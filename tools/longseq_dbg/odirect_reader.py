#!/usr/bin/env python3
"""O_DIRECT parallel expert reader for the streaming-prefill pool build.

Roofline (handoff §2c-ii-c4): the build is bottlenecked by reading 277GB of expert int8 from
the W8A8 checkpoint. Python get_tensor / buffered reads cap at ~0.7-1 GB/s; parallel O_DIRECT
hits the NVMe ceiling ~3.5 GB/s (cache-independent, measured). This reads a layer's experts
into the final pinned ND pool buffers via O_DIRECT large reads + bulk rearrange.

Layout target (matches kt_stream_prefill): w13[E,2I,H] = concat([w1(gate),w3(up)],dim=1);
w2[E,H,I] = down. Scales are tiny -> normal safetensors get_tensor.
"""
import json, os, struct, mmap
import torch
from concurrent.futures import ThreadPoolExecutor

_ALIGN = 4096


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def _read_region_odirect(path, base, lo, hi, prefaulted_buf):
    """O_DIRECT read file bytes [base+lo, base+hi) into prefaulted_buf[0:hi-lo]. Returns view."""
    a_lo = ((base + lo) // _ALIGN) * _ALIGN
    skip = (base + lo) - a_lo
    end = base + hi
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        pos = a_lo
        dst = memoryview(prefaulted_buf)
        got = 0
        need = end - a_lo
        while got < need:
            n = os.preadv(fd, [dst[got:]], pos)
            if n <= 0:
                break
            got += n
            pos += n
    finally:
        os.close(fd)
    return memoryview(prefaulted_buf)[skip : skip + (hi - lo)]


def read_layer(ckpt, idx, L, E, H, I, flat13, flat2, s13, s2, scratch):
    """Fill flat13[E,2I,H]/flat2[E,H,I] (int8) + s13/s2 (fp32) for layer L. scratch: pre-faulted
    bytearray big enough for one shard's expert region (reused). Single-thread, one layer."""
    from safetensors import safe_open

    # group expert weight tensors by shard file
    byfile = {}
    for e in range(E):
        for w in ("w1", "w2", "w3"):
            k = f"layers.{L}.ffn.experts.{e}.{w}.weight"
            byfile.setdefault(idx[k], []).append((k, e, w))
    f13 = flat13.view(E, 2 * I, H)
    f2 = flat2.view(E, H, I)
    for fn, items in byfile.items():
        path = os.path.join(ckpt, fn)
        hdr, base = _header(path)
        offs = {k: hdr[k]["data_offsets"] for k, _, _ in items}
        lo = min(o[0] for o in offs.values())
        hi = max(o[1] for o in offs.values())
        region = _read_region_odirect(path, base, lo, hi, scratch)  # bytes view, len hi-lo
        for k, e, w in items:
            o0, o1 = offs[k]
            blk = torch.frombuffer(region[o0 - lo : o1 - lo], dtype=torch.int8)
            if w == "w1":
                f13[e, 0:I].copy_(blk.view(I, H))
            elif w == "w3":
                f13[e, I : 2 * I].copy_(blk.view(I, H))
            else:
                f2[e].copy_(blk.view(H, I))
    # scales via normal get_tensor (tiny)
    for fn in {idx[f"layers.{L}.ffn.experts.0.w1.weight_scale"]}:
        pass
    sfiles = {}
    for e in range(E):
        for w in ("w1", "w2", "w3"):
            k = f"layers.{L}.ffn.experts.{e}.{w}.weight_scale"
            sfiles.setdefault(idx[k], []).append((k, e, w))
    for fn, items in sfiles.items():
        with safe_open(os.path.join(ckpt, fn), framework="pt") as f:
            for k, e, w in items:
                t = f.get_tensor(k)
                if w == "w1":
                    s13[e, 0:I] = t.reshape(I, 1)
                elif w == "w3":
                    s13[e, I : 2 * I] = t.reshape(I, 1)
                else:
                    s2[e] = t.reshape(H, 1)


if __name__ == "__main__":
    import time, sys

    CKPT = "/workspace/models/DeepSeekV4/DeepSeek-V4-Flash-W8A8"
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    E, H, I = 256, 4096, 2048
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    flat13 = torch.empty(E * 2 * I * H, dtype=torch.int8)
    flat2 = torch.empty(E * H * I, dtype=torch.int8)
    s13 = torch.empty(E, 2 * I, 1, dtype=torch.float32)
    s2 = torch.empty(E, H, 1, dtype=torch.float32)
    scratch = mmap.mmap(-1, 8 * 1024 ** 3)  # page-aligned (O_DIRECT needs it), reused per layer
    t = time.time()
    read_layer(CKPT, idx, L, E, H, I, flat13, flat2, s13, s2, scratch)
    dt = time.time() - t
    gb = E * (2 * I * H + H * I) / 1e9
    print(f"[odirect read_layer] layer{L}: {dt:.2f}s {gb:.2f}GB = {gb/dt:.2f} GB/s")
    # correctness vs get_tensor
    from safetensors import safe_open

    e = 100
    refs = {}
    for w in ("w1", "w2", "w3"):
        k = f"layers.{L}.ffn.experts.{e}.{w}.weight"
        with safe_open(os.path.join(CKPT, idx[k]), framework="pt") as f:
            refs[w] = f.get_tensor(k)
    ok13_gate = torch.equal(flat13.view(E, 2 * I, H)[e, 0:I], refs["w1"])
    ok13_up = torch.equal(flat13.view(E, 2 * I, H)[e, I : 2 * I], refs["w3"])
    ok2 = torch.equal(flat2.view(E, H, I)[e], refs["w2"])
    print(f"[correctness e={e}] gate={ok13_gate} up={ok13_up} down={ok2}")
