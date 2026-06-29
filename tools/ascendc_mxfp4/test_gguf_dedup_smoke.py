#!/usr/bin/env python3
"""Integration smoke test for the GGUF dedup streaming path (no server, ~1 min).

Drives the REAL kt_stream_prefill._streaming_forward dedup branch on a couple of real layers
through the REAL npu_fused_experts, with dynamic-resident OFF, and checks each layer's output
against a deterministic golden (synced convert of the same GGUF half-block feed). Catches:
  - integration glue: the dedup branch is actually taken; imports/wrapper plumbing work
  - reused pinned staging buffer correctness across layers (buffer reuse)
  - whether the production (unsynced) convert is reliable in a normal single forward call

Run:
  NPU_DEVICE_ID=<free> KT_MXFP4_DEPOOL=1 KT_MXFP4_GGUF_DEDUP=1 \
  KT_GGUF_TEMPLATE='/workspace/models/cache/dsv4_layer{layer_idx}_mxfp4.gguf' \
  python3 tools/ascendc_mxfp4/test_gguf_dedup_smoke.py
"""
import ctypes
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tools" / "ascendc_mxfp4"))
sys.path.insert(0, str(_REPO / "third_party" / "llama.cpp" / "gguf-py"))
import mxfp4_fused_op as MX  # noqa: E402
from gguf import GGUFReader  # noqa: E402

_MOD = _REPO / "third_party/sglang/python/sglang/srt/layers/moe/kt_stream_prefill.py"
_MXFP4_CKPT = os.environ.get("KT_MXFP4_CKPT", "/workspace/models/DeepSeekV4/DeepSeek-V4-Flash")
_GGUF_TMPL = os.environ["KT_GGUF_TEMPLATE"]
_NZ = 29


def _load_module():
    spec = importlib.util.spec_from_file_location("ksp", _MOD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _gguf_halfblock_feed(layer, dev):
    """[E,2I,H/2]+scale, [E,H,I/2]+scale on device (same de-interleave the wiring does)."""
    r = GGUFReader(_GGUF_TMPL.format(layer_idx=layer))

    def proj(name):
        t = next(t for t in r.tensors if t.name == f"blk.{layer}.{name}.weight")
        E, N, b17 = t.data.shape
        nb = b17 // 17
        blk = torch.from_numpy(np.array(t.data)).reshape(E, N, nb, 17)
        return blk[..., 1:17].reshape(E, N, nb * 16).contiguous().to(dev), blk[..., 0].contiguous().to(dev)

    cg, sg = proj("ffn_gate_exps")
    cu, su = proj("ffn_up_exps")
    cd, sd = proj("ffn_down_exps")
    return torch.cat([cg, cu], 1), torch.cat([sg, su], 1), cd, sd


def _convert_synced(codes, scale, IN, dev, bd=40):
    """Deterministic (per-chunk sync) half-block convert -> golden NZ slots."""
    E, OUT, HALF = codes.shape
    NB = scale.shape[2]
    HALFp = IN // 2
    ll, lh, le, so = MX._consts(HALF, NB, dev)
    stp = torch.npu.current_stream().npu_stream
    P = lambda t: ctypes.c_void_p(t.data_ptr())
    lib = MX.get_lib()
    out_nz = None
    oscale = torch.empty((E, OUT), dtype=torch.bfloat16, device=dev)
    for c in range(0, E, 32):
        ce = min(c + 32, E)
        Ec = ce - c
        Rc = Ec * OUT
        cd = codes[c:ce].reshape(Rc, HALF).contiguous()
        sd = scale[c:ce].reshape(Rc, NB).contiguous()
        out = torch.empty((Rc, IN), dtype=torch.int8, device=dev)
        Rp = (Rc + 511) // 512 * 512
        osc = torch.empty((Rp,), dtype=torch.float32, device=dev)
        lib.launch_mxfp4_fused(ctypes.c_void_p(stp), bd, P(cd), P(sd), P(out), P(osc),
                               P(ll), P(lh), P(le), P(so), Rc, HALF, NB, IN)
        torch.npu.synchronize()
        nb = HALFp // 16
        q = torch.cat([out[:, :HALFp].reshape(Rc, nb, 16),
                       out[:, HALFp:].reshape(Rc, nb, 16)], 2).reshape(Ec, OUT, IN)
        nd = q.to(torch.float16).transpose(1, 2).contiguous().to(torch.int8)
        nz = torch_npu.npu_format_cast(nd, _NZ)
        if out_nz is None:
            out_nz = torch.empty((E,) + tuple(nz.shape[1:]), dtype=torch.int8, device=dev)
        out_nz[c:ce].copy_(nz)
        oscale[c:ce] = osc[:Rc].reshape(Ec, OUT).to(torch.bfloat16)
        torch.npu.synchronize()
        del out, q, nd, nz, osc, cd, sd
    return out_nz, oscale


def main():
    assert os.environ.get("KT_MXFP4_DEPOOL") == "1" and os.environ.get("KT_MXFP4_GGUF_DEDUP") == "1", \
        "set KT_MXFP4_DEPOOL=1 KT_MXFP4_GGUF_DEDUP=1"
    dev = f"npu:{os.environ.get('NPU_DEVICE_ID', '0')}"
    torch.npu.set_device(dev)
    torch.manual_seed(0)
    cfg = json.load(open(os.path.join(_MXFP4_CKPT, "config.json")))
    E, H, I = (int(cfg[k]) for k in ("n_routed_experts", "hidden_size", "moe_intermediate_size"))

    m = _load_module()
    assert m._KT_MXFP4_DEPOOL and m._KT_GGUF_DEDUP, "module did not pick up dedup env"
    assert not m._KT_DYN_RESIDENT, "run with KT_DYNAMIC_RESIDENT unset for the smoke test"
    m._CFG.update(E=E, H=H, I=I, num_layers=int(cfg["num_hidden_layers"]))

    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput  # noqa: F401

    M, tk = 600, 6  # >512 so it mirrors a streaming chunk
    x = torch.randn(M, H, dtype=torch.bfloat16, device=dev)
    tw, tid = torch.topk(torch.softmax(torch.randn(M, E, device=dev), -1), tk, -1)
    tid = tid.to(torch.int32)
    topk = types.SimpleNamespace(topk_weights=tw, topk_ids=tid)

    def cos(a, b):
        a, b = a.reshape(-1).float(), b.reshape(-1).float()
        return float((a @ b) / (a.norm() * b.norm() + 1e-9))

    # SMOKE_RESERVE_SLOT=1: reserve the streaming slot so _streaming_forward exercises the
    # out_w13/out_w2 reserved-slot write path (the OOM fix) instead of the fresh-alloc path.
    if os.environ.get("SMOKE_RESERVE_SLOT") == "1":
        m.reserve_slot_depool(E, H, I, dev)
        print(f"  [reserved ND slot: {tuple(m._SLOT['w13'].shape)}+{tuple(m._SLOT['w2'].shape)}]")

    # Warm up kernels/allocators/streams once (production's streaming convert always runs warm, after
    # model load + attention). Set SMOKE_NO_WARMUP=1 to test the cold path. Discard the result.
    if os.environ.get("SMOKE_NO_WARMUP") != "1":
        _ = m._streaming_forward(16, x, topk, tk)
        torch.npu.synchronize()

    ok = True
    for L in (16, 17):  # two layers in sequence -> exercises the reused pinned staging buffer
        ci = m._streaming_forward(L, x, topk, tk)        # REAL dedup path + REAL operator
        out = ci.hidden_states
        # deterministic golden for the same layer/inputs
        gc13, gs13, gc2, gs2 = _gguf_halfblock_feed(L, dev)
        w13, s13b = _convert_synced(gc13, gs13, H, dev)
        w2, s2b = _convert_synced(gc2, gs2, I, dev)
        from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import npu_fused_experts
        ref = npu_fused_experts(hidden_states=x.clone(), w13=w13, w13_scale=s13b, w2=w2,
                                w2_scale=s2b, topk_weights=tw.to(x.dtype),
                                topk_ids=tid, top_k=tk)
        torch.npu.synchronize()
        finite = bool(torch.isfinite(out).all())
        c = cos(out, ref)
        shape_ok = tuple(out.shape) == (M, H)
        layer_ok = finite and shape_ok and c >= 0.999
        ok = ok and layer_ok
        print(f"  L={L}: shape={tuple(out.shape)} finite={finite} "
              f"cos(dedup_path, synced_golden)={c:.6f}  {'OK' if layer_ok else 'BAD'}")
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
