"""桩-0: Triton-Ascend 可行性 smoke test (Session G 起点).

证明 nibble-mask (x & 0xF) + per-block scale 广播这套 dequant 内核形状能在 NPU 上
正确 lower。Session C 跑过: max_abs_err 0.0, bit-exact.

下一步 (§4 桩1): 把这里的 `(x & 0xF) * scale` 换成真正的 FP4 e2m1 解码 + e8m0
per-block-32 scale, 对 mxfp4_conv_vectorized_npu.py 的参考输出做 bit-exact 对账.

跑: /usr/local/python3.11.14/bin/python3.11 tools/longseq_dbg/triton_ascend_smoke.py
"""
import torch, torch_npu, triton, triton.language as tl


@triton.jit
def k(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    x = tl.load(x_ptr + off, mask=m)
    y = tl.load(y_ptr + off, mask=m)
    # dequant 内核形状: nibble unpack + scale 广播
    lo = (x & 0xF).to(tl.float32)
    scale = y.to(tl.float32)
    tl.store(o_ptr + off, lo * scale, mask=m)


def main():
    n = 4096
    x = torch.randint(0, 255, (n,), dtype=torch.int32, device="npu")
    y = torch.randint(1, 8, (n,), dtype=torch.int32, device="npu")
    o = torch.empty(n, dtype=torch.float32, device="npu")
    k[(triton.cdiv(n, 256),)](x, y, o, n, BLOCK=256)
    torch.npu.synchronize()
    ref = ((x & 0xF).float()) * y.float()
    err = (o - ref).abs().max().item()
    print("max_abs_err", err, "ok", torch.allclose(o, ref))
    assert torch.allclose(o, ref), "Triton-Ascend smoke FAILED"


if __name__ == "__main__":
    main()
