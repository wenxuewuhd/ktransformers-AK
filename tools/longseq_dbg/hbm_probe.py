# 固定 M,扫 E:若 NPU 耗时 ∝ E(权重字节)→ HBM-bandwidth-bound,可反推 HBM BW。
# 再固定 E,扫 M:看 compute 线性段斜率 → 算力。
import time, torch, torch_npu
torch.npu.set_device(0)
H, I, TOPK = 4096, 2048, 6
def make_grp(N, E):
    base = N // E
    c = torch.full((E,), base, dtype=torch.int64); c[:(N-base*E)] += 1
    return c.npu()
def run(M, E, dt=torch.bfloat16, iters=5):
    N = M*TOPK
    hs = torch.randn(N, H, dtype=dt, device='npu')
    w13 = (torch.randn(E, H, 2*I, dtype=dt, device='npu')*0.02)
    w2  = (torch.randn(E, I, H, dtype=dt, device='npu')*0.02)
    grp = make_grp(N, E)
    def f():
        h = torch.ops.npu.npu_grouped_matmul(x=[hs],weight=[w13],bias=None,split_item=2,group_list_type=1,group_type=0,group_list=grp,output_dtype=dt)[0]
        h = torch.ops.npu.npu_swiglu(h)
        h = torch.ops.npu.npu_grouped_matmul(x=[h],weight=[w2],bias=None,split_item=2,group_list_type=1,group_type=0,group_list=grp,output_dtype=dt)[0]
        return h
    f(); torch.npu.synchronize()
    t0=time.perf_counter()
    for _ in range(iters): f()
    torch.npu.synchronize()
    ms=(time.perf_counter()-t0)/iters*1e3
    wgb = E*(H*2*I+I*H)*2/1e9
    del hs,w13,w2; torch.npu.empty_cache()
    return ms, wgb
print("# E-sweep @ M=256 (compute 极小,凸显权重读 → HBM bound)")
print(f"{'E':>5}{'ms':>9}{'wGB':>8}{'impliedHBM_GB/s':>16}")
for E in (64,128,192,256):
    ms,wgb = run(256, E)
    print(f"{E:>5}{ms:>9.3f}{wgb:>8.1f}{wgb/(ms/1e3):>16.0f}", flush=True)
print("# M-sweep @ E=256 (compute 线性段斜率 → TFLOPS)")
print(f"{'M':>7}{'ms':>9}")
for M in (256,512,4096,16384,32768):
    ms,_ = run(M,256)
    print(f"{M:>7}{ms:>9.3f}", flush=True)
