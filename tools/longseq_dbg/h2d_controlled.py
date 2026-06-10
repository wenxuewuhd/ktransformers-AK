# 受控 H2D 带宽实验:只变"哪个 stream / 几个 stream",其余(buffer、warmup、迭代数)全固定。
import time, torch, torch_npu
torch.npu.set_device(0)
GiB = 1<<30
SZ = 4*GiB
ITERS = 30
WARM = 5

host = torch.empty(SZ, dtype=torch.uint8, pin_memory=True)   # 固定 pinned 源
assert host.is_pinned()

def bw(fn):
    for _ in range(WARM): fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS): fn()
    torch.npu.synchronize()
    dt = (time.perf_counter()-t0)/ITERS
    return SZ/1e9/dt

# A) default stream, non_blocking
dev = torch.empty(SZ, dtype=torch.uint8, device='npu')
print(f"A default stream,        non_blocking=True : {bw(lambda: dev.copy_(host, non_blocking=True)):.1f} GB/s")
print(f"B default stream,        non_blocking=False: {bw(lambda: dev.copy_(host, non_blocking=False)):.1f} GB/s")

# C) 单 side stream
s1 = torch.npu.Stream()
def c():
    with torch.npu.stream(s1): dev.copy_(host, non_blocking=True)
print(f"C 单 side stream,         non_blocking=True : {bw(c):.1f} GB/s")

# D) 单 side stream, blocking
def d():
    with torch.npu.stream(s1): dev.copy_(host, non_blocking=False)
print(f"D 单 side stream,         non_blocking=False: {bw(d):.1f} GB/s")

# E) 2 个 side stream 各搬一半,并发
s2 = torch.npu.Stream()
h_a, h_b = host[:SZ//2], host[SZ//2:]
d_a, d_b = dev[:SZ//2], dev[SZ//2:]
def e():
    with torch.npu.stream(s1): d_a.copy_(h_a, non_blocking=True)
    with torch.npu.stream(s2): d_b.copy_(h_b, non_blocking=True)
print(f"E 2 side stream 并发(各半): {bw(e):.1f} GB/s")
