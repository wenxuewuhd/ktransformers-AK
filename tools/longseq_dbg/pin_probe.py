# 探 NPU pinned(page-locked)host 内存上限 + pageable->pinned memcpy 带宽(ring-staging 备选)。
import time, torch, torch_npu
torch.npu.set_device(0)
GiB = 1<<30
print("# 逐步分配 pinned host buffer,看能撑到多大(每块 8GiB)")
blocks=[]; total=0
try:
    for i in range(40):  # up to 320 GiB
        b = torch.empty(8*GiB, dtype=torch.uint8, pin_memory=True)
        blocks.append(b); total += 8
        print(f"  pinned {total} GiB OK", flush=True)
except Exception as e:
    print(f"  STOP at {total} GiB: {repr(e)[:100]}", flush=True)
print(f"# 达到 pinned 总量 ~{total} GiB(目标 277GB int8 全量源)")
# pageable->pinned memcpy 带宽(ring staging 用)
if blocks:
    src = torch.empty(2*GiB, dtype=torch.uint8)  # pageable
    dst = blocks[0][:2*GiB]
    dst.copy_(src); 
    t0=time.perf_counter()
    for _ in range(3): dst.copy_(src)
    dt=(time.perf_counter()-t0)/3
    print(f"# pageable->pinned memcpy: {2/dt:.1f} GiB/s")
