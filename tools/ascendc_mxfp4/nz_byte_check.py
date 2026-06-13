"""Byte-level FRACTAL_NZ validation harness for the direct-NZ kernel (path 3).

Delivered by the kernel-advisor session for G to validate a kernel that writes int8
FRACTAL_NZ (acl_format=29) GM directly, instead of going through the torch
de-interleave + transpose + npu_format_cast post-step.

Two independent references, so a kernel bug can't hide behind a matching transform:
  - verify_against_format_cast(nz_kernel, nd_ref): byte-compare the kernel's NZ output
    against torch_npu.npu_format_cast(nd_ref, 29) -- the proven path.
  - nz_bytes_from_formula(nd_ref): build the expected NZ bytes purely from the layout
    formula p(in,out) (no format_cast at all) -- catches a format_cast that itself drifts.

Layout (independently verified 2026-06-13, 0/2048 mismatch):
  int8 NZ tile = [16 IN-rows x 32 OUT-cols] row-major, block order W(out) then H(in):
  p(in,out) = ((out//32)*ceil(IN/16) + (in//16))*512 + (in%16)*32 + (out%32)
  one contiguous 512B tile = 16 input channels x 32 output channels.

Usage in a kernel test:
  from nz_byte_check import verify_against_format_cast, nz_bytes_from_formula, raw_bytes
  nd_ref = quantized_int8_ND  # [E, IN, OUT] int8, the golden W8A8 weights
  ok, msg = verify_against_format_cast(nz_kernel, nd_ref); assert ok, msg
"""
import ctypes, math, numpy as np, torch, torch_npu

_NZ = 29
_acl = ctypes.CDLL("libascendcl.so")
_D2D = 3  # ACL_MEMCPY_DEVICE_TO_DEVICE


def raw_bytes(t: torch.Tensor) -> np.ndarray:
    """Physical bytes of an NPU tensor as np.uint8. Reads via aclrtMemcpy D2D into a
    PLAIN ND buffer -- t.cpu() on an NZ tensor triggers NZ->ND and hides the layout.
    MUST sync first: a format_cast issued on the torch stream is not yet done when the
    raw memcpy runs, else you read a half-written buffer (rows read as 0, self-contradicting)."""
    n = t.numel() * t.element_size()
    plain = torch.empty(n, dtype=torch.uint8, device=t.device)
    torch.npu.synchronize()
    rc = _acl.aclrtMemcpy(ctypes.c_void_p(plain.data_ptr()), ctypes.c_size_t(n),
                          ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(n), ctypes.c_int(_D2D))
    if rc != 0:
        raise RuntimeError(f"aclrtMemcpy rc={rc}")
    torch.npu.synchronize()
    return plain.cpu().numpy()


def nz_bytes_from_formula(nd_ref: torch.Tensor) -> np.ndarray:
    """Expected NZ physical bytes built ONLY from the layout formula (no format_cast).
    nd_ref: [E, IN, OUT] int8 (cpu or npu). Returns flat np.int8 [E*ceil(OUT/32)*ceil(IN/16)*512]."""
    assert nd_ref.dim() == 3, "expect [E,IN,OUT]"
    E, IN, OUT = nd_ref.shape
    a = nd_ref.detach().to("cpu").to(torch.int8).numpy()
    tiles_in, tiles_out = math.ceil(IN / 16), math.ceil(OUT / 32)
    out = np.zeros(E * tiles_out * tiles_in * 512, dtype=np.int8)
    in_idx = np.arange(IN)[:, None]
    out_idx = np.arange(OUT)[None, :]
    p = ((out_idx // 32) * tiles_in + (in_idx // 16)) * 512 + (in_idx % 16) * 32 + (out_idx % 32)
    per_e = tiles_out * tiles_in * 512
    for e in range(E):
        out[e * per_e + p.reshape(-1)] = a[e].reshape(-1)
    return out


def assert_nz_bytes_equal(nz_a: torch.Tensor, nz_b: torch.Tensor, tag="") -> tuple[bool, str]:
    ba, bb = raw_bytes(nz_a).view(np.int8), raw_bytes(nz_b).view(np.int8)
    if ba.shape != bb.shape:
        return False, f"{tag} size {ba.shape} != {bb.shape}"
    nbad = int((ba != bb).sum())
    return nbad == 0, f"{tag} byte mismatches={nbad}/{ba.size}"


def verify_against_format_cast(nz_kernel: torch.Tensor, nd_ref: torch.Tensor) -> tuple[bool, str]:
    """Primary check for a direct-NZ kernel: does its physical output equal the proven
    npu_format_cast(nd_ref, 29)?  nd_ref = [E,IN,OUT] int8 golden W8A8 weights."""
    nz_ref = torch_npu.npu_format_cast(nd_ref.npu().contiguous(), _NZ)
    ok_fc, m1 = assert_nz_bytes_equal(nz_kernel, nz_ref, "vs format_cast")
    # also cross-check format_cast itself against the pure formula (defense in depth)
    fc_bytes = raw_bytes(nz_ref).view(np.int8)
    fm_bytes = nz_bytes_from_formula(nd_ref)
    ok_fm = fc_bytes.shape == fm_bytes.shape and int((fc_bytes != fm_bytes).sum()) == 0
    return (ok_fc and ok_fm), f"{m1}; format_cast==formula:{ok_fm}"


def _selftest():
    E, IN, OUT = 1, 32, 64
    val = (torch.arange(IN).view(IN, 1) * OUT + torch.arange(OUT).view(1, OUT)) % 127
    nd = torch.zeros(E, IN, OUT, dtype=torch.int8); nd[0] = val.to(torch.int8)
    nz = torch_npu.npu_format_cast(nd.npu().contiguous(), _NZ)
    ok, msg = assert_nz_bytes_equal(nz, nz, "identity")
    fc = raw_bytes(nz).view(np.int8); fm = nz_bytes_from_formula(nd)
    same = fc.shape == fm.shape and int((fc != fm).sum()) == 0
    print(f"selftest: identity={ok} format_cast==formula={same} ({msg})")
    assert ok and same


if __name__ == "__main__":
    _selftest()
