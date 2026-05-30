#!/usr/bin/env bash
# Ensure `import kt_kernel` works without pip editable install (container restart).
# setup.py maps package name kt_kernel -> python/; symlink + PYTHONPATH parent dir suffices.
# Still requires: kt-kernel/python/kt_kernel_ext*.so on disk + libhwloc.so.15 at runtime.

p27_ensure_kt_kernel() {
  local repo="${1:?repo root required}"
  if [[ ! -e "${repo}/kt-kernel/kt_kernel" ]]; then
    ln -sfn python "${repo}/kt-kernel/kt_kernel"
  fi
  export PYTHONPATH="${repo}/third_party/sglang/python:${repo}/kt-kernel${PYTHONPATH:+:$PYTHONPATH}"
}
