// Adapted from
// https://github.com/Mozilla-Ocho/llamafile/blob/0.8.8/llamafile/iqk_mul_mat_arm82.cpp
// Copyrigth 2024 Iwan Kawrakow.
// Copyright(c) 2024 by KVCache.AI, All Rights Reserved.

#ifdef __aarch64__
// Mirror the zen4 path: rename the symbols emitted from iqk_mul_mat_arm.inc
// so that sgemm.cpp's iqk_mul_mat_{,moe_}arm82 references resolve at link
// time. Without this rename, .inc emits the un-suffixed iqk_mul_mat / _moe
// and `kt_kernel_ext.so` reports `undefined symbol: iqk_mul_mat_moe_arm82`
// at import time.
#define iqk_mul_mat iqk_mul_mat_arm82
#define iqk_mul_mat_moe iqk_mul_mat_moe_arm82
#include "iqk_mul_mat_arm.inc"
#endif  // __aarch64__
