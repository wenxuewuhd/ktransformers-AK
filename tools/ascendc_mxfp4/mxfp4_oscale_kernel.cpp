// AscendC kernel: per-output-channel oscale = amax/127 ONLY (sidestep 1).
// Reads MXFP4 codes+scale, dequants per row, reduces amax over IN, accumulates the per-row
// oscale into a UB block, and flushes each FULL block as ONE large contiguous DataCopy at an
// 8-aligned boundary -- avoids the tiny-per-row-store-interleaved-with-loads failure mode.
// Pair with the int8-weight kernel (which is already correct). Decode/scale/reduce are the same
// verified math as mxfp4_dq_kernel.cpp.
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t HALF_MAX = 2048;
constexpr int32_t IN_MAX = 4096;
constexpr int32_t NB_MAX = 128;
constexpr int32_t ACC = 512;          // oscale flush block (floats), 8-aligned, 2KB

extern "C" __global__ __aicore__ void mxfp4_oscale(
    GM_ADDR codes, GM_ADDR scaleg, GM_ADDR oscaleg,
    GM_ADDR lutLoG, GM_ADDR lutHiG, GM_ADDR lutE8G, GM_ADDR scOffG,
    uint32_t R, uint32_t HALF, uint32_t NB, uint32_t IN)
{
    const int32_t blkid = GetBlockIdx();
    const int32_t nblk = GetBlockNum();

    GlobalTensor<uint8_t> gCodes, gScale;
    GlobalTensor<float> gOscale, gLutLo, gLutHi, gLutE8;
    GlobalTensor<uint32_t> gScOff;
    gCodes.SetGlobalBuffer((__gm__ uint8_t *)codes);
    gScale.SetGlobalBuffer((__gm__ uint8_t *)scaleg);
    gOscale.SetGlobalBuffer((__gm__ float *)oscaleg);
    gLutLo.SetGlobalBuffer((__gm__ float *)lutLoG);
    gLutHi.SetGlobalBuffer((__gm__ float *)lutHiG);
    gLutE8.SetGlobalBuffer((__gm__ float *)lutE8G);
    gScOff.SetGlobalBuffer((__gm__ uint32_t *)scOffG);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> qCodes, qScale;
    pipe.InitBuffer(qCodes, 1, HALF_MAX * sizeof(uint8_t));
    pipe.InitBuffer(qScale, 1, (NB_MAX + 32) * sizeof(uint8_t));
    TBuf<TPosition::VECCALC> tLutLo, tLutHi, tLutE8, tScOff;
    TBuf<TPosition::VECCALC> tComb, tOff, tOffH, tScI, tScF, tScHalf, tAbs, tWork, tAcc;
    pipe.InitBuffer(tLutLo, 256 * sizeof(float));
    pipe.InitBuffer(tLutHi, 256 * sizeof(float));
    pipe.InitBuffer(tLutE8, 256 * sizeof(float));
    pipe.InitBuffer(tScOff, HALF_MAX * sizeof(uint32_t));
    pipe.InitBuffer(tComb, 2 * HALF_MAX * sizeof(float));   // vlo || vhi
    pipe.InitBuffer(tOff, HALF_MAX * sizeof(int32_t));
    pipe.InitBuffer(tOffH, HALF_MAX * sizeof(half));
    pipe.InitBuffer(tScI, NB_MAX * sizeof(int32_t));
    pipe.InitBuffer(tScF, NB_MAX * sizeof(float));
    pipe.InitBuffer(tScHalf, HALF_MAX * sizeof(float));
    pipe.InitBuffer(tAbs, HALF_MAX * sizeof(float));
    pipe.InitBuffer(tWork, HALF_MAX * sizeof(float));
    pipe.InitBuffer(tAcc, ACC * sizeof(float));

    LocalTensor<float> lutLo = tLutLo.Get<float>();
    LocalTensor<float> lutHi = tLutHi.Get<float>();
    LocalTensor<float> lutE8 = tLutE8.Get<float>();
    LocalTensor<uint32_t> scOff = tScOff.Get<uint32_t>();
    LocalTensor<float> comb = tComb.Get<float>();
    LocalTensor<int32_t> off = tOff.Get<int32_t>();
    LocalTensor<half> offH = tOffH.Get<half>();
    LocalTensor<int32_t> scI = tScI.Get<int32_t>();
    LocalTensor<float> scF = tScF.Get<float>();
    LocalTensor<float> scHalf = tScHalf.Get<float>();
    LocalTensor<float> absb = tAbs.Get<float>();
    LocalTensor<float> work = tWork.Get<float>();
    LocalTensor<float> acc = tAcc.Get<float>();

    DataCopy(lutLo, gLutLo, 256);
    DataCopy(lutHi, gLutHi, 256);
    DataCopy(lutE8, gLutE8, 256);
    DataCopy(scOff, gScOff, HALF);
    PipeBarrier<PIPE_ALL>();

    // block partition: each core owns a contiguous, ACC-aligned row range
    const uint32_t chunk = ((R + nblk - 1) / nblk + (ACC - 1)) / ACC * ACC;
    const uint32_t rStart = (uint32_t)blkid * chunk;
    uint32_t rEnd = rStart + chunk;
    if (rEnd > R) rEnd = R;
    const uint32_t scLoad = (NB + 31) / 32 * 32;

    for (uint32_t base = rStart; base < rEnd; base += ACC) {
        uint32_t bend = base + ACC;
        if (bend > rEnd) bend = rEnd;
        for (uint32_t r = base; r < bend; r++) {
            LocalTensor<float> vlo = comb;
            LocalTensor<float> vhi = comb[HALF];
            LocalTensor<uint8_t> cu = qCodes.AllocTensor<uint8_t>();
            DataCopy(cu, gCodes[(uint64_t)r * HALF], HALF);
            qCodes.EnQue(cu);
            LocalTensor<uint8_t> cuU = qCodes.DeQue<uint8_t>();
            LocalTensor<uint8_t> su = qScale.AllocTensor<uint8_t>();
            DataCopy(su, gScale[(uint64_t)r * NB], scLoad);
            qScale.EnQue(su);
            LocalTensor<uint8_t> suU = qScale.DeQue<uint8_t>();

            Cast(offH, cuU, RoundMode::CAST_NONE, HALF);
            Muls(offH, offH, (half)4.0, HALF);
            Cast(off, offH, RoundMode::CAST_RINT, HALF);
            LocalTensor<uint32_t> offU = off.ReinterpretCast<uint32_t>();
            Gather(vlo, lutLo, offU, (uint32_t)0, HALF);
            Gather(vhi, lutHi, offU, (uint32_t)0, HALF);
            qCodes.FreeTensor(cuU);

            Cast(offH, suU, RoundMode::CAST_NONE, NB);
            Muls(offH, offH, (half)4.0, NB);
            Cast(scI, offH, RoundMode::CAST_RINT, NB);
            Gather(scF, lutE8, scI.ReinterpretCast<uint32_t>(), (uint32_t)0, NB);
            qScale.FreeTensor(suU);
            PipeBarrier<PIPE_V>();
            Gather(scHalf, scF, scOff, (uint32_t)0, HALF);
            Mul(vlo, vlo, scHalf, HALF);
            Mul(vhi, vhi, scHalf, HALF);
            PipeBarrier<PIPE_V>();

            Abs(absb, vlo, HALF);
            Abs(work, vhi, HALF);
            Max(scHalf, absb, work, HALF);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> fa = scHalf, fb = absb;
            for (uint32_t h = HALF >> 1; h >= 8; h >>= 1) {
                Max(fb, fa, fa[h], h);
                PipeBarrier<PIPE_V>();
                LocalTensor<float> tmp = fa; fa = fb; fb = tmp;
            }
            PipeBarrier<PIPE_ALL>();
            float amax = fa.GetValue(0);
            for (int i = 1; i < 8; i++) { float v = fa.GetValue(i); if (v > amax) amax = v; }
            if (amax < 1e-8f) amax = 1e-8f;
            acc.SetValue(r - base, amax / 127.0f);   // scalar -> UB accumulator
        }
        // flush this block as ONE contiguous DataCopy (8-aligned base, count = ACC even on the
        // tail -> may write up to ACC-1 padding floats past rEnd; oscale GM is padded to /ACC).
        PipeBarrier<PIPE_ALL>();
        DataCopy(gOscale[base], acc, ACC);
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" void launch_mxfp4_oscale(void *stream, uint32_t blockdim,
    uint8_t *codes, uint8_t *scale, uint8_t *oscale,
    uint8_t *lutLo, uint8_t *lutHi, uint8_t *lutE8, uint8_t *scOff,
    uint32_t R, uint32_t HALF, uint32_t NB, uint32_t IN)
{
    mxfp4_oscale<<<blockdim, nullptr, stream>>>(
        (GM_ADDR)codes, (GM_ADDR)scale, (GM_ADDR)oscale,
        (GM_ADDR)lutLo, (GM_ADDR)lutHi, (GM_ADDR)lutE8, (GM_ADDR)scOff,
        R, HALF, NB, IN);
}
