// AscendC fused MXFP4 -> int8 (+ per-output-channel scale) dequant/requant kernel.
// One output row (channel) per grid-stride iteration.
//
// STATUS: WIP (see STATUS.md). int8 weight output is CORRECT single+multi core (decode/scale/
// reduce/requant all verified; cast must be f32->half->int8). The per-channel `oscale` GM write
// is the one unresolved piece (lands in isolation, writes nothing embedded in the full kernel).
// Authoritative spec/golden/acceptance: tools/mxfp4_w8a8_op/.
//
// Host-uploaded byte-indexed LUTs keep the kernel to count-form vector ops only:
//   lutLo[b]=FP4[b&0xF]  lutHi[b]=FP4[b>>4]  lutE8[b]=2^(b-127)   (256 f32 each)
//   scOff[j]=(j>>4)*4 bytes  (block j/16 -> f32 offset into per-row scale)
//   gidxF[i]=((i&1)*HALF+(i>>1))*4 bytes  (interleave: outF[i]=comb[lo||hi])
//
// Notes / hw pitfalls (verified): Gather uses BYTE offsets; in-place Max(x,x,x[h]) yields 0
// (fold needs distinct dst); reduce via non-in-place ping-pong fold to 8 + scalar tail;
// interleave done on f32 (gather works on f32) then cast f32->i32->i16->i8.
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t HALF_MAX = 2048;
constexpr int32_t IN_MAX = 4096;
constexpr int32_t NB_MAX = 128;

extern "C" __global__ __aicore__ void mxfp4_dq(
    GM_ADDR codes, GM_ADDR scaleg, GM_ADDR outg, GM_ADDR oscaleg,
    GM_ADDR lutLoG, GM_ADDR lutHiG, GM_ADDR lutE8G, GM_ADDR scOffG, GM_ADDR gidxFG,
    uint32_t R, uint32_t HALF, uint32_t NB, uint32_t IN)
{
    const int32_t blkid = GetBlockIdx();
    const int32_t nblk = GetBlockNum();

    GlobalTensor<uint8_t> gCodes, gScale, gOut;
    GlobalTensor<float> gOscale, gLutLo, gLutHi, gLutE8;
    GlobalTensor<uint32_t> gScOff, gGidxF;
    gCodes.SetGlobalBuffer((__gm__ uint8_t *)codes);
    gScale.SetGlobalBuffer((__gm__ uint8_t *)scaleg);
    gOut.SetGlobalBuffer((__gm__ uint8_t *)outg);
    gOscale.SetGlobalBuffer((__gm__ float *)oscaleg);
    gLutLo.SetGlobalBuffer((__gm__ float *)lutLoG);
    gLutHi.SetGlobalBuffer((__gm__ float *)lutHiG);
    gLutE8.SetGlobalBuffer((__gm__ float *)lutE8G);
    gScOff.SetGlobalBuffer((__gm__ uint32_t *)scOffG);
    gGidxF.SetGlobalBuffer((__gm__ uint32_t *)gidxFG);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> qCodes, qScale;
    TQue<QuePosition::VECOUT, 1> qOut, qOsc;
    pipe.InitBuffer(qCodes, 1, HALF_MAX * sizeof(uint8_t));
    pipe.InitBuffer(qScale, 1, (NB_MAX + 32) * sizeof(uint8_t));
    pipe.InitBuffer(qOut, 1, IN_MAX * sizeof(uint8_t));
    pipe.InitBuffer(qOsc, 1, 32);

    TBuf<TPosition::VECCALC> tLutLo, tLutHi, tLutE8, tScOff;
    TBuf<TPosition::VECCALC> tComb, tOff, tOffH, tQ16, tScI, tScF, tScHalf, tAbs, tWork, tOsc;
    pipe.InitBuffer(tLutLo, 256 * sizeof(float));
    pipe.InitBuffer(tLutHi, 256 * sizeof(float));
    pipe.InitBuffer(tLutE8, 256 * sizeof(float));
    pipe.InitBuffer(tScOff, HALF_MAX * sizeof(uint32_t));
    pipe.InitBuffer(tComb, 2 * HALF_MAX * sizeof(float));   // vlo || vhi
    pipe.InitBuffer(tOff, IN_MAX * sizeof(int32_t));
    pipe.InitBuffer(tOffH, HALF_MAX * sizeof(half));
    pipe.InitBuffer(tQ16, IN_MAX * sizeof(int16_t));
    pipe.InitBuffer(tScI, NB_MAX * sizeof(int32_t));
    pipe.InitBuffer(tScF, NB_MAX * sizeof(float));
    pipe.InitBuffer(tScHalf, HALF_MAX * sizeof(float));
    pipe.InitBuffer(tAbs, HALF_MAX * sizeof(float));
    pipe.InitBuffer(tWork, HALF_MAX * sizeof(float));
    pipe.InitBuffer(tOsc, 256 * sizeof(float));

    LocalTensor<float> lutLo = tLutLo.Get<float>();
    LocalTensor<float> lutHi = tLutHi.Get<float>();
    LocalTensor<float> lutE8 = tLutE8.Get<float>();
    LocalTensor<uint32_t> scOff = tScOff.Get<uint32_t>();
    LocalTensor<float> comb = tComb.Get<float>();
    LocalTensor<int32_t> off = tOff.Get<int32_t>();
    LocalTensor<half> offH = tOffH.Get<half>();
    LocalTensor<int16_t> q16 = tQ16.Get<int16_t>();
    LocalTensor<int32_t> scI = tScI.Get<int32_t>();
    LocalTensor<float> scF = tScF.Get<float>();
    LocalTensor<float> scHalf = tScHalf.Get<float>();
    LocalTensor<float> absb = tAbs.Get<float>();
    LocalTensor<float> work = tWork.Get<float>();
    LocalTensor<float> oscBuf = tOsc.Get<float>();

    DataCopy(lutLo, gLutLo, 256);
    DataCopy(lutHi, gLutHi, 256);
    DataCopy(lutE8, gLutE8, 256);
    DataCopy(scOff, gScOff, HALF);
    PipeBarrier<PIPE_ALL>();


    const uint32_t scLoad = (NB + 31) / 32 * 32;
    for (uint32_t r = blkid; r < R; r += nblk) {
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

        // decode lo/hi via byte*4 gather from 256-entry LUTs
        Cast(offH, cuU, RoundMode::CAST_NONE, HALF);
        Muls(offH, offH, (half)4.0, HALF);
        Cast(off, offH, RoundMode::CAST_RINT, HALF);
        LocalTensor<uint32_t> offU = off.ReinterpretCast<uint32_t>();
        Gather(vlo, lutLo, offU, (uint32_t)0, HALF);
        Gather(vhi, lutHi, offU, (uint32_t)0, HALF);
        qCodes.FreeTensor(cuU);

        // e8m0 -> f32, broadcast to HALF, apply
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

        // amax: max(|vlo|,|vhi|) then non-in-place ping-pong fold to 8 + scalar tail
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
        float inv = 127.0f / amax;
        // oscale through the same VECOUT TQue idiom as the int8 output (proven robust multi-core);
        // per-row 8-float (cache-line) slot in a [R,8] oscale layout.
        PipeBarrier<PIPE_ALL>();
        LocalTensor<float> oscV = qOsc.AllocTensor<float>();
        Duplicate(oscV, amax / 127.0f, 8);
        qOsc.EnQue(oscV);
        LocalTensor<float> oscVU = qOsc.DeQue<float>();
        DataCopy(gOscale[(uint64_t)r * 8], oscVU, 8);
        qOsc.FreeTensor(oscVU);

        PipeBarrier<PIPE_ALL>();  // S->V: scalar inv visible to vector Muls

        // requant in float (in-place on comb), clamp
        Muls(vlo, vlo, inv, HALF);
        Muls(vhi, vhi, inv, HALF);
        PipeBarrier<PIPE_V>();
        Mins(vlo, vlo, 127.0f, HALF); Maxs(vlo, vlo, -127.0f, HALF);
        Mins(vhi, vhi, 127.0f, HALF); Maxs(vhi, vhi, -127.0f, HALF);
        PipeBarrier<PIPE_V>();

        // store TWO CONTIGUOUS PLANES: out[r,0:HALF]=qlo, out[r,HALF:IN]=qhi (no in-kernel
        // interleave -> avoids Gather's small-src cap). Interleave deferred to a cheap torch
        // post-step: out_correct[...,0::2]=plane_lo, [...,1::2]=plane_hi.
        LocalTensor<uint8_t> outrow = qOut.AllocTensor<uint8_t>();
        LocalTensor<int8_t> outI = outrow.ReinterpretCast<int8_t>();
        Cast(offH, vlo, RoundMode::CAST_NONE, HALF);     // f32 -> half
        PipeBarrier<PIPE_V>();
        Cast(outI, offH, RoundMode::CAST_RINT, HALF);    // half -> int8 (round)
        PipeBarrier<PIPE_V>();
        Cast(offH, vhi, RoundMode::CAST_NONE, HALF);
        PipeBarrier<PIPE_V>();
        Cast(outI[HALF], offH, RoundMode::CAST_RINT, HALF);
        PipeBarrier<PIPE_V>();
        qOut.EnQue(outrow);
        LocalTensor<uint8_t> outU = qOut.DeQue<uint8_t>();
        DataCopy(gOut[(uint64_t)r * IN], outU, IN);
        qOut.FreeTensor(outU);
    }
}

extern "C" void launch_mxfp4_dq(void *stream, uint32_t blockdim,
    uint8_t *codes, uint8_t *scale, uint8_t *out, uint8_t *oscale,
    uint8_t *lutLo, uint8_t *lutHi, uint8_t *lutE8, uint8_t *scOff, uint8_t *gidxF,
    uint32_t R, uint32_t HALF, uint32_t NB, uint32_t IN)
{
    mxfp4_dq<<<blockdim, nullptr, stream>>>(
        (GM_ADDR)codes, (GM_ADDR)scale, (GM_ADDR)out, (GM_ADDR)oscale,
        (GM_ADDR)lutLo, (GM_ADDR)lutHi, (GM_ADDR)lutE8, (GM_ADDR)scOff, (GM_ADDR)gidxF,
        R, HALF, NB, IN);
}
