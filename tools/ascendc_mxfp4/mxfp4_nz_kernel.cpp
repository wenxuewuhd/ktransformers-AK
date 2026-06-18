// AscendC MXFP4 -> W8A8 that writes FRACTAL_NZ int8 DIRECTLY (no per-layer transpose / format_cast).
// Requires codes/scale PRE-TRANSPOSED at load to channel-inner layout:
//   ct: [E, IN/2, OUT] uint8   (ct[e,k,oc] = nibbles for channel oc, inputs 2k (lo) / 2k+1 (hi))
//   st: [E, NB,   OUT] uint8   (st[e,sb,oc] = e8m0 for channel oc, input-block sb; NB=IN/32)
// Output NZ int8 layout (per expert, H=IN W=OUT):
//   p(in,out) = ((out/32)*(IN/16) + in/16)*512 + (in%16)*32 + (out%32)
// OUT-wide vectorized: for each code byte k, decode ALL OUT channels at once (wide vectors),
// accumulate per-channel amax (lane = channel). Then a second pass quantizes and writes each
// input row as OUT/32 contiguous 32-byte tile slices (32B-aligned strided DataCopy). The transpose
// is gone because codes are already channel-inner; the dequant naturally emits tile rows.
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t OUT_MAX = 4096;

extern "C" __global__ __aicore__ void mxfp4_nz(
    GM_ADDR codesT, GM_ADDR scaleT, GM_ADDR outNZ, GM_ADDR oscaleg,
    GM_ADDR lutLoG, GM_ADDR lutHiG, GM_ADDR lutE8G,
    uint32_t E, uint32_t OUT, uint32_t IN)
{
    const int32_t blkid = GetBlockIdx();
    const int32_t nblk = GetBlockNum();
    const uint32_t HALF = IN / 2;     // code bytes per channel
    const uint32_t NB = IN / 32;      // scale blocks per channel
    const uint32_t IN16 = IN / 16;    // tile-rows per channel-group column
    const uint32_t WBN = OUT / 32;    // number of 32-col tile groups

    GlobalTensor<uint8_t> gC, gS;
    GlobalTensor<int8_t> gOut;
    GlobalTensor<float> gOsc, gLutLo, gLutHi, gLutE8;
    gC.SetGlobalBuffer((__gm__ uint8_t *)codesT);
    gS.SetGlobalBuffer((__gm__ uint8_t *)scaleT);
    gOut.SetGlobalBuffer((__gm__ int8_t *)outNZ);
    gOsc.SetGlobalBuffer((__gm__ float *)oscaleg);
    gLutLo.SetGlobalBuffer((__gm__ float *)lutLoG);
    gLutHi.SetGlobalBuffer((__gm__ float *)lutHiG);
    gLutE8.SetGlobalBuffer((__gm__ float *)lutE8G);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> qC, qS;
    TQue<QuePosition::VECOUT, 1> qO;
    pipe.InitBuffer(qC, 2, OUT_MAX * sizeof(uint8_t));
    pipe.InitBuffer(qS, 2, OUT_MAX * sizeof(uint8_t));
    pipe.InitBuffer(qO, 2, OUT_MAX * sizeof(uint8_t));
    TBuf<TPosition::VECCALC> tLo, tHi, tSc, tAmaxA, tAmaxB, tInv, tOffH, tOff, tAbs, tLL, tLH, tLE;
    pipe.InitBuffer(tLo, OUT_MAX * sizeof(float));
    pipe.InitBuffer(tHi, OUT_MAX * sizeof(float));
    pipe.InitBuffer(tSc, OUT_MAX * sizeof(float));
    pipe.InitBuffer(tAmaxA, OUT_MAX * sizeof(float));
    pipe.InitBuffer(tAmaxB, OUT_MAX * sizeof(float));
    pipe.InitBuffer(tInv, OUT_MAX * sizeof(float));
    pipe.InitBuffer(tOffH, OUT_MAX * sizeof(half));
    pipe.InitBuffer(tOff, OUT_MAX * sizeof(int32_t));
    pipe.InitBuffer(tAbs, OUT_MAX * sizeof(float));
    pipe.InitBuffer(tLL, 256 * sizeof(float));
    pipe.InitBuffer(tLH, 256 * sizeof(float));
    pipe.InitBuffer(tLE, 256 * sizeof(float));

    LocalTensor<float> lo = tLo.Get<float>();
    LocalTensor<float> hi = tHi.Get<float>();
    LocalTensor<float> sc = tSc.Get<float>();
    LocalTensor<float> amA = tAmaxA.Get<float>();
    LocalTensor<float> amB = tAmaxB.Get<float>();
    LocalTensor<float> inv = tInv.Get<float>();
    LocalTensor<half> offH = tOffH.Get<half>();
    LocalTensor<int32_t> off = tOff.Get<int32_t>();
    LocalTensor<float> absb = tAbs.Get<float>();
    LocalTensor<float> lutLo = tLL.Get<float>();
    LocalTensor<float> lutHi = tLH.Get<float>();
    LocalTensor<float> lutE8 = tLE.Get<float>();

    DataCopy(lutLo, gLutLo, 256);
    DataCopy(lutHi, gLutHi, 256);
    DataCopy(lutE8, gLutE8, 256);
    PipeBarrier<PIPE_ALL>();

    // decode one k-slice of OUT channels into lo[OUT], hi[OUT] (scaled). cu already in UB.
    auto decode = [&](LocalTensor<uint8_t> &cu, LocalTensor<uint8_t> &su) {
        Cast(offH, cu, RoundMode::CAST_NONE, OUT);
        Muls(offH, offH, (half)4.0, OUT);
        Cast(off, offH, RoundMode::CAST_RINT, OUT);
        LocalTensor<uint32_t> offU = off.ReinterpretCast<uint32_t>();
        Gather(lo, lutLo, offU, (uint32_t)0, OUT);
        Gather(hi, lutHi, offU, (uint32_t)0, OUT);
        Cast(offH, su, RoundMode::CAST_NONE, OUT);
        Muls(offH, offH, (half)4.0, OUT);
        Cast(off, offH, RoundMode::CAST_RINT, OUT);
        Gather(sc, lutE8, off.ReinterpretCast<uint32_t>(), (uint32_t)0, OUT);
        PipeBarrier<PIPE_V>();
        Mul(lo, lo, sc, OUT);
        Mul(hi, hi, sc, OUT);
        PipeBarrier<PIPE_V>();
    };

    for (uint32_t e = blkid; e < E; e += nblk) {
        const uint64_t cBase = (uint64_t)e * HALF * OUT;
        const uint64_t sBase = (uint64_t)e * NB * OUT;
        const uint64_t oBase = (uint64_t)e * IN * OUT;

        // ---- pass 1: per-channel amax over all inputs ----
        Duplicate(amA, (float)0.0, OUT);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> cur = amA, nxt = amB;
        for (uint32_t k = 0; k < HALF; k++) {
            LocalTensor<uint8_t> cu = qC.AllocTensor<uint8_t>();
            DataCopy(cu, gC[cBase + (uint64_t)k * OUT], OUT);
            qC.EnQue(cu); cu = qC.DeQue<uint8_t>();
            LocalTensor<uint8_t> su = qS.AllocTensor<uint8_t>();
            DataCopy(su, gS[sBase + (uint64_t)(k >> 4) * OUT], OUT);
            qS.EnQue(su); su = qS.DeQue<uint8_t>();
            decode(cu, su);
            qC.FreeTensor(cu); qS.FreeTensor(su);
            Abs(absb, lo, OUT);
            Abs(lo, hi, OUT);                 // reuse lo as |hi|
            Max(absb, absb, lo, OUT);         // max(|lo|,|hi|) this k
            PipeBarrier<PIPE_V>();
            Max(nxt, cur, absb, OUT);         // accumulate (distinct dst, ping-pong)
            PipeBarrier<PIPE_V>();
            LocalTensor<float> t = cur; cur = nxt; nxt = t;
        }
        // oscale = max(amax,1e-8)/127 ; inv = 127/that
        Maxs(cur, cur, (float)1e-8, OUT);
        PipeBarrier<PIPE_V>();
        Muls(absb, cur, (float)(1.0 / 127.0), OUT);   // oscale
        PipeBarrier<PIPE_V>();
        // write oscale[e*OUT .. +OUT] (contiguous)
        LocalTensor<float> oscOut = qO.AllocTensor<float>();
        DataCopy(oscOut, absb, OUT);
        qO.EnQue(oscOut); oscOut = qO.DeQue<float>();
        DataCopy(gOsc[(uint64_t)e * OUT], oscOut, OUT);
        qO.FreeTensor(oscOut);
        Duplicate(inv, (float)127.0, OUT);
        PipeBarrier<PIPE_V>();
        Div(inv, inv, cur, OUT);
        PipeBarrier<PIPE_V>();

        // ---- pass 2: quantize + write tile rows ----
        const uint16_t bc = (uint16_t)WBN;
        const uint16_t dstr = (uint16_t)(IN16 * 16 - 1);   // 32B-block gap between tile groups
        for (uint32_t k = 0; k < HALF; k++) {
            LocalTensor<uint8_t> cu = qC.AllocTensor<uint8_t>();
            DataCopy(cu, gC[cBase + (uint64_t)k * OUT], OUT);
            qC.EnQue(cu); cu = qC.DeQue<uint8_t>();
            LocalTensor<uint8_t> su = qS.AllocTensor<uint8_t>();
            DataCopy(su, gS[sBase + (uint64_t)(k >> 4) * OUT], OUT);
            qS.EnQue(su); su = qS.DeQue<uint8_t>();
            decode(cu, su);
            qC.FreeTensor(cu); qS.FreeTensor(su);
            Mul(lo, lo, inv, OUT);
            Mul(hi, hi, inv, OUT);
            PipeBarrier<PIPE_V>();
            Mins(lo, lo, (float)127.0, OUT); Maxs(lo, lo, (float)-127.0, OUT);
            Mins(hi, hi, (float)127.0, OUT); Maxs(hi, hi, (float)-127.0, OUT);
            PipeBarrier<PIPE_V>();
            // in = 2k (lo)
            uint32_t in0 = 2 * k, in1 = 2 * k + 1;
            LocalTensor<uint8_t> o0 = qO.AllocTensor<uint8_t>();
            LocalTensor<int8_t> oi0 = o0.ReinterpretCast<int8_t>();
            Cast(offH, lo, RoundMode::CAST_NONE, OUT);
            PipeBarrier<PIPE_V>();
            Cast(oi0, offH, RoundMode::CAST_RINT, OUT);
            PipeBarrier<PIPE_V>();
            qO.EnQue(o0); o0 = qO.DeQue<uint8_t>();
            uint64_t off0 = oBase + (uint64_t)(in0 >> 4) * 512 + (uint64_t)(in0 & 15) * 32;
            DataCopy(gOut[off0], o0.ReinterpretCast<int8_t>(), DataCopyParams(bc, 1, 0, dstr));
            qO.FreeTensor(o0);
            // in = 2k+1 (hi)
            LocalTensor<uint8_t> o1 = qO.AllocTensor<uint8_t>();
            LocalTensor<int8_t> oi1 = o1.ReinterpretCast<int8_t>();
            Cast(offH, hi, RoundMode::CAST_NONE, OUT);
            PipeBarrier<PIPE_V>();
            Cast(oi1, offH, RoundMode::CAST_RINT, OUT);
            PipeBarrier<PIPE_V>();
            qO.EnQue(o1); o1 = qO.DeQue<uint8_t>();
            uint64_t off1 = oBase + (uint64_t)(in1 >> 4) * 512 + (uint64_t)(in1 & 15) * 32;
            DataCopy(gOut[off1], o1.ReinterpretCast<int8_t>(), DataCopyParams(bc, 1, 0, dstr));
            qO.FreeTensor(o1);
        }
    }
}

extern "C" void launch_mxfp4_nz(void *stream, uint32_t blockdim,
    uint8_t *codesT, uint8_t *scaleT, int8_t *outNZ, float *oscale,
    uint8_t *lutLo, uint8_t *lutHi, uint8_t *lutE8,
    uint32_t E, uint32_t OUT, uint32_t IN)
{
    mxfp4_nz<<<blockdim, nullptr, stream>>>(
        (GM_ADDR)codesT, (GM_ADDR)scaleT, (GM_ADDR)outNZ, (GM_ADDR)oscale,
        (GM_ADDR)lutLo, (GM_ADDR)lutHi, (GM_ADDR)lutE8, E, OUT, IN);
}
