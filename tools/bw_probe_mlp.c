// bw_probe_mlp.c — read-only DRAM bandwidth/latency probe for the MoE roofline 实锤.
//   gcc -O3 -march=armv8.2-a+fp16+dotprod -pthread -o /tmp/bw_probe_mlp tools/bw_probe_mlp.c
//
// Modes:
//   bw  <cpu_list> <streams_per_thread> <prfm_dist_lines> <mb_per_thread> <secs>
//       Each thread pins to one cpu, first-touch allocates its own buffer (NUMA-local),
//       then streams it read-only with N independent streams (independent address bases
//       interleaved in the inner loop -> raises per-core memory-level parallelism).
//       Reports aggregate GB/s. streams=1 ~= current GEMV (one weight row stream).
//   lat <cpu> <mb>
//       Single-thread random pointer chase (64B nodes) -> loaded memory latency ns.
//
// cpu_list: comma list or a-b range, e.g. "0-15" or "0,1,2".
#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <arm_neon.h>

static double now_s(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

typedef struct {
  int cpu, streams, prfm, stop_flag_unused;
  size_t bytes;
  double secs_target, secs_used;
  size_t bytes_done;
  uint64_t sink;
} targ_t;

static volatile int g_go = 0, g_stop = 0;
static _Atomic int g_ready = 0;

static void pin(int cpu) {
  cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
  pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
}

static void* bw_worker(void* p) {
  targ_t* a = (targ_t*)p;
  pin(a->cpu);
  uint8_t* b = aligned_alloc(4096, a->bytes);
  memset(b, 1, a->bytes);              // first-touch local
  const int S = a->streams;
  size_t ps = (a->bytes / S) & ~(size_t)255;
  const uint8_t* base[8];
  for (int s = 0; s < S; s++) base[s] = b + (size_t)s * ps;
  uint8x16_t acc = vdupq_n_u8(0);
  __atomic_fetch_add(&g_ready, 1, __ATOMIC_SEQ_CST);
  while (!g_go) {}
  double t0 = now_s();
  size_t done = 0;
  while (!g_stop) {
    for (size_t off = 0; off + 256 <= ps; off += 256) {
      for (int s = 0; s < S; s++) {
        const uint8_t* q = base[s] + off;
        if (a->prfm) __builtin_prefetch(q + (size_t)a->prfm * 64, 0, 0);
        uint8x16_t v0 = vld1q_u8(q);
        uint8x16_t v1 = vld1q_u8(q + 64);
        uint8x16_t v2 = vld1q_u8(q + 128);
        uint8x16_t v3 = vld1q_u8(q + 192);
        acc = veorq_u8(acc, veorq_u8(veorq_u8(v0, v1), veorq_u8(v2, v3)));
      }
      done += (size_t)256 * S;   // counts the 4 touched lines per stream * 64B... see note
    }
    if (g_stop) break;
  }
  a->secs_used = now_s() - t0;
  // we only LOAD 64B of each 256B chunk per stream explicitly (4x16B spread over 4 lines)
  // but the hardware fetches whole cachelines: bytes actually moved = lines touched * 64.
  // 4 lines per 256B chunk per stream => done already equals touched bytes (4*64=256). OK.
  a->bytes_done = done;
  a->sink = vgetq_lane_u8(acc, 0);
  free(b);
  return NULL;
}

static int parse_cpus(const char* s, int* cpus, int max) {
  int n = 0;
  char* dup = strdup(s);
  for (char* tok = strtok(dup, ","); tok && n < max; tok = strtok(NULL, ",")) {
    char* dash = strchr(tok, '-');
    if (dash) {
      int a = atoi(tok), b = atoi(dash + 1);
      for (int c = a; c <= b && n < max; c++) cpus[n++] = c;
    } else cpus[n++] = atoi(tok);
  }
  free(dup);
  return n;
}

static int run_bw(int argc, char** argv) {
  if (argc < 7) { fprintf(stderr, "bw <cpus> <streams> <prfm> <mb_per_thread> <secs>\n"); return 2; }
  int cpus[256];
  int nt = parse_cpus(argv[2], cpus, 256);
  int streams = atoi(argv[3]);
  int prfm = atoi(argv[4]);
  size_t mb = (size_t)atoi(argv[5]);
  double secs = atof(argv[6]);
  if (streams < 1 || streams > 8) return 2;

  pthread_t th[256];
  targ_t ta[256];
  for (int i = 0; i < nt; i++) {
    ta[i] = (targ_t){.cpu = cpus[i], .streams = streams, .prfm = prfm,
                     .bytes = mb << 20, .secs_target = secs};
    pthread_create(&th[i], NULL, bw_worker, &ta[i]);
  }
  while (__atomic_load_n(&g_ready, __ATOMIC_SEQ_CST) < nt) {
    struct timespec w = {0, 50 * 1000 * 1000}; nanosleep(&w, NULL);
  }
  g_go = 1;
  struct timespec run = {(time_t)secs, (long)((secs - (time_t)secs) * 1e9)};
  nanosleep(&run, NULL);
  g_stop = 1;
  double tot = 0, max_s = 0; uint64_t sink = 0;
  for (int i = 0; i < nt; i++) {
    pthread_join(th[i], NULL);
    tot += (double)ta[i].bytes_done;
    if (ta[i].secs_used > max_s) max_s = ta[i].secs_used;
    sink ^= ta[i].sink;
  }
  printf("BW threads=%d streams=%d prfm=%d  %.1f GB/s  (per-thread %.2f GB/s) sink=%lu\n",
         nt, streams, prfm, tot / max_s / 1e9, tot / max_s / 1e9 / nt, (unsigned long)sink);
  return 0;
}

static int run_lat(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "lat <cpu> <mb>\n"); return 2; }
  pin(atoi(argv[2]));
  size_t mb = (size_t)atoi(argv[3]);
  size_t bytes = mb << 20, nodes = bytes / 64;
  uint64_t* buf = aligned_alloc(4096, bytes);
  memset(buf, 0, bytes);
  // Sattolo shuffle -> single cycle random permutation over node indices
  size_t* perm = malloc(nodes * sizeof(size_t));
  for (size_t i = 0; i < nodes; i++) perm[i] = i;
  uint64_t rng = 88172645463325252ull;
  for (size_t i = nodes - 1; i > 0; i--) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    size_t j = rng % i;
    size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  for (size_t i = 0; i < nodes; i++)
    buf[perm[i] * 8] = (uint64_t)(perm[(i + 1) % nodes] * 8);
  free(perm);
  volatile uint64_t idx = 0;
  size_t steps = 20 * 1000 * 1000;
  double t0 = now_s();
  uint64_t cur = 0;
  for (size_t i = 0; i < steps; i++) cur = buf[cur];
  double dt = now_s() - t0;
  idx = cur;
  printf("LAT cpu=%s mb=%zu  %.1f ns/load  (%zu steps) sink=%lu\n",
         argv[2], mb, dt / steps * 1e9, steps, (unsigned long)idx);
  free(buf);
  return 0;
}

static int run_gemv(int argc, char** argv);
int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s bw|lat ...\n", argv[0]); return 2; }
  if (!strcmp(argv[1], "bw")) return run_bw(argc, argv);
  if (!strcmp(argv[1], "lat")) return run_lat(argc, argv);
  if (!strcmp(argv[1], "gemv")) return run_gemv(argc, argv);
  return 2;
}
// ---- gemv mode: real mxfp4 dot streaming from DRAM (bypasses KT scheduling) ----
typedef struct { uint8_t e; uint8_t qs[16]; } blk_mx;
static const int8_t kv_mx[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static inline float e8m0h(uint8_t x){uint32_t b=(x<2)?(uint32_t)0x00200000<<x:(uint32_t)(x-1)<<23;float f;memcpy(&f,&b,4);return f;}
typedef struct { int cpu; size_t rows; double secs_used; size_t rows_done; float sink; } garg_t;
static void* gemv_worker(void* p){
  garg_t* a=(garg_t*)p; pin(a->cpu);
  const int nb=128; // K=4096
  size_t rows=a->rows;
  blk_mx* w=aligned_alloc(4096, rows*nb*sizeof(blk_mx));
  memset(w,0x35,rows*nb*sizeof(blk_mx));
  int8_t* act=aligned_alloc(64, nb*34); memset(act,3,nb*34);
  const int8x16_t values=vld1q_s8(kv_mx); const uint8x16_t m4b=vdupq_n_u8(0x0f);
  __atomic_fetch_add(&g_ready,1,__ATOMIC_SEQ_CST);
  while(!g_go){}
  double t0=now_s(); size_t done=0; float sum=0;
  while(!g_stop){
    for(size_t r=0;r<rows && !g_stop;r++){
      const blk_mx* x=w+r*nb; const int8_t* q8=act;
      float sumf=0;
      for(int ib=0;ib+1<nb;ib+=2){
        uint8x16_t q40=vld1q_u8(x[ib].qs), q41=vld1q_u8(x[ib+1].qs);
        int8x16_t a0=vld1q_s8(q8+ib*34+2), a1=vld1q_s8(q8+ib*34+18);
        int8x16_t a2=vld1q_s8(q8+(ib+1)*34+2), a3=vld1q_s8(q8+(ib+1)*34+18);
        int8x16_t b0=vqtbl1q_s8(values,vandq_u8(q40,m4b));
        int8x16_t b1=vqtbl1q_s8(values,vshrq_n_u8(q40,4));
        int8x16_t b2=vqtbl1q_s8(values,vandq_u8(q41,m4b));
        int8x16_t b3=vqtbl1q_s8(values,vshrq_n_u8(q41,4));
        int32x4_t p1=vdotq_s32(vdotq_s32(vdupq_n_s32(0),b0,a0),b1,a1);
        int32x4_t p2=vdotq_s32(vdotq_s32(vdupq_n_s32(0),b2,a2),b3,a3);
        sumf+=e8m0h(x[ib].e)*0.25f*vaddvq_s32(p1)+e8m0h(x[ib+1].e)*0.25f*vaddvq_s32(p2);
      }
      sum+=sumf; done++;
    }
  }
  a->secs_used=now_s()-t0; a->rows_done=done; a->sink=sum;
  free(w); free(act); return NULL;
}
// ---- gemv2: vector-accumulate variant (kills the per-2-block vaddvq+scalar serial chain) ----
static void* gemv2_worker(void* p){
  garg_t* a=(garg_t*)p; pin(a->cpu);
  const int nb=128;
  size_t rows=a->rows;
  blk_mx* w=aligned_alloc(4096, rows*nb*sizeof(blk_mx));
  memset(w,0x35,rows*nb*sizeof(blk_mx));
  int8_t* act=aligned_alloc(64, nb*34); memset(act,3,nb*34);
  const int8x16_t values=vld1q_s8(kv_mx); const uint8x16_t m4b=vdupq_n_u8(0x0f);
  __atomic_fetch_add(&g_ready,1,__ATOMIC_SEQ_CST);
  while(!g_go){}
  double t0=now_s(); size_t done=0; float sum=0;
  while(!g_stop){
    for(size_t r=0;r<rows && !g_stop;r++){
      const blk_mx* x=w+r*nb; const int8_t* q8=act;
      float32x4_t accv0=vdupq_n_f32(0), accv1=vdupq_n_f32(0);
      for(int ib=0;ib+1<nb;ib+=2){
        uint8x16_t q40=vld1q_u8(x[ib].qs), q41=vld1q_u8(x[ib+1].qs);
        int8x16_t a0=vld1q_s8(q8+ib*34+2), a1=vld1q_s8(q8+ib*34+18);
        int8x16_t a2=vld1q_s8(q8+(ib+1)*34+2), a3=vld1q_s8(q8+(ib+1)*34+18);
        int8x16_t b0=vqtbl1q_s8(values,vandq_u8(q40,m4b));
        int8x16_t b1=vqtbl1q_s8(values,vshrq_n_u8(q40,4));
        int8x16_t b2=vqtbl1q_s8(values,vandq_u8(q41,m4b));
        int8x16_t b3=vqtbl1q_s8(values,vshrq_n_u8(q41,4));
        int32x4_t p1=vdotq_s32(vdotq_s32(vdupq_n_s32(0),b0,a0),b1,a1);
        int32x4_t p2=vdotq_s32(vdotq_s32(vdupq_n_s32(0),b2,a2),b3,a3);
        // vector accumulate: two independent FMA chains, no horizontal add in the loop
        accv0=vfmaq_n_f32(accv0, vcvtq_f32_s32(p1), e8m0h(x[ib].e)*0.25f);
        accv1=vfmaq_n_f32(accv1, vcvtq_f32_s32(p2), e8m0h(x[ib+1].e)*0.25f);
      }
      sum+=vaddvq_f32(vaddq_f32(accv0,accv1));   // one horizontal per ROW
      done++;
    }
  }
  a->secs_used=now_s()-t0; a->rows_done=done; a->sink=sum;
  free(w); free(act); return NULL;
}
static void* gemv3_worker(void* p){
  garg_t* a=(garg_t*)p; pin(a->cpu);
  const int nb=128;
  size_t rows=a->rows;
  blk_mx* w=aligned_alloc(4096, rows*nb*sizeof(blk_mx));
  memset(w,0x35,rows*nb*sizeof(blk_mx));
  int8_t* act=aligned_alloc(64, nb*34); memset(act,3,nb*34);
  const int8x16_t values=vld1q_s8(kv_mx); const uint8x16_t m4b=vdupq_n_u8(0x0f);
  __atomic_fetch_add(&g_ready,1,__ATOMIC_SEQ_CST);
  while(!g_go){}
  double t0=now_s(); size_t done=0; float sum=0;
  while(!g_stop){
    for(size_t r=0;r<rows && !g_stop;r++){
      const blk_mx* x=w+r*nb; const int8_t* q8=act;
      const uint8_t* nxt=(const uint8_t*)(w+((r+1<rows)?(r+1):0)*nb);
      float32x4_t accv0=vdupq_n_f32(0), accv1=vdupq_n_f32(0);
      for(int ib=0;ib+1<nb;ib+=2){
        __builtin_prefetch((const uint8_t*)&x[ib]+512,0,0);   // ~30 blocks ahead in-row
        if((ib&15)==0) __builtin_prefetch(nxt+(ib<<2),0,0);   // warm next row head
        uint8x16_t q40=vld1q_u8(x[ib].qs), q41=vld1q_u8(x[ib+1].qs);
        int8x16_t a0=vld1q_s8(q8+ib*34+2), a1=vld1q_s8(q8+ib*34+18);
        int8x16_t a2=vld1q_s8(q8+(ib+1)*34+2), a3=vld1q_s8(q8+(ib+1)*34+18);
        int8x16_t b0=vqtbl1q_s8(values,vandq_u8(q40,m4b));
        int8x16_t b1=vqtbl1q_s8(values,vshrq_n_u8(q40,4));
        int8x16_t b2=vqtbl1q_s8(values,vandq_u8(q41,m4b));
        int8x16_t b3=vqtbl1q_s8(values,vshrq_n_u8(q41,4));
        int32x4_t p1=vdotq_s32(vdotq_s32(vdupq_n_s32(0),b0,a0),b1,a1);
        int32x4_t p2=vdotq_s32(vdotq_s32(vdupq_n_s32(0),b2,a2),b3,a3);
        accv0=vfmaq_n_f32(accv0, vcvtq_f32_s32(p1), e8m0h(x[ib].e)*0.25f);
        accv1=vfmaq_n_f32(accv1, vcvtq_f32_s32(p2), e8m0h(x[ib+1].e)*0.25f);
      }
      sum+=vaddvq_f32(vaddq_f32(accv0,accv1));
      done++;
    }
  }
  a->secs_used=now_s()-t0; a->rows_done=done; a->sink=sum;
  free(w); free(act); return NULL;
}
static int run_gemv(int argc,char**argv){
  if(argc<5){fprintf(stderr,"gemv <cpus> <mb_per_thread> <secs>\n");return 2;}
  int cpus[256]; int nt=parse_cpus(argv[2],cpus,256);
  size_t mb=(size_t)atoi(argv[3]); double secs=atof(argv[4]);
  size_t rows=(mb<<20)/(128*17);
  int variant = (argc>5)?atoi(argv[5]):1;
  void* (*wfn)(void*) = (variant==3)?gemv3_worker:((variant==2)?gemv2_worker:gemv_worker);
  pthread_t th[256]; garg_t ga[256];
  for(int i=0;i<nt;i++){ga[i]=(garg_t){.cpu=cpus[i],.rows=rows};pthread_create(&th[i],NULL,wfn,&ga[i]);}
  while(__atomic_load_n(&g_ready,__ATOMIC_SEQ_CST)<nt){struct timespec w={0,50000000};nanosleep(&w,NULL);}
  g_go=1;
  struct timespec run={(time_t)secs,(long)((secs-(time_t)secs)*1e9)}; nanosleep(&run,NULL);
  g_stop=1;
  double tot=0,max_s=0; float sink=0;
  for(int i=0;i<nt;i++){pthread_join(th[i],NULL);tot+=(double)ga[i].rows_done*128*17;if(ga[i].secs_used>max_s)max_s=ga[i].secs_used;sink+=ga[i].sink;}
  printf("GEMV threads=%d  %.1f GB(w)/s aggregate (per-thread %.2f GB/s) sink=%f\n",nt,tot/max_s/1e9,tot/max_s/1e9/nt,sink);
  return 0;
}
