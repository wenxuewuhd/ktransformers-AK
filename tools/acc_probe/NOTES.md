# acc_probe 工作笔记(910B vs 910C GPQA ~2pp 定位)

## 进行中
- 样本集 `data/samples_gpqa_r1.json`:24 条真实(R1 逐题,prompt 渲染与服务端 input_tokens 逐条一致)+ syn4k/syn8k 合成长文本。
- probe:teacher-forced 纯 prefill,`/generate` input_ids + return_logprob + logprob_start_len=0,temp=0,bs=1 串行,逐条落盘。
- 服务配置:两边 `KT_PREFILL_STREAM=0`(对称;规避 910C 流式 prefill 拷贝线程 segfault,亦更贴近评测真实路径 —— GPQA prompt 大多 <512 本就走 hybrid)。

## 顺手发现的 bug(与 2pp 无直接关系,但都在嫌疑链路上)
1. **KT_PREFILL_STREAM=1 + return_logprob 全位置 → segfault**
   `kt_stream_prefill.py:154 _par_copy` 拷贝线程(mmap→pinned)SIGSEGV,触发条件:本服务首条 ≥512 token(首次流式 prefill)。
   注:当时 card 0 有另一个评测服务共享 host(memcg 环境)。待复现/收窄。
2. **NSA single 模式 2-chunk prefill 断言崩**(★ 嫌疑链路上的实锤边界 bug)
   n=9612(>8192,两 chunk)第二 chunk:`memory_pool_npu.py:532 set_kv_buffer`
   `AssertionError: loc.numel()=2403 vs cache.shape[0]=2049`(2403=ceil(9612/4) 全序列压缩槽 vs 2049=本 chunk)。
   `nsa_indexer.py:593 compressor_epilog → hybrid_swa_c4_c128_memory_pool.py:403 set_compress_buffer`。
   GPQA prompt < 8192 踩不到 → 不解释 2pp;但说明 single 实现的 chunked 状态处理有边界错误。
   TODO:910B/split 跑 syn8k 对照(采集完 910B 主数据后最后做,防崩服务)。

## Phase 1 结果(2026-07-28 深夜,25 样本跨机对拍)
1. **同机自控 = bit 级零差**:两台各自重启后 6 样本逐位置 logprob 全 0 差(910C 邻卡还跑着评测)。
   → 跨机全部 Δ 都是真实的 芯片+CANN(8.5 split vs 9.0 single)差异,不是抖动。
2. **跨机分歧显著大于 FP 本底**:mean|Δlogprob| 0.07–0.42/样本,逐位置 top-1 翻转 2–11%。
   形态:头部(前 256 tok)最大、随上下文衰减 —— 高熵位置对数值扰动敏感,属预期。
3. **无方向性偏置**:25 条回答总 seq-logprob 910C=-8173.1 vs 910B=-8180.4(几乎打平,C 略高);
   逐样本 signed mean ±0.02 两侧摆。910C 没有"给好答案打低分"。
4. **锐度打平**:pooled top1 logprob -0.2407(C) vs -0.2413(B);top1-top2 margin 6.192 vs 6.199。
   "910C 噪声大→分布更平→temp=1 更容易走岔"机制不成立。
5. **长上下文微弱回升信号**:syn4k 是唯一后半段>前半段的样本;idx127(2795tok)max|Δ|=17.3;
   [2048,4096) 位置桶 mean|Δ| 是 [1024,2048) 的 2 倍。与 NSA 压缩长序列占比升高一致,但 GPQA 数据
   都 <3.2k,尚不足以归因 2pp。待加密 5-8k 样本细化曲线。

→ 倾向:2pp 更像统计噪声;决定性证据 = 197 题同 CoT 字母级贪心精度 A/B(letter_probe,进行中)。

## Phase 1.5 追加(07-29 凌晨)
6. **letter 判决:197 题 0 翻转**,同 CoT 下两机贪心字母精度同为 65.48%,margin 11.5/11.7。
   → 最终决策位完全一致;若差距为真,只能来自 temp=1 的 CoT 轨迹分岔。
7. **长曲线加密(2.8k-7.5k 5 条)**:|Δ| 不随长度增长(平台 0.065-0.075,翻转 2-4% 平稳)。
   → NSA "随上下文放大偏差"机制否掉;跨机差异 = 全深度均匀数值本底(通用 CANN/芯片 FP 形态)。
8. **syn8k 两 chunk 崩溃 = split/single 通杀**(910B 同断言 2403 vs 2049)→ 不是 single 特有,
   根因 = fused compressor 无 chunked-prefill 支持:ascend_backend.py:869 `TODO support chunk prefill`,
   compressor 输出按本 forward q tokens 定尺寸(min(t,t//4+bs):8192→2049),c4_loc 按全序列分配
   (common.py:682 prefix_lens_kv 硬编码 0,`TODO: prefixcache` → 9612//4=2403)。>8192 prompt 必崩(缺陷,待修)。
9. **今日 910C 5×(用户跑,card0,R5 未完)= 64.65/66.16/68.69/67.17,mean 66.67** —— 910C 14 次
   累计 mean ~68.3 vs 910B 6 次 ~71.0,差 ~2.6pp、~4σ。**噪声解释承压,更像真差距。**
10. **★新头号嫌疑:temp=1 采样内核。** eval 路径 = sampler.py:272 `torch.softmax + torch.multinomial`
    直接跑在 NPU 上(sampling_backend=pytorch、top_p=1 → simple case)。teacher-forced 全套指标都
    绕过它 —— "前向无偏 + 采样掉点"与全部证据吻合。两机 torch_npu(post2 vs post4)/CANN 内核不同。
    审计:sampler_audit.py 构造已知分布在两边 NPU 各抽 2M 次做卡方(进行中)。

## ★★ 根因(07-29 01:20 定案):evalscope 版本差 → 15/198 题选项文本被污染
- **910C = evalscope 1.8.1,910B = 1.9.0**。1.8.1 的 gpqa_adapter.preprocess 对选项做
  `re.sub('\[.*?\]','')` 等清洗;1.9.0 移除了该清洗(只留 strip)。
- 15/198 题选项含方括号(化学命名 `spiro[4.5]decan`、稠环位标、量子态记号等)→ 910C 出题时
  **选项关键信息被删**(实证:idx24 题干 `spiro[4.5]decan-6-one` vs 910C 选项 `spirodecan-6-ol`,
  910B 选项完好)。
- **量化**:受污染 15 题 910C 35.2%(5 runs) vs 910B 60.0%(3 runs),z≈2.6 → 贡献 **+1.88pp**
  ≈ 全部历史差距(68.99 vs 70.71/71.21);其余 183 题差 2.1pp ≈ 0.85σ = 噪声。
- 与前面所有"跨机前向无偏、同 CoT 零翻转、采样内核忠实"完全自洽:喂同样文本时两机等价,
  差距只在"各自 harness 渲染的题目不同"。**芯片/CANN/NSA 全部洗清(就 GPQA 2pp 而言)。**
- **修复**:910C 升 evalscope==1.9.0(等过夜配对跑完再动环境,避免序列内不一致);
  次日验证跑预期 910C ≈ 70-71%。
- 过夜配对数据的用法修正:affected-15 用于复证根因;rest-183 的逐 seed 配对 = 纯净的跨机 A/B。

## 时间线
- 2026-07-28 22:0x 910C R1 probe(stream=1)崩 @ 首条 ≥512;22:2x(stream=0)跑到 25/26,崩 @ syn8k。
