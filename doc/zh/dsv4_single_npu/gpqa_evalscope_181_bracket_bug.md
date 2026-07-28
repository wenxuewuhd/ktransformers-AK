# GPQA 跑分低约 3pp 的根因:evalscope 1.8.1 删掉了选项里的方括号

> 定稿 2026-07-28。**结论:历史上所有用 evalscope 1.8.1 测出的 GPQA-Diamond 分数,系统性低约 3pp。**
> 与模型、服务端、芯片、CANN、NSA、采样器**全部无关**。910B 与 910C 两条独立排查路径得到同一结论。

---

## 1. 一句话

evalscope **1.8.1** 的 GPQA 适配器会对 4 个选项做 `re.sub(r'\[.*?\]', '', text)`,
把方括号里的内容整段删掉。GPQA-Diamond 的 198 题里有 **15 题**的选项含方括号
(IUPAC 化学命名的定位符/环大小、量子态记号),被删后选项与题干对不上,模型自然答不对。
**1.9.x 已移除该清洗。**

## 2. 代码位置

`evalscope/benchmarks/gpqa/gpqa_adapter.py` → `_process_input()` → 内嵌 `preprocess()`:

```python
# 1.8.1 —— 有问题
def preprocess(text):
    if text is None:
        return ' '
    text = text.strip()
    text = text.replace(' [title]', '. ')
    text = re.sub('\\[.*?\\]', '', text)   # ★ 删掉所有方括号内容
    text = text.replace('  ', ' ')
    return text

# 1.9.1 —— 已修
def preprocess(text):
    if text is None:
        return ' '
    return text.strip()
```

`preprocess()` **只作用于选项**(`Incorrect Answer 1/2/3` + `Correct Answer`),题干不过它。
所以被破坏的是 `A) B) C) D)` 四行,题干完好 —— 这正是"题目问 spiro[4.5]decan,选项却写
spirodecan"的来历。

## 3. 破坏样例(实测)

| 原文(1.9.1) | 被删后(1.8.1) |
|---|---|
| `benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone` | `benzotrifuran-1,3,4,6,7,9-hexaone` |
| `2,8-dimethylspiro[4.5]decan-6-ol` | `2,8-dimethylspirodecan-6-ol` |
| `1,2,3,4-tetrahydro-[1,1'-biphenyl]-4-ol` | `1,2,3,4-tetrahydro--4-ol` |

## 4. 受影响题目(0-based index,共 15 题)

```
8  24  27  32  55  64  76  77  81  97  118  156  171  177  185
```

判定方法:同一份数据集,分别用两个版本跑,逐题比 `predictions/*.jsonl` 里
`messages[role=user].content`。同版本之间 0 题不同,跨版本 15 题不同。

## 5. 量化对账(910B,同一模型、同一服务端配置)

| 分组 | 受污染 15 题 | 干净 183 题 | 全部 198 题 |
|---|---|---|---|
| **evalscope 1.8.1**(6 轮 / 2 boot) | **41.33%** | 70.16% | **67.85% ± 1.83pp** |
| **evalscope 1.9.1**(9 轮 / 3 boot) | **61.11%** | 71.77% | **70.88% ± 0.87pp** |
| 差 | **−19.8pp** | −1.6pp(噪声) | **−3.03pp** |

- 置换检验 6v9(5005 种划分):**p = 0.0006**;两组除 69.70 并列外完全分离。
- 加权还原:`0.0758 × 19.8 + 0.924 × 1.6 = 3.0pp`,与实测 3.03pp 吻合。
- **干净的 183 题两组只差 1.6pp(噪声内)⇒ 模型与服务端完全一致,差异 100% 来自 harness。**

逐 boot 均值(每 boot 3 轮):

| boot | 拉起方式 | 卡 | chunk | harness | mean |
|---|---|---|---|---|---|
| A | 命令行 launcher | 3 | 32768 | 1.8.1 | **67.85** |
| E | 命令行 launcher | 4 | 32768 | 1.8.1 | **67.85** |
| B | `1_serve.sh` | 6 | 8192 | 1.9.1 | 71.21 |
| D | `1_serve.sh` | 6 | 32768 | 1.9.1 | 70.71 |
| C | `1_serve.sh`(另一台机器) | — | 8192 | 1.9.1 | 70.71 |

两个 1.8.1 的 boot 给出**完全相同**的均值 67.85%。

## 6. 910C 独立复现

910C 侧独立排查(不同机器、不同人)得到同一根因,且:

- 同样是 **15/198** 题受影响;
- 举的实锤样例同样是 **idx 24** 的 `spiro[4.5]decan`;
- 受污染题上 35.2%(旧) vs 60.0%(新),与本仓的 41.3% / 61.1% 同量级;
- 其余 183 题差 2.1pp ≈ 0.85σ,判为噪声 —— 与本仓的 1.6pp 一致。

910C 侧同时排除了:芯片、CANN、NSA(teacher-forced logprob 无方向偏置、不随上下文增长)、
采样内核(2M 抽样卡方,同 seed 随机流跨机一致)、197 题同 CoT 最终答案 0 翻转。

## 7. 为什么这么难发现:版本和调用路径绑死了

`script/dsv4_single_npu/2_gpqa_5x.sh` 原来有:

```bash
command -v evalscope >/dev/null 2>&1 || "$PY" -m pip install -q evalscope   # ← 没锁版本
EVALSCOPE="$(command -v evalscope)"
```

于是同一台机器上出现两份 evalscope:

- 走 `2_gpqa_5x.sh` → 脚本发现系统 python 没有 → **自动 pip 装最新版(1.9.1)** → 用它;
- 显式 `EVALSCOPE=/workspace/code/dsv4-acc-compare/evalvenv/bin/evalscope` → **1.8.1**。

本机 `/usr/local/python3.11.14/bin/evalscope` 的创建时间(03:15)与第一次跑
`2_gpqa_5x.sh`(03:14:29)完全对应。

**后果**:"用不用脚本拉服务"与"用哪个 harness"完美共线,导致一整天的排查方向被带偏 ——
先后误判为 `chunked_prefill_size`、boot 间方差、命令行 launcher 路径。
其中"命令行 launcher 有问题"曾达到 p=0.0015,但用探针对比两条拉起路径的进程状态
(11 个模块的 realpath+sha256、`sys.path`、完整环境、cwd)证明**逐字节相同**,
从机制上排除,才把注意力逼回评测端。

## 8. 已做的修复

1. `script/dsv4_single_npu/2_gpqa_5x.sh`:锁 `EVALSCOPE_VERSION=1.9.1`;
   每次运行打印实际使用的 **bin 路径 + 版本**;检测到 1.8.x 时打印显式警告。
2. `tools/p27_launch_ds4flash_npu.sh`:启动时 dump `[p27][env]` 环境快照
   (`KT_*` / `SGLANG_*` / `CHUNKED_PREFILL_SIZE` / `MEM_FRACTION` / …)。
   绝大多数 `KT_*` 既不进 sglang 的 `ServerArgs`、也不被逐条回显,事后无法从日志还原。

## 9. 对外口径的影响(★ 需要复核所有历史数字)

**修正后:910B / depool / GPQA-off = `70.88% ± 0.87pp`(9 轮 / 3 boot / evalscope 1.9.1)。**

需要重新审视的历史数字 —— 凡用 1.8.1 测的都要 **+约 3pp** 才是真值:

| 出处 | 原值 | 说明 |
|---|---|---|
| `accuracy_report.md` §0 clean-code 前 10× | 68.99% | 与 910C 本次 5 轮配对种子的 68.99% 分毫不差 ⇒ 确认是 1.8.1 |
| `accuracy_report.md` §0 clean-code 后 10× | 67.53% | 同上 |
| `accuracy_report.md` §0 早期 9 次单跑 | 68.13% | 同上 |

★ **一个反转**:memory / 报告里曾把 **68.99% 定为权威、把 71.72% 判为作废**。
现在看 68.99% 是坏 harness 的产物,真值 ≈ 70.9%,**反而是被作废的 71.72% 更接近**。
但 71.72% 那一组仍有独立的溯源问题(其 R2=72.73% 全机无产物、R3=73.23% 是
PR #25144 在 tp16 上自报的基线),**不能因数字接近就恢复它**;
"用 68.99 取代 71.72"这个决定方向是错的,应改用本次 9 轮的 70.88%。

## 10. 复现方法

```bash
# 看某一次跑用了哪个 harness(1.9 之后脚本会直接打印;历史归档用这个)
grep -m1 evalscope_version <run_dir>/*/configs/task_config.yaml

# 跨版本逐题比 prompt(同版本应 0 题不同,跨版本应 15 题不同)
python3 - <<'PY'
import glob,json
def load(b):
    d=sorted(glob.glob(b+"/*/"))[-1]; r={}
    for p in glob.glob(d+"predictions/**/*.jsonl",recursive=True):
        for l in open(p):
            o=json.loads(l)
            u=[m for m in (o.get("messages") or []) if m.get("role")=="user"]
            if u: r[o["index"]]=u[0]["content"]
    return r
A=load("<用 1.8.1 跑的 run 目录>"); B=load("<用 1.9.x 跑的 run 目录>")
print(sorted(i for i in set(A)&set(B) if A[i]!=B[i]))
PY
```

## 11. 给 910C 的同步要点

1. **查是否有多份 evalscope**。本次事故在 910B 上不是"机器装了老版本",而是
   同一台机器上 venv 与系统 python 各有一份,**取决于这次调用解析到哪一个**。
   `command -v evalscope`、`python -c "import evalscope;print(evalscope.__file__)"`、
   以及各 venv 都要查一遍。
2. **版本号统一**。910C 计划回归 1.9.0,本仓锁的是 **1.9.1**。
   两边必须用同一个版本号,否则下次又多一个不可比因素。
   (1.9.0 与 1.9.1 是否在 GPQA 路径上还有差异,未核对。)
3. **重测别只比均值,比"恒对集合"**。198 题里约 105 题恒对、20 题恒错、73 题翻转;
   均值要跨 ±3.3pp 噪声带,而恒对集合的变化几乎是确定性的,灵敏度高一个量级。
4. **随机化单位是 boot,不是轮**。同一次服务内连跑 N 轮,共享 seed 与该 boot 的其他状态;
   跨配置比较必须多 boot(本仓实测:同配置两个 boot 可差 2.9pp,比配置效应还大)。
