# H3 rewriter：统一评价、TRL SFT 与 GRPO runbook

此次用户指定的新评价规则见 [固定评价规范](h3_prompt_rewriter_fixed_evaluation.md)：Astra low 101 条模型评测与 Luna 日常 reward。FAL/9B v2 的 30 条校准已复核并冻结规则，新重训 EP1–EP5 已完成标准评审；以下各节保留历史协议，不应把旧状态审查、待复核或未要求配乐扣分混入新评测。

更新：2026-10-06。**v4 已完成训练；新版 reward 见 §14，统一 FAL/EP3/GRPO 审查见 §15；§7–8 保留旧 v3 实验协议与迁移记录。**适用于 HB10 的 Qwen3.5-9B、retention v2 system、官方 Qwen 模板，以及原始 101 条 t2va/i2va 测试集。代码位于 [仓库根目录](../../)。数据制作、官方 Context-IR 标签验证及历史五组比较见 [数据 runbook](h3_prompt_rewriter_data_runbook.md)。

## 1. 评价对象与独立指标

每条审查同时读取原始用户文字、任务/时长/比例/引用角色、未经改写的模型原始输出。原始要求是内容判断依据；teacher rewrite 不是绝对正确答案。判断视觉、动作、物体、人物、音效、音乐、台词及其绑定关系；不按关键词相似度打分。

以下指标分开报告，错误可重叠，不能相加成“总错误率”：

| 维度 | 怎么检查 | 通过意味着什么 |
|---|---|---|
| 原文保留 | 模型逐项语义审查，保存原文及输出证据 | 没发现明确遗漏或冲突；不保证实际生成视频正确 |
| 错误严重程度 | 对每个问题分类，再取每个 case 的最高等级 | 用于区分训练优先级，不等于历史“明确问题”分类 |
| 机械结构 | `grpo/format_rules.py`，保留原始输出 | 字段、顺序、镜头、时间、标签符合本检查器 |
| v2 语义遵循 | 不新增未要求配乐/确切台词，不改变剧情，正确归属说话者等 | 需语义判断，不能由字段存在证明 |
| 引用图像忠实度 | 必须提供原始图像另做视觉审查 | 本次纯文本 Luna/Sol 审查未验证这一项 |
| H3 可接收性 | H3 tokenizer、引用索引、任务头、实际 encoder/generation gate | 独立于 Qwen 的 2048-token 输出上限和机械格式通过率 |

## 2. 内容判定 codebook

先拆出原文的可核对要求：谁/什么物体，动作及受事，数量，速度，位置，镜头，否定约束，先后/因果，中间过程与结局，文字/台词，音效身份/强弱/同步，音乐。逐项找 rewrite 的直接证据；同义表达可以保留，静态结果不等于动作，慢镜头移动不等于主体慢动作，准备执行不等于实际执行。

| 状态 | 定义 |
|---|---|
| preserved | 要求的含义完整保留 |
| partial | 保留了一部分，但丢失限定或部分含义 |
| omitted | 找不到支持该要求的输出内容 |
| contradicted | 输出含明确不兼容的描述 |

当前 GRPO v3 返回问题清单，只对发现的问题返回 partial/omitted/contradicted；没有问题时返回空清单。它不是持久化的完整逐原子 checklist，不能把“空清单”当作所有要求已被形式化证明。完整独立审查仍应按上面的步骤复核。

| 严重程度 | 判定 | 例子 |
|---|---|---|
| severe／严重 | 关键主体、物体或身体部分缺失；关键动作/绑定、状态变化、因果或结局被改错 | 手没了；雾气应消失，却消失后又生成；把小孩的台词改成爷爷说 |
| general／一般 | 核心事件和结局保留，但局部范围、强度或镜头要求偏差 | 部分 zoom in 改成全程 zoom in |
| review／待复核 | 是否影响要求存在真实歧义，尚未确认错误 | 近义表述或镜头关系不够清楚，无法确定是否违背原意 |
| none／未发现问题 | 未发现上述问题 | 兼容原意的光线、材质、服装等合理扩写 |

镜头偏差若导致关键事件不可见，应升级严重程度；不能仅凭“镜头问题”自动降为一般。兼容的制作细节及合理物理音效不因原文没提而自动算错。上下文比例/时长不需要逐字重复进正文；但镜头时间不能越界。未要求的外部配乐必须为 `N/A`，未指定内容的讲话不能凭空添加确切台词。

记录每个问题的分类、状态、等级、原文证据、输出证据/缺失说明、简短理由。一个 case 可有多个问题；最高等级 case 分布互斥，逐类问题数不互斥。训练 reward 的音乐/台词问题包含在问题清单中，不能额外重复扣分。

## 3. 机械格式：t2va / i2va

权威实现：[format_rules.py](../.././grpo/format_rules.py)。对 raw rewrite 检查全部 13 项；全部为 true 才算机械结构通过：

1. `integrated_multimodal_description`、`overall_soundscape`、`non_diegetic_music` 三字段齐全。
2. 每个字段只出现一次。
3. 三字段按上述顺序出现。
4. 三字段内容非空；字面 `N/A` 是非空内容。
5. t2va 无前置正文；i2va 有合法首帧边界引用句。当前接受 `Image 1` 或 `<Picture 1>`、`[Shot 1]` 或 `Shot 1` 的已支持写法，允许 `from`。
6. 镜头 `[Shot 1]` 起连续编号，无跳号或重复。
7. 后续镜头以 `At MM:SS.mmm` 引入。
8. 秒字段小于 60，时间在请求 duration 内。
9. `At` 时间严格递增，不重复。
10. raw 输出不含 `<think>` 或 `</think>`。
11. 不含 Markdown 代码围栏。
12. `<d>[Language]...</d>` 标签闭合且有语言字段。
13. 每段台词之前存在 `S数字` 说话者标签。

第 13 项只检查标签存在，不验证该标签是否绑定正确人物；英文描述、无额外推理 prose、引用忠实度、音乐 `N/A` 适用性及台词内容正确性需要语义检查。当前实现不覆盖 ref2va 六字段、fl2va/l2va 任务头或完整 H3 940-token gate，不能把结果推广到这些任务。

运行独立检查器，sources 每行至少含 `id/task/duration_s`；若两边有 `original_prompt`，会校验原文一致：

```bash
python review/check_format.py --input results.jsonl --sources source_metadata.jsonl --output format_audit.json
```

不给输出删 `<think>`、补字段、重排或去掉错误段落再计算严格通过率。若展示清洗后的诊断结果，必须另列，不能替代 raw 指标。

## 4. 101 条比较与统计分母

所有模型保留同一 101 ID、原文、任务、时长、比例、50 张原图；保存首次完成结果，失败格式、超长和截断样本都保留，不挑选更好的重采样。记录 checkpoint、system/template SHA、框架版本、媒体顺序、processor 限制、EOS、解码参数及输出 token 数。

TRL SFT/GRPO 评测统一为 `do_sample=False`（greedy）、`max_new_tokens=2048`、`repetition_penalty=1`、官方 processor、`enable_thinking=False`、tokenizer EOS 248046/pad 248044。训练候选使用采样，不能拿训练采样输出冒充 greedy 基准。

新 GRPO 评价同时列三组：历史 EP3 原 adapter、合并后但关闭 GRPO adapter 的初始化控制、训练后的 GRPO。这样可以区分 BF16 adapter 合并舍入与 GRPO 本身的变化。不能把旧 LLaMA-Factory 的 18/101 直接与本次结果比较。

| 表格行 | 分母和含义 |
|---|---|
| 严重错误 case 率 | 至少一个 severe 的 case / 已完成且有效审查的 case |
| 一般错误 case 率 | 可报“存在 general”或“最高等级 general”；必须注明是哪种 |
| 待复核率 | 同理，待复核不是确定错误 |
| 格式问题率 | raw 机械检查失败 case / 全部生成 case（完整集为 101） |
| 新增未要求音乐/台词率 | 对应 v2 违规 case / 有效语义审查 case，独立列出 |
| judge 失败率 | 重试后仍失败的 case / 待审查 case；不算“无问题” |
| 两模型严重程度一致率 | 最高等级相同的有效配对 / 全部有效配对 |
| 严重问题漏判/降级 | Sol severe、Luna general/review/none 的数 / Sol severe 的有效配对数；Sol 本身也需复核 |

最终完整语义比例应在 101 条全部有效审查后给出；若仍有失败，报告覆盖数及未知数，不把部分集比例标成 `/101` 的完整结论。`evaluate.py` 的 `clear_cases` 只统计 omitted/contradicted，不包含 partial；这是辅助字段，不等于 severe，也不等于历史 Codex“明确问题”口径。历史审查表见 [四轮 SFT 审计](h3_prompt_rewriter_trl_sft_audit_20261006.md)，原始旧标签未追溯改成新版严重程度。

## 5. 发布代码和外部目录

仓库代码：`./{sft,grpo,review}`。宿主目录 `/data/xiaotong/h3_rewriter_sft_20261002` 挂载为 `/work`；脚本保留真实实验目录，不在 Git 中包含这些外部数据：

```text
/work/models/Qwen3.5-9B/                         原始模型及官方 processor/template
/work/lf_dataset/{train,val}.jsonl               固定 9659/200 的原始 chat 数据
/work/media/                                    图片/视频；引用路径须可读取
/work/grpo_20261005/env/                         已固定的训练 Python 环境
/work/grpo_20261005/system_prompt.txt            system_prompt_v2.txt 的原始字节
/work/trl_sft_official_v2_20261006/               发布 sft 源码、处理数据及四轮 checkpoint
/work/benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl
/work/trl_sft_official_v2_20261006/epoch_benchmarks/step906/results.jsonl
/work/grpo_ep3_luna_v2_20261006/                  发布 grpo 源码及运行产物
```

Luna 现在直接在 HB10 的本机 Codex CLI 上运行，不需要 Mac 或 SSH 隧道。Codex 登录保留在本机，认证文件不复制进容器。实际文本/输出/缓存、runtime receipts、env、模型、媒体和日志不提交。

## 6. 环境与 TRL SFT

固定 Python 训练依赖见 `requirements.txt`：Torch 2.8.0/cu128、torchvision .23、Transformers 5.6.0、TRL 1.14.1、PEFT .18.1、Accelerate 1.11、Datasets 4、TorchCodec .7、PyAV 16。原 SFT Dockerfile 以 Torch2.9.1 镜像加 FFmpeg 系统库，但实际执行的是绑定目录中 Torch2.8 的 env；只 build 原 Dockerfile 不会安装正确的 Python 环境。

新环境安装示例（在隔离容器/新目录做，不覆盖活跃训练 env）：

```bash
python3 -m venv --system-site-packages /work/grpo_20261005/env
/work/grpo_20261005/env/bin/pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
/work/grpo_20261005/env/bin/pip install -r /work/h3_prompt_rewriter/requirements.txt
```

运行镜像必须有 FFmpeg 共享库；检查当前 `Dockerfile`。历史运行的 task-local TorchCodec vendor 在 `/work/trl_sft_official_v2_20261006/vendor`；若直接安装正确 TorchCodec，可不依赖 vendor。完整训练、cache 一致性、模板和视觉预检是实际验证；上面的新环境安装示例本次未新建环境重跑，不替代预检。

把 sft 源码复制到 SFT JOB，把原始 v2 字节复制到 `/work/grpo_20261005/system_prompt.txt`。执行：

```bash
/work/grpo_20261005/env/bin/python /work/trl_sft_official_v2_20261006/prepare.py
python /work/h3_prompt_rewriter/bootstrap_runtime.py sft --root /work
/work/grpo_20261005/env/bin/python /work/trl_sft_official_v2_20261006/run_pipeline.py
```

pipeline 运行环境需设 `HF_HUB_OFFLINE=1`、`OMP_NUM_THREADS=1`、`PYTHONPATH=/work/trl_sft_official_v2_20261006/vendor:/work/trl_sft_official_v2_20261006`，并在有 8 GPU 的已准备容器中执行。prepare 拒绝覆盖已有处理数据，pipeline 拒绝覆盖既有 run。

SFT 使用原始 Qwen3.5-9B，语言 LoRA r64/alpha128/dropout.05，冻结视觉，8卡 microbatch1、累积4、LR1e-4、4ep，共1208step。用官方未修改模板，保留原始图片/文字占位顺序；官方非思考空 think 前缀属于输入、被 loss mask，不应出现在最终 raw rewrite。collator 只对未改 teacher answer 和 EOS 计算 loss，不静默截断，序列上限8192。全量9859预检与每类视觉组合的独立 token/tensor 检查通过后训练。EP3 是 `run/checkpoint-906`。

## 7. 当前 GRPO reward 与多线程

模型 `gpt-6-luna`、reasoning `high`、revision `ep3-v2-luna-severity-v3`。使用一次完整问题审查，返回 severity/status/source span IDs/rewrite span IDs 和理由；证据 ID、来源范围、数值 reward 校验失败会进入重试。

```text
语义 reward = 1
             - 0.8 × min(severe 问题数, 2)
             - 0.1 × min(general 问题数, 3)
             - 0.02 × min(review 问题数, 3)
最终 reward = 语义 reward - 0.3（若机械格式不通过）
```

严重错误不能被大量保留要求的平均分稀释。兼容扩写不扣分。音乐/台词违规在问题清单内计入，不能按布尔 flag 再重复扣。reward 是实验评分设计，尚不保证最佳；训练 reward 增长不能证明独立101严重错误率下降。

8 个 GPU rank 并发提交，每个 rank 的线程池最多4，Luna gateway semaphore 上限16个 CLI调用。当前 microbatch1，活跃并发可能只有8；“16”是上限。CLI 禁用 Apps/shell/web/multi-agent/技能工具初始化；只发必要文本，已有同版本结果按完整请求哈希缓存。

HB10 本机 gateway 默认监听 `127.0.0.1:8792`；训练容器使用 host network 直接访问。`H3_LUNA_PORT` 可覆盖端口，gateway 和训练进程必须设为同一值。请求失败重试3次；仍失败写 `judge_failed=true`、操作性reward -1，若格式也失败再-.3，训练继续。记录失败次数/最终失败数，不能把操作性低分当作模型发现的严重错误。

## 8. 启动与恢复 GRPO

EP3完整后，部署 grpo 源码，运行 `prepare_merge.py`：挑 t2va/i2va各128，排除视频、验证及101原文哈希、验证媒体重合，保持 SFT `prompt_json`。它不声称独立验证了101全部媒体哈希的排除。保存 EP3 adapter/template/EOS receipt 和 merged初始化；只检查一个实际文本样本合并前后 next-token argmax，相同不等于所有输出完全相同。

```bash
/work/grpo_20261005/env/bin/python /work/grpo_ep3_luna_v2_20261006/prepare_merge.py
python /work/h3_prompt_rewriter/bootstrap_runtime.py grpo --root /work
```

初始化用一个空闲 GPU，不能在活跃训练占满卡时启动。`sft_init` 已存在时 prepare 会拒绝覆盖；不要为了重启训练重跑 merge。bootstrap 为已授权任务生成本地 runtime receipts：357 个 source hash（256 训练 + 101 评测）、external-data approval 和 direct-GRPO authorization。生成文件都不入 Git。历史 HB10 的 allowlist 仅含 256 条，迁移时已补齐 101 条评测；启动新 gateway 前务必使用当前 bootstrap，不能复制旧训练专用 allowlist。

**HB10 本机启动 Luna（Linux，必须已有本机 Codex 登录）：**

```bash
codex login status
TASK_ROOT=/data/xiaotong/h3_rewriter_sft_20261002
REWRITER_SRC=/path/to/repo/.
LUNA_RUNTIME=/home/xiaotong/grpo_ep3_luna_v2_20261006/local_luna_runtime
mkdir -p "$LUNA_RUNTIME"
cp "$REWRITER_SRC/grpo/"{luna_base,luna_judge,evidence,start_luna_runtime,runtime_watchdog}.py "$LUNA_RUNTIME/"
# 在 prepare_merge 和 bootstrap grpo 后复制当前已授权 receipts。
cp "$TASK_ROOT/grpo_ep3_luna_v2_20261006/"{luna_approved_sources,luna_external_data_approval}.json "$LUNA_RUNTIME/"
cd "$LUNA_RUNTIME"
H3_LUNA_PORT=8792 python3 start_luna_runtime.py
curl -fsS http://127.0.0.1:8792/
```

本机 supervisor 保存 `runtime_watchdog.pid`、`watchdog_status.json`、`watchdog_events.jsonl` 和 `runtime_pids.json`。只有自身创建的 gateway 子进程才会被终止或重启；连续 3 次健康失败时恢复。健康但不属于本 supervisor 的 gateway 会继续使用，失败后才接管。停止本任务守护进程可创建 `RUNTIME_STOP`；重新启动前明确移除该 stop marker。不管理 SSH、Mac 或其他会话。

本次迁移先在 8792 验证一条实际训练请求（Luna high，约 7.4 秒），然后修改后续进程默认端口。已经加载 Python 的 pilot worker 继续使用旧 8791，完成 64 step 后自动启动的 101 评测使用本机 8792。这个过渡事实不应写成整轮训练全部在本机 judge 上完成。allowlist 补齐前产生的评测 judge 失败保留原始记录；可使用 `review/retry_failed_audits.py --root "$TASK_ROOT"` 在原始 rewrite 上重审，输出另存 `evaluation_101/recovered/`，不重采样或改写输出。

HB10正式运行命令（使用已准备的 `h3-rewriter-grpo:20261005` 镜像及绑定env；镜像不由本仓库自动下载）：

```bash
docker run -d --name xiaotong-grpo-ep3-luna-20261006 --gpus all --network host --shm-size=16g \
  -v /data/xiaotong/h3_rewriter_sft_20261002:/work \
  -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=1 -e H3_LUNA_PORT=8792 \
  -e PYTHONPATH=/work/grpo_ep3_luna_v2_20261006:/work/trl_sft_official_v2_20261006/vendor:/work/trl_sft_official_v2_20261006 \
  h3-rewriter-grpo:20261005 /work/grpo_20261005/env/bin/python -u /work/grpo_ep3_luna_v2_20261006/run_pipeline.py
```

`--network host` 必须能访问 host loopback 8792；训练依赖本机 gateway，不再依赖 Mac。启动前检查GPU归属，避免抢占别的运行。pipeline先2step联调，然后重新从EP3初始化跑64step；smoke权重不用于pilot。语言LoRA r32/alpha64，microbatch1、累积2、8候选/输入、LR5e-6、beta.02、DAPO、采样温度.8/top_p.95，保存step16/32/48/64及final。每step2次输入、16候选；64step共128次输入呈现、1024候选，不是全遍历256输入。

用户已决定跳过等待完整256个Luna/Sol对照，直接训练；可选审查脚本保留，训练不会等待它。不要改用旧 `runtime_controller.py` 及旧 minimum-retention calibration gate。

监控 `pipeline_status.json`、`pilot/status.json`、`pilot.log`、`rollouts.rank*.jsonl`、`judge_failures.rank*.jsonl`、`pilot/checkpoint-*/trainer_state.json`。统计最终judge_failed从rollouts读取，重试日志条数不是最终失败样本数。

非judge训练错误会停并记录状态。恢复时检查完整checkpoint、原环境和服务健康，停止自己的残留worker后，以新container名称运行同一pipeline，它选择最大step checkpoint续训；没有完整checkpoint的失败输出目录要单独保留后明确从头重跑，不直接删除覆盖。也可用相同8卡torchrun运行 `train_grpo.py --steps 64 --resume /work/grpo_ep3_luna_v2_20261006/pilot/checkpoint-N`。

## 9. 训练后验证与选择

`evaluate.py` 自动读取同一101输入，比较历史EP3、matched merged初始化、训练后GRPO，原始输出和每条审查保留，给severity、格式和judge失败指标。Luna paired audit是探索性指标；应独立核对关键分歧，不能因为reward模型给高分就称训练成功。

先看严重错误是否减少，再看一般错误、格式及新增音乐/台词是否回退；报告修复了哪些case、又新增了哪些问题。不要只给平均reward或训练loss。未完成全101审查、仍有调用失败或仅训练reward上升时，不给“GRPO已改善”的结论。当前四轮SFT表不含此次GRPO结果，训练完成后以新的协议列添加，保留旧统计的历史口径。

## 10. 数据、S3 资产与恢复

**实际数据**：原始标签 9899 条，排除 40 个未来媒体泄漏 ID，固定 9659 train / 200 val。`dataset/split_report.json` 记录实际 mode 分布、排除 ID、seed 42；不能把旧配额表当成本次实际任务统计。原始 teacher 保留，TRL prepare 仅替换完整 v2 system、保持 user/media/answer。两阶段均不提供 audio waveform 给 Qwen。

| 当前 TRL 输入 | 行数 | SHA256 |
|---|---:|---|
| `trl_sft_official_v2_20261006/train.jsonl` | 9659 | `c9f4f14894a1af29310b60c289bcacb0018548dc8f8b5eb4415bfc257485cc4c` |
| `trl_sft_official_v2_20261006/val.jsonl` | 200 | `8a0757cb4a531edfc7d361ff58d8f0e7075fc0d787bdc726bfacf2bd4edf5f02` |
| `system_prompt.txt` 原始 v2 字节 | — | `9b95942a5c905bfd8ca703cbb4c31275387e282775395278ff92a889e81d2eda` |

R2 使用 S3 API：bucket `data-transfer-research`，endpoint `https://f25b0ac4c45a2442f62961145a64d158.r2.cloudflarestorage.com`，已配置 profile `r2w`。凭据在机器外部配置中，不能写入仓库或 Docker 镜像。

统一发布根：

```text
s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/
  final/h3_rewrite_results.jsonl              历史已上传原始 teacher 标签
  final/manifest_api.jsonl                    历史已上传原始 manifest
  final/media/                               历史媒体，当前发布补传缺失对象
  training_bundle/{dataset,lf_dataset}/       历史固定 split 和 chat 导出
  trl_sft_official_v2_20261006/                当前统一发布目标
    READY.json                               发布成功及对象大小/SHA 核验 receipt
    data/{source,dataset,lf_dataset}/         当前原始数据及训练 chat 快照
    sft/{train,val}.jsonl                     实际 TRL v2 输入
    sft/system_prompt.txt                    实际 v2 原始字节
    sft/{data_report,preflight_report,official_template_receipt}.json
    sft/run/checkpoint-{302,604,906,1208}/     四轮完整可恢复 checkpoint
    sft/epoch_benchmarks/step{302,604,906,1208}/
    sft/epoch{1,2,3,4}_*audit_101.*            历史内容/格式审计
    benchmark_jobs/arena_original_101_retention_v2_20261005/
```

2026-10-06 已只读核验旧数据：S3 `training_bundle/lf_dataset/{train,val}.jsonl` 与当前本地两文件 SHA256 完全一致；全部 14116 个训练媒体引用均在 `final/media/` 中存在且大小一致，缺失数为 0。媒体没有完成逐对象远端 SHA 校验。本次无需重传已有训练媒体。当前新增 TRL 发布尚待具体写入范围批准。

发布完成状态以 `READY.json` 的 `uploaded=true/ready=true` 为准；当前目录树是明确的发布目标，不能仅凭这份文档假定尚未上传的对象存在。历史 `releases/qwen3.5-9b-h3-rewriter-step604` 和 `h3_rewriter_sft_retention_v2_20261006/assets/adapter.tar` 是旧 LLaMA-Factory adapter，**不是当前 TRL EP2 或 EP3**。已有 `h3_rewriter_sft_retention_v2_20261006/assets/base.tar` 是 HB10 导出的基础模型；使用前核对对应 `assets/READY.json` 大小/SHA，并核对官方 template SHA，不使用那个旧 adapter 作为新 SFT 初始化。

`publish_assets.py` 默认只输出本地清单；`--upload` 才执行写入。显式白名单包含数据、四轮 checkpoint（optimizer/scheduler/8 rank RNG）、历史 SFT 审计和 101 输入/图片，排除凭据、模型登录文件、env、processed cache、judge cache 和 runtime approval。已有媒体核验 key/size，并记录本地 SHA；历史对象没有远端 SHA metadata，receipt 明确写 `remote_key_and_size`，不能称之为远端 SHA256 验证。新上传资产核对远端大小和 SHA256 metadata，不把 multipart ETag 当作 SHA。

```bash
# 使用本机 boto3 工具环境；不要为上传而改动活跃训练 Python env。
export AWS_CONFIG_FILE=/data/xiaotong/h3_rewriter_sft_20261002/aws/config
export AWS_SHARED_CREDENTIALS_FILE=/data/xiaotong/h3_rewriter_sft_20261002/aws/credentials
python3 ./publish_assets.py \
  --root /data/xiaotong/h3_rewriter_sft_20261002 --receipt /tmp/h3_asset_plan.json
# 在明确授权上传目的地与范围后：同一命令添加 --upload。
```

恢复示例（先读取 READY；恢复到空目录，checkpoint-906 是 EP3）：

```bash
RESTORE_ROOT=/data/xiaotong/h3_rewriter_restore
R2_ENDPOINT=https://f25b0ac4c45a2442f62961145a64d158.r2.cloudflarestorage.com
S3_ROOT=s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002
S3_TRL="$S3_ROOT/trl_sft_official_v2_20261006"
mkdir -p "$RESTORE_ROOT"
aws s3 cp "$S3_TRL/READY.json" "$RESTORE_ROOT/READY.json" --profile r2w --endpoint-url "$R2_ENDPOINT"
aws s3 sync "$S3_TRL/data/lf_dataset/" "$RESTORE_ROOT/lf_dataset/" --profile r2w --endpoint-url "$R2_ENDPOINT"
aws s3 sync "$S3_TRL/data/dataset/" "$RESTORE_ROOT/dataset/" --profile r2w --endpoint-url "$R2_ENDPOINT"
aws s3 sync "$S3_ROOT/final/media/" "$RESTORE_ROOT/media/" --profile r2w --endpoint-url "$R2_ENDPOINT"
aws s3 sync "$S3_TRL/sft/" "$RESTORE_ROOT/trl_sft_official_v2_20261006/" --profile r2w --endpoint-url "$R2_ENDPOINT"
aws s3 sync "$S3_TRL/benchmark_jobs/" "$RESTORE_ROOT/benchmark_jobs/" --profile r2w --endpoint-url "$R2_ENDPOINT"
```

上述 data JSONL 保留 `/work` 路径。容器必须把 `RESTORE_ROOT` 挂载到 `/work`；不要在 JSONL 上批量替换或重新排序媒体。基础模型和固定训练 env 仍需按 §6 准备；它们不在当前数据/checkpoint 发布包中。

## 11. TRL SFT 实际结果、checkpoint 与启动

SFT 已完成 4 epoch / 1208 optimizer step；`run/COMPLETE` 和四轮 `trainer_state.json` 均存在。最后重复 on_save 触发已完成 benchmark 覆盖保护，随后由 checkpoint-1208 完成收尾，**没有新增 optimizer step**；事实记录见 `finalization_recovery.json`、`pipeline_status.json`。完整文本审计见 [四轮审计](h3_prompt_rewriter_trl_sft_audit_20261006.md)。

| 轮次 | checkpoint（HB10：`trl_sft_official_v2_20261006/run/`） | 验证 loss | 明确遗漏/冲突 | 格式问题 | 未要求配乐 |
|---|---|---:|---:|---:|---:|
| EP1 | `checkpoint-302` | 1.02736 | 35/101 | 0/101 | 66/101 |
| EP2 | `checkpoint-604` | 1.00575 | 31/101 | 9/101 | 59/101 |
| EP3 | `checkpoint-906` | 1.02714 | 27/101 | 1/101 | 58/101 |
| EP4 | `checkpoint-1208` | 1.06513 | 30/101 | 8/101 | 62/101 |

EP3 按这批 held-out 内容与格式结果选为 GRPO 初始化，并非按最低 val loss 选择。这些历史“明确问题”标签不等同于新 severity severe；未要求配乐独立统计，不能相加。S3 checkpoint 发布路径为 §10 的 `.../sft/run/checkpoint-N/`，以 READY 核验为准。

**从头启动 SFT**（新实验根，不能对已有 `run/` 重跑 pipeline）：

```bash
TASK_ROOT=/data/xiaotong/h3_rewriter_fresh
REWRITER_SRC=/path/to/repo/.
# TASK_ROOT 先具备 models/Qwen3.5-9B、lf_dataset、media、grpo_20261005/env。
mkdir -p "$TASK_ROOT/trl_sft_official_v2_20261006" "$TASK_ROOT/grpo_20261005"
cp "$REWRITER_SRC/sft/"*.py "$TASK_ROOT/trl_sft_official_v2_20261006/"
cp "$REWRITER_SRC/system_prompt_v2.txt" "$TASK_ROOT/grpo_20261005/system_prompt.txt"
# 使用固定 Python env，在 /work 挂载下 prepare 与生成 template receipt。
docker run --rm --network none -v "$TASK_ROOT:/work" -v "$REWRITER_SRC:/code:ro" \
  h3-rewriter-grpo:20261005 /work/grpo_20261005/env/bin/python /work/trl_sft_official_v2_20261006/prepare.py
docker run --rm --network none -v "$TASK_ROOT:/work" -v "$REWRITER_SRC:/code:ro" \
  h3-rewriter-grpo:20261005 python /code/bootstrap_runtime.py sft --root /work
docker run -d --name xiaotong-trl-sft-fresh --gpus all --network none --shm-size=16g \
  -v "$TASK_ROOT:/work" -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=1 \
  -e PYTHONPATH=/work/trl_sft_official_v2_20261006/vendor:/work/trl_sft_official_v2_20261006 \
  h3-rewriter-grpo:20261005 /work/grpo_20261005/env/bin/python -u /work/trl_sft_official_v2_20261006/run_pipeline.py
```

镜像需 FFmpeg 共享库，绑定 env 需固定 Python 依赖；历史 vendor 只在未正确安装 TorchCodec 时需要。pipeline 在一次运行中执行 runtime validation、3-step smoke、9859 行 preflight 和 4 轮训练。恢复已有 checkpoint 时使用相同 8 卡/env：

```bash
/work/grpo_20261005/env/bin/python -m torch.distributed.run --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29616 \
  /work/trl_sft_official_v2_20261006/train.py --epochs 4 \
  --resume /work/trl_sft_official_v2_20261006/run/checkpoint-N
```

只恢复未完成的训练；现有四轮已完成，不需要为“验证完成”再次训练。

## 12. 本次 GRPO 运行状态（2026-10-06）

`pilot/status.json` 确认 64/64 step、epoch 0.5、train_runtime 6307.8448 秒；训练已完成，pipeline 进入 `evaluation_101`。101 独立评测尚在运行，不填写完整严重错误率或宣称改善。本机 gateway 健康且已成功返回评测评分。allowlist 修复前的失败配对须使用 §8 的独立重审工具恢复，保留初始结果；只有全部有效审查后才按 §9 更新 GRPO 对比表。

## 13. v4 修复与全本机重跑

新 JOB：`/data/xiaotong/h3_rewriter_sft_20261002/grpo_ep3_luna_v4_20261006`，容器 `/work/grpo_ep3_luna_v4_20261006`。旧 v3 JOB 及其原始评价保留，不覆盖、不从旧 GRPO adapter 续训。本次共享同一冻结 merged EP3 初始化；EP3 adapter SHA 与官方 template SHA 已核验，`rerun_receipt.json` 记录父实验及固定配置。

旧 1024 候选审计发现：42 个格式失败候选获得正优势；step36 有15次最终judge失败被赋操作性负分；30个音乐flag缺v2项，其中12个空问题清单得满分。旧完整101 recovered Luna结果为 matched merged severe 6→11、格式2→4；原6→9摘要只有93个有效配对。独立审查的10→13属于另一个判定版本，应分别报告。

v4 revision：`ep3-v2-luna-checklist-v4`，Luna high，CLI最多16并发；本机runtime在 `/home/xiaotong/grpo_ep3_luna_v4_20261006/local_luna_runtime`，端口8793。全程无需Mac或SSH。缓存与旧v3隔离。

1. 原文拆为缓存的完整要求清单；source sentences全覆盖，每个requirement恰好判定一次，结合全篇和结尾检查；关键要求的明确错误不得降级。
2. flags与音乐/台词问题清单严格一致；通过covered_requirement_id去重，不双扣。格式仍按raw13项检查，不修复输出后算通过。
3. judge最终失败返回None，固定TRL将其排除出组内均值/方差；完成mask同时置零以排除包括KL在内的梯度。不得写成语义错误。
4. 在TRL实际优势上增加正优势门禁：格式失败、severe、音乐/台词违规及critical待复核不得有正优势；既有负优势保留。门禁有逐候选token对齐断言；`advantages.rank*.jsonl`记录原始与实际优势。**这是显式改动的GRPO更新规则。**
5. 强制真实本机Luna合成校准通过，再运行2step smoke。smoke权重不用作pilot初始化；pilot重新从EP3开始64step，然后同协议greedy101评价。

已通过固定Torch/TRL环境14项回归测试和真实Luna14/14合成校准；校准只使用5个合成原文，不使用101的输出或独立审查标签。模板、256训练数据、媒体顺序、LR5e-6、beta.02、采样配置、LoRA和64step保持原设置。修复不能预先保证严重错误下降，须等待完整独立评价。

```bash
TASK_ROOT=/data/xiaotong/h3_rewriter_sft_20261002
REWRITER_SRC=/path/to/h3-rewriter
JOB_NAME=grpo_ep3_luna_v4_20261006
# 新JOB先具备pilot_inputs、system、冻结EP3初始化和MERGE_READY，见prepare_merge.py。
python3 "$REWRITER_SRC/bootstrap_runtime.py" grpo --root "$TASK_ROOT" --job "$JOB_NAME" --include-calibration-sources
# 本机runtime复制luna_base/luna_judge/evidence/reward_policy/run_config/
# start_luna_runtime/runtime_watchdog/calibrate_judge/format_rules及当前两份Luna receipts。
# 上述bootstrap显式追加5个合成原文hash；两份Luna receipts复制到runtime。
cd /home/xiaotong/grpo_ep3_luna_v4_20261006/local_luna_runtime
H3_LUNA_PORT=8793 python3 start_luna_runtime.py
python3 calibrate_judge.py --output "$TASK_ROOT/$JOB_NAME/calibration_receipt.json"
# 先确认GPU空闲，再启动；每次只允许一个本任务pipeline。
docker run -d --name xiaotong-grpo-ep3-luna-v4-20261006 --gpus all --network host --shm-size=16g \
  -v "$TASK_ROOT:/work" -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=1 \
  -e H3_GRPO_JOB="$JOB_NAME" -e H3_LUNA_PORT=8793 \
  -e PYTHONPATH=/work/$JOB_NAME:/work/trl_sft_official_v2_20261006/vendor:/work/trl_sft_official_v2_20261006 \
  h3-rewriter-grpo:20261005 /work/grpo_20261005/env/bin/python -u /work/$JOB_NAME/run_pipeline.py
```

监控新JOB的`pipeline_status.json`、`smoke.log`、`pilot/status.json`、`pilot.log`、`rollouts.rank*.jsonl`和`advantages.rank*.jsonl`，以及本机runtime的watchdog状态。完成101后只对judge失败重审，保留原始rewrites；重审工具需显式`--job grpo_ep3_luna_v4_20261006 --port 8793`。不得把v4训练reward与旧v3数值当作同一评价器的可比指标，也不得拿训练采样输出替代greedy101。


## 14. v5 状态转换 reward 修复（2026-10-06）

v4 在笑脸车窗 case 中漏判末句反转：雾蒸发后却仍遮住玻璃。v5 (`ep3-v2-luna-state-v5`, 本机端口 8794，默认新 JOB `grpo_ep3_luna_v5_20261006`) 增加独立 Luna 状态审查。该调用不读取第一轮结论，按每个原文句子对照初始状态、变化、最终状态及完整输出的最终状态。缺失/矛盾为严重错误；不确定禁止正优势；审查缺失或不完整按不可评分处理。重复的同句严重错误不重复扣分，两个审查的完整证据均保留。旧 v4 运行及缓存不改写。

该修复增加一次模型调用，不能保证模型评审零误判。已知笑脸 benchmark 用于针对性回归，属于开发样本，不能作为独立泛化证据。针对性验证：实际 step32 错误输出在 v5 得 -0.6；正确消散与原文明确要求重新起雾的对照均得 1.0。20 项固定环境回归测试通过；17/17 条真实 Luna 合成校准通过。完整验证文件在 HB10 `/home/xiaotong/grpo_ep3_luna_v5_20261006/reward_fix_validation.json`。

后续新训练先用 `bootstrap_runtime.py grpo --job grpo_ep3_luna_v5_20261006 --include-calibration-sources` 生成新 JOB 授权及校准来源，复制当前 `grpo/*.py`，启动 `start_luna_runtime.py`。然后运行 `calibrate_judge.py --port 8794 --output <JOB>/calibration_receipt.json`；17 条合成校准必须全部通过才能进入新训练。`run_pipeline.py` 随后预提取 256 条要求。当前未启动 v5 模型训练；v4 step32/64 的原评测保持原口径。


## 15. FAL、当前 EP3 与 GRPO 的统一重审

五组使用同一批 101 ID/原文：FAL 原始输出、当前 TRL EP3 checkpoint-906 未合并输出、matched merged EP3 初始化、v4 GRPO checkpoint-32、v4 GRPO final（64step）。已核对原文完全一致。不重生成，不清洗格式，不把历史审查快照直接拼接。使用 GPT-6 Astra high，冻结 `ep3-v2-astra-state-unified-v1`（v5 checklist + 独立状态审查），评审不接收模型标签；先通过17条合成校准。每条保存原始输出、SHA、要求、两个审查与证据。

入口 `review/unified_compare.py`。语义主表只统计五组共同有效配对；某一组 judge 失败，该 ID 在五组语义统计中均排除，报告未知数，不能冒充全101。raw格式使用同一检查器（支持 Image 1 和 <Picture 1>），分母为各组已审输出。语义保留、严重程度、状态结尾、配乐/台词单列，不能相加为总错误率。该比较统一评分口径，不宣称各组历史生成过程相同；图像忠实度及生成视频效果未验证。笑脸 case 已用于评分开发，不能再称其独立 held-out。

实际本机工作目录：`/home/xiaotong/unified_fal_ep3_grpo_101_20261006`，独立 Astra gateway `127.0.0.1:8796`。结果文件 `COMPARISON.md`、`summary.json`、`audits.jsonl`；`COMPLETE` 仅在五组101条全部有效后写入。结果当前执行中，不能填写未完成的全量结论。


## 16. 代码同步与 checkpoint 发布（2026-10-07）

当前独立代码仓库为 `git@github.com:xiaotong-bao/h3-rewriter.git`（HB10：`/home/xiaotong/h3-rewriter`）。当前代码默认 JOB 是 `grpo_ep3_luna_2k_20261006`；reward profile 使用 Luna / `h3-reward-luna-v1` / 8797，evaluation profile 使用 Astra / `h3-eval-astra-v1` / 8798。两者共享 `grpo/judging_standard.py` 的严重程度准则，并对初次判为 severe 的问题独立复核；源要求审查与状态审查并发。历史 v2/v4/v5 的训练配置、缓存与统计口径保留。固定训练环境中 23 项回归测试通过；这不等于新 profile 已完成真实评审校准或新训练。

本次明确选择四轮 TRL SFT（302/604/906/1208），以及 EP3 初始化的 v2、v4 GRPO 的 checkpoint-16/32/48/64 和 final_adapter，共 14 个目录。完整 checkpoint 包含 adapter、optimizer、scheduler、trainer_state 和八个 rank RNG；final_adapter 用于推理，不声明可恢复优化器训练。GRPO adapter 需要其对应的 merged EP3 初始化；本清单不包含 merged base，应按对应实验 prepare_merge 和模板 receipt 重建，不能直接把 GRPO adapter 加载到未经 EP3 合并的原始 Qwen。

已上传并核验（2026-10-07 01:53:19 UTC / 北京时间 09:53:19）：14 个目录、264 个文件、19,332,433,153 bytes（18.005 GiB）。每个文件均核验远端大小及 SHA256 metadata，上传回执已读回核对，`uploaded=true`、`ready=true`。

```text
s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/
  trl_sft_official_v2_20261006/sft/run/checkpoint-{302,604,906,1208}/
  grpo_ep3_luna_v2_20261006/pilot/checkpoint-{16,32,48,64}/
  grpo_ep3_luna_v2_20261006/pilot/final_adapter/
  grpo_ep3_luna_v4_20261006/pilot/checkpoint-{16,32,48,64}/
  grpo_ep3_luna_v4_20261006/pilot/final_adapter/
  checkpoint_publications/h3_checkpoint_publication_20261007.json
```

当前 EP3 本地路径为 `/data/xiaotong/h3_rewriter_sft_20261002/trl_sft_official_v2_20261006/run/checkpoint-906/`；历史 v2 GRPO 最终可恢复 checkpoint 为 `/data/xiaotong/h3_rewriter_sft_20261002/grpo_ep3_luna_v2_20261006/pilot/checkpoint-64/`，v4 使用对应 v4 目录。

本次使用 checkpoint 专用工具，避免再次扫描或上传原始媒体、评审缓存及认证文件：

```bash
cd /home/xiaotong/h3-rewriter
python3 publish_checkpoints.py \
  --root /tmp/h3_checkpoint_publish_20261007 \
  --folders /tmp/h3_checkpoint_publish_20261007/folders.json \
  --receipt /home/xiaotong/h3_checkpoint_publication_20261007.json \
  --profile r2w --upload
```

默认不上传。上传时逐文件核对远端大小及 SHA256 metadata，已存在且相同的对象复用，内容不同则拒绝覆盖。只有全部核验后才发布本次 receipt，并读回核对；以其 `uploaded=true`、`ready=true` 为准。本次独立 receipt 不替代 §10 覆盖数据与媒体的旧 READY.json。


本次成功上传回执：

```text
s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/checkpoint_publications/h3_checkpoint_publication_20261007.json
```

常用权重地址（完整目录；EP3 / 历史 v2 GRPO / v4 GRPO 分别列出）：

```text
s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/trl_sft_official_v2_20261006/sft/run/checkpoint-906/
s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/grpo_ep3_luna_v2_20261006/pilot/checkpoint-64/
s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/grpo_ep3_luna_v4_20261006/pilot/checkpoint-64/
```

下载示例（恢复到新目录，使用 R2 endpoint）：

```bash
aws --profile r2w \
  --endpoint-url https://f25b0ac4c45a2442f62961145a64d158.r2.cloudflarestorage.com \
  s3 cp --recursive \
  s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/trl_sft_official_v2_20261006/sft/run/checkpoint-906/ \
  /data/xiaotong/h3_rewriter_restore/ep3/checkpoint-906/
```

下载后应按本次回执逐文件核对大小和 SHA256。本次未上传 merged base 或新 profile 训练产物，也不改变历史 GRPO 质量结论。
