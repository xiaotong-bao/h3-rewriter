# Inference + evaluation

唯一推理与评价操作入口；训练配置见 [SFT](sft.md)，强化学习见 [GRPO](grpo.md)。更新：2026-10-07。

## 当前筛选 SFT EP2：权重和加载

本轮为 5682 条内容干净新版数据、普通 LLaMA-Factory SFT；EP2 = checkpoint-356。不是旧 9B v2 checkpoint-604，也不是历史 TRL EP2。

本地目录：

```text
/mnt/nfs/xiaotong/h3-rewriter/runs/content_clean_9bv2_sft_20261007/runs/content_clean_9bv2/checkpoint-356/
```

S3：

```text
s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/content_clean_9bv2_sft_20261007/ep2/checkpoint-356/
```

上级 `READY.json` 已写入，所有 checkpoint、配置和评测对象已核对远端大小与 SHA256 metadata；这不等于逐文件远端下载重新计算 SHA。`evaluation/` 包含 manifest、audits、summary、COMPARISON。

这是 LoRA adapter。先加载 Qwen3.5-9B base，再用 PEFT 加载下载目录；不要把 adapter 当完整模型。adapter_config 里的 `/work/models/Qwen3.5-9B` 是训练容器路径，加载时显式指定实际本地 base。base 备份：

```text
s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_retention_v2_20261006/assets/base.tar
```

模板 `qwen3_5_nothink`；使用未修改的官方 processor/template，enable_thinking=False。101 对照推理为 greedy、max_new_tokens=2048、repetition_penalty=1，保留原始媒体与文本顺序、retention v2 system、EOS 和截断信息。每个 ID 一次，不重抽挑选更好的输出。sampling 另建输出目录并记录 seed、temperature=0.8、top_p=0.95，不与 greedy 混算。

下载（使用本机已配置 r2w，凭据不入 Git）：

```bash
aws --profile r2w --endpoint-url https://f25b0ac4c45a2442f62961145a64d158.r2.cloudflarestorage.com   s3 cp s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/content_clean_9bv2_sft_20261007/ep2/checkpoint-356/   /path/to/ep2/checkpoint-356/ --recursive
```

## 固定 101 评价标准

所有组按 ID 核对原文、任务、图片身份、时长和比例。每个模型都以原始要求为依据，不拿 FAL 当标准答案。输入缺失或条件不一致不得冒充完整 101 比较。

评审模型固定 gpt-6-astra，reasoning_effort=low。调用仅提供原文、任务条件和原始改写，不包含模型名称或实验标签。一轮覆盖全部明确要求，在该轮检查状态变化和结局；没有独立状态审查。等级只有 none/general/severe，不设 review/uncertain。没有明确违反原文的证据不扣分，调用失败单独记录并重试，不伪装成无错误或低分。

| 等级 | 判定 | 内容分 |
|---|---|---:|
| none | 所有明确关键要求保留，未发现确证错误 | 100 |
| general | 核心事件、主体关系和结局正确，存在局部范围、强度、时间、镜头、视觉或声音细节偏差 | 70 |
| severe | 关键实体/身体部位遗漏、主动作遗漏或改变、主体受事/说话者绑定错误、关键数量改变、因果/方向/状态变化/结局错误、违反明确禁止要求 | 0 |

每条取最高等级，不按错误条数重复扣内容分。镜头偏差只有使关键事件不可见或改变核心事件时才升为 severe；不能把所有明确细节都当作严重问题。兼容原文的合理补充、光线、服装、音效及新增配乐允许。配乐仅在违反原文明确音乐/静音要求时计错。

30 条试评复核后的边界：原文未指定 wipe 的动作主体，不自行限定必须由某物遮挡镜头；朝镜头伸手递出已取出的物体可以等价保留交给摄影者，不强求接收动作特写。主体事件保持时，未指定确切数量的泛称复数变成单数至多为一般问题；确切关键人数/数量或多人互动被改变才属于严重问题。合理写实描写可等价保留 hyper-realistic，不要求字面复制。树苗长成巨大树木但未说明高过森林属于一般尺度问题，未长大/变小/成长事件缺失才严重；合理表述明确支配周围森林景观可保留尺度要求。未指定从地面起飞或穿越边界时，向太空深处高速前进可保留 into space，不额外发明地面起点。这些规则适用于所有组，不按模型或特定 ID 改判。

格式分独立：integrated_multimodal_description、overall_soundscape、non_diegetic_music 三字段各出现一次、顺序正确且非空，满足得 100，否则 0。旧 13 项检查可作为辅助诊断，但不混入本次三字段格式分。上下文时长/比例不要求逐字复述，明确冲突才按内容错误处理。

30 条固定校准样本：15 t2va + 15 i2va，随机种子 20261006；ID 清单单独保存。先评审所有可用组，逐项复核严重错误与一般错误边界、遗漏与等价表达、配乐允许规则。若修改 codebook，增加版本号并重跑全部 30 条；正式版本确定后，全部 101 使用同一版。试评属于本批开发校准，不宣称独立泛化验证。

最终只提供分数比较，不做逐条胜平负。表格列：模型/解码、有效条数、平均内容分、平均格式分、严重问题 case 数、一般问题 case 数、无问题 case 数。后三列互斥，相加为有效条数；多错误 issue 数不能混当 case 数。每组另给 Top 5 具体问题：优先 severe、再 general，同级按 ID 和问题顺序确定，附样本 ID、原文证据、输出证据/遗漏说明和原因。问题不足 5 个时报告实际数量，不编造。每个模型/解码组分别给 Top 5。

规范实现：[single_pass_compare.py](../../review/single_pass_compare.py)。保存原始输出、原文、SHA、完整逐项 coverage、证据、问题级别、judge 模型/effort、规则版本、失败记录、汇总分及 30 条复核记录。纯文本评审不验证未展示图片的视觉忠实度。

## 2. 日常 reward：GPT-6 Luna

评审模型固定 gpt-6-luna，reasoning_effort=high。沿用同一单轮内容 codebook：取消独立状态审查、取消待复核、新增配乐允许。要求完整 coverage 和每个问题的原文/输出证据；重复的同一错误只计一次。按问题数连续扣分，用于日常 reward，不拿 Luna reward 替代 Astra 101 模型评测分。

令 S 为 severe 问题数、G 为 general 问题数、F 为三字段格式失败（0/1）：

`reward = max(-1, 1 - 0.8 * min(S, 2) - 0.1 * min(G, 3) - 0.3 * F)`

无错误且格式通过为 1；一个一般错误为 0.9；一个严重错误为 0.2；两个严重错误为 -0.6；仅格式失败为 0.7。严重错误或格式失败的样本不能获得正优势（训练接入时门禁），即使原始 reward 为正。judge 失败为不可评分，重试后仍失败记录 judge_failed，训练接入时梯度贡献为 0，不能把服务失败算作内容问题。

当前 userlike GRPO 的在线 reward 接入见 [GRPO](grpo.md)；历史 v4/v5 使用旧协议，不能混用其缓存或静默替换活跃服务。

固定规则后修改需新版本、新缓存和重新评测，禁止跨版本合并结果。


## 对已有输出运行评测

manifest 每行需包含 id、label、original、rewrite、task、duration_s、aspect；checkpoint、SHA、截断信息一并保留。label 只用于汇总，不向 judge 暴露模型身份。必须逐 ID 核对输入，不能拼凑不一致的 101 条。

```bash
cd /mnt/nfs/xiaotong/h3-rewriter
python3 review/single_pass_compare.py   --manifest runs/content_clean_9bv2_sft_20261007/epoch_reviews/ep2/manifest.jsonl   --output /path/to/new_ep2_review   --model gpt-6-astra --effort low --workers 8
```

输出 `summary.json`、`audits.jsonl`、`COMPARISON.md`；仅 valid=expected=101 且 failures=[] 才算完成。独立重跑可能出现模型判级波动；保留旧记录，不覆盖分数。Top 5 在 COMPARISON 中，完整证据在 audits 中。

当前 SFT 的每轮推理由训练 callback 自动完成；每轮 checkpoint、推理文件和 watcher 见 [SFT](sft.md)。不要在活跃训练上额外启动 GPU 推理，除非明确需要重新生成。

## 当前筛选 SFT 评测快照

| 模型 | 内容分 | 格式分 | 严重样本 | 一般样本 | 无问题 |
|---|---:|---:|---:|---:|---:|
| FAL 固定参考 | 87.13 | 100 | 7 | 20 | 74 |
| 原 9B v2 固定参考 | 87.43 | 100 | 7 | 19 | 75 |
| 筛选 SFT EP1 | 79.11 | 98.02 | 16 | 17 | 68 |
| 筛选 SFT EP2 | 86.73 | 100 | 8 | 18 | 75 |
| 筛选 SFT EP3 | 83.56 | 100 | 10 | 22 | 69 |

这是固定参考评审快照，不与另一轮 9B v2 重评的 86.83 混用。最新各轮结果读取 `runs/content_clean_9bv2_sft_20261007/epoch_reviews/EPOCH_COMPARISON.md`。自动评审存在已知边界波动，尚未人工统一改判。

## GRPO checkpoint 推理和评价

### 已发布 userlike GRPO step320

本机权重：`/data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/pilot/checkpoint-320/`。

完整 checkpoint S3 地址：

```text
s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/backups/step-320/checkpoint-320/
```

以下命令在 HB10 主机运行；复用现有 base、镜像、benchmark 输入和本机 Codex 登录。checkpoint 是 LoRA，不能作为独立完整9B模型加载；必须搭配 Qwen3.5-9B base。远程机器还需要准备这几个依赖及 benchmark 图片。

下载到新的目录（本机原 checkpoint 已在，不必重复下载）：

```bash
aws --profile r2w   --endpoint-url https://f25b0ac4c45a2442f62961145a64d158.r2.cloudflarestorage.com   s3 cp   s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/backups/step-320/checkpoint-320/   /data/xiaotong/h3_rewriter_sft_20261002/restored_step320/   --recursive
```

复现 inference（8卡并行；使用新输出目录，避免混入旧结果）：

```bash
cd /home/xiaotong/h3-rewriter
docker run --rm --gpus all --network none --shm-size=8g   -v /data/xiaotong/h3_rewriter_sft_20261002:/work   -v /home/xiaotong/h3-rewriter:/code:ro   -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=1   -e PYTHONPATH=/work/trl_sft_official_v2_20261006/vendor:/work/grpo_20261005/env/lib/python3.11/site-packages   h3-rewriter-grpo:20261005   /work/grpo_20261005/env/bin/python -u -m torch.distributed.run   --nproc_per_node=8 --master_port=29862   /code/grpo/ablate_9bv2_inputs.py   --checkpoint /work/grpo_9bv2_userlike_1k_20261007/pilot/checkpoint-320   --output /work/grpo_9bv2_userlike_1k_20261007/manual_step320/generation   --templates HF --merges unmerged --autocast --label-suffix GRPO_step320
```

Astra evaluation（主机运行，需要网络和 `codex` 登录）：

```bash
cd /home/xiaotong/h3-rewriter
python3 grpo/prepare_eval_manifest.py   --results /data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/manual_step320/generation/results.jsonl   --output /data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/manual_step320/manifest.jsonl
python3 review/single_pass_compare.py   --manifest /data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/manual_step320/manifest.jsonl   --output /data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/manual_step320   --model gpt-6-astra --effort low --workers 8
```

输出 `summary.json`（分数和失败数）、`audits.jsonl`（101条证据）、`COMPARISON.md`。只有 valid=expected=101 且 failures=[] 才是完整有效评测。生成固定 greedy；Astra 重新调用可能有评分波动。

直接复用已有 step320 输出评测时，把上述 results 路径换成 `astra_eval_every32/step-0320/generation/results.jsonl`，无需重跑 inference。
