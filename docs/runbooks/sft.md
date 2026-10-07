# 当前 SFT：Luna 内容筛选后普通 LLaMA-Factory 训练

唯一 SFT 操作入口；推理权重和评价规则见 [inference + eval](inference_eval.md)，强化学习见 [GRPO](grpo.md)。更新：2026-10-07。

## 当前任务和数据

训练根目录：`/mnt/nfs/xiaotong/h3-rewriter/runs/content_clean_9bv2_sft_20261007/`。
容器：`xiaotong-h3-clean-9bv2-sft-20261007`；8×H100，普通 SFT，不使用 MTP。从 Qwen3.5-9B base 新训练，沿用原 9B v2 的 LLaMA-Factory 方法，不加载旧 step604 adapter。

原新版用户数据 9659 条，Gemini Flash 改写用户格式；teacher 答案、媒体顺序和原划分保留。Luna high 按固定内容标准筛选：只取 maximum_severity=none 且 issues=[]，**忽略格式分**，不因 subject_definitions / detailed_description 等 teacher 结构淘汰样本。冻结快照来自9608条有效评审，51条调用失败未纳入；5682条训练，原200来源的新版验证集，验证集未经此筛选。不把本轮称为20k混合训练。

冻结数据：`lf_dataset/train.jsonl`、`lf_dataset/val.jsonl`；内容筛选导出 SHA256：`7cc1f25ea7b0935f0a7161db80fdb7e6b808e2375357d5ff64551304bc3906f7`。
Luna 原评审目录：`/mnt/nfs/xiaotong/h3-rewriter/runs/userlike_luna_filter_20261007/`。后续重试结果不自动改变活跃训练快照。

源数据 S3：

```text
s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/userlike_gemini_20261007/
```

此前19318条原版+新版实验已按用户要求停止，数据和checkpoint保留；它不是当前任务。

## 固定训练配置

LLaMA-Factory 来源：Mellis-Labs/pika-llama-factory，h3-rewriter 分支，版本 `ec25d69537eb93a0f52e8f2bbc002b451898b6a8`。
实际完整配置：当前训练根目录下 `lf_configs/sft.yaml`。JSON 是合法 YAML；此文件为参数来源。

| 参数 | 当前值 |
|---|---|
| Framework / stage | LLaMA-Factory / ordinary SFT LoRA |
| Epoch / optimizer steps | 6 / 1068，每轮178步 |
| Batch | 每GPU 1 × 累积4 × 8GPU = 32 |
| LR / scheduler / warmup | 1e-4 / cosine / 18 steps |
| LoRA | rank64 / alpha128 / dropout0.05；语言及线性注意力投影 |
| Vision / projector | 冻结 |
| Template / max length | qwen3_5_nothink / 8192 |
| Loss / packing | 仅答案 / 不packing |
| Precision / attention | BF16 / SDPA |
| Gradient checkpointing | 开启，非reentrant |
| Seed / data_seed | 42 / 42 |
| Image pixels min/max | 3136 / 200704 |
| Video pixels min/max | 3136 / 50176；fps1，最多8帧 |
| Save / validation | 每25步及每epoch保存，全部保留；每100步val |

## 启动、监控与恢复

当前任务已经运行，**不要再次启动 pipeline**。pipeline 会拒绝已有正式输出目录，防止覆盖。运行快照：`pipeline.py`、`derive_validated_cache.py`、`train_with_epoch_review.py`。

仓库对应实现：`sft/lf_clean_pipeline.py`、`sft/derive_clean_lf_cache.py`、`sft/lf_train_with_epoch_review.py`。依赖原生 LLaMA-Factory，不改训练器、collator、LoRA 或 loss；新增callback只记录进度、epoch保存和101推理。

预检按ID精确选择已完成原生token和全量视觉审计的父数据子集，逐条检查源行完全相同、答案mask/解码、媒体、长度、划分不泄漏；继承未变视觉样本的完整审计证据。报告 `logs/lf_preflight_report.json`，train5682/val200，答案不匹配均0；八卡4-step smoke已通过。

```bash
docker logs --tail 20 xiaotong-h3-clean-9bv2-sft-20261007
cat /mnt/nfs/xiaotong/h3-rewriter/runs/content_clean_9bv2_sft_20261007/trl_sft_official_v2_20261006/progress.json
tail -n 20 /mnt/nfs/xiaotong/h3-rewriter/runs/content_clean_9bv2_sft_20261007/logs/training.log
```

`status.json` / `trl_sft_official_v2_20261006/pipeline_status.json` 为phase；progress含step、epoch、loss。路径里的 TRL 名称仅为历史复用，当前框架是 LLaMA-Factory。

容器镜像 `h3-rewriter-sft-retrain:20261006`，network none、shm16g；依赖base、媒体、LF源码和固定env挂载。训练Python `/work/lf_env/bin/python`，torchrun静态 rendezvous：8进程、master_addr127.0.0.1、port29624。不要改活跃env，也不要使用 network-none 下的 standalone hostname rendezvous。

恢复时先确认任务停止和最后完整checkpoint；使用原配置加 `resume_from_checkpoint=/work/runs/content_clean_9bv2/checkpoint-N`，以同一八卡torchrun运行 `/work/train_with_epoch_review.py`。不要重跑新训练pipeline，不重建数据，不改变世界大小。未确认停止时不启动第二训练进程。

## 每轮 checkpoint、推理与评价

正式checkpoint目录：`runs/content_clean_9bv2/checkpoint-N/`；每轮步数：178、356、534、712、890、1068。保存adapter、processor/tokenizer、optimizer、scheduler、trainer state和八rank RNG。

每epoch在训练中分片生成同一原101条greedy输出，保存到 `trl_sft_official_v2_20261006/epoch_benchmarks/stepN/`。推理完成后训练继续；Astra评分在CPU/网络独立进行。

评价service：`h3-clean-9bv2-epoch-review.service`；入口：

```bash
python3 review/epoch_reviews.py   --run-root /mnt/nfs/xiaotong/h3-rewriter/runs/content_clean_9bv2_sft_20261007   --epochs 6 --workers 8
```

活跃service已有文件锁，不并行再开watcher。输出 `epoch_reviews/epN/` 的manifest、audits、summary、COMPARISON、COMPLETE；总表 `epoch_reviews/EPOCH_COMPARISON.md`。单轮仅101有效、无失败才算完成。详细规则和Top5见 [inference + eval](inference_eval.md)。

Slack沿用用户授权每20分钟进度：`h3-clean-9bv2-slack-progress.timer`；六轮训练和评测全完成后停止。

## 已发布 EP2

EP2为checkpoint-356，完整可恢复权重已上传并写READY：

```text
s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/content_clean_9bv2_sft_20261007/ep2/checkpoint-356/
```

上级含sft.yaml和evaluation。加载、base和下载命令只在 [inference + eval](inference_eval.md) 维护；不再新增按日期/epoch命名的runbook。
