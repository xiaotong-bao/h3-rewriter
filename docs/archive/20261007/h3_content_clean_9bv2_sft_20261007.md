> 历史快照：仅供追溯，不是当前操作入口。当前仅使用 [inference + eval](../../runbooks/inference_eval.md)、[SFT](../../runbooks/sft.md)、[GRPO](../../runbooks/grpo.md)。旧状态及评分协议不代表当前状态。

# 内容无问题新版数据：普通 9B v2 方法 SFT

用户明确停止19318条混合数据实验，改用5682条 Luna判定内容无问题的新版输入。忽略格式分：subject_definitions / detailed_description 等原 teacher 结构保留。选择条件为 maximum_severity=none 且 issues=[]；冻结快照有9608条已评审、51条失败未纳入。原数据、答案、媒体和训练/验证划分不改；验证集为原200个来源的新版输入，未经此次训练筛选。

用户随后明确普通SFT，不使用MTP。从Qwen3.5-9B基础权重开始；LLaMA-Factory h3-rewriter 分支原9B方法，LR1e-4、单卡BS1、累积4、八卡、有效BS32，LoRA r64/alpha128/dropout0.05，冻结视觉，BF16、SDPA、qwen3_5_nothink，8192上限，无packing，答案loss，cosine/warmup18，seed42。用户指定6EP，预计1068steps，以实际trainer为准。

新实验目录：`/mnt/nfs/xiaotong/h3-rewriter/runs/content_clean_9bv2_sft_20261007/`，容器 `xiaotong-h3-clean-9bv2-sft-20261007`；正式权重 `runs/content_clean_9bv2/`，每25步和每epoch完整保存。每epoch用同一原101测试输入做greedy推理，Astra low固定标准评审，结果位于epoch_reviews。

预检精确核对新数据是先前完整token/全部视觉审计通过数据的未改动子集，按原ID选择native tokenized cache，并逐条重新检查答案mask/解码、长度、媒体路径、划分不泄漏。不重新解码未变化的全部视觉素材，保留父审计SHA和精确子集关系证据。八卡4-step smoke通过后自动进入正式六轮训练。

当前phase/step：`trl_sft_official_v2_20261006/pipeline_status.json`、progress.json（路径名保留仅为复用评审）。训练日志logs/training.log。Slack沿用用户授权每20分钟实际进度。


## EP2 checkpoint 发布

本次内容干净新版 5682 条普通 LLaMA-Factory SFT 的 EP2，checkpoint-356。区别于历史 9B v2 step604 和旧 TRL EP2。

S3 发布目录：

```
s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/content_clean_9bv2_sft_20261007/ep2/
```

- `checkpoint-356/`：LoRA adapter、tokenizer、processor、模板，以及 optimizer、scheduler、8 rank RNG、trainer state，可用于恢复训练。
- `sft.yaml`：实际训练配置。
- `evaluation/`：101 条原始输入及输出 manifest、逐条审计、summary 和 COMPARISON。
- `READY.json`：完成上传后写入的文件清单，记录每个文件的大小及 SHA256；核验方式为远端大小和 SHA256 metadata，非远端下载重新哈希。

推理需先加载 Qwen3.5-9B base，再通过 PEFT 加载下载的 checkpoint-356 adapter；此目录不是独立全参数模型。adapter_config 中 `/work/models/Qwen3.5-9B` 是训练容器路径，加载时用实际本地 base 路径。基础权重沿用已发布的 `s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_retention_v2_20261006/assets/base.tar`。模板为 qwen3_5_nothink。

Astra low 固定标准原始评测：101 条，内容 86.73，格式 100；严重样本 8、一般样本 18、无问题 75。按每个样本最高严重性归类；未进行人工重新分级。
