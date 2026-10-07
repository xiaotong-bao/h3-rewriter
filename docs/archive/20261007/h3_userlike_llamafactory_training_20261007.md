> 历史快照：仅供追溯，不是当前操作入口。当前仅使用 [inference + eval](../../runbooks/inference_eval.md)、[SFT](../../runbooks/sft.md)、[GRPO](../../runbooks/grpo.md)。旧状态及评分协议不代表当前状态。

# 新版用户格式数据：LLaMA-Factory 9B 训练

用户指定沿用 Mellis-Labs/pika-llama-factory 的 h3-rewriter 分支 9B 训练配置。当前代码版本 ec25d69537eb93a0f52e8f2bbc002b451898b6a8，包含原版视频修复。基座为备份恢复的 Qwen3.5-9B；新模型从基础模型开始，不加载旧 adapter。硬件 8×H100，旧版为 H200，不声明数值复现。

## 数据

S3：`s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/userlike_gemini_20261007/`。采用 `paired_2to1/train.jsonl`（19318 条）和 val（400 条），这里 two-to-one 表示两个输入对应同一个答案；原版 : userlike = 1:1。各对原版/新版严格保留 teacher 答案和媒体列表顺序，仍按原 9659/200 个来源 ID 划分。已核对下载文件 SHA 与远端 manifest，并逐条与旧实际训练数据比较答案和媒体。原数据、旧 checkpoint 不改动。

## 配置

单卡 BS1、梯度累积4、8卡、有效 BS32；LR1e-4、6 epochs（用户追加要求；旧配置为2 epochs）、cosine、warmup18；LoRA r64/alpha128/dropout0.05，冻结视觉和 projector。模板 qwen3_5_nothink；cutoff8192、无 packing、仅答案 loss；BF16、SDPA、gradient checkpointing；seed/data_seed42。完整实际配置在实验的 lf_configs/sft.yaml。新训练预计约3624 steps，以 Trainer 实际进度为准。

保留原每25步 checkpoint、每100步验证；另增加 epoch 保存和同一批原101条 greedy 推理/Astra low标准评审，并保留全部 checkpoint 以免 epoch 权重被自动删除。训练器、LoRA、collator、预处理均使用 LLaMA-Factory 官方实现，新增 callback 仅记录进度与做 epoch 推理。

## 运行

实验目录：`/mnt/nfs/xiaotong/h3-rewriter/runs/userlike_llamafactory_20261007/`。容器 `xiaotong-h3-userlike-lf-20261007`，network none。流水线先原生全量 token/答案 mask 预检、全部视觉 token-grid 审计、36条跨格式多模态八卡4-step smoke，再正式训练。预检失败不启动正式训练。

日志：`logs/preflight.log`、`logs/visual_audit.log`、`logs/smoke.log`、`logs/training.log`；当前 phase 和 progress 在 `trl_sft_official_v2_20261006/pipeline_status.json`、progress.json，保留旧路径名仅为复用原101评测代码。最终权重在 `runs/llamafactory_userlike/`。评审在 `epoch_reviews/`，由 h3-userlike-lf-epoch-review.service 自动运行。
