# H3 prompt rewriter

Qwen3.5-9B 的 SFT、GRPO、推理和独立评测。操作文档只保留三本，后续更新写入对应文件。

| Runbook | 内容 |
|---|---|
| [1. Inference + eval](docs/runbooks/inference_eval.md) | checkpoint / S3 / base加载、101推理、Astra low固定评价、Luna reward、分数与证据 |
| [2. SFT](docs/runbooks/sft.md) | 当前5682条Luna内容干净新版数据、普通LLaMA-Factory六轮训练、配置、监控、恢复、每轮评测 |
| [3. GRPO](docs/runbooks/grpo.md) | 9B v2 userlike GRPO初始化、数据、参数、Luna reward、checkpoint和评测流水线 |

当前SFT运行目录：`/mnt/nfs/xiaotong/h3-rewriter/runs/content_clean_9bv2_sft_20261007/`。此前19318条混合训练已停止。EP2 checkpoint-356已发布，路径见第一本。

代码目录：`data/` 数据处理、`sft/` 训练与epoch推理、`grpo/` 强化学习、`review/` 固定评测与过滤。目录README仅介绍代码，不作为额外操作runbook。

旧文档保留为 `docs/archive/20261007/` 的历史快照，不代表当前状态或协议；当前操作以以上三本为准。模型、媒体、完整原文/输出、缓存和凭据保留在外部，不入Git。
