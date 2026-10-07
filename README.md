# H3 prompt rewriter

Qwen3.5-9B 的数据准备、官方模板 TRL SFT、Luna reward GRPO 与独立评测代码。
Luna 直接运行在 Linux 本机已登录的 Codex CLI 上，不依赖 Mac。

| 目录 / 文件 | 用途 |
|---|---|
| [data/](data/README.md) | 固定数据 split 和 ShareGPT 导出；teacher answer 保持原样 |
| [sft/](sft/README.md) | 官方 Qwen 模板、视觉预检、TRL SFT、各轮 101 条评测 |
| [grpo/](grpo/README.md) | EP3 合并初始化、Luna reward / Astra 独立评测、checklist + 状态与严重程度复核、正优势门禁、本机 supervisor、训练与恢复 |
| [review/](review/README.md) | raw 格式检查与失败 judge 重审 |
| [docs/runbooks/](docs/runbooks/h3_prompt_rewriter_evaluation_training_runbook.md) | 环境、启动、恢复、数据/S3 路径和统计口径 |
| `bootstrap_runtime.py` | 为指定任务生成本地模板/source-hash/授权 receipts |
| `publish_assets.py` | 数据与四轮 SFT checkpoint 的显式清单和 S3 发布工具；默认 dry-run |
| `requirements.txt` | 固定训练 Python 依赖 |
| `system_prompt_v2.txt` | SFT/GRPO 使用的完整 v2 system 原始字节 |

## 实际状态

- “10k”数据实际为 9899 条标签；排除 40 条后，固定 9659 train / 200 val。
- TRL SFT 已完成 4 epoch / 1208 step；EP3 checkpoint-906 被选作 GRPO 初始化。
- 旧 v3 GRPO pilot 已完成 64 step，未证明改善；v4 已修复 reward 校验、失败梯度与格式正优势问题，使用新 JOB 在 HB10 本机重跑。
- 训练数据此前已在私有 R2/S3：两份 chat JSONL 的 SHA 与本地一致，14116 个训练媒体引用均存在且大小一致。新增 TRL checkpoint 发布状态以远端 READY.json 为准。

## 运行顺序

1. 从已验证的数据快照恢复 `lf_dataset/` 和 `media/`；重新制作 split 时使用 [data 说明](data/README.md)。
2. 按 [统一 runbook](docs/runbooks/h3_prompt_rewriter_evaluation_training_runbook.md) 准备模型、固定 env 和 FFmpeg 镜像，将实验根挂载为 `/work`。
3. 复制 `sft/*.py` 到 `/work/trl_sft_official_v2_20261006/`，执行 prepare、bootstrap、SFT pipeline。
4. 复制 `grpo/*.py` 到 GRPO JOB，执行 merge 和 bootstrap；在 Linux 本机启动 Luna supervisor，按 profile 验证 reward 8797 / evaluation 8798，再启动 GRPO 容器。
5. 按同一 101 输入做 greedy 比较，报告语义严重程度、格式和 judge 失败；训练 reward 不能替代独立评价。

脚本保留已验证运行的 `/work` 实验布局，目录重组没有改变训练数据或超参数。
`REWRITER_SRC=/path/to/h3-rewriter`，不再需要原来的 monorepo 包路径。
数据、模型权重、媒体、完整原文/输出、缓存、凭据及运行 receipts 均保留在外部，不提交到 Git。

修复回归测试：在固定训练环境运行 `python -m unittest discover -s tests -v`。
真实校准：本机 gateway 启动后执行 `python grpo/calibrate_judge.py --output <JOB>/calibration_receipt.json`；启动前使用 bootstrap 的 `--include-calibration-sources` 把5个合成原文hash加入该任务allowlist。生产训练/101 allowlist仍由bootstrap生成，不能放宽到任意原文。

当前代码使用 `H3_JUDGE_PROFILE=reward`（Luna，8797）或 `evaluation`（Astra，8798），共享严重程度准则。历史实验配置及 checkpoint 不随默认配置变化。checkpoint 专用上传工具为 `publish_checkpoints.py`，默认只生成清单，`--upload` 才上传；已有不同内容的对象拒绝覆盖。
