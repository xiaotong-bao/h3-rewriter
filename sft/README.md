# SFT 代码

当前训练操作只看 [SFT runbook](../docs/runbooks/sft.md)，推理和评分看 [inference + eval](../docs/runbooks/inference_eval.md)。

- `lf_clean_pipeline.py`：当前5682条内容干净新版普通LLaMA-Factory六轮训练。
- `derive_clean_lf_cache.py`：精确数据子集和原生cache预检。
- `lf_train_with_epoch_review.py`：原生训练加进度、epoch保存和101推理callback。
- `epoch_benchmark.py`：同一101条多卡生成。
- `prepare.py`、`train.py`、`run_pipeline.py`：历史TRL实验实现；不作为当前LF启动入口。

历史配置保留在 `docs/archive/20261007/`。依赖和挂载以当前实验快照及runbook为准。
