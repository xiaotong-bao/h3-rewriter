# 独立检查与 judge 恢复

- `check_format.py`：检查 raw t2va/i2va 输出的 13 项机械规则，不代表完整语义或 H3 encoder 接收性。
- `retry_failed_audits.py`：在完整 101 条生成保存后，针对失败的 Luna verdict 重试；保留原始输出与原始审计，另存 `evaluation_101/recovered/`。

```bash
python review/check_format.py --input results.jsonl --sources source_metadata.jsonl --output format_audit.json
python review/retry_failed_audits.py --root /data/xiaotong/h3_rewriter_sft_20261002 --job grpo_ep3_luna_v4_20261006 --port 8793
```

后一命令需要结果目录写权限和已就绪的本机 Luna gateway；不会重采样模型输出。
评价 codebook、严重程度、分母和限制见 [统一 runbook](../docs/runbooks/h3_prompt_rewriter_evaluation_training_runbook.md)。
