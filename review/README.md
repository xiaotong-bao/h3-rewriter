# 独立检查与 judge 恢复

- `check_format.py`：检查 raw t2va/i2va 输出的 13 项机械规则，不代表完整语义或 H3 encoder 接收性。
- `retry_failed_audits.py`：在完整 101 条生成保存后，针对失败的 Luna verdict 重试；保留原始输出与原始审计，另存 `evaluation_101/recovered/`。

```bash
python review/check_format.py --input results.jsonl --sources source_metadata.jsonl --output format_audit.json
python review/retry_failed_audits.py --root /data/xiaotong/h3_rewriter_sft_20261002 --job grpo_ep3_luna_v4_20261006 --port 8793
```

后一命令需要结果目录写权限和已就绪的本机 Luna gateway；不会重采样模型输出。
评价 codebook、严重程度、分母和限制见 [统一 runbook](../docs/runbooks/h3_prompt_rewriter_evaluation_training_runbook.md)。

统一 FAL/当前 EP3/合并 EP3/GRPO step32/step64：`unified_compare.py --root <data-root> --output <new-review-dir> --port <local-astra-port> --workers 8`。保存已有 raw 输出，不重生成；同一 GPT-6 Astra high 与冻结 v5 checklist + 独立状态审查协议，不向 judge 发送模型标签。五组共用有效语义配对分母，raw 格式单列；保存每条证据及哈希，失败可续跑。模型自动评分不是人工独立审查，笑脸属于已用来修复评分的开发案例。
