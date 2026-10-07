# 评审代码

固定标准、参数和复现命令只看 [inference + eval runbook](../docs/runbooks/inference_eval.md)。

- `single_pass_compare.py`：GPT Astra low 101模型评测，或GPT Luna high日常reward；单轮coverage和逐字证据。
- `epoch_reviews.py`：当前SFT每轮101评测watcher与汇总。
- `filter_userlike_luna.py`：新版训练数据Luna审查；本次筛选只按内容无问题，忽略格式分。
- `check_format.py`：历史13项辅助结构诊断，不混入当前三字段格式分。
- `retry_failed_audits.py`、`unified_compare.py`：历史gateway/多轮协议工具，不作为当前评分默认入口。
