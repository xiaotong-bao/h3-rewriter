# GRPO 代码

训练操作只看 [GRPO runbook](../docs/runbooks/grpo.md)，统一推理和评分看 [inference + eval](../docs/runbooks/inference_eval.md)。

- `train_userlike_grpo.py`、`existing_lora.py`：继续训练原9B v2 policy LoRA，冻结初始reference adapter。
- `single_pass_gateway.py`、`single_pass_reward.py`：Luna high单轮reward与证据校验。
- `train_grpo.py`：正优势门禁及不可评分候选梯度屏蔽。
- `watch_userlike_astra_eval.py`、`report_userlike_astra_eval.py`：每32步Astra101流水评测和报告。
- `ablate_9bv2_inputs.py`、`prepare_eval_manifest.py`：多卡原101推理和评审manifest。

`luna_judge.py`、旧pipeline、v4/v5状态审查等属于历史实验；参数和缓存不能与当前single-pass混用。
