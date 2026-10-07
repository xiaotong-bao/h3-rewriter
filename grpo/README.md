# EP3 GRPO with Linux-local Luna v5

当前入口默认使用 `grpo_ep3_luna_v5_20261006`、`ep3-v2-luna-state-v5` 和本机 `127.0.0.1:8794`。
`H3_GRPO_JOB`、`H3_LUNA_PORT` 可显式覆盖。新实验从冻结 EP3 初始化，不续训退化的旧 step64 adapter。
v5 状态审查与验证记录见 [runbook](../docs/runbooks/h3_prompt_rewriter_evaluation_training_runbook.md#14-v5-状态转换-reward-修复2026-10-06)。

## 修复

- `luna_judge.py`：先缓存原文完整 atomic requirements，再逐项审查整个 rewrite。所有 source sentences 必须覆盖，所有 requirement IDs 必须恰好判定一次；critical 确认违规不能降级为 general。
- `reward_policy.py`：音乐/台词 flag 必须与 v2 清单一致；与源 checklist 的同一违规通过 covered_requirement_id 关联，只计一次。矛盾判定不进入缓存，调用方会重试。
- `train_grpo.py`：重试后仍失败的 judge 返回 None，固定 TRL 将其排除出组均值/方差；门禁同时清空其 completion_mask，所以连 KL 梯度也不参与。
- `QualityGatedGRPOTrainer`：在 TRL 计算实际 advantage 后，把格式失败、severe、未要求音乐/台词以及关键要求待复核的正优势置零；保留已有负优势。逐条核对 completion token IDs 与 verdict 对齐，记录 raw/gated advantage。这是标准 GRPO 之上的显式约束，不靠加大格式扣分假装硬门槛。
- `calibrate_judge.py`：14 个合成回归样本，包含正确保留、方向/动作/结局/数量错误及音乐/台词/格式违规。不得使用 held-out 101 输出来调训练规则；pipeline 强制校准通过后才启动。

## 运行

保留原来的 8 ranks、microbatch1、累积2、8 candidates/input、64 steps、LR5e-6、beta.02、DAPO、temperature.8/top_p.95、completion cap2048，以及语言 LoRA32/64。每步两个输入，共128个输入呈现和1024候选，并非完整遍历256原文。
`prepare_merge.py` 可从 EP3 重新合并；本次新 JOB 复用经原 EP3 adapter SHA 和官方 template SHA 核验的同一冻结 merged 初始化。
`bootstrap_runtime.py grpo --root /work --job grpo_ep3_luna_v5_20261006` 生成本地授权/source allowlist。

`start_luna_runtime.py` 启动本机 supervisor 并等待 gateway 就绪；不使用 Mac、SSH 或 caffeinate。
`run_pipeline.py` 检查 local Codex runtime/revision 和真实校准 receipt，执行2步smoke、重新从EP3跑64步pilot，再执行matched greedy101评价。
只有完整 adapter/optimizer/scheduler/trainer_state/8 rank RNG 的 checkpoint 才允许自动恢复。
`monitor_training.py --job /data/.../grpo_ep3_luna_v5_20261006` 直接监控本机文件，无 SSH。
可选256 Luna/Sol审查直接读取同目录生成文件，也不再 rsync 到远端。

结果缓存、运行 receipts、模型/媒体和真实原文/输出均留在外部；revision隔离旧v3缓存。
训练 reward 与 Luna 自审仅是探索性指标，不能替代完整101独立审查。

启动流程在 GPU 训练前调用 `precompute_requirements.py`，并发预提取全部训练 prompt 的要求并写入本机 Luna 缓存；全部校验通过才进入 smoke/pilot。正在运行的实验可单独执行该脚本补齐缓存，建议 `--workers 4` 为训练评审保留并发容量。

## v5 独立状态审查
每条输出增加一次独立 Luna 审查，直接读取原文和完整输出，不看第一轮 checklist 判定。逐句明确初始状态、变化、最终状态及输出的最终状态；缺失、矛盾计严重错误，不确定禁止正优势。结构不完整按评审失败处理。该审查增加一次模型调用，不能保证语义零误判。v4 历史训练及评测保留原评分口径。笑脸 benchmark 仅作为已知失败回归验证，不能再作为独立泛化证明。

## 2k 评分问题预设计
`prepare_prompts.py --sft <SFT> --output <JOB> --count 2000` 单独选取去重、隔离 held-out 的真实训练数据；按可用数据平衡文本/图文，不重复填充不足的任务类型。`design_questions.py --inputs <JOB>/pilot_inputs.jsonl --output <JOB>/scoring_questions.jsonl --port <local-port> --workers 16` 预提取要求，为每项要求和原文每个句子保存检查问题，逐条落盘并支持重跑补齐。结构校验不等于语义认证；该文件是评分设计资产，不会自动替换正在运行的 reward。


2026-10-07 当前代码配置：`H3_JUDGE_PROFILE=reward` 使用 Luna / `h3-reward-luna-v1` / 8797；`evaluation` 使用 Astra / `h3-eval-astra-v1` / 8798。默认 JOB 为 `grpo_ep3_luna_2k_20261006`。两个 profile 共享 `judging_standard.py`，源要求审查和独立状态审查并发，再对 severe 候选做严重程度复核。以上为当前源码行为，历史 v2/v4/v5 运行保持其原配置，修改默认值不会迁移旧缓存或证明新 profile 已校准。
