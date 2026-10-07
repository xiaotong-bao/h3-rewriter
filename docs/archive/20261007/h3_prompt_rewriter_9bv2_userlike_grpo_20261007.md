> 历史快照：仅供追溯，不是当前操作入口。当前仅使用 [inference + eval](../../runbooks/inference_eval.md)、[SFT](../../runbooks/sft.md)、[GRPO](../../runbooks/grpo.md)。旧状态及评分协议不代表当前状态。

# 9B v2 userlike GRPO：当前训练和评测

训练根目录：`/data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/`。

## 数据和初始化

新版 `userlike_gemini_20261007/userlike/train.jsonl` 中抽取 1,000 条：t2va 500、i2va 500；seed 20261007。用户文字去技术引用 tag，媒体放文字前。原数据保留。101 benchmark 和验证集原文及媒体内容排除在训练之外。

选样：`selected_userlike_1000.jsonl`；实际 GRPO 输入：`pilot_inputs.jsonl`；选样 SHA256：`9c4f03bbf3cb54ccb5f538a7ebb317474e99e03025f0beea583d4dbc4bc8b3c6`。

原 base `/work/models/Qwen3.5-9B` + `/work/runs/llamafactory_sft_v2/checkpoint-604` 原 SFT LoRA；不合并、不创建新的 policy LoRA。r=64、alpha=128。冻结参考 adapter 精确复制初始 policy 的值和 dtype；联调确认初始 logits 差为 0。原 SFT checkpoint 保留；GRPO 新建 optimizer。

## 训练

入口：`grpo/train_userlike_grpo.py`；参考 adapter：`grpo/existing_lora.py`；质量门控 trainer：`grpo/train_grpo.py`。

8 GPU DDP；500 optimizer steps；microbatch 1/GPU、梯度累积 2、每个 prompt 8 generations；LR 5e-6、beta 0.02、DAPO、batch reward scaling、BF16、梯度 checkpointing、dropout disabled。生成最多 2048 tokens、temperature 0.8、top_p 0.95；HF enable_thinking=False，显式 EOS 248046、pad 248044。每 16 steps 保存，保留全部 checkpoint。

容器：`xiaotong-9bv2-userlike-grpo-20261007`。

```bash
docker logs --tail 10 xiaotong-9bv2-userlike-grpo-20261007
cat /data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/pilot/status.json
```

启动记录和代码快照：`formal_training_launch_receipt.json`、`formal_training_code/`。输出 `pilot/checkpoint-N/`，完成后 `pilot/final_adapter/`、`pilot/initialization_receipt.json`、`pilot/COMPLETE`。未完成时不能宣称 reference 全程不变；该断言在结束保存前执行。

## Reward

入口：`grpo/single_pass_gateway.py`，本机 127.0.0.1:8799；GPT-6 Luna high，`retention-single-pass-v3-luna-userlike-v1`；单次完整语义覆盖审核并检验逐字证据。六个校准 case 已通过。

`grpo/single_pass_reward.py`：reward=max(-1, 1-0.8*min(severe,2)-0.1*min(general,3)-0.3*format_invalid)。严重问题或格式不合格不允许正优势；运行失败不伪造分数，屏蔽该样本梯度。

## 自动 101 条评测和报告

`grpo/watch_userlike_astra_eval.py`：每 32 steps（32…480）及最终500，按顺序排队；101 个 original benchmark，GPT-6 Astra low，冻结规则 `retention-single-pass-v3-pilot`。

`grpo/ablate_9bv2_inputs.py`：8 卡分片推理，HF no-think、base+当前 LoRA、不合并、BF16 autocast、greedy、max2048、EOS248046。与训练共用 GPU，会影响吞吐。保留原先单卡 step192 的24条后改为8卡；无重复生成已完成样本。生成与 Astra 8 路评分流水并行。失败自动重试；只有最终101/101成功才写 COMPLETE。

`grpo/report_userlike_astra_eval.py`：每次完成生成 `step-N/REPORT.md`，含完整严重/一般问题和相对前轮变化。

结果根目录：`astra_eval_every32/`；`COMPARISON.md`、`results.json` 为评分汇总，`REPORTS.md` 为报告索引，`status.json`/`step-N/stream_status.json` 为状态。报告文件自动更新；本对话没有后台消息推送。

## 已完成评测快照

以下为写入时快照，最新以结果根目录为准。

| Step | Valid | Content | Format | Severe | General | None |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 101/101 | 82.87 | 100.00 | 11 | 21 | 69 |
| 64 | 101/101 | 85.64 | 100.00 | 10 | 15 | 76 |
| 96 | 101/101 | 87.03 | 100.00 | 8 | 17 | 76 |
| 128 | 101/101 | 85.74 | 100.00 | 9 | 18 | 74 |
| 160 | 101/101 | 88.51 | 100.00 | 8 | 12 | 81 |
| 192 | 101/101 | 89.60 | 100.00 | 6 | 15 | 80 |
| 224 | 101/101 | 90.00 | 100.00 | 5 | 17 | 79 |
| 256 | 101/101 | 87.23 | 100.00 | 9 | 13 | 79 |
| 288 | 101/101 | 90.20 | 100.00 | 6 | 13 | 82 |

## Step 224 S3 备份

完整 checkpoint 21文件、2,790,762,269 bytes，含 optimizer、scheduler、RNG、trainer state 和参考 adapter：

```text
s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/backups/step-224/checkpoint-224/
```

远端清单及大小匹配，主 LoRA 权重回读 SHA256 `f6c0356d9f991cc74000a893ad4c0586e68263dda086fbbda0bb387d156698b6`。上级目录含逐文件 SHA256 manifest 和 verification receipt。

## Step 320：下载、inference 和 evaluation 复现

本机权重：`/data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/pilot/checkpoint-320/`。

完整 checkpoint S3 地址：

```text
s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/backups/step-320/checkpoint-320/
```

以下命令在 HB10 主机运行；复用现有 base、镜像、benchmark 输入和本机 Codex 登录。checkpoint 是 LoRA，不能作为独立完整9B模型加载；必须搭配 Qwen3.5-9B base。远程机器还需要准备这几个依赖及 benchmark 图片。

下载到新的目录（本机原 checkpoint 已在，不必重复下载）：

```bash
aws --profile r2w   --endpoint-url https://f25b0ac4c45a2442f62961145a64d158.r2.cloudflarestorage.com   s3 cp   s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/backups/step-320/checkpoint-320/   /data/xiaotong/h3_rewriter_sft_20261002/restored_step320/   --recursive
```

复现 inference（8卡并行；使用新输出目录，避免混入旧结果）：

```bash
cd /home/xiaotong/h3-rewriter
docker run --rm --gpus all --network none --shm-size=8g   -v /data/xiaotong/h3_rewriter_sft_20261002:/work   -v /home/xiaotong/h3-rewriter:/code:ro   -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=1   -e PYTHONPATH=/work/trl_sft_official_v2_20261006/vendor:/work/grpo_20261005/env/lib/python3.11/site-packages   h3-rewriter-grpo:20261005   /work/grpo_20261005/env/bin/python -u -m torch.distributed.run   --nproc_per_node=8 --master_port=29862   /code/grpo/ablate_9bv2_inputs.py   --checkpoint /work/grpo_9bv2_userlike_1k_20261007/pilot/checkpoint-320   --output /work/grpo_9bv2_userlike_1k_20261007/manual_step320/generation   --templates HF --merges unmerged --autocast --label-suffix GRPO_step320
```

Astra evaluation（主机运行，需要网络和 `codex` 登录）：

```bash
cd /home/xiaotong/h3-rewriter
python3 grpo/prepare_eval_manifest.py   --results /data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/manual_step320/generation/results.jsonl   --output /data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/manual_step320/manifest.jsonl
python3 review/single_pass_compare.py   --manifest /data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/manual_step320/manifest.jsonl   --output /data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/manual_step320   --model gpt-6-astra --effort low --workers 8
```

输出 `summary.json`（分数和失败数）、`audits.jsonl`（101条证据）、`COMPARISON.md`。只有 valid=expected=101 且 failures=[] 才是完整有效评测。生成固定 greedy；Astra 重新调用可能有评分波动。

直接复用已有 step320 输出评测时，把上述 results 路径换成 `astra_eval_every32/step-0320/generation/results.jsonl`，无需重跑 inference。
