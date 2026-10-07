# H3 prompt rewriter：固定评价规范

2026-10-06。FAL/9B v2 各 30 条校准已完成并复核，规则冻结为 `retention-single-pass-v3-pilot`（ID 中 pilot 为校准阶段历史命名，正式全量评测沿用同一规则）。EP3 两组试评和全量推理等待可访问的 checkpoint-906，不声称四组已经完成。旧 runbook §1–15 保留历史实验口径，本规范是用户此次指定的新协议，不追溯修改旧 reward 或旧评分。

## 1. 101 prompt 模型评测：GPT-6 Astra low

评测对象：FAL、旧 Qwen3.5-9B checkpoint-604 retention_v2（9B v2）、当前 TRL EP3 checkpoint-906 greedy 和 EP3 checkpoint-906 sampling。共三个模型、四组结果，每组 101 条。

FAL/9B v2 使用已保存原始输出；EP3 两组重新推理。greedy：do_sample=false；sampling：do_sample=true、temperature=0.8、top_p=0.95，每个 ID 一次，不挑选最佳候选。两组 max_new_tokens=2048、repetition_penalty=1.0、官方未修改模板、enable_thinking=false、原始图片与文字顺序、同一 retention v2 system。固定逐 ID 种子并记录 SHA、参数、EOS 与截断信息。sampling 的单次结果不代表多次采样的期望表现。

四组按 ID 核对原文、任务、图片身份、时长和比例。每个模型都以原始要求为依据，不拿 FAL 当标准答案。输入缺失或条件不一致不得冒充完整 101 比较。

评审模型固定 gpt-6-astra，reasoning_effort=low。调用仅提供原文、任务条件和原始改写，不包含模型名称或实验标签。一轮覆盖全部明确要求，在该轮检查状态变化和结局；没有独立状态审查。等级只有 none/general/severe，不设 review/uncertain。没有明确违反原文的证据不扣分，调用失败单独记录并重试，不伪装成无错误或低分。

| 等级 | 判定 | 内容分 |
|---|---|---:|
| none | 所有明确关键要求保留，未发现确证错误 | 100 |
| general | 核心事件、主体关系和结局正确，存在局部范围、强度、时间、镜头、视觉或声音细节偏差 | 70 |
| severe | 关键实体/身体部位遗漏、主动作遗漏或改变、主体受事/说话者绑定错误、关键数量改变、因果/方向/状态变化/结局错误、违反明确禁止要求 | 0 |

每条取最高等级，不按错误条数重复扣内容分。镜头偏差只有使关键事件不可见或改变核心事件时才升为 severe；不能把所有明确细节都当作严重问题。兼容原文的合理补充、光线、服装、音效及新增配乐允许。配乐仅在违反原文明确音乐/静音要求时计错。

30 条试评复核后的边界：原文未指定 wipe 的动作主体，不自行限定必须由某物遮挡镜头；朝镜头伸手递出已取出的物体可以等价保留交给摄影者，不强求接收动作特写。主体事件保持时，未指定确切数量的泛称复数变成单数至多为一般问题；确切关键人数/数量或多人互动被改变才属于严重问题。合理写实描写可等价保留 hyper-realistic，不要求字面复制。树苗长成巨大树木但未说明高过森林属于一般尺度问题，未长大/变小/成长事件缺失才严重；合理表述明确支配周围森林景观可保留尺度要求。未指定从地面起飞或穿越边界时，向太空深处高速前进可保留 into space，不额外发明地面起点。这些规则适用于所有组，不按模型或特定 ID 改判。

格式分独立：integrated_multimodal_description、overall_soundscape、non_diegetic_music 三字段各出现一次、顺序正确且非空，满足得 100，否则 0。旧 13 项检查可作为辅助诊断，但不混入本次三字段格式分。上下文时长/比例不要求逐字复述，明确冲突才按内容错误处理。

30 条固定校准样本：15 t2va + 15 i2va，随机种子 20261006；ID 清单单独保存。先评审所有可用组，逐项复核严重错误与一般错误边界、遗漏与等价表达、配乐允许规则。若修改 codebook，增加版本号并重跑全部 30 条；正式版本确定后，全部 101 使用同一版。试评属于本批开发校准，不宣称独立泛化验证。

最终只提供分数比较，不做逐条胜平负。表格列：模型/解码、有效条数、平均内容分、平均格式分、严重问题 case 数、一般问题 case 数、无问题 case 数。后三列互斥，相加为有效条数；多错误 issue 数不能混当 case 数。每组另给 Top 5 具体问题：优先 severe、再 general，同级按 ID 和问题顺序确定，附样本 ID、原文证据、输出证据/遗漏说明和原因。问题不足 5 个时报告实际数量，不编造。每个 EP3 解码组分别给 Top 5。

规范实现：[single_pass_compare.py](../../review/single_pass_compare.py)。保存原始输出、原文、SHA、完整逐项 coverage、证据、问题级别、judge 模型/effort、规则版本、失败记录、汇总分及 30 条复核记录。纯文本评审不验证未展示图片的视觉忠实度。

## 2. 日常 reward：GPT-6 Luna

评审模型固定 gpt-6-luna，reasoning_effort=high。沿用同一单轮内容 codebook：取消独立状态审查、取消待复核、新增配乐允许。要求完整 coverage 和每个问题的原文/输出证据；重复的同一错误只计一次。按问题数连续扣分，用于日常 reward，不拿 Luna reward 替代 Astra 101 模型评测分。

令 S 为 severe 问题数、G 为 general 问题数、F 为三字段格式失败（0/1）：

`reward = max(-1, 1 - 0.8 * min(S, 2) - 0.1 * min(G, 3) - 0.3 * F)`

无错误且格式通过为 1；一个一般错误为 0.9；一个严重错误为 0.2；两个严重错误为 -0.6；仅格式失败为 0.7。严重错误或格式失败的样本不能获得正优势（训练接入时门禁），即使原始 reward 为正。judge 失败为不可评分，重试后仍失败记录 judge_failed，训练接入时梯度贡献为 0，不能把服务失败算作内容问题。

这是一份新版 reward 规范及独立评测入口；现有 GRPO v4/v5 在线训练代码仍按原版运行，接入新版需要显式选择版本并独立缓存，不能静默替换活跃训练服务。

固定规则后修改需新版本、新缓存和重新评测，禁止跨版本合并结果。

## 3. 本次 101 评测结果与推理状态

Astra low 自动评测，规则 `retention-single-pass-v3-pilot`；内容取最高问题等级，分数不是问题数平均。两组各 101 条有效，调用失败为 0。以下是评审模型的原始判级结果，30 条校准已经人工复核，全量保留每个问题的证据和原始判级，不伪装为逐条人工裁决。图片身份核验仍需 EP3 的官方输入 manifest，现有两组已核对原文、任务、时长、比例，纯文本分数不包含图像忠实度。

| 模型/解码 | 有效/目标 | 内容分 | 格式分 | 严重 case | 一般 case | 无问题 case |
|---|---:|---:|---:|---:|---:|---:|
| fal | 101/101 | 87.13 | 100.00 | 7 | 20 | 74 |
| 9bv2 | 101/101 | 87.43 | 100.00 | 7 | 19 | 75 |
| EP3 greedy | 0/101，待推理 | — | — | — | — | — |
| EP3 sampling | 0/101，待推理 | — | — | — | — | — |

EP3 阻塞：当前机器没有 HB10 的 `/data/xiaotong/h3_rewriter_sft_20261002`；只读检查 R2，`trl_sft_official_v2_20261006/` 前缀未发现发布对象。需要 checkpoint-906 adapter、对应基础模型、system 和官方 101 输入/引用媒体的可访问位置，或 HB10 连接地址。不能把旧 checkpoint-604 当作 EP3。当前两组得分不构成三个模型比较结论。

### 9bv2 Top 5

- **i2v_024_time_lapse_reverse_steak / severe**：The required sauce appearance is explicitly reversed into disappearance. No later sauce appearance restores the requested transition.
  原文：Sauce and garnish appearing
  输出：As the time-lapse reverse begins, the brown sauce and yellow garnish pieces smoothly slide off the plate and vanish into the surrounding white tablecloth.
- **i2v_031_montage_couple_leaving_waving / severe**：Generic transitions do not establish the explicitly required snap transitions, and the entire soundscape omits transition snaps. This removes the requested audiovisual transition mechanism throughout the montage.
  原文：Each moment snaps to the next. Snap sound on each transition
  输出：At 03.500, the shot transitions to a medium shot of the couple walking away from the camera, hand in hand, down a path lined with guests holding lit sparklers.
- **i2v_044_nothing_happens_s_just / severe**：A continuous push-in replaces the explicitly required still frame with a moving shot. Because the entire requested event is an unchanged still frame, this changes its core identity.
  原文：Nothing happens. It's just a stillframe.
  输出：Throughout the 5.16667-second duration, the camera executes a very slow, continuous push-in, gradually magnifying the heart and the wooden stick while maintaining the exact center framing.
- **t2v_004_reality_warp_city_folds / severe**：Bending an individual skyscraper does not depict the principal city-wide action of folding upon itself.
  原文：A city folds impossibly upon itself
  输出：In the center midground, a massive skyscraper bends impossibly like thick rubber, its glass windows reflecting the shifting sky.
- **t2v_004_reality_warp_city_folds / severe**：The required ongoing reversal of spatial orientation is absent throughout the rewrite. Ordinary camera movements and gravity-defying surfaces do not establish this transition.
  原文：Perspective constantly shifts – up becomes down
  输出：未找到对应内容

### fal Top 5

- **i2v_022_eggs_sizzle_hot_pan / severe**：The principal action requires flipping the eggs, but only one of the two established eggs is flipped; the other egg never receives the required action.
  原文：A spatula slides underneath and flips them gently.
  输出：At 00:06.500, a metal spatula enters the frame from the right, sliding gently underneath the right egg. The spatula lifts the egg slightly, revealing the lacy, cooked white underneath. The spatula then performs a gentle flip, with the yolk remaining firm as it settles back onto the heated surface, the steam billowing momentarily with the movement.
- **t2v_000_fisherman_reeling_large_fish / severe**：The rewrite depicts resisting the fish, line paying out, and subsequently netting it, but omits the principal requested action of reeling it in. The spinning reel during drag release does not establish retrieval.
  原文：A fisherman reeling in a large fish.
  输出：The fishing line zips out from the reel at a high speed.
- **t2v_015_basketball_swishing_through_net / severe**：Two rim impacts do not preserve the required two revolutions around the rim. The specified count is attached to a different action.
  原文：spinning around twice
  输出：hits the metal iron twice, spinning violently around the cylinder with a metallic clatter before gravity takes.
- **t2v_015_basketball_swishing_through_net / severe**：The ball drops through in Shot 3 at 00:05.000, and the shooter is first shown celebrating afterward. Being already in celebration at that later point does not establish the required pre-basket celebration.
  原文：already celebrating before the ball goes in.
  输出：[Shot 4] At 00:07.500, the camera cuts to a wide shot looking 50 feet down the court. (S1), a young male player in a jersey, is already mid-stride, arms wide and fists pumping in a wild celebration.
- **t2v_025_kid_teaching_their_grandparent / severe**：The rewrite jumps from walking into a wall to a generic victory screen without establishing Grandpa defeating a basic enemy. The specific principal achievement that motivates the celebration is missing.
  原文：Grandpa finally defeats a basic enemy
  输出：At 00:07.000, the camera cuts back to a medium shot as the TV flashes a bright "VICTORY" screen in gold.

机器可读评测结果、各轮校准记录、冻结规则 SHA 与原始输出在本机 `/home/xiaotong/h3_three_model_eval_20261006/`。训练与评审代码随本仓库提交；运行数据和 checkpoint 不纳入 Git。

## 4. HB10 丢失后的重训：2026-10-06

用户确认 LR `5e-5`、`5 epochs`，其余训练参数沿用原配置。硬件为当前 8×H100 80GB，历史 HB10 为 H200，不能声称逐位重现旧权重。原 Qwen3.5-9B 与固定数据从只读 R2 备份恢复，不加载旧 step604 adapter。重建后的 train/val SHA256 分别完全匹配 `c9f4f14894a1af29310b60c289bcacb0018548dc8f8b5eb4415bfc257485cc4c` 与 `8a0757cb4a531edfc7d361ff58d8f0e7075fc0d787bdc726bfacf2bd4edf5f02`。原始基础模型 tar 的 SHA、官方模板 SHA、实际依赖版本均已核验。

运行目录：`/mnt/nfs/xiaotong/h3-rewriter/runs/sft_retrain_20261006/`；当前容器：`xiaotong-h3-sft-5ep-20261006-r1`（首个容器缺少 C 编译器，在 smoke 开始计算时失败且未完成 optimizer step，已保留证据；镜像补齐 build-essential 后重跑）。八卡 smoke 和全量 9859 条 preflight 通过后自动开始正式训练，保留每 25 steps 恢复点和每轮 checkpoint/101 greedy 输出。预计总 1510 steps，EP3 约 906，以实际 trainer_state 为准。

依赖：Torch 2.8.0+cu128、Transformers 5.6.0、TRL 1.14.1、PEFT 0.18.1、TorchCodec 0.7.0；Python 3.11.13（备份环境原 3.11.14）。容器使用 PyTorch 2.8 runtime + FFmpeg；网络隔离。新 101 输入由已保存原文、任务条件和原图重建，并保留来源 receipt；不能声称旧 HB10 101 输入文件 SHA 已复现。

启动命令：

```bash
docker run -d --name xiaotong-h3-sft-5ep-20261006 --gpus all --network none --shm-size=16g \
  -v /mnt/nfs/xiaotong/h3-rewriter/runs/sft_retrain_20261006:/work \
  -e OMP_NUM_THREADS=1 h3-rewriter-sft-retrain:20261006 \
  /work/grpo_20261005/env/bin/python -u /work/trl_sft_official_v2_20261006/run_pipeline.py \
  --epochs 5 --learning-rate 5e-5
```

Slack 进度：用户明确授权每 20 分钟向本人 Pika 私信发送实际状态。`h3-rewriter-slack-progress.timer` 调用 `sft/slack_progress.py`；每次读取实际 phase/step/loss/GPU，积累足够实测跨度后计算 ETA，不编造进度。首次动态发送已成功。训练完成发送最终通知后停 timer；失败状态如实通知。停止通知：`systemctl --user stop h3-rewriter-slack-progress.timer`。

## 5. 五轮逐 epoch 标准评审

用户要求每个 epoch 都做标准 101 评审。固定 gpt-6-astra / low、`retention-single-pass-v3-pilot`，同一原文/任务/时长/比例、未经清洗的 greedy 输出；单轮审查、无独立状态审查、无待复核等级、新增配乐允许。内容分 none/general/severe 为 100/70/0，三字段格式分独立。

`review/epoch_reviews.py` 由 `h3-rewriter-epoch-review.service` 持续运行，每 30 秒检查各轮 benchmark 的 COMPLETE 和 101 个唯一 ID。逐条核对原文、冻结 manifest SHA；只有全部 101 条有效、无 judge 失败才写评审 COMPLETE。失败保留并重试，未完成不填写正式分数。不改变训练进程或 loss，不占用训练 GPU。每轮保存原始输出/证据、分数、严重/一般/无问题 case 数、Top 5。

实时表：[各 epoch 与 FAL/9B v2 固定参考](../../runs/sft_retrain_20261006/epoch_reviews/EPOCH_COMPARISON.md)。每轮详细报告在 `runs/sft_retrain_20261006/epoch_reviews/epN/COMPARISON.md`，完整评审在 `audits.jsonl`。Slack 每 20 分钟通知同时报告各轮评审状态/得分，训练及五轮评审都完成后才停止。

首次逐轮结果：重训 EP1 / checkpoint-302 的 Astra low 101 条评审已完成，无 judge 失败。内容分 69.31，三字段格式分 96.04；严重 25、一般 20、无问题 56 个 case。完整证据和 Top 5 见 [EP1 报告](../../runs/sft_retrain_20261006/epoch_reviews/ep1/COMPARISON.md)。该记录为新重训的 EP1，不能与历史 HB10 EP1 或旧“明确问题”标签混同。后续轮次结果自动更新实时表。

### 2026-10-07：五轮标准评审完成

训练完成 1510/1510 steps、5 epochs。下表均为固定 Astra low 对同一批 101 条输出的自动评审原始分，无人工改分。

| 模型 | 内容分 | 格式分 | 严重 case | 一般 case | 无问题 case |
|---|---:|---:|---:|---:|---:|
| FAL | 87.13 | 100.00 | 7 | 20 | 74 |
| 9B v2 | 87.43 | 100.00 | 7 | 19 | 75 |
| EP1 | 69.31 | 96.04 | 25 | 20 | 56 |
| EP2 | 76.24 | 100.00 | 15 | 30 | 56 |
| EP3 | 75.05 | 99.01 | 18 | 24 | 59 |
| EP4 | 76.44 | 100.00 | 16 | 26 | 59 |
| EP5 | 81.19 | 100.00 | 13 | 20 | 68 |

EP3 sampling 独立记录于 `runs/sft_retrain_20261006/ep3_sampling_101/`，只有 101 条全部有效才视为正式完成；不混入上述 greedy 结果。

## 6. 旧 9B v2 权重 S3 路径（2026-10-07 核验）

此处按用户“980 v2”指此前比较的旧 **9B v2 / retention v2** 记录：历史 LLaMA-Factory checkpoint-604、epoch 2，101 条 Astra low 内容分 87.43。不是 step980，也不是本次 TRL 重训 EP2 或 EP5。

权重已在私有 R2 的 S3 兼容存储，无需重复上传；本次 head_object 核验 adapter、base 和 env 的远端大小及 SHA256 metadata 与 READY.json 一致。该核验没有重新下载全部对象计算 SHA。

- LoRA adapter（692654080 bytes）：`s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_retention_v2_20261006/assets/adapter.tar`
- 对应 Qwen3.5-9B 基础权重（19329361920 bytes）：`s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_retention_v2_20261006/assets/base.tar`
- 校验记录：`s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_retention_v2_20261006/assets/READY.json`
- 备份运行环境：同目录 `env.tar`。

R2 endpoint：`https://f25b0ac4c45a2442f62961145a64d158.r2.cloudflarestorage.com`，AWS profile：`r2w`。adapter 为 LoRA 权重，加载时必须配合上述 base，不是独立完整模型。

```bash
aws s3 cp s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_retention_v2_20261006/assets/adapter.tar ./9bv2_adapter.tar \
  --profile r2w --endpoint-url https://f25b0ac4c45a2442f62961145a64d158.r2.cloudflarestorage.com
```

SHA256：adapter `2f5d06342a7ef9bfcc8370dd86508ebf1ba9f85047ead9ad32d2ce8b7aba4940`；base `b062b422a48bc4346e96bf59f51ceddf2aa59bb9fd384ba6fb229fc801236e0b`。
