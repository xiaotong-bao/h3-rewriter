# 9B v2 全量 userlike 严格恢复 GRPO（2026-10-08）

## 恢复来源与数据

完整恢复点 `/data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_1k_20261007/pilot/checkpoint-500/`，不是只包含权重的 final_adapter。保留 policy、原 SFT 冻结 reference、optimizer moments/step、scheduler counter 和8个rank的RNG。

训练输入 `userlike_gemini_20261007/userlike/train.jsonl` 全量9659条：ref2va5813、t2va1452、l2va960、i2va954、fl2va480。验证集和original101的文本/媒体内容排除核查通过，无新增排除。teacher答案不参与GRPO。数据SHA、计数和排除见新运行目录的 data_report.json。

新运行目录：`/data/xiaotong/h3_rewriter_sft_20261002/grpo_9bv2_userlike_full_resume_20261008/`。原1k运行和所有 checkpoint 保留。

## 学习率和恢复语义

从 global step500 接续，新增4830步（2个unique prompt/optimizer step，约一个全量数据池采样轮次），目标global step5330。LR在恢复时重设为2e-6，再线性衰减至目标；保留旧optimizer moments及step，不重建KL参考模型，不合并LoRA。

新数据池已改变，ignore_data_skip=True：新数据从头采样，不能宣称与旧数据训练轨迹完全一致。RNG由Trainer从原checkpoint恢复。回调在开始训练时校验完整optimizer状态哈希，policy/reference哈希，以及scheduler.last_epoch==500；写入pilot/resume_receipt.json。

入口 `grpo/train_userlike_grpo.py`，恢复验证 `grpo/strict_resume.py`。针对Transformers把ref子目录当全部adapter而漏载根default的情况，显式恢复根policy；ref先分配FP32再载入，避免BF16临时分配造成舍入。

## 启动前 Luna 审核和 reward 修正

对原step500四个严重输出重新用Luna high打分，原口径漏判地球/滑板上下关系和钓鱼收线；镜子漏手和呼气起雾判为严重。纯Luna加原子覆盖核查后仍偶发漏判。

新revision `retention-single-pass-v3-luna-userlike-v3-astra-guard`：Luna high主评分，凡Luna没有severe，额外用Astra low冻结原标准复核；Astra确认severe则加入错误项、降低reward并禁止正优势。含severe的Luna结果本身已不允许正优势，无需额外调用。reward公式不变。兼容五种任务；ref2va检查六字段，其他检查三字段。

启动校准14/14通过：原有6个合成边界样例、四个错误输出、四个最小修正对照。真实benchmark原文仅用于诊断/校准，不混入训练dataset。记录single_pass_calibration.json保留逐条判定和critical_check。

原1k阶段后期gateway退出，480–499步骤共320候选均评分失败；梯度被mask。新gateway以独立后台进程运行；个别失败样本继续mask，若所有rank的reward均失败，训练直接停止，避免静默完成。新端口8809，日志reward_gateway.log。

## 视频运行和验证

不删除视频样本、不把视频默默替换成图片。`grpo/video_grpo.py`兼容补丁显式把pixel_values_videos/video_grid_thw传入生成、旧policy/ref logps和loss，并按每个样本视频数切分缓存。

运行保持原Torch2.8.0、Transformers5.6.0、PEFT0.18.1。使用原镜像h3-rewriter-grpo:20261005；从本机SFT镜像导出FFmpeg共享库到runtime/ffmpeg_libs，LD_LIBRARY_PATH额外包括该目录及pulseaudio子目录。采样1fps，最多8frames，视频每帧max50176pixels；图片max200704pixels。真实视频前向和LoRA梯度验证通过，见video_forward_preflight.json。

32项单元测试通过，包含多任务格式、LR续期、moment校验和视频缓存切分；另跑恢复后的一步真实视频GRPO诊断，输出resume_probe/，正式阶段仍从原checkpoint500恢复。

## 保存、inference、evaluation 和报告

每100 global steps保存checkpoint，save_total_limit=None。600、700、800……5300各做一次固定original101推理和Astra low评测；最终5330额外评测。8卡推理，HF no-think、未合并LoRA、BF16、greedy、max2048、EOS248046。每次自动生成REPORT.md和汇总表。原step500评测作为此阶段baseline保留。

后台入口 `grpo/watch_userlike_astra_eval.py` 和 `grpo/report_userlike_astra_eval.py`，环境 `H3_GRPO_JOB=grpo_9bv2_userlike_full_resume_20261008`、`H3_EVAL_INTERVAL=100`。

结果目录 `astra_eval_every100/`，正式训练 `pilot/`，进度pilot/status.json，保存恢复证据pilot/resume_receipt.json。数据完整但模型长度预检为每种任务/媒体数量组合的最长用户文本代表，不宣称遍历了全部视频预处理；遇到运行错误需保留日志并从最近完整checkpoint恢复。

## 正式运行恢复检查

正式容器 `xiaotong-9bv2-full-userlike-grpo-20261008` 已启动。8个rank均通过严格恢复检查，逐rank RNG校验结果见 `pilot/rng_restore.rank0.json` 至 `rank7.json`。完整启动参数、容器ID和代码SHA记录在 `formal_training_launch_receipt.json`，运行代码快照在 `formal_training_code/`。
