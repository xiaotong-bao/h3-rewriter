> 历史快照：仅供追溯，不是当前操作入口。当前仅使用 [inference + eval](../../runbooks/inference_eval.md)、[SFT](../../runbooks/sft.md)、[GRPO](../../runbooks/grpo.md)。旧状态及评分协议不代表当前状态。

# 旧 9B-v2 原始 101 输出重新评审（2026-10-07）

使用用户提供的原始 rewrite 文件，101 个 ID 无重复，原文、任务、时长、比例与原 HB10 EP3 的 101 条一致。固定 Astra low / `retention-single-pass-v3-pilot`；不重采样、不清洗输出、不沿用原文件 audit 判级。纯文本评审不验证图片内容或生成视频质量。

源文件：`/data/xiaotong/benchmark_outputs/pika_stage_h3_qwen9b_retention_v2_original101_768p_20261005/h3_rewrite_results.jsonl`。
SHA256：`f72c284bd4f988074824f1153d7dbf0f270b2d304f74804063d3ab26f91404b5`。

| 模型 | 有效/目标 | 内容分 | 三字段格式分 | 严重 case | 一般 case | 无问题 case |
|---|---:|---:|---:|---:|---:|---:|
| 旧 9B-v2，本次重评 | 101/101 | 86.83 | 100.00 | 7 | 21 | 73 |
| 原 HB10 EP3 | 101/101 | 80.59 | 100.00 | 13 | 22 | 66 |
| matched merged EP3 初始化 | 101/101 | 81.68 | 100.00 | 11 | 25 | 65 |
| GRPO v2 step64 | 101/101 | 81.68 | 100.00 | 11 | 25 | 65 |
| GRPO v4 step64 | 101/101 | 78.61 | 100.00 | 15 | 22 | 64 |

本次 9B-v2 比原 HB10 EP3 高 6.24 分，严重 case 少 6 个；仍是这些组中内容分最高。历史 9B-v2 的 87.43 / 7 severe / 19 general / 75 none 保留，不被本次 86.83 替换。同一模型规则重复自动评审存在判级波动，严重 case 总数相同也不表示 ID 完全相同，不能宣称差异显著。

本次首轮及普通重试有一条 `t2v_025_kid_teaching_their_grandparent` 因 partial 问题缺少输出引用未通过校验；单条重试把既有非空引用要求加入支持的返回 schema，未改变评分规则。最终该条为 general，101 条全部有效、最终失败 0。原失败 summary/audits 保存在 `attempt1/`，额外 schema 与单条 cache key 在 `evidence_schema_retry_receipt.json` 和总 rubric receipt 中保存。

完整运行目录：`/home/xiaotong/h3_9bv2_fixed_eval_20261007/`，包含 manifest、101 条 audits、summary、COMPARISON、冻结规则/source SHA、失败记录及单条重试脚本。原 HB10 EP3/GRPO 结果沿用 [对应报告](h3_prompt_rewriter_hb10_ep3_grpo_fixed_evaluation_20261007.md)。

## 全部 7 个严重 case

### i2v_024_time_lapse_reverse_steak

- 原文：Sauce and garnish appearing
- 输出：As the time-lapse reverse begins, the brown sauce and yellow garnish pieces smoothly slide off the plate and vanish into the surrounding white tablecloth.
- 判定理由：The required sauce appearance is reversed into disappearance. Sauce never subsequently appears; steak juices do not establish sauce. This reverses an explicitly required transition within the dinner's assembly.

### i2v_044_nothing_happens_s_just

- 原文：Nothing happens. It's just a stillframe.
- 输出：Throughout the 5.16667-second duration, the camera executes a very slow, continuous push-in, gradually magnifying the heart and the wooden stick while maintaining the exact center framing.
- 判定理由：The required entire event is an unchanged still frame. Continuous magnification explicitly replaces that static presentation with camera motion.

### t2v_000_fisherman_reeling_large_fish

- 原文：A fisherman reeling in a large fish.
- 输出：The fisherman leans forward, his shoulders tensing, as he continues to pull the line.
- 判定理由：The rewrite describes pulling the line, gripping the rod, and landing the fish, but never establishes reeling it in. Reel clicking and drag screaming alone do not establish the requested retrieval action.

### t2v_004_reality_warp_city_folds

- 原文：A city folds impossibly upon itself
- 输出：In the center midground, a massive skyscraper bends impossibly like thick rubber, its glass windows reflecting the shifting sky.
- 判定理由：An individual skyscraper bending does not depict the principal citywide folding action.

- 原文：Perspective constantly shifts – up becomes down, sidewalks lead onto building sides.
- 输出：未找到对应内容
- 判定理由：The required ongoing spatial transformations are absent. Changing camera angles and showing fixed vertical routes do not preserve up becoming down or sidewalks leading onto building sides.

### t2v_009_massive_congratulations_graduates_banner

- 原文：A massive "CONGRATULATIONS GRADUATES" banner being unfurled across a university building by maintenance workers on the roof.
- 输出：The banner finally drops into place, resting flat against the roof.
- 判定理由：The required banner placement across the university building is changed to lying flat against its roof, giving the principal action the wrong target placement and final outcome.

### t2v_041_high_tech_anime_style

- 原文：The target is a heat-signature silhouette in a building 2km away.
- 输出：In the center of the scope's view, a glowing red heat-signature silhouette of a target stands motionless on a distant, illuminated building ledge.
- 判定理由：The target is placed on a building ledge instead of inside a building, changing the required target placement.

### t2v_048_breath_fogging_cold_car

- 原文：then a smiley face drawn in it.
- 输出：A person's breath is visible as a thick, white cloud of fog against the glass, which then begins to spread and form a crude smiley face.
- 判定理由：This replaces the principal drawing action with the fog forming a face itself. The later reference to a drawn face and the finger-rubbing sound do not establish that the smiley is actually drawn in the fog.

