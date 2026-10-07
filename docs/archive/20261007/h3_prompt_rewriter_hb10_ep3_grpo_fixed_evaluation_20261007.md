> 历史快照：仅供追溯，不是当前操作入口。当前仅使用 [inference + eval](../../runbooks/inference_eval.md)、[SFT](../../runbooks/sft.md)、[GRPO](../../runbooks/grpo.md)。旧状态及评分协议不代表当前状态。

# 原 HB10 EP3 与 GRPO：新版固定标准评测（2026-10-07）

本报告对应原 HB10 checkpoint-906（历史 LR 1e-4 / 4 epoch），不是新 H100 LR 5e-5 / 5 epoch 重训的 EP3。使用此前首次保存的 101 条 greedy 输出，不重新采样，不清洗 rewrite；四组原文和输入条件核对一致。两次 GRPO 的合并初始化输出完全一致，合并控制只列一组。

固定评审为 GPT-6 Astra / low，规则 `retention-single-pass-v3-pilot`：单轮检查全部原文要求及结局，允许兼容补充和未被原文禁止的新增配乐；none/general/severe 内容分为 100/70/0，取 case 最高等级。格式只检查三字段各出现一次、顺序正确、非空。各组均为 101/101 有效，最终 judge 失败为 0。所有判级为自动评审原始结果，未作人工改分；纯文本审查不证明图像忠实度或实际视频效果。

| 模型/解码 | 有效/目标 | 内容分 | 格式分 | 严重 case | 一般 case | 无问题 case |
|---|---:|---:|---:|---:|---:|---:|
| HB10_EP3_greedy | 101/101 | 80.59 | 100.00 | 13 | 22 | 66 |
| HB10_EP3_merged_control | 101/101 | 81.68 | 100.00 | 11 | 25 | 65 |
| HB10_GRPO_v2_step64 | 101/101 | 81.68 | 100.00 | 11 | 25 | 65 |
| HB10_GRPO_v4_step64 | 101/101 | 78.61 | 100.00 | 15 | 22 | 64 |

GRPO v2 比原 EP3 高 1.09 分，但与 matched merged 初始化控制同为 81.68，严重 case 也同为 11，因此本次评分未显示 v2 相对其实际初始化有整体改善。v4 比 matched merged 控制低 3.07 分，严重 case 从 11 增至 15，未显示改善。单次自动评审有判级波动，这些差异不构成统计显著性结论，也不能与旧 Luna reward 或历史明确问题数直接比较。

具体变化证据：v2 的 `i2v_019_tilt_camera_up_slowly` 不再被判为严重结局错误，但新增 `t2v_025_kid_teaching_their_grandparent` 的严重问题。v4 的 `i2v_032_lightning_cracks_across_sky` 从严重降为一般，但新增 `i2v_025_spoon_stirred_coffee_cup` 的严重结局错误，以及 `i2v_031_montage_couple_leaving_waving` 的实际离开动作缺失。以下按固定规则列各组 Top 5，不制作逐条胜平负表。

## 数据与复现

运行目录：`/home/xiaotong/h3_hb10_ep3_grpo_fixed_eval_20261007/`。`manifest.jsonl` 保存原始原文/rewrite/条件/checkpoint/SHA；`audits.jsonl` 保存全部 coverage、问题证据及判级；`summary.json` 保存完整统计；`rubric_receipt.json` 保存规则、输入、原始输出文件和本次结果 SHA。缓存及日志不入 Git。原始解码协议为 greedy、max_new_tokens=2048、repetition_penalty=1、官方 Qwen 模板、enable_thinking=false、原始媒体顺序。

```bash
cd /home/xiaotong/h3-rewriter
python3 review/single_pass_compare.py --manifest /home/xiaotong/h3_hb10_ep3_grpo_fixed_eval_20261007/manifest.jsonl \
  --output /home/xiaotong/h3_hb10_ep3_grpo_fixed_eval_20261007 --model gpt-6-astra --effort low \
  --revision retention-single-pass-v3-pilot --workers 8
```



## HB10_EP3_greedy Top 5

- **i2v_007_rides_spin_move_excitement / severe**：The required closing logo-and-tagline appearance after the camera sweep is omitted. Lettering already present on the carousel does not establish the requested logo reveal, and the tagline never appears.
  原文：Camera sweeps the park, then Funworld logo appears with the tagline "Where the fun never stops."
  输出：The central pillar of the carousel is a tall, cylindrical structure wrapped in a repeating pattern of illuminated white letters spelling out "FUN WORLD".
- **i2v_023_stallholder_pockets_cash_shifts / severe**：The rewrite substitutes an unidentified parcel for the required sweet potato. Nothing elsewhere in the rewrite establishes the identity of this central transaction object.
  原文：retrieving a sweet potato wrapped in aluminium foil
  输出：retrieves a small, foil-wrapped parcel
- **i2v_044_nothing_happens_s_just / severe**：A continuous push-in with perspective and shadow changes replaces the required stillframe with an evolving shot, violating the central requirement that nothing happens.
  原文：Nothing happens. It's just a stillframe.
  输出：Throughout the duration of the clip, the camera executes a very slow, subtle push-in, gradually bringing the pink paper heart and the wooden stick closer to the viewer. As the camera moves, the perspective shifts slightly, causing the soft shadow on the right to appear to stretch and deepen marginally against the dark teal surface.
- **i2v_045_person_wearing_highly_reflective / severe**：Moving the sign out of frame removes the background sign needed to show the central requested event: its flickering in perfect synchronization with the reflected text. Only the reflected flicker is visible.
  原文：flickering in perfect sync with the sign in the background.
  输出：The lenses are highly reflective, mirroring the bright, flickering neon sign reading "OPEN" in vibrant red letters, which is positioned just out of frame.
- **t2v_000_fisherman_reeling_large_fish / severe**：The rewrite depicts line paying out, followed by surfacing and netting, but never depicts the fisherman reeling the fish in. This omits the principal requested action.
  原文：A fisherman reeling in a large fish.
  输出：The metallic fishing reel attached to the rod spins rapidly, pulling thick monofilament line out into the water.

### HB10_EP3_merged_control Top 5

- **i2v_019_tilt_camera_up_slowly / severe**：The requested terminal wipe becomes an intermediate transition. The rewrite then ends with a stationary locomotive in a nighttime depot, changing the specified ending.
  原文：Ending by wiping the camera.
  输出：This sudden, sharp wipe acts as a visual transition, completely obscuring the approaching train and the sunset for a brief moment, before revealing a new, contrasting scene in the final seconds.
- **i2v_032_lightning_cracks_across_sky / severe**：The required lightning-then-thunder sequence is changed: thunder and its rumble open the soundscape, and another crack occurs as the first lightning strikes rather than afterward.
  原文：Thunder rumbles after.
  输出：The soundscape begins with the loud, dominant crack of thunder echoing across the open water, immediately followed by a deep, resonant rumble that vibrates through the air. As the first massive lightning bolt strikes, a sharp, explosive crack is clearly heard in the foreground, accompanied by the heavy, rushing sound of wind gusting across the landscape.
- **i2v_044_nothing_happens_s_just / severe**：The required event is complete visual stillness. Continuous magnification and the accompanying perspective changes replace the still frame with a moving shot, directly contradicting the core requirement.
  原文：Nothing happens. It's just a stillframe.
  输出：Throughout the duration of the clip, the camera executes a very slow, deliberate push-in, gradually magnifying the pink heart and the wooden stick while the dark teal background remains a consistent, unchanging canvas.
- **i2v_045_person_wearing_highly_reflective / severe**：Placing the sign out of frame makes the central requested visual relationship—perfectly synchronized flickering of the visible background sign and its lens reflection—invisible. Synchronizing sound with the reflection does not preserve that comparison.
  原文：flickering in perfect sync with the sign in the background.
  输出：The lenses are highly reflective, mirroring the bright, flickering neon sign reading "OPEN" in vibrant red letters, which is positioned just out of frame.
- **t2v_000_fisherman_reeling_large_fish / severe**：Outgoing line preserves the drag event, but the entire rewrite omits the principal action of the fisherman reeling the fish in. Gripping the rod and subsequently netting the fish do not establish retrieval with the reel.
  原文：A fisherman reeling in a large fish.
  输出：The metallic fishing reel attached to the rod spins rapidly, pulling thick monofilament line out of the spool.

### HB10_GRPO_v2_step64 Top 5

- **i2v_032_lightning_cracks_across_sky / severe**：Thunder precedes and coincides with the lightning instead of following it. The subsequent rolling boom does not undo this explicit change to the required lightning-then-thunder sequence.
  原文：Thunder rumbles after.
  输出：The soundscape begins with the low, continuous, and ominous rumble of a distant thunderstorm, layered over the faint, steady howling of wind sweeping across the open water. Early in the clip, a loud, deafening crack of thunder dominates the foreground, perfectly synchronized with the massive lightning strike, accompanied by a sharp, high-pitched electrical sizzle as the bolt branches out.
- **i2v_044_nothing_happens_s_just / severe**：Continuous camera movement and visual changes directly contradict the explicit requirement that nothing happens and the video is a still frame. Keeping the heart and stick motionless does not preserve that requirement.
  原文：Nothing happens. It's just a stillframe.
  输出：Throughout the duration of the clip, the camera executes a very slow, subtle push-in, gradually bringing the pink paper heart and the wooden stick closer to the viewer. As the camera moves, the perspective shifts slightly, causing the soft shadow on the right to appear to stretch and deepen marginally against the dark teal backdrop.
- **t2v_000_fisherman_reeling_large_fish / severe**：No action retrieves line by reeling in. Holding the rod and letting line run out do not preserve the requested principal action.
  原文：A fisherman reeling in a large fish.
  输出：The metallic reel on the rod spins rapidly, pulling thick monofilament line out as the rod creaks visibly under the strain.
- **t2v_000_fisherman_reeling_large_fish / severe**：The capture mechanism changes from the net scooping beneath the fish to the fish leaping and sliding into it. Lowering the net beforehand and subsequently lifting it do not establish the required scoop.
  原文：The net scoops underneath
  输出：The tuna leaps upward, its silver scales catching the ambient light, and slides smoothly into the basket of the net.
- **t2v_009_massive_congratulations_graduates_banner / severe**：The rewrite also unfurls the banner across the roof, changing the required placement across the building into a banner lying on the roof.
  原文：A massive "CONGRATULATIONS GRADUATES" banner being unfurled across a university building by maintenance workers on the roof.
  输出：The massive banner finally drops back down, settling flat against the roof surface.

### HB10_GRPO_v4_step64 Top 5

- **i2v_025_spoon_stirred_coffee_cup / severe**：Withdrawal does not establish the required act of setting the spoon down. The rewrite replaces the requested concluding action with renewed stirring using a different utensil, which continues through the end.
  原文：set down.
  输出：The hand then smoothly withdraws the spoon from the frame entirely.
- **i2v_028_petals_drift_off_leaves / severe**：Blossoms explicitly remain during snowfall; their subsequent restoration replaces the required seasonal return of flowers, changing the cycle’s final transition.
  原文：flowers return
  输出：The white plum blossoms, now dusted with a few lingering, melting water droplets, return to their pristine, vibrant state, swaying gently in the breeze as the camera continues its slow push-in, holding on the enduring beauty of the flowers.
- **i2v_031_montage_couple_leaving_waving / severe**：Preparing to drive off does not show the required departure. The final shot leaves the bride outside the vehicle, so the couple never drives away.
  原文：and driving off.
  输出：Finally, as the groom settles into the seat, the deep, resonant rumble of a car engine starting up is clearly heard, growing louder and more powerful as the vehicle prepares to drive off.
- **i2v_044_nothing_happens_s_just / severe**：Continuous camera movement and changing shadow perspective contradict the central requirement that nothing happens and the video is a still frame. Keeping the depicted objects stationary does not preserve that requirement.
  原文：Nothing happens. It's just a stillframe.
  输出：Throughout the duration of the clip, the camera executes a very slow, subtle push-in, gradually bringing the pink paper heart and the wooden stick closer to the viewer. As the camera moves, the soft shadow on the right side of the background shifts slightly in perspective, enhancing the three-dimensional depth of the scene.
- **i2v_045_person_wearing_highly_reflective / severe**：The required gaze target is absent. Reflecting the sign in her sunglasses does not establish the principal action of looking at that sign.
  原文：is looking at a flickering neon sign.
  输出：As the video progresses, the woman maintains a steady, contemplative gaze, her expression calm and slightly distant.
