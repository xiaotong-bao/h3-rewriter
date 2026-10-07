# Astra low 101 评测：当前结果

EP3 greedy 与 sampling 尚待可访问权重；当前仅 FAL/9B v2 完成。完整规则和推理状态见 [固定评价规范](h3_prompt_rewriter_fixed_evaluation.md)。

| 模型/解码 | 有效/目标 | 内容分 | 格式分 | 严重 case | 一般 case | 无问题 case |
|---|---:|---:|---:|---:|---:|---:|
| 9bv2 | 101/101 | 87.42574257425743 | 100.0 | 7 | 19 | 75 |
| fal | 101/101 | 87.12871287128714 | 100.0 | 7 | 20 | 74 |

模型 gpt-6-astra / low；规则 retention-single-pass-v3-pilot。问题按等级、样本 ID、问题顺序确定，最多 5 项。

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
