# H3 prompt-rewriter data runbook: a 10k (user request → H3 prompt) set labeled with H3-Context-IR

## 当前实际交付（2026-10-06）

“10k”是项目名。实际已完成标签 9899 条；固定排除 40 个未来媒体泄漏样本后，当前训练集合为 9859 条（9659 train / 200 val），seed 42，teacher targets 未改。以下原始 2026-10-02 的 10000 条配额与待建设状态属于历史规划，不代表当前任务尚未制作数据，也不是当前数据的实际任务分布。

- HB10 原始标签及 manifest：`/data/xiaotong/h3_rewriter_sft_20261002/source/{h3_rewrite_results,manifest_api}.jsonl`。
- split：`dataset/{train_partial,val}.jsonl`、`dataset/split_report.json`；固定 40 个排除 ID 见 split report。
- SFT 原始 chat 输入：`lf_dataset/{train,val}.jsonl`；TRL v2 实际输入：`trl_sft_official_v2_20261006/{train,val}.jsonl`。
- 媒体：`media/`，容器挂载根为 `/work`；JSONL 中原始媒体顺序、时长、比例和 teacher 保留。音频 waveform 没有送入 vision-only Qwen，不能据此宣称模型验证了音频引用忠实度。
- 已存在 S3 原始数据：`s3://data-transfer-research/turboscale_migration_202603/xiaotong/h3_rewriter_sft_20261002/final/`（标签、manifest、媒体）和 `training_bundle/`（旧 split/chat 导出）。R2 endpoint、当前完整数据核验状态、TRL checkpoint 与恢复命令统一见 [评价与训练 runbook](h3_prompt_rewriter_evaluation_training_runbook.md#10-数据s3-资产与恢复)。

Goal: training data for our own Qwen prompt rewriter for MiniMax-H3. Each sample is what a user sends — raw text,
optional media, target duration and ratio — paired with the complete H3 prompt that MiniMax's own rewriter
(**H3-Context-IR**, `POST /v2/h3_context_ir`) writes for it. First set: **10,000 labeled samples — 5,000 across
t2va / i2va / fl2va / l2va and 5,000 ref2va covering all 43 reference-mode categories.**

Written 2026-10-02. Worktree `/mnt/nfs/cade/codes/pika3.0-h3` (branch `features/h3-distillation`); everything
below is untracked unless noted.

**Status**

| Piece | State |
|---|---|
| API access, contract, per-task output formats | verified, 17 calls, §2 |
| Labeling driver `packages/pika-data/scripts/label_h3_context_ir.py` + `pika_data/h3_context_ir.py` | built; 6 unit tests; live 4-row run labeled, resumed and assembled |
| Python env `/mnt/nfs/cade/venvs/h3-rewriter-data` (stdlib driver + `tokenizers`) | built |
| Seed-source audit (what each source can supply) | done, §4 |
| Manifest builder, user-ask synthesizer, media normalizer, training-format export | **not built** — steps 1–3 and 8 |
| Dan's `ref2va_v3` media for 24 of 43 categories | **blocked on a regroup by Dan**, §4.2 |
| Pilot, full run | not started |

Decisions still open are listed in §8. Nothing here trains a model; the hand-off to training is step 8.

---

## 1. What a sample is

```json
{"sample_id": "pika_i2v_media_024c8e84", "task": "i2va", "user_text": "a dog is running",
 "duration": 5, "ratio": "adaptive",
 "media": [{"type": "image", "role": "first_frame", "path": "/abs/in_0.png"}],
 "source": "pika_api_user/minimax-h3", "stratum": "i2va_real", "category": null, "org_id": "…", "lang": "en"}
```

| Task | `media` roles, in order | `ratio` | Label starts with |
|---|---|---|---|
| t2va | none | explicit (`16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9`) | `integrated_multimodal_description:` |
| i2va | `first_frame` | `adaptive` | `For the target video, at 0.00 seconds …` |
| fl2va | `first_frame`, `last_frame` | `adaptive` | `How the reference pictures align … Picture 1 … Picture 2 …` |
| l2va | `last_frame` | `adaptive` | `How the reference pictures align … <Picture 1> …` |
| ref2va | `reference_image` ≤ 9, `reference_video` ≤ 3, `reference_audio` ≤ 3, any mix | explicit or `adaptive` | `subject_definitions:` (six sections) |

`duration` is an integer 4–15. Media order is the `<Picture/Video/Audio N>` order. Extra columns are carried through
to the output. The label is the whole prompt, instruction line included — feed it to H3-Base verbatim.

## 2. The teacher: H3-Context-IR

One endpoint for every task; the media roles select the task. Create returns a `task_id`; poll
`GET /v2/query/video_generation/{task_id}` until `task.status == "succeeded"`; the label is `task.content.prompt`.
No video is generated. Docs: `platform.minimax.io/docs/api-reference/video-generation-v2-h3-context-ir`.

- Key: `/mnt/nfs/cade/codes/pika3.0/env/.minimax_console_key`, global endpoint `https://api.minimax.io`
  (console `platform.minimax.io`). Never put the key in a command line, config or log; the driver reads the file.
- Price: $0.90 / M input tokens, $3.60 / M output tokens. Billed tokens are much larger than the visible text: a
  fixed input of ~5.7k (t2va), 13–18k (keyframe tasks), ~30k (ref2va), and output billed at 4–6× the returned prompt.
- Limits: text ≤ 7,000 chars; image JPG/PNG/WEBP ≤ 30 MB, sides 256–5760 px, aspect 0.4–2.5; video MP4/MOV
  H.264/H.265, 2–15 s each and ≤ 15 s in total, ≤ 50 MB, 23.976–60 fps; audio WAV/MP3 2–15 s each and ≤ 15 s in
  total, ≤ 15 MB; request body ≤ 64 MB. Errors: 422 content filter, 429 rate limit, 402 balance.

Measured on 2026-10-02 (requests and responses in `$RUN/probes_20261002/`):

| Task | Calls | Cost per call | Wall time | Label length (H3 tokens) |
|---|---|---|---|---|
| t2va | 5 | $0.012–0.022 | 23–41 s | 313–490 |
| i2va | 2 (+1 README) | $0.028–0.048 | 48–66 s | 864–1,204 |
| fl2va | 3 | $0.024–0.029 | 35–98 s | 401–618 |
| l2va | 2 | $0.032–0.043 | 54–75 s | 910–957 |
| ref2va | 5 (+1 README) | $0.040–0.060 | 55–112 s | 693–920 |

Facts the pipeline depends on:

1. **No seed.** The same request returns a different prompt each time. A label that fails a gate is drawn again.
2. **Token cap.** Our encoder rejects presentations over 1,024 text tokens (`h3_core/geometry.py`,
   `conditioning.py`). The presentation adds labels and the chat template, so the gate is **940 prompt tokens** with
   the H3 tokenizer. 2 of 17 labels were over it (i2va 1,204, l2va 957).
3. **Header time is the frame grid.** The keyframe line carries the smallest `17 n + 5` frames ≥ `24 × duration`,
   over 24: 5 s → 5.17, 6 s → 6.58, 8 s → 8.00. `pika_data.h3_context_ir.api_seconds` reproduces it.
4. **A reference video's soundtrack takes an `<Audio N>` label** ahead of the uploaded audios. A user's "audio 1"
   can therefore be `<Audio 2>` in the label. Strip the audio stream from any video whose sound is not meant to be
   used.
5. **WAV must be sent as `audio/wav`**; `audio/x-wav` is rejected (error 2013, empty usage record). MP3 is sent as
   `audio/mp3` — not yet exercised.
6. Local files go inline as base64 data URLs (images, mp4, wav verified). No upload step.
7. Non-English input gives an English description with the spoken line kept in the original language inside
   `<d>[Language] …</d>`.
8. Context-IR decides the reference mode itself. A terse ref2va request can be read as editing or as generation;
   the label's `summary` tag records which.
9. Not known: the rate limit for this endpoint (video generation is documented at 300 RPM, 30 tasks in flight);
   whether 422 responses are billed; the share of user prompts the content filter rejects.

## 3. Environment

```bash
cd /mnt/nfs/cade/codes/pika3.0-h3
PY=/mnt/nfs/cade/venvs/h3-rewriter-data/bin/python          # stdlib + tokenizers 0.23 (H3 tokenizer for the cap)
TOOL=packages/pika-data/scripts/label_h3_context_ir.py
RUN=/mnt/nfs/cade/codes/pika3.0/misc/dit_train_data/h3_rewriter/ctxir_10k_v1
```

The driver is stdlib-only and also runs under the hosts' `python3`; without `tokenizers` it falls back to a
character estimate of the token count that rejects some valid labels, so use `$PY`. No GPU, no docker.

Driver modes: `--mode check` (no network: limits, per-task counts, projected cost), `label` (default), `assemble`.
Every attempt is appended to `<output>.sidecar.jsonl`; re-running skips rows that are `ok` or `blocked`,
re-checks `unsendable` rows for free, and re-queues `invalid` / `error` rows only with `--retry-failed`. It stops
submitting when the measured spend reaches `--max-spend-usd` (default 5) and exits 3; 401/402 abort with exit 2.
Statuses: `ok`, `invalid` (label failed a gate), `blocked` (content filter), `error`, `unsendable`, `missing`.

Gates in `validate_label` (errors reject the label; warnings are stored in `ctxir_meta.warnings`):

- all tasks: ≤ 940 tokens; no code fence or URL; `<Picture/Video/Audio N>` indices within what was supplied
  (video soundtracks count as audios); cut times increasing and inside the clip;
- base tasks: the exact instruction line for the task, header time on the frame grid, three fields present,
  shots numbered from 1;
- ref2va: six sections in order, every `<Subject N>` defined, `summary` opens with a `[task]` tag;
- warnings: a quoted user line missing from the dialogue, a supplied reference never mentioned, no `[Shot N]`
  marker in a ref2va description, an unknown task tag, `<d>` without a language tag.

## 4. Composition of the 10k

Held out everywhere (never in the training manifest): `arena_ti2v_v1` (100 pool rows, join `arena_prompt_id`),
`arena_v2v_v1` (100, same key), `pika_r2v_v2` (86 = Dan's `ref2va_v3_eval` case dirs). `candy_v1`,
`nova03_female_v1` and `h3post_mixed_v1` share no prompt or image with any source below.

### 4.1 t2va / i2va / fl2va / l2va — 1,250 each

Sources: the API-user mirror `/mnt/nfs/collab/data/dit_train_data/pika_api_user/{minimax-h3,bytedance-seedance-2.5,bytedance-seedance-2.0}`
(`jsonls/t2v_i2v.jsonl`; `caption` is the raw user prompt; inputs at `inputs/dt=<created_at[:10]>/<id>/in_<i>.*`;
outputs are local mp4s) and the arena pool
`/mnt/nfs/cade/codes/pika3.0/misc/dit_train_data/artificial_analysus_arena_samples/merged-till-20260917/arena-inputs.json`.

| Task | Stratum | Rows | Source and rule |
|---|---|---|---|
| t2va | `t2va_arena` | 450 | arena textToVideo, 600 left after the held-out rows; median 265 chars |
| | `t2va_real` | 500 | API-user t2v (726 + 1,329 + 2,664 usable); median ~800 chars |
| | `t2va_short` | 300 | synthesized asks under 80 chars, 30 % non-English, from the `nova03_gemini3p6f_avjoint_natural` captions of `h3_distillation/arena_balanced_20k_20260914` (step 2) |
| i2va | `i2va_arena` | 400 | arena imageToVideo, 580 left; median 118 chars; local PNGs |
| | `i2va_real` | 850 | API-user i2v jobs with one input image (639 + 2,551 + 1,984 usable) |
| fl2va | `fl2va_real` | 700 | API-user i2v jobs with two different input images (166 + 438 + 270 usable) |
| | `fl2va_derived` | 550 | API-user one-image i2v job: its input image as first frame, the **last frame of its output video** as last frame, its caption as text |
| l2va | `l2va_derived` | 1,250 | API-user t2v or i2v job: caption + the last frame of its output video. Rows disjoint from the strata above |

Rules for API-user rows:

- **Frame roles come from the job metadata, never from the file index.** MiniMax jobs store `in_0` = first,
  `in_1` = last (`request.first_frame_image` / `last_frame_image`); Seedance jobs store them reversed
  (`in_0` = `end_image_url`, `in_1` = `image_url`). Checked on 210 + 999 jobs and visually on 8.
- Drop captions already in H3 format (`integrated_multimodal_description:`), JSON or template captions, exact
  duplicates per model, captions over 7,000 chars, and jobs whose `request_duration` is not an integer in 4–15.
- Cap rows per `org_id` at 3 % of a stratum: one org wrote 92 % of the MiniMax t2v rows and 90 % of Seedance 2.0's.
- Stratify each real stratum by caption length (< 60, 60–200, 200–600, 600–1,500, > 1,500 chars) with at least
  15 % in each of the two shortest buckets where the source has them; keep the natural language mix.
- There are no native l2va jobs, and the fl2va 50k / arena-balanced frames exist only on S3, so l2va and part of
  fl2va are derived from generated outputs. Their text was not written with an end frame in mind; they carry
  `stratum: *_derived` so the share can be changed or dropped at export.

### 4.2 ref2va — 5,000 over all 43 categories

The taxonomy is Dan's `ref2va_v3`: groups A identity (A01–A05), B scene and composition (B01–B04), C action
(C01–C05), D camera (D01–D04), E style (E01–E04), F shot structure (F01–F04), G audio (G01–G07), H editing
(H01–H07), I control signal (I01–I03).

| Stratum | Rows | Source and rule |
|---|---|---|
| `ref_taxonomy` | 2,580 | `/mnt/nfs/dan/datas/minimax-H3/training_datas/ref2va_v3/source_manifest.jsonl` (5,600 cases, `primary_category`, ordered `references[]` with `kind`, `has_audio`, `media_path`). 60 per category; the nine categories with 50 cases (F04, G03, G05, G07, H02–H06) give what they have and the shortfall goes to the larger categories of the same group |
| `ref_real` | 2,273 | API-user `jsonls/r2v.jsonl` (22,714 real reference jobs; `references` = ordered `{path, kind}`), stratified by reference combination |
| `ref_arena` | 147 | arena videoToVideo rows left after the held-out 100; short real editing asks, 10 s videos |

`ref_taxonomy` rules:

- **User text is synthesized (step 2).** The set's own `user_prompt.txt` is not user language: it is written in H3
  labels ("Keep `<Subject 1>` recognizable …") and for H01–H07 it is the constant "Edit the source video as
  assigned." The ask is written from the case's expanded prompt, which is inline in the manifest (`prompt`).
- **Access.** Only C02–C05, D, E and H (19 categories, 2,150 cases) are readable by `cade`. A, B, C01, F and I case
  directories are `dan:dan` without group access, and G's `user_prompt.txt` and G05 audio likewise. Ask Dan to run
  `chgrp -R research` and `chmod -R g+rX` on `ref2va_v3/`. Do not start `ref_taxonomy` for a category until its
  media is readable.
- Exclude the 86 eval case dirs. 610 other cases share a reference asset with an eval case: mark them
  `shares_eval_asset` and sample them last.
- 2,946 of the 3,378 reference videos are declared `has_audio: false`: strip their audio stream in step 3, or
  Context-IR will label and may reuse a soundtrack the case does not intend (§2 fact 4).
- `target_duration_seconds` is set on 2,600 cases; round to an integer in 4–15. Sample the rest per §4.3.

`ref_real` rules:

- Same caption filters and `org_id` cap as §4.1. Keep rows within the count limits (665 Seedance 2.5 rows have more
  than 9 images) and with `request_duration` an integer in 4–15.
- **Drop rows whose media breaks a limit rather than trimming it**: the text refers to the whole clip. About 17 %
  of reference videos are too long (sample of 60).
- Sampling weights by combination, because image-only jobs are 80 % of traffic: images only 45 %, image + video
  25 %, video only 10 %, any audio 20 % (1,266 rows carry an audio reference; take all that pass).
- These rows have no category. Coverage is reported after labeling from the `summary` tags Context-IR assigns.
- The 968 cases of `ref2va_v1/user-real` are a subset of this file; do not add them separately.

Coverage target: every one of the 43 categories ends with at least 45 `ok` rows; every `summary` tag
(`reference generation`, `video editing`, `video continuation`, `audio reference`, `audio reuse`,
`keyframe completion`) appears.

### 4.3 Duration and ratio

- Real rows keep the job's `request_duration`. Ratio: the nearest of the six fixed ratios to the output size for
  t2va and ref2va; `adaptive` for the keyframe tasks.
- Rows without a duration (arena, synthesized): 5 s 45 %, 8 s 15 %, 10 s 25 %, 15 s 10 %, other integers in 4–14
  5 %. t2va ratio where unknown: 16:9 60 %, 9:16 25 %, 1:1 10 %, 4:3 / 3:4 / 21:9 5 %.

## 5. Procedure

### Step 0 — before spending

1. Dan regroups `ref2va_v3` (§4.2). The base tasks, `ref_real`, `ref_arena` and 19 taxonomy categories do not wait.
2. Confirm MiniMax's platform terms allow training a model on Context-IR outputs.
3. Top up the MiniMax balance for the budget in §6; the run aborts on HTTP 402.

### Step 1 — build the manifest (to build: `build_rewriter_manifest.py`)

One adapter per stratum writing the §1 schema to `$RUN/manifest.jsonl`, with the filters, caps and exclusions of §4,
a fixed seed, and **1.2× each quota** (the extra rows are the top-up reserve, marked `reserve: true`). Derived
strata extract the last frame with `ffmpeg -sseof -0.2 -i out.mp4 -update 1 -q:v 2 last.jpg` into `$RUN/media/`.
Write `$RUN/manifest_report.json`: rows per stratum, category, length bucket, language, org share.

### Step 2 — synthesize user asks (to build: `synth_user_asks.py`, Gemini)

Only for `t2va_short` and `ref_taxonomy`. Env `/mnt/nfs/cade/venvs/gemini-rewrite`, key
`pika3.0/env/.gemini_key`, same sidecar pattern as `rewrite_captions_h3_gemini.py`.

- `t2va_short`: caption → one user-style request under 80 chars, in a requested language and register.
- `ref_taxonomy`: expanded prompt + reference list → the request a user would type, 1–4 sentences, in one of three
  styles (`@Image 1` tags, plain "image 1", unnumbered "the video"). It must state the intent the category is about,
  quote any dialogue, and **number references per uploaded kind** — never copy `<Subject N>` or the label of a video
  soundtrack. Reject asks that contain `<` labels.

Quote the cost from a 50-row pilot of the final prompt, not from this document.

### Step 3 — normalize reference media (to build: `normalize_rewriter_media.py`)

Into `$RUN/media/`, content unchanged: video → MP4 H.264, audio stream removed when `has_audio` is false; audio →
WAV; images with a side over 5,760 px or outside aspect 0.4–2.5 → row dropped. Rewrite the manifest paths.

### Step 4 — check (free)

```bash
$PY $TOOL --mode check --input $RUN/manifest.jsonl
```

Exit 0 means every row is sendable. Fix or drop what it lists. Rows over the 64 MB body limit need a public or
presigned URL in `media[].url` instead of `path`.

### Step 5 — pilot (~350 rows, ~$15)

40 rows per base task, 3 per ref category, 60 `ref_real` rows, picked by id into `$RUN/pilot_ids.txt`.

```bash
$PY $TOOL --input $RUN/manifest.jsonl --ids-file $RUN/pilot_ids.txt \
    --output $RUN/labeled_pilot.jsonl --workers 8 --max-spend-usd 40
```

Then repeat on fresh ids at `--workers 16` and `--workers 30` to find where 429s start. Gates to pass before the
full run:

| Gate | Pass |
|---|---|
| First-attempt `ok` rate, per task | ≥ 80 % |
| `invalid` because of the token cap, per task | ≤ 15 % (else lower the share of that task's long-caption bucket) |
| `blocked` rate, per stratum | recorded; size the reserve from it |
| Cost per `ok` row, per task | within 1.5× the §2 range; update §6 |
| Throughput | a worker count with no sustained 429s |
| Manual read | 10 labels per base task and 2 per ref category: faithful to the request, media used as asked, mode read correctly |
| Render | 20 pilot labels through H3-Base (bucket config `configs/multires_h3/h3_480_768_1080_1440_buckets_v1.yaml`): no encoder rejection, output follows the label |

### Step 6 — full run

Run per task so a problem in one does not stall the rest; `ref2va` last. Same output file, so the sidecar resumes.

```bash
for T in t2va fl2va i2va l2va ref2va; do
  $PY $TOOL --input $RUN/manifest_primary.jsonl --tasks $T \
      --output $RUN/labeled.jsonl --workers <from pilot> --max-spend-usd <task budget, §6>
done
```

`manifest_primary.jsonl` is the manifest without the reserve rows. Exit 3 means the spend cap was reached: raise it
deliberately and re-run. Watch the `[label]` progress line (rows/min, spend, status counts) and
`labeled.spend.jsonl`. Never edit a sidecar; never run two drivers on the same output.

### Step 7 — retry and top up to quota

```bash
$PY $TOOL --input $RUN/manifest_primary.jsonl --output $RUN/labeled.jsonl --retry-failed --max-attempts 2 --max-spend-usd <cap>
```

Then, per stratum and category, take `quota − ok` rows from the reserve into `$RUN/topup_ids.txt` and label them
from the full manifest with `--ids-file`. Stop when every stratum is at quota or its reserve is empty; record
shortfalls in `$RUN/REPORT.md`.

### Step 8 — QA and export (to build: `export_rewriter_dataset.py`)

1. Report from `labeled.jsonl`: `ok` per task / stratum / category, token-length histogram, warning counts,
   `summary`-tag coverage for ref2va, cost per task.
2. Manual review of 30 rows per base task and 3 per ref category; drop rows with the warnings "quoted user text not
   found" or "supplied but never mentioned" unless the review clears them.
3. Export `ok` rows as chat samples: system = a short fixed instruction; user = the media in manifest order + one
   line `task / duration / ratio` + `user_text`; assistant = `ctxir_prompt`. Split 95 / 5 by `org_id` and by
   reference asset so no user or asset is on both sides.
4. Test references: label the three held-out sets' user inputs (287 rows, ~$12) into `$RUN/eval_refs.jsonl`; they
   are only ever used for evaluation.

## 6. Budget and time

First-pass projection from the §2 means ($0.015 / 0.038 / 0.027 / 0.037 / 0.051 per call):

| Part | Rows | First pass |
|---|---|---|
| t2va | 1,250 | ~$19 |
| i2va | 1,250 | ~$48 |
| fl2va | 1,250 | ~$34 |
| l2va | 1,250 | ~$46 |
| ref2va | 5,000 | ~$255 |
| **Total** | 10,000 | **~$400** |

Add redraws and top-up (assume 25 % until the pilot measures it), the pilot (~$15), eval references (~$12) and the
Gemini asks (unpriced until its pilot): **plan for $550, cap the MiniMax spend at $600.** Per-task
`--max-spend-usd`: t2va 30, i2va 70, fl2va 50, l2va 70, ref2va 350. These numbers come from 17 calls; replace them
with the pilot's.

Time: a call takes 25–110 s. At 30 in flight the 10k is roughly 5–7 hours of wall time; at 8 in flight about a
day. The real ceiling is whatever the pilot finds for 429s.

## 7. Troubleshooting

| Symptom | Cause and action |
|---|---|
| `error 2013 … audio format ".x-wav" not allowed` | wrong MIME; the driver maps `.wav` to `audio/wav`. Other 2013 errors name the offending field |
| many `invalid: N tokens > 940` on i2va / l2va | long image descriptions; the redraw usually fits. If the rate stays above 15 %, the cap can go to 960 only after an encoder check on real labels |
| `instruction line time X != frame grid` | the service changed its rounding; re-derive `api_frames` before accepting labels |
| `<Audio [n]> beyond the k supplied` | a video with an unintended soundtrack was not stripped (step 3) |
| exit 2, `HTTP 402` | balance empty; top up, re-run (resume is automatic) |
| exit 3 | spend cap reached; check the cost per row before raising it |
| sustained 429 | lower `--workers`; the driver already backs off 4–60 s per call |
| `blocked` | content filter; not retried. Counts toward the reserve, not toward cost as far as observed |
| label ignores a reference (warning) | common with terse asks; review, or drop the row |

## 8. Open decisions

1. Dan's regroup of `ref2va_v3` — without it 24 of 43 categories have no media and `ref_taxonomy` covers 19.
2. MiniMax terms on training with Context-IR outputs.
3. The 1,250-each split of the base tasks, the share of derived fl2va / l2va rows, and the §4.3 duration weights.
4. Content-filtered requests: accept the SFW skew, or label those rows with another model later.
5. Student input for audio references: Context-IR hears the audio; a vision-only Qwen cannot. Either an
   audio-capable student or audio captions in the user turn — decide before step 8's export format is fixed.
6. One label per input (this plan) or two draws on a subset to teach the spread.

## 9. Evidence

`$RUN/probes_20261002/`: 15 request/response pairs (`*.json`, `*_out*.json`), the reconstructed
`probe_manifest.jsonl`, and the driver's live run (`driver_smoke_*`). Total spend so far ≈ $0.55. Source audit
numbers in §4 were measured on 2026-10-02 against the paths given there.

## 10. Held-out 101-case retention audit (2026-10-06)

The official H3-Context-IR rewriter completed the same 101 original inputs on node21, using eight concurrent requests. Original prompt text, target duration and aspect ratio match the 9B v2 benchmark exactly; all 50 image inputs use the same original first-frame files. The official API received full-resolution originals, while 9B resized internally, so image preprocessing differs.

| Text review (101 cases per column) | FAL | 9B old prompt | 9B retention_v2 | 9B retention_v2 + GRPO | Official H3 |
|---|---:|---:|---:|---:|---:|
| Clear omission or constraint conflict | 12 | 18 | 18 | 27 | 14 |
| Deviation or added setting requiring review | 19 | 28 | 24 | 19 | 5 |
| No clear omission or conflict found | 70 | 55 | 59 | 55 | 82 |
| Clear issue rate | 11.9% | 17.8% | 17.8% | 26.7% | 13.9% |
| Clear issues plus review cases | 30.7% | 45.5% | 41.6% | 45.5% | 18.8% |

These are direct text-review case counts covering visual, sound-effect and music requirements, not independent human blind review or generated-video success. FAL and the two pre-GRPO 9B columns reuse their earlier complete text reviews; Codex directly inspected the completed GRPO and official outputs. Multiple issues in one case count once. Unrequested added music alone is not a clear issue under this shared criterion.

For this benchmark, retain the **first completed response** for every original input, including labels that fail the runbook's token/structure gates; do not redraw or truncate to select better outputs. This differs deliberately from training-label retry/top-up in §5. Official gate failures were **16/101**: 14 above 940 H3 prompt tokens (including 7 above 1024), plus 2 structure/time validation failures. All 16 remain in the 101-case retention denominator; gate failures and semantic failures can overlap and must not be added as disjoint counts.

Official usage at the runbook's reference token prices gives an estimated cost of **$2.1311**, not a confirmed invoice. The final GRPO checkpoint's direct review worsened to 27 clear issues from 18, so it is not recommended for promotion. [GRPO implementation and recovery instructions](https://github.com/Mellis-Labs/pika-llama-factory/tree/h3-rewriter/examples/h3_rewriter/grpo_20261005) were pushed in [ec25d69](https://github.com/Mellis-Labs/pika-llama-factory/commit/ec25d69537eb93a0f52e8f2bbc002b451898b6a8).

See the five-group review artifacts, including all 14 official clear cases and 5 review cases with evidence, at HB10 `official_h3_context_ir_101_20261006/report.md` and `full_retention_audit_101.json`. Private run artifacts are under node21 `/mnt/nfs/xiaotong/benchmark_outputs/h3_official_context_ir_original101_20261006`; this documentation commit includes no complete source/output dataset, images, credentials or model weights.


## 11. Official-template TRL SFT, GRPO and evaluation method (2026-10-06)

The unified [evaluation and training runbook](h3_prompt_rewriter_evaluation_training_runbook.md) defines source-retention review, severe/general/review error levels, all 13 mechanical t2va/i2va format checks, semantic music/dialogue compliance, evidence requirements, denominators and matched inference controls. Mechanical structure passing is not full semantic compliance or H3 encoder acceptance.

[Published TRL SFT and GRPO sources](../../.) contain the official-template fresh SFT collator/preflight/training/epoch evaluation, EP3 checkpoint-906 GRPO initialization, severity-aware concurrent GPT-6 Luna reward, local service supervision, checkpoint recovery and matched greedy 101-case evaluation. Training uses sampling; evaluation uses greedy. Dataset manifests, media, weights, credentials and runtime receipts stay external. The instructions preserve the actual HB10 directory layout and pinned Python environment.

The [four-epoch fresh TRL SFT review table](h3_prompt_rewriter_trl_sft_audit_20261006.md) uses its own same-template inference protocol. Its clear-error labels are historical direct Codex text reviews, not a retrospective severe/general reclassification, and are not interchangeable with the old LLaMA-Factory 18/101. The new EP3 GRPO result must be added only after its held-out review completes; no improvement is inferred from training reward alone.
