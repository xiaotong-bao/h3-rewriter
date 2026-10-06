# Qwen3.5-9B rewriter: official-template TRL SFT

Run on HB10, `/data/xiaotong/h3_rewriter_sft_20261002/trl_sft_official_v2_20261006`.
Container name: `xiaotong-trl-sft-official-v2-20261006`; host root is mounted at `/work`.

This is a fresh SFT from original Qwen3.5-9B, not a continuation of the old step604
adapter or the GRPO adapter. The frozen split remains 9,659 train / 200 validation.
The full retention v2 system replaces the previous system rules; request task,
duration, ratio, reference roles, user text, teacher answer and media are preserved.
The 101 benchmark cases are not added to training.

## Official format

The processor loads the model's unmodified `chat_template.jinja`, independently
verified against official Qwen revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
Template SHA256: `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`.

`enable_thinking=False` produces the official empty think prefix. The training
sequence uses the same official processor as inference. It masks all prompt
tokens, including empty think and vision tokens, and trains on the unchanged
teacher answer plus `<|im_end|>`. No replacement Jinja template or special
generation markers are injected by TRL. EOS is tokenizer EOS 248046.

The custom collator only adapts mixed text/image/video data and target masking to
TRL SFTTrainer; it does not implement or edit a chat template. Original media
placeholder positions and same-kind reference order are preserved. Video decoding
uses Transformers' preferred TorchCodec 0.7.0 backend, compatible with Torch 2.8.
TorchCodec is installed in this task's vendor directory, not the old environment;
the isolated task image adds FFmpeg shared libraries. No audio waveform is
provided, consistent with the original dataset.

Images: min/max pixels 3136/200704. Videos: 3136/50176, target fps 1, min 2 / max 8
frames, official processor sampling and timestamps. Complete input sequences are
validated against the 8192-token budget; no input or target is silently truncated.

## Training

- TRL SFTTrainer 1.14.1; Transformers 5.6.0; PEFT 0.18.1; Torch 2.8.0.
- 8 H200, BF16, SDPA; microbatch 1/GPU, accumulation 4, effective batch 32.
- Fresh language-only LoRA rank 64 / alpha 128 / dropout 0.05.
- Vision encoder and multimodal projector frozen.
- 4 epochs; LR 1e-4, cosine, warmup 18 updates, seed 42.
- Recovery checkpoints every 25 optimizer steps; all epoch checkpoints retained;
  fixed 200-case validation loss each epoch.
- Each saved epoch generates the original 101 held-out rewrites using the same
  official processor/template; outputs are under `epoch_benchmarks/stepN`.
  All four epochs were subsequently reviewed case by case by Codex; see the
  central runbook and its four-epoch audit summary for the historical labels.
- Expected approximately 1,208 optimizer steps; trainer state is authoritative.

The selected 4 epochs are the initial retraining configuration, not a claim that
four is best. Final checkpoint selection requires held-out generation review.

## Durable execution and checks

`run_pipeline.py`: eight-GPU three-step text/image/video smoke, full 9,859-row
preflight, then four-epoch SFT. Full preflight checks template/target prefix and
token mask per case, and independently compares generation-time prompt token IDs
and all visual tensors for each media-count combination on each worker. No
generation-time pixel hash claim is made for every case.

The container is network-isolated. It uses the already installed GRPO environment
without changing the old LLaMA-Factory or GRPO environment. A task-local TorchCodec
package and FFmpeg-equipped task image provide video decoding.
`environment.freeze.txt` records actual dependencies.

Processed tensors are cached after their first training use as safe tensor files.
Cache keys include the source row, official template, image/video limits and mask
version, preventing reuse of old LLaMA-Factory token caches. This avoids decoding
the same videos again every epoch. Cache equality is checked on actual text,
image and video examples before the full training starts.

Monitoring: `pipeline_status.json`, `progress.json`, `smoke.log`,
`preflight_all.log`, `training.log`, `preflight_report.json`,
`run/checkpoint-*/trainer_state.json`.

After interruption, inspect the last valid checkpoint and run `train.py --resume
/work/trl_sft_official_v2_20261006/run/checkpoint-N --epochs 4` with the same eight-GPU
torchrun command and environment. Do not rerun preparation over existing data or
reuse a historical tokenized cache.
