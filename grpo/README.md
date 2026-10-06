# EP3 GRPO with GPT-6 Luna reward

Snapshot of `grpo_ep3_luna_v2_20261006` on HB10. Start from fresh official-template
TRL SFT EP3 `checkpoint-906`, not old LLaMA-Factory step604 or an old GRPO adapter.

`prepare_merge.py` selects 128 t2va + 128 i2va training inputs, excluding videos,
validation/101 prompt hashes and validation image overlaps. It keeps the original
SFT `prompt_json` ordering, merges EP3 into the frozen base, writes the processor
and checks the official template/EOS plus a next-token argmax comparison. The
merged reference may differ numerically from unmerged EP3; `evaluate.py` therefore
includes both the historical EP3 and a matched merged-zero-GRPO control.

`train_grpo.py`: 8 ranks, microbatch 1, accumulation 2, 8 candidates/input,
64 steps, LR 5e-6, language-only new LoRA 32/64, beta .02, DAPO loss,
`temperature=.8`, `top_p=.95`, maximum completion 2048, official non-thinking
template, EOS 248046/pad 248044. Each step uses two prompt presentations and
16 candidates; the 64-step pilot is 128 presentations, not a full pass over 256.

`luna_judge.py`: `gpt-6-luna`, reasoning `high`, revision
`ep3-v2-luna-severity-v3`. One grounded structured review per candidate, model calls
through ephemeral Codex CLI with Apps, shell, web and multi-agent tools disabled.
The locally logged-in Codex CLI keeps authentication on HB10. Training uses local
loopback 8792, configurable through H3_LUNA_PORT; no SSH forward is required.
Eight training ranks submit concurrently; each rank uses
up to four request threads, and the gateway caps concurrent CLI processes at 16.
With current microbatch 1, active requests need not reach that cap.

Reward: `1 - .8*min(severe,2) - .1*min(general,3) - .02*min(review,3)`;
mechanical format failure subtracts .3. Failures are retried three times and then
receive an operational score of -1 (and another -.3 if format fails), explicitly
marked `judge_failed`; no false semantic verdict is created.

`start_luna_runtime.py` starts the Linux supervisor. `runtime_watchdog.py`
checks local health and restores only gateway children it owns. `run_pipeline.py`
checks the local Luna model/revision before starting and
executes a two-step smoke, then the 64-step pilot, then matched greedy 101-case
inference and Luna exploratory review. It resumes the most recent valid saved
checkpoint on a restart, and fails visibly for non-judge training errors.
The user's explicit decision to start directly waived the complete 256-pair
audit gate; these receipts must be recreated locally, never copied as credentials.

`generate_256.py`, `compare_256.py`, `report_comparison.py` implement the optional
independent Luna/Sol severity comparison. They are not the training reward
implementation and were paused when direct GRPO was requested. The old
`runtime_controller.py`/minimum-retention calibration workflow is intentionally
not the entry point of this severity-aware run.

See the central runbook for installation, source layout, health checks, commands,
checkpoint recovery and evaluation denominators. Completed training must still
be judged using held-out outputs; a rising training reward alone is insufficient.
