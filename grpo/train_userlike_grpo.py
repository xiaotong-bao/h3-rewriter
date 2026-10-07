"""Continue the legacy SFT LoRA directly, with single-pass Luna reward."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, TrainerCallback
from trl import GRPOConfig

import train_grpo as previous
from single_pass_reward import REVISION, score_record
from existing_lora import freeze_reference

ROOT = Path('/work') / os.environ.get('H3_GRPO_JOB', 'grpo_9bv2_userlike_1k_20261007')
PORT = int(os.environ.get('H3_LUNA_PORT', '8799'))
previous.ROOT = ROOT


def call(payload):
    errors = []
    for attempt in range(3):
        try:
            request = urllib.request.Request(f'http://127.0.0.1:{PORT}/score',
                json.dumps(payload).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=1800) as response:
                result = json.load(response)
            assert result['judge_revision'] == REVISION and result['judge_model'] == 'gpt-6-luna'
            assert result['reasoning_effort'] == 'high'
            for item in result['coverage']:
                assert item['source_quote'] in payload['original']
            for issue in result['issues']:
                assert issue['source_quote'] in payload['original']
                assert not issue['rewrite_quote'] or issue['rewrite_quote'] in payload['rewrite']
                assert issue['status'] == 'omitted' or issue['rewrite_quote'].strip()
            scored = score_record(result)
            assert math.isclose(scored['reward'], result['reward'], abs_tol=1e-8)
            assert scored['positive_eligible'] == result['positive_eligible']
            result.update(scored)
            return result
        except Exception as error:
            errors.append(error.read().decode(errors='replace') if isinstance(error, urllib.error.HTTPError) else repr(error))
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    return dict(reward=None, positive_eligible=False, judge_failed=True, errors=errors)


def reward(completions, original, context, task, trainer_state, completion_ids, **kwargs):
    payloads = [dict(original=source, context=ctx, rewrite=previous.completion_text(text))
               for source, ctx, text in zip(original, context, completions, strict=True)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verdicts = list(pool.map(call, payloads))
    valid = sum(not v['judge_failed'] for v in verdicts)
    if torch.distributed.is_initialized():
        count = torch.tensor(valid, device=torch.cuda.current_device())
        torch.distributed.all_reduce(count)
        valid = count.item()
    assert valid > 0, 'Reward service unavailable on all ranks: stop before optimizer updates; resume from saved checkpoint'
    previous._LATEST_RECORDS = []
    for index, (payload, verdict) in enumerate(zip(payloads, verdicts, strict=True)):
        previous._LATEST_RECORDS.append(dict(eligible=verdict['positive_eligible'],
            judge_failed=verdict['judge_failed'], token_ids=completion_ids[index],
            original=payload['original'], reward=verdict['reward'], format_valid=verdict.get('format_valid', False)))
        with (ROOT / f'rollouts.rank{os.environ.get("RANK", "0")}.jsonl').open('a') as stream:
            stream.write(json.dumps({**payload, **verdict, 'step': trainer_state.global_step}, ensure_ascii=False) + '\n')
    return [v['reward'] for v in verdicts]


def adapter_hash(model, name):
    digest = hashlib.sha256()
    for key, parameter in sorted(model.named_parameters()):
        if f'.{name}.' in key:
            digest.update(key.replace(f'.{name}.', '.ADAPTER.').encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--output-name')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--save-steps', type=int, default=16)
    parser.add_argument('--full-preflight', action='store_true')
    parser.add_argument('--inputs-file', default='pilot_inputs.jsonl')
    args = parser.parse_args()
    authorization = json.loads((ROOT / 'single_pass_authorization.json').read_text())
    assert authorization['approved'] and authorization['revision'] == REVISION
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/', timeout=5) as response:
        health = json.load(response)
    assert health['revision'] == REVISION and health['model'] == 'gpt-6-luna'
    calibration = json.loads((ROOT / 'single_pass_calibration.json').read_text())
    assert calibration['passed'] and calibration['revision'] == REVISION
    rows = [json.loads(s) for s in (ROOT / args.inputs_file).read_text().splitlines()]
    if args.smoke:
        rows = [max((r for r in rows if r['task'] == task), key=lambda r: len(r['original']))
                for task in ('t2va', 'i2va')]
    dataset = Dataset.from_list(rows).with_transform(previous.clean_batch)
    processor = AutoProcessor.from_pretrained('/work/models/Qwen3.5-9B')
    processor.image_processor.size = {'shortest_edge': 3136, 'longest_edge': 200704}
    processor.video_processor.size = {'shortest_edge': 3136, 'longest_edge': 50176}
    processor.video_processor.fps = 1.0
    processor.video_processor.min_frames = 2
    processor.video_processor.max_frames = 8
    check_rows = rows
    if args.resume and not args.full_preflight:
        groups = {}
        for row in rows:
            key = (row['task'], sum(b['type'] == 'image' for b in row['prompt'][1]['content']), sum(b['type'] == 'video' for b in row['prompt'][1]['content']))
            if key not in groups or len(row['original']) > len(groups[key]['original']):
                groups[key] = row
        check_rows = list(groups.values())
    for row in check_rows:
        ids = processor.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True)['input_ids']
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        assert len(ids) <= 16384
    base = Qwen3_5ForConditionalGeneration.from_pretrained('/work/models/Qwen3.5-9B',
        dtype=torch.bfloat16, attn_implementation='sdpa')
    policy_checkpoint = args.resume or Path('/work/runs/llamafactory_sft_v2/checkpoint-604')
    model = PeftModel.from_pretrained(base, policy_checkpoint, is_trainable=True)
    model.enable_input_require_grads()
    model.generation_config.eos_token_id = processor.tokenizer.eos_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    initial_hash = adapter_hash(model, 'default')
    output = ROOT / (args.output_name or ('smoke' if args.smoke else 'pilot'))
    assert not output.exists(), f'Never overwrite a run: {output}'
    config = GRPOConfig(output_dir=str(output), learning_rate=args.learning_rate,
        max_steps=2 if args.smoke else args.steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=2, num_generations=8, max_completion_length=2048,
        beta=.02, temperature=.8, top_p=.95, loss_type='dapo', scale_rewards='batch',
        mask_truncated_completions=True, disable_dropout=True, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, ddp_find_unused_parameters=False,
        remove_unused_columns=False, logging_steps=1, save_steps=args.save_steps, save_total_limit=None,
        ignore_data_skip=bool(args.resume),
        report_to='none', warmup_steps=2, seed=42, data_seed=42,
        chat_template_kwargs={'enable_thinking': False},
        generation_kwargs=dict(use_cache=True, eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id), log_completions=False, log_multimodal=False)
    trainer_class = previous.QualityGatedGRPOTrainer
    if args.resume:
        from video_grpo import install_video_support, video_trainer_class
        install_video_support()
        trainer_class = video_trainer_class(trainer_class)
        class StrictTrainer(trainer_class):
            def _load_rng_state(self, checkpoint):
                super()._load_rng_state(checkpoint)
                import random
                import numpy as np
                rank = self.args.process_index
                saved = torch.load(Path(checkpoint) / f'rng_state_{rank}.pth', map_location='cpu', weights_only=False)
                assert torch.equal(torch.get_rng_state(), saved['cpu'])
                assert random.getstate() == saved['python']
                current_numpy = np.random.get_state()
                assert current_numpy[0] == saved['numpy'][0] and np.array_equal(current_numpy[1], saved['numpy'][1])
                assert current_numpy[2:] == saved['numpy'][2:]
                cuda = saved['cuda']
                expected_cuda = cuda[torch.cuda.current_device()] if isinstance(cuda, list) else cuda
                assert torch.equal(torch.cuda.get_rng_state(), expected_cuda)
                (Path(self.args.output_dir) / f'rng_restore.rank{rank}.json').write_text(json.dumps(dict(rank=rank, cpu=True, cuda=True, python=True, numpy=True)) + '\n')
            def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
                super()._load_from_checkpoint(resume_from_checkpoint, model)
                # Transformers sees ref/ and otherwise skips the root default adapter.
                self.model.load_adapter(resume_from_checkpoint, 'default', is_trainable=True)
                self.model.set_adapter('default')
                for name, parameter in self.model.named_parameters():
                    if '.ref.' in name:
                        parameter.requires_grad_(False)
        trainer_class = StrictTrainer
    trainer = trainer_class(model=model, args=config, processing_class=processor,
        train_dataset=dataset, reward_funcs=reward, callbacks=[previous.Status()])
    assert 'ref' in model.peft_config, 'KL must use the frozen SFT adapter, not the bare base'
    assert adapter_hash(model, 'default') == initial_hash, 'Trainer changed the initial SFT policy'
    # PEFT add_adapter can initialize the reference in BF16 while the loaded
    # pretrained policy LoRA is FP32. Preserve exact initial values and dtype.
    if args.resume:
        # Allocate ref in the exact policy dtype before loading saved FP32 weights;
        # loading into freshly created BF16 modules would round them first.
        freeze_reference(model)
        model.load_adapter(args.resume / 'ref', 'ref', is_trainable=False)
        model.set_adapter('default')
        expected_reference = json.loads((args.resume.parent / 'initialization_receipt.json').read_text())['reference_hash']
        assert adapter_hash(model, 'ref') == expected_reference, 'Resume reference differs from original frozen SFT'
    else:
        freeze_reference(model)
        assert adapter_hash(model, 'ref') == initial_hash
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all('.default.' in name and 'language_model' in name for name in trainable)
    reference_hash = adapter_hash(model, 'ref')
    from trl.trainer.utils import use_adapter
    model.eval()
    probe = processor.apply_chat_template(rows[0]['prompt'], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
        return_dict=True, return_tensors='pt').to(trainer.accelerator.device)
    probe.pop('video_metadata', None)
    with torch.inference_mode():
        policy_logits = model(**probe, logits_to_keep=1).logits.float().cpu()
        with use_adapter(model, 'ref'):
            reference_logits = model(**probe, logits_to_keep=1).logits.float().cpu()
    initial_logits_difference = (policy_logits - reference_logits).abs().max().item()
    if not args.resume:
        assert initial_logits_difference == 0., 'Initial policy/reference logits differ'
    if args.resume:
        from strict_resume import ResumeSchedule
        trainer.add_callback(ResumeSchedule(trainer, args.resume, args.learning_rate, initial_hash, reference_hash))
    model.train()
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    assert adapter_hash(model, 'ref') == reference_hash, 'Reference adapter changed'
    trainer.save_model(str(output / 'final_adapter'))
    if trainer.is_world_process_zero():
        processor.save_pretrained(output / 'final_adapter')
        (output / 'initialization_receipt.json').write_text(json.dumps(dict(
            checkpoint=str(policy_checkpoint), merge_used=False,
            train_existing_sft_adapter=True, r=model.peft_config['default'].r,
            alpha=model.peft_config['default'].lora_alpha, initial_adapter_hash=initial_hash,
            final_adapter_hash=adapter_hash(model, 'default'), reference_hash=reference_hash,
            reference_unchanged=True, dropout_disabled=True,
            initial_logits_max_difference=initial_logits_difference,
            revision=REVISION, steps=trainer.state.global_step), indent=2))
        (output / 'COMPLETE').write_text(str(trainer.state.global_step) + '\n')


if __name__ == '__main__':
    main()
