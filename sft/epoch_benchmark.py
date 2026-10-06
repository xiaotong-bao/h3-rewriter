"""Generate all 101 held-out rewrites from each saved epoch with official inputs."""
import json
import os
import time

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from common import JOB, ROOT, encode
from prepare import user_content


class EpochBenchmark(TrainerCallback):
    def __init__(self, p):
        self.p = p

    def on_save(self, args, state, control, model=None, **kwargs):
        # Recovery checkpoints are every 25 steps; full benchmark only at epochs.
        if abs(state.epoch - round(state.epoch)) > 1e-6:
            return
        rank = int(os.environ.get('RANK', 0))
        world = int(os.environ.get('WORLD_SIZE', 1))
        data = ROOT / 'benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl'
        rows = [json.loads(line) for line in data.open()]
        assert len(rows) == len({r['id'] for r in rows}) == 101
        rules = (JOB / 'system_prompt.txt').read_text().strip()
        output = JOB / 'epoch_benchmarks' / f'step{state.global_step}'
        output.mkdir(parents=True, exist_ok=True)
        # Trainer may save the last epoch twice. Reuse only a fully validated
        # benchmark; preserve the existing results rather than overwriting them.
        if (output / 'COMPLETE').is_file():
            existing = [json.loads(line) for line in (output / 'results.jsonl').open()]
            assert len(existing) == 101
            assert {r['id'] for r in existing} == {r['id'] for r in rows}
            assert all(r['step'] == state.global_step and
                       abs(r['epoch'] - state.epoch) < 1e-6 for r in existing)
            return
        path = output / f'rank{rank}.jsonl'
        assert not path.exists(), 'Never overwrite an epoch benchmark'
        if rank == 0:
            (JOB / 'epoch_benchmark_status.json').write_text(json.dumps({
                'phase': 'generating', 'epoch': state.epoch, 'step': state.global_step,
                'started_at': time.time(), 'output': str(output)}, indent=2))
        if dist.is_initialized():
            dist.barrier()
        was_training = model.training
        model.eval()
        with path.open('w') as f, torch.inference_mode():
            for row in rows[rank::world]:
                assert row['system'].startswith(rules)
                original = row['conversations'][0]['value']
                messages = [{'role': 'system', 'content': [{'type': 'text', 'text': row['system']}]},
                    {'role': 'user', 'content': user_content(original, row.get('images', []), row.get('videos', []))}]
                inputs = encode(self.p, messages, generation=True).to(model.device)
                inputs.pop('video_metadata', None)
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=2048,
                    repetition_penalty=1.0, eos_token_id=self.p.tokenizer.eos_token_id,
                    pad_token_id=self.p.tokenizer.pad_token_id, use_cache=True)
                ids = generated[0, inputs['input_ids'].shape[1]:]
                result = {'id': row['id'], 'epoch': state.epoch, 'step': state.global_step,
                    'checkpoint': f'checkpoint-{state.global_step}',
                    'original_prompt': original.replace('<image>', '').strip(),
                    'rewrite': self.p.decode(ids, skip_special_tokens=True).strip(),
                    'generated_tokens': len(ids), 'at_token_cap': len(ids) >= 2048,
                    'last_token_id': int(ids[-1]), 'system_prompt_variant': 'retention_v2',
                    'protocol': 'Unmodified official Qwen template, empty think prefix, original media ordering'}
                f.write(json.dumps(result, ensure_ascii=False) + '\n'); f.flush()
                print('EPOCH_BENCHMARK', state.global_step, rank, row['id'], len(ids), flush=True)
        model.train(was_training)
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            results = [json.loads(line) for i in range(world) for line in (output / f'rank{i}.jsonl').open()]
            assert len(results) == len({r['id'] for r in results}) == 101
            (output / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
                for r in sorted(results, key=lambda r: r['id'])))
            (output / 'summary.json').write_text(json.dumps({'epoch': state.epoch, 'step': state.global_step,
                'rows': 101, 'token_cap_hits': sum(r['at_token_cap'] for r in results),
                'text_audit_status': 'pending_case_by_case_review'}, indent=2))
            (output / 'COMPLETE').write_text('101\n')
            (JOB / 'epoch_benchmark_status.json').write_text(json.dumps({
                'phase': 'complete', 'epoch': state.epoch, 'step': state.global_step,
                'completed_at': time.time(), 'output': str(output)}, indent=2))
