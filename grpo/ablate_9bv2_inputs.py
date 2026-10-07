"""Paired LF/HF x unmerged/merged inference of the frozen legacy SFT adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import peft
import torch
import torch.distributed as dist
import transformers
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

sys.path.insert(0, '/work/LLaMA-Factory/src')
from llamafactory.data.template import TEMPLATES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--templates', nargs='+', choices=('LF', 'HF'), default=['LF', 'HF'])
    parser.add_argument('--merges', nargs='+', choices=('unmerged', 'merged'), default=['unmerged', 'merged'])
    parser.add_argument('--label-suffix', default='current_env')
    parser.add_argument('--autocast', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=Path('/work/runs/llamafactory_sft_v2/checkpoint-604'))
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    if world > 1:
        dist.init_process_group('nccl')
    root = Path('/work')
    source = root / 'benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl'
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == 101
    args.output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(root / 'models/Qwen3.5-9B')
    processor.image_processor.size = {'shortest_edge': 3136, 'longest_edge': 200704}
    processor.image_max_pixels = 200704
    processor.image_min_pixels = 3136
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        root / 'models/Qwen3.5-9B', dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map={'': torch.cuda.current_device()})
    model = PeftModel.from_pretrained(base, args.checkpoint).eval()
    path = args.output / f'rank{rank}.jsonl'
    completed = {(r['id'], r['label']) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    for merge in args.merges:
        if merge == 'merged':
            model = model.merge_and_unload().eval()
        for row in rows[rank::world]:
            raw = row['conversations'][0]['value']
            for template_name in args.templates:
                label = f'9Bv2_{template_name}_{merge}_{args.label_suffix}'
                if (row['id'], label) in completed:
                    continue
                if template_name == 'LF':
                    template = TEMPLATES['qwen3_5_nothink']
                    messages = [{'role': 'user', 'content': raw}, {'role': 'assistant', 'content': ''}]
                    messages = template.mm_plugin.process_messages(messages, row.get('images', []), [], [], processor)
                    ids, _ = template.encode_oneturn(processor.tokenizer, messages, system=row['system'])
                    image_inputs = template.mm_plugin._get_mm_inputs(row.get('images', []), [], [], processor)
                    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in image_inputs.items()}
                    inputs['input_ids'] = torch.tensor([ids], device=model.device)
                    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
                else:
                    import re
                    media = iter(row.get('images', []))
                    content = []
                    for part in re.split(r'(<image>)', raw):
                        if part == '<image>':
                            content.append({'type': 'image', 'image': next(media)})
                        elif part:
                            content.append({'type': 'text', 'text': part})
                    assert next(media, None) is None
                    inputs = processor.apply_chat_template([
                        {'role': 'system', 'content': [{'type': 'text', 'text': row['system']}]},
                        {'role': 'user', 'content': content}], tokenize=True,
                        add_generation_prompt=True, enable_thinking=False,
                        return_dict=True, return_tensors='pt').to(model.device)
                input_ids = inputs['input_ids'][0].tolist()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.autocast):
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=2048,
                        repetition_penalty=1., eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id, use_cache=True)
                tokens = output[0, len(input_ids):].tolist()
                record = dict(id=row['id'], label=label, original_prompt=raw.replace('<image>', '').strip(),
                    rewrite=processor.decode(tokens, skip_special_tokens=True).strip(),
                    generated_tokens=len(tokens), at_token_cap=len(tokens) >= 2048,
                    last_token_id=tokens[-1], eos_token_id=processor.tokenizer.eos_token_id,
                    input_tokens=len(input_ids), input_ids_sha256=hashlib.sha256(json.dumps(input_ids).encode()).hexdigest(),
                    input_tail=processor.tokenizer.decode(input_ids[-20:]),
                    image_grid=inputs['image_grid_thw'].tolist() if 'image_grid_thw' in inputs else None,
                    pixels_sha256=hashlib.sha256(inputs['pixel_values'].cpu().numpy().tobytes()).hexdigest() if 'pixel_values' in inputs else None)
                with path.open('a') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                print('GENERATED', label, row['id'], len(tokens), flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        results = [json.loads(s) for i in range(world) for s in (args.output / f'rank{i}.jsonl').read_text().splitlines()]
        count = 101 * len(args.merges) * len(args.templates)
        assert len(results) == len({(r['id'], r['label']) for r in results}) == count
        (args.output / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sorted(results, key=lambda r: (r['id'], r['label']))))
        (args.output / 'environment.json').write_text(json.dumps(dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
            template_sha256=hashlib.sha256((root / 'models/Qwen3.5-9B/chat_template.jinja').read_bytes()).hexdigest(),
            checkpoint=str(args.checkpoint),
            adapter_sha256=hashlib.sha256((args.checkpoint / 'adapter_model.safetensors').read_bytes()).hexdigest(),
            generation=dict(do_sample=False, max_new_tokens=2048, repetition_penalty=1., eos=processor.tokenizer.eos_token_id, autocast=args.autocast)), indent=2))
        (args.output / 'COMPLETE').write_text(str(count) + '\n')
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
