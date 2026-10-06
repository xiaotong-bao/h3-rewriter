import json

import torch

from common import JOB, ROOT, OfficialCollator, encode, processor
from prepare import user_content


def main():
    torch.set_num_threads(1)
    p = processor()
    raw = [json.loads(line) for line in (JOB / 'train.jsonl').open()]
    selected = {}
    for row in raw:
        kind = 'video' if row['videos'] else 'image' if row['images'] else 'text'
        selected.setdefault(kind, row)
    collator = OfficialCollator(p, cache=True)
    for kind, row in selected.items():
        original = collator.inspect(row)[0]
        first = collator([row])
        cached = collator([row])
        assert set(original) == set(first) == set(cached)
        assert all(torch.equal(original[k], first[k]) and torch.equal(first[k], cached[k]) for k in original)
        print('CACHE_EQUAL', kind, row['id'], flush=True)
    regression = next(row for row in raw if row['id'] == 'F02_case00097')
    _, receipt = collator.inspect(regression, compare_prompt=True)
    print('VIDEO_DECODER_REGRESSION_PASSED', receipt, flush=True)
    benchmark = ROOT / 'benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl'
    rows = [json.loads(line) for line in benchmark.open()]
    assert len(rows) == len({r['id'] for r in rows}) == 101
    rules = (JOB / 'system_prompt.txt').read_text().strip()
    benchmark_sources = set()
    for row in rows:
        assert row['system'].startswith(rules)
        user = row['conversations'][0]['value']
        content = user_content(user, row.get('images', []), row.get('videos', []))
        messages = [{'role': 'system', 'content': [{'type': 'text', 'text': row['system']}]},
                    {'role': 'user', 'content': content}]
        prompt = p.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert prompt.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
        benchmark_sources.add(' '.join(user.replace('<image>', '').strip().split()))
    overlaps = []
    for row in raw:
        prompt = json.loads(row['prompt_json'])
        user_text = ''.join(x.get('text', '') for x in prompt[-1]['content'])
        if ' '.join(user_text.strip().split()) in benchmark_sources:
            overlaps.append(row['id'])
    assert not overlaps, ('Benchmark source text overlaps training', overlaps)
    result = {'passed': True, 'cache_modalities': sorted(selected), 'video_regression': receipt,
              'benchmark_rows': 101,
              'benchmark_training_source_overlaps': overlaps}
    (JOB / 'runtime_validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
