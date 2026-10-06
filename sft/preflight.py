"""Validate every source and target, plus independent multimodal prompt equality."""
import argparse
import collections
import hashlib
import json
import os
import pathlib
import time

import torch

from common import JOB, OfficialCollator, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    a = parser.parse_args()
    torch.set_num_threads(1)
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    p = processor()
    collator = OfficialCollator(p)
    rows = [json.loads(line) for split in ['train', 'val'] for line in (JOB / f'{split}.jsonl').open()]
    if a.limit:
        selected = {}
        for row in rows:
            selected.setdefault((len(row['images']), len(row['videos'])), row)
        rows = list(selected.values())[:a.limit]
    start = time.time()
    seen = set()
    output = JOB / f'preflight{a.limit}.rank{rank}.jsonl'
    assert not output.exists()
    with output.open('w') as f:
        for i, row in enumerate(rows[rank::world]):
            kind = (len(row['images']), len(row['videos']))
            independently_checked = kind not in seen
            _, receipt = collator.inspect(row, compare_prompt=independently_checked)
            seen.add(kind)
            receipt['independent_prompt_check'] = independently_checked
            f.write(json.dumps(receipt) + '\n'); f.flush()
            if i % 50 == 0:
                print(rank, i, receipt, 'elapsed', round(time.time() - start), flush=True)
    print('PREFLIGHT_RANK_COMPLETE', rank, len(rows[rank::world]), flush=True)


if __name__ == '__main__':
    main()
