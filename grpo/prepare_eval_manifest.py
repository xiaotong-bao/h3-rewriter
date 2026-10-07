"""Prepare the frozen original101 Astra manifest from generated rewrites."""
import argparse
import json
from pathlib import Path
from watch_userlike_astra_eval import ROOT, manifest_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'benchmark_jobs/arena_original_101_retention_v2_20261005/data/inputs.jsonl')
    args = parser.parse_args()
    sources = {r['id']: r for r in map(json.loads, args.source.read_text().splitlines())}
    rows = [json.loads(s) for s in args.results.read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == 101
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(manifest_row(row, sources), ensure_ascii=False) + '\n' for row in rows))
    print('Manifest ready: 101 cases', args.output)


if __name__ == '__main__':
    main()
