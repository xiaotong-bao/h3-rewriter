"""Keep frozen sources/targets/media; replace only the system rules with v2."""
import collections
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path('/work')
JOB = ROOT / 'trl_sft_official_v2_20261006'


def user_content(text, images, videos):
    media = {'image': iter(images), 'video': iter(videos)}
    content = []
    for part in re.split(r'(<image>|<video>)', text):
        if part in ('<image>', '<video>'):
            kind = part[1:-1]
            path = next(media[kind])
            assert pathlib.Path(path).is_file(), path
            content.append({'type': kind, kind: path})
        elif part:
            content.append({'type': 'text', 'text': part})
    assert all(next(v, None) is None for v in media.values())
    return content


def main():
    JOB.mkdir(exist_ok=True)
    source_rules = (ROOT / 'grpo_20261005/system_prompt.txt').read_text()
    assert hashlib.sha256(source_rules.encode()).hexdigest() == '2fdac332a23b7fd235c89b2c42caf476a2bc3c2bedd7e82c518bbc29aab6687e'
    rules = source_rules.strip()
    report = {'system_sha256': hashlib.sha256(rules.encode()).hexdigest(), 'splits': {}}
    all_ids = set()
    for split, expected in [('train', 9659), ('val', 200)]:
        rows = []
        kinds = collections.Counter()
        for line in (ROOT / f'lf_dataset/{split}.jsonl').open():
            old = json.loads(line)
            assert old['id'] not in all_ids
            all_ids.add(old['id'])
            assert len(old['conversations']) == 2
            user, answer = old['conversations']
            assert user['from'] == 'human' and answer['from'] == 'gpt'
            assert '<think>' not in answer['value'] and '</think>' not in answer['value']
            suffix = old['system'].split('Requested task:', 1)
            assert len(suffix) == 2
            system = rules + '\n\nRequested task:' + suffix[1]
            content = user_content(user['value'], old.get('images', []), old.get('videos', []))
            prompt = [{'role': 'system', 'content': [{'type': 'text', 'text': system}]},
                      {'role': 'user', 'content': content}]
            row = {'id': old['id'], 'prompt_json': json.dumps(prompt, ensure_ascii=False),
                   'answer': answer['value'], 'images': old.get('images', []),
                   'videos': old.get('videos', []), 'source_sha256': hashlib.sha256(line.encode()).hexdigest()}
            rows.append(row)
            kinds[f"images={len(row['images'])},videos={len(row['videos'])}"] += 1
        assert len(rows) == expected
        path = JOB / f'{split}.jsonl'
        assert not path.exists(), 'Preparation must not overwrite an existing dataset'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        report['splits'][split] = {'rows': len(rows), 'modalities': dict(kinds),
                                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (JOB / 'system_prompt.txt').write_text(rules + '\n')
    (JOB / 'data_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
