"""Export frozen H3 splits to the original ShareGPT input consumed by TRL prepare."""
import hashlib, json, pathlib

ROOT = pathlib.Path('/work')
OUT = ROOT / 'lf_dataset'
OUT.mkdir(exist_ok=True)
def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
requests = {r['sample_id']: r for r in read(ROOT / 'source/manifest_api.jsonl')}
system = (ROOT / 'dataset/system_prompt.txt').read_text()

def convert(row):
    refs = row.get('references', [])
    videos = [m for m in requests[row['id']].get('media', []) if m['type'] == 'video']
    assert len(videos) == sum(r['kind'] == 'video' for r in refs)
    context = [f"Requested task: {row['task']}; aspect ratio: {row['ratio']}; duration: {row['duration']}s."]
    counts, assets, markers = {}, {'images': [], 'videos': []}, []
    for ref in refs:
        kind = ref['kind']; counts[kind] = counts.get(kind, 0) + 1
        label = ref.get('h3_label') or '<' + {'image':'Picture','video':'Video','audio':'Audio'}[kind] + f" {counts[kind]}>"
        extra = ''
        if kind == 'video':
            extra = f"; H3 audio slot=<Audio {counts[kind]}>; source audio enabled={bool(videos[counts[kind]-1].get('has_audio', False))}"
        if kind == 'audio':
            extra = f"; standalone audio reference {counts[kind]}; official H3 audio label=<Audio {len(videos)+counts[kind]}>"
        context.append(f"{label}: kind={kind}; role={ref.get('role') or 'reference'}{extra}.")
        if kind in ('image', 'video'):
            source = ref['media_path']
            path = ROOT / 'media' / (hashlib.sha256(source.encode()).hexdigest() + pathlib.Path(source).suffix)
            assert path.is_file(), str(path)
            assets['images' if kind == 'image' else 'videos'].append(str(path))
            markers.append('<image>' if kind == 'image' else '<video>')
    if any(r['kind']=='audio' for r in refs):
        context.append('Audio reference content is not provided to this VLM. Follow the user-requested audio relationship without guessing a transcript or unprovided sound details.')
    user = row['user_prompt'] + (('\n' + ''.join(markers)) if markers else '')
    assert user.removesuffix('\n' + ''.join(markers)) == row['user_prompt'] if markers else user == row['user_prompt']
    return dict(id=row['id'], system=system+'\n\n'+'\n'.join(context),
                conversations=[{'from':'human','value':user},{'from':'gpt','value':row['target']}], **assets)

splits = {'train': read(ROOT/'dataset/train_partial.jsonl'), 'val': read(ROOT/'dataset/val.jsonl')}
info, report = {}, {}
for name, rows in splits.items():
    converted = [convert(r) for r in rows]
    if name == 'train':
        # Arrow infers JSONL column types from its first block. Seed both
        # visual list columns before the long initial text-only segment.
        seeds = []
        for column in ('images','videos'):
            index = next(i for i,r in enumerate(converted) if r[column])
            if index not in seeds: seeds.append(index)
        converted = [converted[i] for i in seeds] + [r for i,r in enumerate(converted) if i not in seeds]
    assert len(converted) == len(rows)
    path = OUT / (name+'.jsonl')
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in converted))
    report[name] = {'rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'source_sha256':hashlib.sha256((ROOT/('dataset/train_partial.jsonl' if name=='train' else 'dataset/val.jsonl')).read_bytes()).hexdigest()}
    info['h3_'+name] = {'file_name':path.name,'formatting':'sharegpt',
        'columns':{'messages':'conversations','system':'system','images':'images','videos':'videos'},
        'tags':{'role_tag':'from','content_tag':'value','user_tag':'human','assistant_tag':'gpt'}}
smoke_ids = set(); smoke=[]
for row in splits['train']:
    key = row['mode']
    if key not in smoke_ids or row['id'] in ('A03_case00202','A03_case00051','A03_case00053','C02_case00249'):
        smoke.append(convert(row)); smoke_ids.add(key)
(OUT/'smoke.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in smoke))
info['h3_smoke'] = dict(info['h3_train'], file_name='smoke.jsonl')
(OUT/'dataset_info.json').write_text(json.dumps(info, indent=2))
(OUT/'conversion_report.json').write_text(json.dumps(report, indent=2))
print(json.dumps({"conversion": report, "smoke_rows": len(smoke)}, indent=2))
