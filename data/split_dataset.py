import argparse, collections, hashlib, json, pathlib, random

ROOT = pathlib.Path(__file__).resolve().parent
SYSTEM = '''You are a professional H3 prompt rewriter for joint audio-video generation. Convert the user's short instruction and supplied references into a coherent, production-ready official H3 rewrite for the requested task, duration and aspect ratio. Return only the rewrite, without Markdown fences, commentary or reasoning.
Preserve intent, reference identities, appearance, spatial relationships and temporal continuity. Treat first/last images as the requested boundary frames; distinguish reference images, videos and audio by their labels and roles. Never invent reference content that is unavailable. Add concrete visual action, camera movement, lighting and composition consistent with the supplied evidence. Keep pacing appropriate to the requested duration and composition appropriate to the requested aspect ratio.
For t2va, i2va, l2va and fl2va, use integrated_multimodal_description, overall_soundscape and non_diegetic_music in that order, preserving the official boundary-frame reference instructions when applicable. For ref2va, use subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape and non_diegetic_music in that order. Keep reference labels and subject identities stable. Distinguish fully_preserved, partially_preserved, attribute_transfer and weak_reference for visual relationships; distinguish fully_copy, partially_copy, reference and weak_reference for audio relationships.
Use numbered shots [Shot 1], [Shot 2], etc.; introduce subsequent cuts with At MM:SS.mmm when appropriate. Write the description in English, retaining original-language dialogue, lyrics and visible text. Mark dialogue with <d>[Language]...</d> and assign S1, S2, etc. in speaking order. Describe synchronized physical and ambient sound separately from external background music. Transcribe audio only when the user's request requires its spoken or lyrical content; otherwise use the supplied sound characteristics, rhythm, timbre, mood and ambient descriptions, without adding a transcript. Do not fabricate speech, lyrics or speakers. Preserve the user-requested scene ending and any cross-cut continuity.'''

def read(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]

def keys(row):
    result = ['prompt:' + hashlib.sha256(row['user_prompt'].strip().encode()).hexdigest()]
    for r in row.get('references', []):
        for v in (r.get('sha256'),r.get('media_path') or r.get('path') or r.get('url')):
            if v: result.append('media:' + str(v))
    return result

def dump(path, rows):
    with open(path, 'w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')

def convert(r, manifests):
    m = manifests.get(r['id'])
    if m is None: raise ValueError('Missing request manifest: ' + r['id'])
    refs = r.get('references', [])
    # Never expose source_h3_prompt, h3_prompt or generated video_path as inputs.
    public = [{k: v for k, v in x.items() if k in ('kind','media_path','role','h3_label','description','has_audio','duration_seconds')} for x in refs]
    return dict(id=r['id'], task=r['task'], mode=r['mode'], user_prompt=r['user_prompt'],
                duration=m.get('duration'), ratio=m.get('ratio'), references=public,
                target=r['h3_prompt'], rewrite_meta=r.get('rewrite_meta', {}),
                leakage_keys=keys(r))

def main():
    a = argparse.ArgumentParser(); a.add_argument('--source', required=True); a.add_argument('--manifest', required=True)
    a.add_argument('--out', required=True); a.add_argument('--frozen-val', default=None); args=a.parse_args()
    out=pathlib.Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=read(args.source); manifests={r['sample_id']: r for r in read(args.manifest)}
    good=[r for r in rows if r.get('rewrite_status')=='succeeded' and r.get('user_prompt','').strip() and r.get('h3_prompt','').strip()]
    assert len({r['id'] for r in good}) == len(good)
    if args.frozen_val:
        val=read(args.frozen_val); ids={r['id'] for r in val}; blocked={k for r in val for k in r['leakage_keys']+keys(r)}
        train=[convert(r,manifests) for r in good if r['id'] not in ids and not blocked.intersection(keys(r))]
        excluded=[r['id'] for r in good if r['id'] not in ids and blocked.intersection(keys(r))]
    else:
        parent=list(range(len(good)))
        def find(i):
            while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
            return i
        seen={}
        for i,r in enumerate(good):
            for k in keys(r):
                if k in seen: parent[find(i)]=find(seen[k])
                else: seen[k]=i
        groups=collections.defaultdict(list)
        for i,r in enumerate(good): groups[find(i)].append(r)
        counts=collections.Counter(r['mode'] for r in good); want={m:max(1,round(n*200/len(good))) for m,n in counts.items()}
        have=collections.Counter(); selected=[]; remaining=list(groups.values()); random.Random(42).shuffle(remaining)
        # Whole connected components prevent repeated prompts or media crossing the split.
        while remaining and len(selected)<200:
            candidates=[(i,g) for i,g in enumerate(remaining) if len(selected)+len(g)<=240]
            if not candidates: break
            def score(item):
                c=collections.Counter(r['mode'] for r in item[1]); benefit=sum(min(v,max(0,want[m]-have[m])) for m,v in c.items())
                return benefit/max(1,len(item[1]))
            i,g=max(candidates,key=score); selected.extend(g); have.update(r['mode'] for r in g); remaining.pop(i)
        ids={r['id'] for r in selected}; val=[convert(r,manifests) for r in selected]
        train=[convert(r,manifests) for r in good if r['id'] not in ids]; excluded=[]
    assert not ({k for r in train for k in r['leakage_keys']} & {k for r in val for k in r['leakage_keys']})
    dump(out/'train_partial.jsonl',train); dump(out/'val.jsonl',val)
    dump(out/'rejected.jsonl',[r for r in rows if r not in good])
    (out/'system_prompt.txt').write_text(SYSTEM+'\n')
    valhash=hashlib.sha256((out/'val.jsonl').read_bytes()).hexdigest()
    report=dict(source_rows=len(rows),success_rows=len(good),train_rows=len(train),val_rows=len(val),val_sha256=valhash,
                train_modes=dict(collections.Counter(r['mode'] for r in train)),val_modes=dict(collections.Counter(r['mode'] for r in val)),
                future_media_leakage_excluded_ids=excluded,seed=42,raw_teacher_targets_preserved=True,
                sources={'official_guide':'https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md',
                         'open_rewriter':'https://huggingface.co/lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA/blob/main/prompt_template.py'})
    (out/'split_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__': main()
