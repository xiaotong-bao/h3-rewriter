import json,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent
command="python3 - <<'REMOTE'\nimport pathlib,json,glob\nr=pathlib.Path('/data/xiaotong/h3_rewriter_sft_20261002/grpo_ep3_luna_v2_20261006')\nstate=json.loads((r/'pipeline_status.json').read_text())\nstate['optimizer_status']={s:json.loads((r/s/'status.json').read_text()) for s in ['smoke','pilot'] if (r/s/'status.json').exists()}\nstate['rollout_counts']={p.name:len(p.read_text().splitlines()) for p in r.glob('rollouts.rank*.jsonl')}\nstate['judge_failure_counts']={p.name:len(p.read_text().splitlines()) for p in r.glob('judge_failures.rank*.jsonl')}\nprint(json.dumps(state))\nREMOTE"
while True:
 try:
  result=subprocess.run(['ssh','-o','ConnectTimeout=15','hyperbolic-pika-node-0010',command],capture_output=True,text=True,timeout=40)
  if result.returncode:raise RuntimeError(result.stderr)
  state=json.loads(result.stdout);state['local_time']=time.time();(ROOT/'training_status.json').write_text(json.dumps(state,indent=2))
  with (ROOT/'training_history.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
  print(state,flush=True)
  if state.get('state')=='failed' or state.get('stage')=='evaluation_101' and state.get('state')=='complete':break
 except Exception as error:print(repr(error),flush=True)
 time.sleep(30)
