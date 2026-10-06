"""Monitor a local experiment directly; no SSH dependency."""
import argparse,json,pathlib,time
p=argparse.ArgumentParser();p.add_argument('--job',type=pathlib.Path,required=True);a=p.parse_args()
while True:
 try:
  state=json.loads((a.job/'pipeline_status.json').read_text())
  state['optimizer_status']={s:json.loads((a.job/s/'status.json').read_text()) for s in ['smoke','pilot'] if (a.job/s/'status.json').exists()}
  state['rollout_counts']={p.name:len(p.read_text().splitlines()) for p in a.job.glob('rollouts.rank*.jsonl')}
  state['time']=time.time();print(json.dumps(state),flush=True)
  if state.get('state')=='failed' or state.get('stage')=='evaluation_101' and state.get('state')=='complete':break
 except Exception as e:print(repr(e),flush=True)
 time.sleep(30)
