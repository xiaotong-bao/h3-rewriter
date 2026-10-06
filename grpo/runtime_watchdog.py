"""Supervise only this job's local Linux gateway; never manage SSH sessions."""
import fcntl,json,os,pathlib,signal,subprocess,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
PORT=int(os.environ.get('H3_LUNA_PORT','8792'))
REVISION='ep3-v2-luna-severity-v3'
def event(kind,**details):
 with (ROOT/'watchdog_events.jsonl').open('a') as log:log.write(json.dumps({'time':time.time(),'event':kind,**details})+'\n')
def health():
 try:
  with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/',timeout=5) as response:r=json.load(response)
  return r.get('ready') and r.get('revision')==REVISION and r.get('model')=='gpt-6-luna',None
 except Exception as e:return False,repr(e)
def main():
 lock=(ROOT/'runtime_watchdog.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('This job already has a local supervisor')
 (ROOT/'runtime_watchdog.pid').write_text(str(os.getpid()))
 event('started',pid=os.getpid(),port=PORT)
 child=None;failures=0
 while not (ROOT/'RUNTIME_STOP').exists():
  healthy,error=health();failures=0 if healthy else failures+1
  if not healthy and (child is None or child.poll() is not None or failures>=3):
   if child is not None and child.poll() is None:
    child.terminate()
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:child.kill();child.wait()
   with (ROOT/'gateway.log').open('a') as log:
    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'luna_judge.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   (ROOT/'runtime_pids.json').write_text(json.dumps({'gateway':child.pid,'watchdog':os.getpid()}))
   event('gateway_started',pid=child.pid);failures=0
  (ROOT/'watchdog_status.json').write_text(json.dumps({'time':time.time(),'pid':os.getpid(),'port':PORT,'local_healthy':bool(healthy),'error':error,'gateway_pid':child.pid if child else None},indent=2))
  time.sleep(15)
 if child is not None and child.poll() is None:child.terminate();child.wait(timeout=10)
 event('stopped')
if __name__=='__main__':main()
