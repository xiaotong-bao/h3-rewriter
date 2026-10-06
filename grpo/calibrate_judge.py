"""Regression calibration on synthetic prompts, never held-out 101 rewrites."""
import argparse,concurrent.futures,hashlib,json,pathlib,time,urllib.request
from run_config import PORT,JUDGE_REVISION
from format_rules import check_format
from reward_policy import positive_eligible

def rewrite(description,music='N/A',sound='Quiet physical scene ambience.'):
 return 'integrated_multimodal_description: [Shot 1] '+description+'\noverall_soundscape: '+sound+'\nnon_diegetic_music: '+music

def cases():
 groups=[
  ('A red ball rolls to the right, strikes a wooden post, and comes to a stop.',
   'A red ball rolls to the right, strikes a wooden post, and comes to a stop.',
   ['A red ball rolls to the left, strikes a wooden post, and comes to a stop.',
    'A red ball rolls to the right, strikes a wooden post, and continues rolling without stopping.']),
  ('A woman closes a yellow umbrella, puts it on a bench, and waves goodbye.',
   'A woman closes a yellow umbrella, puts it on a bench, and waves goodbye.',
   ['A woman closes a yellow umbrella and puts it on a bench. She remains still, with no farewell gesture.']),
  ('A puddle of water shrinks until the pavement is dry.',
   'A puddle of water shrinks and completely disappears, leaving the pavement dry.',
   ['A puddle of water shrinks until the pavement is dry. In the final view the same pavement remains fully covered by the puddle.']),
  ('A cyclist pedals forward while a dog runs alongside.',
   'A cyclist pedals forward while a dog runs alongside.',
   ['A cyclist pedals backward while a dog runs alongside.']),
  ('Two robots swap a red box and a blue box, then stand still.',
   'Two robots swap a red box and a blue box, then stand still.',
   ['Three robots swap a red box and a blue box, then continue dancing.'])]
 result=[]
 for i,(source,good,bads) in enumerate(groups):
  result.append({'id':f'{i}_faithful','original':source,'rewrite':rewrite(good),'expect':'faithful'})
  for n,bad in enumerate(bads):result.append({'id':f'{i}_critical_{n}','original':source,'rewrite':rewrite(bad),'expect':'critical'})
 result.extend([
  {'id':'state_correct_clear','original':'A misted glass pane clears until transparent.','rewrite':rewrite('A misted glass pane clears until fully transparent.'),'expect':'faithful'},
  {'id':'state_wrong_final','original':'A misted glass pane clears until transparent.','rewrite':rewrite('A misted glass pane clears until transparent. In the final view the same glass remains opaque with mist.'),'expect':'critical'},
  {'id':'state_requested_remist','original':'A misted glass pane clears, then becomes misted again.','rewrite':rewrite('A misted glass pane clears, then becomes misted again.'),'expect':'faithful'}])
 source,good,_=groups[0]
 result.extend([
  {'id':'unrequested_music','original':source,'rewrite':rewrite(good,music='A slow piano melody plays.'),'expect':'music'},
  {'id':'unrequested_dialogue','original':source,'rewrite':rewrite(good+' An off-screen speaker S1 says <d>[English]I won!</d>.'),'expect':'dialogue'},
  {'id':'bad_format','original':source,'rewrite':rewrite(good).replace('non_diegetic_music: N/A',''),'expect':'format'}])
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--port',type=int,default=PORT);a=p.parse_args()
 with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/',timeout=5) as r:health=json.load(r)
 assert health['revision']==JUDGE_REVISION and health['runtime']=='local_codex'
 def check(row):
  payload={'original':row['original'],'context':'Requested task: t2va; aspect ratio: 16:9; duration: 5s.','rewrite':row['rewrite']}
  errors=[];verdict=None
  for attempt in range(3):
   try:
    req=urllib.request.Request(f'http://127.0.0.1:{a.port}/score',json.dumps(payload).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=650) as r:verdict=json.load(r)
    break
   except Exception as e:errors.append(repr(e));time.sleep(2)
  ok=False
  if verdict:
   eligible=positive_eligible(verdict,all(check_format(row['rewrite'],'t2va',5).values()))
   ok={'faithful':not verdict['items'] and eligible,
       'critical':verdict['severity_counts']['severe']>0 and not eligible,
       'music':verdict['unrequested_music'] and not eligible,
       'dialogue':verdict['unrequested_dialogue'] and not eligible,
       'format':not eligible}[row['expect']]
  return {**row,'passed':ok,'verdict':verdict,'errors':errors}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(check,cases()))
 report={'passed':all(r['passed'] for r in results),'cases':len(results),'passed_cases':sum(r['passed'] for r in results),'judge_revision':JUDGE_REVISION,'synthetic_only':True,'results':results}
 a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
 if not report['passed']:raise SystemExit('Calibration failed: '+','.join(r['id'] for r in results if not r['passed']))
if __name__=='__main__':main()
