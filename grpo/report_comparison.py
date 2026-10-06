import collections,json,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parent
while True:
 rows=[json.loads(p.read_text()) for p in (ROOT/'comparison_results').glob('*.json')]
 valid=[r for r in rows if not r['luna'].get('judge_failed') and not r['sol'].get('judge_failed')]
 generated=sum(len(p.read_text().splitlines()) for p in ROOT.glob('ep3_samples_rank*.jsonl'))
 stats={'generated':generated,'paired':len(rows),'valid_pairs':len(valid),'failed_pairs':len(rows)-len(valid)}
 table=['| 指标 | Luna | Sol |','|---|---:|---:|']
 for level,label in [('severe','严重错误'),('general','一般错误（最高等级）'),('review','待复核（最高等级）'),('none','无问题')]:
  a=sum(r['luna']['maximum_severity']==level for r in valid);b=sum(r['sol']['maximum_severity']==level for r in valid)
  stats[level]={'luna':a,'sol':b};table.append(f'| {label} | {a} / {len(valid)} | {b} / {len(valid)} |')
 stats['agreement']=sum(r['luna']['maximum_severity']==r['sol']['maximum_severity'] for r in valid)
 stats['sol_severe_luna_general']=sum(r['sol']['maximum_severity']=='severe' and r['luna']['maximum_severity']=='general' for r in valid)
 stats['sol_severe_luna_review_or_none']=sum(r['sol']['maximum_severity']=='severe' and r['luna']['maximum_severity'] in ('review','none') for r in valid)
 stats['format_failures']=sum(not all(r['format_checks'].values()) for r in valid)
 diff=[r for r in valid if r['luna']['maximum_severity']!=r['sol']['maximum_severity']]
 (ROOT/'severity_statistics.json').write_text(json.dumps(stats,indent=2))
 (ROOT/'severity_disagreements.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in diff))
 text=f"已生成 {generated}/256；完成配对 {len(rows)}/256；有效配对 {len(valid)}；调用失败 {len(rows)-len(valid)}。\n\n"+'\n'.join(table)+f"\n\n最高严重程度一致：{stats['agreement']}/{len(valid)}。\nSol 严重、Luna 一般：{stats['sol_severe_luna_general']}。\nSol 严重、Luna 待复核或无问题：{stats['sol_severe_luna_review_or_none']}。\n格式失败：{stats['format_failures']}/{len(valid)}（同一机械检查器）。\n\n这是双模型对照，Sol 判断也需复核；不把分歧直接视作 Luna 绝对错误。原始图片未交给审查模型，图片细节忠实度不在本次范围。GRPO 尚未启动。\n"
 (ROOT/'SEVERITY_COMPARISON.md').write_text(text)
 if len(rows)==256:break
 time.sleep(15)
