"""Fail-closed reward validation and eligibility for positive policy advantages."""
import math
from evidence import numbered_spans, selected_spans
LEVELS = ('severe', 'general', 'review')

def validate_requirements(source, requirements):
    spans = numbered_spans(source)
    assert requirements and [r['id'] for r in requirements] == list(range(1, len(requirements)+1)), 'Requirement IDs must be complete and ordered'
    covered = set()
    for r in requirements:
        assert isinstance(r['critical'], bool) and r['text'].strip()
        evidence = selected_spans(r['source_ids'], spans)
        assert evidence, 'Requirement lacks source evidence'
        covered.update(r['source_ids'])
    assert covered == {s['id'] for s in spans}, 'Source sentences missing from checklist'
    return requirements

def validate_and_score(source, rewrite, requirements, result):
    validate_requirements(source, requirements)
    lookup = {r['id']:r for r in requirements}
    checklist = result['checklist']
    assert len(checklist) == len(lookup) and {i['requirement_id'] for i in checklist} == set(lookup), 'Incomplete or duplicated checklist verdicts'
    spans = numbered_spans(rewrite); original = numbered_spans(source); issues = []
    for item in checklist:
        req = lookup[item['requirement_id']]
        assert item['status'] in ('preserved','partial','omitted','contradicted')
        assert item['severity'] in ('none',*LEVELS) and item['reason'].strip()
        evidence = selected_spans(item['evidence_ids'],spans)
        if item['status'] != 'omitted':assert evidence, 'Missing rewrite evidence'
        if item['status'] == 'preserved':
            assert item['severity'] == 'none', 'Preserved requirement has an issue severity'
            continue
        assert item['severity'] in LEVELS, 'Violation cannot have severity none'
        # Confirmed damage to an extracted critical requirement cannot silently be downgraded.
        if req['critical'] and item['severity'] != 'review':
            assert item['severity'] == 'severe', 'Critical requirement violation was downgraded'
        issues.append({'severity':item['severity'],'status':item['status'],'category':'source_retention',
            'requirement_id':req['id'],'source_ids':req['source_ids'],'evidence_ids':item['evidence_ids'],
            'source_evidence':' ... '.join(s['text'] for s in selected_spans(req['source_ids'],original)),
            'evidence':' ... '.join(s['text'] for s in evidence),'reason':item['reason']})
    for item in result['v2_issues']:
        assert item['category'] in ('v2_music','v2_dialogue') and item['severity'] in LEVELS
        evidence=selected_spans(item['evidence_ids'],spans)
        assert evidence and item['reason'].strip(), 'Ungrounded v2 issue'
        covered=item['covered_requirement_id']
        if covered:
            assert covered in lookup and lookup[covered].get('category')==item['category'].removeprefix('v2_'), 'Invalid v2 coverage link'
            assert any(i['requirement_id']==covered and i['status']!='preserved' for i in checklist), 'V2 coverage points to preserved requirement'
            continue
        issues.append({**item,'status':'contradicted','source_ids':[],
            'source_evidence':'','evidence':' ... '.join(s['text'] for s in evidence)})
    for flag,category in [('unrequested_music','v2_music'),('unrequested_dialogue','v2_dialogue')]:
        assert isinstance(result[flag],bool)
        assert result[flag] == any(i['category']==category for i in result['v2_issues']), 'Flag and issue list disagree: '+flag
    for i,item in enumerate(issues,1):item['id']=i
    counts={level:sum(i['severity']==level for i in issues) for level in LEVELS}
    reward=1.-.8*min(counts['severe'],2)-.1*min(counts['general'],3)-.02*min(counts['review'],3)
    assert math.isfinite(reward)
    result.update(requirements=requirements,items=issues,severity_counts=counts,reward=reward,retention=reward,
        maximum_severity=next((s for s in LEVELS if counts[s]),'none'))
    return result

def positive_eligible(verdict, format_valid):
    if verdict.get('judge_failed') or not format_valid:return False
    if verdict['severity_counts']['severe'] or verdict['unrequested_music'] or verdict['unrequested_dialogue']:return False
    critical={r['id'] for r in verdict['requirements'] if r['critical']}
    return not any(i['requirement_id'] in critical and i['status']!='preserved' for i in verdict['checklist'])

def gate_advantage(value, eligible, judge_failed=False):
    if judge_failed:return 0.
    return min(value,0.) if not eligible else value
