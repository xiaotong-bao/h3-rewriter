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

def validate_and_score(source, rewrite, requirements, result, require_state_audit=False):
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
    if require_state_audit:
        assert 'state_audit' in result, 'Independent state audit missing'
    if 'state_audit' in result:
        states=result['state_audit']
        assert len(states)==len(original) and {s['source_id'] for s in states}=={s['id'] for s in original}, 'Incomplete state audit'
        for state in states:
            assert state['status'] in ('preserved','omitted','contradicted','uncertain','not_applicable')
            assert all(isinstance(state[k],str) and state[k].strip() for k in ('expected_initial','expected_transition','expected_final','actual_final','reason'))
            evidence=selected_spans(state['evidence_ids'],spans)
            if state['status'] in ('preserved','contradicted'):assert evidence, 'State judgment lacks evidence'
            if state['status']=='not_applicable':
                assert all(state[k]=='N/A' for k in ('expected_initial','expected_transition','expected_final')), 'State constraint cannot be skipped'
            if state['status'] in ('preserved','not_applicable'):continue
            severity='review' if state['status']=='uncertain' else 'severe'
            # Retain audit separately; avoid recharging a same-source severe violation.
            if any(i['severity']==severity and state['source_id'] in i['source_ids'] for i in issues):continue
            issues.append({'severity':severity,'status':state['status'],'category':'state_transition',
                'source_ids':[state['source_id']],'evidence_ids':state['evidence_ids'],
                'source_evidence':selected_spans([state['source_id']],original)[0]['text'],
                'evidence':' ... '.join(s['text'] for s in evidence),'reason':state['reason']})
    for i,item in enumerate(issues,1):item['id']=i
    counts={level:sum(i['severity']==level for i in issues) for level in LEVELS}
    reward=1.-.8*min(counts['severe'],2)-.1*min(counts['general'],3)-.02*min(counts['review'],3)
    assert math.isfinite(reward)
    result.update(requirements=requirements,items=issues,severity_counts=counts,reward=reward,retention=reward,
        maximum_severity=next((s for s in LEVELS if counts[s]),'none'))
    if 'severity_review' in result:
        apply_severity_review(result)
    return result

def apply_severity_review(result):
    import copy
    issues=copy.deepcopy(result['items'])
    severe={i['id'] for i in issues if i['severity']=='severe'}
    decisions=result['severity_review']
    assert len(decisions)==len(severe) and {d['issue_id'] for d in decisions}==severe, 'Incomplete severe issue review'
    lookup={d['issue_id']:d for d in decisions}
    reviewed=[]
    for item in issues:
        if item['id'] in lookup:
            d=lookup[item['id']]
            assert d['final_severity'] in (*LEVELS,'none') and d['reason'].strip()
            item.update(severity=d['final_severity'],severity_review=d)
        if item['severity']!='none':reviewed.append(item)
    counts={level:sum(i['severity']==level for i in reviewed) for level in LEVELS}
    reward=1.-.8*min(counts['severe'],2)-.1*min(counts['general'],3)-.02*min(counts['review'],3)
    result.update(primary_items=issues,items=reviewed,severity_counts=counts,reward=reward,retention=reward,
                  maximum_severity=next((s for s in LEVELS if counts[s]),'none'))
    return result

def positive_eligible(verdict, format_valid):
    if verdict.get('judge_failed') or not format_valid:return False
    if 'severity_review' not in verdict and any(s['status'] not in ('preserved','not_applicable') for s in verdict.get('state_audit',[])):return False
    if verdict['severity_counts']['severe'] or verdict['unrequested_music'] or verdict['unrequested_dialogue']:return False
    if 'severity_review' in verdict:
        return not any(i['severity']=='review' for i in verdict['items'])
    critical={r['id'] for r in verdict['requirements'] if r['critical']}
    return not any(i['requirement_id'] in critical and i['status']!='preserved' for i in verdict['checklist'])

def gate_advantage(value, eligible, judge_failed=False):
    if judge_failed:return 0.
    return min(value,0.) if not eligible else value
