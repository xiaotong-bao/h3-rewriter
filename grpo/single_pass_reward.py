"""Frozen userlike reward: single-pass content issues plus three-field format."""
import math

REVISION = 'retention-single-pass-v3-luna-userlike-v1'


def score_record(record):
    if record.get('judge_failed'):
        return dict(reward=None, positive_eligible=False, judge_failed=True)
    issues = record['issues']
    assert all(i['severity'] in ('severe', 'general') for i in issues)
    severe = sum(i['severity'] == 'severe' for i in issues)
    general = sum(i['severity'] == 'general' for i in issues)
    assert record['format_score'] in (0, 100)
    invalid = record['format_score'] == 0
    reward = max(-1., 1. - .8 * min(severe, 2) - .1 * min(general, 3) - .3 * invalid)
    assert math.isfinite(reward)
    return dict(reward=reward, positive_eligible=severe == 0 and not invalid,
                judge_failed=False, severity_counts=dict(severe=severe, general=general),
                format_valid=not invalid)


def evidence_schema(schema):
    """Encode existing partial/contradiction citation requirements in the schema."""
    import copy
    result = copy.deepcopy(schema)
    omitted = copy.deepcopy(result['properties']['issues']['items'])
    omitted['properties']['status']['enum'] = ['omitted']
    present = copy.deepcopy(result['properties']['issues']['items'])
    present['properties']['status']['enum'] = ['partial', 'contradicted']
    present['properties']['rewrite_quote']['minLength'] = 1
    result['properties']['issues']['items'] = {'anyOf': [omitted, present]}
    return result
