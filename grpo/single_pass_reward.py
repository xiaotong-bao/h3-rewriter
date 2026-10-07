"""Frozen userlike reward: single-pass content issues plus three-field format."""
import math

REVISION = 'retention-single-pass-v3-luna-userlike-v3-astra-guard'

LUNA_RULES_APPENDIX = ''' Explicit action and relationship audit: for each original main event,
identify the actor, the performed action, its object, and the direction or spatial relationship.
Require an explicit or unambiguous equivalent in the completed rewrite. Nearby related actions,
object presence, passive completed states, and generic motion do not establish a missing main action.
Distinguish active retrieval from resisting an outgoing force: a mechanism rotating under external
tension does not establish that the actor operates it to retrieve the object. Inspect physical
support/contact relationships: placing a support object on an actor reverses the actor riding or
standing on that support. A later generic assertion does not cancel an explicit incompatible
arrangement. Report such confirmed missing main actions or reversed actor-object relationships
as severe, with verbatim evidence. Do not demand a visual action in progress when the original
requests only a completed state; preserve explicitly stated attribution in that case.
For each coverage entry use one atomic requirement, splitting conjunctions when they contain
different actions or relations. Before marking preserved, cite its exact supporting rewrite_quote
and search the entire rewrite for counterevidence_quote. If an explicit counterexample exists,
mark violated and report the issue even when another sentence repeats the original request.
For requested causal state creation, the actor must create that state: an already existing
state plus the actor nearby does not establish the creation action. Do not conflate an emitted
cloud near a surface with causing a new layer of condensation on that surface.
Severity boundaries: changing slowly to rapidly while the same actor, action, direction and
outcome remain is GENERAL, not severe. Speed adjectives alone do not change main event identity.
An explicit prohibition (including no music or required silence) violated is SEVERE, not general;
compatible music remains allowed when no such prohibition exists.
For reference-driven tasks do not invent unseen reference details; judge explicit text only.'''


def luna_evidence_schema(schema):
    result = evidence_schema(schema)
    coverage = result['properties']['coverage']['items']
    for key in ('rewrite_quote', 'counterevidence_quote'):
        coverage['properties'][key] = {'type': 'string'}
        coverage['required'].append(key)
    return result


def format_for_task(raw, task):
    import re
    keys = (['subject_definitions', 'summary', 'retention_analysis', 'detailed_description',
             'overall_soundscape', 'non_diegetic_music'] if task == 'ref2va' else
            ['integrated_multimodal_description', 'overall_soundscape', 'non_diegetic_music'])
    hits = [list(re.finditer(r'(?m)^\s*' + key + r'\s*:', raw)) for key in keys]
    if not all(len(h) == 1 for h in hits):
        return False
    positions = [h[0].start() for h in hits]
    return positions == sorted(positions) and all(
        raw[hits[i][0].end():positions[i+1] if i+1 < len(keys) else len(raw)].strip()
        for i in range(len(keys)))


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
