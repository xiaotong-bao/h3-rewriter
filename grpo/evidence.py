"""Validate exact, ordered quote fragments; ellipses denote skipped source text."""
import re
def numbered_spans(source):
    matches=list(re.finditer(r'.+?(?:[.!?。！？](?=\s|$)|\n|$)',source,re.S))
    spans=[]
    for match in matches:
        if match.group().strip():spans.append({'id':len(spans)+1,'start':match.start(),'end':match.end(),'text':match.group()})
    assert spans,'Cannot segment empty source'
    assert all(source[s['start']:s['end']]==s['text'] for s in spans)
    return spans

def selected_spans(ids,spans):
    lookup={s['id']:s for s in spans}
    assert isinstance(ids,list) and all(isinstance(i,int) and not isinstance(i,bool) and i in lookup for i in ids),'Invalid evidence sentence IDs'
    return [lookup[i] for i in sorted(set(ids))]

def evidence_spans(evidence,source):
    fragments=evidence if isinstance(evidence,list) else [x for x in re.split(r'\s*(?:\.\.\.|…)\s*',evidence) if x.strip()]
    assert fragments and all(isinstance(x,str) and x.strip() for x in fragments),'Empty quote fragment'
    offset=0;spans=[]
    for fragment in fragments:
        fragment=fragment.strip()
        start=source.find(fragment,offset)
        assert start>=0,'Quote fragment not found in order: '+repr(fragment)
        end=start+len(fragment);spans.append({'start':start,'end':end,'text':source[start:end]});offset=end
    return spans

def quoted_reason_evidence(reason,source):
    # Only recover the judge's own exact quotes already present in its reasoning.
    # Never synthesize evidence or change a semantic verdict.
    quoted=re.findall(r"['\"‘“]([^'\"’”]+)['\"’”]",reason)
    matches=sorted({q for q in quoted if len(q.strip())>=8 and q in source},key=source.find)
    assert matches,'No grounded quote in judge reasoning'
    return matches,evidence_spans(matches,source)
