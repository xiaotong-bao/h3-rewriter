"""Task-scoped Linux-local Luna high gateway for the frozen single-pass rubric."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
import threading
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'review'))
import single_pass_compare as judge
from single_pass_reward import REVISION, evidence_schema, score_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8799)
    args = parser.parse_args()
    approval = json.loads((args.job / 'single_pass_authorization.json').read_text())
    assert approval['approved'] and approval['revision'] == REVISION
    allowed = set(approval['source_sha256'])
    output = args.job / 'single_pass_reward'
    output.mkdir(exist_ok=True)
    slots = threading.Semaphore(16)
    judge.SCHEMA = evidence_schema(judge.SCHEMA)
    locks = {}
    lock_guard = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(dict(ready=True, revision=REVISION, model='gpt-6-luna', effort='high', port=args.port)).encode())

        def do_POST(self):
            try:
                assert self.path == '/score'
                size = int(self.headers['Content-Length'])
                assert 0 < size <= 2_000_000
                payload = json.loads(self.rfile.read(size))
                assert hashlib.sha256(payload['original'].strip().encode()).hexdigest() in allowed
                context = payload['context']
                row = dict(id='blind', label='blind', original=payload['original'], rewrite=payload['rewrite'],
                    task=re.search(r'Requested task:\s*(\w+)', context)[1],
                    duration_s=float(re.search(r'duration:\s*([0-9.]+)s', context)[1]),
                    aspect=re.search(r'aspect ratio:\s*([^;]+)', context)[1])
                key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                with lock_guard:
                    lock = locks.setdefault(key, threading.Lock())
                with slots, lock:
                    result = judge.grade_retry(row, output, 'gpt-6-luna', 'high', REVISION)
                result.update(score_record(result))
                result.update(judge_revision=REVISION, judge_model='gpt-6-luna', reasoning_effort='high')
                self.send_response(200)
                body = json.dumps(result, ensure_ascii=False).encode()
            except Exception as error:
                traceback.print_exc()
                self.send_response(500)
                body = json.dumps(dict(error=repr(error))).encode()
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

    (output / 'rubric_receipt.json').write_text(json.dumps(dict(revision=REVISION, codebook_revision=judge.REVISION,
        model='gpt-6-luna', effort='high', rules_sha256=hashlib.sha256(judge.RULES.encode()).hexdigest(),
        schema_sha256=hashlib.sha256(json.dumps(judge.SCHEMA, sort_keys=True).encode()).hexdigest()), indent=2))
    print('SINGLE_PASS_LUNA_READY', args.port, REVISION, flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
