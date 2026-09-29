"""Small atomic provenance helpers; no writes outside this research directory."""
from pathlib import Path
import hashlib, json, os, sys, time
ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / p) for p in ('generation_recovery', 'work', 'followup')]

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def dump(path, value):
    path = Path(path)
    assert HERE in path.resolve().parents, path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def lock(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Frozen record changed: {path}'
    else:
        dump(path, value)
    return value

def event(stage, **values):
    path = HERE / 'journal.jsonl'
    with path.open('a') as f:
        f.write(json.dumps({'time': time.time(), 'pid': os.getpid(), 'stage': stage, **values}) + '\n')

def status(stage, state, **values):
    path = HERE / 'status.json'
    value = json.loads(path.read_text()) if path.exists() else {
        'objective': 'Execute the full completion_v2 next research plan',
        'order': ['R1', 'R3', 'R2', 'R4', 'R5'],
        'stages': {k: {'state': 'pending'} for k in ['R1', 'R3', 'R2', 'R4', 'R5']},
        'independent_human_evaluation': 'not_started_requires_actual_participants',
        'completion': 'unproven'}
    value['stages'][stage] = {'state': state, 'updated_unix': time.time(), **values}
    dump(path, value)
