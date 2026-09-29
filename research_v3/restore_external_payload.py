"""Recover omitted original payload bytes without regenerating frozen manifests.

Uses a retained exact SQLite index and a hash-matched official archive. Writes
only a new payload file/receipt in an explicitly chosen directory; no extraction
of archive paths, downloads, training, or changes to original evidence.
"""
import argparse
import hashlib
import json
import sqlite3
import tarfile
import time
import traceback
from pathlib import Path
from v3_common import HERE, dump, sha
from r5_intake import BASE, TASKS, NAMES


def specification(family, task, archive, output_dir):
    assert family == 'dsprites' or task == 'Single_Atomic'
    data_root = BASE / 'data' / f'{family}_hard'
    manifest_path = data_root / 'manifest.json' if family == 'dsprites' else data_root / task / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    prefix = task + '/' if family == 'dsprites' else ''
    index = data_root / task / 'index.sqlite'
    assert sha(index) == manifest['files'][prefix + 'index.sqlite']
    connection = sqlite3.connect(f'file:{index}?mode=ro', uri=True)
    assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    count, low, high, size = connection.execute('SELECT count(*),min(id),max(id),max(offset+size) FROM records').fetchone()
    assert (count, low, high) == (432000, 0, 431999)
    end = 0
    for offset, length in connection.execute('SELECT offset,size FROM records ORDER BY offset'):
        assert offset == end and length > 0
        end += length
    assert end == size
    connection.close()
    output_dir = output_dir.resolve()
    assert HERE in output_dir.parents, 'Recovery output must remain inside this research checkout'
    return {'source_sha256': sha(__file__), 'family': family, 'task': task,
            'manifest_path': str(manifest_path.relative_to(HERE)), 'manifest_sha256': sha(manifest_path),
            'index_path': str(index.relative_to(HERE)), 'index_sha256': sha(index),
            'archive_path': str(archive.resolve()), 'expected_archive_sha256': manifest['archive_sha256'],
            'output_directory': str(output_dir.relative_to(HERE)), 'expected_records': count, 'expected_payload_bytes': size,
            'expected_payload_sha256': manifest['files'][prefix + 'payload.bin'],
            'scope': 'Restore original PNG/JSON/mask bytes at retained index offsets. Preserve all original manifests and timing/hash references. No dataset regeneration, new independent data, training, or upload.'}


def restore(family, task, archive, output_dir, execute=False):
    spec = specification(family, task, archive, output_dir)
    if not execute:
        print(json.dumps(spec, indent=2))
        return spec
    output_dir = output_dir.resolve()
    payload = output_dir / 'payload.bin'
    temporary = output_dir / 'payload.restore.partial'
    receipt = output_dir / 'payload_recovery.json'
    assert not payload.exists() and not temporary.exists() and not receipt.exists(), 'Existing/partial recovery must be inspected; never overwrite it'
    assert sha(archive) == spec['expected_archive_sha256'], 'Official archive bytes differ from the recorded experiment'
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(output_dir / 'payload_recovery_spec.json', spec)
    connection = sqlite3.connect(f'file:{HERE / spec["index_path"]}?mode=ro', uri=True)
    seen = bytearray(spec['expected_records'])
    count = 0
    started = time.perf_counter()
    try:
        with temporary.open('xb') as stream:
            stream.truncate(spec['expected_payload_bytes'])
            with tarfile.open(archive, mode='r|gz') as tar:
                for member in tar:
                    parts = Path(member.name).parts
                    if not member.isfile() or len(parts) != 5:
                        continue
                    difficulty, member_task, split, number, filename = parts
                    if difficulty != 'Hard' or member_task != task or split not in ('train', 'test') or filename not in NAMES:
                        continue
                    number = int(number)
                    assert 0 <= number < (64000 if split == 'train' else 8000)
                    key = (number + (64000 if split == 'test' else 0)) * 6 + NAMES.index(filename)
                    assert not seen[key], 'Duplicate archive member'
                    row = connection.execute('SELECT offset,size,sha256 FROM records WHERE id=?', (key,)).fetchone()
                    assert row is not None
                    raw = tar.extractfile(member).read()
                    assert member.size == len(raw) == row[1] and hashlib.sha256(raw).hexdigest() == row[2]
                    stream.seek(row[0])
                    stream.write(raw)
                    seen[key] = 1
                    count += 1
                    if count % 50000 == 0:
                        print('Verified original members', count, flush=True)
            assert count == spec['expected_records'] and all(seen)
        assert temporary.stat().st_size == spec['expected_payload_bytes']
        assert sha(temporary) == spec['expected_payload_sha256']
        # Never rewrite original intake/audit/verification JSON: their hashes
        # are inputs to the frozen model and evaluator protocols.
        assert sha(HERE / spec['manifest_path']) == spec['manifest_sha256']
        assert sha(HERE / spec['index_path']) == spec['index_sha256']
        temporary.rename(payload)
        result = {'spec_sha256': sha(output_dir / 'payload_recovery_spec.json'), 'records_replayed_from_archive': count,
                  'payload_sha256': sha(payload), 'payload_bytes': payload.stat().st_size,
                  'original_manifest_and_index_unchanged': True, 'seconds': time.perf_counter() - started,
                  'independent_second_download': False, 'independent_model_retraining': False}
        dump(receipt, result)
        print(json.dumps(result, indent=2), flush=True)
        return result
    except Exception:
        dump(output_dir / 'payload_recovery_failure.json', {'unix': time.time(), 'traceback': traceback.format_exc(), 'partial_preserved': temporary.exists()})
        raise
    finally:
        connection.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=['dsprites', 'clevr', 'clevrtex'], required=True)
    parser.add_argument('--task', choices=TASKS, default='Single_Atomic')
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Omission prints the plan without writing payload bytes')
    args = parser.parse_args()
    restore(args.family, args.task, args.archive, args.output_dir, args.execute)
