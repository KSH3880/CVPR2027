"""Compatibility ETA annotator for already-running writers, without restarting jobs."""
import argparse,json,os,time
from pathlib import Path
from sct_scheduler.progress import StageETA
p=argparse.ArgumentParser();p.add_argument('pipeline',type=Path);a=p.parse_args()
trackers={}
while True:
    root=a.pipeline/'status.json'
    try:parent=json.loads(root.read_text())
    except (OSError,json.JSONDecodeError):time.sleep(2);continue
    paths=[root]
    for key in ('api_output','eval_output'):
        if parent.get(key):paths.append(Path(parent[key])/'status.json')
    for path in paths:
        try:
            # Only annotate a settled snapshot; recheck it before atomic replacement.
            before=path.stat()
            if time.time()-before.st_mtime < .2:continue
            raw=path.read_bytes();state=json.loads(raw)
            trackers.setdefault(str(path),StageETA()).annotate(state)
            tmp=path.with_name(f'.{path.name}.eta.{os.getpid()}.tmp')
            tmp.write_text(json.dumps(state,indent=2))
            current=path.stat()
            if (current.st_ino,current.st_mtime_ns)==(before.st_ino,before.st_mtime_ns):
                tmp.replace(path)
            else:tmp.unlink()
        except (OSError,json.JSONDecodeError):continue
    if parent.get('completed') or str(parent.get('stage','')).startswith(('failed','stopped')):break
    time.sleep(2)
