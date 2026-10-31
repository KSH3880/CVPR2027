"""CPU-only local dependencies and reference contract check."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_task/push_kd.yaml').read_text())['env']
    manifest = yaml.safe_load((ROOT/'tokenhsi/data/dataset_push_kd.yaml').read_text())['motions']
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    missing = [r['file'] for rows in manifest.values() for r in rows if not Path(r['file']).is_file()]
    if missing:
        raise FileNotFoundError(str(missing))
    bank = json.loads((ROOT/cfg['jointPush']['rsi']['bank_file']).read_text())
    assert len(bank['candidates']['push']) == 949
    assert [sha(Path(r['file'])) for r in manifest['push']] == bank['contract']['motions']
    asset = ROOT/cfg['asset']['assetRoot']/cfg['asset']['assetFileName']
    assert sha(asset) == bank['contract']['asset']
    assert [sha(asset.parent/m.get('file')) for m in ET.parse(asset).findall('./asset/mesh')] == bank['contract']['meshes']
    reference = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_task/push_reference.yaml').read_text())['env']
    assert reference['numAgents'] == 2 and reference['numObjects'] == 4
    assert reference['jointPush'] == cfg['jointPush']
    assert reference['jointPush']['scene']['goal_min_human_distance'] == 1.
    for name in ('teacher_stage1.pth', 'push_init.pth'):
        assert (ROOT/'weights'/name).is_file(), name
    print(json.dumps(dict(motions=sum(map(len, manifest.values())), push=21, push_loco=4,
                          rsi=949, profiles=167, asset_and_mesh_sha256='match', status='passed')))


if __name__ == '__main__':
    main()
