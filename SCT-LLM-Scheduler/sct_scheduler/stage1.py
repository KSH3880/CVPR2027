"""Stage1 policy identity and native-action verification for local simulator entries."""
import hashlib
import json
from pathlib import Path
from .core import require

ROOT = Path(__file__).resolve().parents[1]


def policy_manifest():
    manifest = json.loads((ROOT / 'stage1_policy.json').read_text())
    require(manifest['epoch'] == 26000 and manifest['stage2_coordination'] is False,
            'Stage1 SCT policy required')
    return manifest


def require_stage1_request(request):
    manifest = policy_manifest()
    require(request['current_scene']['controller_id'] == manifest['controller_id'],
            'Schedule controller mismatch: capture/export a Stage1 scene and obtain a Stage1 schedule')
    return manifest


def load_stage1(player, observation, request):
    manifest = require_stage1_request(request)
    path = Path(manifest['checkpoint_local']).resolve(strict=True)
    require((ROOT / 'teacher_runtime/checkpoints').resolve() in path.parents,
            'Use the local checkpoint copy')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == manifest['sha256'], 'Stage1 checkpoint hash mismatch')
    # Simulator entry adds the local eval directory after importing Isaac Gym.
    from stage1_protocol_policy import Stage1Policy
    return Stage1Policy(player, str(path), observation)
