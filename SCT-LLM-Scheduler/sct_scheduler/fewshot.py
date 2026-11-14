"""Explicit opt-in to recorded behavior references; leave source scene untouched."""
import copy
import json
from pathlib import Path
from .core import validate_request


def attach_references(request):
    request = copy.deepcopy(request)
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'prompts/fewshot/manifest.json').read_text())
    request['demonstrations'] = json.loads((root / manifest['demonstrations_path']).read_text())
    request['environment_spec']['controller']['timing_calibration'] = (
        'Ten single-agent Stage1 behavioral recordings are supplied: five AT completions '
        'and five PUSH first-contact trajectories. They provide empirical timing references, '
        'not coordinated release schedules. Follower SIT/CLIMB/ON_TOP durations are not measured.')
    return validate_request(request)


def matching_initial_request(actual, captured):
    """Only reference data and its calibration description may differ."""
    validate_request(actual)
    validate_request(captured)
    actual = copy.deepcopy(actual)
    if 'demonstrations' in actual:
        actual.pop('demonstrations')
        controller = actual['environment_spec']['controller']
        original = captured['environment_spec']['controller']
        if 'timing_calibration' in original:
            controller['timing_calibration'] = original['timing_calibration']
        else:
            controller.pop('timing_calibration', None)
    return actual == captured
