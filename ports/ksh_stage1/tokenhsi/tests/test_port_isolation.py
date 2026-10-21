"""Import/config boundaries of the additive port."""
from pathlib import Path
import json
import hashlib

import yaml

PORT = Path(__file__).resolve().parents[2]


def test_imports_use_private_snapshot():
    import env.tasks.multi_agent.edge_ontop_task as task
    import learning.multi_agent.amp_network_builder_ma as network
    import utils.size_rsi as sampler
    for module in (task, network, sampler):
        assert PORT in Path(module.__file__).resolve().parents


def test_snapshot_matches_recorded_upstream_source():
    manifest = json.loads((PORT / 'UPSTREAM.json').read_text())
    for entry in manifest['files']:
        assert hashlib.sha256((PORT / entry['path']).read_bytes()).hexdigest() == entry['sha256']


def test_large_data_links_are_local_and_configs_are_private():
    repo = PORT.parents[1]
    data = PORT / 'tokenhsi/data'
    assert not (data / 'cfg').is_symlink()
    for link in data.iterdir():
        if link.name == 'cfg':
            continue
        assert link.is_symlink()
        assert link.resolve() == (repo / 'tokenhsi/data' / link.name).resolve()


def test_three_variants_share_environment_conditions():
    directory = PORT / 'tokenhsi/data/cfg/multi_agent'
    stem = 'approach_scenario_stage1_unified_size_rsi'
    baseline = yaml.safe_load((directory / (stem + '.yaml')).read_text())
    for suffix in ('_typed_bias', '_typed_bias_message'):
        other = yaml.safe_load((directory / (stem + suffix + '.yaml')).read_text())
        other['env']['relationReward']['stage1_variant'] = baseline['env']['relationReward']['stage1_variant']
        assert baseline == other
