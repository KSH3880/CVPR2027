import pytest
import torch
import yaml

from test_student_rsi import make_bank, row
from utils.hybrid_late_rsi import HybridLateRSIBank, phase_samples, validate_hybrid_rsi


def make_hybrid(tmp_path, records):
    student = make_bank(tmp_path, [row(skill='sit')])
    spec = yaml.safe_load(open('tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_hybrid_late_rsi.yaml'))['env']['referenceRsi']
    spec['student_bank'] = str(tmp_path/'bank.json')
    spec['validated_fraction'] = 1.
    return HybridLateRSIBank(spec, student.libs, tmp_path/'dataset.yaml',
                            tmp_path/'asset.xml', 'cpu', records, 'fixture')


def test_bounds_and_disjoint_bands(tmp_path):
    bank = make_hybrid(tmp_path, {})
    phases = [p for band in range(4) for p in phase_samples(bank.spec, band)]
    assert min(phases) == .5 and max(phases) == .95
    assert len(phases) == len(set(phases)) == 32
    with pytest.raises(ValueError):
        validate_hybrid_rsi(dict(bank.spec, phase_bands=[[.8, .98]]))


def test_same_xyz_rejection_moves_backward_then_loco(tmp_path):
    bank = make_hybrid(tmp_path, {('carryWith', (20, 25, 30)):
                       [(0, .85, 0), (0, .75, 1), (0, .65, 2), (0, .55, 3)]})
    size = torch.tensor([[.2, .25, .3]])
    before = size.clone()
    for phase in [.85, .75, .65, .55]:
        _, times, empty = bank.sample('carryWith', size)
        assert times.item() == pytest.approx(phase) and not empty.item()
        token = bank.last_tokens.repeat(3)
        bank.observe(token, torch.tensor([True, False, True]))
    _, _, empty = bank.sample('carryWith', size)
    assert empty.item()
    torch.testing.assert_close(size, before)
    bank.on_epoch(1000)
    assert bank.sample('carryWith', size)[1].item() == pytest.approx(.85)


def test_support_pair_and_missing_xyz_never_substitute(tmp_path):
    bank = make_hybrid(tmp_path, {('putDownOnTop', (60, 25, 20), (40, 50, 60)): [(1, .9, 0)]})
    size = torch.tensor([[.6, .25, .2]]*2 + [[.65, .25, .2]])
    support = torch.tensor([[.4, .5, .6], [.4, .5, .55], [.4, .5, .6]])
    _, _, empty = bank.sample_context('putDownOnTop', size, support)
    assert empty.tolist() == [False, True, True]


def test_clip_weight_independent_of_candidate_count_and_resume(tmp_path):
    torch.manual_seed(1)
    bank = make_hybrid(tmp_path, {('putDown', (20, 20, 20)):
        [(0, p, 0) for p in [.8, .85, .9, .95]] + [(1, .85, 0)]})
    ids, _, _ = bank.sample('putDown', torch.full((4000, 3), .2))
    assert .72 < (ids == 1).float().mean().item() < .78
    bank.observe(torch.tensor([0, 0, 0]), torch.tensor([True]*3))
    state = bank.state_dict()
    restored = make_hybrid(tmp_path, {('putDown', (20, 20, 20)):
        [(0, p, 0) for p in [.8, .85, .9, .95]] + [(1, .85, 0)]})
    restored.load_state_dict(state)
    torch.testing.assert_close(bank.counts, restored.counts)
    assert restored.counts[0] == 3


def test_validated_preference_and_sit_unchanged(tmp_path):
    bank = make_hybrid(tmp_path, {('pickUp', (20, 20, 20)): [(0, .8, 0), (1, .9, 0)]})
    bank.observe(torch.tensor([0]*3), torch.tensor([True]*3))
    ids, _, _ = bank.sample('pickUp', torch.full((100, 3), .2))
    assert (ids == 0).all()
    _, times, _ = bank.sample('sit', torch.full((1, 3), .65))
    assert times.item() == pytest.approx(.95)
    assert bank.last_tokens.item() == -1


def test_physical_screen_nonfinite_and_displacement(tmp_path):
    from env.tasks.multi_agent.hybrid_rsi_screen import screen_states
    bank = make_hybrid(tmp_path, {})
    initial_root = torch.zeros(4, 13)
    initial_boxes = torch.zeros(4, 2, 13)
    root, boxes = initial_root.clone(), initial_boxes.clone()
    root[1, 7] = 3.1
    boxes[2, 1, 0] = .31
    boxes[3, 0, 0] = float('nan')
    assert screen_states(root, boxes, initial_root, initial_boxes,
                         bank.spec['screen']).tolist() == [True, False, False, False]


def test_actual_yaml_contract():
    from utils.unified_training import validate_unified_env
    from utils.edge_scenario_spec import validate_sampler
    from utils.reference_rsi import validate_reference_rsi
    env = yaml.safe_load(open('tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_hybrid_late_rsi.yaml'))['env']
    validate_reference_rsi(env['referenceRsi'])
    validate_unified_env(env)
    validate_sampler(env['relationGraph'], 2, 4)
    assert env['templateRsi']['HOLDING_ON_TOP'][-1] == .1
    assert env['numEnvs'] == 2048


def test_xyz_diagnostics_counts_and_only_observed_candidates(tmp_path):
    import csv
    from types import SimpleNamespace
    from env.tasks.multi_agent.hybrid_rsi_runtime import HybridRSIRuntime
    bank = make_hybrid(tmp_path, {('putDownOnTop', (65, 60, 55), (40, 40, 40)):
                                [(0, .8, 0), (1, .9, 0)]})
    bank.observe(torch.tensor([1]*3), torch.tensor([True]*3))
    runtime = HybridRSIRuntime.__new__(HybridRSIRuntime)
    runtime.env = SimpleNamespace(cfg={'args': SimpleNamespace(output_path=str(tmp_path))})
    runtime.bank, runtime.session, runtime.last_dump_epoch = bank, 'fixture', 0
    runtime.counts = torch.zeros(5000, 8, dtype=torch.long)
    runtime.counts[4987] = torch.tensor([2, 10, 1, 1, 1, 0, 1, 1])
    runtime.dump(1000)
    with (tmp_path/'diagnostics/hybrid_xyz_episodes_fixture.csv').open() as f:
        entries = list(csv.DictReader(f))
    assert len(entries) == 1
    assert [entries[0][k] for k in ('task', 'x_cm', 'y_cm', 'z_cm', 'completed', 'final_success')] == ['holding_ontop', '65', '60', '55', '1', '1']
    with (tmp_path/'diagnostics/hybrid_candidates_fixture.csv').open() as f:
        candidates = list(csv.DictReader(f))
    assert len(candidates) == 1 and candidates[0]['observations'] == '3'
