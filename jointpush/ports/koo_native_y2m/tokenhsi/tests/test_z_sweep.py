import pytest
import torch
from utils.z_sweep import make_conditions, box_sizes, graph_rows, summarize
from utils.edge_scenario_spec import compose_graph
from env.tasks.multi_agent.edge_context_reward import owner_sum


def test_default_grid_and_fixed_xy_separate_source_and_support():
    conditions = make_conditions()
    assert len(conditions) == 102
    sizes = torch.tensor(box_sizes(conditions, 2))
    assert sizes.shape == (204,4,3)
    assert torch.all(sizes[...,:2] == .4)
    for i,c in enumerate(conditions):
        if c['axis'] == 'source_size_z':
            assert sizes[2*i,0,2].item() == pytest.approx(c['value'])
            assert sizes[2*i,2,2].item() == pytest.approx(.4)
        elif c['axis'] == 'support_size_z':
            assert c['task'] == 'ontop'
            assert sizes[2*i,0,2].item() == pytest.approx(.4)
            assert sizes[2*i,2,2].item() == pytest.approx(c['value'])
        else:
            assert torch.all(sizes[2*i,:,2] == .4)


def test_graph_is_two_independent_tasks_with_matching_geometry_slots():
    conditions = make_conditions(sizes=[.4], offsets=[0.])
    templates, bindings = graph_rows(conditions, 2)
    graph = compose_graph(templates, bindings, num_objects=4)
    assert owner_sum((graph.required_goal & graph.edge_valid).long(), graph).eq(1).all()
    for e, c in enumerate(c for c in conditions for _ in range(2)):
        for a in (0,1):
            required = graph.edge_valid[e] & graph.required_goal[e] & (graph.edge_owner[e]==a)
            index = required.nonzero().item()
            if c['task'] in ('at','ontop'):
                assert graph.edge_src[e,index] == 2+a
                assert graph.edge_dst[e,index] == (6+a if c['task']=='at' else 4+a)
            else:
                assert graph.edge_dst[e,index] == 2+a


def test_summary_counts_trials_not_steps_and_preserves_missing_bins():
    conditions = make_conditions(sizes=[.4], offsets=[0.])
    rows = [dict(condition=0,repeat=0,scene=0,agent=a,ever_success=1,
        final_success=a,initial_success=0,terminated=1) for a in (0,1)]
    result = summarize(rows, conditions)
    assert result[0]['trials'] == 2
    assert result[0]['scene_trials'] == 1
    assert result[0]['ever_success_rate'] == 1
    assert result[0]['final_success_rate'] == .5
    assert result[0]['final_success_scene_all_rate'] == 0
    assert result[0]['ever_success_scene_all_rate'] == 1
    assert result[1]['trials'] == 0
    assert result[1]['final_success_rate'] is None


@pytest.mark.parametrize('args',[dict(x=0),dict(y=float('nan')),dict(offsets=[-.1]),dict(sizes=[])])
def test_bad_geometry_rejected(args):
    with pytest.raises(ValueError): make_conditions(**args)
