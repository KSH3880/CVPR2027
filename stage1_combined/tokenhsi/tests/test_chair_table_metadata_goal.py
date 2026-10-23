from pathlib import Path
import torch
import yaml
from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_ontop_spec import batched

ROOT=Path(__file__).resolve().parents[1]


def test_metadata_goal_changes_only_climb_target_and_keeps_physical_surface():
    env=yaml.safe_load((ROOT/'data/cfg/multi_agent/chair_table_metadata_full_ft.yaml').read_text())['env']
    graph=compose_canonical_graph(torch.tensor([[1,2],[2,1]]))
    objects=torch.zeros(2,4,13);objects[...,6]=1
    objects[...,0]=torch.arange(4).float();objects[...,2]=.3
    sizes=torch.full((2,4,3),.6)
    hands=torch.zeros(2,2,2,3);feet=torch.zeros_like(hands);roots=torch.zeros(2,2,3);goals=torch.zeros(2,2,3)
    args=(hands,feet,roots,objects,sizes,goals,graph,env['relationReward'],.94)
    old_phi,old=evaluate_interaction_edges(*args)
    # Deliberately differs from bbox; includes a chair-style 10cm offset.
    height=torch.tensor([[.4,.27,.3,.3],[.4,.27,.3,.3]])
    target=objects[...,:3].clone();target[...,2]+=height+.94
    new_phi,new=evaluate_interaction_edges(*args,climb_targets=target)
    sit=batched(graph.edge_relation,2)==9
    climb=batched(graph.edge_relation,2)==10
    dst=batched(graph.edge_dst,2)-2;row=torch.arange(2)[:,None]
    torch.testing.assert_close(new['target'][climb],target[row,dst.clamp_min(0)][climb])
    torch.testing.assert_close(new_phi[sit],old_phi[sit],rtol=0,atol=0)
    for key,value in old.items():
        if torch.is_tensor(value) and value.shape[:2]==sit.shape:
            torch.testing.assert_close(new[key][sit],value[sit],rtol=0,atol=0)
    for key in ['z_surface','feet_height_error','feet_individual_max_error','region_error']:
        torch.testing.assert_close(new[key],old[key],rtol=0,atol=0)


def test_metadata_variant_only_adds_climb_goal_option():
    cfg=ROOT/'data/cfg/multi_agent'
    old=yaml.safe_load((cfg/'chair_table_full_ft.yaml').read_text())
    new=yaml.safe_load((cfg/'chair_table_metadata_full_ft.yaml').read_text())
    assert new['env']['chairFT'].pop('climb_goal')=='tokenhsi_metadata'
    assert new==old
