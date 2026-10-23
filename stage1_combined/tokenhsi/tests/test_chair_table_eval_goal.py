from pathlib import Path
from types import SimpleNamespace
import json
import torch
import yaml
from env.tasks.multi_agent.humanoid_chair_ft import HumanoidChairFT
from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_ontop_spec import batched

ROOT=Path(__file__).resolve().parents[2]


def test_training_targets_match_object_comparison_for_every_chair_and_table():
    cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/chair_table_eval_full_ft.yaml').read_text())
    catalog=json.loads((ROOT/'chair_assets/catalog.json').read_text())
    chairs=catalog['train']+catalog['test'];tables=list(catalog['tables'])
    for folder in sorted((ROOT/'tokenhsi/data/dataset_amass_climb/objects/test/Table_Square').iterdir()):
        if folder.is_dir():
            raw=json.loads((folder/'config.json').read_text());tables.append(dict(path=str(folder),bbox=raw['bbox']))
    n=len(chairs)
    task=HumanoidChairFT.__new__(HumanoidChairFT)
    task.cfg=cfg;task.device='cpu';task._relation_cfg=cfg['env']['relationReward'];task._char_h=.94
    task._box_states=torch.zeros(n,4,13);task._box_states[...,6]=1
    task._box_states[...,0]=torch.arange(4).float();task._box_states[...,2]=.7
    task._chair_kinds=torch.tensor([2,1,2,1]).expand(n,-1)
    task._box_size=torch.zeros(n,4,3)
    task._chair_features=torch.zeros(n,4,6);task._chair_seats=torch.zeros(n,4,3)
    task._chair_seat_extents=torch.zeros(n,4,2);task._climb_metadata_heights=torch.zeros(n,4)
    for row,chair in enumerate(chairs):
        table=tables[row%len(tables)]
        for slot,entry in enumerate([chair,table,chair,table]):
            raw=json.loads((Path(entry['path'])/'config.json').read_text())
            task._box_size[row,slot]=torch.tensor(entry['bbox'])
            if slot%2==0:
                seat=entry['seat'];extent=entry['seat_extent'];feature=entry['feature']
            else:
                seat=[0,0,entry['bbox'][2]/2];extent=entry['bbox'][:2];feature=[1,0,0,*seat]
            task._chair_seats[row,slot]=torch.tensor(seat)
            task._chair_seat_extents[row,slot]=torch.tensor(extent)
            task._chair_features[row,slot]=torch.tensor(feature)
            task._climb_metadata_heights[row,slot]=raw['tarSitPos'][2]
    task._logical_box_values=lambda values,env_ids=None: values if env_ids is None else values[env_ids]
    task._rigid_body_pos=torch.zeros(n,2,4,3);task._key_body_ids=torch.arange(4)
    for templates in [[1,2],[2,1]]:
        graph=compose_canonical_graph(torch.tensor(templates).expand(n,-1))
        task.relation_runtime=SimpleNamespace(graph=graph);task._tar_pos=torch.zeros(n,2,3)
        phi,diag=task._evaluate_ontop_context()
        relation=batched(graph.edge_relation,n);valid=batched(graph.edge_valid,n)
        dst=(batched(graph.edge_dst,n)-2).clamp_min(0);rows=torch.arange(n)[:,None]
        expected=task._box_states[rows,dst,:3].clone()
        surface=torch.where((dst%2)==0,task._chair_seats[rows,dst,2],task._climb_metadata_heights[rows,dst])
        sit_surface=task._chair_seats[rows,dst,2]
        expected[...,2]+=torch.where(relation==9,sit_surface+.1,surface+.94)
        torch.testing.assert_close(diag['target'][valid],expected[valid],atol=2e-7,rtol=0)
        task.cfg['env']['chairFT']['climb_goal']='tokenhsi_metadata'
        old_phi,old_diag=task._evaluate_ontop_context()
        sit=valid&(relation==9)
        torch.testing.assert_close(phi[sit],old_phi[sit],rtol=0,atol=0)
        torch.testing.assert_close(diag['target'][sit],old_diag['target'][sit],rtol=0,atol=0)
        task.cfg['env']['chairFT']['climb_goal']='object_comparison'
