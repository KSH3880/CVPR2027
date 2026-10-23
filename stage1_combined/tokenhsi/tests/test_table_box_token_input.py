from pathlib import Path
import json
import torch
from utils.chair_geometry import table_box_token_inputs


def test_all_table_metadata_faces_only_change_input_upper_bbox():
    root=Path(__file__).resolve().parents[1]/'data/dataset_amass_climb/objects'
    configs=[json.loads(p.read_text()) for split in ['train','test'] for p in (root/split/'Table_Square').glob('*/config.json')]
    assert len(configs)==18
    n=len(configs);size=torch.tensor([c['bbox'] for c in configs])
    signs=torch.tensor([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
    objects=torch.randn(n,3,30);corners=size[:,None,:]/2*signs[None]
    objects[...,6:30]=corners.reshape(n,1,24)
    desc=torch.randn(n,3,6);desc[...,0]=torch.tensor([0.,1.,1.])
    kinds=torch.tensor([0,1,2]).expand(n,-1);top=torch.tensor([c['tarSitPos'][2] for c in configs])[:,None].expand(-1,3)
    original=objects.clone();saved=desc.clone()
    obs,features=table_box_token_inputs(objects,desc,kinds,top)
    torch.testing.assert_close(objects,original);torch.testing.assert_close(desc,saved)
    torch.testing.assert_close(obs[:,[0,2]],original[:,[0,2]])
    torch.testing.assert_close(obs[...,:6],original[...,:6])
    b=obs[:,1,6:30].reshape(n,8,3)
    torch.testing.assert_close(b[...,:2],corners[...,:2])
    torch.testing.assert_close(b[...,2].max(-1).values,top[:,1])
    torch.testing.assert_close(b[...,2].min(-1).values,-size[:,2]/2)
    assert not features[:,1].any()
    torch.testing.assert_close(features[:,[0,2]],desc[:,[0,2]])
