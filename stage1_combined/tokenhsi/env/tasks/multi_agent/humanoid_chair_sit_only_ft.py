"""Chair SIT-only full fine-tuning with audited native RSI."""
import json
from pathlib import Path
import torch
from env.tasks.multi_agent.humanoid_chair_mixed_ft import HumanoidChairMixedFT
from env.tasks.multi_agent.humanoid_chair_ft import NoBoxRsi
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from utils.edge_scenario_spec import classify_templates

class HumanoidChairSitOnlyFT(HumanoidChairMixedFT):
    def __init__(self,*args,**kwargs):
        self._solo_cache_class=NoBoxRsi
        HumanoidMACarry.__init__(self,*args,**kwargs)
        if self.num_agents!=2 or self.num_objects!=4:raise ValueError('Requires two humans and four chairs')
    def _load_box_asset(self):
        super()._load_box_asset()
        # Two active chairs plus two chair distractors; no boxes or tables in physics.
        for row in range(self.num_envs):
            for dst,src in [(2,0),(3,1)]:
                self._box_assets[row*4+dst]=self._box_assets[row*4+src]
                self._native_asset_keys[row][dst]=self._native_asset_keys[row][src]
        for dst,src in [(2,0),(3,1)]:
            for name in ['_box_size','_chair_kinds','_chair_features','_chair_seats','_chair_seat_extents','_climb_metadata_heights']:
                a=getattr(self,name);a[:,dst]=a[:,src]
        self._box_scale=self._box_size/torch.tensor(self._build_base_size,device=self.device)
    def _init_relation_runtime(self):
        self._size_template_probabilities.zero_();self._size_template_probabilities[...,1]=1
        return HumanoidMACarry._init_relation_runtime(self)
    def _reset_envs(self,env_ids):
        result=super()._reset_envs(env_ids)
        if len(env_ids)==self.num_envs and not hasattr(self,'_sit_only_checked'):
            row=env_ids.repeat_interleave(self.num_agents);owner=torch.arange(self.num_agents,device=self.device).repeat(len(env_ids))
            kinds=classify_templates(self.relation_runtime.graph,row,owner,True)
            assert (kinds==1).all() and (self._chair_kinds==2).all()
            pool=getattr(self,'_native_pool',None)
            if not self._is_eval:assert pool is not None and pool.counts['sit']>0 and pool.counts['climb']==0
            Path(self.cfg['args'].output_path,'chair_sit_only_validation.json').write_text(json.dumps(dict(num_envs=self.num_envs,task_counts={'SIT':int((kinds==1).sum()),'CLIMB':int((kinds==2).sum())},all_physics_objects_chairs=True,native_counts=pool.counts if pool else {},native_active=int(pool.active.sum()) if pool else 0,native_amp_finite=bool(torch.isfinite(self._amp_obs_buf).all()),rsi_bank=str(pool.path) if pool else None),indent=2))
            self._sit_only_checked=True
        return result
