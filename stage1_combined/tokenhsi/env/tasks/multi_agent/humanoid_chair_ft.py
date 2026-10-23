import json
from pathlib import Path
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
from utils.edge_ontop_spec import select_graph
from utils.chair_geometry import EXTRA, table_box_token_inputs

class NoBoxRsi:
    def __init__(self,task):pass

class HumanoidChairFT(HumanoidMACarry):
    def __init__(self,*args,**kwargs):
        self._solo_cache_class=NoBoxRsi
        super().__init__(*args,**kwargs)
        self._size_template_probabilities.zero_()
        self._size_template_probabilities[...,1:3]=.5
        if self.num_agents!=2 or self.num_objects!=4:raise ValueError('Chair FT requires 2 humans and 4 objects')

    def get_obs_size(self):return super().get_obs_size()+self.num_objects*EXTRA
    def get_scene_entity_sizes(self):
        result=super().get_scene_entity_sizes();result[1]+=EXTRA;return result

    def _load_box_asset(self):
        root=Path(__file__).resolve().parents[4]/'chair_assets'
        catalog=json.loads((root/'catalog.json').read_text())
        split=self.cfg['env']['chairFT'].get('asset_split','train')
        chairs=catalog[split];tables=catalog['tables'];N,O=self.num_envs,self.num_objects
        self._chair_ids=torch.randint(len(chairs),(N,O),device=self.device)
        self._chair_table_ft = self.cfg['env']['chairFT'].get('target_categories') == ['chair', 'table']
        if self._chair_table_ft:
            # One chair and one table target; randomize owner assignment each environment.
            kinds=torch.randint(1,3,(N,O),device=self.device)
            first=torch.randint(1,3,(N,),device=self.device)
            kinds[:,0]=first;kinds[:,1]=3-first
        else:
            kinds=torch.randint(3,(N,O),device=self.device);kinds[:,:2]=2 # 0 box, 1 table, 2 chair
        self._chair_kinds=kinds
        self._chair_features=torch.zeros(N,O,EXTRA,device=self.device)
        self._chair_seats=torch.zeros(N,O,3,device=self.device)
        self._climb_metadata_heights=torch.zeros(N,O,device=self.device)
        self._chair_seat_extents=torch.zeros(N,O,2,device=self.device)
        self._box_size=torch.empty(N,O,3,device=self.device)
        self._box_density=torch.full((N*O,),100.,device=self.device)
        self._box_assets=[];cache={}
        model_ids=self._chair_ids.cpu().tolist();kind_ids=kinds.cpu().tolist()
        for e in range(N):
            for o in range(O):
                kind=kind_ids[e][o];index=model_ids[e][o]
                options=gymapi.AssetOptions();options.fix_base_link=True
                options.angular_damping=.01;options.linear_damping=.01;options.max_angular_velocity=100.
                options.default_dof_drive_mode=gymapi.DOF_MODE_NONE
                if kind==0:
                    size=[.4,.4,.4];key=('box',)
                    if key not in cache:cache[key]=self.gym.create_box(self.sim,*size,options)
                else:
                    entry=chairs[index] if kind==2 else tables[index%len(tables)]
                    metadata=json.loads((Path(entry['path'])/'config.json').read_text())
                    self._climb_metadata_heights[e,o]=float(metadata['tarSitPos'][2])
                    size=entry['bbox'];key=(kind,entry['path'] if kind==1 else entry['id'])
                    if key not in cache:
                        folder=Path(entry['path']);options.use_mesh_materials=True
                        options.mesh_normal_mode=gymapi.COMPUTE_PER_VERTEX;options.vhacd_enabled=True
                        options.vhacd_params.resolution=100000;options.vhacd_params.max_convex_hulls=128
                        options.vhacd_params.max_num_vertices_per_ch=64
                        cache[key]=self.gym.load_asset(self.sim,str(folder),'asset.urdf',options)
                        if cache[key] is None or self.gym.get_asset_rigid_body_count(cache[key])!=1:raise ValueError('Invalid rigid asset')
                    if kind==2:
                        self._chair_features[e,o]=torch.tensor(entry['feature'],device=self.device)
                        self._chair_seats[e,o]=torch.tensor(entry['seat'],device=self.device)
                        self._chair_seat_extents[e,o]=torch.tensor(entry['seat_extent'],device=self.device)
                    elif self._chair_table_ft:
                        # A table has no backrest; its horizontal top is the support surface.
                        top=[0.,0.,float(size[2])/2]
                        self._chair_features[e,o]=torch.tensor([1.,0.,0.,*top],device=self.device)
                        self._chair_seats[e,o]=torch.tensor(top,device=self.device)
                        self._chair_seat_extents[e,o]=torch.tensor(size[:2],device=self.device)
                self._box_size[e,o]=torch.tensor(size,device=self.device)
                self._box_assets.append(cache[key])
        self._box_scale=self._box_size/torch.tensor(self._build_base_size,device=self.device)
        # Only a graph-template eligibility proxy; actual physics/observations retain full native bbox.
        self._rsi_asset_sizes=torch.full_like(self._box_size,.45)
        print('[chair FT] '+('targets: one chair + one table; distractors: chair/table only' if self._chair_table_ft else 'objects: two task chairs + two fixed random box/table/chair distractors')+'; split='+split,flush=True)

    def _compute_clean_scene_obs(self,env_ids=None):
        obs=super()._compute_clean_scene_obs(env_ids)
        B=obs.shape[0];H=self.num_agents*self.get_clean_humanoid_obs_size();O=self.num_objects
        objects=obs[:,H:H+O*30].reshape(B,O,30)
        shape=self._logical_box_values(self._chair_features,env_ids)
        if self.cfg['env']['chairFT'].get('table_input_bbox') == 'metadata_top_box_tokenizer':
            objects,shape=table_box_token_inputs(objects,shape,self._logical_box_values(self._chair_kinds,env_ids),self._logical_box_values(self._climb_metadata_heights,env_ids))
        return torch.cat([obs[:,:H],torch.cat([objects,shape],-1).reshape(B,-1),obs[:,H+O*30:]],-1)

    def _reset_ref_state_init(self,slot_env,slot_agent):
        # Shared loco initialization, no box-specific RSI retargeting/cache.
        saved=self._is_eval
        try:
            self._is_eval=True
            return super()._reset_ref_state_init(slot_env,slot_agent)
        finally:self._is_eval=saved

    def _evaluate_ontop_context(self,env_ids=None):
        bodies=self._rigid_body_pos if env_ids is None else self._kinematic_humanoid_rigid_body_states[env_ids,...,:3]
        goals=self._tar_pos if env_ids is None else self._tar_pos[env_ids]
        objects=self._logical_box_values(self._box_states,env_ids).clone()
        sizes=self._logical_box_values(self._box_size,env_ids).clone()
        climb_targets=None
        if self.cfg['env']['chairFT'].get('climb_goal') in ('tokenhsi_metadata','object_comparison'):
            # Original TokenHSI: object origin XY, world Z + metadata Z + char_h.
            climb_targets=objects[...,:3].clone()
            climb_targets[...,2]+=self._logical_box_values(self._climb_metadata_heights,env_ids)+self._char_h
        desc=self._logical_box_values(self._chair_features,env_ids)
        seats=self._logical_box_values(self._chair_seats,env_ids)
        extent=self._logical_box_values(self._chair_seat_extents,env_ids)
        chair=desc[...,0]>.5
        seat_world=objects[...,:3]+quat_rotate(objects[...,3:7].reshape(-1,4),seats.reshape(-1,3)).reshape_as(seats)
        if self.cfg['env']['chairFT'].get('climb_goal') == 'object_comparison':
            chair_asset=self._logical_box_values(self._chair_kinds,env_ids)==2
            climb_targets=torch.where(chair_asset[...,None],seat_world+torch.tensor([0.,0.,self._char_h],device=self.device),climb_targets)
        objects[...,:3]=torch.where(chair[...,None],seat_world-torch.tensor([0.,0.,.01],device=self.device),objects[...,:3])
        seat_size=torch.cat([extent,torch.full_like(extent[...,:1],.02)],-1)
        sizes=torch.where(chair[...,None],seat_size,sizes)
        graph=self.relation_runtime.graph if env_ids is None else select_graph(self.relation_runtime.graph,env_ids)
        config=dict(self._relation_cfg);config['sit']=dict(config['sit']);config['sit']['pelvis_clearance']=self.cfg['env']['chairFT']['sit_pelvis_clearance']
        phi,diag=evaluate_interaction_edges(bodies[...,self._key_body_ids[[0,1]],:],bodies[...,self._key_body_ids[[2,3]],:],bodies[...,0,:],objects,sizes,goals,graph,config,self._char_h,climb_targets=climb_targets)
        if env_ids is None:self._ontop_last_diag=diag
        return phi,diag

    def _reset_envs(self,env_ids):
        result=super()._reset_envs(env_ids)
        if not hasattr(self,'_chair_scene_saved') and len(env_ids)==self.num_envs:
            graph=select_graph(self.relation_runtime.graph,env_ids)
            _,diag=self._evaluate_ontop_context(env_ids)
            obj=self._logical_box_values(self._box_states,env_ids)
            seat=self._logical_box_values(self._chair_seats,env_ids)
            seat_world=obj[...,:3]+quat_rotate(obj[...,3:7].reshape(-1,4),seat.reshape(-1,3)).reshape_as(seat)
            target_index=(graph.edge_dst-self.num_agents).clamp(0,self.num_objects-1)
            row=torch.arange(len(env_ids),device=self.device)[:,None]
            sit=(graph.edge_relation==9)&graph.edge_valid
            climb=(graph.edge_relation==10)&graph.edge_valid
            expected=seat_world[row,target_index,2]+torch.where(sit,self.cfg['env']['chairFT']['sit_pelvis_clearance'],self._char_h)
            if self.cfg['env']['chairFT'].get('climb_goal') in ('tokenhsi_metadata','object_comparison'):
                height=self._logical_box_values(self._climb_metadata_heights,env_ids)
                expected=torch.where(climb,obj[row,target_index,2]+height[row,target_index]+self._char_h,expected)
                if self.cfg['env']['chairFT'].get('climb_goal') == 'object_comparison':
                    is_chair=self._logical_box_values(self._chair_kinds,env_ids)[row,target_index]==2
                    expected=torch.where(climb & is_chair,seat_world[row,target_index,2]+self._char_h,expected)
            error=(diag['target'][...,2]-expected).abs()[sit|climb].max()
            assert float(error)<1e-5 and not (graph.edge_valid&~(sit|climb)).any()
            report={'num_envs':self.num_envs,'num_agents':self.num_agents,'num_objects':self.num_objects,
                    'sit_edges':int(sit.sum()),'climb_edges':int(climb.sum()),
                    'target_objects_all_chairs':bool((self._chair_kinds[:,:2]==2).all()),
                    'target_counts_box_table_chair':torch.bincount(self._chair_kinds[:,:2].flatten(),minlength=3).cpu().tolist(),
                    'task_counts_by_object_category':{str(kind):{'sit':int((sit & (self._chair_kinds[row,target_index]==kind)).sum()),'climb':int((climb & (self._chair_kinds[row,target_index]==kind)).sum())} for kind in (1,2)},
                    'distractor_counts_box_table_chair':torch.bincount(self._chair_kinds[:,2:].flatten(),minlength=3).cpu().tolist(),
                    'climb_goal':self.cfg['env']['chairFT'].get('climb_goal','support_surface'),
                    'max_seat_target_z_error':float(error),'descriptor_width':EXTRA,
                    'object_obs_width':30+EXTRA,'training_split':self.cfg['env']['chairFT']['asset_split']}
            folder=Path(self.cfg['args'].output_path);folder.mkdir(parents=True,exist_ok=True)
            (folder/'chair_scene_validation.json').write_text(json.dumps(report,indent=2))
            self._chair_scene_saved=True
        return result
