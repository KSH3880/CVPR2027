"""BEFORE pairs retain their roles; only independent people can select PUSH."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import torch
from utils.before_spec import sample_roles
from utils.stage1_jointpush import compose_graph
from utils.edge_ontop_spec import select_graph

SAMPLER='stage2_before_push_groups'


def validate_env(env, evaluation=False):
    from utils.unified_training import validate_common_env, SKILLS
    from utils.experiments import native_config
    from utils.task_role_spec import TASK_PROBS
    from utils.before_object_alias import MODE, VIRTUAL_MODE
    virtual = env['relationGraph'].get('before_virtual_object')
    if virtual is not None and (virtual != VIRTUAL_MODE or env['relationGraph'].get('before_object_alias') != MODE):
        raise ValueError('BEFORE virtual objects require at_goal_xyz_v1 and object aliases')
    minimum = env['numAgents'] if evaluation else 2*env['numAgents']
    if env['numAgents']<1 or env['numObjects']<minimum:
        raise ValueError('BEFORE PUSH requires one object token per person for inference, two for training')
    if env.get('stage2Policy')!={'version':1,'observation':'task_packet_groups_v1','relations':['NONE','SELF','BEFORE','COUPLED']}:
        raise ValueError('Stage2 PUSH requires the common task/group observation and four human relations')
    source=native_config('stage1_push')['env']
    if env['relationGraph']['sampler']!=SAMPLER or env['relationGraph']['template_probabilities']!=source['relationGraph']['template_probabilities']:
        raise ValueError('BEFORE solo action probabilities must match stage1_push')
    if tuple(env['skill'])!=SKILLS+('push',):
        raise ValueError('BEFORE PUSH needs nine skills and four AMP families')
    if any(v != [1.]+[0.]*8 for v in env['templateRsi'].values()):
        raise ValueError('BEFORE PUSH always starts from loco')
    if env['eval']['skill']!=['loco'] or env['eval']['skillInitProb']!=[1.]:
        raise ValueError('BEFORE PUSH evaluation is loco-only')
    expected=deepcopy({k:v for k,v in source['jointPush'].items() if k!='rsi'})
    if env['before'].get('source_max_distance') == 5.:
        expected['scene']['source_distance']=[2.,5.]
    if 'goal_distance' in env['before']:
        expected['scene']['goal_distance']=[2.,5.]
    if env['jointPush']!=expected or env['ampExpertMapping']!=source['ampExpertMapping']:
        raise ValueError('BEFORE solo PUSH requires the Stage1 PUSH physics/contact/expert mapping')
    from utils.before_relative_sampling import DISTANCE_MIXTURE
    from utils.before_goal_sampling import DISTANCE_MIXTURE as GOAL_DISTANCE_MIXTURE
    original=native_config('stage2_before')['env']['before']
    near=deepcopy(original);near['distance_mixture']=deepcopy(DISTANCE_MIXTURE)
    limited=deepcopy(near);limited.update(pairing_probability=.5,source_max_distance=5.)
    before=deepcopy(env['before'])
    goal_distance=before.pop('goal_distance',None)
    asymmetric=before.pop('asymmetric_progress',None)
    if 'wait_clear' in before:
        from env.tasks.multi_agent.before_wait_reward import validate_wait_clear
        validate_wait_clear(before.pop('wait_clear'))
        if virtual != VIRTUAL_MODE or asymmetric is not None or before.get('distance_mixture') != GOAL_DISTANCE_MIXTURE:
            raise ValueError('wait_clear requires goalroute virtual without asymmetric progress')
    if before.get('distance_mixture') == GOAL_DISTANCE_MIXTURE:
        if goal_distance != [1.,5.] or before.get('source_max_distance') != 5.:
            raise ValueError('Goal-centered BEFORE requires source5 and goal5')
        before['distance_mixture']=deepcopy(DISTANCE_MIXTURE)
    if 'asymmetric_progress' in env['before']:
        if (not isinstance(asymmetric,dict) or set(asymmetric)!={'lambda_inner'}
                or type(asymmetric['lambda_inner']) not in (int,float)
                or not math.isfinite(asymmetric['lambda_inner']) or asymmetric['lambda_inner']<0
                or before!=limited or virtual!=VIRTUAL_MODE):
            raise ValueError('Asymmetric progress requires source5 virtual and a finite nonnegative lambda_inner')
    if 'goal_distance' in env['before']:
        if (type(goal_distance) is not list or goal_distance!=[1.,5.]
                or any(type(value) not in (int,float) for value in goal_distance) or before!=limited):
            raise ValueError('BEFORE goal distances require source5 and [1.0, 5.0]')
    if before not in (original,near,limited):
        raise ValueError('BEFORE PUSH requires the mix226 or near50 distance contract with loco-only initialization')
    old=deepcopy(env)
    old.pop('jointPush')
    old['relationGraph']['template_probabilities']=dict(zip(('HOLDING','SIT','CLIMB','HOLDING_AT','HOLDING_ON_TOP'),TASK_PROBS))
    old['relationGraph']['sampler']='stage2_before_groups'
    old['skill']=old['skill'][:-1]
    old['templateRsi']={k:v[:-1] for k,v in old['templateRsi'].items() if k!='PUSH_AT'}
    validate_common_env(old)


def sample_graph(sizes,ranges,probabilities,preset='random_scenario',num_objects=None,pairing_probability=.8):
    n,m=sizes.shape[:2]
    o=2*m if num_objects is None else num_objects
    if m<1 or o<m:
        raise ValueError('BEFORE PUSH requires at least one object token per person')
    required = {'holding_ontop': 2*m, 'before_stack': m+m//2, 'one_pair_stack': m+1}
    if o<required.get(preset,m):
        raise ValueError('BEFORE PUSH preset exceeds the object token budget: '+preset)
    probs=[probabilities[k] for k in ('SIT','CLIMB','HOLDING_AT','HOLDING_ON_TOP','PUSH_AT')]
    base='all_solo' if preset=='push' else preset
    predecessor,templates=sample_roles(sizes,ranges,base,solo_probabilities=probs,pairing_probability=pairing_probability)
    if preset=='push':templates.fill_(5)
    if o<2*m:
        pending=(m+(templates==4).sum(-1)>o).nonzero(as_tuple=True)[0]
        for _ in range(1024):
            if not len(pending):break
            pre,sampled=sample_roles(sizes[pending],ranges,base,solo_probabilities=probs,pairing_probability=pairing_probability)
            fits=m+(sampled==4).sum(-1)<=o
            predecessor[pending[fits]],templates[pending[fits]]=pre[fits],sampled[fits]
            pending=pending[~fits]
        if len(pending):
            raise RuntimeError('BEFORE PUSH task graph exceeds object token budget after 1024 retries; increase numObjects')
    graph=compose_graph(templates,num_objects=2*m)
    rows,follower=(predecessor>=0).nonzero(as_tuple=True)
    edge=((graph.edge_owner[rows]==follower[:,None])&graph.required_goal[rows]).long().argmax(-1)
    leading=((graph.edge_owner[rows]==predecessor[rows,follower,None])&(graph.edge_relation[rows]==7)&graph.edge_valid[rows]).long().argmax(-1)
    graph.edge_dst[rows,edge]=m+predecessor[rows,follower]
    graph.prereq_mask[rows,edge,leading]=True
    return graph,predecessor,templates


def compile_graph(spec,m,o,device=None):
    return select_graph(compose_graph(torch.full((1,m),3,device=device,dtype=torch.long),num_objects=o),0)


def push_contract(env):
    return dict(joint_push=deepcopy(env['jointPush']),skills=env['skill'],experts=env['ampExpertMapping'],
        stage2_policy=deepcopy(env.get('stage2Policy')),motion_manifest_sha256=hashlib.sha256(Path(env['motion_file']).read_bytes()).hexdigest())
