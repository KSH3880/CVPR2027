"""BEFORE pairs retain their roles; only independent people can select PUSH."""
from copy import deepcopy
import hashlib
from pathlib import Path
import torch
from utils.before_spec import sample_roles
from utils.stage1_jointpush import compose_graph
from utils.edge_ontop_spec import select_graph

SAMPLER='stage2_before_push_groups'


def validate_env(env):
    from utils.unified_training import validate_common_env, SKILLS
    from utils.experiments import native_config
    from utils.task_role_spec import TASK_PROBS
    if env['numAgents']<1 or env['numObjects']<2*env['numAgents']:
        raise ValueError('BEFORE PUSH requires at least two object tokens per person')
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
    if env['jointPush']!=expected or env['ampExpertMapping']!=source['ampExpertMapping']:
        raise ValueError('BEFORE solo PUSH requires the Stage1 PUSH physics/contact/expert mapping')
    from utils.before_relative_sampling import DISTANCE_MIXTURE
    original=native_config('stage2_before')['env']['before']
    near=deepcopy(original);near['distance_mixture']=deepcopy(DISTANCE_MIXTURE)
    limited=deepcopy(near);limited.update(pairing_probability=.5,source_max_distance=5.)
    if env['before'] not in (original,near,limited):
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
    probs=[probabilities[k] for k in ('SIT','CLIMB','HOLDING_AT','HOLDING_ON_TOP','PUSH_AT')]
    base='all_solo' if preset=='push' else preset
    predecessor,templates=sample_roles(sizes,ranges,base,solo_probabilities=probs,pairing_probability=pairing_probability)
    if preset=='push':templates.fill_(5)
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
