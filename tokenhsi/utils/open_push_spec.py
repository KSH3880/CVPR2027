"""Joint OPEN/PUSH semantic graph contracts, without simulator dependencies."""
import math
import torch
from utils.edge_context_spec import EdgeContextGraph, AT
from utils.edge_door_spec import OPEN
from utils.edge_ontop_spec import expand_graph
PUSH = 12
SAMPLER = 'two_agent_open_push'
VARIANT = 'stage1_open_push'


def compile_graph(spec, m, o, device=None):
    push_only=spec.get('push_only',False)
    allowed={'sampler','semantic_only','task_probabilities'}
    if type(push_only) is not bool or (m,o)!=(2,2 if push_only else 4) or not allowed<=set(spec)<=allowed|{'shuffle_token_order','push_only'} or spec['sampler']!=SAMPLER or spec['semantic_only'] is not True:
        raise ValueError('Joint OPEN/PUSH requires 2 humans, 2 doors, 2 boxes')
    if type(spec.get('shuffle_token_order',False)) is not bool:
        raise ValueError('shuffle_token_order must be boolean')
    p=spec['task_probabilities']
    if set(p)!={'OPEN','PUSH'} or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in p.values()) or not math.isclose(sum(p.values()),1.):
        raise ValueError('Invalid OPEN/PUSH probabilities')
    if push_only:
        if p!={'OPEN':0.,'PUSH':1.}:raise ValueError('Door-free scenes require PUSH-only probabilities')
        src=torch.tensor([0,2,1,3],device=device)
        dst=torch.tensor([2,4,3,5],device=device)
        rel=torch.tensor([PUSH,AT,PUSH,AT],device=device)
        owner=torch.tensor([0,0,1,1],device=device)
        valid=torch.ones(4,dtype=torch.bool,device=device)
        return EdgeContextGraph(('task0','goal0','task1','goal1'),m,o,src,dst,rel,owner,valid,valid.clone(),
            torch.zeros(4,4,dtype=torch.bool,device=device),torch.full_like(src,-1))
    src=torch.tensor([0,4,1,5],device=device)
    dst=torch.tensor([2,6,3,7],device=device)
    rel=torch.tensor([OPEN,AT,OPEN,AT],device=device)
    owner=torch.tensor([0,0,1,1],device=device)
    valid=torch.tensor([True,False,True,False],device=device)
    return EdgeContextGraph(('task0','goal0','task1','goal1'),m,o,src,dst,rel,owner,valid,valid.clone(),
        torch.zeros(4,4,dtype=torch.bool,device=device),torch.full_like(src,-1))


def graph_for_tasks(spec, tasks):
    push_only=spec.get('push_only',False)
    g=expand_graph(compile_graph(spec,2,2 if push_only else 4,tasks.device),len(tasks))
    if push_only:
        if (tasks!=1).any():raise ValueError('Door-free scenes cannot assign OPEN')
        return g
    for a in range(2):
        e=2*a;push=tasks[:,a]==1
        g.edge_dst[:,e]=2+a+2*push.long()
        g.edge_relation[:,e]=torch.where(push,PUSH,OPEN)
        g.edge_valid[:,e+1]=push
        g.required_goal[:,e+1]=push
    return g


def validate_reward(c):
    if 'contact_at' in c:
        from copy import deepcopy
        a=c['contact_at']
        fields={'near_distance_scale','satisfaction_threshold','z_tolerance','progress_delta','at_progress_delta','progress_sigma'}
        if not isinstance(a,dict) or not fields<=set(a)<=fields|{'max_tilt','tilt_full_angle','tilt_zero_angle'}:
            raise ValueError('Invalid CONTACT+AT parameters')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in a.values()):
            raise ValueError('CONTACT+AT parameters must be finite and positive')
        if a['satisfaction_threshold']!=.9 or a['near_distance_scale']!=10 or ('max_tilt' in a and a['max_tilt']>=math.pi/2):
            raise ValueError('Invalid CONTACT threshold/tilt')
        if {'tilt_full_angle','tilt_zero_angle'} & set(a):
            if not {'tilt_full_angle','tilt_zero_angle','max_tilt'} <= set(a) or not 0 < a['tilt_full_angle'] < a['max_tilt'] < a['tilt_zero_angle'] < math.pi/2:
                raise ValueError('Invalid CONTACT+AT tilt shaping angles')
        if c.get('edge_aggregation')!='self_sum_teammate_mean' or c['state_reward_weight']!=.4:
            raise ValueError('CONTACT+AT requires split edges and state weight 0.4')
        if c.get('at',{}).get('tilt_shaping') or c.get('push_at',{}).get('contact_shaping'):
            raise ValueError('CONTACT+AT cannot combine with old shaping modes')
        legacy=deepcopy(c);legacy.pop('contact_at');legacy['state_reward_weight']=.2
        validate_reward(legacy)
        return
    split=c.get('edge_aggregation')=='self_sum_teammate_mean'
    expected={'mode','schema_version','stage1_variant','state_reward_weight','progress_reward_weight',
        'success_reward_weight','door','push','task_sharing','success','observation','diagnostics'}
    if split:
        expected |= {'edge_aggregation','satisfaction_threshold','at','push_at'}
    if set(c)!=expected or c['mode']!='state_relation_edge_stage1_v1' or c['schema_version']!=12 or c['stage1_variant']!=VARIANT:
        raise ValueError('Invalid joint OPEN/PUSH reward schema')
    if c['observation']!={'graph_packet_fields':['valid','src','dst','relation','owner']} or c['task_sharing']!={'self':.9,'teammate':.1}:
        raise ValueError('Preserve unified semantic fields and sharing')
    saturation='paired_placement_current_success' if split else 'current_success'
    if c['success']!={'saturation':saturation,'terminate_when_all_subgoals_done':False}:
        raise ValueError('Joint tasks use current success')
    if [c[k] for k in ('state_reward_weight','progress_reward_weight','success_reward_weight')]!=([.2,.2,.2] if split else [.4,.2,.2]):
        raise ValueError('Joint tasks use 0.4/0.2/0.2 reward scales')
    if split:
        if c['push'].get('control_mode')!='midpoint_xy_net_credit':
            raise ValueError('Split PUSH+AT requires midpoint hand geometry')
        hand=dict(c['push_at'])
        if type(hand.pop('require_tilt',True)) is not bool:
            raise ValueError('PUSH require_tilt must be boolean')
        reference=hand.pop('hand_reference','box_rear_midpoint_xy')
        if reference not in ('box_rear_midpoint_xy','goal_opposite_surface_xy'):
            raise ValueError('Invalid PUSH+AT hand reference')
        shaping=hand.pop('contact_shaping',None)
        if shaping is not None:
            fields={'mode','alignment_weight','alignment_sigma2','gate_full_distance','gate_start_distance'}
            if not isinstance(shaping,dict) or set(shaping)!=fields or shaping['mode']!='two_hand_face_xy_alignment':
                raise ValueError('Invalid CONTACT shaping fields/mode')
            if reference!='goal_opposite_surface_xy':
                raise ValueError('CONTACT shaping requires goal-opposite surface reference')
            values=[shaping[k] for k in fields-{'mode'}]
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
                raise ValueError('CONTACT shaping parameters must be finite numbers')
            if not (0<shaping['alignment_weight']<1 and shaping['alignment_sigma2']>0 and
                    0<shaping['gate_full_distance']<shaping['gate_start_distance']):
                raise ValueError('Invalid CONTACT shaping weight/scale/distances')
        at=dict(c['at'])
        success_tilt=at.pop('success_max_tilt',None)
        if success_tilt is not None:
            if isinstance(success_tilt,bool) or not isinstance(success_tilt,(int,float)) or not math.isfinite(success_tilt) or not 0<success_tilt<math.pi/2:
                raise ValueError('Invalid AT success_max_tilt')
            if any(at.get(k) for k in ('tilt_shaping','state_require_upright','require_grounded')):
                raise ValueError('AT success_max_tilt cannot combine with other AT tilt gates')
        tilt=at.pop('tilt_shaping',None)
        if tilt is not None:
            if not isinstance(tilt,dict) or set(tilt)!={'full_angle','zero_angle','success_angle'}:
                raise ValueError('Invalid AT tilt shaping fields')
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in tilt.values()):
                raise ValueError('AT tilt angles must be finite numbers')
            if not 0<=tilt['full_angle']<tilt['success_angle']<tilt['zero_angle']<math.pi/2:
                raise ValueError('Invalid AT tilt angle ordering')
            if at.get('state_require_upright',False) or at.get('require_grounded',False):
                raise ValueError('AT tilt shaping cannot combine with legacy AT gates')
        if type(at.pop('state_require_upright',False)) is not bool:
            raise ValueError('AT state_require_upright must be boolean')
        if type(at.pop('require_grounded',False)) is not bool:
            raise ValueError('AT require_grounded must be boolean')
        progress_delta=at.pop('progress_delta',None)
        if isinstance(progress_delta,bool) or not isinstance(progress_delta,(int,float)) or not math.isfinite(progress_delta) or progress_delta<0:
            raise ValueError('AT progress_delta must be a finite nonnegative distance')
        if hand!={'hand_distance_scale':10.,'approach_delta':.5,'approach_sigma':1.} or at!={'near_distance_scale':10.,'z_tolerance':.001,'progress_sigma':1.} or c['satisfaction_threshold']!=.9:
            raise ValueError('Split PUSH+AT preserves HOLDING+AT geometry and progress scales')
    d=c['door'];p=c['push']
    push_fields={'target_distance','goal_tolerance','ground_tolerance','max_tilt','max_speed','max_angular_speed','hand_distance','min_hand_force','min_box_force','min_contact_progress','hand_lateral_offset','hand_target_height'}
    if 'control_mode' in p and p['control_mode']!='midpoint_xy_net_credit':
        raise ValueError('Invalid PUSH control mode')
    if set(d)!={'target_angle','angle_tolerance','hand_distance_scale','approach_delta','approach_sigma'} or set(p) not in (push_fields,push_fields|{'control_mode'}):
        raise ValueError('Invalid joint task reward parameters')
    for key,value in list(d.items())+[(k,v) for k,v in p.items() if k!='control_mode']:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (key!='angle_tolerance' and value==0):
            raise ValueError('Reward parameters must be finite and positive')
    if not 0<=d['angle_tolerance']<d['target_angle']<1.7 or not 0<p['goal_tolerance']<p['target_distance'] or not 0<p['max_tilt']<math.pi/2 or not p['min_contact_progress']<p['target_distance']-p['goal_tolerance']:
        raise ValueError('Invalid task thresholds')


def checkpoint_contracts_match(saved,current):
    """Allow relocating an identical RSI bank, retaining all content checks."""
    if saved==current:return True
    if not isinstance(saved,dict) or not isinstance(current,dict):return False
    if not saved.get('rsi_bank_sha256') or saved['rsi_bank_sha256']!=current.get('rsi_bank_sha256'):return False
    from copy import deepcopy
    left,right=deepcopy(saved),deepcopy(current)
    for contract in (left,right):
        if isinstance(contract.get('rsi'),dict):contract['rsi'].pop('bank_file',None)
    return left==right


def validate_env(env):
    if env.get('pushGoalContact',False) is not bool(env.get('relationReward',{}).get('contact_at')):
        raise ValueError('pushGoalContact must match the CONTACT+AT reward mode')
    from utils.push_scene import validate_scene
    validate_scene(env)
    if env['relationReward']['push'].get('control_mode')=='midpoint_xy_net_credit' and env.get('manipulationContact') is None:
        raise ValueError('Midpoint PUSH requires physics-tick movement credit collection')
    if env.get('manipulationContact') is not None:
        from utils.manipulation_contact import validate_contact
        validate_contact(env['manipulationContact'])
    compile_graph(env['relationGraph'],env['numAgents'],env['numObjects'])
    validate_reward(env['relationReward'])
    if env['relationReward'].get('edge_aggregation') and not env['relationGraph'].get('push_only',False):
        raise ValueError('Split PUSH+AT is a dedicated door-free experiment')
    if env['policyObsMode']!='clean_scene' or env['ampTaskConditioning'] is not True or env['ampEnabled'] is not True:
        raise ValueError('Joint tasks require clean scene and task-conditioned AMP')
    if env['skill']!=['loco','doorOpen','push'] or env['skillInitProb']!=[1.,0.,0.] or env['stateInit']!='Start':
        raise ValueError('Both tasks initialize from locomotion clip starts')
    if env.get('openPushRSI') is not None:
        from utils.open_push_rsi import validate_rsi
        validate_rsi(env['openPushRSI'])
    if set(env['ampExpertMapping'])!={'OPEN','PUSH'}:
        raise ValueError('Explicit task expert mappings required')
    for row in env['ampExpertMapping'].values():
        if len(row)!=3 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in row) or not math.isclose(sum(row),1.):
            raise ValueError('Invalid task expert mixture')
    if env['ampExpertMapping']['OPEN'][2] or env['ampExpertMapping']['PUSH'][1]:
        raise ValueError('Never mix door and box-push experts across tasks')
    p=env['relationGraph']['task_probabilities']
    nominal=[p['OPEN']*x+p['PUSH']*y for x,y in zip(env['ampExpertMapping']['OPEN'],env['ampExpertMapping']['PUSH'])]
    if len(env['skillDiscProb'])!=3 or any(abs(x-y)>1e-8 for x,y in zip(nominal,env['skillDiscProb'])):
        raise ValueError('Nominal expert mixture must match task frequencies')
    if env.get('referenceRsi') or env.get('templateRsi') or env['box']['reset']['randomHeight'] or env['box']['build']['randomSize']:
        raise ValueError('Joint task uses fixed box geometry and locomotion reset')
    # Reuse the existing hold validation with its original one-person contract.
    from utils.door_hold import validate_hold
    proxy=dict(env,numAgents=1,numObjects=1)
    validate_hold(proxy)
    lo,hi=env['door']['resetAngleRange']
    if not 0<=lo<=hi<env['relationReward']['door']['target_angle']-env['relationReward']['door']['angle_tolerance']:
        raise ValueError('Doors must reset unsolved')
    box=env['pushBox']
    validate_variable_push(env)
    if set(box)-{'size_profiles'}!={'size','mass','friction'} or len(box['size'])!=3 or any(not math.isfinite(v) or v<=0 for v in box['size']+[box['mass'],box['friction']]):
        raise ValueError('Invalid push box physics')
    if box['size']!=env['box']['build']['baseSize']:
        raise ValueError('Push box size settings must agree')
    p=env['relationReward']['push']
    if not 0<p['hand_lateral_offset']<box['size'][0]/2 or not 0<p['hand_target_height']<box['size'][2]:
        raise ValueError('PUSH hand targets must lie inside the rear face')


def validate_variable_push(env):
    profiles=env['pushBox'].get('size_profiles')
    if profiles is None:return
    expected=[[round(1+i*.1,1),round(1+j*.1,1),1.3] for i in range(6) for j in range(6)]
    y2=[[round(1+i*.1,1),2.,1.3] for i in range(6)]
    if not ((profiles==expected and env['pushBox']['size']==[1.5,1.5,1.3]) or
            (profiles==y2 and env['pushBox']['size']==[1.5,2.,1.3])):
        raise ValueError('Variable PUSH requires X1..1.5m, Y1..1.5m or fixed2m, Z1.3m')
    if not env['relationGraph'].get('push_only') or env.get('openPushRSI',{}).get('pose_mode')!='native_phase':
        raise ValueError('Variable boxes require PUSH-only native phase RSI')
