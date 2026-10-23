"""Joint OPEN/PUSH semantic graph contracts, without simulator dependencies."""
import math
import torch
from utils.edge_context_spec import EdgeContextGraph, AT
from utils.edge_door_spec import OPEN
from utils.edge_ontop_spec import expand_graph
PUSH = 12
SAMPLER = 'two_agent_open_push'
VARIANT = 'stage1_open_push'


def scene_layout(env):
    """Station centers; absent layout preserves existing experiment contracts."""
    if env.get('pushOnly',False):
        layout=env['openPushLayout']
        if set(layout)!={'push_x','lane_spacing'} or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in layout.values()) or layout['lane_spacing']<=0:
            raise ValueError('Invalid PUSH layout')
        return dict(layout)
    layout = env.get('openPushLayout', {'door_x': 1., 'push_x': -1.5, 'lane_spacing': 5.})
    if not isinstance(layout, dict) or set(layout) != {'door_x', 'push_x', 'lane_spacing'}:
        raise ValueError('Invalid OPEN/PUSH layout fields')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in layout.values()) or layout['lane_spacing'] <= 0 or layout['door_x'] <= layout['push_x']:
        raise ValueError('Invalid OPEN/PUSH station coordinates')
    if 'openPushLayout' in env:
        # Nominal paths only: include box sweep, approach distance and body margin.
        required_lane = (env['relationReward']['push']['target_distance'] +
                         env['pushBox']['size'][1] + env['doorSpawnDistanceRange'][1] + 1.)
        required_x = env['pushBox']['size'][0] / 2 + 1.1 + 1.
        if layout['lane_spacing'] < required_lane or layout['door_x'] - layout['push_x'] < required_x:
            raise ValueError('OPEN/PUSH layout overlaps nominal approach/swept corridors')
    return dict(layout)


def compile_graph(spec, m, o, device=None):
    if (m,o) not in ((2,4),(2,2)) or set(spec) not in ({'sampler','semantic_only','task_probabilities'}, {'sampler','semantic_only','task_probabilities','shuffle_token_order'}) or spec['sampler']!=SAMPLER or spec['semantic_only'] is not True:
        raise ValueError('Joint OPEN/PUSH requires 2 humans, 2 doors, 2 boxes')
    if type(spec.get('shuffle_token_order',False)) is not bool:
        raise ValueError('shuffle_token_order must be boolean')
    p=dict(spec['task_probabilities'])
    if p=={'PUSH':1.}:p['OPEN']=0.
    if set(p)!={'OPEN','PUSH'} or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in p.values()) or not math.isclose(sum(p.values()),1.):
        raise ValueError('Invalid OPEN/PUSH probabilities')
    if o==2:
        if p!={'OPEN':0.,'PUSH':1.}:
            raise ValueError('Two-object policy requires PUSH-only tasks')
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
    g=expand_graph(compile_graph(spec,2,4,tasks.device),len(tasks))
    for a in range(2):
        e=2*a;push=tasks[:,a]==1
        g.edge_dst[:,e]=2+a+2*push.long()
        g.edge_relation[:,e]=torch.where(push,PUSH,OPEN)
        g.edge_valid[:,e+1]=push
        g.required_goal[:,e+1]=push
    return g


def validate_reward(c):
    expected={'mode','schema_version','stage1_variant','state_reward_weight','progress_reward_weight',
        'success_reward_weight','door','push','task_sharing','success','observation','diagnostics'}
    if 'door' not in c:expected.remove('door')
    if set(c) not in (expected, expected | {'contact_at'}) or c['mode']!='state_relation_edge_stage1_v1' or c['schema_version']!=12 or c['stage1_variant']!=VARIANT:
        raise ValueError('Invalid joint OPEN/PUSH reward schema')
    if 'contact_at' in c:
        a=c['contact_at']
        if set(a)-{'success_mode','at_progress_delta','max_tilt','tilt_fade_start','tilt_fade_end'} != {'near_distance_scale','satisfaction_threshold','z_tolerance','progress_delta','progress_sigma'}:
            raise ValueError('Invalid contact/AT parameters')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for k,v in a.items() if k!='success_mode') or a['satisfaction_threshold']>1:
            raise ValueError('Invalid contact/AT thresholds')
        if 'max_tilt' in a and not 0<a['max_tilt']<math.pi/2:
            raise ValueError('Invalid AT tilt limit')
        fade_keys={'tilt_fade_start','tilt_fade_end'}
        if fade_keys & set(a):
            if not fade_keys <= set(a) or not 0<a['tilt_fade_start']<a['tilt_fade_end']<math.pi/2:
                raise ValueError('AT tilt fade requires ordered start/end angles in radians')
            if 'max_tilt' not in a or not a['tilt_fade_start']<=a['max_tilt']<=a['tilt_fade_end']:
                raise ValueError('AT success tilt must lie within the fade interval')
        if a.get('success_mode','state_and_height') not in ('state_and_height','state_only'):
            raise ValueError('Invalid contact/AT success mode')
    if c['observation']!={'graph_packet_fields':['valid','src','dst','relation','owner']} or c['task_sharing']!={'self':.9,'teammate':.1}:
        raise ValueError('Preserve unified semantic fields and sharing')
    if c['success']!={'saturation':'current_success','terminate_when_all_subgoals_done':False}:
        raise ValueError('Joint tasks use current success')
    if [c[k] for k in ('state_reward_weight','progress_reward_weight','success_reward_weight')]!=[.4,.2,.2]:
        raise ValueError('Joint tasks use 0.4/0.2/0.2 reward scales')
    d=c.get('door',{});p=c['push']
    if (d and set(d)!={'target_angle','angle_tolerance','hand_distance_scale','approach_delta','approach_sigma'}) or set(p)-{'contact_mode'}!={'target_distance','goal_tolerance','ground_tolerance','max_tilt','max_speed','max_angular_speed','min_hand_force','min_box_force','min_contact_progress','hand_lateral_offset','hand_target_height','hand_distance'}:
        raise ValueError('Invalid joint task reward parameters')
    if p.get('contact_mode','goal_face') not in ('goal_face','any_surface','fixed_face_midpoint','fixed_face_surface'):
        raise ValueError('Invalid PUSH contact mode')
    if p.get('contact_mode') in ('goal_face','any_surface') and ('contact_at' not in c or c['contact_at'].get('success_mode')!='state_only'):
        raise ValueError('Surface contact requires state-only contact/AT reward')
    for key,value in list(d.items())+[(k,v) for k,v in p.items() if k!='contact_mode']:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (key!='angle_tolerance' and value==0):
            raise ValueError('Reward parameters must be finite and positive')
    if (d and not 0<=d['angle_tolerance']<d['target_angle']<1.7) or not 0<p['goal_tolerance']<p['target_distance'] or not 0<p['max_tilt']<math.pi/2 or not p['min_contact_progress']<p['target_distance']-p['goal_tolerance']:
        raise ValueError('Invalid task thresholds')


def validate_env(env):
    if env["relationReward"]["push"].get("contact_mode") in ("fixed_face_midpoint","fixed_face_surface") and (not env.get("pushGoalContact") or "contact_at" not in env["relationReward"]):
        raise ValueError("Fixed face requires goal-face contact/AT reward")
    from utils.open_push_scene import validate_scene
    validate_scene(env)
    if env.get('pushOnly',False):
        if env['pushOnly'] is not True or env['relationGraph']['task_probabilities']!={'PUSH':1.}:
            raise ValueError('PUSH-only profile requires PUSH probability 1')
        if any('door' in k.lower() for k in env) or 'door' in env['relationReward'] or 'OPEN' in env['ampExpertMapping']:
            raise ValueError('PUSH-only profile must not contain door settings')
        scene_layout(env);compile_graph(env['relationGraph'],env['numAgents'],env['numObjects']);validate_reward(env['relationReward'])
        if env['skill']!=['loco','push'] or env['skillInitProb']!=[1.,0.] or env['skillDiscProb']!=[.33,.67] or env['ampExpertMapping']!={'PUSH':[.33,.67]}:
            raise ValueError('PUSH-only AMP requires loco/push 0.33/0.67')
        if env['stateInit']!='Start' or env['policyObsMode']!='clean_scene' or env['ampTaskConditioning'] is not True or env['ampEnabled'] is not True:
            raise ValueError('Invalid PUSH-only initialization/AMP/observation')
        from utils.open_push_rsi import validate_rsi
        validate_rsi(env['openPushRSI'])
        from utils.open_push_sizes import size_values
        size_values(env)
        box=env['pushBox']
        if set(box)!={'size','mass','friction'} or len(box['size'])!=3 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in box['size']+[box['mass'],box['friction']]) or box['size']!=env['box']['build']['baseSize']:
            raise ValueError('Invalid PUSH box physics')
        if env['box']['reset']['randomHeight'] or env['box']['build']['randomSize'] or env.get('referenceRsi') or env.get('templateRsi'):
            raise ValueError('PUSH-only uses size-specific RSI and flat ground')
        return
    if type(env.get('pushNoDoorPolicy',False)) is not bool or (env.get('pushNoDoorPolicy',False) and env['relationGraph']['task_probabilities']!={'OPEN':0.,'PUSH':1.}):
        raise ValueError('Door-free policy requires boolean option and PUSH-only tasks')
    if type(env.get('pushGoalContact',False)) is not bool or (env.get('pushGoalContact',False) and 'pushGoalRandomization' not in env):
        raise ValueError('Goal-facing PUSH contact requires random goal geometry')
    if type(env.get('openPushRandomizeLanes', False)) is not bool:
        raise ValueError('openPushRandomizeLanes must be boolean')
    scene_layout(env)
    compile_graph(env['relationGraph'],env['numAgents'],env['numObjects'])
    validate_reward(env['relationReward'])
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
    from utils.open_push_sizes import size_values
    size_values(env)
    box=env['pushBox']
    if set(box)!={'size','mass','friction'} or len(box['size'])!=3 or any(not math.isfinite(v) or v<=0 for v in box['size']+[box['mass'],box['friction']]):
        raise ValueError('Invalid push box physics')
    if box['size']!=env['box']['build']['baseSize']:
        raise ValueError('Push box size settings must agree')
    p=env['relationReward']['push']
    if not 0<p['hand_lateral_offset']<box['size'][0]/2 or not 0<p['hand_target_height']<box['size'][2]:
        raise ValueError('PUSH hand targets must lie inside the rear face')
