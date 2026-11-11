"""Export pre-execution observations only; never read rollout trajectories or outcomes."""
import argparse
import json
from pathlib import Path
import torch
from export_demonstrations import vec, yaw, NAMES, DESCRIPTIONS, completion_definition
from sct_scheduler.core import build_messages, scene_index
from sct_scheduler.stage1 import policy_manifest

ROOT = Path(__file__).resolve().parent


def export(folder, output, env_index=0, raw=None):
    capture = json.loads((folder / 'capture.json').read_text())
    manifest = policy_manifest()
    assert capture.get('policy_controller_id') == manifest['controller_id'], 'Fresh Stage1 capture metadata required'
    if raw is None:
        raw = torch.load(folder / 'initial_scene.pt', map_location='cpu', weights_only=False)
    assert capture['completed'] and 0 <= env_index < capture['environments']
    data = {k: v[env_index] for k, v in raw.items()}
    packet = data['packet'].tolist()
    m = len(packet)
    mapping = data['logical_box_order'].tolist()
    assignment = data['assignment'].tolist()
    parent = data['predecessor'].tolist()
    groups = data['groups'].tolist()
    aid = lambda a: f'A{a+1:02d}'
    oid = lambda b: f'object_{b}'
    scene = dict(scene_id=folder.name + '_env%04d' % env_index, controller_id=manifest['controller_id'],
        coordination_type='mixed', initial_observation={}, tasks=[], event_definitions={}, coordination_constraints=[])
    physical_ids = sorted(set(mapping) | set(assignment))
    scene['initial_observation'] = dict(
        agents=[dict(agent_id=aid(a), task_id='task_' + aid(a),
            root_position_m=vec(r[:3]), root_orientation_xyzw=vec(r[3:7]),
            root_yaw_rad=round(yaw(r), 6), root_linear_velocity_mps=vec(r[7:10]),
            left_hand_position_m=vec(data['hands'][a, 0]), right_hand_position_m=vec(data['hands'][a, 1]))
            for a, r in enumerate(data['roots'])],
        objects=[dict(object_id=oid(b), position_m=vec(data['physical_boxes'][b, :3]),
            orientation_xyzw=vec(data['physical_boxes'][b, 3:7]), dimensions_m=vec(data['physical_sizes'][b]),
            linear_velocity_mps=vec(data['physical_boxes'][b, 7:10])) for b in physical_ids],
        obstacles=[], obstacle_note='Distractor object tokens remain observable, but their physical bodies are parked outside the arena; flat ground is implicit.')
    push_groups = {}
    for a, row in enumerate(packet):
        valid, kind, actor, payload, target = map(int, row)
        assert valid and actor == a
        name = NAMES[kind]
        task = dict(task_id='task_' + aid(a), agent_id=aid(a), task_label=name,
                    description=DESCRIPTIONS[name], group_id=f'group_{groups[a]}')
        if name == 'PUSH':
            event = aid(a) + '_approach'
            task.update(object_id=oid(assignment[a]), readiness_event_id=event,
                object_goal_position_m=vec(data['center_goals'][a]),
                assigned_push_region=dict(center_world_m=vec(data['logical_boxes'][a, :3]),
                    orientation_xyzw=vec(data['logical_boxes'][a, 3:7]), dimensions_m=vec(data['logical_sizes'][a])))
            scene['event_definitions'][event] = (
                f'Estimated end of {aid(a)} approach from its initial pose to its assigned PUSH region, '
                'ready to begin pushing. This is a planning milestone without a numeric distance threshold '
                'or an implemented contact detector. It is not arrival of the box at its final goal. '
                'No stop, freeze, or runtime gate is applied at this milestone.')
            push_groups.setdefault(groups[a], []).append(event)
        else:
            event = aid(a) + '_complete'
            task['completion_event_id'] = event
            scene['event_definitions'][event] = completion_definition(name)
            if kind >= 2:
                task['object_id'] = oid(mapping[payload-m])
            if name == 'AT':
                task['object_goal_position_m'] = vec(data['goals'][target-m-len(mapping)])
            else:
                task['support_object_id'] = oid(mapping[target-m])
            if parent[a] >= 0:
                leader = parent[a]
                goal_index = int(packet[leader][4])-m-len(mapping)
                task['support_intended_final_center_m'] = vec(data['goals'][goal_index])
                task['dependency_sensitive_event_id'] = event
                task['dependency_sensitive_event_note'] = 'Existing formal task-completion proxy; a distinct interaction-onset detector is unavailable. Do not describe this event as first contact.'
                scene['coordination_constraints'].append(dict(type='enabling',
                    enabling_event_id=aid(leader)+'_complete', dependent_event_id=event))
            if a in parent:
                task['enabling_event_id'] = event
                task['description'] += ' Enabling predecessor: start at episode time 0.'
        scene['tasks'].append(task)
    # Follower timing must target the dependency-sensitive interaction onset,
    # not completion of sitting/climbing/placement.
    for task in scene['tasks']:
        old = task.get('dependency_sensitive_event_id')
        if old:
            new = task['agent_id'] + '_interaction'
            task.pop('completion_event_id', None)
            task['dependency_sensitive_event_id'] = new
            onset = {'SIT': 'beginning seated contact with the support',
                     'CLIMB': 'first load-bearing foothold on the support',
                     'ON_TOP': 'beginning placement of the carried payload onto the support'}[task['task_label']]
            scene['event_definitions'].pop(old)
            scene['event_definitions'][new] = 'Estimated first dependency-sensitive interaction: ' + onset + '. Include approach, turning and any pickup/carry preparation before this point. This is an unmeasured planning milestone, not task completion and not a runtime gate.'
            task['dependency_sensitive_event_note'] = 'Interaction onset at the support intended final state; not task completion.'
            for c in scene['coordination_constraints']:
                if c.get('dependent_event_id') == old: c['dependent_event_id'] = new
    if 'task_slots' in data:
        required_physical={mapping[i] for i,valid in enumerate(data['task_slots'].tolist()) if valid}
        for obj in scene['initial_observation']['objects']:
            obj['physical_collision_enabled'] = int(obj['object_id'].split('_')[1]) in required_physical
    for events in push_groups.values():
        if len(events) > 1:
            assert 2 <= len(events) <= 4
            scene['coordination_constraints'].append(dict(type='align_events', event_ids=events))
    if not scene['coordination_constraints']:
        scene['coordination_type']='independent'
    env = dict(units=dict(position='meters', time='seconds', angle='radians'),
        coordinate_system=dict(frame='shared simulator world frame within this scene', up_axis='z', heading='yaw counterclockwise from +x', quaternion='xyzw'),
        controller=dict(description='Fixed pretrained Stage1 SCT 26k policy with original weights and normalization. Stage2 learned coordination is disabled; LLM initial release is the only external timing gate.',
            scheduler_control='Only initial freeze release: enabling predecessors and independent agents at 0; followers and coupled members individually scheduled.',
            before_release='Held at initial pose by the simulator freeze adapter when a schedule is executed.',
            after_release='Policy acts autonomously without further scheduler freezing or PUSH start commands.',
            timing_calibration='Unavailable. No historical demonstrations, measured walking speed, or event durations are supplied.'),
        task_glossary=DESCRIPTIONS,
        field_definitions=dict(assigned_push_region='Initial logical region of the shared physical box assigned by the simulator; not its final goal.',
            object_goal_position_m='Intended final object center; not a measured future state.',
            support_intended_final_center_m='Enabling predecessor goal used to interpret the follower target.'),
        limitations=['Fresh reset snapshot only; no policy rollout or future observations.',
            'Coupled approach is a qualitative planning milestone, not an exact measured detector.',
            'Enabling follower interaction onset is a planning estimate; formal success is scored separately with the original evaluator.',
            'Provide numeric timing estimates, explain their basis, and mark uncalibrated estimates with low confidence.'])
    request = dict(environment_spec=env, current_scene=scene)
    messages = build_messages(request)
    scene_index(scene)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'request.json').write_text(json.dumps(request, indent=2))
    (output / 'user_prompt.json').write_text(json.dumps(json.loads(messages[1]['content']), indent=2))
    template = (ROOT / 'prompts/user_template.txt').read_text()
    fields = dict(UNITS_COORDINATE_FRAME_AND_FIELD_DEFINITIONS={k:v for k,v in env.items() if k!='task_glossary'},
        PLAIN_LANGUAGE_TASK_DEFINITIONS=env['task_glossary'], EXACT_READINESS_ENABLING_AND_COMPLETION_CONDITIONS=scene['event_definitions'],
        CURRENT_INITIAL_OBSERVATION_TASKS_TARGETS_AND_DEPENDENCIES_JSON=scene)
    for key, value in fields.items():
        template = template.replace('{{'+key+'}}', json.dumps(value, indent=2))
    (output / 'user_template_filled.txt').write_text(template)
    (output / 'provenance.json').write_text(json.dumps(dict(source_directory=str(folder.resolve()),
        capture=capture, checkpoint_sha256=manifest['sha256'], api_calls=0),indent=2))
    return request


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.folder, args.output)
