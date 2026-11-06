"""Formal evaluation settings. General scenes and Stage1-only single tasks differ."""
import torch
import statistics

HORIZON = 900
TASK_NAMES = ('SIT', 'CLIMB', 'AT', 'ON_TOP', 'PUSH')
SINGLE_PRESETS = {'AT': ('holding_at', 1), 'ON_TOP': ('holding_ontop', 2),
                  'SIT': ('sit', 1), 'CLIMB': ('climb', 1), 'PUSH': ('push', 1)}
AT_SIZES = ((.22,)*3, (.27,)*3, (.32,)*3, (.37,)*3, (.42,)*3,
            (.47,)*3, (.52,)*3, (.57,)*3, (.30,.30,.40))
INTERACTION_METRICS = ('Independent Interaction Success Rate',
    'Enabling Interaction Group Success Rate', 'Coupled Interaction Group Success Rate',
    'Mixed Interaction Group Success Rate')


def configure(args, cfg, name):
    profile = getattr(args, 'eval_protocol', '')
    if not profile:
        return
    if not (args.eval and args.test and args.headless):
        raise ValueError('Formal evaluation requires --test --eval --headless')
    from utils.experiments import STAGE2_NAMES
    unified = name in STAGE2_NAMES
    if args.cfg_env or args.episode_length or ((args.num_agents or args.num_objects) and not unified):
        raise ValueError('Formal evaluation reads training settings; scene overrides are forbidden')
    if args.eval_repeats != 1 or args.seed not in (0, 1, 2):
        raise ValueError('Formal evaluation: one first episode, independent seed 0/1/2')
    if args.before_follower_near or args.before_follower_short or args.before_pickup_carry_rsi != 'loco' or args.task_role_swap:
        raise ValueError('Viewer overrides are forbidden in formal evaluation')
    if args.eval_skills not in ('', 'loco') or args.eval_skill_probs not in ('', '1.0'):
        raise ValueError('Formal evaluation requires loco')
    if profile == 'stage1_single':
        if name not in ('stage1', 'stage1_push') or args.eval_single_task not in SINGLE_PRESETS or (name == 'stage1' and args.eval_single_task == 'PUSH'):
            raise ValueError('Single-task evaluation is exclusive to Stage1')
        preset, objects = SINGLE_PRESETS[args.eval_single_task]
        cfg['env'].update(numAgents=1, numObjects=objects)
        args.task_graph = preset
    elif profile == 'general':
        if args.eval_single_task or args.task_graph != 'random_scenario':
            raise ValueError('General evaluation uses the training random_scenario sampler')
    else:
        raise ValueError('Unknown evaluation protocol')
    from utils.experiments import experiment_horizon
    cfg['env']['episodeLength'] = experiment_horizon(name)
    cfg['env']['eval'].update(skill=['loco'], skillInitProb=[1.0])


def single_at_sizes(n, device):
    if n < len(AT_SIZES):
        raise ValueError('AT single evaluation needs at least nine environments')
    ids = torch.cat((torch.arange(9, device=device), torch.randint(9, (n-9,), device=device)))
    return torch.tensor(AT_SIZES, device=device)[ids[torch.randperm(n, device=device)]]


def spatial_success(packet, roots, boxes, sizes, goals, dtype=torch.float64):
    """Float64 geometry, entirely independent of native reward success/gates."""
    from env.tasks.multi_agent.edge_ontop_reward import vertical_extent, inner_xy_region_error
    n, m, _ = packet.shape
    o = boxes.shape[1]
    valid, task, actor, payload, target = packet.long().unbind(-1)
    rows = torch.arange(n, device=boxes.device)[:, None]
    roots, boxes, sizes, goals = [x.to(dtype=dtype) for x in (roots, boxes, sizes, goals)]
    support = boxes[rows, (target-m).clamp(0, o-1)]
    support_size = sizes[rows, (target-m).clamp(0, o-1)]
    payload_idx = (payload-m).clamp(0, o-1)
    point = torch.where((task >= 2)[..., None], boxes[rows, payload_idx, :3],
                        roots[rows, actor.clamp(0, m-1), :3])
    goal = support[..., :3].clone()
    goal[..., 2] += vertical_extent(support[..., 3:7], support_size/2)
    offset = torch.where(task == 0, .10, torch.where(task == 1, .94, sizes[rows, payload_idx, 2]/2))
    goal[..., 2] += offset
    goal = torch.where(((task == 2) | (task == 4))[..., None], goals[rows, (target-m-o).clamp(0, m-1)], goal)
    point_ok = (point-goal).norm(dim=-1) <= .20
    region_ok = ((inner_xy_region_error(point[..., :2], support, support_size, .10) <= 0)
                 & ((point[..., 2]-goal[..., 2]).abs() <= .20))
    region_ok = torch.where((task == 2) | (task == 4), point_ok, region_ok)
    return torch.stack((point_ok & valid.bool(), region_ok & valid.bool()), -1)


def coupled_current_success(current, groups):
    same = groups[:, :, None] == groups[:, None, :]
    return (current[:, None] | ~same[..., None]).all(2)


def coupled_scores(groups, individual_ever, individual_final, group_ever, group_final):
    members = (groups[:, :, None] == groups[:, None, :]).sum(-1)
    leader = groups == torch.arange(groups.shape[1], device=groups.device)[None]
    result = {}
    for label, index in (('point', 0), ('region', 1)):
        result[label] = {}
        for size in range(2, max(4, groups.shape[1])+1):
            people = members == size
            selected = people & leader
            result[label][str(size)] = dict(
                groups=int(selected.sum()), people=int(people.sum()),
                individual_ever=int(individual_ever[..., index][people].sum()),
                individual_final=int(individual_final[..., index][people].sum()),
                simultaneous_ever=int(group_ever[..., index][selected].sum()),
                simultaneous_final=int(group_final[..., index][selected].sum()))
    return result


def shared_center_success(centers, goals, groups, valid, dtype=torch.float64):
    same = groups[:, :, None] == groups[:, None, :]
    active = (same & valid[:, None, :]).any(-1)
    distance = (centers.to(dtype)-goals.to(dtype)).norm(dim=-1)
    return (distance <= .20) & active & (same.sum(-1) > 1)


def shared_centers_from_tokens(packet, boxes, goals, groups):
    n, m = groups.shape
    rows = torch.arange(n, device=boxes.device)[:, None]
    payload = (packet[..., 3]-m).long().clamp(0, boxes.shape[1]-1)
    target = (packet[..., 4]-m-boxes.shape[1]).long().clamp(0, m-1)
    same = (groups[:, :, None] == groups[:, None, :]).to(torch.float64)
    count = same.sum(-1, keepdim=True)
    centers = same @ boxes[rows, payload, :3].double()/count
    center_goals = same @ goals[rows, target].double()/count
    return centers, center_goals


def shared_center_scores(groups, ever, final):
    members = (groups[:, :, None] == groups[:, None, :]).sum(-1)
    leader = groups == torch.arange(groups.shape[1], device=groups.device)[None]
    result = {}
    for size in range(2, max(4, groups.shape[1])+1):
        selected = (members == size) & leader
        count = int(selected.sum())
        row = dict(groups=count, ever_success=int(ever[selected].sum()),
                   final_success=int(final[selected].sum()))
        for metric in ('ever', 'final'):
            row[metric+'_rate'] = row[metric+'_success']/count if count else None
        result[str(size)] = row
    return result


def interaction_group_scores(valid, parent, groups, ever, center_ever):
    follower = parent >= 0
    enabling = follower.clone()
    rows, people = follower.nonzero(as_tuple=True)
    enabling[rows, parent[rows, people]] = True
    members = (groups[:, :, None] == groups[:, None, :]).sum(-1)
    coupled = (members > 1) & ~enabling
    leader = groups == torch.arange(groups.shape[1], device=groups.device)[None]
    independent = valid & ~enabling & ~coupled
    both = ever & ever[torch.arange(len(parent), device=parent.device)[:, None], parent.clamp_min(0)]
    result = {}
    for label, index in (('point', 0), ('region', 1)):
        values = (ever[..., index][independent], both[..., index][follower & valid],
                  center_ever[coupled & leader & valid])
        scores = {}
        for name, successes in zip(INTERACTION_METRICS, values):
            count, success = successes.numel(), int(successes.sum())
            scores[name] = dict(groups=count, success=success, rate=success/count if count else None)
        rates = [row['rate'] for row in scores.values()]
        scores[INTERACTION_METRICS[-1]] = dict(rate=sum(rates)/3 if None not in rates else None)
        result[label] = scores
    return result


def summarize_interaction_groups(runs):
    result = {}
    for label in ('point', 'region'):
        result[label] = {}
        for name in INTERACTION_METRICS:
            rows = [run[label][name] for run in runs]
            rates = [row['rate'] for row in rows]
            complete = bool(rates) and None not in rates
            item = dict(mean=statistics.mean(rates) if complete else None,
                std=statistics.stdev(rates) if complete and len(rates) > 1 else None, seeds=rates)
            if name != INTERACTION_METRICS[-1]:
                item.update(groups=sum(row['groups'] for row in rows),
                            successes=sum(row['success'] for row in rows))
            result[label][name] = item
    return result


def predecessors(graph, n, device):
    """Owner of the prerequisite AT edge for each follower; -1 for independent tasks."""
    from utils.edge_ontop_spec import batched
    result = torch.full((n, graph.num_agents), -1, dtype=torch.long, device=device)
    mask = graph.prereq_mask
    if mask.dim() == 2:
        mask = mask[None].expand(n, -1, -1)
    valid = batched(graph.edge_valid & graph.required_goal, n)
    owner = batched(graph.edge_owner, n)
    rows, edge = (valid & mask.any(-1)).nonzero(as_tuple=True)
    parent = mask[rows, edge].long().argmax(-1)
    result[rows, owner[rows, edge]] = owner[rows, parent]
    return result
