"""Formal evaluation settings. General scenes and Stage1-only single tasks differ."""
import torch

HORIZON = 900
TASK_NAMES = ('SIT', 'CLIMB', 'AT', 'ON_TOP', 'PUSH')
SINGLE_PRESETS = {'AT': ('holding_at', 1), 'ON_TOP': ('holding_ontop', 2),
                  'SIT': ('sit', 1), 'CLIMB': ('climb', 1), 'PUSH': ('push', 1)}
AT_SIZES = ((.22,)*3, (.27,)*3, (.32,)*3, (.37,)*3, (.42,)*3,
            (.47,)*3, (.52,)*3, (.57,)*3, (.30,.30,.40))


def configure(args, cfg, name):
    profile = getattr(args, 'eval_protocol', '')
    if not profile:
        return
    if not (args.eval and args.test and args.headless):
        raise ValueError('Formal evaluation requires --test --eval --headless')
    if args.cfg_env or args.num_agents or args.num_objects or args.episode_length:
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
