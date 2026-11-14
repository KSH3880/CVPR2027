import torch


def dependencies(packet):
    valid, task, actor, payload, target = packet.long().unbind(-1)
    return (valid.bool()[:, :, None] & valid.bool()[:, None, :]
        & ((task[:, :, None] == 2) | (task[:, :, None] == 3))
        & (task[:, None, :] == 3)
        & (payload[:, :, None] == target[:, None, :])
        & (actor[:, :, None] != actor[:, None, :]))


def alias_layout(packet, num_agents, num_objects):
    records = packet.reshape(len(packet), -1, 5).long()
    valid, task, actor, payload, target = records.unbind(-1)
    dependent = dependencies(records).any(1)
    objects = torch.stack((payload, target), -1)
    used = torch.stack((valid.bool() & (task >= 2), valid.bool()), -1)
    used &= (objects >= num_agents) & (objects < num_agents+num_objects)
    slots = (objects-num_agents).clamp(0, num_objects-1).flatten(1)
    counts = torch.zeros(len(packet), num_objects, device=packet.device, dtype=torch.long)
    counts.scatter_add_(1, slots, used.long().flatten(1))
    count = (counts > 0).sum(-1)
    rank = (dependent[:, None, :] & (actor[:, None, :] < actor[:, :, None])).sum(-1)
    return dependent, target, num_agents+count[:, None]+rank, count


def human_relations(packet, num_agents):
    records = packet.reshape(len(packet), -1, 5).long()
    owners = records[..., 0].bool()[..., None] & (
        records[..., 2, None] == torch.arange(num_agents, device=packet.device))
    predecessor = torch.einsum('nqi,nji,njk->nqk', owners.transpose(1, 2).float(),
                              dependencies(records).float(), owners.float()).bool()
    return torch.where(torch.eye(num_agents, device=packet.device, dtype=torch.bool),
                       1, torch.where(predecessor, 2, 0))


def goal_aligned_scene(scene, num_agents, num_objects):
    m, o = num_agents, num_objects
    packet = scene[:, -5*m:].reshape(len(scene), m, 5).long()
    result = scene.clone()
    poses = result[:, -(5*m+7*(2*m+o)):-5*m].reshape(len(scene), 2*m+o, 7)
    corners = scene[:, m*223:m*223+o*30].reshape(len(scene), o, 30)[..., 6:].reshape(len(scene), o, 8, 3)
    height = corners[..., 2].amax(-1)-corners[..., 2].amin(-1)
    expected = poses[:, m:m+o, :3].clone()
    rows, records = (packet[..., 0].bool() & (packet[..., 1] == 2)).nonzero(as_tuple=True)
    expected[rows, packet[rows, records, 3]-m] = poses[rows, packet[rows, records, 4], :3]
    rows, records = (packet[..., 0].bool() & (packet[..., 1] == 3)).nonzero(as_tuple=True)
    payload, support = packet[rows, records, 3]-m, packet[rows, records, 4]-m
    for _ in range(m-1):
        xyz = expected[rows, support].clone()
        xyz[:, 2] += (height[rows, support]+height[rows, payload])/2
        expected[rows, payload] = xyz
    dependent, target, alias, _ = alias_layout(packet, m, o)
    rows, records = dependent.nonzero(as_tuple=True)
    poses[rows, alias[rows, records], :3] = expected[rows, target[rows, records]-m]
    return result


def qualified_success(raw, graph):
    result = raw.clone()
    for _ in range(graph.num_agents):
        result = raw & (result[:, None] | ~graph.prereq_mask).all(-1)
    return result


def install():
    from isaacgym import gymapi, gymutil
    from env.tasks.multi_agent.humanoid_ma_before import HumanoidMABefore
    from env.tasks.multi_agent.humanoid_ma_before_push import HumanoidMABeforePush
    from env.tasks.multi_agent.scene_reset import SceneResetMixin
    from env.tasks.multi_agent.scene_validation import scene_infeasible
    from env.tasks.multi_agent import before_reward, before_wait
    from learning.multi_agent import stage2_push_network, human_coordination
    from utils import before_object_alias
    from utils.stage1_jointpush import compose_graph
    from utils.edge_ontop_spec import batched

    before_object_alias.alias_layout = alias_layout
    stage2_push_network.goal_aligned_scene = goal_aligned_scene
    human_coordination.human_relations = human_relations

    def graph_for(self, ids):
        templates = torch.tensor([3, 4, 4], device=self.device).expand(len(ids), -1).clone()
        pre = torch.tensor([-1, 0, 1], device=self.device).expand_as(templates).clone()
        graph = compose_graph(templates, num_objects=6)
        placement = graph.required_goal & graph.edge_valid
        edges = torch.stack([((graph.edge_owner == owner) & placement).long().argmax(-1)
                             for owner in range(3)], -1)
        rows = torch.arange(len(ids), device=self.device)
        for follower in (1, 2):
            graph.edge_dst[rows, edges[:, follower]] = 3+follower-1
            graph.prereq_mask[rows, edges[:, follower], edges[:, follower-1]] = True
        return graph, pre, templates

    HumanoidMABeforePush._draw_episode_graph = graph_for
    original_load = HumanoidMABeforePush._load_box_asset
    # Set physical dimensions before assets, scale and BPS are constructed.
    import inspect
    import textwrap
    source = textwrap.dedent(inspect.getsource(original_load))
    anchor = '    profiles=torch.tensor(self._joint[\'box\'][\'profiles\'],device=self.device)'
    if source.count(anchor) != 1:
        raise ValueError('Box asset loader changed')
    source = source.replace(anchor,
        "    self._box_size[:, :3] = self._box_size.new_tensor([[.4,.4,.25],[.4,.4,.25],[.3,.3,.25]])\n"+anchor)
    namespace = dict(original_load.__globals__)
    exec(compile(source, __file__, 'exec'), namespace)
    HumanoidMABeforePush._load_box_asset = namespace['_load_box_asset']

    def reset_boxes(self, ids):
        self._push_hits[ids] = False
        self._push_layout_failed[ids] = False
        HumanoidMABefore._reset_boxes(self, ids)
        desired = self._humanoid_root_states.new_tensor([[-4.,0.],[0.,-5.],[5.,0.]])
        delta = self._env_origins[ids, None, :2]+desired-self._humanoid_root_states[ids, :, :2]
        self._humanoid_root_states[ids, :, :2] += delta
        self._kinematic_humanoid_rigid_body_states[ids, :, :, :2] += delta[:, :, None]
        self._box_states[ids, :3, :2] = self._env_origins[ids, None, :2]+desired.new_tensor([[-2.,0.],[0.,-3.],[3.,0.]])
        self._box_states[ids, :3, 2] = self._box_size[ids, :3, 2]/2
        self._box_states[ids, :3, 3:13] = 0
        self._box_states[ids, :3, 6] = 1
        self._before_near_pairs[ids] = False
        self._before_distance_modes[ids] = 0

    def reset_task(self, ids):
        SceneResetMixin._reset_task(self, ids)
        self._tar_pos[ids, :, :2] = self._env_origins[ids, None, :2]
        heights = self._box_size[ids, :3, 2]
        self._tar_pos[ids, :, 2] = heights.cumsum(-1)-heights/2

    HumanoidMABeforePush._reset_boxes = reset_boxes
    HumanoidMABeforePush._reset_task = reset_task
    HumanoidMABeforePush._ontop_scene_infeasible = lambda self, ids: scene_infeasible(self, ids, margin=.001)

    original_reset = before_wait.BeforeWait.reset
    def wait_reset(self, ids, graph, raw_success):
        original_reset(self, ids, graph, raw_success)
        owner = batched(graph.edge_owner, len(self.armed))
        self.goal = owner[self.rows, self.predecessor]
    before_wait.BeforeWait.reset = wait_reset

    def progress_to_chain(progress, roots, objects, sizes, goals, graph, config,
                          raw_success=None, lambda_inner=0., radius_override=None):
        n, m, o = len(roots), graph.num_agents, graph.num_objects
        rel, src, dst, owner = [batched(getattr(graph, k), n) for k in
                               ('edge_relation','edge_src','edge_dst','edge_owner')]
        rows, edges = (graph.edge_valid & graph.prereq_mask.any(-1)).nonzero(as_tuple=True)
        pred = graph.prereq_mask[rows, edges].long().argmax(-1)
        goal = owner[rows, pred]
        distance = (objects[rows, src[rows, edges]-m, :2]-goals[rows, goal, :2]).norm(dim=-1)
        p = config['progress']
        radius = torch.full_like(distance, float(p.get('delta', 0.)))
        if 'bbox_buffers' in p:
            radius = sizes[rows, dst[rows, edges]-m, :2].norm(dim=-1)/2+p['bbox_buffers']['ON_TOP']
        if radius_override is not None:
            radius = torch.full_like(distance, radius_override)
        result = progress.clone()
        result[rows, edges] = 1/(1+(distance-radius).clamp_min(0)/p['sigma'])
        return result
    before_reward.predecessor_at_progress = progress_to_chain
    original_success = before_reward.BeforeRuntime._raw_success
    def chain_success(self, *args):
        return qualified_success(original_success(self, *args), args[-1])
    before_reward.BeforeRuntime._raw_success = chain_success

    original_camera = HumanoidMABeforePush._update_camera
    def camera(self):
        if not hasattr(self, '_humanoid_root_states'):
            return original_camera(self)
        if self.viewer is not None and self._camera_follow_flag:
            x, y = self._env_origins[0, :2].tolist()
            self.gym.viewer_camera_look_at(self.viewer, None,
                gymapi.Vec3(x+8,y-11,7), gymapi.Vec3(x,y-1,.7))
    HumanoidMABeforePush._update_camera = camera
    original_draw = HumanoidMABeforePush._draw_ontop_context
    def draw(self):
        original_draw(self)
        if self.viewer is None:
            return
        for support in (0, 1):
            color = self._agent_color(support+1)
            wire = gymutil.WireframeBoxGeometry(*self._box_size[0, support].tolist(),
                color=(color.x, color.y, color.z))
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*self._tar_pos[0, support].tolist())
            pose.r = gymapi.Quat(*self._box_states[0, support, 3:7].tolist())
            gymutil.draw_lines(wire, self.gym, self.viewer, self.envs[0], pose)
    HumanoidMABeforePush._draw_ontop_context = draw
