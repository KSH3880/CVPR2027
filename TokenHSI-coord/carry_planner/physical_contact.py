"""CPU-pipeline PhysX contact-pair measurement, excluding expected contacts."""
import numpy as np

KINDS = ("agent_agent", "agent_box", "box_box", "total")


def classify_contacts(contacts, owners, force_threshold=0.0):
    """Count positive-force contact pairs by physical actor ownership.

    owners maps DOMAIN_ENV rigid-body indices to (agent|box, physical index).
    Ground, self contact, own-box grasp, and other scene objects are excluded.
    lambda is documented by this Isaac Gym release as contact force magnitude.
    """
    hits = {key: False for key in KINDS}
    for contact in contacts:
        a, b = owners.get(int(contact["body0"])), owners.get(int(contact["body1"]))
        if a is None or b is None or not float(contact["lambda"]) > force_threshold:
            continue
        if a[0] == b[0]:
            if a[1] == b[1]:
                continue
            kind = "agent_agent" if a[0] == "agent" else "box_box"
        else:
            if a[1] == b[1]:
                continue
            kind = "agent_box"
        hits[kind] = True
        hits["total"] = True
    return hits


class PhysicalContactTracker:
    def __init__(self, task, force_threshold=0.0):
        from isaacgym import gymapi
        if task.gym.get_sim_params(task.sim).use_gpu_pipeline:
            raise ValueError("physical contact pairs require CPU pipeline")
        if not np.isfinite(force_threshold) or force_threshold < 0:
            raise ValueError("contact force threshold must be finite and nonnegative")
        self.task, self.threshold = task, force_threshold
        self.owners = []
        for env_id, env in enumerate(task.envs):
            owners = {}
            for kind, handles in (("agent", [task.gym.get_actor_handle(env, a) for a in range(2)]),
                                  ("box", task._box_handles[env_id])):
                for actor_id, handle in enumerate(handles):
                    for body in range(task.gym.get_actor_rigid_body_count(env, handle)):
                        index = task.gym.get_actor_rigid_body_index(env, handle, body, gymapi.DOMAIN_ENV)
                        owners[index] = (kind, actor_id)
            self.owners.append(owners)
        self.steps = {k: np.zeros(task.num_envs, dtype=np.int64) for k in KINDS}
        self.queries = np.zeros(task.num_envs, dtype=np.int64)
        self.raw_contacts = np.zeros(task.num_envs, dtype=np.int64)
        self.positive_contacts = np.zeros(task.num_envs, dtype=np.int64)

    def reset(self, env_ids):
        ids = env_ids.detach().cpu().numpy()
        for value in self.steps.values():
            value[ids] = 0
        self.queries[ids] = 0
        self.raw_contacts[ids] = 0
        self.positive_contacts[ids] = 0

    def sample(self):
        result = {k: np.zeros(self.task.num_envs, dtype=bool) for k in KINDS}
        for i, env in enumerate(self.task.envs):
            contacts = self.task.gym.get_env_rigid_contacts(env)
            self.queries[i] += 1
            self.raw_contacts[i] += len(contacts)
            self.positive_contacts[i] += int(np.sum(contacts['lambda'] > self.threshold))
            hits = classify_contacts(contacts, self.owners[i], self.threshold)
            for key in KINDS:
                result[key][i] = hits[key]
        return result

    def commit(self, hits):
        for key in KINDS:
            self.steps[key] += hits[key]
