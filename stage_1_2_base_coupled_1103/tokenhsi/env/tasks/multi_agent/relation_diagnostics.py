"""TensorBoard naming for Stage1 graph, sampling and penalty metrics."""


def relation_tensorboard_tag(key):
    return "relation/" + key


def split_reward_tensorboard_tag(key):
    """Alias actual AT/CONTACT edge means, including their native gate/weights.

    These are means over the participating edges of each relation, not over all
    agents. In mixed tasks they need not sum to the overall task reward. AT also
    includes independent AT tasks; CONTACT only includes PUSH edges.
    """
    parts = key.split("/")
    if len(parts) != 3 or parts[0] != "edge" or parts[1] not in ("at", "contact"):
        return None
    component = {"state_component": "state", "progress_component": "progress",
                 "success_component": "success", "total": "total"}.get(parts[2])
    if component is not None:
        return "reward_terms/" + parts[1] + "_" + component
    return None
