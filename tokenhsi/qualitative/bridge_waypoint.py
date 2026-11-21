import argparse
import math

from utils.bridge_waypoint import bridge_goal, update_switch, waypoint_scene


def switch_margin(value):
    value = float(value)
    if not math.isfinite(value) or value < 0 or not (value * 2).is_integer():
        raise argparse.ArgumentTypeError('switch margin must be nonnegative, in 0.5 m increments')
    return value


