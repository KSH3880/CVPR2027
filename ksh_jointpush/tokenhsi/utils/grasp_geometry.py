ANCHOR_INSET = .15


def grasp_y(half_y, side, inset=ANCHOR_INSET):
    return (half_y - inset) * side
