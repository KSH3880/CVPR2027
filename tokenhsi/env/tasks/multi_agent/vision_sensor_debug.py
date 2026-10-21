import numpy as np
import torch


def draw_fan_sensor(gym, viewer, env, data, origin, resolution, uniform_rings=False, square=False):
    colors = np.array([[.95, .15, .85], [.1, .85, 1.], [1., .7, .05]], dtype=np.float32)
    for agent in range(len(data['eye'])):
        status = data['status'][agent]
        z = torch.where(status != 0, data['filled'][agent], data['eye'][agent, 2] + .15)
        points = torch.cat((data['xy'][agent], z[:, None] + .04), -1) - origin
        dx, dy = points.new_tensor([.055, 0, 0]), points.new_tensor([0, .055, 0])
        lines = torch.cat((torch.stack((points - dx, points + dx), 1),
                           torch.stack((points - dy, points + dy), 1)), 0).cpu().numpy()
        rgb = np.tile(colors[status.cpu().numpy()], (2, 1))
        gym.add_lines(viewer, env, len(lines), lines, rgb)
        if square:
            grid = points.reshape(resolution, resolution, 3)
            outline = torch.cat((grid[0], grid[1:, -1], grid[-1, :-1].flip(0),
                                 grid[1:-1, 0].flip(0), grid[:1, 0]))
            lines = torch.stack((outline[:-1], outline[1:]), 1).cpu().numpy()
        else:
            eye = data['eye'][agent] - origin
            arc = (points[(resolution - 1)**2:] if uniform_rings
                   else points.reshape(resolution, resolution, 3)[:, -1])
            boundary = torch.stack((eye.expand(2, -1), arc[[0, -1]]), 1)
            arc_lines = torch.stack((arc[:-1], arc[1:]), 1)
            lines = torch.cat((boundary, arc_lines), 0).cpu().numpy()
        rgb = np.tile(np.array([[.3, .5, 1.]], dtype=np.float32), (len(lines), 1))
        gym.add_lines(viewer, env, len(lines), lines, rgb)
