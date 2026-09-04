"""Path math for steerable carry. No isaacgym import: runnable and testable on CPU.

A path is a tensor [N, V, 2] of xy vertices, resampled to a fixed arc-length
spacing so that index j sits at arc length j * ds. Paths shorter than V vertices
are extended along their final direction, which keeps every tensor rectangular
and makes "past the end" mean "keep going straight" rather than "stop here".
"""

import torch
import math

DS = 0.1          # vertex spacing along a stored path
V = 320           # vertices per stored path (32 m)
# 32 m covers the worst full-task path: the target is sampled 1-10 m from the
# box, the character spawns a similar distance away, and bowing adds ~20 %.
# A path longer than V * DS would be truncated before the target, so
# path_stats reports the length and the caller is expected to watch it.


def _extend(pts, n_valid, total):
    """Extend each path to `total` vertices along its last direction."""
    N = pts.shape[0]
    idx = torch.arange(total, device=pts.device)
    last = (n_valid - 1).clamp(min=1)
    tail_dir = pts[torch.arange(N), last] - pts[torch.arange(N), last - 1]
    tail_dir = tail_dir / tail_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    out = pts[:, :total].clone() if pts.shape[1] >= total else torch.cat(
        [pts, pts[:, -1:].expand(N, total - pts.shape[1], 2)], dim=1)
    beyond = idx[None, :] >= n_valid[:, None]
    steps = (idx[None, :] - (n_valid[:, None] - 1)).clamp(min=0).float()
    anchor = pts[torch.arange(N), last]
    ext = anchor[:, None, :] + steps[..., None] * DS * tail_dir[:, None, :]
    return torch.where(beyond[..., None], ext, out)


def resample(waypoints, ds=DS, total=V, with_end=False):
    """Arc-length resample a polyline [N, M, 2] to [N, total, 2] at spacing ds."""
    N, M, _ = waypoints.shape
    seg = waypoints[:, 1:] - waypoints[:, :-1]
    seg_len = seg.norm(dim=-1)
    cum = torch.cat([torch.zeros(N, 1, device=waypoints.device),
                     seg_len.cumsum(dim=-1)], dim=-1)
    total_len = cum[:, -1]

    s = torch.arange(total, device=waypoints.device, dtype=waypoints.dtype) * ds
    s = s[None, :].expand(N, total)
    j = torch.searchsorted(cum.contiguous(), s.contiguous().clamp(max=total_len[:, None] - 1e-6))
    j = (j - 1).clamp(0, M - 2)

    ar = torch.arange(N, device=waypoints.device)[:, None]
    seg_start = cum.gather(1, j)
    frac = ((s - seg_start) / seg_len.gather(1, j).clamp(min=1e-6)).clamp(0, 1)
    p0 = waypoints[ar, j]
    p1 = waypoints[ar, j + 1]
    out = p0 + frac[..., None] * (p1 - p0)

    n_valid = (total_len / ds).long().clamp(min=2, max=total)
    # **실제 경로가 어디서 끝나는지.** 뒤쪽은 직선 외삽이라 이 값이 없으면 래칫이
    # 그리로 미끄러져도 막을 수 없다 (실측: 경로 10.5 m 인데 호길이가 17.9 m 까지 갔다).
    # 기본 반환은 그대로라 기존 호출부는 영향이 없다.
    ext = _extend(out, n_valid, total)
    return (ext, n_valid) if with_end else ext


def project(root_xy, path, s_lo=None, s_hi=None):
    """Nearest point on the path.

    Returns arc length s (m), signed lateral offset (left positive), and the
    unit tangent there. Batched over N with no python loop.

    s_lo / s_hi restrict the search to one leg. A full-task path doubles back
    near the box -- the approach leg arrives there and the transport leg leaves
    -- so an unrestricted nearest-point search can snap the carried box onto the
    approach leg and run the arc length backwards. Bounds are per-env tensors.

    The search is exhaustive over all V-1 segments, which looks wasteful at six
    calls a step but is not worth optimising. A coarse-to-fine variant (scan
    every 8th segment, refine around the winner) was tried and reverted: it was
    WRONG on 0.5-1.5 % of envs, by up to 12 m, because a whole-task path doubles
    back at the box so distance-vs-arc-length is not unimodal and the coarse pass
    lands in the wrong basin. It was also 2.4x SLOWER, the gathers costing more
    than the dense arithmetic they were meant to avoid.
    """
    a = path[:, :-1]
    b = path[:, 1:]
    ab = b - a
    ap = root_xy[:, None, :] - a
    t = (ap * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-12)
    t = t.clamp(0, 1)
    proj = a + t[..., None] * ab
    d = (root_xy[:, None, :] - proj).norm(dim=-1)

    if s_lo is not None or s_hi is not None:
        seg_s = torch.arange(a.shape[1], device=path.device, dtype=path.dtype) * DS
        bad = torch.zeros_like(d, dtype=torch.bool)
        if s_hi is not None:
            bad |= seg_s[None, :] > s_hi[:, None]
        if s_lo is not None:
            bad |= (seg_s[None, :] + DS) < s_lo[:, None]
        bad &= ~bad.all(dim=1, keepdim=True)   # never mask a row into nothing
        d = d.masked_fill(bad, 1e9)

    k = d.argmin(dim=1)
    ar = torch.arange(path.shape[0], device=path.device)
    s = (k.float() + t[ar, k]) * DS

    tang = ab[ar, k]
    tang = tang / tang.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    rel = root_xy - proj[ar, k]
    lateral = tang[:, 0] * rel[:, 1] - tang[:, 1] * rel[:, 0]
    return s, lateral, tang


def window(path, s, k, spacing):
    """The k command points ahead of arc length s, spaced `spacing` apart."""
    offs = torch.arange(1, k + 1, device=path.device, dtype=path.dtype) * spacing
    q = (s[:, None] + offs[None, :]) / DS
    lo = q.floor().long().clamp(0, V - 2)
    frac = (q - lo.float())[..., None]
    ar = torch.arange(path.shape[0], device=path.device)[:, None]
    return path[ar, lo] + frac * (path[ar, lo + 1] - path[ar, lo])


def to_local(pts_xy, root_xy, heading):
    """World xy [N, K, 2] into the root's yaw frame."""
    c = torch.cos(-heading)[:, None]
    s = torch.sin(-heading)[:, None]
    d = pts_xy - root_xy[:, None, :]
    return torch.stack([c * d[..., 0] - s * d[..., 1],
                        s * d[..., 0] + c * d[..., 1]], dim=-1)


def max_turn_deg(path, over=1.5):
    """Largest heading change over any `over` metres of arc, in degrees."""
    seg = path[:, 1:] - path[:, :-1]
    head = torch.atan2(seg[..., 1], seg[..., 0])
    span = max(int(round(over / DS)), 1)
    d = head[:, span:] - head[:, :-span]
    d = (d + math.pi) % (2 * math.pi) - math.pi
    return d.abs().max(dim=1).values * 180.0 / math.pi


def gen_gt(start_xy, target_xy, curvature, seed):
    """Transport path: start -> target with a single lateral bulge.

    One leg only. A root -> box -> target path would need a median 133-degree
    corner at the box, which no carried-box turn rate can follow; the approach
    leg is left to the pretrained policy and steering applies to transport.
    curvature is the bulge height as a fraction of the straight-line length.
    """
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    N = start_xy.shape[0]
    d = target_xy - start_xy
    L = d.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    n = torch.stack([-d[:, 1], d[:, 0]], dim=-1) / L
    sign = (torch.randint(0, 2, (N, 1), generator=g).to(start_xy) * 2 - 1)
    ts = torch.linspace(0, 1, 17, device=start_xy.device)[None, :, None]
    bulge = torch.sin(ts * math.pi) * curvature * L[:, None, :] * sign[:, None, :]
    pts = start_xy[:, None, :] + ts * d[:, None, :] + bulge * n[:, None, :]
    return resample(pts)


def _bow(a, b, curvature, g, modes, n):
    """One leg from a to b, bowed sideways by a random sine series.

    offset(t) = L * sum_m  c_m sin(m pi t),   c_m ~ U(-1, 1) / m

    Every mode vanishes at t=0 and t=1, so both endpoints are hit exactly no
    matter what is drawn. That is the whole reason for the sine basis: the path
    must start at the character, pass through the box, and end at the target,
    and those three are not negotiable. 1/m keeps the high modes small so the
    curve wiggles rather than folds.

    The series is then rescaled so its peak equals curvature * L * U(0.2, 1),
    which pins down what `curvature` means: the largest sideways departure from
    the straight line, as a fraction of leg length. Without that rescale the
    summed coefficients have a fat tail -- a single draw can bow a leg by most
    of its own length, tripling the path and running it off the stored buffer.
    """
    N = a.shape[0]
    d = b - a
    L = d.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    nrm = torch.stack([-d[:, 1], d[:, 0]], dim=-1) / L
    ts = torch.linspace(0, 1, n, device=a.device, dtype=a.dtype)[None, :, None]
    off = torch.zeros(N, n, 1, device=a.device, dtype=a.dtype)
    for m in range(1, modes + 1):
        cm = (torch.rand(N, 1, 1, generator=g).to(a) * 2 - 1) / m
        off = off + cm * torch.sin(m * math.pi * ts)
    peak = off.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    want = curvature * (0.2 + 0.8 * torch.rand(N, 1, 1, generator=g).to(a))
    off = off / peak * want
    return a[:, None, :] + ts * d[:, None, :] + off * L[:, None, :] * nrm[:, None, :]


def gen_full(root_xy, box_xy, tar_xy, curvature, seed, modes=3, n=33):
    # modes controls how wiggly a path can be: one mode is a single bow,
    # more modes add shorter-wavelength detours at 1/m amplitude.
    """Whole-task path: character -> box -> placement target.

    Returns (path [N, V, 2], s_box [N]) where s_box is the arc length at which
    the box sits, i.e. the boundary between the approach leg and the transport
    leg. No turn limit is applied: the character stops at the box to pick it up,
    so an arbitrarily sharp corner there costs nothing to follow.
    """
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    leg1 = _bow(root_xy, box_xy, curvature, g, modes, n)
    leg2 = _bow(box_xy, tar_xy, curvature, g, modes, n)
    pts = torch.cat([leg1, leg2[:, 1:]], dim=1)
    s_box = (leg1[:, 1:] - leg1[:, :-1]).norm(dim=-1).sum(dim=-1)

    # A path longer than the stored buffer is silently cut short of the target
    # by resample, which looks like a policy that stops early rather than a
    # generator that ran out of room. Curvature 0.15 peaks near 25 m against a
    # 32 m buffer, but 0.25 already reaches it, so this has to be loud.
    total = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1).sum(dim=-1)
    over = total > V * DS
    if over.any():
        print(f"[steer_path] WARNING: {int(over.sum())}/{len(over)} paths exceed the "
              f"{V * DS:.0f} m buffer (longest {total.max():.1f} m) and are truncated "
              f"before the target. Raise V.", flush=True)
    return resample(pts), s_box


def _bow_v2(a, b, g, lat_min, lat_max, turn_max_deg, n=65, iters=7, p_two=0.0,
            skew=0.0, spread=(1.0, 1.0), lat_frac=0.0):
    """One leg bowed by one -- or with probability p_two, two -- sine humps.

    Two properties matter and mode 1 is the only choice that gets both:

    * It is the flattest way to reach a given sideways offset. At 0.8 m of bow
      on a 6 m leg, one mode turns 18 deg per 1.5 m where three modes turn 51.
    * It is the most distinguishable from going straight. Higher modes flip sign
      several times, so a straight line sits near their average -- which is why
      the no-steering control scored only 0.05 m worse than the steered policy
      on the old three-mode paths. One hump leaves the straight line 0.51 m off
      instead of 0.39, and the gap grows with the bow.

    The size is drawn in metres, not as a fraction of leg length, because a
    detour is set by what is being avoided, not by how far there is left to go.
    Short legs would then turn very sharply for the same detour (1.5 m over 3 m
    of leg is 89 deg), so the amplitude is shrunk until the turn fits under
    turn_max_deg. Shrinking only -- growing a leg into its cap would invent
    curvature the draw did not ask for.

    p_two mixes in sin(2*pi*t), an S weaving past two obstacles rather than
    around one. It stays distinguishable from a straight line because it only
    changes side once; three humps change side twice and that is precisely the
    v1 shape a straight walk scored almost as well on. It also needs about twice
    the turn for the same offset, so turn_max_deg shrinks it on its own.
    """
    N = a.shape[0]
    d = b - a
    L = d.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    nrm = torch.stack([-d[:, 1], d[:, 0]], dim=-1) / L
    ts = torch.linspace(0, 1, n, device=a.device, dtype=a.dtype)[None, :, None]
    modes = torch.where(torch.rand(N, 1, 1, generator=g).to(a) < p_two, 2.0, 1.0)
    base = torch.sin(math.pi * modes * ts)
    if spread != (1.0, 1.0):
        # |sin|^p keeps the crest height (everything is renormalised below) and
        # only changes how the turning is distributed: p > 1 concentrates it at
        # the crest, p < 1 spreads it toward the ends. Below about 0.85 the
        # slope at t=0,1 blows up and the leg leaves the character -- and
        # arrives at the box -- at a sharp angle, which is why the low end of
        # the range is clamped rather than left to the caller.
        lo, hi = max(spread[0], 0.85), max(spread[1], 0.85)
        pw = lo + (hi - lo) * torch.rand(N, 1, 1, generator=g).to(a)
        base = base.sign() * base.abs().clamp(min=1e-9) ** pw
    hump = base
    if skew > 0:
        # Move the crest off centre. sin(pi t)*(1 + c cos(pi t)) is mode 1 plus
        # c/2 of mode 2, so it still vanishes at both ends -- the three points
        # stay exact -- and |c| < 1 keeps the offset single-signed, one bulge.
        # Without this the crest sits at t=0.5 in every leg ever generated, so
        # two paths with the same amplitude have the same shape.
        c = (torch.rand(N, 1, 1, generator=g).to(a) * 2 - 1) * skew
        hump = hump * (1.0 + c * torch.cos(math.pi * modes * ts))
    hump = hump / hump.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    sign = (torch.randint(0, 2, (N, 1, 1), generator=g).to(a) * 2 - 1)
    hi = torch.full_like(L, lat_max)
    if lat_frac > 0:
        # A bow drawn in metres means something different on a 1 m leg than on a
        # 9 m one: legs are 1-10.6 m here and 28 % are under 4 m, so a 2.2 m bow
        # on a short leg asks the character to reverse direction inside 1.5 m.
        # Capping the DRAW by leg length fixes that before any shape is built,
        # and it reads only the leg length -- spread and skew are untouched,
        # which is the whole reason for doing it here instead of by turn rate.
        hi = torch.minimum(hi, lat_frac * L)
    lo = torch.minimum(torch.full_like(hi, lat_min), hi)
    amp = (lo + (hi - lo) * torch.rand(N, 1, generator=g).to(a))[:, :, None] * sign

    def pts(A):
        return a[:, None, :] + ts * d[:, None, :] + (hump * A) * nrm[:, None, :]

    for _ in range(iters):
        turn = max_turn_deg(resample(pts(amp)), 1.5).clamp(min=0.5)
        over = (turn / turn_max_deg).clamp(min=1.0)          # shrink only
        amp = amp / over[:, None, None]
    return pts(amp)


def gen_full_v2(root_xy, box_xy, tar_xy, seed, lat_min=0.3, lat_max=1.5,
                turn_max_deg=35.0, p_two=0.0, skew=0.0, spread=(1.0, 1.0),
                lat_frac=0.0, with_end=False):
    """Whole-task path whose legs bow once, by a distance drawn in metres."""
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    leg1 = _bow_v2(root_xy, box_xy, g, lat_min, lat_max, turn_max_deg, p_two=p_two,
                   skew=skew, spread=spread, lat_frac=lat_frac)
    leg2 = _bow_v2(box_xy, tar_xy, g, lat_min, lat_max, turn_max_deg, p_two=p_two,
                   skew=skew, spread=spread, lat_frac=lat_frac)
    pts = torch.cat([leg1, leg2[:, 1:]], dim=1)
    s_box = (leg1[:, 1:] - leg1[:, :-1]).norm(dim=-1).sum(dim=-1)
    total = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1).sum(dim=-1)
    over = total > V * DS
    if over.any():
        print(f"[steer_path] WARNING: {int(over.sum())}/{len(over)} paths exceed the "
              f"{V * DS:.0f} m buffer (longest {total.max():.1f} m). Raise V.", flush=True)
    if with_end:
        path, n_valid = resample(pts, with_end=True)
        return path, s_box, n_valid
    return resample(pts), s_box


def path_stats(path, s_box=None, corner_skip=0.8):
    """Diagnostics for one batch of paths: length, sharpness, corner angle.

    Measurement only -- nothing here shapes a path. The point is to learn what
    the character can actually follow before deciding whether to constrain the
    generator, rather than guessing a limit up front.

    The turn at the box is reported on its own and excluded from the leg
    statistics. It is the one place the path is allowed to be arbitrarily sharp
    (the character stops there to pick the box up), and left in, it swamps every
    other number: an unmasked median turn of 85 deg per 1.5 m is just the corner,
    and says nothing about whether the legs are walkable.
    """
    seg = path[:, 1:] - path[:, :-1]
    head = torch.atan2(seg[..., 1], seg[..., 0])
    out = {}
    if s_box is not None:
        out["s_box_m"] = s_box
        arc = torch.arange(head.shape[1], device=path.device, dtype=path.dtype) * DS
        near = (arc[None, :] - s_box[:, None]).abs() < corner_skip
        i = (s_box / DS).long().clamp(1, head.shape[1] - 2)
        ar = torch.arange(path.shape[0], device=path.device)
        turn = head[ar, i + 1] - head[ar, i - 1]
        out["corner_deg"] = ((turn + math.pi) % (2 * math.pi) - math.pi).abs() * 180 / math.pi
    else:
        near = torch.zeros_like(head, dtype=torch.bool)

    for over in (1.5, 0.5):
        span = max(int(round(over / DS)), 1)
        d = head[:, span:] - head[:, :-span]
        d = ((d + math.pi) % (2 * math.pi) - math.pi).abs()
        bad = near[:, span:] | near[:, :-span]
        out[f"turn_{over}m_deg"] = d.masked_fill(bad, 0.0).max(dim=1).values * 180 / math.pi

    d = (head[:, 1:] - head[:, :-1] + math.pi) % (2 * math.pi) - math.pi
    bad = near[:, 1:] | near[:, :-1]
    out["max_curv_1/m"] = (d.abs() / DS).masked_fill(bad, 0.0).max(dim=1).values
    return out


def synth_step(gt, s, radius, sigma, seed):
    """Per-step command path: identical to gt out to `radius`, different beyond.

    Mimics a world model regenerating the trajectory every step. With
    radius >= the observation horizon the emitted window is bit-identical to
    gt's, which is exactly the property we want to hold.
    """
    if sigma <= 0:
        return gt
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    N = gt.shape[0]
    seg = gt[:, 1:] - gt[:, :-1]
    head = torch.atan2(seg[..., 1], seg[..., 0])

    arc = torch.arange(head.shape[1], device=gt.device, dtype=gt.dtype) * DS
    past = (arc[None, :] - (s[:, None] + radius)).clamp(min=0)
    beyond = past > 0
    ramp = (past / 1.0).clamp(max=1.0)
    amp = (torch.rand(N, 1, generator=g).to(gt) * 2 - 1) * sigma
    head = head + amp * ramp * (math.pi / 4)

    # inside the agreement radius keep gt's own steps, so the emitted window is
    # bit-identical to gt's whenever radius >= the observation horizon
    step = torch.stack([torch.cos(head), torch.sin(head)], dim=-1) * DS
    step = torch.where(beyond[..., None], step, seg)
    out = torch.cat([gt[:, :1], gt[:, :1] + step.cumsum(dim=1)], dim=1)
    keep = torch.cat([torch.zeros(N, 1, dtype=torch.bool, device=gt.device), beyond], dim=1)
    return torch.where(keep[..., None], out, gt)


# --- time-indexed trajectory ------------------------------------------------
#
# The arc-length window (`window` above) is anchored to where the body already
# is, so it can only ever say "1.2 m further along". It cannot say "stand still
# for two seconds", "take this corner slowly", or "be here at t = 5 s" -- the
# command and the body's own progress are the same number.
#
# tau[t] is where the body should be AT STEP t, independent of where it is. That
# makes speed a property of the command (point spacing) instead of a constant in
# the reward, and a stop is simply a repeated point.
#
# Measured (2026-08-19, 505k frames): the character CAN hold 1.5 m/s through a
# 46 deg/m bend -- it just cannot track while doing it. Deviation there is 0.184
# at 1.4-1.6 m/s, 0.080 at 0.8-1.2, and 0.044 below 0.8, i.e. the same as a
# straight. So the curve fix is a speed command, not a feasibility limit.

def curvature(path):
    """Local curvature at every vertex, rad/m, measured over 0.6 m of arc."""
    n = path.shape[1]
    i = torch.arange(n, device=path.device).clamp(3, n - 4)
    u1 = path[:, i] - path[:, (i - 3).clamp(0, n - 1)]
    u2 = path[:, (i + 3).clamp(0, n - 1)] - path[:, i]
    cr = u1[..., 0] * u2[..., 1] - u1[..., 1] * u2[..., 0]
    dp = (u1 * u2).sum(-1)
    return torch.atan2(cr.abs(), dp) / (6 * DS)


def build_tau(path, H, dt, v_nom=1.5, g_kappa=0.0, v_min=0.4,
              stops=None, stop_len=0, end=None):
    """Resample `path` into H points spaced by time instead of arc length.

    Vectorised over envs and closed-form in t: the time to cross each 0.1 m
    segment is DS / v(s), so the cumulative sum of those is the arrival time at
    every vertex, and inverting it with searchsorted gives the point for any t.
    A stop is a constant added to every arrival time past its arc index, which
    makes the inversion return the same point for that many steps -- no branch,
    and the reward reads a zero commanded velocity for free.

    g_kappa = 0 gives uniform v_nom, which reproduces the arc-length behaviour
    exactly and is what stage 1 regresses against.

    `end` is the vertex the target sits on, and tau HOLDS there. Without it the
    command runs off the end of the task: the stored path is 320 vertices (32 m)
    while a real one reaches its target after 11.4 m on median, and a 600-step
    episode at 1.5 m/s asks for 30 m. The policy then tracks the command
    faithfully straight past the target and never puts the box down -- measured
    at deviation 0.041 (the best ever recorded) with delivery 0.035. Holding at
    the target reuses the stop machinery: repeated points mean a commanded speed
    of zero, i.e. "arrive and stay".
    """
    N, V_, _ = path.shape
    v = torch.full((N, V_), v_nom, device=path.device)
    if g_kappa > 0:
        v = (v_nom / (1.0 + g_kappa * curvature(path))).clamp(v_min, v_nom)
    arrive = torch.cumsum(DS / v, dim=1) - DS / v[:, :1]      # arrival time per vertex
    t = torch.arange(H, device=path.device, dtype=path.dtype)[None, :] * dt
    if stops is not None and stop_len > 0:
        # Freeze the clock, do not delay it. Adding a constant to the arrival
        # times leaves a single long segment that searchsorted then interpolates
        # ACROSS -- a crawl, not a stop. Subtracting the elapsed hold from t
        # holds the sampled arc length exactly still, so tau repeats its point
        # and the reward reads a commanded speed of zero.
        hold = stop_len * dt
        for j in range(stops.shape[1]):
            t_stop = arrive.gather(1, stops[:, j:j + 1].clamp(0, V_ - 1))
            t = t - (t - t_stop).clamp(0.0, hold)
    k = torch.searchsorted(arrive.contiguous(), t.expand(N, H).contiguous())
    k = k.clamp(1, V_ - 1)
    if end is not None:
        k = torch.minimum(k, end[:, None].clamp(1, V_ - 1))
    t0, t1 = arrive.gather(1, k - 1), arrive.gather(1, k)
    w = ((t - t0) / (t1 - t0).clamp(min=1e-6)).clamp(0, 1)[..., None]
    ar = torch.arange(N, device=path.device)[:, None]
    return path[ar, k - 1] + w * (path[ar, k] - path[ar, k - 1])
