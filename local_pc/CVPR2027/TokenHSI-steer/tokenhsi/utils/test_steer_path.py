"""CPU unit tests for steer_path. Run: python tokenhsi/utils/test_steer_path.py"""

import math
import torch
import steer_path as sp

torch.manual_seed(0)
ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print("  PASS  " + name)
    else:
        fail += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def straight_path(N=4, ang=0.0):
    s = torch.arange(sp.V, dtype=torch.float32) * sp.DS
    d = torch.tensor([math.cos(ang), math.sin(ang)])
    return (s[None, :, None] * d[None, None, :]).expand(N, sp.V, 2).clone()


print("resample")
wp = torch.tensor([[[0., 0.], [3., 0.], [3., 4.]]])
p = sp.resample(wp)
seg = (p[:, 1:] - p[:, :-1]).norm(dim=-1)
check("spacing is DS everywhere", torch.allclose(seg, torch.full_like(seg, sp.DS), atol=1e-4),
      f"max dev {(seg - sp.DS).abs().max():.2e}")
check("starts at first vertex", torch.allclose(p[0, 0], wp[0, 0], atol=1e-5))
check("passes through the corner", (p[0] - wp[0, 1]).norm(dim=-1).min() < 0.06)
check("extends straight past the end", torch.allclose(
    (p[0, -1] - p[0, -2]), (p[0, -2] - p[0, -3]), atol=1e-4))

print("project")
path = straight_path()
root = torch.tensor([[2.0, 0.5], [2.0, -0.5], [0.0, 0.0], [5.0, 0.0]])
s, lat, tang = sp.project(root, path)
check("arc length", torch.allclose(s, torch.tensor([2.0, 2.0, 0.0, 5.0]), atol=1e-3), f"{s}")
check("lateral sign: left positive", lat[0] > 0 and lat[1] < 0, f"{lat}")
check("lateral magnitude", torch.allclose(lat.abs()[:2], torch.tensor([0.5, 0.5]), atol=1e-3))
check("tangent is +x", torch.allclose(tang, torch.tensor([1.0, 0.0]).expand(4, 2), atol=1e-4))

diag = straight_path(ang=math.pi / 4)
s2, lat2, t2 = sp.project(torch.tensor([[1.0, 1.0]]), diag[:1])
check("diagonal arc length", abs(s2.item() - math.sqrt(2)) < 1e-2, f"{s2.item():.4f}")
check("diagonal on-path lateral ~0", lat2.abs().item() < 1e-3)

print("window")
w = sp.window(path, torch.zeros(4), 6, 0.4)
check("window shape", tuple(w.shape) == (4, 6, 2))
check("first point at spacing", abs(w[0, 0, 0].item() - 0.4) < 1e-3, f"{w[0,0,0]}")
check("last point at k*spacing", abs(w[0, 5, 0].item() - 2.4) < 1e-3)
w2 = sp.window(path, torch.full((4,), 3.0), 6, 0.4)
check("window slides with s", abs(w2[0, 0, 0].item() - 3.4) < 1e-3)

print("to_local")
pts = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
loc = sp.to_local(pts, torch.zeros(1, 2), torch.tensor([math.pi / 2]))
check("yaw rotation", torch.allclose(loc[0], torch.tensor([[0.0, -1.0], [1.0, 0.0]]), atol=1e-5),
      f"{loc[0]}")
loc2 = sp.to_local(pts, torch.tensor([[1.0, 0.0]]), torch.zeros(1))
check("translation", torch.allclose(loc2[0, 0], torch.zeros(2), atol=1e-6))

print("gen_gt")
box = torch.tensor([[3.0, 0.0], [1.0, 4.0]])
tar = torch.tensor([[6.0, 2.0], [-2.0, 4.0]])
gt = sp.gen_gt(box, tar, curvature=0.15, seed=0)
check("gt shape", tuple(gt.shape) == (2, sp.V, 2))
seg = (gt[:, 1:] - gt[:, :-1]).norm(dim=-1)
check("gt spacing is DS", (seg - sp.DS).abs().max() < 1e-3, f"{(seg - sp.DS).abs().max():.2e}")
check("curvature 0.05 stays inside the 46 deg limit",
      sp.max_turn_deg(sp.gen_gt(box, tar, 0.05, 0)).max() < 46)
turns = [sp.max_turn_deg(sp.gen_gt(box, tar, c, 0)).median().item()
         for c in (0.0, 0.05, 0.10, 0.20)]
check("turn angle rises monotonically with curvature",
      all(turns[i] < turns[i + 1] for i in range(3)), f"{[round(t) for t in turns]}")
check("gt starts at the box", (gt[:, 0] - box).norm(dim=-1).max() < 1e-4)
d_tar = (gt - tar[:, None, :]).norm(dim=-1).min(dim=1).values
check("gt reaches the target", d_tar.max() < 0.1, f"{d_tar}")


print("synth_step  (the invariance property)")
gt = sp.gen_gt(box, tar, curvature=0.2, seed=1)
s0 = torch.zeros(2)
K, SPACING = 6, 0.4
L = K * SPACING
for R, expect_same in ((L + 1.5, True), (L + 0.1, True), (0.5, False)):
    a = sp.synth_step(gt, s0, radius=R, sigma=1.0, seed=11)
    b = sp.synth_step(gt, s0, radius=R, sigma=1.0, seed=22)
    wa = sp.window(a, s0, K, SPACING)
    wb = sp.window(b, s0, K, SPACING)
    same = torch.allclose(wa, wb, atol=1e-6)
    check(f"R={R:.1f} (L={L:.1f}) window identical across far-field seeds = {expect_same}",
          same == expect_same, f"max diff {(wa - wb).abs().max():.4f}")

a = sp.synth_step(gt, s0, radius=L + 1.0, sigma=1.0, seed=7)
check("window matches the GT window when R >= L",
      torch.allclose(sp.window(a, s0, K, SPACING), sp.window(gt, s0, K, SPACING), atol=1e-6))
check("but the far field really did move",
      (a - gt).norm(dim=-1).max() > 0.2, f"{(a - gt).norm(dim=-1).max():.3f}")

print("batched consistency")
N = 4096
big = straight_path(N)
r = torch.randn(N, 2) * 2
s, lat, tang = sp.project(r, big)
check("no nans at scale", torch.isfinite(s).all() and torch.isfinite(lat).all())
check("lateral equals |y| on a straight x-axis path",
      torch.allclose(lat, r[:, 1], atol=1e-3), f"max {(lat - r[:,1]).abs().max():.2e}")

print("gen_full  (the whole task: character -> box -> target)")
N = 512
root = torch.randn(N, 2) * 3
box = root + torch.randn(N, 2) * 2 + torch.tensor([4.0, 0.0])
tgt = box + torch.randn(N, 2) * 2 + torch.tensor([5.0, 0.0])
full, s_box = sp.gen_full(root, box, tgt, curvature=0.15, seed=1)

TOL = sp.DS * 1.5   # vertices sit on a DS grid, so "hits the point" means within one cell
check("starts at the character", (full[:, 0] - root).norm(dim=-1).max() < 1e-3,
      f"max {(full[:, 0] - root).norm(dim=-1).max():.2e}")
d_box = (full - box[:, None, :]).norm(dim=-1).min(dim=1).values
check("passes through the box", d_box.max() < TOL, f"max {d_box.max():.3f}")
d_tar = (full - tgt[:, None, :]).norm(dim=-1).min(dim=1).values
check("reaches the target", d_tar.max() < TOL, f"max {d_tar.max():.3f}")

# s_box must be the arc length where the box sits, since every phase switch
# and both projection masks are keyed off it
at_s = sp.window(full, s_box - sp.DS, 1, sp.DS)[:, 0]
check("s_box lands on the box", (at_s - box).norm(dim=-1).max() < TOL,
      f"max {(at_s - box).norm(dim=-1).max():.3f}")

# the true path ends at the target; everything past that is _extend padding, so
# measure the real length as s_box plus the transport leg's own arc
seg = (full[:, 1:] - full[:, :-1]).norm(dim=-1)
i_tar = (full - tgt[:, None, :]).norm(dim=-1).argmin(dim=1)
real_len = torch.stack([seg[j, :i_tar[j]].sum() for j in range(N)])
check("real path length fits inside the stored buffer",
      real_len.max() < sp.V * sp.DS, f"longest {real_len.max():.1f} m of {sp.V * sp.DS:.1f} m")

# controlled geometry: legs long enough that a 0.5 m turn window fits inside one
r2 = torch.zeros(64, 2)
b2 = torch.stack([torch.full((64,), 8.0), torch.zeros(64)], -1)
t2 = torch.stack([torch.full((64,), 8.0), torch.full((64,), 8.0)], -1)
n1 = 60   # 6 m, safely inside the 8 m approach leg
straight = sp.gen_full(r2, b2, t2, 0.0, seed=1)[0]
bent = sp.gen_full(r2, b2, t2, 0.3, seed=1)[0]
check("curvature 0 gives straight legs", sp.max_turn_deg(straight[:, :n1], 0.5).max() < 1.0,
      f"max {sp.max_turn_deg(straight[:, :n1], 0.5).max():.2f} deg")
check("higher curvature bends more",
      sp.max_turn_deg(bent[:, :n1], 0.5).mean() > sp.max_turn_deg(straight[:, :n1], 0.5).mean())
# curvature must mean what the docstring says, or the sweep axis is meaningless
for c in (0.1, 0.3):
    one = sp._bow(root, box, c, torch.Generator().manual_seed(3), 3, 65)
    dd = box - root
    LL = dd.norm(dim=-1, keepdim=True)
    nn = torch.stack([-dd[:, 1], dd[:, 0]], -1) / LL
    lat = ((one - root[:, None, :]) * nn[:, None, :]).sum(-1).abs().amax(dim=1) / LL[:, 0]
    check(f"curvature {c} bounds the sideways bow at {c} of leg length",
          lat.max() <= c + 1e-4, f"max {lat.max():.4f}")

print("project  (leg masking)")
# a path that doubles back: approach leg arrives from the left, transport leaves
# to the left again, so the box's own position is ambiguous without a mask
there = torch.stack([torch.linspace(0, 4, 41), torch.zeros(41)], -1)[None]
back = torch.stack([torch.linspace(4, 0, 41), torch.full((41,), 0.3)], -1)[None]
dbl = sp.resample(torch.cat([there, back[:, 1:]], dim=1))
q = torch.tensor([[3.0, 0.15]])
s_free, _, _ = sp.project(q, dbl)
s_late, _, _ = sp.project(q, dbl, s_lo=torch.tensor([4.5]))
check("unmasked projection can snap to the wrong leg", s_free.item() < 4.0,
      f"s={s_free.item():.2f}")
check("s_lo forces the later leg", s_late.item() > 4.5, f"s={s_late.item():.2f}")
s_early, _, _ = sp.project(q, dbl, s_hi=torch.tensor([4.0]))
check("s_hi forces the earlier leg", s_early.item() <= 4.0, f"s={s_early.item():.2f}")
s_none, _, _ = sp.project(q, dbl, s_lo=torch.tensor([999.0]))
check("a bound that masks everything falls back instead of returning garbage",
      torch.isfinite(s_none).all() and s_none.item() >= 0)

print("gen_full_v2  (single hump, metres, turn-capped)")
r2 = torch.randn(600, 2) * 3
b2 = r2 + torch.randn(600, 2) * 2 + torch.tensor([5.0, 0.0])
t2 = b2 + torch.randn(600, 2) * 2 + torch.tensor([5.0, 0.0])
p2, sb2 = sp.gen_full_v2(r2, b2, t2, seed=2, lat_min=0.3, lat_max=1.5, turn_max_deg=35.0)
check("starts at the character", (p2[:, 0] - r2).norm(dim=-1).max() < 1e-3)
check("passes through the box", (p2 - b2[:, None, :]).norm(dim=-1).min(dim=1).values.max() < TOL)
check("reaches the target", (p2 - t2[:, None, :]).norm(dim=-1).min(dim=1).values.max() < TOL)
st2 = sp.path_stats(p2, sb2)
tmax = st2["turn_1.5m_deg"].max()
check("turn stays under the cap", tmax <= 35.0 + 1.0, f"max {tmax:.1f} deg")
# one hump means the offset keeps its sign along a leg; that is what makes a
# straight line a visibly wrong answer instead of an averagely-right one
i = int(sb2.min() / sp.DS) - 3
d = b2 - r2
nr = torch.stack([-d[:, 1], d[:, 0]], -1) / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
lat = ((p2[:, :i] - r2[:, None, :]) * nr[:, None, :]).sum(-1)
same = ((lat > -1e-6).all(dim=1) | (lat < 1e-6).all(dim=1)).float().mean()
check("the bow does not change side within a leg", same > 0.95, f"{same:.3f}")
check("a wider cap allows sharper paths",
      sp.path_stats(sp.gen_full_v2(r2, b2, t2, 2, 0.3, 1.5, 60.0)[0], sb2)["turn_1.5m_deg"].max()
      > st2["turn_1.5m_deg"].max())

print("gen_full_v2  (S legs mixed in)")
p2s, sb2s = sp.gen_full_v2(r2, b2, t2, seed=2, lat_min=0.3, lat_max=1.5,
                           turn_max_deg=35.0, p_two=1.0)
check("S legs still pass through the box",
      (p2s - b2[:, None, :]).norm(dim=-1).min(dim=1).values.max() < TOL)
check("S legs still reach the target",
      (p2s - t2[:, None, :]).norm(dim=-1).min(dim=1).values.max() < TOL)
# The cap is applied per leg, before the two are joined. sin(2*pi*t) arrives at
# the box steeply, so a 1.5 m window straddling the corner picks up part of the
# corner itself -- which is deliberately unconstrained -- and reads ~37 deg. Skip
# a full window either side and what is left is the leg, which does hold the cap.
sts = sp.path_stats(p2s, sb2s, corner_skip=1.6)
check("S legs stay under the same turn cap", sts["turn_1.5m_deg"].max() <= 35.0 + 1.0,
      f"max {sts['turn_1.5m_deg'].max():.1f} deg")
# The point of two humps is that they DO change side -- once. This has to be
# measured over each env's OWN leg 1: the single-hump test above uses one shared
# prefix (the shortest leg), and an S only crosses at the halfway point, so that
# prefix reports every long leg as single-sided.
def leg1_crossings(path, s_box):
    out = []
    for n in range(0, path.shape[0], 7):
        end = int(s_box[n] / sp.DS) - 3
        lat_n = ((path[n, :end] - r2[n]) * nr[n]).sum(-1)
        out.append(int(((lat_n[:-1] * lat_n[1:]) < 0).sum()))
    return torch.tensor(out, dtype=torch.float)

cs = leg1_crossings(p2s, sb2s)
check("S legs change side within a leg", (cs >= 1).float().mean() > 0.9,
      f"{(cs >= 1).float().mean():.3f}")
# and only once: three humps would cross twice, which is the v1 shape
check("S legs cross the centre at most once", cs.max() <= 1, f"max {int(cs.max())}")
check("single-hump legs never change side", leg1_crossings(p2, sb2).max() == 0)
pm, sbm = sp.gen_full_v2(r2, b2, t2, seed=2, lat_min=0.3, lat_max=1.5,
                         turn_max_deg=35.0, p_two=0.3)
frac = (leg1_crossings(pm, sbm) >= 1).float().mean()
check("p_two=0.3 gives a mix of both shapes", 0.15 < frac < 0.5, f"{frac:.3f} S")

print("gen_full_v2  (crest moved off centre)")
# 8 m legs at right angles: long enough that the crest index is a meaningful
# fraction, and the same geometry for both arms so only skew differs.
rc = torch.zeros(64, 2)
bc = torch.stack([torch.full((64,), 8.0), torch.zeros(64)], -1)
tc = torch.stack([torch.full((64,), 8.0), torch.full((64,), 8.0)], -1)


def crest_frac(skew):
    path, sb = sp.gen_full_v2(rc, bc, tc, seed=2, lat_min=0.3, lat_max=1.5,
                              turn_max_deg=35.0, skew=skew)
    n_hat = torch.tensor([0.0, 1.0])
    out = []
    for n in range(0, 64, 3):
        end = int(sb[n] / sp.DS) - 3
        lat = ((path[n, :end] - rc[n]) * n_hat).sum(-1)
        out.append(float(lat.abs().argmax()) / max(end - 1, 1))
    return path, torch.tensor(out)


p_flat, c_flat = crest_frac(0.0)
p_skew, c_skew = crest_frac(0.6)
check("skewed legs still pass through the box",
      (p_skew - bc[:, None, :]).norm(dim=-1).min(dim=1).values.max() < TOL)
check("skewed legs still reach the target",
      (p_skew - tc[:, None, :]).norm(dim=-1).min(dim=1).values.max() < TOL)
def crossings_flat(path, sb):
    """Sign changes of the sideways offset inside leg 1, on the rc/bc geometry."""
    n_hat = torch.tensor([0.0, 1.0])
    out = []
    for n in range(0, 64, 3):
        end = int(sb[n] / sp.DS) - 3
        lat = ((path[n, :end] - rc[n]) * n_hat).sum(-1)
        out.append(int(((lat[:-1] * lat[1:]) < 0).sum()))
    return torch.tensor(out)

_, sb_skew = sp.gen_full_v2(rc, bc, tc, seed=2, lat_min=0.3, lat_max=1.5,
                            turn_max_deg=35.0, skew=0.6)
check("skew keeps one bulge per leg", int(crossings_flat(p_skew, sb_skew).max()) == 0,
      f"max crossings {int(crossings_flat(p_skew, sb_skew).max())}")
check("without skew every crest sits at the halfway mark",
      float((c_flat - 0.5).abs().max()) < 0.06, f"max off {float((c_flat-0.5).abs().max()):.3f}")
check("with skew the crest spreads out",
      float(c_skew.std()) > 5 * float(c_flat.std()),
      f"std {float(c_skew.std()):.3f} vs {float(c_flat.std()):.3f}")
check("skew keeps the crest inside the leg",
      float(c_skew.min()) > 0.3 and float(c_skew.max()) < 0.75,
      f"{float(c_skew.min()):.2f}~{float(c_skew.max()):.2f}")

print("gen_full_v2  (spread redistributes turning, not height)")
# Amplitude must be pinned to compare shapes: spread draws its own random number,
# so with a range the two arms would not even be drawing the same bows.
V3 = dict(lat_min=1.5, lat_max=1.5, turn_max_deg=999.0, skew=0.0, p_two=0.0)


def crest_and_turn(spread):
    path, sb = sp.gen_full_v2(rc, bc, tc, seed=5, spread=spread, **V3)
    n_hat = torch.tensor([0.0, 1.0])
    h = []
    for n in range(0, 64, 3):
        end = int(sb[n] / sp.DS) - 3
        h.append(float(((path[n, :end] - rc[n]) * n_hat).sum(-1).abs().max()))
    return path, torch.tensor(h), sp.max_turn_deg(path, 1.5)


p_flat2, h_flat, t_flat = crest_and_turn((1.0, 1.0))
p_wide, h_wide, t_wide = crest_and_turn((1.8, 1.8))
check("spread keeps the path through the box",
      (p_wide - bc[:, None, :]).norm(dim=-1).min(dim=1).values.max() < TOL)
check("spread keeps the path reaching the target",
      (p_wide - tc[:, None, :]).norm(dim=-1).min(dim=1).values.max() < TOL)
check("crest height stays at the drawn amplitude",
      float((h_flat - 1.5).abs().max()) < 0.05 and float((h_wide - 1.5).abs().max()) < 0.05,
      f"flat {float((h_flat-1.5).abs().max()):.3f}  wide {float((h_wide-1.5).abs().max()):.3f}")
check("but the turning really is redistributed",
      float((t_wide - t_flat).abs().max()) > 3.0,
      f"max change {float((t_wide - t_flat).abs().max()):.1f} deg")

print("gen_full_v2  (lat_frac: short legs must not get long bows)")
# 1 m and 9 m legs side by side, same everything else.
rs = torch.zeros(48, 2)
bs = torch.stack([torch.cat([torch.full((24,), 1.0), torch.full((24,), 9.0)]),
                  torch.zeros(48)], -1)
ts_ = bs + torch.tensor([5.0, 0.0])
FRAC = 0.25


def crests(frac):
    path, sb = sp.gen_full_v2(rs, bs, ts_, seed=4, lat_min=0.0, lat_max=2.2,
                              turn_max_deg=999.0, skew=0.8, p_two=0.0,
                              spread=(0.85, 1.8), lat_frac=frac)
    n_hat = torch.tensor([0.0, 1.0])
    out = []
    for n in range(48):
        end = max(int(sb[n] / sp.DS) - 3, 5)
        out.append(float(((path[n, :end] - rs[n]) * n_hat).sum(-1).abs().max()))
    return torch.tensor(out)

c_off, c_on = crests(0.0), crests(FRAC)
check("without lat_frac a 1 m leg can get a bow longer than the leg",
      float(c_off[:24].max()) > 1.0, f"max {float(c_off[:24].max()):.2f} m on a 1 m leg")
check("lat_frac holds short legs to their share",
      float(c_on[:24].max()) <= FRAC * 1.0 + 0.05,
      f"max {float(c_on[:24].max()):.2f} m, allowed {FRAC * 1.0:.2f}")
check("lat_frac leaves long legs alone",
      float(c_on[24:].max()) > 1.5, f"max {float(c_on[24:].max()):.2f} m on a 9 m leg")

print(f"\n{ok} passed, {fail} failed")
raise SystemExit(1 if fail else 0)
