#!/usr/bin/env python3
"""Renders commanded-vs-executed paths from a run's npz into a standalone page.

    python scripts/steer/plot_paths.py out.html run_a.npz run_b.npz ...

Each run becomes a card of small multiples: the commanded ground-truth path
dashed, the BOX's actual track solid, the HUMAN's root track thin, the placement
target ringed.

**Both bodies are drawn, always.** The box alone hides the whole approach leg --
the box does not move until it is picked up, so a box-only plot silently starts
the story at the grasp and the approach deviation (our largest error term, 3x the
carry term) is invisible. If you add a plot here, plot root as well as box.
"""

import json
import sys
from pathlib import Path

import numpy as np

W, H, PAD = 210, 150, 14


def episodes(npz, k=3):
    d = np.load(npz)
    box, gt, ep, tar = d["box"], d["gt"], d["episode"], d["final_tar"]
    root = d["root"] if "root" in d else None
    held = d["held"] if "held" in d else np.ones(ep.shape, np.uint8)
    act = d["active"] if "active" in d else held
    out = []
    for n in range(box.shape[1]):
        m = ep[:, n] == ep[-1, n]
        a = act[m, n].astype(bool)
        if a.sum() < 40:
            continue
        # ROOT is drawn over the WHOLE episode, not the held frames. `held` is
        # 1 only while the box is carried, so masking by it deletes the entire
        # approach leg -- the walk from spawn to the box, which is where our largest
        # error term lives. The box track stays on held frames because before the
        # grasp it is a stationary dot.
        b = box[m, n][a][:, :2]
        r = root[m, n][:, :2] if root is not None else None
        grab = int(np.argmax(a)) if a.any() else 0
        g = gt[n]
        keep = int(np.linalg.norm(g - tar[n, :2], axis=-1).argmin()) + 1
        g = g[:max(keep, 2)]
        # how close the box ever got to the target: under 0.2 m counts as delivered
        err = float(np.linalg.norm(box[m, n][:, :3] - tar[n], axis=-1).min())
        # how far the box strayed from the commanded path, averaged over the run
        seg_a, seg_b = g[:-1], g[1:]
        ab = seg_b - seg_a
        den = (ab * ab).sum(-1); den[den < 1e-12] = 1e-12
        t = np.clip(((b[:, None, :] - seg_a[None]) * ab[None]).sum(-1) / den[None], 0, 1)
        proj = seg_a[None] + t[..., None] * ab[None]
        xtrack = float(np.linalg.norm(b[:, None, :] - proj, axis=-1).min(axis=1).mean())
        out.append((err, g, b, tar[n, :2], xtrack, r, grab))
    # Show typical GOOD behaviour: pick among delivered episodes, spread over
    # tracking quality (best / median / worst-of-delivered). The old best/median/
    # worst-overall made one panel in three a failure by construction.
    ok = [o for o in out if o[0] <= 0.2] or out
    ok.sort(key=lambda x: x[4])
    picks = [ok[0], ok[len(ok) // 2], ok[-1]] if len(ok) >= 3 else ok
    return d, picks[:k]


def panel(g, ex, tar, rt=None, grab=0):
    if len(ex) > 160:
        ex = ex[:: max(1, len(ex) // 160)]
    if rt is not None and len(rt) > 160:
        rt = rt[:: max(1, len(rt) // 160)]
    allp = np.vstack([g, ex, tar[None, :]] + ([rt] if rt is not None else []))
    lo, hi = allp.min(0), allp.max(0)
    span = max((hi - lo).max(), 1e-3)

    def T(p):
        q = (p - lo) / span
        return PAD + q[..., 0] * (W - 2 * PAD), H - PAD - q[..., 1] * (H - 2 * PAD)

    def path(p):
        x, y = T(p)
        return "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))

    tx, ty = T(tar)
    sx, sy = T(g[0])
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="commanded versus executed">'
            f'<path class="cmd" d="{path(g)}"/>'
            + (f'<path class="root" d="{path(rt)}"/>' if rt is not None else '')
            + f'<path class="exe" d="{path(ex)}"/>'
            + (lambda gx, gy: f'<circle class="grab" cx="{gx:.1f}" cy="{gy:.1f}" r="3.0"/>')(*T(rt[min(grab, len(rt)-1)])) if rt is not None else ''
            f'<circle class="start" cx="{sx:.1f}" cy="{sy:.1f}" r="3.4"/>'
            f'<circle class="tgt" cx="{tx:.1f}" cy="{ty:.1f}" r="5.5"/>'
            f'<circle class="tgtdot" cx="{tx:.1f}" cy="{ty:.1f}" r="1.8"/></svg>')


def card(npz, label, note):
    d, picks = episodes(npz)
    if not picks:
        return ""
    def cap(e, xt):
        ok = "배달 성공" if e <= 0.2 else "배달 실패"
        return f'{ok} · 목표까지 {e:.2f} m<br>경로 이탈 평균 {xt:.2f} m'
    body = "".join(
        f'<figure class="panel">{panel(g, b, t, r, gb)}<figcaption>{cap(e, xt)}</figcaption></figure>'
        for e, g, b, t, xt, r, gb in picks)
    return f'''<article class="card">
      <header class="card-hd"><h2>{label}</h2><p>{note}</p></header>
      <div class="panels">{body}</div>
    </article>'''


CSS = """
:root{--ground:#F5F8F9;--panel:#fff;--ink:#0F171B;--ink-2:#53636C;--ink-3:#8496A0;
--line:#DBE3E7;--cmd:#94A3AD;--exe:#0E7F90;--tgt:#B9782A;--root:#C2557B;--shadow:0 1px 2px rgba(15,23,27,.05)}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--ground:#0C1316;--panel:#131D22;
--ink:#E7EEF1;--ink-2:#9AAAB3;--ink-3:#6C7C86;--line:#22313A;--cmd:#6B7D89;--exe:#3FBACB;
--tgt:#DFA451;--root:#E1799C;--shadow:none}}
:root[data-theme=dark]{--ground:#0C1316;--panel:#131D22;--ink:#E7EEF1;--ink-2:#9AAAB3;
--ink-3:#6C7C86;--line:#22313A;--cmd:#6B7D89;--exe:#3FBACB;--tgt:#DFA451;--root:#E1799C;--shadow:none}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);line-height:1.55;
font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:48px 24px 80px;display:flex;flex-direction:column;gap:32px}
h1{font-size:clamp(1.6rem,3.2vw,2.2rem);font-weight:680;letter-spacing:-.022em;margin:0;text-wrap:balance}
.lede{margin:0;color:var(--ink-2);max-width:62ch}
.eyebrow{font:600 .72rem/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.14em;
text-transform:uppercase;color:var(--ink-3);margin:0 0 10px}
.legend{display:flex;flex-wrap:wrap;gap:20px;padding:14px 16px;background:var(--panel);
border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow)}
.legend div{display:flex;align-items:center;gap:9px;font-size:.86rem;color:var(--ink-2)}
.sw{width:26px;height:0;border-top:2.5px solid var(--exe);flex:none}
.sw.cmd{border-top:2px dashed var(--cmd)}
.sw.root{border-top:2px solid var(--root)}
.sw.dot{width:11px;height:11px;border:0;border-radius:50%;background:var(--tgt)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;
box-shadow:var(--shadow);display:flex;flex-direction:column;gap:14px}
.card-hd h2{margin:0;font-size:1.05rem;font-weight:640}
.card-hd p{margin:3px 0 0;color:var(--ink-2);font-size:.9rem}
.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.panel{margin:0;display:flex;flex-direction:column;gap:4px}
.panel svg{width:100%;height:auto;display:block}
.panel figcaption{font:500 .68rem/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;
color:var(--ink-3);text-align:center;font-variant-numeric:tabular-nums}
.cmd{fill:none;stroke:var(--cmd);stroke-width:2;stroke-dasharray:5 4;stroke-linecap:round}
.grab{fill:var(--root);stroke:var(--panel);stroke-width:1.2}
.root{fill:none;stroke:var(--root);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round;opacity:.95}
.exe{fill:none;stroke:var(--exe);stroke-width:2.4;stroke-linejoin:round;stroke-linecap:round}
.start{fill:var(--exe)}.tgt{fill:none;stroke:var(--tgt);stroke-width:2}.tgtdot{fill:var(--tgt)}
"""


def main(out, specs):
    cards = "".join(card(p, lbl, note) for p, lbl, note in specs if Path(p).exists())
    html = f'''<title>Trained Steering Paths</title>
<style>{CSS}</style>
<div class="wrap">
  <header>
    <p class="eyebrow">평가 · 512 envs × 3 trials</p>
    <h1>학습한 정책이 명령한 경로를 따라가는가</h1>
    <p class="lede">점선이 명령한 경로, 실선이 상자가 실제로 지나간 자취.
    각 조건마다 가장 잘된 것, 중간, 가장 안 된 것 세 개를 골랐다.
    <strong>목표까지</strong>는 상자가 목표에 가장 가까이 간 거리다 (0.2 m 이내면 배달 성공).
    <strong>경로 이탈</strong>은 상자가 명령한 경로에서 떨어진 거리의 평균이다.</p>
  </header>
  <div class="legend">
    <div><span class="sw cmd"></span>명령한 경로</div>
    <div><span class="sw root"></span>사람이 간 경로</div>
    <div><span class="sw"></span>상자가 간 경로</div>
    <div><span class="sw dot" style="background:var(--root)"></span>집는 지점</div>
    <div><span class="sw dot"></span>목표 지점</div>
  </div>
  {cards}
</div>'''
    Path(out).write_text(html)
    print("wrote", out, len(html), "bytes")


if __name__ == "__main__":
    main(sys.argv[1], [(p, Path(p).stem, "") for p in sys.argv[2:]])
