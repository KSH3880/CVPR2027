# Imported trajectory coordinator checkpoints

2026-09-09에 `traj` 브랜치 작업자로부터 전달받은 C2 coordinator PTH다.
두 파일 모두 `tokenhsi-coord-c2-v1` strict loader와 실제 CPU forward를 통과했다.

| 파일 | 확인된 설정 | step | SHA-256 |
|---|---|---:|---|
| `c5.pth` | C5-r2, pointwise physical speed caps, `unnecessary_slow=2`, random priority | 2000 | `d610a40286e9756762569c29cd145a0e52637d8313ec2b875841d9b8780e1c13` |
| `c13.pth` | C13 path-guard-30, `path_residual=30`, random priority | 300 | `f59977714315a7f6d830990ea232ea0e8b50bf386f8f9ad90cf45f54b924775f` |

기존 GT 실행은 기본 task를 그대로 사용한다.

```bash
MA_GPU=7 bash scripts/masteer/view.sh ms18_maskteam_origscale_c06_s0
```

Coordinator trajectory는 A=2 전용이다. positional 인자는 coordinator PTH가 아니라
frozen ms18 executor tag이며, C5/C13 선택은 `COORD_CKPT`로 한다.

```bash
MA_GPU=7 MS_SINGLE=0 MS_SCEN=cross \
MS_TASK=HumanoidMACoordCarry COORD_MODEL=c2 COORD_PROVIDER=learned \
COORD_CKPT=/home/hwanhee/koo_cvpr/runs/coord/imported/c5.pth \
bash scripts/masteer/view.sh ms18_maskteam_origscale_c06_s0
```

`c13.pth`를 보려면 `COORD_CKPT`만 바꾼다.
