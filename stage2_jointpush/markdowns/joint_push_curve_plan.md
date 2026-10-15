# 다음 실험: 60kg 직선 Push·360° agent 시작

2026-10-14 갱신. 곡선 혼합 초안은 사용자 결정으로 제외한다. Stage1의 직선 Push를 활용하고 Stage2는 공동 접근·자리 배치·동시 밀기를 학습한다.

| 항목 | 설정 |
|---|---|
| 실제 상자 | 2.6m cube/60kg/마찰0.6 |
| 목표 방향 | 초기 box local+Y 정렬, 곡선 각도 없음 |
| 미는 거리 | reset마다2.2–5m 균등 랜덤, 환경별 독립 |
| agent 위치 | 상자 주변360° 방위각, XY외접원+1–2m 반경 |
| agent heading | 각각360°독립 랜덤 |
| agent 최소 XY 간격 | 0.8m |
| 미는 면 | 초기local−Y면, 두 사람이 공유하며 에피소드동안동일로컬면유지 |
| 모델/AMP/RSI/보상 | 기존60kg설정과동일, loco-only RSI |
| 시작 | Stage1초기값전이에서새Stage2, 기존Stage2resume없음 |

거리는 상자 중심이 초기 위치에서 목표 중심까지 이동해야 하는 거리다. agent 시작 거리와 별개다. 목표는 reset 이후 고정하며, 다음 reset에서 다시 샘플링한다. 초기 정렬은 상자 회전을 물리적으로 잠그는 설정이 아니다.

다음 실행 후보는 연속AT gate 프로필 `joint_push_aligned_surround60_softgate_ca4.yaml`, 명령은 `bash stage2_jointpush/scripts/joint_push_softgate_train.sh`. hard-gate비교용은기존surround60프로필을유지한다. GPU6/2048환경. 현재 학습을 바꾸거나 새 학습을 자동 실행하지 않았다. 실행/평가/뷰어 상세는 config.md 상단 참조.

로그에서는 접촉gate 비율과 실제상자이동/목표거리감소/목표도달률을분리해보는것을권장한다. 현재success보상에는손배치가포함되고progress는현재거리점수이다. rear-only기존모델과360°시작모델은초기분포가달라같은평가분포에서비교해야한다.
