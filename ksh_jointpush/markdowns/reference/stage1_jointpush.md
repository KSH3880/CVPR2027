# Stage1 jointpush

현재 checkout의 `stage1_v2_fixed` 네 과제에 KOO PUSH를 연결한 스크래치 실험이다. 이름에 jointpush가 들어가지만 **아직 Stage2 coupled 학습은 아니다.**

## 설정

| 항목 | 값 |
| --- | --- |
| 규모 | 2명·4상자·2048환경·1200step |
| SIT/CLIMB/AT/ONTOP/PUSH | 10/20/25/20/25% 크기 생성 prior 및 크기 조건부 과제 sampler |
| 사람 기준 위치·방향 | 기존 Stage1의 참조 root + 반경2.5m 사람별 offset, 참조 yaw 유지 |
| PUSH loco 사람→상자 | 실제 사람 XY 기준 반경 U(2,10)m·360° |
| PUSH 상자→목표 | loco/접촉 RSI 모두 반경 U(2,10)m·360°, 사각형/정면 cone 제한 없음 |
| PUSH 상자/목표 높이 | 바닥 Z/2, 목표는 reset 후 고정 |
| 크기 | X/Y1.6~2.2m(0.1m 간격), Z1.45~1.7m(0.05m), min(X,Y)≥1.1Z인167종 |
| 물성 | 20kg·마찰0.4·별도 밑판 없음 |
| PUSH RSI | loco50%/접촉50%, KOO949후보·후반30% 요청70%, clip7 제외, 진행률에 따른 상자 이동 없음 |
| PUSH AMP | push_loco50%/push50%, 기존 과제는 기존 family 유지 |
| 기존 상자/받침/distractor | Stage1 크기/물성, distractor는 반경4.5m 원판·무작위 yaw |
| 목표 최소거리 | PUSH 목표와 담당 사람 중심1m 이상 |
| 경로 검사 | 제안 후 거부 샘플링. PUSH끼리3.5m 분리, 뒤쪽1.5m 포함, 다른 사람·과제 상자·SUPPORT와 여유 검사 |
| 학습 | teacher 없음, 학습률2e-5, 입력/AMP 정규화 통계 갱신, 본인 보상100% |
| 평가 | 모든 과제 loco100%, 과제 비율·물성·loco 배치/목표 sampler는 학습과 동일. 기존 PTH는 가중치/정규화만 로드 |

크기는 환경 생성 시 고정된다. PUSH 크기로 생성한 담당 슬롯은 PUSH를 유지하고, 다른 슬롯은 그 크기로 가능한 기존 과제를 reset마다 고른다. SUPPORT는 ONTOP이면 과제 받침이고, 그 외에는 물리 장면에 남는 distractor다. 숨겨진 물리 후보 bank는 없다. 간섭 없는 표본만 채택하므로 최종 거리/방향 빈도는 균등 제안 분포와 다를 수 있다.

## 보상

KOO의 CONTACT+AT를 사용한다. 각 관계의 state/progress/success 가중치는0.2다. CONTACT는 목표 방향으로 밀 수 있는 유한한 면과 양손의 XY 거리 중 최악값을 사용한다. 상태 `exp(-10d²)>0.9`와 양손 반력/상자 힘0.5N이 접촉 성공 조건이다. 접근 점수는 사람→상자 중심 거리의 `1/(1+max(d-0.5,0))`다.

AT는 중심3D 거리의 `exp(-10d²)`, XY 접근 `1/(1+max(dxy-0.1,0))`, 성공은 중심 거리20cm 및 Z오차1mm를 사용한다. AT에는 CONTACT 선행 gate가 없다. 관계 성공 시 해당 세 항을 포화하고, AT 성공 시 CONTACT 보상도 포화한다. 두 관계의 과제 보상 최대는1.2이며 AMP/페널티 합산은 기존 학습기 규칙이다.

## 출처와 연결

- 기반: 로컬 `stage1_v2_fixed.yaml` 및 `HumanoidMACarry`.
- PUSH 기하/RSI/보상: [KOO 포트](https://github.com/KSH3880/CVPR2027/tree/koo_push_1026/ports/stage1_base_push/tokenhsi), 원본 [가이드](https://github.com/KSH3880/CVPR2027/blob/koo_push_1026/PUSH_RSI_UNIFORM_GUIDE.md).
- KOO의 고정 사람 anchor·yaw 추가 회전·square/cone 목표·넓은 distractor 영역은 가져오지 않았다. 새 배치는 `jointpush_scene.py`에 분리한다.
- 원본 RSI bank JSON과 후보를 보존하고 실제 모션/asset/mesh 및20kg·0.4·167종 물성을 검사한다. 새 거리/방향 배치는 원본 bank의 scene 계약과 구분하며 최종 reset 간섭 검사를 적용한다.
- 학습/평가 명령: [config](../config.md). 체크포인트·환경 옵션: [USAGE](../USAGE.md#stage1-jointpush-체크포인트와-환경).
- 검증 기록: [changelog](../../changelog.md), `output_etc/check/stage1_jointpush/`.
