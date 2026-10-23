# 5과제 PUSH 학습 지표

기존 CONTACT/AT 보상·성공 조건·샘플러를 유지하고 관측 지표만 추가한다. 재개 전 기록에는 새 지표가 없으며 소급 계산하지 않는다. TensorBoard 실제 prefix는 `relation/`이다.

| TensorBoard 이름 (`relation/` 뒤) | 정의 |
| --- | --- |
| `edge/push_contact/phi_raw` | PUSH 양손의 공통 유한 면 XY state. 높이 무관 |
| `edge/push_contact/own_success`, `push/contact` | control step에서 한 번 이상 기존 양손 접촉 조건을 만족한 PUSH agent 비율 |
| `edge/push_contact/distance_xy` | 두 손 중 더 먼 손의 공통 면 XY 거리(m). `distance`도 동일, 높이 오차는 0 |
| `push/contact_tick_fraction` | 물리 tick별 기존 접촉 조건 만족 비율 |
| `push/at` | PUSH agent만의 현재 AT 자체 조건 만족 비율. CARRY 제외 |
| `push/contact_and_at` | 같은 control step의 CONTACT와 현재 AT 동시 만족 비율 |
| `push/goal_distance_m` | PUSH 상자 중심에서 목표까지 XY 거리 |

이동량은 물리 tick 전후의 상자 중심 XY를 비교하여 control step 내에서 합산한다. 순간 접촉을 control step 전체 이동에 적용하지 않는다. episode reset/RSI 배치 이동은 포함하지 않는다.

| 이름의 중간 부분 | 계산 |
| --- | --- |
| `travel` | 상자의 XY 이동 경로 길이 |
| `contact_travel` | 같은 물리 tick에 양손 접촉 조건을 만족했을 때의 XY 이동 길이 |
| `goal_progress` | 이전 목표거리 − 현재 목표거리. 가까워지면 양수, 멀어지면 음수 |
| `contact_goal_progress` | 접촉 tick만의 부호 있는 목표거리 감소량 |
| `contact_goal_forward` | 접촉 tick에서 양수인 목표거리 감소량만 합산. 후퇴 차감 전 전진량 |

각 이름에 다음 suffix를 붙인다. 예: `relation/push/episode_contact_goal_progress_m`.

- `_m_per_step`: 모든 활성 PUSH agent/control step에 대한 평균(m). 비접촉 step도 분모에 포함한다.
- `_m_per_s`: 위 값을 control step 시간으로 나눈 평균속도(m/s).
- `episode_<이름>_m`: 종료된 PUSH agent episode별 누적값의 평균(m).

`relation/edge/at/*`는 기존 CARRY_AT/PUSH_AT 혼합이다. PUSH만 비교할 때는 `relation/push/at`를 사용한다. 접촉 중 이동은 관측된 동시성 지표이며 접촉력이 이동의 원인인지 판정하는 별도 분석은 아니다.

## 500회 재개

각 기존 실험의 번호 checkpoint `*_00000500.pth`를 읽을 수 있고 epoch=500·모델/optimizer/관측 및 AMP 통계·관계 계약이 존재함을 확인한 뒤, 명세에 기록한 원래 PID만 SIGTERM하고 같은 전용 train wrapper의 `RESUME_CHECKPOINT`로 재개한다. 환경 수2048·GPU6·기존 출력 root를 유지하며 새 timestamp run에 기록한다. 모델·optimizer·통계·epoch/frame을 checkpoint에서 복원하고 시뮬레이터 episode와 진단 누적값은 새로 시작한다.

이번 두 학습의 고정 명세·상태·재개 로그는 `output_etc/check/push_metrics_resume500_20261021/{manifest.json,status.json,controller.log,stage1_*.log}`에 있다. 컨트롤러는 검증 완료 marker가 있어야 전환하며 PID 시작 시각을 대조한다. 기존 status가 있으면 중복 실행을 거부한다.

두 학습 모두 500회 재개를 완료했고 첫 새 epoch501/frame65536000·새지표32개finite를 확인했다. 새run 경로와 검증 시점 값은 같은 폴더의 `verification.json`을 참조한다.

```bash
# 사본 push/ 루트에서 한 번만 실행하는 명시적 500회 재개 명세
python tokenhsi/scripts/multi_agent/resume_five_task_metrics.py \
  output_etc/check/push_metrics_resume500_20261021/manifest.json

# 개별 수동 재개 (컨트롤러와 동시에 실행하지 않는다)
TOKENHSI_GPU=6 RESUME_CHECKPOINT=/absolute/path/Stage1FiveTaskDepth18020kg_00000500.pth \
  bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth180_20kg_train.sh 2 2048 4
TOKENHSI_GPU=6 RESUME_CHECKPOINT=/absolute/path/Stage1FiveTaskDepth20020kgFriction04_00000500.pth \
  bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04_train.sh 2 2048 4
```
