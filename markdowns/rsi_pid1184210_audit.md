# PID 1184210 RSI 설정 점검

확인 시각: 2026-10-02 약 14:35 KST. 프로세스에 연결·수정·종료하지 않고
`/proc` 실행 경로, 실행 인자, 실행 당시 저장 설정과 소스의 reset/학습 경로를 읽었다.
미래의 동일 PID/설정 상태는 이 문서로 판단하지 않는다. 프레임별 실제 충격 발생률은
이번 읽기 전용 점검에서 측정하지 않았다.

## 실행 기준

- cwd: `/home/hwanhee/koo_cvpr/stage1_fix`
- env config: `../../runs/manual_resume_289953/shared_random_binding_full_amp.yaml`
- output: `../output/shared_random_binding_resume1099002_full_amp_7500/`
- 저장 설정: `../output/shared_random_binding_resume1099002_full_amp_7500/SharedRandomBindingResume1099002FullAMP7500_01-22-53-00/relation_config.yaml`
- 원본 모션 목록: `../tokenhsi/data/dataset_loco_sit_carry_climb.yaml`
- 2인·4물체·2048환경, shared random binding, full AMP resume.
- 박스 X/Y 40~65cm, Z 25~55cm. `stateInit: Random`.
- 저장 설정에 `referenceRsi` 없음. 안전 후보표, 전 충돌체 1cm 관통 screen,
  물리 충격 기반 시점 선택, 해당 opt-in의 참조 박스 속도 복원은 사용하지 않는다.
- `relationReward.hard_skill_training.climb_rsi.late_fraction: 0.0`.
  phase_range 값 65~98%는 남아 있으나 확률 0이므로 후반 override는 비활성이다.

## 어떤 구간에서 RSI를 사용하는가

skill은 과제 template별 확률로 정하고, 선택된 skill의 clip을 weight로 뽑은 뒤
`MotionLib.sample_time_rsi()`로 원래 허용 구간에서 시간을 균등 샘플링한다.
모든 행동의 전 구간을 사용하는 것은 아니다. 박스 크기별로 허용 구간을 바꾸지도 않는다.
clip의 RSI 제외 프레임은 `frame_id / num_frames * motion_length`로 시간 변환된다.

| skill/clip | 설정상 RSI 시간 구간 |
|---|---|
| carryWith, B20 | 전체 0~100% |
| putDown, B21 | 전체 0~100% |
| pickUp, B19 (100 frames) | 0~30% 및 75~100%; 30~75% 제외 |
| pickUp, A5 (189 frames) | 0~34.39% 및 55.56~100%; 34.39~55.56% 제외 |
| sit, 20 clips | clip별 초반만. 허용 끝 지점은 44.44~80.82% 범위 |
| climb, 36_04 (70 frames) | 0~30% |
| climb, 36_14 (85 frames) | 0~51.76% |
| climb, 141_08 (85 frames) | 0~61.18% |
| climb, 40_08 (55 frames) | 0~18.18% |
| climb, 107_10 (50 frames) | 0~22% |
| climb, BEAMN01 (100 frames) | 0~56% |
| climb, jump-13-matrix (60 frames) | 0~16.67% |

표의 비율은 reset의 모션 시작 시점이다. 에피소드 중 정책이 그 뒤 구간과 비슷한
행동을 하는지와는 별개다. 실제 채택된 시간 분포는 장면 배치 재추첨 때문에 원래
시간 샘플링 분포와 달라질 수 있다.

## 손/사람 관통 때문에 박스가 튀면 어떻게 되는가

1. **초기 reset:** 기존 검사는 박스–박스 겹침, goal 배치 등을 재추첨한다.
   sit/climb에는 박스 안쪽 6cm 여유를 둔 영역에 body origin이 들어갔는지와
   발 원점이 바닥 아래 2cm보다 깊은지 보는 별도 검사가 있다.
   carryWith/pickUp/putDown의 담당 박스에 손·팔 충돌체가 1cm 이상 관통하는지를
   일반적으로 검사해 버리는 새 안전 screen은 이 실행에서 비활성이다.
2. **물리 진행:** 관통 상태가 채택되면 다음 simulation에서 접촉 반응이 생길 수 있다.
   튀는지 사전 rollout 검사하는 단계는 없다. carry 박스 속도는 기존 경로에서 0이며,
   사람 참조 속도와의 불일치도 남아 있다.
3. **보상:** 박스 위치의 step 차이로 계산한 속도가 2.5m/s를 넘으면 속도 페널티가
   적용된다. 이는 probe의 Δv>3m/s·속력>5m/s 진단과 다른 기준이다.
4. **종료:** 박스 속도 급증만으로 종료하지 않는다. 조기 종료는 비허용 신체의 접촉과
   낮은 body 높이로 넘어짐을 판정하며 progress>1 이후 적용된다. 한 사람이 넘어지면
   장면 전체를 reset한다. 타임아웃도 reset 사유다.
5. **학습:** 관측·action·보상·다음 관측·done·AMP 관측은 rollout buffer에 저장된다.
   튕김 때문에 학습 데이터를 특별히 버리는 경로가 없다. 넘어져 종료된 전이도 저장하고,
   실제 넘어진 agent의 terminate flag로 value bootstrap을 처리한다.

따라서 **현재는 관통/튕김 사례가 학습 데이터에 들어갈 수 있다.** 보상과 종료를 통해
학생이 처리하게 되는 구조이지, 안전한 RSI 시점만 미리 골라주는 구조가 아니다.
그런 사례가 얼마나 자주 발생하고 수렴을 얼마나 늦추는지는 별도 측정이 필요하다.

## 근거 코드

- `../tokenhsi/utils/motion_lib.py`: `sample_time_rsi`, 제외 프레임의 시간 변환.
- `../tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py`: skill/template RSI,
  `_reset_ref_state_init`, `_reset_boxes`, referenceRsi opt-in.
- `../tokenhsi/env/tasks/multi_agent/edge_ontop_task.py`: reset 재추첨,
  `_ontop_scene_infeasible`의 기존 배치·sit/climb 검사.
- `../tokenhsi/env/tasks/multi_agent/edge_context_task.py`, `relation_reward.py`:
  속도 페널티, 보상 합산.
- `../tokenhsi/env/tasks/multi_agent/humanoid_ma.py`: `_compute_reset`, `compute_humanoid_reset`.
- `../tokenhsi/learning/multi_agent/ma_agent.py`: `play_steps`의 rollout buffer 저장.

목표와 과거 검사 해석은 [reference_rsi_archive.md](reference_rsi_archive.md)를 참조한다.
