# 레포 구조

유지하는 컨피그는 `stage1`, `stage2_before_nonemask_rsi433_near50`, `stage2_before`, `stage2_before_rsi433`, `stage1_push`, `stage2_before_push`, `stage2_coupled_push`, `stage2_before_push_near50`, `stage2_coupled_push_goal5`, `stage2_coupled_push_individual_goal`, `stage2_before_push_near50_pair50_source5`, `stage2_coupled_push_individual_goal_solo5`, `stage2_before_push_near50_pair50_source5_virtual`, `stage2_before_push_near50_pair50_source5_virtual_asym`, `stage2_before_push_near50_pair50_source5_goal5`, `stage2_before_push_near50_pair50_source5_virtual_goal5`, `stage2_before_push_near50_pair50_source5_virtual_asym_goal5`, `stage2_before_push_near50_pair50_source5_goal5_goalroute`, `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute`, `stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute`, `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_clear`다. 실행 명령은 [config](config.md), 설치·추가 옵션은 [USAGE](USAGE.md)에 있다.

## 실행과 설정

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/run.py`, `utils/parse_task.py` | MA 학습/평가·Stage1/BEFORE 모델 등록과 환경 생성 |
| `tokenhsi/utils/config.py`, `experiments.py` | CLI, 20개 실험 계약, 학습/추론/resume env/sim 로딩 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/` | 환경·학습 YAML 각 21개 |
| `tokenhsi/scripts/multi_agent/*_{train,test,vnc}.sh` | 실험별 학습·viewer·VNC 진입점 63개 |
| `tokenhsi/scripts/multi_agent/*_eval.sh`, `evaluate.sh` | 일반 정식 평가21개, Stage1/Stage1 PUSH 전용 1명 단독 평가2개·공통 runtime |
| `tokenhsi/evaluate.py`, `rescore.py` | 독립 seed 실행·통계, 저장 trace CPU 재채점 |
| `tokenhsi/utils/evaluation_protocol.py`, `learning/multi_agent/formal_eval.py` | 학습 인원/단독 구분·점/영역/BEFORE 지표·개인 낙상 제어 |
| `tokenhsi/scripts/multi_agent/launch.sh`, `runtime_env.sh` | 공통 인자·출력 경로·conda·GPU |
| `tokenhsi/scripts/multi_agent/{gui_gpu_env,run-gui,mps_auto_env}.sh` | viewer GPU, VNC, 기존 MPS 연결 |
| `tokenhsi/utils/checkpoint_compat.py` | 기존 checkpoint의 이름·RSI 경로를 현재 표기로 변환 |
| `checkpoints/` | 외부 checkpoint 심링크 |
| `output/`, `output_etc/check/` | 본학습·공용 캐시, 확인 학습·평가·viewer |

## 환경

아래 경로는 `tokenhsi/env/tasks/multi_agent/` 기준이다.

| 파일 | 역할 |
| --- | --- |
| `humanoid_ma.py`, `humanoid_ma_carry.py` | 다중 사람·제어·상태와 공통 환경·물체 생성 |
| `scene_observation.py`, `scene_reset.py`, `reference_motion.py` | 관측, reset/배치, 모션·RSI·AMP |
| `edge_ontop_task.py`, `edge_context_task.py` | 과제 그래프·물리 재시도·보상·진단 |
| `humanoid_ma_push.py`, `jointpush_reward.py` | 1/2명 PUSH 자산·배치·접촉 RSI·CONTACT+AT 보상 |
| `before_approach_timeout.py` | 선행 최초 성공 이후 후행 root→goal 최초 접근 마감 타이머와 환경 전체 terminal 처리 |
| `humanoid_ma_before_push.py`, `before_push_reward.py` | BEFORE solo PUSH 자산·배치·접촉 보상, 기존 BEFORE gate 유지 |
| `humanoid_ma_coupled_push.py`, `coupled_push_reward.py` | 공동 상자·가상 구역·그룹 gate·goal saturation |
| `push_goal_markers.py` | PUSH 목표의 viewer/VNC 선 표시만 추가; actor·reset·scale 수정 없음 |
| `humanoid_ma_before.py` | BEFORE 역할·공유 물체·AT progress·후행 초기화 |
| `scene_layout.py`, `scene_validation.py` | 활성 물체·distractor·loco 참조·물리 유효성·새 goal5 후보의 수치 여유 |
| `before_rsi_cache.py`, `native_rsi_cache.py` | 후행 pickUp/carryWith native 크기별 screening |
| `before_pickup_carry_rsi.py`, `before_relative_sampling.py` | RSI433 적용/fallback, near50 배치 |
| `size_rsi_cache.py` | Stage1 크기별 RSI 캐시 |
| `*_reward.py` | 과제 보상·BEFORE gate·선택적 비대칭 progress·CPA·물체 속도 |

## 정책과 과제 정의

- `tokenhsi/learning/multi_agent/amp_network_builder_ma.py`, `task_role_encoder.py`: 공통 GTA·task bias·사람 action MLP·actor/critic.
- 같은 디렉터리의 `before_network.py`, `human_coordination.py`: BEFORE 사람 attention·object packet·Stage1 전이·actor encoder 동결.
- `stage2_push_network.py`: BEFORE PUSH와 coupled의 동일 5과제/4family 정책·6N task/group suffix·4종 사람 relation·packet 기반 alias 전처리·선택적 AT 목표 XYZ 투영·Stage1 PUSH 전이. `before_push_network.py`는 같은 builder의 호환 import다.
- `viewer_diagnostics.py`: viewer 전용 native 보상 성공 진단, 정식 평가와 별도.
- `ma_agent.py`, `ma_players.py`, `scene_normalizer.py`: 장면 단위 PPO/AMP·저장/복원·평가·정규화.
- `tokenhsi/utils/{edge_*,task_role_spec,before_spec}.py`: 그래프·역할·packet·checkpoint 계약.
- `tokenhsi/utils/{coupled_push_spec,coupled_push_scene}.py`: 공동 그룹·인원별 물리 bank·직선 목표·충돌 검사.
- `tokenhsi/utils/coupled_push_contact.py`: 개인 목표 실험의 공통 4면 선택·담당 구역 CONTACT·개인 도착 gate.
- `tokenhsi/utils/{before_push_spec,before_push_layout}.py`: solo PUSH 샘플링·계약·viewer 가변 토큰 예산의 과제 재샘플링·goal5 거리 설정·다중 사람 경로 배치·source5 초기 후보16개 묶음 검사·후보 소진 시 순차 보완.
- `tokenhsi/utils/{before_sampling,before_relative_sampling,before_object_alias}.py`: mix226/near50·토큰 alias·가상 상자 GTA XYZ 변환.
- `tokenhsi/utils/before_goal_sampling.py`, `env/tasks/multi_agent/before_goal_sampling.py`: goalroute의 목표 중심 후행 배치·수행 거리 near50·조건부 거리 직접 샘플링과 reset 연결.
- `tokenhsi/utils/stage1_sampling.py`: 기존 접근/정사각형 목표·선택적 상자 중심 반경 목표 샘플링.
- `tokenhsi/learning/multi_agent/stage1_push_network.py`: 5과제 정책·4 family AMP·선택적78k 전이.
- `tokenhsi/utils/{stage1_jointpush,jointpush_scene,jointpush_geometry,jointpush_rsi,jointpush_rsi_geometry}.py`: PUSH 계약·1/2명 장면·bank/모션/기하 검증.
- `tokenhsi/utils/{stage1_geometry,stage1_config,stage1_platforms,size_rsi}.py`: Stage1/BEFORE 공통 기하·RSI·물리 설정.
- `lpanlib/poselib/`: 기존 모션 배열 로딩 라이브러리.

## 검증과 문서

`test_before_goal_sampling.py`는 목표 중심 거리·조건부 분포·near50·기존 goal5와의 계약 분리를 검사한다. `tokenhsi/tests/`는 보상·배치·GTA·mask·RSI·전이·재로드·출력 경로를 검사한다. `before_test_utils.py`, `stage1_test_utils.py`는 CPU fixture다. `check_before_asymmetric_progress.py`는 현재 AT 성공 전환·비대칭 progress·기존 항 보존을 검사한다. `check_before_virtual_object.py`는 가상 Alias XYZ·실제 회전·원본 상태 보존을 검사한다. `check_before_object_alias.py`·`check_before_push.py`·`check_coupled_push.py`는 실제 물리·부분 reset·추론 검사 진입점이다.

[PUSH](reference/stage1_push.md), [BEFORE PUSH](reference/stage2_before_push.md), [공통 Stage2/coupled](reference/stage2_coupled_push.md), [평가 규약](reference/evaluation.md), [Stage1 단독](../evaluation_single.md), [공통 정책](reference/ma_clean_scene_gta.md), [BEFORE](reference/stage2_before.md), [지표](reference/relation_diagnostics.md), [이관](reference/PORTING_GUIDE.md)을 참고한다. 이전 실험은 [legacy](legacy/README.md)에 보존한다.

- `tokenhsi/env/tasks/multi_agent/before_wait_reward.py`: 선행 사람/상자→goal 통로와 안전 대기 구역의 선택적 보상. `test_before_wait_clear.py`는 CPU 기하/계약 검사, `check_before_wait_clear.py`는 실제 환경 보상 합산·부분 reset 검사다. [설계](reference/before_wait_clear.md).

- `tokenhsi/learning/multi_agent/attention_viewer.py`, `attention_dashboard.html`: viewer 전용 actor GTA/coordination attention 추출과 localhost 대시보드. `test_attention_viewer.py`는 행동·RNG·가중치 보존, MHA 확률 일치와 셔플 복원을 검사한다.

- `before_reward.py`의 선택적 state 구간 정규화: [설계와 예외](reference/before_state_rescale.md). `test_before_state_rescale.py`는 CPU 보상/계약 검사, `check_before_state_rescale.py`는 실제 환경 연결·보상합·부분 reset 검사다.

- `stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute_approach_timeout2s`: [2초 이내 최초 접근 확인](reference/before_approach_timeout.md). CPU 테스트 `tests/test_before_approach_timeout.py`, 환경 연결 검사 `tests/check_before_approach_timeout.py`.

- `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_approach_timeout2s`: 일반 virtual의 최초 접근2초 비교 실험. 같은 timeout 구현을 사용한다.

- BEFORE asym 3m: `before_reward.py::predecessor_at_progress`의 `wait_radius_m`는 gate 닫힘 때만 고정 경계를 사용한다. 전용 `_radius3m` 설정과 wrapper를 사용한다.

- BEFORE 연속 progress: `before_continuous_progress.py`의 최초 개방 anchor와 `BeforeRuntime.step` 연결. [명세](reference/before_continuous_progress.md), CPU/물리 검사 `test_before_continuous_progress.py` / `check_before_continuous_progress.py`.

- 새 base wait_penalty_v2: `before_wait.py`는 ksh 원본의 읽기 전용 복사본이며 `HumanoidMABefore` reset과 `edge_context_task`의 속도 벌점으로 연결한다. [명세](reference/before_wait_penalty_v2.md).

- 대기 속도벌점 soft entry: `before_wait.py`의 `entry_weight`/`soft_entry_width`, CPU/물리 검사 `test_before_soft_wait_entry.py` / `check_before_soft_wait_entry.py`. [명세](reference/before_soft_wait_entry.md).

- 가져온 relfusion_zerobg 설정: `tokenhsi/data/cfg/imported/relfusion_zerobg/`의 env/train YAML·실제train_used·saved_run·transfer_report·manifest. 설정 원본 보관본. 실행용 fusion 구현과 wrapper는 [relfusion 명세](reference/before_relfusion_zerobg.md)를 따른다.

- BEFORE relation fusion: `learning/multi_agent/before_relation_fusion.py`, `human_coordination.py`, `stage2_push_network.py`. 원본 동작 이식 및 기존 attention-order 진단 유지.

- relfusion 목표 속도 L2 대기벌점: `before_target_velocity.py`의 순수 계산 + `before_wait.py`의 SIT/CLIMB 분기. [명세](reference/before_target_velocity.md).

- 속도 상한 실험 `_relfusion_zerobg_speed_cap`: 동일 helper의 speed-cap 분기. [명세](reference/before_speed_cap.md).

- 제곱 속도오차 + 3m 감속: `_velocity_l2_squared`, [명세](reference/before_velocity_squared.md).

- 먼거리 접근 하한/추종 혼합: `_velocity_l2_squared_floorblend`, [명세](reference/before_velocity_floorblend.md).

- `*_velocity_l2_squared_floorblend_speed08`: floorblend 비교 설정. 속도 기준0.8m/s, tau3.75로 감속거리3m 유지. 기존 구현·전용 train/test/VNC/eval wrapper 사용.
