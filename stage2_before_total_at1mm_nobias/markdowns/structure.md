# 레포 구조

유지하는 컨피그는 `stage1`, `stage2_before_nonemask_rsi433_near50`, `stage2_before`, `stage2_before_rsi433`, `stage1_push`, `stage2_before_push`, `stage2_coupled_push`, `stage2_before_push_near50`, `stage2_coupled_push_goal5`, `stage2_coupled_push_individual_goal`, `stage2_before_push_near50_pair50_source5`, `stage2_coupled_push_individual_goal_solo5`, `stage2_before_push_near50_pair50_source5_virtual`, `stage2_before_push_near50_pair50_source5_virtual_asym`, `stage2_before_push_near50_pair50_source5_goal5`, `stage2_before_push_near50_pair50_source5_virtual_goal5`, `stage2_before_push_near50_pair50_source5_virtual_asym_goal5`, `stage2_before_push_near50_pair50_source5_goal5_goalroute`, `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute`, `stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute`, `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait`, `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v2`, `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v3`, `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty`, `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2`, `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg`, `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v2`, `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v3`, `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v4`, `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v3_ep900`, `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_human3_resume21k`, `stage2_before_total`, `stage2_before_total_ft21k`, `stage2_before_total_progress0`, `stage2_before_total_at1mm`, `stage2_before_total_at1mm_climb2`, `stage2_before_at1mm`, `stage2_before_total_at1mm_nobias`다. 실행 명령은 [config](config.md), 설치·추가 옵션은 [USAGE](USAGE.md)에 있다.

## 실행과 설정

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/run.py`, `utils/parse_task.py` | MA 학습/평가·Stage1/BEFORE 모델 등록과 환경 생성 |
| `tokenhsi/view_before_chain.py`, `utils/before_chain_viewer.py`, `scripts/multi_agent/before_chain3_{test,vnc}.sh` | 3명 AT→ON_TOP→ON_TOP 고정 장면 viewer; 해당 프로세스에서만 연쇄 의존·가상 받침 확장 |
| `tokenhsi/utils/config.py`, `experiments.py` | CLI, 38개 실험 계약, 학습/추론/resume env/sim 로딩 |
| `tokenhsi/utils/before_total_at1mm_climb2_resume.py` | total12k를 AT 보상1mm·CLIMB+2m 설정으로 이관하며 학습 상태와 원본을 보존 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/` | 환경·학습 YAML 각 38개 |
| `tokenhsi/scripts/multi_agent/*_{train,test,vnc}.sh` | 실험별 학습·viewer·VNC 진입점 114개 |
| `tokenhsi/scripts/multi_agent/*_eval.sh`, `evaluate.sh` | 일반 정식 평가38개, Stage1/Stage1 PUSH 전용 1명 단독 평가2개·공통 runtime |
| `tokenhsi/evaluate.py`, `rescore.py` | 독립 seed 실행·통계, 저장 trace CPU 재채점 |
| `tokenhsi/summarize_before_stops.py`, `utils/before_stop_evaluation.py` | 정식 평가의 동일512환경×3seed 기록에서 정지/거리/동일쌍 성공을 CPU 집계; 별도64환경 실행 없음 |
| `tokenhsi/utils/evaluation_protocol.py`, `learning/multi_agent/formal_eval.py` | 학습 인원/단독 구분·점/영역/BEFORE 지표·개인 낙상 제어 |
| `tokenhsi/scripts/multi_agent/launch.sh`, `runtime_env.sh` | 공통 인자·출력 경로·conda·GPU |
| `tokenhsi/scripts/multi_agent/{gui_gpu_env,run-gui,mps_auto_env}.sh` | viewer GPU, VNC, 기존 MPS 연결 |
| `tokenhsi/utils/before_total_progress_resume.py` | total9k Progress 반경0 재개용 복사본·optimizer/epoch 보존 |
| `tokenhsi/utils/before_total_finetune.py` | total_ft21k 전용 v1 21k 모델·정규화 초기화, optimizer/epoch/frame 미복원 |
| `tokenhsi/utils/before_fusion_resume.py` | human3 전용 v1 21k 재개용 복사본 생성·모델/optimizer/epoch 보존·새 저장본은 그대로 재개 |
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
| `humanoid_ma_before_push.py`, `before_push_reward.py` | BEFORE solo PUSH 자산·배치·접촉 보상, 기존 BEFORE gate 유지 |
| `humanoid_ma_coupled_push.py`, `coupled_push_reward.py` | 공동 상자·가상 구역·그룹 gate·goal saturation |
| `push_goal_markers.py` | PUSH 목표의 viewer/VNC 선 표시만 추가; actor·reset·scale 수정 없음 |
| `before_wait.py` | 선택적 BEFORE 후행 대기 진입·유예·XY/XYZ 속력·연속/누적 위반 종료와 AT 성공 후 해제·선택적 ON_TOP 사람 기준 진입 |
| `humanoid_ma_before.py` | BEFORE 역할·공유 물체·AT progress·후행 초기화 |
| `scene_layout.py`, `scene_validation.py` | 활성 물체·distractor·loco 참조·물리 유효성·새 goal5 후보의 수치 여유 |
| `before_rsi_cache.py`, `native_rsi_cache.py` | 후행 pickUp/carryWith native 크기별 screening |
| `before_pickup_carry_rsi.py`, `before_relative_sampling.py` | RSI433 적용/fallback, near50 배치 |
| `size_rsi_cache.py` | Stage1 크기별 RSI 캐시 |
| `*_reward.py` | 과제 보상·BEFORE gate·선택적 AT 보상 전용1mm·선택적 비대칭 progress·CPA·물체 속도 |

## 정책과 과제 정의

- `tokenhsi/learning/multi_agent/amp_network_builder_ma.py`, `task_role_encoder.py`: 공통 GTA·task bias·사람 action MLP·actor/critic.
- 같은 디렉터리의 `before_network.py`, `human_coordination.py`: BEFORE 사람 attention·object packet·Stage1 전이·actor encoder 동결.
- `stage2_push_network.py`: BEFORE PUSH와 coupled의 동일 5과제/4family 정책·6N task/group suffix·4종 사람 relation·packet 기반 alias 전처리·선택적 AT 목표 XYZ 투영·Stage1 PUSH 전이. `before_push_network.py`는 같은 builder의 호환 import다.
- `tokenhsi/learning/multi_agent/before_relation_fusion.py`: 후행 전용 BEFORE embedding을 Human Attention bias와 출력 projection 전 잔차 MLP에 공유하며 SELF/NONE bias는0으로 고정한다. `before_zerobg_nobias`는 actor/critic의 bias projection 없이 mask 없는 attention을 쓰고 MLP fusion을 유지한다.
- `viewer_diagnostics.py`: viewer 전용 native 보상 성공 진단, 정식 평가와 별도.
- `ma_agent.py`, `ma_players.py`, `scene_normalizer.py`: 장면 단위 PPO/AMP·저장/복원·평가·정규화.
- `tokenhsi/utils/{edge_*,task_role_spec,before_spec}.py`: 그래프·역할·packet·checkpoint 계약.
- `tokenhsi/utils/before_box_size.py`: BEFORE PUSH viewer의 공유상자 행동별/전체공통 교집합 크기·시작 시 고정할 쌍/행동 선택. `test_before_box_size.py`는 범위·격자·역할·비공유 상자·토큰 예산을 검사한다.
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

`test_before_total_at1mm_climb2.py`는 CLIMB 전용+2m·기존 반경 보존·다중쌍/셔플·Ever 해제·total12k 재개 상태 보존을 검사한다. `test_before_total_at1mm_nobias.py`는 bias 생략·fusion gradient·actor/critic 전이·재로드·계약 분리를 검사한다. `test_before_total_at1mm.py`는 AT 보상만1mm·후행20cm Current/Ever 유지·단독/PUSH 포화·새 학습 계약을 검사한다. `test_before_total_finetune.py`는 v1 21k 모델/정규화 이관·optimizer/epoch/frame 초기화·소스/대상 계약 분리를 검사한다. `test_before_total.py`는 Stage1 26k 신규 학습·900step·ON_TOP payload 진입·사람 XYZ 속력 페널티·Ever 진입/해제를 검사한다. `test_before_goal_sampling.py`는 목표 중심 거리·조건부 분포·near50·기존 goal5와의 계약 분리를 검사한다. `check_before_wait.py`는 실제 reset 연결·해제·Actor/Critic·부분 reset을 확인한다. `test_before_wait_v2.py`는 유예 없는 XYZ 누적 위반·AT 3D 성공/Gate·기존 설정 보존을 검사한다. `test_before_fusion_human3.py`는 +3m 사람 기준 진입·AT Ever 해제와 v1 21k 재개 상태/원본 보존을 검사한다. `test_before_fusion_v2.py`는 ON_TOP 사람 기준 진입·+2.5m 경계·기존 설정/계약 보존을 검사한다. `test_before_fusion_v3_ep900.py`는 새 학습의900스텝·정식 평가 연결과 기존 실험1200스텝 보존을 검사한다. `test_before_wait_penalty.py`는 개인 XYZ 속도 감점·진입/성공 기록·부분 reset·기존 설정 보존을 검사한다. `test_before_wait.py`는 대기 타이머·XY 기준·다중 쌍·부분 reset·설정 분리를 검사한다. `tokenhsi/tests/`는 보상·배치·GTA·mask·RSI·전이·재로드·출력 경로를 검사한다. `before_test_utils.py`, `stage1_test_utils.py`는 CPU fixture다. `check_before_asymmetric_progress.py`는 현재 AT 성공 전환·비대칭 progress·기존 항 보존을 검사한다. `check_before_virtual_object.py`는 가상 Alias XYZ·실제 회전·원본 상태 보존을 검사한다. `check_before_object_alias.py`·`check_before_push.py`·`check_coupled_push.py`는 실제 물리·부분 reset·추론 검사 진입점이다.

[PUSH](reference/stage1_push.md), [BEFORE PUSH](reference/stage2_before_push.md), [공통 Stage2/coupled](reference/stage2_coupled_push.md), [평가 규약](reference/evaluation.md), [Stage1 단독](../evaluation_single.md), [공통 정책](reference/ma_clean_scene_gta.md), [BEFORE](reference/stage2_before.md), [지표](reference/relation_diagnostics.md), [이관](reference/PORTING_GUIDE.md)을 참고한다. 이전 실험은 [legacy](legacy/README.md)에 보존한다.
