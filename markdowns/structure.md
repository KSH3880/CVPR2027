# 레포 구조

Multi-agent 실험은 `stage1`, `stage1_v2`, `stage1_v2_fixed`, `stage2`, 공유 받침 성공 기준 비교 두 개와 `stage2_before`다. 컨피그 목록·기본 실행 명령·추론 옵션은 [config.md](config.md), 추가 옵션·checkpoint 규칙은 [USAGE.md](USAGE.md)를 따른다.

## 실행과 설정

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/run.py` | 학습·평가 등록과 실행 |
| `tokenhsi/data/cfg/multi_agent/stage1*.yaml` | 세 실험의 환경 설정 |
| `tokenhsi/data/cfg/train/rlg/stage1*.yaml` | 세 실험의 PPO/AMP/모델 설정 |
| `tokenhsi/scripts/multi_agent/stage1*_{train,test,vnc}.sh` | 실험별 학습·로컬 평가·서버 viewer, 총9개 |
| `tokenhsi/scripts/multi_agent/{runtime_env,mps_auto_env,gui_gpu_env,run-gui}.sh` | 공통 conda·GPU·MPS·VNC 준비 |
| `output/<실험>/<run>/` | 본학습 checkpoint·TensorBoard·진단, Git 제외 |
| `output/rsi_cache*` | 운영용 공용 RSI 캐시 |
| `output_etc/check/<실험>/{train,eval}/` | 확인 학습·평가·viewer 결과 |
| `tokenhsi/utils/output_paths.py`, `tokenhsi/scripts/multi_agent/output_paths.sh` | 확인 경로 검사·구 output 경로 전환 |
| `output_etc/check/stage1_cleanup_20261015/` | 실험 정리의 동등성·실행 검증·이동 경로표 |
| `output_etc/check/output_routing_20261015/` | 문서/출력 정리의 CPU·학습·경로 전환 검증 |

## 메인 구현

| 경로 | 역할 |
| --- | --- |
| `env/tasks/multi_agent/humanoid_ma.py`, `humanoid_ma_carry.py` | `tokenhsi/` 아래의 다중 사람·상자·선반 물리, scene 관측·RSI·AMP 통합 |
| `env/tasks/multi_agent/edge_ontop_task.py`, `edge_context_task.py` | graph/배치 reset·물리 재추첨·보상·trace·진단 |
| `env/tasks/multi_agent/edge_stage1_reward.py` | 자기 edge 합산·placement 성공 시 HOLDING 보상 포화·task packet |
| `env/tasks/multi_agent/edge_interaction_reward.py`, `edge_ontop_reward.py` | SIT/CLIMB/AT/ON_TOP 기하·상판 영역 성공 |
| `env/tasks/multi_agent/collision_reward.py`, `relation_reward.py` | 사람 CPA·graph 담당 물체 속도 패널티 |
| `env/tasks/multi_agent/size_rsi_cache.py` | 크기별 물리 screening과 프레임 cache |
| `utils/edge_scenario_spec.py`, `task_role_spec.py` | canonical 독립 과제·크기 조건 sampler·가변 인원·task packet |
| `utils/edge_context_spec.py`, `edge_ontop_spec.py`, `edge_interaction_spec.py` | graph 자료형·배치/부분 선택·순열·관계 ID |
| `utils/edge_stage1_spec.py`, `relation_task_spec.py` | 남은 Stage1 reward·checkpoint 계약 |
| `utils/size_rsi.py`, `unified_training.py` | 실제 크기·RSI pool/fallback·과제별 AMP family와 모델 계약 |
| `utils/stage1_platforms.py`, `stage1_sampling.py` | 선반 높이·graph binding·원본 거리 분포·구 stage1 snapshot 호환 |
| `utils/stage1_v2.py`, `stage1_v2_fixed.py` | v2 기하·실제 높이 phase·fixed env/sim 복원 |
| `learning/multi_agent/amp_network_builder_ma.py`, `task_role_encoder.py` | 공유 task bias·사람 residual MLP·GTA actor/critic |
| `learning/multi_agent/ma_agent.py`, `ma_players.py`, `scene_normalizer.py` | scene 단위 PPO·AMP·checkpoint·평가·노드별 정규화 |
| `learning/multi_agent/edge_context_eval.py` | 현재/누적 task·scene 성공 평가 |

표의 구현 경로는 `tokenhsi/` 기준이다. `edge_ontop_*`, `edge_context_*`, `task_role_*`는 세 실험이 사용하는 공통 구현 이름이다. 파일명만 보고 제거하지 않는다.

## 검증

- `test_output_paths.py`: 본학습/확인 분리·기존 경로 전환·경로 이탈 거부.
- `tokenhsi/tests/stage1_test_utils.py`: 과거 실험에 의존하지 않는 CPU fixture.
- `test_stage1.py`, `test_stage1_sampling.py`, `test_stage1_v2.py`, `test_stage1_v2_fixed.py`: 세 config·성공·RSI·선반·snapshot 계약.
- `test_stage1_policy.py`, `test_stage1_gta.py`, `test_stage1_wrapper.py`, `test_ma_scene_features.py`: task/edge/token 순열·gradient·optimizer·GTA 수식·정규화.
- `test_generic_task_scenario.py`, `test_graph_box_speed_penalty.py`, `test_agent_cpa_collision.py`, `test_gui_gpu.py`: 가변 인원·담당 물체·CPA·GPU 선택.
- `smoke_stage1_sim.py`, `smoke_stage1_v2_training.py`: 실제 reset/선반/CPA 및 학습 중 RSI/AMP 정렬 검사.

## 문서와 별도 자료

원본 단일인 TokenHSI의 `basic_interaction_skills/`, `adapt_interaction_skills/`, `comp_interaction_skills/`, `multi_task/`, `longterm_task_completion/` 설정·코드와 데이터 전처리는 이번 multi-agent 실험 정리 범위 밖이다. [원본 태스크 참고](legacy/tokenhsi_original_carry_sit_climb.md), [v2 비교](legacy/tokenhsi_vs_stage1_v2_train_eval.md)를 참조한다.

설치·추가 실행 옵션은 [USAGE.md](USAGE.md), 머신 이관은 [PORTING_GUIDE.md](reference/PORTING_GUIDE.md), 지표는 [relation_diagnostics.md](reference/relation_diagnostics.md)를 확인한다.

일상 안내는 `README.md`, `changelog.md`, `AGENTS.md`와 `markdowns/{config,structure,USAGE}.md`다. 상세 모델·진단·포팅은 `markdowns/reference/`, 과거 변경·분석은 [legacy](legacy/README.md)에 보관한다. MPS 운영 안내는 [mps/README.md](../mps/README.md)를 따른다.

## Stage2 추가 구현

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_stage2.py`, `stage2_scene.py` | fixed 기반 4명 그룹·8개 장면 상자·공동 파지·GraspView·CPA 예외 |
| `tokenhsi/utils/stage2_spec.py`, `stage2_amp.py`, `stage2_rsi.py` | 그룹 graph·task별 AMP·15cm 공동 RSI 캐시와 누락 시 자동 생성 |
| `tokenhsi/env/tasks/multi_agent/stage2_solo_rsi_cache.py` | 단독 RSI의 4명 asset bank 연결 |
| `tokenhsi/learning/multi_agent/stage2_network_builder.py`, `human_coordination.py`, `grasp_view.py` | 현재 stage1 가중치 전이·동결·human SA4·사람별 앵커 관측 |
| `joint_carry/` | 복사한 공동 reference·보정·크기별 RSI 생성과 물리 screening |
| `joint_carry/render_size_rsi.py` | 원본/대표27크기의 native asset RSI 영상·검사 통과 프레임 표시 |
| `joint_carry/single_source.py`, `single_source_manifest.json` | 1인 원본 B19/B20/B21과 공통 다리 보정·공동 배치 생성 |
| `joint_carry/audit_size_rsi.py`, `tokenhsi/utils/grasp_geometry.py` | 생성 범위의 데이터 검사·RSI/GTA/리워드 공통 앵커 |
| `tokenhsi/tests/test_stage2.py`, `check_stage2.py` | CPU 계약·전이·gradient와 GPU reset/추론·부분 reset 검증 |

세부 계약과 데이터 준비는 [Stage2 참고](reference/stage2.md)에 있다.

## 공유 받침 비교 실험

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_shared_{full_success,region_success}.yaml` | 전체 받침/선택 구역 성공 비교, 4명·1024환경 |
| `tokenhsi/scripts/multi_agent/stage2_shared_*_{train,test,vnc}.sh` | 두 실험 전용 실행, 기본 GPU5 |
| `tokenhsi/utils/shared_support.py` | 순차 그룹 sampler·최소 토큰 수·구역/distractor 압축 배치·강체 변환·checkpoint 계약 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_shared_support.py` | 물리 bank·공유 관측·loco reset·현재 최근접 구역 선택 |
| `tokenhsi/env/tasks/multi_agent/shared_support_{reward,scene}.py` | 성공 영역 비교·원본 물리 높이·실제 물체 배치 검사 |
| `tokenhsi/learning/multi_agent/shared_support_network.py` | padding 없는 공통 장면 encoder·물리 ID 기반 coordination |
| `tokenhsi/tests/{test,check}_shared_support.py` | CPU 계약·보상·전이와 GPU 추론/부분 reset 검사 |

설정·관측·초기화 계약은 [공유 받침 참고](reference/shared_support.md)에 있다.

## BEFORE 실험

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_before.yaml`, `tokenhsi/scripts/multi_agent/stage2_before_{train,test,vnc}.sh` | GPU5 기본 4명·1024환경·8물체, 가변 N명·M≥N 물체 학습/평가 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_before_ground.yaml`, `tokenhsi/scripts/multi_agent/stage2_before_ground_{train,test,vnc}.sh` | BEFORE 기반 바닥 시작·AT 바닥 목표, 인공 받침 비활성 실험 |
| `tokenhsi/utils/before_spec.py` | 2인 BEFORE/단독 sampler·물체 예산 초과 시 graph 재샘플링·크기 조건 역할·과제/distractor 토큰·checkpoint 계약 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_before.py`, `before_reward.py` | 현재 AT 성공 gate·loco 배치·일반 물리 bank·reset 시 과제 소유자 색상 갱신 |
| `tokenhsi/env/tasks/multi_agent/before_follower_near.py`, `tokenhsi/utils/before_layout.py` | 평가 전용 무작위 후행 근접 배치와 수평 강체 변환 |
| `tokenhsi/learning/multi_agent/before_network.py`, `human_coordination.py` | 공통 장면 encoder·SELF/NONE/단방향 BEFORE SA4 |
| `tokenhsi/tests/test_before.py`, `test_before_follower_near.py`, `check_before.py` | CPU gate/분포/전이·GPU 과제 연결/부분 reset/추론 검사 |

추가 혼합 실험 `stage2_before_ground_mix50`의 env/train YAML은 `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_before_ground_mix50.yaml`, 실행은 `tokenhsi/scripts/multi_agent/stage2_before_ground_mix50_{train,test,vnc}.sh`다. `tokenhsi/utils/before_sampling.py`가 쌍별 모드와 후행 거리를 추첨하며 `tokenhsi/tests/test_before_sampling.py`가 혼합 비율·거리·기존 경로·checkpoint 분리를 검사한다.

세부 계약은 [BEFORE 참고](reference/stage2_before.md)에 있다.

`stage2_before_ground_mix334_1200`은 같은 `before_sampling.py`의 3단계 거리 모드와 선행 접근 거리 축소를 사용한다. env/train YAML과 전용 train/test/VNC는 각각 `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_before_ground_mix334_1200.yaml`, `tokenhsi/scripts/multi_agent/stage2_before_ground_mix334_1200_{train,test,vnc}.sh`에 있다.

`stage2_before_ground_mix226_1200`의 env/train YAML과 전용 train/test/VNC도 같은 디렉터리에 있다. `before_sampling.py`의 20/20/60·ON_TOP 전용 거리·상자 기준 AT 목표 샘플러를 `humanoid_ma_before.py`에서 BEFORE 쌍에만 적용한다. `tokenhsi/tests/test_before_mix226.py`는 분포·거리·목표 제외 영역·기존 설정/체크포인트 분리를, `check_before.py`는 실제 배치와 단독 목표 보존을 검사한다.


`stage2_before_ground_mix226_atprogress_1200`의 env/train YAML과 전용 train/test/VNC는 같은 디렉터리에 있다. `before_reward.py`의 `predecessor_at_progress`와 `humanoid_ma_before.py`의 기하 평가 연결이 BEFORE 후행 progress의 목적지만 선행 AT 목표로 바꾼다. `test_before_atprogress.py`는 보상·gate·과제/edge/부분 배치 보존·checkpoint 분리를, `check_before.py`는 물리 환경의 목표 연결과 부분 reset을 검사한다.


`stage2_before_ground_mix226_actorbias0_1200`과 `stage2_before_ground_mix226_atprogress_actorbias0_1200`은 같은 위치의 전용 env/train YAML·train/test/VNC를 사용한다. `relationGraph.actor_human_bias`를 `stage2_network_builder.py`에서 actor coordination에만 전달하며 `human_coordination.py`가 BEFORE bias만 파라미터로 저장한다. `test_before_actor_bias0.py`는 고정 0·gradient·actor/critic 분리·초기 출력 동등성·체크포인트 분리를 검사한다.


`stage2_before_ground_mix226_atprogress_carry50_1200`의 전용 env/train YAML·train/test/VNC는 위와 같은 디렉터리에 있다. `env/tasks/multi_agent/before_carry_rsi.py`는 carryWith 전용 크기별 물리 캐시를, `utils/before_carry_rsi.py`는 후행 선택을 담당한다(`tokenhsi/` 기준). `humanoid_ma_before.py`에서 RSI 사람/상자를 함께 배치하며 `test_before_carry_rsi.py`는 선택·변환·설정/체크포인트 분리를, `check_before_carry_rsi.py`는 GPU 배치·AMP·부분 reset·loco/RSI 평가를 검사한다.

- `tokenhsi/utils/before_full_rsi.py`: 단독·BEFORE 전체 Stage1 RSI 확률·실험 계약과 평가 전용 선후행 carryWith 선택.
- `tokenhsi/env/tasks/multi_agent/before_full_rsi{,_cache}.py`: 실제 물체 binding별 물리 프레임 캐시·역할 배치·크기/합성 fallback·진단.
- `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage2_before_ground_mix226_atprogress_fullrsi_1200.yaml`, 동명 train/test/VNC: 전체 RSI 별도 실행 경로.
- `tokenhsi/tests/test_before_full_rsi.py`, `check_before_full_rsi.py`: CPU 계약/확률·GPU reset/추론/부분 reset 검증.

`stage2_before_ground_mix226_atprogress_adapter_1200`의 env/train YAML·전용 train/test/VNC는 같은 디렉터리에 있다. `before_network.py`가 actor hidden Linear1·2의 공유 0-init adapter를 추가하며 `stage2_network_builder.py`가 Stage1 전이와 기존 head 전체 동결을 처리한다. `test_before_adapter.py`는 초기 출력 동등성·gradient·동결·전이·checkpoint 분리를 검사한다.

`stage2_before_ground_mix226_atprogress_adapter_bias0_1200`은 adapter 기반 actor·critic SELF/NONE bias0 비교 설정으로 같은 위치의 env/train YAML·train/test/VNC를 사용한다. `stage2_network_builder.py`에서 `actor_human_bias`·`critic_human_bias`를 각각 전달하고 `before_spec.py`에서 checkpoint 계약을 분리한다. `test_before_adapter_bias0.py`는 양쪽 고정0·BEFORE/QK gradient·전이·동결·재로드를 검사한다.

`stage2_before_ground_mix226_atprogress_bias0_1200`은 adapter 없는 AT progress의 actor·critic SELF/NONE bias0 설정으로 같은 위치의 env/train YAML·train/test/VNC를 사용한다. actor encoder/RMS는 동결하고 기존 action head는 학습한다. `test_before_adapter_bias0.py`에서 adapter 유무별 고정0·head 학습/동결·전이·재로드를 함께 검사한다.

`stage2_before_ground_mix226_atprogress_nonemask_1200`의 env/train YAML·전용 train/test/VNC는 같은 위치에 있다. `human_coordination.py`의 `before_only_mask_none` 모드가 actor·critic 사람 SA의 NONE을 차단하며 SELF bias는0, BEFORE bias만 학습한다. `test_before_none_mask.py`는 연결 차단·gradient·순서 변경·전이/재로드·checkpoint 분리를 검사한다.
