# 레포 구조

유지하는 컨피그는 `stage1`, `stage2_before_nonemask_rsi433_near50`, `stage2_before`, `stage2_before_rsi433`, `stage1_push`, `stage2_before_push`, `stage2_coupled_push`, `stage2_before_push_near50`, `stage2_coupled_push_goal5`, `stage2_coupled_push_individual_goal`, `stage2_before_push_near50_pair50_source5`, `stage2_coupled_push_individual_goal_solo5`, `stage2_coupled_push_individual_goal_solo5_carry5`, `stage2_coupled_push_individual_goal_solo5_countnorm_coupled`, `stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none`, `stage2_coupled_push_individual_goal_solo5_keep1`, `stage2_coupled_push_individual_goal_solo5_shrink1`, `stage2_coupled_push_individual_goal_solo5_carry5_ground`, `stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg`, `stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg`, `stage2`, `stage2_distill`, `stage2_900`, `stage2_distill_900`, `stage2_distill_900_v2`, `stage2_distill_900_v3`이다. 실행 명령은 [config](config.md), 설치·추가 옵션은 [USAGE](USAGE.md)에 있다.

## 실행과 설정

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/run.py`, `utils/parse_task.py` | MA 학습/평가·Stage1/BEFORE 모델 등록과 환경 생성 |
| `tokenhsi/utils/config.py`, `experiments.py` | CLI, 스물여섯 실험 계약, 학습/추론/resume env/sim 로딩 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/` | 환경·학습 YAML 각 스물여섯 개 |
| `tokenhsi/scripts/multi_agent/*_{train,test,vnc}.sh` | 실험별 학습·viewer·VNC 진입점 78개 |
| `tokenhsi/scripts/multi_agent/*_eval.sh`, `evaluate.sh` | 일반 정식 평가26개, Stage1/Stage1 PUSH 전용 1명 단독 평가2개·공통 runtime |
| `tokenhsi/scripts/multi_agent/stage2_teachers_{test,vnc,eval}.sh` | enable/coupled 두 PTH의 관계별 제로샷 viewer·공간 채점, 5명 이상 그룹 배치 |
| `tokenhsi/evaluate.py`, `rescore.py` | 독립 seed 실행·통계, 저장 trace CPU 재채점 |
| `tokenhsi/utils/evaluation_errors.py` | 공통 Point/Region Error의 유효 frame 최솟값·성공 필터·공동 그룹 중복 제거·seed 통계 |
| `eval/` | 평가 보조 작업공간. `compare_enabling_success.py`는 각각-ever·동시 성공, `compare_region_extent.py`는 XY80%·100%를 저장 trace에서 비교하며 메인 eval은 기존 위치 유지 |
| `eval/stage2_protocol_eval.py`, `stage2_distractor_physics.py`, `summarize_stage2_protocol.py` | 새 프로토콜 4명·512환경의 초기 snapshot 생성·관측/물리 분리 실행·초기 상태/CPU 재채점 검증·seed별 비교 |
| `eval/run_stage2_scalability.py`, `stage2_scalability_eval.py`, `stage2_scalability_layout.py`, `stage2_protocol_sampling.py`, `stage2_xy100.py`, `summarize_stage2_scalability.py` | 보조 경로의 Independent/Mixed 2·4·8·16명 평가·그룹별 distractor 배분·XY100% 채점·CPU 검증·기존4H Mixed 재사용 집계 |
| `eval/stage2_sampling_audit.py` | 선택적 초기 배치 단계별 RNG/상태 비교; 공유 상자 중복 초기화 진단 |
| `eval/stage1_protocol_eval.py`, `stage1_protocol_policy.py`, `stage2_full_observation.py`, `compare_stage1_scalability.py` | 동일 초기 장면에서 Stage1 원본 정책 비교·가상 받침/개인 PUSH 입력 변환·native 행동 검증·Stage1/Full distractor attention 보존·읽기 전용 원본 재사용 비교표 |
| `eval/run_table2.py`, `table2_eval.py`, `table2_sampling.py` | Table 2 전용 단일 Enable 쌍/공동 PUSH 그룹의 Stage1·Stage2 평가·3명 지원·동일 초기 장면 검증 |
| `eval/table2_metrics.py`, `rescore_table2.py`, `summarize_table2.py`, `test_table2.py` | Time·t_coop 기록·손 원시 좌표 저장·접촉 거리/시작 변위/손 접근 CPU 재집계·표/seed별 집계·경계/실패/낙상 테스트 |
| `eval/rescore_table2_speed.py` | t_coop 시작의 XY 속도·지속 시간/손 접근 조건 CPU 비교·인원/seed별 비율·시작률·trial별 프레임 수 보존 |
| `FSM/` | SCT 26k + 관계별 state freeze/일회 해제 viewer·Enable3종/Coupled2/3/4명·공통 formal SR·VNC/로컬 진입점·상태 테스트. 명령은 `FSM/README.md` |
| `FSM/collision_probe.py` | 같은 자세/상자에서 freeze 충돌·해제 충돌·해제 무충돌을 비교하는 물리 확인 viewer. SR 평가와 분리 |
| `FSM/eval_run.py`, `eval_entry.py`, `eval_verify.py`, `eval_summary.py` | SCT26k+FSM Table1 Independent/Mixed·Table2 단일 관계 평가, 기존 초기 장면 재사용·CPU SR/FSM 재검증·seed별 집계 |
| `tokenhsi/utils/evaluation_protocol.py`, `learning/multi_agent/formal_eval.py` | 학습 인원/단독 구분·점/영역/BEFORE·개인/공유 중심 공동 PUSH·interaction group 성공률과 seed 통계·개인 낙상 제어 |
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
| `humanoid_ma_before_push.py`, `before_push_reward.py` | BEFORE solo PUSH 자산·배치·접촉 보상, 기존 BEFORE gate 유지 |
| `humanoid_ma_stage2.py`, `stage2_layout.py`, `stage2_reward.py`, `stage2_wait.py` | 통합 Stage2의 관계별 배치·AT/접촉/선후행 gate·대기 속도 감점 |
| `humanoid_ma_coupled_push.py`, `coupled_push_reward.py` | 공동 상자·가상 구역·그룹 gate·goal saturation |
| `push_goal_markers.py` | PUSH 목표의 viewer/VNC 선 표시만 추가; actor·reset·scale 수정 없음 |
| `viewer_design.py` | `DESIGN=1`의 distractor 관측/물리 분리·그룹 색 확장·Enable 역할별 채도/밝기·공유 상자 중간 톤·목표 구 숨김·표시 호출; 기본 실행은 비활성 |
| `viewer_goal_markers.py` | 디자인 전용 과제별 목표 테두리·윗면 표시·방향 화살표 |
| `humanoid_ma_before.py` | BEFORE 역할·공유 물체·AT progress·후행 초기화 |
| `scene_layout.py`, `scene_validation.py` | 활성 물체·distractor·loco 참조·물리 유효성 |
| `before_rsi_cache.py`, `native_rsi_cache.py` | 후행 pickUp/carryWith native 크기별 screening |
| `before_pickup_carry_rsi.py`, `before_relative_sampling.py` | RSI433 적용/fallback, near50 배치 |
| `size_rsi_cache.py` | Stage1 크기별 RSI 캐시 |
| `*_reward.py` | 과제 보상·BEFORE gate·CPA·물체 속도 |

## 정책과 과제 정의

- `tokenhsi/learning/multi_agent/amp_network_builder_ma.py`, `task_role_encoder.py`: 공통 GTA·task bias·사람 action MLP·actor/critic.
- 같은 디렉터리의 `before_network.py`, `human_coordination.py`: BEFORE 사람 attention·object packet·Stage1 전이·actor encoder 동결.
- `stage2_push_network.py`: BEFORE PUSH와 coupled의 동일 5과제/4family 정책·기본6N task/group suffix(keep1/shrink1은 선택 상대 ID N개를 더한7N)·4종 사람 relation·packet 기반 alias 전처리·Stage1 PUSH 전이. `human_coordination.py`는 선택한 관계의 query별 유효 key 수로 log-count 보정하며 `test_human_attention_count_normalization.py`에서 관계 비중·선호 보존·mask·전이/계약을 검증한다. `before_push_network.py`는 같은 builder의 호환 import다.
- `stage2_relation_fusion.py`, `utils/stage2_spec.py`, `utils/before_goal_sampling.py`: BEFORE/COUPLED 독립 임베딩과 공통 fusion MLP·세 장면 유형 1:1:1·충돌 없는 alias 배정·goalroute 배치. `test_stage2.py`에서 관계·토큰·보상·전이·gradient·재로드를 검사한다.
- `stage2_teachers.py`, `before_relation_fusion.py`: 원본 teacher별 네트워크·RMS 복원과 사람별 행동 선택. `utils/stage2_group_sampling.py`, `env/tasks/multi_agent/stage2_group_layout.py`는 추론 전용 순차 그룹 선택·5m 중심 배치·BEFORE 중심 복원·distractor 제외·카메라를 담당한다. `test_stage2_teachers.py`, `tests/check_stage2.py`에서 가중치/담당 선택·관측 제외·물리/부분 reset을 검사한다.
- `coupled_relation_fusion.py`: ground A/B의 COUPLED 64차원 공유 임베딩·레이어/헤드 bias 투영·output projection 전 residual fusion. SELF/NONE만 A는 lookup 학습, B는 0 고정이다. `test_coupled_relation_fusion.py`에서 계산 동치·단독 생략·인원/셔플·gradient·전이/재로드를 검사한다.
- `utils/stage2_eval_sizes.py`: general 평가 전용 enable 행동별 크기 교집합·공유 물리 상자/질량 생성·첫 episode 관계/역할 고정. `test_stage2_eval_sizes.py`에서 범위·격자·solo/coupled 보존·native 역할 샘플링을 검사한다.
- `stage2_distill_agent.py`, `distill_replay.py`, `utils/stage2_distill_spec.py`: 고정 두 teacher의 사람별 평균 행동 모방·scene 단위 teacher/student 실행 전환·누적 replay·actor coordination/action head 업데이트·체크포인트/optimizer/replay 복원. [distillation 명세](reference/stage2_distill.md).
- `viewer_diagnostics.py`: viewer 전용 native 보상 성공 진단, 정식 평가와 별도.
- `ma_agent.py`, `ma_players.py`, `scene_normalizer.py`: 장면 단위 PPO/AMP·저장/복원·평가·정규화.
- `tokenhsi/utils/{edge_*,task_role_spec,before_spec}.py`: 그래프·역할·packet·checkpoint 계약.
- `tokenhsi/utils/{coupled_push_spec,coupled_push_scene}.py`: 공동 그룹·인원별 물리 bank·직선 목표·충돌 검사.임의 N≥1·M≥N과 물체 부족 장면의 과제 재샘플링을 지원한다.
- `tokenhsi/utils/coupled_push_contact.py`: 개인 목표 실험의 공통 4면 선택·담당 구역 CONTACT·개인 도착 gate.
- `tokenhsi/utils/coupled_push_peer_selection.py`: viewer의 keep3/shrink3과 학습·추론의 keep1/shrink1 접촉 기반 상대 선택·독립 RNG·부분 reset. 선택되지 않은 COUPLED는 NONE으로 바꾸며, `test_coupled_peer_selection.py`·`test_single_coupled_peer.py`가 관계 전환·셔플·PPO 선택 재사용·전이를 검사한다.
- `tokenhsi/utils/coupled_push_assignment.py`: viewer 전용 좌우 정렬·Hungarian·교차 우선 구역 배정. reset 후 개인 목표/관측/접촉에 같은 배정을 연결한다. `test_coupled_assignment.py`는 배정 목적함수·회전/그룹·접촉/목표 연결·9명 이상 근사 탐색을 검사한다.
- `tokenhsi/utils/{before_push_spec,before_push_layout}.py`: solo PUSH 샘플링·계약·다중 사람 경로 배치.
- `tokenhsi/utils/{before_sampling,before_relative_sampling,before_object_alias}.py`: mix226/near50·토큰 alias.
- `tokenhsi/learning/multi_agent/stage1_push_network.py`: 5과제 정책·4 family AMP·선택적78k 전이.
- `tokenhsi/utils/{stage1_jointpush,jointpush_scene,jointpush_geometry,jointpush_rsi,jointpush_rsi_geometry}.py`: PUSH 계약·1/2명 장면·bank/모션/기하 검증.
- `tokenhsi/utils/stage1_sampling.py`: 기본 목표 정사각형 샘플링과 carry5 전용 상자 기준 거리/방향 샘플링.
- `tokenhsi/utils/{stage1_geometry,stage1_config,stage1_platforms,size_rsi}.py`: Stage1/BEFORE 공통 기하·RSI·물리 설정과 coupled carry5_ground 지면 배치 계약.
- `lpanlib/poselib/`: 기존 모션 배열 로딩 라이브러리.

## 검증과 문서

논문 Table 1–3의 평가 조건·테이블 구성·미정 항목은 루트의 [CVPR2027 평가 프로토콜](../cvpr2027_evaluation_protocol.md)에 둔다.

`tokenhsi/tests/`는 보상·배치·GTA·mask·RSI·전이·재로드·출력 경로를 검사한다. `before_test_utils.py`, `stage1_test_utils.py`는 CPU fixture다. `check_before_object_alias.py`·`check_before_push.py`·`check_coupled_push.py`는 실제 물리·부분 reset·추론 검사 진입점이다.

`test_coupled_scene_sampling.py`는 비PUSH 사람의 경로 장애물 유지·공동 상자의 전체 크기·distractor 재시도와 ON_TOP 받침 보존을 검사한다.

[통합 Stage2](reference/stage2.md), [PUSH](reference/stage1_push.md), [BEFORE PUSH](reference/stage2_before_push.md), [공통 Stage2/coupled](reference/stage2_coupled_push.md), [4명 초과 그룹 스폰](reference/group_spawn_over4.md), [평가 규약](reference/evaluation.md), [Stage1 단독](../evaluation_single.md), [공통 정책](reference/ma_clean_scene_gta.md), [BEFORE](reference/stage2_before.md), [지표](reference/relation_diagnostics.md), [이관](reference/PORTING_GUIDE.md)을 참고한다. 이전 실험은 [legacy](legacy/README.md)에 보존한다.
