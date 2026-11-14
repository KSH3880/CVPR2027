# 레포 구조

유지하는 컨피그는 `stage1`, `stage2_before_nonemask_rsi433_near50`, `stage2_before`, `stage2_before_rsi433`, `stage1_push`, `stage2_before_push`, `stage2_coupled_push`, `stage2_before_push_near50`, `stage2_coupled_push_goal5`, `stage2_coupled_push_individual_goal`, `stage2_before_push_near50_pair50_source5`, `stage2_coupled_push_individual_goal_solo5`, `stage2_coupled_push_individual_goal_solo5_carry5`, `stage2_coupled_push_individual_goal_solo5_countnorm_coupled`, `stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none`, `stage2_coupled_push_individual_goal_solo5_keep1`, `stage2_coupled_push_individual_goal_solo5_shrink1`, `stage2_coupled_push_individual_goal_solo5_carry5_ground`, `stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg`, `stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg`, `stage2`, `stage2_distill`, `stage2_900`, `stage2_distill_900`, `stage2_distill_900_v2`, `stage2_distill_900_v3`, `stage2_distill_900_v4`, `stage2_900_v4_reward`, `stage2_distill_900_v5`이다. 실행 명령은 [config](config.md), 설치·추가 옵션은 [USAGE](USAGE.md)에 있다.

## 실행과 설정

| 경로 | 역할 |
| --- | --- |
| `EVAL_HANDOFF.md` | 전달용 평가 요약·버전 대조·실행 및 검증 순서 |
| `eval/finalize_v4_large_radius.py` | 대규모 평가의 정상 전원 낙상 조기 종료 CPU 재검증·기존 보고서 보존·큐 종료 후 결과 반영 |
| `eval/run_v4_large_radius.py` | V4 2k·32/64H Ind/Mixed64환경·seed1 탐색, R10부터5m 증가·180초 배치 제한·GPU6/7·SR/정책/반경 검증 |
| `eval/tables123_handoff.md` | PDF (15) Table 1/2/3 인수인계·현재 checkpoint/결과·공통 snapshot·실행 경로·검증 조건 |
| `tokenhsi/run.py`, `utils/parse_task.py` | MA 학습/평가·Stage1/BEFORE 모델 등록과 환경 생성 |
| `tokenhsi/utils/config.py`, `experiments.py` | CLI, 스물아홉 실험 계약, 학습/추론/resume env/sim 로딩 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/` | 환경·학습 YAML 각 스물아홉 개 |
| `tokenhsi/scripts/multi_agent/*_{train,test,vnc}.sh` | 실험별 학습·viewer·VNC 진입점 87개 |
| `tokenhsi/scripts/multi_agent/*_eval.sh`, `evaluate.sh` | 일반 정식 평가28개, Stage1/Stage1 PUSH 전용 1명 단독 평가2개·공통 runtime |
| `tokenhsi/scripts/multi_agent/stage2_teachers_{test,vnc,eval}.sh` | enable/coupled 두 PTH의 관계별 제로샷 viewer·공간 채점, 5명 이상 그룹 배치 |
| `tokenhsi/evaluate.py`, `rescore.py` | 독립 seed 실행·통계, 저장 trace CPU 재채점 |
| `tokenhsi/utils/evaluation_errors.py` | 공통 Point/Region Error의 유효 frame 최솟값·성공 필터·공동 그룹 중복 제거·seed 통계 |
| `eval/` | 평가 보조 작업공간. `compare_enabling_success.py`는 각각-ever·동시 성공, `compare_region_extent.py`는 XY80%·100%를 저장 trace에서 비교하며 메인 eval은 기존 위치 유지 |
| `eval/benchmark_policy.py` | TokenHSI/SCT/Ours V4의1/2/4/8/16H·장면1개 정책 정규화/actor 속도·PyTorch 상주/피크 메모리·실제 입력/strict checkpoint 로드 비교; 시뮬레이션 제외 |
| `eval/stage2_protocol_eval.py`, `stage2_distractor_physics.py`, `summarize_stage2_protocol.py` | 새 프로토콜 4명·512환경의 초기 snapshot 생성·관측/물리 분리 실행·초기 상태/CPU 재채점 검증·seed별 비교 |
| `eval/run_stage2_scalability.py`, `stage2_scalability_eval.py`, `stage2_scalability_layout.py`, `stage2_protocol_sampling.py`, `stage2_xy100.py`, `summarize_stage2_scalability.py` | 보조 경로의 Independent/Mixed 2·4·8·16명 평가·그룹별 distractor 배분·XY100% 채점·CPU 검증·기존4H Mixed 재사용 집계 |
| `eval/stage2_sampling_audit.py` | 선택적 초기 배치 단계별 RNG/상태 비교; 공유 상자 중복 초기화 진단 |
| `eval/bias_ablation.py`, `bias_ablation_eval.py`, `run_bias_ablation.py`, `summarize_bias_ablation.py`, `test_bias_ablation.py` | 추론 전용 SCT/DCT bias 출력 제거·첫 정책 입력/옵션 off 행동/가중치 불변 검증·Independent1H 지원·GPU6/7 병렬 평가와 기존 baseline 비교 |
| `eval/stage1_protocol_eval.py`, `stage1_protocol_policy.py`, `stage2_full_observation.py`, `compare_stage1_scalability.py` | 동일 초기 장면에서 Stage1 원본 정책 비교·가상 받침/개인 PUSH 입력 변환·native 행동 검증·Stage1/Full distractor attention 보존·읽기 전용 원본 재사용 비교표 |
| `eval/run_table2.py`, `table2_eval.py`, `table2_sampling.py` | Table 2 전용 단일 Enable 쌍/공동 PUSH 그룹의 Stage1·Stage2 평가·3명 지원·동일 초기 장면 검증 |
| `eval/table2_metrics.py`, `rescore_table2.py`, `summarize_table2.py`, `test_table2.py` | Time·t_coop 기록·손 원시 좌표 저장·접촉 거리/시작 변위/손 접근 CPU 재집계·표/seed별 집계·경계/실패/낙상 테스트 |
| `eval/rescore_table2_speed.py` | t_coop 시작의 XY 속도·지속 시간/손 접근 조건 CPU 비교·인원/seed별 비율·시작률·trial별 프레임 수 보존 |
| `eval/reaggregate_coupled_cooperation.py`, `coupled_cooperation_metrics.py` | SCT/FSM/Coop. FT/Ours 기존 공동 PUSH 궤적의 TeamHOI 방식 전체 누적 t_coop·원본64점 Jerk CPU 재집계·20/50cm 비교·seed/인원별 통계·원본 보존 검증 |
| `eval/reaggregate_enabling_post_at_time.py` | TokenHSI/SCT/FSM/Coop. FT/Ours 기존 Enable45개 기록의 최초 AT·최초 동시 성공·Post-AT Time CPU 재집계·SR/최초 성공 exact 검증·성공 trial 조건부 시간과 행동/seed별 통계 |
| `eval/reaggregate_v4_enabling_post_at_time.py` | 현재 V4 2k/Coop. FT6k로 Post-AT 비교 갱신·TokenHSI/SCT/FSM27개 재사용+18개 CPU 재계산·원시 SHA/현재 Table2 SR·Time 검증·기존 표 보존 |
| `eval/analyze_enabling_success_selection.py` | TokenHSI/Ours V4의4608개 동일 장면에서 성공 표본별 초기 거리/상자/회전각·공통 성공402개 조건부 시간·거리 구간별 SR CPU 비교·장면별 CSV |
| `eval/analyze_enabling_movement_speed.py` | 같은 두 모델의 성공 trace에서 사람 접근·물체 운반·전체/AT 전후 수평 이동속력 CPU 집계·공통 성공/각자 성공·접근 및 이동 문턱 민감도·구간별 CSV/제외 수 |
| `eval/analyze_cooperative_mechanisms.py` | PDF (15) §4.3용 Enabling4k/Coupled6k/Ours V4의90개 기록 CPU 분석·실패 단계/조건부 성공/공통 성공 시간/후행 준비·공동 근접/목표 정확도·분해 그림; 공식 평가 기준 유지 |
| `eval/reaggregate_enabling_completion_time.py` | 기존Enable54개 기록의Time Succ./All CPU 재집계·실패에900step 최대 시간 부여·원래 성공 건수/Time·원본 SHA 검증·Coop. FT4k/6k 구분·LLM Scheduler 제외 |
| `eval/reaggregate_v4_enabling_time.py` | 현재 V4 통합 표 다섯 모델 Enable45회에서 Time Succ./All만 CPU 재집계·실패 최대 시간·원본/다른 지표 보존 검증·시간 열 갱신 |
| `eval/run_coopft_tables.py`, `coopft_checkpoint_eval.py`, `summarize_coopft_tables.py` | 외부 Coop. FT 원본 환경/정책·기존4k snapshot을 읽기 전용으로 재사용한3k/6k84회 평가·GPU6/7 메모리 예약/동적 배정·실제 첫 입력/가중치/RMS·CPU SR/Error·최신20cm t_coop/Jerk 검증·4k 포함 비교 |
| `eval/group_spacing_eval.py`, `run_group_spacing.py`, `run_group_spacing_ready.py`, `run_group_spacing_comparison.py` | 그룹 강제 이격 해제/3m의64환경·seed1·8/16H Independent/Mixed 다섯 모델36회 확인·공통 초기 장면/CPU SR·Error 검증·GPU6/7 병렬 실행·기준 생성 인계/준비된 평가 동시 배정·FSM Independent 공유/기존 입력 차이 기록 |
| `eval/run_group_spacing_1m.py`, `group_spacing_1m_eval.py` | 같은 조건의1m·18회 추가 평가를 GPU5에서만 실행·Coop. FT GPU 제한의 프로세스 내부 예외·최대10개 병렬 작업/메모리 예약·공통 초기 장면/CPU SR·Error/간격 검증·검증 완료마다 중간 표 갱신 |
| `eval/run_group_spacing_2m.py`, `group_spacing_2m_eval.py` | 같은 조건의2m·18회를 GPU5/6/7에 배정·GPU5의1m 큐 시작 예약량 반영·GPU별 최대10개/10,000MiB 여유 메모리·동일 검증/중간 표 갱신 |
| `eval/run_group_spacing_formal.py` | 3m·512환경·seed1/6/7의8/16H Table1 정식42회·TokenHSI/SCT/FSM/Coop. FT6k·Ours 제외·GPU5/6/7 메모리 예약/동적 배정·공통 장면12개/CPU SR·Error 검증·seed별/평균±표본std 집계 |
| `eval/run_group_spacing_4m_16h.py` | 3m16H 중단 후 대체4m·16H·512환경·seed1/6/7 정식21회·공통 장면6개·기존 정식 실행기/검증 재사용·GPU5/6/7·Ours 제외/FSM Independent 공유 |
| `eval/group_spacing_sampling_probe.py` | 원래 초기 배치의 첫4회만 진단해 PUSH/전체 장면 거절·사람0.7m 미만·단계별 시간·거절 상태를 저장; SR 평가와 분리 |
| `eval/run_group_radius_screen.py`, `group_radius_probe.py` | 그룹 중심 고정 반경의8/16H Independent/Mixed 배치 생성만 검사·시도/시간 제한·GPU5/6/7 분산·실제 그룹 반경 검증·정책 SR 실행 없음 |
| `eval/run_group_radius_eval.py` | 고정 반경8H3m/16H7.5m·64환경·seed1·SCT/TokenHSI/Ours12회 정책 평가·GPU6/7 전용·공통 장면/실제 반경/첫 입력/CPU SR·Error 검증·중간 표 갱신 |
| `eval/run_group_radius_formal.py` | 같은 반경의512환경·seed1/6/7·TokenHSI/SCT/FSM/Coop. FT6k 정식42회·Ours 제외/FSM Independent 공유·GPU5/6/7·실제 반경/공통 상태/CPU SR·Error 검증 |
| `eval/run_v4_group_radius_eval.py` | V4 1.5k/2k·64환경·seed1·8H3m/16H7.5m Independent/Mixed8회·기존 공통 장면 재사용·GPU5/6/7·CPU SR/Error/정책 보존 검증·checkpoint별 비교 |
| `eval/run_v5_formal.py`, `summarize_v5_formal.py` | V5 2k Table1/2/3 42회·GPU6/7·공통 snapshot/검증 재사용·SR/Time Succ-All/t_coop 집계 |
| `eval/run_v4_formal.py`, `accelerate_v4_formal.py`, `v4_formal_eval.py`, `summarize_v4_formal.py` | 채택 V4 2k의512환경·seed1/6/7 Table1/2 42회·8H3m/16H7.5m·공통 장면/실제 입력/정책 보존/CPU 채점·최신20cm t_coop/Jerk·기존 비교군 통합 표 |
| `eval/group_radius_handoff.md` | 다른 Codex용8H3m/16H7.5m 그룹 중심 고정 반경 정의·유지 조건·실행 코드/checkpoint·64환경 결과·512환경 배치 검사와 SR 평가 구분 |
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
- `stage2_teachers.py`, `before_relation_fusion.py`: 원본 teacher별 네트워크·RMS 복원과 사람별 행동 선택. `utils/stage2_group_sampling.py`, `env/tasks/multi_agent/stage2_group_layout.py`는 추론 전용 순차 그룹 선택·8H/16H 정식 평가 기본 반경3m/7.5m(일반 viewer/그 외 인원은 기존5m 간격)·BEFORE 중심 복원·distractor 제외·카메라를 담당한다. `test_stage2_group_radius.py`, `test_stage2_teachers.py`, `tests/check_stage2.py`에서 실제 반경/그룹 내부 기하/RNG·가중치/담당 선택·관측 제외·물리/부분 reset을 검사한다.
- `coupled_relation_fusion.py`: ground A/B의 COUPLED 64차원 공유 임베딩·레이어/헤드 bias 투영·output projection 전 residual fusion. SELF/NONE만 A는 lookup 학습, B는 0 고정이다. `test_coupled_relation_fusion.py`에서 계산 동치·단독 생략·인원/셔플·gradient·전이/재로드를 검사한다.
- `utils/stage2_eval_sizes.py`: general 평가 전용 enable 행동별 크기 교집합·공유 물리 상자/질량 생성·첫 episode 관계/역할 고정. `test_stage2_eval_sizes.py`에서 범위·격자·solo/coupled 보존·native 역할 샘플링을 검사한다.
- `stage2_distill_agent.py`, `distill_replay.py`, `utils/stage2_distill_spec.py`: 고정 두 teacher의 사람별 평균 행동 모방·scene 단위 teacher/student 실행 전환·누적 replay·actor coordination/action head 업데이트·체크포인트/optimizer/replay 복원. [distillation 명세](reference/stage2_distill.md).
- `viewer_diagnostics.py`: viewer 전용 native 보상 성공 진단, 정식 평가와 별도.
- `ma_agent.py`, `ma_players.py`, `scene_normalizer.py`: 장면 단위 PPO/AMP·저장/복원·평가·정규화.
- `tokenhsi/utils/{edge_*,task_role_spec,before_spec}.py`: 그래프·역할·packet·checkpoint 계약.
- `tokenhsi/utils/{coupled_push_spec,coupled_push_scene}.py`: 공동 그룹·인원별 물리 bank·직선 목표·충돌 검사.임의 N≥1·M≥N과 물체 부족 장면의 과제 재샘플링을 지원한다.
- `tokenhsi/utils/premature_push.py`: V4 teacher 보상 실험의 공동 PUSH 접촉 준비 전 이동 감점.
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

논문 Table 1–3의 유일한 기준은 checkout 옆 공용 폴더의 [전체 프로토콜](../../evaluation_protocol/README.md)이다. [레포별 구현·실행 안내](../../evaluation_protocol/implementations.md), [통합 결과 목록](../../evaluation_protocol/results.md), [공통 계산 검증](../../evaluation_protocol/checks/README.md)을 함께 둔다. 루트의 `cvpr2027_evaluation_protocol.md`는 공용 문서 안내이며, 이전 본문은 `markdowns/legacy/cvpr2027_evaluation_protocol_20261111_before_shared.md`에 보존한다.

`tokenhsi/tests/`는 보상·배치·GTA·mask·RSI·전이·재로드·출력 경로를 검사한다. `before_test_utils.py`, `stage1_test_utils.py`는 CPU fixture다. `check_before_object_alias.py`·`check_before_push.py`·`check_coupled_push.py`는 실제 물리·부분 reset·추론 검사 진입점이다.

`test_coupled_scene_sampling.py`는 비PUSH 사람의 경로 장애물 유지·공동 상자의 전체 크기·distractor 재시도와 ON_TOP 받침 보존을 검사한다.

[통합 Stage2](reference/stage2.md), [V4 teacher 보상 PPO](reference/stage2_v4_reward.md), [PUSH](reference/stage1_push.md), [BEFORE PUSH](reference/stage2_before_push.md), [공통 Stage2/coupled](reference/stage2_coupled_push.md), [4명 초과 그룹 스폰](reference/group_spawn_over4.md), [평가 규약](reference/evaluation.md), [Stage1 단독](../evaluation_single.md), [공통 정책](reference/ma_clean_scene_gta.md), [BEFORE](reference/stage2_before.md), [지표](reference/relation_diagnostics.md), [이관](reference/PORTING_GUIDE.md)을 참고한다. 이전 실험은 [legacy](legacy/README.md)에 보존한다.
