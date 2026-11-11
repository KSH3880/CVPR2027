# 변경 기록

## 2026-11-11

### 사용자 지정 기존 freeze로 soft wait 전체 재평가

- 사용자 정정에 따라 evaluate_saved_api.py에 --freeze legacy/snapshot_resume 선택 옵션을 추가하고 기본값을 legacy로 변경. legacy 코드 자체는 수정하지 않음. 각 결과에 freeze_selection.json으로 audit 일치 검사 기록.
- 같은 api_replay_soft_wait_20261111_120506의 1,920개 새 스케줄·같은 초기 reference·Stage1 26k로 기존 legacy freeze 평가30회 재제출. GPU6/7, 64환경·900step. API 재호출0이며 snapshot 평가 결과 보존.
- CPU44 tests 및 구문 검사 통과. 실행 output_etc/check/eval_soft_wait_legacy_20261111_135137, 완료 여부는 status.json 기준. 기존 학습/MPS/ksh 변경 없음.


### Soft wait API 응답 전체 시뮬레이션 평가 제출

- evaluate_saved_api.py로 api_replay_soft_wait_20261111_120506의 1,920개 새 스케줄을 평가. 기존 reference를 새 출력 트리에서 참조하여 capture/API 반복 없이 Mixed2/4/8/16 + Table2 6설정 × seed1/6/7 총30회, 각64환경·900step을 GPU6/7에 순차 제출한다.
- Stage1 26k + snapshot_resume freeze. 실행 가능한 검증 실패3건도 원문 그대로 포함하며 스케줄 수정 없음. 기존 전체 결과는 legacy freeze이므로 프롬프트 효과만의 통제 비교라고 주장하지 않음.
- 각 실행 후 초기 상태·실제 release tick·CPU rescore·Table2 replay 및 snapshot 속도복원 오차0 검증, 전체 완료 후 summary.json/report.md 자동 집계. 소스 해시 변경 시 다음 작업 제출 거부.
- 실행 output_etc/check/eval_soft_wait_snapshot_20261111_131017. 실제 완료/실패 상태는 status.json 기준. GPU 허용6/7, ksh 쓰기/MPS 제어/기존 프로세스 종료 없음.


### 초기 장면 재사용·soft wait 프롬프트 API 재호출

- replay_saved_api.py 추가. 기존 full_mixed_table2_env64_20261111_003631의 1,920 request를 검증·그대로 재사용하고 현재 system prompt로 재호출한다. 물리 초기화/rollout/GPU 작업 없이 API 전용이며 기존 응답은 보존한다.
- gpt-6-luna, reasoning effort medium, 최대16병렬, timeout180초, 자동재시도0. 현재 prompt 원문·SHA256·장면별 request 해시·raw/validated response·실패 원인·진행 상태를 별도 저장. 호출 중 prompt 변경 시 HTTP 전송 전에 해당 호출 거부.
- 실행 output_etc/check/api_replay_soft_wait_20261111_120506. 완료 여부와 실패 수는 해당 status.json 기준이며 시뮬레이션 평가는 아직 제출하지 않음.


### FSM vs LLM 다른 16인 Mixed 장면 paired 비교

- compare_fsm_llm_episodes.py로 기존 env64 seed1/6/7 reference와 API 스케줄 재사용. 각 seed의 FSM/snapshot LLM을 같은 GPU에서 순차 실행(GPU6 seed1·7, GPU7 seed6), 각64환경·900step, 6개 실행 exit0. API 추가 호출 없음.
- 실행 불가능한 seed6 env1을 양쪽 집계에서 제외하여 191장면·선행544명을 대응. packet/predecessor/groups/goals/sizes exact 일치 검증. 원본 FSM/새 snapshot LLM/기존 LLM을 별도 집계.
- 선행 성공278/280/284명(51.10/51.47/52.21%), 선행 낙상120/176/171명(22.06/32.35/31.43%). 새 freeze의 단일 장면 낙상 개선은 전체 표본에서는 재현되지 않음. 성공은 region ever 기준, 낙상과 상호 배타적이지 않음.
- FSM과 snapshot LLM에서 선행 모두 성공207, FSM만 성공71, LLM만 성공73, 둘 다 실패193. 성공 여부 차이는 존재하지만 선행 성공률은 거의 동일하고 LLM 낙상률이 높음. 스케줄·관측 갱신·freeze 구현 차이가 함께 있으므로 특정 원인으로 단정하지 않음.
- 최초 후처리에서 excluded 레코드를 정수로 간주한 인덱스 오류를 검출하고 environment 필드를 읽도록 수정. GPU 결과는 재실행/수정하지 않고 summarize_fsm_llm_episodes.py로 재집계. 이전 집계 오류는 status.json에 보존.
- 결과 output_etc/check/fsm_llm_paired_16h_env64_20261111/comparison.json, 선행별 predecessor_pairs.csv. 모든 GPU 실행 종료, ksh/MPS/기존 학습 변경 없음.


### FSM 비의존 snapshot freeze 실험

- sct_scheduler/snapshot_freeze.py 신규 구현. FSM 상태·접촉·성공 gate 없이 초기 snapshot을 대기 중 고정하고, release 시 root/DOF/body 초기 속도까지 1회 복원한다. 물체는 투영하지 않으며 충돌은 유지한다. 실제 kinematic actor 전환은 아니다.
- batch_eval_entry.py의 SCT_FREEZE_IMPL=snapshot_resume로 선택하며 기본 legacy 실행은 보존. 같은 모델·장면·저장 스케줄로 GPU6 1환경16명900step 실행, 정상 종료. API 추가 호출 없음.
- 최초 delayed release(frame45) 이전 root 궤적이 기존과 exact 일치, frame45부터 차이 발생. 복원 오차0, 초기 root/body gap0. 낙상6→3, ON_TOP A04는 여전히 실패(해제 후 물건 중심 최대 높이0.2055m). 단일 장면 결과로 일반적 개선을 주장하지 않음.
- CPU 테스트44개 통과. 새 테스트는 batch actor 매핑, frozen 속도0, release 시 초기 속도 복원, 이후 재복원 방지, simulator 제출 actor IDs, hook 정리를 확인한다.
- 결과 output_etc/check/viewer_one_env_20261111_015804/snapshot_freeze_comparison.json 및 teacher_runtime 동명 경로/snapshot_resume. 기존 전체평가 결과는 수정하지 않음.


### 동일 16인 장면 FSM 동작 진단

- 원본 FSM eval_entry/controller/state를 teacher_runtime/fsm_diagnostic에 그대로 복사하고 ksh 쓰기 금지 audit guard로 GPU6에서 1환경·900step 실행. 초기 observed_scene exact 검증 통과, 기존 headless resolved_env 설정 일치, Stage1 native actor/RMS 검증 통과.
- GPU7에서는 fsm_schedule_diagnostic으로 원본 freeze/관측 갱신 경로에 저장 LLM release frame만 적용한 대조 실행. 운영 scheduler와 기존 전체 평가 결과는 수정하지 않음. 두 실행 모두 exit 0, API 추가 호출 없음.
- 기존 LLM / 원본 FSM / FSM freeze+동일 LLM 스케줄의 낙상은 각각 6/5/2명. ON_TOP A04는 세 실행 모두 실패. 원본 FSM은 frame634(21.13초)에 실제 선행 성공 후 해제했지만 물건 높이가 0.15m에 머물고 frame894에 낙상.
- 같은 스케줄 대조도 첫 프레임부터 root 최대 약0.00597m 차이. 관측 초기화·갱신/상태 제출 경로와 GPU 차이를 추가 분리하기 전에는 freeze 코드 결함이나 LLM 타이밍만의 원인으로 단정하지 않음. 단일 장면 진단이며 집계 성능 비교 아님.
- 결과 output_etc/check/viewer_one_env_20261111_015804/fsm_behavior_comparison.json, 재집계 compare_fsm_behavior.py. 기존 viewer/headless 동일 스케줄 재생은 trace 전체 exact 일치(viewer_headless_comparison.json).


### Viewer를 1환경으로 제한

- 사용자 요청에 따라 기존64환경 viewer에 Q 키를 보내 GPU 동기화 정상 종료. viewer 진입점에 num_envs==1 검사를 추가하고 선행/후행 색상·카메라를 입력 관계에서 선택하도록 수정.
- GPU6에서 새1환경·16명 seed1 snapshot 생성 후 현재 prompt로 API1회 호출, 저장 스케줄을 6086 viewer에 적용. 기존64환경 평가의 실패 장면과 동일하다고 주장하지 않음. 경로 output_etc/check/viewer_one_env_20261111_015804 및 teacher_runtime 동명 경로.

### GPU6 평가 장면 viewer

- 별도 view_batch_entry.py로 현재 batch evaluator를 그대로 감싸 GPU viewer 연결. 64환경/16명 seed1, 같은 reference·Stage1 26k·원본 API 스케줄·낙상 처리로 실행하며 env0의 A11→A04 실패 사례를 표시한다. 기존 1환경/GPU7 viewer는 수정하지 않음.
- GPU6 UUID 실행 확인, noVNC6086, 첫 control step 후 일시정지. 초기 roots/goals/groups/logical_box_order exact, 최대 root 이동0.091865m 및 root/body 차이0.000021579m 확인. 화면 캡처로 준비 상태 확인. 색상은 선행 A11 빨강/후행 A04 하늘색, Space 재생·정지/숫자 카메라/Q 정상 종료.
- 실행 경로 teacher_runtime/output_etc/check/viewer_gpu6_eval16_20261111_015305. API 추가 호출·기존 평가/학습 종료·MPS 변경 없음. 전체 궤적의 headless 일치 여부는 아직 검증 전이며 초기 상태/첫 step 검증만 완료.

### Mixed + Table2 전체 64환경 평가 큐

- 원본 최신 run_suite의 42개 조합(Independent/Mixed 2/4/8/16H × seed1/6/7 + Table2 6설정 × 3 seeds)을 확인. 사용자 최종 요청에 따라 Independent 전용 12회를 제외하여 30회 × 64환경 = 1,920장면/API 호출로 구성.
- Mixed 12회와 Table2 enable SIT/CLIMB/ON_TOP·coupled2/3/4의18회, 900step, Stage1 원본26k 정책. Mixed 내부 독립 agent는 0초 출발을 prompt/검증에 명시. GPU6/7 각각 한 subprocess, API 최대16병렬, 자동 재시도·출발 시각 보정 없음.
- 현재 exporter로 새 snapshot에서 request를 생성하며 명목 속도 지시 부재 검사. 실행 코드와 prompt snapshot/hash 저장, 실행 중 코드가 변하면 다음 단계 제출 중단. 과거 API 결과 재사용 없음.
- Table2까지 초기 freeze 경로 연결. 실행 불가능한 장면 제외 시 formal trace/initial과 Table2 tracker/cooperation 배열을 같은 인덱스로 필터하고 full initial 별도 보존. 원본 CPU rescore·Table2 replay 및 실제 release tick 검증 자동 수행.
- CPU 43 tests 및 30개 조합/구문 검사 통과. GPU 사전 검증은 mixed16/enable_sit/coupled3 × 2환경×30step이며 Table2는 마지막 슬롯을 제외하는 합성 검사 포함, 논문 수치에서 제외.
- 최초 사전 검증(003425)은 freeze로 0이 된 초기 속도와 원본 속도의 직접 비교 때문에 CPU 검사에서 중단(API 0건). 속도 초기화만 반영해 검사 수정 후 기존 궤적 재검증 통과. 실패 기록은 보존.
- 활성 실행 경로 output_etc/check/full_mixed_table2_env64_20261111_003631. status.json이 실제 진행 상태 기준. 원본 ksh·기존 학습·MPS 수정/종료 없음.
- Mixed16 / Enable SIT2 / Coupled3 사전 GPU 검증 모두 통과. Table2의 제외 슬롯 필터·CPU/GPU SR·Error·Time/contact replay 일치 확인. 64환경 본 캡처 시작, 첫4개 설정 캡처 완료 및 실패0 확인 후 자동 큐 유지.

## 2026-11-10

### 응답 limitations 필드 제거

- 사용자 요청에 따라 출력 schema, 필수 필드 검증, 합성 응답 및 preflight에서 limitations 제거. 출력은 predictions/schedule/predicted_episode_events 세 필드이며 가정·불확실성은 prediction basis/confidence로 설명한다.
- 입력 environment_spec의 환경 설명은 유지. 과거 응답은 batch 로드 시 메모리에서만 구 필드를 제거하며 저장된 원본은 수정하지 않음.
- tokenhsi_koo CPU 42 tests 통과. API/GPU 재실행 없음.

### 응답 status 필드와 상태 분기 제거

- 사용자 요청에 따라 system/user prompt에서 ok/infeasible 및 status 필드를 제거. 통합 출력은 predictions/schedule/predicted_episode_events/limitations 네 필드이며 완전한 스케줄을 요구한다.
- 검증기, CLI, episode/Isaac adapter, batch API/평가 경로, 합성 fixture를 새 형식에 맞춤. 빈 스케줄·제약 위반·시간 합산 오류는 계속 거부한다. 기존 배치 응답의 status=ok는 GPU batch 로드 시 메모리에서만 제거하며 저장된 과거 응답과 평가 결과는 보존한다.
- 프로젝트 tokenhsi_koo 환경에서 CPU 41 tests 통과. 기본 Python에는 torch가 없어 테스트 환경을 바로잡아 재검증함. API/GPU 재실행 없음.

### 출력 규칙 통합

- system prompt의 상태별 출력 규칙을 하나의 OUTPUT RULES 목록으로 통합하고 중복 표현을 정리. JSON schema 및 ok/infeasible 처리와 타이밍 제약은 유지. 코드 변경·API/GPU 실행 없음.

### 사용하지 않는 계획 경로 설명 삭제

- 현재 입력에 계획 경로가 없으므로 system prompt의 INFORMATION BOUNDARY에서 planned geometric path 문단 삭제. 실행 전 정보 및 장면 데이터/지시 구분 규칙은 유지. API 재호출 없음.

### TIMING ESTIMATES의 명시적 속도 지시 제거

- 사용자 요청에 따라 system prompt의 해당 절에서 nominal/walking/measured speeds 언급과 속도·준비시간 가정 지시를 제거. 참조 대상이 없어진 Adapt them 문장도 삭제.
- 관측·가정 설명, active duration에서 초기 대기 제외, 미보정 추정의 low confidence 및 수치 출력 조건은 유지. 코드·기존 결과 변경 및 API/GPU 실행 없음.

### Enabling 최소 대기 조건 제거

- 사용자 요청에 따라 system/user template/16인 예시에서 earliest release 요구, 대기 최소화 목적, safety margin 금지를 제거. 선행 release=0 및 후행 상호작용 event가 선행 enabling event 이후라는 조건만 유지하고 추가 대기를 허용한다. 고정 시간 간격은 지정하지 않는다.
- 응답 검증은 최소 해와의 동등성 대신 제출된 episode event의 실제 부등식/동기화 조건을 검사한다. 참고 solver는 합성 fixture/preflight에만 유지. 기존 API 응답 및 완료 평가 결과는 수정하지 않음.
- CPU 41 tests 통과: 추가 enabling 대기 허용, 동시 event 거부, 선행 release=0, coupled 동기화, 시간 합산 검증 포함. API/GPU 재실행 없음.

### 저장된 원본 스케줄 실행 평가 (검증 오류도 포함)

- 사용자 요청에 따라 예측 시간·제약·필드 오류를 실행 차단 조건에서 분리. 기존 strict 검증은 유지하며 명시적 execution-only 평가에서 agent별 유한·비음수 출발 시각 완비만 요구한다. 출발 시각 재계산과 API 재호출 없음.
- 1,536건 중 1,535건 포함(기존 검증 실패 67건 포함). seed7/env0102만 A04 release 누락으로 제외. 원본 API 파일은 읽기 전용 재사용하며 새 실행 manifest에 모든 오류/제외를 기록.
- 초기 샘플 재현을 위해 GPU batch는 512 슬롯 유지. 제외 슬롯은 horizon 전체 동결하고 로컬 formal_eval에서 점수 및 trace의 모든 환경 축을 필터링한다. 원래 환경 인덱스 매핑 별도 저장, 초기 snapshot은 512 슬롯 그대로 보존. 원본 ksh 파일은 수정하지 않음.
- 실행: output_etc/check/eval16_raw_20261110_214845, GPU7, Stage1 26k, seed1/6/7 순차 900step. CPU 39 tests 통과 및 전건 prepare 검사 완료. seed1 초기 snapshot 4항목 exact, 첫 step 최대 root 이동 0.099482m / root-body 차이 0.000086317m. 최종 GPU 결과는 실행 상태 파일 기준.
- 완료: seed별 512/512/511환경, 900step. 원본 CPU rescore가 세 seed 모두 GPU 성공 판정·JSON count·오차에 일치. 24,560 agent 해제 시각이 원본 schedule의 ceil(control_dt)와 전부 일치, 제외 슬롯은 해제 0회. 새 API 호출 0.
- Region 그룹 성공률(seed 평균±표준편차): Independent 76.57±0.81%, Enabling 10.63±0.67%, Coupled 53.34±2.35%, Mixed 46.84±0.69%. Enabling은 원본 simultaneous 성공 기준. summary.json / verification.json / release_verification.json에 원시 집계와 검증 보관.

### 원본 16인 설정의 Stage1 + LLM 일괄 평가 시작

- 사용자 승인: mixed 16명/32 object tokens, seed1/6/7 × 512환경 = 1,536장면, 900step. 원본 run_stage2_scalability/eval 프로토콜과 동일 V2 carrier + Stage1 26k, XY100%, distractor 관측 유지/물리 제거, 그룹간 배치와 full attention 사용. 원본 eval 핵심 파일 해시 일치 확인.
- 초기 순간 이동 수정: before_action(0)의 frozen-only indexed 제출이 pending 전체 reset을 덮어쓰지 않도록 모든 actor/DOF 재제출. 첫 관측의 rigid-body buffer도 reset kinematic 상태로 맞춤.
- CPU 36개 테스트 통과. GPU7 사전 검증(2환경×16명×30step)에서 첫 스텝 최대 root 이동 0.043434m, root/body 차이 0.000001907m, Stage1 native actor/normalization exact 및 coordination_calls=0 확인. 사전 검증은 합성 스케줄이며 본평가에 포함하지 않음.
- 초기 장면 → 장면별 API 저장(16 concurrent) → 전부 검증 후 seed별 formal eval → 원본 CPU rescore/summary 경로 추가. API 오류/비실행 응답은 보존하고 미완성 스케줄로 평가하지 않음. 후행 timing target은 completion 대신 의존 상호작용 onset으로 변경.
- 진행 기록: output_etc/check/eval16_full_20261110_202645/status.json. 이 항목 작성 시 실행 시작 상태이며 최종 결과는 아직 없음. 원본 ksh 쓰기 및 MPS 종료 명령 없음.
- 후속 상태: seed1/6/7 각각 512개 초기 요청 생성 완료, 총 1,536개. API 병렬 저장 단계 진입 및 최초 검증 응답 성공 확인. 전체 응답 검증 후 GPU 평가·CPU 재채점 자동 실행 예정이며 최종 완료와 구분한다.

### 실제 실행 정책을 Stage1 SCT로 정정

- 원본 FSM이 쓰는 `eval/stage1_protocol_policy.py` 경로를 로컬 replay/view 진입점에 연결. Stage1JointPushDensity3Cone90Finetune_00026000.pth를 로컬 복사하고 SHA256 ec4215803b529bab1dd1be430154604418392e24faed59bca69b5c61adafe67c 및 epoch26000 CPU 확인.
- Stage1 가중치·정규화 복원, goal-aligned observation 변환, native actor 일치 검사 및 Stage2 coordination 호출 거부 hook 사용. 종료 시 policy proof 기록. Stage2 checkpoint는 환경/네트워크 구성용이며 행동 정책으로 사용하지 않음.
- capture/export 메타데이터를 Stage1 controller_id로 변경하고 이전 Stage2 스케줄 실행 거부. CPU 테스트 36개 통과 및 진입점 문법 검증. GPU/API 재실행 없음; 실제 GPU policy proof와 초기 상태 동기화 오류 수정은 아직 미완료.

### Native viewer 종료 및 초기 상태 동기화 오류 확인

- 사용자 요청으로 :99 Isaac Gym 창에 Q 입력, VIEWER_NORMAL_EXIT 및 viewer/noVNC 프로세스·6087/5999 포트 종료 확인. GPU process signal/MPS 제어 명령 없이 종료.
- 기존 native_view16_20261110_200753/rollout.npz에서 t=0 root와 pelvis rigid-body 위치의 최대 불일치 8.2049m 확인. 첫 저장 구간(약 0.067초)에 일부 즉시 출발 agent의 root가 3.53~8.27m 이동. 시작 순간 상태가 일관되지 않으며, 단순 화면 갱신 문제가 아닌 초기 물리 상태 적용 오류 조사 필요.
- 앞선 초기 tensor 일치 및 frozen projection 오차 0은 전체 초기화의 정당성을 보장하지 않음. 이 rollout을 스케줄 성능·물리 성공 검증으로 사용하지 않는다. GPU 재실행하지 않음.

### 실제 GPU Isaac Gym viewer 제공

- 도식 재생 대신 native Gym viewer를 GPU7 CUDA/Vulkan 양쪽에 연결하고 noVNC localhost:6087로 제공. nvidia-smi에서 viewer PID의 GPU7 M+C+G context 확인.
- 같은 초기 장면 검증 후 Space 시작 대기. 0은 전체, 1~7은 그룹 카메라, Q는 CUDA 동기화 후 정상 종료. 저장된 strict-after 스케줄을 그대로 사용하며 API 재호출 없음. 실행 완료 후에도 화면 유지.
- 첫 기동은 formal 설정의 headless 요구로 config 단계 실패; 새 viewer 진입점에서 장면 설정만 재사용하도록 제한적으로 처리. formal 평가 점수는 생성하지 않음.
- 기동 기록: teacher_runtime/output_etc/check/native_view16_20261110_200753. 기존 학습·MPS 종료 명령 없음.

### 실제 16명 스케줄 freeze rollout 및 재생 뷰

- API strictly-after 응답을 재사용해 GPU7 기존 MPS에서 600step(20초) 정상 실행(rc0), 추가 API 호출 없음. 동일 seed reset의 root/box/크기/task packet/그룹/목표를 저장 초기 상태와 대조해 모두 일치 확인.
- InitialReleaseFreeze를 실제 Stage2 정책에 적용. 16명 전원 release 기록, frozen root/DOF의 projection 후 최대 오차 0. A12 요청 0.51초는 control frame에서 0.533333초에 해제. 전원 해제 후 재동결 없음.
- 실제 root/rigid body/물체/frozen/종료 flag를 기록하고 독립 HTML 시간·그룹·시점 재생기 및 CPU skeleton/box MP4 생성. Gym 원본 mesh 렌더가 아닌 실측 궤적 도식임을 표시. MP4 미리보기 육안 확인 및 HTML canvas mock 그룹별 렌더 검증.
- A10/A14 종료 flag 및 낮은 최종 root 높이 관찰; 성공 rollout으로 주장하지 않음. 정책은 기존 Stage2 learned coordination 포함, 추가 초기 시간 gate만 적용.
- 결과: teacher_runtime/output_etc/check/replay16_20261110_200020/{view.html,replay.mp4,rollout.npz,release_audit.json,initial_match.json}.

### Strictly-after 검증 및 실제 장면 재호출

- enabling 검증을 출력 정밀도의 엄격 부등식으로 변경. 고정 물리 지연은 추가하지 않으며 선행 release=0도 검증. coupled 정렬은 유지. 합성 예제 갱신 및 CPU 테스트 34개 통과.
- 같은 16명 초기 관측으로 gpt-6-luna / medium API 1회 호출, ok 및 검증 통과. A02 예상 완료 5.60초, A12 release 0.51초 + 예상 소요 5.10초 = 5.61초.
- 입력의 A12 이벤트는 여전히 기존 formal completion proxy이므로 최초 앉기 접촉이 선행 완료 뒤라는 검증은 아님. 실제 rollout 없음. 원문: output_etc/check/api_scene16_strict_after_20261110_195414.

### Enable 프롬프트의 엄격한 선후 관계

- 사용자 요청대로 dependency-sensitive event를 선행 enabling event보다 strictly after로 변경하고 동시 발생 금지. 고정 추가 지연 초수는 명시하지 않음.
- 기존 동시 허용 공식 대신 release + q > p를 만족하는 최소 대기 조건으로 system/user template을 정리. 이번 수정은 프롬프트에 한정; 기존 코드 검증기의 비엄격 부등식은 별도 수정 전까지 유지됨. API 호출 없음.

### 가정 기반 프롬프트로 실제 16명 장면 재호출

- 동일한 초기 관측을 재사용하고 입력의 옛 insufficient_information 지시만 교체. 기존 감사 기록은 보존하고 새 경로에 API 1회 호출.
- gpt-6-luna / reasoning medium 응답 ok. 16명 예측·출발 시각·episode event 산술 및 enabling/coupled 최단 대기 제약 검증 통과.
- Enable A02=0초, A12=2초. coupled 그룹 예상 접근 시각은 각각 9.42/11.63/12.29/4.64초. 모든 예측은 low confidence이며 속도·준비 시간은 모델 가정. 실제 rollout 없음.
- 결과: output_etc/check/api_scene16_best_effort_20261110_194522/response.validated.json.

### 정보 부족 응답 선택지 제거

- system/user 프롬프트에서 calibration 부재에도 명시적 속도·준비 시간 가정으로 숫자 예측과 전체 스케줄을 생성하도록 변경. 가정은 실측 사실과 구분하고 low confidence로 표시.
- 응답 스키마의 insufficient_information 상태 제거. infeasible은 시간 제약 충돌에만 사용하도록 명시. exporter의 정보 부족 반환 지시도 제거하고 가독용 16명 예시 갱신.
- CPU 테스트 32개 통과, 제거한 상태의 응답 거부 포함. 기존 실제 요청·응답 감사 기록은 보존. 추가 API/GPU 호출 없음.

### 실제 16명 reset 관측으로 API 1회 호출

- `capture_initial_entry.py`로 GPU7 기존 MPS에서 지정 V3 checkpoint를 로딩하고 환경 1개/16명/seed21 초기 상태를 캡처. policy inference/env_step 없이 CUDA 동기화 후 정상 종료(rc0). 첫 시도는 필수 --group_spawn_over4 누락으로 config 단계 종료; 실패 기록 보존 후 옵션을 추가한 새 경로에서 완료.
- 실제 구성: Enable A02→A12, coupled A01/A08/A15/A16 및 A03/A05/A07/A10 및 A06/A11 및 A09/A13, 독립 A04 PUSH/A14 SIT. `export_initial_scene.py`로 초기 관측만 JSON 및 user template으로 내보내고 스키마 검증.
- Coupled 접근은 수치 임계값 없는 계획상 도달 시점으로 명시. Enable 후행 이벤트는 기존 formal completion proxy이며 실제 최초 접촉 detector가 아님을 입력에 공개.
- OpenAI 설정 모델, reasoning medium으로 API 정확히 1회 호출. 응답은 insufficient_information, predictions/schedule 빈 배열: controller 속도·실측 event duration이 없어 추정 근거 부족. 응답 스키마 검증 성공, CLI rc2는 정상 non-ok 상태 표시. LLM 스케줄 적용 rollout은 수행하지 않음.
- 원시 캡처: teacher_runtime/output_etc/check/initial16_20261110_194117. 입력/template: output_etc/check/real_scene16_20261110_194117. API 감사 기록: output_etc/check/api_scene16_20261110_194117.

### 16명 혼합 장면 입력 예시

- Enable 2쌍 + coupled 2/3/3/4명 그룹으로 총 16명 가상 초기 장면 작성. `examples/hypothetical_16_agents_request.json`은 실제 API 입력 형식, `prompts/user_example_16_agents.txt`는 user template을 채운 가독용 예시.
- 위치·목표·이벤트 조건은 설명용 가정이며 실측/성공 데이터가 아님을 명시. 입력 스키마, 16명 이벤트 소유권, 그룹별 구성, 실제 메시지 직렬화 검증 완료. API/GPU 호출 없음.

### Coupled PUSH 그룹 크기 정정

- 사용자 정정에 따라 system prompt와 user template의 4명 기본 전제를 제거. 그룹은 2~4명이며 실제 구성원은 입력에서 읽도록 명시.
- 장면 전체 인원과 개별 PUSH 그룹 크기를 구분. 그룹 구성원별 초기 release 결정 방식은 유지.
- API 메시지 조립에서 수정된 프롬프트 로딩 확인. API/GPU 호출 없음.

### Enable / coupled 제어 역할을 프롬프트에 명시

- Enable은 2명 단위로 선행 release=0 고정, 후행 freeze 해제 시각만 결정. 출력 스키마 호환을 위해 선행도 schedule에 0으로 포함.
- Coupled 기본 대상은 4명 PUSH 그룹이며 전원의 개별 출발 시각과 선출발자를 결정. 명시된 다른 그룹 크기도 동일 규칙 적용. 동시 출발 허용, 출발 후 별도 PUSH gate 없음.
- system prompt와 참고 user template 갱신. API 메시지 조립에서 변경된 system prompt가 로딩되는지 CPU 확인; API/GPU 호출 없음.

### 10-shot API 경로 제거

- 사용자 요청에 따라 API 입력을 environment_spec/current_scene으로 축소. 데모 10개 필수 검증, 프롬프트의 calibration 지시, supporting_example_ids 출력 필드와 --calibration 옵션 제거.
- 프롬프트는 초기 장면의 근거와 시간 가정을 명시하며 추정할 수 없으면 insufficient_information을 반환. 초기 release 제약과 물리 동결 동작은 유지.
- 합성 예제와 CLI 테스트 갱신. 실측 trajectory·과거 실행 결과는 보존하고 기록 검증 도구는 API 입력 검증에서 분리.
- CPU unittest 31개 통과. 유료 API 호출·GPU 작업 없음.

### OpenAI 직접 호출 및 reasoning medium

- 기본 base URL을 OpenAI `/v1`로 변경. 로컬 `.env`의 기존 OpenAI URL·사용자 모델·키를 보존하고 `SCT_LLM_REASONING_EFFORT=medium` 추가.
- `.env`/환경변수/CLI reasoning effort 설정을 Chat Completions 최상위 `reasoning_effort`로 전달. 빈 값·미지정은 CLI에서 medium, 명시 CLI가 우선하며 inference audit에도 기록.
- CPU 테스트 33개 통과: 요청 본문, 빈 값 fallback, CLI override, 오타 거부 포함. 실제 유료 OpenAI 호출은 수행하지 않음.

### Stage2 실측 성공 trajectory 10-shot 수집

- 사용자 지정 V3 2000 checkpoint와 sidecar·코드를 `teacher_runtime/`에 독립 복사. ksh 쓰기 거부 audit hook 및 로컬 CUDA/extension 캐시 사용.
- GPU7 기존 MPS에 연결하여 6개 setting을 순차 실행, 모두 900step 정상 종료(rc0). 외부 FSM 대기·Stage1 가중치 교체 없음.
- SIT 7/8, CLIMB 13/16, ON_TOP 12/16, 2인 PUSH 13/16, 3인 PUSH 14/16, 4인 PUSH 15/16 native group success. 전체 88장면 중 74성공이며, 생존·readiness 추가 조건을 통과한 후보는 69개.
- 최종 SIT2/CLIMB1/ON_TOP2 + PUSH2인2/3인1/4인2, 총 10개 성공 사례 선택. 초기 상태·root·손·물체 궤적·실측 이벤트를 JSON으로 추출하고 dense raw와 실패 기록도 보존.
- Enabling dependency-sensitive event는 명시적으로 formal completion. 별도 approach/readiness로 오기하지 않음. Coupled readiness는 실제 box 표면 양손 12cm 미만.
- `--calibration`으로 실제 demonstration 묶음을 API 메시지에 로딩. CPU 테스트 31개 통과, 10개 예제 원시 데이터 대조·request 스키마 검증 통과.
- 결과: `output_etc/check/calibration_20261110/`. 실측 raw: `teacher_runtime/output_etc/check/demo_{probe,diverse}_20261110/`. API 호출 0회, 기존 작업·MPS 종료 없음. GPU7 사용 메모리는 종료 후 실행 전 42,557 MiB로 복귀.

### Baze gateway API 설정 및 호출

- 기본 base URL `https://api.baze.co/v1/gateway`에 `/chat/completions`를 붙여 호출. 전체 endpoint override 유지.
- 프로젝트 `.env`에서 키·모델 로딩, 환경변수/CLI override, 빈 설정 템플릿 추가. 로컬 `.env`는 0600 및 Git ignore 확인.
- JSON mode는 명시 옵션으로 변경. Bearer 인증·HTTP 오류 메시지 비밀값 제외 및 재시도 없음.
- CPU 테스트 30개 통과: URL 경로, 인증 헤더, payload, 설정 우선순위, 실제 CLI 설정 연결, 오류 처리 포함.
- Baze 공개 endpoint 규격을 확인하지 못해 Chat Completions 호환을 가정. API 키·모델 미입력으로 실제 인증 호출은 미실행. GPU/MPS 사용 없음.

### Few-shot 10개로 확대

- 사용자 요청에 따라 system/user prompt, 입력 검증, 합성 예제·생성기 및 README를 10개 demonstration 기준으로 변경.
- 10개 입력 허용 및 5/9/11개 입력 거부 확인. CPU unittest 24개 통과.
- API 호출과 GPU 실행 없이 검증. 기존 5-shot 실행 기록은 보존.

### SCT + LLM Scheduler 독립 base 생성

- 제공된 프롬프트와 JSON 메시지 계약, 5개 데모 검증, 일회 호출 및 추론 audit 구현.
- enabling/coupled 혼합 제약을 정수 centisecond difference constraints로 검증. 잘못된 응답 자동 보정·재시도·fallback 없음.
- 원본 FSM의 projection 방식을 로컬 adapter로 이식. 초기 시간 기반 해제만 사용하고 release 후 추가 gate 없음.
- 실제 event recorder, censored label, bounded episode callback 연결, GPU6/7 UUID guard 추가.
- CPU unittest 24개 통과. 실제 Isaac Gym·LLM 호출·성공률은 미검증.
- coupled/enabling 오프라인 CLI 모두 exit 0. 각각 A/B release 2/0초, 0/2초 확인. 11개 Python 파일 구문 검사 통과. 로그는 `output_etc/check/offline_{coupled,enabling}/`.
- ksh 코드·기존 프로세스·MPS·큐·baseline 변경 없음. 합성 데이터는 실제 calibration용이 아님.
