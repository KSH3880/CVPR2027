# 16인 Stage1 + LLM 평가

## 활성 전체 평가: Mixed + Table2, 환경당 64개

`run_full_suite.py`는 Independent 전용 평가 없이 Mixed 2/4/8/16명과
Table2 enable_sit/enable_climb/enable_ontop/coupled_2/coupled_3/coupled_4를
seed1/6/7로 실행한다. 총30회, 64환경씩 총1,920장면, 900step이다.
Mixed 내부 독립 agent는 즉시 출발한다.

현재 경로: `output_etc/check/full_mixed_table2_env64_20261111_003631/status.json`.
사전 검증 → 30개 초기 장면 캡처(GPU6/7 각각1작업) → API16병렬 호출 →
저장된 원본 스케줄 평가 → CPU 재채점과 Table2 replay → summary/report 순서다.
API 자동 재시도 없음. 실행 가능한 출력은 검증 오류와 함께 평가하며 실행 불가능한
장면만 제외하고 분모·trace·Table2를 일관되게 필터한다. 원문과 실패 내역은 보존한다.
코드/프롬프트가 실행 도중 변경되면 새 단계 제출을 중단하므로 완료 전 수정하지 않는다.
`manifest.json`, `api_audit.json`, `api_failures.json`, `summary.json`, `report.md`와
GPU 쪽 각 run의 `suite_verification.json`에 설정/실행/검증을 저장한다.

아래는 이전 16인 평가 기록이다.

원본 `eval/run_stage2_scalability.py` 및 최신 Table1 seed 설정을 따른다.
Mixed 16명, 32 object tokens, seed 1/6/7, seed당 512환경, 900step,
XY100%, 모든 관측 object token attention 유지, distractor 물리만 arena 밖으로 제거한다.
Stage2 V2는 동일 장면/네트워크 구성용 carrier이고 행동은 Stage1 26k 원본으로 교체한다.

`run_eval16_batch.py --output output_etc/check/<새이름> --workers 16`은 다음 순서로 동작한다.

1. 2환경 사전 검증: 초기 root jump, root/body 일관성, Stage1 native action/정규화 검사.
2. seed별 512환경의 초기 snapshot과 장면별 request 생성. 사후 trajectory는 API 입력에 포함하지 않음.
3. 1,536장면 API 요청과 원문·검증 응답 저장. 본평가 중 API 호출 없음.
4. 모든 스케줄이 ok인 경우에만 seed별 GPU 평가. 오류는 `api_failures.json`에 보존하고 평가를 중단한다.
5. 원본 rescore로 CPU/GPU 결과 대조 후 summary/report 생성.

현재 실행: `output_etc/check/eval16_full_20261110_202645/status.json`.
API 기록은 같은 폴더의 `api/general_seed*/env_*/`에, GPU snapshot·평가 기록은
`teacher_runtime/output_etc/check/eval16_full_20261110_202645/`에 저장한다.
실행 중인 작업을 같은 명령으로 중복 실행하지 않는다.

초기 freeze만 사용하며 release 후 재정지하지 않는다. formal 성공 기준은 원본 그대로다.
LLM의 후행 event는 의존 상호작용 시작의 추정 시점이며 런타임 detector/gate가 아니다.
출력은 소수 둘째 자리까지, 실제 release는 해당 시간이 지난 첫 control tick에 적용된다.

## 원본 출력 그대로 실행하는 평가

`run_saved_eval16.py --source <기존 API 결과> --output <새 출력>`은 추가 API 호출이나
스케줄 수정 없이 실행한다. prediction/constraint 검증 오류는 audit에 남기되 실행한다.
agent별 출발 시각이 누락·중복되거나 유효한 시간이 아니면 해당 장면만 제외한다.
기존 strict 실행의 검증은 그대로 유지한다.

현재 `output_etc/check/eval16_raw_20261110_214845/status.json`에 진행 상태를 저장한다.
1,535장면 포함, seed7/env0102(A04 release 누락)만 제외. 원래 초기 장면을 재현하도록
512 슬롯으로 시뮬레이션하고 제외 슬롯은 계속 동결한 뒤 모든 평가 지표·trace에서 뺀다.
`evaluated_environment_indices.json`은 필터된 trace와 원래 환경 인덱스의 대응이다.

평가 완료: seed별 512/512/511환경 × 900step. 세 seed 모두 CPU 재채점과 GPU 집계가
일치하고, 포함된 24,560 agent의 실제 해제 시각이 원본 schedule과 일치했다
(control tick 올림 적용). `summary.json`, `verification.json`, `release_verification.json` 참조.

### 독립 snapshot freeze 대조 옵션

`SCT_FREEZE_IMPL=snapshot_resume`을 batch evaluator 실행 환경에 지정하면 초기 snapshot의 자세와 속도를 release 시 한 번 복원합니다. 기본값 `legacy`는 기존 실행을 보존합니다. 이 옵션은 FSM 성공/접촉 gate를 사용하지 않으며, 대기 중 상태 고정과 물리 충돌은 유지합니다. 16인 1장면 대조는 완료됐지만 전체 평가로 검증한 기본 방식은 아닙니다.

### 공용 프로토콜 v1 재평가

`run_protocol_v1_eval.py`는 저장된 no-unnecessary-wait API 응답과 초기 장면으로 legacy freeze 평가30회를 실행하고 `protocol_v1_rescore.py`로 최신 지표를 집계합니다. 결과의 `protocol_v1_report.md`, `protocol_v1_summary.json`, `protocol_v1_verification.json`을 사용합니다. 기존 `report.md`/`table2.json`의 구간 내6cm 비율은 과거 지표이며 최신 t_coop으로 인용하지 않습니다. 64환경 탐색이며 Independent전용 장면은 제외되어 Table1 Ind. 공식 열을 제공하지 않습니다. Enable 완료 시간은 행동동일가중→seed평균, Coupled t_coop/Jerk는 공용20cm counter/NaN/0제외 및 합산trial 인원동일가중 규칙을 적용합니다.
