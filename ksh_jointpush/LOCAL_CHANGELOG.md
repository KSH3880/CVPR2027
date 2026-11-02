# 로컬 실행 변경

## 2026-11-01

### 이동 페널티 학습 TensorBoard 연결

- 기존6017 TensorBoard의 logdir 목록에 `coupled_premature_motion_26000`을 추가하고 TensorBoard만 재시작했다. 기존 실험 목록과 새 run의286개 scalar·premature_push3종 및 실제 비영 벌점 데이터를 HTTP API로 확인했다. 학습PID1043657은 계속 실행한다. 주소는 `http://localhost:6017`, TensorBoard PID1044661이다. 실행/검증은 현재 학습 launch 폴더의 `tensorboard.sh`·`tensorboard.json`에 기록했다.

### 이동 페널티 설정 본학습 시작

- 사용자 요청으로 현재 구현 그대로 Stage1 PUSH26k에서 새 optimizer·seed42·GPU0/tokenhsi118·4명×1024환경 본학습을 시작했다. hard gate는 현재 소스의 aligned 제거 상태이며 전원 접촉 전 수평 속도0.05m/s 초과분에0.2가중치 벌점을 힘 비율로 배분한다. 출발 후 해제하는 latch는 없고 episode 전체에 적용한다. 기존15500iter와는 aligned 조건도 다르다.
- PID1043657, tmux `codex-ksh:premature_motion_20261101_163616`. epoch2에서 actor/RMS 고정·coordination/head 갱신·유한값·26k 전이를 확인했고 저장 설정 및 비영 벌점 scalar3종도 확인했다. CHECK/MAX_ITERATIONS/resume는 해제했다. 출력: `/home/injesus1010/repos/CVPR2027/ksh_jointpush/coupled_base/output/stage2_coupled_push_premature_motion_20261101_163616`. 실행/검증: `/home/injesus1010/repos/CVPR2027/runs/jointpush/premature_motion_train_20261101_163616`. 현재 상태는 `runs/jointpush/ksh_jointpush_active_training.json`이다.

### 전원 접촉 전 이동 벌점·개인 힘 비율 배분

- 사용자 승인한 보상 실험을 선택적 `stage2JointPush.premature_motion_penalty`로 구현했다. 공동 PUSH의 전원 양손 뒷면 근접 조건이 미완료이고 물리 상자 수평 속도가0.05m/s를 넘을 때 그룹당 `-0.2*(speed-0.05)`를 만들고, 뒷면 근처 양손의 상자 이동 방향 양의 힘 비율로 개인에게 배분한다. 단독/다른 과제/전원 준비/정지/힘 없는 관성은0이며 인원수로 벌점이 중복되지 않는다. 실제 simulate sample별 계산을 control step에서 평균한다. 손 net contact force 기반 귀속 추정이며 접촉쌍 직접 측정은 아니다.
- 기존 YAML은 보존하고 별도 `stage2_coupled_push_premature_motion.yaml` 및 train/test/vnc wrapper를 추가했다. 선택 옵션의 유한값·허용 키를 검증하고 checkpoint 계약에 포함해 기존 Stage2 checkpoint의 설정 변경 resume를 거부한다. 기존 actor 입력·hard gate·CONTACT 보상·공식 성공 채점은 변경하지 않았다. 최종 reward와 total 로그에 포함하고 `relation/premature_push/`에 벌점·미준비 이동 비율·양의 손 힘을 기록한다.
- 로컬 attention 호환 적용 CPU61테스트 통과: 비접촉자0·1:3 힘 분배·그룹 격리·단독/다른 과제/관성/제동 제외·한손 접촉·전원 준비 해제·순열·설정/계약·최종 reward 합산을 포함한다. 호환 미적용 정책 테스트6건은 PyTorch2.0 MHA mask 오류였으며 실제 실행과 같은 호환 적용 후 통과했다.
- 1024환경·4명·Stage1 PUSH26k에서 MAX_ITERATIONS=2 통합 검증을 수행해 exit0·유한 모델·actor/RMS 고정·coordination/head 갱신과 실제 비영 벌점 기록을 확인했다. 기존 runner 종료 조건으로 scalar update3건(step0/131072/262144)이 기록됐으며 검증은 종료됐다. 본학습 성능 개선은 아직 측정하지 않았다. 결과: `output_etc/check/premature_motion_check_20261101_161930/verification.json`. baseline15500iter와의 aligned 조건 차이는 USAGE에 명시했다.

### 사용자 요청으로 학습 종료

- 사용자 요청에 따라 현재26k기반 Stage2 학습PID617217에 SIGINT를 보내 종료하고 프로세스 소멸을 확인했다. 마지막 정기 checkpoint는15500iter(`Stage2CoupledPushRegionGoals_00015500.pth`)이며 저장 결과를 보존했다. 활성 학습 상태를 `stopped_by_user`로 갱신했다.

### 15000iter 혼합 과제 native viewer

- 사용자 요청으로 15000iter 체크포인트를 GPU0·tokenhsi118·4명1환경·random_scenario로 실행했다. N=다음 장면, F=카메라 추적, L=이미지 저장, Esc=종료다. 90frame의 유한 관측·행동·보상과 실제 사람·상자·바닥 렌더링을 확인했다. 첫 장면은4명 공동 PUSH이며 optimizer step0회, 본학습PID617217은 유지한다.
- 실행/미리보기/검증: `output_etc/check/view_15000_20261101_132947/`, viewer PID1009339, tmux `codex-ksh:coupled_view_15000_20261101_132947`. 재실행은 해당 폴더의 `run.sh`, 현재 상태는 `runs/jointpush/ksh_jointpush_active_view.json`이다.

### 최신15000iter 정식 평가·8000iter 비교 완료

- 사용자 요청으로 최신 저장본 `Stage2CoupledPushRegionGoals_00015000.pth`의epoch15000·269개tensor 유한값·SHA256 `75f5aa08...`를 확인하고 같은general 프로토콜(512환경×평가seed0/1/2·1200step)을 실행했다. 저장 학습환경 및 평가/집계 코드 SHA256은 검증된8000iter 평가와 동일하며 기존CPU37테스트 결과를 참조한다. 본학습PID617217은 유지한다.
- task별 점/영역 ever·종료/연속유지, PUSH 단독/공동 전원 성공률과 종료·목표 통과 부호거리,8000→15000 비교표를 생성했다. 학습은 aligned 포함, 평가 코드는 기존과 같은 aligned 제거 상태다.
- 실행/로그: `output_etc/check/formal_eval_15000_tasks_terminal_push_20261101_130229/`, tmux `codex-ksh:formal_eval_15000_20261101_130229`, launcher PID1002523. 진행·완료 상태는 `runs/jointpush/ksh_jointpush_active_evaluation.json`을 따른다.
- 평가·후처리 모두exit0으로 완료했다.1,536장면·6,144개task·5,418개PUSH구역·2,189개PUSH그룹이며 세seed1200frame의CPU/GPU/JSON 판정·분모/분자가 정확히 일치했다.8000iter와 모든 초기 상태tensor가 동일하고 source·checkpoint·실행스크립트·원시trace/지표 입력 SHA256도 유지됐다. 완료 상태JSON 갱신은 별도 기록했으며15000iter 분포PNG와 보고서의iteration 표기를 확인했다. 본학습PID617217은 계속 실행한다.
- PUSH 점 ever61.39±2.14%, 종료55.76±2.25%, 연속유지55.65±2.19%로8000iter 대비+1.64/+0.28/+0.32pt다. 그룹 전원 연속유지(3seed 합산)는 단독52.91%,2명59.33%,3명53.28%,4명44.21%다. 구역 종료절대거리 평균2.621→3.547m, 통과 시+인 부호거리 평균+1.979→+3.010m,20cm 초과 통과율25.18→30.68%로 증가했다. ON_TOP 점 종료성공률49.80→53.45%로 상승했고 SIT는62.85→59.76%로 하락했다. `COMPARE_8000_15000.md`에 task·그룹·거리 비교를 저장했다.

## 2026-10-31

### 8000iter 정식 평가·7500iter 비교 실행

- 사용자 요청으로 `Stage2CoupledPushRegionGoals_00008000.pth`의epoch8000·269개tensor 유한값·SHA256 `15c16729...`를 확인하고 같은general 프로토콜(512환경×평가seed0/1/2·1200step)을 시작했다. 저장 학습환경 및 평가/집계 코드 SHA256은 검증된7500iter 평가와 동일하며 기존CPU37테스트 결과를 참조한다. 본학습PID617217은 유지한다.
- task별 점/영역 ever·종료/연속유지, PUSH 단독/공동 전원 성공률과 종료·목표 통과 부호거리,7500→8000 비교표를 생성한다. 학습은 aligned 포함, 평가 코드는 기존과 같은 aligned 제거 상태다.
- 실행/로그: `output_etc/check/formal_eval_8000_tasks_terminal_push_20261031_195033/`, tmux `codex-ksh:formal_eval_8000_20261031_195033`, launcher PID860551. 진행·완료 상태는 `runs/jointpush/ksh_jointpush_active_evaluation.json`을 따른다.
- 평가·후처리 모두exit0으로 완료했다.1,536장면·6,144개task·5,418개PUSH구역·2,189개PUSH그룹이며 세seed1200frame의CPU/GPU/JSON 판정·분모/분자가 정확히 일치했다.7500iter와 모든 초기 상태tensor가 동일하고 source·checkpoint·원시trace/지표 입력 SHA256도 유지됐다. 완료 상태JSON 갱신은 별도 기록했으며8000iter 분포PNG를 확인했다. 본학습PID617217은 계속 실행한다.
- PUSH 점 ever59.74±0.63%, 종료55.48±0.85%, 연속유지55.33±0.85%로7500iter 대비+2.31/+1.59/+1.60pt다. 그룹 전원 연속유지(3seed 합산)는 단독47.40%,2명59.76%,3명52.09%,4명42.82%다. 구역 종료절대거리 평균3.784→2.621m, 통과 시+인 부호거리 평균+3.214→+1.979m,20cm 초과 통과율31.51→25.18%로 감소했다. CLIMB 점 종료성공률84.09→94.55%로 상승했고 SIT는77.98→62.85%로 하락했다. `COMPARE_7500_8000.md`에 task·그룹·거리 비교를 저장했다.

### 최신7500iter 정식 평가·7000iter 비교 실행

- 사용자 요청으로 최신 저장본 `Stage2CoupledPushRegionGoals_00007500.pth`의epoch7500·269개tensor 유한값·SHA256 `715e4da2...`를 확인하고 같은general 프로토콜(512환경×평가seed0/1/2·1200step)을 시작했다. 저장 학습환경 및 평가/집계 코드 SHA256은 검증된7000iter 평가와 동일하며 기존CPU37테스트 결과를 참조한다. 본학습PID617217은 유지한다.
- task별 점/영역 ever·종료/연속유지, PUSH 단독/공동 전원 성공률과 종료·목표 통과 부호거리,7000→7500 비교표를 생성한다. 학습은 aligned 포함, 평가 코드는 기존과 같은 aligned 제거 상태다.
- 실행/로그: `output_etc/check/formal_eval_7500_tasks_terminal_push_20261031_190525/`, tmux `codex-ksh:formal_eval_7500_20261031_190525`, launcher PID852541. 진행·완료 상태는 `runs/jointpush/ksh_jointpush_active_evaluation.json`을 따른다.
- 평가·후처리 모두exit0으로 완료했다.1,536장면·6,144개task·5,418개PUSH구역·2,189개PUSH그룹이며 세seed1200frame의CPU/GPU/JSON 판정·분모/분자가 정확히 일치했다.7000iter와 모든 초기 상태tensor가 동일하고 source·checkpoint·원시trace/지표 입력 SHA256도 유지됐다. 완료 상태JSON 갱신은 별도 기록했으며7500iter 분포PNG를 확인했다. 본학습PID617217은 계속 실행한다.
- PUSH 점 ever57.44±1.91%, 종료53.90±2.14%, 연속유지53.73±2.13%로7000iter 대비−3.89/−2.69/−2.79pt다. 그룹 전원 연속유지(3seed 합산)는 단독42.81%,2명58.14%,3명50.70%,4명43.75%다. 구역 종료절대거리 평균2.921→3.784m, 통과 시+인 부호거리 평균+2.368→+3.214m,20cm 초과 통과율25.71→31.51%로 증가했다. SIT 점 종료성공률은63.72→77.98%로 올랐다. `COMPARE_7000_7500.md`에 task·그룹·거리 비교를 저장했다.

### 7000iter 정식 평가·6000iter 비교 실행

- 사용자 요청으로 `Stage2CoupledPushRegionGoals_00007000.pth`의epoch7000·269개tensor 유한값·SHA256 `c9fe2c5b...`를 확인하고 같은general 프로토콜(512환경×평가seed0/1/2·1200step)을 실행했다. 저장 학습환경과 평가/집계 코드 SHA256은 검증된6000iter 평가와 동일하다. 기존CPU37테스트 검증은 해당 평가의 verification.json을 참조한다. 본학습PID617217은 유지한다.
- task별 점/영역 ever·종료/연속유지 성공률, PUSH 단독/공동 그룹 전원 성공률, 종료 거리·목표 통과 부호거리 및6000→7000 비교표를 자동 생성한다. 기존과 같이 학습은 aligned 포함, 평가 코드는 aligned 제거 상태다.
- 실행/로그: `output_etc/check/formal_eval_7000_tasks_terminal_push_20261031_172323/`, tmux `codex-ksh:formal_eval_7000_20261031_172323`, launcher PID834835. 실행·완료 상태는 `runs/jointpush/ksh_jointpush_active_evaluation.json`을 따른다.
- 평가·후처리 모두exit0으로 완료했다.1,536장면·6,144개task·5,418개PUSH구역·2,189개PUSH그룹이며 세seed1200frame의CPU/GPU/JSON 판정이 정확히 일치했다.6000iter와 모든 초기 상태tensor가 동일하며 코드·checkpoint·원시trace/지표 입력의 SHA256도 유지됐다. 완료 상태JSON의 정상 갱신은 별도 표시했고7000iter 분포PNG를 확인했다. 본학습PID617217은 계속 실행한다.
- PUSH 점 ever61.33±0.85%, 종료56.59±0.38%, 연속유지56.52±0.39%로6000iter 대비+2.60/+3.06/+3.03pt다. 그룹 전원 연속유지(3seed 합산)는 단독48.62%,2명62.03%,3명51.69%,4명42.36%다. 구역 종료절대거리 평균은3.19→2.92m, 통과 시+인 부호거리 평균은+2.657→+2.368m로 감소했다. `COMPARE_6000_7000.md`에 task별 점/영역·그룹별 유지·거리 비교를 저장했다. AT·ON_TOP의 점 종료성공률은55.56/50.78%로6000iter보다 낮았다.

### 최신6000iter 정식 평가·5000iter 비교 실행

- 사용자 요청으로 최신 저장본 `Stage2CoupledPushRegionGoals_00006000.pth`의epoch6000·269개tensor 유한값·SHA256 `4e12d61e...`를 확인하고 같은general 프로토콜(512환경×seed0/1/2·1200step) 평가를 시작했다. 본학습PID617217을 유지하며 GPU0에서 실행한다. 학습은 aligned 포함, 평가 환경은5000iter 평가와 동일한 aligned 제거 코드다.
- 기존 task별 점/영역 ever와 PUSH 종료거리·목표 통과 부호거리·종료/연속유지 성공률을 모두 생성하고, CPU 재채점·기존 거리/낙상과의 독립 대조·그룹 전원 판정·5000/6000 초기 snapshot 대조와 비교표를 자동 실행한다. 세 보고서의5000iter 고정 문구를 실제source manifest/checkpoint의iter 표기로 변경했다. 관련CPU37테스트 통과.
- 실행: `output_etc/check/formal_eval_6000_tasks_terminal_push_20261031_160042/`, tmux `codex-ksh:formal_eval_6000_20261031_160042`, launcher PID818047. 진행/완료 상태는 `runs/jointpush/ksh_jointpush_active_evaluation.json`, 원시로그와 검증은 해당 평가 폴더를 따른다.
- 평가·후처리 모두exit0으로 완료했다.1,536장면·6,144개task·5,418개PUSH구역·2,189개PUSH그룹이며 세seed1200frame의CPU/GPU/JSON 판정이 정확히 일치했다.5000iter와 초기packet/그룹/크기/goal/root/box/DOF/논리순서가 전부 동일하고 실행 중source·checkpoint SHA256도 유지됐다. 분포PNG의6000iter 표기와 내용을 확인했다. 본학습PID617217은 계속 실행한다.
- PUSH 점 ever58.73±1.58%, 종료성공53.53±1.13%, 연속유지53.49±1.07%다(평가seed 평균±표본표준편차).5000iter 대비 각각+4.14/+3.32/+3.30pt다.4명그룹 전원 연속유지는188/432=43.52%(기존36.57%)다. 구역 종료절대거리 평균은3.19m, 목표통과 시+인 부호거리 평균은+2.657m로 증가했다. `COMPARE_5000_6000.md`·`REPORT.md`·`TERMINAL_SUCCESS.md`·`PUSH_SIGNED_DISTANCE.md` 및 원시CSV/검증JSON에 기록했다.

### 5000iter 평가의 종료 성공·연속 유지 성공률 추가

- 사용자 요청으로 기존512환경×평가seed0/1/2의 완전한trace를 `ksh_terminal_success_eval.py`에서 CPU 재채점했다. 종료 순간 성공과 개인 첫 성공부터 마지막까지 매 프레임 성공을 구분하며, 낙상자는 유지 성공에서 제외하고 전체 분모에는 포함한다. PUSH는 개인 구역/그룹 전원을1·2·3·4명으로 분리한다. 기존 ever와 모든task 분모/분자는 정확히 일치했고 원시trace·기존JSON/CSV SHA256도 유지했다.
- 공식 판정 및 새 집계 관련CPU25테스트 통과. 이탈 후 재진입·마지막frame 첫 성공·점/영역 독립·낙상 후 상자만 목표에 남은 경우·전원낙상 종료시점·packet순열·그룹 각각ever와 동일종료 성공의 차이를 검증했다. 실제6,144개task·2,189개PUSH그룹을 집계했으며 시뮬레이션/학습은 재실행하지 않았다.
- 점 기준 종료 성공률(평가seed평균)은 AT53.34%, ON_TOP55.13%, SIT52.86%, CLIMB93.93%, PUSH50.20%다. PUSH 연속유지는50.18%; 4명그룹 전원 종료/연속유지는158/432=36.57%다. 기존평가 폴더의 `TERMINAL_SUCCESS.md`·`terminal_success_summary.json`·개인/그룹CSV·검증JSON에 기록하고 USAGE·structure를 갱신했다.

### PUSH 종료 목표 거리의 통과 방향 부호 추가

- 사용자 요청으로 기존5000iter 평가의 reset `initial.pt`와 종료trace를 재사용해 목표선 통과 시+, 미통과 시−인 기존3D 거리 평균을 계산했다. 고정축은 초기 상자/구역→목표의 XY 방향이며 현재 상자 방향·첫action frame을 쓰지 않는다. 횡방향 오차를 뺀 진행축 투영오차와 종료 시 목표선/20cm 초과 통과율도 별도 저장했다. 원시데이터·기존 절대거리·성공률은 SHA256으로 불변을 확인했다.
- 관련CPU12테스트 통과. 횡방향·회전/이동 불변·목표선/정확한goal·정의불능 초기축·초기reset 방향·horizon/전원낙상 종료frame을 검증했고 실제5,418개 구역·2,189개 그룹의 부호/크기 일치를 확인했다. 도구는 `ksh_push_signed_eval.py`, 결과는 기존평가 폴더의 `PUSH_SIGNED_DISTANCE.md`·`push_signed_terminal_summary.json`·두CSV·`push_signed_verification.json`이다.
- 구역 기준 ±목표거리 평균: 전체+2.212m, 단독−0.713m,2명+1.642m,3명+2.294m,4명+3.306m. 진행축 투영 평균은 각각+1.926/−0.698/+1.444/+1.933/+2.933m다. 시뮬레이션/학습은 재실행하지 않았다.

### 5000iter 평가의 학습 조건과 실행 코드 표기 정정

- 사용자 지적대로5000iter 체크포인트는 aligned가 있는hard gate로 학습한 모델이다. “aligned 제거 코드 기준”은 새 평가 프로세스가 수정된 환경 코드를 로드했다는 의미이며, 재학습한 가중치라는 뜻이 아니다. 학습 당시 gate와 평가 코드 조건을 보고서·상태/source manifest에 분리했다. 공식 점/영역 성공은 reward gate와 독립된 기하 판정임을 명시했다.
- 보고서 생성기의 조건 문구와 USAGE를 정정했으며 기존 결과의 REPORT.md만 다시 생성했다. 시뮬레이션·checkpoint·원시trace·성공률·거리 수치는 그대로 보존했다. 변경patch는 해당 평가 폴더의 `report_conditions_correction.patch`에 기록했다.

### 5000iter 정식 task별 평가와 PUSH 종료 거리

- 사용자 요청으로 고정5000iter를 공식general 프로토콜(학습 random_scenario·4명·512환경×seed0/1/2·1200step·deterministic·개인 낙상)을 통해 task별 채점한다. `evaluate.py`가 `TOKENHSI_RUN_ENTRY`를 받아 로컬 torch2.0/CUDA11 호환 진입점을 사용하도록 연결했고, root의 `scripts/coupled_region_goals_eval.sh`를 추가했다. 평가 전 사용자가 viewer를 직접 닫았으며 본학습PID617217은 유지한다.
- `ksh_push_terminal_eval.py`는 완전한 첫 episode trace에서 환경의1200step 또는 전원 낙상 최초step을 골라 PUSH 구역→개인목표와 물리상자→공통목표 종료 거리를 계산한다. 성공/실패/낙상을 모두 포함하며 개인 마지막 유효step 거리는 별도 저장한다. 물리상자 중심과 공통목표는 동일폭 구역/고정목표 평균으로 복원해 그룹당 한 번 집계한다. 최종20cm 비율과 ever 성공률을 구분하고 mean/median/p90·seed별 통계·CSV·분포PNG를 생성한다.
- 공식 점/영역 판정과 새 종료 거리 CPU22테스트 통과: horizon/전원낙상 시점·개인낙상 이후 다른task 계속·실패 포함·task packet 인덱스/순열·회전으로 구역만 빗나간 경우의 전체상자 중심을 검증했다. 실행결과는 `output_etc/check/formal_eval_5000_tasks_terminal_push_20261031_135950/`, 로컬 재실행법은 USAGE에 기록한다. 새 평가 프로세스는 aligned 제거 후 코드를 사용한다.

- 정식 평가 완료:1,536장면·6,144개 개인task이며 PUSH5,418개 구역/2,189개 물리상자를 집계했다. PUSH 점 ever54.60±1.29%, 종료 시20cm50.31%, 구역 종료거리 평균2.86m·중앙값0.20m·p9011.86m다. 세seed1200frame의CPU/GPU 성공기록·JSON분모/분자·유한값이 전부 정확히 일치했고 평가 중 source/checkpoint SHA256도 유지됐다. `REPORT.md`·`push_terminal_summary.json`·원시CSV·분포PNG·`verification.json`에 기록했다. 본학습PID617217은 계속 실행한다.


### 공동 PUSH hard gate의 aligned 조건 제거

- 사용자 요청으로 `coupled_push_spec.group_contact_gate`의 상자 local +Y·목표 방향 내적 조건을 제거했다. 공동 전원 양손의 뒤쪽 면18cm 거리·local Y≤0 조건은 유지하며 목표가 옆/뒤에 있어도 이를 만족하면 열린다. goal saturation의 현재 거리20cm·높이1mm 예외와 단독 gate는 유지한다. 목표를 지나쳐 saturation이 풀려도 접촉을 유지하면 gate가 열릴 수 있다.
- 관련 CPU8테스트 통과:1+3·2+2·4명 그룹과 회전한 상자에서 앞/옆/뒤 목표·한 손 이탈·그룹별 차단·단독 유지·얇은 상자의 앞면 거부·목표 범위/높이 이탈 시 saturation 해제를 검증했다. 로그와 변경 patch는 `output_etc/check/remove_gate_alignment_20261031_134400/`에 기록했다.
- 실행 중인 본학습PID617217은 기존 함수를 로드한 상태로 유지했다. 이번 변경은 새 프로세스부터 적용하며 재시작은 수행하지 않았다. 적용 대기 기록은 `runs/jointpush/ksh_jointpush_pending_gate_change.json`이다.

### Stage2 1+3·2+2·4명 공동의 bias / raw QK / softmax 비교

- 사용자 요청으로 같은5000iter의 세 배치를 비교한다. 기존1+3의162시점을 보존·재사용하고,2+2와4명공동은 각각3개 실제rollout·180시점을 추가 수집했다. 전 시점의 실제상자 배정은 `[16,22,22,22]` / `[20,20,21,21]` / `[23,23,23,23]`이며 각 수집에서 행동차이0·model/RMS불변·optimizer0회를 확인했다. 본학습PID617217은 유지한다.
- `render_coupled_stage2_comparison.py`와 HTML template은 같은layer/head/시점의 bias | raw QKᵀ | 최종softmax를 한 줄에 표시한다. 세 배치 선택·전체layer/head·장면/step·수집평균·QK/√d 전환·bias/QK 동일색상척도·실제배치를 제공한다. 열별색상범위는 세 배치에 공통이며 raw와scaled NPZ/PNG를 구분한다. 평균확률은 각 시점 softmax의 평균이다.
- raw QK는 스케일전 투영내적을 직접 수집하며 `softmax(raw QK/√32+bias)`와 독립PyTorch MHA를 대조한다. 관련CPU2테스트 통과, 세 배치 원본GPU출력 대비 재생오차최대4.77e-7·MHA3.58e-7·softmax4.77e-7·행합2.38e-7을 확인했다. JS194개 전환·전체72지도·실제Chrome 렌더링 및 PNG 시각검증을 통과했다.
- 결과·원시배열·검증·재실행스크립트: `output_etc/check/stage2_attention_comparison_5000_20261031_132618/`. 수집 entry의 `STAGE2_ATTN_PATTERNS`는1plus3·2plus2·4를 선택하며 여러 배치를 동일player에서 순차수집한다. 실행법은 coupled_base/markdowns/USAGE.md에 기록했다. 성공사례필터링이나 동일초기배치의 대응비교는 아니다.

### Stage2 actor attention1+3 관찰·시점별 지도

- 사용자요청으로현재5000iter의1+3고정rollout3개에서실제player의정규화된actor encoder출력·task/group packet·coordination출력을162시점수집했다. 원래human shuffle·정책·RNG를유지하는읽기전용hook이며계측전후행동출력차이0·model/RMS동일·optimizer0회를확인했다. 본학습PID617217은유지한다.
- `ksh_stage2_attention.py`는원래agent순서의입력으로Stage2복사본을재생해4layer×2head×4×4의QK/관계bias/softmax를추출한다. 독립PyTorch MHA대비·순열복원·원본가중치/RNG불변·identity인경우에도attention은비균등일수있음을확인하는CPU2테스트를통과했다. 실제GPU원본출력대비재생최대오차4.77e-7, MHA참조3.28e-7, 행합2.38e-7다.
- 평균/초기지도·관계bias·시간변화·초기배치PNG와시점/평균/QK/bias전환HTML을생성했다. JS구문및8지도×3표시모드전환과PNG시각확인을통과했다. SELF평균0.2592, 같은공동그룹peer0.2546, 다른그룹NONE0.2393다. 표본3장면의관찰이며attention은행동인과기여량과다르다. QK에는앞선layer의bias영향이남으므로별도ablation으로해석하지않는다.
- 수집/원시배열/상태/검증/그림: `/home/injesus1010/repos/CVPR2027/ksh_jointpush/coupled_base/output_etc/check/stage2_attention_1plus3_5000_20261031_131023`. 재사용진입점은 `scripts/coupled_stage2_attention_capture.py`와 `scripts/render_coupled_stage2_attention.py`이며실행법은USAGE에둔다.

### 5000iter 고정1명 단독+3명 공동 PUSH viewer

- 사용자요청으로현재26k기반학습의최신저장본5000iter를1+3으로고정해native viewer에띄웠다. viewer프로세스의sample_groups만 `[0,1,1,1]`로바꾸고 `push` preset으로전원PUSH를선택한다. 실제member `[1,3,3,3]`, 물리상자 `[16,22,22,22]`를확인했다. 원본checkpoint·학습설정·PID617217은유지한다.
- 기본카메라는초록agent0의단독PUSH를따라간다. 1=단독, 2=3명공동, 3=전체, N=다음1+3장면, F=카메라추적, L=이미지저장, Esc=종료다. 90frame정책추론finite·거리범위·실제단독agent/상자화면을확인했으며optimizer step0회다.
- PID768757, tmux `codex-ksh:coupled_1plus3_20261031_124256`. 실행명령은 `bash /home/injesus1010/repos/CVPR2027/ksh_jointpush/coupled_base/output_etc/check/view_1plus3_5000_20261031_124256/run.sh`, 검증·그룹정보·미리보기·저장프레임은같은폴더에둔다.

## 2026-10-30

### 21k 실행 중지·출력 삭제 후 Stage1 PUSH26k로 새 학습

- 사용자 요청으로 PID613529를종료하고 해당21k기반본학습출력 `stage2_coupled_people_first_source2_10_goal2_5_20261030_234641`의checkpoint·summary·로그50,556,219byte와TensorBoard링크를삭제했다. 기존다른실행·원본모델·코드는유지한다. 삭제기록은기존launch의 `user_stop_and_delete.json`이다.
- 사용자가제공한 `checkpoints/Stage1JointPushDensity3Cone90Finetune_00026000.pth`를확인했다. epoch26000·165tensor·유한값·SHA256 `ec421580...`·동일tensor전이·RMS호환을검증한뒤 새optimizer로Stage2를처음부터시작했다. 공동사람–상자2~10m·상자–goal2~5m·단독두거리2~10m·사람우선배치를유지한다.
- PID617217, tmux `codex-ksh:coupled_26k_20261030_235619`. 실제2epoch에서26k전이기록·고정actor/RMS의원본일치·coordination/head갱신·finite·AT/CONTACT를확인했다. 실행/검증: `/home/injesus1010/repos/CVPR2027/runs/jointpush/fresh_stage1_26k_people_first_20261030_235619`; 출력: `/home/injesus1010/repos/CVPR2027/ksh_jointpush/coupled_base/output/stage2_coupled_26k_people_first_source2_10_goal2_5_20261030_235619/Stage2CoupledPushRegionGoals_30-23-56-23`. TensorBoard6017을같은기존기록과새 `coupled_region_goals_26000`으로연결했다. YAML의기본21k는변경하지않고명시적STAGE1_CHECKPOINT로26k를선택한다.

### 사람 우선 배치 복원·공동 사람–상자2~10m·새 Stage2 시작

- 사용자 요청대로 공동 사람–전체 상자 중심XY만2~10m로 변경하고 상자–goal2~5m·단독PUSH두거리2~10m를 유지했다. 이어진 정정에 따라 native의 `placement_order`를 제거해 사람을 먼저 정한 뒤 위치를 고정하고 상자·goal을 찾는 기존 순서로 복원했다. 구 상자 우선 구현은 과거 저장 설정용으로 남겨 뒀다.
- CPU72개 통과: 혼합2048장면의두범위·실제5m초과source표본·단독bitwise동일·사람root/참조body불변·goal/정책/보상계약을확인했다. 중간 상자우선실행PID612688은후속정정으로종료했고 최종PID613529는Stage1 PUSH21000의165tensor에서새optimizer로처음부터시작했다. 실제2epoch에서1024환경·native설정·encoder/RMS동결·coordination/head갱신·finite·AT/CONTACT를확인했다.
- 실행/검증: `/home/injesus1010/repos/CVPR2027/runs/jointpush/fresh_people_first_source2_10_goal2_5_20261030_234641`; 출력: `/home/injesus1010/repos/CVPR2027/ksh_jointpush/coupled_base/output/stage2_coupled_people_first_source2_10_goal2_5_20261030_234641/Stage2CoupledPushRegionGoals_30-23-46-44`. tmux `codex-ksh:coupled_people_first_20261030_234641`, TensorBoard6017 `coupled_region_goals_21000/people_first_source2_10_goal2_5_fresh`. 이전학습·중간실행·zero-shot자료는보존한다.

### 현재 2~5m 배치의 Stage1 PUSH21000 zero-shot viewer

- 사용자 요청으로 Stage1 PUSH21000의165 tensor/RMS를 보존하고 Stage2 coordination을 identity로 붙인 관찰용 checkpoint를 생성했다. optimizer step은0이다. 구역goal·공동2~5m·상자/goal 우선 배치로4명 공동 PUSH를 로컬 native viewer에서 재생한다. N은다음장면, F는카메라추적, L은이미지저장, Esc는종료다. 본학습PID601645는계속실행한다.
- 초기 관측 변환 오류를 viewer 전용 코드에서 수정했다. 기본 Vulkan 선택에서는 실제화면/캡처가검게나왔고, NVIDIA를 명시한 작은 물리장면은정상렌더링됐다. `scripts/coupled_region_goals_test.sh`의local graphics경로에도공통 `gui_gpu_env.sh`를적용했다. bash구문·165tensor동일·coordination identity·90frame추론finite·실제사람/상자/바닥PNG를확인했다.
- tmux `codex-ksh:coupled_zeroshot_20261030_231952`, PID608719. 실행/전이/검증/미리보기: `/home/injesus1010/repos/CVPR2027/ksh_jointpush/coupled_base/output_etc/check/zeroshot_box_first_20261030_231952`. 첫장면의모든agent–상자중심4.265~4.522m, 상자–goal4.219m다. 초기오류/검은화면로그와그래픽비교결과도같은폴더에보존했다.

### 사용자 정정: 공동 PUSH 새 Stage2 학습 시작

- 이어서 학습하지 말라는 요청으로 PID600184를 종료하고 Stage1 PUSH21000에서 새 optimizer로 Stage2를 처음부터 시작했다. 공동2~5m·상자/goal 우선 배치·단독2~10m·AT/CONTACT 기록은 유지한다.
- PID601645, tmux `codex-ksh:coupled_fresh_20261030_231402`. 실제2epoch에서 설정·동결/갱신·finite를 검증했다. 실행/검증: `/home/injesus1010/repos/CVPR2027/runs/jointpush/fresh_box_first_20261030_231402`; 출력: `/home/injesus1010/repos/CVPR2027/ksh_jointpush/coupled_base/output/stage2_coupled_push_box_first_fresh_20261030_231402/Stage2CoupledPushRegionGoals_30-23-14-05`. TensorBoard6017의 `coupled_region_goals_21000/box_first_2_5m_fresh`에 연결했다.

### 승인된 상자·goal 우선 2~5m 학습 재시작

- 사용자 승인으로 이전 PID571143을 종료하고 최신500epoch의 모델·optimizer195state를 복원했다. 원본 checkpoint는 보존하고 복사본의 배치 계약·내장 환경 세 항목만 명시적으로 이관했다. 첫 시도는 내장 환경 불일치로 시작 전에 종료됐으며 내장 환경을 일치시킨 뒤 정상 재시작했다.
- 새 PID600184, tmux `codex-ksh:coupled_box_first_20261030_230938`. 실제502epoch에서 구역goal·공동2~5m/단독2~10m·상자 우선 배치·encoder/RMS 동결·head/coordination 갱신·finite·AT/CONTACT 기록을 확인했다. 실행/이관/검증은 `/home/injesus1010/repos/CVPR2027/runs/jointpush/resume_box_first_20261030_230938`, 본학습 출력은 `/home/injesus1010/repos/CVPR2027/ksh_jointpush/coupled_base/output/stage2_coupled_push_box_first_20261030_230938/Stage2CoupledPushRegionGoals_30-23-10-59`다. 기존 TensorBoard6017의 `coupled_region_goals_21000/box_first_2_5m`에 연결했다.

### 공동 PUSH 상자·goal 우선 배치

- 공동 그룹은 상자·goal을 먼저 정하고 구역 뒤에 사람을 배치하도록 수정했다. 공동 두 거리2~5m, 단독2~10m를 유지한다. CPU69개와 실제192장면/부분reset24회 검증을 통과했다. 현재 본학습에는 미적용이며 상세는 coupled_base/changelog.md를 따른다.

### 공동 PUSH 거리2~5m 정정

- 사용자 정정에 따라 공동 agent–상자/상자–goal만2~5m로 수정했다. 단독 PUSH2~10m와 나머지 과제는 유지한다. CPU64개를 통과했고 실행 중인 학습은 재시작하지 않았다. 현재 native 설정과 실행 중인 구 설정의 차이는 EXECUTION.md에 명시했다.

### 공동 PUSH 거리만1~10m로 변경

- 사용자 최종 요청대로 coupled_base 구역goal 모드의 공동 agent–상자/상자–goal 거리를1~10m로 분리했다. 단독 PUSH2~10m와 나머지 과제는 유지하며 CPU64개를 통과했다. 기존 학습PID571143은 재시작하지 않아 변경 전 배치로 계속 실행한다. 상세와 검증 경로는 coupled_base/changelog.md를 따른다.

### 현재 goal 분리 학습 AT/CONTACT 로깅과 resume

- coupled_base에서 누락된 CONTACT 진단을 직접 기록하고 기존과 같은 `reward_terms/at_*`, `reward_terms/contact_*`8항목을 추가했다. 혼합 과제의 다른 보상을 CONTACT로 오인하지 않으며 보상/학습 설정은 유지한다. CPU29개·실제4epoch 업데이트 검증을 통과했다.
- 사용자 승인으로 최신2epoch checkpoint에서 모델·optimizer를 복원했다. PID571143·tmux `coupled_split_20261030_210932`, 실행/검증 `../runs/jointpush/resume_coupled_reward_split_20261030_210932/`. TensorBoard6017의 현재 `coupled_region_goals_21000`에 새 로그를 연결하고 이전 세 run도 보존한다. 세부 평균 정의와 변경은 coupled_base/changelog.md에 기록했다.

### 구역별 goal 모드21000 전이 본학습 시작

- 사용자 수신 완료 후 새 Stage1 PUSH21000(165tensor·SHA `5ac3ee58...`)을 검증하고 coupled_base의 goal분리모드를 GPU0·tokenhsi118·4명1024환경·LOCO 시작으로 실행했다. PID564900·tmux `codex-ksh:coupled_region_goals_20261030_200442`, 출력은 `coupled_base/output/stage2_coupled_push_region_goals_20261030_200442/Stage2CoupledPushRegionGoals_30-20-43-15/`다.
- CPU26개·실제2epoch actor/RMS동결·coordination/action head 갱신·새goal checkpoint계약·263scalar finite를 확인했다. 실행/설정/검증은 `../runs/jointpush/launch_coupled_region_goals_20261030_200442/`다. 앞서 중단한7000 KSH실험은 보존하고 현재학습metadata를 새실행으로 갱신했다.

### pull의 구역별 goal 모드 학습 준비 · 21000 PTH 대기

- 사용자 요청한 새 모드는 `coupled_base/stage2_coupled_push_region_goals`다. 기존 KSH19500 hard/RSI 모델과 구분하고 Stage1 PUSH21000 전이·4명1024환경 설정과 실행 파일을 준비했다. 로컬 데이터235행/21PUSH 해시·CPU6개·GPU4환경 초기화/고정goal 검증을 통과했다. 원본19500/7000 저장본은 사용하지 않았다.
- 21000 PTH가 로컬에 없고 안내된Pro1 자동인증이 실패해 아직 학습을 시작하지 않았다. 수신 위치는 `../runs/assets/coupled_base/Stage1PushFinetune_00021000.pth`, 실행/검증 경로는 `../runs/jointpush/launch_coupled_region_goals_20261030_200442/`다. 세부 호환 처리는 `coupled_base/changelog.md`에 기록했다.

### 공중 받침대의 goal·관측·보상 영향 확인

- 사용자 지적에 따라 현재7000 hard/PUSH-only 설정에서 받침대와 실제goal 경로를 점검했다. 관측은 사람4·논리상자8·goal4이며 goal pose/AT 보상은 `_tar_pos`, 받침대는 별도actor28~35다. 현재받침대는 fixed이며 높이20/20.5m, 실제PUSH goal높이는0.725~0.85m다.
- 별도 headless 프로브의 random/all_solo/all_coupled 각16환경에서 판 위치를 `[71,-63,47]`m 이동해도 전체관측·행동·관계보상·goal성공·gate 차이0을 확인했다. 실제goal을 상자위치로 바꾸는 양성대조에서는 모두 변화했다. 추론에서도 사람순서를 섞으므로 A/B RNG를 일치시켰다. 각16step의 판 접촉력은0N이다. 물리actor 자체는 남아 있으므로 일반적인 물리영향 불가를 주장하지 않고 현재배치의 비접촉 및 입력분리를 확인한다. viewer/원본코드/학습상태는 유지했다.
- 원시는 `output_etc/check/platform_goal_audit_20261030_165751/{probe.py,verification.json,conclusion.json}`에 있다.

### 사용자 요청으로 hard CONTACT 학습 종료

- 16:51KST 사용자 요청으로 출력 경로와 명령을 확인한 학습PID140502에 SIGINT를 보내 프로세스 종료를 검증했다. 마지막 정기 저장본은7000epoch이며 기존 checkpoint와 로그를 보존한다. 현재 학습 metadata를 `stopped`로 갱신했으며 종료 기록은 현재 launch의 `stop.json`이다. 뷰어·TensorBoard에는 종료 신호를 보내지 않았다.

### 현재 hard 학습 최신7000 checkpoint native viewer

- 사용자 요청으로 최신 정기 저장본7000을 사본으로 고정하고 학습과 같은 hard gate·공동 jhh/단독 KSH CONTACT 계약으로 native viewer를 실행했다. GPU0·tokenhsi118·4명1환경·random_scenario·LOCO100% 평가이며 정책 추론과 실제 물리 시뮬레이션을 사용한다. 학습PID140502는 계속 실행한다.
- tmux `codex-ksh:ksh_hard_view_20261030_164652`, viewer PID526965, 데스크톱 `:0`의 Isaac Gym 창을 확인했다. 체크포인트 계약·두 reward adapter·모델 로딩·키 구독·정책 추론 로그와 IsViewable을 검증했다. N=다음episode, Esc=뷰어 종료다.
- 설정/체크포인트 사본·실행 명령·검증은 `output_etc/check/native_hard_view_7000_20261030_164652/`, 현재 viewer metadata는 `../runs/jointpush/ksh_jointpush_active_view.json`이다.

## 2026-10-29

### 이전 soft · 현재 hard 학습 TensorBoard 비교

- 사용자 요청으로 TensorBoard6017을 두 run 비교로 재연결했다. 이전 `ksh_soft_balanced_19500`의 분리 로그를 보존하고 현재 `ksh_hard_balanced_jhh_contact_19500`의 원본 scalar와 AT/CONTACT8항목을 새 `derived_summaries/`로 미러링한다. CPU writer는30초마다 갱신하며 현재 학습PID140502를 감시한다. 학습은 계속 실행한다.
- HTTP에서 두 run 각각205개 scalar와 실제 AT/CONTACT 값을 확인했고 writer의 후속 갱신도 검증했다. TensorBoard PID141790·writer PID141691이며 학습PID140502가 계속 실행 중이다. 설정·실행 스크립트·원시 검증은 현재 launch의 `tensorboard_compare*`·`reward_split*`이다. 현재 서버의 실행 안내를 갱신했다.

### 새 공동 CONTACT 기준 · hard gate 본학습

- 사용자 요청으로 pull의 `coupled_jhh_solo_ksh_contact_v1`을 적용해 GPU0·tokenhsi118·4명×1024환경 학습을 시작했다. 공동 CONTACT는 손 중점 높이1.04m·우회 접근 progress·양손 거리/힘 조건, 단독 CONTACT는 기존 KSH다. gate는 사용자 최종 지정인 hard이며 balanced RSI·초기 안전 검사·잉여 상자 이격은 유지한다. 이전5500 soft 모델을 resume하지 않고 Stage1 19500에서 새 optimizer로 전이한다.
- 이전 로컬 설정 사본에서 reward_gate만 hard로 바꾼 native 참조를 만들고, upstream 생성기로 CONTACT 항목만 추가했다. 원본 source/data는 보존하고 upstream 문서3개 변경을 구분했다. 실제 생성 설정을 fixture로 지정한 CPU57개 통과, GPU2epoch에서 actor/RMS 동결·coordination 갱신·gradient 및 새 checkpoint 계약과 전체197scalar 유한값을 확인했다.
- tmux `codex-ksh:ksh_hard_contact_20261029_222331`, PID140502. 출력은 `output/train_base19500_balanced_hard_jhh_contact_gpu0_20261029_222331/Stage2JointPushRsi_29-22-24-29/`, 설정·명령·검증은 `../runs/jointpush/launch_ksh_hard_contact_20261029_222331/`다. 기존 TensorBoard6017은 이전 soft 로그를 유지한다.

### 손 접촉 viewer 후방 추적 카메라

- 사용자 요청으로 hand-quality 사례의 기본 카메라를 미는 방향 뒤쪽으로 옮겼다. 네 명의 평균 위치와 상자 중심을 함께 보며 그룹/상자 폭에 맞게 거리를 잡고 초기 미는 축을 유지해 목표를 지나도 시점이 뒤집히지 않는다. `KSH_VIEW_CAMERA=legacy`로 기존 시점을 선택할 수 있고 F로 추적을 해제할 수 있다.
- 카메라 계산은 `ksh_view_camera.py`, 실행은 strict사례 폴더의 `view_behind.sh`에 둔다. CPU2개에서 후방 배치·그룹 이동과 회전·폭에 따른 framing을 확인했다. 실제 뷰어PID106875의 `behind_camera_check.png`에서 네 명과 양손이 보이는 후방 시점을 확인했고 이전 뷰어는 새 화면 준비 후 제어 파일로 정상 교체했다. 원래 보상·정책·저장된 시작 상태는 유지한다.

### 4명 양손 접촉을 유지하는 성공 사례 추가 탐색

- 사용자 요청으로 `ksh_hand_success.py`의 평가 전용 `hand_collect`를 추가했다. 5500PTH·soft·LOCO·deterministic 설정에서4명 공동 그룹만 탐색한다. 성공 판정/학습 보상을 바꾸지 않고 목표 도달과 별도로8개 손 중심의 finite rear face3D거리·실제PUSH hit를 기록한다. 목표 포화가 접촉으로 오인되지 않는다.
- 선별 기준은8손 모두8cm이내를2초 window의90%이상 유지, 같은 window에서4명 모두의실제PUSH hit50%이상, 목표방향진전25cm이상, 연속이탈0.2초이하, 최초공식도달순간에도8손 모두8cm이내다. 목표별 접촉 유지 시간·미달 후보는 진단용으로 구분하며 strict사례와 섞지 않는다. root/DOF/body/손거리/접촉의 실제 전체 궤적과 초기 상태를 보존한다.
- 사용자가1건이면충분하다고지정하여 strict1건 확보 또는64회 예산 종료로 설정했다. GPU0·256평가환경 실행은 `output_etc/check/hand_success_5500_20261029_205318/`, tmux `ksh_hand_success_5500`, 상태는 `../runs/jointpush/ksh_hand_success_active.json`이다. 학습은 실행하지 않았다.
- 사용자 요청으로 해당 strict1건을 기존 native policy viewer에 연결했다. 실행은 같은 폴더의 `view.sh`, tmux `ksh_hand_live_5500`이며 실제 준비 완료와 창 생성을 확인했다.
- CPU10개에서8손·실제힘 필수, goal포화·정지상태/도달순간이탈·초기성공/연속이탈 배제, 회전·높이·유한면3D거리를 확인했다. 원본554파일 해시·diff 검사 통과. 첫256episode에서 strict1건(case001/env32)을 확보해 rc0으로 종료했다.8손8cm이내 연속29.5초, 도달시최대거리4.36cm,2초선별창의8손접근율100%/4명PUSH hit98.3%/진전68.1cm였다. 도달프레임541의공식20cm·높이1mm와기록궤적을 독립검증했다(`trajectory_verification.json`).

### 그룹 지정 viewer 재실행

- 사용자가 4명 공동밀기를 다시 요청하여 `ksh_success_live.py --category {1+3,2+2,4}`로 시작 유형을 지정할 수 있게 했다. 종료된 viewer를 GPU0/tokenhsi118/5500PTH의4명 공동 사례001로 다시 열었다. 수집은 유형별2건씩 총6건을 확보하여 rc0으로 완료했다(`cases/cases.json`, `exit_code`).

### native viewer 키 전환 중 기존 화면 유지

- 기존 N/1/2/3 전환은 프로세스를 먼저 종료해 다음 모델·에셋 로딩 동안 창이 사라졌다. `scripts/ksh_success_live.py`와 `ksh_live_control.py`로 현재 화면을 유지한 채 다음 장면을 준비하고, 새 창의 첫 물리 frame이 준비된 뒤 교체한다. 없는 유형·현재 유형 선택과 새 장면 초기화 실패는 현재 화면을 유지한다.
- 관리되는 viewer에서 전환 키는 종료 코드 대신 원자적 JSON 요청을 보내고 GPU 동기화 후 현재 화면을 유지하며 기다린다. N은 파일명을 기준으로 다음 사례를 선택하므로 수집 목록 증가로 순서가 밀리는 문제도 방지한다. Esc/닫기는 viewer를 종료한다. 수집 평가·학습·TensorBoard와 별도다.
- CPU9개(목록 증가·없는 유형·원자적 요청 포함)를 통과했다. 실제 N 요청44에서 기존PID77187/창이 IsViewable인 채 새PID77601을 준비하고 첫 frame 준비 후 2+2 화면으로 전환한 것을 확인했다. 실제2키 요청42는 같은PID77601을 유지하고 재개했다. 원시 검증은 `output_etc/check/group_success_5500_20261029_195057/viewer_key_handoff_verification.json`, `viewer_switch_20261029_201500/`와 `viewer_managed_v2.log`다.

### 그룹별 전원 목표 도달 검색과 원래 물리 viewer

- 사용자 요청으로 `ksh_group_success.py`를 추가했다. 평가에서만 1+3·2+2·4명 그룹을 균등 탐색하며 초기 성공을 제외하고 모든 agent가 자기 그룹의 공식 목표에 도달한 완결 episode만 선택한다. 두 그룹의 도달 시간이 달라도 허용하며 동시 성공·종료 성공 여부를 별도 기록한다.
- 시작 시점의 scene/root/DOF/graph/runtime와 물리 상자 크기·밀도를 저장하고, live mode는 동일 에셋을 만든 원래 태스크에서 초기 상태를 복구한 뒤 매 step 정책 추론·물리 시뮬레이션을 실행한다. 기록 자세 재생과 별도다. 원래 viewer와 N 키를 유지하고 1/2/3 그룹 선택을 추가했다. 그룹 전환은 에셋을 맞춘 새 viewer 프로세스로 이어진다.
- 복수 환경에서 선택된 장면의 큰 world offset 때문에 native 바닥 렌더 범위 밖으로 나가는 문제를 발견했다. world 위치·목표·kinematic reference를 env0 원점으로 함께 평행 이동하고 속도와 로컬 형상을 보존한다. 원점 이동 CPU 회귀를 추가했다.
- 학습은 중단 상태로 유지한다. 실행·검증 원시는 `output_etc/check/group_success_5500_20261029_195057/`에 기록한다. 20:05 기준 2+2 두 건·4명 공동 두 건을 수집했으며 1+3은 탐색 중이다. CPU6개·원본554파일 해시·diff 검사를 통과했고 바닥 복구는 `floor_fixed_composite.png`로 직접 확인했다. 원점 이동 전 native4명 사례는526frame에 전원 재도달했으나 일부1환경 재실행은 목표를 놓쳤다. 선별된 성공 원본과 live 재실행의 성공을 구별한다.

### 목표 도달 에피소드만 수집·기록 자세 재생

- 사용자 요청으로 `ksh_success_cases.py`의 headless 수집과 `scripts/ksh_success_replay.py`의 독립 native 재생을 추가했다. 공식goal success가초기화후한번이라도참인완결episode만저장하며root/DOF/body/상자크기·목표·그룹·성공frame을보존한다.4명전원성공과한명/일부그룹성공을구분하고종료시유지사례를먼저표시한다.재생은기록자세로구동하며모델추론/새정책롤아웃은없다.
- 실제5500모델·LOCO100%·평균행동·64환경×2회에서사례6개를확보했다.6개모두원래20cm/높이1mm성공조건·초기미성공을검증했고1개는전원종료성공을유지했다.수집을6개에서멈췄으므로성공률평가로사용하지않는다.초기부터episode종료까지기록해도착후이탈도재생한다.
- CPU3개·native8frame재생·첫도착body위치최대오차2.91e-5m와PNG를검증했다.결과는 `output_etc/check/success_cases_5500_20261029_181452/`,현재창은 `KSH Success Replay - 5500`·PID56377·tmux `ksh_success_replay_5500`이다.N/P=다음/이전,R=처음,G=첫도달,Space=일시정지,Esc=종료.현재viewer메타데이터를갱신했으며학습은종료상태를유지했다.

### 사용자 요청으로 학습 완전 종료

- 17:54KST 사용자 요청으로 학습PID3984199의명령/출력/MPS미연결을검증하고SIGINT로종료했다. 프로세스소멸과종료기록을확인했으며기존로그/weight를보존했다. 마지막정기저장본은5500epoch다. 현재metadata는 `stopped`, 종료검증은현재launch의 `stop.json`이다. viewerPID47988·TensorBoard는유지했으며 보상분리writer는학습PID소멸을감지해마무리한다.

### 최신5500 checkpoint native viewer

- 사용자 요청으로 현재학습의 최신숫자5500저장본을 독립사본으로 고정하고 GPU0·desktop `:0`에 native `Isaac Gym` viewer를 실행했다. 학습과동일soft/balanced/safe-start/clear-paths계약·4명1환경·random_scenario·기본LOCO100%평가다. tmux `ksh_jointpush_view_20261029_173516`,PID47988이며실제창/정책추론/N키구독을확인했다. 학습PID3984199는 유지했다.
- 체크포인트/설정사본·명령·로그·검증은 `output_etc/check/native_view_5500_20261029_173516/`, 현재viewer메타데이터는 `../runs/jointpush/ksh_jointpush_active_view.json`이다. N=다음episode,Esc=viewer종료다.

### 사용자 요청으로 일시정지 학습 재개

- 16:04KST 사용자 요청으로 동일명령/출력의 정지PID3984199를 확인하고 SIGCONT로 재개했다. `R (running)`과새학습로그를 확인했으며 optimizer·epoch·출력·설정을 유지한다. 현재metadata를 `running_verified`로 갱신하고현재launch의 `resume.json`에 검증을 기록했다. TensorBoard와보상분리writer도 유지한다.

### 사용자 요청으로 현재 학습 일시정지

- 12:59KST 사용자 요청으로 현재출력/명령/MPS미연결을 확인한 학습PID3984199에 SIGSTOP을 보내 `T (stopped)` 상태를 검증했다. 프로세스·optimizer·메모리 상태와GPU메모리는 유지하며 자동재개하지 않는다. TensorBoard와보상분리writer는 유지했다. 현재metadata는 `paused`, 재개신호는SIGCONT이며현재launch의 `pause.json`에 기록했다.

### 현재 학습 CONTACT/AT 보상 분리 표시

- 사용자 요청으로 기존 `scripts/ksh_reward_split.py`를 CPU writer(PID4119578,tmux `ksh_jointpush_reward_split`)로 연결했다. PUSH-only의 전체 과제 항에서 AT를 빼 CONTACT를 산출하며 가중치/gate/포화 반영 state/progress/success/total8개를 추가한다. 기존4442step을 backfill하고 후속7step 갱신도 확인했다. 모든 변환에서 전체합·항별합 일치와 CONTACT음수없음을 검사한다.
- 원본event는 유지하며 별도 `derived_summaries/` 단일writer미러에 원본197태그와분리8태그를 기록한다. TensorBoard6017만 새미러로 재연결해 HTTP응답·205태그·8개실제값을 확인했다. 학습PID3984199는 그대로 실행 중이며 상태/명령/검증은현재launch의 `reward_split*`·`tensorboard_split*`에 있다.

### 현재 학습 TensorBoard6017 연결

- 사용자 요청으로 `scripts/tensorboard_compat.py`와 `tokenhsi118`을 사용해 localhost6017에 현재학습 `ksh_soft_balanced_19500`만 연결했다.15초 갱신·CPU실행·tmux `ksh_jointpush_tensorboard_6017`이며 명령/로그는 현재launch폴더에 보존한다. 기존6006/6016포트와 학습은 유지했다. HTTP200·scalar197개 로딩을 확인했다.

### RTX5090 GPU0 soft/balanced 본학습 시작

- 사용자 요청으로 `tokenhsi118`·Stage1 19500 전이·새 optimizer/epoch0·4명×1024환경·soft gate·balanced각1/3·초기0.7m 안전 검사·잉여 상자8~12m/경로1.5m 이격 본학습을 시작했다. tmux `ksh_jointpush_soft_20261029_001531`, PID3984199, 출력 `output/train_base19500_balanced_soft_safe_clear_gpu0_20261029_001531/Stage2JointPushRsi_29-00-15-41/`다.
- RTX5090/CUDA11.8의 NVRTC 미지원으로 초기화 중 실패한 시도 로그를 보존했다. 로컬 호환 모듈에서 상자 부피/크기격자3축곱을 동등한 명시적 곱셈으로 연결하고 JIT fusion을 비활성화했다. PyTorch2.0의 float attention bias native 경로 오류는 identity view로 일반 attention 경로를 선택해 우회한다. source·모델구조·보상·학습조건은 유지하며 soft adapter보다 task가 먼저 import되지 않도록 연결 순서를 맞췄다.
- 실제GPU1024환경 부피/격자·TorchScript모션보간·행렬역전파, CPU호환2개(크기/질량/확률·attention출력/gradient동등성)를 검증했다. 본학습2epoch에서 coordination gradient/갱신·actor encoder/scene RMS동결·startupweight 저장을 확인했다. 초기shared RSI그룹은전원push382/혼합422/전원loco413이었다. GPU메모리약22GB이며 실행/설정사본/검증은 `../runs/jointpush/launch_ksh_jointpush_soft_20261029_001531/`, 현재메타데이터는 `../runs/jointpush/ksh_jointpush_active_training.json`이다. 다른 프로세스는 종료하지 않았다.

### 수신 데이터 검증·로컬 연결과 PyTorch2.0 호환

- 수신한2.1GB 데이터에서 누락 메타데이터111개를 복원하고13개 payload 경로를 로컬로 연결했다. 원본554파일 해시, 모션130개/모션·물체161개 경로, PUSH21개 모션과 캐릭터/mesh2개 해시를 검증했다. Stage1 19500·soft gate·balanced RSI·초기0.7m 안전 검사·잉여 상자8~12m/경로 여유1.5m 실행 설정을 생성했다.
- 실제 모션 목록에 남아 있던 이전 서버 상대경로(`../../../../CVPR2027/TokenHSI/tokenhsi/data/...`)를 로컬로 변환하도록 준비 스크립트를 보완했다. `tokenhsi118`의 PyTorch2.0에서 미지원인 `any((1,2))`는 새 로컬 `ksh_torch_compat.py`로 해당5개 함수만 동등한 단일축 순차 연산에 연결한다. frozen source와 보상/안전 기준은 유지한다.
- CPU33개 검사·IsaacGym gymtorch 확장 import·원본 해시·diff 검사 통과. 확장 빌드 캐시는 `runs/jointpush/torch_extensions/`를 사용했다. 실제 simulator reset/GPU 학습은 실행하지 않았다. 준비/검증/CPU검사 결과는 `output_etc/check/data_ready_20261029/`에 보존했다.

## 2026-10-28

### 현재 서버 tokenhsi118·soft/balanced 실행 경로 이식

- 사용자 요청으로 launcher 기본 환경을 `tokenhsi118`, RSI를 balanced, gate를 soft로 정하고 로컬 Stage1 19500 checkpoint를 연결했다. GPU 번호를 선택 가능하게 하고 검증된 기존 MPS 자동 연결 또는 일반 CUDA를 사용한다. 초기 배치 검사·주변 박스 이격·1024환경은 유지한다. view wrapper도 같은 conda 기본값을 사용한다.
- 원본 데이터는 `runs/assets/ksh_jointpush/data/`로 받는다. 준비 스크립트는 원본 manifest를 보존하면서 누락 메타데이터의 해시를 검사하고 로컬 payload를 연결하며 모션 경로를 변환한다. 사용하지 않는10500/14000 복원 의존성을 제거했다. `EXECUTION.md` 상단에 현재 서버 scp·준비·학습 명령을 기록했다.
- `tokenhsi118`에서 CPU13개 검사(데이터 이식·누락/해시 오류·모션 경로·soft/balanced 설정 생성·기존 보상/RSI), shell/Python 구문 검사를 통과했다. 실제 입력 검사는111개 메타데이터+13개 payload 경로 누락을 보고하며 데이터 전송 전이므로 simulator/GPU 학습은 실행하지 않았다. 확인 로그는 `output_etc/check/local_port_20261028/`에 둔다.

### 최종 safe-start·장애물 이격 설정 GPU5 본학습

사용자 승인으로해당native초기뷰PID3832983만종료하고19500베이스fresh Stage2를GPU5 MPS에시작했다. hard18cm·balanced각1/3·PUSH-only·단독±45°/공동0°·인당20kg·1024환경을유지하고모든agent초기0.7m/몸체겹침검사와주변박스8~12m/접근선·sweep여유1.5m를적용했다. tmux`jointpush_hardgate_safe19500`,PID3851733,output`output/train_base19500_hard_safe_clear_gpu5_20261028_214039/Stage2JointPushRsi_28-21-40-42/`.

원본554파일해시·19500실제전이·두adapter설치·실제2epoch encoder/RMS동결·coordination gradient/업데이트·startupweight를확인했다. CPU보상분리미러PID3854708과TensorBoard6016새run`current_ksh_jointpush_base19500_hard_safe`를연결하고기존14k비교2개를유지했다. MPS서버와기존다른클라이언트는중단하지않았다.

### 주변 랜덤 박스와 agent 접근선·상자 경로 이격

사용자 요청으로로컬`ksh_distractor_clearance.py`를추가했다. 주변4개박스를환경원점반경8~12m에배치하고, 모든agent초기위치→담당미는면바깥1m접근점선분과박스외곽여유1.5m·물리상자sweep/다른물체여유1.5m를검사한다. 실패를최대256회재추첨하고미해결시reset재시도로전달한다. 새학습기본clear_paths/구checkpoint view기본legacy와별도workspace/계약을추가했다.

CPU5tests·원본554파일해시·shell구문통과. 실제CPU64장면에서전체4개박스의반경/접근선/sweep이격assert와agent초기안전성·8step추론을통과했고초기낙상종료0건이다. 외부source·보상·RSI비율은유지했다. 해당native정지뷰만교체해GPU5·6086에서새배치를로딩했다.

### GPU5 균등 세 상태 정지 초기 view

사용자 요청으로GPU5 native·6086·tmux`jointpush_initial_33`에safe-start/all_coupled/balanced 초기자세view를띄웠다. N은전원LOCO/전원PUSH/혼합을각1/3추첨하며정책/rollout은진행하지않는다. 초기화렌더동기화1마이크로초1회후root/DOF5frame불변과PNG를확인했고첫화면은push2/loco2였다. 결과`output_etc/check/initial_view_equal33_20261028_211853/`.

balanced view에서R/L확률1/0을sampler의0.5계약검증에보내지않고전원PUSH/LOCO강제선택으로처리하도록로컬bootstrap을수정했다. controls표시도equal-thirds로정정했다. 외부source와MPS서버/다른작업은유지했다.

### 모든 agent 쌍의 최종 초기 배치 안전 검사

사용자 승인으로로컬`ksh_initialization_safety.py`에최종rootXY최소0.7m·모든쌍몸체covering-sphere겹침검사를추가했다. loco-loco와다른그룹도검사하며실패환경의human reference/배치를재추첨한다. 기존그룹·구역·RSI선택과보상은유지한다. loco-only는human도재추첨하는Mixin reset경로로연결했다. 새학습기본all_pairs,기존view기본legacy;새workspace_safe_start/checkpoint계약으로분리했고실행중PID3306277에는반영하지않았다.

CPU11tests·원본554파일해시·shell/Python구문통과. 실제CPU4환경에서balanced160장면(최소간격0.790m)+loco-only64장면(최소0.978m)reset·각8step추론을검증했다. 모든쌍0.7m미만/몸체검사실패0건·초기낙상종료0건. Stage1 19500전이모델로초기배치를검사했으며장기성능/GPU본학습검증은아니다. 결과는`output_etc/check/initialization_safe_start{,_loco}_19500/summary.json`이다. 외부ksh와원본source는수정하지않았다.

### RSI 실제 초기 배치 CPU 진단

사용자 문의로 현재19500 기반hard/balanced 설정과1000epoch 모델을CPU4환경에서진단했다. 초기48장면에서드문root근접을발견했고추가160장면에서7장면의agent간XY간격0.7m미만(최소0.254m)을재현했다. loco-loco4쌍·push/loco3쌍이며root와참조body위치는최대3.8e-6m차이로buffer불일치가아니다. 현배치검사는같은그룹이며한명이라도pushRSI인경우만몸체겹침을검사하고,모든agent쌍의root최소간격을요구하지않는다.

반면전원push인공동50그룹은모두초기gate열림·양손면거리약6cm였다. 각8step(약0.27초)에서낙상종료0/160이며장기안정성검증은아니다. 진단은 `output_etc/check/initialization_audit_19500_v3/{summary.json,results/initialization_audit.json}`에보존했다. 기존학습/환경/보상은바꾸지않았다.

### Contact progress·분할 할당 검증 및 로그 미러 갱신 복구

사용자 문의로 현재Contact progress가가중치0.2×1/(1+max(몸통XY−담당구역중심거리−0.5,0))임을확인했다. 물리상자할당이그룹간중복없고그룹내공유인지·구역순위유일성·n분할폭/중심을무작위1만장면CPU에서검증했다. 코드변경/학습중단없음.

추가한별도eventwriter가원본summaries에동시기록하면서TensorBoard가더이상원본파일을tail하지않는문제를발견했다. 원본지표가79epoch에멈춘동안파생태그만갱신되고있었다. 파생event파일만제거하고CPU변환기가원본scalar전체와분리보상을별도단일-writer `derived_summaries/`에복사하도록수정했다. 현재TensorBoard run은미러를가리키며원본event·학습·optimizer는그대로다. 미러parser에서205개태그·512epoch이상기록을확인했다.

### 보상 태그 그룹 통합

사용자 요청으로 분리보상 태그를 `reward_terms/at_progress`, `reward_terms/contact_progress`처럼같은reward_terms그룹의평면이름으로변경했다. CPU파생로그만재생성하고TensorBoard캐시를갱신해새8개태그와구중첩태그제거를확인했다. 학습과원본event는유지했다.

### 실행 중 Contact / AT TensorBoard 보상 분리

사용자 요청으로 CPU event변환기 `scripts/ksh_reward_split.py`를 추가했다. 현재PUSH-only agent별CONTACT+AT각1개 구성에서 기존전체항−AT항으로CONTACT를 산출하고 state/progress/success/total8개태그를같은summaries에기록한다. 원래가중치·gate·목표포화가반영된agent-step평균이며AMP/패널티는포함하지않는다. 전체과제항과3개component합계일치·CONTACT음수없음을각기록에서검증한다.

학습PID3306277을재시작하지않고GPU없는CPUwatcher3324286을연결했다. 실제59개step변환과TensorBoard8개태그등장을확인했다.30초마다새step을추가하고학습종료시끝난다. 원본source/학습/보상/optimizer는변경하지않았다.

### Stage1 19500 기반 GPU5 재학습·TensorBoard 표시 정리

사용자 지정19500 체크포인트(sha256 `4eb295a8…`)를 받아 Stage1 전이 경로를 선택 가능하게 하고 GPU5/6을 지원하도록 로컬 launcher를 확장했다. 기본14000은 유지하며 custom base는 파일명/해시를 workspace에 반영해 설정 충돌을 막는다. 기존 checkpoint와 model이름·shape/관계metadata/Stage1물성계약이 같음을 CPU에서 확인했고 관련CPU8tests·원본554파일 해시·shell구문 통과.

GPU5 기존MPS서버4104165에 PID3306277을 연결해4명·1024환경·hard18cm·balancedRSI·목표포화로19500부터 fresh Stage2를 시작했다. 실제source_epoch19500 전이·epoch2 encoder/RMS동결·coordination gradient/갱신·startupweight 저장 확인. 기존MPS클라이언트3103831은 유지했다. 실행은 `output/train_base19500_balanced_hardgate_gpu5_20261028_165533/`이며 tmux `jointpush_hardgate_19500`다.

TensorBoard6016에서 이전KSH와5000iter미만 실행은 표시만 제거했다. 새`current_ksh_jointpush_base19500_hard`와기존14k비교2개만 남은 것을API로확인했고 원시weight/log는 보존했다. 이전설정은 해당 launch의`tensorboard_before_cleanup.json`에 남겼다.

### 18cm hard gate + balanced RSI 본학습

사용자 요청으로 전원 양손이 미는 면에서18cm 이내일 때 AT state/progress를 여는 hard gate adapter를 추가했다. CONTACT shaping과 목표 위치 도달 시 gate 면제·CONTACT/AT 포화는 유지한다. 기존 balanced soft 설정과 gate 계약·workspace 경로 외 설정은 같으며, RSI 전원push/혼합/전원loco 각1/3·solo50:50, PUSH-only·단독±45°/공동0°·물성·1024환경을 유지한다. 기존 checkpoint와 분리한 hard workspace/출력에 Stage1 14000부터 Stage2 fresh optimizer로 시작했다.

CPU12tests·설정 동등성·원본554파일 해시·shell 구문 통과. GPU6 MPS PID3260458, tmux `jointpush_hardgate_balanced`에서 실제2epoch 업데이트·encoder/RMS동결·coordination gradient/갱신을 확인했다. output=`output/train_balanced_hardgate_20261028_162947/Stage2JointPushRsi_28-16-29-51`; epoch2 startup weight 저장. TensorBoard6016에 `current_ksh_jointpush_rsi_hard`를 추가하고 이전soft6000도 유지했다. 다른 MPS작업은 유지했다.

### 학습 중단 후 실제 출력 디렉터리 전환

사용자 요청으로 balanced 학습 PID1872208만 공식 MPS terminate_client(rc0)로 종료했다. runs의 실제 output/output_etc를 이 프로젝트로 이동하고 역방향 심링크와 구 runs 출력 경로를 제거했다. 생성 설정·launch 기록은 유지한다. 새 학습/평가의 출력 검증 root도 이 프로젝트로 수정하고 현재 실행 metadata를 stopped·새 경로로 갱신했다.

weight28개 inode/크기 보존, 원본554파일 해시, balanced 설정 일치, Python·shell 구문, 학습/평가 출력 경로 검증 통과. TensorBoard6016만 새 경로로 재연결해 최종6074epoch 로그를 확인했다. 마지막 숫자weight는6000epoch이며 다른 MPS클라이언트2개와 서버는 유지했다. 새 학습·GPU검증은 실행하지 않았다. 이동 기록: `../runs/ksh_output_relocation_20261028.json`.

### 최상위 프로젝트 분리 및 결과 경로 통합

사용자 요청으로 frozen source·전용 adapter/scripts/tests를 `CVPR2027/ksh_jointpush/`에 배치했다. 기존 실행 경로는 원자적 교환 및 호환 링크로 유지하고 현재 학습 결과는 `output/`, 확인 결과는 `output_etc/check/`에서 접근한다. 원본554파일 해시·기존balanced 설정 일치·CPU19tests·이동안전성4tests 통과. 새 경로의2환경 CPU reset/정책/부분reset도rc0·optimizer0으로 통과했다. 학습/supervisor/TensorBoard를 중단·재시작하지 않았다. [실행 안내](EXECUTION.md), [전체 이동 기록](../docs/OUTPUT_LAYOUT.md).
