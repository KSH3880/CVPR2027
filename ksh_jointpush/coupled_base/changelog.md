# Changelog

현재 실행법은 [config](markdowns/config.md), 코드 위치는 [structure](markdowns/structure.md)를 따른다. 정리 전 문서·변경 기록은 [보관 자료](markdowns/legacy/pre_four_configs/README.md)에 있다.

## 2026-11-02

### Stage2 attention에4명 단독 PUSH 구성 추가

- 사용자 요청으로 capture의 `4solo=[0,1,2,3]` 고정 구성을 추가하고 각reset에서4명 모두PUSH template임을 검증한다. renderer/HTML은 선택적4solo와 가변 배치 수를 지원하며 기존 세 배치 출력도 유지한다.
- 기존11.5k 1+3·2+2·4명 기록은 읽기 전용 링크로 재사용하고4solo만3episode×600step 새로 수집한다. 실행/출력: `output_etc/check/stage2_attention_four_layouts_11500_20261102_213606/`. 완료 후 새 비교 HTML을 브라우저로 연다.
- CPU2테스트 통과:4solo를 포함한 원래 shuffled actor와canonical attention출력·relation bias·softmax 일치, 모델/RNG 보존. Python/JS 구문 검사 통과. 실제 수집·renderer 검증 결과는 해당폴더 state/로그에 기록한다.

### 11.5k Stage2 attention 세 배치 비교 실행

- 사용자 요청으로 기존 관찰 entry/비교 renderer를 사용해11.5k의1+3·2+2·4명 PUSH에서 각각3episode×600step,10step간격 actor attention을 수집한다. GPU여유6000MiB 이상일 때 시작하며 학습/진행 중 평가를 종료하지 않는다.
- bias·raw QK·최종softmax 비교 HTML을 생성하고 모델/RMS 불변·원래 actor출력과의 수치 대조를 renderer에서 검증한다. 완료 시 브라우저 열기를 시도한다. 실행 상태는 `runs/jointpush/ksh_jointpush_stage2_attention.json`, 실행/로그는 `output_etc/check/stage2_attention_comparison_11500_20261102_211202/`이다.

### 11k checkpoint 저장 후 일회 평가 예약

- 사용자 요청으로 현재 premature_motion 학습의11000 checkpoint를30초마다 확인하는 일회 watcher를 등록했다. 파일 크기/mtime 안정화·epoch11000·모델 유한값·예약 당시 source hash를 검증하고512환경×seed0/1/2 평가 및 성공/종료/미접촉10·20·50cm 집계를 실행한다. 기존 평가가 있거나GPU여유가6500MiB 미만이면 기다리며 다른 프로세스는 종료하지 않는다.
- 실행: `runs/jointpush/eval_11000_when_ready_20261102_191303/`(CVPR2027 루트 기준), 상태: `runs/jointpush/scheduled_eval_11000.json`. 중복 실행 방지 claim을 사용하며11k 한 번 완료 후 종료한다. shell/Python 구문 검증 및 waiting_checkpoint 로그를 확인했다.

### 최초 출발 대기 지표와 이전15k·현재7.5k 비교 평가

- 사용자 요청으로 같은 capture의 출발 기준 민감도를 추가했다. 속도0.1/0.2m/s·지속0.2/0.5초와 초기 XY변위10/20cm를 비교하며 접촉조건은 유지한다. 별도 `sensitivity.py`가 완료된seed 수를 명시하고 SENSITIVITY.md/JSON에 저장한다. 진행 중인 평가/기존 지표는 변경하지 않았다.

- 사용자 요청으로 `scripts/coupled_departure_capture.py`에 정식 평가 전용 관찰 entry를 추가했다. 기존 학습/보상 코드는 수정하지 않고 초기 reset과 매 control step에서 양손 뒷면 근접·물리 상자 XY/속도를 저장한다. 실제 contact gate(goal 예외 해제)와 준비 판정 일치를 매 sample 확인한다.
- `ksh_departure_eval.py`는 물리 상자당 최초 출발(0.05m/s 초과6sample≈0.2초), 최초 전원 접촉 전 선출발·출발 순간 미준비·무출발/미준비·준비 인원·접촉 전 경로/변위·준비 후 출발 지연을2/3/4명별 집계한다. 첫 팀원 낙상에서 관측을 중단하고 censor 비율을 별도 기록한다. 접촉은 힘 조건이 아닌 기존 gate의18cm 양손 위치 조건이다.
- CPU7테스트 통과: 기다린 뒤 출발, 선출발/거리, 순간 흔들림, 낙상, 준비 후 손 이탈, reset속도/경계값, 물리 그룹 중복 방지. conda118의 LD_LIBRARY_PATH를 지정해 실행했다.
- 실제512환경 첫 seed에서 관찰 entry 초기화·native contact gate 일치 검증과100/1200step 진행을 확인했다. 최종 비교 수치는 아직 집계 중이다.
- 기존 trace에는 손 위치가 없어 두 checkpoint를512환경×seed0/1/2×1200step으로 순차 재평가한다. 기존 평가와 초기 상태·CPU 재채점, 모델 간 초기 상태 일치와 source hash를 자동 검증하며 `runs/jointpush/departure_comparison.json`에 상태를 기록한다. 실행/집계: `output_etc/check/departure_compare_15000_7500_20261102_112157/run.sh`; 완료 수치는 REPORT.md를 확인한다. 두 모델은 학습량·aligned·패널티가 달라 패널티만의 효과로 해석하지 않는다.

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

- 구역goal native 설정에 `placement_order: box_goal_people_v1`을 추가했다. 공동 그룹은 기존 사람 위치와 독립적으로 상자·goal을 먼저 정한 뒤, 각 구역 뒤쪽에 사람을 배치한다. 모든 참여자의 root XY–전체 상자 중심 XY와 상자–goal은2~5m다. 사람 간 간격·다른 상자의 이동 경로와 충돌도 검사한다. 초기 자세·속도는 유지하며 root/kinematic body XY를 함께 이동한다.
- 단독 PUSH2~10m와 다른 과제는 유지한다. 새 필드가 없는 저장 설정은 기존 배치를 재현한다. 앞선2~5m 물리 배치 실패를 해결했으며 실행 중인 본학습은 재시작하지 않았다.
- 관련 CPU69개 통과. 실제8환경에서4명 공동·혼합·단독·독립AT·1+3·2+2 각4회(192장면), 부분reset24회와 물리step의 유한값·거리·충돌·구역goal을 확인했다. 검증은 `output_etc/check/box_first_joint_20261030_230220/physics_verification.json`이며 checkpoint 로드/optimizer step은0회다.

### 공동 PUSH 거리 최종2~5m 정정

- 후속 실제 물리 검증에서 `Stage2 jointpush layout exhausted`가 발생했다. CPU 테스트 통과만으로 실행 가능성을 확정할 수 없다. 기존 반경2.5m의 사람 원형 배치와 모든 공동 참여자–상자 중심2~5m/뒤쪽 배치 조건을 함께 만족시키는 배치를 추가로 해결해야 한다. 새 설정은 본학습에 미적용이며 현재 학습은 기존 설정으로 유지한다. 실패 로그는 같은 검증 폴더의 `physics_retry.log`다.

- 사용자 정정대로 구역goal 모드의 공동 전용 source/goal 범위와 validator를 `[2,5]`로 바꿨다. 단독 PUSH2~10m와 다른 과제 배치, 공동/단독 분리 sampler는 유지한다. 문서·테스트·미적용 설정 metadata를 같은 값으로 갱신했다. 실행 중인 학습은 재시작하지 않았다.
- CPU64개 통과: 공동2048개 혼합 장면의2~5m 범위와 양 끝 표본·단독 기존 배치 bitwise 동일·구 checkpoint 재현·native 설정·구역goal/CONTACT/정책을 확인했다. 현재 검증 경로는 `output_etc/check/joint_distance_2_5_20261030_224333/`다.

### 구역goal 공동 PUSH만 거리1~10m로 분리

- 사용자 최종 지정에 따라 현재 구역goal native YAML의 `stage2JointPush.source_distance/goal_distance`를 각각 `[1,10]`으로 추가했다. 공동 그룹(2명 이상)에만 적용하며 agent–전체 물리 상자 중심 XY는 모든 참여자에 대해 검사한다. 단독 PUSH는 `jointPush.scene`의 기존2~10m를 사용하고 SIT/CLIMB/AT/ON_TOP 배치는 유지한다.
- sampler의 고정2~10m 수식을 공동/단독 설정 선택으로 바꾸고 거리 설정을 checkpoint 계약으로 검증한다. 새 필드가 없는 기존 저장 설정은 원래2~10m로 재현하며 새/구 거리 계약 혼용은 거부한다. 실행 중인 학습PID571143은 재시작하지 않았으므로 원래 저장 설정으로 계속 실행한다.
- CPU64개 통과: 2048개 공동/단독 혼합 장면 범위·공동1~2m 실제 표본·단독 배치 bitwise 동일·기존 저장 설정 재현·native9설정·구역goal/CONTACT/공통정책을 확인했다. 실제8환경에서 공동/혼합/단독/독립AT 각4회 reset·4step 관측/보상finite·고정goal·거리범위를 검증했다. 공동 source최소1.547m·goal최소1.212m, 단독source최소2.137m·goal최소2.034m다. checkpoint/optimizer0회이며 검증은 `output_etc/check/joint_distance_1_10_20261030_223845/`에 둔다.

### 혼합 과제 AT/CONTACT 보상 기록 분리와 승인된 resume

- 공통 진단에서 빠졌던 CONTACT(PUSH relation12)를 직접 집계하고 `reward_terms/{at,contact}_{state,progress,success,total}`8개 별칭을 추가했다. gate·가중치·포화가 적용된 각 관계 활성 edge 평균이며 AT는 독립 AT를 포함한다. SIT/CLIMB 등이 섞이므로 전체−AT로 CONTACT를 추정하지 않는다. 보상 계산·학습 설정은 유지한다.
- 혼합/전원PUSH/CONTACT없음 집계와 보상 불변, 구역goal·공통정책 CPU29개 통과(tokenhsi118 attention 호환 적용). 사용자 승인으로 PID564900을 종료하고 최신 저장본2epoch의 모델·optimizer195state를 복원했다. PID571143·tmux `coupled_split_20261030_210932`, 실제4epoch에서 encoder/RMS동결·coordination/head 갱신·finite를 확인했다. 재시작 전 미저장 진행분은 보존된 로그에만 남는다.
- 실행/검증은 CVPR2027 루트 `runs/jointpush/resume_coupled_reward_split_20261030_210932/`, 새 출력은 `output/stage2_coupled_push_region_goals_split_20261030_210932/Stage2CoupledPushRegionGoals_30-21-09-47/`다. TensorBoard6017 HTTP에서 기존205/205/263태그와 현재283태그·AT/CONTACT 실제 값·11epoch를 확인했다.8개 별칭은 native 진단과 일치하고 각3개 component 합은 total과 일치한다.

### 구역별 goal 본학습 시작 · 로컬21000 수신

- 사용자 수신 완료 후 Stage1 PUSH21000의165 tensor·계약·유한값과 SHA256 `5ac3ee58...`을 확인하고 GPU0·tokenhsi118·4명×1024환경·8토큰·LOCO 시작·새 optimizer로 `stage2_coupled_push_region_goals` 본학습을 시작했다. 이전 공통goal Stage2 모델은 resume하지 않았다.
- 구역goal/공통정책 CPU26개 통과. 실제 epoch2에서 구역goal활성, actor encoder·입력RMS 동결, coordination·action head 갱신, optimizer195state·모델/263scalar 유한값을 확인하고 새goal계약의 startup checkpoint를 저장·재검증했다. 로컬 entry의 초기2epoch 감사는 학습규칙을 변경하지 않는다.
- 학습PID564900·tmux `codex-ksh:coupled_region_goals_20261030_200442`. 출력 `output/stage2_coupled_push_region_goals_20261030_200442/Stage2CoupledPushRegionGoals_30-20-43-15/`, 실행/설정/검증은 CVPR2027 루트 `runs/jointpush/launch_coupled_region_goals_20261030_200442/`다. TensorBoard6017에 이전 두 run과 새 `coupled_region_goals_21000`을 연결했고 HTTP에서205/205/263태그를 확인했다. 기존 두 실험은 AT/CONTACT 미러를 보존하고 새 혼합과제는 원본summary를 사용한다.

### 로컬 tokenhsi118 구역별 goal 학습 준비

- 사용자 학습 요청에 따라 로컬 모션235항목·PUSH21clip 해시를 확인하고 기존 자료로5개 payload 디렉터리를 연결했다. Stage1 PUSH21000 PTH는 로컬에 없으며 Pro1 비대화식 인증이 실패해 현재 본학습은 미시작이다. 기존19500/Stage2 저장본으로 대체하지 않았다.
- 루트 `scripts/coupled_base_run.py`에서 PyTorch2.0 tuple-any·float-bias MHA 및 CUDA11.8/RTX5090의3축곱/JIT fusion 호환을 실행 중에만 적용한다. 구역별 goal train wrapper가 이 entry를 선택하고 공통launch는 기본 native entry를 유지한다. 원래 source/ports·goal/물성/학습 규칙은 바꾸지 않았다.
- CPU goal 테스트6개와 GPU4환경의 공동/단독PUSH 각4step 관측·보상finite, 구역별goal입력·고정·2~10m를 확인했다. 가중치 로드/optimizer0회다. 검증은 `output_etc/check/local_ready_20261030_200442/`, 준비된1024환경 명령과 설정은 `../../../runs/jointpush/launch_coupled_region_goals_20261030_200442/`에 있다.

### Git 배포와 로컬 데이터 준비 경로 보완

- 사용자 로컬 다운로드 실행을 위해 모션목록YAML2개를Git제외에서해제하고서버절대경로body_models/PUSH자료링크는추적에서제외했다. 실제서버자료·링크는보존한다. 준비명령은명시적data/21k경로를받으며coupled에불필요한78k/body_models를요구하지않는다.
- 별도/tmp checkout에서실제235개모션항목·21개PUSH모션/캐릭터해시·21k복사·준비기록을검증했다. [로컬실행안내](LOCAL_RUN.md)에다운로드·별도자료수신·준비·학습·viewer명령을추가했다. 대용량자료/PTH/결과는Git에포함하지않으며검증은 `output_etc/check/region_goals/portable_prepare.json`에있다.

### 공동 PUSH 구역별 고정 goal 추가

- 사용자 요청으로 앞서 받은coupled/Stage1 PUSH21k를 기준으로 `stage2_coupled_push_region_goals`를 추가했다. 초기구역중심−초기전체중심 offset을 기존공통goal에 더해 사람별고정goal을 만들고, 각자의task목표·GTA관측·AT보상/성공·goal saturation·marker·정식평가를 현재구역중심과대응시켰다. 목표는상자회전을따라가지않으며각구역의초기밀기방향/거리2~10m는원래와같다.
- 기존coupled설정·물리상자·배치/충돌경로·CONTACT·18cm접촉gate·PPO·모델·베이스는유지했다. 전용YAML·train/test/VNC/eval·루트wrapper와별도checkpoint계약으로구실험resume를거부한다. 외부ksh/고정ports·기존프로세스는변경하지않았다.
- CPU전체324개·shell9개구문·변경문서링크검사통과. 실제CPU4환경에서21k165tensor전이후공동/혼합/단독PUSH/독립AT각8step관측·행동·보상finite,구역별관측목표·거리·목표고정·부분reset보존·목표도착AT/gate·회전시오차증가를확인했다. optimizer0회·가중치불변·새checkpoint생성없음·본학습미실행. 검증은 `output_etc/check/region_goals/{summary.json,pytest.log,live/verification.json}`, 상세는[구역별목표](markdowns/reference/stage2_coupled_push.md).

## 2026-10-29

### PUSH viewer 목표구 확대

- PUSH 포함 설정의 viewer/VNC/영상 목표구를 반지름3cm에서20cm로 확대했다(지름 약6.7배). 공통 marker 생성부의 표시 scale만 변경했으며 목표 좌표·성공 판정·학습 설정과 비PUSH 설정의 marker 크기는 유지했다.
- coupled1환경의 동일 카메라에서 기존/확대 목표구를 offscreen 렌더링해 가시성을 확인했다. marker scale 변경 전후 정책 관측이 bitwise 동일하고 짧은 시뮬레이션 관측/보상 finite를 확인했다. [확대 화면](output_etc/check/push_goal_marker/render/after.png), [검증](output_etc/check/push_goal_marker/render/verification.json). USAGE에 표시 크기를 기록했다.


### PUSH26k 평가와 Stage2 제로샷 검증

- `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`를 검사했다.5과제/4family·165 tensor·모델/RMS finite를 확인했고 두 입력 PTH의 실행 전후 SHA256은 동일하다. 제공78k는 기존 기준78k와 해시가 같고 새 pth 경로의4환경32step headless 재로드를 완료했다.
- 26k Stage1 PUSH는512환경×3seed·900step 일반3회/1명 행동별 단독15회를 완료했다. 점/영역5과제 macro는 일반 **84.55/88.32%**, 단독 **82.20/86.95%**다. PUSH 점 성공률은 일반87.12%·단독88.61%로21k보다 높지만 다른 과제의 하락으로 전체 점 macro는 낮았다.21k와18개 초기 장면의9개 필드가 bitwise 일치한다.
- near50·coupled는26k tensor165개를 그대로 복사하고 공통 coordination identity/Linear bias0인 제로샷 저장본을 각각 만들었다. 추가 학습0회·두 모델 tensor 동일을 확인했다. 각16환경·4명·1200step 실행과4환경32step headless viewer 재로드를 완료했다. BEFORE ON_TOP 쌍/4명 공동 PUSH viewer 명령을 USAGE에 추가했다. 관찰용 파일에는 optimizer가 없고 기본 YAML 전이21k는 변경하지 않았다.
- 총20개 trace의 CPU/GPU·JSON 건수 일치, float32/64 ever/simultaneous 차이0·finite를 확인했다. [리포트](output_etc/check/push_26k/report.md), [검증](output_etc/check/push_26k/verification.json). Stage2 결과는 작은 제로샷 실행 검사이며 학습된 Stage2 성능·실제 GUI 화면 검사는 아니다. 기존 원본/프로세스는 보존했다.


### BEFORE PUSH near50 거리 변형 추가

- `stage2_before_push_near50` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. `stage2_before_push`의 mix226 대신 참고 설정의 `relative_sum` near50 거리만 적용했다. 기존 모든 YAML은 보존하고 공통 모델·PUSH/독립 행동 비율·4명/1024환경/8토큰/1200step·loco·21k 전이는 유지했다. RSI433과 attention mask는 이식하지 않았다.
- near 모드는 BEFORE 쌍의50%이며 SIT/CLIMB는 후행→받침 거리가 선행→받침보다 작고, ON_TOP은 후행→받침+후행→payload 거리 합이 선행→받침보다 작다. 나머지50%는 original 샘플링이다. validator·viewer/eval의 새 이름/모션/task 경로·checkpoint 혼용 거부를 연결했다.
- CPU313개 통과.1024환경 CHECK epoch2/frame262144 저장·모델/RMS/optimizer finite·actor encoder73 tensor/입력 RMS 동결을 확인했다.32환경5preset에서 near 부등식·PUSH 경로/물리·alias 관측·부분 reset·가중치 불변을 확인했고4환경32step headless viewer와16환경×seed0·1200step 일반 평가를 완료했다. CPU/GPU·JSON 건수 일치, float32/64 ever 차이0·finite다. [검증](output_etc/check/before_push_near50/verification.json).
- config/structure/USAGE/README·BEFORE PUSH 상세에 설정과 명령을 갱신했다. 장기 본학습·512환경×3seed 성능·GUI/VNC 실화면은 실행하지 않았다.


### 공통 Stage2 alias 전처리와 coordination 초기화 수정

- 공통 `Stage2PushBuilder`의 alias 처리를 config 플래그 대신 task packet의 BEFORE 의존 관계로 결정하도록 통일했다. encoder용 복사본만 변환하고 복제 물체 token feature/pose·원본 packet·coordination relation은 보존한다. coupled·독립 packet은 그대로 통과한다. 기존4과제 BEFORE 구현은 유지했다.
- 새 coordination의 모든 Linear bias를 원본처럼0으로 초기화했다. 초기 출력 identity와 Stage1 PUSH165 tensor 전이는 유지하며 기존 학습 PTH 로드 시 저장된 bias를 덮어쓰지 않는다. coupled 제로샷 PTH를 같은 경로에 새 초기화로 재생성하고 이전 관찰용 저장본은 `output_etc/check/stage2_common_fix/previous_zeroshot.pth`에 보존했다.
- CPU305개 통과. BEFORE SIT/CLIMB/ON_TOP·두 쌍·coupled·solo의 같은 가중치/입력에서 config에 무관하게 actor/critic 출력 bitwise 일치, 기존 encoder 동작·복제 feature/pose·원본 packet 보존을 확인했다. 실제 GPU BEFORE 관측과 기존 학습 가중치의 교차 실행도 action/critic 최대 차이0이다.
- 두 설정 모두4명·1024환경 CHECK epoch2/frame262144 저장, 모델/RMS/optimizer finite·actor encoder73 tensor/입력 RMS 동결을 확인했다. BEFORE32환경5preset의 alias 관측/부분 reset/finite 검사와 재생성 제로샷4환경·32step headless 재로드를 완료했다. 이전/새 제로샷의 coupled·solo 출력은 동일하다. [검증](output_etc/check/stage2_common_fix/verification.json). 구조/설정 상세 문서를 갱신했고 장기 성능·실제 GUI 화면은 검증하지 않았다.


### Coupled 제로샷 viewer 저장본

- 제공 Stage1 PUSH21k의 기존165 tensor를 그대로 복사하고 공통 Stage2 coordination을 identity로 초기화한 `output_etc/check/coupled_zeroshot/Stage2CoupledPush_ZeroShot_Stage1_21000.pth`를 만들었다. 추가 optimizer step0·tensor 일치·identity 초기화를 확인했다. 원본 PTH는 보존했다.
- `TASK_GRAPH=all_coupled`로4환경·32step headless test wrapper 재로드를 완료했다. 실제 GUI 화면은 실행하지 않았다. 제로샷 viewer 명령을 USAGE에 추가했고 전이 기록은 `output_etc/check/coupled_zeroshot/transfer.json`에 있다.


### 공통 Stage2 PUSH 모델과 coupled 이식

- `stage2_coupled_push` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. 원격 `stage1_2_jointpush_st`의 coupled straight·밀도3·인원별 공동 상자/구역·그룹 gate·goal saturation을 이식했다. 독립 행동과 AMP 혼합은 현재 Stage1 PUSH 설정(AT25/ON_TOP20/SIT10/CLIMB15/PUSH30)을 읽도록 바꾸고 미사용 Stage2 RSI/adapter 스택은 가져오지 않았다.
- `stage2_before_push`와 coupled는 동일 `Stage2PushBuilder`·관측1272/AMP1330·5과제·NONE/SELF/BEFORE/COUPLED4종 사람 relation·6N task/group suffix를 사용한다. tensor 이름·크기가 같고 Stage1 PUSH21k의165 tensor를 그대로 전이한다. 기존 BEFORE PUSH의5N 관측 검사 저장본은 새 모델에 resume하지 않는다.
- 사용자 기준대로 Stage1은900step을 유지하고 모든 Stage2 학습/일반 평가/viewer 기본 시간을1200step으로 변경했다. 기존4과제 BEFORE 모델은 보존했다. coupled 평가의 PUSH 성공은 가상 구역이 아닌 실제 전체 상자 중심에서 계산한다.
- CPU292개 통과. 두 Stage2 모두4명·1024환경 CHECK가 epoch2/frame262144 저장 후 종료했고 coupled resume은 epoch3/frame393216까지 완료했다. actor encoder73 tensor/입력 RMS 동결·action head6 tensor 갱신·모델/RMS/optimizer finite를 확인했다.
- coupled32환경8preset·BEFORE32환경5preset의 배치/구역/AMP/부분 reset/가중치 불변을 확인했다. 두 설정 모두 일반32환경×seed0·1200step 및 headless viewer4환경·32step 재로드를 완료했다. CPU/GPU·JSON 건수 일치, float32/64 ever/simultaneous 차이0·finite다. [검증 보고](output_etc/check/coupled_integration/report.md), [전체 결과](output_etc/check/coupled_integration/verification.json).
- config/structure/USAGE/README·BEFORE 상세와 [공통 Stage2/coupled 상세](markdowns/reference/stage2_coupled_push.md)를 갱신했다. Stage1 YAML·원본 checkout·외부 PTH·기존 프로세스는 보존했다. 장기 본학습·정규512환경×3seed 성능·GUI/VNC 실화면은 검증하지 않았다.


### Stage2 BEFORE solo PUSH 추가

- `stage2_before_push` 환경/학습 YAML과 전용 train/test/VNC/일반 eval을 추가했다. 독립 역할만 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%를 사용한다. 기존 0.8 쌍 구성·선행 AT·크기 호환 후행 SIT/CLIMB/ON_TOP 균등 선택·mix226·공유 alias·AT gate/progress는 유지한다. 기본4명·1024환경·8토큰·900step·모두 loco다.
- PUSH 전용 큰 상자 bank를 추가해 기본20개 물리 후보에서 배정하며 일반 distractor/BEFORE 받침으로 섞이지 않게 했다. Stage1 PUSH의 밀도3·167종·cone90·거리·접촉 가중치를 적용하고 다중 사람 경로/활성 장애물 검사를 추가했다. 정책5과제·AMP4family를 기존 BEFORE coordination에 연결해 기본21k PUSH에서 tensor165개를 그대로 전이한다. 기존5개 환경/학습 YAML은 변경하지 않았다.
- CPU280개 통과. 최종 CHECK1024환경은 epoch2/frame262144, resume은 epoch3/frame393216 저장을 완료했다. actor encoder73 tensor·입력 RMS 동결, action head6 tensor 갱신, 모델/정규화/optimizer finite를 확인했다. [학습·재개](output_etc/check/before_push/training_validation.json).
- 32환경7개 preset에서 물체 alias/숨김·거리·밀도/마찰·PUSH 경로·AT progress·AMP·부분 reset·가중치 불변을 확인했다. 최종 저장본 일반32환경×seed0·900step과 headless viewer4환경·32step 재로드를 완료했다. 평가 CPU/GPU/JSON 건수 일치·float32/64 ever 차이0·finite를 확인했다. [검증 보고](output_etc/check/before_push/report.md), [전체 검증](output_etc/check/before_push/verification.json). 이는 실행 검사이며 장기 학습 성능·512환경×3seed 평가·GUI/VNC 실화면은 검증하지 않았다.
- config/structure/USAGE/README와 [설정 상세](markdowns/reference/stage2_before_push.md)를 갱신했다. 원본 checkout·외부 checkpoint·기존 실행 프로세스는 보존했다.


### Stage1 PUSH 행동 비율 변경

- 사용자 지정대로 `stage1_push`의 새 학습 크기/행동 prior를 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%로 변경했다. 검증기의 허용 계약·샘플링 테스트·현재 PUSH/USAGE/평가 문서를 갱신했다. 과거 저장본의 분포와 평가 결과는 보존하며21k PTH의 기본 viewer/일반 평가는 저장된PUSH50%를 복원한다.
- CPU35개 통과:8만 사람 슬롯의 크기 조건부 샘플링 빈도·합계1·설정 검증·이전 저장본 복원/호환. [검증](output_etc/check/push_sampling_30/tests.log), [설정](output_etc/check/push_sampling_30/config_check.json). 비율 변경 검증에 GPU 학습/평가는 실행하지 않았다.


### Stage1 PUSH 통합과900step 통일

- `stage1_push` 환경/학습 YAML·train/test/VNC·일반/1명 단독 eval을 추가했다. scratch prior는 SIT7.5/CLIMB18.75/AT24.375/ON_TOP24.375/PUSH25%로 기존 네 과제 상대 비율을 유지한다. 정책은5과제·AMP4family로 확장하고 기존4과제/3family 기본값은 보존했다. PUSH167종·밀도3·cone90·접촉 RSI bank/모션을 이식했으며 모션21개와21k PTH는 외부 원본에 심링크로 연결했다.
- 학습·일반/단독 평가·viewer 기본 시간을900step으로 통일하고 시간 종료의 `-1` 경계를 수정했다. 새5과제 macro와 기존4과제 macro를 구분한다. PTH 옆 sidecar가 없으면 내부 저장 env/sim을 읽는다. 제공21k 일반 평가는 학습 당시 PUSH50% 분포를 복원하고 새 scratch는PUSH25%다. 서로 다른 학습 계약의 resume는 거부한다.1명 PUSH와1명 ON_TOP의 물체/받침 인덱스, 여섯 번째 RSI 통계 축을 일반화했다.
- 제공 `Stage1JointPushDensity3Cone90Finetune_00021000.pth`로512환경×3seed 일반3회·단독15회, 모두900step을 완료했다.5과제 macro 점/영역은 일반 **85.84±1.07 / 88.80±0.81%**, 단독 **83.32±0.51 / 88.78±0.32%**다. PUSH는 일반 **81.78±1.93%**, 단독 **81.51±1.47%**다.18개 trace CPU/GPU·JSON 건수 일치, float32/64 ever 차이0, 상태 finite를 확인했다. [리포트](output_etc/check/push_900/report.md), [검증](output_etc/check/push_900/verification.json).
- 이전78k1200step trace의 앞900frame을 별도로 재채점해 같은 제한 시간의 비교를 추가했다. 기존 원본 trace/보고서는 보존했다. 기존 네 과제의 단독 평균은21k에서 낮아졌으며 수치와 비교 조건은 리포트에 기록했다.
- CPU259개 통과. 실제78k→PUSH 전이 후 기존 네 과제 actor/critic 출력 보존, 원본 브랜치 정책·task encoder·clean scene 관측 코드 일치를 확인했다. Stage1/BEFORE 기존 PTH4환경·32step 재로드도 통과했다. PUSH scratch CHECK는2048환경으로 RSI5517 profile 준비 후 epoch2/frame262144를 저장했고 resume은 epoch3/frame393216까지 진행했다. 모델/RMS finite, scratch 시작·PUSH25%·900step 계약을 확인했다. [학습/재개](output_etc/check/push_900/training_validation.json). 원본 PTH/코드/기존 프로세스는 보존했으며 장기 본학습·GUI/VNC 실화면은 실행하지 않았다.


### 일반 평가와 Stage1 전용 단독 평가 분리

- 일반 `*_eval.sh`는 checkpoint 저장본의 학습 인원·물체·크기·과제/BEFORE 배치를 유지한다. Stage1만 `stage1_single_eval.sh`로1명·행동별 독립 평가를 제공한다. 단독 AT만 기존9종 크기, 나머지는 학습 크기/동적 물리를 따른다. viewer native 성공 진단은 `viewer_diagnostics.py`로 이름과 결과 태그를 분리했다.
- 정식 평가는 loco·정확히1200step·독립 seed0/1/2×512환경 첫 episode, 같은 궤적의 점/영역 ever·seed 평균/표본 std·macro다. 개인 낙상 frame까지 채점하고 해당 관절 drive를 해제하며 다른 사람은 계속 수행한다. BEFORE의 개인/follower 개인·둘 다 ever·동시 성공을 별도 집계한다. 학습 reward/종료 조건은 유지했다. viewer 기본 horizon도1200으로 맞췄다.
- Stage1 원본78000으로 일반3회(2명·4물체), 단독12회(행동별1명)를 완료했다. macro 점/영역은 일반 **95.17±0.84 / 97.17±0.18%**, 단독 **96.58±0.26 / 98.83±0.40%**다. [리포트](output_etc/check/eval_unified/stage1_78k_report.md). 15개 trace CPU float64/GPU·JSON 건수 및 float32/64 ever가 전부 일치했고 상태 finite·원본 checkpoint 해시 보존을 확인했다.
- CPU242개 통과. BEFORE 세 설정은 검사 저장본으로16환경·4명·1200step 실행/재채점(낙상8/10/15명, 각각24pair), viewer는4환경·32step의 새 진단 경로를 확인했다. Stage2 본학습 성능과 GUI/VNC 화면은 평가하지 않았다. [검증](output_etc/check/eval_unified/verification.json), [BEFORE](output_etc/check/eval_unified/before_validation.json). 새 규약·명령·코드 위치를 evaluation_single/config/structure/USAGE에 갱신하고 이전 단독 규약은 legacy에 보존했다.


### 네 실험 이름 통일

- 환경/학습 YAML·전용 train/test/VNC12개·실험/출력 이름·테스트·현재 문서를 `stage1`, `stage2_before_nonemask_rsi433_near50`, `stage2_before`, `stage2_before_rsi433`로 통일했다. Stage1 보조 구현은 `stage1_config.py`, `stage1_geometry.py`로 바꿨으며 기본 전이 링크는 `checkpoints/Stage1_00078000.pth`다. 외부 원본 파일과 과거 결과 경로는 보존했다.
- 기존78000 및 이전 BEFORE checkpoint의 reward 태그·RSI 경로를 읽기 경계에서 변환한다. 보상·관측·모델 tensor·실험 간 혼용 거부는 유지한다. 기존 네 저장본을 새 test wrapper로4환경·32step headless 재로드해 정상 종료했고 네 모델의 전체 tensor·actor/critic 출력은 변경 전과 bitwise 동일했다. [재로드](output_etc/check/config_rename/smoke.json), [모델](output_etc/check/config_rename/model_equivalence.log). CPU224개 통과. 셸/Python/YAML·문서 링크·이전 실행 이름 잔여 검사 결과는 `output_etc/check/config_rename/`에 있다. GUI/VNC 실화면과 본학습은 실행하지 않았다.

### 네 컨피그와 학습·추론 경로 정리

- `stage1_v2_fixed`, `stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200`, `stage2_before_ground_mix226_atprogress_objalias_1200`, `stage2_before_ground_mix226_atprogress_objalias_rsi433_1200`만 유지했다. 나머지21쌍 YAML·63개 wrapper와 공동 운반·공유 받침·adapter·full RSI·carry50·단일인 태스크·전처리·미사용 에셋/모듈을 제거했다. 기존 파일322개를 삭제했으며 과거 문서는 legacy에 보존했다. [변경 목록](output_etc/check/four_config_cleanup/change_manifest.json).
- BEFORE의 공통 기하·배치·물리 검사·native 후행 RSI를 분리하고 모델을 공통 정책의 직접 확장으로 정리했다. 공통 환경의 관측·reset·모션/AMP를 별도 mixin으로 분리했다. 파라미터 이름·관측·보상·거리·초기화·동결/학습 범위를 유지했다.
- 전용12개 wrapper는 공통 `launch.sh`를 사용한다. conda 기본값은 모두 `tokenhsi`, seed는42, test 기본은 viewer다. GPU는 한 번 지정하며 기존 MPS만 연결한다. 설정 로더는 새 학습의 native YAML과 추론/resume의 저장 env/sim을 같은 경로에서 처리한다. `ENV_CONFIG`로 명시적 대체가 가능하며 fixed 임시 YAML 생성·실험 이름별 중복 분기를 제거했다.
- Stage1 78000은 `checkpoints/Stage1V2Fixed_00078000.pth` 심링크로 연결했다. 네 모션 데이터 디렉터리도 공유 원본에 연결했다. 원본 checkout·가중치·기존 학습 PID3432492는 보존했다. 확인 산출물·RSI 캐시·pytest 임시 파일은 `output_etc/check/`에 둔다.
- CPU214개 통과: 네 설정/계약 혼용 거부·실제78000 전이·설정 복원·wrapper/conda 선택·GTA·보상·거리·mask·RSI/fallback·재로드. [CPU](output_etc/check/four_config_cleanup/cpu.log). 네 모델의 전체 tensor·actor/critic 출력·계약이 정리 전과 bitwise 동일했다. [모델](output_etc/check/four_config_cleanup/model_equivalence.log). 네 설정의 전체 reset2회·부분 reset1회에서 관측·사람/상자 상태·크기·목표·물체 연결의 최대 차이0을 확인했다. [reset](output_etc/check/four_config_cleanup/reset_equivalence.json). 후행 RSI sampler의 출력·RNG도 동일했다.
- GPU0/MPS에서 fixed2명·2048환경과 BEFORE4명·1024환경의 CHECK가 각각 epoch2/frame262144 저장 후 종료했다. fixed native RSI6413 profile을 생성했다. BEFORE 세 설정은 source78000·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/scalar finite를 확인했다. mix226 resume은 epoch3 저장 후 종료했다. [학습/재개](output_etc/check/four_config_cleanup/training_validation.json).
- 실제78000 fixed headless 추론과 BEFORE 전용 test의4환경·32step 저장본 재로드를 확인했다. BEFORE32환경의 loco/mix·ON_TOP pickup/carry·단독 등14개 구성에서 alias 관측/pose·활성 물리 수·거리·AT progress·AMP·부분 reset·가중치 불변을 확인했다. [물리 검사 진행 결과](output_etc/check/four_config_cleanup/physics.log), [wrapper](output_etc/check/four_config_cleanup/wrapper_validation.json). 셸/Python/YAML·현재 문서 링크·4쌍 YAML/12개 wrapper·미사용 import/실험 참조 제거를 검사했다. VNC 실행 의존성이 없어 실제 GUI/VNC 화면과 장기 성능은 검증하지 않았다. 본학습은 시작하지 않았다.

## 2026-10-28

- 코드 정리용 독립 복사본을 만들었다. 원본과 복사 범위·초기 검증은 `code_copy_manifest.json`과 [당시 changelog](markdowns/legacy/pre_four_configs/changelog.md)에 보존한다.
