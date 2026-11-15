# Changelog

현재 실행법은 [config](markdowns/config.md), 코드 위치는 [structure](markdowns/structure.md)를 따른다. 정리 전 문서·변경 기록은 [보관 자료](markdowns/legacy/pre_four_configs/README.md)에 있다.

## 2026-11-15

### Stage1 PUSH task attention bias 어블레이션

- `stage1_push_no_bias` 환경/학습 YAML과 전용 train/test/VNC·일반/단독 eval wrapper를 추가했다. `output/stage1_push`의 고정 비율 scratch 설정에서 actor/critic의 task attention bias 계산만 생략하고 GTA·task embedding·Action MLP·보상·RSI·물리·PPO를 유지한다. 본학습 출력은 `output/stage1_push_no_bias/`이며 기존 bias 활성 PTH와의 재개·viewer·평가 혼용을 거부한다. [설정 상세](markdowns/reference/stage1_push_no_bias.md), [명령](markdowns/config.md), USAGE·structure를 갱신했다.
- CPU250개 통과·외부 PTH6개 skip. bias 계산 미호출·초기0-bias 기준 출력 일치·task embedding/Action MLP gradient·padding/순열·5행동 eval 명령 경로를 확인했다. 실제2명·2048환경·4물체의 짧은 PPO 저장(epoch2)과 재개(epoch3), 모델/optimizer/RMS finite·Action MLP 갱신·bias projection0 유지,2환경·16step headless 추론을 확인했다. [CPU](output_etc/check/stage1_push_no_bias/cpu_tests.log), [PUSH 회귀](output_etc/check/stage1_push_no_bias/push_regression.log), [저장 검증](output_etc/check/stage1_push_no_bias/training_verification.json), [재개·추론](output_etc/check/stage1_push_no_bias/resume_verification.json). 본학습·정식 성능/FPS 비교·GUI/VNC는 실행하지 않았으며 기존 학습 PID54094를 유지했다.

## 2026-11-14

### Adaptive 재개 후 콘솔 FPS 표시

- 영역 기준 재개 스크립트의 tee 파이프로 Python stdout이 버퍼링되어 adaptive 초기 로그 뒤 FPS 출력이 지연됨을 확인했다. TensorBoard에서는 학습이 계속 진행 중이었다. 향후 학습의 원래 FPS print에 flush=True를 추가하고 재개 스크립트에 PYTHONUNBUFFERED=1을 지정했다. 실행 중인 PID54094는 재시작하지 않고 실제 TensorBoard FPS를 읽어 base_2의 TTY에 `[TB]` 표시로 전달했다. epoch2603의 FPS가 pane에 보이는 것을 확인했고 변경 Python 구문·diff 검사를 통과했다. [표시 상태](output_etc/check/stage1_push_adaptive_console/status.json).

### Stage1 PUSH adaptive 영역 성공률 전환

- 영역(region) 성공률로 adaptive의 집중 선택·90% 숙련·80% 망각을 변경했다. 점/영역이 다른 모의 평가에서 영역 점수를 읽고 숙련을 판정하는 회귀 검증을 포함해 CPU 검증을 통과했다. 2k의 전체 controller 상태가 영역 기준 재계산과 동일함을 확인하고2500 저장본에서 재개했다. 새 run `Stage1PushAdaptive_14-16-54-58`의 학습 PID54094에서 region 로드 로그·MPS 서버 client 등록·epoch2501 PPO 진행·CLIMB50% 슬롯을 확인했다. 일반 저장본 재개라 AMP replay는 새로 채운다. [테스트](output_etc/check/stage1_push_adaptive_region/tests_full.log), [2k 이력 검증](output_etc/check/stage1_push_adaptive_region/history_equivalence.json), [적용 검증](output_etc/check/stage1_push_adaptive_region/activation_verification.json).

### GPU0 MPS 준비 및 adaptive 자동 전환 감시

- GPU0 전용 MPS 서버를 시작하고 runtime 자동 연결·작은 CUDA 연산·서버 client PID 일치를 확인했다. 기존 adaptive를 강제 종료하지 않고2000 epoch 평가 뒤 원래 supervisor의 재개로 전환했으며, 새 학습 PID4190730가 MPS 서버 client 목록에 실제 등록된 것을 확인했다. 기존 schedule PID22685는 점검 당시 이미 종료된 상태였고 이 작업에서 종료하지 않았다. [초기 접속 검증](output_etc/check/mps_recovery_20261114/probe.json), [학습 연결 검증](output_etc/check/mps_recovery_20261114/connected.json).

### Stage1 PUSH schedule109000 정식 단독 평가

- 사용자 요청으로 최신109000 저장본을5행동·각1명·512환경×seed0/1/2·900step·loco로 순차 평가했다. 점/영역 평균은 AT62.11/62.11%, ON_TOP66.28/67.51%, SIT76.17/89.00%, CLIMB87.83/95.70%, PUSH91.54/91.54%다. macro5 점76.78%·영역81.17%. CPU float64/GPU·JSON 건수·finite·checkpoint 해시 보존을 검증했고 float32/64 ever 차이는0건이다. 자원 guard 발동 없이 종료했고 기존 두 학습 PID가 평가 종료 시 유지됨을 확인했다. [리포트](output_etc/check/stage1_push_schedule_109000_single512_seeds012/report.md), [검증](output_etc/check/stage1_push_schedule_109000_single512_seeds012/verification.json), [자원 요약](output_etc/check/stage1_push_schedule_109000_single512_seeds012/resource_summary.json).

### Stage1 PUSH 성공률 기반 집중 샘플링 실험

- 별도 scratch용 `stage1_push_adaptive` 환경/학습 YAML과 train/test/VNC·일반/단독 eval wrapper를 추가했다. 최초2k epoch는5행동 각20%, 이후2k마다 행동별512환경·seed0·loco 단독 점 성공률로 판단한다. 한 행동50%·각 행동 최소5%·집중6k~12k,90% 이상2회 숙련·80% 미만2회 망각, 전체 숙련 시 균등20%로 계속 학습한다. epoch/성공 종료 조건은 없고 기존 보상·RSI·물리·PPO 설정은 유지한다. [설정 상세](markdowns/reference/stage1_push_adaptive.md), [실행 명령](markdowns/config.md).
- 새 실험은 행동별 물리 슬롯을 배분하고 확률이 변할 때만 supervisor가 같은 run을 재개한다. 정책/optimizer/RMS·전역 epoch/frame·controller 상태·AMP replay를 이어받고 expert demo는 새 family 비율로 생성한다. 수동 resume은 원본 보존을 위해 새 run에 저장한다. 단독 평가/viewer는 배치 간격·인원 조정 전의 학습 계약으로 호환성을 확인한다. config/USAGE/structure를 갱신했다.
- CPU191개 통과·외부 PTH 관련5개 skip. 실제2명·2048환경·4물체의 reset·슬롯/AMP 비율·PPO2회, 합성 성공률로 유도한 전환 저장, native train wrapper 재개 epoch2001의 가중치 갱신·optimizer/RMS finite·replay20만 복원과 CLIMB50%/나머지12.5%를 확인했다. epoch2000 경계는 검증용 counter 주입이며2k epoch 학습 성능 검증이 아니다. 실제 자동 evaluator 경로도5행동×16환경·seed0·loco로 확인했고 낙상 종료 trace의 CPU float64/GPU/JSON 일치·finite·float32/64 ever 차이0을 확인했다. [검증 보고](output_etc/check/stage1_push_adaptive_build/report.md), [검증 JSON](output_etc/check/stage1_push_adaptive_build/verification.json).
- 균등·CLIMB 집중 조합의 RSI 캐시3182개를 검증 후 로컬 공용 캐시에 추가해7681→10863개로 보강했다. 기존 모든 mask/phase를 보존하고 원본을 검증 폴더에 백업했다. 다른 분포의 미등록 조합은 추후 추가 screening이 필요할 수 있다. 기존 본학습 PID22685는 유지했고 새 본학습은 시작하지 않았다. [캐시 보존 검증](output_etc/check/stage1_push_adaptive_build/shared_cache_update.json).

## 2026-11-13

### Stage1 PUSH schedule102500 정식 단독 재평가

- 사용자 요청으로 평가 시작 시 최신 `Stage1PushSchedule_00102500.pth`를5행동·각1명·512환경×seed0/1/2·900step·loco100%로 평가했다. 점/영역 평균은 AT63.93/63.93%, ON_TOP69.01/70.44%, SIT66.21/88.35%, CLIMB92.51/99.67%, PUSH86.39/86.39%다. macro5는 점75.61±1.59%·영역81.76±1.15%(표본std)로99000보다 점1.93%p·영역1.03%p 높으며86500의 전체 평균보다 낮다. [리포트·명령](output_etc/check/stage1_push_schedule_102500_single512_seeds012/report.md).
- 공식 evaluator의15개 명령을 최대3개씩 병렬 실행했다. 초기 장면10개 tensor가15회 모두99000과 bitwise 동일하며 전체900frame·CPU float64/GPU·JSON 건수 일치·finite·PTH 해시 보존·float32/64 ever 차이0을 확인했다. 본학습 PID22685와 코드·설정은 유지했다. [검증](output_etc/check/stage1_push_schedule_102500_single512_seeds012/verification.json), [CLIMB 진단](output_etc/check/stage1_push_schedule_102500_single512_seeds012/climb_diagnostics.json).

### Stage1 PUSH schedule99000 정식 단독 재평가

- 사용자 요청으로 평가 시작 시 최신 `Stage1PushSchedule_00099000.pth`를5행동·각1명·512환경×seed0/1/2·900step·loco100%로 평가했다. 점/영역 평균은 AT59.44/59.44%, ON_TOP68.68/70.18%, SIT64.45/89.97%, CLIMB91.60/99.80%, PUSH84.24/84.24%다. macro5는 점73.68±1.11%·영역80.73±0.91%(표본std)로86500보다 점8.36%p·영역5.14%p 낮다. 다섯 행동 모두 점 성공률이 하락했고 특히 AT·SIT의 하락이 크다. [리포트·명령](output_etc/check/stage1_push_schedule_99000_single512_seeds012/report.md).
- 공식 evaluator의15개 명령을 최대3개씩 병렬 실행했다. 초기 장면10개 tensor가15회 모두86500과 bitwise 동일하며 전체900frame·CPU float64/GPU·JSON 건수 일치·finite·PTH 해시 보존·float32/64 ever 차이0을 확인했다. 본학습 PID22685와 코드·설정은 유지했다. [검증](output_etc/check/stage1_push_schedule_99000_single512_seeds012/verification.json), [CLIMB 진단](output_etc/check/stage1_push_schedule_99000_single512_seeds012/climb_diagnostics.json).

## 2026-11-12

### Stage1 PUSH schedule86500 정식 단독 재평가

- 사용자 요청으로 평가 시작 시 최신 `Stage1PushSchedule_00086500.pth`를5행동·각1명·512환경×seed0/1/2·900step·loco100%로 평가했다. 점/영역 평균은 AT70.05/70.05%, ON_TOP76.89/77.86%, SIT80.79/95.70%, CLIMB96.35/99.61%, PUSH86.13/86.13%다. macro5는 점82.04±0.49%·영역85.87±0.28%(표본std)로82000보다 점7.64%p·영역6.48%p 높다. CLIMB 점은1.17%p 낮아졌고 나머지 네 행동 점 성공률은 회복했다. [리포트·명령](output_etc/check/stage1_push_schedule_86500_single512_seeds012/report.md).
- 공식 evaluator의15개 명령을 최대3개씩 병렬 실행했다. 초기 장면10개 tensor가15회 모두82000과 bitwise 동일하며 전체900frame·CPU float64/GPU·JSON 건수 일치·finite·PTH 해시 보존·float32/64 ever 차이0을 확인했다. 본학습 PID22685와 코드·설정은 유지했다. [검증](output_etc/check/stage1_push_schedule_86500_single512_seeds012/verification.json), [CLIMB 진단](output_etc/check/stage1_push_schedule_86500_single512_seeds012/climb_diagnostics.json).

## 2026-11-11

### Stage1 PUSH schedule82000 정식 단독 재평가

- 사용자 요청으로 평가 시작 시 최신 `Stage1PushSchedule_00082000.pth`를5행동·각1명·512환경×seed0/1/2·900step·loco100%로 평가했다. 점/영역 평균은 AT64.71/64.71%, ON_TOP64.91/65.76%, SIT67.90/90.56%, CLIMB97.53/98.96%, PUSH76.95/76.95%다. macro5는 점74.40±1.20%·영역79.39±0.84%(표본std)이며76500보다 점7.47%p·영역5.61%p 낮다. 다섯 행동 모두 점 성공률이 하락했고 원인을 단정하거나 학습 설정을 바꾸지 않았다. [리포트·명령](output_etc/check/stage1_push_schedule_82000_single512_seeds012/report.md).
- 공식 evaluator의15개 명령을 최대3개씩 병렬 실행했다. 초기 장면10개 tensor가15회 모두76500과 bitwise 동일하며 전체900frame·CPU float64/GPU·JSON 건수 일치·finite·PTH 해시 보존·float32/64 ever 차이0을 확인했다. 본학습 PID22685와 코드·설정은 유지했다. [검증](output_etc/check/stage1_push_schedule_82000_single512_seeds012/verification.json), [CLIMB 진단](output_etc/check/stage1_push_schedule_82000_single512_seeds012/climb_diagnostics.json).

### Stage1 PUSH schedule76500 정식 단독 재평가

- 사용자 요청으로 평가 시작 시 최신 `Stage1PushSchedule_00076500.pth`를5행동·각1명·512환경×seed0/1/2·900step·loco100%로 평가했다. 점/영역 평균은 AT72.53/72.53%, ON_TOP76.24/78.06%, SIT79.49/92.19%, CLIMB98.57/99.67%, PUSH82.55/82.55%다. macro5는 점81.88±0.70%·영역85.00±0.10%(표본std)로64500보다 상승했다. 점 성공률은 다섯 행동 모두 상승했고 PUSH도77.60→82.55%로 회복했다. [리포트·명령](output_etc/check/stage1_push_schedule_76500_single512_seeds012/report.md).
- 공식 evaluator의15개 명령을 최대3개씩 병렬 실행했다. 초기 장면10개 tensor가15회 모두64500과 bitwise 동일하며 전체900frame·CPU float64/GPU·JSON 건수 일치·finite·PTH 해시 보존·float32/64 ever 차이0을 확인했다. 본학습 PID22685와 코드·설정은 유지했다. [검증](output_etc/check/stage1_push_schedule_76500_single512_seeds012/verification.json), [CLIMB 진단](output_etc/check/stage1_push_schedule_76500_single512_seeds012/climb_diagnostics.json).

## 2026-11-10

### Stage1 PUSH schedule64500 정식 단독 재평가

- 사용자 요청으로 평가 시작 시 최신 `Stage1PushSchedule_00064500.pth`를5행동·각1명·512환경×seed0/1/2·900step·loco100%로 평가했다. 점/영역 평균은 AT63.02/63.02%, ON_TOP68.36/70.05%, SIT78.45/92.90%, CLIMB95.90/99.93%, PUSH77.60/77.60%다. macro5는 점76.67±1.29%·영역80.70±0.34%(표본std)로59000보다 상승했다. AT·ON_TOP·SIT은 개선됐으며 CLIMB 점은97.01→95.90%, PUSH는80.40→77.60%로 낮아졌다. [리포트·명령](output_etc/check/stage1_push_schedule_64500_single512_seeds012/report.md).
- 공식 evaluator의15개 명령을 최대3개씩 병렬 실행했다. 초기 장면10개 tensor가15회 모두59000과 bitwise 동일하며 전체900frame·CPU float64/GPU·JSON 건수 일치·finite·PTH 해시 보존·float32/64 ever 차이0을 확인했다. 본학습 PID22685와 코드·설정은 유지했다. [검증](output_etc/check/stage1_push_schedule_64500_single512_seeds012/verification.json), [CLIMB 진단](output_etc/check/stage1_push_schedule_64500_single512_seeds012/climb_diagnostics.json).

## 2026-11-09

### Stage1 PUSH schedule59000 정식 단독 재평가

- 사용자 요청으로 평가 시작 시 최신 `Stage1PushSchedule_00059000.pth`를5행동·각1명·512환경×seed0/1/2·900step·loco100%로 평가했다. 점/영역 평균은 AT57.55/57.55%, ON_TOP64.78/66.73%, SIT70.12/92.32%, CLIMB97.01/99.87%, PUSH80.40/80.40%다. macro5는 점73.97±0.24%·영역79.38±0.40%(표본std)로53500보다 상승했다. CLIMB는 높은 수준을 유지했고 PUSH는82.88→80.40%로 낮아졌다. [리포트·명령](output_etc/check/stage1_push_schedule_59000_single512_seeds012/report.md).
- 공식 evaluator가 생성한15개 명령을 최대3개씩 병렬 실행했다. 15개 초기 장면10개 tensor가53500과 bitwise 동일하며 전체900frame·CPU float64/GPU·JSON 건수 일치·finite·PTH 해시 보존·float32/64 ever 차이0을 확인했다. 본학습 PID22685와 코드·설정은 유지했다. [검증](output_etc/check/stage1_push_schedule_59000_single512_seeds012/verification.json), [CLIMB 진단](output_etc/check/stage1_push_schedule_59000_single512_seeds012/climb_diagnostics.json).

### Stage1 PUSH schedule53500 정식 단독 평가

- 사용자 요청으로 평가 시작 시 최신 `Stage1PushSchedule_00053500.pth`를5행동·각1명·512환경×seed0/1/2·900step·loco100%로 평가했다. 점/영역 seed 평균은 AT56.45/56.45%, ON_TOP59.24/61.13%, SIT58.07/79.49%, CLIMB97.14/99.67%, PUSH82.88/82.88%다. macro5는 점70.75±0.74%·영역75.92±0.37%(표본std)다. 기존 고정 비율75000과 다른 학습 run·epoch임을 구분했다. [리포트·명령](output_etc/check/stage1_push_schedule_53500_single512_seeds012/report.md).
- CLIMB1536시도 중1492회 점 성공·1531회 내부 XY와 목표 높이 동시 만족을 확인했다. 15개 초기 장면10개 tensor가 기존75000과 bitwise 동일하며 전체900frame·CPU float64/GPU·JSON 건수 일치·finite·PTH 해시 보존·float32/64 ever 차이0을 확인했다. 본학습 PID22685와 코드·설정은 유지했다. [검증](output_etc/check/stage1_push_schedule_53500_single512_seeds012/verification.json), [CLIMB 진단](output_etc/check/stage1_push_schedule_53500_single512_seeds012/climb_report.md).

## 2026-11-05

### Stage1 PUSH 고정 epoch 스케줄 컨피그 추가

- 새 scratch용 `stage1_push_schedule` 환경/학습 YAML과 train/test/VNC·수동 일반/단독 eval wrapper를 추가했다. 초반 AT20/ON_TOP15/SIT10/CLIMB30/PUSH25%,30k~40k 선형 전환 후 AT25/ON_TOP20/SIT5/CLIMB25/PUSH25%로 유지한다. 본학습 epoch 제한은 해제했고 자동 eval·성공률 전환/종료 조건은 없다. 보상·RSI·물리·PPO는 유지한다.
- 고정 자산의 크기 호환성을 보존하면서 자산 집단의 평균 과제 확률을 목표 비율에 맞췄고 AMP family 비율도 같은 epoch를 따른다. 스케줄 명세와 전역 epoch 상태를 PTH에 저장해 resume/viewer가 저장 시점 분포를 복원하며 기존 고정 분포 PTH와 혼용을 거부한다. config/USAGE/structure·[상세](markdowns/reference/stage1_push_schedule.md)를 갱신했다.
- CPU117개 통과·외부 PTH 관련5개 skip. 실제2명·2048환경·4물체의0/30k/35k/40k/100만 epoch reset·finite·크기/PUSH 불변·AMP 비율·35k 상태 복원을 확인했다. 짧은 학습 epoch2와 전용 wrapper 재개 epoch3에서 모델/RMS/optimizer finite·가중치 변경을 확인했다. 공용 RSI 캐시는 기존5360개 항목을 보존하고 검증된2321개만 추가했다. 본학습 PID1908198은 유지했으며 새 본학습·정책 eval·GUI/VNC는 실행하지 않았다. [검증 보고](output_etc/check/stage1_push_schedule/report.md), [검증 JSON](output_etc/check/stage1_push_schedule/verification.json).

### Stage1 PUSH scratch75000 정식 단독 재평가

- 평가 시작 시 최신 `Stage1Push_00075000.pth`를5행동·각1명·512환경×seed0/1/2·900step·loco100%로 평가했다. 점/영역 seed 평균은 AT80.53/80.53%, ON_TOP89.45/92.06%, SIT87.57/96.68%, CLIMB0/0%, PUSH88.80/88.80%다. macro5는 점69.27±0.59%·영역71.61±0.25%(표본std)로63500보다 개선됐으며 CLIMB는 회복되지 않았다.
- CLIMB1536시도의 상자 외곽20cm 이내 접근은384회(25%)로63500의70.18%보다 낮아졌다. 골반25cm 이상 상승과 내부 XY·목표 높이 동시 만족은 모두0회다. 샘플링 변경 제안은 적용하지 않았고 본학습·보상·YAML은 유지했다. [CLIMB 진단](output_etc/check/stage1_push_75000_single512_seeds012/climb_report.md).
- 15개 초기 장면10개 tensor 필드가63500과 bitwise 동일하며900frame·CPU float64/GPU·JSON 건수 일치·float32/64 ever 차이0·finite·PTH 해시 보존을 확인했다. [리포트·명령](output_etc/check/stage1_push_75000_single512_seeds012/report.md), [검증](output_etc/check/stage1_push_75000_single512_seeds012/verification.json).

## 2026-11-04

### Stage1 PUSH scratch63500 정식 단독 평가·CLIMB 진단

- 평가 시작 시 최신 `Stage1Push_00063500.pth`를5행동·각1명·512환경×seed0/1/2·900step·loco100%로 평가했다. 점/영역 seed 평균은 AT65.36/65.36%, ON_TOP76.95/80.86%, SIT84.96/94.14%, CLIMB0/0%, PUSH83.01/83.01%다. macro5는 점62.06±0.37%·영역64.67±0.24%(표본std)다.52500 대비 AT·SIT 점·PUSH는 개선됐고 ON_TOP은 낮아졌다.
- CLIMB1536시도 중1078회(70.18%)는 상자 외곽20cm 이내 접근했지만 골반25cm 이상 상승은0회, 내부 XY·목표 높이 동시 만족도0회였다. 학습과 평가의 root 목표 높이는 모두 상자 윗면+0.94m이며 평가는 발 조건을 요구하지 않는다. RSI 혼합 학습 프레임 지표와 loco 단독 성공률을 구분해 기록했다. 원인을 단정하거나 본학습·보상·YAML을 변경하지 않았다. [CLIMB 진단](output_etc/check/stage1_push_63500_single512_seeds012/climb_report.md).
- 15개 초기 장면10개 tensor 필드가52500과 bitwise 동일하며900frame·CPU float64/GPU·JSON 건수 일치·float32/64 ever 차이0·finite·PTH 해시 보존을 확인했다. 본학습 PID1908198은 유지했다. [리포트·명령](output_etc/check/stage1_push_63500_single512_seeds012/report.md), [검증](output_etc/check/stage1_push_63500_single512_seeds012/verification.json).

## 2026-11-03

### Stage1 PUSH viewer 인원·물체 수 제한 해제

- viewer/test/VNC의1/2명·M≤2N 제한을 제거해 N≥1·M≥N으로 확장했다. ON_TOP은 남는 M−N개 받침을 서로 다르게 배정하고 초과 과제만 크기 호환 대체 과제로 재샘플링한다. `holding_ontop` 고정 요청도 부족하면 재샘플링한다. 다인 PUSH의 모든 사람·경로·상자 검사와 실제 받침을 제외한 distractor 재배치를 연결했다. 기존 학습2명·4물체·2048환경과 정식 단독1명 경로·RNG는 유지한다.
- CPU57개 통과·외부 PTH 관련3개 skip. 실제52500 PTH로3/3·4/4·4/5·4/12·4/8·8/8 등7개 구성의4환경·32step·부분 reset·유효 과제/받침·PUSH 경로·finite·가중치 불변을 확인했다. test wrapper4명/4물체의 headless와 실제 GUI32step도 정상 종료했다. config/USAGE/structure/PUSH 상세와 전용 simulator checker를 갱신했다. 본학습은 재시작하지 않았다. [검증 보고](output_etc/check/push_viewer_counts/report.md), [검증](output_etc/check/push_viewer_counts/verification.json).

### Stage1 PUSH scratch52500 정식 단독 재평가

- `Stage1Push_00052500.pth`의5행동·각1명·512환경×seed0/1/2·900step·loco100% 평가를 완료했다. 점/영역 seed 평균은 AT62.30/62.30%, ON_TOP82.10/86.26%, SIT78.39/94.21%, CLIMB0/0%, PUSH77.86/77.86%다. macro5는 점60.13±0.13%·영역64.13±0.23%(표본std)다.47000 대비 ON_TOP은 개선됐고 전체 평균은 소폭 낮아졌다.
- 코드 수정 중 시작된 CLIMB seed1의 초기 상태 차이를 발견해 해당 실행을 `pre_fix_cases/`에 보존하고 완료된 코드로 다시 평가했다. 최종15개 초기 장면의10개 tensor 필드는47000과 모두 bitwise 동일하다. CPU float64/GPU·JSON 건수 일치·float32/64 ever 차이0·finite·PTH 해시 보존을 확인했다. [리포트·명령](output_etc/check/stage1_push_52500_single512_seeds012/report.md), [검증](output_etc/check/stage1_push_52500_single512_seeds012/verification.json).

### Stage1 PUSH scratch47000 정식 단독 재평가

- 최신 `Stage1Push_00047000.pth`를5행동·각1명·512환경×seed0/1/2·900step·loco100%로 총15회 평가했다. 점/영역 seed 평균은 AT66.80/66.80%, ON_TOP77.99/83.59%, SIT79.23/94.27%, CLIMB0/0%, PUSH79.04/79.04%다. macro5는 점60.61±0.86%·영역64.74±0.75%(표본std)다.38000 대비 운반 두 과제·SIT가 개선됐고 PUSH는80.79→79.04%로 소폭 낮아졌다.
- 15개 초기 장면의10개 tensor 필드가38000과 모두 bitwise 동일하다.15개 trace의900frame·CPU float64/GPU·JSON 건수 일치·float32/64 ever 차이0·finite·PTH 해시 보존과 seed별 서로 다른 초기 상태를 확인했고 본학습은 유지했다. [리포트·명령](output_etc/check/stage1_push_47000_single512_seeds012/report.md), [집계](output_etc/check/stage1_push_47000_single512_seeds012/summary.json), [검증](output_etc/check/stage1_push_47000_single512_seeds012/verification.json).

## 2026-11-02

### Stage1 PUSH scratch38000 정식 단독 평가

- 사용자 요청대로 최신 `Stage1Push_00038000.pth`를5행동·각1명·512환경×seed0/1/2·900step·loco100%로 총15회 평가했다. 행동당1536시도이며 점/영역 seed 평균은 AT30.86/30.86%, ON_TOP47.07/51.69%, SIT71.22/88.48%, CLIMB0/0%, PUSH80.79/80.79%다. macro5는 점45.99±0.89%·영역50.36±0.55%(표본std)다. 이전32환경 평가와는 PTH·표본 구성이 달라 동일 장면 비교가 아니다.
- 15개 trace 모두900frame·CPU float64/GPU·JSON 건수 일치·float32/64 ever 차이0·finite·PTH 해시 보존을 확인했다. 각 행동의 세 seed 초기 상태가 서로 다름을 확인했고 본학습은 유지했다. [리포트·명령](output_etc/check/stage1_push_38000_single512_seeds012/report.md), [집계](output_etc/check/stage1_push_38000_single512_seeds012/summary.json), [검증](output_etc/check/stage1_push_38000_single512_seeds012/verification.json).

### Stage1 PUSH scratch33000 소규모 단독 재평가

- 최신 `Stage1Push_00033000.pth`를 이전과 같은5행동·각1명·32환경·seed0·900step·loco100%로 평가했다. 점/영역 성공은 AT3/3, ON_TOP9/10, SIT27/32, CLIMB0/0, PUSH25/25(각32시도), macro5는40/43.75%다.26500 대비 AT +3/+3, ON_TOP +8/+7, SIT -1/+2, PUSH +2/+2다. 낙상은 AT/ON_TOP/SIT 각1명이며512환경×3seed 정식 평가가 아니다.
- 5행동의 초기 장면10개 tensor 필드가26500과 모두 bitwise 동일하다.5개 trace의900frame·CPU/GPU·JSON 건수 일치·float32/64 ever 차이0·finite·PTH 해시 보존을 확인했고 본학습은 유지했다. [리포트·명령](output_etc/check/stage1_push_33000_single32_seed0/report.md), [검증](output_etc/check/stage1_push_33000_single32_seed0/verification.json).

## 2026-11-01

### Stage1 PUSH scratch26500 소규모 단독 재평가

- 최신 `Stage1Push_00026500.pth`를 이전과 같은5행동·각1명·32환경·seed0·900step·loco100%로 평가했다. 점/영역 성공은 AT0/0, ON_TOP1/3, SIT28/30, CLIMB0/0, PUSH23/23(각32시도), macro5는32.5/35%다.24500 대비 ON_TOP 0/+2, SIT +11/+10, PUSH +1/+1이다. 낙상은 ON_TOP1명·SIT2명이며512환경×3seed 정식 평가가 아니다.
- 5행동의 초기 장면10개 tensor 필드가24500과 모두 bitwise 동일하다.5개 trace의900frame·CPU/GPU·JSON 건수 일치·float32/64 ever 차이0·finite·PTH 해시 보존을 확인했고 본학습은 유지했다. [리포트·명령](output_etc/check/stage1_push_26500_single32_seed0/report.md), [검증](output_etc/check/stage1_push_26500_single32_seed0/verification.json).

### Stage1 PUSH scratch24500 소규모 단독 재평가

- 최신 `Stage1Push_00024500.pth`를 이전과 같은5행동·각1명·32환경·seed0·900step·loco100%로 평가했다. 점/영역 성공은 AT0/0, ON_TOP1/1, SIT17/20, CLIMB0/0, PUSH22/22(각32시도), macro5는25/26.875%다.21000 대비 ON_TOP +1/+1, SIT +1/0, PUSH +5/+5다. 낙상은 ON_TOP3명·SIT2명·PUSH1명이며512환경×3seed 정식 평가가 아니다.
- 5행동의 초기 장면10개 tensor 필드가21000과 모두 bitwise 동일하다.5개 trace의900frame·CPU/GPU·JSON 건수 일치·float32/64 ever 차이0·finite·PTH 해시 보존을 확인했고 본학습은 유지했다. [리포트·명령](output_etc/check/stage1_push_24500_single32_seed0/report.md), [검증](output_etc/check/stage1_push_24500_single32_seed0/verification.json).

### Stage1 PUSH viewer 고정 과제 옵션 문서화

- 기본1환경 viewer에서 PUSH가 보이지 않는 경우를 확인했다. PUSH용 물체 슬롯은 생성 때 고정되므로 해당 슬롯이 없으면 reset으로 PUSH가 새로 생기지 않는다. 기존 `TASK_GRAPH=push` 옵션과 viewer 명령을 config/USAGE에 보완했다. 실행 중인 viewer의 최근 CSV에서 ON_TOP/운반 AT 관계를 확인했고 코드·실행 프로세스는 변경하지 않았다. 문서 diff·경로를 확인했다.

### Stage1 PUSH scratch21000 소규모 단독 평가

- 현재 scratch 학습의 최신 `Stage1Push_00021000.pth`를 이전과 같은5행동·각1명·32환경·seed0·900step·loco100%로 평가했다. 점/영역 성공은 AT0/0, ON_TOP0/0, SIT16/20, CLIMB0/0, PUSH17/17(각32시도)이며 macro5는20.625/23.125%다. 낙상은 SIT4명, 나머지0명이다. 이전12000의 전 행동0/32 대비 SIT·PUSH가 개선됐다. 외부21k finetune과 다른 PTH이며512환경×3seed 정식 평가가 아니다.
- 5행동의 초기 장면10개 tensor 필드가12000과 모두 bitwise 동일하다.5개 trace의900frame·CPU/GPU·JSON 건수 일치·float32/64 ever 차이0·finite·PTH 해시 보존을 확인했고 본학습은 유지했다. [리포트·명령](output_etc/check/stage1_push_21000_single32_seed0/report.md), [검증](output_etc/check/stage1_push_21000_single32_seed0/verification.json).

## 2026-10-31

### Stage1 PUSH12000 소규모 단독 재평가

- 평가 시작 시 최신 `Stage1Push_00012000.pth`를 이전과 같은5행동·각1명·32환경·seed0·900step·loco100%로 평가했다. 점·영역 모두 각0/32·macro5 0%로9000 대비 성공수 개선은 없었다. 낙상은 AT/ON_TOP 각1명, 나머지0명이다.512환경×3seed 정식 평가가 아니다.
- 5행동의 초기 장면10개 tensor 필드가9000과 모두 bitwise 동일하다.5개 trace의900frame·CPU/GPU·JSON 건수 일치·float32/64 ever 차이0·finite·PTH 해시 보존을 확인했고 본학습은 유지했다. [리포트·명령](output_etc/check/stage1_push_12000_single32_seed0/report.md), [검증](output_etc/check/stage1_push_12000_single32_seed0/verification.json).

### Stage1 PUSH9000 소규모 단독 재평가

- 최신 저장본 `Stage1Push_00009000.pth`를 이전과 같은5행동·각1명·32환경·seed0·900step·loco100%로 평가했다. 점·영역 모두 각0/32·macro5 0%로7000 대비 성공수 개선은 없었다. 낙상은 ON_TOP1명, 나머지0명이다.512환경×3seed 정식 평가가 아니다.
- 5개 초기 장면의10개 tensor 필드가7000과 모두 bitwise 동일하다.5개 trace의900frame·CPU/GPU·JSON 건수 일치·float32/64 ever 차이0·finite·PTH 해시 보존을 확인했다. 본학습은 유지했다. [리포트·명령](output_etc/check/stage1_push_9000_single32_seed0/report.md), [검증](output_etc/check/stage1_push_9000_single32_seed0/verification.json).

### Stage1 PUSH7000 소규모 단독 평가

- 사용자 요청대로 최신 저장본 `Stage1Push_00007000.pth`의5행동을 각각1명·32환경·seed0 한 번·900step·loco100%로 평가했다. AT/ON_TOP/SIT/CLIMB/PUSH 점·영역은 모두0/32이며 macro5도0%다. 낙상은 ON_TOP/PUSH 각1명, 나머지0명이다. 학습 중 frame 성공률과 다른 loco 단독 결과이며512환경×3seed 정식 평가가 아니다.
- 5개 trace 모두900frame·CPU/GPU·JSON 건수 일치·float32/64 ever 차이0·finite·PTH 해시 보존을 확인했다. 기존 본학습은 유지했다. 실행 명령·결과는 [리포트](output_etc/check/stage1_push_7000_single32_seed0/report.md), [검증](output_etc/check/stage1_push_7000_single32_seed0/verification.json)에 기록했다. 코드·YAML·기본 평가 명령은 변경하지 않았다.

## 2026-10-30

### PUSH 접촉과 목표 도달 지표 분리

- PUSH 포함 설정의 TensorBoard 관계 지표를 `edge/contact/*`와 `edge/push_at/*`로 추가하고 `edge/at/*`를 운반 AT 전용으로 바꿨다. 유효 PUSH 관계의 소유자로 AT를 구분해 edge 순서·alias와 무관하게 집계한다. `own_success`는 현재 성공 비율이며 접촉 보상 포화와 실제 접촉 성공은 구분한다. 보상·성공 조건·모델·실행 명령은 유지하며 기존 프로세스는 재시작하지 않았다. 변경은 다음 시작·재개부터 적용된다.
- CPU65개 통과·외부 파일 관련3개 skip. 혼합 과제/edge shuffle에서 contact·PUSH AT·운반 AT의 서로 다른 성공률,1/2/4명·비활성 관계·단일 그래프 배치, 진단 on/off 보상과 runtime 결과 bitwise 동일을 확인했다. Stage2 전이 검사1개는 로컬21k PTH 부재로 실패해 후속 실행에서 제외했다. GPU 학습과 속도 비교는 실행하지 않았다. [전체 검사](output_etc/check/push_metrics/tests.log), [사용 가능한 검사](output_etc/check/push_metrics/tests_available.log). 지표 상세·structure를 갱신했다.

### 새 checkout의 Stage1 PUSH 데이터 연결 복구

- tmux `base`의 `stage1_push`가 SIT `ref_motion.npy`의 `FileNotFoundError`로 종료된 원인은 Git에서 제외된 외부 데이터 링크 누락이었다. 이 checkout에 공유 `TokenHSI`의 기본 네 데이터셋 `motions` 링크와 Downloads의 PUSH 모션21개 링크를 복구했다. PUSH는 manifest 경로마다 bank SHA256과 일치하는 원본 파일을 연결했으며 외부 원본과 기존 학습 프로세스는 변경하지 않았다. USAGE에 실제 공유 경로와 새 checkout의 링크 복구 주의점을 기록했다.
- 두 manifest의 고유 참조 파일121개 존재·PUSH21개 해시 일치를 확인했다.2048환경 CHECK에서 모션 로딩·첫 물리 step을 통과하고 RSI5360 profile screening 진행을 확인했다. 전체 캐시 생성/PPO 완료 전 이번 CHECK만 SIGINT로 종료했으므로 학습 완료 검증은 아니다. [파일 검증](output_etc/check/stage1_push_data_links/verification.json), [확인 로그](output_etc/check/stage1_push_data_links/train.log).

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
