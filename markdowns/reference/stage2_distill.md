# Stage2 distillation

`stage2_distill`은 고정된 enable/coupled teacher 두 개의 행동을 단일 통합 student에 모방시키는 실행 가능한 학습 설정이다. 환경·관측·관계·보상·물리 조건은 [stage2](stage2.md)와 같으며 기본4명·1024환경·8물체 토큰이다. M≥N의 예산 검사/재샘플링과5명 이상 추론 확장도 공유한다. [실행법](../USAGE.md#stage2-distillation).

`stage2_distill_900`은 에피소드 제한만900step인 별도 컨피그다. Student 초기화·동결·손실·20+180iteration 전환·replay·업데이트 수는 동일하다. 이900 버전의 teacher는 사용자 지정 enable21k(11월5일22:58 run)·coupled17k이며 기존1200step 학습본을 그대로 사용한다. Teacher 경로는 이 버전의 학습 YAML에서 각각 선택한다. 전용 train/test/VNC/eval wrapper를 사용하며 viewer/eval도 기본900step이다. GPU는2/6/7만 허용하고 기본6이다.

`stage2_distill_900_v4`는 V3의 환경·학습 설정을 유지하고 enable34k(`Stage2BeforeTotalAt1mm`)·coupled30k(`Stage2CoupledContact`) teacher를 사용한다. Student는 Stage1 PUSH26k에서 새로 전이하며 두 경로는 전용 학습 YAML에 지정한다. 출력과 train/test/VNC/eval wrapper는 V4 이름으로 분리한다. V4의 학습·test·VNC·eval은 GPU2/5/6/7을 허용하며 기본6이다.

`stage2_distill_900_v5`는 V4를 복사하고 **coupled teacher만 같은 run의38k**(`Stage2CoupledContact_00038000.pth`)로 변경한다. Enable/solo는 기존34k를 유지한다. Student는 V4를 이어 학습하는 것이 아니라 Stage1 PUSH26k에서 새로 전이하며 환경·900step·학습 설정·GPU 허용 범위는 V4와 같다. 전용 train/test/VNC/eval wrapper와 `output/stage2_distill_900_v5/`를 사용한다.

## 초기화와 정답

- Student는 Stage1 PUSH26k에서 전이하고 actor encoder와 입력 RMS를 고정한다. Actor의 coordination(관계 임베딩·fusion 포함)과 action head만 학습한다. Critic·AMP discriminator·sigma도 고정하며 PPO/AMP 손실은 사용하지 않는다.
- 1200step `stage2_distill`의 teacher는 enable13.5k/coupled25.5k, `stage2_distill_900`은 enable21k/coupled17k 정기 저장본을 기본으로 각각 원래 네트워크/전처리/RMS로 복원하고 고정한다. 학습 YAML의 `distillation.teacher_before`, `teacher_coupled`로 경로를 지정한다.
- 두 teacher는 동일한 물리 상태의 전체 장면을 관측한다. 공동 PUSH 그룹원은 coupled, enable 선행/후행 및 모든 solo는 enable teacher의 평균 행동을 정답으로 사용한다. Teacher/학생의 평균 행동을 clipping 이전 공간에서 비교한다.

## 수집과 모방

매 iteration32step을1024환경에서 진행한다. 현재 raw 관측과 사람별 teacher 정답을 scene 단위 replay에 저장한다. Teacher 실행 여부는 장면마다 하나의 Bernoulli로 선택하여 같은 그룹 안에서 teacher/student 제어가 갈리지 않게 한다. 선택한 평균 행동에 기존 clipping을 적용해 환경을 진행한다. 종료한 환경만 다음 step 전에 reset한다.

처음20iteration은 teacher로만 움직인다. 이후180iteration 동안 teacher 실행 확률을 선형으로 줄이며200iteration부터 student로만 움직인다. 어느 실행 정책을 선택하든 방문 상태의 teacher 정답을 수집한다. Student rollout에서 발생한 상태도 teacher에게 질의하는 DAgger 계열 학습이다.

Replay는 최대65,536개 scene의 관측/정답을 저장하는 ring buffer다. 매 iteration512scene minibatch로32회 평균 행동 MSE를 업데이트한다. 학습률3e-4, gradient norm 상한1이다. 입력 정규화는 전이 통계를 그대로 유지한다. 환경 reward는 진단용으로만 기록한다.

학습 YAML의 `params.config.distillation`에서 rollout/replay/minibatch/update 수와 teacher 전환 일정을 조정한다. `max_epochs` 기본10,000은 전체 목표 iteration이며 `MAX_ITERATIONS`도 같은 총 iteration 기준이다.

## 저장과 복원

기존 `relation_config.yaml`·관계/충돌/환경 계약·전이 보고서를 유지한다. `teachers.json`은 각 PTH의 경로/epoch/frame/SHA256와 담당 규칙을 기록한다. `distill_metrics.jsonl`과 TensorBoard의 `distill/*`에 MSE 전후·평균 학습 MSE·teacher 확률·실행 scene 수·관계별 정답 수·replay 크기·frame을 기록한다.

PTH에는 student와 정규화 통계, actor optimizer, epoch/frame, replay 데이터/크기/cursor, teacher 해시와 distillation 설정을 저장한다. 같은 teacher/설정일 때 optimizer/replay와 iteration을 복원한다. 환경은 새 장면으로 reset하며 물리 궤적의 완전한 연속 복원을 보장하는 기능은 아니다. 정기 저장은500iteration 간격이며 종료 시 최신 PTH를 저장한다.

추론은 student PTH 하나를 `stage2_distill_{test,vnc,eval}.sh`에 전달한다. Teacher 파일을 불러오지 않는다. Viewer/eval 기본 구역 배정은 ordered이며5명 이상은 공통5m 그룹 배치 경로를 사용한다. 기본 공간 평가 조건은1200step과 독립 float64 채점이고 CPU 재채점을 지원한다.
