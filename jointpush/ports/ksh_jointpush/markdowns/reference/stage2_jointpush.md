# Stage2 jointpush

실행 명령은 [config](../config.md), 검증 이력은 [changelog](../../changelog.md)에 둔다.

## 설정

- 현재 `stage1_jointpush`의 과제 보상·PUSH profiles·질량20kg·본체 마찰0.4·밑판 없음·기존 플랫폼 물성을 사용한다. teacher/KL은 없다.
- 4명·1024환경·1200step. 사람 순서를 섞고 남은 인원이 2명 이상이면 coupled 0.8/단독0.2를 순차 선택한다. coupled는 2~남은 인원 균등, 단독은 SIT/CLIMB/AT/ONTOP/PUSH=10/20/25/20/25를 새로 뽑는다. 마지막 1명은 단독이다.
- 학습·평가 모두 loco100%. 기존 과제는 loco, PUSH는 push_loco 참조 자세·시점을 사용하며 추가 yaw 회전은 없다. AMP는 기존 4계열 expert mapping을 유지한다. AMP family prior는 순차 그룹 sampler의 사람 기준 기대 분포를 따른다(공동 사람 비율 약82.27%). PUSH family 내부 loco/push는 50/50이다.
- 환경 생성 시 기본 PUSH profile 4개를 뽑는다. coupled는 X만 인원수 배, Y/Z 유지, 질량20×인원수 kg다. 크기는 reset 동안 고정이다.

## 물리 후보와 관측

사용자가 승인한 후보 bank 방식이다. 환경마다 carry4·SIT4·CLIMB4·SUPPORT4·1인PUSH4·2인PUSH2·3인PUSH1·4인PUSH1, 총24개 물리 actor를 생성한다. 그룹 조합에 따라 실제 과제 상자1~4개와 SUPPORT4개를 배치하고 나머지16~19개 후보는 장면 밖에 격리한다. SUPPORT 중 ON_TOP 목적지로 할당되지 않은 것이 distractor다. 격리 후보는 distractor가 아니며 관측에 포함하지 않는다.

object 관측은 항상 담당4+SUPPORT4=8토큰이다. coupled는 하나의 실제 상자를 로컬X 방향 동일 폭 구역으로 나누고 멤버에게 무작위 일대일 배정한다. 에피소드 중 배정은 유지한다. 각 토큰은 현재 강체 pose로 구역 중심을 변환하고 속도에 `omega × offset`을 더한다. GTA도 이 pose를 사용한다. 별도 group ID를 coordination에 전달한다.

## 배치와 보상

사람을 먼저 초기화한다. 이후 상자 후보를 32개씩 병렬 제안한다. 사람 배치는 재시도 중 유지한다. 실패한 환경만 후보를 다시 뽑으며, 후보는 수치 배열이므로 물리 actor 32개를 생성하지 않는다.

- 모든 PUSH: 기준 사람→실제 상자 중심 반경2~10m, 상자→목표 반경2~10m, 상자 yaw360°. 목표는 reset 뒤 고정한다.
- coupled: 멤버 전원이 실제 상자 중심에서2~10m, 로컬−Y 뒤쪽180° 안에 있어야 한다. 목표는 로컬+Y±45°. 실제 상자 폭을 사용하는 이동 직사각형 검사로 다른 과제 상자·사람·목표 경로와의 간섭을 거절한다.
- 단독PUSH: 목표 방향360°와 기존 Stage1 PUSH의 경로·후방1.5m·경로 간격3.5m 조건을 유지한다. 장면의 나머지 사람·상자에도 간섭 검사를 적용한다.
- 기존 네 단독 과제: 기존 사람 기준 상자/목표 sampler·높이/플랫폼 규칙을 사용한다. 미할당 SUPPORT는 기존 반경4.5m 영역에서 배치하고 PUSH 경로와 겹치면 다시 뽑는다.
- CONTACT는 자기 구역의 실제 뒤쪽 외부 면에 대해 계산한다. 가상 절단면은 접촉 면이 아니다. AT 보상은 구역 중심 대신 실제 공유 상자 중심과 공통 목표를 사용한다.
- coupled AT의 state/progress/success에는 현재 전원 양손18cm 접촉 gate를 적용한다. 실제 상자 전체 뒤쪽 면·뒤쪽 위치·목표 방향 정렬을 검사한다. CONTACT 보상은 gate로 막지 않으며, 단독에는 gate를 적용하지 않는다. AT 성공 시 CONTACT 포화는 Stage1과 같다.

## 전이·제로샷

Stage1 PUSH의165개 모델 항목과 scene/AMP 정규화 통계를 strict 전이한다. actor encoder와 scene RMS를 동결하고 coordination·action head·critic·discriminator를 학습한다. coordination은 actor/critic 각4층·64차원·2head·FF128이며 identity 초기화한다. 새 optimizer/epoch으로 시작한다.

학습률2e-5 constant, gamma0.99, tau0.95, horizon32, minibatch16384(장면4096), mini_epochs6, critic_coef5, entropy0, clip0.2, bounds10, gradient clipping 없음. AMP buffer 각200000, batch512/minibatch4096, discriminator coef5·gradient penalty5·logit0.01·weight decay0.0001·reward scale2. task/AMP=0.5/0.5, 저장500epoch이다.

test/VNC에 Stage1 PUSH `.pth`를 직접 주면 coordination을 identity로 초기화하고 추가 학습 없이 실행한다. 기존 네 과제만 가진 원본 Stage1 checkpoint는 PUSH network와 shape가 다를 수 있으므로 이 전이 경로의 대상이 아니다. Stage2 checkpoint는 저장된 Stage2 계약을 확인하고 그대로 복원한다. 학습 재개는 `RESUME_CHECKPOINT`를 사용한다.

## 검증 범위

GPU5·1024환경 reset/정책 추론과1환경 VNC를 검사한다. `check_stage2_jointpush.py`의 학습 경로 검사는 `MAAgent.train`을 검사 함수로 교체하여 optimizer step을 실행하지 않는다. PPO 학습 속도·장기 안정성·공동 과제 성공률 검증과는 구분한다. CUDA allocator 수치는 PyTorch 할당량이며 PhysX·드라이버 메모리 전체가 아니다. 격리 후보가 없는 동일 장면과의 비용 비교는 수행하지 않았다.

## PUSH 목표 표시

jointpush viewer의 PUSH 목표는 기존 중심점에 지름20cm 구형 테두리를 더한다. 물리 marker actor의 크기는 바꾸지 않는다. 목표 XY에 반경35cm 바닥 링과 목표 높이까지 세로선을 함께 표시한다. 단독은 담당 사람 색, coupled는 노란색이며 공통 목표 슬롯당 하나만 표시한다. 일반 carry AT 마커는 기존 지름6cm·사람 색을 유지한다. `stage2_jointpush_rsi`도 같은 표시를 사용한다. 정책 관측·보상·성공 거리 설정은 변경하지 않는다.

## 목표 도착 시 gate 면제 설정

`stage2_jointpush_goal_saturation`과 `stage2_jointpush_rsi_goal_saturation`은 각각 일반/RSI 원본에서 `stage2JointPush.goal_saturation: true`만 추가한 보상 비교 설정이다. 과제 샘플링·초기화·PPO는 각 원본과 같고, RSI 캐시는 기존 경로를 재사용한다. 실행 config·스크립트·출력 디렉터리는 분리한다.

공유 상자 중심이 목표에서 3D 거리20cm 이내이고 높이 오차1mm 이내이면 전원 접촉 gate를 면제한다. 이때 AT 성공과 짝지어진 CONTACT의 state/progress/success가 모두 포화되어 사람별 과제 보상은1.2다(패널티·task/AMP 혼합 전). 목표 밖에서는 기존 전원 접촉 gate를 적용한다. 성공 이력을 영구 유지하지 않으며 상자가 목표에서 벗어나면 다시 gate 조건을 검사한다. 진단 `group_gate`는 도착 면제를 포함한 유효 gate다. 기존 설정은 옵션이 없으므로 기존 동작을 유지한다.

기본 학습은 기존과 동일하게 Stage1 PUSH checkpoint에서 시작한다. 새 옵션은 Stage2 checkpoint 계약에도 포함되므로 기존 일반/RSI Stage2 checkpoint를 새 설정으로 직접 resume/test하면 계약 불일치로 거절된다. 새 설정에서 생성된 checkpoint로 재개·평가한다.

## 내부 action adapter 비교 설정

`stage2_jointpush_goal_saturation_adapter`는 일반 loco 기반 목표 포화 설정의 actor 학습 방식을 바꾼다. Stage1 actor encoder와 기존 action head 전체(64→1024→512→32)를 동결하고, 사람 간 attention4층 및 앞의 두 Linear에 병렬로 더하는 adapter(64→1024, 1024→512)를 학습한다. 각 adapter는 bias를 포함한 Linear 하나이며 기존 Linear 출력과 합한 뒤 ReLU를 적용한다. 최종32D 출력에는 adapter를 추가하지 않는다.

Adapter weight/bias는0, coordination은 identity로 초기화하여 전이 직후 같은 관측에 대한 Stage1 action mean을 유지한다. 동결 head를 통과하는 gradient는 차단하지 않아 앞단 attention과 adapter가 학습된다. Critic·critic coordination·AMP discriminator 및 scene/AMP 정규화 처리는 기존 Stage2와 같다. 단독·공동 모두 같은 adapter를 사용하므로 학습 후 단독 행동 보존까지 보장하지는 않는다.

환경의 `stage2JointPush.action_adapter: tokenhsi_internal`과 학습 network의 `use_internal_adaptation: true`를 함께 사용한다. 기존 설정은 adapter를 생성하지 않는다. 보상·샘플링·loco 초기화·PPO는 `stage2_jointpush_goal_saturation`과 동일하다. checkpoint 계약 및 모델 구조가 달라 기존 Stage2 checkpoint를 직접 resume하지 않으며, Stage1 PUSH에서 새로 전이하거나 이 adapter 설정의 checkpoint를 재개한다. 기본 wrapper 전이 경로는 기존10500이고, 문서의 비교 실행 명령은 사용자가 실행 중인 기준에 맞춰14000을 명시한다.
