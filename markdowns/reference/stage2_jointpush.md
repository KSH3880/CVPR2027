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
