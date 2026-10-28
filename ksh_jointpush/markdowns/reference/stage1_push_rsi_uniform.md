# PUSH RSI 균등 거리

`stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform`은 raw cuboid에서 RSI 시작 거리와 커리큘럼만 바꾼 실험이다. 실행 명령은 [config](../config.md)에 있다.

아래 배경과 기존 검증 기록은 제공된 문서의 수치다. 현재 checkout의 구현 검증은 마지막 절에 별도로 기록한다.

## 배경

raw cuboid 35000을 256환경으로 평가했다(push agent 512명, loco 시작과 RSI 시작이 반반).

| 행동 | loco 시작 | RSI 시작 |
| --- | --- | --- |
| 평균 행동(기본 test) | 0.24 | 0.90 |
| 행동 noise 포함 | 0.70 | 0.99 |

- 평균 행동으로 평가했을 때 실패의 85%는 상자를 0.5m쯤 밀고, 남은 시간 동안 손을 면에 댄 채 멈춰 있다. 상자→목표 5m 이상에서 loco 시작 성공률은 0.00~0.12다.
- 기존 RSI 시작은 bank frame(11개, clip 7개)의 `box_progress` p(0.47~0.77)만큼 상자를 목표 쪽으로 옮긴다. 그래서 남은 거리 (1−p)·D가 0.47~5.3m로 묶였다(중앙값 2.05m, 4m 초과는 3%). 멀리서 미는 도중의 상태가 학습에 거의 없었다.
- `late_probability`는 후보가 여러 개인 clip 2개에서만 작용하므로, 이 범위를 바꾸지 못한다.

## 변경

1. **`rsi.uniform_goal_distance: true`**: `humanoid_ma_stage1_push._reset_push`에서 RSI 상자를 p만큼 옮기지 않는다. `rsi.goal_distance_max: 12.0`에 따라 RSI 남은 거리는 U(2,12)m이며, loco는 기존 U(2,10)m를 유지한다. 사람 자세·속도는 기존과 같은 mocap frame이다. flag 기본값은 false이며, 거리 상한 옵션이 없는 이전 checkpoint는 기존 상한을 사용한다.
2. **커리큘럼 제거**: `navigation.curriculum`을 마지막 단계 하나로 둔다(epoch 0부터 loco 목표 최대 10m, 사람 시작 반경 1.65~6m, 360°). 장면 내 상자 간격은 RSI 상한12m까지 고려한다.
3. **초기 모델**: 현재 기본 새 학습은 `STAGE1_CHECKPOINT=output/stage1/Stage1V2Fixed_00078000.pth`에서 전이하고 optimizer/epoch을 새로 시작한다. 기존 raw cuboid에서 시작하려면 `PUSH_CHECKPOINT`를 지정한다. 이 전이는 `uniform_goal_distance`·`goal_distance_max` 차이를 허용한다(navigation은 원래 무시한다). 같은 teacher 파일의 상대/절대경로는 정규화해 비교하고, 다른 teacher는 계속 거부한다. 일반 resume의 설정 계약은 유지한다.

보상, bank, 상자 물성, 기존 과제, RSI 비율(0.5), PUSH episode 1200 step은 cuboid와 같다.

## 파일

- 환경·학습 설정: `tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform.yaml`. 학습 yaml은 cuboid와 같다.
- 실행 스크립트: `tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform_{train,test,vnc}.sh`. 공용 `stage1_v2_fixed_push_{train,test}.sh`와 `train_vnc.sh`의 실험 목록에도 추가했다.
- 테스트: `tokenhsi/tests/test_push_cuboid.py::test_rsi_uniform_only_changes_rsi_distance_and_final_stage_and_warm_starts_from_cuboid`

## 제공된 문서의 기존 검증 기록

GPU5에서 2048환경 reset의 상자→목표 거리를 쟀다(push agent 2048명).

| 설정 | RSI 시작 | loco 시작 |
| --- | --- | --- |
| cuboid, 마지막 단계 | 0.47~5.08m, 중앙값 2.26 | 2.01~9.95m, 중앙값 5.89 |
| rsi_uniform, epoch 0 | 2.00~10.00m, 중앙값 6.25, 1m 구간마다 11~14% | 2.01~9.95m, 중앙값 5.89 |

- CPU: push 테스트 41개 중 40개가 통과했다. 실패한 1개는 서버에 없는 `Stage1V2Fixed_00060000.pth`를 읽는 기존 테스트다.
- `CHECK=1` 학습: 35000 이어받기(fresh optimizer), epoch 2 저장, resume 계약, weight finite를 확인했다. 이 확인은 커리큘럼 제거 전에 했다. 커리큘럼 제거 뒤에는 이어받기용 config 생성만 다시 확인했다.
- 자료: 분석은 `output_etc/check/stage1_v2_fixed_push_360_raw_cuboid/push_diag/{e20000,e35000,e35000_stoch}`, 검증은 `output_etc/check/stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform/`에 있다.

## 주의

- 기존 RSI 시작의 48%를 차지하던 "목표 2m 이내에서 시작"이 사라진다. 목표 앞에서 멈추는 동작은 미는 도중에 지나가는 구간으로만 학습된다.
- 새 checkpoint의 평가도 RSI 시작을 2~12m에서 뽑는다. cuboid와 비교할 때는 loco 시작 성공률로 비교한다. 이전 2~10m checkpoint의 일반 resume는 저장된 기존 범위를 유지하므로 새 범위 적용에는 `PUSH_CHECKPOINT` 전이를 사용한다.
- `TEACHER_KL_COEF=0`이면 teacher를 만들지 않고, warm-start/resume의 teacher 전용 로드와 KL 관측 처리·forward도 생략한다. 기본 계수0.001은 유지한다.
- 학습 로그의 PUSH 성공률은 noise가 섞인 rollout에서 매 step 성공 비율이다. 성공 여부는 평균 행동 평가로 판단한다.

## 초기 구현 검증 — 2026-10-24, RSI 2~10m

- CPU 관련 테스트 60개와 새 wrapper·공용 스크립트의 shell 구문 검사를 통과했다.
- GPU0·2048환경의 최초 reset에서 RSI 시작 1025명은 2.005~9.996m(중앙값 6.116m), loco 시작 1023명은 2.004~9.999m였다. RSI의 1m 구간별 비중은 10.8~13.9%였다. epoch 0부터 목표 최대10m·반경6m·360°를 확인했다.
- 실제 cuboid epoch24500의 모델·정책/AMP 정규화 일치, optimizer 상태0개에서 시작, epoch2 저장·epoch4 재개와 optimizer step96→192를 확인했다. 두 실행의 모델/optimizer tensor와 각각271개 scalar가 finite였다.
- 저장본의 전용 test wrapper로 1환경·8step 화면 없는 추론을 통과했다. 시각화·장기 학습 성능은 평가하지 않았다. 기존 cuboid 본학습·설정·bank·보상·RSI 모션은 유지했다.
- 현재 검증 자료: [validation.json](../../output_etc/check/push_rsi_uniform_20261024/validation.json). 새 variant의 본학습은 시작하지 않았다.

## KL0·RSI 12m 변경 검증 — 2026-10-24

- CPU66개 통과. KL0의 resume·warm-start는 teacher 전용 파일을 읽지 않고, 최초 fixed 전이는 학생 가중치 로드를 유지함을 검사했다. 양수 KL의 teacher 생성·동결·기존 과제 전용 loss도 확인했다.
- GPU0·2048환경에서 KL0의 전이/재개 모두 teacher 생성·복제·전용 로드0회였다. RSI1025명은2.006~11.995m(10m 초과216명), loco1023명은2.004~9.999m였다.
- epoch2 저장→epoch4 재개·optimizer step96→192·각271 scalar와 모델/optimizer finite, 전용 wrapper의1환경8step 추론을 확인했다. 기존 본학습은 유지하고 새 본학습은 시작하지 않았다. [검증](../../output_etc/check/push_teacher0_rsi12_20261024/validation.json).

## 현재 본학습 — 2026-10-24

사용자 요청으로 기존 cuboid를 중단하고 fixed78000에서 KL0·새 optimizer/epoch의 본학습을 시작했다. GPU0·2048환경의 사용자 서비스 `tokenhsi-push-rsi-uniform-kl0-20261024-194136.service`로 실행하며, `Linger=yes`로 로그아웃 후에도 유지된다. epoch1/frame0~epoch17·271개 scalar finite·KL loss0을 확인했다. [검증](../../output_etc/check/push_uniform_kl0_fresh_20261024_194136/validation.json), 현재 상태·로그 명령은 [config](../config.md)에 있다.

20:05 CUDA illegal memory access로 종료되어20:16에 같은 서비스를 재시작했다. 저장본이 없어 fixed78000에서 다시 시작했으며, 새 실행의 epoch1~13·271개 scalar finite·KL loss0을 확인했다. [재시작 검증](../../output_etc/check/push_uniform_kl0_restart_20261024_201610/validation.json).
