# Stage1 PUSH adaptive

`stage1_push_adaptive`는 `stage1_push`의 보상·성공 조건·모션/RSI·물리·정책·PPO를 바탕으로 별도 scratch 학습한다. 기존 `stage1_push`/`stage1_push_schedule` 실행에는 적용하지 않는다. 실행 명령은 [config](../config.md), 재개 방법은 [USAGE](../USAGE.md)에 있다.

## 전환 규칙

| 항목 | 값 |
| --- | --- |
| 첫 평가까지 | 0~2,000 epoch, AT/ON_TOP/SIT/CLIMB/PUSH 각20% |
| 자동 평가 | 매2,000 epoch, 행동별1명·512환경·seed0·1회·900step·loco100% |
| 판단 지표 | 정식 단독 평가의 **영역(region) ever 성공률**. 집중 선택·90% 숙련·80% 망각 모두 동일한 영역 지표를 사용 |
| 집중 행동 | 50% |
| 다른 행동 | 각각 최소5%; 남는 비율은 아직 숙련되지 않은 다른 행동에 균등 분배 |
| 집중 구간 | 최소6,000 epoch, 최대12,000 epoch |
| 숙련 | 2회 연속90% 이상 |
| 망각 | 이미 숙련한 행동이 2회 연속80% 미만이면 숙련 해제·다음 집중 후보 우선 |
| 모두 숙련 | 즉시 각20%로 돌아가 계속 학습; 자동 성공/epoch 종료 없음 |

최초 집중 대상은 첫 평가에서 성공률이 가장 낮은 행동이다. 동점 순서는 CLIMB→PUSH→AT→ON_TOP→SIT이다. 이후 미숙련 행동 중 가장 오래 집중하지 않은 행동을 먼저 선택하고, 같은 경우 성공률·동점 순서를 적용한다. 확인된 망각 행동은 다음 구간 경계에서 우선하며 현재 구간을 중간에 끊지 않는다.

집중 행동이 일찍 숙련되어도 최소6,000 epoch까지 유지하고 다음 평가 경계에서 전환한다. 12,000 epoch 안에 숙련되지 않으면 다른 후보로 순환한다. 미숙련이 하나뿐이면 같은 행동의 집중을 이어간다. 모든 행동의 숙련이 확인된 경우에는 최소 집중 시간과 무관하게 균등 유지로 전환한다. 80~90% 구간은 기존 숙련 상태를 유지한다. 고정 검증 seed를 반복 사용하므로 자동 점수는 curriculum 판단용이며 수동512환경×seed0/1/2의 정식 보고와 구분한다.

예: 첫 CLIMB 집중이면 CLIMB50%, 나머지 각12.5%다. CLIMB 숙련 후 PUSH에 집중하면 PUSH50%, CLIMB5%, AT/ON_TOP/SIT 각15%다. 미숙련 행동이 집중 대상 하나뿐이면 다른 네 행동에 남는50%를 균등 분배하여 각각12.5%로 둔다. 모든 경우 합계100%·최소5%·집중50%를 유지한다.

## 실제 샘플링과 자산

PUSH 상자는 생성 때 크기·질량·마찰이 정해진다. 따라서 기존 고정 상자에 task prior만 바꾸는 방법으로는 PUSH 비율을 바꿀 수 없다. 새 실험은 목표 비율대로 사람 슬롯을 배정하고 각 슬롯의 행동별 원래 크기 분포에서 물리 자산을 만든다. 해당 집중 구간 동안 슬롯의 행동은 고정되고 배치·RSI·모션·정책 탐색은 매 episode 계속 샘플링한다. 전체 rollout의 실제 행동별 프레임 비율을 조절하며 PPO loss 가중치는 바꾸지 않는다. episode 길이가 다르므로 reset 횟수 비율은 프레임 비율과 다를 수 있다.

2명·2048환경의4096슬롯에서 정수 배정을 하며5%는 올림하여 최소205슬롯을 보장한다. 목표와 실제 비율의 오차는 약1슬롯 이내다. 학습은 기존 Stage1의2명·4물체 RSI 경로를 유지하며 두 사람 모두 ON_TOP을 수행할 수 있는 받침을 확보한다. viewer는 기존 N≥1·M≥N 지원과 ON_TOP 초과 재샘플링을 유지한다.

집중 비율이 변할 때만 해당 새 학습 프로세스를 정상 종료하고 supervisor가 같은 run·epoch/frame으로 재개한다. 정책·critic·discriminator·optimizer·모든 정규화 통계와 AMP replay를 저장한다. AMP expert demo는 새 family 비율로 다시 생성하고 family 내부 혼합·RSI 비율은 유지한다. 새 물리 환경에서 episode를 다시 시작하므로 경계의 진행 중 episode와 난수열까지 연속인 재개는 아니다. 비율이 그대로인 평가에서는 환경을 재생성하지 않는다.

자동 평가 중에는 본학습 rollout을 잠시 멈추고 독립 subprocess에서 다섯 행동을 순차 평가한다. 평가 완료 후 원래 구간을 계속하거나 새 구간으로 재개한다. 2,000 epoch는 학습 epoch 기준이며 평가 시간은 별도로 걸린다. GPU 지정은 wrapper의 `TOKENHSI_GPU`를 그대로 상속한다.

## 저장과 오류 처리

PTH의 `stage1_push_adaptive_state`에 평가 epoch·집중 행동/시작 epoch·확률·숙련/망각·연속 판정 횟수·최근 집중 기록·점수를 저장한다. 물리 자산을 만들기 전에 이 상태를 읽고, 계약 revision3과 설정을 확인한다. 기존 고정 비율/epoch 스케줄 PTH의 전이·resume은 허용하지 않는다. 본학습 epoch 제한은 해제되어 있으며 기존 학습 PTH를 초기 가중치로 쓰지 않는다.

일반 저장은500 epoch마다 유지하고 자동 평가 후 별도 `Stage1PushAdaptive_adaptive_<epoch>.pth`를 저장한다. 환경 재생성 전 저장에만 AMP replay를 추가한다. 일반 저장에서 수동 재개하면 기존 resume처럼 replay를 새로 채우며, 전환 PTH를 사용하면 저장된 replay도 복원한다. 이전 일반 PTH의 저장 시점 분포는 해당 PTH 내부 state로 복원하므로 최신 sidecar에 의존하지 않는다.

평가 snapshot·명령·로그·trace·manifest·decision은 `output_etc/check/stage1_push_adaptive/<run>/epoch_<epoch>/attempt_*/`에 둔다. 다섯 행동의 결과·시도 수·checkpoint 해시가 맞아야 상태를 갱신한다. 오류 시 새 학습은 임의로 판단하지 않고 실패를 알린다. 기존 다른 학습 프로세스를 중단하지 않는다. 중복 supervisor는 run별 lock으로 거부한다.

TensorBoard에는 `adaptive/eval_region_*`, `adaptive/probability_*`, `adaptive/mastered_*`, `adaptive/evaluation_epoch`와 기존 샘플링·RSI·보상 지표를 남긴다. `decision.json`의 `criterion`도 `region`으로 기록한다. 실제 프레임 분배는 `relation/sampling/asset_expected_*`, reset 분포는 기존 `relation/sampling/template_*`로 구분한다. 행동 이름 접미사는 `holding_at`, `holding_ontop`, `sit`, `climb`, `push`다.

2026-11-14 사용자 요청으로 기존 점 기준을 영역 기준으로 변경했다. 변경 전2k 자동 평가는 다섯 행동 모두 점·영역 성공수가0으로 같아서 그때의 집중 행동·확률·숙련 streak는 영역 기준으로 재계산해도 동일하다. 이전 `adaptive/eval_point_*` 기록은 당시 점 지표이며 영역 지표로 이름을 바꾸지 않는다. 실행 중인 Python에는 파일 수정만으로 적용되지 않아 다음 일반 저장본에서 재개해야 한다. 일반 PTH 재개 시 AMP replay는 위 설명대로 다시 채운다.
