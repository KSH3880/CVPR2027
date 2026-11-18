# TokenHSI 원본과 stage1_v2: 학습·평가 비교

`stage1_v2`는 기존 `stage1`에서 거리/운반 RSI 정렬을 유지하고 크기·성공 조건·RSI 검사를 확장한 별도 실험이다. 아래는 논문 수치를 추측한 비교가 아니라 **연결된 TokenHSI 코드·데이터와 적용 YAML**을 기준으로 한다. 원본의 세부 식과 asset 목록은 [TokenHSI 원본 문서](tokenhsi_original_carry_sit_climb.md)에 있다.

- 적용 env: [stage1_v2.yaml](../../tokenhsi/data/cfg/multi_agent/stage1_v2.yaml)
- 적용 train: [stage1_v2.yaml](../../tokenhsi/data/cfg/train/rlg/stage1_v2.yaml)
- 크기/RSI: [size_rsi.py](../../tokenhsi/utils/size_rsi.py), [stage1_v2.py](../../tokenhsi/utils/stage1_v2.py), [size_rsi_cache.py](../../tokenhsi/env/tasks/multi_agent/size_rsi_cache.py)
- 성공 판정: [edge_interaction_reward.py](../../tokenhsi/env/tasks/multi_agent/edge_interaction_reward.py)
- 실행 명령: [config.md](../config.md#stage1-v2)

## 1. 이번에 적용한 범위

| 항목 | stage1_v2 |
| --- | --- |
| Climb 대상 | X/Y 각각 0.40~0.80m, Z 0.20~0.70m |
| ONTOP 받침·배경 | X/Y 각각 0.40~0.80m, Z 0.25~0.50m |
| Carry payload | X/Y/Z 각각 0.20~0.60m 유지 |
| Sit 대상 | X/Y 각각 0.25~0.80m, Z 0.35~0.50m 유지 |
| 크기 간격 | 축별 0.05m; 환경 생성 시 고정 |
| AT 성공 | 목표 중심까지 3D 거리 ≤0.20m **그리고** 높이 오차 ≤0.001m |
| Climb 성공 XY | 상판 각 축 양쪽 10%씩 제외, 내부 80% |
| Sit/ONTOP 성공 XY | 기존 내부 80% 유지 |
| RSI | 새 크기에 대한 별도 캐시, 관통·발 바닥 침범·물리 충격·낙상 검사, 실제 크기에 따른 phase |

`stage1`과 분리된 variant는 `stage1_v2`, output은 `output/stage1_v2`, run 접두사는 `Stage1V2`다. 기존 checkpoint와 직접 resume/eval을 혼용하면 검사에서 거부한다. actor/critic 구조는 stage1과 같고 각각 독립 파라미터를 가진다.

## 2. 학습 환경·샘플링

| 항목 | 원본 TokenHSI | stage1_v2 |
| --- | --- | --- |
| 사람/물체 | 1명, 담당 물체 중심 | 2명·4상자, 사람·배경 간섭 존재 |
| 과제 | TRAJ/SIT/CARRY/CLIMB = 10/30/30/30% | SIT/CLIMB/AT/ONTOP = 10/25/32.5/32.5%의 명목 주변분포 |
| 담당 물체 접근 거리 | SIT U(1,5m), Carry/Climb U(1,10m) | 같은 규칙, ONTOP도 U(1,10m) |
| 방향 | 원본 2D 정규 난수 벡터 정규화 | 같은 규칙 |
| Carry 목적지 XY | 환경 원점 기준 각 축 U(-4.5,4.5m), 사람 및 payload에서 최소 1m | AT 목표 및 ONTOP 받침에 동일 규칙 |
| payload→목적지 거리 | 별도 반경으로 뽑지 않음 | 동일. 사람→payload 반경과 혼동하지 않음 |
| putDown RSI 목적지 | 모션 마지막 상태의 위치 | 모션 마지막 XY 유지. ONTOP 받침은 그 XY에 바닥 배치 |
| 추가 배치 검사 | 단일 사람 장면의 원본 검사 | 다중 상자/사람 간섭 재추첨; 최종 수락 분포는 원본과 완전히 같지 않음 |
| 배경 | 우리와 같은 추가 배경 슬롯 없음 | 2개 받침 후보 슬롯 중 ONTOP이 사용하지 않는 슬롯이 배경 |

우리의 물체 크기는 환경 생성 때 고정되고, 매 reset의 과제는 해당 크기에 적격인 과제 중 조건부 확률로 선택한다. 크기만 바꾸고 과제 확률 계산을 예전 범위에 두지 않도록 새 범위를 양쪽 경로에 전달한다. 예를 들어 높이 0.65/0.70m의 담당 상자는 Carry payload로 선택되지 않는다.

배경 상자는 바닥·랜덤 yaw로 두며 기존 반경 4.5m 원판 내 면적 균등 배치 경로를 유지한다. AT 목적지/ONTOP 받침의 ±4.5m 정사각형 샘플링과 다르다. 배경도 관측 토큰·물리 충돌에는 참여하지만 직접적인 task edge 성공 보상은 없다.

## 3. 학습 물체·선반

| 대상 | 원본 TokenHSI | stage1_v2 |
| --- | --- | --- |
| Carry | 학습 정육면체 한 변 0.20~0.60m, 0.05m 간격 | 각 축 독립 0.20~0.60m의 직육면체 |
| Sit | train 의자 mesh 49개, 전체 bbox X 0.4091~0.6941/Y 0.4334~0.6430/Z 0.6902~1.0144m | XY 0.25~0.80m, Z 0.35~0.50m 상자 |
| Sit 앉는 높이 | 좌면 대표 높이 약 0.3764~0.5195m; 등받이 높이와 다름 | 상자 상판 0.35~0.50m |
| Climb Box | XY (0.4,0.4)/(0.5,0.5)/(0.6,0.6)m × Z 0.2/0.3/0.4/0.5/0.6/0.7m, 18개 | XY 각각 0.40~0.80m × Z 0.20~0.70m, 0.05m 간격 |
| Climb 가구 | Cabinet 10개·Table 10개 포함 총 38개; 일부 실제 상판 약 0.725m | 가구 mesh 없이 직육면체 |
| Sit/Climb 물리 | 고정 mesh | 동적 상자, 움직이거나 기울 수 있음 |
| Carry 출발 선반 | loco에서 50% 높이 추가, 밑면 U(0,1m), 윗면 1.2m 제한, 출발에 +5cm 여유 | AT/ONTOP payload에 같은 규칙 |
| Carry 도착 선반 | 출발과 독립 추첨, 같은 제한; 도착에는 +5cm 없음 | AT에 같은 규칙 |
| 선반 RSI 예외 | 참조 상태에 따른 예외 | 출발 선반은 loco만. AT 도착은 loco/pickUp/carryWith, putDown은 모션 마지막 XY·바닥 목표 |
| ONTOP 도착 | 별도 ONTOP 과제 없음 | XY 0.40~0.80/Z 0.25~0.50m 받침 상자, 바닥 배치·별도 도착 선반 없음 |

우리 선반은 GPU 재배치 시 충돌 형상 이동을 확인한 중력 비활성·고질량 dynamic 판이며 원본 fixed 판과 물리 구현이 완전히 같지는 않다. 비활성 판은 장면 밖으로 이동한다. Carry 최대 높이 0.6m와 받침 최대 높이 0.5m의 합은 1.1m다. Climb 상자 두 개는 적층 관계가 아니므로 두 높이를 더해서 금지하지 않는다.

## 4. RSI와 AMP: 같은 부분과 다른 부분

| 항목 | 원본 TokenHSI | stage1_v2 |
| --- | --- | --- |
| SIT 초기 skill | loco/sit 50/50 | 동일한 요청 비율 |
| CLIMB 초기 skill | loco/climb 50/50 | 동일한 요청 비율 |
| Carry 초기 skill | loco/pickUp/carryWith/putDown 50/10/30/10 | AT/ONTOP 모두 같은 요청 비율 |
| 시간 선택 | 원본 motion YAML의 RSI 구간을 사용하는 sampler | 실제 크기에서 유효한 프레임 캐시; 70%는 interaction phase의 마지막 30%, 나머지 30%는 전체 valid |
| 후반 후보 없음 | 원본 방식 | 전체 valid 프레임으로 대체 |
| 특정 skill 유효 프레임 없음 | 원본 방식 | 같은 task의 사용 가능한 skill 중 원래 가중치로 재추첨; 필요 시 loco |
| 높이 retarget | 원본 모션 경로 | 수행하지 않음. 사람 자세·속도·과거 모션 좌표를 임의 이동시키지 않음 |
| AMP | 원본 task/skill 구성과 discriminator 설정 | Carry/SIT/CLIMB family 65/10/25%, 해당 task family의 demo/replay를 행별 매칭 |

**요청 RSI 비율과 실제 실행 비율은 다를 수 있다.** 캐시 공백에 따른 fallback과 장면 간섭 재추첨을 기록한다. 높은 Climb 상판에 성공 자세가 없더라도, 물리적으로 유효한 접근/중간 프레임은 사용할 수 있다. 성공 자세만 골라 초기화하는 규칙은 아니다.

새 RSI 캐시는 `output/rsi_cache_stage1_v2/`를 사용한다. 설정·모션·인체 asset·구현·크기 범위가 해시에 포함된다. 기존 캐시를 덮어쓰거나 새 크기에서 검증 없이 재사용하지 않는다. 새로운 seed/환경 수에서 처음 보는 크기 조합은 추가 screening이 필요할 수 있다.

검사 항목:

1. 참조 body point가 상자 내부로 6cm 이상 깊게 들어가는지, 발 point가 바닥 아래 2cm보다 낮은지 검사한다. 정확한 mesh 간섭 검사는 아니며 보수적인 point 검사다.
2. 참조 DOF 목표로 0.1초 진행하며 root 속도 변화≤3m/s, 상자 속도≤5m/s, root/상자 이동≤0.3m, finite 조건을 검사한다.
3. 시뮬레이션 body 위치에서도 같은 관통·발 검사를 하고, 접촉과 몸 높이를 이용한 낙상 조건을 검사한다.
4. SIT/CLIMB phase는 원본 정적 object origin의 Z를 상판으로 추측하지 않고, 실제 생성 상자의 높이로 계산한다.
5. 최종 장면 배치에서는 다른 사람·배경 물체와의 간섭도 검사한다. 통과는 장시간 안정성이나 학습 수렴을 보장하지 않는다.

AMP 이력은 **실제로 선택된 skill의 motion ID/time**에서 만들고, family 라벨은 **현재 담당 task**를 따른다. 예를 들어 Climb의 초기 skill이 loco로 대체돼도 task family는 Climb이다. 정책 토큰 셔플은 모델 내부에서 pose·task endpoint와 함께 수행하고 출력은 사람 순서로 복원한다. PPO 행 셔플 후에도 rollout/demo/replay의 family가 행별로 같아야 한다. 원본 모션을 공간적으로 보정하지 않으므로 보정 자세와 보정되지 않은 AMP 과거 이력을 섞지 않는다.

## 5. 학습 보상

원본은 과제마다 접근·root/상자 목표 근접·운반/발 높이 등의 보상을 조합한다. 우리 쪽은 동일한 전체 보상을 복제하지 않고 관계별 state/progress/success를 각각 0.2로 합산한다. 본인 task 기여만 합산하고 동료 보상은 0이며, PPO task/AMP 가중치는 0.5/0.5다. 사람 간 CPA 충돌 보상, power·상자 속도 벌점도 있다.

AT의 dense state 식 `exp(-10*d²)`와 공통 satisfaction threshold 0.9는 유지한다. **AT 성공만** `d≤0.2m`에 해당하는 별도 임계와 높이 1mm를 사용한다. 따라서 HOLDING의 성공 조건이나 다른 행동의 state 곡선을 함께 완화하지 않는다. Climb XY inset은 5%→10%로 변경한다. 성공 시 보상 포화 및 paired placement 처리는 기존 방식을 따른다.

## 6. 평가 조건·성공 판정

| 항목 | 원본 TokenHSI | stage1_v2 기본 평가 |
| --- | --- | --- |
| 초기화 | 해당 과제의 locomotion 100% | loco 100%; 중간 interaction RSI로 시작하지 않음 |
| 사람→물체 거리 | 학습 loco와 동일한 SIT 1~5/나머지 1~10m | 동일 규칙; 다중 장면 거절 샘플링 차이는 유지 |
| Carry test 크기 | 한 변 0.22/0.27/0.32/0.37/0.42/0.47/0.52/0.57m 정육면체와 (0.3,0.3,0.4)m | 학습과 같은 축별 0.20~0.60m 격자; 별도 held-out 크기 split 없음 |
| Sit/Climb test | 의자 26개, 로컬 Climb 35개 test mesh | 학습과 같은 상자 크기 분포; 별도 test mesh 없음 |
| Carry/AT | 상자 중심→목표 3D 거리≤0.20m | 같은 거리 조건 **그리고** 목표 높이 오차≤0.001m |
| SIT | root→저장된 좌면 목표점+0.10m의 3D 거리≤0.20m | root XY가 상판 내부 80%, root Z가 상판+0.12m에서 ±0.07m |
| CLIMB | root→저장된 상판 목표점+0.94m의 3D 거리≤0.20m | root XY 내부 80%, root Z가 상판+0.94m에서 ±0.20m, 양발 평균 Z가 상판에서 ±0.07m |
| ONTOP | 없음 | payload 중심 XY가 받침 내부 80%, 회전 bbox 밑면–받침 상판 간격 절댓값≤0.001m |
| 시간 | Carry 20초, SIT/Climb 10초 | 기본 모든 과제 20초 |
| 집계 | 한 번이라도 성공한 episode 비율(ever), precision 별도 | 마지막 유효 step(final)·episode 중 한 번이라도 성공(ever), agent/scene 지표 |
| 종료 | 본인 낙상, 시간 제한; 평가 IET 비활성 | 사람 중 누구든 낙상하면 장면 종료, 시간 제한; 성공 자체로 즉시 종료하지 않음 |
| 반복/환경 수 | 공식 wrapper 512환경×3회 | test wrapper 기본 16환경×3회, 명령 인자로 별도 지정 |

내부 80%는 각 축 양쪽에서 전체 길이의 10%씩 제외한다는 뜻이다. **면적은 원 상판의 64%**다. bbox 상판/밑면의 월드 Z는 회전된 bbox의 수직 extent로 계산하고, XY 내부 영역은 yaw 좌표계로 계산한다. SIT/CLIMB는 root, ONTOP은 payload 중심의 XY를 검사하며 신체/물체 전체 footprint의 포함을 요구하지 않는다.

우리 학습 성공 보너스와 평가의 순간 성공 판정은 같은 함수를 사용한다. final/ever는 이 순간 판정의 집계 방식이다. 성공 판정에는 별도 유지 시간·실접촉·양발 각각의 XY 포함을 요구하지 않는다. Climb 발 높이는 **양발 평균**이다. 원본 Climb의 발 높이 항은 보상에는 있지만 원본 평가 성공에는 없다. 원본 Carry의 내려놓기 보상에도 높이 1mm 조건이 있지만 원본 평가 성공에는 없다.

## 7. 공정한 비교를 위해 구분할 결과

1. **공통 조건 비교:** 두 정책에 같은 물체·목표 정보·출발/도착 분포·시간·낙상 종료 조건을 제공하고 같은 성공식으로 평가한다. 원본 20cm 지표와 우리 plane 지표는 각각 별도 열로 제시할 수 있다.
2. **원본 환경 전이 평가:** 원본 test mesh와 1인 환경에 우리 정책을 옮겨 평가한다. 우리 입력에 원본 좌면 목표점이 없다는 차이와 물리 호환 처리를 명시한다. 우리 환경에서 20cm 수식만 적용한 결과와 구분한다.
3. **우리 확장 평가:** 2인·배경 물체·ONTOP에서 평가한다. ONTOP에는 원본 공식 수치가 없으므로 원본 과제 성능과 같은 열의 공식 비교치로 취급하지 않는다.

TokenHSI 논문의 Climb test 수 26과 현재 연결된 로컬 목록 35, 공식 학습 4096환경과 현재 서버 학습 2048환경도 구분한다. 동일 거리/RSI 비율만으로 동일 benchmark가 되지는 않는다.

## 8. 검증 기록

- 2026-10-14 stage1_v2 검증(원본 보관 레포의 `output_etc/stage1_v2_check_20261014/validation.json`): CPU70개, 2048환경 학습 epoch2·resume epoch3, 771 scalar finite·물리 reset 실패0. 확인 학습은 `output/stage1_v2_check/`와 `output/stage1_v2_resume_check/`이며 새 본학습은 실행하지 않았다.
- 새 캐시6,413 profile 중 SIT487·Climb715개는 모두 유효 프레임이 있다. SIT의 phase 공백은 0이고, Climb129개는 phase가 없어 전체 valid로 대체한다. CarryWith35/712개는 유효 프레임이 없어 skill fallback을 사용한다. 이 값은 seed42·2048환경에 배정된 크기의 결과이며 전체 Cartesian 크기 조합 전수 검사는 아니다.
- 실제 reset23회·참조 모션10,570개·AMP 과거95,130프레임·높은 Climb RSI521개의 연결을 검사했다. PPO 미니배치48개·786,432행에서 담당 task와 rollout/demo/replay family가 일치하고 AMP 정규화 후 one-hot 라벨이 유지되는 것을 확인했다. Actor/critic 토큰·edge/task 순열의 출력/gradient 검사는 CPU에서 통과했다.
- 표준 headless32환경32step, 별도13과제/초기 skill·416step·310reset·772거리 검사와 CPA·선반 재배치 접촉을 통과했다. 선반 검사는 착지 구간의 실제 접촉과 정지 높이/속도를 함께 확인한다. VNC wrapper는 실제 호출했으나 기존 NVML 불일치로 GPU 선택에서 종료돼 화면은 미검증이다. 짧은 실행 검증을 장기 학습 성능이나 모든 RSI의 장시간 안정성 검증으로 해석하지 않는다.

현재 실행 목록과 이번 정리의 검증 범위는 [config.md](../config.md)를 따른다. 위 수치는 원본 보관 레포의 과거 검증 기록이다.
