# 기존 네 과제 + PUSH_AT

`stage1_v2_fixed_five_task`는 `stage1_v2_fixed`를, `stage2_shared_full_success_five_task`는 공유 Stage2의 정책·human coordination·전이 구조를 기반으로 한다. 기존 네 과제를 교체하지 않고 같은 정책에 PUSH_AT을 추가한다. 기존 네 과제/PUSH 전용 YAML과 실행 중인 학습은 그대로 둔다. 실행 명령은 [config](../config.md)를 따른다.

| 정책 task ID | 과제 | graph template ID | 관계 |
| --- | --- | --- | --- |
| 0 | SIT | 1 | SIT(9) |
| 1 | CLIMB | 2 | CLIMB(10) |
| 2 | CARRY_AT | 3 | HOLDING(6) + AT(7) |
| 3 | CARRY_ONTOP | 4 | HOLDING(6) + ON_TOP(8) |
| **4** | **PUSH_AT** | **5** | **PUSH_CONTACT(11) + AT(7)** |

HOLDING-only template 0은 종전과 같이 확률0이며 정책 과제가 아니다. 정책 packet은 그대로 `[valid, task, actor, payload, target]`이고, PUSH도 actor→payload/actor→target/payload→target 연결을 사용한다. GTA·tokenizer·Action MLP·actor/critic 구조와 입력 차원은 유지하고 task embedding은 새 설정에서 4→5행으로 늘린다. AMP는 PUSH family 조건을 추가해 입력만 확장하며 discriminator의 층 수와 hidden 폭은 유지한다. 기존 설정은 4행/3-family를 유지한다.

GPU6 물성을 적용하고 최종 목표1~5m·360° 안전 시작으로 바꾼 두 설정은 [별도 명세](five_task_gpu6_variants.md)를 따른다. 아래 일반 기본 설정과 구분한다.

## Stage1

- 비율은 SIT/CLIMB/CARRY_AT/CARRY_ONTOP/PUSH_AT = **8/20/26/26/20%**. 기존 네 과제의 상대 비율10/25/32.5/32.5%를 보존한다. 종전 크기 조건부 sampler를 유지하며, source asset 중 PUSH 크기인 슬롯은 PUSH를 수행한다. 크기별 실제 비율과 고정된 환경 표본에 따라 관측 비율은 달라질 수 있다.
- 2명·2048학습환경·4물체·634D를 유지한다. PUSH source만 X1.2~1.5m/5cm·Y1.8m·Z1.3m·20kg·상자마찰0.6이며 바닥마찰1.0이다. 다른 과제의 크기/밀도/선반/보상은 기존 경로를 따른다. 작업영역은 PUSH 경로를 위해 envSpacing24다.
- PUSH는 현재 목표 방향60° 이내 동적 공통 유한 면 CONTACT, state>0.9 및 각 손/상자0.5N 동시 반력, 기존 AT와 paired 포화를 사용한다. CARRY_AT의 HOLDING 보상에는 적용하지 않는다. 목표1~4m·loco시작1~5m·AT 기울기 제한 없음이다.
- 기존 네 과제의 RSI 가중치와 SizeRsiCache는 유지한다. PUSH는 검증된 Y1.8m/20kg 캐시를 재사용하며 loco50%/실제밀기RSI50%, 원본 DOF position/velocity·손중점≤1m를 유지한다. 첫 학습에서 기존 과제용 크기 캐시가 없으면 종전 방식으로 생성하므로 준비 시간이 걸린다.

## Stage2

- 4명·1024학습환경·12물체토큰·1420D, 기존 shared-support human coordination·Stage1 actor 동결/전이 구조를 유지한다. coupled_probability=0.8이며 **coupled 그룹은 PUSH_AT만** 수행한다. 그룹 크기는 원래 순차 sampler를 따른다. solo에서는 기존 네 과제와 PUSH_AT 모두 가능하다.
- 물리 상자는 일반 source/PUSH source/ON_TOP support 각4개다. 모든 상자를 실제 관측하며, 공동 PUSH는 그룹 리더의 PUSH 상자와 AT 목표를 공유한다. PUSH는 움직이는 전체 상자이므로 기존 SIT/CLIMB/ON_TOP 받침의 가상 구역 분할을 적용하지 않는다. 기존 Stage2 YAML의 공동 SIT/CLIMB/ON_TOP 경로는 유지한다.
- 원래 shared-full-success와 같이 **전 과제 loco 초기화**다. 그룹별 같은 방향으로 접근하며 사람 사이 측면 간격0.9m를 둔다. Stage1 선별 RSI의 공동 배치/공동 PD 검수를 했다는 의미가 아니다.
- Stage1 다섯 과제 checkpoint만 전이할 수 있다. 기존 네 과제 checkpoint의 embedding/AMP 차원을 자동 확장하지 않는다. `STAGE1_CHECKPOINT`에 새 Stage1 학습 저장본을 지정한다.

## AMP·checkpoint

원본 모션 특징은129D 그대로이며, AMP 조건 family는 carry/sit/climb에 PUSH를 추가해 **3→4차원 one-hot**이다. 따라서 **한 프레임132→133D·10프레임1320→1330D**, discriminator 첫 Linear 입력1320→1330이다. hidden1024/512와 출력·층 수는 유지한다. 기존 family ID0/1/2를 보존하고 PUSH는 ID3·`[0,0,0,1]`이다. 모션 수가 늘어서 특징 차원을 늘린 것이 아니라, 별도 PUSH 전문가를 조건화하기 위한 라벨 확장이다. PUSH expert는 loco⅓/PUSH⅔이며 carry expert에 PUSH 모션을 섞지 않는다. 현재/RSI history/demo/replay/정규화 모두 네 family 라벨을 보존하고 같은 family끼리 비교한다. 평가 결과는 다섯 과제별 성공률과 PUSH_CONTACT edge 성공을 별도로 기록한다.

새 다섯 과제 설정은 task embedding과 AMP 입력 크기가 늘므로 기존 네 과제/PUSH 단독 checkpoint와 직접 resume/test할 수 없다. checkpoint metadata에 four-family one-hot 계약을 기록해 이전 zero-code 버전도 거부한다. 새 Stage1→새 Stage2의 정책/critic/discriminator 전이만 차원 일치·taxonomy·fixed-platform 조건을 검증한 뒤 허용한다. 본학습과 장기 성능은 이번 코드 작업의 대상이 아니다.

## 검증

- CPU: `tokenhsi/tests/test_five_task.py`는 기존 ID/packet·순열·다섯 과제 sampling·PUSH gradient·AMP one-hot/매칭·checkpoint taxonomy·공동 source/goal·정책/discriminator forward·Stage1→Stage2 전이를 검사한다. 원래 Stage1/Stage2와 parameter key/shape를 비교해 **task embedding 두 행렬의 4→5행 및 discriminator 첫 입력1320→1330 외에는 동일**한지도 검사한다.
- `tokenhsi/tests/check_five_task.py`는 추후 reset/부분reset/관측/AMP demo·물리/PUSH RSI 확인용으로 준비한 코드다. 사용자 지시에 따라 **현재 최종 버전의 시뮬레이션·RSI 검사는 실행하지 않는다**. 이전 중간 버전의 GPU 결과를 최종 코드 검증으로 간주하지 않는다.
- 코드 확인 결과는 `output_etc/check/five_task*.log`, 범위·결과는 [changelog](../../changelog.md)에 기록한다.
