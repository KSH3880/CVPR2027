# 정식 평가 규약 (2026-10-29)

프로토콜은 `general`과 `stage1_single` 두 개다. **일반 평가는 학습 인원**, **1명 단독 평가는 Stage1 전용**이다. BEFORE의 solo는 다인 장면 안에서 선행 관계가 없는 역할을 뜻하며 1명 단독 평가와 다르다. viewer용 `*_test.sh`/`*_vnc.sh`의 진단값도 정식 평가표와 구분한다.

## 장면·초기화

| 항목 | 일반 (`general`) | Stage1 단독 (`stage1_single`) |
| --- | --- | --- |
| 정책 | Stage1·Stage1 PUSH 또는 세 BEFORE 모델 | Stage1 또는 Stage1 PUSH |
| 인원·물체 | checkpoint 옆 `relation_config.yaml`의 학습 인원·토큰 수 | 1명; AT/SIT/CLIMB 물체1개, ON_TOP payload+받침2개 |
| 기본 구성 | 현재 Stage1/Stage1 PUSH 2명·4물체, BEFORE 4명·8토큰 | 행동별 독립 실행, distractor 없음 |
| 과제 | 학습 `random_scenario` sampler | AT, ON_TOP, SIT, CLIMB 및 PUSH 모델의 PUSH 각각 지정 |
| 크기 | 저장된 학습 크기 범위·크기 조건부 과제 분포 | AT만 지정9종; 나머지는 저장된 학습 범위 |
| 배치·물리 | 학습 설정 유지 | 같은 학습 배치·물리 경로; 물체 수만 위처럼 축소 |
| 시작 | loco 100% | loco 100% |

일반 Stage1의 명목 과제 prior는 SIT .10, CLIMB .25, AT .325, ON_TOP .325다. 실제로는 먼저 크기를 뽑고 그 크기에서 가능한 과제의 조건부 분포를 뽑으므로 표본 비율은 매번 다르다. 인원별 그래프/물체 연결도 학습의 reset 경로를 사용한다. 단독은 네 과제를 균등한 개수로 독립 실행하므로 일반 평가의 행동 혼합과 같지 않다.

BEFORE는 학습 pair/solo·역할·크기 호환·공유 물체 alias·거리 분포를 유지한다. follower를 가능한 모든 행동 중 균등하게 바꾸는 조작은 아직 하지 않는다. 논리 토큰8개가 항상 물리 상자8개를 의미하는 것은 아니다. 선행 payload와 후행 받침을 공유하는 pair 수만큼 활성 물체가 줄며, 나머지 물체/distractor 처리도 학습과 같다.

단독 AT의 전체 크기(m): `.22³, .27³, .32³, .37³, .42³, .47³, .52³, .57³, (.30,.30,.40)`. 각 유형 최소1개를 포함하고 나머지503개는 균등 복원 추출 후 shuffle한다. 나머지 단독 범위는 학습 YAML의 `taskSizeRanges`에서 읽는다. 현재 ON_TOP payload 각축 .20–.60, 받침 XY .40–.80/Z .25–.50; SIT XY .25–.80/Z .35–.50; CLIMB XY .40–.80/Z .20–.70이며 각축 .05m 격자다.

loco 모션·시점·사람 방향, 출발 상자 방사 거리(SIT 1–5m, 나머지 1–10m), 목표 XY 샘플링과 충돌 재시도는 학습 배치 경로를 따른다. Stage1의 운반 출발 선반/AT 목표 높이도 유지한다. SIT/CLIMB 상자를 고정하는 예전 단독 규약과 달리 이번에는 **학습처럼 동적 상자**다. BEFORE의 ground/near50/mix226 배치도 각 학습 설정 그대로다. viewer에서는 별도로 RSI를 선택할 수 있지만 정식 평가에서는 거리·RSI·인원 override를 거부한다.

## 시간·낙상

- seed **0,1,2**를 독립 프로세스로 실행하고, seed당 **512환경의 첫 episode 1회**만 채점한다. 단독은 행동당1,536시도, 네 행동 합6,144시도다. 일반 Stage1은1,536장면·3,072사람 과제이며 행동별 분모는 샘플링 결과다.
- 최대 **정확히900 제어 step**, 제어 간격1/30초로30초다. 기존 `episodeLength-1` 경계를 학습·viewer까지 수정해 시간 종료도900번째 제어step으로 맞췄다. 이번 개정에서 학습 horizon도900으로 맞추며 학습 보상은 유지한다. viewer도Stage1 900step/Stage2 1200step이지만 개인 낙상 처리와 성공 집계는 정식 평가와 분리한다.
- deterministic 정책·저장 RMS를 고정한다. reset 직후 warmup/초기 상태는 성공 frame에서 제외하고 첫 action 이후부터 채점한다. 성공해도 조기 종료하지 않으며 ever 성공은 보존한다.
- 낙상 판정 자체는 학습의 접촉·높이·grace 기준을 사용한다. **한 명 낙상으로 다른 사람을 종료하지 않는다.** 낙상 frame까지 채점하고 그 사람은 이후 action을0으로 만들며 관절 drive mode/stiffness/damping을 해제해 수동 물리로 남긴다. 사람·상자는 삭제하거나 이동하지 않는다. 다른 사람은 계속 수행한다. 전원이 낙상하면 실행을 마칠 수 있다.

## 점·영역 판정

같은 rollout을 두 기준으로 채점한다. 학습 `own_success`나 BEFORE gate를 재사용하지 않는다. float32 simulator 상태를 float64로 변환하여 판정한다.

| 과제 | 검사점 | 목표점 |
| --- | --- | --- |
| AT | payload 중심 | 샘플링된 AT 목표점 |
| ON_TOP | payload 중심 | 현재 받침 중심 XY, 회전을 반영한 현재 상판 최고 Z + payload **직립 높이/2** |
| SIT | 골반/root | 현재 받침 중심 XY, 상판 최고 Z + .10m |
| CLIMB | 골반/root | 현재 받침 중심 XY, 상판 최고 Z + .94m |

점 기준: 목표점까지 **3D 거리 ≤.20m**. 영역 기준: 받침 yaw 로컬 좌표에서 **|x|≤.4Lx, |y|≤.4Ly**이고 **목표 높이 오차 ≤.20m**. AT는 두 기준이 같다. 내부80%는 각축 기준(면적64%)이며 몸/상자 전체가 아닌 검사점만 본다. 접촉·속도·손 놓기·발 높이·방향·유지시간을 성공 조건으로 추가하지 않는다. 작은 받침에서는 영역 성공률이 더 낮을 수도 있다.

BEFORE는 점·영역 각각 다음 세 지표를 저장한다.

1. **개인 ever**: 각자의 기하 조건을 한 번이라도 만족. follower는 선행자 성공 전에 만족해도 이 지표에 들어가며 BEFORE 완료율이라는 뜻은 아니다.
2. **둘 다 ever**: `ever(A) AND ever(B)`. 시간/순서와 무관하게 두 사람 모두 성공한 pair 비율이다.
3. **동시 성공 ever**: 같은 유효 frame에서 `A(t) AND B(t)`가 한 번이라도 참. 학습 gate의 동시 판정 의미를 비교하는 진단값이며 거리 임계값은 새 평가 기준을 쓴다.

둘 다/동시 성공의 분모는 **실제로 선행 관계가 있는 follower**만이다. solo/leader에는 해당 없음이다. follower 개인 성공도 같은 pair 분모로 별도 저장한다. 선행자가 낙상 전 얻은 ever는 둘 다 ever에 남지만 낙상 후 상태로 새로운 동시 성공을 얻지는 않는다. 엄격한 ‘A 이후에만 B’ 시간 순서를 강제하는 지표는 추가하지 않았다.

## 통계·산출물·실행

각 행동의 seed별 성공률3개의 **평균 ± 표본 표준편차(ddof=1)**를 %/%p로 보고한다. macro는 각 seed에서 네 행동 성공률을 먼저 평균낸 뒤 세 seed 평균/표준편차를 계산한다. 일반 평가의 전체 시도 수로 가중한 평균과 구분한다. solo/leader가 포함된 전체 개인 성공률과 follower 전용 pair 성공률도 분모를 명시한다.

정식 진입점은 `*_eval.sh`; Stage1 단독은 `stage1_single_eval.sh`다. 명령은 [config](../config.md)의 추론 옵션에 있다. `evaluate.py`가 독립 seed별 `run.py`를 실행한다. 작은 `--envs`나 일부 `--seeds`/`--tasks`는 검사 전용이며 `manifest.json`의 `formal_report=false`로 표시한다.

모든 결과는 `output_etc/check/` 아래에 둔다. 각 실행의 `initial.pt`는 초기 사람·관절·물체·크기·목표·논리 연결을, `trace.npz`는 매 frame root/물체 pose·채점 유효 mask·낙상 step·누적 성공을 보존한다. `result.json`은 seed별 지표, 상위 `summary.json`은 통계, `manifest.json`은 checkpoint 경로/SHA256·실행 조건이다. `tokenhsi/rescore.py <결과폴더>`로 simulator 없이 CPU float64 재채점과 GPU 결과 일치를 확인한다. 큰 trace는 재채점·재현 자료이므로 보고서 숫자만 남기고 지우지 않는다.

구현: [프로토콜](../../tokenhsi/utils/evaluation_protocol.py), [rollout/낙상 제어](../../tokenhsi/learning/multi_agent/formal_eval.py), [seed 실행/집계](../../tokenhsi/evaluate.py), [재채점](../../tokenhsi/rescore.py). 이전 Stage1 900step/Stage2 1200step·고정 상자 규약은 [2026-10-22 단독 문서](../legacy/evaluation_single_20261022.md)에 보존했다.


## PUSH 확장 및Stage1 900step/Stage2 1200step 개정

현재 전 실험의 학습·정식 평가·viewer 기본은Stage1 900step/Stage2 1200step이다. 이전1200step 결과는 당시 조건의 과거 결과로 보존한다. 기본 Stage1 단독은4행동, `stage1_push` 단독은5행동으로 각각512환경×3seed를 실행한다. PUSH 검사점은 payload 중심, 목표점은 reset 때 지정한 AT 목표이며 점/영역 모두3D거리20cm로 동일하다. 평가에 CONTACT·손 힘·유지시간을 추가하지 않는다. PUSH 크기는 학습167종이며 AT9종 override와 섞지 않는다.

Stage1 PUSH 일반 평가는 checkpoint 저장 분포·2명·4물체·배치를 복원한다. 새 scratch는 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%, 제공21k finetune은 PUSH50%다. 결과의 `macro4`는 기존네행동, `macro5`는 PUSH포함5행동이고 `macro`는 해당 모델의 전체 과제 평균이다. 점/영역 각각 seed별 macro를 먼저 만든 뒤 평균/표본std를 계산한다.
