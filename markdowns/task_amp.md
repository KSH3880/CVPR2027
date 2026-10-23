# 현재 할당 task에 맞는 AMP

2026-10-02. ksh PID 1099002의 unified task AMP를 읽기 전용으로 확인하고 우리 레포의 기존 경로를 재사용했다. 수정은 우리 레포의 새 학습 설정·테스트·문서에만 적용했다.

## 적용 상태

| 실행/설정 | task AMP |
| --- | --- |
| ksh PID 1099002 unified | 기존부터 활성 |
| 우리 PID 1184210 full AMP 설정 | 기존부터 활성 |
| 새 `approach_scenario_stage1_size_progress.yaml` | 이번에 활성화 |
| 기존 shared_random_binding 기본 YAML | 기존 설정 유지 |

PID는 확인 당시 값이다. 실행 중인 학습·RSI 검사는 수정하지 않았고 새 학습도 시작하지 않았다.

## 선택 기준

RSI 초기 동작이 아니라 **각 agent에 할당된 graph task family**를 따른다.

| 할당 과제 | AMP family | 조건부 전문가 분포 |
| --- | --- | --- |
| HOLDING / HOLDING_AT / HOLDING_ON_TOP | carry | loco 1/3, OMOMO 1/3, pickUp 1/6, putDown 1/6 |
| SIT | sit | loco 1/3, sit 2/3 |
| CLIMB | climb | loco / climb / climbNoRSI 각각 1/3 |

CLIMB이 loco RSI로 시작해도 climb family를 유지한다. 3개 family를 조건으로 쓰는 단일 판별기이며 AT/ON_TOP을 별도 판별기로 나누거나 성공 단계에 따라 전문가를 전환하지 않는다. `carryWith` 전용 전문가 비중은 원본 unified처럼 0이며 RSI의 carryWith 사용과 별개다.

## 연결

- `humanoid_ma_carry.py::_amp_family`: graph에서 각 agent의 family 결정.
- `_fetch_conditioned_amp_demo`: family 분포의 실제 참조 모션 생성.
- live 관측 및 RSI 참조 이력에 task one-hot 3차원 추가. 부분 reset도 해당 task 사용.
- `MAAgent.train_epoch`: rollout 행마다 같은 family의 expert/replay 선택. replay warm-up에 family가 없을 때만 같은 family의 live rollout 사용.
- `_preproc_amp_obs`: 모션 특성은 정규화하고 one-hot 라벨은 보존.

AMP 입력은 프레임당 129+3=132차원, 10프레임 총 1320차원이다. 정책 입력 크기는 변경하지 않는다.

## 새 설정과 검증

새 size_progress YAML의 `ampTaskConditioning: true`, `ampConditionedResume: true`로 기존 full AMP 경로를 활성화했다. 두 번째 키는 random-binding에서 conditioned AMP를 허용하는 계약 옵션이며 자동 resume을 뜻하지 않는다. `skillDiscProb`도 원본 unified의 명목 분포로 정리했다. 실제 conditioned 전문가 생성은 `EXPERT_PROBS`를 사용한다.

RSI·박스 범위는 유지했다. 실행 명령은 [size_progress.md](size_progress.md). 기존 checkpoint와 새 progress의 보상 계약 차이를 우회하지 않는다.

CPU 30개 테스트 통과: 5개 graph 과제의 family, 실제 expert/replay marker 매칭, 명목 분포, 라벨 정규화 보존, 기존 unified/shared random binding/progress 회귀. 로그: `../runs/reference_rsi/task_amp_tests.log`. 새 설정의 GPU 학습 및 수렴 효과는 아직 측정하지 않았다.
