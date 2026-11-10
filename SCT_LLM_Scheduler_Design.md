# SCT + LLM Scheduler — Baseline 설계 정리

> **상태:** 논의된 설계안. 아래의 시간 예측 및 스케줄링은 구현·실험으로 검증 완료된 사실이 아니라, 현재 합의한 baseline의 동작 원칙이다.

## 1. 핵심 아이디어

**SCT + LLM Scheduler는 사전 학습된 SCT 정책을 추가 학습 없이 유지하고, LLM이 초기 장면 정보로 각 에이전트의 예상 소요 시간을 추정하여 출발 시점을 조정하는 비교군이다.**

- **SCT:** 에이전트별 관절 행동을 생성하는 기존 정책
- **LLM Scheduler:** 주어진 과제와 의존성 관계를 바꾸지 않고, 에이전트의 **언제 출발할지** 결정
- **외부 제어:** 정해진 시점까지 해당 에이전트를 freeze하고 출발 시 release
- **예측 입력:** 평가 시작 시점의 장면 관측, 에이전트·물체·목표 정보, 작업 할당 및 의존성
- **금지할 정보:** 실제 평가 rollout의 미래 완료 시각이나 성공 여부 등 사후 정보

### 실행 흐름

```text
Initial scene observation + tasks + dependencies
                     │
                     ▼
               LLM Scheduler
          (예상 소요 시간 추정)
                     │
                     ▼
             Start-time schedule
                     │
                     ▼
        Pretrained SCT + freeze/release
                     │
                     ▼
              Physical execution
```

**중요:** 이 baseline은 Method 3.4의 **LLM Planner**와 역할이 다르다.

| 구분 | LLM Planner (시나리오 실행) | SCT + LLM Scheduler (비교군) |
|---|---|---|
| 목적 | 어떤 에이전트가 어떤 작업을 맡고 어떤 의존성을 가질지 결정 | 이미 배정된 작업을 언제 시작할지 결정 |
| 출력 | Task / dependency assignments | 에이전트별 시작 지연 시간 |
| 저수준 제어 | 정책이 생성 | 변경 없이 기존 SCT 사용 |

## 2. Enabling — 선행 작업과 후행 준비의 타이밍 맞추기

**예:** AT → ONTOP

선행 에이전트의 AT가 완료될 예상 시간과 후행의 접근·준비 예상 시간을 LLM이 추정한다.

- 선행 AT 완료 예상: **5초**
- 후행 접근·준비 예상: **3초**
- 후행 시작 전 대기: **2초**

후행은 2초간 freeze한 후 출발하여, 선행 AT가 완료될 것으로 예상되는 5초 시점에 후행도 작업 위치에서 준비되도록 시도한다.

```text
시간(초)         0       1       2       3       4       5
선행 AT          ├──────────── AT 수행 ──────────────┤
후행 ONTOP       ├── 대기 ───┤── 접근·준비 ──────────┤
                                                 예상 동기화
```

개념적으로, 선행이 즉시 출발하고 후행의 추정 접근·준비 시간이 \(\hat{t}_{\mathrm{follower}}\)일 때:

\[
t_{\mathrm{wait}}=\max\left(0,\hat{t}_{\mathrm{AT}}-\hat{t}_{\mathrm{follower}}\right).
\]

### FSM과의 차이

- **SCT + FSM:** 선행 AT의 *실제 성공*을 확인한 후 후행을 해제한다.
- **SCT + LLM Scheduler:** 선행 AT의 *예상 완료 시간*에 맞춰 후행을 미리 출발시킨다.

LLM Scheduler의 의도는 선행 작업이 끝날 때까지 후행이 출발하지 않는 데서 발생하는 대기 시간을 줄이는 것이다. 단, 예상보다 후행이 일찍 도착한 경우 **선행 완료 전 최종 상호작용을 별도로 막을지**는 아직 미정이다.

## 3. Coupled — 공동 PUSH 참여자들의 동시 도착 유도

공동 PUSH 참여자들이 각자 할당받은 **PUSH Region**에 접근하는 데 걸릴 시간을 추정한다. 가장 오래 걸릴 것으로 예상되는 에이전트를 기준으로 다른 에이전트의 출발을 늦춘다.

### 예시

| 에이전트 | 예상 접근 시간 | 출발 전 대기 | 예상 도착 시간 |
|---|---:|---:|---:|
| A | 4초 | 4초 | 8초 |
| B | 6초 | 2초 | 8초 |
| C | 8초 | 0초 | 8초 |

예상 접근 시간이 \(\hat{t}_{\mathrm{approach},i}\)라면:

\[
t_{\mathrm{wait},i}=\max_j\hat{t}_{\mathrm{approach},j}-\hat{t}_{\mathrm{approach},i}.
\]

따라서 가장 멀리 있거나 오래 걸릴 것으로 예상되는 에이전트는 즉시 출발하고, 나머지는 시간 차이만큼 대기한다.

**주의:** 동시 출발/예상 동시 도착이 실제 동시 접촉을 보장하지는 않는다. 접근 속도와 물리적 상호작용은 예측과 달라질 수 있다.

### FSM과의 차이

- **SCT + FSM:** 각 참여자의 **실제 양손 12cm 근접 조건**을 확인해 도착을 기록하고 freeze한 뒤, 마지막 사람이 도착하면 모두 해제한다.
- **SCT + LLM Scheduler:** 각 참여자의 **예상 접근 소요 시간**을 비교하여 시작 시간을 조정한다.

## 4. 비교군 간 차이 요약

| 항목 | SCT + FSM | SCT + LLM Scheduler |
|---|---|---|
| 저수준 정책 | 동일한 사전 학습 SCT | 동일한 사전 학습 SCT |
| 추가 정책 학습 | 없음 | 없음 |
| 조율 근거 | 실행 중 관측되는 실제 상태 | 초기 장면에서 추정한 시간 |
| Enabling | 선행 AT 성공 후 후행 해제 | 선행 완료 예상 시점에 맞춰 후행 출발 |
| Coupled | 참여자 도착 확인 → 동결 → 전원 동시 해제 | 예상 접근 시간 차이에 따라 출발 지연 |
| 조율 유형 | **Reactive, state-based** | **Predictive, time-based** |
| 주요 한계 | 후행이 지나치게 늦게 출발할 수 있음 | 시간 예측 오차로 동기화가 어긋날 수 있음 |

모든 비교군에는 동일한 **task assignments**와 동등한 **virtual-object / pushing-region target information**을 제공하고, FSM과 LLM Scheduler는 동일한 SCT checkpoint를 사용한다는 것이 비교 설계의 원칙이다.

## 5. 아직 구현 전 확정할 사항

| 항목 | 확인해야 할 선택 |
|---|---|
| Enabling 소요 시간 | 후행의 이동 시간만 계산할지, 집기·접근·준비까지 포함할지 |
| 예측 실패 시 대응 | 선행 완료 전 후행 도착 시 추가 안전 대기/제어를 허용할지 |
| LLM 호출 횟수 | 현재 논의: **초기 1회 호출**. 실행 중 재계획은 별도 결정 필요 |
| Freeze 동작 | SCT + FSM과 동일한 물리적 동결·해제 방식을 사용할지 |
| Coupled 접촉 후 행동 | 예상 동시 도착 이후 조율 개입 여부 |
| 보고 항목 | 사용한 LLM, 프롬프트, 입력 정보, 출발 시간 계산, 예측 실패 처리, 추론 비용 |

이 결정 사항은 최종 구현과 논문 Supplementary 설명에 정확히 반영해야 한다.

## 6. 논문에서의 기술 수준

### 본문 (Sec. 4.1 Baselines)

실행 원리만 한 문장으로 설명한다.

```latex
SCT + LLM Scheduler estimates task-completion and approach times from initial scene observations to coordinate agent start times.
```

여기서 **task-completion times**는 Enabling에서의 선행 작업 예상 완료 시간, **approach times**는 후행·Coupled 참여자의 예상 접근 시간을 뜻한다.

### Supplementary

- LLM의 모델명·버전·입력 스키마·프롬프트
- Enabling과 Coupled의 시간 추정 및 대기 시간 계산
- freeze/release 처리
- 예측 오류 및 예외 상황 처리
- 초기 1회 호출 여부와 실행 중 추가 개입 여부
- 평가 시 동일 SCT checkpoint, 장면·목표·성공 조건의 일치 여부

## 7. 실험에서 검증하려는 질문

> **초기 장면에 기반한 외부 시간 스케줄링만으로도, 학습된 dependency-aware coordination에 필적하는 협력 성능을 낼 수 있는가?**

이 질문을 검증할 때는 **Group SR**뿐 아니라 Enabling의 **Completion Time**과 Coupled의 **\(t_{\mathrm{coop}}\)**를 함께 해석한다. 단, 현재 \(t_{\mathrm{coop}}\)의 최종 정의는 아직 확정되지 않았다.
