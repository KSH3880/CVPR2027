# SCT + LLM Scheduler baseline

고정된 SCT 정책에 **초기 1회 시간 예측 → initial freeze → 시간에 따른 영구 release**를 추가하는 독립 코드 베이스다. Enabling, coupled, 두 관계의 혼합 제약을 지원한다. 주어진 system prompt는 [prompts/system.txt](prompts/system.txt)에 저장했다.

현재 CPU에서 입력·응답 검증, 오프라인 실행, 상태 전이 및 모의 Gym tensor projection을 검증했다. 실제 LLM 호출·Isaac Gym rollout·성공률은 검증하지 않았다. 전체 teacher 저장소나 checkpoint를 복제한 프로젝트는 아니며, 이미 만들어진 Stage1 task/player에 연결하는 모듈이다. 실제 5개 demonstration, 현재 reset 장면 export, checkpoint와 event detector는 실험 환경에서 제공해야 한다. 예제는 모두 **합성 plumbing fixture**다.

## 구조

| 경로 | 역할 |
|---|---|
| `sct_scheduler/core.py` | 입력 경계·데모·LLM JSON 검증, 혼합 제약의 최단 대기 계산 |
| `sct_scheduler/client.py` | 주입 가능한 transport, 1회 호출, 원본 요청·응답·latency·usage 기록 |
| `sct_scheduler/runtime.py` | 시뮬레이션 프레임 기준 release clock |
| `sct_scheduler/isaac_adapter.py` | 기존 FSM 방식의 root/DOF/관측/PD 상태 동결 |
| `sct_scheduler/episode.py` | reset 완료 task에서 bounded episode 실행 연결 |
| `sct_scheduler/recording.py` | 실제 event detector 시간 기록, 미도달 이벤트 censoring |
| `examples/` | coupled/enabling 합성 요청·응답 JSON 및 생성기 |
| `tests/` | 표준 라이브러리 unittest, 어댑터 검사만 CPU torch 사용 |
| `provenance.json`, `THIRD_PARTY_LICENSE.txt` | 읽기 전용 참고 소스와 라이선스 |

## CPU에서 바로 확인

프로젝트 폴더에서 실행한다. core/CLI는 Python 표준 라이브러리만 사용한다. Python 3.8 이상을 기준으로 작성했다.

```bash
cd /home/hwanhee/juan/CVPR2027/SCT-LLM-Scheduler
python3 -m unittest discover -s tests -v

python3 -m sct_scheduler examples/synthetic_coupled_request.json \
  --response-file examples/synthetic_coupled_response.json \
  --output output_etc/check/offline_coupled

python3 -m sct_scheduler examples/synthetic_enabling_request.json \
  --response-file examples/synthetic_enabling_response.json \
  --output output_etc/check/offline_enabling

python3 -m sct_scheduler examples/synthetic_coupled_request.json --messages-only
```

출력 폴더는 매번 새로운 경로를 지정한다. 기존 결과를 덮어쓰지 않는다. `ok` 외의 응답은 CLI exit code 2이며 실행 가능한 schedule이 없다. 잘못된 JSON·산술·제약은 오류로 종료한다. 재시도·자동 스케줄 수정·동시 출발 fallback은 없다.

## LLM 연결

기본 메시지는 사용자가 제시한 구조 그대로다. 별도 template 문자열 치환 없이 environment/task/event 설명을 JSON에 넣는다.

```python
from sct_scheduler import build_messages
messages = build_messages({
    'environment_spec': environment_spec,
    'demonstrations': demonstrations,  # 정확히 5개, 동일 checkpoint
    'current_scene': current_scene,   # 실행 전 상태만
})
```

provider SDK를 쓰면 다음 transport 인터페이스만 구현하면 된다. 반환 raw text를 파싱하기 전 원본 그대로 보존한다.

```python
from sct_scheduler.client import OneShotScheduler

def transport(messages):
    result = my_llm_call(messages)  # 사용자 provider 호출
    return result.text, {'model': result.model, 'usage': result.usage}

scheduler = OneShotScheduler(transport)
answer = scheduler.schedule(request, 'output_etc/check/scene_001')
```

chat-completions 호환 HTTP endpoint도 지원한다. 모델·URL은 명시적으로 지정하며, 키는 `SCT_LLM_API_KEY` 환경변수에서만 읽는다. `response_format={"type":"json_object"}`를 지원하는 endpoint가 필요하다. SDK별 모델 옵션은 transport에서 설정한다. 실제 데이터가 준비된 후 아래 명령을 사용할 수 있다.

```bash
python3 -m sct_scheduler real_request.json \
  --endpoint "$SCT_LLM_ENDPOINT" --model "$SCT_LLM_MODEL" \
  --output output_etc/check/real_scene_001
```

요청·system prompt·raw response·검증 응답·추론 latency·provider usage를 저장한다. 통화 비용은 provider 요율이 없으므로 추정하지 않는다. 추론 시간은 시뮬레이션 시간에 포함하지 않는다. 입력 JSON은 trusted exporter에서 생성해야 한다. 알려진 미래 데이터 필드 유입을 거부하지만 자유 텍스트에 숨은 사후 정보를 자동 판별하지는 못한다.

## 입력 계약

- `environment_spec`: 제시된 units, coordinate_system, controller, trajectory_format을 사용한다. `task_glossary`에 실제 task label 뜻을 추가하고 각 task에도 plain-language `description`을 넣는다.
- `controller_id`: 별칭보다 checkpoint SHA256 등 실제 동일성을 확인할 수 있는 식별자를 권장한다. 검증기는 문자열 일치를 확인하며 파일 내용을 대신 증명하지 않는다.
- event ID는 장면 안에서 유일하다. 소유자는 task의 `readiness_event_id`, `enabling_event_id`, `dependency_sensitive_event_id`, `completion_event_id` 등으로 지정한다. 같은 에이전트의 추가 이벤트는 `event_agents: {"event_id": "agent_id"}`로 지정할 수도 있다.
- `event_definitions`는 모든 event ID의 정확한 조건을 포함한다. 이름만 보고 AT/SIT/ONTOP 등의 의미를 추측하지 않는다.
- 데모에는 모든 정의된 이벤트를 `reached=true/false`로 기록한다. false인 이벤트의 `episode_time_s`, `elapsed_since_release_s`는 null이다. trajectory의 null은 unavailable이며 0으로 치환하지 않는다.
- `elapsed_since_release_s = episode_time_s - executed_release_times_s[agent]`다. 실제 frame quantization이 있었다면 **실제 해제 시각**을 사용한다.
- agent trajectory 열 이름은 사용자 예시처럼 `trajectory_columns`이며, object는 `object_trajectory_columns`다. 모든 agent/object ID를 넣고 샘플이 없으면 빈 배열을 넣는다. observation window 밖 또는 역순 timestamps는 거부한다.
- `outcomes`는 가능한 경우 데모에 추가한다. 주어진 예시와의 호환을 위해 필수로 강제하지 않지만 미도달 이벤트 표기는 필수다. 실패 데모를 성공 데모로 교체하지 않는다.
- current scene에는 scene/controller/coordination type, initial_observation, tasks, event_definitions, coordination_constraints와 선택적인 event_agents/planned_geometric_paths만 허용한다.

Coupled 제약은 다음과 같다. 정확한 predicted event alignment를 검사한다. 물리적 접촉 동기화 보장은 아니다.

```json
{"type": "align_events", "event_ids": ["A_ready", "B_ready"]}
```

Enabling 제약은 아래와 같다. follower가 dependency-sensitive event에 도달하는 episode time이 선행 enabling event보다 빠를 수 없다. `enable`이라는 별칭은 사용하지 않는다.

```json
{"type": "enabling", "enabling_event_id": "A_enabled", "dependent_event_id": "B_ready"}
```

현재 baseline은 모든 제약을 `release[v] >= release[u] + elapsed[u_event] - elapsed[v_event]`로 검사한다. coupled에는 양방향 제약을 둔다. nonnegative release에서 componentwise earliest schedule을 계산하므로 isolated enabling/coupled 공식과 일치하고 혼합 관계도 공동으로 처리한다. 양의 cycle이면 release만으로 해결할 수 없다. LLM이 낸 schedule을 자동 수정하지 않고, 최소 대기 해와 다르면 거부한다. 시간은 0.01초 정수로 검증하여 float 오차로 모순을 만드는 것을 피한다.

## 기존 FSM과 연결

참고 코드: `/home/hwanhee/ksh/stage1_2_base_teacher/FSM/controller.py`, `FSM/entry.py`. 원본은 수정·실행·import하지 않는다. 새 어댑터에 물리 projection 방식을 옮겼고 state-based release 조건은 사용하지 않는다.

**이벤트 차이를 주의해야 한다.** 현재 원본 FSM은 실제 상자 표면에 양손이 12cm 미만인지 검사한다. 사용자가 제시한 JSON 예시는 각각 할당된 손 목표점과의 거리 12cm 이하이다. 둘은 다른 이벤트다. 실제 평가에서 채택한 조건을 동일한 detector·설명으로 데모와 current scene에 제공해야 한다. Enabling의 이동하는 물체도 follower의 intended final-state target을 명시해야 한다.

reset 직후, physics를 진행하기 전에 initial observation을 export한다. 그 task를 그대로 유지한 채 `run_episode`를 호출한다. batch에서는 환경마다 request 및 유일한 scene_id를 제공하고 actor slot 순서를 명시한다. `Stage1Policy`와 기존 normalization 복원은 환경을 준비하는 caller의 책임이다. Stage2 coordinator나 기존 FSM controller를 동시에 설치하면 안 된다.

```python
# Isaac Gym 환경에서는 isaacgym을 torch보다 먼저 import한다.
from sct_scheduler.episode import run_episode

answers = run_episode(
    task, requests, agent_order, scheduler,
    'output_etc/check/episode_001', physical_gpu=6,
    observe=read_current_observation,
    policy=frozen_stage1_policy,
    step=step_without_auto_reset,
    max_steps=900,
    record_step=record_events_and_samples,
)
```

callbacks의 계약은 `observe() -> obs`, `policy(obs) -> actions`, `step(actions) -> bool done`, `record_step(episode_time_s, task, controller)`다. observe는 현재 tensor에서 관측을 다시 계산해야 한다. 이미 reset한 task를 받으며 자동 reset을 사용하지 않는다. batch 일부가 reset되어야 하면 전체 호출을 종료하고 새 episode/request로 시작한다. 기존 formal eval runner를 사용하려면 동일 순서로 `InitialReleaseFreeze.before_action(frame)`과 `mask_actions`를 연결할 수도 있다.

release는 `ceil(release_time / task.dt)`번째 control frame에서 일어난다. 최대 한 control step 미만 지연되며 actual release 시각을 별도로 저장한다. inference latency와 관계없고, release 후에는 성공·접촉·낙상을 보고 추가 정지를 하지 않는다. 어댑터의 `close()`는 Python hook만 복구한다. simulator 또는 MPS를 종료하는 함수가 아니다.

동결은 원본 FSM처럼 root/DOF 상태, 속도, rigid-body 관측 및 PD target을 복원한다. **진짜 kinematic actor는 아니며**, solver 내부의 반력·미세 움직임이 존재할 수 있다. 상자는 고정하지 않는다. 실제 physics 동작과 정책 가중치 불변은 후속 GPU 검증 대상으로 남는다.

## GPU/MPS 운영

이 코드 베이스에는 학습/GPU 작업 launcher나 MPS 관리·종료 코드가 없다. CPU CLI는 GPU를 사용하지 않는다. 어댑터는 physical GPU **6 또는 7**만 허용하고, 프로세스 시작 전 `CUDA_VISIBLE_DEVICES`를 해당 GPU의 UUID로 지정했는지 검사한다. task는 logical `cuda:0`을 사용한다. 이는 기존 MPS의 ordinal remapping 혼동을 피하기 위한 조건이며, MPS pipe/log 환경은 이미 승인된 서버 설정을 유지해야 한다.

GPU 사용 여부·checkpoint·scene export를 검토한 뒤 실제 rollout을 실행한다. signal로 작업을 끄거나 MPS daemon/client를 건드리지 않는다. 실행은 bounded loop에서 정상 종료하고 GPU synchronize 후 hook을 복구한다. `ksh` 아래에는 코드·출력·캐시를 쓰지 않는다. 외부 teacher를 직접 실행하는 launcher를 추가할 때에는 별도의 로컬 checkout과 캐시/데이터 경로 검토가 먼저 필요하다.

## 검증 범위

CPU 24 tests: isolated/mixed/infeasible 제약, 시간 산술, 검열된 데모, 미래 정보 필드, 잘못된 응답, 일회 호출, release quantization, 두 환경·서로 다른 actor 순서·비연속 tensor view에서 projection, release 후 재동결 금지, hook 복구. 모의 Gym은 실제 물리를 검증하지 않는다. 실험 결과나 성공률을 주장하지 않는다.
