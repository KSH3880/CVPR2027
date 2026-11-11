# SCT + LLM Scheduler baseline

고정된 SCT 정책에 **초기 1회 시간 예측 → initial freeze → 시간에 따른 영구 release**를 추가하는 독립 코드 베이스다. Enabling, coupled, 두 관계의 혼합 제약을 지원한다. 주어진 system prompt는 [prompts/system.txt](prompts/system.txt)에 저장했다.

현재 실행 정책은 `stage1_policy.json`에 지정된 **Stage1 SCT 26k**다. 로컬 `Stage1Policy`가 원본 가중치·정규화를 복원하고 native Stage1 출력 일치 및 Stage2 coordination 미호출을 검사한다. Stage2 checkpoint 인자는 장면/네트워크 초기화용으로만 남으며 실제 action은 Stage1로 교체된다. 과거 Stage2 controller_id로 생성한 스케줄은 실행하지 않는다. 초기 root/rigid-body 동기화 오류는 아직 미해결이므로 다음 GPU 실행 전에 수정·검증해야 한다.

스케줄러 core는 CPU에서 입력·응답 검증, 오프라인 실행, 상태 전이 및 모의 Gym tensor projection을 검증했다. Baze 짧은 인증 요청도 확인했다. 별도로 지정된 Stage2 checkpoint의 실제 trajectory 기록 경로를 추가했으며 [기록 안내](RECORDING.md)를 따른다. LLM이 생성한 스케줄을 적용한 실제 rollout은 아직 검증하지 않았다. `examples/`의 예제는 **합성 plumbing fixture**이고 실제 calibration과 구분한다.

## 구조

| 경로 | 역할 |
|---|---|
| `sct_scheduler/core.py` | 입력 경계·LLM JSON 검증, 혼합 제약의 최단 대기 계산 |
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

API 입력은 환경 설명과 현재 초기 장면만 포함한다. 과거 데모를 로딩하지 않는다. 별도 template 문자열 치환 없이 environment/task/event 설명을 JSON에 넣는다.

```python
from sct_scheduler import build_messages
messages = build_messages({
    'environment_spec': environment_spec,
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

기본 base URL은 `https://api.openai.com/v1`이며 실제 POST 주소는 `https://api.openai.com/v1/chat/completions`다. 기존 Baze를 사용하려면 base URL을 `https://api.baze.co/v1/gateway`로 지정한다. 경로가 다르면 `--endpoint`로 전체 URL을 지정한다.

**API 키 입력 위치:** 이 프로젝트의 `.env` 파일. OpenAI API 프로젝트의 키와 해당 계정에서 사용 가능한 모델 ID를 입력한다. 현재 로컬 모델 설정은 사용자 지정값을 보존한다.

```dotenv
SCT_LLM_BASE_URL=https://api.openai.com/v1
SCT_LLM_API_KEY=여기에_키_입력
SCT_LLM_MODEL=사용할_모델_ID
SCT_LLM_REASONING_EFFORT=medium
```

`.env`는 Git에서 제외되며 명령 실행 없이 문자열로 읽는다. 같은 이름의 환경변수가 있으면 파일 값보다 우선한다. `--model`, `--base-url`은 설정을 덮어쓰고, `--env-file`로 다른 설정 파일을 선택할 수 있다. `.env.example`에는 키가 없는 설정 틀만 보관한다.

키·모델 입력 후 연결 테스트 명령은 다음과 같다. 합성 데이터를 사용하는 **실제 유료 API 호출**이 될 수 있으며 GPU 시뮬레이션은 실행하지 않는다. 출력 경로는 매번 새로 지정한다.

```bash
python3 -m sct_scheduler examples/synthetic_coupled_request.json \
  --output output_etc/check/baze_api_test_001
```

기본 CLI 요청은 `model`, `messages`, `stream=false`, `reasoning_effort="medium"`을 전송한다. reasoning 설정이 없거나 비어 있어도 medium을 적용하고 `--reasoning-effort`로 덮어쓸 수 있다. 선택 모델이 해당 effort를 지원해야 한다. 사용한 effort는 `inference.json`에 기록한다. [공식 Chat Completions 규격](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)을 따른다.

provider가 지원하면 `--json-mode`를 추가해 `response_format={"type":"json_object"}`를 보낼 수 있다. 이 옵션 없이도 prompt의 JSON 출력 지시와 로컬 검증은 적용된다. 자동 재시도는 없으며 HTTP 오류 본문·인증 헤더는 로그에 저장하지 않는다. 실제 데이터가 준비된 후 전체 endpoint를 직접 지정할 수도 있다.

```bash
python3 -m sct_scheduler real_request.json \
  --endpoint "$SCT_LLM_ENDPOINT" --model "$SCT_LLM_MODEL" \
  --output output_etc/check/real_scene_001
```

요청·system prompt·raw response·검증 응답·추론 latency·provider usage를 저장한다. 통화 비용은 provider 요율이 없으므로 추정하지 않는다. 추론 시간은 시뮬레이션 시간에 포함하지 않는다. 입력 JSON은 trusted exporter에서 생성해야 한다. 알려진 미래 데이터 필드 유입을 거부하지만 자유 텍스트에 숨은 사후 정보를 자동 판별하지는 못한다.

## 입력 계약

타이밍 calibration이 없어도 수치 예측과 출발 스케줄을 생성하도록 요청한다. 미보정 추정은 low confidence와 basis에 표시한다. 응답은 `predictions`, `schedule`, `predicted_episode_events` 세 필드로 구성하며 `status` 필드는 없다. 모든 agent의 스케줄을 요구하고 실제 타이밍 제약과 시간 합산을 검증한다.

- `environment_spec`: 제시된 units, coordinate_system, controller를 사용한다. `task_glossary`에 실제 task label 뜻을 추가하고 각 task에도 plain-language `description`을 넣는다.
- `controller_id`: 별칭보다 checkpoint SHA256 등 실제 동일성을 확인할 수 있는 식별자를 권장한다. 이 식별자는 메타데이터이며 검증기가 checkpoint 내용을 대신 증명하지 않는다.
- event ID는 장면 안에서 유일하다. 소유자는 task의 `readiness_event_id`, `enabling_event_id`, `dependency_sensitive_event_id`, `completion_event_id` 등으로 지정한다. 같은 에이전트의 추가 이벤트는 `event_agents: {"event_id": "agent_id"}`로 지정할 수도 있다.
- `event_definitions`는 모든 event ID의 정확한 조건을 포함한다. 이름만 보고 AT/SIT/ONTOP 등의 의미를 추측하지 않는다.
- current scene에는 scene/controller/coordination type, initial_observation, tasks, event_definitions, coordination_constraints와 선택적인 event_agents/planned_geometric_paths만 허용한다.

Coupled 제약은 다음과 같다. 정확한 predicted event alignment를 검사한다. 물리적 접촉 동기화 보장은 아니다.

```json
{"type": "align_events", "event_ids": ["A_ready", "B_ready"]}
```

Enabling 제약은 아래와 같다. follower가 dependency-sensitive event에 도달하는 episode time이 선행 enabling event보다 엄격히 늦어야 한다. 동시는 허용하지 않지만 추가 대기는 허용하며 최소 출발 시각이나 고정 여유시간을 요구하지 않는다. `enable`이라는 별칭은 사용하지 않는다.

```json
{"type": "enabling", "enabling_event_id": "A_enabled", "dependent_event_id": "B_ready"}
```

검증은 LLM이 제출한 release와 elapsed의 합으로 episode event time을 계산한다. Enabling은 선행 release=0 및 후행 event > 선행 event를 검사하고, coupled는 지정 event들의 episode time 일치를 검사한다. 최소 대기 해와 같을 필요는 없으며 LLM schedule을 자동 수정하지 않는다. 시간은 정수 centisecond로 비교해 부동소수점 오차를 피한다. `earliest_schedule`은 합성 예제·preflight용 참고 해를 만드는 도구로 남아 있지만 응답 통과 조건에는 사용하지 않는다.

## 기존 FSM과 연결

참고 코드: `/home/hwanhee/ksh/stage1_2_base_teacher/FSM/controller.py`, `FSM/entry.py`. 원본은 수정·실행·import하지 않는다. 새 어댑터에 물리 projection 방식을 옮겼고 state-based release 조건은 사용하지 않는다.

**이벤트 차이를 주의해야 한다.** 현재 원본 FSM은 실제 상자 표면에 양손이 12cm 미만인지 검사한다. 사용자가 제시한 JSON 예시는 각각 할당된 손 목표점과의 거리 12cm 이하이다. 둘은 다른 이벤트다. 실제 평가에서 채택한 조건을 동일한 detector·설명으로 current scene에 제공해야 한다. Enabling의 이동하는 물체도 follower의 intended final-state target을 명시해야 한다.

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

스케줄러 API CLI는 GPU를 사용하지 않는다. 별도의 `collect_trajectories.py`는 로컬 teacher 복사본으로 유한 길이 trajectory를 기록하며 GPU **6 또는 7**만 허용한다. 학습이나 MPS 관리·종료는 수행하지 않는다. 스케줄러 어댑터도 physical GPU **6 또는 7**만 허용하고, 프로세스 시작 전 `CUDA_VISIBLE_DEVICES`를 해당 GPU의 UUID로 지정했는지 검사한다. task는 logical `cuda:0`을 사용한다. 이는 기존 MPS의 ordinal remapping 혼동을 피하기 위한 조건이며, MPS pipe/log 환경은 이미 승인된 서버 설정을 유지해야 한다.

GPU 사용 여부·checkpoint·scene export를 검토한 뒤 실제 rollout을 실행한다. signal로 작업을 끄거나 MPS daemon/client를 건드리지 않는다. 실행은 bounded loop에서 정상 종료하고 GPU synchronize 후 hook을 복구한다. `ksh` 아래에는 코드·출력·캐시를 쓰지 않는다. 외부 teacher를 직접 실행하는 launcher를 추가할 때에는 별도의 로컬 checkout과 캐시/데이터 경로 검토가 먼저 필요하다.

## 검증 범위

CPU 테스트는 isolated/mixed/infeasible 제약, 시간 산술, 데모 없는 입력, 미래 정보 필드, 잘못된 응답, 일회 호출, release quantization, 두 환경·서로 다른 actor 순서·비연속 tensor view에서 projection, release 후 재동결 금지, hook 복구와 API URL·인증 헤더·설정 로딩·오류 처리를 검사한다. 모의 Gym은 실제 물리를 검증하지 않는다. 실험 결과나 성공률을 주장하지 않는다.
