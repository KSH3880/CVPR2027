# SCT-LLM-Scheduler 사용 안내

이 문서는 `/home/hwanhee/juan/CVPR2027/SCT-LLM-Scheduler`의 현재 구현을 기준으로 한다.
운영 규칙은 상위 [OPERATIONS.md](../docs/OPERATIONS.md), 저장소 규칙은 [AGENTS.md](AGENTS.md)를 따른다.
기존 결과는 [실험 경로](#11-실험-경로)에서 찾는다. 결과를 덮어쓰지 않고 새 출력 폴더를 지정한다.

## 1. 동작 구조

```text
초기 장면 capture (GPU)
  → request.json 생성
  → LLM API 호출 (CPU/네트워크)
  → response.raw.txt 저장 및 검증
  → 저장된 초기 장면과 응답으로 rollout (GPU)
  → CPU 재채점·실제 release 검증
  → 최신 프로토콜 지표 재집계 (CPU)
  → report.md / protocol_v1_summary.json
```

LLM은 각 장면의 실행 전에 한 번 호출한다. 실행 중 재호출하지 않는다.
현재 장면 정보만 입력하며 demonstration/few-shot은 사용하지 않는다.
LLM 추론 시간은 simulation clock에 포함하지 않는다.
여러 장면을 API batch runner에서 병렬 호출하지만, 각 장면의 `messages`는 독립적이다.

실제 동작은 [stage1_policy.json](stage1_policy.json)의 **Stage1 SCT 26k**가 생성한다.
CLI에 보이는 Stage2 V2 checkpoint는 환경·네트워크 생성용 carrier다.
첫 행동 전에 Stage1 가중치·정규화를 복원하고, native Stage1 출력 일치와 Stage2 coordination 호출 0을 검사한다.

| 관계 | 출발 시각 결정 |
|---|---|
| Enable | 선행 AT는 0초. LLM이 후행의 초기 release만 결정. 예측상 `후행 release + q > 선행 p` |
| Coupled | 그룹별 2~4명 모두 개별 release. `T=max(a_i)`, `release_i=T-a_i` |
| Independent | 0초 출발. Mixed 내부 independent도 동일 |

후행은 선행 완료 전에 출발·접근할 수 있다. 선행 이후여야 하는 것은 지정된 dependency-sensitive event다.
실제 선행 성공을 기다리는 gate는 없다. 예측한 시각이 되면 성공 여부와 관계없이 release한다.
Coupled도 초기 위치에서만 대기하며, 접촉 지점에서 재동결하거나 별도 PUSH 시작 명령을 주지 않는다.

## 2. 환경과 API 설정

서버에 이미 설치된 환경을 사용한다. 이 폴더에 Isaac Gym을 새로 설치하는 절차는 포함하지 않는다.

```bash
cd /home/hwanhee/juan/CVPR2027/SCT-LLM-Scheduler
conda activate tokenhsi_koo
python -m unittest discover -s tests -q
```

일부 실행기는 검증용 Python을 `/home/hwanhee/anaconda3/envs/tokenhsi_koo/bin/python`으로 지정한다.
다른 서버로 이식할 때는 `evaluate_saved_api.py`, `run_protocol_v1_eval.py`, `run_group_gap_eval.py`의 환경 경로를 확인한다.
GPU entry는 runtime script가 conda/MPS/GPU visibility를 설정한다.

[.env.example](.env.example)을 참고하여 프로젝트 루트의 `.env`에 입력한다. 기존 `.env`를 덮어쓰지 않는다.

```dotenv
SCT_LLM_BASE_URL=https://api.openai.com/v1
SCT_LLM_API_KEY=여기에_실제_키
SCT_LLM_MODEL=사용할_모델_ID
SCT_LLM_REASONING_EFFORT=medium
```

- `.env`는 문자열로 읽으며 shell 명령으로 실행하지 않는다. 같은 이름의 환경변수가 우선한다.
- reasoning effort가 없거나 비어 있으면 `medium`이다.
- 실제 모델·effort는 실행 manifest와 `inference.json`을 확인한다. 최근 실행은 `gpt-6-luna`를 사용했다.
- base URL 뒤에 `/chat/completions`를 붙인다. `.env`에는 전체 endpoint가 아닌 base URL을 넣는다.
- Baze를 쓸 경우 base URL은 `https://api.baze.co/v1/gateway`다.
- API key는 출력·보고서·manifest에 쓰지 않는다.
- 아래 실제 호출 명령은 비용이 발생한다. 현재 transport는 자동 재시도를 하지 않는다.

## 3. API 입력과 프롬프트

실제로 전송하는 구조는 다음과 같다.

```python
from sct_scheduler.core import build_messages

request = {
    "environment_spec": environment_spec,
    "current_scene": current_scene,
}
messages = build_messages(request)
# [
#   {"role": "system", "content": prompts/system.txt의 내용},
#   {"role": "user", "content": json.dumps(request, ...)},
# ]
```

| 파일 | 사용 방식 |
|---|---|
| [prompts/system.txt](prompts/system.txt) | `build_messages()`가 실제 읽는 system prompt |
| [prompts/user_template.txt](prompts/user_template.txt) | 사람이 보는 입력 설명용 template. API에서 자동 치환하지 않음 |
| [prompts/user_example_16_agents.txt](prompts/user_example_16_agents.txt) | 지시문은 현재 user template과 동일한 설명용16명 예시. 실측 장면이나 실제 호출 입력으로 간주하지 않음 |

프롬프트 실험은 `system.txt`를 바꾼 뒤 **API를 새로 호출**해야 한다. 기존 응답으로 eval만 재실행하면 이전 스케줄을 사용한다.
호출 중 prompt를 바꾸지 않는다. replay runner는 전송 직전에 snapshot과 일치하는지 검사한다.

`environment_spec`은 단위·좌표계·controller 설명·task glossary 등을 담는다.
`current_scene`은 다음 정보를 담는다.

- scene/controller ID, 관계 종류
- 초기 사람 위치·방향·속도, 물체 pose·크기
- 사람별 task와 목표, 실제 coupled membership
- event ID와 정의, Enable/Coupled 제약

미래 trajectory·성공 시각·실행 결과는 현재 장면 입력에 넣지 않는다.
초기 기록에서 요청을 생성하는 코드는 [export_initial_scene.py](export_initial_scene.py)다.

현재 system prompt는 성공적인 협력을 우선하고 `Avoid unnecessarily long initial waiting.`을 지시한다.
불확실성 때문에 추가 대기를 권하는 문장은 제거했다. 특정 시간 여유나 최소 대기 해는 강제하지 않는다.

## 4. API 호출·출력·검증

[client.py](sct_scheduler/client.py)의 `chat_transport()`는 표준 라이브러리 HTTP client로 다음 POST를 전송한다.

```json
{
  "model": "설정한 모델 ID",
  "messages": ["system 메시지", "user 메시지"],
  "stream": false,
  "reasoning_effort": "medium",
  "response_format": {"type": "json_object"}
}
```

위 `messages` 항목은 설명용 축약이다. 실제 값은 3절의 role/content 객체다.
`response_format`은 CLI의 `--json-mode` 또는 batch runner가 활성화할 때만 추가된다.

`OneShotScheduler.schedule(request, output_dir)`가 입력·실제 messages를 먼저 저장하고 한 번 호출한다.
원문 저장 → JSON 파싱 → [validate_response](sct_scheduler/core.py) 순서로 처리한다.

응답은 다음 세 필드다. `status`, `limitations` 필드는 요구하지 않는다.

```json
{
  "predictions": [
    {
      "agent_id": "A01",
      "event_id": "A01_approach",
      "estimated_elapsed_time_s": 6.0,
      "confidence": "low",
      "basis": "장면 관측과 시간 추정 근거"
    }
  ],
  "schedule": [{"agent_id": "A01", "release_time_s": 0.0}],
  "predicted_episode_events": [
    {"agent_id": "A01", "event_id": "A01_approach", "episode_time_s": 6.0}
  ]
}
```

위 응답은 필드 설명용 축약이며 실제 장면의 모든 agent·필수 event를 포함해야 한다.
검증기는 모든 사람의 release, 유한한 음수 아닌 시간·소수점 두 자리, 선행/독립 agent의 0초 출발,
Enable strict-after, Coupled 예측 이벤트 시각 일치, `episode=release+elapsed`를 검사한다.
`earliest_schedule()`은 fixture/preflight 참고용이며 LLM 응답을 자동 수정하지 않는다.

### 단일 장면 CLI

```bash
# messages 확인: API 호출·GPU 작업 없음
python -m sct_scheduler examples/synthetic_coupled_request.json --messages-only

# 저장된 응답 검증: API 호출·GPU 작업 없음
python -m sct_scheduler examples/synthetic_coupled_request.json \
  --response-file examples/synthetic_coupled_response.json \
  --output output_etc/check/offline_check_001

# 실제 API 한 번 호출: GPU 시뮬레이션 없음
python -m sct_scheduler examples/synthetic_coupled_request.json \
  --json-mode --output output_etc/check/api_test_001
```

`examples/`는 합성 연결 테스트 자료다. 실제 demonstration·성능 평가 장면이 아니다.
단일 CLI는 strict 검증 실패 시 오류로 종료한다. Batch eval은 아래 별도의 executable 검사도 사용한다.

### 장면별 API 파일

```text
output_etc/check/<API_RUN>/api/
  table1/mixed_16h/general_seed1/env_0000/
    request.json
    messages.json
    response.raw.txt
    response.validated.json   # strict 검증 통과 시
    inference.json            # 응답 수신 시 model/usage/latency 등
    failure.json              # 호출 또는 검증 실패 시
```

단일 CLI는 위 파일을 지정한 output 폴더에 직접 저장한다.

- raw 응답은 성공 여부와 관계없이 수신하면 먼저 보존한다.
- strict 검증 오류가 있어도 모든 사람의 실행 가능한 release가 있으면 현재 batch 평가 정책에서는 원문 그대로 실행한다.
- 실행 불가능한 응답만 해당 장면에서 제외한다. 동시 출발 fallback·스케줄 보정은 없다.
- `api_failed`는 검증 실패도 포함하므로 곧바로 실행 제외 건수를 뜻하지 않는다.
- `api_nonexecutable`이 실제 실행 불가능 건수다. 정확한 원인은 `api_failures.json`, `api_audit.json`을 확인한다.
- 장면별 `failure.json`의 executable 표시는 scheduler strict 실패의 기록이다. 최종 실행 여부는 batch audit이 판단한다.

## 5. 실행 작업 선택

| 목적 | 실행기 | API/GPU |
|---|---|---|
| 새 장면부터 전체 실행 | `run_full_suite.py` | capture → API → eval |
| 같은 초기 장면에 prompt만 재호출 | `replay_saved_api.py` | API만 |
| 저장된 새 스케줄 평가 | `evaluate_saved_api.py` | GPU eval·기본 검증/집계 |
| 최신 프로토콜 CPU 지표 | `protocol_v1_rescore.py` | API/GPU 없음 |
| 8H3m·16H4m만 새 평가 | `run_group_gap_eval.py` | capture → 384 API → 6 eval |

현재 전체 범위는 **Mixed 2/4/8/16명 × 3seed + Table2 6설정 × 3seed = 30회**다.
각64환경이면 **1,920장면**이다. Independent 전용12회는 사용자 요청으로 제외했다.
Mixed 내부 independent는 실행·집계한다. Table1의 정식 Ind.와 혼동하지 않는다.
정식 공용 프로토콜512환경 규모와 현재64환경 탐색 결과를 구분한다.

### 새 장면부터 전체 실행

```bash
python run_full_suite.py --prepare-only --output output_etc/check/full_new_run_001 --envs 64
python run_full_suite.py --output output_etc/check/full_new_run_001 --envs 64 --workers 16
```

`--prepare-only`는 작업 목록만 출력한다. 실제 명령은 초기 capture와 API 호출까지 포함한다.
기본 그룹 간격은5m다. 기본 freeze는 legacy다.
이 실행기의 기본 보고서에는 과거 Coupled 구간 지표도 포함되므로 최신 지표는 7절 후처리를 사용한다.
현재 검증·재사용 경로의 중심은64환경이다. 특히 아래 두 replay/eval 실행기는1,920장면·64환경을 고정 가정한다.

### 저장된 장면으로 API만 다시 호출

```bash
python replay_saved_api.py \
  --source output_etc/check/full_mixed_table2_env64_20261111_003631 \
  --output output_etc/check/api_prompt_variant_001 \
  --workers 16
```

- `--source`는 `api/.../request.json` 1,920개를 가진 이전 전체 run이다.
- 입력 JSON은 그대로 사용하고 현재 `prompts/system.txt`로 messages를 새로 만든다.
- `--settings mixed_4h mixed_8h`로 설정을 제한할 수 있다(설정당 64환경×3seed=192개).
- `--additional-source <run>`을 붙이면 같은 설정/seed/env 경로의 입력은 추가 source가 우선한다. 예를 들어 8H 최신 3m 배치 source를 추가하고 4/8H만 선택할 수 있다. manifest의 입력 해시와 추가 출처를 확인한다.

- 물리 capture/GPU eval은 수행하지 않는다. 입력 해시·prompt snapshot을 manifest에 저장한다.
- batch timeout은180초, 기본 최대16병렬, 자동 재시도0이다.

### 저장된 API 응답으로 eval만 실행

```bash
SCT_PROTOCOL_VERSION=cvpr2027-evaluation-v1.0.0 \
python evaluate_saved_api.py \
  --api output_etc/check/api_prompt_variant_001 \
  --output output_etc/check/eval_prompt_variant_001 \
  --freeze legacy
```

해당 API run은 완료 상태·1,920건이어야 한다.
`manifest.json.source`의 이름에 대응하는 `teacher_runtime/output_etc/check/<SOURCE>/reference`가 실제로 존재해야 한다.
입력 JSON만 남아 있고 물리 reference가 없으면 동일 초기 장면의 rollout을 재현할 수 없다.
GPU6/7에서 각각 한 subprocess를 순차 실행한다. 추가 API 호출은 없다.
`SCT_PROTOCOL_VERSION`을 지정하면 첫 실제 정책 입력 검증을 활성화한다.
최신 t_coop/Jerk/Time 집계는 실행 완료 후 7절 명령을 별도로 실행한다.

### 경로가 고정된 자동화 스크립트

다음 파일은 현재 실험 경로가 코드 안에 들어 있는 편의 wrapper다. 다른 run에 쓰려면 경로를 검토하거나 위 일반 실행기를 사용한다.

- `run_wait_prompt_experiment.py`: 고정된 기존1,920개 request로 API 재호출 후 legacy 평가.
- `run_protocol_v1_eval.py`: `api_no_unnecessary_wait_20261111_151400` 응답으로 전체 평가 후 최신 CPU 집계.
- `run_eval16_batch.py`, `run_saved_eval16.py`: 초기16명 전용 실험 경로. 현재 전체 suite의 일반 실행기로 쓰지 않는다.
- `compare_fsm_llm_episodes.py`: 저장된16명 장면의 FSM/LLM 대조 진단.

## 6. GPU 평가와 freeze

`run_full_suite.gpu_job()`가 runtime shell을 통해 `teacher_runtime/batch_eval_entry.py`를 실행한다.
`REFERENCE_ONLY=1`이면 초기 state와 request만 기록한다. eval이면 저장된 reference와 입력을 대조하고 adapter를 설치한다.

정책 제어 순서:

```text
reset
→ reference 초기 상태 비교
→ InitialReleaseFreeze 설치
→ 각 frame에서 before_action(frame)
→ 현재 관측 계산 → Stage1 평균 행동
→ 고정 agent action mask
→ simulator step → 성공/낙상/trace 기록
```

release는 `ceil(release_time_s/control_dt)` frame에 적용되어 요청보다 먼저 풀리지 않는다.
실제 release frame·시각은 `release_audit.json`에 저장한다.

| freeze | 동작 | 사용 |
|---|---|---|
| `legacy` | root/DOF/관측/PD target을 고정, 속도0. 해제 때 초기 속도를 복원하지 않음 | 현재 기본·주 평가 |
| `snapshot_resume` | 대기 후 초기 root/관절/몸체 속도까지1회 복원 | 별도 비교 실험 |

두 방식 모두 실제 kinematic actor 전환이 아닌 상태 tensor projection이다.
상자는 고정하지 않고 사람 충돌체를 유지한다. 해제 후 재동결은 없다.
공통 낙상 판정은 별도이며, 낙상자는 motor를 해제한 수동 actor로 남는다.

GPU는 **물리6/7만** 사용한다. runtime이 GPU UUID visibility와 해당 MPS pipe를 확인한다.
`nvidia-smi`로 점유를 확인하되 MPS 제어·GPU reset·다른 client 종료는 하지 않는다.
viewer 종료가 필요하면 정상 종료 경로를 사용하며 signal로 GPU 프로세스를 끄지 않는다.
`/home/hwanhee/ksh`에는 코드·캐시·출력을 쓰지 않는다.

## 7. 프로토콜·검증·보고서

규칙 원본은 `/home/hwanhee/ksh/evaluation_protocol/README.md`, 고정된 로컬 사본은 [protocol_v1/README.md](protocol_v1/README.md)다.
공용 파일이 바뀌면 로컬 사본·구현·기존 결과의 적용 버전을 대조해야 한다.

```bash
# 계산 예제만 검사; GPU 없음
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
python protocol_v1/checks/verify.py

# 완료된 전체30개 suite에서 최신 지표 재집계; API/GPU 없음
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
python protocol_v1_rescore.py output_etc/check/eval_prompt_variant_001
```

`protocol_v1_rescore.py`는 검증된30개 실행을 전제로 한다.
6개 실행뿐인 간격 변형 run에 그대로 실행하지 않는다. Table2는 기존 전체 run에서 재집계한다.

검증 항목:

- reference와 초기 root·목표·그룹·물체 배정 일치
- Stage1 native actor/RMS 일치, 가중치 불변, coordination 호출0
- 첫 physics step의 큰 위치 점프/root-body 불일치 없음
- 프로토콜 활성 시 실제 첫 정책 입력과 초기 freeze 반영 관측 exact 일치
- GPU trace의 SR·건수·Error를 CPU로 재채점
- Table2 최초 성공 시각과 actual release가 원문 응답의 control frame에 일치
- distractor 관측/attention 유지와 물리 제거

주요 집계:

| 지표 | 계산 |
|---|---|
| Enable SR | 같은 유효 frame에서 선행 AT와 후행이 동시에 성공한 적 있는 쌍 |
| Enable Time Succ. | 성공 trial만 최초 그룹 성공 step/30을 평균 |
| Enable Time All | 성공은 실제 시간, 실패는 최대30초를 부여해 전체 trial 평균 |
| Coupled SR | 물리 공유 상자 중심이 공동 목표에 도달한 그룹 |
| t_coop | 전체 기록의 원래 전원 양손12cm 근접 수 / 공용20cm 시작·종료 counter 차이. float32 clip/NaN 규칙 |
| Mean Absolute Jerk | 회전하는 실제 상자 둘레64점, dt0.0333, 원본 순차 차분·유효 mask. trial0 제외 |

Time은 initial waiting도 포함한다. Enable 전체는 **각 seed에서 세 행동 동일 가중 평균 → seed 평균±표본표준편차**다.
Coupled 전체 Avg.도 **각 seed에서2/3/4명 동일 가중 평균 → seed 평균±표본표준편차**다.
표준편차는 `ddof=1`이다. 인원별 표준편차를 단순 평균하지 않는다.
합산 trial의 pooled 값은 seed 평균과 별도 보존하며 pooled 평균에 seed표준편차를 붙이지 않는다.

| 파일 | 의미 |
|---|---|
| `report.md` | 최신 프로토콜 재집계 후 사용하는 기본 보고서 |
| `protocol_v1_report.md` | 위와 같은 내용의 별칭 |
| `report_legacy_metrics.md` | 재집계 전 과거 지표 보고서 보존본 |
| `summary.json` | 기본 suite 집계. 과거 Table2 협력 지표가 포함될 수 있음 |
| `protocol_v1_summary.json` | 최신 SR/Time/t_coop/Jerk, 인원별·전체·seed별 값 |
| `protocol_v1_verification.json` | 공통 계산 예제·첫 입력·suite 검증·코드 해시 |
| run별 `protocol_v1_trials.npz` | t_coop/Jerk trial별 값·counter·전체 공동 frame 수 |

새 지표 재집계를 거치기 전의 일반 `report.md`는 자동으로 최신 지표가 되지 않는다.
Table1만 있는 간격 변형 run의 `report.md`는 해당 Mixed 결과만 담는다.

## 8. 저장 위치

```text
SCT-LLM-Scheduler/
  .env                         # 사용자 키·모델 설정, Git 제외
  prompts/system.txt
  stage1_policy.json
  sct_scheduler/               # API·검증·release clock·adapter
  protocol_v1/                # 공용 프로토콜 고정 사본
  output_etc/check/<RUN>/       # CPU/API 결과·manifest·status·보고서
  teacher_runtime/
    batch_eval_entry.py
    eval/                      # 장면·기록·Stage1 연결
    tokenhsi/                  # 로컬 simulation 코드
    checkpoints/               # 로컬 checkpoint
    output_etc/check/<RUN>/     # GPU 실제 실행·trace·검증
```

CPU 결과와 GPU 결과의 `<RUN>` 이름이 같다. CPU `status.json.gpu_output`으로 실제 GPU 경로를 확인한다.
API 폴더·reference를 symlink로 연결하는 실행기도 있으므로 참조 원본을 이동·삭제하지 않는다.

GPU run별 주요 파일:

- `launch.json`, `process.json`, `exit.json`, `run.log`
- `initial.pt`, `initial_scene.pt`, `resolved_env.yaml`, `physics_only_reset.pt`
- `trace.npz`: root·물체·유효 mask·낙상·성공·목표·관계
- `stage1_policy.json`, `release_audit.json`, `llm_initial_match.json`
- `first_policy_input.pt/json`: 프로토콜 활성 시 첫 실제 정책 입력 증거
- `suite_verification.json`, `verification.log`
- Table2: `table2.json`, `table2_trials.npz`, Coupled의 `cooperation.npz`

`cooperation.npz`의 원본 손/공유 상자 pose가 없으면 새 t_coop/Jerk를 정확하게 재집계할 수 없다.

## 9. 진행 상태와 실패 확인

```bash
cat output_etc/check/<RUN>/status.json
watch -n 5 cat output_etc/check/<RUN>/status.json
```

경로의 `<RUN>`은 실제 폴더명으로 바꾼다.
`cat`/`watch`는 매번 파일을 다시 열기 때문에 상태 파일의 atomic replace도 반영한다.
`tail -f status.json`은 교체 전 파일을 계속 볼 수 있어 진행 확인용으로 권장하지 않는다.

| 필드 | 해석 |
|---|---|
| `stage` | capture/api/eval/protocol_metrics/complete 등 현재 단계 |
| `api_done`, `api_total` 또는 `total` | 완료 API 건수와 전체 건수 |
| `api_failed` | 통신 또는 strict 응답 검증 실패 |
| `api_nonexecutable` | 실제로 실행할 수 없는 응답 |
| `captured` | 초기 기록을 만든 job 목록 |
| `evaluated` | rollout 완료 job 목록 |
| `verified` | CPU·초기상태·release 검증까지 통과한 job 목록 |
| `running` | 현재 GPU별 작업 |
| `failures` | 평가·집계 오류 목록 |

`completed=true`만 보지 말고 `stage`, `failures`, `verified` 개수도 확인한다.
전체 suite는30개, 간격 변형 suite는6개가 정상 완료 수다.
프로토콜 wrapper는 eval이 끝난 뒤 CPU 지표 집계가 이어지므로 wrapper도 complete인지 확인한다.
상위 wrapper 상태에 카운터가 없다면 하위 API/eval `status.json`을 확인한다.
`mirror_pipeline_progress.py <상위 status.json>`은 표시만 보완하며 실제 작업을 재시작하지 않는다.

## 10. 8H·16H 그룹 간격 변형

```bash
python run_group_gap_eval.py
```

이 명령은 비용이 발생하는 **새 초기 기록·384 API 호출·6개 GPU 평가**를 실행한다.
현재 구현은8H/16H, seed1/6/7, 64환경, legacy freeze, 현재 system prompt로 고정되어 있다.
GPU6은8H, GPU7은16H를 처리한다.

- 8H 그룹 중심 최소3m, 16H 최소4m.
- `SCT_GROUP_GAP_VARIANT=8h3m_16h4m`일 때만 적용. 옵션이 없으면 기본5m.
- 원 배치 반경, 최소간격 검증, metadata가 같은 설정을 사용한다.
- 내부 상대 배치를 유지하지만 그룹 간 배치와 관측이 달라지므로 새 스케줄을 생성한다.
- 기존5m 장면의 API 응답을 새 장면 응답으로 재사용하지 않는다.
- API 완료 후 GPU 평가·첫 입력/간격/CPU 재채점 검증·부분 Table1 집계가 자동으로 이어진다.
- Table2는 새로 실행하지 않는다. Time Succ./All은 기존 Table2 기록에서 CPU로 재집계한다.

## 11. 실험 경로

상태는 각 폴더의 `status.json`을 기준으로 판단한다.

| 경로 (`output_etc/check/` 아래) | 내용 |
|---|---|
| `full_mixed_table2_env64_20261111_003631` | 초기 전체64환경 장면·API·legacy 평가. replay 입력의 출처 |
| `api_no_unnecessary_wait_20261111_151400` | 추가 대기 권장 문구를 제거한 프롬프트의1,920 응답 |
| `eval_no_unnecessary_wait_legacy_20261111_151400` | 위 응답의 legacy 평가 |
| `eval_protocol_v1_legacy_20261111_165027` | 첫 입력 검증을 포함한 최신 프로토콜 재평가·Table2 Succ./All·Coupled 전체±std 보고서 |
| `mixed8_16_gap3_4_20261111_231041` | 8H3m·16H4m 새 배치 실험 |

[최신 전체 보고서](output_etc/check/eval_protocol_v1_legacy_20261111_165027/report.md)
· [8H/16H 간격 변형 상태](output_etc/check/mixed8_16_gap3_4_20261111_231041/status.json)

## 12. 자주 헷갈리는 점

- **API만 호출했는데 eval이 안 돈다:** `replay_saved_api.py`는 API 전용이다. eval 실행기를 별도로 사용한다.
- **프롬프트를 바꿨는데 결과가 같다:** 저장된 스케줄을 재사용하면 새 prompt가 적용되지 않는다.
- **단독 CLI와 batch 실패 수가 다르다:** strict response 검증과 실행 가능성 검사가 다르다.
- **새 report와 옛 report가 다르다:** 최신 프로토콜 후처리 완료 여부를 확인한다.
- **두 방식의 seed가 같은데 다른 장면이다:** 환경 수·RNG·그룹 간격이 달라지면 동일 장면이 아니다.
- **선행이0초인데 왜 행동이 달라지나:** 관측과 다른 actor의 움직임·충돌이 달라질 수 있다. 같은 checkpoint만으로 동일 행동을 보장하지 않는다.
- **Table1 Ind.가 왜 없나:** Independent 전용 평가는 현재 범위에서 제외했다. Mixed solo로 대체하면 안 된다.
- **표준편차가 무엇인가:** 여기서 표시는3개 평가 seed의 표본표준편차다. trial별 산포나 학습 seed 산포가 아니다.

## 13. 새 장면 생성용 2명 연속 GPU viewer

```bash
python view_continuous_enable.py --episodes 10 --seed 2 --port 6086 \
  --settings enable_sit enable_climb enable_ontop
```

GPU6·1환경·2명이며, 기본10에피소드다. 설정 목록을 순환하고 seed는 에피소드마다1씩 증가한다.
각 장면을 새로 capture하고 현재prompt로 API1회 호출한 뒤 legacy freeze·Stage1으로 재생한다.
기존64환경 결과의 특정 장면을 추출한 viewer는 아니며, viewer 진단 결과는 정식평가와 구분한다.
`--settings enable_sit`로 같은 task의 서로 다른 장면만 볼 수도 있다. `coupled_2`도 선택 가능하다.

VS Code에서6086 포트 포워딩 후 `http://localhost:6086/vnc.html?autoconnect=1&resize=remote`에 접속한다.
VNC 연결은 전환 중에도 유지된다. 초기 capture/API/새 환경 로딩 동안 화면이 잠시 비어 있을 수 있다.

- **Space:** 재생/일시정지. 최초 에피소드만 첫frame에서 일시정지하며 이후 장면은 자동 재생.
- **N:** 현재 에피소드를 건너뛰고 다음 장면 준비. 일시정지 중에도 사용 가능.
- **Q:** 연속viewer 전체를 GPU동기화 후 정상 종료. 창닫기도 전체 종료로 취급.
- **에피소드 완료:** 최대900step 또는 전원낙상 조기 종료 후 약2초간 마지막 상태를 보여주고 자동으로 다음 장면 준비.
- **0 / 1:** 전체/그룹 카메라. viewer창에 키보드 focus가 있어야 한다.

N/Q는 활성 Isaac viewer창에서 처리한다. capture/API 대기 중에는 다음 viewer가 열릴 때까지 기다린다.
각 GPU episode process는 정상 종료 후 새로 생성하므로 이전 낙상자의 해제된motor가 다음장면으로 이어지지 않는다.
새 scene마다 별도 API비용이 발생하고, 최대 `--episodes`개 실행 후 종료한다. MPS/GPU신호 종료를 사용하지 않는다.
상태는 `output_etc/check/viewer_continuous_2h_<시각>/status.json`의 episode/seed/setting/history를 확인한다.
GPU기록은 teacher_runtime 동명폴더의 `viewer/episode_XXXX/`에 보존한다.


## 14. 이미 평가한 장면 연속 viewer — API 호출 없음

```bash
python view_saved_eval.py --setting enable_sit --seed 1 --start-env 0 --episodes 10
```

이 절은 과거 64환경 방식이다. 현재는 아래 15절의 단일 환경 viewer를 사용한다. 이전 평가를 보려면 새 장면 생성기 대신 이 명령을 사용한다. 기본source는 `eval_protocol_v1_legacy_20261111_165027`이며 `--source`로 바꿀 수 있다. source는 완료된 전체suite와 API/reference symlink가 남아 있어야 한다.
저장된 초기상태·task·물체배정·스케줄·Stage1으로 물리 재실행하며 API key/transport를 읽거나 호출하지 않는다. 저장영상 재생이 아니라 기존평가 조건의 재현이다.
정확한초기배치를위해 내부64환경배치를보존하고 카메라는선택된2명장면1개에맞춘다. 새1환경을뽑아기존응답을적용하는방식과구분한다.
Space재생/정지, N은다음저장env를처음부터재현, Q전체정상종료. 완료시에도다음env로자동전환한다. 장면전환시원래배치를재초기화하므로환경/모델로딩대기는있지만API대기는없다.
기본noVNC6086을사용한다. 실행중인동일포트viewer를중복실행하지않고Q로정상종료한다.


## 15. 영속 단일환경 저장 장면 viewer — 현재 권장

```bash
python view_saved_single.py --seed 1 --start-env 0 --episodes 10 --port 6086
```

AT→SIT은 기본값이며, Coupled 2명은 `python view_saved_single.py --setting coupled_2 --episodes 64`로 조회한다. GPU6, 실제1환경2명이며 Stage1 모델/정규화/모션라이브러리는 시작시1회만 로딩한다. 다음장면에서 프로세스·시뮬레이터를 다시 생성하지 않는다. API호출·키로딩도 없다.
Space재생·정지, N다음저장env복원, Q전체정상종료, 완료후자동다음장면. 최초장면은첫물리프레임에서일시정지하며이후자동재생한다.
저장된64개장면의물체자산을시작시한번만준비하고, 현재장면의작업물체만활성화하며나머지는멀리보관한다. 사람과물리환경은1개이며원래64환경전체를계산하지않는다.
원래root/DOF/물체배정·크기·관계·목표·스케줄을복원한다. Source세계좌표를해당장면의로컬좌표로옮기고, 원래223D 관측에서초기rigid-body 상태를복원한다. 관측복원은허용오차1e-4이하검증하며정식평가의bit-exact 영상재생이아닌동일초기조건의단일환경진단이다.
각reset의restore_check.json, first_step_check.json, first_frame.png와 live_status.json을확인한다. 모션라이브러리를재로딩하던 view_saved_eval.py 방식은이viewer로대체한다.

단일 환경 viewer는 원본 평가의 ordered 접촉 영역 배정을 복원한다. 초기화 때 생성된 구형 goal marker는 저장 장면의 목표와 혼동되지 않도록 숨긴다.
