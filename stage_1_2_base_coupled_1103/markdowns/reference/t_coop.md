# 공동 PUSH 접촉 시간 비율: t_coop

`tokenhsi/utils/cooperative_push_metric.py` 구현. 시작 정의가 미정이므로 단일 시작 방식을 확정하지 않고 두 방식의 결과를 모두 보존한다. 기존 보상·hand gate·성공률 계산을 변경하지 않는 평가 전용 지표다.

## 정의

각 trial(최초 배정된 공동 PUSH 그룹)에 대해:

```text
t_coop (%) = 100 × 전원 양손 근접 프레임 수 / 측정 구간 프레임 수
```

- 대상: 최초 배정 2·3·4명 공동 PUSH. 참가자는 도중에 제외하지 않는다.
- 거리: 손 중심에서 실제 공유 상자의 회전된 직육면체 전체 표면까지의 최소 3D 거리. 개인 분할 구역·선택된 밀기 면·gate를 사용하지 않는다. 상자 내부 점은 가장 가까운 경계까지의 양의 거리다.
- 접촉: 모든 참가자의 양손 거리가 문턱 **미만**. 기본 출력은 6·8·12cm 전부이며 최근 비교 기준은 12cm다. 접촉력·시작 시 한 손 접근 조건은 없다.
- 종료: 공유 상자 중심과 공유 중심 목표의 3D 거리가 **20cm 이하**가 되는 최초 프레임. 도달하지 못하면 episode 종료까지 포함한다.
- 중간 정지·손 이탈·실패 trial을 포함한다. 팀 첫 낙상에서 잘라내지 않는다.
- 밀기 시작 실패는 trial 점수 0%. 성공이 reset 상태부터 성립하면 측정 프레임 없이 0%다.
- 집계는 **trial별 t_coop의 산술평균**이다. 그룹 크기별 2·3·4명 및 공동 전체를 제공한다. 프레임 수로 가중 평균하면 긴 trial이 과대표집되고 시작 실패가 누락되므로 사용하지 않는다.

## 시작 방식

| 방식 | 조건 |
|---|---|
| `displacement` | 공유 상자 중심이 reset 위치에서 XY 20cm 이상 떨어진 최초 프레임 |
| `speed` | 공유 상자 XY 속도가 문턱 이상으로 1초 연속 유지된 첫 구간의 시작 프레임 |

속도 방식은 변위 조건을 대체한다. 기본 sweep은 0.2~1.0m/s, 0.1 간격이다. 속도 지속 판정이 **최초 목표 도달 프레임보다 엄격히 이전**에 완료되어야 한다. 성공이 없으면 episode 마지막 프레임까지 지속 판정을 완료할 수 있다. 문턱 아래로 한 번이라도 내려가면 연속 카운트를 초기화한다.

샘플은 제어 주기다. reset을 frame 0으로 저장하지만 측정은 post-step frame 1부터 시작하며 종료 프레임은 포함한다. 지속 조건은 `ceil(1초 / control_dt)`개의 연속 post-step 속도 샘플로 정의한다. 해당 개수의 제어 시간 구간을 의미하며, 연속 시뮬레이션 시각의 모든 물리 substep을 검사한 결과는 아니다. 실제 dt와 필요 프레임 수를 결과 JSON에 기록한다.

## 저장과 실행

individual-goal 공동 PUSH의 기존 정식 평가가 끝나면 seed 폴더에 `tcoop_trace.npz`를 추가로 저장한다. GPU에서 재평가를 돌리는 명령은 바꾸지 않는다. 이 변경의 시뮬레이션 실행 검증은 아직 하지 않았다.

저장 필드:

| 필드 | 형태 / 의미 |
|---|---|
| `hands` | `[T+1, env, agent, 2, 3]`, 양손 세계 좌표 |
| `boxes` | `[T+1, env, agent, 13]`, 실제 공유 상자의 위치·xyzw 회전·선속도·각속도 |
| `sizes`, `goals` | `[env, agent, 3]`, 실제 상자 전체 크기와 공유 중심 목표 |
| `groups`, `templates` | `[env, agent]`, 최초 그룹/과제 (PUSH=5) |
| `dt` | 제어 주기(초) |
| `end_step` | `[env]`, 각 환경의 최종 포함 프레임 인덱스 |
| `schema_version` | 1 |

현재 정식 평가는 horizon 또는 환경의 모든 사람이 낙상한 시점을 episode 끝으로 본다. `end_step`은 이 규칙을 따른다. 다른 수집기에서 데이터를 가져오면 실제 episode 끝을 지정해야 한다. 동일 그룹의 모든 사람에게 동일한 물리 상자 상태와 중심 목표를 연결해야 하며, 개인 region token을 전달하면 오류로 중단한다.

저장소 루트에서 **GPU 없이** 재집계:

```bash
# TRACE_DIR: 정식 평가의 seed 결과 폴더 (tcoop_trace.npz가 들어 있음)
python tokenhsi/score_tcoop.py "$TRACE_DIR" \
  --output output_etc/check/tcoop_rescore/results.json

# 특정 조건만 계산
python tokenhsi/score_tcoop.py "$TRACE_DIR" \
  --start-mode speed --speeds 0.4 --contact 0.12 \
  --sustain-seconds 1.0 \
  --output output_etc/check/tcoop_rescore/speed04_contact12.json

# 여러 seed 통합: 각 seed 폴더를 위치 인자로 나란히 전달
python tokenhsi/score_tcoop.py "$SEED0_DIR" "$SEED1_DIR" "$SEED2_DIR" \
  --output output_etc/check/tcoop_rescore/combined.json
```

기본 실행은 (변위 1종 + 속도 9종) × 접촉 문턱 3종 = **30개 조건**을 저장한다. `--displacement`로 변위 문턱도 변경할 수 있다. Python + numpy만 필요하며 torch/Isaac Gym을 import하지 않는다.

출력은 `combined`(여러 입력의 trial 단위 통합), `sources`(입력별 조건·그룹별 집계·trial 상세)다. trial 상세에 시작·종료·최초 성공 프레임, 분자·분모를 남긴다. `start_failures`도 별도로 보고한다. 입력 간 평균의 min/max는 평가 입력 간 산포이며 반복 학습의 산포가 아니다. **서로 다른 모델의 trace는 별도로 실행**해야 하며 한 명령에 혼합하지 않는다.

과거 `departure.npz`의 선택면 near boolean·XY 좌표·속도만으로는 전체 표면 거리 6·8·12cm를 복원할 수 없다. 위 필드가 들어 있는 실제 손/상자 궤적이 필요하며, 누락 시 추정하지 않고 오류를 낸다. 새 정식 평가에서 원시 데이터를 수집한 뒤에는 시작 방식과 거리 문턱 변경에 시뮬레이션을 다시 실행할 필요가 없다.

## 검증

CPU 합성 궤적으로 회전 상자 표면 거리, 내부·모서리 거리, strict 접촉 문턱, 단일 손 이탈, 정지 포함, 지속 카운트 초기화·시작 소급, 목표 전 지속 판정, 시작 실패 0%, 2·3·4명, 3D 목표 및 episode 끝을 검사한다.

```bash
python -m unittest discover -s tokenhsi/tests -p test_cooperative_push_metric.py
```

GPU 시뮬레이션 및 실제 정책 궤적의 수치 검증은 미실행이다.
