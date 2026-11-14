# SCT + FSM

원본 SCT(Stage1 26k) 정책에 대기·해제만 추가한다. Enable은 목표 정렬 가상 받침, Coupled는 실제 공유 상자의 개인별 PUSH 구역을 입력한다. Stage2 checkpoint는 환경·네트워크 생성에만 사용하며 첫 행동 전에 SCT 가중치·정규화를 복원한다. Stage2 coordination은 실행하지 않는다.

## 동작

| 관계 | Freeze | 한 번만 해제하는 조건 |
| --- | --- | --- |
| Enable | 후행을 초기 자세에서 정지 | 선행 AT의 실제 물체 중심이 목표3D 거리20cm 이하에 도달 |
| Coupled | 각 참여자가 양손 모두 자기 그룹의 실제 상자 표면12cm 미만에 접근하면 도착 기록 후 정지 | 최초 참여자 전원이 한 번씩 도착하면 일괄 해제. 마지막 도착자는 정지하지 않음 |

Coupled 도착 기록은 접촉이 끊겨도 유지한다. 해제 후에는 다시 멈추지 않으며, 서로 다른 그룹의 도착·해제는 독립적이다. 낙상으로 유효하지 않은 사람의 새 접촉은 도착으로 기록하지 않는다.

Enable의 선행은 성공 후에도 원본 SCT 정책으로 계속 제어한다. 선행의 도착 후 freeze·상자 고정은 추가하지 않는다.

Freeze는 root 위치·회전과 DOF 자세를 저장하고 매 물리 서브스텝 후 복원하며 속도를0으로 한다. 대기 중 PD 목표도 저장 자세로 유지한다. 사람 충돌체는 남고 상자 상태는 덮어쓰지 않는다. 물리 solver 안에서 진짜 kinematic actor로 전환하는 방식은 아니므로 서브스텝 중 미세 움직임·반력이 생길 수 있다. 관측의 사람 자세·속도도 고정 상태로 맞춘다.

**대기 중인 사람은 파란색**, 해제하면 원래 그룹/역할 색이다. 콘솔의 `[FSM] frame=... freeze/release`는 env·사람 인덱스(0부터)를 출력한다.

## 서버에서 보기

저장소 루트에서 실행한다. GPU6/7만 허용하며 기본은7이다. noVNC 포트를 VS Code에서 포워딩한다. 동시에 켜려면 서로 다른 `PORT`를 사용한다.

```bash
cd /home/hwanhee/ksh/stage1_2_base_teacher

# Enable: AT → SIT
TOKENHSI_GPU=7 PORT=6081 bash FSM/vnc.sh enable_sit --seed 1

# Enable: AT → CLIMB / ON_TOP
TOKENHSI_GPU=7 PORT=6081 bash FSM/vnc.sh enable_climb --seed 1
TOKENHSI_GPU=7 PORT=6081 bash FSM/vnc.sh enable_ontop --seed 1

# Coupled: 2 / 3 / 4명
TOKENHSI_GPU=7 PORT=6081 bash FSM/vnc.sh coupled_2 --seed 1
TOKENHSI_GPU=7 PORT=6081 bash FSM/vnc.sh coupled_3 --seed 1
TOKENHSI_GPU=7 PORT=6081 bash FSM/vnc.sh coupled_4 --seed 1
```

접속: `http://localhost:6081/vnc.html?autoconnect=1&resize=remote`. 종료는 Ctrl+C다. 기본1환경·900step·10개 episode이며 장면마다 FSM을 초기화하고 낙상으로 해제한 motor도 복원한다.

## 로컬 viewer / 짧은 확인

```bash
TOKENHSI_GPU=7 bash FSM/view.sh enable_sit --episodes 1
TOKENHSI_GPU=6 bash FSM/view.sh coupled_2 --envs 2 --episodes 1 --steps 300 --headless
```

`--envs`, `--episodes`, `--steps`(최대900), `--seed`, `--headless`, `--capture`, `--output`을 지원한다. `--capture`는 viewer의 초기/전환/주기 화면을 episode당 최대8장 저장한다. `--output`은 비어 있는 `output_etc/check/` 하위 경로만 허용한다. 기본 경로는 `output_etc/check/fsm_<setting>_<시각>`이다.

SR은 기존 formal eval의 XY100%·Enable 동시 성공·공동 상자 중심 성공·사람별 낙상 처리를 재사용한다. 설정별 Table2 샘플링·공통 교집합·near50 해제·ordered·M=2N·distractor 관측/attention 유지·물리 제거를 적용한다. 이 viewer 확인 결과는512환경×3seed 본평가가 아니다. t_coop 정의는 변경하지 않는다.

결과는 episode별 `result.json`, `fsm.json`(전환·고정 오차·구현 정보), `fsm_trace.npz`(제어 전 손 거리·접촉·유효 상태·freeze/release), 기존 `trace.npz`에 저장한다. `initial_before_fsm.pt`는 freeze 적용 전 물리 상태다. 최상위 `stage1_policy.json`은 원본 정책/정규화 exact·가중치 불변·coordination 호출0을 검증한다.

기본 SCT: `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`. `--stage1-checkpoint`로 같은26k 호환 checkpoint 경로를 지정할 수 있다.

## 코드

- `state.py`: 관계별 대기·일회 해제 상태.
- `controller.py`: 상태 고정·접촉 판정·전환 기록·대기 색상.
- `entry.py`: 현재 평가 장면·SCT 정책·공통 SR을 viewer에 연결.
- `run.py`, `view.sh`, `vnc.sh`: 인자·출력 경로·runtime·noVNC 진입점.
- `test_state.py`: 순서·누적 도착·마지막 즉시 해제·그룹 독립성·낙상·일회 해제 검사.

## Table 1·2 평가

`eval_run.py`가 기존 장면 샘플러·물리/관측 분리·Stage1 입력 변환·메인 SR·Table2 지표를 호출하고 `eval_entry.py`가 FSM을 연결한다. 메인/eval 코드 수정 없이 `FSM/`에서 실행한다. 기본512환경·900step·임시 seed1/3/6·XY100%·M=2N이며, 5명 이상 그룹 간격5m·그룹별 잔여 distractor 관측/attention 유지·물리 제거를 적용한다.

```bash
TOKENHSI_GPU=6 python FSM/eval_run.py --output output_etc/check/sct_fsm_eval_v2_20261109 --gpu 6 \
  --jobs mixed:2 enable_sit enable_ontop &

TOKENHSI_GPU=6 python FSM/eval_run.py --output output_etc/check/sct_fsm_eval_v2_20261109 --gpu 6 --worker a \
  --jobs mixed:8 independent:2 &

TOKENHSI_GPU=6 python FSM/eval_run.py --output output_etc/check/sct_fsm_eval_v2_20261109 --gpu 6 --worker b \
  --jobs independent:16 enable_climb &

TOKENHSI_GPU=7 python FSM/eval_run.py --output output_etc/check/sct_fsm_eval_v2_20261109 --gpu 7 \
  --jobs coupled_2 mixed:4 coupled_3 &

TOKENHSI_GPU=7 python FSM/eval_run.py --output output_etc/check/sct_fsm_eval_v2_20261109 --gpu 7 --worker a \
  --jobs mixed:16 coupled_4 &

TOKENHSI_GPU=7 python FSM/eval_run.py --output output_etc/check/sct_fsm_eval_v2_20261109 --gpu 7 --worker b \
  --jobs independent:8 independent:4 &

wait

python FSM/eval_summary.py output_etc/check/sct_fsm_eval_v2_20261109
```

Conda `tokenhsi`에서 실행하며 각 시뮬레이션은 공통 runtime을 로드한다. GPU당3개 worker를 동시에 실행하는 예시다. `--worker a/b`로 같은 GPU의 상태 파일을 분리하고 설정별 파일 잠금으로 중복 rollout을 방지한다. 기존 Stage1 평가의 동일 seed 초기 snapshot을 읽기 전용으로 재사용한다. `--envs`가512와 다르면 확인용 snapshot을 새로 생성한다. 완료 rollout/검증은 재사용하며 미완료 산출물은 덮어쓰지 않는다.

기존 canonical manifest가 있는 설정의 seed만 분리할 때는 `--seed-worker --jobs coupled_4 --seeds 3 --worker coupled4_seed3`처럼 한 설정·한 seed를 지정한다. 설정 전체를 처리하는 기존 큐의 해당 작업은 먼저 재배분해야 한다. Seed별 잠금을 사용하며 원래 manifest의 전체 seed가 검증된 뒤에만 설정 완료와 평균/표준편차를 집계한다.

`worker_gpu6.json`, `worker_gpu7.json`과 각 `_a.json`/`_b.json`에 진행 상태를 기록하고, 완료된 설정의 seed별 결과·평균±표본표준편차는 `report.md`, `summary.json`에 집계한다. 초기 장면/관측 exact·원본 Stage1 가중치/정규화 exact·coordination0·distractor901회·CPU SR/Error·FSM 상태 replay를 검증한다. Table2는 성공 Time과 손/실제 상자 좌표도 기록한다. `t_coop`은 시작 조건 미정이므로 기존20cm 시작·6/8/12cm 접촉 값은 참고값으로만 표시하며, `cooperation.npz`로 추후 재집계한다.

## Freeze 충돌 확인

```bash
TOKENHSI_GPU=7 PORT=6083 bash FSM/vnc.sh enable_sit --collision-probe --steps 180 --episodes 3 --capture
```

같은 사람 자세·같은 상자로 다음3개 장면을 순서대로 실행한다. 시작1초 동안 자세를 고정한 뒤 상자를3m/s로 발사하거나 사람을 해제한다.

| 순서 | 장면 |
| --- | --- |
| 1 · `frozen_hit` | 파란 사람 freeze 유지 + 빨간 상자 충돌 |
| 2 · `free_hit` | 초록 사람 freeze 해제 + 같은 상자 충돌 |
| 3 · `free_nohit` | freeze 해제, 상자는 발사하지 않음. 원래 자세의 불안정성 확인 |

사람의 PD 목표는 세 경우 모두 같은 저장 자세다. SCT 정책 행동·FSM 성공 해제는 사용하지 않는 **물리 확인 전용 장면**이며 SR 평가가 아니다. 다른 사람·상자·목표 표시는 밖으로 옮기고 camera는 대상 사람을 고정해서 본다. `--collision-probe`는 `enable_sit`·1환경·31step 이상에서만 사용할 수 있고 기본 FSM viewer에는 적용되지 않는다.

Freeze된 사람은 충돌 계산에 참여하고 상자의 속도·방향을 바꾸지만, 사람의 root/DOF 상태는 서브스텝 끝에 복원된다. 진짜 정적/kinematic 장애물과 완전히 같은 동역학은 아니다. 각 장면의 `collision_trace.npz`에 사람/상자 상태·접촉력을, `probe.json`·`collision_summary.json`에 요약을 보존한다. 접촉력에는 바닥 접촉도 포함되므로 힘의 최댓값 전체를 상자→사람 충격으로 해석하지 않는다.

실제 확인에서 고정된 사람의 root 변위는0이고 공중 충돌 시 상자 X 속도는2.998→1.717m/s였다. [결과·화면](../output_etc/check/fsm_collision_20261109/report.md).

## 검증

누적 도착·마지막 즉시 해제·그룹 독립성 등을 CPU7개 테스트로 확인했다. 초기 freeze 뒤 전체 root/DOF reset을 재제출해 Isaac Gym의 대기 중 reset 덮어쓰기를 방지한다. GPU6의 Mixed2H64환경40step, GPU7의 Enable SIT/Mixed8H 각2환경220step은 초기 장면·첫 프레임 위치·원본 정책·CPU SR/Error/FSM 재검증·고정 root 변위0을 통과했다. [현재 검증](../output_etc/check/fsm_reset_smoke_20261109/verification.json). 이전 초기 freeze가 있는 viewer/소규모 rollout은 성능 근거에서 제외한다.
