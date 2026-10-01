# Frozen TokenHSI Carry의 담당 box 교환

원본 stage1 TokenHSI checkpoint를 고정하고, 기존 A=2 Carry 환경에서
agent→box 담당 매핑만 교환한다. path planner·속도 planner·world 모델은
사용하지 않는다. `TokenHSI/` baseline 폴더와 checkpoint는 수정하지 않는다.

추가된 `launch_ms18.sh`는 frozen ms18 실행기에 명시적 경로를 주는 별도 모드다.
아래 원본 Stage1 실행과 구분한다.

## ms18 + 경로를 주고 담당 교환

```bash
# viewer: 운반 중 교환. env0은 대조군, env1은 교환 대상.
CARRY_BOX_SWAP_VIEW=1 CARRY_BOX_SWAP_ENVS=2 MA_GPU=0 \
  bash TokenHSI-coord/carry_box_swap/launch_ms18.sh ms18_swap_view_01 carryWith

# headless: 집기 전 / 운반 중을 각각 비교
MA_GPU=0 bash TokenHSI-coord/carry_box_swap/launch_ms18.sh ms18_swap_walk_01 loco_carry
MA_GPU=0 bash TokenHSI-coord/carry_box_swap/launch_ms18.sh ms18_swap_carry_01 carryWith
```

- 기본 ms18 weight: `TokenHSI-masteer/output/ms18_maskteam_origscale_c06_s0_00009000.pth`.
  세 번째 인자로 다른 호환 ms18 checkpoint를 지정한다.
  frozen 하위 Stage1 weight는 `MS_CKPT`로 지정하며 기본은
  `TokenHSI-masteer/output/ckpt_stage1.pth`다.
- 경로는 학습된 path planner 출력이 아니라 명시적 33점 꺾은선이다.
  미보유 상태는 현재 root→담당 box→그 box 목표, 보유 상태는 root→목표.
  속도는 1.5m/s 명령이며 실제 속도 보장은 아니다.
- 담당 변경 시 이전 경로와 진행 커서를 폐기하고 현재 root에서 새 담당
  box/목표로 즉시 설치한다. 이후 경로는 유지하다가 보유/접근/도착 phase가
  바뀔 때 재설치한다. 보행을 멈추거나 접촉을 해제하는 명령은 추가하지 않는다.
- task obs·reward·경로 생성 모두 동일한 box permutation을 사용한다.
  중첩 호출에서는 다시 permutation하지 않는다. 물리 tensor alias는 복원한다.
- viewer는 ms18 경로 리본, 현재 제어 창, 초록 목표 box와 담당 연결선을 그린다.
  경로 리본 끝도 교환된 담당 목표를 사용한다. GUI 실행은 별도 확인이 필요하다.
- 모든 env의 첫 episode는 동일한 전체 600스텝 한도를 사용한다.
  PPO 학습용 초기 timeout 분산은 이 테스트에서 제거한다.
- 경로는 장애물 회피/곡률 validity 검사나 candidate 선택을 거치지 않는다.
  콘솔의 COORD invalid=0은 이 경로의 안전성을 검증했다는 뜻이 아니다.
- trace에 명령 `path`를 함께 저장한다. 교환 이벤트에서 물리 root/box 불변,
  관측 변화, 새 root/box/goal 앵커를 검증한다.

### ms18 운반 중 기능 검증 (2026-10-01)

GPU0, 기본 ms18 weight, seed0, 16환경(대조군/교환 각8), 599 실행 스텝.
대조군의 두 box 배송 proxy는 7/8, 실제 교환된 환경에서는 3/7이었다.
실제 교환 7/7에서 교환 후 바디 중심 <0.3m 근접이 있었다.
1개 교환 대상은 조건이 충족되지 않아 실제 교환 결과 분모에서 제외했다.
담당 변경 후 두 agent가 각각 새 담당 box를 held한 이력은 6/7이며,
동시에 보유하거나 실제 안전하게 건네줬음을 뜻하지 않는다.
단일 seed 기능 테스트이며 초기 상태가 agent/환경 간 일대일로 대응하지 않는다.
실제 물리 접촉/배송 성공을 확정하는 지표가 아니다.

최종 운반 중 로그: `runs/carry_box_swap/ms18_path_box_swap_carry_20261001_c/`.
집기 전 로그: `runs/carry_box_swap/ms18_path_box_swap_walk_20261001_a/`.
집기 전 대조군의 두 box 배송 proxy는 6/8, 실제 교환은 1/8,
교환 후 바디 근접은 3/8이었다. 두 테스트의 `analysis.json`은 실제 교환된
env만 포함하고 `trace.active & trace.swapped` 이후의 근접만 집계한다.
앞선 `_a`는 배치 설정 오류, `_b`는 초기 timeout 분산이 적용된 개발 중
실행으로 보존했으며 최종 비교에 사용하지 않는다.

## 교환의 의미

- actor 0/1, box 0/1의 물리 위치·회전·속도·접촉 상태는 그대로 유지한다.
- agent 0은 box 1, agent 1은 box 0의 위치·회전·속도·BPS·크기와 목표를
  관측한다. 목표는 기본적으로 box를 따라가므로 실제 각 box 목적지는 그대로다.
- 관측·보상을 계산하는 동안에만 논리적으로 입력을 모아 읽고, 원래 simulator
  tensor alias를 즉시 복원한다. reset은 물리 actor 순서대로 진행한다.
- `set_box_assignment(env_ids, assignment)`가 담당 변경 API다.
  `assignment`는 `[N,2]` long permutation으로 중복 담당을 허용하지 않는다.
  world 모델의 조건부 담당 선택은 이 API 위에 별도로 구현할 수 있다.
- 담당 변경은 손의 접촉을 강제로 풀거나 box를 넘겨주는 동작 명령이 아니다.
  frozen 정책이 새 관측에 어떻게 반응하는지를 측정한다.

## 실행

repo 루트에서:

```bash
# 운반 중: carryWith로 시작, 1초 이후 두 agent 모두 held proxy이면 교환
MA_GPU=0 bash TokenHSI-coord/carry_box_swap/launch.sh swap_carry_01 carryWith

# 집기 전: loco_carry로 시작, 0.2초 이후 두 agent 모두 미보유이면 교환
MA_GPU=0 bash TokenHSI-coord/carry_box_swap/launch.sh swap_walk_01 loco_carry

# viewer: cyan/orange 선은 각 agent의 현재 담당 box
CARRY_BOX_SWAP_VIEW=1 CARRY_BOX_SWAP_ENVS=2 MA_GPU=0 \
  bash TokenHSI-coord/carry_box_swap/launch.sh swap_view_01 carryWith
```

새 tag를 사용하고 GPU 점유를 먼저 확인한다. 세 번째 인자로 original stage1
checkpoint 경로를 지정할 수 있다. 로컬 기본값은
`TokenHSI-masteer/output/ckpt_stage1.pth`; 서버에는 코드와 경로를 별도 반영한다.
adapt/ms18 checkpoint는 모델 구조가 달라 이 runner에 바로 넣지 않는다.

설정: `CARRY_BOX_SWAP_ENVS=16`, `STEPS=600`, `SEED=0`.
실제 변수 이름은 모두 `CARRY_BOX_SWAP_` 접두사를 붙인다.
`CARRY_BOX_SWAP_AFTER`는 교환 조건 검사 시작 스텝이며 조건이 충족될 때까지
기다린다. `CARRY_BOX_SWAP_GOALS_FOLLOW_BOX=0`이면 box만 교환하고 agent의
목표는 유지한다. 기본은 1이다.

짝수 env는 교환 없음, 홀수 env는 교환 대상이다. 각 env의 첫 episode만
측정하고, 넘어진 env는 이후 집계에서 제외한다. 같은 설정의 대조군이지만
서로 다른 초기 모션·box 표본이므로 일대일 인과 비교는 아니다.
산출물은 `runs/carry_box_swap/<tag>/`의 config, sidecar, log,
`summary.json`, 물리 상태·매핑·held·active를 기록한 `trace.npz`다.

## 2026-10-01 실제 GPU 테스트

Original stage1 checkpoint, GPU0, seed0, 16환경(그룹별 8개), 최대 600스텝.
목표는 box를 따라가며 두 agent는 2.5m 떨어진 평행 lane에서 시작한다.

| 시작·그룹 | 실제 교환 수 | 두 담당 box 보유 관측 | 두 box 배송 proxy | 바디 근접 episode |
|---|---:|---:|---:|---:|
| 집기 전 대조군 | — / 8 | 8 / 8 | 8 / 8 | 1 / 8 |
| 집기 전 교환 | 8 / 8 | 5 / 8 | 5 / 8 | 3 / 8 |
| 운반 중 대조군 | — / 8 | 8 / 8 | 5 / 8 | 0 / 8 |
| 운반 중 실제 교환 env만 | 7 / 8 | 2 / 7 | 2 / 7 | 7 / 7 |

운반 중 1개 env는 두 agent 모두 보유 조건에 도달하지 않아 교환되지 않았다.
교환 후 수치는 이 env를 제외한 `analysis.json` 기준이다. 교환 순간 physical
root/box 전체 상태가 동일함과 관측 변화가 0이 아님을 실제 실행에서 검증했다.

held는 높이·root/손 거리 proxy, 배송은 box의 목표 3D 거리 <0.3m,
box 속도 <0.2m/s, 해당 agent 미보유를 10스텝 유지한 proxy다. 두 agent의
담당 box 보유는 episode 내 각자 한 번 이상 관측했다는 뜻이며 동시 보유나
물리 접촉 검증은 아니다. 근접은 캐릭터 바디 중심 최소 3D 거리 <0.3m다.
단일 시드의 기능 probe로 일반 성공률을 주장하지 않는다.

집기 전에는 새 담당 box를 처리한 표본이 있었지만, 운반 중에는 담당 교환만으로
안전하게 인계되지 않았다. 후속 관제는 내려놓기·재집기 등 전환 동작과 교환
조건을 별도로 정의하고 검증해야 한다. viewer 구문은 준비됐지만 GUI 실행은
이번 headless 검증에 포함되지 않았다.

로그: `original_walk_box_swap_20261001_a`, `original_carry_box_swap_20261001_a`.
매핑·형상/목표/이전 상태 일치·예외 시 tensor alias 복원 테스트 3개 통과.
