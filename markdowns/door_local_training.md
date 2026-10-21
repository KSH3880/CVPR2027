# 절충 경첩 Door 로컬 학습 인계

현재 구현 범위는 `HumanoidMADoor`의 두 agent가 각각 자기 문을 여는 독립 OPEN 학습이다. 목표로 말한 “한 agent가 문을 열고 다른 agent가 carry한 박스로 같은 문을 통과”하는 공동 과제는 이 wrapper에 포함되지 않는다. DoorHold/BONES 등 다른 실험과 혼동하지 않는다.

수동 회전 경첩·DOF_MODE_NONE·stiffness0, `env.door.hinge: {damping: 2.0, friction: 0.1}`. 분리/공유 loco full-AMP 두 설정에 반영했다. 몸 접근0.2·손 접근0.2·progress0.2·success0.2, AMP 거리 taper 없음, 기존 loco reset 유지. 손 사용은 필수가 아니며 success는 문 각도1.15rad 이상이다. 닫힘 spring 없음. checkpoint 물리 계약에 경첩 값이 포함되므로 예전 물리 checkpoint를 이어 쓰지 않는다.

현재 작업 트리에는 Door 관련 untracked 파일과 core 수정이 있다. commit/push만으로는 로컬에 전달되지 않을 수 있다. 로컬 Codex는 최신 작업 트리의 task·reward·utils·env/train YAML·train/test/VNC·tests 및 기존 core의 Door 연동 변경까지 존재하는지 확인한다. 필요한 motion 파일은 `tokenhsi/data/dataset_door_loco.yaml`에 기재된4개다. 서버 데이터 심링크의 절대 경로는 로컬에서 유효하지 않을 수 있으므로 실제 로컬 데이터로 연결한다. 기존 humanoid asset과 motion 로딩 의존성도 확인한다. 원본 데이터는 수정하지 않는다.

아래 블록을 로컬 Codex에 전달한다.

```text
이 저장소의 절충 경첩 Door 분리형 학습을 내 로컬 Linux/NVIDIA 환경에서 실행해줘. 서버에서는 학습을 실행하지 마.

먼저 AGENTS.md, git status --short, changelog.md, markdowns/structure.md, markdowns/config.md, markdowns/door_local_training.md를 읽고 최신 작업 트리가 제대로 전달됐는지 확인해.

목표와 범위:
- 이번 학습은 두 agent가 각자 자기 문을 여는 기존 HumanoidMADoor OPEN 학습이다.
- 한 agent OPEN + 다른 agent CARRY + 같은 문 통과는 이후 별도 통합 과제다. 이번 작업에서 그 과제가 구현됐다고 보고하지 마.
- 스치면 조금 움직이고 멈추되, 힘을 유지해 밀면 열린다. 열린 문을 계속 잡을 필요가 없도록 닫힘 spring/motor는 추가하지 마.

현재 구현 확인:
- tokenhsi/env/tasks/multi_agent/humanoid_ma_door.py: optional door.hinge를 actor DOF damping/friction에 적용. driveMode NONE, stiffness0. 경첩 값은 checkpoint 계약에 포함.
- tokenhsi/data/cfg/multi_agent/door_open_unified_approach_amp.yaml: hinge damping2.0/friction0.1, AMP gate_enabled false, state_reward_weight0.4.
- 대응 train YAML의 share_edge_encoder false. shared_edge variant는 비교용 공유형이므로 이번에는 unified 분리형을 사용.
- 몸/손 보상은 각각0.2. progress/success는 각각0.2. success는 angle>=1.15rad이며 손 사용 필수 조건 없음.
- 현재 task/reward/utils/YAML/scripts/tests 중 untracked 파일과 core 연동 수정이 누락되지 않았는지 확인. git pull만으로 최신 서버 작업 트리가 전달됐다고 가정하지 마.

로컬 준비:
- nvidia-smi로 사용 가능한 GPU를 확인하고 TOKENHSI_GPU를 해당 로컬 번호로 한 번 지정.
- tokenhsi/scripts/multi_agent/runtime_env.sh 사용. conda env 기본 tokenhsi. Isaac Gym은 torch보다 먼저 import.
- motion manifest tokenhsi/data/dataset_door_loco.yaml의4개 파일 및 humanoid/door asset 존재 확인. 서버 절대 경로를 향하는 심링크는 로컬 데이터에 맞게 연결.
- 학습은2048환경 유지. 부족하면 임의 축소하지 말고 원인을 보고.

검증:
1) 올바른 import 순서로 CPU 테스트:
   PYTHONPATH=tokenhsi python -c 'import isaacgym; import pytest; raise SystemExit(pytest.main(["-q", "tokenhsi/tests/test_door_open.py", "tokenhsi/tests/test_door_reference.py", "tokenhsi/tests/test_door_approach_amp.py"]))'
   conda 준비는 먼저 runtime_env.sh를 source해.
2) TOKENHSI_GPU=<로컬 GPU> OUTPUT_PATH=output/door_balanced_hinge_local_physics_check bash tokenhsi/scripts/multi_agent/validate_door_physics.sh
   이것은2환경 비학습 물리 검증. humanoid 접촉은 prescribed pose이며 자연스러운 학습 정책의 증거가 아님.
3) TOKENHSI_GPU=<로컬 GPU> SEED=42 MAX_ITERATIONS=2 OUTPUT_PATH=output/door_separate_balanced_hinge_local_check bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
   RESUME_CHECKPOINT는 unset하고 scratch 확인. check가 끝난 뒤 본학습과 출력 경로를 분리.

검증 통과 시 본학습 실행:
unset MAX_ITERATIONS RESUME_CHECKPOINT
TOKENHSI_GPU=<로컬 GPU> TOKENHSI_CONDA_ENV=tokenhsi OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SEED=42 OUTPUT_PATH=output/door_separate_balanced_hinge_local bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2

기존 학습을 임의 종료하지 말고, 이전 물리 checkpoint를 resume하지 마. 본학습은 로그를 파일로 남기고 PID·output 경로·실제 경첩 값·GPU·환경 수를 보고해.

로컬 viewer:
TOKENHSI_GPU=<로컬 GPU> HEADLESS=0 OUTPUT_PATH=output/door_separate_balanced_hinge_local_view bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <이번 학습 checkpoint.pth> 2 1 2 3

TensorBoard success는 현재 성공 상태의 시간 비율이며 episode 성공률과 같다고 하지 마. viewer에서 스침/지속 밀기/관성/몸 충돌을 함께 확인하고 판단해.
```
