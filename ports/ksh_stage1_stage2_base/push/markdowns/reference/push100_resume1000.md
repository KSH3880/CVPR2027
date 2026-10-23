# 1000회 저장본에서 PUSH 100% 재개

기존 5과제 모델·optimizer·AMP·정규화 통계·epoch를 복원하고 `TASK_GRAPH=push_at`으로 모든 agent의 과제를 PUSH_AT에 고정한다. YAML의 혼합 비율은 checkpoint 호환을 위해 그대로 두며 고정 preset이 실제 그래프와 PUSH 상자 생성을 우선한다. `relation_config.yaml`의 `task_graph`로 유효 preset을 확인한다. AMP expert 혼합은 기존 설정을 유지한다.

PUSH 시작은 기존 보행 50%/선별 RSI 50%, 물성과 보상·목표 범위는 유지한다. 아래는 본학습 명령이며 기존 학습을 종료하지 않는다. 사용자가 기존 실행을 정리한 뒤 각 명령을 별도 터미널에서 실행한다. 출력은 기존 혼합 학습과 분리한다.

```bash
cd /home/hwanhee/koo_cvpr/stage1_fix/ports/ksh_stage1_stage2_base/push

TOKENHSI_GPU=6 TASK_GRAPH=push_at \
OUTPUT_PATH=output/stage1_depth180_20kg_push100_from1000 \
RESUME_CHECKPOINT="$PWD/output/stage1_v2_fixed_five_task_depth180_20kg/Stage1FiveTaskDepth18020kg_21-17-59-13/nn/Stage1FiveTaskDepth18020kg_00001000.pth" \
bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth180_20kg_train.sh 2 2048 4

TOKENHSI_GPU=6 TASK_GRAPH=push_at \
OUTPUT_PATH=output/stage1_depth200_20kg_friction04_push100_from1000 \
RESUME_CHECKPOINT="$PWD/output/stage1_v2_fixed_five_task_depth200_20kg_friction04/Stage1FiveTaskDepth20020kgFriction04_21-18-02-20/nn/Stage1FiveTaskDepth20020kgFriction04_00001000.pth" \
bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_five_task_depth200_20kg_friction04_train.sh 2 2048 4
```

추가 재개에도 `TASK_GRAPH=push_at`을 지정한다. 시각화는 기존 해당 물성 test/VNC wrapper에 새 checkpoint를 전달하고 `TASK_GRAPH=push_at`을 함께 지정한다. `HEADLESS=0`은 로컬 test, 서버는 VNC wrapper를 사용한다.

새 PUSH100% 두 학습 전용 TensorBoard (사본 루트에서 실행):

```bash
tensorboard --logdir_spec depth180_push100:output/stage1_depth180_20kg_push100_from1000,depth200_friction04_push100:output/stage1_depth200_20kg_friction04_push100_from1000 --host 127.0.0.1 --port 6012
```

VS Code에서 6012를 포워딩하고 `http://localhost:6012`에 접속한다. 기존 6008과 별도 실행이다.
