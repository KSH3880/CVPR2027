# PUSH 목표 거리 커리큘럼

두 `stage1_push_depth180_*rsi1m*_goal_curriculum` 실험은 원래 손 높이/AT 기울기/RSI/물성/보상을 유지하며 목표 거리만 epoch 0~999: 1~5m, 1000~2999: 1~7m, 3000부터: 1~10m로 균등 샘플링한다. 목표 방향은 상자 로컬 +Y다. 단계 변경 이후 reset부터 새 거리를 사용하며 진행 중 episode의 목표는 유지한다. 작업 구역 중심을 ±16m로 벌리고 envSpacing=32를 사용한다.

학습 epoch는 첫 reset 전과 각 rollout 전에 연결한다. resume은 저장된 epoch를 이어가고, test/VNC도 checkpoint epoch에 해당하는 거리를 사용한다. TensorBoard `push/goal_max_m`으로 단계를 확인한다.

기존 실행에는 파일 수정이 실시간 반영되지 않는다. 기존 프로세스는 자동 중단/재시작하지 않는다. 새 커리큘럼으로 이어가려면 별도 checkpoint 사본을 이관한 뒤 전용 train wrapper에 RESUME_CHECKPOINT를 지정한다. 이관은 원래 1~4m 실험과 일치하는 설정만 허용하고 goal/장면 간격 외의 변경을 거부한다. 모델·optimizer·정규화·epoch/frame 등 checkpoint의 학습 상태를 보존하며 시뮬레이터 episode는 새로 시작한다.

저장소 루트에서 환경을 준비한 뒤 사본으로 이동한다.

```bash
TOKENHSI_GPU=6 bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
cd ports/ksh_stage1_stage2_base/push
export PYTHONPATH=tokenhsi
```

각 실험의 원래 checkpoint 경로를 `SOURCE`로 지정한다. 이관 출력은 기존 파일을 덮어쓰지 않는다.

```bash
# stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum
python tokenhsi/scripts/multi_agent/migrate_push_goal_curriculum.py --checkpoint "$SOURCE" --cfg_env tokenhsi/data/cfg/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum.yaml --output output/stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum/migration/resume.pth
TOKENHSI_GPU=6 RESUME_CHECKPOINT=output/stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum/migration/resume.pth bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_goal_curriculum_train.sh 2 2048 4
```

```bash
# stage1_push_depth180_h080_rsi1m_goal_curriculum
python tokenhsi/scripts/multi_agent/migrate_push_goal_curriculum.py --checkpoint "$SOURCE" --cfg_env tokenhsi/data/cfg/multi_agent/stage1_push_depth180_h080_rsi1m_goal_curriculum.yaml --output output/stage1_push_depth180_h080_rsi1m_goal_curriculum/migration/resume.pth
TOKENHSI_GPU=6 RESUME_CHECKPOINT=output/stage1_push_depth180_h080_rsi1m_goal_curriculum/migration/resume.pth bash tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_goal_curriculum_train.sh 2 2048 4
```

새 학습·로컬·VNC 기본 명령은 [config.md](../config.md)에 있다. 현재 준비된 500회 이관본은 각 새 output의 `migration/epoch500.pth`이며 본학습 전환은 아직 수행하지 않았다.
