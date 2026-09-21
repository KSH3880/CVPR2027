# Carry collision-avoidance planner

`stack_planner`의 Transformer, 33-point joint path/speed head, decision
history, PPO action distribution, checkpoint format을 그대로 사용한다. 실행 task만
sequential stack이 아닌 기존 simultaneous `HumanoidMASteerCarry` 계열이다.

- 두 agent 모두 `root -> own box -> own goal`을 동시에 수행한다.
- sequential phase, A1 retreat, A2 handoff, stacking reward는 없다.
- pickup(index 16)과 placement(index 32)는 정확한 hard anchor다.
- reward는 frozen ms18의 원래 Carry reward와 task progress를 유지하면서
  agent-agent, agent-other-box, box-box proximity cost를 감점한다.
- 기본 stress distribution은 `MS_SCEN=cross`다.

Smoke 예시:

```bash
CARRY_PLANNER_ENVS=8 CARRY_PLANNER_ITERS=1 \
CARRY_PLANNER_HORIZON=4 CARRY_PLANNER_LOW_STEPS=6 \
MA_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi118 \
  bash TokenHSI-coord/carry_planner/train.sh carry_smoke \
  /path/to/ms18/Humanoid.pth
```

출력은 `runs/carry_planner/<tag>/`에 저장한다. 동일 tag는 덮어쓰지 않는다.
