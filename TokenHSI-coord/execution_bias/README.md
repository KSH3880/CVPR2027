# Deterministic execution-bias MLP

Frozen executor가 33-point 계획을 커브 안쪽이나 goal 방향으로 잘라 실행하는 systematic
bias를 학습한다. 확률분포를 출력하지 않는다.

## 계약

```text
input  plan_xyv [B,A,33,3]
output error_xy [B,A,33,2]
actual_hat = plan_xyv[..., :2] + error_xy
```

모델은 joint plan 전체를 shared local frame에서 flatten한다. 따라서 뒤쪽 goal과 두 agent의
상대 geometry를 보존한 채 각 점의 deterministic world-coordinate `(dx,dy)`를 출력한다.
첫 점은 현재 실제 위치이므로 error를 정확히 0으로 고정한다.

```python
from execution_bias import ExecutionBiasMLP

model = ExecutionBiasMLP()
predicted_error = model(plan_xyv)
predicted_actual = model.executed_path(plan_xyv)
planner_cost = collision_cost(predicted_actual)
```

기존 planner의 `[B,K,A,33]` candidate 출력에는 다음 adapter를 사용한다. error model의
parameter를 freeze해도 `detach`하지 않으므로 collision/task cost gradient가 corrected path를
거쳐 원래 planned `(x,y,v)`로 전달된다.

```python
from execution_bias import attach_execution_prediction, load_bias_checkpoint

bias_model, _ = load_bias_checkpoint("bias.pth", device=path.device)
bias_model.requires_grad_(False).eval()
output = attach_execution_prediction(planner_output, bias_model)
cost = collision_cost(output["executed_path_world"])
```

## 데이터

NPZ는 정확히 세 array를 가진다.

```text
plan_xyv [N,A,33,3]
error_xy [N,A,33,2]
valid    [N,A,33]
```

`error_xy[i]`는 spatial index가 아니라 planned `(x,y,v)`에서 계산한 nominal timestamp의
실제 executor 위치와 planned 위치의 차이다. `align_actual_to_plan()`으로 30 Hz simulator
trace를 이 계약으로 변환할 수 있다. plan hold가 끝나거나 episode가 실패한 뒤의 점은
`valid=False`로 둔다.

```bash
cd TokenHSI-coord
python -m execution_bias.train train.npz \
  --validation validation.npz \
  --output ../runs/execution_bias/anti_feat_top_s2.pth
```

검증:

```bash
cd TokenHSI-coord
python -m unittest discover -s execution_bias/tests -v
```
