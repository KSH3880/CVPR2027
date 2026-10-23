# 실제 박스 크기를 반영하는 progress

2026-10-02 구현. 박스 범위를 확정하기 전에도 매 환경의 실제 `sizes`와 박스 회전으로 계산한다. RSI 모션·프레임·초기화와 박스 랜덤화 범위는 이번 변경에 포함하지 않는다.

## 계산

`progress = 1 / (1 + e_xy / sigma)`, `sigma = 1m`.

| 관계 | e_xy |
| --- | --- |
| SIT | 사람 root XY에서 기존 SIT 성공 사각형까지의 최단거리 |
| CLIMB | 사람 root XY에서 기존 CLIMB 성공 사각형까지의 최단거리 |
| ON_TOP | 옮기는 상자 중심 XY에서 받침의 기존 성공 사각형까지의 최단거리 |
| HOLDING | 사람 root XY에서 대상 상자 외곽 사각형까지의 거리에서 팔 접근 여유 0.3m를 빼고 0으로 제한 |
| AT | 옮기는 상자 중심 XY에서 goal까지의 거리에서 기존 허용 거리 0.5m를 빼고 0으로 제한 |

기존 성공 판정과 같은 yaw 기준 좌표 변환을 쓴다. SIT/ON_TOP은 각 변에서 전체 폭의 10%씩 안쪽으로 들어간 영역(폭 80%), CLIMB은 각 변에서 5%씩 들어간 영역(폭 90%)이다. 중앙을 제외한다는 뜻이 아니다. HOLDING의 0.3m 여유는 0.4m 박스의 한 변 방향에서 기존 중심 거리 0.5m와 일치한다.

progress가 1이어도 성공이라고 판정하지 않는다. 기존 높이·발·파지 등 성공 조건, phi와 나머지 보상은 유지한다. 실제 수렴 가속 효과는 후속 비교 학습에서 확인해야 한다.

## 적용 경로

- 환경: `tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_size_progress.yaml`
- 옵션: `relationReward.progress.kind: success_region`
- 기반: unified shared random binding. progress 변경에 더해 후속 요청으로 task별 full AMP를 활성화했다. 선택 기준은 [task_amp.md](task_amp.md).
- 새 실행: `tokenhsi/scripts/multi_agent/approach_scenario_stage1_size_progress_train.sh`
- headless 평가: 같은 경로의 `approach_scenario_stage1_size_progress_test.sh`
- 기본 출력: `../runs/size_progress`

기존 설정은 기존 계산을 유지한다. 실행 중인 학습에는 자동 적용되지 않는다. 보상 설정이 checkpoint 계약에 포함되므로 기존 distance 설정 checkpoint는 새 설정으로 일반 resume할 수 없다. 현재 구현은 이 계약을 유지하며, 기존 정책 이식을 위한 별도 변환은 구현하지 않았다.

아래는 새 학습 실행 명령이며 이번 작업에서는 실행하지 않았다. 현재 YAML의 크기 범위는 기존 XY 0.40~0.65m, Z 0.25~0.55m다. XYZ 0.20~0.65m RSI 검사 결과가 나오면 사용자가 결정한 범위와 RSI 설정을 별도로 적용한다.

```bash
cd /home/hwanhee/koo_cvpr/stage1_fix
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_size_progress_train.sh 2 2048 4
```

## 검증

최초 progress CPU 25개, task AMP 활성화 후 CPU 30개 테스트 통과. 사각형 거리·회전·기존 성공 조건 보존·과제별 expert/replay 매칭·기존 설정 회귀를 확인했다. 실행 스크립트 shell 문법 확인. GPU 학습 및 수렴 비교는 수행하지 않았다.

테스트 로그: `../runs/reference_rsi/size_progress_tests.log`.
