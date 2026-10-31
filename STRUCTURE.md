# 코드 위치

- `tokenhsi/`: 원본 TokenHSI 독립 소스 snapshot. 외부 원본 수정 없음.
- `tokenhsi/env/tasks/multi_task/humanoid_push_kd.py`: PUSH 물리·배치·RSI·보상·expert, 고정50% 환경 분리.
- `tokenhsi/env/tasks/multi_task/push_integration.py`: 전달 코드에서 observation/등록/indexed reset만 추출.
- `tokenhsi/utils/{jointpush_geometry,jointpush_rsi_geometry,jointpush_rsi}.py`:26k 기하/RSI sampler snapshot.
- `tokenhsi/utils/push_reward.py`: 순수PyTorch CONTACT+AT.
- `tokenhsi/learning/push_kd_agent.py`: 고정teacher·masked PPO/critic·PUSH-only AMP/replay·KL·resume 계약.
- `tokenhsi/learning/push_kd_losses.py`: loss/normalization 관측 변환, CPU 사용 가능.
- `tokenhsi/learning/push_kd_player.py`: student-only PUSH final/ever 첫episode 평가.
- `scripts/`: checkpoint 확장·preflight·학습/평가 진입점.
- `tests/`: CPU 정책/데이터/실제optimizer 경로 검증.
- `weights/`: 원본 teacher 및 확장 초기student. 학습된 PUSH 최종 모델 아님.
- `SOURCE.json`: 데이터/소스 출처와 checkpoint hash.

## 현재 reference scene 경로

- `push_reference/`: 원본Stage1 26k env/utils snapshot. sourcehash는PROVENANCE.json. 정책/network는가져오지않음.
- `tokenhsi/env/mixed_reference_env.py`:2인4상자mixed reference pool + native3과제KD pool, slot reset매핑과loss route.
- `tokenhsi/env/reference_projection.py`: 사람별397D 관측·4family→5task AMP변환.
- `tokenhsi/data/cfg/multi_task/push_reference.yaml`:26k저장환경. dataset경로와실행pool수만별도연결.
- 초기 `humanoid_push_kd.py` 1인adapter는현재runner에서사용하지않는다.
