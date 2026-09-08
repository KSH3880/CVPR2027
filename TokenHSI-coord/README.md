# Coordinator 모듈 — juan 브랜치

`traj`의 `5f39199f7ea9396532fe251c46aa634734ee9f10`에서
`TokenHSI-coord/coordinator/` 26개 파일을 원본 그대로 가져왔다.
현재 실제 두 agent의 root/box/goal/phase 상태로 joint 경로와 속도를 생성하는
상위 모듈이다. 입력 계약은 `coordinator/schema.py`, 모델은 `coordinator/model.py`에 있다.

## 사용 범위

- PyTorch 기반 모델, planner, loss, policy, checkpoint loader와 자체 테스트를 포함한다.
- `TokenHSI-masteer`의 기존 학습·평가·sequential-stack 실행에는 연결하지 않았다.
- 학습된 PTH, `trajectory_predictor/`, simulator 복사본, 실행 스크립트는 포함하지 않는다.
- `coordinator/train_closed_loop.py`와 `measure_executor.py`는 원본 보존용이다.
  이 파일들의 실행에는 별도 Isaac Gym 및 원본 coord simulator bridge가 필요하다.
  이 브랜치의 현재 독립 모듈만으로 closed-loop 학습이나 viewer 실행은 지원하지 않는다.
- `coordinator/README.md`는 traj 원본 문서이므로 그 안의 `scripts/coord/` 실행 예제와
  과거 학습 결과는 이 브랜치에 해당 구성요소가 있다는 뜻이 아니다.

## 독립 검사

프로젝트 루트에서 PyTorch가 설치된 Python으로 실행한다.

```bash
cd TokenHSI-coord
OMP_NUM_THREADS=1 python -m unittest discover -s coordinator/tests -v
```

모듈 사용 시 `TokenHSI-coord`를 Python import 경로에 두고
`from coordinator.model import JointCoordinator` 등으로 import한다.
기존 체크포인트를 사용할 때는 해당 C1/C2/simple loader로 저장된 설정과 함께 읽는다.
모델을 새로 생성하면 무작위 초기 가중치이므로 학습 완료 모델과 구분해야 한다.
