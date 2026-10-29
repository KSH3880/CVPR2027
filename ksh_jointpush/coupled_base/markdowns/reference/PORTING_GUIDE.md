# 다른 머신으로 이관

1. GPU/CUDA 호환 PyTorch, Isaac Gym, `requirements.txt` 의존성을 준비한다. Isaac Gym을 torch보다 먼저 import한다.
2. `dataset_loco_sit_carry_climb.yaml`의 네 데이터셋 `motions`를 공유 데이터에 심링크로 연결한다.
3. `checkpoints/Stage1_00078000.pth`를 가중치에 연결한다. fixed 추론에는 원래 run의 `relation_config.yaml`도 필요하다. BEFORE 전이 경로는 `STAGE1_CHECKPOINT`로 지정할 수 있다.
4. `TOKENHSI_GPU`를 지정한다. conda 기본은 `tokenhsi`이며 `TOKENHSI_CONDA_ENV`, `CONDA_BASE`로 변경한다.
5. [config](../config.md)의 명령에 `CHECK=1`을 붙이고 학습 환경 수를 유지한다. 결과는 `output_etc/check/`에 둔다.
6. VNC 의존성과 옵션은 [USAGE](../USAGE.md#서버-viewer)를 따른다.

모션 로딩용 `lpanlib/poselib`와 기본 humanoid 에셋이 필요하다. 캐시가 없으면 native screening을 생성한다. 외부 데이터·원본 결과·실행 프로세스는 보존한다.
