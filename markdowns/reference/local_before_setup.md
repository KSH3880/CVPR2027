# BEFORE 로컬 준비

- 경로: `/home/inkyu/projects/CVPR_stage2/stage_1_2_base_before`
- GitHub 브랜치: `stage_1_2_base_before`. jointpush checkout과 독립 복제.
- conda: `tokenhsi_koo`, GPU는 `TOKENHSI_GPU=0` 또는 `1`로 지정한다.
- 데이터는 기존 로컬 배포 데이터에 심링크 연결하며 원본을 수정하지 않는다.
- 기존 BEFORE용 `checkpoints/Stage1_00078000.pth`는 로컬 `Stage1V2Fixed_00078000.pth` 복사본이다. 이름 alias이며 브랜치 원본 Stage1 78k와 동일 파일임을 의미하지 않는다. 전이 호환성/실제 학습은 별도 검증이 필요하다.
- PUSH BEFORE용 `checkpoints/Stage1PushFinetune_00021000.pth`는 로컬 Stage1JointPushDensity3Cone90Finetune 21k 복사본이다. 최신 source5/virtual/asym 계열용 26k도 `pth/`에 복사했다. 원본 checkpoint는 보존했다.
- Stage2 학습 시작에는 Stage1 가중치를 사용한다. 이미 학습된 Stage2의 평가/이어 학습에는 해당 Stage2 checkpoint와 relation_config.yaml이 추가로 필요하다.
- 본학습과 GPU 시뮬레이션은 자동 실행하지 않았다. 데이터 경로·패키지·shell 검사는 학습 성공을 보장하지 않는다.

저장소 루트에서 실행하며 GPU 점유를 먼저 확인한다.

```bash
nvidia-smi
# stage2_before
TOKENHSI_GPU=0 HEADLESS=1 bash tokenhsi/scripts/multi_agent/stage2_before_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5_goal5_goalroute
TOKENHSI_GPU=0 HEADLESS=1 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_goal5_goalroute_train.sh 4 1024 8
```

짧은 검사는 같은 명령 앞에 `CHECK=1`을 추가한다. 출력은 자동으로 `output_etc/check/`를 사용한다. 다른 BEFORE 실험의 명령은 [config](../config.md)를 따른다.
