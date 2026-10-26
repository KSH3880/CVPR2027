# TokenHSI CLIMB 환경으로 78k full fine-tuning

기존 TokenHSI SIT FT와 같이 Stage1V2Fixed_00078000.pth에서 전체 모델과 정규화 통계를 학습한다. SIT FT 결과 weight에서 이어가는 것이 아니다. 35D adapter/GTA 모델 및 AMP 3채널은 78k 호환을 유지한다. 원본 단일 태스크 정책의 네트워크로 교체하지 않는다.

1명·1물체·2048환경, lr2e-5, 추가20000 iteration, 100회마다저장. 원본4096환경 대비 환경수와 pretrained 모델·FT 학습률/예산은 기존 SIT FT를 따른다. 이 서버는 GPU0/tokenhsi118이며 CUDA11.8 NVRTC 호환 때문에 PYTORCH_JIT=0을 기본으로 한다.

원본 HumanoidClimb.ObjectLib로 Box/Cabinet/Table_Square를 각1/3 확률로 뽑고 종류 내 모델은 균등 replacement로 환경당 고정한다. native URDF/bbox/tarSitPos를 사용한다. 걷기50%·CLIMB RSI50%; 걷기 시작 시 물체는 사람 주위1~10m/360도, RSI 시작 시 참조 object XY/yaw를 사용한다. 물체는 bbox 반높이에 바닥 배치한다. 별도 RSI cache/정면/support 필터를 쓰지 않는다.

목표는 물체 원점 XY·물체Z+metadata tarSitPosZ+char_h. 원본 compute_climb_reward를 AST 동일하게 복사한다. 접근속도.2+목표근접.5+발높이.3, 원본속도벌점·power.0005, task/AMP.5/.5. 원본 CLIMB의 낙상 및 IET 함수 사용: episodeLength300, 성공거리.2m 이내 누적59frame, 평가 IET 없음.

원본 dataset_amass_climb.yaml 그대로 사용: loco/climb/climbNoRSI 전문가.3/.4/.3, 초기화.5/.5/0. AMP family는 CLIMB 채널만 사용한다. 모델 입력은 기존 SIT FT와 같은35D adapter를 통과하며 backrest/seat 추가 입력은0이다.

실행 로그/산출물과 PID는 project runs/tokenhsi_climb_78k_20261024/active_run.json을 따른다. 검증 결과는 changelog.md에 기록한다.

2026-10-25 최종20k평가완료:512환경/seed0/3repeat,성공80.66/80.86/80.08%,0회차제외평균80.47%,성공최소거리평균3.638cm. 독립trace재채점통과. 상세 `../../output_etc/check/tokenhsi_climb_20k_official_20261025/REPORT.md`,평가기 `../../evaluation/tokenhsi_climb/README.md`.
