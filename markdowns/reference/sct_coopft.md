# SCT + Coop. FT

`sct_coopft`는 `stage2_900` 기반의 일반 PPO+AMP 협력 baseline이다. 환경·관측 전처리·물리·보상·대기 감점·샘플링을 공유하며 FSM은 사용하지 않는다. 환경 YAML의 `human_relation_fusion`은 공유 환경 계약을 보존하며, 실제 협력 모듈은 학습 YAML의 전용 `amp_multi_agent_sct_coopft` builder가 선택한다. Teacher와 행동 증류는 사용하지 않는다.

## 모델

Stage1 26k에서 초기화한다. SCT actor encoder·입력 RMS는 고정하고 actor 협력 모듈·action head를 PPO로 학습한다. Critic encoder·협력 모듈·value head와 AMP discriminator는 일반 Stage2처럼 함께 학습한다. Sigma는 원본 설정의 `learn_sigma: false`를 유지한다. Actor와 critic의 협력 모듈은 모두 `MeanCooperation`이다.

각 block은 `LayerNorm → [자기64; partner평균64; 전체 다른 사람평균64; 관계one-hot3] → Linear(195,80) → ReLU → Linear(80,64) → residual → LayerNorm → Linear(64,128) → ReLU → Linear(128,64) → residual`이다. 4개 block은 별도 가중치를 가지며 모든 사람이 공유한다. 각 block에서 갱신된 feature를 다시 집계한다. 두 residual branch의 마지막 Linear를 0으로 초기화한다.

- Partner는 enabling 후행자의 선행자, coupled 참여자의 자신을 제외한 같은 그룹이다. Independent·enabling 선행자는 비어 있다.
- 전체 다른 사람은 자신을 제외한 모든 사람이다. 두 평균은 대상이 없으면 64차원 0이며 없는 사람을 평균 분모에 넣지 않는다.
- 관계는 `NONE=[1,0,0]`, `ENABLING=[0,1,0]`, `COUPLED=[0,0,1]`이다. 별도 relation embedding·attention·dependency bias/fusion은 없다.
- 협력 모듈은 150,784개 파라미터로 DCT의 146,944개보다 약2.6% 크다. SCT와 action head의 구조·전이 가중치는 같다.

## 학습과 실행

`stage2_900`의 일반 `ma` trainer를 재사용한다. PPO actor/critic loss와 AMP discriminator loss로 학습하며 task/AMP reward 가중치는 각각0.5다. 학습률2e-5·32step rollout·scene minibatch4,096·mini_epochs6·최대1,000,000epoch 등 모든 학습 설정은 원본과 동일하다. 기본4명·1024환경·8토큰·900step이다.

전이는 `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`를 사용한다. 현재 checkout은 기존 jointpush repo의 동일 PTH에 심링크로 연결하며 공유 motion 데이터도 심링크로 읽는다. 출력은 `output/sct_coopft/SCTCoopFT_*`, 확인 실행은 `output_etc/check/`다.

Train/test/VNC/eval은 **GPU6/7만** 허용하며 기본6이다. 기본 명령은 [config](../config.md), 추가 확인·resume·공간 평가 명령은 [USAGE](../USAGE.md#sct--coop-ft)에 둔다. 기존에 잘못 시작한 증류 run은 PPO 결과로 사용하지 않으며 그 checkpoint로 resume하는 것도 차단한다.
