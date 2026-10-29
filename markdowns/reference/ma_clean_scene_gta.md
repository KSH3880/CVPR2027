# 공통 장면 정책

환경마다 사람 H223·물체 O30·목표 T1, 각 토큰 pose7, 사람별 task packet5를 생성한다. pose와 task packet은 RMS를 그대로 통과한다. actor와 critic은 별도 encoder/head다.

`amp_network_builder_ma.py`의64차원·2head·4layer GTA와 `task_role_encoder.py`의 공유 task bias·사람 residual action MLP를 공유한다. tensor 이름과 관측 순서는 기존 Stage1 78000과 동일하다.

환경은 `humanoid_ma_carry.py`에서 관측·모션/AMP·reset·과제 mixin과 `HumanoidMA`를 조합한다. 모션의 native 상대 자세·크기별 물리 screening·과제별 AMP family를 사용한다. 학습기는 장면 단위 PPO minibatch와 사람별 action/reward를 연결한다.

저장본은 `relation_config.yaml`과 reward/task/BEFORE 계약을 포함한다. [structure](../structure.md)를 따른다.
