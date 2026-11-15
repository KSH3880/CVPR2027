# Stage1 PUSH task attention bias 어블레이션

`stage1_push_no_bias`는 `stage1_push`와 같은 환경·물리·보상·RSI·과제 비율·AMP·PPO에서 actor/critic의 task attention bias만 제거하는 scratch 실험이다. 기준 학습 결과는 `output/stage1_push/`이며 기존 checkpoint를 이어 학습하지 않는다. 기본2명·2048환경·4물체·900step·seed42를 유지한다.

환경 YAML은 `stage1_push.yaml`과 동일하다. 학습 YAML의 차이는 `params.network.transformer.relation_bias: false`와 저장 이름 `Stage1PushNoBias`뿐이다. 본학습 출력은 `output/stage1_push_no_bias/`로 분리한다. 설정과 train/test/VNC·일반/단독 평가 명령은 [config](../config.md)에 있다.

`RelationEncoder`는 bias background·task edge 투영·dense/scatter fusion을 호출하지 않고 attention에 task bias를 전달하지 않는다. GTA의 Q/K/V 변환, 5×64 task embedding과 이를 사용하는 사람 residual Action MLP, attention/MLP의 일반 Linear bias는 유지한다. 물체 padding이 있으면 유효하지 않은 token에 대한 hard mask는 그대로 적용한다.

state dict 구조는 유지하며 사용하지 않는 `bias_projection`만 학습 대상에서 제외한다. task embedding과 Action MLP는 계속 학습한다. 초기 bias projection이0인 기준 모델과 같은 가중치를 사용하면 actor/critic encoder 출력은 일치한다. 학습 후 성능·FPS 개선은 별도로 측정해야 한다.

checkpoint의 `relation_experiment_config.train`에 저장된 `relation_bias`를 읽어 bias/no-bias 재개·viewer·평가 간 혼용을 거부한다. 해당 값이 없는 과거 checkpoint는 bias 활성으로 취급한다. no-bias 재개는 같은 실험의 checkpoint와 전용 wrapper를 사용한다. [추가 실행 옵션](../USAGE.md).
