# Box·Chair·Table 혼합 전체 FT

`box_chair_table_mixed_full_ft`는 기존 `chair_mixed_full_ft`의 native 환경에 table을 추가한 설정이다. Box·Chair·Table 목표 비율은 각1/3이다. 2048환경에서는683개가 Box 목표2개,1365개가 chair/table 목표 하나씩·주변box2개다. 전체4096개 목표는 Box1366/Chair1365/Table1365개로 정수 오차만 있다. 물체 배정은 환경 생성 때 고정되며 pose는 매episode 갱신한다.

사용자 선택에 따라 모든 물체는 SIT/CLIMB50:50만 학습하며 AT·ON_TOP은 제외한다. Box의 크기·reward·size-aware RSI는 stage1_v2_fixed 경로를 유지한다. AMP도 loco/sit/climb 계열로 맞춘다. native 환경에서 Chair SIT/CLIMB는 seat+.10/.94, Table SIT는 bbox상판+.10, CLIMB는 metadata+.94다. 소스78k, 새35D tokenizer, 전체 레이어 학습, lr2e-5,20kepochs는 유지한다.

native RSI는 `output/rsi_cache_native_chair_table_20261023`의 검사 통과 후보를50% 확률로 사용한다. 비어 있는 asset/task는 loco로 대체한다. 기존 Box size-aware RSI와 함께 사용하며, native 후보의 관절한계 보정과 일치하는9프레임 AMP history를 재계산해 reset에 적용한다. 초기 yaw·XY는 실제 target 물체 pose에 맞춘다. 기존 장면 검사에 두 사람 body-point 최소거리15cm 검사를 추가했다.

실행 명령은 [config](../config.md)의 동일 이름 행/명령을 사용한다.

## Tokenizer 입력

Box·Table은 기존30D tokenizer를 공유하고 Chair만 추가 좌판·방향 입력의35D tokenizer를 사용한다. `chairFT.table_input_bbox: metadata_top_box_tokenizer`에서 Table 관측의8개 bbox 꼭짓점 중 윗면Z만 `tarSitPos.z`로 맞춘다. 아래면·XY·object pose와 속도는 유지한다. 물리 mesh/bbox, RSI geometry, reward·성공·평가 목표는 바꾸지 않는다. flag는 입력 복사본에서만0으로 만들어 기존Box branch로 보낸다. 옵션이 없는 과거 checkpoint의 입력/라우팅은 유지된다. 학습 Table10은 원래 두 높이가 같고 held-out Table8의 차이는 최대2.03cm다.
