# Box·Chair·Table 혼합 전체 FT

`box_chair_table_mixed_full_ft`는 기존 `chair_mixed_full_ft`의 native 환경에 table을 추가한 설정이다. Box·Chair·Table 목표 비율은 각1/3이다. 2048환경에서는683개가 Box 목표2개,1365개가 chair/table 목표 하나씩·주변box2개다. 전체4096개 목표는 Box1366/Chair1365/Table1365개로 정수 오차만 있다. 물체 배정은 환경 생성 때 고정되며 pose는 매episode 갱신한다.

사용자 선택에 따라 모든 물체는 SIT/CLIMB50:50만 학습하며 AT·ON_TOP은 제외한다. Box의 크기·reward·size-aware RSI는 stage1_v2_fixed 경로를 유지한다. AMP도 loco/sit/climb 계열로 맞춘다. native 환경에서 Chair SIT/CLIMB는 seat+.10/.94, Table SIT는 bbox상판+.10, CLIMB는 metadata+.94다. 소스78k, 새35D tokenizer, 전체 레이어 학습, lr2e-5,20kepochs는 유지한다.

native RSI는 `output/rsi_cache_native_supported_20261023`의 검사 통과 후보를50% 확률로 사용한다. 비어 있는 asset/task는 loco로 대체한다. 기존 Box size-aware RSI와 함께 사용하며, native 후보의 관절한계 보정과 일치하는9프레임 AMP history를 재계산해 reset에 적용한다. 초기 yaw·XY는 실제 target 물체 pose에 맞춘다. 기존 장면 검사에 두 사람 body-point 최소거리15cm 검사를 추가했다.

실행 명령은 [config](../config.md)의 동일 이름 행/명령을 사용한다.

## Tokenizer 입력

Box·Table은 기존30D tokenizer를 공유하고 Chair만 추가 좌판·방향 입력의35D tokenizer를 사용한다. `chairFT.table_input_bbox: metadata_top_box_tokenizer`에서 Table 관측의8개 bbox 꼭짓점 중 윗면Z만 `tarSitPos.z`로 맞춘다. 아래면·XY·object pose와 속도는 유지한다. 물리 mesh/bbox, RSI geometry, reward·성공·평가 목표는 바꾸지 않는다. flag는 입력 복사본에서만0으로 만들어 기존Box branch로 보낸다. 옵션이 없는 과거 checkpoint의 입력/라우팅은 유지된다. 학습 Table10은 원래 두 높이가 같고 held-out Table8의 차이는 최대2.03cm다.

## 지지 상태를 반영한 RSI 수정 (2026-10-23)

기존 constant-Z retarget은 물체 높이 차이를 접근 자세까지 적용하여 양발이 뜨는 후보를 만들었다. `rsi_native/repair_support.py`는 원본 FK상 지면을 딛던 nonphase 후보의 최저 발바닥을5mm에 맞추고 해당 발의 수직속도를 제거한다. phase 후보는 물체 지지면을 유지한다. 새 static 검사에서는 발/골반의 지면 또는 native mesh 수평면 근접(3cm), 관통 한계, 이후0.1/0.5초 PhysX 안정성을 모두 요구한다. history 각 프레임에도 같은 지면/속도 보정을 적용한다. 물체 높이가 달라 그대로 적용할 수 없는 후보는 관통·지지 검사에서 제외한다.

기존 Box RSI도 `require_support`에서 native 발 mesh·골반 sphere와 바닥/위쪽 Box 면 근접을 확인하고 부유한 reset을 재추출한다. 모델·reward 변경은 없다. native RSI50%, 모션 가중치 유지·late draw70%/full draw30%는 유지한다. late phase가 없는 모션은 안전한 full pool로 대체하므로 실제 nonphase 비율은30%와 다를 수 있다.

최종 통합 검증: CPU19개·2048환경1epoch rc0, native활성1381·nonphase745·높이기준중간283, Box 부유 reset104건 재추출, AMP history오차0/전체finite. 근접/표본 검사와 짧은 PD 검사이며 장기 policy 안정성을 보장하지 않는다. 결과 `output_etc/check/native_rsi_supported_78k_smoke/`.

## 의자 정면·발 표면 교차 검증 및 학습 보류

최신 bank는 `output/rsi_cache_native_front_foot_20261023`이다. `chair_front_only`는 RSI뿐 아니라 의자 과제의 loco 초기화도 정면으로 제한한다. 외부 접근은±45도, 중앙12cm는 착석/상판 위 상태로 허용한다. 뒤쪽 접근 후보를180도 돌린 뒤 mesh/물리 검사를 다시 통과시킨다.

`foot_surface.py`는 foot-edge/object-triangle 교차2mm를 검사하고 물리1·2·3·6·30step에도 적용한다. `NativeFootGuard`는 실제 reset에서 모든 native 물체와 두 사람의 발을 검사한다. 테스트 자산은 catalog에서 필요할 때 불러온다. `validate_reset.py`는 학습용 reset을 사용하되 정책 행동·optimizer 없이 검증한다. 최종 review는 `output_etc/check/native_rsi_front_foot_review_validation/REPORT.md`다. 사용자 지시로 **학습은 보류 중**이다.


## 관절 IK RSI 검증 (2026-10-23, 학습 보류)

`rsi_native/ik.py`에서 실패 phase의 다리18DOF/root를 보정했다. 236개 중 정적164·단독 PhysX67·9프레임 history45개 통과(Chair SIT30/CLIMB5, Table SIT6/CLIMB4). 기존 완료 RSI가 없던 Table SIT1종이 복구됐지만 전체 탈락 pose를 해결한 것은 아니다. 기존 실행 bank와 YAML을 교체하지 않았다.

최종 결과와 한계: `output_etc/check/native_rsi_ik_physics_20261023_v3/REPORT.md`. 별도 history bank: `output_etc/check/native_rsi_ik_history_20261023_v3`. BEFORE/AFTER snapshot: `output_etc/check/native_rsi_ik_review_20261023_v3`, 로컬 viewer: `output_etc/check/native_rsi_ik_local_view_20261023_v3`. N/P로 물리 통과 pose의 보정 전후를 비교하고 SPACE 고정PD/R reset. 본학습은 사용자 RSI 검토 전까지 보류한다.
