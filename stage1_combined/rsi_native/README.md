# Native chair/table RSI 후보 구성·검사

기존 box 참조 SIT20/CLIMB7개를 학습용 native chair49/table10종으로 rigid retarget한다. 원본 `object_state`의 z는 지면 원점이다. 지지면은 안정된 SIT 골반 높이−10cm / CLIMB 발 높이에서 추정한다. 물체 좌판·상판과 facing에 맞춰 root/body 위치·회전·속도를 함께 변환한다.

`generate.py`는 원본 관절각 후보를 보존한다. `repair.py`는 실제32DOF 한계로 clamp하고 최대 보정량30/100rad를 넘는 후보를 제외한다. 관절속도와 전신 FK를 다시 계산한다. 1축 팔꿈치로 재현되지 않는 참조 손 위치도 재계산한다. 원본 관절각·보정량은 별도 필드로 남긴다. 보정 후 정적 통과를 위해 추가로 전방0/7.5/15cm, 위0/2.5/5/7.5/10cm의 유한 탐색을 한다. 초기 goal 오차20cm 이하인 phase 후보만 추가 이동을 허용한다.

`geometry.py`는 native URDF collision mesh에 sphere/capsule 중심선과 발 mesh 표본을 대조한다. 겹침 proxy15mm·바닥20mm가 한계다. 다수 물체 mesh가 닫혀 있지 않아 parity 기반 검사는 정확한 체적 교차 판정이 아니다.

`launch_audit.py` / `audit_backend.py`는 원래 학습 환경을 수정하지 않고 별도 task/player를 주입한다. 실제 native VHACD 물체와2048환경(1명·1물체), 초기 관절각 PD 고정으로0.1초/0.5초 검사를 한다. optimizer는 실행하지 않는다. head200N, root이동15/30cm, 속도변화3m/s, 최종속도2m/s, 최종geometry/floor, phase 최종goal25cm를 검사한다. `usable`은 static+short+hold+phase 모두 통과한 후보다.

검사 통과는 정책 안정성·정확한 무충돌을 보장하지 않는다. 두 사람·주변 물체 충돌, 물체yaw별 reset, task별10-frame AMP history의 보정 일관성은 별도 통합 검증이 필요하다. 초기 audit bank는 후보 산출물이다. `build_training_bank.py`가 관절 보정과 동일한9프레임 AMP history를 계산한 뒤, `utils/native_rsi.py`가 혼합 FT의 실제 물체 yaw/XY에 맞춰 reset한다. held-out 물체는 사용하지 않는다.

실행은 stage1_combined에서 runtime을 먼저 적용한다. 출력은 새 `output_etc/check/` 경로를 사용한다. 아래는 CPU 생성·보정 명령이며 학습 명령이 아니다.

```bash
. tokenhsi/scripts/multi_agent/runtime_env.sh
export PYTHONPATH="$TOKENHSI_ROOT:$TOKENHSI_ROOT/tokenhsi"
PYTORCH_JIT=0 python rsi_native/generate.py --output output_etc/check/native_rsi_raw_NEW
python rsi_native/repair.py --source output_etc/check/native_rsi_raw_NEW --output output_etc/check/native_rsi_repaired_NEW
```

물리 검사는 `output_etc/check/native_rsi_physics_20261023_v8/env.yaml`과 동일한1명/1물체 audit 설정 및 `launch_audit.py --task NativeRsiAuditTask`를 사용한다. YAML template probability의 key 순서를 보존해야 한다. 최종 결과·허용 후보·그림은 해당 audit 폴더의 `REPORT.md`, `physics_report.json`, `audited_bank/`, `preview.png`에 둔다. 최초 raw 물리 검사는 별도 `native_rsi_physics_20261023/`에 보존한다.

## 로컬 Table SIT 착석 확인

```bash
. tokenhsi/scripts/multi_agent/runtime_env.sh
. tokenhsi/scripts/multi_agent/gui_gpu_env.sh
python rsi_native/view_table_sit.py --output output_etc/check/table_sit_view_NEW
```

검사 통과 RSI가 있는 Table9종을 표시한다. 학습 정책 rollout이 아닌 초기 착석 자세이며 시작은 일시정지다. 빨강=상판+.10m 목표, 노랑=점 성공거리20cm 구, 청록=골반. N/P는 다음/이전 Table, SPACE는 고정 초기관절각 PD 물리 진행/정지, R은 초기자세 복원이다. `scene.json`·`viewer.png`를 저장한다. 성공구는 점 기준 시각화이며 학습의 다른 feet/영역 조건까지 표현하지 않는다.

## 78k 실제 학습 reset 확인

`build_training_bank.py`는 source clip/time에서9프레임을 다시 읽고 clamp/FK/velocity·동일 retarget transform으로 history를 만든다. rootHeightObs=false, heading-local AMP이므로 이후 물체yaw/XY 변환에도 AMP는 불변이다. 실행용 bank는 `output/rsi_cache_native_chair_table_20261023/`다. 실험 초기화에서 mesh·AMP 크기/유한값을 검사하며 확률50%, 누락asset/task는 loco 대체다. 기존 장면 재시도와 두 사람15cm body-point 거리를 적용한다.

실제2048환경 reset snapshot: `output_etc/check/native_rsi_78k_train_smoke/`. native활성1285슬롯, history cache 오차0, 전체AMP finite, source epoch78000. 장면 viewer:

```bash
. tokenhsi/scripts/multi_agent/runtime_env.sh
. tokenhsi/scripts/multi_agent/gui_gpu_env.sh
python rsi_native/view_reset_snapshot.py --snapshot output_etc/check/native_rsi_78k_train_smoke --output output_etc/check/RSI_VIEW_NEW
```

N/P는 Box SIT→Box CLIMB→Table SIT→Table CLIMB→Chair SIT→Chair CLIMB 실제 학습 reset을 전환한다. SPACE는 고정관절 PD 물리 진행, R은 자세 복원이다. 두 사람·주변4물체를 포함하며 정책 추론이 아니다. 체크포인트별 policy 안정성·장기 성공은 별도 평가해야 한다.

## 부유 RSI 수정·최종 실행 bank

`repair_support.py`와 `support.py`가 source-grounded nonphase 발을5mm에 투영하고 수직속도를 정합한다. 수평 물체 면 또는 지면의 발/골반 근접3cm를 요구하며, 벽면 접촉은 지지로 인정하지 않는다. `audit_backend.py`는 수정 candidate를2048환경에서0.1/0.5초 다시 검사하고 최종 지지 근접도 요구한다. `build_training_bank.py --full-motion`가 해당 후보만으로 매 history 프레임을 동일하게 보정한다. 최종 bank는 `output/rsi_cache_native_supported_20261023`이며 앞선 phase-only/full-motion bank는 실패·과거 결과로 보존한다.

Box 원래 RSI도 `utils/rsi_support.py`의 발 mesh·골반 sphere/바닥/위쪽 Box면 근접 검사로 부유한 reset을 재추출한다. 실제 초기화 검증과 snapshot은 `output_etc/check/native_rsi_supported_78k_smoke/`, 로컬 viewer 산출물은 `output_etc/check/native_rsi_supported_view/`다.

## 최신 검증·review (학습 보류)

`front_filter.py`는 의자 접근을 정면±45도로 바꾸고 후보를 재검사한다. `foot_surface.py`는 열린/얇은 상판에도 foot-edge/mesh-triangle 교차 검사를 적용한다. `utils/native_foot_guard.py`는 실제2명·4물체 장면의 모든 native 물체에 대해 발을 검사한다.

`validate_reset.py`는 훈련 reset/RSI/AMP를 그대로 사용하지만 optimizer·정책 행동은0이다. 최종 결과는 `output_etc/check/native_rsi_front_foot_review_validation/`, 실행 bank는 `output/rsi_cache_native_front_foot_20261023`이다. 사용자 지시로 본학습은 시작하지 않았다.

```bash
. tokenhsi/scripts/multi_agent/runtime_env.sh
. tokenhsi/scripts/multi_agent/gui_gpu_env.sh
python rsi_native/view_reset_snapshot.py --snapshot output_etc/check/native_rsi_front_foot_review_validation --scene-index 6 --output output_etc/check/RSI_REVIEW_NEW
```

10장면에서 접근/완료를 나눠 볼 수 있다. 초록선은 정면, 빨간선은 뒤쪽이다. 중앙 착석/상판 위 상태는 정면 cone의 예외다. 일부 물체/모션의 완료 phase는 엄격한 검사에서 모두 탈락했으므로 안전 접근 후보로 대체한다. 상세 후보 수와 한계는 REPORT를 따른다.

## 관절 접촉 IK 보정 (학습 보류)

`ik.py`는 기존 audit에서 탈락한 완료 자세를 clip별로 골라 양쪽 다리18DOF와 root XYZ를 bounded least-squares로 보정한다. 상체·heading은 유지하고, 관절 변화는 각 축0.1rad(약6도), root 이동은 전체 거리2cm 이내다. 이 범위에서 맞지 않으면 제외한다. 기존 v3의 큰 변경 결과는 사용자 피드백으로 채택하지 않았으며, 새 제한으로 재생성은 아직 하지 않았다. CLIMB는 발바닥을 좌판/상판 footprint와 높이에, SIT는 지면 근처 발만 바닥에 맞춘다. 높은 Table SIT의 매달린 발은 강제로 내리지 않는다. 접촉·충돌·원래 자세 보존은 soft objective이고, 최종 통과는 전신 표본 geometry·foot-edge 교차·지지·goal20cm·의자 정면 조건으로 별도 판정한다. 모든 물체/모션을 맞출 수 있다는 보장은 없다.

원본과 보정 전 상태, 관절/root delta를 함께 보존한다. 결과는 별도 check bank이며 기존 실행 캐시를 덮어쓰지 않는다. PhysX 재검사를 통과한 뒤 `build_training_bank.py`가 같은 관절 delta를 이전9프레임에도 적용하고, 관절한계·속도·FK를 재계산한다. history 중 관통/바닥 침투가 있으면 후보를 제외한다. 이것은 짧은 local window의 constant-offset 보정이며 전체 모션의 전역 IK 최적화는 아니다.

`ik_preview.py --audit <IK 물리 audit> --output output_etc/check/<새 이름>`은 각 물체군/SIT·CLIMB의 통과 후보를 보정 전/후 쌍으로 export한다. `view_reset_snapshot.py`의 N/P로 BEFORE/AFTER를 비교한다. 노란 성공구는 물체당 하나만 표시한다. 원본 실패 자세도 비교용으로 보이므로 BEFORE에서는 관통·불안정이 남아 있다.
