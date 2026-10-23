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
