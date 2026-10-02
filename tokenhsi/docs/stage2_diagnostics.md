# Stage 2 진단 도구

실측·RSI 검사·attention 분석의 재현 명령을 모은다. 모든 명령은 저장소 루트에서 실행한다. 본학습·viewer 명령은 [config_stage2.md](../../markdowns/config_stage2.md)를 따른다.

GPU 번호는 각 명령의 `TOKENHSI_GPU`로 지정한다. `$GPU` 예제를 사용할 때는 먼저 사용할 번호를 설정한다.

Rescue CA 학습 추세는 아래 진단으로 측정한다. 지정 run의 latest pth를 처음 실행 때 output에 고정하며, 500 epoch 간격의 모든 checkpoint를 같은 최신-policy 관측 bank에서 비교한다. 실제 rollout은 epoch 500/2000/4000/6000/8000/10000/latest, CLIMB·SIT·STACK 양쪽 역할·독립 대조군 각 16환경×3seed·600step이다. 최신에는 CA 제거·균등 routing·동료 edge 제거 rollout도 수행한다. 기존 학습 YAML·프로세스에는 적용하지 않으며 `report.html`·PNG·JSON·NPZ를 별도 저장한다. 새로운 latest를 분석할 때는 다른 `--output`을 지정한다.

```bash
TOKENHSI_GPU=0 bash -c '
source tokenhsi/scripts/multi_agent/runtime_env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tokenhsi/scripts/multi_agent/analyze_stage2_attention.py \
  --run output/approach_stage2_rescue_klclimb50/ApproachStage2RescueKLClimb50_01-20-37-44 \
  --output output/stage2_attention_trends
'
```

같은 받침을 공유하는 ON_TOP/SIT/CLIMB 6조합의 정적 공간 측정은 아래 명령을 사용한다. 학습 환경을 유지하며 별도 CPU PhysX와 실제 충돌 형상을 사용한다. 결과·조건·한계는 `output/shared_support_measurement/`에 저장하며, 정책 성공률·운반 경로·균형 안정성 검증과 구분한다. 추가 라이브러리는 전용 폴더에만 설치한다.

```bash
TOKENHSI_GPU="$GPU" bash -c '
source tokenhsi/scripts/multi_agent/runtime_env.sh
MEASURE_DEPS=output/shared_support_measurement/deps
python -m pip install --no-deps --upgrade --target "$MEASURE_DEPS" numpy==1.24.4 python-fcl==0.7.0.6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tokenhsi/scripts/multi_agent/measure_shared_support.py --deps "$MEASURE_DEPS"
'
```

공유 9조합의 RSI·상자 크기 진단은 아래 명령으로 재현한다. 기존 Rescue 설정에 여섯 조합을 주입하는 진단이며, 새 Shared9 본학습은 위 전용 config·wrapper를 사용한다. Stage 1 epoch 8000·학습 0회로 평가하고, 공유 SIT/CLIMB을 loco RSI로 제한한 대안도 비교한다. CPU PhysX 접촉 확인은 GPU 진단과 별도 프로세스에서 실행한다. 결과와 크기 권장안은 `output/shared_scenario_audit/README.md`에 있다.

```bash
TOKENHSI_GPU="$GPU" bash -c '
source tokenhsi/scripts/multi_agent/runtime_env.sh
MEASURE_DEPS=output/shared_support_measurement/deps
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python tokenhsi/scripts/multi_agent/measure_shared_support.py --deps "$MEASURE_DEPS" --output output/shared_scenario_audit/geometry --payload-sides 0.20 0.25 0.30 0.40 --payload-height 0.30 --heights 0.30 0.40 0.45 0.50 --max-side 1.25
python tokenhsi/scripts/multi_agent/measure_shared_support.py --deps "$MEASURE_DEPS" --output output/shared_scenario_audit/geometry35 --payload-sides 0.35 --payload-height 0.30 --heights 0.30 0.40 0.45 0.50 --max-side 1.25
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --output output/shared_scenario_audit/reset_detailed
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --verify-snapshots output/shared_scenario_audit/reset_detailed/reset_physx_snapshots.json --output output/shared_scenario_audit/reset_detailed
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --only-resets --shared-loco --output output/shared_scenario_audit/reset_shared_loco
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --verify-snapshots output/shared_scenario_audit/reset_shared_loco/reset_physx_snapshots.json --output output/shared_scenario_audit/reset_shared_loco
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --case-set midrange --resets 0 --output output/shared_scenario_audit/midrange
'
```

아래는 크기 43설정과 all-loco 초기 배치의 진단이다. 운반을 먼저 측정한 최신 공유 범위는 `output/shared_scenario_audit/transport_grid/README.md`를 따른다. 9조합 본학습은 위 Shared9 config·전용 wrapper를 사용한다.

```bash
TOKENHSI_GPU="$GPU" bash -c '
source tokenhsi/scripts/multi_agent/runtime_env.sh
MEASURE_DEPS=output/shared_support_measurement/deps
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python tokenhsi/scripts/multi_agent/measure_shared_rectangles.py --deps "$MEASURE_DEPS"
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --case-set carryable --resets 0 --output output/shared_scenario_audit/carryable
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --case-set carryable --only-resets --all-loco --resets 3 --output output/shared_scenario_audit/reset_carryable_loco
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --verify-snapshots output/shared_scenario_audit/reset_carryable_loco/reset_physx_snapshots.json --output output/shared_scenario_audit/reset_carryable_loco
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --case-set carryable --cases current_range support60_75_50_payload30 support60_75_50_payload35 support60_80_50_payload35 support65_65_50_payload30 --only-resets --all-loco --resets 1 --snapshot-all --output output/shared_scenario_audit/loco_unbiased
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --verify-snapshots output/shared_scenario_audit/loco_unbiased/reset_physx_snapshots.json --output output/shared_scenario_audit/loco_unbiased
'
```

Stage 1 epoch 8000의 독립 운반 515크기를 먼저 검사하고, 그 결과 안에서 공유 SIT/CLIMB/ON_TOP 6조합의 공간과 실제 rollout을 비교한다. 진단의 운반·coverage 필터는 기존 학습 성공 조건을 변경하지 않는다.

```bash
TOKENHSI_GPU="$GPU" bash -c '
source tokenhsi/scripts/multi_agent/runtime_env.sh
MEASURE_DEPS=output/shared_support_measurement/deps
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --case-set transport_grid --resets 0 --output output/shared_scenario_audit/transport_grid
python tokenhsi/scripts/multi_agent/summarize_transport_sharing.py --select
python tokenhsi/scripts/multi_agent/measure_shared_rectangles.py --deps "$MEASURE_DEPS" --supports-file output/shared_scenario_audit/transport_grid/supports_for_geometry.json --payload-sides 0.25 0.30 0.35 0.40 --clearances 0.0 0.04 --output output/shared_scenario_audit/transport_geometry
python tokenhsi/scripts/multi_agent/measure_shared_rectangles.py --deps "$MEASURE_DEPS" --supports-file output/shared_scenario_audit/transport_grid/supports_tall_payload.json --payload-sides 0.30 0.35 --payload-heights 0.40 0.45 0.50 --clearances 0.0 0.04 --output output/shared_scenario_audit/transport_geometry_tall
python tokenhsi/scripts/multi_agent/measure_shared_rectangles.py --deps "$MEASURE_DEPS" --supports-file output/shared_scenario_audit/transport_grid/supports_thin.json --payload-sides 0.30 0.35 --payload-heights 0.30 0.40 0.45 0.50 --clearances 0.0 0.04 --output output/shared_scenario_audit/transport_geometry_thin
python tokenhsi/scripts/multi_agent/measure_shared_rectangles.py --deps "$MEASURE_DEPS" --verify-results output/shared_scenario_audit/transport_geometry_tall/results.json --output output/shared_scenario_audit/transport_geometry_verified
python tokenhsi/scripts/multi_agent/measure_shared_rectangles.py --deps "$MEASURE_DEPS" --verify-results output/shared_scenario_audit/transport_geometry_thin/results.json --verify-supports-file output/shared_scenario_audit/transport_grid/verify_thin_supports.json --output output/shared_scenario_audit/transport_geometry_thin_verified
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --size-cases output/shared_scenario_audit/transport_grid/sharing_cases.json --sharing-only --resets 0 --output output/shared_scenario_audit/sharing_after_transport
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --size-cases output/shared_scenario_audit/transport_grid/sharing_thin_cases.json --sharing-only --resets 0 --output output/shared_scenario_audit/sharing_thin
python tokenhsi/scripts/multi_agent/summarize_transport_sharing.py
'
```

역할별 최대 크기(받침60×70×50cm·올리는 상자30×30×50cm)의 독립 4과제와 연합 9조합 진단은 아래 명령을 사용한다. 학습 config가 아닌 진단 입력이며 결과는 `output/shared_scenario_audit/max_sizes/README.md`에 있다.

```bash
TOKENHSI_GPU="$GPU" bash -c '
source tokenhsi/scripts/multi_agent/runtime_env.sh
MEASURE_DEPS=output/shared_support_measurement/deps
python tokenhsi/scripts/multi_agent/summarize_transport_sharing.py --max-sizes
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --size-cases output/shared_scenario_audit/max_sizes/cases.json --independent-only --resets 0 --output output/shared_scenario_audit/max_sizes/independent
python tokenhsi/scripts/multi_agent/audit_shared_scenarios.py --deps "$MEASURE_DEPS" --size-cases output/shared_scenario_audit/max_sizes/cases.json --resets 0 --output output/shared_scenario_audit/max_sizes/all_nine
'
```
