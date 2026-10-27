# Table HOLDING+AT

기존 chair HOLDING+AT의 1명/동적물체1개 환경을 상속한다. 원본 `TokenHSI/tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_box2objs_table.yaml` 및 `humanoid_adapt_carry_box2objs.py`를 읽기 전용으로 대조했다.

- 카테고리 `Table_Circle`. 환경마다 train 모델 균등 복원 추출, 원본 URDF/질량/동적 물리 사용.
- 초기 거리 1~5m, 바닥+5cm, **yaw 0~360도 균등**. chair의 사람을 향하는 정면 보정은 적용하지 않는다. 목표 XY ±4.5m, 사람/물체와 squared distance >=2, 바닥 배치.
- 원본 Table YAML의 randomHeightProb=1/maxTopSurfaceHeight=1.35에도 실제 코드에서는 Table_Circle의 높이 무작위화를 제외한다. 이 구현도 바닥만 사용한다. 상속한 stage1 검증 계약의 maxTopSurfaceHeight=1.2는 유지하며 이 reset에서는 사용하지 않는다.
- 원본 초기화 loco_more/loco=50:50, AMP expert loco_more/loco/pickUp/carryWith/putDown=.5/.2/.1/.1/.1. 전용 `dataset_table_hold_at.yaml`은 원본 모션 목록의 해당 여섯 skill을 보존하며 SIT/CLIMB를 제외한다.
- 기존 chair와 같은 HOLDING_AT100% 관계 보상, 30D 물체 관측, 78k strict 전이/normalizer, 전체층 FT/lr2e-5/20k/500회 저장, 2048환경/spacing5/600step. 원본 TokenHSI의 transformer 정책·carry 보상·4096환경 학습 전체를 복제한 설정은 아니다.
- train/test/VNC는 `table_hold_at_full_ft_{train,test,vnc}.sh`. test는 checkpoint의 환경 설정을 복원하고 asset split만 test로 전환한다. `HOLD_AT_OBJECT_ROOT`는 train/Table_Circle 및 test/Table_Circle을 포함하는 dataset_dynamic_objects 루트다. 평가 환경 수/반복은 `TABLE_TEST_ENVS`/`TABLE_REPEATS`.

현재 데이터: `tokenhsi/data/dataset_dynamic_objects` → `/home/injesus1010/repos/TokenHSI/tokenhsi/data/dataset_dynamic_objects`, train21/test9. config/URDF/mesh 및 모션 경로 모두 존재함을 확인했다. 공유 데이터는 수정하지 않는다.

검증: `tokenhsi/tests/check_table_hold_at_cpu.py` 통과(78k 165tensor strict/normalizer/finite CPU 추론/보상 동일/부분reset/무작위yaw/목표분포). shell 구문 검사 통과. 로그는 상위 `runs/table_hold_at_implementation_20261026/output_etc/check/cpu.log`. 현재 세션은 NVIDIA driver 접근 불가로 GPU 시뮬레이션·PPO update 미검증, 본학습 미실행. 이 로컬 설치의 conda 환경은 `tokenhsi118`이므로 실행 시 `TOKENHSI_CONDA_ENV=tokenhsi118`로 지정한다.

## 2026-10-26 실행

사용자 요청 후 샌드박스 밖 GPU 접근 확인. GPU0 RTX5090/tokenhsi118/PYTORCH_JIT=0에서 2048환경 1회 PPO·저장·자연종료/finite/양tokenizer 업데이트 검증 통과 후 본학습 시작. 현재 실행은 `../../table_hold_at_active_run.json`, 상세는 상위 `runs/table_hold_at_20261026_1606/REPORT.md`를 따른다. 앞 문단의 GPU 미검증 상태를 대체한다.
