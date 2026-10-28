# TokenHSI native Table evaluation of our10k policy

原본 `HumanoidAdaptCarryBox2Objs`/환경 YAML/`trans_players.run_eval`을 고정 사본에서 사용하고 행동 정책만 우리 Table residual10k로 바꾼다. `policy.py`는 native 상태를 우리280D관측으로 변환하고171tensor/전체scene RMS를strict복원한다. CPU기존관측생성기와bitwise동등,평가전후동결검증을수행한다. `worker.py`는 원본loop/seed/simulator함수body를AST로읽어그대로실행하고trace만기록한다. `validate.py`는별도NumPy채점기로위치+직립/종료frame/첫episode/precision을재검증한다. `test_validate.py`는직립실패·종료frame성공·다음episode누수·통계변조를검사한다.

512환경/test9/seed0/연속3repeat/600step(20초),위치≤20cm+원본updot≥0.86615. 반복표준편차를독립seed표준편차와구분한다. 모델관측은우리모델의30Dobject/GTA/HOLDING_AT를유지한다. 원본환경은원논문과달리9×9m목표를쓴다. 논문83.6%그대로재현이라고표현하지않는다.

2026-10-28 실행사본·명령(프로젝트루트):

```bash
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi118 bash runs/table_residual_10k_tokenhsi_native_20261028/run.sh
```

이미완료된출력폴더에서재실행하지않는다. 다음평가는새실험사본/출력폴더를만든다. 현재결과는 `runs/table_residual_10k_tokenhsi_native_20261028/REPORT.md`/summary.json/validation.json을확인한다. 전체3회84.44±1.08%,첫회제외1·2회84.47±1.52%. 기존30초위치-only90.89±3.03%와조건/표준편차유형이다르다.

Native loader는asset_root="./"와파일명을직접붙이므로object root는원본상대경로 `dataset_dynamic_objects`를유지한다. 실제데이터는기존symlink를통해참조한다. 절대경로는asset None/actor-1/CUDA오류로이어지므로actor index범위를검증한다. baseline/실행중Chair학습은수정하지않는다.
