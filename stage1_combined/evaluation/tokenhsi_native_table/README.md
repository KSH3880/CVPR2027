# TokenHSI native Table evaluation of our10k policy

## 공개 TokenHSI Table 모델의 원본 평가

공개 `ckpt_stage2_objShape_table.pth`(내부epoch40000)를 원본환경·20초·test9·512환경·seed0·연속3회로 측정하려면 저장소루트에서 다음을 실행한다.

```bash
PYTHONPATH=stage1_combined /home/injesus1010/anaconda3/envs/tokenhsi118/bin/python \
  -m evaluation.tokenhsi_native_table.run_public --gpu 0
```

`run_public.py`는 읽기전용baseline의 실행·env·정책·관측·seed/sim·player소스를 새 코드사본으로 고정하고 `public_worker.py`는 관측/행동을 바꾸지 않는 기록wrapper와 strict복원·가중치/RMS/AMP RMS 불변검사를 추가한다. 원본`get_action` 전체를 유지하므로 deterministic행동을 선택하는 과정의 RNG소비도 원본과 같다. 우리280D관측 변환을 사용하지 않고 원본265D관측/RMS와102tensor를 복원한다. RTX5090용math attention을 사용한다.

성공은 물체중심–목표3D거리≤20cm AND up-dot≥0.86615이며 첫episode 종료frame까지ever다. 시간은600설정step/599action, 시드는반복사이재설정하지 않는다. 대표값은repeat1/2 평균±반복표본std, repeat0와3회전체통계도 보존한다. `--checkpoint`, `--base-checkpoint`로 로컬배포캐시 기본파일을 바꿀 수 있고 `--prepare-only`는 고정사본만 만든다. `--smoke`는4환경의같은20초×3repeat연결검증이며성능결과가아니다.

결과는 `output_etc/check/hold_at_eval/tokenhsi_table_native20s_<시각>/`의 REPORT/summary/validation/원본metrics와 measurement/초기상태·trace에 저장한다. 진행은 결과폴더state.json/measurement/progress.json 및 저장소루트`runs/tokenhsi_table_native20s_active.json`에서 확인한다. 기존측정 결과를 덮어쓰지 않는다. 공개코드의9×9m목표조건이므로 논문83.6%와 같은 조건의 재현이라고 표현하지 않는다.

## 우리 residual10k를 원본 환경에서 평가한 기존 기록

原본 `HumanoidAdaptCarryBox2Objs`/환경 YAML/`trans_players.run_eval`을 고정 사본에서 사용하고 행동 정책만 우리 Table residual10k로 바꾼다. `policy.py`는 native 상태를 우리280D관측으로 변환하고171tensor/전체scene RMS를strict복원한다. CPU기존관측생성기와bitwise동등,평가전후동결검증을수행한다. `worker.py`는 원본loop/seed/simulator함수body를AST로읽어그대로실행하고trace만기록한다. `validate.py`는별도NumPy채점기로위치+직립/종료frame/첫episode/precision을재검증한다. `test_validate.py`는직립실패·종료frame성공·다음episode누수·통계변조를검사한다.

512환경/test9/seed0/연속3repeat/600step(20초),위치≤20cm+원본updot≥0.86615. 반복표준편차를독립seed표준편차와구분한다. 모델관측은우리모델의30Dobject/GTA/HOLDING_AT를유지한다. 원본환경은원논문과달리9×9m목표를쓴다. 논문83.6%그대로재현이라고표현하지않는다.

2026-10-28 실행사본·명령(프로젝트루트):

```bash
TOKENHSI_GPU=0 TOKENHSI_CONDA_ENV=tokenhsi118 bash runs/table_residual_10k_tokenhsi_native_20261028/run.sh
```

이미완료된출력폴더에서재실행하지않는다. 다음평가는새실험사본/출력폴더를만든다. 현재결과는 `runs/table_residual_10k_tokenhsi_native_20261028/REPORT.md`/summary.json/validation.json을확인한다. 전체3회84.44±1.08%,첫회제외1·2회84.47±1.52%. 기존30초위치-only90.89±3.03%와조건/표준편차유형이다르다.

Native loader는asset_root="./"와파일명을직접붙이므로object root는원본상대경로 `dataset_dynamic_objects`를유지한다. 실제데이터는기존symlink를통해참조한다. 절대경로는asset None/actor-1/CUDA오류로이어지므로actor index범위를검증한다. baseline/실행중Chair학습은수정하지않는다.
