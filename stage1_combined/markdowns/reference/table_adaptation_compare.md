# Table: frozen encoder residual — 총10,000회

사용자최신요청으로 tokenizer 학습예약은취소하고 **residual만총10,000회** 학습한다. 기존두모드구현과검증은아래에보존한다. 이전 TableTokenHSIFT 학습3544421은중단상태다. 두모드모두원본78k에서시작하며성능하락한FT모델/optimizer를재개하지않는다. Table 원본carry보상·종료·PPO/AMP계수·4096환경·seed42·걷기참조초기화유지,carry RSI미사용. 최초계획은각1000회비교였으나현재는residual만10,000회/100회번호저장이다.

| 항목 | residual | tokenizer |
|---|---|---|
| 기존 actor encoder 전체 | 동결 | 동결(기존object tokenizer도보존) |
| 기존 actor action head/sigma | 동결 | 동결 |
| 기존 critic encoder 전체 | 동결 | 동결 |
| actor학습부분 | encoded64→256→128→32 action residual MLP | 별도30D→256→128→64 object tokenizer |
| 결합 | mu=mu_78k+delta_mu,기존action clip/rescale유지 | 기존object tokenizer의출력자리를새tokenizer로대체 |
| 초기화 | finalLinear=0,뒤에활성함수없음 | 원본78k object tokenizer를복사초기화(랜덤초기화아님) |
| scene RMS | 전부78k로복원후동결 | 동일 |
| critic value head / discriminator | 학습 / fresh학습 | 동일 |

토크나이저를통과한gradient는동결된Transformer를거쳐새tokenizer로전달된다. 기존encoder파라미터는변하지않는다. old/newtoken을동시에추가하는원본TokenHSI구조나42D변환을복제한것은아니며,우리30D/GTA모델에서적응지점만분리한다. TokenHSI의gated compensate action을그대로복제하지않고사용자요청의additive residual을적용한다.

이전실험과의차이:이전에는object RMS와critic encoder도업데이트됐고oldactorobject tokenizer를직접학습했다. 이번에는**전체scene RMS/양쪽기존encoder를동결**하고별도actor모듈만학습한다. 따라서이전12100회실험과한축만다른엄밀한ablation은아니지만,이번두모드끼리는같은설정이다.

## 실행 상태

`../../table_adaptation_active_run.json` 및프로젝트 `runs/table_adaptation_compare_20261027/state.json`이현재상태다. 단일GPU0에서기존residual1000회구간을중단없이완료한뒤,모델/optimizer/scene·AMP·value normalizer/epoch/frame을full-state복원해1001~10000회를이어간다. **tokenizer를실행하는controller는종료했고새controller에는residual만있다.** 현재프로세스메모리의max_epochs는1000으로고정돼있어기존코드/실행중bash는변경하지않는다. resume_code/는별도사본이며full_experiment_name을동일하게지정해같은TensorBoard곡선으로이어간다. 시뮬레이터상태와replay buffer는재초기화되므로물리궤적까지bitwise연속은아니다. 큐파일은수정하지않는다. 동결코드사본 `code/`,분리로그/체크포인트 `residual/train`, `tokenizer/train`. optimizer는각각새로시작한다.

config.md에개별train/test/VNC명령이있다. 현재controller와동시에수동학습을중복실행하지않는다. checkpoint는각모드의것을사용한다. test는checkpoint의train/network설정을복원하므로extra adapter tensor를strict하게로드한다. 공통정식평가 `evaluation/table_hold_at/run_eval.py --checkpoint ...`도추가된network를보존하도록지원했다.

## 검증

- CPU:두모드의초기추론이78k와일치(tokenizer bitwise,residual최대2.1e-7수치오차). 두번backward/Adam step후새모듈gradient비영,기존base weight불변. residual은두번째backward에서첫층gradient도확인했다.
- GPU:각4096환경2회PPO·epoch2저장·rc0. 동결153tensor는78k와exact,**scene RMS6tensor전부exact**,새모듈6tensor학습,finite. 두모드초기construction의root/modelID표본도exact.
- 각모드1환경headless추론으로checkpoint복원/평가정상종료rc0. 실제형상·긴성능평가는별도이며2회검증결과를성능으로인용하지않는다.
- 원시 `runs/table_adaptation_compare_20261027/output_etc/check/{cpu_final.log,residual.log,tokenizer.log,validation.json,view_*.log}`.

실행확인: residual본학습PID3634237/GPU0 유지, tokenizer취소. 10k continuation supervisor3637486. 원본구간이정상1000회저장을완료해야만resume한다. 그전장애시새로시작하지않고실패를기록한다. [TensorBoard](http://localhost:6013)는두모드로그를분리표시하며tokenizer 이벤트는새로생기지않는다. 공통회귀16개통과.

추가검증: 실제2회PPO checkpoint를CPU에서동일한restore 경로로복원해모델·전체normalizer·optimizer18개상태·epoch/frame exact를확인했다. 로그 `output_etc/check/resume_cpu_final.log`. 1000회최종checkpoint와구간설정/메타정보는 `residual/phase1_1000/`에보존후이어간다.


## 2026-10-28 Table4k–10k 연속 평가

사용자 요청 범위는4000/5000/6000/7000/8000/9000/10000 번호 checkpoint다. 고정 코드 사본으로 GPU0에서 순차512환경×독립seed0/1/2·30초·AT20cm 첫episode ever를 측정하고,현재Chair4096학습은 유지한다. 기존10k `runs/table_hold_at_eval_20261028_110441/`는 trace/precision/hash/epoch검증 후 재사용한다. 새4k–9k도각seed초기root/DOF/물체/모델/목표전배열이10k와exact인지 검사한다. 모든checkpoint의동결153tensor/scene RMS는원본78k exact·finite를사전검증한다.

산출물: `runs/table_residual_eval_4k_10k_20261028/{state.json,results.csv,REPORT.md}`. 상태와결과는batch진행중갱신된다. `batch.log`와 `eval_<8자리epoch>.log`는실행로그,checkpoint별원시결과는 `runs/epoch_<8자리epoch>/`다. `completed.json`이있을때만7개전체평가완료다.

평가완료 결과(평균±평가seed표본std):

| Iteration | Success (%) |
|---:|---:|
| 4000 | 80.21 ± 4.24 |
| 5000 | 81.25 ± 1.28 |
| 6000 | 85.94 ± 2.93 |
| 7000 | 86.39 ± 2.46 |
| 8000 | 88.41 ± 0.69 |
| 9000 | 87.63 ± 1.96 |
| 10000 | 90.89 ± 3.03 |

평균최고는10k90.89%. 전체21seed(18신규+3기존10k재사용)trace/precision/hash/통계검산·초기전배열동일성검증통과. 평가종료후에도Chair학습유지확인. 그래프와CSV는위산출물루트에있다.
