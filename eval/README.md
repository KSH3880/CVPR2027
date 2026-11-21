# 평가 보조 작업

2026-11-17 사용자지적으로Table3 힘조건평가의CoopFT checkpoint를6k에서7k로정정완료했다. GPU5/6/7 각기존점유포함90GB상한으로512환경·900step·2/3/4H×seeds1/6/7 신규9회를실패0으로검증했다. 다른4모델36회와앞선6k결과는보존하고동일한snapshot/힘gate/집계로통합했다.7k SR평균87.83/85.87/83.40%, Avg85.70%·동일신규궤적t_coop근접80.7875%→힘추가80.7819%다. 초기입력/정책·CPU SR/Error·원문힘gate·frame수·SHA검증통과·자체GPU worker0·기존학습보존이다. [정정통합표/seed별값](../output_etc/check/table3_force_coopft7k_gpu567_20261117/report.md), [집계](../output_etc/check/table3_force_coopft7k_gpu567_20261117/summary.json), [검증](../output_etc/check/table3_force_coopft7k_gpu567_20261117/completion_verification.json), [운영검증](../output_etc/check/table3_force_coopft7k_gpu567_20261117/operation_verification.json). 실행은 `bash output_etc/check/table3_force_coopft7k_gpu567_20261117/run.sh`이며중복시작을거부한다.

2026-11-17 사용자 요청으로 Table3의5개 모델·2/3/4H·512환경·900step·seeds1/6/7 총45회를GPU5/6/7에서19.62분에완료·검증했다. Checkpoint는TokenHSI40.5k·SCT/FSM26k·CoopFT6k·Ours V4 2k다. 전원의양손표면12cm 근접에TeamHOI의팔/손순접촉력norm>0·팀원한명이상조건을추가하고기존/추가 t_coop을동일궤적으로산출했다. 원본SR판정·20cm출발·counter/clip을유지하고FSM최초관측만canonical로보정했다. 전수첫입력/초기상태·CPU SR/Error·공개원문힘mask·CPU/GPU근접/힘frame수·FSM replay·정책/SHA검증을통과했다. 초기GPU5가88.57GB로요청85GB를초과해자체1회를중단/재실행했고보정후모든관측점은85GB이내다. 기존학습보존·자체GPU worker0이다. [최종표/seed별값](../output_etc/check/table3_force_teamhoi_gpu567_20261117/report.md), [집계](../output_etc/check/table3_force_teamhoi_gpu567_20261117/summary.json), [검증](../output_etc/check/table3_force_teamhoi_gpu567_20261117/completion_verification.json), [운영검증](../output_etc/check/table3_force_teamhoi_gpu567_20261117/operation_verification.json). 실행은 `bash output_etc/check/table3_force_teamhoi_gpu567_20261117/run.sh`이며기존manifest가있으면중복시작을거부한다.

2026-11-17 SCT+FSM의2/4/8/12H를GPU5/6/7 각전체85GB 이내에서FSM24회+현재SCT Independent12회로완료했다. 각512환경·900step·M=2N·2/4/8H seeds1/6/7,12H seeds13/20/25·8H R3m/12H R6m다. 최초reset 관측은canonical 반환값을유지하고FSM freeze/release 규칙은보존한다. 공식 `canonical_entry.py`는Independent에서controller없이SCT 원본함수를호출하므로두방법의Ind는현재SCT 공통실행결과를공유한다. 별도FSM무개입진단은900frame native행동exact/freeze0이나독립실행SR이최대0.3418pp달랐으며 `report.md`/`summary.json`에원시값을보존했다. [공식표/설명](../output_etc/check/fsm_aligned_24812_env512_gpu567_20261117/formal_report.md), [공식검증](../output_etc/check/fsm_aligned_24812_env512_gpu567_20261117/formal_completion_verification.json), [원시진단검증](../output_etc/check/fsm_aligned_24812_env512_gpu567_20261117/completion_verification.json). 공통입력/정책·정규화/반경/CPU SR·Error/FSM replay/SHA를전수검증했고자체GPU worker는모두종료했다. 실행은 `bash output_etc/check/fsm_aligned_24812_env512_gpu567_20261117/run.sh`이며기존manifest가있으면중복시작을거부한다.

2026-11-17 후속사용자요청으로GPU5 일시정지controller1개·평가/배치검사4개를종료했다. 완료결과SHA보존·잔존프로세스0·GPU5 utilization0%/70MiB를확인했다. 자동재개하지않는다. [종료기록](../output_etc/check/table1_expand20_14_18_10_gpu5_20261117/user_terminated_gpu5.json).

2026-11-17 사용자요청으로GPU5 확장평가큐전체를일시정지했다. 추가배정차단·controller/GPU프로세스STOP·GPU utilization0%를확인했으며자동재개하지않는다.20H·R10m는27회중11회rollout완료로최종CPU검증/3seed표는미완료다. [완료분만보기](../output_etc/check/table1_expand20_14_18_10_gpu5_20261117/completed_at_user_pause.md), [정지기록](../output_etc/check/table1_expand20_14_18_10_gpu5_20261117/user_paused_all_gpu5.json).14H는7m 배치검사중정지,18H/10H는미시작이다. 아래12H ablation 완료결과는유지된다.

2026-11-17 Table4의12H·6m·512환경·900step·24토큰·seeds13/20/25 재학습 ablation3개를GPU5에서9회완료·검증했다. TaskBias1500 Independent0.08±0.02%, V4NoBias2000 Mixed72.17±2.06%, NoRelFusion2000 Mixed70.02±0.85%다. TaskFusion은사용자가로컬에서별도실행한다. 기존12H 초기snapshot공유·첫입력exact·정책보존·실제구성요소·반경·CPU재채점·SHA보존을검증했다. 최대9개동시·75.37GB·실패0·자체rollout잔존0이다. [seed별표](../output_etc/check/component_ablation12_r6_seeds13_20_25_gpu5_20261117/report.md), [검증](../output_etc/check/component_ablation12_r6_seeds13_20_25_gpu5_20261117/completion_verification.json). 실행은 `bash output_etc/check/component_ablation12_r6_seeds13_20_25_gpu5_20261117/run.sh`이며중복시작을거부한다. 기존20→14→18→10H 큐는실행중인작업을유지하고신규배정만차단했으며, ablation 채팅보고후원래제한으로재개했다.

2026-11-16 Table1 추가12H·R6m·각512환경·900step·seed13/20/25의Independent/Mixed를GPU5에서모든5모델로완료·검증했다. TokenHSI40.5k·SCT/FSM26k·CoopFT7k·Ours V4 2k, 신규24회·재사용/공유포함30결과다. Ours Independent전용93.39±0.47%, Mixed81.54±0.88%이며Mixed내부I/E/C와모든모델seed별값은 [전체표](../output_etc/check/table1_12h_r6_seeds13_20_25_allmodels_gpu5_20261116/report.md)에있다. FSM 최초관측은canonical반환값으로보존해모든seed의실제첫입력exact와제어동작을검증했고외부원본은수정하지않았다. 최대12개동시·관측84.59GB·자체rollout잔존0이다. [검증](../output_etc/check/table1_12h_r6_seeds13_20_25_allmodels_gpu5_20261116/operation_verification.json), [manifest](../output_etc/check/table1_12h_r6_seeds13_20_25_allmodels_gpu5_20261116/manifest.json). 실행: `bash output_etc/check/table1_12h_r6_seeds13_20_25_allmodels_gpu5_20261116/run.sh`(중복시작거부).

2026-11-16 Ours V4 2k·12H Mixed·R6m·각512환경·900step·seed26~30의5회 신규 평가/검증을 GPU5에서 완료했다. Independent92.92±0.56%, Enabling61.55±1.92%, Coupled87.36±1.24%, Mixed80.61±0.34%(5seed 평균±표본std)다. 최대5개 동시 rollout·관측37.46GB·자체 rollout 잔존0이며 기존 결과는 보존했다. [전체seed표](../output_etc/check/ours_mixed12_r6_env512_seeds26to30_gpu5_20261116/report.md), [검증](../output_etc/check/ours_mixed12_r6_env512_seeds26to30_gpu5_20261116/operation_verification.json). 실행기는 `bash output_etc/check/ours_mixed12_r6_env512_seeds26to30_gpu5_20261116/run.sh`(중복 시작 거부)다.

2026-11-16 Ours V4 2k·12H Mixed·R6m·각512환경·900step·seed16~25의10회 신규 평가/검증을 GPU5에서 완료했다. Independent92.54±0.65%, Enabling62.16±1.46%, Coupled87.09±1.17%, Mixed80.60±0.75%(10seed 평균±표본std)다. 최대10개 동시 rollout·관측72.79GB·자체 rollout 잔존0이며 기존seed0~15 정책/정규화와같다. [전체seed표](../output_etc/check/ours_mixed12_r6_env512_seeds21to25_gpu5_20261116/combined16to25_report.md), [통합 집계](../output_etc/check/ours_mixed12_r6_env512_seeds21to25_gpu5_20261116/combined16to25_summary.json). [seed16~20 실행기](../output_etc/check/ours_mixed12_r6_env512_seeds16to20_gpu5_20261116/run.sh), [seed21~25 실행기](../output_etc/check/ours_mixed12_r6_env512_seeds21to25_gpu5_20261116/run.sh)는중복 시작을거부한다.

2026-11-16 Ours V4 2k·12H Mixed·R6m·각512환경·900step·seed0~15의16회 신규 평가/검증을 GPU5에서 완료했다. Independent92.89±0.81%, Enabling62.17±1.32%, Coupled87.28±0.92%, Mixed80.78±0.56%(seed 평균±표본std)다. 최대11개 동시 rollout·관측 메모리81.21GB·자체 rollout 잔존0이다. seed1은MPS 확인 실패 로그 보존 후 재실행했다. [전체seed표](../output_etc/check/ours_mixed12_r6_env512_seeds0to15_gpu5_20261116/report.md), [7.5m 비교](../output_etc/check/ours_mixed12_r6_env512_seeds0to15_gpu5_20261116/radius_comparison.md), [검증](../output_etc/check/ours_mixed12_r6_env512_seeds0to15_gpu5_20261116/operation_verification.json). 실행기는 `bash output_etc/check/ours_mixed12_r6_env512_seeds0to15_gpu5_20261116/run.sh`(중복 시작 거부)다.

2026-11-16 12H 그룹 중심 원 반경의3/4/5/6m 검사를 GPU5에서 완료했다. 각512환경·Independent/Mixed×seed1/6/7·준비 포함240초·1024회 시도 한도에서 최초 전체 통과 반경은 **6m**다.3m 0/6·4m 2/6·5m 3/6·6m 6/6 통과이며6m는29.8–58.0초에 완료됐다. 성공11개 snapshot의 실제 반경/균등 간격과 원본 보존을 CPU 검증했다. 정책 SR은 실행하지 않았고 기존7.5m 평가는 별도 보존한다. [비교표](../output_etc/check/group_radius12_env512_seeds167_gpu5_20261116/report.md), [검증](../output_etc/check/group_radius12_env512_seeds167_gpu5_20261116/verification.json). 실행: `bash output_etc/check/group_radius12_env512_seeds167_gpu5_20261116/run.sh`(완료 결과 보존).

평가 확인·비교·후처리용 보조 스크립트는 이 폴더에서 만든다.

2026-11-16 Ours V4 2k·12H Mixed·R7.5m·각512환경·seed0~15의16회 신규 평가/검증을 GPU5에서 완료했다. Independent91.10±0.82%, Enabling59.33±1.76%, Coupled85.36±1.41%, Mixed78.59±0.69%(seed 평균±표본std)이며 최대 관측 메모리80.75GB였다. 별도12H 최소 반경 탐색과 구분한다. [전체seed표](../output_etc/check/ours_mixed12_r75_env512_seeds0to15_gpu5_20261116/report.md), [검증](../output_etc/check/ours_mixed12_r75_env512_seeds0to15_gpu5_20261116/operation_verification.json). 실행기는 `bash output_etc/check/ours_mixed12_r75_env512_seeds0to15_gpu5_20261116/run.sh`다.

2026-11-16 Ours V4 2k·10H Mixed·R7.5m·각64환경·20토큰·900step·seed0~4 총5회를 GPU5에서 동시 완료·검증했다. Independent90.61±2.84%, Enabling65.97±2.50%, Coupled86.41±6.36%(5seed 평균±표본std)다. GPU5 최대 관측27.20GB·자체 rollout 잔존0. [전체seed표](../output_etc/check/ours_mixed10_r75_env64_seeds0to4_gpu5_20261116/report.md), [검증](../output_etc/check/ours_mixed10_r75_env64_seeds0to4_gpu5_20261116/operation_verification.json). 실행기는 `bash output_etc/check/ours_mixed10_r75_env64_seeds0to4_gpu5_20261116/run.sh`이며 완료한 rollout은 재사용한다.

2026-11-16 Ours V4 2k·12H Mixed·R7.5m·각64환경·24토큰·900step·seed0~4 총5회를 GPU5에서 동시 완료·검증했다. Independent88.95±1.18%, Enabling62.09±4.88%, Coupled84.59±2.18%(5seed 평균±표본std)다. GPU5 최대 관측27.48GB·자체 rollout 잔존0. [전체seed표](../output_etc/check/ours_mixed12_r75_env64_seeds0to4_gpu5_20261116/report.md), [검증](../output_etc/check/ours_mixed12_r75_env64_seeds0to4_gpu5_20261116/operation_verification.json). 실행기는 `bash output_etc/check/ours_mixed12_r75_env64_seeds0to4_gpu5_20261116/run.sh`이며 완료한 rollout은 재사용한다.

2026-11-16 Enabling 원인 분석은 사용자 요청으로 추가 진행을 중단했다. 저장 궤적22회·관찰 GPU9회·대조 GPU9회의 검증 자료는 `output_etc/check/enabling_scale_diagnosis_20261116/`에 보존한다. 최종 보고서 통합은 진행하지 않았다. [중단 상태](../output_etc/check/enabling_scale_diagnosis_20261116/design_status.json), [대조 실험 검증](../output_etc/check/enabling_scale_diagnosis_20261116/causal/verification.json).

2026-11-16 Ours V4 2k·16H Mixed·R7.5m·각512환경·900step의 seed9~15 총7회를 GPU5에서 추가 완료했다. 7개 rollout 동시 실행·메모리 최대 관측56.70GB로 상한85GB 이내였고 모두 종료됐다. Independent88.20±0.90%, Coupled81.82±0.88%, Enabling44.13±1.26%, Mixed71.38±0.75%(7seed 평균±표본std)다. [전체seed표](../output_etc/check/ours_mixed16_r75_env512_seeds9to15_gpu5_20261116/report.md), [운영 검증](../output_etc/check/ours_mixed16_r75_env512_seeds9to15_gpu5_20261116/operation_verification.json). 실행기는 `bash output_etc/check/ours_mixed16_r75_env512_seeds9to15_gpu5_20261116/run.sh`이며 완료한 rollout은 재사용한다.

2026-11-16 Ours V4 2k·16H Mixed·R7.5m·각512환경·900step을 seed0~8로 확장 완료했다. 기존1/6/7은 원본 보존·검증 확인 후 재사용했고0/2/3/4/5/8의6회는 GPU5에서 새로 평가했다. GPU 최대 관측47.73GB로 사용자 상한85GB 이내였고 모두 종료됐다. Independent88.39±0.48%, Coupled81.96±0.92%, Enabling44.47±1.57%, Mixed71.61±0.77%(9seed 평균±표본std)다. [전체seed·원본 경로](../output_etc/check/ours_mixed16_r75_env512_seeds0to8_gpu5_20261116/report.md), [검증](../output_etc/check/ours_mixed16_r75_env512_seeds0to8_gpu5_20261116/operation_verification.json). 실행기는 `bash output_etc/check/ours_mixed16_r75_env512_seeds0to8_gpu5_20261116/run.sh`다. seed1/6/7의 `reuse.json`은 외부 원본 경로를 가리키며, 새6회 raw는 해당 `seed_*/rollout/`에 있다.

2026-11-16 Ours V4 2k·16H Mixed·R7.5m·각64환경·900step을 seed0~8 총9회 GPU5에서 완료했다.9개 동시 rollout, GPU 메모리 최대 관측50.78GB로 사용자 상한85GB 이내였다. Independent87.08±3.18%, Coupled81.93±1.72%, Enabling42.01±3.06%, Mixed70.34±1.47%(seed별 SR 평균±표본std)다. [모든seed 표](../output_etc/check/ours_mixed16_r75_env64_seeds0to8_gpu5_20261116/report.md), [검증](../output_etc/check/ours_mixed16_r75_env64_seeds0to8_gpu5_20261116/operation_verification.json). 실행기는 `bash output_etc/check/ours_mixed16_r75_env64_seeds0to8_gpu5_20261116/run.sh`이고 원본/로그는 같은 폴더의 `seed_0/`~`seed_8/`에 있다. 기존seed1과 초기 상태는 exact였지만 새 rollout Independent 성공2건 차이가 있어 보고서에 따로 기록했다.

2026-11-16 Enabling 후속 진단은 [실패 단계·목표 기울기·곱 효과 분석](../output_etc/check/enabling_failure_diagnosis_20261116/report.md)에 있다. 반경4조건과 기존16H·3seed 총7개 궤적의4,818쌍을 CPU 분석했다. 4/8/16H의 단독 SR 제곱 및 선행 ever×후행 ever|선행 ever×동시|둘 다 ever의 정확한 분해를 비교했다. 단순 단독 저하의 누적만으로 Enabling 저하가 설명되지는 않으며, 후행 정렬/동작 미완료가 주 병목이다. 새 GPU 실행은 없다. [검증](../output_etc/check/enabling_failure_diagnosis_20261116/verification.json).

2026-11-16 Ours V4 2k의16H Mixed를64환경·seed1·900step으로 그룹 반경7.5/8/9/10m 각각 GPU5에서 동시 평가했다. 관계별 SR·건수와 CPU 재채점·초기 입력·정책 보존·실제 반경을 검증했고 모두 종료됐다. [비교표](../output_etc/check/ours_mixed16_radius64_gpu5_seed1_20261116/report.md), [검증](../output_etc/check/ours_mixed16_radius64_gpu5_seed1_20261116/completion_verification.json). 과제/관계/크기는 같지만 초기 자세·그룹 내부 배치는 일부 달라 반경만 바꾼 paired 비교가 아니다. 실행기는 `bash output_etc/check/ours_mixed16_radius64_gpu5_seed1_20261116/run.sh`이며 완료한 rollout은 재사용한다.

2026-11-16 경로 장애·목표 접근·충돌 위험의 [CPU 추가 분석](../output_etc/check/mixed_enabling_path_analysis_20261116/report.md)을 완료했다. AT 성공·생존 시작 시점이 있는9,173쌍에서 ON_TOP 회수/운반 두 구간까지 반영했고 경로 장애와 낮은 SR의 연관성을 확인했다. 전수 실제 충돌력이나 물리 불가능성을 확정한 결과는 아니다. [대표 궤적4개](../output_etc/check/mixed_enabling_path_analysis_20261116/trajectory_cases.png), [검증](../output_etc/check/mixed_enabling_path_analysis_20261116/verification.json). GPU 사용·신규 시뮬레이션은0회다.

2026-11-16 Mixed Enabling 저하를 GPU·신규 시뮬레이션 없이 저장 궤적12개(Ours4/8/16H·Coop. FT7k16H, 각3seed)의11,137쌍으로 CPU 분석했다. 실패 단계·선행 완료 시간·낙상·외부 사람/과제 물체 근접·후행 접근/정체를 기록했다. [분석 보고서](../output_etc/check/mixed_enabling_cpu_analysis_20261116/report.md), [쌍별 원자료](../output_etc/check/mixed_enabling_cpu_analysis_20261116/pairs.csv), [검증](../output_etc/check/mixed_enabling_cpu_analysis_20261116/verification.json). 근접 지표는 실제 충돌 측정이 아니며, 외부 공간 혼잡을16H 저하의 주원인으로 단정할 근거는 확인되지 않았다.

2026-11-12 `sct_coopft` PPO7k의 최신 Table1/2/3 총42회(24+9+9)를 GPU6/7에서 새로 완료했다. 512환경·seed1/6/7·900step·공통 snapshot·8H R=3m/16H R=7.5m·Time Succ./All·공용20cm/12cm t_coop를 검증했다. GPU6 최대8개/GPU7 최대12개 worker로 실제17개 동시 실행했고 실패0·평가 rollout/이전 controller 종료를 확인했다. Enabling 평균 SR55.06±1.75%, Coupled 평균 SR85.70±0.25%, t_coop pooled80.83%(별도seed80.83±1.25%)다. 이전7k Coupled 결과는 보존하고 재사용하지 않았다. [최종 표·seed 통계](../output_etc/check/sct_coopft_7000_tables123_radius_seeds167_20261112/report.md), [집계 JSON](../output_etc/check/sct_coopft_7000_tables123_radius_seeds167_20261112/summary.json), [완료 검증](../output_etc/check/sct_coopft_7000_tables123_radius_seeds167_20261112/completion_verification.json), [운영 검증](../output_etc/check/sct_coopft_7000_tables123_radius_seeds167_20261112/operation_verification.json). 아래 명령은 완료 검증을 재사용한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hwanhee/anaconda3/envs/tokenhsi/bin/python -B output_etc/check/sct_coopft_7000_tables123_radius_seeds167_20261112/run_suite.py
```

2026-11-12 `sct_coopft` PPO8k의 최신 Table1/2/3 총42회(24+9+9)를 GPU6/7에서 완료했다. 512환경·seed1/6/7·900step·공통 snapshot·8H R=3m/16H R=7.5m·Time Succ./All·공용20cm/12cm t_coop를 검증했다. GPU6 최대8개/GPU7 최대12개 worker로 실제19개 동시 실행했고 실패0·최종 중복0·평가 rollout과 이전 controller 종료를 확인했다. Enabling 평균 SR56.34±0.90%, Coupled 평균 SR85.85±0.82%, t_coop pooled84.66%(별도seed84.66±0.19%)다. [최종 표·seed 통계](../output_etc/check/sct_coopft_8000_tables123_radius_seeds167_20261112/report.md), [집계 JSON](../output_etc/check/sct_coopft_8000_tables123_radius_seeds167_20261112/summary.json), [완료 검증](../output_etc/check/sct_coopft_8000_tables123_radius_seeds167_20261112/completion_verification.json), [운영 검증](../output_etc/check/sct_coopft_8000_tables123_radius_seeds167_20261112/operation_verification.json). 아래 명령은 완료 검증을 재사용한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hwanhee/anaconda3/envs/tokenhsi/bin/python -B output_etc/check/sct_coopft_8000_tables123_radius_seeds167_20261112/run_suite.py
```

2026-11-12 `sct_coopft` PPO15k는 자동 준비의Python3.8 호환 오류를 수정한 뒤 GPU6/7에서8개 rollout을 시작했으나, 사용자 요청으로 watcher/supervisor/rollout을 모두 중단했다. 완료 검증은0/42회이며 부분 산출물과 실패 이력을 보존했다. [중단 상태](../output_etc/check/sct_coopft_15000_tables123_radius_seeds167_20261112/status.json), [watcher 상태](../output_etc/check/sct_coopft_15000_tables123_radius_seeds167_20261112/watch_state.json). 아래 명령은 사용자 재개 지시가 있을 때만 사용한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hwanhee/anaconda3/envs/tokenhsi/bin/python -B output_etc/check/sct_coopft_15000_tables123_radius_seeds167_20261112/watch_15k.py
```

2026-11-12 `sct_coopft` PPO10k의 최신 Table1/2/3 총42회를 GPU6/7·512환경·seed1/6/7·900step·XY100%로 완료했다. Ours V4의 같은 초기 snapshot·8H R=3m/16H R=7.5m·Enabling Time Succ./All·공용20cm/12cm t_coop를 검증했다. [최종 표·seed 통계](../output_etc/check/sct_coopft_10000_tables123_radius_seeds167_20261112/report.md), [집계 JSON](../output_etc/check/sct_coopft_10000_tables123_radius_seeds167_20261112/summary.json), [완료 검증](../output_etc/check/sct_coopft_10000_tables123_radius_seeds167_20261112/completion_verification.json). 큐 종료 후 아래 명령은 완료 검증을 재사용한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hwanhee/anaconda3/envs/tokenhsi/bin/python -B output_etc/check/sct_coopft_10000_tables123_radius_seeds167_20261112/run_suite.py
```

2026-11-11 `sct_coopft` PPO7k의 공용 `cvpr2027-evaluation-v1.0.0` t_coop/Jerk 재집계를 완료했다. 2/3/4H×seed1/6/7의9개 저장 궤적을 CPU로 읽고20cm 출발·12cm 전체 프레임 누적·원본counter/clip을 적용했다. t_coop 주값80.83%, Jerk14.97m/s³이며 기존 구간 내6cm 결과와 구분한다. 공용 계산 예제34개·최초 성공/SR·4,608개 trial의 분자/counter/clip·통계·원본 SHA를 검증했다. [4k/6k/7k 공용 규약 비교표](../output_etc/check/sct_coopft_7000_table2b_protocol_v1_20261111/report.md), [집계 JSON](../output_etc/check/sct_coopft_7000_table2b_protocol_v1_20261111/summary.json), [완료 검증](../output_etc/check/sct_coopft_7000_table2b_protocol_v1_20261111/completion_verification.json).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hwanhee/anaconda3/envs/tokenhsi/bin/python -B output_etc/check/sct_coopft_7000_table2b_protocol_v1_20261111/run.py
```

2026-11-11 `sct_coopft` PPO7k의 Table2(b) Coupled2/3/4H × seed1/6/7, 총9회를 GPU6/7·512환경·900step·XY100%로 완료했다. 4k/Ours의 같은 초기 reference를 재사용했고 초기 상태/관측·CPU SR/Error/접촉 재집계 검증을 모두 통과했다. [4k/7k 비교·seed별 결과](../output_etc/check/sct_coopft_7000_table2b_seeds167_20261111/report.md), [집계 JSON](../output_etc/check/sct_coopft_7000_table2b_seeds167_20261111/summary.json), [완료 검증](../output_etc/check/sct_coopft_7000_table2b_seeds167_20261111/completion_verification.json). 아래 명령은 완료 검증을 재사용하며 큐 종료 후 미완료 항목만 이어 실행할 수 있다.

```bash
/home/hwanhee/anaconda3/envs/tokenhsi/bin/python output_etc/check/sct_coopft_7000_table2b_seeds167_20261111/run_suite.py
```

2026-11-11 `sct_coopft` PPO4k의 Table1/2 정식 평가42회를 GPU6/7·512환경·900step·seed1/6/7·XY100%로 완료했다. Ours의 초기 reference를 재사용했고 모든 초기 장면/관측·CPU SR/Error·Table2 Time/접촉 검증을 통과했다. [최종 표·seed별 결과](../output_etc/check/sct_coopft_4000_tables12_seeds167_20261111/report.md), [집계 JSON](../output_etc/check/sct_coopft_4000_tables12_seeds167_20261111/summary.json), [완료 검증](../output_etc/check/sct_coopft_4000_tables12_seeds167_20261111/completion_verification.json)을 참고한다. [전용 큐](../output_etc/check/sct_coopft_4000_tables12_seeds167_20261111/run_suite.py)는 완료된 검증을 재사용하며, 큐 종료 후 같은 명령으로 미완료 항목만 이어 실행할 수 있다.

```bash
/home/hwanhee/anaconda3/envs/tokenhsi/bin/python output_etc/check/sct_coopft_4000_tables12_seeds167_20261111/run_suite.py
```

2026-11-11 `sct_coopft` 최신 PPO4k의4명·64환경·seed0 평가는 기존 Independent 전용/Mixed 샘플러를 사용했다. Independent SR은 전용 장면, Enable·Coupled·Mixed SR은 Mixed 장면에서 집계한다. 고정2쌍/4명1그룹 구성은 사용하지 않는다. GPU6/7·900step·XY100%의 [보고서](../output_etc/check/sct_coopft_4000_four_scenes_env64_seed0_20261111/report.md), [집계](../output_etc/check/sct_coopft_4000_four_scenes_env64_seed0_20261111/summary.json), [검증](../output_etc/check/sct_coopft_4000_four_scenes_env64_seed0_20261111/verification.json)을 참고한다.

2026-11-10 V3 2k·seed1/6/7의 Table1/2 실행은 [전용 큐](../output_etc/check/v3_2000_tables12_seeds167_20261110/run_suite.py)를 사용한다. SCT26k·SCT+FSM을 같은 초기 장면으로 비교하며, 기존59회 재사용·신규67회로 총126회를 검증한다. Table2의 seed1/6은 `run_table2_extra.py 6`·`run_table2_extra.py 7`에서 먼저 실행하고 본 큐가 완료 검증을 재사용한다. 기존 V2 진입점·학습 코드는 변경하지 않는다. [실행 상태](../output_etc/check/v3_2000_tables12_seeds167_20261110/status.json), [GPU6 추가 큐](../output_etc/check/v3_2000_tables12_seeds167_20261110/extra_status_gpu6.json), [GPU7 추가 큐](../output_etc/check/v3_2000_tables12_seeds167_20261110/extra_status_gpu7.json), [비교표](../output_etc/check/v3_2000_tables12_seeds167_20261110/report.md).

```bash
/home/hwanhee/anaconda3/envs/tokenhsi/bin/python output_etc/check/v3_2000_tables12_seeds167_20261110/run_suite.py
```

완료된 seed 검증은 재사용한다. `supervisor.pid`의 큐가 종료된 뒤 이 명령으로 이어서 실행하며, 현재 진행 상황만 집계하려면 같은 폴더의 `summarize.py`를 실행한다. 최종 큐는 원본/checkpoint SHA·초기 장면·CPU 재채점을 확인한 뒤 `summary.json`·`report.md`를 생성한다.

- 메인 평가 진입점은 기존 [evaluate.py](../tokenhsi/evaluate.py)를 유지한다.
- 성공 판정·낙상 처리·집계는 기존 함수를 재사용한다.
- 로그·결과·trace·임시 산출물은 `output_etc/check/`에 저장한다.
- 평가 조건은 [논문 프로토콜](../cvpr2027_evaluation_protocol.md)을 따른다.
- Table 2는 `run_table2.py`·`table2_eval.py`로 별도 실행한다. 메인 eval의 성공 판정·낙상을 재사용하며 Enabling Time·Coupled t_coop만 보조 경로에 추가한다.

`table2_sampling.py`는 2명 Enable 한 쌍(AT→SIT/CLIMB/ON_TOP 각각) 또는 2/3/4명 공동 PUSH 한 그룹만 만든다. M=2N·공통 교집합·거리·ordered·distractor 관측 유지/물리 제거와 동일 초기 snapshot 검증을 사용한다. Stage1은 원본26k 정책 입력 변환을 재사용하고 Stage2는 V2 2k다. 각 GPU에서 Stage1·Stage2를 순서대로 실행하면 GPU당18회이며 reference 생성 후512환경×인자로 지정한 세 seed를 평가한다. 두 GPU에 서로 다른 설정을 배정해 reference 경합을 피한다.

```bash
python eval/run_table2.py --output output_etc/check/table2_20261109 --model stage1 --gpu 6 --settings coupled_4 enable_sit coupled_2
python eval/run_table2.py --output output_etc/check/table2_20261109 --model stage2 --gpu 6 --settings coupled_4 enable_sit coupled_2

python eval/run_table2.py --output output_etc/check/table2_20261109 --model stage1 --gpu 7 --settings enable_climb enable_ontop coupled_3
python eval/run_table2.py --output output_etc/check/table2_20261109 --model stage2 --gpu 7 --settings enable_climb enable_ontop coupled_3

python eval/summarize_table2.py output_etc/check/table2_20261109 --seeds 1 3 6
```

각 GPU에서는 위의 Stage1 명령 완료 후 Stage2 명령을 실행한다. seed 기본값1/3/6은 이번 임시 Table 2 실행용이며 공통 프로토콜의 확정 seed가 아니다. `--seeds`·`--envs`로 검사를 분리한다. Table 2는 XY100%를 보조 채점에 적용하고 학습 보상·기본 메인 기준은 유지한다.

`table2_metrics.py`는 Enabling 최초 동시 성공 Time(성공만), Coupled 최초 수평 변위20cm부터 최초 성공까지의 전원 양손 접촉 비율을 계산한다. 실패는 기록된 에피소드 끝까지 포함, 미시작0%, 낙상 이후 접촉0, 시작/끝 프레임 포함·30Hz다. 손 중심과 회전한 실제 공유 상자 표면의3D 거리6cm 미만이며 힘 조건은 없다. 8/12cm도 같은 raw 기록에서 산출한다. 각 trial 평균 후 행동/인원별 동일 가중 평균·seed별 평균±표본std를 계산하며 성공이 없는 Time은 N/A다. Table 2에 Error 열은 넣지 않는다.

Coupled의 `cooperation.npz`에는900프레임 전체의 양손 위치·실제 상자 위치/회전·크기를 저장한다. `trace.npz`의 유효 상태·목표와 결합해 문턱 변경을 CPU로 재집계한다. `table2_trials.npz`는 최초 성공·시작·분모/분자 프레임 수이며 GPU 기록과 CPU 재계산의 exact 일치를 확인한다. 저장 용량·압축 소요 시간은 `table2.json`에 기록한다.

```bash
python eval/rescore_table2.py output_etc/check/table2_20261109/stage2/coupled_3/general_seed1 --distance-cm 6 8 12
python eval/rescore_table2.py output_etc/check/table2_20261109/stage2/coupled_3/general_seed1 --distance-cm 6 8 12 --start-displacement-cm 40
python eval/rescore_table2.py output_etc/check/table2_20261109/stage2/coupled_3/general_seed1 --distance-cm 12 --start-displacement-cm 20 --start-hand-distance-cm 16
```

이 명령은 기본적으로 원본을 읽기만 한다. `--start-displacement-cm`은 밀기 구간의 시작 변위이며 기본20cm다. 시작 변위를 바꾸면 시작/분모/분자만 재계산하고 최초 성공 프레임·SR은 원본과 검증한다. 별도 결과가 필요하면 `--output output_etc/check/<결과>.json`을 지정한다.

`--start-hand-distance-cm 16`은 같은 프레임에 유효 참여자의 한 손 이상이 상자 표면16cm 미만인 조건을 시작 변위 조건과 함께 요구한다. 한 번 시작하면 유지하며 높이/접촉력 조건은 없다. 옵션 생략 시 기존 이동량만 사용하는 시작 조건이다. TeamHOI를 참고한 PUSH용 비교 옵션이며 기본 프로토콜은 변경하지 않는다.

시작20cm+손16cm 조건으로 저장 궤적을 재집계한 비교·seed별 결과는 [손 접근 시작 조건 비교](../output_etc/check/table2_handstart16_20261109/report.md)에 있다.

속도·지속 시간으로 시작을 정하는 비교는 다음 명령으로 CPU 재집계한다. 기본9조합은 XY 속도0.05/0.1/0.2m/s × 0.1/0.3/0.5초다. 초기 물리 상자 중심부터 프레임 간 XY 변위를 control dt로 나눈 속도를 쓰며 평활화하지 않는다. 연속 구간의 매 프레임에 유효 참여자 한 손 이상이 표면16cm 미만이어야 한다. 최초 성공까지 조건이 확인되면 시작을 연속 구간 첫 프레임으로 소급한다. 변위20cm 조건은 대체하며 공동 접촉은 전원 양손12cm 미만·실패 포함·미시작0%다. 원본 최초 성공/12cm 접촉 프레임 수를 exact 검증하고 기본 프로토콜은 변경하지 않는다.

```bash
python eval/rescore_table2_speed.py --output output_etc/check/table2_speed_start_20261109 --speeds 0.05 0.1 0.2 --durations 0.1 0.3 0.5
python eval/rescore_table2_speed.py --output output_etc/check/table2_speed_02_10_1s_nohand_20261109 --speeds 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --durations 1 --no-start-hand
```

`summary.json`·`report.md`에 인원별/seed별 t_coop와 시작률, 실행별 JSON·NPZ에 조합별 시작/분모/분자 프레임 수를 저장한다. [속도·지속 시간 비교 결과](../output_etc/check/table2_speed_start_20261109/report.md).

`--no-start-hand`는 시작의 손 접근 조건만 제거하고 XY 속도·지속 시간만 사용한다. 전원 양손 접촉12cm 판정은 유지한다. `--start-hand-cm`과 동시 지정할 수 없다. [손 접근 조건 제거 비교](../output_etc/check/table2_speed_02_10_1s_nohand_20261109/comparison.md).

이번 Stage1 26k·Stage2 V2 2k의 Table 2 36회 결과는 [report.md](../output_etc/check/table2_20261109/report.md), 초기 장면·CPU 재집계·기록 검증은 [verification.json](../output_etc/check/table2_20261109/verification.json)에 있다.

시작 변위만40cm로 바꾼 CPU 재집계의20cm 비교·seed별 수치는 [시작 변위 비교](../output_etc/check/table2_start40_20261109/report.md)에 있다. 기본 프로토콜은20cm로 유지한다.

일반 `*_test.sh`·`*_vnc.sh`는 학습 장면 샘플러와 보상 기반 viewer 진단을 사용한다. `TASK_GRAPH=random_scenario`만으로 논문 프로토콜의 공통 교집합·균등 관계/행동·near50 해제·distractor 물리 제거가 적용되지는 않는다. Viewer의 성공률/낙상 처리도 formal SR과 다르며, `EVAL_PROTOCOL=general`은 headless 전용이라 일반 VNC에 붙이는 방식으로 전환할 수 없다.

### 공식16H Mixed VNC

```bash
TOKENHSI_GPU=6 PORT=6080 bash eval/mixed16_formal_vnc.sh --seed 1
```

이 전용 진입점은 외부 `stage1_2_base_teacher/eval/group_spacing_eval.py`와 Ours V4 2k checkpoint를 읽기 전용으로 재사용한다. 사용자 요청대로 **1환경**·16명·32토큰·900step·그룹 중심 반경7.5m를 사용하고 균등 관계/행동·크기 교집합·distractor 관측 유지/물리 제거·전체 토큰 attention·낙상 제어·XY100% 규칙을 같은 평가 경로로 처리한다. GPU5/6/7을 허용하며 기본6이다. `--seed`는0 이상의 임의 정수이고 기본1이다. 환경 수가 달라 기존512환경의 특정 장면과 같지는 않다. 먼저 같은1환경 초기 reference를 별도 headless 프로세스로 생성(물리step0)한 뒤, viewer 프로세스에서 재샘플링해 초기 관측/최초 입력 일치를 검사한다. 두 프로세스는 순차 실행하며 각각1환경이다.

환경 생성은 `--headless --no_video`를 유지하고 graphics device만 활성화한 뒤 viewer를 별도로 만든다. 일반 viewer의 추가 marker actor가 생기지 않으며 카메라·화면 갱신만 추가한다. Space로 시작/일시정지, F로 카메라 자동 추적, Esc로 종료한다. 처음은 일시정지하며900step 후 마지막 화면을 유지한다. `PORT`를 VS Code에서 포워딩하고 브라우저의 `http://localhost:6080/vnc.html`로 접속한다.

출력은 실행마다 이 checkout의 `output_etc/check/mixed16_formal_vnc_*/`에 새로 생성하고 원래 평가 결과는 덮어쓰지 않는다. `viewer_manifest.json`·`reference/`·`rollout/`에 설정·초기 상태·검증/trace를 남긴다. `bash eval/mixed16_formal_vnc.sh --dry-run --seed 1`은 GPU import·파일 생성·VNC 실행 없이 설정을 확인한다. CPU 검증만 수행했으며, 실제 graphics device/초기 입력 일치와 전체 rollout은 GPU 실행 시 확인해야 한다. 초기 입력이 다르면 기존 평가 검사가 실행을 중단한다. 논문 결과를 산출하는512환경 평가와 구분한다.

`compare_region_extent.py`는 저장 궤적의 XY 중앙80%·전체100% 성공률을 CPU에서 비교한다. 기존80%의 SR·ever/simultaneous 재현과100% 성공의 포함 관계를 검증하며, 메인 기준·정책·원본 결과·Error는 변경하지 않는다.

`run_stage2_scalability.py`는 `stage2_scalability_eval.py`로 Independent 전용·Mixed를 분리해 2/4/8/16명 평가를 실행한다. 초기 snapshot·매 frame 물리/관측 분리·CPU 재채점을 검증하며, 출력은 `output_etc/check/`에 둔다. 2명 제한 완화와 5명 이상 그룹별 distractor 배분·관측 보존은 보조 경로에만 적용한다. `stage2_xy100.py`는 채점·Error의 XY만100%로 바꾸고 학습 보상·관측 계산은 유지한다. `summarize_stage2_scalability.py`는 기존4H Mixed XY100% 결과를 재사용해 인원수별 표를 만든다. seed는 실행 인자로 지정하며 프로토콜 기본값을 변경하지 않는다.

5명 이상 초기 공동 상자는 그룹 대표만 쓴다. 여러 참가자가 같은 물리 상자에 동시에 다른 초기 상태를 쓰던 중복을 막으며, 최종 공동 PUSH 후보·거리·ordered 배정은 기존 함수를 사용한다. `STAGE2_SAMPLING_AUDIT=1`은 초기 배치 단계별 상태/RNG 비교 파일을 추가하며 기본은 꺼져 있다.

`stage2_full_observation.py`는 관측용 distractor를 encoder attention에도 포함시킨다. 기존 `MAPlayerContinuous`의 5명 이상 task-slot mask는 이 보조 경로에서 해제하며 encoder 호출마다 확인한다. 이 수정 이전 8·16명 결과는 distractor attention 조건이 다르므로 재평가 결과와 구분한다.

Stage1 비교는 같은 runner에 `--stage1-checkpoint <Stage1_00026000.pth>`와 `--reference-evaluations <기존Full평가폴더>`를 추가한다. `--checkpoint`는 동일 초기 장면 생성을 위한 Full checkpoint이며, 첫 행동 전에 Stage1 가중치·정규화를 복원한다. `stage1_protocol_policy.py`는 enable의 목표 정렬 가상 받침과 coupled의 개인별 PUSH 구역을 입력하고 Stage2 coordination을 우회한다. 원본 Stage1 network와 행동 평균/표준편차 exact 일치·가중치 불변·coordination 호출0·초기 물리/관측 일치를 검증한다. Stage1과 Full 모두 모든 관측 토큰을 attention에 포함한다.

`compare_stage1_scalability.py <Stage1평가폴더> <Full재평가폴더> <기존Full평가폴더>`는 Stage1 전체·Full8/16H의 검증 완료를 확인하고 기존 Full2/4H를 읽기만 해 비교표와 seed별 SR·그룹 수를 만든다. 원본 결과는 변경하지 않는다.

```bash
python eval/run_stage2_scalability.py --checkpoint <checkpoint.pth> \
  --output output_etc/check/<평가명> --gpu 6 --seeds <seed1> <seed2> <seed3> \
  --jobs independent:2 independent:4 mixed:2 independent:16
python eval/run_stage2_scalability.py --checkpoint <checkpoint.pth> \
  --output output_etc/check/<평가명> --gpu 7 --seeds <seed1> <seed2> <seed3> \
  --jobs mixed:16 independent:8 mixed:8
python eval/summarize_stage2_scalability.py output_etc/check/<평가명> \
  --seeds <seed1> <seed2> <seed3> --mixed4-cache <기존XY100비교/comparison.json>
```

```bash
/home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/compare_region_extent.py \
  output_etc/check/stage2_distill_900_v2_2000_new_protocol_seeds678_20261108 \
  output_etc/check/stage2_v2_2000_xy100_seeds678_20261108
```

`stage2_protocol_eval.py`는 기존 검증된 새 프로토콜 샘플러·distractor 관측/물리 분리 hook을 재사용하는 **4명·512환경 전용** 진입점이다. `stage2_distractor_physics.py`는 기준 snapshot 경로를 제외하면 기존 검증 hook과 같다. 메인 채점·Error·CPU 재채점은 공통 코드를 사용한다.

기준 경로 기본값은 `output_etc/check/stage2_new_protocol_1500_env512_20261108/reference/`다. 새 seed는 `STAGE2_REFERENCE_ROOT=output_etc/check/<평가명>/reference`를 지정하고, 아래 명령을 `REFERENCE_ONLY=1`·출력 `<기준경로>/general_seed<seed>`로 먼저 실행해 초기 snapshot·관측만 저장한다. 이후 `REFERENCE_ONLY=0`·출력 `<평가명>/general_seed<seed>`로 같은 checkpoint·설정의 실제 평가를 실행한다. 기준 생성은 rollout·성공 집계를 하지 않는다.

```bash
TOKENHSI_GPU=6 bash -c '
source tokenhsi/scripts/multi_agent/runtime_env.sh
python eval/stage2_protocol_eval.py \
  --task HumanoidMACoupledPush --cfg_train tokenhsi/data/cfg/train/rlg/stage2_distill_900_v2.yaml \
  --cfg_env "" --motion_file tokenhsi/data/dataset_stage1_jointpush.yaml \
  --checkpoint "$1" --output_path "$2" --seed "$3" --num_envs 512 \
  --test --eval --headless --no_video --eval_protocol general --eval_repeats 1 \
  --task_graph random_scenario --enable_size_mode per_action_intersection --coupled_assignment ordered
' _ <checkpoint.pth> output_etc/check/<평가명>/general_seed0 0
```

크기 CLI는 기존 hook 진입용이며 실제 공유 상자는 전체 공통 교집합이다. 결과에는 `common_intersection`·`cvpr2027_mixed_v1`·동시 성공 기준·두 Error가 기록된다. 지정한 seed를 각각 실행한 뒤 다음 보조 코드로 초기 상태·CPU 재채점·seed 통계를 확인한다. 평가 폴더에는 checkpoint·reference·seeds를 기록한 manifest가 필요하다.

```bash
PYTHONPATH=tokenhsi /home/hwanhee/anaconda3/envs/tokenhsi/bin/python \
  eval/summarize_stage2_protocol.py output_etc/check/<평가명>
```
