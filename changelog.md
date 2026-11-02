# Changelog

현재 실행법은 [config](markdowns/config.md), 코드 위치는 [structure](markdowns/structure.md)를 따른다. 정리 전 문서·변경 기록은 [보관 자료](markdowns/legacy/pre_four_configs/README.md)에 있다.

## 2026-11-02

### BEFORE 세 모델 8명·16토큰·64환경·1회 평가

- 추가 GPU/시뮬레이션 없이 기존4명(6000/5500/6000·512환경×3seed 평균)과8명(7000/6500/7000·64환경×seed2)의 Plane Ever 차이를 집계했다. checkpoint·평가 규모가 달라 순수 인원 수 효과로 해석할 수 없음을 명시했다. [4명→8명 행동별 차이](output_etc/check/before8_eval64_20261102/comparison_4_to_8.md), [집계 JSON](output_etc/check/before8_eval64_20261102/comparison_4_to_8.json).
- 사용자 지정대로 GPU5·6만 사용해 source5 7000·virtual6500·asym7000 저장본을8명·16토큰·64환경·seed2·첫 episode1회·1200step·deterministic으로 평가했다. 저장된 random_scenario/near50 설정을 유지하고 기존 Plane Ever 기하 판정·개인 낙상 freeze를 재사용했다. 학습4명에서 인원을 바꾼 진단 평가이며, 정식 general의 인원 변경 금지 규칙은 운영 코드에서 유지하고 확인용 driver만 별도 경로에 뒀다.
- 후행 동시 Plane Ever는 SIT21.05/15.79/13.16%(각38쌍), CLIMB30.77/2.56/2.56%(각39쌍), ON_TOP10.11/13.48/8.99%(각89쌍), 전체17.47/11.45/8.43%(각166쌍)였다. 선행 AT81.33/77.71/79.52%이며 단독 행동은 양쪽 BEFORE 역할을 제외해 집계했다. 1seed이므로 표준편차를 계산하지 않았다. [행동별·단독·선행 AT 표](output_etc/check/before8_eval64_20261102/report_plane.md).
- 세 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 차이0·모델 간 초기 상태 정확 일치·checkpoint 및 운영 파일281개 해시 보존을 확인했다. 기존 학습 프로세스도 유지됐다. 8명 초기화에서 기존 PUSH sequential 경로 배치가 오래 걸렸으며 이번 예산 수정에서 해당 물리 sampler를 최적화하거나 검사를 완화하지 않았다. [입력 조건](output_etc/check/before8_eval64_20261102/plan.json), [전체 수치](output_etc/check/before8_eval64_20261102/comparison.json), [검증](output_etc/check/before8_eval64_20261102/verification.json).

### BEFORE PUSH viewer의 가변 인원·물체 예산 지원

- 추론 validation을 CLI와 task 생성까지 전달해 BEFORE PUSH viewer/test/VNC에서 N≥1·M≥N을 허용했다. Alias 포함 `N+ON_TOP 수`가 예산을 초과한 과제 조합만 최대1024회 재샘플링하며 불가능한 고정 ON_TOP preset은 즉시 거절한다. 학습/resume의 M≥2N·기존 충분한 예산의 샘플링/RNG·모델/보상/Gate/계약과 정식 general 규약은 유지했다. config/USAGE/structure를 갱신했다.
- CPU132개 통과(기존 checkpoint 전이1개 제외), 수정 전후 CPU5개·CUDA4개 사례의 그래프/역할/template/RNG 정확 일치, GPU5의 작은 토큰 예산2560개 장면 검사를 통과했다. GPU6에서 기존 asym6000을8명·1환경·12토큰·seed2·32step으로 실제 VNC 실행/종료하고 CUDA/Vulkan GPU 일치를 확인했다. 같은 GPU에서 실제 test wrapper의1명·1토큰/source5 7000과8명·8토큰/virtual6500을 각각1환경·seed2·16step으로 완료했다. 보조8환경 다중preset 검사는 random_scenario16step·부분reset 이후 all_solo의 기존 PUSH 배치 지연으로 해당 보조 프로세스만 중단했으며 전체 preset 통과로 간주하지 않는다. 기존 학습·참고 jointpush 레포는 보존했다. [검증/제한](output_etc/check/before_viewer_budget_20261102/report.md), [이번 구현 diff](output_etc/check/before_viewer_budget_20261102/implementation_diff.patch).

### 최신 goalroute 세 BEFORE 실험 Plane Ever 평가

- 실제 초기 후행 수행거리가 선행 접근+운반거리보다 짧은1322/1689쌍(78.27%)을 CPU로 분리 집계했다. 후행 SIT45.36/22.51/10.51%, CLIMB24.15/2.60/1.00%, ON_TOP19.66/19.57/17.55%였다. 강제 near 표시는 저장되지 않아 실제 거리로 분리했으며, 모델별 분류 동일·near/far 합의 기존 행동/seed별 건수 정확 재현을 확인했다. [가까운 쌍/나머지 표](output_etc/check/goalroute_eval_latest_20261102/report_near_plane.md).
- 평가 시작 시 최신 고정 저장본인 source5 6000/frame786432000, virtual 5500/frame720896000, virtual_asym 6000/frame786432000을 각각512환경×4명×seed0·1·2×1200step deterministic 첫 episode로 평가했다. 직전 허용 GPU0·3 중 GPU0은 여유 메모리 약2GB라 GPU3만 사용했고 기존 프로세스·학습 설정은 유지했다. near50을 포함한 checkpoint의 학습 샘플링을 사용하며 모델별 epoch가 다르다.
- BEFORE 후행 동시 Plane Ever는 SIT44.68/21.81/11.46%, CLIMB25.96/6.17/6.09%, ON_TOP21.31/20.69/18.27%였다. 단독 SIT96.55/92.31/85.80%, 선행 AT82.84/83.00/84.72%였다(source5/virtual/asym 순서). [행동별 표·선행 AT](output_etc/check/goalroute_eval_latest_20261102/report_plane.md), [1500→최신 비교](output_etc/check/goalroute_eval_latest_20261102/comparison_1500_to_latest.md).
- 9개 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 판정 차이0·입력 checkpoint 해시 보존을 통과했다. 세 모델과1500↔최신의 같은seed 초기 상태가 정확히 같았고 운영 코드/설정281개 해시도 같았다. 현재 세 학습 PID 생존·누적 학습 지표 finite를 확인했다. [입력 조건](output_etc/check/goalroute_eval_latest_20261102/plan.json), [전체 수치](output_etc/check/goalroute_eval_latest_20261102/comparison.json), [전체 검증](output_etc/check/goalroute_eval_latest_20261102/verification.json).

## 2026-11-01

### 최신 goalroute 세 실험 1500 checkpoint Plane Ever 평가

- 추가 CPU 분리 집계에서 초기 후행 수행거리가 선행 접근+운반거리보다 짧은 쌍은1322/1689개(78.27%)였다. near50은 약50%에 짧은 거리를 강제하며 나머지 무제약 표본도 가까울 수 있다. 저장되지 않은 강제 near 선택 표시 대신 실제 초기 거리로 분리했고 세 모델/seed별 분류 동일·near/far 건수 합의 기존 결과 정확 재현을 확인했다. [가까운 쌍/나머지 Plane 표](output_etc/check/goalroute_eval1500_20261101/report_near_plane.md).
- 사용자 지정 GPU0·3에서 세 goalroute 실험의 epoch1500/frame196608000 저장본을 각각512환경×4명×seed0·1·2×1200step deterministic 첫 episode로 평가했다. BEFORE는 선행AT·후행 동시 Plane Ever, 단독은 혼합 장면에서 BEFORE 양쪽 역할을 제외한 개인 Ever다. 기존 프로세스·학습 설정은 유지했다.
- BEFORE 동시 전체 평균±표본std는 source5 `21.42±0.77%`, virtual `12.27±1.03%`, asym `12.22±1.57%`였다. 후행 SIT38.07/13.79/11.27%, CLIMB19.91/5.13/6.64%, ON_TOP15.86/14.64/14.86%였다. 단독 SIT94.64/82.35/84.62%로1000 대비 각각+3.48/+2.72/-3.82%p였다. [행동별 Plane 표](output_etc/check/goalroute_eval1500_20261101/report_plane.md), [1000→1500 비교](output_etc/check/goalroute_eval1500_20261101/comparison_1000_to_1500.md).
- 9개 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 차이0·checkpoint 해시 보존을 통과했다. 세 모델과1000↔1500의 같은seed 초기 상태가 정확히 같았고 운영 코드/설정281개 해시도 보존됐다. 모든seed GPU0/3 실행과 현재 세 학습 PID·지표 finite를 확인했다. [입력 조건](output_etc/check/goalroute_eval1500_20261101/plan.json), [전체 검증](output_etc/check/goalroute_eval1500_20261101/verification.json).

### 최신 goalroute 세 실험 1000 checkpoint Plane Ever 평가

- 동일 세 goalroute 실험의 epoch1000/frame131072000 저장본을 GPU5·6만 사용해 각512환경×4명×seed0·1·2×1200step의 deterministic 첫 episode로 정식 평가했다. BEFORE는 선행AT·후행 동시 Plane Ever, 단독은 혼합 장면에서 BEFORE 양쪽 역할을 제외한 개인 Plane Ever다. 기존 학습·checkpoint·운영 코드는 유지했다.
- BEFORE 동시 전체 평균±표본std는 source5 `18.93±0.91%`, virtual `13.39±0.44%`, asym `12.29±1.66%`였다. 후행 SIT25.65/15.07/9.99%, CLIMB22.50/6.75/5.91%, ON_TOP14.90/15.54/15.72%였다. 단독 SIT는91.16/79.63/88.44%로500 대비 각각-5.34/+1.99/+0.03%p였다. [행동별 Plane 표](output_etc/check/goalroute_eval1000_20261101/report_plane.md), [500→1000 비교](output_etc/check/goalroute_eval1000_20261101/comparison_500_to_1000.md).
- 9개 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 차이0·입력 checkpoint 해시 보존을 통과했다. 세 모델 및500↔1000의 같은seed 초기 상태가 정확히 같았고 두 평가의 운영 코드/설정281개 해시도 같았다. 모든seed GPU5/6 실행과 현재 세 학습 PID·지표 finite를 확인했다. [입력 조건](output_etc/check/goalroute_eval1000_20261101/plan.json), [전체 검증](output_etc/check/goalroute_eval1000_20261101/verification.json), [seed별 변화](output_etc/check/goalroute_eval1000_20261101/delta_from_500.json).

### 최신 goalroute 세 실험 500 checkpoint Plane Ever 평가

- source5_goal5_goalroute·virtual_goal5_goalroute·virtual_asym_goal5_goalroute의 epoch500/frame65536000 저장본을 GPU5·6만 사용해 각512환경×4명×seed0·1·2×1200step의 deterministic 첫 episode로 정식 평가했다. 시작 시 virtual은466epoch여서 해당500 저장본이 생성된 뒤 평가했으며, 각 checkpoint의 goal_route_sum·목표 거리·Virtual/비대칭 설정을 유지했다.
- BEFORE 동시 Plane Ever 전체 평균±표본std는 source5 `14.26±1.15%`, virtual `11.60±0.90%`, asym `10.71±1.23%`였다. 후행 SIT22.22/13.83/9.99%, CLIMB19.76/8.45/9.24%, ON_TOP9.14/12.09/11.57%이며 단독은 BEFORE 양쪽 역할을 제외해 따로 집계했다. [행동별 Plane 표](output_etc/check/goalroute_eval500_20261101/report_plane.md), [seed별 건수·통계](output_etc/check/goalroute_eval500_20261101/plane_tables.json).
- 9개 rollout 모두1200step을 완료하고 CPU float64 재채점·GPU 누적·JSON 건수가 일치했다. trace finite·float32/64 누적 판정 차이0·같은seed 초기 상태 정확 일치·입력 checkpoint 해시 보존과 모든seed GPU5/6 실행 기록을 확인했다. 운영 코드/설정281개 해시와 기존 세 학습 PID·지표 finite도 유지됐다. [입력/실행 조건](output_etc/check/goalroute_eval500_20261101/plan.json), [전체 검증](output_etc/check/goalroute_eval500_20261101/verification.json).

### 이전 source5·virtual·virtual_asym 5000 checkpoint Plane Ever 평가

- 이전 세 run의 epoch5000/frame655360000 저장본을 GPU5·6에서 각512환경×4명×seed0·1·2×1200step의 deterministic 첫 episode로 정식 평가했다. checkpoint에 저장된 기존 relative_sum 배치를 사용했고 현재 goalroute 학습은 유지했다. BEFORE는 선행AT·후행 같은 frame의 동시 Plane Ever, 단독은 BEFORE 선행·후행을 모두 제외한 개인 Plane Ever다.
- BEFORE 동시 전체 평균±표본std는 source5 `15.12±1.78%`, virtual `8.12±0.68%`, asym `9.10±1.16%`였다. 행동별 SIT33.85/7.59/9.72%, CLIMB13.82/2.07/2.25%, ON_TOP8.85/10.85/11.66%로 source5의 SIT/CLIMB 우위가 유지됐다. [Plane 행동별 표](output_etc/check/eval5000_20261101/report_plane.md), [세부 수치](output_etc/check/eval5000_20261101/plane_tables.json).
- 9개 rollout의 CPU float64 재채점·GPU 누적·JSON 건수가 일치했고 trace finite·float32/64 누적 판정 차이0·동일seed 초기 상태 정확 일치·입력 checkpoint 해시 보존을 확인했다. 비교 코드 삭제 후 운영 코드/설정281개 해시도 유지됐고 현재 세 goalroute 학습 PID·지표 finite를 확인했다. [입력 조건](output_etc/check/eval5000_20261101/plan.json), [전체 검증](output_etc/check/eval5000_20261101/verification.json).

### 사용하지 않는 최적화 비교·계측 코드 정리

- 사용자 요청으로 `output_etc/check/push_opt123_20261101`·`reset_share_20261101` 아래 일회성 Python17개와 bytecode9개만 삭제했다. 운영 경로의 참조·실행 프로세스가 없음을 확인했고 코드/설정281개 해시를 전후 비교해 전부 보존했다. 현재 학습·원본·모든 checkpoint·보고서·JSON·로그는 유지했다. 보고서의 삭제 코드 링크를 정리하고 삭제 표시를 추가했다. [삭제 목록·검증](output_etc/check/eval5000_20261101/cleanup_result.json).

### source5 goalroute 실제 학습의 reset 시간 비중 계측

- GPU5·원본 source5_goal5_goalroute·4명/1024환경/8토큰·seed42·Stage1 26k에서 CHECK40epoch(1280step)를 실행했다. 계측은 별도 프로세스에서 reset 앞뒤 동기화와 후보 sampler CUDA event로 추가했고 운영 코드·모델/보상/샘플링 설정·기존 프로세스는 유지했다. 시작 전체 초기화와 첫2epoch는 집계에서 제외했다.
- 측정38epoch275.584초 중 reset39.994초(14.51%), 나머지 rollout80.372초(29.16%), PPO/AMP 업데이트·배치 준비 등155.217초(56.32%)였다. reset 내부 PUSH 후보 sampler는7.369초로 전체2.67%·reset18.42%였다. reset465회 중464회는1–5환경,1회는550환경이었고 호출당 환경 수 중앙값1이었다. 큰 전체 초기화의63–66% 단축을 학습 중 작은 부분 reset에 그대로 대입할 수 없으며, 이번 후보 sampler 최적화가 줄일 수 있는 전체 시간 비중은 제한적이다.
- 40epoch/frame5242880 완료·학습 지표/모델/optimizer/정규화 finite·encoder73 tensor/입력 RMS 동결·운영 sampler 해시 보존을 확인했다. 단일seed 초기 학습과 동기화 계측의 근사 비중이며 성숙한 정책·세 모델 동시 학습이나 최적화 적용 후 처리량을 보장하지 않는다. [보고서](output_etc/check/reset_share_20261101/report.md), [전체 집계](output_etc/check/reset_share_20261101/verification.json), [checkpoint 검사](output_etc/check/reset_share_20261101/checkpoint_validation.json).

### source5 goalroute PUSH sampler 최적화1·2·3 분리 비교

- 사용자 요청으로 GPU5만 사용해 source5_goal5_goalroute 하나에서 원본/실제 PUSH 경로 벡터화/반경 재사용/후보 메타데이터 공유를 누적 비교했다. 구현은 `output_etc/check/push_opt123_20261101/variants.py`의 프로세스별 교체로 격리했고 운영 코드·세 설정·기존 프로세스는 보존했다. 최종 전체 장면 validator는 원본을 유지한다.
- CPU/GPU 각각 경로147,456건·경계30건·sampler54건의 판정/후보/미해결 index/입력/RNG 동일성을 통과했다. 교차32,768회 reset의 상태·관측이 정확히 같았고, 4명/1024환경/8토큰 스트레스65,596회에서 거리/near/충돌/경로/각도/성공 환경 보존·부분 reset을 통과했다. 최종 실패0·최대10라운드·버전별 fallback2,216건 전부 해결·집계 동일이었다. 네 버전의32환경5preset×16step Actor/Critic·alias·보상/관측 finite·부분 reset·모델 보존과 각5epoch CHECK 학습의26k165 tensor 전이·encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite를 확인했다.
- 같은 GPU에서 순서를 교차한6개 측정 seed의 전체 reset 중앙값은 원본1.749초, 1번0.645초, 1+2번0.629초, 1+2+3번0.589초였다. 순서를 뒤집은 최종 단일seed 학습에서 첫 epoch 제외 중앙값은 각각6.890/7.003/7.239/7.130초로 전체 PPO 속도 개선은 확인되지 않았다. 후보 중간 메모리는108.0→179.5/162.8/146.5MiB로 늘었고 전체 PyTorch 학습 peak는 약13.17GiB였다. 장기 처리량·정책 성능은 미검증이므로 정식 반영은 보류한다. 기존 mix226 전용 물리 검사 도구가 goalroute 설정을 거절한 것은 비교용 검사만 현재 거리 검증으로 교체해 재검증했다. [보고서](output_etc/check/push_opt123_20261101/report.md), [전체 결과](output_etc/check/push_opt123_20261101/verification.json).

### goal5 기반 목표 중심 BEFORE 배치·수행 거리 near50 실험3개

- 세 goal5 설정에 `_goalroute`를 붙인 환경/학습 YAML과 train/test/VNC/eval을 추가했다. 선행 A→O·O→G 각각1–5m, SIT/CLIMB 후행 G→B 1–10m, ON_TOP G→P·P→B 각각1–5m와 B→G≥1m를 적용한다. near50은 생성된 쌍 절반에 후행 수행 거리 합 < 선행 접근+운반 거리 합을 강제한다. 기존 goal5 여섯 YAML 해시·단독 거리·모델/보상/Gate/Virtual/비대칭 설정을 보존하고 새로운 before 계약·출력·20개 실험의 문서/명령 목록을 분리했다. 기본은 동일 Stage1 26k 전이이며 기존 goal5 저장본과 resume/추론을 혼용하지 않는다.
- near 거리 합의 조건부 CDF와 조건부 각도로 반복 rejection을 피한다. 새 설정에서만 AT 목표·후행 배치를 PUSH 후보 검사보다 먼저 확정하고, PUSH/distractor/단독 ON_TOP 받침처럼 나중에 배치할 물체를 제외한 고정 장면을 선검사한다. 최종 전체 충돌·경로·각도·strict near 판정과64회 외부 재시도는 유지한다. 이 선검사는 중간 검사에서 발견한 불가능한 장면의 긴 PUSH 순차 fallback을 줄이기 위한 것이다.
- GPU5/6에서 최종 코드의 각4명·1024환경·8토큰을16,399회씩, 총49,197회 reset했다. 혼합·전원PUSH·BEFORE 두쌍/한쌍·독립과 부분 reset, 모든 거리·near50·충돌·경로·각도·graph/성공 환경 보존을 통과했고 최종 실패0·최대10라운드·순차 fallback 미해결0이었다. 설정별 후행 SIT2645·CLIMB2461·ON_TOP11594건을 포함했고 강제 near5739/11571쌍(49.60%)이었다. 세 설정의 기하 검사 집계도 동일했다.
- CPU213개 통과(없는 기존 checkpoint4개 skip·기존21k 전이1개 제외), 세 새 wrapper CHECK 학습 epoch2/frame262144·26k165 tensor 전이·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·finite를 확인했다. 최종 선검사 포함 asym CHECK 재확인과32환경5preset×16step의 actor/critic·Virtual XYZ/실제 회전·같은 스텝 AT 성공 전환·State/Success/HOLDING/원본 상태 보존, 세 wrapper의4환경32step headless 재로딩을 통과했다.12개 wrapper 구문·20개 config 표/세 명령 목록·문서 링크·diff 검사도 통과했다.
- 같은 GPU6에서 순서를 교차한8쌍 중 warmup2쌍을 제외한 혼합1024환경 reset 중앙값은 기존 goal5 5.246초→goalroute 3.827초(27.1% 감소)였다. 기존 본학습과 함께 실행된 reset 측정이며 장기 PPO 전체 처리량 개선을 보장하지 않는다. 기존 사용자 프로세스를 유지했으며 본학습·새 정책 성능 평가·GUI/VNC 실화면은 실행하지 않았다. [검증 보고서](output_etc/check/before_goalroute/report.md), [전체 수치/실행 조건](output_etc/check/before_goalroute/verification.json).

### 세 모델4k BEFORE·단독 Plane Ever 재집계

- 사용자 요청으로 기존 source5/virtual/virtual_asym epoch4000·512환경×4명×3seed·1200step 결과에서 `region`을 Plane Ever 주 지표로 정리했다. BEFORE는 선행 AT와 후행의 같은 프레임 동시 Ever, 단독은 선행·후행 모두 제외한 개인 Ever다. 기존 Point 표는 보존하고 [Plane 표](output_etc/check/latest3_eval_20261101_132836/report_plane.md)·[seed별 건수/통계](output_etc/check/latest3_eval_20261101_132836/plane_tables.json)를 추가했다.
- 기존9개 CPU/GPU 재채점 결과와 finite를 확인하고 저장 trace의 Plane 성공 건수/JSON 일치·기존 Point 단독 표 재현을 통과했다. 새 시뮬레이션·GPU 작업 없이 재집계했다. 후행 SIT 동시 Ever29.07/5.82/14.12%, 단독 SIT95.37/79.60/93.07%이며 값은 source5/virtual/asym 순서의 seed 평균이다.

### 원본 Stage1 26k 단독 SIT·CLIMB 거리별 평가

- 사용자 지정 주 지표는 **Plane Ever**(기존 코드 `region`)다. 저장·CPU 검증한 결과를 다시 정리했으며 추가 시뮬레이션은 하지 않았다. Stage1 26k는 SIT1–5m83.75%·6–10m90.31%, CLIMB100.00%·99.69%다. [Stage1 Plane 표](output_etc/check/stage1_26k_distance_eval_20261101/report_plane.md), [source5 Plane 표](output_etc/check/source5_distance_eval_20261101/report_plane.md). 아래 Point 수치는 별도 지표로 보존한다.
- `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`의 저장 env/sim을 복원하고 Stage1 단독 기준1명·1상자·loco·deterministic·900step·seed0으로 평가했다. 사람 root→자기 상자 중심 초기 XY 거리1·2·…·10m마다64환경, 행동별640시도이며 GPU5(SIT)·6(CLIMB)만 사용했다. 기존 학습 프로세스·코드·설정·원본 checkpoint는 보존했다.
- Point Ever는 SIT1–5m70.00%·6–10m71.88%, CLIMB98.75%·98.12%였다. 앞선 source5는4명·8토큰·1200step이므로 모델만 바꾼 통제 비교와 구분한다. 두 rollout900step·총1280시도의 CPU float64 재채점/GPU 누적/JSON 집계 일치, float32/64 누적 판정 차이0·trace finite·초기 거리 오차 최대0.067mm·checkpoint SHA256 보존을 확인했다. [거리별 결과](output_etc/check/stage1_26k_distance_eval_20261101/report.md), [실행 조건/검증](output_etc/check/stage1_26k_distance_eval_20261101/comparison.json).

### source5 5k 독립 SIT·CLIMB 거리별 평가

- 요청 시점 최신 source5 epoch5000 저장본을 사람 root→자기 상자 중심 초기 XY 거리1·2·…·10m에서 평가했다. 행동·거리별64환경×4명=256시도, seed0·loco·deterministic·1200step이며 전원 독립 SIT 또는 CLIMB로 배치했다. 별도 평가 프로세스에서 거리만 고정하고 기존 학습·설정·보상·충돌 검사를 유지했다. 단일 seed 거리 진단이며 일반 혼합 과제 평가와 구분한다.
- Point Ever는 SIT1–5m 평균83.05%·6–10m82.89%, CLIMB95.00%·95.63%였다. 거리별 결과는 [결과표](output_etc/check/source5_distance_eval_20261101/report.md)에 있다. 두 rollout의1200step·전체5120시도 CPU float64 재채점과 GPU 누적/JSON 집계 일치, float32/64 누적 판정 차이0·trace finite·최대 초기 거리 오차0.168mm·checkpoint SHA256 보존을 확인했다.
- 평가는 GPU0/1에서 완료됐고 해당 프로세스는 종료됐다. 완료 직후 사용자 지시에 따라 이후 GPU 작업은5/6만 사용한다. [실행 조건/검증](output_etc/check/source5_distance_eval_20261101/comparison.json).

### source5 계열 goal5 실험3개 추가

- `stage2_before_push_near50_pair50_source5_goal5`·`stage2_before_push_near50_pair50_source5_virtual_goal5`·`stage2_before_push_near50_pair50_source5_virtual_asym_goal5` 환경/학습 YAML과 train/test/VNC/eval을 추가했다. `before.goal_distance: [1.0, 5.0]`으로 단독 AT·BEFORE 선행 AT의 현재 상자→목표와 단독 ON_TOP의 들 상자→받침을 반경 샘플링하고, PUSH 상자→목표는2–5m로 바꾼다. 사람과 목표 최소1m·기존 Z 계산은 유지하고 새 목표에는 원점±4.5m 제한을 적용하지 않는다. 기존 세 실험의 환경/학습 YAML6개 해시를 보존했다.
- 새 설정에서만 후보 상자 간격·AT 목표와 다른 상자 간격에1mm 여유, PUSH 후보 cone89.98°를 적용한다. 최종 충돌·90° cone 기준은 유지한다. 사용자 선택대로 BEFORE 후행 최종 수행 거리에는 상한을 추가하지 않았고 현재 공유 상자/들 상자 배치·near50·보상·Gate·Virtual 관측·lambda10·모델·AMP/PPO는 유지한다. 새 계약/출력을 분리하고 기본 전이는 모두 기존 실험 실행과 같은 Stage1 26k다. config/structure/USAGE/README·거리 상세를 갱신했다.
- CPU191개 통과(없는 기존 가중치4개 skip·21k 전이1개 제외), 기존 정사각형 sampler 출력/RNG 회귀를 통과했다. GPU0/3/1·각4명/1024환경/8토큰에서 혼합·전원PUSH·BEFORE 두쌍/한쌍·독립과 부분 reset을 각각16,399회, 총49,197회 검사해 최종 실패0·장면 검사 최대9라운드·거리/near/충돌/경로/각도/성공 환경 보존을 확인했다. 각 순차 fallback2건은 기존 장면 재시도로 해결했다.
- 새 wrapper 세 개의 CHECK 학습을 각각 epoch2/frame262144까지 저장했다.26k165 tensor 전이·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite와 각4환경32step headless 재로딩을 통과했다. 새 asym 저장본의32환경5preset×16step·부분 reset에서 Virtual XYZ/실제 회전·같은 스텝 AT 성공 전환·State/Success/HOLDING/원본 상태 보존을 확인했다.17개 컨피그 표·세 명령 목록·12개 새 wrapper·문서 링크/구문 검사를 통과했다. 본학습·정규 성능 평가·GUI/VNC 실화면은 실행하지 않았고 기존 학습을 유지했다. [전체 검증](output_etc/check/before_goal5/verification.json).

### PUSH 목표 거리2–5m 검증과 reset 수치 경계 확인

- 기존 source5/virtual/asym 설정·코드와 학습 프로세스는 보존하고, 별도 검증 프로세스의 메모리에서 PUSH 상자→목표만2–5m로 바꿨다. 기존2–10m는 변경 범위를 접근 거리로 한정하면서 유지한 값이며,5m 축소 검증 실패로 되돌렸다는 기록은 확인되지 않았다. 현재 validator는 기존 Stage1 PUSH의2–10m 계약을 요구하므로 새 범위는 별도 계약 지원이 필요하다.
- 막힌4명 PUSH fixture는 기존/축소 각각20seed에서 통과했다. GPU0/3·4명·1024환경·8토큰에서 기존2–10m는16,399회, 무여유2–5m는 혼합/전원PUSH/BEFORE 두쌍·한쌍/독립의 다섯 preset24,591회 reset을 완료했다.64회 한도 소진0·최대 장면 검사12/10라운드였지만, 축소 검사에서는 PhysX 반영 뒤 상자 간격 경계가 약0.007–0.008mm 뒤집힌 두 장면을 발견했다. 탐색 검사의 추가 강제SIT preset은 random asset bank와 호환되지 않아 제외하고 기존 범위는 다섯 preset으로 재검증했다.
- 후보 간격1mm 여유만 준 중간 검사에서는 PUSH 방향44.99903°가 반영 뒤45.00025°로 바뀌는 각도 경계도 발견했다. 검증용 프로세스에서 후보 간격1mm·후보 cone89.98°를 함께 적용하자16,399회 reset과 부분 reset·그래프/성공 환경 보존·거리/near/경로/충돌/각도 최종 검사를 통과했다. 최종 간격50mm·cone90° 및 나머지 판정은 유지했고 최대 장면 검사11라운드였다. 보완안은 실제 코드·config에 적용하지 않았다. [검증 요약](output_etc/check/push_goal5_reset/report.md), [전체 결과/제한](output_etc/check/push_goal5_reset/verification.json).

### source5·virtual·virtual_asym 최신4k 정식 평가

- 재시작한 세 run의 요청 시점 최신 저장본은 모두 epoch4000/frame524288000이었다. 전용 eval wrapper로 각512환경×seed0·1·2·4명·1200step·deterministic 첫 episode를 GPU0/2/3에서 평가했다. 기존 GPU5/6 학습과 checkpoint를 보존했다.
- 점 기준 개인 macro5는 source5 `74.54±0.75%`, virtual `56.00±2.13%`, virtual_asym `61.26±2.28%`였다. BEFORE 동시 성공은 각각 `15.82±0.92%`, `8.41±3.26%`, `11.70±1.92%`, 후행 개인 성공은 `46.15±1.70%`, `9.24±3.73%`, `12.11±1.99%`였다. 실험당1689쌍이며 값은 seed 평균±표본std다. 비대칭은 virtual보다 개선됐지만 현재 시점에서는 원본이 우세하며, 후행 SIT/CLIMB 동시 성공에서 차이가 컸다. 선행 AT는 `67.62/67.00/75.35%`였다.
- 모든 seed에서 세 실험의 저장 초기 상태가 정확히 동일했다.9개 rollout이1200step을 완료하고 CPU float64 재채점·GPU 누적 성공·JSON 집계가 정확히 일치했다. float32/64 판정 차이0·trace finite·입력 checkpoint SHA256 보존을 확인했다. [결과표](output_etc/check/latest3_eval_20261101_132836/report.md), [전체 수치/검증](output_etc/check/latest3_eval_20261101_132836/comparison.json), [입력/실행 조건](output_etc/check/latest3_eval_20261101_132836/plan.json).

### source5 계열 PUSH 초기 후보 검사 배치화

- source5·virtual·virtual_asym의 초기 PUSH 후보128개를16개씩 최대8묶음으로 검사하도록 변경했다. 기존 `sample_layout(..., max_attempts=1)`의 유효성 검사를 재사용하며 환경별 첫 유효 후보·성공 환경 보존·후보 예산·순차 fallback·장면 재시도를 유지한다. 보상·Virtual 관측·lambda10·모델·checkpoint 계약·실행 명령과 다른 설정은 유지한다. 난수 소비 순서가 바뀌므로 같은 seed의 배치는 달라질 수 있다. 상세/structure/USAGE를 갱신했다.
- 관련 CPU173개 통과(없는 기본 가중치4개 skip·21k 전이1개 제외). GPU6·1024환경 reset16,399회에서 최종 실패0, 순차 fallback2,658건 해결·성공 환경/graph/near/거리/경로 조건 보존을 확인했다. GPU5·1024환경 CHECK 학습 epoch4/frame524288와 재확인 학습에서 actor encoder73 tensor·입력 RMS 동결, head6 tensor 갱신, 모델/optimizer/지표 finite를 확인했다.32환경5preset의 물리·Virtual XYZ/실제 회전·현재 AT 성공 전환·기존 State/Success/HOLDING 보존과4환경32step headless 재로딩을 통과했다.
- 후보 검사만 비교하면 어려운1환경은 중앙값1.290→0.084초(15.28배), 일반1024환경은0.0259→0.0109초(2.37배)였다. GPU5에 다른 학습이 없는 상태에서 같은 설정·seed42·4epoch의 평균 rollout+update는8.735→7.416초로15.1% 감소했다. 난수 순서가 다른 짧은 단일 seed 측정이며 장기/동시 학습 속도 보장은 아니다. 기존 프로세스를 종료하지 않았고 본학습은 시작하지 않았다. [전체 검증](output_etc/check/push_batched_sampling/verification.json).

### Virtual BEFORE 비대칭 progress 실험 추가

- `stage2_before_push_near50_pair50_source5_virtual_asym` 환경·학습 YAML과 train/test/VNC/eval을 추가했다. 기존 virtual에서 `before.asymmetric_progress.lambda_inner: 10.0`만 추가하고 출력·checkpoint 계약을 분리했다. 기본26k 전이·Virtual GTA·물리 배치·모델·PPO/AMP·가중치·State/Success Gate·성공 saturation·HOLDING/독립/PUSH 보상은 유지한다. config/structure/USAGE·상세 설계를 갱신했다.
- `predecessor_at_progress()`에 현재 AT raw 성공과 lambda를 전달해 미완료 시 반경 안쪽 거리만10배 감쇠한다. 거리/root·payload/반경·바깥 감쇠는 유지하고 성공 시 기존 식으로 즉시 복원한다. 호출부는 Runtime과 같은 `interaction_own_success()`로 현재 phi/오차를 판정하며 이전 own_success/achieved를 읽지 않는다. 기존 설정은 lambda0 경로로 동일 계산을 유지하고 새 옵션은 source5 virtual에 한정해 검증한다.
- CPU165개 통과(없는 기본 가중치4개 skip·21k 전이1개 제외). GPU5·1024환경 CHECK epoch2/frame262144 저장, actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite를 확인했다.32환경7preset의 각16step·부분 reset과 성공/실패3회 전환을 검사해 같은 스텝 반영·성공 이력 무시·State/Success/HOLDING/독립 항 보존·Virtual XYZ/실제 회전 유지를 확인했다.4환경32step headless 재로딩과 eval wrapper help·구문·문서 링크를 통과했다. 기존 본학습은 유지했고 새 본학습·정규 성능 평가·실제 GUI/VNC는 실행하지 않았다. [전체 검증](output_etc/check/before_asymmetric_progress/verification.json).

## 2026-10-31

### source5 Goal-Aligned Virtual Object 실험 추가

- `stage2_before_push_near50_pair50_source5_virtual` 환경·학습 YAML과 train/test/VNC/eval을 추가했다. 환경 차이는 `relationGraph.before_virtual_object: at_goal_xyz_v1` 하나이며 source5 배치 보완·보상·Gate를 유지한다. 기본 전이는 연결된26k이고 비교 원본도 같은26k를 지정한다. 출력과 checkpoint 계약을 분리하고 기존 계약은 보존했다.
- `before_object_alias.py`에서 packet의 actor/payload/target으로 선행 AT Goal을 찾아 복사한 scene의 후행 Alias GTA XYZ만 변경한다. `stage2_push_network.py`의 Actor/Critic Encoder 직전에 함께 적용하고 Quaternion·intrinsic·원본 관측/물리 상태/graph·coordination packet은 유지한다. 기존 목표 XY progress가 곧 가상 상자 접근 progress이며 추가 보상이나 구조 변경은 없다. config/structure/USAGE·BEFORE 상세를 갱신했다.
- 관련 CPU140개 통과(없는 기본 가중치4개 skip·21k 전이1개 제외). GPU5·1024환경 CHECK epoch2/frame262144 저장,26k165 tensor 전이·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite를 확인했다.32환경7preset에서 각16step과 부분 reset, SIT/CLIMB/ON_TOP Alias XYZ·실제 회전·원본 상태/graph 보존·정책/critic/AMP finite를 통과했다.4환경32step headless 재로드와 전용 eval의4환경·seed0·1200step을 완료했고 CPU 재채점/JSON 집계 일치·float32/64 차이0·finite·입력 저장본 해시 보존을 확인했다. 소표본 실행 검사이며 실제 GUI/VNC·장기 성능은 검증하지 않았다. 기존 본학습은 유지했다. [전체 검증](output_etc/check/before_virtual_object/verification.json).

### source5 PUSH 후보 소진·reset 좌표 반올림 보완

- 4명 독립 PUSH에서128개 동시 후보가64회 모두 소진되는 종료를 재현했다. `stage2_before_push_near50_pair50_source5`에서 기존 후보가 소진된 환경에만 순차 경로 배치를 적용하고, 유효한 경로를 보존하며 막히면 순서를 다시 뽑는다. 그래프·near 선택·성공 환경은 보존하고 소진된 환경만 기존 장면 재시도로 넘긴다.
- 상태 반영 뒤 좌표가 최대0.061mm 달라져 PUSH 경로·near 경계가 뒤집히는 사례를 확인했다. 해당 설정의 후보 생성에만 경로 여유와 near 거리 차이1mm를 추가했다. 거리 범위·최종 부등식/충돌 조건·기존 near50/mix226 기본 샘플링은 유지하며, 보완된 장면의 공간적 배치 분포는 달라진다.
- 관련 CPU92개 통과(없는 기본21k 전이 검사1개 skip·1개 제외). GPU5·1024환경에서 일반51,200·4명 PUSH5,120·BEFORE 두 쌍/한 쌍 각5,120·독립5,120과 부분 reset15개, 총71,695회를 통과했다. 순차 보완7,951건 중7,949건을 해결하고2건은 기존 장면 재시도로 통과했다. 최종 reset 실패0·그래프/near/성공 환경 보존·기존 거리/경로/충돌 조건을 확인했다. 거리 경계 측정에는 float32 허용0.2mm를 사용했으며 최대 이탈은0.038mm였다.
- CHECK 학습 epoch11/frame1441792 저장·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite를 확인했다. 저장본의4환경32step headless 재로딩과32환경5preset의 물리·alias·AMP·부분 reset 검사를 통과했다. [전체 검증](output_etc/check/reset_1031/verification.json). 본학습은 재시작하지 않았으며 장기 수렴 성능은 검증하지 않았다.

### 1031 checkout 모션·26k 전이 연결 복구

- tmux `1031`의 SIT 모션 `FileNotFoundError`를 확인했다. `stage1_2_base_jointpush`를 참고해 누락된 SIT/CARRY/LOCO/CLIMB 네 모션 링크를 공유 원본에 연결하고, `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`를 참고 레포의 파일에 연결했다. 기존 PUSH 링크와 외부 원본은 유지했다.
- PUSH 모션 YAML161개·기존 모션 YAML136개 파일 참조가 모두 존재하며, 오류가 난 SIT 배열과26k checkpoint(epoch26000·model165 tensor)의 CPU 로딩을 확인했다. [검증](output_etc/check/data_links_1031/verification.json). 학습은 실행·재시작하지 않았다.

### BEFORE near50 쌍 선택50·접근 최대5m 설정 추가

- `stage2_before_push_near50_pair50_source5` 환경·학습 YAML과 train/test/VNC/eval을 추가했다. 기존 near50에서 쌍 생성 확률0.5·일반 사람→상자1–5m·독립 PUSH 접근2–5m만 적용한다. near50·AT/ON_TOP 목표 배치·PUSH 이동2–10m·보상·모델·loco·4명/1024환경/8토큰/1200step·기본21k 전이는 유지하고 출력·저장본 계약을 분리했다.
- 1024환경 첫 검사에서 고정 주변 배치에 대한 PUSH 후보 소진을 확인했다. 새 설정에서 성공 환경을 보존하고 실패 환경만 기존 장면 재샘플링으로 넘기도록 연결했으며 거리·경로·충돌 조건과 기존 실험 기본값은 유지했다. 전원 독립 PUSH32환경 검사는 재샘플링에 수분이 소요됐다.
- 최종 CPU196개 통과(없는checkpoint4개 skip·21k 전이1개 제외). GPU6·1024환경 CHECK epoch2/frame262144 저장·finite·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신을 확인했다. 32환경7preset의 거리·near·alias·PUSH 경로·부분 reset과4환경32step headless 재로드를 통과했다. AT 목표 최대14.04m·BEFORE ON_TOP 운반 최대9.08m를 관측해 별도 목표 거리 상한을 추가하지 않았음을 확인했다.
- 일반8환경·seed0·1200step 검사와 CPU 재채점의 건수 일치·float32/64 차이0·finite를 확인했다. 기본21k 파일이 없어 검증과 문서 학습 명령에는26k를 명시했다. [전체 검증](output_etc/check/before_pair50_source5/verification.json). config/structure/USAGE/README·BEFORE 상세를 갱신했다. 본학습·GUI/VNC 실화면·정규512환경×3seed 성능 평가는 실행하지 않았다.

### 개인 목표 solo5를 별도 컨피그로 분리

- 요청에 따라 `stage2_coupled_push_individual_goal`의 단독 거리 상한 변경만 되돌렸다. 공동 확률0.5와 개인 목표·접촉 설정은 유지한다. 단독 접근/PUSH 이동 최대5m는 새 `stage2_coupled_push_individual_goal_solo5` 환경·학습 YAML과 train/test/VNC/eval로 분리하고 출력·저장본 계약도 구분했다. 단독 CARRY AT 목표 높이 보존 수정은 유지한다.
- CPU71개 통과(기존21k 전이1개 제외), 원본/변형 환경 차이는 단독 상한 두 값뿐임을 확인했다. GPU6에서 기존5m 검증 저장본을 새 wrapper로 재로딩하여4환경32step 실행을 완료했다. [CPU 로그](output_etc/check/coupled_individual_solo5_split/cpu.log), [실행 로그](output_etc/check/coupled_individual_solo5_split/reload.log). shell 구문·diff 검사 통과. 본학습은 실행·재시작하지 않았다.

### 개인 목표 실험의 단독 접근·PUSH 이동 최대5m

- 새 개인 목표 컨피그에 `solo_source_max_distance: 5.0`·`solo_goal_max_distance: 5.0`을 추가했다. SIT/CLIMB/CARRY AT/ON_TOP의 사람→상자는1–5m, 단독 PUSH 접근·이동은2–5m다. 공동 접근2–10m·이동2–5m와 CARRY 목표 XY 샘플링은 유지한다. 공통 샘플러의 기본값은 유지하고 현재 설정에만 상한을 전달한다.
- 개인 목표 변환을 공동 구성원에게만 적용하여 단독 CARRY AT의 목표 높이 덮어쓰기를 수정했다. 기존 검사의 목표 벡터 비교를 PUSH에 한정하고 단독 접근 범위·AT 목표/받침 높이 검사를 추가했다.
- CPU53개 통과(기존21k 전이1개 제외). GPU6·확률0.5·1024환경 CHECK epoch2/frame262144 저장, finite·encoder/RMS 동결·head 갱신 확인.32환경8preset의 거리·고정 목표·부분 reset 검사를 통과했고 holding_at의 높은 목표61개를 포함해 받침과 목표 높이 일치를 확인했다. [검증](output_etc/check/coupled_individual_solo5/verification.json). 기존 본학습은 재시작하지 않았다.

### 개인 목표 실험의 공동 선택 확률50%

- `stage2_coupled_push_individual_goal`의 `coupled_probability`를0.8→0.5로 변경했다. 그룹 선택 확률 기준이며4명 기대 인원 비율은 공동63.54%·단독36.46%다. 확률 validator와 설정 비교 검사를 갱신했고 기존 coupled/goal5는0.8을 유지한다.
- 관련 CPU44개 통과(기존21k 전이1개 제외), diff 검사 통과. [로그](output_etc/check/coupled_individual_goal/probability50_cpu.log). 아래 GPU 검증은 변경 전0.8 설정의 결과이며 이번에는 학습을 실행·재시작하지 않았다.

### 개인 목표·공통 4면 coupled 실험 연결

- `stage2_coupled_push_individual_goal` 환경/학습 YAML·train/test/VNC/eval을 추가했다. 공동2–5m 이동 벡터를 초기 분할 토큰마다 적용하고 개인 AT 연결·보상·도착 예외·현재 성공 포화를 사용한다. 공유 중심 목표는 물리 경로와 공통 외곽4면 선택에만 사용한다. 목표 방향60° 후보 중 담당 구역까지 그룹 최대 손 거리가 가장 작은 면을 선택하고 전원 gate도 같은 면으로 검사한다. 별도 `aligned` 조건은 제거했다. 기존 coupled/goal5·독립 행동은 유지하고 저장본 계약을 분리했다.
- CPU83개 통과(없는 기본21k 저장본을 요구하는 기존 전이1개 제외). GPU6·1024환경 CHECK epoch2/frame262144 저장·finite·actor encoder73 tensor/입력 RMS 동결·action head6 tensor 갱신 확인.32환경8preset의 거리/개인 목표 연결·고정 목표·부분 reset·정책/AMP finite 검사와4환경32step 실제 렌더를 통과했다. [학습 검사](output_etc/check/coupled_individual_goal/training_validation.json), [장면 검사](output_etc/check/coupled_individual_goal/scene/physics_validation.json), [목표 표시](output_etc/check/coupled_individual_goal/viewer/frame_32.png).
- GPU6·32환경·seed0·1200step 평가를 완료하고 개인/그룹 동시 성공의 ever/final 집계를 연결했다. CPU 재채점·JSON 건수 일치, float32/64 개인/BEFORE 지표 차이0·상태 finite·입력 checkpoint 해시 보존을 확인했다. [재채점](output_etc/check/coupled_individual_goal/eval/general_seed0/rescore.json). config/structure/USAGE/README·상세 설계를 갱신했다. 본학습은 시작하지 않았으며 기존 학습 프로세스는 유지했다.

### PUSH·CARRY AT 목표구 반지름20cm 통일

- 요청에 따라 모든 설정의 AT 목표구를 생성 시 반지름20cm로 통일하고 PUSH 와이어 구를25cm에서20cm로 줄였다. 바닥 링은35cm로 유지하며 reset 중 scale 변경은 추가하지 않았다.
- URDF 시각 반지름0.1m×생성 scale2.0=0.2m·Python 구문·diff 검사를 통과했다. 이번 크기 조정의 viewer 실행 검증은 하지 않았다.

### PUSH viewer 표시 롤백 및 선 표시 재구현

- 사용자가 viewer에서 물체가 날아가는 비정상 동작을 보고해 직전 PUSH 표시 이식 전체를 롤백했다. reset 시 marker scale을 변경하는 코드·표시 모듈·호출부를 제거하고 기존 목표구 표시와 문서를 복구했다. 짧은 관측/렌더 검사만으로 정상 동작을 판단한 기존 검증은 불충분했다. 기존 학습·컨피그 변경은 보존했다.
- 이후 기존 목표구·actor·scale·reset을 유지하고 `add_lines`로 PUSH 목표의 바닥 링·세로선·외곽 와이어 구만 덧그리도록 다시 구현했다. 공유 목표 선은 노란색이며 일반 AT에는 추가 선이 없다. 공통 reset 파일은 변경 전과 동일하다.
- GPU5의 실제 viewer에서5500 정책·4환경·동일 seed의 기존/선 표시 실행을 각각1200step 비교했다. 전체 actor root state·정책 관측·행동이 bitwise 동일하고 최대 차이0·finite다. [궤적 비교](output_etc/check/push_goal_viewer_safe/trajectory_comparison.json). 일반 AT·독립 PUSH·2명 두 그룹·4명 공동의 선 표시만 호출하는 것도 확인했다. [표시 검사](output_etc/check/push_goal_viewer_safe/draw_only_verification.json).

### Coupled goal5 5500 간단 평가

- 최신 정기5500 저장본을 GPU5·32환경·seed0·1200step 첫 episode1회로 평가했다. 공동 PUSH의 실제 상자 목표 도달은2명6/24(25.0%)·3명1/7(14.3%)·4명1/9(11.1%)다. 혼합 장면의 독립 역할은 SIT1/1·CLIMB2/2·AT5/6·ON_TOP5/6·PUSH1/8이며 점/영역 결과는 같다. 사람 번호별 공동 참여 성공은 별도 JSON에 저장했다. [상세](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/breakdown.json).
- CPU 재채점과 GPU·JSON 건수가 일치하고 float32/64 성공 차이0·상태 finite·입력 checkpoint 해시 보존을 확인했다. 낙상1/128명이다. 소표본 검사이며 본학습 프로세스는 유지했다. [검증](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/verification.json).
- 같은 궤적의1200step final은 공동2명4/24·3명0/7·4명0/9, 독립 AT4/6이며 나머지 독립 행동은 ever와 같다. 이전 낙상자를 제외하고 final이 ever에 포함됨을 확인했다. [final 상세](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/final_breakdown.json).

## 2026-10-30

### 공동 PUSH goal5 컨피그 추가

- `stage2_coupled_push_goal5` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. 공동 PUSH는 사람→실제 상자 중심2–10m·상자→목표2–5m, 독립 PUSH는 두 거리2–10m다. 기존 coupled YAML은 원래 설정으로 복구했으며 사람 초기 배치·그 외 환경/학습 값은 유지했다. 새 실험의 기본 전이는 제공26k이고 출력·저장본 계약은 분리했다.
- CPU28개 통과(공통 모델26·새 설정 로딩2, 기본21k 파일이 필요한 기존 전이1개 제외). GPU5의1024환경 CHECK가 epoch2/frame262144 저장 후 정상 종료했다. 모델/정규화/optimizer finite·actor encoder73 tensor와 입력 RMS 동결·action head6 tensor 갱신을 확인했다. [학습 검사](output_etc/check/coupled_goal5/training_validation.json).
- 전용 test wrapper로 저장본을4환경·32step·4명 공동 PUSH로 headless 재로드했다. 셸/Python 구문·새 YAML/문서 경로를 확인하고 config/structure/USAGE/README·공동 PUSH 상세를 갱신했다. 장기 본학습·수렴 성능·실제 GUI/VNC 화면은 검증하지 않았다.

### 모션 데이터 연결 복구

- loco·sit·climb·carry 모션은 `/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/`의 각 원본으로, PUSH 모션은 `/home/hwanhee/ksh/stage1_2_jointpush/tokenhsi/data/jointpush_motions`로 심링크를 연결했다. 외부 원본은 수정하지 않았다.
- 현재 PUSH 모션 YAML의 파일 참조161개가 모두 존재함을 확인했다. 학습은 실행하지 않았다.

### 공동 PUSH 거리2–5m 분석

- 공동 PUSH의 두 XY 거리를2–5m로 줄이는 안을 검사했다.32환경의41개 공동 그룹에 각각104만 후보를 검사하고, 실패4환경은 위치·회전 구간 검사를 통해 고정된 사람·상자 크기에서 거리/뒤쪽/여유 조건을 동시에 만족할 수 없음을 확인했다. [분석](output_etc/check/coupled_distance_analysis/summary.json).
- 공통 Stage2 CPU23개 통과(기본21k 파일이 필요한 전이1개 제외). 제공26k의165 tensor 전이와 입력 RMS 로드를 확인했다. 이후 사람→상자2–10m·상자→목표2–5m 조합은32환경의 혼합/4명 공동/독립 PUSH reset·짧은 시뮬레이션을 통과했다. 사람 초기 배치는 변경하지 않았고 기존 coupled YAML은 두 거리2–10m로 복구했다.
- 최초 검증은 GPU1에서 실행 후 종료했다. 이후 사용자 지시에 따라 GPU5·6만 사용하며 GPU5에서 같은 배치 오류를 재현했다. [로그](output_etc/check/coupled_distance_2to5/scene_gpu5.log), [초기 배치](output_etc/check/coupled_distance_2to5/scene/reset_failure.json).

## 2026-10-29

### PUSH viewer 목표구 확대

- PUSH 포함 설정의 viewer/VNC/영상 목표구를 반지름3cm에서20cm로 확대했다(지름 약6.7배). 공통 marker 생성부의 표시 scale만 변경했으며 목표 좌표·성공 판정·학습 설정과 비PUSH 설정의 marker 크기는 유지했다.
- coupled1환경의 동일 카메라에서 기존/확대 목표구를 offscreen 렌더링해 가시성을 확인했다. marker scale 변경 전후 정책 관측이 bitwise 동일하고 짧은 시뮬레이션 관측/보상 finite를 확인했다. [확대 화면](output_etc/check/push_goal_marker/render/after.png), [검증](output_etc/check/push_goal_marker/render/verification.json). USAGE에 표시 크기를 기록했다.


### PUSH26k 평가와 Stage2 제로샷 검증

- `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`를 검사했다.5과제/4family·165 tensor·모델/RMS finite를 확인했고 두 입력 PTH의 실행 전후 SHA256은 동일하다. 제공78k는 기존 기준78k와 해시가 같고 새 pth 경로의4환경32step headless 재로드를 완료했다.
- 26k Stage1 PUSH는512환경×3seed·900step 일반3회/1명 행동별 단독15회를 완료했다. 점/영역5과제 macro는 일반 **84.55/88.32%**, 단독 **82.20/86.95%**다. PUSH 점 성공률은 일반87.12%·단독88.61%로21k보다 높지만 다른 과제의 하락으로 전체 점 macro는 낮았다.21k와18개 초기 장면의9개 필드가 bitwise 일치한다.
- near50·coupled는26k tensor165개를 그대로 복사하고 공통 coordination identity/Linear bias0인 제로샷 저장본을 각각 만들었다. 추가 학습0회·두 모델 tensor 동일을 확인했다. 각16환경·4명·1200step 실행과4환경32step headless viewer 재로드를 완료했다. BEFORE ON_TOP 쌍/4명 공동 PUSH viewer 명령을 USAGE에 추가했다. 관찰용 파일에는 optimizer가 없고 기본 YAML 전이21k는 변경하지 않았다.
- 총20개 trace의 CPU/GPU·JSON 건수 일치, float32/64 ever/simultaneous 차이0·finite를 확인했다. [리포트](output_etc/check/push_26k/report.md), [검증](output_etc/check/push_26k/verification.json). Stage2 결과는 작은 제로샷 실행 검사이며 학습된 Stage2 성능·실제 GUI 화면 검사는 아니다. 기존 원본/프로세스는 보존했다.


### BEFORE PUSH near50 거리 변형 추가

- `stage2_before_push_near50` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. `stage2_before_push`의 mix226 대신 참고 설정의 `relative_sum` near50 거리만 적용했다. 기존 모든 YAML은 보존하고 공통 모델·PUSH/독립 행동 비율·4명/1024환경/8토큰/1200step·loco·21k 전이는 유지했다. RSI433과 attention mask는 이식하지 않았다.
- near 모드는 BEFORE 쌍의50%이며 SIT/CLIMB는 후행→받침 거리가 선행→받침보다 작고, ON_TOP은 후행→받침+후행→payload 거리 합이 선행→받침보다 작다. 나머지50%는 original 샘플링이다. validator·viewer/eval의 새 이름/모션/task 경로·checkpoint 혼용 거부를 연결했다.
- CPU313개 통과.1024환경 CHECK epoch2/frame262144 저장·모델/RMS/optimizer finite·actor encoder73 tensor/입력 RMS 동결을 확인했다.32환경5preset에서 near 부등식·PUSH 경로/물리·alias 관측·부분 reset·가중치 불변을 확인했고4환경32step headless viewer와16환경×seed0·1200step 일반 평가를 완료했다. CPU/GPU·JSON 건수 일치, float32/64 ever 차이0·finite다. [검증](output_etc/check/before_push_near50/verification.json).
- config/structure/USAGE/README·BEFORE PUSH 상세에 설정과 명령을 갱신했다. 장기 본학습·512환경×3seed 성능·GUI/VNC 실화면은 실행하지 않았다.


### 공통 Stage2 alias 전처리와 coordination 초기화 수정

- 공통 `Stage2PushBuilder`의 alias 처리를 config 플래그 대신 task packet의 BEFORE 의존 관계로 결정하도록 통일했다. encoder용 복사본만 변환하고 복제 물체 token feature/pose·원본 packet·coordination relation은 보존한다. coupled·독립 packet은 그대로 통과한다. 기존4과제 BEFORE 구현은 유지했다.
- 새 coordination의 모든 Linear bias를 원본처럼0으로 초기화했다. 초기 출력 identity와 Stage1 PUSH165 tensor 전이는 유지하며 기존 학습 PTH 로드 시 저장된 bias를 덮어쓰지 않는다. coupled 제로샷 PTH를 같은 경로에 새 초기화로 재생성하고 이전 관찰용 저장본은 `output_etc/check/stage2_common_fix/previous_zeroshot.pth`에 보존했다.
- CPU305개 통과. BEFORE SIT/CLIMB/ON_TOP·두 쌍·coupled·solo의 같은 가중치/입력에서 config에 무관하게 actor/critic 출력 bitwise 일치, 기존 encoder 동작·복제 feature/pose·원본 packet 보존을 확인했다. 실제 GPU BEFORE 관측과 기존 학습 가중치의 교차 실행도 action/critic 최대 차이0이다.
- 두 설정 모두4명·1024환경 CHECK epoch2/frame262144 저장, 모델/RMS/optimizer finite·actor encoder73 tensor/입력 RMS 동결을 확인했다. BEFORE32환경5preset의 alias 관측/부분 reset/finite 검사와 재생성 제로샷4환경·32step headless 재로드를 완료했다. 이전/새 제로샷의 coupled·solo 출력은 동일하다. [검증](output_etc/check/stage2_common_fix/verification.json). 구조/설정 상세 문서를 갱신했고 장기 성능·실제 GUI 화면은 검증하지 않았다.


### Coupled 제로샷 viewer 저장본

- 제공 Stage1 PUSH21k의 기존165 tensor를 그대로 복사하고 공통 Stage2 coordination을 identity로 초기화한 `output_etc/check/coupled_zeroshot/Stage2CoupledPush_ZeroShot_Stage1_21000.pth`를 만들었다. 추가 optimizer step0·tensor 일치·identity 초기화를 확인했다. 원본 PTH는 보존했다.
- `TASK_GRAPH=all_coupled`로4환경·32step headless test wrapper 재로드를 완료했다. 실제 GUI 화면은 실행하지 않았다. 제로샷 viewer 명령을 USAGE에 추가했고 전이 기록은 `output_etc/check/coupled_zeroshot/transfer.json`에 있다.


### 공통 Stage2 PUSH 모델과 coupled 이식

- `stage2_coupled_push` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. 원격 `stage1_2_jointpush_st`의 coupled straight·밀도3·인원별 공동 상자/구역·그룹 gate·goal saturation을 이식했다. 독립 행동과 AMP 혼합은 현재 Stage1 PUSH 설정(AT25/ON_TOP20/SIT10/CLIMB15/PUSH30)을 읽도록 바꾸고 미사용 Stage2 RSI/adapter 스택은 가져오지 않았다.
- `stage2_before_push`와 coupled는 동일 `Stage2PushBuilder`·관측1272/AMP1330·5과제·NONE/SELF/BEFORE/COUPLED4종 사람 relation·6N task/group suffix를 사용한다. tensor 이름·크기가 같고 Stage1 PUSH21k의165 tensor를 그대로 전이한다. 기존 BEFORE PUSH의5N 관측 검사 저장본은 새 모델에 resume하지 않는다.
- 사용자 기준대로 Stage1은900step을 유지하고 모든 Stage2 학습/일반 평가/viewer 기본 시간을1200step으로 변경했다. 기존4과제 BEFORE 모델은 보존했다. coupled 평가의 PUSH 성공은 가상 구역이 아닌 실제 전체 상자 중심에서 계산한다.
- CPU292개 통과. 두 Stage2 모두4명·1024환경 CHECK가 epoch2/frame262144 저장 후 종료했고 coupled resume은 epoch3/frame393216까지 완료했다. actor encoder73 tensor/입력 RMS 동결·action head6 tensor 갱신·모델/RMS/optimizer finite를 확인했다.
- coupled32환경8preset·BEFORE32환경5preset의 배치/구역/AMP/부분 reset/가중치 불변을 확인했다. 두 설정 모두 일반32환경×seed0·1200step 및 headless viewer4환경·32step 재로드를 완료했다. CPU/GPU·JSON 건수 일치, float32/64 ever/simultaneous 차이0·finite다. [검증 보고](output_etc/check/coupled_integration/report.md), [전체 결과](output_etc/check/coupled_integration/verification.json).
- config/structure/USAGE/README·BEFORE 상세와 [공통 Stage2/coupled 상세](markdowns/reference/stage2_coupled_push.md)를 갱신했다. Stage1 YAML·원본 checkout·외부 PTH·기존 프로세스는 보존했다. 장기 본학습·정규512환경×3seed 성능·GUI/VNC 실화면은 검증하지 않았다.


### Stage2 BEFORE solo PUSH 추가

- `stage2_before_push` 환경/학습 YAML과 전용 train/test/VNC/일반 eval을 추가했다. 독립 역할만 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%를 사용한다. 기존 0.8 쌍 구성·선행 AT·크기 호환 후행 SIT/CLIMB/ON_TOP 균등 선택·mix226·공유 alias·AT gate/progress는 유지한다. 기본4명·1024환경·8토큰·900step·모두 loco다.
- PUSH 전용 큰 상자 bank를 추가해 기본20개 물리 후보에서 배정하며 일반 distractor/BEFORE 받침으로 섞이지 않게 했다. Stage1 PUSH의 밀도3·167종·cone90·거리·접촉 가중치를 적용하고 다중 사람 경로/활성 장애물 검사를 추가했다. 정책5과제·AMP4family를 기존 BEFORE coordination에 연결해 기본21k PUSH에서 tensor165개를 그대로 전이한다. 기존5개 환경/학습 YAML은 변경하지 않았다.
- CPU280개 통과. 최종 CHECK1024환경은 epoch2/frame262144, resume은 epoch3/frame393216 저장을 완료했다. actor encoder73 tensor·입력 RMS 동결, action head6 tensor 갱신, 모델/정규화/optimizer finite를 확인했다. [학습·재개](output_etc/check/before_push/training_validation.json).
- 32환경7개 preset에서 물체 alias/숨김·거리·밀도/마찰·PUSH 경로·AT progress·AMP·부분 reset·가중치 불변을 확인했다. 최종 저장본 일반32환경×seed0·900step과 headless viewer4환경·32step 재로드를 완료했다. 평가 CPU/GPU/JSON 건수 일치·float32/64 ever 차이0·finite를 확인했다. [검증 보고](output_etc/check/before_push/report.md), [전체 검증](output_etc/check/before_push/verification.json). 이는 실행 검사이며 장기 학습 성능·512환경×3seed 평가·GUI/VNC 실화면은 검증하지 않았다.
- config/structure/USAGE/README와 [설정 상세](markdowns/reference/stage2_before_push.md)를 갱신했다. 원본 checkout·외부 checkpoint·기존 실행 프로세스는 보존했다.


### Stage1 PUSH 행동 비율 변경

- 사용자 지정대로 `stage1_push`의 새 학습 크기/행동 prior를 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%로 변경했다. 검증기의 허용 계약·샘플링 테스트·현재 PUSH/USAGE/평가 문서를 갱신했다. 과거 저장본의 분포와 평가 결과는 보존하며21k PTH의 기본 viewer/일반 평가는 저장된PUSH50%를 복원한다.
- CPU35개 통과:8만 사람 슬롯의 크기 조건부 샘플링 빈도·합계1·설정 검증·이전 저장본 복원/호환. [검증](output_etc/check/push_sampling_30/tests.log), [설정](output_etc/check/push_sampling_30/config_check.json). 비율 변경 검증에 GPU 학습/평가는 실행하지 않았다.


### Stage1 PUSH 통합과900step 통일

- `stage1_push` 환경/학습 YAML·train/test/VNC·일반/1명 단독 eval을 추가했다. scratch prior는 SIT7.5/CLIMB18.75/AT24.375/ON_TOP24.375/PUSH25%로 기존 네 과제 상대 비율을 유지한다. 정책은5과제·AMP4family로 확장하고 기존4과제/3family 기본값은 보존했다. PUSH167종·밀도3·cone90·접촉 RSI bank/모션을 이식했으며 모션21개와21k PTH는 외부 원본에 심링크로 연결했다.
- 학습·일반/단독 평가·viewer 기본 시간을900step으로 통일하고 시간 종료의 `-1` 경계를 수정했다. 새5과제 macro와 기존4과제 macro를 구분한다. PTH 옆 sidecar가 없으면 내부 저장 env/sim을 읽는다. 제공21k 일반 평가는 학습 당시 PUSH50% 분포를 복원하고 새 scratch는PUSH25%다. 서로 다른 학습 계약의 resume는 거부한다.1명 PUSH와1명 ON_TOP의 물체/받침 인덱스, 여섯 번째 RSI 통계 축을 일반화했다.
- 제공 `Stage1JointPushDensity3Cone90Finetune_00021000.pth`로512환경×3seed 일반3회·단독15회, 모두900step을 완료했다.5과제 macro 점/영역은 일반 **85.84±1.07 / 88.80±0.81%**, 단독 **83.32±0.51 / 88.78±0.32%**다. PUSH는 일반 **81.78±1.93%**, 단독 **81.51±1.47%**다.18개 trace CPU/GPU·JSON 건수 일치, float32/64 ever 차이0, 상태 finite를 확인했다. [리포트](output_etc/check/push_900/report.md), [검증](output_etc/check/push_900/verification.json).
- 이전78k1200step trace의 앞900frame을 별도로 재채점해 같은 제한 시간의 비교를 추가했다. 기존 원본 trace/보고서는 보존했다. 기존 네 과제의 단독 평균은21k에서 낮아졌으며 수치와 비교 조건은 리포트에 기록했다.
- CPU259개 통과. 실제78k→PUSH 전이 후 기존 네 과제 actor/critic 출력 보존, 원본 브랜치 정책·task encoder·clean scene 관측 코드 일치를 확인했다. Stage1/BEFORE 기존 PTH4환경·32step 재로드도 통과했다. PUSH scratch CHECK는2048환경으로 RSI5517 profile 준비 후 epoch2/frame262144를 저장했고 resume은 epoch3/frame393216까지 진행했다. 모델/RMS finite, scratch 시작·PUSH25%·900step 계약을 확인했다. [학습/재개](output_etc/check/push_900/training_validation.json). 원본 PTH/코드/기존 프로세스는 보존했으며 장기 본학습·GUI/VNC 실화면은 실행하지 않았다.


### 일반 평가와 Stage1 전용 단독 평가 분리

- 일반 `*_eval.sh`는 checkpoint 저장본의 학습 인원·물체·크기·과제/BEFORE 배치를 유지한다. Stage1만 `stage1_single_eval.sh`로1명·행동별 독립 평가를 제공한다. 단독 AT만 기존9종 크기, 나머지는 학습 크기/동적 물리를 따른다. viewer native 성공 진단은 `viewer_diagnostics.py`로 이름과 결과 태그를 분리했다.
- 정식 평가는 loco·정확히1200step·독립 seed0/1/2×512환경 첫 episode, 같은 궤적의 점/영역 ever·seed 평균/표본 std·macro다. 개인 낙상 frame까지 채점하고 해당 관절 drive를 해제하며 다른 사람은 계속 수행한다. BEFORE의 개인/follower 개인·둘 다 ever·동시 성공을 별도 집계한다. 학습 reward/종료 조건은 유지했다. viewer 기본 horizon도1200으로 맞췄다.
- Stage1 원본78000으로 일반3회(2명·4물체), 단독12회(행동별1명)를 완료했다. macro 점/영역은 일반 **95.17±0.84 / 97.17±0.18%**, 단독 **96.58±0.26 / 98.83±0.40%**다. [리포트](output_etc/check/eval_unified/stage1_78k_report.md). 15개 trace CPU float64/GPU·JSON 건수 및 float32/64 ever가 전부 일치했고 상태 finite·원본 checkpoint 해시 보존을 확인했다.
- CPU242개 통과. BEFORE 세 설정은 검사 저장본으로16환경·4명·1200step 실행/재채점(낙상8/10/15명, 각각24pair), viewer는4환경·32step의 새 진단 경로를 확인했다. Stage2 본학습 성능과 GUI/VNC 화면은 평가하지 않았다. [검증](output_etc/check/eval_unified/verification.json), [BEFORE](output_etc/check/eval_unified/before_validation.json). 새 규약·명령·코드 위치를 evaluation_single/config/structure/USAGE에 갱신하고 이전 단독 규약은 legacy에 보존했다.


### 네 실험 이름 통일

- 환경/학습 YAML·전용 train/test/VNC12개·실험/출력 이름·테스트·현재 문서를 `stage1`, `stage2_before_nonemask_rsi433_near50`, `stage2_before`, `stage2_before_rsi433`로 통일했다. Stage1 보조 구현은 `stage1_config.py`, `stage1_geometry.py`로 바꿨으며 기본 전이 링크는 `checkpoints/Stage1_00078000.pth`다. 외부 원본 파일과 과거 결과 경로는 보존했다.
- 기존78000 및 이전 BEFORE checkpoint의 reward 태그·RSI 경로를 읽기 경계에서 변환한다. 보상·관측·모델 tensor·실험 간 혼용 거부는 유지한다. 기존 네 저장본을 새 test wrapper로4환경·32step headless 재로드해 정상 종료했고 네 모델의 전체 tensor·actor/critic 출력은 변경 전과 bitwise 동일했다. [재로드](output_etc/check/config_rename/smoke.json), [모델](output_etc/check/config_rename/model_equivalence.log). CPU224개 통과. 셸/Python/YAML·문서 링크·이전 실행 이름 잔여 검사 결과는 `output_etc/check/config_rename/`에 있다. GUI/VNC 실화면과 본학습은 실행하지 않았다.

### 네 컨피그와 학습·추론 경로 정리

- `stage1_v2_fixed`, `stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200`, `stage2_before_ground_mix226_atprogress_objalias_1200`, `stage2_before_ground_mix226_atprogress_objalias_rsi433_1200`만 유지했다. 나머지21쌍 YAML·63개 wrapper와 공동 운반·공유 받침·adapter·full RSI·carry50·단일인 태스크·전처리·미사용 에셋/모듈을 제거했다. 기존 파일322개를 삭제했으며 과거 문서는 legacy에 보존했다. [변경 목록](output_etc/check/four_config_cleanup/change_manifest.json).
- BEFORE의 공통 기하·배치·물리 검사·native 후행 RSI를 분리하고 모델을 공통 정책의 직접 확장으로 정리했다. 공통 환경의 관측·reset·모션/AMP를 별도 mixin으로 분리했다. 파라미터 이름·관측·보상·거리·초기화·동결/학습 범위를 유지했다.
- 전용12개 wrapper는 공통 `launch.sh`를 사용한다. conda 기본값은 모두 `tokenhsi`, seed는42, test 기본은 viewer다. GPU는 한 번 지정하며 기존 MPS만 연결한다. 설정 로더는 새 학습의 native YAML과 추론/resume의 저장 env/sim을 같은 경로에서 처리한다. `ENV_CONFIG`로 명시적 대체가 가능하며 fixed 임시 YAML 생성·실험 이름별 중복 분기를 제거했다.
- Stage1 78000은 `checkpoints/Stage1V2Fixed_00078000.pth` 심링크로 연결했다. 네 모션 데이터 디렉터리도 공유 원본에 연결했다. 원본 checkout·가중치·기존 학습 PID3432492는 보존했다. 확인 산출물·RSI 캐시·pytest 임시 파일은 `output_etc/check/`에 둔다.
- CPU214개 통과: 네 설정/계약 혼용 거부·실제78000 전이·설정 복원·wrapper/conda 선택·GTA·보상·거리·mask·RSI/fallback·재로드. [CPU](output_etc/check/four_config_cleanup/cpu.log). 네 모델의 전체 tensor·actor/critic 출력·계약이 정리 전과 bitwise 동일했다. [모델](output_etc/check/four_config_cleanup/model_equivalence.log). 네 설정의 전체 reset2회·부분 reset1회에서 관측·사람/상자 상태·크기·목표·물체 연결의 최대 차이0을 확인했다. [reset](output_etc/check/four_config_cleanup/reset_equivalence.json). 후행 RSI sampler의 출력·RNG도 동일했다.
- GPU0/MPS에서 fixed2명·2048환경과 BEFORE4명·1024환경의 CHECK가 각각 epoch2/frame262144 저장 후 종료했다. fixed native RSI6413 profile을 생성했다. BEFORE 세 설정은 source78000·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/scalar finite를 확인했다. mix226 resume은 epoch3 저장 후 종료했다. [학습/재개](output_etc/check/four_config_cleanup/training_validation.json).
- 실제78000 fixed headless 추론과 BEFORE 전용 test의4환경·32step 저장본 재로드를 확인했다. BEFORE32환경의 loco/mix·ON_TOP pickup/carry·단독 등14개 구성에서 alias 관측/pose·활성 물리 수·거리·AT progress·AMP·부분 reset·가중치 불변을 확인했다. [물리 검사 진행 결과](output_etc/check/four_config_cleanup/physics.log), [wrapper](output_etc/check/four_config_cleanup/wrapper_validation.json). 셸/Python/YAML·현재 문서 링크·4쌍 YAML/12개 wrapper·미사용 import/실험 참조 제거를 검사했다. VNC 실행 의존성이 없어 실제 GUI/VNC 화면과 장기 성능은 검증하지 않았다. 본학습은 시작하지 않았다.

## 2026-10-28

- 코드 정리용 독립 복사본을 만들었다. 원본과 복사 범위·초기 검증은 `code_copy_manifest.json`과 [당시 changelog](markdowns/legacy/pre_four_configs/changelog.md)에 보존한다.
