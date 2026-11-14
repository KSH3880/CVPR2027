# 평가 보조 작업

평가 확인·비교·후처리용 보조 스크립트는 이 폴더에서 만든다.

Enabling의 **Coordination Behavior**는 기존45개 rollout을 CPU로 재집계한다. AT 직전 준비율·root footprint 근사 사전 점유율·Post-AT 지연과 공통 성공 장면/민감도를 제공한다. [실행](../markdowns/USAGE.md#coordination-behavior-cpu-재집계) · [결과](../output_etc/check/coordination_behavior_v4_2000_20261114/report.md) · [해석](../output_etc/check/coordination_behavior_v4_2000_20261114/interpretation.md).

SCT26k/Ours V4 2k의 IND 차이를 분석하는 단독/2H·분리/교차·목표 정밀도 평가는 [설계](../markdowns/reference/ind_mechanisms.md)와 [실행](../markdowns/USAGE.md#ind-성능-차이-보조-평가)을 따른다. GPU6/7·512환경·seed1/6/7·900step의 신규24회이며 기존2H6회는 읽기 전용으로 재사용한다.

**최신 논문 Table 1·2·3 평가는 [인수인계](tables123_handoff.md)부터 읽는다.** 아래 과거 실행의 `Table 1/2`·`table2` 명칭은 보존한다. 현재 Table 2는Enabling, Table 3은Coupled이며 실행 폴더에서는 모두 `table2` 아래에 있다. 새 작업은 사용자 지정 GPU6/7 범위를 따르며 아래GPU5 포함 고정 큐를 그대로 실행하지 않는다.

Ours V4 2k의 **32H/64H × Independent/Mixed 탐색**은 `run_v4_large_radius.py`로 실행한다.64환경·seed1·900step이며 GPU6/7에 한 작업씩 배정한다. 조건별 그룹 중심 원 반경10m부터5m씩 증가하고, 준비 포함180초 안에 배치/반경 검증을 통과한 첫 반경에서 평가한다(검색 상한100m). 성능을 보고 반경을 바꾸지 않으며 실패/시간초과 로그를 보존한다. [상태](../output_etc/check/v4_2000_32h64h_radius_env64_seed1_20261112/status.json) · [보고서](../output_etc/check/v4_2000_32h64h_radius_env64_seed1_20261112/report.md). 기존 실행 중이거나 출력이 있으면 중복 실행하지 않는다. 전원 낙상 시900step 이전 종료를 허용하며 실제 frame 수·전원 낙상·CPU 재채점을 검증한다. 기존 큐가900step 고정 검증에 걸린 결과는 `finalize_v4_large_radius.py`에서 CPU 재검증하고 큐 종료 후 보고서를 갱신한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_v4_large_radius.py
```

PDF (15) §4.3의 결과 해석은 `analyze_cooperative_mechanisms.py`로 CPU 분석한다. 사용자 지정대로 Enabling Coop. FT4k·Coupled6k·Ours V4 2k를 사용한다. 기존90개 기록의 Enabling 실패 단계·공통 성공 시간·후행 준비 상태, Coupled 공동 근접 이후 성공·밀기 구간 비율·목표 거리 분포를 비교한다. 추가 지표는 공식 SR/t_coop과 구분하며 원시 기록이 없는 LLM Scheduler는 새 집계에서 제외한다. [분석](../output_etc/check/paper43_analysis_20261112/report.md), [추가 대조](../output_etc/check/paper43_analysis_20261112/supplement.md), [그림](../output_etc/check/paper43_analysis_20261112/failure_partitions.pdf). 실행: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/analyze_cooperative_mechanisms.py`.

정책 추론 속도·메모리는 `benchmark_policy.py`로1/2/4/8/16H·장면1개·M=2N을 비교한다. TokenHSI40.5k/SCT26k/Ours V4 2k의 정규화+actor+평균 행동을 측정하며 입력 변환·물리 시뮬레이션은 제외한다. TokenHSI도 N명을 한 배치로 처리한다. 메모리는 actor/critic/AMP를 포함한 로드 네트워크의 PyTorch 할당량과 추론 피크이며 CUDA context·다른 프로세스·시뮬레이션은 제외한다. 공유 GPU의 참고 측정이며 전용 GPU의 논문용 latency와 구분한다.

```bash
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
exec python eval/benchmark_policy.py --output output_etc/check/<측정명>'
```

[GPU6 간단 측정 결과](../output_etc/check/policy_benchmark_n1_2_4_8_16_20261112/warm_cache/report.md). 세 모델 공통 CUDA/JIT 캐시 워밍업 후20회 워밍업·3구간×50회로 측정했고 초기 캐시 포함 결과는 상위 폴더에 보존했다.

8H R=3m /16H R=7.5m의 배치 정의·실행 경로·완료 결과는 [고정 반경 인수인계](group_radius_handoff.md)를 따른다.

현재 이 checkout의 일반 정식 평가와 `run_stage2_scalability.py`도8H/16H에서 이 반경을 기본 적용한다(`--eval_protocol`·`--group_spawn_over4`). 별도 `GROUP_SPAWN_RADIUS` 지정이 필요하지 않다. 2H/4H·그룹 내부 배치·과제/성공 판정은 유지하며 이전5m 결과를 같은 출력 폴더에서 재사용하지 않는다. 아래 간격 비교용 실행기는 명시한 실험 조건을 유지한다.

같은 반경의 **512환경 × seed1/6/7 정식 평가**는 `run_group_radius_formal.py`를 사용한다. TokenHSI40.5k/SCT26k/FSM/Coop. FT6k의8/16H Independent/Mixed42회와 공통 장면12개이며 Ours는 제외하고 FSM Independent는 SCT와 공유한다. GPU5/6/7별 최대8개, 시작 예약량8H10,000MiB/16H14,000MiB를 반영하고 GPU5는16,000MiB·GPU6/7은12,000MiB 여유를 유지한다. 실제 반경·공통 초기 상태·CPU SR/Error·정책 보존을 검증하며 TokenHSI397D 입력 차이/FSM 기존 첫 관측 차이는 기록한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_group_radius_formal.py
```

[정식 평가 결과](../output_etc/check/table1_group_radius_8r3_16r7p5_env512_seeds167_baselines_20261112/report.md) · [최종 검증](../output_etc/check/table1_group_radius_8r3_16r7p5_env512_seeds167_baselines_20261112/final_audit.json).42회·공통 장면12개 모두 완료했고 실패0이다. 실행 중 같은 큐를 중복 실행하지 않는다.

TokenHSI/SCT/FSM/Coop. FT/Ours의 결과 원본·집계 필드·알려진 차이는 [통합 결과 목록](../../evaluation_protocol/results.md)에서 관리한다. 최신 Coupled 주 값은20cm 조건의 pooled 평균이며 seed 평균±표본표준편차와 구분한다.

V5 **2k Table 1/2/3**는 `run_v5_formal.py`로 별도 실행한다. GPU6/7만, 각 GPU 최대3개·12GB 여유를 유지하며 기존42개 snapshot을 재사용한다.16H Mixed/3H Coupled 선검증 후 전체 큐를 진행하고 `summarize_v5_formal.py`가 V5만의 SR·Time(Succ./All)·최신20cm t_coop을 집계한다. seed 통계와 pooled 값은 구분한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_v5_formal.py
```

[진행 상태](../output_etc/check/v5_2000_tables123_radius_seeds167_20261113/status.json) · [중간 표](../output_etc/check/v5_2000_tables123_radius_seeds167_20261113/partial_report.md). 완료 후 같은 폴더의 `report.md`·`summary.json`·`completion_verification.json`을 확인한다.

V4 **2k 정식 Table 1/2**는 `run_v4_formal.py`를 사용한다. **512환경·seed1/6/7·900step·XY100%·M=2N**의 Ours 신규42회(Table1 24·Table2 18)다. Table1의8H/16H는 그룹 중심 원 반경3m/7.5m의 baseline 공통 snapshot,2H/4H와 Table2는 기존 검증 snapshot을 재사용한다.16H Mixed·Coupled3H를 먼저 검증한 뒤 나머지를 실행하며 GPU5/6/7을 사용한다. 초기 GPU별3개 제한 이후 사용자 요청으로 `accelerate_v4_formal.py`에 기존9개 자식을 재시작 없이 인계했고, 후속 요청으로 신규 작업은 GPU5/6에 최대10개씩 배정하고 GPU7 기존 작업은 유지한다. 실제 PID 메모리를 뺀 시작 예약량·최소10,000MiB 여유를 적용한다. 첫 인계의 세션 종료로 중단된8회는 시도/로그를 보존한 뒤 동일 조건으로 재실행했으며, 두 번째 인계에서는 이전 supervisor만 정지하고 새 supervisor를 독립 세션으로 시작해 자식 실행을 유지했다. 인계 당시 기록은 결과 폴더의 `acceleration/`·`acceleration_adoption.json`·`acceleration_source_hashes.json`에 보존한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_v4_formal.py
```

[상태](../output_etc/check/v4_2000_tables12_radius_seeds167_20261112/status.json) · [중간 표](../output_etc/check/v4_2000_tables12_radius_seeds167_20261112/partial_report.md). 완료 후 같은 폴더의 `report.md`·`summary.json`·`completion_verification.json`을 확인한다. 최신20cm t_coop/Jerk는 저장 궤적 CPU 재집계이며 원시6cm 출력과 구분한다. `summarize_v4_formal.py <결과폴더>`는 기존 비교군 검증 결과와 Ours를 합쳐 표를 만들며 미완료 baseline은 N/A로 표시한다. TokenHSI397D 입력/FSM 기존 첫 관측 차이는 보존·표시한다. 기준은 공용 프로토콜 **v1.1.0**이며 이전5m 결과는 새8H/16H 칸에 넣지 않는다. 신규42회와 비교군 재사용168회·총210개 seed 결과를 검증했고 [최종 표](../output_etc/check/v4_2000_tables12_radius_seeds167_20261112/report.md)·[완료 검증](../output_etc/check/v4_2000_tables12_radius_seeds167_20261112/completion_verification.json)을 생성했다.

V4 **1.5k/2k 선택용 반경 비교**는 `run_v4_group_radius_eval.py`를 사용한다. **64환경·seed1·8H R=3m/16H R=7.5m·Independent/Mixed 총8회**이며 기존 반경 평가의 공통 초기 장면4개를 읽기 전용으로 재사용한다. GPU5/6/7만 사용하며 GPU별 최대3개·시작 예약8,000MiB/worker·여유10,000MiB를 적용한다. V4 config/checkpoint를 명시하고 실제 반경·첫 정책 입력·가중치/RMS 보존·CPU SR/Error를 검증한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_v4_group_radius_eval.py
```

[진행 상태](../output_etc/check/v4_1500_2000_group_radius_env64_seed1_20261112/status.json) · [중간 비교표](../output_etc/check/v4_1500_2000_group_radius_env64_seed1_20261112/partial_report.md). 완료 후 같은 폴더의 `report.md`·`summary.json`·`completion_verification.json`을 확인한다. 기존 실행 중이면 중복 실행하지 않는다. 8회 모두 완료·실패0이며 [최종 비교표](../output_etc/check/v4_1500_2000_group_radius_env64_seed1_20261112/report.md)·[검증](../output_etc/check/v4_1500_2000_group_radius_env64_seed1_20261112/completion_verification.json)을 확인한다. 같은 초기 장면에서2k가 네 조건 모두 높았고 Mixed는8H/16H에서 각각+1.54/+1.55pp다.

그룹 중심 고정 반경의 **정책 SR 확인**은 `run_group_radius_eval.py`를 사용한다. **8H R=3m /16H R=7.5m·64환경·seed1·900step·XY100%·M=2N**, SCT26k/TokenHSI40.5k/Ours V3 2k의 Independent/Mixed 총12회와 공통 장면4개다. **GPU6/7만** 사용하고 준비된 장면부터 최대GPU별10개를 배정하며 실제/시작 예약 메모리를 포함해10,000MiB 여유를 유지한다. 실제 그룹 반경·공통 초기 상태/첫 입력·CPU SR/Error·가중치/RMS·소스 SHA를 검증한다. TokenHSI의397D 입력 차이는 유지한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_group_radius_eval.py
```

[상태](../output_etc/check/table1_group_radius_8r3_16r7p5_env64_seed1_20261112/status.json) · [중간 표](../output_etc/check/table1_group_radius_8r3_16r7p5_env64_seed1_20261112/partial_report.md). 완료 시 같은 폴더에 `report.md`·`summary.json`·`completion_verification.json`을 생성한다. 아래 배치 생성 검사와 구분하며 실행 중 같은 큐를 중복 실행하지 않는다.

그룹 중심 **고정 반경**의 빠른 배치 검사는 `run_group_radius_screen.py`·`group_radius_probe.py`를 사용한다. 그룹 수와 무관하게 중심을 지정 반경의 원 둘레에 놓고 그룹 순서/회전 RNG·그룹 내부 원래 배치·Enable 후행 재배치·거리/크기·distractor 조건은 유지한다. `GROUP_SPAWN_RADIUS`는 이 보조 경로의 옵션이며 사람의 기본 `agentSpawnRadius=2.5`와 다르다. 정책 행동/SR은 실행하지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_group_radius_screen.py \
  --output output_etc/check/group_radius_screen_env64_seed1_20261111/coarse
```

1차는64환경·seed1의 Independent/Mixed 각각8H 반경1.5/2.5/3.5m·16H4/6/8m를 비교한다. 실행당 준비 포함120초·배치시도64회 이내에 전체 장면과 실제 그룹 중심 반경 검증까지 완료해야 통과다. 제한 초과를 기하학적 불가능으로 해석하지 않는다. `--radii8 2 --radii16 7`처럼 범위를 좁히고 이후 `--envs 512 --seeds 1 6 7`로 재확인한다. 한 번에 최대12개를 GPU5/6/7에 분산하며 실제 여유 메모리가 시작 예약량보다 부족하면 실행하지 않는다. [1차 진행/결과](../output_etc/check/group_radius_screen_env64_seed1_20261111/coarse/report.md) · [추가 후보](../output_etc/check/group_radius_screen_env64_seed1_20261111/fine/report.md).

검사64회(64환경28회·512환경36회)를 완료했다. 추가 검사/512환경은 원래1024회 시도 한도를 유지했고 실행당 준비 포함240초 이내 전체 생성·실제 반경 검증을 통과 기준으로 삼았다. 512환경·seed1/6/7·Independent/Mixed6조건을 모두 통과한 가장 작은0.5m 후보는 **8H R=3m·16H R=7.5m**, 더 빨리 생성된 후보는 **8H R=3.5m·16H R=8m**다. 수학적 최소값/정책 SR 결과가 아니며 최종 평가 반경을 확정·적용하지 않았다. [전체 비교·정의](../output_etc/check/group_radius_feasibility_20261111/report.md) · [원시 집계](../output_etc/check/group_radius_feasibility_20261111/summary.json) · [검증/실행 당시 소스](../output_etc/check/group_radius_feasibility_20261111/verification.json).

**3m의16H는 사용자 요청으로 중단했다.** 완료된3m8H 전체21회와16H 일부11회, 총32회는 보존하고 미완료16H10회는 취소했다. 아래3m 큐는 재실행하지 않는다.

대체 **4m·16H 정식 평가**는 `run_group_spacing_4m_16h.py`를 사용한다. **512환경 × seeds1/6/7**, Independent/Mixed·TokenHSI40.5k/SCT26k/FSM/Coop. FT6k의21회와 공통 초기 장면6개다. FSM Independent는 SCT 공유, Ours 제외다. GPU5/6/7과 메모리 예약/검증은3m 정식 실행기를 재사용하고 간격과 평가 인원만4m·16명으로 바꾼다. 기존3m 결과와 별도 저장한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_group_spacing_4m_16h.py
```

[4m 상태](../output_etc/check/table1_4m_16h_env512_seeds167_baselines_20261111/status.json) · [manifest](../output_etc/check/table1_4m_16h_env512_seeds167_baselines_20261111/manifest.json) · [중간 표](../output_etc/check/table1_4m_16h_env512_seeds167_baselines_20261111/partial_report.md). 완료 시16H 두 열의 `report.md`·`summary.json`·`completion_verification.json`을 생성한다. 실행 중 중복 실행하지 않는다.

Table1 **3m 정식 평가**는 `run_group_spacing_formal.py`를 사용한다. **512환경 × seeds1/6/7**, 8/16H Independent/Mixed, TokenHSI40.5k/SCT26k/FSM/Coop. FT6k의42회와 공통 초기 장면12개다. **Ours는 제외**하고 FSM Independent는 SCT 결과를 공유한다. 900step·XY100%·M=2N·기존 샘플링/크기/거리/ordered/distractor 조건은 유지한다. GPU5/6/7별 최대8개, 시작 중 메모리 예약(8H10,000MiB·16H14,000MiB)을 포함해 GPU6/7은12,000MiB·GPU5는16,000MiB 여유를 남긴다. `group_spacing_2m_eval.py`는 GPU 허용 어댑터로 재사용하며 실제 간격은 `SPACING_METERS=3.0`이다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_group_spacing_formal.py
```

[상태](../output_etc/check/table1_3m_env512_seeds167_baselines_20261111/status.json) · [manifest](../output_etc/check/table1_3m_env512_seeds167_baselines_20261111/manifest.json) · [중간 표](../output_etc/check/table1_3m_env512_seeds167_baselines_20261111/partial_report.md). 완료 시 `report.md`·`summary.json`·`completion_verification.json`을 생성한다. 실제 첫 입력/공통 장면·CPU SR/Error·checkpoint/소스 SHA를 검사하며 TokenHSI397D 차이와 FSM 기존 첫 관측 차이는 보존·명시한다. 실행 중 큐를 중복 실행하지 않는다.

병렬 수 확대 시 기존 자식7개는 종료 없이 `concurrency_adoption.json`으로 인계했다. 변경 전 실행기/manifest/상태/SHA는 `concurrency_handoff_before/`, 실행기만 변경된 SHA 전환은 `concurrency_source_transition.json`에 보존한다.

Table1 그룹 간격 확인은 **64환경·seed1 하나**에서 강제 이격 해제(`native`)와 최소3m 원형 그룹 배치(`3m`)를 비교한다. 각 조건은8/16H Independent/Mixed, TokenHSI40.5k/SCT26k/FSM/Coop. FT6k/Ours V3 2k의18회이며 FSM Independent는 SCT를 공유한다. 총36회다. 그룹 샘플러·거리/크기·2k 토큰 예산·distractor 관측/물리·900step·XY100%는 유지한다. FSM의 기존 reset 관측 재계산은 보존하며 실제 첫 입력 차이를 별도 기록한다. 기존512환경 결과와는 환경 수·초기 장면도 달라 이격만의 효과로 해석하지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_group_spacing_comparison.py
```

[진행 상태](../output_etc/check/table1_group_spacing_env64_seed1_20261111/status.json) · [manifest](../output_etc/check/table1_group_spacing_env64_seed1_20261111/manifest.json). GPU6/7별 최대10개 worker, 여유 메모리/시작 중 예약량을 확인하며 최소10,000MiB를 남긴다. 기준이 준비된 평가를 함께 실행한다. 기존64회 배치 재시도 한도가 일부 장면에서 소진돼 수용 조건은 유지하고 두 조건 모두1024회로 늘렸다. [첫 시도 기록](../output_etc/check/table1_no_group_spacing_env64_seeds167_20261111/first_attempt_status.json)을 보존했다. 실행 중 같은 큐를 중복 실행하지 않는다.

`run_group_spacing_comparison.py`가 공통 실행기·준비된 작업 배정기를 조합한다. 기존 seed1 작업은 종료 없이 인계하고 seed6/7은 사용자 지시에 따라 중단했다. 인계 근거·실행기 SHA는 결과 폴더에 저장한다. 간격 진단에서는 Enabling 선행·후행을 한 그룹으로 합치며 최초 실제 정책 입력·CPU SR/Error를 확인한다. 실행 중인 자식이 있으면 supervisor를 임의 재실행하지 않는다.

**0m의16H는 사용자 지시로 중단했다.** 진행 중 기준 생성2개·0m 진단1개를 종료하고 대기 평가9회를 취소했다. 완료된0m8H9회·3m18회는 보존하며 위 비교 큐를 재실행하지 않는다.

**1m·2m도 사용자 지시로 중단했다.** 각 조건의8H9회는 보존하고 미완료16H9회씩은 취소했다. 아래 명령은 실행 기록이며 재실행하지 않는다. 미완료 값은 실패 SR로 집계하지 않는다.

추가1m 조건은 같은64환경·seed1·checkpoint·8/16H Independent/Mixed·18회로 **GPU5만** 사용한다. 기존 GPU6/7 큐와 별도이며, GPU5 최대10개 worker·최소10,000MiB 여유 메모리를 유지한다. [상태](../output_etc/check/table1_group_spacing_env64_seed1_20261111/1m/status.json) · [manifest](../output_etc/check/table1_group_spacing_env64_seed1_20261111/1m/manifest.json) · [중간 표](../output_etc/check/table1_group_spacing_env64_seed1_20261111/1m/partial_report.md). 준비된 기준 장면의 평가를 함께 실행하며 중간 표는 검증 완료마다 갱신한다.

`group_spacing_1m_eval.py`는 Coop. FT 원본의GPU6/7 제한을 이 평가 프로세스에서만GPU5로 바꾼다. 외부 파일은 수정하지 않고 `TOKENHSI_GPU=CUDA_VISIBLE_DEVICES`가 가리키는 실제GPU가5인지 검사한다(MPS UUID 지정 포함). 첫 제한 오류와 진행 중 작업 인계 기록은 `1m/adoption_gpu5_guard.json`에 보존한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_group_spacing_1m.py
```

추가2m 조건은 같은64환경·seed1·checkpoint·18회로 **GPU5/6/7**에 배정한다. GPU별 최대10개 worker·최소10,000MiB 여유 메모리를 유지하고 GPU5의1m 큐에 예약된 시작 메모리도 반영한다. [상태](../output_etc/check/table1_group_spacing_env64_seed1_20261111/2m/status.json) · [manifest](../output_etc/check/table1_group_spacing_env64_seed1_20261111/2m/manifest.json) · [중간 표](../output_etc/check/table1_group_spacing_env64_seed1_20261111/2m/partial_report.md). GPU5에서만 Coop. FT의 기존 평가 어댑터를 사용하고 GPU6/7은 원본 GPU 검사를 유지한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_group_spacing_2m.py
```

배치 거절 진단은 `group_spacing_sampling_probe.py`를 사용한다. seed1·64환경·16H Mixed의 원래 수용 조건/RNG를 유지하고 첫4회만 실행해 PUSH/전체 장면 거절·사람0.7m 미만·단계별 시간을 기록한다. [1m 진단](../output_etc/check/group_spacing_sampling_diagnosis_20261111/1m/sampling_probe.json)은 SR 평가가 아니며 제한 종료 시 layout exhausted는 진단 한도 소진이다.0m 진단은 사용자 지시로 중단했다.

Coop. FT의3k/6k Table1/2는 `run_coopft_tables.py`로 실행한다. 기존4k의 공통 snapshot·원본 Coop. FT 환경/모델 코드를 읽기 전용으로 재사용하며, 이 checkout의 `output_etc/check/`에 새 결과를 저장한다. 512환경·seed1/6/7·900step·XY100%로 신규84회,4k42회는 재사용한다. GPU6/7별 최대8개 worker·실제 여유 메모리와 시작 중 예약량을 확인하고 최소10,000MiB를 남긴다. checkpoint별16H Mixed/Coupled3H 검증 후 본 큐가 시작된다. 실제 첫 정책 입력·checkpoint 가중치/정규화 보존·CPU SR/Error·Table2 replay와 최신20cm t_coop/Jerk를 검증한다. Post-AT/Collision은 포함하지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_coopft_tables.py \
  --workers-per-gpu 8 --reserve-mib 10000
```

[완료 결과·3k/4k/6k 비교](../output_etc/check/coopft_3000_6000_tables12_seeds167_20261111/report.md) · [완료 검증](../output_etc/check/coopft_3000_6000_tables12_seeds167_20261111/completion_verification.json) · [상태](../output_etc/check/coopft_3000_6000_tables12_seeds167_20261111/status.json) · [manifest](../output_etc/check/coopft_3000_6000_tables12_seeds167_20261111/manifest.json). 신규84회 완료·실패0이다. `supervisor.lock`으로 중복 큐를 막으며 실행 중에는 다시 실행하지 않는다. 완료 후 `summarize_coopft_tables.py <결과폴더>`로3k/4k/6k 비교를 다시 집계할 수 있다.

현재 **V4 2k 통합 표의 Time (Succ. / All)**은 `reaggregate_v4_enabling_time.py`로 재집계한다. TokenHSI40.5k/SCT26k/FSM/Coop. FT6k/Ours V4 2k의Enable45개 기록에서 성공은 실제 그룹 완료 시간, 실패는900×기록된control_dt(약30초)를 사용한다. 행동별 평균→세 행동 동일 가중→seed 평균으로 집계하고, 현재 Table2(a) 시간 열에 두 평균을 `Succ. / All`로 표기한다. 각 std/seed 값은 JSON에 저장한다. 기존 통합 report/summary는 `time_succ_all/before/`에 보존하며 SR·Table1·Coupled는 유지한다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/reaggregate_v4_enabling_time.py
```

[현재 다섯 모델 시간표](../output_etc/check/v4_2000_tables12_radius_seeds167_20261112/time_succ_all/report.md) · [원본/집계 검증](../output_etc/check/v4_2000_tables12_radius_seeds167_20261112/time_succ_all/verification.json). 아래는 이전 V3/Coop. FT4k·6k 기록의 별도 재집계다.

Enabling의 **Time (Succ./All)** 재집계는 `reaggregate_enabling_completion_time.py`를 사용한다. 기존512환경·seed1/6/7·900step 기록에서 Succ.는 최초 그룹 동시 성공 시간의 성공 trial 평균, All은 실패에 최대 episode 시간(900×기록된dt, 약30초)을 부여한 전체 trial 평균이다. 행동별 평균 → 세 행동 동일 가중 → seed 평균±표본std이며 원래 SR·성공 Time·원본은 유지한다. 표의Coop. FT4k와 추가6k를 구분하고 외부 LLM Scheduler는 제외한다. [결과·seed별 수치](../output_etc/check/enabling_completion_time_succ_all_seeds167_20261111/report.md) · [검증](../output_etc/check/enabling_completion_time_succ_all_seeds167_20261111/verification.json).

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/reaggregate_enabling_completion_time.py
```

Enabling Post-AT Time 추가 분석은 `reaggregate_enabling_post_at_time.py`를 사용한다. 최초 유효 그룹 동시 성공−최초 유효 AT 성공을30Hz 초로 변환하며 그룹 성공 trial만 포함한다. 행동별 평균 → seed별 세 행동 동일 가중 평균 → seed 평균±표본표준편차다. 원래 Completion Time·SR은 유지한다. [결과·seed별 값](../output_etc/check/enabling_post_at_time_seeds167_20261111/report.md), [충돌 지표 문헌 검토안](../output_etc/check/enabling_post_at_time_seeds167_20261111/collision_metric_review.md). 충돌 정의·문턱은 미확정이며 새 충돌 평가 코드는 없다.

현재 V4 2k/Coop. FT6k로 맞춘 다섯 모델의 Post-AT은 `reaggregate_v4_enabling_post_at_time.py`를 사용한다. 기존 TokenHSI/SCT/FSM27개를 원시 SHA 검증 후 재사용하고 V4/Coop. FT6k18개를 CPU 재계산했다. [현재 결과](../output_etc/check/v4_2000_tables12_radius_seeds167_20261112/post_at_time/report.md) · [검증](../output_etc/check/v4_2000_tables12_radius_seeds167_20261112/post_at_time/verification.json). 기존 Table1/2·Time Succ./All은 변경하지 않는다. 실행: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/reaggregate_v4_enabling_post_at_time.py`.

TokenHSI/Ours V4의 성공 표본 선택 효과는 `analyze_enabling_success_selection.py`로 분석한다.4608개 동일 초기 장면의 선행/후행 XY 경로·공유 상자 크기·회전각과 각자 성공/공통 성공402개에서의 AT·완료·Post-AT 시간을 비교한다. 거리 구간별 SR과 장면별 CSV를 별도 저장하고 기존 지표는 변경하지 않는다. [거리·시간 비교](../output_etc/check/v4_2000_tables12_radius_seeds167_20261112/success_selection_tokenhsi_vs_v4/report.md).

실제 이동속력은 `analyze_enabling_movement_speed.py`로 CPU 재집계한다. 같은 두 모델의 공통 성공/각자 성공에서 사람 접근(중심거리1m), 물체 운반(초기 수평 변위20cm 이후), 사람 전체·후행 AT 전후를 구분한다. 수평 궤적 길이÷시간이며 정지 프레임도 포함한다. 접근0.5/1.5m 및 이동 프레임 문턱0.2/0.5m/s는 보조 민감도 분석이다. 구간 없는 trial은 제외·건수를 기록하고 논문 지표나 기존 표는 변경하지 않는다. [구간별·행동별 결과](../output_etc/check/v4_2000_tables12_radius_seeds167_20261112/movement_speed_tokenhsi_vs_v4/report.md). 실행: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/analyze_enabling_movement_speed.py`.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/reaggregate_enabling_post_at_time.py --workers 3
```

**평가 기준은 [공용 프로토콜](../../evaluation_protocol/README.md) 한 곳을 따른다.** 현재 버전 `cvpr2027-evaluation-v1.1.0`, Table1의8H/16H 그룹 중심 반경3m/7.5m·공동 PUSH 주 시작 변위20cm. 레포별 연결·checkpoint·알려진 차이는 [구현 안내](../../evaluation_protocol/implementations.md), 계산 예제는 [공통 검증](../../evaluation_protocol/checks/README.md)을 본다. 아래 기존6cm/구간 내 비율은 원시 기록기의 과거 출력이며 최신 t_coop/Jerk는 별도 재집계의20cm 결과다.

공동 PUSH의 TeamHOI 방식 t_coop·Mean Absolute Jerk는 저장 궤적에서 CPU로 재집계한다. SCT26k·기존 FSM·Coop. FT4k·Ours V3 2k, 2/3/4명·512환경·seed1/6/7·900step의36개 기록을 사용한다. 출발 변위20/50cm를 따로 산출하며 메인 eval·기존 기록은 변경하지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/reaggregate_coupled_cooperation.py --workers 3
```

t_coop은 전원 양손–실제 전체 상자 표면<12cm·힘/추가 생존 필터 없이 전체900frame의 공동 프레임 수를 원본 counter 차이로 나누고 clip한다. Jerk는 보존한 원본 함수의 float32·dt0.0333·중간 높이 둘레64점·순차 차분/마스크·0 trial 제외 규칙이다. 이 정의는 아래의 기존 구간 내6cm t_coop과 구분한다. [결과·seed별 표](../output_etc/check/coupled_cooperation_four_models_20261111/report.md), [프로토콜 사본](../output_etc/check/coupled_cooperation_four_models_20261111/reference/coupled_push_tcoop_jerk_protocol.md), [검증](../output_etc/check/coupled_cooperation_four_models_20261111/verification.json). 합산 trial 평균과 seed 평균±표본표준편차를 따로 저장하며, FSM은 첫 정책 입력의 관측 불일치가 확인된 기존 궤적의 참고값이다.

추론 시 bias 제거는 `bias_ablation_eval.py`·`run_bias_ablation.py`를 사용한다. Stage1 26k의 Task-Relation Bias는 SCT의 `bias_table()` 출력만, V3 2k의 Dependency-Relation Bias는 DCT의 `effective_relation_bias()` 출력만0으로 만든다. Actor/critic에 모두 적용하며 checkpoint·관계 할당·공유 임베딩·feature fusion은 보존한다. Stage1 원본 정책 검사는 제거 적용 전에 수행한다. 실제 첫 정책 입력과 canonical 관측의 exact 일치·제거 옵션 off의 행동 exact·probe 이후 RNG 복원·가중치/정규화 불변과 매 frame bias 호출을 검증한다. 관측을 재계산하거나 FSM 경로를 호출하지 않는다.

```bash
/home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_bias_ablation.py \
  --output output_etc/check/bias_ablation_20261110 --seeds 1 6 7 --workers-per-gpu 5

/home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/summarize_bias_ablation.py \
  output_etc/check/bias_ablation_20261110
```

GPU6/7에 각각 최대5개 평가를 실행한다. Stage1 Independent1/2/4/8/16H의15회·V3 Mixed2/4/8/16H의12회와 같은 프로토콜의 Stage1 1H baseline3회, 총30회다. 각512환경·900step·XY100%·M=2N이며 다른 baseline24회는 기존 Table1 결과를 재사용한다. 1H 지원은 이 보조 entry의 Independent 검증에 한정한다. 결과·실행 상태·로그는 위 output의 `report.md`·`summary.json`·`status.json`·`logs/`에 저장한다. `test_bias_ablation.py`는 bias 제거/복원·공유 fusion 출력과 가중치 보존을 확인한다. 아직 실행 중이면 같은 큐 명령을 중복 실행하지 않는다.

위 seed1/6/7의30회는 완료됐다. [최종 비교표](../output_etc/check/bias_ablation_20261110/report.md), [검증](../output_etc/check/bias_ablation_20261110/completion_verification.json).

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
