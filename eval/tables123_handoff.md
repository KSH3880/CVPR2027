# Table 1·2·3 평가 인수인계

2026-11-12 · PDF `CVPR2027_tmp_2 (15).pdf` · 계산 규약 `cvpr2027-evaluation-v1.1.0`.

**이 문서부터 읽고 [공용 평가 규약](../../evaluation_protocol/README.md)을 적용한다.** 기존 결과를 재사용할 때도 checkpoint·초기 snapshot·평가 조건·검증 기록이 일치해야 한다. 새 checkpoint는 별도 출력/manifest로 평가한다.

## 1. 평가할 표

| 표 | 장면 | 출력 | 모델당 실행 |
| --- | --- | --- | ---: |
| Table 1 — Versatility & Scalability | 2/4/8/16H × Independent/Mixed 각각 | Independent Interaction Success Rate / Mixed Interaction Group Success Rate | 24회 |
| Table 2 — Enabling Cooperation | AT→SIT / AT→CLIMB / AT→ON_TOP, 각각 2명 한 쌍 | 행동별 Group SR·Avg.·Time (Succ./All) | 9회 |
| Table 3 — Coupled Cooperation | 공동 PUSH 2/3/4명, 각각 한 그룹 | 인원별 Group SR·Avg.·Cooperative Time Ratio | 9회 |

각 설정512환경×seeds **1/6/7**, 총42회다. 이미 일치하는 결과는 재사용하며 FSM Independent는 SCT와 공유한다. Table 2·3에는 다른 solo/그룹을 추가하지 않는다. Ablation·Post-AT·Collision·Jerk·속력 분석은 이번 세 표의 필수 평가에 포함하지 않는다.

**기존 코드/폴더 명칭은 유지한다:** `table1`은 Table 1, `table2/enable_*`는 Table 2, `table2/coupled_*`는 Table 3이다. 과거 문서의 Table 2(a)/(b)와 같은 설정이다.

## 2. 공통 조건과 채점

- 900step=30초, loco 초기화100%, 평균 행동, 정규화 통계 고정, 첫 episode만 평가. 성공해도 reset하지 않는다.
- M=2N. Distractor는 물리만 제거하고 관측·attention은 유지한다(TokenHSI 입력 예외는5절).
- Table 1의8H/16H는 interaction-group 중심을 반경 **3m/7.5m** 원 둘레의 균등 각 간격에 배치한다. 최소 그룹 간격이 아니다. 2H/4H·Table 2/3 및 그룹 내부 배치는 기존 규칙을 유지한다.
- Independent는 다섯 행동 각각20%. Mixed는 가능한 I/E/C를 균등 선택하고1/2/2–4명을 배정하는 기존 샘플러다. 매 장면 세 관계를 모두 강제하지 않는다.
- Enable 공유 상자는 공통 교집합 X/Y=0.40–0.60m, Z=0.35–0.50m. Near50 해제. 거리 구간은 공용 규약2.2절을 따른다. 각 쌍의 선행/후행 거리합을 같은 값으로 맞추지 않는다.
- Coupled 상자는 k명에 가로폭·질량 k배, 기존 거리·ordered 배정 유지. 전체 상자 크기·물성·거리·방향은 공용 규약1–3절을 적용한다.
- SIT/CLIMB/ON_TOP은 받침 XY **100%**와 해당 목표 높이±20cm. AT/PUSH는 실제 물체 중심의 목표3D 거리≤20cm. 학습 reward/gate의1mm 조건은 사용하지 않는다.
- Enable Group SR은 **동일 유효 frame에서 선행·후행 동시 성공**. `ever(AT) AND ever(follower)`로 집계하지 않는다.
- Coupled Group SR은 실제 공유 상자의 목표 도달. 개인별 PUSH 구역·전원 접촉을 성공 조건으로 사용하지 않는다. 공통 사람별 낙상 처리·이전 성공 보존 유지.
- Mixed SR은 같은 seed의 Independent/Enabling/Coupled SR을1/3씩 평균. Mixed의 solo SR을 Table 1 Independent 전용 결과로 대체하지 않는다.
- SR/Time은 seed별 계산→3seed 평균±표본표준편차(ddof=1). Table 2/3 Avg.는 각 seed에서 세 행동/인원 설정 동일 가중 평균.

**Table 2 Time:** Succ.는 성공 쌍의 최초 동시 성공 step/30 평균. All은 성공에 실제 시간, 실패에30초를 부여한 전체 평균이다. 각각 행동별 평균→세 행동 동일 가중 평균→seed 통계. 성공0건의 Succ.는 N/A, All은30초.

**Table 3 t_coop:** 초기 상자 중심 대비 수평 직선 변위20cm가 출발 기준. 원래 전원의 양손 중심과 실제 상자 전체 표면의 거리<12cm를 사용한다. 분자는 **기록 전체(출발 전·성공 후 포함)**의 공동 근접 frame 수, 분모는 공용 규약3.4절 counter 차이다. 0/NaN·clip 처리를3.5절대로 유지한다. 순수 밀기 구간 접촉 비율로 바꾸지 않는다.

t_coop은 **seed 평균±표본std**와 **pooled 평균**을 모두 저장하되 혼용하지 않는다. ± 표기에는 seed 평균/seed std를 한 쌍으로 사용하고 pooled 평균에 seed std를 붙이지 않는다. 기존6cm·구간 내 접촉률은 과거 지표다.

## 3. 현재 checkpoint와 결과 선택

| 모델 | 현재 선택 |
| --- | --- |
| TokenHSI (+ PUSH) | KD40.5k |
| SCT / SCT + FSM | Stage1 26k |
| SCT + Coop. FT | Table 1 현재 정식 결과6k / **Table 2 현재 PDF4k / Table 3 현재 PDF6k** |
| Ours | **V4 2k**, 하나의 통합 student |
| SCT + LLM Scheduler | 외부 작업 결과. 이 checkout에 원시 기록이 없어 새 집계/실행 대상에 자동 포함하지 않음 |

Coop. FT의 Table 2/3 checkpoint 차이는 사용자가 현재 PDF 기준으로 유지하라고 지정한 상태다. 새 비교군·checkpoint를 평가할 때는 별도로 명시하며 위 결과를 임의로 섞지 않는다.

```text
TokenHSI:
/home/hwanhee/ksh/tokenhsi_push/checkpoints/TokenHSIPushKD_00040500.pth
SCT / FSM:
/home/hwanhee/ksh/stage1_2_base_teacher/pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth
Coop. FT4k:
/home/hwanhee/ksh/sct_coopft/output/sct_coopft/SCTCoopFT_10-22-24-41/nn/SCTCoopFT_00004000.pth
Coop. FT6k:
/home/hwanhee/ksh/sct_coopft/output/sct_coopft/SCTCoopFT_10-22-24-41/nn/SCTCoopFT_00006000.pth
Ours V4 2k:
/home/hwanhee/ksh/stage1_2_base_teacher/output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth
```

현재 결과 확인:

- [V4/Coop6k 통합 결과](../output_etc/check/v4_2000_tables12_radius_seeds167_20261112/report.md) · [summary](../output_etc/check/v4_2000_tables12_radius_seeds167_20261112/summary.json). `seed_runs[].source`에서 모델/설정/seed별 원시 기록을 찾는다. `models.<model>.enabling_average`와 `table2` 등의 기존 key를 유지한다.
- **Table 2 Coop4k만** [4k 포함 SR·Time Succ./All](../output_etc/check/enabling_completion_time_succ_all_seeds167_20261111/report.md) · [summary](../output_etc/check/enabling_completion_time_succ_all_seeds167_20261111/summary.json)의 `coopft`를 사용한다. 이 파일의 `ours`는 과거V3이므로 현재Ours로 사용하지 않는다.
- [전체 출처·알려진 차이](../../evaluation_protocol/results.md). 과거8H/16H 최소5m 결과와 현재 원 반경 결과를 혼용하지 않는다.

## 4. 공통 초기 장면과 실행 경로

레포 기준 경로는 `/home/hwanhee/ksh/stage1_2_base_teacher`다. 같은 seed만으로 같은 장면이라고 간주하지 말고 아래 기존 snapshot을 재사용한다.

| 조건 | Reference 경로 (`output_etc/check/` 아래) |
| --- | --- |
| Table 1의2H/4H | `v3_2000_tables12_seeds167_20261110/reference/table1/{mode}_{N}h/general_seed{seed}/` |
| Table 1의8H/16H | `table1_group_radius_8r3_16r7p5_env512_seeds167_baselines_20261112/reference/{mode}_{N}h/general_seed{seed}/` |
| Table 2·3 | `v3_2000_tables12_seeds167_20261110/reference/table2/{setting}/general_seed{seed}/` |

`mode=independent/mixed`, `setting=enable_sit/enable_climb/enable_ontop/coupled_2/coupled_3/coupled_4`다. `initial.pt`·`observed_scene.pt`·`reference_info.json` 및 검증 기록을 확인한다.

| 작업 | 기존 구현 |
| --- | --- |
| Table 1 | `eval/run_stage2_scalability.py`, `stage2_scalability_eval.py` |
| Table 2·3 | `eval/table2_eval.py`, `table2_sampling.py`, `table2_metrics.py` |
| 현재 V442회 실행 구성·reference 연결 | `eval/run_v4_formal.py`, `v4_formal_eval.py` |
| Coop. FT3k/6k | `eval/run_coopft_tables.py`, `coopft_checkpoint_eval.py` |
| Time Succ./All | `eval/reaggregate_enabling_completion_time.py`, `reaggregate_v4_enabling_time.py` |
| 현재 t_coop 재계산 | `eval/coupled_cooperation_metrics.py`의 `calculate`; V4 실행기의 연결 참고 |
| 상세 옵션/이식 | [구현 안내](../../evaluation_protocol/implementations.md), [eval 안내](README.md) |

**기존 큐를 새 평가 명령으로 무조건 실행하지 않는다.** `run_table2.py`는V2 checkpoint가 기본이고, V4/Coop 큐와 재집계 스크립트는 checkpoint·출력·출처가 고정되어 있다. 새 모델은 기존 adapter를 활용하되 checkpoint/config/reference/출력을 명시한 별도 실행 구성이 필요하다. 모든 레포에 공통으로 적용되는 자동 launcher는 없다.

이번 인수인계의 새 GPU 작업은 **6/7만** 사용한다. 기존5/6/7 고정 큐는 그대로 실행하지 않는다. 점유를 확인하고 기존 프로세스를 보존한다. Conda `tokenhsi`, 준비는 `tokenhsi/scripts/multi_agent/runtime_env.sh`, GPU 선택은 `TOKENHSI_GPU`로 전달한다. 결과는 `output_etc/check/<새 평가명>/`에 저장한다.

## 5. 완료 조건

1. Manifest에 checkpoint/config/SHA, seed, 환경 수,900step, snapshot 출처/SHA, 계산 버전, 실제 GPU를 기록한다.
2. 초기 장면·실제 첫 정책 입력·8H/16H 반경·물체/관계 할당·XY100%·distractor 관측 유지/물리 제거를 검증한다.
3. 최초 성공 step·성공/전체 수·raw pose/손/생존 상태를 저장하고 CPU 재채점으로 SR/Time을 대조한다. t_coop은 분자·분모·NaN·clip 건수도 확인한다.
4. Seed별 결과와 평균±표본std, 별도 pooled 값, `report.md`·`summary.json`·`manifest.json`·검증 결과를 제공한다.

**알려진 차이:** TokenHSI는 고정397D 입력으로 다른 사람/distractor 토큰 attention을 지원하지 않는다. 기존 FSM은 첫 정책 입력의 관측 불일치가 확인된 참고 결과이며 새 공식 평가에는 수정·첫 입력 검증·재평가가 필요하다. 후처리로 해결됐다고 표시하지 않는다.

**이번 정리는 문서만 변경했다. 평가 실행·코드 수정·기존 결과 덮어쓰기는 하지 않았다.**
