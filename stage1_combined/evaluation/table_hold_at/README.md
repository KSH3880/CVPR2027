# Table carry HOLDING+AT 체크포인트 평가

공개TokenHSI Table과우리residual10k를 우리영역식·40초·동일seed연속3회로 비교하려면 [Table carry 40초 명령](../climb_paired/README.md#table-carry-40초-비교)을 사용한다. 아래기존30초·독립3seed평가와 집계를구분한다.

Chair의 [전용 평가](../hold_at/README.md)와 같은 AT 성공식/시간/집계를 사용한다. Table 전용 프로토콜 `native_table_hold_at_30s_v1`로 결과를 구분한다.

| 항목 | 조건 |
|---|---|
| 장면/물체 | 1명·동적Table1개, Table_Circle test9종 균등복원추출 |
| 초기 배치 | 학습 native reset 그대로: loco100%, 거리1~5m, 바닥+5cm, yaw0~360도 균등 |
| 목표 | XY±4.5m·바닥높이, 사람/물체와 squared XY distance≥2 |
| 표본/시간 | 512환경 × 독립seed0/1/2, 최대900step(30초), 시간종료899step |
| 성공 | 물체 중심–목표 3D거리≤20cm, AT 점/영역 동일 |
| 집계 | 첫action 이후~첫episode 종료frame 포함 ever. 초기frame/후속episode 제외 |
| 추가 조건 | 파지·손놓기·직립·속도·유지시간 조건 없음. 기존 낙상종료 유지 |
| 통계 | seed별 성공률 평균±표본표준편차(ddof=1) |
| 모델 | 저장된30D관측/모델/normalizer 고정·deterministic·학습없음 |

Chair와 방향 분포와 물체 종류가 다르다. Chair는 사람을 향하도록 정렬하지만 Table은 원본 Table reset의 무작위 yaw를 유지한다. 원본 TokenHSI 공개모델의 공식20초 평가와도 구분한다.

프로젝트 루트에서 실행:

```bash
# 활성 Table run의 가장 큰 번호 체크포인트를 선택해 정식512×3 평가
TOKENHSI_GPU=0 bash stage1_combined/tokenhsi/scripts/multi_agent/table_hold_at_full_ft_eval.sh

# 특정 체크포인트 고정
TOKENHSI_GPU=0 bash stage1_combined/tokenhsi/scripts/multi_agent/table_hold_at_full_ft_eval.sh "$CKPT"

# GPU 실행 없이 모델/설정/test9/코드 독립사본 준비
bash stage1_combined/tokenhsi/scripts/multi_agent/table_hold_at_full_ft_eval.sh --prepare-only

# 4환경·seed0·전체900step 경로검증(성능평가용 아님)
TOKENHSI_GPU=0 bash stage1_combined/tokenhsi/scripts/multi_agent/table_hold_at_full_ft_eval.sh --smoke
```

체크포인트를 생략하면 `stage1_combined/table_hold_at_active_run.json`의 experiment_dir/nn에서 `TableHoldAtFullFT_<번호>.pth`의 가장 큰 번호를 선택한다. best/final 별칭은 자동선택하지 않는다. 선택한 checkpoint를 새 runs폴더의 `checkpoint.pth`로 **byte 그대로 복사**하고 env/sim/train도 해당 checkpoint에서 복원하므로 실행 도중 더 최신 파일이 생겨도 평가 모델은 바뀌지 않는다.

출력은 상위 `runs/table_hold_at_eval_<시각>/`. 별도 `--output /절대경로/runs/새폴더` 지정 가능. `summary.json`에 seed별 성공률과 평균/표본표준편차, `REPORT.md`에 요약, `source/output_etc/check/table_hold_at_seed*/`에 initial/trace와 독립재채점·해시검증 기록이 남는다. 기존 학습은 유지하며 평가 실행은 별도 사본에서 수행한다.

이 머신 기본 Python/conda는 tokenhsi118, JIT0이다. 다른 환경은 `TOKENHSI_EVAL_PYTHON`과 `TOKENHSI_CONDA_ENV`를 함께 지정한다. 78k zero-shot 비교는 Table FT checkpoint를 `--template-checkpoint`로 제공하고 원본78k를 평가 checkpoint로 지정한다.

검증(2026-10-26): Chair 채점/첫episode/통계/재채점4함수 AST동일, CPU5개통과. 최신3000회checkpoint 고정·GPU0 4환경900step 자연종료rc0, 시간종료899step/32·64ever/trace·해시재채점 통과. `runs/table_hold_at_eval_smoke_20261026/validation.json`. 정식512×3 성능평가는 미실행이며 기존학습은 유지했다.

2026-10-26 원본78k zero-shot 정식측정: **26.50±0.56%**(134/134/139명),동일초기상태의FT3000은0%. 세seed초기전배열exact일치·trace재채점·해시검증통과. 상위 `runs/table_hold_at_78k_zeroshot_20261026/REPORT.md` 참조. FT성능저하원인은미확정.


## 공개 TokenHSI Table 모델을 공통 기준으로 평가

`--public-table-checkpoint`와 `--public-base-checkpoint`를 함께 지정한다. `--checkpoint`는 우리 공통 장면/기본 player 생성에만 사용하고, 행동은 공개 TokenHSI Table 정책이 계산한다. 원본 human223D+carry42D=265D 관측 함수, adaptation network,265D normalizer를 복원한다. 최종 모델102tensor와 RMS strict 복원 및 값 동등 검사를 수행한다. 최초8개 관측에서 CPU/GPU action을 대조하고 원시 입력/출력을 저장한다. RTX5090/PyTorch2.0에서 호환되지 않는 fused SDP 대신 math attention을 사용하며 기존 우리 정책 평가 경로에는 이 변경을 적용하지 않는다.

```bash
TOKENHSI_CONDA_ENV=tokenhsi118 PYTORCH_JIT=0 \
/home/injesus1010/anaconda3/envs/tokenhsi118/bin/python stage1_combined/evaluation/table_hold_at/run_eval.py \
  --checkpoint "$SCENE_CKPT" --gpu 0 \
  --public-table-checkpoint "$PUBLIC_TABLE_CKPT" \
  --public-base-checkpoint "$PUBLIC_STAGE1_CKPT"
```

경로는 프로젝트 루트 기준이다. 공개 Table 모델의 배포 파일명은 `ckpt_stage2_objShape_table.pth`이며 현재 캐시 파일 내부 epoch는40000이다. 이 결과는 논문 Table3의83.6% 재현이 아니라 공통 native_table_hold_at_30s_v1 측정이다. test9/512×독립seed0·1·2/30초/AT20cm 첫episode ever, 추가 upright 조건 없음. 초기 자세/물체/목표는 같은 seed의 우리 평가와 비교한다. `public_policy_provenance.json`,seed별manifest.public_policy 및 `public_policy_cpu_parity.{json,pth}`에 실제 사용 모델/해시/전이/CPU대조 증거를 남긴다. 상위 manifest의checkpoint는 장면 생성용 파일이며 public_policy가 행동 정책을 식별한다.

2026-10-27 공통 기준 공개 Table pth 측정:476/454/475명,**91.47±2.43%**(평가seed표본std). 원본265D/102tensor/RMS복원,math attention CPU/GPU대조,모든seed초기전배열residual2200과exact/trace·precision·해시검증통과. 결과·최초fusedSDP실패기록은 `runs/tokenhsi_table_common_eval_20261027/`에 보존한다.
