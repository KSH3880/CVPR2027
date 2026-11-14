# 8H·16H 그룹 중심 고정 반경 평가 인수인계

작업 저장소: `/home/hwanhee/ksh/stage1_2_base_teacher`  
기록일: 2026-11-12

## 적용할 배치

| 인원 | 그룹 중심 원 반경 R | 적용 장면 |
|---|---:|---|
| 8H | 3m | Independent / Mixed 모두 |
| 16H | 7.5m | Independent / Mixed 모두 |

- 장면의 그룹 중심들을 **고정 반경 원 둘레에 균등한 각 간격**으로 배치한다. 원의 중심은 해당 환경의 원점이며 기존 그룹 순서·회전 샘플링을 유지한다.
- 그룹은 Independent 1명, Enabling 선행·후행 2명, Coupled PUSH 참여자 2~4명 단위다. 진단에서도 Enabling 두 사람을 한 그룹으로 합친다.
- 중심은 그룹 구성원 초기 root XY의 평균이다. 그룹을 평행 이동하며 내부 상대 배치·자세와 기존 Enabling 후행 재배치/중심 보정 절차를 유지한다.
- 기존 사람 기본 스폰 반경 `agentSpawnRadius=2.5`와 별개다.
- **3m/7.5m는 그룹 간 최소 간격이 아니다.** 그룹 수 G에 따른 인접 중심 간격은 `d = 2R sin(π/G)`다. Independent에서 G=N이므로 8H는 약2.30m,16H는 약2.93m다. Mixed는 그룹 수에 따라 달라진다.

이전의 그룹 간격5m,8H 간격3m,16H 간격4m 결과는 별도 조건이다. 이 문서의 설정과 혼용하지 않는다.

## 유지하는 평가 조건

- Independent 전용 장면과 Mixed 장면을 각각 평가한다. 모든 모델에 같은 초기 장면·과제 할당·성공 판정을 적용한다.
- Independent: SIT/CLIMB/AT/ON_TOP/PUSH 각각20%.
- Mixed: 미할당 사람에서 가능한 I/E/C를 균등 선택하고 할당을 반복한다. I는1명,E는2명,C는 가능한2~4명을 균등 선택한다. E 후행 SIT/CLIMB/ON_TOP은 각각1/3.
- 900스텝, Region XY100%, 물체 토큰 M=2N. 그룹 k명당2k 토큰 예산을 유지한다.
- Distractor는 관측에 유지하고 물리에서는 제외한다. 기존 배치 유효성 검사는 유지한다.
- 상자 크기·과제 거리·Coupled ordered 배정·사람별 낙상 처리·성공 보존 등 기존 규칙은 유지한다. 세부 공통 기준은 [공용 프로토콜](/home/hwanhee/ksh/evaluation_protocol/README.md)을 따른다.
- Enabling 그룹 성공: 같은 유효 프레임에서 선행 AT와 후행 행동이 동시에 성공한 적이 있어야 한다.
- Coupled 그룹 성공: 실제 공유 상자 중심의 목표 도달 기준을 사용한다.
- Table 1의 Independent 열은 개인 성공률, Mixed 열은 Independent/Enabling/Coupled 유형별 그룹 성공률의 동일 가중 평균이다.

## 구현 진입점

아래 파일은 모두 위 작업 저장소 기준이다.

| 파일 | 역할 |
|---|---|
| `eval/group_spacing_eval.py` | `GROUP_SPAWN_RADIUS`를 읽어 그룹 중심 고정 반경 배치를 적용하는 GPU 평가 진입점 |
| `tokenhsi/env/tasks/multi_agent/stage2_group_layout.py`, `humanoid_ma_stage2.py` | 일반 정식 평가의8H/16H 기본 반경·그룹 평행 이동·실제 반경 검증 |
| `eval/run_group_radius_eval.py` | 아래 완료된64환경·seed1·세 모델12회 전용 실행기 |
| `eval/run_group_radius_formal.py` | 512환경·seed1/6/7·TokenHSI/SCT/FSM/Coop. FT6k42회·Ours 제외·GPU5/6/7 전용 실행기 |
| `eval/run_group_spacing.py` | 모델별 checkpoint·공통 초기 장면 생성·실행·CPU 검증 공용 코드 |
| `eval/run_group_spacing_ready.py` | GPU6/7 메모리 예약·준비된 작업 병렬 배정 |
| `eval/run_group_radius_screen.py`, `eval/group_radius_probe.py` | 정책 행동 없이 배치 생성 가능 여부만 검사 |

GPU 자식의 배치 옵션은 `GROUP_SPAWN_RADIUS=3` 또는 `7.5`, `SPACING_METERS=native`다. 전용 실행기가 인원별로 전달한다. 일반 train/test/VNC에 환경변수만 붙여도 자동 적용되는 전역 기능은 아니다.

**현재 기본 동작:** 이 checkout의8H/16H 정식 평가(`--eval_protocol`·`--group_spawn_over4`)와 일반 scalability 진입점은 환경변수 없이도3m/7.5m를 적용한다. 그룹 배치 재시도는 기존 검증 경로와 같은1024회이며 수용 조건은 유지한다. 결과에는 `group_spawn_radius=3.0/7.5`, `group_spawn_gap=null`을 기록한다. 기존 간격 비교용 실행기는 명시한 간격/반경으로 덮어쓴다. 일반 viewer와 다른 인원수의 배치는 유지한다.

`run_group_radius_eval.py`는 **64환경·seed1·모델3개·출력 폴더가 코드에 고정**돼 있다. CLI로 환경 수/seed를 바꾸는 실행기가 아니다. 다른 실험은 이 경로를 재사용하되 새 출력 폴더와 요청받은 설정을 사용한다. 기존 완료 결과를 덮어쓰거나 같은 큐를 중복 실행하지 않는다. 512환경으로 확장하면 동시 실행 수/시작 메모리 예약량도 다시 확인한다.

기존 실행 명령(완료된 구성 확인용):

```bash
cd /home/hwanhee/ksh/stage1_2_base_teacher
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/run_group_radius_eval.py
```

CPU supervisor는 GPU를 직접 사용하지 않고 자식만 GPU6/7에 배정한다. 실행 전 `nvidia-smi`를 확인하고 기존 학습·다른 프로세스는 보존한다. 자식의 실행 환경은 `tokenhsi/scripts/multi_agent/runtime_env.sh`를 사용한다.

## 완료된 정책 평가

64환경 × seed1 한 번,8/16H × Independent/Mixed ×3모델 =12회. 공통 초기 장면4개. GPU6/7에서 약9.3분에 전부 완료했고 실패0건이다.

| 모델 | 8H Independent | 8H Mixed | 16H Independent | 16H Mixed |
|---|---:|---:|---:|---:|
| TokenHSI (+ PUSH),40.5k | 80.86 | 49.67 | 83.11 | 46.64 |
| SCT,26k | 87.89 | 58.51 | 81.45 | 45.23 |
| Ours V3,2k | 98.05 | 82.80 | 91.60 | 72.38 |

단위%. 단일 seed이므로 표준편차는 없다. TokenHSI의 고정397D 자기/담당 과제 입력 차이는 유지한다. 반경/간격 조건을 바꾸면 유효성 재샘플링으로 초기 장면도 달라질 수 있으므로 결과 차이 전체를 이격 효과로 해석하지 않는다.

평가 checkpoint:

```text
SCT:
/home/hwanhee/ksh/stage1_2_base_teacher/pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth

TokenHSI:
/home/hwanhee/ksh/tokenhsi_push/checkpoints/TokenHSIPushKD_00040500.pth

Ours:
/home/hwanhee/ksh/stage1_2_base_teacher/output/stage2_distill_900_v3/Stage2Distill900V3_09-20-41-11/nn/Stage2Distill900V3_00002000.pth
```

결과 폴더:
`/home/hwanhee/ksh/stage1_2_base_teacher/output_etc/check/table1_group_radius_8r3_16r7p5_env64_seed1_20261112/`

- `report.md`: 표와 seed별 네 가지 성공률.
- `summary.json`: 원시 수치·설정별 집계.
- `manifest.json`: 실행 설정·checkpoint·출처.
- `completion_verification.json`:12회/기준4개 완료, 실제 반경·공통 초기 상태·CPU SR/Error·checkpoint SHA 검증.
- 각 실행의 `group_spacing.json`: `group_placement=fixed_radius_group_center_ring`, `group_spawn_radius=3.0/7.5`, `group_spawn_gap=null`과 실제 최소 중심 간격.

## 반경 선택 근거와 적용 범위

512환경·seed1/6/7·Independent/Mixed의 배치 생성만 검사했을 때, 준비 포함240초·기존1024회 시도 한도 안에서 모두 통과한 가장 작은0.5m 후보가8H3m/16H7.5m였다. 수학적 최소 반경이라는 의미는 아니다.

배치 검사 기록: `/home/hwanhee/ksh/stage1_2_base_teacher/output_etc/check/group_radius_feasibility_20261111/report.md`.

**512환경·seed1/6/7의 정책 SR 평가도2026-11-12에 완료했다.** TokenHSI/SCT/FSM/Coop. FT6k42회·공통 장면12개, 실패0이며 Ours는 제외했다. 실행기는 `eval/run_group_radius_formal.py`, 결과 폴더는 `output_etc/check/table1_group_radius_8r3_16r7p5_env512_seeds167_baselines_20261112/`다. [정식 결과](../output_etc/check/table1_group_radius_8r3_16r7p5_env512_seeds167_baselines_20261112/report.md)와 해당 `completion_verification.json`·`final_audit.json`을 참고한다. 위 표는 별도의64환경·seed1 결과다. 반경은 공용 프로토콜v1.1.0과 이 checkout의8H/16H 정식 평가 기본값에 반영돼 있으며2H/4H·일반 viewer 배치는 유지한다.

## 정식 채택 ·2026-11-12

사용자 확정으로8H R=3m/16H R=7.5m를 Table 1의 Independent/Mixed 공통 정식 배치로 채택했다. 공용 프로토콜v1.1.0에 반영했으며2H/4H·Table 2 단일 그룹은 기존 배치를 유지한다. Ours는 V4 2k다. `eval/run_v4_formal.py`가512환경·seed1/6/7의 Table1/2 신규42회를 GPU5/6/7에서 실행한다. 결과는 `output_etc/check/v4_2000_tables12_radius_seeds167_20261112/`에 저장하며 완료 여부는 `status.json`·`completion_verification.json`으로 확인한다. 위64환경 V3 표와 이전5m 결과는 과거 결과로 보존한다.

V4 2k의 정식42회는 모두 완료했다. `output_etc/check/v4_2000_tables12_radius_seeds167_20261112/report.md`에 최신 다섯 모델 Table1/2, `completion_verification.json`에 신규42회 검증, `comparison_verification.json`에 재사용168회 공통 초기 상태 검증을 저장했다.
