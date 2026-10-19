# 원본 크기 물체 zero-shot

사용자가 지정한 TokenHSI `tokenhsi/data`와 로컬 `TeamHOI/assets`를 사용한다. ksh·공유 데이터·TeamHOI 원본은 수정하지 않는다. 기존 `round_table`은 박스 크기로 변형한 이전 probe로 보존한다.

| 이름 | 실제 크기 X/Y/Z(m) | 질량(kg) |
|---|---|---:|
| tokenhsi_table_native | 0.65/0.65/0.669157 | 7 |
| tokenhsi_armchair_native | 0.60/0.6938324/0.9196421 | 7 |
| teamhoi_default_round | 2.0/2.0/0.86 | 78.31662 |
| teamhoi_default_square | 1.6/1.6/0.86 | 64.3584 |
| teamhoi_default_rectangle | 2.0/1.2/0.86 | 60.5184 |
| teamhoi_large_round | 2.4/2.4/0.86 | 111.4914 |
| teamhoi_large_square | 2.2/2.2/0.86 | 119.0784 |
| teamhoi_large_rectangle | 3.0/1.4/0.86 | 103.7184 |

TokenHSI test에는 원형 테이블9개·안락의자27개가 있으며 대표2개를 준비했다. 전수 평가는 아니다. 전체 원본 경로·SHA256·bbox·질량은 `runs/jointcarry_zeroshot/native_assets/catalog.json`에 기록한다.

## 적용 방식

- 크기와 질량을 유지한다. bbox 중심을 actor 원점으로 삼는 평행이동만 하며 mesh/texture는 로컬에 복사한다.
- TeamHOI 고정5link는 충돌·visual 배치를 유지한1강체로 통합한다. 질량과 COM, 원래 관성에 평행축 정리를 적용한다. TokenHSI 원본에서 관성이 없는 경우만 bbox 근사를 사용하고 manifest에 명시한다.
- O0만 교체한다. 실제 형상의 bbox를 `_box_size`와8개 꼭짓점 입력에 반영한다. 기존 물체 입력은 속도6+bbox24=30차원, pose는 별도다. mesh의 빈 공간과 질량은 새로운 관측으로 추가하지 않는다.
- epoch21500 고정, 추가 학습 없음. 크기·질량·형상이 동시에 달라지므로 형상 하나의 효과를 분리한 실험은 아니다.
- `joint_carry_at`의 접근(loco) 시작만 지원한다. 박스용 paired pickup/carryWith/putDown RSI는 거부한다. 사람 초기Y거리는 `max(1.0, bbox_y/2+0.6)+U(0,1)` m로 물체 내부 시작을 피한다. 기존 locoPlatform50%는 유지한다. AT에서는 쌓기가 없으므로 불필요한 stack 높이 상한 검사를 분리했다. ON_TOP은 미지원이다.

## GPU 실행

사용자 요청에 따라 **물리·tensor pipeline·정책 모두 GPU6**인 `native_gpu_test.sh`를 사용한다.
기존 학습 MPS와 공유하지 않도록 `CUDA_MPS_PIPE_DIRECTORY=''`를 첫 CUDA 초기화 전에 지정한다. 이는 [NVIDIA의 MPS 우회 방법](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)이다. 물리 시작 전에 physicalGPU6 UUID와 nvidia-smi의 본인 process type `C`를 확인하며, MPS client이면 실행을 거부한다. `gpu_context.json`에 증거를 저장한다. 기존 daemon/GPU compute mode는 바꾸지 않는다. 디바이스 오류가 다른 작업에 절대 영향을 주지 않는다는 보장은 아니다.

CVPR2027 루트에서:

```bash
# GPU6 PhysX + GPU pipeline + GPU policy
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUM_ENVS=8 EVAL_REPEATS=1 bash stage2_jointcarry/zeroshot/native_gpu_test.sh tokenhsi_table_native
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUM_ENVS=8 EVAL_REPEATS=1 bash stage2_jointcarry/zeroshot/native_gpu_test.sh teamhoi_large_rectangle
# CPU 사전 확인 경로(최종 GPU 평가와 구분)
EPISODE_LENGTH=32 EVAL_REPEATS=1 bash stage2_jointcarry/zeroshot/native_cpu_test.sh tokenhsi_armchair_native
```

`native_test.sh` 기본 실행도 GPU 물리 경로로 연결된다. 혼합 배치는 `native_batch_test.sh native_source_batch`로 실행하며 환경별 asset 매핑을 `native_env_assets.json`에 남긴다. 사람 스폰 여유는 배치 내 최대 bbox Y를 기준으로 통일한다.

GPU viewer는 `PORT=6086 bash stage2_jointcarry/zeroshot/native_vnc.sh teamhoi_large_rectangle`로 실행한다. 2026-10-15 실제 대형 직사각형 화면·GPU물리 rollout·HTTP200을 확인했다. 로그/화면은 `runs/jointcarry_zeroshot/native_view_6086_teamhoi_large_rectangle/`. 실행 상태는 현재 프로세스와 포트로 확인한다.

## 검증 및 기록

- `native_assets/cpu_physics_check.json`: 8종180step CPU 로드·낙하,1body/0DOF, 실측 질량 일치.
- `native_cpu_smoke`, `native_cpu_smoke_v2`: 대표3종32프레임 CPU checkpoint 추론rc0. 최초 의자 검사는 AT에 stack 높이 제한이 적용돼 거부됐으며 실패 로그를 보존했다.
- `native_probe_20261015`: 대표3종 각8장면600프레임, CPU물리/GPU정책 사전 탐색으로 모두rc0. **전체GPU 평가가 아니다.**
- `native_gpu_smoke`, `native_gpu_probe_20261015`: 독립 GPU6 CUDA 컨텍스트 확인과 GPU PhysX/pipeline/정책 평가 결과를 보존한다.

최종 GPU6 비교는 [RESULTS.md](../../runs/jointcarry_zeroshot/native_gpu_probe_20261015/RESULTS.md)에 기록했다. 24환경600프레임 rc0, 물체별 목표 완료0/8이며 높이/이동 대리 지표는 운반 성공률로 해석하지 않는다.
