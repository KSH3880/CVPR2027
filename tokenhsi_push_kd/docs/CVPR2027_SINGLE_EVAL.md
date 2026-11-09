# CVPR2027 단독 스킬 평가

2026-11-09 갱신: `cvpr2027_evaluation_protocol_1109.md` + 사용자 지정 native SIT/CLIMB 예외.

2026-11-08 사용자 지정 범위: **우선 단독 스킬만**. SIT, CLIMB, AT, ON_TOP, PUSH를 각각 한 사람으로 평가한다. AT/ON_TOP/PUSH는 물체 슬롯 두 개를 사용한다. 이 결과는 Table 1의 다인원 Independent 또는 Mixed 결과가 아니다.

## 기존 코드와 차이

| 항목 | 기존 평가 | 새 단독 평가 |
|---|---|---|
| 규모 | PUSH seed 0 × 512, native seed 0 × 512 × 3회 | 스킬별 512장면 × 사용자가 명시한 서로 다른 seed 3개 |
| 시간 | PUSH 1200, SIT/CLIMB 300, CARRY 600 | 모두 900 control step, dt=1/30 |
| 워밍업 | native 700 step | 없음; 최초 loco episode만 |
| 환경 | native 의자/지형/운반 데이터 + reference PUSH | SIT/CLIMB: 원본 native; 나머지: reference 물리 상자 |
| 거리 | PUSH 2–10m, 기존 native 설정 | PUSH 접근/이동 각각 2–5m; AT/ON_TOP 1–5m; SIT/CLIMB native 배치 |
| 성공 | native reward success, PUSH float32 거리 | AT/ON_TOP/PUSH: XY100 Region ever·Point; SIT/CLIMB: native Point ever |
| 목표 | native 목표 | SIT/CLIMB native 목표; ON_TOP 상판+payload 높이/2 |
| 종료 | 종료 환경 reset 및 반복 | 성공 reset 없음; 낙상 frame 포함 후 제어 해제; 이후 성공 무효 |
| 통계 | 반복/개인 평균 | seed별 SR 후 평균±표본표준편차(ddof=1) |
| 검증 | 최종 수치/행 저장 | initial, 900 frame 물리 pose·관절·접촉·생존·낙상·성공, 소스 hash, CPU 재채점 |

기존 `push_kd_player.py` 경로는 과거 수치 재현용으로 유지한다. `scripts/eval.sh`의 기본값은 새 프로토콜이며 `EVAL_PROTOCOL=legacy`로 기존 경로를 명시한다. 학습 경로는 바꾸지 않는다.


## 1109 성공·Error 및 native 예외

- 기본 revision은 `1109`. ON_TOP은 yaw XY100%를 사용한다. SIT/CLIMB는 사용자 요청으로 원본 TokenHSI native 환경·mesh·크기·배치·목표를 유지한다. 원본 SIT은 StraightChair_Normal, CLIMB는 Box/Cabinet/Table_Square 혼합이다.
- 원본 native 성공은 root와 native 목표의 3D 거리≤20cm다. 의자 등받이 최고점을 좌판으로 취급하거나 임의의 좌판 Region을 만들지 않는다. 해당 결과는 `scoring_mode=native_point`, `region_definition=native_point_alias`로 표시하며, 공통 출력의 Region SR/Error는 Point와 같은 값이다. **이 두 과제는 논문 XY100 box 결과가 아니며 다른 방법도 같은 native 조건에서 비교해야 한다.**
- 공통 실행 조건은 900 step·loco100%·평균 행동·첫 에피소드·낙상 frame 포함·정규화 고정이다. 과거 native 300step/700warmup 결과와는 구분한다.
- AT/ON_TOP/PUSH는 문서 1.8의 Point/Region Error를 함께 출력한다. 둘 다 Region 성공 trial만 포함하고 각 trial의 모든 유효 frame에서 최소 거리를 구해 cm로 변환한다. Region은 목표 평면까지 거리이며 ±20cm 허용치를 빼지 않는다.
- `result.json`의 `errors.{point,region}.{mean_cm,samples}`와 원시 `distances_m.npy`, `minimum_distances_m.npy`를 저장한다. seed별 평균을 같은 가중치로 집계하며 성공 표본이 없는 seed가 하나라도 있으면 전체 평균·표준편차는 null이다. 행동 간 전체 macro는 만들지 않는다.
- 정식 실행은 `--seeds S1 S2 S3` 또는 `EVAL_SEEDS="S1 S2 S3"`를 요구한다. 아래 0/1/2는 실행 예시이며 확정 seed가 아니다. `--smoke`는 seed0·2장면의 구현 검사만 한다.
- `--protocol 1108` / `EVAL_REVISION=1108`은 이전 box/XY80 경로를 명시적으로 선택한다. 학습과 legacy eval 기본 성공 판정은 바꾸지 않았다.
- 기존 원시 trajectory는 GPU 재실행 없이 재채점 가능하다. 새 출력 파일로만 저장하며 asset/배치를 바꾸지는 않는다. 예: `python scripts/rescore_cvpr2027.py --input <task/seed_dir> --output <new_report.json>`.

## 실행

```bash
EVAL_SEEDS="0 1 2" TOKENHSI_GPU=0 CHECKPOINT=/absolute/student.pth bash scripts/eval.sh
EVAL_SEEDS="0 1 2" TOKENHSI_GPU=0 CHECKPOINT=/absolute/student.pth EVAL_TASK=push bash scripts/eval.sh
# AT와 ON_TOP 분리. 새 평가에서 carry라는 하나의 task로 합치지 않는다.
/home/injesus1010/anaconda3/envs/tokenhsi118/bin/python scripts/eval_cvpr2027.py \
  --checkpoint /absolute/student.pth --tasks at on_top --seeds 0 1 2 --output output_etc/eval/my_run
# CPU 2장면 smoke: 공식 결과 아님, 현재 GPU 학습과 분리하여 구현 검증
/home/injesus1010/anaconda3/envs/tokenhsi118/bin/python scripts/eval_cvpr2027.py \
  --checkpoint /absolute/student.pth --smoke --cpu
```

기본 결과는 `output_etc/eval/cvpr2027_<timestamp>/<task>/seed_<seed>/`에 저장한다. 각 seed의 `result.json`, `initial.npz`, pose·관절·접촉 등의 `.npy`, `cpu_rescore.json`과 전체 `summary.json`을 보존한다. CPU 재채점은 저장 궤적에서 float64 공간 판정과 낙상 유효 구간을 다시 계산해 online 결과와 비교한다. 원시 float32 물리 상태를 float64로 변환하여 채점하며 시뮬레이터 자체의 정밀도를 바꾸지는 않는다.

`--snapshot-root <previous_run>`은 같은 task/seed의 initial 상태를 재사용한다. asset 크기·밀도·환경 원점이 다르면 거절한다. 같은 seed만으로 서로 다른 코드베이스의 장면 동일성을 주장하지 않는다. 다른 방법의 adapter에도 같은 snapshot과 scoring 규칙을 적용해야 한다. Native SIT/CLIMB에서는 전체 actor state·asset 경로까지 보관하고 asset 일치를 검증한다.

## 물체와 표본

AT/ON_TOP/PUSH reference 경로에 적용: PUSH 167 profile 균등, 밀도3·마찰.4. 일반 상자는 지정된 축별 .05m 격자, 밀도100. 과제를 고정한 후 해당 크기를 생성한다. 상자 초기 yaw는 랜덤이며 지면에 직립한다. 접근 반지름/방향을 균등 추출하고 PUSH 목표 방향은 상자 local +Y ±45도다. 원형 XY bounding radius로 겹침·경로 간섭을 보수적으로 거절하며, 최대4096회 재시도 초과는 오류로 처리한다. 정책 실행 결과로 표본을 제외하지 않는다.

ON_TOP의 두 번째 슬롯은 실제 받침이다. 그 외 과제의 두 번째 슬롯은 distractor이며 초기 배치 검사 후 물리 공간 밖으로 이동한다. sampled pose·회전·속도 및 형상 크기는 initial archive에 유지한다.

## 적용 범위와 제한

- 단독 스킬 각각을 측정한다. 5개 행동 20%의 단일 Independent 혼합장면 SR은 생성하지 않는다. 다인원 Independent / Enable / Coupled / Mixed, 그룹 배치·alias·공동 상자 확장은 이번 범위 밖이다.
- TokenHSI는 고정된 397차원 self/task 관측을 쓰며, 전체 M개 object-token attention 자체가 없다. 작업 물체와 ON_TOP 받침은 기존 projection을 사용하고 distractor 물리 충돌은 제거한다. distractor 토큰을 소비하는 모델과 **관측 구조가 같다고 주장하지 않는다**. 명시적 전체 물체 토큰 입력은 checkpoint architecture 변경 없이 구현할 수 없다.
- 문서 5절의 미정 사항은 임의로 확정하지 않는다. 이번 범위의 N=1 distractor는 반경4.5m 원의 면적 균등 배치를 사용하고 결과 구현 버전에 기록한다(문서는 4명 배치 영역만 지정). 이는 공통 snapshot 생성 측과 맞춰야 하는 운영 선택이다.
- 낙상 판정은 각 환경의 humanoid 접촉/높이/grace 구현을 사용하며 threshold와 관련 소스 hash를 저장한다(SIT/CLIMB native, 나머지 reference). 낙상 frame까지 채점하고 motors를 해제한 뒤 수동 actor로 남긴다. 1인 장면이므로 해당 사람 낙상 이후 그 장면에서 새 성공을 기록하지 않는다.
- 이 구현의 첫 CPU smoke 결과는 성능 비교용이 아니다. 공식 결과는 동일 장면·같은 GPU/physics 조건에서 512×3 seeds로 재평가해야 한다. 기존에 보고한 50.78% 등의 성공률은 구 프로토콜이며 새 결과로 사용할 수 없다.

## 구현 검증 (2026-11-09)

- scoring/reference scene/checkpoint contract 테스트 23개 통과; Python 및 shell 문법 검사 통과.
- `output_etc/check/cvpr2027_1109_native_smoke`: native SIT/CLIMB 각각 CPU 2장면·900 step 완료.
- `output_etc/check/cvpr2027_1109_smoke`: AT/ON_TOP/PUSH 각각 CPU 2장면·900 step 완료. 이 폴더의 box SIT/CLIMB는 native 변경 전 검사이므로 최종 설정 결과로 사용하지 않는다.
- 위 실행 모두 정규화 불변 및 저장 궤적 CPU 재채점 일치. `cvpr2027_1109_native_replay`에서 SIT 초기 snapshot 전체 필드 일치와 복원 후 900 step 재채점 확인.
- 512×3 정식 성능 평가는 실행하지 않았다. GPU 학습 프로세스를 조작하지 않았다.

## 이전 구현 검증 (2026-11-08)

- `tests/test_cvpr2027_scoring.py`, `tests/test_reference_scene.py`, `tests/test_contract.py`: 18개 통과.
- `output_etc/check/cvpr2027_final_smoke`: 스킬 5종 각각 CPU 2장면, 최대900 step 실행 완료. 모든 실행에서 normalization 불변 및 저장 궤적 CPU 재채점 일치 확인.
- `output_etc/check/cvpr2027_replay_smoke`: PUSH 초기 snapshot 복원 실행 완료. initial의 모든 필드가 원본과 동일하고 CPU 재채점 통과.
- 512장면×3 seeds 전체 성능 평가는 아직 실행하지 않았다. 진행 중 GPU 학습은 유지했다.
