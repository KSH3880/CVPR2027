# Stage1 jointpush

현재 checkout의 `stage1_v2_fixed` 네 과제에 KOO PUSH를 연결한 스크래치 실험이다. 이름에 jointpush가 들어가지만 **아직 Stage2 coupled 학습은 아니다.**

## 설정

| 항목 | 값 |
| --- | --- |
| 규모 | 2명·4상자·2048환경·1200step |
| SIT/CLIMB/AT/ONTOP/PUSH | 10/20/25/20/25% 크기 생성 prior 및 크기 조건부 과제 sampler |
| 사람 기준 위치·방향 | 기존 Stage1의 참조 root + 반경2.5m 사람별 offset, 참조 yaw 유지 |
| PUSH loco 사람→상자 | 실제 사람 XY 기준 반경 U(2,10)m·360° |
| PUSH 상자→목표 | loco/접촉 RSI 모두 반경 U(2,10)m·360°, 사각형/정면 cone 제한 없음 |
| PUSH 상자/목표 높이 | 바닥 Z/2, 목표는 reset 후 고정 |
| 크기 | X/Y1.6~2.2m(0.1m 간격), Z1.45~1.7m(0.05m), min(X,Y)≥1.1Z인167종 |
| 물성 | 20kg·마찰0.4·별도 밑판 없음 |
| PUSH RSI | loco50%/접촉50%, KOO949후보·후반30% 요청70%, clip7 제외, 진행률에 따른 상자 이동 없음 |
| PUSH AMP | push_loco50%/push50%, 기존 과제는 기존 family 유지 |
| 기존 상자/받침/distractor | Stage1 크기/물성, distractor는 반경4.5m 원판·무작위 yaw |
| 목표 최소거리 | PUSH 목표와 담당 사람 중심1m 이상 |
| 경로 검사 | 제안 후 거부 샘플링. PUSH끼리3.5m 분리, 뒤쪽1.5m 포함, 다른 사람·과제 상자·SUPPORT와 여유 검사 |
| 학습 | teacher 없음, 학습률2e-5, 입력/AMP 정규화 통계 갱신, 본인 보상100% |
| 평가 | 모든 과제 loco100%, 과제 비율·물성·loco 배치/목표 sampler는 학습과 동일. 기존 PTH는 가중치/정규화만 로드 |

크기는 환경 생성 시 고정된다. PUSH 크기로 생성한 담당 슬롯은 PUSH를 유지하고, 다른 슬롯은 그 크기로 가능한 기존 과제를 reset마다 고른다. SUPPORT는 ONTOP이면 과제 받침이고, 그 외에는 물리 장면에 남는 distractor다. 숨겨진 물리 후보 bank는 없다. 간섭 없는 표본만 채택하므로 최종 거리/방향 빈도는 균등 제안 분포와 다를 수 있다.

## 보상

KOO의 CONTACT+AT를 사용한다. 각 관계의 state/progress/success 가중치는0.2다. CONTACT는 목표 방향으로 밀 수 있는 유한한 면과 양손의 XY 거리 중 최악값을 사용한다. 상태 `exp(-10d²)>0.9`와 양손 반력/상자 힘0.5N이 접촉 성공 조건이다. 접근 점수는 사람→상자 중심 거리의 `1/(1+max(d-0.5,0))`다.

AT는 중심3D 거리의 `exp(-10d²)`, XY 접근 `1/(1+max(dxy-0.1,0))`, 성공은 중심 거리20cm 및 Z오차1mm를 사용한다. AT에는 CONTACT 선행 gate가 없다. 관계 성공 시 해당 세 항을 포화하고, AT 성공 시 CONTACT 보상도 포화한다. 두 관계의 과제 보상 최대는1.2이며 AMP/페널티 합산은 기존 학습기 규칙이다.

## 출처와 연결

- 기반: 로컬 `stage1_v2_fixed.yaml` 및 `HumanoidMACarry`.
- PUSH 기하/RSI/보상: [KOO 포트](https://github.com/KSH3880/CVPR2027/tree/koo_push_1026/ports/stage1_base_push/tokenhsi), 원본 [가이드](https://github.com/KSH3880/CVPR2027/blob/koo_push_1026/PUSH_RSI_UNIFORM_GUIDE.md).
- KOO의 고정 사람 anchor·yaw 추가 회전·square/cone 목표·넓은 distractor 영역은 가져오지 않았다. 새 배치는 `jointpush_scene.py`에 분리한다.
- 원본 RSI bank JSON과 후보를 보존하고 실제 모션/asset/mesh 및20kg·0.4·167종 물성을 검사한다. 새 거리/방향 배치는 원본 bank의 scene 계약과 구분하며 최종 reset 간섭 검사를 적용한다.
- 학습/평가 명령: [config](../config.md). 체크포인트·환경 옵션: [USAGE](../USAGE.md#stage1-jointpush-체크포인트와-환경).
- 검증 기록: [changelog](../../changelog.md), `output_etc/check/stage1_jointpush/`.


## 밀도3·목표±45° variant

`stage1_jointpush_density3_cone90`는 위 설정에서 PUSH 상자를 밀도3kg/m³로 만들고, 목표 방향을 상자 로컬+Y 기준±45°(총90°)로 제한한다. 질량은 부피×3이며167종 균등 평균18.0014kg, 범위11.136~24.684kg이다. PhysX box asset 생성 시 density를 전달하여 해당 크기의 질량·관성을 계산한다. 다른 과제와 받침은 밀도100을 유지한다.

상자/목표를 기존 방식으로 제안한 뒤 cone 밖이면 그 장면의 PUSH 상자 위치·yaw·목표를 함께 재샘플링한다. 혼합 장면의 비PUSH 상자는 유지하며, 경로·간섭 검사도 유지한다. 최대 시도는2048회다. 거리2~10m·1200step·과제 혼합·RSI 비율·보상·학습 설정은 기반과 같으며, 원본 checkpoint의 square 목표/60초 환경 전체를 복제한 설정은 아니다.

RSI의 `reference_mass: 20.0`은 기존 bank가20kg에서 선별되었다는 출처다. 기존 후보·가중치·모션/asset/mesh 해시·마찰·크기 검사를 유지하며 원본 bank를 변경하지 않는다. 밀도3에서의 새 물리 screening을 뜻하지 않으며, GPU32환경에서 실제 질량/관성, loco/RSI 진단 reset·각32step 추론은 확인했다. 전크기·전후보 RSI screening, 장기 안정성·성공률은 미검증이다.

전용 환경/학습 YAML·train/test/VNC·출력 경로를 사용한다. 학습은 기반과 같이 스크래치이며, test/VNC 기본 입력은14000 checkpoint다. 실행 명령은 [config](../config.md), CPU 검증 결과는 `output_etc/check/stage1_jointpush_density3_cone90/`에 둔다.


## Fixed78000 파인튜닝

`stage1_jointpush_density3_cone90_finetune`는 밀도3·목표±45° variant의 물성·장면·RSI·보상·PPO 설정을 유지하고, 78k 전이와 과제 비율만 바꾼다. PUSH50%, SIT5%, CLIMB12.5%, AT16.25%, ONTOP16.25%로 기존78k의 네 과제 비율을 절반으로 줄였다. 이는 환경 생성의 과제/크기 prior이며 PUSH 슬롯은 고정되고 나머지는 크기 조건부로 과제를 재선택하므로 유한 표본의 실제 비율은 약간 다르다.

`experiment.transfer.checkpoint`는 `pth/Stage1V2Fixed_00078000.pth`다. 로컬 원본은 이웃 `TokenHSI_stage1_push/output/stage1/Stage1V2Fixed_00078000.pth`이며 같은 내용으로 이 checkout에 복사했다(SHA256 `f5b1637716611c673dd33f3dd894bf081a7dd88b7025943505d8287eba5ba13f`). 학습 wrapper의 `STAGE1_CHECKPOINT`로 경로를 바꿀 수 있다.

기존 `stage1_push_network.transfer()`로 actor·critic·AMP 가중치를 가져온다. 네 과제 embedding은 보존하고 새 PUSH embedding은 carry AT에서 복사한다. 판별기의 기존132개/frame 입력 연결을 복사하며 추가 PUSH label 연결만0으로 초기화한다. 관측·가치·AMP 정규화 통계도 전이한다. optimizer/epoch은 새로 시작하며, 모든 기존 학습 대상과 정규화 업데이트·학습률2e-5를 유지한다. 별도 teacher KL이나 encoder 동결은 추가하지 않는다.

`RESUME_CHECKPOINT`를 지정하면78k 초기화를 건너뛰고 동일 환경 계약의 파인튜닝 checkpoint를 복원한다. 원래20kg/360° 또는 밀도3/90° 스크래치 결과는 덮어쓰지 않는다. test/VNC의 기본14000 입력은 기존 PUSH 정책 평가용이며 새 학습 결과를 보려면 그 checkpoint 경로를 인자로 준다.


## PUSH AT progress 이동 gate

`stage1_jointpush_density3_cone90_finetune_atprogress_gate`는 기존 fine-tune 설정에서 `jointPush.at_progress_min_delta: 0.001`만 추가한다. 78k 초기화·PUSH50%·밀도3·cone90·AMP·CONTACT·동력 비용은 유지한다.

상자 중심과 목표의 수평 거리를 d_t, 감소량을 Δd_t = d_(t-1) - d_t로 둔다. PUSH의 AT progress는 현재 AT 성공이면 0.2, 미달이면 `0.2 * 1[Δd_t > 0.001] / (1 + max(d_t - 0.1, 0))`이다. 1mm와 같은 값은 지급하지 않는다. 성공 시 세 AT 항의 합계0.6과 paired CONTACT 포화는 그대로다. 성공 여부는 현재 step 기준이며 영구 latch가 아니다.

직전 거리는 제어 step 시작 때 저장하는 graph-bound 상자 위치에서 계산한다. 부분 reset에서는 기존 위치 이력을 새 배치로 갱신하며 reset 기하 평가에는 gate를 적용하지 않는다. 일반 carry AT와 다른 과제에는 적용하지 않는다. 30Hz에서 1mm/step은 목표 접근 속도 약0.03m/s에 해당한다. task 가중치0.5는 이후 기존 경로에서 적용한다.

새 설정은 checkpoint 계약에 포함되므로 기존 fine-tune checkpoint의 직접 resume는 계약 불일치로 거부한다. 기본 학습은 기존과 동일한78k 전이로 시작하며, 재개에는 새 설정으로 만든 checkpoint를 사용한다. 추론은 다른 호환 checkpoint도 가능하지만 보상 설정은 이 gate 컨피그를 따른다. 실행 명령은 [config](../config.md)를 따른다.
