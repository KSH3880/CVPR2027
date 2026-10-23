# 기존 물체 tokenizer의 독립 fine-tuning

객체별 기존 버전(`sit_chair`, `carry_chair`, `carry_table`)은 KSH39000 모델의 구조/관측634D·물체30D를 그대로 가져온다. 새로운 tokenizer·adapter·40D 입력은 추가하지 않는다. 기존 actor의 `actor_encoder.tokenizers.1`만 학습하고 사람/목표 tokenizer·관계 embedding·transformer·action head는 동결한다. Critic encoder/value head와 기존 AMP 판별기는 복사한 가중치에서 학습한다. Actor 관측 RMS와 AMP RMS는 복사 후 고정한다. 처음 시작할 때 source optimizer·epoch/frame/env state를 이어받지 않는다. 같은 variant의 fine-tune checkpoint에 한해서만 명시적 resume가 가능하다.

원래 Obj tokenizer까지 보존하는 버전은 `carry_chair_copied_adapter`다. 별도 복사본 tokenizer와 adapter만 Actor에서 학습하며, base 분기는 원래 tokenizer를 사용하고 adapter를 비활성화한다. [설정·분기별 명령](config.md#armchair-carry-copied-adapter)을 따른다.

ArmChair 단독 adapter 버전은 `carry_chair_adapter`다. 기존 CARRY 의자 설정의 과제/객체/RSI를 유지하고 Obj tokenizer와 층별 action adapter를 학습한다. 기존 head는 동결하며 [학습 가이드](object_finetune_train_guide.md)와 [학습·추론·VNC 명령](config.md#armchair-carry-adapter)을 따른다.

## 공동 학습 비교: Adapter / Head

사용자 요청으로 추가한 `joint_adapter`와 `joint_head`는 같은39000 사본에서 시작하는 별도 모델이다. 두 버전 모두 SIT StraightChair50%·CARRY ArmChair25%·CARRY Table_Circle25%를 한 optimizer에서 함께 학습하고 모델 하나에 저장한다. 2048환경 중1024/512/512환경을 고정 배정하고 객체는 각 category의 train split 전체를 순환 배치한다. 같은 환경의 두 agent는 같은 task/category를 사용한다. CLIMB와 상자 replay는 이 비교에 포함하지 않으므로 기존 전체 task 성능 보존은 보장하지 않는다.

| 부분 | `joint_adapter` | `joint_head` |
| --- | --- | --- |
| 기존 Actor Obj tokenizer | 학습 | 학습 |
| Actor 사람/Goal tokenizer·Transformer·관계 embedding | 동결 | 동결 |
| 기존 Action head | 동결 | 전체 학습 |
| Action adapter | 각 Linear층의 병렬 잔차 학습 | 없음 |
| Critic 전체·AMP 판별기 | 학습 | 학습 |
| 관측/AMP RMS·행동 sigma | 동결 | 동결 |

Adapter는 TokenHSI `amp_network_builder_transformer_adapt.py`의 `internal_adapt_mlp`처럼 head의 각 Linear층 입력에 같은 입출력 차원의 Linear를 적용해 원래 층 출력에 더한다. hidden층과 마지막 action층 모두 적용하며 원래 활성함수는 유지한다. 모든 adapter weight/bias를0으로 초기화하므로 초기 행동은 기존 모델과 동일하다. Obj tokenizer는 새로 만들지 않는다. Head 버전은 원래 head를 직접 학습한다.

SIT 의자는 고정 지지체·loco/SIT RSI50/50, CARRY 물체는 동적·loco 초기화를 유지한다. AMP demo family도 SIT/CARRY50/50이며 각 family 안의 모션 혼합은 아래 원본 대응 표를 따른다. SIT만 실제 mesh별 후보 검사를 수행한다. 평가 wrapper는 같은 공동 모델을 test split의 세 category에 함께 적용하며 기본16환경은8/4/4로 나뉜다.

```bash
# 둘 중 실행할 버전을 선택한다. 기존 출력이 있으면 새 OUTPUT_PATH를 지정한다.
env -u MAX_ITERATIONS -u RESUME_CHECKPOINT -u OUTPUT_PATH -u ASSETS_PER_CATEGORY TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_joint_adapter_train.sh 2048
env -u MAX_ITERATIONS -u RESUME_CHECKPOINT -u OUTPUT_PATH -u ASSETS_PER_CATEGORY TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_joint_head_train.sh 2048

# Adapter: local / headless / VNC
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/object_tokenizer_finetune/joint_adapter/local_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_joint_adapter_test.sh /absolute/path/adapter.pth 4 3
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/object_tokenizer_finetune/joint_adapter/eval_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_joint_adapter_test.sh /absolute/path/adapter.pth 64 3
TOKENHSI_GPU=6 PORT=6128 OUTPUT_PATH=output/object_tokenizer_finetune/joint_adapter/vnc_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_joint_adapter_vnc.sh /absolute/path/adapter.pth 4 100

# Head: local / headless / VNC
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/object_tokenizer_finetune/joint_head/local_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_joint_head_test.sh /absolute/path/head.pth 4 3
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/object_tokenizer_finetune/joint_head/eval_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_joint_head_test.sh /absolute/path/head.pth 64 3
TOKENHSI_GPU=6 PORT=6129 OUTPUT_PATH=output/object_tokenizer_finetune/joint_head/vnc_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_joint_head_vnc.sh /absolute/path/head.pth 4 100
```

## TokenHSI 코드와의 대응

원본의 `basic_interaction_skills/amp_humanoid_sit.yaml`과 `adapt_interaction_skills/amp_humanoid_adapt_carry_box2objs_{chair,table}.yaml`, 해당 task Python을 확인했다.

| 실험 | 객체 | 초기화 | AMP 참조 |
| --- | --- | --- | --- |
| `sit_chair` | StraightChair_Normal train49 / test26 | loco50%·sit50%; 실제 객체별 PD 검사를 통과한 SIT 프레임 | loco20%·sit80% |
| `carry_chair` | ArmChair_Normal train63 / test27 | loco100%, 원본의 loco_more/loco50/50을 기존 checkpoint의 단일 loco 슬롯으로 투영 | loco70%·pickUp/carryWith/putDown 각10% |
| `carry_table` | Table_Circle train21 / test9 | loco100%, 위와 동일 | 위와 동일 |

TokenHSI의 형상 적응도 Chair/Table을 별도로 학습하며, CARRY config의 집기/운반/내려놓기 초기화 확률은0이다. 해당 동작은 AMP 데모에 사용한다. 우리 모델의 entity tokenizer 구조·관측/보상은 TokenHSI의 task tokenizer와 다르므로 원본의 전체 네트워크/학습 목표를 복제한 것은 아니다. 사용자의 기존 tokenizer 유지 요청에 따라 원본 적응 코드의 action adapter도 추가하지 않는다. SIT reward/성공은 좌판 metadata 높이를 사용하지만 정책에는 전체 native bbox30D를 전달한다. 방향/좌판 자체를 추가 관측하지 않는 표현 한계와 장기 수렴은 별도 평가 대상이다.

## RSI

원본 TokenHSI SIT는 `fix_base_link=True`로 의자를 고정한다. 이 포트도 SIT 지지체를 고정하되 GPU PhysX에서 tensor reset 위치를 따라가도록 중력 없음·속도 제한0·질량1e10kg의 reset 가능한 dynamic fixture를 사용한다. CARRY 의자/테이블은 원래 질량의 움직이는 물체다. 따라서 SIT 물성은 native URDF 질량 그대로가 아니다.

`ports/ksh_object_finetune`의 독립 SizeRsiCache를 사용한다. 원본 `output/rsi_cache`는 읽거나 덮어쓰지 않는다. 캐시 키는 실제 객체 ID/파일 fingerprint·물성/시뮬 설정·참조 모션·retarget/검사 코드를 포함한다. 동일 bbox의 다른 mesh가 후보를 공유하지 않는다.

SIT의 참조 `object_state.npy` Z는 바닥 원점일 수 있다. 실제 모션의 낮은 골반 phase(clip별 root-Z10분위−기존 pelvis clearance0.12m)를 기준 좌판 높이로 삼고, native 좌판 높이와의 차이를 사람 root/kinematic body에 같은 translation으로 적용한다. 정책 입력은 바꾸지 않는다. 이는 rigid translation retarget이며 관절 IK를 수행한 것은 아니다.

각 clip의 최대8개 프레임을 실제 collider/PD로0.1초 검사한다. 기존 root/물체 이동·속도·finite 기준에 발 바닥 높이·root 이동≤0.12m·물체속도≤2m/s를 추가한다. 후보 검사 때 mesh teleport 접촉 캐시를 비우고 candidate root/DOF 상태를 복원한다. 실제 partial reset은 다른 환경을 물리 step시키지 않는다. 통과 후보가 없으면 기존 availability 기반 fallback을 쓰고 요청/실행/fallback을 기록한다. `rsi_candidate_frames_per_clip: 0`은 전 프레임 검사다.

## 격리·실행

원본 `/home/hwanhee/ksh/stage1_task`의 code/config를 `ports/ksh_object_finetune`에 독립 복사했다. `UPSTREAM.json`에 출처·파일 hash를 보관한다. 시작 가중치는 private `weights/source_39000.pth`의 읽기 전용 사본이다. 원본 KSH 코드·checkpoint를 수정하지 않는다. 데이터는 읽기 전용 심링크, native asset/VHACD 입력은 각 새 output에 복사한다. main `tokenhsi/env`, `learning`, `utils`에는 fine-tuning 코드를 넣지 않는다. private cache는 `ports/ksh_object_finetune/output/rsi_cache`이며 기존 실행의 cache와 분리된다.

각 실행 출력은 `output/object_tokenizer_finetune/` 아래 **새 디렉터리**만 허용한다. 기존 디렉터리는 덮어쓰지 않는다. `ASSETS_PER_CATEGORY=1`은 검증용 대표 객체, 기본0은 전체 train/test split이다. 학습 환경 수는2048이다.

```bash
# GPU6 점유 확인 후, 원하는 실험의 명령을 선택한다. 각각 독립된 39000 출발 사본이다.
nvidia-smi -i 6
env -u MAX_ITERATIONS -u RESUME_CHECKPOINT -u OUTPUT_PATH -u ASSETS_PER_CATEGORY TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_sit_chair_train.sh 2048
env -u MAX_ITERATIONS -u RESUME_CHECKPOINT -u OUTPUT_PATH -u ASSETS_PER_CATEGORY TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_train.sh 2048
env -u MAX_ITERATIONS -u RESUME_CHECKPOINT -u OUTPUT_PATH -u ASSETS_PER_CATEGORY TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_table_train.sh 2048

# 짧은 학습 확인: 환경 수는 유지하고 반복만 제한한다. 각 variant에 동일하게 적용 가능.
TOKENHSI_GPU=6 MAX_ITERATIONS=1 OUTPUT_PATH=output/object_tokenizer_finetune/sit_chair/check_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_sit_chair_train.sh 2048

# Local viewer / headless / VNC: 같은 variant의 fine-tuned checkpoint만 사용
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/object_tokenizer_finetune/sit_chair/local_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_sit_chair_test.sh /absolute/path/fine_tuned.pth 1 3
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/object_tokenizer_finetune/sit_chair/eval_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_sit_chair_test.sh /absolute/path/fine_tuned.pth 64 3
TOKENHSI_GPU=6 PORT=6128 OUTPUT_PATH=output/object_tokenizer_finetune/sit_chair/vnc_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_sit_chair_vnc.sh /absolute/path/fine_tuned.pth 1 100
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/object_tokenizer_finetune/carry_chair/local_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_test.sh /absolute/path/fine_tuned.pth 1 3
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/object_tokenizer_finetune/carry_chair/eval_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_test.sh /absolute/path/fine_tuned.pth 64 3
TOKENHSI_GPU=6 PORT=6129 OUTPUT_PATH=output/object_tokenizer_finetune/carry_chair/vnc_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_vnc.sh /absolute/path/fine_tuned.pth 1 100
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/object_tokenizer_finetune/carry_table/local_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_table_test.sh /absolute/path/fine_tuned.pth 1 3
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/object_tokenizer_finetune/carry_table/eval_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_table_test.sh /absolute/path/fine_tuned.pth 64 3
TOKENHSI_GPU=6 PORT=6130 OUTPUT_PATH=output/object_tokenizer_finetune/carry_table/vnc_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_table_vnc.sh /absolute/path/fine_tuned.pth 1 100
```

저장마다 `finetune_update_audit.json`에서 동결 actor와 RMS의 tensor 완전일치를 검사한다. `finetune_initialization.json`에는 초기 model tensor 전체가39000 사본과 같은지, 실제 학습 parameter 목록을 기록한다. `integrity.json`은 KSH 원본 코드/config와 초기 checkpoint 사본 무변경을 검사한다. 장기 본학습은 사용자 명령으로 실행한다.

## 확인한 범위

CPU 동결/기존 모듈 보존 검사2개와 private Python·9개 wrapper 문법을 통과했다. GPU6의 SIT 전체49종·128평가환경 검사에서 후보3763프레임을 확보했다. 한 객체는 통과 후보가 없어 보행 fallback하며, 전체 fallback은91/5554요청(1.64%)이었다. 다른 환경 root/DOF/AMP 이력은 partial reset 전후 완전일치했고, 12step 관측·행동·AMP는 finite였다. SIT 물체 최대속도1.16e-8m/s·사람 root 최대속도2.63m/s 미만이었다. 결과는 `output/object_tokenizer_finetune/sit_fixed_full_rsi_check/rsi_audit.json`이다.

CARRY 두 실험은 전체 train63/21종·2048환경의 짧은 학습을 마쳤다. actor의 나머지70개 parameter와 관측/AMP RMS는 완전일치했고 기존 물체 tokenizer·critic·판별기는 실제 업데이트됐다. 저장본은 대표 test 객체·2환경·12step으로 재로딩했다(`carry_chair`, `carry_table` 각각의 `train_check`·`eval_check_v2`). 이는 장기 학습 성공률이나 모든 test 객체의 성능 검증이 아니다.

SIT 수정본도 전체 train49종·2048환경의 짧은 학습과 동일한 동결/업데이트 검사를 통과했다(`sit_chair/train_fixed_check`). 해당 저장본의 test 대표 객체·2환경·12step 재로딩도 rc0였다(`sit_chair/eval_fixed_check`). 초기39000 가중치는 model tensor 전체가 완전일치했고, 세 실험 모두 종료 시 KSH 원본 코드/config와 시작 checkpoint 사본의 해시가 유지됐다.
