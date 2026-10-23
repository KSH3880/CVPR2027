# Stage1 객체 파인튜닝 실행 가이드

모든 명령은 저장소 루트(`/home/hwanhee/koo_cvpr/stage1_fix`)에서 실행한다. 우선 학습 명령만 정리한다.

## 공통 규칙

- Config·스크립트·실행 옵션 변경 시 아래 표와 학습 명령을 함께 갱신한다. 새 컨피그는 표와 명령 목록의 맨 아래에 추가하고 실험별 별도 섹션은 늘리지 않는다.
- 실행 전 `nvidia-smi`로 점유를 확인하고 명령에 `TOKENHSI_GPU`를 지정한다(ArmChair Adapter는 GPU4). 기존 프로세스는 임의 종료·재시작하지 않는다.
- 이 전용 wrapper의 학습 인자는 **`[envs]` 하나**다. `2048`을 전달하면 **2명·2048환경·4물체**로 실행된다. 일반 Stage1 wrapper의 `2 2048 4` 형식과 다르다.
- 본학습 전 `CHECK`, `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT`, `ASSETS_PER_CATEGORY` 잔여값을 해제한다. 기본값은 전체 train 객체다.
- 각 실험은 동일한39000 pth의 사본에서 시작해 별도 checkpoint를 만든다. KSH 원본 레포·원본 checkpoint는 수정하지 않는다.
- 출력은 `output/object_tokenizer_finetune/<실험명>/train/`이다. 이미 존재하는 출력 경로는 덮어쓰지 않으므로 재실행은 이 출력 루트 아래의 새 `OUTPUT_PATH`를 지정한다.
- 현재 wrapper는 `CHECK=1`을 지원하지 않는다. 짧은 확인은 `MAX_ITERATIONS=1`과 별도 `OUTPUT_PATH`를 사용하고2048환경을 유지한다. `output_etc/check/`는 현재 launcher의 허용 경로가 아니다.
- **기존 세 버전은 Obj tokenizer만 Actor에서 학습한다. 맨 아래 `CARRY 의자 Adapter`는 Obj tokenizer와 action adapter를 함께 학습한다. 두 경우 모두 기존 action head는 동결한다. `CARRY 의자 Copied Adapter`는 기존 Obj tokenizer도 동결하고 별도 복사본+adapter를 학습한다. Head 직접 학습의 객체별 분리 버전은 아직 이 목록에 없다.** Critic·AMP 판별기는 학습하고 사람/Goal tokenizer·Transformer·관측/AMP 정규화 통계는 동결한다.
- CARRY 의자·테이블은 각각 별도 학습한다. SIT는 별도 선택 항목이다. 공동 `joint_adapter`/`joint_head` 실험은 이 가이드에서 실행하지 않는다. 구조와 검증 상세는 [파인튜닝 문서](object_tokenizer_finetune.md)를 참조한다.

## 현재 컨피그

| 실험 | 객체 카테고리 | 환경 Config | 학습 Config |
| --- | --- | --- | --- |
| CARRY 의자 | `ArmChair_Normal` | [object_tokenizer_finetune_carry_chair.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/multi_agent/object_tokenizer_finetune_carry_chair.yaml) | [object_tokenizer_finetune_carry_chair.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/train/rlg/object_tokenizer_finetune_carry_chair.yaml) |
| CARRY 테이블 | `Table_Circle` | [object_tokenizer_finetune_carry_table.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/multi_agent/object_tokenizer_finetune_carry_table.yaml) | [object_tokenizer_finetune_carry_table.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/train/rlg/object_tokenizer_finetune_carry_table.yaml) |
| SIT 의자 | `StraightChair_Normal` | [object_tokenizer_finetune_sit_chair.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/multi_agent/object_tokenizer_finetune_sit_chair.yaml) | [object_tokenizer_finetune_sit_chair.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/train/rlg/object_tokenizer_finetune_sit_chair.yaml) |
| CARRY 의자 Adapter | `ArmChair_Normal` | [object_tokenizer_finetune_carry_chair_adapter.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/multi_agent/object_tokenizer_finetune_carry_chair_adapter.yaml) | [object_tokenizer_finetune_carry_chair_adapter.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/train/rlg/object_tokenizer_finetune_carry_chair_adapter.yaml) |
| CARRY 의자 Copied Adapter | `ArmChair_Normal` | [object_tokenizer_finetune_carry_chair_copied_adapter.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/multi_agent/object_tokenizer_finetune_carry_chair_copied_adapter.yaml) | [object_tokenizer_finetune_carry_chair_copied_adapter.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/train/rlg/object_tokenizer_finetune_carry_chair_copied_adapter.yaml) |

## 학습

잔여 옵션을 해제하고 점유를 확인한 뒤, 필요한 컨피그의 한 줄만 실행한다. 각 명령은 독립 학습이다.

```bash
unset CHECK MAX_ITERATIONS OUTPUT_PATH RESUME_CHECKPOINT ASSETS_PER_CATEGORY
nvidia-smi -i 6

# CARRY — ArmChair_Normal
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_train.sh 2048

# CARRY — Table_Circle
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_table_train.sh 2048

# SIT — StraightChair_Normal
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_sit_chair_train.sh 2048

# CARRY — ArmChair_Normal / Obj tokenizer + Action adapter
nvidia-smi -i 4
TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_adapter_train.sh 2048

# CARRY — ArmChair_Normal / 기존 Obj 보존 + 복사본 tokenizer·adapter 학습
TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_copied_adapter_train.sh 2048
```
