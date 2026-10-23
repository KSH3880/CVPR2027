# ksh Stage 1 세 방식 이식

기존 학습 코드에 합치지 않고 `ports/ksh_stage1/`에 독립 실행본을 추가했다. 원본은 `/home/hwanhee/ksh/stage1_fix`의 2026-10-05 스냅샷이며 파일별 SHA256은 `ports/ksh_stage1/UPSTREAM.json`에 있다. 실행 시 이 복사본의 env/learning/utils/config만 사용한다. 원본 레포의 이후 변경은 자동 반영되지 않는다.

| 방식 | 네트워크 | 기본 결과 경로 |
| --- | --- | --- |
| `size_rsi` | actor/critic 별도 semantic EdgeEncoder MLP + GTA | `output/ksh_size_rsi` |
| `size_rsi_typed_bias` | actor/critic 별도 task·relation·방향별 attention bias + GTA | `output/ksh_size_rsi_typed_bias` |
| `size_rsi_typed_bias_message` | typed bias + attention value에 typed edge message | `output/ksh_size_rsi_typed_bias_message` |

세 방식의 task 분포·박스 크기·size RSI·보상은 원본 설정을 함께 복사했다. 기존 hybrid sampler를 결합한 실험은 아니다. task 분포는 HOLDING/SIT/CLIMB/ON_TOP/AT = 5/10/25/30/30%, carry 계열은 PD 유효 후보 중 후반 30%를 70% 확률로 요청한다. SIT/CLIMB의 행동 구간 RSI도 원본과 같다.

## 실행

저장소 루트에서 실행한다. GPU 점유 확인 후 번호를 지정하고 학습 환경은 2048을 사용한다. conda 기본값은 `tokenhsi`다. 각 방식의 checkpoint는 같은 방식의 test/resume에 사용한다. 기존 hybrid/unified checkpoint를 섞지 않는다.

```bash
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_train.sh 2 2048 4
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_train.sh 2 2048 4
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_message_train.sh 2 2048 4
```

학습 resume는 `RESUME_CHECKPOINT=<같은 방식 checkpoint.pth>`로 지정한다. 짧은 검증은 `MAX_ITERATIONS=1 OUTPUT_PATH=output/ksh_<방식>_check`를 사용한다. 본학습 전 이 변수들의 잔여값을 확인한다.

각 `<checkpoint.pth>`를 해당 방식 파일로 바꾼다. 기본 평가 시작 skill은 loco이며 RSI 시작 학습 성공률과 구분한다.

```bash
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_vnc.sh <checkpoint.pth> 2 1 4 10

TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_vnc.sh <checkpoint.pth> 2 1 4 10

TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_message_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_message_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_message_vnc.sh <checkpoint.pth> 2 1 4 10
```

## 경로와 격리

- 공통 launcher: `tokenhsi/scripts/multi_agent/ksh_port_common.sh`. 실행 cwd·Python 검색 경로를 private snapshot으로 설정한다. `OUTPUT_PATH`는 이 레포의 `output/ksh_*` 아래만 허용한다.
- 코드·YAML: `ports/ksh_stage1/tokenhsi/`. 기존 `tokenhsi/env`, `learning`, `utils`, cfg를 수정하지 않는다.
- 원본 PD 캐시는 private `ports/ksh_stage1/output/rsi_cache/`에 복사했다. 새 캐시도 이 디렉터리에 저장되어 기존 캐시와 분리된다.
- 대용량 데이터는 이 레포의 기존 `tokenhsi/data`를 상대 심링크로 참조하며 공유 원본은 읽기 전용이다. `lpanlib`, `body_models`도 이 레포를 참조한다. 실행에 ksh 원본 경로는 필요 없다.
- 로컬로 복사할 때 이 레포의 `ports/`, wrappers, 라이브러리를 함께 복사하고 기존 데이터 심링크 대상은 로컬 데이터 위치에 맞춰 연결한다. 코드만 복사하면 motion/asset 데이터는 포함되지 않는다.

CPU 검증: `bash ports/ksh_stage1/verify.sh`. 원본 계약·세 네트워크·size RSI·import 격리·스냅샷 해시를 검사한다. private 테스트는 기본 root pytest 수집에서 제외하고 이 명령으로 실행한다.

2026-10-05 검증: CPU 50개 통과, 세 방식 각각 GPU6/tokenhsi·2048환경 짧은 학습/epoch2 checkpoint 저장·유한값 검사 및 16환경·32step headless 재로드 평가 완료. typed bias/message table shape와 비영 norm도 확인했다. 결과는 `output/ksh_port_validation_20261005`와 별도 `output/ksh_*_check`, `output/ksh_*_eval_check`에 있다. 기존 코드 및 ksh 원본 파일 해시는 유지됐다. 장기 수렴 비교는 수행하지 않았으며 GUI/VNC는 문법·경로 검증만 했다.
