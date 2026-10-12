# 공동운반 데이터 · 다른 서버에서 실행

현재 학습 입력은 `teamhoi_retarget_posture_all/`의 AMP 후진9개·옆걸음3개와
`carry_joint_corrected/`의 공동 RSI다. 두 데이터는 Git 포함 대상이므로 별도 ZIP이나 재보정이 필요 없다.
기존 TokenHSI 데이터·Isaac Gym 실행 환경·체크포인트는 사용자가 준비한 것을 재사용한다.

## Clone 후 최초 실행

저장소 루트에서 기존 `tokenhsi/data` 경로를 한 번 연결한다. 기존 데이터는 읽기만 하며,
이 checkout의 없거나 끊어진 데이터 링크만 만든다. 이미 유효한 다른 데이터는 덮어쓰지 않는다.

```bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
python tokenhsi/scripts/prepare_joint_carry.py --data-root /path/to/TokenHSI/tokenhsi/data
```

이미 데이터가 올바르게 연결되어 있으면 `--data-root` 없이 검사만 해도 된다.
또는 학습·평가 명령 앞에 `TOKENHSI_DATA_ROOT=/path/to/TokenHSI/tokenhsi/data`를 지정하면
두 LocoAMP 실험의 train/test/VNC 실행 경로에서 자동 연결·누락 파일·AMP/RSI SHA256 검사를 수행한다.
직접 `python tokenhsi/run.py`를 호출할 때는 위 준비 명령을 먼저 실행한다.

Stage1 체크포인트는 Git에 포함하지 않는다. 기존 파일의 절대 경로를 지정하거나
`stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00023000.pth`에 두면 된다.

```bash
TOKENHSI_GPU=5 \
STAGE1_CHECKPOINT=/path/to/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00023000.pth \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_task_embedding_train.sh 2 2048 4

TOKENHSI_GPU=5 \
STAGE1_CHECKPOINT=/path/to/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00023000.pth \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_head_only_train.sh 2 2048 4
```

GPU 번호는 새 서버에 맞춘다. 평가·VNC 명령은 [config.md](../markdowns/config.md#stage2-mixed80-locoamp)를 따른다.
새 checkout에는 `output/rsi_cache`가 없으므로 첫 실행에서 크기별 RSI 물리 검사를 수행해 자동 생성한다.
이 초기 준비는 평소 실행보다 오래 걸릴 수 있다. 학습/보상/AMP80·10·10/RSI 분포는 변경하지 않는다.

검증: Git 배포 대상만 공백 포함 새 경로에 복원한 뒤 GPU5·2048환경에서 두 모델의 학습·저장을 완료했다.
Stage1 epoch23000을 사용했고 frozen encoder69개는 원본과 동일했다. 최초 RSI 캐시 생성과 다음 실행의 재사용도 확인했다.
다른 서버의 드라이버·패키지는 기존에 동작하는 TokenHSI 환경을 전제로 한다.

## 포함 파일

| 경로 | 용도 |
| --- | --- |
| `teamhoi_retarget_posture_all/*.npy`, `amp_motions.yaml`, `manifest.json` | 검증된 현재 AMP 및 데이터 계약 |
| `teamhoi_retarget_posture_all/*_targets.npz` | AMP 보정 검사 재현용 목표 배열 |
| `carry_joint_corrected/{pickUp,carryWith,putDown}/{cooperative_rsi_candidate,screened_rsi_snapshots}.npz` | 공동 RSI 로더가 사용하는6개 파일 |
| `runtime_manifest.json` | 현재 학습용 AMP12개·RSI6개의 SHA256 |
| `teamhoi_reference/` | 재보정용 TeamHOI 원본12개·출처·라이선스 |
| `scripts/` | 생성·보정·물리 검사·영상 코드 |

원본과 보정본의 재생성은 선택 사항이다. 이미 검증한 파일을 그대로 쓰면 환경 차이에 따른 재보정 결과 차이를 피할 수 있다.
보정 재생성 후에는 검증·config 데이터 계약·`runtime_manifest.json`도 함께 갱신해야 한다.
이전 굽힌 보정본/영상/물리 보고서/학습 output은 Git에 포함하지 않는다.
4열 비교 영상을 다시 만들려면 먼저 `retarget_teamhoi_reference.py --all`로 이전 비교용 보정본을 생성한 다음
[현재 자세 보존 보정·검사·영상 명령](../markdowns/config.md)을 실행한다. 학습에는 현재 자세 보존 세트만 쓴다.

`HANDOFF.md`, `handoff_manifest.json`은 reset 연결 이전의 전달 기록이다.
현재 reset은 `tokenhsi/utils/joint_carry_rsi.py`로 연결되어 있으며 예전 전달 ZIP 검사는 clone 준비 절차가 아니다.
RSI319개·보정 AMP는 초기 물리/입력 검사를 통과한 데이터이며 연속 보행 안정성이나 학습 성공을 보장하지 않는다.
