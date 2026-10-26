# Stage 1 원본 + PUSH_AT

`KSH3880/CVPR2027`의 `stage1_stage2_base`, commit `28f02b3278f0a3e703cfa22a1e8cc376bb8ec512`에서 가져온 별도 실행본이다. 원본 파일 목록과 Git blob hash는 `UPSTREAM.json`에 있다. 실행 cwd와 Python 경로를 이 폴더로 제한하며, 메인 체크아웃의 실행 중 PUSH 학습 코드는 수정하지 않는다. 데이터는 메인 체크아웃으로 상대 심링크 연결한다.

원본 `stage1_v2_fixed.yaml`의 SIT·CLIMB·carry_at·carry_ontop, 해당 graph/RSI/보상 구현과 원본 config를 보존한다. `carry_at`을 교체하지 않고 `PUSH_AT`을 별도 과제로 추가한다. 기본 과제 확률은 기존 과제 0%, PUSH_AT 100%이며 Stage 2/coupled 학습은 사용하지 않는다.

| 항목 | 설정 |
| --- | --- |
| 정책 | 원본 task-shared embedding + TaskActionFusion + GTA, 별도 actor/critic, 64D·2head·4layer |
| 과제 ID | 기존 sit/climb/carry_at/carry_ontop = 0/1/2/3 유지, PUSH_AT = 4 추가 |
| embedding | 원본 4×64를 새 과제용 5×64로 확장; 기존 정책 과제 행을 재사용하지 않음. actor/critic embedding에 각64개 추가. AMP family 추가로 discriminator 입력1320→1330; 이 세 weight 외에는 모든 파라미터 이름·크기가 원본과 동일 |
| 입력 | 634D = 사람2×223 + 물체4×30 + 목표2×1 + pose8×7 + packet2×5, 8토큰 |
| AMP | 기존 carry/sit/climb family0/1/2 유지, 별도 PUSH family3 추가;133D×10 = 1330D |
| 장면 | 사람2·실제 상자4·목표2, 학습2048환경·600step |
| PUSH 상자 | 네 상자 모두 X1~1.5m/0.1m격자·Y2m·Z1.3m·각20kg·마찰0.6 |
| 배정 | 원본 canonical payload0/1 유지, 나머지2개도 실제 actor/관측/충돌에 포함 |
| 배치 | 출발1~2m·상자 yaw 랜덤·박스 로컬+Y 목표2~5m; 추가 상자는 전체 PUSH/RSI 경로 밖에 배치 |
| AT | 원본 그대로 exp(-10·거리²), XY progress buffer0.1m/σ1, 성공거리20cm·Z오차1mm, state/progress/success 각0.2 |
| CONTACT | 현재 PUSH의 공통 유한 면 양손 XY 거리·양손 반력/상자 힘 조건, 접근 progress; state0.4/progress0.2/success0.2 |
| 합산 | 원본 self_sum·본인1.0/동료0.0; 원본 AT 성공 시 선행 CONTACT 포화 유지 |
| 기울기 | AT은 원본 판정을 사용하므로 현재 PUSH의 AT soft-tilt shaping/성공 tilt gate는 적용하지 않음 |
| RSI | loco/PUSH50/50·원본 DOF·clip7 제외949후보·후반30%를70% 확률로 요청 |

PUSH의 기하·초기0.1초 PD로 승인된 Y2m bank를 source 계약으로 엄격히 검사하여 재사용한다. 새 AT 보상과 4물체 장면을 source 계약과 같다고 가장하지 않는다. 추가 상자는 이동/RSI 경로를 피하도록 새로 배치하고, 별도 CPU 장면 검사를 수행했다. 949개 후보 전체의 새 4물체 장기 rollout을 검증한 것은 아니다.

다른 과제를 다시 활성화하려면 `env.relationGraph.template_probabilities`를 조정한다. 과제에 맞는 원본 크기 분포와 원본 reset/RSI 코드가 남아 있으며 PUSH 크기는 PUSH 과제에만 적용한다. 이 로컬 체크아웃에는 기존 SIT/CLIMB/carry 모션 경로 중 누락된 데이터가 있으므로, 그 과제를 활성화하려면 해당 원본 데이터를 먼저 연결해야 한다. 이번 실제 물리 검증은 PUSH_AT 100% 설정이다.

저장소 루트에서 실행한다. 새 train/test/VNC는 `tokenhsi/scripts/multi_agent/stage1_base_push_{train,test,vnc}.sh`이며 출력은 `output/stage1_base_push*`로 분리한다. 학습은 scratch이며 기존 570D PUSH의 6000회 파일과 원본 4과제 모델을 직접 resume/test할 수 없다. 새 checkpoint에는 전용 graph/geometry/RSI/manifest 계약을 저장하고 simulator 생성 전에 CPU preflight로 검사한다. 검사용 무학습 fixture도 학습 resume를 거부한다.

```bash
# CPU 단위검사 / 실제 CPU 물리·player·checkpoint 검사 포함
bash ports/stage1_base_push/verify.sh
WITH_SIM=1 bash ports/stage1_base_push/verify.sh

# 본학습: 지정 GPU에 기존 Python CUDA client가 있으면 시작 전에 거부한다.
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo \
OUTPUT_PATH=output/stage1_base_push \
bash tokenhsi/scripts/multi_agent/stage1_base_push_train.sh 2 2048 4

# 짧은 학습 확인: 환경 수는2048 유지. GPU가 비어 있을 때 별도로 실행한다.
env -u RESUME_CHECKPOINT TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo \
OUTPUT_PATH=output/stage1_base_push_check_train MAX_ITERATIONS=2 \
bash tokenhsi/scripts/multi_agent/stage1_base_push_train.sh 2 2048 4

# 새 실험에서 학습한 checkpoint로 loco 시작 평가
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo \
TOKENHSI_VIEWER_LOW_VRAM=1 HEADLESS=0 \
bash tokenhsi/scripts/multi_agent/stage1_base_push_test.sh <Stage1BasePush.pth> 2 1 4 10
TOKENHSI_GPU=1 TOKENHSI_CONDA_ENV=tokenhsi_koo HEADLESS=1 \
bash tokenhsi/scripts/multi_agent/stage1_base_push_test.sh <Stage1BasePush.pth> 2 1 4 10
TOKENHSI_GPU=1 TOKENHSI_GUI_GPU_MODE=native TOKENHSI_CONDA_ENV=tokenhsi_koo \
TOKENHSI_VIEWER_LOW_VRAM=1 RESOLUTION=1280x720x24 PORT=6107 \
bash tokenhsi/scripts/multi_agent/stage1_base_push_vnc.sh <Stage1BasePush.pth> 2 1 4 10
```

VNC 접속: `http://localhost:6107/vnc.html?autoconnect=1&resize=remote`. 실행 전 [로컬 viewer 보호 규칙](../../markdowns/config.md#local-vnc-viewer)을 따른다. 새 경로의 GPU/PPO/GUI·학습 병행 안전성·정책 성능은 미검증이다. 학습/MPS 종료나 재시작은 수행하지 않았다.

검증 결과는 [REPORT.md](../../output/stage1_base_push_check/REPORT.md)에 기록한다. 무학습 fixture는 성공한 학습 정책이 아니며 성능 평가에 사용하지 않는다.

## PUSH50% 파인튜닝

`stage1_push_finetune.yaml`은 사용자 제공 `weights/Stage1V2Fixed_00078000.pth`의 기존4과제 가중치를 보존하고 PUSH를 추가하는 별도실험이다. PUSH50%·SIT5%/CLIMB12.5%/AT·ON_TOP각16.25%이며 기존과제 RSI/모션/보상을 유지한다. CONTACT각0.2, PUSH크기는XY1.6~2.2m/Z1.45~1.7m·min(X,Y)≥1.1Z의167종, 20kg/마찰0.4다. 출발1~10m·환경중심XY±4.5m목표/로컬+Y±45°·최초목표고정은유지한다. 현재bank/전이본/출력은stable이름으로구분한다. [명령·검증·실행상태](../../markdowns/config.md#stage1-push-finetune).
