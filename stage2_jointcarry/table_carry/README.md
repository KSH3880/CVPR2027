# 상판 아래 파지 · 직사각형 테이블 carry fine-tune

기존 앵커(grasp-view) SA4 epoch21000의 policy/critic/discriminator와 정규화 통계를 복사한다. PPO optimizer·epoch/frame은 새로 시작한다. Actor scene encoder와 actor 관측 RMS는 기존 Stage2처럼 동결하고 SA4/action head·Critic·판별기를 미세조정한다. 학습률은 기존2e-5에서1e-5로 줄였다.

실제 물체는 TeamHOI 기본 직사각형2.0×1.2×0.86m·60.5184kg다. `runs/table_carry/assets/`의 원본 크기·질량·관성 단일강체를 사용한다. 추론 전용 zero-shot 환경변수 경로와 분리된 `HumanoidMATableCarry`에서 학습한다.

- 두 사람, 4물체, 공동 AT만 사용한다. Mixed80 sampler 구현은 재사용하지만 전용 진입점이 joint_carry_at를 강제하므로 실제 초기화는 공동100%다. 단독/ON_TOP/집기는 이번 fine-tune 범위가 아니다.
- 물체 로컬 파지 중점은 `(0, ±0.45, 0.351)m`다. 상판 아래면은 로컬z=.39m, 손 충돌 sphere반경=.04m를 반영했다. 관측과 HOLDING 보상에 동일한 z 오프셋을 사용한다. 손 두 개는 각자 로컬x 약±.22m로 IK를 맞췄다.
- 들고 있는6가지 초기 자세는 기존 carryWith reference의 하체를 유지하고 팔 IK·발 높이를 보정했다. 테이블 다리 최하단은 초기 바닥 위30cm, 초기 속도는0이다. 정적 관절 기하 검사를 통과한 후보이며 동적 안정성은 별도 runtime 기록으로 판정한다.
- Goal360·CPA·AMP80/10/10·원래 배치 조건은 유지한다. 목표높이는 테이블 바닥 배치 높이다. 손 해제 보상이나 성공 조건은 추가하지 않았다.
- `table_carry` checkpoint 계약으로 원래 박스 모델과 resume/test 혼용을 거부한다. 원래 박스 모델은 최초 warm start에서만 명시적으로 허용한다. 원본 checkpoint와 asset은 수정하지 않는다.

CVPR2027/stage2_jointcarry에서 실행한다. 물리·pipeline·정책 모두 GPU6이며 기존 MPS daemon은 유지하고 이 작업만 직접 CUDA 컨텍스트를 사용한다.

```bash
# 새 fine-tune: 기본 source는 로컬 anchor epoch21000, 2048환경/seed42/2000 iteration
OUTPUT_PATH=output/table_carry/my_train bash table_carry/train.sh
# 학습 검증: 환경 수는2048 유지, 반복 횟수만 줄임
MAX_ITERATIONS=2 OUTPUT_PATH=output_etc/check/table_carry/my_check bash table_carry/train.sh
# 동일 table checkpoint에서 이어 학습
RESUME_CHECKPOINT=/absolute/path/table_checkpoint.pth OUTPUT_PATH=output/table_carry/my_resume bash table_carry/train.sh
# 화면 없는 평가 / VNC
OUTPUT_PATH=output_etc/check/table_carry/my_eval NUM_ENVS=16 REPEATS=3 bash table_carry/test.sh /absolute/path/table_checkpoint.pth
PORT=6088 OUTPUT_PATH=output_etc/check/table_carry/my_view bash table_carry/vnc.sh /absolute/path/table_checkpoint.pth
```

`table_carry/prepare_rsi.py`는 RSI 후보를 생성한다. 실행하면 출력 pool의 해시가 바뀔 수 있으므로 학습/재개 중에는 다시 생성하지 않는다. config의 `initial_states_sha256`와 일치해야 하며 임의로 계약 검증을 끄지 않는다.

본학습 체크포인트는 `output/table_carry/`, 기존 상태·검증 기록은 `../runs/table_carry/`에서 확인한다. 명령 파일의 존재와 운반 성능 검증은 구분한다.

2026-10-19 검증 기록: `../runs/table_carry/VALIDATION.md`(workspace 기준) 및 `current_training.json`. CPU23개·GPU 물리/부분reset·2048환경3epoch 저장/동결/갱신·16환경64step 재로드 통과. 고정 초기 PD는 약1초 전원 파지 후 붕괴하므로 학습 성공이나 장기 안정성은 별도 확인해야 한다.
