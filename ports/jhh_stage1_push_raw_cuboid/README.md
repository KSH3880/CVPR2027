# JHH Stage1 기존 과제 + PUSH fine-tuning 이식

원본 `/home/hwanhee/jhh/TokenHSI_stage1_stage2_base`의 GPU5 학습 `stage1_v2_fixed_push_360_raw_cuboid`를 2026-10-23에 독립 복사했다. 원본의 미커밋 코드도 포함한 시점 사본이며 [manifest](PORT_MANIFEST.json)에 원본 HEAD·파일 SHA256·데이터 링크·로컬 수정 목록을 기록했다. 원본 레포 및 실행 중인 학습에는 쓰기/중지/재시작을 하지 않았다. 루트 `tokenhsi/`와 기존 `ports/ksh_*`도 수정하지 않았다.

## 설정과 보존 범위

- 기존 SIT/CLIMB/CARRY_AT/CARRY_ON_TOP 50%, PUSH 50%; 원본 교사 KL 0.001, 학습률 1e-5, 2048환경·2명·4물체·634D 관측·4-family AMP 유지.
- PUSH X1.2~1.7/Y1.45~1.7/Z1.45~1.7m(Y≥Z),15~20kg,상자마찰0.3. 목표는 초기 면 법선 방향이며 epoch0/1000/3000에서 최대5/7/10m, 접근 반각45/90/180°. PUSH episode1200step.
- PUSH 시작은 원본 raw RSI50%/loco50%, 원본 bank/모션/보상 그대로. 이 설정은 현재 레포의 손중점≤1m·실제밀기구간 재선별 RSI와 별개의 외부 실험이다. 해당 기준으로 재선별하거나 새 영상/전수 물리 검수를 수행하지 않았다.
- 저장 완료된 **22500회** 모델과 교사 **73000회** 모델, 기존 RSI cache를 일반 파일로 복사했다. 원본 학습이 진행돼도 자동 갱신되지 않는다. 데이터12개 링크는 읽기 전용 공유 데이터의 절대 경로로 연결했고, PUSH 모션/asset/code는 사본에 있다. output은 원본과 공유하지 않는다.
- 로컬 변경은 conda 기본값 `tokenhsi`, 사본으로 고정한 `PYTHONPATH`, VNC 기본 포트6126, CPU 검사 교사 경로뿐이다. YAML·보상·RSI·학습 알고리즘은 원본 그대로다.

## 실행

모든 명령은 현재 저장소 루트에서 실행한다. wrapper가 사본으로 이동하므로 아래 체크포인트와 OUTPUT_PATH는 **사본 루트 기준**이다. 학습 명령은 준비만 했으며 실행하지 않았다. GPU5의 원본 학습에 영향을 주지 않도록 예시는 GPU6을 사용한다.

```bash
# 현재 복사본의 무결성 확인
python3 ports/jhh_stage1_push_raw_cuboid/verify_port.py

# 복사한22500회에서 별도 output으로 재개 학습 (직접 실행할 때만 시작)
TOKENHSI_GPU=6 RESUME_CHECKPOINT=output/stage1_v2_fixed_push_360_raw_cuboid/Stage1V2FixedPush360RawCuboid_21-16-00-23/nn/Stage1V2FixedPush360RawCuboid_00022500.pth OUTPUT_PATH=output/stage1_v2_fixed_push_360_raw_cuboid_resume22500 bash ports/jhh_stage1_push_raw_cuboid/tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_train.sh 2 2048 4

# 원본73000회 교사에서 새 fine-tuning (직접 실행할 때만 시작)
env -u RESUME_CHECKPOINT -u PUSH_CHECKPOINT -u CHECK -u MAX_ITERATIONS TOKENHSI_GPU=6 OUTPUT_PATH=output/stage1_v2_fixed_push_360_raw_cuboid_finetune73000 bash ports/jhh_stage1_push_raw_cuboid/tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_train.sh 2 2048 4

# 로컬 viewer: PUSH 보행 접근 시작
TOKENHSI_GPU=6 HEADLESS=0 bash ports/jhh_stage1_push_raw_cuboid/tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_test.sh output/stage1_v2_fixed_push_360_raw_cuboid/Stage1V2FixedPush360RawCuboid_21-16-00-23/nn/Stage1V2FixedPush360RawCuboid_00022500.pth 2 1 4 10

# 화면 없는 평가
TOKENHSI_GPU=6 HEADLESS=1 bash ports/jhh_stage1_push_raw_cuboid/tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_test.sh output/stage1_v2_fixed_push_360_raw_cuboid/Stage1V2FixedPush360RawCuboid_21-16-00-23/nn/Stage1V2FixedPush360RawCuboid_00022500.pth 2 4 4 10

# 서버 VNC: PUSH RSI 시작
TOKENHSI_GPU=6 PORT=6126 EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 bash ports/jhh_stage1_push_raw_cuboid/tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_vnc.sh output/stage1_v2_fixed_push_360_raw_cuboid/Stage1V2FixedPush360RawCuboid_21-16-00-23/nn/Stage1V2FixedPush360RawCuboid_00022500.pth 2 1 4 10

# 서버 VNC: 보행 접근 시작
TOKENHSI_GPU=6 PORT=6126 EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 bash ports/jhh_stage1_push_raw_cuboid/tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_vnc.sh output/stage1_v2_fixed_push_360_raw_cuboid/Stage1V2FixedPush360RawCuboid_21-16-00-23/nn/Stage1V2FixedPush360RawCuboid_00022500.pth 2 1 4 10

# 기존 과제 + PUSH 혼합 평가 (loco 시작)
TOKENHSI_GPU=6 TASK_GRAPH=random_scenario HEADLESS=1 bash ports/jhh_stage1_push_raw_cuboid/tokenhsi/scripts/multi_agent/stage1_v2_fixed_push_360_raw_cuboid_test.sh output/stage1_v2_fixed_push_360_raw_cuboid/Stage1V2FixedPush360RawCuboid_21-16-00-23/nn/Stage1V2FixedPush360RawCuboid_00022500.pth 2 16 4 10
```

VNC: `http://localhost:6126/vnc.html`(SSH 포트 포워딩). `EVAL_SKILLS=loco,push EVAL_SKILL_PROBS=0.5,0.5`는 PUSH 혼합 시작이다. `EVAL_SKILLS=push`는 `TASK_GRAPH=push`(평가 기본값)에서만 사용한다. 추론 마지막 인자로 episode step 수를 줄일 수 있다. 학습 전 셸에 남은 CHECK/MAX_ITERATIONS/RESUME_CHECKPOINT 등의 값을 확인한다.

## 검증

원본 CPU 검사40개 통과. GPU6에서22500회 모델의 PUSH loco/RSI 각4평가환경·12step 정상 종료, 각각8/8agent가 요청한 시작 모드로 집계됐다. 이 검사는 실행 경로 확인이며 밀기 성능·장기 성공률 검증은 아니다. 기존 과제 혼합16환경·12step에서 SIT4/CLIMB4/CARRY_AT5/CARRY_ON_TOP3/PUSH16 agent trial을 확인했다. VNC6126·1환경·12step도 정상 종료했다. 모델 유한값·YAML/RSI bank 계약·161개 데이터 경로·430개 파일/12개 공유 링크·Python/셸 문법을 확인했다. 결과 요약은 `output_etc/check/port_verification.json`이다. 검사 산출물은 사본 `output_etc/check/port_*/`에 둔다.
