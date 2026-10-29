# 단독 KSH · 공동 jhh CONTACT 비교

현재 KSH19500 hard/balanced/safe/clear 실행의 `launch_base19500_hard_safe_clear_gpu5_20261028_214039/configs`를 복제한다. 생성기는 `--reference-configs`를 필수로 받고 CONTACT가 이미 바뀐 참조 설정은 거부한다. 원본 스냅샷 및 실행 중 학습에는 쓰지 않는다.

공동 PUSH(`members > 1`)에만 jhh CONTACT를 적용하고, 단독 PUSH(`members == 1`)의 state·progress·성공 판정은 KSH 그대로 유지한다. AT·gate는 양쪽 모두 현재 KSH 기준이다.

변경은 `env.stage2JointPush.contact_reward` 하나다. `train.yaml`·motion manifest는 바이트가 같고 환경의 다른 모든 값도 같다. CONTACT 구현 SHA를 모델 계약에 포함해 기존 Stage2 모델과 혼용하지 않는다. 기본 실행은 기존 Stage1 19500에서 시작하며 기존 Stage2 학습의 resume가 아니다.

아래 jhh 조건은 공동 PUSH에만 적용한다.

- state: 담당 rear region 손 중점 XY와 목표 높이1.04m. 높이 dead band ±3.25cm, 이후 sigma4cm.
- progress: 상자 장애물을 고려해 담당 rear region 1m 뒤까지 가는 경로 거리 `1/(1+d)`. 여유 폭45cm.
- 성공: 양손 region rear face 거리≤18cm, 중점XY≤18cm, 높이±3.25cm, 각 손/상자 반작용≥0.5N. control step 내 물리 substep hit를 OR한다.

기존 `jointpush/partial/reward.py:approach_distance`, `jointpush/ports/jhh_raw_cuboid/tokenhsi/env/tasks/multi_agent/push_reward.py:contact_geometry`를 그대로 복사했다. 실행 때 예전 소스를 import하지 않는다. AT state·progress·성공 조건, gate, 도착 시 CONTACT 포화, 보상 가중치, 그룹 합산은 현재 구현을 유지한다. 예전 jhh 실험 전체를 재현하는 설정은 아니다.

RSI·배치·물성·AMP·학습률은 유지한다. 특히 공동 PUSH의 현재 RSI 손 배치 높이0.8m와 CONTACT 목표1.04m가 다르므로 RSI 직후 CONTACT state는 낮을 수 있다. 이를 보상 외 초기 조건 변경과 혼동하면 안 된다.

전용 train/test/VNC 명령은 ../config.md에 있다. 기본 GPU5·conda tokenhsi_juan, 4명·1024학습환경·8 object token이다. test/VNC는 이 CONTACT 계약으로 학습한 `CHECKPOINT_OVERRIDE`를 요구하며 기본1환경이다. `KSH_CONTACT_REFERENCE_CONFIGS`로 다른 참조 launch를 명시할 수 있다. 새 실행 전에 기존 GPU 작업 수·MPS를 확인한다. 준비만 수행했으며 본학습은 실행하지 않았다.

CPU 테스트7개 통과: 원본 수식 수치 일치, 높이/성공 조건, 단독·공동 state/progress/physics 성공 분기, AT/gate 보존, 설정 차이 한 항목, checkpoint 격리, native no-op. CPU4환경·3preset 각8step 및 부분 reset 통과. 실행 검증 로그는 `output_etc/check/coupled_jhh_solo_ksh_contact_20261029/`.
