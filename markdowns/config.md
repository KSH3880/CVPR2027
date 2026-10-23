# Multi-Agent Carry 실행 가이드

현재 실행 가능한 실험은 원본 1번과 Stage 1 **27·28·30·31·32·33·34·35·36·37번 및 paired/unified 변형**, Stage 2 **29번 및 SIT plane 변형**이다. 과거 9~26번 config와 전용 실행 스크립트는 정리했다. 모든 명령은 저장소 루트에서 실행한다.

## 공통 규칙

- **단독 점/영역 평가 + PUSH 평가**: [전용 실행 안내](../evaluation/single_match/README.md#명령). `evaluation/single_match/batch.py`로 KSH73000/PUSH22500의 기존4과제를 같은 초기 장면에서 비교하고, PUSH22500만 loco/RSI·2~5/5~10m를 분리한다. 기존 코드·학습 수정 없이 GPU6, 정식512환경×3seed·최대30초·첫episode ever, 점20cm/영역 기준을 같은 궤적으로 채점한다. headless·로컬·VNC6127 명령 포함. 현재는 짧은 실행 검증까지다.

- **Pro1/JHH 기존 과제+PUSH fine-tuning 이식**: `stage1_v2_fixed_push_360_raw_cuboid`는 독립 `ports/jhh_stage1_push_raw_cuboid/`에서 실행한다. [설정·22500회 재개/새 학습·로컬/headless/VNC 전체 명령](../ports/jhh_stage1_push_raw_cuboid/README.md#실행). 원본 GPU5 학습 유지, 사본 conda `tokenhsi`·GPU6·2048학습환경·VNC6126. 본학습은 실행하지 않았다.

- GPU6 PUSH의 학습 크기 중심 대칭XYZ 크기 제로샷은 [사본 명령](../ports/ksh_stage1_stage2_base/push/markdowns/USAGE.md#push-크기-제로샷)을 따른다. 보행 시작·위치 전환 없음·기존20kg/마찰 유지, 평가에서만 실제 상자 크기를 바꾼다.

- GPU6 PUSH의 목표 고정·2초 후 상자 이동(목표 반대편/사람 뒤) 시각화는 [사본 명령](../ports/ksh_stage1_stage2_base/push/markdowns/USAGE.md#push-중간-상자-이동-제로샷)을 따른다. `PUSH_BOX_SWITCH_MODES`는 골 전환 옵션과 별도로 사용한다.

- GPU6 PUSH의 실행 중90°/180° 목표 전환 시각화는 [사본 명령](../ports/ksh_stage1_stage2_base/push/markdowns/USAGE.md#push-중간-목표-전환-제로샷)을 따른다. 보행 시작·기존checkpoint로 평가하며 기본5초 뒤 목표만 바꾼다.

- GPU6 5과제 Stage1의 RSI 시각화는 `TASK_GRAPH=push_at EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0 TASK_CAMERA=agent`로 전용 test/VNC를 실행한다. [사본의 명령·추론 옵션](../ports/ksh_stage1_stage2_base/push/markdowns/config.md)을 따른다. 기존 viewer는 재실행해야 평가 RSI 수정이 적용된다.

- **GPU6 두 실험 1000회→PUSH 100% 재개**: [별도 출력·재개 명령](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push100_resume1000.md). 5과제 모델을 보존하고 `TASK_GRAPH=push_at`으로 과제를 고정한다.

- **5과제 PUSH 학습 로그·500회 재개**: CONTACT·PUSH 전용 AT·접촉 중 이동거리·목표거리 감소·episode 누적량은 [지표/재개 명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push_training_metrics.md)를 따른다.

- **GPU6 물성·다섯 과제 새 학습 두 설정**: `stage1_v2_fixed_five_task_depth180_20kg` / `stage1_v2_fixed_five_task_depth200_20kg_friction04`. 실제 상자4개·2048환경·최종 상자→목표1~5m·360° 보행/전신 표면간격1~5m. [조건·검증·영상](../ports/ksh_stage1_stage2_base/push/markdowns/reference/five_task_gpu6_variants.md), [학습·로컬·VNC 명령](../ports/ksh_stage1_stage2_base/push/markdowns/config.md).

- PUSH RSI 신규/변경 작업은 [PUSH RSI 기준](push_rsi_guidelines.md)을 먼저 따른다.

- **재선별 RSI 새 학습 두 설정**: `stage1_push_depth180_h100_rsi1m_at5deg_phase`·`stage1_push_depth180_h080_rsi1m_phase`; 커리큘럼 없이 목표1~4m, loco50%·선별 PUSH50%. [학습·로컬·VNC 명령](../ports/ksh_stage1_stage2_base/push/markdowns/config.md).

- **PUSH RSI 재선별 준비**: 손 높이≤1m와 실제 양팔 밀기 자세를 함께 만족하는5모션·1196프레임의 [별도 캐시·검수·재현 명령](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push_rsi_phase_selection.md). 기존 활성 학습 설정은 변경하지 않았다.

- **GPU6 두 RSI≤1m PUSH 거리 커리큘럼**: epoch0/1000/3000부터 목표1~5/1~7/1~10m. [설정·상태보존 이관](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push_goal_curriculum.md), [전용 학습·로컬·VNC](../ports/ksh_stage1_stage2_base/push/markdowns/config.md). 기존 프로세스는 자동 전환하지 않는다.

- **Y1.8m·20kg·RSI≤1m 선별 두 실험**: [1.0m/AT5°·0.8m/무제한 학습·로컬·VNC](#y18m20kgrsi1m-선별-두-실험). 원본RSI관절보정없음·목표1~4m·시작1~5m.

- **Y1.8m PUSH 손 목표1.0m 고정**: [전용 새 학습·로컬·VNC](#y18m-push-손-목표10m-고정). 20kg·완화state·성공state>0.9+0.5N 유지.

- **Y1.8m PUSH 높이 비교**: [0.65m 고정 / AMP 원본 높이·완화 state·실행 명령](#y18m-push-손-높이-비교). CONTACT 자체 성공은 state>0.9와 0.5N 반력이며 별도 거리 gate는 없다.

- **safe PUSH 3종 CONTACT 힘 조건 복구(2026-10-20)**: `stage1_push_midpoint_safe`, `stage1_push_depth180_safe`, `stage1_push_midpoint_safe_at5deg` 기본 YAML은 손 중점 점수>0.9와 양손 각각의 면 법선 반력≥0.5N·상자 면 법선 반력≥0.5N을 같은 물리 substep에서 요구한다. 기존 train/test/VNC 명령은 동일하다. 현재 실행 중 학습은 종전 조건을 유지한다. 변경 전 checkpoint는 새 기본 설정으로 직접 resume할 수 없으며, test/VNC가 학습 때 저장된 설정을 읽으면 종전 힘 조건 없는 판정을 유지한다. AT 연동 CONTACT 보상 포화는 유지한다.

- **GPU5 AT 성공 기울기≤5°**: [별도 scratch 학습·로컬·VNC](#gpu5-push-at-성공-기울기-5도).

- **GPU5/6 초기 관통 수정**: [각각의 새학습/로컬/VNC 명령](#push-초기-관통-수정). GPU5 시작1~2m·GPU6 시작1~5m 유지.

- **X만가변·Y1.8m고정 PUSH**: [GPU6 새학습/로컬/VNC](#stage1-push-y18m-깊이x-가변).

- **중점만 CONTACT·20kg Stage1 PUSH**: [새 실험과 학습·로컬·VNC](#stage1-push-중점-contact20kg). CONTACT 자체 성공은 state>0.9, 원본AT/paired포화 유지, X=Y1.2~1.5m·Z1.3m 고정·목표1~4m다.

- **PUSH·Stage1 SIT/CARRY 방식 RSI**: X/Y 1.2~1.5m, Z 1.3m 고정·25kg·중점0.65m. 물리조건 변경으로 RSI cache를 재생성한 뒤 `TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_rect_native_rsi_train.sh 2 2048 4`로 학습한다. [캐시 재생성·설정·검증 명령](../ports/ksh_stage1_stage2_base/push/markdowns/reference/stage1_push_rect_native_rsi.md).

- **직육면체 PUSH·양손 중점 고정점**: `TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_rect_point_train.sh 2 2048 4`. XYZ독립1.2~1.5m/5cm격자·20kg·뒷면중점높이0.65m·직선1~5m, [설정/RSI 검증](../ports/ksh_stage1_stage2_base/push/markdowns/reference/stage1_push_rect_point.md), [학습·로컬·VNC 명령](../ports/ksh_stage1_stage2_base/push/markdowns/config.md). 기존 PUSH checkpoint와 직접 혼용하지 않는다.

- **XYZ1.2~1.5m·동일질량·기울기 우선 분석**: `TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/analyze_push_rect_stability.sh`. 접촉체/실제humanoid를구분하는비학습검사이며학습설정은변경하지않는다. [조건·결과·한계](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push_rect_stability.md).

- **외부 Stage1 PUSH 힘 조건 제거 버전**: `TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_no_force_train.sh 2 2048 4`. 위치/높이 gate·AT·직선1~5m는 유지한다. 기존 힘 버전과 checkpoint는 직접 혼용하지 않는다. [설정·로컬/VNC 명령](../ports/ksh_stage1_stage2_base/push/markdowns/config.md).

- **외부 Stage1 기반 PUSH100%**: [명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/stage1_push.md), [사본의 학습·로컬·VNC 명령](../ports/ksh_stage1_stage2_base/push/markdowns/config.md). 기존2명·4상자입력/634D·GTA·AT를 유지하고1.3m/20kg/마찰0.6·직선1~5m·사용자CONTACT를 연결했다. 저장소 루트에서 `TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_train.sh 2 2048 4`로 실행한다. 원본 `upstream/` 보존. 기존 네 과제에 PUSH_AT을 추가하고 Stage2 coupled를 PUSH로 제한한 별도 설정은 [다섯 과제 명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/five_task_push_at.md)·[학습/로컬/VNC](../ports/ksh_stage1_stage2_base/push/markdowns/config.md)를 따른다. 최종 버전은 코드/CPU 확인까지이며 RSI·시뮬레이션·본학습은 추가 실행하지 않는다.

- **PUSH 상자를 돌릴 필요 없는 직선 목표**: [초기 면 법선 방향·거리2~5m·학습/로컬/VNC](#push-상자-정면-직선-목표). 높이 정렬 RSI 설정에서 목표 방향만 제한한 별도 실험.

- **PUSH 원본/crop/AMP 영상**: 저장소 루트에서 `python3 tokenhsi/scripts/multi_agent/render_push_crop_comparison.py`. numpy/scipy/Pillow/ffmpeg가 있는 CPU 환경용이며 GPU·checkpoint 불필요. 대표1clip의3열 스켈레톤 MP4를 `output/push_crop_comparison/`에 생성하며 재실행 시 이 영상·metadata·캡처를 갱신한다. 학습/원본 데이터는 변경하지 않는다.

- **객체별 파인튜닝 학습 명령 모음**: [별도 실행 가이드](object_finetune_train_guide.md). 현재 실행 가능한 CARRY 의자/테이블·SIT의 학습 명령만 정리하며, 기존 tokenizer-only와 ArmChair Adapter 버전을 구분한다.

- **물체 tokenizer 공동 fine-tuning 두 버전**: [Adapter / Head 설정·학습·로컬·VNC](object_tokenizer_finetune.md#공동-학습-비교-adapter--head). `object_tokenizer_finetune_joint_{adapter,head}_{train,test,vnc}.sh`, SIT50%/CARRY 의자25%/테이블25%, 기존 Obj tokenizer와 행동 보정 방식 비교.

- **PUSH 높이 정렬 RSI·중심 입력/점XY·Z구간**: [초기높이0.585~0.650m선별·GPU2학습](#push-높이-정렬-rsi). 기존408후보 RSI와 별도 비교.

- **PUSH 중심 입력·점XY/높이구간 보상(기존 RSI)**: [중점Z0.585~0.650m무감점·GPU2학습/GPU6검증](#push-중심-입력점xy높이구간-보상). 물체30D·570D 유지.

- **PUSH 높이 정렬 RSI 목표2~5m 재개**: [500회 상태보존 이관·학습/로컬/VNC](#push-높이-정렬-rsi-목표25m-resume).

- **PUSH 중심 입력·고정점 보상 비교용**: [고정접촉점Z0.6175m·RSI·학습/로컬/VNC](#push-중심-입력점-보상-scratch). 기존 물체30D·570D 입력 유지.

- **PUSH 접촉 앵커·고정1.3m·목표3~5m scratch**: [앵커Z0.585~0.650m·RSI·학습/로컬/VNC](#push-접촉-앵커-13m목표35m-scratch). 물체30D·570D 유지.

- **PUSH 크기별 손 중점 높이 분석**: [GPU6·실제접촉/PD·힘조건별5도검사](#push-크기별-손-중점-높이-물리-분석). 비학습·상자1.0~1.5m·마찰0.6.

- **PUSH 고정면·목표3~5m resume**: [이관·학습·로컬·VNC](#push-고정면-목표35m-resume). 기존 fixed-face의 모델/optimizer/AMP·통계/학습 횟수를 보존하는 별도 실험.

- **PUSH 고정면·양손 중점 중앙선**: [설정·학습·로컬·VNC](#push-고정면양손-중점-중앙선). `push_fixed130_fixed_face_midpoint_{train,test,vnc}.sh`, GPU6 고정1.30m/20kg 기반 별도 scratch 실험.

- **PUSH 2번·고정1.30m/20kg·AT 기울기 제한 없음**: 아래 [전용 설정·명령](#push-2번-고정130m20kgat-기울기-제한-없음). `push_fixed130_20kg_no_tilt_{train,test,vnc}.sh` 사용. 문 없는570D PUSH 전용scratch,크기변동없음·AT감쇠/성공각도제한없음·기존AT거리/높이조건유지. 구가변크기저장본은직접resume/test하지않는다.

- **PUSH 1번·크기1.20~1.40m + AT 기울기 감쇠**: 아래 [전용 설정·명령](#push-1번-크기120140m--at-기울기-감쇠). `push_size120to140_at_tiltfade_{train,test,vnc}.sh` 사용. 문 없는570D PUSH 전용 scratch이며 AT state/progress만5~15° 선형감쇠, AT 성공≤10°. 기존 학습과 checkpoint를 섞지 않는다.

- **GPU4·GPU6 PUSH 정책 무게별 비교**: `TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/compare_push_mass_success.sh`. GPU4 한 장에서 두 정책의 최신 번호 checkpoint를 동시 headless 평가한다(각248환경·600step·8회·seed8429, RSI혼합/보행 시작 각각3968 agent episode). `NUM_ENVS`는62배수, `EVAL_REPEATS`, `EPISODE_LENGTH`, `SEED`, `CHECKPOINT_GPU4/6`로 조절한다. `OUTPUT_PATH`는 존재하지 않는 경로만 허용하며 기본 `output/push_mass_success_gpu4_XXXXXX/` 자동 생성. 정책별log/JSON·`REPORT.md`·`per_mass.csv`·`mass_bins.csv` 출력. AT 에피소드 도달률과 성공 에피소드의 첫 성공 시점/성공 중 최대 기울기(도, yaw 제외)를 기록한다. 질량12.12~24.98kg와 크기1.10~1.40m가 함께 변하므로 질량 단독 효과는 아니다. 새 학습 config가 아닌 평가 전용이며 GPU 실행은 미검증이다.

- **기존 물체 tokenizer 독립 fine-tuning**: [TokenHSI SIT/ArmChair/Table별 설정·학습·로컬·VNC 명령](object_tokenizer_finetune.md). `ports/ksh_object_finetune`의 세 전용 env/train YAML만 사용하며 원본 KSH·main core·39000 checkpoint는 수정하지 않는다. 30D/기존 tokenizer 유지, critic/AMP 학습, 새 tokenizer/adapter 없음.

- **CLIMB 학습 크기 범위 내 비교**: [범위·native/동일 크기 상자 대조 명령](object_zero_shot.md#climb-학습-크기-범위-내-비교). GPU6·비학습 평가이며 원래부터 X/Y0.4~0.8m·Z0.25~0.5m에 드는 Box6/Cabinet2/Table3종을 사용한다. 리스케일 없이 bbox·질량·마찰을 맞춘 상자와 비교하며 형상에 따른 관성·COM은 다를 수 있다.

- **CLIMB 올라가기 사건 평가**: [Box/Cabinet/Table_Square 명령·정의](object_zero_shot.md#climb-올라가기-사건-평가). GPU6·39000 모델에 `--climb-ascent`를 사용하며 중앙 목표/종료 유지 없이 양발 윗면·접촉을 에피소드 중 한 번 만족하면 집계한다. 정책·학습 보상은 바꾸지 않는다.

- **39000 SIT·의자 CLIMB 좌판 입력 보정**: [세 조건 비교·로컬/VNC 명령](object_zero_shot.md#좌판-입력-보정-sitclimb). `compare_seat_proxy.py`는 GPU6의 비학습 평가이며 전체 bbox/좌판 높이/높이+폭을 동일 좌판 판정으로 비교한다. 새 학습 config는 없다.

- 의자 CLIMB 교차 과제 시각화는 [객체 시각화 문서](object_zero_shot.md)의 `--cases climb_chair`를 사용한다. SIT용 StraightChair_Normal의 전체 bbox와 기존 CLIMB 판정을 유지하며 GPU1·6123 명령을 제공한다.

- 39000 distill의 **객체 종류 제로샷·한 장면 시각화**: [원래 크기의 실제 메시·30D bbox 입력·GPU1 headless/로컬/VNC 명령](object_zero_shot.md). 기본은 원래 Isaac Gym native viewer/stack 카메라다. 수동10분 유지 또는 `--scene-mode loop --episode-length 600` 자동 반복을 선택한다. Space=일시정지/N·R=같은 객체 새 배치이며 reset 로그를 남긴다. 재실행 output은 자동 분리한다. KSH 원본은 읽기만 한다.

- PUSH만 학습하는 현재 AT0.1m 실험의 **문 관측·토큰 삭제 버전**은 아래 `PUSH 문 없는 정책` 섹션을 따른다. 현재 실행 설정은 변경하지 않으며 별도 YAML/output을 사용한다.

- OPEN/PUSH warmstart29000의 **4m 우회 접근·문 숨김·연속 녹화**는 아래 전용 평가 wrapper를 사용한다. 원래 YAML/checkpoint 계약을 유지하고 평가 시 시작 위치와 문 visual만 바꾼다. 녹화 전용 임시 YAML은 실행 중 생성한다. 새 학습 실험이 아니다.

  ```bash
  # 서버 VNC + 연속 녹화: Ctrl+C까지 사실상 계속 실행, 기본 60초 단위 MP4
  TOKENHSI_GPU=6 PORT=6106 bash tokenhsi/scripts/multi_agent/open_push_far_record.sh
  # 탑뷰 녹화: 두 사람·박스·목표 전체를 위에서 촬영
  TOKENHSI_GPU=6 PORT=6107 VIDEO_CAMERA=top OUTPUT_PATH=output_records bash tokenhsi/scripts/multi_agent/open_push_far_record.sh
  # 짧은 확인: 학습이 아닌 1평가환경·12step 두 번
  TOKENHSI_GPU=6 PORT=6116 EVAL_REPEATS=2 EPISODE_LENGTH=12 VIDEO_SEGMENT_SECONDS=0.2 VIDEO_WIDTH=640 VIDEO_HEIGHT=360 OUTPUT_PATH=output/open_push_far_record_check bash tokenhsi/scripts/multi_agent/open_push_far_record.sh
  # 녹화 없는 로컬 viewer: 같은 4m 시작·문 visual 숨김
  TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
  PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/view_open_push_far.py --task HumanoidMAOpenPush --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml --cfg_env tokenhsi/data/cfg/multi_agent/open_push_push180_contact_at_warmstart.yaml --motion_file tokenhsi/data/dataset_open_push_push180.yaml --checkpoint output/open_push_push180_contact_at_warmstart/OpenPushPush180ContactAtWarmstart_09-17-51-36/nn/OpenPushPush180ContactAtWarmstart_00029000.pth --num_agents 2 --num_envs 1 --num_objects 4 --task_graph push --task_camera agent --eval_skills loco --eval_skill_probs 1.0 --eval_repeats 100 --episode_length 1200 --no_video --test --eval'
  ```

  VNC는 `http://<서버주소>:6106/vnc.html`, 영상은 `output_records/videos/<실행시각>/part_000001.mp4`부터 저장한다. 기본 출력 루트는 사용자 지정 `output_records`이며 `OUTPUT_PATH`를 지정하면 해당 경로를 우선한다. 기본 1280×720/30FPS·agent0 중심 센서 영상이므로 viewer 설정 패널은 포함하지 않는다. `VIDEO_SEGMENT_SECONDS`, `VIDEO_WIDTH/HEIGHT`, `EVAL_REPEATS`, `EPISODE_LENGTH`, `SEED`, `OUTPUT_PATH`로 조절한다. Ctrl+C/정상종료/SIGTERM에는 마지막 짧은 구간도 마감한다. 강제종료/프로세스 crash에는 현재 구간 마감을 보장하지 않는다. 4m는 학습 시작거리1~2m 밖이며 우회 경로/성공을 보장하지 않는다.

  이 서버에서는 카메라를 켠 기존 직접 headless 실행이 `gym.create_sim()`에서 exit139로 재현됐다. wrapper는 기존 GUI와 동일한 GPU/Vulkan/Xvfb 초기화를 사용한다. `MAPlayerContinuous.run_eval()`에는 기본 녹화 시작 호출이 없으므로 전용 wrapper에서 명시적으로 녹화를 시작한다. GPU6·640×360·12step×2회 종료코드0·4개MP4(6/6/6/4frame) 전체디코딩 및 중단 시 partial 구간 마감/RGB색상 확인. Gymnasium 전환은 수행하지 않았다.

  `VIDEO_CAMERA=follow`(기본)는 agent0 추적, `top`은 문을 제외한 두 사람·박스·목표의 XY 범위와 화면비에 맞춰 탑뷰 높이를 조절한다. 탑뷰는 저장 센서와 VNC viewer 카메라에 모두 적용하며 기존 캐릭터 추적이 덮어쓰지 않도록 viewer 갱신도 교체한다. 탑뷰 저장 폴더는 `<실행시각>_top`이다. GPU6·1평가환경·12step·640×360 탑뷰 평가 정상종료, MP4 두 개6/5frame 전체디코딩·실제탑뷰프레임 확인(`output/open_push_far_top_record_check/`). VNC 탑뷰 화면 캡처는 `output/open_push_top_viewer_check/viewer.png`다.

  seed 기본값은 원래 `--no_video` viewer와 같은 `-1`(매 실행 무작위)이다. 같은 장면을 비교하려면 두 명령에 동일한 seed를 지정한다(viewer `--seed <값>`, 녹화 `SEED=<값>`). 녹화 폴더의 `metadata.json`에 실제 torch seed·checkpoint·카메라·해상도를 저장한다. 이전 녹화 기본값7931의비교에서GPU6·같은seed의원래viewer/녹화각69step전체관측·행동·사람/박스상태가정확히일치하고둘다같은step에조기종료했다(`output/open_push_far_compare/comparison.json`). 이 한 장면 검증은 seed별장기넘어짐비율을측정한것은아니다.

- ksh 세 방식 독립 이식: [코드 격리·checkpoint·명령](ksh_stage1_port.md). 기존 core 대신 `ports/ksh_stage1` 복사본을 실행하며 결과는 `output/ksh_*`로 분리한다.

- 5개 task의 고정 XY·크기/위치 Z 평가: [조건·명령·결과 해석](z_sweep.md). `evaluate_z_sweep.sh`는 학생 checkpoint의 headless 평가 전용이다.

- 후반 hybrid RSI: [설정·학습·평가·VNC 명령](hybrid_late_rsi.md). 같은 XYZ에서 80~95 → 70~80 → 60~70 → 50~60% 순으로 탐색하며 PD 검사와 현재 정책의 초기 rollout 관측을 결합한다.

- 학생 RSI 학습: [설정과 실행 명령](student_rsi_sampling.md). `approach_scenario_stage1_student_rsi_train.sh`의 agent별 task 비율은 HOLDING/SIT/CLIMB/ON_TOP/AT = **10/10/20/30/30%**다. ON_TOP·AT은 각각 HOLDING_ON_TOP·HOLDING_AT template이며 RSI skill 혼합과 별도다.

- 사용자 지정 조건부 AMP resume: `bash /home/hwanhee/koo_cvpr/runs/manual_resume_289953/launch_epoch7500_gpu6.sh`.
  별도 YAML의 `ampTaskConditioning: true`, `ampConditionedResume: true`로 프레임당 132차원 AMP를 복원한다.
  원본 unified 7500회 가중치/통계 전체, 초기화한 optimizer, 대상 random-binding reward/graph 계약을 사용하며 GPU 6·2048환경·총 40000회다.
  출력은 `output/shared_random_binding_resume1099002_full_amp_7500`, 로그/PID는 `../runs/manual_resume_289953/`에 저장한다. 기존 실행이 살아 있으면 중복 실행하지 않는다.

- 랜덤 크기 참조 RSI의 시간 후반/행동 단계 두 실험, distill 충돌 probe와 제한 사항:
  [reference_rsi.md](reference_rsi.md). 기준은 `unified_shared_random_binding`이며 학생 학습에 teacher를 사용하지 않는다.

- 학습 인자는 `[num_agents] [num_envs] [num_objects]`; Stage 1 기본값은 `2 2048 3`이다. 짧은 확인도 환경 수 2048을 유지하고 `MAX_ITERATIONS`만 줄인다.
- 실행 전 `nvidia-smi`로 점유를 확인하고 명령 앞에 `TOKENHSI_GPU`를 지정한다. 실행 중인 프로세스는 임의 종료하지 않는다.
- 본학습 전 셸의 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 확인한다.
- 학습은 기본적으로 scratch다. 기존 실험의 reward/checkpoint 계약을 섞지 않는다. Stage 2 전이는 아래의 `STAGE1_CHECKPOINT`를 사용한다.
- 평가·VNC 인자는 `<checkpoint.pth> [agents] [envs] [objects] [repeats]`다. `HEADLESS=0`은 로컬 viewer, `HEADLESS=1`은 화면 없는 평가다.
- 로컬 viewer와 서버 VNC의 상자 색은 매 reset의 과제 배정을 따른다. 단독 대상은 해당 에이전트 색, 공동 대상은 노란색, 나머지는 회색이다. AT goal 점도 해당 edge 담당 에이전트 색이며, goal 슬롯이 바뀌어도 배정을 따른다. 실행 중인 뷰어는 다시 시작해야 반영된다.
- 데이터 원본 `/home/hwanhee/CVPR2027/TokenHSI`는 읽기 전용이며, 이 저장소는 심링크로 사용한다.




39000 상자 CARRY/SIT/CLIMB 마찰0.6·장면별 자동 MP4/VNC 명령은 [객체 시각화 문서](object_zero_shot.md#상자-마찰06세-과제-한-장면씩-자동-녹화)를 따른다. 평가 전용이며 사용자가 직접 실행한다.


## GPU5 PUSH AT 성공 기울기 5도

`stage1_push_midpoint_safe_at5deg`는 GPU5 safe 설정의 중점0.65m·물성·RSI를 유지하고 AT 성공에만 상자 수직축 기울기≤5°(yaw 제외)를 추가한다. AT 성공에 따른 paired 포화도 이 조건을 따른다. 연속 state/progress 감쇠나 초과 시 종료는 추가하지 않는다. 기존 checkpoint와 직접 resume/test 혼용은 불가하며 새 scratch 학습이다. 본학습은 자동 실행하지 않는다.

```bash
env -u CHECK -u MAX_ITERATIONS -u OUTPUT_PATH -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=5 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_midpoint_safe_at5deg_train.sh 2 2048 4
TOKENHSI_GPU=5 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_midpoint_safe_at5deg_test.sh /absolute/path/to/checkpoint.pth 2 1 4 10
TOKENHSI_GPU=5 PORT=6115 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_midpoint_safe_at5deg_vnc.sh /absolute/path/to/checkpoint.pth 2 1 4 10
```

화면 없는 평가는 `HEADLESS=1`. 출력은 사본의 `output/stage1_push_midpoint_safe_at5deg/`다.

## PUSH 초기 관통 수정

발메시·구체/캡슐크기를반영해상자/바닥초기관통을방지하는전용설정이다. GPU5는X=Y가변·일반시작1~2m유지, GPU6는X가변/Y1.8m고정·일반시작1~5m유지이며둘다20kg·상자→목표1~4m다. [명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push_safe_start.md). 기존학습은자동중단하지않고새output으로시작한다.

```bash
# 저장소루트에서실행. 기존checkpoint와직접혼용하지않는다.
env -u CHECK -u MAX_ITERATIONS -u OUTPUT_PATH -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=5 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_midpoint_safe_train.sh 2 2048 4
env -u CHECK -u MAX_ITERATIONS -u OUTPUT_PATH -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_train.sh 2 2048 4

# 새실험checkpoint의절대경로를사용
TOKENHSI_GPU=5 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_midpoint_safe_test.sh /absolute/path/to/checkpoint.pth 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_test.sh /absolute/path/to/checkpoint.pth 2 1 4 10
TOKENHSI_GPU=5 PORT=6115 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_midpoint_safe_vnc.sh /absolute/path/to/checkpoint.pth 2 1 4 10
TOKENHSI_GPU=6 PORT=6116 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_vnc.sh /absolute/path/to/checkpoint.pth 2 1 4 10
```

## Stage1 PUSH Y1.8m 깊이·X 가변

X만1.20~1.50m가변·Y1.80m/Z1.30m고정·20kg·상자→목표1~4m·일반시작root–상자중심1~5m다. 손목표는−Y면중앙이고이동목표는+Y방향이다. 중점CONTACT/원본AT포화/native RSI를유지한다. [명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/stage1_push_depth180_native_rsi.md).

```bash
# 저장소루트에서실행. 기존학습은자동중단하지않는다.
nvidia-smi -i 6
env -u CHECK -u MAX_ITERATIONS -u OUTPUT_PATH -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_native_rsi_train.sh 2 2048 4

# 새실험checkpoint의절대경로를사용한다.
TOKENHSI_GPU=6 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_native_rsi_test.sh /absolute/path/to/checkpoint.pth 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_native_rsi_test.sh /absolute/path/to/checkpoint.pth 2 16 4 3
TOKENHSI_GPU=6 PORT=6116 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_native_rsi_vnc.sh /absolute/path/to/checkpoint.pth 2 1 4 10
```

## Stage1 PUSH 중점 CONTACT·20kg

[명세·캐시 준비](../ports/ksh_stage1_stage2_base/push/markdowns/reference/stage1_push_midpoint_native_rsi.md), [사본 실행 목록](../ports/ksh_stage1_stage2_base/push/markdowns/config.md). CONTACT는 양손중점 점수만 사용하고 `state>0.9`로 자체 성공을 판정한다. 원본AT·paired포화·AMP를 유지하며 X=Y1.20~1.50m·Z1.30m고정·20kg·목표1~4m다. 구CONTACT checkpoint와 혼용하지 않는다.

저장소 루트에서 실행:

```bash
# 전용 캐시가 준비되지 않았을 때 먼저 생성
TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/build_stage1_push_midpoint_native_rsi_cache.sh

# 새 본학습
nvidia-smi -i 6
env -u CHECK -u MAX_ITERATIONS -u OUTPUT_PATH -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_midpoint_native_rsi_train.sh 2 2048 4

# 로컬 viewer / 화면 없는 평가 / 서버 VNC: 새 실험 checkpoint의 절대경로 사용
TOKENHSI_GPU=6 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_midpoint_native_rsi_test.sh /absolute/path/to/checkpoint.pth 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_midpoint_native_rsi_test.sh /absolute/path/to/checkpoint.pth 2 16 4 3
TOKENHSI_GPU=6 PORT=6116 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_midpoint_native_rsi_vnc.sh /absolute/path/to/checkpoint.pth 2 1 4 10
```

## PUSH 상자 정면 직선 목표

`push_fixed130_point_zband_heightmatch_straight_goal2to5.yaml`은 기존 높이 정렬 RSI·목표2~5m 설정을 기준으로 `pushGoalRandomization.sampling: box_forward_uniform`을 사용한다. reset 시 상자의 초기 로컬 +Y 방향(한 면의 수평 법선)을 월드 좌표로 변환해 그 방향으로만2~5m 균등 거리에 목표를 만든다. 따라서 방향을 맞추기 위한 상자 회전이 필요 없다. 초기 상자 위치·yaw 무작위화는 유지하며, 목표는 이후 상자가 움직이거나 회전해도 고정된다. 정책의 실제 회전/기울기를 강제로 막는 기능은 아니다.

1.3m/20kg/마찰0.6·점XY/높이구간 보상·관측·AMP·2048환경은 유지한다. 전용 bank는 기존11개 높이 선별 자세를 동일하게 재사용하고 목표 분포 계약/출처hash만 분리했다. 전후보 PD 재검사는 수행하지 않았다. 기존360° 목표 checkpoint는 계약이 달라 직접resume/test할 수 없으며, 아래 명령은 새로 학습한다.

```bash
# 본학습: 기존 GPU6 학습을 자동으로 중단하지 않는다. 실행 전 점유 확인.
nvidia-smi -i 6
env -u TRANSFER_CHECKPOINT -u RESUME_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=6 SEED=42 OUTPUT_PATH=output/push_fixed130_point_zband_heightmatch_straight_goal2to5 bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_straight_goal2to5_train.sh 2 2048 4

# 로컬 viewer / 화면 없는 평가: 새 직선 목표 실험 checkpoint 사용
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/push_straight_view bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_straight_goal2to5_test.sh <직선목표_checkpoint.pth> 2 1 4 100
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/push_straight_eval bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_straight_goal2to5_test.sh <직선목표_checkpoint.pth> 2 16 4 3

# 서버 VNC
TOKENHSI_GPU=6 PORT=6116 OUTPUT_PATH=output/push_straight_vnc bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_straight_goal2to5_vnc.sh <직선목표_checkpoint.pth> 2 1 4 100
```

## PUSH 높이 정렬 RSI 목표2~5m resume

기존 높이 정렬 RSI 목표3~5m 실험의500회 checkpoint에서 목표 반경만2~5m 균등으로 바꾼다. 중심입력/손 높이구간/보상/물리/11개 RSI후보/AMP는 유지하며 별도 YAML·bank·output을 사용한다. 모델/optimizer/AMP/RMS·epoch/frame을 보존한 이관본만 직접 재개할 수 있다.

```bash
# 최초 이관: 기존 출력이 있으면 덮어쓰지 않는다(현재 생성 완료).
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/prepare_push_heightmatch_goal2to5_resume.py output/push_fixed130_point_zband_heightmatch_goal3to5/PushFixed130PointZBandHeightMatchGoal3To5_18-20-51-17/nn/PushFixed130PointZBandHeightMatchGoal3To5_00000500.pth output/push_fixed130_point_zband_heightmatch_goal2to5/resume/PushFixed130PointZBandHeightMatchGoal2To5_00000500.pth'

# GPU2 본학습 재개: 환경2048, MAX_ITERATIONS 잔여값 해제
nvidia-smi -i 2
env -u TRANSFER_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=2 SEED=42 OUTPUT_PATH=output/push_fixed130_point_zband_heightmatch_goal2to5 RESUME_CHECKPOINT=output/push_fixed130_point_zband_heightmatch_goal2to5/resume/PushFixed130PointZBandHeightMatchGoal2To5_00000500.pth bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_goal2to5_train.sh 2 2048 4

# 로컬 viewer / 화면 없는 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/push_fixed130_point_zband_heightmatch_goal2to5_view bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_goal2to5_test.sh output/push_fixed130_point_zband_heightmatch_goal2to5/resume/PushFixed130PointZBandHeightMatchGoal2To5_00000500.pth 2 1 4 100
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/push_fixed130_point_zband_heightmatch_goal2to5_eval bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_goal2to5_test.sh output/push_fixed130_point_zband_heightmatch_goal2to5/resume/PushFixed130PointZBandHeightMatchGoal2To5_00000500.pth 2 64 4 8
TOKENHSI_GPU=6 PORT=6132 OUTPUT_PATH=output/push_fixed130_point_zband_heightmatch_goal2to5_vnc bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_goal2to5_vnc.sh output/push_fixed130_point_zband_heightmatch_goal2to5/resume/PushFixed130PointZBandHeightMatchGoal2To5_00000500.pth 2 1 4 1000
```

검증: 이관 전후685tensor 동일·epoch500/frame65536000 보존·셸문법 통과. GPU6·2048환경500→502회/frame65798144·685tensor finite·정상저장 rc0, 전용test2평가환경12step×2재로딩 rc0(`output/push_fixed130_point_zband_heightmatch_goal2to5_{resume_check,eval_check}/`). 기존 높이선별 후보의 재사용이며 전후보 PD screen을 다시 수행하지 않았다. 본학습/GUI는 실행하지 않았다.

## PUSH 높이 정렬 RSI

`push_fixed130_point_zband_heightmatch_goal3to5.yaml`·동명 train/test/VNC/RSI wrapper는 기존408후보 RSI를 보존한 별도 실험이다. 중심30D/570D입력·점XY/높이0.585~0.650m무감점 보상·상자1.3m/20kg/마찰0.6·목표3~5m·AMP21expert·RSI혼합50%/후반요청70%는 유지한다. 바뀌는 것은 RSI 후보와 그 해시 계약이다.

`build_push_height_matched_rsi.py`는 원 point bank408후보를 각각2개 독립장면에서 시험한다. 초기 retarget 기준자세와 첫 물리스텝의 양손 중점이 모두 높이 구간 안이어야 하며,12controlstep동안 finite/낙상종료없음/root>0.65m/root drift<0.45m/box높이±0.08m·발지지/손면근접을 확인한다. 통과 후보만 새 `push_fixed130_point_zband_heightmatch_goal3to5_bank.json`에 저장하고 원본을 수정하지 않는다. 후속12step동안 손 높이가 계속 구간에 머무르는 조건은 아니며 정책의 높이 유지 성능을 보장하지 않는다.

실제 선별 결과는 **11자세/7모션(ID0/1/2/5/6/19/20)**이다. GPU6 RSI64환경/32768 agent reset 중 RSI16,388개 기준자세의 높이구간내비율100%·원중심관측동일·부분reset/AMP/12stepfinite·bank불변 rc0(`output/push_fixed130_point_zband_heightmatch_goal3to5_rsi_check/audit.json`). 전408후보×2장면의 첫스텝높이/12step안정성 원자료는 같은 경로의 `height_screen.json`이다.

후보 선별은 모션 다양성을 줄인다. RSI에 남지 않는 clip도 AMP expert에서는 유지한다. 학습 checkpoint는 bank경로/해시가 달라 기존408후보 실험에서 직접 재개하지 않는다. RSI wrapper는 최초 bank생성 후32768 agent reset에서 RSI 기준자세 구간내100%를 검증하며, 기존 bank가 있으면 출처/계약을 검사하고 감사만 반복한다.

```bash
# GPU6: 최초 전후보 screen + 실제 RSI/reset/관측 감사
nvidia-smi -i 6
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_goal3to5_rsi.sh

# GPU2: 높이 정렬 RSI로 scratch 본학습
nvidia-smi -i 2
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=2 SEED=42 OUTPUT_PATH=output/push_fixed130_point_zband_heightmatch_goal3to5 bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_goal3to5_train.sh 2 2048 4

# GPU6: 별도 짧은 학습 확인
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 SEED=42 MAX_ITERATIONS=2 OUTPUT_PATH=output/push_fixed130_point_zband_heightmatch_goal3to5_check bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_goal3to5_train.sh 2 2048 4

# 해당 실험 checkpoint: 로컬 / headless / 서버VNC
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/push_fixed130_point_zband_heightmatch_goal3to5_view bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_goal3to5_test.sh /absolute/path/checkpoint.pth 2 1 4 100
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/push_fixed130_point_zband_heightmatch_goal3to5_eval bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_goal3to5_test.sh /absolute/path/checkpoint.pth 2 64 4 8
TOKENHSI_GPU=6 PORT=6133 OUTPUT_PATH=output/push_fixed130_point_zband_heightmatch_goal3to5_vnc bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_heightmatch_goal3to5_vnc.sh /absolute/path/checkpoint.pth 2 1 4 1000
```

추가 검증: CPU48개·문법/diff통과, GPU6·2048환경 scratch3epoch/frame393216·685tensor finite·새bank계약확인·2평가환경12step×2재로딩 rc0(`output/push_fixed130_point_zband_heightmatch_goal3to5_{check,eval_check}/`).

본학습/장기 성공률/GUI는 미실행이다.

## PUSH 중심 입력·점XY/높이구간 보상

`push_fixed130_point_zband_goal3to5.yaml`·동명 train/test/VNC/RSI wrapper는 기존 ±2cm 고정점 버전과 구분한 현재 선택이다. 기존 상자 중심 관측30D/전체570D·1.3m/20kg/상자마찰0.6/바닥1·목표3~5m·600step은 유지한다. 학습 GPU2, 검증 GPU6이며 학습 환경은2048개다.

양손 중점 XY는 최초 목표 반대 고정면의 가로 중앙으로 유도하고, 높이는 **0.585~0.650m 전체를 무감점 구간**으로 인정한다. RSI 기준점은 구간 중앙0.6175m로 고정해 초기 자세/기존 bank를 유지하며, 숨겨진 무작위 목표 높이는 없다. 물체에 고정된 기준점이 상자 이동/회전을 따라가고 그 기준점의 월드Z±0.0325m를 구간으로 쓴다(기재한 지면 높이는 상자가 reset 자세일 때).

PUSH 점수는 `exp(-10*d²-(e/0.04)²)`다. `d`는 중점XY오차와 각 손의 면거리 중 최댓값, `e=max(|중점Z-기준점Z|-0.0325,0)`는 구간 밖으로 벗어난 거리다. 다른 오차가 없을 때 구간 내1.0, 경계 밖1cm0.9394·2cm0.7788·4cm0.3679다. 접촉 성공은 양 끝을 포함한 구간 안의 중점 높이·중점XY≤0.18m·각 손 면거리≤0.18m·손 각각/상자힘≥0.5N을 요구한다. ±2cm 추가 제한은 없다. 경계 판정에는 float32 오차용1e-6m 수치 허용만 둔다.

기존 접촉/AT 도착 포화·접근 progress·보너스·0.9/0.1공유·실제 상자 중심 AT와 task/AMP0.5/0.5를 유지한다. 각 손의 개별 높이/간격과 전도각을 새로 제한하지 않는다. 목적 계약은 `fixed_material_point_height_band_v2`이며 구 고정점/앵커 checkpoint는 직접 재개하지 않는다. RSI 기하/기준높이는 같아 기존 `push_fixed130_point_goal3to5_bank.json`408후보를 해시 확인하며 재사용한다. AMP21clip 유지·RSI만 생존후보 없는 ID7/8 제외다.

```bash
# GPU6 실제 RSI/reset/관측 감사
nvidia-smi -i 6
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_goal3to5_rsi.sh

# GPU2 scratch 본학습
nvidia-smi -i 2
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=2 SEED=42 OUTPUT_PATH=output/push_fixed130_point_zband_goal3to5 bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_goal3to5_train.sh 2 2048 4

# GPU6 별도 짧은 scratch 확인
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 SEED=42 MAX_ITERATIONS=2 OUTPUT_PATH=output/push_fixed130_point_zband_goal3to5_check bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_goal3to5_train.sh 2 2048 4

# 이실험 저장본으로 로컬 / headless / 서버VNC
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/push_fixed130_point_zband_goal3to5_view bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_goal3to5_test.sh /absolute/path/checkpoint.pth 2 1 4 100
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/push_fixed130_point_zband_goal3to5_eval bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_goal3to5_test.sh /absolute/path/checkpoint.pth 2 64 4 8
TOKENHSI_GPU=6 PORT=6132 OUTPUT_PATH=output/push_fixed130_point_zband_goal3to5_vnc bash tokenhsi/scripts/multi_agent/push_fixed130_point_zband_goal3to5_vnc.sh /absolute/path/checkpoint.pth 2 1 4 1000
```

검증: CPU48개 통과·GPU6 RSI64환경/32768 agent reset/12step rc0(`output/push_fixed130_point_zband_goal3to5_rsi_check/audit.json`),2048환경 scratch3epoch/frame393216·685tensor finite 및2평가환경12step×2재로딩 rc0(`output/push_fixed130_point_zband_goal3to5_{check,eval_check}/`).

RSI 감사에서는 초기 retarget 기준 자세의 중점이 높이 구간 안인 비율이 약4.95%였다. 어깨 회전만으로 팔 길이를 보존하는 리타겟은 도달 불가능한 낮은 목표를 제한하므로 초기 손이 항상 구간 안에 있지는 않다. 물리적 안정성/유한값 검증과 보상 접촉 성공은 구분한다. 기존 bank 재사용이며 새로운 전후보 PD screen은 아니다.

본학습은 자동 시작하지 않는다. 장기 성공률/GUI/전도 안정성은 새로 검증하지 않았다.

## PUSH 중심 입력·점 보상 scratch

`HumanoidMAOpenPushPoint`·`push_fixed130_point_goal3to5.yaml`은 앵커 입력 버전과 분리한 실험이다. 관측은 기존 `HumanoidMAOpenPush`의 상자 중심/중심속도·회전·bbox·AT목표·그래프 그대로이며 물체30D/전체570D다. 접촉점 좌표를 추가하거나 중심을 치환하지 않는다. 상자는1.3m/20kg/마찰0.6·바닥1,목표는초기상자중심에서3~5m/360°·600step·2048학습환경이다.

접촉점은 최초 목표 반대 주축면의 가로 중앙/표면에 둔다. 높이는 지면 위0.6175m(이전0.585~0.650m범위의중앙)로 고정한다. 점은 상자로컬좌표에 고정해 상자 이동/전체회전을 따른다. 높이를 입력에 숨겨 무작위로 바꾸지 않도록 `height_range_m: [0.6175, 0.6175]`만 허용한다. Stage2에서 임의 높이를 지시하려면 별도 입력 또는 알려진 결정 규칙이 필요하다.

양손 중점의 접촉점 XY오차와 두 손의 면거리 중 최댓값을 `d`, 중점Z오차를 `dz`로 두고 PUSH state를 `exp(-10*d²-(dz/0.04)²)`로 계산한다. 두 손 각각 면거리≤0.18m·손/상자힘≥0.5N·중점XY≤0.18m·중점Z≤0.02m일 때 접촉 성공이다. 각 손이 같은 점에 겹치도록 강제하지 않으며 손 간격/개별 높이는 자유다. 위치가 틀린 면접촉만으로 PUSH 보상이 포화되는 것을 막는다. 기존 접근 progress·접촉/도착 포화·보너스·공유와 실제 상자 중심 기준 AT는 유지한다. 새 전도 제약/5° 보장은 없다.

전용 `push_fixed130_point_goal3to5_bank.json`은 동일 기하/retargeting/모션 해시와 donor의0.6175m PD검사 포함 여부를 대조한 뒤, 기존3높이 PD screen408생존후보를 보수적으로 재사용한다. 원본 bank/AMP21clip은 유지하고, RSI에서만 생존후보 없는 ID7/8을 제외한다. 접촉 성공2cm는 초기RSI가모두2cm이내라는의미가아니다. 신규 point목적 계약 `fixed_material_point_midpoint_v1`을 사용하므로 기존 앵커/중앙선 저장본으로 직접 resume/test하지 않는다.

```bash
# GPU6·64평가환경 RSI/reset/중심관측 감사
nvidia-smi -i 6
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/push_fixed130_point_goal3to5_rsi.sh

# scratch 본학습: GPU2
nvidia-smi -i 2
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=2 SEED=42 OUTPUT_PATH=output/push_fixed130_point_goal3to5 bash tokenhsi/scripts/multi_agent/push_fixed130_point_goal3to5_train.sh 2 2048 4

# 별도 짧은 scratch 검증
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 SEED=42 MAX_ITERATIONS=2 OUTPUT_PATH=output/push_fixed130_point_goal3to5_check bash tokenhsi/scripts/multi_agent/push_fixed130_point_goal3to5_train.sh 2 2048 4

# 해당 point실험 저장본만: 로컬 / headless / 서버VNC
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/push_fixed130_point_goal3to5_view bash tokenhsi/scripts/multi_agent/push_fixed130_point_goal3to5_test.sh /absolute/path/checkpoint.pth 2 1 4 100
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/push_fixed130_point_goal3to5_eval bash tokenhsi/scripts/multi_agent/push_fixed130_point_goal3to5_test.sh /absolute/path/checkpoint.pth 2 64 4 8
TOKENHSI_GPU=6 PORT=6131 OUTPUT_PATH=output/push_fixed130_point_goal3to5_vnc bash tokenhsi/scripts/multi_agent/push_fixed130_point_goal3to5_vnc.sh /absolute/path/checkpoint.pth 2 1 4 1000
```

검증: CPU44개, GPU6·64평가환경/32768 agent reset의 원 중심 관측 완전동일·고정높이/부분reset/AMP·12step감사(`output/push_fixed130_point_goal3to5_rsi_check/audit.json`),2048환경 scratch3epoch/frame393216·685tensor finite 및2평가환경12step×2저장본재로딩 rc0(`output/push_fixed130_point_goal3to5_{check,eval_check}/`).

빨강/파랑 십자는 보상 접촉점이며 정책에 추가되는 입력이 아니다. 본학습은 자동 시작하지 않는다. 장기 성공률/GUI는 미검증이다.

## PUSH 접촉 앵커 1.3m·목표3~5m scratch

`push_fixed130_anchor_goal3to5.yaml`·`HumanoidMAOpenPushAnchor`는 별도 scratch 실험이다. 상자는1.3m정육면체/20kg/마찰0.6,바닥마찰1,AT도착점은초기상자중심에서3~5m/360°균등,600step·2048학습환경이다. 기존GPU6학습/원본Stage2는 변경하지 않는다.

에피소드마다각상자의접촉높이를0.585~0.650m에서균등샘플링한다. 최초목표반대주축면의가로중앙/표면위에앵커를만들고상자로컬좌표에고정한다. 면/높이는리셋한환경만갱신한다. 상자가회전/이동하면full quaternion으로앵커가따라간다. 표면안쪽15cm inset은사용하지않는다.

관측은물체30D/전체570D를유지한다. 상자GTA위치에는앵커XYZ,물체로컬선속도에는`v+omega×앵커로컬오프셋`을전달한다. 물리bbox/각속도/회전·사람/AT도착점·그래프는보존한다. 원본graspview와같은위치/강체점속도변환개념이며두사람이별도상자를담당하므로장면을두벌복제하지않는다. 별도XYZ추가33D초안은사용하지않는다. 기존Stage2와의직접checkpoint전이호환성을검증한것은아니다.

PUSH state는 지정한 물리 면까지의 두 손 거리 중 최댓값으로 계산한다(`exp(-10*d²)`). 접촉은 두 손 각각 면 거리≤0.18m·손/상자 힘 조건으로 판정한다. **양손 중점의 XY/높이 거리 보상·성공/접촉 gate는 없다.** 앵커 높이는 관측과 RSI 초기 자세에만 사용하므로, 손이 그 높이를 유지한다고 보장하지 않는다. 원본 graspview도 입력 치환 개념은 같지만 원본의 양손 중점 앵커 보상까지 복제하지는 않는다.

AT 목표 관측은 이동하지 않고, 보상·성공은 실제 상자 중심 기준이다. 기존 접근 progress·힘 접촉/도착 포화·보너스·공유는 유지한다. 새 기울기 제약은 없으며 전도/5° 제한은 보장하지 않는다. 계약은 `push_contact_anchor_view_30d_v2`/`fixed_material_face_no_midpoint_v1`로 중점 보상이 있었던 초안 저장본의 직접 재개를 거부한다.

RSI는 `push_fixed130_anchor_goal3to5_viewonly_bank.json`을 사용한다. 동일 높이/기하/retargeting/모션 해시를 대조한 뒤 기존 낮은 앵커 PD screen의 408후보를 출처 해시와 함께 복사한다. 원 bank는 수정하지 않는다. 21개 AMP expert는 유지하며, PD 생존 후보가 없는 2개 clip(ID7/8)은 RSI에서만 제외한다. 전후보 PD screen은 3개 높이×2장면×12step의 유한 검사이며 연속 높이/장기 안정성을 입증하지 않는다. 최초 생성 후 RSI wrapper는 reset/관측 감사를 반복하며, `ANCHOR_RSI_RESCREEN=1`은 전후보 screen을 다시 실행한다.

```bash
# 낮은높이 PD bank 재사용 검증 + 실제 reset/관측/AMP 감사: GPU6·64평가환경
nvidia-smi -i 6
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/push_fixed130_anchor_goal3to5_rsi.sh

# scratch 본학습: 기존가중치재개/전이없음·2048환경
nvidia-smi -i 6
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=6 SEED=42 OUTPUT_PATH=output/push_fixed130_anchor_goal3to5 bash tokenhsi/scripts/multi_agent/push_fixed130_anchor_goal3to5_train.sh 2 2048 4

# 별도 짧은 scratch 확인
TOKENHSI_GPU=6 SEED=42 MAX_ITERATIONS=2 OUTPUT_PATH=output/push_fixed130_anchor_goal3to5_viewonly_check bash tokenhsi/scripts/multi_agent/push_fixed130_anchor_goal3to5_train.sh 2 2048 4

# 이실험 저장본만: 로컬 / headless / 서버VNC
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/push_fixed130_anchor_goal3to5_view bash tokenhsi/scripts/multi_agent/push_fixed130_anchor_goal3to5_test.sh /absolute/path/checkpoint.pth 2 1 4 100
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/push_fixed130_anchor_goal3to5_eval bash tokenhsi/scripts/multi_agent/push_fixed130_anchor_goal3to5_test.sh /absolute/path/checkpoint.pth 2 64 4 8
TOKENHSI_GPU=6 PORT=6130 OUTPUT_PATH=output/push_fixed130_anchor_goal3to5_vnc bash tokenhsi/scripts/multi_agent/push_fixed130_anchor_goal3to5_vnc.sh /absolute/path/checkpoint.pth 2 1 4 1000
```

검증: CPU40개 통과. GPU6·64평가환경/32768 agent reset의 관측/속도/부분reset/AMP/12step 감사는 `output/push_fixed130_anchor_goal3to5_viewonly_rsi_check/audit.json`, 2048환경 scratch3epoch/frame393216·685tensor finite는 `output/push_fixed130_anchor_goal3to5_viewonly_check/`, 2평가환경12step×2 재로딩은 `output/push_fixed130_anchor_goal3to5_viewonly_eval_check/`에 있다. 장기 성공률·GUI/VNC는 미검증이다.

viewer의빨강/파랑십자는각owner접촉앵커,기존도착goal표시는AT목표다. 전용 bank만 사용하며 구 중앙선/중점 보상 초안 checkpoint는 직접 resume/test하지 않는다. 이실험자체의저장본은`RESUME_CHECKPOINT`로재개할수있다. 본학습은자동시작하지않는다.

## PUSH 크기별 손 중점 높이 물리 분석

[보고서](../output/push_height_analysis_20261018/REPORT.md)·[JSON](../output/push_height_analysis_20261018/summary.json)·[그림](../output/push_height_analysis_20261018/height_limits.png). GPU6·1.0~1.5m/0.05m간격·동일밀도(1.3m20kg)·상자마찰0.6·바닥1이다. 새학습config/보상/RSI변경이 아닌 비학습검사다.

```bash
nvidia-smi -i 6
TOKENHSI_GPU=6 OUTPUT_PATH=output/push_height_analysis_new bash tokenhsi/scripts/multi_agent/analyze_push_height.sh
```

새 출력경로만 허용한다. wrapper는 정확히알려진두점힘12705조건→두물리손접촉4950조건→실제humanoid관절PD1488조건→요약/그림을 순차실행한다. 학습정책은 사용하지 않으며 기존학습은 유지한다. 검사결과는5초 접촉체/4초 humanoid·60Hz계측·내부2substep의 유한조건이다. `HEIGHT_PD_QUICK=1`은PD검사기만16조건으로 제한하는 개발용옵션이고 wrapper는해제한다.

접촉체0.45/0.475/0.50H의495시험에서0.802~0.862m이동·최대0.040°였다. 실제humanoid775/1488은10cm이동/≤5°이나 지지/낙상필터후49,손외접촉의심도제외하면2건으로 전체크기의순수손밀기안전범위를확정하지못했다. 알려진수평력1~5mg는공통0.375~0.50H,수직±20%/양손불균형포함1~2mg는0.25~0.375H시험점들이통과했다. 임의힘에대한절대안전구간은아니다. 자세/힘을제한하지않고중점만맞추는보상으로≤5°를보장할수없다. 원자료는위보고서의3개output경로에보존했다.

## PUSH 고정면 목표3~5m resume

`push_fixed130_fixed_face_midpoint_goal3to5.yaml`은 초기 상자 중심→목표 수평거리3~5m 균등·방향360°를 사용한다. 고정면/양손 중점,1.30m/20kg,마찰,RSI50%·후반70%,600step·AT성공/보상·네트워크는 동일하다. 목표는 reset 후 고정한다. 최대5m 이동 작업구역을 분리하려고 배치반경8m·두상자최소거리12m·envSpacing20으로 넓혔다. 사람 시작거리1~2m는 유지한다.

구 checkpoint는 목표/RSI 계약이 달라 직접 resume하지 않는다. `prepare_push_goal3to5_resume.py`가 기존 fixed-face 계약을 검증하고 모델·optimizer·AMP/RMS·epoch/frame 전체를 보존한 별도 저장본을 만든다. bank943후보는 이전 PD검증 포즈를 그대로 보존하며 전용 계약과 출처hash를 기록한다. 목표 방향 분포/지역 reset 기하는 같고 반지름·전역배치만 달라 실제 목표 reset을 별도로 검사한다. 전후보 PD screen을 새로 수행했다는 의미는 아니다.

```bash
# 현재 확인된8500회에서 full-state 이관(출력 checkpoint가 이미 있으면 거부)
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/prepare_push_goal3to5_resume.py output/push_fixed130_fixed_face_midpoint/PushFixed130FixedFaceMidpoint_17-22-21-17/nn/PushFixed130FixedFaceMidpoint_00008500.pth output/push_fixed130_fixed_face_midpoint_goal3to5/resume/PushFixed130FixedFaceMidpointGoal3To5_00008500.pth'

# 재개 학습: GPU6·2048환경·기존 프로세스 종료/재시작은 자동 수행하지 않음
nvidia-smi -i 6
env -u TRANSFER_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=6 SEED=42 OUTPUT_PATH=output/push_fixed130_fixed_face_midpoint_goal3to5 RESUME_CHECKPOINT=output/push_fixed130_fixed_face_midpoint_goal3to5/resume/PushFixed130FixedFaceMidpointGoal3To5_00008500.pth bash tokenhsi/scripts/multi_agent/push_fixed130_fixed_face_midpoint_goal3to5_train.sh 2 2048 4

# 로컬 viewer / headless / 서버 VNC: 이관본 또는 새 실험 저장본만
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/push_fixed130_fixed_face_midpoint_goal3to5_view bash tokenhsi/scripts/multi_agent/push_fixed130_fixed_face_midpoint_goal3to5_test.sh output/push_fixed130_fixed_face_midpoint_goal3to5/resume/PushFixed130FixedFaceMidpointGoal3To5_00008500.pth 2 1 4 100
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/push_fixed130_fixed_face_midpoint_goal3to5_eval bash tokenhsi/scripts/multi_agent/push_fixed130_fixed_face_midpoint_goal3to5_test.sh output/push_fixed130_fixed_face_midpoint_goal3to5/resume/PushFixed130FixedFaceMidpointGoal3To5_00008500.pth 2 64 4 8
TOKENHSI_GPU=6 PORT=6129 OUTPUT_PATH=output/push_fixed130_fixed_face_midpoint_goal3to5_vnc bash tokenhsi/scripts/multi_agent/push_fixed130_fixed_face_midpoint_goal3to5_vnc.sh output/push_fixed130_fixed_face_midpoint_goal3to5/resume/PushFixed130FixedFaceMidpointGoal3To5_00008500.pth 2 1 4 1000

# 실제 RSI 감사: 비학습64환경·32768 agent reset
TOKENHSI_GPU=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/check_push_fixed130_rsi.py --task HumanoidMAOpenPush --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml --cfg_env tokenhsi/data/cfg/multi_agent/push_fixed130_fixed_face_midpoint_goal3to5.yaml --motion_file tokenhsi/data/dataset_push_only_push180.yaml --num_agents 2 --num_envs 64 --num_objects 4 --headless --no_video --test --eval --task_graph push --num_threads 1 --output_path output/push_fixed130_fixed_face_midpoint_goal3to5_rsi_check'

# 짧은 resume: MAX_ITERATIONS는 추가 횟수가 아닌 절대 epoch 상한
TOKENHSI_GPU=6 SEED=42 MAX_ITERATIONS=8501 OUTPUT_PATH=output/push_fixed130_fixed_face_midpoint_goal3to5_resume_check RESUME_CHECKPOINT=output/push_fixed130_fixed_face_midpoint_goal3to5/resume/PushFixed130FixedFaceMidpointGoal3To5_00008500.pth bash tokenhsi/scripts/multi_agent/push_fixed130_fixed_face_midpoint_goal3to5_train.sh 2 2048 4
```

검증: CPU21개·셸문법/diff 통과. RSI64환경·32768agent reset(GPU4에서 사용자GPU6지정 전 완료): 목표3.000006~4.999993m/평균3.997986m·초기AT0·RSI50.15%/후반70.36%·상자간최소12.000035m·고정면/부분reset/12stepfinite·bank불변 (`output/push_fixed130_fixed_face_midpoint_goal3to5_rsi_check/audit.json`). 이관685tensor 완전동일. 이후GPU6·2048환경8500→8502저장rc0/모델finite·전용평가2환경12step×2재로딩rc0. `_resume_check`·`_eval_check` 출력에 보존했다. 이후 검증은GPU6만 사용한다. 장기성공률/GUI미검증·본학습미실행.

## PUSH 고정면·양손 중점 중앙선

`push_fixed130_fixed_face_midpoint.yaml`은 GPU6 고정1.30m/20kg 설정의 별도 scratch 실험이다. 목표를 생성할 때 초기 상자 yaw 기준으로 목표 반대쪽 면 하나(주축, 동률은 Y)를 선택한다. 에피소드 동안 면 번호를 다시 선택하지 않는다. 면의 위치/yaw는 상자를 따라 움직이며 월드 좌표에 고정하는 방식은 아니다. 기존과 같은 yaw/XY 접촉 기하를 사용하며 roll/pitch를 반영한 실제 3D 접촉선 판정은 아니다.

`relationReward.push.contact_mode: fixed_face_midpoint`에서 양손 위치의 평균이 그 면의 세로 중앙선으로 가도록 PUSH state를 계산한다. 높이는 자유이며 각 손을 중앙선에 모으거나 좌우 간격을 강제하지 않는다. 양손 중점의 XY 거리와 각 손의 면까지 XY 거리 중 최댓값을 사용해 한 손이 먼데 평균만 맞는 경우도 방지한다. 접촉은 이 거리≤0.18m와 기존 양손 반력/상자 힘 조건을 요구한다. AT 보상·도착 시 PUSH 포화·무게/마찰·AMP·관측/네트워크는 그대로다.

기존 RSI bank를 재사용한다. 초기 목표 반대 주축면과 reference retarget/물성은 같으므로 PD screen 계약은 유지하며, 새로운 reward 계약은 기존 checkpoint와 다르다. `check_push_fixed130_rsi.py`는 새 모드에서 실제 초기면 일치·RSI 양손 중점 거리·부분 reset 면 보존을 추가 검사한다. 구 저장본을 직접 resume/test하지 않는다. 이 실험의 저장본만 `RESUME_CHECKPOINT`로 재개한다. 실행 중인 기존 학습에는 소스 수정이 실시간 반영되지 않는다.

```bash
# 새 본학습: scratch·2048환경
nvidia-smi -i 6
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=6 SEED=42 OUTPUT_PATH=output/push_fixed130_fixed_face_midpoint bash tokenhsi/scripts/multi_agent/push_fixed130_fixed_face_midpoint_train.sh 2 2048 4
# 짧은 확인: 별도 출력·2048환경
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 SEED=42 MAX_ITERATIONS=2 OUTPUT_PATH=output/push_fixed130_fixed_face_midpoint_check bash tokenhsi/scripts/multi_agent/push_fixed130_fixed_face_midpoint_train.sh 2 2048 4
# 이 실험의 저장본: 로컬 / headless / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/push_fixed130_fixed_face_midpoint_view bash tokenhsi/scripts/multi_agent/push_fixed130_fixed_face_midpoint_test.sh /absolute/path/checkpoint.pth 2 1 4 100
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/push_fixed130_fixed_face_midpoint_eval bash tokenhsi/scripts/multi_agent/push_fixed130_fixed_face_midpoint_test.sh /absolute/path/checkpoint.pth 2 64 4 8
TOKENHSI_GPU=6 PORT=6128 OUTPUT_PATH=output/push_fixed130_fixed_face_midpoint_vnc bash tokenhsi/scripts/multi_agent/push_fixed130_fixed_face_midpoint_vnc.sh /absolute/path/checkpoint.pth 2 1 4 1000
```
```bash
# 새 모드 실제 RSI/AMP/부분 reset 감사: 비학습64평가환경·256회 reset
TOKENHSI_GPU=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/check_push_fixed130_rsi.py --task HumanoidMAOpenPush --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml --cfg_env tokenhsi/data/cfg/multi_agent/push_fixed130_fixed_face_midpoint.yaml --motion_file tokenhsi/data/dataset_push_only_push180.yaml --num_agents 2 --num_envs 64 --num_objects 4 --headless --no_video --test --eval --task_graph push --num_threads 1 --output_path output/push_fixed130_fixed_face_midpoint_rsi_check'
```

검증: CPU48개·셸문법/diff 통과. GPU6·2048환경·MAX_ITERATIONS=2(기존 종료조건상3epoch) 저장 rc0, 전용 평가2환경·12step×2 재로딩 rc0. 출력은 위 `_check`와 `output/push_fixed130_fixed_face_midpoint_eval_check/`다. 추가 RSI 감사는 GPU6·64평가환경·32768 agent reset rc0: 초기 목표면 일치·RSI 중점 중앙선 최대0.060001m·부분 reset 면 보존·12step finite·bank 불변(`output/push_fixed130_fixed_face_midpoint_rsi_check/audit.json`). 기존 PD screen 전후보 재검사는 아니다. 본학습·장기성공률·GUI/VNC는 미검증이다.

## PUSH 2번: 고정1.30m/20kg·AT 기울기 제한 없음

`push_fixed130_20kg_no_tilt.yaml`은 기존GPU6 문없는PUSH 전용 설정에서 `pushSizeRandomization`을 제거하고 RSI bank만 전용사본으로 분리했다. 실제박스는모두1.30×1.30×1.30m·20kg·마찰0.6이며생성/reset중크기·무게를바꾸지않는다. 관측570D/물체30D/6토큰/AMP1310D·목표1~2.5m·PUSH/AT보상·손접촉·네트워크·AMP·RSI50%/후반요청70%는동일하다.

`contact_at`에는 `max_tilt`와`tilt_fade_start/end`가없어 AT state/progress의기울기감쇠나AT성공각도제한이없다. AT성공은기존근접도≥0.9·높이오차≤1mm이다. 기존 `push.max_tilt=0.35`는진단등에쓰이는값이며현재contact_at보상/성공/접촉을기울기로제한하지않는다. 손접촉은같은목표반대면의양손XY거리≤0.18m·각손반력≥0.5N·상자전방힘≥0.5N을그대로사용한다. 기존처럼AT도착시PUSH보상도포화한다.

새 실험은scratch기본이며장기학습은자동실행하지않았다. 크기·bank계약이달라기존가변크기checkpoint를직접resume/test하지않는다. `RESUME_CHECKPOINT`는이실험에서학습한저장본에만사용한다. GPU4의AT감쇠실험과별도실험이다.

```bash
# GPU6 본학습: scratch·2048환경
nvidia-smi -i 6
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=6 SEED=42 OUTPUT_PATH=output/push_fixed130_20kg_no_tilt bash tokenhsi/scripts/multi_agent/push_fixed130_20kg_no_tilt_train.sh 2 2048 4

# 짧은확인: 별도check출력·2048환경유지(기존loop는epoch>MAX_ITERATIONS에서종료)
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 SEED=42 MAX_ITERATIONS=2 OUTPUT_PATH=output/push_fixed130_20kg_no_tilt_check bash tokenhsi/scripts/multi_agent/push_fixed130_20kg_no_tilt_train.sh 2 2048 4

# 이실험에서학습한저장본으로 로컬viewer / headless평가 / 서버VNC
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/push_fixed130_20kg_no_tilt_view bash tokenhsi/scripts/multi_agent/push_fixed130_20kg_no_tilt_test.sh /absolute/path/checkpoint.pth 2 1 4 100
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/push_fixed130_20kg_no_tilt_eval bash tokenhsi/scripts/multi_agent/push_fixed130_20kg_no_tilt_test.sh /absolute/path/checkpoint.pth 2 64 4 8
TOKENHSI_GPU=6 PORT=6126 OUTPUT_PATH=output/push_fixed130_20kg_no_tilt_vnc bash tokenhsi/scripts/multi_agent/push_fixed130_20kg_no_tilt_vnc.sh /absolute/path/checkpoint.pth 2 1 4 1000
```

RSI bank `tokenhsi/data/push_fixed130_20kg_no_tilt_bank.json`은기존31크기전체PD통과943후보를동일하게보존하고검증범위내1.30m만유지한사본이다. 원본계약·모션hash·물성·reset기하동일성·strict subset을검사하고출처hash/검증범위를기록한다. 신규전후보PD재검사를주장하지않으며실제고정설정reset은아래감사로확인한다.

```bash
# CPU bank복구/동일성검사(다른내용파일덮어쓰기거부)
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/build_push_fixed130_bank.py'

# 비학습64평가환경:실제질량/RSI/AMP/부분reset/12step검사
TOKENHSI_GPU=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/check_push_fixed130_rsi.py --task HumanoidMAOpenPush --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml --cfg_env tokenhsi/data/cfg/multi_agent/push_fixed130_20kg_no_tilt.yaml --motion_file tokenhsi/data/dataset_push_only_push180.yaml --num_agents 2 --num_envs 64 --num_objects 4 --headless --no_video --test --eval --task_graph push --num_threads 1 --output_path output/push_fixed130_20kg_no_tilt_rsi_check'
```

검증: CPU115개·Python/셸문법·diff통과. GPU6·64평가환경·32768agent reset에서고정1.30m/20kg·초기AT성공0·RSI50.41%/후반요청69.47%·초기손목표거리최대0.1735m·부분reset보존·관측570D/물체30D/AMP1310D·12stepfinite·bank불변확인(`output/push_fixed130_20kg_no_tilt_rsi_check/audit.json`). GPU6·2048환경·MAX_ITERATIONS=2의scratch확인은기존종료조건에따라3epoch후저장rc0·모델finite·고정크기/질량/무감쇠계약을확인했다(`output/push_fixed130_20kg_no_tilt_check`). 전용test2환경12step×2재로드rc0(`output/push_fixed130_20kg_no_tilt_eval_check`). 장기본학습·GUI/VNC·장기성공률은미검증이다.

### 고정 PUSH의 이동 중 기울기·도착 전 재수평 평가

`output/push_fixed130_transit_tilt_20261017/`는 10000/12500/14500 저장본을 같은 seed8429·128scene·600step·2회씩 RSI 혼합/보행 시작으로 비교하는 별도 비학습 평가다. 초기≤5°→10°/20° 초과 연속3control step→기울어진 구간 순 목표접근량≥0.10m→첫 AT 성공≤5°를 기록한다. 전체 episode 비율과 성공 조건부 비율을 구분하며, 실패 episode의 도착 전 지표는 전체 유효 구간이다. physics substep 피크나 기울이기의 인과효과는 측정하지 않는다.

```bash
# GPU6 점유 확인 후 실행: 기존 학습은 유지, 세 저장본 순차 평가
nvidia-smi -i 6
bash output/push_fixed130_transit_tilt_20261017/run.sh
# 기존 NPZ/JSON에서 재집계·궤적 독립 일치 검사만 실행(CPU)
python3 output/push_fixed130_transit_tilt_20261017/report.py
```

결과는 `REPORT.md`·`summary.json`, 개별 결과는 `epoch_*/trials.json`·`audit.json`·`trajectory_*.npz`다. 같은 경로로 재실행하면 이 평가 결과를 갱신한다. 원래 학습 YAML/checkpoint/RSI bank는 유지한다.

### 고정 PUSH 접촉 형태·엄격90% 최신 평가

`output/push_fixed130_contact_analysis_20261017/ANALYSIS_REPORT.md`는 손목3D위치/순body반력으로 같은면·인접두면·반대두면·모호한모서리를추정하고기울기직전특징을비교한다. 최소15frame·접촉형태90%엄격판정의최신16500비교는 `output/push_fixed130_contact_latest16500_20261017/STRICT_REPORT.md`다. 물체별실제접촉센서가아니며접촉형태는정책행동결과다.

```bash
# 기존학습유지·GPU6에서16500평가재실행(같은평가결과경로갱신)
nvidia-smi -i 6
bash output/push_fixed130_contact_latest16500_20261017/run.sh
# 저장NPZ에서CPU접촉분석/엄격비교만재집계
python3 output/push_fixed130_contact_analysis_20261017/analyze.py
python3 output/push_fixed130_contact_analysis_20261017/strict_report.py output/push_fixed130_contact_latest16500_20261017
```

### 현재 고정 PUSH의 상자 마찰 0.2/0.4/0.6 시각화

비학습 전용 `push_fixed130_friction_test.sh`/`push_fixed130_friction_vnc.sh`는 원래 YAML/checkpoint/RSI 계약 검사를 유지한 뒤 실제 상자 actor의 shape 마찰만 평가 override한다. 새 학습 config가 아니며 학습에 사용하지 않는다. 바닥/발 마찰과 사람·물체 초기화는 유지하고 `BOX_FRICTION`은0.2/0.4/0.6만허용한다. 원래config의0.6과실제override값은 output의 `friction_override.json`에구분한다.

```bash
# 동일17000저장본·동일seed, 하나씩실행하고Ctrl+C종료후다음조건
CKPT=output/push_fixed130_20kg_no_tilt/PushFixed13020KgNoTilt_16-13-56-41/nn/PushFixed13020KgNoTilt_00017000.pth
TOKENHSI_GPU=6 BOX_FRICTION=0.2 SEED=8429 PORT=6132 bash tokenhsi/scripts/multi_agent/push_fixed130_friction_vnc.sh "$CKPT" 2 1 4 1000
TOKENHSI_GPU=6 BOX_FRICTION=0.4 SEED=8429 PORT=6134 bash tokenhsi/scripts/multi_agent/push_fixed130_friction_vnc.sh "$CKPT" 2 1 4 1000
TOKENHSI_GPU=6 BOX_FRICTION=0.6 SEED=8429 PORT=6136 bash tokenhsi/scripts/multi_agent/push_fixed130_friction_vnc.sh "$CKPT" 2 1 4 1000
# 로컬viewer / 화면없는평가: BOX_FRICTION만바꿔동일test wrapper사용
TOKENHSI_GPU=6 BOX_FRICTION=0.2 SEED=8429 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_fixed130_friction_test.sh "$CKPT" 2 1 4 1000
TOKENHSI_GPU=6 BOX_FRICTION=0.2 SEED=8429 HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_fixed130_friction_test.sh "$CKPT" 2 64 4 2
```

VNC접속은 `http://<서버주소>:6132/vnc.html`(0.4는6134,0.6은6136). 기본 output은 `output/push_fixed130_friction_view/f<마찰>/`. GPU6·각마찰2평가환경·12step·seed8429에서원래17000재로딩/실제shape마찰일치/정상종료확인(`output/push_fixed130_friction_view_check/`). VNC GUI자체는새로띄우지않았다. seed는같지만물성이바뀐이후궤적은달라질수있다.

warmstart29000실제손높이비교는 `output/warmstart29000_hand_height_20261017/REPORT.md`·`hand_height.json`·원시 `trace_*.npz`에저장한다. 평가명령은같은폴더 `run.sh`, CPU재집계는 `python3 output/warmstart29000_hand_height_20261017/report.py`. 원래warmstart도상자위치/yaw는랜덤이고목표는상자로컬전방2m이며현재360°랜덤목표와다르다.

## PUSH 1번: 크기1.20~1.40m + AT 기울기 감쇠

`push_size120to140_at_tiltfade.yaml`은 문 없는 PUSH 전용570D/물체30D/6토큰/AMP1310D를 기준으로 한다. 상자 한 변1.20~1.40m·0.01m간격21종을 생성 시 균등 배정하며 reset마다 크기를 바꾸지 않는다. 밀도20kg/(1.3m)^3 유지·실제 질량15.73~24.98kg·마찰0.6. 목표1~2.5m·손 접촉·네트워크·AMP·RSI 비율은 원래 설정을 유지한다.

`relationReward.contact_at`에 `tilt_fade_start=0.0872664626`, `tilt_fade_end=0.2617993878`, `max_tilt=0.1745329252`(rad)를 추가한다. 박스 로컬Z/월드Z 사이 각도θ를 사용하므로 yaw는 자유다. 배율은≤5°에서1,5~15°에서(15°−θ)/10°,≥15°에서0. AT state와progress의 성공포화 **후** 배율을 곱하므로 목표 도착 후에도 감쇠한다. AT progress는 `0.2/(1+max(d_xy−0.1,0))*배율`이며 성공시 기존 포화 규칙을 따른다(성공거리 약0.103m의0.1m초과 구간에서는 포화1을 적용). AT 성공은 기존근접도≥0.9·높이오차≤1mm에θ≤10°를 AND한다. AT 성공보너스0.2에는 배율을 곱하지 않는다. PUSH state/progress/접촉/보너스는 감쇠하지 않으며 기존 거리·높이 도착에 따른 PUSH 포화 규칙까지 유지한다. 기본 옵션 없는 실험과 기존tilt035는 종전 동작을 유지한다.

새 실험은 **scratch** 기본이다. 보상·크기·RSI bank 계약이 달라 기존GPU4/6 저장본을 직접resume/test하지 않는다. `RESUME_CHECKPOINT`는 이 실험에서 학습한 저장본에만 사용한다. 장기본학습은 자동 실행하지 않았으며 기존 학습을 중단하지 않았다.

```bash
# GPU6 예시. 실제 점유 확인 후 실행하며 GPU4를 사용할 때는 TOKENHSI_GPU만 바꾼다.
nvidia-smi -i 6
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=6 SEED=42 OUTPUT_PATH=output/push_size120to140_at_tiltfade bash tokenhsi/scripts/multi_agent/push_size120to140_at_tiltfade_train.sh 2 2048 4

# 짧은 확인은2048환경 유지·별도 output·반복수 제한. 기존 loop는epoch>MAX_ITERATIONS에서 종료.
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 SEED=42 MAX_ITERATIONS=2 OUTPUT_PATH=output/push_size120to140_at_tiltfade_check bash tokenhsi/scripts/multi_agent/push_size120to140_at_tiltfade_train.sh 2 2048 4

# 위 실험에서 학습한 저장본으로 로컬 viewer / headless 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/push_size120to140_at_tiltfade_view bash tokenhsi/scripts/multi_agent/push_size120to140_at_tiltfade_test.sh /absolute/path/checkpoint.pth 2 1 4 100
TOKENHSI_GPU=6 HEADLESS=1 OUTPUT_PATH=output/push_size120to140_at_tiltfade_eval bash tokenhsi/scripts/multi_agent/push_size120to140_at_tiltfade_test.sh /absolute/path/checkpoint.pth 2 64 4 8
TOKENHSI_GPU=6 PORT=6126 OUTPUT_PATH=output/push_size120to140_at_tiltfade_vnc bash tokenhsi/scripts/multi_agent/push_size120to140_at_tiltfade_vnc.sh /absolute/path/checkpoint.pth 2 1 4 1000
```

RSI bank `tokenhsi/data/push_size120to140_at_tiltfade_bank.json`은 원본1.10~1.40m의31종 전체PD검사를 통과한943개 후보를 그대로 보존하고 생성 크기만21종으로 제한한 별도 사본이다. 원본계약·motion hash·물성·reset기하의 동일성과 strict subset을 검사하고 출처SHA256·원래검증크기·유지크기를 기록한다. 모든후보에 신규PD검사를 다시 했다는 의미는 아니다. 원본bank는 수정하지 않는다.

```bash
# bank 복구/동일성 확인(CPU; 파일이 다른 내용으로 존재하면 덮어쓰기 거부)
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/build_push_size_subset_bank.py'

# 새 설정의 비학습 RSI/실제 질량/AMP/부분reset 검사:21종·42평가환경
TOKENHSI_GPU=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/check_push_at_tiltfade_rsi.py --task HumanoidMAOpenPush --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml --cfg_env tokenhsi/data/cfg/multi_agent/push_size120to140_at_tiltfade.yaml --motion_file tokenhsi/data/dataset_push_only_push180.yaml --num_agents 2 --num_envs 42 --num_objects 4 --headless --no_video --test --eval --task_graph push --num_threads 1 --output_path output/push_size120to140_at_tiltfade_rsi_check'
```

검증: 관련CPU108개·Python/셸문법·diff 확인. GPU6·42평가환경·21504agent reset에서21크기/실제질량15.7305~24.9795kg·초기AT성공0·RSI50.09%/후반요청69.60%·초기RSI손목표거리최대0.1735m·부분reset보존·AMP1310D/관측570D finite·12step 확인(`output/push_size120to140_at_tiltfade_rsi_check/audit.json`). 2048환경·MAX_ITERATIONS=2의scratch학습은기존종료조건에따라3epoch실행·저장후rc0(`output/push_size120to140_at_tiltfade_check`),전용test로2환경12step×2회재로드rc0(`output/push_size120to140_at_tiltfade_eval_check`). 장기수렴·성공률·GUI/VNC는미검증이다.

## PUSH 전용 570D AT 기울기 제한 (현재 GPU6 설정 기반)

현재 `push_only_physical_resume3500` 학습과 같은 문 없는 물리 PUSH 전용 설정이다. 입력570D·물체30D·6토큰·AMP1310D, loco/PUSH 전용 manifest, 기존RSI bank/샘플링/후반비율·물성·네트워크·보상식을 유지하며 `contact_at.max_tilt: 0.35`만 추가한다. AT 성공에 수직대비 박스기울기≤0.35rad(약20°)를 AND한다. yaw는 자유이며 성공보너스/성공포화만 새 조건을 따른다. 아래684D 문 포함 실험과 checkpoint를 혼용하지 않는다.

원본570D checkpoint를 명시적으로 이관한 사본을 사용한다. `convert_push_only_at_tilt_checkpoint.py`는 config/input/보상/네트워크/물리/RSI/모션 계약을 검사한 뒤 기울기 조건과 metadata만 바꾸고 model/optimizer/RMS/epoch/frame를 보존한다. 생성된사본을덮어쓰지않는다. 기존GPU6프로세스는변경하지않으며새프로세스부터조건이적용된다.

```bash
# 최초 한 번 이관: 현재 실행의 570D·6000회 저장본 (이미 생성한 경우 생략)
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/convert_push_only_at_tilt_checkpoint.py output/push_only_physical_resume3500/OpenPushContactAtFriction06RandomSizeGoal1To25At01NoDoorsPhysicalScratch_15-09-49-36/nn/OpenPushContactAtFriction06RandomSizeGoal1To25At01NoDoorsPhysicalScratch_00006000.pth output/push_only_at_tilt035_migration/epoch6000.pth'

# GPU0에서570D 상태그대로resume·2048환경 (새학습을자동실행하지않음)
nvidia-smi -i 0
env -u TRANSFER_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=0 SEED=42 RESUME_CHECKPOINT="$PWD/output/push_only_at_tilt035_migration/epoch6000.pth" OUTPUT_PATH=output/push_only_at_tilt035_resume6000_gpu0 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_tilt035_train.sh 2 2048 4

# 로컬viewer / headless평가 / 서버VNC (동일570D 설정의 저장본 사용)
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_tilt035_test.sh output/push_only_at_tilt035_migration/epoch6000.pth 2 1 4 100
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_tilt035_test.sh output/push_only_at_tilt035_migration/epoch6000.pth 2 64 4 8
TOKENHSI_GPU=6 PORT=6122 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_tilt035_vnc.sh output/push_only_at_tilt035_migration/epoch6000.pth 2 1 4 1000

# GPU6 비학습RSI/AMP검사 (62평가환경/31크기)
TOKENHSI_GPU=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/validate_at_tilt.py --task HumanoidMAOpenPush --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml --cfg_env tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_tilt035.yaml --motion_file tokenhsi/data/dataset_push_only_push180.yaml --checkpoint output/push_only_at_tilt035_migration/epoch6000.pth --num_envs 62 --num_agents 2 --num_objects 4 --headless --no_video --test --eval --task_graph push --num_threads 1 --output_path output/push_only_at_tilt035_rsi_amp_check'
```

검증: CPU57개통과. GPU6·570D/물체30D/AMP1310D·62평가환경/31크기·31744reset에서초기AT성공0·기울기초과0·RSI49.97%/후반70.52%·bank SHA256동일·부분reset보존·원본모델/optimizer/RMS/epoch/frame완전보존및실제actor/critic/AMP추론12stepfinite(`output/push_only_at_tilt035_rsi_amp_check/audit.json`). 전용test스크립트2환경12step×2재로드rc0(`output/push_only_at_tilt035_eval_check/`). 본학습미실행·기존GPU6학습유지,새조건장기성공률미측정.

## PUSH AT 기울기 제한 (별도 문 포함 684D 실험)

`open_push_contact_at_friction06_random_size_goal1to25_at01_tilt035.yaml`은 문 포함 AT01의 관측684D·물체40D·AMP1310D·네트워크·물성·RSI bank/비율·기타 보상식을 유지하고 `contact_at.max_tilt: 0.35`(약20°)만 추가한다. 박스 로컬Z와 월드Z의 각도로 판정하므로 yaw는 허용한다. AT 기존 근접도≥0.9·높이오차≤0.001m에 기울기≤0.35rad를 AND하며 성공 보너스와 성공에 따른 state/progress 포화 전에 적용한다. 따라서 기울기 초과에서는 성공 보너스/포화만 사라지고 거리 기반 보상은 유지된다. 기존 옵션 없는 실험의 판정은 바꾸지 않는다.

보상 계약이 달라 원본 checkpoint를 직접 resume/eval하지 않는다. 아래 명시적 이관은 원본 문 포함 AT01과 네트워크/보상/RSI bank/manifest를 검사하고 AT 옵션과 관련 metadata만 바꾼 사본을 만든다. 모델·optimizer·RMS·epoch/frame 등 학습 상태는 그대로 보존한다. actor-only 전이는 거부한다. 생성된 사본은 덮어쓰지 않는다.

```bash
# 최초 한 번만 이관 (이미 생성된 경우 생략)
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/convert_at_tilt_checkpoint.py output/open_push_contact_at_friction06_random_size_goal1to25_at01/OpenPushContactAtFriction06RandomSizeGoal1To25At01_14-18-17-32/nn/OpenPushContactAtFriction06RandomSizeGoal1To25At01_00006000.pth output/at_tilt035_migration/epoch6000.pth'

# GPU4 점유 확인 후 사용자 본학습: 6000회부터 재개, 2048환경
nvidia-smi -i 4
env -u TRANSFER_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=4 SEED=42 RESUME_CHECKPOINT="$PWD/output/at_tilt035_migration/epoch6000.pth" OUTPUT_PATH=output/open_push_contact_at_friction06_random_size_goal1to25_at01_tilt035_resume6000_gpu4 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_tilt035_train.sh 2 2048 4

# 새 설정 checkpoint: 로컬 viewer / 화면 없는 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_tilt035_test.sh output/at_tilt035_migration/epoch6000.pth 2 1 4 100
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_tilt035_test.sh output/at_tilt035_migration/epoch6000.pth 2 64 4 8
TOKENHSI_GPU=6 PORT=6121 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_tilt035_vnc.sh output/at_tilt035_migration/epoch6000.pth 2 1 4 1000

# GPU6 비학습 RSI/AMP/보존 검사 (31종크기·62평가환경)
TOKENHSI_GPU=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/validate_at_tilt.py --task HumanoidMAOpenPush --cfg_train tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml --cfg_env tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_tilt035.yaml --motion_file tokenhsi/data/dataset_open_push_push180.yaml --checkpoint output/at_tilt035_migration/epoch6000.pth --num_envs 62 --num_agents 2 --num_objects 4 --headless --no_video --test --eval --task_graph push --num_threads 1 --output_path output/at_tilt035_rsi_amp_check'
```

검증: CPU87개(기울기 경계/yaw·보상/RSI 불변·이관 상태 보존/오설정 거부) 통과. GPU6·62평가환경/31크기·31744reset에서 초기AT성공0·기울기초과0·RSI49.44%/후반요청69.97%·부분reset보존·원본bank SHA256동일, 실제6000회 actor/critic/AMP 및 정규화 재로드·12step 추론finite 확인(`output/at_tilt035_rsi_amp_check/audit.json`). 전용test스크립트의2환경·12step×2재로드평가rc0(`output/at_tilt035_eval_check/`). 새 학습은 실행하지 않았으며 장기수렴/새 조건의 성공률은 미측정이다.

## PUSH 문 없는 정책

현재 `open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors.yaml`은 `pushOnly: true`인 PUSH 전용 실행이다. 물리 장면에는 사람2명·상자2개만 생성하며 문 asset/actor/관절/충돌체가 없다. 정책은 상자30D·전체570D·6토큰을 유지한다. CLI의 `num_objects=4`는 기존 graph/RSI의 내부 슬롯 번호 호환용이며 실제 물리 상자는2개다. 내부 빈 슬롯은 simulator actor가 아니다.

학습 YAML에서 문 물성·각도·자동닫힘·hold/posture·문 스폰·OPEN 확률/보상/AMP mapping을 삭제했다. `dataset_push_only_push180.yaml`에는 loco4개와 push21개만 있으며 문 모션을 로드하지 않는다. `push_only_friction06_random_size_goal1to25_bank.json`은 기존 bank에서 PUSH 후보/물성 계약만 정확히 추출한 사본이며 원본 bank를 변경하지 않는다. 보상 계산은 `PushOnlyRuntime`에서 PUSH/AT만 실행한다. 목표1~2.5m·AT progress0.1m·RSI50%/후반요청70%·물성·AMP loco33%/push67%는 유지한다. AMP1310D의 고정 label2열은 기존 정책 호환을 위한 폭이며 상수 PUSH label만 사용한다. `openPush*`, `two_agent_open_push`, `stage1_open_push`라는 공통 식별자는 코드 호환용으로 남지만 OPEN 실행 설정을 포함하지 않는다.

```bash
# 기본은 scratch. 기존 학습과 분리한 새 output이며 GPU6·2048환경 유지
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_train.sh 2 2048 4
# 이 설정으로 저장한 checkpoint: 로컬 viewer / headless 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_test.sh /absolute/path/checkpoint.pth 2 1 4 100
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_test.sh /absolute/path/checkpoint.pth 2 64 4 8
TOKENHSI_GPU=6 PORT=6120 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_vnc.sh /absolute/path/checkpoint.pth 2 1 4 1000
```

기본 출력은 `output/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_physical_scratch/`이다. 기본 pretrained/전이 없음, `TRANSFER_CHECKPOINT` 거부. 실행 중인 기존 학습에는 소스 수정이 자동 반영되지 않으며 종료/재시작하지 않는다.

기존30D no_doors checkpoint는 문 설정을 포함한 옛 계약이므로 직접 로드하지 않는다. 필요할 때만 명시적으로 이관한 사본으로 재개한다. 이관 도구는 기존 PUSH/보상/물성·RSI후보·모션/네트워크 일치를 검사하고 model/optimizer/RMS/카운터를 그대로 보존한다. 원본을 덮어쓰지 않으며 문 포함 정책이나31D 정책은 거부한다. 이관은 기본 scratch 명령에 포함되지 않는다.

```bash
# 선택 사항: 기존30D 정책에서 이어 학습할 경우에만 실행
TOKENHSI_GPU=6 source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/convert_push_only_checkpoint.py /absolute/path/old_30d_checkpoint.pth output/push_only_physical_migration/resume.pth
# 총 학습 반복 상한은 checkpoint epoch보다 크게 지정
TOKENHSI_GPU=6 RESUME_CHECKPOINT=output/push_only_physical_migration/resume.pth OUTPUT_PATH=output/push_only_physical_resume bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_train.sh 2 2048 4
```

검증: CPU192개 통과(PUSH 보상/성공 수치 완전 일치 포함), GPU6·62환경/31종크기에서 물리 문0·관절0·loco/push만 로드·RSI50.44%/후반71.94%·부분reset/AMP/12step 유한값 확인(`output/push_only_physical_sim_check/audit.json`). 별도 GPU6·2048환경 scratch 확인 학습 저장(`output/push_only_physical_train_check/`), 기존3000회 이관 사본4환경×2반복 평가 rc0(`output/push_only_physical_eval_check/`). 새 저장본2환경×2반복 재로드평가 및 VNC1환경12step rc0. 별도2048환경3000→3001회 resume 저장·이관전후 모델/optimizer/RMS/카운터일치·두확인학습148종scalar와저장tensor finite(`output/push_only_physical_train_check/saved_check.json`). 확인학습은모두종료했으며사용자지시로추가학습을실행하지않는다. 장기 학습 결과는 아직 측정하지 않았다.

## 현재 실험

| 번호 | Config | 역할 |
| ---: | --- | --- |
| ksh size RSI | [size_rsi.yaml](../ports/ksh_stage1/tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi.yaml) | 크기별 PD RSI·별도 actor/critic semantic MLP |
| ksh typed bias | [typed_bias.yaml](../ports/ksh_stage1/tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias.yaml) | size RSI + typed attention bias |
| ksh typed message | [typed_bias_message.yaml](../ports/ksh_stage1/tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message.yaml) | typed bias + typed value message |
| 1 | [amp_humanoid_ma_carry.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml) | 원본 multi-agent carry 기준 |
| 27 | [approach_scenario_independent_with_climb.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_independent_with_climb.yaml) | 독립 물체·goal binding, 점 기준 성공 |
| 28 | [approach_scenario_stage1_plane.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_plane.yaml) | 27번의 ON_TOP·CLIMB을 상판 영역 성공으로 변경; Stage 2 기본 출발점 |
| 29 | [approach_stage2_coordination.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_coordination.yaml) | Stage 1 checkpoint에서 협력 graph 학습 |
| Stage 2 plane | [approach_stage2_coordination_sit_plane.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_coordination_sit_plane.yaml) | 34번 기반 SIT plane·보상·독립 과제 분포 |
| 30 | [approach_scenario_stage1_sit_plane_normalized.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_sit_plane_normalized.yaml) | 28번 + SIT 상판 성공·유효 edge 평균 보상 |
| 31 | [approach_scenario_stage1_skill_curriculum.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_skill_curriculum.yaml) | 30번 + 가까운 시작·후반 CLIMB RSI·어려운 과제 증량·보상 shaping |
| 32 | [approach_scenario_stage1_skill_curriculum_reward_preserved.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved.yaml) | 31번의 시작 상태·RSI·샘플링을 유지하고 state/progress는 30번 식 |
| 33 | [approach_scenario_stage1_at_goal_fix.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_at_goal_fix.yaml) | 32번 + AT 목표 slot reset 오류 수정 |
| 34 | [approach_scenario_stage1_self_sum.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_self_sum.yaml) | 33번 + 자기 edge 보상 합계·동료 edge 평균, 과제 비율 10/10/10/35/35 |
| 35 | [approach_scenario_stage1_slow_near_start.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_slow_near_start.yaml) | 34번 + 가까운 시작 80%를 60만 step 유지하고 120만 step까지 30%로 감소 |
| 36 | [approach_scenario_stage1_placement_focus.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_placement_focus.yaml) | 34번 + HOLDING+AT/HOLDING+ON_TOP만 각 50% 샘플링 |
| 37 | [approach_scenario_stage1_ontop_putdown.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_ontop_putdown.yaml) | 36번 + ON_TOP에도 충돌 없는 중간 구간의 `putDown` RSI 10% 적용 |
| Paired AT | [approach_scenario_stage1_paired_at_no_near.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_paired_at_no_near.yaml) | 2-agent/3-object, 매 scene AT+AT, 가까운 시작 없음 |
| Paired placement | [approach_scenario_stage1_paired_placement_no_near.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_paired_placement_no_near.yaml) | 2-agent/4-object, scene별 AT+AT 또는 ON_TOP+ON_TOP 각 50%, 가까운 시작 없음 |
| Paired owner HOLDING | [approach_scenario_stage1_paired_placement_owner_holding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_paired_placement_owner_holding.yaml) | Paired placement + AT/ON_TOP edge에 담당자의 선행 HOLDING `φ` 입력 |
| Unified semantic | [approach_scenario_stage1_unified.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified.yaml) | 2H/4O 독립 5과제, canonical binding·토큰 순열·과제 조건 AMP·unified RSI |
| Unified shared edge | [approach_scenario_stage1_unified_shared_edge_encoder.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder.yaml) | Unified semantic의 actor·critic semantic EdgeEncoder 파라미터 공유 |
| OPEN/PUSH unified | [open_push_unified_amp.yaml](../tokenhsi/data/cfg/multi_agent/open_push_unified_amp.yaml) | 사람별 OPEN/PUSH 50/50 공동 학습·문 지지 및 바닥 박스 밀기·과제 조건부 AMP |
| OPEN/PUSH PUSH 2m | [open_push_late_rsi_hinge_friction_push2m.yaml](../tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction_push2m.yaml) | 문 마찰0.5 late RSI + PUSH 목표2m·전용 bank/output |
| OPEN/PUSH PUSH180 21개 | [open_push_late_rsi_hinge_friction_push2m_push180.yaml](../tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction_push2m_push180.yaml) | PUSH2m·넓은 배치 + 앞 양손 밀기21개 AMP·전용 RSI bank/output |
| OPEN/PUSH PUSH180 구역 무작위 | [open_push_push180_random_lanes.yaml](../tokenhsi/data/cfg/multi_agent/open_push_push180_random_lanes.yaml) | PUSH180 유지·reset마다 담당 구역50/50 교환·기존 bank 재사용·별도 checkpoint 계약 |
| OPEN/PUSH PUSH180 실제 장면 무작위 | [open_push_push180_random_scene.yaml](../tokenhsi/data/cfg/multi_agent/open_push_push180_random_scene.yaml) | 고정1.3m 박스·실제XY/yaw/밀기방향/1~2m시작/물리binding 무작위·전용bank/output |
| OPEN/PUSH 접촉+AT | [open_push_push180_contact_at.yaml](../tokenhsi/data/cfg/multi_agent/open_push_push180_contact_at.yaml) | PUSH는 양손 접촉, AT는 기존 목표 근접·높이 판정; 별도 scratch/output |
| PUSH 자유 표면·state≥0.9 | [push_contact_at_friction06_free_surface_state09.yaml](../tokenhsi/data/cfg/multi_agent/push_contact_at_friction06_free_surface_state09.yaml) | 고정1.3m/20kg·어느 표면이든 양손접촉·raw state≥0.9+실제접촉 성공 |
| PUSH 고정크기·state≥0.9 | [push_contact_at_friction06_fixed_size_state09.yaml](../tokenhsi/data/cfg/multi_agent/push_contact_at_friction06_fixed_size_state09.yaml) | PUSH100%·1.3m/20kg·접촉/목표 raw state 평균≥0.9 성공·마찰0.6 |
| PUSH 크기 랜덤·목표1~2.5m·AT progress0.1m (전이 선택) | [open_push_contact_at_friction06_random_size_goal1to25_at01.yaml](../tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01.yaml) | goal25 10000 actor/AMP/RMS 전이·critic/optimizer 초기화·AT만0.1m포화 |
| PUSH 전용570D AT 기울기≤0.35rad | [open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_tilt035.yaml](../tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_tilt035.yaml) | 현재GPU6 PUSH전용기반·입력/AMP/RSI유지·AT성공만제한·570D사본resume·전용train/test/VNC |
| PUSH AT 기울기≤0.35rad | [open_push_contact_at_friction06_random_size_goal1to25_at01_tilt035.yaml](../tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_tilt035.yaml) | 문 포함 AT01·AT 성공 기울기만 제한·6000회 명시적 이관 resume·전용 train/test/VNC |
| PUSH 문 없는 정책 | [open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors.yaml](../tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors.yaml) | PUSH 전용·물리 문/문 설정/문 모션 삭제·물체30D/570차원·6토큰·scratch/별도 output |
| PUSH 크기 랜덤·목표1~2.5m (scratch 비교) | [open_push_contact_at_friction06_random_size_goal1to25.yaml](../tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25.yaml) | 중심거리1~2.5m 균등·360°·scratch·전용bank/output |
| PUSH 크기 랜덤·목표0.5~2.5m (기존 실행) | [open_push_contact_at_friction06_random_size_goal25.yaml](../tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal25.yaml) | 중심이동0.5~1m/1~2.5m 각50%·360°·scratch·전용bank/output |
| PUSH 크기 랜덤·마찰0.6 (기존 비교) | [open_push_contact_at_friction06_random_size.yaml](../tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size.yaml) | PUSH100%·1.1~1.4m/1cm·동일밀도12.1~25kg·크기별RSI |
| OPEN/PUSH 목표 반대면·마찰0.6 (고정크기 비교) | [open_push_contact_at_friction06_goal_face.yaml](../tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_goal_face.yaml) |20kg·바닥1·랜덤 시작/목표·목표 반대면 PUSH/RSI·29000actor/AMP 전이 |
| OPEN/PUSH 목표 반대면·마찰1 (비교용) | [open_push_contact_at_friction1_goal_face.yaml](../tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction1_goal_face.yaml) | 같은 기하/보상·20kg·별도 마찰1 bank/output |
| OPEN/PUSH 마찰1·랜덤목표 | [open_push_contact_at_friction1_random_goal.yaml](../tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction1_random_goal.yaml) | 접촉+AT·박스 마찰1·독립XY목표·29000 actor/AMP 전이·전용 RSI bank |
| OPEN/PUSH 접촉+AT 전이 | [open_push_push180_contact_at_warmstart.yaml](../tokenhsi/data/cfg/multi_agent/open_push_push180_contact_at_warmstart.yaml) | random-scene4500의 actor/AMP/입력RMS 전이·critic/optimizer/카운터 초기화 |
| PUSH 랜덤 goal 제로샷 평가 | [전용 test](../tokenhsi/scripts/multi_agent/open_push_zero_shot_random_goal_test.sh) | 기존 warmstart checkpoint·보행시작·평가 reset의 goal만 무작위화; 학습용 config 없음 |
| OPEN/PUSH hinge friction | [open_push_late_rsi_hinge_friction.yaml](../tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction.yaml) | late RSI revision5 + 문 힌지 마찰0.5·별도 bank/output |
| OPEN/PUSH late RSI | [open_push_late_rsi.yaml](../tokenhsi/data/cfg/multi_agent/open_push_late_rsi.yaml) | 공동 과제·보행/조작50/50·PD검사 후보의 뒤쪽30%를70% 확률로 요청 |
| Door AMP OFF | [door_open_amp_off.yaml](../tokenhsi/data/cfg/multi_agent/door_open_amp_off.yaml) | 2H/2Door 독립 OPEN, shared EdgeEncoder·기존 MAAgent PPO, AMP 비활성 |
| Door AMP ON | [door_open_amp_on.yaml](../tokenhsi/data/cfg/multi_agent/door_open_amp_on.yaml) | 같은 OPEN/PPO·reward에 기존 비조건부 AMP 추가 |
| Door pull approach AMP | [door_pull_unified_approach_amp.yaml](../tokenhsi/data/cfg/multi_agent/door_pull_unified_approach_amp.yaml) | 사람 쪽으로 열리는 −Z hinge·당김 접촉 공간, 별도 checkpoint/output |
| Door loco AMP | [door_open_unified_approach_amp.yaml](../tokenhsi/data/cfg/multi_agent/door_open_unified_approach_amp.yaml) | 기존 unified·몸/손 계수0.2·전 구간 loco AMP |
| Door BONES left AMP | [door_open_bones_left_amp.yaml](../tokenhsi/data/cfg/multi_agent/door_open_bones_left_amp.yaml) | 왼손·왼쪽 개방 human reference + loco30/door70 full AMP·loco-only reset |
| Door open/hold AMP | [door_open_hold_amp.yaml](../tokenhsi/data/cfg/multi_agent/door_open_hold_amp.yaml) | 사람1·문1, 밀어 열고 손으로5초 지지·놓으면 자동 닫힘 |
| Door shared loco AMP | [door_open_shared_edge_approach_amp.yaml](../tokenhsi/data/cfg/multi_agent/door_open_shared_edge_approach_amp.yaml) | 같은 몸/손0.2·전 구간 loco AMP + Actor/Critic EdgeEncoder 공유 |
| Door unified AMP ON | [door_open_unified_amp_on.yaml](../tokenhsi/data/cfg/multi_agent/door_open_unified_amp_on.yaml) | 기존 unified의 Actor/Critic별 EdgeEncoder + Door proxy AMP/RSI |
| Door shared-edge AMP ON | [door_open_shared_edge_amp_on.yaml](../tokenhsi/data/cfg/multi_agent/door_open_shared_edge_amp_on.yaml) | 같은 Door proxy AMP/RSI + Actor/Critic EdgeEncoder 공유 |
| Shared random binding | [approach_scenario_stage1_unified_shared_random_binding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_shared_random_binding.yaml) | 사용자 12옵션 조합: 무작위 binding·큰 상자·34번 near 시작·비조건부 AMP |
| Unified owner HOLDING | [approach_scenario_stage1_unified_owner_holding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_owner_holding.yaml) | Unified semantic + AT/ON_TOP 담당 HOLDING φ 입력 |
| Hybrid late RSI | [approach_scenario_stage1_hybrid_late_rsi.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_hybrid_late_rsi.yaml) | student 기반, 같은 XYZ의 후반 PD/현재 정책 후보 선택·ON_TOP 받침별 putDown |
| Object FT Joint Adapter | [object_tokenizer_finetune_joint_adapter.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/multi_agent/object_tokenizer_finetune_joint_adapter.yaml) | SIT/CARRY 공동·Obj+층별 잔차 adapter 학습·기존 head 동결 |
| Object FT Joint Head | [object_tokenizer_finetune_joint_head.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/multi_agent/object_tokenizer_finetune_joint_head.yaml) | 같은 공동 데이터·Obj+기존 head 학습·adapter 없음 |
| Object FT Carry Chair Adapter | [object_tokenizer_finetune_carry_chair_adapter.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/multi_agent/object_tokenizer_finetune_carry_chair_adapter.yaml) | ArmChair63종 단독 HOLDING_AT·Obj+층별 adapter 학습·head 동결·GPU4 |
| Object FT Carry Chair Copied Adapter | [object_tokenizer_finetune_carry_chair_copied_adapter.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/multi_agent/object_tokenizer_finetune_carry_chair_copied_adapter.yaml) | 기존 Obj 보존·복사본+adapter 학습·base/adapt 분기·GPU4 |
| Object FT SIT | [object_tokenizer_finetune_sit_chair.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/multi_agent/object_tokenizer_finetune_sit_chair.yaml) | StraightChair 전용·기존30D tokenizer·좌판 RSI50/50·독립 사본 |
| Object FT Chair | [object_tokenizer_finetune_carry_chair.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/multi_agent/object_tokenizer_finetune_carry_chair.yaml) | ArmChair 운반 전용·TokenHSI식 loco 초기화·critic/AMP 학습 |
| Object FT Table | [object_tokenizer_finetune_carry_table.yaml](../ports/ksh_object_finetune/tokenhsi/data/cfg/multi_agent/object_tokenizer_finetune_carry_table.yaml) | Table_Circle 운반 전용·위와 별도 checkpoint/output |

27~37번 Stage 1은 2-agent/3-object 독립 graph와 schema 9 semantic 관찰을 쓴다. ON_TOP 두 개를 동시에 샘플하지 않는다. 28~37번의 팀 보상은 자기 0.9, 동료 0.1이다. 30~33번은 유효 edge를 agent별로 평균 내어 단일 edge와 두 edge의 최대 task 보상을 0.6으로 맞춘다. 34~37번은 자기 edge를 합산하고 동료 edge만 평균 내며 state·progress·성공 식은 33번과 같다.

Paired AT/paired placement도 schema 9 semantic 관찰과 34번의 reward·RSI·AMP를 유지한다. 두 agent의 **과제 relation만 같은 타입**으로 샘플하고 실제 motor action은 각자 선택한다. Paired placement는 두 ON_TOP에 서로 다른 받침을 주기 위해 4개 물체를 쓴다. 두 기준 실험은 가까운 시작 확률이 처음부터 끝까지 0이며, 각자 별도 checkpoint/output에서 scratch로 시작한다.

Paired owner HOLDING은 paired placement의 샘플링·reward·RSI·AMP·가까운 시작 없음 설정을 유지한다. 관측만 edge당 5개에서 6개 필드로 늘리고 AT/ON_TOP에는 같은 owner의 `H→source O` HOLDING `φ`, 나머지 유효 edge에는 1을 기록한다. actor·critic의 기존 semantic 64차원 뒤에서 AT/ON_TOP에만 상태 잔차를 더한다. 기존 paired checkpoint와 관측·네트워크 형식이 달라 별도 output에서 scratch로 학습한다.

2026-09-30 공통 코드 수정 이후 scenario의 상자 속도 패널티는 현재 graph에 연결된 담당 물체를 따른다. 이전에는 legacy agent assignment를 사용해 랜덤 binding에서 다른 물체의 패널티가 들어갈 수 있었다. 모델/관측 shape는 같아서 기존 checkpoint를 로드할 수 있지만, 수정 전후 학습의 penalty 의미는 다르다. 이미 실행 중인 프로세스에는 자동 적용되지 않으며 이후 새 프로세스부터 적용된다.

33번은 32번의 보상·샘플링·RSI를 유지한다. AT graph가 두 번째 목표 slot을 가리킬 때 reset도 해당 목표를 배치하도록 공통 코드의 slot 선택을 수정했다. 수정 후 새로 시작하는 27~32번 실행에도 적용되지만, 이미 실행 중인 프로세스와 checkpoint는 바뀌지 않는다. 비교 실험은 33번 전용 output에 scratch로 시작한다.

## OPEN/PUSH Unified — 두 과제 공동 학습

### PUSH 전용·자유 표면 양손 접촉·고정크기·state ≥ 0.9

`push_contact_at_friction06_free_surface_state09.yaml`은 위 랜덤크기 PUSH 기준에서 박스를1.3m/20kg으로 고정하고 `relationReward.push.contact_mode: any_surface`를 사용한다. 랜덤 시작/목표/binding·마찰0.6/바닥1·PUSH100%·AMP보행33/PUSH67·RSI비율·구역간격6.3m는 유지한다.

손마다 **회전된 박스 전체 표면의 최근접점까지 XYZ거리**를 계산한다. 앞/뒤/옆/위/아래 어느 면과 위치든 허용하며 양손이 같은 면일 필요도 없다. 내부점을 표면거리0으로 처리하지 않는다. 접촉 φ=`exp(-10*max(d_right,d_left)²)`, 목표 φ=`exp(-10*||box_center-goal||²)`이며 **raw state=(접촉φ+목표φ)/2 ≥0.9 AND 실제양손접촉**일 때 개인성공이다. 각 손의 표면거리≤0.18m와 표면밖방향 반력≥0.5N을 같은 물리 tick에서 요구한다. 손 반력은 선택된 면/목표방향으로 제한하지 않으며 상쇄될 수 있는 박스 net force는 접촉 gate에 사용하지 않는다. 목표에 닿아도 손을 떼거나 한손접촉만 있으면 성공하지 않는다. 성공 판정은 saturation/가중치 적용 전에 수행한다.

정책 보상/성공의 손 위치 지정은 제거하지만 **RSI 초기자세와 AMP PUSH 참조는 기존 밀기 자세**를 유지한다. `pushGoalContact: true`는 RSI 정렬과 기존 방향 진단용으로 남으며 자유 표면 보상/접촉 검사를 제한하지 않는다. 기존 고정1.3m 물리 bank를 재사용하고 새 보상 옵션은 checkpoint 보상 metadata에 포함한다. 기존 저장본 직접resume/test는 거부하고 원본29000 actor/AMP/입력RMS 전이·critic/optimizer/카운터 초기화로 시작한다.

```bash
# 본학습·2048환경 (현재 bank 준비됨)
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/push_contact_at_friction06_free_surface_state09_train.sh 2 2048 4
# 짧은학습 확인·별도output
TOKENHSI_GPU=6 SEED=42 MAX_ITERATIONS=1 OUTPUT_PATH=output/push_free_surface_state09_train_check bash tokenhsi/scripts/multi_agent/push_contact_at_friction06_free_surface_state09_train.sh 2 2048 4
# 로컬 viewer / 화면없는평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_contact_at_friction06_free_surface_state09_test.sh <new_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_contact_at_friction06_free_surface_state09_test.sh <new_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 PORT=6099 bash tokenhsi/scripts/multi_agent/push_contact_at_friction06_free_surface_state09_vnc.sh <new_checkpoint.pth> 2 1 4 100
# 후보 재생성 없이 기존 bank의 reset/AMP 호환 검사
TOKENHSI_GPU=6 RSI_REUSE_BANK=1 bash tokenhsi/scripts/multi_agent/build_push_free_surface_state09_rsi.sh
# bank가 없을 때만 전체 PD 후보 생성
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_push_free_surface_state09_rsi.sh
```

검증 결과는 [changelog](../changelog.md)의 최신 항목 참조. 장기 본학습/정책 성공률은 미검증이다.

### PUSH 전용·고정 1.3m·state ≥ 0.9 성공

`push_contact_at_friction06_fixed_size_state09.yaml`은 크기 랜덤 PUSH 실험에서 `pushSizeRandomization`을 제거한 별도 비교다. 박스1.3m/20kg·마찰0.6/바닥1, PUSH100%·AMP 보행33/PUSH67, 구역간격6.3m·랜덤 시작/목표/담당박스·RSI50%/후반70%는 유지한다.

`contact_at.success_mode: state_only`로 가중치와 성공 saturation 적용 전의 **state_score = (PUSH 접촉 φ + AT 목표 φ)/2 ≥ 0.9**이면 개인성공이다. PUSH φ는 양손 중 더 먼 손의 목표반대면 XY거리로 `exp(-10*d_hand²)`, AT φ는 박스-목표 XYZ거리로 `exp(-10*||delta||²)`다. 로그의 가중 state 보상(최대0.4)에0.9를 직접 비교하지 않는다. 기존의 별도 높이1mm/힘접촉 조건은 성공에 요구하지 않으며 Z오차는 AT φ에 포함한다. 목표에서 손을 멀리 놓으면 raw state가 낮아져 성공이 해제된다. 판정 후 기존 saturation/state/progress/bonus 계산을 사용한다. 옵션이 없는 실험의 성공식은 유지한다.

원본29000 actor/AMP/입력RMS 전이·critic/optimizer/카운터 초기화, 새 RSI bank와 checkpoint 보상계약/output을 사용한다. 기존 랜덤크기/고정크기 저장본의 직접resume/test는 계약 불일치로 거부한다.

```bash
# 전용 고정크기 RSI bank 생성 (학습없음)
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_push_fixed_size_state09_rsi.sh
# 본학습·2048환경
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/push_contact_at_friction06_fixed_size_state09_train.sh 2 2048 4
# 짧은검증·별도output
TOKENHSI_GPU=6 SEED=42 MAX_ITERATIONS=1 OUTPUT_PATH=output/push_fixed_size_state09_train_check bash tokenhsi/scripts/multi_agent/push_contact_at_friction06_fixed_size_state09_train.sh 2 2048 4
# 로컬 viewer / 화면없는평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/push_contact_at_friction06_fixed_size_state09_test.sh <new_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/push_contact_at_friction06_fixed_size_state09_test.sh <new_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 PORT=6099 bash tokenhsi/scripts/multi_agent/push_contact_at_friction06_fixed_size_state09_vnc.sh <new_checkpoint.pth> 2 1 4 100
```

검증 결과는 [changelog](../changelog.md)의 최신 항목 참조. 장기 본학습/정책 성공률은 미검증이다.

### PUSH 손 접촉 중 방향 독립 이동량 측정

`measure_push_hand_motion.sh`는 checkpoint를 원래 계약으로 로드하고 학습RSI혼합/보행시작을 각각평가한다. 물리tick마다 박스전체표면거리≤0.18m·표면밖 손반력≥0.5N의한손/양손접촉을검사하며목표방향은요구하지않는다. 접촉중박스중심XY누적이동거리/직립접촉이동거리와episode전체순변위를분리한다. 누적거리는왕복/흔들림을포함하고손접촉과동시이동이손의단독인과를증명하지않는다. 실행중학습로그가아닌별도정책평가다. checkpoint기본계약은기존goal25이며다른실험은일치하는 `CFG_ENV`를명시한다.

```bash
TOKENHSI_GPU=6 SEED=42 OUTPUT_PATH=output/push_hand_motion_9500 bash tokenhsi/scripts/multi_agent/measure_push_hand_motion.sh output/open_push_contact_at_friction06_random_size_goal25/OpenPushContactAtFriction06RandomSizeGoal25_13-16-10-31/nn/OpenPushContactAtFriction06RandomSizeGoal25_00009500.pth 64 2
# 짧은측정기확인: MOTION_EPISODE_LENGTH=12, 별도OUTPUT_PATH, 2환경/1회
```

출력은 `audit.json`(집계/metadata), `trials.json`(사람episode별수치)이며평가환경수/반복은checkpoint뒤두인자다. 기본600step상한·seed42. 현재정책밀기분석에는원래600step을사용한다.

### PUSH 목표1~2.5m·AT progress0.1m·10000 정책 전이 — 현재 선택

`open_push_contact_at_friction06_random_size_goal1to25_at01.yaml`은 기존goal25의10000저장본에서actor/행동분산·AMP discriminator·관측/AMP입력정규화를전이하고critic/value head·optimizer·보상정규화·epoch/frame·AMP replay를초기화한다. 목표는1~2.5m거리균등/360°이며 `contact_at.at_progress_delta: 0.1`로박스→목표progress만0.1m에서포화한다. 사람→박스progress는기존 `progress_delta: 0.5`를유지한다. 옵션없는기존실험의보상은변하지않는다. 성공XYZφ≥0.9/Z오차1mm·release·크기/물성·RSI비율/AMP는유지한다.

전이는기존goal25→새목표분포/AT포화/대응bank만허용하고source snapshot·보상metadata·물리/network계약을검사한다. `TRANSFER_CHECKPOINT`로같은goal25계약의다른epoch를선택할수있으며 `RESUME_CHECKPOINT`와동시사용은거부한다. 후속resume/test는새실험저장본을사용한다. 기본전이source는goal25의 `OpenPushContactAtFriction06RandomSizeGoal25_00010000.pth`다.

RSI물리계약은contact_at보상을제외하므로검증된1~2.5m bank를재사용한다. 전용RSI wrapper는기본 `RSI_REUSE_BANK=1`로reset/AMP/초기AT성공을감사하고, bank재생성은 `RSI_REUSE_BANK=0`이다. RSI최대85%선행이동에서도남은거리최소0.15m로성공반경보다크다. 구체적검증은changelog의2026-10-14 항목을참조한다.

```bash
# 준비된bank 재사용 검사 (완료)
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_open_push_friction06_random_size_goal1to25_at01_rsi.sh
# 본학습·2048환경·기본source10000·별도output
TOKENHSI_GPU=6 SEED=42 MAX_ITERATIONS= RESUME_CHECKPOINT= TRANSFER_CHECKPOINT= OUTPUT_PATH=output/open_push_contact_at_friction06_random_size_goal1to25_at01 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_train.sh 2 2048 4
# 짧은전이확인 (본학습과분리)
TOKENHSI_GPU=6 SEED=42 RESUME_CHECKPOINT= TRANSFER_CHECKPOINT= MAX_ITERATIONS=1 OUTPUT_PATH=output/push_goal1to25_at01_transfer_check bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_train.sh 2 2048 4
# 새실험checkpoint 로컬 viewer / 보행시작 headless / 서버VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_test.sh <new_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 SEED=42 OUTPUT_PATH=output/push_goal1to25_at01_loco_eval bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_test.sh <new_checkpoint.pth> 2 64 4 8
TOKENHSI_GPU=6 PORT=6099 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_vnc.sh <new_checkpoint.pth> 2 1 4 100
```

기본학습output은 `output/open_push_contact_at_friction06_random_size_goal1to25_at01`이다. 위명령에서쉘의 `OUTPUT_PATH`가남아있다면비우거나이경로를명시한다. 장기본학습은준비작업에서실행하지않는다.

### GPU4 PUSH8500 크기·무게별 비학습 평가

[결과·해석](../output/push_gpu4_size_mass_audit_20261015/REPORT.md), [크기별 그림](../output/push_gpu4_size_mass_audit_20261015/size_mass_comparison.png). 원래 문 포함 AT01 YAML·8500 checkpoint·정규화를 재사용하며 학습/물성/보상은 변경하지 않는다. GPU1·248평가환경·31크기 균등·각 크기/시작조건128 agent episode·600step·seed8429다. 학습 RSI 혼합과 보행 시작을 분리하고 AT 도달/수직 AT 도달·yaw·기울기·좌우 이탈·목표 방향 전진을 집계한다. 두 agent가 같은 장면 종료를 공유하므로 통계는 장면 단위로 처리한다. 크기와 질량이 함께 변하는 평가이며 무게만의 인과효과를 측정한 것은 아니다.

```bash
# 비학습 headless 평가만 실행; 기존 결과를 보존할 별도 output 지정
TOKENHSI_GPU=1 OUTPUT_PATH=output/push_gpu4_size_mass_audit_repeat bash output/push_gpu4_size_mass_audit_20261015/run.sh
```

새 실행용 YAML/학습 실험이 아니며 viewer를 실행하지 않는다. 기존 학습은 중단/재시작하지 않는다.

### PUSH 전용·박스 크기 랜덤·목표1~2.5m — scratch 비교

`open_push_contact_at_friction06_random_size_goal1to25.yaml`은 초기 박스 중심→목표 중심 거리를 **1~2.5m 거리 균등**, 방향을360°균등으로 선택한다(면적균등 아님). 기존0.5~1m/1~2.5m 50/50 실행과 config·bank·output을 분리했다. 크기/물성·보상/성공·AMP·RSI50%/후반요청70%·7m 작업구역은 유지한다. scratch이며 기존checkpoint의 직접resume/test는 목표/bank 계약이 달라 거부된다. 후속resume은 새 실험 저장본만 사용한다.

RSI는 원래 목표 거리의 최대85%를 선행 이동하므로 **reset 후 남은 거리는 최소0.15m**다. AT 성공반경 약0.103m보다 크며 전용bank 검증에서 실제 초기AT 성공 건수를 확인한다. 1~2.5m는 RSI 이전 거리이며 보행평가에는 조작RSI가 없다. AT progress의0.5m 포화식은 유지한다.

```bash
# 전용PD RSI bank 생성·검증
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_open_push_friction06_random_size_goal1to25_rsi.sh
# scratch 본학습·2048환경
TOKENHSI_GPU=6 SEED=42 RESUME_CHECKPOINT= TRANSFER_CHECKPOINT= bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_train.sh 2 2048 4
# 별도출력 짧은 확인
TOKENHSI_GPU=6 SEED=42 RESUME_CHECKPOINT= TRANSFER_CHECKPOINT= MAX_ITERATIONS=1 OUTPUT_PATH=output/push_random_size_goal1to25_train_check bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_train.sh 2 2048 4
# 로컬 viewer / 화면 없는 보행 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_test.sh <new_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 SEED=42 OUTPUT_PATH=output/push_random_size_goal1to25_loco_eval bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_test.sh <new_checkpoint.pth> 2 64 4 8
TOKENHSI_GPU=6 PORT=6099 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_vnc.sh <new_checkpoint.pth> 2 1 4 100
```

검증 결과는 [changelog](../changelog.md)의2026-10-14 항목을 참조한다. 장기 본학습은 이 변경 작업에서 시작하지 않는다.

### PUSH 전용·박스 크기 랜덤·목표0.5~2.5m — 기존 실행

`open_push_contact_at_friction06_random_size_goal25.yaml`은 기존 랜덤크기 PUSH의 목표 분포를 실제 박스 중심 간 거리로 지정한다. **0.5~1m를50%, 1~2.5m를50%** 확률로 선택하고, 선택한 구간 안에서 거리를 균등하게 뽑는다(면적균등 아님). 방향은360°균등이다. 목표Z는 각 박스 실제높이÷2, 목표는 episode 동안 고정이며 보행사람 초기중심에서1m 배제한다.

최대거리2.5m의 이동영역을 반영해 두 작업구역 최소간격을7m로 늘렸고, 같은arena에서 큰간격의배치가실패하지않도록 배치 rejection 상한을512회로 높였다. 크기1.1~1.4m/동일밀도·물성·보행거리·PUSH100%·AMP·접촉+AT/release·AT의XYZφ≥0.9 및Z오차1mm 성공식은 기존실험과 같다. RSI 비율/후반요청도 같고, 새목표방향/거리로 전용bank를 검사한다. 50/50 목표분포는RSI 선행이동 전 초기박스 기준이며, 조작RSI는 그거리의일부를 이미진행한상태에서시작한다. 평가`--eval`은조작RSI없이보행시작이다.

사용자 요청으로 **scratch 학습**한다. actor/critic/AMP와정규화통계·optimizer·카운터를새로초기화하며config에전이source가없고전용train은`TRANSFER_CHECKPOINT`를거부한다. RSI bank는신경망가중치가아닌참조초기자세후보이므로scratch에서도사용한다. 기존10500의직접resume/test는목표/scene/bank계약불일치로거부하며, 후속resume은새goal25저장본만사용한다. 새학습시셸의`RESUME_CHECKPOINT`를비워야한다.

```bash
# 전용PD RSI bank 준비 (학습없음; 최초1회)
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_open_push_friction06_random_size_goal25_rsi.sh
# scratch 본학습·2048환경
TOKENHSI_GPU=6 SEED=42 RESUME_CHECKPOINT= TRANSFER_CHECKPOINT= bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal25_train.sh 2 2048 4
# 별도output의짧은검증
TOKENHSI_GPU=6 SEED=42 RESUME_CHECKPOINT= TRANSFER_CHECKPOINT= MAX_ITERATIONS=1 OUTPUT_PATH=output/push_random_size_goal25_train_check bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal25_train.sh 2 2048 4
# 새goal25 checkpoint의로컬viewer / RSI없는headless평가 / 서버VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal25_test.sh <new_goal25_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 SEED=42 OUTPUT_PATH=output/push_random_size_goal25_loco_eval bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal25_test.sh <new_goal25_checkpoint.pth> 2 64 4 8
TOKENHSI_GPU=6 PORT=6099 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal25_vnc.sh <new_goal25_checkpoint.pth> 2 1 4 100
```

검증 결과는 [changelog](../changelog.md)의 최신 항목을 참조한다. 본학습은 실행하지 않았다.

### PUSH 전용·박스 크기 랜덤·동일 밀도 — 기존 비교

`open_push_contact_at_friction06_random_size.yaml`은 두 에이전트 모두 PUSH만 학습한다(OPEN0/PUSH1). AMP는 보행33%·PUSH67%이며 OPEN 참조 샘플링은0이다. 원본29000actor/AMP/입력RMS를전이하고 critic/optimizer/카운터를초기화한다. 관측/모델 호환을 위해 문 actor/OPEN 토큰과 파일 loader는 유지하지만 OPEN 학습/RSI 선택은 하지 않는다.

박스는 정육면체1.1~1.4m를 **1cm 간격31종에서균등샘플링**한다. 각 물리박스는 시뮬레이션 생성 때 크기를 정하고 해당환경의reset동안 유지한다(episode마다 크기를바꾸는 방식은아님). 생성시각크기의asset에 같은밀도20/1.3³≈9.1033kg/m³를적용하여 질량12.1165~24.9795kg·관성을함께설정한다. 박스마찰0.6·바닥1은유지한다. 박스마다 크기/무게는다르며20kg은1.3m기준값이다.

실제크기를BPS/크기관측·담당박스mapping·목표반대면/힘법선·박스초기높이/목표Z·보행침투검사·RSI손정렬·바닥corner진행검사에연결한다. RSI손높이는지면0.8m,후보는초기양손목표오차18cm미만과31종크기×독립2장면의PD12step를모두통과해야한다. 공통후보bank를전용으로저장하며 최소각PUSHclip의생존후보를요구한다. 크기범위를반영해PUSH작업구역최소간격을6.3m로늘렸다. 시작XY/yaw/heading/담당박스·목표XY랜덤과독립접촉+AT/release는유지한다. 크기/과제비율은checkpoint계약에포함하며이전저장본직접resume은거부한다.

```bash
# GPU6·31종크기전체RSI 검사 (학습없음)
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_open_push_friction06_random_size_rsi.sh
# PUSH만 본학습·2048환경
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_train.sh 2 2048 4
# 별도output의짧은검증
TOKENHSI_GPU=6 SEED=42 MAX_ITERATIONS=1 OUTPUT_PATH=output/push_random_size_train_check bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_train.sh 2 2048 4
# 로컬viewer / 화면없는평가 / 서버VNC: 새실험저장본
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_test.sh <new_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_test.sh <new_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 PORT=6099 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_vnc.sh <new_checkpoint.pth> 2 1 4 100
```

검증: CPU118개 통과. GPU6·31종×2장면/후보의 초기양손오차+PD12step 검사에서PUSH943/1744 후보가통과했고21clip모두생존후보를갖는다(`output/open_push_friction06_random_size_rsi_check/`). 62환경/124실제박스의31종크기·질량12.11652~24.97952kg·마찰0.6·크기BPS·PUSH만의과제/AMP·랜덤시작/목표/네면·부분reset·보행/RSI12step을확인했고초기양손오차최대0.16965m였다(`output/push_random_size_sim_check/audit.json`). 2048환경최종bank의29000source actor/AMP88텐서전이·critic81초기화·epoch2/frame262144 저장·모델/161종scalar유한값(`output/push_random_size_train_check/`),2환경30step재로드평가종료코드0(`output/push_random_size_eval_check/`)를확인했다. 초기손오차필터보완전검사본은`output/push_random_size_train_check_before_initial_RSI_filter/`에구분보존했다. 장기본학습/성공률은미검증이며 본학습은실행하지않았다.

### PUSH 목표 반대면·박스20kg·마찰0.6 — 고정크기 비교

`open_push_contact_at_friction06_goal_face.yaml`은 고정크기 비교 실행설정이다. **박스20kg·박스마찰0.6·바닥정지/동마찰1**이다. 비교용 `open_push_contact_at_friction1_goal_face.yaml`은 박스마찰1이며별도bank/output을쓴다. 과거 `open_push_contact_at_friction1_random_goal.yaml`과원본실험은보존한다. 과거랜덤목표설정은접촉이박스local-Y고정뒷면이어서 랜덤목표와접촉면이일치하지않을수있었고 이번설정에서이를연결했다.

- PUSH박스XY/yaw·사람1~2m거리/임의방향시작·사람heading·중복없는담당박스는reset마다랜덤이다. 사람보행시작을항상목표반대면에배치하는것은아니며,접촉/보상이반대면으로이동하도록안내한다.
- 목표는RSI선행이동전박스중심의XY±1.5m사각형에서샘플하고 박스/보행사람과1m이상떨어뜨린다. 목표Z는중심0.65m,episode동안고정한다. 크기1.3m·작업영역최소간격6.2m는유지한다.
- `pushGoalContact: true`일때목표를박스yaw좌표로변환하고, 조작RSI의양손목표는목표방향과가장잘맞는내향법선을가진옆면의반대쪽에배치한다. ±X/±Y네면을모두사용한다. 접촉/보상은현재목표방향과60°이내로밀수있는반대면중양손이같이면에가장가까운것을선택한다. 대각선에서두인접반대면을허용해45°경계의90°접촉면전환을완화하고 목표쪽앞면/직교면은제외한다. 손힘/박스힘검사도그면의내향법선을따른다.
- RSI박스선행이동은실제고정목표변위를따르고 사람/양손은그목표반대면에맞춘다. 접촉은기존처럼손높이자유이고 `hand_target_height:0.8`은RSI자세설정이다. AT목표근접/높이성공과release허용은유지한다. 네트워크/관측shape는같고기존관측의박스pose·goal·손pose로면을판단한다.
- `push_contact:goal_opposite_faces_v2`를checkpoint와RSI bank계약에포함해이전고정뒷면/초기dominant-only면checkpoint 직접resume/test를거부한다. 기본원본29000source의actor/AMP/입력RMS를전이하고 critic/optimizer/카운터를초기화한다. 후속resume은새실험저장본만쓴다. 초기dominant-only v1검사는별도보존했고최종v2검증과구분한다.

```bash
# GPU6·현재마찰0.6 전용 RSI PD12step검사
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_open_push_friction06_goal_face_rsi.sh
# 본학습 (2048환경, 원본29000actor/AMP warm start)
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_goal_face_train.sh 2 2048 4
# 확인용 짧은학습은 별도output
TOKENHSI_GPU=6 SEED=42 MAX_ITERATIONS=1 OUTPUT_PATH=output/open_push_goal_face06_train_check bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_goal_face_train.sh 2 2048 4
# 로컬 viewer / 화면없는평가 / 서버VNC: 새실험checkpoint
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_goal_face_test.sh <new_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=random_open_push bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_goal_face_test.sh <new_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 PORT=6099 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_goal_face_vnc.sh <new_checkpoint.pth> 2 1 4 100
# 비교마찰1의 bank/학습/로컬viewer/서버VNC는별도wrapper/output
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_open_push_friction1_goal_face_rsi.sh
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction1_goal_face_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction1_goal_face_test.sh <friction1_new_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction1_goal_face_test.sh <friction1_new_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 PORT=6099 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction1_goal_face_vnc.sh <friction1_new_checkpoint.pth> 2 1 4 100
```

검증: 관련 CPU102개 통과. GPU6에서 생성한 v2 RSI는 OPEN6/13·PUSH1045/1744 통과(비교 마찰1 PUSH1204/1744). `output/open_push_goal_face06_sim_check/audit.json`에서 실제20kg/0.6·200reset 랜덤목표/배정·목표 반대 네면·부분reset·네task 보행/RSI12step을 확인했고 RSI양손오차최대0.13863m였다. `output/open_push_goal_face06_train_check/`에서2048환경·원본29000actor/AMP88텐서전이·critic81텐서초기화·epoch2/frame262144 저장·모델/161종scalar유한값을 확인했다. `output/open_push_goal_face06_eval_check/`의2환경30step 재로드평가는 종료코드0이다. 짧은검사는 실행/호환성 확인이며 성공률이나 최대밀기질량 검증이 아니다. 장기 본학습은 실행하지 않았다.

바닥1과박스0.6의평균접촉마찰0.8에서20kg의수평미끄러짐필요힘은약157N이다. 앞선물리상한검사의약62.9kg은무수직손힘/무충돌/무외부지지의준정적발마찰필요조건상한이며 실제에이전트가그질량을밀었다는검증이아니다. 기존학습정책의관측상한과도구분한다.

### PUSH 마찰1·ksh식 랜덤 XY 목표

`open_push_contact_at_friction1_random_goal.yaml`은 중지한 접촉+AT warm start의 별도 후속 실험이다. PUSH 박스 마찰만1.0으로 변경하고, 목표는 초기 박스중심 기준 XY±1.5m 사각형에서 균등 샘플링해 박스와1m 미만이면 재추출한다. 보행 사람도 목표에서1m 이상 떨어뜨린다. 목표Z는 박스중심0.65m이며 에피소드 도중 고정한다. ksh의 사각형 샘플링/근접 재추출을 PUSH 작업영역에 적용한 것으로, ksh의 사람별 고정중심이나 참조모션 종점 목표를 그대로 복제하지 않는다. 박스yaw와 목표방향은 독립이다. 영역분리를 위해 두 박스 최소 간격은6.2m다.

OPEN·접촉+AT 보상·AMP·보행/조작50/50·후반RSI70%는 유지한다. RSI 박스 선행이동은 랜덤목표까지의 변위를 따른다. 새물성/목표 계약의 전용 bank를 PD12step으로 다시 생성하며 이전 bank는 재사용하지 않는다. 기본 source는 중지한 학습의29000 checkpoint다. actor/AMP/입력RMS를 가져오고 critic/optimizer/보상RMS/카운터/replay는 초기화한다. source 저장설정과 계약을 검사하고 마찰/목표/장면간격/bank만 변경을 허용한다. 기존checkpoint 직접resume/test는 불가하며 이후 새설정 저장본으로resume한다.

```bash
# 비학습 RSI bank 생성 (실행 전 GPU 점유 확인)
TOKENHSI_GPU=1 bash tokenhsi/scripts/multi_agent/build_open_push_friction1_random_goal_rsi.sh
# 별도 본학습 (2048환경)
TOKENHSI_GPU=1 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction1_random_goal_train.sh 2 2048 4
# 짧은 확인용 별도 output
TOKENHSI_GPU=1 SEED=42 OUTPUT_PATH=output/open_push_friction1_random_goal_train_check MAX_ITERATIONS=1 bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction1_random_goal_train.sh 2 2048 4
# 새설정 checkpoint로 로컬 viewer / headless 평가 / 서버 VNC
TOKENHSI_GPU=1 HEADLESS=0 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction1_random_goal_test.sh <new_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=1 HEADLESS=1 TASK_GRAPH=random_open_push bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction1_random_goal_test.sh <new_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=1 PORT=6099 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_contact_at_friction1_random_goal_vnc.sh <new_checkpoint.pth> 2 1 4 100
```

검증: CPU85개·Python/셸문법·diff 통과. GPU1 새물성PD screen은 OPEN6/13·PUSH1201/1744 후보 통과(21개 PUSH clip 모두 생존),1024 reset 조작RSI50.49%·후반요청70.99%·AMP/부분reset 통과(`output/open_push_friction1_random_goal_rsi_check/audit.json`).2환경200reset·독립목표방향·부분reset·네task 보행16/RSI32reset×12step 통과,RSI양손오차 최대0.16024m(`output/open_push_friction1_random_goal_sim_check/audit.json`);기존2m장면 회귀도 통과(`output/open_push_friction1_goal_legacy_regression_check/`).2048환경29000source actor/AMP88개 복사·critic81개 초기화·epoch2/frame262144 저장·161종scalar유한값·새checkpoint2환경30step 재로드 평가 통과(`output/open_push_friction1_random_goal_train_check/`, `output/open_push_friction1_random_goal_eval_check/`). 장기 본학습은 실행하지 않았다.

### 학습 정책 없는 물리 상한·직접 PD 구동 검사

`check_push_physics_envelope.sh`는 checkpoint/신경망/학습 없이 실제humanoid 질량50.2997kg·발마찰1·중력9.81을감사한다. 모든관절PD effort는3.4028e38(사실상토크제한없음)이고 motor gear100/70은현재PD토크상한이아니다. 수직손힘0·무외부지지·무충돌·준정적수평밀기라는조건에서 발마찰필요조건은493.44N이다. 평균박스/바닥마찰결합을가정한이상적질량상한은박스마찰0.1~1.0에서91.5/83.8/77.4/71.9/67.1/62.9/59.2/55.9/52.9/50.3kg이다. 생성시물성을적용한60비학습환경에서 각상한의80/100/120%질량·30조건+무외력대조군으로493.44N·높이0.5m의박스반응을검증했다. 이는박스역학의필요조건검증이며 에이전트가그질량을실제로밀었다는검증은아니다.

`check_push_actuator_load.sh`는 source모션4자세·ankle/torso/hip 목표36종·명령부하6종의864비학습환경을실제nativePD로5초구동한다. Jacobian-transpose offset과동일한action목표범위clip을사용하고 1000kg박스에서뒤2초의반력을기록한다. checkpoint/정책/optimizer/외부bodyforce는사용하지않는다. 지지/손근접조건90%이상인78조건을확인했고 최대평균net손반력기록159.9N·순간341.2N이었다. 탐색한자세/구동에서의기록이며 전역물리최대값이아니다. 별도센서대조에서20kg정지박스net Z가중량보다낮고100N수평외력시바닥net Y가0으로기록돼,접촉력텐서값을정확한출력상한이나완전한힘평형/질량환산에사용하지않는다.

```bash
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/check_push_physics_envelope.sh
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/check_push_actuator_load.sh
```

GPU6의30개상한주변박스조건/무외력대조·864개직접PD조건 및 실제물성감사·Python/셸문법·diff를확인했다. 보고서는[이상적상한/박스역학](../output/push_physics_envelope_20261012/REPORT.md),[직접PD구동/측정제한](../output/push_actuator_load_20261012/REPORT.md)에있다. source50.2997kg와foot/bottom마찰1의실제속성이바뀌면검사는assert로중단한다. 학습config/기존checkpoint/RSI bank는변경하지않았다.

### 실제 정책의 무게·마찰 교차 검사

`check_open_push_policy_capacity.sh`는 기본29000접촉+AT 정책을 정상 계약 검사로 로드하고 학습 없이 실제 action을 실행한다.1.3m 박스의 질량11종(5/10/12/15/20/25/30/40/60/100/219.7kg)과 박스마찰0.1~1.0의10종을 actor생성 시 적용한다. 바닥마찰1과 정책/학습config/원본bank는 유지한다. GPU PhysX에서는prepare_sim 이후shape마찰setter의조회값이바뀌어도 실제힘응답은바뀌지않는것을 대조검사로 확인해 생성시물성설정을 사용한다. 모든110조합의기하/관절초기상태를같게 복사하고 질량/마찰getter 및 초기상태 동일성을 검사한다.

접촉시작은 기존screened pose를박스진행0에정렬·정면2m목표·10초×3반복이다. 보행시작은기존보행reset·박스중심XY±1.5m 독립목표/사람1m배제·20초×2반복이다.220환경(평가전용)·조합별 접촉12/보행8 agent-trial을쓴다. 외력은 넣지 않는다. 이동은 네바닥모서리각각의 목표방향변위 중최소값으로측정하고 기울기0.35rad초과/높이오차8cm초과를분리한다. 안정이동률은 해당거리달성과 에피소드동안전도없음을함께요구한다. 현재고정정책의표본성능이며 관절최대힘이나재학습된정책의물리한계가아니다.

```bash
# 실행 전 nvidia-smi 확인. 기존 프로세스는 유지한다.
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/check_open_push_policy_capacity.sh
# 같은 원본계약의 다른checkpoint를 비교할 때만 첫인자로 지정
TOKENHSI_GPU=6 OUTPUT_PATH=output/push_policy_capacity_other bash tokenhsi/scripts/multi_agent/check_open_push_policy_capacity.sh <contact_at_warmstart_checkpoint.pth>
```

검증: GPU6 전체220환경·110조합·접촉/보행2200 agent-trial 완료,조합별getter·동일초기상태·유한관측 검사를통과했다. 마찰1은5/6/7/8/9/10/11/12/15/20kg·40평가환경·접촉48trial씩480개추가검사했다. 마찰1에서안정50cm는9kg64.6%·10kg37.5%·12kg6.25%·20kg0%,10kg은안정10cm100%였다. 접촉시작안정50cm≥50%의가장무거운시험조건은마찰0.1~0.5에서40kg,0.6에서30kg,0.7에서20kg,0.8에서15kg,0.9에서12kg,1에서9kg(추가검사)다. 작은표본/무게간격/현재정책조건이며 물리적절대상한이아니다. 보행·랜덤목표는모든조합에서안정50cm50%미만이다. [전체결과](../output/push_policy_mass_friction_grid_20261012/REPORT.md)에trial별원본/CSV/PNG/PDF를기록했다. 생성시마찰0.1/1.0의실제120N응답차이도별도대조확인했다. 기존runtime-setter첫결과는invalid표시로분리했다. Python/셸문법·diff통과, 학습은실행하지않았다.

```bash
# 마찰1 무게경계 추가검사:48 agent-trial/무게, 별도output
TOKENHSI_GPU=6 CAPACITY_MASSES=5,6,7,8,9,10,11,12,15,20 CAPACITY_FRICTIONS=1.0 CAPACITY_SCENES=4 CAPACITY_CONTACT_REPEATS=6 CAPACITY_WALK_REPEATS=0 CAPACITY_NUM_ENVS=40 OUTPUT_PATH=output/push_policy_friction1_mass_boundary_20261012 bash tokenhsi/scripts/multi_agent/check_open_push_policy_capacity.sh
```

결과는 `output/push_policy_mass_friction_grid_20261012/`의 `audit.json`, `trials.json`, `summary.csv`에저장한다. `CAPACITY_MASSES`, `CAPACITY_FRICTIONS`, `CAPACITY_SCENES`로축소검사할때는 `CAPACITY_NUM_ENVS=무게종수×마찰종수×scene수`도맞춰야한다. `CAPACITY_CONTACT_REPEATS/WALK_REPEATS`, `CAPACITY_CONTACT_STEPS/WALK_STEPS`는 검사반복/길이만바꾼다. 평가환경설정이므로학습환경2048규칙과별개다.

### PUSH 접촉 + 기존 AT 배치

`open_push_push180_contact_at.yaml`은 실제 장면 무작위 PUSH180 설정에 `relationReward.contact_at`를 추가한 실행 가능한 별도 실험이다. graph는 `H→PUSH→O→AT→G`로 유지하고 두 edge의 state·progress·success 보상을 각각 계산해 합산한다. OPEN 보상, 박스1.3m/20kg/마찰0.6, 배치 무작위화, AMP21개, RSI 비율은 기반 실험과 같다.

- PUSH state는 양손 중 더 먼 손의 뒤면XY 거리로 `exp(-10*d²)`, progress는 사람–상자XY 거리의 `1/(1+max(d-0.5,0))`다. 성공은 기존 양손 뒤면XY 근접·반력 및 상자 전방 힘 접촉 검사다. 손 높이는 기존 접촉 검사처럼 자유다. 물체 이동 실적을 요구하지 않는다.
- AT state는 기존 식 `exp(-10*||상자중심-목표||²)`, progress는 상자–목표XY 거리의 같은 거리식이다. 접촉·credit으로 제한하지 않는다. 성공은 state≥0.9와 높이 오차≤0.001m이며 속도·기울기·손 이동 이력을 요구하지 않는다. 높이 기준은 상자 중심 목표Z다. 기존 PUSH의 `goal_tolerance: 0.15`가 새 AT의 성공 반경을 정하지 않는다(수평 기준 약0.103m).
- edge당 state/progress/success 계수는 각각0.2로 두 edge를 합산한다(기존 공통 state0.4를 두 edge에0.2씩 배분). 팀 공유는 기존 자기0.9/동료0.1 합계를 유지한다. 최종 task 성공은 AT다. AT 성공 시 HOLDING+AT처럼 PUSH reward도 포화되어 손을 놓을 수 있지만 PUSH 자체의 `own_success`는 현재 접촉 여부다.

기존 RSI bank는 같은 물리·손 자세를 검사한 더 엄격한 이전 PUSH screen으로 재사용한다. bank 계약에서는 정책 보상 전용 `contact_at`만 제외한다. checkpoint 계약에는 새 보상 종류와 AT 파라미터를 기록하므로 이전 PUSH checkpoint의 직접 resume/test는 거부한다. 새 scratch 학습을 사용하며 이번 수정에서 장기학습은 실행하지 않았다.

```bash
# 별도 scratch 학습 (2048환경)
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/open_push_push180_contact_at_train.sh 2 2048 4
# 짧은 학습 확인용 별도 output
TOKENHSI_GPU=6 OUTPUT_PATH=output/open_push_contact_at_train_check MAX_ITERATIONS=2 bash tokenhsi/scripts/multi_agent/open_push_push180_contact_at_train.sh 2 2048 4
# 새 실험에서 생성한 checkpoint로 로컬 viewer / 화면 없는 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_push180_contact_at_test.sh <contact_at_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_push180_contact_at_test.sh <contact_at_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 PORT=6099 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_push180_contact_at_vnc.sh <contact_at_checkpoint.pth> 2 1 4 100
```

검증: 접촉/AT 독립 성공·무접촉 이동 보상·목표에서 release·높이 실패·부분reset·RSI screen 계약 동등성 및 기존 OPEN/PUSH/RSI/구역/장면 CPU 테스트74개·문법/스크립트경로/diff 통과. GPU6 비학습2환경200reset 배치/독립binding·부분reset·네task 보행16reset/조작RSI32reset×12step 관측/보상/AMP 유한값 및 새 보상계약 확인 통과(`output/open_push_contact_at_sim_check/audit.json`, RSI 양손 오차 최대0.1609m). 장기 정책 효과와 GUI/VNC는 미검증이다.

### PUSH 접촉+AT 전이 학습 — OPEN도 공동 학습

26000 저장본의 **OPEN/PUSH 단독** GPU6·각64환경×8반복(각512장면) 결과/명령은 [단독 평가 보고서](open_push_26000_separate_eval_20261012.md)에 있다. 개인 목표 성공 OPEN0%·PUSH93.65%; PUSH는release허용이며 원본접촉장면지표와구분한다.

26000 저장본의 GPU6·64환경×8반복(총512장면) headless 평가 결과와 재실행 명령은 [평가 보고서](open_push_26000_eval_20261012.md)에 있다. 기본random OPEN/PUSH·보행 시작, 개인49.41%/장면최종1.17%이며 두 지표의 성공 조건은 다르다.

VNC 공통 실행기는 X display와 대응 VNC 포트가 모두 비어 있는 번호를 선택한다. 26000 저장본을 OPEN/PUSH 한 명씩 시각 평가하는 명령:

```bash
TOKENHSI_GPU=6 PORT=6099 TASK_GRAPH=open_push OUTPUT_PATH=output/open_push_26000_eval_gpu6_20261012 bash tokenhsi/scripts/multi_agent/open_push_push180_contact_at_warmstart_vnc.sh output/open_push_push180_contact_at_warmstart/OpenPushPush180ContactAtWarmstart_09-17-51-36/nn/OpenPushPush180ContactAtWarmstart_00026000.pth 2 1 4 1000
```


`open_push_push180_contact_at_warmstart.yaml`은 위 접촉+AT와 같은 env/network/reward/AMP/RSI이며 기본 source는 `output/open_push_push180_random_scene_scratch/OpenPushPush180RandomScene_09-03-35-26/nn/OpenPushPush180RandomScene_00004500.pth`다. 새 학습 초기화에서 actor encoder/action head/sigma와 AMP discriminator, scene 입력RMS·AMP 입력RMS를 복사한다. Critic encoder/value head·optimizer·reward RMS·epoch/frame·AMP replay는 새 초기값이며 actor/AMP를 고정하지 않고 계속 학습한다. 사람별 OPEN/PUSH50/50,2048환경과 기존 물성을 유지한다.

전용 wrapper는 GPU 숫자/UUID를 받아 이전 GPU6 제한 없이 지정 장치에서 실행한다. `TRANSFER_CHECKPOINT`는 같은 기존 random-scene 보상·AMP/RSI·물성·network 계약의 다른 source로 바꿀 때만 사용한다. `RESUME_CHECKPOINT`와 동시에 지정하면 거부한다. 새 전이 checkpoint의 후속 resume/test는 일반 새 접촉+AT 계약을 검사하며 과거 source를 다시 복사하지 않는다. CPU 전이 검사는 모델·통계 전체 형상/유한값·계약을 확인한 후 복사하고 이전critic/optimizer/보상RMS/counter를 로드하지 않는다.

```bash
# 본학습: 기본4500 source에서 새 counter로 시작, 별도 output. 실행 전 nvidia-smi 점유 확인.
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS -u TRANSFER_CHECKPOINT TOKENHSI_GPU=1 SEED=42 OUTPUT_PATH=output/open_push_push180_contact_at_warmstart bash tokenhsi/scripts/multi_agent/open_push_push180_contact_at_warmstart_train.sh 2 2048 4
# 확인용1회 제한: 본학습 output과 분리
env -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 SEED=42 OUTPUT_PATH=output/open_push_contact_at_warmstart_check MAX_ITERATIONS=1 bash tokenhsi/scripts/multi_agent/open_push_push180_contact_at_warmstart_train.sh 2 2048 4
# 새 전이 학습 저장본으로 후속 resume (초기4500 source 전이를 다시 하지 않음)
env -u TRANSFER_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=1 RESUME_CHECKPOINT=<new_contact_at_checkpoint.pth> bash tokenhsi/scripts/multi_agent/open_push_push180_contact_at_warmstart_train.sh 2 2048 4
# 로컬 viewer / 화면 없는 평가 / 서버 VNC: 새 접촉+AT checkpoint
TOKENHSI_GPU=1 HEADLESS=0 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_push180_contact_at_warmstart_test.sh <new_contact_at_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=1 HEADLESS=1 TASK_GRAPH=random_open_push bash tokenhsi/scripts/multi_agent/open_push_push180_contact_at_warmstart_test.sh <new_contact_at_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=1 PORT=6099 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_push180_contact_at_warmstart_vnc.sh <new_contact_at_checkpoint.pth> 2 1 4 100
```

검증: 전이CPU7개·접촉+AT/기존OPEN/PUSH 포함56개·Python/셸문법·diff 통과. GPU6·2048환경 실제4500source에서 actor/AMP88텐서 복사·critic81텐서 초기화 보고서·epoch2/frame262144 저장·새optimizer state·161종 TensorBoard scalar 유한값 확인(`output/open_push_contact_at_warmstart_check/`). 새전용test의2환경30step 재로드평가 종료코드0 확인(`output/open_push_contact_at_warmstart_eval_check/`). 본학습과 장기 성능 개선은 미검증이다.

### PUSH 랜덤 goal 제로샷 시각화 — warmstart 10000/29000

기존 직선2m목표로 학습한 `OpenPushPush180ContactAtWarmstart` 정책을 원래 YAML과 checkpoint 계약 검사로 로드한다. 전용 평가 진입점이 **보행 초기화 후, 관측/marker/보상 history 초기화 전** goal만 교체한다. 두 사람 모두 PUSH·loco 시작이며 물성/배치/접촉면/성공식은 기존 정책 기준이다. 새 학습 config나 학습 명령은 없다.

각 박스의 초기 중심 기준 world XY 각각±1.5m에서 균등 후보를 뽑고 거리1~2m, 두 사람의 초기 root에서1m이상인 goal만 수락한다. Z는 초기박스높이를 유지하고 에피소드동안 목표는 고정한다. 최대2m로 기존 작업영역을 유지한다. goal RNG는 별도이며 두 checkpoint에 같은 seed42를 사용할 수 있다. 정책에 따라 종료/reset 시점이 달라지면 이후 장면까지 완전히 같은 비교를 보장하지 않는다. 목표의 관측과 화면 점은 실제 `_tar_pos`를 사용하며 `random_goal_resets.jsonl`에 초기박스/사람/배정/goal을 기록한다. 기본카메라는 두사람/박스/목표 전체를 위에서 보며 `TASK_CAMERA=agent`는 기존근접시점을사용한다. 기존 학습 보상·bank·checkpoint는 수정하지 않는다. 성공률은 별도 평가 결과로 확인해야 한다.

```bash
CKPT_DIR=output/open_push_push180_contact_at_warmstart/OpenPushPush180ContactAtWarmstart_09-17-51-36/nn
# 서버 VNC: 29000 / 10000, 각각 별도 터미널에서 실행
TOKENHSI_GPU=6 PORT=6102 SEED=42 OUTPUT_PATH=output/open_push_zero_shot_29000_viewer bash tokenhsi/scripts/multi_agent/open_push_zero_shot_random_goal_vnc.sh "$CKPT_DIR/OpenPushPush180ContactAtWarmstart_00029000.pth" 2 1 4 100
TOKENHSI_GPU=6 PORT=6103 SEED=42 OUTPUT_PATH=output/open_push_zero_shot_10000_viewer bash tokenhsi/scripts/multi_agent/open_push_zero_shot_random_goal_vnc.sh "$CKPT_DIR/OpenPushPush180ContactAtWarmstart_00010000.pth" 2 1 4 100
# 로컬 viewer / 화면없는 평가
TOKENHSI_GPU=6 HEADLESS=0 SEED=42 OUTPUT_PATH=output/open_push_zero_shot_29000_local bash tokenhsi/scripts/multi_agent/open_push_zero_shot_random_goal_test.sh "$CKPT_DIR/OpenPushPush180ContactAtWarmstart_00029000.pth" 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 SEED=42 OUTPUT_PATH=output/open_push_zero_shot_29000_eval bash tokenhsi/scripts/multi_agent/open_push_zero_shot_random_goal_test.sh "$CKPT_DIR/OpenPushPush180ContactAtWarmstart_00029000.pth" 2 64 4 8
```

### PUSH180 실제 위치·회전·시작·담당 물체 무작위화

`open_push_push180_random_scene.yaml`은 구역 배정 실험에서 PUSH의 실제 기하를 바꾸는 별도 실험이다. 크기는1.3m 정육면체,20kg·마찰0.6 고정이며 reward/성공·AMP21개·보행/조작50/50과 후반RSI70%를 유지한다. `pushSceneRandomization`이 PUSH reset을 제어하며 기존 generic `box.reset`의 rotation/assignment 필드 대신 task 전용경로에서 아래 샘플링을 수행한다.

- 두 물리 박스의XY를 중심 `(-4,0)`·반경4.5m 원 안에서 샘플하고 작업영역을 분리하기 위해 박스중심 사이6m 이상을 요구한다. 임의방향2m경로·박스반대각선 및 보행시작을 포괄하는 영역과 고정문/환경extent를 config에서 검증한다. 정책이 이 영역 밖으로 나가는 모든 충돌을 막는 조건은 아니다.
- 박스별yaw는0~2π 무작위. 목표는 **RSI 이동 전 박스 중심에서 reset yaw의local +Y 방향2m**이며 에피소드 도중 고정한다. 접촉은 현재박스법선, 목표거리/credit은 같은 고정목표를 따르고 전진 진단은 초기밀기방향으로 투영한다.
- 담당 박스는 두 사람에게 중복 없이 무작위 배정한다. OPEN문 구역 배정과 PUSH박스 배정은 독립이다. graph의canonical 논리슬롯에 해당 물리박스를 연결하고 관측·RSI·접촉력·credit·색상을 같은binding으로 모은다.
- PUSH보행 시작은 담당박스 중심에서 무작위 방향·거리1~2m, 사람heading도 무작위다. 초기body중심의 박스 침투 후보는 재추출한다. RSI는 통과후보의 사람/발/팔 자세를 박스의새XY/yaw로 강체 정렬하고, 선행진행은 같은 초기밀기방향으로 배치한다. 크기/관절/보상은 바꾸지 않는다.
- **OPEN문의 힌지 위치·방향·물리·초기화는 기존대로 유지**한다. 이 실험은 움직일 수 있는 PUSH박스에 SIT/HOLDING식 배치변화를 적용하며, 두 고정힌지 문을 임의 재배치하지 않는다. `OPEN_PUSH_LANES=fixed/swapped`는 OPEN구역에만 적용되고 PUSH의XY/yaw/binding은 계속 무작위다.

bank는 `tokenhsi/data/open_push_push180_random_scene_bank.json`이다. 이전단일과제PD통과후보를 그대로 보존하고 강체XY/yaw 불변성을 이용한재사용 출처와 새scene계약을 기록한다. 이는 전체무작위장면의 신규PD검사를 모든후보에 수행했다는 의미가 아니며 실제새배치의RSI는 아래소규모감사로검증했다. 기존bank는보존한다. checkpoint에 `push_scene` 계약을 추가하여 이전checkpoint 직접resume/test를 거부하고, `transfer_open_push_random_scene.sh`로 구역무작위PUSH180 checkpoint에서 scene/bank/snapshot만 전이한다. 모델/optimizer/정규화/epoch/frame과reward를 유지하며 기존파일은덮어쓰지 않는다. 고정구역checkpoint는 먼저random-lanes전이를거쳐야한다.

```bash
# 새scene계약의SE(2)재사용bank만 준비 (원본bank 유지·같은결과면재사용)
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/transfer_open_push_random_scene.sh --bank-only
# 구역무작위 계약checkpoint에서 명시적 전이 (아래3500 전이본은 준비됨)
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/transfer_open_push_random_scene.sh <random_lane_checkpoint.pth> <new_random_scene_checkpoint.pth>
# 별도 본학습: 이번 구현에서는 장기학습 미실행
TOKENHSI_GPU=6 RESUME_CHECKPOINT=output/open_push_push180_random_scene/transfer/epoch3500_random_scene.pth MAX_ITERATIONS=1000000 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_push180_random_scene_train.sh 2 2048 4
# 로컬 viewer / headless 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_push180_random_scene_test.sh <random_scene_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_push180_random_scene_test.sh <random_scene_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 PORT=6099 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_push180_random_scene_vnc.sh output/open_push_push180_random_scene/transfer/epoch3500_random_scene.pth 2 1 4 100
# 비학습2환경: XY/yaw/거리/독립binding/관측/접촉/부분reset/RSI
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/validate_open_push_scene.sh
# 별도output·2048환경 짧은resume
TOKENHSI_GPU=6 OUTPUT_PATH=output/open_push_random_scene_resume_check RESUME_CHECKPOINT=output/open_push_push180_random_scene/transfer/epoch3500_random_scene.pth MAX_ITERATIONS=3501 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_push180_random_scene_train.sh 2 2048 4
```

검증: CPU70개·Python/셸문법·diff 통과. GPU6 비학습2환경에서200reset의XY/yaw/독립binding4조합·거리1.005~1.998m·부분reset·네task보행16reset×12step·네task조작RSI32reset×12step·접촉력/목표/관측/AMP유한값 통과,RSI양손목표오차최대0.1609m(`output/open_push_random_scene_sim_check/audit.json`). 실제3500전이의모델/optimizer/정규화/카운터전체보존·원본PD후보동일,2048환경3500→3502 resume/저장·scene계약/크기/물성/reward/AMP확인 통과(`output/open_push_random_scene_resume_check/audit.json`). 새test wrapper의전이본2환경30step복원종료코드0 확인(`output/open_push_random_scene_test_check/metrics/`). 장기정책개선·GUI/VNC는 미검증이다.

### PUSH180 담당 구역50/50 무작위화

`open_push_push180_random_lanes.yaml`은 PUSH18021개 설정에 `openPushRandomizeLanes: true`만 추가한다. reset마다 장면별로 `[0,1]` 또는 `[1,0]` 배정을50/50으로 선택하고 부분 reset은 해당 장면만 변경한다. 문/박스의 물리 배치는 고정하며 사람의 구역·담당 물체를 함께 교환한다. graph의 논리 슬롯은 담당자 순서를 유지하고 관측·RSI·목표·접촉력·이동 credit·보상·물체 색상은 같은 물리 배정을 따른다. 기존 보상/성공·물성·AMP·초기화 비율과 네트워크는 유지한다. 구역 편향만을 비교하는 실험이며 정지/발·몸 밀기/목표 초과 문제의 해결 효과는 아직 확인하지 않았다.

기존 physical RSI bank는 그대로 재사용한다. checkpoint에는 `lane_randomization: {version: 1, swap_probability: 0.5}` 계약을 추가하므로 기존 고정 구역checkpoint를 새 실험에 직접resume/test할 수 없다. `transfer_open_push_random_lanes.sh`는 고정구역PUSH180 계약·bank·학습 차원·network를 확인하고 구역 계약/실험 snapshot만 변경하며 모델·optimizer·정규화·epoch/frame을 보존한다. source와 output을 분리하고 이미 존재하는 output은 거부한다. 기존3500 저장본의 전이본은 아래 경로에 생성했다.

평가/VNC의 `OPEN_PUSH_LANES=random`(기본)은 구역을50/50으로 선택한다. `fixed`는 초록/아래·파랑/위, `swapped`는 초록/위·파랑/아래로 고정한다. 학습은 이 환경변수를 무시하고 항상 무작위 배정한다. 기존 config에서 옵션이 없으면 기존 고정 구역 동작/계약을 유지한다. `open_push_agent_lanes` extra는 사람별 물리구역0/1을 기록한다.

```bash
# CPU/full-state 명시적 전이: source는 고정구역 PUSH180 checkpoint
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/transfer_open_push_random_lanes.sh <fixed_push180_checkpoint.pth> <new_random_lane_checkpoint.pth>
# 준비된3500 전이본으로 별도 본학습 (이번 구현 작업에서는 장기학습 미실행)
TOKENHSI_GPU=6 RESUME_CHECKPOINT=output/open_push_push180_random_lanes/transfer/epoch3500_random_lanes.pth MAX_ITERATIONS=1000000 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_push180_random_lanes_train.sh 2 2048 4
# scratch는 resume 환경변수를 제거하고 별도output 사용
env -u RESUME_CHECKPOINT TOKENHSI_GPU=6 OUTPUT_PATH=output/open_push_random_lanes_scratch SEED=42 bash tokenhsi/scripts/multi_agent/open_push_push180_random_lanes_train.sh 2 2048 4
# 로컬 viewer / headless 평가: 새 실험 계약의 checkpoint 사용
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=push OPEN_PUSH_LANES=random bash tokenhsi/scripts/multi_agent/open_push_push180_random_lanes_test.sh <random_lane_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=push OPEN_PUSH_LANES=random bash tokenhsi/scripts/multi_agent/open_push_push180_random_lanes_test.sh <random_lane_checkpoint.pth> 2 16 4 3
# 서버 VNC: fixed/swapped로 각 색의 구역을 고정해 비교 가능
TOKENHSI_GPU=6 PORT=6099 TASK_GRAPH=push OPEN_PUSH_LANES=random bash tokenhsi/scripts/multi_agent/open_push_push180_random_lanes_vnc.sh <random_lane_checkpoint.pth> 2 1 4 100
# 비학습2환경 mapping·RSI·부분reset 검증
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/validate_open_push_lanes.sh
# 2048환경 짧은 resume 검증용: 본학습output과 분리
TOKENHSI_GPU=6 OUTPUT_PATH=output/open_push_random_lanes_resume_check RESUME_CHECKPOINT=output/open_push_push180_random_lanes/transfer/epoch3500_random_lanes.pth MAX_ITERATIONS=3501 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_push180_random_lanes_train.sh 2 2048 4
```

검증: CPU총66개(기존/신규62개+이동credit 추가4개)·Python/셸 문법·diff 통과. GPU6 비학습2환경에서 네 task 조합×기본/교환배정12step, 비대칭 접촉력 담당자mapping, 목표/관측pose/AMP label·200reset 교환49.5%·부분reset·네 task RSI의 양쪽배정48reset×12step 유한값 통과(`output/open_push_random_lanes_sim_check/audit.json`). 실제3500 전이본의 모델/optimizer/정규화/카운터 등 state전체보존,2048환경3500→3502 resume/저장·새구역계약/기존reward와bank보존 통과(`output/open_push_random_lanes_resume_check/audit.json`). 새test wrapper로 전이본2환경30step교환배정복원 종료코드0 확인(`output/open_push_random_lanes_test_check/metrics/`). 기존본학습은 유지하며 새 장기학습/정책 개선·GUI/VNC는 미검증이다.

### RSI 전체 모션·프레임 타임라인 (checkpoint 불필요)

`build_open_push_rsi_timeline.sh`는 로드된 AMP clip의 **모든 프레임**과 해당 config bank의 RSI 시작 후보를 독립 HTML로 출력한다. 기본은 문 마찰0.5·PUSH2m다. 드롭다운으로 OPEN/PUSH 두 clip을 선택하고, 처음부터/재생/←→ 한 프레임/이전·다음 RSI 후보/0.25~1배속을 사용할 수 있다. 주황 표시는 각 clip의 통과 후보 뒤쪽30%다. 후보가 없는 구간도 재생에서 생략하지 않고 끝에서 멈춘다. 후보 이동은 의도적인 시간 이동이며 순차 재생과 구분한다. PUSH 원본 AMP와 같은 `lower_push_arms` 보정 자세를 겹쳐 표시하고 높이 보정을 끄면 원본만 볼 수 있다. 뼈대 시각화이며 물체 접촉·정책 rollout 검사는 아니다. root를 화면 중심에 따라가고 회전/확대가 가능하다.

```bash
# CPU 생성: 기본2m config (GPU 학습/시뮬레이션 실행 안 함)
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_open_push_rsi_timeline.sh
# 현재 학습 중인1.5m 실험의 후보로 생성
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_open_push_rsi_timeline.sh --cfg-env tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction.yaml --output-path output/open_push_hinge_friction_rsi_timeline
# 기본2m 결과를 서버에서 보기: VS Code PORTS에서6090 포워딩
python3 -m http.server 6090 --bind 127.0.0.1 --directory output/open_push_push2m_rsi_timeline
# 브라우저: http://localhost:6090/rsi_timeline.html
```

현재 제공 중인1.5m 뷰어는 `output/open_push_push2m_rsi_timeline/current_1p5m/rsi_timeline.html`에 있으며 기본2m 폴더를6090으로 서빙할 때 `http://localhost:6090/current_1p5m/rsi_timeline.html`로 접근한다. 검증:3clip 프레임/후보 이동·끝 정지 JS 검사·Firefox 렌더링·HTTP200 통과.

### PUSH180 AMP 21개·2m·넓은 배치 (2000 resume)

`dataset_open_push_push180.yaml`은 요청 목록의180 계열21개를 같은 weight1로 연결한다. 기존2개 변환 파일은 보존하고 신규19개는 `dataset_bones_push180/`에 변환했다. 원본0초부터 metadata의 첫 밀기 멈춤/손 해제 시작 전까지30FPS로 변환하며 crop과 전체 event는 `bones_push180_selection.json`에 기록한다. 270/360은 포함하지 않는다. loco4개·OPEN1개와 task별 AMP 혼합 비율, 양손 credit,2m 목표·넓은 배치는 기존과 같다.

config는 `open_push_late_rsi_hinge_friction_push2m_push180.yaml`, bank는 `open_push_late_rsi_cube_low_hinge_friction_push2m_wide_push180_bank.json`, output은 `output/open_push_late_rsi_cube_low_midpoint_xy_hinge_friction_push2m_wide_push180`이다. 기존2모션/1.5m checkpoint는 직접 resume/test할 수 없으므로 `transfer_open_push_push180.{py,sh}`로 명시적으로 전이한다. 도구는 기존마찰0.5/revision5 계약을 확인하고 PUSH1.5→2m·넓은배치·21개manifest/RSI 및 reward metadata만 갱신하며 모델/optimizer/정규화/epoch/frame을 보존한다. 원본을 덮어쓰지 않는다. 사용자 종료한 기존 학습의2000 저장본을 전이해 `tmux koo_train:0.0`에서 GPU6·2048환경으로 resume했다. 상한은 원래값1000000을 유지하며9000으로 변경하지 않았다. launcher/log/실행PID·run 경로는 output의 `launch_resume2000.sh`, `train_resume2000.log`, `latest_launch.json`에서 확인한다.

```bash
# CPU 신규19개 변환(완료된 동일 source/crop 결과 재사용; 기존2개 유지)
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash -c '. tokenhsi/scripts/multi_agent/runtime_env.sh; python tokenhsi/scripts/multi_agent/convert_bones_push.py --expanded-180 --workers 4'
# 비학습16환경(후보당2환경×8후보) PD 검사·bank 생성
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_open_push_hinge_friction_push2m_push180_rsi.sh
# 명시적 전이(아래 전이 파일은 이미 생성됨; 새 output 파일에만 작성 가능)
SOURCE_CHECKPOINT=output/open_push_late_rsi_cube_low_midpoint_xy_hinge_friction/OpenPushLateRSIHingeFriction_08-19-24-10/nn/OpenPushLateRSIHingeFriction_00002000.pth
TARGET_CHECKPOINT=output/open_push_late_rsi_cube_low_midpoint_xy_hinge_friction_push2m_wide_push180/transfer/epoch2000_push180_2m_wide.pth
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/transfer_open_push_push180.sh "$SOURCE_CHECKPOINT" "$TARGET_CHECKPOINT"
# 실제 본학습 resume 명령(2048환경·총epoch 상한1000000, 현재 실행 중이면 중복 실행하지 않음)
TOKENHSI_GPU=6 MAX_ITERATIONS=1000000 RESUME_CHECKPOINT="$TARGET_CHECKPOINT" SEED=42 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_push2m_push180_train.sh 2 2048 4
# 현재 본학습 화면
tmux attach -t koo_train
# scratch 실행은 별도output으로 구분
# env -u RESUME_CHECKPOINT TOKENHSI_GPU=6 MAX_ITERATIONS=1000000 OUTPUT_PATH=output/open_push_push180_scratch SEED=42 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_push2m_push180_train.sh 2 2048 4
# 이 실험 checkpoint로 로컬 추론 / 화면 없는 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_push2m_push180_test.sh <push180_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_push2m_push180_test.sh <push180_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 PORT=6089 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_push2m_push180_vnc.sh <push180_checkpoint.pth> 2 1 4 100
# checkpoint 없이 모든 AMP 프레임·RSI 후보 보기
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_open_push_rsi_timeline.sh --cfg-env tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction_push2m_push180.yaml --motion-file tokenhsi/data/dataset_open_push_push180.yaml --output-path output/open_push_push2m_rsi_timeline/push180
# 기존6090 서버 사용 시 http://localhost:6090/push180/rsi_timeline.html
# 실제 물리 RSI 시작 자세: N 새 표본, Space 12PDstep, P PUSH
TOKENHSI_GPU=6 PORT=6091 RSI_CFG_ENV=tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction_push2m_push180.yaml RSI_MOTION_FILE=tokenhsi/data/dataset_open_push_push180.yaml OUTPUT_PATH=output/open_push_push180_rsi_view bash tokenhsi/scripts/multi_agent/open_push_rsi_view_vnc.sh
```

검증: 전체21개 변환의 실제 DOF 범위/속도·접지·유한값·crop과 기존2개 hash 보존 통과(최대 endpoint 오차2.02cm 미만), CPU55개 통과. GPU6 비학습16환경(후보당2환경)12step PD에서 OPEN13/13·PUSH1048/1744 통과,21개 모두 후보 확보·1024reset/부분reset/AMP/고정2m목표 통과(`output/open_push_hinge_friction_push2m_wide_push180_rsi_check/audit.json`). 기존2환경 경로도 bank 복사본으로1000reset/AMP 통과,headless RSI viewer3표본/12step 통과.22clip 프레임/후보/끝정지 JS·HTTP200·Firefox 렌더링 확인. 전이 전후 학습state/hash 보존 및 잘못된 변경5종 거부 확인,2048환경2000→2002 짧은resume/저장·새계약 확인(`output/open_push_push180_resume_check/audit.json`). 본학습도2000부터 진행 중이며 장기 성공 성능은 아직 평가하지 않았다.

### PUSH 목표 거리 2m·구역 간격 확대 (학습 미실행)

`open_push_late_rsi_hinge_friction_push2m.yaml`은 기존 마찰0.5·1.5m 실험을 보존하고 PUSH 목표를2m로 늘린 별도 설정이다. `openPushLayout: {door_x: 4, push_x: -4, lane_spacing: 10}`로 문/박스 가로 간격2.5→8m, 두 작업 구역Y 간격5→10m를 넓힌다. 문 힌지 중심은 `(4,-5)/(4,5)`, 박스 시작 중심은 `(-4,-5)/(-4,5)`, 목표 중심은 `(-4,-3)/(-4,7)`이다. 보행 사람은 자기 과제의 손잡이/박스 뒤에서 기존1.5~2.5m 거리로 시작하고 조작 RSI는 이동한 자기 물체에 맞춰 정렬한다. 목표와 부분 reset도 새 위치를 따른다. `envSpacing: 14`로 확대 배치의 환경 영역을 확보한다. 구역 사이 정상2m 경로를 분리한 것이며 정책이 경로 밖으로 돌아다니는 모든 충돌을 막는 것은 아니다.

문 물리·에피소드600step·AMP·손 credit·성공 허용거리15cm는 유지한다. 출력은 `output/open_push_late_rsi_cube_low_midpoint_xy_hinge_friction_push2m_wide`, bank는 `tokenhsi/data/open_push_late_rsi_cube_low_hinge_friction_push2m_wide_bank.json`이다. 배치를 bank/checkpoint 계약에 포함하며 기존 배치의 config·bank·checkpoint는 보존했다. 배치 키가 없는 실험은 원래 좌표와 계약을 유지한다. 기존1.5m/좁은2m checkpoint는 직접 resume/test할 수 없다. **1500회 checkpoint 전이는 미구현이며 새 학습/기존 프로세스 전환은 하지 않았다.**

```bash
# 비학습2환경 bank 생성/검증
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_open_push_hinge_friction_push2m_rsi.sh
# 네 과제 조합·보행/RSI 배치·목표·부분 reset·짧은 PD 간격 검사
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/check_open_push_layout.sh
# 향후 scratch 학습·2048환경 (이번 작업에서는 실행하지 않음)
env -u MAX_ITERATIONS -u RESUME_CHECKPOINT TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_push2m_train.sh 2 2048 4
# 넓은2m 설정으로 저장한 checkpoint의 로컬 추론 / headless 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_push2m_test.sh <2m_wide_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_push2m_test.sh <2m_wide_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 PORT=6089 TASK_GRAPH=push bash tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_push2m_vnc.sh <2m_wide_checkpoint.pth> 2 1 4 100
# checkpoint 없이 현재2m RSI 배치 검사·C 키로 다른 사람/구역
TOKENHSI_GPU=6 PORT=6091 RSI_CFG_ENV=tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction_push2m.yaml OUTPUT_PATH=output/open_push_push2m_wide_rsi_view bash tokenhsi/scripts/multi_agent/open_push_rsi_view_vnc.sh
```

검증: CPU54개·GPU6 비학습2환경PD OPEN13/13·PUSH154/171·1000reset/부분reset/고정목표/AMP 통과(`output/open_push_hinge_friction_push2m_wide_rsi_check/audit.json`). 네 과제×보행/RSI160scene·960PDstep에서 사람 관절 중심 사이 최소8.28m(`output/open_push_push2m_wide_layout_check/layout_audit.json`), RSI headless3sample/12step 통과.

기존 좁은2m 검증 기록: CPU49개·GPU 비학습 후보 OPEN13/PUSH114·1000reset/AMP 통과(`output/open_push_hinge_friction_push2m_rsi_check/audit.json`)이며 현재 넓은 배치의 결과와 구분한다.

### OPEN 초기 밀기 저항 비교: hinge friction 0.5

`open_push_late_rsi_hinge_friction.yaml`은 기존 revision5 late RSI에서 문 `hinge.friction`만0→0.5로 높인다. 마찰은 양방향 회전을 저항하므로 초기 밀기뿐 아니라 닫힘에도 저항한다. 자동 닫힘 spring(stiffness2/damping0.5/max_torque4), hinge damping0.5, OPEN/PUSH 보상·성공·AMP는 유지한다. 기존 config/bank/output을 보존하므로 실행 중인 학습과 기존 checkpoint 평가에는 영향을 주지 않는다. 실제 문 개방 저항/학습 효과는 정책 rollout 비교가 필요하다.

전용 bank는 `tokenhsi/data/open_push_late_rsi_cube_low_hinge_friction_bank.json`, 출력은 `output/open_push_late_rsi_cube_low_midpoint_xy_hinge_friction`이다. 물리와 bank hash가 checkpoint 계약에 포함되므로 기존 마찰0 checkpoint는 새 설정에 직접 resume/test할 수 없다. 전용 `transfer_open_push_hinge_friction.sh`는 revision5 마찰0 checkpoint에서 마찰0.5로의 전이만 허용한다. 원본/대상 config·bank 계약이 마찰/bank 경로 외에는 동일하고 PD 후보가 완전히 같아야 하며, source bank hash와 보상 버전을 검사한다. 모델·optimizer·정규화 통계·epoch/frame은 보존하고 물리 전이 provenance를 새 checkpoint에 기록한다. 원본은 덮어쓰지 않는다. 변환본을 `RESUME_CHECKPOINT`로 전달한다.

```bash
nvidia-smi
# 새 물리의 비학습2환경 PD bank 생성/검증
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_open_push_hinge_friction_rsi.sh
# 기존 revision5 checkpoint를 새 마찰 조건으로 변환(원본 보존·새 파일만 허용)
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/transfer_open_push_hinge_friction.sh <기존_checkpoint.pth> output/open_push_late_rsi_cube_low_midpoint_xy_hinge_friction/transfer/<epoch>_friction05.pth
# 변환 checkpoint에서 이어 학습·2048환경; MAX_ITERATIONS는 추가 횟수가 아닌 전체 epoch 상한
# 기존 학습의 종료/전환은 별도 사용자 지시에 따른다.
env -u MAX_ITERATIONS TOKENHSI_GPU=6 RESUME_CHECKPOINT=output/open_push_late_rsi_cube_low_midpoint_xy_hinge_friction/transfer/<epoch>_friction05.pth bash tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_train.sh 2 2048 4
# 별도 scratch 학습·2048환경
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_train.sh 2 2048 4
# 새 마찰 설정으로 학습한 checkpoint만 사용
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=open bash tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_test.sh <새_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_test.sh <새_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 PORT=6088 TASK_GRAPH=open bash tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_vnc.sh <새_checkpoint.pth> 2 1 4 100
```

검증: GPU6 비학습2환경 PD에서 OPEN13/13·PUSH145/171 후보 통과,1000reset·partial reset·AMP 유한값·전용 bank 계약 일치 확인(`output/open_push_hinge_friction_rsi_check/audit.json`). YAML 변경 범위·wrapper4개 bash 문법·diff 검사 통과. 변환 전후 원본 학습 상태 완전 일치와 불일치5종 거부 확인. GPU6·2048환경1000→1002 짧은 resume/저장 통과(`output/open_push_hinge_friction_resume_check/audit.json`). 실제1500 후속 checkpoint는 `output/open_push_late_rsi_cube_low_midpoint_xy_hinge_friction/transfer/epoch1500_friction05.pth`로 준비했다. 새 조건의 본학습/정책 평가는 미실행이다.

### 현재 revision5: PUSH 중점 XY·높이 자유

PUSH 손 state는 `exp(-10 × ||양손 중점의 box yaw-local XY − [0, -depth/2]||²) × grounded`다. 오른손/왼손 목표점 배정을 제거하고 손 높이·양손 높이 차이에 보상이나 제한을 주지 않는다. 접촉은 각 손의 yaw-local XY가 뒤면 수평 선분에서18cm 이내이고 양손 전방 반력/박스 반력이 각각0.5N 이상일 때 인정한다. Z 검사는 없으며 GPU net force proxy이므로 실제 충돌 pair를 식별하는 것은 아니다. 중점만 맞고 두 손이 뒤면 폭 밖에 멀리 있으면 손 state는 높을 수 있지만 접촉 전진 실적은 적립되지 않는다.

기존 기울기0.35rad(약20도)·시작 대비 중심 높이±8cm의 grounded 제한, physics tick별 양손 전진 credit·RSI 초기 progress·성공 조건 및 OPEN 보상은 유지한다. RSI 어깨 정렬의 초기 높이0.8m와 기존 PD bank/AMP 원본도 유지하며, `hand_target_height`/`hand_lateral_offset`는 이제 RSI 정렬과 과거 기준 진단에만 쓰인다. 기존 bank는 고정점 조건을 통과한 시작 자세를 그대로 재사용하며 새 접촉 기준으로 후보를 확장하지 않았다.

**후속 계획:** 높이 자유 학습에서 높은 손 자세·기울기 위반이 지속되고 접촉 전진/성공이 개선되지 않으면, 그때 낮은 높이를 부드럽게 선호하는 보상을 추가한다. 현재는 이 보상을 넣지 않는다. `push/physics/{right_height,left_height,hand_height_difference,midpoint_xy_distance,tilt_violation,credited_forward_m}`와 성공률을 함께 확인한다. `right_distance/left_distance/worst_hand_distance/both_hands_near`는 이제 뒤면 XY 거리 기준이며 `*_height_error`는 RSI0.8m 기준의 진단일 뿐이다.

checkpoint revision5·`rsi_baseline_plus_hand_distance_midpoint_xy_v2`로 이전 revision1~4 직접 resume/test를 거부한다. 기본 output은 `output/open_push_late_rsi_cube_low_midpoint_xy_hand_credit`, `output/open_push_unified_amp_cube_low_midpoint_xy_hand_credit`다. 기존 실행 프로세스에는 자동 반영되지 않는다.

```bash
# 실행 전 점유 확인
nvidia-smi
# GPU6·2048환경 scratch 본학습 (late RSI)
env -u MAX_ITERATIONS -u RESUME_CHECKPOINT TOKENHSI_GPU=6 SEED=42 OUTPUT_PATH=output/open_push_late_rsi_cube_low_midpoint_xy_hand_credit bash tokenhsi/scripts/multi_agent/open_push_late_rsi_train.sh 2 2048 4
# 새 revision5 checkpoint: 로컬 추론 / headless 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_test.sh <새_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_test.sh <새_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 PORT=6087 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_vnc.sh <새_checkpoint.pth> 2 1 4 100
# unified는 loco-only 초기화 비교: 같은 revision5 손 보상
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_unified_amp_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_unified_amp_test.sh <새_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 PORT=6087 bash tokenhsi/scripts/multi_agent/open_push_unified_amp_vnc.sh <새_checkpoint.pth> 2 1 4 100
```

revision5 검증: 관련 CPU49개·train/test/VNC wrapper6개 bash 문법·diff 검사 통과. 이번 변경은 보상/접촉 계산 변경으로 GPU 학습/평가는 실행하지 않았다. 높이 자유 학습 성능은 미검증이다.

아래 revision4 및 고정 손 목표 설명은 이전 구현·검증 기록이다. 현재 PUSH 손 보상/접촉/출력은 위 revision5를 따른다.

2026-10-08 사용자 요청으로 확인용 학습/평가·physics/RSI 검사·뷰어 output30개를 삭제했다. 아래 과거 검증 결과의 output 경로는 현재 없을 수 있다. 본학습 `output/open_push_90_standing`, `output/open_push_late_rsi_cube_low`, `output/open_push_late_rsi_cube_low_joint_hand_credit`는 보존했으며 코드/실행 wrapper·모션·RSI bank는 유지한다.

**이전(2026-10-08) revision4: RSI 초기 진행도 + 손으로 만든 추가 진행도**를 보상한다. 기본 출력은 `output/open_push_unified_amp_cube_low_rsi_progress_hand_credit`, `output/open_push_late_rsi_cube_low_rsi_progress_hand_credit`다. 이전 revision1/2/3 checkpoint는 직접 resume/test할 수 없다. 기존 실행 중 프로세스는 이전 보상을 유지한다.

OPEN 개방 progress는 `0.5 × min(현재 각도, RSI 후 시작 각도 + 손 인정 개방각)/90도 × 현재 손 지지 × 자세`이며0~0.5로 제한한다. 유지 progress는 기존 `0.5 × 유지시간/5초`다. PUSH progress는 `min(실제 전체 목표 진행도, RSI 초기 진행도 + 손 인정 전진/전체 목표거리) × grounded × 현재 양손 접촉`이며0~1로 제한한다. 각 progress에 기존0.2 가중치를 적용한다. RSI 배치 직후에도 손 접촉 조건을 만족하면 초기 progress 보상이 나오지만, 초기 배치로 성공 자격은 주지 않는다. 몸으로 먼저 이동한 뒤 마지막에 손을 대도 초기 RSI 진행도 이상의 보상을 받으려면 추가 손 조작 실적이 필요하다. 인정 실적이 후퇴 때 차감되고 실제 reset 이후 순진행 이하로 제한되는 규칙은 유지한다.

성공 판정은 revision3의 손 실적 규칙을 유지한다. OPEN은 reset 이후 남은 성공각을 손으로 열고90도 이상·손 지지·서 있는 자세로5초 유지해야 성공한다. PUSH는 reset 이후 남은 거리−목표 허용15cm(최소5cm)를 접촉 전진으로 확보한 뒤 목표 위치/grounded/안정 조건을 만족해야 성공한다. 정상 PUSH 완료 후 release는 허용하고 state 손 거리 항은 성공 시 최대값으로 덮어쓰지 않는다. OPEN RSI는 PD검사 후보 중85도 이하만 사용하며 현재13개는 모두68.1도 이하라 원본 자세/후반 sampler와 bank를 유지한다. OPEN 유지의0.1초 접촉 grace는 기존대로이며 몸과 손의 동시 접촉 자체를 배제한 것은 아니다.

진단 태그 `relation/90_debug/open/{credited_angle_rad,required_angle_rad,hand_contact,eligible,hold_seconds}`, `relation/90_debug/push/physics/{credited_forward_m,required_forward_m}`는 계속 **reset 이후 손 실적**을 기록한다. `manipulation_progress`는 이제 RSI 초기분을 포함하고 접촉으로 제한한 보상 진행도다. 평균은 사람×control step 표본이다.

```bash
# GPU6·2048환경 scratch 본학습
env -u MAX_ITERATIONS -u RESUME_CHECKPOINT TOKENHSI_GPU=6 SEED=42 OUTPUT_PATH=output/open_push_late_rsi_cube_low_rsi_progress_hand_credit bash tokenhsi/scripts/multi_agent/open_push_late_rsi_train.sh 2 2048 4
# 새 revision4 checkpoint: 로컬/화면 없는 평가/서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_test.sh <새_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_test.sh <새_checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 PORT=6087 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_vnc.sh <새_checkpoint.pth> 2 1 4 100
# unified(loco-only) 비교 학습/로컬/서버 VNC
env -u MAX_ITERATIONS -u RESUME_CHECKPOINT TOKENHSI_GPU=6 SEED=42 OUTPUT_PATH=output/open_push_unified_amp_cube_low_rsi_progress_hand_credit bash tokenhsi/scripts/multi_agent/open_push_unified_amp_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_unified_amp_test.sh <새_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 PORT=6087 bash tokenhsi/scripts/multi_agent/open_push_unified_amp_vnc.sh <새_checkpoint.pth> 2 1 4 100
```

이하 revision2/3의 구현·검증 기록에서 손 인정 거리/각도만으로 계산한 progress는 역사 설명이다. 현재 보상 식은 위 revision4를 따른다.

2026-10-08부터 PUSH는 **physics tick별 양손 접촉 전진 거리**를 사용한다. 해당 tick의 양손 목표18cm 이내·손별/박스 전방 반력0.5N 이상·grounded 조건을 만족할 때만 목표XY 거리 감소를 누적한다. 목표에서 멀어지는 이동은 접촉 여부와 무관하게 누적 거리를 차감하고 reset 이후 실제 순전진 거리 이하로 제한한다. RSI 배치 직후 남은 거리를 기준으로 다시 초기화해 RSI 선행 이동은 인정하지 않는다. 성공 자격은 `인정 거리 >= max(reset 남은 거리−목표 허용15cm, 최소5cm)`이며 목표 위치·바닥/기울기·안정 조건도 유지한다. 성공 전 progress는 인정 거리/reset 남은 거리와 현재 control step의 양손 접촉으로 계산한다. 정상적으로 밀어 성공한 뒤 손을 떼어도 목표/안정 조건을 유지하면 성공·progress·bonus는 유지하지만, state의 손 거리 항은 최대값으로 덮어쓰지 않는다. OPEN 보상은 유지한다. 동시 몸 접촉 자체를 금지한 것은 아니며 GPU net force 기반 접촉 proxy의 한계는 유지된다.

PUSH만 먼저 수정한 revision2 출력은 `output/open_push_unified_amp_cube_low_hand_credit`, `output/open_push_late_rsi_cube_low_hand_credit`였다. 현재는 위의 revision4 출력을 사용한다. 기존 train/test/VNC wrapper를 사용한다. checkpoint 계약 revision2·`per_tick_hand_credit_v1`로 이전 정책의 직접 resume/test를 거부한다. 물리/자세 PD bank는 동일하여 재생성하지 않는다. 기존 실행 중 학습·뷰어에는 자동 반영되지 않는다. 이전 `_cube_low` 출력과 아래 과거 검증 결과는 이전 보상 실행이다.

```bash
# 수정 보상의 scratch 본학습(기존 학습은 자동 종료하지 않음)
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_train.sh 2 2048 4
# 새 보상으로 저장된 checkpoint만 사용
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_test.sh <새_checkpoint.pth> 2 1 4 10
TOKENHSI_GPU=6 PORT=6087 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_vnc.sh <새_checkpoint.pth> 2 1 4 10
```

새 TensorBoard 진단은 `relation/90_debug/push/physics/{credited_forward_m,required_forward_m}`다. 전자는 접촉으로 인정한 reset 이후 거리, 후자는 성공에 필요한 거리이며 사람×control step 평균이다. 기존 `eligible`은 이제 충분한 접촉 전진 거리를 확보한 비율이다. `manipulation_progress`도 접촉 거리 기반 보상 의미로 변경됐다.

현재 PUSH 상자는 **1.3m 정육면체**,20kg·마찰0.6이며 **양손 목표 높이0.8m**다. 깊이0.6m의 높은 상자는 뒤면을 밀 때 바닥에서 미끄러지기보다 기울어지는 물리 비교 결과에 따라 변경했다. 현재 기본 output은 위의 `_hand_credit`이며, 이전 출력은 `output/open_push_unified_amp_cube_low`, `output/open_push_late_rsi_cube_low`다. 이전 크기/목표 checkpoint 직접 resume/test는 계약상 거부된다. 기존 실행 프로세스에는 자동 적용되지 않는다.

PUSH 손 목표는 박스 뒤면의 두 점이다: 오른손 local X=+0.2m, 왼손 X=-0.2m, Y=-depth/2=-0.65m, Z=0.8-height/2=0.15m(바닥 기준0.8m). 목표는 박스 회전/이동을 따른다. 손 상호작용은 `exp(-10*max(d_right²,d_left²))`에 바닥/기울기 조건을 곱한다. 밀기 접촉은 양손 모두 각자의 목표18cm 이내·각 손 전방 반력0.5N 이상·박스 전방 반력0.5N 이상을 같은 physics tick에 만족해야 한다. RSI는 몸통/발 및 팔 길이를 보존하고 양쪽 어깨에서 팔을 회전해 손Z를 목표에 맞춘 뒤 양손 XY를 강체 yaw/이동·6cm stand-off로 정렬한다. 어깨 DOF와 reset AMP 이력도 보정하며 AMP 전문가 원본은 유지한다. 물리적으로 목표에 닿지 못하거나 불안정한 후보는 PD screen에서 제외한다. OPEN의 한 손 조건은 유지한다.

2026-10-09 현재 PUSH18021개에 대한 실제 손 밀기 검사는 [결과와 한계](../output/push180_hand_physics_audit_20261009/REPORT.md)에 있다. 학습 없이2환경에서 외력12조건·전체관절PD189회·초기속도0/다리·몸통유지/팔만재생42회를 정상충돌과 손전용상자충돌로 비교했다. 보행제거·손전용 조건의 원본팔 전진 중앙값2.48cm/최대25.23cm,0.8m보정팔0.048cm/최대7.23cm였다. 손으로 이동은 가능하지만 안정적2m 성공은 확인하지 않았다. 보정팔의 균형·접촉 지속이 후속 확인 대상이며 물성/학습 설정은 변경하지 않았다. 실행 코드·원자료는 같은 output의 `audit.py`, `arms_only.py`, `audit.json`, `arms_only_target/audit.json`이다. 이는 능동 균형 정책의 성공 평가가 아니다. 실제 에이전트PD 비교 영상은 같은 output의 `index.html`이며 현재6090에서 제공한다. VS Code PORTS에서6090을 포워딩하고 `http://localhost:6090/`을 열면 clip0/16·팔만/전체동작·정상/손전용충돌·배속을 선택해 원본과 보정자세를 함께 재생할 수 있다. 서버가 없으면 `python3 -m http.server 6090 --bind 127.0.0.1 --directory output/push180_hand_physics_audit_20261009`로 제공한다.

물리 비교 검사는 다음 명령으로 실행한다. 학습하지 않는2환경 검사이며 YAML/RSI bank를 수정하지 않는다. 기존 상자의 목표 높이1.05/0.8/0.6/0.4m와1.3m 정육면체의1.05/0.8m를20kg·동일 마찰로 비교한다. 외력 검사는 대칭 양손 힘과 같은 합력80/120/160/200N을 뒤면 중앙에2초간 적용하고, 별도 환경은 사람을 멀리 둔 무외력 대조군이다. 이는 힘 높이/형상의 물리 비교이며 정책 성공 검사가 아니다. 사람 PD 검사는 clip별6개 동일 원본 자세를 각2환경에서 최대2초 유지하고 양손 접촉·시작 손 거리·기울기·바닥 edge 전진을 기록한다. 현재 사람 PD 검사는 어깨 회전 RSI를 적용한다. 아래의 이전 비교 결과는 팔 보정을 적용하기 전 원본 손Z를 유지한 검사다. `PHYSICS_FORCE_ONLY=1`로 외력 검사만 실행할 수 있다.

```bash
TOKENHSI_GPU=6 OUTPUT_PATH=output/open_push_physics_comparison_verified bash tokenhsi/scripts/multi_agent/check_open_push_physics.sh
```

외력 분리 비교에서160N을 적용하면 기존 얕은 상자는 목표1.05/0.8/0.6/0.4m 모두 기울어졌고, 정육면체도1.05m에서는 기울어졌다. 정육면체+0.8m는 기울기0·바닥 전진31.56cm를 기록했다(`output/open_push_physics_comparison_isolated/comparison.json`). 무외력 대조군의 최대 전진은 기존 상자0.008mm/정육면체0이었다.같은 설정도200N에서는 넘어졌으므로 모든 힘에서 안정성을 보장하지 않는다. 현재 기본 설정은 이 정육면체/0.8m 조합이며 낮은 손 RSI 팔 보정을 적용했다. 새 학습 정책의 장시간 밀기/AMP 적합성은 별도 확인이 필요하다.

전체 검사24개 외력 조건·120개 사람PD 결과는 `output/open_push_physics_comparison_verified/comparison.json`이다. 원본 자세/1.05m 목표에서 기울기20도 이내·root 높이65cm 이상 구간의 바닥 edge 이동은 기존 상자 최대0.46cm, 정육면체 최대11.89cm였다.0.8m 목표의 첫 step 손 거리 중앙값은 약28cm로 현재18cm 접촉 허용보다 크다. 이 PD는 초기 자세를 고정하므로 목표 변경만으로 팔 동작을 낮추거나 보행/밀기 정책을 학습하지 않는다.

TensorBoard 진단(2026-10-07 추가)은 `relation/90_debug/{open,push}/{current_success,sample_count,distance_xy,xy_progress,manipulation_progress}`다. 담당 task의 전체 rollout 사람×step 표본으로 평균한다. `distance_xy`는 root→문 손잡이 또는 root→박스 중심의 수평 거리(m), `xy_progress`는 OPEN의 `1/(1+max(distance-approach_delta,0)/approach_sigma)`와 PUSH의 `1/(1+max(distance-0.5,0))`다. 현재 OPEN의 delta0.5/sigma1과 PUSH는 같은 식이며 XY거리0.5m 이내에서1로 포화한다. PUSH state의 접근 항도 같은 식에0.2를 곱한다(기존 `1/(1+distance)`에서 변경). `manipulation_progress`는 기존 progress 보상을0.2로 나눈 값으로 문 개방/지지 또는 박스 목표 이동 진행이며 접근 점수와 구분한다. 두 공동 실험(unified/late RSI)에 적용하며 조작 progress·성공·초기화·checkpoint 계약은 유지한다. 기존 실행 프로세스에는 자동 반영되지 않고 새 실행부터 변경 보상을 사용한다.

낮은 손/AMP 충돌 진단은 `relation/90_debug/push/` 아래에 기록한다. 모든 physics 값은 PUSH 담당 사람×control step 평균이며 `sample_count`와 함께 본다.

| 하위 태그 | 확인할 문제 |
| --- | --- |
| `physics/{right_distance,left_distance,worst_hand_distance}` | 손별 목표 거리(m), 한 손만 붙는 문제 |
| `physics/{right_height_error,left_height_error,both_hands_near,paired_contact}` | 목표 대비 손Z 오차(m), 양손 근접/동시 반력 접촉률 |
| `physics/{right_reaction_N,left_reaction_N,box_forward_force_N}` | 마지막 physics tick의 법선 반력(N); paired_contact는 control step 내 같은 tick의 양손 접촉을 수집 |
| `physics/{tilt_rad,tilt_violation,grounded,box_speed}` | 기울기(rad),20도 초과율,바닥/기울기 유효율,속도(m/s) |
| `physics/{box_forward_m,bottom_edge_forward_m,since_reset_forward_m,since_reset_bottom_edge_forward_m}` | 원래 시작/이번 reset 이후 전진(m),기울어짐에 의한 중심 이동과 바닥 edge 전진 구분 |
| `rsi/{right_reset_distance,left_reset_distance,retarget_count}` | 보정된 kinematic reset 손 거리(m)와 PUSH 조작 초기화 수 |
| `reward_balance/{task_reward,amp_reward,weighted_task_reward,weighted_amp_reward,combined_reward,sample_count}` | 실제 PPO task/AMP 보상 및 가중 후 기여; OPEN도 같은 태그 제공 |
| `reward_balance/{paired_contact,both_hands_near}_{amp_reward,task_reward,sample_count}` | 양손 접촉/근접 PUSH 표본에서 AMP와 과제 보상이 경쟁하는지; 표본0의 값0은 성능으로 해석하지 않음 |

바닥 edge 값은 reset의 수직·기본방향 상자를 기준으로 현재 최저 앞/뒤 edge 중점을 추적하는 진단이며 yaw/다른 축 회전에서는 단순 전진 proxy다. `since_reset_*`은 RSI가 박스를 미리 전진시킨 양을 제외한다. 현재 성공 조건/보상에 이 진단 값을 사용하지 않는다.

GPU6·2048환경·MAX_ITERATIONS2(실행 엔진상3epoch)에서 checkpoint 저장과 새 PUSH 진단37개×3회 유한값 기록 확인(`output/open_push_cube_low_train_check_v3/tensorboard_audit.json`). 보정 초기 손 거리 평균 약6.5cm이며 기존28cm 차이를 줄였다. checkpoint2환경/30step headless 복원 및 RSI 뷰어3sample/12step 확인 완료. 이 짧은 실행은 장기 정책 성공/AMP 적합성을 입증하지 않는다.

Actor/Critic의 실제 attention 가산 bias는 `relation_attention/{actor,critic}/edge_bias/{open,push,at}/{mean,rms,sample_count}`로 기록한다. 기존 진단 주기(기본100 forward)에 canonical graph의 유효 edge를 최대32 PPO 표본에서 추출하고 layer/head/edge 전체의 부호 평균·RMS를 계산한다. AT는 PUSH의 박스→목표 관계다. 표본이 없는 관계는 count/mean/rms가0이므로 count와 함께 해석한다. semantic-only 기존5과제도 holding/sit/climb/at/ontop 같은 이름으로 기록한다. 프로세스에 이미 import된 코드에는 자동 적용되지 않고 다음 실행/checkpoint 재개부터 적용한다. 기존 PID의 종료·재시작은 하지 않는다.

사용자 선택에 따라 기존 Unified처럼 사람2명 각각의 task를 매 reset마다 독립 샘플링한다. `relationGraph.shuffle_token_order: true`로 actor/critic의 매 forward마다 H2/O4/G2 전체8개 토큰 순서를 무작위로 섞는다. 토큰·관계 bias·GTA 위치/회전 정보를 함께 순열하고 출력 전에 복원하므로 담당 대상과 사람별 action/value 순서는 유지된다. OPEN50%·PUSH50%이며 OPEN/OPEN·OPEN/PUSH·PUSH/OPEN·PUSH/PUSH 모두 가능하다. 하나의 Actor/Critic·PPO 학습에서 graph가 담당 물체와 task를 알려준다. 기존 Unified와 같은 Actor/Critic별 별도 EdgeEncoder를 사용한다. 기존 HOLDING/SIT/CLIMB/AT/ON_TOP 및 Door Hold 이외의 문 단독 설정은 그대로 둔다.

환경 `HumanoidMAOpenPush`는 사람2·문2·박스2(총물체4)다. 문은 현재 Door Hold의 전용 `door_hold.urdf`·닫힘 spring·마찰0을 재사용하며90도(π/2rad) 이상 개방한 상태에서 손으로5초 지지해야 현재 성공이다. 유지 중 골반 높이는 asset 정상 선 자세의70% 이상, torso 수직축 기울기는40도 이내, 최소 한 발은 바닥 근처에서 위쪽 지지력을 내야 한다. 손 접촉 공백은0.1초까지 허용하고 이탈하면 유지 시간이 초기화된다. PUSH 박스는1.3m 정육면체·20kg·마찰0.6이며 초기 위치에서+Y로1.5m 떨어진 고정 목표까지 민다. 목표XY15cm 이내·초기 바닥 높이±8cm·기울기0.35rad 이내·이동 속도0.2m/s 이하·각속도0.3rad/s 이하가 성공 조건이다. 양손의 같은 physics tick 접촉으로 인정한 전진 거리가 위의 reset 기준 필요 거리를 충족해야 한다. GPU net contact force 기반 공간/힘 판정이며 collision pair/grasp constraint는 아니다. 박스를 직접 움직이는 actuator는 없다.

사람별 초기화는 보행100%·`Start`, 담당 문 손잡이/박스 뒤면에서1.5~2.5m 떨어져+Y를 보도록 한다. 문/박스는 고정된 두 작업 공간에 함께 존재한다. graph는 OPEN의H→Door 또는 PUSH의H→Box 및Box→Goal을 사용한다. semantic packet20·intrinsic O40·총관측684차원이다. task 보상은 state0.4/progress0.2/success0.2와 기존 자기0.9/동료0.1 공유를 유지한다. 성공은 현재 상태이며 성공만으로 episode를 종료하지 않는다.

AMP는 담당 graph task를 따르는 조건부 단일 판별기다. 프레임당 human129+OPEN/PUSH one-hot2=131차원,10frame 총1310차원이다. 기존3-family AMP 기본 동작은 유지하고 label 크기만 선택 가능하게 확장했다. expert/replay는 해당 task의 실제 모션끼리 매칭한다.

| Task | AMP 전문가 | 초기화 |
| --- | --- | --- |
| OPEN | 보행4개33% + 기존 BONES 왼손 문 열기1개67% | 보행4개·Start |
| PUSH | 보행4개33% + 선택한 밀기2개67%(두 clip 동일 weight·각33.5%) | 보행4개·Start |

토큰 섞기 적용 후 CPU28개 통과:4가지 task 조합의 actor/critic 출력·gradient 순열 불변 및 자동 randperm 경로 확인. GPU6 비학습2환경 reset·AMP·actor/critic·4가지 task 조합 시뮬레이션 통과(`output/open_push_shuffle_check_gpu6/audit.json`). shuffle 설정도 checkpoint 계약에 기록한다. 이번 변경에서 학습은 실행하지 않았다.

원래 `open_push_unified_amp`는 loco100%·Start 초기화를 유지한다. 보행50%·조작50%와 후보 뒤쪽30%를70% 확률로 요청하는 별도 실행은 아래 OPEN/PUSH Late RSI 절을 참조한다. 초기화 비율과 AMP 전문가33/67 비율은 별개다.

현재33/67 비율에서 관련 CPU16개 통과(`/tmp/open_push_amp33_cpu.log`); 새 GPU/학습 실행은 하지 않았다. 두 task 모두 보행/조작 전문가를33/67로 맞췄다. OPEN/PUSH50/50 기준 명목 `skillDiscProb`는 `[0.33,0.335,0.335]`(loco/doorOpen/push)다. expert 비율은 checkpoint 계약에 포함되므로 이전 문100%·PUSH30/70 확인용 checkpoint의 직접 resume/test는 거부된다. 새 비율로 scratch 학습한다.

[dataset_open_push.yaml](../tokenhsi/data/dataset_open_push.yaml)은 기존 Door manifest의 보행4개·문1개와 새 밀기2개만 읽는다. 밀기는 `push_obstacle_180_101__A341`의0–12.2초(367frame), `push_obstacle_180_003__A342`의0–11.7초(352frame)를30FPS humanoid15body/32DOF로 변환했다. 준비와 지속 밀기를 남기고 손을 빼는 후반은 제외했다. 원본 두 BVH는 다운로드된 archive에서만 선별 추출했다. source에는 박스 궤적/접촉이 없으므로 human AMP만 사용하며 박스 RSI는 추가하지 않았다. 변환 보고서는 `tokenhsi/data/dataset_bones_push/motions/<clip>/phys_humanoid_v3/conversion_report.json`이다.

GPU는 사용자 지정 **6번만** 사용한다. 정책 test wrapper는 `nvidia-smi`로 물리 GPU 번호를 확인하므로 VNC가 전달하는 GPU6 UUID도 허용하고 CUDA/Vulkan 선택을 유지한다. 새 train/test/VNC/검증 wrapper는GPU6을 기본값으로 하고 다른 숫자를 거부한다. 기존 실행과 checkpoint는 변경하지 않는다. 문/박스 물리·expert manifest·task 비율·초기화·모델 공유 설정을 별도 checkpoint 계약으로 저장하므로 기존 Door/Unified checkpoint와 직접 resume/test할 수 없다. 기본 output은 `output/open_push_unified_amp_cube_low_rsi_progress_hand_credit`다.

```bash
# CPU 모션 변환 재생성: 원본 두 BVH 추출이 완료된 상태
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; python tokenhsi/scripts/multi_agent/convert_bones_push.py'
# 비학습 loader/reset/AMP/actor/critic/4가지 task 조합 검증
TOKENHSI_GPU=6 OUTPUT_PATH=output/open_push_check_gpu6 bash tokenhsi/scripts/multi_agent/validate_open_push.sh
# scratch 본학습: 실행 전 nvidia-smi와 셸의 MAX_ITERATIONS/OUTPUT_PATH/RESUME_CHECKPOINT 확인
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_unified_amp_train.sh 2 2048 4
# 짧은 확인용 학습: 본학습 output과 분리
TOKENHSI_GPU=6 MAX_ITERATIONS=2 OUTPUT_PATH=output/open_push_unified_amp_check bash tokenhsi/scripts/multi_agent/open_push_unified_amp_train.sh 2 2048 4
# 공동 정책 로컬 viewer / 화면 없는 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_unified_amp_test.sh <checkpoint.pth> 2 2 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/open_push_unified_amp_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/open_push_unified_amp_vnc.sh <checkpoint.pth> 2 1 4 10
# 평가에서 조합 고정: open / push / open_push / push_open (기본 random_open_push)
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=open_push bash tokenhsi/scripts/multi_agent/open_push_unified_amp_test.sh <checkpoint.pth> 2 2 4 1
```

초기 전문가 비율(OPEN문100%·PUSH보행30/밀기70%)에서 CPU34개·GPU6 비학습2환경 검사 통과. PUSH 변환 관절 제한 위반0·최대 DOF속도6.95rad/s 미만·end-effector fit 최대1.55cm·발 mesh최소5mm 확인. reset30회에서두 사람 모두 OPEN/PUSH 샘플링 및4가지 조합·partial reset 보존·task별 AMP label·actor/critic/discriminator 유한값 확인. GPU6·2048환경·`MAX_ITERATIONS=2` 확인용 PPO/AMP 실행 및 checkpoint 저장, 같은 checkpoint의2환경/30step headless 평가 복원 완료. 실행 엔진은 max_epochs를 넘는 시점에 멈추므로 이 확인은3epoch까지 진행했다. 장기 본학습·학습된 정책의 성공·GUI/VNC 실행은 미검증이다. 비학습 결과 `output/open_push_check_gpu6/audit.json`, 확인용 checkpoint `output/open_push_unified_amp_check/OpenPushUnifiedAMP_07-16-40-29/nn/OpenPushUnifiedAMP.pth`, 로그 `/tmp/open_push_{cpu_tests_gpu6,sim_gpu6,train_check_gpu6,checkpoint_gpu6}.log`.

90도·자세 조건은 단독 Door Hold와 공동 OPEN에 동일하게 적용된다. 공동 PUSH 보상은 바꾸지 않았다. 자세가 무효이면 OPEN 손 상호작용·각도 진행·유지 진행·성공 보상을 지급하지 않고 접근 보상만 남긴다. 자세 설정은 doorHold와 checkpoint 계약에 포함되므로 이전 조건의 checkpoint는 직접 resume/test 계약과 다르다.

## OPEN/PUSH Late RSI — PD 후보 후반 시작

현재 양손 PUSH bank는 `tokenhsi/data/open_push_late_rsi_cube_low_bank.json`이며 출력 기본값은 `output/open_push_late_rsi_cube_low_rsi_progress_hand_credit`다. GPU6 비학습2환경12step PD 검사에서 OPEN13/13·PUSH145/171개(clip별69/76개) 통과,1000reset 조작48.6%·후반 요청68.72%·부분 reset/고정 목표/보정 AMP history 유한값 확인(`output/open_push_cube_low_rsi_check/audit.json`). 후보 screen은 양손 근접·발 지지/안정성을 검사하며 양손 접촉력을 지속적으로 내는 정책 성공 검증은 아니다. 후보 계약revision3에 어깨 회전 retarget 방식을 포함하며 이전 크기/한 손/원본 손Z bank는 직접 사용할 수 없다. 이전 양손1.05m bank `open_push_late_rsi_two_hand_bank.json`와166개 통과 결과는 역사 artifact로 남긴다.

1.3m 높이만 변경했던 당시 bank는 `tokenhsi/data/open_push_late_rsi_h13_bank.json`이다. 재검증에서 OPEN13·PUSH155개 통과,1000reset 조작48.6%·후반 요청68.72%,partial reset/고정 목표/AMP 통과(`output/open_push_late_rsi_h13_check/audit.json`). 이전1.1m bank·확인용 checkpoint는 역사 artifact로 남기며 현재 크기 계약에는 사용할 수 없다. 당시 RSI 목표 손 높이는 실제 뒤면의 바닥+5cm~상단-5cm 범위로 clamp했다. 현재는 양손 목표높이0.8m와 어깨 회전 RSI를 쓴다. 기존0.55m 고정 중심 높이를 사용하던 PD 감사도 config의 절반 높이를 따르도록 수정했다.

사람·문·박스의 RSI 초기화를 눈으로 확인할 때는 checkpoint 없는 전용 검사 뷰어를 쓴다. 기본 `late` 모드는 사람0 OPEN/사람1 PUSH를 조작100%·후반 요청100%로 표시한다(검사 프로세스에서만 강제하며 학습 YAML/bank는 수정하지 않음). 모션clip/time/원본phase/후반 요청 여부가 화면에 표시되고 `latest_sample.json`·`samples.jsonl`에 기록된다. 한 physics frame 뒤 정지하며 **학습된 정책 행동이 아니다**. Space는 sampled DOF를PD로 최대12step 유지한다. 성공/장시간 안정성 확인과 구분한다.

PUSH 양손 목표는 오른손 빨강·왼손 청록 cross로 표시한다. `RSI_VIEW_FOCUS=1`로 PUSH 사람 카메라에서 시작할 수 있다.

키: `N` 새 샘플, `Space`12step PD 실행/일시정지, `C` 다른 사람 카메라, `L` 후반 조작 강제, `M` 학습의50/50·후반70% 혼합, `O` OPEN/OPEN, `P` PUSH/PUSH, `B` OPEN/PUSH. 카메라는 마우스로 조정할 수 있다. 일반 `open_push_late_rsi_test/vnc.sh`는 보행 시작 정책 평가라서 RSI 자세 검사용으로 쓰지 않는다.

```bash
# 로컬 RSI 초기화 검사(1환경, checkpoint 불필요)
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_rsi_view.sh
# 서버 VNC: VS Code PORTS에서6086 포워딩 후 localhost:6086/vnc.html 접속
TOKENHSI_GPU=6 PORT=6086 bash tokenhsi/scripts/multi_agent/open_push_rsi_view_vnc.sh
# 화면 없는3sample/12step PD 검사
TOKENHSI_GPU=6 HEADLESS=1 RSI_VIEW_CYCLES=3 bash tokenhsi/scripts/multi_agent/open_push_rsi_view.sh
```

검증 결과: CPU35개 통과. GPU6·비학습2환경에서 문13개·PUSH155개(clip별72/83)의12step PD 후보를 확보했다. 문 통과 후보의 원본phase는 약0.186~0.657이며 마지막30% 구간이 아니라 이 **유효 후보의 뒤쪽**을 우선 선택한다. PUSH는 원본 후반phase도 포함한다. 1000reset 실측 조작48.9%·조작 중 후반 요청69.33% 및4과제 조합·partial reset·고정 목표·AMP history 검증 통과(`output/open_push_late_rsi_reset_check/audit.json`). GPU6·2048환경 확인용3epoch 학습·checkpoint 저장·2환경30step 보행 시작 평가 복원 및XY/edge bias/RSI TensorBoard 생성 확인. 정책 성공·장기 학습·GUI/VNC는 미검증이다. 본학습은 실행하지 않았다.

`open_push_late_rsi.yaml`은 기존 공동 과제의 보상·90도/자세/5초 유지·AMP33/67·토큰 순열·모델을 유지한다. 예전 task별 보행/조작50/50 혼합과 Door/ksh 후보 후반 우선 방식을 참고해 사람별 OPEN/PUSH task에 맞는 초기화를 적용한다.

| Task | 학습 초기화 | 조작 후보 시간 선택 |
| --- | --- | --- |
| OPEN | 보행50%·doorOpen50% | 각 clip의 PD 통과 후보 뒤쪽30%를70% 확률로 요청; 나머지30%는 전체 통과 후보 |
| PUSH | 보행50%·push clip1 25%·clip2 25% | 각 clip 동일 비중, 위와 같은 후반 우선 선택 |

전체 초기화 중 후반 요청은35%다. 전체 후보에서 고르는15%에도 후반이 포함될 수 있어 실제 후반 선택 비율은35%보다 높을 수 있다. 후보 뒤쪽30%는 원본 모션의 마지막30%와 다르다. 보행은 기존0초 Start·1.5~2.5m 시작을 유지한다. 평가/test/VNC는 학습 RSI를 끄고 보행100% Start로 접근부터 평가한다.

사람 reference에 맞춰 OPEN은 문각도 `0.4+1.2*원본phase`rad의 손잡이 앞6cm에 손을 정렬하고 사람 전체 위치/회전/속도를 함께 변환한다. PUSH는 고정1.5m 목표 거리 중 `0.85*원본phase`만큼 박스를 미리 전진시키고 뒤면의 양손 목표점 앞6cm에 손 pair XY를 정렬한다. 원본에 물체 궤적이 없어 이 각도/위치는 **기하 proxy**이며 BONES에서 측정된 문/박스 상태가 아니다. 목표와 reset 기준 박스 위치는 이동 전 원래 상태를 유지한다. 문 유지 시간·접촉 유예/이력·박스 성공 eligibility는0부터 시작한다. 기존 성공 조건을 통과하는 실제 접촉/5초 유지가 필요하다.

`build_open_push_rsi.sh`는 원본모션 파일 hash·문/박스 물리·보상/어깨 retarget 방식에 묶인 `tokenhsi/data/open_push_late_rsi_cube_low_bank.json`을 만든다. geometry/AMP history 후보를12 control step(약0.4초)의 고정 reference DOF PD로 검사해 유한값·넘어짐/높이·root drift45cm 미만·발 지지·OPEN 손잡이20cm/PUSH 양손 목표 각각18cm 접근·PUSH 바닥 높이를 확인한다. OPEN은 현재 문 선 자세 판정으로 지지를 확인하고 PUSH는 별도의 foot body35cm/위쪽힘1N 기준을 쓴다. 각 clip에 통과 후보가 없으면 실행을 거부한다. 이는 짧은 reset 안정성 검사이며 접촉 유지·학습된 정책 성공을 보장하지 않는다.

기존 실행용 env는 변경하지 않았다. 새 env/output·RSI spec/후보 bank hash가 checkpoint 계약에 추가되므로 원래 loco-only checkpoint를 이 설정으로 직접 resume/test할 수 없다. 새 scratch 학습이며 기존 실행 PID를 종료/재시작하지 않는다. train/network YAML은 기존 공동 실험의 것을 재사용한다. 새 학습에서도 앞서 추가한 과제별 XY 접근·edge bias 로그가 기록된다. `relation/90_debug/{open,push}/rsi/{reset_count,interaction_count,late_request_count}`는 rollout 간 reset 표본 수다.

```bash
# 실행 전 GPU6 점유 확인; 기존 학습 유지
nvidia-smi
# 비학습2환경 후보 생성/PD검사·학습 reset 경로/AMP 감사
TOKENHSI_GPU=6 OUTPUT_PATH=output/open_push_late_rsi_check bash tokenhsi/scripts/multi_agent/build_open_push_rsi.sh
# 새 scratch 본학습(자동 실행하지 않음)
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_train.sh 2 2048 4
# 별도 짧은2048환경 학습 확인
TOKENHSI_GPU=6 MAX_ITERATIONS=2 OUTPUT_PATH=output/open_push_late_rsi_train_check bash tokenhsi/scripts/multi_agent/open_push_late_rsi_train.sh 2 2048 4
# 로컬 viewer / 화면 없는 평가 / 서버 VNC: 보행 Start로 평가
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_test.sh <checkpoint.pth> 2 2 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/open_push_late_rsi_vnc.sh <checkpoint.pth> 2 1 4 10
```

## Door Open/Hold — 1인 밀기·지지

`HumanoidMADoorHold`는 사람1·문1의 별도 과제다. 90도(π/2rad) 이상 열고 자기 손을 손잡이12cm 이내에 두며 문을 밀어 받치는 접촉이5초 이어지면 현재 성공으로 판정한다. 접촉에는 손의 반력과 문짝의 개방 방향 힘이 각각0.5N 이상 필요하다. 손과 발 지지력을 각 physics tick에서 확인한다. 발은 해당 control step 내의 접촉 충격을 수집하고 다음 step으로 이월하지 않는다. 손은 최대0.1초의 짧은 접촉 공백을 허용한다. 손이 멀어지거나 공백이 길어지면 타이머와 현재 성공을 초기화한다. 성공 후에도 지지를 유지해야 한다. 추가 자세 조건은 골반 높이≥asset 정상 선 자세의70%, torso의 수직축 기울기≤40도, 최소 한 발의 body 높이≤18cm 및 위쪽 접촉력≥1N이다. 정상 높이는 reset/RSI 자세가 아닌 고정 asset 초기 자세에서 가져온다. 자세 조건이 깨지면 유지 시간을 즉시0으로 초기화하고 손 상호작용·열기 진행 보상을0으로 만든다. 접근 보상은 남긴다. 양손 중 어느 손이든 인정하며 몸통-문 접촉 금지·통과 판정·추가 넘어짐 종료는 적용하지 않았다. target_angle=π/2, angle_tolerance=0으로90도 미만 성공을 허용하지 않는다. 손가락 grasp constraint는 없으며 agent2의 운반/통과도 아직 없다.

전용 `door_hold.urdf`는 바닥과 문짝 사이4cm 틈을 두어 바닥 마찰에 걸리지 않도록 했다. 손잡이는 높이1m이며 기존 push 문 방향을 유지한다. 경첩 마찰0·감쇠0.5에 환경 닫힘 토크 `clamp(-2*angle-0.5*velocity, -4, 4)`를 적용한다. 정책은 사람 관절32개만 제어한다. 기존 문 asset/실험의 경첩 설정은 보존한다.

관측은288차원이며 물체 특징O38에 지지 시간·접촉 상태·마지막 접촉 후 시간이 포함된다. 사용자 요청으로 AMP 전문가를 BONES 왼손 open/walk100%(loco0%)로 변경했다. `skillDiscProb`와 `ampExpertMapping.OPEN`은 둘 다 `[0,1]`이다. RSI는 loco-only·`Start`를 유지하며 닫힌 문 손잡이 앞1.5~2.5m에서 매 reset마다 균등하게 거리를 뽑는다. 학습·평가 모두 적용하며 `doorSpawnDistanceRange: [1.5,2.5]`가 범위, `doorApproachAMP.spawn_distance: 2.5`는 재배치 기준인 상한이다. 문을 향한 방향·좌우 정렬은 유지한다. loco 파일은 초기화용으로 manifest에 남긴다. 긴 유지 동작의 전문가를 추가한 상태는 아니다. 새로운 관측·인원수·물리 계약 때문에 기존 checkpoint를 직접 사용할 수 없고 별도 output에서 scratch 학습해야 한다. 비율·시작 거리 범위도 계약에 포함되어 이전 고정 거리/30·70 설정 checkpoint와 직접 resume/test할 수 없다. 학습2048환경; 아래2환경 검사는 물리·loader 검증이며 학습된 정책의 성공을 의미하지 않는다.

```bash
# 비학습 검증: GPU6 / CPU 실제 손-문 collision pair
TOKENHSI_GPU=6 OUTPUT_PATH=output/door_hold_check_gpu bash tokenhsi/scripts/multi_agent/validate_door_hold.sh
TOKENHSI_GPU=6 AUDIT_SIM=cpu OUTPUT_PATH=output/door_hold_check_cpu bash tokenhsi/scripts/multi_agent/validate_door_hold.sh
# scratch 학습 (현재 실행하지 않음)
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/door_open_hold_amp_train.sh 1 2048 1
# 학습 후 로컬 viewer / 화면 없는 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_hold_amp_test.sh <hold_checkpoint.pth> 1 1 1 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_hold_amp_test.sh <hold_checkpoint.pth> 1 16 1 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/door_open_hold_amp_vnc.sh <hold_checkpoint.pth> 1 1 1 10
```

기본 output은 `output/door_open_hold_amp`다. 유지 시간은 env YAML의 `doorHold.duration`으로 바꾼다. 실행 전 GPU6 점유 및 `MAX_ITERATIONS/OUTPUT_PATH/RESUME_CHECKPOINT` 잔여값을 확인한다. 검증 결과는 `output/door_hold_check_{cpu,gpu}/physics_audit.json`을 확인한다.

## BONES Door — 왼손·왼쪽 개방 AMP

`door_open_bones_left_amp`는 기존 분리형 `door_open_unified_approach_amp`의 Door/PPO/reward·닫힌 문·1.5m loco reset을 유지한다. OPEN 전문가만 새 [dataset_door_bones_left.yaml](../tokenhsi/data/dataset_door_bones_left.yaml)의 loco30%/doorOpen70%로 바꿨다. `skill=[loco,doorOpen]`, 초기화 확률 `[1,0]`, expert `[0.3,0.7]`, gate 비활성/full AMP다. 비율은 최초 설정이며 학습으로 검증한 최적값이 아니다. 기존 YAML·output·checkpoint는 보존하고 새 manifest hash 계약으로 이전 loco/proxy checkpoint의 직접 resume/test는 거부한다.

원본 `inside_door_handle_right_side_open_walk_R_001__A515_M.bvh`는 **왼손→왼쪽** 반전 motion이다. Uniform77관절→phys_humanoid_v3의15body/32DOF, cm/Y-up→m/Z-up, constrained IK·손 고정·발 mesh grounding을 거쳐0–3.4초/30FPS/103frame으로 변환했다. 원본 full passage 대신 reach/open/초반 전진을 남겼다. source에는 문 위치/각도/접촉이 없으며 **human AMP reference로만 사용**한다. 문 상태는 simulator와 OPEN reward가 제공하며 source 문 RSI/그립 constraint를 새로 만들지 않았다.

CPU40개·GPU6 비학습2환경 loader/AMP64×1290·FK·52frame 재생 통과. 무접촉 Gym/Poselib 차이최대2.2mm, 문을 겹친 replay 차이최대9.5cm이다. CPU [문 비교 HTML](../tokenhsi/data/dataset_bones_door/motions/inside_door_handle_right_side_open_walk_R_001__A515_M/phys_humanoid_v3/door_preview.html)·GIF와 [검증 보고서](../output/door_bones_left_amp_check/REPORT.md)를 먼저 확인한다. preview의 문 각도는 손 방향으로 추정한 것으로 전문가/접촉 정답에 포함하지 않는다. GPU6 offscreen graphics 초기화는 실패했고 GPU0으로 대체하지 않았다. 아래 학습·policy test/VNC wrapper는 준비했으나 본학습·정책 성공·GUI 실행은 미검증이다.

```bash
# 원본 추출 완료 상태에서 CPU 변환 / 문 형상 비교 미리보기 재생성
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; python tokenhsi/scripts/multi_agent/convert_bones_door.py'
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; python tokenhsi/scripts/multi_agent/build_bones_door_preview.py'
# GPU6 전용 비학습 검증. CAPTURE_BONES는 지정하지 않는다.
TOKENHSI_GPU=6 OUTPUT_PATH=output/door_bones_left_amp_check bash tokenhsi/scripts/multi_agent/validate_bones_door_amp.sh
# 본학습: 별도 output, 2048환경, scratch. 이번 작업에서는 실행하지 않음.
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/door_open_bones_left_amp_train.sh 2 2048 2
# 해당 실험에서 새로 학습한 checkpoint만 사용
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_bones_left_amp_test.sh <bones_checkpoint.pth> 2 1 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_bones_left_amp_test.sh <bones_checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/door_open_bones_left_amp_vnc.sh <bones_checkpoint.pth> 2 1 2 10
```

기본 output은 `output/door_open_bones_left_amp`다. 실행 전에 `nvidia-smi`와 `MAX_ITERATIONS/OUTPUT_PATH/RESUME_CHECKPOINT` 잔여값을 확인한다. 변환 자체는 CPU이며 runtime을 통해 GPU6만 노출한다. 공유 원본 데이터에는 쓰지 않는다.

## Door unified / shared-edge — 같은 Door proxy AMP·RSI 비교

사용자 최종 선택은 기존 `approach_scenario_stage1_unified`와 `unified_shared_edge_encoder`의 비교다. 기존 unified는 **Actor/Critic별 별도 semantic EdgeEncoder**, shared-edge는 **같은 EdgeEncoder 파라미터 공유**다. 두 방식 모두 tokenizer·GTA Transformer·출력 head는 Actor/Critic별 별도다. ksh size RSI의 기본 semantic network도 별도 encoder이지만 이번 비교에 ksh 독립 코드나 typed bias/message를 추가하지 않는다.

새 `door_open_unified_amp_on` / `door_open_shared_edge_amp_on`의 env YAML은 동일하며, train YAML은 `transformer.share_edge_encoder: false/true`만 다르다. Door observation/reward/success·PPO·AMP·RSI는 동일하다. 기존 Door `amp_on/off` 실험은 유지한다.

실제 Door reference dataset이 없는 상태에서 [dataset_door_proxy.yaml](../tokenhsi/data/dataset_door_proxy.yaml)은 기존 loco·pickUp·carryWith 파일만 읽는다. **이들은 box 동작에서 고른 대체 참조이며 실제 문 열기 전문가가 아니다.** sit/climb/putDown·OMOMO 혼합은 새 두 실험에서 제외한다. AMP 전문가 비율은 loco/pickUp/carryWith=20/70/10%, RSI 초기화는30/60/10%다. `doorReference`는 pickUp/carryWith의 root 높이≥0.85m, root XY 속도≤0.8m/s, 양발 높이≤0.2m, 적어도 한 손의 높이1.0±0.25m·root heading 앞방향 reach≥0.2m를 검사한다. 전체 10프레임 AMP history가 조건을 만족하는 후보만 사용한다. 현재 참조 후보는 pickUp59·carryWith22개다.

RSI는 같은 후보에서 각 clip의 유효 후보 뒤쪽30%를70% 확률로 요청한다. 이는 ksh의 후반 후보 우선 방식에 맞춘 것이며, box 크기별 PD 후보 bank를 Door에 그대로 적용한 것은 아니다. Human을 +Y로 향하게 회전시키고 조건에 맞는 손을 **자기 handle XY에서 앞쪽8cm**에 정렬한다. root/손 높이는 원본을 유지하고 몸·root의 위치/회전/속도를 일관되게 변환한다. 문은 닫힌 상태로 별도 reset하며 binding은 바꾸지 않는다. loco 시작은 자기 문 앞에서 기존 loco 상태를 사용한다. 평가는 기존처럼 loco 시작이며 학습 RSI 시작 성공률과 구분한다.

```bash
# 동일 seed·2 humans·2048 학습환경·2 doors. 이번 작업에서는 실행하지 않음.
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/door_open_unified_amp_on_train.sh 2 2048 2
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/door_open_shared_edge_amp_on_train.sh 2 2048 2

# 로컬 viewer / headless 평가
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_unified_amp_on_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_unified_amp_on_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_shared_edge_amp_on_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_shared_edge_amp_on_test.sh <checkpoint.pth> 2 16 2 3

# 서버 VNC
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/door_open_unified_amp_on_vnc.sh <checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/door_open_shared_edge_amp_on_vnc.sh <checkpoint.pth> 2 1 2 10
```

학습 전 `nvidia-smi`로 GPU6 점유와 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 확인한다. 기본 output은 `output/door_open_unified_amp_on`, `output/door_open_shared_edge_amp_on`이다. 같은 방식·같은 Door reference 계약의 checkpoint로만 resume/test한다. checkpoint에는 공유 여부·reference 조건·expert/RSI 비율·motion manifest SHA256을 기록해 혼용을 거부한다. 추후 실제 Door motion을 기존 MotionLib 형식으로 등록하고 manifest·expert row·reference 조건을 교체할 수 있다. TRUMANS 원본과 기존 ksh 이식본은 수정하지 않았다. 긴 학습의 문 열기 성능 개선은 아직 확인하지 않았다.

비학습 CPU physics check는 아래 Door ON/OFF 섹션의 명령에서 env/train YAML을 위 두 파일로, motion 파일을 `tokenhsi/data/dataset_door_proxy.yaml`로 바꾸면 된다. 새 reference 설정에서는 실제 interaction RSI 경로도 검사한다. CPU 회귀는 기존 Door 테스트와 `tokenhsi/tests/test_door_reference.py`를 함께 실행한다.

## Door 당김 방향 — 별도 approach AMP 설정

`door_pull_unified_approach_amp`는 기존 approach-only에서 경첩 축을−Z로 반전하고 handle offset을 `(0.78, -0.25, 1)`로 변경한다. 사람은 손잡이 앞1.5m·+Y 방향이며 문은 사람 쪽(−Y)으로 열린다. DOF 양수는 개방량으로 관측/reward에 사용하고 panel 물리/초기 관측 회전만 부호를 반영한다. 기존 밀기 설정과 checkpoint/output은 분리하며 밀기 checkpoint의 resume/평가는 거부한다.

물리 검증은 CPU/GPU 각2평가환경·한 코어·1스레드로 통과했다. 실제 손–손잡이 접촉, 무접촉20초, 접촉 해제4초, 관측/전체 reward 대조 결과는 [검증 보고서](../output/door_pull_physics_audit_20261006/REPORT.md)에 있다. 손은 sphere이고 grasp 제약은 없어서 스크립트 접촉으로 당김 방향의 힘 전달을 확인한 것이며 정책 성공/실제 grasp 검증은 아니다. 이 설정의 학습은 실행하지 않았다.

```bash
# 물리 검증만(각각 한 코어·1스레드, 학습 없음)
TOKENHSI_GPU=6 AUDIT_SIM=cpu DOOR_EXPERIMENT=door_pull_unified_approach_amp OUTPUT_PATH=output/door_pull_physics_check_cpu bash tokenhsi/scripts/multi_agent/validate_door_physics.sh
TOKENHSI_GPU=6 AUDIT_SIM=gpu DOOR_EXPERIMENT=door_pull_unified_approach_amp OUTPUT_PATH=output/door_pull_physics_check_gpu bash tokenhsi/scripts/multi_agent/validate_door_physics.sh
# 본학습(2048환경, scratch; 이번 작업에서는 실행하지 않음)
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/door_pull_unified_approach_amp_train.sh 2 2048 2
# 로컬 viewer / 화면 없는 평가
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_pull_unified_approach_amp_test.sh <pull_checkpoint.pth> 2 1 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_pull_unified_approach_amp_test.sh <pull_checkpoint.pth> 2 16 2 3
# 서버 VNC
TOKENHSI_GPU=6 PORT=6086 bash tokenhsi/scripts/multi_agent/door_pull_unified_approach_amp_vnc.sh <pull_checkpoint.pth> 2 1 2 3
```

## Door Open — 기존 unified + 전 구간 loco AMP

### Door 추가 물리 검증 (학습 없음)

아래 명령은 평가환경2개·CPU 한 코어 affinity·PyTorch/PhysX/BLAS 각1스레드로 제한한다. 무접촉20초 안정성, 스크립트 사람 접촉으로 passive hinge 개방, 물리/관측/전체 reward 일치를 검사한다. CPU에서는 실제 손–문 contact pair도 확인한다. 사람을 스크립트로 움직이므로 학습 정책 성공률 검증과 구분한다.

```bash
TOKENHSI_GPU=6 AUDIT_SIM=gpu OUTPUT_PATH=output/door_physics_check_gpu bash tokenhsi/scripts/multi_agent/validate_door_physics.sh
TOKENHSI_GPU=6 AUDIT_SIM=cpu OUTPUT_PATH=output/door_physics_check_cpu bash tokenhsi/scripts/multi_agent/validate_door_physics.sh
```

CPU/GPU 검증 통과 결과: [검증 보고서](../output/door_physics_audit_20261006/REPORT.md). 장기 학습 실행과 분리되며 checkpoint가 필요 없다.

### 2026-10-07 공유/분리 비교 — 같은 HOLDING 계수·전 구간 AMP

`door_open_unified_approach_amp`는 Actor/Critic EdgeEncoder **분리**(`share_edge_encoder: false`), 새 `door_open_shared_edge_approach_amp`는 **공유**(`true`)다. 환경 YAML은 같고 train YAML은 공유 플래그·실험 name만 다르다. 두 설정 모두 몸/손0.2·각도progress/success0.2·gate_enabled=false·loco expert/RSI·reset revision3·Start·task/disc0.5/0.5·PPO를 사용한다. output/checkpoint는 분리하고 공유 여부 계약이 다른 checkpoint의 resume/test는 거부한다. 아래 명령은 준비만 했으며 본학습은 실행하지 않았다.

```bash
# Actor/Critic EdgeEncoder 분리 — 터미널1
unset MAX_ITERATIONS RESUME_CHECKPOINT OUTPUT_PATH
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SEED=42 OUTPUT_PATH=output/door_separate_full_amp_balanced_hinge bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
# Actor/Critic EdgeEncoder 공유 — 터미널2
unset MAX_ITERATIONS RESUME_CHECKPOINT OUTPUT_PATH
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SEED=42 OUTPUT_PATH=output/door_shared_full_amp_balanced_hinge bash tokenhsi/scripts/multi_agent/door_open_shared_edge_approach_amp_train.sh 2 2048 2
# 로컬 viewer / 화면 없는 평가
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <separate_checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <separate_checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_shared_edge_approach_amp_test.sh <shared_checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_shared_edge_approach_amp_test.sh <shared_checkpoint.pth> 2 16 2 3
# 서버 VNC (포트 별도)
TOKENHSI_GPU=6 PORT=6086 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_vnc.sh <separate_checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=6 PORT=6087 bash tokenhsi/scripts/multi_agent/door_open_shared_edge_approach_amp_vnc.sh <shared_checkpoint.pth> 2 1 2 10
```

기본 shared output은 `output/door_open_shared_edge_approach_amp_holding_scale_full_amp_balanced_hinge`다. 기존 `door_open_shared_edge_amp_on`은 box proxy AMP/RSI 실험이므로 이번 loco-only 비교와 구분한다.

### 2026-10-07 현재 설정 — HOLDING 접근 계수·전 구간 AMP

사용자 요청으로 GPU6의 `reset_v2_run1/run2` 학습 두 개를 종료했다. 각 마지막 저장 checkpoint는 epoch10500이며 본학습은 재시작하지 않았다. 현재 미는 `door_open_unified_approach_amp`의 `state_reward_weight`는0.4로, 몸 접근·손 상호작용은 각각0.1→0.2이다. `doorApproachAMP.gate_enabled: false`로 거리 taper·손접촉/각도 latch를 비활성화해 에피소드 내내 AMP 가중치1을 사용한다. 접근 delta0.5/sigma1.0·손 거리 scale10·각도 progress0.2·success0.2·sharing0.9/0.1·AMP task/disc0.5/0.5·PPO/reset은 유지한다. 성공 시 edge 최대 보상은0.8이다. 다른 Door/당김·기존 HOLDING 설정은 유지한다.

기본 output은 `output/door_open_unified_approach_amp_holding_scale_full_amp_balanced_hinge`로 분리했다. 보상/gate 계약이 달라 기존0.1 또는 gate 활성 checkpoint는 현재 wrapper의 resume/test에서 거부한다. 아래는 새 계수·전 구간 AMP로 scratch 학습할 때의 명령이며 **이번 작업에서 실행하지 않았다**. 기존 checkpoint를 자동 변환하거나 덮어쓰지 않는다.

```bash
unset MAX_ITERATIONS RESUME_CHECKPOINT OUTPUT_PATH
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SEED=42 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
# 새 보상 계약의 checkpoint만 사용: 로컬 viewer / 화면 없는 평가 / 서버 VNC
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <new_checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <new_checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 PORT=6086 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_vnc.sh <new_checkpoint.pth> 2 1 2 10
```

gate 수정 전 접근 정체 진단은 `output/door_approach_debug_20261007/REPORT.md`를 참조한다. 당시 기존 checkpoint를 저장된 원래 reward config로 비학습 평가했다. 이후 사용자 요청으로 거리 기반 AMP 보상 감소를 제거했으며, 전 구간 AMP에서의 학습 성능은 미검증이다. 새 설정의 물리/보상 검증은 `output/door_full_amp_check_20261007/`에 저장한다.

### 2026-10-06 reset·학습 이력 (이하 기존0.1 계수의 기록)

**아래 재개·평가 명령은 당시 이력이다. 현재0.2 계수 wrapper로 기존 checkpoint를 resume/test할 수 없다. 새 실행에는 위 현재 설정 명령을 사용한다.**

2026-10-06 미는 문 두 실행을 각각 epoch3000/frame393216000 checkpoint에서 resume했다. 원본 checkpoint는 보존하고 같은 run1/run2 output 아래 새 시간 디렉터리에 로그·checkpoint를 저장한다. Actor/Critic·AMP/discriminator·normalizer·optimizer·epoch/frame을 기존 restore 경로로 복원한다. 당김 asset/grasp는 사용하지 않는다. epoch3008 이후 진행을 확인했다. 운영 기록은 각 `latest_launch.json`, 검증/launch 기록은 `output/door_push_resume_20261006/`에 있다.

아래는 중단 후 재개용 명령이다. 같은 실행이 살아 있으면 중복 실행하지 않는다. 이후에는 각 실행의 가장 최신 checkpoint를 지정한다. CPU 사용을 줄이기 위해 OMP/MKL/OpenBLAS/NumExpr 연산 스레드를1로 제한하고 학습환경 수2048은 유지한다.

```bash
unset MAX_ITERATIONS
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SEED=42 OUTPUT_PATH=output/door_open_unified_approach_amp_reset_v2_run1 RESUME_CHECKPOINT=output/door_open_unified_approach_amp_reset_v2_run1/DoorOpenUnifiedApproachAMP_06-12-22-51/nn/DoorOpenUnifiedApproachAMP_00003000.pth bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
# 별도 터미널
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SEED=42 OUTPUT_PATH=output/door_open_unified_approach_amp_reset_v2_run2 RESUME_CHECKPOINT=output/door_open_unified_approach_amp_reset_v2_run2/DoorOpenUnifiedApproachAMP_06-12-22-51/nn/DoorOpenUnifiedApproachAMP_00003000.pth bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
```



현재 reset 수정본은 `stateInit: Start`로 네 loco clip의 시작 프레임을 사용한다. 무작위 시점 참조 유지 검사에서 일부 첫-step 물리 불안정성을 확인했고, 시작 프레임은 GPU2048환경×10회 모두 통과했다. own handle 앞1.5m·+Y 방향·env-local 관측, root/DOF 단일 setter·reset 중 즉시 refresh 제거·GPU index tensor 수명 보존을 적용한다. checkpoint `reset_revision: 3`와 `state_init: Start` 계약으로 수정 전 checkpoint의 평가/resume를 거부한다.

최종 재학습은 아래 별도 output에서 scratch로 실행한다(각 터미널에서 한 번씩):

```bash
unset MAX_ITERATIONS RESUME_CHECKPOINT
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi SEED=42 OUTPUT_PATH=output/door_open_unified_approach_amp_reset_v2_run1 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
# 별도 터미널
unset MAX_ITERATIONS RESUME_CHECKPOINT
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi SEED=42 OUTPUT_PATH=output/door_open_unified_approach_amp_reset_v2_run2 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
# 로컬 추론 / 서버 VNC (각 run의 수정 후 checkpoint)
TOKENHSI_GPU=6 HEADLESS=0 OUTPUT_PATH=output/door_reset_v2_eval bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 PORT=6086 OUTPUT_PATH=output/door_reset_v2_eval bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_vnc.sh <checkpoint.pth> 2 1 2 10
```

운영 실행 기록은 각 output의 `latest_launch.json`·`launches/<날짜>/train.log`에 저장한다. 같은 실행이 살아 있으면 위 명령을 다시 실행하지 않는다. 검증 스크립트·로그는 `output/door_reset_v2_validation_20261006/`; 확인용 output은 `output/door_reset_v2_check`다.

2026-10-06 추가 수정: approach-only loco RSI의 사람 pose를 env-local own handle 앞1.5m로 재배치하고 +Y 방향으로 일관되게 회전한다. 이전에는 사람에만 arena origin이 더해져 다중환경에서 문과 멀어졌으며 단일환경 viewer로는 검출되지 않았다. 이 수정 전 시작한 `reset_fixed_run1/run2`에도 해당하므로 결과를 정상 학습 성능으로 해석하지 않는다. 새 프로세스부터 적용되며 기존 실행은 자동 갱신되지 않는다. GPU16환경 reset 검증 통과·수정 후 학습 성능은 미검증이다.

수정된 GPU reset으로 기존 두 실행과 같은 설정(seed42·2048환경·2Human/2Door)을 scratch 재학습하려면 서로 다른 터미널에서 아래 명령을 실행한다. 두 실행은 동일 설정 반복이며 ON/OFF 비교가 아니다. 이전 checkpoint는 resume하지 않는다.

```bash
# 터미널1
unset MAX_ITERATIONS RESUME_CHECKPOINT
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi SEED=42 OUTPUT_PATH=output/door_open_unified_approach_amp_reset_fixed_run1 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
# 터미널2
unset MAX_ITERATIONS RESUME_CHECKPOINT
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi SEED=42 OUTPUT_PATH=output/door_open_unified_approach_amp_reset_fixed_run2 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
```


환경 reset만 확인하는 비학습 뷰어(체크포인트 불필요):

```bash
# 서버 VNC (6086 포트 전달 후 localhost:6086/vnc.html 접속)
TOKENHSI_GPU=6 PORT=6086 bash tokenhsi/scripts/multi_agent/door_reset_vnc.sh
# 로컬 화면
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_reset_view.sh
# GUI 없이 GPU reset 5회 확인
TOKENHSI_GPU=6 HEADLESS=1 DOOR_RESET_CYCLES=5 bash tokenhsi/scripts/multi_agent/door_reset_view.sh
```

실제 Door reset 후 참조 관절 자세를 PD target으로 1물리 step 적용하고, 그 화면을3초 유지한 뒤 재초기화한다. 학습 policy를 사용하지 않는다. terminal의 `DOOR_RESET_CHECK`는 reset 전후 root quaternion 유사도(1에 가까움)·골반 높이·문 각도를 출력한다. 초기 자세 적용 확인용이며 걷기 안정성·문열기 정책 성능 검증은 아니다. GPU6/1환경5회 headless 확인 통과. VNC에서 실제 viewer 생성·렌더링·reset2회 정상 종료를 확인했다. 로컬 viewer에는 DISPLAY가 필요하며, 없으면 안내와 함께 종료한다.

2026-10-06 GPU reset 진단: 사람과 Door의 indexed root/DOF setter를 통합했다. 이전 코드로 실행한 Door 학습은 사람 RSI root reset이 GPU 첫 step에 반영되지 않는 현상을 재현했다. 기존 실행에는 수정이 자동 적용되지 않으며, 이전 학습 current-success 지표를 정상 정책 성능으로 해석하지 않는다. 기존 5500 checkpoint는 수정 후 2환경 평가에서도 성공0·조기종료100%였다. 정상 reset에서의 재학습 성능은 미검증이다.

`door_open_unified_approach_amp`는 기존 unified의 Actor/Critic별 EdgeEncoder와 MAAgent PPO를 그대로 쓴다. H0→Door0/H1→Door1, Door 관측·성공 조건은 기존 Door 설정과 같으며 현재 몸 접근·손 상호작용 계수는 각각0.2이다. 실제 문 모션은 사용하지 않는다. `dataset_door_loco.yaml`의 기존 stand-to-walk·walk-to-stand·walk1 네 clip만 AMP expert와 loco RSI로 쓰고 box interaction proxy RSI는 제외했다. Human은 자기 handle 앞 1.5m에서 시작한다.

현재 push 설정은 `doorApproachAMP.gate_enabled: false`로 거리·손접촉·문각도와 무관하게 매 step AMP 가중치1을 사용한다. 에피소드 전 구간의 합산은 `0.5×task + 0.5×AMP`이며 discriminator·demo/replay·network는 유지한다. 기존 임계값은 spawn/reset 호환과 다른 설정을 위해 보존하지만 현재 push 보상에는 적용하지 않는다. 이 필드가 없거나true인 설정(당김 포함)은 기존1.2→0.6m taper·접촉/각도 latch를 유지한다. 실제 Door 모션 대신 기존 loco expert를 전 구간 사용한다.

TensorBoard의 `open/amp_weight`, `open/amp_active_fraction`은 현재 설정에서 모두1이어야 한다. checkpoint에는 gate·loco expert/RSI·manifest 해시·network 공유 여부를 기록한다. 기존 full-proxy AMP·접근0.1·gate 활성 checkpoint와 혼용하지 않는다. 현재 기본 출력은 `output/door_open_unified_approach_amp_holding_scale_full_amp_balanced_hinge`다.

```bash
# 점유 확인 후 본학습: 2Human / 2048환경 / 2Door
nvidia-smi -i 6
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_train.sh 2 2048 2
# 로컬 viewer / 화면 없는 평가
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_test.sh <checkpoint.pth> 2 16 2 3
# 서버 VNC
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_vnc.sh <checkpoint.pth> 2 1 2 10
```

실행 전 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 확인한다. 2026-10-07 전 구간AMP 수정 후 CPU47개·GPU2환경 물리/보상 대조 및 근접/개방 상태의AMP1 검증을 통과했다. 이전 gated checkpoint의 GPU16환경 진단과 구분한다. 현재 설정의 학습 및 장기 성공률은 미검증이다.

## Door Open — AMP OFF / ON

`HumanoidMADoor`는 Stage 1 semantic 5필드 packet·shared EdgeEncoder·Actor/Critic Transformer와 MAAgent의 scene PPO를 재사용한다. `H0 --OPEN--> Door0`, `H1 --OPEN--> Door1` 고정 binding이며 각 문은 별도의 passive revolute hinge다. 협동 Door graph나 HOLDING 대체 relation을 사용하지 않는다. 두 config는 env의 `ampEnabled`, train의 `amp_enabled`만 다르고, 서로 일치해야 한다. 두 task multiplier는 기존처럼 0.5, ON의 AMP multiplier는 0.5다. OFF에서는 style reward·discriminator forward/loss·expert demo/replay 처리를 생략한다. PPO 설정과 network 구조는 동일하다.

Object node는 기존 O30에 hinge angle·angular velocity·panel 좌표계 handle XYZ를 추가한 O35다. 문 pose는 기존 GTA pose7로 전달한다. Human223·Goal1·6개 token·semantic suffix10을 유지하여 scene 관측은 570차원이다. 문 상태는 `door_hinge_angle`, `door_hinge_velocity`, `door_handle_position`, `door_pose`로 접근한다. `reset_door_angles(env_ids, angles)`는 `[B,2]` 문 각도만 바꾸고 binding은 유지한다. 기본 reset은 0rad, hinge 제한은 [0,1.7]rad다.

Reward는 owner의 root→handle XY 접근(기존 거리 progress 형태), 가까운 손→handle 지수 상호작용, `clamp(angle/1.2,0,1)` 열기 progress, 목표각 1.2rad·허용오차 0.05rad의 current success로 분리한다. approach/interaction은 각각 최대0.1, progress/success는 각각 최대0.2로 기존 edge 최대0.6을 유지한다. 현재 성공 중 term을 포화하며 성공이 풀리면 포화도 풀린다. 각 edge는 자기 문만 계산하고, 이후 기존 자기0.9·동료0.1 공유를 적용한다. power·Human collision penalty는 유지하고 box speed penalty는 사용하지 않는다. 성공 조기종료는 하지 않는다.

TensorBoard `reward_terms/door_approach`, `door_interaction`, `door_progress`, `door_success`, `total`에서 term을 확인한다. 네 Door term은 공유 전 owner 값이며 total은 공유·penalty 이후다. `open/hinge_angle`, `open/hinge_velocity`, `open/current_agent_success`, `open/current_scene_success` 진단도 기존 TensorBoard 경로로 전달한다. 평가 JSON은 기존 `metrics/edge_context_*.json`에 agent·scene·open_0/open_1 성공률을 기록한다.

AMP ON은 현재 `dataset_loco_sit_carry_climb.yaml`과 기존 MotionLib·AMP discriminator를 사용한다. Door motion 전문가는 아직 없으며 `ampExpertMapping.OPEN`은 기존 비조건부 expert 혼합의 명시적 설정이다. 향후 Door reference를 기존 motion YAML/skill 목록에 등록하고 이 row를 교체한다. TRUMANS repo·dataset은 이번 구현에서 사용하거나 수정하지 않았다. 기본 초기화는 기존 loco reference이며 기존 box/hybrid RSI 설정에는 영향을 주지 않는다.

```bash
# 학습 명령만 제공; 이번 작업에서는 학습하지 않음. 기본 2 2048 2.
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi bash tokenhsi/scripts/multi_agent/door_open_amp_off_train.sh 2 2048 2
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi bash tokenhsi/scripts/multi_agent/door_open_amp_on_train.sh 2 2048 2

# 로컬 viewer / headless 평가
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_amp_off_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_amp_off_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/door_open_amp_on_test.sh <checkpoint.pth> 2 16 2 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/door_open_amp_on_test.sh <checkpoint.pth> 2 16 2 3

# 서버 VNC
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/door_open_amp_off_vnc.sh <checkpoint.pth> 2 1 2 10
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/door_open_amp_on_vnc.sh <checkpoint.pth> 2 1 2 10
```

CPU 단위 검증은 `PYTHONPATH=tokenhsi python -c 'import isaacgym; import pytest; raise SystemExit(pytest.main(["-q", "tokenhsi/tests/test_door_open.py"]))'`로 실행한다. 실제 CPU PhysX 초기화·8step·부분 reset·passive hinge 회전·handle 위치·AMP demo를 확인하는 비학습 스크립트도 제공한다:

```bash
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi bash -c '
source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python tokenhsi/scripts/multi_agent/check_door_open_cpu.py \
  --task HumanoidMADoor \
  --cfg_env tokenhsi/data/cfg/multi_agent/door_open_amp_off.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/door_open_amp_off.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --num_envs 2 --num_agents 2 --num_objects 2 --headless --no_video \
  --test --eval --task_graph open --sim_device cpu --pipeline cpu --rl_device cpu
'
```

ON 검증은 위 두 YAML 이름의 `amp_off`를 `amp_on`으로 바꾼다. 이 스크립트는 CPU 평가 환경만 생성하고 trainer·학습은 실행하지 않는다.

Output은 각각 `output/door_open_amp_off`, `output/door_open_amp_on`이다. 기존 box checkpoint와 schema·Object tokenizer·relation embedding 크기가 다르므로 Door scratch checkpoint를 사용한다. 학습 resume는 같은 AMP 모드와 graph·reward 계약만 허용한다. GUI/VNC·GPU pipeline·장기 성공률은 별도 검증 대상이다.

## ksh 세 방식 독립 이식

ksh의 옛 rescue AT 정책은 원본의 평가 전용 데모 로더를 사용한다. `RescueAtKLClimb50_00008000.pth`는 원본 AT 데모에 지정된 checkpoint이며 전체 실험 최고 AT 성공률 순위는 확인되지 않았다. 아래는2명·1환경의 목표높이0.6/1.0m×중간/옆 위치10반복 시각화 명령이다. 기존 checkpoint·학습은 변경하지 않으며 출력만 이 repo에 저장한다. checkpoint 로드·noVNC 페이지HTTP200·실제 화면 렌더링을 확인했다.

```bash
TOKENHSI_GPU=6 PORT=6098 VNC_DIR=/home/hwanhee/opt/vnc GOAL_HEIGHTS=0.6,1.0 GOAL_XY_MODES=midpoint,side REPEATS=10 OUTPUT_PATH=/home/hwanhee/koo_cvpr/stage1_fix/output/ksh_at_rescue_viewer bash /home/hwanhee/ksh/stage1_fix/tokenhsi/scripts/multi_agent/at_goal_demo_vnc.sh /home/hwanhee/ksh/stage1_fix/distill_pth/ApproachScenarioStage1RescueAtKLClimb50_00008000.pth
```

VS Code PORTS에서6098을 포워딩하고 `http://localhost:6098/vnc.html?autoconnect=1&resize=remote`에 접속한다. 데모는 목표 위치에 대한 반응을 관찰하는 용도이며 원래 학습 분포의 성공률 평가와 구분한다.

기존 core를 유지하고 별도 `ports/ksh_stage1` 코드·config를 실행한다. 기본 output은 `output/ksh_size_rsi`, `output/ksh_size_rsi_typed_bias`, `output/ksh_size_rsi_typed_bias_message`다. 같은 방식 checkpoint로만 평가/resume한다. 데이터·캐시·테스트 경로는 [이식 안내](ksh_stage1_port.md).

```bash
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_vnc.sh <checkpoint.pth> 2 1 4 10

TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_vnc.sh <checkpoint.pth> 2 1 4 10

TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_message_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_message_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_message_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/ksh_size_rsi_typed_bias_message_vnc.sh <checkpoint.pth> 2 1 4 10
```

## 후반 hybrid RSI

별도 scratch 실험으로 XYZ20~65cm와 task 비율 10/10/20/30/30%를 유지한다. 후반 물리 후보를 현재 정책의 초기 2 step 관측으로 갱신한다. 같은 크기에서 80~95 → 70~80 → 60~70 → 50~60%를 사용하고 전부 없으면 같은 크기의 loco 시작으로 돌아간다. ON_TOP putDown은 source·받침 XYZ 조합을 따른다. 상세 조건·resume·진단은 [hybrid_late_rsi.md](hybrid_late_rsi.md).

```bash
TOKENHSI_GPU=6 SEED=42 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_hybrid_late_rsi_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_hybrid_late_rsi_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_hybrid_late_rsi_test.sh <checkpoint.pth> 2 16 4 3
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_hybrid_late_rsi_vnc.sh <checkpoint.pth> 2 1 4 10
```

기본 output은 `output/approach_scenario_stage1_hybrid_late_rsi`다. 학습 resume는 동일 hybrid 후보/config 상태를 가진 checkpoint만 허용한다. 기존 학생 후보 파일은 SIT/CLIMB에 계속 필요하다.

## 학습

아래 `GPU`를 사용할 장치 번호로 바꾼다. 각 Stage 1 실험은 별도 output에 scratch로 시작한다.

```bash
GPU=5
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_independent_with_climb_train.sh 2 2048 3  # 27
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_plane_train.sh 2 2048 3  # 28
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_sit_plane_normalized_train.sh 2 2048 3  # 30
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_train.sh 2 2048 3  # 31
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_train.sh 2 2048 3  # 32
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_at_goal_fix_train.sh 2 2048 3  # 33
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_train.sh 2 2048 3  # 34
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_slow_near_start_train.sh 2 2048 3  # 35
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_train.sh 2 2048 3  # 36
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_train.sh 2 2048 3  # 37
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_train.sh 2 2048 3  # Paired AT
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_train.sh 2 2048 4  # Paired placement
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_train.sh 2 2048 4  # Paired owner HOLDING
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_train.sh 2 2048 4  # Unified semantic
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_train.sh 2 2048 4  # Unified shared edge
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_random_binding_train.sh 2 2048 4  # Shared random binding
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_train.sh 2 2048 4  # Unified owner HOLDING
```

한 번에 필요한 실험 **한 줄만** 실행한다. 짧은 확인은 별도 output을 쓴다.

```bash
MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_scenario_stage1_skill_curriculum_reward_preserved_check \
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_train.sh 2 2048 3
```

정기 checkpoint 저장 간격은 학습 설정에서 500 epoch다. `.pth`는 각 run의 `nn/`, TensorBoard는 `summaries/`, edge 진단은 `diagnostics/`에 저장된다.

## 로컬 평가와 서버 VNC

학습과 **같은 번호의 checkpoint와 스크립트**를 사용한다. `CKPT`를 실제 `.pth` 절대 경로로 설정한다.

```bash
GPU=5
CKPT='/absolute/path/to/checkpoint.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_independent_with_climb_test.sh "$CKPT" 2 1 3 10  # 27
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_plane_test.sh "$CKPT" 2 1 3 10  # 28
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=sit bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_sit_plane_normalized_test.sh "$CKPT" 2 1 3 10  # 30
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_test.sh "$CKPT" 2 1 3 10  # 31
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_test.sh "$CKPT" 2 1 3 10  # 32
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_at_goal_fix_test.sh "$CKPT" 2 1 3 10  # 33
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_test.sh "$CKPT" 2 1 3 10  # 34
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_slow_near_start_test.sh "$CKPT" 2 1 3 10  # 35
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_test.sh "$CKPT" 2 1 3 10  # 36
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_test.sh "$CKPT" 2 1 3 10  # 37
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_test.sh "$CKPT" 2 1 3 10  # Paired AT
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_test.sh "$CKPT" 2 1 4 10  # Paired placement
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_test.sh "$CKPT" 2 1 4 10  # Paired owner HOLDING
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_test.sh "$CKPT" 2 1 4 10  # Unified semantic
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_test.sh "$CKPT" 2 1 4 10  # Unified shared edge
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_random_binding_test.sh "$CKPT" 2 1 4 10  # Shared random binding
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_test.sh "$CKPT" 2 1 4 10  # Unified owner HOLDING
```

화면 없는 학습 분포 평가는 해당 test 명령에서 `HEADLESS=1 TASK_GRAPH=random_scenario`와 평가 환경 수 `64`를 지정한다. 로컬 test와 서버 VNC는 모두 기본적으로 `output/<실험명>/metrics/`와 `diagnostics/`에 평가 결과를 쓴다. 반복 평가 시 진단 CSV는 같은 파일에 이어 쓰므로 실행별 분석은 `OUTPUT_PATH`를 따로 지정한다. 학습 checkpoint는 별도 run 디렉터리의 `nn/`에 저장된다. 아래는 서버 viewer 실행 명령이다.

```bash
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_independent_with_climb_vnc.sh "$CKPT"  # 27
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_plane_vnc.sh "$CKPT"  # 28
TOKENHSI_GPU="$GPU" TASK_GRAPH=sit bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_sit_plane_normalized_vnc.sh "$CKPT"  # 30
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_vnc.sh "$CKPT"  # 31
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_vnc.sh "$CKPT"  # 32
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_at_goal_fix_vnc.sh "$CKPT"  # 33
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_vnc.sh "$CKPT"  # 34
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_slow_near_start_vnc.sh "$CKPT"  # 35
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_vnc.sh "$CKPT"  # 36
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_vnc.sh "$CKPT"  # 37
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_vnc.sh "$CKPT"  # Paired AT
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_vnc.sh "$CKPT"  # Paired placement
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_vnc.sh "$CKPT"  # Paired owner HOLDING
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_vnc.sh "$CKPT"  # Unified semantic
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_vnc.sh "$CKPT"  # Unified shared edge
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_random_binding_vnc.sh "$CKPT"  # Shared random binding
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_vnc.sh "$CKPT"  # Unified owner HOLDING
```

## Unified 독립 5과제 · semantic / shared edge / owner-HOLDING 비교

Semantic과 owner-HOLDING은 **edge 입력 상태 유무만 다르다**. Shared edge는 semantic과 과제·관측·보상이 같고 actor·critic의 semantic EdgeEncoder 임베딩·MLP·bias projection만 공유한다. 토크나이저·GTA/Transformer·head는 분리한다. 기존 paired/focus 실험은 보존하며 각자 새 output에서 scratch로 시작한다. 새 AMP 입력은 프레임당 기존 129차원에 과제 one-hot 3차원을 더한 132차원, 10프레임 총 1320차원이다. 기존 paired checkpoint와는 AMP 및 variant 계약이 달라 이어 학습하지 않는다.

- 학습: 2명·4물체. 각 agent가 독립적으로 `HOLDING/SIT/CLIMB/HOLDING_AT/HOLDING_ON_TOP = 5/5/20/35/35%`를 샘플한다. 두 ON_TOP도 가능하며 명목 빈도는 12.25%다.
- 논리 할당(0-based): H₀→O₀→G₀, H₁→O₁→G₁. SIT/CLIMB도 각자의 Oᵢ를 사용한다. ON_TOP은 O₀→O₂, O₁→O₃. 물리 asset slot만 reset마다 섞이며 모든 관측·RSI·보상은 동일 logical-to-physical mapping을 따른다.
- H/O/G별 인코딩 뒤 transformer 입력을 모든 타입에 걸쳐 매 forward 무작위 순열로 섞는다. 한 forward의 batch는 같은 순열을 사용하며, GTA와 bias의 두 축도 함께 이동한다. 출력은 canonical 순서로 복원한 다음 human별 actor/value head에 전달한다. 순열 등가성에 따른 수치 동등성은 검증하되 추가 학습 이득을 보장하지 않는다.
- Goal GTA quaternion은 `(0,0,0,1)`로 고정한다. 상자 크기는 X=0.40m, Y=0.30m, Z=0.30~0.50m(0.05m 간격)다.
- loco RSI의 담당 상자는 **owner root XY에서 반경 1~2m**에 배치한다. AT 목표/ON_TOP 받침에 가까운 시작은 0%이며 별개 설정이다. pickUp/carryWith/putDown/sit/climb RSI의 참조 물체 위치는 유지한다. 물리 배치 부적합 시 graph를 유지한 채 reset을 다시 뽑는다.
- state/progress/success 0.2/0.2/0.2, 자기 edge 합계·동료 edge 평균의 0.9/0.1 공유, placement 성공 시 matching HOLDING 포화는 유지한다. TERM 입력은 추가하지 않는다.

| 과제 | RSI |
| --- | --- |
| HOLDING | loco 50%, pickUp 50% |
| SIT | loco 50%, sit 50% |
| CLIMB | loco 50%, climb 50%; 원래 허용 시점 사용, 후반 강제 샘플링 없음 |
| HOLDING+AT | loco 50%, pickUp 10%, carryWith 30%, putDown 10% |
| HOLDING+ON_TOP | loco 50%, pickUp 10%, carryWith 40%; putDown 없음 |

AMP는 단일 판별기에 **각 agent의 할당 과제 family**를 매 프레임 붙인다. HOLDING/AT/ON_TOP은 carry, SIT는 sit, CLIMB은 climb이다. RSI로 걷기를 뽑았어도 family는 할당 과제를 따른다. 전문가의 과제 내부 분포는 원본 unified에 맞춘다: carry=loco/OMOMO/pickUp/putDown 1/3·1/3·1/6·1/6, sit=loco/sit 1/3·2/3, climb=loco/climb/climbNoRSI 각 1/3. `carryWith`는 RSI에는 있지만 이 AMP 전문가 혼합에는 없다.

`skillDiscProb`는 리셋 시 family 비율 75/5/20으로 환산한 명목 전체 분포다. 실제 조건부 expert 생성은 `utils/unified_training.py`의 분포를 사용한다. 학습에서는 전문가·리플레이를 현재 rollout의 family와 **행별로 맞춰** 뽑아 PPO minibatch마다 family 빈도가 일치한다. 리플레이에 해당 family가 아직 없을 때만 같은 family의 현재 rollout을 사용한다. 과제 라벨은 AMP 정규화에서 통과시키며 참조 이력·부분 reset에도 유지한다. task/AMP reward 비중은 기존 0.5/0.5다.

Semantic 실험은 edge당 `[valid,src,dst,relation,owner]` 5필드이고 semantic type/relation만 bias에 인코딩한다. Owner 실험은 6번째 값에 동일 owner·source의 HOLDING φ를 넣고 AT/ON_TOP의 semantic64 뒤에서 상태 잔차를 더한다. 다른 유효 edge의 입력 값은 1이며, 상태 잔차는 placement edge에만 적용한다.

아래 학습 명령에서 필요한 실험을 선택한다. `TOKENHSI_GPU=0`은 로컬 단일 GPU 기준이다.

```bash
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_train.sh 2 2048 4
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_train.sh 2 2048 4
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_train.sh 2 2048 4
```

각 checkpoint에는 같은 실험의 test/VNC를 사용한다. 로컬 viewer는 다음과 같다.

```bash
CKPT='/absolute/path/to/checkpoint.pth'
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_test.sh "$CKPT" 2 1 4 10
```

`TASK_GRAPH=holding|sit|climb|holding_at|holding_ontop`은 두 사람 모두 해당 과제로 평가한다. 기본 평가는 loco 시작이다. placement의 들고 있는 시작을 보려면 `TASK_GRAPH=holding_at` 또는 `holding_ontop`에 `EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0`을 추가한다. `HEADLESS=1`은 화면 없는 평가다.

```bash
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_vnc.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_vnc.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_vnc.sh "$CKPT" 2 1 4 10
```

각 기본 output은 `output/approach_scenario_stage1_unified`, `output/approach_scenario_stage1_unified_shared_edge_encoder`, `output/approach_scenario_stage1_unified_owner_holding`이다. Shared edge는 별도 variant/checkpoint로 scratch 학습하며 기존 unified checkpoint를 resume하지 않는다. 짧은 검증은 별도 `_check` output과 `MAX_ITERATIONS=2`를 사용한다. 2명·4물체 sampler이며 가변 인원 평가 sampler 확장은 포함하지 않는다.

## Stage 1 AT/ON_TOP 집중 실험 · 36번

34번과 동일한 reward·RSI·가까운 시작 설정에서 `HOLDING/SIT/CLIMB/HOLDING_AT/HOLDING_ON_TOP = 0/0/0/0.5/0.5`로 샘플한다. 한 scene에 AT와 ON_TOP이 각각 하나씩 배정되고, 두 과제 모두 자기 HOLDING edge를 포함한다. 단독 HOLDING 및 SIT/CLIMB 과제는 학습 중 나오지 않는다. `templateRsi`의 AT/ON_TOP 행은 그대로 적용되고, CLIMB 후반 RSI는 CLIMB 과제가 없어 사용되지 않는다. `skillDiscProb`의 SIT/CLIMB 모션 비율은 34번과 동일하다. 새 variant·output에서 scratch로 시작한다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachScenarioStage1PlacementFocus.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_vnc.sh "$CKPT"
```

## Stage 1 같은 과제 쌍 · 가까운 시작 없음

**Paired AT**은 3개 물체에서 두 agent에게 각각 다른 물체·목표를 배정한다. 각 agent의 edge는 `H→O HOLDING`, `O→G AT`이며 한 scene에 AT edge가 두 개다. **Paired placement**는 4개 물체에서 scene마다 `AT+AT` 또는 `ON_TOP+ON_TOP`을 50%씩 뽑는다. ON_TOP scene의 두 source와 두 support는 모두 다른 물체다. 두 실험 모두 34번의 RSI·AMP·보상식을 유지하고 `near_start` 확률만 0으로 고정한다. `TASK_GRAPH=holding_at`/`holding_ontop`으로 paired placement 평가 장면을 각각 고를 수 있다. 다른 물체 수와 variant는 서로 checkpoint 호환되지 않는다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachScenarioStage1PairedAtNoNear.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_vnc.sh "$CKPT"

TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_train.sh 2 2048 4
CKPT='/absolute/path/to/ApproachScenarioStage1PairedPlacementNoNear.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_vnc.sh "$CKPT"
```

## Stage 1 Paired placement · 담당자 HOLDING 상태 입력

위 Paired placement와 동일한 2-agent/4-object·AT+AT/ON_TOP+ON_TOP·가까운 시작 없음·RSI·AMP·reward를 사용한다. AT/ON_TOP edge의 여섯 번째 관측 필드는 동일 owner와 source 물체의 직전 HOLDING edge `φ`다. HOLDING 등 다른 유효 edge에는 1을 넣고, 무효 edge는 0으로 마스킹한다. `φ`는 손–상자 중심 거리 기반의 0~1 상태이며 실제 물리적 파지 플래그는 아니다. 기존 paired checkpoint는 새 관측 폭과 맞지 않는다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_train.sh 2 2048 4
CKPT='/absolute/path/to/ApproachScenarioStage1PairedPlacementOwnerHolding.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_vnc.sh "$CKPT"
```

## Stage 1 ON_TOP putDown RSI 실험 · 37번

36번과 동일한 과제·보상 분포에서 `HOLDING_ON_TOP` RSI 중 `carryWith` 10%를 `putDown`으로 옮긴다. ON_TOP `putDown`은 원본 모션의 45~70% 구간에서만 시작하고, 받침 상자를 모션의 최종 놓기 XY에 배치한다. 이 구간은 원본 모션의 후기 상자·받침 겹침을 피한다. 받침의 물리 크기는 기존 분포 그대로이며, reset 시 상자 또는 사람과 겹치면 재추첨한다. AT `putDown` RSI 10%와 나머지 설정은 36번과 같다. 별도 variant·output에서 scratch로 시작한다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachScenarioStage1OntopPutdown.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_vnc.sh "$CKPT"
```

## Stage 2 협력 실험 · 29번

29번은 지정한 Stage 1 `.pth`에서 actor/critic/AMP weight와 관찰 통계를 옮겨 새 실행을 시작한다. `RESUME_CHECKPOINT`는 이미 시작한 **같은 Stage 2 변형**을 이어 학습할 때만 쓴다. 기존 29번의 SIT 성공과 보상 정규화는 **28번 정의**다.

```bash
STAGE1_CHECKPOINT='/absolute/path/to/Stage1.pth' TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_train.sh 2 2048 3
RESUME_CHECKPOINT='/absolute/path/to/ApproachStage2Coordination.pth' TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachStage2Coordination.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=place_stack \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=place_climb \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_vnc.sh "$CKPT"
```

Stage 2 plane은 34번의 SIT plane 성공·자기 edge 합계·동료 edge 평균을 적용한다. 협력 과제 비율은 `place_climb/place_sit/place_stack = 20/20/40%`, 독립 과제는 20%다. 독립 과제 내부의 HOLDING/SIT/CLIMB/HOLDING+AT/HOLDING+ON_TOP은 34번과 같이 10/10/10/35/35이며 RSI·가까운 시작 일정도 34번과 같다. 협력 과제는 공유 물체에 맞춘 Stage 2 RSI를 유지한다. 별도 config·스크립트·output을 사용한다.

```bash
STAGE1_CHECKPOINT='/absolute/path/to/ApproachScenarioStage1SelfSum.pth' TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachStage2CoordinationSitPlane.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=place_stack \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=place_stack \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_vnc.sh "$CKPT"
```

3·4인 평가 graph는 `tokenhsi/data/cfg/multi_agent/graphs/`에 있다. 입력 크기 검증용이며 성공은 별도 학습으로 확인한다.

## 원본 1번

원본 비교가 필요할 때 `ma_carry_train.sh`, `ma_carry_test.sh`, `ma_carry_watch.sh`를 사용한다. 실험별 GPU 지정과 데이터 경로는 위 공통 규칙을 따른다.

지표 정의는 [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 참고한다. TensorBoard는 `tensorboard --logdir output --host 127.0.0.1 --port 6006`으로 실행한다.

## Shared random binding · 사용자 12옵션 조합

선택 순서 `2 2 2 1 2 1 2 1 2 1 2 2`를 별도 scratch 실험으로 구현했다. 기존 unified/shared/34/35 config와 output을 유지한다.

- Actor/critic semantic EdgeEncoder 공유, PRE/TERM·owner HOLDING 상태 입력 없음. H/O/G token 순열은 매 forward 유지한다.
- reset마다 논리 물체 4개와 goal 2개를 무작위 순열로 배정한다. 두 담당 물체와 두 ON_TOP 받침은 서로 분리된다. 물리 slot도 기존 방식으로 섞는다. graph·RSI·관측·보상·속도 penalty는 같은 binding을 사용한다.
- 실제 상자 크기는 XY 0.40~0.65m, Z 0.25~0.55m이며 34번 크기 풀·평가 크기를 사용한다. loco 담당 상자는 사람 root XY에서 1~2m다.
- 배치 가까운 시작은 34번 일정: per-env 0→300000 step에 80→30%, 이후 30%. AT 목표 XY는 source에서 0.75~1.5m, ON_TOP 받침 XY는 source에서 1.1~1.8m에 놓는다. 평가에는 적용하지 않으며 AT putDown RSI의 참조 목표는 보존한다.
- 과제 비율은 HOLDING/SIT/CLIMB/HOLDING_AT/HOLDING_ON_TOP 5/5/20/35/35다. RSI는 shared의 과제별 비율을 유지하고 CLIMB 후반 강제 RSI는 끈다. 가까운 목표 배치는 RSI 자세 초기화와 별개다.
- AMP는 34/35번 비조건부 혼합: loco/sit/climb/climbNoRSI/omomo/pickUp/carryWith/putDown = .26/.20/.08/.06/.10/.10/.10/.10. AMP 과제 one-hot·family 매칭을 끄며 10프레임 입력은 1290차원이다.
- Goal 회전은 identity, 속도 penalty는 graph 담당 물체 기준이다. state/progress/success .2/.2/.2와 자기 합계·동료 평균 .9/.1은 유지한다.
- 기본 정책 관측은 644차원이지만 AMP·binding·variant가 다르므로 기존 unified/shared checkpoint의 resume는 거부한다. 아래 전용 train에서 scratch로 시작한다.

```bash
nvidia-smi
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_random_binding_train.sh 2 2048 4
```

GPU는 실행 시 점유를 확인해 선택한다. 본학습 전에 셸에 남은 MAX_ITERATIONS/OUTPUT_PATH/RESUME_CHECKPOINT를 해제한다. 평가에는 새 실험 checkpoint만 사용한다.

```bash
CKPT='/absolute/path/to/ApproachScenarioStage1UnifiedSharedRandomBinding.pth'
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_random_binding_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_random_binding_test.sh "$CKPT" 2 64 4 3
TOKENHSI_GPU=6 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_random_binding_vnc.sh "$CKPT" 2 1 4 10
```

### 2026-10-07 push 경첩 저항 강화

현재 분리/공유 `door_open_*_approach_amp` 두 설정에 `env.door.hinge: {damping: 2.0, friction: 0.1}`를 적용했다. 기존 회전 경첩의0.5/0.05보다 저항을 높이고 DOF_MODE_NONE·stiffness0의 수동 문을 유지한다. 공용 URDF·pull·BONES 설정은 변경하지 않는다. 경첩 물리는 checkpoint 계약에 포함되어 기존 저항으로 학습한 checkpoint의 resume/test는 거부한다. 현재 wrapper는 새 물리 설정으로 scratch 학습하며 기본 output도 `_balanced_hinge`로 분리한다. 기존 프로세스에는 소급 적용되지 않는다.

물리 검증: `TOKENHSI_GPU=6 OUTPUT_PATH=output/door_hinge_resistance_check_20261007 bash tokenhsi/scripts/multi_agent/validate_door_physics.sh`. 비학습2환경에서 설정값·무접촉 안정성·충격 후 자유회전·지속 humanoid 접촉·관측/보상 대응을 검사한다. 손 사용은 여전히 성공의 필수 조건이 아니며 angle-only 성공 보상은 유지했다.

2026-10-07 로컬 학습 인계: 사용자 승인한 절충 경첩2.0/0.1을 분리/공유 loco full-AMP 두 설정에 반영했다. 로컬 인계 가이드는 [door_local_training.md](door_local_training.md). 5.0/0.5 결과는 이전 강화 설정의 기록이며 새 검증은 `output/door_balanced_hinge_check_20261007/`를 사용한다.


## ArmChair CARRY Adapter

기존39000 사본에서 출발한다. 원래 Obj tokenizer와 각 action-head Linear층의0초기화 잔차 adapter, critic·AMP 판별기를 학습한다. 기존 head·Transformer·사람/Goal tokenizer는 동결한다. HOLDING_AT·ArmChair train63/test27종·loco 초기화를 사용하며 기존 tokenizer-only 출력과 분리된다. 인자는 train `[envs]`, test/VNC `<checkpoint> [envs] [repeats]`다.

```bash
# 학습: 전체 ArmChair63종,2agent/2048env/4object
unset CHECK MAX_ITERATIONS OUTPUT_PATH RESUME_CHECKPOINT ASSETS_PER_CATEGORY
TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_adapter_train.sh 2048

# Local / headless / VNC: 같은 adapter variant의 저장본 사용
CKPT='/absolute/path/to/adapter_checkpoint.pth'
TOKENHSI_GPU=4 HEADLESS=0 OUTPUT_PATH=output/object_tokenizer_finetune/carry_chair_adapter/local_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_adapter_test.sh "$CKPT" 1 10
TOKENHSI_GPU=4 HEADLESS=1 OUTPUT_PATH=output/object_tokenizer_finetune/carry_chair_adapter/eval_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_adapter_test.sh "$CKPT" 64 3
TOKENHSI_GPU=4 PORT=6134 OUTPUT_PATH=output/object_tokenizer_finetune/carry_chair_adapter/vnc_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_adapter_vnc.sh "$CKPT" 1 100
```


## ArmChair CARRY Copied Adapter

원래 Actor Obj tokenizer·head·Transformer·사람/Goal tokenizer를 동결하고,39000 Obj 가중치를 복사한 별도 tokenizer와0초기화 adapter·critic·AMP 판별기를 학습한다. Actor encoder의 `tokenizers.1`과 `adapted_object_tokenizer`, action adapter가 하나의pth에 함께 저장된다. 이전 adapter-only variant와 구분해 처음부터 학습하며 해당 저장본을 직접 resume하지 않는다. 객체/과제/RSI는 ArmChair63종·HOLDING_AT·loco 시작으로 유지한다.

평가의 `POLICY_BRANCH=adapt`(기본)는 복사본+adapter, `POLICY_BRANCH=base`는 원래 Obj tokenizer+adapter 비활성화다. 모드는 checkpoint의 가중치를 변경하지 않는다. Base의 actor 파라미터와 입력 정규화는 원본 그대로 유지하지만 critic/AMP는 학습하므로 학습 보상·가치까지 원본 보존한다는 뜻은 아니다.

```bash
# 학습
unset CHECK MAX_ITERATIONS OUTPUT_PATH RESUME_CHECKPOINT ASSETS_PER_CATEGORY
TOKENHSI_GPU=4 bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_copied_adapter_train.sh 2048

CKPT='/absolute/path/to/copied_adapter_checkpoint.pth'
# Local / headless / VNC: 기본 adapt 분기
TOKENHSI_GPU=4 HEADLESS=0 OUTPUT_PATH=output/object_tokenizer_finetune/carry_chair_copied_adapter/local_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_copied_adapter_test.sh "$CKPT" 1 10
TOKENHSI_GPU=4 HEADLESS=1 OUTPUT_PATH=output/object_tokenizer_finetune/carry_chair_copied_adapter/eval_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_copied_adapter_test.sh "$CKPT" 64 3
TOKENHSI_GPU=4 PORT=6135 OUTPUT_PATH=output/object_tokenizer_finetune/carry_chair_copied_adapter/vnc_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_copied_adapter_vnc.sh "$CKPT" 1 100
# Base 분기 평가: 원래 Obj·adapter 끔
TOKENHSI_GPU=4 POLICY_BRANCH=base HEADLESS=1 OUTPUT_PATH=output/object_tokenizer_finetune/carry_chair_copied_adapter/base_new bash tokenhsi/scripts/multi_agent/object_tokenizer_finetune_carry_chair_copied_adapter_test.sh "$CKPT" 64 3
```


## Y1.8m PUSH 손 높이 비교

X1.2~1.5m·Y1.8m·Z1.3m/20kg·접근1~5m·목표1~4m·기존 safe reset/native RSI를 기준으로 두 새 scratch 실험을 분리한다. 손 목표 높이는 0.65m 고정 또는 원본 AMP에서 샘플한 양손 중점 높이를 에피소드별로 고정한다. 세부 수식·높이·checkpoint 계약은 [명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push_depth180_height_compare.md)를 따른다. 연속 state 감쇠만 완화하며 CONTACT 자체 성공은 state>0.9와 양손 각각/상자 반력≥0.5N이다. 별도 수평≤15cm·높이±5cm 성공 gate는 없다. AT/paired 포화는 유지하며 기존 checkpoint나 두 높이 버전은 직접 혼용하지 않는다. 기존 학습은 재시작하지 않는다.

저장소 루트에서 실행한다. 아래는 새 본학습 명령이며 자동 실행하지 않았다.

```bash
# 0.65m 고정: 학습 / 로컬 viewer / 서버 VNC
env -u CHECK -u MAX_ITERATIONS -u OUTPUT_PATH -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_h065_relaxed_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_h065_relaxed_test.sh /absolute/path/to/checkpoint.pth 2 1 4 10
TOKENHSI_GPU=6 PORT=6125 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_h065_relaxed_vnc.sh /absolute/path/to/checkpoint.pth 2 1 4 10

# AMP 원본 높이: 학습 / 로컬 viewer / 서버 VNC
env -u CHECK -u MAX_ITERATIONS -u OUTPUT_PATH -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_hamp_relaxed_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_hamp_relaxed_test.sh /absolute/path/to/checkpoint.pth 2 1 4 10
TOKENHSI_GPU=6 PORT=6126 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_hamp_relaxed_vnc.sh /absolute/path/to/checkpoint.pth 2 1 4 10

```

화면 없는 평가는 `HEADLESS=1`이다.


## Y1.8m 높은 손 위치 전도 검사

저장소 루트에서 실행하는 비학습 물리 검사다. 학습 설정의 질량을 변경하지 않는다. 접촉체와 실제humanoid PD/IK를 구분하며 학습정책 안정성 보장은 아니다. 기존 출력은 덮어쓰지 않는다.

```bash
TOKENHSI_GPU=6 PUSH_DEPTH_MASS=20 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/analyze_push_depth180_high.sh
TOKENHSI_GPU=6 PUSH_DEPTH_MASS=25 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/analyze_push_depth180_high.sh
```

[조건·출력·재집계](../ports/ksh_stage1_stage2_base/push/markdowns/USAGE.md#y18m-높은-손-위치-전도-검사). 동일 하중/mg 비율 비교이며 절대 힘은 질량과 함께 달라진다.


## Y1.8m PUSH 손 목표1.0m 고정

`stage1_push_depth180_safe_h100_relaxed`는 손 목표 높이1.0m·Y1.8m·X1.2~1.5m·Z1.3m·20kg이다. state감쇠완화·state>0.9와반력0.5N·원본AMP/RSI·초기관통방지·접근1~5m·목표1~4m를유지한다. RSI관절을1m로retarget하지않고접촉목표점만고정한다. [높이/체크포인트명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push_depth180_height_compare.md#손-목표-높이10m-고정). 기존학습에는자동반영되지않는다.

저장소루트에서실행한다. 새본학습은자동실행하지않았다.

```bash
env -u CHECK -u MAX_ITERATIONS -u OUTPUT_PATH -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_h100_relaxed_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_h100_relaxed_test.sh /absolute/path/to/checkpoint.pth 2 1 4 10
TOKENHSI_GPU=6 PORT=6127 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_h100_relaxed_vnc.sh /absolute/path/to/checkpoint.pth 2 1 4 10
```

화면없는평가는 `HEADLESS=1`이다.


## Y1.8m·20kg·RSI≤1m 선별 두 실험

공통 X1.2~1.5m/5cm·Y1.8m고정·Z1.3m고정·20kg·상자→목표1~4m·사람→상자시작1~5m. 원본PUSH AMP에서양손중점높이≤1m인프레임만선택하고,관절/속도/손높이를retarget하지않는다. 기존safe강체배치후에도≤1m인프레임만사용한다. 21/21모션유지·크기별1833프레임선별. AMP전문가데이터는필터링하지않으며RSI초기화풀만선별한다. 완화state>0.9+양손각각/상자0.5N·AT/paired포화는유지한다. [RSI/보상/체크포인트명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push_depth180_rsi_le1m.md).

두명령은각각별도터미널에서실행한다. 각2048학습환경이며두실험모두GPU6을사용한다. 기존checkpoint와직접혼용하지않는새scratch학습이다. 본학습은자동실행하지않았다.

```bash
# 손목표1.0m·AT성공기울기≤5°
env -u CHECK -u MAX_ITERATIONS -u OUTPUT_PATH -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_test.sh /absolute/path/to/checkpoint.pth 2 1 4 10
TOKENHSI_GPU=6 PORT=6131 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_vnc.sh /absolute/path/to/checkpoint.pth 2 1 4 10

# 손목표0.8m·AT성공기울기제한없음
env -u CHECK -u MAX_ITERATIONS -u OUTPUT_PATH -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_test.sh /absolute/path/to/checkpoint.pth 2 1 4 10
TOKENHSI_GPU=6 PORT=6132 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_vnc.sh /absolute/path/to/checkpoint.pth 2 1 4 10

```

화면없는평가는 `HEADLESS=1`이다.

## PUSH 공통 면 XY CONTACT (2026-10-21)

[명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push_common_face_xy.md) · [사본 실행 목록](../ports/ksh_stage1_stage2_base/push/markdowns/config.md). 기존 checkpoint와 직접 혼용하지 않는 별도 scratch 설정이다.

### stage1_push_depth180_h080_rsi1m_phase_face_xy

```bash
TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_phase_face_xy_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_phase_face_xy_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_phase_face_xy_test.sh "$CKPT" 2 64 4 3
TOKENHSI_GPU=6 PORT=6140 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_h080_rsi1m_phase_face_xy_vnc.sh "$CKPT" 2 1 4 10
```

### stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy

```bash
TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy_test.sh "$CKPT" 2 64 4 3
TOKENHSI_GPU=6 PORT=6141 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_depth180_h100_rsi1m_at5deg_phase_face_xy_vnc.sh "$CKPT" 2 1 4 10
```

## PUSH 공통 면 XY·Y1.8m/20kg와 Y2.0m/25kg

두 설정 모두 X1.2~1.5m/0.05m격자·Z1.3m·마찰0.6, 목표1~4m·loco시작1~5m·loco/PUSH50%·원본RSI≤1m·AT기울기제한없음이다. 새 물성별 전수 재검증 캐시를 사용한다. [명세·RSI검증·영상](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push_face_pair.md). 각 checkpoint 계약은 독립이며 기존 checkpoint를 직접 resume하지 않는다.

### stage1_push_face_xy_depth180_20kg

```bash
TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth180_20kg_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth180_20kg_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth180_20kg_test.sh "$CKPT" 2 64 4 3
TOKENHSI_GPU=6 PORT=6142 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth180_20kg_vnc.sh "$CKPT" 2 1 4 10
```

### stage1_push_face_xy_depth200_25kg

```bash
TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth200_25kg_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth200_25kg_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth200_25kg_test.sh "$CKPT" 2 64 4 3
TOKENHSI_GPU=6 PORT=6143 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth200_25kg_vnc.sh "$CKPT" 2 1 4 10
```

## PUSH 공통 면 XY·Y2.0m/20kg·마찰0.4

`stage1_push_face_xy_depth200_20kg_friction04`은 기존 Y2.0m/25kg·마찰0.6에서 질량/상자전체마찰을20kg/0.4로 바꾼 별도 scratch 설정이다. X1.2~1.5m·Z1.3m·바닥마찰1.0·공통면CONTACT·AT·RSI분포는 유지한다. 전용 재검증 RSI캐시를 사용하며 기존 checkpoint를 직접 resume하지 않는다. [명세·검증·영상](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push_face_depth200_20kg_friction04.md).

```bash
# 학습
TOKENHSI_GPU=6 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth200_20kg_friction04_train.sh 2 2048 4
# 로컬 viewer
TOKENHSI_GPU=6 HEADLESS=0 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth200_20kg_friction04_test.sh "$CKPT" 2 1 4 10
# 화면 없는 평가
TOKENHSI_GPU=6 HEADLESS=1 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth200_20kg_friction04_test.sh "$CKPT" 2 64 4 3
# 서버 VNC
TOKENHSI_GPU=6 PORT=6144 bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_face_xy_depth200_20kg_friction04_vnc.sh "$CKPT" 2 1 4 10
```
