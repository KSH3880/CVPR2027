# 레포 구조

- `evaluation/single_match/`: 업로드한 [단독 평가 명세](evaluation_single.md)의 독립 실행 코드. `backend.py` 평가 전용 subclass/strict frozen actor, `evaluate.py` 첫episode 기록, `metrics.py`·`rescore.py` 점/영역 재채점, `batch.py` 기존4과제 두모델 비교 및 별도PUSH 평가·seed통계, `run.sh`/`vnc.sh`·CPU테스트. [조건·실행](../evaluation/single_match/README.md).

- `ports/jhh_stage1_push_raw_cuboid/`: Pro1/JHH GPU5 `stage1_v2_fixed_push_360_raw_cuboid`의 독립 코드·설정·raw RSI·22500회 모델/73000회 교사 사본. `PORT_MANIFEST.json`·`verify_port.py`로 복사 무결성 확인; [실행](../ports/jhh_stage1_push_raw_cuboid/README.md).

- GPU6 PUSH 실제XYZ 크기 제로샷: 사본 `tokenhsi/utils/push_eval_sizes.py`·`tests/{test,check}_push_eval_sizes.py`; [명령](../ports/ksh_stage1_stage2_base/push/markdowns/USAGE.md#push-크기-제로샷).

- GPU6 PUSH 목표 고정·중간 상자 이동: 사본 `tokenhsi/utils/push_box_switch.py`; [명령](../ports/ksh_stage1_stage2_base/push/markdowns/USAGE.md#push-중간-상자-이동-제로샷).

- GPU6 PUSH 평가 중90°/180° 목표 전환: 사본 `tokenhsi/utils/push_goal_switch.py`와 `learning/multi_agent/edge_context_eval.py`; [명령](../ports/ksh_stage1_stage2_base/push/markdowns/USAGE.md#push-중간-목표-전환-제로샷).

현재 실행 실험은 원본 multi-agent carry(1), 독립 Stage 1(27·28·30·31·32·33·34·35·36·37 및 paired AT/placement·unified 독립 5과제), 협력 Stage 2(29 및 SIT plane 변형)다. 실행 명령과 checkpoint 규칙은 [config.md](config.md)에 있다.

## 진입점

| 경로 | 역할 |
| --- | --- |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/tests/check_five_task_eval_rsi.py` | 5과제 Stage1 평가의 PUSH/loco/혼합 RSI 선택·원본 관절/속도·부분 reset·12step 검사 |
| `ports/ksh_stage1_stage2_base/push/markdowns/reference/push100_resume1000.md` | GPU6 두 물성의 1000회 checkpoint에서 `TASK_GRAPH=push_at`·별도 output으로 PUSH 100% 재개 |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/utils/push_motion_metrics.py`, `scripts/multi_agent/resume_five_task_metrics.py` | 5과제 PUSH 접촉·목표 진척·에피소드 거리 집계/500회 재개; [명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push_training_metrics.md) |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_face_xy_depth200_20kg_friction04.yaml`, 동명 train/test/VNC·전용 RSI cache | Y2m·20kg·상자마찰0.4 공통면CONTACT·물성별 RSI 재검증 |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_face_xy_depth{180_20kg,200_25kg}.yaml`, 동명 train/test/VNC·`tests/{recheck,audit}_push_face_rsi.py`·`scripts/multi_agent/render_push_face_rsi.py` | 마찰0.6·공통면CONTACT 두물성/전수RSI검사·실제충돌체/발메시영상 |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/utils/push_point.py::common_face_xy`, `*_phase_face_xy` env/train/test/VNC, `tests/{test,check}_push_common_face_xy.py` | 동적 공통 유한 면 XY CONTACT·state/반력 success·독립 보상 계약 |
| [PUSH RSI 기준](push_rsi_guidelines.md) | 향후 PUSH RSI의 원본 보존·높이/밀기 구간 선별·loco 분리·상자 동반 검수 기준 |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_depth180_{h100_rsi1m_at5deg,h080_rsi1m}_phase.yaml`, 동명 train/test/VNC | 새 RSI5모션·loco50%·목표1~4m(커리큘럼 없음)의 두 새 학습 설정 |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/select_push_rsi_phase.py`, 동명 `tests/test_push_rsi_phase_selection.py`·`data/stage1_push_depth180_rsi_le1m_push_phase_*` | 원본 손 높이≤1m·양팔 밀기 구간만 RSI 재선별; 기존 학습 미적용 |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/utils/push_goal_curriculum.py`, 전용 `*_goal_curriculum` env/train/test/VNC | 목표1~5→1~7→1~10m·epoch 기준 단계/상태보존 이관 |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_depth180_{h100_rsi1m_at5deg,h080_rsi1m}.yaml`, 동명 train/test/VNC·`tests/filter_push_rsi_height.py`, `test_push_rsi_height_selection.py` | Y1.8m/20kg·원본RSI손중점≤1m 선별·1.0m/AT5°와0.8m/무제한 비교 |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_depth180_safe_h100_relaxed.yaml`, 동명 train/test/VNC | Y1.8m·20kg·손 목표1.0m 고정·완화state·성공state>0.9+0.5N |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/analyze_push_depth180_high.sh`, `report_push_depth180_high.py` | Y1.8m/정방형 대조·20/25kg·고위치 접촉체/21AMP PD 전도 검사·trace 재검증 보고서 |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_depth180_safe_{h065,hamp}_relaxed.yaml`, 동명 train/test/VNC·`tests/test_push_depth180_relaxed.py` | Y1.8m·0.65m/AMP 높이 비교, 연속 state 완화·성공 state>0.9+0.5N |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/tests/test_push_midpoint_force.py` | safe PUSH 3종 0.5N 기본값·법선 반력/위치 동시성·부분 reset·checkpoint 계약 CPU 검사 |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/data/cfg/{multi_agent,train/rlg}/stage1_push_midpoint_safe_at5deg.yaml`, 동명 train/test/VNC·`tests/test_push_at5deg.py` | GPU5 safe 기반 AT 성공 기울기≤5°·독립 output/checkpoint |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/tests/{audit_push_topple,analyze_push_topple_geometry,test_push_reset_geometry}.py` | 초기학습전도사건추적·발mesh내부정점분석·collider/바닥회귀검사 |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/utils/push_reset_geometry.py`, `stage1_push_{midpoint,depth180}_safe` config/train/test/VNC/cache | 발mesh·sphere/capsule의상자/바닥초기관통방지·GPU5/6별도계약; [명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push_safe_start.md) |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/data/cfg/multi_agent/stage1_push_depth180_native_rsi.yaml`, 동명 train/test/VNC·cache 생성기 | X만1.2~1.5가변·Y1.8/Z1.3고정·20kg·−Y면/+Y목표1~4m; [명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/stage1_push_depth180_native_rsi.md) |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/data/cfg/multi_agent/stage1_push_midpoint_native_rsi.yaml`, 동명 train/test/VNC·cache 생성기 | 중점만 CONTACT/state>0.9·원본AT/포화·20kg/X=Y/Z1.30고정/목표1~4m; [명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/stage1_push_midpoint_native_rsi.md) |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/data/cfg/multi_agent/stage1_push_rect_native_rsi.yaml`, 동명 train/train-test/VNC와 크기별21모션 RSI cache | Stage1 SIT/CARRY 방식 전체 유효프레임·후반30%추출, 원본관절/AMP 자세 보존; [간단검증/실행](../ports/ksh_stage1_stage2_base/push/markdowns/reference/stage1_push_rect_native_rsi.md) |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/data/cfg/multi_agent/stage1_push_rect_point.yaml` | 독립XYZ1.2~1.5m/20kg·뒷면중앙0.65m 양손중점·3 RSI 전수선별·전용train/test/VNC; [명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/stage1_push_rect_point.md) |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/{analyze_push_rect_stability.sh,analyze_push_rect_contacts.py,analyze_push_rect_humanoid.py,push_stability_common.py,report_push_rect_stability.py}` | 동일질량·XYZ독립가변·양손중점의기울기/전진분석, 학습설정미변경; [검사와결과](../ports/ksh_stage1_stage2_base/push/markdowns/reference/push_rect_stability.md) |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/data/cfg/multi_agent/stage1_push_no_force.yaml` | 사용자CONTACT의 손/상자 힘 gate만 제거한 비교용Stage1·전용train/test/VNC; 기존위치/높이·AT 유지 |
| `ports/ksh_stage1_stage2_base/push/tokenhsi/utils/push_random_spawn.py`, `*five_task_depth{180_20kg,200_20kg_friction04}` env/train·train/test/VNC | GPU6 물성의5과제·최종 목표1~5m·전신표면 간격/360° 무관통 생성·RSI/영상 [명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/five_task_gpu6_variants.md) |
| `ports/ksh_stage1_stage2_base/` | 외부 commit `28f02b3`의 읽기전용 `upstream/`·독립 `push/` 사본, SHA256 manifest/검증기. `push/`의 `stage1_push.yaml`·`humanoid_ma_stage1_push.py`는2명/4상자·기존AT·CONTACT·직선1~5m 별도 실행. [작업 규칙](../ports/ksh_stage1_stage2_base/README.md), [명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/stage1_push.md). 별도 `*five_task` 설정과 `humanoid_ma_five_task{,_stage2}.py`는 기존 네 과제+PUSH_AT 및 Stage2 PUSH 전용 coupled 코드. [명세](../ports/ksh_stage1_stage2_base/push/markdowns/reference/five_task_push_at.md) |
| `push_fixed130_point_zband_heightmatch_straight_goal2to5.yaml`, 동명 `_{train,test,vnc}.sh`·전용bank, `tests/test_push_straight_goal.py` | 초기 상자 로컬+Y 면 법선의2~5m 직선 목표·높이정렬11자세 유지; `open_push_scene.py::sample_goal_offsets`의 `box_forward_uniform`; [명령](config.md#push-상자-정면-직선-목표) |
| `tokenhsi/scripts/multi_agent/render_push_crop_comparison.py` | CPU 원본 전체/crop/기존 AMP3열 스켈레톤 MP4·metadata; `output/push_crop_comparison/`에 저장 |
| `markdowns/push_reward_physics_amp_handoff.md` | GPU6에서 확인한 높이 정렬 RSI·목표2~5m PUSH의 보상·물성·AMP/25개 데이터 절대 경로·BONES 사용 조건·BVH/AMP 변환·RSI 보정/선별 전달 명세; 정책 입력·네트워크 제외 |
| `prepare_push_heightmatch_goal2to5_resume.py`, `push_fixed130_point_zband_heightmatch_goal2to5.yaml`, 동명 `_{train,test,vnc}.sh` | 높이 정렬 RSI500회 상태보존·목표2~5m 별도 resume; [명령](config.md#push-높이-정렬-rsi-목표25m-resume) |
| `build_push_height_matched_rsi.py`, `push_fixed130_point_zband_heightmatch_goal3to5.yaml`, 동명 `_{train,test,vnc,rsi}.sh` | 기존408후보를 초기/첫물리높이·PD안정성으로 선별한 별도 RSI bank·GPU2학습/GPU6검증; [명령](config.md#push-높이-정렬-rsi) |
| `push_fixed130_point_zband_goal3to5.yaml`, 동명 `_{train,test,vnc,rsi}.sh`, `tests/test_push_point_zband.py`, `push_point.py::point_height_excess/valid` | 중심30D/570D입력·중점XY점/높이0.585~0.650m무감점·GPU2학습/GPU6검증; [명령](config.md#push-중심-입력점xy높이구간-보상) |
| `humanoid_ma_open_push_point.py`, `utils/push_point.py`, `push_fixed130_point_goal3to5.yaml`, 동명 `_{train,test,vnc,rsi}.sh`, `tests/test_push_point.py` | 기존 중심30D/570D 관측·고정0.6175m접촉점/양손중점보상·중심AT·공유 RSI검사기; [명령](config.md#push-중심-입력점-보상-scratch) |
| `humanoid_ma_open_push_anchor.py`, `utils/push_point.py`, `push_fixed130_anchor_goal3to5.yaml`, 동명 `_{train,test,vnc,rsi}.sh`, `check_push_anchor_rsi.py`, `tests/test_push_anchor.py` | 고정면/낮은점 앵커 관측30D·570D/위치·강체점속도·개별손면접촉(중점보상/gate없음)·낮은높이PD RSI; [명령](config.md#push-접촉-앵커-13m목표35m-scratch) |
| `analyze_push_height.sh`, `analyze_push_height_{forces,contacts,pd}.py`, `summarize_push_height_analysis.py` | GPU6비학습·1.0~1.5m/마찰0.6·실제humanoid/두물리손접촉/알려진힘·5도이하이동높이분석; [명령](config.md#push-크기별-손-중점-높이-물리-분석) |
| `push_fixed130_fixed_face_midpoint_goal3to5.yaml`, 동명 `_{train,test,vnc}.sh`, `utils/push_goal_resume.py`, `prepare_push_goal3to5_resume.py`, `tests/test_push_goal_resume.py` | 목표3~5m·충돌 없는 작업구역 확대·기존 fixed-face full-state resume 이관·전용 RSI bank; [명령](config.md#push-고정면-목표35m-resume) |
| `ports/ksh_object_finetune/object_tokenizer_copy.py`, `object_tokenizer_finetune_carry_chair_copied_adapter_{train,test,vnc}.sh` | 원래 Obj tokenizer 보존·복사본+adapter 학습·명시적 base/adapt 평가 분기; [명령](config.md#armchair-carry-copied-adapter) |
| `object_tokenizer_finetune_carry_chair_adapter_{train,test,vnc}.sh`, private 동명 env/train YAML | ArmChair 단독·Obj tokenizer+층별 action adapter·head 동결; [명령](config.md#armchair-carry-adapter) |
| `markdowns/object_finetune_train_guide.md` | 객체별 CARRY 의자/테이블·SIT의 config 표와 학습 명령만 모은 실행 가이드 |
| `ports/ksh_object_finetune/action_adaptation.py`, `finetune_contract.py`, `object_tokenizer_finetune_joint_{adapter,head}_{train,test,vnc}.sh` | 공동 SIT/CARRY 두 비교·기존 Obj tokenizer+층별 residual adapter 또는 기존 head 직접 학습; [설정/명령](object_tokenizer_finetune.md#공동-학습-비교-adapter--head) |
| `ports/ksh_object_finetune/`, `object_tokenizer_finetune_{sit_chair,carry_chair,carry_table}_{train,test,vnc}.sh`, `tests/test_object_finetune_contract.py` | KSH39000 독립 사본·기존 물체 tokenizer만 actor에서 fine-tuning·critic/AMP 학습·정확한 객체별 RSI·세 객체군 분리; [실행/검증](object_tokenizer_finetune.md) |
| `tokenhsi/scripts/multi_agent/compare_climb_size_range.py`, `evaluate_object_zero_shot.py::{size_in_range,prepare,load_objects}` | CLIMB 학습 범위 native 필터·동일 bbox/질량 상자 대조·초기상태/실제 bbox/질량/마찰 감사; [명령](object_zero_shot.md#climb-학습-크기-범위-내-비교) |
| `evaluate_object_zero_shot.py::climb_ascent_success`, `tests/test_object_zero_shot.py` | 중앙 root 목표·종료 유지 없는 CLIMB 올라가기 사건 집계·양발 윗면 geometry/접촉·최초 step/연속3step 및 기존 성공률 병기; [명령](object_zero_shot.md#climb-올라가기-사건-평가) |
| `tokenhsi/scripts/multi_agent/compare_seat_proxy.py`, `evaluate_object_zero_shot.py::{support_proxy_geometry,proxy_policy_observation}` | KSH39000의 SIT·의자 CLIMB 비학습 좌판 bbox 보정·GTA origin/속도 일관성·3조건 동일 초기상태/물성 대조; [명령](object_zero_shot.md#좌판-입력-보정-sitclimb) |
| `tokenhsi/scripts/multi_agent/evaluate_object_zero_shot.py` (`--box-friction` 실제 shape 평가 override·`--record-video` 장면별 viewer MP4), `object_zero_shot_{test,vnc}.sh`, `tokenhsi/tests/test_object_zero_shot.py` | KSH39000을 읽기 전용으로 재사용하는 native 객체5종 평가·634D/물체30D 유지·판정 전용 좌판 metadata·원본 무변경 감사; [실행/제한](object_zero_shot.md) |
| `open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_tilt035.yaml`, 동명 `_{train,test,vnc}.sh`, `convert_push_only_at_tilt_checkpoint.py`, `utils/at_tilt_checkpoint.py::migrate_push_only_checkpoint`, `tests/test_push_only_at_tilt.py` | 현재GPU6과동일570D/물체30D/PUSH전용·AT성공기울기만0.35rad제한·상태보존이관·공통validate_at_tilt.py로RSI/AMP검증 |
| `open_push_contact_at_friction06_random_size_goal1to25_at01_tilt035.yaml`, 동명 `_{train,test,vnc}.sh`, `utils/at_tilt_checkpoint.py`, `convert_at_tilt_checkpoint.py`, `validate_at_tilt.py`, `tests/test_at_tilt.py` | 문 포함 AT01의 AT 성공에만0.35rad 기울기 제한·학습 상태 보존 checkpoint 이관·비학습 RSI/AMP/정책 검증 |
| `tokenhsi/utils/push_only.py`, `convert_push_only_checkpoint.py`, `validate_push_only.py`, `tokenhsi/tests/test_push_only.py`, `dataset_push_only_push180.yaml`, `push_only_friction06_random_size_goal1to25_bank.json` | 문 설정 없는 PUSH 계약·기존30D checkpoint의 엄격한 이관·물리 actor/RSI/부분reset 감사·보상 동등성 검사·loco/PUSH 전용 데이터 |
| `tokenhsi/utils/push_policy_observation.py`, `tokenhsi/tests/test_push_no_doors.py`, `open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors.yaml`, 동명 `_{train,test,vnc}.sh` | PUSH 전용: 물리 문 actor/관절 및 문 학습설정/모션 삭제·상자30D/570차원/6토큰·scratch·기존30D는 명시적 이관 후 resume |
| `open_push_contact_at_friction06_random_size_goal1to25_at01.yaml`, `open_push_contact_at_friction06_random_size_goal1to25_at01_{train,test,vnc}.sh`, `build_open_push_friction06_random_size_goal1to25_at01_rsi.sh`, `learning/multi_agent/open_push_transfer.py::goal_progress_source` | goal25 actor/AMP/RMS 전이·goal1~2.5m·AT만0.1m포화·critic/optimizer초기화; 기존1~2.5m물리RSI bank재사용 |
| `tokenhsi/scripts/multi_agent/measure_push_hand_motion.{py,sh}` | checkpoint별 비학습 물리tick 감사: 목표방향과무관한 박스표면 한손/양손 힘접촉·접촉중XY누적거리·순변위·학습RSI혼합/보행시작분리; audit/trials JSON |
| `tokenhsi/scripts/multi_agent/view_open_push_far.py`, `open_push_far_record.sh`, `tokenhsi/utils/segmented_video.py` | 기존 warmstart29000 평가의4m 시작·문 visual 숨김·VNC/별도카메라 연속MP4; GUI GPU초기화·평가녹화시작·메모리상수 streaming/구간마감; 기본저장 `output_records/videos/<실행시각>/` |
| `tokenhsi/utils/open_push_scene.py::goal_travel_limit/sample_goal_offsets`, `open_push_contact_at_friction06_random_size_goal1to25.yaml`, `open_push_contact_at_friction06_random_size_goal1to25_{train,test,vnc}.sh`, `build_open_push_friction06_random_size_goal1to25_rsi.sh` | PUSH 중심목표1~2.5m 거리균등·360°·전용bank/output·scratch; RSI 초기 남은거리/AT성공 건수 감사 |
| `tokenhsi/utils/open_push_scene.py::goal_travel_limit/sample_goal_offsets`, `tokenhsi/tests/test_push_goal_radius.py`, `tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal25.yaml`, `tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_goal25_{train,test,vnc}.sh`, `build_open_push_friction06_random_size_goal25_rsi.sh` | PUSH 목표중심거리0.5~1m/1~2.5m 각50%·360°균등·7m작업구역·전용PD RSI bank·scratch 학습; 기존정사각목표분포보존 |
| `tokenhsi/scripts/multi_agent/view_open_push_random_goal.py`, `open_push_zero_shot_random_goal_{test,vnc}.sh` | 기존 warmstart10000/29000 제로샷 PUSH 평가·시각화; 원래 checkpoint 계약 검사 유지·loco reset 후 목표만XY±1.5m/거리1~2m로교체·초기장면/goal JSONL 기록 |
| `tokenhsi/scripts/multi_agent/check_push_physics_envelope.{py,sh}`, `check_push_actuator_load.{py,sh}` | 정책/학습없는 actual PD effort감사·발마찰이상적상한과박스힘대조·864조건직접PD부하/지지검사; 전역상한/실현가능구동/센서기록을구분 |
| `tokenhsi/scripts/multi_agent/check_open_push_policy_capacity.{py,sh}` | GPU6 실제29000정책의 무게11×마찰10 actor생성시 물성검사·같은초기배치·접촉/보행시작·외력없는 안정이동/전도 감사 |
| `tokenhsi/data/cfg/multi_agent/push_contact_at_friction06_free_surface_state09.yaml`, `tokenhsi/scripts/multi_agent/push_contact_at_friction06_free_surface_state09_{train,test,vnc}.sh`, `build_push_free_surface_state09_rsi.sh`, `tokenhsi/tests/test_push_free_surface.py` | PUSH自由 표면 XYZ거리/양손반력·고정1.3m·state≥0.9 성공; `open_push_reward.py::push_any_surface_geometry`·물리bank재사용 |
| `tokenhsi/data/cfg/multi_agent/push_contact_at_friction06_fixed_size_state09.yaml`, `tokenhsi/scripts/multi_agent/push_contact_at_friction06_fixed_size_state09_{train,test,vnc}.sh`, `build_push_fixed_size_state09_rsi.sh` | PUSH100%·고정1.3m/20kg·가중치/saturation 적용 전  state평균≥0.9 성공 비교; `open_push_reward.py::contact_at_reward`, `test_push_contact_at.py` 경계/회귀검증 |
| `tokenhsi/utils/open_push_sizes.py`, `tokenhsi/tests/test_push_random_size.py`, `tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size.yaml`, `tokenhsi/scripts/multi_agent/open_push_contact_at_friction06_random_size_{train,test,vnc}.sh`, `build_open_push_friction06_random_size_rsi.sh` | PUSH만 학습·생성시31종균등크기/동일밀도·박스별실제기하·전체크기공통PD RSI bank·전용output |
| `tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction{06,1}_goal_face.yaml`, `tokenhsi/scripts/multi_agent/open_push_contact_at_friction{06,1}_goal_face_{train,test,vnc}.sh`, `build_open_push_friction{06,1}_goal_face_rsi.sh`, `tokenhsi/tests/test_push_goal_face.py` | 현재20kg/박스0.6/바닥1·랜덤시작/목표·목표반대면PUSH/RSI·대각선면경계·전용bank/checkpoint계약;1은비교용 |
| `tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction1_random_goal.yaml`, `tokenhsi/scripts/multi_agent/open_push_contact_at_friction1_random_goal_{train,test,vnc}.sh`, `build_open_push_friction1_random_goal_rsi.sh` | PUSH 마찰1·ksh식 독립XY목표·29000 actor/AMP 전이·새PD RSI bank; 목표 샘플링은 `utils/open_push_scene.py` |
| `tokenhsi/learning/multi_agent/open_push_transfer.py`, `tokenhsi/tests/test_open_push_transfer.py`, `tokenhsi/data/cfg/multi_agent/open_push_push180_contact_at_warmstart.yaml`, `tokenhsi/scripts/multi_agent/open_push_push180_contact_at_warmstart_{train,test,vnc}.sh` | OPEN/PUSH random-scene actor·AMP·입력RMS를 접촉+AT에 전이; critic/optimizer/카운터 초기화·source 계약검사·GPU 선택 실행 |
| `tokenhsi/data/cfg/multi_agent/open_push_push180_contact_at.yaml`, `tokenhsi/scripts/multi_agent/open_push_push180_contact_at_{train,test,vnc}.sh`, `tokenhsi/tests/test_push_contact_at.py` | PUSH 양손 접촉과 기존 AT 배치의 독립 edge 보상 합산·scratch 실험; `open_push_reward.py::contact_at_reward`, 기존 물리 RSI screen 재사용·별도 checkpoint 보상 계약 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_open_push.py`, `open_push_reward.py`, `tokenhsi/utils/open_push_spec.py` | 사람별 OPEN/PUSH50/50 graph·PUSH 중점XY/높이자유·뒤면XY/양손반력 접촉·혼합 actor·task 조건 AMP·revision5 checkpoint 계약 |
| `tokenhsi/utils/open_push_rsi.py`, `tokenhsi/data/open_push_late_rsi_cube_low_bank.json` | task별 PD검사 후보·clip 균등 후반 선택·발/몸통 보존 어깨 회전으로 낮은 손 RSI·후보 hash 계약 |
| `tokenhsi/data/cfg/multi_agent/open_push_late_rsi.yaml`, `tokenhsi/scripts/multi_agent/open_push_late_rsi_{train,test,vnc}.sh` | 보행/조작50/50·후반 우선 별도 실험; 기존 공동 train/network YAML 재사용 |
| `tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction.yaml`, `tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_{train,test,vnc}.sh`, `build_open_push_hinge_friction_rsi.sh` | 문 힌지 마찰0.5 비교; 전용 `open_push_late_rsi_cube_low_hinge_friction_bank.json`·별도 output, 기존 물리 checkpoint 직접 복원 불가 |
| `tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction_push2m.yaml`, `tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_push2m_{train,test,vnc}.sh`, `build_open_push_hinge_friction_push2m_rsi.sh` | 기존1.5m 보존·PUSH2m·마찰0.5·문/박스 가로8m·구역10m·전용 wide bank/output; 학습·1500 전이 미실행 |
| `tokenhsi/scripts/multi_agent/transfer_open_push_hinge_friction.{py,sh}` | revision5 마찰0→0.5만 검증한 checkpoint 전이; 동일 RSI 후보 요구·모델/optimizer/epoch 보존·원본 덮어쓰기 금지 |
| `tokenhsi/scripts/multi_agent/check_open_push_layout.{py,sh}`, `utils/open_push_spec.py::scene_layout` | 구역 배치 검증·기존 좌표 기본값·네 과제/보행·RSI 위치·목표/부분 reset/짧은 PD 상호 간격 감사 |
| `tokenhsi/scripts/multi_agent/build_open_push_rsi.{py,sh}`, `tokenhsi/tests/test_open_push_rsi.py` | 비학습2환경12step PD 후보 검사·학습 reset/AMP/partial reset 감사 및 CPU 비율/정렬 검증 |
| `tokenhsi/scripts/multi_agent/check_open_push_physics.{py,sh}` | 비학습2환경의 상자 모양/힘 높이 비교·외력 대조군 및 원본 손 Z를 보존한 사람 PD·기울기/바닥 edge 이동 JSON |
| `tokenhsi/scripts/multi_agent/build_open_push_rsi_timeline.{py,sh}`, `viewer_assets/open_push_rsi_timeline.html` | CPU 전체 AMP clip·원본/낮춘 팔·RSI 후보/후반 표시·프레임 슬라이더·무생략 재생 HTML 생성; checkpoint 불필요 |
| `tokenhsi/scripts/multi_agent/view_open_push_rsi.py`, `open_push_rsi_view{,_vnc}.sh` | checkpoint 없는1환경 RSI 자세 검사·화면 모션정보·키보드 샘플/카메라/12step PD·JSON 샘플 기록 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/open_push_unified_amp.yaml`, `tokenhsi/data/dataset_open_push.yaml`, `tokenhsi/scripts/multi_agent/open_push_unified_amp_{train,test,vnc}.sh` | 문/박스 두 과제 공동 정책 설정·보행4/문1/밀기2 manifest·GPU6 전용 실행 |
| `tokenhsi/scripts/multi_agent/convert_bones_push.py`, `tokenhsi/data/dataset_bones_push/` | 선택한 BONES 밀기2개의30FPS/15body/32DOF 변환·수치 보고서·preview 데이터 |
| `tokenhsi/scripts/multi_agent/transfer_open_push_push180.{py,sh}`, `output/open_push_late_rsi_cube_low_midpoint_xy_hinge_friction_push2m_wide_push180/latest_launch.json` | 기존마찰0.5·1.5m·2모션 checkpoint에서 PUSH2m·넓은배치·21모션/RSI로 full-state 명시적 전이 및 현재 tmux 본학습 위치 |
| `tokenhsi/data/bones_push180_selection.json`, `tokenhsi/data/dataset_bones_push180/`, `tokenhsi/data/dataset_open_push_push180.yaml` | 요청180 계열21개의crop/동작 구간·신규19개 retarget·기존2개를 포함한 AMP 목록; 변환은 `convert_bones_push.py --expanded-180` |
| `tokenhsi/data/cfg/multi_agent/open_push_late_rsi_hinge_friction_push2m_push180.yaml`, `tokenhsi/scripts/multi_agent/open_push_late_rsi_hinge_friction_push2m_push180_{train,test,vnc}.sh`, `tokenhsi/scripts/multi_agent/build_open_push_hinge_friction_push2m_push180_rsi.sh` | PUSH2m 넓은 배치·21개 AMP·전용 RSI bank/output; 기존 실행과 분리 |
| `tokenhsi/utils/open_push_lanes.py`, `tokenhsi/data/cfg/multi_agent/open_push_push180_random_lanes.yaml`, `tokenhsi/scripts/multi_agent/open_push_push180_random_lanes_{train,test,vnc}.sh` | reset별 담당구역50/50 교환·물체 고정·관측/RSI/접촉/credit의 owner mapping; 기존 PUSH180 보상/bank 유지 |
| `tokenhsi/utils/open_push_scene.py`, `tokenhsi/data/cfg/multi_agent/open_push_push180_random_scene.yaml`, `tokenhsi/scripts/multi_agent/open_push_push180_random_scene_{train,test,vnc}.sh` | 고정1.3m PUSH박스의실제XY/yaw·local전방2m목표·1~2m시작·독립물리binding·침투/작업영역검사; OPEN고정힌지유지 |
| `tokenhsi/scripts/multi_agent/transfer_open_push_random_scene.{py,sh}`, `tokenhsi/scripts/multi_agent/validate_open_push_scene.{py,sh}`, `tokenhsi/tests/test_open_push_scene.py` | 단일PD후보SE(2)재사용전용bank·scene전이·실제reset/RSI검사; 결과 `output/open_push_random_scene_sim_check/audit.json` |
| `tokenhsi/scripts/multi_agent/transfer_open_push_random_lanes.{py,sh}`, `tokenhsi/scripts/multi_agent/validate_open_push_lanes.{py,sh}`, `tokenhsi/tests/test_open_push_lanes.py` | 고정구역PUSH180 full-state를 구역 계약만 전이·네task/두배정/부분reset/실제RSI 감사; 결과 `output/open_push_random_lanes_sim_check/`,2048환경resume `output/open_push_random_lanes_resume_check/audit.json` |
| `tokenhsi/scripts/multi_agent/validate_open_push.{py,sh}`, `tokenhsi/tests/test_open_push.py` | 공동 sampler·RSI 초기 progress + physics tick별 OPEN 손 개방각/PUSH 양손 전진 credit·RSI 기준점/85도 OPEN 후보·grounded 성공·2-family AMP/replay·변환·GPU6 reset/네트워크/비학습 검증 |
| `tokenhsi/run.py` | 학습·평가 태스크와 알고리즘 등록 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_door_hold.py`, `tokenhsi/utils/door_hold.py`, `tokenhsi/assets/door/door_hold.urdf` | 1인90도 개방·골반/torso/발 지지 자세 판정·연속 손 지지, reset별1.5~2.5m 거리 샘플링, 자동 닫힘 spring, 접촉 공백 처리·타이머 관측, 바닥 틈이 있는 전용 문 |
| `tokenhsi/data/cfg/multi_agent/door_open_hold_amp.yaml`, `tokenhsi/data/cfg/train/rlg/door_open_hold_amp.yaml`, `tokenhsi/scripts/multi_agent/door_open_hold_amp_{train,test,vnc}.sh` | 사람1·문1·BONES full AMP의 별도 scratch 학습/평가/viewer 설정 |
| `tokenhsi/scripts/multi_agent/validate_door_hold.{py,sh}`, `tokenhsi/tests/test_door_hold.py` | 비학습 손 접촉 개방·발 지지 부족의 거짓 성공 차단·release/자동 닫힘·AMP discriminator 및 CPU90도/자세/5초 유지 조건 검사 |
| `bones_seed/metadata/`, `bones_seed/soma_uniform.tar.gz`, `bones_seed/soma_uniform/bvh/`, `bones_seed/validation/REPORT.md` | BONES-SEED metadata·공식 SHA256 검증 archive·선별 문184개·PUSH71개 원본 BVH·원본 수치 검증/로컬 skeleton 재생 |
| `bones_seed/validation/support_lift_candidates/` | 협동 승강 확장용 원본 BVH4개·손목 상승 분석·4행 자세 비교·후보 보고서; 지지 접촉/협동 물리는 미검증 |
| `bones_seed/validation/requested_push_71/` | 요청 PUSH71개(180/270/360)의 metadata 동작 구간·선별 추출 경로/크기/SHA256·원본 BVH parser/FK 검증; AMP/RSI 적용 범위는 선택 대기 |
| `tokenhsi/utils/bones_bvh.py`, `tokenhsi/scripts/multi_agent/convert_bones_door.py` | SOMA BVH 절대 Hips/XYZ·Euler FK parser와 선택한 왼손 clip의 humanoid constrained IK·crop·30FPS SkeletonMotion 변환 |
| `tokenhsi/data/dataset_bones_door/`, `tokenhsi/data/dataset_door_bones_left.yaml` | 변환된103frame AMP motion·매핑/수치 보고서·현재 문을 겹친 CPU HTML/GIF·loco/doorOpen manifest |
| `tokenhsi/scripts/multi_agent/build_bones_door_preview.py`, `validate_bones_door_amp.{py,sh}`, `tokenhsi/tests/test_bones_door_amp.py` | CPU 문 형상 비교 미리보기·GPU6 전용 비학습 loader/AMP/FK/replay 검증·좌표/관절/설정 회귀 검사 |
| `tokenhsi/data/cfg/multi_agent/door_open_bones_left_amp.yaml`, `tokenhsi/scripts/multi_agent/door_open_bones_left_amp_{train,test,vnc}.sh` | 기존 loco reset·OPEN reward와 full AMP를 유지하는 BONES 왼손 별도 실험; 정책 성공 미검증 |
| `ports/ksh_stage1/`, `tokenhsi/scripts/multi_agent/ksh_*` | ksh 세 방식의 독립 스냅샷·전용 train/test/VNC; 기존 학습 코드와 분리, [실행·격리](ksh_stage1_port.md) |
| `tokenhsi/scripts/multi_agent/` | 현재 실험별 train/test/VNC, `runtime_env.sh`, `run-gui.sh` |
| `tokenhsi/data/cfg/multi_agent/` | 현재 실험 YAML과 평가 graph 예제 |
| `tokenhsi/data/cfg/train/rlg/` | PPO/AMP/Transformer 학습 설정 |
| `output/<실험>/<run>/` | checkpoint `nn/`, TensorBoard `summaries/`, 진단 `diagnostics/` |

2026-10-08 OPEN/PUSH 확인용 output30개를 정리했다. 본학습3개(`open_push_90_standing`, `open_push_late_rsi_cube_low`, `open_push_late_rsi_cube_low_joint_hand_credit`)와 학습용 코드/설정·모션·RSI bank는 유지한다. 과거 검사 output 경로는 남아 있지 않을 수 있다.

## 현재 실험이 사용하는 코드

실제 박스 크기를 사용하는 새 progress 설정과 실행 명령은 [size_progress.md](size_progress.md)를 참조한다. RSI 변경은 별도로 진행한다.
현재 task를 따르는 AMP 선택과 새 설정의 활성 상태는 [task_amp.md](task_amp.md)에 기록했다.
학생 반복 검사 기반 크기별 RSI sampler와 학습 실행 명령은 [student_rsi_sampling.md](student_rsi_sampling.md)에 있다.
같은 크기의 후반 PD/현재 정책 hybrid RSI 구현과 실행은 [hybrid_late_rsi.md](hybrid_late_rsi.md)에 있다.

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_door.py`, `edge_door_reward.py` | 기존 Carry 기반 2H/2Door 물리 상태·독립 OPEN reward·진단 |
| `tokenhsi/assets/door/door_pull.urdf`, `tokenhsi/data/cfg/{multi_agent,train/rlg}/door_pull_unified_approach_amp.yaml`, `tokenhsi/scripts/multi_agent/door_pull_unified_approach_amp_*` | 당김 −Z hinge·손잡이 접촉 공간·별도 train/test/VNC; grasp 제약 없음 |
| `tokenhsi/utils/door_approach_amp.py`, `tokenhsi/data/dataset_door_loco.yaml`, `tokenhsi/tests/test_door_approach_amp.py` | 선택적 own handle taper/latch·gate_enabled=false이면 전 구간 AMP1·loco-only expert/RSI·CPU 검증 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/door_open_unified_approach_amp.yaml`, `tokenhsi/scripts/multi_agent/door_open_unified_approach_amp_*` | 기존 unified + 몸/손0.2·전 구간 loco AMP 전용 train/test/VNC |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/door_open_shared_edge_approach_amp.yaml`, `tokenhsi/scripts/multi_agent/door_open_shared_edge_approach_amp_*` | 같은 몸/손0.2·전 구간 loco AMP에서 Actor/Critic EdgeEncoder 공유 비교·별도 train/test/VNC |
| `tokenhsi/utils/door_reference.py`, `tokenhsi/data/dataset_door_proxy.yaml` | 기존 상호작용 모션의 Door 기하 조건/AMP history 선별·후반 RSI·own handle 정렬 |
| `tokenhsi/tests/test_door_reference.py`, `tokenhsi/data/cfg/{multi_agent,train/rlg}/door_open_{unified,shared_edge}_amp_on.yaml` | 기존 unified/shared 파라미터 공유 차이·같은 Door proxy AMP/RSI 비교 |
| `tokenhsi/utils/edge_door_spec.py`, `tokenhsi/assets/door/door.urdf` | OPEN=11·고정 scenario binding·schema11 계약, repo 소유 passive hinge asset |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/door_open_amp_{off,on}.yaml`, `tokenhsi/scripts/multi_agent/door_open_amp_*` | 기존 MAAgent/shared-edge를 쓰는 AMP ON/OFF 동일 PPO 설정·train/test/VNC |
| `tokenhsi/scripts/multi_agent/view_door_reset.py`, `door_reset_view.sh`, `door_reset_vnc.sh` | checkpoint 없이 Door GPU reset 1step 자세를3초씩 보여주는 비학습 뷰어·headless 진단 |
| `tokenhsi/scripts/multi_agent/validate_door_physics.py`, `validate_door_physics.sh` | CPU 한 코어·1스레드 비학습 2환경: 무접촉20초·실제 손 접촉·물리/관측/전체 reward 대조 |
| `tokenhsi/scripts/multi_agent/check_door_open_cpu.py` | 학습 없는 2환경 CPU PhysX·hinge 회전·handle 위치·부분 reset·AMP demo 검사 |
| `tokenhsi/tests/test_door_open.py` | CPU binding·reward·current success·부분 reset·관측·passive DOF·공유 encoder·AMP OFF 검증 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` | 물체 배정·RSI·물리 환경·관측 통합 |
| `tokenhsi/utils/hybrid_late_rsi.py` | 동일 source/support XYZ의 후반 후보 선택·관측 창·checkpoint 상태 |
| `tokenhsi/env/tasks/multi_agent/hybrid_rsi_screen.py`, `hybrid_rsi_runtime.py` | 초기 PD 캐시·현재 정책 초기 관측·task별 XYZ 완료 episode 진단 |
| `tokenhsi/tests/test_hybrid_late_rsi.py` | 후반 단계 fallback·크기/받침 보존·관측/복원·물리 검사·YAML 계약 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_task.py`, `edge_context_task.py` | graph reset, 가까운 시작, 물리 타당성·진단 |
| `tokenhsi/env/tasks/multi_agent/edge_stage1_reward.py` | Stage 1 edge 보상·포화·팀 공유 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_reward.py`, `edge_interaction_reward.py` | AT·ON_TOP·SIT·CLIMB state/progress와 plane 성공 |
| `tokenhsi/utils/edge_scenario_spec.py` | 독립 물체·goal binding, 3물체·4물체 같은 과제 쌍 및 canonical/무작위 binding 독립 5과제 sampler |
| `tokenhsi/utils/unified_training.py` | unified 과제별 AMP 전문가 분포·라벨·리플레이 매칭·설정 검증 |
| `tokenhsi/utils/edge_stage1_spec.py` | Stage 1 variant·semantic/owner-HOLDING packet·후반 CLIMB/ON_TOP putDown RSI 검증 |
| `tokenhsi/utils/edge_stage2_spec.py` | Stage 2 협력 graph·평가 preset |
| `tokenhsi/utils/relation_task_spec.py` | reward config·checkpoint 계약 검증 |
| `tokenhsi/learning/multi_agent/amp_network_builder_ma.py`, `edge_context_encoder.py`, `ma_agent.py` | GTA actor/critic, edge bias·HOLDING 상태 융합, AMP·PPO 학습 |
| `tokenhsi/learning/multi_agent/coordination_head.py`, `stage2_transfer.py` | 협력 attention·Stage 1 weight 이식 |
| `tokenhsi/tests/test_scenario_independent_with_climb.py`, `test_scenario_stage1_plane.py`, `test_scenario_stage1_sit_plane_curriculum.py`, `test_stage2_coordination.py` | 현재 실험 CPU 검증 |
| `tokenhsi/tests/test_stage1_unified.py` | 세 unified config 계약·공유 edge gradient·독립 분포·goal 회전·AMP family 매칭·토큰 순열의 actor/value 및 gradient 검증 |
| `tokenhsi/tests/test_unified_shared_random_binding.py` | 사용자 12옵션 계약·checkpoint 분리·무작위 target·부분 reset의 near-start/penalty 연결 검증 |
| `tokenhsi/tests/test_graph_box_speed_penalty.py` | graph 담당 물체의 속도 패널티·부분 reset history·기존 할당 경로 회귀 검증 |

`edge_context_spec.py`, `edge_ontop_spec.py`, `edge_interaction_spec.py`와 관련 reward 함수는 과거에 도입됐지만 현재 Stage 1·2의 graph·관찰·보상 계산에서도 호출된다. 파일명만 보고 삭제하면 현재 실험이 깨진다.

## 작업별 확인

- **GPU4 PUSH8500 크기·무게별 평가:** `output/push_gpu4_size_mass_audit_20261015/`의 `evaluate.py`·`run.sh`는 GPU1 비학습 측정, `report.py`는 장면 bootstrap/CSV/PNG 생성, `REPORT.md`·`full/audit.json`·`full/trials.json`은 집계/원시결과다. 31크기·학습 RSI 혼합/보행 시작 분리·yaw/기울기/실제 전진량을 확인한다. [명령](config.md#gpu4-push8500-크기무게별-비학습-평가).

- **PUSH180 손으로 밀 수 있는지:** `output/push180_hand_physics_audit_20261009/REPORT.md`·`audit.py`·`arms_only.py`. `render_physics.py`는 clip0/16 물리 재생영상16개, `index.html`은6090 비교페이지, `videos/audit.json`은 영상재실행 측정값이다. 현재물성 외력대조·원본/낮춘팔PD·손전용상자충돌·초기속도0/보행제거 팔8DOF 재생 검사이며 안정적2m 정책 성공과 구분한다.

- **OPEN/PUSH 접근·edge bias 추이:** `open_push_reward.py:approach_xy_diagnostics`와 `humanoid_ma_open_push.py:consume_relation_diagnostics`가 task별 XY 거리·접근 점수·조작 progress를 집계한다. `edge_context_encoder.py:PackedEdgeSemanticFusion.bias_diagnostics`는 canonical 유효 관계의 실제 bias 평균·RMS·표본 수를 계산하고 `amp_network_builder_ma.py`→`ma_agent.py`가 Actor/Critic별 TensorBoard에 기록한다. 의미·태그·기존 실행 적용 범위는 [config.md](config.md)의 OPEN/PUSH 절을 참조한다.

- **Door 접근 정체 진단:** `output/door_approach_debug_20261007/REPORT.md`·`audit_policy.py`, 각 `run1/run2/audit.json`·`trajectory.csv`. 학습 없이 epoch10500 정책의 관측·action·접근과 거리 AMP gate 손실을 확인한다. 역사 checkpoint의 원래 reward config로 평가하며 새 계수 학습과 구분한다.

- **Door 수정 후 전체 환경 reset/PPO 검증:** `output/door_reset_v2_validation_20261006/`의 `audit_train.py`·`train_check_complete.log`; 기존 결과와 분리한 두 학습 output의 `latest_launch.json`에서 PID·로그를 확인한다.

- **5개 task Z별 평가:** `scripts/multi_agent/evaluate_z_sweep.py`와 `.sh`, `utils/z_sweep.py`, `tests/test_z_sweep.py`. [고정 XY·크기/위치 높이·ON_TOP 받침 분리](z_sweep.md).

- **3개 학습 추이 기록:** [2026-10-03 성공률·edge bias 비교](training_three_runs_success_bias_20261003.md). PID 2978991·2734782·1099002의 1000 epoch 구간별 종료/도달 성공률과 같은 checkpoint epoch의 task별 actor/critic bias를 보존한 시점 기록이다.

- **충격 기반 RSI 후보 검사:** `utils/impact_candidates.py`로 예비 원본 시점을 추출하고
  `scripts/multi_agent/probe_student_candidates.sh`로 현재 학생 정책에서 재검사한다.
  [명령과 검사 제한](reference_rsi_candidate_commands.md). 학습 sampler에 자동 적용하지 않는다.

- **GPU 메모리 계측:** `../../runs/memory_audit_289953/profile_train.py`는 별도 실행용 단계별 계측 진입점이다. 실행 스크립트·원시 JSONL·REPORT.md를 같은 디렉터리에 보관한다. 기존 장기 학습에는 적용되지 않는다.

- **랜덤 크기 참조 RSI:** `utils/reference_rsi.py`의 크기별 모션/시점 sampler·참조 박스 속도·전 장면 관통 screen,
  `scripts/multi_agent/build_reference_rsi_bank.py`의 CPU 후보표 생성,
  `scripts/multi_agent/probe_reference_rsi.py`의 teacher 전용 물리 검사.
  두 config·train/test/VNC와 실행법은 [reference_rsi.md](reference_rsi.md)에 있다.
  박스 크기 유지·후반 RSI를 통한 학습 가속 목표, 크기별 실측 결과와 미검증 제안은
  [reference_rsi_archive.md](reference_rsi_archive.md)에 기록했다.
  X/Y/Z별 실측 충격과 해석 제한은 [reference_rsi_axis_analysis.md](reference_rsi_axis_analysis.md),
  PID 1184210의 당시 RSI 설정은 [rsi_pid1184210_audit.md](rsi_pid1184210_audit.md)에 있다.

- **Stage 1 graph·보상:** 해당 YAML → `edge_scenario_spec.py` → `edge_ontop_task.py` → `edge_interaction_reward.py`·`edge_stage1_reward.py`; 위 Stage 1 테스트로 검증한다.
- **Stage 2 협력·전이:** `approach_stage2_coordination.yaml` → `edge_stage2_spec.py` → `coordination_head.py`·`stage2_transfer.py`; `test_stage2_coordination.py`로 검증한다.
- **실행·GPU:** [config.md](config.md), `runtime_env.sh`, `mps/`, `run-gui.sh`를 확인한다.
- **지표 해석:** [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 확인한다.

경첩 저항은 `HumanoidMADoor._build_box`에서 선택적 `env.door.hinge`로 적용한다. `validate_door_physics.py`의 `hinge_impulse_coast`는 설정이 있는 실험의 무접촉 충격 후 멈춤을 검사한다. 결과는 `output/door_hinge_resistance_check_20261007/`에 저장한다.

로컬 절충 경첩 학습 인계와 복사용 Codex 프롬프트: [door_local_training.md](door_local_training.md).

## PUSH 무게별 두 정책 비교

- `tokenhsi/scripts/multi_agent/evaluate_push_mass_success.py`: 기존 크기·실제 질량 평가 + 성공 에피소드의 첫 성공/성공 중 최대 기울기 집계.
- `tokenhsi/scripts/multi_agent/compare_push_mass_success.sh`: 같은 GPU에서 두 최신 PUSH 정책 동시 headless 평가.
- `tokenhsi/scripts/multi_agent/report_push_mass_success.py`: 무게별 성공률·조건부 기울기 CSV/Markdown 출력.

## PUSH 1번 AT 기울기 감쇠·1.20~1.40m

- `tokenhsi/data/cfg/multi_agent/push_size120to140_at_tiltfade.yaml`, `tokenhsi/scripts/multi_agent/push_size120to140_at_tiltfade_{train,test,vnc}.sh`: PUSH-only scratch·21크기·AT state/progress5~15° 감쇠·성공≤10° 전용 실행.
- `tokenhsi/utils/push_size_subset.py`, `tokenhsi/scripts/multi_agent/build_push_size_subset_bank.py`, `tokenhsi/data/push_size120to140_at_tiltfade_bank.json`: 기존31크기PD통과후보를21크기strict subset에 재사용하고 출처계약을 보존.
- `tokenhsi/scripts/multi_agent/check_push_at_tiltfade_rsi.py`, `tokenhsi/tests/test_push_at_tiltfade.py`: 실제크기/질량·RSI/AMP/부분reset 검사와 AT전용감쇠·PUSH불변·경계·구checkpoint거부 CPU 검증.

## PUSH 2번 고정1.30m·20kg·AT 기울기 제한 없음

- `tokenhsi/data/cfg/multi_agent/push_fixed130_20kg_no_tilt.yaml`, `tokenhsi/scripts/multi_agent/push_fixed130_20kg_no_tilt_{train,test,vnc}.sh`: 가변크기를제거한문없는PUSH 전용scratch·기존AT보상/성공조건유지.
- `tokenhsi/scripts/multi_agent/build_push_fixed130_bank.py`, `tokenhsi/data/push_fixed130_20kg_no_tilt_bank.json`: 공통size subset검증기로기존31크기PD통과후보를고정1.30m에재사용하고출처보존.
- `tokenhsi/scripts/multi_agent/check_push_fixed130_rsi.py`, `tokenhsi/tests/test_push_fixed130.py`: 고정실제크기/질량·RSI/AMP/부분reset·고정/가변checkpoint계약분리검증.

## 고정 PUSH 이동 중 기울기 평가

- `output/push_fixed130_transit_tilt_20261017/evaluate.py`, `run.sh`: 고정 PUSH 세 저장본의 첫 성공 전 기울기·순 목표접근량·연속 기울기 구간 및 control-step 전체 궤적 평가.
- 같은 경로의 `report.py`: 성공 조건부/전체 빈도·scene bootstrap CI·저장 NPZ에서 지표 독립 재계산. 결과는 `REPORT.md`·`summary.json`, 원시는 `epoch_*/trials.json`·`trajectory_*.npz`.

## 고정 PUSH 접촉면 분석·최신 엄격평가

- `output/push_fixed130_contact_analysis_20261017/`: `evaluate.py`/`run.sh`는손/힘/상자전체궤적수집, `geometry.py`/`analyze.py`는실제3D면접촉추정과기울기직전대조, `ANALYSIS_REPORT.md`/`analysis.json`은분석.
- 같은경로의 `strict_report.py`와 `output/push_fixed130_contact_latest16500_20261017/run.sh`: 최신16500평가및90%접촉형태비교. `STRICT_REPORT.md`/`strict_summary.json`/`strict_episodes.json`에전저장본비교저장.

## 고정 PUSH 마찰별 viewer·warmstart 실제손높이

- `tokenhsi/scripts/multi_agent/view_push_friction.py`, `push_fixed130_friction_{test,vnc}.sh`: 기존checkpoint계약을보존한평가전용상자shape마찰override(0.2/0.4/0.6). metadata는friction_override.json.
- `output/warmstart29000_hand_height_20261017/`: warmstart29000의원래설정PUSH추론·손높이측정. `run.sh`/`report.py`/`REPORT.md`/`hand_height.json`·전체trace_NPZ.

## PUSH 전도 직전 모멘트 CPU 분석

- `output/push_tipping_moment16500_20261017/analyze.py`/`report.py`: 16500보행시작저장궤적에서전도축손모멘트·중력복원·회전가속도를추정하고대조군/네모서리위험도를비교한다. 결과는 `REPORT.md`·`moments.json`. 실제물체별접촉모멘트측정이아니다.

## PUSH 고정면·양손 중점 중앙선

- `tokenhsi/data/cfg/multi_agent/push_fixed130_fixed_face_midpoint.yaml`, 동명 `tokenhsi/scripts/multi_agent/`의 `_{train,test,vnc}.sh`: GPU6 고정1.30m/20kg 기반 별도 scratch 실행.
- `open_push_reward.py::push_face_geometry`의 `face_side`, `humanoid_ma_open_push.py::_reset_task`: 목표 생성 시 owner별 면 고정, 부분 reset 보존. `contact_at_reward`/`push_hand_contact`: 양손 중점 중앙선 거리와 각 손 면 거리 동시 적용.
- `tokenhsi/tests/test_push_fixed_face_midpoint.py`: 면 불변·상자 yaw 추종·손 벌림/중점 이동·높이 자유·부분 reset·기존 RSI bank 호환 CPU 검사.
- `tokenhsi/scripts/multi_agent/check_push_fixed130_rsi.py`: 새 고정면 모드에서 실제 초기 목표면·RSI 양손 중점 거리·부분 reset 면 보존을 추가 감사한다.
