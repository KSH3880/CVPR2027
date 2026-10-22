# 레포 구조

`ports/stage1_base_push/`는 GitHub `stage1_stage2_base`의 독립 스냅샷·원본Stage1 config와 추가 `stage1_base_push.yaml`(env/train)을 포함한다. `utils/stage1_push_spec.py`는 기존과제ID 보존·추가PUSH graph/크기/전용checkpoint 계약, `env/tasks/multi_agent/humanoid_ma_stage1_push.py`는4상자·원본reset/RSI 확장·PUSH 연결, `stage1_push_reward.py`는 원본AT+현재CONTACT 보상을 담당한다. 새과제 embedding1행과 별도PUSH AMP family1칸(4family/1330D)을 추가하고 원본TaskActionFusion/GTA 구조를 유지한다. `verify.sh`는CPU회귀, `check_scene.py`/`check_player.py`는CPU물리·평가/player·무학습fixture 복원 검사다. root의 `tokenhsi/scripts/multi_agent/stage1_base_push_{train,test,vnc}.sh`가 private cwd/Python경로를 고정한다. [명령](config.md#stage1-base-push), [상세](../ports/stage1_base_push/README.md).

`output_etc/check/stage1_base_push_rsi_review/`의 `render_review.py`·`launch.sh`는 실제4상자 CPU reset 및 동일bank949후보를 합친 검수영상 생성 진입점이며 `review.json`에 출처와 검수대기를 기록한다.

`push_native_phase_y2m.yaml`(env/train), 동명`_{train,test,vnc}.sh`는 기존native5000회에서 Y길이2m·X6종으로 이어가는 별도resume 실험이다. `push_native_phase_y2m_rsi_bank.json`은 기존949후보를 새6종/양담당자 초기PD로 재검사한다. `check_native_push_y2m.sh`·`check_push_native_y2m_scene.sh`가CPU물리/네트워크 연결 진입점이고 `transfer_push_native_y2m_checkpoint.py`가 모델/RMS/optimizer/epoch/frame을 보존하며 새계약으로 전이한다. `tests/test_push_native_y2m.py`가 장면·후보 보존/거부조건을 검사한다(모두tokenhsi 아래). 렌더/기하/샘플러 도구는 `--cfg-env`로 전용config를 받는다. [명령](config.md#push-native-y2m), [검수](../output_etc/check/push_native_phase_y2m/REPORT.md).

`utils/push_view_geometry.py`(tokenhsi 아래)는 `TOKENHSI_VIEWER_PUSH_LENGTH_Y`로 native GUI 평가에서만 상자 로컬Y를 늘린다. `humanoid_ma_open_push.py`가 평가 asset/크기 tensor·장면 sampler에 적용하고 source checkpoint/RSI 계약은 보존한다. `tests/test_push_view_geometry.py`는 학습/RSI/잘못된 모드 거부·원본불변·2m상자2048scene CPU clearance를 확인한다. 명령은 [가변 PUSH](config.md#push-native-phase-xy) 참조.

`push_native_phase_xy.yaml`(env/train), `dataset_push_native_phase_xy.yaml`, 동명 `_{train,test,vnc}.sh`는 기존 common-face soft-tilt의 원본 자세 RSI/XY 가변 상자 전용 scratch 실험이다. 현재목표는초기박스로컬+Y정면2~5m(`utils/push_scene.py`의`box_forward`)이며 MD후보에서clip7만뺀4클립/949후보의`push_native_phase_xy_forward_2to5m_no_clip7_rsi_bank.json`을 사용한다. `utils/native_push_rsi.py`는 원본 관절 보존·구간 선별·collider/발mesh 안전 정렬·계약/전체·후반 pool을 관리한다. `humanoid_ma_open_push.py`는36종 실제 asset·담당자별 크기와 native RSI를 연결하고 `open_push_reward.py`는 실제 크기 tensor로 접촉면/진단을 계산한다. `scripts/multi_agent/{build_native_push_phase,check_native_push_phase,audit_native_push_geometry,audit_native_push_sampler,render_native_push_rsi}.py`와 `check_native_push_phase.sh`, `tests/test_native_push_phase.py`가 선별/CPU물리/전수기하/샘플링/영상/회귀 진입점이다(모두 `tokenhsi/` 아래). `utils/rsi_preview_render.py`는 실제MJCF primitive/발OBJ의CPU triangle깊이 렌더·정측면을 제공하며 `tests/test_rsi_preview_render.py`가 가림/primitive범위를 검사한다. 현재검수영상은 `output_etc/check/push_native_phase_xy_no_clip7_review/rsi_candidates_no_clip7.mp4`이며 이전1,196영상에는clip7이포함되어있다. 렌더러의`--bank/--output/--candidates-only/--fps/--repeat`로 보존된선별본을 별도영상으로 만들 수 있다. 현재검수는 `output_etc/check/push_native_phase_xy_no_clip7_review/`이며 추가접지51후보·3~5m 이전검수는 각`output_etc/check/push_native_phase_xy_grounded/`·`output_etc/check/push_native_phase_xy/`에 보존, 명령은 [가변 PUSH](config.md#push-native-phase-xy) 참조.

현재 실행 실험은 원본 multi-agent carry(1), 독립 Stage 1(27·28·30·31·32·33·34·35·36·37 및 paired AT/placement·unified 독립 5과제), 협력 Stage 2(29 및 SIT plane 변형)다. 실행 명령과 checkpoint 규칙은 [config.md](config.md)에 있다.

로컬 시각화 명령과 VRAM·Wayland 설정은 [로컬 VNC 뷰어](config.md#local-vnc-viewer)를 먼저 참고한다.

## 진입점

`tokenhsi/data/stage1_push_motion_data_20261020/`는 사용자 제공 서버 모션 묶음을 별도로 해제한 폴더다. 내부 `tokenhsi/data/dataset_stage1_push.yaml`의 PUSH21/보행4개 및 원본 RSI cache/YAML을 보존한다. 로컬 검사·파생 후보·자세 표본은 `output_etc/check/push_motion_archive_20261020/{audit.py,report.py,REPORT.md}`에 있으며 기존 학습 설정에는 연결하지 않았다.

`learning/multi_agent/edge_context_eval.py`의 `TOKENHSI_PUSH_DISTANCE_EVAL=1`은 loco 시작 PUSH 평가에 초기 상자→목표 거리/접근거리·최초/종료 성공과 도달시간을 추가한다. `utils/push_distance_eval.py`가3~5m를0.5m 간격으로 집계하고 매 완료 repeat를 `metrics/distance_{completed_results,summary}.json`에 저장한다. `tests/test_push_distance_eval.py`는 거리 경계·비동기 scene 종료·중간 저장을 CPU 검사한다(모두 `tokenhsi/` 아래). 실행은 `output/push_distance_eval_20261019/launch.sh`, 조건은 [거리별 평가](config.md#push-distance-eval) 참조.

`push_contact_face_at_tilt_soft_3to5m.yaml`(env/train)과 동명`_{train,test,vnc}.sh`는 상자 초기 중심 기준3~5m 목표의 별도 전이 실험이다. `utils/push_scene.py`의 opt-in `sampling: box_radial`이 기존 owner-square 경로를 보존한다. `check_push_contact_face_at_3to5m_scene.sh`·`screen_push_contact_face_at_3to5m_rsi.sh`는 CUDA 비노출 CPU 연결/89후보 읽기 전용 검사이며 `check_push_random_scene.py`의 `TOKENHSI_CHECKPOINT`로 전이본 strict network load도 확인한다. `transfer_push_goal_distance_checkpoint.py`는 CPU audit와 원본 계약을 비교해 목표 배치만 바꾼 전이본을 만든다. `tests/test_push_goal_distance.py`가 거리/경로/RSI 기하/전이 경계를 검사한다(모두 `tokenhsi/` 아래). 실행/검증 한계는 [config.md](config.md#push-goal-3to5m) 참조.

`ports/ksh_object_finetune/`는 서버에서 복사한 native 의자/탁자 object tokenizer fine-tune 전용 스냅샷이다. `launch.py`는 해당 복사본을 import하고 로컬 파일의 실행 전후 무결성을 확인한다. `resume_local.sh {check|train}`은 `carry_chair_copied_adapter` 500epoch의 로컬2048환경 headless resume 진입점이며, 초기/재개 가중치는 `weights/{source_39000,resume_500}.pth`다. [명령·검증 범위](config.md#object-tokenizer-local-resume)를 따른다.

`push_contact_face_at_tilt_soft.yaml`(env/train), `push_contact_face_at_tilt_soft_{train,test,vnc}.sh`, `push_contact_face_at_tilt_soft_rsi_bank.json`은 공통 면 CONTACT를 유지하면서 AT5~15도 감쇠/성공10도 이하로 전이하는 별도 실행 경로다. `open_push_reward.py:contact_at_reward`가 포화 후 AT 감쇠를 적용하고 `open_push_spec.py`가 기울기 계약을 검사한다. `transfer_push_contact_face_tilt_checkpoint.py`는 동일 env의 보상 변경만 허용하는 CPU 전이 도구이며 `tests/test_push_contact_face_tilt_soft.py`가 경계/포화/복원/RSI 회귀를 검사한다(모두 `tokenhsi/` 아래).

`utils/push_face_audit.py`는 평가 전용 CPU 면/모서리·양손 XY 배치 계측이다. `learning/multi_agent/edge_context_eval.py`에서 `TOKENHSI_PUSH_FACE_AUDIT=1`일 때만 사용하며, `scripts/multi_agent/summarize_push_faces.py`가 결과 JSON 또는 완료 repeat 로그를 집계한다(모두 `tokenhsi/` 아래). `tests/test_push_face_audit.py`는 회전·모서리 모호성·종료 masking을 검사한다. 보상/학습 경로는 변경하지 않는다.

같은 opt-in 계측은 full quaternion/유한3D 면·손 반력의 면별 법선 성분을 사용하는 OBB/힘 proxy 분류도 기록한다. 첫 AT 성공 이전 이동 구간의 면별 기울기·10/15도 초과·거리 감소 및 episode 주사용 면별 성공률을 집계한다. `output/push_face_tilt_eval_20261017/launch.sh`가 공통 면 soft tilt10000epoch의1환경40회 평가 진입점이다. 힘은 control 경계의 net force proxy이며 충돌 쌍 측정이 아니다.

`push_soft_legacy40_at05.yaml`, `push_soft_legacy40_at05_{train,test,vnc}.sh`, `push_soft_legacy40_at05_rsi_bank.json`은 이전40D·AT0.5m 저장본 전용 복원 경로다. `humanoid_ma_open_push.py`의 `pushObjectObsSize` 기본30/호환40과 `open_push_spec.py:checkpoint_contracts_match`의 동일hash bank 경로 이동 검사를 사용한다. 기존 soft tilt train network YAML을 재사용한다.

`humanoid_ma_open_push.py:get_clean_object_obs_size/_clean_object_features`는 문 없는 `push_only`에서 박스 물리30D·전체570D를 사용한다. OPEN 포함 설정은40D이며, `test_push_only.py`에서30D 관측과570D Actor/Critic 연결을 CPU로 확인한다. 기존40D checkpoint는 새 관측 계약과 호환되지 않는다.

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/env/tasks/multi_agent/open_push_reward.py:push_contact_shaping`, `tokenhsi/tests/test_push_contact_align.py` | CONTACT 양손 면 XY 조화평균·사람 위치 정렬·근거리 접근 감쇠·기존 AT/완료 포화/담당 공유 회귀 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/push_toy_contact_align_at.yaml`, `tokenhsi/scripts/multi_agent/push_toy_contact_align_at_{train,test,vnc}.sh`, `check_push_contact_align_scene.sh`, `tokenhsi/data/push_toy_contact_align_at_rsi_bank.json` | CONTACT 세 변경 + AT 성공만10도·전용 scratch/bank/output·2환경 비학습 RSI/물리/네트워크 검사 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/push_toy_random_at_10deg_at_success_tilt.yaml`, `tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_success_tilt_{train,test,vnc}.sh`, `check_push_at_success_tilt_scene.sh`, `tokenhsi/tests/test_push_at_success_tilt.py` | AT 성공만10도 제한·state/progress 기존 거리식·전용 bank/output·GPU0 scratch 및 회귀 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/push_toy_random_at_10deg_at_tilt_soft.yaml`, `tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_tilt_soft_{train,test,vnc}.sh`, `check_push_at_tilt_soft_scene.sh`, `tokenhsi/tests/test_push_at_tilt_soft.py` | AT state/progress 기울기5~15도 선형 감쇠·성공10도 이하·PUSH 각도 gate 없음·전용 bank/output/회귀 및 RSI 연결 검사 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/push_toy_random_at_10deg_at_state.yaml`, `tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_state_{train,test,vnc}.sh`, `tokenhsi/tests/test_push_at_state_tilt.py` | PUSH 각도 제한 제거·AT state만10도 제한·높이/진행/성공 유지·별도 bank/output 및 회귀 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/push_toy_random_at_10deg_at_grounded.yaml`, `tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_at_grounded_{train,test,vnc}.sh` | PUSH·AT 모두10도/높이±8cm 보상 gate·별도 RSI bank/output/checkpoint |
| `tokenhsi/scripts/multi_agent/transfer_push_at_grounded_checkpoint.py`, `tokenhsi/tests/test_push_at_grounded.py` | AT grounded 한 조건만 변경하는 명시적 checkpoint 전이·optimizer 초기화·보상 gate/완료 해제 회귀 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/push_toy_random_at_10deg.yaml`, `tokenhsi/scripts/multi_agent/push_toy_random_at_10deg_{train,test,vnc}.sh` | PUSH만10도 제한·ksh식 사람별 XY±2m 랜덤 AT·목표 반대편 손 State·별도 bank/checkpoint/output |
| `tokenhsi/env/tasks/multi_agent/open_push_reward.py:push_goal_opposite_xy_target` | 현재 박스→목표 반대 ray와 yaw 기준 XY 직사각형 경계 교점·random AT 손 중점 State |
| `tokenhsi/env/tasks/multi_agent/open_push_reward.py:push_rsi_hand_targets` | 목표 반대편 RSI 양손 pair·사람 방향/속도 yaw 정렬·기존 실험은 로컬 뒤면 유지 |
| `tokenhsi/env/tasks/multi_agent/open_push_reward.py:push_contact_distance`, `push_forward_normal`, `push_hand_contact`, `push_debug_metrics` | 랜덤 AT 목표 반대편 면 XY 거리·현재 목표 방향 반력의 접촉 proxy/진단; 목표 반대편 reference는 CONTACT 조화평균 state>0.9 및 기존 힘 조건, 로컬 reference는 거리 기준 유지 |
| `tokenhsi/utils/push_scene.py:sample_random_at_scene`, `tokenhsi/tests/test_push_random_at.py`, `tokenhsi/scripts/multi_agent/check_push_random_at_scene.sh` | 사람/박스 최소1m 목표·두 이동/RSI 경로2.2m 여유·CPU/GPU reset/부분 reset/네트워크 연결 검사 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/{open,push}_toy.yaml`, `tokenhsi/scripts/multi_agent/{open,push}_toy_{train,test,vnc}.sh` | 로컬 과제별 100% 분리 학습·최신 후반 RSI·손 조작/몸 접촉 gate·실행별 독립 Actor/Critic/D·기본 최대9000회 및 독립 출력 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/push_toy_only_random_scene.yaml`, `tokenhsi/scripts/multi_agent/push_toy_only_random_scene_{train,test,vnc}.sh`, `tokenhsi/tests/test_push_only.py` | 문 없는2H/2Box/6토큰·PUSH_ONLY=1 선택·전용 계약/output·OPEN 미호출/PUSH 보상 동일성 검사 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/push_toy_only_random_scene_at.yaml`, `tokenhsi/scripts/multi_agent/push_toy_only_random_scene_at_{train,test,vnc}.sh` | 독립 PUSH/AT 각0.2/0.2/0.2·자기 합계/동료 평균·AT 도착 시 선행 PUSH 포화·전용 bank/checkpoint/output |
| `tokenhsi/env/tasks/multi_agent/open_push_reward.py:push_at_edge_reward`, `tokenhsi/tests/test_push_at.py`, `tokenhsi/scripts/multi_agent/check_push_at_scene.sh` | HOLDING+AT 방식의 손 중점 근접/목표 거리 보상·접촉 없는 AT·부분 reset/실제 CPU 물리/6토큰 네트워크 검사 |
| `tokenhsi/scripts/multi_agent/check_push_only_scene.sh`, `check_push_random_scene.py` | CPU/GPU 비학습 문 actor/DOF 부재·6토큰 Actor/Critic/AMP·RSI/reset/물리 검사 |
| `tokenhsi/utils/push_scene.py`, `tokenhsi/data/cfg/{multi_agent,train/rlg}/push_toy_midpoint_2m_random_scene.yaml`, `tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_random_scene_{train,test,vnc}.sh` | 고정1.3m·owner-relative1~2m/전 방향 yaw·경로 간섭 제외·참조 정렬·별도 배치 계약/최대100만epoch |
| `tokenhsi/scripts/multi_agent/check_push_random_scene.{py,sh}`, `tokenhsi/tests/test_push_random_scene.py` | CPU/GPU 비학습 reset/부분 reset/RSI/AMP/물리 검사·방향 회전/성공/후진 실적 회귀 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/push_toy_midpoint_2m.yaml`, `tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_{train,test,vnc}.sh` | PUSH 단독·중점 XY/높이 자유·2m 목표·후진 차감 순이동 실적·몸 접촉 허용·기존2모션·별도 checkpoint/output |
| `tokenhsi/scripts/multi_agent/build_push_midpoint_2m_rsi.sh`, `tokenhsi/data/push_toy_midpoint_2m_rsi_bank.json`, `tokenhsi/tests/test_push_midpoint_2m.py` | CPU/GPU 비학습 RSI 검증·전용2m bank·중점/접촉/후진/RSI 실적/성공 회귀 검사 |
| `tokenhsi/utils/manipulation_contact.py`, `tokenhsi/tests/test_manipulation_contact.py` | MJCF 신체 형상·물체 근접/수평 반력 기반 몸 접촉 proxy 및 손 조작/RSI/완료 후 손 떼기 회귀 |
| `tokenhsi/tests/test_open_push_toy.py` | 단독 reset/AMP 라벨·반대 과제 전문가 배제·복사된 RSI bank/모션 hash 계약 검증 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/open_toy_strong_close.yaml`, `tokenhsi/scripts/multi_agent/open_toy_strong_close_{train,test,vnc}.sh`, `tokenhsi/data/open_toy_strong_close_rsi_bank.json` | OPEN 토이의 초기 닫힘 preload·강한 개방 저항/자동 닫힘·별도 계약/출력 및 재검사 RSI |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/open_toy_reference_close.yaml`, `tokenhsi/scripts/multi_agent/open_toy_reference_close_{train,test,vnc}.sh`, `tokenhsi/data/open_toy_reference_close_rsi_bank.json` | 참조 PD 후보 비교로 자동 닫힘 저항1/0.5/6/preload1을 선택한 별도 OPEN 실험·기존 checkpoint와 계약 분리 |
| `tokenhsi/scripts/multi_agent/check_door_autoclose.py` | 실제 문 asset CPU PhysX에서 기존/강화 설정의 충격 개방·손 없는 놓기 비교, `output/open_toy_strong_close_physics_check/audit.json` |
| `tokenhsi/scripts/multi_agent/check_open_reference_push.{py,sh}` | CPU2환경·참조 관절 PD/자유 root·실제 손/비손 충돌 쌍·세 저항18조건 및 이상적 손잡이 힘9조건 비교; `output/open_reference_push_check/{REPORT.md,audit.json,reference_traces.csv,comparison.png}` |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_open_push.py`, `open_push_reward.py`, `tokenhsi/utils/open_push_spec.py` | 사람별 OPEN/PUSH50/50 graph·문 지지/바닥 박스 밀기·혼합 actor·task 조건 AMP·checkpoint 계약 |
| `tokenhsi/utils/open_push_rsi.py`, `tokenhsi/data/open_push_late_rsi_cube_low_bank.json` | task별 PD검사 후보·clip 균등 후반 선택·발/몸통 보존 어깨 회전으로 낮은 손 RSI·후보 hash 계약 |
| `tokenhsi/data/cfg/multi_agent/open_push_late_rsi.yaml`, `tokenhsi/scripts/multi_agent/open_push_late_rsi_{train,test,vnc}.sh` | 보행/조작50/50·후반 우선 별도 실험; 기존 공동 train/network YAML 재사용 |
| `tokenhsi/scripts/multi_agent/build_open_push_rsi.{py,sh}`, `tokenhsi/tests/test_open_push_rsi.py` | 비학습2환경12step PD 후보 검사·학습 reset/AMP/partial reset 감사 및 CPU 비율/정렬 검증 |
| `tokenhsi/scripts/multi_agent/check_open_push_physics.{py,sh}` | 비학습2환경의 상자 모양/힘 높이 비교·외력 대조군 및 원본 손 Z를 보존한 사람 PD·기울기/바닥 edge 이동 JSON |
| `tokenhsi/scripts/multi_agent/view_open_reference_motion.py`, `open_reference_motion_view{,_vnc}.sh` | checkpoint 없이 새 닫힘 저항에서 AMP 관절 PD 시간 재생·자유 root·1환경·화면 각도/손 총힘/낙상·닫힘/RSI 시작 비교·JSONL trace |
| `tokenhsi/scripts/multi_agent/view_open_push_rsi.py`, `open_push_rsi_view{,_vnc}.sh` | checkpoint 없는1환경 RSI 자세 검사·화면 모션정보·키보드 샘플/카메라/12step PD·JSON 샘플 기록 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/open_push_unified_amp.yaml`, `tokenhsi/data/dataset_open_push.yaml`, `tokenhsi/scripts/multi_agent/open_push_unified_amp_{train,test,vnc}.sh` | 문/박스 두 과제 공동 정책 설정·보행4/문1/밀기2 manifest·GPU6 전용 실행 |
| `tokenhsi/scripts/multi_agent/convert_bones_push.py`, `tokenhsi/data/dataset_bones_push/` | 선택한 BONES 밀기2개의30FPS/15body/32DOF 변환·수치 보고서·preview 데이터 |
| `tokenhsi/scripts/multi_agent/validate_open_push.{py,sh}`, `tokenhsi/tests/test_open_push.py` | 공동 sampler·접촉/grounded 성공·2-family AMP/replay·변환·GPU6 reset/네트워크/비학습 검증 |
| `tokenhsi/run.py` | 학습·평가 태스크와 알고리즘 등록 |
| `tokenhsi/utils/config.py:parse_sim_params` | YAML 물리 설정·평가16환경 이하 전용 `TOKENHSI_VIEWER_LOW_VRAM` PhysX 접촉 버퍼 축소 |
| `tokenhsi/env/tasks/base_task.py:render`, `tokenhsi/learning/multi_agent/ma_players.py:_amp_debug` | 선택적 viewer 렌더 간격/draw 대기/AMP debug 생략; 물리·정책 step 유지 |
| `tokenhsi/learning/multi_agent/edge_context_eval.py:run_edge_context_eval` | required-goal 평가와 선택적 PUSH 접촉/목표 고정/손 떼기 보상 감사; 살아 있는 첫 episode만 집계 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_door_hold.py`, `tokenhsi/utils/door_hold.py`, `tokenhsi/assets/door/door_hold.urdf` | 1인90도 개방·골반/torso/발 지지 자세 판정·연속 손 지지, reset별1.5~2.5m 거리 샘플링, 자동 닫힘 spring, 접촉 공백 처리·타이머 관측, 바닥 틈이 있는 전용 문 |
| `tokenhsi/data/cfg/multi_agent/door_open_hold_amp.yaml`, `tokenhsi/data/cfg/train/rlg/door_open_hold_amp.yaml`, `tokenhsi/scripts/multi_agent/door_open_hold_amp_{train,test,vnc}.sh` | 사람1·문1·BONES full AMP의 별도 scratch 학습/평가/viewer 설정 |
| `tokenhsi/scripts/multi_agent/validate_door_hold.{py,sh}`, `tokenhsi/tests/test_door_hold.py` | 비학습 손 접촉 개방·발 지지 부족의 거짓 성공 차단·release/자동 닫힘·AMP discriminator 및 CPU90도/자세/5초 유지 조건 검사 |
| `bones_seed/metadata/`, `bones_seed/soma_uniform.tar.gz`, `bones_seed/soma_uniform/bvh/`, `bones_seed/validation/REPORT.md` | BONES-SEED metadata·공식 SHA256 검증 archive·선별184개 원본 BVH·원본 수치 검증/로컬 skeleton 재생 |
| `tokenhsi/utils/bones_bvh.py`, `tokenhsi/scripts/multi_agent/convert_bones_door.py` | SOMA BVH 절대 Hips/XYZ·Euler FK parser와 선택한 왼손 clip의 humanoid constrained IK·crop·30FPS SkeletonMotion 변환 |
| `tokenhsi/data/dataset_bones_door/`, `tokenhsi/data/dataset_door_bones_left.yaml` | 변환된103frame AMP motion·매핑/수치 보고서·현재 문을 겹친 CPU HTML/GIF·loco/doorOpen manifest |
| `tokenhsi/scripts/multi_agent/build_bones_door_preview.py`, `validate_bones_door_amp.{py,sh}`, `tokenhsi/tests/test_bones_door_amp.py` | CPU 문 형상 비교 미리보기·GPU6 전용 비학습 loader/AMP/FK/replay 검증·좌표/관절/설정 회귀 검사 |
| `tokenhsi/data/cfg/multi_agent/door_open_bones_left_amp.yaml`, `tokenhsi/scripts/multi_agent/door_open_bones_left_amp_{train,test,vnc}.sh` | 기존 loco reset·OPEN reward와 full AMP를 유지하는 BONES 왼손 별도 실험; 정책 성공 미검증 |
| `ports/ksh_stage1/`, `tokenhsi/scripts/multi_agent/ksh_*` | ksh 세 방식의 독립 스냅샷·전용 train/test/VNC; 기존 학습 코드와 분리, [실행·격리](ksh_stage1_port.md) |
| `tokenhsi/scripts/multi_agent/` | 현재 실험별 train/test/VNC, `runtime_env.sh`, `run-gui.sh` |
| `tokenhsi/scripts/multi_agent/run-gui.sh` | 전용 Xvfb·x11vnc·noVNC 수명 관리 및 x11vnc의 상속 Wayland 환경 제거 |
| `tokenhsi/scripts/multi_agent/gui_gpu_env.sh`, `runtime_env.sh`, `tokenhsi/utils/config.py`, `tokenhsi/env/tasks/base_task.py` | VNC native 옵션의 Mesa 우회·뷰어 전용 MPS 분리·CUDA/Vulkan 인덱스 및 기본 CUDA 장치 일치 |
| `tokenhsi/data/cfg/multi_agent/` | 현재 실험 YAML과 평가 graph 예제 |
| `tokenhsi/data/cfg/train/rlg/` | PPO/AMP/Transformer 학습 설정 |
| `output/<실험>/<run>/` | checkpoint `nn/`, TensorBoard `summaries/`, 진단 `diagnostics/` |

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

### PUSH 구역 교환

- `tokenhsi/env/tasks/multi_agent/humanoid_ma_open_push.py`: `_agent_zone` 배정, 물리 슬롯→담당자 조회, graph용 물체 순서, RSI/힘/credit 연결, 네 구역 진단.
- `tokenhsi/data/cfg/{multi_agent,train/rlg}/push_toy_midpoint_2m_zone_swap.yaml`: 전용 구역 교환 설정/학습 이름.
- `tokenhsi/scripts/multi_agent/push_toy_midpoint_2m_zone_swap_{train,test,vnc}.sh`: 전용 실행/출력.
- `tokenhsi/scripts/multi_agent/check_open_push_zones.{py,sh}`: 양 배정·부분 reset·실제 CPU 물리 연결 검사. 실행법/학습 보류 상태는 config의 PUSH 구역 교환 절 참조.


`push_contact_face_at.yaml`(env/train), `push_contact_face_at_{train,test,vnc}.sh`, `push_contact_face_at_rsi_bank.json`은 고정1.3m·20kg의 공통 면 CONTACT+AT 전용 실험이다. `open_push_reward.py:push_face_geometry/push_rear_xy/contact_at_reward`가60도 허용 공통 면·최대 손거리 state·힘 접촉 포화를 계산한다. `push_rsi_hand_targets`는 면 접선 기준 RSI 두 점을 생성하며 `goal_contact_face_xy_v1`을 bank/checkpoint 계약에 기록한다. `test_push_contact_face_at.py`는 면 선택/회전·엄격0.9/힘 경계·RSI·접촉/AT 포화·기울기 무제한 및 runtime 연결을 검사한다. `screen_push_contact_face_rsi.{py,sh}`는 전용 bank만 CPU PD12step 재screen하며, `check_push_contact_face_at_scene.sh`는 기존 reset/물리/네트워크 검사기를 새 설정에 연결한다(위 파일 모두 `tokenhsi/` 아래).
