# Multi-Agent Carry 실행 가이드

현재 실행 가능한 실험은 원본 1번과 Stage 1 **27·28·30·31·32·33·34·35·36·37번 및 paired/unified 변형**, Stage 2 **29번 및 SIT plane 변형**이다. 과거 9~26번 config와 전용 실행 스크립트는 정리했다. 모든 명령은 저장소 루트에서 실행한다.

## 빠른 확인

- **[stage1_v2](#stage1-v2)**: 새 크기·성공 조건·RSI 검사 실험. Climb Z 0.2~0.7m, 받침/배경 XY 0.4~0.8·Z 0.25~0.5m, AT 20cm+높이 1mm, Climb 내부 80%. [TokenHSI 학습·평가 비교](tokenhsi_vs_stage1_v2_train_eval.md).
- **[stage1_v2_fixed](#stage1-v2-fixed)**: v2 전용 고정 선반·첫 reset 전 물리 1회 초기화. 기존 stage1/v2와 output 분리, 저장 env+sim 복원.

- **[Action → TokenHSI 원본 환경 평가](#action의-tokenhsi-원본-환경-평가)**: 원본 test mesh·1인 환경에서 frozen Action을 평가하는 별도 headless 진입점. ON_TOP은 별도 확장이다.

- **[stage1](#stage1)**: Action 기반 CPA·타입 간 공유 bias·AT/ON_TOP payload 선반. `stage1_train.sh`/`stage1_test.sh`/`stage1_vnc.sh`, 기본 2명·2048환경·4물체이며 기존 Action/CPA checkpoint와 혼용하지 않는다.

- 신규 **[Task/Role Gate + Entmax](#unified-size-rsi-task-role-gate-entmax)**는 `task_embedding` 기반 독립 실험이다. Additive bias/message 없이 task·role로 GTA K/V를 함께 조절하고 전체 scene에 Entmax 1.5를 적용한다. 기존 5종과 checkpoint를 섞지 않는다.

- 최근 비교 대상은 **Task message / 공유 Task MLP / Task MLP split / Task embedding / Task embedding action**이다. [다섯 실험 공통 설정과 차이](#task-5종-공통-설정과-비교), [최근 확인 기록](#task-3종-최근-확인-기록)을 먼저 확인한다.
- 실행 명령: [Task message](#unified-size-rsi-task-message), [공유 Task MLP](#unified-size-rsi-task-mlp), [Task MLP split](#unified-size-rsi-task-mlp-split), [Task embedding](#unified-size-rsi-task-embedding), [Task embedding action](#unified-size-rsi-task-embedding-action). 다섯 실험 모두 **2명·2048환경·4물체**, 전용 train/test/VNC와 output을 사용한다. 서로의 checkpoint를 resume/eval에 섞지 않는다.
- `split`은 **task별 concat 이후 MLP와 layer/head projection 분리**다. 역할 embedding은 공유하며, actor/critic은 독립이다. 두 MLP 실험 모두 관계 message는 없다.
- `task_embedding`은 역할 입력·중간 MLP·message 없이 task/NONE/SELF의 6개 embedding과 H/O/G 타입 쌍별 projection을 사용한다. Actor/critic은 독립이며 각 branch 안에서 모든 카테고리가 projection을 공유한다.
- `task_embedding_action`은 task embedding 기반으로 SELF/NONE bias를 0으로 고정하고, GTA 복귀 후 사람 토큰에 `z + MLP([z, e_task])`를 적용한다. MLP는 128→64→64 ReLU·4개 layer 공유, actor/critic 독립이며 edge message는 없다.
- 학습 중 HOLDING/AT/ON_TOP 성공률은 보상 primitive 지표다. 정책의 네 task 및 별도 평가의 운반 완수율과 구분한다. Bias는 원래 값과 같은 물리 타입의 NONE 대비 차이를 구분한다.
- 현재 GPU/PID·epoch는 문서의 과거 기록으로 판단하지 않는다. 실행 전 `nvidia-smi`, 실행 중인 `tokenhsi/run.py` 인자와 해당 run의 `relation_config.yaml`·`summaries/`·`nn/`을 확인한다.

## Action의 TokenHSI 원본 환경 평가

`tokenhsi/evaluate_action_tokenhsi.py`는 학습하지 않는 headless 평가 전용 진입점이다. 기존 Action actor/normalizer를 strict-load하고 원본 `HumanoidTrajSitCarryClimb`의 1인 환경·test mesh·초기화·시간 제한·성공 누적 판정을 사용한다. 관측만 Action의 사람/물체 bbox/task packet으로 변환한다. 기존 multi-agent test와 분리하며, 원본 세팅은 [상세 문서](tokenhsi_original_carry_sit_climb.md)를 참조한다.

```bash
TOKENHSI_GPU=0 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
python tokenhsi/evaluate_action_tokenhsi.py \
  --checkpoint output/approach_scenario_stage1_unified_size_rsi_task_embedding_action/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingAction_08-15-04-39/nn/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingAction_00055000.pth \
  --task carry --num-envs 64 --repeats 3 --seed 0 \
  --static-mode relocatable --output output_etc/action_tokenhsi_eval/carry'
```

출발 물체를 가까이 두는 대조 평가는 위 명령에 **`--approach-max-distance 2`**를 추가하고 `--output output_etc/action_tokenhsi_near2m_eval/carry`처럼 결과를 분리한다. 사람–담당 물체의 초기 XY 반경만 **U(1,2m)**로 제한한다. 원본 하한 1m를 유지하며 AT 도착 목표·ON_TOP 받침의 위치/높이 샘플링, 실제 mesh, 성공 기준과 시간 제한은 유지한다. 옵션을 생략하면 원본 Sit 1~5m·나머지 1~10m다.

`--task`는 `carry`, `sit`, `climb`, `ontop` 중 선택한다. **Sit/Climb은 `--static-mode fixed_scene`, Carry/ON_TOP은 `--static-mode relocatable`**을 사용한다. `ontop`은 **원본에 없는 확장**으로, Carry의 도착 선반을 현재 ON_TOP 크기 규칙의 동적 받침상자로 대체한다. 받침의 현재 상판 중심에 payload가 놓였을 때의 중심점을 목표로 삼고 3D 20cm 도달을 누적한다. 출발 선반은 원본 Carry 규칙이며 안정적 적층/유지 지표는 아니다.

`--static-mode original`은 원본 fixed actor다. 이 서버에서는 GPU reset 후 선반의 collision 위치가 따라오지 않아 낙하 접촉 검사를 통과하지 못했다. `relocatable`은 재배치 가능한 중력 비활성·고질량 선반으로 대체하는 Carry 호환 모드다. Sit/Climb의 `fixed_scene`은 **원본 fixed mesh**를 유지하고 장면 전체의 공통 XY 이동·yaw 회전으로 원본에서 샘플링한 상대 배치를 보존한다. 모델 관측은 다시 원래 좌표로 복원하므로 neutral goal의 위치·방향도 달라지지 않는다. 이 차이를 manifest에 기록하며 논문의 완전 동일 재현으로 부르지 않는다. `--probe-only`로 출발 선반에 상자를 떨어뜨려 충돌 위치를 확인할 수 있다.

원본의 512환경 대신 간단 평가 기본값은 64환경×3회다. 로컬 Climb test는 논문의 26개와 달리 35개다. Action은 학습 당시 2명·4물체 대신 1명·과제 물체만 보며, 의자의 좌판 목표점은 모델 입력에 추가하지 않는다. `manifest.json`, `results.json`, `summary.json`, 회차별 초기 상태·거리 trace를 저장한다. 기존 Action/stage1 학습 설정과 성공 조건을 바꾸지 않는다.

2026-10-14 Action 55k 간단 평가(각 192시도)의 Carry/Sit/Climb/ON_TOP ever는 **19.8/49.5/54.2/17.7%**, 종료 시는 **17.7/44.3/9.9/15.6%**다. 실행 조건·물리 검사·초기 디버깅 결과 제외 기준은 [평가 기록](../output_etc/action_tokenhsi_20261014/README.md), 독립 재집계는 [validation.json](../output_etc/action_tokenhsi_20261014/validation.json)에 있다.

원본 환경의 **1~2m 제한 평가**(Action 55k·각 192시도)는 AT/ON_TOP/SIT/CLIMB ever **52.1/48.4/50.5/92.7%**, 종료 시 **45.8/46.4/44.3/7.8%**다. [평가 기록](../output_etc/action_tokenhsi_near2m_20261014/README.md)과 [검증 결과](../output_etc/action_tokenhsi_near2m_20261014/validation.json)에 저장했다.

## 공통 규칙

- `output/`에는 본학습 폴더와 `rsi_cache`를 유지한다. 완료된 검증·평가·분석 14폴더는 2026-10-10에 `output_etc/`로 이동했다. 이동 목록·원래 경로는 `output_etc/archive_manifest_20261010.json`에서 확인한다. 아래 실행 명령은 새 실행의 `output/` 경로를 사용한다.

- 학습 인자는 `[num_agents] [num_envs] [num_objects]`; Stage 1 기본값은 `2 2048 3`이다. 짧은 확인도 환경 수 2048을 유지하고 `MAX_ITERATIONS`만 줄인다.
- 실행 전 `nvidia-smi`로 점유를 확인하고 명령 앞에 `TOKENHSI_GPU`를 지정한다. 실행 중인 프로세스는 임의 종료하지 않는다.
- 본학습 전 셸의 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 확인한다.
- 학습은 기본적으로 scratch다. 기존 실험의 reward/checkpoint 계약을 섞지 않는다. Stage 2 전이는 아래의 `STAGE1_CHECKPOINT`를 사용한다.
- 평가·VNC 인자는 `<checkpoint.pth> [agents] [envs] [objects] [repeats]`다. `HEADLESS=0`은 로컬 viewer, `HEADLESS=1`은 화면 없는 평가다.
- 로컬 viewer와 서버 VNC의 상자 색은 매 reset의 과제 배정을 따른다. 단독 대상은 해당 에이전트 색, 공동 대상은 노란색, 나머지는 회색이다. 상자 정리 데모는 모든 상자를 같은 회색으로 표시한다. AT goal 점도 해당 edge 담당 에이전트 색이며, goal 슬롯이 바뀌어도 배정을 따른다. 실행 중인 뷰어는 다시 시작해야 반영된다.
- 공유 데이터 원본은 읽기 전용으로 취급하고 이 저장소에서는 심링크로 사용한다. 현재 로컬 링크는 `/home/cvlab/Desktop/CVPR2027/TokenHSI`로 연결된다(2026-10-07, 12개 링크 모두 유효). 서버 문서의 `/home/hwanhee/CVPR2027/TokenHSI`와 경로가 다르므로 다른 머신에서는 실제 링크 대상을 확인한다. RSI 캐시는 이 저장소의 `output/rsi_cache`다.
- 4명·16상자 정리 데모는 아래 **상자 정리 VNC 데모** 절을 따른다. 지정된 rescue checkpoint를 사용하는 평가 전용 실행이다.

## Carry 관절 영상·공동운반 RSI 후보

관련 코드·산출물은 저장소 루트 `joint_carry/`에 모은다. **보정본 RSI 연결은 [인계 문서](../joint_carry/HANDOFF.md)를 먼저 읽는다.** 최종 파일은 [joint_carry 안내](../joint_carry/README.md), 전용 코드는 `joint_carry/scripts/`, 보정 전 자료는 `joint_carry/archive/`다. 전달 ZIP에는 최신 보정본·코드·물리 검사·인계 문서를 포함한다.

```bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
python joint_carry/scripts/check_handoff.py  # 학습/물리 실행 없이 무결성과 snapshot 계약 확인
```

학습 없이 CPU에서 원본 `pickUp` B19·`carryWith` B20·`putDown` B21의 관절값을 재생하고 두 사람 합성 영상을 만든다. 실행 가능한 **시각화·후보 export**이며, 학습용 공동운반 reset에는 연결하지 않았다.

```bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
python joint_carry/scripts/carry_joint_preview.py --output joint_carry/archive/carry_joint_preview
python joint_carry/scripts/carry_joint_preview.py --skill all --grip-inset 0.15 --output joint_carry/archive/carry_joint_preview_all_inset15
```

`joint_carry/archive/carry_joint_preview/`의 `comparison.mp4`는 원본/공동운반을 나란히 보이는 30fps·129프레임(4.3초) 영상이다. `original.mp4`, `cooperative.mp4`와 `comparison_half_speed.mp4`도 저장한다. 원근·평면 시점은 동일한 관절 데이터에서 렌더링한다.

`--skill all`은 `pickUp/`, `carryWith/`, `putDown/`에 각각 같은 파일 세트를 저장하고, 루트에 `comparison_all_skills.mp4` 등 모음 영상과 `all_skills.json`을 만든다. 세 클립은 각각 100/129/135프레임이며 모음은 364프레임(약 12.13초)이다. **서로 다른 원본 클립을 순서대로 보여주는 영상이며 연속 궤적으로 연결한 데이터가 아니다.** 단일 동작은 `--skill pickUp`, `--skill carryWith`(기본값), `--skill putDown`으로 선택한다. `joint_candidates_all_skills.zip`에는 세 동작의 원본/합성 NPZ·메타데이터와 root/관절/상자 배열 사용법 `README.txt`를 묶는다.

원본 0.4m 상자를 공동운반에서는 0.52×0.80×0.40m로 바꾼다. 첫 사람을 상자 로컬 Y의 `-(0.4 - grip_inset)`m만큼 이동하고 두 번째 사람은 매 프레임 상자 중심 주위로 180° 회전한다. `--grip-inset 0.15`는 끝부분 배치 대비 각 사람의 몸 전체를 15cm 안쪽으로 이동한다. 몸 내부 관절각과 뼈 길이는 그대로이며 IK는 없다. `original_joints.npz`, `cooperative_rsi_candidate.npz`에 로컬 quaternion(xyzw), 32-DOF 각도·속도, root/body 위치·회전·속도, 공유 상자 상태·크기를 저장한다. 사람 배열은 `[frame, agent, joint/body, component]`이며 root/DOF는 해당 축을 생략한다. 속도는 합성 궤적에서 다시 계산한다. 영상과 NPZ는 동일한 상태를 사용하며 장면 재현에는 root·상자 상태도 함께 필요하다.

영상의 full quaternion FK와 32-DOF 물리 캐릭터의 FK는 정확히 같지 않다. 물리 팔꿈치는 Y축 1-DOF라 원본 X/Z 회전이 제외되고, 이번 세 클립에서는 손 위치가 최대 약 8.45cm 달라졌다. NPZ의 root/DOF/상자 상태를 넣으면 이 **32-DOF 근사 자세**가 재현된다. full quaternion/body 배열은 영상 자세를 보존한다.

NPZ의 `source_rsi_allowed`는 원본 YAML 제외 구간을 기존 `MotionLib` 시간 변환식으로 반영한 마스크다. 합성 데이터의 물리 유효성을 뜻하지 않으며 `physics_valid_frame_mask`는 생성 시 미검증이므로 모두 false다. 원본 상자의 높이도 그대로 보존하므로 바닥과의 간격이 있을 수 있다.

`metadata.json`에 합성 방법·손 관절점의 상자 표면 거리·낮은 발의 수평 속도를 기록한다. 이는 실제 접촉 판정이 아니다. 반대편 복제 사람은 발 미끄러짐이 크므로 `rsi_ready=false`, `physics_validated=false`다. RSI 채택 전 발 접촉 보정·충돌·물리 안정성 검증과 두 사람 동시 reset 경로가 필요하다. 원본 모션과 기존 학습 설정은 수정하지 않는다.

### 합성 carry 물리 검사

```bash
TOKENHSI_GPU=0 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
python joint_carry/scripts/validate_carry_physics.py --input joint_carry/archive/carry_joint_preview_all_inset15 --output joint_carry/archive/carry_physics_validation'
```

Isaac Gym **CPU PhysX·1환경씩**으로 원본/합성 세 동작을 검사한다. 학습은 하지 않는다. 기존 asset·PD gain·중력·마찰·밀도 100kg/m³·1/60초·substeps 2를 사용한다. 모든 프레임에 기존 RSI의 0.1초 고정 PD target 충격 기준을 적용하고, 실제 시뮬레이터 FK에서 구/캡슐과 상자의 관통·발 mesh의 바닥 침범을 계산한다. 초기 상자 속도는 NPZ를 사용하므로 실제 학습 reset의 0 속도 설정과 다르다. 2cm 관통 제한은 이번 보수적 검사 기준이며 물리 엔진의 절대 허용 오차를 뜻하지 않는다. 캡슐은 중심선 5mm 이하 간격 표본, 사람 간 겹침은 더 성긴 표본의 하한이며 mesh-상자·self collision은 실제 PhysX에서 처리한다.

`report.json`, 동작/원본·합성별 `frames.csv`, `screen_masks.npz`, `simulator_reset_geometry.npz`, `pd_rollout.npz`를 저장하며 입력 후보·학습 캐시를 수정하지 않는다. 연속 PD 검사는 root를 고정하거나 강제 이동하지 않는다. 학습된 균형 제어기가 없으므로 장기 PD 실패가 학습 불가능을 증명하지는 않는다.

2026-10-07의 15cm 안쪽 합성본은 원본 RSI 허용 프레임 중 **pickUp 29/55·carryWith 0/129·putDown 0/135**가 충격+관통 검사를 함께 통과했다. pickUp의 통과 구간은 집기 전 0~28프레임이다. 충격만 검사하면 각각 55/55·129/129·128/135여서 그것만으로 접촉 타당성을 보장할 수 없다. 운반 중 손/팔의 최대 관통은 약 11.5/10.1cm다. 이 결과로 학습용 RSI 승인을 부여하지 않았다.

### 공동운반 보정·검사 후 RSI snapshot 추출

```bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
python joint_carry/scripts/correct_carry_candidates.py --no-render
TOKENHSI_GPU=0 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
python joint_carry/scripts/validate_carry_physics.py --input joint_carry/carry_joint_corrected --output joint_carry/carry_corrected_physics'
python joint_carry/scripts/correct_carry_candidates.py --finalize
```

기본 입력은 `joint_carry/archive/carry_joint_preview_all_inset15`, 보정 출력은 `joint_carry/carry_joint_corrected`다. 실제 32-DOF 팔꿈치·고정 손목에 맞춘 뒤 상자 옆면 접촉과 팔 capsule의 관통을 함께 최소화하는 IK를 적용한다. 상자 크기와 15cm 안쪽 잡기 위치, 하체·몸통·머리 관절각은 유지하고, root 높이만 클립별 상수 1.4~2.6cm를 올린다. **팔 각도는 변경된다**(해당 입력의 최대 DOF 변화 pickUp 42.0°·carryWith 29.4°·putDown 36.6°). 속도는 수정 궤적에서 다시 계산한다.

`--finalize`는 보고서의 입력 SHA256을 확인한 후 원본 RSI 제외·0.1초 충격·관통 검사를 통과한 프레임만 추출한다. carryWith에는 두 사람 모두 6 step 중 3 step 이상 손 접촉이 생겨야 한다는 조건을 추가한다. `screened_rsi_snapshots.npz`의 `root_state`는 `[sample,2,13]`, `dof_position/dof_velocity`는 `[sample,2,32]`, `box_state`는 `[sample,1,13]`이고 각 13-D state는 위치·xyzw quaternion·선속도·각속도다. `source_frame/source_time`으로 보정 reference의 원래 프레임을 찾는다.

2026-10-07 보정본은 **55/55·129/129·135/135, 총 319개** 원본 RSI 허용 프레임이 짧은 검사를 통과했다. carryWith 손/팔 최대 관통은 11.51/10.10cm에서 약 0.50/0cm로 감소했고 실제 PhysX FK와 렌더 FK의 최대 위치 차이는 6.44e-7m 미만이다. pickUp 전체 클립 중 원본 제외 구간은 계속 제외한다. `before_after/comparison_all_skills.mp4`에 실제 물리 관절 기준 보정 전후 영상, `corrected_rsi_bundle.zip`에 보정 reference·snapshot·메타데이터·물리 결과를 묶는다.

**이는 단기 reset용 snapshot 후보이며 연속 전문가 모션이나 장기 균형을 검증한 데이터가 아니다.** 복제 사람의 연속 발 미끄러짐은 남아 있고 native PD의 장기 운반도 실패한다. `training_loader_integrated=false`, `continuous_motion_validated=false`로 저장하며 현재 독립 과제 reset에는 연결하지 않았다. 영상·snapshot에서 모든 관절을 원본 그대로 유지했다는 의미도 아니다. 기존 학습·원본 데이터는 수정하지 않는다.

## 현재 실험

| 번호 | Config | 역할 |
| ---: | --- | --- |
| Stage1 | [stage1.yaml](../tokenhsi/data/cfg/multi_agent/stage1.yaml) | Action MLP 유지, 타입 간 공유 task bias·사람 CPA·carry payload/AT 목적지 선반 |
| Stage1 v2 | [stage1_v2.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2.yaml) | Stage1 + Climb 0.2~0.7m·받침/배경 새 범위·AT 20cm/Climb 내부80%·별도 RSI 캐시 |
| Stage1 v2 fixed | [stage1_v2_fixed.yaml](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml) | v2 크기·보상 유지, 고정 선반·일회 초기화·env/sim 복원 |
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
| Unified owner HOLDING | [approach_scenario_stage1_unified_owner_holding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_owner_holding.yaml) | Unified semantic + AT/ON_TOP 담당 HOLDING φ 입력 |
| Unified size RSI | [approach_scenario_stage1_unified_size_rsi.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi.yaml) | 행동별 실제 asset 크기·bbox progress·물리 검사한 후반 RSI, 과제 비율 5/10/25/30/30 |
| Unified size RSI typed bias | [approach_scenario_stage1_unified_size_rsi_typed_bias.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias.yaml) | Size RSI 유지, 타입·관계·타입별 독립 layer/head bias 표 |
| Unified size RSI typed message | [approach_scenario_stage1_unified_size_rsi_typed_bias_message.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message.yaml) | Typed bias 유지, 64-D 관계 메시지 추가 |
| Unified size RSI task message | [approach_scenario_stage1_unified_size_rsi_task_message.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message.yaml) | 복합 carry 과제·역할별 bias/message, 단방향 3연결·자기 보상만 |
| Unified size RSI task MLP | [approach_scenario_stage1_unified_size_rsi_task_mlp.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp.yaml) | Task message와 동일한 과제·RSI·AMP·자기 보상, task·역할 MLP bias만 사용 |
| Unified size RSI task MLP split | [approach_scenario_stage1_unified_size_rsi_task_mlp_split.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split.yaml) | Task MLP의 역할 embedding 공유, task별 MLP·projection 독립 |
| Unified size RSI task embedding | [approach_scenario_stage1_unified_size_rsi_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding.yaml) | Task/NONE/SELF embedding과 H/O/G 타입 쌍별 projection, 중간 MLP·메시지 없음 |
| Unified size RSI task embedding action | [approach_scenario_stage1_unified_size_rsi_task_embedding_action.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action.yaml) | SELF/NONE=0, task embedding bias + 사람 토큰의 concat/비선형 residual MLP |
| Unified size RSI task embedding action CPA | [approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa.yaml) | Action 구조 유지, 사람 간 현재 거리 collision을 예측 최근접 거리 CPA로 교체 |
| Unified size RSI task role gate entmax | [approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax.yaml) | Task64 + role3 → layer별 GTA block K/V gate, 전체 scene Entmax 1.5; bias/message 없음 |

27~37번 Stage 1은 2-agent/3-object 독립 graph와 schema 9 semantic 관찰을 쓴다. ON_TOP 두 개를 동시에 샘플하지 않는다. 28~37번의 팀 보상은 자기 0.9, 동료 0.1이다. 30~33번은 유효 edge를 agent별로 평균 내어 단일 edge와 두 edge의 최대 task 보상을 0.6으로 맞춘다. 34~37번은 자기 edge를 합산하고 동료 edge만 평균 내며 state·progress·성공 식은 33번과 같다.

Paired AT/paired placement도 schema 9 semantic 관찰과 34번의 reward·RSI·AMP를 유지한다. 두 agent의 **과제 relation만 같은 타입**으로 샘플하고 실제 motor action은 각자 선택한다. Paired placement는 두 ON_TOP에 서로 다른 받침을 주기 위해 4개 물체를 쓴다. 두 기준 실험은 가까운 시작 확률이 처음부터 끝까지 0이며, 각자 별도 checkpoint/output에서 scratch로 시작한다.

Paired owner HOLDING은 paired placement의 샘플링·reward·RSI·AMP·가까운 시작 없음 설정을 유지한다. 관측만 edge당 5개에서 6개 필드로 늘리고 AT/ON_TOP에는 같은 owner의 `H→source O` HOLDING `φ`, 나머지 유효 edge에는 1을 기록한다. actor·critic의 기존 semantic 64차원 뒤에서 AT/ON_TOP에만 상태 잔차를 더한다. 기존 paired checkpoint와 관측·네트워크 형식이 달라 별도 output에서 scratch로 학습한다.

2026-09-30 공통 코드 수정 이후 scenario의 상자 속도 패널티는 현재 graph에 연결된 담당 물체를 따른다. 이전에는 legacy agent assignment를 사용해 랜덤 binding에서 다른 물체의 패널티가 들어갈 수 있었다. 모델/관측 shape는 같아서 기존 checkpoint를 로드할 수 있지만, 수정 전후 학습의 penalty 의미는 다르다. 이미 실행 중인 프로세스에는 자동 적용되지 않으며 이후 새 프로세스부터 적용된다.

33번은 32번의 보상·샘플링·RSI를 유지한다. AT graph가 두 번째 목표 slot을 가리킬 때 reset도 해당 목표를 배치하도록 공통 코드의 slot 선택을 수정했다. 수정 후 새로 시작하는 27~32번 실행에도 적용되지만, 이미 실행 중인 프로세스와 checkpoint는 바뀌지 않는다. 비교 실험은 33번 전용 output에 scratch로 시작한다.

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
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_train.sh 2 2048 4  # Unified owner HOLDING
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_train.sh 2 2048 4  # Unified size RSI
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_train.sh 2 2048 4  # Unified size RSI typed bias
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_train.sh 2 2048 4  # Unified size RSI typed message
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_train.sh 2 2048 4  # Unified size RSI task message
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_train.sh 2 2048 4  # Unified size RSI task MLP
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_train.sh 2 2048 4  # Unified size RSI task MLP split
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_train.sh 2 2048 4  # Unified size RSI task embedding
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_train.sh 2 2048 4  # Unified size RSI task embedding action
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax_train.sh 2 2048 4  # Unified size RSI task role gate entmax
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
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_test.sh "$CKPT" 2 1 4 10  # Unified owner HOLDING
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_test.sh "$CKPT" 2 1 4 10  # Unified size RSI
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_test.sh "$CKPT" 2 1 4 10  # Unified size RSI typed bias
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_test.sh "$CKPT" 2 1 4 10  # Unified size RSI typed message
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_test.sh "$CKPT" 2 1 4 10  # Unified size RSI task message
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_test.sh "$CKPT" 2 1 4 10  # Unified size RSI task MLP
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_test.sh "$CKPT" 2 1 4 10  # Unified size RSI task MLP split
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_test.sh "$CKPT" 2 1 4 10  # Unified size RSI task embedding
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_test.sh "$CKPT" 2 1 4 10  # Unified size RSI task embedding action
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax_test.sh "$CKPT" 2 1 4 10  # Unified size RSI task role gate entmax
TOKENHSI_GPU=3 HEADLESS=0 bash tokenhsi/scripts/multi_agent/box_cleanup_demo_test.sh  # 4명·16상자 정리 데모
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
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_vnc.sh "$CKPT"  # Unified owner HOLDING
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_vnc.sh "$CKPT"  # Unified size RSI
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_vnc.sh "$CKPT"  # Unified size RSI typed bias
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_vnc.sh "$CKPT"  # Unified size RSI typed message
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_vnc.sh "$CKPT"  # Unified size RSI task message
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_vnc.sh "$CKPT"  # Unified size RSI task MLP
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_vnc.sh "$CKPT"  # Unified size RSI task MLP split
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_vnc.sh "$CKPT"  # Unified size RSI task embedding
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_vnc.sh "$CKPT"  # Unified size RSI task embedding action
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax_vnc.sh "$CKPT"  # Unified size RSI task role gate entmax
TOKENHSI_GPU=3 PORT=6080 bash tokenhsi/scripts/multi_agent/box_cleanup_demo_vnc.sh  # 4명·16상자 정리 데모
```

## 상자 정리 VNC 데모

별도 학습/config 없이 34번 YAML을 바탕으로 `box_cleanup_demo.py`에서 평가 전용 task/player를 등록한다. 기본 checkpoint는 `distill_pth/ApproachScenarioStage1RescueAtKLClimb50_00008000.pth`다. 사람 4명·물리 상자 16개·8개 semantic edge로 정책을 그대로 로드하며, H223/O30 정규화 통계도 재사용한다. 데모 로더만 옛 rescue variant와 추가 AT/CLIMB shaping 설정을 현재 self-sum runtime에 대응시킨다. schema·packet·나머지 reward 계약은 검사하고, 원본 checkpoint와 일반 학습/평가 로더는 변경하지 않는다. 데모의 결과는 운반 진행 기록이며 rescue 보상 비교 평가는 아니다.

- 중앙에 상자 16개를 불규칙한 2단 더미로 모아 쌓는다. 가로·세로 0.40/0.45/0.50m, 높이 0.30/0.35/0.40m를 독립적으로 섞으며 밀도는 100kg/m³다. 기본 중심 간격 70cm에서 두 줄의 가로 위치·각 상자의 위치를 흐트러뜨리고 수평 회전을 ±15° 섞는다. 받침 위 대상도 조금 어긋나게 놓으며 시작 시 물체끼리 겹치지 않는다. 크기는 실행 시작 시 고정되고 위치·회전은 매 반복 다시 샘플한다. 배정 여부와 관계없이 같은 회색이다.
- 동서남북 사람 4명이 자기 방향의 윗단 상자를 하나씩 옮기고, 모두 끝나면 남은 윗단 상자 네 개를 배정한다. 운반 대상 8개 모두 처음에는 받침 상자 위에 있으며 실제 높이에 맞춰 초기 중심·바닥 목적지 높이를 계산한다. 해당 checkpoint의 저장된 학습 설정은 가로·세로 0.40~0.65m, 높이 0.25~0.55m다. 데모 크기는 이 범위 안이며 크기별 성공률을 보장하지 않는다.
- 목적지 XY 오차 12cm 미만·회전 bbox 바닥 오차 2.5cm 미만·선속도 0.15m/s 미만·각속도 0.5rad/s 미만을 15 control step(0.5초) 유지하면 내려놓기로 기록한다. 손 접촉 해제는 요구하지 않는다. 먼저 끝난 사람은 기존 배정을 유지하며 정책이 계속 자세를 제어한다.
- 네 상자가 동시에 안정적으로 놓이면 graph와 목적지만 바꾼다. 사람·상자를 teleport/reset하지 않고 두 번째 운반을 시작한다. 두 라운드 후 종료하며 viewer는 마지막 화면을 유지한다. 중앙 상자도 실제 물리 물체여서 접촉하면 움직이거나 더미가 무너질 수 있다.
- 이전 동일 크기·정렬된 중앙 더미 버전의 seed 42 검사에서는 7개 운반 후 마지막 윗단 집기가 정체되어 시간 초과했다. 현재 불규칙·크기 혼합 더미의 8개 완주는 아직 확인하지 못했다. 사용자 선택에 따라 4명·16물체 전체 관측을 그대로 입력하며 주변 관측 추론은 적용하지 않는다.

```bash
# 서버 VNC: 기본 10회·한 회 최대 1500 step(50초), 6080 포워딩 후 noVNC 접속
TOKENHSI_GPU=3 PORT=6080 bash tokenhsi/scripts/multi_agent/box_cleanup_demo_vnc.sh
# 로컬 viewer / 화면 없는 검사
TOKENHSI_GPU=3 HEADLESS=0 bash tokenhsi/scripts/multi_agent/box_cleanup_demo_test.sh
TOKENHSI_GPU=3 HEADLESS=1 REPEATS=1 OUTPUT_PATH=output/box_cleanup_demo_check bash tokenhsi/scripts/multi_agent/box_cleanup_demo_test.sh
```

checkpoint를 바꾸려면 첫 인자로 경로를 넘긴다. `REPEATS` 기본값은 10이며 각 반복마다 두 라운드 장면을 처음부터 재생한다. `KEEP_OPEN=0`은 viewer가 마지막 반복 종료 시 자동 닫히게 한다. `EPISODE_LENGTH` 기본값은 1500 control step(시뮬레이션 시간 50초), `SEED`는 42다. 실패·시간 초과도 `OUTPUT_PATH/cleanup_results.json`에 실제 상자 크기와 함께 기록하고 다음 라운드를 강제 성공시키지 않는다. Viewer는 `frame_0/120/300/600.png`와 `final.png`를 저장한다.

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

## Unified size RSI

`approach_scenario_stage1_unified_size_rsi`는 unified semantic에서 분리한 scratch 실험이다. 기존 unified checkpoint는 reward 계약이 달라 재개할 수 없다. 2-agent/4-object canonical 배정, semantic edge 관찰, 토큰·edge 순열, state·성공 조건·보상 가중치, 가까운 시작 없음은 유지한다.

| 과제/물체 | X·Y 각각 (m) | Z (m) |
| --- | --- | --- |
| HOLDING, HOLDING_AT, HOLDING_ON_TOP source | 0.20–0.60 | 0.20–0.60 |
| SIT source | 0.25–0.80 | 0.35–0.50 |
| CLIMB source | 0.40–0.80 | 0.25–0.50 |
| ON_TOP support | 0.50–0.80 | 0.25–0.45 |

- X/Y/Z는 각각 독립적인 5cm 격자 균등분포이고 밀도는 100kg/m³다. 큰 carry 상자는 최대 21.6kg이며, 범위 안의 모든 크기에서 행동 성공이 보장되는 것은 아니다.
- 물리 asset 크기는 scene 생성 때 확정한다. 먼저 과제 비율 **HOLDING/SIT/CLIMB/AT/ON_TOP = 5/10/25/30/30**의 혼합 크기 분포를 만들고, reset에서는 해당 크기가 허용하는 과제를 `p(task)/허용 크기 조합 수`에 비례해 다시 선택한다. 이는 전체 혼합 분포에서 목표 과제 비율과 과제별 균등 크기를 보존한다. 유한한 환경 pool의 실측 비율에는 표본 오차가 있다. 물리 상자 randomAssignment를 끄고 source 0/1·support 2/3을 고정해 실제 크기와 RSI 캐시의 owner 연결을 보존한다. 토큰·edge 입력 순열은 계속 사용한다. 고정 평가 preset은 처음부터 해당 과제 크기로 asset을 생성한다.
- Progress는 `1/(1+max(dxy-r_valid,0)/1m)`다. `r_valid=목표 상자 XY 반대각선+buffer`; buffer는 HOLDING 0.20m, SIT/CLIMB 0.30m, ON_TOP 0.10m다. AT는 점 목표여서 반대각선 없이 0.10m다. ON_TOP은 받침 크기를 사용한다.
- AT/ON_TOP RSI는 모두 **loco/pickUp/carryWith/putDown=40/10/40/10**. HOLDING·SIT·CLIMB의 기존 50/50 skill 비율은 유지한다. AMP expert 내부 skill 분포는 원본 unified와 같고 family 사전분포만 carry/sit/climb=65/10/25로 맞춘다.
- 학습 첫 로딩에서 실제 asset의 크기별로 SIT/CLIMB/pickUp/carryWith/putDown **전체 프레임**을 0.1초 물리 검사한다. ON_TOP putDown은 source·받침 크기 쌍과 reference 최종 XY에 둔 접지 받침을 함께 검사한다. 이 RSI 받침 회전은 identity다. 학습 reset 중에는 시뮬레이션 검사를 하지 않는다.
- 통과 조건은 모든 검사 step에서 finite, root 속도 변화 ≤3m/s, 상자 속도 ≤5m/s, root·상자 이동 ≤0.30m다. 미세 접촉만으로 탈락시키지 않으며 own_success를 초기화 필수 조건으로 요구하지 않는다. Reset에서는 다른 owner/남는 상자가 reference 몸을 깊게 침범하면 배치를 다시 뽑는다. 이 검사는 초기 충격 검사이고 장기 성공 보장은 아니다.
- SIT/CLIMB은 reference의 상자 접근·높이로 상호작용 구간을 추정한다. 마지막 상호작용 뒤의 퇴장 프레임은 배제하고, **70%는 유효 상호작용 프레임의 마지막 30%, 30%는 전체 유효 프레임**에서 선택한다. 후반 후보가 없으면 앞의 유효 프레임을 쓰며, 불연속 유효 구간을 유지한다. 가능한 clip끼리는 원래 clip 가중치를 사용한다.
- 요청 skill에 유효 프레임이 없으면 같은 과제의 다른 허용 skill, 최종적으로 loco를 선택한다. 실제 선택한 skill·motion ID·시간으로 humanoid pose/velocity, 담당 source, AMP history를 함께 초기화한다. humanoid pose의 크기 보정·IK는 하지 않는다. SIT/CLIMB source는 reference XY/회전과 실제 높이를, carry source는 reference 궤적과 바닥 높이 clamp를 사용한다.
- 캐시는 `output/rsi_cache/*.pkl.gz`에 저장한다. motion tensor·상자 reference·asset 파일·물리 설정·검사 기준·RSI 코드가 바뀌면 별도 캐시를 만든다. 기본 학습 SEED는 42이며 `SEED=...`로 변경할 수 있다. 같은 설정에 새 크기 조합이 생기면 누락 조합만 추가 검사한다. 첫 실행은 이 과정으로 오래 걸릴 수 있다.
- Size RSI 실행 경로는 고정 asset의 과제 배정 확률을 한 번 계산하고 reset graph를 GPU에서 배치 생성한다. 캐시 RSI 전에 기존 motion/time을 뽑아 버리던 중복 작업도 제거했다. 설정·분포·graph 의미·AMP 연결·기존 물리 캐시는 유지한다. 기존 unified 실행 경로는 그대로이며 새 프로세스부터 적용된다. 중복 난수 추출 제거로 같은 seed의 학습 궤적은 달라질 수 있다.
- TensorBoard `sampling/rsi_<task>_actual_<skill>`은 최종 수락된 reset의 실제 skill 비율, `sampling/rsi_<task>_fallback_attempt_fraction`은 전체 초기화 시도의 대체 비율이다. `sampling/rsi_ontop_putdown_fraction`도 확인한다. `sampling/rsi_<relation>_reference_success`와 `first_step_success`로 reference 초기 성공과 첫 물리 step 뒤의 성공을 비교한다. 기존 `own_success`는 실제 물리 rollout의 성공 지표다.

```bash
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=6 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_vnc.sh "$CKPT" 2 1 4 10
```


## Unified size RSI typed bias

`approach_scenario_stage1_unified_size_rsi_typed_bias`는 위 Size RSI의 과제 분포·물리 크기·후반 RSI·캐시·AMP·보상·성공 조건·`r_valid`를 유지하고 semantic bias 생성부만 교체한 별도 scratch 실험이다.

- 전용 train config는 `amp_ma_carry_relation_unified_size_rsi_typed_bias.yaml`이며 `relation_bias_mode: typed_lookup`을 사용한다. 기존 레거시 `lookup`과 다르다. Env variant와 train mode가 맞지 않거나 actor/critic 표 공유·bias 비활성화를 지정하면 실행 전에 거부한다.
- `(source 타입, relation, target 타입)`으로 `[3,11,3,4,2]` 표에서 layer/head bias 8개를 직접 조회한다. 표는 0으로 초기화하고 PPO/critic 역전파로 학습한다. Actor·critic은 각각 792개 파라미터의 독립 표를 갖는다. 16/32/16 edge embedding·공유 edge MLP·projection은 이 실험에서 사용하지 않는다.
- 유효 edge의 방향·endpoint에 bias를 넣고, 나머지는 타입별 NONE·대각선 SELF를 사용한다. Edge 순열·토큰/GTA/bias 동시 순열·human readout 복원은 기존과 같다. Node tokenizer·GTA·attention·action/value head는 유지한다.
- 같은 타입/관계 조합은 agent·물체 슬롯과 관계없이 같은 표 값을 사용한다. 연속 관측에 따른 attention 변화는 기존 QK/GTA가 담당한다.
- Output은 `output/approach_scenario_stage1_unified_size_rsi_typed_bias`다. 기존 Size RSI checkpoint로 직접 resume/eval할 수 없다. 이 실험에서 저장한 checkpoint는 `RESUME_CHECKPOINT`로 재개한다. 기존 output과 실행 중인 학습을 변경하지 않는다.

```bash
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_train.sh 2 2048 4
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=5 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_vnc.sh "$CKPT" 2 1 4 10
```


## Unified size RSI typed bias + relation message

`approach_scenario_stage1_unified_size_rsi_typed_bias_message`는 typed-bias 실험에 관계 메시지만 추가한 별도 scratch 실험이다. Size RSI의 과제 분포·크기·후반 RSI 캐시·AMP·보상·성공 조건·`r_valid`를 유지한다.

- 기존 scalar bias 표 `[3,11,3,4,2]`와 QK→bias→softmax는 유지한다. 새 관계 표 `[3,11,3,64]`를 `(source 타입, relation, target 타입)`으로 조회하고 두 head에 32차원씩 나눈다. 표는 layer 간 공유하고 actor/critic은 분리한다. 추가 학습 파라미터는 각 6,336개, 합계 12,672개다.
- `relation_message: {enable: true, alpha: 1.0, init_std: 0.02}`가 전용 train config에 있다. 과제 relation은 작은 정규분포로 초기화하고 NONE/SELF·비과제 메시지는 0이다. 유효한 방향성 과제 edge에만 같은 attention 가중치를 곱해 source 노드로 합산한다. 기존 V 메시지의 GTA 복귀 변환 뒤, head 결합·output projection 전에 관계 메시지를 더한다.
- 저장된 edge만 gather/scatter로 처리하며 dense `[batch,head,node,node,32]`를 만들지 않는다. 토큰 셔플 때 src/dst도 함께 바꾸며 edge 순열·goal slot 교환에 대해 action/value/gradient가 유지되는지 검사한다.
- TensorBoard `relation_attention/{actor,critic}/layer{0..3}/node_message_rms`, `relation_message_rms`, `relation_to_node_rms`에 두 메시지의 크기를 기록한다. 기존 attention 진단과 동일하게 샘플된 관측에서 측정하며 relation 값에는 alpha가 적용된다.
- Env variant와 메시지 설정을 함께 검사한다. 기존 Size RSI/typebias checkpoint를 이 실험에 직접 resume/eval할 수 없다. 새 실험에서 생성한 checkpoint로만 `RESUME_CHECKPOINT` 재개한다. Output은 `output/approach_scenario_stage1_unified_size_rsi_typed_bias_message`다.

```bash
# Scratch 학습: GPU 점유를 확인한 뒤 지정
RESUME_CHECKPOINT='' TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_train.sh 2 2048 4
# 로컬 viewer / headless 평가 / 서버 VNC
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=5 TASK_GRAPH=holding_at EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_vnc.sh "$CKPT" 2 1 4 10
# 같은 실험 checkpoint 재개
RESUME_CHECKPOINT="$CKPT" TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_train.sh 2 2048 4
```


## Task 5종 공통 설정과 비교

아래 다섯 실험의 환경 설정은 checkpoint 격리용 `stage1_variant`만 다르다. Task bias/message 표현을 비교하며 과제·보상·크기·RSI·AMP와 634-D 관측을 유지한다.

| 항목 | Task message | 공유 Task MLP | Task MLP split | Task embedding | Task embedding action |
| --- | --- | --- | --- | --- | --- |
| 실험명 접미사 | `task_message` | `task_mlp` | `task_mlp_split` | `task_embedding` | `task_embedding_action` |
| Train mode | `task_role_lookup` | `task_role_mlp` | `task_role_mlp_split` | `task_type_embedding` | `task_type_embedding_action` |
| Task bias | task·출발/도착 역할·layer/head별 표 | 역할/task embedding → 공유 MLP → 공유 projection | 역할/task embedding → task별 MLP → task별 projection | task embedding → 물리 타입 쌍별 projection | task embedding → 물리 타입 쌍별 projection |
| 관계 message | task·역할별 학습 벡터 표 | 없음 | 없음 | 없음 | 없음; 사람 토큰에 상태/task concat MLP 직접 주입 |
| NONE/SELF 배경 | 물리 타입·관계·물리 타입별 표 | 별도 EdgeEncoder MLP | 별도 EdgeEncoder MLP, task별 분리하지 않음 | NONE/SELF embedding → task와 공유하는 타입 쌍별 projection | 학습 파라미터 없이 항상 0 |
| Actor/critic | 독립 | 독립 | 독립 | 독립 | 독립 |

다섯 이름 앞에는 `approach_scenario_stage1_unified_size_rsi_`가 붙는다. Env YAML은 `tokenhsi/data/cfg/multi_agent/`, train YAML은 `tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_<접미사>.yaml`, wrapper는 `tokenhsi/scripts/multi_agent/<실험명>_{train,test,vnc}.sh`다. Output은 `output/<실험명>/<run>/`이며 저장 설정은 `relation_config.yaml`, 모델은 `nn/`, TensorBoard는 `summaries/`에 있다.

### Task와 edge

H=actor(사람), O=payload(운반 상자), G=바닥 target, S=받침 target이다. Sit/climb의 O는 payload가 아닌 target 역할이다. 각 agent가 독립 배정받으며 비율은 크기 조건부 샘플링 전의 사전 확률이다.

| Task ID | 정책 task | 사전 비율 | 정책의 단방향 연결 | 보상·RSI·AMP에 사용하는 primitive |
| ---: | --- | ---: | --- | --- |
| 0 | `sit` | 10% | H→O(target) | SIT |
| 1 | `climb` | 25% | H→O(target) | CLIMB |
| 2 | `carry_at` | 32.5% | H→O, H→G, O→G | HOLDING(H→O) + AT(O→G) |
| 3 | `carry_ontop` | 32.5% | H→O, H→S, O→S | HOLDING(H→O) + ON_TOP(O→S) |

Carry의 세 연결에는 **모두 같은 `carry_at` 또는 `carry_ontop` task ID**가 들어간다. 역할 쌍 `(actor,payload)`, `(actor,target)`, `(payload,target)`이 다르므로 bias도 달라질 수 있다. Split에서도 한 task 내부 세 연결은 같은 전용 MLP·projection을 쓴다. 단독 HOLDING task는 없고 추가 H→target 연결에는 보상을 중복 부여하지 않는다.

### 보상·RSI·AMP 및 평가

- 보상은 자기 primitive 합계 ×1, 동료 보상 ×0이며 carry를 2로 나누지 않는다. PPO의 `task_reward_w=0.5`, `disc_reward_w=0.5`는 그대로다. 팀 보상 제거와 task/AMP 혼합 가중치는 다른 설정이다.
- RSI는 SIT의 loco/sit 50/50, CLIMB의 loco/climb 50/50, 두 carry의 loco/pickUp/carryWith/putDown 40/10/40/10이다. 물리 검사한 유효 프레임에서 후반 70%/전체 30% 방식으로 뽑고, 유효 skill이 없는 경우 fallback으로 실제 비율이 달라질 수 있다.
- AMP family는 carry/sit/climb 65/10/25이며 두 carry task는 같은 carry expert family를 사용한다. RSI skill 분포와 AMP expert 분포를 동일한 것으로 해석하지 않는다.
- 공유 MLP는 `16-D src role + 32-D task + 16-D dst role → 64→64→64 → layer/head bias` 경로 전체를 task 간 공유한다. Split은 embedding 표를 유지하고 뒤의 MLP·projection만 네 task로 분리한다. NONE/SELF 배경 encoder는 두 경우 모두 task encoder와 별도다.
- Task message의 메시지는 task·역할별 **상태와 무관한 학습 벡터**다. 같은 attention으로 집계하며, 현재 동작을 입력받아 행동 지시를 생성하는 별도 네트워크는 아니다.
- Test wrapper는 기본 `HEADLESS=1`, `EVAL_SKILLS=loco`, `EVAL_SKILL_PROBS=1.0`이다. 이미 든 상태를 확인할 때는 `TASK_GRAPH=carry_at` 또는 `carry_ontop`과 `EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0`을 함께 지정한다. `holding_at/holding_ontop`은 별칭이다. 로컬 viewer는 `HEADLESS=0`, 서버 viewer는 전용 VNC wrapper를 쓴다.

### 지표 읽는 법

- 성공률 비교는 `info/epochs`에 맞춰 최근 200 epoch 평균을 사용한다. 시작 시간이 다른 split은 같은 epoch 구간도 비교한다.
- `relation/edge/{holding,at,ontop,sit,climb}/own_success`는 개별 primitive 조건이다. Carry 전체 완료율과 같지 않으며 추가 정책 edge H→G/H→S의 별도 성공률은 없다. `relation/goal/final_agent_success`는 에피소드 종료 시 담당 과제 전체 성공이다. RSI를 포함한 학습 지표이므로 처음부터 수행하는 평가 결과로 해석하지 않는다.
- Bias는 **task bias 원래 값**과 **Δ = task bias − 같은 물리 타입 쌍의 NONE bias**를 함께 본다. 4 layer × 2 head의 부호 있는 평균은 head별 편차를 가린다. 큰 양의 Δ가 NONE 억제에서 나올 수 있으며 실제 attention은 QK 점수와 함께 결정된다. Bias 증가만으로 성공률 개선 원인을 확정하지 않는다.

## Task 3종 최근 확인 기록

**2026-10-06 23:10 KST 로그 확인분**을 2026-10-07 문서에 반영했다. 실시간 상태가 아니며 아래 성공률은 각 run의 최근 200 epoch 평균이다. 세 실험 모두 2명·2048환경·4물체, seed 42로 본학습을 시작했다.

| 항목 | Task message | 공유 Task MLP | Task MLP split |
| --- | ---: | ---: | ---: |
| 확인 epoch | 10,409 | 10,767 | 3,006 |
| HOLDING | 61.6% | 61.7% | 18.6% |
| AT | 5.7% | 6.0% | 6.3% |
| ON_TOP | 4.4% | 15.7% | 4.2% |
| SIT | 9.2% | 20.4% | 0.9% |
| CLIMB | 19.9% | 20.4% | 16.2% |
| 종료 시 담당 과제 전체 성공 | 8.7% | 18.2% | 6.5% |

동일한 2,801~3,000 epoch의 전체 성공은 message 5.41%, 공유 MLP 5.70%, split 6.48%였다. 해당 확인에서 scalar NaN/Inf는 없고 최근 구간의 물리 reset 실패·동료 보상 기여는 0이었다. Split의 actor/critic 각각 네 task MLP가 본학습 checkpoint 500→1,000 사이에 모두 업데이트된 것도 확인했다. 한 seed의 학습 기록이며 분리 효과나 별도 평가 성공률을 확정하지 않는다.

Bias 확인 checkpoint는 message 10,000 / 공유 MLP 10,500 / split 3,000 epoch다. 공유 MLP의 carry_at H→O는 원래 bias +2.398, NONE −16.112, Δ +18.511로 NONE 억제의 영향이 컸다. Split의 동일 연결은 원래 +0.717, Δ +4.006이었다. 전체 edge·actor/critic 값은 로컬 분석 산출물 [CSV](../output_etc/trend_review_20261006/three_edge_vs_none_latest.csv)·[그래프](../output_etc/trend_review_20261006/three_edge_vs_none_latest.png)를 참조한다. 이 `latest` 파일은 당시 분석의 보관본이며 Git에 포함되지 않는다.

검증 이력은 [changelog](../changelog.md)에 있다. Split 관련 CPU 84개 및 2048환경 짧은 학습, carry_at/carry_ontop 각각 16환경·32-step headless 평가가 통과했다. 장기 성능 비교와 VNC 화면 검증을 대신하는 결과는 아니다.

## Unified size RSI task message

`approach_scenario_stage1_unified_size_rsi_task_message`는 typed message 기반의 별도 scratch 실험이다. 과제를 `sit / climb / carry_at / carry_ontop`으로 입력하며, 각 agent의 사전 샘플링 비율은 **10 / 25 / 32.5 / 32.5%**다. 실제 고정 asset에는 기존 크기 조건부 샘플링을 적용한다. 단독 HOLDING 과제는 제외한다.

- 정책 suffix는 agent당 `[valid, task, actor, payload, target]` 한 레코드다. Task ID는 위 순서의 0~3, endpoint는 canonical token ID다. 2H/4O/2G에서 `568 scene + 56 pose + 10 task = 634`차원이며, suffix는 관측 정규화를 우회해 PPO sample에 그대로 저장한다.
- `carry_at(H,O,G)`와 `carry_ontop(H,O,S)`는 각각 actor→payload, actor→target, payload→target의 **단방향 3연결**을 만든다. 셋 모두 같은 task ID지만 역할 쌍으로 별도 조회한다. `carry_ontop(H,O,S)`의 O는 위에 놓을 물체, S는 받침이다. Sit/climb은 actor→target 한 연결이다. 역방향은 일반 QK attention과 NONE bias를 사용한다.
- Actor/critic 각각 bias `[4 task,3 src role,3 dst role,4 layer,2 head]`와 message `[4,3,3,64]` 표를 학습한다. 역할은 actor=0/payload=1/target=2다. NONE/SELF 배경 bias는 기존 물리 타입 기준 `[3,2,3,4,2]` 표다. 합계 각 **2,736개** 파라미터이며 각자 독립이다. Bias 0 초기화, message std=0.02·alpha=1, layer 공유·head별 32차원·GTA 복귀 뒤 가중합은 이전 message 실험과 같다.
- 정책 task packet은 현재 보상 graph의 owner와 endpoint에서 매번 구성한다. 보상 graph의 HOLDING+AT/ON_TOP은 유지하며 새 H→target 연결에는 별도 보상을 만들지 않는다. `self=1, teammate=0, edge_aggregation=self_sum`: 자기 primitive 보상을 합산한다. 최대 task 보상은 sit/climb 0.6, carry 1.2이며 carry를 2로 나누지 않는다. State/progress/success·r_valid·성공 saturation·패널티 식은 유지한다.
- 내부 template 슬롯은 기존 순서 `[HOLDING,SIT,CLIMB,HOLDING_AT,HOLDING_ON_TOP]`, 비율 `[0,.10,.25,.325,.325]`이다. RSI 프레임·물리 캐시와 AMP family/expert 연결은 기존 primitive graph를 사용한다. Carry family 총량 65%와 크기 분포가 같으므로 기존 asset 생성·RSI 캐시를 재사용한다. YAML의 HOLDING 크기/RSI 항목은 내부 공통 형식으로 남지만 단독 과제는 뽑지 않는다.
- Token 순열 시 bias 두 축·메시지 endpoint·GTA pose를 함께 바꾸고 human readout을 복원한다. Task/edge 레코드 순서에는 의존하지 않는다. `TASK_GRAPH=carry_at/carry_ontop`을 지원하고 기존 `holding_at/holding_ontop`은 별칭이다. `holding` 단독 preset은 거부한다.
- 전용 train YAML은 `amp_ma_carry_relation_unified_size_rsi_task_message.yaml`, mode는 `task_role_lookup`, output은 `output/approach_scenario_stage1_unified_size_rsi_task_message`다. Packet version 5와 별도 variant로 기존 checkpoint의 직접 resume/eval을 거부한다. 같은 새 실험 checkpoint로만 재개한다. 성공률 로그의 HOLDING/AT/ON_TOP은 계속 **보상 primitive** 기준이며 정책 task 분류와 구분한다.

```bash
# Scratch 학습
RESUME_CHECKPOINT='' TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_train.sh 2 2048 4
# 로컬 viewer / headless 평가 / carryWith 상태의 서버 VNC
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=5 TASK_GRAPH=carry_at EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_vnc.sh "$CKPT" 2 1 4 10
# 같은 실험 checkpoint 재개
RESUME_CHECKPOINT="$CKPT" TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_train.sh 2 2048 4
```

## Unified size RSI task MLP

`approach_scenario_stage1_unified_size_rsi_task_mlp`는 위 task message의 env 설정을 그대로 사용하는 별도 scratch 실험이다. Env YAML은 checkpoint 분리용 `stage1_variant`만 다르다. 과제 비율 10/25/32.5/32.5%, 단독 HOLDING 제외, carry의 단방향 3연결, 자기 primitive 보상 합계(`self=1, teammate=0`), 크기 분포·물리 검사 RSI·AMP family/전문가 연결·634-D task packet을 모두 공유한다. Carry의 loco/pickUp/carryWith/putDown RSI는 AT와 ON_TOP 모두 40/10/40/10이다.

- 정책은 `task_role_mlp`다. 출발 역할 16-D·task 32-D·도착 역할 16-D 임베딩을 연결한 뒤 `64→64→64` ReLU MLP와 layer/head projection으로 attention bias를 만든다. 36가지 task/역할 조합을 먼저 인코딩하고 각 연결에 조회한다. Actor/critic은 별도 파라미터를 갖는다.
- NONE/SELF 배경은 기본 Size RSI처럼 물리 타입·관계·물리 타입의 `EdgeEncoder` MLP를 사용한다. 두 bias projection은 0으로 초기화한다. GTA·토큰 순열은 유지한다.
- `relation_message.enable=false`이며 관계 메시지 표와 가산 경로는 없다. 일반 attention의 value 집계는 유지한다. 기존 task message와 네트워크 및 checkpoint 계약이 달라 직접 resume/eval을 거부한다.
- 전용 env/train YAML·train/test/VNC·output을 사용한다. RSI 캐시 경로는 `output/rsi_cache`로 같으며 내용 hash가 맞으면 재사용한다. 아래 GPU 0은 현재 로컬 서버 예시이며 실행 전 점유를 확인한다.

```bash
# Scratch 학습: 2명, 2048환경, 4물체
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_train.sh 2 2048 4
# 짧은 학습 확인: 본학습과 output 분리
TOKENHSI_GPU=0 MAX_ITERATIONS=1 RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_train.sh 2 2048 4
# CKPT에 같은 task MLP 실험의 checkpoint 절대 경로를 지정
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=carry_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=0 TASK_GRAPH=carry_ontop EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_vnc.sh "$CKPT" 2 1 4 10
# 같은 task MLP checkpoint에서 재개
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_train.sh 2 2048 4
```

## Unified size RSI task MLP split

`approach_scenario_stage1_unified_size_rsi_task_mlp_split`는 task MLP의 공유 영향을 비교하는 별도 scratch 실험이다. Env는 `stage1_variant`만 다르며 과제·3연결·자기 보상·크기·RSI·AMP·634-D packet은 task MLP 및 task message와 같다. Carry의 RSI는 AT/ON_TOP 모두 loco/pickUp/carryWith/putDown 40/10/40/10이고 기존 물리 캐시를 재사용한다.

- 출발 역할 `Embedding(3,16)`·task `Embedding(4,32)`·도착 역할 `Embedding(3,16)`을 유지한다. 출발·도착 표는 별도이며 네 task가 공유한다. Task 표는 task별 다른 행을 쓴다.
- Concat 이후 `64→64→64` MLP를 sit/climb/carry_at/carry_ontop별로 네 개 만든다. Bias projection도 `[task=4,layer=4,head=2,64]`로 분리한다. 한 task 내부의 세 역할 연결은 같은 전용 MLP·projection을 사용한다. Task 간 역할 embedding 공유의 영향은 계속 남는다.
- Task별 9개 역할 조합만 처리한 뒤 환경별로 조회한다(총 36개). Actor/critic은 전체 파라미터가 독립이다. NONE/SELF 배경은 기존 별도 MLP를 유지하며 task별로 분리하지 않는다. 관계 메시지는 없고 GTA는 유지한다.
- Task encoder는 actor/critic 각각 35,552개 파라미터다(기존 9,056개). 전체 네트워크는 4,209,730개로 52,992개(+1.27%) 증가한다. 추가 FP32 weight 약 0.20 MiB, gradient·Adam 상태까지 약 0.81 MiB이며 이는 전체 실행 메모리 증가의 실측값은 아니다.
- 전용 train mode는 `task_role_mlp_split`이다. Packet v5는 유지하지만 variant/fusion 및 weight shape가 달라 task message·공유 task MLP checkpoint를 직접 resume/eval할 수 없다. 전용 config·train/test/VNC·output에서 scratch로 시작한다.

```bash
# Scratch 학습: 2명, 2048환경, 4물체
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp_split bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_train.sh 2 2048 4
# 짧은 확인용 output
TOKENHSI_GPU=0 MAX_ITERATIONS=1 RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp_split_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_train.sh 2 2048 4
# CKPT에 같은 task MLP split 실험의 checkpoint 절대 경로를 지정
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=carry_at EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=0 TASK_GRAPH=carry_ontop EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_vnc.sh "$CKPT" 2 1 4 10
# 같은 split checkpoint 재개
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT="$CKPT" OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp_split bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_train.sh 2 2048 4
```

## Unified size RSI task embedding

`approach_scenario_stage1_unified_size_rsi_task_embedding`은 공유 task MLP 기반의 별도 scratch 실험이다. Env YAML은 `stage1_variant`만 다르고, train YAML은 `relation_bias_mode=task_type_embedding`만 다르다. 보상·성공 조건·과제 분포·634-D task packet·RSI·AMP·GTA·토큰/edge 셔플을 유지한다.

- Actor/critic 각각 `Embedding(6,64)` 하나를 둔다. 카테고리는 `[sit, climb, carry_at, carry_ontop, NONE, SELF]`다. 역할/물리 타입/agent ID를 embedding 입력으로 넣지 않으며 중간 MLP·관계 message가 없다.
- Projection은 `[src type=3, dst type=3, layer=4, head=2, 64]`다. `b[c,s,t,l,h] = dot(E[c], W[s,t,l,h])`이며 H/O/G 방향 쌍·layer/head마다 독립이다. Task·NONE·SELF는 **같은 W를 공유**하고 카테고리 embedding만 다르다. Actor와 critic은 전체 파라미터가 독립이다.
- 같은 task·타입 쌍·layer/head라면 agent/물체 ID와 무관하게 bias가 같다. `carry_at`의 H→O/H→G/O→G는 embedding을 공유하고 projection은 다르다. `climb H→O`와 `carry_at H→O`는 projection을 공유하고 embedding은 다르다.
- 받침 S는 O 타입이다. 따라서 `carry_ontop`의 H→운반 상자와 H→받침은 같은 bias를 받는다. 운반 상자→받침은 O→O projection을 쓴다. 이전 actor/payload/target **역할 쌍별** 설계와 구분한다.
- NONE은 task 연결이 없는 방향, SELF는 대각선에 적용한다. 배경 위에 task bias를 중복 가산하지 않고 유효 task 연결을 교체한다. 역방향·다른 agent의 비연결 토큰은 NONE으로 남는다.
- Canonical endpoint의 실제 H/O/G 타입으로 bias를 만든 뒤 토큰 순열에 맞춰 bias 두 축과 GTA pose를 함께 이동한다. Human readout은 역순열로 복원한다. Task record/primitive edge 순서로 projection을 고르지 않는다.
- Embedding은 std=.02 랜덤, projection은 0으로 초기화한다. 최초 bias는 0이며 학습 후 성장 속도·성공률 개선은 보장하지 않는다. Encoder당 embedding 384 + projection 4,608 = **4,992개** 파라미터다. 전체 네트워크는 4,130,626개로 공유 task MLP보다 26,112개 적다.
- RSI는 SIT loco/sit 50/50, CLIMB loco/climb 50/50, 두 carry loco/pickUp/carryWith/putDown 40/10/40/10을 유지한다. AMP carry/sit/climb family 65/10/25 및 expert matching, size RSI 캐시 `output/rsi_cache`도 그대로다. 보상은 자기 primitive 합계 ×1·동료 ×0이고 PPO task/AMP 가중치는 0.5/0.5다.
- 전용 variant/fusion과 weight 구조로 기존 세 실험 checkpoint의 직접 resume/eval을 거부한다. 학습·평가 모두 task embedding 전용 wrapper와 checkpoint를 사용한다.

2026-10-07 검증: 관련 CPU 91개, GPU 0의 2048환경 짧은 학습(저장 epoch 2), 저장 모델의 carry_at/carry_ontop + carryWith 각각 16환경·32-step headless 평가를 완료했다. 학습 scalar 233종/466값 모두 finite, 물리 reset 실패·물체 binding/보상 합산 오류 0이며 6개 카테고리와 타입 쌍 projection의 학습 경로를 확인했다. 검증 output은 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_check/`와 `_check_carry_at/`, `_check_carry_ontop/`다. 본학습·장기 성능 및 VNC 화면은 아직 검증하지 않았다.

```bash
# Scratch 학습: 2명·2048환경·4물체. 실행 전 GPU 점유 확인
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_train.sh 2 2048 4
# 짧은 검증은 별도 output
TOKENHSI_GPU=0 MAX_ITERATIONS=1 RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_train.sh 2 2048 4
# CKPT에는 task embedding 실험의 checkpoint 경로 지정
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=carry_at EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=0 TASK_GRAPH=carry_ontop EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_vnc.sh "$CKPT" 2 1 4 10
# 같은 실험 checkpoint 재개
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT="$CKPT" OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_train.sh 2 2048 4
```

## Unified size RSI task embedding action

`approach_scenario_stage1_unified_size_rsi_task_embedding_action`은 기존 task embedding 기반의 별도 scratch 실험이다. Env YAML은 `stage1_variant`, train YAML은 `relation_bias_mode=task_type_embedding_action`만 다르다. 보상·샘플링·RSI·AMP·GTA·634-D packet·토큰/edge 셔플은 동일하다.

- Actor/critic 각각 `Embedding(4,64)`와 `[src type=3,dst type=3,layer=4,head=2,64]` projection을 사용한다. 카테고리는 sit/climb/carry_at/carry_ontop이며, NONE/SELF embedding은 없다. 비과제 연결·대각선 bias는 항상 **0**이고 attention 연결 자체를 마스킹하지 않는다. Task bias는 부호·크기를 제한하거나 강제로 증폭하지 않는다.
- 같은 branch에서 task embedding은 bias와 직접 주입 경로가 공유한다. 각 layer의 `A×V → GTA 복귀 → head concat` 뒤, output projection 전에 사람 토큰에만 `z_H += MLP(concat(z_H, e_task))`를 적용한다. MLP는 **Linear(128,64) → ReLU → Linear(64,64)**다. Task 벡터에 A를 곱하거나 상대 j축으로 합산하지 않는다. 사람마다 유효 task record 하나를 사용하며 carry 연결 수로 주입 횟수가 늘지 않는다. 물체·목표·padding에는 직접 추가하지 않는다.
- MLP 하나를 네 layer가 공유하고 actor/critic은 별도다. Task·agent별 MLP는 만들지 않는다. Canonical packet의 `actor` endpoint를 token permutation의 역매핑으로 옮긴 뒤 gather/scatter하므로 task record 순서나 인간 토큰의 현재 위치에 의존하지 않는다. Human readout은 기존처럼 역순열 복원 후 수행한다.
- Embedding std=.02, task bias projection=0, **추가 MLP 출력층 weight/bias=0**으로 시작한다. Builder의 전역 Linear 초기화 뒤에도 이 초기화를 보존한다. 첫 optimizer step에는 출력층/projection이 먼저 학습되고 이후 MLP 앞층과 embedding으로 gradient가 전달된다. 초기 주입은 정확히 0이다.
- 주입 MLP는 branch당 12,416개, 두 branch 합계 24,832개다. NONE/SELF embedding 256개를 제거해 전체 모델은 **4,155,202개**로 기존 task embedding보다 24,576개(+0.595%) 증가한다. Layer 공유는 파라미터 공유이며 forward는 네 번 실행한다.
- 보상은 자기 primitive 합계 ×1·동료 ×0, PPO task/AMP 0.5/0.5다. Task prior는 sit/climb/carry_at/carry_ontop 10/25/32.5/32.5%, AMP family carry/sit/climb 65/10/25를 유지한다. SIT loco/sit 50/50, CLIMB loco/climb 50/50, carry loco/pickUp/carryWith/putDown 40/10/40/10 RSI와 기존 물리 캐시 `output/rsi_cache`를 사용한다.
- `relation_attention/{actor,critic}/layer{0..3}/task_action_{input_rms,delta_rms,to_input_rms}`를 기록한다. 이는 사람 토큰의 주입 전 RMS·추가량 RMS·비율이며 task 의미를 유용하게 사용한다는 증거 자체는 아니다. CPU 검증은 bias/QK를 고정한 task 벡터 교체의 출력 영향도 확인한다. 학습된 정책의 task 입력 제거/교체 성공률 평가는 별도로 수행해야 한다.
- 새 variant·fusion 계약·4행 embedding·주입 MLP로 기존 checkpoint의 resume/eval 혼용을 거부한다. 기존 task embedding checkpoint를 직접 이어 학습하지 않는다.

2026-10-08 검증: 관련 CPU 103개, GPU 0·2048환경 확인 학습(epoch 2/frame 262144), carry_at 및 역할 교환 ON_TOP의 carryWith·16환경·32-step headless 평가 통과. 기존 RSI 캐시 hit, scalar 257종/514값 finite, 물리 reset 실패·binding/보상 합산 오류 0. 네 task embedding·MLP·bias projection의 실제 업데이트와 SELF/NONE=0을 확인했다. Output은 `_task_embedding_action_check` 및 `_check_carry_at`, `_check_carry_ontop`이다. 이는 실행/학습 경로 검증이며 장기 성공률·VNC 화면·학습된 task 입력 제거/교체 평가는 미검증이다.

```bash
# Scratch 본학습: 실행 전 GPU 점유 확인, 2명·2048환경·4물체
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_action bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_train.sh 2 2048 4
# 짧은 확인 학습: 별도 output
TOKENHSI_GPU=0 MAX_ITERATIONS=1 RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_action_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_train.sh 2 2048 4
# CKPT에는 이 새 실험의 checkpoint 지정
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=carry_at EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=0 TASK_GRAPH=carry_ontop EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_vnc.sh "$CKPT" 2 1 4 10
# 한 명씩 SIT / CLIMB 고정: 첫 번째 사람 SIT, 두 번째 사람 CLIMB, loco 시작
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=sit_climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 TASK_GRAPH=sit_climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_vnc.sh "$CKPT" 2 1 4 10
# 가변 인원 추론 예시: M>=1, O>=M이면 원하는 수 지정 가능
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_test.sh "$CKPT" 9 1 14 10
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_vnc.sh "$CKPT" 9 1 14 10
# 같은 새 실험 checkpoint 재개
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT="$CKPT" OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_action bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_train.sh 2 2048 4
```

Action의 `random_scenario` 평가는 **사람 M≥1·물체 O≥M**를 지원한다. 인자는 `<checkpoint> M 환경수 O 반복수`다. 각 사람에게 서로 다른 주물체를 배정하고, 남는 O−M개는 받침/배경으로 쓴다. ON_TOP이 받침보다 많이 뽑히면 무작위로 초과 담당자를 골라 실제 크기에 적격인 다른 과제로 재추첨한다. O=M이면 ON_TOP 없이 나머지 과제를 수행한다. 모두 ON_TOP으로 고정하는 `carry_ontop`은 과제 정의상 O≥2M이 필요하며, `sit_climb`은 사람 순서대로 SIT/CLIMB를 번갈아 배정한다. 가변 인원 평가에서는 사람 간 출발 간격과 장면 반경을 늘려 밀집 reset 실패를 줄인다. 기존 2명·4물체 학습/평가의 샘플링 및 checkpoint 계약은 유지한다. 기존 2명 정책의 가변 인원 **실행 지원**이며 동일한 성공률을 보장하는 것은 아니다.

Action의 반복 평가는 기존 `_test.sh`의 마지막 인자로 횟수를 지정한다. 다음은 저장본 하나를 64환경·3회·600 step·loco 시작으로 평가하는 명령이다. 저장본별 `CKPT`와 `OUTPUT_PATH`를 바꾸면 최근 checkpoint들을 비교할 수 있다. 결과는 `<OUTPUT_PATH>/metrics/edge_context_*.json`의 `repeat_0..2`에 저장되며 `final_agent_goal_success`는 종료 시 개인 목표 성공률, `ever_agent_goal_success`는 평가 중 한 번이라도 성공한 비율이다. `final_edge_own_success`에는 SIT/CLIMB/AT/ON_TOP별 종료 비율이 있다. 과제별 회차 평균은 표본 수로 가중한 합산 성공률과 구분한다.

```bash
TOKENHSI_GPU=0 HEADLESS=1 SEED=42 EPISODE_LENGTH=600 TASK_GRAPH=random_scenario EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0 OUTPUT_PATH=output_etc/action_eval_repeat3 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_test.sh "$CKPT" 2 64 4 3
```

### Action 시작 거리 대조 평가

[거리 평가 스크립트](../output_etc/action_distance_eval_20261014/evaluate.py)는 Action 55k·2명·4물체·64환경·3회·600 step·loco 시작·seed 42·기존 성공/낙상 조건을 유지하고, 평가 시 `box.reset.ownerLocoDistanceRange`만 바꾼다. 표준 config/checkpoint 검사를 통과한 뒤 측정 episode reset 전에 메모리에서만 적용하며 학습 YAML은 수정하지 않는다.

```bash
TOKENHSI_GPU=0 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
python output_etc/action_distance_eval_20261014/evaluate.py --range 1 2
python output_etc/action_distance_eval_20261014/evaluate.py --range 3 5
python output_etc/action_distance_eval_20261014/evaluate.py --range 5 10
python output_etc/action_distance_eval_20261014/analyze.py'
```

각 구간의 `initial_*.pt`·`trace_*.pt`·`results.json`과 통합 `summary.json`을 별도 보관한다. 실제 시작 거리 범위·finite·성공/접근/낙상 재집계와 1~2m 기준의 기존 표준 평가 재현을 검사한다. 접근 지표는 **담당 물체 중심 XY 1m 이내 도달**이며 과제 성공과 구분한다. 두 사람 중 한 명이 넘어지면 장면이 끝나는 기존 규칙을 유지한다. 같은 seed라도 배치 거절·종료 후 reset 때문에 구간 간 개별 장면은 정확히 짝지어지지 않는다.

2026-10-14의 구간별 384시도 합산 종료 성공은 **93.2/76.8/9.9%**, 물체 접근은 **100/86.2/12.2%**, 낙상 종료 장면은 **1.0/14.1/88.0%**다. [전체 결과와 해석](../output_etc/action_distance_eval_20261014/README.md), [독립 재집계](../output_etc/action_distance_eval_20261014/summary.json)를 참조한다.

## Stage1 v2 fixed

고정 선반 검증을 반영한 별도 실험 **`stage1_v2_fixed`**다. [Env](../tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml)·[train](../tokenhsi/data/cfg/train/rlg/stage1_v2_fixed.yaml), 전용 `stage1_v2_fixed_{train,test,vnc}.sh`를 사용한다. 기본 2명·2048환경·4물체, output `output/stage1_v2_fixed`, run `Stage1V2Fixed_*`다. 본학습은 자동 실행하지 않는다.

- v2의 크기·샘플링·성공 조건·보상·모델·AMP를 유지하고 `box.reset.platformPhysics: fixed_warmup_v1`만 선택한다. 기존 stage1/v2 YAML에는 이 옵션이 없어 기존 동적 선반 경로를 유지한다.
- 선반은 `fix_base_link=True`이며 에피소드 reset마다 다시 배치한다. **위치가 에피소드 간 고정이라는 뜻은 아니다.** GPU PhysX의 최초 재배치를 위해 프로세스의 첫 전체 reset 전에 물리 계산을 한 번 한다. 이후 일반 reset이 사람·상자·진행 상태·AMP 이력을 초기화한다.
- AT는 상자 **중심** 목표다. 목표 Z = 선반 상판 Z + 상자 높이/2, 선반 중심 Z = 목표 Z − 상자 높이/2 − 0.01m다. 바닥 목표는 상판 Z=0이다. 고정 선반은 바닥과 겹쳐도 떠밀리지 않는다.
- 새 run의 `relation_config.yaml`과 checkpoint metadata에 env와 **sim을 함께 저장**한다. test/VNC는 checkpoint의 저장 env를 읽고 고정 선반 모드를 명시적으로 적용한다. `STAGE1_V2_FIXED_ENV_CONFIG`가 있으면 그 파일을 우선한다. 옛 v2 snapshot에 sim이 없을 때만 새 실험 YAML의 전체 sim을 보충하고 출처를 출력한다. `rest_offset=0`을 요구하며 불완전한 sim은 거부한다. 해석된 입력은 output의 `fixed_env_*.yaml`에 보관한다.
- 기존 **v2** checkpoint는 새 물리 환경에서 평가하거나 명시적 resume할 수 있다. 이는 물리 환경을 바꾸는 후속 실험이며 이전 실험의 동일 조건 재현이 아니다. stage1 checkpoint는 v2 보상/크기 계약과 달라 혼용하지 않는다.
- RSI 캐시는 기존 v2 디렉터리를 사용한다. 이번 재검사 6,413개 profile의 프레임/phase mask는 기존 캐시와 모두 일치했다. 코드·입력 조건으로 cache key가 달라지면 최초 screening 시간이 필요하다.

```bash
# 새 본학습: 직접 실행할 때만 시작
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/stage1_v2_fixed bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_train.sh 2 2048 4
# 짧은 확인 학습
TOKENHSI_GPU=0 MAX_ITERATIONS=1 OUTPUT_PATH=output/stage1_v2_fixed_check bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_train.sh 2 2048 4
# CKPT: 기존 v2 또는 새 fixed 실험의 checkpoint
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_test.sh "$CKPT" 2 32 4 3
TOKENHSI_GPU=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_vnc.sh "$CKPT" 2 1 4 10
# 저장 상태에서 새 output으로 계속 학습
TOKENHSI_GPU=0 RESUME_CHECKPOINT="$CKPT" OUTPUT_PATH=output/stage1_v2_fixed_resume bash tokenhsi/scripts/multi_agent/stage1_v2_fixed_train.sh 2 2048 4
```

검증 자료: [결과 JSON](../output_etc/v2_fixed_validation_20261015/validation.json). 격리 후보에서 CPU 86개, 32환경의 13과제/초기 skill·부분 reset·AT 마커 및 바닥/0.005/0.02/0.25/0.75m 선반 접촉, 2048환경 epoch3 및 resume epoch4를 확인했다. 선반 정지 오차0·상자 밑면 오차≤0.0043mm, 추가 초기 물리 계산 약21ms였다. 세 epoch 속도 비교는 비슷한 범위이며 장기 성능 보장은 아니다. 테스트의 연속 reset 호출과 float32 XY 허용 오차 오류는 격리 검사 코드에서 바로잡았고 최초 실패 로그도 보관했다. GUI/VNC 화면은 기존 NVML 버전 불일치 때문에 미검증이다.
최종 연결 후 CPU **97개**, 새 train wrapper의 **2048환경 epoch2/frame262144**·514 scalar finite·reset 실패0을 확인했다. 운영 코드로도 동일13과제·다섯 높이 접촉 검사를 통과했다. 새 checkpoint 및 기존 v2 8k의 표준 headless32환경32step 평가도 통과했다. VNC wrapper는 GPU 선택 단계에서 기존 NVML 오류로 종료됐다.

## Stage1 v2

새 실험은 **`stage1_v2`**다. [Env](../tokenhsi/data/cfg/multi_agent/stage1_v2.yaml)·[train](../tokenhsi/data/cfg/train/rlg/stage1_v2.yaml), 전용 `stage1_v2_train.sh`/`stage1_v2_test.sh`/`stage1_v2_vnc.sh`를 사용한다. 기본 2명·2048환경·4물체, output `output/stage1_v2`, run 이름 `Stage1V2_*`다. 기존 stage1/Action checkpoint와 직접 resume/eval을 혼용하지 않는다.

- stage1의 공유 task bias·Action MLP·CPA·원본 규칙 출발/AT 목적지 선반·TokenHSI 거리·운반 RSI 50/10/30/10을 유지한다.
- 크기는 축별 0.05m 간격. Climb XY 0.4~0.8·Z 0.2~0.7m, 받침/배경 XY 0.4~0.8·Z 0.25~0.5m다. Carry XYZ 0.2~0.6, Sit XY 0.25~0.8·Z 0.35~0.5m는 유지한다. 적격 과제 계산과 실제 asset 생성에 같은 범위를 사용한다.
- AT는 3D 20cm+높이 오차1mm, Climb은 내부80%+root/양발 평균 높이 조건이다. HOLDING·SIT·ONTOP 조건과 dense state 곡선은 유지한다. 학습 성공 보너스와 eval 순간 성공은 같은 함수를 사용한다.
- RSI는 `output/rsi_cache_stage1_v2`에 별도 생성한다. 처음 실행할 때 실제 물리 screening에 시간이 걸린다. 크기별 관통·바닥 침범·초기 충격·낙상을 검사하고, 실제 상자 높이로 SIT/CLIMB phase를 계산한다. 모션을 수직 이동시키지 않는다. 유효 프레임이 없는 skill의 fallback과 명목/실행 비율을 구분한다. 새 seed·환경 수가 새 크기 조합을 만들면 추가 검사가 필요하다.
- AMP family/과거 이력은 담당 task/실제 선택 motion에 연결한다. 모델 토큰·edge/task 셔플 및 PPO 행별 demo/replay 매칭을 유지한다. 상세 비교와 해석은 [학습·평가 비교 문서](tokenhsi_vs_stage1_v2_train_eval.md)를 따른다.

```bash
# 본학습 — 실행 전에 GPU 점유 확인
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/stage1_v2 bash tokenhsi/scripts/multi_agent/stage1_v2_train.sh 2 2048 4
# 별도 확인 학습
TOKENHSI_GPU=0 MAX_ITERATIONS=1 OUTPUT_PATH=output/stage1_v2_check bash tokenhsi/scripts/multi_agent/stage1_v2_train.sh 2 2048 4
# CKPT는 stage1_v2에서 생성한 checkpoint
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/stage1_v2_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/stage1_v2_test.sh "$CKPT" 2 32 4 3
TOKENHSI_GPU=0 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/stage1_v2_vnc.sh "$CKPT" 2 1 4 10
# 같은 실험 재개
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT="$CKPT" OUTPUT_PATH=output/stage1_v2 bash tokenhsi/scripts/multi_agent/stage1_v2_train.sh 2 2048 4
```

`TASK_GRAPH=sit|climb|carry_at|carry_ontop|random_scenario`, `TASK_ROLE_SWAP`, `EVAL_SKILLS`, `EVAL_SKILL_PROBS`, `EPISODE_LENGTH`, `SEED`를 지원한다. test/VNC는 기본적으로 checkpoint의 `<run>/relation_config.yaml`을 자동 선택한다. 다른 동일 실험 env를 의도적으로 사용하려면 `STAGE1_V2_ENV_CONFIG`를 지정한다. 저장 config가 없으면 명시적 지정을 요청하며 종료한다. 서로 다른 크기/RSI 설정으로의 resume는 거부한다.

2026-10-14 검증: CPU **70개**, GPU 0·**2048환경** 확인 학습 epoch2와 resume epoch3·771 scalar finite·물리 reset 실패0 통과. 새 캐시6,413 profile, 실제 RSI10,570개/AMP 과거95,130프레임·PPO48 minibatch/786,432행 정렬·actor/critic 관련 파라미터 업데이트를 확인했다. 표준 headless32환경32step, 별도13과제/RSI416step·310reset·772거리·CPA·선반 접촉도 통과했다. [검증 결과](../output_etc/stage1_v2_check_20261014/validation.json). 확인 output은 `output/stage1_v2_check/`·`output/stage1_v2_resume_check/`이며 본학습은 실행하지 않았다. VNC는 wrapper를 실제 호출했지만 기존 NVML 불일치로 GPU 선택에서 막혀 화면 미검증이다. CarryWith35/712 profile은 fallback, Climb129/715 profile은 phase 공백으로 전체 valid 프레임 선택이므로 명목 RSI 비율과 실제 실행 분포를 구분한다.

## stage1

`stage1`은 `approach_scenario_stage1_unified_size_rsi_task_embedding_action` 기반의 별도 scratch 실험이다. Env/train YAML 모두 `stage1.yaml`, 전용 실행은 `stage1_train.sh`·`stage1_test.sh`·`stage1_vnc.sh`, 기본 output은 `output/stage1`이다. 기존 Action/CPA checkpoint와 직접 resume/eval을 섞지 않는다.

- **Bias:** actor/critic 각각 `Embedding(4,64)`와 `[layer=4,head=2,64]` projection을 사용한다. 같은 task·layer·head면 H/O/G 타입·역할과 무관하게 같은 additive bias다. AT와 ON_TOP의 세 방향 task 연결 모두 동일한 값을 받는다. NONE/SELF=0, softmax/GTA, 사람 토큰의 `z += MLP([z,e_task])` 및 layer 공유·actor/critic 독립성은 유지한다. Branch당 bias encoder 768개, 전체 모델 4,147,010개 파라미터다. 선반 actor는 추가하지만 H/O/G 토큰과 634-D packet은 유지한다.
- **CPA:** 사람 root XY 상대 위치/속도의 접근도 × 최근접 거리 위험 × `0.99^(t/control_dt)`, 거리 0.7m·계수 0.5다. 기존 현재 거리 충돌항을 교체하며 중복 합산하지 않는다.
- **출발 선반:** AT/ON_TOP의 loco/default 초기화 payload만 적용한다. 50% 확률로 밑면 높이 U(0,1m)를 더하고 payload 윗면 1.2m로 clamp한다. 0.4×0.4×0.02m 판을 밑면에 맞추고 출발 상자만 접촉 초기화 여유 5cm를 추가한다(일시적으로 윗면 1.25m 가능). 바닥을 선택해도 원본처럼 판 윗면을 z=0에 둔다. `pickUp/carryWith/putDown` RSI 출발 상자는 모션을 따르며 선반을 비활성화한다.
- **도착 선반:** AT의 loco/pickUp/carryWith 시작에서 목적지 높이를 출발과 독립적으로 같은 규칙으로 추첨한다. putDown AT는 원본 마지막 프레임 XY·바닥 높이를 유지한다. ON_TOP은 기존 받침 상자가 도착지이며 별도 도착 선반을 두지 않는다. SIT/CLIMB·받침·배경 상자는 높이지 않는다.
- GPU PhysX에서 reset 위치에 실제 충돌 형상이 따라가도록 기존 중력 비활성·고질량 dynamic 판을 재사용한다. 원본 고정 받침의 동작을 근사하며 발 충돌 필터도 기존 repo 설정을 유지한다. 비활성 판은 20m 위로 이동한다.
- **거리·RSI (2026-10-14 변경):** `box.reset.samplingMode: tokenhsi`에서 loco 사람→물체 XY 반경은 SIT U(1,5m), CLIMB/AT/ON_TOP U(1,10m)다. AT 목표와 ON_TOP 받침 XY는 공통 환경 원점 기준 각 축 U(-4.5,4.5m), 사람·payload와 각각 최소 1m로 뽑는다. 사람별 spawn offset으로 목표 영역을 옮기거나 축소하지 않는다. loco/pickUp/carryWith에 적용하며 putDown은 기존 모션 마지막 위치를 유지한다. 다중 물체의 관통/간섭 재추첨은 유지하므로 최종 수락 분포가 원본 1인 장면과 완전히 같지는 않다. ON_TOP 받침 높이는 기존 크기에 따른 바닥 배치를 유지한다.
- **운반 RSI:** AT/ON_TOP 모두 loco/pickUp/carryWith/putDown **50/10/30/10**. SIT·CLIMB 50/50, 크기 분포·과제/성공 보상·AMP·후반 프레임 우대·물리 캐시는 유지한다. 과제 prior도 10/25/32.5/32.5%다.
- 새 학습 run 이름은 `Stage1TokenHSISampling_*`, config와 output 루트는 `stage1`을 유지한다. 이미 실행 중인 구버전 근거리 학습에는 적용되지 않는다. 구버전 config는 `samplingMode` 없음·1~2m·40/10/40/10으로 계속 지원하며, 다른 거리/RSI 설정으로의 암묵적 학습 resume는 거부한다. 평가에서는 모델 구조가 같아 기존 stage1 checkpoint도 사용할 수 있으며 결과 JSON의 `stage1_sampling_mode`로 평가 분포를 구분한다.
- test/VNC는 기본적으로 checkpoint의 `<run>/relation_config.yaml`을 자동 선택하므로 구버전 checkpoint는 학습 당시 1~2m 분포를 사용한다. 다른 평가 분포를 의도적으로 사용하려면 `STAGE1_ENV_CONFIG`를 지정한다(최신 분포: `tokenhsi/data/cfg/multi_agent/stage1.yaml`). 저장 config가 없으면 명시적 지정을 요청하며 종료한다. 기존 분포로 학습을 재개할 때는 여전히 `STAGE1_ENV_CONFIG=output/stage1/<기존_run>/relation_config.yaml`을 train 명령 앞에 지정한다. 새로운 기본 설정과 기존 Action/CPA checkpoint는 혼용하지 않는다.

```bash
# 본학습: 실행 전 GPU 점유 확인. GPU 번호는 사용할 장치로 변경.
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/stage1 bash tokenhsi/scripts/multi_agent/stage1_train.sh 2 2048 4
# 확인 학습: 환경 수 유지, 반복 횟수만 축소
TOKENHSI_GPU=0 MAX_ITERATIONS=1 RESUME_CHECKPOINT='' OUTPUT_PATH=output/stage1_check bash tokenhsi/scripts/multi_agent/stage1_train.sh 2 2048 4
# CKPT에는 stage1에서 생성한 checkpoint 경로 지정
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/stage1_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/stage1_test.sh "$CKPT" 2 32 4 3
TOKENHSI_GPU=0 TASK_GRAPH=carry_at bash tokenhsi/scripts/multi_agent/stage1_vnc.sh "$CKPT" 2 1 4 10
# 기존 stage1 학습 재개
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT="$CKPT" OUTPUT_PATH=output/stage1 bash tokenhsi/scripts/multi_agent/stage1_train.sh 2 2048 4
```

`TASK_GRAPH=sit|climb|carry_at|carry_ontop|random_scenario`, `EVAL_SKILLS`/`EVAL_SKILL_PROBS`, `TASK_ROLE_SWAP`, `EPISODE_LENGTH`, `SEED`를 기존 wrapper와 동일하게 지원한다. 선반 규칙은 학습·평가에 공통 적용한다. 저장 env의 거리·크기·보상 설정을 사용하되, viewer의 인원/환경/물체 수·graph·초기 skill(기본 loco) 등 평가 인자는 계속 우선한다. 이미 실행 중인 viewer는 다음 실행부터 새 config 선택이 적용된다.

2026-10-14 **RSI 사전 감사·아직 미적용인 제안:** 현재 Climb 높이는 0.25~0.50m, 성공 XY 내부 90%, AT 성공은 기존 phi≥0.9·높이 1mm다. 논의한 Climb 0.20~0.70m·내부 80%, AT 3D 20cm·높이 1mm는 아직 운영 설정에 적용하지 않았다. [감사 결과](../output_etc/stage1_rsi_height_audit_20261014/validation.json), [높이별 분석](../output_etc/stage1_rsi_height_audit_20261014/geometry_summary.json)을 참조한다. 독립된 높은 Climb 상자 둘을 적층으로 잘못 제한하는 검사와, 0.60~0.70m 상판의 성공 자세를 제공하지 못하는 기존 RSI를 먼저 다뤄야 한다. 초기 성공 자세가 없다는 사실만으로 정책 학습 불가능을 뜻하지는 않는다. 기존 캐시는 초기 충격 검사이며 장기 안정성 보장이 아니다. Carry의 빈 profile은 다른 skill로 대체되고, 현재 SIT late pool은 phase가 비어 전체 valid로 대체되므로 명목 비율과 실제 선택 분포를 구분한다. 받침·배경 크기 변경은 미정이다.

2026-10-14 **Action 55k 높이 대조 평가:** 실제 Climb 상자 Z=0.40/0.50/0.55/0.60/0.65/0.70m, 각 높이 XY 81조합·486 agent 시도, loco 출발 1~2m·20초에서 제안 내부 80% final **99.8/99.4/98.6/96.7/91.6/53.9%**. 상자 기울기≤15°·상판 높이 오차≤5cm를 추가한 final은 **99.8/99.4/98.6/96.3/90.1/43.4%**다. [결과](../output_etc/action_climb_height_eval_20261014/summary.json)·[검증](../output_etc/action_climb_height_eval_20261014/validation.json). 높은 상자에서 동작 수행 사례가 확인됐지만 1~10m·새 학습·RSI 안정성은 이 실험의 검증 대상이 아니다. 운영 설정은 유지하고, 평가 프로세스에서만 실제 asset 크기와 Climb 전용 높이 검사를 적용했다. 재현: `TOKENHSI_GPU=0 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; python output_etc/action_climb_height_eval_20261014/evaluate.py'`, 이후 `python output_etc/action_climb_height_eval_20261014/analyze.py`. 동일 경로로 재실행하면 기존 평가 산출물을 덮어쓴다.

2026-10-14 거리/RSI 변경 검증: CPU 67개, 2048환경 확인 학습 epoch 2·514 scalar finite·물리 reset 실패 0, 실제 초기 거리 765건·13개 과제/RSI·416 step·부분 reset·선반 접촉 통과. 신/구 config의 표준 test 각 16환경·32 step도 통과했다. [검증 결과](../output_etc/stage1_tokenhsi_sampling_check_20261014/validation.json). 확인 학습만 수행했으며 장기 성능 측정은 아니다.

2026-10-13 검증: **CPU 168개**, GPU 0·**2048환경** 확인 학습 epoch 2/frame 262144 및 checkpoint resume epoch 3/frame 393216 통과. 총 771 scalar finite·물리 reset 실패 0, actor/critic 공유 projection·네 task embedding·Action MLP의 optimizer 업데이트와 step 96→144를 확인했다. 네 과제 각각 전용 test wrapper의 **32환경·64-step headless 평가** 완료. 별도 시뮬레이터는 **13개 과제/RSI 조합·416 rollout step·319 reset 호출**에서 선반 범위/graph binding/부분 reset 및 CPA를 검증했고, source/target 판을 0.25m→0.75m로 재배치해 실접촉 지지·높이 오차 최대 1.28e-5m를 확인했다. 이는 실행 검증이며 장기 성능 평가가 아니다.

검증 자료: `output_etc/stage1_check_20261013/validation.json` 및 로그·평가 metrics. 확인 checkpoint는 `output/stage1_check/`, resume 확인은 `output/stage1_resume_check/`다. 본학습은 시작하지 않았다. VNC wrapper는 실제 호출했으나 현재 머신의 **NVML 580.178.04 / 로드된 드라이버 580.173.02 불일치**로 GPU 선택 단계에서 종료돼 화면 검증은 미완료다. Headless CUDA 경로는 정상이며 기존 학습/시스템 드라이버는 변경하지 않았다.

## Unified size RSI task embedding action CPA

`approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa`는 Action의 별도 scratch 실험이다. 모델·타입별 bias projection·task concat MLP·과제/성공 조건·AMP·RSI·토큰 셔플은 유지하고 사람 간 collision 보상만 변경한다. Env의 Action variant와 모델 구조는 같지만 collision checkpoint 계약으로 기존 static/CPA checkpoint의 직접 resume/eval 혼용을 거부한다. 기존 Action weight를 CPA에 이식하는 기능은 추가하지 않았다.

- Git `595688e`의 `collision_reward.py` 및 `approach_stage2_rescue_shared9_cpa.yaml`을 참고했다. `agentCollisionMode: cpa`, `agentCollisionTTCDiscount: 0.99`, 계수 0.5·기준 거리 0.7m다.
- 사람 root XY 상대 위치/속도로 등속 운동의 최근접 시각 `t=max(-p·v/|v|²,0)`과 거리를 계산한다. 접근 정렬도 × 최근접 거리 위험 × `0.99^(t/control_dt)`를 사람 쌍별로 계산하고 최댓값에 -0.5를 곱한다. 정지·동일 속도 평행 이동·멀어지는 경우는 0이며, 기존 거리 페널티를 중복해서 더하지 않는다. 사람–물체 CPA는 포함하지 않는다.
- Train YAML·train/test/VNC·output을 분리했다. 기존 설정에서 mode를 생략하면 static으로 유지된다. 과거 checkpoint는 저장된 env를 통해 static 계약을 복원한다.

2026-10-13 검증: CPU 37개 통과, GPU 0·2048환경 확인 학습 epoch 2 저장·514 scalar finite·물리 reset 실패 0. 저장본 16환경·32-step 평가에서 31회 collision reward와 독립 CPA 계산의 차이 0을 확인했다. 자료는 `output_etc/action_cpa_check_20261013/`, 확인 학습은 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa_check/`다. 본학습·장기 성능·VNC 화면 검증은 하지 않았다.

```bash
# 새 본학습 (2명·2048환경·4물체)
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa_train.sh 2 2048 4
# 짧은 확인은 별도 output
TOKENHSI_GPU=0 MAX_ITERATIONS=1 RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa_train.sh 2 2048 4
# CKPT에는 CPA 실험의 checkpoint 경로 지정
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa_vnc.sh "$CKPT" 2 1 4 10
# 같은 CPA 실험에서 재개
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT="$CKPT" OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa_train.sh 2 2048 4
```

## Unified size RSI task role gate entmax

2026-10-10 속도 최적화: 학습 시 gate 표 조회의 gradient 합산을 one-hot 행렬곱으로 변경하고, 16D 조회 뒤 4D block을 확장한다. No-grad/frozen/FP32 TF32 경로는 compact index lookup을 사용한다. 수식·weight key·optimizer·config와 Entmax는 유지하므로 **같은 실험의 기존 checkpoint로 resume 가능**하다. 현재 실행 중인 Python에는 파일 변경이 반영되지 않으므로 저장 checkpoint에서 다시 실행해야 적용된다. 저장 전 종료하면 아직 저장되지 않은 진행은 잃는다. 기존 본학습은 자동으로 중단하지 않는다.

GPU에서 같은 checkpoint·입력의 8192 scene actor+critic forward/backward는 170.0→87.3ms, 다른 교차 측정에서는 332.6→163.0ms로 약 49~51% 단축됐다. **전체 iteration 단축률이나 장기 성공률 개선을 뜻하지 않는다.** CPU 137개·CUDA 1개 및 기존 optimizer를 복원한 2048환경 epoch 2→3 학습을 확인했다. 자료는 `output_etc/task_role_gate_lookup_opt_20261010/`와 `output_etc/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax_lookup_opt_check/`에 있다.

`approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax`는 `task_embedding` 기반의 독립 scratch 실험이다. Env YAML은 `stage1_variant`만 변경하므로 기존 과제·보상·크기·RSI·AMP·634-D packet·edge/token 순열을 그대로 사용한다. Train YAML에서는 attention 설정만 변경한다. 별도 variant/fusion 계약으로 이전 checkpoint의 resume/eval을 거부한다.

- Actor/critic은 각각 `Embedding(4,64)` 하나를 네 layer에서 공유한다. `task64 + role one-hot3`의 12개 조합을 만들고, 각 layer의 독립 `67→64→16` SiLU MLP가 `1+tanh` gate를 생성한다. 2 heads × 8 blocks이며 각 block의 4개 차원에는 같은 스칼라를 곱한다. Task별·agent별 MLP는 없다.
- `relation_attention_mode=task_role_gate_entmax`, `relation_bias_mode=none`, `relation_bias=false`, `relation_message.enable=false`, `attention_normalizer=entmax15`를 함께 검증한다. 기존 bias encoder·relation message·사람 토큰 직접 주입 MLP는 생성하지 않는다. Builder의 전역 초기화 후 gate 출력층을 다시 0으로 만들어 초기 gate=1을 보존한다.
- GTA Q/K/V 적용 뒤 task edge의 K score와 V 집계만 gather/scatter로 보정하고, V를 합친 후 GTA 역변환을 한 번 수행한다. 토큰을 섞으면 src/dst도 역순열로 재매핑한다. Invalid packet은 embedding 조회와 gather 전에 안전한 index로 바꾸고 correction을 0으로 처리한다. Non-task·self·reverse는 gate=1이며 attention 후보로 남는다.
- Entmax는 full sort 기반 정확한 1.5 정규화와 해석적 backward를 사용한다. [원 논문](https://arxiv.org/abs/1905.05702)의 threshold/Jacobian을 독립 bisection oracle 및 finite-difference로 확인한다. Dense score/attention은 유지하므로 계산은 여전히 N² 규모다. 초기 gate가 1이어도 Softmax baseline과 출력은 다르다.
- RSI는 SIT loco/sit 50/50, CLIMB loco/climb 50/50, carry loco/pickUp/carryWith/putDown 40/10/40/10, 후반/전체 70/30을 유지한다. AMP family carry/sit/climb 65/10/25·행별 expert/replay 매칭·원래 캐시를 사용한다. 보상은 자기 primitive 합계 ×1, 동료 ×0, PPO task/AMP 0.5/0.5다.
- `relation_attention/{actor,critic}/layer{0..3}/` 아래 task attention mass·edge zero-rate·평균 nonzero support 수·gate delta RMS·과제별 zero-rate와 gate gradient norm을 기록한다. 각 task embedding 행의 gradient norm도 기록한다. Attention 통계는 진단 시점의 앞 32개 sample, gradient norm은 마지막 minibatch 기준이다. 기존 환경의 task 성공·collision 진단은 유지한다. 별도 distractor/근접 장애물 행동 평가와 장기 성능 비교는 수행하지 않았다.
- Gate 파라미터는 branch당 21,824개(embedding 256 + 4 MLP 21,568), 전체 모델은 4,164,290개다. 파라미터 수는 entity 개수와 무관하지만 현재 unified sampler는 2H/4O용이다. 4H/8O·4H/16O rollout 지원이나 Stage 2 연결을 뜻하지 않는다.

2026-10-10 검증: 관련 CPU **135개 통과**, GPU 0·2048환경·MAX_ITERATIONS=1 학습(epoch 2/frame 262144) 및 네 과제 각 16환경·32-step headless 평가 완료. Carry는 carryWith, ON_TOP은 role swap을 사용했다. RSI 캐시 hit·scalar 321종/642값 finite·물리 reset 실패 0·CSV binding/보상 합산 오류 0이며 양 branch의 모든 gate와 네 task embedding 행이 업데이트됐다. 검증 자료는 `output_etc/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax_check/verification.json`·`validation/`, 평가 결과는 네 `_check_<task>/metrics/`다. `_check`는 실행 경로 확인 전용이며 본학습·장기 성공률·VNC 화면은 검증하지 않았다.

```bash
# Scratch 본학습: 실행 전 nvidia-smi, 2명·2048환경·4물체
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax_train.sh 2 2048 4
# 짧은 확인: 본학습과 별도 output
TOKENHSI_GPU=0 MAX_ITERATIONS=1 RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax_train.sh 2 2048 4
# CKPT에는 이 실험에서 저장한 checkpoint 경로 지정
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=carry_at EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=0 TASK_GRAPH=carry_ontop EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax_vnc.sh "$CKPT" 2 1 4 10
# 같은 실험 checkpoint 재개
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT="$CKPT" OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax_train.sh 2 2048 4
```
