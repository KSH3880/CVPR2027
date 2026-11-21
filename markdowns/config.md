# 실행 가이드

모든 명령은 저장소 루트에서 실행한다.

## 공통 규칙

- 유지하는 실험은 아래 컨피그 표를 따른다. 설정과 실험별 train/test/VNC 진입점을 함께 관리한다.
- 기존 실행기는 기본 conda `tokenhsi_jhh`, 새 `stage2_sensor_900_bridge_follow_cap*` 실행기는 이 서버의 `tokenhsi`를 사용한다. 실행 전 `nvidia-smi`로 점유를 확인하고 `TOKENHSI_GPU`를 한 번 지정한다. 기존 프로세스는 보존한다.
- 이 checkout의 현재 학습은 `stage2_sensor_900_cliff_bridge_surface`이며 GPU0에서 실행한다. Run 이름은 `Stage2Sensor900CliffBridgeSurface`다. 기존 실험 명령은 유지한다. 학습 시각화 noVNC는 기본6080, 테스트 시각화는6081이다. 현재 서비스·Distill1500 새 전이는 [USAGE](USAGE.md#bridge-surface-본학습-서비스), 전이·동결·센서 옵션은 [USAGE](USAGE.md#stage2-sensor-900)를 따른다.
- 학습 인자: `[agents] [envs] [objects]`. 기본값은 Stage1·Stage1 PUSH **2명·2048환경·4물체**, BEFORE·coupled **4명·1024환경·8토큰**이다. BEFORE는 인원 N≥1·토큰 M≥2N, coupled는 인원 N≥1·토큰 M≥N을 지정할 수 있다.
- viewer/진단 인자: `<checkpoint.pth> [agents] [envs] [objects] [repeats]`. 추론과 resume은 checkpoint 옆 `relation_config.yaml`의 env/sim을 읽는다. `ENV_CONFIG`를 명시하면 해당 YAML을 우선하며 checkpoint 계약도 검사한다.
- 본학습은 `output/`, 확인 학습·평가·viewer는 **`output_etc/check/`**에 저장한다. 짧은 학습은 `CHECK=1`을 붙이며 학습 환경 수를 유지한다. 본학습 전 `CHECK`, `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 해제한다.
- 기존 BEFORE의 기본 전이 가중치는 `checkpoints/Stage1_00078000.pth`, PUSH 기반 Stage2 설정은 `checkpoints/Stage1PushFinetune_00021000.pth`, `stage2_coupled_push_goal5`·`stage2_coupled_push_individual_goal`·`stage2_coupled_push_individual_goal_solo5`·`stage2_coupled_push_individual_goal_solo5_carry5`·`stage2_coupled_push_individual_goal_solo5_countnorm_coupled`·`stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none`·`stage2_coupled_push_individual_goal_solo5_keep1`·`stage2_coupled_push_individual_goal_solo5_shrink1`·`stage2_coupled_push_individual_goal_solo5_carry5_ground`·`stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg`·`stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg`은 `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`다. 데이터·checkpoint 연결과 추가 옵션은 [USAGE](USAGE.md)를 따른다.
- 학습·평가·viewer 기본 제한 시간은 **Stage1 900step, Stage2 1200step**이며 `stage2_900`·`stage2_distill_900`·`stage2_sensor_900`·`stage2_sensor_900_bridge_init`·`stage2_sensor_900_bridge_curriculum`·`stage2_sensor_900_room_before`·`stage2_sensor_900_cliff_bridge`·`stage2_sensor_900_cliff_bridge_surface`·`stage2_sensor_900_cliff_bridge_compact`·`stage2_sensor_900_bridge_before_fixed`·`stage2_sensor_900_bridge_before_fixed_carry_rsi`·`stage2_sensor_900_bridge_before_fixed_layout_cache`·`stage2_sensor_900_bridge_follow_cap*`는 학습·평가·viewer 모두 **900step**이다.
- 정식 평가의 기존 과제/관계 지표와 추가 interaction group 성공률 정의는 [평가 규약](reference/evaluation.md#통합-stage2의-interaction-group-성공률)을 따른다.
- 모든 test의 기본값은 viewer다. 로컬 추론은 `HEADLESS=0`, 화면 없는 진단은 `HEADLESS=1`이다. **정식 평가는 `*_eval.sh`**, 1명 단독 평가는 **Stage1 계열 전용 `stage1_single_eval.sh`·`stage1_push_single_eval.sh`**로 구분한다. 일반 평가는 저장된 학습 인원·물체 수를 유지한다.

- `stage2`·`stage2_distill`·`stage2_900`·`stage2_distill_900` 학습은 4명·1024환경·8토큰이 기본이다. N≥4·M≥N을 지원하며 5명 이상은 전용 그룹 스폰 옵션을 사용한다. 두 teacher viewer와 확장 인자는 [USAGE](USAGE.md#두-teacher-제로샷-viewer)를 따른다. 전이는 `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`를 사용한다.

## 현재 컨피그

| 환경 Config | 학습 Config |
| --- | --- |
| [stage1.yaml](../tokenhsi/data/cfg/multi_agent/stage1.yaml) | [stage1.yaml](../tokenhsi/data/cfg/train/rlg/stage1.yaml) |
| [stage2_before_nonemask_rsi433_near50.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_nonemask_rsi433_near50.yaml) | [stage2_before_nonemask_rsi433_near50.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_nonemask_rsi433_near50.yaml) |
| [stage2_before.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before.yaml) | [stage2_before.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before.yaml) |
| [stage2_before_rsi433.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_rsi433.yaml) | [stage2_before_rsi433.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_rsi433.yaml) |
| [stage1_push.yaml](../tokenhsi/data/cfg/multi_agent/stage1_push.yaml) | [stage1_push.yaml](../tokenhsi/data/cfg/train/rlg/stage1_push.yaml) |
| [stage2_before_push.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push.yaml) | [stage2_before_push.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push.yaml) |
| [stage2_coupled_push.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push.yaml) | [stage2_coupled_push.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push.yaml) |
| [stage2_before_push_near50.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50.yaml) | [stage2_before_push_near50.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50.yaml) |
| [stage2_coupled_push_goal5.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_goal5.yaml) | [stage2_coupled_push_goal5.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_goal5.yaml) |
| [stage2_coupled_push_individual_goal.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal.yaml) | [stage2_coupled_push_individual_goal.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal.yaml) |
| [stage2_before_push_near50_pair50_source5.yaml](../tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5.yaml) | [stage2_before_push_near50_pair50_source5.yaml](../tokenhsi/data/cfg/train/rlg/stage2_before_push_near50_pair50_source5.yaml) |
| [stage2_coupled_push_individual_goal_solo5.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5.yaml) | [stage2_coupled_push_individual_goal_solo5.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5.yaml) |
| [stage2_coupled_push_individual_goal_solo5_carry5.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5.yaml) | [stage2_coupled_push_individual_goal_solo5_carry5.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_carry5.yaml) |
| [stage2_coupled_push_individual_goal_solo5_countnorm_coupled.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled.yaml) | [stage2_coupled_push_individual_goal_solo5_countnorm_coupled.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_countnorm_coupled.yaml) |
| [stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none.yaml) | [stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none.yaml) |
| [stage2_coupled_push_individual_goal_solo5_keep1.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_keep1.yaml) | [stage2_coupled_push_individual_goal_solo5_keep1.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_keep1.yaml) |
| [stage2_coupled_push_individual_goal_solo5_shrink1.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_shrink1.yaml) | [stage2_coupled_push_individual_goal_solo5_shrink1.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_shrink1.yaml) |
| [stage2_coupled_push_individual_goal_solo5_carry5_ground.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground.yaml) | [stage2_coupled_push_individual_goal_solo5_carry5_ground.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_carry5_ground.yaml) |
| [stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg.yaml) | [stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg.yaml) |
| [stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg.yaml](../tokenhsi/data/cfg/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg.yaml) | [stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg.yaml](../tokenhsi/data/cfg/train/rlg/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg.yaml) |
| [stage2.yaml](../tokenhsi/data/cfg/multi_agent/stage2.yaml) | [stage2.yaml](../tokenhsi/data/cfg/train/rlg/stage2.yaml) |
| [stage2_distill.yaml](../tokenhsi/data/cfg/multi_agent/stage2_distill.yaml) | [stage2_distill.yaml](../tokenhsi/data/cfg/train/rlg/stage2_distill.yaml) |
| [stage2_900.yaml](../tokenhsi/data/cfg/multi_agent/stage2_900.yaml) | [stage2_900.yaml](../tokenhsi/data/cfg/train/rlg/stage2_900.yaml) |
| [stage2_distill_900.yaml](../tokenhsi/data/cfg/multi_agent/stage2_distill_900.yaml) | [stage2_distill_900.yaml](../tokenhsi/data/cfg/train/rlg/stage2_distill_900.yaml) |
| [stage2_sensor_900.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900.yaml) | [stage2_sensor_900.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900.yaml) |
| [stage2_sensor_900_bridge_init.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_init.yaml) | [stage2_sensor_900_bridge_init.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_bridge_init.yaml) |
| [stage2_sensor_900_bridge_curriculum.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_curriculum.yaml) | [stage2_sensor_900_bridge_curriculum.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_bridge_curriculum.yaml) |
| [stage2_sensor_900_room_before.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_room_before.yaml) | [stage2_sensor_900_room_before.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_room_before.yaml) |
| [stage2_sensor_900_cliff_bridge.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_cliff_bridge.yaml) | [stage2_sensor_900_cliff_bridge.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_cliff_bridge.yaml) |
| [stage2_sensor_900_cliff_bridge_surface.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_cliff_bridge_surface.yaml) | [stage2_sensor_900_cliff_bridge_surface.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_cliff_bridge_surface.yaml) |
| [stage2_sensor_900_cliff_bridge_compact.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_cliff_bridge_compact.yaml) | [stage2_sensor_900_cliff_bridge_compact.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_cliff_bridge_compact.yaml) |
| [stage2_sensor_900_bridge_before_fixed.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_before_fixed.yaml) · [명세](reference/stage2_sensor_900_bridge_before_fixed.md) | [stage2_sensor_900_bridge_before_fixed.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_bridge_before_fixed.yaml) |
| [stage2_sensor_900_bridge_before_fixed_carry_rsi.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_before_fixed_carry_rsi.yaml) | [stage2_sensor_900_bridge_before_fixed_carry_rsi.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_bridge_before_fixed_carry_rsi.yaml) |
| [stage2_sensor_900_bridge_before_fixed_layout_cache.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_before_fixed_layout_cache.yaml) · [명세](reference/stage2_sensor_900_bridge_before_fixed_layout_cache.md) | [stage2_sensor_900_bridge_before_fixed_layout_cache.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_bridge_before_fixed_layout_cache.yaml) |
| [stage2_sensor_900_bridge_follow_cap.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_follow_cap.yaml) | [stage2_sensor_900_bridge_follow_cap.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_bridge_follow_cap.yaml) |
| [stage2_sensor_900_bridge_follow_cap_waypoint.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint.yaml) | [stage2_sensor_900_bridge_follow_cap_waypoint.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_bridge_follow_cap_waypoint.yaml) |
| [stage2_sensor_900_bridge_follow_cap_waypoint_progress.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_progress.yaml) | [stage2_sensor_900_bridge_follow_cap_waypoint_progress.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_bridge_follow_cap_waypoint_progress.yaml) |
| [stage2_sensor_900_bridge_follow_cap_waypoint_single_pass.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_single_pass.yaml) | [stage2_sensor_900_bridge_follow_cap_waypoint_single_pass.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_bridge_follow_cap_waypoint_single_pass.yaml) |
| [stage2_sensor_900_bridge_follow_cap_waypoint_progress_single_pass.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_progress_single_pass.yaml) | [stage2_sensor_900_bridge_follow_cap_waypoint_progress_single_pass.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_bridge_follow_cap_waypoint_progress_single_pass.yaml) |
| [stage2_sensor_900_bridge_follow_cap_rear90.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_follow_cap_rear90.yaml) | [stage2_sensor_900_bridge_follow_cap_rear90.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_bridge_follow_cap_rear90.yaml) |
| [stage2_sensor_900_bridge_follow_cap_rear90_random.yaml](../tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_follow_cap_rear90_random.yaml) | [stage2_sensor_900_bridge_follow_cap_rear90_random.yaml](../tokenhsi/data/cfg/train/rlg/stage2_sensor_900_bridge_follow_cap_rear90_random.yaml) |

## 학습

```bash
# stage1
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_train.sh 2 2048 4

# stage2_before_nonemask_rsi433_near50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_nonemask_rsi433_near50_train.sh 4 1024 8

# stage2_before
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_train.sh 4 1024 8

# stage2_before_rsi433
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_rsi433_train.sh 4 1024 8

# stage1_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_push_train.sh 2 2048 4

# stage2_before_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_train.sh 4 1024 8

# stage2_coupled_push
TOKENHSI_GPU=6 STAGE1_CHECKPOINT=pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth bash tokenhsi/scripts/multi_agent/stage2_coupled_push_train.sh 4 1024 8

# stage2_before_push_near50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_train.sh 4 1024 8

# stage2_coupled_push_goal5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_goal5_train.sh 4 1024 8

# stage2_coupled_push_individual_goal
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_train.sh 4 1024 8

# stage2_before_push_near50_pair50_source5
TOKENHSI_GPU=6 STAGE1_CHECKPOINT=pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_carry5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_countnorm_coupled
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none
TOKENHSI_GPU=3 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_keep1
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_keep1_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_shrink1
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_shrink1_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_carry5_ground
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg
TOKENHSI_GPU=2 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg_train.sh 4 1024 8

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg
TOKENHSI_GPU=2 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg_train.sh 4 1024 8

# stage2
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_train.sh 4 1024 8

# stage2_distill
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_distill_train.sh 4 1024 8

# stage2_900
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_900_train.sh 4 1024 8

# stage2_distill_900
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_distill_900_train.sh 4 1024 8

# stage2_distill_900 (학습 시각화, noVNC 6080)
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_distill_900_train_vnc.sh 4 1024 8

# stage2_sensor_900
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_train.sh 4 1024 8

# stage2_sensor_900 (학습 시각화, noVNC 6080)
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_train_vnc.sh 4 1024 8

# stage2_sensor_900_bridge_init
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_init_train.sh 4 1024 8

# stage2_sensor_900_bridge_init (학습 시각화, noVNC 6080)
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_init_train_vnc.sh 4 1024 8

# stage2_sensor_900_bridge_curriculum
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_curriculum_train.sh 4 1024 8

# stage2_sensor_900_bridge_curriculum (학습 시각화, noVNC 6080)
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_curriculum_train_vnc.sh 4 1024 8

# stage2_sensor_900_room_before
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_room_before_train.sh 4 1024 8

# stage2_sensor_900_cliff_bridge
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_train.sh 4 1024 8

# stage2_sensor_900_cliff_bridge_surface
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_surface_train.sh 4 1024 8

# stage2_sensor_900_cliff_bridge_compact
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_compact_train.sh 4 1024 8

# stage2_sensor_900_bridge_before_fixed
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_before_fixed_train.sh 4 1024 8

# stage2_sensor_900_bridge_before_fixed_carry_rsi
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_before_fixed_carry_rsi_train.sh 4 1024 8

# stage2_sensor_900_bridge_before_fixed_layout_cache
TOKENHSI_GPU=5 ENV_CONFIG=tokenhsi/data/cfg/multi_agent/stage2_sensor_900_bridge_before_fixed_layout_cache.yaml bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_before_fixed_layout_cache_train.sh 4 1024 8

# stage2_sensor_900_bridge_follow_cap
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_train.sh

# stage2_sensor_900_bridge_follow_cap_waypoint
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_train.sh

# stage2_sensor_900_bridge_follow_cap_waypoint_progress
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_progress_train.sh

# stage2_sensor_900_bridge_follow_cap_waypoint_single_pass
TOKENHSI_GPU=7 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_single_pass_train.sh

# stage2_sensor_900_bridge_follow_cap_waypoint_progress_single_pass
TOKENHSI_GPU=7 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_progress_single_pass_train.sh

# stage2_sensor_900_bridge_follow_cap_rear90
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_rear90_train.sh 4 1024 8

# stage2_sensor_900_bridge_follow_cap_rear90_random
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_rear90_random_train.sh 4 1024 8
```

## 로컬 추론

```bash
CKPT='/absolute/path/to/checkpoint.pth'

# stage1
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_test.sh "$CKPT" 2 1 4 10

# stage2_before_nonemask_rsi433_near50
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_nonemask_rsi433_near50_test.sh "$CKPT" 4 1 8 10

# stage2_before
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_test.sh "$CKPT" 4 1 8 10

# stage2_before_rsi433
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_rsi433_test.sh "$CKPT" 4 1 8 10

# stage1_push
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_push_test.sh checkpoints/Stage1PushFinetune_00021000.pth 2 1 4 10

# stage2_before_push
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_goal5
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_goal5_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_test.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_countnorm_coupled
TOKENHSI_GPU=0 HEADLESS=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none
TOKENHSI_GPU=3 HEADLESS=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_keep1
TOKENHSI_GPU=5 HEADLESS=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_keep1_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_shrink1
TOKENHSI_GPU=6 HEADLESS=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_shrink1_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5_ground
TOKENHSI_GPU=6 HEADLESS=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg
TOKENHSI_GPU=2 HEADLESS=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg_test.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg
TOKENHSI_GPU=2 HEADLESS=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg_test.sh "$CKPT" 4 1 8 10

# stage2
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_test.sh "$CKPT" 4 1 8 10

# stage2_distill
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_distill_test.sh "$CKPT" 4 1 8 10

# stage2_900
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_900_test.sh "$CKPT" 4 1 8 10

# stage2_distill_900
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_distill_900_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_init
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_init_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_curriculum
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_curriculum_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_room_before
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_room_before_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_cliff_bridge
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_cliff_bridge_surface
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_surface_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_cliff_bridge_compact
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_compact_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_before_fixed
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_before_fixed_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_before_fixed_carry_rsi
TOKENHSI_GPU=0 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_before_fixed_carry_rsi_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_before_fixed_layout_cache
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_before_fixed_layout_cache_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_follow_cap
HEADLESS=0 TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_follow_cap_waypoint
HEADLESS=0 TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_follow_cap_waypoint_progress
HEADLESS=0 TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_progress_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_follow_cap_waypoint_single_pass
TOKENHSI_GPU=7 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_single_pass_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_follow_cap_waypoint_progress_single_pass
TOKENHSI_GPU=7 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_progress_single_pass_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_follow_cap_rear90
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_rear90_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_follow_cap_rear90_random
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_rear90_random_test.sh "$CKPT" 4 1 8 10
```

## 서버 VNC 추론

```bash
CKPT='/absolute/path/to/checkpoint.pth'

# stage1
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_vnc.sh "$CKPT" 2 1 4 10

# stage2_before_nonemask_rsi433_near50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_nonemask_rsi433_near50_vnc.sh "$CKPT" 4 1 8 10

# stage2_before
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_rsi433
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_rsi433_vnc.sh "$CKPT" 4 1 8 10

# stage1_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_push_vnc.sh checkpoints/Stage1PushFinetune_00021000.pth 2 1 4 10

# stage2_before_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_goal5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_goal5_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_vnc.sh "$CKPT" 4 1 8 10

# stage2_before_push_near50_pair50_source5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_countnorm_coupled
TOKENHSI_GPU=0 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none
TOKENHSI_GPU=3 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_keep1
TOKENHSI_GPU=5 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_keep1_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_shrink1
TOKENHSI_GPU=6 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_shrink1_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5_ground
TOKENHSI_GPU=6 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg
TOKENHSI_GPU=2 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg
TOKENHSI_GPU=2 COUPLED_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg_vnc.sh "$CKPT" 4 1 8 10

# stage2
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_vnc.sh "$CKPT" 4 1 8 10

# stage2_distill
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_distill_vnc.sh "$CKPT" 4 1 8 10

# stage2_900
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_900_vnc.sh "$CKPT" 4 1 8 10

# stage2_distill_900
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_distill_900_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_init
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_init_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_curriculum
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_curriculum_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_room_before
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_room_before_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_cliff_bridge
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_cliff_bridge_surface
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_surface_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_cliff_bridge_compact
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_compact_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_before_fixed
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_before_fixed_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_before_fixed_carry_rsi
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_before_fixed_carry_rsi_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_before_fixed_layout_cache
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_before_fixed_layout_cache_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_follow_cap
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_follow_cap_waypoint
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_follow_cap_waypoint_progress
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_progress_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_follow_cap_waypoint_single_pass
TOKENHSI_GPU=7 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_single_pass_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_follow_cap_waypoint_progress_single_pass
TOKENHSI_GPU=7 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_waypoint_progress_single_pass_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_follow_cap_rear90
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_rear90_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_bridge_follow_cap_rear90_random
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_rear90_random_vnc.sh "$CKPT" 4 1 8 10
```

## 추론 옵션

- `TASK_GRAPH=random_scenario`가 기본이다. Stage1: `sit`, `climb`, `holding_at`, `holding_ontop`, `sit_climb`. BEFORE: `before_sit`, `before_climb`, `before_stack`, `one_pair_sit`, `one_pair_climb`, `one_pair_stack`, `all_solo`, `sit`, `climb`, `holding_at`, `holding_ontop`.
- `EVAL_SKILLS`와 `EVAL_SKILL_PROBS`는 함께 지정한다. 예: `EVAL_SKILLS=loco EVAL_SKILL_PROBS=1.0`. BEFORE의 기본 초기화는 loco이며 후행 RSI는 별도 옵션으로 선택한다.
- `TASK_CAMERA=stack` 또는 `agent`로 viewer 카메라를 선택한다. 그 외 초기화·거리·resume·실행 환경 옵션은 [USAGE](USAGE.md)를 따른다.
- DistillV4 원본의 직선 횡단·선행 GTA 중간 목표 viewer는 `--bridge-physics baseline|fixed|carry|fixed_carry|platform|baseline_nocap|carry_nocap|follow_cap`, `--bridge-start present|near|insert`(기본 present), `--cap-lift 0.02`(follow_cap 전용, m 단위·기본 0)를 지원한다. [제로샷 실행법](USAGE.md#distillv4-bridge-중간-목표-제로샷-viewer)을 따르며 기존 학습 config를 변경하지 않는 추론 전용 실행기다.
- 위 viewer의 `--no-waypoint`는 다리 중앙 가상목표·반경 전환을 끄고 처음부터 원래 목표를 사용한다. 이때 `--switch-margin`은 사용하지 않으며 물성·캡 높이·시작 배치는 유지한다.
- coupled test/VNC의 `COUPLED_ASSIGNMENT=random|ordered|distance|optimal`(기본 random)은 공동 사람↔구역 배정이다. ordered는 좌우 정렬, distance는 총 접근 거리 최소화, optimal은 교차 우선이다. 직접 실행은 `--coupled_assignment`; 학습/resume·기본 정식 평가의 비random 변경은 받지 않는다. 통합 Stage2 네 컨피그의 viewer/eval은 ordered가 기본이며 인원·토큰 확장 평가도 지원한다. [계산 방식·인원별 탐색](USAGE.md#공통-stage2-push와-coupled).
- 개인 목표 coupled test/VNC의 `COUPLED_PEERS=all|keep3|shrink3`(기본 all)은 5명 이상 공동 그룹의 사람 attention 상대 선택이다. keep3은 미접촉 상대 우선 최대3명을 유지하고, shrink3은 접촉한 상대를 빼 전원 접촉 시0명으로 줄인다. 직접 실행은 `--coupled_peers`; [접촉·갱신 규칙](USAGE.md#공통-stage2-push와-coupled).
- 이후 사용자 공동 PUSH 평가는 ordered를 사용하며, 현재는 전용 비교 실행기로 채점한다. [ordered 평가 실행법](USAGE.md#공통-stage2-push와-coupled).

```bash
# stage2_coupled_push_individual_goal_solo5: ordered를 distance 또는 optimal로 교체
TOKENHSI_GPU=6 COUPLED_ASSIGNMENT=ordered TASK_GRAPH=all_coupled bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_vnc.sh "$CKPT" 4 1 8 10

# stage2_coupled_push_individual_goal_solo5: 미접촉 우선3명 유지
TOKENHSI_GPU=5 PORT=6080 COUPLED_ASSIGNMENT=ordered COUPLED_PEERS=keep3 TASK_GRAPH=all_coupled bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_vnc.sh "$CKPT" 8 1 8 10

# stage2_coupled_push_individual_goal_solo5: 접촉 완료 상대 제외 → 전원 접촉 시0명
TOKENHSI_GPU=5 PORT=6080 COUPLED_ASSIGNMENT=ordered COUPLED_PEERS=shrink3 TASK_GRAPH=all_coupled bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_vnc.sh "$CKPT" 8 1 8 10
```

- 11000 PTH의 감량/액션 샘플링 비교 viewer는 아래 명령을 사용한다. 첫 인자8을4/6으로 바꾸면4/6명이고 물체는 자동2N이다. `mass` 모드에서 `MASS_SCALE=0.8`은20% 감량, `0.6`은40% 감량이다. `baseline`·`mass80`·`mass60`·`noise`도 선택할 수 있다. ordered·1환경·seed0·씬당1200step이고 마지막 인자가 순차 재생 횟수(생략 시10)다. `SEED`로 시작 장면을 바꾼다. 상세는 [USAGE](USAGE.md#공통-stage2-push와-coupled)를 따른다.

```bash
# stage2_coupled_push_individual_goal_solo5: 8명 감량20%
TOKENHSI_GPU=6 PORT=6086 HEADLESS=0 MASS_SCALE=0.8 bash output_etc/check/solo5_mass_noise_n68_20261101/view.sh 8 mass 10

# stage2_coupled_push_individual_goal_solo5: 8명 원래 질량+액션 노이즈
TOKENHSI_GPU=5 PORT=6085 HEADLESS=0 bash output_etc/check/solo5_mass_noise_n68_20261101/view.sh 8 noise 10

# stage2_coupled_push_individual_goal_solo5: 8명 감량40%
TOKENHSI_GPU=6 PORT=6084 HEADLESS=0 MASS_SCALE=0.6 bash output_etc/check/solo5_mass_noise_n68_20261101/view.sh 8 mass 10
```

- 정식 평가는 loco·Stage1 900step/Stage2 1200step·512환경×seed0/1/2이며 점/영역 ever를 함께 보고한다. 인원/물체/거리 override를 받지 않는다. `--envs`·`--seeds`를 줄인 실행은 검사 전용으로 표시한다. [평가 규약](reference/evaluation.md), [Stage1 단독](../evaluation_single.md).

```bash
CKPT=checkpoints/Stage1_00078000.pth

# stage1
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_eval.sh "$CKPT" --output output_etc/check/stage1_general

# stage2_before_nonemask_rsi433_near50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_nonemask_rsi433_near50_eval.sh /absolute/path/to/near50.pth --output output_etc/check/stage2_before_nonemask_rsi433_near50_general

# stage2_before
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_eval.sh /absolute/path/to/before.pth --output output_etc/check/stage2_before_general

# stage2_before_rsi433
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_rsi433_eval.sh /absolute/path/to/rsi433.pth --output output_etc/check/stage2_before_rsi433_general

# stage1 전용 1명 단독 비교
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_single_eval.sh "$CKPT" --output output_etc/check/stage1_single
```

```bash
# stage1_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_push_eval.sh checkpoints/Stage1PushFinetune_00021000.pth --output output_etc/check/stage1_push_general

# stage1_push 전용 1명 단독 비교
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_push_single_eval.sh checkpoints/Stage1PushFinetune_00021000.pth --output output_etc/check/stage1_push_single
```

```bash
# stage2_before_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_eval.sh "$CKPT" --output output_etc/check/stage2_before_push_general
```

- `stage2_before_push`도 일반 평가는 학습 인원을 유지한다. `TASK_GRAPH=push`는 모든 사람을 독립 PUSH로 고정하는 viewer 옵션이다. [BEFORE PUSH](reference/stage2_before_push.md).

- `stage1_push`는 체크포인트 없이 스크래치 학습한다. 추론은 PTH 내부 저장 환경도 읽으며 일반 평가는 해당 학습 분포를 유지한다. `TASK_GRAPH=push`로 PUSH를 관찰한다. 설정 상세는 [PUSH](reference/stage1_push.md)를 따른다.

```bash
# stage2_coupled_push
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_general
```

- `stage2_coupled_push` 계열은 인원 상한 없이 N≥1·토큰 M≥N을 지원한다. `TASK_GRAPH=all_coupled`는 지정 인원 전원 공동 PUSH, `all_solo`는 독립 행동 혼합, `push`는 전원 독립 PUSH다. 물체가 부족한 임의 과제 조합은 다시 샘플링한다. 전원 ON_TOP 고정인 `holding_ontop`은 M≥2N이어야 한다. [공통 Stage2 PUSH 설정](reference/stage2_coupled_push.md).

```bash
# stage2_before_push_near50
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_eval.sh "$CKPT" --output output_etc/check/stage2_before_push_near50_general
```

```bash
# stage2_coupled_push_goal5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_goal5_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_goal5_general
```

```bash
# stage2_coupled_push_individual_goal
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_individual_goal_general
```

```bash
# stage2_coupled_push_individual_goal_solo5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_individual_goal_solo5_general
```

```bash
# stage2_before_push_near50_pair50_source5
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_pair50_source5_eval.sh "$CKPT" --output output_etc/check/stage2_before_push_near50_pair50_source5_general
```

- 개인 목표 solo5의 임의 인원·물체 수 관찰 예시다. N은 사람 수, M은 물체 토큰 수(M≥N)다. 혼합 장면은 `TASK_GRAPH=random_scenario`로 바꾼다. 로컬 viewer는 `_vnc.sh`를 `_test.sh`로 바꾸고 `HEADLESS=0`을 지정한다.

```bash
N=10
M=10

# stage2_coupled_push_individual_goal_solo5
TOKENHSI_GPU=5 PORT=6085 TASK_GRAPH=all_coupled bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_vnc.sh "$CKPT" "$N" 1 "$M" 10
```

```bash
# stage2_coupled_push_individual_goal_solo5_carry5
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_individual_goal_solo5_carry5_general
```

```bash
# stage2_coupled_push_individual_goal_solo5_carry5_ground
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_individual_goal_solo5_carry5_ground_general

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg
TOKENHSI_GPU=2 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg_general

# stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg
TOKENHSI_GPU=2 bash tokenhsi/scripts/multi_agent/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg_eval.sh "$CKPT" --output output_etc/check/stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg_general
```

- `stage2`의 `TASK_GRAPH`는 `random_scenario`, `before`, `coupled`, `mixed`, `all_solo`, `all_coupled`, `before_sit`, `before_climb`, `before_stack` 및 기존 단독 과제 preset을 지원한다.

- 기존 `stage2_sensor_900_bridge_follow_cap*` test/VNC의 담당 영역 기본값은 학습과 같은 `COUPLED_ASSIGNMENT=random`이다. 새 `_rear90`은 배치 후 `ordered`, `_rear90_random`은 배치 후 무작위 담당 배정을 학습·평가·VNC·캐시 복원에 공통 적용한다.
- Follow-cap `view_eval.sh`는 최종 배치 후 상자 좌표계 기준 `ordered` 배정을 사용한다(기존 학습 배정과 다름).
- Follow-cap eval VNC의 `--pusher-gap <m>`는 두 사람 공통 거리, `--pusher-gaps <m1> <m2>`는 밀기 담당 에이전트 번호순 개별 거리다. [인자와 명령](USAGE.md#follow-cap-5cm-학습-비교).
- Follow-cap 3500·4500·5000 checkpoint의 insert·횡단 전용 512환경×3seed 평가는 [추가 실행법](USAGE.md#follow-cap-5cm-학습-비교)을 따른다.
- 학습 모델의 선행·상자·목표 위치와 공동밀기 삽입 거리를 지정하는 `bridge_wait_vnc.sh`는 [위치 지정 viewer](USAGE.md#학습-모델의-위치-지정-viewer)를 따른다.
- Follow-cap의 학습 혼합 배치·캐시·확률적 행동을 재현하는 명령은 [학습 규칙 viewer](USAGE.md#학습-규칙-그대로-보는-viewer)를 따른다.

- `_rear90` 학습 배치의 원본 V4 2k 제로샷 VNC는 전용 `_rear90_zeroshot_vnc.sh`를 사용한다. [명령·모드 선택](USAGE.md#follow-cap-5cm-학습-비교).

- 두 follow-cap 6k의 최종 배치 후 좌우 정렬 평가는 [USAGE](USAGE.md#follow-cap-5cm-학습-비교)의 `bridge_crossing_eval_6000_ordered_20261118` 실행기를 사용한다.

- 기존 학습 배정 viewer는 `view_training.sh baseline 42 --bridge-start insert --pusher-gap 3.0`으로 모드와 공동밀기 사람의 뒷면 거리를 선택한다. [학습 규칙 viewer](USAGE.md#학습-규칙-그대로-보는-viewer).

- 가상목표 없는 follow-cap 6k의 기존 학습 담당 배정 평가는 [USAGE](USAGE.md#follow-cap-5cm-학습-비교)의 `bridge_crossing_eval_6000_training_assignment_20261118` 실행기를 사용한다.

- rear90·rear90_random의 동일 1k 정렬 평가는 [USAGE](USAGE.md#follow-cap-5cm-학습-비교)의 `bridge_rear90_eval_1000_ordered_20261118` 실행기를 사용한다.

- rear90·rear90_random test/VNC는 `PUSHER_ASSIGNMENT=ordered|random`으로 최종 담당 배정을 선택한다. 생략하면 checkpoint config를 따른다. 직접 실행은 `--pusher_assignment`이며 학습/resume에는 적용하지 않는다. [명령](USAGE.md#follow-cap-5cm-학습-비교).
- rear90 두 test/VNC에서 담당별 초기 밀기 거리는 다섯 위치 인자 뒤 `--pusher-gaps 1.5 3.0`으로 지정한다. 거리 기준·대응 순서·실행 예시는 [USAGE](USAGE.md#follow-cap-5cm-학습-비교)를 따른다.

- Follow-cap의 공동밀기 인원·선행/후행 쌍 수·보행로 폭을 조절하는 추론 전용 viewer는 [USAGE](USAGE.md#follow-cap-5cm-학습-비교)를 따른다.
