# 직육면체 PUSH·Stage1 native RSI

이 설정은 Stage1의 SIT/CARRY RSI 방식으로 크기별 유효 프레임 풀을 사용한다. 기존의 3개 고정 자세 제한을 없애고 PUSH AMP의 **21개 원본 모션**에서 크기별로 살아남은 모든 프레임을 고른다. 원본 모션의 관절/손 높이를 바꾸지 않는다.

| 설정 | 값 |
| --- | --- |
| 상자 | X/Y 각각1.20~1.50m·5cm 격자 독립 추출, Z 1.30m 고정, 각 상자25kg, friction0.6 |
| 목표 | 초기 상자 뒤 중앙선의 바닥 기준0.65m, 상자에 붙는 로컬점 `(0,-Y/2,0.65-Z/2)`; 양손 중점으로 보상 |
| PUSH 목표 이동 | 초기 면 법선 직선 방향1~5m, 고정 목표 |
| CONTACT | 양손 각각 뒷면≤18cm, 중점 XY≤18cm·Z±3cm; 손/상자 힘0.5N gate 없음 |
| AT·AMP | 가져온 Stage1 AT/보상/네트워크 유지; PUSH AMP 원본21개 및 expert 구성 유지 |
| RSI motion 분포 | MotionLib의 원래 motion weight로21개 중 선택; clip별 통과/검사 후보 수로 가중치를 재분배하지 않음 |
| RSI 시간 | 선택 모션의 SIT/CARRY 크기별 cache와 같은 유효 프레임 풀/기하+짧은 물리 선별. 30%는 유효 프레임 전체에서, 70%는 유효 프레임 중 마지막30%에서 균등 추출 |
| 관절/속도 | 선택한 원본 frame의 전체 humanoid DOF position/velocity를 그대로 사용; 손 높이 retarget 없음. RSI scene 배치에 필요한 root translation/yaw는 적용 |
| 학습 기울기 제한 | 새 tilt reward/종료 제한 없음. 사전 프레임 필터도 2° 기준을 사용하지 않음 |

후반30%는 모션의 시간 퍼센트가 아니라, 같은 물리 검사에서 유효한 **프레임 pool**의 뒤쪽30%다. 크기 profile은 X/Y의 7×7 조합, Z=1.3m 고정으로 총49개이며, 21개 PUSH 모션의 프레임 cache는 현재 물성 기준으로 다시 생성해야 한다. 25kg/49 profile cache를 새로 만들고 YAML checksum을 갱신하려면 GPU6에서 아래를 실행한다. 기존 cache는 20kg·XYZ 가변 검사 결과라 새 물리 계약에 재사용하지 않는다.

```bash
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_stage1_push_rect_native_rsi_cache.sh
```

전신 겹침/바닥·root/box 물리 안정성은 기존 SIT/CARRY의 0.1초 cache 조건을 사용한다. 목표 손점에 원본 손이 항상 맞는다는 필터는 추가하지 않아, 학습이 원본 pose에서 손 중점 높이로 팔을 조정해야 한다.

빠른 실행 준비 검증 명령은 사본 루트에서 실행한다. 64환경 reset·12 control step이며 본학습을 시작하지 않는다.

```bash
TOKENHSI_GPU=6 bash -c '
source tokenhsi/scripts/multi_agent/runtime_env.sh
export PYTHONPATH="$TOKENHSI_ROOT/tokenhsi:$TOKENHSI_ROOT" OMP_NUM_THREADS=1
python tokenhsi/tests/check_stage1_push.py \
  --task HumanoidMAStage1Push \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage1_push_rect_native_rsi.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage1_push_rect_native_rsi.yaml \
  --motion_file tokenhsi/data/dataset_stage1_push.yaml \
  --num_agents 2 --num_envs 64 --num_objects 4 \
  --task_graph random_scenario --headless --no_video \
  --output_path output_etc/check/stage1_push_rect_native_rsi_reset
'
```

실행 전용 `stage1_push_rect_native_rsi_{train,test,vnc}.sh`는 [config](../config.md)의 마지막에 등록돼 있다. RSI cache는 `data/stage1_push_rect_native_rsi_cache.npz`에 저장되며 YAML의 SHA256와 모션 hash를 검증하고, 파일이 달라지면 checkpoint/config를 거부한다. 출력 cache build 감사와 reset sampling 감사는 각각 `output_etc/check/stage1_push_rect_native_rsi_build/` 및 `..._reset/`에 있다.
