# Bridge 고정 신체 높이·상면 마찰 variant

`stage2_sensor_900_cliff_bridge_surface`는 기존 `stage2_sensor_900_cliff_bridge`의 샘플링·보상·모델·RSI를 유지하고 아래 두 환경 옵션만 추가한다. 실행 명령은 [config](../config.md), 검증 결과는 [changelog](../../changelog.md)를 따른다.

```yaml
sensorScene:
  bridge:
    humanoid_height_reference: fixed_start_surface
    top_surface:
      thickness: 0.005
      friction: 1.0
```

Bridge에서 `_start_surface > FLOOR`인 사람은 환경 원점Z+해당 환경의 무대 높이를 신체 위치에서 뺀다. 바닥에서 시작한 사람은 환경 원점Z만 뺀다. 무대 높이는 다리Z크기+`top_drop`(현재1cm)이며, 순간 골반XY가 틈 위에 놓여도 바뀌지 않는다. 실제 신체 위치는 그대로 두고 관측의 기준만 바꾸며, 앉기/오르기/낙상의 실제 높이 변화는 유지한다. 부분 reset도 같은 env-id 집합으로 계산한다. Cliff/Room은 기존 관측을 유지한다.

높이맵과 낙상 판정의 `_terrain_height()`는 실제 지형을 계속 사용한다. 골반 바로 아래 틈을 지날 때 인간 토큰의 높이만 급변하는 문제를 분리해서 수정한다. 옵션이 없는 구형 설정은 기존 계산을 사용한다.

다리는 원래 높이 안의 윗5mm를 별도 충돌 형상으로 나눈다. 시각 메시 하나·강체 하나·충돌 형상 둘이며 전체 외형·질량·COM·관성을 유지한다. 상면 마찰1.0·본체/바닥0.4·반발0을 actor 생성 때 설정한다. 일반 PUSH 상자는 변경하지 않는다. 외력 다리는 같은 자산 생성기를 사용하되 밀도500·중력 해제·회전 제한·제어 규칙을 유지한다. URDF와 메모리 캐시는 일반/외력·크기·밀도·판 두께를 구분한다.

`bridge_surface.py`는 평가 전용5mm 판 구현의 자산 부분만 옮긴 것이다. 평가의5명 역할 강제·1.5m 높이 고정·추론용 spawn/heading은 옮기지 않았다. 학습 다리 높이는 환경별1.45–1.70m의 기존 profile을 유지한다.

입력 차원·entity 수·actor encoder/coordination/sigma 동결·입력 정규화 동결·critic 전체와 value 정규화 학습은 기존과 같다. Cliff40%/Bridge60%, Bridge 내부40/30/30,4명·1024환경·8토큰·900step, Distill1500 새 전이를 유지한다. 새 출력명은 `Stage2Sensor900CliffBridgeSurface`다.

두 옵션은 기존 `coupled_push_contract.sensorScene`과 `relation_config.yaml`에 저장된다. 구형/신형 checkpoint를 서로 다른 variant로 추론하거나 resume하면 거부한다. 구형10K 학습을 새 물리로 자동 변환하지 않으며, 신규 variant checkpoint끼리의 resume만 지원한다.

## 검증 명령

CPU 테스트는 Isaac Gym을 torch보다 먼저 import한다.

```bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p output_etc/check/bridge_surface_20261112
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tokenhsi python -c 'import isaacgym, pytest; raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "--basetemp=output_etc/check/bridge_surface_20261112/pytest_tmp", "tokenhsi/tests/test_bridge_surface.py", "tokenhsi/tests/test_cliff_bridge.py", "tokenhsi/tests/test_sensor_stage2.py"]))'
python tokenhsi/tests/check_bridge_surface_material.py
```

다음 GPU 검사와1024환경 확인 학습은 각각 최소8GiB/32GiB 여유가 있을 때만 실행한다. GPU 검사는4명·8환경의 추론 검사이며 학습 환경 수 축소가 아니다. 명시적인 학습 교체 요청 없이는 기존 학습을 중단해 메모리를 확보하지 않는다.

```bash
TOKENHSI_GPU=0 python tokenhsi/tests/check_cliff_bridge.py --task HumanoidMACoupledPush \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage2_sensor_900_cliff_bridge_surface.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage2_sensor_900_cliff_bridge_surface.yaml --test --eval \
  --checkpoint output/stage2/Stage2Distill900_00001500.pth \
  --motion_file tokenhsi/data/dataset_stage1_jointpush.yaml --headless --num_agents 4 --num_envs 8 --num_objects 8 \
  --output_path output_etc/check/bridge_surface_20261112/runtime

TOKENHSI_GPU=0 CHECK=1 OUTPUT_PATH=output_etc/check/bridge_surface_20261112/ppo \
  bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_surface_train.sh 4 1024 8
```

`CHECK=1`은 기존 루프의 종료 조건 때문에 최대2update를 수행한다. 검사에서 생성한 초기화 전용/짧은 학습 checkpoint를 본학습 시작점으로 사용하지 않는다. CPU 물리 검사는 실제 GPU 재질·정책 성능 검증과 구분한다.

2026-11-12 후속 사용자 요청으로 기존 학습만 안전 종료하고1024환경2update 검증을 완료했다. 새 Distill1500 본학습의 서비스·로그·종료 명령은 [USAGE](../USAGE.md#bridge-surface-본학습-서비스), 실제 검증은 [운영 기록](../../output_etc/check/bridge_surface_launch_20261112-160350/report.md)을 따른다.
