# Fixed BEFORE Bridge 배치 캐시

`stage2_sensor_900_bridge_before_fixed_layout_cache`는 기존 fixed 실험의 보상·역할·Distill2000 전이를 유지하면서, 최종 거리 범위에서 무작위 생성·검사를 마친 초기 상태를 재사용한다. 실행 명령은 [config](../config.md)에 있다.

새 학습의 기본 초기화는 fixed와 같은 `stateInit: Default`(RSI 없음)다. 캐시에도 기본 자세와 현재 자세를 반복한 AMP 이력을 저장하며 AMP 전문가 학습은 유지한다. 거리 mixture는 `[0,0,1]`로 모든 update에서 alpha=1이다. 최종 거리 범위 안에서 무작위 배치하며, 각 객체를 정확한 최대 거리에 두는 설정은 아니다.

## 저장과 복원

- 물체와 무대 자산의 크기가 환경마다 다르므로 캐시는 같은 환경 안에서만 사용한다.
- 같은 무대/crossing 2종 × 시작 모드 3종 × 후행 과제 SIT/CLIMB/ON_TOP 3종의 18개 조합에 두 개씩 저장한다. 두 무대 구성의 캐시는 섞어 쓰지 않는다.
- 처음 reset하기 전에 alpha=1로 18개 조합을 2회씩 생성한다. 기존 배치 검사·재시도를 모두 통과한 환경당 36개 상태를 GPU 메모리에 보관한다. 준비 중 통계는 학습 통계에서 제외한다. 거리별 빈 공간을 없애 이전 108개 슬롯 대비 할당량은 1/3이다.
- 사람/상자/무대 actor 상태, DOF, 사람의 기구학적 body 상태, 목표, 역할, 관계 그래프, 공간 메타데이터, AMP 현재 프레임과 과거 프레임을 함께 저장한다.
- Default 초기화에서는 전체 배치·발 높이 보정이 끝난 뒤 비활성 후행 Target을 자기 root로 정규화한다. 초기 사람 좌표와 scene origin을 섞어 Target이 수 km 벗어나는 오류를 막는다. 교정된 Target을 캐시에 저장하며 실제 AT/PUSH 목표·사람·상자·RSI 초기화는 유지한다.
- reset마다 같은 출발 무대 20%/crossing 80%, 시작 모드 0.2/0.4/0.4, 후행 과제 균등을 새로 뽑고 해당 조합의 두 캐시 중 하나를 무작위 선택한다. 역할·위치의 연속 분포는 유한한 캐시 표본으로 근사한다. crossing 선행 전용 다리 중심 보상은 [fixed 명세](stage2_sensor_900_bridge_before_fixed.md)를 따른다.
- `refresh_probability: 0`이므로 준비 후에는 새 배치 생성·검사를 실행하지 않는다. 모든 슬롯의 alpha=1이 보장되어 reset은 miss 조회 없이 캐시를 직접 선택·복원한다.
- 이전 run의 저장 설정처럼 연속 거리 mixture를 사용하는 경우에는 기존 거리별 캐시·정확한 alpha 검사·miss 생성 경로를 유지한다. 캐시 없는 fixed와 carry-RSI 실험의 거리 설정은 변경하지 않았다.
- 복원 후 simulator에 상태를 반영하고 에피소드 시간, 종료 플래그, 성공·보상·BEFORE 대기 이력과 다리 중심 통과 상태를 새로 초기화한다. PPO rollout·optimizer 상태는 저장하지 않는다.

## 범위와 검증

새 env/train YAML과 train/test/VNC/train-VNC/eval wrapper를 사용한다. 캐시 준비·재사용은 학습에서만 적용하며 평가와 viewer는 기존 fixed 샘플링을 사용한다. 캐시는 checkpoint나 디스크에 저장하지 않으며 새 실행·resume에서 다시 준비한다. 기존 fixed checkpoint와 새 실험의 설정 계약은 구분된다.

변경 전 연속 거리/1% refresh checkpoint도 저장 설정으로 평가·재개할 수 있다. 실험 종류를 검사할 때 거리·캐시 sampler의 지원되는 차이는 허용하지만, 실제 checkpoint와 실행 환경의 계약은 별도로 엄격하게 비교한다. 기존 checkpoint에 새 native 설정을 강제로 지정하는 재개는 거부한다. wrapper의 재개 명령은 [USAGE](../USAGE.md#설정과-resume)를 따른다. 기존 실행 프로세스에는 코드·YAML 변경이 소급 적용되지 않는다.

구현은 `humanoid_ma_sensor_bridge_layout_cache.py`, 저장소는 `utils/episode_layout_cache.py`다. CPU 테스트는 `test_episode_layout_cache.py`, 실제 1024환경의 상태/AMP 복원·부분 reset·alpha=1 고정·온라인 생성 없음·시뮬레이션 검사는 `check_bridge_layout_cache.py`를 사용한다. 이번 GPU 계측은 `output_etc/check/layout_cache_final_20261117/`에 둔다. `relation/layout_cache/hits`, `misses`, `hit_fraction`으로 적중률을 기록한다.
