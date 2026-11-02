# BEFORE 통로 확보와 정지 대기

실험: `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_clear`.
기반: `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute`.
가상 상자 관측, progress/state/success, 성공 gate, 초기화·거리·task 비율, 모델과 PPO/AMP는 유지한다. 비대칭 progress는 추가하지 않는다. 두 페널티는 환경 보상에 더해지고 기존 task 가중치0.5가 적용된다. 새 config의 계약에는 `before.wait_clear` 전체가 포함된다.

## 통로

각 BEFORE 후행은 자기 선행의 현재 사람→goal 및 운반 상자→goal XY 선분 주변 캡슐을 피한다. 무한 직선이 아니므로 끝점 밖에서는 끝점까지의 거리를 사용한다. 길이0이면 원으로 취급한다. 폭은 양쪽 물체의 반경 합+clearance다. 사람 반경은0.35m, 상자는 임의 회전에 보수적인 3D 대각선 절반이다. 후행 사람을 검사하며 ON_TOP은 후행의 실제 운반 상자도 검사한다. 두 통로와 후행 사람/상자의 가장 큰 침범도를 사용한다.

각 검사에서 signed clearance `c = distance_to_segment - combined_radius`를 계산한다. `max(0, -c / combined_radius)^2`(최대1)에 blocking_weight0.5를 곱해 음수 보상으로 준다. 선행 AT가 현재 성공한 프레임에는0이며, 다시 실패하면 재활성화된다. 판정은 실제 물리 상태이며 가상 좌표를 사용하지 않는다.

## 정지 대기

대기점이나 방향을 저장하지 않는다. 목표 중심 반경만 계산한다. 기존 progress 영역 반경과 goal 끝부분의 통로 반경 중 큰 값에 safe_margin0.10m를 더해 대기 중심 반경을 정한다. 그래야 큰 상자에서도 금지 구역 바깥에 대기 공간이 남는다. SIT/CLIMB는 후행 사람, ON_TOP은 후행 상자의 goal 거리를 사용한다.

대기 가중치는 `smoothstep(1 - abs(distance-radius)/wait_band)`와 `smoothstep(min_clearance/safe_margin)`의 곱이다. smoothstep은 입력을0–1로 잘라 `x*x*(3-2*x)`로 계산한다. wait_band0.50m는 반경 주변 띠의 반폭이다. 통로 안에서는 안전 가중치가0이므로 정지 페널티 없이 비켜줄 수 있다. 충분히 먼 접근 구간에서도0이다.

이동 페널티는 `-motion_weight * waiting_weight * horizontal_speed_squared`다. motion_weight 초기값은0.2이며, ON_TOP은 사람과 상자의 XY 속도 제곱 합을 사용한다. 선행 AT 성공 시 해제된다. 전후좌우/접선 이동에 모두 적용하며 관절 속도나 몸통 yaw 회전은 제약하지 않는다. 균형 유지·잡기·AMP 보상은 유지한다.

## 진단과 제한

`reward_terms/before_blocking`, `reward_terms/before_wait_motion`을 별도로 기록하고 total에도 포함한다. 환경 extras에는 waiting_active/weight/radius와 corridor_intrusion도 제공한다. 기존 설정은 새 항이 없어 보상·로그 크기가 유지된다.

초기 계수는 효과가 검증된 값이 아니다. 각 후행과 자기 선행 쌍의 통로만 다루며 다른 쌍까지 확장한 전역 통행 계획은 아니다. 직선 캡슐은 실제 운반 궤적의 근사이고 충돌이나 정지를 보장하지 않는다. 통로의 이동에 따라 안전 여부도 매 스텝 바뀐다. 추가적인 정지점 보상·강제 action clamp·WAIT 모드는 없다.

학습 후에는 대기 속도/누적 이동, 통로 침범, 선행 완료 시간·성공률, 사람/상자 접촉, 선행 성공 후 후행 수행 지연을 함께 비교해야 한다. 기존 virtual/asym 가중치에 새 보상만 켜고 추론하는 것으로 개선을 주장하지 않는다.
