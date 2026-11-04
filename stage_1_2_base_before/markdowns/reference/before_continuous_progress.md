# BEFORE asym3m 연속 progress

설정: `stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute_radius3m_continuous`.
원본 virtual_asym + 반경3m, 접근 timeout 및 state 정규화 없음.

선행 완료 전 수평거리 d의 progress는 `0.1 / (1 + (max(d-3,0) + 10*max(3-d,0))/sigma)`이다. 3m에서0.1로 최대다. SIT/CLIMB는 사람 rootXY, ON_TOP은 payload XY와 선행 goal 사이의 거리다.

첫 gate 개방 순간 같은 위치에서 대기 보상 r0와 접근 곡선 q0를 저장한다. 접근 곡선은 기존 역거리 형태 `q(d)=1/(1+d/sigma)`를 사용하되 bbox 반경의 평탄부는 없애 goal에서1에 도달한다. 개방 이후 보상은 `clip(r0 + (0.2-r0)*(q(d)-q0)/(1-q0), 0, 0.2)`다. d가 같으면 gate 자체에 의한 변화는0이며, 가까워지면 증가하고 멀어지면 감소한다. 시간에 따른 보상 증가는 없다.

progress 전환은 최초 개방 이후 유지한다. gate가 다시 닫히거나 열려도 anchor를 갱신하거나 대기 곡선으로 되돌리지 않으므로 gate 반복으로 보상을 올릴 수 없다. state/success gate는 기존 current AT success 그대로다. 따라서 첫 개방 이후에는 asym 내부 접근 페널티도 다시 켜지지 않는다.

후행 progress의 기존 성공 saturation은 이 모드에서 사용하지 않는다(불연속 재발 방지). state/success 보상은 그대로이므로 전체 reward까지 연속이라는 의미는 아니다. goal 도착 전 성공 영역에 들어가더라도 progress가 즉시0.2로 점프하지 않는다.

예외: 최초 개방 순간 이미 정확히 d=0이면 남은 접근 구간이 없다. 이때는 보상 연속성을 우선해 r0를 유지하며, 후퇴하면 `r0*q(d)`로 낮아진다. 그 에피소드에서는 이 예외만으로0.2까지 증액하지 않는다.

anchor는 실제 reward step에서만 저장하고 부분 reset은 해당 환경의 anchor만 초기화한다. 단순 geometry 진단은 anchor를 변경하지 않는다. 새 설정은 기존 checkpoint와 계약을 분리하며 기존 학습에는 소급 적용하지 않는다.

구현: `before_continuous_progress.py`, `BeforeRuntime.step`, `HumanoidMABefore._evaluate_ontop_context`. 검증: `test_before_continuous_progress.py`, `check_before_continuous_progress.py`.
