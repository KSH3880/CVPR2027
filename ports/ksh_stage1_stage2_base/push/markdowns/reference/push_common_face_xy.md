# PUSH 공통 면 XY CONTACT

GPU6 phase 두 기준 설정에서 CONTACT state와 자체 success만 동적 공통 면으로 분리한다. 기존 실행은 재시작하지 않는다. GPU1 코드는 직접 검증하지 않았고 사용자 제공 명세를 구현했다.

네 실제 유한 옆면의 XY 투영에서 각 손의 최단 거리를 구한다. 현재 목표 방향과 안쪽 밀기 법선 XY 방향이60° 이내인 후보 중 양손 최대거리가 최소인 면을 매 계산마다 고른다. 면 밖은 가장자리 거리를 포함한다. 손 Z·중점 기준점·높이 감점/gate는 없다. 동률은 면 인덱스 순, 목표 XY가0이면 현재 로컬+Y 방향이다. 기울기로 XY 법선이 사라진 면은 후보에서 제외하고 후보가 없으면 state0·자체성공false다. state=`exp(-10*max(d_left,d_right)^2)`다.

자체 success는 같은 물리 substep의 state>0.9와 양손 각각/상자 법선 반력≥0.5N이다. 제어 step OR 누적·reset 제거·AT paired 포화는 기존대로다. progress는 점 목표가 없는 기존 root→상자 중심 접근점수 `1/(1+max(root_box_xy_distance-0.5,0))`를 유지한다. state/progress/success 가중치는 각각0.2다.

두 실험은 기존 h080/AT무제한, h100/AT5° 기준의 RSI·상자·목표·AMP·관측을 유지한다. `height_m`은 기존 reset 배치 메타데이터이며 보상에서 참조하지 않는다. RSI≤1m 선별은 유지한다. 새 contact_objective=`common_face_xy_v1`, pushPointReward는 빈 dict이며 기존 checkpoint 직접 resume/test는 계약 검사에서 거부한다. 새 scratch 전용이다.

실행 명령은 [config](../config.md)의 `*_phase_face_xy`를 사용한다. 저장소 루트에서 wrapper의 전체 경로로도 실행 가능하다.
