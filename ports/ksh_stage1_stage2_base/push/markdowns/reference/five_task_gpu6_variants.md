# GPU6 물성의 다섯 과제 학습 두 설정

기존 네 과제+PUSH_AT, 4-family AMP의 [공통 구현](five_task_push_at.md)을 사용한다. 두 설정 모두 사람2명·실제 상자4개·목표2개·8토큰/634D·AMP133D/프레임·2048학습환경이다. 기존 네 과제와 네트워크·PUSH21모션/공유loco12모션은 유지한다. 학습/로컬/VNC 명령은 [config](../config.md)의 맨 아래 두 항목에 있다.

| 설정 | PUSH 상자 X/Y/Z(m) | 질량 | 상자 마찰 |
| --- | --- | --- | --- |
| `stage1_v2_fixed_five_task_depth180_20kg` | 1.2~1.5 / 1.8 / 1.3 | 20kg | 0.6 |
| `stage1_v2_fixed_five_task_depth200_20kg_friction04` | 1.2~1.5 / 2.0 / 1.3 | 20kg | 0.4 |

X는5cm격자·바닥마찰1.0이다. 물성별 기존 선별 cache와 SHA를 각각 연결한다. 기존 PUSH 전용 또는 다른 조건 checkpoint를 직접 resume하지 않는다. output과 실험명은 두 설정별로 분리했다.

## 최종 목표와 생성 위치

- **최종 reset 상자 중심→목표: 1~5m 균등.** 초기 상자 로컬+Y 방향을 사용하며, RSI로 상자가 진행한 경우 목표도 같은 변위만큼 옮겨 **RSI 적용 후에도** 실제 거리가1~5m다. 에피소드 도중에는 목표를 고정한다.
- **PUSH 보행 시작은 상자 주변360° 무작위**, 사람 heading도 상자/목표와 독립적으로 무작위다. 목표 쪽에 서서 시작하면 밀 수 있는 반대편으로 돌아가 접근해야 한다. 정책이 이미 우회 행동을 학습했다는 의미는 아니다.
- **거리 기준은 전신 collider bounds와 상자 표면 사이 XY 최소 간격1~5m**다. root/상자 중심 거리를 사용하지 않는다. 실제 MJCF sphere/capsule와 발OBJ 전체 bounds를 포함한다. 전체 사람을 강체 이동·yaw 회전할 뿐 관절이나 높이를 맞추려고 자세를 retarget하지 않는다.
- 모든 PUSH 사람과 **네 실제 상자 전부**의3D bounds를 확인한다. 보행 시작은 어느 상자와도 최소0.2m 여유를 요구한다. 실패하면 장면을 다시 샘플링하고64회 실패 시 중단한다. 초기 겹침을 허용하는 fallback은 없다. 이후 정책/물리 rollout에서 충돌이 없다는 보장은 아니다.
- PUSH RSI는 기존 원본 관절/속도·손중점≤1m·실제밀기 구간 pool·loco50%/RSI50%를 유지한다. 접촉 직전 자세이므로 보행용1~5m 조건을 RSI에 적용하지 않고 기존 collider margin2cm를 유지하며 다른 상자와도 겹치지 않는지 검사한다.

## 검증과 영상

`tokenhsi/tests/test_push_random_spawn.py`는 회전·크기·방향을 바꿔도 전신 표면 간격을 지키는지, 360° 시작인지, root가 아닌 팔/발 또는 미할당 상자 충돌도 제외하는지 검사한다. `check_five_task.py`는 실제64평가환경에서32reset·실제 물성·다섯 과제·634D/AMP133D·부분reset·PUSH RSI 원본 DOF/속도·손높이·0.1초 원래 관절각 PD 유지·12제어step finite를 확인한다. 다른 네 과제는 이 검사 프로세스에서만 명시적으로 loco fallback한다. 네 과제 전체 RSI cache 재검사나 PPO 학습은 아니다.

산출물은 `output_etc/check/five_task_random_depth180/`와 `output_etc/check/five_task_random_depth200_friction04/`다. 각 `audit.json`, `rsi_review.mp4`, `review.jpg`(선별 PUSH), `starts_review.jpg`(보행8방향)를 확인한다. 영상은 **실제 reset에서 추출한 RSI 자세 샘플**과 실제 MJCF collider/발OBJ, 사람2명·상자4개·목표를 표시한 탑뷰다. 원본 시간의 연속 모션·정책/물리 rollout 영상과 구분한다. 검사 결과와 한계는 [changelog](../../changelog.md)에 기록한다.
