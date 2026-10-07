# 왼쪽 힌지·오른쪽 손잡이 자동 닫힘 문

정면 관측자는 local -X에서 +X를 본다. +Y는 왼쪽, +Z는 위쪽이다. hinge 위치는 `(0,+0.45,0)`, 닫힌 앞 손잡이는 `(-0.075,-0.36,1.05)`m이며 양의 회전으로 안쪽·왼쪽으로 열린다. 문짝은 0.9×2.1×0.045m, 하단 간격 1.5cm, 개방 제한은 0~110°다.

`left_hinge_right_handle.urdf`는 고정 base의 문틀, panel, 앞 handle와 handle_back의 4개 rigid body와 1개 revolute DOF를 가진다. 손잡이는 문짝에 고정되어 있으며 충돌 가능하다. 다른 actor와 충돌하며 articulation 자체의 self-collision은 제외한다. 잠금장치나 회전식 손잡이는 없다.

자동 닫힘 설정은 URDF의 강한 motor가 아니라 `DoorFixture`의 position drive에 있다. 기본 `stiffness=6.0`, `damping=3.0`, target=0으로 90°에서 초기 토크 약 9.42N·m를 만든다. 문을 손으로 잡으면 복원력이 걸리고 놓으면 닫힌다. 다른 물리 로더에서 asset만 불러올 때는 이 스프링 설정을 별도로 적용해야 한다.

재생·검증 명령은 [config.md](../../../../markdowns/config.md)의 **왼쪽 힌지·오른쪽 손잡이 자동 닫힘 문**을 참조한다. 생성 코드는 `tokenhsi/utils/door_asset.py`, 환경 모듈은 `tokenhsi/env/tasks/multi_agent/door_scene.py`다. 학습 task·reward·AMP 통합은 `tokenhsi/env/tasks/multi_agent/humanoid_ma_push_door.py`와 config 문서의 **Push + Door open/hold Stage 1**에서 제공한다.
