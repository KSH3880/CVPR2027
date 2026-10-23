# Stage1 PUSH 중점 CONTACT·원본 AT

전용 환경/train YAML과 train/test/VNC wrapper는 `stage1_push_midpoint_native_rsi`다. 기존 실험과 checkpoint를 혼용하지 않고 scratch로 학습한다. 실행 중인 학습과 원본 보관본은 변경하지 않는다.

- 물성: X=Y를 함께1.20~1.50m·5cm격자로 샘플링, Z1.30m 고정, 각 상자20kg·마찰0.6. 2명·2048학습환경·실제4상자·634D/8토큰·네트워크·AMP 및 native RSI 추출 방식은 유지한다.
- 초기 상자에서 목표까지 거리는 지정면 법선 방향1~4m 균등이다. RSI로 상자를 목표 쪽으로 이동한 뒤에는 남은 거리가1m보다 짧을 수 있다.
- 목표점은 초기 지정 면 중앙의 바닥 기준0.65m인 상자 로컬점이다. 상자 pose를 따라 이동/회전한다.
- `m=(왼손+오른손)/2`, `delta=m-target`, `phi_C=exp(-10*(delta_x²+delta_y²)-(delta_z/0.04)²)`이다. 높이 무감점 구간, 개별 손의 면 거리, 힘 gate, 별도18cm/±3cm gate를 사용하지 않는다. 손의 간격/개별 위치는 점수에 직접 영향을 주지 않는다.
- CONTACT 자체 성공은 제어 step 끝의 원래 `phi_C > 0.9`이다. `0.9`와 같은 값은 실패다. physics tick에서 성공한 적이 있다는 이유로 제어 step 끝까지 성공을 유지하지 않는다.
- CONTACT 접근점수는 기존 `1/(1+max(root_box_xy_distance-0.5,0))`다.
- 원본 Stage1 AT는 그대로: `phi_A=exp(-10*||box_center-goal||²)`, 접근점수 `1/(1+max(distance_xy-0.1,0))`, 성공은3D거리≤0.2m에 해당하는 state 기준+Z오차≤0.001m. 회전/속도 gate를 추가하지 않는다.
- 원본 paired 포화 유지: CONTACT 자체 성공 또는 현재 AT 성공이면 CONTACT의 state/progress/success를 각각0.2로 포화한다. AT는 현재 AT 성공이면 각각0.2로 포화한다. 둘 다 최대일 때 과제보상1.2이며, AT에서 벗어나면 CONTACT 자체 판정으로 돌아온다. `own_success`와 `reward_saturated`는 구분한다.
- 본인 과제 보상만 합산하고 기존 페널티와 `0.5*env_reward+0.5*AMP`를 유지한다.
- 보상 계약은 `pushTask.contact_objective: midpoint_only_v1`로 분리한다. 구면/힘 CONTACT checkpoint는 직접 resume/test하지 않는다.
- 전용20kg/X=Y/Z고정7 profile RSI cache를 사용한다. 이전독립XY49 profile·25kg 설정이나 XYZ가변343 profile metadata를 가진 캐시는 직접 로드하지 않는다. 캐시 생성 후 전용 YAML SHA를 갱신한다. 기존 실험 cache/YAML은 덮어쓰지 않는다.

## RSI 방식

기존 `stage1_push_rect_native_rsi`와 동일한 `PushNativeRsiCache`/원본21 PUSH 모션을 사용한다. reset의50%는 PUSH RSI이며 나머지는 loco다. MotionLib 원가중으로 clip을 선택하고,70%는 유효프레임 후반30%·30%는 전체유효프레임에서 추출한다. 기존0.1초 물리/겹침 필터를 사용하고 관절/손높이 retarget은 없다. 장면에 맞춘 사람 전체의 yaw/XY 정렬과 초기 상자 진행률은 기존과 같고, 캐시는 새7크기에 맞춰 재생성한다.

## 관통 검사

기존 native 필터는 모든 body 중심이 상자 면보다6cm 이상 안쪽에 있는 깊은 관통과 발의 바닥 아래 위치를 검사한다. RSI 정렬 시 관절은 그대로 두고 사람 전체를 뒤로 이동시켜 상자 주변8cm 여유를 확보한다. 이 필터 자체가 collider 무관통을 보장하지는 않는다.

추가 감사는 실제MJCF 손 collider(반지름4cm구체)를 기준으로 손 중심을 상자 로컬좌표로 변환하고, oriented box의 signed distance로 구체 겹침 깊이를 측정한다. 전7크기×21모션의 모든 원본프레임에서 reset과기존0.1초물리검사각tick을 측정해 `xy_tied_build/audit.json`의 `hand_box_audit`에기록한다. 이는측정이며 native 유효프레임필터나학습보상에새gate를추가하지않는다. 장기정책rollout중관통은이초기화검사와별개다.

실측: reset손224168건에서겹침/중심내부0건, 0.1초물리손1345008건에서중심내부0건·표면겹침176280건·최대2.361mm. cache56042조합전부유효. 결과는 `output_etc/check/stage1_push_midpoint_native_rsi_xy_tied_build/audit.json`에보존했다.

## 실행

사본 루트 `ports/ksh_stage1_stage2_base/push`에서 실행한다. 본학습/로컬/VNC 명령은 [config](../config.md)의 목록을 따른다.

```bash
# 전용 캐시가 없거나 물리 계약이 바뀌었을 때 생성
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_stage1_push_midpoint_native_rsi_cache.sh

# CPU 보상/AT/포화 검사
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
PYTHONPATH=tokenhsi python -m pytest -q tokenhsi/tests/test_stage1_push_midpoint.py tokenhsi/tests/test_stage1_push.py tokenhsi/tests/test_stage1_push_rect_point.py'

# 2048환경 짧은 확인. 본학습과 출력 분리.
CHECK=1 MAX_ITERATIONS=1 TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_midpoint_native_rsi_train.sh 2 2048 4

# 본학습. 기존 학습을 자동 중단하지 않는다.
env -u CHECK -u MAX_ITERATIONS -u OUTPUT_PATH -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_midpoint_native_rsi_train.sh 2 2048 4
```

확인 결과와 미검증 항목은 [changelog](../../changelog.md)를 따른다. 본학습 성공률/GUI/장기안정성은 짧은 실행 검증으로 보장하지 않는다.
