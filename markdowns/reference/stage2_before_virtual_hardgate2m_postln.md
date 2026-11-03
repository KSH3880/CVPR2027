# BEFORE virtual 2m hard gate Post-LN

`stage2_before_virtual_asym_postln`을 독립 복사한 `experiments/stage2_before_virtual_hardgate2m_postln/`에서 asym 안쪽 감쇠를 2m 하드게이트로 교체한다. 기존 실험과 학습은 유지한다.

후행 SIT/CLIMB/ON_TOP의 progress는 선행 AT가 현재 실패하고 `d < 2m`이면 0, 그 외에는 `1/(1+max(d-R,0)/sigma)`다. 정확히 2m는 기존 값을 유지한다. 가중치 0.2, sigma 1, R은 기존 상자 XY 반대각선+과제 buffer다. 0.6×0.6m 상자에서 2m 보상은 SIT/CLIMB 약0.088, ON_TOP 약0.081이다.

거리는 선행 AT 최종 목표와의 XY 거리이며 SIT/CLIMB은 후행 사람 root, ON_TOP은 운반 물체를 쓴다. 선행 AT 성공 시 같은 frame에 차단을 해제하고, 다시 실패하면 재적용한다. 속도·정지 보상을 추가하지 않으므로 2m 바깥에서 정지 자체를 보장하지 않는다.

state/success/termination·HOLDING/독립/PUSH·AMP·샘플링·Post-LN은 기준 그대로다. 전용 YAML의 `before.hard_gate_progress.distance: 2.0`을 계약에 포함해 기존 asym checkpoint 혼용을 거부한다. Stage1 PUSH26k부터 전이하며 사용자 지정4명·1024환경·8토큰이다. 실행은 [config](../config.md)의 전용 train/test/VNC를 따른다. 본학습은 `koo_stage2:0.0`에서 GPU2로 실행하며, 실행 정보는 `output/stage2_before_virtual_hardgate2m_postln/launch_1024.json`을 따른다. 초기2048환경 로그와 설정은 보존한다.
