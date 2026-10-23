# Stage1 PUSH 밀리는 방향 깊이1.8m

- 새 실험 `stage1_push_depth180_native_rsi`: **X만1.20~1.50m/5cm 가변**, Y1.80m 고정, Z1.30m 고정, 각 상자20kg·마찰0.6. 실제상자4개·사람2명·2048학습환경·634D/8토큰을 유지한다.
- X는 손을 대는 면의 좌우 폭, Y는 사람→상자→목표의 깊이다. 상자초기yaw는무작위지만 축은상자로컬기준이다. 손목표는로컬 `(0,-0.9,0.65-Z/2)`인−Y면중앙이고, 목표는초기+Y방향1~4m균등이다. 로컬±X옆면을밀거나긴Y면의폭방향으로이동하도록설정하지않는다.
- 중점만CONTACT/state>0.9·원본AT/현재AT성공시CONTACT포화·페널티·AMP50%는기존중점설정과동일하다.
- 일반시작은 root–상자중심 거리1~5m를사용한다. 원본 HOLDING/CARRY의1~10m를참고하되사용자요청으로상한을5m로설정했다. 표면거리가아니다. 원본은360도방향이지만 PUSH는−Y뒤쪽으로한정한다. 실제크기+4cm영역에몸체중심이겹치면최대64회거리를재추출하고실패시초기화를중단한다. RSI모션시작과상자→목표1~4m범위는별개다.
- native RSI도동일한21모션/원가중·50%PUSH RSI와50%loco·70%유효프레임후반30%추출·원본DOF position/velocity보존·관절/손높이retarget없음이다. 장면에맞춘전체yaw/XY정렬과상자초기진행률은유지한다. 새7크기cache를분리한다.
- 기존RSI의0.1초물리/겹침필터와8cm뒤이동을유지한다. 전크기/전프레임의실제4cm손구체관통감사는build `audit.json`의 `hand_box_audit`를따른다. 이검사는초기구간이며장기정책의무전도를보장하지않는다.
- `PushNativeRsiCache`의크기index는캐시metadata의축별최소/최대로구성하여Y1.8m를지원한다. 이전실험의추출법은변경하지않는다.
- 물성계약이다르므로기존X=Y/독립XY checkpoint를직접resume/test하지않고새로학습한다. 기존실행중학습/config/cache/output은보존한다.

## 검증

CPU34개·GPU6전7크기56042 RSI조합통과. RSI초기손겹침0,0.1초물리손중심내부0/구체표면겹침최대2.416mm.일반시작2126건에서도손구체/몸체중심겹침0,실제root중심거리1.035~4.998m였다. 전신메시교차검사는아니며장기정책무전도는미검증이다.

## 실행

사본루트에서실행한다. [실행목록](../config.md)에학습/로컬/VNC명령이있다.

```bash
# 캐시가 없거나물성이달라진경우만준비
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_stage1_push_depth180_native_rsi_cache.sh

# 새본학습
nvidia-smi -i 6
env -u CHECK -u MAX_ITERATIONS -u OUTPUT_PATH -u RESUME_CHECKPOINT -u TRANSFER_CHECKPOINT TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_native_rsi_train.sh 2 2048 4

# 짧은학습은별도check출력으로분리
CHECK=1 MAX_ITERATIONS=1 TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_native_rsi_train.sh 2 2048 4
```

검증결과/미검증항목은[변경기록](../../changelog.md)을따른다.
