# 공통 면 CONTACT 두 물성 비교

| 설정 | X | Y | Z | 질량 | 상자 마찰 |
| --- | --- | --- | --- | --- | --- |
| `stage1_push_face_xy_depth180_20kg` | 1.2~1.5m, 0.05m격자 | 1.8m | 1.3m | 20kg | 0.6 |
| `stage1_push_face_xy_depth200_25kg` | 동일 | 2.0m | 1.3m | 25kg | 0.6 |

두 env YAML은 상자의 Y·질량 및 이에 따른 RSI cache/SHA만 다르다. X는 사용자 지정 현재 범위를 유지한다. 밑면 별도 마찰·collision shape 분리는 적용하지 않는다. 바닥 static/dynamic 마찰1.0도 유지한다.

[공통 면 CONTACT](push_common_face_xy.md)는 양손에서 같은 유한 면까지의 XY 거리 중 최댓값으로 `exp(-10*d²)`를 계산한다. 현재 목표방향과 밀기 법선60° 이내 후보 중 minimax 공통면을 매번 선택하며 손 Z·중점 기준점/높이 보상은 없다. 자체성공은 같은 substep state>0.9+양손 각각/상자 반력≥0.5N, progress는 기존 root→상자 접근점수다. AT·paired포화·관측634D·AMP는 기준 h080 phase_face_xy를 유지하고 두 설정 모두 AT 기울기 제한은 없다. `height_m: 0.8`은 기존 reset 배치 metadata이고 CONTACT 높이 목표가 아니다.

## RSI 검증

원본 관절/속도를 유지한다. loco/PUSH50%·시작1~5m·목표1~4m·late70%/유효pool마지막30%·손중점≤1m와 양팔 밀기 자세 선별을 유지한다. 새 캐시는 기존5모션 선별pool의 부분집합만 허용하고 각각의 실제 크기/질량으로 모든 후보를 다시 물리 검사했다. 캐시는 `tokenhsi/data/stage1_push_face_xy_{depth180_20kg,depth200_25kg}_rsi_cache.npz`이며 SHA는 각 YAML에 고정한다. 원본 캐시와 원본 모션은 덮어쓰지 않는다.

각 설정에서7크기×1196프레임=8372조합·두사람16744초기상태를 원본 관절각 PD 유지0.1초/6tick 검사했다. 모두 통과,낙상/비유한/초기관통0·원본DOF position/velocity오차0이다. 실제 시뮬레이터 질량20/25kg·상자마찰0.6을 확인했다. 별도 실제 RSI 전수 배치에서도 몸관통/목표반대편위반/손면투영이탈0,손높이0.945716~0.972829m·손면간격약8cm·머리최소간격약0.323m였다. 초기거리state>0.9는8372/8372지만, 약8cm 간격이 있으므로 힘접촉 성공을 뜻하지 않는다.

각64평가환경·4096agent reset에서 원본 관절/속도·높이·loco거리·선별5모션·부분reset·634D/물체30D·12step finite를 확인했다. 두 설정 각각PUSH RSI비율은 결과JSON의 `rsi_fraction`을 참조한다. 초기PD검사는 장기균형·상자전도·학습정책성공의 검증이 아니다. 본학습은 실행하지 않았다.

검증/영상 디렉터리: `output_etc/check/push_face_pair_20261021/{depth180_20kg,depth200_25kg}/`. `rescreen.json`은 전수 초기물리검사, `audit.json`은 배치기하, `runtime/audit.json`은 학습reset 경로 감사다.

## 영상

두 `rsi_review.mp4`는 각각1280×720/10fps/47초이며, 실제 RSI reset에서 추출한 원본시간자세열을 CPU 투영했다. 실제 MJCF sphere/capsule·원본 발OBJ·해당크기의상자를 함께 표시한다. 다섯 PUSH구간과 보행시작·최대손거리/최소머리거리/최대발높이 사례를 포함한다. 정책 또는 물리 rollout 영상이 아니며 질량에 따른 동작 차이를 보여주는 영상으로 해석하지 않는다. 대표시간열의 X는1.3m이고 전수검증은7크기 전부다. 발바닥 수치는 실제 OBJ vertex 변환값이며 원본 보행자세 특성상 양발이 항상바닥에닿는조건은 아니다.

- [Y1.8m·20kg 영상](../../output_etc/check/push_face_pair_20261021/depth180_20kg/rsi_review.mp4)
- [Y2.0m·25kg 영상](../../output_etc/check/push_face_pair_20261021/depth200_25kg/rsi_review.mp4)

학습·로컬·서버VNC는 [config](../config.md) 목록 맨아래의 두 전용 명령을 사용한다. 기존학습은 중단/재시작하지 않았고, 각 scratch output을 사용한다.

## 재현 도구

사본 루트에서 전용 env/train/motion 인자와 headless·test/eval을 사용한다. 실행 GPU는 `TOKENHSI_GPU=6`,환경은 공통 `runtime_env.sh`를 따른다.

- `tokenhsi/tests/recheck_push_face_rsi.py`: 설정의 원본 선별cache를 새 물성에서 전수 재검사하여 출력디렉터리에 `verified_cache.npz` 생성(기존파일 덮어쓰기금지). 새설정의cache를 입력하면 해당pool을 다시검사한다.
- `tokenhsi/tests/audit_push_face_rsi.py`: 실제초기배치 전수검사·모든클립연속자세추출.
- `tokenhsi/tests/check_stage1_push.py`: 32회학습reset·부분reset·12step·원본관절속도/샘플러검사.
- `python tokenhsi/scripts/multi_agent/render_push_face_rsi.py --input output_etc/check/push_face_pair_20261021/depth180_20kg --config tokenhsi/data/cfg/multi_agent/stage1_push_face_xy_depth180_20kg.yaml`: 영상과발메시감사생성.25kg는동명depth200_25kg경로사용.
