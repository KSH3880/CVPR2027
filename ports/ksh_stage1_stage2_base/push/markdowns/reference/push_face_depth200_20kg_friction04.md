# Y2.0m·20kg·상자 마찰0.4 PUSH

`stage1_push_face_xy_depth200_20kg_friction04`는 기존 `stage1_push_face_xy_depth200_25kg`에서 상자 질량25→20kg·마찰계수0.6→0.4 및 전용cache/SHA만 바꾼 별도 실험이다. 상자 전체 surface 마찰이며 밑면만의 설정은 아니다. X1.2~1.5m/5cm격자·Y2.0m·Z1.3m, 바닥 static/dynamic마찰1.0, 목표1~4m·loco시작1~5m·loco/PUSH50%·원본관절/속도/손중점≤1m·공통면CONTACT·AT/paired포화·634D/AMP를 유지한다. AT기울기제한은 없다.

새 전용 cache는 `tokenhsi/data/stage1_push_face_xy_depth200_20kg_friction04_rsi_cache.npz`이며 SHA는 YAML에 고정한다. 기존 두설정과 cache/output을 덮어쓰지 않고 기존checkpoint 직접resume/test는계약검사에서거부한다. 본학습은 실행하지 않았다. 학습·로컬·VNC명령은 [config](../config.md) 목록 맨아래를 사용한다.

## 검증

검증 디렉터리는 `output_etc/check/push_face_depth200_20kg_friction04_20261021/`이다. `rescreen.json`에 새 물성에서 선별8372 size-frame·두사람16744상태의 전수0.1초/6tick PD 검사결과를 기록했다. 모두통과·낙상/비유한/초기관통0·원본DOF position/velocity오차0이다. 실제질량약20kg·상자shape마찰약0.4를조회확인했다. 초기짧은PD검사이며장기균형/전도/학습정책성능은입증하지않는다.

`audit.json`은 실제RSI배치전수/보행기하, `runtime/audit.json`은4096agent학습reset·부분reset·634D·12step 확인, `runtime/cpu.log`는계약/샘플러/보상회귀검사다. 원본모션/기존선별pool을변형하지않고새물성에서다시검사한다. 검증도구는 [두물성명세](push_face_pair.md)의 recheck/audit/check/render를재사용한다.

## 영상

[새 조건 RSI 영상](../../output_etc/check/push_face_depth200_20kg_friction04_20261021/rsi_review.mp4)은 실제새설정에서추출한초기자세를상자·MJCF충돌체·발OBJ와함께그린CPU영상이다. 질량/마찰에따른동작을보여주는정책/물리rollout영상은아니다. 전반0~40초는PUSH5구간·40~43초는PUSH경계사례·43~46초는loco시작검수·46~47초는발높이최대사례다.
