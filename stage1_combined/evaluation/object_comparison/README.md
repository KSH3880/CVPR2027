# 물체 종류별 SIT / CLIMB 비교

현재 기본 평가: **Box, Chair, Cabinet, Table 각각 독립적으로 SIT와 CLIMB**를 측정한다. 한 환경에 종류를 섞지 않는다. 각 종류 안에서 테스트 모델을 균등 복원 추출하며 모델은 환경 생성시 고정된다. 두 정책은 동일 seed의 같은 초기 사람/관절DOF/물체 모델·위치에서 시작한다.

| 종류 | 테스트 모델 | 실제 크기/형상 |
|---|---:|---|
| Box |18|원본 TokenHSI CLIMB test18 크기와 URDF 그대로|
| Chair |26|StraightChair_Normal test native meshes|
| Cabinet |9|Cabinet test native meshes|
| Table |8|Table_Square test native meshes|

Box XY는0.4×0.5,0.5×0.3,0.6×0.4m이고, 각각 높이0.2~0.7m(0.1m간격)다. 우리 일반 single_match 랜덤 크기 박스 분포와 구분한다.

목표 정의는 기존 평가 기준을 유지한다.
- Chair SIT: 원본 좌판/앉을 목표(`tarSitPos`)와 동일. 좌판+10cm.
- Box/Cabinet/Table SIT: bbox 상판중심+10cm. metadata SIT로 변경하지 않는다.
- Chair CLIMB: 좌판+94cm.
- Box/Cabinet/Table CLIMB: 원본 `tarSitPos` 높이+94cm.

검사점은 골반/root. 점 성공은 목표3D거리≤20cm, 영역 성공은 좌판/상판의각축내부80%XY와목표Z±20cm.1명1고정물체,loco시작,최대900step(일반종료899),첫episode종료frame포함ever. 접촉/발높이/방향/유지시간을 성공조건에 추가하지 않는다.

```bash
cd /home/hwanhee/juan/CVPR2027/stage1_combined
# 준비 및 CPU 목표/Box 크기 검증
python3 evaluation/object_comparison/test_targets.py
# 전체8조건×2정책 각4환경×seed0 구현 검증(900step)
python3 evaluation/object_comparison/run_eval.py --smoke
# 정식 전체 비교: 조건·정책당512환경×독립seed0/1/2, 총48개 순차실행, GPU5
python3 evaluation/object_comparison/run_eval.py
# 필요한 물체/행동만 선택
python3 evaluation/object_comparison/run_eval.py --categories Box Cabinet --actions CLIMB
```

공개 TokenHSI Stage1 40k, 우리 Stage1V2Fixed73k 고정. 새로운 학습/토크나이저 없음. 사람/물체 관측은 기존 두 모델 각자의 입력을 유지한다. 공개정책의 명시적목표는 위 표면기준으로 생성하고, 우리모델의 관계목표 및 평가도 동일표면기준이다. 우리모델은 별도의 CLIMB높이 입력이 없으므로 관측 차원을 늘리지 않는다. SIT의 물체 앞방향은 의자metadata, 나머지는 기존 로컬+X 규약이다.

코드는 이 디렉터리에 관리하며 실제 backend/자산/결과는 `CVPR2027/runs/object_comparison_20261022/`에 둔다. 검증된 이전 평가 사본 `runs/tokenhsi_native_objects_20261022/chair_source`를 template로 복사해 준비한다. baseline/ksh는 수정하지 않는다. `--workspace`로 새산출물루트를 선택할 수 있다. 새실행마다 고유tag사용,기존출력재사용거부. 코드변경시 새workspace로 준비한다.

정식 REPORT.md는8개행에 TokenHSI/우리모델의점/영역성공률 평균±표본표준편차를출력한다. summary.json에는 seed별결과,모델별결과를보존한다. 저장trace 독립재채점/해시/종료frame/float32float64, 정책간전체초기배열exact검증포함. smoke는정식성공률통계를생성하지않는다. 이전 SIT/CLIMB 결과는덮어쓰지않고보존하며,이번 종류내복원추출표본과 이전balanced표본결과를혼합해새통계로 만들지않는다.

검증 완료: CPU61자산·원본Box18크기·두행동목표/회전/높이경계. GPU5의8조건×두정책 각4env×seed0×900step(16실행) rc0, paired전체초기배열8건exact, 전trace독립재채점/종료frame/hash/float32float64 통과. 배치: runs/object_comparison_20261022/eval_20261022_205431_smoke/. 정식512×3seed 평가는 아직 실행하지 않았다.
