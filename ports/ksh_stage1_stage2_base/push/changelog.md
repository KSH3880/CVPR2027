# Changelog

최신 변경부터 기록한다. 현재 실행법은 [config](markdowns/config.md), 코드 위치는 [structure](markdowns/structure.md)를 따른다. 과거 변경·검증 전체는 [보관 changelog](markdowns/legacy/changelog_before_cleanup.md)에 있다.

## 2026-10-23

### 기존 범위 중심 대칭XYZ PUSH 크기 제로샷

- 평가 전용 `PUSH_EVAL_SIZE_X/Y/Z`로 실제 PUSH asset크기와 bbox입력을 함께 바꾼다. Y2m모델 명령은 X0.9~1.8/Y1.7~2.3/Z1.0~1.6m로 기존중심1.35/2.0/1.3m를 유지한다. 보행 시작만 허용·중간 위치/골전환 차단, 학습YAML/checkpoint/보상·질량20kg·마찰0.4 유지. 크기는 actor생성시0.05m격자로 독립샘플링하고 실행중고정하며 물성/실제크기를 `size_zero_shot.json`에 기록한다. Z로PUSH를판별하던 초기화는 평가에명시적assetmask를전달해 보정했다. [명령](markdowns/USAGE.md#push-크기-제로샷).
- CPU 기존5과제/새크기회귀12개·Python/diff 통과. GPU6 Y2m17500checkpoint로4평가환경 headless·1환경 VNC6096 각12step×2회 rc0. 별도4환경 물성검사에서실제20kg/마찰0.4·bbox입력일치·접지·PUSH배정·RSI0·12step finite통과. 결과 `output_etc/check/push_size_zero_shot_{depth200_smoke,vnc_smoke,physics}/`. 장기성공률 미평가·기존학습유지.

### 목표 고정·중간 상자 이동 두 조건

- 평가 전용 `PUSH_BOX_SWITCH_MODES=across_goal,behind_agent`를 추가했다. 기본60step(2초)에 목표 반대편/현재 pelvis 뒤쪽으로 반복별 교대 이동한다. 목표·사람/AMP/episode는 유지하고 상자 높이/회전/물성을 보존하며 상자속도만0으로 만든다. 현재 콜라이더/다른 상자와 겹치면 같은 ray를 따라 더 멀리 배치한다. 새 배치의 접촉 면/관측·성공 이력을 갱신하며 이전 성공/teleport거리를 새 성공/밀기거리로 세지 않는다. 결과는 `push_box_switch_*.json`; [명령](markdowns/USAGE.md#push-중간-상자-이동-제로샷).
- CPU 방향/90°·180° 회귀4개·문법/diff 통과. GPU6 두17500회모델: Y2m각4평가환경·12step/3step전환 headless 및 Y1.8m각1평가환경·90step/60step전환 VNC6096의 두조건 rc0. 목표/사람 불변·겹침0·상자높이/회전보존/속도0·관측finite 확인. 결과 `output_etc/check/push_box_switch_{depth200_smoke,depth180_vnc_smoke}/`. 장기 대응 성공률은 미평가·기존 학습 유지.

### PUSH 평가 중90°/180° 목표 전환

- 평가에서만 `PUSH_GOAL_SWITCH_ANGLES=90,180`으로 반복별 각도를 교대한다. 기본150step 후 초기 목표 방향을 회전해 현재 상자에서2m 위치로 목표/마커만 갱신한다. 사람/상자 reset·재학습 없이 현재 관측을 재계산하며 이전 성공 이력을 분리한다. `push_goal_switch_*.json`에 좌표와 전환된 장면의 성공률을 기록한다. 명령은 [USAGE](markdowns/USAGE.md#push-중간-목표-전환-제로샷).
- CPU 회전 좌표1개 통과. GPU6/tokenhsi 17500회checkpoint로 Y2m각2평가환경 headless 및 Y1.8m각1평가환경 VNC6096에서90°/180° 각12step·3step 전환 완료(rc0), JSON의 각도/전환장면/목표거리2m 확인. 결과 `output_etc/check/push_goal_switch_{depth200_smoke,depth180_vnc_smoke}/`. 장기 새목표 도달 성능은 미평가. 기존 학습은 유지했다.

## 2026-10-22

### 5과제 Stage1 평가에서 PUSH RSI 선택 복구

- `humanoid_ma_five_task.py`의 평가 RSI 확률0 고정을 제거하고 평가 skill의 PUSH 정규화 확률을 사용한다. 학습 RSI50%·캐시/관절/속도·보상/모델은 유지하며 기존 GPU6 두 학습은 종료/재개하지 않았다.
- `check_five_task_eval_rsi.py`로 GPU6/tokenhsi 두 물성 각16평가환경 확인: PUSH128/128 RSI·loco0/128·혼합112/256 및120/256·원본 DOF position/velocity·초기관통0·목표1~5m·부분 reset·12step finite 통과. 결과 `output_etc/check/five_task_eval_rsi_fix_depth{180,200}/eval_rsi.json`. `EVAL_SKILLS=push EVAL_SKILL_PROBS=1.0`으로 viewer를 다시 실행해야 적용된다.

## 2026-10-21

### 두 물성 1000회에서 PUSH 100% 재개

- 두 새PUSH100% output 전용 TensorBoard6012 실행·HTTP200·두run인식 확인. 기존6008은 유지하며 재실행 명령은 아래 명세에 기록했다.
- 5과제 Stage1 전용 `push_at` 학습을 허용하고 두 depth180/depth200 train wrapper가 `TASK_GRAPH`를 받도록 했다. `relation_config.yaml`/checkpoint에 실제 `task_graph`를 남긴다. YAML·checkpoint 계약·기존 AMP expert·보행/RSI 비율은 유지하며 [재개 명령](markdowns/reference/push100_resume1000.md)의 output은 기존 혼합 학습과 분리한다.
- CPU8개·각4096 agent PUSH_AT·셸/Python 문법 통과. GPU6 각2048환경에서 원래1000회checkpoint 재개→1002회/frame131334144 종료, 모델finite·optimizer 존재·preset/환경수 저장 확인. `output_etc/check/push100_resume1000_verification.json`, `push100_resume1000_depth{180,200}/train.log`. 기존 학습은 종료하지 않았고 새 본학습은 미실행이다.

### 5과제 PUSH CONTACT·접촉 이동/목표 진척 로그·500회 재개

- `edge_context_task.py`의 관계별 집계에 PUSH_CONTACT11을 추가하고 `HumanoidMAFiveTask`에 PUSH만의 CONTACT/AT/동시만족률·물리tick XY 이동·접촉중 이동·부호있는 목표거리감소/양수전진량·종료episode누적을 연결했다. step평균/초당속도와 미터 단위를 분리한다. CONTACT 거리진단은 기존 HOLDING 값 대신 실제 공통면XY거리다. 보상·성공·모델·RSI 계약은 유지한다.
- CPU12개·GPU6 두물성 각64평가환경/12step·finite/부분reset 통과. 각2048환경 짧은PPO(1~2회차)에서 실제 접촉·episode이동량 TensorBoard 생성·목표거리약3m·모든추가지표finite 확인. `check_five_task.py`는 연속reset/부분reset 검수 후 rollout직전fullreset으로 미반영root setter가 이동에 섞이는 검사오류를 보정했다.
- 사용자 지시에 따라 두 기존 학습의 500회 번호checkpoint를 기다리는 고정PID/시작시각 검증 감시기를 활성화했다. 완성checkpoint의epoch/필수학습상태를 확인한 뒤 해당PID만종료·전용wrapper의RESUME_CHECKPOINT로2048환경재개한다. [지표·재개명세](markdowns/reference/push_training_metrics.md), 상태/로그 `output_etc/check/push_metrics_resume500_20261021/`.
- 두학습500회재개완료: 첫새epoch501/frame65536000·각2048환경/GPU6·기존PID종료·추가지표32개finite확인. 확인시Y1.8m522회/Y2m503회. `verification.json`에새run·관측지표·검증회차를저장했다.

### GPU6 두 물성의 5과제·360° 안전 시작·목표1~5m

- `stage1_v2_fixed_five_task_depth180_20kg`·`stage1_v2_fixed_five_task_depth200_20kg_friction04` env/train·train/test/VNC·독립output을 추가했다. 기존5과제/4상자/634D·AMP133D를 유지하고 PUSH 물성은 각각Y1.8m/20kg/마찰0.6, Y2m/20kg/마찰0.4다. 목표는RSI 적용 후 실제상자중심에서도1~5m, 보행은360°/독립heading·전신collider표면간격1~5m다. 네상자전부와보행최소0.2m/RSI기존2cm여유를검사하고 실패시재추출·한도초과시중단한다. 기존4과제/모션/관절/원본cache는보존한다.
- CPU47개통과. GPU6각64평가환경·32reset에서PUSH672상태(보행338/RSI334),원본DOF position/velocity·손중점≤1m·초기상자겹침0·8방향·부분reset·12step finite확인. 실제목표범위각1.0025~4.9929/1.0020~4.9941m,전신표면간격각1.0135~4.9969m. 각RSI334상태의0.1초PD(6tick) 낙상/비유한0,실제20kg·마찰0.6/0.4조회확인. 다른네과제는이검사에서만loco fallback,전체RSI pool 재검사/장기안정/PPO 본학습은아니다.
- 실제reset자세·MJCF콜라이더/발OBJ·사람2/상자4/목표탑뷰 영상두개(각1280×720·10fps·42.4초/424frame)를생성·실제프레임검토·전체디코딩확인했다. 정책/물리rollout영상이아니다. `output_etc/check/five_task_random_depth{180,200_friction04}/`의 `audit.json`·`rsi_review.mp4`·`review.jpg`·`starts_review.jpg`; [명세](markdowns/reference/five_task_gpu6_variants.md). 셸6개·설정/문서링크/diff 확인,본학습미실행.

### 기존 네 과제에 PUSH_AT 추가·모델 구조 유지

- `stage1_v2_fixed_five_task`·`stage2_shared_full_success_five_task`를 별도 env/train·train/test/VNC로 추가했다. 기존 SIT/CLIMB/CARRY_AT/CARRY_ONTOP ID0~3과 구현을 유지하고 PUSH_AT=4(template5, CONTACT11+AT7)를 같은 정책의 다섯 번째 과제로 연결했다. Stage1 비율8/20/26/26/20%, Stage2는 기존 네 과제의 solo를 유지하고 coupled 그룹만 PUSH_AT 상자/목표를 공유한다.
- 원본 GTA/tokenizer/Action MLP·actor/critic 구조는 유지하고 task embedding 두 행렬만4→5행이다. 사용자 확인에 따라 AMP는 carry/sit/climb/PUSH의4-family one-hot으로 연결했다. 모션특징129D 유지·조건3→4D·프레임132→133D/10frame1320→1330D이며 discriminator 첫 입력만 확장하고 hidden1024/512와 층 수는 유지한다. zero-code 우회는 제거했다. 다섯 과제 Stage1→Stage2 전이·별도 checkpoint 계약·과제별 평가 지표를 연결했다. 이 사본의 끊어진 SIT/CARRY/CLIMB 데이터 링크만 읽기 전용 원본에 다시 연결했다.
- 사용자 최종 지시에 따라 RSI·시뮬레이션 추가 검사는 중단하고 코드/config만 마무리했다. 최종4-family/133D 버전 CPU43개 통과(기존 ID/packet·정책/discriminator forward·원본 대비 embedding/AMP 첫 입력 외 shape 동일·4-family matching/정규화·공동 source/goal·Stage1→Stage2 전이·이전zero-code checkpoint 거부). 여섯 셸문법·YAML/Python 문법·문서 링크·157모션/물체 경로 확인. 이전 중간 버전 GPU 결과는 최종 코드 검증으로 간주하지 않는다. 본학습/최종 시뮬레이션 미실행·기존 학습 유지. 명세 `markdowns/reference/five_task_push_at.md`, CPU결과 `output_etc/check/five_task_amp4_cpu.log`.

### Y2m PUSH를 20kg·상자 마찰0.4로 분리

- 기존 `stage1_push_face_xy_depth200_25kg`를 보존하고 질량25→20kg·상자전체마찰0.6→0.4만 바꾼 `stage1_push_face_xy_depth200_20kg_friction04` env/train·train/test/VNC·독립output/cache를 추가했다. X1.2~1.5m·Y2m·Z1.3m·바닥마찰1.0·공통면CONTACT/AT·RSI분포는 유지한다. 기존checkpoint와직접혼용하지않는다. 검증기는 고정0.6 대신 해당YAML의 실제마찰을비교하도록수정했다.
- GPU6 선별8372 size-frame·두사람16744상태의0.1초/6tick PD 전수재검사 모두통과,낙상/비유한/초기관통0·원본DOF position/velocity오차0,실제20kg/마찰0.4조회확인. 실제배치전수몸관통/목표반대편/손면투영위반0. 각64평가환경4096agent reset·부분reset·634D·12step finite 통과(RSI0.4917),CPU44개통과.
- 새조건의실제RSI자세를추출한47초/470frame 영상 `output_etc/check/push_face_depth200_20kg_friction04_20261021/rsi_review.mp4`를생성하고전체디코딩/미리보기확인했다. 정책/물리rollout영상은아니다. 명세 `markdowns/reference/push_face_depth200_20kg_friction04.md`. 장기성능/본학습은미실행·실행중1번학습은유지했다.

### 공통 면 CONTACT·Y1.8m/20kg와 Y2.0m/25kg RSI 검증·영상

- 사용자 최종 조건대로 상자 마찰0.6·X1.2~1.5m/5cm격자·Z1.3m 유지, Y1.8m/20kg와 Y2.0m/25kg의 `stage1_push_face_xy_depth{180_20kg,200_25kg}` env/train·train/test/VNC·독립output을 추가했다. 공통면state·기존0.5N성공/AT paired포화,목표1~4m·loco시작1~5m·RSI50%·원본관절/속도/손중점≤1m를 유지하며 두설정 모두 AT기울기제한없음이다. 옆면/밑면별 마찰은 변경하지 않았다.
- 물성별 기존선별8372 size-frame을 두사람16744상태로 전수0.1초/6tick PD 재검사해 모두통과(낙상/비유한/초기관통0·DOF position/velocity오차0),별도cache/SHA를 연결했다. 실제배치전수의 몸관통/목표반대편/손면투영위반0·손면거리약8cm·손높이0.945716~0.972829m. 각64평가환경4096agent reset/부분reset·634D·12step finite 통과,RSI비율0.4932/0.4917. 초기힘접촉·장기균형/전도·정책성능 검증은 아니다.
- 실제MJCF충돌체·발OBJ·상자 및5클립/보행/경계사례를 표시한 RSI원본시간자세열 영상 두개(각1280×720·10fps·47초)를 생성했다. 정책/물리rollout영상이아님을표기했다. 발바닥낮은쪽5.0~41.3mm·양발>40mm는3/1196프레임으로원본발걸음을유지한다. 검증: 사본 `output_etc/check/push_face_pair_20261021/`,명세 `markdowns/reference/push_face_pair.md`. CPU41개·여섯셸문법·문서경로/diff·영상470frame전체디코딩통과. 본학습은 미실행·기존학습유지.

### CONTACT를 동적 공통 유한 면 XY 거리로 분리

- GPU6 phase 두 설정에서 별도 `*_phase_face_xy` env/train·train/test/VNC·output을 추가했다. 각 손의 동일 유한 면 XY 거리 중 최댓값으로 `exp(-10*d²)`를 계산하고 현재 목표와 밀기 법선60° 이내 후보 중 minimax 면을 매번 선택한다. 중점 기준점·높이 state/success gate를 제거하고 같은 면 state>0.9+기존0.5N 반력 성공으로 연결했다. CONTACT progress는 점/높이 조건 없는 기존 root→상자 접근점수, AT/paired포화·RSI·AMP는 유지한다. 기존 checkpoint 직접 혼용 금지·기존 학습은 유지했다.
- 전달 문서·공통 면 명세·config/structure를 갱신했다. CPU 공통면/기존 중점 회귀44개 통과(높이 무관·가장자리·공통면·60° 경계·회전·반력·후보 없음0점·checkpoint 거부). GPU6 두 설정 각64평가환경·12step, reset/부분reset·634D·보상 finite·공통면 state 일치 확인. 본학습/PPO·장기 성능은 미검증. 결과: 사본 `output_etc/check/push_common_face_xy/`.

## 2026-10-20

### 앞으로의 PUSH RSI 구성·검수 기준 문서

- `markdowns/push_rsi_guidelines.md`에 현재 GPU6 두 phase 실행을 기준으로 원본 관절/속도 보존·손중점≤1m와 양팔 밀기 구간 동시 선별·loco50% 유지·실제 상자/collider/발 메시 검수·샘플러/초기 물리 검사·cache/checkpoint 호환 기준을 정리했다. 손 보상 목표와 RSI 높이 제한, 정적 배치와 동적 성능을 구분한다.
- 루트/사본 `AGENTS.md` 및 config/structure·기존 RSI 명세에 문서를 연결했다. 사용자 요청에 따른 문서/작업 지침만 변경했으며 현재 실행 YAML·cache·학습은 변경하지 않았다. 코드/YAML과 조건을 대조했고 문서 링크·경로·diff를 검증했다. 추가 학습/시뮬레이션은 실행하지 않았다.

### 선별 PUSH RSI 두 설정의 새 학습 명령

- `stage1_push_depth180_h100_rsi1m_at5deg_phase`·`stage1_push_depth180_h080_rsi1m_phase` env/train·train/test/VNC·독립 output을 추가했다. 기존 설정 대비 새 선별 RSI cache/SHA만 변경했다. 사용자 최종 지시대로 두 설정 모두 거리 커리큘럼 없이 목표1~4m·loco50%·PUSH50%·시작1~5m를 유지한다. 각각 손 목표1.0m/AT5° 및0.8m/AT 제한 없음이다. 기존 학습은 유지했다.
- GPU6 각64평가환경·4096agent reset에서 새5모션만 선택·RSI높이≤1m·원본관절/속도·무관통·부분reset·634D·12step finite 확인. 실제 RSI비율0.493·목표거리1.0006~3.9988m, 각 손목표1.0/0.8m 확인. 확인 경로 `output_etc/check/push_phase_launch/`. 설정 차이/hash·6개 셸문법·diff 통과. 새 본학습/PPO는 실행하지 않았다.

### PUSH RSI: 1m 이하에서 양팔 밀기 자세만 재선별

- 사용자 허용 loco 접근 시작은 유지하고 PUSH의 팔 내린 준비/정리 자세를 제외하는 CPU 선별기·별도 캐시/구간 manifest·회귀 검사를 추가했다. 손 중점≤1m와 원본 관절/속도는 유지한다. 중간의 높이 무제한 비교는 최종 적용 대상에서 제외했다.
- 21개 중5개(0/3/4/7/8),1196프레임×7크기=8372조합. 실제 배치 전수 무관통·손 면 투영 이탈0·원본 DOF 오차0, 손 높이0.945716~0.972829m. 문제 clip15/0.067s 제외·허용 clip8/3.933s 유지. 5구간 시작/중간/끝15장 검수와 실제 발 메시 측정·40초 영상 저장.
- CPU20개 통과. GPU6 별도35평가환경·대표210agent조건·0.1초 PD screen210/210통과(fall/nonfinite0), 영상 전체 디코딩·diff 통과. 초기 접촉/장기 전도 해결을 보장하지 않는다. 기존 학습/활성 YAML/cache는 미변경이며 새 캐시의 학습 적용은 별도 계약/이관 작업이다. 검증: 사본 `output_etc/check/push_rsi_phase_selection_20261020/`, 상세 `markdowns/reference/push_rsi_phase_selection.md`.

### PUSH 목표 거리 1~5→1~7→1~10m 커리큘럼 준비

- 사용자 지정대로 두 RSI≤1m 실험에 별도 `*_goal_curriculum` env/train·train/test/VNC를 추가했다. epoch0/1000/3000 기준이며 새 reset부터 적용한다. 10m 경로를 위해 작업 구역 중심±16m·envSpacing32를 사용한다. 원래 손높이/AT/물성/RSI/보상은 유지한다.
- 학습 첫 reset 전·각 rollout과 평가 checkpoint epoch를 연결했다. 원래1~4m checkpoint의 학습 상태를 보존하는 명시적 이관 도구와 500회 사본 두 개를 준비했다. 기존 두 본학습은 중단/전환하지 않았다.
- CPU45개 통과. GPU6 각64검증환경·32reset에서 거리 단계/RSI/634D·12step 확인,0.8m 이관본2평가환경·12step×2 재로딩 rc0. 각2048환경 확인resume500→501회 rc0. 검증 산출물은 사본 `output_etc/check/push_goal_curriculum/`. 장기 커리큘럼 성능은 미검증이다.


### PUSH 학습 모션 로컬 반출 묶음

- 현재 두 Y1.8m 학습의 PUSH21개·보행4개 원본 NPY, manifest·≤1m RSI cache·두 env/train 설정을 `output_etc/exports/stage1_push_motion_data_20261020.tar.gz`로 묶었다(약4.43MiB). 실제 파일을 담아 심링크 의존성을 제거했고 31파일 SHA256·archive33항목 검증 통과. 학습 코드/asset/checkpoint는 포함하지 않았으며 원본 데이터는 변경하지 않았다.

### Y1.8m·20kg·원본 RSI 높이≤1m 선별 두 실험

- 사용자 요청으로 `stage1_push_depth180_h100_rsi1m_at5deg`(손 목표1.0m·AT 성공≤5°)와 `stage1_push_depth180_h080_rsi1m`(손 목표0.8m·AT 기울기 제한 없음)의 env/train·train/test/VNC·output을 분리했다. X1.2~1.5m·Y1.8m·Z1.3m·20kg, 상자→목표1~4m·사람→상자시작1~5m, 완화 state>0.9+양손각각/상자0.5N·AT/paired 포화를 유지한다.
- 원본 모션 관절·속도·손높이를 수정하지 않고 RSI pool만 선별했다. 실제 MotionLib 보간·기존 safe 배치를 반영해 원본 및 배치 후 손 중점≤1m 조건을 적용했다. 초기 경계 검사의 1.000036m 사례는 1mm 선별 여유로 해당 프레임을 제외했다. 7크기×1833=12831 size-frame·21/21모션 유지, 원본 높이0.867878~0.998610m/배치 후0.867878~0.998943m. 원본 cache 불변·별도 cache/SHA·checkpoint 계약 확인.
- CPU84개 통과. GPU6 각64검증환경4096agent reset에서 높이 제한·원본 DOF position/velocity·강체 Z 이동만 적용·634D·무관통/부분reset·12step 확인. 각2048환경 확인학습2epoch/frame262144 rc0·669tensor finite, 전용test2환경·RSI100%·12step×2 재로딩 rc0. 새 본학습은 미실행. 검증은 사본 `output_etc/check/push_depth180_rsi_le1m_selection_v2/` 및 `<실험명>_check/`. 문서 명령/경로·Python/셸문법·diff·원본387파일 검사 통과.

### Y1.8m PUSH 손 목표 높이1.0m 고정

- 사용자 선택으로 `stage1_push_depth180_safe_h100_relaxed` env/train·train/test/VNC·output을 분리했다. 손목표만1.0m고정이며Y1.8m·X1.2~1.5m·Z1.3m·20kg·완화state·성공state>0.9+양손각각/상자0.5N·AT/paired포화를유지한다. 기존0.65m/AMP비교설정·cache와기존실행은보존한다. 원본RSI관절을1m로retarget하지않는다.
- CPU68개통과. GPU6 64검증환경4096agent reset에서실제목표높이1.0m·실제20kg·관절position/velocity보존·초기관통방지·부분reset·634D·12stepfinite통과. 명령/경로·셸문법/diff확인. 본학습/PPO/정책성능미실행. 검증 `output_etc/check/stage1_push_depth180_safe_h100_relaxed_reset/`.

### Y1.8m 높은 손 위치 20/25kg 전도 비교

- 비학습 `analyze_push_depth180_high.sh`·보고서 도구를 추가했다. 기존 접촉체/실제humanoid 분석에 Y1.8m·정방형대조·질량·원본AMP높이 옵션을 연결했다. 접촉체는 각질량616조건(7개X·7높이·2하중·3배치/무구동대조 포함), 실제humanoid는2940조건(21AMP선택자세·4고정높이/원본높이·2하중)이며 초기 collider/바닥 보정을 적용했다.
- GPU6: 두질량모두1.0m높이에서42조건무전도·약0.82~0.85m이동. 최대기울기20/25kg은1.10m에서3.95°/3.44°,1.12m에서34.79°/38.83°,1.15m에서각21/42조건45°초과. 높은위치의실제humanoid유효안정손밀기는입증하지못했다. 각질량실측·접촉체trace각도/최종변위재검증통과. 하중한도는질량비례이므로동일절대힘질량단독대조는아니다. [20/25kg 비교](output_etc/check/push_depth180_high_25kg_20261020/COMPARE_20_25.md). 학습설정질량/기존프로세스는변경하지않았다.

### Y1.8m 손 높이0.65m/AMP 비교·state 기반 성공 복구

- 별도 `stage1_push_depth180_safe_{h065,hamp}_relaxed` env/train·train/test/VNC·output을 추가했다. Y1.8/Z1.3m·X1.2~1.5m·20kg·접근1~5m·목표1~4m·native관절/RSI/cache·AT/paired포화를 유지한다. 사용자수정에따라0.80m초안을원본AMP손중점높이로교체하고별도수평/높이성공gate를제거했다. state감쇠만완화하고CONTACT자체성공은state>0.9+양손각각/상자반력≥0.5N이다. AMP목표는episode별고정하며정책손을따라가지않는다.
- CPU67개통과. GPU6 각64검증환경4096agent reset·원본관절position/velocity보존·634D·초기관통/부분reset·12stepfinite통과. 추가4096reset의AMP목표전체범위0.8683~1.1695m 확인(마지막reset만의0.9475~1.1228m와구분). 새본학습/PPO·정책성능은미검증. 문서명령/링크·셸/Python문법·diff·원본387파일검사통과,기존세학습생존확인. 검증은사본 `output_etc/check/stage1_push_depth180_safe_{h065,hamp}_relaxed_reset/` 및 `*_hamp_relaxed_height_range/`.

### safe PUSH 3종 CONTACT 0.5N 조건 복구

- 사용자 요청으로 `stage1_push_midpoint_safe`, `stage1_push_depth180_safe`, `stage1_push_midpoint_safe_at5deg` 기본 YAML에 양손 각각/상자 면 법선 반력≥0.5N을 복구했다. 기존 중점 점수>0.9와 같은 물리 substep에서 검사·제어 step 내 누적하며 reset 접촉은 제거한다. 연속 state·AT/paired 포화·RSI/cache는 유지한다.
- CPU54개 통과(0.5N 포함/미만·각 손/상자·힘 방향/yaw·위치/힘 동시성·부분reset·기존 AT/중점/reset 회귀). 새 기본 설정과 기존 no-force checkpoint의 직접 resume 계약은 분리된다. 기존 학습은 중단/재시작하지 않았으며 실제 GPU 시뮬레이션·새 학습 성능은 미검증이다. 기본 명령·적용 범위를 config/명세에 기록했다.

### GPU5 중점 safe PUSH AT 성공 기울기 5도

- `stage1_push_midpoint_safe_at5deg` env/train·train/test/VNC·output을 분리했다. 기존 GPU5 중점0.65m·X=Y/20kg·목표1~4m·native RSI/cache·초기관통 방지를 유지하고 AT 성공에만 수직축 기울기≤5°(yaw 제외)를 추가했다. AT 기반 paired 포화도 해당 조건을 따르며 연속 state/progress 감쇠·초과 종료는 추가하지 않았다. 기존 checkpoint 직접 혼용 불가·새 scratch 명령이며 기존 학습은 중단하지 않았다.
- CPU32개(5° 포함/초과·yaw·부분reset·거리/높이·paired포화·기존중점/reset회귀) 및 GPU6 64검증환경/12step 검사 rc0. 셸문법·문서 명령/링크·diff 확인. 본학습·PPO·장기성능은 미실행. 검증은 사본 `output_etc/check/stage1_push_midpoint_safe_at5deg_reset/`.

## 2026-10-19

### GPU5/6 PUSH 발메시·바닥 초기 관통 수정

- 몸체중심만검사하던일반시작에서발끝mesh관통을놓친원인을확인했다. 동일환경/seed의별도초기학습96step재현에서GPU5 45°전도47건/GPU6 Y1.8설정15건,각8개대표사례발정점초기상자관통확인. GPU5발끝만정리한진단대조47→0, collider기반첫수정의남은1건은초기발바닥관통2.4cm를추가확인했다. 실행중현재정책직접재생이나장기전도율비교는아니다.
- 전용 `stage1_push_midpoint_safe`(GPU5/X=Y가변/시작1~2m), `stage1_push_depth180_safe`(GPU6/X가변/Y1.8/시작1~5m) env/train·train/test/VNC·cache를분리했다. 실제MJCF sphere/capsule반경/offset·전체foot mesh bounds를반영해상자2cm여유로거리재추출·바닥2mm여유로전체사람Z배치보정; native관절/velocity/clip추출·중점CONTACT/원본AT포화/목표1~4m/물성은유지한다. asset+mesh SHA/reset계약을cache/checkpoint에고정한다. 기존학습/설정은보존하며기존checkpoint직접resume불가다.
- CPU41개통과. 두캐시각7×8006=56042조합전부유효. 최종각2048환경3epoch/frame393216 rc0·각669tensorfinite·전용test2환경12step×2재로딩rc0. 초기학습96step에서45°/90°전도두설정모두0; GPU5 21592/GPU6 23156 agent reset collider상자관통0·잔여바닥보정최대2.6e-8m. 거리각1~2m/1~5m유지확인. 장기정책의전도차단은미보장이다.
- [비교보고서](output_etc/check/push_safe_start_report/REPORT.md), [수정·명령](markdowns/reference/push_safe_start.md). 원본387파일·셸문법/diff·config별명령중복검사통과, 기존학습생존확인. 실행목록중복을정리해각config의train/test/VNC를한줄씩복구했다.

### PUSH 깊이Y1.8m·X가변·일반시작1~5m 전용 설정

- 사용자 요청으로 `stage1_push_depth180_native_rsi` env/train·train/test/VNC·전용cache를 분리했다. X만1.20~1.50m/5cm가변·Y1.80/Z1.30m고정·20kg·4상자이며 손목표는−Y면중앙, 이동목표는+Y1~4m다. 일반시작은원본HOLDING의중심거리기준을참고하되사용자지정1~5m로설정하고실제크기+4cm겹침거부를유지했다. 중점CONTACT/state>0.9·원본AT/포화·native RSI/원본관절은유지하며기존학습/config/checkpoint와분리했다.
- RSI크기index를cache축별범위로일반화하여Y1.8m를지원한다. CPU34개통과. GPU6: 7×8006=56042조합전부유효·초기손224168건무겹침·0.1초물리1345008건손중심내부0/표면최대2.416mm. 64환경4096agent reset에서X7종/Y1.8/Z1.3·20kg·−Y접촉/+Y목표·634D/부분reset/12step통과. 일반시작2126건의손구체/몸체중심겹침0·root중심거리1.035~4.998m·root-뒷면여유최소0.135m. 이는손구체/몸체중심기준이며전신메시무관통보장은아니다.
- 최종1~5m설정2048환경확인학습2epoch/frame262144 rc0·669tensorfinite·전용test2환경12step×2재로딩rc0. 검증은사본 `output_etc/check/stage1_push_depth180_native_rsi_{build,loco1to5_reset,loco1to5_train,loco1to5_eval}/`. 방향검사는짧은RSI잔여거리로세계좌표반올림이증폭되지않도록변위벡터로대조했다. 원본387파일·셸문법/diff통과. 본학습·GUI·장기전도율개선은미검증이다.

### 중점 PUSH X=Y 연동·목표1~4m·손 관통 감사

- 사용자 요청으로 `stage1_push_midpoint_native_rsi`만 X=Y1.20~1.50m/5cm·Z1.30m·20kg·초기목표1~4m로 변경했다. 실행중native RSI와동일한50%PUSH/원가중21모션/70%후반30%/원본관절보존을유지하고7크기별cache를새파일로분리했다. 독립XY/목표1~5m checkpoint와직접혼용불가. 공통cache생성기는실제config의크기profile을열거하도록수정했다.
- CPU32개 통과. GPU6: 7×8006=56042조합 전부유효. RSI초기손4cm구체224168건에서겹침/중심내부0건; 기존0.1초물리검사1345008건에서중심내부0건·표면겹침176280건/최대2.361mm. 이는초기물리감사이며장기정책관통보장은아니다. native기존6cm중심관통필터/8cm전신뒤이동은유지했다.
- 64환경4096agent reset에서X=Y7종·실제20kg·초기거리복원1.0005~4.0000m·RSI49.66%/후반69.57%·21/21모션/원본DOF보존·634D/부분reset/12step통과. 2048환경확인학습2epoch/frame262144 rc0·669tensorfinite, 전용test2환경12step×2재로딩rc0. 원본387파일/셸문법/diff통과·기존학습생존확인. 출력은사본 `output_etc/check/stage1_push_midpoint_native_rsi_xy_tied_{build,reset,train,eval}/`. 본학습/GUI/장기성능미검증.

### 중점만 CONTACT·20kg 전용 Stage1 PUSH

- 사용자 요청으로 `stage1_push_midpoint_native_rsi` env/train·train/test/VNC·전용RSI cache를 분리했다. 양손중점-상자기준점 오차만 state에 사용하고 원래state>0.9로 CONTACT 자체성공을 판정한다. 개별손면거리/힘/별도높이gate는 제거하고 원본AT·현재AT성공시CONTACT까지paired포화·기존가중치/AMP는 유지한다. X/Y1.20~1.50m·Z1.30m고정·20kg이며 이전CONTACT checkpoint와혼용불가다.
- CPU30개 통과. GPU6: 49크기×8006frame=392294조합 전부RSI통과·빈크기0·전용cache/SHA설치. 64환경4096agent reset·실제20kg·Z고정·634D원관측동일·21/21모션·부분reset·12stepfinite통과. 2048환경확인학습2epoch/frame262144 rc0·669tensorfinite, 전용test2환경12step×2재로딩rc0. 셸문법·원본387파일불변확인. 검증은 사본 `output_etc/check/stage1_push_midpoint_native_rsi_{20kg_build,reset,train,eval}/`. 본학습·GUI·장기성공률은미검증이다.

### 직육면체 PUSH 계약 검사와 25kg/Z고정 RSI cache 갱신 경로 수정

- strict PUSH 물리 계약을 X/Y 1.20~1.50m·Z 1.30m·25kg으로 맞추고, 실제 상자 크기 샘플링도 Z를 고정하도록 했다. 이 설정용 49 profile RSI cache 재생성 및 SHA 갱신 스크립트를 추가했다.
- CPU 설정/크기 샘플링 테스트와 GPU6 reset 검증은 아직 실행하지 않았다. GPU6은 기존 학습 프로세스가 사용 중이므로 새 cache 생성은 실행하지 않았다. cache를 재생성하기 전에는 25kg config의 native RSI cache metadata가 불일치한다.

### 직육면체 PUSH 크기와 질량 고정값 변경

- `stage1_push_rect_native_rsi`의 X/Y는 1.20~1.50m 가변으로 유지하고 Z는 1.30m 고정, 상자 질량은 25kg으로 설정했다. 기존 실행 중인 학습에는 적용되지 않고 새 실행부터 반영된다.
- YAML과 실행/참고 문서 값을 확인했다. GPU 학습이나 물리 검증은 실행하지 않았다.

### 직육면체 PUSH 질량을 25kg으로 변경

- `stage1_push_rect_native_rsi`의 상자 질량만 20kg에서25kg으로 변경했다. 이미 실행 중인 학습에는 적용되지 않으며, 변경 후 새로 실행하는 프로세스부터 적용된다.
- YAML 값과 실행 명령/설정 문서를 확인했다. GPU 학습이나 물리 재검증은 실행하지 않았다.

### 직육면체 PUSH·SIT/CARRY native RSI

- 새 `stage1_push_rect_native_rsi`는 크기별343 profile에서21개 PUSH 모션의 모든 유효 물리 프레임을 pool로 저장한다. MotionLib 원가중으로 모션을 선택하고, 70%는 유효 프레임 후반30%·30%는 전체 유효 프레임에서 추출한다. 0.1초 기존 SIT/CARRY 물리/겹침 필터만 적용하고 0.65m 목표 높이/2°를 초기 RSI 필터로 쓰지 않으며 관절/손높이 retarget을 제거했다.
- GPU6 검사: 343×21×8006=2,746,058 size-frame 전부 유효·빈 크기0. 64환경 reset에서21/21모션 각74~107회·1,998 RSI start·후반 추출70.8%·1,550 distinct clip/time·관절position/velocity 원본과완전일치,4상자·634D/부분reset/12step finite통과. CPU20개 통과, cache/hash/셸검사 통과. 학습 정책 성능·장기무기울기는 미검증. [설정·검증 명령](ports/ksh_stage1_stage2_base/push/markdowns/reference/stage1_push_rect_native_rsi.md).


### Stage1 SIT/CARRY native RSI를 가변 상자 PUSH에 연결

- `stage1_push_rect_native_rsi`를 추가했다.343크기×21 PUSH 모션×8006프레임=2,746,058 조합에서 기존 SizeRsiCache와 동일한0.1초물리/geometry검사를 적용했고전조합통과·empty size0. MotionLib원가중·유효frame전체/후반30%선택(후반추출70%)·원본DOF position/velocity그대로. 점CONTACT/물성/목표는직전가변점설정유지.
- GPU6·64환경4096agent/12step reset에서21/21clip샘플·각clip74~107회·late70.8%·1,550개고유clip/time·원본DOF완전일치·실제20kg/마찰0.6·XYZ 180조합·부분reset·목표고정·finite통과. CPU20개통과. 원본387파일불변. 학습본설정/정책성공률/장기무기울기는미검증. 명세 `markdowns/reference/stage1_push_rect_native_rsi.md`; 출력`output_etc/check/stage1_push_rect_native_rsi_{build,reset}/`.


### 직육면체 PUSH·중점 고정점·RSI 재선별

- `stage1_push_rect_point` env/train·train/test/VNC·전용bank 추가. 실제XYZ독립1.2~1.5m/5cm격자·크기별동일20kg·목표로컬점(0,-Y/2,0.65-Z/2)·중점높이±3cm·힘gate없음. 기존AT/입력/AMP유지·2°는검증/진단만. [명세](markdowns/reference/stage1_push_rect_point.md).
- GPU6 RSI7546초기화/343크기/12control step0.4초검사에서3후보만전수통과·bank해시고정. 채택분최대각<0.0506°/첫step높이오차<8.5mm. CPU17개·64환경4096agent reset/부분reset/634D원관측동일/실제20kg·12stepfinite통과. 2048환경학습2epoch/frame262144 rc0·669tensorfinite/모델165tensor,전용test2환경12step×2재로딩rc0. 검증출력 `output_etc/check/stage1_push_rect_point_{rsi,reset,train,eval}/`. 원본387파일불변·셸문법통과. 기존학습유지·본학습/GUI/장기성공률은미검증.


### 직육면체·동일질량·중점 접촉 안정성 분석

- `analyze_push_rect_stability.sh`와접촉체/실제humanoid분석·집계·측정CPU테스트추가. XYZ각1.2~1.5m·질량10~30kg·중점목표, 전구간최대2°우선. 학습YAML/reward/RSI/기존학습불변. [명세·결과](markdowns/reference/push_rect_stability.md).
- GPU6: 저점접촉체4050/4050회2°이하·0.5m이상, 고점일부95.226°전도. 고정0.65m/20kg·343크기×6조건2058/2058회통과(최대0.0873°·0.818~0.849m), 대조343회통과. 실제21AMP선택자세PD28350회는유효손밀기+2°/0.1m기준7회통과하여최적질량/안정학습정책은미검증. 결과 `output_etc/check/push_rect_stability_20261019/`. trace재계산·질량감사통과; CPU측정3개·Python/셸문법·원본387파일불변검사통과. wrapper전체일괄실행은미실행이며각검사구성요소는개별완료했다.

### CONTACT 힘 조건 없는 비교 설정

- `stage1_push_no_force` env/train·train/test/VNC를 추가하고revision2에서손/상자힘gate만생략했다. 양손면거리·중점XY·높이와나머지보상/AT·목표/물성/RSI/AMP는유지한다. 기존힘버전·checkpoint계약/output은분리했다.
- CPU44개·GPU6/64환경4096agent reset·부분reset/634D관측동일·목표고정·12stepfinite rc0(`output_etc/check/stage1_push_no_force_reset/`). 셸문법/diff확인. PPO학습·새checkpoint재로딩·GUI/장기성공률은미실행이다.

### Stage1 구조 유지·PUSH100% 별도 추가

- `stage1_push` env/train·train/test/VNC와 `HumanoidMAStage1Push`를 추가했다. 기존2명·4물체30D/8토큰·634D·네트워크·AT를 유지하고,1.3m/20kg/마찰0.6·초기면법선1~5m·기존양손CONTACT/중점높이·11자세RSI·PUSH AMP를 분리해 연결했다. 원본HOLDING과 기존 config는 보존하며 checkpoint 교차로딩을 거부한다.
- CPU40개 통과. GPU6·64환경4096agent reset: 원관측동일·RSI49.15%·부분reset/목표고정·12stepfinite·실제20kg/마찰0.6. 2048환경짧은학습2epoch/frame262144 rc0·669tensorfinite/모델165tensor,전용test2환경12step×2재로딩rc0(`output_etc/check/stage1_push_{reset,train_final,eval_final}/`). 셸문법/diff·원본upstream387개와기존config/wrapper54개해시검사통과. 본학습/GUI/장기성공률·Stage2 PUSH·전후보PD재검사는미실행이다.


### 공유 받침 토큰 예산과 distractor 유지

- 두 공유 실험의 X/Y를 1.2~1.6m로 변경했다(Z 0.35~0.5m 유지). 기본 분할은 2×2, 구역 한 변은 0.6~0.8m다. train/test/VNC의 마지막 물체 인자는 총 obj 토큰 수이며 기본·학습 최소는 12다. 인원·공유 크기 상한에 따른 최소값보다 적으면 오류를 낸다.
- 과제용 물체·공유 구역을 빈자리 없이 모으고 미할당 일반 상자를 실제 distractor로 추가해 지정 토큰 수를 채운다. 개인 받침 전용 빈자리와 관측 padding mask를 제거했으며 task 인덱스·물리 ID 연결을 갱신했다. 기존 Stage2의 반경 4.5m 초기 배치를 사용하고, 과제용 물체 위치 조정 후 겹치는 distractor만 다시 배치한다. 큰 받침·24토큰 검사의 전체 장면 재시도 한도 실패를 이 과정으로 해결했다.
- CPU 회귀 129개 및 마지막 전이 fixture 수정 후 공유 테스트 20개 통과. GPU5에서 전체 성공 12토큰·구역 성공 16토큰 각각 4환경·10구성·16step, 토큰/실제 distractor 수·영역 면적·task 물리 연결·부분 reset·AMP finite·가중치 불변을 확인했다. [12토큰](output_etc/check/shared_support_tokens_20261019/full12/validation.json), [16토큰](output_etc/check/shared_support_tokens_20261019/region16/validation.json).
- 전용 test wrapper로 6명·24토큰·2환경의 checkpoint 재로드/16step 추론, 별도 XY 1.2~2.4m·24토큰·4환경의 3구성/8step도 확인했다. [6명 평가](output_etc/check/shared_support_tokens_20261019/full6person24/metrics/edge_context_20261019_103443.json), [큰 받침 검사](output_etc/check/shared_support_tokens_20261019/large24/validation.json). 본학습·1024환경 메모리·장기 성능은 검증하지 않았고 기존 변경·실행 프로세스는 보존했다.
- 공유 checkpoint 계약을 version 2로 올렸다. 이전 padding 방식의 공유 checkpoint는 직접 resume할 수 없으며 Stage1 checkpoint 전이는 유지한다. [실행 명령](markdowns/config.md), [토큰·배치 계약](markdowns/reference/shared_support.md)을 갱신했다.

## 2026-10-18

### 공유 받침의 전체/구역 성공 비교

- 현재 Stage2 학습 설정을 기반으로 `stage2_shared_full_success`, `stage2_shared_region_success`의 환경·학습 YAML과 train/test/VNC를 추가했다. 기본 GPU5·4명·1024환경, 순차 coupled/NONE 샘플링, 공동운반 제외, 전 과제 loco 초기화와 Stage1 AMP를 적용했다. 기존 Stage2 설정은 보존했다.
- 공유 물리 받침 하나를 고정 로컬 구역 토큰으로 표현하고 padding을 마스킹한다. 물리 ID로 coupled를 판정하며 사람별 최근접 구역을 state/progress에 연결한다. 두 실험은 공유 과제 성공 XY 영역만 다르고 실제 높이·충돌은 원본 받침을 따른다. 상세 계약은 [공유 받침 참고](markdowns/reference/shared_support.md)에 있다.
- CPU 121개 통과. GPU5에서 두 버전 각각 4환경·10구성·각16step 추론, 부분 reset, AMP finite, Stage1 30500 strict 전이·가중치 불변을 확인했다. loco 평가가 AMP expert 선택까지 제한해 데모 일부를 채우지 않던 경로를 수정했다. [전체 성공 검사](output_etc/check/shared_support/full/validation.json), [구역 성공 검사](output_etc/check/shared_support/region/validation.json).
- 전용 test wrapper로 전체 성공 4명·4환경 및 구역 성공 6명·2환경·XY 2.4~3.2m 받침의 checkpoint 재로드/16step 평가를 확인했다. [전체 평가](output_etc/check/shared_support/full_wrapper/metrics/edge_context_20261018_233022.json), [확장 평가](output_etc/check/shared_support/region_6person_wrapper/metrics/edge_context_20261018_233105.json). optimizer 학습·1024환경 메모리·학습 성능·실제 viewer 화면은 검증하지 않았다. 기존 미커밋 변경과 실행 프로세스는 보존했다.

### Stage1 fixed45000 기반 Stage2 미학습 viewer

- `stage1/Stage1V2Fixed_00045000.pth`의165개 tensor를 strict 전이하고,2환경·단독/공동 AT/ON_TOP5구성의 각16step 추론·부분 reset·AMP finite·가중치 불변을 확인했다. 검증용 `Stage2Transfer.pth`를 생성했으며 optimizer 학습은 실행하지 않았다.
- GPU5에서4명·1환경·8상자·고정 선반·15cm 앵커의 무작위 과제 viewer를 noVNC6081/:103으로 띄워 실제 화면과 움직임을 확인했다. 별도6080 viewer와 기존 프로세스는 보존했다. [검증·실행 정보](output_etc/check/stage2_from_stage1_45000/validation.json), [화면](output_etc/check/stage2_from_stage1_45000/viewer.png). 추가 협력 모듈은 항등 초기화 상태이며 공동 운반 성공률은 검증하지 않았다.
- Stack 카메라가 장면 밖에 격리한 물리 bank 상자까지 포함해 화면이 멀어지는 문제를 고쳤다. 현재 논리 상자만 카메라 범위에 포함하며135개 CPU 테스트와 실제 viewer로 확인했다. `run-gui.sh`는 사용 중인 VNC 포트도 건너뛰고 서비스 로그를 `output_etc/check/`에 저장한다. 재실행·접속 방법은 [USAGE](markdowns/USAGE.md#stage1-가중치로-미학습-stage2-viewer-보기)에 기록했다.

## 2026-10-17

### 이전 앵커 실험 자료 정리

- 사용자 요청에 따라 이전 앵커 전용 캐시·검증 폴더·임시 생성 보고서·비교 영상과 문서의 참조를 삭제했다. 현재15cm 검토 페이지와 재검증 스크립트도 현재 데이터만 사용하도록 정리했다.
- 관련 파일11734개를 삭제했다. 현재15cm 캐시8264개 파일의 전체 해시가 정리 전후 동일하며, CPU134개와 GPU5의4131개 RSI pool 로딩 검사가 추가 생성 없이 통과했다. 검토 영상29개의 상태와 링크를 확인했으며 학습·새 물리 screening은 실행하지 않았다. [정리 검증](output_etc/check/stage2_cache_cleanup/validation.json).

### 공동 앵커·자동 RSI를15cm로 통일

- 사용자 결정에 따라 공동 RSI 배치·GTA·HOLDING 리워드의 앵커를 Y 끝면에서15cm 안쪽으로 통일했다. 기본 캐시는 `joint_carry/stage2_size_rsi/inset15_single`이다.
- 실행 중 캐시 누락·구버전 조건이면 1인 원본 기반의15cm 생성기를 명시적으로 호출한다. 유효한 캐시는 재사용한다. CPU134개와 GPU의 실제 누락 캐시 생성→재로드 검사가 통과했다.
- GPU5에서 1인 원본 기반 전체1377크기×세 모션을 생성했다. NPZ4131개·1,450,746,074bytes(약1.35GiB), 전체 최소 유효 프레임은 pickUp100/carryWith114/putDown129다. 누락·빈 pool·hash/finite·관절 범위 오류0이고4131개 pool의 실제 GPU 로더 샘플링도 추가 생성 없이 통과했다. 전체 생성·물리 screening은 약27분18초이며, 기존 대표 캐시를 일부 재사용했다. 전체 생성의 worker5크기는 재시도 후 통과했다.
- 4환경의 두 pair AT/ON_TOP carry 초기 자기 앵커 phi 최솟값0.99724와2환경의 단독/공동 template4구성·각16step·부분 reset·AMP finite·가중치 불변을 확인했다. Template 검사는 작은 단독 상자 fixture다. 표준 test wrapper의4환경 carryWith·16step strict 재로드도 통과했다. 원본 checkpoint 불변을 확인했다. 학습·장기 운반 성공률은 검사하지 않았다.
- [전체 검사](output_etc/check/stage2_anchor15/full1377_audit.json), [검증 요약](output_etc/check/stage2_anchor15/validation.json), [15cm27크기 영상](output_etc/check/stage2_anchor15/index.html). 영상은 native asset 자세 재생이며 정책/물리 rollout은 아니다.

### 공동 RSI 범위·ON_TOP 초기화 확정과 native 영상

- stage2 공동 크기를 X0.2~0.6/Y0.4~1.2/Z0.2~0.6m로 변경했다. 단독/공동 ON_TOP 초기화는50/10/40/0으로 확정하고 AT는50/10/30/10을 유지했다. 공동 reset이 AT/ON_TOP별 비율을 선택하도록 고쳤으며, 공동 RSI는 payload 크기만으로 공유하고 단독 ON_TOP putDown 캐시 준비도 제외한다. ON_TOP 평가에 putDown을 요청하면 중단한다.
- 기존 제작기의 보정 결과를 native humanoid·상자로 재생하는 `joint_carry/render_size_rsi.py`를 추가했다. 원본1개·새 범위27크기를 각12.2초로 출력하고 검사 통과/제외 프레임을 표시했다. 실제 캐시의 통과 프레임과 재생 자세 일치를 확인했다. 이는 reference 자세 재생이며 정책/물리 rollout은 아니다. [비교 영상](output_etc/check/stage2_rsi_video/comparison.mp4), [27크기 재생 목록](output_etc/check/stage2_rsi_video/index.html).
- CPU129개 통과. GPU5에서 단독·공동 AT/ON_TOP의 template RSI4구성·각16step 추론·부분 reset·AMP finite·ON_TOP putDown 제외를 확인했다. 이 검사는2환경·작은 단독 상자 fixture이며 학습은 실행하지 않았다. 원본 stage1 YAML·다른 레포·기존 프로세스는 보존했다. [검증](output_etc/check/stage2_rsi_video/validation.json).

### 공동 RSI의 좁은 X 범위 사전 검사

- 공동 payload의 새 범위 X0.2~0.6/Y0.4~1.2/Z0.2~0.6m에서 각 축 최소·중간·최대의 27크기를 GPU5로 보정·물리 검사했다. 세 모션 모두 유효 프레임을 확보했으며 최소 통과 수는 pickUp100/carryWith112/putDown126이다. 관절 범위 초과0, 저장 배열 finite·SHA256 일치를 확인했다. 학습은 실행하지 않았다.
- 81개 NPZ 합계26.74MiB이며, 같은 평균으로 전체1377크기를 생성하면 약1.33GiB로 추정한다. 전체 생성과 ON_TOP 별도 RSI 검사는 실행하지 않았다. 이번 검사는 데이터 준비만 수행했으며 실행 config의 X 범위·ON_TOP 초기화 비율은 아직 변경하지 않았다. [검증·용량](output_etc/check/stage2_rsi_xnarrow/validation.json).

### fixed 기반 Stage2 공동 운반 연결

- 현재 `stage1_v2_fixed`를 기반으로 `stage2` env/train·전용 train/test/VNC를 추가했다. 4명·1024환경·8장면 상자, 단독 SIT/CLIMB/운반과 stage1 거리·선반·성공/보상 포화를 유지한다. 기존 Stage2의 랜덤 pairing·GraspView·human SA4·협력 상대 CPA 예외를 연결하며 미할당 장면 상자는 관측에 남긴다.
- fixed30500의 모델165 tensor를 strict 전이하고 actor encoder·관측 RMS를 동결한다. 기존 공동 reference를 이 checkout으로 복사해 세 모션의 다리 관절 범위·접지를 보정하고, 공동 크기별 배치/상체/팔 보정·GPU 물리 screening을 추가했다. 초기화는 loco/pickUp/carryWith/putDown=50/10/30/10, ON_TOP putDown은 실제 받침까지 검사한다. [설정·데이터 계약](markdowns/reference/stage2.md).
- CPU **129개 통과**. GPU5에서 접근 시작8구성·공동 RSI6구성·template RSI2구성의 **16-step 추론·부분 reset·AMP reference finite**, 모델/관측 RMS 불변을 확인했다. Template 검사는2환경·작은 단독 상자 fixture이며 PPO/optimizer 학습은 실행하지 않았다. Stage2 표준 test의 strict 재로드와 기존 fixed30500 표준 test도4환경·16-step headless로 정상 종료했다.
- 공동 payload27크기×세 모션과 ON_TOP 받침6조합에서 유효 RSI frame을 확보했다. 원본 carryWith/putDown의 무릎 범위 초과도 보정하여 검사 후보의 관절 범위 초과0을 확인했다. 복사한 RSI6파일·AMP12파일 hash 일치. 전체2601크기/받침 조합을 사전 생성한 것은 아니며, 필요한 profile은 사용 전 보정·검사 후 캐시한다. 빈 profile은 중단하며 크기 축소·공동 loco 대체를 하지 않는다.
- 학습·VNC 실화면·장기 성공률은 검사하지 않았다. 다른 레포와 기존 프로세스는 수정·종료하지 않았다. 모든 최종 검사 종료코드0. [검증 요약](output_etc/check/stage2_integration/validation.json), [RSI 미리보기](output_etc/check/stage2_integration/rsi_preview.png).

## 2026-10-15

### 정리 후 학습·원본 checkpoint 간단 재확인

- GPU0에서 fixed 표준 train wrapper를 **2048환경·MAX_ITERATIONS=1**로 실행해 epoch2/frame262144 checkpoint 저장·모델/514 scalar finite·물리 reset 실패0·운영 RSI cache hit를 확인했다. 이번 재확인의 학습 대상은 fixed다.
- 원본 레포의 stage1 **19,000**, v2 **8,000**, fixed **1,000** epoch 파일과 저장 config를 복사해 각 표준 test wrapper로 **32환경·32step headless 추론**을 완료했다. metadata 검사·모델 strict load 통과, 세 실행 모두 종료 코드0. 원본 checkpoint/config hash 불변이며 기존 프로세스는 변경하지 않았다. GUI/VNC·장기 성능은 검사하지 않았다. [결과·명령·로그](output_etc/check/compat_recheck_20261015/validation.json).

### 사용하지 않는 공동운반 자료 제거

- 별도 작업에서 관리하는 공동운반의 후보 생성·미리보기·보정·물리 검사·인계용 `joint_carry/` 7파일을 제거했다. 현재 세 실험과의 import·설정·실행 연결이 없음을 확인하고 README·structure의 안내와 전용 gitignore 항목도 정리했다. Stage1 공통 RSI·운반 구현과 원본 레포는 변경하지 않았다.
- CPU **119개 통과**, 실행 코드/현재 문서의 잔여 참조 없음·문서 링크·diff 확인. 학습·시뮬레이션은 실행하지 않았다. 검사 로그는 `output_etc/check/joint_carry_cleanup_20261015/cpu.log`에 있다.

### 컨피그 순서·추론 옵션 안내

- config의 표·학습·로컬 추론·VNC를 `stage1` → `stage1_v2` → `stage1_v2_fixed` 순으로 정렬하고 메인/비교 표시를 없앴다. TASK_GRAPH·EVAL_SKILLS 값과 혼합 확률 예시를 끝에 추가했다. 새 컨피그는 목록 아래에 추가하도록 AGENTS에 명시하고 README·structure도 맞췄다.
- 문서 링크·셸 문법·컨피그/명령 순서와 기존 실행 명령 불변을 확인했다. 문서만 변경했으며 학습·평가는 실행하지 않았다.

### config 문서 형식 고정

- 사용자 예시에 맞춰 config를 공통 규칙·현재 컨피그 표·학습·로컬 추론·서버 VNC 명령으로 정리했다. 실험별 설명·검증 기록을 빼고 추가 옵션·checkpoint/캐시 안내는 USAGE로 옮겼다. AGENTS에 같은 형식을 유지하도록 명시하고 structure·참고 문서의 안내를 맞췄다.
- 학습·로컬 추론·VNC 명령마다 `# 컨피그 이름`을 붙이고 컨피그 사이를 한 줄씩 띄웠다. 같은 작성 규칙을 AGENTS에 반영했다.
- 문서 링크·명령의 스크립트/인자·셸 문법·diff 확인. 문서만 변경했으며 학습·평가는 실행하지 않았다.

### 문서·확인 출력 경로 정리

- 실행 안내를 `stage1`·`stage1_v2`·`stage1_v2_fixed` 세 실험 중심으로 줄였다. 상세 정책·진단·포팅은 `markdowns/reference/`, 과거 분석·MPS 설치/실험·이전 changelog는 `markdowns/legacy/`로 옮겼다. README·config·structure·USAGE·MPS 안내와 링크를 함께 갱신했다.
- AGENTS와 세 실험의 train/test/VNC 및 직접 `run.py` 경로 검사를 맞췄다. 본학습은 `output/`, 확인 학습·평가·viewer는 **`output_etc/check/`만** 사용한다. `CHECK=1`은 기본 `MAX_ITERATIONS=1`, 양수 `MAX_ITERATIONS`도 확인 출력으로 분리한다. 구 `output/..._check`는 자동 전환하고 확인 루트 밖 지정은 거부한다.
- 기존 확인 폴더9개·검증 기록·중간 RSI `b847…`를 `output_etc/check/stage1_cleanup_20261015/`로 이동했다. 이동 시 전 파일 hash 일치, [경로표](output_etc/check/stage1_cleanup_20261015/path_migration.json) 보관. 현재 RSI `c227…`는 중간 캐시와 byte 단위로 같으며 운영용으로 유지했다. 원본 레포·기존 실행 프로세스는 변경하지 않았다.
- CPU **119개 통과**, 셸14개 문법·수정 문서 링크·diff 확인. fixed **2048환경 epoch2/frame262144**·모델/514 scalar finite·물리 reset 실패0·현재 RSI cache hit. 이동 checkpoint의 **32환경·32step headless 평가**와 구 output 경로 전환 통과. GUI/VNC 화면은 기존 NVML 불일치로 미검증이다. [검증 기록](output_etc/check/output_routing_20261015/validation.json).

### Multi-agent 실험 정리: 세 개만 유지

- 나머지29개 실험과 전용 구현·스크립트·테스트·설계 문서를 정리했다. 남은 세 실험의 env/train YAML6개와 checkpoint parameter 이름·shape·관측/reward 계약은 유지한다. 비활성 경로의 난수 소비 제거로 새 scratch run의 같은 seed 궤적까지 과거와 같다는 보장은 없다.
- 정리 당시 CPU105개·세 모델 strict load/출력/gradient/reward/metadata 대조·v2/fixed 2048환경 학습/resume·세 표준 평가·13과제 시뮬레이션·가변 인원 평가를 확인했다. stage1 전체 학습·장기 수렴·GUI 화면은 미검증이다. [상세 결과](output_etc/check/stage1_cleanup_20261015/validation.json)와 보관 changelog를 참조한다.
