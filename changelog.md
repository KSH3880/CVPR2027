# Changelog

최신 변경부터 기록한다. 현재 실행법은 [config.md](markdowns/config.md), 코드 위치는 [structure.md](markdowns/structure.md)를 참조한다. 실행 중인 GPU/PID는 이 파일에 고정하지 않고 실제 프로세스로 확인한다.

## 2026-10-23

### 단독 점/영역 비교 및 1인 PUSH 평가 파일 추가

- 업로드한 `markdowns/evaluation_single.md`에 맞춰 `evaluation/single_match/`에 독립 평가 subclass·strict frozen actor·첫episode trace·점/영역 재채점·두모델 batch/seed통계·headless/로컬/VNC6127을 추가했다. KSH73000/PUSH22500의 기존4과제는 같은 초기 장면에서 비교하며, 별도 승인한 PUSH는22500만 loco/RSI·목표2~5/5~10m·1m정사각형 구역/점20cm·도달시간/거리진척/전도율을 분리한다. [명령·판정식·범위](evaluation/single_match/README.md). 기존 학습 코드·YAML·모델과 외부원본은 수정하지 않았다.
- CPU6개·Python/셸문법·diff 통과. GPU6의12조건 각4평가환경/12step/seed0 rc0, 기존4과제의 두모델 초기root/DOF위치·속도/상자/목표/크기/배정/질량 정확일치,12개 궤적 독립재채점 및 float32/64 ever일치·PUSH 실제초기거리/목표고정 확인. 추가4환경 RSI 실제mass/전체shape마찰 설정대조, VNC6127의1환경 RSI/12step rc0. 사본 `output_etc/check/single_match_20261023/validation.json` 및 각run `completed.json`. 사본430파일/12링크 무결성·GPU5 원래PID/시작시각 유지 확인. 정식512×3seed·30초 성능평가와 학습은 실행하지 않았다.

### Pro1/JHH 기존 과제 + PUSH fine-tuning 독립 이식

- GPU5에서 실행 중인 원본 `stage1_v2_fixed_push_360_raw_cuboid`를 `ports/jhh_stage1_push_raw_cuboid/`로 독립 복사했다. 기존 과제50%/PUSH50%·raw RSI·물성·보상/YAML 유지, 저장 완료22500회 모델·73000회 교사·RSI cache 복사 및 공유 데이터12개 절대링크 연결. 사본 conda `tokenhsi`/PYTHONPATH·VNC6126·테스트 교사경로만 조정했다. [학습/재개·로컬·headless·VNC 명령](ports/jhh_stage1_push_raw_cuboid/README.md), 원본 SHA/로컬 변경 `PORT_MANIFEST.json`, 검사기 `verify_port.py`.
- CPU40개·161데이터경로·모델finite/YAML/bank계약·430파일/12링크·Python/셸문법·diff 통과. GPU6의22500회 모델: loco/RSI 각4평가환경·혼합16환경·VNC6126의1환경, 모두12step/rc0. 시작모드각8/8 및 혼합SIT4/CLIMB4/CARRY_AT5/CARRY_ON_TOP3/PUSH16 agent trial 확인. 사본 `output_etc/check/port_verification.json`. 장기성능/학습은 미검증·미실행, 원본소스 SHA와GPU5 원래PID/시작시각 유지 확인. 현재 레포의 RSI≤1m/밀기구간 기준으로 재선별한 실험은 아니다.

### GPU6 MPS 서버 복구

- 사용자 요청으로 `mps/shell.sh`의 `mps_start 6`을 실행해 기존 GPU6 전용 제어 데몬의 GPU 서버를 다시 띄웠다. GPU5/6의 제어 데몬이 서로 다른 GPU UUID와 `gpu5/pipe`·`gpu6/pipe`를 사용함을 확인했다. GPU6 짧은 CUDA heartbeat/클라이언트 등록 통과·검사 프로세스 정상 종료, GPU5 기존 MPS/두 작업 유지. 당시 GPU6 기존 두 학습 PID는 이미 종료돼 있었으며 학습은 재실행하지 않았다.

### GPU6 PUSH 실제XYZ 크기 제로샷

- 사본에평가전용 `PUSH_EVAL_SIZE_X/Y/Z`를추가했다. 기존중심을유지한 X0.9~1.8/Y1.7~2.3/Z1.0~1.6m로실제asset/bbox입력을변경하고20kg/마찰0.4·보상/학습계약을유지한다. 보행시작만허용·위치전환없음·실행중크기고정. OOD Z에서도PUSH배정을유지하도록평가assetmask를연결했다. [명령](ports/ksh_stage1_stage2_base/push/markdowns/USAGE.md#push-크기-제로샷).
- CPU12개·문법/diff통과. GPU6 Y2m17500모델4평가환경 headless/1환경 VNC6096 각12step×2회rc0; 별도4환경실제20kg/마찰0.4·bbox입력/접지·PUSH배정·RSI0·12step finite확인. 사본 `output_etc/check/push_size_zero_shot_{depth200_smoke,vnc_smoke,physics}/`. 장기성능미평가·기존학습유지.

### GPU6 PUSH 목표 고정·중간 상자 이동

- 사본 평가에 `PUSH_BOX_SWITCH_MODES=across_goal,behind_agent`를 추가했다. 기본2초 후 목표 반대편/현재 사람 뒤로 상자만 옮긴다. 목표·사람 상태 유지, 상자회전/높이/물성 보존·속도0, 실제 콜라이더/다른 상자 겹침 회피·새 접촉 면/관측·성공 이력 갱신. [명령](ports/ksh_stage1_stage2_base/push/markdowns/USAGE.md#push-중간-상자-이동-제로샷).
- CPU 방향/목표전환 회귀4개·문법/diff 통과. GPU6 두17500모델로 Y2m각4평가환경 headless12step 및 Y1.8m각1평가환경 VNC6096/90step·실제2초전환 두조건 rc0. 목표/사람 불변·겹침0·상자높이/회전보존·속도0·관측finite 확인. 사본 `output_etc/check/push_box_switch_{depth200_smoke,depth180_vnc_smoke}/`. 장기성공률 미평가·기존 학습 유지.

### GPU6 PUSH 목표 중간 전환 제로샷 viewer

- 사본 평가에 반복별90°/180° 전환 옵션을 추가했다. 기본5초 후 초기 목표 방향을 회전해 현재 상자에서2m 지점으로 목표/마커를 갱신하며 사람·상자·AMP/episode를 reset하지 않는다. 이전 목표 성공 이력을 분리한 JSON과 [로컬/headless/VNC 명령](ports/ksh_stage1_stage2_base/push/markdowns/USAGE.md#push-중간-목표-전환-제로샷)을 추가했다.
- CPU 회전좌표1개 통과. GPU6 두17500checkpoint의 짧은90°/180° 평가: Y2m각2환경 headless·Y1.8m각1환경 VNC6096, 각12step/전환3step rc0·전환장면/목표2m 확인. 사본 `output_etc/check/push_goal_switch_{depth200_smoke,depth180_vnc_smoke}/`. 장기 성공 성능은 미평가·기존 학습 유지.

## 2026-10-22

### 5과제 PUSH 평가의 RSI 옵션 적용

- `HumanoidMAFiveTask`가 평가에서 PUSH RSI 확률을 0으로 고정하던 오류를 수정했다. 평가만 `EVAL_SKILLS`/정규화한 확률을 적용하며 학습의 RSI50%·원본 캐시/보상/체크포인트는 유지한다. 실행 중 GPU6 두 학습 PID/시작 시각 유지 확인.
- GPU6 두 물성 각16평가환경에서 PUSH128/128·loco0/128 RSI, 혼합 시작·원본 DOF position/velocity·무관통·목표1~5m·부분 reset·12step finite 통과. 사본 `output_etc/check/five_task_eval_rsi_fix_depth{180,200}/eval_rsi.json`. 새 평가 검사 `tokenhsi/tests/check_five_task_eval_rsi.py`; 기존 viewer는 재실행해야 적용된다.

## 2026-10-21

### GPU6 두 1000회 checkpoint의 PUSH 100% 재개 명령

- 새 PUSH100% 두 output만 읽는 별도 TensorBoard를127.0.0.1:6012에 실행했다. HTTP200·두run인식 확인, 기존6008 유지. 재실행 명령은 아래 명세에 기록했다.
- 사본 5과제 Stage1에 한해 `push_at` 학습 preset을 허용하고 두 물성 train wrapper에 `TASK_GRAPH`를 연결했다. 기존 YAML/모델/RSI·기본 혼합 학습은 유지한다. 저장 설정에 실제 `task_graph`를 기록하며 [재개 명령](ports/ksh_stage1_stage2_base/push/markdowns/reference/push100_resume1000.md)은 별도 output을 사용한다.
- CPU 회귀8개·물성별4096/4096 agent PUSH_AT 생성·셸/Python 문법 통과. GPU6 각2048환경에서 원래1000회 저장본→1002회 짧은PPO 종료, 모델finite·optimizer/epoch/frame·push_at 저장 확인. 결과는 사본 `output_etc/check/push100_resume1000_verification.json`. 기존 두 학습은 유지했으며 PUSH100% 본학습은 실행하지 않았다.

### GPU6 5과제 PUSH 접촉·실제 밀기 거리 로그와 500회 재개

- 누락된 `edge/push_contact/*`와 PUSH 전용 CONTACT/AT/동시만족률을 추가했다. 물리 tick별 XY 이동·접촉 중 이동·부호 있는 목표거리 감소·접촉 전진량을 step/초/종료episode 단위로 기록한다. 기존 보상·성공·모델·RSI 계약은 유지한다. [지표·재개 명세](ports/ksh_stage1_stage2_base/push/markdowns/reference/push_training_metrics.md).
- CPU12개·GPU6 두물성 각64평가환경/12step·부분reset·유한값 통과. 각2048환경 짧은PPO(1~2회차)에서 실제 접촉/episode 이동량 TensorBoard 생성·목표거리약3m·finite 확인. 평가검사의 연속reset 뒤 마지막 부분setter가 앞선reset을 덮는 문제는 rollout 직전 full reset으로 검사 순서만 보정했다.
- 사용자 요청으로 두 원래 PID/시작시각·500회 번호checkpoint·전용wrapper를 고정한 감시기를 활성화했다. checkpoint epoch/필수복원상태 검증 후 대상만 종료·재개한다. 전환 상태/로그: 사본 `output_etc/check/push_metrics_resume500_20261021/`.
- 두 학습500회저장본에서재개완료. 첫새epoch501/frame65536000·각2048환경/GPU6·기존PID종료·추가지표32개finite를확인했다. 확인시Y1.8m522회/Y2m503회, 근거는 위경로 `verification.json`이다.

### GPU6 물성의 다섯 과제 두 설정·랜덤 안전 시작과 RSI 영상

- 사본 `ports/ksh_stage1_stage2_base/push/`에 `stage1_v2_fixed_five_task_depth180_20kg`·`stage1_v2_fixed_five_task_depth200_20kg_friction04` env/train·train/test/VNC·분리output을 추가했다. 5과제·입력상자4개 유지, 목표는RSI후에도상자중심1~5m, 보행은360°/독립방향·전신표면간격1~5m·네상자모두무관통검사다. 물성은기존GPU6의Y1.8m/20kg/마찰0.6 및Y2m/20kg/마찰0.4다.
- CPU47개·GPU6각64평가환경/32reset의PUSH672상태(보행338/RSI334),원본관절/속도·손중점≤1m·초기겹침0·8방향·부분reset·12step확인. 각RSI334상태0.1초PD 낙상/비유한0. 네과제전체RSI검증/장기성능/본학습은아니다. 각42.4초/424frame실제콜라이더/발메시·4상자/목표탑뷰영상 생성·전체디코딩확인. [명세·검증·영상 경로](ports/ksh_stage1_stage2_base/push/markdowns/reference/five_task_gpu6_variants.md). 본학습은실행하지않았다.

### 기존 Stage1/2 구조에 다섯 번째 PUSH_AT 통합

- `ports/ksh_stage1_stage2_base/push/`에 기존 네 과제 ID0~3을 유지하고 PUSH_AT=4를 추가한 `stage1_v2_fixed_five_task`·`stage2_shared_full_success_five_task` env/train·train/test/VNC를 추가했다. Stage2 coupled는 PUSH_AT만 공유한다. 기존 GTA/actor/critic 유지·task embedding4→5행이며, AMP는 PUSH를 네 번째 family로 추가해 one-hot3→4D·프레임132→133D/10frame1330D로 연결한다. discriminator 첫 입력 외 층/hidden 폭과 원래 YAML/실행 중 학습은 보존한다.
- 사용자 최종 지시에 따라 RSI/시뮬레이션 검사를 중단하고 코드/config를 정리했다. 최종4-family 버전 CPU43개 통과(원본 대비 task embedding/AMP 첫 입력 외 모델 parameter key/shape 동일·ID·AMP·그래프·전이), 셸/YAML/Python 문법·문서/157데이터 경로 확인. 이전 중간 버전의 GPU 결과를 최종 코드 검증으로 사용하지 않는다. 최종 시뮬레이션/본학습 미실행. [명세](ports/ksh_stage1_stage2_base/push/markdowns/reference/five_task_push_at.md).

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
- GPU6: 두질량모두1.0m높이에서42조건무전도·약0.82~0.85m이동. 최대기울기20/25kg은1.10m에서3.95°/3.44°,1.12m에서34.79°/38.83°,1.15m에서각21/42조건45°초과. 높은위치의실제humanoid유효안정손밀기는입증하지못했다. 각질량실측·접촉체trace각도/최종변위재검증통과. 하중한도는질량비례이므로동일절대힘질량단독대조는아니다. [20/25kg 비교](ports/ksh_stage1_stage2_base/push/output_etc/check/push_depth180_high_25kg_20261020/COMPARE_20_25.md). 학습설정질량/기존프로세스는변경하지않았다.

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
- [비교보고서](ports/ksh_stage1_stage2_base/push/output_etc/check/push_safe_start_report/REPORT.md), [수정·명령](ports/ksh_stage1_stage2_base/push/markdowns/reference/push_safe_start.md). 원본387파일·셸문법/diff·config별명령중복검사통과, 기존학습생존확인. 실행목록중복을정리해각config의train/test/VNC를한줄씩복구했다.

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

### 직육면체 PUSH RSI를 기존 SIT/CARRY 방식으로 전환

- 새 `stage1_push_rect_native_rsi`는 크기별343 profile에서21개 PUSH 모션의 모든 유효 물리 프레임을 pool로 저장한다. MotionLib 원가중으로 모션을 선택하고, 70%는 유효 프레임 후반30%·30%는 전체 유효 프레임에서 추출한다. 0.1초 기존 SIT/CARRY 물리/겹침 필터만 적용하고 0.65m 목표 높이/2°를 초기 RSI 필터로 쓰지 않으며 관절/손높이 retarget을 제거했다.
- GPU6 검사: 343×21×8006=2,746,058 size-frame 전부 유효·빈 크기0. 64환경 reset에서21/21모션 각74~107회·1,998 RSI start·후반 추출70.8%·1,550 distinct clip/time·관절position/velocity 원본과완전일치,4상자·634D/부분reset/12step finite통과. CPU20개 통과, cache/hash/셸검사 통과. 학습 정책 성능·장기무기울기는 미검증. [설정·검증 명령](ports/ksh_stage1_stage2_base/push/markdowns/reference/stage1_push_rect_native_rsi.md).


### 직육면체 PUSH·양손 중점0.65m 별도 구현

- 사본 `stage1_push_rect_point` env/train·train/test/VNC·별도bank 추가. XYZ독립1.20~1.50m/5cm격자를자산생성시추출·각상자20kg/마찰0.6·실제크기로reset/관측·직선1~5m·중점높이0.65m/±3cm·힘gate없음. 기존Stage1입력/AT/AMP21모션/기존설정·원본·학습유지. 2°는검증/진단기준이며새학습제한은추가하지않았다.
- GPU6 RSI343크기×11후보×2사람7546초기화·12control step/0.4초검사에서전크기통과3후보채택(채택분최대각0.0506°미만·첫step중점Z오차8.5mm미만), bank SHA256/checkpoint계약고정. CPU17개·64환경4096agent reset/634D원관측동일·180개크기/실제20kg·부분reset·12stepfinite통과. 2048환경짧은학습2epoch/frame262144 rc0/669tensorfinite·모델165tensor,전용2환경12step×2checkpoint재로딩rc0. 원본387파일불변·셸문법검사통과. [명세/명령](ports/ksh_stage1_stage2_base/push/markdowns/reference/stage1_push_rect_point.md). 검증 `push/output_etc/check/stage1_push_rect_point_{rsi,reset,train,eval}/`; 본학습·GUI·장기성공률/무기울기보장은미검증.


### 동일 질량 직육면체 PUSH·기울기 우선 물리 분석

- 사본에 XYZ독립1.2~1.5m·10~30kg·양손중점 접촉 분석/보고서/재실행wrapper를 추가했다. 학습설정·reward·bank·원본·기존학습은 변경하지 않았다. [검사 명세](ports/ksh_stage1_stage2_base/push/markdowns/reference/push_rect_stability.md).
- GPU6: 접촉체6480회+대조270회, 저점4050회전부2°/0.5m기준통과·고점일부전도95.226°. 20kg/고정높이0.65m의343크기×6조건2058회전부통과(최대0.0873°/이동0.818~0.849m)+대조343회통과. 실제21AMP선택자세PD28350회는균형등유효조건+2°/0.1m기준7회만통과하여 안정정책/최적질량은미검증이다. 결과 `push/output_etc/check/push_rect_stability_20261019/REPORT.md`. 전구간trace각도/변위재검증·실제질량감사통과; CPU측정3개·Python/셸문법·원본387파일불변검사통과. 기존학습PID생존확인.

### Stage1 PUSH CONTACT 힘 gate만 제거

- 사용자 요청으로 사본 `stage1_push_no_force` env/train·train/test/VNC/output을 분리했다. `require_contact_force=false`인revision2에서 손/상자0.5N 검사를 생략하고 기존양손면거리/중점XY18cm·높이구간·상태/접근보상·AT·1~5m·물성/RSI/AMP를 유지한다. 기존힘버전과checkpoint는직접혼용불가이며원본보관본·기존학습은유지한다.
- CPU44개(0/양/음힘독립·양손거리/중점XY/높이경계·힘버전/AT회귀·계약분리)통과. GPU6·64환경4096agent reset·관측634D동일·부분reset·고정목표·12stepfinite rc0(`push/output_etc/check/stage1_push_no_force_reset/`). 셸문법·문서/diff확인. PPO본학습·새checkpoint재로딩·GUI·장기성공률은미실행이다.

### 복사한 Stage1 구조에 PUSH100%·직선1~5m 추가

- 사용자 요청으로 `ports/ksh_stage1_stage2_base/push/`에 `stage1_push` env/train·train/test/VNC·전용CONTACT 클래스/RSI bank·모션manifest를 추가했다. 기존2명·4물체30D/8토큰·634D·GTA/네트워크·AT 보상/20cm·Z1mm 성공을 유지하며,상자1.3m/20kg/마찰0.6·초기면법선직선1~5m·중점높이구간/실제양손힘접촉을 연결했다. 담당상자2개와미할당2개를모두관측하며,11자세RSI50%/AMP33·67을사본에서만연결하고기존checkpoint와분리했다.
- CPU40개·셸문법/diff통과. GPU6·64환경4096agent reset에서원본관측완전동일/RSI49.15%·부분reset·고정목표·12stepfinite·실제4상자질량약20kg/마찰0.6 확인. 2048환경짧은학습2epoch/frame262144 rc0·669tensorfinite/모델165tensor,전용test2환경12step×2재로딩rc0. 검증은사본 `output_etc/check/stage1_push_{reset,train_final,eval_final}/`에보존했다.
- 원본upstream387파일/읽기전용과작업사본기존config·wrapper54개해시불변확인. root/사본문서갱신. 기존학습은유지하며새본학습·GUI·장기성공률·Stage2공동PUSH·전후보PD재검사는미실행이다.

### 외부 Stage 1·2 원본 고정 보관·PUSH 작업 사본

- 사용자 요청으로 `stage1_stage2_base` commit `28f02b3278f0a3e703cfa22a1e8cc376bb8ec512`의387개 파일을 `ports/ksh_stage1_stage2_base/{upstream,push}/`에 동일하게 가져왔다. upstream은 일반파일/디렉터리 읽기전용, push는 독립 사본이며 기존 루트 코드·외부 원본·학습 프로세스는 변경하지 않았다. 출처/파일별Git blob·SHA256 manifest, 검증기·작업안내를 추가했다.
- 두 사본387/387개 원본blob/해시동일·원본12개심링크문자열보존·일반파일별inode독립·upstream쓰기권한없음 확인. config/structure에 위치·실행미준비 상태를 기록했다. 데이터/체크포인트 연결·PUSH 이식·학습/시뮬레이션은 수행하지 않았다.

### 39000 상자 세 과제 마찰·자동 녹화

- 사용자 요청으로 객체 평가기에 `--box-friction` 실제 shape 설정/재조회 감사와 `--record-video` 장면별 viewer MP4/예외 마감을 추가했다. CARRY/SIT/CLIMB 각 한 장면 자동 종료 명령을 object_zero_shot/config 문서에 기재했다. KSH 원본·checkpoint 불변, 학습 설정 변경 없음.
- 관련 CPU17개·Python 문법 통과. 앞선 CARRY 마찰0.6 viewer는 step90 이상 실행 후 사용자 요청으로 종료했다. 새 녹화는 사용자가 직접 실행하도록 GPU/GUI를 실행하지 않았으며 MP4 생성·재생은 미검증이다.

### PUSH 상자 회전이 필요 없는 직선 목표

- 사용자 요청으로 `box_forward_uniform` 목표 생성과 `push_fixed130_point_zband_heightmatch_straight_goal2to5` YAML·train/test/VNC·전용bank를 추가했다. 상자 초기yaw를 유지하고 초기로컬+Y 면 법선의2~5m 균등거리에 고정목표를 둔다. 기존360°실험과 구분하며11개 높이정렬 RSI자세·보상·물성·관측·AMP는 유지한다. 목표 분포 계약이 달라 기존checkpoint 직접resume/test는 불가하며 전후보PD재검사는 미실행이다.
- CPU22개(방향/거리/회전불변/원설정차이/bank계약/기존목표회귀), 셸문법·diff검사 통과. GPU6·2048환경 `MAX_ITERATIONS=2`는3epoch/frame393216 저장 rc0·685tensor finite,전용test2환경12step×2평가 rc0(`output/push_fixed130_point_zband_heightmatch_straight_goal2to5_{check,eval_check}/`). config/structure갱신. 기존GPU6학습은 유지하며 새 본학습·GUI·장기성공률은 미실행이다.

### PUSH 원본·crop·AMP 비교 영상

- `render_push_crop_comparison.py`로 대표 `push_obstacle_180_001__A344_M`의 원본 전체·원본 유지구간·기존 AMP 변환본을3열 MP4로 생성했다.10.65초 crop 이후 원본은 계속 재생하고 나머지는 종료 표시·마지막 자세 고정으로 구분한다. 원본의 고정 Root를 제외하고 Hips 추적, CPU 전용·학습/데이터 불변이다.
- `output/push_crop_comparison/`에15.53초·1440×680·30FPS/466frame 영상·원본hash metadata·캡처 저장. 화면 확인·ffmpeg 전체디코딩·diff검사 통과. 스켈레톤 재생이며 정책/접촉 물리 영상은 아니다.

### PUSH 보상·물성·AMP 전달 문서

- 전달 문서에 공식 BONES 사용 자격/출처/재배포 조건, BVH→30FPS 제약IK·AMP expert 변환, RSI 어깨 보정·408→11후보 PD/높이 선별·초기 AMP history 처리와 절대 경로를 추가했다. 외부에 모션 파일을 그대로 전달하라는 기존 안내를 정정했다. 공식 약관·변환 코드/report·현재 bank 대조이며 추가 GPU 실행은 없다.
- 사용자 요청으로 `markdowns/push_reward_physics_amp_handoff.md`에 현재 GPU6 PUSH의 접촉/AT 보상·가중치·포화·페널티, 고정1.3m/20kg/마찰0.6·밀도, AMP33/67 설정·전체25개 모션 경로·RSI 구분을 정리했다. 정책 입력·네트워크는 제외했다.
- 실제 프로세스 인자·YAML/코드 대조, AMP 파일25/25 존재·문서 로컬 링크 확인. 실행 설정/코드 변경 및 추가 학습·시뮬레이션은 없다.

### 높이 정렬 PUSH 목표2~5m 상태보존 resume

- 사용자 요청으로 어제 GPU2 높이정렬 RSI 실험의500회에서 목표 반경만3~5→2~5m로 변경하는 전용 YAML·train/test/VNC·bank/output과 `prepare_push_heightmatch_goal2to5_resume.py`를 추가했다. 원 실험을 보존하며 같은11개후보·보상·관측·물리·AMP를 유지한다. 이관 전후685tensor 동일·epoch500/frame65536000 보존을 확인했다.
- GPU6·2048환경 짧은재개500→502/frame65798144 정상저장 rc0·685tensor finite, 전용test2평가환경12step×2재로딩 rc0(`output/push_fixed130_point_zband_heightmatch_goal2to5_{resume_check,eval_check}/`). 셸문법/diff검사 통과. config/structure에 GPU2본학습 및 평가/VNC명령을 추가했다. 본학습/GUI·장기성공률·전후보PD재검사는 미실행이다.

## 2026-10-18

### PUSH 초기 손 중점 높이 정렬 RSI

- 사용자 요청에 따라 기존408후보/어깨retarget를 보존하고 `push_fixed130_point_zband_heightmatch_goal3to5` YAML·train/test/VNC/RSI wrapper/output과 전용 bank를 추가했다. 중심입력/점XY·Z구간보상/상자/목표/AMP는 유지하며,RSI 후보만 초기높이가맞는 자세로 선별한다. 기존 bank 직접resume은 새경로/hash 계약으로 거부된다.
- `build_push_height_matched_rsi.py`가 GPU6에서408후보×2장면을 다시 시험했다. 초기기준자세·첫물리스텝의중점Z0.585~0.650m와12step finite/낙상없음/root높이/이동량/box높이·발지지/손면근접을 통과한11자세/7모션(ID0/1/2/5/6/19/20)을 저장했다. 원point bank hash불변·AMP21expert유지,14clip은이 RSI에서만제외한다. 후속12step내내 높이구간 유지 조건은 아니며 예비408후보검사에서는 이를 만족한 후보가0개였다.
- GPU6 RSI64환경/32768 agent reset 중 RSI16,388개 기준자세 높이구간내100%·원중심관측동일·부분reset/AMP/12stepfinite·bank불변 rc0(`output/push_fixed130_point_zband_heightmatch_goal3to5_rsi_check/{audit,height_screen}.json`). 현재방식은 도달가능자세를 고르는 것이며 전체21clip을낮은높이에맞게재설계한것은아니다.
- CPU48개·Python/셸문법/diff통과. GPU6·2048환경 scratch MAX_ITERATIONS=2는3epoch/frame393216 저장 rc0·685tensor finite,전용test2환경12step×2재로딩 rc0(`output/push_fixed130_point_zband_heightmatch_goal3to5_{check,eval_check}/`). config/structure에 GPU2본학습/GPU6검증 및 로컬/headless/VNC명령을 갱신했다. 본학습·장기성공률·GUI는 미실행이다.

### PUSH 점XY·높이0.585~0.650m무감점 구간

- 사용자 요청으로 기존 ±2cm 고정점과 비교 가능한 `push_fixed130_point_zband_goal3to5.yaml`·동명 train/test/VNC/RSI wrapper를 분리했다. 중심30D/570D 입력·1.3m/20kg/상자0.6·목표3~5m·중심AT는 유지한다. 중점XY는 면중앙의 점으로 유도하고 높이는0.585~0.650m(양끝포함)에서 무감점/접촉인정,구간밖거리만sigma0.04m로 감점한다. 힘/면거리/XY 조건과 보상 가중치·포화는 유지한다.
- `push_point.py`에 높이구간/밖거리 검증과 경계float32허용1e-6m를 추가했다. 기존 고정점/앵커 동작은 유지하며 새 계약은 `fixed_material_point_height_band_v2`다. RSI기준0.6175m·모션/retargeting/기하가 같아 기존 point bank408후보를 불변 재사용한다.
- CPU48개 통과(양끝포함 전구간점수1/접촉,밖거리대칭감점/접촉거부,AT/기존앵커회귀). GPU6 RSI64환경/32768 agent reset의 원중심관측동일·부분reset·AMP/12stepfinite·bank불변 rc0(`output/push_fixed130_point_zband_goal3to5_rsi_check/audit.json`). 초기 retarget기준자세 중점의 높이구간내비율은4.95%로 모든RSI초기손높이가맞는것은아니다. 어깨회전/팔길이보존 retarget 도달제한을 문서화했으며 전후보 PD screen을 새로 실행한 것은 아니다.
- GPU6·2048환경 scratch MAX_ITERATIONS=2는3epoch/frame393216 저장 rc0·685tensor finite,전용test2환경12step×2재로딩 rc0(`output/push_fixed130_point_zband_goal3to5_{check,eval_check}/`). Python/셸문법·diff통과. config/structure에 GPU2본학습/GPU6검증·로컬/headless/VNC 명령을 반영했다. 본학습·장기성공률·GUI는 미실행이다.

### PUSH 중심 입력 유지·고정점 보상·GPU2 본학습

- 사용자 선택 변경으로 `HumanoidMAOpenPushPoint`·`push_fixed130_point_goal3to5.yaml`·동명 train/test/VNC/RSI wrapper를 별도 추가했다. 기존 앵커 입력 실험은 보존한다. 원래 중심 위치/속도/크기/회전·AT목표/그래프 관측30D/570D를 그대로 쓰며 좌표 치환/추가 입력은 없다.
- 양손 중점이 최초 목표 반대 고정면의 표면 중앙·높이0.6175m 접촉점에 접근하도록 XY/면거리·Z 보상을 준다. 접촉 성공에는 양손 면거리/힘·중점XY≤0.18m/Z≤0.02m를 요구해 잘못된 높이의 접촉 보상 포화를 막는다. AT는 실제 중심 기준·기존 접근/포화/공유를 유지한다. 높이는 입력에 숨긴 무작위 목표를 피하려고 이전 범위 중앙값으로 고정하며 손 간격/개별 높이는 강제하지 않는다.
- `push_point.py` 점 보상/검증과 앵커 runtime의 phi/contact hook을 분리해 기존 앵커 동작을 유지했다. 새 목적 계약은 `fixed_material_point_midpoint_v1`이다. 전용 RSI bank408후보는 donor의0.6175m PD검사 포함·기하/모션hash를 대조해 보수적으로 재사용하고 원 bank/AMP21clip은 유지한다(전후보 재검사 아님). 현재 입력에서 Stage2 임의 접촉점 지시는 지원하지 않는다.
- CPU44개·Python/셸문법·diff통과. GPU6 RSI64평가환경/32768 agent reset에서 원 관측 완전동일·중심pose/속도·고정높이0.617500m·목표3.000145~4.999974m·부분reset/AMP/12step finite·bank불변 rc0(`output/push_fixed130_point_goal3to5_rsi_check/audit.json`). GPU6·2048환경 scratch MAX_ITERATIONS=2는3epoch/frame393216 저장 rc0·685tensor finite,전용test2환경12step×2 재로딩 rc0(`output/push_fixed130_point_goal3to5_{check,eval_check}/`).
- 사용자 요청으로 본학습 wrapper 기본값/문서 명령은 GPU2,검증은 GPU6이다. config/structure에 전용 학습·로컬/headless/VNC/RSI 명령을 갱신했다. 본학습/장기성공률/GUI는 미실행이다.

### PUSH 앵커 입력·중점 보상 없는 scratch

- `HumanoidMAOpenPushAnchor`·`push_point.py`·`push_fixed130_anchor_goal3to5.yaml`과 train/test/VNC/RSI wrapper를 추가했다. 1.3m/20kg/상자0.6·목표3~5m·2048환경이며, 목표 반대 고정면 표면의 앵커 높이0.585~0.650m를 reset마다 샘플링한다. 물체30D/전체570D에서 위치·선속도만 앵커/강체점속도로 치환하고 bbox·회전·AT 목표를 유지한다.
- 사용자 최종 선택에 따라 양손 중점 XY/Z 보상·접촉 gate를 제거했다. 개별 손의 지정 면3D거리/힘 접촉과 실제 중심 AT·기존 접근/포화/공유를 유지한다. 낮은 앵커는 입력/RSI용이며 손 높이/5° 안정성을 강제하지 않는다. 원본 graspview 입력 개념을 적용한 것이며 그 원본의 중점 보상은 복제하지 않았다. v2 관측/보상 계약으로 구 초안/중앙선 checkpoint와 구분한다.
- 동일 높이/retargeting/기하/모션 해시 대조 후 기존 낮은높이 PD screen408후보를 전용 `push_fixed130_anchor_goal3to5_viewonly_bank.json`에 출처hash와 함께 복사했다. 원 bank 불변·AMP21clip 유지·RSI만 생존후보 없는 ID7/8 제외. 원 screen은943+추가84후보×3높이×2장면×12step이며 이번 보상 변경 후 전후보 PD screen을 재실행한 것은 아니다.
- CPU40개·Python/셸문법·diff 통과. GPU6 RSI64평가환경/32768 agent reset에서 앵커0.585000~0.649995m·목표3.000145~4.999974m·관측/속도/부분reset/AMP·12step finite·bank 불변 확인(`output/push_fixed130_anchor_goal3to5_viewonly_rsi_check/audit.json`). 2048환경 scratch MAX_ITERATIONS=2는3epoch/frame393216 저장 rc0·685tensor finite·전용test2환경12step×2 재로딩 rc0(`output/push_fixed130_anchor_goal3to5_viewonly_{check,eval_check}/`). 본학습/장기성공률/GUI는 미실행. config/structure에 최종 구조와 명령을 갱신했다.

### PUSH 크기별 실제 밀기·손 중점 높이/힘 분석

- GPU6에서1.0~1.5m/5cm간격·밀도20kg/1.3³·상자0.6/바닥1의 전도분석을 수행했다. 실제humanoid PD1488조건·두물리손접촉4950조건·알려진두점외력12705조건을 분리하고 원자료/CSV/궤적·보고서/그림을 보존했다(`output/push_height_analysis_20261018/REPORT.md`). 기존학습/정책/보상/RSI bank는 유지했다.
- 물리손접촉0.45/0.475/0.50H495시험은0.802~0.862m이동·최대0.040°; humanoid775/1488이10cm이동/≤5°,지지/낙상제외49·손외접촉의심도제외2건이다. 전체크기의순수손밀기범위/임의힘절대안전은확정하지않았다. 알려진수평력1~5mg공통0.375~0.50H,수직±20%/양손불균형포함1~2mg공통0.25~0.375H시험점이통과했다.
- 영힘대조·접촉체초기높이1e-5m일치·유한물리실행·원bank불변·저장궤적최대각/변위/접촉수/실제중점독립재계산·Python/셸문법/diff검사통과. 접촉체초기tensor동기화오류예비실행은INVALIDATED.json으로제외하고수정실행만사용했다. 60Hz계측/내부2substep·4~5초·정육면체/중심COM·제어기/힘조건한계를보고서에명시했다. 재현wrapper/config/structure갱신.

### 고정면 PUSH 목표3~5m full-state resume

- `push_fixed130_fixed_face_midpoint_goal3to5` YAML·train/test/VNC/output을 분리했다. 목표 반경3~5m 균등·360°·reset후고정,배치반경8m·상자간최소12m·envSpacing20으로 작업구역을 확보했다. 기존 고정면/중점·보상/AT·600step·1.30m/20kg·사람1~2m·RSI혼합/AMP/네트워크는 유지한다.
- `prepare_push_goal3to5_resume.py`·`utils/push_goal_resume.py`로 현재 fixed-face8500의 엄격한 계약검사 후 전용 bank943후보/출처hash와 full-state 이관본을 생성했다. 모델·optimizer·AMP/RMS·epoch/frame 등685tensor의 완전동일성을 독립확인(`output/push_fixed130_fixed_face_midpoint_goal3to5/resume/*.preservation.json`). 구 저장본 직접resume은 불가하며 이관본을 사용한다. 기존 학습은 유지했다.
- CPU21개·셸문법/diff통과. GPU4에서 지시변경 전 완료한 RSI64환경·32768agent reset: 거리3.000006~4.999993m(평균3.997986)·초기AT0·RSI50.15%/후반70.36%·상자간최소12.000035m·초기목표면/부분reset/AMP/12stepfinite·bank불변확인(`output/push_fixed130_fixed_face_midpoint_goal3to5_rsi_check/audit.json`). 기존 PD-screen후보의 rigid XY/yaw 재사용이며 신규전후보PD검사는아니다.
- 이후 사용자 지시에 따라 GPU6으로 검증했다.2048환경8500→8502·frame증가·모델finite·optimizer유지/업데이트·저장rc0(`output/push_fixed130_fixed_face_midpoint_goal3to5_resume_check/`),전용test2평가환경12step×2재로딩rc0(`output/push_fixed130_fixed_face_midpoint_goal3to5_eval_check/`). 장기성공률/GUI는미검증,본학습은자동실행하지않았다. config/structure 및 앞으로검증GPU6 지침을 갱신했다.

## 2026-10-17

### PUSH 목표 생성 시 면 고정·양손 중점 중앙선

- GPU6 고정1.30m/20kg 설정에서 `push_fixed130_fixed_face_midpoint` YAML·train/test/VNC/output을 분리했다. 최초 목표 반대 주축면 번호를 owner별로 고정하고 reset한 환경만 갱신한다. 기존 yaw/XY 기하에서 상자 위치/yaw는 추종하며 손 위치/목표 상대 방향으로 면을 다시 고르지 않는다.
- 양손 중점의 중앙선 XY 거리와 각 손의 면 XY 거리 중 최댓값으로 PUSH state/접촉을 계산한다. 높이/손 간격은 자유, 기존 힘·AT·도착 포화 조건은 유지한다. 기존 RSI bank의 초기 retarget/PD screen은 같아 재사용하며 reward 계약이 다른 구 checkpoint 직접재개는 허용하지 않는다. 기존 학습 프로세스는 유지했다.
- CPU48개 통과(대칭 손 벌림·중점 이동·높이 자유·면 불변/yaw 추종·부분reset·기존 보상 회귀), 셸문법/diff 확인. GPU6·2048환경·MAX_ITERATIONS=2는 기존 종료조건에 따라3epoch 후 저장 rc0 (`output/push_fixed130_fixed_face_midpoint_check/`). 전용 test2환경·12step×2 checkpoint재로딩 rc0 (`output/push_fixed130_fixed_face_midpoint_eval_check/`). 초기 설정검사 변수 참조 오류를 수정한 뒤 재검증했다. 본학습·장기성공률·GUI/VNC는 미실행이다.
- 추가 실제 RSI 감사: GPU6·64평가환경·32768 agent reset에서 초기 목표면 일치·RSI 중점 중앙선 거리 최대0.060001m(<0.18m)·부분 reset 면 보존·AMP/관측 finite·12step·bank 불변 rc0 (`output/push_fixed130_fixed_face_midpoint_rsi_check/audit.json`). 기존 검사기를 새 모드에 맞게 확장했으며 PD screen 전후보를 새로 재검사한 것은 아니다.

### 한 면PUSH 전도 직전 모멘트 추정

- `output/push_tipping_moment16500_20261017/`에16500보행시작저장궤적의CPU분석/JSON/보고서를추가했다. 첫20°초과3step이전의마지막≤5°구간0.5초와직전0.2초를비교하고목표방향·시간·접촉여부를맞춘대조군을사용했다. 새시뮬레이션/학습변경없음.
- 같은한면대조76사건에서총손힘105.8/105.2N·관측전도축손모멘트89.0/72.8Nm였지만네모서리최대손/중력위험비0.760/0.773으로구분되지않았다. 전도축방향의연관은있으나손목순반력/추정접촉점이며바닥및다른신체접촉모멘트가없어원인을확정하지않는다. 힘방향/중력/회전4개기하검사·Python문법확인.

### 고정PUSH 마찰별 평가 viewer·warmstart29000 손 높이

- `view_push_friction.py`·`push_fixed130_friction_{test,vnc}.sh`를추가했다. 원래YAML/checkpoint계약검사는유지하고평가actor상자마찰만0.2/0.4/0.6으로변경한다. 바닥/발/RSI·학습프로세스는유지하고선언값과실제shape값을분리기록한다. config/structure에17000의세마찰VNC·로컬/headless명령을추가했다. 새학습config는아니다.
- GPU6·17000·마찰별2평가환경·12step의checkpoint재로딩/실제shape마찰일치/rc0확인(`output/push_fixed130_friction_view_check/`). Python/셸문법·diff검사통과. VNC GUI는미실행이다.
- warmstart29000은원래YAML·GPU6·64scene·600step·seed8429·RSI혼합/보행각2회rc0·bank불변확인. 보행한면90%그룹147/255성공·이동양손접촉손Z평균0.979m(현재16500동일그룹1/189·1.079m). 목표분포/정책차이가있어높이차이의인과효과로단정하지않는다. `output/warmstart29000_hand_height_20261017/REPORT.md`·JSON/NPZ에보존했다.
- InterMimic은논문표object/ground0.9와현재공개코드가다르며실제object0.6·바닥0.9임을공식코드로재확인하고위보고서에근거링크를남겼다.

### PUSH 접촉 형태 분리·최신16500 엄격 평가

- `output/push_fixed130_contact_analysis_20261017/`에 세저장본 손위치/순body힘/상자자세궤적·접촉면기하분류·기울기직전대조분석을 보존했다. 기하/90°yaw8검사·동일초기상자/root/목표/RSI·이전성공건수일치확인. 갱신전초기손캐시는제외하며 실제물체별contact pair측정은아니다.
- 최신16500은 `output/push_fixed130_contact_latest16500_20261017/`에서GPU6·기존학습유지·128scene·600step·seed8429·RSI혼합/보행각2회rc0. 이동중수평구간최소15frame·접촉유형90%엄격기준의보행한면성공1/189(0.5%)·20°초과82.0%, 인접두면100/104(96.2%)·20°초과1.0%. 14500의0/179·90/116과비교했다. 전체보행31.25→38.28%/혼합48.05→57.23%개선이한면밀기개선을보장하지않는다. YAML/checkpoint/RSI bank는변경하지않았다.
- 보고서/JSON·config/structure에재현경로기록. Python/셸문법·이전14500엄격건수독립일치·문서diff확인. 단일seed·행동에따른사후접촉분류의한계를보고서에명시했다.

### 고정 PUSH의 이동 중 기울기·도착 전 재수평 평가

- `output/push_fixed130_transit_tilt_20261017/`에 비학습 평가·전체 control-step 궤적·독립 재계산 보고서를 추가하고 config/structure에 재현 명령을 연결했다. GPU6·10000/12500/14500 저장본·seed8429·128scene·600step·RSI혼합/보행 각각2회(조건별512 agent episode) 완료. 기존 학습과 YAML/checkpoint/RSI bank는 유지했다.
- 초기≤5°→기울기>20° 연속3step→기울어진 구간 순 목표접근≥0.10m→첫AT≤5° 사건은 전조건0건. >10° 기준은 혼합0/4→1/90→5/246, 보행0/5→0/13→4/160(성공중)이다. 14500 성공군의 최대기울기 평균4.36°/3.88°는12500보다낮고 실패군20°초과경험은36.8%/41.2%로증가했다. 큰기울기재수평전략이도착률상승의주된양상이라는근거는없다. 단일seed·이른저장본성공표본소수·control-step표본의한계를보고서에명시했다.
- Python/셸문법·문서diff 검사, 저장NPZ의최대각/순접근량/첫성공 독립재계산, 이전평가성공건수 전조건일치·bank불변 확인. 0건bootstrap CI는 퇴화하므로 확률0의 근거로 표기하지 않는다.

## 2026-10-16

### BONES 협동 승강용 모션 후보 조사

- Stage2 탑승자 포함 물체 승강을 고려해 양팔 상승/바닥 지지 후보를 검색하고 공식 viewer 원본BVH4개·손목분석·자세비교를 `bones_seed/validation/support_lift_candidates/`에 저장했다. A520/A522는 샘플 구간에서 양손목 약0.82~0.85m/0.73~0.74m 상승 확인. BVH parser/FK·전프레임 finite 검사 통과. 아래 받침 접촉·협동 물리·retarget/학습은 미검증이며 기존 config는 변경하지 않았다.

### ArmChair Obj tokenizer 복사본·원래 정책 보존 분기

- `carry_chair_copied_adapter` env/train YAML·train/test/VNC를 추가했다.39000 Obj 가중치를 별도 `adapted_object_tokenizer`에 복사해 adapter와 함께 학습하고 원래 Obj/head/Transformer를 동결한다. 평가의 `POLICY_BRANCH=base`는 원래 tokenizer·adapter 끔, 기본adapt는 복사본+adapter를 선택한다. 하나의pth에 두 tokenizer를 저장하며 기존 실험/원본KSH는 변경하지 않는다. 학습 가이드/config/structure에 명령을 추가했다.
- CPU5개·문법·링크/diff 검사 통과. CPU는 초기출력 일치·학습 후base 행동 불변·저장/복원 양분기를 검사했다. GPU4·ArmChair63종·2048환경·짧은2epoch rc0: 복사본/adapter/critic/판별기 업데이트, 원래 Obj/head 업데이트없음·동결76개와RMS 불변. 실제 저장본의 원래Actor76개 tensor도39000과 완전일치했다(`output/object_tokenizer_finetune/carry_chair_copied_adapter/train_check`). base/adapt 각각 test대표객체·2환경·12step 재로딩 rc0·원본해시 불변도 확인했다(`eval_base`, `eval_adapt`). 장기성공률은 미측정이다.

### ArmChair 전용 action adapter 파인튜닝

- `carry_chair_adapter` env/train YAML·전용 train/test/VNC를 추가했다. TokenHSI의 층별0초기화 Linear 잔차 adapter를 재사용하며 기존 Obj tokenizer와 adapter·critic/AMP만 학습한다. 기존 action head/Transformer/사람·Goal tokenizer와 관측/AMP RMS는 동결한다. ArmChair63종·HOLDING_AT·loco 초기화를 유지하고 기존 출력과 분리했다. 학습 가이드/config/structure를 갱신했다.
- CPU4개·Python/셸 문법·링크/diff 확인. GPU4·전체63종·2048환경·짧은2epoch 확인 rc0: Obj tokenizer/adapter/critic/판별기 업데이트, head 업데이트 없음·동결 actor70개/RMS 불변·원본 코드/config 및 시작 가중치 사본 해시 유지(`output/object_tokenizer_finetune/carry_chair_adapter/train_check`). test 대표 객체·2환경·12step 재로딩 rc0·해시 불변도 확인했다(`carry_chair_adapter/eval_check`). 장기 본학습/성공률은 미측정이다.

### 객체별 파인튜닝 학습 가이드 분리

- 사용자 요청 형식에 맞춰 [학습 전용 가이드](markdowns/object_finetune_train_guide.md)에 config 표·CARRY 의자/테이블·SIT 명령을 모았다. 실제 wrapper의 단일 env 인자·출력 제한·CHECK 미지원과 기존 tokenizer-only/미완료 객체별 Adapter·Head 버전을 구분했다. config/structure에 연결했다. 문서 경로·명령 및 diff만 확인했으며 학습은 새로 실행하지 않았다.

### PUSH 2번·고정1.30m/20kg·AT 기울기 제한 없는 학습

- 기존GPU6 문없는PUSH 전용설정에서크기가변만제거한 `push_fixed130_20kg_no_tilt` YAML·train/test/VNC/output을분리했다. AT기울기감쇠/성공각도제한없음·기존거리/높이조건과PUSH/AT보상·손접촉·AMP/네트워크를유지한다. PUSH전용RSI/학습계약이누락된가변크기키를None으로표현하도록고정크기를지원했으며기존가변설정계약은불변이다.
- 기존31크기PD통과943후보를동일하게보존한고정1.30m subset bank에출처hash/검증범위를기록했다. CPU115개·Python/셸문법·diff통과. GPU6·64평가환경·32768agent reset: 실제질량19.999998kg·고정1.30m·초기AT성공0·RSI50.41%/후반69.47%·초기손목표거리최대0.1735m·부분reset/관측570D/AMP1310D/12stepfinite·bank불변확인(`output/push_fixed130_20kg_no_tilt_rsi_check/audit.json`). 신규전후보PD재검사는주장하지않는다.
- GPU6·2048환경·MAX_ITERATIONS=2의scratch확인은기존loop종료조건에따라3epoch후저장rc0·모델finite·고정크기/질량/무감쇠계약을확인했다(`output/push_fixed130_20kg_no_tilt_check`). 전용test2환경12step×2재로드rc0(`output/push_fixed130_20kg_no_tilt_eval_check`). config/structure명령갱신. 장기본학습/GUI는미실행이며기존프로세스종료/재시작은하지않았다.

### PUSH 1번·1.20~1.40m와 AT 전용 기울기 감쇠

- 사용자 최종범위1.20~1.40m·21종(15.73~24.98kg)의 문없는570D scratch YAML·train/test/VNC를 분리했다. AT state/progress는5~15° 선형감쇠를 성공포화후 적용하고 AT성공은기존조건+≤10°,성공보너스는감쇠하지않는다. PUSH접촉/보상과기존도착포화규칙·AMP·네트워크는보존한다. 기존실험은옵션없이종전동작을유지하며구checkpoint직접재개는거부한다.
- 기존31크기PD검사통과943후보를그대로보존한21크기subset bank에원본SHA256/검증범위/계약을기록했다. 신규전후보PD검사를주장하지않으며GPU6·42평가환경·21504agent reset에서초기AT성공0·RSI50.09%/후반69.60%·손목표거리최대0.1735m·실제질량/21크기·부분reset보존·관측/AMP/12step유한값을확인했다(`output/push_size120to140_at_tiltfade_rsi_check/audit.json`).
- CPU108개·Python/셸문법·diff 확인. GPU6·2048환경·MAX_ITERATIONS=2의짧은학습은기존loop종료조건으로3epoch후저장rc0(`output/push_size120to140_at_tiltfade_check`),전용test2환경12step×2재로드rc0(`output/push_size120to140_at_tiltfade_eval_check`). config/structure에명령을연결했다. 장기본학습/GUI는미실행·기존GPU4/6학습은유지했다.

### 두 PUSH 정책의 무게별 성공·성공 시 기울기 비교 명령

- 기존 크기/실제 actor 질량 평가기를 전용 `evaluate_push_mass_success.py`로 옮기고 첫 AT 성공 순간·성공 상태 중 최대 기울기를 성공 에피소드에 한해 평균/중앙값/P90으로 집계한다. 실패 기울기는 null이며 yaw와 구분한다.
- `compare_push_mass_success.sh`는 GPU4에서 두 학습의 최신 번호 저장본을 동시에 비학습 평가하고 정책별 JSON/log와 무게별 CSV·REPORT를 새 output에 저장한다. 원래 YAML/manifest/checkpoint 계약·학습 프로세스는 유지한다. CPU 조건부 성공/성공0건/빈 그룹·두 정책 CSV 생성 및 Python/셸 문법·diff 검사 통과. 실행 명령만 준비했으며 GPU 평가 결과는 미측정이다.

### 기존 물체 tokenizer 독립 fine-tuning·TokenHSI 객체군 분리

- 원본 TokenHSI config/task 코드를 확인해 SIT StraightChair, CARRY ArmChair/Table_Circle을 별도 실험으로 만들었다. `ports/ksh_object_finetune`의 KSH39000 독립 사본에서 기존 물체30D tokenizer·critic·AMP 판별기만 학습하며 새 tokenizer/adapter는 추가하지 않는다. 전용 YAML·train/test/VNC 각3종과 [실행 문서](markdowns/object_tokenizer_finetune.md)를 config/structure에 연결했다.
- CARRY는 원본처럼 loco 초기화·운반 AMP를 사용한다. SIT는 좌판 높이 translation과 객체 ID/fingerprint별 PD 후보 캐시를 사용하고, 원본의 고정 의자에 대응하는 reset 가능한 지지체로 수정했다. 초기 movable 의자 감사에서 발견한 접촉 튐은 수정 후 전체49종·128환경·12step 검사에서 물체 최대속도1.16e-8m/s 미만이었다. 후보3763개·1종 후보없음·fallback91/5554(1.64%), partial reset의 다른 환경 root/DOF/AMP 완전보존·관측/AMP finite를 확인했다(`output/object_tokenizer_finetune/sit_fixed_full_rsi_check`).
- CPU2개·private Python/9개 셸 문법·diff 검사 통과. GPU6·2048환경·전체 train49/63/21종의 짧은 학습은 모두 rc0이며 기존 actor70개/RMS 불변·물체 tokenizer/critic/판별기 실제 업데이트·초기 model39000 완전복사를 확인했다. SIT 최종 검사는 `sit_chair/train_fixed_check`, CARRY는 각 `train_check`에 저장했다. 세 실험 저장본은 test 대표 객체·2환경·12step 재로딩 rc0(SIT `eval_fixed_check`, CARRY `eval_check_v2`). KSH 코드/config와 시작 checkpoint 사본 해시 불변이며 장기 본학습·성공률 측정은 수행하지 않았다.

## 2026-10-15

### CLIMB 학습 크기 범위 내 native·동일 크기 상자 대조

- 사용자 요청으로 저장 YAML의 `env.box.build.taskSizeRanges.CLIMB`를 읽는 native 필터와 동일 bbox/URDF질량 상자 대조·전 객체 실제 물성 감사·비교 launcher를 추가했다. 원래 크기가 범위 안인 Box6/Cabinet2/Table3종만 사용하며 리스케일/학습은 하지 않는다. 관성/COM은 collider별로 다름을 명시했다. 준비 단계 YAML 키 오류를 수정하고 실제 선택6/2/3종을 확인한 뒤 본평가했다.
- CPU17개 통과. GPU6/tokenhsi·각64scene/128agent·600step·seed42: native 올라가기99.22/99.22/81.25%, 같은 크기·질량 상자99.22/97.66/100%. 초기 root/물체/goal·bbox·실제 질량/마찰 대조 전부일치,6worker rc0·finite·KSH 코드/YAML/39000 checkpoint hash 불변. Table은 범위 내에서도 차이가 남으며 COM/관성/지지구조를 분리한 인과효과는 미측정이다. [분석](output/climb_training_range_matched_gpu6/REPORT.md). config/structure/객체 문서에 재현 명령을 연결했다.

### CLIMB 올라가기 사건 평가 기록

- 직전 사용자 요청으로 중앙 root 목표/종료 유지 대신 양발 윗면 geometry·접촉을 집계하는 `--climb-ascent`를 추가했다. CPU16개 및 GPU6/tokenhsi·Box18/Cabinet9/Table8종·각32scene/64agent·600step·seed42 완료: 올라가기79.69/45.31/51.56%,3step유지78.13/43.75/40.63%. 기존 final/ever도 병기하며 서로 다른 조건의 지표임을 명시했다. 세 worker rc0·원본/39000 hash 불변(`output/climb_ascent_39000_gpu6/`). 학습/GUI/기존 프로세스 변경 없음.

### 39000 SIT·의자 CLIMB 좌판 입력 보정 비교

- 사용자 요청으로 기존 native 객체 평가에 `full/seat-height/seat-inset` 정책 bbox와 `--climb-surface seat`를 추가했다. 실제 mesh·물성·state는 유지하고 관측634D/물체30D의 가상 bbox·GTA origin·가상 중심 선속도만 일관되게 보정한다. CLIMB도 등받이 최상단 대신 좌판에서 평가하며 세 조건에 같은 판정을 적용한다. XY0.75배는 정확한 좌판 annotation이 아닌 근사다. 비교 launcher·정량/로컬/VNC 명령을 config/structure/객체 문서에 연결했다.
- CPU11개 통과(원본 관측/state 불변·회전·속도·부분 row·634D 계약 포함). GPU1 2환경12step×SIT/CLIMB 추론 확인 후 사용자 GPU 변경에 따라 정량 비교는 GPU6/tokenhsi에서 전부 재실행했다. 의자26종·각32scene/64agent·600step·seed42: 종료 성공률 SIT7.81→40.63→48.44%, CLIMB1.56→12.50→10.94%(full→높이→높이/폭). 초기 root/물체/goal·실제 물성/충돌체 감사가 세 조건 완전일치, KSH 코드/YAML·checkpoint SHA256 불변(`output/seat_proxy_comparison_gpu6/comparison.json`). 단일 seed의 초기 비교이며 등받이 접촉 직접검사/정확한 좌판 polygon/GUI는 미검증이다. 학습·기존 프로세스 변경 없음.

### GPU4 PUSH8500의 크기·무게별 GPU1 비학습 평가

- 사용자 요청으로 원래 AT01 8500 저장본을 GPU1·248평가환경·31크기 균등·600step·seed8429로 평가했다. RSI혼합/보행 시작 각3968 agent episode(크기별128), 실제 actor 질량12.12~24.98kg·원래 checkpoint 계약/정규화 재사용·bank hash 보존을 확인했다. 결과/측정기/CPU장면bootstrap/CSV/그림은 `output/push_gpu4_size_mass_audit_20261015/REPORT.md`에 연결했다. 기존 학습을 중단/재시작하지 않았다.
- 보행 시작에서 작은/중간/큰 상자 AT도달36.8/39.6/33.0%, 기울기20°초과62.7/48.7/14.7%,45°초과13.5/0.14/0%. 수평yaw30°초과17.3/16.7/14.8%로 작은−큰 차이의95%장면bootstrap구간은0을 포함한다. 크기와질량이함께변하므로질량단독인과효과로해석하지않는다.
- 62환경12step 짧은확인·본평가종료rc0·7936표본/크기균등/보행RSI0/유한값·Python/셸문법·CPU보고서/그림생성확인. 이후사용자의GPU1사용금지에따라추가GPU작업없이CPU로그만조회했다.

### 의자 CLIMB 시각화 조합 추가

- 사용자 요청으로 `climb_chair`를 등록해 SIT용 StraightChair_Normal을 기존 CLIMB 정책으로 시각화하도록 했다. 전체 bbox·CLIMB 최상단 판정 유지, GPU1/6123 명령을 config/객체 문서에 연결했다. Python 문법·실제 URDF/metadata 경로·diff 검사만 수행했고 viewer는 실행하지 않았다. KSH 수정 없음.

### 객체 시각화의 원래 renderer 복구·reset 입력 확인

- 사용자 요청으로 객체 viewer 기본을 KSH 원래 Isaac Gym renderer/UI/stack 카메라로 바꿨다. proxy는 추가 N/R/Space만 처리하고 기존 F/V/K/L/Esc 이벤트는 그대로 전달한다. 이전 근접 renderer는 `--viewer-style focus`로 남겼다. N/R은 같은 객체 새 배치이며 다른 mesh 교체가 아님을 문서에 명시했다.
- reset 환경 row를 명시하고 키 수신/새 장면 로그를 추가했다. 기존 자동 반복 명령(`--scene-mode loop --episode-length 600`)을 config/객체 문서에 기록했다. CPU9개(press/release 및 여러 render 호출 사이 reset 유지·기본 단축키 전달 포함), Python/셸 문법 확인. GUI는 사용자 요청대로 실행하지 않았으며 KSH 원본 수정 없음.

### 객체 viewer 관측 오류 수정·수동 장면 유지

- 사용자 로그의 `CommonPlayer.get_action` IndexError 원인은 viewer의 두 번째 step에서 Tensor 관측을 dict로 전달하지 않은 것이었다. reset/step 입력을 동일 dict 계약으로 맞췄다. 시각화 재실행은 기존 output을 보존하고 `_runNNN` 새 폴더를 자동 선택한다.
- 기본 viewer는 한 장면18000step(약10분) 후 화면 유지·성공/넘어짐 자동 reset 없음·Space 일시정지/N 새 장면이다. 기존 자동 반복은 `--scene-mode loop`로 선택한다. config/객체 시각화 문서 갱신. CPU7개 및 셸 문법 통과, 수정 후 viewer/GPU 평가는 사용자 요청에 따라 실행하지 않았다. KSH 원본 수정 없음.

### 39000 distill 객체 제로샷 경로·객체별 시각화 명령

- KSH39000의 코드/설정/정규화를 읽기 전용 재사용하는 `evaluate_object_zero_shot.py`와 test/VNC wrapper를 현재 레포에 추가했다. 실제 native URDF·충돌체·원래 크기/질량·634D/물체30D를 사용하며 SIT 좌판 metadata는 판정에만 적용한다. GPU PhysX의 mesh teleport 초기 충격은 접촉을 비운 뒤 전체 root/DOF 시작 상태 복원으로 제거했다. KSH 파일·checkpoint는 수정하지 않는다.
- CPU5개 통과, GPU1 대표5종2환경12step 실제39000 추론/finite/차원·충돌체 생성 확인. 초기 충격 대조검사·79개 native 객체의32scene/600step 결과는 별도output에 보존했고 원본 코드/설정 및 checkpoint SHA256 불변을 확인했다. 이후 대조/진단 평가는 사용자 요청으로 중단했다.
- 최종 요청에 따라 성공률 집계 없는 반복 재생 모드와 SIT1/CLIMB3/CARRY3의7개 VNC 명령을 `markdowns/object_zero_shot.md`에 추가하고 config/structure에 연결했다. 사용자가 직접 실행하도록 viewer는 띄우지 않았다. 재생 모드는 문법/CPU 검사만 했으며 GUI 실행은 미검증이다.

### 현재 PUSH 전용570D 학습에 AT 기울기 제한 적용

- GPU6에서실행중인PUSH전용물리프로파일을기준으로 `open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors_tilt035` YAML/train/test/VNC/output을추가했다. 원본과차이는 `contact_at.max_tilt=0.35` 하나이며570D/물체30D/6토큰/AMP1310D·물리·PUSH보상·RSI/모션을유지한다. 앞서준비한문포함684D실험과별도이며이번대상은570D다.
- `convert_push_only_at_tilt_checkpoint.py`와엄격한570D계약검사로현재실행의6000회저장본을 `output/push_only_at_tilt035_migration/epoch6000.pth`에이관했다. model/optimizer/RMS/epoch/frame등비metadata상태전부원본과완전일치,원본bank/hash유지·실제runtime계약일치확인. config에GPU0 resume와로컬/서버VNC/비학습검사를기록했다.
- CPU57개통과(570D보존·기울기경계/yaw·기존PUSH/AT회귀·이관보존/타계약거부). GPU6·62평가환경/31크기·31744reset: 초기AT성공0·기울기초과0·RSI49.97%/후반70.52%·부분reset/기존AMP데모및actor/critic/판별기추론finite·12step정상(`output/push_only_at_tilt035_rsi_amp_check/audit.json`). 570D전용test경로2환경12step×2재로드rc0(`output/push_only_at_tilt035_eval_check/`),셸문법/diff검사통과. 본학습은실행하지않고기존GPU6프로세스를그대로유지했다. 장기수렴/새성공률은미측정.

### 문 포함 AT01의 AT 성공 기울기 제한·6000회 재개 준비

- 사용자가 아침에 중단한 문 포함 AT01의6000회 저장본 재개를 위해 별도 `open_push_contact_at_friction06_random_size_goal1to25_at01_tilt035` YAML/train/test/VNC/output을 추가했다. AT 성공에만 로컬Z/월드Z 기울기≤0.35rad를 AND하며 성공보너스/포화 이전에 적용한다. 관측684D·물체40D·AMP1310D·거리보상·PUSH접촉·물성·RSI bank/비율은 유지한다. 원본 옵션없는 실험은 그대로다.
- `convert_at_tilt_checkpoint.py`/`utils/at_tilt_checkpoint.py`로 보상계약만 명시적으로 바꾼 사본 `output/at_tilt035_migration/epoch6000.pth` 생성. GPU검사에서 원본과 model/optimizer/RMS/epoch/frame 등 비metadata 상태 전체 완전일치·실제 runtime 계약일치를 확인했다. 원본 bank/checkpoint를 덮어쓰지 않는다. config에GPU4 resume·로컬평가·서버VNC·GPU6 검사명령과structure진입점을 기록했다.
- CPU87개 통과(기울기경계/yaw·다른보상/RSI불변·이관보존/오설정거부). GPU6·62평가환경/31크기·31744reset: RSI49.44%/후반69.97%·초기AT성공0·기울기초과0·bank SHA256동일·부분reset/AMP데모finite·실제6000회actor/critic/AMP판별기와정규화재로드12step정상(`output/at_tilt035_rsi_amp_check/audit.json`). 전용test스크립트2평가환경/12step×2재로드rc0(`output/at_tilt035_eval_check/`). Python/셸문법·diff공백확인. 새학습은실행하지않고기존프로세스를중단/재시작하지않았다. 장기수렴/새조건의성공률은미측정.

### PUSH 전용 학습의 물리 문·문 설정 제거

- 사용자 요청으로 기존 no_doors 실행 YAML을 `pushOnly`로 바꿔 문 asset/actor/관절/힘·hold/posture·문 reset/보상/AMP mapping·문 모션 로딩을 제거했다. 실제 사람2/상자2이며 기존 내부4슬롯/관측570D·6토큰/AMP1310D와 PUSH 보상·크기·목표·RSI 비율은 유지한다. 전용 loco/PUSH manifest·PUSH만 추출한 bank를 사용하고 기본scratch output/실험명을 physical_scratch로 분리했다. config/structure 명령·위치 갱신.
- CPU192개 통과(기존PUSH 보상/성공 완전일치·문 설정/모션 부재 포함). GPU6·62환경/31크기·12step 실제 문0/문관절0·loco/push만 로드·RSI50.44%/후반71.94%·부분reset/AMP 통과(`output/push_only_physical_sim_check/audit.json`). GPU6·2048환경 scratch epoch2/frame262144 및 기존3000 이관사본 resume epoch3001 저장,148종scalar/저장본tensor finite·이관시모델/optimizer/RMS/카운터완전보존(`output/push_only_physical_train_check/saved_check.json`). 기존사본4환경/새저장본2환경 재로드평가·VNC1환경12step rc0. 셸문법/diff 확인.
- 기본은 전이 없는scratch이며 기존30D checkpoint는 `convert_push_only_checkpoint.py`로 문 비활성설정만 제거한 사본을 명시적으로 만든 뒤 RESUME_CHECKPOINT 지정 시에만 재개한다. 원본 bank/checkpoint·기존 학습프로세스를 변경하지 않았다. 사용자의 추가 학습 금지 지시에 따라 이후 학습은 실행하지 않으며 새 장기 학습은 시작하지 않았다. 장기수렴/성공률은 미측정.

## 2026-10-14

### 문 없는30D scratch의 RSI·AMP 실행 검증

- 사용자 확인 요청에 따라 GPU6·2048환경·MAX_ITERATIONS=1·별도output으로scratch짧은학습epoch2/frame262144저장확인(`output/push_no_doors_scratch_train_check/saved_check.json`). load_checkpoint=false·전이config없음·물체30D/정책570D·AMP1310D·모델/161종scalar유한값·실제AMP보상확인. 새저장본2환경30step재로드rc0(`output/push_no_doors_scratch_eval_check/`).
- 기존PD검사bank를재생성/덮어쓰기하지않는검사사본으로496환경/31크기·31744reset표본을확인했다. RSI50.12%·후반요청69.38%·초기AT성공0·AMP이력/데모finite·부분reset보존통과(`output/push_no_doors_30d_rsi_amp_check/audit.json`). 원본bank SHA256동일,기존GPU6학습PID3071308유지. 본학습/장기수렴/성공률은미검증이다.


### 문 없는 물체30D PUSH를 scratch로 전환

- 사용자 결정으로 no_doors 실행 YAML의 experiment.transfer를 제거하고 train의TRANSFER_CHECKPOINT를거부한다. 기본출력/실험명을no_doors_scratch로분리하고test/VNC출력·config명령/전체목록·structure를갱신했다. GPU6·seed42·2명/2048환경/물리객체4(정책상자2)로기존보상/물성/AMP/RSI설정을유지한다.
- CPU111개통과, YAML전이없음·load_checkpoint=false·전이변수실제거부·셸3개문법·diff확인. 기존투영utility검증은테스트내명시적source설정으로유지했다. 요청은명령제공이므로새학습을실행하지않았으며기존학습을중단/재시작하지않았다.


### 문 없는 PUSH 물체 입력을 원래30D로 정렬

- 사용자 요청으로 문 제거 버전의 `push_eligible` 1D도 tokenizer 입력에서 삭제했다. 물체는선속도3·각속도3·bbox24의30D, MLP30→256→128→64·전체정책570D/6토큰이다. source tokenizer/RMS는원본40열중0~29열만전이하며 관측계약을 `push_no_doors_30d_v2`로갱신했다. 기존31D관측계약과직접혼용불가. 기존문포함학습/보상내부credit계산/AMP는유지한다.
- CPU111개통과(eligible/runtime무접근·실제10000전이·RMS·양encodergradient·기존회귀). GPU6 4평가환경12step×2회Base정책570차원·물체30D·6토큰정상종료(`output/push_no_doors_30d_sim_check/`). Python문법/diff확인. 학습/저장/재개/장기성능미검증,기존학습중단/재시작없음.


### PUSH 정책의 문 관측·토큰 삭제

- 사용자 요청으로 현재 레포 OPEN/PUSH의 선택적 `pushNoDoorPolicy`와 전용 `open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors` YAML/train/test/VNC/output을 추가했다. 문2개 특징·GTA pose와 상자 슬롯의 문 전용9열을 실제 삭제해684→572차원·Actor/Critic8→6토큰·상자40→31특징으로 줄인다. 물리슬롯4개는 유지하며 정책객체2개/정규화/graph endpoint를 일치시킨다. 원본10000 actor tokenizer/RMS는 남는열로 투영전이하고 critic/optimizer를 초기화한다. 새 관측계약으로 기존모델의 직접resume/eval을 거부한다.
- CPU111개 통과(실제10000가중치/RMS전이·문값/poseNaN독립성·문특징무접근·binding/graph·순열·양encoder학습gradient·기존OPEN/PUSH/전이회귀). GPU6·4평가환경×2반복·12step Base정책572차원/6토큰 실제연결rc0(`output/push_no_doors_sim_check/`). Python/셸문법·YAML·새문서링크·diff검사 확인. 학습/저장/재개/장기성능은미검증이며현재GPU6학습PID3071308은종료/재시작/설정변경하지않았다. 수정은현재stage1_fix레포에서만수행했다.

### PUSH 목표1~2.5m·AT progress0.1m·10000 정책 전이

- 사용자 요청으로 `open_push_contact_at_friction06_random_size_goal1to25_at01.yaml`과train/test/VNC/RSI wrapper를추가했다. 사람→박스progress0.5m를유지하고AT전용 `at_progress_delta=0.1`을분리했다. 옵션없는기존실험은기존식을유지한다. goal25의10000 actor/행동분산·AMP·관측/AMP입력RMS전이,critic/value/optimizer/보상RMS/카운터초기화;목표/AT포화/대응bank만변경허용·source snapshot/metadata/network/물리계약검사추가. config명령/목록·structure진입점갱신.
- CPU88개·셸4개/Python문법·diff검사통과. 기존PD943/1744·21clip bank의물리계약일치로재사용하고GPU6 최종설정31744reset초기AT성공0·남은거리0.150236~2.499918m·RSI50.12%/후반요청69.38%·AMP/부분reset확인(`output/open_push_friction06_random_size_goal1to25_at01_rsi_check/`). GPU6 62환경/31종크기·목표1.000130~2.499995m·네면/보행/RSI12step·부분reset검사통과(`output/push_goal1to25_at01_sim_check/`).
- GPU6 2048환경별도짧은전이학습epoch2/frame262144정상저장·source10000/88텐서전이·critic81텐서초기화보고서·모델/161종scalar유한값확인(`output/push_goal1to25_at01_transfer_check/`). 새저장본2환경30step재로드종료코드0(`output/push_goal1to25_at01_eval_check/`). 장기수렴/성공률은미측정이며본학습은시작하지않았다. 기존프로세스를중단/재시작하지않았다.

### PUSH9500 손 접촉 중 방향 독립 이동량 측정

- 사용자 요청으로 `measure_push_hand_motion.{py,sh}`를추가했다. 원래checkpoint계약/정책/물성으로학습RSI혼합과보행시작을평가하고물리tick의전체표면 한손/양손 힘접촉·방향독립XY누적거리/순변위를분리한다. config명령·structure진입점갱신.
- GPU6·2환경12step측정기확인후9500저장본64환경×2반복·600step상한·각모드256명정상종료(`output/push_hand_motion_9500/REPORT.md`). 보행접촉누적평균0.633m/중앙값0.00955m/p90 3.037m·10cm이상25%/50cm이상19.53%·순변위평균0.638m;학습혼합접촉누적평균0.345m.원래목표방향strict접촉/credited전진량과조건이다르며손단독인과는미확인이다.
- Python/셸문법·diff검사,512행접촉tick/누적거리/순변위집계일관성검사통과. 현재학습을중단/재시작하지않았다.

### PUSH 랜덤크기·목표1~2.5m·RSI 초기 성공 검사

- 사용자 요청으로 `open_push_contact_at_friction06_random_size_goal1to25.yaml`과 전용train/test/VNC/PD RSI wrapper를 추가했다. 중심거리1~2.5m/360°균등·scratch·전용bank/output이며 기존0.5~2.5m 실행과분리했다. 보상/성공·물성·RSI비율/후반요청은유지하고 `radial_uniform` 샘플링을추가했다. config목록/명령·structure진입점갱신.
- CPU29개·셸4개문법·Python문법/문서신규링크/diff검사통과. GPU6 PD12step 후보943/1744·21clip생존,31744초기표본AT성공0·남은거리0.150213~2.499984m·RSI50.16%/후반요청70.28%·AMP/부분reset통과(`output/open_push_friction06_random_size_goal1to25_rsi_check/`).
- GPU6 62환경/31종크기·200reset 목표1.000130~2.499995m·네면/물성/보행/RSI12step·부분reset통과,초기손오차최대0.16838m(`output/push_random_size_goal1to25_sim_check/`). 2048환경별도scratch짧은학습epoch2/frame262144·load_checkpoint=false·모델/161종scalar유한값확인(`output/push_random_size_goal1to25_train_check/`). 새저장본2환경30step재로드평가종료코드0(`output/push_random_size_goal1to25_eval_check/`). 기존장기학습을중단/재시작하지않았고새본학습은시작하지않았다.

### 탑뷰 녹화의 VNC 화면 카메라 동기화

- 저장 센서만 탑뷰이고 VNC viewer는 기존 캐릭터 추적 구도였던 누락을 수정했다. `VIDEO_CAMERA=top`에서 센서/viewer가 같은 탑뷰 위치·목표 계산을 사용하고 `_update_camera`도 교체한다. config 설명 갱신.
- GPU6·1평가환경·12step 정상종료, viewer 카메라 높이33.12m 및 실제 VNC 탑뷰 PNG 확인(`output/open_push_top_viewer_check/viewer.png`). Python 문법·diff 공백 검사 통과. 사용자 장기 녹화는 재시작하지 않았다.

### 원래 viewer와 연속 녹화의 동작 비교·seed 기본값 정렬

- 사용자가제공한 `/tmp/open_push_far_view.py --no_video`를기준으로GPU6·1평가환경·seed7931·120step상한의viewer/녹화를비교했다. 두실행모두69step조기종료,69step전체관측hash·행동·사람/박스상태정확히일치(`output/open_push_far_compare/comparison.json`). 이장면에서는녹화에따른동작변화가없으며장기넘어짐비율은미측정이다.
- 녹화wrapper에추가했던기본seed7931을원래viewer와같은-1(무작위)로정렬하고영상폴더 `metadata.json`에실제torch seed·checkpoint·카메라·해상도를저장하도록수정했다. `SEED`명시지정유지·config비교방법갱신.
- 새기본값GPU6·1평가환경·4step정상종료,실제seed2806/요청seed-1 metadata와마지막3frameMP4저장확인(`output/open_push_far_random_seed_check/`). 셸/Python문법·diff공백검사통과.

### 연속 녹화 탑뷰 선택 옵션

- `view_open_push_far.py`에 `VIDEO_CAMERA=top`을 추가했다. 문을 제외한 두 사람·박스·목표 XY 범위와 화면비로 센서 카메라 높이를 조절하고 저장 폴더에 `_top`을 붙인다. 기본 follow/정책/물리는 유지한다. config 실행명령 갱신.
- GPU6·1평가환경·12step·640×360 정상종료. MP4 두 개6/5frame 전체디코딩·실제탑뷰프레임 확인(`output/open_push_far_top_record_check/`), Python/셸문법 통과. 장기녹화 미실행.

### 연속 녹화 기본 저장 경로 변경

- 사용자 지정 디렉터리에 맞춰 `open_push_far_record.sh`의 기본 출력 루트를 `output_records`로 변경했다. 영상은 `output_records/videos/<실행시각>/part_*.mp4`에 생성되며 명시한 `OUTPUT_PATH`는 우선한다. config/structure 경로 갱신, 디렉터리 존재·셸 문법·diff 공백 검사 확인. 시뮬레이션은 재실행하지 않았다.

### PUSH 4m 접근·문 숨김 연속 녹화

- 임시 viewer를 `view_open_push_far.py`로 보존하고 `open_push_far_record.sh`·`utils/segmented_video.py`를 추가했다. 기존 warmstart29000 YAML/checkpoint 계약과4m 시작·문visual 지하치환 유지, 평가 시작 시 녹화를 명시적으로 요청한다. 별도카메라의깨끗한1280×720영상을기본60초단위MP4로streaming저장하고Ctrl+C/정상종료/SIGTERM에마지막구간도마감한다. 실행시각별폴더로덮어쓰기를피한다. config/structure 실행명령·진입점 갱신.
- 기존 직접headless+카메라명령의exit139를재현하고faulthandler로 `BaseTask.create_sim()` native초기화지점확인(`output/open_push_far_record_check/repro_headless.log`). 기존GUI의GPU/Vulkan/Xvfb초기화를사용하는새경로는GPU6·1평가환경·12step×2회정상종료했다. 640×360MP4네개6/6/6/4frame전부디코딩·설정패널없는실제프레임확인. writer의구간전환/RGB색상/중단후partial구간디코딩확인·Python/셸문법통과. Gymnasium전환/기존학습종료/장기녹화는수행하지않았다.

## 2026-10-13

### PUSH 랜덤크기·중심목표0.5~2.5m·50/50·scratch

- 사용자 요청으로 `open_push_contact_at_friction06_random_size_goal25.yaml`과 전용train/test/VNC/PD RSI wrapper를 추가했다. 중심거리0.5~1m/1~2.5m를각50%로선택하고구간별거리/360°방향균등샘플링·목표Z크기별고정·episode목표고정을적용한다. 최대2.5m의작업영역에맞춰구역간격7m와배치rejection512회를적용했다. 기존정사각목표실험·보상/성공/물성은보존했다.
- 최종설정은scratch이며전이source없음·train의TRANSFER_CHECKPOINT거부·후속resume은새계약저장본만가능하다. config전체목록/실행명령·structure진입점갱신. 전이로준비했던중간짧은검증은 `output/push_random_size_goal25_transfer_check/`로구분했다.
- 최종CPU72개·셸4개문법통과. GPU6 전용PD12step bank는PUSH938/1744후보·21clip생존,31744reset표본조작RSI49.44%/후반요청69.91%·AMP/부분reset통과(`output/open_push_friction06_random_size_goal25_rsi_check/`). 62환경/31종×2물리박스·200reset목표0.500056~2.499601m/가까운비율50.2903%·목표반대네면·보행/RSI12step·부분reset검사통과(`output/push_random_size_goal25_sim_check/`,최대초기손오차0.16838m).
- 별도output의2048환경scratch짧은학습epoch2/frame262144저장·`load_checkpoint=false`/전이설정없음/전이보고서없음·모델/161종scalar유한값확인(`output/push_random_size_goal25_train_check/`). 새저장본2환경30step재로드평가종료코드0(`output/push_random_size_goal25_eval_check/`),최종diff공백검사통과. 장기본학습은시작하지않았다.

### PUSH 랜덤크기10500 보행시작 평가 결과 분석

- 사용자 완료평가 `output/push_random_size_10500_loco_eval/metrics/edge_context_20261013_153604.json`의8반복·512장면/1024명을집계했다. 최종개인AT568/1024(55.47%)·ever570/1024(55.66%)·실제최종접촉409/1024·접촉+AT최종장면0/512·ever7/512·조기종료1/512. 조작RSI없는제공명령과학습tick성공률의차이를구분했다.
- 분석은같은폴더 `REPORT.md`. JSON에checkpoint/실행metadata와기울기/크기별결과가없어로드경로독립확인·전도원인분리는불가함을명시했다.8개repeat와정수건수검증,새학습/평가미실행.

### 종료한 PUSH 랜덤크기 학습 checkpoint 정리

- 사용자 요청으로 PID1483875 종료를 확인하고 `output/open_push_contact_at_friction06_random_size/OpenPushContactAtFriction06RandomSize_12-19-58-19/nn/`에서 최신 `OpenPushContactAtFriction06RandomSize_00010500.pth`만 남겼다. 나머지18개pth 삭제·잔여1개 확인, 약898MB 확보.

### PUSH10000/29000 랜덤 goal 제로샷 시각화

- 기존 contact+AT warmstart의 원래 YAML/checkpoint 계약 검사로 정책을 로드하고 평가 reset 후 goal만 교체하는 전용Python/test/VNC를 추가했다. 두사람PUSH·loco시작·XY±1.5m/거리1~2m·초기박스/두사람1m배제·Z고정·episode목표고정이며 물성/접촉/성공식/학습/bank는 유지한다. 별도goal RNG·초기상태/배정/목표JSONL과 전체장면카메라를 제공한다. config/structure에 재실행명령 기록.
- CPU2048goal의범위/배제/Z/seed재현성과 두checkpoint의실제owner-goal배정 확인. GPU6 각각2평가환경×2반복×30step 재로드종료코드0(`output/open_push_zero_shot_{10000,29000}_check/`);짧은실행이며 정책성공률검증은아니다. GPU6 각각1환경100반복VNC를6102(29000)/6103(10000)로실행해 rollout·HTTP200·실제렌더링캡처 확인(`output/open_push_zero_shot_{10000,29000}_viewer/`). 셸문법/diff확인;본학습미실행.

## 2026-10-12

### PUSH 고정크기·자유 표면 양손 접촉·state≥0.9

- 크기랜덤 PUSH 기준에서 고정1.3m/20kg·목표반대면 비교와 자유 표면 비교 YAML/train/test/VNC를 추가했다. PUSH100%·마찰0.6/바닥1·랜덤시작/목표/binding·구역6.3m·AMP보행33/PUSH67·RSI비율 유지. 자유 표면은 박스 회전을 반영한 전체 표면 XYZ거리와 손별 외향반력으로 양손접촉을 판정하며 면/위치/목표방향을 강제하지 않는다. raw state=(양손최악거리φ+목표XYZφ)/2≥0.9와 실제양손접촉이 개인성공이며 saturation 전에 판정한다. 고정면 비교는 raw state만 요구한다. 기존 옵션없는 성공식은 유지했다.
- 기존 고정크기 PD bank를 재사용하고 policy-only 접촉옵션은 RSI 물리계약에서 제외했다. RSI 초기정렬/AMP 참조의 기존 밀기자세는 유지하며 자유로운 초기자세 분포로 바꾼 것은 아니다. 원본29000 actor/AMP/입력RMS 전이·critic/optimizer/카운터 초기화, 보상metadata로 checkpoint 호환 분리. config/structure에 전용명령 기록.
- CPU131개와 확장 전이55개 통과(일부 중복); 셸문법·새문서링크·diff 확인. GPU6 bank 재사용1024명 reset/AMP/부분reset 통과(`output/push_free_surface_state09_rsi_check/`);2환경200reset·랜덤goal/binding·보행/RSI12step·고정크기20kg/0.6 확인(`output/push_free_surface_state09_sim_check/`). GPU6·2048환경 짧은학습 epoch2/frame262144·88텐서전이/critic81초기화·모델/161종scalar유한값 확인(`output/push_free_surface_state09_train_check/`),2환경30step재로드평가종료코드0(`output/push_free_surface_state09_eval_check/`). 장기본학습/정책성공률 미검증·본학습 미실행.

### PUSH만 학습·박스1.1~1.4m 랜덤·동일밀도

- 사용자 요청으로 전용random_size YAML/train/test/VNC/RSI를추가했다. 두사람PUSH100%·AMP보행33/PUSH67/OPEN0,기준1.3m20kg의밀도9.1033 유지·생성시1cm간격31종균등샘플/환경별크기고정·무게12.1165~24.9795kg·박스마찰0.6/바닥1. 실제크기를생성asset/질량/관성·BPS/관측·접촉/보상/힘법선·RSI·초기/목표Z·보행침투/바닥corner에연결했고작업간격6.3m로확대했다. 원본29000actor/AMP/RMS전이·critic/optimizer/카운터초기화,크기/과제비율checkpoint/bank계약 분리. config/structure 명령갱신.
- CPU118개·문법/링크/diff통과. GPU6 전체31종×2장면의초기양손오차+PD12step검사에서PUSH943/1744·21clip생존(`output/open_push_friction06_random_size_rsi_check/`). 62환경/124물리박스의전체크기/질량/0.6마찰·BPS·PUSH만과제/AMP·랜덤시작/목표/네면·부분reset·보행/RSI통과,최대초기양손오차0.16965m(`output/push_random_size_sim_check/`). 최종bank·2048환경88텐서전이/critic81초기화·epoch2/frame262144저장·모델/161종scalar유한값(`output/push_random_size_train_check/`),2환경30step재로드종료코드0(`output/push_random_size_eval_check/`). 장기본학습/성공률미검증·본학습미실행.

### OPEN/PUSH 26000 단독 평가 완료

- GPU6·각64환경×8반복·각512장면/1024명·seed42·보행시작으로 OPEN/PUSH를 분리했다. 두 최종JSON/8반복 확인: OPEN0/1024, PUSH959/1024(93.65%),양쪽terminate0. PUSH 목표도달후release허용의개인성공을주결과로사용하며,접촉까지요구하는원본장면3.12%는팀목표성공률로사용하지않는다.
- 보고서 `markdowns/open_push_26000_separate_eval_20261012.md`, 원본/집계/로그는 `output/open_push_26000_{open,push}_eval_gpu6_20261012/`. 집계수·링크·diff검증. 기존혼합결과와분리하고 학습/평가코드는변경하지않았다.

### OPEN/PUSH 26000·512장면 평가 완료

- GPU6·headless64환경×8반복·seed42·기본random OPEN/PUSH 보행 시작 평가 완료,최종JSON/8반복 확인. 개인목표506/1024(49.41%)·장면최종6/512(1.17%)·장면ever68/512(13.28%)·terminate1/512. PUSH AT506건·OPEN0건으로집계되며 과제별시도수가저장되지않아PUSH단독성공률은미확정. 개인AT와장면접촉+AT조건차이를명시했다.
- 분석/재실행/원본링크는 `markdowns/open_push_26000_eval_20261012.md`, 원본은 `output/open_push_26000_eval_512_gpu6_20261012/metrics/edge_context_20261012_151625.json`. 집계수/512장면/문서링크/diff확인; 학습설정변경없음.

### PUSH20kg·마찰0.6·랜덤목표 반대면 접촉/RSI

- 사용자 최종선택인 박스20kg·마찰0.6·바닥1의 전용YAML/bank/train/test/VNC를 추가했다. 랜덤XY/yaw·사람시작/heading·담당박스·목표를 유지하고 목표 반대면을 양손접촉/보상/힘법선/RSI/선행이동에 연결했다. RSI는주목표축,실제접촉은목표방향60°이내의가장가까운공통반대면을사용해대각선의면전환을완화한다. 손높이자유·독립AT/release 유지, v2bank/checkpoint계약으로이전면계약직접resume거부. 원본29000actor/AMP/RMS만전이하고 critic/optimizer/카운터초기화; 비교마찰1은별도보존. config/structure 갱신.
- CPU102개 통과. GPU6 RSI OPEN6/13·PUSH1045/1744(비교마찰1 1204/1744). 실제20kg/0.6·200reset·랜덤goal/네면·부분reset·네task 보행/RSI12step 통과,양손오차최대0.13863m(`output/open_push_goal_face06_sim_check/`). 2048환경 source29000의88텐서전이/critic81초기화·epoch2/frame262144 저장·모델/161종scalar유한값(`output/open_push_goal_face06_train_check/`),2환경30step 재로드평가종료코드0(`output/open_push_goal_face06_eval_check/`). 장기 본학습/정책성공률/실제최대밀기질량은미검증.

### OPEN/PUSH 26000 VNC 평가와 포트 선택 수정

- 사용자 요청으로 접촉+AT warmstart26000 checkpoint를 GPU6·1평가환경·OPEN/PUSH 한 명씩(`TASK_GRAPH=open_push`)·1000반복 VNC로 실행했다. 로그는 `output/open_push_26000_eval_gpu6_20261012/viewer.log`, noVNC는6099다. 실제 rollout·GPU6 프로세스·HTTP200을 확인했으며 기존 학습은 유지했다. 완료된 성공률 평가는 아니다.
- `run-gui.sh`가 비어 있는 X display라도 대응 VNC 포트가 점유되어 있으면 건너뛰도록 수정했다. 기존6001점유에서 다음display로 실행 성공, 셸문법 확인.

### 학습 없는 물리 상한과 직접 PD 구동 검사

- 사용자 의도는학습정책성능이아닌물리한계여서 `check_push_physics_envelope.{py,sh}`, `check_push_actuator_load.{py,sh}`를추가했다. GPU6·checkpoint/신경망/optimizer없음,actual humanoid질량50.2997kg·발마찰1·PD effort3.4028e38를확인했다. motorgear100/70은PD토크상한이아니다. 수직손힘/충돌/외부지지없는준정적수평밀기에서 발마찰필요조건493.44N,마찰1의이상적질량상한50.3kg을계산했다. 이것을실제에이전트달성값으로간주하지않는다.
- GPU6·60비학습환경·마찰10종×상한80/100/120%질량의박스조건30개+무외력대조 검증통과. 알려진493.44N을초기높이0.5m에1초가하면 마찰1·40.24kg은1.234m/0.30°·50.30kg은0.0256m·60.36kg은0.0002m이동했다. 이는별도외력에대한상자역학검증이며 에이전트가그질량을민검사가아니다(`output/push_physics_envelope_20261012/REPORT.md`).
- GPU6·864직접PD조건·5초/후반2초반력기록 완료;78조건이지지/근접90%이상,최대평균net손반력159.9N·순간341.2N·관측DOF힘181.6Nm였다. 탐색한자세/구동의기록으로전역물리최대가아니다. 센서대조에서20kg정지박스net Z가중량과불일치하고100N수평외력시바닥net Y가0으로기록돼 그텐서를완전반력/정확한힘상한으로환산하지않는다(`output/push_actuator_load_20261012/REPORT.md`, `output/push_contact_force_unit_check_20261012/`, `output/push_horizontal_contact_unit_check_20261012/`). Python/셸문법·diff통과. config/structure에실행법과제한을기록했고 학습설정/checkpoint/bank는보존했다.

### 실제 정책의 무게×마찰 밀기 검사

- 사용자 요청으로GPU6의비학습`check_open_push_policy_capacity.{py,sh}`를추가했다. 원본접촉+AT29000정책·박스1.3m·바닥마찰1,질량11종×박스마찰0.1~1.0의110조합을같은초기기하/관절상태에서실행하고 접촉정면시작/보행랜덤목표의이동·박스전도·목표도달·손반력을분리한다. 물성은검사actor생성시에만변경하며 학습config/원본checkpoint/bank는보존한다. config/structure에명령과제한을기록했다.
- prepare_sim이후마찰setter는getter만변경되고 실제힘응답은유지되는것을별도대조로확인했다. 첫grid결과는`output/push_policy_mass_friction_grid_20261012_runtime_ignored/`에invalid표시로보존·제외했다. 생성시마찰수정후같은sim의20kg·높이0.5m·120N에서마찰0.1은1.204m/마찰1은0.000019m이동을확인했다(`output/push_creation_friction_grid_calibration/`). 외력검증은별도sim이며 정책검사에는외력을넣지않았다.
- GPU6·220평가환경·110조합·2200 agent-trial 및 마찰1·40평가환경·10무게·480추가trial 완료,물성getter/동일초기상태/유한관측·Python/셸문법·diff통과. 마찰1의안정50cm(박스전도없음)는9kg64.6%·10kg37.5%·12kg6.25%·20kg0%;10kg은안정10cm100%. 현재정책/자세표본의결과이며 절대물리상한이아니다. 보행랜덤목표는전조합안정50cm50%미만이다. 전체CSV/원본/PNG/PDF는`output/push_policy_mass_friction_grid_20261012/REPORT.md`,마찰1추가는`output/push_policy_friction1_mass_boundary_20261012/`에저장했다. 학습은실행하지않았다.

### PUSH 마찰1·ksh식 랜덤목표와 RSI 재검사

- 중지한 접촉+AT warm start 후속으로 `open_push_contact_at_friction1_random_goal.yaml`·전용train/test/VNC·PD bank 생성 wrapper를 추가했다. 박스마찰1.0·초기박스기준XY±1.5m 독립목표·박스/보행사람1m 배제·고정Z·구역간격6.2m를 적용하고 RSI 선행이동도 목표변위를 따른다. 기존실험/bank는 보존한다.29000source actor/AMP/입력RMS 전이·critic/optimizer/보상RMS/카운터 초기화 경로는 source설정/계약과 허용변경을 검사한다. config/structure에 명령과 호환규칙을 기록했다.
- RSI 검사도구가 랜덤binding의 실제 담당박스를 검사하고 회전/랜덤목표를 감사하도록 수정했다. CPU85개·Python/셸문법·diff 통과. GPU1 새물성PD12step은 OPEN6/13·PUSH1201/1744 후보(21clip 모두 생존) 통과;1024reset 조작RSI50.49%·후반요청70.99%·AMP/부분reset 통과(`output/open_push_friction1_random_goal_rsi_check/audit.json`).새/기존장면2환경200reset·네task 보행/조작RSI·부분reset 회귀 통과;새RSI양손오차최대0.16024m(`output/open_push_friction1_random_goal_sim_check/`, `output/open_push_friction1_goal_legacy_regression_check/`).
- 2048환경 실제29000source actor/AMP88개 복사·critic81개 초기화·짧은학습epoch2/frame262144 저장·161종scalar유한값·2환경30step 재로드평가 종료코드0 확인(`output/open_push_friction1_random_goal_train_check/`, `output/open_push_friction1_random_goal_eval_check/`). 장기 본학습과 정책개선은 미검증이다.

## 2026-10-09

### OPEN/PUSH 접촉+AT 전이 학습 경로

- 사용자 요청으로 random-scene4500에서 actor/AMP/scene·AMP입력RMS를 가져오고 critic/optimizer/rewardRMS/epoch/frame/replay를 초기화하는 warm start와 전용YAML·train/test/VNC를 추가했다. OPEN/PUSH50/50·2048환경·보상/물성/RSI 유지, 새 전용wrapper만 GPU를 선택 가능하게 했고 source 보상/물리/참조/network·전체텐서/통계형상을 검사한다. 후속resume에서는 전이를 반복하지 않는다. config/structure에 명령을 기록했다.
- 전이CPU7개 포함56개·Python/셸문법·diff 통과. GPU6·2048환경 실제4500source actor/AMP88텐서 복사·critic81텐서 초기화·짧은 학습·epoch2/frame262144 저장·새optimizer state·161종 scalar 유한값 확인(`output/open_push_contact_at_warmstart_check/`). 새전용test의2환경30step 재로드평가 종료코드0 확인(`output/open_push_contact_at_warmstart_eval_check/`). 본학습은 실행하지 않았고 장기개선은 미검증이다.

### ksh rescue AT 정책 시각화

- 사용자 요청으로 ksh 원본의 `distill_pth/ApproachScenarioStage1RescueAtKLClimb50_00008000.pth`를 원본 AT 전용 VNC 데모로 재생했다. GPU6·1환경·2명 HOLDING+AT, 목표중심0.6/1.0m×중간/옆 위치·10반복이며 output은 이 repo의 `output/ksh_at_rescue_viewer/`로 분리했다. checkpoint 로드·noVNC HTTP200·실제 화면 캡처 렌더링 확인. 전체 실험 최고 AT 성공률 순위는 남은 기록으로 확인하지 못했으며 재측정하지 않았다.

### PUSH 접촉과 기존 AT 배치 분리

- 사용자 요청으로 `open_push_push180_contact_at.yaml`과 전용 train/test/VNC를 추가했다. PUSH는 양손 뒤면 접촉, AT는 기존 거리 state/progress·근접/높이 성공을 독립 계산해 합산하며 AT에 손 이동credit·속도·기울기 조건을 요구하지 않는다. AT 성공 시 HOLDING+AT식 접촉 reward 포화로 release를 허용한다. 실제 장면 무작위화·물성·AMP·RSI는 유지하고 새 checkpoint 보상 계약과 scratch output으로 기존 실험을 보존한다. config/structure에 실행법·수식·호환 규칙을 기록했다.
- CPU74개(신규4개 포함)·Python/셸문법·스크립트경로·diff 통과. GPU6 비학습2환경에서200reset 배치/독립binding4조합·부분reset·네task 보행16reset/조작RSI32reset×12step 관측/보상/AMP 유한값 및 새 보상계약 확인 통과(`output/open_push_contact_at_sim_check/audit.json`, RSI 양손 오차 최대0.1609m). 장기학습·정책 개선·GUI/VNC는 미검증이다.

### PUSH180 실제 장면 무작위화·크기 고정

- 공통 코드 변경 후 기존 구역 교환 실험의 실제 시뮬레이션 회귀 검증도 통과했다(네 task·두 배정·부분 reset·RSI, `output/open_push_random_lanes_scene_regression_check/`).

- 사용자 요청으로 SIT/HOLDING식 배치변화를 PUSH에 적용한 `open_push_push180_random_scene.yaml`·전용train/test/VNC·scene sampling/전이/검사도구를 추가했다.1.3m·20kg·마찰0.6와reward/AMP/초기화비율을 유지하며 박스XY/yaw·1~2m보행방향/heading·중복없는담당박스를무작위화한다. 작업영역분리·초기body침투검사·reset yaw의local전방2m고정목표·RSI선행진행·관측/접촉/credit/진단을연결하고 부분reset을보존한다. OPEN고정힌지배치는유지한다.
- 이전단일PD통과후보를강체XY/yaw로재사용하는출처/scene계약전용bank를생성하고 원본을보존했다. CPU70개·Python/셸문법·diff통과. GPU6비학습2환경200reset의XY/yaw/독립binding4조합·거리1.005~1.998m·부분reset·네task보행16reset/조작RSI32reset×12step·접촉/목표/관측/AMP검사통과(RSI양손오차최대16.1cm,`output/open_push_random_scene_sim_check/audit.json`). 실제3500전이state전체/PD후보동일·물성/reward/AMP보존,2048환경3500→3502 resume/저장확인(`output/open_push_random_scene_resume_check/audit.json`),새test wrapper의2환경30step복원종료코드0 확인. 기존학습·checkpoint유지. 새장기학습·개선효과·GUI/VNC는미검증이다.

### PUSH180 담당 구역50/50 무작위화 별도 실험

- 매번 초록은 목표를 지나 이동하고 파랑은 손을 대고 정지하는 구역별 정책 편향을 줄이기 위해 `open_push_push180_random_lanes.yaml`·전용train/test/VNC를 추가했다. 물체 배치는 고정하고 reset별 owner→물리구역 배정을50/50 교환하며 관측·graph 논리슬롯·보행/RSI·목표·접촉/credit·색상을 같은배정으로 연결했다. 보상/성공·물성·AMP·기존bank를 유지하고 부분reset은 해당장면만 변경한다. 평가 `OPEN_PUSH_LANES=random/fixed/swapped`·checkpoint 구역계약·전용전이/검사도구와 config/structure 실행법을 추가했다.
- CPU총66개(기존/신규62개+이동credit 추가4개)·Python/셸문법·diff통과. GPU6 비학습2환경 네task×두배정12step·비대칭접촉력/목표/관측/AMP·200reset교환49.5%·부분reset·네task조작RSI48reset×12step 통과(`output/open_push_random_lanes_sim_check/audit.json`). 원본3500의 모델/optimizer/정규화/카운터 등 전이state전체동일·잘못된조건거부검사 및 GPU6·2048환경3500→3502 resume/저장·새계약/원래reward/bank확인(`output/open_push_random_lanes_resume_check/audit.json`), 새test wrapper의2환경30step교환배정복원 종료코드0 확인. 원본checkpoint·학습PID2545261 유지. 새본학습·구역편향개선·GUI/VNC 성능은 미검증이다.

### 실제 에이전트 손 밀기 영상 비교

- 사용자 요청으로 이전검사 clip0/16을 실제 에이전트PD로 다시 실행해 원본/낮춘팔×팔만/전체동작×정상/손전용충돌16개 H.264 영상을 생성했다. root강제이동·상자외력 없이 GPU6·2환경이며 넘어짐 프레임에서 영상을 고정하고 이동/양손credit를 표시한다. 별도 `render_physics.py`·`videos/audit.json`·동시재생/배속/조건전환 `index.html`을 같은 검사output에 저장하고 localhost6090으로 제공했다.
- 8개 재생 정상종료·16영상각90frame/H.264·페이지/영상HTTP200·추출프레임실제구도·JS문법 확인. 원본학습/config/bank 유지. 화면의 값은 영상 생성 시 재실행 측정이며 전체21clip 집계와 구분한다.

### PUSH180 실제 손 밀기 물리 검사 (학습 유지)

- 사용자 요청으로 현재20kg·마찰0.6·1.3m 상자를 GPU6 비학습2환경에서 검사했다. 외력12조건·21clip×3후보×3방식189rollout과 초기속도0/다리·몸통 목표유지/팔8DOF만 재생42rollout을 정상 충돌·손 전용 상자 충돌로 비교했다. config/AMP/RSI bank/checkpoint를 변경하지 않고 학습PID2545261 유지를 확인했다.
- 높이0.8m/160N/2초에서31.70cm 미끄럼,120N은 거의정지.1m/160N은 넘어짐. 보행제거·손 전용 충돌에서 원본AMP팔 전진 중앙값2.48cm/최대25.23cm(21개중4개≥5cm),낮춘팔0.048cm/최대7.23cm(2개≥5cm). 물리적 손 이동 가능성은 확인했지만 안정적2m 정책 성공은 미검증이다. PD균형/초기상태·충돌필터의 자기충돌 영향과 공유episode 종료 한계는 `output/push180_hand_physics_audit_20261009/REPORT.md`에 기록했다. 두 최종 검사 정상종료·총231rollout 집계 완료; config/structure에 결과 경로를 연결했다.

## 2026-10-08

### PUSH180 새 설정으로2000 resume·tmux 본학습

- 사용자 요청으로 `transfer_open_push_push180.{py,sh}`를 추가해 기존마찰0.5/revision5·1.5m/2모션 계약과 양쪽PD bank를 검사하고 PUSH2m/넓은배치/21모션·새RSI·reward metadata만 전이했다. 원본2000 checkpoint를 보존하고 target output의 `transfer/epoch2000_push180_2m_wide.pth`와 hash/변경 기록을 생성했다. 모델/optimizer/정규화/epoch/frame 및 graphstate 완전동일 비교·잘못된 물리/샘플링/revision/optimizer누락/clip후보누락5종 거부 통과.
- GPU6·2048환경 확인용output에서2000→2002 resume/저장·목표2m/새manifest/계약 확인(`output/open_push_push180_resume_check/audit.json`). 기존 학습은 사용자가 종료한 상태를 확인하고 `koo_train:0.0`에서 별도 본학습output으로 원본전이2000부터 시작했다. 사용자 최신 지시에 따라 기존상한1000000을 유지했다(9000 미적용). launcher/log/현재PID/run은 target output의 `latest_launch.json`에서 확인한다. config/structure 실행법·전이 제한을 갱신했다. 장기정책 성능은 아직 확인하지 않았다.

### PUSH180 AMP 21개·2m RSI bank 확장 (학습 유지)

- 사용자 선택대로180 계열21개(기존2개 보존+신규19개)를 `dataset_open_push_push180.yaml`에 같은weight로 추가했다.270/360은 제외하고 metadata의 첫 멈춤/손 해제 전까지30FPS retarget했다. crop/events 선택자료·전용2m 넓은배치 config/bank·train/test/VNC·bank 생성 wrapper와 config/structure 명령을 추가했다. 기존manifest/bank/현재학습은 유지하고 새학습·checkpoint전이는 실행하지 않았다.
- RSI 마지막frame float32시간의 미세한 길이 초과를 clamp하고 회귀검사를 추가했다. PD 생성기는 기본2환경을 보존하며 후보당2환경씩 묶음 검사하는 옵션을 추가했다. 프레임뷰어 `--motion-file`·물리viewer `RSI_MOTION_FILE` override로 확장목록을 연결했다.
- 전체21개 실제변환 DOF/속도/접지/finite/crop·기존2개 hash 통과(최대endpoint오차2.02cm 미만),CPU55개 통과. GPU6 비학습16환경12step PD OPEN13/13·PUSH1048/1744·21clip모두후보확보·1024reset/부분reset/AMP/고정2m목표 통과(`output/open_push_hinge_friction_push2m_wide_push180_rsi_check/audit.json`). 기본2환경1000reset(bank복사본)·headless RSI3표본/12step·22clip JS프레임/후보/끝정지·HTTP200·Firefox 렌더링 통과.6090의 `/push180/rsi_timeline.html`에 출력했다. Python/셸 문법·diff 확인. 장기정책 성능과 기존checkpoint 직접resume는 검증하지 않았다(계약상 직접복원 불가).

### 요청 PUSH 71개 원본 검토·선별 추출

- 사용자 목록71개(180:21/270:24/360:26)의 temporal label을 정리하고 archive에서 신규69개 BVH만 추출했다. 기존2개 원본/AMP manifest/RSI bank/실행 중 학습을 보존했다. 목록·동작 구간·파일별SHA256·검증은 `bones_seed/validation/requested_push_71/`에 기록했다.
- 71개 모두 event 시간 순서·BVH parser/FK·78노드/240채널·정적Root·유한값 검증 통과, 누락0개. 270은옆몸/한손,360은등/엉덩이 밀기여서 현재 양손 접촉 credit과 불일치한다. 적용 범위 선택 대기이며 새retarget·AMP 추가·RSI 재생성·학습은 아직 실행하지 않았다.

### PUSH2m 문·박스·에이전트 구역 간격 확대

- 사용자 요청으로2m config에 `openPushLayout`(doorX4/pushX-4/Y구역10m)·envSpacing14를 추가했다. 문/박스 가로 간격2.5→8m·구역5→10m, 보행/조작RSI/고정목표/부분reset을 같은 배치로 연결했다. 구역 검증·checkpoint/bank 계약·전용wide bank/output과 viewer config override·실제 위치 카메라를 추가했다. 배치 키가 없는 기존1.5m 좌표/계약·실행 중 학습은 유지하고 새 학습/1500전이는 실행하지 않았다.
- CPU54개·Python/셸 문법·diff 통과. GPU6 비학습2환경PD OPEN13/13·PUSH154/171·1000reset/부분reset/고정목표/AMP 통과(`output/open_push_hinge_friction_push2m_wide_rsi_check/audit.json`). 네 과제×보행/RSI160scene·960PDstep·목표/부분reset 검사에서 사람 관절 중심 사이 최소8.28m 확인(`output/open_push_push2m_wide_layout_check/layout_audit.json`). RSI headless3sample/12step·6090 프레임뷰어 재생성/JS조작 검사 통과. 이는 정상 배치/짧은PD 확인이며 정책 이탈의 모든 충돌·장기 성능 보장은 아니다.

### RSI 전체 프레임 시각화

- 사용자 요청으로 `build_open_push_rsi_timeline.{py,sh}`·`viewer_assets/open_push_rsi_timeline.html`을 추가했다. AMP clip 전체를 무생략 재생하고 RSI/후반 후보를 표시하며 처음부터·한 프레임·후보 이동·배속·원본/낮춘 팔 겹침·회전/확대를 제공한다. 끝에서는 정지하고 물리 접촉/정책 rollout과 구분했다. config/structure에 생성/서버접속 명령을 기록했다.
- CPU로2m·현재1.5m 후보 뷰어 생성(OPEN103/PUSH367·352frame), JS3clip의 프레임/후보 이동·끝 정지 검사와 Firefox 실제 렌더링/캡처 통과(`output/open_push_rsi_timeline_browser_check/screenshot.png`). 6090 HTTP200·생성 파일 일치·Python/셸 문법·diff 확인. 학습PID2217554 유지·학습/정책 실행 없음.

### PUSH 목표2m 별도 설정 (학습 유지)

- 사용자 요청으로 현재 문 마찰0.5 실험의 PUSH 목표1.5→2m인 `open_push_late_rsi_hinge_friction_push2m.yaml`·전용 bank·train/test/VNC·bank 생성 wrapper를 추가했다. 기존1.5m 설정/학습은 유지하고 새 학습·1500 checkpoint 전이는 실행하지 않았다. config/structure에 실행법과 기존 checkpoint 직접 복원 불가를 기록했다.
- RSI 감사의 목표1.5m 하드코딩을 실제 config 거리로 수정했다. CPU49개·wrapper4개 bash 문법·YAML 변경 범위·diff 검사 통과. GPU6 비학습2환경12step PD에서 OPEN13/13·PUSH114/171 후보 통과,1000reset·부분 reset·고정2m 목표·AMP 유한값 감사 통과(`output/open_push_hinge_friction_push2m_rsi_check/audit.json`). 지정된 `tokenhsi` 환경에서도 bank 재사용1000reset·부분 reset·AMP 검사 종료코드0 확인(`output/open_push_hinge_friction_push2m_reset_check/audit.json`). 2m 정책 성능·resume는 미검증이다.

### OPEN 마찰 변경 checkpoint 전이·resume 검증

- 사용자 요청으로 `transfer_open_push_hinge_friction.{py,sh}`를 추가했다. revision5 마찰0→0.5와 전용 bank 경로/hash 변경만 허용하고 나머지 config·bank 계약·PD 후보 동일성을 검사한다. 원본을 덮어쓰지 않고 모델/optimizer/정규화 통계/epoch/frame을 보존하며 source hash·epoch를 기록한다. config/structure에 전이·resume 명령과 전체epoch 상한 의미를 갱신했다.
- 실제1000 checkpoint 변환 전후 원본 학습 상태 완전 일치와 잘못된revision·spring·reward·후보·bank hash5종 거부 확인. GPU6·2048환경MAX_ITERATIONS1001에서1000→1002 resume/저장·새 마찰0.5 계약 확인(`output/open_push_hinge_friction_resume_check/audit.json`). 실제1500 checkpoint 변환본은 새 실험의 `transfer/epoch1500_friction05.pth`에 준비했다. bash 문법·diff 검사 통과. 기존PID1644702는 유지했으며 새 조건의 본학습/정책 성능 평가는 실행하지 않았다.

### OPEN 초기 밀기 저항용 힌지 마찰 비교

- 사용자 요청으로 닫힘 spring은 유지하고 문 hinge friction0→0.5인 `open_push_late_rsi_hinge_friction.yaml`을 추가했다. 마찰은 양방향 회전 저항이며 초기 밀기만 별도 제한하는 것은 아니다. 기존 late RSI config/bank/output·학습PID1644702는 유지하고 전용 bank·train/test/VNC·bank 생성 wrapper 및 config/structure 명령을 분리했다.
- GPU6 비학습2환경12step PD에서 OPEN13/13·PUSH145/171 후보 통과,1000reset·partial reset·AMP 유한값 검증 통과(`output/open_push_hinge_friction_rsi_check/audit.json`). 전용 bank 계약 일치·YAML 변경 범위(마찰/bank만)·wrapper4개 bash 문법·diff 검사 통과. 새 학습/정책 평가는 실행하지 않았다. 기존 물리 checkpoint 직접 resume는 계약상 거부되며500 가중치의 물리 변경 전이 경로는 미구현이다.

### PUSH 양손 중점 XY 보상·높이 자유

- 사용자 요청으로 PUSH 손 state를 뒤면 중앙의 양손 중점 yaw-local XY 거리로 변경했다. 고정 좌우 두 점·손 높이 제한을 제거하고 각 손의 뒤면XY18cm 근접·양손/박스 전방반력0.5N 접촉으로 전진 credit을 인정한다. 기존 기울기0.35rad·중심높이±8cm·RSI 초기 progress/손 전진·성공 규칙·OPEN을 유지한다. RSI0.8m 초기 정렬·PD bank·AMP 원본은 재사용한다.
- checkpoint revision5·`*_cube_low_midpoint_xy_hand_credit` 출력으로 이전 보상과 분리하고 config/structure·GPU6 학습/평가/VNC 명령을 갱신했다. 손별 실제높이/높이차·중점XY거리 진단을 추가했다. 학습이 개선되지 않고 높은 손/기울기 위반이 지속되면 낮은 높이를 부드럽게 선호하는 보상을 후속 추가하기로 config에 기록했다(현재 미적용).
- 관련 CPU49개 통과: 다른 손높이·목표점 배정 제거·중점 오차·뒤면 폭/양손반력/yaw 및 기존 기울기/높이/progress/reset/RSI 회귀 포함. wrapper6개 bash 문법·diff 검사 통과. GPU 학습/평가는 실행하지 않았으며 기존 프로세스는 유지했다. 높이 자유 정책의 장기 성능은 미검증이다.

### OPEN/PUSH 확인용 출력 정리

- 사용자 요청으로 `output/`의 OPEN/PUSH 확인 학습·평가·physics/RSI 검사·뷰어 출력30개(239.5MiB)를 삭제했다. 본학습 `open_push_90_standing`, `open_push_late_rsi_cube_low`, `open_push_late_rsi_cube_low_joint_hand_credit`3개와 파일23개의 크기/mtime 보존을 확인했다. 학습용 코드/YAML·모션·RSI bank·회귀 테스트는 유지했다.
- 실행 중 사용 경로가 없는지 확인 후 삭제했고 config/structure에 보존 범위를 기록했다. 아래 과거 검증 기록의 확인용 output 경로는 정리로 삭제된 artifact를 가리킬 수 있으며 기록된 검증 결과는 당시 실행 결과다. 삭제 목록은 `/tmp/open_push_output_cleanup_20261008.json`이다. 문서 diff 검사 통과; 학습/평가는 실행하지 않았다.

### OPEN/PUSH RSI 초기 진행도 복원·손 조작 치팅 제한 유지

- 사용자 요청대로 RSI 초기 진행도 보상을 복원했다. OPEN은 `min(현재 각도, reset각도+손 인정 개방각)`, PUSH는 `min(실제 전체 진행도, RSI 초기 진행도+손 인정 전진/전체거리)`를 현재 손 접촉(OPEN 자세/PUSH grounded 포함)으로 제한한다. 몸으로 먼저 이동한 뒤 손만 대도 추가 progress를 얻지 못하며 초기분은 보상한다. 성공 자격은 RSI 이후 실제 손 조작 누적·OPEN5초 지지 규칙을 유지하고 초기 RSI는 성공 실적에 포함하지 않는다.
- checkpoint 계약revision4·`*_cube_low_rsi_progress_hand_credit` 출력으로 보상 버전을 분리했다. 기존 train/test/VNC wrapper·bank/샘플링·물리·네트워크 형식은 유지하고 config/structure의 현재 식과 GPU6 명령을 갱신했다. 이전revision1/2/3 checkpoint의 직접 resume/test는 거부한다.
- CPU49개 통과: RSI 초기 손 접촉에서 progress 지급/접촉 없으면0/성공 자격0, 몸 선행 이동 후 손 접촉의 추가progress 차단 및 기존 손 조작/유지/reset 회귀 포함. bash 문법·diff 검사 통과. 이번 수정은 보상 식 변경으로 GPU 학습/평가는 실행하지 않았으며 기존 프로세스를 종료·재시작하지 않았다. 새 보상의 본학습 성능은 미검증이다.

### OPEN RSI 이후 손 개방각·5초 지지 성공

- OPEN도 RSI 적용 후 각도에서 손 실적/유지타이머0으로 시작한다. 같은 physics tick의 손 접촉/서 있는 자세에서 증가 각도만 인정하고 닫힘은 차감·실제 순개방으로 상한을 둔다. 남은 성공각을 손으로 연 뒤 기존90도·손 지지·자세로5초 유지해야 성공한다. 개방 progress를 손 실적에 연결하고 성공 시 손 거리 state 최대값 덮어쓰기를 제거했다. 유지의0.1초 접촉 공백 허용은 유지하며 PUSH 규칙은 그대로다.
- PD검사 bank 원본/자세를 수정하지 않고 OPEN85도 이하 후보만 선택한다(현재 bank13개 모두68.1도 이하). 후반 요청/선택은 유효 pool에서 유지한다. OPEN 인정/필요각도·손접촉·자격·유지시간 진단을 추가했다. checkpoint 계약revision3·전용 `*_cube_low_joint_hand_credit` 출력으로 이전 보상 정책을 분리하고 config/structure·GPU6 명령을 갱신했다.
- CPU46개 통과: 몸 개방 후 마지막 손접촉 성공 차단·손 개방 후5초 유지/release·성공 손거리 보상·닫힘/reset 및85도 후보필터/후반선택 포함. GPU6·2048환경 MAX_ITERATIONS2(3epoch) 학습·checkpoint 저장·OPEN 새 진단 유한값 확인(`output/open_push_joint_hand_credit_train_check/audit.json`), 같은 checkpoint2환경·30step headless 복원 평가 통과(`output/open_push_joint_hand_credit_eval_check/metrics/`). 기존 학습PID4143193은 유지하며 새 본학습의 정책 품질은 미검증이다.

### PUSH 양손 접촉 전진 기반 보상·성공

- 순간 손 접촉 후 몸으로 밀어도 성공 자격이 유지되던 latch를 제거했다. 같은 physics tick의 양손 접촉/grounded에서 실제 목표거리 감소만 누적하고 후퇴는 차감·reset 순전진으로 상한을 둔다. RSI 배치 뒤 남은 거리를 다시 기준으로 잡아 선행 이동을 제외하며 목표 허용영역에 필요한 접촉 전진을 확보해야 성공한다. 성공 전 progress는 누적 접촉거리와 현재 양손 접촉을 따른다. 정상 완료 후 release는 허용하되 state 손 거리 항의 성공 최대값 덮어쓰기는 제거했다. OPEN은 유지한다. 손/몸 동시 접촉 배제나 GPU collision pair 식별을 추가한 것은 아니다.
- 새 거리/필요거리 TensorBoard 진단과 checkpoint 계약revision2를 추가했다. train/test/VNC 기본 output을 `*_cube_low_hand_credit`로 분리하고 이전 checkpoint 직접 복원은 거부한다. 물리 RSI bank는 재사용한다. config/structure를 갱신했다.
- CPU42개 통과: 짧은 접촉→몸 전진·몸 전진 후 마지막 접촉·RSI 선행 배치·정상 완료/release·공중/왕복·부분 reset 포함. GPU6·2048환경 MAX_ITERATIONS2 제한 학습3epoch·checkpoint 저장·새 진단 유한값 확인(`output/open_push_hand_credit_train_check/audit.json`). 같은 새 checkpoint의2환경·30step headless 복원/평가 통과(`output/open_push_hand_credit_eval_check/metrics/`). 초기 성공률0이며 본학습 성능/몸 접촉 비중은 미검증이다. 기존 학습PID4143193은 종료·재시작하지 않았다.

### OPEN/PUSH 정책 VNC GPU UUID 검사 수정

- VNC의 GPU6→UUID 전달 뒤 내부 test wrapper가 숫자6만 허용해 종료되던 오류를 수정했다. late RSI/unified 정책 test는 `nvidia-smi`로 실제 물리 index6을 검사하고 UUID는 유지한다. config에 전달 규칙을 기록했다.
- 두 test wrapper의 bash 문법 검사 통과. GPU6·1환경·epoch4000 checkpoint로 late RSI VNC6087 복원 및120step 이상 rollout·화면 캡처·포트 유지 확인. 기존 학습 PID4143193은 유지했다.

## 2026-10-07

### PUSH 정육면체·0.8m 손 목표·팔 RSI 및 AMP 충돌 로그

- 사용자 승인한1.3m 정육면체·양손 목표 높이0.8m를 공동 unified/late RSI에 적용했다(20kg·마찰0.6 유지). 어깨에서 팔을 회전해 손 높이를 낮추고 torso/발/팔 길이를 보존한 뒤 XY pair/6cm stand-off를 정렬한다. 어깨 DOF·속도와 reset AMP 이력도 보정하며 전문가 모션은 원본을 유지한다. 출력`*_cube_low`·새 bank/revision3로 이전 크기/목표 artifact와 분리했다. config/structure에 현재 설정/실행법/진단 의미를 기록했다.
- TensorBoard에 PUSH 손별 거리/높이 오차·양손 근접/접촉·반력·기울기/grounded·속도·원래 시작/reset 이후 중심/바닥 edge 전진과 RSI reset 거리/횟수를 기록한다. OPEN/PUSH task/AMP 원 보상·가중 기여 및 PUSH 접촉/근접 표본의 조건부 보상도 기록해 AMP 경쟁을 확인한다. 물리 진단은 현재 보상/성공 조건을 바꾸지 않는다.
- CPU46개 통과(거리별 접근 식·어깨 FK/속도·발/팔 길이·로그 scalar/기울어짐 진단 포함). GPU6 비학습2환경 PD에서 OPEN13/13·PUSH145/171(clip별69/76),1000reset 조작48.6%/후반68.72%·partial reset/AMP 통과(`output/open_push_cube_low_rsi_check/audit.json`).2048환경 MAX_ITERATIONS2 제한 학습3epoch/checkpoint 저장·새 PUSH 진단37개×3회 유한값 기록 확인(`output/open_push_cube_low_train_check_v3/tensorboard_audit.json`), 초기 손 거리 평균 약6.5cm. 같은 checkpoint2환경/30step 평가와 headless RSI 뷰어3sample/12step 통과. rollout 진단의 사람 축/정수 counter 오류를 수정한 최종 검증 결과다. PUSH 접근 보상/XY진단의0.5m 포화 식이 적용된 소스로 학습했다. 기존 장기 학습/뷰어 프로세스는 변경하지 않았으며 새 본학습·장시간 성공/AMP 적합성은 미검증이다.

### PUSH 접근 보상 0.5m 포화

- 사용자 요청으로 공동 OPEN/PUSH의 PUSH 접근 state와 XY 진단 점수를 `1/(1+max(distance-0.5,0))`로 맞췄다. 사람 root→박스 중심 XY거리0.5m 이내에서 최대 접근 보상0.2를 지급하며 현재 HOLDING/OPEN 접근 식과 같다. unified/late RSI 공통 코드에 적용하고 조작 progress·양손 interaction·접촉/성공 조건은 유지한다. config의 식을 갱신했다. 기존 프로세스는 재시작하지 않았으며 새 실행부터 적용된다.
- 관련 CPU28개 통과(`test_open_push.py`):0/0.2/0.5/1/1.5m에서 보상·진단 점수 일치와 기존 접촉/성공·graph/순열 회귀 확인. GPU 시뮬레이션·학습은 실행하지 않았다.

### PUSH 손 높이·정육면체 물리 비교

- 사용자 요청으로 비학습2환경 `check_open_push_physics.{py,sh}`를 추가했다. 기존0.8×0.6×1.3m/정육면체1.3m를20kg·동일 마찰로 비교하며 대칭 양손과 같은 합력의 작용 높이와 원본 AMP 관절 PD를 분리한다. 외력 대조군에서는 사람을10m 떨어뜨려 우연한 충돌을 배제하고 상자 중심 이동과 기울기/바닥 edge 이동을 구분한다. 기본 YAML·RSI bank·실행 중 학습/뷰어는 변경하지 않았다. config/structure에 실행법·한계를 기록했다.
- GPU6 외력24조건·사람PD120회 비교 완료(`output/open_push_physics_comparison_verified/comparison.json`).160N에서 기존 상자는 작용 높이1.05/0.8/0.6/0.4m 모두 기울어졌고 정육면체도1.05m에서는 기울어졌다. 정육면체+0.8m는 기울기0·바닥 전진31.56cm지만200N에서는 넘어졌다. 무외력 대조군 최대 전진0.008mm/정육면체0. 원본 자세 PD의1.05m 목표에서 기존 바닥 이동 최대0.46cm/정육면체11.89cm를 기록했다. 목표만0.8m로 내리면 초기 손 거리 중앙값 약28cm(허용18cm 초과)이므로 낮은 손 자세/AMP 적합성과 학습 정책 성공은 미검증이다. Python/셸 문법·diff 검사 통과.

### PUSH 뒤면 양손 목표·동시 접촉·RSI 정렬

- 사용자 승인한 두 점(상자local 오른손X+20cm/왼손X-20cm·뒤면Y-30cm·바닥 기준Z1.05m)을 보상/접촉/RSI에 연결했다. 손 interaction은 더 먼 손의 제곱 거리로 계산하며 양손 모두 각 목표18cm 이내·각 반력0.5N 이상 및 박스 전방반력0.5N 이상을 같은physics tick에 만족해야 밀기 접촉 이력을 갱신한다. 목표는 박스 이동/회전을 따른다. RSI는 손 pair XY를 한 번의 rigid yaw/translation으로 맞추고6cm stand-off·원본Z를 유지한다. OPEN 한 손·상자1.3m·AMP/초기화 비율·유지 타이머/성공조건은 유지한다. 새 양손 output/bank·계약revision2로 이전 한 손 artifact와 분리했다.
- CPU37개 통과: 한 손 위치/반력 부족·손 목표 바뀜 거부, 박스 회전/양손 목표·발Z를 보존한 pair 정렬 및 기존 OPEN/graph/gradient 회귀. GPU6 비학습2환경12step PD에서 OPEN13/13·PUSH166/171 후보 통과,1000reset 조작48.6%·후반 요청68.72%·partial reset/고정 목표/AMP 통과(`output/open_push_two_hand_rsi_check/audit.json`, `/tmp/open_push_two_hand_rsi_build.log`). GPU6 VNC6086을 양손 검사 화면으로 교체하고 오른손 빨강/왼손 청록 목표·PUSH 자세·sample 목표좌표 캡처 확인(`output/open_push_rsi_view_two_hand_live/`, `/tmp/open_push_rsi_view_two_hand_vnc.log`). 장기 학습·실제 양손 지속 접촉 정책 성공은 미검증이며 학습 프로세스 변경/새 학습은 하지 않았다.

### PUSH 상자 높이1.3m·크기에 맞는 RSI 정렬

- 사용자 요청/AMP 손 높이 약1.07~1.10m의 상단 여유를 위해 공동 loco-only·후반RSI PUSH 상자 높이를1.1→1.3m로 늘렸다(폭0.8/깊이0.6m·20kg 유지). 두 실험 기본 output에`_h13`을 붙이고1.3m RSI bank를 별도로 생성해 이전 크기의 artifact를 보존한다. RSI 뒤면 손 높이 clamp와 PD 감사의 바닥 중심 기준을 실제 상자 크기로 계산한다. 손 보상은 뒤면 전체 최근점 거리·양손 최솟값, 접촉은 한 손 허용으로 유지하며 config에 명시했다.
- CPU35개 통과. GPU6 비학습2환경12step PD 재검사에서 OPEN13/13·PUSH155/171 및1000reset 조작48.6%·후반 요청68.72%·partial reset/고정 목표/AMP 유한값 통과(`output/open_push_late_rsi_h13_check/audit.json`, `/tmp/open_push_rsi_h13_build.log`). 검사 VNC6086을 새 크기로 다시 열어 박스 중심Z0.65m/reset·모션정보·렌더링 실행 확인(`output/open_push_rsi_view_h13_live/latest_sample.json`, `/tmp/open_push_rsi_view_h13_vnc.log`). 기존 검사 뷰어만 교체했고 학습 프로세스는 변경하지 않았다. 새 크기의 학습/정책 성공은 미검증이다.

### OPEN/PUSH 후반 RSI 시각 검사 뷰어

- 사용자 눈으로 RSI 배치를 검사할 수 있도록 checkpoint 없는1환경 `view_open_push_rsi.py`와 로컬/VNC wrapper를 추가했다. 기본 OPEN/PUSH 조작·후반 강제 모드와 학습 혼합 모드, 새 샘플·사람별 카메라·12step sampled DOF PD 검사 키를 제공한다. 화면에 task/clip/time/원본phase/후반 여부를 표시하고 sample JSON을 기록한다. 검사 모드 변경은 프로세스 내부에만 적용하며 학습 YAML/후보 bank·기존 학습은 수정하지 않는다. config/structure에 실행법과 정책 평가와의 차이를 기록했다.
- GPU6 headless3sample reset/12step PD·유한값 확인 통과(`/tmp/open_push_rsi_view_check.log`). GPU6 실제 VNC6086에서 OPEN/PUSH 초기화·모션정보 표시 및 N/C/Space 키 동작·화면 캡처 확인(`output/open_push_rsi_view_live/{latest_sample.json,samples.jsonl}`, `/tmp/open_push_rsi_view_vnc.log`). GUI wrapper의 index→UUID 전달과 debug line float32 입력을 수정했다. wrapper 문법·diff 검사 통과. 뷰어를 사용자 검사용으로 실행 유지하며 학습 정책의 성공/장시간 안정성 검사는 아니다.

### OPEN/PUSH PD 후보 후반 RSI 별도 실험

- 사용자 요청으로 기존 task별 보행/조작50/50과 ksh/Door의 후보 뒤쪽30%·70% 요청 비율을 참고한 `open_push_late_rsi` env·train/test/VNC를 추가했다(기존 공동 train/network YAML 재사용). OPEN은 보행50/문50, PUSH는 보행50/각 clip25다. 원본 물체 궤적이 없어 문각도/박스 전진량을 phase 기반 기하 proxy로 배치하고 손 앞6cm에 사람 자세/속도를 함께 정렬한다. 물체 원래 기준/고정 목표를 유지하며 유지 시간·접촉/성공 이력을0부터 시작한다. 평가 초기화는 보행100% Start다. 원래 설정/실행 PID3576291은 보존하며 새 bank hash/RSI checkpoint 계약으로 이전 checkpoint 직접 resume/test를 차단한다. config/structure에 실행법·proxy 한계·로그 태그를 기록했다.
- GPU6 비학습2환경12step PD 검사에서 OPEN13/13·PUSH155/171 후보 통과(clip별72/83개; `output/open_push_late_rsi_check/{screen_counts,accepted,rejected}.json`). OPEN은 문 선 자세 조건, PUSH는35cm 이내 foot body의 실제1N 이상 지지를 검사한다. 초기 PUSH에 문 posture를 적용한 후보 거부는 별도 기준으로 수정했다. 1000개 학습 reset 표본에서 조작48.9%·조작 중 후반 요청69.33%,4가지 task 조합·task별 reference 매칭·AMP history label/유한값·부분 reset·고정 목표·초기 유지/eligibility0 통과(`output/open_push_late_rsi_reset_check/audit.json`). 감사 도구의 eval→train 전환 시 skill 목록 복원 누락을 수정했다.
- CPU35개 통과: clip 동일비중·후반 확률·후보 계약 거부·강체 정렬·기존 config 보존 및 공동 reward/순열 회귀. GPU6·2048환경·MAX_ITERATIONS2의 별도 확인용 PPO/AMP가 엔진 종료 조건으로3epoch 완료하고 checkpoint 저장(`output/open_push_late_rsi_train_check/OpenPushLateRSI_07-20-21-32/nn/OpenPushLateRSI.pth`),2환경/30step headless 평가 복원 통과(`/tmp/open_push_late_rsi_{train_check,eval_check}.log`). 실제 TensorBoard에 과제별XY/조작 progress·Actor/Critic edge bias·RSI counts 생성/유한값 확인. 장기 학습·정책 성공·후반 자세의 지속 접촉/5초 지지·GUI/VNC는 미검증이며 본학습은 실행하지 않았다.

### OPEN/PUSH XY 접근·실제 edge bias 로깅

- 성공률만으로 접근과 조작을 구분하기 어려워 OPEN/PUSH별 root→손잡이/박스 중심 XY거리(m)·기존 접근 식의0~1점수·조작 progress를 rollout 표본 가중 평균으로 추가했다. semantic-only edge fusion에 Actor/Critic별 실제 유효 task attention bias의 부호 평균·RMS·표본 수를 추가했다(기존100 forward 진단 주기·최대32표본; OPEN/PUSH/AT 및 기존5관계). config/structure에 태그와 정의를 기록했다. 보상·관측·checkpoint 계약은 유지하며 이미 실행 중인 PID3576291에는 자동 반영되지 않는다. 종료/재시작·학습은 실행하지 않았다.
- CPU30개 통과(`test_open_push.py`, `test_stage1_unified.py`): 활성/비활성 관계 마스킹·부호 평균/RMS·gradient 분리·XY 접근과 박스 목표 progress 분리 및 Actor/Critic4과제 조합 순열/gradient 회귀 포함. GPU 시뮬레이션·실제 장기 학습 로그 생성은 미검증이다.

### Door Hold / 공동 OPEN의90도·선 자세 성공 조건

- 사용자 요청으로 단독 `door_open_hold_amp`와 공동 `open_push_unified_amp`의 문 성공 문턱을90도(π/2, 각도 허용오차0)로 변경했다. asset 고정 선 자세 대비 골반 높이70% 이상·torso 수직축 기울기40도 이내·최소 한 발의 바닥 근처 지지(foot body 높이≤18cm, 위쪽 힘≥1N)를 동시에 만족해야5초 유지 시간을 누적한다. 발 접촉 충격은 각 physics tick에서 수집한다. 자세 무효 시 유지 시간·현재 성공과 손 상호작용/열기 진행 보상을 초기화/차단하고 접근 보상은 남긴다. 양손 허용·몸통 접촉 판정·통과/추가 넘어짐 종료·PUSH/AMP 설정은 유지했다. 자세 설정은 기존 doorHold checkpoint 계약에 포함되므로 이전 계약으로 직접 resume/test할 수 없다. config/structure 문서를 갱신했다.
- CPU53개 통과(`/tmp/door_posture_cpu.log`):90도 미만·저골반·torso 기울기·발 지지 없음/뜬 발 힘·제어 초반 접촉 충격·연속5초/초기화 및 공동 PUSH 보상 독립성 포함. GPU6 비학습2환경 공동4과제 조합·reset/AMP/network 검사 통과(`output/open_push_posture_check_gpu6/audit.json`). 단독 물리 회귀 검사 통과(`output/door_hold_posture_check_gpu6/physics_audit.json`):문 최대97.38도·손 지지 약9.93초로 기존 각도/접촉만이면 성공하는 자세가 발 지지 부족으로 새 유지 시간 최대0.27초에 그쳐 거부되고 놓으면 닫힘을 확인했다. 강화 조건에서 정상 선 자세의 실제5초 물리 유지·학습된 정책 성공은 미검증이며 CPU 정상 유지 조건만 통과했다. 초기 GPU binding 호출 오류는 actor-local body handle 사용으로 수정했다. compileall·diff 확인; 학습/기존 프로세스 변경은 하지 않았다.

### OPEN/PUSH 전체 토큰 순서 섞기

- 사용자 요청으로 공동 sampler에 `shuffle_token_order: true`를 추가하고 boolean 검증·checkpoint 계약을 연결했다. 기존 네트워크 순열 경로로 actor/critic 매 forward에서H2/O4/G2 전체8토큰·관계 bias·GTA pose를 함께 섞고 사람 출력 순서를 복원한다. 과제/AMP33·67·RSI 및 물체/goal binding은 유지했다. config에 현재 동작과 미적용 RSI 추천안을 구분해 기록했다.
- CPU28개 통과(`/tmp/open_push_shuffle_cpu.log`):4가지 과제 조합의 actor/critic 출력·gradient 및 자동randperm 확인 포함. GPU6 비학습2환경 reset/partial reset·AMP128×1310·actor4×32/critic4×1·4가지 조합 시뮬레이션 통과(`output/open_push_shuffle_check_gpu6/audit.json`, `/tmp/open_push_shuffle_gpu6.log`). 학습/RSI 변경은 실행하지 않았다.

### OPEN/PUSH AMP 보행33%·조작67% 정렬

- 사용자 요청으로 공동 `open_push_unified_amp`의 OPEN/PUSH 전문가를 모두 보행33%·각 조작67%로 맞췄다. `ampExpertMapping.OPEN=[0.33,0.67,0]`, `PUSH=[0.33,0,0.67]`, 명목 `skillDiscProb=[0.33,0.335,0.335]`다. 두 밀기 clip은 동일 weight로각33.5%다. 과제50/50·loco100% Start 초기화·모델/보상 및 단독 Door 실험은 유지했다. 현재 config 문서를 갱신했다.
- 관련 CPU16개 통과(`/tmp/open_push_amp33_cpu.log`). 새 비율·명목 분포·초기화/task 유지 및 기존 계약과 expert mapping 차이를 확인했다. 기존 확인용 checkpoint는 새 비율의 직접 resume/test 계약과 달라 새 scratch 학습이 필요하다. 이번 비율 수정에서 GPU 시뮬레이션/학습은 실행하지 않았다.

### OPEN/PUSH 50/50 공동 학습

- 사용자 요청으로 별도 PUSH 단독 실험 대신 기존 Unified처럼 사람2명 각각 OPEN/PUSH를50/50 샘플링하는 `HumanoidMAOpenPush`를 추가했다. 문2·박스2의 혼합 actor, 현재 Door Hold의 열기/5초 지지, 손 밀기 이력·바닥/기울기/안정성·고정 목표1.5m 기반 PUSH를 한 정책으로 학습한다. 과제 graph 및2-family 조건 AMP/expert/replay 매칭·별도 physics/reference checkpoint 계약을 연결했다. 기존3-family AMP 기본값과 Door/Unified 실험은 보존한다.
- 선택한 BONES `push_obstacle_180_101__A341`/`push_obstacle_180_003__A342`만 archive에서 추출해367/352frame·30FPS·15body/32DOF로 변환했다. 기존 보행4개·문1개와 새밀기2개 manifest 및 공동 env/train·train/test/VNC를 추가했다. OPEN 문100%·PUSH 보행30/밀기70%, reset은 loco100%·Start다. 원본/shared 데이터는 수정하지 않았다. config/structure에 실행법을 기록했다.
- CPU34개 통과. 관절 제한 위반0·최대DOF속도6.95rad/s 미만·endpoint fit 최대1.55cm·발mesh최소5mm 확인. GPU6 비학습2환경에서30회 task reset·4가지 조합·partial reset 보존·AMP128×1310·actor4×32/critic4×1/discriminator 유한값 및 무접촉 박스 안정성 통과(`output/open_push_check_gpu6/audit.json`). GPU6·2048환경·MAX_ITERATIONS2 확인용 PPO/AMP 실행·checkpoint 저장 및2환경/30step headless 복원 완료. 엔진 종료 조건으로실제3epoch 진행; 정책 성공/장기 학습·GUI/VNC는 미검증이다.
- 초기 비학습 검증은GPU5에서 완료했으나 사용자GPU6 전용 지시 후 GPU6에서 다시 확인했다. 새 wrapper는GPU6만 허용한다. 확인용 output은 `output/open_push_unified_amp_check`, 본학습 기본은 `output/open_push_unified_amp`이며 장기 본학습은 실행하지 않았다.

### Door Hold 시작 거리1.5~2.5m 무작위화

- 사용자 요청으로 고정2.5m 대신 reset마다 env/사람별 손잡이 앞 거리를1.5~2.5m 균등 샘플링한다. 학습/평가 모두 적용하며 참조 body/root 위치를 함께 이동시킨다. 부분 reset은 다른 환경을 보존하고 거리 범위를 checkpoint 계약에 포함한다. 문 전문가100%·loco Start RSI·모델 및 유지 조건은 유지했다. config/structure 문서를 갱신했다.
- CPU28개·GPU6 비학습2환경 검증 통과.5회 reset10샘플에서1.509~2.415m 및 범위 준수·거리 변화·부분 reset 보존 확인. AMP/actor/critic·5초 지지·release/자동 닫힘 검사 통과, 학습은 실행하지 않았다. 결과 `output/door_hold_start15_25_check/physics_audit.json`.

### Door Hold 문 전문가100%·시작 거리2.5m

- 사용자 요청으로 `door_open_hold_amp`의 `skillDiscProb`·`ampExpertMapping.OPEN`을 `[0,1]`로 바꾸고 손잡이 앞 시작 거리를1.5→2.5m로 늘렸다. RSI는 loco100%·Start를 유지한다. 관측/모델/유지 조건 및 다른 Door 실험은 변경하지 않았다. 현재 config 문서를 갱신했다.
- 관련 CPU28개·GPU6 비학습2환경 검증 통과: 실제 초기 손잡이 수평 거리2.5m, 전문가/초기화 비율·AMP discriminator·actor/critic 유한값·5초 접촉 지지·release/자동 닫힘 확인. 검증 결과 `output/door_hold_door100_start25_check/physics_audit.json`. 접근·지지를 지정한 자세 물리 검사이며2.5m 접근 정책의 성공은 미검증; 학습 실행 없음.

### 1인 Door Open/Hold 과제

- agent2 운반을 제외하고 사람1·문1의 `HumanoidMADoorHold`, 별도 env/train YAML·train/test/VNC를 추가했다. BONES 왼손+loco full AMP를 유지하고 문을1.15rad 이상 열어 손으로5초 지지할 때 현재 성공으로 판정한다. 손 접촉 반력/문 개방 방향 힘을 physics tick마다 확인하며0.1초 공백만 허용한다. 손 이탈/장기 공백/각도 하락은 타이머·현재 성공을 초기화한다. O38/총288 관측에 타이머·접촉·공백 시간을 포함하고1인 graph/encoder 타입을 연결했다.
- 전용 문 asset은 문짝 하단4cm 틈·경첩 마찰0을 적용하여 닫힘 토크가 바닥/정적 마찰에 막히지 않게 했다. 환경 spring(k2,c0.5,max4Nm)만 경첩에 작용하며 정책의 문 actuator는 없다. 기존2인 문 설정·asset은 보존한다. 별도 output/checkpoint 계약으로 기존 정책과 분리한다.
- GPU6·CPU 각 비학습2환경에서 실제 손으로 밀기·5초 지지·손 release 시 성공 해제·10초 후 닫힘 통과. CPU 실제 손-문 collision pair204step 확인. AMP64×1290→discriminator64×1 유한값. 지정 자세로 수행한 물리 검증이며 본학습·정책의 자연스러운 지속 지지는 미검증; 장시간 유지 전문가를 새로 추가하지 않았다. 결과 `output/door_hold_check_{cpu,gpu}/physics_audit.json`.
- CPU51개·GPU6 actor2×32/critic2×1 입출력 유한값·wrapper 문법·diff 검사 통과. 기존 BONES 비교 테스트는 이후 추가된 분리 실험 경첩 설정만 명시적으로 예외 처리하고 다른 설정 동일성 검사를 유지한다. 검증 프로세스 정상 종료, 학습 실행 없음.

### 절충 경첩 적용 및 로컬 Codex 학습 인계

- 사용자 승인한 damping2.0/friction0.1을 분리/공유 loco full-AMP 두 설정에 반영했다. 이전5.0/0.5와 output/checkpoint가 섞이지 않도록 wrapper 기본 경로를 `_balanced_hinge`로 분리했다. CPU55개·GPU6 비학습2환경 물리 검증 통과: 실제 설정·충격 후 멈춤·무접촉 안정성·prescribed humanoid 접촉 개방·관측/보상 대응 확인. 결과 `output/door_balanced_hinge_check_20261007/physics_audit.json`.
- `markdowns/door_local_training.md`에 작업 트리/untracked·데이터 심링크 이식·로컬 준비·검증·2048환경 scratch 학습/viewer 명령과 전체 Codex 프롬프트를 추가했다. 현재 독립2agent/2door OPEN과 이후 공동 OPEN+CARRY 통과 과제의 범위를 구분했다. 서버 학습은 실행하지 않았다.

### Push Door 경첩 마찰·감쇠 강화

- 순간 충격으로 문이 관성 개방되는 현상을 줄이기 위해 현재 분리/공유 approach full-AMP 설정에 선택적 경첩 damping5.0/friction0.5를 적용했다(기존0.5/0.05). 수동 revolute joint·stiffness0을 유지하고 공용 URDF·pull·BONES 설정은 보존했다. 경첩 물리를 checkpoint 계약에 포함하고 train/test/VNC 기본 output을 `_resistant_hinge`로 분리했다.
- CPU55개 통과. GPU6·비학습2환경에서20초 무접촉 drift<3e-12rad·설정값·관측/보상 대응 통과. 초기0.1rad에서0.25/2.64667/3rad/s 충격 후 최대0.1015/0.2566/0.2954rad에서 멈춰 성공각1.15rad 미달; prescribed humanoid 접촉에서는1.69999rad까지 개방했다. 결과 `output/door_hinge_resistance_check_20261007/physics_audit.json`.
- 기존 학습 종료/재시작 및 새 학습은 실행하지 않았다. 기존 PID2705174는 확인 시 이미 부재. 변경은 새 프로세스에 적용되며 학습 성능·자연스러운 손 사용은 미검증; angle-only 성공 조건은 유지했다.

### BONES 왼손·왼쪽 개방 motion 변환 및 OPEN AMP 연결

- 선택한 Uniform 반전 BVH를 읽는 parser와 source 위치·anatomical frame 기반 constrained IK 변환기를 추가했다. cm/Y-up·절대 Hips translation을 m/Z-up으로 변환하고 humanoid bone length·32 DOF 제한·손 고정·발 mesh grounding을 적용했다. crop0–3.4초·30FPS·103frame·15body의 `SkeletonMotion .npy`와 수치 보고서·현재 문 치수를 겹친 CPU HTML/GIF를 생성했다. 원본/shared 데이터와 문 asset은 보존했다.
- 새 `dataset_door_bones_left.yaml`·`door_open_bones_left_amp` env/train·train/test/VNC를 추가했다. 기존 loco-only reset·reward·전 구간 AMP를 유지하고 OPEN expert를 loco30%/doorOpen70%로 연결했다. full push AMP의 검증 조건만 다중 전문가를 허용하도록 확장했으며 gated/pull의 기존 제한은 유지한다. 학습은 실행하지 않았다.
- CPU40개 통과. GPU6·2평가환경에서 실제 loader·32DOF·AMP64×1290 및 discriminator64×1 유한값·두 expert pool·checkpoint 계약·52frame 비학습 재생 통과. 무접촉 Gym/Poselib FK 차이최대2.2mm, 문과 겹친 replay 차이최대9.5cm. 관절 제한 위반0·최대 DOF 속도6.05rad/s·발 mesh 바닥5–21mm 확인. 개방 비교구간 손–손잡이 거리중앙값7.6cm로 접촉 정확성은 미확정이다.
- GPU6 offscreen graphics 초기화가 segfault로 실패하여 GPU0으로 전환하지 않고 CPU 미리보기를 제공했다. source에는 문 궤적/접촉이 없으며 추정 door angle은 시각 비교에만 사용하고 AMP에는 넣지 않는다. 보고서 `output/door_bones_left_amp_check/REPORT.md`; 실제 정책 문열기·통과 성공은 미검증이다.

### BONES-SEED Uniform 다운로드·Door 후보 원본 검증

- 사용자 승인으로 `bones_seed/soma_uniform.tar.gz` 45,188,088,297bytes를 다운로드했다. 공식 LFS SHA256 `0c1fd91628d2644d0d7c0b735ae64faa5fa79834280bc1ce7860a871e55f912c`와 일치한다. 선정 후보184개만 추출(373,134,682bytes); 전체 archive를 풀지 않았다.
- 184개 모두 metadata frame 수·공통77-joint skeleton·240 channels·유한값 검사 통과. BVH Frame Time0.008333초, 정적 Root wrapper를 제외한 Hips translation은Y-up/cm이다. 공식 viewer에서 먼저 받은11개는 archive 파일과 SHA256이 모두 일치한다.
- 우선11개 root/FK 분석·원본 skeleton HTML 재생·그림·수치표는 `bones_seed/validation/REPORT.md`에 정리했다. TYPE1 Top5의 root 순이동0.98~2.27m 확인; TYPE2 검토 후보는 개방 이후0.10~0.44m 후퇴하나 측면 이동이 작다. 문 평면·접촉·doorway clearance는 미확정이다. 기존 코드/YAML·AMP 연결·retarget·학습·평가는 변경/실행하지 않았다.

### Door 전 구간 loco AMP의 EdgeEncoder 공유/분리 명령 분리

- 동일 설정 반복run1/run2 대신 사용자 요청한 Actor/Critic EdgeEncoder 공유/분리 비교를 준비했다. 새 `door_open_shared_edge_approach_amp` env/train YAML·train/test/VNC를 추가했다. 기존 unified 분리형과 env는 동일하고 train 차이는 공유 플래그와name뿐이다. 몸/손0.2·전 구간AMP·loco RSI/reset/PPO를 맞추고 output/checkpoint를 분리했다.
- CPU50개 통과: 두 새 설정의 동일 조건 및 실제 encoder 객체 공유·Actor/Critic gradient 확인 포함. 공유형 GPU16평가환경 reset1step에서 quaternion 유사도≥0.99903·거리1.47~1.50m·유한 관측 확인. wrapper 문법·diff 검사 통과. 로그는 `/tmp/door_shared_full_amp_check.log`. 본학습은 실행하지 않았다.

### Door 전 구간 loco AMP — 거리 taper·조작 latch 비활성화

- 사용자 요청으로 현재 미는 `door_open_unified_approach_amp`에 `doorApproachAMP.gate_enabled: false`를 적용했다. 거리·손접촉·문각도에 따른 감소/차단 없이 AMP 가중치1을 유지한다. 몸/손0.2·task/disc0.5/0.5·reset/PPO는 유지하고, 당김 등 필드가 없는 설정의 기존 gate는 보존한다.
- 기본 train/test/VNC output을 `door_open_unified_approach_amp_holding_scale_full_amp`로 분리했다. gate 설정이 checkpoint 계약에 포함되어 이전 gated checkpoint의 직접 resume/test는 거부된다. 학습은 실행하지 않았다.
- CPU47개·wrapper 문법·diff 검사 통과. GPU2환경 비학습 무접촉/손접촉·관측/보상 대조 및 근접/개방 상태의AMP1 assertion 통과. 결과는 `output/door_full_amp_check_20261007/`, 로그는 `/tmp/door_full_amp_check_20261007.log`. 새 설정 학습성능은 미검증이다.

### Door HOLDING 접근 계수 정렬·두 학습 종료·접근 정체 진단

- 사용자 요청으로 GPU6의 미는 Door `reset_v2_run1/run2` 두 학습만 SIGTERM 종료·프로세스 부재를 확인했다. 마지막 번호 checkpoint는 각epoch10500이며 원본은 보존했다. 새 본학습은 실행하지 않았다.
- 현재 push approach-only의 몸/손 계수를 각각0.1→0.2(state weight0.4)로 변경했다. progress/success0.2·sharing·AMP/PPO/reset 및 다른 실험은 유지했다. 기본 output을 `door_open_unified_approach_amp_holding_scale`로 분리하고 기존 reward checkpoint의 직접 resume/test 거부를 검증했다.
- CPU46개·계수변경 후 GPU2환경 무접촉/손접촉·관측/보상 대조 통과. 각checkpoint를16평가환경 deterministic/stochastic599step 비학습 진단했다. reset 관측 좌표 오차0·문 위치에 대한action 반응·finite·DOF padding 확인. 평균 최근접 거리약1.2m·조작거리0.6m 도달0·문열기0이며, 접근 시 AMP 감소가 접근 이득보다 큰 설계 충돌을 확인했다. 단일 원인 확정·계수수정 후 학습성능은 미검증이다. 보고서와 로그는 `output/door_approach_debug_20261007/`.

## 2026-10-06

### 미는 Door 두 실행 epoch3000 resume

- 사용자 요청으로 미는 문 `reset_v2_run1/run2`를 GPU6·각2048환경·seed42에서 재개했다. 기존 프로세스 부재/GPU 여유 확인 후 각 최신 epoch3000/frame393216000 checkpoint의 가중치 유한값·Door/reward/AMP 계약 및 optimizer 존재를 확인했다. 당김·grasp 변경은 적용하지 않았다.
- 기존 restore로 모델/통계·optimizer·epoch/frame을 복원하고 원본 checkpoint 보존·같은 output의 새 시간 디렉터리로 기록한다. CPU 연산 thread pool을1로 제한했다. 두 실행의 checkpoint 로드·GPU6 배치·TensorBoard epoch3008/frame394133504 진행 확인; CPU 사용은 확인 시 각약105%였다.
- 실행/PID/로그는 각 `latest_launch.json`, launch/검증 결과는 `output/door_push_resume_20261006/`에 저장했다. 재개 명령은 config.md에 기록했고 diff 검사 통과. resume 후 장기 성능은 미확인이다.


### Door 당김 방향 분리 및 CPU/GPU 물리 검증

- 기존 밀기 설정은 유지하고 `door_pull_unified_approach_amp` 환경/train YAML·train/test/VNC·당김 asset을 추가했다. 경첩 축−Z·손잡이 offset−0.25m, reset FK 부호·spawn 위치·당김 checkpoint 계약을 적용한다. DOF 양수 개방량·보상식은 유지한다. 학습은 실행·종료·재시작하지 않았다.
- 축 반전만 적용한 첫 접촉 검사는 실패했다. 손이 걸릴 공간 확보 및 느린 스크립트 손 궤적 후 CPU/GPU 각각 약69°·사람 방향 Y변위−0.568m 확인. CPU 실제 손–손잡이 pair 확인; 무접촉20초·다른 문 유지·해제4초 hinge 범위/finite·관측/전체 reward 대조 통과. 모든 검증은 한 코어·1스레드로 제한 후 종료했다.
- 관련 CPU45개·shell/Python/YAML 문법 검사 통과. 결과는 `output/door_pull_physics_audit_20261006/REPORT.md`. sphere 손의 접촉 검사이며 실제 grasp/인장 부착·정책 성공은 미검증이다.


### Door 실제 손 접촉·무접촉 안정성·관측/보상 추가 검증

- 비학습 2환경 CPU/GPU 검증 스크립트·wrapper를 추가했다. CPU affinity 한 코어·PyTorch/PhysX/BLAS 각1스레드로 제한했고 검증 후 종료했다. 기존 본학습과 환경/reward 설정은 변경하지 않았다.
- 사람이 떨어진 상태20초에서 문 각도 변화3e-12rad 미만, 스크립트로 사람을 움직인 접촉에서 CPU/GPU 모두97.4° 개방·나머지 문 유지 확인. CPU 실제 손–문 contact pair 확인, GPU에서는 pair API 대신 손 contact force와 passive hinge 반응을 확인했다.
- 물리 DOF/손잡이/panel과 관측, 성공 경계, 독립 계산한 전체 reward·공유분·패널티 대조 모두 통과. 손잡이 오차1e-6m 미만. 결과는 `output/door_physics_audit_20261006/REPORT.md`에 보존했다. 학습 정책 성능·grasp/잠금장치 검증과 구분한다.


### Door GPU 전체·부분 reset 검증 후 두 본학습 재시작

- 사용자 종료/재시작 요청 시 기존 두 학습은 이미 종료된 상태임을 실제 GPU/프로세스로 확인했다. Door reset의 중복 DOF setter·reset 직후 refresh를 제거하고 GPU index tensor를 유지한다. approach-only 관측에서 환경 격자 오프셋을 제거해 actor/body의 env-local 좌표와 일치시켰다.
- 무작위 loco RSI 시점은 일부 reset 직후 물리 불안정성을 재현했다. 자기 충돌 OFF 비교도 해결하지 못해 충돌 설정은 유지했다. 네 clip의 시작 프레임을 쓰는 `stateInit: Start`로 변경했으며 GPU2048환경×10회 참조 유지 검사 모두 통과(최소 root quaternion 유사도0.99903). 보상·AMP·PPO·환경 수·seed는 유지했다.
- `reset_revision: 3`/`state_init: Start` checkpoint 계약으로 이전 모델 혼용을 거부한다. 새 좌표/회전·GPU setter 회귀2개 포함 CPU43개 통과. GPU2048환경·10회 PPO에서 reset 후 첫-step42132개 검사 통과·epoch11 checkpoint 저장·가중치 유한값 및 16환경60step headless 평가/JSON 정상 종료를 확인했다. 짧은 평가로 문열기 성능은 판단하지 않는다.
- GPU6/tokenhsi/seed42·각2048환경의 두 scratch 본학습을 `output/door_open_unified_approach_amp_reset_v2_run1/run2`로 분리 실행했다. launch/PID/로그는 각 `latest_launch.json`에, 확인용 스크립트·로그는 `output/door_reset_v2_validation_20261006/`에 보존한다. 본학습 두 프로세스의 GPU6 배치와 epoch4/5 진행을 확인했으며 approach term은 약0.055였다(수정 전 약0.00042). 장기 성능은 아직 미확인이다.

### Door approach RSI의 환경 좌표·시작 방향 수정

- 다중 환경에서 사람에만 arena origin이 더해져 Door와 다른 좌표계로 reset되는 추가 오류를 확인했다(CPU2환경의 두 번째 root–handle 거리9.94~10.92m). 단일환경 viewer에서는 드러나지 않았다.
- approach-only Door RSI를 env-local own spawn XY에 재배치하고 +Y로 향하게 root/body 위치·회전·속도를 함께 변환했다. clip 누적 이동도 제거하며 참조 motion ID/time·관절·AMP history는 유지한다. viewer에 root–handle 거리 진단을 추가했다.
- GPU16평가환경·3회 reset/1step에서 모든 root–handle 거리1.44~1.50m·quaternion 유사도≥0.99725·유한 관측 확인. 관련 CPU41개 및 diff 검사 통과. 기존 실행 중 프로세스에는 적용되지 않는다. 수정 후 본학습은 실행하지 않았다.

### Door reset 수정 후 동일 두 실행 재학습 명령

- 기존 두 실행은 동일 approach-only AMP·seed42 설정임을 명확히 하고 scratch 재학습 output을 `door_open_unified_approach_amp_reset_fixed_run1/run2`로 분리한 명령을 안내했다. GPU6 학습 프로세스 부재·wrapper 인자와 경로를 확인했다. 학습은 실행하지 않았다.

### Door reset viewer GUI 실행 보완

- 로컬 viewer에 DISPLAY 검사와 기존 `gui_gpu_env.sh`의 CUDA/Vulkan GPU 일치 설정을 추가했다. DISPLAY 없는 서버 터미널에는 VNC 명령을 안내하고 segfault 전 종료한다.
- tokenhsi/GPU6/PORT6086에서 실제 VNC viewer 생성·렌더링·reset2회 정상 종료 확인(자세 유사도 최소0.99994). shell 문법·diff 확인. 사용자 로컬 DISPLAY 환경에서의 성공 여부는 미확인이다.

### Door reset 확인용 비학습 뷰어

- 사용자 직접 확인용 checkpoint 없는 reset viewer·로컬/VNC wrapper를 추가했다. 참조 자세 PD로1물리 step 후3초 화면 유지·재초기화하며 root quaternion 유지·높이·문 각도를 출력한다. 정책 성능/걷기 검증과 구분한다.
- tokenhsi/GPU6·1평가환경5회 headless reset 통과: quaternion 유사도 최소0.99994·골반 높이0.895~0.943m·닫힌 문 유지. shell 문법·diff 확인. 실제 GUI/VNC 화면은 미검증이다.

### Door GPU reset 상태 덮어쓰기 수정

- 사람/문 indexed root·DOF reset을 한 번에 적용하도록 Door 경로를 수정했다. GPU2환경에서 수정 전 RSI root 방향이 첫 step에 identity 부근으로 돌아가고 수정 후 참조 방향을 유지하는 것을 확인했다. CPU reset만으로는 드러나지 않았다.
- 관련 CPU 회귀41개 및 `git diff --check` 통과. 기존5500 checkpoint는 수정 전후2환경 평가 모두 성공0·조기종료100%였다. 이전 학습 current-success 추이를 정상 문열기 성능으로 해석하지 않는다. 진단은 `output/door_pose_audit_20261006/{before,after}.log`; 기존 학습은 종료·재시작하지 않았다. 재학습 성능·viewer는 미검증이다.

### Door 기존 unified·접근만 locomotion AMP

- 사용자 선택에 따라 `door_open_unified_approach_amp`와 train/test/VNC를 추가했다. 기존 unified/MAAgent/PPO·Door reward를 유지하며 기존 걷기/서기4clip만 AMP·loco RSI에 사용한다. 실제 Door motion이나 box interaction proxy는 사용하지 않는다.
- own handle root XY 거리1.2→0.6m에서 AMP 보상만 taper하고 root≤0.6m/hand≤0.25m/angle≥0.05rad에서 Human별 reset까지0으로 고정한다. rollout별 gate를 보존해 보상을 계산하며 task0.5/disc0.5·discriminator 학습 경로를 유지했다. 시작 거리1.5m·전환 진단·checkpoint gate 계약을 명시했다.
- CPU Door 테스트41개 통과(신규7개), 2환경8step CPU PhysX·유한 관측570/AMP1290·loco demo 생성을 확인했다. GPU 학습은 실행하지 않았다. 문 열기 학습 성능·GPU pipeline·viewer는 미검증이다.


### Door 기존 unified / shared-edge 비교·과제 관련 proxy AMP/RSI

- 사용자 최종 선택에 따라 `door_open_unified_amp_on`(Actor/Critic별 semantic EdgeEncoder)과 `door_open_shared_edge_amp_on`(공유)을 별도 config·train/test/VNC·output으로 추가했다. 두 env는 동일하고 train 차이는 공유 플래그뿐이다. 기존 unified 네트워크와 PPO를 유지하며 ksh 원본/이식본은 수정하지 않았다.
- 실제 Door dataset이 없어 기존 loco/pickUp/carryWith를 대체 참조로 명시했다. 새 manifest·`utils/door_reference.py`에서 서 있는 자세·handle 높이·앞방향 손 reach·저속 root·전체 AMP history를 선별한다. AMP20/70/10%, RSI30/60/10%와 clip별 뒤쪽30% 후보70% 요청을 두 실험에 동일 적용한다. 자신의 닫힌 문 handle 앞에 Human pose/velocity를 일관되게 정렬하며 기존 box RSI·reward는 유지한다.
- CPU 회귀94개 통과(새 비교/reference12개 포함). 두 설정 각각2평가환경에서 interaction RSI·8step CPU PhysX·hinge 회전·handle 위치·유한 관측570/AMP1290 및 기존 AMP demo 생성을 확인했다. 실제 유효 history 후보는 pickUp59/carryWith22개이며 network 공유/분리와 양쪽 gradient를 검사했다. Python/shell 문법·config 경로·diff 검사를 수행했다. GPU 학습·GUI/VNC·장기 성능은 미검증이다.
- 새 checkpoint에 공유 여부·reference/expert/RSI 계약과 motion manifest 해시를 보존하고 다른 계약의 resume/test를 거부한다. 기존 Door ON/OFF는 유지하며 TRUMANS에 의존하지 않는다. GPU6 본학습 명령만 제공하고 실행하지 않았다.

### Stage 1 shared-edge 기반 독립 Door Open·AMP ON/OFF

- `HumanoidMADoor`·repo 소유 revolute Door URDF·OPEN=11 고정 H0→Door0/H1→Door1 graph·별도 `edge_door_reward.py`를 추가했다. 기존 semantic packet/shared EdgeEncoder/Actor·Critic Transformer/MAAgent PPO를 재사용하고 Object30에 hinge angle·velocity·local handle XYZ만 추가했다. passive Door DOF는 사람 제어·force penalty와 분리한다.
- approach·hand interaction·opening progress·current success를 분리하고 기존 edge 최대0.6·자기0.9/동료0.1 공유·power/collision penalty convention을 유지한다. term TensorBoard와 OPEN 진단·기존 graph 평가 JSON을 연결했다. 두 config는 AMP 활성 플래그만 다르며 OFF의 discriminator/demo/replay 처리는 생략한다. ON은 기존 MotionLib·reference mixture와 명시적 OPEN expert row를 사용한다. 평가 reset skill과 AMP expert 목록은 Door에서 분리했다.
- 전용 ON/OFF train/test/VNC·CPU 물리 check와 실행/체크포인트 문서를 추가했다. 기존 task reward/RSI 및 TRUMANS repo·dataset은 변경하지 않았다. Door schema11과 AMP 모드로 학습 resume 계약을 구분한다.
- CPU 회귀 82개(신규 Door 22개 포함) 통과. ON/OFF 각 2평가환경·8step CPU PhysX 초기화/부분 reset·관측570/AMP1290 유한값과 기존 demo 생성 확인; OFF에서 초기 angular velocity로 hinge 실제 회전 및 handle 위치 일치도 확인했다. Python import/문법·6개 shell 문법·문서 경로·diff 검사를 수행했다. GPU 학습은 실행하지 않았고 GPU pipeline·GUI/VNC·장기 성공률은 미검증이다.

## 2026-10-05

### ksh size RSI·typed bias·typed message 독립 이식

- 사용자 요청에 따라 세 방식을 `ports/ksh_stage1`의 독립 스냅샷과 전용 train/test/VNC로 추가했다. env/learning/utils/config·PD 캐시를 분리하고 대용량 데이터는 기존 읽기 전용 링크를 사용한다. 기존 hybrid 및 ksh 원본 코드·설정·학습 결과는 수정하지 않았다. 출처/해시와 실행법은 `UPSTREAM.json`, `markdowns/ksh_stage1_port.md`에 기록했다.
- CPU 50개·새 shell 11개 문법·diff 검사 통과. GPU6/tokenhsi에서 세 방식 각각 2048환경·MAX_ITERATIONS=1로 정상 학습/epoch2 저장, 각 16환경·32step headless 평가/JSON 저장을 확인했다. 가중치 유한값·typed bias/message shape와 비영 norm을 확인했다. 검증은 `output/ksh_*_check`, `output/ksh_*_eval_check` 및 `output/ksh_port_validation_20261005`에 분리했다.
- 기존 파일 및 ksh 원본 해시 비교 통과, 사용자 학습 PID1678859 유지. 장기 본학습은 시작하지 않았고 GUI/VNC는 shell/경로만 확인했다. 짧은 검증 결과로 장기 성공률 개선은 판단하지 않는다.

## 2026-10-03

### 중단된 GPU 6 hybrid 학습 재개

- 사용자 요청으로 중단된 hybrid 본학습을 최신 1000 epoch checkpoint에서 재개했다. 이전 로그의 마지막 epoch는 1049였고 CUDA illegal memory access로 종료되어 저장 이후 49 epoch는 다시 진행한다. 재개 전 GPU 6 MPS 정상 응답과 중복 학습 부재를 확인했으며 다른 실행은 유지했다.
- 동일 seed42·2048환경·tokenhsi에서 optimizer/정규화·hybrid 후보 관측 창을 포함해 복원했다. GPU 6 배치·후보 캐시 일치·TensorBoard 1001~1003 epoch 진행을 확인했다. 재개 checkpoint·이전 run·새 PID/로그/run 경로는 `output/approach_scenario_stage1_hybrid_late_rsi/latest_launch.json`에 보존했다.

### 5개 task의 고정 XY·크기/위치 Z별 학생 평가

- 사용자 요청으로 평가 전용 `evaluate_z_sweep.sh/.py`와 집계 `utils/z_sweep.py`를 추가했다. 박스 XY40cm·Z20~65cm, 밑면 높이0~50cm를 분리하고 ON_TOP source/받침·AT goal 높이를 별도로 비교한다. 기존 충돌 받침을 사용하며 canonical binding·고정 배치는 해당 평가 프로세스에만 적용한다. 실제 asset 크기 override는 `--eval`에서만 허용한다.
- CPU 20개(Z 집계·기존 progress/성공 조건 회귀)·Python/shell 문법·diff 검사 통과. GPU6의 26환경/30step smoke 뒤 학생11000 checkpoint로 1632평가환경·600step·4회 완료: 102조건×128agent=13,056기록, 중복/누락 없음·초기 성공0·실제 초기 source 밑면 높이 오차<1e-6m. 기존 학습은 유지했다.
- HOLDING의 Z20cm 종료 성공0%, 25~65cm는90.6~100%; ON_TOP source40→65cm는75→19.5%; AT 바닥 목표 종료0%, goal 밑면20cm는49.2%. SIT 최대1.6%, CLIMB0%는 loco 시작 평가 결과다. 질량도 Z에 따라 변하며 장기 학습/RSI 시작 성능으로 해석하지 않는다.
- 결과·그래프·전체 표는 `output/student_rsi_z_sweep_11000_xy40_seed42/REPORT.md`, 조건·재실행은 `markdowns/z_sweep.md`. 원시 trial에서 agent 성공과 scene 전체 성공을 분리했다.

### GPU 6 학생 학습을 hybrid 본학습으로 교체

- 사용자 요청에 따라 GPU 6의 기존 student_rsi 실행을 종료하고 hybrid_late_rsi를 seed42·2048환경·2 agents·4 objects로 scratch 시작했다. 요청의 PID 오기는 실제 실행 명령/GPU를 대조해 기존 학생 실행으로 식별했다. 다른 학습은 유지했다.
- MAX_ITERATIONS/RESUME_CHECKPOINT/OUTPUT_PATH 잔여값을 제거하고 tokenhsi 환경·분리된 백그라운드 세션으로 실행했다. GPU 6 배치·물리 캐시 재사용·TensorBoard epoch 1~6 진행을 확인했다. 실행 PID/로그/run 경로는 `output/approach_scenario_stage1_hybrid_late_rsi/latest_launch.json`에 기록했다.

### 같은 XYZ를 유지하는 후반 hybrid RSI 구현

- 별도 hybrid_late_rsi 설정·train/test/VNC를 추가했다. carry 계열은 실제 크기의 PD 검사와 현재 학습 정책 첫 2 step 관측을 결합하며 80~95→70~80→60~70→50~60% 순으로 선택한다. 전 구간 실패 시 크기·task를 유지한 loco 시작을 사용하고 성공률로 크기를 재추첨하지 않는다. SIT/CLIMB은 기존 학생 후보를 유지한다.
- ON_TOP putDown은 source/받침 XYZ별로 검사하고 RSI 혼합에 10% 추가했다. task 분포·보상·progress·shared edge·조건부 AMP는 학생 설정을 유지한다. 1000 epoch 관측 창 갱신, 후보 상태 checkpoint 보존, 500 epoch task별 XYZ episode/후보 진단을 추가했다.
- CPU 39개·Python/shell 문법·diff 검사 통과. GPU 6·2048환경 짧은 학습/checkpoint 저장 및 cache hit/resume를 확인했고 후보 관측 3202→5079 보존·누적과 모델 유한값을 확인했다. 기본 tokenhsi 환경에서도 GPU 6 resume 정상 종료. 16환경·30step headless 평가 정상 종료; GUI/VNC는 shell/경로 확인만 했다.
- 기존 학습 PID는 유지했고 새 장기 학습은 시작하지 않았다. 수렴 속도 개선은 장기 비교 전까지 미확인이다. 실행·후보 조건·진단은 markdowns/hybrid_late_rsi.md.

## 2026-10-02

### 학생 RSI 학습 task 샘플링 비율 조정

- 사용자 요청으로 student_rsi YAML의 agent별 HOLDING/SIT/CLIMB/HOLDING_ON_TOP/HOLDING_AT 비율을 5/5/20/35/35%에서 10/10/20/30/30%로 변경했다. RSI skill 혼합은 유지한다.
- CPU에서 YAML 값·합계·sampler 계약 및 seed42의 2048 scene/4096 agent task 생성을 확인했다. 학습은 실행하거나 기존 프로세스를 재시작하지 않았다.
- 실행 시 unified 검증의 고정 비율 검사에 막히는 누락을 수정했다. student_candidates 모드는 YAML 분포의 sampler 계약을 검사하고 기존 실험의 고정 비율 검사는 유지한다. 실제 student_rsi YAML의 unified 검증 통과·잘못된 확률 거부를 포함해 CPU 회귀 23개 통과. GPU 학습은 실행하지 않았다.

### 학생 RSI 끝부분 우선 요청을 50%로 조정

- 사용자 요청에 따라 새 student_rsi YAML의 late_fraction을 0.8에서 0.5로 낮췄다. pickUp/carryWith/putDown의 90~98% 우선 요청 50%, 검증된 65~98% 전체 pool 선택 50%다. 후보 부족 fallback 및 sit/climb 설정은 유지한다. 실제 끝부분 선택률은 전체 pool 구성과 fallback에 따라 다르다.
- YAML 값과 referenceRsi 계약을 CPU로 확인했다. 새 학습을 실행하거나 기존 실행을 재시작하지 않았다.

### 학생 반복 검사 후보의 크기별 학습 RSI sampler 구현

- 새 student_candidates 모드로 같은 실제 크기·skill의 원본 motion ID/phase를 선택한다. n≥3·모든 초기 Δv≤2 및 carry 손 거리 증가≤20cm 조건을 적용하고 원본 clip 가중치/학생 품질 가중치를 유지한다. 초기 CPU 표 준비 뒤 GPU 일괄 샘플링으로 reset의 크기별 Python 루프를 피한다.
- carry/pickUp/putDown은 80% 확률로 90~98% pool을 요청하고 부족하면 같은 크기의 덜 늦은 후보를 우선 사용한다. sit/climb은 추가 끝부분 편향 없이 후보 pool을 사용한다. pool이 전혀 없으면 같은 크기·skill의 원래 허용 RSI를 사용하고 누적 fallback을 기록한다. 크기/skill을 조용히 바꾸지 않는다.
- 별도 XYZ20~65cm student_rsi YAML 및 GPU6/7 제한 train/headless-test wrapper를 추가했다. 새 progress/task AMP도 활성화한다. 사람/박스 참조 초기화와 박스 속도 복원을 재사용하고 학생 검사에 없던 1cm 전신 screen은 추가하지 않는다.
- CPU 42개 회귀 통과, 실제 후보 fingerprint 및 1000개 크기를 CPU stub으로 확인했다. 기존 실행은 유지하며 GPU 실환경 검증과 장기 학습은 시작하지 않았다. 구현/명령/미검증 범위는 markdowns/student_rsi_sampling.md.

### 확장 학생 RSI 256회 검사 완료·반복 저충격 후보 정리

- 동일 student16500/seed1, XYZ20~65cm, 2048환경×256회×20step 완료를 확인했다. NaN 0, 부분 reset 보존 통과. 203,375후보 중 166,480관측/36,895미관측, n≥3·모든 초기 Δv≤2 후보 62,949개를 검토용 표로 보존했다.
- 크기·구간·축별 matched agent 통계와 coverage를 결과 옆 ANALYSIS.md/analysis_*.csv에 저장했다. 최대65cm carryWith에도 낮은 충격 후보가 있으나 CLIMB는 낮은 Δv에서도 root 하강이 많아 성공 유지 검증과 구분한다. 정상 pickUp 상승/putDown release를 실패로 자동 판정하지 않는다.
- 기존 학습·RSI sampler·progress 설정은 변경하지 않았다. 기존 RSI의 동일 학생 대조군 및 수렴 비교 없이 개선율을 주장하지 않는다. 적용 방향은 markdowns/reference_rsi_expanded_candidates.md에 기록했다.

### 새 size_progress 학습 설정의 task AMP 활성화

- ksh PID 1099002를 읽기 전용으로 참고해 task family 기준 AMP 경로를 확인했다. 우리 현재 full AMP 학습은 이미 활성 상태였지만 새 size_progress YAML은 비활성이어서 기존 conditioned 경로를 켰다. carry/sit/climb 전문가 분포와 명목 skillDiscProb를 원본 unified에 맞춘다.
- RSI 초기 skill과 무관하게 graph task 라벨을 붙이고 실제 같은 family의 expert/replay를 선택하는 기존 코드를 재사용한다. 부분 reset·AMP 정규화 라벨 보존도 확인했다. carryWith 전용 전문가 비중은 원본과 같이 0이다.
- CPU 30개 테스트 통과. 기존 학습·RSI 검사·ksh 코드는 변경하지 않았고 새 GPU 학습은 시작하지 않았다. 적용 상태·분포·계약은 `markdowns/task_amp.md`에 기록했다.

### XYZ 20~65cm RSI 검사 완료·원본 시점 예비 후보 재추출

- 262,144 scenes/128회/20step distill 검사 정상 완료, NaN 0 및 부분 reset 보존을 확인했다. 행동·후반 구간·축·정확한 1000크기별 CPU 집계를 원시 결과 옆 ANALYSIS.md/analysis_*.csv에 보존했다. agent 첫 두 step Δv와 scene 20step 전체 Δv를 구분한다.
- 원본 참조 motion ID/phase의 Δv≤2 예비 후보 203,375개(preferred 137,008/conditional 66,367)를 새 파일로 추출했다. 최대 크기 carryWith도 시점별 충격 차이가 크지만 학생 안전/수렴 개선을 확정하지 않는다. 새 범위는 일부 teacher 학습 범위 밖이다.
- 기존 학생 보상/RSI를 유지한 확장 box.build 전용 재검사 설정과 동일 16500 snapshot 재검사 명령을 준비했다. 설정의 변경 범위와 reward 계약 보존을 CPU로 확인했다. 실행 명령과 반영 방향은 `markdowns/reference_rsi_expanded_candidates.md`에 기록했다.
- 기존 원시 결과·학습·RSI sampler는 변경하지 않았다. 확장 후보 학생 재검사는 다음 단계이며 기존 학생 검사의 작은 범위 config로 실행하면 새 크기를 검사하지 못한다.

### 실제 박스 크기·회전을 반영하는 progress 추가

- SIT·CLIMB·ON_TOP progress를 기존 성공 사각형까지의 XY 거리로 계산하는 `success_region` 옵션을 추가했다. HOLDING은 박스 외곽까지의 거리와 접근 여유 0.3m, AT은 기존 goal 허용 거리 0.5m를 사용한다. 높이·발·파지 등 성공 조건과 phi는 유지한다.
- 별도 size_progress YAML 및 GPU 6/7 제한 headless train/test wrapper를 추가했다. 기반 설정과의 차이는 progress뿐이며 RSI·크기 범위·실행 중인 학습은 변경하지 않았다. 기존 보상 설정 checkpoint와의 계약 분리를 유지한다.
- CPU 25개 테스트 통과 및 shell 문법 확인. GPU 학습/수렴 비교는 수행하지 않았다. 계산·적용 명령·checkpoint 제약은 `markdowns/size_progress.md`에 기록했다.

### XYZ 0.20~0.65m 전용 RSI 충격 검사 시작

- 사용자 최종 요청에 따라 1000개 크기·후반 65~98%·5종 참조 RSI의 distill probe를 GPU 6에서 분리 실행했다. 2048환경·128회·20step, measurement_version=2다. 기존 학습은 유지했다.
- wrapper에 RSI_PROBE_CFG 입력을 추가하고 전용 snapshot에서 XYZ scaleRange=[0.5,1.625], maxTopSurfaceHeight=1.30을 설정했다. 기본 학습 YAML은 수정하지 않는다. 입력과 로그는 ../runs/reference_rsi/impact_xyz20_65_seed0/, 조건/해석은 markdowns/reference_rsi_expanded_probe.md에 기록했다.
- shell 문법과 1000개 조합/최대 스택 설정을 확인했다. 첫 background 실행은 유지되지 않아 새 session으로 분리 시작했다. 완료/전체 분석은 아직 미확인이며 새 크기 일부는 teacher 학습 범위 밖이다.

### 학생 RSI 후보 재검사 v2 완료 분석

- 사용자 실행의 measurement_version=2, 2048환경×64회×20step 완료를 확인했다. 동일 학생 snapshot/seed로 후보 42,555개를 관측하고 7,250개는 미측정. 3회 이상 관측·모든 첫2step Δv≤2m/s 후보는 16,294개다. NaN 0, 부분 reset 보존 통과.
- 수정된 손 거리 결과와 정확한 원본 크기/시점별 후보를 분석했다. 50×50cm carryWith 90–98%의 Δv>2m/s는 0.89%, 손 거리 증가>20cm는 0%였지만 65×65cm는 각각 37.23/27.57%였다(Z 합산). 거리 threshold는 진단 예시이며 실제 파지 성공 판정이 아니다.
- 최신 ANALYSIS.md/analysis_summary.json과 문서 링크를 저장했다. 회전/수직 이동의 정상 동작과 실패를 아직 분리하지 않았으며 학습용 후보표나 학습 설정은 변경하지 않았다. 기존 학생 RSI 대조군/수렴 가속은 미검증이다.

### 학생 RSI 후보 64회 재검사 분석 · 손 거리 기준 정정

- 사용자 실행 2048환경×64회×20step 완료를 확인했다. 49,805후보 중 42,538개를 관측했고 7,267개는 미측정. 3회 이상 관측된 20,011개 중 첫2step Δv가 모두 2m/s 이하인 후보는 16,120개다. 동일 학생의 원래 RSI 대조군이 없어 성능/수렴 개선율은 확정하지 않는다.
- 손 거리 증가의 초기 기준을 reset 직후 Gym rigid-body tensor에서 읽어 이전 자세가 남을 수 있는 문제를 확인했다. 이번 실행의 해당 지표는 파지 판정에서 제외하며 원시 결과를 보존했다. 이후 측정은 초기 참조 FK를 사용하고 measurement_version=2로 표시한다. 수정은 문법 확인했으며 version2 GPU 검사는 아직 수행하지 않았다.
- 원시 결과 옆 ANALYSIS.md/analysis_summary.json과 RSI 문서 링크에 결과·측정 제한을 기록했다. 실행 중인 학습과 학습 설정은 변경하지 않았다.

### 충격 기반 RSI 예비 후보 추출 · 현재 학생 재검사 구현

- `utils/impact_candidates.py`로 기존 impact_sweep의 실제 크기/skill/원본 motion ID·phase를 추출한다. Δv≤1m/s preferred, 1~2m/s conditional의 49,805개 예비 후보를 생성했다. 원본 데이터 fingerprint와 출력 덮어쓰기 방지를 적용하고 teacher rollout 상태나 학습용 안전 판정은 만들지 않는다.
- `probe_student_candidates.sh`와 probe의 `candidate_replay`를 추가했다. 학생은 기존 checkpoint 계약 검증/strict model load 후 실제 크기에 맞는 후보를 재생한다. 후보가 없는 조합은 같은 크기·skill의 원래 RSI로 검사하고 candidate_id=null로 구분한다. 수직 Δv/상승량·회전·손–박스 중심 거리 증가·사람 root 높이 감소와 후보별 관측 수를 저장한다. 학습 sampler는 변경하지 않는다.
- CPU 24개 통과. 작은 GPU 실행의 minibatch/AMP 제약을 보정한 뒤 GPU 6의 128환경·20step 학생 검사 정상 완료: 124후보 일치, NaN 0, 부분 reset 보존 통과. 학생에서 일치 후보 16개가 첫2step Δv 2m/s를 넘어 teacher shortlist만으로 안전을 확정할 수 없음을 확인했다. 실패 실행도 보존했다.
- 구현/명령/제한은 `markdowns/reference_rsi_candidate_commands.md`. 1184210은 유지했고 장기 학습·전체 후보 검증은 실행하지 않았다.

### RSI 축별 충격 분석 · 후속 세션 문서 연결

- `markdowns/reference_rsi_axis_analysis.md`에 carryWith/pickUp의 Y 경향, sit/climb의 Z 경향, 정확한 크기를 고정한 비교와 원시 통계 경로를 기록했다. 집계는 단일 teacher/seed의 accepted 초기 상태 관측이며 인과·안전·학습 한계 판정이 아님을 명시했다.
- RSI 아카이브와 구조 문서에서 축별 분석 및 PID 1184210 점검으로 연결했다. 링크·원시 수치·문서 형식을 확인했으며 학습 코드·설정·프로세스는 변경하지 않았다.

### 박스 크기 유지 · 후반 RSI 학습 가속 논의 아카이브

- 사용자 목표를 박스 범위를 최대한 유지하면서 쓸 만한 후반 RSI를 확보해 학습을 빠르게 하는 것으로 명시하고 `markdowns/reference_rsi_archive.md`에 구현·기하 검사·distill 물리 검사·전체 크기별 표·해석 수정과 남은 작업을 정리했다.
- 1cm 관통 기준과 Δv 3m/s·속력 5m/s는 진단 기준이며 학습 불가능 판정이 아님을 기록했다. 최대 크기/후반 구간별 충격, 소표본·teacher 제어의 한계, Δv 1~2m/s 허용안이 미검증 제안임을 구분했다. 수렴 가속은 아직 측정하지 않았다.
- 기존 RSI 가이드와 구조 문서에 아카이브 링크를 연결했다. 문서·결과 링크와 주요 통계를 검증했으며 학습 코드/설정·실행 프로세스는 변경하지 않았다.

### 별도 실행의 AMP GPU 메모리 계측

- 실행 중인 학습을 유지하고 `../runs/memory_audit_289953/`의 별도 profiler로 GPU 6·2048환경에서 AMP 129/132 각 6회, 132 41회와 본학습 동일 시드 6회를 확인했다. 모두 정상 종료했다.
- 6회 비교에서 실제 텐서는 약 +189MiB, PyTorch 예약량은 +1328MiB였다. 조건부 AMP family 매칭의 임시 할당과 캐시 반환을 확인했으나 본학습의 26084MiB 전체는 재현하지 못했다. 상세 수치·한계는 REPORT.md에 기록했다.
- 계측/empty_cache는 별도 프로세스에만 적용했다. 1184210·1099002는 유지했고 학습 코드에는 추가 변경하지 않았다.

## 2026-10-01

### Shared random binding 조건부 AMP 7500회 resume

- 사용자 요청으로 기존 학습 289953만 종료하고 원본 unified의 7500회 모델·AMP 통계 전체를 별도 pth로 이식했다. 대상 reward/graph 메타데이터를 유지하고 optimizer는 초기화했다. 1099002와 ksh 원본은 변경하지 않았다.
- `ampConditionedResume: true`를 명시한 random-binding 설정에 한해 132차원 조건부 AMP를 허용한다. 기본 129차원 실험 계약은 유지한다. CPU에서 새 설정/기본 설정 통과와 opt-in 없는 조건부 설정 거부를 확인했다.
- 실행 산출물과 GPU 6 런처는 `../runs/manual_resume_289953/`. 2048환경으로 7500회 pth 로드 및 GPU 6 UUID 배정을 확인했다. 총 40000회 종료 설정이며 학습 수렴은 아직 검증하지 않았다.

### 크기·행동·시점별 초기 충격 probe 옵션

- `probe_reference_rsi.sh impact_sweep`을 추가했다. 기존 크기 범위를 유지하고 상자 관련 5종의 원본 참조 시점을 지정 phase 구간에서 검사한다. 검사에서만 RSI 제외 구간을 허용하며 1cm 후보표 필터/후보 부족 loco 대체를 사용하지 않는다. 기존 장면 배치 검사는 유지한다.
- carry 박스 참조 속도를 복원하고 distill 정책으로 실행해 담당 박스별 첫 2step Δv·최대 속도/각속도·참조 궤적 이탈과 모션 ID/phase를 저장한다. 크기·skill·10% phase 구간별 표본 수와 분위수를 CSV로 출력한다. 정책 영향이 포함된 랜덤 표본 검사이며 학습/안전 후보표 생성은 하지 않는다.
- Python·셸 문법과 diff 검증을 통과했다. 새 옵션의 GPU 실행은 사용자가 수행할 예정이며 물리 검증 완료로 주장하지 않는다. 기존 학습 프로세스는 유지했다.

### 랜덤 크기 참조 RSI · 시간 후반/행동 단계 및 distill 물리 probe

- PID 289953의 shared random binding 기준으로 별도 `approach_reference_rsi_{time_late,semantic_late}` config·train/test/VNC를 추가했다. 기존 보상·AMP·과제·skill 비율은 유지하고 CLIMB 70% 선호 기준만 시간 65~98% 또는 발의 상판 접근으로 나눈다. 학생 학습은 scratch이며 distill pth는 probe의 action 계산에만 사용한다.
- CPU 크기별 참조 모션 후보표, 원본 제외 구간/CLIMB 후반 override, 같은 크기의 명시적 loco fallback 및 계수, 참조 박스 선/각속도, 재추첨 중 고정 물체 배정, 모든 사람–박스의 구체/캡슐/발 mesh 정점 관통 screen, RSI/후보표 SHA256 resume 계약을 추가했다. 기존 옵션 없는 실행 경로는 유지한다.
- CPU 관련 20개 통과. GPU 6의 2048환경·2회 reset·10step distill probe에서 >5m/s 장면이 기준 175/4096, 시간 후반 2/4096, 행동 단계 7/4096이었다. 최종 두 조건의 NaN/조기 종료 0, 부분 reset의 비대상 상태/AMP 이력 유지 통과. 원본 teacher는 3물체/rescue 메타데이터이며 4물체 target 모델 weight strict load를 probe에서만 확인했다. 일반 checkpoint 검사는 우회하지 않는다.
- 두 scratch 학습을 각각 `MAX_ITERATIONS=1`, 2048환경으로 실행해 epoch 2 checkpoint 저장을 확인했다. 각각 scalar 384개가 모두 finite이고 teacher/distill 필드가 없었다. 두 checkpoint의 일반 시작 64환경·32step·3회 평가도 정상 종료했다(짧은 학생 성공률 0, 로드/실행 검증). 박스 속도 급증의 완전 제거와 수렴 개선은 검증하지 않았다. carryWith의 안전 후보가 14/252 크기에만 남아 loco 대체로 초기 skill 분포가 크게 바뀌므로 예비 속도 감소를 그대로 성능 개선으로 해석하지 않는다.
- 산출물은 `../runs/reference_rsi/`, 상세 실행법·제한은 `markdowns/reference_rsi.md`. 기존 학습·ksh·baseline·외부 소스는 변경/종료하지 않았다. 장기 학습·GUI/VNC는 실행하지 않았다.

### Shared 무작위 binding · 사용자 12옵션 조합

- 선택 `2 2 2 1 2 1 2 1 2 1 2 2`를 별도 config·train/test/VNC·output·checkpoint variant로 구현했다. Shared semantic EdgeEncoder·PRE/TERM 없음·토큰 순열·4물체·5/5/20/35/35·owner loco 1~2m·identity goal·graph 속도 penalty를 유지한다.
- 논리 물체/goal binding을 reset별 disjoint 무작위 배정하고 34번 상자 크기(XY .4~.65m/Z .25~.55m)와 near 시작(80→30%, 30만 per-env step)을 적용했다. AMP는 34/35번 비조건부 혼합·1290차원이며 RSI는 shared 기준이다. 기존 unified config는 canonical·조건부 AMP 계약을 유지한다.
- 관련 CPU **50개 통과**, 새 shell 문법·diff 확인. GPU 4의 학습 없는 **2048환경·8-step** 실행에서 관측644/AMP1290·actor/critic edge 공유·전체 논리 slot 사용·부분 reset·finite 관측/보상·물리 reset 실패0 확인. 결과: `output/approach_scenario_stage1_unified_shared_random_binding_check/environment_check.json`.
- 본학습은 실행하지 않았다. 사수님 원본·기존 두 학습 프로세스는 변경하지 않았다. 실행 명령과 12옵션을 config.md에 기록했다. 장기 수렴은 미검증이다.

## 2026-09-30

### AT goal 마커의 edge 담당 색·슬롯 표시

- Viewer/VNC의 AT goal 점을 고정 빨강 대신 현재 graph의 AT edge owner 색으로 표시한다. marker 표시 여부도 owner 번호가 아닌 실제 AT destination goal 슬롯을 따른다. 레거시 marker는 agent 슬롯 색을 사용한다.
- `config.md` 상단의 학습·로컬 평가·서버 VNC 전체 명령 목록에 unified semantic/shared edge/owner-HOLDING 세 실험을 기재했다.
- Goal 슬롯이 owner 번호와 바뀐 graph를 포함한 관련 CPU 테스트 **6개 통과**, Python 문법·diff 확인. GPU 5의 1환경·16step AT 서버 VNC 평가가 종료되고 결과 JSON을 저장했다. 실행 중이던 학습은 재시작하지 않았다.

### Unified semantic actor·critic EdgeEncoder 공유 실험

- 기존 unified semantic의 과제·관측·보상은 유지하고, actor·critic의 semantic EdgeEncoder 임베딩·MLP·bias projection만 공유하는 별도 config·train/test/VNC·output·checkpoint variant를 추가했다. owner-HOLDING 경로는 변경하지 않았다.
- CPU unified 테스트 **7개 통과**: 두 encoder의 모듈 동일성·모델 파라미터 열거 시 중복 없음·양쪽 gradient·기존 checkpoint 계약 분리를 확인했다. 셸 문법·diff 검사도 통과했다.
- GPU 5에서 2048환경·`MAX_ITERATIONS=1` 별도 `_check` 학습과 checkpoint 저장을 완료했다. TensorBoard scalar **177개/354값 모두 finite**, 저장 weight의 actor/critic edge tensor 8개가 동일했다. 저장 checkpoint로 1환경·32step headless 평가도 완료했다. 이는 실행 경로 확인이며 장기 수렴 검증은 아니다.

### Unified 독립 5과제 · semantic/owner-HOLDING 두 실험

- 두 agent가 HOLDING/SIT/CLIMB/HOLDING+AT/HOLDING+ON_TOP을 5/5/20/35/35로 독립 샘플링하는 4물체 config·train/test/VNC·별도 output을 추가했다. 두 실험은 edge의 담당 HOLDING φ 입력 유무만 다르다. 기존 실험의 설정·실행 중 학습은 유지했다.
- Hᵢ/Oᵢ/Gᵢ 논리 할당과 agent별 별도 ON_TOP 받침을 고정하고, 타입 인코딩 뒤 토큰·GTA·bias를 함께 순열화한 다음 human readout 전에 복원한다. 새 실험에서만 goal quaternion을 identity로, 상자를 0.40×0.30×Z(0.30~0.50m)로, loco 담당 상자를 owner 주변 1~2m로 설정했다. placement 가까운 시작은 0이다.
- 원본 unified 기준 RSI와 carry/sit/climb 조건부 단일 AMP를 연결했다. AMP 10프레임에 과제 one-hot을 각각 추가하고 전문가·리플레이 family를 rollout과 행별 매칭한다. RSI history·부분 reset·정규화에서도 라벨을 유지한다. CLIMB 후반 강제 RSI는 끄고 ON_TOP putDown은 사용하지 않는다. 보상식·TERM 미사용은 유지한다.
- CPU 전체 **139개 통과**. 각각 2048환경·`MAX_ITERATIONS=2`(기존 runner 기준 저장 epoch 3) scratch 학습·저장 완료, scalar **177개 모두 finite**, 물리 reset 실패·공유 담당 물체 **0**. Owner actor/critic 상태 분기 weight 갱신을 확인했다. 토큰 순열의 action/value·gradient 동등성과 AMP family 매칭을 테스트했다.
- 각 checkpoint의 128환경 training-RSI 검사(전체 reset 4회·부분 reset·64step)에서 goal 회전·크기·loco 1~2m·graph/RSI·관측 packet·AMP 이력/정규화 연결을 확인했다. 전용 test wrapper의 64환경·32step headless 평가도 두 실험 모두 완료했다. 검증 자료는 `output/unified_validation/`에 보관했다. 이는 실행 검증이며 장기 수렴이나 모든 자세의 무충돌을 보장하는 결과는 아니다.

### Scenario graph 기준 상자 속도 패널티 수정

- 랜덤 object binding과 무관한 기존 `_agent_box_assignment`로 속도 패널티를 계산하던 오류를 수정했다. 현재 위치·pre-step history·부분 reset history 모두 graph의 담당 HOLDING/SIT/CLIMB 물체를 동일하게 조회한다. 기존 비-scenario 경로는 기존 할당을 유지한다.
- 소유자/물리 slot/edge 순서 변경·부분 reset·기존 경로 회귀를 포함한 관련 CPU 테스트 **19개 통과**. 저장 owner checkpoint로 128환경·64step 시뮬레이션에서 실제 패널티와 graph 기준 재계산값의 오차 **0**을 확인했다. 현재 실행 중인 학습은 재시작하지 않았다.
- 추가 분석은 `output/placement_code_audit/`에 보관했다. 기본 loco source 배치는 과거 owner 주변 1~2m와 현재 arena 반경 4.5m로 다르며, GTA goal 회전은 여전히 같은 번호 human heading을 사용한다. 상자 크기·spawn·goal 회전은 이번에 변경하지 않았다.

## 2026-09-29

### Paired placement · 담당자 HOLDING 상태 edge 입력

- 기존 4물체 paired placement의 과제·보상·RSI·AMP·가까운 시작 없음 설정을 유지한 별도 config·train/test/VNC·output을 추가했다. 새 관측 패킷은 edge당 `valid/src/dst/relation/owner/owner_holding_state` 6개 값이며, AT/ON_TOP에는 동일 owner·source의 선행 HOLDING `φ`, 다른 유효 edge에는 1을 넣는다. actor·critic의 기존 semantic 64차원 뒤에서 placement edge에만 상태 잔차를 더하고 checkpoint 계약을 분리했다.
- 관련 CPU 테스트 **32개 통과**, Python·셸 문법과 diff 확인. GPU 0의 기존 학습은 유지한 채 2048환경·1 iteration 별도 output 학습과 checkpoint 저장을 완료했다. TensorBoard scalar **137개 모두 finite**, AT+AT/ON_TOP+ON_TOP 샘플 비율 **0.501/0.499**였고 actor·critic 상태 분기 weight가 모두 0에서 갱신됐다. 저장 checkpoint로 각 1환경·32-step headless AT/ON_TOP 평가를 완료했다. 이는 실행 경로 검증이며 장기 수렴 결과는 아니다.

### 같은 과제 쌍 · 가까운 시작 없는 Stage 1 비교

- 기존 3물체 sampler에서 두 agent 모두 AT만 받는 config와, 새 4물체 sampler에서 scene별 AT+AT/ON_TOP+ON_TOP을 50/50으로 뽑는 config를 분리했다. 두 실험의 가까운 시작 확률은 0이며 34번의 RSI·AMP·보상식은 유지한다. 4물체 ON_TOP은 서로 다른 source 2개와 support 2개를 사용한다. 각 config에 train/test/VNC와 별도 output·checkpoint variant를 연결했다.
- 관련 CPU 테스트 **53개 통과**, 스크립트 문법·diff 확인. GPU 0에서 각각 2048환경·1 iteration scratch 학습과 checkpoint 저장을 완료했다. 물리 reset 실패는 둘 다 **0**, paired placement의 AT/ON_TOP 비율은 **0.509/0.491**이었다. 저장 checkpoint로 각각 1환경·32-step headless AT/ON_TOP 평가를 마쳤고, ON_TOP 평가에서 물리 상자 4개가 서로 다른 edge 역할에 연결됨을 확인했다. 장기 정책 학습 결과는 아직 확인하지 않았다.

## 2026-09-28

### 37번 ON_TOP putDown RSI 실험

- 36번을 유지하고 별도 config·train/test/VNC·output·checkpoint variant에서 ON_TOP `putDown` RSI 10%를 추가했다. 원본 후기 모션은 받침 상자와 겹치므로 45~70% 구간에서 시작하며, 받침을 모션 최종 XY에 배치하고 상자·사람 침투 reset을 재추첨한다. AT `putDown`과 36번 과제·보상 분포는 유지한다.
- 관련 CPU 테스트 31개, 셸·Python 문법·diff 검사 통과. GPU 0에서 2048환경·1 iteration 학습과 checkpoint 저장 완료: ON_TOP `putDown` RSI 비율 9.46%, 물리 reset 실패 0·재시도 98, scalar 138개 모두 finite. 저장 checkpoint의 1환경·32-step ON_TOP headless 평가와 JSON 저장도 완료했다. 이는 실행 경로 확인이며 장기 수렴 검증은 아니다.

### 36번 AT/ON_TOP 집중 학습 실험

- 34번의 reward·RSI·가까운 시작을 유지하고 HOLDING/SIT/CLIMB/HOLDING+AT/HOLDING+ON_TOP 샘플 비율만 0/0/0/50/50으로 바꾼 별도 Stage 1 config·train/test/VNC·output을 추가했다. 새 variant로 checkpoint 보상 계약을 분리했다.
- 관련 CPU 테스트 18개, 스크립트 문법·문서 링크·diff 검사 통과. GPU 0에서 2048환경·1 iteration 확인과 checkpoint 저장을 완료했다. AT/ON_TOP 샘플 비율 각각 0.5, 물리 reset 실패 0, scalar 137개 모두 finite였다. 저장 checkpoint의 1환경·32-step headless 평가와 JSON 저장도 완료했다. 이는 실행 경로 검증이며 장기 수렴 검증은 아니다.

### 뷰어 상자 색을 과제 배정에 연결

- Stage 1·2 로컬/VNC 뷰어에서 graph의 실제 대상 상자를 에이전트 색으로 칠한다. 공동 대상은 노란색, 비대상은 회색이다. 물리 상자 순서와 graph 배정이 달라도 색이 맞도록 변경했다.
- Stage 1 ON_TOP, Stage 2 place_climb/place_stack graph를 사용한 색 지정 호출 검증과 Python 문법 확인을 마쳤다. 실제 뷰어 화면 확인은 아직 하지 않았다.

## 2026-09-27

### Stage 2 SIT plane·34번 보상/독립 과제 변형

- 기존 29번과 분리한 `approach_stage2_coordination_sit_plane` config·train/test/VNC·output을 추가했다. SIT도 상판 안쪽 10%와 높이 오차 7cm로 판정하고, 자기 edge 합계·동료 edge 평균에 0.9/0.1 팀 보상을 적용한다. 에피소드는 place_climb/place_sit/place_stack/독립 20/20/40/20이며, 독립 과제 내부는 34번의 10/10/10/35/35와 RSI·가까운 시작 일정을 따른다. 협력 과제의 RSI는 기존 Stage 2 규칙을 유지한다.
- Stage 2 checkpoint 로드에서 reward config 불일치를 검사하도록 수정했다. 34번 epoch 500 checkpoint로 GPU 6에서 2048환경·1 iteration 이식 학습과 1환경 place_sit 평가를 완료했다. 물리 reset 실패 0, scalar 177개 모두 finite, 관찰·AMP 통계 이식 확인. 관련 CPU 테스트 22개 통과. 장기 수렴은 아직 확인하지 않았다.

### output_etc Git 제외

- `.gitignore`에 `/output_etc/`를 추가했다. `git check-ignore`로 내부 파일 제외와 `git status`에서 해당 폴더가 사라진 것을 확인했다.

### VNC 평가 결과 기본 경로 통일

- 현재 실험 9개의 VNC wrapper가 기본 `OUTPUT_PATH`를 각 실험의 `output/<실험명>`으로 전달하도록 바꿨다. 로컬 test와 같은 위치에 `metrics/`, `diagnostics/`가 생기며 학습 checkpoint의 run 디렉터리와 분리된다. 기존 `_vnc` 폴더와 실행 중인 학습은 변경하지 않았다. 셸 문법과 기본 경로를 확인했다.

### 34·35번 자기 보상 합계와 가까운 시작 일정 비교

- 33번의 AT reset 수정·state/progress/성공 식을 유지하고, 34번은 자기 edge 보상 합계·동료 edge 평균과 HOLDING/SIT/CLIMB/HOLDING+AT/HOLDING+ON_TOP 10/10/10/35/35 샘플링을 적용했다. 35번은 34번에서 가까운 시작 80%를 60만 per-env step까지 유지하고 120만 step까지 30%로 감소시킨다. 두 실험은 별도 config·train/test/VNC·output에서 scratch로 시작하며 33번과 기존 학습 프로세스는 변경하지 않았다.
- 관련 Stage 1·2 CPU 테스트 **30개 통과**, 스크립트 문법·diff 확인. GPU 6에서 각각 2048환경·1 iteration 학습과 checkpoint 저장을 완료했다. 두 실험 모두 물리 reset 실패 **0**, 공유 주 물체 **0**, scalar **177개 모두 finite**였고, 샘플 비율은 목표값에 근접했다. 짧은 진단 표본에서 AT 두 번째 목표 slot 높이도 정상이다. 저장 checkpoint의 1환경·32-step AT headless 평가와 JSON 생성도 각각 완료했다. 이는 실행 경로 검증이며 장기 수렴 검증은 아니다.

### 33번 AT 목표 reset 수정

- 독립 graph의 AT가 두 번째 목표 slot을 지정해도 reset이 첫 번째 slot에 목표를 배치하던 선택 오류를 수정했다. 32번의 보상·샘플링·RSI를 유지한 33번 config와 전용 train/test/VNC·output을 추가했다. 공통 reset 수정은 이후 새로 시작하는 27~32번에도 적용되며, 실행 중인 프로세스는 변경하지 않았다.
- 관련 CPU 테스트 **23개 통과**. GPU 6에서 2048환경·1 iteration scratch 학습과 checkpoint 저장 완료. 진단 CSV의 AT 목표 높이는 0보다 컸고, TensorBoard scalar **177개 모두 finite**, 공유 주 물체 **0**이었다. 이는 AT 목표 배치와 실행 경로 검증이며 정책 수렴 검증은 아니다.

## 2026-09-25

### 과거 실험 진입점 정리

- 실행 중인 27·28·30·32번과 Stage 2 29번, 비교용 31번 및 원본 1번을 남겼다. Git에 있던 9~26번 config 18개·전용 실행 스크립트 50개·전용 테스트 11개·오래된 설계 문서 10개·미사용 graph 예제 2개·미사용 학습 smoke config 1개를 제거했다. 현재 문서를 다시 작성하고 25·26번 전용 sampler 분기를 제거했다. 공유 graph·reward 함수는 현재 실험도 사용하므로 유지했다.
- 기존 학습 프로세스와 output은 변경하지 않았다. 삭제된 config·스크립트는 Git 이력에서 확인할 수 있다.
- 남긴 테스트 전체 **109개 통과**, 실행 스크립트 문법·Markdown 로컬 링크·diff 확인. 새 코드로 32번 2048환경·1 iteration 확인에서 checkpoint 저장, 물리 reset 실패 **0**·공유 주 물체 **0**, scalar **177개 모두 finite**였다. 이는 실행 경로 검증이며 장기 수렴 검증은 아니다.

### 32번 보상식 유지 커리큘럼

- 31번의 가까운 AT/ON_TOP 시작·후반 CLIMB RSI·과제/RSI 비율을 유지하고, AT/ON_TOP progress와 CLIMB state만 30번 식으로 되돌린 별도 schema 9 variant와 train/test/VNC를 추가했다. 31번 설정과 checkpoint는 그대로 두며 32번은 scratch로 시작한다.
- 관련 CPU 테스트 **17개 통과**. 2048환경·`MAX_ITERATIONS=1` GPU 1 확인에서 checkpoint 저장, 물리 reset 실패 **0**·재시도 **135회**, scalar **177개 모두 finite**, 템플릿 비율 약 **0.095/0.105/0.298/0.255/0.247**을 확인했다. 저장 checkpoint의 1환경·20-step ON_TOP headless 평가와 JSON 생성도 완료했다. 이는 실행 경로 검증이며 정책 수렴 검증은 아니다.

### 30번 SIT plane 정규화·31번 hard-skill 시작 상태 실험

- 28번을 보존하고 새 schema 9 variant 두 개를 분리했다. 30번은 SIT 성공을 root XY의 상자 윗면 안쪽 10%·목표 Z ±7cm로 바꾸고 agent별 active-edge 보상을 평균내 단일/두 edge 최대 task reward를 0.6으로 맞춘다. 31번은 AT·ON_TOP 가까운 시작 비율을 환경 step 0→300,000에서 80→30%로 낮추고, 장거리 progress·CLIMB root/발 높이 state·hard-skill sampling을 추가한다. 두 실험 모두 scratch이며 팀 공유 0.9/0.1은 유지한다.
- 기존 plane CSV 표본에서 AT/ON_TOP은 에피소드 첫 관측 3.33/4.27m에서 마지막 관측 5.70/6.36m로 목표에서 멀어졌다. CLIMB reference 7개·505프레임을 상자 높이 0.4m·XY 0.5m로 실측한 결과 성공 가능 29프레임 전부 원본 RSI 제외 구간에 있었다. 31번은 CLIMB RSI의 70%를 후반 65~98% 구간에 배정하며 기존 데이터셋과 27·28번 학습을 변경하지 않는다.
- 관련 CPU **34개 통과**, 스크립트 문법·diff 확인. GPU 1에서 두 실험 각각 2048환경·1 iteration scratch와 checkpoint 저장, 저장 checkpoint의 1환경·32-step headless 평가를 완료했다. 30번 물리 리셋 실패 0·재시도 121회, 31번 실패 0·재시도 156회, 각 scalar 177개 finite였다. 31번 CLIMB root/발 조건 동시 통과율은 후반 RSI 이전 짧은 확인 0%에서 1.26%로 바뀌었다. 이는 초기 상태·실행 경로 확인이며 장기 정책 수렴 검증은 아니다.

### 29번 Stage 2 coordination 전이 실험

- 28번 plane 성공 조건과 사용자 지정 전역 팀 보상 0.9/0.1을 유지하고, `place_climb/place_sit/place_stack` 협력 90%와 독립 bundle 10%를 샘플하는 schema 10 config·train/test/VNC를 추가했다. Stage 1 schema 9 semantic `.pth`를 `STAGE1_CHECKPOINT`로 선택해 actor/critic/AMP weight와 actor·AMP 관측 통계를 엄격히 복사하고, actor encoder·관측 RMS를 고정했다. 새 grounded-edge cross-attention과 Stage 1 head의 `[W,0]` 확장으로 협력 입력을 연결했다. Stage 2 resume는 별도 `RESUME_CHECKPOINT`로 분리했다.
- 공유 물체 SIT/CLIMB agent는 loco에서 시작하고, downstream 성공 또는 전체 graph 성공으로 시작한 reset은 재시도한다. 3인 공유 물체·4인 독립 쌍의 explicit 평가 graph와 가변 agent/edge 입력을 추가했다. Stage 2 taxonomy의 CLIMB 누락과 평가 시 학습 graph 동일성 검사를 수정했다.
- CPU 전체 **278개 통과**, 2048환경·1 iteration GPU 학습 및 checkpoint 저장 완료. 저장 checkpoint의 첫 head 협력 입력 weight norm은 **0.0556**, TensorBoard scalar **162개 모두 finite**, 물리 reset 실패 **0**이었다. 실제 Stage 1 plane checkpoint에서 전이 직후 동일 입력의 action mean·sigma 최대 차이는 각각 **0**이었다. 같은 Stage 2 checkpoint로 2·3·4인 1환경·20-step headless 평가와 JSON 저장을 완료했다. 2048환경 Stage 2 resume 뒤 협력 입력 weight norm은 **0.0817**로 변했고 attention weight도 갱신됐으며 actor encoder 75개 tensor와 actor RMS는 동일했다. Resume scalar **174개 모두 finite**, 물리 reset 실패 **0**이었다. 이는 실행·가변 입력 경로 검증이며 협력 성공률·장기 수렴 검증은 아니다.

## 2026-09-24

### 28번 stage1_plane 독립 시나리오 실험

- 27번의 독립 물체·goal 배정, 다섯 행동 각 0.2, RSI·AMP와 팀 보상 0.9/0.1을 유지했다. CLIMB·ON_TOP의 current-success만 원격 `ontop_climb_plane` 방식으로 변경했다. ON_TOP은 source 중심 XY의 support 윗면 안쪽 10% 영역 및 면 높이 오차 1mm, CLIMB은 root XY의 안쪽 5% 영역 및 root 높이 오차 20cm·평균 발 높이 오차 7cm를 요구한다. 연속 state·progress 보상은 기존 중심점 기준이며, 새 reward variant·config·train/test/VNC·output과 checkpoint 계약을 분리했다.
- CPU 전체 **274개 통과**, shell 문법·diff 공백 확인. GPU 3·2048환경·1 iteration scratch에서 다섯 template 비율 **0.194~0.205**, shared primary object **0**, 물리 reset 실패 **0**, TensorBoard scalar **174개 모두 finite**, checkpoint 저장을 확인했다. 저장 checkpoint로 1환경·32-step ON_TOP headless 평가와 JSON 저장도 종료했다. 이는 실행 경로 확인이며 장기 학습 수렴 검증은 아니다.
- GPU 3 본학습 로그 점검에서 CSV의 `self_task_contribution`/`other_task_contribution`이 Stage 1 보상을 자기 100%로 잘못 표시하는 문제를 발견해 config의 0.9/0.1 비율을 따르도록 수정했다. 실제 학습 reward와 TensorBoard 공유 지표는 이미 0.9/0.1로 정상이며, 실행 중 프로세스는 재시작하지 않아 해당 프로세스의 CSV 표시는 이전 방식으로 남는다. 관련 CPU 테스트 **17개 통과**·diff 공백 확인.

### 27번 독립 물체 binding + standalone CLIMB scratch 실험

- 26번의 5-template reward·0.9/0.1 팀 보상·AMP·template RSI를 유지하고, 물체 3개를 reset마다 무작위 순열로 두 agent의 서로 다른 primary object와 남은 ON_TOP support에 배정한다. AT goal도 비공유 무작위 배정하며 동시 ON_TOP은 제외한다. 유효한 template pair의 각 agent 주변 확률은 기존 0.2를 유지한다. 새 sampler/config/train/test/VNC/output과 별도 checkpoint variant를 연결했다.
- CPU 전체 **270개 통과**. GPU 2·2048환경·1 iteration 확인에서 5-template 비율 **0.196~0.203**, shared primary object **0**, 물리 reset 실패 **0**, TensorBoard scalar **172개 모두 finite**, GTA position p95 **4.4012m**, checkpoint 저장을 확인했다. 저장 checkpoint로 1환경·32-step ON_TOP headless 평가와 JSON 저장도 종료했다. 이는 실행 경로 검증이며 장기 수렴·물체 수 확대 추론은 미검증이다.
