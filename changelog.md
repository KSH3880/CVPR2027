# Changelog

최신 변경부터 기록한다. 현재 실행법은 [config.md](markdowns/config.md), 코드 위치는 [structure.md](markdowns/structure.md)를 참조한다. 실행 중인 GPU/PID는 이 파일에 고정하지 않고 실제 프로세스로 확인한다.

## 2026-10-22

### Stage1 PUSH 5500회 정책 VNC

- 사용자 요청으로 `output/stage1_base_push_view_20261022/launch.sh`를 추가하고 5500회 저장본·GPU1 native/MPS 우회·1환경·2명/4상자·loco 시작·1280×720 VNC6107을 실행했다. 모델 계약/복원·실제 정책 step·화면 캡처를 확인했다. tmux는 `koo_stage1_push_view_20261022`, 로그/화면은 같은 폴더 `viewer.log`/`preview.png`다.
- 실행 후 두 학습 PID 생존·GPU1 이벤트 로그 갱신·MPS 로그 크기 불변·viewer MPS 변수 제거를 확인했다. GPU1 여유는 관측 시523MiB이고 학습 처리량은 약9600→7600~8000FPS로 감소했다. 장기 병행 안전성 검증은 아니다.

## 2026-10-21

### Stage1 PUSH_AT RSI 검수 영상

- `output_etc/check/stage1_base_push_rsi_review/stage1_push_rsi_review.mp4` 생성: 앞24초 실제CPU reset 2명/4상자 8장면, 이후 동일bank의 PUSH949후보 전체 영상 재사용. 55.633초·720p·30FPS·1,669프레임이며 사용자 검수 대기다.
- bank/소스 해시 대조·전체 디코딩·대표 프레임 시각 검수·스크립트 문법/diff 확인. GPU 실행·학습설정 변경·새학습 시작 없음. 작업 후 GPU1 종료는 사용자 직접 종료로 확인했으며 재시작하지 않았다. 상세는 같은 폴더 `REPORT.md`.

### PUSH_AT 과제·독립 AMP family 추가 유지

- 사용자 확정대로 정책PUSH_AT ID4와 별도PUSH AMP family3을 유지한다. 직전 AMP3family 복원을 철회하고4family/1330D로 되돌렸다. 기존과제ID0~3·AMP family0~2는 유지한다. 원본과 모든 state_dict 키/shape를 비교해 actor/critic 과제embedding4→5행 및 discriminator 입력1320→1330만 다름을 확인했다. Transformer/TaskActionFusion/GTA의 층/폭은 유지한다.
- CPU66회귀·1000agent reset·부분reset·실제물리24step·모델strict복원 및 MAPlayer/RMS복원·loco24step 재검증 통과. 정책634D/8토큰·AMP1330D다. 기존 GPU학습/물성은 이번 수정에서 변경하지 않았으며 새 GPU/PPO는 실행하지 않았다.

### 원본 Stage 1 네트워크·4상자에 PUSH_AT 추가

- 사용자 정정대로 GitHub `stage1_stage2_base`/`28f02b3`를 `ports/stage1_base_push`에 격리하고 기존SIT/CLIMB/carry_at/carry_ontop을 보존했다. 기존과제0~3 유지·새PUSH_AT=4/샘플링100%, 원본task-shared embedding/Action MLP/GTA·634D/8토큰·2H/4Box·학습2048환경이다. embedding1행·AMP PUSH family1칸을 추가하고 Stage2/coupled는 제외했다.
- 원본AT(각0.2·20cm/1mm·본인100%)와 현재CONTACT(0.4/0.2/0.2·공통면/양손반력)를 결합한다. 현재Y2m/X1~1.5m/Z1.3m·20kg·마찰0.6을 네 상자에 적용하고 추가2개는 실제actor/관측에 포함해 경로 밖에 배치한다. 승인949후보/nativeDOF·loco/PUSH50/50·후반70%를 유지한다. 전용train/test/VNC·출력·checkpoint CPU preflight와 점유GPU 시작 거부를 추가했다.
- CPU66회귀(기존16과제쌍 graph/보상동일성·현재CONTACT함수 AST동일성·2048scene 경로검사), 실제1000agent reset·부분reset·질량/마찰·물리24step·Actor/Critic/AMP·모델strict복원, 실제player 모델/RMS복원·loco24step 통과. 기존6000회/검사fixture 학습resume 거부·mock점유GPU 차단 확인. 기존두학습PID 생존·GPU1로그진행 확인. GPU/PPO/GUI·장기성공 검증 및 새학습 실행은 하지 않았다. 결과는 `output/stage1_base_push_check/REPORT.md`.

### 오래된 체크포인트 용량 정리

- 사용자 요청으로 비활성 실행의 `.pth`를 실행 폴더·모델명별 최대 학습 단계 1개씩 보존하고 343개(할당 용량 16.30GiB)를 삭제했다. 현재 두 학습의 출력 전체와 출력 외 가중치는 보존했으며, fine-tune `source_39000`/`resume_500` 및 PUSH Y2m 재개본/원본5000회 파일은 전후 SHA256 동일성을 확인했다.
- 정리 당시 `.pth` 420→77개, 레포 약73→57GiB. 삭제 목록·보존 목록·결과는 `output_etc/checkpoint_cleanup_20261021/{manifest,result}.json`. 두 학습 PID의 실행·GPU 점유와 최근 TensorBoard 이벤트 갱신을 확인했다. 학습/설정 변경이나 GPU 연산 검사는 하지 않았다.

### PUSH 미는 방향Y2m·5000회 상태 보존 resume 준비

- 사용자 요청으로 `push_native_phase_y2m` env/train/test/VNC·전용bank/output·명시적checkpoint전이를 추가했다. 상자Y2m/X1~1.5m6종/Z1.3m·20kg·마찰0.6·경로간격2.8m이며 기존보상/AMP·정면목표2~5m·949후보·원본관절을 유지한다. 원본5000회 모델/RMS/optimizer168개/epoch/frame/최고보상 tensor동일성 확인, live simulator/replay 미복원이다. 기존실험은 보존하고 새 학습은 실행하지 않았다.
- CPU기하5,694배치 위반0·초기0.1초PD11,388상태 전부통과·대표0.4초고정PD48/144·원본DOF/부분reset·10만sampler·1000agent reset/담당교환·24step물리/AMP/570D정책strict복원·관련CPU70회귀 통과. 재검사때승인bankSHA를보존하도록보완하고새6종전수재검사로확인했다. 기존GPU1client가 있으면 새train이실행전차단되는것도 확인했다. GPU/PPO/장기균형 검증은 하지 않았다.
- `output_etc/check/push_native_phase_y2m/rsi_candidates_y2m.mp4`는949후보전체·1280×720/30FPS/31.63초CPU초기자세열이다. bankSHA/949frame/clip0·3·4·8/Y2m대조·처음/후반프레임시각검수·문법/링크/diff확인통과. 검사보고서는같은폴더`REPORT.md`다. 작업전후두학습PID생존·GPU1로그갱신·MPS로그신규기록없음확인, 기존학습/viewer/MPS를중단하지않았다.

### 기존 PUSH 정책의 미는 방향2m viewer 옵션

- 사용자 요청으로 `TOKENHSI_VIEWER_PUSH_LENGTH_Y=2` 평가전용 옵션을 추가했다. source YAML/bank/checkpoint 계약은 그대로 검사하며 실제 asset·크기 관측/접촉 tensor·장면 clearance에 로컬Y2m를 적용한다. X1~1.5m/Z1.3m·20kg·마찰0.6 유지, 경로간격2.8m다. native GUI/loco/16환경이하로 제한하고 학습·headless·interaction RSI는 거부한다. 실제 viewer 크기는 로그에 표시한다.
- CPU12검사(원본불변·모드/길이거부·2048scene clearance) 통과. 5000저장본의 GPU1/1환경/VNC6105 명령을config에 기록했다. GPU/물리rollout/GUI 실행은 하지 않았으며 변경크기 정책성공·학습병행 안전성은 미검증이다.

## 2026-10-20

### 목표를박스정면2~5m로제한·clip7만제외한949후보검수영상

- 사용자요청으로목표sampler에`box_forward`를추가하고초기박스로컬+Y정면2~5m로제한했다. 월드방향은박스yaw를따르며반대/측면목표를검증에서거부한다. 출발랜덤/상자/보상/AMP는유지한다. 이후사용자가지적한clip7은RSI에실제로포함됨을확인해전체247후보만제외하고clip0/3/4/8·949후보로설정했다. 추가발2cm필터/다른구간제외없음.
- 현재bank는`push_native_phase_xy_forward_2to5m_no_clip7_rsi_bank.json`. 직선1196후보의새CPU86,112상태0.1초PD/원본DOF/부분reset통과결과에서동일949부분집합임을확인하고sourceSHA/상속검증으로기록했다. CPU150회귀(2048scene×3seed로컬+Y/거리/방향거부·949후보동일성·기존radial호환), 현재bank샘플링·계약/문법/diff확인. 과거bank/영상보존, GPU/학습실행없음.
- 검수용949후보만의원본속도31.63초/30FPS/949frame영상은 `output_etc/check/push_native_phase_xy_no_clip7_review/rsi_candidates_no_clip7.mp4`. 이전MD영상의정확한해당frame선택/표시수정·clip7/극단사례없음·현재bankSHA/949frame대조를확인했다. 후속사용자가영상검수를승인했다. 승인메타데이터만갱신하고949후보동일성을재확인했으며명령만제공하고학습은실행하지않았다. 보고서는같은폴더`REPORT.md`다.

### PUSH 목표2~5m·사용자 확인 MD RSI 복원

- 사용자 확정대로 `push_native_phase_xy`의목표최솟값3→2m(최대5m/방향랜덤)을적용하고, 사용자가영상으로확인한MD조건clip0/3/4/7/8·1,196후보를복원했다. 추가발2cm/clip7전체제외는적용하지 않는다. 새`push_native_phase_xy_2to5m_rsi_bank.json`은원본후보/순서/구간과동일하고목표최솟값/새bank경로외계약동일성을확인했다. 원본3~5m/51후보bank보존·출발/상자/보상/AMP유지·학습미실행.
- CPU74회귀(2048scene 목표2~5m/경로·출발1~2m·후보동일성 포함), 36크기/양담당자86,112상태0.1초PD·원본DOF/부분reset·10만sampler·계약/문법/링크/diff 확인통과. 확인기는추가발지지필터를설정한경우에만힘지지를요구하고, 초기PD실패시요청pool을조용히축소하지않고실패하도록수정했다. 검사출력은새2to5m폴더로분리했다. 결과는`output_etc/check/push_native_phase_xy_2to5m/REPORT.md`. GPU/학습/PPO실행없음.

### MD 조건 RSI 5클립·1,196후보만 영상

- 사용자 요청으로 추가접지필터/clip7 제외 전의보존bank를 사용한 `output_etc/check/push_native_phase_xy/md_rsi_candidates_only.mp4`를 생성했다. clip0/3/4/7/8의1,196후보 전부·원본속도30FPS/39.87초·1280×720, 추가극단사례/검수정지화면 없음. 정측면/실제collider·발OBJ·불투명깊이 렌더다. 원본bankSHA·클립별출력합·ffprobe1,196frame·문법/diff확인 통과. 현재학습config/bank 변경 및 GPU실행 없음.

### 실제 RSI 후보만 별도 영상

- 사용자 요청으로 현재bank의clip0/8·51개후보 전부만 포함한 `output_etc/check/push_native_phase_xy_grounded/rsi_candidates_only.mp4`를 생성했다. 추가극단사례 없이0.25배속6.8초·1280×720/15FPS/102출력frame·bankSHA 대조를 확인했다. 후보/학습설정 변경 및 GPU실행 없음.

### RSI clip8 영상 겹침 재검수·정측면/불투명 깊이 렌더

- 사용자 지적clip8/4.700초는 원본 발mesh 우17.18cm/좌1.85cm·머리collider 간격32.25cm이며 CPU기하검사상 상자관통 없음이었다. 이전영상은 투명상자와 뼈대에 깊이가림을 적용하지 않아 투영겹침/발높이를 판별하기 어려웠다. RSI후보/원본관절/학습설정은 유지하고 실제MJCF sphere/capsule·발OBJ·불투명상자의CPU triangle깊이 렌더와 정측면으로 수정했다. 이전영상은 보존한다.
- 현재영상은 `output_etc/check/push_native_phase_xy_grounded/rsi_preview_v2.mp4`, 지적프레임은`reported_frame_v2.jpg`, 원본시각/발간격은`rechecked_frames.json`. 실제물리rollout/정책안정성 검증이 아니며 GPU실행 없음. 깊이순서/primitive범위 및 기존mesh지지 회귀5개·문법/영상메타/링크/diff확인 통과. 0.4초 감사의 절대골반높이0.65m 기준 미달을 실제넘어짐으로 단정하지 않도록 현재config/보고서도 정정했다.

### PUSH 초기 발 지지 조건 누락 수정·clip7 RSI 제외

- 사용자 영상 지적 프레임clip7/5.133초의 실제 발mesh 바닥이 우2.67cm/좌5.43cm 떠 있음을 확인했다. 이전 검사는 높이를 계측했지만 초기 접지를 선별 조건으로 쓰지 않았고0.1초 검사도 힘 지지를 요구하지 않았다. clip7을 RSI에서 제외하고 최소 한 발 메시 바닥≤2cm·연속구간/양끝제거 교집합으로 재선별하여clip0/8의51프레임만 새grounded bank에 남겼다. 이전bank/영상과 AMP전문가21개는 보존한다.
- CPU기하1,836배치 관통/면투영/높이/초기발간격 위반0, 최소 한 발 간격1.14~1.85cm. 3,672상태0.1초PD·마지막 실제발수직힘≥1N·부분reset 통과. 별도0.4초 대표고정PD는0/432통과로 골반하강 문제가 여전히 남는다. 후보 다양성 감소/초기 정확히0cm·장기균형 미보장은 현재config와보고서에 명시했다. GPU/학습 실행 없음.
- 양발 간격 숫자 표시·장면 중심 자동 카메라의 약13초 새CPU RSI영상/검수는 `output_etc/check/push_native_phase_xy_grounded/REPORT.md`. 지적 프레임을 재현하는 실제mesh 회귀를 추가하고 관련CPU22검사·문법/diff를 통과했다.

### PUSH 원본 자세 RSI·XY 가변 상자·검수 영상

- 사용자 요청으로 `push_native_phase_xy` 별도 env/train/test/VNC/output을 추가했다. 출발1~2m·방향/배치/담당자 랜덤·3~5m 목표·기존soft tilt 보상을 유지하고 상자X/Y각1~1.5m/0.1m격자·Z1.3m/20kg/마찰0.6·경로간격2.5m로 구성했다. 원본PUSH21개 AMP는 유지하며 RSI만 높이/실제밀기/연속구간 선별5클립1,196프레임·DOF위치/속도 원본 보존으로 바꾼다. 기존bank/checkpoint는 보존하며 직접resume하지 않는다.
- CPU기하43,056배치 관통/면투영/높이위반0·0.1초 원본PD86,112상태 통과·원본DOF/부분reset·10만샘플 pool/계약검사·CPU87회귀 통과. 별도0.4초 고정PD 대표검사는373/1080통과로 골반하강 문제가 남으며 장기균형/정책성공을 보장하지 않는다. GPU학습/PPO/전이 미실행. 시작/종료GPU1은75MiB/PUSH PID없음·GPU0기존PID생존 확인, 학습/MPS중단 없음.
- 실제발mesh/머리collider·상자·양시점·5클립연속/극단사례의 약46초 CPU RSI 자세열 영상과 검수결과는 `output_etc/check/push_native_phase_xy/REPORT.md`. 물리rollout 영상과 구분했다. 현재 명령은config의`push-native-phase-xy`절을 따른다.

### 서버 PUSH 모션 묶음 압축 해제·로컬 데이터 검사

- 사용자 제공 `tokenhsi/data/stage1_push_motion_data_20261020.tar.gz`를 동명 별도 폴더에 내부 경로 그대로 해제했다. 기존 데이터는 덮어쓰지 않았으며31파일 SHA256·manifest 경로·캐시 motion/asset hash를 확인했다.
- CPU 실제 MotionLib 로드/보간·FK/유한값 검사: PUSH21개/8,006frame/266.17초·보행4개/24.33초다. PUSH 전 frame 머리>골반·관절 한계 오차1e-4rad 이내다. 양손 각각≤1m·과거9frame·DOF 후보1,617개, 동봉 cache의 배치 후 양손 높이까지 제한한 별도 로컬 후보1,616개를 저장했다. 원본 모션/cache는 보존했다.
- 동봉 cache는 손 중점≤1m·X1.2~1.5/Y1.8/Z1.3이며 각 손 기준과 다르다. h080 YAML에 AT 기울기 제한이 없고 h100_at5deg는 손목표1m다. 실행 코드 미포함이므로 학습에 설치하지 않았다. 로컬 physics/PD/전도·정책 검사는 미실행이다. 결과·자세 표본·보행 DOF 한계는 `output_etc/check/push_motion_archive_20261020/REPORT.md`.

## 2026-10-19

### 3~5m 정책 loco 시작 거리별 평가

- 평가 opt-in `TOKENHSI_PUSH_DISTANCE_EVAL=1`을 추가했다. 실제 초기 상자→목표 XY 거리3~3.5/3.5~4/4~4.5/4.5~5m별 최초/종료 AT 성공, 최초 도달시간, 종료 목표거리와 사람→상자 접근거리를 기록한다. interaction RSI 시작을 거부하며 첫 scene 종료 후 reset 표본은 제외한다. 매 완료 repeat의 원본/집계를 원자적으로 저장한다. 학습·보상·목표 배치는 변경하지 않는다.
- CPU8개 회귀(거리 경계/분모·최초/종료 구분·비동기 종료 masking·중간 JSON 저장 포함), Python/셸 문법·diff 확인. `output/push_distance_eval_20261019/launch.sh`로23000저장본·GPU1 native/MPS 우회·저메모리1환경·loco·seed20261019·50scene 목표/최대1시간 평가를 시작했다. 가상1280×720 화면은 유지하되 렌더 간격을 늘린다. 실제 checkpoint 복원/물리360step, 평가 프로세스 MPS 변수 미상속·GPU1 선택, 두 학습PID 생존·로그 갱신 및 GPU1 MPS 로그 신규 기록 없음을 확인했다. 최종 성공률/전체50scene 완료는 아직 미확인이다.
- 후속12:00:39에50scene/100agent 완료·평가 종료를 확인했다. 최초/종료AT32/100, 거리순6/22·9/24·11/28·6/26, 두 agent 동시5/50scene, 조기termination0이다. 실패68건 평균 목표거리 감소0.154m·66건0.5m 미만으로 이동 부족이 관측되나 접촉/자세 원인은 미계측이다. 결과/한계는 `output/push_distance_eval_20261019/REPORT.md`. 종료 후GPU1 epoch23343→23599/GPU0 2586→2640·기존PID 생존·GPU1 MPS 로그11516byte 무변경 확인. 같은23000저장본의GPU1 native·loco 시각화 명령을config에 추가했다(새 viewer 실행은 하지 않음).

## 2026-10-18

### 공통 면 soft tilt 상자 기준3~5m 목표 전이 점검

- 사용자 요청으로 기존 사람 기준XY±2m 목표를 보존하면서 상자 초기 중심 기준 거리3~5m·랜덤 방향의 opt-in sampler와 `push_contact_face_at_tilt_soft_3to5m` env/train/test/VNC·별도 output을 추가했다. 두 담당 이동/RSI 경로 간격·clearance를 검사하며 보상/물리/episode600·570D·기존 RSI bank는 유지한다. 후반 RSI는 초기 이동 거리 비율을 적용하므로 남은 거리까지3~5m로 제한하지 않는다.
- CUDA 비노출·MPS 미연결 CPU147테스트,2048scene×3seed 거리/경로 조건, 실제2환경1,000agent reset·양 담당 RSI/부분reset/목표 고정·24step 물리/AMP/Actor/Critic 연결 통과. 초기 거리3.0005~4.9985m·RSI 후0.5190~4.9939m·reset 기울기0도. 기존89후보 모두 새 배치12step PD 자세/발 지지/손18cm근접 screen 통과·bank 무변경(실제 힘 접촉/정책 성공 검사가 아님).
- CPU 명시적 전이 도구는 전체 원본 계약과 새 연결 audit를 비교하고 목표 분포 외 변경을 거부한다.17500epoch 저장본의181개 모델/관측·value·AMP 정규화 텐서·epoch/frame 보존, optimizer/best reward/env state 초기화·새 계약 수용/원본 거부·strict CPU network load/유한 forward 확인. 결과 `output/push_contact_face_at_tilt_soft_3to5m_transfer/{REPORT.md,validation.json}` 및 인접`_check`,`_checkpoint_check`,`_screen`폴더. 실제 전환 시 최종 저장본으로 재전이한다.
- GPU resume/PPO·정책 물리 rollout·장거리 성공률·600step 충분성은 미검증이다. GPU0/1 학습 중이라 GPU 실행/학습 중단·재시작은 하지 않았다. 검사 중 기존 학습 epoch17676→17692/1807→1808 및 마지막 두 PID 생존·MPS server.log 갱신 없음 확인. Python/셸 문법·diff 확인.

## 2026-10-17

### 공통 면 soft tilt 정책의 같은 면/다른 면 평가

- 평가 opt-in 면 계측에 full quaternion·유한3D 측면/상하면 분리와 각 손의 해당 면 외향 net force≥0.5N·박스 수평힘≥0.5N proxy를 추가했다. 첫 AT 성공 전 이동의 면별 기울기/10·15도 초과 및 episode 주사용 면별·초기 목표 기하별 성공을 집계한다. 학습/보상은 변경하지 않았다. CPU5개 회귀·summary 통합 예제·Python/셸 문법·diff 확인.
- GPU1 native·MPS 우회·저메모리1환경·10000epoch·보행 시작40회 평가가1시간 timeout으로37scene/74agent까지 완료됐다. 완료 로그를 복원했으며 AT 최초/종료5/74다. 힘 proxy의 도달 전 이동에서 같은 면9333step/평균7.61도/10도 초과25.85%, 다른 면18step/평균0.41도였다. 양면 주사용 episode가 없어 양면 성공 우월성은 미확인이다. 결과/정의/한계: `output/push_face_tilt_eval_20261017/{REPORT.md,summary.json,completed_results.json}`.
- 실제 평가 환경의 MPS 변수 미상속·CUDA/Vulkan GPU1 일치·checkpoint/물리/렌더링 및 종료 확인. 전후 기존 두 학습PID 생존·epoch10508→10819/766→816·MPS 로그 추가0byte 확인. 학습/MPS 중단·재시작은 수행하지 않았다.

### 서버 object tokenizer500 로컬 resume 준비

- 후속 사용자 GPU0 학습 종료 후 GPU0 MPS client 없음·GPU1 학습PID851709를 확인하고2048환경으로 짧은 GPU resume 검증을 수행했다. 로컬PyTorch2.0.0에서500→502epoch/frame65798144·optimizer step24000→24096·실제 물리/PPO/AMP 업데이트·checkpoint 저장·정상 종료 확인. 동결 actor76개·관측/AMP RMS·코드/초기pth 무결성 유지, adapter/복제tokenizer/critic/discriminator93개 텐서 변경 및 모델 전체 유한값 확인. 결과 `output/object_tokenizer_finetune/local_validation_20261017/check/`, 로그는 상위 `check.log`. 종료 후GPU0 client 없음·GPU1 PID/로그 갱신·MPS 오류 신규 기록 없음 확인. 본학습은 미실행이다.
- 복사된 `ports/ksh_object_finetune/launch.py`의 서버 절대경로 무결성 참조를 로컬 스냅샷으로 변경하고, `resume_local.sh {check|train}`을 추가했다. GPU0·`tokenhsi_koo`·2048환경·seed42·500저장본·별도 timestamp 출력이며 기존 학습/MPS 변경이나 자동 실행은 하지 않는다.
- CPU pth 로드(epoch500/frame65536000/optimizer)·relation/task 계약·63의자 asset fingerprint·136모션 참조·초기/재개 관측 및 AMP RMS 동일성·188파일 무결성 조회·GPU 숨긴 Isaac Gym→PyTorch2.0.0 import·셸 문법/diff 확인. 복사본의 upstream 대비 기존3파일 차이는 유지했다. GPU0 기존 학습/높은 메모리 점유를 확인하여 GPU 시뮬레이션·resume/PPO·본학습은 실행하지 않았다. 서버 PyTorch2.4 개발버전과의 GPU 호환성은 미검증이다.

### 공통 면 CONTACT+AT의 AT soft tilt 전이 준비

- 사용자 선택대로 AT state/progress를5~15도 선형 감쇠하고 성공10도 이하로 제한하는 `push_contact_face_at_tilt_soft` env/train/test/VNC·별도 bank/output을 추가했다. 목표 도달 포화 후에도 감쇠한다. CONTACT 자체 보상/접촉·박스1.3m/20kg·물리·RSI 후보는 유지한다.
- 엄격한 원본/대상 env·보상/RSI/hash 계약 검사 CPU 전이 도구로 원본8000epoch 저장본의 정책/가치/AMP/정규화·epoch/frame을 보존하고 optimizer/best reward/env state를 초기화한 전이본을 만들었다. 경로: `output/push_contact_face_at_tilt_soft_transfer/`. 저장 이후 진행·rollout/AMP replay는 복원되지 않는다.
- CPU74개 관련 회귀·181개 가중치/정규화 텐서 동일·대상 metadata와 원본 계약 거부·셸/Python 문법/diff 확인. GPU 복원/PPO·새 물리 rollout/GUI·장기 성능은 미검증이다. 기존 학습 중단/재시작·GPU/MPS 실행은 하지 않았다.

## 2026-10-16

### 고정 크기 공통 면 CONTACT+AT 전용 설정

- 사용자 제공 GPU4·6 방식과 최종 선택한 접촉 score>0.9를 `push_contact_face_at` 새 실험으로 구현했다. 현재1.3m·20kg·마찰0.6·570D/6토큰·랜덤 AT 배치/AMP를 유지하고, 목표 방향60도 이내 면 중 최대 손거리 최소의 공통 면을 선택한다. 접근 state는 `exp(-10*max(d_R,d_L)^2)`, 접촉은 이 값>0.9 및 각 손/상자 힘≥0.5N이다. force normal도 선택 면과 일치하며 물리 tick OR 누적은 유지한다. CONTACT/AT 각0.4/0.2/0.2, 실제 접촉/AT에 의한 포화·AT 자체 성공/접촉 기록 구분을 연결하고 기존 AT 기울기 감쇠는 새 경로에서 제거했다.
- RSI를 면 중심±접선0.2m·바닥 기준0.8m/stand-off6cm로 정렬하고 전용 bank·`goal_contact_face_xy_v1` 계약을 분리했다. CPU PD12step에서 기존90후보 중89개(clip별30/59)를 수용하고 자세 안정성 실패1개를 제외했다. 이는18cm 근접·발 지지/자세 screen이며 실제0.9/힘 접촉이나 정책 성공 검사는 아니다(`output/push_contact_face_at_rsi_screen/screen.json`).
- env/train YAML·train/test/VNC·CPU 검사/screen wrapper 및 문서를 추가했다. 로컬 test viewer도 GPU 환경 helper를 거쳐 native의 CUDA/Vulkan 일치·MPS 우회를 전달한다(새 GUI 실행은 미검증). 새 설정은 별도 scratch/output이며 기존 checkpoint 직접 resume/test는 불가하다. 사용자 지시대로 기존 학습을 변경/중단/재시작하거나 본학습을 시작하지 않았다.
- CPU199개 기존/신규 회귀 및 추가3개 면 선택 최대거리/60도 경계 테스트 통과. CPU2환경1,000agent reset·양 담당 배정 RSI·partial reset·24step 유한 물리/보상·목표 고정·570D Actor/Critic/AMP 연결 통과(`output/push_contact_face_at_connections/audit.json`). 셸 문법/diff 확인. CPU 확장 캐시 쓰기 제한은 GPU 비노출 호스트 검사로 해소했고 GPU/MPS 실행은 하지 않았다. 검사 후 기존 학습PID42928/46498 생존 확인. GPU/PPO 업데이트·checkpoint 복원·GUI·장기 성능은 미검증이다.

### 목표 반대편 PUSH 접촉 근접 판정 state0.9

- 사용자 요청으로 GPU1 대상의 목표 반대편 reference 공통 접촉/진단에서 양손18cm 조건을 각 손 면 XY 점수의 조화평균 CONTACT state `H>0.9`로 교체했다. 각 손 반력·상자 전방 힘≥0.5N 및 손 높이 자유는 유지한다. 보상 완료 포화값을 재사용하지 않으며 `both_hands_near`도 같은 기하 판정을 사용한다. 기존 로컬 reference의 거리 판정·보상·면 선택·RSI·YAML/checkpoint 계약은 유지했다.
- CPU140개(엄격 threshold·15cm 거부·양손 중점 오인 방지·0.5N 경계·높이 자유·회전/기존 reference 및 관련 보상 회귀) 통과. 실행 중 학습/GPU/MPS는 변경하지 않았고 새 물리 rollout은 실행하지 않았다. 새 프로세스부터 적용되며 목표 반대편 reference를 쓰는 다른 실험의 접촉 진단에도 적용된다.

### PUSH 면/모서리 평가 계측

- 사용자 요청으로 평가에만 `TOKENHSI_PUSH_FACE_AUDIT=1` CPU 계측을 추가했다. 초기 목표 반대 ray의 면 중앙/비스듬/모서리 구간, 양손의 같은 면/다른 면/모서리 모호 XY 근접 proxy, AT 최초/종료 성공·도달시간·목표거리 감소를 기록한다. 학습/보상/기존 평가 기본값은 유지한다. 집계는 `scripts/multi_agent/summarize_push_faces.py`를 사용한다(`tokenhsi/` 아래).
- CPU3개(회전·양손 배치/모서리 tie·종료 masking) 및 diff/launcher 셸 문법 통과. 사용자 뷰어 종료 후 GPU1 native·저메모리1환경·12500저장본·40회 직렬 평가를 `output/push_face_eval_20261016/`에서 시작했다. 실제 checkpoint/물리 step/첫 scene 평가 완료 및 평가 프로세스의 MPS 변수 미상속, 두 학습PID42928/46498·학습 로그 갱신 확인. 최종40scene/80agent 평가 완료(13:07:59). 전체AT40/80, 면중앙12/44·비스듬17/24·모서리11/12이며 실패사례 평균 목표거리 감소0.052m다. 모서리의 도달 전 이동 구간에서 다른 면 XY 배치86.3%를 관측했으나 힘 접촉/인과 우월성은 검증하지 않았다. 신규 공통 면 정책의 결과가 아니다. 집계/분석은 `output/push_face_eval_20261016/{summary.json,analysis.md}`에 저장했다.

## 2026-10-14

### 이전40D·AT0.5m GPU0 resume 경로

- 사용자 요청으로 이전PID9090의 최신9000epoch 저장본용 `push_soft_legacy40_at05` env/train/test/VNC·별도 bank/output을 만들었다. 기본30D는 유지하고 명시적40D 옵션으로 이전 조작 상태10칸을 복원한다. bank 파일 위치만 달라도 hash와 나머지 계약이 동일하면 checkpoint를 수용한다.
- CPU61개(30D/40D 관측·bank 경로 이동 및 내용/관측 불일치 거부·관련 보상/네트워크) 통과.9000 저장본40D·AT0.5m·원본bank SHA256 일치·셸 문법/diff 확인. GPU0 MPS client 없음·GPU1 새 학습PID42928 확인. 사용자 최종 지시대로 명령만 제공하고 GPU 학습/resume는 실행하지 않았다.

### 30D·AT0.1m RSI/AMP GPU 연결 검사

- 사용자 요청으로 현재 soft tilt 설정(사람223·박스30·목표1D/총570D, AT progress_delta0.1m)을 비어 있는 GPU1 MPS에서2환경 비학습 검사했다. 검사 전후 학습PID 없음·MPS 연결 client 없음·GPU1 점유75MiB 확인. 학습/MPS 재시작은 수행하지 않았다.
- 1,000agent reset·양 담당 배정 RSI 정렬·부분 reset·24step 유한 물리/관측/보상·목표 고정·6토큰 Actor/Critic·AMP 전문가128개/Discriminator 연결 통과. 조작 시작51.1%·후반 요청67.7%·reset 박스 기울기0도·bank 보존 확인. 결과/로그: `output/push_soft_box30_at01_connections/{audit.json,check.log}`. PPO/AMP optimizer 업데이트·본학습은 미실행이다.

### soft tilt AT progress 포화0.1m·30D scratch 명령

- 사용자 요청으로 현재 soft tilt YAML의 AT `progress_delta`를0.5→0.1m로 변경하고 validator가 유한 비음수 거리를 받도록 했다. 기울기 감쇠/성공·물리/RSI 후보는 유지하고 bank reward 계약/재사용 사유를 갱신했다. 사람223·박스30·목표1D(총570D) scratch 학습 명령 및 별도 출력 `output/push_soft_box30_at01`을 config에 기록했다.
- CPU0.1m 경계·bank 계약·570D Actor/Critic·관련 보상 회귀68개 통과. wrapper 셸 문법·diff 확인. GPU1 읽기 전용 점유75MiB·이전 학습PID 없음 확인. 새 학습/물리 rollout은 실행하지 않았다.

### 문 없는 PUSH 박스 관측30D

- 사용자 요청으로 `push_only` 박스 관측에서 추가 조작/접촉 상태10칸을 제거하고 물리 정보30D만 입력한다. 전체 관측590→570D·6토큰 유지. OPEN 포함 설정은40D 유지. 보상/진단의 내부 상태는 유지하며 checkpoint 계약의 기존 `object_obs_size` 비교로40D 저장본 직접 resume/test를 차단한다. 실행 중 학습에는 적용되지 않는다.
- CPU 관측30D/570D Actor·Critic 연결 및 PUSH only·soft tilt·OPEN/PUSH 회귀55개 통과. 초기 샌드박스 확장 캐시 쓰기 실패 후 GPU를 숨긴 호스트 CPU 검사 통과. 문서·diff 확인. 새 물리 rollout·GPU·학습·checkpoint 전이는 수행하지 않았다.

### GPU0·GPU1 MPS 활성화 및 resume 명령 안내

- 사용자 요청으로 두 GPU의 학습 프로세스 없음과 MPS 비활성을 확인한 뒤 사용자 전용 MPS를 켰다. 두 GPU server ready·연결 CUDA client 없음 확인. 학습은 시작하지 않았다.
- 기존 train wrapper의 runtime 자동 연결을 확인하고 CONTACT+AT1500 / soft tilt8500 저장본 resume 명령을 안내했다.

### GPU0 단일 GPU CONTACT+AT 1500 저장본 재개

- GPU1 제거 후 사용자 요청으로 남은 GPU0의 CONTACT+AT를 최신1500epoch checkpoint·2048환경·seed1485·최대100만epoch로 재개했다. MPS 시작/종료/복구는 수행하지 않았다. 이전 로그는 보존하고 실행 기록은 `output/contact_resume1500_single_gpu_20261014/launch.json`에 분리했다.
- checkpoint cuda:0 복원·1501/1502epoch·약11900~12300FPS·실제 GPU0 점유와 학습PID 확인. launcher 셸 문법 확인. 1500 이후 미저장 진행·rollout/AMP replay는 복원되지 않는다.

### GPU1 soft tilt 사용자 중단 후 6000 저장본 재개

- 사용자 요청으로 GPU1 학습 종료·MPS 준비/연결 client 없음 확인 후 최신6000epoch 저장본에서 기존2048환경·seed356·최대100만epoch로 재개했다. 이전 실행/로그는 보존하고 새 launcher·로그·run 기록을 `output/push_soft_resume6000_20261014/launch.json`에 분리했다.
- checkpoint 복원·6001/6002epoch·약12400FPS·실제 GPU1 배치·GPU0 기존 학습PID 유지 확인. launcher 셸 문법 확인. 6000 이후 미저장 진행과 실행 중 환경/AMP replay는 복원되지 않는다.

### GPU1 PUSH+AT soft tilt 5500 저장본 재개

- 사용자 요청으로 GPU1 학습/viewer 없음과 MPS 준비 상태를 확인한 뒤 같은 실험5500epoch checkpoint에서2048환경·seed356·기존 최대100만epoch로 재개했다. viewer/native 환경변수를 제거하고 기존 학습 MPS 자동 연결을 사용했다. 실행/로그/새 run은 `output/push_soft_resume_20261014/launch.json`에 기록했다.
- checkpoint 로드·`load_checkpoint: true`·5501/5502epoch 진행·학습 fps 약12696·실제 GPU1/MPS 연결 및 GPU0 기존 학습PID 유지를 확인했다. launcher 셸 문법·문서 경로·diff 확인. 종료 전5500 이후 미저장 진행과 rollout/AMP replay는 복원되지 않는다.

### 시각화 작업의 기존 학습 보호 지침

- 사용자 요청으로 `AGENTS.md`에 학습과 viewer의 MPS 공유 금지·로컬 native 경로 명시·새 GPU 조합의 별도 GPU 검증·GPU 오류 즉시 재시도 중단·학습 PID/로그 전후 확인·영향 즉시 보고를 필수 규칙으로 추가했다. 초기화 검사만으로 병행 안전을 단정하지 않으며 학습/MPS 복구는 사용자 지시를 따른다. `config.md` 로컬 VNC 절에서 해당 지침을 연결했다.
- 문서 링크·경로·diff를 확인했다. 문서만 수정했으며 GPU 실행·학습·MPS 변경은 수행하지 않았다.

### 로컬 VNC Vulkan 충돌 우회

- GPU1 soft tilt VNC의 초기 segfault를 재현하고 gdb로 Gym 내장 Vulkan `vkGetDeviceProcAddr` 충돌을 확인했다. `TOKENHSI_GUI_GPU_MODE=native` 옵션으로 Mesa 레이어를 우회하고 뷰어만 MPS 연결을 분리하며 CUDA/Vulkan·기본 CUDA 장치를 지정 물리 GPU로 일치시켰다. 기본 mesa 경로와 학습 설정은 유지한다. 실행 명령/구조 문서를 갱신했다.
- GPU 선택 CPU7개·Python/셸 문법 통과. GPU1 soft tilt5500 및 GPU0 CONTACT1000 checkpoint 복원·짧은 VNC rollout/평가 정상 종료를 확인했다. 중간 MPS 연결 검사09:57에 CUDA 오류가 발생하면서 GPU1 학습PID436397도 종료됐다(MPS 로그 동시 종료 기록). GPU0 학습PID459946은 유지됐으며 GPU1 학습/MPS 재시작은 수행하지 않았다. 로그는 `output/vnc_native_fix_20261014/`, 평가 결과는 각 실험의 `metrics/`에 있다. 장기 시각화는 미검증이다.

## 2026-10-13

### CONTACT 양손 표면 근접·접근 정렬 + AT 구현

- 사용자 합의한 세 변경을 `push_toy_contact_align_at` 별도 실험으로 구현했다. CONTACT state/자체 성공은 목표 반대편 면까지 각 손 XY 거리의 조화평균, progress는 거리 접근·위치 정렬50/50 및 중심 거리2→1m 근거리 접근 감쇠를 사용한다. 힘 접촉 조건은 추가하지 않았다. 최신 AT 성공만10도 설정의 AT 보상·높이 gate·자체/AT 성공 포화는 유지하며 기존 설정 계산도 유지한다.
- 전용 env/train·train/test/VNC·검사 wrapper·bank/output/checkpoint 계약 및 문서를 추가했다. 물리/참조/RSI 배치 불변이므로 후보/PD screen을 보존하고 bank reward 계약/재사용 출처만 변경했다. CPU176개 회귀 통과. 호스트 runtime CPU2환경에서1,000agent reset·양 담당 배정 RSI·부분 reset·24step 물리/관측/보상·6토큰 Actor/Critic/AMP 연결·목표 고정/bank 보존 확인(`output/push_toy_contact_align_at_connections/audit.json`). Python/YAML/셸 문법·문서 링크·diff 확인. 본학습·checkpoint 복원·GUI·장기 개선은 미검증이다.

### 중단 후 GPU0/GPU1 MPS 복구

- 사용자 요청으로 두 GPU의 학습/client 없음과 GPU0 fatal GPU error 기록을 확인한 뒤 사용자 MPS를 정상 종료/재시작했다. 종료 후 비활성 소켓은 helper로 `pipe.stale.*`에 보관하고 새 pipe를 생성했다. 두 GPU runtime 자동 연결·Isaac Gym 선행 import·CPU→CUDA 전송/연산/동기화 통과. 학습/output/checkpoint는 삭제하거나 시작하지 않았다.

### GPU0 MPS 및 AT 성공 전용 기울기 scratch 설정

- 사용자 요청으로 GPU0 점유 확인 후 전용 MPS를 활성화하고 서버 준비/상태 응답을 확인했다. `push_toy_random_at_10deg_at_success_tilt` 전용 env/train·train/test/VNC·검사 wrapper·bank/output을 추가했다. AT 성공에만10도 이하를 요구하고 초과 시 성공/AT bonus/AT에 의한 PUSH 포화를 해제한다. AT state/progress의 거리식·PUSH 각도 자유/높이±8cm·기존 실험은 유지한다. 기존 RSI 후보/PD screen을 보존하고 보상 계약/출처를 분리했다.
- CPU145개 회귀·GPU0 MPS2환경1,000agent reset·양 담당 배정 RSI·부분 reset·24step 물리/관측/보상·6토큰 Actor/Critic/AMP·bank 보존 통과(`output/push_toy_random_at_10deg_at_success_tilt_connections/audit.json`). YAML/셸/Python 문법·diff 확인. GPU1 기존 학습 유지. 본학습은 시작하지 않고 scratch 명령을 제공했다. 장기 성능/새 checkpoint 복원/GUI는 미검증이다.

### AT 기울기 감쇠 scratch 설정 및 RSI 연결 검사

- 사용자 요청으로 `push_toy_random_at_10deg_at_tilt_soft` 전용 env/train·train/test/VNC·검사 wrapper·RSI bank/output을 추가했다. AT state/progress는5도 이하 유지·5~15도 선형 감쇠·15도 이상0이며 도착 포화 이후에도 적용한다. AT 성공은 기존 거리/높이 조건에10도 이하를 추가하고 초과 시 현재 성공/보너스/AT에 의한 PUSH 포화를 해제한다. PUSH 각도 gate는 제거하고 높이±8cm·손 접근/자체 성공은 유지한다. 이전 실험 기본값은 유지한다.
- 물리/참조/RSI/배치 불변 비교와 기존 후보/PD screen 보존 확인으로 전체 후보 재screen은 생략하고 bank 보상 계약/출처를 분리했다. CPU123개 회귀·GPU1 MPS2환경1,000agent reset(박스 기울기0도)·양 담당 배정 RSI·부분 reset·24step 유한 물리/관측/보상·목표 고정·6토큰 Actor/Critic/AMP 연결 통과(`output/push_toy_random_at_10deg_at_tilt_soft_connections/audit.json`). 샌드박스 GPU 접근 실패 후 호스트 실행. YAML/셸/Python 문법·diff 확인. 본학습은 시작하지 않았고 장기 개선/새 checkpoint 복원/GUI는 미검증이다.

### GPU1 MPS 활성화

- 사용자 요청으로 GPU1 점유를 확인한 뒤 `source mps/shell.sh; mps_start 1`로 사용자 전용 MPS를 활성화했다. GPU 서버 준비·`mps_status 1` 응답·연결된 CUDA client 없음 확인. 학습은 시작하지 않았다. 접속 경로는 `/tmp/mps-test-1000/gpu1/pipe`이며 기존 runtime 자동 연결을 사용한다.

### PUSH 기울기 제한을 AT state로 이동

- 사용자 요청으로 `push_toy_random_at_10deg_at_state` 전용 env/train·train/test/VNC·RSI bank/output을 추가했다. PUSH 각도 제한은 제거하고 높이±8cm는 유지한다. AT state만10도 초과 시0이며 도착 포화도 우회하지 못한다. AT progress/성공·도착 시 PUSH 포화와 기존 실험 기본값은 유지한다. RSI 후보/PD screen을 보존하고 보상 계약/재사용 출처만 갱신했다.
- CPU 회귀102개 통과(10도 경계·목표 도착 전후·state 복구·PUSH 높이 gate·bank 계약·기존 실험). YAML/셸 문법·diff 확인. PID174445는 작업 환경에서 조회되지 않아 기존 학습 변경/재시작은 수행하지 않았다. 새 학습/물리/복원/GUI는 미검증이며 기존 checkpoint 직접 resume/test는 새 계약과 호환되지 않는다.

### GPU1 PUSH 최신 저장본 VNC

- 사용자 요청으로 GPU1 PUSH goal-rear RSI의5500epoch checkpoint를 같은 GPU·저메모리1환경·2사람/2박스·1280×720·6093포트·보행 시작으로 시각화했다. 실행/로그는 `output/push_gpu1_view_20261013/{launch.sh,viewer.log}`, tmux는 `koo_push_gpu1_view_20261013`이다.
- checkpoint 복원·정책150step·6093포트·GPU0/GPU1 기존 학습 프로세스 유지 확인. GPU1 총 점유23707/24576MiB 확인. 별도 에피소드 성공률 평가는 수행하지 않았다.

### 빠른 PUSH VNC와 GPU0 정책 평가

- 사용자 요청으로 비어 있는 GPU0에서4500checkpoint VNC를 다시 실행했다. 렌더2틱 간격·draw 대기 해제·매step AMP debug 생략 옵션을 추가했다(기본 동작 유지, 물리/정책 step 불변). 실행/로그는 `output/push_gpu0_view_20261013/{launch.sh,viewer_fast.log}`다.
- 선택적 PUSH 평가 감사와 평가용 학습 RSI 혼합 옵션을 추가했다. 보행 시작·RSI 혼합 평가는 각각 `output/push_gpu0_eval_20261013/{launch.sh,launch_train_mix.sh}`로 재현하며 metrics에 완료 전후 접촉·에피소드 접촉·목표 고정·완료 보상 포화를 기록한다. 학습 설정/checkpoint 계약은 유지한다.
- CPU 회귀90개·Python/셸 문법·diff 검사 통과. GPU0 checkpoint 복원/뷰어 rollout 및 두 평가 완료: 보행 시작16환경×10회 목표 도착0/320agent episode, RSI 혼합16환경×3회9/96·이동 중 접촉proxy50.5%·완료 후 비접촉step99.4%·완료 보상 포화 위반0. 두 평가 모두 목표 drift0. GPU1 학습은 유지했다. 보행 접근부터 전체 과제를 수행하는 능력은 아직 부족하다. 결과는 `output/push_gpu0_eval_20261013/{REPORT.md,summary.json}` 및 각 `metrics/`에 기록했다.

### PUSH 4500 checkpoint GPU0 VNC 실행

- 사용자 요청으로 GPU1 PUSH goal-rear RSI의4500epoch checkpoint를 GPU0·평가1환경·2사람/2박스·저메모리·1280×720·6090포트에서 실행했다. 실행 파일/로그는 `output/push_gpu0_view_20261013/{launch.sh,viewer.log}`, tmux 세션은 `koo_push_view_20261013`이다.
- VNC 포트·실제 GPU0 배치·정책 rollout30step 이상 및 GPU0 OPEN/GPU1 PUSH 학습 프로세스 유지 확인. 수정된 접촉 진단은 새 뷰어에 적용되며 기존 학습 프로세스에는 자동 반영되지 않는다. 정책 성공률 평가는 수행하지 않았다.

### 랜덤 AT 목표 방향 접촉 진단 수정

- 목표 반대편 보상/RSI와 달리 로컬−Y 면·로컬+Y 반력을 사용하던 접촉 진단을 수정했다. `goal_opposite_surface_xy`에서 목표 반대 ray가 만나는 박스 면의 양손 XY 거리와 현재 박스→목표 방향 손/박스 반력을 사용한다. 목표 XY 일치 시 기존 yaw+Y로 fallback하며 손 높이 자유·거리18cm·각 손/박스 힘0.5N·같은 physics tick 양손 조건을 유지한다. 거리/근접/힘 진단과 all/moving/completed 접촉 집계에 같은 판정을 연결했다.
- 보상·AT 도착 후 PUSH 포화/손 떼기·checkpoint 계약은 유지한다. 기존 실험은 기존 판정을 유지한다. 실행 중 학습은 재시작하지 않았으며 수정은 새 프로세스부터 반영된다. 이전 TensorBoard 접촉 수치는 소급 수정할 수 없다.
- CPU 회귀90개 통과(전후좌우/대각/XY 일치·yaw3조건·반대 힘·한 손 힘 없음·표면 이탈·기존 분기 포함). GPU 시뮬레이션/수정 후 정책 rollout은 미검증이다.

## 2026-10-12

### 랜덤 AT 목표 반대편 RSI 손 정렬 수정

- 보상만 목표 반대편을 사용하고 RSI는 로컬−Y 뒤면을 사용하던 불일치를 수정했다. 현재 reset의 담당 목표로 양손 중점 표면 교점·목표 방향 수직 pair·6cm stand-off를 정하고 사람 자세/방향/선속도/각속도를 함께 yaw 정렬한다. 발 높이와 기존 실험 정렬은 유지한다. reset 순서상 RSI 전에 `_tar_pos`가 갱신되지 않으므로 새 샘플 목표를 직접 조회한다. 초기 배치의 뒤쪽1.5m RSI 경로 여유 검사도 목표 방향을 따른다.
- RSI bank/체크포인트에 `goal_opposite_surface_xy_v1` 계약을 추가하고 bank 후보/이전 hash를 보존했다(전체 후보 재screen 없음). train/test/VNC 기본 output은 `output/push_toy_random_at_10deg_goal_rear_rsi`. 이전 goal-rear checkpoint 직접 resume/test는 불가하며 기존 결과는 보존한다.
- CPU82개 회귀 통과(전후좌우/대각/XY일치·yaw3조건의 손 중점/방향/속도/발높이 및 기존 분기 포함). GPU1·2환경 1,000agent reset·양 담당 배정 RSI·부분 reset·24step 유한 물리/관측/보상·Actor/Critic/AMP 연결·bank 보존 통과(`output/push_toy_random_at_10deg_goal_rear_rsi_gpu_connections/audit.json`). GPU 접근/Isaac Gym 캐시 샌드박스 제한 후 호스트 재실행. GPU1 학습 종료와 GPU0 OPEN 유지를 확인했다. 본학습은 시작하지 않았고 새 checkpoint 복원/GUI/장기 성공 개선은 미검증이다.

### GPU0 OPEN 6000 checkpoint 재개

- 사용자 요청으로 종료된 OPEN strong-close를 최신 저장6000epoch checkpoint에서 GPU0·2048환경·seed1015·기존 최대9000epoch로 재개했다. checkpoint 로드·새 run의6001/6002epoch 진행·학습 fps·실제 GPU0 배치 및 GPU1 PUSH 프로세스 유지를 확인했다. 실행/로그/새 run은 `output/open_resume_20261012/launch.json`에 기록했다. 이전 종료 시6354epoch까지의 미저장 진행과 rollout/AMP replay는 복원되지 않는다.

### 랜덤 AT PUSH 손 보상을 목표 반대편 표면으로 변경

- 사용자 요청으로 `push_toy_random_at_10deg`에 `push_at.hand_reference: goal_opposite_surface_xy`를 적용했다. 매 step 현재 박스→목표 반대 ray와 yaw 기준 XY 직사각형 경계의 교점을 손 중점 기준으로 사용하고 XY 일치 시 로컬−Y 중앙으로 fallback한다. 높이 자유·10도/±8cm·성공0.9·Progress·AT/포화는 유지하며 다른 실험은 기존 기준을 유지한다. RSI 초기 참조 정렬은 기존 박스 yaw 기준이다.
- 보상/checkpoint 계약과 RSI bank reward metadata를 갱신했다(후보 그대로·재screen 없음·이전 hash 기록). train/test/VNC 기본 출력을 `output/push_toy_random_at_10deg_goal_rear`로 분리하고 기존 학습 output을 보존했다. 수정 전 checkpoint 직접 resume/test는 불가하다.
- CPU84개 회귀 통과(샌드박스 Isaac Gym 캐시 제한 후 호스트 재실행), 목표 전후좌우/대각·박스 yaw4조건·비정사각형·목표/박스 이동·XY 일치·기존 분기/AT 포화 확인. 실제 CPU2환경 1,000agent reset·양 배정 RSI·부분 reset·24step 물리/관측/보상 유한값·6토큰 Actor/Critic/AMP 연결·bank 보존 통과(`output/push_toy_random_at_10deg_goal_rear_connections/audit.json`). 본학습·새 GPU 물리·checkpoint 복원·GUI·정책 성공 개선은 미검증이며 학습은 시작하지 않았다.

## 2026-10-11

### PUSH만10도 제한 / ksh식 랜덤 AT 별도 설정

- 사용자 요청으로 `push_toy_random_at_10deg` env/train·train/test/VNC·별도 RSI bank/output을 추가했다. PUSH만10도 이하·초기 중심 높이±8cm로 제한하고 AT 성공/도착 시 PUSH 포화는 유지했다. ksh `_reset_task`의 사람별 사각 랜덤 목표/사람·박스 최소1m 방식을 참고해 XY±2m 랜덤 AT와 두 박스/뒤쪽1.5m RSI 이동 경로2.2m·상대 사람 기준점 여유 검사를 구현했다. 목표는 초기화 후 고정하고 RSI 선행 이동은 실제 샘플된 목표 거리 비율을 따른다. 기존 실험/PD 후보/공유 원본은 유지하며 새 보상·배치 계약으로 checkpoint를 분리했다.
- 관련 CPU49개 회귀 통과(Isaac Gym 캐시 권한 실패 후 호스트 재실행), 최종 새12개 경계/분포/경로 회귀 통과. 실제 CPU 및 GPU1 각2환경·1,000agent reset·양 담당 배정 RSI·부분 reset·24step 유한 물리/관측/보상·목표 고정·6토큰 Actor/Critic/AMP 연결·bank 보존 확인(`output/push_toy_random_at_10deg{,_gpu}_connections/audit.json`). YAML/셸 문법·diff 확인. 새 본학습/checkpoint 복원/GUI/장기 정책 성공은 미검증이며 본학습은 시작하지 않았다. 초기 배치/의도 경로 여유 검사로 정책의 임의 이동 충돌까지 보장하지 않는다.

### 리부트 후 GPU0 OPEN strong-close 재개

- 사용자 요청으로 최신 저장5500epoch checkpoint에서 GPU0·2048환경·seed1015·기존 최대9000epoch 설정으로 재개했다. checkpoint 로드·새 run의5501epoch 이후 진행·정상 학습 fps·실제 GPU0 배치를 확인했다. 실행/로그/새 run은 `output/open_resume_20261011/launch.json`에 기록했다. 리부트 전5500epoch 이후 미저장 진행과 rollout/AMP replay는 복원되지 않는다.

## 2026-10-09

### PUSH + AT GPU1 scratch 본학습 시작

- 사용자 요청으로 기존 GPU1 문 없는 PUSH를 SIGINT 종료한 뒤 `tmux koo_train`의 `push_at` 창에서 독립 PUSH+AT를 checkpoint 없이 새로 시작했다. 2048환경·2박스·seed5306·최대100만epoch이며 짧은 반복/resume/viewer 버퍼 환경변수는 제거했다. 실제 GPU1 배치·새 보상 config·`load_checkpoint: false`·첫 학습 fps 약12520을 확인했다. GPU0 OPEN은 유지했다. 실행/로그/창은 `output/push_toy_only_random_scene_at/launch.json`에 기록했다. 장기 성공률은 아직 확인하지 않았다.

### 문 없는 PUSH + AT 독립 edge 보상 실험

- 사용자 요청으로 기존 HOLDING+AT 방식의 `push_toy_only_random_scene_at` env/train·train/test/VNC/output을 분리했다. PUSH 손 중점 XY 근접/사람 접근과 AT 박스–목표3D 근접/XY 접근에 각각 state/progress/success0.2씩 지급한다. 자기2edge 합계0.9·동료2edge 평균0.1, AT 현재 성공 시 선행 PUSH 포화·손 떼기 완료 유지·목표 이탈 해제를 적용했다. AT은 손 접촉/순이동/속도 gate 없이 기존φ≥0.9·높이1mm로 판정한다. PUSH 자체는 근접 proxy 성공이며 물리 접촉/손 이동 입증과 다르다. 관계별 성공/보상 진단과 checkpoint 보상 계약을 추가했다.
- 기존/신규 CPU81개 통과(캐시 쓰기 샌드박스 제한 후 호스트 재실행). 실제 CPU2환경1,000agent reset·양 배정 RSI·부분 reset·24step 유한 물리/관측/보상·6토큰 Actor/Critic/AMP forward 통과(`output/push_toy_only_random_scene_at_connections/audit.json`). 동일 물리 PD 후보를 전용 bank로 복사하고 새 보상 metadata 및 재사용 출처를 기록했으며 후보 재screen은 하지 않았다. 기존 bank와 실행 중 학습은 유지했다. 새 본학습·GPU 시뮬레이션·checkpoint 복원·GUI/VNC·장기 성공률은 미검증이다.

### 문 없는 PUSH GPU1 scratch 본학습 시작

- 사용자 요청으로 `PUSH_ONLY=1`을 통해 `push_toy_only_random_scene`를 GPU1·2048환경·2박스·seed5306·최대100만epoch로 새로 시작했다. checkpoint/짧은 반복/viewer 버퍼 환경변수는 제거했다. 실제 GPU1·전용 config/2물체 인자·정상 학습 fps 로그를 확인했고 기존 GPU0 OPEN은 유지했다. 실행/로그는 `output/push_toy_only_random_scene/launch.json`에 기록했다. 속도 비교 benchmark는 하지 않았다.

### 문 없는 PUSH_ONLY 옵션 구현

- 사용자 요청으로 `relationGraph.push_only: true`의2H/2Box 모드를 구현하고 전용 `push_toy_only_random_scene` env/train YAML·train/test/VNC/output을 추가했다. 기존 random-scene wrapper에서 `PUSH_ONLY=1`로 선택한다. 문 asset/actor/DOF·토크·문/손잡이 접촉·OPEN reward 계산을 제외하고6토큰/590차원 관측과2개 물체 graph/RSI/힘/credit 연결을 적용했다. 박스 크기/물성·PUSH 보상·AMP/RSI·무작위 배치·최대100만epoch는 유지했다. 관측 계약이 달라 기존 checkpoint와 분리하며 공통2m bank/manifest metadata는 재사용한다. 문 없는 배치에서 문 간섭 검사도 생략한다.
- 기존 CPU78개·새3개 계약/보상 동일성/OPEN 미호출 검사 통과. 실제 CPU/GPU 각2환경1,000개 agent reset·두 강제 배정 RSI·부분 reset·24step 물리·bank 보존·6토큰 Actor/Critic/AMP forward 통과(`output/push_toy_only_random_scene{,_gpu}_connections/audit.json`). 문 actor/경첩 없음·32body/64DOF 확인. wrapper3개의 옵션 dispatch·Python/셸 문법·diff 확인. config/structure 반영. 사용자 요청으로 속도 benchmark는 생략했다. 장기 성공·checkpoint 복원·GUI/VNC는 미검증이다.

### PUSH 실제 무작위 배치 scratch 본학습 시작

- 사용자 실행 요청으로 `push_toy_midpoint_2m_random_scene`를 GPU1·2048환경·seed5306·최대1000000epoch로 새로 시작했다. MAX_ITERATIONS/RESUME_CHECKPOINT/viewer 버퍼 환경변수를 제거하고 checkpoint 없이 실행했다. 실제 GPU1 배치·프로세스 메모리19544MiB·첫 학습 fps 로그를 확인했다. 기존 GPU0 OPEN은 유지했다. 실행/로그는 `output/push_toy_midpoint_2m_random_scene/launch.json`에 기록했다. 장기 성공률·checkpoint 복원·viewer는 미검증이다.

### 고정 크기 PUSH 실제 위치·방향·담당 배정 무작위화

- 사용자 요청으로 기본 unified SIT/HOLDING의 물리 배정/owner 주변 위치·yaw/참조 정렬 방식을 `push_toy_midpoint_2m_random_scene` 전용 env/train·train/test/VNC/output으로 분리했다. 보행 시작 박스는 사람 주변 무작위 방향1~2m·yaw/사람 heading 전 방향, 담당자는50% 교환한다. 목표는 reset 박스의 로컬+Y2m로 고정하고 조작 RSI는 같은 방향으로 phase 전진·양손 정렬·속도 yaw 회전을 적용한다. graph/관측/BPS/힘/credit 연결을 유지한다. 전진 진단도 episode 방향에 투영한다.
- 크기1.3m·질량/마찰·보상·AMP2모션·초기화 비율·2m PD bank·최대100만epoch는 유지했다. 구역 기준점(-4,±4) 및 swept 경로/문 간섭 제외를 추가했다. 새 배치 계약으로 이전 checkpoint와 분리했다. config/structure에 실제 동작·충돌 제외 분포·학습/로컬 추론/VNC 명령을 기록했다.
- CPU 회귀78개 통과(2048배치 샘플·회전5조건 접촉/성공/후진/진단 포함). 실제 CPU/GPU 각각2환경1,000개 agent reset·두 강제 배정 RSI 손 정렬·graph/관측/목표·부분 reset·AMP·24step 물리 유한값/목표 고정·bank 원본 보존 통과(`output/push_toy_midpoint_2m_random_scene{,_gpu}_connections/audit.json`). CPU 교환52.2%·조작49.9%·후반 요청69.5%, yaw/위치 방향 전 구간 확인. 본학습/뷰어는 시작하지 않았고 기존 OPEN은 유지했다. 새2048환경 학습·장기 정책 성공·checkpoint 복원·GUI/VNC는 미검증이다.

### PUSH 구역 교환 최대 epoch 100만

- 사용자가 학습을 종료한 뒤 요청한 대로 `push_toy_midpoint_2m_zone_swap` train YAML의 `max_epochs`만9000→1000000으로 변경했다. YAML 파싱과 다른 설정 동일 여부·diff 확인 통과. config 문서 반영. 학습은 재실행하지 않았다.

### PUSH 구역 교환 scratch 본학습 시작

- 사용자 시작 요청으로 `push_toy_midpoint_2m_zone_swap`를 GPU1·2048환경·seed5306·최대9000epoch로 새로 실행했다. MAX_ITERATIONS/RESUME_CHECKPOINT/viewer 버퍼 환경변수를 제거하고 checkpoint 없이 시작했다. 실제 GPU1 배치·약19.3GiB 프로세스 메모리·첫 학습 fps 로그를 확인했으며 기존 GPU0 OPEN은 유지했다. 실행 명령/로그는 `output/push_toy_midpoint_2m_zone_swap/launch.json`에 기록했다. 장기 성공률·checkpoint 복원·viewer는 아직 미검증이다.

### PUSH 구역 배정 50% 교환 구현·학습 보류

- `push_toy_midpoint_2m_zone_swap` env/train YAML·train/test/VNC를 분리했다. 매 환경 reset에서 양 사람의 담당 구역을 50% 확률로 교환하고 물리 actor 인덱스·배치를 유지한다. 담당 문/박스/목표·관측/BPS·graph·RSI·접촉력·순이동 실적·보상을 같은 배정으로 조회한다. 물리 문 토크는 원래 actor 순서로 적용한다. 보상/물성/AMP/RSI 비율과 기존 2m RSI bank는 유지한다. 네 agent/구역 조합별 성공·순전진·손 인정 거리 진단을 추가했다.
- CPU 회귀60개 통과. 실제 CPU2환경에서 양 배정×네 과제 조합·보행 시작·RSI 소유자·graph/관측/힘 연결·같은 tick 이동 실적·부분 reset·10000회 교환 비율·12step 유한 물리 검사 통과(`output/push_toy_midpoint_2m_zone_swap_connections/audit.json`). 사용자 요청으로 학습을 보류했다. 새2048환경 검증 학습은 시작되지 않았으며 기존 OPEN 프로세스만 유지 중임을 확인했다. 장기 효과·checkpoint 복원·viewer는 미검증이다.

### 로컬 PUSH 중점 XY·2m 목표·순이동 실적 설정

- 사용자 전달 기준을 선택적 `midpoint_xy_net_credit`로 구현하고 `push_toy_midpoint_2m` 전용 env/train YAML·train/test/VNC·RSI 검사/bank/output을 추가했다. 손 위치 보상은 양손 중점 XY↔뒷면 중앙, 접촉은 각 손의 yaw 기준 뒷면 XY 선분18cm 및 각0.5N/박스0.5N 반력이다. PUSH 몸 접촉은 차단하지 않고 같은 physics tick의 바닥/손 접촉에서 새 거리를 인정한다. 후진은 접촉 없이도 차감하고 reset 이후 실제 순이동으로 제한하며 RSI 선행 이동은 밀기 실적에서 제외한다. 목표2m·남은 거리 기반 성공 실적·손 떼기 후 현재 성공을 적용했다. 초기 RSI의 높이0.8/좌우±0.2는 유지한다.
- 사용자 요청으로 로컬 기존 PUSH 참조2개와 PUSH 단독 구성을 유지했다(서버21모션·OPEN/PUSH50/50 혼합은 이식하지 않음). 기존 OPEN/PUSH/공동 설정은 기본 분기로 유지하고 새 reward/목표/bank/checkpoint 계약을 분리했다. config/structure에 명령·차이·새 scratch 학습 필요를 기록했다. RSI 검사 목표 하드코딩 제거·수동 정렬 후 credit 초기화·bank 재사용 시 파일 보존·선택적12step 물리 smoke 검사를 추가했다.
- CPU 회귀77개 통과, 최종 새/기존 토이 계약18개 통과(샌드박스 캐시 쓰기 제한 실패는 호스트 재실행으로 확인). CPU2환경 OPEN13/13·PUSH90/171(모션별31/59)·1000reset/partial reset/고정 목표/AMP 통과(`output/push_toy_midpoint_2m_rsi_check`). GPU0·2환경에서 CPU bank 그대로1000reset/partial reset/AMP 및12step 물리·유한 보상/관측/이동 실적 통과(`output/push_toy_midpoint_2m_gpu_check`). Python/셸 문법·YAML 변경 범위·경로/diff 확인. 기존 GPU0 OPEN/GPU1 PUSH 본학습 유지, 뷰어는 띄우지 않았다. 새2048환경 학습/checkpoint/GUI/장기 정책 성공은 미검증이다.

### 사용자 요청 OPEN/PUSH GPU 교체 재개

- 사용자 확인에 따라 기존 OPEN/PUSH를 SIGINT로 종료하고 같은 설정·seed·2048환경으로 OPEN strong-close는 GPU0/2500epoch, PUSH는 GPU1/500epoch checkpoint에서 재개했다. 모델/optimizer/epoch/frame 복원 경로와 실제 GPU 배치·학습 fps 로그를 확인했다. 저장 이후 진행 및 실행 중 환경/AMP replay는 보존되지 않는다. 재개 명령/checkpoint/로그는 `output/gpu_swap_20261009/launch.json`, config에 일반 재개 절차를 기록했다. PID는 실제 프로세스로 확인한다.

### checkpoint 없는 AMP 물리 모션 VNC 뷰어

- `view_open_reference_motion.py`·`open_reference_motion_view{,_vnc}.sh`를 추가했다. 새 reference-close 저항에서1환경·시간별 AMP 관절 PD·자유 root로 재생하며 화면에 참조시각/문각도/손 총 접촉력/낙상을 표시한다. RSI/닫힌 문 전환·일시정지·다음 시작 및 JSONL 로그를 지원한다. config/structure에 checkpoint 없는 명령과 낙상/통과 성공 미보장 한계를 기록했다.
- GPU0·저메모리2097152/4·1280×720·VNC6091의 중간/후반 RSI2회 재생 정상 종료(`output/open_reference_motion_vnc_check`); 두 번 모두 이후 낙상했다. CPU1환경 닫힌 문/중간 참조 재생도 정상 종료·최대67.74도 후 낙상(`output/open_reference_motion_closed_check`). GPU 참조 시간 텐서의 장치 불일치를 수정 후 재검증했다. Python/셸 문법·유한 관측/관절·diff 확인 통과. 기존2048환경 본학습 두 개는 유지했다. 화면 캡처는 Pillow XCB 지원 부재로 확인하지 못했고 학습 정책/문 통과/5초 지지 성공 검증은 아니다.

### AMP 참조 PD 비교에 따른 OPEN 닫힘 저항 별도 설정

- 저항 후보9개를 추가 비교하고 preload1N·m 이상·중간/후반 RSI 두 시작의 비손 접촉 전 최대각95도 이상·선 자세90도 도달 조건에서90도 저항이 가장 큰 후보1/0.5/최대6/preload1을 선택했다(`output/open_reference_resistance_sweep/SELECTED.md`). `open_toy_reference_close` env/train YAML·train/test/VNC·별도 bank/output을 추가했다. 기존 reward·AMP·모델·90도/5초 조건은 유지하며 물리/bank가 달라 기존 checkpoint와 분리한다. config/structure에 명령·선택 근거·한계를 기록했다.
- 실제 문 CPU8조건에서 새 설정90→10도2.03초·2N·m/0.15초 충격 최대0.52도(기존5.28도)·20N·m 충격42.86도 확인(`output/open_toy_reference_close_physics_check`). CPU69개 통과. CPU2환경 RSI OPEN13/13·PUSH116/171·1000reset/partial reset/AMP 통과. GPU0·2환경 OPEN13/13·1000reset/partial reset/AMP 통과(`output/open_toy_reference_close_rsi_gpu_check`); 미사용 PUSH 후보는 CPU 검사본 보존. 최종 bank 계약8개·Python/셸 문법·YAML 변경 범위·diff/문서 경로 확인 통과.
- 부분 RSI 검사에는 기존 계약 일치 bank를 요구하는 `RSI_SCREEN_SKILLS` 옵션을 추가했고 기본 양쪽 task 검사는 유지했다. 물리 비교 스크립트는 선택 실험/output 지정이 가능하며 기존 strong-close 기본 검증은 유지한다. 참조 PD는 새 저항에서 RSI2/3시작에서 선 자세·비손 접촉 전90도에 도달했지만 닫힌 문 시작3조건은 실패하고 이후 낙상했다. 손 접촉 후 관성 개방도 포함하므로 유효 손 진행/5초 정책 성공을 입증하지 않는다. GPU0/1 본학습은 유지했으며 새2048환경 학습·checkpoint 복원·GUI/VNC는 미검증이다.

## 2026-10-08

### OPEN 참조 PD·손잡이 외력으로 개방 가능성 비교

- 비학습 CPU2환경 `check_open_reference_push.{py,sh}`를 추가했다. 기존/중간/현재강화 저항에서 원본 참조 관절 PD·자유 root·3참조시각×닫힘/RSI각18조건과 이상적 손잡이 접선 힘20/40/60N9조건을 분리한다. CPU 실제 충돌 쌍의 손/비손 접촉과 비손 첫 접촉 이전 최대각을 기록하며 다른 에이전트의 낙상으로 시험을 자르지 않는다. YAML/bank/checkpoint는 수정하지 않았다.
- 최종27조건·유한값·무외력 대조군 거의0도·trace수·Python/셸 문법·diff 확인 통과. 현재강화 외력20N은 최대52.2/최종42.1도,40N은97.4/97.2도다. 참조0.633초/닫힌 문은 비손 접촉 이전 최대각 기존15.2/중간5.7/현재3.1도 및 첫 비손 접촉0.87/0.67/0.43초였다. 현재강화 참조6조건 모두90도 미도달, 세 저항18조건 모두 담당자 낙상으로 종료했다. 따라서 저항이 밀기를 방해한다는 근거는 있으나 저항만 줄이면 해결된다는 결론·학습 정책의 손 힘/5초 지지 능력은 미검증이다. 보고서·JSON·CSV·그래프 `output/open_reference_push_check/`; config/structure 반영. 기존 학습은 유지했다.

### 로컬 VNC 실행 지침 고정·뷰어 버퍼 2배 확대

- 다른 Codex가 viewer 명령을 찾을 때 참고하도록 AGENTS·structure·config 상단에서 config의 `local-vnc-viewer` 절로 연결했다. OPEN VNC 명령·1280×720·로컬 conda·Wayland 수정·로그 경로·checkpoint 선택 기준을 모았다.
- 사용자 요청으로 저메모리 옵션의 접촉 버퍼1048576→2097152·배율2→4로 각각2배 확대했다. 실제 VRAM 사용량의2배를 보장하지 않으며 학습/16환경 초과 차단은 유지한다. GPU0 OPEN1000checkpoint·1환경30step VNC6099 평가 정상 종료(`output/open_toy_strong_close_low_vram_double_check`); Python/셸 문법·문서 링크/경로·diff 확인 통과. 기존 학습/뷰어는 재시작하지 않았다.

### Wayland 데스크톱의 VNC 시작 실패 수정

- `run-gui.sh`의 x11vnc에만 `WAYLAND_DISPLAY`·`XDG_SESSION_TYPE`을 제거했다. 명시한 Xvfb 화면 대신 상속 Wayland 세션을 감지해 즉시 종료하던 원인을 실제 non-quiet 로그로 확인했다. config/structure에 반영했다.
- Xvfb/x11vnc/noVNC6089 시작·정리 확인, GPU0 OPEN1000checkpoint·저메모리 중간 설정·1280×720·1환경30step VNC 평가 정상 종료(`output/open_toy_strong_close_low_vram_vnc_check`). shell 문법·diff 통과. 기존 학습은 유지했다.

### 학습 병행 뷰어의 PhysX VRAM 절약 옵션

- `TOKENHSI_VIEWER_LOW_VRAM=1`로 평가16환경 이하에서만 PhysX 접촉 버퍼8388608→1048576·배율10→2를 적용한다. 사용자 요청대로 완만한 축소를 선택했으며 학습 사용은 차단한다. config/structure에 로컬·VNC 사용법을 기록했다.
- 기본값 보존·축소 값·학습/환경수 차단 검사와 Python 문법·diff 확인 통과. GPU0 OPEN1000checkpoint·1환경30step headless 복원 평가 완료(`output/open_toy_strong_close_low_vram_moderate_check`). 더 작은 임시 설정의 GUI 초기화 VRAM404MiB를 확인했지만 :0 X 연결 종료가 남았고 VNC는 x11vnc 시작 실패로 검증하지 못했다. 최종 중간 설정의 GUI 성공·VRAM 사용량은 미검증이며 기존 학습은 유지했다.

### OPEN 초기 개방 저항·자동 닫힘 강화

- 가볍게 건드려도 문이 열리는 현상을 줄이도록 `open_toy_strong_close` 전용 env/train YAML·train/test/VNC·RSI bank를 추가했다. 강성2→16·감쇠0.5→4·최대 토크4→32N·m, 닫힌 위치에서도 preload4N·m를 적용한다. 선택적 preload 검증/토크를 공통 커널에 추가하고 미설정 시 기존 식을 유지한다. 기존 OPEN/PUSH 설정과 프로세스는 변경하지 않았으며 새 출력/물리·RSI 계약으로 분리했다. config/structure에 실행법과 검증 한계를 기록했다.
- CPU67개 통과. 실제 문 asset CPU PhysX8조건에서90도 놓기 후10도 미만까지1.93→0.63초, 개방 각속도3rad/s는1.98→0.70초. 닫힌 문에2N·m/0.15초 입력 시 최대 개방5.28도→거의0도,20N·m/0.15초는52.80도→9.97도(`output/open_toy_strong_close_physics_check/audit.json`). GPU1 비학습2환경 RSI OPEN13/13·PUSH145/171·1000reset/partial reset/AMP 통과(`output/open_toy_strong_close_rsi_check`). GPU1·2048환경·MAX_ITERATIONS2에서3epoch/checkpoint 저장 및2환경/30step headless 복원 평가 통과(`output/open_toy_strong_close_{check,eval_check}`). wrapper3개 문법·diff 확인. 본학습·강화 설정 GUI/VNC·정상 손 지지5초의 장기 정책 성공은 미검증이다.

### 로컬 OPEN/PUSH 손 조작 보상·몸 접촉 차단

- 사용자 요청으로 `open_toy`·`push_toy`에 선택적 `manipulationContact`를 적용했다. RSI 시작 진행도는 유지하고 이후 같은 physics tick의 유효 손 접촉·몸 접촉 없음 구간에서 얻은 새 진행만 인정한다. 몸으로 먼저 진행한 뒤 손만 대는 소급 인정과 반복 전진 이력 누적을 차단했다. OPEN은 개방 보상과5초 지지, PUSH는 양손 전진 자격/완료를 연결하며 PUSH 완료 후 손을 떼어도 목표 안정 보상을 지급한다. 손 거리 항의 성공 최대값 덮어쓰기는 제거했다. 담당 사람의13개 비손 body 형상 근접/수평 반력 proxy 및 이동 중/완료 후 접촉 진단을 추가했다. 충돌 pair·힘 기여도 측정은 아니다.
- 수정 전 checkpoint와 의미가 다른 관측 예약 슬롯/보상 계약을 분리하고 기본 output을 `output/{open,push}_toy_hand_contact`로 바꿨다. 두 설정은 최대9000회·2048환경 유지, 각각 실행별 독립 Actor/Critic/D/optimizer/replay다. 기존 공동 YAML과 SIT/ON_TOP/CLIMB는 변경하지 않았다. config/structure에 실행법·보상·진단·한계를 기록했다.
- CPU54개 통과(기존42+접촉/진행12): 손→몸 밀기·몸→손 재접촉·후반 RSI 유지·완료 후 손 떼기·OPEN 몸 지지·부분 reset·형상 회전·바닥 수직 반력 배제 검사. RTX3090 GPU0·2048환경·MAX_ITERATIONS2에서 각각3epoch/checkpoint 저장(`output/{open,push}_toy_hand_contact_check`), 각 checkpoint2환경/30step headless 복원 통과(`output/{open,push}_toy_hand_contact_eval_check`). 새 접촉37개 태그×3epoch 유한값·담당131072/반대0 표본 및 이전 계약과의 차이 확인(`output/toy_hand_contact_tensorboard_audit.json`). wrapper6개 문법·YAML/diff 확인. 장기 손 사용 개선·실제 접촉 proxy 정확도·GUI/VNC는 미검증이며 기존 실행 프로세스는 변경하지 않았다.

### 토이 학습 기본 최대 반복 수 조정

- 사용자 요청으로 OPEN/PUSH 토이 train YAML의 `max_epochs`를2→9000으로 올리고 실행 문서의 기본 명령을 맞췄다. 두 YAML 값과 diff 확인 완료. 학습은 새로 실행하지 않았다.

### 로컬 OPEN/PUSH 분리 토이 학습

- 서버의 공동 후반 RSI 설정을 기준으로 `open_toy`·`push_toy` 전용 env/train YAML과 train/test/VNC를 추가했다. 사람2명 모두 한 과제만 학습하도록 graph 확률과 AMP 명목 전문가 비율을 함께 조정하고 checkpoint/output을 분리했다. 물리·보상·후반 RSI·모델 구조는 유지한다. 로컬 wrapper는 GPU6 제한 없이 기본GPU0·활성 conda 또는 `tokenhsi_koo`를 사용하고 저장소 상대 경로로 실행한다. 기본 학습 길이는 최대9000회이며 `MAX_ITERATIONS`로 덮어쓸 수 있다. config/structure에 실행법과 checkpoint 계약을 기록했다.
- CPU42개 통과(기존38+단독4): 단독 과제 reset·조건부 AMP 라벨/반대 전문가 배제·복사된 RSI bank의 로컬 모션 hash 계약 확인. RTX3090 GPU0·2048환경·MAX_ITERATIONS2에서 각각3epoch 완료 및 checkpoint 저장(`output/{open,push}_toy_check`). TensorBoard에서 매 epoch 담당 과제131072표본·반대 과제0표본 확인. 각 checkpoint2환경·30step headless 평가 복원 통과(`output/{open,push}_toy_eval_check`). wrapper6개 문법·diff 확인. GUI/VNC 실제 실행·장기 정책 성공은 미검증이며 기존 프로세스는 변경하지 않았다.

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
