# 로컬 실행 변경

## 2026-10-28

### 최종 safe-start·장애물 이격 설정 GPU5 본학습

사용자 승인으로해당native초기뷰PID3832983만종료하고19500베이스fresh Stage2를GPU5 MPS에시작했다. hard18cm·balanced각1/3·PUSH-only·단독±45°/공동0°·인당20kg·1024환경을유지하고모든agent초기0.7m/몸체겹침검사와주변박스8~12m/접근선·sweep여유1.5m를적용했다. tmux`jointpush_hardgate_safe19500`,PID3851733,output`output/train_base19500_hard_safe_clear_gpu5_20261028_214039/Stage2JointPushRsi_28-21-40-42/`.

원본554파일해시·19500실제전이·두adapter설치·실제2epoch encoder/RMS동결·coordination gradient/업데이트·startupweight를확인했다. CPU보상분리미러PID3854708과TensorBoard6016새run`current_ksh_jointpush_base19500_hard_safe`를연결하고기존14k비교2개를유지했다. MPS서버와기존다른클라이언트는중단하지않았다.

### 주변 랜덤 박스와 agent 접근선·상자 경로 이격

사용자 요청으로로컬`ksh_distractor_clearance.py`를추가했다. 주변4개박스를환경원점반경8~12m에배치하고, 모든agent초기위치→담당미는면바깥1m접근점선분과박스외곽여유1.5m·물리상자sweep/다른물체여유1.5m를검사한다. 실패를최대256회재추첨하고미해결시reset재시도로전달한다. 새학습기본clear_paths/구checkpoint view기본legacy와별도workspace/계약을추가했다.

CPU5tests·원본554파일해시·shell구문통과. 실제CPU64장면에서전체4개박스의반경/접근선/sweep이격assert와agent초기안전성·8step추론을통과했고초기낙상종료0건이다. 외부source·보상·RSI비율은유지했다. 해당native정지뷰만교체해GPU5·6086에서새배치를로딩했다.

### GPU5 균등 세 상태 정지 초기 view

사용자 요청으로GPU5 native·6086·tmux`jointpush_initial_33`에safe-start/all_coupled/balanced 초기자세view를띄웠다. N은전원LOCO/전원PUSH/혼합을각1/3추첨하며정책/rollout은진행하지않는다. 초기화렌더동기화1마이크로초1회후root/DOF5frame불변과PNG를확인했고첫화면은push2/loco2였다. 결과`output_etc/check/initial_view_equal33_20261028_211853/`.

balanced view에서R/L확률1/0을sampler의0.5계약검증에보내지않고전원PUSH/LOCO강제선택으로처리하도록로컬bootstrap을수정했다. controls표시도equal-thirds로정정했다. 외부source와MPS서버/다른작업은유지했다.

### 모든 agent 쌍의 최종 초기 배치 안전 검사

사용자 승인으로로컬`ksh_initialization_safety.py`에최종rootXY최소0.7m·모든쌍몸체covering-sphere겹침검사를추가했다. loco-loco와다른그룹도검사하며실패환경의human reference/배치를재추첨한다. 기존그룹·구역·RSI선택과보상은유지한다. loco-only는human도재추첨하는Mixin reset경로로연결했다. 새학습기본all_pairs,기존view기본legacy;새workspace_safe_start/checkpoint계약으로분리했고실행중PID3306277에는반영하지않았다.

CPU11tests·원본554파일해시·shell/Python구문통과. 실제CPU4환경에서balanced160장면(최소간격0.790m)+loco-only64장면(최소0.978m)reset·각8step추론을검증했다. 모든쌍0.7m미만/몸체검사실패0건·초기낙상종료0건. Stage1 19500전이모델로초기배치를검사했으며장기성능/GPU본학습검증은아니다. 결과는`output_etc/check/initialization_safe_start{,_loco}_19500/summary.json`이다. 외부ksh와원본source는수정하지않았다.

### RSI 실제 초기 배치 CPU 진단

사용자 문의로 현재19500 기반hard/balanced 설정과1000epoch 모델을CPU4환경에서진단했다. 초기48장면에서드문root근접을발견했고추가160장면에서7장면의agent간XY간격0.7m미만(최소0.254m)을재현했다. loco-loco4쌍·push/loco3쌍이며root와참조body위치는최대3.8e-6m차이로buffer불일치가아니다. 현배치검사는같은그룹이며한명이라도pushRSI인경우만몸체겹침을검사하고,모든agent쌍의root최소간격을요구하지않는다.

반면전원push인공동50그룹은모두초기gate열림·양손면거리약6cm였다. 각8step(약0.27초)에서낙상종료0/160이며장기안정성검증은아니다. 진단은 `output_etc/check/initialization_audit_19500_v3/{summary.json,results/initialization_audit.json}`에보존했다. 기존학습/환경/보상은바꾸지않았다.

### Contact progress·분할 할당 검증 및 로그 미러 갱신 복구

사용자 문의로 현재Contact progress가가중치0.2×1/(1+max(몸통XY−담당구역중심거리−0.5,0))임을확인했다. 물리상자할당이그룹간중복없고그룹내공유인지·구역순위유일성·n분할폭/중심을무작위1만장면CPU에서검증했다. 코드변경/학습중단없음.

추가한별도eventwriter가원본summaries에동시기록하면서TensorBoard가더이상원본파일을tail하지않는문제를발견했다. 원본지표가79epoch에멈춘동안파생태그만갱신되고있었다. 파생event파일만제거하고CPU변환기가원본scalar전체와분리보상을별도단일-writer `derived_summaries/`에복사하도록수정했다. 현재TensorBoard run은미러를가리키며원본event·학습·optimizer는그대로다. 미러parser에서205개태그·512epoch이상기록을확인했다.

### 보상 태그 그룹 통합

사용자 요청으로 분리보상 태그를 `reward_terms/at_progress`, `reward_terms/contact_progress`처럼같은reward_terms그룹의평면이름으로변경했다. CPU파생로그만재생성하고TensorBoard캐시를갱신해새8개태그와구중첩태그제거를확인했다. 학습과원본event는유지했다.

### 실행 중 Contact / AT TensorBoard 보상 분리

사용자 요청으로 CPU event변환기 `scripts/ksh_reward_split.py`를 추가했다. 현재PUSH-only agent별CONTACT+AT각1개 구성에서 기존전체항−AT항으로CONTACT를 산출하고 state/progress/success/total8개태그를같은summaries에기록한다. 원래가중치·gate·목표포화가반영된agent-step평균이며AMP/패널티는포함하지않는다. 전체과제항과3개component합계일치·CONTACT음수없음을각기록에서검증한다.

학습PID3306277을재시작하지않고GPU없는CPUwatcher3324286을연결했다. 실제59개step변환과TensorBoard8개태그등장을확인했다.30초마다새step을추가하고학습종료시끝난다. 원본source/학습/보상/optimizer는변경하지않았다.

### Stage1 19500 기반 GPU5 재학습·TensorBoard 표시 정리

사용자 지정19500 체크포인트(sha256 `4eb295a8…`)를 받아 Stage1 전이 경로를 선택 가능하게 하고 GPU5/6을 지원하도록 로컬 launcher를 확장했다. 기본14000은 유지하며 custom base는 파일명/해시를 workspace에 반영해 설정 충돌을 막는다. 기존 checkpoint와 model이름·shape/관계metadata/Stage1물성계약이 같음을 CPU에서 확인했고 관련CPU8tests·원본554파일 해시·shell구문 통과.

GPU5 기존MPS서버4104165에 PID3306277을 연결해4명·1024환경·hard18cm·balancedRSI·목표포화로19500부터 fresh Stage2를 시작했다. 실제source_epoch19500 전이·epoch2 encoder/RMS동결·coordination gradient/갱신·startupweight 저장 확인. 기존MPS클라이언트3103831은 유지했다. 실행은 `output/train_base19500_balanced_hardgate_gpu5_20261028_165533/`이며 tmux `jointpush_hardgate_19500`다.

TensorBoard6016에서 이전KSH와5000iter미만 실행은 표시만 제거했다. 새`current_ksh_jointpush_base19500_hard`와기존14k비교2개만 남은 것을API로확인했고 원시weight/log는 보존했다. 이전설정은 해당 launch의`tensorboard_before_cleanup.json`에 남겼다.

### 18cm hard gate + balanced RSI 본학습

사용자 요청으로 전원 양손이 미는 면에서18cm 이내일 때 AT state/progress를 여는 hard gate adapter를 추가했다. CONTACT shaping과 목표 위치 도달 시 gate 면제·CONTACT/AT 포화는 유지한다. 기존 balanced soft 설정과 gate 계약·workspace 경로 외 설정은 같으며, RSI 전원push/혼합/전원loco 각1/3·solo50:50, PUSH-only·단독±45°/공동0°·물성·1024환경을 유지한다. 기존 checkpoint와 분리한 hard workspace/출력에 Stage1 14000부터 Stage2 fresh optimizer로 시작했다.

CPU12tests·설정 동등성·원본554파일 해시·shell 구문 통과. GPU6 MPS PID3260458, tmux `jointpush_hardgate_balanced`에서 실제2epoch 업데이트·encoder/RMS동결·coordination gradient/갱신을 확인했다. output=`output/train_balanced_hardgate_20261028_162947/Stage2JointPushRsi_28-16-29-51`; epoch2 startup weight 저장. TensorBoard6016에 `current_ksh_jointpush_rsi_hard`를 추가하고 이전soft6000도 유지했다. 다른 MPS작업은 유지했다.

### 학습 중단 후 실제 출력 디렉터리 전환

사용자 요청으로 balanced 학습 PID1872208만 공식 MPS terminate_client(rc0)로 종료했다. runs의 실제 output/output_etc를 이 프로젝트로 이동하고 역방향 심링크와 구 runs 출력 경로를 제거했다. 생성 설정·launch 기록은 유지한다. 새 학습/평가의 출력 검증 root도 이 프로젝트로 수정하고 현재 실행 metadata를 stopped·새 경로로 갱신했다.

weight28개 inode/크기 보존, 원본554파일 해시, balanced 설정 일치, Python·shell 구문, 학습/평가 출력 경로 검증 통과. TensorBoard6016만 새 경로로 재연결해 최종6074epoch 로그를 확인했다. 마지막 숫자weight는6000epoch이며 다른 MPS클라이언트2개와 서버는 유지했다. 새 학습·GPU검증은 실행하지 않았다. 이동 기록: `../runs/ksh_output_relocation_20261028.json`.

### 최상위 프로젝트 분리 및 결과 경로 통합

사용자 요청으로 frozen source·전용 adapter/scripts/tests를 `CVPR2027/ksh_jointpush/`에 배치했다. 기존 실행 경로는 원자적 교환 및 호환 링크로 유지하고 현재 학습 결과는 `output/`, 확인 결과는 `output_etc/check/`에서 접근한다. 원본554파일 해시·기존balanced 설정 일치·CPU19tests·이동안전성4tests 통과. 새 경로의2환경 CPU reset/정책/부분reset도rc0·optimizer0으로 통과했다. 학습/supervisor/TensorBoard를 중단·재시작하지 않았다. [실행 안내](EXECUTION.md), [전체 이동 기록](../docs/OUTPUT_LAYOUT.md).
