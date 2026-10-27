# Changelog

## 2026-10-27

### 사용자 요청으로 GPU 자원 실행 가드 제거

- 새 KSH launcher의 50GiB 여유·Python client 수 차단만 제거했다. GPU6 제한·MPS 연결 검증은 유지한다. shell 구문을 확인했으며 학습은 실행하지 않았다.

### PUSH 목표를 상자 정면에 직선 정렬

- 사용자 요청으로 Stage1 및 Stage2 단독/공동 목표를 상자 로컬+Y로 생성한다. ±45도 목표편차를 없애고 거리2~10m·agent출발위치/그룹구성/PUSH-only/softgate/RSI는유지한다. 새workspace_push_only_aligned에분리해실행중인기존workspace_push_only학습은수정/종료하지않았다.
- CPU회귀29개와실제CPU4환경 단독/공동/랜덤그룹RSI각16step·부분reset의각도0검사를통과했다. 원본/snapshot은보존한다.

### 목표 도달 성공에서 손 gate 제거 및 CONTACT/AT 완전 포화

- 사용자 요청으로soft adapter의AT성공hard-gate조건을제거했다. 위치성공원래기준20cm/Z1mm를유지하고도달전AT state/progress만soft gate를적용한다. 도달시원래paired처리로CONTACT/AT각state/progress/success최대값과유효gate1을보장한다. 목표를벗어나면포화가해제된다. power/collision은유지한다.
- workspace_goal45_goal_success와box_goal_position 계약/코드SHA로기존설정을보존·분리했다. CPU26tests(접촉0/gate0도달보상1.2,도달전soft shaping,목표이탈/높이/부분reset)와실제CPU RSI두환경각16step/부분reset을통과했다. 원본/snapshot은수정하지않고학습/뷰어중단을유지했다.

### 정지 viewer의 초기 GPU 렌더 동기화

- reset tensor만쓴상태에서상자/관절그래픽이구pose에남는문제를확인했다. reset후중력/속도0·dt1e-6 동기화1회로변환을반영하고이후물리진행없이정지렌더링한다. 정책step은0이며기록의materialization단계와정지중physics step을구분한다.

### RSI 초기자세 고정 viewer

- 사용자 정정에 따라rollout view만종료하고초기reset자세만렌더링하는평가전용모드를추가했다. policy/env_step/simulate호출없이카메라조작과N(50/50새자세),R(push),L(loco)를지원한다. 학습은중단상태를유지하고원본/snapshot은수정하지않았다.

### PUSH RSI 검사용 viewer50/50

- 사용자 요청으로새goal45환경/Stage1베이스14000의공동4명view를GPU6 native/MPS우회로실행했다. loco50/pushRSI50은그룹단위·에피소드별샘플링이며평가전용PUSH_RSI_MIX로분리한다. 기존100%강제RSI view만교체했고학습은중단상태를유지한다.

### Stage1 및 Stage2 단독 PUSH의 목표 ±45도 재샘플링

- 사용자 지적에 따라 Stage1 sample_layout와 Stage2 단독 sample_push의 후보승인에 상자+Y 대비goal 수평각±45도검사를추가했다. 공동기존cone/배치·거리·물성·RSI는유지한다. snapshot원문함수를로컬adapter에분리복사하고거절조건만추가했으며원본/snapshot은수정하지않았다.
- 생성설정/launcher를workspace_goal45로분리해이전config를보존한다. Stage1수정config도생성하고bootstrap에서Stage1/Stage2모두지원한다. 각도/adapter SHA를checkpoint계약에포함했다.
- CPU24tests와실제CPU2환경의공동/단독PUSH RSI각16step·부분reset후목표각도검사를통과했다. 학습은중단상태를유지했다.

### GPU6에서 새 KSH RSI+soft gate 본학습 시작

- 사용자 승인으로 기존Partial PID3873224만 MPS terminate_client 성공0으로 중단했다. 타학습769426·MPS서버2946883는유지한다. 이전weight/로그를보존했다.
- 새 Stage1PushFinetune14000에서 epoch0/새optimizer,4명×1024환경,인당20kg/마찰0.4,원본혼합과제·RSI와로컬평균soft gate로시작했다. GPU6전용MPS연결을검증했다.
- 실제coordination gradient401.784·2epoch후actor/RMS불변/coord업데이트/optimizer195state·startup저장을확인했다. TensorBoard current_ksh_jointpush_rsi_soft와 latest_partial_training.json을갱신했다. 새실행은 runs/jointpush/ksh_jointpush_rsi/output/soft_rsi_gpu6_20261027_162444.

### 새 KSH RSI 실험의 shaping gate를 soft로 변경

- 사용자 요청으로 bootstrap 로컬adapter를추가했다. 두손중최대유한면거리 q=exp(-(d/.18)^2)의그룹평균을 AT state/progress에곱한다. 전원hard조건은 최종성공/성공보너스에만유지하고CONTACT·단독과기존성공포화를유지한다. 원본ksh/snapshot·현재학습은변경하지않는다.
- 생성config에gate설정과adapter SHA를포함해기존hard-gate Stage2계약과분리했다. CPU회귀2개로부분접촉shaping/전원성공/CONTACT·단독불변을확인하고,실제CPU2환경공동/단독RSI 각16step과부분reset을통과했다. 새본학습은시작하지않았다.

### 새 KSH Stage2 jointpush RSI 독립 실행 준비

- 사용자 요청으로 ksh stage1_2_jointpush commit5e30a72의554일반파일·13읽기링크를 독립snapshot으로 보존했다. 원본checkout/수정/GPU실행/학습교체는 하지 않았다. 제공최신Stage1PushFinetune14000을로컬고정하고10500회귀용도보존한다.
- prepare/train/view/CPUcheck bootstrap을추가했다. 생성manifest에서상대경로130모션을절대경로로정규화하고소스는그대로유지한다. 캐시는복사하고모든출력은runs로돌린다. GPU6 index/UUID검증·학습메모리가드·MPS검증·nativeGUI우회를연결했다.
- CPU20tests,실제CPU2환경all_coupled/push 각16step·14000복원/원본RSI관절·고정목표·부분reset을통과했다. 별도최신14000 전이/coord gradient·업데이트/actor동결CPU검증도완료했다. 새GPU학습/GUI와전체mixed RSI캐시생성은실행하지않았다. 명령과한계는 KSH_JOINTPUSH.md.

### 질량만 낮춘11500 viewer

- 사용자 요청으로 같은11500 checkpoint의 task상자 질량만 인당15~25→10~15kg으로 변경하는 test/eval전용 PARTIAL_VIEW_MASS_PER_AGENT 옵션을 추가했다. 원래 균등 난수를 재사용해 크기·배치·그룹·ordinary질량·마찰·checkpoint 계약을 보존하며 실제물성 차이를 physics_override.json에 명시한다. 학습사용은 거부한다.
- CPU1환경2step strict복원/finite관측·정책·실제actor질량 확인을 통과했다. GPU6 nativeviewer만 교체했고 학습/MPS는 유지한다. 별도 view workspace는 runs/jointpush/view11500_mass10to15_20261027_113411/workspace다.

## 2026-10-26

### Viewer N키로 다음 에피소드

- 사용자 요청으로 Partial viewer에 N키를 등록했다. render에서는 요청만 저장하고 다음 physics step의 정상reset 계산 후 env0을 done처리하므로 네agent/공동상자/goal이 기존 reset경로로 함께 초기화된다. success를 강제하지 않으며 학습/headless에는 적용하지 않는다. 기존 V/L/F/K/Esc 입력과 카메라/그래픽 동작을 유지한다.
- AST실행프로브로 N입력→deferred reset·1회소비·다른env/headless불변·기존K키를 확인했다. MPS를 사용하지 않는 native viewer만 재시작해 적용했다.

### 이전 baseline36000 Partial5000 resume 및 TensorBoard 연결

- 사용자 요청으로 RSI uniform실험만 공식 MPS terminate_client 성공0으로 중단하고 이전baseline36000/1024환경×4명/인당15~25kg/상자전체마찰0.3 설정의 최신5000 pth를 resume했다. 원본설정을 새 workspace로 복사하며 모델·optimizer195state·epoch5000/frame655360000의 정확한복원을 검증한다.
- 기존 current_base36000_e1024 TensorBoard 이름을 유지한다. 원본event는 보존하고 checkpoint지점 이전기록425154개를 새연속로그로 복사, 미저장12936개를 제외한다. frame/iter축은 복원step으로 이어지고 time축은 기존max offset을 더한다. CPU전용TensorBoard6016만 캐시갱신을 위해 재시작했다.
- runtime복원감사와 resume후2epoch actor/RMS동결·coord업데이트 검증을 추가했다. MPS서버2946883·타클라이언트3349106은 유지한다. 경로는 runs/jointpush/latest_partial_training.json.

### RSI uniform11500 baseline 및 밑판 전용 마찰

- 사용자 요청으로 jhh RSI uniform 실험의 최신 numbered checkpoint11500을 CPU확인·로컬복사·strict Stage1계약검사했다. 원본/기존결과는 보존하며 새Stage2 epoch0/1024환경×4명/인당15~25kg를 유지한다.
- 로컬 Partial에 한 rigid body·두 충돌box URDF를 선택적으로 추가했다. 본체0.9·밑판1cm0.3 마찰, 전체 cuboid 질량/관성/크기/COM 및 기존actor/body인덱싱을 보존한다. surface_mode가 없는 구 workspace의 물리는 그대로이며 새checkpoint계약에는 surface설정을 포함한다. prepare 기본baseline을RSI uniform으로 갱신했다.
- CPU4환경8step에서 strict165tensor전이/104coord추가·gradient37.9167·actor/RMS불변·유한관측/action/물리·질량/두shape마찰을 확인했다. MPS terminate_client로 기존PID2861739만 종료(성공0)했고 다른클라이언트3349106·서버2946883를 유지했다. GPU6 새실행에서 coord gradient399.312와2epoch actor/RMS동결·coord업데이트·startup저장을 확인했다. TensorBoard current_uniform11500_bottom03을 연결하고 최신경로/PID를 latest_partial_training.json에 기록했다.

### 원본과 전체 agent 샘플 수 일치

- 사용자 요청으로 Partial 기본 env를 2048에서1024로 줄였다. 4명×1024=4096 rows로 원본2명×2048과 같으며 horizon32/minibatch16384는 유지한다. 기존 workspace는 보존하고 baseline36000에서 새 optimizer/epoch0로 다시 시작한다.
- GPU6 MPS 공식 terminate_client 2946883 1996870의 성공코드0을 확인한 뒤 기존 학습만 종료됐다. 다른 클라이언트922451와 MPS 서버2946883는 유지한다. 인당15~25kg·reward·배정·출발 설정과 jhh 원본은 변경하지 않았다. 새 GPU6 PID2861739에서 coordination gradient457.97, 2epoch 후 actor/RMS동결·coord업데이트·startup저장을 확인했다. TensorBoard current_base36000_e1024로 연결했다.

## 2026-10-25

### 최신 baseline36000에서 저질량 Partial 파인튜닝 새 시작

- 사용자 정정에 따라 이전Partial14000을 resume하지 않고 jhh stage1_v2_fixed_push_360_raw_cuboid의 최신36000 baseline에서 새Stage2를 시작했다. numbered/일반checkpoint 모두epoch36000/frame4718592000을CPU확인하고 로컬복사·원본contract검사를 통과했다. 인당질량은15~25kg,4명/2048환경·기존reward/gate/후방180/랜덤배정을 유지한다.
- 별도 workspace raw_cuboid_partial_base36000_mass15to25_20261025, GPU6 전용MPS/새tmux·새optimizer·identitycoord/epoch0다. CPU4환경8step strict165tensor/새104coord/gradient38.996·actor/RMS불변과 GPU첫coord gradient405.44를 확인했다. jhh/이전weight·결과·다른작업/MPS daemon은 수정하지 않았다. 최신실행메타데이터는 runs/jointpush/latest_partial_training.json.

## 2026-10-23

### TensorBoard 현재/이전 Partial 비교

- 사용자 요청으로6016전용TensorBoard만 비교모드로 재시작했다. 현재 RawCuboid rear180과 이전Koo Partial summaries를 별도run으로 연결하며 event원본/학습은 수정하지 않았다. 비교경로와서버설정은 runs/jointpush/tensorboard_raw_cuboid, tmux tensorboard_partial_comparison이다.

### 현재 GPU6 Partial TensorBoard 제공

- 사용자 요청으로 현재 실행의 summaries를 읽는 전용TensorBoard를 localhost6016/tmux tensorboard_partial_gpu6에 실행했다. GPU는 사용하지 않고 기존서버/학습/MPS는 유지한다. HTTP scalar API에서85태그와 접촉/group gate/AT성공/goal거리/보상 표시를 확인했다. 설정과로그는 runs/jointpush/tensorboard_raw_cuboid에 기록했다.

### GPU6 rear180 Partial 본학습 실행

- 사용자 요청으로 실행 중이던 자체 viewer만 종료하고 GPU6에서4명/2048환경 Partial Stage2를 시작했다. RawCuboid22500 Stage1 base, 후방180°/그룹·구역 랜덤 배정, 인당20~30kg 및 실제 공동상자 AT progress/양손group gate를 유지한다. 기존GPU6 타 평가·GPU5 학습과 MPS daemon은 유지한다.
- GPU6 전용 MPS pipe 자동검증·strict165tensor전이/새104tensor·coord gradient387.86·2epoch 후 actor/RMS불변/coord업데이트 및 startup 저장을 확인했다. 충돌은 enable=true, root XY 최근접거리0.7m 미만에서 coefficient0.5 선형페널티다. 최신 세션/실험/weight경로는 runs/jointpush/latest_partial_training.json 참조.

### Partial 인당 질량20~30kg 복원

- 사용자 요청으로 GPU6 zero-shot viewer3050304만 종료했다. 새 prepare와 준비된 rear180 workspace의 partitionPush.mass_per_agent를20~30kg로 변경했다. 그룹 n명에 대해 실제 질량n×U(20,30)kg이며1/2/3/4인 각각20~30/40~60/60~90/80~120kg다. baseline checkpoint/원본jhh 물성·마찰·배정·출발은 변경하지 않는다.

### GPU6 native zero-shot viewer 장치 매핑 수정

- 사용자 invalid device ordinal 보고의 원인은 단일visible CUDA0과 물리번호를 사용한 player(cuda5)의 불일치였다. 기존 jhh runtime은 GUI에서도 MPS/Mesa 경로를 다시 활성화하므로 local launcher에서 native GUI helper와 MPS 탐색 제외를 적용하고, unfiltered CUDA/Vulkan 순서의 물리index를 sim/RL/player/current CUDA에 모두 지정했다. 원본 snapshot/jhh 및 학습 경로는 수정하지 않았다.
- GPU6 기존 Python client가 없는 것을 확인한 뒤1환경 zero-shot viewer를 실행했다. CUDA7장 공개/sim·RL·policy cuda6/Vulkan6, 환경변수상 MPS pipe/log 없음과 checkpoint strict transfer·rollout·noVNC6086을 확인했다. GPU5 기존학습3729650/MPS4104165와 GPU6 MPS2946883은 유지한다. 기록 runs/jointpush/partial_zeroshot_gpu6_20261023_153601. 새 본학습은 실행하지 않았다.

### Goal 반대쪽180도 출발·중복 없는 랜덤 배정

- 사용자 요청으로 위치를 실제 상자→goal 반대 방향 중심±90°에서 샘플링한다. 기존 중심반경1.65~6m·상자 외접원+0.45m clearance·사람 최소0.9m·독립heading은 유지한다.
- 거리 기반 구역 최적화·최근접 상자 확인/재샘플링을 제거했다. 랜덤 그룹에 물리 상자와 중복 없는 랜덤 station을 먼저 배정하고, 같은 그룹의 구역은 균등 랜덤 순열로 겹치지 않게 배정한다. 공동 상자 공유만 허용하며 다른 그룹의 할당은 분리한다. 구역은 reset 때만 재배정한다. partitionPush.version4/별도 rear180 workspace로 이전reset checkpoint와 구분한다.
- CPU13회귀 통과:2048샘플 후방180°/clearance/비중첩·4096그룹 중복검사·랜덤 구역24순열 확인. 실제CPU32환경24step/finite·4인8토큰 strict복원·반원/그룹 내 고정구역 assertion 통과. 학습·MPS/외부jhh는 변경하지 않았다.

### RawCuboid baseline 기준 Partial Joint Push 재구성

- 사용자 지정 jhh의 stage1_v2_fixed_push_360_raw_cuboid/22500회 checkpoint를 로컬에 복사해 고정했다. 원본230파일과 checkpoint hash 불변을 확인했다. 새 partial은 로컬 source snapshot·geometry·scene·reward·실행 wrapper로 독립했으며 외부 및 stage2_jointpush 코드를 수정하지 않는다.
- RawCuboid midpoint/1.04m/실제 힘과 물리tick OR·navigation·0.2×contact/AT 보상·AMP33/67·물성15~20kg/마찰0.3/크기분포를 유지한다. 그룹 인원만큼 X·질량을 확장하고 모든 구성원의 양손이 실제 goal 쪽으로 미는 뒷면에 있어야 AT state/progress/success gate가 열린다. AT progress는 분할 token이 아닌 실제 공동 상자 중심과 공통 goal로 계산한다. 접근/contact는 gate 이전에도 학습된다. 같은 그룹90/10 공유, baseline 선형 충돌을 사용한다.
- CPU9회귀·4/32환경24물리step·strict transfer165개/새coord104개·coordination gradient L1=33.059/optimizer update·actor/RMS불변 통과. 검증 결과는 runs/jointpush/raw_cuboid_partial_20261023. 새 GPU/PPO/GUI/본학습은 실행하지 않았다.
- 기존 koo_partial_* 및 koo_1021_* 전용 코드·복사본17항목을 제거하고 runs/jointpush/retired_koo_partial_20261023.tar.gz로 보관했다. 기존 weight·결과와 과거Native분석은 보존했다. 이전 partial 실행 중인 프로세스가 없음을 확인했다.

## 2026-10-22

### 7000iter 무개입 4인 viewer 명령

- `view_push7000_4.sh`와실행진입점을추가했다. 평가와동일한local4인검증/스폰hook을재사용하기위해five_task_scaling의실행부만main guard로감쌌다. 원본player loop,4명8박스/PUSH_AT/loco시작/원본bias를유지한다. GUI CUDA·Vulkan·정책device를사용자가지정한GPU로맞추고첫action후4인overview를설정한다.
- shell문법/help·Python컴파일을확인했다. 사용자요청은명령제공이므로GPU/GUI는시작하지않았다. 실제실행시viewer_verification.json에장치/인원/박스/무개입여부를기록한다.

### PUSH100 7000iter 체크포인트 4명 확장 평가

- 사용자 제공 depth200_20kg_friction04_push100_from1000의7000iter/frame917504000을확인하고저장env/sim/train을runs/jointpush/five_task_push7000_scaling_20261022/workspace로복사했다. 원본2명4박스→4명8박스(담당4+여분4)/4goal의기존local확장만사용했다. model은task_shared_embedding_action/GTA/64D4층2head이며추가bias/mask/학습없음.
- 기존500평가를보존하고epoch가동적인five_task_scaling.py·전용wrapper/집계기를추가했다. taskpacket·관측1268D·counts[4,8,4]·DOF128·actor20·배정·20kg/마찰0.4를assert검증했다. CPU2장면24step→GPU5기존MPS4장면60step→64장면600step/seed2141을순차완료했다.
- 4인낙상9/64(14.0625%),손접촉22/256(8.59375%),원본AT목표도달0/256. 평균상자이동0.1343m이나목표방향순진척0.00735m,중앙이동0.01187m. 4인구동은가능하지만PUSH수행확장은충분하지않다. 사용자기존선호대로2인평가는생략하여원래수행부족과확장저하를분리할수없다. 단일seed/firstscene종료검열을명시했다.
- runtime.done의ATrequired와contact동시만족을별도로계측했다. result/initial/256agent CSV/summary PNG/REPORT/검증JSON을위runs하위에저장했다. 외부Python184파일·checkpoint·savedconfig hash불변,가중치/RMS불변·finite통과. GPU5메모리21817MiB반환·기존MPS4104165유지확인.


## 2026-10-21

### KSH / Koo NONE relation 평균 및 2→4인 정보 유입 비교

- 실제directed taskedge와SELF를제외해NONE을정의했다. Koo H→담당B/B→담당G, KSH Sit/Climb H→담당B만task로분류하고 같은owner의간접/역방향NONE과다른owner항목을구분했다. human/box/goal query·3×3physicaltype·layer/head별bias와attentionmass,자기box task−남의box NONE margin을저장했다. float64집계로반복동일값의float32평균오차를피했다.
- 박스수=agent수인2명/4명 각각32초기장면·seed2141·원래biasedactor/no intervention. CPU로Koo2/KSH2/KSH4를추가추출하고Koo4는기존32장면canonical자료재사용. 외부ksh142파일hash/모든가중치RMS불변을확인했고GPU/MPS는쓰지않았다. source별학습task/배치차이는통제되지않았음을명시했다.
- 4명,4층평균전체NONEbias Koo-4.082/KSH-4.327; human만은-4.533/-7.270. 마지막human foreign mass Koo2인0.752%→4인6.612%,KSH0.173%→0.516%. 같은모델의type별NONEbias는2/4인에서동일했다. 마지막층HB task−NONE gap은5.808/9.307. KSH말층B/G querybias0을포함한전체평균의한계를보고서에설명했다.
- runs/jointpush/attn_view/none_relation_comparison_20261021에3개비교PNG,전체layer/head/type CSV,summary/verification JSON,REPORT 저장. attentionmass는V/GTA/잔차를포함한실제action기여도와동일하지않으며학습과정의단독원인으로확정하지않는다.

### KSH Sit/Climb 2명·2박스 raw attention

- 이전4인 비교와 같은 stage1_task Distill39000 checkpoint로 2human/2box/2goal(6토큰), H0=SIT/H1=CLIMB의 CPU추출 wrapper를 추가했다. 원본지원counts를사용하고task/size sampler만process-local로고정하며원본소스·가중치·설정은바꾸지않는다.
- CPU4장면/seed42/loco시작/300step에서0/30/60/120/180/299의24sample을저장했다. 원래actor/GTA/bias 유지, masking/음수bias개입없음. canonical probe와원래shuffle action오차3.58e-7, 층출력재구성0, softmax재구성1.79e-7, weights/RMS불변, 외부Python142파일hash불변확인. GPU/MPS미사용.
- runs/jointpush/attn_view/ksh_sitclimb2_boxes2_raw_20261021에 rawNPZ(6,4,4,2,6,6), 상태/검증JSON,24개rawQK·bias·실제attention복합PNG, 초기/120step rawQK전용2PNG, index갤러리/README를저장했다. QK는bias와softmax전GTA QK/sqrt(d), 후반입력에는앞층bias효과가반영됨을표시했다.

### 전체층 -4의 4인 물리 viewer

- `view4_bias.sh`와 `bias_view.py`로 평가와 같은 BiasCorrection을 checkpoint 복원 이후 원본 player/viewer loop에 연결했다. 모든 H/B/G query의 다른 owner key에 actor1~4층 -4를 적용하며 hard mask는 아니다. 첫 action의 실제 보정/층출력을 검사하고 hooks를 계속 유지한다. 기본 native4_view는 플래그 없으면 기존동작이다.
- GPU5 CUDA·Vulkan/4명4박스4목표/seed2141/loco시작/포트6086에서 실행. 첫action 검증 actor·physics 모두cuda:5, 층 출력 재구성0, bias오차4.77e-7, hard_mask=false. 초기카메라가박스내부를향해첫reset후전체4인/박스범위의overview로조정했고자동agent-follow를해제했다. 최종검증json·화면캡처는 runs/jointpush/y2m/view4_bias_all4_overview_20261021. 기존MPS4104165와 외부 원본 유지. viewer는 사용자 확인을 위해 계속 실행한다.

### 층별 all-owner ablation 및 전체층 hard mask

- 선택층 API에 first2/layer1~4/1-based 쉼표목록과 λ8/12를 추가했다. CPU 초반2층12·1층단독4 검증을 통과했다. 사용자 요청으로 완료한2명4조건을 보존하고 남은2명 평가는 생략,4명11조건(기준/초반4·6·8·12/각층단독4/후반4/전체4)을 GPU5 기존MPS·64장면/seed2141/최대600step으로 완료했다. 결과·비교PNG·11개실제attention PNG·층별foreign attention 곡선은 runs/jointpush/attn_view/native_y2m_layer_bias_ablation_20261021에 저장했다.
- 단일층4는낙상57~63/64. 초반2층4/6/8/12는51/46/47/46낙상·목표12.50/13.28/13.28/11.72%. 후반4는49/64·8.20%, 전체4는2/64·80.47%재현. 초기2장면에서초반12의foreign attention은앞층거의0이나후반다시증가했다. 전체64장면 attention통계와구분한다.
- 사용자 후속 요청으로 JOINTPUSH_BIAS_HARD_MASK=1을 추가했다. 선택 cross-owner bias를-inf로 대체하고 같은owner값불변·차단확률정확히0·실제층출력재구성을검증했다. 0*inf를피하고softmax유한값을검사한다. CPU검증 후GPU5동일4명조건의전체층hard mask는낙상2/64(3.125%),목표200/256(78.125%),전원도달27/64. 원본63/64·1/256,전체층-4는2/64·206/256. 학습/체크포인트는바꾸지않았다.
- hard mask 산출물은 runs/jointpush/attn_view/native_y2m_allowner_hardmask_20261021. 서로다른box를미는task별추론분리효과이며협력학습의증거로해석하지않는다. 두비교모두초기hash일치·weights/RMS불변·원본preflight무결성검증통과. 평가후GPU5메모리21817MiB로반환·MPS4104165유지확인.

### 모든 human·box·goal query의 cross-owner 음수 bias 평가

- 사용자 범위 확장에 맞춰 H_i/B_i/G_i 모든 query에서 자기 owner 토큰3개를 제외한 모든 key에 -λ를 한 번 더하는 all_owner scope를 추가했다. 같은 owner 기존bias, critic, checkpoint/RMS는 유지한다. 원래 token shuffle을 보존하며 canonical owner별9m(m-1)개 보정 항목·나머지 항목 불변을 검증한다. human-only 중간 배치는 사용자 변경에 따라 전용 runner/평가만 종료하고 interrupted.json을 남겼다. 다른 작업/MPS는 종료하지 않았다.
- CPU4인24step/1000agent reset·readout검사 통과. GPU5 기존MPS에서64장면/seed2141/최대600step, 후반2층·전체4층×λ2/4/6×2명/4명+대조군=14조건 완료. 같은 인원수 초기hash 일치, model/RMS 전부불변, 층 출력 재구성 오차0, bias 오차최대4.77e-7. 원본768파일은 각 preflight검증 통과.
- 4인 원본 낙상63/64·목표1/256 → 전체층 -4에서2/64·206/256(80.47%), 전원도달31/64. 같은2인은2/64·98/128→2/64·107/128. 후반2층 -4는49/64·8.20%, 전체층 -6은3/64·79.30%. 다른 owner 정보 전달의 개입효과가 크지만 특정 query/층의 단독원인이나 독립seed 성능은 검증하지 않았다.
- runs/jointpush/attn_view/native_y2m_allowner_bias_20261021에 summary CSV/비교PNG/14개전체층attention PNG/rawNPZ/초기foreign attention CSV/REPORT를 저장했다. 전용 평가 프로세스 종료·기존MPS4104165 유지 확인. 다른 사용 작업의GPU점유는 그대로 유지했다.

### 5과제 depth180 1000iter 원본2인 viewer

- 최신 재실행 폴더의 relation_config를 runs/jointpush/five_task_view_depth180_1000_20261021로 복사하고 외부 소스·데이터·체크포인트는 읽기 전용으로 사용하는 로컬 viewer 진입점을 추가했다. 출력 resolver만 로컬 workspace로 지정한다. 학습/환경/관계 bias는 수정하지 않는다.
- 2명·박스4개/PUSH_AT/loco 평가시작/seed2141, GPU5 CUDA·Vulkan을 명시한다. 원본 player의 device_name 기본값도5로 지정해 다른GPU 정책 할당을 방지한다. 기존MPS 서비스와 다른작업을 유지한다.

### Native Y2m 후반층 owner bias 보정 20조건 평가

- 원본/체크포인트 수정 없이 actor3·4층에서 Hi→다른 owner의 사람·박스·목표 key에만 -λ(2/4/6)를 더하는 로컬 계측을 추가했다. 자기 key, 앞 두 층, 다른 query, critic, 가중치/RMS를 유지하고 원래 token shuffle을 보존한다. raw QK/원래 bias/적용 bias/실제 attention을 canonical 순서로 저장한다.
- CPU λ0는 기존4인 결과·궤적과 완전 일치, GPU smoke 통과. GPU5 기존MPS에서2명/4명 각각 기준+9보정=20조건,64장면/seed2141/최대600step을 순차 완료했다. 같은 인원수 초기state hash 일치, 모든 model/RMS 불변, 층 출력 재구성 오차0, bias 오차 최대4.77e-7을 확인했다.
- 4인 기준 낙상63/64·목표1/256 → 남의 목표 -6에서48/64·19/256. 같은2인 기준2/64·98/128 →2/64·102/128. 사람/박스만 보정은4인61~64/64낙상. 목표 경로 개입 효과가 있으나 여전히75%낙상이며 단일seed 탐색 결과다. 앞층/간접경로까지 검증한 것은 아니다.
- runs/jointpush/attn_view/native_y2m_bias_correction_20261021에 전체 CSV/비교PNG/조건별20개 attention PNG/REPORT를 저장했다. 평가 후GPU5 메모리21817MiB로 반환, 기존MPS4104165 유지. 원본768파일은 각 조건 preflight 검증을 통과했다.

### 기존원자료에서자기/남의박스QK·bias margin 분해

- 추가정책/물리실행없이agentwise초기32장면의각Hi에대해자기Bi와다른Bj(j!=i)의score차를QK/bias별로분해했다. log(p_own/p_foreign)=QK차+bias차를각항목수치검증해softmax분모영향과구분했다.
- 첫층QK margin은거의0. Push/HOLDING+AT 첫층bias margin은5.870/5.925로비슷하지만마지막층은5.808/10.981로차이가컸다. Sit/Climb 마지막층8.378/10.235. 마지막층QK margin은Push+.367,HOLDING+AT-.782로오히려Push가더자기박스를선호하므로직접box attention분리차이는후반learned bias가중요한후보로보인다. 단독인과로판정하지않으며다른human/goal/V기여는별도다.
- runs/jointpush/attn_view/agentwise_attention_20261021/logit_decomposition에summary/margins CSV/한국어REPORT/층별비교PNG를저장했다. 평균margin의exp를평균확률비로해석하지않는주의를명시했다. GPU/MPS/원본미사용·미수정.

### 두5과제500iter모델의원래2인안정성·낙상검사

- 사용자제공koo_cvpr/.../ksh_stage1_stage2_base/push의두500checkpoint(epoch500/frame65536000)를CPU에서확인하고저장relation_config에서env/sim/train을local workspace로복사했다. 깊이1.8m20kg마찰.6와2m20kg마찰.4. 외부원본184Python파일/두checkpoint SHA를보존·재검증했다. 원본output 제한은process-local로local workspace/output_etc/check에적용해원본쓰기없음.
- CPU4env/12step에서2인634D/4인1268D와두물성·allPUSH_AT packet·실제20kg/friction·finite·strictrestore를검증했다. 원본Stage1이2인/4box만허용하므로4명8box는count검증/계약count와2인전용PUSHreset만로컬에서일반화했다. 원래정사각형4모서리를source로사용하고extra4box는base reset을유지했다. GPU5기존MPS4env/60step후각64장면/최대600step을순차실행했다.
- 우선PUSH_AT전용2인/4인네조건은모두64/64장면낙상. 사용자추가지시로원래2명/박스4개/다섯과제혼합baseline을두모델각64장면·loco-only원본test초기화로검사했다. 두모델모두64/64낙상,첫낙상중앙값1.100s/1.133s. PUSH없는43장면도각43/43낙상. 4명확장판정은baseline미확보로보류하며500iter자체를원인으로단정하지않는다.
- 원본발제외접촉+.15m기본body별높이종료식과최초scene종료검열사용. 전조건weights/RMS불변/finite,원본2인actor10/DOF64·확장4인actor20/DOF128·globalactorID범위/중복없음확인. GPU메모리21817MiB복귀/MPS4104165생존,다른학습/데몬미종료.
- runs/jointpush/five_task_500_fall_20261021에REPORT/summary.csv/all_results.json/firstfall곡선PNG/원본copy설정/source_manifest/검증자료·6조건로그를저장했다.실행스크립트five_task_fall_prepare.py/five_task_fall.py/five_task_fall.sh·순차실행run_five_task_fall.py·집계report_five_task_fall.py.

### 에이전트 개인 소유 기준 attention 재집계

- 사용자 지적으로기존2명group단위비중을개인소유기준으로교정했다. 각Hi의자기human/다른human/자기담당box/다른agent담당box/비대상box/자기goal/다른goal7개분류를계산해합1검사,logical앞4box와physical담당assignment일치를검사했다. 같은pair안의다른agent도남의것으로포함한다.
- Push4box,KSH Sit/Climb8box,구HOLDING+AT8box의feature실험과같은seed2141·초기32장면을CPU로재수집했다. 초기상태JSON이앞선feature실험과정확히같음을확인했다. 원본forward유지·층재구성오차0·가중치/RMS불변. GPU/MPS와KSH원본미수정.
- 최종층/2head/32장면평균Hi→Bj(j!=i):Push H0..3=1.228/1.090/1.415/1.127%,SitClimb=.034/.023/.033/.025%,HOLDING+AT=.054/.055/.070/.028%. Push에서다른human평균2.012%,다른goal3.385%로,다른box평균1.215%보다크다. 기존12.6/.29/.21%는다른샘플/2명group기준이라이개인별통계와혼용하면안된다.
- runs/jointpush/attn_view/agentwise_attention_20261021에원자료/개인owner별4x4그림/카테고리별그림/전체층PNG/agent-layer-head CSV/summary/REPORT를저장했다. 모델간태스크·배치·물체수·학습이력차이는남으며확률만으로V크기나낙상인과는판정하지않는다.

### 구버전 HOLDING+AT unified6000의2명→4명 feature 비교

- 사용자 요청으로KSH stage1_fix의ApproachScenarioStage1Unified_00006000을원본수정없이CPU에서32장면·300step실행했다. 기존물리재현의HOLDING+AT그래프확장을process-local로재사용하되2인counterfactual의graph compile도지원했다. 4명/8box/4goal에서2인/4box/2goal을추출하고semantic edge는owner로골라src/dst/owner index만remap했다. edge순서가shuffle되어도대응한다.
- 초기32장면×4명에서최종feature상대L2 3.02%,cos .999421,action상대L2 2.37%,RMS .004900. tokenizer입력차0, 모든시점4인pair-mask와2인계산feature최대차1.67e-6/action5.96e-7, 가중치/RMS불변확인. CPU추론/물리만사용했고GPU/MPS와원본KSH는건드리지않았다.
- 구버전HOLDING+AT는Push와같은edge_mlp인데도feature변화가작다. 따라서edge_mlp라는bias구조만으로Push확장실패를설명할수없다. 모델간태스크/배치/학습이력교란은남는다. 같은물리상태는모델내2명vs4명비교에적용되며서로다른모델의장면까지동일하다는뜻은아니다.
- 원자료와15PNG는 runs/jointpush/attn_view/ksh_holding_at_feature_2vs4_20261021. Push/Sit/Climb/HOLDING+AT 통합PNG/summary/REPORT는 runs/jointpush/attn_view/push_sit_climb_holding_at_feature_comparison_20261021에저장했다.

### KSH Sit/Climb의2명→4명 feature 변화와Push대조

- 사용자 요청으로KSH stage1_task 39000 distill 모델을read-only로실행하고Sit/Climb/Sit/Climb의32장면을CPU에서비교했다. 원본2명/4box에맞춰4명/8box/4goal의같은관측을pair별2인/4box/2goal로추출했다. human/물체/goal/GTApose는그대로유지하고taskpacket actor/target index만remap했다. extra logical4/5를첫pair,6/7을둘째pair로정했다.
- tokenizer동일human입력차0, 모든시점pair-mask4인과2인계산의feature최대차2.03e-6/action최대차4.92e-7(수치정밀도내). strict 모델복원, 가중치/RMS불변·유한값 확인. 원본KSH코드수정없음/GPU·MPS사용없음.
- 초기32장면의 task별비교: Sit 최종feature상대L2 1.87%,cos .999787,action상대L2 1.63%,actionRMS .003490. Climb 각각2.66%,.998530,2.19%,.009215. 앞선Push는41.22%,.885337,32.74%,.075839. KSH는encoder출력부터안정적이며큰feature변화를head만흡수한결과는아니다. 단일seed/다른task·layout·물체수·학습이력교란으로bias방식이나distillation의단독인과를확정하지않는다.
- KSH원자료/층별PNG는 runs/jointpush/attn_view/ksh_sitclimb_feature_2vs4_20261021, task별비교PNG/summary/한국어REPORT는 runs/jointpush/attn_view/push_vs_sitclimb_feature_comparison_20261021. 초기값주비교, 시간샘플은원래4인궤적의최초장면종료전만사용. 같은장면의agent·시점을독립표본으로해석하지않는다.

### 같은 상태의2명·4명 human feature 비교

- 사용자 요청으로 CPU PhysX32장면에서4명/4box/4goal unmasked 궤적을 추적하고 초기/15/30/60/120/180step의 최초종료 전128장면·시점(512agent·시점)을 저장했다. 각4인관측을같은사람·박스·목표·pose의2인pair×2로분리하고graph index만remap했다. 동일 가중치/RMS의2인encoder,4인fullencoder,4인pair-mask를비교한다. 비대상박스는추가하지않아2→4인context 차이를측정한다.
- tokenizer및각Transformer층의64Dhuman feature와같은head의raw action mean 저장. 초기32장면×4명에서cosine은tokenizer1.0→L1 .9745→L2 .9409→L3 .9006→L4 .8853. 평균relativeL2는0→.2211→.3164→.3885→.4122. 최종feature RMS차 .47139. action RMS차 .07584/최대절대차 .70622, DOF별절대차 .1초과비율10.89%.
- 모든샘플에서동일human tokenizer입력차0,4인pair-mask와2인pair의모든층feature/action차0. 원본 가중치/RMS불변·유한값·문법확인. 별도rollout끼리비교한것이아니며차이자체로낙상인과나head단독문제를판정할수없다. GPU/MPS미사용.
- runs/jointpush/attn_view/native_y2m_feature_2vs4_20261021에NPZ/상태/수치요약/한국어REPORT와비교PNG1장+샘플PNG6장/index를저장했다. 64D를8x8로표시한그림은공간구조가아니다.

### 원본2명 Native Y2m attention PNG 및 담당박스 차단

- 사용자의 우선 요청에 따라 원본2인/2박스/2goal/6token/570D를 mask없이 CPU에서4장면·최대300step 캡처했다. 원본6000 strict복원과가중치/RMS불변, 실제층출력재구성오차0, token순서정렬 전후 action최대차1.79e-7 확인. 최초비교forward의RNG를복구해추가shuffle소비를막았다. attention_cpu.sh의 PROBE_AGENTS=2 진입과렌더러의동적인원표시를지원한다. 19유효샘플 PNG는 runs/jointpush/attn_view/native_y2m_2agent_boxes2_20261021에저장한다.
- 앞선4인 담당박스 요청은 H_i→B_j(j!=i)12셀을모든층/head에서차단하는두조건(비대상표시/차단)을 GPU5/MPS에서64장면/최대600step으로완료했다. 기존초기상태hash와동일. 비대상표시64/64낙상,비대상차단63/64낙상,목표도달둘다0. 대조군2개와함께 runs/jointpush/attn_view/native_y2m_owner_box_ablation_20261021에저장한다. 직접mask만으로다른토큰을거친간접전파를막지는않는다.
- 추가로에이전트별4개encoder pass에서자기박스외의담당박스key를모든query에서차단하고자기human행만읽는 strict owner모드를구현했다. CPU2env/24step과1000agent reset·확률0·층재구성오차0만확인했으며, 2명map우선요청시점의이모드GPU평가는미실행이다. perspective별PNG원자료를따로저장한다.

### 비대상 박스 mask ablation + 실제 masked attention PNG

- 사용자 요청으로4인/8box에서 비대상4개(X0..X3)의 모든16subset을 순차 비교했다. process-local actor층 hook으로 선택 key열만 -inf 처리하며 모든query/head/layer에 적용한다. 실제 물리8box와 담당4box/goal/보상/critic/가중치/RMS는 유지한다. 원래 random token shuffle을 forward당1회 유지하고 그림 저장 시 canonical 순서로 되돌린다.
- CPU2env/24step에서 mask-none 계측과 원래 evaluator의 전체 지표·traces가 정확히 같음을 확인했다. all-extra mask도 CPU에서 확률0·층연산 재구성 오차0. GPU5 기존MPS 4env/60step smoke 통과 후64env/최대600step/seed2141의16조건을 실행했다. 모든조건의 초기physics+obs hash동일, 가중치/RMS 불변, masked probability와 층출력 재구성 최대오차0.
- 결과: X0..X3 모두 mask는 장면낙상63/64(98.4375%), 나머지15조건64/64(100%). 모든조건 agent목표도달0. 단일seed의 같은64장면 반복이며 조합을 독립표본으로 해석하지 않는다. 첫scene종료에서 동료를 검열한다. 비대상박스만 가려서는 큰 개선이 없으며 담당박스/다른사람 등의 원인을 판정한 실험은 아니다.
- runs/jointpush/attn_view/native_y2m_boxes8_mask_ablation_20261021에 raw NPZ/상태/로그/결과와 첫2장면의 종료 전84샘플 PNG를 저장한다. PNG는4층×2head의 raw QK/sqrt(d), 원래bias, 실제masked attention이며 막힌열은 빗금 처리한다. 조건간 공통logit색상범위를 사용한다. 비교 mask_progression.png/갤러리/REPORT도 생성한다. 렌더러의 확률재구성검사에도 mask를 반영하도록 수정했다.
- GPU평가 종료 후 GPU5메모리21817MiB로 반환, 기존MPS서버4104165 생존 확인. 다른 프로세스/원본소스는 수정·종료하지 않았다.

### 4명·8박스 attention PNG

- CPU PhysX/원본6000 가중치로4장면을 추적하고 종료 전10유효 장면·시점의 raw QK/sqrt(d), bias, 최종attention을 수집했다. 토큰 순서를 정렬한 별도 관측 probe이며 실제 행동은 원래 player.get_action을 유지한다. 정렬 전후 action 최대차 약9.93e-7, 층 출력 재구성 오차0, 가중치/RMS 불변 확인. probe의 최초 비교 forward는 shuffle RNG를 추가 소비하므로 기존 rollout의 bitwise 재현은 주장하지 않는다.
- 각 샘플4층×2head의 PNG와 index.html을 runs/jointpush/attn_view/native_y2m_4agent_boxes8_20261021에 저장했다. H=human/B=담당박스/X=비대상박스/G=goal. QK와bias는 전체샘플 공통색상범위, 확률은0~1. GPU/MPS는 사용하지 않았다.

### 4명·박스8개 view 설정

- 로컬 평가 확장에서 JOINTPUSH_BOXES=8을 지원하고 view4_boxes8.sh를 추가했다. 대상4개와 비대상4개를 실제 물리 객체·관측으로 포함한다. 기존4박스 기본값과 원본 소스는 유지한다.
- 16토큰/1288D, 대상4개의 중복 없는 할당·목표4개·8개 task edge. 추가 상자는 각 그룹 바깥쪽4m에 배치한다. 진행량 버퍼는 대상4개 기준으로 분리했다.
- CPU 체크포인트 strict 복원, 1000agent reset, 부분reset 격리, 12actor/128DOF/68body, 전체상자20kg, 24step 유한값·가중치/RMS 불변 확인. GPU/MPS와 GUI는 실행하지 않았다.

### Native Y2m 4명·상자4개 평가 확장

- 사용자 요청으로 원본2인 sampler 두개를 X±10m로 배치하는4인/4box/4goal 단일PhysX task를 추가했다. 원본 물리/크기6종/20kg/30D물체·bias·정책가중치는 유지하고 기반 클래스 물체수/packet폭/배정/강체·DOF tensor를 일반화했다. 원본 소스는 수정하지 않고 local install만 적용한다. runtime 보상은 원래2인 계산을 두pair에 적용하며 loco 초기화 평가만 지원한다.
- CPU1000agent reset·중복배정 없음·부분reset 격리·상자20kg·8actor/128DOF/64body·1140D/12token strict 복원/24step finite 통과. 동일관측 pair-mask4인 encoder/readout과 원본2인×2 feature/action 최대차0. CPU검사용 rl_games device_name도 명시적으로CPU를 사용한다.
- GPU5 기존MPS/seed2141/64장면/최대600step, mask 없이 원본가중치/RMS 불변 평가:4인 낙상63/64(98.4375%), agent목표도달1/256(0.390625%); 기존2인2/64·98/128. 4인낙상중59장면은 넘어진 agent의 이전손접촉기록 없음. 첫 scene 종료에서 동료검열/단일seed/배치확장교란을 보고서에 명시했다. `runs/jointpush/y2m/uploaded_6000_scaling_comparison_20261021/REPORT.md`.
- 4명 view wrapper도 추가했으며 shell/Python 문법만 검사했다. 4명 renderer는 실행하지 않았다. 평가프로세스 종료·MPS 생존, 원본768파일/데이터 계약 재검증 통과. 학습은 하지 않았다.

### Viewer 내부 VNC 포트 충돌 수정

- 원본 GUI wrapper가 빈 X display만 고르고 연결 VNC 포트6000 점유는 나중에 검사해 종료하던 문제를 수정했다. 원본 스냅샷은 유지하고 로컬 `scripts/run_gui.sh`에서 X socket/lock와 VNC 포트를 함께 검사한다. `view.sh`는 로컬 GUI wrapper와 원본 test script를 조합한다.
- 실제 현재 포트 점유에서 :104/6004 선택 확인, shell 문법·원본768파일/데이터 계약 무결성 재검증 통과. GPU/renderer는 실행하지 않았고 기존 서비스/MPS는 종료하지 않았다.

### 업로드6000회 모델의 2명 낙상·목표 도달 검사

- 업로드 가중치의 CPU physics/reference 계약과 Actor/Critic/Discriminator strict 복원을 통과했다. 원본 actor/RMS를 유지하는 first-episode 계측 `scripts/scaling_eval.{py,sh}`를 추가했다.
- GPU5 기존MPS, seed2141, loco 시작,64장면/최대600step에서 장면낙상2/64(3.125%), agent목표도달98/128(76.5625%), 양쪽도달장면39/64(60.9375%). 평균 초기/종료 목표거리3.484/0.641m. 낙상은54/70step이며 장면 첫 종료에서 동료를 검열한다. 네트워크/RMS 불변과 유한값 확인, 메모리 반환·MPS 유지 확인.
- 사용자 조건인 2명 무낙상을 만족하지 못해4명 실험은 실행하지 않았다. 원시 결과: `runs/jointpush/y2m/uploaded_6000_2agent_seed2141_20261021/result.json`. 단일 평가 seed이며 학습 seed 간 일반화 결과가 아니다.

### Native PUSH Y2m 재현 패키지 준비

- 사용자 요청으로 `juan` 브랜치의 `jointpush/`에 Koo `c3b71cf` 스냅샷과 source hash manifest를 보존하고 준비/CPU검사/평가/VNC/resume wrapper를 추가했다. 원본26개 motion 경로는 내용이 일치하는 외부 데이터를 읽는 심링크로 연결하며 YAML/bank/asset/외부소스를 수정하지 않는다.
- checkpoint 기본경로는 `jointpush/checkpoints/PushNativePhaseY2m_00006000.pth`, 임의 경로 인자/환경변수도 지원한다. 가중치를 복사하지 않았으며 부재 시 GPU 초기화 전 종료한다. 생성물은 `runs/jointpush/y2m/`에만 저장한다.
- 준비 검사: 원본768파일, asset/mesh/PUSH 모션 및949후보 bank 계약 일치. CPU 전용으로1000 agent reset/양방향 배정/부분reset/AMP/24 physics step·570D/6토큰 Actor/Critic/Discriminator finite 검사 통과. shell/Python 문법과 checkpoint 부재 조기종료 확인. `runs/jointpush/y2m/preparation.json`, `runs/jointpush/y2m/cpu_check_20261021_151555/audit.json`.
- GPU·MPS 연산·학습·viewer는 시작하지 않았다. 업로드 checkpoint의 strict 복원과 GPU 실행은 아직 확인하지 않았다.
