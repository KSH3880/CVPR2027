## 2026-10-28

### runs의 미사용 실험·로그 압축 보관

사용자 요청으로 실행 프로세스·코드·심링크 의존성이 없는 과거 실험을 output_etc/archive/runs에 보관했다. 이 프로젝트 실험 폴더 28개를 포함해 전체81폴더·140개별파일을 tar 원본 대조·압축본 SHA 기록 후 runs에서 제거했다. 원시 결과와 실패 근거는 보존했으며 학습/supervisor/TensorBoard/viewer 생존과 학습 event 갱신을 확인했다. 보관 목록·복원은 ../runs/archive/cleanup_20261028/INDEX.md, 정책은 ../docs/OUTPUT_LAYOUT.md.


### 기존 Chair 본학습 결과 output 통합

runs의 Chair HOLDING_AT 본학습과 장기 bbox FT 결과를 이 프로젝트 output으로 이동했다. 소스사본·검증 기록은 유지하고 구 결과 경로는 호환 링크로 연결했다. checkpoint inode/크기·양쪽 경로 접근 검사 통과. 학습을 새로 실행하거나 중단하지 않았다. 상세 ../docs/OUTPUT_LAYOUT.md.


### Table residual10k TokenHSI 원본 환경·평가 방식 측정
사용자요청. 原본HumanoidAdaptCarryBox2Objs/환경YAML/seed·sim함수/Transformer player run_eval body를고정사본에서실행하고정책만우리10k로교체. 512환경/test9/seed0/연속3repeat/20초/AT20cm+native updot≥.86615. 성공432/438/427,전체**84.44±1.08%**(연속repeat표본std),이전Chair집계처럼0회제외하면**84.47±1.52%**. 성공최소거리오차7.09/6.91/6.91cm. native→우리280D관측기존생성기bitwise exact·171tensor/RMS strict/평가전후동결,4환경3repeat smoke·metric회귀3개·정식3trace/precision/599step/첫episode재채점·원본core/ckpt해시통과. 첫시도절대에셋경로전달실수로asset None/actor-1/첫reset CUDA오류→상대경로수정·actor검증추가,실패기록보존. 성공경로에별도동기화/수명변경없음. Chair3831871유지. ../runs/table_residual_10k_tokenhsi_native_20261028/REPORT.md 및validation.json. 공개코드조건이며논문10×10m와달라논문83.6%재현과구분;30초위치-only평가와도구분.

### Table residual4k–10k 연속 평가 완료
사용자요청4000부터1000간격10000까지7개checkpoint/GPU0·test9/512×독립seed0·1·2/30초AT20cm첫episode ever. 평균±평가seed표본std:4k80.21±4.24%,5k81.25±1.28%,6k85.94±2.93%,7k86.39±2.46%,8k88.41±0.69%,9k87.63±1.96%,10k90.89±3.03%. 기존완료10k재채점재사용·4k–9k18seed신규실행. 모든checkpoint frozen153/scene RMS원본78k exact·finite,21seed trace/precision/hash·초기전배열10k및이전2200과exact·통계검산통과. Chair3831871평가후에도유지. CSV/PNG/SVG/validation.json 및../runs/table_residual_eval_4k_10k_20261028/REPORT.md.

### Chair 예약 경로 오류 복구 및 본학습 시작
상태조회로Table10k/rc0완료·Chair예약04:45 preflight경로검사rc2/PPO미진입확인. 원인은고정code checkout의형제출력경로;원본실패보존후새recovery에서code/output_etc/check/chair_preflight/train으로수정,등록전실제resolver호출추가. 옛경로거부/새경로허용포함회귀4개및모의검증통과. 실제4096환경2PPO rc0/epoch2·frozen153/sceneRMS exact·trainable18·양gradient통과후11:08:33에fresh78k/GPU0/4096/seed42/총10k본학습시작(PID3831871,supervisor3831386). 사용자Table평가미종료. ../runs/chair_residual_after_table_20261028_recovery/REPORT.md 및preflight_validation.json;현재state는chair_residual_active_run.json.

## 2026-10-27

### Table 종료 후 Chair HOLDING_AT residual 예약 활성화
사용자실행예약요청. Table3648228·기존supervisor3637486유지,독립예약supervisor3685235가Table10k정상종료marker/번호checkpoint/PID종료→GPU0여유23GiB확인→Chair4096환경2PPO·frozen153/sceneRMS/양gradient검증→fresh78k에서총10k/save100본학습을자동수행하도록등록. seed42·TokenHSI보상/PPO·encoder/action head/sigma/scene RMS동결·residual/valuehead/AMP학습18tensor유지. 코드/78k/설정사본·해시고정,PIDstarttime/lock/launch marker로중복방지,선행실패시미시작·실패시무재시도. CPU모의predecessor성공/실패/누락/PID재사용/검증→본학습/중복방지통과;실제Chair는아직대기. 상태chair_residual_active_run.json,산출물../runs/chair_residual_after_table_20261027/ 및reference/chair_adapt_residual.md.

### 公開 TokenHSI Table pth 공통 AT 평가
사용자요청GPU0·기존Table학습유지,배포ckpt_stage2_objShape_table.pth(내부epoch40000)/base40k의원본265D관측·42Dcarry·RMS·102tensor strict복원으로test9/512×seed0·1·2/30초/AT20cm첫episode ever평가. 성공476/454/475, **91.47±2.43%**(평가seed표본std). 우리residual2200의66.28±1.18%와전seed초기전배열bitwise동일. fusedSDP가RTX5090에서FATAL→사용자승인으로오류worker3674693만중단·실패기록보존,math attention재실행및각seed8개CPU/GPU action최대오차4.2e-7검증. 전seedrc0/trace재채점/32·64ever/해시·통계/metrics10개통과. 논문83.6%·10k재현과구분. Table3648228유지·Chair미실행. evaluator public-checkpoint옵션/공유원본policy/문서추가. ../runs/tokenhsi_table_common_eval_20261027/REPORT.md 및validation.json.

### Chair carry residual 학습 준비 (본학습 미시작)
Table residual과동일한78k fresh·encoder/action head/sigma/scene RMS동결·zero action residual·value head/fresh AMP학습·4096환경·원본carry보상/PPO·총10k/save100을Chair에추가. native Chair정면reset/train63/test27실제에셋사용,carry RSI/tokenizer추가없음. 전용train/test/vnc/eval및설정/구조문서추가. evaluator의adapter이름보존·GPU/환경override·구모션manifest없을때공용carry목록fallback지원. CPU설정동등/초기78k출력일치/2backward·원본동결/부분reset/정면·에셋검증,1환경native추론rc0,정식평가기4환경900step smoke rc0·trace/precision/hash재채점,metrics5개통과. Table3648228유지·Chair본학습미실행;GPU여유상4096 PPO미검증. 공통check_docs는기존/hwanhee하드코딩으로실패,새문서링크·Python/shell구문통과. reference/chair_adapt_residual.md 및../runs/chair_adapt_residual_prepare_20261027/output_etc/check/REPORT.md.

### Residual1200회 공통평가
사용자요청GPU0·학습유지,1200회고정·test9/512×독립seed0·1·2/30초/AT20cm첫episode ever. 성공267/259/266명, **51.56±0.85%**(평가seed표본std),800회44.27%대비+7.29%p. 세seed초기전배열800회와bitwise동일. 동결153tensor/sceneRMS 원본78k exact,전seedrc0/trace재채점/32·64ever/해시·통계검산통과. ../runs/table_residual_eval1200_20261027/REPORT.md 및validation.json.

### Residual800회 공통평가
사용자요청GPU0·학습유지,800회번호checkpoint고정·test9/512×독립seed0·1·2/30초/AT20cm첫episode ever. 성공234/212/234명, **44.27±2.48%**(평가seed표본std),zero-shot26.50%대비+17.77%p. 동결153tensor/sceneRMS 원본78k exact확인. 전seedrc0/trace재채점·32/64ever/해시·통계검산통과. 초기모션/물체동일·origin제외좌표0.05mm이내일치. ../runs/table_residual_eval800_20261027/REPORT.md 및validation.json.

### Residual만총10k로연장·tokenizer취소
사용자최신지시. 비교supervisor3634212만종료하고residual3634237유지,tokenizer미실행. 실행중max1000은유지한채정상1000저장후optimizer/모델/전체RMS/epoch/frame full-state재개→총10000을수행할별도resume_code/continuation supervisor3637486준비·실행. 같은full_experiment_name/TB6013유지,기존메타/1000저장본archive. real restore의CPU검증으로실제epoch2checkpoint 모델/RMS/optimizer18상태/frame exact확인. 미래재개때도검증기록. 시뮬레이터/replay재초기화는별도. reference/table_adaptation_compare.md 및runs/table_adaptation_compare_20261027/state.json.

### Frozen encoder residual/tokenizer 두pilot 시작
사용자요청으로Table3544421만중단(MPS미사용확인). 기존78k actor/critic encoder·action head·sigma·전체scene RMS동결,정책은zero-init additive residual 또는복사초기화별도object tokenizer만학습. critic value head/fresh discriminator공통학습. 원본carry보상/PPO계수/4096환경/seed42유지,각1000회·100회저장순차비교. CPU초기출력동등/2backward·동결확인,각4096환경2PPO rc0/저장/finite/frozen153tensor·RMS6tensor exact,각1환경추론rc0. residual본pilot시작,정상종료후tokenizer. 상세reference/table_adaptation_compare.md 및 ../runs/table_adaptation_compare_20261027/REPORT.md. 정식eval의adapter network복원지원추가.

### TokenHSI Table 설정 FT12100 공통평가
사용자요청GPU0·학습유지,최신저장12100고정복사·test9/512×독립seed0·1·2/30초/AT20cm첫episode ever. 성공9/17/13명, 2.54±0.78%(평가seed표본std). zero-shot26.50±.56%보다낮음. 전seedrc0·trace재채점/32·64ever/원본SHA/통계검산통과. 초기모션/모델동일,origin제외좌표0.05mm이내일치. ../runs/table_tokenhsi_eval_20261027_latest/REPORT.md 및validation.json.

## 2026-10-26

### RSI 제외·원본TokenHSI Table recipe로 본학습
사용자최종요청. 우리78k/30D/GTA·graph입력유지,원본7보상·종료함수AST동일/PPO·AMP계수전항목동일/4096환경/AMPscale3/에너지OFF/물체낙하종료/걷기50:50. 원본동결adaptation에대응해actor물체tokenizer만학습·critic학습·fresh판별자/AMP RMS,old actor/sigma/사람RMS동결. CPU3개·4096환경1PPO저장rc0/finite·frozen74tensor exact/학습91tensor변경/양gradient확인후GPU0 PID3544421 본학습시작·연속update확인. 원본max1m/save100유지,구학습·RSI실험미재개. 새TB6012. 상세reference/table_tokenhsi_ft.md 및 ../runs/table_tokenhsi_20261026/REPORT.md.

### HOLDING_AT Table carry RSI 적용·검증
사용자요청대로보상/30D입력/원본78k계수유지,B20 carryWith 모션양팔native32 IK·실제mesh상판아랫면지지·9frame보정AMP이력추가. 정적·2048환경0.3초물리검증통과train16/21모델37자세/test4/9모델6자세만뱅크공개. 미지원모델명시적loco fallback(요청50%,실제약38%);viewer는검증test4subset. CPU2개·2048환경1회PPO/저장/finite/양gradient·2048부분reset/history exact통과. 본학습미시작·기존학습중단유지. 상세reference/table_carry_rsi.md 및 ../runs/table_rsi_20261026/REPORT.md.

### Table viewer 상대 체크포인트 경로 수정
사용자제보: runtime_env의cd 이후realpath로프로젝트루트상대경로실패. runtime진입전realpath -e로고정하고인자검사/LD_LIBRARY_PATH초기화추가. 실제3000회파일존재·shell구문확인. GPU viewer미실행.

### Table78k zero-shot 평가 및 FT3000 비교
사용자요청GPU0/원본78k가중치·normalizer/FT는env템플릿만사용/test9/512×seed0·1·2/30초. AT점=영역ever134/134/139명, 26.50±0.56%(평가seed표본std). FT3000의0/0/0과세seed초기root·DOF·물체·modelID·목표전배열exact일치. 전trace재채점·32/64ever·원본해시·통계검산통과. 동일조건FT성능저하확인,원인미확정·학습유지. ../runs/table_hold_at_78k_zeroshot_20261026/REPORT.md 및validation.json.

### Table carry 최신 PTH 전용 평가
Chair hold_at 평가와동일한AT20cm/첫episode종료frame포함ever/30초/512×독립seed0·1·2/ddof1로evaluation/table_hold_at 및전용eval wrapper추가. Table test9·무작위yaw·학습env/모델/normalizer고정,최신번호PTH자동선택/byte동일사본/trace·해시재채점. Chair평가기준4함수AST동일·CPU5개통과. GPU0/최신3000회/4환경900step실행rc0·899시간종료/32·64ever/전trace재채점통과(성능평가아님). 정식512×3미실행,기존학습유지. runs/table_hold_at_eval_smoke_20261026/validation.json 및evaluation/table_hold_at/README.md.

### Chair 논문88.8%와공개코드측정97.85% 구분정정
사용자지적으로논문Table3/4.2.2와공개소스재대조. Chair종류·test27·공개모델은맞지만논문랜덤orientation/10×10m와공개코드front사람정렬/9×9m차이확인. 저장initial에서도정렬/±4.5m확인,기록I/O제거후eval루프AST원본동일. 97.85%를논문재현으로표현한것정정;공개checkpoint/code의현서버측정값으로한정. 수치차이원인은미확정·추가GPU실험없음. runs/tokenhsi_chair_official_20261026_160009/PAPER_AUDIT.md 및paper_audit.json.

### 原본 TokenHSI Chair 공식환경 평가
사용자후속요청으로공통MA장면대신baseline원본HumanoidAdaptCarryBox2Objs/환경YAML/player3repeat루프를독립사본에서실행. Chair10k/base40k·GPU5·test27·512환경·seed0·600step(20초),위치20cm+updot≥.86615직립성공과원본물체조기종료유지. 성공501/503/499,운영0회제외1·2평균97.85%(반복sample std0.55%p;독립seed산포아님),전체3회평균도97.85%. 3trace성공/precision독립재채점·32/64ever·599시간종료·checkpoint/source해시통과. 기록runs/tokenhsi_chair_official_20261026_160009/REPORT.md. 평가client자연종료후기존Chair3359530/PUSH2013210·MPS유지,baseline/ksh미수정.

### 原본 TokenHSI Chair carry 공통 AT 평가
사용자요청으로공개ckpt_stage2_objShape_chair.pth(epoch10000)/baseStage1(epoch40000)발견. 원본265D관측/RMS/102tensor strict복원·가중치동결·CPUbitwise동일확인후공통native AT512×seed0/1/2/test27/30초평가. GPU5만사용,기존두학습유지하이번평가한정세번째잡사용자승인. 성공500/506/504,98.31±0.60%(평가seed표본std). 전seedrc0·trace재채점·32/64ever·899시간종료·해시/통계검산통과. 이전78kzero31.64±1.86%와세seed초기root/DOF/object/modelID/goal등모든배열exact일치. 종료후MPSclient2013210/3359530만남음,학습/MPS/ksh미변경. runs/tokenhsi_chair_hold_at_20261026_prepared/REPORT.md 및validation.json.

### 原본78k Chair HOLDING+AT zero-shot 정식평가
사용자요청GPU0순차512환경×독립seed0/1/2/test27/1명1동적의자/30초/native초기분포. 원본78kbyte-identical·30D토크나이저/normalizer직접복원,ChairFT는env/sim/train템플릿만제공하도록옵션추가. AT점=영역ever171/163/152(총486/1536),31.64±1.86%(평가seed표본표준편차). 전seedrc0·trace재채점·32/64ever·899시간종료·초기root상이·각seed27종포함·해시/통계검산통과. 기존GPU5Chair3359530/PUSH2013210·MPS/ksh보존. runs/chair_hold_at_78k_zeroshot_20261026_153255/REPORT.md 및validation.json.

### Native Chair HOLDING+AT 단독 평가 정리
사용자지정docs/evaluation_single.md의AT위치≤20cm/점=영역/첫episode종료frame포함ever·30초·512×독립seed0/1/2·ddof1을evaluation/hold_at에분리. native학습1~5m/front/바닥배치유지와일반박스AT분포차이명시. 저장env/sim/30D모델고정,독립checkpoint/코드/test27사본·trace/DOF/해시/재채점기록. CPU5개통과,GPU0의4환경900step검증rc0·899시간종료/406낙상·전trace32/64재채점통과. runs/chair_hold_at_eval_smoke_20261026_152916;첫runtime pipefail실패는이전run에보존후기존runtime과동일한셸옵션으로수정. 정식512×3미실행,GPU5학습·MPS·ksh미변경.

### Chair HOLDING+AT 78k GPU5 본학습 시작
사용자실행승인. 원본동적ArmChair공유데이터train63/test27발견후runs독립코드·에셋사본준비,실제경로/1인학습허용/접촉4M/CUDA동기화정상종료보완. 2048환경1회PPO·저장·자연종료,finite/actor·critic30D각6tensor업데이트확인. 78kfresh전이전체FT/lr2e-5/20k/500저장본학습PID3359530/GPU5시작,첫update·양tokenizergradient확인. 기존PUSH2013210/MPS유지,ksh미변경. runs/chair_hold_at_78k_20261026_135444/REPORT.md 및 chair_hold_at_active_run.json.

### Chair HOLDING+AT 78k 준비 (학습 보류)
사용자요청으로1명1동적ArmChair/2048환경/원본초기·목표분포/기존HOLDING_AT보상설정유지. 78k의actor/critic30D토크나이저와전체165tensor·normalizer strict복원,35D확장없음. 전용train/test/vnc·문서추가. CPU추론/부분reset/정면/목표분포검증통과,optimizer0/GPU미사용. 원본dynamic ArmChair데이터미발견으로train은GPU연결전경로검사exit2,실제시뮬레이션미검증. 기존MPS/잡미변경. 상세 markdowns/reference/chair_hold_at_ft.md.

### 사용자요청 SIT MPS 안전중단
GPU5 SIT1887068만 MPS terminate_client4104165/1887068 성공코드0확인후프로세스종료. MPS로그Client termination succeeded,기존PUSH2013210 및서버유지,새fatal/reset없음. 마지막32k저장본보존,활성JSON stopped_by_user. 원시mps_stop_request.json은해당run에보존.

### 최신32k 우리점·영역평가
GPU5,최신32000고정복사·동일30초512×독립seed0/1/2·test26·1명1고정의자평가. 점82.23±0.34%,영역82.68±0.49%,20k95.51/95.64%보다낮음. 전trace재채점통과·rc0,초기배열비교보존,기존학습유지. runs/chair_sit_point_region_32000_20261026_130917/REPORT.md.

### SIT 23k/26k/29k 우리점·영역평가
사용자요청GPU5,기존20k와동일30초·512×seed0/1/2·test26·1명1고정의자 평가총9개rc0. 점/영역23k85.94±1.56/86.20±1.47%,26k86.98±0.79/87.17±0.69%,29k83.66±0.41/84.11±0.49%. 20k95.51/95.64%보다모두낮음. 전trace독립재채점통과,초기배열비교validation.json보존,학습및다른작업유지. runs/chair_sit_point_region_23k_26k_29k_20261026_122850/REPORT.md.

### 78k zero-shot 두 SIT 평가
사용자요청GPU5,원본30D bbox/tokenizer/normalizer그대로인78k고정평가. 원본조건10초·seed0×3repeat 점0회제외46.58±4.56%(전체47.01±3.30). 우리30초512×seed0/1/2 점53.32±0.00%,영역57.29±0.30%. 점273/273/273이나초기root배열서로다름확인. 전체trace독립재채점통과·rc0,기존학습유지. runs/chair_sit_78k_zeroshot_dual_20261026_115055/REPORT.md. catalog경로첫실패보완후공통평가재실행.

### 20k Chair SIT 기존점·영역평가
사용자요청기존FrozenChairSit/single_match평가기그대로복제,20k checkpoint만교체.GPU5,의자test26균등복원추출·각seed26종포함,512×독립seed0/1/2·30초·첫episodeever. 점95.51±0.34%,영역95.64±0.23%,성공488/491/488과489/491/489. 3trace독립재채점/목표오차0/32·64ever/해시검증통과. 10초TokenHSI반복평가와구분. runs/chair_sit_point_region_20k_20261026_113657/REPORT.md. 학습유지.

### 최신30500 Chair SIT 평가
사용자요청GPU5,학습유지하며30500고정복사·기존동일TokenHSI조건평가. 3회78.91/79.88/79.10%, 전체79.30±0.52%,0회제외79.49±0.55%;20k90.82%대비11.33%p하락. 독립trace재채점모두일치·rc0·자연종료. runs/chair_sit_tokenhsi_eval_30500_20261026_110059/REPORT.md. 기존학습/다른GPU작업유지.

## 2026-10-25

### 저장500으로 즉시 재시작
사용자즉시중단지시로1881817 SIGTERM/종료확인후20k저장본에서전체상태재개. 새run runs/tokenhsi_sit_resume20k_save500_20261025_192226, GPU5/1명1의자2048환경/무제한유지. 실제저장config save_frequency500,max_epochs=.inf와첫20001gradient확인. 다음정기저장20500. 직전미저장진행분은복원되지않음.

### Chair SIT 저장주기500 준비
사용자지시로기본train YAML save_frequency500변경. 현재프로세스는메모리내100유지하므로실행중파일은미변경, run/train_save500.yaml에무제한재개설정준비. 운영규칙상체크포인트후중단/재시작승인대기.

### Chair SIT 20k에서 무제한 재개
사용자지시로GPU5에서20k체크포인트모델/optimizer/normalizer/epoch복원, max_epochs=.inf로종료횟수해제. 1명1의자2048환경/fullFT/lr2e-5/원본SIT환경유지,100회저장. runs/tokenhsi_sit_resume20k_unlimited_20261025_191957/run.sh 및env/train YAML보존. 초기transfer는78k,실제resume20k분리(FT를transfer소스로쓴첫실패보존). 첫20001학습update및양쪽tokenizergradient비영확인. 활성JSON갱신.

### Chair SIT +7000 동일 조건 평가
사용자요청 GPU5,20k와동일코드/설정에서7k만교체. 3회77.15/80.66/81.84%, 전체79.88±2.44%, 0회제외81.25±0.83%. 20k90.82±1.38%대비9.57%p낮음. 전trace독립재채점통과·rc0. runs/chair_sit_tokenhsi_eval_7k_20261025_191048/REPORT.md. 기존GPU작업유지.

### Chair SIT 최종20000 TokenHSI 조건 평가
GPU5 본학습 자연종료/최종20000저장 확인. runs/chair_sit_tokenhsi_eval_20261025_182833 독립사본으로 test26·1명1의자·512환경·loco_sit100%·거리1~5m/360도·300step·seed0프로세스내3repeat 평가. 점ever90.04/91.80/89.84%, 3회90.56±1.08%, 운영0회제외평균90.82%. NumPy64 첫episode/종료frame 재채점전부일치·rc0. 우리정책구조유지, 공개TokenHSI정책결과아님. 상세 runs REPORT, 실행코드는독립사본에보존. MPS공통파일누락첫실패로그보존후보완. 활성학습JSON completed로갱신.

## 2026-10-24

### 1명1의자 원본TokenHSI SIT 환경으로 재시작
사용자승인으로기존GPU5 학습3450803/3450704 종료. 최종지시1명1의자에맞춰tokenhsi_sit_full_ft 추가. 원본걷기/SIT 참조50:50·의자1~5m/360도·참조의자pose·보상AST동일·10초/IET 적용. nativecache/정면필터제외. AMP YAML을걷기12+sit20으로분리하고teacher SIT family만실제샘플링(CLIMB미로딩),78k형태호환라벨3차원유지. 78k부터fullFT20kiteration GPU5. CPU13개 및2048환경1iteration rc0,loadedlib2개/AMPfinite/초기1040대1008/source78k/frozen0/gradient확인. 설정 markdowns/reference/tokenhsi_sit_ft.md, 실행 tokenhsi_sit_active_run.json. baseline/ksh/기존산출물보존.

## 2026-10-24

### Chair SIT +6100 학습 조건 일치 평가
사용자 요청으로 같은6100 weight를 저장된 학습 환경에서 재평가. 독립사본에서 train49의자/2048환경×2명/4의자/SIT100%/정면/RSI50%/원래600step 약20초 유지, 학습은 계속 실행. seed42 단일4096명 시도, 점ever33.94%·영역ever33.18%; 초기이미성공21.90%/21.97%, 초기실패자새성공15.54%/14.77%. RSI그룹점67.23%,loco그룹0.15%. AMP history오차0/finite 및 독립NumPy64 첫episode·종료frame 재채점통과. 첫tensor/dict인터페이스실패기록보존. 결과 ../runs/chair_sit_trainmatch_eval_20261024_124625/REPORT.md.

## 2026-10-24

### 原본 TokenHSI 학습/평가 설정 감사
사용자요청으로공식Stage1 train/eval YAML·reset·player를직접대조해 reference/tokenhsi_train_eval_settings.md에정리. SIT eval360도·1~5m,CLIMB1~10m,학습각interaction/loco50:50 vs eval loco전용(reference sampling자체는유지),SIT/CLIMB299step≈9.97초,IET train만,eval seed0한프로세스3repeat확인. 객체split/종류별개수,목표/보상/점성공,물리/네트워크전체원본설정JSON도 runs/tokenhsi_train_eval_settings_20261024/에보존. baseline·학습프로세스·평가코드는수정하지않음.

## 2026-10-24

### Chair SIT full FT +6100 단독 평가
사용자요청으로78000시작FT의추가6100회정기저장본을고정복사해1명1의자·test26·512환경×seed0/1/2·loco/RSI없음·30초점/영역ever평가. 두지표모두1/1536,0.07±0.11%. 전3trace재채점통과,학습과actor/taskrole/selfobs/scene정규화동일성점검. 학습2명/4의자/정면RSI와단독평가조건차이명시; 학습유지. 상세 ../runs/chair_sit_ft_eval_20261024_122404/REPORT.md.

## 2026-10-23

### 78k chair SIT 전용 전체 모델 FT
사용자요청으로78000weight·native_front_foot RSI 기반 chair SIT100% 전용환경추가. 2명×2048환경·4물체모두의자, CLIMB/Box/Table목표제외. 기존혼합모드는유지하고sit_only opt-in만SIT확률1·chair fraction1 허용. 학습random_scenario계약유지. 전체층/정규화학습,lr2e-5,20kiteration. 검증1iteration rc0: SIT4096/CLIMB0,RSI누락0·AMP오차0/finite,source78k·frozen0 및gradient확인. 초기잘못된preset의실패로그는보존. 관련CPU10개 통과. GPU5 본학습 연속update/fps 및 source78000·SIT4096/CLIMB0·frozen0·native RSI active2063 확인. 본학습실행정보 chair_sit_only_active_run.json, 상세 markdowns/reference/chair_sit_only_ft.md.

## 2026-10-23

### IK 없이 기존 RSI로 78k 본학습 시작

- 사용자 명시 승인으로 기존 `output/rsi_cache_native_front_foot_20261023` 사용, 두 IK bank 미적용. 78k 소스·Box/Chair/Table 균등·SIT/CLIMB·전체 레이어·2048환경·lr2e-5 유지.
- GPU0 wrapper3032593, 출력 `output/box_chair_table_mixed_full_ft/local_78k_noik_20261023_174510/`, 실행 기록 `../runs/box_chair_table_78k_noik_20261023_174510/active_run.json`. 초기화 후 연속4회 update/fps 확인. 추가 평가/검사 실행하지 않았다.

### 미세 보정 RSI 재생성·viewer 교체

- 사용자 요청으로 관절±0.1rad/root 총2cm 제한으로 실패 phase236개를 재생성해 정적110개 통과(관절최대0.1rad·root최대1.999cm). 별도 `output_etc/check/native_rsi_small_ik_20261023`에 보존하며, preview에 `--static-bank` 옵션을 추가해 물리 재검사 전임을 명시한다. 대량 PhysX/AMP 검사와 학습은 실행하지 않고 기존 실행 cache는 유지한다.

### 사용자 피드백: RSI IK를 미세 보정으로 제한

- 기존 ±0.8rad/축별8cm는 과도한 변경으로 사용자 기각. 관절 축당±0.1rad·root 총 이동2cm 이내, 원자세 penalty 강화. 이 범위에서 해결되지 않는 자세는 제외한다. preview도 최대 변경 대신 최소 변경 후보를 선택한다.
- 기존 IK v3 결과/viewer는 과거 검토 자료이며 채택하지 않는다. 이 제한으로 후보를 재생성하거나 추가 검사는 실행하지 않았고 학습/실행 cache도 변경하지 않았다.

### Native RSI 관절 접촉 IK 구현·검증 (학습 보류)

- 사용자 요청으로 `rsi_native/ik.py` 추가. 다리18DOF·root XYZ bounded least-squares로 발 접촉/좌판·상판 footprint·collision·원자세를 함께 맞춘다. 상체·heading 유지, 관절 축당±0.8rad/root 축당±8cm. 원본 실패 자세·delta 보존. 열린 mesh 판정 한계와 기존15mm proxy/20mm floor/2mm foot-edge 허용은 유지한다.
- 기존 audit 실패 phase에서 물체/과제별 서로 다른 clip 최대2개씩, 총236개 시도→정적164→2048환경0.1/0.5초 PhysX67개→과거9프레임 geometry/foot-edge·AMP finite까지45개 통과. Chair SIT30/CLIMB5·Table SIT6/CLIMB4. 기존 완료 후보0이던 Table SIT1종 복구. 모든 탈락 자세를 복구한 결과는 아니다.
- `build_training_bank.py`는 IK delta를 source clamp 후 history에도 적용하고 limit/velocity/FK를 정합, history 관통 후보 제외. 실행 cache/YAML은 변경하지 않았다. CPU13개 통과, batching 최적화 후 IK 테스트 재통과. 최종 audit rc0/optimizer0, 상세 `output_etc/check/native_rsi_ik_physics_20261023_v3/REPORT.md`.
- `ik_preview.py`가 물리 통과 후보의 BEFORE/AFTER8장면 export. 로컬 viewer 화면 확인 `output_etc/check/native_rsi_ik_local_view_20261023_v3/`, N/P 전후 전환·SPACE 고정PD·R reset. AFTER는 물리 통과 기준이며 history 통과와는 별도다. 사용자가 검토 후 결정하도록 **학습은 시작하지 않았다**.

### 발 관통·의자 정면 제한 검증 완료, 사용자 지시로 학습 보류

- 앞서 승인된 수정 절차로 supported 학습3015658/3015685를 종료. foot mesh 모서리와 실제 native 물체 삼각형 교차 검사를2mm 기준으로 추가하고, 단독2048환경의1/2/3/6/30step에도 적용했다. `NativeFootGuard`는 실제 reset에서 자기·상대 에이전트의 발과 장면의 모든 native 물체를 검사한다. 열린 얇은 상판도 edge 교차로 검출하며, 완전한 체적 무충돌 보장은 아니다.
- 의자49종의 front가 실제 등받이 반대임을 mesh로 대조. 접근은 정면±45도, 중앙12cm 착석/상판 상태는 허용한다. 뒤쪽 접근을 좌판 중심180도 회전 후 재검사하고, loco 초기화도 정면이 되도록 배치한다. 실행 준비 bank `output/rsi_cache_native_front_foot_20261023`, `chair_front_only`/`strict_foot_collision` 설정.
- CPU22개 통과. native 물리 audit rc0 `output_etc/check/native_rsi_front_foot_physics_20261023_v3/`. 최종2048환경 reset은 `validate_reset.py`로 **optimizer0·정책행동0** 검증: accepted발교차0·의자정면위반0·실제1step후 발교차0, AMP history오차0/전체finite. native활성1368·중간/접근941, 장면발교차2건·Box부유100건 재추출. 결과 `output_etc/check/native_rsi_front_foot_review_validation/REPORT.md`.
- 완료 phase 보존 물체: Chair SIT47/49·CLIMB28/49, Table SIT9/10·CLIMB5/10. phase 없는 모션은 안전 접근 pool로 대체하므로 late draw70%가 완료 자세70%를 의미하지 않는다. 모든59종에 안전 후보는 있다.
- Viewer를 root/DOF tensor reset으로 바꾸고 CPU 환경 좌표계 원점 중복을 수정, 활성 목표만 표시. 접근/완료10장면, 초록=정면·빨강=뒤쪽. 최종 로컬 viewer `output_etc/check/native_rsi_front_foot_final_review/`. **사용자가 RSI 확인 후 학습을 결정한다고 지시했으므로 본학습은 재시작하지 않았다.**

### 지면·물체 지지 보정 후 RSI 재검증·78k 재시작

- 사용자 승인으로 full-motion 학습 그룹3012276 종료. 물체 높이 차이에 따른 constant-Z 이동을 중간 과정 전체에 적용해 발이 뜨는 실제 문제를 확인했다. 원본FK상 지면을 딛던 nonphase 발은5mm로 투영하고 지지점 수직속도를 제거한다. `support.py`에서 발/골반의 지면/수평 mesh면 근접3cm를 요구하고 벽면 접촉·부유는 제외한다. history 각 프레임도 같은 방식으로 보정한다.
- 수정 후보 native2048환경0.1/0.5초 PhysX 재검증 rc0. phase-only 필터를 제거한 full-safe bank에 Chair SIT10629/CLIMB1273, Table SIT1753/CLIMB224개 보존. 기존 Box RSI도 발mesh/골반 sphere 지지 필터로 부유 reset을 재추출한다. 실행bank `output/rsi_cache_native_supported_20261023/`, late70/full30·source78k·Box/Table/Chair균등·전체FT 유지.
- CPU19개·통합2048환경1epoch rc0. native활성1381/nonphase745, Box 부유104건 재추출, AMP history 오차0/전체finite. Table/Chair SIT snapshot 최저발5mm, native4조건 지지검사true 확인. 검증 `output_etc/check/native_rsi_supported_78k_smoke/`, 새 viewer Table SIT부터 표시·스크린샷확인 `output_etc/check/native_rsi_supported_view/`.
- 사용자 승인대로 78k부터 본학습 재시작: wrapper3015658, `output/box_chair_table_mixed_full_ft/local_78k_supported_rsi_20261023_163941/`, 실행기록 `../runs/box_chair_table_78k_supported_rsi_20261023_163941/active_run.json`. 연속4회차 update 확인. 앞선 잘못된 bank/학습/검증은 보존하고 활성 기준에서 제외한다.

### 78k 소스·native RSI/AMP 통합 후 혼합 본학습 시작

- 사용자 승인으로 기존 Table SIT viewer 종료. 78k checkpoint의 epoch78000·73k와 모델key/shape동일 확인 후 혼합 FT 소스를78k로 변경했다. Box/Table/Chair 목표 각1/3·SIT/CLIMB50:50·전체레이어·lr2e-5·2048환경 유지.
- `build_training_bank.py`가 통과 native 후보의9프레임 참조를 native32 관절한계로 보정하고 FK/velocity·동일 rigid retarget으로 AMP를 재생성한다. `utils/native_rsi.py`에서 실제 물체yaw/XY로 배치하고50% 확률 적용, 누락asset/task는 loco 대체. 장면 재시도에 두 사람 body-point15cm 필터를 추가. 실행bank `output/rsi_cache_native_chair_table_20261023/`.
- CPU13개 통과, GPU2048환경1epoch rc0. 초기 native활성1285슬롯, AMP history 오차0·전체finite·sourceepoch78000, frozen0 확인. 로그 `output_etc/check/native_rsi_78k_train_smoke.log`. 실제 reset6조건 snapshot을 로컬 viewer에 표시·화면확인 완료(`output_etc/check/native_rsi_78k_view/`). N/P 전환·SPACE 고정PD·R reset. viewer 유지.
- 사용자 요청으로 GPU0 본학습 시작: wrapperPID2998041, `output/box_chair_table_mixed_full_ft/local_78k_rsi_20261023_144809/`, 실행기록 `../runs/box_chair_table_78k_rsi_20261023_144809/active_run.json`. 연속4회차 update 로그 확인. 장기 policy 성공률은 아직 평가하지 않았다.

### Table SIT 목표·착석 RSI 로컬 viewer

- 사용자 요청으로 `rsi_native/view_table_sit.py`를 추가하고 로컬 DISPLAY:0에 실행. 실제 native Table9종·humanoid를 불러와 검사 통과 RSI 자세와 SIT 목표(상판+.10), 점 성공20cm 구, 골반 위치를 표시한다. N/P 물체 전환, SPACE 고정PD 물리, R reset. 정책 추론·학습은 실행하지 않는다.
- 로컬 실행 `VIEWER_READY`와 스크린샷 육안 확인 완료. 첫 table의 초기 root-goal 거리2.11cm. 산출물 `output_etc/check/table_sit_local_view_20261023/`와 동명log. Table1종은 통과 RSI가 없어 제외했다. viewer는 사용자 확인을 위해 실행 상태로 유지한다.

### Table 관측 bbox 상판 보정·Box tokenizer 공유

- 사용자 요청으로 혼합 FT에 `table_input_bbox: metadata_top_box_tokenizer`를 설정했다. Table 관측의 bbox 상단Z만 dataset tarSitPos.z로 맞추고 관측 flag/추가descriptor를0으로 보내 Box30D tokenizer를 공유한다. Chair35D branch 유지. 물리 bbox·mesh·RSI·reward/평가 목표·pose는 변경하지 않는다. 옵션이 없는 이전 checkpoint는 기존 입력 유지.
- 학습10/평가8개 Table metadata와 bbox 꼭짓점, 아래면/XY·속도·Box/Chair 불변·입력 원본 불변을 검사했다. 혼합·목표 회귀 포함 CPU9개 통과(`output_etc/check/table_box_input_regression.log`). 본학습 미실행, native RSI 통합은 별도 미완료.

### Box·Chair·Table 목표 비율 균등화

- 사용자 요청으로 Box75%/native25%를 폐기하고 세 물체군 목표 비율을 각1/3로 수정했다. native환경 비율2/3, 각환경 chair/table 목표 하나씩이므로2048환경의4096목표는 Box1366/Chair1365/Table1365다. SIT/CLIMB50:50 유지. CPU 혼합 경로7개 통과(`output_etc/check/box_native_equal_ratio_cpu.log`), 본학습 미실행.
- Box heading은 기존대로 pose7의 quaternion을 통해 GTA transform/attention에 전달된다. 30D Box tokenizer 자체에는 각도 항목이 없으며 local 선속도3+각속도3+bbox24다. 입력 코드는 변경하지 않았다.

### Box도 SIT/CLIMB만 학습하도록 범위 확정

- 사용자 답변에 맞춰 `box_chair_table_mixed_full_ft`의 전 물체 과제를 SIT/CLIMB50:50으로 수정하고 AT·ON_TOP을 제외했다. AMP 분포도 기존 Chair/Table SIT/CLIMB 설정으로 정렬했다. Box75%/native25%, Box 크기·reward·RSI 경로는 유지한다. 명시적 `training_tasks`에 한해 validator가 전용 확률을 허용한다.
- RSI 캐시의 기존 task확률이 YAML을 덮는 문제도 수정하고 회귀 검사 추가. CPU8개 통과. GPU2048환경 1회차 rc0: Box SIT1549/CLIMB1523, native SIT509/CLIMB515, AT·ON_TOP0개. 최종 로그 `output_etc/check/box_chair_table_sit_climb_smoke_v2.log`. 앞선 smoke는 실패/과거 검증으로 보존한다. native RSI reset/AMP 통합 전, 본학습 미실행.

### Box·Chair·Table 기존 혼합 설정 복원

- 사용자 요청으로 `box_chair_table_mixed_full_ft` YAML/train/test/VNC를 추가. Box75%는 stage1_v2_fixed task확률·크기·reward·size-aware RSI 유지, native25%는 chair/table 목표 하나씩 SIT/CLIMB50:50. native goal은 object_comparison, 새35D tokenizer·73k 소스·전체레이어 FT 유지.
- `HumanoidChairMixedFT`의 native 자산 배정을 table까지 확장하고 metadata/seat를 제공. 기존chair-only 혼합 동작과 Box 경로 보존. CPU7개·wrapper 문법 통과. GPU2048환경 한 회차 rc0: Box1536/native512, table512/chair512 목표, Box크기 불변, frozen0·actor/critic 새tokenizer gradient 양수 확인. 로그 `output_etc/check/box_chair_table_mixed_smoke.log`.
- 본학습은 실행하지 않음. native RSI bank는 아직 reset/AMP history 통합 전이라 해당 구성은 native loco 초기화로 명시. 상세 `markdowns/reference/box_chair_table_mixed_full_ft.md`.

### Native chair/table RSI 후보 구성·검사

- `rsi_native/`에 SIT20/CLIMB7 참조의 native chair49/table10별 retarget·관절한계 보정·전신 FK·mesh 검사와 별도 PhysX audit를 추가했다. object_state의 z는 지면 원점이며 지지면을 참조 자세에서 추정한다. 참조 손 위치와32DOF FK 차이 최대23.1cm도 재계산했다. 원본 후보·실패 결과를 보존한다.
- CPU geometry/FK4개 통과. GPU2048환경·1명/1native물체·0.1/0.5초 고정PD 검사 rc0·optimizer0. 보정 후 통과 물체/후보: Chair SIT49/49·9472, CLIMB48/49·663, Table SIT9/10·1753, CLIMB9/10·111. 3개 asset/task 조합은 후보가 없어 제외 목록에 기록했다.
- 결과 `output_etc/check/native_rsi_physics_20261023_v8/REPORT.md`, 원본/보정 bank와 정적 검사도 check 아래 보존. 통과는 장기 정책 안정성·정확한 체적 무충돌 보장이 아니다. 두 사람/주변물체 reset·보정 AMP history는 아직 연결하지 않았고 학습은 재시작하지 않았다.

### Fine-tuned tokenizer 정책용 단독 점/영역 평가

- `evaluation/finetuned/`에 35D tokenizer checkpoint용 1명/1고정물체 어댑터와 runner를 추가했다. Chair26/Table8 각각 SIT/CLIMB, 512환경×평가 seed0/1/2·900step·종료 frame을 포함한 root 점/영역 ever를 측정한다. 기존 public/73k와 paired 비교는 하지 않는다. 초기 상태·trace·hash·종료 frame 검증과 독립 float64 재채점을 기록한다.
- 사용자 승인으로 로컬 viewer를 종료하고 학습은 유지했다. 평가 대상은 화면의 8600회차로 고정했다. 정식 결과 경로는 `output_etc/check/finetuned_eval_8600_formal/`, 실행 로그는 `../runs/checks/chair_full_ft/finetuned_formal.log`다.
- 평가 Table8에서 metadata와 bbox 높이 차이를 검출했다. object_comparison CLIMB의 의자 판정을 tokenizer 라우팅 flag 대신 실제 물체 종류로 수정했다. 학습 Table10은 두 높이가 같아 수치상 영향이 없다. 전체 chair75/train-table10/test-table8 목표 회귀와 점/영역 판정 CPU3개 통과. 네 조건의 소규모 GPU rollout과 trace 재채점도 통과했다.

## 2026-10-22

### Object comparison 목표점에 맞춘 Chair/Table 전체FT

- 사용자 승인으로 metadata-all 학습 PID2869928을 종료. `chair_table_eval_full_ft` YAML/train/test/VNC 추가: Chair SIT/CLIMB=좌판+.10/.94, Table SIT=bbox상판+.10, Table CLIMB=metadata+.94로 `evaluation/object_comparison/prepare.py`의 목표점과 정렬했다. SIT와 기존 물리 지지면·feet/영역 학습 조건은 유지한다.
- CPU3개 통과: 전체chair75/table10 목표 대조, 기존metadata목표회귀·SIT불변. GPU2048환경 epoch1/저장 rc=0, 최대목표높이오차2.4e-7m·frozen0·actor/critic 새tokenizer gradient양수. 저장 모델16환경/16step 추론 rc=0. 로그 `../runs/checks/chair_full_ft/evalgoal_{train,eval}.log`.
- 기존 결과를 resume하지 않고 73000부터 새 본학습 시작. GPU0 PID2883544, 2048환경·전체레이어·lr2e-5·20,000회. 실행 기록 `../runs/chair_table_eval_full_ft_20261022_212639/active_run.json`, 출력 `output/chair_table_eval_full_ft/local_20261022_212639/`.

## 2026-10-22

### 물체 종류별 독립 SIT/CLIMB 평가
혼합 대신 evaluation/object_comparison/을 기본 평가로 지정. Box18/Chair26/Cabinet9/Table8 각각 독립 SIT·CLIMB, 종류내균등복원추출, public40k vs우리73k paired 비교. Box는 원본TokenHSI test18 XY3종×높이6종 그대로. SIT기존목표 및 CLIMB의자좌판/기타metadata+94cm 유지. 기존혼합코드·결과 보존. CPU61자산/Box크기/회전/높이경계 통과. GPU5에서 전체8조건×두정책 각4env·seed0·900step의16실행 rc0, 8paired초기장면/DOF exact, 전16trace재채점/종료frame/hash/float32float64 검증 통과. 검증배치 runs/object_comparison_20261022/eval_20261022_205431_smoke. 정식512×3seed 재평가는 미실행.

## 2026-10-22

### CLIMB 원본 category 혼합 평가
`evaluation/climb_mixed/`에 Box/Cabinet/Table_Square 각1/3·종류내균등복원추출 평가를 추가. 테스트35종(18/9/8), native mesh와 metadata+94cm 목표, paired frozen TokenHSI40k/우리73k. SIT 및 기존결과 보존. 실제 category/model배정, 전체/종류별 seed통계 기록. 원본가중치/추출 seed0/1/2 CPU동일성, 자산35개 경로/구문, GPU5 두정책각16env×seed0×900step, 전체초기배열exact, trace재채점/hash/float32float64/종료frame 검증 통과. 정식512×3seed 결과는 아직 없음. 실행법 markdowns/config.md 및 evaluation/climb_mixed/README.md.

## 2026-10-22

### Git 대상에서 생성 데이터 제외
독립 폴더 복사에 따라 노출된 joint_carry 모션/RSI bank, 로컬 데이터 링크, 배열 산출물을 .gitignore에 추가. 소스·설정·문서는 유지하고 실제 파일과 학습 프로세스는 변경하지 않음. git status/check-ignore로 확인.

## 2026-10-22

### stage1_combined 독립 폴더
사용자 요청으로 코드·의자/테이블 에셋·73000 초기 weight를 독립 복사하고 catalog와 checkpoint 참조를 새 폴더로 변경. 실행 중 학습은 기존 경로 보존. 복사본 파일/경로/구문 검증 수행.

## 2026-10-22

### Compact chair bbox35 tokenizer 및 20,000 iteration 실행
PointNet을 35→256→128→64 MLP로 대체. 속도6, bbox24, 등받이XY2, 좌판XYZ3 입력. 기존 가중치/정규화 동결 및 actor/critic gradient 짧은 GPU 검증 통과. chair FT에만 inclusive 종료 조건 적용. 상세 실행은 ../active_run.json 참조.

# Changelog

## 2026-10-22

### 의자 좌판 SIT/CLIMB + 등받이 PointNet 토크나이저 FT

- 73000에서 의자 object MLP를 복사하고 로컬 표면128점(일반64/등받이32/좌판32)과 좌판·등받이·방향 descriptor를 추가했다. 점군별 PointNet pooling, zero-init Linear residual. actor/critic의 chair tokenizer만 학습하며 나머지165개 원본 parameter와 scene/AMP/value 통계는 고정한다.
- 2명/2048환경/4물체, 대상 의자2개+주변 random Box/Table/Chair2개. 의자train49/test26 분리, 좌판 SIT+0.10/CLIMB+0.94 목표, loco 초기화, 고정 native mesh를 사용한다. 박스용 RSI는 의자에 쓰지 않는다.
- CPU75종 descriptor/순서 불변/등받이 반응/gradient 검증 통과. 실제2048환경 epoch3 저장·원본165텐서/통계 bitwise 불변·토크나이저/PointNet gradient·좌판 목표0오차·test16환경 추론 rc=0. 상위 README.md와 검증 로그 참조.
- GPU5에1000 epoch 설정 파일럿을 시작했다. 실행 경로/PID/log는 상위 active_run.json.


### 의자 6종 순환

- ksh test 의자 6종을 독립 복사하고 CHAIR_MESH로 선택. 동일 seed/AABB·정책 유지.
- cycle_chairs.sh가 의자당 1회 평가 후 정상 종료를 기다려 다음 의자를 로드. viewer는 동일 noVNC 세션에서 기본 2바퀴 수행. 기존 실행 프로세스 종료 없음.
- 검증 로그는 상위 chair_cycle_smoke.log. GUI 렌더는 미검증.


### 독립 사본의 의자 ON_TOP 시각화 probe

- CHAIR_ONTOP=1인 test/holding_ontop에서 운반 슬롯만 의자 메시로 교체. 기존 샘플링 AABB에 맞춰 중심·축별 크기 조정, VHACD 충돌과 단일 rigid body/0 DOF 검증. 받침·정책 관측·reward·성공 기준 유지.
- GPU5 headless 2명/1환경/1반복 rc=0. viewer 명령은 상위 view_chair_ontop.sh. viewer 렌더 미검증.


최신 변경부터 기록한다. 현재 실행법은 [config](markdowns/config.md), 코드 위치는 [structure](markdowns/structure.md)를 따른다. 과거 변경·검증 전체는 [보관 changelog](markdowns/legacy/changelog_before_cleanup.md)에 있다.

## 2026-10-21

### PUSH raw 직육면체·마찰0.3 설정 추가

- 1.04m 손 높이에서 1.2~1.35m 정육면체가 미끄러지기 전에 기우는 문제를 줄이려고 `stage1_v2_fixed_push_360_raw_cuboid` config/train/test/VNC를 분리했다. push 상자만 x1.2~1.7m·y/z1.45~1.7m(y≥z)·15~20kg·마찰0.3으로 바꾸고 PUSH 에피소드를 1200step으로 줄였다. 보상·원본 RSI·접촉 높이·기존 과제와 그 상자 마찰1.0은 유지한다. `sample_box_properties`에 축별 범위를 추가했고 기존 정육면체 샘플링은 그대로다.
- CPU 203개 통과, 2개는 서버에 없는 `Stage1V2Fixed_00060000.pth` 로드로 실패. GPU5 2048환경 짧은 학습 epoch4 저장·73000 전이·모델 finite, 실제 생성된 push 상자 256개의 마찰0.3·질량·크기·y≥z와 나머지 상자 768개의 마찰1.0, test wrapper 평가를 확인했다. 본학습·GUI viewer는 실행하지 않았고 기존 학습은 유지했다. [구현·검증](markdowns/reference/stage1_push_raw.md#직육면체-variant), [자료](output_etc/check/stage1_v2_fixed_push_360_raw_cuboid/).

## 2026-10-20

### 현재 서버 실행 환경·PUSH raw 전이 기준

- 실행 기본 conda를 `tokenhsi_jhh`로 바꾸고(runtime_env·run-gui·VNC wrapper), 문서 명령을 GPU5 기준으로 갱신했다. GPU5의 기존 MPS daemon에 자동 연결된다. 데이터 링크는 `/home/hwanhee/CVPR2027/TokenHSI`로 정상 연결돼 있다.
- `stage1_v2_fixed_push_360_raw`의 `source_checkpoint`를 `output/stage1/Stage1V2Fixed_00073000.pth`로 바꿨다. config의 raw 학습/학습 VNC 명령은 `PUSH_CHECKPOINT` 없이 Stage1에서 바로 전이한다. 60000 기반 PUSH360 checkpoint의 raw warm-start는 teacher 불일치로 거부되며 `test_push_raw.py`에 반영했다. 이 서버의 isaacgym은 torch 선 import를 거부하므로 USAGE CPU 검증 명령이 isaacgym을 먼저 import한다.
- CPU 199개 통과, 2개는 이 서버에 없는 `Stage1V2Fixed_00060000.pth` 로드로 실패. GPU5 2048환경 짧은 학습에서 공용 `output/rsi_cache_stage1_v2`(4335 profile, 최초 약25분)를 생성하고 epoch3 저장·source_epoch 73000 전이·모델 finite를 확인했다. 1환경 headless 평가, 4환경 학습 noVNC(6080)·1환경 테스트 noVNC(6081)의 실제 렌더·움직임·HTTP200/WebSocket101을 확인했고 검증 프로세스는 종료했다. 기존 GPU5 학습은 유지했다. 본학습은 시작하지 않았다. [자료](output_etc/check/push_raw_jhh_20261020/).

### PUSH 원본 RSI·1.04m 접촉점·가변 정육면체

- `stage1_v2_fixed_push_360_raw` config/train/test/VNC를 분리했다. 원본 RSI/AMP 이력·접촉 높이1.04m·한 변1.2~1.7m·질량25~30kg을 적용하고, 실제 크기의 면/바닥 위치와 모든 사람 충돌 형상의 최소 수평 후퇴를 계산한다. 기존 과제/비율/KL·기존 checkpoint·공용 cache/원본 NPY는 유지한다. 기존 PUSH는 명시적 warm-start로 전이한다.
- CPU47개 통과. 51크기×11후보×3질량+무작위365건의 실제 FK/별도 GPU PhysX2048건에서 초기 침투0·후퇴 최대6.784mm·1초 PD 최대 기울기0.740도·머리 충격0N을 확인했다. 실제2048환경 epoch2/frame262144 저장과 모델/542 scalar finite·입력/AMP 정규화 불변, 1환경64회128쌍의 초기 침투/머리 충격0·최대 기울기3.964도와 원본 관절/속도/AMP 이력·실제 물성을 확인했다.
- 기존 보정 reset의 이전 상태 bitwise 동일·새 wrapper/4환경 배치 전달을 확인했다. 본학습·GUI viewer는 시작하지 않았고 기존 학습은 유지했다. [구현·검증 범위](markdowns/reference/stage1_push_raw.md), [자료](output_etc/check/push_raw_variable_20261020/).

### PUSH 추론의 원본 RSI 선택 옵션

- test/VNC에 `PUSH_RSI_MODE=corrected|raw`와 직접 실행 인자 `--push_rsi_mode`를 추가했다. 기본 보정을 유지하고 `raw`에서 원본 관절·손 높이·참조 속도 및 초기 AMP 이력을 사용한다. 선별 bank·상자/goal 배치·보상·checkpoint 계약은 유지하며 직접 학습 실행의 raw override는 거부한다. [사용법](markdowns/USAGE.md#push-추론-rsi-자세).
- 관련 CPU31개 통과. 1환경 추론에서 원본11후보의 반환 상태와 관절·속도·9프레임 AMP 이력 일치,8회 원본 reset의 손 높이 약0.958~1.093m를 확인했다. 옵션 생략 시 이전 보정 모드의 root·관절·속도·상자·goal·선별 시각이 bitwise 동일했다. 학습 재실행·공용 cache 변경은 하지 않았다.
- 사용자 checkpoint와100회·300step 인자로6081 원본 RSI noVNC를 실행하고 HTTP200/WebSocket101/RFB·실화면·움직임을 확인했다. 기존 본학습은 유지했다. [검증](output_etc/check/push_rsi_raw_20261020/raw/validation.json), [기본값 비교](output_etc/check/push_rsi_raw_20261020/default/validation.json), [viewer](output_etc/check/push_rsi_raw_20261020/viewer_validation.json).

### PUSH RSI 실제 reset·초기 충돌 정밀 분석

- 원래 샘플러와 배치를 유지한2048환경 학습 rollout·1환경 테스트64회에서 접촉 RSI1016/128명을 관측했다. 원본의 머리 침투는0이었으며 팔 높이 보정 후 손 기준 정렬로 몸이41~52cm 전진하면서 reset의 머리 침투가1016/1016·128/128로 나타났다. 상자 초기 바닥 간격은 수치 오차 수준이며 기울기/속도0이었다.
- 같은 초기 상태의 머리/사람 접촉 제외·2cm 낙하·초기 속도0 대조로 바닥보다 초기 사람-상자 침투가 지배적임을 확인했다. [정밀 분석·MP4](markdowns/reference/stage1_push_rsi_collision_audit.md), [원시 수치](output_etc/check/push_rsi_precise_20261020/summary.json). 본학습·학습 코드/config·모션/캐시는 변경하지 않았고 optimizer 갱신0회다.

### PUSH 참조 모션 전체 MP4

- 현재 AMP의 밀기21개·보행4개를 native humanoid 정면/측면 MP4로 변환했다. 접촉 RSI 원본7개·후보11시각을 표시하고 [영상 목록](output_etc/check/push_ref_motion_mp4_20261020/videos/README.md)과 전체 ZIP을 생성했다. 원본 사람 모션 재생이며 상자/goal 궤적·reset 팔 보정·정책/물리 rollout은 포함하지 않는다.
- 1280×720·30fps·총8740프레임의 전체 디코딩, PUSH bank와 원본21개 hash 일치, ZIP 무결성을 확인했다. 학습 코드/config·원본 모션은 변경하지 않았다. [검증](output_etc/check/push_ref_motion_mp4_20261020/validation.json).

### PUSH 초기 충돌·크기별 RSI 설계 메모

- [별도 설계 문서](markdowns/reference/stage1_push_size_rsi_plan.md)에 기존 초기 충돌 진단 근거, 정육면체 1.0~1.5m·접촉 높이50%·질량20~25kg 조건과 크기별 RSI 보정·캐시·검증 방향을 정리했다. 선형 질량 매핑과 11종 크기는 제안으로 구분했다.
- 문서와 구조 안내만 갱신했다. 링크·diff를 확인했으며 코드/config 변경·캐시 생성·학습/시뮬레이션은 실행하지 않았다.

### PUSH 테스트/VNC의 에피소드 길이 인자

- PUSH/PUSH 360 test/VNC 명령 마지막에 `[episode_steps]`를 추가했다. 마지막 인자 → `EPISODE_LENGTH` → 저장 설정 순으로 적용하며 기존 생략 동작을 유지한다. [사용법](markdowns/USAGE.md#push-추론-에피소드-길이)을 추가하고 config의 인자 안내를 갱신했다.
- VNC 명시/생략 인자 전달4건·길이 우선순위3건·셸 구문·diff 검사를 통과했다. 기존8500 checkpoint의 RSI 시작1환경·16step headless 추론을 확인했다. 학습과 VNC viewer는 실행하지 않았다. [검증](output_etc/check/push_episode_argument/forwarding.json).

## 2026-10-19

### PUSH 360 추론의 직선 배치 옵션

- test/VNC에 `PUSH_SPAWN_MODE=random|straight`를 추가했다. 기본 360도 생성과 미는 면 뒤쪽 직선 생성을 선택하며 보행 반경 1.65~6m·목표 거리·RSI·학습 커리큘럼은 유지한다. checkpoint 설정과 별도의 추론 인자로 적용하고 학습/구 PUSH의 직선 override는 거부한다. 사용법은 [USAGE](markdowns/USAGE.md#push-360-추론-배치)에 기록했다.
- 관련 CPU29개 통과. 기존 midpoint500 checkpoint로 두 모드 각각4환경·2회·16step GPU 추론에서 위치 범위·관측 finite와 직선 횡오차 최대 약1.3e-6m를 확인했다. 공개 test wrapper의 직선1환경·16step도 통과했다. 학습·VNC viewer는 실행하지 않았다. [검증](output_etc/check/push_straight_spawn/validation.json).

### 실행 가이드·테스트 시각화 명령 정리

- `config.md`를 새 학습·4환경 학습 시각화·1환경 테스트 시각화 기준으로 정리했다. PUSH 기본 학습은 fixed60000에서 새로 시작하며 PUSH 전이/재개는 USAGE로 연결한다. 실제 PUSH360 midpoint500 checkpoint 예시와 과제/시작 동작 기본값·PUSH 옵션을 반영하고 중복 로딩 설명을 제거했다.
- 문서 섹션 순서·링크·스크립트/체크포인트 경로·셸 구문과 선택 checkpoint의 navigation/midpoint 설정을 확인했다. 문서 변경이며 학습·시뮬레이션은 실행하지 않았다.

### PUSH 접촉 상태 점수를 양손 중점 오차로 변경

- 새 학습용 `stage1_v2_fixed_push_360`에 `contact_midpoint_only: true`를 적용했다. 상태 점수의 개별 손-면 거리 항을 제외하고 중점 XY 오차·기존 Z 허용 구간만 사용한다. 유효 접촉의 거리/힘 판정과 옵션이 없는 기존 설정은 유지한다. 이전 PUSH/360 모델은 `PUSH_CHECKPOINT`로 전이하고 일반 resume는 저장 보상을 복원한다.
- CPU191개 통과. 손별 거리가 달라도 중점이 같으면 상태 점수가 같고 유효 접촉 판정은 유지됨을 확인했다. PUSH1500의 새 보상 전이 계약·diff 검사 통과. 이번 보상 변경의 GPU 재학습은 실행하지 않았다. [검증](output_etc/check/push_midpoint_reward/validation.json).

### PUSH 360도 접근·장거리 목표

- `stage1_v2_fixed_push_360` 전용 설정/train/test/VNC를 추가했다. 최종 목표 2~10m·보행 root 반경1.65~6m·360도, 지정 면까지의 우회 접근 보상·초기 겹침 재추첨을 적용한다. 0/1000/3000 epoch에 범위를 확대하고 평가는 최종 범위다. PUSH1800step·기존 과제600step, 기존/PUSH50:50·물성·AMP·RSI를 유지한다.
- `PUSH_CHECKPOINT`로 기존 PUSH 모델/정규화를 가져오고 새 optimizer/epoch으로 시작한다. 일반 resume는 저장 epoch의 단계를 첫 reset부터 복원하며, 구 PUSH checkpoint의 직접 재개/평가는 전용 wrapper에서 거부한다. 기존 본학습은 유지했다. [구현 메모](markdowns/reference/stage1_push_360.md).
- CPU190개 통과. PUSH1500 전이의 모델/정책·AMP RMS 일치·새 optimizer, 2048환경 짧은 학습/저장·모델/gradient/271 scalar finite·teacher/RMS 불변과 단계별 범위를 확인했다. 최종 박스 간격 보완 후64환경 보행/RSI·각16step 평가,4환경 학습 재개와1환경 평가 noVNC의 실제 장면/움직임·HTTP200/WebSocket101/RFB를 확인했다. 초기 진단값 자료형 오류는 수정 후 재실행했고, 장기 수렴은 미평가다. [검증](output_etc/check/push360_setup/validation.json).

### 테스트 task별 모션 로딩 축소

- 테스트/VNC는 양수 확률의 시작 skill만 읽는다. PUSH 보행은 4clip만 로드하고 PUSH RSI는 선별 bank의 7clip만 로드·검증하며 원본 motion ID를 매핑한다. 보행 테스트의 불필요한 PUSH bank/21clip 검사와 Stage2 테스트의 backward/sideways expert 로딩을 생략했다. 학습 로딩은 유지한다.
- CPU178개 통과. PUSH 보행4/PUSH RSI7/SIT20 모션의 각각1환경·16step headless 평가, 별도6082 포트의1환경 PUSH noVNC 실제 장면·움직임·HTTP200/WebSocket101/RFB를 확인했다. 기존 학습·6081 viewer는 유지했다. Stage2 보조 expert 제외는 정적/CPU 경로만 확인했다. [검증](output_etc/check/eval_motion_load_20261019/validation.json).

### PUSH 상자 질량 25kg

- 사용자 요청으로 `stage1_v2_fixed_push`의 담당 PUSH 상자 질량을 20→25kg으로 변경했다. 크기·마찰·기존 과제 물성은 유지한다. 기존 20kg RSI bank를 보존하고 후보 11개를 재사용하는 25kg 전용 bank와 변경 이력을 분리했다.
- 관련 CPU12개 통과. 2048환경에서 simulator rigid-body 질량 약25.000004kg·관측/모델/loss/gradient finite·짧은 학습/저장과 teacher/RMS 불변을 확인했다. 25kg 전 후보 PD 재선별은 하지 않았다. [검증 기록](output_etc/check/stage1_push_mass25/runtime_audit.json).

### Stage1 fixed 기반 PUSH fine-tuning

- `stage1_v2_fixed_push` 전용 환경/학습 config와 train/test/VNC를 추가했다. 기존/PUSH 환경군 50:50·상자 로컬 면의 수직 목표 2~5m·접촉 보상/물성/RSI/AMP를 분리하고 기존 config 경로를 유지했다.
- fixed60000 전이에서 기존 가중치를 보존하고 5-task/4-family로 확장한다. 새 optimizer·학습률1e-5, 기존 과제에만 동결 teacher KL, 정책/AMP 입력 정규화 고정을 적용했다. PUSH resume는 optimizer/epoch을 복원하고 비율/KL override를 허용한다.
- CPU167개·기존 4과제 전이 출력 비교 통과. 2048환경 epoch2/frame262144 저장, teacher/RMS 불변·모델/loss/gradient finite·PUSH 임베딩 갱신, 4환경 epoch3 resume, headless 평가와 4환경 학습/1환경 테스트 noVNC의 실제 렌더·움직임·HTTP200/WebSocket101/RFB·종료 정리를 확인했다. 장기 수렴·기존 성공률 보존은 미평가다.
- [구현 메모](markdowns/reference/stage1_push.md), [검증 기록](output_etc/check/stage1_v2_fixed_push_setup/validation.json). 본학습은 시작하지 않았고 검증 프로세스는 종료했다.

### tokenhsi_sm120 실행 환경과 noVNC

- 현재 서버의 `tokenhsi_sm120`·GPU0에 맞춰 conda/공유 라이브러리와 데이터 링크 12개를 연결했다. 로컬 Xvfb/x11vnc·noVNC 탐색, 실행별 프로세스 종료와 포트 정리를 보완했다. 원본 데이터·참고 repo는 수정하지 않았다.
- 여섯 컨피그의 headless 학습/테스트, 공통 `train_vnc.sh`의 실제 4환경 학습과 첫 4환경 overview 카메라, 기존 VNC의 1환경 테스트를 연결했다. 시각화 학습만 PPO/AMP minibatch를 조정하고 본학습 YAML의 배치는 유지한다. 학습/테스트 VNC 기본 포트는 6080/6081이다.
- 사용자 제공 `output/stage1/Stage1V2Fixed_00060000.pth`를 Stage2 세 설정의 기본 전이에 반영했다. 별도 이동한 fixed checkpoint는 내부 저장 env/sim도 복원한다. 실행 명령은 [config](markdowns/config.md), 환경·옵션은 [USAGE](markdowns/USAGE.md)에 갱신했다.
- CPU 155개 통과. Stage1 세 설정은 2048환경, 공유 받침 두 설정은 1024환경의 짧은 학습·저장/재로드·headless 테스트·4환경 학습 noVNC·1환경 테스트 noVNC를 확인했다. 10개 학습 checkpoint의 상태/loss finite·동결 actor 불변과 실제 렌더/움직임·HTTP200/WebSocket101/RFB·종료 정리를 확인했다. 초기 실행 중 수정한 셸의 종료 오류는 최종 스크립트 재실행으로 해소했다.
- 기본 Stage2는 단독/공동 RSI 최초 생성 중 사용자의 검증 중단 요청으로 종료했다. 생성된 캐시는 유지했으며 기본 Stage2의 optimizer 학습·최종 테스트/viewer는 미검증이다. 이번 검증 프로세스와 포트는 모두 종료했고 기존 서버 프로세스는 보존했다. 장기 학습 성능은 평가하지 않았다. [검증 요약](output_etc/check/runtime_setup_20261019/validation.json), [실행 결과](output_etc/check/runtime_setup_20261019/matrix.json).

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
