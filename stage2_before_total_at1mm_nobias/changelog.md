# Changelog

현재 실행법은 [config](markdowns/config.md), 코드 위치는 [structure](markdowns/structure.md)를 따른다. 정리 전 문서·변경 기록은 [보관 자료](markdowns/legacy/pre_four_configs/README.md)에 있다.

## 2026-11-15

### stage2_before_total_at1mm_nobias — BEFORE attention bias 어블레이션

- 기존 total_at1mm의 환경·학습 설정을 복사하고 `human_relation_fusion: before_zerobg_nobias`로 분리했다. actor/critic 사람 attention의 bias projection 파라미터·bias 계산을 생략하고 `attn_mask=None`을 전달한다. BEFORE embedding/후행 잔차 MLP fusion·Stage1 task bias·보상·샘플링은 유지한다.
- Stage1 26k 신규 전이·4명/1024환경/8토큰·900step, 기본 출력 `output/stage2_before_total_at1mm_nobias`. 기존 Stage2 bias checkpoint와 계약을 분리하고 train/test/VNC/eval wrapper·실험 등록·정식 평가 경로·config/structure/참고 문서를 추가했다. 기존 output의 두 실행 env/sim과 기준 YAML의 일치도 확인했다(런타임 메타데이터 제외).
- 관련 CPU 테스트33개 통과: bias 계산 미호출/mask=None, zero-bias 기준 출력·입력 gradient 일치, fusion 학습·사람 순서 불변성·actor/critic 전이·strict 재로드·계약 분리. shell 구문·문서 링크·diff 검사 통과.
- GPU6 `CHECK=1 MAX_ITERATIONS=1`·1024환경 짧은 학습 정상 종료(기존 runner 기준 저장 epoch2), 모델 유한값·bias projection 키 없음·양쪽 fusion embedding 보존 확인. 저장본으로 전용 test wrapper 4명·1환경·2step headless 추론 정상 종료. 기존 GPU6 학습 PID239158/145746 생존 확인. 현재 PyTorch 빌드의 Flash/memory-efficient attention 미지원 경고를 확인했다. 본학습·정식 평가·속도 비교는 실행하지 않았다. [CPU](output_etc/check/before_total_at1mm_nobias_20261115/tests.log), [학습](output_etc/check/before_total_at1mm_nobias_20261115/train.log), [추론](output_etc/check/before_total_at1mm_nobias_20261115/viewer.log), [검증](output_etc/check/before_total_at1mm_nobias_20261115/verification.json).

## 2026-11-11

### at1mm 리슘36k·새학습7k — 추가 정수1k 평가

- 최신36.5k/7.5k 중1k간격 미평가36k/7k만 GPU6/7에서 각각1모델×3seed 실행. 기존 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·512환경×3seed×900step,정지는 같은 기록에서 집계했다.
- 전체 XY80/XY100/동일쌍 선행AT(%): 리슘36k74.97±2.13/77.53±2.24/97.49; 새학습7k62.39±2.58/65.17±2.55/94.89. 리슘35k70.36 대비+4.61pp로3seed 모두 회복했으나34k78.58 최고 유지. 새학습6k60.77 대비+1.62pp로 해당 계열 전체 최고 갱신.
- 행동별 XY80후행/동일쌍AT: 리슘 SIT71.80/98.04·CLIMB79.10/98.29·ON_TOP74.92/96.97; 새학습66.35/94.62·55.66/96.07·63.08/94.53. 정지root중앙값·정지/관찰가능: 리슘1.72m14/36·1.57m22/28·3.12m192/239(payload1.40m); 새학습1.56m22/34·1.47m30/35·3.17m200/248(payload1.39m).
- 6회 정상 완료,CPU/GPU/JSON·초기조건·정지telemetry 검증 통과. 원본PTH·운영401파일·학습PID238346/239158 보존 확인. [결과](output_etc/check/at36_scratch7_eval_20261111/report.txt),[전체추세](output_etc/check/at36_scratch7_eval_20261111/overall_trends.txt),[정지추세](output_etc/check/at36_scratch7_eval_20261111/all_stops.csv),[검증](output_etc/check/at36_scratch7_eval_20261111/verification.json).

### at1mm 리슘35k·새학습6k — 추가 정수1k 평가

- GPU6/7에 각각1저장본×3seed만 실행했다. 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·512환경×3seed×900step. 정지는 동일512기록에서 집계했다.
- 전체 XY80/XY100/동일쌍 선행AT(%): 리슘35k70.36±1.56/73.43±1.89/96.92; 새학습6k60.77±1.01/64.21±.33/95.29. 리슘34k78.58 대비−8.22pp이며3seed 모두 하락,34k 최고 유지. 새학습5k61.82 대비−1.05pp,XY100은64.26→64.21로 유사하다.
- XY80 행동별 후행/동일쌍AT: 리슘35k SIT68.39/98.58·CLIMB71.97/96.04·ON_TOP70.63/96.55; 새학습6k58.21/94.44·57.08/96.63·63.20/95.19. 정지root중앙값·정지/관찰가능: 리슘1.67m12/33·1.39m18/23·3.11m197/238(payload1.40m); 새학습1.63m19/27·1.46m36/50·2.89m194/242(payload1.41m).
- 6회 정상 완료,CPU/GPU/JSON·동일 초기장면·정지telemetry 검증 통과. 새 리슘run 설정은 이름/초기화 항목 제외 기존과 동일. 원본PTH·운영401파일 보존,기존학습PID238346/239158 생존 및 tmux 업데이트 확인. [결과](output_etc/check/integer2_eval_20261111/report.txt),[전체추세](output_etc/check/integer2_eval_20261111/overall_trends.txt),[정지추세](output_etc/check/integer2_eval_20261111/all_stops.csv),[검증](output_etc/check/integer2_eval_20261111/verification.json).

### at1mm 리슘32/33/34k·새학습3/4/5k 평가 및 CUDA 오류 기록

- 사용자 요청으로 GPU6·7에 최초6저장본18seed를 동시 시작했으나 초기화 중 OOM/illegal-memory-access가 발생했다. 같은 시점 GPU6의 기존at1mm리슘PID85967·새학습PID1030761도 CUDA 오류로 종료됨을 tmux에서 확인했다. 별도 coupled 평가가 GPU6·7에 동시 실행된 것도 확인했으나 정확한 CUDA 오류 전파 인과는 확정하지 않았다. 학습 보존 성공으로 기록하지 않는다.
- 이번 평가 프로세스만 종료하고 실패 로그/산출물을 각폴더attempt1에 보존했다. GPU당1저장본×3seed로 재시도했고 마지막새학습5k는 비어 있는GPU6으로 변경했다. 사용자에게 최신34.5k/5.5k의 ZIP무결성·SHA256을 확인한 재개 명령만 제공했으며 학습을 직접 재시작하지 않았다. [오류 기록](output_etc/check/integer6_eval_20261111/incident.json),[리슘 체크포인트](output_etc/check/integer6_eval_20261111/recovery_checkpoints.json).
- 재시도18회는 모두 정상 완료. 기존 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·512환경×3seed×900step이며 동일512기록에서 정지를 집계했다. CPU/GPU/JSON 재채점·초기조건·정지 telemetry·원본PTH/운영401파일 보존 검증 통과. 원래 학습PID의종료 여부는 검증JSON에false로명시했다.
- 전체 XY80/XY100/동일쌍 선행AT(%): 리슘32k70.55±1.98/72.47±1.85/96.57,33k77.01±2.50/79.03±2.42/97.32,34k78.58±3.19/79.93±2.53/97.38; 새학습3k58.34±.45/61.41±.67/93.37,4k58.58±3.12/62.23±2.13/94.47,5k61.82±2.51/64.26±1.70/93.89.
- 리슘34k가 동일조건 평가 전체 최고를 갱신. XY80 SIT79.65/AT97.82·CLIMB77.65/98.01·ON_TOP78.54/96.97,행동균등78.61/97.60. 정지root중앙값 SIT1.63m17/34·CLIMB1.61m14/21·ON_TOP3.25m203/237(payload1.42m). 33/34k 두저장본 연속상승이며 새학습은3/4k정체후5k상승.
- [신규 결과](output_etc/check/integer6_eval_20261111/report.txt),[동일조건60개 전체 추세](output_etc/check/integer6_eval_20261111/overall_trends.txt),[행동별 전체 추세](output_etc/check/integer6_eval_20261111/all_trends.txt),[정지 추세](output_etc/check/integer6_eval_20261111/all_stops.csv),[검증](output_etc/check/integer6_eval_20261111/verification.json).

## 2026-11-10

### 미평가 1k 저장본7개 — total38/39·at1mm30/31·climb2_24·새at1mm1/2k

- GPU6·7에서 총21회, 각512환경×3seed×900step 평가했다. 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지, 정지는 동일512기록으로 집계했다. GPU6 최대3모델·GPU7 최대2모델 동시 실행 후 새학습2개를 이어서 실행했다.
- 전체 XY80/XY100/동일쌍 선행AT(%): total38k63.12±1.99/64.92±1.73/97.91,39k68.06±2.10/69.40±1.85/98.02; at1mm30k71.45±2.34/72.95±2.32/96.74,31k70.88±2.42/72.62±2.91/96.86; climb2_24k75.36±.70/78.32±.78/97.38; Stage1 26k 기반 새at1mm1k24.70±1.56/28.02±2.02/85.54,2k46.66±.72/50.78±.60/92.73.
- climb2_24k가 동일조건 평가된 모델 전체 최고를 갱신했다. XY80 행동별 후행/동일쌍AT:SIT70.35/97.60·CLIMB79.57/98.24·ON_TOP76.02/96.97,행동균등75.31/97.61. CLIMB 정지root중앙값2.02m·48/57,ON_TOP root3.12m/payload1.41m·188/238,SIT1.57m·20/30. 기존total27k·at1mm27k 모델별 최고는 유지. 단일 저장본 상승의 지속성은 미확인이다.
- 총54개 동일조건 저장본의 전체/행동별 XY80·XY100·동일쌍AT 추세를 재집계했다(total12k 기존 물리제외 결과 포함). 정지 추세에는 동일512기록만 포함하고 과거 별도64결과는 제외했다. 새at1mm는 이름·초기화 경로/재개 여부 외 환경·학습설정 동일성도 검증했다.
- 21회 정상 완료, CPU/GPU/JSON 재채점·초기조건·정지 telemetry 일치, 원본PTH·운영401파일·기존학습3PID 보존 검증 통과. [신규 결과](output_etc/check/integer7_eval_20261110/report.txt),[전체 추세](output_etc/check/integer7_eval_20261110/overall_trends.txt),[행동별 전체 추세](output_etc/check/integer7_eval_20261110/all_trends.txt),[정지 추세](output_etc/check/integer7_eval_20261110/all_stops.csv),[검증](output_etc/check/integer7_eval_20261110/verification.json).

### total36/37k·at1mm29k — 신규 1k 저장본 정식 평가

- GPU7 total36/37k, GPU6 at1mm리슘29k를 각각512환경×3seed×900step 평가했다. 기존 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지 조건이며 정지는 동일512기록에서 집계했다.
- 전체 후행 XY80/XY100/동일쌍 선행AT(%): total36k67.45±2.51/68.91±3.26/98.20,37k68.68±.54/70.25±.20/98.66; at1mm29k71.66±1.89/73.94±2.43/97.22. total35k69.51 대비36k하락·37k부분회복,at1mm28k67.57 대비29k회복. 모델별 최고는total27k73.15/75.71(AT97.55),at1mm27k72.27/75.01(AT97.04) 유지.
- XY80 행동별 후행/동일쌍AT: total36k SIT70.89/98.04·CLIMB64.83/98.61·ON_TOP66.97/98.10;37k66.57/99.09·64.17/98.30·71.28/98.65;at1mm29k71.99/97.60·73.49/97.27·70.96/97.07.
- 정지root거리 중앙값·정지/관찰가능: total36k SIT1.58m16/23·CLIMB1.60m15/21·ON_TOP3.40m198/229(payload1.41m);37k1.44m8/18·1.65m19/25·3.31m185/232(payload1.42m);at1mm29k1.59m17/24·1.69m19/20·3.08m196/246(payload1.39m).
- 9회 정상 완료, CPU/GPU/JSON 재채점·초기조건·통합정지기록 일치, 원본PTH·운영코드·기존학습3PID 보존 검증 통과. [결과](output_etc/check/total36_37_at1mm29_eval_20261110/report.txt),[직전/최고 비교](output_etc/check/total36_37_at1mm29_eval_20261110/comparison.csv),[정지](output_etc/check/total36_37_at1mm29_eval_20261110/stops.csv),[검증](output_etc/check/total36_37_at1mm29_eval_20261110/verification.json).

### stage2_before_at1mm — Stage1 26k 기반 새 학습 이름 분리

- 사용자 요청으로 기존stage2_before_total_at1mm의환경·학습YAML을이름만바꿔복사했다. 새이름stage2_before_at1mm/Stage2BeforeAt1mm,기본출력output/stage2_before_at1mm. 기본전이는Stage1JointPushDensity3Cone90Finetune_00026000.pth,load_checkpoint=false이며RESUME_CHECKPOINT 없이새학습을시작한다. 기존설정/저장본은보존했다.
- 전용train/test/VNC/eval wrapper4개,실험등록·900step·HumanoidMABeforePush·jointpush motion·정식평가PUSH지원목록을연결하고config표/명령및structure를갱신했다. 보상/모델/샘플링은동일하며AT보상1mm·후행current gate20cm 유지.
- 이름외YAML동일성·전이파일존재·native환경로딩/계약·shell구문·문서링크검증통과. 기존CPU테스트98통과/3skip. GPU6에서기존at1mm27k를새test wrapper로4명·1환경·2step headless추론하여정상종료확인. 새학습PPO실행은하지않았다. [CPU검증](output_etc/check/before_at1mm_config_20261110/tests.log),[실행검증](output_etc/check/before_at1mm_config_20261110/smoke.log).

### total 대비 at1mm 선행 높이 실제 준수 분석

- 기존 동일조건512×3seed×900step trace를 CPU로 재집계: total/at1mm 각각20/24/25/26/27/28k와total35k,총13저장본39회. 코드상1mm는실제상자중심Z−기존AT목표Z이며,3D거리≤.2m·선행alive와동시에충족하는지측정했다. 손놓기/접촉 여부의 직접 측정은 아니다.
- 동일27k,선행AT Ever total97.55%/at1mm97.04%. 도착쌍중1mm Ever89.56→97.56%,연속1초(30frame)충족88.14→96.96%,gate열린시간중1mm준수비율84.04→94.04%(쌍별시간비율평균후seed평균). 후행첫성공까지의구간으로제한해도76.21→89.09%. 동일초기쌍중양모델AT Ever를만족한1643쌍만비교해도1mm1473→1605,1초1450→1595.
- 비교6epoch 모두·각3seed 모두at1mm가1mm Ever조건부비율상승. 최신total35k는도착중85.03%·1초81.85%,at1mm28k96.53%·96.29%.27k행동별도착중1mm total/at1mm:SIT90.75/97.69,CLIMB90.57/97.96,ON_TOP88.72/97.40.
- total27k는AT도착1679쌍중1mm미충족175,at1mm27k1670쌍중41. 이미충족부분집합의후행동시성공27/175(15.43%)/4/41(9.76%);두집합선행AT Ever100%. 충족집합은1232/1504(81.91%)/1240/1629(76.12%),선행AT100%. 궤적사후분류이므로인과추정이아니다. at1mm28k는1mm유지성능이높아도후행전체67.57%(선행96.79%)로하락해높이만으로전체실패를설명할수없다.
- 39회원본후행동시성공·선행Ever정확복원,seed별packet/크기/goal/쌍매핑일치,strict⊆gate·연속유지⊆strict검증통과. 학습/평가조건변경·GPU실행없음. [요약](output_etc/check/total_at1mm_z_analysis_20261110/report.txt),[행동별/epoch별](output_etc/check/total_at1mm_z_analysis_20261110/summary.csv),[동일쌍비교](output_etc/check/total_at1mm_z_analysis_20261110/matched_comparison.json),[검증](output_etc/check/total_at1mm_z_analysis_20261110/verification.json).

### at1mm28k — 추가 1k 저장본 평가

- 확인 시 신규 정수1k 미평가는at1mm28k 하나였다(total35.5k·climb2_23.5k·at1mm28.5k 제외). GPU6에서 기존 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·512환경×3seed×900step 평가.
- XY80 후행/동일쌍 선행AT(%): SIT66.97/97.31,CLIMB73.29/98.90,ON_TOP65.82/95.82,행동균등68.69/97.34,전체67.57±2.04/96.79. XY100은72.17/97.31,74.11/98.90,66.55/95.82,균등70.94/97.34,전체69.37±1.70/96.79.27k전체72.27/75.01 대비−4.70/−5.64pp로하락하며27k모델최고유지. CLIMB70%대는유지되지만SIT·ON_TOP도동반하락했다.
- 동일512기록 정지/관찰가능·root거리 중앙값: SIT13/25·1.35m,CLIMB17/21·1.72m,ON_TOP204/244·3.17m(payload1.39m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초 기준.
- 3회 정상 완료·CPU/GPU/JSON 일치·초기장면/크기/거리/물리관측분리·통합정지기록·원본PTH/운영401파일/기존학습2PID 보존 검증 통과. [결과](output_etc/check/at1mm28_eval_20261110/report.txt),[전체추세](output_etc/check/at1mm28_eval_20261110/all_trends.csv),[정지추세](output_etc/check/at1mm28_eval_20261110/all_stops.csv),[검증](output_etc/check/at1mm28000_common_sumoff_physics_20261110/verification.json).

### 미평가 1k 저장본11개 — GPU6·7 병렬 정식 평가

- total32/33/34/35k,at1mm25/26/27k,climb2_20/21/22/23k를 GPU당 최대2모델(각3seed)로 평가했다. 각512환경×3seed×900step,공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지. 정지는 같은512기록에서 집계했다.
- 전체 XY80/XY100/동일쌍 선행AT(%): total32k67.94/70.04/98.09,33k67.17/68.68/97.61,34k69.15/70.66/98.31,35k69.51/71.54/98.54. at1mm25k68.53/70.39/97.16,26k70.89/73.16/97.49,27k72.27/75.01/97.04. climb2_20k71.10/73.20/97.84,21k70.61/73.16/96.76,22k71.81/73.74/97.68,23k70.47/73.43/95.75.
- 모델별 전체최고: total27k73.15±1.68/75.71±2.15(AT97.55) 유지,at1mm27k72.27±.71/75.01±.82(97.04) 갱신,climb2_22k71.81±.51/73.74±.61(97.68) 갱신. 최고 선택은 동일조건 평가된 저장본 내 기준이다.
- 최신 행동별 XY80 후행/동일쌍AT: total35k SIT73.00/99.53,CLIMB65.57/98.31,ON_TOP69.46/98.23;at1mm27k72.46/97.09,76.37/98.02,70.76/96.64;climb2_23k61.37/95.28,72.32/97.07,73.83/95.49. at1mm CLIMB25–27k72.89→70.83→76.37(AT97.71→97.72→98.02)로70%대를 유지하며,climb2는20–23k전체70–72%대이나23k SIT가 하락했다.
- 최신 정지/관찰가능·root거리 중앙값: total35k SIT12/25·1.50m,CLIMB26/38·1.64m,ON_TOP180/240·3.30m(payload1.40m);at1mm27k13/22·1.67m,14/22·1.72m,201/246·3.14m(payload1.39m);climb2_23k17/42·1.55m,46/53·2.02m,195/234·3.18m(payload1.39m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초 기준.
- 33회 정상 완료·CPU/GPU/JSON 일치·초기장면/크기/거리/물리관측분리·통합정지기록·원본PTH/운영401파일/기존학습2PID 보존 검증 통과. [11개결과](output_etc/check/latest11_eval_20261110/report.txt),[전체추세](output_etc/check/latest11_eval_20261110/all_trends.txt),[추세CSV](output_etc/check/latest11_eval_20261110/all_trends.csv),[정지추세](output_etc/check/latest11_eval_20261110/all_stops.csv),[검증](output_etc/check/latest11_eval_20261110/verification.json).

## 2026-11-09

### at1mm24k 평가·세 모델 동일조건 추세 정리

- GPU6에서 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·512환경×3seed×900step 완료. XY80 후행/동일쌍 선행AT(%): SIT74.78/97.38,CLIMB73.80/97.38,ON_TOP68.27/96.54,행동균등72.28/97.10,전체70.93±1.16/96.92. XY100은78.13/97.38,74.95/97.38,70.38/96.54,균등74.49/97.10,전체73.13±1.58/96.92.
- at1mm 모델 전체/행동균등 최고 갱신. CLIMB22→23→24k67.98→69.02→73.80(같은쌍AT98.88→96.86→97.38),ON_TOP23→24k63.58→68.27로 회복. total27k 전체최고73.15/75.71은 미갱신.
- 동일512기록 정지/관찰가능·root거리 중앙값: SIT19/33·1.56m,CLIMB23/27·1.66m,ON_TOP207/238·3.16m(payload1.38m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초 기준. 별도64환경 실행 없음.
- 동일조건의 기존 평가 total18/19.5/20–31k,at1mm15–24k,climb2_13–19k를 재집계했다. 전체는 실제 샘플 수 기준,행동균등 별도. 과거64환경 정지는 합치지 않고 stops512 존재분만 사용했다. [전체추세](output_etc/check/at1mm24_trends_20261109/all_trends.txt),[CSV](output_etc/check/at1mm24_trends_20261109/all_trends.csv),[정지추세](output_etc/check/at1mm24_trends_20261109/all_stops.csv),[출처](output_etc/check/at1mm24_trends_20261109/sources.json).
- 3회 정상 완료·CPU/GPU/JSON 일치·초기장면/크기/거리/물리관측분리·통합정지기록·원본PTH/운영401파일/기존학습2PID 보존 검증 통과. [24k결과](output_etc/check/at1mm24_trends_20261109/report.txt),[검증](output_etc/check/at1mm24000_common_sumoff_physics_20261109/verification.json).

### total31k·CLIMB+2m19k — 정식 평가

- GPU7(total)·6(climb2)에서 각512환경×3seed×900step 완료. 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지. 같은512 기록에서 정지 집계.
- XY80 후행/동일쌍 선행AT(%): total31k SIT71.44/97.89,CLIMB63.50/98.61,ON_TOP67.37/98.42,행동균등67.44/98.31,전체67.54±2.09/98.32. XY100전체69.10±1.86/98.32. climb2_19k70.06/97.17,71.37/96.62,62.58/96.75,균등68.00/96.85,전체66.13±.32/96.80;XY100전체68.11±.94/96.80.
- total30→31k 전체67.23→67.54로 유사하며27k 최고73.15 유지. climb2_18→19k CLIMB58.71→71.37 회복,ON_TOP69.40→62.58 하락. 행동균등66.54→68.00 상승과 샘플수 기준전체67.65→66.13 하락이 공존하며15k 전체최고71.55 유지.
- 정지/관찰가능·root거리 중앙값: total31k SIT10/17·1.60m,CLIMB22/23·1.70m,ON_TOP195/242·3.30m(payload1.42m);climb2_19k14/23·1.58m,47/55·2.03m,186/238·2.99m(payload1.40m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초 기준.
- 6회 정상 완료·CPU/GPU/JSON 일치·초기장면/크기/거리/물리관측분리·통합정지기록·원본PTH/운영401파일/기존학습2PID 보존 검증 통과. [결과](output_etc/check/total31_climb219_20261109/report.txt),[추세CSV](output_etc/check/total31_climb219_20261109/trends.csv),[정지](output_etc/check/total31_climb219_20261109/stops.csv),[검증](output_etc/check/total31_climb219_20261109/verification.json).

### total30k·at1mm23k·CLIMB+2m18k — 정식 평가

- GPU6(at1mm)·7(total/climb2)에서 각512환경×3seed×900step 완료. 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지, 같은512 기록으로 정지를 집계했다.
- XY80 후행/동일쌍 선행AT(%): total30k SIT66.39/98.15,CLIMB62.76/97.95,ON_TOP69.29/97.06,행동균등66.15/97.72,전체67.23±.40/97.50. XY100전체68.45±.15/97.50. at1mm23k74.46/95.93,69.02/96.86,63.58/96.53,균등69.02/96.44,전체67.31±1.94/96.46;XY100전체68.93±1.57/96.46. climb2_18k71.51/98.07,58.71/96.81,69.40/96.56,균등66.54/97.15,전체67.65±1.60/96.97;XY100전체70.16±2.53/96.97.
- 직전 대비 XY80전체 total−1.45pp,at1mm−1.59pp,climb2−2.36pp. 모델별 전체 최고는 total27k·at1mm22k·climb2_15k 유지. at1mm는SIT/CLIMB 상승과ON_TOP 하락이 공존하며,climb2 CLIMB는17k65.16→18k58.71로 하락했다.
- 정지/관찰가능·root거리 중앙값: total30k SIT16/26·1.58m,CLIMB29/36·1.50m,ON_TOP206/243·3.29m(payload1.38m);at1mm23k17/27·1.69m,27/28·1.57m,199/255·3.18m(payload1.40m);climb2_18k13/19·1.62m,43/53·2.09m,196/237·3.19m(payload1.40m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초 기준.
- 9회 정상 완료·CPU/GPU/JSON 일치·초기장면/크기/거리/물리관측분리·통합정지기록·원본PTH/운영401파일/기존학습2PID 보존 검증 통과. [결과](output_etc/check/total30_at1mm23_climb218_20261109/report.txt),[추세CSV](output_etc/check/total30_at1mm23_climb218_20261109/trends.csv),[정지](output_etc/check/total30_at1mm23_climb218_20261109/stops.csv),[검증](output_etc/check/total30_at1mm23_climb218_20261109/verification.json).

### at1mm22k·CLIMB+2m15/17k 실패 구간 비교

- total27k와 같은 공통/합일치/near해제/물리distractor제외512×3seed×900step 기록을 CPU 재집계했다. 후행개인Ever·동시성공·동일쌍선행AT Ever를 원본과 정확히 복원하고 대기trace row/owner 매핑도 검증했다. GPU/학습변경없음.
- 각1721쌍: at1mm22k 선행Ever1668·후행1186,실패535=AT미성공53+접근반경미진입303+근처이나XY미진입148+XY진입후동시높이미달12+기타19. climb2_15k1666/1231,실패490=55+251+160+8+16;climb2_17k1680/1205,실패516=41+271+175+12+17. 분류는 앞선total27k와 동일한 순차배타적 기준이다.
- CLIMB352쌍: at1mm22k 후행239/선행348,climb2_15k245/343,climb2_17k229/346. 접근했지만XY미진입67/72/74 중 같은gate·근처구간에 평가높이도 전혀못맞춘61/70/70쌍. 후행실패 중current선행성공·Ever해제·살아있음·속력≤.3·접근반경밖 연속3초 사례30/21/31쌍(다른분류와중복;시도전/후미구분). 시간·동작의 인과확정은 아니다.
- ON_TOP 접근미완료198/178/188 중payload바닥이실제지면에서10cm초과로1초연속떠있던사례175/146/160쌍. 집기확정아닌이동/들림대리지표로,전부미집기로설명할수없다. 가상받침XYZ재채점의회복/손실은전체40/5,36/5,46/6;CLIMB4/1,2/0,1/0으로작다. 운영성공조건/궤적은변경없음.
- [모델·행동별비교](output_etc/check/at1mm22_climb2_failure_20261109/comparison.csv),[검증](output_etc/check/at1mm22_climb2_failure_20261109/verification.json),[CPU분석](output_etc/check/at1mm22_climb2_failure_20261109/analyze.py).

### total27k 실패 구간 CPU 재집계

- 기존 공통·합일치·near50해제·물리distractor제외512×3seed trace만 사용했다. 원래 XY80 동시성공을 정확히 복원: 총1721쌍,선행AT Ever1679,후행동시1259. 실패462 중 선행AT미성공42,후행개인성공 있지만 동시성공없음8,유효한 후행/선행current 동시구간없음7,그 구간에 접근반경 미진입233,접근했지만XY80영역 미진입161,XY진입했으나 동시높이미충족11이다. 접근반경은 실물 XY대각반경+SIT/CLIMB.3m·ON_TOP.1m,ON_TOP은payload 기준이다. 분류는 순차·배타적이며 행동/물리 인과 판정은 아니다.
- 같은궤적 XY100성공1303,높이만무시1271,가상받침XYZ재채점1307(기존실패56회복·기존성공8손실). 가상위치재채점은 실제동작개선 예측이 아니다. GPU실행·학습/설정변경없음. [집계](output_etc/check/total27_failure_partition_20261109/summary.csv),[seed별](output_etc/check/total27_failure_partition_20261109/seeds.csv),[분석코드](output_etc/check/total27_failure_partition_20261109/analyze.py).

### total29k·at1mm22k — 동일조건 정식 평가

- total29k GPU7 시작 후 at1mm22k 저장 완료·archive 무결성을 확인해 GPU6에서 평가했다. 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·각512환경×3seed×900step, 동일512 기록에서 정지 집계.
- XY80 후행/동일쌍 선행AT(%): total29k SIT66.72/98.58,CLIMB70.27/99.45,ON_TOP69.12/97.79,행동균등68.70/98.61,전체68.68±.72/98.32. XY100전체71.76±1.73/98.32.28k전체69.47/71.92보다 소폭 하락,27k 최고 유지.
- at1mm22k XY80은 SIT71.11/96.84,CLIMB67.98/98.88,ON_TOP68.37/96.22,균등69.15/97.31,전체68.90±1.29/96.92. XY100균등71.01/97.31,전체70.82±1.22/96.92.21k CLIMB58.96/94.83에서 회복; 두XY 기준에서 기존 모델최고16k 전체68.01/70.39·균등68.27/70.57을 갱신했으나 격차는 작다.
- 정지/관찰가능·root거리 중앙값: total29k SIT13/19·1.47m,CLIMB17/21·1.67m,ON_TOP205/238·3.32m(payload1.40m);at1mm22k16/27·1.56m,27/31·1.61m,189/248·3.12m(payload1.39m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초 기준이다.
- 6회 정상 완료·CPU/GPU/JSON 일치·초기장면/크기/거리/물리관측분리·통합정지기록·모델별 이전 환경설정 동일·원본PTH/운영401파일/기존학습2PID 보존 검증 통과. [total 추세](output_etc/check/total29000_common_sumoff_physics_20261109/report.txt),[at1mm 추세](output_etc/check/at1mm22000_common_sumoff_physics_20261109/report.txt),[통합CSV](output_etc/check/total29_at1mm22_20261109/trends.csv),[정지](output_etc/check/total29_at1mm22_20261109/stops.csv),[검증](output_etc/check/total29_at1mm22_20261109/verification.json).

### CLIMB +2m17k — XY80/100·정지 평가

- GPU7에서 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·512환경×3seed×900step 평가, 동일 기록으로 정지를 집계했다.
- XY80 후행/동일쌍 선행AT(%): SIT69.91/96.18,CLIMB65.16/98.36,ON_TOP71.91/98.01,행동균등68.99/97.52,전체70.01±1.91/97.62. XY100은76.20/96.18,65.69/98.36,73.99/98.01,균등71.96/97.52,전체72.81±1.59/97.62.16k전체67.66/70.74보다 회복했으나15k71.55/73.06은 미갱신. CLIMB XY80은15→16→17k69.79→66.74→65.16(동일쌍AT97.42→97.49→98.36)이다.
- 정지/관찰가능·root거리 중앙값: SIT15/26·1.60m,CLIMB46/58·2.02m,ON_TOP185/220·3.26m(payload1.45m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초 기준.
- 3회 정상 완료·CPU/GPU/JSON 일치·기존 초기장면/크기/거리/물리관측분리·통합정지기록·16k 환경설정 동일·원본PTH/운영401파일/기존학습2PID 보존 검증 통과. [13–17k 추세](output_etc/check/climb217000_common_sumoff_physics_20261109/report.txt),[CSV](output_etc/check/climb217000_common_sumoff_physics_20261109/trends.csv),[정지](output_etc/check/climb217000_common_sumoff_physics_20261109/stops.csv),[검증](output_etc/check/climb217000_common_sumoff_physics_20261109/verification.json).

### total28k — XY80/100·정지 평가

- GPU7에서 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·512환경×3seed×900step 완료, 동일512 기록으로 정지를 집계했다.
- XY80 후행/동일쌍 선행AT(%): SIT72.38/98.80,CLIMB66.90/97.69,ON_TOP69.22/98.02,행동균등69.50/98.17,전체69.47±2.09/98.13. XY100은74.82/98.80,67.74/97.69,72.24/98.02,균등71.60/98.17,전체71.92±2.57/98.13.27k 전체73.15/75.71 대비 각각−3.68/−3.79pp이며 기존27k 최고 유지. 선행AT는97.55→98.13으로 상승했다.
- 정지/관찰가능·root거리 중앙값: SIT18/44·1.53m,CLIMB37/45·1.63m,ON_TOP181/245·3.15m(payload1.42m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초 기준이다.
- 3회 정상 완료·CPU/GPU/JSON 일치·기존 초기장면/크기/거리/물리관측분리·통합정지기록·27k 환경설정 동일·원본PTH/운영401파일/기존학습2PID 보존 검증 통과. [추세](output_etc/check/total28000_common_sumoff_physics_20261109/report.txt),[CSV](output_etc/check/total28000_common_sumoff_physics_20261109/trends.csv),[정지](output_etc/check/total28000_common_sumoff_physics_20261109/stops.csv),[검증](output_etc/check/total28000_common_sumoff_physics_20261109/verification.json).

### at1mm21k — 동일조건 정식 평가

- 새21000 번호 저장본을 GPU6에서 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·512환경×3seed×900step으로 평가하고 같은 기록에서 정지를 집계했다.
- XY80 후행/동일쌍 선행AT(%): SIT69.95/98.04,CLIMB58.96/94.83,ON_TOP66.44/96.87,행동균등65.12/96.58,전체65.73±1.46/96.74. XY100은75.25/98.04,59.54/94.83,68.03/96.87,균등67.61/96.58,전체67.99±.87/96.74.20k전체65.94→65.73으로 정체하며 모델 최고16k68.01을 갱신하지 못했다.
- 정지/관찰가능·root거리 중앙값: SIT21/43·1.57m,CLIMB34/38·1.59m,ON_TOP196/244·3.16m(payload1.41m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초 기준.
- 3회 정상 완료·CPU/GPU/JSON 일치·초기장면/크기/거리/물리관측분리·통합정지기록·20k 환경설정 동일·원본PTH/운영401파일/기존학습2PID 보존 검증 통과. [추세·XY80/100](output_etc/check/at1mm21000_common_sumoff_physics_20261109/report.txt),[CSV](output_etc/check/at1mm21000_common_sumoff_physics_20261109/trends.csv),[정지](output_etc/check/at1mm21000_common_sumoff_physics_20261109/stops.csv),[검증](output_etc/check/at1mm21000_common_sumoff_physics_20261109/verification.json).

### CLIMB +2m16k — XY80/100 및13–16k 추세

- GPU7에서 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·512환경×3seed×900step 완료. 같은512 기록으로 정지를 집계했다.
- XY80 후행/동일쌍 선행AT(%): SIT64.64/97.82,CLIMB66.74/97.49,ON_TOP69.39/97.38,행동균등66.92/97.56,전체67.66±2.14/97.50.15k 균등71.30/97.01·전체71.55/96.81보다 하락,특히SIT−7.35pp. XY100은 SIT71.22/97.82,CLIMB67.31/97.49,ON_TOP71.92/97.38,균등70.15/97.56,전체70.74±1.88/97.50. 두기준 모두15k가 현재 모델 최고다.
- 정지/관찰가능·root거리 중앙값: SIT15/25·1.45m,CLIMB45/53·2.00m,ON_TOP206/227·3.36m(payload1.41m). CLIMB13→16k 중앙값1.96→2.04→1.99→2.00m로유사하다. 정지기준 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초.
- CPU/GPU/JSON 일치·기존 초기장면·크기/거리/물리관측분리·통합정지기록·15k와 환경설정 동일·원본PTH/운영401파일/기존학습2PID 보존 검증 통과. [XY80/100 추세](output_etc/check/climb216000_common_sumoff_physics_20261109/report.txt),[CSV](output_etc/check/climb216000_common_sumoff_physics_20261109/trends.csv),[정지](output_etc/check/climb216000_common_sumoff_physics_20261109/stops.csv),[검증](output_etc/check/climb216000_common_sumoff_physics_20261109/verification.json).

### total27k·at1mm20k — 1k 단위 평가와 동일조건 추세

- total27k GPU7,at1mm20k GPU6에서 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·각512환경×3seed×900step으로 평가했다. 정지는 같은512 기록에서 집계했다.
- Plane Ever XY80 후행/동일쌍 선행AT(%): total27k SIT76.02/98.80,CLIMB70.70/96.65,ON_TOP72.99/97.39,행동균등73.23/97.62,전체73.15±1.68/97.55. 동일조건 기존 total22k 균등69.81,CLIMB+2m15k71.30을 넘어 비교3모델 중 최고이나 단일 저장본 상승이다. at1mm20k69.08/97.35,61.50/97.74,66.26/97.28,균등65.61/97.46,전체65.94±.78/97.38로19k전체65.95와 유사하다. XY100전체 total75.71±2.15(AT97.55),at1mm68.37±1.42(97.38).
- 정지/관찰가능·root거리 중앙값: total27k SIT11/25·1.74m,CLIMB29/41·1.59m,ON_TOP189/241·3.27m(payload1.42m);at1mm20k18/31·1.56m,24/30·1.65m,195/241·3.23m(payload1.42m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초 기준이다.
- 6회 정상 완료, CPU/GPU/JSON 일치·과거 초기장면·크기/거리/물리관측분리·통합정지기록 일치·모델별 환경설정 동일·원본PTH/운영401파일/기존학습2PID 보존 검증 통과. [동일조건 행동별 추세](output_etc/check/total27_at1mm20_20261109/report.txt),[전체CSV](output_etc/check/total27_at1mm20_20261109/trends.csv),[정지](output_etc/check/total27_at1mm20_20261109/stops.csv),[검증](output_etc/check/total27_at1mm20_20261109/verification.json). 이전 일반거리·near50 결과와 혼합하지 않았다.

### 3명 AT→ON_TOP→ON_TOP 고정 장면 viewer

- 사용자 요청으로 O2 높이0.20→0.25m 변경. 현재 O0/O1은0.40×0.40×0.25m, O2는0.30×0.30×0.25m. 가상 받침 중심Z0.125/0.375m 유지, O2 최종 중심Z는 높이에 따라0.625m로 계산된다. GPU7 headless60step 정상 완료·크기/가상pose 진단 통과. [기록](output_etc/check/before_chain3_height25_20261109/chain_viewer.json). 성공 개선 여부는 미확인.
- 사용자 요청으로 O0/O1 XY를 모두0.50→0.40m로 축소했다. 높이0.25m·O2 0.30×0.30×0.20m·total22k·가상 중심Z 유지. GPU7 headless60step 정상 완료와 크기/가상pose 검증 통과. [기록](output_etc/check/before_chain3_base40_20261109/chain_viewer.json). 성공 개선 여부는 미확인.
- 아래 상자 돌출부의 발 접촉 가설 비교를 위해 O0 XY만0.60→0.50m로 줄여 O1과 맞췄다. 현재 크기는0.50×0.50×0.25 / 0.50×0.50×0.25 / 0.30×0.30×0.20m. total22k와 가상 중심Z는 유지. GPU7 headless60step 정상 완료, 실제 크기·가상pose 진단 통과. [기록](output_etc/check/before_chain3_equal_base_20261109/chain_viewer.json). 성공 개선은 아직 미확인.
- C의 내려놓기 여유 비교를 위해 마지막 payload XY만0.40→0.30m로 축소했다(높이0.20m 유지). total22k·받침 크기·가상 중심Z0.125/0.375m는 유지했다. GPU7 headless60step 정상 완료, 실제 크기와 가상pose 검증 통과. [기록](output_etc/check/before_chain3_small_payload_20261109/chain_viewer.json). 동작 개선 여부는 아직 검증하지 않았다.
- 사용자 요청으로 기본 checkpoint·학습 YAML 선택을 total22k로 변경했다. GPU7 headless60step 정상 완료와 연쇄/가상pose 진단 통과. [기록](output_etc/check/before_chain3_total22k_20261109/chain_viewer.json). 아래 CLIMB+2m15k 검증은 최초 구현 당시 기록이다.
- `before_chain3_{test,vnc}.sh`, `view_before_chain.py`, `utils/before_chain_viewer.py`를 추가했다. 별도 프로세스에서만 ON_TOP 선행·순차 의존·예상 높이의 가상 받침을 확장한다. A가 O0 운반, B가 O1을 O0 위에, C가 O2를 O1 위에 놓는다. 기존 학습 코드·설정·체크포인트는 수정하지 않았다.
- 기본은 CLIMB+2m15k,3명·1환경·6토큰,고정 크기/배치,1800step×10회다. C의 대기Ever 해제에는 A 조건을 만족한 B ON_TOP 성공을 사용한다. 실행법과 제한은 [USAGE](markdowns/USAGE.md#before-chain3-viewer)에 기록했다.
- CPU2테스트(순서 변경·aliasXYZ만 변경·연쇄gate)와 shell syntax 통과. GPU7 실제checkpoint headless120step×2회 및 VNC300step×1회 정상 완료, 연결·가상 위치/회전·원본pose 보존·유한action 확인. 화면저장 보조 API 오류는 제거 후 VNC 재검증했다. [실제 화면](output_etc/check/before_chain3_20261109/scene.png), [실행 기록](output_etc/check/before_chain3_20261109/render/chain_viewer.json). 이는 실행 검증이며 3단 쌓기 성공률 검증은 아니다.

### at1mm CLIMB +2m 13–15k — 1k 단위 평가

- `stage2_before_total_at1mm_climb2`의 미평가13/14/15k를 GPU6/7에서 평가했다. 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·각512환경×3seed×900step, Plane Ever XY80 후행 동시 성공이다. 정지는 같은512 기록에서 집계하며 별도64 실행은 없다. 평가 hook의 초기 반경 기록에도 CLIMB 전용+2m를 반영했다(샘플링·모델 변경 없음).
- 후행/동일쌍 선행AT(%):13k SIT75.00/98.80,CLIMB57.76/98.27,ON_TOP63.38/96.32,행동균등65.38/97.80;14k68.97/97.86,58.37/97.74,63.07/96.22,균등63.47/97.27;15k71.99/97.17,69.79/97.42,72.12/96.44,균등71.30/97.01.15k 전체71.55±1.76(AT96.81),XY100 전체73.06±1.63(96.81).
- 같은조건 at1mm15k CLIMB60.30(AT97.17) 대비+9.49pp.15k 행동균등71.30은 최근 비교한 total22k69.81(AT98.51),at1mm16k68.27(95.92)보다 높다.13/14k는 낮았으므로 단일 저장본으로 반경 확대의 인과효과나 지속 상승을 확정하지 않는다.
- 정지/관찰가능·root 목표거리 중앙값:13k SIT16/33·1.50m,CLIMB54/70·1.96m,ON_TOP198/240·3.14m(payload1.40m);14k15/34·1.64m,53/70·2.04m,196/237·3.21m(payload1.41m);15k28/46·1.43m,51/71·1.99m,196/228·3.28m(payload1.40m). at1mm15k CLIMB38/47·1.66m에 비해 정지거리가 길다. 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초, 개인낙상 방식이다.
- 9회 정상 완료, CPU/GPU/JSON 일치·과거 초기 과제장면 동일·크기/거리/물리관측분리·통합 정지기록 일치 검증 통과. at1mm 대비 환경 차이는 CLIMB 반경과 재개 경로뿐임을 확인했다. 원본PTH·운영401파일 보존. [보고서](output_etc/check/climb2_1k_batch_20261109/report.txt),[추세·SD](output_etc/check/climb2_1k_batch_20261109/trends.csv),[정지](output_etc/check/climb2_1k_batch_20261109/stops.csv),[검증](output_etc/check/climb2_1k_batch_20261109/verification.json).

### total23–26k·at1mm17–19k — 1k 단위 일괄 평가

- GPU6에서 at1mm17/18/19k, GPU7에서 total23/24/25/26k를 평가했다. 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·각512환경×3seed×900step, Plane Ever XY80 후행 동시 성공 기준이다. 정지는 같은512 기록에서 집계하며 별도64환경은 없다.
- 후행 행동균등 평균/동일쌍 선행AT 행동균등 평균(%): total22k69.81/98.51→23k63.01/97.29→24k66.81/97.62→25k67.28/97.92→26k63.75/98.97. at1mm16k68.27/95.92→17k65.10/96.98→18k67.62/96.96→19k65.74/97.34. 두 모델 모두 이전 최고를 넘지 못했고, 연속 상승보다 저장본별 변동이 나타났다.
- 최신 total26k 후행/동일쌍AT: SIT68.50/99.49,CLIMB58.58/98.88,ON_TOP64.17/98.53,전체64.02/98.84. at1mm19k는72.82/97.17,58.54/98.59,65.85/96.25,전체65.95/96.91. 각 seed SD·XY100·모든 중간 저장본 결과는 아래 CSV에 보존했다.
- 모든7개 정지/관찰가능 건수와 root/payload 목표거리 중앙값·사분위수를 집계했다. 최신 total26k SIT7/20·1.80m,CLIMB24/34·1.67m,ON_TOP180/232·3.28m(payload1.41m);at1mm19k SIT14/23·1.54m,CLIMB26/33·1.64m,ON_TOP200/240·3.29m(payload1.40m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초, 개인낙상 방식이다.
- 21회 정상 완료, CPU/GPU/JSON 일치·과거 초기 과제장면 동일·크기/거리/물리관측분리·통합 정지기록 일치 검증 통과. 각 이전 저장본과 환경설정 동일, 원본PTH·운영401파일·기존학습2PID 보존 확인. [보고서](output_etc/check/total_at1mm_1k_batch_20261109/report.txt), [추세·SD](output_etc/check/total_at1mm_1k_batch_20261109/trends.csv), [정지](output_etc/check/total_at1mm_1k_batch_20261109/stops.csv), [검증](output_etc/check/total_at1mm_1k_batch_20261109/verification.json). 개별 평가 폴더는 batch `plan.json`에 기록했다.

### at1mm16k·total22k — 공통·합일치·near50 해제·물리 제외 동시 평가

- at1mm16000은 GPU6, total22000은 GPU7에서 동시에 평가했다. 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·각512환경×seed0/1/2×900step, 정지도 같은512 기록에서 집계하며 별도64 실행은 없다.
- Plane Ever XY80 후행 동시 평균±SD%(동일쌍 선행AT Ever): at1mm16k SIT72.17±3.00(95.13), CLIMB65.28±4.17(97.13), ON_TOP67.35±3.72(95.50), 전체68.01±2.35(95.75), 행동균등68.27(AT균등95.92).15k 균등61.47(97.09)에서+6.80pp로 회복했다.15k 결과에 근거한 중단 추천은 보류하며,1mm의 인과적 이득은 확정하지 않는다.
- total22k SIT77.04±3.15(99.02), CLIMB66.88±3.52(98.89), ON_TOP65.50±1.40(97.62), 전체68.52±1.37(98.19), 행동균등69.81(98.51).21k 균등67.61(98.32)보다+2.20pp: SIT+9.30·CLIMB+1.32·ON_TOP−4.02pp. 같은궤적 XY100 전체는 at1mm70.39±2.49(95.75),total70.16±2.09(98.19).
- 정지/관찰가능·root 목표거리 중앙값: at1mm SIT23/47·1.52m,CLIMB33/43·1.61m,ON_TOP202/262·3.12m(payload1.38m); total SIT15/29·1.70m,CLIMB22/32·1.57m,ON_TOP187/252·3.23m(payload1.39m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초, 개인낙상 방식이다.
- 6회 정상 완료(최장160초), CPU/GPU/JSON 일치·기존 초기 과제장면 동일·크기/거리/물리관측분리·통합 정지기록 일치 검증 통과. 각 이전 저장본과 환경설정 동일, 원본PTH·운영401파일·기존학습2PID 보존 확인. [at1mm 보고서](output_etc/check/at1mm16000_common_sumoff_physics_20261109/report.txt), [total 보고서](output_etc/check/total22000_common_sumoff_physics_20261109/report.txt). 각 폴더의 `verification.json`, `physics_only/stops512/stops.json`에 검증·정지 결과를 저장했다.

## 2026-11-08

### total21k — 공통·합일치·near50 해제·물리 distractor 제외

- 새21000 저장본을 GPU7에서 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·512환경×seed0/1/2×900step으로 평가했다. 정지는 같은512 궤적에서 집계했으며 별도64환경 실행은 없다.
- Plane Ever XY80 후행 동시 평균±SD%(동일쌍 선행AT Ever): SIT67.73±4.59(98.29), CLIMB65.56±1.89(99.17), ON_TOP69.53±2.23(97.49), 전체68.26±2.15(98.02), 행동균등67.61(AT균등98.32). 같은조건20k 전체63.48(96.92)/균등63.50(97.44)에서 회복했다.12k 전체67.25/균등67.09,19.5k66.71/66.75보다 높지만 지속 상승 여부는 추가 저장본이 필요하다.
- 같은궤적 XY100 전체70.58±1.83(AT98.02). 정지/관찰가능·root 목표거리 중앙값: SIT15/28·1.40m, CLIMB20/28·1.64m, ON_TOP209/237·3.34m(payload1.42m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초, 개인낙상 방식이다.
- 3회 정상 완료(최장268초), CPU/GPU/JSON 일치·과거 초기 과제장면 동일·거리/크기/물리관측분리·통합 정지기록 일치 검증 통과. 원본PTH·운영401파일·기존학습2PID 보존 확인. [보고서](output_etc/check/total21000_common_sumoff_physics_20261108/report.txt), [비교](output_etc/check/total21000_common_sumoff_physics_20261108/comparison.csv), [정지](output_etc/check/total21000_common_sumoff_physics_20261108/physics_only/stops512/stops.json), [검증](output_etc/check/total21000_common_sumoff_physics_20261108/verification.json).

### at1mm15k — 공통·합일치·near50 해제·물리 distractor 제외

- GPU7에서 최신 번호 저장본15000을 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·512환경×seed0/1/2×900step으로 평가했다. 같은512 궤적에서 정지를 집계하고 별도64환경은 실행하지 않았다. AT 보상1mm와 평가/후행gate20cm는 분리 유지했다.
- Plane Ever XY80 후행 동시 평균±SD%(동일쌍 선행AT Ever): SIT60.21±1.97(97.57), CLIMB60.30±6.84(97.17), ON_TOP63.91±2.57(96.53), 전체62.29±2.21(96.92), 행동균등61.47(AT균등97.09). 동일조건 출발점 total12k 전체67.25(97.27)/균등67.09(97.57) 대비 하락하며 SIT 차이가 가장 크다. 이전 at1mm14k 일반거리·near50 결과와 직접 추세 비교하지 않는다.
- 같은궤적 XY100 전체65.94±2.07(AT96.92). 정지/관찰가능·root 목표거리 중앙값: SIT25/41·1.66m, CLIMB38/47·1.66m, ON_TOP199/237·3.19m(payload1.42m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초, 개인낙상 방식이다.
- 3회 정상 완료(최장141초), CPU/GPU/JSON 일치·초기 과제장면 동일·거리/크기/물리관측분리·통합 정지기록 일치 검증 통과. 원본PTH·운영401파일·기존학습2PID 보존 확인. [보고서](output_etc/check/at1mm15000_common_sumoff_physics_20261108/report.txt), [비교](output_etc/check/at1mm15000_common_sumoff_physics_20261108/comparison.csv), [정지](output_etc/check/at1mm15000_common_sumoff_physics_20261108/physics_only/stops512/stops.json), [검증](output_etc/check/at1mm15000_common_sumoff_physics_20261108/verification.json).

### total20k — 공통·합일치·near50 해제·물리 distractor 제외

- GPU6에서 total20000 고정본을 공통 교집합·합일치·near50 해제·대소제약 없음·distractor 물리만 제외/관측 유지·512환경×seed0/1/2×900step으로 평가했다. 정지 진단은 같은512 궤적에서 집계했으며 별도64환경 실행은 없다.
- Plane Ever XY80 후행 동시 평균±SD%(동일쌍 선행AT Ever): SIT67.59±2.46(98.07), CLIMB59.77±4.85(98.33), ON_TOP63.13±2.88(95.91), 전체63.48±1.99(96.92), 행동균등63.50(AT균등97.44). 같은조건19.5k 전체66.71(97.32)/균등66.75(97.70) 대비 각각−3.23/−3.26pp. 단일 저장본 하락으로 지속 하락을 단정하지 않는다.
- 같은궤적 XY100 전체66.22±1.41(AT96.92). 통합 정지/관찰가능 및 root 목표거리 중앙값: SIT19/32·1.65m, CLIMB22/28·1.64m, ON_TOP209/249·3.28m(payload1.40m). 선행Ever 전 활성·rootXYZ≤.3m/s 연속1초, 개인낙상 방식이며 과거 native64와 구분한다.
- 3회 정상 완료(최장190초), CPU/GPU/JSON 점수 일치·초기 과제장면 동일·거리/크기/near 조건·물리/관측분리·통합 정지기록과 독립 기록 일치 검증 통과. 원본PTH·운영401파일·기존학습2PID 보존 확인. [보고서](output_etc/check/total20000_common_sumoff_physics_20261108/report.txt), [비교](output_etc/check/total20000_common_sumoff_physics_20261108/comparison.csv), [정지](output_etc/check/total20000_common_sumoff_physics_20261108/physics_only/stops512/stops.json), [검증](output_etc/check/total20000_common_sumoff_physics_20261108/verification.json).

### 정지 진단을 정식 512환경 평가에 통합

- 사용자 요청에 따라 별도64환경 정지 실행을 없애고 같은512환경×3seed 궤적에서 집계한다. `formal_eval.py`에 root XYZ 속력·대기 활성/해제 기록을 추가하고 `evaluate.py` 요약에 정지 집계를 연결했다. 학습·행동·보상·종료·성공 조건은 변경하지 않았다.
- `before_stop_evaluation.py` 및 `summarize_before_stops.py`는 첫 연속1초 정지 거리·관찰가능/정지 건수·사람/payload 거리·동일쌍 선행AT Ever를 CPU로 집계한다. 과거 `waiting_trace.pt`도 지원하며, 기록이 없다고 추가 시뮬레이션을 실행하지 않는다. 개인낙상 방식의 정식 평가와 과거 장면낙상64환경 진단은 구분한다.
- AGENTS·평가 참고문서·USAGE·config 공통 규칙·structure를 갱신하고, 향후 조건별 실행용512전용 runner 예시를 준비했다. 과거 평가 runner와 결과는 보존했다.
- CPU 테스트27개 통과(모의 정식 rollout 포함), 기존512×3seed 기록 집계1.89초, 과거64환경 저장 기록의 정지 건수/거리 재현 및 `git diff --check` 통과. 새 GPU 시뮬레이션은 실행하지 않았다. [검증](output_etc/check/before_stop512_integration_20261108/cpu_validation.json), [테스트](output_etc/check/before_stop512_integration_20261108/tests.log).

### total19.5k — 공통·합일치·near50 해제·distractor 물리만 제외

- GPU6에서 total19500 고정본을 공통 교집합·합일치·near50 해제·거리합 대소제약 없음·distractor 물리만 제외/관측 유지·512환경×seed0/1/2×900step으로 평가했다. 별도 같은조건64환경 seed0 정지 진단 포함4회 완료했다. 사용자가 함께 요청한 at1mm14k 기본 평가는 직전 완료 기록을 재사용했다.
- XY80 후행 동시 평균±SD%(동일쌍 선행AT Ever): SIT70.71±1.96(98.55),CLIMB63.29±2.20(98.00),ON_TOP66.26±.80(96.56),전체66.71±.97(97.32),행동균등66.75(선행AT균등97.70). 동일조건 total12k 전체67.25±4.53(97.27)/균등67.09(97.57),18k62.94±1.36(96.05)/균등62.71(96.58)와 비교했다.19.5k는18k 대비 회복했지만12k를 넘지는 못했다. 기본평가와 여러조건이 달라 distractor 단독효과로 해석하지 않는다.
- 같은궤적 XY100 재채점은 SIT74.78±1.57(98.55),CLIMB63.90±1.27(98.00),ON_TOP68.15±1.38(96.56),전체68.85±1.07(97.32),행동균등68.94(97.70). 높이와선행20cm current gate는 그대로다.
- 정지/관찰가능: SIT0/0,CLIMB1/1·root.36m,ON_TOP9/9·root2.77m/payload1.47m. 선행Ever 전 페널티활성·rootXYZ≤.3m/s 연속1초 첫시점, formal 개인낙상동결 방식으로 일반 native 장면낙상 진단과 별도다. SIT/CLIMB 표본이 부족해 대표정지거리로 해석하지 않는다.
- CPU/GPU/JSON 정확일치·XY80 점수복원·XY100 성공손실0·finite900frame·제거실행각901회 물리/관측분리 검증 통과. 과거 공통 초기 사람/과제상자/목표/역할 정확일치·관측 distractor 반올림오차1e-4m 이내·nearflag0/거리/크기 검사,원본PTH·운영398파일 보존 및 기존학습2PID 생존 확인. 운영 코드/학습설정 수정 없음.
- [보고서](output_etc/check/total19500_common_sumoff_physics_20261108/report.txt),[동일조건 비교](output_etc/check/total19500_common_sumoff_physics_20261108/comparison.csv),[정지](output_etc/check/total19500_common_sumoff_physics_20261108/stops.csv),[검증](output_etc/check/total19500_common_sumoff_physics_20261108/verification.json).

### total19.5k·at1mm14k — 최신 기본 평가·정지 진단

- 조회 시 최신 번호 저장본 total19500/at1mm14000을 각각 GPU6/7에서 일반거리·near50 유지·행동별 교집합·distractor 포함·512환경×seed0/1/2×900step·Plane Ever XY80으로 평가했다. 모델별 native64 seed0 정지 진단까지 총8회 완료했다. at1mm13.5k는 최신 요청 대상이 아니므로 추가 평가하지 않았다.
- total19.5k 후행 동시 평균±SD%(동일쌍 선행AT Ever): SIT67.33±2.06(99.31),CLIMB57.47±3.34(94.85),ON_TOP66.02±1.87(95.30),전체64.57±2.16(96.16). 19k 대비전체+3.09pp,행동균등59.49→63.61(선행AT균등95.55→96.49). total 내 전체/행동균등/SIT/CLIMB 최고를 갱신했다. 기존전체 최고12k62.99(AT96.46) 대비+1.58pp이며 공통·합일치 조건은 아직 미평가다.
- at1mm14k는 SIT63.37±3.05(97.57),CLIMB49.25±2.52(91.48),ON_TOP61.44±3.16(93.50),전체59.38±1.17(94.08).13k55.14(95.34) 대비전체+4.24pp·행동균등52.32→58.02(선행AT균등95.83→94.18)로 회복했으나 출발 total12k전체62.99/균등59.67에는 미달한다. 보상Z1mm는 평가/후행20cm gate와 분리 유지했다.
- native 정지/관찰가능·root 목표거리 중앙값:total SIT8/8·1.40m,CLIMB5/5·1.74m,ON_TOP12/14·2.12m(payload1.21m);at1mm SIT6/6·1.18m,CLIMB6/6·1.48m,ON_TOP14/15·1.97m(payload1.21m). 선행Ever 전 페널티활성·rootXYZ≤.3m/s 연속1초 첫시점, 장면낙상 종료로 정식 개인낙상 지표와 별도다.
- CPU/GPU/JSON 정확일치·finite900frame·과거 일반평가 초기상태 정확일치·near50/크기/거리·native 진입/Ever해제 검증 통과. frozen encoder73개/RMS·원본PTH·운영398파일 보존, 기존학습2PID 생존 확인. 운영 코드·학습 설정 변경 없음.
- [total 보고서](output_etc/check/total19500_general_20261108/report.txt),[total 행동균등 추세](output_etc/check/total19500_general_20261108/macro_trend.csv),[at1mm 보고서](output_etc/check/at1mm_14000_general_20261108/report.txt),[at1mm 추세](output_etc/check/at1mm_14000_general_20261108/trend.csv),[total 검증](output_etc/check/total19500_general_20261108/verification.json),[at1mm 검증](output_etc/check/at1mm_14000_general_20261108/verification.json).

### total18.5k·19k — 기본 평가·행동균등 추세

- GPU6만 사용해 고정18500/19000 저장본을 일반거리·near50 유지·행동별 교집합·distractor 포함·512환경×seed0/1/2×900step·Plane Ever XY80으로 각각 평가했다. 각 native64 seed0 정지 진단 포함 총8회 정상 완료했다. 사용자가18.5k 후19k를 추가 요청해 순서대로 실행했다.
- 18.5k 후행 동시 평균±SD%(동일쌍 선행AT Ever): SIT62.21±4.69(97.49), CLIMB45.91±4.38(93.18), ON_TOP65.26±3.29(95.06), 전체60.56±2.55(95.25). 19k는59.81±.86(98.04),53.47±3.22(92.59),65.18±2.97(96.02),전체61.48±2.38(95.82). 18.5→19k CLIMB+7.56pp·SIT−2.40pp·ON_TOP−.08pp로 CLIMB가 회복했다.
- total 행동균등 후행 평균/동일쌍 선행AT 행동균등 평균은12k59.67/96.36→18k59.07/94.97→18.5k57.79/95.24→19k59.49/95.55. 평가된 total에서 CLIMB는19k 최고지만 전체 및 행동균등 최고는12k 유지다. 단일저장본 회복을 지속 상승으로 단정하지 않는다.
- native 정지/관찰가능·root 목표거리 중앙값:18.5k SIT5/5·1.22m,CLIMB4/4·1.67m,ON_TOP13/14·2.08m(payload1.22m);19k SIT7/8·1.19m,CLIMB5/5·1.79m,ON_TOP13/14·1.99m(payload1.22m). 선행Ever 전 페널티활성·rootXYZ≤.3m/s 연속1초 첫시점, 장면낙상 종료로 formal 개인낙상과 별도다.
- 각 CPU/GPU/JSON 정확일치·finite900frame·과거 일반평가 초기상태 정확일치·near50/크기/거리·native 진입/Ever해제 검증 통과. frozen encoder73개/RMS·설정·원본PTH·운영398파일 보존, 기존학습2PID 생존 확인. 운영 코드/학습 설정 수정 없음.
- [18.5k 보고서](output_etc/check/total18500_general_20261108/report.txt),[19k 보고서](output_etc/check/total19k_general_20261108/report.txt),[행동별 추세](output_etc/check/total19k_general_20261108/full_action_history.csv),[행동균등 추세](output_etc/check/total19k_general_20261108/macro_trend.csv),[19k 정지](output_etc/check/total19k_general_20261108/stops.csv),[검증](output_etc/check/total19k_general_20261108/verification.json).

### at1mm13k — 기본 평가·12.5k 및 출발점 추세

- GPU7만 사용해 total12k 재개 후1000epoch인 at1mm13000 고정본을 일반거리·near50 유지·행동별 교집합·distractor 포함·512환경×seed0/1/2×900step·Plane Ever XY80으로 평가했다. AT 보상Z1mm와 별개로 평가/후행gate의20cm 기준은 유지했다. native64 seed0 정지 진단까지4회 완료했다.
- 후행 동시 평균±SD%(같은쌍 선행AT Ever): SIT58.03±.79(98.76), CLIMB39.06±6.38(94.57), ON_TOP59.88±3.31(94.15), 전체55.14±.71(95.34). total12k→at1mm12.5k→13k 전체62.99→59.32→55.14(AT96.46→95.34→95.34), 행동균등 평균59.67→56.09→52.32(같은쌍AT 행동균등96.36→95.27→95.83). 12.5k 대비SIT 회복, CLIMB/ON_TOP 하락이며 원인 인과는 미검증이다.
- 단독 SIT97.53±3.23,CLIMB99.26±.75,AT97.44±.81,ON_TOP97.54±1.66,PUSH96.29±.10%. native 정지/관찰가능 및 root 목표거리 중앙값 SIT6/6·1.17m,CLIMB4/4·1.76m,ON_TOP14/14·2.08m(payload1.25m). 선행Ever 전 페널티활성·rootXYZ≤.3m/s 연속1초 첫시점, 장면낙상 종료이며 formal 개인낙상 점수와 별도다.
- CPU/GPU/JSON 정확일치·900frame finite·과거 일반평가 초기상태 정확일치·near50/크기/거리·native 진입/Ever해제 검증 통과. frozen encoder73개/RMS·원본checkpoint·운영398파일 보존, 기존학습2PID 생존 확인. 운영 코드/학습 config 수정 없음.
- [보고서](output_etc/check/at1mm_13000_general_20261108/report.txt),[추세](output_etc/check/at1mm_13000_general_20261108/trend.csv),[정지](output_etc/check/at1mm_13000_general_20261108/stops.csv),[검증](output_etc/check/at1mm_13000_general_20261108/verification.json).

### at1mm12.5k — 기본 평가·total12k 출발점 비교

- GPU7에서 total12k 재개 후500epoch인 at1mm12500 고정본을 일반 goalroute·near50·행동별 교집합·distractor 포함·512환경×seed0/1/2×900step·Plane Ever XY80으로 평가했다. AT 보상만Z1mm이며 정식 AT 공간 성공/후행gate의20cm 기준은 기존과 동일하다. 별도 native64 seed0 정지 진단 포함4회 완료했다.
- 후행 동시 평균±SD%(동일쌍 선행AT Ever): SIT54.54±3.01(99.09), CLIMB48.06±1.75(91.63), ON_TOP65.65±2.41(95.10), 전체59.32±.57(95.34). 출발 total12k64.72/45.64/68.64/62.99(AT98.73/94.04/96.32/96.46) 대비 −10.17/+2.42/−2.99/−3.67pp. 첫500epoch 결과이며 지속 추세나1mm의 단독 인과효과로 단정하지 않는다.
- 단독 Plane Ever SIT94.36±2.39, CLIMB98.77±.44, AT97.30±1.20, ON_TOP97.72±1.86, PUSH92.56±1.17%. native 정지/관찰가능과 root 목표거리 중앙값: SIT6/6·1.13m, CLIMB5/5·1.65m, ON_TOP13/14·2.13m(payload1.21m). 선행미완료·페널티활성·rootXYZ≤.3m/s 연속1초 첫시점, 장면낙상 종료이며 정식 개인낙상 점수와 별도다.
- CPU/GPU/JSON 정확일치·900frame finite·기존 일반평가 초기상태 정확일치·near50/크기/거리 검증 통과. frozen encoder73개/RMS·원본checkpoint·운영398파일 보존, 기존학습2PID 생존 확인. 설정 차이는AT 보상1mm·실험명·재개정보뿐이며 운영 코드 변경 없음.
- [보고서](output_etc/check/at1mm_12500_general_20261108/report.txt),[출발점 비교](output_etc/check/at1mm_12500_general_20261108/comparison.csv),[정지](output_etc/check/at1mm_12500_general_20261108/stops.csv),[검증](output_etc/check/at1mm_12500_general_20261108/verification.json),[설정 비교](output_etc/check/at1mm_12500_general_20261108/checkpoint_verification.json).

### total18k — 공통·합일치·near50 해제, distractor 물리만 제외 비교

- GPU7만 사용해 동일18000 저장본을 공통 교집합·합일치·near50 해제·거리합 대소제약 없음·512환경×seed0/1/2×900step으로 평가했다. distractor 유지/물리만 제외 두 조건이며, 제거 조건도 관측 토큰은 원래 sampled pose/zero velocity로 유지한다. 조건별64환경 seed0 정지 진단 포함 총8회 완료했다.
- XY80 후행 동시 평균±SD%(동일쌍 선행AT Ever), 유지→물리만 제외: SIT65.99±3.38(96.58)→67.26±4.87(97.35), CLIMB54.66±5.11(96.89)→57.78±4.06(97.44), ON_TOP65.20±.64(95.49)→63.10±2.91(94.96), 전체63.18±1.75(96.05)→62.94±1.36(96.05). 전체−.24pp로 개선은 없으며, 행동별 방향도 다르다. XY100 재채점 전체65.28±1.95→64.57±1.25(각AT96.05).
- 정지/관찰가능: 유지 SIT0/0·CLIMB0/0·ON_TOP8/10, 제외 SIT0/0·CLIMB0/1·ON_TOP10/10. ON_TOP root 중앙값2.76→2.41m, payload각1.49m. 페널티활성·선행미완료 중rootXYZ≤.3m/s 연속1초 첫시점이며 공통/합일치/개인낙상동결 진단이다. SIT/CLIMB 표본 부족으로 정지거리 비교 불가.
- CPU/GPU/JSON 정확일치·900frame finite·각제거실행901회 물리/관측분리 확인, 초기사람/과제상자/목표/역할 정확일치·관측 distractor 반올림오차1e-4m 이내·거리/크기 검증 통과. 원본PTH·운영398파일 보존 및 기존학습2PID 생존 확인. 운영 코드/학습 config 수정 없음.
- [보고서](output_etc/check/total18k_common_nearoff_distractor_20261108/report.txt),[비교](output_etc/check/total18k_common_nearoff_distractor_20261108/comparison.csv),[정지](output_etc/check/total18k_common_nearoff_distractor_20261108/stops.csv),[검증](output_etc/check/total18k_common_nearoff_distractor_20261108/verification.json).

### total18k — 일반 near50 평가·정지 진단

- GPU6에서 고정18000 저장본을 일반 goalroute·near50·행동별 교집합·distractor 물리/관측 포함·512환경×seed0/1/2×900step·Plane Ever로 평가하고 native64 seed0 정지 진단을 완료했다. 공통/합일치/near해제/물리제거/at1mm/CLIMB2 옵션은 적용하지 않았다.
- 후행 동시 평균±SD% (같은 쌍 선행AT Ever): SIT64.83±4.58(97.64), CLIMB46.46±4.51(92.91), ON_TOP65.92±1.68(94.35), 전체61.70±.88(94.83).17k 대비 +1.63/−.98/+5.55/+3.30pp다. 전체 회복은 주로ON_TOP이며 CLIMB 개선은 없다. 전체 최고 total12k62.99±2.82(AT96.46)에는 미달한다.
- native 정지/관찰가능 및 root 목표거리 중앙값: SIT6/7·1.22m, CLIMB5/5·1.75m, ON_TOP13/14·1.92m(payload1.22m). 선행미완료·페널티활성 후XYZ≤.3m/s 연속1초 첫시점이며 formal 개인낙상동결 점수와 별도다.
- 4회 정상 완료, CPU/GPU/JSON 정확일치·900frame finite·과거7k와 초기장면 정확일치·near50/거리/크기 검증, frozen encoder73개/RMS·원본 PTH·운영398파일 보존 및 작업 시작 시 기존학습2PID 생존 확인. 평가 전용 파일 외 운영 코드 수정 없음.
- [보고서](output_etc/check/total18k_general_20261108/report.txt),[이전 비교](output_etc/check/total18k_general_20261108/comparison.csv),[추세](output_etc/check/total18k_general_20261108/full_action_history.csv),[정지](output_etc/check/total18k_general_20261108/stops.csv),[검증](output_etc/check/total18k_general_20261108/verification.json).

### at1mm CLIMB +2m — total12k 재개용 코드·컨피그 준비

- `stage2_before_total_at1mm_climb2`를 추가했다. 사용자 선택대로 기존 at1mm 재개와 같은 total12k에서 시작하며, BEFORE CLIMB 수행 edge만 `r_box+2m`로 변경한다. SIT/ON_TOP은+1.5m, ON_TOP payload 기준·사람XYZ 속력 감점·가중치1.0·AT 보상1mm·후행20cm Current/Ever gate·900step·모델/AMP/PPO는 유지한다.
- BeforeWait에 선택적 `climb_radius_buffer`를 추가하고 reset에서 실제 relation별 반경을 계산한다. 기존 설정에서는 scalar 공통 반경 경로를 유지한다. validator/계약·실험 등록·train/test/VNC/eval·문서를 연결했다.
- train wrapper는 total12k 원본을 보존하고 `before_total_at1mm_climb2_resume.py`로 초기 재개 복사본을 준비한다. 모델/optimizer/정규화/epoch/frame 유지, 새 실험의 저장본은 그대로 resume하며 다른 epoch의 total 출발은 거부한다. 기본 이관본은 wrapper 실행 시 생성되며 이번 작업에서는 본학습용 이관본을 생성하거나 학습을 실행하지 않았다.
- 관련 CPU134개 통과, 재개 변환 추가 후 신규 파일17개 테스트 통과(앞선16개와 중복). CLIMB 추가0.5m 구간·SIT/ON_TOP 보존·XYZ 감점·다중쌍/셔플·진입/해제 Ever·부분reset·단독/PUSH 무영향·잘못된 옵션 거부·모델/optimizer 등 재개 상태/캐시/원본 보존을 검사했다. 실제 total12k metadata 계약 dry check, 기존 포함36개 실험 설정·Python AST·shell5개 문법 통과.
- 사용자 요청대로 GPU 시뮬레이션/확인 학습/본학습은 실행하지 않았다. 실제 기동 검증은 미실시다. [환경](tokenhsi/data/cfg/multi_agent/stage2_before_total_at1mm_climb2.yaml),[재개 변환](tokenhsi/utils/before_total_at1mm_climb2_resume.py),[테스트](tokenhsi/tests/test_before_total_at1mm_climb2.py),[검증 로그](output_etc/check/total_at1mm_climb2_20261108/tests.log),[추가 테스트](output_etc/check/total_at1mm_climb2_20261108/new_tests.log),[설정/문법](output_etc/check/total_at1mm_climb2_20261108/static_checks.json),[실행법](markdowns/USAGE.md).

### fuv1 21k·total12k — distractor 물리만 제외, 관측 유지

- GPU6에서 공통 교집합·합일치·near50 해제·거리합 대소제약 없음·512환경×seed0/1/2×900step으로 두 모델을 평가했다. 각 모델 별도64환경 seed0 정지 진단까지 총8회 정상 완료. 사용자 정정에 따라 total21k가 아닌 기존 비교의 total12k를 사용했다.
- 평가 전용 hook으로 샘플링 완료 후 genuine distractor의 물리 actor만 장면 밖으로 옮겼다. 관측에는 원래 sampled pose/zero velocity/BPS 및 모든 토큰을 유지하고, 과제/alias 물리 상태·reward·gate·채점용 상태는 치환하지 않았다. 관측 distractor는 고정이며 움직이는 실제 장애물과 동일한 동적 관측은 아니다. 기존 물리·관측 동시 제거 실험과 구분한다.
- XY80 후행 동시 평균±SD% (같은 쌍 선행 AT Ever): fuv1 SIT61.23±4.50(97.35), CLIMB63.53±3.05(97.40), ON_TOP70.15±2.08(97.92), 전체66.60±1.94(97.67). total SIT71.62±7.29(98.04), CLIMB62.66±6.17(98.03), ON_TOP66.98±3.82(96.64), 전체67.25±4.53(97.27). 기존 전체64.10(AT96.69)/66.01(AT96.75) 대비 +2.49/+1.24pp로 표본 평균상 소폭 상승이며 모든 행동의 개선이나 통계적 유의성을 뜻하지 않는다.
- 동일 궤적 XY100 재채점 전체는 fuv1 68.68±2.38(AT97.67), total69.23±3.81(AT97.27). 기존66.67/68.28 대비 +2.01/+0.95pp. 높이/current AT 조건은 유지했다.
- 64환경 정지/관찰가능과 root 목표거리 중앙값: fuv1 SIT4/4·2.83m, CLIMB3/4·3.35m, ON_TOP24/24·3.95m(payload2.24m); total SIT0/1, CLIMB1/1·1.23m, ON_TOP9/10·2.81m(payload1.47m). 공통/합일치/near해제의 개인낙상동결 궤적이며 native 장면낙상 진단과 혼합하지 않는다.
- 전8회 CPU/GPU/JSON 정확일치·901회 관측/물리분리 검사·finite900frame 통과. 기존 장면의 사람/과제 상자/목표/역할/DOF는 정확일치, 관측 distractor는 reset 전후 world float 반올림 오차 최대0.0000611m 이내다. 원본 PTH·운영390파일·기존학습3개 보존 확인. 운영 코드/학습 config 수정 없음.
- [보고서](output_etc/check/fuv1_21k_total12k_physics_only_20261108/report.txt),[비교](output_etc/check/fuv1_21k_total12k_physics_only_20261108/comparison.csv),[정지](output_etc/check/fuv1_21k_total12k_physics_only_20261108/stops.csv),[검증](output_etc/check/fuv1_21k_total12k_physics_only_20261108/verification.json),[평가 전용 hook](output_etc/check/fuv1_21k_total12k_physics_only_20261108/physics_only.py).

### progress0 16k — 일반 평가·정지 진단·CLIMB 1mm 조건부 집계

- GPU6만 사용해 고정16000 저장본을 일반 goalroute/near50 유지·행동별 교집합·512환경×seed0/1/2×900step·Plane Ever로 평가했다. 별도 native64 seed0 정지 진단도 완료했다. 모델/physics/reward 변경 없음.
- 후행 동시 평균±표준편차(동일쌍 선행AT Ever 평균)%: SIT60.18±4.77(97.86),CLIMB52.00±7.49(91.22),ON_TOP65.39±2.76(94.66),전체61.33±2.66(94.71).15k 대비CLIMB+4.03pp·전체+0.77pp로 일부 회복했지만10k 최고전체62.47/CLIMB55.66에는 미달한다.
- CLIMB3seed 합산352쌍에서 선행AT성공321쌍의 후행182/321=56.70%,그중20cm+Z1mm Ever 충족271쌍181/271=66.79%,끝까지1mm미충족50쌍1/50=2.00%(각조건부동일쌍AT100%).1mm이후gate열림90frame이상257쌍은180/257=70.04%. 높이와 성공의 연관이며1mm 조건변경의 인과효과는 아니다.
- native 정지/관찰가능: SIT5/5(root중앙값1.18m),CLIMB5/5(1.69m),ON_TOP11/14(root1.92m·payload1.21m). 선행미완료·활성화후rootXYZ≤.3m/s 연속1초의 첫시점이며 formal 개인낙상 방식과 별도다.
- CPU/GPU/JSON점수 정확일치, 초기장면 이전7k와 정확일치·near50/거리/크기 검증, 동결encoder73개/RMS 불변, 원본checkpoint·production파일 보존, 기존학습3PID 생존 확인. [보고서](output_etc/check/progress0_16k_general_20261108/report.txt),[추세](output_etc/check/progress0_16k_general_20261108/full_action_history.csv),[CLIMB조건부](output_etc/check/progress0_16k_general_20261108/height/summary.csv),[검증](output_etc/check/progress0_16k_general_20261108/verification.json).

### CLIMB 저하 — total/progress0 1mm 조건부·동일장면·거리 집계

- 기존 일반 near50·행동별 교집합512환경×3seed에서 total12k/17k,progress0 10k/15k,fuv1 21k,fuv2 20k/22k 총21trace를CPU 재집계했다. 전부900frame 공통 창이며 각 모델CLIMB352쌍·모델간seed별 초기 상태 정확일치·원본전체horizon AT/후행Ever/동시 점수복원 확인. GPU 실행/운영변경 없음.
- 선행AT Ever 조건부 후행동시%는 total12k48.34,17k50.61,progress10k60.12,15k51.21. 그중 기존20cm+Z1mm를한번이라도 만족한 부분집합은58.62/66.01/68.57/62.31%,끝까지 미충족은10.00/0/8.70/3.23%다. 조건부 부분집합의 같은쌍 선행AT Ever는모두100%. 평균seed%가 아닌3seed합산건수 비율이다.
- progress10→15k 양쪽1mm성공 동일230장면에서 후행163→148,초기후행1–3m84쌍61→47,3–5m63쌍43→42,5–10m83쌍59→59. 근거리 실패 중1mm이후CurrentAT/alive/Ever해제 상태에서 rootXYZ≤.3·목표거리>r_box+.3을연속3초 유지한 사례8→21. 양쪽1mm이후gate열림90frame이상·시간순 성공으로 제한해도1–3m54/73→42/73으로 하락한다. 반경1.5m는학습중불변이므로반경수치의인과효과로단정하지않는다.
- fuv1 21k(+3m)와total17k(+1.5m) 양쪽1mm성공214장면은166/214→149/214,1–3m54/78→54/78·3–5m52/61→50/61·5–10m60/75→45/75다(각부분집합양쪽AT100%). total의차이를초기근거리페널티만으로설명할수없다. 학습horizon/epoch/ON_TOP 기준등이달라반경독립비교는아니다.
- [설명](output_etc/check/climb_total_progress0_height_20261108/report.txt),[조건부집계](output_etc/check/climb_total_progress0_height_20261108/extended_summary.csv),[실패분할](output_etc/check/climb_total_progress0_height_20261108/extended_failure_partition.csv),[동일장면비교](output_etc/check/climb_total_progress0_height_20261108/matched_pairs_summary.csv),[검증](output_etc/check/climb_total_progress0_height_20261108/extended_verification.json).

### total_at1mm — total12k 재개용 이관본 준비

- 일반 near50·행동별 교집합 최고 total12k(후행62.99±2.82%,동일쌍 선행96.46%)를 출발점으로 추천하고, 실행 가능한 resume 명령을 위해 `output_etc/check/total_at1mm_resume12k_20261108/`에 복사본/sidecar를 준비했다. 기존 새 학습용 config/train 기본값은 유지하며 원본 PTH를 수정하지 않았다.
- 이관본은 before 보상용1mm 옵션·실험/train 이름·checkpoint 계약/출처만 변경했다. 모델·optimizer·3종정규화·epoch12000·frame1572864000·env_state 등 나머지 모든 저장 항목을 재로드해 원본과 정확일치 확인했다. 기존 task/relation/collision/BEFORE 계약과 실제 saved-env 로드·실험검증 통과, 원본 SHA 불변 확인. 학습/시뮬레이션은 실행하지 않았다.
- [준비/검증](output_etc/check/total_at1mm_resume12k_20261108/verification.json),[재생성 스크립트](output_etc/check/total_at1mm_resume12k_20261108/prepare.py),[resume 명령](markdowns/USAGE.md). 추가1k 학습 후13k로 표기된다.

### total_at1mm 신규 학습 — AT 보상1mm와 후행20cm gate 분리

- 사용자 합의대로 `stage2_before_total_at1mm`을 total에서 분리했다. Stage1JointPushDensity3Cone90Finetune26k 전이·Stage2 새 학습(load_checkpoint=false)이며900step·가상상자·fusion·near50·+1.5m/1.0/payload 대기 페널티를 유지한다. env의 `before.reward_at_z_tolerance: 0.001`만 추가하며 기존 total YAML/학습 결과는 보존했다.
- BeforeRuntime의 보상용 성공만 AT Z≤1mm로 제한한다. 단독/선행/PUSH AT의 성공 보너스·State/Progress 포화 및 연결된 HOLDING/CONTACT의 placement 연동 포화에 적용한다. HOLDING/CONTACT 자체 성공은 기존대로다. native raw/own success·후행 Current gate·대기 Ever 해제·학습 성공 통계·정식 eval은 기존20cm 기준을 유지한다. 모델/관측/후행 자체 성공조건/후행 progress에는 변경이 없다.
- 전용 env/train YAML 및 train/test/VNC/eval wrapper,900step 등록·평가 routing·config 표와3종 명령·structure를 연결했다. 신규 before 옵션은 checkpoint 계약에 포함되므로 기존 total checkpoint의 직접 resume은 거부되며, 이번에는 별도 이관/리슘 기능을 추가하지 않았다.
- CPU98통과·1건 skip(기존 Stage1 PUSH checkpoint 미연결 테스트),35개 기존/신규 실험 계약 검사·Stage126k 파일 존재·Python AST·shell 문법 통과.1mm 양/음 경계·BEFORE3행동 Current gate/Ever 해제/부분reset·task shuffle·단독AT/PUSH 연동 포화·기존 total/progress0/대기 회귀를 확인했다. 본학습/시뮬레이션은 실행하지 않았다.
- [환경](tokenhsi/data/cfg/multi_agent/stage2_before_total_at1mm.yaml),[학습](tokenhsi/data/cfg/train/rlg/stage2_before_total_at1mm.yaml),[테스트](tokenhsi/tests/test_before_total_at1mm.py),[검증](output_etc/check/total_at1mm_config_20261108/verification.json),[로그](output_etc/check/total_at1mm_config_20261108/tests.log).


### total 선행 AT 1mm·바닥 접촉·손 유지 진단

- 일반 near50·행동별 교집합 저장512환경×3seed의 total12k/17k 및 fuv1 21k를900step으로 분석했다. 과거1mm는 상자중심–목표Z 오차이며 손 해제/바닥 접촉 조건은 아니다. AT Ever 중 기존20cm+1mm를 한 번도 동시에 만족하지 못한 쌍은 total12k246/1660,17k281/1632,v1 21k232/1642다. 그중 성공후 회전 반영 바닥간격≤5mm를1초 유지한 쌍은243/246,277/281,227/232로 대부분 공중 유지가 아니다. 마지막 공동 관찰1초의 기울기 중앙값60–120도는164/246,197/281,199/232다.
- GPU6에서 total12k/17k 각64환경seed0·900step 정식 개인낙상동결 재생에 손 위치/속도/순접촉력을 추가 기록했다. 최초AT 성공1초 이후 양손–상자체적 거리≤8cm·각손 순접촉력>1N 연속1초 후보는7/11쌍이며 후행동시3/7·2/11,동일쌍AT7/7·11/11이다.17k CLIMB 후보4쌍은 거리≤5cm에서도7.03–21.13초 유지·후행0/4/선행4/4, 그중1쌍은 기존1mm도 만족했다. 양손후보와 바닥간격>2cm가 함께1초 지속된 사례는 두64환경 모두0이다. 순접촉력은 상대 actor를 식별하지 않으므로 확정 grasp/실패 원인으로 해석하지 않는다.
- 두 추가 실행119/123초 정상 완료·초기 상태 모델간 동일·CPU/GPU/JSON 정확일치·900frame finite·손/상자 프레임 정렬·원본/snapshot SHA 및 기존2학습 생존 확인. 운영코드/학습 설정 변경 없음.1mm 복구의 인과적 개선 효과는 미검증이다.
- [설명](output_etc/check/total_at_height_holding_20261108/report.txt),[512 높이집계](output_etc/check/total_at_height_holding_20261108/height_summary.csv),[바닥/자세 개별쌍](output_etc/check/total_at_height_holding_20261108/placement_pairs.csv),[손 접촉 후보](output_etc/check/total_at_height_holding_20261108/hand_pairs.csv),[검증](output_etc/check/total_at_height_holding_20261108/verification.json).

### fuv1 21k·total12k 공통 교집합·합일치/near50해제·XY80/100 — 512환경×3seed

- 사용자 요청으로 GPU6에서 두 모델을 공통 공유상자 교집합·합일치·near50해제·거리합 대소제약없음·distractor포함·512환경×seed0/1/2·900step으로6회 평가했다. XY80/100은 같은 궤적 재채점이며 높이±.2m·실제AT current gate는 유지했다. 각 모델2개, 총4개 조건이다.
- fuv1 후행동시 평균±SD% XY80→100 /같은쌍 선행AT Ever: SIT58.43±2.99→61.34±2.66/96.62,CLIMB64.77±1.89→65.67±1.73/96.04,ON_TOP66.36±.85→69.38±.90/96.98,전체64.10±.96→66.67±1.54/96.69다.
- total 후행동시 평균±SD% XY80→100 /같은쌍 선행AT Ever: SIT68.79±2.42→72.16±3.06/96.62,CLIMB57.75±2.94→58.60±2.98/98.35,ON_TOP67.79±2.14→70.09±3.14/96.22,전체66.01±1.50→68.28±1.86/96.75다. 이번 표본 평균에서 total 전체/SIT가 높고 fuv1 CLIMB이 높다. 전체 모델간 우열의 통계적 유의성을 주장하지 않는다.
- 모델별 SIT415쌍·CLIMB352쌍·ON_TOP954쌍(전체1721쌍). 64환경1회에서 fuv1 공통XY100 78.46%였으나 확대 평가66.67%로 유지되지 않았다. total은64환경67.69%→68.28%다. 두 모델은 각 seed의 초기상태/역할/목표/크기가 bit-exact이며 원선행/ON_TOP각구간1–5m·후행최소2m·양측합2–10·nearflag0·교집합격자 확인했다.
- 같은 checkpoint·공통·합일치near해제·900step·seed0의 직전64환경 정지진단을 재사용했다. 개인낙상동결 궤적 진단이며 native 장면낙상종료 진단과 구분한다. 재사용 원본 경로/사람·payload 거리/정지·관찰가능 건수는 stop64_reused.csv와 verification.json에 기록했다.
- 6회114–186초 정상완료. CPU/GPU/JSON·XY80원점수정확복원·XY100기존성공손실0·finite·900frame, 원본/snapshot SHA·운영383파일·기존학습2개 보존 검증 통과. 운영 코드/학습 설정 변경 없음.
- [비교표](output_etc/check/fuv1_21k_total12k_sumoff_common_xy512x3_20261108/report.txt),[평균·SD·성공건수](output_etc/check/fuv1_21k_total12k_sumoff_common_xy512x3_20261108/summary.csv),[64환경 비교](output_etc/check/fuv1_21k_total12k_sumoff_common_xy512x3_20261108/comparison64.csv),[정지진단 재사용](output_etc/check/fuv1_21k_total12k_sumoff_common_xy512x3_20261108/stop64_reused.csv),[검증](output_etc/check/fuv1_21k_total12k_sumoff_common_xy512x3_20261108/verification.json).

### fuv1 21k·total12k 합일치/near50해제 × 행동별·공통 × XY80·100 — 64환경

- 사용자 요청으로 GPU6에서 fuv1 21k·total12k 각각 행동별/공통 공유상자 교집합으로64환경·seed0·900step 1회씩 총4회 실행했다. 합일치·near50해제·거리합 대소제약없음·distractor포함을 공통 적용하고, XY80/100은 동일 궤적 CPU 재채점으로8결과를 생성했다. 높이±.2m·실제AT current gate·보상·모델은 유지했다. ON_TOP XY는payload중심 판정이다.
- 후행전체 동시%/같은쌍 선행AT Ever: fuv1 행동별80 64.62/98.46,100 67.69/98.46;공통80 75.38/96.92,100 78.46/96.92. total 행동별80 61.54/95.38,100 64.62/95.38;공통80·100 모두67.69/96.92다. 이번 표본에서는 공통의 전체 성능이 높았으나 일반화/확정우열로 단정하지 않는다.
- 모든 조건에서 SIT12쌍·CLIMB12쌍·ON_TOP41쌍(전체65쌍). 한 SIT/CLIMB 성공 차이는8.33%p이며1seed결과다. 행동별 후행·동일쌍 선행·성공건수를 저장했다. 크기 조건 내 모델 초기 상태 bit-exact, 모든4회 task/쌍/물체매핑 동일, 원래선행두구간/ON_TOP두구간각1–5m·후행사람목표최소2m·양측합2–10·nearflag0·교집합격자검증 통과. XY100 기존성공손실0·XY80 원점수정확복원·CPU/GPU/JSON 재채점 통과.
- 정지 진단은 이번 같은64환경 정식 개인낙상동결 궤적에서 선행Ever전 armed·rootXYZ≤.3m/s 1초 최초정지를 측정했다. 기존 native 장면낙상종료 진단과 프로토콜이 다르다. 사람목표 중앙값m(정지/관찰가능): fuv1행동별 SIT2.56(4/4),CLIMB2.98(3/4),ON_TOP3.66(23/23)/payload2.18;공통 SIT2.83(4/4),CLIMB2.77(3/4),ON_TOP4.11(25/25)/payload2.30. total행동별 SIT1.56(1/2),CLIMB.71(1/1),ON_TOP3.25(10/12)/payload1.15;공통 SIT없음(0/1),CLIMB.27(1/2),ON_TOP2.80(9/11)/payload1.52. 소표본이며 정지 여부가 스킬 실행 이전 대기의 질을 보장하지 않는다.
- 4회 정상완료. 원본/snapshot SHA·운영383파일·기존학습2개 보존, 900frame·finite 검증 통과. 본학습/운영설정 변경 없음.
- [비교표](output_etc/check/fuv1_21k_total12k_sumoff_sizes_xy64_20261108/report.txt),[결과·성공건수](output_etc/check/fuv1_21k_total12k_sumoff_sizes_xy64_20261108/summary.csv),[정지 거리·건수](output_etc/check/fuv1_21k_total12k_sumoff_sizes_xy64_20261108/stops.csv),[검증](output_etc/check/fuv1_21k_total12k_sumoff_sizes_xy64_20261108/verification.json).

### total 17k 일반 near50 평가 — GPU6

- 사용자 요청으로 total17k를 일반 goalroute·near50·행동별 공유상자 교집합·512환경×seed0/1/2·900step·Plane Ever로 평가했다. Distractor 포함, GPU6만 사용하고 native64 seed0 정지 진단을 함께 수행했다.
- 후행 동시 평균±SD%/같은쌍 선행AT Ever는 SIT63.19±2.13/98.15,CLIMB47.44±.15/93.63,ON_TOP60.37±1.67/93.83,전체58.40±.53/94.82다. 16k60.90/95.41 대비 후행−2.50%p이며 ON_TOP65.71→60.37(같은쌍AT95.39→93.83) 하락이 크다. 모델 최고는12k62.99±2.82/96.46 유지다.
- 단독 SIT97.19,CLIMB100.00,AT97.14,ON_TOP97.34,PUSH95.07%다. native64 원크기·장면낙상종료에서 선행Ever전 armed·rootXYZ≤.3m/s 연속1초 첫정지 사람목표 XY 중앙값m(정지/관찰가능)은 SIT1.25(6/6),CLIMB1.73(5/5),ON_TOP1.99(10/14)/payload1.20이다. 정식 개인낙상동결 성공률과 구분한다.
- 4회149–166초 정상완료. 기존 초기상태/near50/거리/크기 격자 정확일치, CPU/GPU/JSON 채점·finite·900frame, checkpoint 원본/snapshot SHA·frozen actor encoder73tensor·input RMS·환경설정·운영383파일·기존학습2개 보존 검증 통과. 본학습 변경 없음.
- [결과](output_etc/check/total17k_general_20261108/success.csv),[16k 비교](output_etc/check/total17k_general_20261108/comparison.csv),[정지](output_etc/check/total17k_general_20261108/stops.csv),[검증](output_etc/check/total17k_general_20261108/verification.json).

### total 12–16k·progress0 11–15k 일반 near50 1k 간격 평가 — GPU6

- 사용자 요청으로 체크포인트 10개의 일반 goalroute·near50·행동별 공유상자 교집합·512환경×seed0/1/2·900step·Plane Ever와 각각 native64 seed0 정지 진단을 수행했다. Distractor는 원래대로 포함하며, 직전 distractor 제외 조건을 사용하지 않았다. GPU6만 사용하고 기존 학습은 유지했다.
- 후행 전체 동시 평균%/같은 쌍 선행AT Ever는 progress0 11k61.89/94.59,12k61.40/95.00,13k57.51/94.67,14k59.05/95.58,15k60.56/95.17이다. 최고는 기존10k62.47±2.31/95.05 유지다. 13k 하락 후 회복했지만 기존 최고를 넘지 못했다.
- total 12k62.99/96.46,13k59.50/96.51,14k58.87/95.69,15k57.07/95.07,16k60.90/95.41이다. 12k62.99±2.82%로 모델별 최고를 갱신했지만 이후 지속 상승하지 않았다. progress0 최고와 차이0.51%p만으로 우열을 단정하지 않는다.
- 최신 후행/같은쌍 AT%는 progress15k SIT58.25/96.26,CLIMB47.97/93.65,ON_TOP66.25/95.28;total16k SIT61.74/97.89,CLIMB46.93/92.58,ON_TOP65.71/95.39다. 전체·행동별 1k 추세, 단독, 표준편차, 쌍별 결과를 저장했다.
- native64 원크기·장면낙상종료 정지진단(선행Ever 전 armed·rootXYZ≤.3m/s 연속1초)의 사람목표 XY 중앙값m(정지/관찰가능)은 progress15k SIT1.16(5/5),CLIMB1.55(5/5),ON_TOP2.14(12/14)/payload1.21;total16k SIT1.16(6/6),CLIMB1.83(5/5),ON_TOP2.04(13/14)/payload1.21이다. 정식 개인낙상동결 성공률과 구분한다.
- 40회 모두 정상 완료(실행별132–206초, 전체 약29분). 30개 formal 초기상태/near50/거리/크기 격자 기존 평가와 정확일치, CPU/GPU/JSON 채점·finite·900frame, 10개 checkpoint 원본/snapshot SHA·frozen actor encoder73개 tensor·input RMS·환경설정, 운영383파일·기존학습2개 보존 검증 통과. 운영 코드·학습 설정 변경 없음.
- [간단 비교표](output_etc/check/total12_16k_progress11_15k_general_20261108/compact_report.txt),[전체 행동별 추세](output_etc/check/total12_16k_progress11_15k_general_20261108/full_action_history.csv),[정지 추세](output_etc/check/total12_16k_progress11_15k_general_20261108/stop_trends.csv),[단독](output_etc/check/total12_16k_progress11_15k_general_20261108/solo.csv),[검증](output_etc/check/total12_16k_progress11_15k_general_20261108/verification.json).

### progress0 10k 일반 near50 — distractor 물리·관측 제외 비교

- 사용자 요청으로 GPU6에서 행동별 공유상자 교집합·512환경×seed0/1/2·900step·Plane Ever를 평가했다. 원래 샘플링을 완료한 뒤 genuine distractor만 기존 비활성 물체 위치로 이동하고 기존 encoder object_valid 마스크로 모든 attention에서 제외했다. 과제용 상자·후행 alias·사람·목표·DOF·task packet·near50 초기 상태는 기존 평가와 정확히 일치한다. 운영 코드·설정·학습·checkpoint 변경 없음.
- 후행 동시 평균±SD% 기존→제외는 SIT61.92±4.11→57.56±3.76,CLIMB55.66±4.25→54.02±3.43,ON_TOP65.27±3.25→64.43±5.43,전체62.47±2.31→60.62±3.41이다. 동일쌍 선행AT Ever는 각각97.82→98.26,92.60→93.37,94.76→96.43,95.05→96.27이다. 물리·관측 동시 제거에서 개선되지 않았으며, 두 효과를 분리하거나 통계적 악화로 단정하지 않는다.
- native64 seed0 원크기·장면낙상종료 정지진단(선행Ever전 armed·rootXYZ≤.3m/s 연속1초): 사람목표 XY 중앙값m(정지/관찰가능) SIT1.19(5/5),CLIMB1.62(4/4),ON_TOP2.05(12/14)/payload1.21. 정식 개인낙상동결 성공률과 구분한다.
- 4회156–181초 정상완료. CPU/GPU/JSON 채점 일치·finite·900frame·거리/크기/alias/required 초기 상태 정확일치·checkpoint/운영파일/기존학습 보존 통과. Actor/Critic 실제 weight CPU검사에서 제외 토큰 feature/pose 교란 영향0·셔플 마스크·runtime hook 정확일치 확인. Gym 전후 내부 좌표 전송 audit만 0.2mm 허용하며, 비교한 과제 초기 상태는 bit-exact다.
- [비교](output_etc/check/progress0_10000_no_distractors_general_20261108/comparison.csv),[보고서](output_etc/check/progress0_10000_no_distractors_general_20261108/report.txt),[정지](output_etc/check/progress0_10000_no_distractors_general_20261108/stops.csv),[검증](output_etc/check/progress0_10000_no_distractors_general_20261108/verification.json),[마스크 검증](output_etc/check/progress0_10000_no_distractors_general_20261108/mask_verification.json).

## 2026-11-07

### XY100% 재채점 — 6조건 SIT/CLIMB/ON_TOP 전체

- 사용자 요청으로 total9k/progress0 9.5k/10k × 일반near50/합일치near해제 6조건의512환경×3seed×900step기존trace를CPU재채점했다. 실제받침/회전/높이±.2m/원래실제AT current gate/alive/궤적은유지하고 XY margin_fraction만.1→0(내부80%→전체100%)으로바꿨다. SIT/CLIMB은사람root,ON_TOP은payload중심판정이며payload전체면적포함기준이아니다. 운영성공조건/학습변경없음.
- SIT 후행동시상승폭은+5.34~7.60%p,CLIMB+0~1.09%p,ON_TOP+1.68~3.67%p다. CLIMB은80%경계폭만의민감도가작다는뜻이며XY접근전체가문제없다는뜻은아니다.
- progress10 일반 기존→XY100 후행동시평균%/동일쌍선행AT는 SIT61.92→67.26/97.82,CLIMB55.66→55.97/92.60,ON_TOP65.27→68.94/94.76,전체62.47→65.84/95.05다. 합일치는 SIT55.78→62.17/98.11,CLIMB55.74→56.58/96.54,ON_TOP65.90→68.51/95.68,전체61.38→64.57/96.46다.
- 18trace의원래후행개인/동시/같은쌍선행점수정확복원·기존성공상실없음·직전SIT재채점정확일치검증통과. 각행동/전체/seed/쌍별결과와표준편차를저장했다.
- [전체표](output_etc/check/xy100_six_cases_20261107/summary.csv),[기준](output_etc/check/xy100_six_cases_20261107/report.txt),[검증](output_etc/check/xy100_six_cases_20261107/verification.json).

### SIT XY 실패 — total9k/progress0 9.5k/10k × 일반/합일치 6조건 CPU 재분석

- 사용자 요청으로6조건의512환경×3seed×900step기존trace를CPU재채점,415 SIT쌍씩18개trace의후행개인/동시/동일쌍선행AT Ever를정확복원했다. 새GPU실행·정책/보상/성공조건변경은없다.
- 선행Ever·SIT개인미성공 분모/접근미달/접근후실제XY80%영역미진입은 일반total9k174/66/103,progress9.5k169/56/104,progress10k147/49/96;합일치total9k190/87/99,progress9.5k165/73/87,progress10k168/69/99다. XY진입후높이동시미충족은각0–2쌍이다. 분류는선행최초성공이후alive프레임기준이며currentgate기준도별도저장했다.
- 실제→가상XYZ받침재채점SIT동시평균%/같은쌍AT는 일반total9 54.33→63.48/97.60,progress9.5 56.83→65.81/98.84,progress10 61.92→68.82/97.82;합일치total9 51.67→60.83/98.11,progress9.5 57.81→63.77/98.40,progress10 55.78→63.52/98.11다. 원래실제ATcurrentgate·사람궤적·크기·q·alive유지한기하가정이며실제개선예측이아니다.
- progress10의가상기준회복은선행Ever/SIT개인미성공147중31,168중35로위치불일치만이전체원인은아니다. currentgate·근처·높이충족실패70/72쌍의장면별최소XY영역오차중앙값6.24/7.38cm다. 높이무시재채점은progress10양조건점수변화없음. 합일치9.5→10에서접근미달73→69이나XY미진입87→99,개인Ever있지만동시성공없음3→8이다.
- [해석](output_etc/check/sit_six_cases_20261107/interpretation.txt),[요약](output_etc/check/sit_six_cases_20261107/summary.json),[쌍별](output_etc/check/sit_six_cases_20261107/pairs.csv),[검증](output_etc/check/sit_six_cases_20261107/verification.json).

### progress0 10k 합일치·near50/거리합 대소 제약 해제 평가 — GPU6

- 사용자 요청으로 progress0 10k를 합일치(선행 각1–5m,후행 SIT/CLIMB 사람목표2–10m,ON_TOP 두구간각1–5m·사람목표최소2m)·near50해제·거리합대소제약없음으로 평가했다. 행동별공유상자교집합·4명8토큰·512환경×seed0/1/2·900step·Plane Ever,GPU6만사용했다.
- 후행동시평균±SD%/동일쌍선행AT는 SIT55.78±2.69/98.11,CLIMB55.74±2.27/96.54,ON_TOP65.90±3.24/95.68,전체61.38±1.90/96.46이다. 같은조건9.5k63.20/96.17에서−1.82%p이며 total9k61.15/95.65와비슷하다. 일반near50 10k62.47/95.05 대비−1.09%p로, 일반조건상승이합일치조건에서는유지되지않았다. Progress변경의인과·전조건개선으로단정하지않는다.
- native64 원크기·장면낙상종료의 미완료선행·armed·rootXYZ≤.3m/s연속1초 첫정지 사람목표XY중앙값m(정지/관찰가능)은 SIT1.58(3/4),CLIMB1.62(1/1),ON_TOP2.92(6/9)/payload1.30다. 소표본이며정식개인낙상동결분모와구분한다.
- 4회184–218초정상완료. 같은조건기존평가초기상태tensor정확일치·nearflag0·원거리구간/합2–10/교집합격자·CPU/GPU/JSON재채점·finite·900frame·원본/snapshotSHA·운영306파일·기존학습2개보존검증통과. 본학습변경없음.
- [보고서](output_etc/check/progress0_10000_sum_matched_no_near_20261107/report.txt),[비교](output_etc/check/progress0_10000_sum_matched_no_near_20261107/comparison.csv),[정지](output_etc/check/progress0_10000_sum_matched_no_near_20261107/stops.csv),[검증](output_etc/check/progress0_10000_sum_matched_no_near_20261107/verification.json).

### progress0 10k·total11k 일반거리·near50 행동별 교집합 평가 — GPU6

- 사용자 요청으로 progress0 10k(total9k 재개후+1k)와 원본total11k를 일반goalroute·near50·행동별공유상자교집합·4명8토큰·각512환경×seed0/1/2·900step·Plane Ever로 평가하고 각각native64 seed0 정지진단을 수행했다. 각직전checkpoint와환경동일·actorencoder73/inputRMS동결을확인했다.
- progress0 후행동시평균±SD%/동일쌍선행AT는 SIT61.92±4.11/97.82,CLIMB55.66±4.25/92.60,ON_TOP65.27±3.25/94.76,전체62.47±2.31/95.05다. 9.5k60.56/95.69 대비후행+1.92%p·선행−.64%p이며 SIT+5.09%p다. 시작점total9k59.23/94.41보다+3.25%p다.
- total11k는 SIT54.00±5.28/98.11,CLIMB42.45±6.84/92.63,ON_TOP60.03±2.24/94.87,전체54.95±3.34/95.18다. total10k56.92/94.43보다−1.96%p,CLIMB−7.24%p이며 total최고는9k를 유지한다. 단독5종 평균progress0 96.98%,total97.19%다.
- native64 원크기·장면낙상종료의 미완료선행·armed·rootXYZ≤.3m/s연속1초 첫정지 사람목표XY중앙값m(정지/관찰가능)은 progress0 SIT1.16(5/5),CLIMB1.93(3/3),ON_TOP1.97(14/15)/payload1.21;total SIT1.20(7/7),CLIMB1.85(4/5),ON_TOP1.96(12/14)/payload1.21다. 정식개인낙상동결평가와분모가다르다.
- 8회각149–174초,GPU6에서최대4개동시실행으로정상완료. CPU/GPU/JSON재채점·finite·900frame·초기장면동일·원near50/거리/교집합격자·checkpointSHA·운영306파일·기존학습2개보존검증통과. 본학습변경없음.
- [보고서](output_etc/check/total11k_progress10k_general_20261107/report.txt),[행동별](output_etc/check/total11k_progress10k_general_20261107/latest_actions.csv),[추세](output_etc/check/total11k_progress10k_general_20261107/full_trends.csv),[정지](output_etc/check/total11k_progress10k_general_20261107/stops.csv),[검증](output_etc/check/total11k_progress10k_general_20261107/verification.json).

### progress0 9.5k 합일치·near50/거리합 대소 제약 해제 평가 — GPU6

- 사용자 요청으로 progress0 9.5k를 합일치(선행 각1–5m,후행 SIT/CLIMB 사람목표2–10m,ON_TOP 두구간각1–5m·사람목표최소2m)·near50해제·선후행경로합대소제약없음으로 평가했다. 행동별 공유상자 교집합·4명8토큰·512환경×seed0/1/2·900step·Plane Ever, GPU6만 사용했다.
- 후행 동시평균±SD%/같은쌍 선행AT Ever는 SIT57.81±4.21/98.40,CLIMB58.40±4.53/94.08,ON_TOP67.46±4.15/96.03,전체63.20±2.40/96.17다. 같은모델 일반거리near50 전체60.56/95.69 대비+2.64%p다. 거리하한과near제약을함께바꿔near해제단독효과로해석하지않는다.
- 같은 합일치near해제 total9k61.15/95.65 대비+2.05%p다. 행동별후행/선행 이전→현재는 SIT51.67/98.11→57.81/98.40,CLIMB58.95/96.40→58.40/94.08,ON_TOP66.24/94.33→67.46/96.03이다.
- native64 seed0 원래크기분포·장면낙상종료의 선행미완료·armed·rootXYZ≤.3m/s 연속1초 첫정지 사람목표XY 중앙값m(정지/관찰가능)은 SIT1.62(2/2),CLIMB1.81(1/1),ON_TOP2.88(7/8),ON_TOP payload1.34m다. 소표본이며 정식 개인낙상동결 평가와 분모가 다르다.
- 4회150–174초 정상완료. 같은조건 기존평가와 초기상태tensor정확일치·nearflag0·원거리구간/합2–10/교집합격자·CPU/GPU/JSON재채점·finite·900frame·원본/snapshotSHA·운영306파일·기존학습2개보존 검증 통과. 본학습변경없음.
- [보고서](output_etc/check/progress0_9500_sum_matched_no_near_20261107/report.txt),[비교](output_etc/check/progress0_9500_sum_matched_no_near_20261107/comparison.csv),[정지](output_etc/check/progress0_9500_sum_matched_no_near_20261107/stops.csv),[검증](output_etc/check/progress0_9500_sum_matched_no_near_20261107/verification.json).

### progress0 첫9.5k 일반거리·near50 행동별 교집합 평가 — GPU6

- 사용자 요청으로 `stage2_before_total_progress0` 첫 정기저장9.5k(total9k 재개후+500epoch)를 일반 goalroute·near50·행동별 공유상자 교집합·4명8토큰·512환경×seed0/1/2·900step·Plane Ever로 평가했다. total9k 대비 Progress반경0 외 환경 동일(재개 checkpoint 경로 제외), actor encoder73/input RMS 동결을 확인했다.
- 후행 동시평균±SD%/같은쌍 선행AT Ever는 SIT56.83±5.09/98.84,CLIMB54.29±6.69/93.99,ON_TOP64.59±2.58/94.98,전체60.56±1.07/95.69다. 시작점total9k59.23/94.41 대비후행+1.33%p, SIT+2.51/CLIMB+2.49/ON_TOP+.30%p다. 선행도+1.28%p 상승하여 Progress변경의 인과효과로 단정하지 않는다. 단독 SIT95.17/CLIMB98.27/AT96.87/ON_TOP98.10/PUSH95.30%다.
- native64 seed0 원래 크기분포·장면낙상종료에서 선행미완료·armed·rootXYZ≤.3m/s 연속1초 첫정지 사람목표XY 중앙값m(정지/관찰가능)은 SIT1.17(5/6),CLIMB1.63(4/4),ON_TOP1.90(13/14),ON_TOP payload1.21m다. 정식 개인낙상동결 평가와 분모를 구분한다.
- 4회135–182초 정상 완료. CPU/GPU/JSON 재채점·finite·900frame·초기장면/near50/거리/교집합격자·원본/snapshotSHA·운영306파일보존·기존학습2개생존 검증 통과. 본학습 변경 없음.
- [보고서](output_etc/check/progress0_9500_general_20261107/report.txt),[행동별](output_etc/check/progress0_9500_general_20261107/success.csv),[정지](output_etc/check/progress0_9500_general_20261107/stops.csv),[검증](output_etc/check/progress0_9500_general_20261107/verification.json).

### FT21k 추가10k 일반거리·near50 행동별 교집합 평가 — GPU6

- 사용자 요청으로 `stage2_before_total_ft21k` 추가10k를 일반 goalroute·near50·행동별 공유상자 교집합·4명8토큰·512환경×seed0/1/2·900step·Plane Ever로 평가했다. 추가8k와 환경/actor encoder73/input RMS 동일성을 확인했다.
- 후행 동시평균±SD%/같은쌍 선행AT Ever는 SIT53.09±5.24/98.11,CLIMB47.19±.86/94.36,ON_TOP68.87±2.19/97.92,전체60.56±1.66/97.21다. 추가8k60.19/96.68 대비+0.37%p이며 기존 최고추가3k60.68/96.21와 비슷하고 최고기록은 갱신하지 않았다. 단독 SIT95.31/CLIMB99.26/AT97.30/ON_TOP98.47/PUSH96.14%다.
- native64 seed0 원래 크기분포·장면낙상종료에서 선행미완료·armed·rootXYZ≤.3m/s 연속1초 첫정지 사람목표XY 중앙값m(정지/관찰가능)은 SIT1.20(5/5),CLIMB2.01(3/3),ON_TOP2.18(14/14),ON_TOP payload1.25m다. 정식 개인낙상동결 평가와 분모를 구분한다.
- 4회187–217초 정상 완료. CPU/GPU/JSON 재채점·finite·900frame·초기장면/near50/거리/교집합격자·원본/snapshotSHA·운영306파일보존·기존학습3개생존 검증 통과. 본학습 변경 없음.
- [보고서](output_etc/check/ft21k_10k_general_20261107/report.txt),[행동별](output_etc/check/ft21k_10k_general_20261107/success.csv),[추세](output_etc/check/ft21k_10k_general_20261107/full_trends.csv),[정지](output_etc/check/ft21k_10k_general_20261107/stops.csv),[검증](output_etc/check/ft21k_10k_general_20261107/verification.json).

### total10k 일반거리·near50 행동별 교집합 평가 — GPU6

- 사용자 요청으로 원본 `stage2_before_total` 10k를 일반 goalroute·near50·행동별 공유상자 교집합·4명8토큰·512환경×seed0/1/2·900step·Plane Ever로 평가했다. progress0 분기가 아니며 total9k와 환경/actor encoder73/input RMS 동일성을 확인했다.
- 후행 동시평균±SD%/같은쌍 선행AT Ever는 SIT54.03±3.56/98.07,CLIMB49.69±6.98/91.30,ON_TOP60.87±2.03/94.04,전체56.92±2.34/94.43다. 9k 전체59.23/94.41보다−2.31%p, ON_TOP−3.41%p이며 total 최고 저장본은9k를 유지한다. 단독 SIT93.03/CLIMB98.03/AT97.29/ON_TOP96.96/PUSH95.08%다.
- native64 seed0 원래 크기분포·장면낙상종료에서 선행미완료·armed·rootXYZ≤.3m/s 연속1초 첫정지 사람목표XY 중앙값m(정지/관찰가능)은 SIT1.18(6/6),CLIMB1.80(5/6),ON_TOP2.03(11/14),ON_TOP payload1.21m다. 정식 개인낙상동결 평가와 분모를 구분한다.
- 4회184–208초 정상 완료. CPU/GPU/JSON 재채점·finite·900frame·초기장면/near50/거리/교집합격자·원본/snapshotSHA·운영306파일보존·기존학습3개생존 검증 통과. 본학습 변경 없음.
- [보고서](output_etc/check/total10k_general_20261107/report.txt),[행동별](output_etc/check/total10k_general_20261107/success.csv),[추세](output_etc/check/total10k_general_20261107/full_trends.csv),[정지](output_etc/check/total10k_general_20261107/stops.csv),[검증](output_etc/check/total10k_general_20261107/verification.json).

### total9k Progress 포화 반경0 재개 실험 추가

- 사용자 요청으로 `stage2_before_total_progress0` 환경/학습 YAML과 train/test/VNC/eval wrapper를 추가했다. BEFORE 후행 SIT/CLIMB/ON_TOP 수행 edge만 `before.progress_radius: 0.0`으로 `p=1/(1+d)`를 사용한다. 기존 설정은 원래 반경을 유지하며 실제 성공 시 포화·current gate·가상상자·대기 페널티·독립/HOLDING·900step·모델 구조는 보존했다.
- train은 total9k를 기본 source로 원본 보존 재개용 복사본을 `output_etc/check/before_total_progress0_initial/`에 준비한다. 모델/optimizer/정규화/epoch/frame 전체 동일성을 확인했고 새 환경/실험명/계약만 변경했다. 일반 checkpoint 계약 검사는 유지하며 새 저장본은 그대로 resume한다. epoch9k부터 이어지므로 추가1k는10k다.
- GPU6·4명1024환경8토큰으로 9000→9002 epoch 실제 재개 학습·저장을 완료했다. 출력 finite·새 반경/900step 반영·frame 증가를 확인했다. CPU 49개 테스트(반경·행동/셔플 매핑·gate·성공 포화·기존/asym/total 회귀) 통과. 기존 학습 프로세스는 보존했으며 장기 본학습은 시작하지 않았다. config/USAGE/structure에 실행법과 역할을 기록했다.
- [테스트](output_etc/check/total_progress0_20261107/tests.log), [재개 보존](output_etc/check/total_progress0_20261107/resume_verification.json), [짧은 학습 검증](output_etc/check/total_progress0_20261107/train_verification.json), [학습 로그](output_etc/check/total_progress0_20261107/train.log).

### total9k 근처 정체의 실제 보상·발 움직임 진단 — GPU6

- 사용자 요청으로 모델구조 변경 없는 다음 실험을 검토하기 위해 total9k 일반거리near50·행동별교집합512환경 seed0 900step을 GPU6에서 추가 재생했다. runtime State/Progress/Success·HOLDING·발위치/속력을 읽기만 했으며 운영파일/정책/입력/물리/보상은 변경하지 않았다. 기존 초기상태는 정확히 같으나20frame부터 궤적이 달라져 별도 진단으로 취급하고 기존3seed 평가를 갱신하지 않았다. 진단 자체 CPU/GPU/JSON 재채점은 일치했다.
- 후행동시/동일쌍선행AT 건수는 SIT77/131·128/131,CLIMB58/110·101/110,ON_TOP211/326·305/326다. 선행Ever·후행미성공47/42/94쌍 중 gate열림·Progress최대·rootXYZ≤.3·양발≤.1m/s 연속30frame은15/18/10쌍이다. CLIMB근처실패23쌍 중 발움직임>.3m/s 연속15frame은12쌍,한발이상상자전체XY범위내/상단높이±.1m 연속15frame은3쌍이다. 접촉방해나동작의도는단정하지않았다.
- 근처실패장면별중앙값의중앙값 State는 SIT.07286/CLIMB.002245/ON_TOP.02624,Progress는모두.2였다. CLIMB높이오차중앙값.513m다. 이정체의인과증명은아니지만완료전추가접근신호를시험할근거다.
- 후보는 total9k 분기에서 BEFORE후행 수행edge Progress포화반경만0으로바꿔 p=1/(1+d)로 만드는 것. 기존가상G/모델/AT20cm/gate/대기페널티/HOLDING/독립과제/성공시포화는유지하는설계이며미구현이다. 바깥접근보상과기울기도줄어드는한계를명시하고추가1k/2k의정식성공·대기·정체진단으로확인하는실험을추천했다.
- checkpointSHA·GPU6UUID·운영301파일·기존학습2개보존검증통과,새재생102초완료. 같은checkpoint의직전native64정지진단을재사용했다. [해석](output_etc/check/total9k_reward_motion_diagnostic_20261107/interpretation.txt),[요약](output_etc/check/total9k_reward_motion_diagnostic_20261107/summary.json),[쌍별](output_etc/check/total9k_reward_motion_diagnostic_20261107/pairs.csv),[검증](output_etc/check/total9k_reward_motion_diagnostic_20261107/verification.json).

### total9k 실제 받침을 목표 쪽으로5·10cm 이동한 가정 재채점

- 사용자 요청으로 total9k 일반거리near50/합일치near해제 각각512환경×3seed×900step 저장 궤적에서 받침XYZ를 G쪽으로 min(5/10cm,현재오차)만큼 이동시켜 재채점했다. 사람/payload/회전/크기/alive/기존 실제20cm AT current mask는 유지했다. 물리 실행/정책 변경/학습 효과 예측이 아니며 XY만 이동하는 민감도와 G완전정렬도 함께 저장했다.
- 합일치near해제의 원래→5cm→10cm 후행동시3seed평균%는 SIT51.67→56.69→60.58(같은쌍AT98.11),CLIMB58.95→60.05→59.75(96.40),ON_TOP66.24→68.24→68.97(94.33),전체61.15→63.70→64.97(95.65)다. 전체5cm는실패47회복/기존성공3상실,10cm는74회복/8상실로 단조개선을 가정하지 않았다. 일반near50 전체59.23→60.86→61.96(AT94.41),SIT54.33→58.65→61.81(97.60)다.
- 고정량 이동과 AT 임계치 강화의 차이를 구분하기 위해, 오차15/10cm를 초과한 프레임만 해당 경계까지 보정하는 가정도 계산했다. 합일치전체61.15→61.43→62.83(AT95.65),일반전체59.23→59.63→60.57(94.41)다. 원래gate를 보존한 기하 보정이며 좁은gate로 실제 실행한 결과가 아니다. XYZ 이동은 지면/접촉 제약을 재시뮬레이션하지 않는다.
- 6개원평가의무이동개인/동시/선행점수정확복원·원본과alias pose동일·이동크기/방향/목표통과없음·Quaternion보존·같은쌍선행상한을검증했다. CPU만 사용했으며 운영코드/config/학습은 변경하지 않았다.
- [보고서](output_etc/check/total9k_support_shift_20261107/report.txt),[전체결과](output_etc/check/total9k_support_shift_20261107/summary.csv),[쌍별회복/상실](output_etc/check/total9k_support_shift_20261107/pairs.csv),[검증](output_etc/check/total9k_support_shift_20261107/verification.json).

### total9k 합일치·near50 해제·거리합 대소 제약 없음 평가 — GPU6

- 사용자 요청으로 total9k를 합일치 거리·near50 해제·선후행 경로합 대소 제약 없음으로 새 평가했다. 선행 두 구간 각1–5m, 후행 SIT/CLIMB 사람목표2–10m, ON_TOP 두 구간 각1–5m 및 사람목표 최소2m·기존 충돌검사를 유지했다. 행동별 공유상자 교집합·4명8토큰·512환경×seed0/1/2·900step·Plane Ever다. 모델 구조/학습 설정/20cm AT 기준은 변경하지 않았다.
- 후행 동시평균±SD%/동일쌍 선행AT Ever는 SIT51.67±5.90/98.11, CLIMB58.95±5.21/96.40, ON_TOP66.24±2.41/94.33, 전체61.15±2.50/95.65다. 기존9k 일반거리near50 전체59.23/94.41 대비후행+1.92%p, 행동별−2.65/+7.14/+1.95%p다. 같은 새조건6k59.45/96.17 대비+1.70%p다. 거리 하한과 near50를 함께 바꿔 near50 단독 효과로 해석하지 않는다.
- 초기 장면 전체 tensor가 같은조건6k와 정식3회/native1회 모두 정확히 일치했다. forced near플래그는0, 실제 후행합짧음876·길거나같음845쌍이다. 해당 부분집합 후행동시/동일쌍 AT는 전체59.70/95.13 대62.65/96.21%, 행동별도 저장했다.
- native64 seed0·기존상자분포·장면낙상종료의 미완료선행·armed·rootXYZ≤.3m/s 연속1초 첫정지 사람목표XY 중앙값m(정지/관찰가능)은 SIT1.53(4/4),CLIMB미관찰(0/1),ON_TOP3.23(6/9),ON_TOP payload1.35m다. 작은 정지표본과 정식 개인낙상동결 분모를 구분했다.
- 정식3회CPU/GPU/JSON일치·finite·900frame·실제GPU6UUID·거리/경로합/near해제/원충돌검사·교집합격자/비공유상자·checkpoint SHA·직전검증된동결encoder73/inputRMS snapshot·운영301파일·기존학습2개보존, native진입/AT Ever해제/가중치불변검증 통과.4회150–167초 정상완료.
- [보고서](output_etc/check/total9k_sum_matched_no_near_20261107/report.txt),[비교](output_etc/check/total9k_sum_matched_no_near_20261107/comparison.csv),[경로합별](output_etc/check/total9k_sum_matched_no_near_20261107/route_order.csv),[정지](output_etc/check/total9k_sum_matched_no_near_20261107/stops.csv),[검증](output_etc/check/total9k_sum_matched_no_near_20261107/verification.json).

### total9k 일반거리/near50 평가 및 AT10·15cm 도달 진단 — GPU6

- 최신 정수1k 저장본 total9k를 기존 일반거리 goalroute·near50 유지·행동별 공유상자 교집합·4명8토큰·512환경×seed0/1/2·900step·Plane Ever로 평가했다. FT21k 최신은 선택 시8.5k여서1k 간격 원칙에 따라 기존8k 결과를 재사용했다. 평가와 별개로10cm 조건 변경/학습 재개는 하지 않았다.
- total9k 후행 동시평균±SD%/동일쌍 선행AT Ever는 SIT54.33±6.32/97.60·CLIMB51.80±5.83/93.19·ON_TOP64.29±2.82/93.52·전체59.23±1.46/94.41다. 8k 전체57.42/94.07 대비후행+1.81%p, 행동별+4.18/+2.58/+0.66%p다. total 기록상 최고이며 단독5종 평균95.73%다. 기존FT8k 전체60.19±1.59/96.68와 비교한다.
- native64 seed0·기존상자분포·900step·장면낙상종료의 정지 진단도 완료했다. 미완료 선행·armed·rootXYZ≤.3m/s 연속1초 첫정지 사람목표XY 중앙값m(정지/관찰가능)은 SIT1.08(7/7),CLIMB1.76(4/6),ON_TOP1.92(13/14),ON_TOP payload1.22m다. 정식 개인낙상동결 분모와 구별한다.
- 기존20cm 정책의 저장 궤적에서 AT3D 임계치20/15/10cm를 CPU 재채점했다. 같은쌍 선행Ever3seed평균은 total9k94.41/91.03/79.32%,FT8k96.68/93.49/80.37%다. 연속30frame 유지비율은 total9k93.84/88.26/70.71%,FT8k95.93/90.82/73.11%다. total8k도 함께 저장했다. 변경된 gate로 실행한 결과나10cm 학습 후 성능 예측이 아니며 도달 지연 통계는 각 임계치에 도달한 쌍에 한정한다.
- 정식3회CPU/GPU/JSON재채점일치·finite·900frame·실제GPU6UUID·초기상태기존7k정확일치·near50/거리/교집합/격자/비공유상자보존·snapshot/원pthSHA·8k대비Stage1 actor encoder73/input RMS/전체실험설정동일·운영301파일·기존학습2개보존 검증을 통과했다. native진입/AT Ever해제/가중치불변도 확인했다.4회150–170초 정상완료.
- [보고서](output_etc/check/total9k_general_20261107/report.txt),[행동별](output_etc/check/total9k_general_20261107/success.csv),[추세](output_etc/check/total9k_general_20261107/full_trends.csv),[정지](output_etc/check/total9k_general_20261107/stops.csv),[AT임계치](output_etc/check/total9k_general_20261107/at_tolerance.csv),[검증](output_etc/check/total9k_general_20261107/verification.json).

### SIT 중앙 정렬 후보·CLIMB 상승·ON_TOP 운반 단계 CPU 진단

- 사용자 요청으로 total8k/FT21k+8k/v1 21k의 동일 초기 장면·일반거리 near50·행동별 교집합·512환경×3seed 기록을 모두900step으로 재분석하고 기존 개인/동시 점수9회를 정확 재현했다. GPU/운영 코드/config/프로세스 변경은 없다.
- 선행 Ever·후행 개인 미성공 중 CLIMB total147/v1126쌍에서 해제 후 초기 root 대비10cm 상승 연속15frame은11/11쌍,20cm는7/9쌍이었다. 근처 연속15frame 도달하지만20cm 상승 지속없음99/80쌍이다. 발/관절/접촉 trace가 없어 시도 자체의 부재나 선행 방해로 단정하지 않았다.
- ON_TOP은 초기 대비 상승 대신 회전된 payload의 실제 바닥높이≥10cm 연속15frame과 사람XY1.2m 이내·초기payloadXY변위≥0.5m 연속15frame 운반 지표를 사용했다. total 미성공265쌍은 공중유지없음48·공중유지하나운반지표없음2·운반지표있음215, v1은232=59+1+172다. total 운반215중 접근미달114(마지막 alive구간에도 공중+사람근처94), 접근후XY미진입83(근처높이충족경험59), XY진입후높이미충족18이다. grasp/놓침/의도적 내려놓기는 이 지표로 확정하지 않았다.
- SIT 실제실패·가상동시성공 total37/v126쌍 중 같은 프레임에서 rootXY를 가상중앙G로 강제하는 가정 재채점은27/18쌍에서 실제영역/높이도 통과했다. 첫 가상성공프레임 상자-G XY오차 중앙값11.5/13.4cm다. 실제 정책 변경 효과로 해석하지 않았다. 성공조건 강화보다 완료 후에도 중앙XY 정렬 보상 신호를 남기는 별도 설계 후보를 제시하며 구현은 하지 않았다.
- [해석·한계](output_etc/check/before_skill_failure_stages_20261107/interpretation.txt), [요약](output_etc/check/before_skill_failure_stages_20261107/summary.json), [쌍별](output_etc/check/before_skill_failure_stages_20261107/pairs.csv), [운반후실패](output_etc/check/before_skill_failure_stages_20261107/ontop_transport_failures.json), [검증](output_etc/check/before_skill_failure_stages_20261107/verification.json).

### total 기반 다음 실험 검토 — v1·FT21k·total 실패 비교

- 기존 동일 초기 장면·일반거리/near50·행동별 교집합·512환경×3seed·900step 진단을 CPU 재집계했다. v1 21k/FT21k+8k/total8k의 선행 Ever 후 후행 개인 미성공은 491/616/595쌍이며, 접근반경 미진입 226/370/244·접근 후 XY 미진입 234/218/312쌍이다. 전체 후행 동시/동일쌍 선행 AT 평균은 65.83/95.40·60.19/96.68·57.42/94.07%다.
- 접근 미달 중 페널티 해제 후 root 최대 XY 변위≤25cm는 53/37/42쌍으로, 접근 실패 전체를 정지 지속으로 해석할 수 없다. CLIMB는 해당 건수 20/35·22/72·20/38로 정체 사례가 있고, ON_TOP 접근 미달의 payload 초기 대비 10cm 상승도 없는 사례는 49/146·32/192·30/155다. 변위는 이동 경로 길이·의도적 보행이 아니며 payload 상승은 집기 성공을 보장하지 않는다.
- 후행 정책에 armed/released/current 상태를 명시적으로 전달하는 별도 실험을 후보로 검토했다. 현재 fusion은 정적 BEFORE embedding이고 Ever 상태를 직접 입력하지 않는다. 인과·개선 효과는 미검증이며 코드/config/프로세스를 변경하지 않았다. [재집계](output_etc/check/total_ft_v1_failure_review_20261107/summary.csv), [기준·한계](output_etc/check/total_ft_v1_failure_review_20261107/provenance.json).

### total8k 후행 실패 병목 CPU 진단 — total6k·FT8k·v121k와 동일900step 비교

- 사용자 요청으로기존일반거리/near50·행동별공유상자교집합·각512환경×3seed저장trace를재분석했다. total6k/8k·ft21k+8k·v121k앞900step의개인/동시/동일쌍선행AT Plane Ever를정확히재구성했고모든초기상태가모델간/seed별정확히일치했다. 새GPU평가/운영코드/학습설정변경은없다.
- total8k후행동시57.42%/개인59.57%/같은쌍선행AT94.07%다. 전체동시실패733쌍은선행미성공102·선행Ever후후행개인미성공595·양쪽개인Ever이나동시없음36이다. 선행성공후후행개인실패595쌍의배타적기하분류는접근반경미진입244(SIT51/CLIMB38/ON_TOP155),접근후실제XY성공영역미진입312(124/102/86),XY진입후높이불충족25(1/6/18),이후생존없음14(7/1/6)다. 이분류는실패지점이며접촉/동작실패의인과원인을단정하지않았다.
- 선행Ever쌍모두Plane AT첫성공과페널티첫해제가정확히일치하고해제후armed프레임은0이다. 후행개인실패595쌍중382쌍은이후살아있는프레임95%이상ATcurrent도유지했다. 이595쌍의선행첫성공중앙값189step,후행성공988쌍의첫동시성공중앙값321step/750step이후첫성공16쌍이다. 시간연장후성공량은추정하지않았다.
- 선행Ever·동시실패631쌍중146쌍은마지막100프레임중30프레임이상ATcurrent·alive·페널티Ever해제·가상목표progress=1을함께만족했다.142쌍은해당조건과rootXYZ속력≤.3m/s도누적30프레임이상만족했다. 연속1초정지나손/팔정지로해석하지않았다.146쌍중137쌍은실제XY미진입분류다. 입력은scene/task/group이며armed/released직접입력없음·Alias는완료후에도G고정·progressG기준/state와success실제상자기준이라는코드를확인했다. 대기→수행전환/마지막정렬이병목후보이나단일원인확정은아니다.
- 동일궤적을support중심G기준으로가정재채점하면total8k전체60.79%·SIT57.96%(원래57.42/50.14%,동일쌍AT94.07/96.84%)다. 실패68쌍추가인정·기존성공10쌍상실이며정책관측변경의효과예측은아니다. XY100%만전체60.43·높이제거만58.35·둘다61.94%로판정완화만으로v121k900step65.83%/AT95.40%가재현되지는않았다.
- v1만성공한413장면중total8k접근후XY미진입199·접근미달118쌍,반대로total만성공한268쌍도있다. 학습량/진입반경/episode길이차이를분리하지않아반경1.5m하나의인과효과로단정하지않았다. 선행완료후Alias실제위치전환은향후별도검증후보로만기록했으며구현하지않았다.
- [보고서](output_etc/check/total8k_failure_compare_20261107/report.md),[행동별비교](output_etc/check/total8k_failure_compare_20261107/scores.csv),[쌍별진단](output_etc/check/total8k_failure_compare_20261107/pairs.csv),[progress포화](output_etc/check/total8k_failure_compare_20261107/late_progress_plateau.csv),[검증](output_etc/check/total8k_failure_compare_20261107/extended_verification.json).

### total8k·ft21k+8k 기존 일반거리/near50 평가 — GPU6

- 사용자에게기준을확인한뒤합일치·무제약대신기존일반거리goalroute·near50유지·행동별공유상자교집합·4명8토큰·512환경×seed0/1/2·900step·원래Plane Ever로GPU6에서두저장본을평가했다. 거리hook을사용하지않고기존7k평가driver/probe를재사용했다. near플래그/모드만읽어서기록했으며전체초기상태가모델간/기존7k와정확히일치했다.
- 후행동시평균±SD%/같은쌍선행AT Ever는total8k SIT50.14±3.87/96.84·CLIMB49.23±9.11/91.15·ON_TOP63.62±1.18/93.92·전체57.42±2.28/94.07,ft21k+8k51.56±6.31/97.31·49.17±0.85/94.63·68.15±5.46/97.18·전체60.19±1.59/96.68다. 분모SIT415/CLIMB352/ON_TOP954/전체1721쌍유지. 7k전체54.80/94.36·59.81/96.16대비후행+2.62/+0.38%p다. total은CLIMB/ON_TOP상승·SIT하락,FT는ON_TOP상승·SIT/CLIMB하락이다. 단독5종산술평균96.16/97.42%다.
- 각64환경·seed0·900step·기존상자분포·native장면낙상종료정지진단완료. 선행rawAT Ever미완료·armed·rootXYZ≤.3m/s1초첫정지사람목표XY중앙값m(정지/관찰가능)은total SIT1.19(6/6)·CLIMB1.87(4/4)·ON_TOP2.13(13/14),FT1.17(4/5)·2.03(3/3)·2.22(14/14)다. ON_TOPpayload중앙값1.22/1.25m·IQR·초기거리감소율도보존했으며정식개인낙상동결분모와구분했다.
- 정식6회CPU/GPU/JSON재채점정확일치·finite·900frame·실제GPU6UUID·기존near50/거리·초기장면정확일치·크기교집합/5cm격자/비공유상자·checkpoint/Snapshot SHA·Stage1동결actor encoder73/input RMS·FT21k초기화출처·운영301파일·기존학습2개보존검증통과. native2회진입/AT Ever해제/가중치불변도검증했다. 총8회정상완료,운영코드/학습설정변경없음.
- 기존검증된이력의스칼라표를재사용해8k두개와전체/행동별추세를갱신했다. total기록상최고는8k57.42%지만이전5k57.11%와차이가작고,FT최고+3k60.68%·v1최고21k1200step66.35%는유지됐다. [최신비교](output_etc/check/total_ft21k_8k_general_20261107/report.md),[전체추세](output_etc/check/total_ft21k_8k_general_20261107/full_trends.csv),[행동별추세](output_etc/check/total_ft21k_8k_general_20261107/full_action_history.csv),[정지](output_etc/check/total_ft21k_8k_general_20261107/stops.csv),[검증](output_etc/check/total_ft21k_8k_general_20261107/verification.json).

### total6k 합일치 거리 + near50/경로합 대소 제약 없는 새 평가 — GPU6

- 사용자 요청으로total6k를GPU6에서만행동별공유상자교집합·4명8토큰·512환경×seed0/1/2·900step·기존Plane Ever로새평가했다. 합일치의선행두구간각1–5m·후행SIT/CLIMB2–10m·ON_TOP두구간각1–5m/사람목표최소2m·기존충돌검사는유지하고near플래그를0으로설정했다. 선후행두합각2–10m이며대소관계거절검사는없다. 새위치1721쌍중후행합짧음876/김845로두경우모두자연발생했다.
- 새후행동시3seed평균±SD%/동일쌍선행AT Ever는SIT53.38±2.37/97.27,CLIMB51.32±5.15/96.91,ON_TOP65.24±2.57/95.40,전체59.45±0.79/96.17이다. 원합일치near50전체58.87/95.64대비후행+0.58%p,전부후행합긴평가60.21/96.28대비−0.76%p다. 각행동분모415/352/954는이전과동일하며역할/크기는유지하고위치는새로뽑았다. 이번무제약기록의짧음/김전체후행58.68/60.20%,동일쌍AT94.96/97.41%도저장했다.
- 별도native64·seed0·900step·기존상자분포·장면낙상종료정지진단도완료했다. 선행rawAT Ever미완료·armed·rootXYZ≤.3m/s1초첫정지정지/관찰가능은SIT3/3(사람목표XY중앙값1.58m),CLIMB0/0,ON_TOP8/10(사람3.00m·payload1.22m)다. 72쌍중관찰가능13쌍으로표본한계를구분하고정식개인낙상동결분모와혼합하지않았다.
- 정식3회CPU/GPU/JSON재채점정확일치·finite·900frame·실제GPU6UUID·near플래그0·두합/세부구간/최소거리·교집합크기/5cm격자/비공유상자·역할/크기보존·checkpoint SHA/직전동결encoder73와input RMS동일snapshot·운영301파일해시·기존학습2개보존검증통과. native진입/AT Ever해제/가중치불변도통과했다. 4회약169–190초완료. 운영코드/학습config변경없음.
- [결과비교](output_etc/check/total6k_sum_matched_no_near_20261107/report.md),[행동별수치](output_etc/check/total6k_sum_matched_no_near_20261107/success.csv),[정지진단](output_etc/check/total6k_sum_matched_no_near_20261107/stops.csv),[실제경로합별](output_etc/check/total6k_sum_matched_no_near_20261107/route_order.csv),[검증](output_etc/check/total6k_sum_matched_no_near_20261107/verification.json).

### total6k 합일치 + 모든 후행 경로합>선행 경로합 새 평가 — GPU6

- 사용자 요청으로GPU6에서만total6k를행동별공유상자교집합·4명8토큰·512환경×seed0/1/2·900step·기존Plane Ever로새평가했다. 초기XY선행사람→공유상자1–5m·공유상자→목표1–5m,후행SIT/CLIMB사람→목표2–10m,ON_TOP두구간각1–5m·사람→목표최소2m를유지하고모든BEFORE쌍에후행합>선행합을강제했다. near50의후행짧음강제와상충하므로이번검증hook에서만near플래그를해제했다. 원sampler배치를거리/경로조건으로거절하며충돌검사는유지했다. 운영config/코드는미변경이다.
- 새후행동시평균±SD%/동일쌍선행AT Ever는SIT46.84±3.07/98.55,CLIMB57.74±1.96/96.04,ON_TOP66.96±1.86/95.37,전체60.21±1.48/96.28이다. 원래합일치전체56.03/98.07·51.85/94.95·62.90/94.85·58.87/95.64대비후행−9.20/+5.89/+4.06/+1.35%p다. 기존긴쌍부분집합전체64.09/96.73(426쌍)과새조건샘플링60.21/96.28(1721쌍)은구분했다. 새평가분모SIT415/CLIMB352/ON_TOP954는기존전체와같다.
- 별도native64·seed0·900step·기존상자분포·장면낙상종료정지진단도완료했다. 선행rawAT Ever미완료·armed·rootXYZ≤0.3m/s1초첫정지의정지/관찰가능은SIT0/0,CLIMB1/1(사람목표XY1.12m),ON_TOP6/8(사람중앙값4.00m·payload1.24m)다. 전체72쌍중관찰가능9쌍으로정지표본이적고정식개인낙상동결평가와분모를혼합하지않았다.
- 정식3회CPU/GPU/JSON재채점정확일치·finite·900frame·실제GPU6UUID·모든1721쌍후행합엄격히큼·거리/원구간·크기교집합/5cm격자/비공유상자·기존평가와역할/크기동일(위치는새샘플)·snapshot/원checkpoint SHA·동결actor encoder73/input RMS·운영301파일해시·기존학습2개보존검증통과. native진입/AT Ever해제/가중치불변도검증했다. 4회모두약170–191초에정상완료했다.
- [결과비교](output_etc/check/total6k_sum_matched_long_20261107/report.md),[행동별수치](output_etc/check/total6k_sum_matched_long_20261107/success.csv),[정지진단](output_etc/check/total6k_sum_matched_long_20261107/stops.csv),[검증](output_etc/check/total6k_sum_matched_long_20261107/verification.json).

### Fusion v1 21k 일반거리·합일치 초기 경로합 비교 재집계

- 사용자 요청으로total6k와같은초기XY선행/후행경로합기준으로fusionv121k의행동별교집합·near50·각512환경×3seed를CPU재집계했다. 주비교900step의후행동시평균%/동일쌍선행AT Ever짧음→길거나같음은기존SIT63.56/95.09→56.67/95.14(N311/104),CLIMB58.31/94.33→61.04/95.52(282/70),ON_TOP68.71/95.56→74.87/96.62(742/212),전체65.31/95.20→67.48/96.02(1335/386)다. 합일치는SIT58.35/97.68→52.72/99.12(299/116),CLIMB60.24/95.15→62.35/94.85(275/77),ON_TOP68.72/94.45→78.11/98.78(721/233),전체64.57/95.36→68.14/98.15(1295/426)다. SIT은짧은쪽이높아모든행동일관개선은아니며CLIMB긴집단SD11.84/19.04%p도기록했다.
- 원래일반거리1200step도별도저장했고전체짧음/김65.84/68.02%,동일쌍AT95.57/96.30%다. 각조건/seed초기경로합/분류가total6k대응쌍과1e-6m이내일치,초기packet/parent/크기/목표trace정확일치·부분집합건수원점수복원·선행상한·capture/near조건검증통과.900step은직전검증된원traceprefix성공값,1200step은원trace값을재사용했다. 후행경로긴조건만으로대기의존성검증이강해진다고단정하지않았다. 새GPU작업·운영코드/조건/학습변경없음.
- [보고서](output_etc/check/fusion21k_route_order_20261107/report.md), [집계](output_etc/check/fusion21k_route_order_20261107/summary.csv), [검증](output_etc/check/fusion21k_route_order_20261107/verification.json).

### total6k 일반거리·합일치 초기 선행/후행 경로합 비교 재집계

- 사용자 요청으로각조건512환경×3seed×900step·행동별공유상자교집합·기존Plane Ever기록을초기XY경로합L_B<L_A/≥로CPU재집계했다. 선행은사람→공유상자+공유상자→목표,후행SIT/CLIMB는사람→목표,ON_TOP은사람→payload+payload→목표다. 실제이동거리/시간은아니다.
- 셀후행동시3seed평균%/동일쌍선행ATEver의짧음→길거나같음은일반SIT52.35/96.08→57.27/98.05(N311/104),CLIMB37.30/95.02→57.80/95.52(282/70),ON_TOP63.11/93.30→68.98/96.25(742/212),전체55.14/94.30→63.70/96.61(1335/386)다. 합일치는SIT53.33/98.04→63.03/98.25(299/116),CLIMB49.74/94.31→61.16/96.87(275/77),ON_TOP62.12/94.64→65.81/95.86(721/233),전체57.27/95.32→64.09/96.73(1295/426)다. SD·건수·초기각구간거리도저장했다. 후행합이긴집단이높지만선행경로/배치도달라서거리인과효과로단정하지않았다.
- 초기packet/parent/크기/목표trace정확일치·각seed/행동부분집합건수합이원점수복원·선행상한·합일치capture거리3e-5m이내일치·near강제쌍후행합짧음검증통과. 동률0쌍. 새GPU작업·운영코드/성공조건/학습변경없음. [보고서](output_etc/check/total6k_route_order_20261107/report.md),[쌍별](output_etc/check/total6k_route_order_20261107/pairs.csv),[집계](output_etc/check/total6k_route_order_20261107/summary.csv),[검증](output_etc/check/total6k_route_order_20261107/verification.json).

### Fusion v1 최고21k 저장본 XY100%·높이완화 CPU 재채점

- 사용자 요청으로 기존 행동별 공유상자 교집합·일반거리·near50·512환경×3seed 평가 전체후행평균 최고인fusionv121k의저장trace를재채점했다. 원래1200step 기준기존80%/높이±20cm→XY100%만→높이제거만→둘다후행동시3seed평균%는SIT62.61→66.21→62.61→66.46(동일쌍AT95.39),CLIMB58.81→59.38→58.81→60.80(94.61),ON_TOP70.77→73.18→72.45→74.97(96.34),전체66.35→68.67→67.28→70.01(95.75)다. 높이제거는영향분리진단이며운영성공정의로제안하지않았다.
- total과비교용900step에서는전체65.83→68.21→66.70→69.43(동일쌍AT95.40)다. 원래1200stepbaseline은원trace와,900stepbaseline은기존검증된screening_pairs.csv의모든1721쌍개인/동시/선행Ever와각seed정확일치했다. 각조건baseline포함관계·선행상한검증통과. ±30/50cm 중간조건·평균/SD/건수도보존했다. GPU사용·새rollout·운영코드/조건/학습변경없음.
- [비교표](output_etc/check/fusion21k_success_geometry_20261107/report.md), [1200step](output_etc/check/fusion21k_success_geometry_20261107/horizon1200/footprint_height.csv), [900step](output_etc/check/fusion21k_success_geometry_20261107/horizon900/footprint_height.csv).

### total6k 뷰어 인상과 정식 점수 차이 CPU 진단

- 사용자 제공 seed9·common·before_sit/climb/stack 뷰어 명령은 가변 `Stage2BeforeTotal.pth`를 읽으며 확인 시 epoch7500이었다. 뷰어 시작 당시 epoch는 확인하지 않았고6k로 단정하지 않았다. preset 두 동종 BEFORE쌍·공통 크기와 random_scenario·행동별 크기·일반/합일치 거리 조건을 구분했다.
- 저장된 total6k 일반거리/합일치 각각512환경×3seed×900step trace를 CPU로 재구성해 개인·동시·같은쌍 선행AT Plane Ever가6회 모두 원점수와 정확히 일치함을 확인했다. 일반거리 후행 동시/같은쌍 선행AT 합산981/1721(57.00%)/1632/1721(94.83%), 개인1001/1721(58.16%)이다. 일반 실패740쌍의 배타적 기하 분류는 선행미성공89·후행개인만Ever17·선행첫성공후생존없음16·접근반경미진입254·접근후실제XY성공영역미진입348·XY진입후높이불충족16이다. 접촉/행동 원인 자체를 단정하지 않았다.
- 일반거리 선행성공·후행개인미성공634쌍 중120쌍(SIT42/172·CLIMB6/187·ON_TOP72/275)은 선행성공후 가상 support 중심 기하에서만 성공했다. 접근후XY미진입348쌍의 영역까지최소부족거리중앙값11.39cm다. 모든 선행AT Ever쌍에서 실제페널티해제첫프레임과Plane AT첫성공프레임이 일치했다. 동시조건 손실은일반1.16%p/합일치1.28%p로 주된손실이아니다. 뷰어장면직접채점·기한연장효과·충돌원인은미확인이다.
- 추가로 일반거리 동일900step 기록에서 높이±20cm·선행AT·동시조건·낙상mask를 유지하고 후행XY성공영역바깥거리5/10cm를허용하는CPU가정재채점을했다. 기존→+5cm→+10cm의3seed평균후행동시%는 SIT53.52→61.48→67.55(동일쌍AT96.58), CLIMB41.41→43.14→44.88(95.24), ON_TOP64.37→67.50→69.17(93.94), 전체57.01→61.02→63.76(94.82)다. 운영성공조건은미변경이며기존0cm재채점정확일치·완화단조성을3seed모두검증했다. [가정재채점](output_etc/check/total6k_failure_20261107/relaxed_xy.csv).
- 사용자추가요청으로같은일반거리기록에서XY각축80/100%와높이±20/30/50cm/제거를분리재채점했다. 기존→XY100%만→높이제거만→둘다의후행동시3seed평균%는SIT53.52→60.28→53.74→60.50(동일쌍AT96.58),CLIMB41.41→43.17→41.71→44.60(95.24),ON_TOP64.37→67.50→65.11→68.86(93.94),전체57.01→60.74→57.53→61.84(94.82)다. ±30cm만의전체는57.13%다. 높이제거는영향분리용이며실제성공정의로제안하지않았다. baseline정확일치·완화단조성·선행상한3seed검증통과,운영조건미변경. [윗면/높이비교](output_etc/check/total6k_failure_20261107/footprint_height.csv).
- [진단 보고서](output_etc/check/total6k_failure_20261107/report.md), [개별 쌍](output_etc/check/total6k_failure_20261107/pairs.csv), [재구성 검증](output_etc/check/total6k_failure_20261107/verification.json). 새GPU평가·운영코드/설정/학습프로세스 변경은 없다.

### 합일치조건 후보 거리 재집계 및 모델별 2개 새 평가 — GPU6

- 사용자 요청으로 기존 행동별 공유상자 교집합·near50·512환경×3seed의 정수1k 저장본33개를 공통900step으로 CPU 재집계했다. 초기 후행 사람root–선행AT 목표XY≥2m인1433쌍에서 전체 후행 동시평균 상위2개씩을 선정했다: total6k/5k·ft21k+3k/+7k·v121k/25k. 각 부분집합의 같은 쌍 선행AT Ever·행동별 분모·개인Ever·seed별 SD를 저장했다. v1은900step을 재구성하고1200step 재구성이 기존 검증된 기록과 일치함을 확인했다.
- 새 합일치조건은 초기XY 기준 선행 사람→공유상자1–5m·공유상자→목표1–5m, 후행SIT/CLIMB 사람→목표2–10m, ON_TOP 사람→payload1–5m·payload→목표1–5m·사람→목표최소2m다. ON_TOP 사람거리5m상한은 없고 최대10m는 두구간합에서 유도된다. 모든역할의 경로합 가능범위를2–10m로 맞추며 실제 경로합/확률분포를 같게 강제하지 않는다. near50의50%쌍 후행경로합<선행경로합과 기존 충돌검사를 유지했다. 단독은 기존goal5다.
- 선정6개를 GPU6에서만 새로샘플링해 행동별 교집합·near50/loco·4명/8토큰·512환경×seed0/1/2·900step·Plane Ever로 평가했다. 후행 동시평균±SD/같은쌍 선행AT Ever 평균%는 total6k58.87±3.30/95.64·5k56.48±1.26/94.54, ft+3k61.08±2.09/96.32·+7k59.20±1.50/96.39, v121k65.38±1.06/96.05·25k62.04±2.83/95.82다. 이번 후보 중 모델별 최고는 total6k·ft+3k·v121k이며 재집계 추정과 새평가를 구분했다. 새평가 분모SIT415/CLIMB352/ON_TOP954/전체1721쌍은 모든저장본에서 동일하다.
- 모델별 최고 후보의 행동별 후행/같은쌍 선행AT%는 total6k SIT56.03/98.07·CLIMB51.85/94.95·ON_TOP62.90/94.85, ft+3k51.42/97.35·53.22/95.36·68.27/96.26, v121k56.87/98.07·59.95/95.18·71.05/95.49다. 기존 일반거리 공통900step의 전체후행/선행AT는 각각57.01/94.82·60.68/96.21·65.83/95.40이다. 새조건에서 일관된 큰폭 상승을 확인한 것은 아니다.
- 같은6개저장본의 별도64환경·seed0·900step·기존상자분포·native장면낙상종료 정지진단도완료했다. 선행raw AT Ever 미완료·armed·root XYZ≤.3m/s1초 첫정지 사람–목표XY중앙값m(정지/관찰가능)의 최고후보는 total6k SIT1.75(2/2)·CLIMB1.16(1/1)·ON_TOP2.60(14/15), ft+3k.66(3/3)·.08(3/3)·3.01(14/15), v121k2.18(6/6)·2.96(5/5)·2.96(23/23)다. ON_TOP payload중앙값1.40/1.50/1.70m와모든후보의IQR·초기거리감소율을저장했다. 속력기준정지가행동전대기를보장하지않으며정식성공률과native분모를혼합하지않는다.
- 신규정식18회CPU/GPU/JSON 재채점·finite·실제GPU6 UUID·모든후보동일초기장면·거리/near·선행과ON_TOP기존구간·행동별크기교집합/5cm격자/비공유payload·동결actor encoder73/RMS·선택checkpoint SHA·native진입/해제Ever·운영301파일·기존학습2개보존을검증했다. 새ONTOP첫seed의사람거리최대9.83m/326쌍중95쌍이5m초과여서5m상한미적용도확인했다. 거리거절검사와reset최대512회는검증폴더에서만적용했고운영코드/학습설정은변경하지않았다.
- [후보 선정·새 평가 비교표](output_etc/check/sum_matched_candidates_20261107/comparison.md), [33개 거리 재집계](output_etc/check/sum_matched_candidates_20261107/screening.csv), [선정 저장본](output_etc/check/sum_matched_candidates_20261107/selected.json), [새 행동별 성공·선행AT](output_etc/check/sum_matched_candidates_20261107/success.csv), [정지거리·건수·감소율](output_etc/check/sum_matched_candidates_20261107/stops.csv), [검증](output_etc/check/sum_matched_candidates_20261107/verification.json).

### total7k·ft21k+7k 일반 거리 평가 — GPU6, v131k 재사용

- 사용자 요청으로 total7k·ft21k+7k를 GPU6에서만 기존goalroute 거리·행동별 공유상자 교집합·near50/loco·4명/8토큰·512환경×seed0/1/2·900step·Plane Ever로 평가했다. 합일치조건이나2–5m/3–5m 거리override를 적용하지 않았다. v1최신파일은31.5k였으나 사용자 확인에 따라1k간격을 유지해 기존31k1200step 결과를 재사용했다.
- 후행 동시평균±SD/같은 쌍 선행AT Ever 평균%는 total7k SIT53.67±3.11/97.13·CLIMB42.96±2.40/91.49·ON_TOP59.75±3.54/94.25·전체54.80±2.12/94.36, ft21k+7k53.56±7.71/97.60·52.51±6.65/95.07·65.17±2.01/95.92·전체59.81±2.47/96.16이다. 행동별/전체415/352/954/1721쌍을 유지했다. 6k대비전체후행은−2.22/+.57%p이며 total은ON_TOP64.37→59.75%하락, 해당쌍선행AT93.94→94.25%다. 단독5종 평균은97.14/96.62%다. 기존최고total5k·ft+3k·v121k는 유지됐다.
- 같은저장본각64환경·seed0·기존상자분포·900step·native장면낙상종료 정지진단도완료했다. 선행raw AT Ever 미완료·armed·사람root XYZ속력≤.3m/s1초의첫정지 사람–목표XY 중앙값m(정지/관찰가능)은 total SIT1.14(6/7)·CLIMB1.94(3/3)·ON_TOP1.94(13/14), ft1.20(4/4)·1.69(4/4)·2.16(14/14)다. ON_TOP payload거리중앙값1.22/1.25m와IQR/초기거리감소율을저장했다. 정식개인낙상동결성공률과native분모를구분한다.
- 신규정식6회CPU/GPU/JSON 재채점·finite·실제GPU6 UUID·이전평가와seed별초기장면정확일치·크기교집합/5cm격자/비공유상자·동결actor encoder73/RMS·checkpoint epoch/SHA·FT21k초기화계약·native진입/해제Ever·운영301파일보존을검증했다. 작업시작시기존학습2개를보존했다. 기존평가를재사용해35개저장본의전체/행동별추세를갱신했다. 운영코드/설정변경은없다.
- [비교표](output_etc/check/total_ft21k_7k_general_20261107/comparison.md), [전체 추세](output_etc/check/total_ft21k_7k_general_20261107/full_trends.csv), [행동별 추세](output_etc/check/total_ft21k_7k_general_20261107/full_action_history.csv), [최신 행동별](output_etc/check/total_ft21k_7k_general_20261107/latest_actions.csv), [최신 정지 거리](output_etc/check/total_ft21k_7k_general_20261107/latest_stop.csv), [검증](output_etc/check/total_ft21k_7k_general_20261107/verification.json).

### 최고 저장본 3개의 후행 사람–목표 최소 2m/3m 새 샘플링 비교 — GPU6

- 사용자 확인 기준으로 total5k·ft21k+3k·v121k를 사람root–선행AT 목표XY 2–5m/3–5m에 새로 샘플링해 행동별 공유상자 교집합·near50·512환경×seed0/1/2·900step·Plane Ever로 평가했다. ON_TOP도 사람거리이며 목표→payload·payload→사람은 기존1–5m, 선행의두구간1–5m를 유지했다. 기존1–10m는 검증된 기록을 재사용하고 v1도 앞900step으로 맞췄다. 새 조건은 기존goalroute 분포에 거리 거절검사를 추가한 조건부분포다.
- 전체 후행 동시평균±SD/같은 쌍 선행AT Ever 평균%의 기존→2–5m→3–5m는 total57.11±1.20/94.82→59.41±2.69/94.60→59.82±2.37/95.07, ft21k60.68±1.26/96.21→63.19±2.54/96.46→62.12±0.74/95.63, v165.83±0.81/95.40→65.31±1.43/95.98→66.26±1.20/96.92다. 행동별·개인Ever·같은쌍선행AT·seed별건수/SD를 저장했으며SIT/CLIMB/ON_TOP/전체415/352/954/1721쌍을 유지했다. 3m가 일관되게 높은 결과는 아니다.
- 시작부터 페널티 진입 반경 안인 비율의2–5m→3–5m는 v1 SIT45.06→16.87%·CLIMB44.89→19.32%·ON_TOP77.36→74.32%, total/ft ON_TOP37.63→32.29%다. ON_TOP 진입은payload 기준이므로 사람3m하한이 초기페널티 회피를 보장하지 않는다. near50 강제표시를 유지했으며 실제근거리비율과 구분해 저장했다.
- 별도 각64환경·seed0·기존 상자 분포·native 장면 낙상 종료의 정지 진단6회도 완료했다. 선행raw AT Ever 미완료·armed·사람XYZ≤.3m/s1초 첫정지의SIT/CLIMB/ON_TOP 사람XY중앙값m(정지/관찰가능)은 total2–5m1.54(1/2)/1.62(2/2)/2.59(11/12), 3–5m미관찰(0/3)/미관찰(0/2)/3.13(10/11); ft2–5m.60(1/1)/1.65(1/1)/2.96(11/13), 3–5m관찰가능없음(0/0)/미관찰(0/2)/3.84(7/9); v12–5m2.87(12/12)/2.94(10/10)/3.09(27/27), 3–5m3.35(3/3)/2.98(6/7)/3.64(26/27)다. ON_TOP payload중앙값은각각1.21/1.32·1.37/1.44·2.00/2.06m이며 정식 성공률과 native분모를 혼합하지 않는다.
- 3–5m seed2의초기장면1개가원래64회재시도상한에도달해검증폴더에서만reset상한을512회로늘려재실행했다. 거리·충돌·near검사를완화하지않았고초기실패로그를보존했다. 정식18회CPU/GPU/JSON 재채점·finite·실제GPU6 UUID·모델간동일조건/seed의초기장면정확일치·거리/선행/ON_TOP구간·행동별크기교집합/5cm격자/비공유상자보존·native해제Ever검증을통과했다. 운영301파일·선택checkpoint SHA·기존학습3개를보존했다. 운영코드/설정변경은없다.
- [비교표·정지 진단](output_etc/check/best_models_min2_min3_resample_20261107/comparison.md), [행동별 성공·선행AT](output_etc/check/best_models_min2_min3_resample_20261107/success.csv), [초기거리·실제near·진입반경](output_etc/check/best_models_min2_min3_resample_20261107/geometry.csv), [정지거리·건수](output_etc/check/best_models_min2_min3_resample_20261107/stops.csv), [검증](output_etc/check/best_models_min2_min3_resample_20261107/verification.json).

### 최고 저장본 초기 사람–목표3–4m 대4–5m 분리 집계

- 사용자 요청으로 total5k·ft21k+3k·v121k의 기존 동일900step·행동별 교집합·near50·512환경×3seed 개별쌍 기록을 CPU에서 재집계했다. 구간은[3,4)/[4,5)m이며 모든 행동에서 초기 후행 사람root–선행AT 목표XY 기준이다. 3–4/4–5m의 표본수는SIT59/54·CLIMB49/46·ON_TOP217/137쌍으로 모델 간 동일하다.
- 4–5m에서3–4m를 뺀 후행 동시성공 차이는 total/ft/v1 순으로 SIT+7.03/+24.45/+4.58%p, CLIMB−5.90/−14.91/−6.21%p, ON_TOP−6.62/−.63/+5.87%p다. 각구간동일쌍 선행AT·개인Ever·성공건수·seed평균/SD를 함께 저장했다. 4m이상이 일관되게 높은 결과는 아니며 ON_TOP 사람거리 집계로payload–목표 하한의 효과를 판단하지 않는다.
- 모델 간각구간의쌍ID 정확 일치·동시성공상한·두구간합계가검증된기존3–5m의행동별표본/성공/선행AT/개인Ever건수와정확일치·원본pairs파일SHA보존을 확인했다. 추가GPU rollout 및 운영 코드/설정 변경은 없다.
- [성공률·분모·선행AT](output_etc/check/best_models_3to4_vs4to5_20261107/success.csv), [거리구간 차이](output_etc/check/best_models_3to4_vs4to5_20261107/delta.csv), [seed별 건수](output_etc/check/best_models_3to4_vs4to5_20261107/seeds.csv), [검증](output_etc/check/best_models_3to4_vs4to5_20261107/verification.json).

### 모델별 최고 저장본의 초기 후행 거리×행동 성공률 재집계 — CPU만 사용

- 사용자 요청으로 동일 행동별 교집합 평가의 전체 후행 동시평균 최고인 total5k·ft21k+3k·v121k를 선택했다. 기존512환경×seed0/1/2·near50/loco·Plane Ever 기록9회를 재사용하고 추가 GPU rollout은 수행하지 않았다. 거리 기준은 모든 행동에서 초기 후행 사람root–같은 쌍 선행AT 목표의XY 거리이며 ON_TOP도사람 기준이다. 기존1–2/2–3/3–5/5–7/7–10m 구간을 사용했다.
- 공정한 거리 비교용 주 집계는 모두900step으로 맞췄다. v121k는 검증된 최초성공 시점 기록에서900step까지의 개인/동시/선행AT Ever를 재구성했으며 기존1200step 결과도 별도모드로 보존했다. 표는3seed 성공건수/표본수 합산 비율이며 seed별 평균/SD도 별도 저장했다. 모델 간각구간의후행/선행은 정확히 같은 쌍·행동·거리 분모를 사용한다.
- SIT 거리구간별 표본은71/64/113/96/71쌍, CLIMB70/67/95/60/60쌍, ON_TOP147/218/354/188/47쌍이다. CLIMB1–2m 후행/동일쌍 선행AT는 total28.57/90.00·ft42.86/91.43·v138.57/92.86%이며3–5m에서는40.00/94.74·64.21/94.74·68.42/97.89%다. SIT3–5m59.29/97.35·55.75/97.35·71.68/96.46%, 7–10m49.30/97.18·32.39/98.59·43.66/92.96%다. ON_TOP7–10m68.09/93.62·78.72/100.00·78.72/97.87%이나표본47쌍이다. 거리 효과의 원인을 이 집계만으로 확정하지 않는다.
- 모든seed의 역할/거리/구간 분류 모델 간 정확 일치·구간별합계가원래 검증된 행동별 성공/선행AT 건수와seed별로일치·v1900step이기존cutoff건수와일치·동시성공상한·원본/snapshot checkpoint SHA 보존 검증을 통과했다. 운영 코드·설정·학습 프로세스는 변경하지 않았다.
- [거리별 비교](output_etc/check/best_models_initial_distance_20261107/comparison.csv), [건수·평균·SD](output_etc/check/best_models_initial_distance_20261107/distance_success.csv), [개별 쌍](output_etc/check/best_models_initial_distance_20261107/pairs.csv), [검증](output_etc/check/best_models_initial_distance_20261107/verification.json).

### total6k·ft21k+6k·v131k 평가 및 모델별 전체 추세 통합 — GPU6

- 사용자 요청 저장본3개를 GPU6에서만 행동별 공유상자 교집합·512환경×seed0/1/2·4명/8토큰·near50/loco·Plane Ever로 평가했다. total/ft21k900step·v11200step을 유지하고 각64환경·seed0·기존 상자 분포·native 장면 낙상 종료 정지 진단도 완료했다.
- 전체 후행 동시평균±SD/같은 쌍 선행AT 평균%는 total6k57.01±0.74/94.82·ft21k+6k59.25±1.71/96.11·v131k59.25±3.37/96.11이다. 5k·+5k·30k 대비후행−.10/−.25/−3.16%p다. total은1~5k 상승 후 이번6k가5k와 비슷하며, ft21k는추가1~6k에서뚜렷한상승없음, v1은최고21k66.35±1.10/95.75 이후등락이다. 단일 저장본으로 장기 정체/악화를 확정하지 않는다.
- 최신 행동별 후행±SD/동일쌍 선행AT%는 total6k SIT53.52±3.47/96.58·CLIMB41.41±5.50/95.24·ON_TOP64.37±1.78/93.94, ft21k+6k49.24±6.31/97.89·52.31±2.81/94.00·66.17±2.24/96.12, v131k56.11±6.28/96.88·49.36±7.17/94.86·64.36±0.95/96.23이다. SIT/CLIMB/ON_TOP/전체 분모415/352/954/1721쌍을 유지했다. 단독5종 산술평균96.81/97.39/97.47%다.
- v131k 앞900step을 저장 trace에서float64로 별도 재구성했다. 전체58.72±3.37/95.99이며1200step 재구성은 정식 점수와 정확히 일치했다. 기존25~30k900/1200step 집계는 재사용했다. 신규3개와기존22개를 합쳐 추세를 갱신하고, 기존 검증된family 이력의v1 5/6/8/10/11/13/15/18k32행을 추가해33개 저장본·132개 행동별 기록으로 통합했다. 기존21~25k20행의 중복 결과/행동별 분모가 정확히 일치함을 확인했으며64환경 screening·일반/공통 크기 결과는 혼합하지 않았다. 과거19.5k는 정수1k 이력에서 제외했다.
- 최신64환경 진단의 선행 미완료·armed·사람root XYZ속력≤.3m/s·1초 첫 정지 사람–목표XY 중앙값/정지·관찰가능은 total6k SIT1.21m(6/6)·CLIMB1.95m(3/3)·ON_TOP1.97m(14/14), ft21k+6k1.20m(5/5)·2.01m(3/3)·2.15m(14/14), v131k1.94m(8/9)·2.20m(8/8)·2.92m(23/23)다. ON_TOP 진입은payload 기준이며 정지 시payload–목표 중앙값1.25/1.25/1.56m다. IQR/초기거리 감소율을 보존했고 정식 개인 낙상 동결 성공률과 native 분모를 구분한다. 정지 표본만으로 행동 전 대기를 보장하지 않는다.
- 신규9회 CPU/GPU/JSON 재채점 일치·finite·25개 신규/기존 조건의 seed별 초기 장면 정확 일치·동일 행동 분모·물리/alias/5cm격자/비공유payload 보존·동결actor encoder73/RMS·checkpoint epoch/SHA·FT21k 초기화 계약·운영301파일·기존학습3개 보존 검증 통과. 신규 정식 평가 실제GPU UUID 및 native 계획 모두GPU6이다. 운영 코드·설정 변경은 없다.
- [전체 추세](output_etc/check/total_ft21k_v1_6k_31k_20261107/full_trends.csv), [전체 행동별](output_etc/check/total_ft21k_v1_6k_31k_20261107/full_action_history.csv), [최신 행동별](output_etc/check/total_ft21k_v1_6k_31k_20261107/latest_actions.csv), [v1 900/1200step](output_etc/check/total_ft21k_v1_6k_31k_20261107/v1_prefix900.csv), [단독](output_etc/check/total_ft21k_v1_6k_31k_20261107/solo.csv), [정지 추세](output_etc/check/total_ft21k_v1_6k_31k_20261107/stops.csv), [검증](output_etc/check/total_ft21k_v1_6k_31k_20261107/verification.json), [과거 이력 대조](output_etc/check/total_ft21k_v1_6k_31k_20261107/history_verification.json).

### total·ft21k 2~5k·v1 27~30k 미평가12개 행동별 교집합 평가 — GPU6

- 사용자 요청 범위12개를 GPU6에서만 행동별 공유상자 교집합·512환경×seed0/1/2·4명/8토큰·near50/loco·Plane Ever로 평가했다. total/ft21k900step·v1기존1200step을 유지했다. 저장본별64환경·seed0·기존 상자 분포·native 장면 낙상 종료 정지 진단도 모두 완료했다. 최대 두 저장본/네 정식 worker를 같은 GPU6에서 처리했으며0.5k 간격이나 이후 새 저장본으로 범위를 늘리지 않았다.
- 전체 후행 동시평균±SD/같은 쌍 선행AT 평균%의 total2/3/4/5k는38.22±1.95/90.65·48.28±1.05/91.22·54.33±1.73/94.30·57.11±1.20/94.82, ft21k추가2/3/4/5k는55.17±2.09/96.17·60.68±1.26/96.21·57.01±4.05/96.68·59.50±0.37/95.93이다. v127/28/29/30k는62.89±2.21/96.75·63.77±2.41/96.51·59.52±1.99/96.22·62.41±1.58/96.34다. SIT/CLIMB/ON_TOP/전체 분모415/352/954/1721쌍을 유지했다. total은1~5k 연속 상승, ft21k는1k 대비개선없음, v1은등락이며 기존최고21k66.35±1.10/95.75를 넘지 못했다.
- 최신 행동별 후행±SD/동일쌍 선행AT%는 total5k SIT51.82±1.53/97.60·CLIMB43.07±5.19/92.35·ON_TOP64.73±4.08/94.56, ft21k+5k48.55±2.95/96.40·49.93±1.72/93.99·67.83±2.19/96.43, v130k56.65±6.50/97.38·55.27±4.57/95.75·67.63±1.51/96.13이다. 최신 단독5종 산술평균은96.98/96.77/97.43%다. 기존 완료분을 재사용해 total/ft21k0.5~5k·v121~30k의 행동별/전체/단독 추세를 통합했다.
- v125~30k 저장 trace를float64로 재구성해 앞900step도 별도 집계했다. 최신30k900step 전체61.72±1.84/96.05이며 total5k57.11±1.20/94.82·ft21k+5k59.50±0.37/95.93과 제한 시간을 맞춰 비교할 수 있다. 모든1200step 재구성이 정식 점수와 일치했고 추가 GPU rollout은 없다.
- 최신64환경 진단의 선행 미완료·armed·사람root XYZ속력≤.3m/s·1초 첫 정지 사람–목표XY 중앙값/정지·관찰가능은 total5k SIT1.08m(5/6)·CLIMB1.79m(5/5)·ON_TOP2.00m(13/14), ft21k+5k1.23m(5/5)·1.69m(4/4)·2.14m(14/14), v130k1.86m(9/10)·2.14m(7/7)·2.93m(23/23)다. ON_TOP 진입은payload 기준이며 정지 시payload–목표 중앙값1.22/1.21/1.56m다. 초기 사람–목표 거리 감소율 중앙값(SIT/CLIMB/ON_TOP)은total31.78/2.56/1.67%, ft19.34/−14.28/2.08%, v112.50/2.28/.68%다. 정식 개인 낙상 동결 성공률과 native 진단 분모를 구분하며 정지 표본만으로 행동 전 대기를 보장하지 않는다. 저장본별IQR/건수/감소율도 보존했다.
- 신규36회 재채점 CPU/GPU/JSON 일치·finite·22개 신규/기존 비교조건의 seed별 초기 장면 정확 일치·동일 행동 분모·물리/alias 크기·5cm격자·비공유payload 보존·동결actor encoder73/RMS·checkpoint epoch/SHA·FT21k 초기화 계약·운영301파일·기존학습3개 보존 검증 통과. 신규 정식 평가 실제GPU UUID 및 native 계획 모두GPU6이다. 운영 코드·설정 변경은 없다.
- [모델별 추세](output_etc/check/total_ft21k_v1_2to5_27to30_20261107/trends.csv), [전체 행동별](output_etc/check/total_ft21k_v1_2to5_27to30_20261107/success.csv), [최신 행동별](output_etc/check/total_ft21k_v1_2to5_27to30_20261107/latest_actions.csv), [v1 900/1200step](output_etc/check/total_ft21k_v1_2to5_27to30_20261107/v1_prefix900.csv), [단독](output_etc/check/total_ft21k_v1_2to5_27to30_20261107/solo.csv), [정지 추세](output_etc/check/total_ft21k_v1_2to5_27to30_20261107/stops.csv), [모델별 최고](output_etc/check/total_ft21k_v1_2to5_27to30_20261107/best.csv), [검증](output_etc/check/total_ft21k_v1_2to5_27to30_20261107/verification.json).

### total·ft21k·v1 신규 정수1k 저장본 평가 — ft21k 1k·v1 26k

- 사용자 요청으로 세 실험의 새 정수1k 저장본과 평가 이력을 대조했다. total은 최신 정수1k가 이미 평가되어 재사용했고, 미평가 ft21k 1k·fusion v1 26k를 GPU6에서만 행동별 공유상자 교집합·512환경×seed0/1/2·4명/8토큰·near50/loco·Plane Ever로 평가했다. total/ft21k900step, v1기존1200step을 유지했으며0.5k 간격 추가 평가는 하지 않았다.
- 후행 동시평균±SD/같은 쌍 선행AT 평균%는 ft21k 1k SIT53.02±1.91/97.38·CLIMB46.62±3.79/93.55·ON_TOP67.41±0.58/96.23·전체59.68±1.02/95.94, v1 26k SIT54.69±4.63/97.46·CLIMB52.91±1.91/93.73·ON_TOP66.31±3.90/95.18·전체60.75±2.67/95.41이다. 분모415/352/954/1721쌍을 유지했다. ft21k0.5k 전체59.12±1.65/95.64 대비+0.57%p, v1 25k65.97±1.52/95.82 대비−5.22%p다. 단독5종 평균은ft21k97.39→97.42%, v197.50→97.27%다.
- 저장 trace를float64로 재구성해 v1 앞900step도 집계했다. v1 25k 전체64.93±1.82/95.70→26k59.87±2.37/95.18이며, 같은900step의ft21k1k59.68±1.02/95.94와 비슷하다. 미세조정 source21k 앞900step65.83±0.81/95.40보다는ft1k가6.14%p 낮다. 모든1200step 재구성 결과가 기존 정식 점수와 일치하며 추가 GPU rollout은 없다.
- 두 새 저장본의 별도64환경·seed0·기존 상자 분포·native 장면 낙상 종료 정지 진단도 GPU6에서 완료했다. 선행 미완료·armed·root XYZ≤.3m/s를1초 유지한 첫 정지의 사람–목표XY 중앙값/정지·관찰가능은ft21k1k SIT1.18m(4/5)·CLIMB2.01m(3/3)·ON_TOP2.27m(15/16), v126k1.83m(9/9)·2.13m(8/8)·2.92m(23/23)다. ON_TOP은payload 진입이며 정지 시payload–목표 중앙값은1.28/1.56m다. 정식 개인 낙상 동결 성공률과 native 진단 분모를 구분하며, 정지 표본을 행동 전 대기 성공으로 단정하지 않는다. 초기거리 감소율/IQR도 보존했다.
- CPU/GPU/JSON 재채점·finite·seed별 초기 장면 정확 일치·동일 행동 분모·동결actor encoder73/RMS·checkpoint epoch/SHA·FT21k 초기화 계약·원본checkpoint·운영301파일·작업 시작 시 기존학습3개 보존 검증 통과. 정식 평가의 실제GPU UUID와 native 계획 모두GPU6임을 확인했다. 운영 코드·설정 변경은 없다.
- [전체 추세](output_etc/check/total_ft21k_v1_integer_20261107/overview.csv), [행동별](output_etc/check/total_ft21k_v1_integer_20261107/success.csv), [v1 900/1200step](output_etc/check/total_ft21k_v1_integer_20261107/v1_prefix900.csv), [단독](output_etc/check/total_ft21k_v1_integer_20261107/solo.csv), [정지](output_etc/check/total_ft21k_v1_integer_20261107/stops.csv), [검증](output_etc/check/total_ft21k_v1_integer_20261107/verification.json), [최종 저장본 확인](output_etc/check/total_ft21k_v1_integer_20261107/final_inventory.json).

### total 0.5k·1k와 v3_ep900 동학습량 비교 — GPU6

- 사용자 요청으로 total 0.5k와 새로 생성된1k, v3_ep900 0.5k를 GPU6에서 행동별 공유상자 교집합·512환경×seed0/1/2×900step·4명/8토큰·near50/loco·Plane Ever로 평가했다. ep900 1k 정식 평가와 ep900 두 저장본의 native64 진단은 동일 SHA의 기존 결과를 재사용했으며 새 GPU 실행은 모두6이다.
- 후행 동시평균±SD/같은 쌍 선행AT 평균%는 0.5k ep900→total: SIT5.92±1.71/88.66→5.09±1.59/86.30·CLIMB5.39±0.28/63.11→5.15±2.40/63.65·ON_TOP15.15±2.92/70.54→19.39±0.54/71.49·전체11.00±1.54/73.37→13.02±1.29/73.48이다. 1k는 SIT8.98±1.51/93.06→8.28±3.91/94.19·CLIMB7.61±2.72/76.29→15.12±1.93/76.45·ON_TOP17.82±0.09/77.06→29.53±3.38/82.39·전체13.60±0.03/80.76→21.51±2.78/84.02다. 분모는415/352/954/1721쌍이며1k 전체 차이+7.91%p다. 단독5종 산술평균은0.5k ep90090.43/total86.15%, 1k91.67/92.29%다. 초기 저장본 비교이며 장기 우위는 미확인이다.
- total 두 저장본의 별도64환경·seed0·기존 상자 분포·native 장면 낙상 종료 진단도 GPU6에서 완료했다. 선행 미완료·armed·root XYZ≤.3m/s·1초 첫 정지의 사람–목표XY 중앙값과 정지/관찰가능은0.5k SIT.68m(3/14)·CLIMB.07m(11/14)·ON_TOP.51m(12/23), 1k .96m(8/9)·.37m(10/14)·1.06m(12/21)다. total ON_TOP 진입은payload 기준이며 정지 시payload–목표 중앙값은.23/.86m다. 정지 표본은 행동 전 대기를 보장하지 않으며 정식 개인 낙상 동결 성공률과 진단 분모를 구분한다.
- CPU/GPU/JSON 재채점·finite·네 조건 seed별 초기 장면 정확 일치·동일 행동 분모·checkpoint epoch/SHA·Stage1 26k 신규 초기화·기하/alias/비공유payload·동결actor encoder73/RMS·운영301파일·기존학습4개 보존 검증 통과. 두 실험의 YAML은 이름과 ON_TOP 진입 human/payload만 다름을 확인했다. 운영 코드·설정 변경은 없다.
- [비교표](output_etc/check/total_vs_ep900_early_20261107/comparison.csv), [단독](output_etc/check/total_vs_ep900_early_20261107/solo.csv), [정지거리·건수·초기거리 감소율](output_etc/check/total_vs_ep900_early_20261107/stops.csv), [검증](output_etc/check/total_vs_ep900_early_20261107/verification.json).

### v3_ep900 3k 행동별 교집합512환경×3회 평가 — GPU6만 사용

- 사용자 요청으로 v3_ep900 3k를 GPU6에서만 행동별 공유상자 교집합·512환경×seed0/1/2×900step·4명/8토큰·near50/loco·Plane Ever로 평가했다. GPU6에서 두 평가 worker로 세 seed를 처리했으며 다른 GPU에는 평가를 올리지 않았다.
- 후행 동시평균±SD/같은 쌍 선행AT 평균%는 SIT45.89±6.21/94.58·CLIMB36.75±9.89/86.42·ON_TOP41.48±4.16/89.41·전체41.60±3.53/90.07다. 분모415/352/954/1721쌍이며2k 전체35.93±2.22/89.64 대비후행+5.68%p다. seed0/1/2 전체도38.45→45.15·35.10→38.10·34.24→41.57%로 모두 상승했다. 단독5종 산술평균95.40→95.27%는 비슷하지만 SIT96.23±1.61→90.60±3.44%가 하락했다.
- 같은3k의 별도64환경·seed0·기존 상자 분포·native 장면 낙상 종료 정지 진단도 GPU6에서 수행했다. 선행 미완료·armed·root XYZ≤.3m/s·1초 첫 정지의 사람–목표XY 중앙값[IQR]은 SIT1.22[1.14,1.25]·CLIMB1.62[1.59,1.87]·ON_TOP1.45[1.12,1.73]m, 정지/관찰가능6/7·5/5·9/12다. ON_TOP 진입은사람 기준이며 정지 시payload–목표 중앙값1.43m다. 초기 사람–목표 거리 감소율 중앙값24.84/13.68/21.27%를 보존했다. 정식 개인 낙상 동결 성공률과 native 진단의 종료·분모를 구분한다.
- armed와 관계없는 추가 root 정지 진단에서는 선행 미완료 중43/72쌍이1초 조건을 만족했고, 최초 정지 시unarmed였던25쌍의 사람–목표 거리 중앙값은2.73m였다. 페널티 활성화 후 정지 표본을 전체 대기 위치로 해석하지 않는다.
- 3회900step·finite·CPU/GPU/JSON 재채점 일치·2k 및 기존21k와 seed별 초기 장면 정확 일치·공유 물리/alias 크기·5cm 격자·비공유payload 보존·동결actor encoder73/RMS·원본checkpoint·운영301파일·기존학습4개 보존 검증 통과. native 진입/AT Ever 해제·동일checkpoint·정식 평가 실제GPU UUID와native계획 모두GPU6만 사용함도 확인했다. 운영 코드·설정 변경은 없다.
- [성공률](output_etc/check/fusionv3_ep900_3000_family512x3_20261107/comparison.csv), [1k→2k→3k](output_etc/check/fusionv3_ep900_3000_family512x3_20261107/history.csv), [2k 대비](output_etc/check/fusionv3_ep900_3000_family512x3_20261107/delta_vs_2k.csv), [단독](output_etc/check/fusionv3_ep900_3000_family512x3_20261107/solo.csv), [정지64](output_etc/check/fusionv3_ep900_3000_family512x3_20261107/stop/metrics.json), [초기거리 감소율](output_etc/check/fusionv3_ep900_3000_family512x3_20261107/stop/closure.json), [진입 전 정지](output_etc/check/fusionv3_ep900_3000_family512x3_20261107/pre_release_stop.json), [검증](output_etc/check/fusionv3_ep900_3000_family512x3_20261107/combined_verification.json).

### total_ft21k 추가학습0.5k 행동별 교집합512환경×3회 평가

- 사용자 요청으로 `stage2_before_total_ft21k`의epoch500 저장본을 GPU2/6/7에서 행동별 공유상자 교집합·512환경×seed0/1/2×900step·4명/8토큰·near50/loco·Plane Ever로 평가했다. 원본 v1 21k는 기존 trace의900step 절단 결과를 재사용해 제한 시간을 맞춰 비교했다. 동일 행동·쌍·분모415/352/954/1721을 유지했다.
- 후행 동시평균±SD/같은 쌍 선행AT 평균%는 SIT52.54±5.35/97.64·CLIMB50.43±4.57/94.28·ON_TOP65.17±2.66/95.29·전체59.12±1.65/95.64다. 원본21k 앞900step은전체65.83±0.81/95.40이며 후행−6.71%p다. BEFORE 선행/후행을 제외한 단독5종 산술평균은97.23→97.39%로 유지됐다. 추가0.5k의 단일 저장본이므로 이후 회복 여부는 아직 판단하지 않는다.
- 같은 저장본의 별도64환경·seed0·기존 상자 분포·native 장면 낙상 종료 정지 진단은 GPU2에서 수행했다. 선행 미완료·armed·root XYZ≤.3m/s·1초 첫 정지의 사람–목표XY 중앙값[IQR]은 SIT1.27[1.22,1.31]·CLIMB1.68[1.03,2.06]·ON_TOP2.30[1.84,3.25]m, 정지/관찰가능5/5·4/4·15/15다. ON_TOP 진입은payload 기준이며 정지 시payload–목표 중앙값1.28m다. 초기 사람–목표 거리 감소율 중앙값21.76/−11.16/1.24%를 보존했다.
- 추가 CPU 진단에서 선행 미완료 중armed 여부와 관계없이 root 속력 조건을1초 만족한65/72쌍 중43쌍은 최초 정지 시점에unarmed였다. 해당43쌍의 사람–목표 거리 중앙값3.88m다. 페널티 활성화 후 표본만으로 전체 정책이 가까이 접근해 대기한다고 해석하지 않는다. 정식 개인 낙상 동결 성공률과 native 진단의 종료·분모도 구분한다.
- 3회900step·finite·CPU/GPU/JSON 재채점 일치·기존21k 및 v3_ep900와 seed별 초기 장면 정확 일치·공유 물리/alias 크기·5cm 격자·비공유payload 보존·동결actor encoder73/RMS·원본checkpoint·운영301파일·기존학습4개 보존 검증 통과. FT epoch500·실제21k source SHA·최초optimizer/epoch/frame 초기화·native 진입/AT Ever 해제·실제GPU UUID2/6/7도 확인했다. 운영 코드·설정 변경은 없다.
- [성공률](output_etc/check/total_ft21k_500_family512x3_20261107/comparison.csv), [원본900step 비교](output_etc/check/total_ft21k_500_family512x3_20261107/comparison_vs_source900.csv), [단독](output_etc/check/total_ft21k_500_family512x3_20261107/solo.csv), [정지64](output_etc/check/total_ft21k_500_family512x3_20261107/stop/metrics.json), [구역 진입 전 정지](output_etc/check/total_ft21k_500_family512x3_20261107/pre_release_stop.json), [검증](output_etc/check/total_ft21k_500_family512x3_20261107/combined_verification.json).

## 2026-11-06

### fusion v1 25k 행동별 교집합512환경×3회 평가

- 사용자 요청으로 fusion v1 25k를 GPU2/6/7에서 행동별 공유상자 교집합·512환경×seed0/1/2×1200step·4명/8토큰·near50/loco·Plane Ever로 평가했다. 공통 교집합 추가 평가는 수행하지 않았다.
- 후행 동시평균±SD/같은 쌍 선행AT 평균%는 SIT61.99±6.20/97.09·CLIMB59.02±5.65/93.17·ON_TOP70.32±0.76/96.22·전체65.97±1.52/95.82다. 분모415/352/954/1721쌍이며24k 전체60.81±2.77 대비+5.16%p, 기존 최고21k66.35±1.10보다0.38%p 낮다. BEFORE 선행/후행을 제외한 단독5종 산술평균은24k96.82→25k97.50%였다.
- 같은25k의 별도64환경·seed0·기존 상자 분포·native 장면 낙상 종료 정지 진단은 GPU2에서 완료했다. 선행 미완료·armed·root XYZ≤.3m/s·1초 첫 정지의 사람–목표XY 중앙값[IQR]은 SIT1.77[1.28,2.31]·CLIMB2.12[1.87,2.41]·ON_TOP2.94[1.98,3.83]m, 정지/관찰가능9/9·8/8·23/23다. ON_TOP 진입은payload 기준이며 정지 시payload–목표 중앙값1.56m다. 초기 사람–목표 거리 감소율 중앙값11.77/4.99/0.92%를 보존했다. 정식 개인 낙상 동결 성공률과 native 진단의 종료/분모를 구분한다.
- 3회1200step·finite·CPU/GPU/JSON 재채점 일치·21k/24k와 seed별 초기 장면 정확 일치·공유 물리/alias 크기·5cm 교집합 격자·비공유payload 보존·동결actor encoder73/RMS·원본checkpoint·운영301파일·기존학습4개 보존 검증 통과. native 진입/AT Ever 해제·동일checkpoint·정식 평가의 실제GPU UUID가2/6/7과 일치함도 확인했다. 운영 코드·설정 변경은 없다.
- [성공률](output_etc/check/fusion_25000_family512x3_20261106/comparison.csv), [21k/24k 비교](output_etc/check/fusion_25000_family512x3_20261106/delta_vs_21k_24k.csv), [추세](output_etc/check/fusion_25000_family512x3_20261106/history.csv), [단독](output_etc/check/fusion_25000_family512x3_20261106/solo.csv), [정지64](output_etc/check/fusion_25000_family512x3_20261106/stop/metrics.json), [초기거리 감소율](output_etc/check/fusion_25000_family512x3_20261106/stop/closure.json), [검증](output_etc/check/fusion_25000_family512x3_20261106/combined_verification.json).

### v3_ep900 2k 행동별 교집합512환경×3회 평가

- 사용자 요청으로 v1/v3_ep900 저장본을 확인했다. v1은 최신24.5k·최신정수24k로 기존 평가와 같아0.5k 간격 재평가는 생략하고, 새 v3_ep900 2k를 GPU2/6/7에서 행동별 공유상자 교집합·512환경×seed0/1/2×900step·4명/8토큰·near50/loco·Plane Ever로 평가했다. 종료 시점에도 v1의 새 정수 저장본은 없다.
- 후행 동시평균±SD/같은 쌍 선행AT 평균%는 SIT32.89±2.20/94.22·CLIMB31.27±2.14/85.36·ON_TOP38.95±2.52/89.22·전체35.93±2.22/89.64다. 분모415/352/954/1721쌍이며1k 전체13.60±0.03/80.76 대비후행+22.33%p다. 선행/후행을 제외한 단독5종 성공률의 산술평균도91.67→95.40%였다.
- 같은2k의 별도64환경·seed0·기존 상자 분포·native 장면 낙상 종료 정지 진단은 GPU2에서 수행했다. 선행 미완료·armed·root XYZ≤.3m/s·1초 첫 정지의 사람–목표XY 중앙값[IQR]은 SIT1.17[1.12,1.76]·CLIMB1.71[1.62,1.74]·ON_TOP1.59[1.22,1.77]m, 정지/관찰가능8/9·6/6·9/12다. ON_TOP 진입은 사람 기준이며 정지 시payload–목표 중앙값1.22m다. 초기 사람–목표 거리 감소율 중앙값30.23/18.75/27.02%를 보존했다. 정식 개인 낙상 동결 성공률과 native 진단의 종료/분모를 구분한다.
- 3회900step·finite·CPU/GPU/JSON 재채점 일치·1k 및 기존 v1 평가와 seed별 초기 장면 정확 일치·공유 물리/alias 크기·5cm 교집합 격자·비공유payload 보존·동결actor encoder73/RMS·원본checkpoint·운영301파일·기존학습4개 보존 검증 통과. native 진입과AT Ever 해제·동일checkpoint·모든 rollout GPU2/6/7 제한도 확인했다. 운영 코드·설정 변경은 없다.
- [성공률](output_etc/check/fusionv3_ep900_2000_family512x3_20261106/comparison.csv), [1k→2k](output_etc/check/fusionv3_ep900_2000_family512x3_20261106/history.csv), [단독](output_etc/check/fusionv3_ep900_2000_family512x3_20261106/solo.csv), [정지64](output_etc/check/fusionv3_ep900_2000_family512x3_20261106/stop/metrics.json), [초기거리 감소율](output_etc/check/fusionv3_ep900_2000_family512x3_20261106/stop/closure.json), [검증](output_etc/check/fusionv3_ep900_2000_family512x3_20261106/combined_verification.json).

### stage2_before_total_ft21k: v1 최고21k에서 모델·정규화만 이관해 미세조정

- `stage2_before_total_ft21k` 환경/학습 YAML과 train/test/VNC/eval wrapper를 추가했다. 환경·보상·모델·PPO/AMP는 total과 같고(+1.5m·ON_TOP payload·가중치1.0·900step), 초기값만 평가했던 fusion v1 21k로 변경했다. 전용 `before_total_finetune.py`와 `_init_before()`의 명시적 `stage2_weights` 분기가 actor/critic/AMP 모델 전체와 입력·AMP·value 정규화를 이관한다. optimizer·epoch·frame·환경 상태는 복원하지 않으며 기존 Stage1 전이·일반 resume 계약은 유지한다.
- 원본21k SHA가 기존 평가와 동일함을 확인했다. 새 학습의 optimizer state/epoch/frame은0이며 전체279텐서를 이관한다. 미세조정 저장본은 일반 `RESUME_CHECKPOINT`로 이어 학습하고, 이때는 원본21k 초기화를 반복하지 않는다. registry·launch·정식 평가 dispatch와33개 설정/명령·구조·사용법 문서를 연결했다.
- CPU103개·shell 문법·문서 경로·GPU2 4명/1024환경/8토큰 CHECK 신규 미세조정(epoch2/frame262144/optimizer step96)·해당 저장본 CHECK 재개(epoch3/frame393216/step144)·test wrapper4환경/6step×2회 재로드 통과. 동결 actor encoder73텐서/입력RMS·finite·유효 업데이트·기존 YAML/wrapper194파일·원본21k·기존학습3개 보존을 확인했다. 장기 본학습은 시작하지 않았으며 성능 개선 여부는 아직 평가하지 않았다.
- [환경](tokenhsi/data/cfg/multi_agent/stage2_before_total_ft21k.yaml), [학습](tokenhsi/data/cfg/train/rlg/stage2_before_total_ft21k.yaml), [초기화](tokenhsi/utils/before_total_finetune.py), [실행 명령](markdowns/config.md), [검증](output_etc/check/before_total_ft21k_config_20261106/verification.json), [학습](output_etc/check/before_total_ft21k_config_20261106/train.log), [재개](output_etc/check/before_total_ft21k_config_20261106/resume.log), [재로드](output_etc/check/before_total_ft21k_config_20261106/reload.log).

### stage2_before_total: Stage1 26k 신규 학습·payload +1.5m·900step

- `v3_ep900`을 기반으로 `stage2_before_total` 환경/학습 YAML과 train/test/VNC/eval wrapper를 추가했다. Stage1 `Stage1JointPushDensity3Cone90Finetune_00026000.pth`에서 새 Stage2 학습을 시작하며 자동 resume는 없다. 학습 저장 이름은 `Stage2BeforeTotal`이다.
- ON_TOP 대기 진입 기준만 human→payload로 변경했다. SIT/CLIMB 사람–목표XY·반경+1.5m·가중치1.0·모든 행동의 사람 root XYZ 속력 감점·진입Ever/선행AT 성공Ever 해제·fusion/virtual/거리/near50/보상/학습 설정은 기반과 같다. 학습·평가·viewer 기본900step을 registry/launch/evaluate에 연결하고32개 설정 및 실행 문서를 갱신했다.
- CPU85개·shell 문법·문서 경로·GPU2 4명/1024환경/8토큰 CHECK 신규 학습(epoch2 저장)·같은 GPU에서 test wrapper4환경/기본900step 재로드를 통과했다. 전이 보고서의 source_epoch26000·동결 actor encoder73텐서/입력RMS·finite·유효 학습 업데이트를 확인했다. 기존 YAML/wrapper188파일과 기존학습3개를 보존했으며 장기 본학습은 시작하지 않았다. 정지 학습 효과/성공률은 아직 측정하지 않았다.
- [환경](tokenhsi/data/cfg/multi_agent/stage2_before_total.yaml), [학습](tokenhsi/data/cfg/train/rlg/stage2_before_total.yaml), [실행 명령](markdowns/config.md), [검증](output_etc/check/before_total_config_20261106/verification.json), [학습 로그](output_etc/check/before_total_config_20261106/train.log), [재로드](output_etc/check/before_total_config_20261106/reload.log).

### fuv1 24k 행동별 교집합512환경×3회 평가

- 사용자 요청으로 새1k 단위 checkpoint를 확인해 fuv1 24k를 GPU2에서 행동별 공유상자 교집합·512환경×seed0/1/2×1200step·4명/8토큰·near50/loco·Plane Ever로 평가했다. fuv2는 확인 시점과 평가 종료 시점 모두 최신22.5k이며 새 정수1k 저장본은 없다.
- 후행 동시평균±SD/같은 쌍 선행AT 평균%는 SIT53.89±6.49/97.35·CLIMB52.71±4.44/94.54·ON_TOP66.98±1.40/96.22·전체60.81±2.77/96.17다. 분모415/352/954/1721쌍이며, 23k 대비전체+4.62%p로 회복했으나21k 최고66.35±1.10%보다5.54%p 낮다.
- 같은24k의 별도64환경·seed0·기존 상자 분포·native 장면 낙상 종료 정지 진단도 GPU2에서 완료했다. 선행 미완료·armed·root XYZ≤.3m/s·1초 첫 정지의 사람–목표XY 중앙값 SIT1.87·CLIMB2.22·ON_TOP2.91m, 정지/관찰가능9/9·8/8·23/23이다. ON_TOP 진입은 기존payload 기준이며 정지 시 payload–목표 중앙값1.56m다. 초기거리 감소율 중앙값13.55/3.35/1.83%를 보존했다. 정식 개인 낙상 동결 성공률과 구분한다.
- 3회1200step·finite·CPU/GPU/JSON 재채점 일치·21k/23k와 세 seed 초기 장면 정확히 동일·기하/alias/비공유payload·동결actor encoder73/RMS·원본checkpoint·운영288파일·기존학습3개 보존 검증 통과. 정지 진단 runtime 진입/AT Ever 해제·동일checkpoint·모든rollout GPU2도 확인했다. 운영코드/설정 변경은 없다.
- [성공률](output_etc/check/fusion_24000_family512x3_20261106/comparison.csv), [21k/23k 비교](output_etc/check/fusion_24000_family512x3_20261106/delta_vs_21k_23k.csv), [정지64](output_etc/check/fusion_24000_family512x3_20261106/stop/metrics.json), [검증](output_etc/check/fusion_24000_family512x3_20261106/combined_verification.json).

### 사람 기준 대기 v3_ep900·v4·v2 초기 checkpoint64환경 비교

- GPU2만 사용해 행동별 교집합·64환경·seed0·near50/loco·4명/8토큰에서 v3_ep900 0.5/1/1.5k, v4 0.5k, v2 0.5/1/1.5k 총7개를 평가했다. v4는0.5k 이후 번호 저장본이 없다. 각 실험 기본900/1200step을 유지하고 저장 trace를float64 CPU로 재구성해 공통900step 결과도 집계했다. 후행 동시성공은1200까지의 평가와900절단값이 모두 같았다. 행동별 분모 SIT12·CLIMB12·ON_TOP41·전체65쌍이며 단일seed 탐색 결과다.
- 공통900step 후행 전체/같은 쌍 선행AT%는 v3_ep900 0.5/1/1.5k에서13.85/81.54→9.23/84.62→16.92/90.77, v4 0.5k13.85/75.38, v2 0.5/1/1.5k에서13.85/73.85→15.38/78.46→24.62/83.08이다. CLIMB는7개 모두0/12다. v2 1.5k ON_TOP36.59/82.93 대 v3_ep90024.39/90.24이며 작은 표본/서로 다른 학습horizon으로 반경 단독 효과를 확정하지 않는다.
- 같은 checkpoint의 별도64환경·기존 상자 분포·native 장면 낙상 종료 정지 진단도 공통900step으로 비교했다. v3_ep900 1k는 앞서 수행한 동일SHA의900step64환경 진단을 재사용하고6회를 추가했다. 선행 미완료·armed·root XYZ≤.3m/s·1초 첫 정지의 사람–목표 중앙값 SIT/CLIMB/ON_TOP은 v3_ep900 1.5k1.10/1.33/.89m(7/7·8/9·11/15), v2 1.5k1.31/1.56/1.90m(9/10·10/12·18/19)다. 정지 후 선행AT가 성공하고3초 이상 관찰 가능한 쌍에서3초 내0.3m/s 초과10프레임 재이동은 v3_ep9008/16·v2 25/29다. 정지 위치·초기거리 감소율·ON_TOP payload 거리·재이동 건수와IQR을 모두 보존했다. 정식 성공률과 native 진단의 장면/분모를 구분한다.
- 7회 공식 CPU/GPU/JSON 재채점·공통900/full trace 재구성 일치·모델 간 동일64환경 초기 장면·기하/alias/비공유payload·동결actor encoder73/RMS·원본checkpoint·운영288파일·기존학습3개 보존 검증 통과. 운영코드/설정 변경은 없다.
- [공통900 성공률](output_etc/check/human_wait_early_family64_20261106/success_common900.csv), [각 원래horizon 포함](output_etc/check/human_wait_early_family64_20261106/success_by_horizon.csv), [정지·감소율·재이동](output_etc/check/human_wait_early_family64_20261106/native_stop_common900.csv), [개별쌍](output_etc/check/human_wait_early_family64_20261106/native_stop_pairs_common900.csv), [검증](output_etc/check/human_wait_early_family64_20261106/combined_verification.json).

### fusion human3: v1 21k에서 ON_TOP 사람 기준으로 재개

- 사용자 요청으로 `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_human3_resume21k` 환경/학습 YAML과 train/test/VNC/eval wrapper를 추가했다. v1 대비 ON_TOP 대기 진입 기준만 payload→human이며 반경+3m·가중치1.0·SIT/CLIMB·1200step·거리/near50·보상·모델·동결/PPO/AMP를 유지한다. registry·launch·정식 평가 dispatch와 config/structure의31개 설정·명령을 연결했다.
- train wrapper는 평가했던 v1 21k를 기본 source로 사용한다. 엄격한 기존 resume 계약은 유지하고, 전용 `before_fusion_resume.py`가 새 설정의 재개용 복사본/sidecar를 `output_etc/check/before_fusion_human3_resume21k_initial/`에 자동 생성한다. 모델·optimizer·모든 정규화·frame·epoch21000은 원본과 정확히 같고 원본 PTH는 변경하지 않는다. 이후 이 실험의 저장본은 변환 없이 resume한다. epoch도21k부터 이어진다.
- CPU93개·네wrapper문법·GPU2 4명/1024환경/8토큰 CHECK 재개 학습 정상 종료(epoch21002 저장)·유효 업데이트·finite·Stage1 actor encoder73텐서/scene RMS 동결·새test wrapper4환경/6step×2회 재로드 통과. 원본21k가 평가에 쓴 SHA와 같으며 기존 YAML/wrapper30개·기존학습3개 보존을 확인했다. 장기 본학습은 시작하지 않았다.
- [환경](tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_human3_resume21k.yaml), [실행 명령](markdowns/config.md), [검증](output_etc/check/before_fusion_human3_resume21k_config_20261106/verification.json), [학습](output_etc/check/before_fusion_human3_resume21k_config_20261106/train.log), [재로드](output_etc/check/before_fusion_human3_resume21k_config_20261106/reload.log).

### fuv2 22k 행동별 교집합512환경×3회 평가

- GPU2만 사용해 fusion v2 22k를 행동별 공유상자 교집합·512환경×seed0/1/2×1200step·4명/8토큰·near50/loco·Plane Ever로 평가했다. 후행 동시평균±SD/같은 쌍 선행AT 평균%는 SIT58.11±6.13/98.55·CLIMB43.81±2.40/93.16·ON_TOP55.31±2.58/96.46·전체53.60±2.33/96.28다. 분모415/352/954/1721쌍이다. 20k 대비 후행SIT+7.16·CLIMB−7.17·ON_TOP−2.88·전체−1.36%p로 전체 개선은 확인되지 않았다.
- 같은22k의 별도64환경·seed0·기존 상자 분포·native 장면 낙상 종료 정지 진단도 GPU2에서 완료했다. 선행 미완료·페널티 활성화 후 root XYZ≤.3m/s를1초 유지한 첫 정지의 사람–목표XY 중앙값은 SIT1.52·CLIMB2.10·ON_TOP1.91m, 정지/관찰가능8/8·7/7·13/14다. 초기 사람–목표 거리 감소율 중앙값14.85/.03/3.26%를 함께 기록했다. ON_TOP 진입은 사람 기준이며 정지 시 payload–목표 중앙값1.46m다. 정식 개인 낙상 동결 성공률과 구분한다.
- 3회1200step·finite·CPU/GPU/JSON 재채점 일치·20k와 세 seed 초기 장면 정확히 동일·기하/alias/비공유payload·동결encoder73/RMS·원본checkpoint·운영281파일·실행 전부터 있던 본학습3개 보존 검증을 통과했다. 정지 진단 runtime 진입/AT Ever 해제·동일checkpoint·모든rollout GPU2도 확인했다. 운영코드/설정 변경은 없다.
- [성공률](output_etc/check/fusionv2_22000_family512x3_20261106/comparison.csv), [20k 비교](output_etc/check/fusionv2_22000_family512x3_20261106/delta_vs_20k.csv), [정지64](output_etc/check/fusionv2_22000_family512x3_20261106/stop/metrics.json), [검증](output_etc/check/fusionv2_22000_family512x3_20261106/combined_verification.json).

### fuv3_ep900 1k 행동별 교집합512환경×3회 평가

- GPU2만 사용해 900스텝 새 학습의1k를 행동별 공유상자 교집합·512환경×seed0/1/2×900step·4명/8토큰·near50/loco·Plane Ever로 평가했다. 후행 동시평균±SD/같은 쌍 선행AT 평균%는 SIT8.98±1.51/93.06·CLIMB7.61±2.72/76.29·ON_TOP17.82±.09/77.06·전체13.60±.03/80.76다. 분모415/352/954/1721쌍이며 이 버전의 첫1k 평가다.
- 같은 저장본의 별도64환경·seed0·기존 상자 분포·native 장면 낙상 종료 정지 진단도 GPU2에서900step 완료했다. 선행 미완료·페널티 활성화 후 root XYZ≤.3m/s를1초 유지한 첫 정지의 사람–목표XY 중앙값은 SIT1.27·CLIMB1.02·ON_TOP.97m, 정지/관찰가능8/11·9/13·14/19다. 초기 사람–목표 거리 감소율 중앙값47.74/75.15/49.94%를 함께 기록했다. ON_TOP 진입은 사람 기준이며 정지 시 payload–목표 중앙값1.14m다. 정식 개인 낙상 동결 성공률과 구분한다.
- 3회900step·finite·CPU/GPU/JSON 재채점 일치·기존512평가와 초기 장면 정확히 동일·기하/alias/비공유payload·동결encoder73/RMS·원본checkpoint·운영281파일·기존학습4개 보존 검증을 통과했다. 정지 진단 runtime 진입/AT Ever 해제·동일checkpoint·모든rollout GPU2를 확인했다. 운영코드/설정 변경은 없다.
- [성공률](output_etc/check/fusionv3_ep900_1000_family512x3_20261106/comparison.csv), [정지64](output_etc/check/fusionv3_ep900_1000_family512x3_20261106/stop/metrics.json), [검증](output_etc/check/fusionv3_ep900_1000_family512x3_20261106/combined_verification.json).

### fuv1 23k 행동별 교집합512환경×3회 평가

- 사용자 지정 GPU2만 사용해 fuv1 23k를 행동별 공유상자 교집합·512환경×seed0/1/2×1200step·4명/8토큰·near50/loco·Plane Ever로 평가했다. 후행 동시평균±SD/같은 쌍 선행AT 평균%는 SIT48.18±7.20/96.37·CLIMB51.17±1.02/93.96·ON_TOP61.65±2.81/96.03·전체56.19±.46/95.69다. 분모415/352/954/1721쌍이며, 전체는22k 대비−3.54%p·21k 대비−10.16%p다. 같은 조건 최고는21k를 유지한다.
- 같은23k의 별도64환경·seed0·기존 상자 분포·native 장면 낙상 종료 정지 진단도 GPU2에서 완료했다. 선행 미완료·페널티 활성화 후 root XYZ≤.3m/s를1초 유지한 첫 정지의 사람–목표XY 중앙값 SIT1.83·CLIMB2.16·ON_TOP2.93m, 정지/관찰가능9/9·10/10·23/23이다. ON_TOP 진입은 기존payload 기준이며 정지 시 payload–목표 중앙값1.56m다. 초기 사람–목표 거리 감소율 중앙값은12.38/7.80/1.35%다. 정식 개인 낙상 동결 성공률과 구분한다.
- 3회1200step·finite·CPU/GPU/JSON 재채점 일치·21k/22k와 세 seed 초기 장면 정확히 동일·기하/alias/비공유payload·동결encoder73/RMS·원본checkpoint·운영281파일·기존학습4개 보존 검증을 통과했다. 정지 진단 runtime 진입/AT Ever 해제·동일checkpoint·모든rollout GPU2도 확인했다.
- [성공률](output_etc/check/fusion_23000_family512x3_20261106/comparison.csv), [21k/22k 비교](output_etc/check/fusion_23000_family512x3_20261106/delta_vs_21k_22k.csv), [정지64](output_etc/check/fusion_23000_family512x3_20261106/stop/metrics.json), [검증](output_etc/check/fusion_23000_family512x3_20261106/combined_verification.json).

### fuv1 22k 행동별 교집합512환경×3회 평가

- GPU2/6/7만 사용해 fuv1 22k를 행동별 공유상자 교집합·512환경×seed0/1/2×1200step·4명/8토큰·near50/loco·Plane Ever로 평가했다. 21k와 세 seed 초기 장면이 정확히 동일하다. 후행 동시평균±SD/같은 쌍 선행AT 평균%는 SIT50.73±6.06/97.27·CLIMB54.29±.96/94.28·ON_TOP65.89±3.78/96.87·전체59.73±.89/96.45다. 분모는415/352/954/1721쌍이다. 21k 전체66.35±1.10% 대비−6.62%p이며 세 seed 모두 하락했다. 동일 조건 최고는21k를 유지한다.
- 같은22k로 GPU7에서 별도64환경·seed0·기존 상자 분포·native 장면 낙상 종료 정지 진단도 완료했다. 선행 미완료·페널티 활성화 후 root XYZ≤.3m/s를1초 유지한 첫 정지의 사람–목표XY 중앙값은 SIT1.87·CLIMB2.15·ON_TOP2.99m, 정지/관찰가능9/10·7/7·23/23이다. ON_TOP 진입은 기존payload 기준이고 정지 시 payload–목표 중앙값1.56m다. 초기 사람–목표 거리 감소율 중앙값은11.70/2.21/.06%다. 개인 낙상 동결 방식의 정식 성공률과 구분한다.
- 3회1200step·finite·CPU/GPU/JSON 재채점 일치·기하/alias/비공유payload·동결encoder73/RMS·원본checkpoint·운영281파일·기존학습4개 보존 검증을 통과했다. 정지 진단의 runtime 진입/AT Ever 해제 및 동일 checkpoint도 확인했다.
- [성공률](output_etc/check/fusion_22000_family512x3_20261106/comparison.csv), [21k 비교](output_etc/check/fusion_22000_family512x3_20261106/delta_vs_21k.csv), [정지64](output_etc/check/fusion_22000_family512x3_20261106/stop/metrics.json), [검증](output_etc/check/fusion_22000_family512x3_20261106/combined_verification.json).

### fuv3_ep900: Stage1 26k 전이로900스텝 새 학습

- 사용자 요청으로 v3를 복사한 `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v3_ep900` 환경/학습 YAML·train/test/VNC/eval wrapper를 추가했다. 기존v3 대비episodeLength1200→900과실행이름만변경했고, 페널티+1.5m·가중치1.0·ON_TOP사람기준·거리/near50·모델·동결·PPO rollout32·AMP를유지한다. Stage1 26k에서Stage2새학습이며기존v3 0.5k리슘이아니다.
- registry·학습검증·viewer제한시간·정식평가 horizon/dispatch를새실험900으로연결했다. 기존실험horizon은그대로다. config/structure의30개설정·명령을갱신했다.
- CPU58개·실제종료함수899/900경계·네wrapper문법·기존v1/v2/v3/v4 YAML/스크립트24개해시보존통과. GPU1 확인학습의AMP데모초기화지연을진단해확인프로세스만중단하고, 당시비어있던GPU5에서동일4명/1024환경/8토큰 CHECK학습정상종료·epoch2저장·fresh flag/전이26k/900/rollout32·finite·동결encoder73/RMS를확인했다. 이후사용자GPU2/6/7제한에따라GPU1확인probe만중단하고GPU2에서저장본재로드·실제899계속/900리셋·finite추론을확인했다. 종료경계probe에서만낙상종료를잠시비활성화했다. 기존학습3개유지·본학습미실행이다.
- [환경](tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v3_ep900.yaml), [명령](markdowns/config.md), [검증](output_etc/check/before_fusion_v3_ep900_config_20261106/verification.json), [학습](output_etc/check/before_fusion_v3_ep900_config_20261106/train_gpu5.log), [GPU2재로드/종료경계](output_etc/check/before_fusion_v3_ep900_config_20261106/probe_gpu2.log).

### fuv1 21k·fuv2 20k 첫 성공 스텝과900스텝 절단 집계

- 기존행동별교집합512환경×3seed trace6회를CPU float64 Plane기하로재채점해 개인Ever·후행동시Ever·같은쌍선행AT의첫성공스텝과300/600/900/1200절단값을계산했다. 추가rollout/모델/환경변경없으며1200스텝원본Ever·동시counts를정확히재현했다. 실패는1201 sentinel로저장하고성공시점분포에서제외했다.
- 전체후행동시평균±SD/같은쌍AT%는fuv1 1200의66.35±1.10/95.75→900의65.83±.81/95.40, fuv2 54.97±2.65/94.42→54.09±2.83/94.30이다. 901–1200에서처음동시성공한쌍은9/1142·15/946으로900이최종성공의99.21%·98.41%를보존한다. 각모델분모1721쌍이며seed평균과counts합산비율을구분했다. 단독은후행/선행을제외한2702건중900이후추가성공3·2건이다.
- 최종동시성공쌍의첫성공중앙값/p90스텝은fuv1후행310/469.3·같은쌍AT169/242, fuv2후행308/522.5·AT170.5/243이다. 현재정책의평가prefix결과이며900스텝학습의효과를평가한것은아니다.
- [절단성공률](output_etc/check/fusion21k_v220k_success_time_20261106/cutoffs.csv), [성공스텝분포](output_etc/check/fusion21k_v220k_success_time_20261106/timing.csv), [개별쌍](output_etc/check/fusion21k_v220k_success_time_20261106/pairs.csv), [단독](output_etc/check/fusion21k_v220k_success_time_20261106/solo_cutoffs.csv), [검증](output_etc/check/fusion21k_v220k_success_time_20261106/verification.json).

### fuv1 21k 초기 후행 거리별 재집계

- 행동별 교집합512환경×3seed 저장결과를 재사용해 후행 사람root→같은 쌍 선행AT 목표XY의1–2/2–3/3–5/5–7/7–10m 구간별 개인/동시/선행AT Ever를 집계했다. ON_TOP도사람 기준이며3seed합산비율과seed평균/SD를분리보존했다. 추가rollout은없다.
- 전체1721쌍의구간합산counts가검증된원본총계와일치했다. CLIMB1–2m 후행동시/AT38.57/92.86%(N70), SIT7–10m45.07/92.96%(N71), ON_TOP7–10m78.72/100%(N47)로행동별거리경향이다르다. 거리와초기페널티의인과관계로해석하지않는다.
- [거리별결과](output_etc/check/fusion21k_initial_distance_20261106/distance_success.csv), [개별쌍](output_etc/check/fusion21k_initial_distance_20261106/pairs.csv), [검증](output_etc/check/fusion21k_initial_distance_20261106/verification.json).

### fuv1 21k·fuv2 20k 행동별 교집합512환경×3회 평가

- fuv1 21k·fuv2 20k를 행동별 공유상자 교집합·512환경×seed0/1/2×1200step·기존near50/loco/Plane Ever·4명/8토큰으로 평가했다. 사용자 GPU 변경 지시에 따라 진행 중fuv1은GPU5/6에서 유지하고, 아직 시작하지 않은fuv2 대기 coordinator만 GPU1로 교체해 세seed를 순차 완료했다. 기존학습은 종료하지 않았다.
- 후행 동시평균±SD/같은 쌍 선행AT 평균%는fuv1 SIT62.61±5.01/95.38·CLIMB58.81±2.50/94.61·ON_TOP70.77±.73/96.34·전체66.35±1.10/95.75, fuv2 SIT50.95±5.60/95.86·CLIMB50.99±4.97/91.42·ON_TOP58.19±.98/94.87·전체54.97±2.65/94.42다. 조건별분모SIT415·CLIMB352·ON_TOP954·전체1721쌍이다. fuv1은18k 대비전체+7.63%p이고 모든행동/seed에서 상승해 기존동일조건 평가 중 최고다. fuv2는16k55.08%와비슷하고18.5k50.40%에서 회복했다.
- 선행AT Ever 해제 전 armed·root XYZ≤.3m/s·30프레임 첫 정지의 사람–목표XY 중앙값SIT/CLIMB/ON_TOP은fuv1 2.16/2.12/3.14m·fuv2 2.06/1.99/2.13m다. 초기거리 감소율 중앙값은+1.67/+1.36/+0.07%·+3.78/+3.97/+4.65%다. 정지/관찰가능은167/184·173/180·669/679 및153/169·152/162·378/429다. 개인낙상동결 평가 내 관측이며native장면 정지검사와 구분한다.
- 6회1200step·finite·공식float64 CPU/GPU/JSON재채점·float32/64성공진단차이0·같은seed 모델간 및 기존512초기장면 정확히 동일·기하/alias/비공유payload·동결encoder73/RMS·원본checkpoint·운영275파일·기존2학습 보존 검증 통과. fuv1은직전64환경평가와checkpoint SHA도동일하다.
- [fuv1 성능](output_etc/check/fusion_21000_family512x3_20261106/comparison.csv), [fuv2 성능](output_etc/check/fusionv2_20000_family512x3_20261106/comparison.csv), [통합추세](output_etc/check/fusionv2_20000_family512x3_20261106/combined_history.csv), [두모델검증](output_etc/check/fusionv2_20000_family512x3_20261106/two_model_verification.json). 정지분포는각폴더 waiting_aggregate.csv에 있다.

### fusion v1 21k 행동별 교집합64환경 평가

- 사용자 요청으로 fuv1 21k 완전 저장본을 GPU5에서 행동별 공유상자 교집합·64환경×seed0×1200step·4명/8토큰·기존near50/loco/Plane Ever로1회 평가했다. 같은 초기65쌍(SIT12·CLIMB12·ON_TOP41)의 후행 동시/같은 쌍 선행AT Ever%는58.33/100·58.33/100·78.05/100·전체70.77/100이다. 개인Ever도동일하다. 기존18k66.15/100·20k53.85/96.92 대비 회복했으나 단일seed 탐색 결과다.
- 선행AT Ever 해제 전 armed·사람XYZ속력≤0.3m/s·30프레임 첫 정지의 사람–목표XY 중앙값은SIT1.69·CLIMB1.68·ON_TOP3.65m, 정지/관찰가능은5/5·7/7·26/27이다. 초기거리 감소율 중앙값−5.24/+0.10/−0.45%를 함께 보존했다. 개인 낙상 동결 평가 내 진단이며native장면 정지검사와 구분한다.
- 1200step·finite·공식float64 CPU/GPU/JSON 재채점 일치·float32/64성공 차이0·18k/20k와초기장면 정확히 동일·공유물리/alias크기/5cm격자/비공유payload보존·동결encoder73/RMS·원본checkpoint·운영275파일·기존2학습 보존을 확인했다.
- [성능](output_etc/check/fusion_21000_family64_20261106/comparison.csv), [10–21k추세](output_etc/check/fusion_21000_family64_20261106/history.csv), [정지](output_etc/check/fusion_21000_family64_20261106/waiting.csv), [검증](output_etc/check/fusion_21000_family64_20261106/combined_verification.json).

### fuv4: fuv3 기반 대기 페널티 버퍼2m 설정

- fuv3를 복사해 `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v4` 환경/학습 YAML·train/test/VNC/eval wrapper를 추가하고 registry·launch·평가 dispatch·config/structure에 연결했다. 실행 이름 외에는 waiting_penalty.radius_buffer만1.5→2.0m다. 가중치1.0·ON_TOP 사람 기준·진입/선행성공 Ever·기존 Gate·fusion·거리/near50를 유지한다. 기본은 동일Stage1 26k 전이로 새 학습이다.
- 관련CPU24개 및 세 행동의2m 경계·XYZ 속도·진입 latch·AT Ever 해제를 확인했다. GPU6 CHECK 학습4명/1024환경/8토큰 정상 종료·epoch2 저장·finite·Stage1 encoder73텐서 보존·새test wrapper 재로드4환경/6step×2회 통과. 기존fusion v1/v2/v3 YAML·wrapper18개 해시 및 기존2학습을 보존했다. 장기 본학습은 시작하지 않았다.
- [환경](tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v4.yaml), [실행 명령](markdowns/config.md), [검증](output_etc/check/before_fusion_v4_config_20261106/verification.json), [학습](output_etc/check/before_fusion_v4_config_20261106/train.log).

### fuv3: fuv2 기반 대기 페널티 버퍼1.5m 설정

- `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v3` 환경/학습 YAML과 train/test/VNC/eval wrapper를 추가하고 registry·공통 launch·평가 dispatch·config/structure의28개 실험 목록을 연결했다. fuv2 대비 실행 이름과 waiting_penalty.radius_buffer만2.5→1.5m다. coefficient1.0·ON_TOP 포함 사람 기준 XY 진입·XYZ 속력·진입/선행성공 Ever·Progress/current State·Success Gate·거리/near50·fusion/zero bias·PPO/AMP를 유지한다. 기본은 동일Stage1 26k에서 새 학습이며 기존fuv2 checkpoint와 계약을 분리했다. 원본fuv2 YAML2개/스크립트4개 해시를 보존했다.
- 관련CPU24개 및 확장검사50개 통과. 확장 공통검사1개는 구형 `checkpoints/Stage1PushFinetune_00021000.pth` 파일 부재로 실행 불가였고, 실제 사용하는26k 전이는 GPU6 `CHECK=1`·4명/1024환경/8토큰 짧은 학습으로 확인했다. 저장epoch2·1.5/1.0/human 설정·finite·동결encoder73텐서/RMS·checkpoint 저장·새test wrapper 재로드/추론 통과. 네 wrapper shell문법·기존2학습 생존을 확인했다. 장기 본학습은 시작하지 않았다.
- [환경](tokenhsi/data/cfg/multi_agent/stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v3.yaml), [실행 명령](markdowns/config.md), [검증](output_etc/check/before_fusion_v3_config_20261106/verification.json), [학습](output_etc/check/before_fusion_v3_config_20261106/train.log), [관련CPU](output_etc/check/before_fusion_v3_config_20261106/cpu_focused.log).

### fusion v1 18k·fusion v2 16k 초기 후행 사람–목표 거리별 집계

- 기존 두 교집합512환경×seed0/1/2 결과12회의 초기 후행 사람root→같은 쌍 선행AT 목표XY 거리를1–2/2–3/3–5/5–7/7–10m로 나눠 SIT·CLIMB·ON_TOP의 개인/동시/같은 쌍 선행AT Ever를 집계했다. ON_TOP도payload가 아닌 사람 거리다. 추가 rollout/모델·물리·보상 변경 없이 저장float64 Plane Ever를 재사용했다. 조건별1721쌍 전부 구간에 포함되고 모델 간 같은 조건/seed 거리·역할·구간이 정확히 같으며, 구간 합산 성공/trials가 기존 검증 전체counts와 일치했다.
- 표/그림은3seed 성공건수/해당 구간 쌍 수의 합산비율이며 기존seed별 평균±표준편차와 구분한다. CSV에seed별 값·평균/표준편차·N·개인/동시/AT를 모두 보존했다. 행동별교집합 CLIMB1–2m(N70)의 동시/AT%는v1 31.4/90.0·v2 30.0/88.6, 5–7m(N60)는63.3/96.7·45.0/93.3으로 단순히 가까울수록 성공률이 높지는 않았다. 공통에서도1–2m(N69)42.0/92.8·31.9/89.9 대5–7m(N56)73.2/100·55.4/96.4였다. 거리만 통제한 인과 실험으로 해석하지 않는다.
- [모든 거리/행동/교집합 결과](output_etc/check/fusion18k_v216k_initial_distance_20261106/distance_success.csv), [행동별 그림](output_etc/check/fusion18k_v216k_initial_distance_20261106/family.png), [공통 그림](output_etc/check/fusion18k_v216k_initial_distance_20261106/common.png), [개별쌍](output_etc/check/fusion18k_v216k_initial_distance_20261106/pairs.csv), [검증](output_etc/check/fusion18k_v216k_initial_distance_20261106/verification.json).

### fusion v2 19k 행동별 교집합 64환경 평가

- 사용자 지정 fusion_v2 19k 완전 저장본을 행동별 공유상자 교집합·64환경×seed0×1200step에서1회 평가했다. GPU6만 사용했고 기존4명/8토큰·random_scenario/near50·loco·Plane Ever·개인 낙상 동결을 유지했다. 원본checkpoint와 기존2학습을 보존했다.
- 동일65쌍(SIT12·CLIMB12·ON_TOP41)의 후행 개인/동시/같은 쌍 선행AT Ever%는 SIT58.33/50.00/100·CLIMB25.00/25.00/91.67·ON_TOP63.41/60.98/95.12·전체55.38/52.31/95.38다. 직전18k 전체56.92/56.92/93.85 대비 개인−1.54·동시−4.62%p다. 16k 전체67.69/67.69/96.92보다 낮았으며 같은64환경 초기 장면의10–19k 지표를 재실행 없이 정리했다.
- 평가 내 선행 AT Ever 해제 전 armed·root XYZ≤.3m/s·30프레임 첫 정지의 사람→목표XY 중앙값은 SIT1.63·CLIMB1.20·ON_TOP1.68m, 정지/관찰가능은5/5·6/6·13/14쌍이다. 초기거리 감소율 중앙값−1.15/+17.93/−.17%와IQR을 저장했다. 개인 낙상 동결 내 관측이며 native 장면 종료 진단과 구분한다.
- 1200step·finite·공식float64 CPU/GPU/JSON 재채점 및 개인 counts 일치·float32/64 성공 진단 차이0·16k/18k와 초기 장면 동일·공유 물리/alias 기하·범위/격자·비공유payload·encoder73텐서/RMS 동결·원본checkpoint·이번 운영파일263개 보존·기존2학습 생존 검증 통과. 이전 탐색 이후 변경된4개viewer 옵션 파일을 별도 기록했고, 기본original 경로의 초기 상태 동일성을 확인했다.
- [현재 성능](output_etc/check/fusionv2_19000_family64_20261106/comparison.csv), [10–19k 이력](output_etc/check/fusionv2_19000_family64_20261106/history.csv), [정지 기록](output_etc/check/fusionv2_19000_family64_20261106/waiting.csv), [검증](output_etc/check/fusionv2_19000_family64_20261106/combined_verification.json).

### BEFORE PUSH viewer 크기 모드와 후행 개인 Ever 조회

- 기존 test/VNC에 `BEFORE_BOX_SIZE=original|family|common`/직접 `--before_box_size`를 연결했다. 기본original은 기존 역할/크기 샘플링을 유지한다. family/common은 기존 교집합 평가와 같은5cm 격자로 선행 공유상자만 선택하고, 사용자 결정대로 시작 시 뽑은 쌍·행동을 고정해 위치·목표만 재샘플링한다. 단독·payload·PUSH 크기 규칙, 학습/YAML/모델/reward/gate/checkpoint 계약은 유지하며 학습·정식 평가·다른 task·오타를 거부한다. 공통 launch와 config, BEFORE PUSH asset/reset 훅, `utils/before_box_size.py`, 범위/역할/예산 CPU 테스트, config/USAGE/structure를 갱신했다.
- CPU35개·shell/Python 문법·4개 CLI 거부 검사 통과. GPU5/6에서 v1 기본/family/common(4명·16환경·8토큰), v2 common(8명·4환경·12토큰)을 각각3회 reset/Actor·Critic48step으로 확인했다. 교집합 역할/물리 크기 고정·목표 재샘플링·Alias 기하·부분 reset 격리·finite·model weight 보존 통과. 기존test wrapper도common/before_stack/4명·2환경·8토큰·6step×2회에서 정상 종료했다. VNC 세션은 새로 열지 않았다. 원본2 checkpoint와 기존2학습을 보존했다.
- 직전 v1 18k/v2 16k 교집합512×3의 저장 trace/JSON에서 BEFORE 후행 개인 Plane Ever를 추가 집계했다. 선행 성공/gate를 요구하지 않는 실제 상자 기준 성공이며 선행 완료 전 수행도 포함한다. 전체 개인Ever 평균±표준편차/동시Ever 평균/같은 쌍 선행AT 평균%는 v1 행동별59.95±2.59/58.72/95.82·공통59.35±2.96/58.31/96.33, v2 행동별56.48±.34/55.08/95.17·공통54.85±.31/53.80/95.12다. 개인−동시 차이는1.0–1.4%p이며 동일1721쌍의 행동별/seed별 값과 AT를 저장했다. 저장JSON과 counts 일치를 확인했고 추가 성능 rollout은 실행하지 않았다.
- [검증](output_etc/check/before_box_size_viewer_20261106/verification.json), [후행 개인/동시/선행 AT 행동별 결과](output_etc/check/before_box_size_viewer_20261106/follower_individual.csv), [CPU](output_etc/check/before_box_size_viewer_20261106/cpu.log), [실제test wrapper](output_etc/check/before_box_size_viewer_20261106/wrapper_common.log).

### fusion v1 18k·fusion v2 16k 교집합 512환경 후보 재평가

- 64환경 탐색에서 선택한 fusion_v1 18k·fusion_v2 16k의 동일SHA256 저장본을 행동별/전체공통 공유상자 교집합에서 각512환경×seed0/1/2×1200step 평가했다(12회). GPU5/6만 사용했고4명/8토큰·혼합 random_scenario/near50·loco·Plane Ever·개인 낙상 동결·5cm 격자를 유지했다. 체크포인트·운영 코드/설정·기존2학습을 보존했다.
- 전체 후행 동시 평균±표준편차/같은 쌍 선행 AT 평균%는 v1 행동별58.72±2.75/95.82·공통58.31±2.68/96.33, v2 행동별55.08±.51/95.17·공통53.80±.94/95.12다. 조건/저장본마다 SIT415·CLIMB352·ON_TOP954·전체1721쌍이며 행동별 선행 AT 상한·분모를 보존했다. 64환경 단일seed 후보값 v1 66.15/100·v2 67.69/96.92%는512×3에서 유지되지 않았다.
- v1 15k 대비 전체 후행−.83/+.21%p, v2 10k 대비+1.10/−1.21%p로 두 조건 모두seed 방향이 혼재했다. 기존6/8/10/11/13/15k 및19.5k·18.5k 교집합 이력과 새 결과를 함께 정리했고, 64환경과512환경 성능은 별도 비교파일로 분리했다.
- 평가 내 armed·선행 AT Ever 해제 전 root XYZ≤.3m/s·30프레임 첫 정지 telemetry를 수집했다. SIT/CLIMB/ON_TOP 사람→목표XY 중앙값(m)은 v1 행동별2.14/2.12/3.12·공통2.25/2.12/3.15, v2 행동별2.01/1.90/2.06·공통2.08/1.86/2.15다. 초기거리 감소율·IQR·정지/관찰가능 분모를 저장했으며 개인 낙상 동결 내 관측을 native 장면 종료 진단과 구분했다.
- 12회1200step·finite·공식float64 CPU/GPU/JSON 재채점 일치·float32/64 성공 진단 차이0·같은 조건/seed 두 모델 및 기존512환경 초기 장면 동일·공유 물리/원본/alias 크기·범위/격자·비공유payload 보존·encoder73텐서/RMS 동결·원본checkpoint·운영파일262개·기존2학습 생존 검증 통과. 물리 GPU5/6 진단 기록을 저장했다.
- [행동별/공통 성능](output_etc/check/fusion18000_fusionv216000_intersections512x3_20261106/comparison.csv), [비교 그림](output_etc/check/fusion18000_fusionv216000_intersections512x3_20261106/comparison.png), [전체 교집합 추세](output_etc/check/fusion18000_fusionv216000_intersections512x3_20261106/trends.png), [64환경 대비](output_etc/check/fusion18000_fusionv216000_intersections512x3_20261106/screening_vs_confirmation.csv), [정지 기록](output_etc/check/fusion18000_fusionv216000_intersections512x3_20261106/waiting_aggregate.csv), [검증](output_etc/check/fusion18000_fusionv216000_intersections512x3_20261106/combined_verification.json).

### fusion v1/v2 10k 이후 1k 간격 행동별 교집합 64환경 탐색

- 사용자 최종 범위에 따라 fusion_v1 10–20k(11개)·fusion_v2 10–18k(9개)의 행동별 공유상자 교집합만 각64환경×seed0×1200step 평가했다(20회). 공통 교집합·두 번째 seed·18.5k는 최종 집계에서 제외했다. 범위 조정 전 완료/중단된 진단은 check 아래 보존했다. GPU5/6만 사용하고 기존4명/8토큰·혼합 random_scenario/near50·loco·Plane Ever·개인 낙상 동결·5cm 격자를 유지했다.
- 저장본별 동일65쌍(SIT12·CLIMB12·ON_TOP41)의 후행 동시/같은 쌍 선행 AT Ever를 집계했다. 이번 탐색 최고는 v1 18k43/65=66.15% /65/65=100%, v2 16k44/65=67.69% /63/65=96.92%다. v1 15k64.62/96.92·13k63.08/100%, v2 11k60.00/98.46%가 뒤를 이었다. 최신20k53.85/96.92·18k56.92/93.85%로 단조 상승은 없었다. 동일 초기 장면의 기존 v1 8k64환경75.38/95.38%도 비교 가능하며, 단일 seed 후보 탐색 결과를 기존512환경×3seed와 분리했다.
- 평가 내 armed·선행 AT Ever 해제 전 root XYZ≤.3m/s·30프레임 첫 정지 telemetry를 함께 수집했다. 최고 저장본의 SIT/CLIMB/ON_TOP 사람→목표XY 중앙값(m)은 v1 18k1.69/1.71/3.68, v2 16k1.70/.95/1.55다. 초기거리 대비 감소율·IQR·정지/관찰가능 분모를 보존했으며, 개인 낙상 동결 내 관측을 native 장면 종료 진단과 구분했다.
- 20회1200step·finite·공식float64 CPU/GPU/JSON 재채점 일치·추가float32/64 성공 진단 차이0·저장본 간 및 이전64환경 초기 장면 동일·역할/크기uniform·공유 물리/원본/alias 크기·교집합/격자·비공유payload 보존·encoder73텐서/RMS 동결·원본checkpoint·운영파일262개·기존2학습 생존 검증을 통과했다. 단일 seed이므로 표준편차를 추정하지 않았다.
- [전체 행동별 성능](output_etc/check/fusion_10k_latest_1k_size64_20261106/comparison.csv), [1k 추세](output_etc/check/fusion_10k_latest_1k_size64_20261106/trends.png), [후보 순위](output_etc/check/fusion_10k_latest_1k_size64_20261106/rankings.csv), [정지 기록](output_etc/check/fusion_10k_latest_1k_size64_20261106/waiting_aggregate.csv), [기존512 평가와 별도 비교](output_etc/check/fusion_10k_latest_1k_size64_20261106/existing512_reference.csv), [검증](output_etc/check/fusion_10k_latest_1k_size64_20261106/combined_verification.json).

### fusion_v1 19.5k·fusion_v2 18.5k 행동별/공통 교집합 평가

- 작업 시작 시 최신 완전 저장본 fusion_v1 19.5k·fusion_v2 18.5k를 고정해 행동별/전체공통 공유상자 교집합에서 각512환경×seed0/1/2×1200step 평가했다(총12회). 기존 random_scenario/near50·4명/8토큰·loco·Plane Ever·개인 낙상 동결·5cm 크기 격자를 유지했다. 최초 GPU2/3에서 완료한2회를 보존하고, 사용자 GPU5/6 제한 이후 실행 중이던2회는 check 아래 보관한 뒤 재시작해 남은10회를 GPU5/6에서 완료했다. 기존 GPU5/6 학습과 원본 checkpoint·운영 코드/설정을 보존했다.
- 후행 동시±표준편차/같은 쌍 선행 AT 평균%는 fusion_v1 행동별 SIT56.94±5.85/97.42·CLIMB53.51±5.87/95.12·ON_TOP63.61±.90/96.14·전체59.93±2.75/96.22, 공통58.25±4.33/95.68·53.68±4.89/95.38·61.94±1.13/96.67·59.39±1.17/96.16이다. 15k 대비 전체+.38/+1.28%p로 seed 방향이 혼재해 큰 추가 상승은 없었다. 공통 CLIMB−3.42%p·ON_TOP+2.50%p는각각3seed 공통 방향이다.
- fusion_v2 행동별 SIT47.60±2.30/96.40·CLIMB37.38±4.54/91.46·ON_TOP56.49±2.39/95.49·전체50.40±1.92/94.89, 공통48.07±2.73/95.02·38.65±.48/92.22·51.18±4.03/93.84·47.85±2.38/93.78%다. 10k 대비 전체−3.58/−7.16%p이고 공통 전체·세 행동은3seed 공통 하락했다. 행동별 CLIMB−13.69%p도3seed 공통 하락이며 같은 쌍 선행 AT 평균은91.78→91.46%로 큰 변화가 없었다.
- 평가 내 armed·선행 AT Ever 해제 전 root XYZ≤.3m/s·30프레임 첫 정지 telemetry를 함께 수집했다. SIT/CLIMB/ON_TOP 사람→목표XY 중앙값(m)은 v1 행동별2.14/2.04/3.12·공통2.23/2.04/3.16, v2 행동별2.00/1.96/2.11·공통2.00/1.92/2.09다. 해당 교집합·개인 낙상 동결 조건의 관측이며 native64 장면 종료 진단과 구분한다.
- 같은 조건/seed의 두 모델 초기 장면 및 기존5k 간격 교집합 초기 장면과 정확 일치했다. 조건/저장본마다 SIT415·CLIMB352·ON_TOP954·전체1721쌍의 동일분모 선행 AT를 보존했다. 12회1200step·finite·공식float64 GPU/CPU/JSON 재채점 일치·float32/64 진단 차이0·공유 물리/alias 크기·범위/격자·비공유payload 보존·encoder73텐서/RMS 동결·원본checkpoint·운영파일262개·기존2학습 생존 검증 통과. 조건 간 충돌 재시도에 따른 배치 차이는 분포 비교로 해석한다.
- [최신 성능](output_etc/check/fusion19500_fusionv218500_intersections512x3_20261106/comparison.csv), [이전 저장본 대비](output_etc/check/fusion19500_fusionv218500_intersections512x3_20261106/changes.csv), [행동별 그림](output_etc/check/fusion19500_fusionv218500_intersections512x3_20261106/comparison.png), [추세](output_etc/check/fusion19500_fusionv218500_intersections512x3_20261106/trends.png), [정지 기록](output_etc/check/fusion19500_fusionv218500_intersections512x3_20261106/waiting_aggregate.csv), [GPU 전환 기록](output_etc/check/fusion19500_fusionv218500_intersections512x3_20261106/gpu_migration.json), [검증](output_etc/check/fusion19500_fusionv218500_intersections512x3_20261106/combined_verification.json).

## 2026-11-05

### 모델별 5k 간격 공유상자 교집합 512환경 평가

- 사용자 요청으로 penalty_v2 5/10/15/20k·fusion_v1 5/10/15k·fusion_v2 5/10k의9개 고정 저장본을 행동별/전체공통 교집합에서 각512환경×seed0/1/2×1200step 평가했다(54회). 동일 조건의 기존5k 격자 결과는 없어 새로 실행했다. 기존 penalty15k near0 결과는 재사용하지 않았다. GPU3/4/5/2/6 사용·GPU0/1 미사용, 기존3학습·운영 코드/설정·원본checkpoint 보존.
- 검증된 check 전용 크기 driver를 재사용해 선행 공유 물리상자만 AT∩후행/전체공통 범위의5cm 격자로 교체했다. 기존 random_scenario/near50·4명/8토큰·loco·Plane Ever·개인 낙상 동결을 유지했다. 저장본마다 같은 조건/seed 초기 장면이 정확히 같았고, 이전 fusion8k near50 교집합의6개 초기 장면과도 일치했다. 조건 간 역할·원본 자산·크기uniform은 같지만 충돌 재시도로 배치가 달라질 수 있어 크기 조건은 분포 비교로 해석한다.
- penalty의 행동별 전체 후행/같은 쌍 선행 AT 평균%는5/10/15/20k39.55/86.91→51.06/90.86→60.21/92.78→56.30/93.13, 공통40.34/87.78→50.37/90.58→61.88/93.15→57.74/94.00이다. 15→20k 하락은 두 조건 모두3seed 공통이며 이번 격자 최고는15k다.
- fusion_v1 행동별 전체5/10/15k56.77/94.48→59.11/94.65→59.55/95.18, 공통57.59/95.64→59.26/95.63→58.11/95.63%다. 10k 이후 행동별은 혼재한 작은 상승·공통은3seed 공통 소폭 하락으로 큰 추가 상승은 없었다. fusion_v2 행동별5→10k48.26/93.97→53.98/94.77·공통45.92/94.88→55.01/94.89%이며 두 조건 모두3seed 공통 상승했다. 모든 행동의 같은 쌍 선행 AT·분모·seed별 값·표준편차를 보존했다. 조건/저장본별 분모는 SIT415·CLIMB352·ON_TOP954·전체1721쌍이다.
- 54회 평가 안에서 armed·선행 AT Ever 해제 전 root XYZ≤.3m/s·30프레임 첫 정지의 사람→목표XY 거리·초기거리 감소율·변위/IQR을 수집했다. 개인 낙상 동결·교집합 크기 평가 내 기록이며 종전 native64 장면 종료 진단과 구분한다. 일반 크기 기존5k 격자 결과도 재실행 없이 비교 그림에 포함했다.
- 54회1200step·finite·공식 float64 GPU/CPU/JSON 재채점 일치·공유 물리/원본 및 alias 관측 크기·격자·비공유payload 보존·encoder73텐서/RMS 동결·기존 일반평가와 동일checkpoint SHA256·운영파일262개·기존3학습 생존 검증 통과. 추가 float32 비교에서 fusion_v2 5k 행동별seed0 ON_TOP1쌍의 Plane Ever/동시 판정 차이가 있었다. 기울어진 받침의 yaw 정밀도에 따른 내부XY 경계 오차 .347mm이며 공식float64 결과를 유지했다(전체3seed평균 대체채점 차이 .059%p). 기존 요약기의 float32/64 차이0 요구를 공식채점 일치 검증과 별도 정밀도 진단 기록으로 분리했고 운영 채점코드는 바꾸지 않았다.
- [행동별/전체 결과](output_etc/check/before_5k_intersections512x3_20261105/comparison.csv), [행동별 추세 그림](output_etc/check/before_5k_intersections512x3_20261105/comparison.png), [일반/두 교집합 전체 비교](output_etc/check/before_5k_intersections512x3_20261105/overall.png), [정지 기록](output_etc/check/before_5k_intersections512x3_20261105/waiting_aggregate.csv), [정밀도1건](output_etc/check/before_5k_intersections512x3_20261105/precision_check.json), [검증](output_etc/check/before_5k_intersections512x3_20261105/combined_verification.json).

### penalty_v2 20k·fusion_v1 15k·fusion_v2 13k 평가 및 정지 추세

- 사용자 지정20k/15k/13k 완전 저장본을 고정해 기존 일반 크기·random_scenario/near50·4명/8토큰·loco·Plane Ever·개인 낙상 동결에서 각512환경×seed0/1/2×1200step 평가했다. 일반 GPU3/4/5·native64환경×seed0 정지 진단GPU6/2를 사용하고 GPU0·1·기존3학습·운영 코드/설정·체크포인트를 보존했다.
- 후행 동시±표준편차/같은 쌍 선행 AT 평균%는 penalty20k SIT57.88±5.73/97.75·CLIMB45.77±1.70/93.16·ON_TOP34.56±5.56/93.87·전체42.01±1.91/94.49, fusion15k58.10±6.25/97.15·50.50±4.77/93.34·36.07±3.98/95.92·43.98±3.01/95.57, fusion_v2 13k54.79±5.60/97.20·51.69±3.00/93.55·27.34±1.57/96.01·38.60±1.05/95.70이다. 단독5행동 평균95.98±.90/95.46±1.01/96.56±1.04%다.
- 직전19k/14k/12k 대비 전체−5.14/+3.33/+1.41%p다. penalty는전체·CLIMB−5.39·ON_TOP−6.53%p가3seed 공통 하락했고18k 이후2회 연속 하락했다. fusion_v1은 전체 seed 방향이 혼재했지만 ON_TOP+5.23%p는3seed 공통 회복, fusion_v2는전체·CLIMB+5.35%p가3seed 공통 상승했다. 컨피그별 최고 penalty18k50.13/95.07·fusion13k45.17/96.39·fusion_v2 9k40.85/96.82%를 모두 넘지 못해 조건부 교집합 추가 rollout0회다.
- armed·선행 AT Ever 전 root XYZ≤.3m/s·30프레임 첫 정지의 사람→목표XY 중앙값 SIT/CLIMB/ON_TOP(m)은 penalty1.21/2.32/2.80·fusion1.90/2.12/2.81·fusion_v2 1.56/2.15/2.01, 쌍별 초기거리 감소율 중앙값17.6/12.4/1.1·14.8/3.4/1.3·16.4/.2/5.3%다. 정지/관찰가능은8/9·10/10·22/26,9/9·8/8·24/24,8/9·7/7·16/17쌍이다. 동일 정지쌍 비교와1k 거리/감소율 추세를 보존하고 조건부 표본 변화·root 첫 정지와 전신/지속 대기를 구분했다.
- 일반9회·native3회1200step 정상 종료, CPU/GPU/JSON 재채점 일치·float32/64 성공 차이0·기존 동일seed 초기 상태 정확 일치·native 진입/AT Ever 해제·encoder/RMS 동결·checkpoint/운영파일262개·기존3학습 생존·최고치 조건부 판정 검증 통과. [최신3개](output_etc/check/penalty20k_fusion15k_fusionv2_13k_eval_stop_20261105/latest_configs.csv), [직전 비교](output_etc/check/penalty20k_fusion15k_fusionv2_13k_eval_stop_20261105/before_comparison.csv), [최고 기록](output_etc/check/penalty20k_fusion15k_fusionv2_13k_eval_stop_20261105/best_by_config.csv), [성공률 추세](output_etc/check/penalty20k_fusion15k_fusionv2_13k_eval_stop_20261105/latest_trends.png), [정지 추세](output_etc/check/penalty20k_fusion15k_fusionv2_13k_eval_stop_20261105/stop_trends.png), [동일쌍 정지 비교](output_etc/check/penalty20k_fusion15k_fusionv2_13k_eval_stop_20261105/stop_common_pairs.csv), [검증](output_etc/check/penalty20k_fusion15k_fusionv2_13k_eval_stop_20261105/combined_verification.json).

### penalty_v2 19k·fusion_v1 14k·fusion_v2 12k 평가 및 정지 추세

- 사용자 지정19k/14k/12k 완전 저장본을 고정해 기존 일반 크기·random_scenario/near50·4명/8토큰·loco·Plane Ever·개인 낙상 동결에서 각512환경×seed0/1/2×1200step 평가했다. GPU3/4/5 일반 평가·GPU6/2 native64환경×seed0 정지 진단을 병행했고 GPU0·1·기존3학습·운영 코드/설정·체크포인트를 보존했다.
- 후행 동시±표준편차/같은 쌍 선행 AT 평균%는 penalty19k SIT59.12±1.79/98.83·CLIMB51.16±1.04/94.38·ON_TOP41.09±3.24/94.22·전체47.15±1.89/95.20, fusion14k55.59±3.75/97.43·51.04±4.71/92.57·30.84±.30/95.18·40.65±1.40/95.05, fusion_v2 12k50.21±4.16/97.37·46.34±7.79/94.41·28.61±1.81/94.43·37.19±1.12/95.04이다. 단독5행동 평균96.99±1.15/96.04±.40/95.52±.59%다.
- 직전18k/13k/11k 대비 전체−2.98/−4.52/−.77%p이며 앞 두 모델은3seed 공통 하락, fusion_v2는 혼재했다. penalty CLIMB−5.85%p·fusion SIT−4.55/ON_TOP−5.86·fusion_v2 CLIMB−3.32는3seed 공통 하락이다. 단일 저장본 변동과 장기 추세를 구분했다. 컨피그별 최고 penalty18k50.13/95.07·fusion13k45.17/96.39·fusion_v2 9k40.85/96.82%를 모두 넘지 못해 조건부 교집합 추가 rollout0회다.
- armed·선행 AT Ever 전 root XYZ≤.3m/s·30프레임 첫 정지의 사람→목표XY 중앙값 SIT/CLIMB/ON_TOP(m)은 penalty1.17/2.20/2.64·fusion1.87/2.08/2.85·fusion_v2 1.48/2.05/2.03, 쌍별 초기거리 감소율 중앙값23.2/9.0/1.0·12.5/.5/1.9·16.7/.5/8.3%다. 정지/관찰가능은9/10·9/9·21/25,9/9·7/7·23/23,8/10·7/8·18/18쌍이다. fusion_v2 ON_TOP의 동일 정지15쌍 감소율은2.61→2.74%로 전체 조건부 중앙값 변화와 구분한다. penalty CLIMB 동일8쌍은15.37→5.99%다. root 첫 정지와 전신/지속 대기를 구분하고1k 정지거리/감소율 추세 그림·CSV를 저장했다.
- 일반9회·native3회1200step 정상 종료, CPU/GPU/JSON 재채점 일치·float32/64 성공 차이0·기존 동일seed 초기 상태 정확 일치·native 진입/AT Ever 해제·encoder/RMS 동결·checkpoint/운영파일262개·기존3학습 생존·최고치 조건부 판정 검증 통과. [최신3개](output_etc/check/penalty19k_fusion14k_fusionv2_12k_eval_stop_20261105/latest_configs.csv), [직전 비교](output_etc/check/penalty19k_fusion14k_fusionv2_12k_eval_stop_20261105/before_comparison.csv), [최고 기록](output_etc/check/penalty19k_fusion14k_fusionv2_12k_eval_stop_20261105/best_by_config.csv), [성공률 추세](output_etc/check/penalty19k_fusion14k_fusionv2_12k_eval_stop_20261105/latest_trends.png), [정지 추세](output_etc/check/penalty19k_fusion14k_fusionv2_12k_eval_stop_20261105/stop_trends.png), [정지 동일쌍](output_etc/check/penalty19k_fusion14k_fusionv2_12k_eval_stop_20261105/stop_common_pairs.csv), [검증](output_etc/check/penalty19k_fusion14k_fusionv2_12k_eval_stop_20261105/combined_verification.json).

### fusion_v2 11k 지정 재평가 및 정지 진단

- 사용자 지정11k 완전 저장본을 고정해 GPU3에서 기존 일반 크기·random_scenario/near50·4명/8토큰·loco·Plane Ever·개인 낙상 동결512환경×seed0/1/2×1200step 평가, GPU4에서64환경×seed0 native 정지 진단을 수행했다. GPU0·1 미사용, 운영 코드/설정·체크포인트·기존3학습 보존.
- 후행 동시±표준편차/같은 쌍 선행 AT 평균%는 SIT47.74±5.65/97.96·CLIMB49.66±6.63/96.92·ON_TOP29.46±.85/95.20·전체37.96±.87/96.17, 단독5행동 평균96.50±1.75%다. 10.5k 대비 전체+2.45%p·CLIMB+2.27·ON_TOP+4.77·SIT−3.65%p로 각각3seed 공통 방향이다. 전체9k40.85/96.82→10k38.26/95.50→10.5k35.51/95.80→11k37.96/96.17%로 최근 하락에서 일부 회복했고9k 최고에는 미달한다.
- 조건부 교집합 최고치 판정은 불충족으로 추가 rollout0회다. 동일 일반 평가 기준 컨피그별 최고는 penalty18k50.13/95.07·fusion_v1 13k45.17/96.39·fusion_v2 9k40.85/96.82%로 유지했다. 기존10.5k 평가를 보존하고 전체 이력·1k 추세·최신3개 집계를 갱신했다.
- armed·선행 AT Ever 전 root XYZ≤.3m/s·30프레임 첫 정지의 사람→목표XY 중앙값 SIT/CLIMB/ON_TOP은1.57/2.10/1.85m, 쌍별 초기거리 감소율 중앙값17.0/.2/2.6%, 정지/관찰가능8/8·7/7·15/15쌍이다. root 첫 정지와 전신/지속 대기를 구분한다.
- 일반3회·native1회1200step 정상 종료, CPU/GPU/JSON 재채점 일치·float32/64 성공 차이0·기존 동일seed 초기 상태 정확 일치·native 진입/AT Ever 해제·encoder/RMS 동결·checkpoint/운영파일262개·기존3학습 생존·최고치 조건부 판정 검증 통과. [직전 비교](output_etc/check/fusionv2_11000_eval_stop_20261105/before_comparison.csv), [최신3개](output_etc/check/fusionv2_11000_eval_stop_20261105/latest_configs.csv), [컨피그별 최고](output_etc/check/fusionv2_11000_eval_stop_20261105/best_by_config.csv), [1k 추세](output_etc/check/fusionv2_11000_eval_stop_20261105/latest_trends.png), [정지 비교](output_etc/check/fusionv2_11000_eval_stop_20261105/stop_comparison.csv), [검증](output_etc/check/fusionv2_11000_eval_stop_20261105/combined_verification.json).

### penalty_v2 18k·fusion_v1 13k·fusion_v2 10.5k 최신 평가 및 컨피그별 최고 기록

- 작업 시작 시 완전 저장된18k/13k/10.5k를 고정해 기존 일반 크기·random_scenario/near50·4명/8토큰·loco·Plane Ever·개인 낙상 동결에서 각512환경×seed0/1/2×1200step 평가했다. 각64환경×seed0 native 정지 진단도 병행했다. GPU2–6만 사용했고 운영 코드/설정·체크포인트·기존3학습을 보존했다.
- 후행 동시/같은 쌍 선행 AT 평균%는 penalty18k SIT62.68/97.74·CLIMB57.00/95.33·ON_TOP42.62/94.01·전체50.13/95.07, fusion13k60.13/98.57·52.28/93.87·36.70/96.65·45.17/96.39, fusion_v2 10.5k51.40/98.52·47.39/94.11·24.68/95.51·35.51/95.80이다. 단독5행동 평균96.72/96.80/96.26%다. 직전 전체 대비+7.51/+6.11/−2.75%p이며 앞 두 모델은3seed 모두 상승했다.
- 동일 일반 평가 전체 후행3seed 평균으로 컨피그별 최고를 판정했다. penalty18k50.13±1.07/95.07가 기존15k보다+5.43%p, fusion13k45.17±1.01/96.39가8k보다+.43%p로 최고를 갱신했다(후자는 작은 차이). fusion_v2 최고는9k40.85±.68/96.82%로 유지했다. 전체 이력·최고 원본pth 경로/SHA256을 저장했고1k 추세에 이번 명시적 최신10.5k를 추가했다.
- 최고를 갱신한 두 모델만 행동별/전체공통 공유상자 교집합을 각512환경×3seed 추가 평가했다(총12회, near50 유지). 행동별 전체 후행±표준편차/같은 쌍 선행 AT%는 penalty62.46±.82/94.35·fusion60.38±1.47/95.69, 공통64.24±2.77/93.73·60.46±2.03/95.98이다. 각 조건1721쌍의 행동별 분모/선행 상한·크기격자·비공유payload 보존을 검사했고 기존fusion8k near50 교집합과 초기 상태가 정확히 일치했다. 일반 크기와의 비교는 크기별 충돌 재시도가 포함된 분포 비교다.
- armed·선행 AT Ever 전 root XYZ≤.3m/s·30프레임 첫 정지의 사람→목표XY 중앙값 SIT/CLIMB/ON_TOP(m)은 penalty1.22/1.92/2.76·fusion1.94/2.21/2.87·fusion_v2 1.48/2.05/1.92, 쌍별 초기거리 감소율 중앙값30.0/15.4/3.0·14.7/.1/1.2·16.7/.9/3.5%다. 정지/관찰가능은9/9·8/8·23/25,10/11·7/7·23/23,8/10·7/7·16/18쌍이며 root 첫 정지와 전신/지속 대기를 구분한다. 교집합 정지는 별도 native가 아닌 정식 평가 내 telemetry다.
- 일반9·native3·교집합12회 정상 종료, CPU/GPU/JSON 재채점 일치·float32/64 성공 차이0·초기 상태 동일·native 진입/AT Ever 해제·동결encoder/RMS·원본checkpoint·운영파일262개·기존3학습 생존·조건부 최고치 판정 검증 통과. [컨피그별 최고](output_etc/check/penalty18k_fusion13k_fusionv2_10500_eval_stop_20261105/best_by_config.csv), [직전 비교](output_etc/check/penalty18k_fusion13k_fusionv2_10500_eval_stop_20261105/before_comparison.csv), [추세](output_etc/check/penalty18k_fusion13k_fusionv2_10500_eval_stop_20261105/latest_trends.png), [정지 비교](output_etc/check/penalty18k_fusion13k_fusionv2_10500_eval_stop_20261105/stop_comparison.csv), [교집합](output_etc/check/penalty18k_fusion13k_fusionv2_10500_intersections_20261105/comparison.csv), [검증](output_etc/check/penalty18k_fusion13k_fusionv2_10500_eval_stop_20261105/combined_verification.json).

### 최고3개 저장본 near 강제 해제·교집합 512환경 평가

- 사용자 지정 penalty_v2 15k·fusion_v1 8k·fusion_v2 9k의 후행 거리 near50 강제를 해제하고 행동별/전체공통 공유상자 교집합을 각512환경×seed0/1/2×1200step 평가했다. 총18회, GPU3·4·5·2·6 사용·GPU0·1 미사용. 기존 혼합 random_scenario·pairing 확률·loco·4명/8토큰·Plane Ever·개인 낙상 동결을 유지했다.
- check 전용 driver에서 기존 near 난수 소비는 보존하면서 반환 mask를0으로 바꿨다. 저장된 YAML/체크포인트의 near_probability=.5는 수정하지 않았고, 실제18회 reset 후 near flags/modes가 모두0임을 저장·검증했다. 후행이 우연히 더 가까운 사례는 유지한다. 선행 두 구간1–5m·후행 SIT/CLIMB 목표거리1–10m·ON_TOP pickup/transport 각1–5m·사람→목표≥1m를 실제 초기 상태에서 전수 검사했다(수치 허용1mm).
- 행동별 교집합의 후행 전체 동시/같은 쌍 선행 AT 평균%는 penalty60.08/94.37·fusion_v1 61.91/94.83·fusion_v2 59.56/96.05, 공통은60.68/94.19·62.42/95.53·59.76/95.99다. 조건/저장본마다 SIT415·CLIMB352·ON_TOP954·전체1721쌍이며 표준편차·행동별 선행 상한을 보존했다.
- 기존 fusion_v1 8k near50 교집합512×3 결과는 재실행 없이 비교용으로 재사용했다. 동일 checkpoint SHA256·운영파일262개·초기 역할/packet/크기/asset uniform을 검증했다. near0/near50 전체 후행 차이는 행동별+.66%p·공통+.22%p로 작았지만 공통 SIT−7.88%p·ON_TOP+3.64%p는3seed 공통 방향이었다. 선행 AT는 행동별94.54→94.83·공통95.53→95.53%다.
- 실제 후행 수행거리가 더 짧은 쌍은 행동별near0 925/1721(53.75%)·near50 1335/1721(77.57%), 공통926/1721(53.81%)·1333/1721(77.45%)였다. near0는 먼 쌍만 고르는 필터가 아니며 기본거리 분포 평가다. 같은 near0 조건/seed 내 세 모델 초기 장면은 정확히 같았다. near0/near50 간 역할/크기는 같지만 거리·충돌 재시도로 초기 위치/모션·목표가 달라지는 분포 비교임을 명시했다.
- 별도 native64 실행 없이18회 평가 내 첫 정지 telemetry·초기거리 감소율·변위/IQR을 수집했고 종전 native 장면 종료 진단과 구분했다. 전수1200step·finite·CPU/GPU/JSON 재채점 일치·float32/64 성공 차이0·공유 물리/alias 크기/격자/비공유 payload 보존·encoder/RMS 동결·원본checkpoint·운영파일262개·기존3학습 생존 검증 통과. [전체 비교](output_etc/check/best3_near0_intersections512x3_20261105/comparison.csv), [그림](output_etc/check/best3_near0_intersections512x3_20261105/comparison.png), [거리 분포](output_etc/check/best3_near0_intersections512x3_20261105/distance_distribution.png), [정지 기록](output_etc/check/best3_near0_intersections512x3_20261105/waiting_aggregate.csv), [검증](output_etc/check/best3_near0_intersections512x3_20261105/combined_verification.json).

### penalty_v2 17k·fusion_v1 12k·fusion_v2 10k 평가 및 최고치 조건부 교집합 판정

- 사용자 지정17k/12k부터 일반512환경×3seed×1200step Plane Ever 평가를 시작하고, fusion_v2 10k는 완전 저장을 확인한 뒤 같은 평가에 추가했다. 각 저장본의64환경×seed0 native 정지 진단을 병행했다. 실제 일반 GPU3/4/2·진단GPU5/6 사용, GPU0·1 미사용; 기존 학습·체크포인트·운영 코드/설정을 보존했다.
- 기존 일반 크기·random_scenario/near50·4명/8토큰·loco·deterministic·개인 낙상 동결을 유지했다. 후행 동시/같은 쌍 선행 AT 평균%는 penalty17k SIT57.78/97.08·CLIMB50.11/95.42·ON_TOP33.95/93.22·전체42.62/94.53, fusion12k53.74/96.60·50.35/93.12·28.97/96.02·39.06/95.45, fusion_v2 10k46.39/97.43·51.65/93.67·29.71/95.58·38.26/95.50이다. 단독5행동 평균은96.00/96.53/96.75%.
- 직전16k/11k/9k 대비 전체 후행 변화는+.47/−3.74/−2.59%p이며각각3seed 공통 방향이다. fusion_v1 ON_TOP−5.15%p와 fusion_v2 SIT−7.86%p도3seed 공통 하락했다. 단일 저장본 변동과 장기 추세를 구분하고 기존1k 이력에 세 결과를 추가했다.
- ‘최고치’는 동일 모델의 이전 모든 일반 평가 중 전체 BEFORE 후행 동시 Plane Ever3seed 평균 최고를 넘는 것으로 정의했다. 기존 penalty15k44.70/93.47·fusion_v1 8k44.74/95.45·fusion_v2 9k40.85/96.82%를 세 저장본 모두 넘지 못했다. 따라서 사용자 조건에 따라 행동별/공통 교집합 추가 rollout은0회이며, 일반 평가와 교집합 평가를 혼동하지 않도록 별도 조건/계획/판정 파일을 남겼다.
- armed·선행 AT Ever 전 root XYZ≤.3m/s·30프레임 첫 정지의 사람→목표XY 중앙값 SIT/CLIMB/ON_TOP(m)은 penalty1.21/1.70/2.08·fusion1.89/2.13/2.93·fusion_v2 1.47/1.97/1.88이다. 초기거리 감소율 중앙값은27.6/16.7/8.4·14.3/2.8/.7·14.3/−2.3/4.8%, 정지/관찰가능은9/11·8/8·23/27,9/9·8/8·23/23,8/9·6/6·15/15쌍이다. 조건부 표본 변화와 전신/지속 대기를 구분하며 공통 정지 쌍 비교도 저장했다.
- 정식9회·native3회1200step 정상 종료·CPU/GPU/JSON 재채점 일치·float32/64 성공 차이0·기존 동일 seed 초기 상태 정확 일치·native 진입/AT Ever 해제·encoder/RMS 동결·원본checkpoint·운영파일262개·기존3학습 생존 검증 통과. 최고치 판정 점수와 정식 집계 일치, 조건부 교집합 대상 공집합을 추가 검증했다. [직전 비교](output_etc/check/penalty17k_fusion12k_fusionv2_10k_eval_stop_20261105/before_comparison.csv), [이전 최고와 비교](output_etc/check/penalty17k_fusion12k_fusionv2_10k_eval_stop_20261105/best_checkpoint_comparison.csv), [추세](output_etc/check/penalty17k_fusion12k_fusionv2_10k_eval_stop_20261105/integer_trends.png), [정지 비교](output_etc/check/penalty17k_fusion12k_fusionv2_10k_eval_stop_20261105/stop_comparison.csv), [검증](output_etc/check/penalty17k_fusion12k_fusionv2_10k_eval_stop_20261105/combined_verification.json).

### BEFORE 성공 방해 요인 기존 trajectory CPU 분석

- 정지거리 조정과 분리해 fusion_v2 9k·fusion_v1 8/11k·penalty_v2 15/16k의 기존 일반 평가15회(512환경×3seed×1200step), 총8445쌍을 CPU에서 재분석했다. 새 GPU rollout·운영 코드/설정 변경 없이 root/box trajectory에서 후행·선행·동시 Plane 성공을 재구성했고15회 모두 저장된 결과와 정확히 일치했다.
- fusion_v2 9k 전체 쌍의 seed 평균%는 동시성공40.85·선행 AT 실패3.18·선행 성공/후행 개인성공 없음55.02·둘 다 Ever/동시 성공 없음.95다. 주된 실패는 후행 수행이며 단순 동시 판정 차이는 작았다. 후행 개인 실패929쌍에서 선행 첫 성공 시점 중앙값5.83초, 이후 두 사람 모두 살아 있는 시간34.03초·선행 gate on32.93초로 시간 부족/대부분의 지속 gate off로 설명되지 않았다.
- 현재 sampler는 SIT/CLIMB 공유상자 크기 호환을 검사하지만 ON_TOP은 항상 허용한다. 기존3seed ON_TOP 받침950개 중830개(87.4%)가 현재 SUPPORT 크기 범위 밖이었다. 범위 안120개/밖830개에서 후행 동시/같은 쌍 선행 AT 평균%는 fusion_v2 9k62.47/97.77 대26.19/96.28, fusion_v1 8k64.26/94.95 대31.85/96.10, penalty15k62.94/93.75 대33.78/92.73이었다. 세 모델에서 같은 경향이며 크기별 다른 장면의 관측 비교로 인과 효과를 단정하지 않는다.
- fusion_v2 9k의 선행 성공/후행 개인실패 SIT148개 중108개, CLIMB170개 중110개는 progress 반경까지 접근했지만 실제 받침의 내부80% XY 영역에 진입하지 못했다. ON_TOP611개 중437개는 선행 성공 후 payload가 실제 받침 progress 반경에 도달하지 못했고5개는 그 전에 후행이 낙상했다. 전체611개 중534개는 payload가 초기보다.2m 이상 올라간 적이 있지만 손 접촉/안정 집기는 trace에 없으므로 이를 집기 성공으로 단정하지 않는다. 사람은 받침1m 이내에 왔지만 payload는 progress 반경에 오지 못한 사례186개를 확인했다.
- 선행 AT 완료 이후 선행 root가 받침1m 안에 머문 시간비율 중앙값은 fusion_v2 9k 후행 개인실패 SIT65.1/CLIMB100.0/ON_TOP20.2%, 동시성공8.7/7.5/7.9%였다. fusion_v1·penalty에서도 같은 방향으로, 선행의 공간 점유가 추가 진단 후보지만 접촉/후퇴 개입이 없어 방해의 인과 증거는 아니다. 전체 동시실패999개 중후행 낙상동반74개(7.4%), 선행 성공 전 낙상8개로 전체 병목을 낙상만으로 설명하지 않는다.
- 단독 SIT98.51·CLIMB97.46·ON_TOP97.15%와 BEFORE 격차, 초기 거리별 비단조 패턴, 가상/실제 pose의 고정 trajectory 재채점도 함께 기록했다. 정지 반경을 바꾸기 전에 공유 받침 크기 분포와 완료 후 선행 공간 점유·후행 운반/안착을 우선 분리 진단할 근거다. 운영파일262개 보존·동일쌍 선행 상한 검증 통과. [요약 그림](output_etc/check/success_blockers_20261105/blockers.png), [분모/선행 포함 부분집합](output_etc/check/success_blockers_20261105/splits.csv), [쌍별 상세](output_etc/check/success_blockers_20261105/detail.csv), [검증/한계](output_etc/check/success_blockers_20261105/verification.json).

### fusion_v2 9000 일반 평가 및 native 정지 진단

- 원래 학습 크기·random_scenario/near50·4명/8토큰·loco·deterministic을 유지해 GPU3에서512환경×seed0/1/2×1200step 일반 Plane Ever 평가, GPU6에서64환경×seed0 native 정지 진단을 수행했다. 크기 교집합·viewer 거리 override 없이9k 저장본을 고정했고 GPU0·1과 기존 학습을 보존했다.
- 후행 동시/같은 쌍 선행 AT 평균%는 SIT54.25/98.25·CLIMB53.10/96.47·ON_TOP30.89/96.43·전체40.85/96.82, 단독5행동 평균97.03±.79%다. 8k36.21/95.85 대비 전체+4.64%p, SIT+7.29·CLIMB+2.85·ON_TOP+4.45%p로 세 행동과 전체가 평가3seed 모두 상승했다. Fusion v2의 기존 평가 중 전체 최고이며6/7/8/9k 전체35.87/36.00/36.21/40.85%로 최근 상승을 확인했다. 선행도+.97%p 상승했으므로 후행 개선 전체를 단독 정책 변화로 분리하지 않는다.
- armed·선행 AT Ever 전 root XYZ≤.3m/s·30프레임 첫 정지의 사람→목표XY 중앙값 SIT/CLIMB/ON_TOP은1.82/2.02/2.01m, 쌍별 초기거리 감소율16.1/−2.8/5.2%, 출발점 대비XY 변위 .44/.40/.35m다. 정지/관찰가능9/9·6/6·16/17쌍이며8k 대비 큰 정지 패턴 변화는 없었다. ON_TOP 진입은human·반경buffer2.5m·AT Ever 해제 latch를 실제 runtime과 대조했다. root 첫 정지는 전신/선행 완료까지 지속 대기와 구분한다.
- 정식3회·진단1회1200step 정상 종료·CPU/GPU/JSON 재채점 일치·float32/64 성공 차이0·기존 동일 seed 초기 상태 정확 일치·native 진입/AT Ever 해제·encoder/RMS 동결·원본checkpoint·운영파일262개·기존3학습 생존 검증 통과. 일반 성공률 이력과 정지 이력에9k를 추가하고8→9k 동일쌍 정지 비교도 보존했다. [8k 비교](output_etc/check/fusionv2_9000_eval_stop_20261105/before_comparison.csv), [1k 추세](output_etc/check/fusionv2_9000_eval_stop_20261105/integer_trends.csv), [그래프](output_etc/check/fusionv2_9000_eval_stop_20261105/integer_trends.png), [정지 비교](output_etc/check/fusionv2_9000_eval_stop_20261105/stop_comparison.csv), [검증](output_etc/check/fusionv2_9000_eval_stop_20261105/combined_verification.json).

### fusion_v1 상위 정수 저장본 6k·8k·11k 공유상자 교집합 512환경 비교

- 기존 일반3seed 후행 전체 동시 Plane Ever가 높은 정수1k 저장본8k44.74%·11k42.79%·6k42.41%를 선택해, 행동별/전체공통 교집합을 각512환경×seed0/1/2×1200step 평가했다. GPU3·4·6에서 총18회 수행했고 GPU0·1·운영 코드·학습 설정·기존 학습을 보존했다.
- 이전64환경 진단 driver를 check 아래에서 재사용했다. 물리 자산 생성 전 선행 공유상자만 AT∩후행 크기 범위로 교체하며5cm 격자·비공유 AT·ON_TOP payload 보존을 검증했다. 행동별 XYZ 범위는 SIT .25–.60/.25–.60/.35–.50, CLIMB .40–.60/.40–.60/.20–.60, ON_TOP .40–.60/.40–.60/.25–.50m; 공통은 .40–.60/.40–.60/.35–.50m다. random_scenario/near50·4명/8토큰·loco·deterministic·개인 낙상 동결·기존 Plane Ever 채점을 유지하고 사용자 지정 크기 진단으로 표시했다.
- 행동별 교집합6/8/11k의 전체 후행 동시/같은 쌍 선행 AT 평균%는61.88/94.48·61.24/94.54·57.36/95.16, 공통은63.98/95.24·62.20/95.53·57.74/95.52다. ON_TOP 후행은 행동별66.69/65.38/59.51%, 공통64.35/63.02/57.65%로11k가 두 조건에서 낮았다. 표준편차·행동별 동일분모 선행 AT·기존 일반 크기 평가 비교를 저장했다. 총분모는 조건/저장본마다 SIT415·CLIMB352·ON_TOP954쌍이다.
- 같은 조건/seed에서 세 저장본의 초기 장면이 정확히 같았다. 조건 간 graph packet·역할·원본자산·크기uniform 난수는 같지만 크기별 충돌 재시도로 사람 초기 배치가 seed별128/118/139환경에서 달라졌다. 기존 일반 크기 및 두 교집합 결과를 크기만 바꾼 완전 동일 배치의 인과 비교로 해석하지 않는다.
- 별도 native 실행을 추가하지 않고18회 평가 내 armed·AT Ever 해제 전 root XYZ≤.3m/s·30프레임 첫 정지 telemetry를 기록했다. 사람→목표XY 중앙값은 조건/저장본별 SIT2.20–2.31·CLIMB2.06–2.13·ON_TOP3.08–3.15m, 쌍별 초기거리 감소율 중앙값은 각각1.0–2.9/1.7–2.8/.1–.5%다. 이번 개인 낙상 동결·교집합 크기 조건 내 진단이며 종전 native64환경 기록과 구분한다.
- 18회 정상 종료·CPU/GPU/JSON 재채점 일치·float32/64 성공 차이0·실제 물리/원본 및 alias 관측 크기·범위/격자/균등난수 대응·encoder/RMS 동결·checkpoint/운영파일262개·기존3학습 생존 검증 통과. [성공률 비교](output_etc/check/fusion_top3_size512x3_20261105/comparison.csv), [그래프](output_etc/check/fusion_top3_size512x3_20261105/comparison.png), [정지 기록](output_etc/check/fusion_top3_size512x3_20261105/waiting_aggregate.csv), [검증](output_etc/check/fusion_top3_size512x3_20261105/verification.json).

### penalty_v2 16000 / fusion_v1 11000 일반 평가 및 native 정지 진단

- GPU3·4에서 각512환경×3seed×1200step 일반 random_scenario/near50 Plane Ever 평가, GPU5·6에서 같은 저장본 각64환경×seed0 native 정지 진단을 병행했다. 공유상자 교집합 override 없이 원래 크기 분포를 사용했고 모든 viewer 옵션을 해제했다. GPU0·1 미사용, 운영 코드·학습 설정 변경 없음.
- penalty16k 후행 동시/같은 쌍 선행 AT 평균%는 SIT60.98/98.25·CLIMB47.19/94.87·ON_TOP33.23/92.81·전체42.16/94.42, 단독5행동 평균96.04%. 15k44.70/93.47 대비 전체−2.54%p·ON_TOP−4.31%p는 평가3seed 공통 하락이며 SIT/CLIMB 변화는 혼재했다. 선행 성공률 하락으로 설명되지 않으며 기존 전체 최고는15k다.
- fusion_v1 11k SIT55.05/97.08·CLIMB52.71/94.19·ON_TOP34.12/95.37·전체42.79/95.44, 단독평균97.00%. 10k40.66/95.99 대비 전체+2.13%p·ON_TOP+3.19%p는3seed 공통 회복했다. 기존 최고8k44.74/95.45에는 미달하며 SIT seed 표준편차9.94%p를 함께 기록했다.
- armed·AT Ever 전 root XYZ≤.3m/s·30프레임 정지의 사람→목표XY 중앙값 SIT/CLIMB/ON_TOP(m)은 penalty1.15/1.50/2.22, fusion1.91/2.15/2.86. 정지/관찰가능 쌍은9/9·9/9·24/28, 9/10·9/9·23/23이다. 초기거리 감소율 중앙값은23.0/52.6/9.7%, 15.0/1.5/1.1%; 출발점 대비XY 변위는 .59/.96/.52m, .42/.39/.41m다. penalty15/16k 모두 정지한 CLIMB7쌍의 감소율17.3→37.4%·변위 .59→.72m로16k에서 더 접근했다. root 정지와 전신/지속 대기를 구분한다.
- 정식6회/진단2회 정상 종료, CPU/GPU/JSON 재채점 일치·float32/64 차이0·기존 일반/64환경 초기 상태 정확 일치·native 진입/AT Ever 해제·동결 encoder/RMS·원본 checkpoint·운영파일262개·기존3학습 보존 검증 통과. 기존 평가와 운영파일 해시 동일, 정수1k 성공률 이력35개와 정지 이력을 갱신했다. [직전 비교](output_etc/check/penalty16k_fusion11k_eval_stop_20261105/before_comparison.csv), [1k 이력](output_etc/check/penalty16k_fusion11k_eval_stop_20261105/integer_trends.csv), [그래프](output_etc/check/penalty16k_fusion11k_eval_stop_20261105/integer_trends.png), [정지 비교](output_etc/check/penalty16k_fusion11k_eval_stop_20261105/stop_comparison.csv), [검증](output_etc/check/penalty16k_fusion11k_eval_stop_20261105/combined_verification.json).

### 최고 저장본 3개 BEFORE 공유상자 교집합 크기 64환경 비교

- 기존3seed 전체 BEFORE 후행 동시 Plane Ever 평균으로 최고 저장본 penalty_v2 15k·fusion_v1 8k·v2 8k를 선택했다. 사용자 선택에 따라 기존 혼합 random_scenario·near50·4명/8토큰·64환경·seed0·1200step에서 행동별 교집합/전체 공통 교집합을 각1회, 총6회 평가했다. GPU3·4·5 사용, GPU0·1 미사용.
- 체크포인트/학습 코드를 바꾸지 않고 check 아래 진단 driver에서 물리 자산 생성 전에 선행 공유상자만 교체했다. AT와 SIT/CLIMB/ON_TOP 받침의 교집합은 각각 XYZ .25–.60/.25–.60/.35–.50, .40–.60/.40–.60/.20–.60, .40–.60/.40–.60/.25–.50m이며 공통은 .40–.60/.40–.60/.35–.50m, 모두 .05m 격자다. 원래 크기에 기존 혼합 역할 sampler를1회 적용하고 역할/의존 그래프를 고정해 조건별 분모를 맞췄다. 독립 AT·ON_TOP payload 등 비공유 크기는 보존했다.
- 각 조건은 SIT12·CLIMB12·ON_TOP41, 전체65쌍이다. 행동별 교집합의 전체 후행 동시/같은 쌍 선행 AT 평균%는 penalty70.77/96.92·fusion_v1 75.38/95.38·v2 50.77/95.38, 공통은64.62/92.31·69.23/93.85·53.85/95.38이다. 단일 seed·작은 표본의 조건 비교이며 기존512환경 결과 대비 크기 효과로 직접 해석하지 않는다.
- 같은 조건 안에서 세 모델의 초기 상태가 정확히 일치했다. 두 조건 간 원본 자산·역할·uniform 크기 난수·graph packet·logical 순서는 같지만 크기에 따른 물리 유효성 재시도로18/64환경의 초기 사람XY 배치가 달라졌다. 두 조건 차이를 크기만 바꾼 동일 배치 실험으로 해석하지 않는다.
- 별도 시뮬레이션을 늘리지 않고6회 평가 안에서 대기 telemetry를 수집했다. armed·AT Ever 해제 전 root XYZ≤.3m/s·30프레임 정지/관찰가능 쌍은 행동별 조건 penalty40/43·fusion38/38·v2 26/26, 공통43/47·39/39·28/31이다. 이번 개인 낙상 동결 평가 내 관측이며 이전 native 장면 종료 정지 프로토콜과 구분한다. 거리·초기거리 감소율·변위·IQR도 저장했다.
- 6회1200step·finite·CPU/GPU/JSON 재채점 일치·float32/64 성공 차이0·실제 공유 자산/원본 및 alias 관측 크기·요청한 범위/5cm 격자·비공유 물체 보존·encoder/RMS 동결·원본 checkpoint/운영파일262개·기존3학습 생존 검증 통과. [성공률](output_etc/check/before_size_intersections_best64_20261105/success.csv), [정지 기록](output_etc/check/before_size_intersections_best64_20261105/waiting.csv), [배치 비교](output_etc/check/before_size_intersections_best64_20261105/placement_comparison.json), [검증](output_etc/check/before_size_intersections_best64_20261105/verification.json).

### 세 모델 정수 1k 평가 공백 보완 및 fusion_v2 8000 추가

- 미평가 penalty_v2 4/5/11/12/13/14k·fusion_v1 6/7/8/9k·fusion_v2 4/6/7k와 추가 요청된 v2 8k를 각각512환경×3seed×1200step Plane Ever로 평가했다. GPU2·3·4·5를 사용했고 같은14개 저장본의64환경×seed0 native 정지 진단은 GPU6 및 v2 8k GPU4에서 수행했다. GPU0·1 미사용, random_scenario/near50·기존 규약 유지.
- 기존19개 정수 k 평가와 합쳐 penalty1–15k·fusion_v1 1–10k·v2 1–8k 총33개 결과를 채웠다. 소수 k는 1k 표·그래프에서 제외했으며 평균/표준편차·같은 쌍/행동/거리 부분집합의 선행 AT·단독5행동 평균을 보존했다. 각 모델 CSV, 구간별3seed 변화량, 그래프를 저장했다.
- penalty10→11→12→13→14→15k 전체 후행 동시/같은 쌍 선행 AT 평균%는33.47/91.34→39.13/91.47→39.78/92.67→38.40/92.73→42.47/94.13→44.70/93.47이다. 13k ON_TOP 하락 뒤14·15k 전체와ON_TOP이3seed 공통 상승했다. 단독평균15k96.99%.
- fusion_v1 6/7/8/9/10k 전체 후행/선행은42.41/94.28·40.84/94.95·44.74/95.45·41.56/94.83·40.66/95.99%로8k가 정수 k 중 최고다. v2 6/7/8k는35.87/95.27·36.00/95.67·36.21/95.85%로 전체는 유지되지만 CLIMB 후행/선행은46.01/93.63→49.17/94.37→50.25/95.14로 개선됐다. 희소3k/7.5k 비교에서의 CLIMB 정체 해석을 보완한다. v2 8k SIT46.96/97.75·ON_TOP26.45/95.47·단독평균97.04%.
- 정규화 정지 이력에 새14개 진단을 합치고 정수 k만 CSV로 추출했다. 같은 쌍 비교에서 penalty CLIMB 14→15k의 목표거리 중앙값 .164→1.888m·초기거리 감소율89.2→19.4%·출발점 대비XY 변위1.421→.643m(8쌍)로 변화했다.11–14k는 목표 가까이까지 접근 후 정지했으므로15k의 변화가 이후에도 유지되는지는 미확인이다. root 정지와 전신/지속 대기는 구분한다.
- 14k는 GPU3에서 완료한3seed를 최종 사용했다. GPU2에서 뒤늦게 겹쳐 시작한 자체 평가 자식만 종료하고 미완료 로그를 `superseded_duplicate/`에 보존했다. 완료 결과의 심링크·재채점은 check 경로 안에서만 처리했고 원본 checkpoint·기존 학습·viewer는 건드리지 않았다.
- 최종 선택된 정식42회·진단14회 정상 완료, CPU/GPU/JSON 재채점 일치·float32/64 차이0·동일 초기 상태·native 진입/AT Ever 해제·동결 encoder/RMS·원본 checkpoint/운영파일262개·기존3학습 생존 확인. 운영 코드·학습 설정 변경 없음. [1k 전체 표](output_etc/check/integer1k_penalty_fusion_fusionv2_eval_stop_20261105/integer_trends.csv), [그래프](output_etc/check/integer1k_penalty_fusion_fusionv2_eval_stop_20261105/integer_trends.png), [정지 이력](output_etc/check/integer1k_penalty_fusion_fusionv2_eval_stop_20261105/stop_integer_trends.csv), [공통 정지 쌍](output_etc/check/integer1k_penalty_fusion_fusionv2_eval_stop_20261105/stop_common_pairs.csv), [통합 검증](output_etc/check/integer1k_penalty_fusion_fusionv2_eval_stop_20261105/combined_verification.json).

### penalty_v2 12500 / fusion_v1 7500 / fusion_v2 5000 중간 저장본 평가

- GPU3·4·5에서 각 512환경×3seed×1200step Plane Ever 정식 평가, GPU6에서 같은 저장본 각각 64환경×seed0 정지 진단을 수행했다. random_scenario/near50 유지, viewer 거리 지정 해제, GPU0·1 미사용. 기존 운영 코드·학습 설정 변경 없음.
- 후행 동시/같은 쌍 선행 AT 평균%: penalty_v2 SIT53.79/98.77·CLIMB43.09/92.88·ON_TOP32.76/91.33·전체39.52/93.25, fusion_v1 56.05/97.06·54.84/94.74·33.41/94.57·전체43.09/95.11, fusion_v2 49.72/94.48·42.51/91.55·22.86/94.02·전체32.96/93.56. 단독5행동 평균은 각각94.04/96.89/95.70%다.
- 중간 저장본을 기존 이력에 삽입해 penalty10→12.5→15k 전체 후행33.47→39.52→44.70%, 선행91.34→93.25→93.47%를 확인했다. 두 구간 모두 전체 후행이3개 평가 seed에서 공통 상승했다. fusion_v1 5→7.5→10k는40.58/95.87→43.09/95.11→40.66/95.99로 등락했고, v2 3→5→7.5k는31.67/93.38→32.96/93.56→35.98/95.33이며 CLIMB는 유지·최근 ON_TOP은3seed 공통 상승했다.
- 같은5k/7.5k에서 fusion_v2 전체 후행은 v1보다 각각7.62/7.11%p 낮고 모두3seed 공통 차이다. 7.5k 선행은v1 95.11/v2 95.33%로 비슷하므로 후행 격차를 선행 성공률만으로 설명하지 않는다. 모델 외에 대기 반경·ON_TOP 진입 기준도 달라 개별 변경의 인과 효과를 단정하지 않는다.
- armed·AT Ever 전 root XYZ≤.3m/s·30프레임 정지의 초기거리 감소율 중앙값(SIT/CLIMB/ON_TOP)은 penalty43.5/67.0/2.2%, fusion_v1 12.4/4.2/1.5%, fusion_v2 15.6/1.5/6.3%; 출발점 대비XY 변위 중앙값은1.08/1.76/.44m, .40/.43/.37m, .46/.51/.40m다. 정지/관찰가능 쌍은10/10·9/9·22/27, 9/9·8/8·23/23, 9/11·7/8·17/17이다. penalty12.5k와15k 모두 정지한 CLIMB8쌍의 목표거리 중앙값은 .34→1.89m, 감소율75.4→19.4%로 변했다.64환경·조건부 진단이며 전신/지속 정지를 뜻하지 않는다.
- 정식9회·진단3회 정상 종료, CPU/GPU/JSON 재채점 일치·float32/64 차이0·동일 초기 상태·native 진입/AT Ever 해제·동결 encoder/RMS·원본 checkpoint·운영파일262개·기존3학습 보존 확인. 이전 평가와 운영파일 해시도 동일하다. [전체 성공률 이력](output_etc/check/midpoint_penalty12500_fusion7500_fusionv2_5000_eval_stop_20261105/before_comparison.csv), [구간/동일 epoch 비교](output_etc/check/midpoint_penalty12500_fusion7500_fusionv2_5000_eval_stop_20261105/comparison.json), [정규화 정지 이력](output_etc/check/midpoint_penalty12500_fusion7500_fusionv2_5000_eval_stop_20261105/stop_comparison.csv), [공통 정지 쌍](output_etc/check/midpoint_penalty12500_fusion7500_fusionv2_5000_eval_stop_20261105/stop_common_pairs.csv), [통합 검증](output_etc/check/midpoint_penalty12500_fusion7500_fusionv2_5000_eval_stop_20261105/combined_verification.json).

### penalty_v2 15000 / fusion_v1 10000 / fusion_v2 7500 최신 평가

- GPU3·4·5에서 시작 시 고정한 세 checkpoint를 각각 512환경×3seed×1200step Plane Ever로 평가하고, GPU6에서 같은 저장본을 각각 64환경×seed0로 정지 진단했다. random_scenario/near50 유지, viewer 거리 지정 해제, GPU0·1 미사용. 운영 코드·학습 설정 변경 없음.
- 후행 동시/같은 쌍 선행 AT 평균%: penalty_v2 SIT59.64/97.39·CLIMB48.93/91.66·ON_TOP37.54/92.79·전체44.70/93.47, fusion_v1 52.13/96.57·53.95/94.09·30.93/96.54·전체40.66/95.99, fusion_v2 46.84/96.60·42.49/93.92·29.23/95.48·전체35.98/95.33. 단독5행동 평균은 각각96.99/95.81/96.50%다.
- 직전10k/5k/3k 대비 전체 후행 변화는 +11.23/+0.08/+4.31%p. penalty 세 행동과 fusion_v2 SIT·ON_TOP은3seed 공통 상승했고, fusion_v1 SIT는3seed 공통 하락했다. 서로 다른 epoch의 비교이며 중간 추세를 보간하지 않는다.
- 첫 armed·선행 AT Ever 전 root XYZ≤.3m/s·30프레임 정지의 사람→목표XY 중앙값(m)은 penalty1.07/1.89/2.51, fusion_v1 1.90/2.11/2.81, fusion_v2 1.49/2.01/1.93(SIT/CLIMB/ON_TOP). 정지/관찰가능 쌍은 각각8/9·8/8·24/28, 9/9·6/7·23/24, 8/9·6/6·15/16이다.
- 쌍별 초기거리 대비 감소율 `100*(1-d_stop/d_initial)` 중앙값은 penalty32.3/19.4/4.6%, fusion_v1 14.0/−5.4/1.5%, fusion_v2 17.8/−3.3/5.0%; 실제 출발점 대비XY 변위 중앙값은 .63/.64/.47m, .40/.35/.33m, .47/.42/.44m다. 양 checkpoint 모두 정지한 같은 쌍 비교도 별도로 저장했다. root 정지는 전신 정지나 선행 완료까지 지속 대기를 뜻하지 않는다.
- 정식9회·진단3회 정상 종료, CPU/GPU/JSON 재채점 일치·float32/64 차이0·기존 초기 상태 일치·native 진입/AT Ever 해제·동결 encoder/RMS·checkpoint/운영파일262개·기존3학습 보존을 확인했다. [성공률 비교](output_etc/check/penalty15k_fusion10k_fusionv2_7500_eval_stop_20261105/before_comparison.csv), [정규화 정지 비교](output_etc/check/penalty15k_fusion10k_fusionv2_7500_eval_stop_20261105/stop_comparison.csv), [공통 정지 쌍](output_etc/check/penalty15k_fusion10k_fusionv2_7500_eval_stop_20261105/stop_common_pairs.csv), [통합 검증](output_etc/check/penalty15k_fusion10k_fusionv2_7500_eval_stop_20261105/combined_verification.json).

### fusion_v2 3000 평가 및 세 모델 64환경 정지 거리 추세

- GPU3에서 고정 fusion_v2 3000을512환경×3seed×1200step Plane Ever로 평가하고, GPU4에서 같은 저장본64환경×seed0 정지 진단을 수행했다. 새 viewer 거리 옵션을 명시적으로 해제하고 기존 random_scenario/near50·평가 규약을 유지했다. 운영 코드·학습 설정 변경 없음.
- 후행 동시/같은 쌍 선행 AT 평균%는 SIT39.97/96.53·CLIMB42.84/90.07·ON_TOP23.95/93.59·전체31.67/93.38. 전체는2k27.33/92.69에서+4.35%p, 단독5행동 평균95.97→96.04. CLIMB/ON_TOP/전체는3seed 공통 상승, SIT 변화는 혼재했다. 같은3k fusion_v1 전체34.40/94.03보다3seed 공통 낮으며 penalty_v2 전체23.16/84.91보다3seed 공통 높다.
- 첫 root XYZ≤.3m/s·1초 연속 정지의 사람→목표XY 중앙값 SIT1.42m(9/10)·CLIMB1.80m(7/7)·ON_TOP1.90m(18/20), ON_TOP payload1.63m. 2k1.42/1.84/1.88m에서 대체로 유지됐다. 출발지 초기XY .5m 이내 유지16/72쌍, 그중 성공 후3초 관찰가능16쌍 중14쌍이 .5m 이상 이동했다.
- penalty_v2 6.5/8.5/9/10k·fusion_v1 1.5/3/4/5k·fusion_v2 .5/1/2/3k의64환경 정지 기록을 집계했다. 12개 기록의 초기 상태 정확 일치, root 속력/시간·native 장면 종료 규약을 확인하고32환경 기록은 제외했다. 같은 초기 장면이어도 정지/관찰가능 부분집합은 checkpoint별로 달라지며 정지 위치만으로 접근 후 대기라고 해석하지 않는다.
- 정식3회/진단1회1200step 정상 종료·CPU/GPU/JSON 재채점 일치·float32/64 차이0·기존 초기 상태 일치·native human 진입/AT Ever 해제·동결 encoder/RMS·checkpoint/운영파일262개·기존3학습 보존 확인. [성공률 비교](output_etc/check/fusion_v2_3000_eval_stop_20261105/before_comparison.csv), [정지 추세](output_etc/check/fusion_v2_3000_eval_stop_20261105/stop_trends.csv), [그래프](output_etc/check/fusion_v2_3000_eval_stop_20261105/stop_trends.png), [검증](output_etc/check/fusion_v2_3000_eval_stop_20261105/combined_verification.json).

### goalroute BEFORE viewer의 목표→후행 사람 거리 지정

- `BEFORE_FOLLOWER_GOAL_DISTANCE=<m>` / `--before_follower_goal_distance <m>`를 추가했다. 사용자 확인에 따라 SIT·CLIMB·ON_TOP 모두 선행 AT 목표→후행 사람 root 초기 XY 거리다. 기존 선행 목표 거리 옵션과 독립 사용/병용 가능하며 생략 시 기존 동작이다.
- ON_TOP은 샘플링된 사람/payload를 함께 평행 이동해 집기 배치를 유지한다. payload→목표 거리는 그에 따라 변할 수 있다. 수동 거리 지정 시 near50 경로 부등식과 목표→사람 최소1m 제약을 대체하고 기존 물리 충돌 검사는 유지한다. 학습·정식 평가·비loco/비goalroute·잘못된 수치는 거부하며 YAML·checkpoint 계약·모델·보상·gate는 변경하지 않는다. config/USAGE에 명령을 기록했다.
- CPU76개 통과: .5/4/12m와 선행 목표 거리 조합, 여러 BEFORE 쌍·셔플된 목표/물체 매핑·부분 reset·선행/독립 목표·Z/회전 보존·ON_TOP 집기 벡터 보존·물리 충돌 검사 유지·formal 거부. 실제 CLI 로더의 유효7m+4m 및 잘못된7경로 거부 확인.
- GPU3·4·5에서 fusion_v1 5000 strict 복원, 각64환경 random_scenario·before_sit·before_climb·before_stack·all_solo 5경우16step finite Actor/Critic·가중치/가상 상자·부분 reset 검증. 총463쌍의 목표7m/후행 사람4m 오차는 .00005m 이내이며 ON_TOP 사람→payload 1–5m를 확인했다. GPU6·PORT6187 실제 VNC wrapper1환경·12step 제한1회 정상 종료. 원본/snapshot checkpoint·비대상 운영파일258개·기존3학습 보존. [검증](output_etc/check/before_follower_goal_distance_20261105/verification.json), [VNC 로그](output_etc/check/before_follower_goal_distance_20261105/vnc.log).

### goalroute BEFORE viewer의 선행 AT 목표 거리 지정

- test/VNC에 `BEFORE_AT_GOAL_DISTANCE=<m>`와 직접 CLI `--before_at_goal_distance <m>`를 추가했다. goalroute BEFORE loco 추론에서 선행 공유상자→AT 목표 초기 XY 거리만 지정하며 기존 목표 방향·Z·사람→상자 샘플링 규칙을 유지한다. 후행은 새 목표/선행 경로 길이를 기준으로 기존 goalroute/near50 배치를 적용한다. 생략 시 기존 동작이며 독립 과제·환경 YAML·checkpoint 계약·모델·보상·gate는 변경하지 않는다.
- 학습·정식 평가·비loco 및 0/음수/NaN/Inf를 거부한다. 기존 launcher·config CLI·formal guard·goalroute 배치에 최소 연결하고 config 추론 옵션/USAGE에 사용법을 기록했다.
- CPU50개 통과(새 거리/다중쌍·goal index 매핑·부분 reset·선행 물리 상태/Goal Z/독립 목표 보존 및 formal 거부 포함), 실제 config 로더의 유효7m/잘못된7경로 확인. GPU3·4·5에서 fusion_v1 5000 strict 복원 후 각64환경 random_scenario·before_sit·before_climb·before_stack·all_solo 5경우, 16step finite Actor/Critic·가중치 보존·가상 XYZ/실물 quaternion·부분 reset·near 조건을 확인했다. 총463쌍 목표거리 오차는 float32 월드좌표 기준 .00005m 이내다.
- GPU6·PORT6186 실제 VNC wrapper의 7m·CLIMB·1환경·12step 제한1회 종료까지 확인했다. 원본/snapshot checkpoint·비대상 운영파일258개·기존3학습 보존. [검증](output_etc/check/before_at_goal_distance_20261105/verification.json), [VNC 로그](output_etc/check/before_at_goal_distance_20261105/vnc.log).

## 2026-11-04

### penalty_v2 10000 / fusion_v1 5000 성공률·정지·출발지 분석

- GPU3·4에서 고정 penalty_v2 10000/fusion_v1 5000을 각각512환경×3seed×1200step Plane Ever로 평가하고, GPU5·6에서 같은 저장본64환경×seed0 정지 진단을 병행했다. 기존 random_scenario/near50·평가 규약 유지, 운영 코드·학습 설정 변경 없음.
- penalty_v2 9k→10k 후행 동시/같은 쌍 선행 AT 평균%는 전체34.42/91.06→33.47/91.34, 단독5행동 평균94.13→93.67. 행동별 후행/선행은 SIT47.12/96.20·CLIMB39.51/89.79·ON_TOP25.97/90.18. 전체 변화는seed별−5.00/+0.89/+1.27%p로 혼재하여 대체로 정체다.
- fusion_v1 4k→5k 전체39.39/94.78→40.58/95.87, 단독평균94.68→96.32. SIT59.17/98.31·CLIMB51.33/93.91·ON_TOP29.38/95.82. SIT/전체는3seed 공통 상승, ON_TOP은3seed 공통 하락, CLIMB 변화는 혼재했다. 모델별epoch가다르므로 같은 학습량 비교로 해석하지 않는다.
- 첫 root XYZ≤.3m/s·1초 연속 정지의 사람→목표XY 중앙값 SIT/CLIMB/ON_TOP(m): penalty .92/.63/2.18(정지/관찰가능9/10·11/11·24/29), fusion1.82/2.01/2.83(9/10·7/8·23/23). ON_TOP payload1.40/1.56m. root 정지는 발/팔 정지를 뜻하지 않으며 정식 성공률과 진단을 합산하지 않는다.
- 선행 성공 전 관측 구간 내내 초기XY .5m 이내 유지한 후행은 penalty11/72·fusion37/72쌍이다. 모두 성공 후3초를 관찰했고 각각11/11·33/37쌍이 그 안에 .5m 이상 이동했다. fusion 출발지 근처 유지 중14쌍은 성공 전 페널티가 활성화되지 않았으며, 정지 거리만으로 목표 접근 후 대기라고 해석하지 않는다.
- 정식6회/진단2회1200step 정상 종료·CPU/GPU/JSON 재채점 일치·float32/64 차이0·동일 초기 상태·native 진입/AT Ever 해제/정지 집계 일치·동결 encoder/RMS·checkpoint/운영파일262개 보존·기존3학습 생존 확인. [성공률 비교](output_etc/check/penalty10k_fusion5k_eval_stop_20261104/before_comparison.csv), [정지 비교](output_etc/check/penalty10k_fusion5k_eval_stop_20261104/stop_comparison.csv), [검증](output_etc/check/penalty10k_fusion5k_eval_stop_20261104/combined_verification.json).

### fusion_v2 2000 성공률·정지 거리·출발지 대기 진단

- GPU3에서 고정 fusion_v2 2000을 512환경×3seed×1200step Plane Ever로 평가하고, GPU4에서 같은 저장본 64환경×seed0 정지 진단을 수행했다. 기존 random_scenario/near50 유지, 운영 코드·학습 설정 변경 없음.
- 후행 동시/같은 쌍 선행 AT 평균%는 SIT40.39/94.28·CLIMB36.74/90.84·ON_TOP18.63/92.88·전체27.33/92.69. 전체는1k11.30/86.10에서 상승했고 후행 세 행동 모두3seed 공통 개선했다. 단독5행동 동일 가중치 평균93.46→95.97%. 같은2k penalty_v2는전체15.58/80.88, fusion_v1은29.41/90.65로 v2의 전체 우위를 확정하지 않는다.
- 첫 root XYZ≤.3m/s·1초 연속 정지의 사람→목표XY 중앙값 SIT1.42m(9/11)·CLIMB1.84m(8/8)·ON_TOP1.88m(18/18), ON_TOP payload1.51m. 분모는 armed·미해제1초 관찰 가능 쌍이며 발/팔 정지를 뜻하지 않는다.
- 별도 출발지 분석에서 선행 성공 전 관측 구간 내내 초기XY .5m 이내인 후행13/72쌍, 그중 성공 후3초 관찰 가능한13쌍 중10쌍이3초 내 .5m 이상 이동했다. 출발지 근처 유지와 목표로 접근한 뒤 정지를 구분하며, 정지 거리만으로 접근 후 대기라고 해석하지 않는다.
- 정식3회/진단1회1200step 정상 완료·CPU/GPU/JSON 재채점 일치·float32/64 차이0·기존 초기 상태 정확 일치·human/+2.5m 진입 및 AT Ever 해제 검증·동결 encoder/RMS·checkpoint/운영파일262개 보존·기존3학습 생존 확인. [비교](output_etc/check/fusion_v2_2000_eval_stop_20261104/before_comparison.csv), [정지](output_etc/check/fusion_v2_2000_eval_stop_20261104/stop_comparison.csv), [출발지 분석](output_etc/check/fusion_v2_2000_eval_stop_20261104/spawn_audit/metrics.json), [검증](output_etc/check/fusion_v2_2000_eval_stop_20261104/combined_verification.json).

### penalty_v2 9000 / fusion_v1 4000 성공률 + 정지 거리

- GPU3·4에서고정penalty_v2 9000/fusion_v1 4000을각512환경×3seed×1200step Plane Ever로평가하고,GPU5·6에서같은저장본각64환경×seed0×1200step 정지진단을병행했다. random_scenario/near50·기존평가규약유지,운영코드/학습설정변경없음.
- penalty_v2 8k→9k 전체후행동시/같은쌍선행AT 평균% 34.21/90.48→34.42/91.06, 단독5행동평균94.41→94.13. 후행SIT49.77/선행96.50·CLIMB38.68/87.48·ON_TOP26.98/90.53. SIT증가와CLIMB감소가겹쳐전체정체,seed별변화는혼재했다.
- fusion_v1 3k→4k 전체34.40/94.03→39.39/94.78, 단독평균96.04→94.68. 후행SIT47.52/선행97.68·CLIMB50.59/91.79·ON_TOP31.67/94.95. 전체/ON_TOP은3seed공통상승,SIT평균+8.21%p는seed별편차가크고CLIMB는유지됐다.
- 정지거리중앙값(사람→목표XY,m)SIT/CLIMB/ON_TOP: penalty_v2 9k .97/.13/2.16, fusion_v1 4k 1.95/2.16/2.84. ON_TOP payload는1.37/1.56m. 정지/관찰가능건수각9/10·10/12·21/27,10/10·9/9·23/23. penalty_v2 CLIMB는8.5k1.84m→9k.13m로가까워졌고progress포화반경밖1초정지는8/12→2/12다. root XYZ≤.3m/s·1초연속첫확인기준이며정식성공률과합산하지않는다.
- 성공률6회/정지2회1200step 정상완료·CPU/GPU/JSON재채점일치·float32/64차이0·각규모동일초기상태·native진입/AT Ever해제/정지집계일치·동결encoder/RMS·checkpoint/운영파일보존·기존3학습생존확인. [성공률 비교](output_etc/check/penalty9k_fusion4k_eval_stop_20261104/before_comparison.csv), [정지 비교](output_etc/check/penalty9k_fusion4k_eval_stop_20261104/stop_comparison.csv), [통합 검증](output_etc/check/penalty9k_fusion4k_eval_stop_20261104/combined_verification.json).

### fusion_v2 1000 성공률 평가 + 64환경 정지 진단

- 같은고정fusion_v2 1000 저장본으로GPU3·512환경×3seed×1200step Plane Ever 성공률과GPU4·64환경×seed0×1200step 일반 장면 낙상 종료 정지 진단을 병행했다. 기존sampler/near50 유지, 운영 코드/학습 설정 변경 없음.
- 전체 후행 동시/같은 쌍 선행 AT 평균%는0.5k 9.29/74.11→1k 11.30/86.10, 단독5행동 동일 가중치 평균83.77→93.46%다. 후행SIT10.01/선행91.74·CLIMB5.58/77.80·ON_TOP14.08/87.48%. ON_TOP/전체는3seed 공통 상승했다.
- 같은1k의fusion_v1은후행/선행전체18.74/86.56%, 단독평균93.41%. v2는후행SIT−10.76·CLIMB−14.91·ON_TOP−3.23·전체−7.44%p이며세행동모두3seed에서v1보다낮았다. 선행전체는비슷해후행차이를선행성공률만으로설명할수없다.
- 정지거리중앙값(사람→목표XY,m)은SIT .65→1.15·CLIMB .13→1.22·ON_TOP .62→1.81, ON_TOP payload .32→1.46으로변했다. 정지/관찰가능SIT10/12·CLIMB10/13·ON_TOP17/21, progress포화반경밖정지는각10/12·9/13·15/21이다. rootXYZ≤.3m/s·1초연속첫확인기준이며발/팔정지나완료후재실행을보장하는판정은아니다.
- 성공률3회·정지진단1회1200step 정상종료·CPU/GPU/JSON재채점일치·float32/64차이0·각규모의기존초기상태정확일치·native human진입/AT Ever해제/정지집계일치·동결encoder/RMS·가중치/checkpoint/운영파일보존·기존3학습생존확인. [성공률 비교](output_etc/check/fusion_v2_1000_eval_stop_20261104/before_comparison.csv), [정지 비교](output_etc/check/fusion_v2_1000_eval_stop_20261104/stop_comparison.csv), [통합 검증](output_etc/check/fusion_v2_1000_eval_stop_20261104/combined_verification.json).

### 최신 penalty_v2/fusion v1/v2 64환경 정지 거리 및 평가 동반 규칙

- 사용자 요청에 따라 향후 대기 페널티 계열의 eval에는 같은checkpoint의64환경·seed0 정지 거리 진단을 함께 수행하도록 AGENTS/평가 규약에 기록했다. 성공률 정식 평가와 장면 낙상 종료 기반 정지 진단은 구분한다.
- GPU3·4·5에서 최신 고정penalty_v2 8500 / fusion_v1 3000 / fusion_v2 500을각64환경·seed0·4명·8토큰·1200step으로 긴급 진단했다. 선행 미완료·armed 이후 root XYZ≤0.3m/s·1초 연속 첫 확인 시점의 XY거리 기준이다.
- 사람→AT목표 중앙값 SIT/CLIMB/ON_TOP(m): penalty_v2 1.06/1.84/2.01, fusion_v1 1.69/2.08/2.84, fusion_v2 .65/.13/.62. ON_TOP payload→목표는1.28/1.56/.32m. 정지/1초 관찰 가능 건수 SIT/CLIMB/ON_TOP은각7/10·12/12·23/29, 9/9·9/9·23/23, 7/15·13/15·12/25다. 서로epoch가달라v2의반경변경효과를분리해해석하지않는다.
- 진단 driver는 task용payload 거리와 실제wait 진입 거리를분리해fusion_v2의human 기준을반영하고, 매step native armed/released가해당거리·반경·AT Ever와일치하는지검사했다. 기존학습/운영코드 변경 없음.
- 3회1200step 정상 종료·finite·동일 초기 상태·runtime 진입/해제/정지 집계 일치·가중치/원본checkpoint/운영파일260개 보존·기존3학습 생존 확인. [거리/건수 CSV](output_etc/check/latest_three_stop64_20261104/distances.csv), [쌍별 기록](output_etc/check/latest_three_stop64_20261104/pairs.csv), [검증](output_etc/check/latest_three_stop64_20261104/verification.json).

### penalty_v2 8000 / fusion3000 / fusion_v2 500 정식 평가

- GPU2·3·4·5에서 고정penalty_v2 8000·fusion3000·fusion_v2 500 및 동일epoch 비교용penalty_v2 3000을각4명·512환경×3seed×1200step, 기존 random_scenario/near50·Plane Ever 규약으로 평가했다. 운영 코드/설정 변경 없음.
- 전체 후행 동시/같은 쌍 선행 AT 평균%: penalty_v2 8000 34.21/90.48(직전7000대비+2.97/+0.37%p), fusion3000 34.40/94.03(직전2000대비+4.99/+3.38%p), fusion_v2 500 9.29/74.11. 기존penalty_v2 CLIMB/ON_TOP/전체와fusion CLIMB/ON_TOP/전체는3seed 공통 상승했다. 단독5행동 동일 가중치 평균94.41/96.04/83.77%.
- 동일3000epoch penalty_v2→fusion 후행 SIT31.61→39.32·CLIMB21.74→50.16·ON_TOP20.64→26.12·전체23.16→34.40%, 같은 쌍 선행93.68→95.95·81.63→91.59·83.05→94.32·84.91→94.03%. 세 행동 모두3seed에서fusion우세.
- 동일500epoch fusion→fusion_v2 후행SIT10.03→8.34·CLIMB6.93→4.89·ON_TOP10.08→11.45·전체9.34→9.29%, 각 쌍 선행79.73→82.16·58.74→67.21·70.35→73.97·69.60→74.11%. ON_TOP은3seed 공통 개선했지만 전체 이득은 아직 없고, 단독평균88.56→83.77%로 저하했다. 초기 checkpoint이고 학습seed 반복 비교가 아니므로 개선/악화를 확정하지 않는다.
- 12회1200step 정상 완료·CPU/GPU/JSON 재채점 일치·float32/64 차이0·동일 초기 상태·같은 쌍 분모/상한·동결 encoder/RMS·checkpoint 및운영파일262개 보존·기존3학습 생존 확인. 실제 정지 거리/대기 행동 추가 진단 없음. [전체 추세](output_etc/check/penalty8k_fusion3k_fusionv2_500_eval_20261104/trends.csv), [seed별 변화](output_etc/check/penalty8k_fusion3k_fusionv2_500_eval_20261104/comparison.json), [검증](output_etc/check/penalty8k_fusion3k_fusionv2_500_eval_20261104/verification.json).

### fusion_v2 — ON_TOP 사람 기준 진입 / 대기 반경 +2.5m

- 기존fusion을 보존하고 `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg_v2` 환경/학습 YAML·train/test/VNC/eval wrapper를 추가했다. ON_TOP 대기 페널티 진입만 `ontop_distance_source: human`으로 선택하고, 모든 후행의 radius_buffer를3→2.5m로 변경했다. coefficient1.0·root XYZ 감점·진입 latch·선행 AT Ever 해제·Progress payload 기준·Virtual/fusion/bias·current Gate·AMP/PPO·샘플링은 유지한다. 같은 Stage1 26k에서 새로 전이한다.
- `before_wait.py`에 선택적 거리 기준을 연결하고 생략 시 기존payload 동작을 유지했다. validator·실험 등록·launcher·정식평가5개 분류 목록과 config/structure/reference 문서를 갱신했다. 기존/새 checkpoint 계약은 분리한다.
- CPU121개 통과(신규10개 포함):3행동 +2.5m 경계·XY 진입/XYZ 감점·ON_TOP 사람/payload 반대 위치·해제/부분reset·기존 설정/계약 보존. 27개 실험 설정 validator 및5개 shell 문법 검사 통과. 기존fusion2000 strict 복원/finite 확인.
- GPU4·4명/1024환경/8토큰 `CHECK=1 MAX_ITERATIONS=1` 학습exit0(기존 루프 저장epoch2). 저장본 strict 복원·Actor encoder/입력RMS 비트단위 동결·Actor/Critic finite·fusion 학습 확인. 새 eval wrapper64환경×seed0×1200step exit0·CPU/GPU/JSON 재채점 일치·float32/64 차이0·실제 runtime human/2.5/1.0 확인. 기존 운영파일250개 보존(승인된공통코드5개수정 제외), 기존프로세스 임의종료/재시작 없음. 본학습 미시작.
- [검증](output_etc/check/fusion_v2_config_20261104/verification.json), [CPU](output_etc/check/fusion_v2_config_20261104/cpu_tests.log), [학습](output_etc/check/fusion_v2_config_20261104/train.log), [GPU 평가](output_etc/check/fusion_v2_config_20261104/eval/general_seed0/result.json). 실행 명령은 [config](markdowns/config.md)에 추가했다.

### penalty_v2 1/2/4/5/6/7k 및 fusion1/2k·32환경 정지 거리 추세

- GPU2~6에서 요청8개 고정 저장본을각4명·32환경·seed0·1200step으로 진단했다. 기존 root XYZ≤0.3m/s·1초 연속·선행 미완료/armed 이후 첫 정지 기준과 일반 장면 낙상 종료를 유지했다. 운영 코드/설정 변경 없음, ON_TOP 진입은 저장본의 기존payload 기준이다.
- 사람→AT목표 XY거리 중앙값 SIT/CLIMB/ON_TOP(m): penalty_v2 1k .46/.09/.64, 2k .88/.11/.57, 4k 1.48/.07/1.21, 5k 1.55/.13/1.68, 6k 2.02/1.33/2.42, 7k 1.28/1.27/2.35; fusion1k 1.74/.63/2.75, 2k 2.43/3.01/2.79. ON_TOP payload 중앙값은각.20/.15/1.23/1.49/1.62/1.49/1.80/2.28m다.
- 동일32장면의 후행 SIT14·CLIMB6·ON_TOP16쌍 중 정지 사례만 거리 집계, 관찰 가능 분모와정지 건수를 CSV에 별도 기록했다. 소표본이고 root정지는발/팔 정지를보장하지않는다. 64환경 진단과표본을혼합하지않는다.
- 8회1200step 정상 완료·finite·모든 초기 상태 정확 일치·AT Ever 해제/runtime 정지 집계 일치·가중치/원본checkpoint/운영파일254개 보존·기존4학습 생존 확인. [거리/사례 수 CSV](output_etc/check/penalty_fusion_stop32_epochs_20261104/distances.csv), [쌍별 기록](output_etc/check/penalty_fusion_stop32_epochs_20261104/pairs.csv), [검증](output_etc/check/penalty_fusion_stop32_epochs_20261104/verification.json).

### fusion2000 6명/8명·64환경 진단 평가

- GPU3·4에서 같은 fusion2000 고정 저장본을6명/12토큰·8명/16토큰, 각각64환경·seed0·1200step으로 평가했다. 기존 인원 변경 확인 driver에서 Plane Ever·개인 낙상 freeze·random_scenario/near50를 재사용했다. 학습4명에서 변경한 소표본 진단으로512환경×3seed 정식 평가와 구분한다.
- 6명→8명 후행 동시/같은 쌍 선행 AT(%): SIT31.43/91.43(35쌍)→45.95/97.30(37쌍), CLIMB37.50/75.00(24)→32.56/81.40(43), ON_TOP18.37/83.67(49)→16.22/86.49(74), 전체26.85/84.26(108)→27.92/87.66(154). 단독5행동 동일 가중치 평균95.24→90.41%, 개인 낙상5.21→12.11%. 인원별 초기 장면/표본이 다르므로 순수 인원 효과나 통계적 우열을 확정하지 않는다.
- 두 실행1200step·CPU/GPU/JSON 재채점 일치·float32/64 차이0·finite·같은 쌍 분모/상한·동결 encoder/RMS·원본checkpoint/운영파일256개 보존·기존4학습 생존 확인. 8명은 초기 배치 검사에 시간이 더 걸렸지만 정책 rollout은 정상 완료됐다. [행동별 집계](output_etc/check/fusion_6_8_eval64_20261104/metrics.json), [BEFORE CSV](output_etc/check/fusion_6_8_eval64_20261104/before.csv), [검증](output_etc/check/fusion_6_8_eval64_20261104/verification.json).

### penalty_v2 7000 / fusion2000 및 동일2000 비교 평가

- GPU3·4·6에서 고정penalty_v2 7000·fusion2000·동일epoch 비교용penalty_v2 2000을각4명·512환경×3seed×1200step, 기존 random_scenario/near50·Plane Ever 규약으로 평가했다.
- 최신 전체 후행 동시/같은 쌍 선행 AT 평균%: penalty_v2 7000은31.24/90.11(6000대비−2.48/−1.42%p), fusion2000은29.41/90.65(1000대비+10.68/+4.09%p). penalty_v2 전체/ON_TOP은3seed 공통 하락, fusion후행3행동 모두3seed 공통 개선했다. 단독5행동 동일 가중치 평균94.84/94.29%.
- 동일2000epoch penalty_v2→fusion 후행SIT21.18→35.43·CLIMB14.51→40.87·ON_TOP13.90→22.42·전체15.58→29.41%, 각 쌍 선행92.70→91.86·76.71→87.73·78.20→91.36·80.88→90.65%. 세 행동 모두3seed에서fusion우세. SIT는선행 개선 없이 후행 개선이 관찰됐으나 단일학습run 비교로 기전의 인과 효과를 확정하지 않는다.
- 9회1200step 정상 완료·CPU/GPU/JSON 재채점 일치·float32/64 차이0·동일 초기 상태·같은 쌍 분모/상한·동결 encoder/RMS·checkpoint/운영파일256개 보존·기존4학습 생존 확인. [전체 추세](output_etc/check/penalty7k_fusion2k_eval_20261104/trends.csv), [seed별 변화](output_etc/check/penalty7k_fusion2k_eval_20261104/comparison.json), [검증](output_etc/check/penalty7k_fusion2k_eval_20261104/verification.json). 실제 대기 행동은이번에추가측정하지 않았다.

### penalty_v1 64환경 정지 거리 비교

- GPU4·최종6500·64환경·seed0·1200step을 직전v2/fusion과 같은 root XYZ≤0.3m/s·1초 연속 기준 및 같은 초기 장면에서 진단했다. 첫 정지 사람→AT목표 XY거리 중앙값 SIT0.34/CLIMB0.15/ON_TOP0.65m, ON_TOP payload→목표0.23m로 v2/fusion보다 중심에 가까웠다.
- armed 미해제1초 관찰 가능 SIT14/CLIMB14/ON_TOP24쌍 중 root정지9/13/14쌍, progress 포화반경 밖1초 정지는0/1/5쌍이다. root정지 중 발속력까지 유지한 쌍18/36. 거리/행동 동작 후보를 엄밀한 행동 분류로 해석하지 않는다.
- 1200step 정상 종료·finite·동일 초기 상태·AT Ever 해제/runtime 정지 집계 일치·가중치/원본checkpoint/운영파일254개 보존·기존4학습 생존 확인. [집계](output_etc/check/penalty_v1_stop64_20261104/metrics.json), [쌍별 거리](output_etc/check/penalty_v1_stop64_20261104/pairs.csv), [검증](output_etc/check/penalty_v1_stop64_20261104/verification.json).

### penalty_v2 / fusion 64환경 실제 정지 거리 진단

- GPU3·4에서 최신 고정 penalty_v2 6500 / fusion1500을각64환경·seed0·4명·8토큰·1200step, 학습 random_scenario/near50·일반 장면 낙상 종료로 진단했다. 선행 미완료·대기 armed 이후 root XYZ속력≤0.3m/s가30step 연속 유지된 첫 확인 시점의 XY거리를 측정했다. 정식 성공률 평가와 구분한다.
- 첫 정지 사람→AT목표 거리 중앙값 SIT/CLIMB/ON_TOP은 penalty_v2 0.93/1.85/2.10m, fusion1.41/1.92/2.26m. ON_TOP payload→목표는1.28/1.46m. 1초 관찰 가능50/47쌍 중42/43쌍 root정지, 이 첫 구간 발속력도≤0.3으로 유지된 쌍은14/13뿐이다. ON_TOP 정지 시 payload높이가 초기보다0.1m 넘게 상승하지 않은 사례18/19·25/25로 집기 전 정지가 많이 포함된다. +3m는 페널티 진입 반경이며 실제 정지 거리를 고정하지 않는다.
- 두 실행1200step 정상 종료·finite·동일 초기 상태·AT Ever 해제 일치·runtime 정지 집계 일치·가중치/원본checkpoint/운영파일254개 보존·기존4학습 생존 확인. [집계](output_etc/check/penalty_v2_fusion_stop64_20261104/metrics.json), [쌍별 거리/시점](output_etc/check/penalty_v2_fusion_stop64_20261104/pairs.csv), [검증](output_etc/check/penalty_v2_fusion_stop64_20261104/verification.json).

### fusion / penalty_v2 동일1000epoch 평가

- GPU3(penalty_v2)·4(fusion)에서각1000 고정 저장본을4명·512환경×3seed×1200step, 기존 random_scenario/near50·Plane Ever 규약으로 평가했다. 500epoch는 기존 결과를 재사용했다.
- 1000epoch 전체 후행 동시/같은 쌍 선행 AT 평균%는 penalty_v2 11.25/74.94 → fusion18.74/86.56. 후행 SIT11.99→20.77·CLIMB9.51→20.49·ON_TOP11.67→17.31, 각 쌍 선행81.72→92.79·66.95→81.33·75.72→86.41. 세 행동 모두3seed에서fusion이 우세했다. 단독5행동 동일 가중치 평균90.14→93.41%. 선행 개선도 동반되어 후행만의 인과 효과로 해석하지 않는다.
- 6회1200step 정상 종료·CPU/GPU/JSON 재채점 일치·float32/64 차이0·기존 초기 상태 동일·같은 쌍 분모/상한·동결 encoder/RMS·원본checkpoint 및 운영파일256개 보존·기존4학습 생존 확인. 정지 행동 추가 진단 없음. [동일epoch 비교](output_etc/check/fusion_penalty_v2_1000_eval_20261104/same_epoch.csv), [행동/거리별 결과](output_etc/check/fusion_penalty_v2_1000_eval_20261104/metrics.json), [검증](output_etc/check/fusion_penalty_v2_1000_eval_20261104/verification.json).

### penalty_v2 6000 평가

- GPU3에서 고정6000 저장본을4명·512환경×3seed×1200step, 기존 random_scenario/near50·Plane Ever 규약으로 평가했다. 전체 후행 동시/같은 쌍 선행 AT는33.73±1.45/91.53±0.25%, 직전5500 대비+3.68/+1.90%p다. 후행 SIT50.39·CLIMB36.90·ON_TOP26.36%, 각 쌍 선행96.34·91.58·89.78%. SIT/ON_TOP/전체는3seed 공통 개선, CLIMB는1seed 상승·2seed 소폭 하락했다. 단독5행동 동일 가중치 평균94.65→93.82%.
- 3회1200step 정상 종료·CPU/GPU/JSON 재채점 일치·float32/64 차이0·기존 초기 상태 동일·동결 encoder/RMS·원본checkpoint 및 운영파일 보존·기존4학습 생존 확인. 정지→해제 행동 추가 진단은 미실시. [추세 CSV](output_etc/check/penalty_v2_6000_eval_20261104/trends.csv), [행동/거리별 결과](output_etc/check/penalty_v2_6000_eval_20261104/metrics.json), [검증](output_etc/check/penalty_v2_6000_eval_20261104/verification.json).

### 실행 중인4종 평가 — 기본25k/asym24.5k/penalty_v2 5.5k/fusion0.5k

- GPU2·3·4·6에서 고정 snapshot4개를4명/8토큰·512환경×3seed×1200step Plane Ever로 평가했다. random_scenario/near50·기존 정식 평가 규약 유지, 중단된 wait/penalty_v1은 재실행하지 않았다. 운영 코드/설정 변경 없음.
- 전체 후행 동시/같은 쌍 선행 AT 평균%: 기본40.64/88.66, asym29.34/93.62, penalty_v2 30.04/89.64, fusion9.34/69.60. 직전 대비 기본−2.11%p(후행SIT는3seed 공통 하락), asym+2.92%p, penalty_v2+4.74%p. penalty_v2는후행3행동 모두3seed 공통 개선했다.
- 동일500epoch penalty_v2→fusion은 전체 후행10.09→9.34·같은 쌍 선행66.29→69.60, 단독SIT60.46±2.84→83.13±0.68이다. 초기 단독 성능 저하 완화는 관측됐으나 BEFORE 후행 개선은 아직 확인되지 않았다. 실제 정지→해제 행동의 추가 진단은 수행하지 않았다.
- 12회1200step·CPU/GPU/JSON 일치·float32/64 차이0·finite·과거와 동일 초기 상태·같은 쌍 분모/상한·encoder/RMS 동결·원본checkpoint/운영파일 보존·실행 중4학습 생존 확인. [행동별/거리별 수치](output_etc/check/active_four_eval_20261104/before_results.csv), [단독](output_etc/check/active_four_eval_20261104/solo_results.csv), [추세](output_etc/check/active_four_eval_20261104/trends.png), [seed별 변화](output_etc/check/active_four_eval_20261104/comparison.json), [검증](output_etc/check/active_four_eval_20261104/verification.json).

### penalty_v2 BEFORE relation fusion / zero background 실험 추가

- `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg` 환경/학습 YAML과 train/test/VNC/eval wrapper를 추가했다. 기존 penalty_v2에서 `human_relation_fusion: before_zerobg`와 실험 이름만 변경했다. +3m/1.0 속력 페널티·Ever 해제·가상 상자·거리/보상/Gate는 유지한다.
- 후행 query→선행 key의 BEFORE bias와 후행 attention 출력의 잔차 MLP에 같은64차원 embedding을 사용한다. embedding/MLP는4층 공유, projection은층/head별, Actor/Critic은별도다. SELF/NONE/미사용 COUPLED bias는0 고정, fusion은출력 projection 전에 후행에게만 적용하고 성공 후에도 역할 조건을 유지한다. 기본 경로/기존 checkpoint는 보존하고 새 fusion 계약을 분리했다.
- CPU 신규14개 포함131개 통과·3개 기존 skip. 별도 기존 테스트1개는 `checkpoints/Stage1PushFinetune_00021000.pth` 부재로 실패했다. 사용 중인26k 전이·초기 항등·gradient·단방향/두 쌍/셔플·Actor/Critic 가상 관측과 원본 packet·기존 penalty_v2 4500 strict 복원은 확인했다.
- GPU4·4명/1024환경/8토큰 `CHECK=1 MAX_ITERATIONS=2` 학습 exit0(기존 루프 저장 epoch3), Actor encoder 비트 단위 동결·양쪽 fusion/bias 학습·finite 확인. 새 eval wrapper로64환경×1seed×1200step 저장본 재실행 exit0. 기존 YAML/실험 wrapper 보존, 본학습 미시작. [검증](output_etc/check/before_relfusion_zerobg_20261104/verification.json), [학습](output_etc/check/before_relfusion_zerobg_20261104/train.log), [checkpoint 검사](output_etc/check/before_relfusion_zerobg_20261104/checkpoint_verification.json).

### penalty / wait_v2 / wait_v3 실제 정지·거리 비교

- GPU2·3·6에서 penalty6500/wait_v2 10500/wait_v3 7500 각512환경×3seed·일반 장면 낙상 종료로 진단했다. wait 대기 종료ON을 유지하고 반경 밖 정지도 별도 집계했다. penalty_v2 4500 직전3seed 결과를 재사용하고, wait_v2/v3는 같은 seed0의 대기 종료만OFF인 보조 실행을 추가했다(신규11회). 운영 코드/학습 변경 없음.
- 선행 미완료·진입 후1초 이상 관찰 집단의 반경 안1초 root 정지 SIT/CLIMB/ON_TOP: penalty132/235·229/303·327/662, 첫 정지 중심거리 중앙값0.36/0.12/0.19m. wait_v2는1/114·3/192·4/323, wait_v3는0/165·4/208·2/604에 그쳤다. wait 대기 종료 장면616/1536·926/1536. 기존penalty의 동작 후보 전 바깥 정지는9.8/7.9/18.4%, penalty_v2는66.7/39.7/55.0%였다. 기하 후보 검출이며 정확한 행동 분류는 아니다.
- 종료OFF seed0 첫 반경 안 정지 거리 SIT/CLIMB/ON_TOP: v2 0.78/0.17/0.20m, v3 0.31/0.22/0.25m. 종료를 제외해도 너무 접근한 뒤 멈추는 사례가 많다. ON/OFF 초기 상태는 동일하지만 분기 이후 공통 생존 궤적도 달라졌으므로 OFF는 별도 재실행 보조 진단으로 기록하고 프레임 단위 인과 비교로 해석하지 않았다.
- native 종료 판정·raw AT=Plane AT·release=raw AT Ever·초기 상태·finite·원본checkpoint/운영파일249개·기존6학습 보존 확인. [전체/행동별 정지·거리·직접 종료 거리·동일 집단 선행/후행](output_etc/check/wait_stop_comparison_20261104/report.md), [시계열 검증](output_etc/check/wait_stop_comparison_20261104/timing_verification.json), [보존/ON-OFF 차이](output_etc/check/wait_stop_comparison_20261104/verification.json).

### penalty_v2 4500 실제 대기→선행 완료→후행 수행 진단

- GPU2·3·6·512환경×3seed·일반 장면 낙상 종료/1200step에서 +3m/1.0 penalty_v2의 실제 armed/released·root XYZ 속력·발 높이/속력·payload·native/Plane 성공을 기록했다. 기존 정식 평가와 checkpoint·초기 상태 동일, 관측/물리/Gate/정책 변경 없음.
- 미완료 상태를1초 이상 관찰한1239쌍 중 root1초 정지973, 행동 후보 전에 progress 반경 밖 정지664. 선행 완료와 이후3초 생존을 관찰한 SIT159/CLIMB213/ON_TOP620쌍에서 연속 바깥 대기 유지77/49/245, 이어 재출발67/30/220, 이어 실제 동시 Plane 성공40/24/47이었다. 같은 대상 선행 AT는159/213/620 전부 성공. 조건부 진단이며 정식 전체 성공률과 구분한다.
- 선행 끝내 미완료230쌍 중179에서 조기 동작 후보 관측. SIT 높이 하강·CLIMB 높은 발 들기·ON_TOP 가상 배치 구역 진입에 기반한 기하 후보이며 행동 분류 정답은 아니다. 정지→해제→성공 사례는 있으나 일관된 대기는 미달. CLIMB 첫 root 정지 거리 중앙값0.57m로 너무 접근한 뒤 정지가 많다. [상세/같은 분모 순서표](output_etc/check/penalty_v2_wait_sequence_20261104/report.md), [성공 시계열](output_etc/check/penalty_v2_wait_sequence_20261104/ordered_success.png), [미완료 중 시도 후보](output_etc/check/penalty_v2_wait_sequence_20261104/premature_without_leader.png).
- 3seed 실제 종료 판정·초기 상태 일치·native AT와 Plane AT 일치·해제=raw AT Ever/영구 유지·finite·가중치/원본checkpoint/운영파일249개·기존6학습 보존 확인. [검증](output_etc/check/penalty_v2_wait_sequence_20261104/verification.json), [추가 시계열 검증](output_etc/check/penalty_v2_wait_sequence_20261104/sequence_details.json).

### 최신6종 후속 평가 — 기본/asym23.5k 및 wait/penalty 추세

- GPU2·3·6만 사용해 기본23500/asym23500/wait_v2 10500/wait_v3 7500/penalty6500/penalty_v2 4500 고정 snapshot을 4명/8토큰·512환경×3seed×1200step Plane Ever로 평가했다. random_scenario/near50·대기 종료 제외 규약과 이전 평가의 환경 설정·초기 상태를 유지했다. 운영 코드/설정 변경 없음.
- 전체 후행 동시/같은 쌍 선행 AT 평균%: 기본42.75/88.66, asym26.42/92.96, v2 23.62/88.55, v3 15.49/82.68, penalty17.59/83.67, penalty_v2 25.31/87.85. 기본/asym 직전 하락 후 반등, v2 정체, v3 소폭 개선, 기존penalty 소폭 하락. 새penalty_v2는500→4500에서 단독SIT60.46→92.60으로 회복했으나 중간 checkpoint 추세·실제 정지 행동은 이번에 검증하지 않았다.
- 18회1200step·CPU/GPU/JSON 일치·float32/64 판정 차이0·finite·동일 초기 상태·같은 쌍 분모/상한·거리 분할 합계·동결 encoder/RMS·checkpoint 및 운영파일249개 보존·기존6학습 생존 확인. [행동별/거리별/단독/최근 추세](output_etc/check/needed_eval_followup_20261104/report_plane.md), [그래프](output_etc/check/needed_eval_followup_20261104/trends.png), [seed별 변화](output_etc/check/needed_eval_followup_20261104/comparison.json), [검증](output_etc/check/needed_eval_followup_20261104/verification.json).

### 필요한 최신6종 평가 — GPU0·1 제외

- GPU2·3·6에서 기본20000/asym19500/wait_v2 8000/wait_v3 4500/penalty3500/penalty_v2 500 고정 snapshot을4명/8토큰·512환경×3seed×1200step Plane Ever로 평가했다. random_scenario/near50와 wait 대기 종료 제외 규약 유지. 중간 checkpoint 추가 평가는 생략하고 기존 최근2개 및500epoch 대조 결과를 재사용했다.
- 전체 후행 동시/같은 쌍 선행 AT 평균%: 기본38.93/88.77, asym24.11/93.36, v2 23.86/86.92, v3 13.10/80.66, penalty18.47/83.66, 새penalty_v2 10.09/66.29. 최근 기본/asym 후행 하락·v3 정체, v2/기존penalty 개선. 같은500에서 기존/new penalty 전체9.18/71.88→10.09/66.29이나 단독SIT85.07→60.46으로 저하. 실제 정지 행동의 추가 진단은 수행하지 않았다. [전체·거리별 같은 쌍 선행/후행·단독·추세·std](output_etc/check/needed_eval_20261104/report_plane.md), [그래프](output_etc/check/needed_eval_20261104/trends.png).
- 앞서 추가한 penalty_v2가 `evaluate.py`의5개 실험 분류 목록에서 누락되어 task/motion/PUSH 경로 등록만 보완했다. 평가 CPU18개 통과. 초기 GPU3/6 MPS 상속 설정 불일치는 실행 전 차단됐고, 평가 자식 환경의 상속 pipe를 해제해 각 GPU 검증 daemon으로 재시도했다. 실제 평가 GPU UUID→2/3/6 확인.
- 최종18회1200step·CPU/GPU/JSON 재채점·float32/64 판정 일치·finite·동일 초기 상태·같은 쌍 분모/상한·거리 분할 합계·encoder/RMS 동결·checkpoint 및 등록 보완 후 운영파일249개 해시 보존 확인. 기존6개 학습 생존. [검증](output_etc/check/needed_eval_20261104/verification.json).

## 2026-11-03

### wait_penalty_v2 — 진입 반경 +3m / 속력 가중치1.0

- 기존 `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty`를 보존하고 `_v2` 환경/학습 YAML과 train/test/VNC/eval wrapper를 추가했다. radius_buffer만2→3m, coefficient만0.6→1.0으로 변경했다. 진입 Ever·선행 성공 Ever 해제, root XYZ 속력, 기존 Progress/current State·Success Gate·AT 높이 제한 해제는 유지하며 대기 조기 종료는 없다. 실험/output 이름을 분리하고 registry/launcher/config/structure를 연결했다.
- CPU103개 통과 및 새 설정의3행동 진입 경계·XYZ 감점·이탈 후 유지·성공 후 해제·부분 reset·기존 checkpoint 계약 분리 확인. GPU0·4명/1024환경/8토큰 `CHECK=1` 짧은 학습 exit0, Actor/Critic 실행·checkpoint 저장·저장된+3/1.0 설정 확인. 기존 penalty YAML/스크립트6개 해시 보존. 본학습은 시작하지 않았다. [검증](output_etc/check/wait_penalty_v2_config_20261103/verification.json), [학습 로그](output_etc/check/wait_penalty_v2_config_20261103/train.log).

### wait_v3 3500 가상 상자 거리 개입 CLIMB 토이

- GPU0·32환경·seed2에서 선택 후행 Alias GTA XY만 root 기준3.5→0.2m로 당겼다가 멀리 보내는12초 주기3회와 동일 초기 상태3.5m 대조군을 실행했다. 대기 종료만 끄고 원래 낙상/시간 종료를 유지했다. 운영 코드/weight/물리 상태 직접 수정 없음.
- 발 중심 Z>0.45m 3step 기준 시도 후보 시작 거리 중앙값1/2/3차0.73/0.99/0.99m, 2회 이상 검출28/32 대 대조군0/32. 선행 AT가 끝내 성공하지 않은 env6에서도0.75/0.66/0.88m에서 높은 발 올리기를 반복했고 실제 영상으로 확인했다. 고정 행동 임계값·정지 성공·학습 성공률로 해석하지 않는다. [결과/한계](output_etc/check/virtual_climb_trigger_20261103/report.md), [거리·발/골반 높이](output_etc/check/virtual_climb_trigger_20261103/distance_and_height.png), [0.5배속 영상2개 ZIP](output_etc/check/virtual_climb_trigger_20261103/virtual_climb_toy_slow.zip).
- 입력 변경 범위·초기 상태 동일·계측 위치/높이 및 선행 성공의 렌더 재현·MP4 decode/ZIP CRC·운영파일241개/checkpoint/기존5학습 보존 확인. 접촉 힘/native done 일부는 렌더 유무에 따라 달라 원본 trace를 집계했으며 대표 env6 생존은 동일하다. [검증](output_etc/check/virtual_climb_trigger_20261103/verification.json).

### source5/asym goalroute 전체 평가 추세 재집계

- 신규 시뮬레이션 없이 현재 goal5_goalroute의0.5/1/1.5/6/7/9/14.5/15.5/17/18k, 두 모델20checkpoint를 통합했다. BEFORE 행동별·거리별 후행/같은 쌍 선행 AT와 단독5행동·표준편차를 [전체 표](output_etc/check/source5_asym_full_trends_20261103/report.md), [BEFORE 그래프](output_etc/check/source5_asym_full_trends_20261103/before_trends.png), [단독 그래프](output_etc/check/source5_asym_full_trends_20261103/solo_trends.png)에 기록했다. 다른 거리 설정·8명·AT9 평가는 섞지 않았다.
- seed별 동일 부분집합 분모·후행 성공 상한·평균/std 재계산 일치를 확인했다. 장기적으로 기본3행동은 개선됐으나 최근 정체, asym은9k 이후 CLIMB 개선이 뚜렷하고 SIT/ON_TOP 개선은 작거나 변동이 크다. 운영 코드/학습 변경 없음.

### wait_v3 3000 +2m 실제 대기 조기 종료 영상

- GPU0·3k 고정 checkpoint·seed2·4명/8물체·행동별 preset으로 SIT/CLIMB/ON_TOP 해당 후행의 실제 대기 종료를 촬영했다. 모두 r_box+2m 반경 밖 접근부터 누적 XYZ 이동 위반1초 종료까지 연속이며 실제 길이3.63/5.27/2.70초. ON_TOP은 자연 실행에서 사례를 먼저 찾아 같은 종료 frame/속력이 재현된 장면을 촬영했다. 처음 고른 부적절한 ON_TOP 촬영만 중단·교체했다.
- 1280×800·0.5배속·시작2초/끝4초 정지, 사람/목표/반경·속력·누적 위반 자막을 표시했다. 초기 미갱신 렌더1프레임만 제외, 물리/정책 상태 조작 없음. [영상 폴더](output_etc/check/wait_v3_3000_videos_20261103/slow_videos/), [ZIP 약4.6MB](output_etc/check/wait_v3_3000_videos_20261103/wait_v3_3000_early_termination_slow.zip).
- 3건 모두 해당 후행 waiting 종료·선행 미해제·누적1초 확인. 전체 MP4 decode·ZIP CRC·시작/종료 화면 확인·checkpoint 및 운영파일241개 해시·기존5개 학습 보존. 선택 사례이며 성능 통계가 아니다. [검증](output_etc/check/wait_v3_3000_videos_20261103/verification.json).

### wait_v2/v3/penalty 동일 epoch 비교 — v2는1k 간격

- 사용자 범위 축소에 따라 v2는1000/2000/3000만, v3는500~3000의500간격, penalty는요청 시점 생성된500~2000의500간격을 비교했다. 진행 중이던 평가 전용 v2 500을 중단하고1500/2500 대기 작업은 취소했으며 본학습은 보존했다. v2 2000/3000·v3 3000·penalty1000만 GPU0·512환경×3seed×1200step 신규 평가12회, 기존9checkpoint는 trace 재집계로 재사용했다.
- 같은3k 전체 후행 동시/같은 쌍 선행 AT 평균%는 v2 15.95/80.91·v3 13.73/77.68. 같은2k는 v2 14.64/77.32·v3 11.42/77.00·penalty15.70/79.04. v2는3k ON_TOP이 높고 CLIMB는v3와 비슷하며, penalty는2k SIT가 높다. [전체/거리별 같은 쌍 선행·후행 및 단독 표](output_etc/check/matched_wait_epochs_20261103/report_plane.md), [같은 epoch 그래프](output_etc/check/matched_wait_epochs_20261103/matched_trends.png).
- 총13checkpoint/39rollout의 CPU/GPU/JSON 일치·finite·float32/64 판정 차이0·동일 초기 상태·seed별 동일 부분집합 분모 확인. 신규 checkpoint/운영파일243개 해시·encoder/RMS 동결·기존5개 학습 보존. wait 조기 종료 제외 규약 유지, 실제 대기 행동 추가 진단 없음. [검증](output_etc/check/matched_wait_epochs_20261103/verification.json).

### source5/asym18000·wait_v2 7000·wait_v3 2500·penalty2000 추세 평가

- 비어 있던 GPU0에서 고정5개 checkpoint·4명/8토큰·512환경×3seed×1200step Plane Ever 평가15회를 완료했다. 학습 random_scenario/near50 및 기존 정식 규약(wait 대기 종료 제외)을 유지했다. 전체 후행 동시/같은 쌍 선행 AT 평균%는 source5 40.33/86.68·asym26.48/90.59·v2 22.34/85.17·v3 13.85/76.66·penalty15.70/79.04.
- 직전 대비 기본/asym 전체 하락, v2/v3/penalty 반등. asym CLIMB는 개선됐지만 SIT는 하락했다. 단독 SIT는 v3 88.15→96.13·penalty80.02→88.02로 회복했다. 학습량·AT 추가 높이 조건 차이를 명시하고 행동별/거리별 같은 쌍 선행·단독·과거 추세·표본std를 [보고서](output_etc/check/next_trends_20261103/report_plane.md), [그래프](output_etc/check/next_trends_20261103/trends.png)에 기록했다. 실제 대기 행동의 추가 진단은 수행하지 않았다.
- 15회1200step·CPU/GPU/JSON 재채점 일치·float32/64 판정 차이0·finite·과거 및 모델 간 초기 상태 동일·같은 쌍 상한·거리 분할 합계 확인. encoder/RMS 동결·checkpoint/운영파일243개 해시·기존5개 학습 보존. 운영 코드/설정 변경 없음. [검증](output_etc/check/next_trends_20261103/verification.json).

### source5/asym17000·wait_v2 6500·wait_v3 2000 평가 및 추세

- GPU4·4명/8토큰·512환경×3seed×1200step Plane Ever, 학습 random_scenario/near50 유지·wait 대기 종료 제외. 새4개 checkpoint를 고정해12회 평가했다. 전체 후행 동시/같은 쌍 선행 AT 평균%는 source5 42.77/87.42·asym29.21/92.19·v2 20.35/84.10·v3 11.42/77.00. 직전 대비 기본/asym 상승·v2 유지·v3 하락이며 v3 단독 SIT95.03→88.15. penalty1500은 기존 평가를 재사용했다.
- 행동별 같은 쌍 선행/후행·거리 부분집합·단독·checkpoint 추세와 표본std를 [통합 보고서](output_etc/check/combined_trends_20261103/report_plane.md), [추세 그래프](output_etc/check/combined_trends_20261103/trends.png)에 기록했다. 모델별 학습량 및 AT 추가 높이 조건 차이를 명시했다. 과거1500 대기 진단은2000 행동 검증으로 간주하지 않는다.
- 12회 완료·CPU/GPU/JSON 재채점 일치·finite·초기 상태 일치·같은 쌍 상한·거리 분할 합계 확인. checkpoint/운영파일243개 해시 보존·기존5개 학습 생존. 운영 코드/설정 변경 없음. [통합 검증](output_etc/check/combined_trends_20261103/verification.json), [v3 검증](output_etc/check/wait_v3_eval2000_20261103/verification.json).

### penalty1500 정식 평가·wait_v3 동일1500 대기/해제 비교

- GPU4·512환경×3seed×1200step Plane Ever. penalty1500 후행 동시/같은 쌍 선행 AT 평균%는 SIT14.17/87.79·CLIMB8.49/69.29·ON_TOP14.86/80.12, 전체13.22/79.24. penalty500 대비 SIT/ON_TOP 상승·CLIMB 정체, 단독 SIT85.07→80.02. 동일1500 wait_v3와 전체 성능은 비슷하고 행동별 차이가 있다. 기본15500/wait_v2 6000은 학습량이 다른 참고값으로 분리했다. [정식 비교·판단 요약](output_etc/check/penalty_eval1500_20261103/report_plane.md).
- 같은 초기512환경·seed0으로 v3 종료ON/OFF 및 penalty 일반 장면 종료 진단. 진입 후1초 root 정지2/402·184/450·221/461, 동일 진입 집단 선행 Plane Ever9/402·261/450·314/461. v3 ON은364/512장면 대기 종료·평균12.05초(OFF32.77, penalty34.85). OFF/penalty 첫 정지 목표 거리 중앙값 약0.20m, 정지 사례 대부분 이미 누적 이동1초 초과. CLIMB 해제 직전 정지·이후3초 관찰 가능 집단의 이동 재개는 v3 OFF4/20·penalty1/18, 해당 집단 장면 끝까지 후행 동시/선행 Ever는1/20·20/20 및0/18·18/18. 1seed 조건부 표본이며 root 정지가 스킬 시작 전 대기를 보장하지 않는다. [대기 상세](output_etc/check/penalty_eval1500_20261103/report_wait.md).
- 정식3회 CPU/GPU/JSON 일치·float32/64 차이0·finite·penalty500/v3 1500 초기 상태 일치. 진단3건 실제 종료 판정·가중치 보존·정식seed0 포함 초기 상태 정확 일치. checkpoint/운영파일243개 해시·기존5개 학습 보존. 운영 코드/설정 변경 없음. [정식 검증](output_etc/check/penalty_eval1500_20261103/verification.json), [진단 검증](output_etc/check/penalty_eval1500_20261103/native_verification.json).

### wait_v3 1500 정식 평가·500/1000 추세

- GPU4·4명/8토큰·512환경×seed0/1/2×1200step Plane Ever, 학습 random_scenario/near50 유지·정식 규약상 대기 종료 제외. 고정1500 snapshot의 후행 동시/같은 쌍 선행 AT 평균%는 SIT11.20/86.82·CLIMB12.90/62.89·ON_TOP13.48/76.59, 전체12.86/75.55였다. 1000 대비 전체12.43→12.86, CLIMB13.34→12.90으로500의26.09에는 회복하지 못했다. 단독·동일 쌍 거리별 선행/후행·3checkpoint 추세와std는 [보고서](output_etc/check/wait_v3_eval1500_20261103/report_plane.md)에 기록했다.
- 3회1200step·CPU/GPU/JSON 일치·float32/64 누적 판정 차이0·finite·500/1000 초기 상태 정확 일치·같은 쌍 상한·거리 분할 합계 확인. encoder/RMS 동결·checkpoint/운영파일243개 해시 보존·기존5개 학습 생존. 운영 코드/설정 변경 없음. [검증](output_etc/check/wait_v3_eval1500_20261103/verification.json).

### wait_v2 6000 정식 평가·5500 대비

- GPU4·4명/8토큰·512환경×seed0/1/2×1200step Plane Ever, 학습 random_scenario/near50 유지 및 정식 규약상 대기 종료 제외. 최신 번호6000 snapshot의 후행 동시/같은 쌍 선행 AT 평균%는 SIT22.81/91.93·CLIMB17.87/81.64·ON_TOP20.40/84.61, 전체20.31/85.42였다. 5500 대비 전체 후행17.99→20.31, 세 행동 평균 모두 상승. 단독·거리 부분집합·표본std는 [보고서](output_etc/check/wait_v2_eval6000_20261103/report_plane.md)에 기록했다.
- 3회1200step·CPU/GPU/JSON 일치·float32/64 누적 판정 차이0·finite·5500 초기 상태 정확 일치·같은 쌍 상한·거리 분할 합계 확인. encoder/RMS 동결·checkpoint/운영파일243개 해시 보존·기존5개 학습 생존. 운영 코드/설정 변경 없음. [검증](output_etc/check/wait_v2_eval6000_20261103/verification.json).

### wait_v3 1000 정식 평가·500 대비

- GPU4·4명/8토큰·512환경×seed0/1/2×1200step Plane Ever, 학습 random_scenario/near50 유지 및 정식 규약상 대기 조기 종료 제외. 고정1000 snapshot으로 후행 동시/같은 쌍 선행 AT 평균%는 SIT10.42/80.38·CLIMB13.34/63.48·ON_TOP12.80/75.31, 전체12.43/73.63이었다. 500 대비 후행 CLIMB26.09→13.34, 전체15.26→12.43. 단독·거리 부분집합·표본std는 [보고서](output_etc/check/wait_v3_eval1000_20261103/report_plane.md)에 기록했다.
- 3회1200step·CPU/GPU/JSON 일치·float32/64 누적 판정 차이0·finite·500 초기 상태 정확 일치·같은 쌍 상한·거리 분할 합계 확인. encoder/RMS 동결·checkpoint/운영파일 해시 보존·기존5개 학습 생존. 정식 성공률이며 대기 행동 학습 여부를 판정하지 않는다. [검증](output_etc/check/wait_v3_eval1000_20261103/verification.json).

### wait_v3 / penalty 500 실제 장면 0.5배속 묶음

- 직전 평가와 같은 두 모델500 snapshot으로 GPU4·seed2·행동별 preset·4명/8물체에서 SIT/CLIMB/ON_TOP을 촬영했다. wait_v3 대기 종료 ON, penalty 개인 감점 유지. 초기부터 종료/최대20초까지 실제 상태를 기록하고 반경·속력·종료 누적/감점량을 표시했다. ON_TOP 다른 쌍 종료 사례를 포함한7개 MP4를 [한 폴더](output_etc/check/wait_v3_penalty_videos_20261103/slow_videos/) 및 [ZIP](output_etc/check/wait_v3_penalty_videos_20261103/wait_v3_penalty_500_slow_videos.zip)에 모았다.
- 1280×800·0.5배속·시작2초/끝4초 정지 화면, reset 직후 미갱신 렌더1프레임만 제외. MP4 전체 decode·ZIP CRC·checkpoint 해시 확인, 기존5개 학습 생존. 촬영 프로세스 중단 후 미완료 penalty2건만 재실행했다. 선택 장면은 성능 통계가 아니며 episode/종료 사유는 README와 verification.json에 기록했다. 운영 코드/설정 변경 없음.

### 실행 중5개 최신 저장본 GPU4 Plane Ever 평가

- 요청 시점 기본15500·asym15500(15k resume run)·wait_v2 5500·wait_v3 500을 고정하고, 아직 저장본이 없던 penalty는 첫500 생성 후 포함했다. 모든 입력을 검증 경로에 snapshot으로 보존해 GPU4·4명/8토큰·512환경×seed0/1/2×1200step로 평가했다. 학습 random_scenario/near50 유지, wait_v2/v3 대기 종료는 정식 규약상 제외했다.
- 전체 후행 동시/같은 쌍 선행 AT Ever 평균%는 기본40.59/86.86·asym22.43/89.24·wait_v2 17.99/84.77·wait_v3 15.26/74.37·penalty9.18/71.88이었다. 행동별·단독·후행/선행 근거리·직전 평가 추세와 표본std는 [보고서](output_etc/check/latest_five_eval_20261103/report_plane.md)에 기록했다. 모델epoch 및 기존 AT 추가1mm 학습 조건 차이를 명시했다.
- 15회 모두1200step·CPU/GPU/JSON 재채점 일치·float32/64 누적 판정 차이0·finite·초기 상태 동일·같은 쌍 상한·거리 분할 합계 검증. encoder73tensor/입력 RMS 동결·checkpoint 해시·운영파일240개 보존·기존5개 학습 생존 확인. 평가 중 요청받은 viewer 토글3개 파일 변경은 별도 해시로 추적했으며 정식평가 동작에는 영향 없다. [검증](output_etc/check/latest_five_eval_20261103/verification.json).

### BEFORE viewer 대기 조기 종료 선택 옵션

- 요청에 따라 wait/wait_v2/wait_v3 test/VNC의 `BEFORE_WAIT_TERMINATION=0`을 추가했다(기본1). `--disable_before_wait_termination`으로 추론 중 대기 위반 종료만 건너뛰며, 낙상/시간 종료·checkpoint 설정·학습·penalty 보상은 유지한다. 학습 직접 CLI 사용은 거부한다. USAGE/config에 명령을 기록했다.
- GPU4·wait_v3 500에서8환경16step 및 상태 주입 ON/OFF 검사로 기존 대기 종료·옵션OFF 시 낙상/시간 종료 버퍼 보존·부분 reset·정식평가 우회를 확인했다. 실제 test wrapper도1환경8step 정상 완료했고 플래그 전달/학습 거부·shell 문법을 확인했다. [물리 검증](output_etc/check/viewer_wait_toggle_20261103/physics/physics_validation.json), [wrapper 로그](output_etc/check/viewer_wait_toggle_20261103/wrapper.log).
- 진행 중 최신5개 정식평가는 원래 대기 종료를 우회하므로 이 추론 옵션 추가는 채점 동작을 바꾸지 않는다. 해당3개 파일의 승인된 변경 해시는 평가 검증에서 별도 추적한다.

### Virtual wait_penalty: 개인 연속 XYZ 속도 감점

- virtual_goal5_goalroute 기반 `_wait_penalty` env/train·train/test/VNC/eval·registry·평가 목록을 추가했다. AT 추가1mm 높이 제한 해제(기존3D20cm 유지), 목표 XY r_box+2m 최초 진입 후 후행 개인 task 보상에서 `0.6×root XYZ 속력`을 매 step 차감한다. 선행 raw AT 최초 성공 시 같은 step부터 영구 해제하며 기존 state/success current gate·progress·HOLDING·AMP는 유지한다. 대기 위반 종료는 없다.
- 기존 BeforeWait의 진입/해제 기록을 공유하고 새 감점을 별도 reward term에 기록했다. 기존 설정/계약은 보존하고 새 계약/출력은 분리했다. config/USAGE/structure/상세 문서에 실행법과 의미를 추가했다.
- CPU183통과·3skip. GPU4·4명/1024환경 짧은 학습2epoch/262144frame 완료·finite·actor encoder73tensor/입력 RMS 동결, wait_penalty 로그 확인. 8환경×6preset 각16step 및 상태 주입 검사에서 개인 감점량·타 보상 항 보존·종료 버퍼 불변·진입/성공 기록·부분 reset·Virtual 관측 확인. 기존 YAML46개 해시 보존. 본학습은 시작하지 않았다. [검증 보고서](output_etc/check/before_wait_penalty_20261103/report.md).

### wait_v3: 대기 진입 반경 +2m

- 사용자 요청으로 wait_v2를 보존하고 `_wait_v3` env/train YAML·train/test/VNC/eval wrapper·registry·실행/평가 목록을 연결했다. 동작 설정 차이는 `waiting_termination.radius_buffer: 1.0 → 2.0` 하나이며 progress·XYZ 속력0.3m/s·유예0초·누적1초 종료·선행 성공 시 영구 해제·AT 기준을 유지했다. 기본 Stage1 26k 전이로 별도 새 학습한다. config/USAGE/structure에 경로와 명령을 추가했다.
- 기존 대기 CPU87개 통과. SIT/CLIMB/ON_TOP별 +2m 경계 안팎 및 +1.5m 진입·30step 누적 종료·선행 성공 즉시/영구 해제9사례, v2 설정/스크립트 해시 보존·반경 외 YAML 동일·checkpoint 계약 분리·shell 문법 통과. GPU4에서4명·1024환경 짧은 학습2epoch/262144frame 정상 종료·저장, 실제 저장 설정+2m·모델 finite 확인. 본학습 시작/기존 프로세스 종료 없음. [설정 검증](output_etc/check/before_wait_v3_20261103/config_check.json), [실행 검증](output_etc/check/before_wait_v3_20261103/training_check.json).

### wait_v2 4500 행동 전 정지 진단·실제 장면

- GPU4·512환경·seed0 첫 episode에서 일반 낙상/대기 종료를 유지했다. 선행 성공 전 반경 진입 후행298건 중 반경 안 root XYZ≤0.3m/s 1초 연속 정지는 SIT0/65·CLIMB2/82·ON_TOP5/151이었다. 대기 위반 종료267/512, 평균18.91초. 같은 쌍 선행 AT Ever를 진입 집단/전체별 함께 기록했다.
- 실제 선택 장면에서 CLIMB·ON_TOP은 반경 진입 후 정지하지 않고 발 들기/상자 내리기 동작을 시작한 뒤 누적 위반1초로 종료됐다. 세 영상 모두 선행 AT 성공 전이며 SIT는 계속 접근하다 종료됐다. 0.5배속 접근부터 종료까지 영상 및 [보고서](output_etc/check/wait4500_stop_20261103/report.md). 정지 지표만으로 행동 전 대기 성공을 뜻하지 않으며 정식평가와 구분한다.
- 1k 진단과 초기 상태 정확 일치·실제 종료 판정 일치·checkpoint/운영파일229개 해시 보존·기존4개 학습 생존 확인. 운영 코드/설정 변경 없음.

### 기본/virtual/asym/wait 최신4개 GPU4 Plane Ever 평가

- 요청 시점 최신 저장본 source5 14500·virtual13500(재개 run)·asym14500·wait_v2 4500을 GPU4에서 각512환경×seed0/1/2×1200step·4명/8토큰으로 평가했다. 사용자 명칭에 맞춰 이번 보고서의 wait는 wait_v2이며 기존 wait 실험과 파일명은 보존했다. 정식평가에서는 대기 종료를 제외했다.
- BEFORE 후행 SIT/CLIMB/ON_TOP 평균%는 기본60.87/53.69/28.09, virtual34.87/7.14/20.08, asym16.14/40.94/17.96, wait23.99/23.52/20.42였다. 전체 후행/같은 쌍 선행AT는40.80/86.75·20.13/83.51·22.89/91.49·21.85/83.31%. 각 행동·전체/후행근거리/선행근거리마다 같은 쌍 AT Ever 및std를 함께 기록했다. 단독5행동·직전 평가 추세도 [보고서](output_etc/check/latest_four_eval_20261103/report_plane.md)에 정리했다.
- 12회 모두1200step·CPU/GPU/JSON 재채점 일치·float32/64 누적 차이0·finite·seed별 초기 상태 동일·같은 쌍 AT 상한·거리 분할 합계 통과. 각 checkpoint의 encoder73tensor/입력 RMS 동결, checkpoint 및 운영파일229개 해시 보존, 기존4개 학습 프로세스 생존을 확인했다. 운영 코드/설정 변경 없음. [검증](output_etc/check/latest_four_eval_20261103/verification.json).

### wait_v2 1000 GPU4 정식 평가·실제 대기 종료 진단

- 사용자 요청으로1k를 GPU4에서4명·8토큰·512환경×seed0/1/2×1200step Plane Ever 평가했다. 정식평가에서 대기 종료는 제외한다. 전체 BEFORE 후행 동시 Ever/같은 쌍 선행 AT Ever는 SIT12.91±0.82/82.31±3.74%, CLIMB21.52±1.44/60.72±4.19%, ON_TOP13.78±1.00/74.26±4.09%였다. 전체15.39±0.94/72.78±4.11%, 기존wait1k 후행9.17% 대비 높았다. 단독 SIT91.55·CLIMB98.59·AT85.38·ON_TOP86.41·PUSH95.65%. 전체/후행근거리/선행근거리·동일1k 비교는 [정식 보고서](output_etc/check/wait_v2_eval1000_20261103/report_plane.md).
- 같은1k·seed0·512환경 첫 episode로 일반 장면 낙상 종료를 유지하고 대기 종료 ON/OFF를 비교했다. 초기 상태 정확 일치. ON은297/512장면이 대기 위반 포함 종료, 평균16.67초(OFF31.99초). 진입327후행 중 AT 성공 전 반경 안에서 root XYZ≤0.3m/s로1초 연속 정지1명. 위반9432프레임 중 XY만으로는 위반하지 않고 XYZ로만 추가된 경우117(1.24%). 같은 쌍 선행 AT Ever는 ON26.72%/OFF62.07%로 종료에 따른 수행 기회 감소를 확인했다. 이는1seed 진단이며 정식3seed 성능표와 구분한다. [대기 보고서](output_etc/check/wait_v2_eval1000_20261103/report_native.md).
- 정식 wrapper 실행 중 `evaluate.py`의 wait/wait_v2 등록 누락으로 task 선택이 실패했다. 반복된 기존 실험 목록5곳에 두 이름만 추가해 task·모션·PUSH 집계를 연결했으며 재실행3seed 완료·CPU18테스트 통과. 학습 코드/보상/종료 설정은 수정하지 않았다.
- CPU float64/GPU/JSON 채점 일치·float32/64 누적 차이0·1200step/finite·기존virtual과seed별 초기 상태 동일·encoder73tensor/입력 RMS 동결·checkpoint/운영파일 해시 보존·기존4개 학습 생존을 확인했다. [검증](output_etc/check/wait_v2_eval1000_20261103/verification.json).

## 2026-11-02

### Virtual wait_v2: 유예 제거·누적 XYZ 위반·AT 높이 제한 해제

- 기존 wait를 보존하고 `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_v2` env/train·train/test/VNC/eval·registry·GPU4 명령을 추가했다. XY 대기 반경을 r_box+1m로 늘리고 유예0초·root XYZ 속력0.3m/s 초과 누적1초 종료를 적용했다. 정지/반경 이탈은 누적값을 지우지 않으며 선행 AT 성공은 즉시 에피소드 끝까지 해제한다. Progress는 그대로다.
- 새 설정의 단독/선행/PUSH AT는3D20cm를 공통 사용한다. `at_z_tolerance=.2`로 추가1mm 제약을 없앴고 ON_TOP 자체 높이 조건은 유지했다. 기존 성공 함수를 재사용하며 AT 설정을 checkpoint 계약에 반영했다. 기존 설정은 XY·유예·연속 모드를 그대로 사용한다.
- CPU174통과·1skip. GPU4에서4명·1024환경 짧은 학습2epoch/262144frame 완료, Stage1 26k165tensor 전이·encoder73tensor/입력 RMS 동결·head6tensor 갱신·finite 확인. 저장본8환경×6preset 실제16step 및 reset 주입 검사에서 수직 이동/누적 종료·AT 성공 즉시/영구 해제·부분 reset·독립 행동·정식평가 bypass 통과. 기존 YAML 해시 및4개 학습 프로세스를 보존했다. [검증](output_etc/check/before_wait_v2_20261102/report.md).

### wait1500 실제 대기 종료 장면 촬영

- 앞부분 부족·카메라 혼잡에 대한 사용자 피드백으로 같은 GPU5·wait1500에서 한 후행의 에피소드 시작부터 연속 재촬영했다. 선행/후행 색 구분·목표/대기 반경의 설명용 표시·추적 카메라를 적용했으며 영상은0.5배속이다. 실제 시간 SIT3.97초·CLIMB12.30초 후 대기 종료, ON_TOP15초 동안 대기 종료 없음을 기록했다. 세 장면 모두 반경 진입 전부터 시작하고 에피소드가 이어짐·checkpoint/기존4개 학습 보존을 확인했다. [연속 영상 검증](output_etc/check/wait_capture_context_20261102/verification.json).
- 사용자 요청으로 GPU5에서 wait1500을 행동별4명·8환경·8토큰·seed2 deterministic 일반 실행으로 촬영했다. 상태/속도 주입 없이 실제 카메라 화면에 거리·root XY 속도·유예/연속 위반 시간을 표시했다. SIT·CLIMB는 연속 위반1초와 장면 reset을 함께 확인했고, ON_TOP은900step 동안 대기 종료가 없었으며 최대 연속 위반0.90초였다. [영상·판정 기록](output_etc/check/wait_capture_20261102/verification.json).
- 촬영기만 output_etc/check 아래 추가했으며 카메라 월드 좌표를 보정했다. 최초 촬영의 시뮬레이터 정리 단계 오류 후 재촬영을 정상 완료했다. 체크포인트 해시와 기존4개 학습 PID 생존을 확인했으며 운영 코드·대기 조건은 변경하지 않았다.

### Virtual wait1500 GPU5 Plane Ever 평가

- 사용자 요청으로 GPU5에서 wait1500을4명·8토큰·512환경×seed0·1·2×1200step·deterministic 첫 episode로 평가했다. 정식 규약상 대기 위반 종료는 제외하고 기존 개인 낙상을 적용했다. wait500/1000·virtual1500과 비교하고 전체/후행 경로 짧음/선행 경로 짧음 모두 행동별 동일 쌍 선행 AT를 함께 집계했다.
- 전체 후행 동시 Plane Ever/같은 쌍 선행 AT는 SIT11.67/86.98%, CLIMB8.45/63.22%, ON_TOP10.82/72.89%, 전체10.48±0.40/73.58±1.38%였다. 후행 경로가 짧은 집단은11.62/86.07·4.28/62.38·10.18/69.13%, 선행 경로가 짧은 집단은12.47/89.80·25.10/66.49·13.19/86.74%였다. 단독 SIT90.40·AT92.39·ON_TOP93.00%였다. [추세·virtual1500·거리별 표](output_etc/check/wait_eval1500_20261102/report_plane.md).
- 3회 모두1200step·CPU float64/GPU/JSON 일치·trace finite·float32/64 누적 차이0, 모든seed 기존 virtual과 초기 상태 정확 일치, 같은 쌍 선행 AT 상한·거리 분할 합 재현·encoder73tensor/입력 RMS 동결·checkpoint/운영 파일223개 해시 보존을 통과했다. 기존4개 학습 PID도 유지했다. [검증](output_etc/check/wait_eval1500_20261102/verification.json).

### BEFORE 실제 근거리 추세·wait와 같은 쌍 선행 AT 보고

- 반대 집단인 선행 접근+운반 경로 < 후행 수행 경로도367쌍(21.73%, SIT83·CLIMB80·ON_TOP204)으로 집계했다. 동거리0건이며 기존 후행 근거리 집단의 정확한 여집합이다. 최신 CLIMB 후행/같은 선행AT는 source5 45.18/84.54%, virtual23.30/85.93%, asym25.77/89.32%, wait1000 23.74/64.30%였다. 실제 완료 순서가 아닌 초기 XY 경로 합 분류다. [전체 checkpoint·평균/std·분모](output_etc/check/goalroute_leader_near_trends_wait_20261102/report.md).
- 기존 goalroute 세 모델의500/1000/1500/중간6000·5500/7000/최신9000·8500 저장본18개와 wait500/1000의 평가2개를 새 시뮬레이션 없이 CPU 집계했다. 초기 후행 수행 경로 < 선행 접근+운반 경로인1322/1689쌍(78.27%)을 동일하게 분리했으며 SIT265·CLIMB311·ON_TOP746쌍이다. 강제 near50 선택 플래그가 아닌 실제 거리 기준이다.
- 최신 근거리 후행 동시 Plane Ever는 source5 SIT44.98·CLIMB40.17·ON_TOP21.86%, virtual35.13·2.91·20.23%, asym11.16·9.34·16.77%였다. 같은 쌍 선행 AT는 각각83.06·73.97·85.93%,93.19·69.90·85.00%,94.76·85.22·89.55%였다. wait1000은7.55·4.27·9.23%, 같은 선행 AT81.60·57.04·64.98%였다. [전체 추세·평균/std·분모](output_etc/check/goalroute_near_trends_wait_20261102/report.md).
- 모든 평가/seed의 거리·분류 일치, near+나머지의 건수/성공 합과 전체 평가 재현, 기존 근거리 보고서의6개 모델·checkpoint 결과 재현, 후행 동시 Ever≤같은 쌍 선행 AT Ever를 확인했다. 사용자 요청으로 AGENTS/evaluation 보고 규칙에 전체·행동별·거리별·추세의 후행과 같은 분모 선행 AT 동시 보고를 추가했다. 학습·관측/보상·평가 지표 코드는 변경하지 않았다. [검증](output_etc/check/goalroute_near_trends_wait_20261102/verification.json).

### Virtual wait 500/1000 GPU5 Plane Ever 평가

- 사용자 지정 GPU5에서 `_wait`의500·1000 고정 checkpoint를 각각4명·8토큰·512환경×seed0·1·2×1200step·deterministic 첫 episode로 평가했다. 기존 정식 규약에 따라 대기 이동 종료를 제외하고 개인 낙상만 적용했으며 기존 virtual500/1000 결과를 비교용으로 재사용했다.
- wait500→1000 후행 동시 Plane Ever는 SIT6.83→7.79%, CLIMB6.83→8.22%, ON_TOP7.79→10.07%, 선행 AT64.85→69.13%였다. 단독 SIT89.64→86.87%이며 기존 virtual1000의79.63%보다는 높았다. virtual1000 대비 후행 SIT/ON_TOP·선행 AT는 낮았다. [500/1000·기존 virtual 비교표](output_etc/check/wait_eval500_1000_20261102/report_plane.md).
- 6회 모두1200step·CPU float64/GPU/JSON 일치·trace finite·float32/64 누적 판정 차이0, 모든seed의 기존 virtual 초기 상태 정확 일치, encoder73tensor/입력 RMS 동결·checkpoint·운영 파일223개 해시 보존을 통과했다. 기존4개 학습 PID도 유지했다. 정지 행동 자체의 진입/속도 지표는 이번 성공률 평가에 포함하지 않았다. [검증](output_etc/check/wait_eval500_1000_20261102/verification.json).

### goalroute 세 모델 최신9000/8500/9000 GPU2 Plane Ever 평가

- 사용자 지정 GPU2만 사용해 시작 시 최신 고정 저장본 source5 9000·virtual8500·virtual_asym9000을4명·8토큰·512환경×seed0·1·2×1200step·deterministic 첫 episode로 정식 평가했다. 각 checkpoint의 기존 random_scenario/near50·크기·거리 설정을 유지했으며 AT9 override와 새 wait 실험은 적용하지 않았다.
- 후행 동시 Plane Ever는 SIT45.17/32.29/12.91%, CLIMB41.16/7.15/12.82%, ON_TOP22.31/20.31/18.24%(source5/virtual/asym 순서)였다. 전체31.39±1.11/19.72±0.77/15.88±0.91%, 단독 SIT93.85/92.36/86.48%, 선행 AT84.20/85.21/90.67%였다. [행동별·단독·선행 AT 표](output_etc/check/goalroute_eval_latest_gpu2_20261102/report_plane.md). 모델epoch는 서로 다르다.
- 9개 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 판정 차이0, 같은seed 세 모델 및 이전7000 평가의 초기 상태 정확 일치, 입력 checkpoint·운영 파일 해시 보존을 통과했다. 기존 세 학습 PID 생존·학습 scalar finite도 확인했다. [입력 조건](output_etc/check/goalroute_eval_latest_gpu2_20261102/plan.json), [전체 수치](output_etc/check/goalroute_eval_latest_gpu2_20261102/comparison.json), [검증](output_etc/check/goalroute_eval_latest_gpu2_20261102/verification.json).

### source5·virtual AT 지정9종·64환경 진단 평가

- GPU4에서 source5 9000·virtual8500을 각각4명·8토큰·64혼합환경·seed2·1200step·deterministic Plane Ever로 평가했다. 평가 프로세스에서9종 제안 크기로 기존 크기 호환 역할 sampler를 한 번 실행해 첫 episode 역할을 고정하고, 실제 단독/선행 AT 물리 자산만9종으로 교체했다. ON_TOP payload 등 비AT 크기·기존 near50/거리/충돌 배치를 유지하며 운영 코드·checkpoint는 변경하지 않았다.
- 단독 AT96.30/96.30%(각26/27), BEFORE 선행 AT87.67/84.93%(64/73·62/73)였다. 후행 SIT/CLIMB/ON_TOP별 선행 AT는76.19/82.35/97.14%와85.71/76.47/88.57%였다. 전체 AT100건에서9종 모두 포함됐고 단독0.42³는 표본0건이다. 작은1seed 진단이며 모델epoch가 다르다. 두 모델 초기 상태 정확 일치·실제 AT 자산/관측 크기 일치·비AT 크기 보존·CPU/GPU/JSON 채점 일치·trace finite·checkpoint/운영 파일 해시 보존을 확인했다. 최초 공용 bank 전체 교체 예비 결과는 제외했다. [행동별·크기별 결과](output_etc/check/at9_eval64_20261102/report.md), [검증](output_etc/check/at9_eval64_20261102/verification.json).

### Virtual goalroute 후행 대기 이동 종료 실험

- 기존 virtual_goal5_goalroute를 보존하고 `_wait` 환경/학습 YAML·train/test/VNC/eval·출력/checkpoint 계약을 추가했다. 새 설정만 BEFORE 후행의 XY 거리(root 또는 ON_TOP payload)가 공유 상자 반대각선+0.5m 이내면 진입을 유지하고,1초 유예 후 root XY 속도>0.3m/s가1초 연속이면 후행 terminate/장면 reset을 적용한다. 선행 AT의 같은 스텝 raw_own_success로 episode 내 영구 해제하며 부분 reset은 해당 환경만 초기화한다. 기존 Progress·보상/Gate·낙상·샘플링·모델을 유지하고 정식 Plane 평가는 기존 규약상 새 종료를 제외한다. config/structure/USAGE/상세 문서를 갱신했다.
- GPU4만 사용해4명·1024환경·8토큰 CHECK를 epoch3/frame393216까지 실행했다.26k165tensor 전이·encoder73tensor/입력 RMS 동결·head6tensor 갱신·모델/optimizer/지표 finite, CPU212개 통과(기존1개 skip·전이3개 제외),6개preset의8환경16step actor/critic·Virtual·종료 버퍼 연결·동시 성공 해제·latch·부분 reset을 확인했다. 강제 SIT/CLIMB 검사는 호환 bank로 별도 초기화했다. 기존 세 goalroute YAML6개 해시·문서 링크/21개 설정 목록·wrapper 구문/diff 검사도 통과했다. 본학습·학습된 대기 성능·장기 속도는 미검증이다. [검증 보고서](output_etc/check/before_wait_20261102/report.md).

### goalroute 세 모델 7000 checkpoint 4명 Plane Ever 평가

- 사용자 지정 GPU5·6에서 세 실험 모두 epoch7000/frame917504000 저장본으로4명·8토큰·512환경×seed0·1·2×1200step·deterministic 첫 episode를 정식 평가했다. random_scenario/near50을 포함한 저장 설정을 유지하고 기존 학습·checkpoint·운영 코드는 보존했다.
- BEFORE 후행 동시 Plane Ever는 SIT34.55/29.61/12.03%, CLIMB31.40/6.36/7.67%, ON_TOP18.82/19.03/16.48%(source5/virtual/asym 순서)였다. 전체25.02±1.35/18.29±2.13/13.54±0.79%, 단독 SIT96.94/92.34/82.69%, 선행 AT84.63/83.29/87.65%였다. [행동별·단독·선행 AT 표](output_etc/check/goalroute_eval7000_20261102/report_plane.md).
- 9개 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 판정 차이0·세 모델의 같은seed 초기 상태 정확 일치·입력 checkpoint 및 운영 파일281개 해시 보존을 통과했다. 기존 세 학습 PID 생존도 확인했다. [입력 조건](output_etc/check/goalroute_eval7000_20261102/plan.json), [전체 수치](output_etc/check/goalroute_eval7000_20261102/comparison.json), [검증](output_etc/check/goalroute_eval7000_20261102/verification.json).

### BEFORE 세 모델 8명·16토큰·64환경·1회 평가

- 추가 GPU/시뮬레이션 없이 기존4명(6000/5500/6000·512환경×3seed 평균)과8명(7000/6500/7000·64환경×seed2)의 Plane Ever 차이를 집계했다. checkpoint·평가 규모가 달라 순수 인원 수 효과로 해석할 수 없음을 명시했다. [4명→8명 행동별 차이](output_etc/check/before8_eval64_20261102/comparison_4_to_8.md), [집계 JSON](output_etc/check/before8_eval64_20261102/comparison_4_to_8.json).
- 사용자 지정대로 GPU5·6만 사용해 source5 7000·virtual6500·asym7000 저장본을8명·16토큰·64환경·seed2·첫 episode1회·1200step·deterministic으로 평가했다. 저장된 random_scenario/near50 설정을 유지하고 기존 Plane Ever 기하 판정·개인 낙상 freeze를 재사용했다. 학습4명에서 인원을 바꾼 진단 평가이며, 정식 general의 인원 변경 금지 규칙은 운영 코드에서 유지하고 확인용 driver만 별도 경로에 뒀다.
- 후행 동시 Plane Ever는 SIT21.05/15.79/13.16%(각38쌍), CLIMB30.77/2.56/2.56%(각39쌍), ON_TOP10.11/13.48/8.99%(각89쌍), 전체17.47/11.45/8.43%(각166쌍)였다. 선행 AT81.33/77.71/79.52%이며 단독 행동은 양쪽 BEFORE 역할을 제외해 집계했다. 1seed이므로 표준편차를 계산하지 않았다. [행동별·단독·선행 AT 표](output_etc/check/before8_eval64_20261102/report_plane.md).
- 세 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 차이0·모델 간 초기 상태 정확 일치·checkpoint 및 운영 파일281개 해시 보존을 확인했다. 기존 학습 프로세스도 유지됐다. 8명 초기화에서 기존 PUSH sequential 경로 배치가 오래 걸렸으며 이번 예산 수정에서 해당 물리 sampler를 최적화하거나 검사를 완화하지 않았다. [입력 조건](output_etc/check/before8_eval64_20261102/plan.json), [전체 수치](output_etc/check/before8_eval64_20261102/comparison.json), [검증](output_etc/check/before8_eval64_20261102/verification.json).

### BEFORE PUSH viewer의 가변 인원·물체 예산 지원

- 추론 validation을 CLI와 task 생성까지 전달해 BEFORE PUSH viewer/test/VNC에서 N≥1·M≥N을 허용했다. Alias 포함 `N+ON_TOP 수`가 예산을 초과한 과제 조합만 최대1024회 재샘플링하며 불가능한 고정 ON_TOP preset은 즉시 거절한다. 학습/resume의 M≥2N·기존 충분한 예산의 샘플링/RNG·모델/보상/Gate/계약과 정식 general 규약은 유지했다. config/USAGE/structure를 갱신했다.
- CPU132개 통과(기존 checkpoint 전이1개 제외), 수정 전후 CPU5개·CUDA4개 사례의 그래프/역할/template/RNG 정확 일치, GPU5의 작은 토큰 예산2560개 장면 검사를 통과했다. GPU6에서 기존 asym6000을8명·1환경·12토큰·seed2·32step으로 실제 VNC 실행/종료하고 CUDA/Vulkan GPU 일치를 확인했다. 같은 GPU에서 실제 test wrapper의1명·1토큰/source5 7000과8명·8토큰/virtual6500을 각각1환경·seed2·16step으로 완료했다. 보조8환경 다중preset 검사는 random_scenario16step·부분reset 이후 all_solo의 기존 PUSH 배치 지연으로 해당 보조 프로세스만 중단했으며 전체 preset 통과로 간주하지 않는다. 기존 학습·참고 jointpush 레포는 보존했다. [검증/제한](output_etc/check/before_viewer_budget_20261102/report.md), [이번 구현 diff](output_etc/check/before_viewer_budget_20261102/implementation_diff.patch).

### 최신 goalroute 세 BEFORE 실험 Plane Ever 평가

- 실제 초기 후행 수행거리가 선행 접근+운반거리보다 짧은1322/1689쌍(78.27%)을 CPU로 분리 집계했다. 후행 SIT45.36/22.51/10.51%, CLIMB24.15/2.60/1.00%, ON_TOP19.66/19.57/17.55%였다. 강제 near 표시는 저장되지 않아 실제 거리로 분리했으며, 모델별 분류 동일·near/far 합의 기존 행동/seed별 건수 정확 재현을 확인했다. [가까운 쌍/나머지 표](output_etc/check/goalroute_eval_latest_20261102/report_near_plane.md).
- 평가 시작 시 최신 고정 저장본인 source5 6000/frame786432000, virtual 5500/frame720896000, virtual_asym 6000/frame786432000을 각각512환경×4명×seed0·1·2×1200step deterministic 첫 episode로 평가했다. 직전 허용 GPU0·3 중 GPU0은 여유 메모리 약2GB라 GPU3만 사용했고 기존 프로세스·학습 설정은 유지했다. near50을 포함한 checkpoint의 학습 샘플링을 사용하며 모델별 epoch가 다르다.
- BEFORE 후행 동시 Plane Ever는 SIT44.68/21.81/11.46%, CLIMB25.96/6.17/6.09%, ON_TOP21.31/20.69/18.27%였다. 단독 SIT96.55/92.31/85.80%, 선행 AT82.84/83.00/84.72%였다(source5/virtual/asym 순서). [행동별 표·선행 AT](output_etc/check/goalroute_eval_latest_20261102/report_plane.md), [1500→최신 비교](output_etc/check/goalroute_eval_latest_20261102/comparison_1500_to_latest.md).
- 9개 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 판정 차이0·입력 checkpoint 해시 보존을 통과했다. 세 모델과1500↔최신의 같은seed 초기 상태가 정확히 같았고 운영 코드/설정281개 해시도 같았다. 현재 세 학습 PID 생존·누적 학습 지표 finite를 확인했다. [입력 조건](output_etc/check/goalroute_eval_latest_20261102/plan.json), [전체 수치](output_etc/check/goalroute_eval_latest_20261102/comparison.json), [전체 검증](output_etc/check/goalroute_eval_latest_20261102/verification.json).

## 2026-11-01

### 최신 goalroute 세 실험 1500 checkpoint Plane Ever 평가

- 추가 CPU 분리 집계에서 초기 후행 수행거리가 선행 접근+운반거리보다 짧은 쌍은1322/1689개(78.27%)였다. near50은 약50%에 짧은 거리를 강제하며 나머지 무제약 표본도 가까울 수 있다. 저장되지 않은 강제 near 선택 표시 대신 실제 초기 거리로 분리했고 세 모델/seed별 분류 동일·near/far 건수 합의 기존 결과 정확 재현을 확인했다. [가까운 쌍/나머지 Plane 표](output_etc/check/goalroute_eval1500_20261101/report_near_plane.md).
- 사용자 지정 GPU0·3에서 세 goalroute 실험의 epoch1500/frame196608000 저장본을 각각512환경×4명×seed0·1·2×1200step deterministic 첫 episode로 평가했다. BEFORE는 선행AT·후행 동시 Plane Ever, 단독은 혼합 장면에서 BEFORE 양쪽 역할을 제외한 개인 Ever다. 기존 프로세스·학습 설정은 유지했다.
- BEFORE 동시 전체 평균±표본std는 source5 `21.42±0.77%`, virtual `12.27±1.03%`, asym `12.22±1.57%`였다. 후행 SIT38.07/13.79/11.27%, CLIMB19.91/5.13/6.64%, ON_TOP15.86/14.64/14.86%였다. 단독 SIT94.64/82.35/84.62%로1000 대비 각각+3.48/+2.72/-3.82%p였다. [행동별 Plane 표](output_etc/check/goalroute_eval1500_20261101/report_plane.md), [1000→1500 비교](output_etc/check/goalroute_eval1500_20261101/comparison_1000_to_1500.md).
- 9개 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 차이0·checkpoint 해시 보존을 통과했다. 세 모델과1000↔1500의 같은seed 초기 상태가 정확히 같았고 운영 코드/설정281개 해시도 보존됐다. 모든seed GPU0/3 실행과 현재 세 학습 PID·지표 finite를 확인했다. [입력 조건](output_etc/check/goalroute_eval1500_20261101/plan.json), [전체 검증](output_etc/check/goalroute_eval1500_20261101/verification.json).

### 최신 goalroute 세 실험 1000 checkpoint Plane Ever 평가

- 동일 세 goalroute 실험의 epoch1000/frame131072000 저장본을 GPU5·6만 사용해 각512환경×4명×seed0·1·2×1200step의 deterministic 첫 episode로 정식 평가했다. BEFORE는 선행AT·후행 동시 Plane Ever, 단독은 혼합 장면에서 BEFORE 양쪽 역할을 제외한 개인 Plane Ever다. 기존 학습·checkpoint·운영 코드는 유지했다.
- BEFORE 동시 전체 평균±표본std는 source5 `18.93±0.91%`, virtual `13.39±0.44%`, asym `12.29±1.66%`였다. 후행 SIT25.65/15.07/9.99%, CLIMB22.50/6.75/5.91%, ON_TOP14.90/15.54/15.72%였다. 단독 SIT는91.16/79.63/88.44%로500 대비 각각-5.34/+1.99/+0.03%p였다. [행동별 Plane 표](output_etc/check/goalroute_eval1000_20261101/report_plane.md), [500→1000 비교](output_etc/check/goalroute_eval1000_20261101/comparison_500_to_1000.md).
- 9개 rollout 모두1200step·CPU float64/GPU/JSON 판정 일치·trace finite·float32/64 누적 차이0·입력 checkpoint 해시 보존을 통과했다. 세 모델 및500↔1000의 같은seed 초기 상태가 정확히 같았고 두 평가의 운영 코드/설정281개 해시도 같았다. 모든seed GPU5/6 실행과 현재 세 학습 PID·지표 finite를 확인했다. [입력 조건](output_etc/check/goalroute_eval1000_20261101/plan.json), [전체 검증](output_etc/check/goalroute_eval1000_20261101/verification.json), [seed별 변화](output_etc/check/goalroute_eval1000_20261101/delta_from_500.json).

### 최신 goalroute 세 실험 500 checkpoint Plane Ever 평가

- source5_goal5_goalroute·virtual_goal5_goalroute·virtual_asym_goal5_goalroute의 epoch500/frame65536000 저장본을 GPU5·6만 사용해 각512환경×4명×seed0·1·2×1200step의 deterministic 첫 episode로 정식 평가했다. 시작 시 virtual은466epoch여서 해당500 저장본이 생성된 뒤 평가했으며, 각 checkpoint의 goal_route_sum·목표 거리·Virtual/비대칭 설정을 유지했다.
- BEFORE 동시 Plane Ever 전체 평균±표본std는 source5 `14.26±1.15%`, virtual `11.60±0.90%`, asym `10.71±1.23%`였다. 후행 SIT22.22/13.83/9.99%, CLIMB19.76/8.45/9.24%, ON_TOP9.14/12.09/11.57%이며 단독은 BEFORE 양쪽 역할을 제외해 따로 집계했다. [행동별 Plane 표](output_etc/check/goalroute_eval500_20261101/report_plane.md), [seed별 건수·통계](output_etc/check/goalroute_eval500_20261101/plane_tables.json).
- 9개 rollout 모두1200step을 완료하고 CPU float64 재채점·GPU 누적·JSON 건수가 일치했다. trace finite·float32/64 누적 판정 차이0·같은seed 초기 상태 정확 일치·입력 checkpoint 해시 보존과 모든seed GPU5/6 실행 기록을 확인했다. 운영 코드/설정281개 해시와 기존 세 학습 PID·지표 finite도 유지됐다. [입력/실행 조건](output_etc/check/goalroute_eval500_20261101/plan.json), [전체 검증](output_etc/check/goalroute_eval500_20261101/verification.json).

### 이전 source5·virtual·virtual_asym 5000 checkpoint Plane Ever 평가

- 이전 세 run의 epoch5000/frame655360000 저장본을 GPU5·6에서 각512환경×4명×seed0·1·2×1200step의 deterministic 첫 episode로 정식 평가했다. checkpoint에 저장된 기존 relative_sum 배치를 사용했고 현재 goalroute 학습은 유지했다. BEFORE는 선행AT·후행 같은 frame의 동시 Plane Ever, 단독은 BEFORE 선행·후행을 모두 제외한 개인 Plane Ever다.
- BEFORE 동시 전체 평균±표본std는 source5 `15.12±1.78%`, virtual `8.12±0.68%`, asym `9.10±1.16%`였다. 행동별 SIT33.85/7.59/9.72%, CLIMB13.82/2.07/2.25%, ON_TOP8.85/10.85/11.66%로 source5의 SIT/CLIMB 우위가 유지됐다. [Plane 행동별 표](output_etc/check/eval5000_20261101/report_plane.md), [세부 수치](output_etc/check/eval5000_20261101/plane_tables.json).
- 9개 rollout의 CPU float64 재채점·GPU 누적·JSON 건수가 일치했고 trace finite·float32/64 누적 판정 차이0·동일seed 초기 상태 정확 일치·입력 checkpoint 해시 보존을 확인했다. 비교 코드 삭제 후 운영 코드/설정281개 해시도 유지됐고 현재 세 goalroute 학습 PID·지표 finite를 확인했다. [입력 조건](output_etc/check/eval5000_20261101/plan.json), [전체 검증](output_etc/check/eval5000_20261101/verification.json).

### 사용하지 않는 최적화 비교·계측 코드 정리

- 사용자 요청으로 `output_etc/check/push_opt123_20261101`·`reset_share_20261101` 아래 일회성 Python17개와 bytecode9개만 삭제했다. 운영 경로의 참조·실행 프로세스가 없음을 확인했고 코드/설정281개 해시를 전후 비교해 전부 보존했다. 현재 학습·원본·모든 checkpoint·보고서·JSON·로그는 유지했다. 보고서의 삭제 코드 링크를 정리하고 삭제 표시를 추가했다. [삭제 목록·검증](output_etc/check/eval5000_20261101/cleanup_result.json).

### source5 goalroute 실제 학습의 reset 시간 비중 계측

- GPU5·원본 source5_goal5_goalroute·4명/1024환경/8토큰·seed42·Stage1 26k에서 CHECK40epoch(1280step)를 실행했다. 계측은 별도 프로세스에서 reset 앞뒤 동기화와 후보 sampler CUDA event로 추가했고 운영 코드·모델/보상/샘플링 설정·기존 프로세스는 유지했다. 시작 전체 초기화와 첫2epoch는 집계에서 제외했다.
- 측정38epoch275.584초 중 reset39.994초(14.51%), 나머지 rollout80.372초(29.16%), PPO/AMP 업데이트·배치 준비 등155.217초(56.32%)였다. reset 내부 PUSH 후보 sampler는7.369초로 전체2.67%·reset18.42%였다. reset465회 중464회는1–5환경,1회는550환경이었고 호출당 환경 수 중앙값1이었다. 큰 전체 초기화의63–66% 단축을 학습 중 작은 부분 reset에 그대로 대입할 수 없으며, 이번 후보 sampler 최적화가 줄일 수 있는 전체 시간 비중은 제한적이다.
- 40epoch/frame5242880 완료·학습 지표/모델/optimizer/정규화 finite·encoder73 tensor/입력 RMS 동결·운영 sampler 해시 보존을 확인했다. 단일seed 초기 학습과 동기화 계측의 근사 비중이며 성숙한 정책·세 모델 동시 학습이나 최적화 적용 후 처리량을 보장하지 않는다. [보고서](output_etc/check/reset_share_20261101/report.md), [전체 집계](output_etc/check/reset_share_20261101/verification.json), [checkpoint 검사](output_etc/check/reset_share_20261101/checkpoint_validation.json).

### source5 goalroute PUSH sampler 최적화1·2·3 분리 비교

- 사용자 요청으로 GPU5만 사용해 source5_goal5_goalroute 하나에서 원본/실제 PUSH 경로 벡터화/반경 재사용/후보 메타데이터 공유를 누적 비교했다. 구현은 `output_etc/check/push_opt123_20261101/variants.py`의 프로세스별 교체로 격리했고 운영 코드·세 설정·기존 프로세스는 보존했다. 최종 전체 장면 validator는 원본을 유지한다.
- CPU/GPU 각각 경로147,456건·경계30건·sampler54건의 판정/후보/미해결 index/입력/RNG 동일성을 통과했다. 교차32,768회 reset의 상태·관측이 정확히 같았고, 4명/1024환경/8토큰 스트레스65,596회에서 거리/near/충돌/경로/각도/성공 환경 보존·부분 reset을 통과했다. 최종 실패0·최대10라운드·버전별 fallback2,216건 전부 해결·집계 동일이었다. 네 버전의32환경5preset×16step Actor/Critic·alias·보상/관측 finite·부분 reset·모델 보존과 각5epoch CHECK 학습의26k165 tensor 전이·encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite를 확인했다.
- 같은 GPU에서 순서를 교차한6개 측정 seed의 전체 reset 중앙값은 원본1.749초, 1번0.645초, 1+2번0.629초, 1+2+3번0.589초였다. 순서를 뒤집은 최종 단일seed 학습에서 첫 epoch 제외 중앙값은 각각6.890/7.003/7.239/7.130초로 전체 PPO 속도 개선은 확인되지 않았다. 후보 중간 메모리는108.0→179.5/162.8/146.5MiB로 늘었고 전체 PyTorch 학습 peak는 약13.17GiB였다. 장기 처리량·정책 성능은 미검증이므로 정식 반영은 보류한다. 기존 mix226 전용 물리 검사 도구가 goalroute 설정을 거절한 것은 비교용 검사만 현재 거리 검증으로 교체해 재검증했다. [보고서](output_etc/check/push_opt123_20261101/report.md), [전체 결과](output_etc/check/push_opt123_20261101/verification.json).

### goal5 기반 목표 중심 BEFORE 배치·수행 거리 near50 실험3개

- 세 goal5 설정에 `_goalroute`를 붙인 환경/학습 YAML과 train/test/VNC/eval을 추가했다. 선행 A→O·O→G 각각1–5m, SIT/CLIMB 후행 G→B 1–10m, ON_TOP G→P·P→B 각각1–5m와 B→G≥1m를 적용한다. near50은 생성된 쌍 절반에 후행 수행 거리 합 < 선행 접근+운반 거리 합을 강제한다. 기존 goal5 여섯 YAML 해시·단독 거리·모델/보상/Gate/Virtual/비대칭 설정을 보존하고 새로운 before 계약·출력·20개 실험의 문서/명령 목록을 분리했다. 기본은 동일 Stage1 26k 전이이며 기존 goal5 저장본과 resume/추론을 혼용하지 않는다.
- near 거리 합의 조건부 CDF와 조건부 각도로 반복 rejection을 피한다. 새 설정에서만 AT 목표·후행 배치를 PUSH 후보 검사보다 먼저 확정하고, PUSH/distractor/단독 ON_TOP 받침처럼 나중에 배치할 물체를 제외한 고정 장면을 선검사한다. 최종 전체 충돌·경로·각도·strict near 판정과64회 외부 재시도는 유지한다. 이 선검사는 중간 검사에서 발견한 불가능한 장면의 긴 PUSH 순차 fallback을 줄이기 위한 것이다.
- GPU5/6에서 최종 코드의 각4명·1024환경·8토큰을16,399회씩, 총49,197회 reset했다. 혼합·전원PUSH·BEFORE 두쌍/한쌍·독립과 부분 reset, 모든 거리·near50·충돌·경로·각도·graph/성공 환경 보존을 통과했고 최종 실패0·최대10라운드·순차 fallback 미해결0이었다. 설정별 후행 SIT2645·CLIMB2461·ON_TOP11594건을 포함했고 강제 near5739/11571쌍(49.60%)이었다. 세 설정의 기하 검사 집계도 동일했다.
- CPU213개 통과(없는 기존 checkpoint4개 skip·기존21k 전이1개 제외), 세 새 wrapper CHECK 학습 epoch2/frame262144·26k165 tensor 전이·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·finite를 확인했다. 최종 선검사 포함 asym CHECK 재확인과32환경5preset×16step의 actor/critic·Virtual XYZ/실제 회전·같은 스텝 AT 성공 전환·State/Success/HOLDING/원본 상태 보존, 세 wrapper의4환경32step headless 재로딩을 통과했다.12개 wrapper 구문·20개 config 표/세 명령 목록·문서 링크·diff 검사도 통과했다.
- 같은 GPU6에서 순서를 교차한8쌍 중 warmup2쌍을 제외한 혼합1024환경 reset 중앙값은 기존 goal5 5.246초→goalroute 3.827초(27.1% 감소)였다. 기존 본학습과 함께 실행된 reset 측정이며 장기 PPO 전체 처리량 개선을 보장하지 않는다. 기존 사용자 프로세스를 유지했으며 본학습·새 정책 성능 평가·GUI/VNC 실화면은 실행하지 않았다. [검증 보고서](output_etc/check/before_goalroute/report.md), [전체 수치/실행 조건](output_etc/check/before_goalroute/verification.json).

### 세 모델4k BEFORE·단독 Plane Ever 재집계

- 사용자 요청으로 기존 source5/virtual/virtual_asym epoch4000·512환경×4명×3seed·1200step 결과에서 `region`을 Plane Ever 주 지표로 정리했다. BEFORE는 선행 AT와 후행의 같은 프레임 동시 Ever, 단독은 선행·후행 모두 제외한 개인 Ever다. 기존 Point 표는 보존하고 [Plane 표](output_etc/check/latest3_eval_20261101_132836/report_plane.md)·[seed별 건수/통계](output_etc/check/latest3_eval_20261101_132836/plane_tables.json)를 추가했다.
- 기존9개 CPU/GPU 재채점 결과와 finite를 확인하고 저장 trace의 Plane 성공 건수/JSON 일치·기존 Point 단독 표 재현을 통과했다. 새 시뮬레이션·GPU 작업 없이 재집계했다. 후행 SIT 동시 Ever29.07/5.82/14.12%, 단독 SIT95.37/79.60/93.07%이며 값은 source5/virtual/asym 순서의 seed 평균이다.

### 원본 Stage1 26k 단독 SIT·CLIMB 거리별 평가

- 사용자 지정 주 지표는 **Plane Ever**(기존 코드 `region`)다. 저장·CPU 검증한 결과를 다시 정리했으며 추가 시뮬레이션은 하지 않았다. Stage1 26k는 SIT1–5m83.75%·6–10m90.31%, CLIMB100.00%·99.69%다. [Stage1 Plane 표](output_etc/check/stage1_26k_distance_eval_20261101/report_plane.md), [source5 Plane 표](output_etc/check/source5_distance_eval_20261101/report_plane.md). 아래 Point 수치는 별도 지표로 보존한다.
- `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`의 저장 env/sim을 복원하고 Stage1 단독 기준1명·1상자·loco·deterministic·900step·seed0으로 평가했다. 사람 root→자기 상자 중심 초기 XY 거리1·2·…·10m마다64환경, 행동별640시도이며 GPU5(SIT)·6(CLIMB)만 사용했다. 기존 학습 프로세스·코드·설정·원본 checkpoint는 보존했다.
- Point Ever는 SIT1–5m70.00%·6–10m71.88%, CLIMB98.75%·98.12%였다. 앞선 source5는4명·8토큰·1200step이므로 모델만 바꾼 통제 비교와 구분한다. 두 rollout900step·총1280시도의 CPU float64 재채점/GPU 누적/JSON 집계 일치, float32/64 누적 판정 차이0·trace finite·초기 거리 오차 최대0.067mm·checkpoint SHA256 보존을 확인했다. [거리별 결과](output_etc/check/stage1_26k_distance_eval_20261101/report.md), [실행 조건/검증](output_etc/check/stage1_26k_distance_eval_20261101/comparison.json).

### source5 5k 독립 SIT·CLIMB 거리별 평가

- 요청 시점 최신 source5 epoch5000 저장본을 사람 root→자기 상자 중심 초기 XY 거리1·2·…·10m에서 평가했다. 행동·거리별64환경×4명=256시도, seed0·loco·deterministic·1200step이며 전원 독립 SIT 또는 CLIMB로 배치했다. 별도 평가 프로세스에서 거리만 고정하고 기존 학습·설정·보상·충돌 검사를 유지했다. 단일 seed 거리 진단이며 일반 혼합 과제 평가와 구분한다.
- Point Ever는 SIT1–5m 평균83.05%·6–10m82.89%, CLIMB95.00%·95.63%였다. 거리별 결과는 [결과표](output_etc/check/source5_distance_eval_20261101/report.md)에 있다. 두 rollout의1200step·전체5120시도 CPU float64 재채점과 GPU 누적/JSON 집계 일치, float32/64 누적 판정 차이0·trace finite·최대 초기 거리 오차0.168mm·checkpoint SHA256 보존을 확인했다.
- 평가는 GPU0/1에서 완료됐고 해당 프로세스는 종료됐다. 완료 직후 사용자 지시에 따라 이후 GPU 작업은5/6만 사용한다. [실행 조건/검증](output_etc/check/source5_distance_eval_20261101/comparison.json).

### source5 계열 goal5 실험3개 추가

- `stage2_before_push_near50_pair50_source5_goal5`·`stage2_before_push_near50_pair50_source5_virtual_goal5`·`stage2_before_push_near50_pair50_source5_virtual_asym_goal5` 환경/학습 YAML과 train/test/VNC/eval을 추가했다. `before.goal_distance: [1.0, 5.0]`으로 단독 AT·BEFORE 선행 AT의 현재 상자→목표와 단독 ON_TOP의 들 상자→받침을 반경 샘플링하고, PUSH 상자→목표는2–5m로 바꾼다. 사람과 목표 최소1m·기존 Z 계산은 유지하고 새 목표에는 원점±4.5m 제한을 적용하지 않는다. 기존 세 실험의 환경/학습 YAML6개 해시를 보존했다.
- 새 설정에서만 후보 상자 간격·AT 목표와 다른 상자 간격에1mm 여유, PUSH 후보 cone89.98°를 적용한다. 최종 충돌·90° cone 기준은 유지한다. 사용자 선택대로 BEFORE 후행 최종 수행 거리에는 상한을 추가하지 않았고 현재 공유 상자/들 상자 배치·near50·보상·Gate·Virtual 관측·lambda10·모델·AMP/PPO는 유지한다. 새 계약/출력을 분리하고 기본 전이는 모두 기존 실험 실행과 같은 Stage1 26k다. config/structure/USAGE/README·거리 상세를 갱신했다.
- CPU191개 통과(없는 기존 가중치4개 skip·21k 전이1개 제외), 기존 정사각형 sampler 출력/RNG 회귀를 통과했다. GPU0/3/1·각4명/1024환경/8토큰에서 혼합·전원PUSH·BEFORE 두쌍/한쌍·독립과 부분 reset을 각각16,399회, 총49,197회 검사해 최종 실패0·장면 검사 최대9라운드·거리/near/충돌/경로/각도/성공 환경 보존을 확인했다. 각 순차 fallback2건은 기존 장면 재시도로 해결했다.
- 새 wrapper 세 개의 CHECK 학습을 각각 epoch2/frame262144까지 저장했다.26k165 tensor 전이·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite와 각4환경32step headless 재로딩을 통과했다. 새 asym 저장본의32환경5preset×16step·부분 reset에서 Virtual XYZ/실제 회전·같은 스텝 AT 성공 전환·State/Success/HOLDING/원본 상태 보존을 확인했다.17개 컨피그 표·세 명령 목록·12개 새 wrapper·문서 링크/구문 검사를 통과했다. 본학습·정규 성능 평가·GUI/VNC 실화면은 실행하지 않았고 기존 학습을 유지했다. [전체 검증](output_etc/check/before_goal5/verification.json).

### PUSH 목표 거리2–5m 검증과 reset 수치 경계 확인

- 기존 source5/virtual/asym 설정·코드와 학습 프로세스는 보존하고, 별도 검증 프로세스의 메모리에서 PUSH 상자→목표만2–5m로 바꿨다. 기존2–10m는 변경 범위를 접근 거리로 한정하면서 유지한 값이며,5m 축소 검증 실패로 되돌렸다는 기록은 확인되지 않았다. 현재 validator는 기존 Stage1 PUSH의2–10m 계약을 요구하므로 새 범위는 별도 계약 지원이 필요하다.
- 막힌4명 PUSH fixture는 기존/축소 각각20seed에서 통과했다. GPU0/3·4명·1024환경·8토큰에서 기존2–10m는16,399회, 무여유2–5m는 혼합/전원PUSH/BEFORE 두쌍·한쌍/독립의 다섯 preset24,591회 reset을 완료했다.64회 한도 소진0·최대 장면 검사12/10라운드였지만, 축소 검사에서는 PhysX 반영 뒤 상자 간격 경계가 약0.007–0.008mm 뒤집힌 두 장면을 발견했다. 탐색 검사의 추가 강제SIT preset은 random asset bank와 호환되지 않아 제외하고 기존 범위는 다섯 preset으로 재검증했다.
- 후보 간격1mm 여유만 준 중간 검사에서는 PUSH 방향44.99903°가 반영 뒤45.00025°로 바뀌는 각도 경계도 발견했다. 검증용 프로세스에서 후보 간격1mm·후보 cone89.98°를 함께 적용하자16,399회 reset과 부분 reset·그래프/성공 환경 보존·거리/near/경로/충돌/각도 최종 검사를 통과했다. 최종 간격50mm·cone90° 및 나머지 판정은 유지했고 최대 장면 검사11라운드였다. 보완안은 실제 코드·config에 적용하지 않았다. [검증 요약](output_etc/check/push_goal5_reset/report.md), [전체 결과/제한](output_etc/check/push_goal5_reset/verification.json).

### source5·virtual·virtual_asym 최신4k 정식 평가

- 재시작한 세 run의 요청 시점 최신 저장본은 모두 epoch4000/frame524288000이었다. 전용 eval wrapper로 각512환경×seed0·1·2·4명·1200step·deterministic 첫 episode를 GPU0/2/3에서 평가했다. 기존 GPU5/6 학습과 checkpoint를 보존했다.
- 점 기준 개인 macro5는 source5 `74.54±0.75%`, virtual `56.00±2.13%`, virtual_asym `61.26±2.28%`였다. BEFORE 동시 성공은 각각 `15.82±0.92%`, `8.41±3.26%`, `11.70±1.92%`, 후행 개인 성공은 `46.15±1.70%`, `9.24±3.73%`, `12.11±1.99%`였다. 실험당1689쌍이며 값은 seed 평균±표본std다. 비대칭은 virtual보다 개선됐지만 현재 시점에서는 원본이 우세하며, 후행 SIT/CLIMB 동시 성공에서 차이가 컸다. 선행 AT는 `67.62/67.00/75.35%`였다.
- 모든 seed에서 세 실험의 저장 초기 상태가 정확히 동일했다.9개 rollout이1200step을 완료하고 CPU float64 재채점·GPU 누적 성공·JSON 집계가 정확히 일치했다. float32/64 판정 차이0·trace finite·입력 checkpoint SHA256 보존을 확인했다. [결과표](output_etc/check/latest3_eval_20261101_132836/report.md), [전체 수치/검증](output_etc/check/latest3_eval_20261101_132836/comparison.json), [입력/실행 조건](output_etc/check/latest3_eval_20261101_132836/plan.json).

### source5 계열 PUSH 초기 후보 검사 배치화

- source5·virtual·virtual_asym의 초기 PUSH 후보128개를16개씩 최대8묶음으로 검사하도록 변경했다. 기존 `sample_layout(..., max_attempts=1)`의 유효성 검사를 재사용하며 환경별 첫 유효 후보·성공 환경 보존·후보 예산·순차 fallback·장면 재시도를 유지한다. 보상·Virtual 관측·lambda10·모델·checkpoint 계약·실행 명령과 다른 설정은 유지한다. 난수 소비 순서가 바뀌므로 같은 seed의 배치는 달라질 수 있다. 상세/structure/USAGE를 갱신했다.
- 관련 CPU173개 통과(없는 기본 가중치4개 skip·21k 전이1개 제외). GPU6·1024환경 reset16,399회에서 최종 실패0, 순차 fallback2,658건 해결·성공 환경/graph/near/거리/경로 조건 보존을 확인했다. GPU5·1024환경 CHECK 학습 epoch4/frame524288와 재확인 학습에서 actor encoder73 tensor·입력 RMS 동결, head6 tensor 갱신, 모델/optimizer/지표 finite를 확인했다.32환경5preset의 물리·Virtual XYZ/실제 회전·현재 AT 성공 전환·기존 State/Success/HOLDING 보존과4환경32step headless 재로딩을 통과했다.
- 후보 검사만 비교하면 어려운1환경은 중앙값1.290→0.084초(15.28배), 일반1024환경은0.0259→0.0109초(2.37배)였다. GPU5에 다른 학습이 없는 상태에서 같은 설정·seed42·4epoch의 평균 rollout+update는8.735→7.416초로15.1% 감소했다. 난수 순서가 다른 짧은 단일 seed 측정이며 장기/동시 학습 속도 보장은 아니다. 기존 프로세스를 종료하지 않았고 본학습은 시작하지 않았다. [전체 검증](output_etc/check/push_batched_sampling/verification.json).

### Virtual BEFORE 비대칭 progress 실험 추가

- `stage2_before_push_near50_pair50_source5_virtual_asym` 환경·학습 YAML과 train/test/VNC/eval을 추가했다. 기존 virtual에서 `before.asymmetric_progress.lambda_inner: 10.0`만 추가하고 출력·checkpoint 계약을 분리했다. 기본26k 전이·Virtual GTA·물리 배치·모델·PPO/AMP·가중치·State/Success Gate·성공 saturation·HOLDING/독립/PUSH 보상은 유지한다. config/structure/USAGE·상세 설계를 갱신했다.
- `predecessor_at_progress()`에 현재 AT raw 성공과 lambda를 전달해 미완료 시 반경 안쪽 거리만10배 감쇠한다. 거리/root·payload/반경·바깥 감쇠는 유지하고 성공 시 기존 식으로 즉시 복원한다. 호출부는 Runtime과 같은 `interaction_own_success()`로 현재 phi/오차를 판정하며 이전 own_success/achieved를 읽지 않는다. 기존 설정은 lambda0 경로로 동일 계산을 유지하고 새 옵션은 source5 virtual에 한정해 검증한다.
- CPU165개 통과(없는 기본 가중치4개 skip·21k 전이1개 제외). GPU5·1024환경 CHECK epoch2/frame262144 저장, actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite를 확인했다.32환경7preset의 각16step·부분 reset과 성공/실패3회 전환을 검사해 같은 스텝 반영·성공 이력 무시·State/Success/HOLDING/독립 항 보존·Virtual XYZ/실제 회전 유지를 확인했다.4환경32step headless 재로딩과 eval wrapper help·구문·문서 링크를 통과했다. 기존 본학습은 유지했고 새 본학습·정규 성능 평가·실제 GUI/VNC는 실행하지 않았다. [전체 검증](output_etc/check/before_asymmetric_progress/verification.json).

## 2026-10-31

### source5 Goal-Aligned Virtual Object 실험 추가

- `stage2_before_push_near50_pair50_source5_virtual` 환경·학습 YAML과 train/test/VNC/eval을 추가했다. 환경 차이는 `relationGraph.before_virtual_object: at_goal_xyz_v1` 하나이며 source5 배치 보완·보상·Gate를 유지한다. 기본 전이는 연결된26k이고 비교 원본도 같은26k를 지정한다. 출력과 checkpoint 계약을 분리하고 기존 계약은 보존했다.
- `before_object_alias.py`에서 packet의 actor/payload/target으로 선행 AT Goal을 찾아 복사한 scene의 후행 Alias GTA XYZ만 변경한다. `stage2_push_network.py`의 Actor/Critic Encoder 직전에 함께 적용하고 Quaternion·intrinsic·원본 관측/물리 상태/graph·coordination packet은 유지한다. 기존 목표 XY progress가 곧 가상 상자 접근 progress이며 추가 보상이나 구조 변경은 없다. config/structure/USAGE·BEFORE 상세를 갱신했다.
- 관련 CPU140개 통과(없는 기본 가중치4개 skip·21k 전이1개 제외). GPU5·1024환경 CHECK epoch2/frame262144 저장,26k165 tensor 전이·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite를 확인했다.32환경7preset에서 각16step과 부분 reset, SIT/CLIMB/ON_TOP Alias XYZ·실제 회전·원본 상태/graph 보존·정책/critic/AMP finite를 통과했다.4환경32step headless 재로드와 전용 eval의4환경·seed0·1200step을 완료했고 CPU 재채점/JSON 집계 일치·float32/64 차이0·finite·입력 저장본 해시 보존을 확인했다. 소표본 실행 검사이며 실제 GUI/VNC·장기 성능은 검증하지 않았다. 기존 본학습은 유지했다. [전체 검증](output_etc/check/before_virtual_object/verification.json).

### source5 PUSH 후보 소진·reset 좌표 반올림 보완

- 4명 독립 PUSH에서128개 동시 후보가64회 모두 소진되는 종료를 재현했다. `stage2_before_push_near50_pair50_source5`에서 기존 후보가 소진된 환경에만 순차 경로 배치를 적용하고, 유효한 경로를 보존하며 막히면 순서를 다시 뽑는다. 그래프·near 선택·성공 환경은 보존하고 소진된 환경만 기존 장면 재시도로 넘긴다.
- 상태 반영 뒤 좌표가 최대0.061mm 달라져 PUSH 경로·near 경계가 뒤집히는 사례를 확인했다. 해당 설정의 후보 생성에만 경로 여유와 near 거리 차이1mm를 추가했다. 거리 범위·최종 부등식/충돌 조건·기존 near50/mix226 기본 샘플링은 유지하며, 보완된 장면의 공간적 배치 분포는 달라진다.
- 관련 CPU92개 통과(없는 기본21k 전이 검사1개 skip·1개 제외). GPU5·1024환경에서 일반51,200·4명 PUSH5,120·BEFORE 두 쌍/한 쌍 각5,120·독립5,120과 부분 reset15개, 총71,695회를 통과했다. 순차 보완7,951건 중7,949건을 해결하고2건은 기존 장면 재시도로 통과했다. 최종 reset 실패0·그래프/near/성공 환경 보존·기존 거리/경로/충돌 조건을 확인했다. 거리 경계 측정에는 float32 허용0.2mm를 사용했으며 최대 이탈은0.038mm였다.
- CHECK 학습 epoch11/frame1441792 저장·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/optimizer/지표 finite를 확인했다. 저장본의4환경32step headless 재로딩과32환경5preset의 물리·alias·AMP·부분 reset 검사를 통과했다. [전체 검증](output_etc/check/reset_1031/verification.json). 본학습은 재시작하지 않았으며 장기 수렴 성능은 검증하지 않았다.

### 1031 checkout 모션·26k 전이 연결 복구

- tmux `1031`의 SIT 모션 `FileNotFoundError`를 확인했다. `stage1_2_base_jointpush`를 참고해 누락된 SIT/CARRY/LOCO/CLIMB 네 모션 링크를 공유 원본에 연결하고, `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`를 참고 레포의 파일에 연결했다. 기존 PUSH 링크와 외부 원본은 유지했다.
- PUSH 모션 YAML161개·기존 모션 YAML136개 파일 참조가 모두 존재하며, 오류가 난 SIT 배열과26k checkpoint(epoch26000·model165 tensor)의 CPU 로딩을 확인했다. [검증](output_etc/check/data_links_1031/verification.json). 학습은 실행·재시작하지 않았다.

### BEFORE near50 쌍 선택50·접근 최대5m 설정 추가

- `stage2_before_push_near50_pair50_source5` 환경·학습 YAML과 train/test/VNC/eval을 추가했다. 기존 near50에서 쌍 생성 확률0.5·일반 사람→상자1–5m·독립 PUSH 접근2–5m만 적용한다. near50·AT/ON_TOP 목표 배치·PUSH 이동2–10m·보상·모델·loco·4명/1024환경/8토큰/1200step·기본21k 전이는 유지하고 출력·저장본 계약을 분리했다.
- 1024환경 첫 검사에서 고정 주변 배치에 대한 PUSH 후보 소진을 확인했다. 새 설정에서 성공 환경을 보존하고 실패 환경만 기존 장면 재샘플링으로 넘기도록 연결했으며 거리·경로·충돌 조건과 기존 실험 기본값은 유지했다. 전원 독립 PUSH32환경 검사는 재샘플링에 수분이 소요됐다.
- 최종 CPU196개 통과(없는checkpoint4개 skip·21k 전이1개 제외). GPU6·1024환경 CHECK epoch2/frame262144 저장·finite·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신을 확인했다. 32환경7preset의 거리·near·alias·PUSH 경로·부분 reset과4환경32step headless 재로드를 통과했다. AT 목표 최대14.04m·BEFORE ON_TOP 운반 최대9.08m를 관측해 별도 목표 거리 상한을 추가하지 않았음을 확인했다.
- 일반8환경·seed0·1200step 검사와 CPU 재채점의 건수 일치·float32/64 차이0·finite를 확인했다. 기본21k 파일이 없어 검증과 문서 학습 명령에는26k를 명시했다. [전체 검증](output_etc/check/before_pair50_source5/verification.json). config/structure/USAGE/README·BEFORE 상세를 갱신했다. 본학습·GUI/VNC 실화면·정규512환경×3seed 성능 평가는 실행하지 않았다.

### 개인 목표 solo5를 별도 컨피그로 분리

- 요청에 따라 `stage2_coupled_push_individual_goal`의 단독 거리 상한 변경만 되돌렸다. 공동 확률0.5와 개인 목표·접촉 설정은 유지한다. 단독 접근/PUSH 이동 최대5m는 새 `stage2_coupled_push_individual_goal_solo5` 환경·학습 YAML과 train/test/VNC/eval로 분리하고 출력·저장본 계약도 구분했다. 단독 CARRY AT 목표 높이 보존 수정은 유지한다.
- CPU71개 통과(기존21k 전이1개 제외), 원본/변형 환경 차이는 단독 상한 두 값뿐임을 확인했다. GPU6에서 기존5m 검증 저장본을 새 wrapper로 재로딩하여4환경32step 실행을 완료했다. [CPU 로그](output_etc/check/coupled_individual_solo5_split/cpu.log), [실행 로그](output_etc/check/coupled_individual_solo5_split/reload.log). shell 구문·diff 검사 통과. 본학습은 실행·재시작하지 않았다.

### 개인 목표 실험의 단독 접근·PUSH 이동 최대5m

- 새 개인 목표 컨피그에 `solo_source_max_distance: 5.0`·`solo_goal_max_distance: 5.0`을 추가했다. SIT/CLIMB/CARRY AT/ON_TOP의 사람→상자는1–5m, 단독 PUSH 접근·이동은2–5m다. 공동 접근2–10m·이동2–5m와 CARRY 목표 XY 샘플링은 유지한다. 공통 샘플러의 기본값은 유지하고 현재 설정에만 상한을 전달한다.
- 개인 목표 변환을 공동 구성원에게만 적용하여 단독 CARRY AT의 목표 높이 덮어쓰기를 수정했다. 기존 검사의 목표 벡터 비교를 PUSH에 한정하고 단독 접근 범위·AT 목표/받침 높이 검사를 추가했다.
- CPU53개 통과(기존21k 전이1개 제외). GPU6·확률0.5·1024환경 CHECK epoch2/frame262144 저장, finite·encoder/RMS 동결·head 갱신 확인.32환경8preset의 거리·고정 목표·부분 reset 검사를 통과했고 holding_at의 높은 목표61개를 포함해 받침과 목표 높이 일치를 확인했다. [검증](output_etc/check/coupled_individual_solo5/verification.json). 기존 본학습은 재시작하지 않았다.

### 개인 목표 실험의 공동 선택 확률50%

- `stage2_coupled_push_individual_goal`의 `coupled_probability`를0.8→0.5로 변경했다. 그룹 선택 확률 기준이며4명 기대 인원 비율은 공동63.54%·단독36.46%다. 확률 validator와 설정 비교 검사를 갱신했고 기존 coupled/goal5는0.8을 유지한다.
- 관련 CPU44개 통과(기존21k 전이1개 제외), diff 검사 통과. [로그](output_etc/check/coupled_individual_goal/probability50_cpu.log). 아래 GPU 검증은 변경 전0.8 설정의 결과이며 이번에는 학습을 실행·재시작하지 않았다.

### 개인 목표·공통 4면 coupled 실험 연결

- `stage2_coupled_push_individual_goal` 환경/학습 YAML·train/test/VNC/eval을 추가했다. 공동2–5m 이동 벡터를 초기 분할 토큰마다 적용하고 개인 AT 연결·보상·도착 예외·현재 성공 포화를 사용한다. 공유 중심 목표는 물리 경로와 공통 외곽4면 선택에만 사용한다. 목표 방향60° 후보 중 담당 구역까지 그룹 최대 손 거리가 가장 작은 면을 선택하고 전원 gate도 같은 면으로 검사한다. 별도 `aligned` 조건은 제거했다. 기존 coupled/goal5·독립 행동은 유지하고 저장본 계약을 분리했다.
- CPU83개 통과(없는 기본21k 저장본을 요구하는 기존 전이1개 제외). GPU6·1024환경 CHECK epoch2/frame262144 저장·finite·actor encoder73 tensor/입력 RMS 동결·action head6 tensor 갱신 확인.32환경8preset의 거리/개인 목표 연결·고정 목표·부분 reset·정책/AMP finite 검사와4환경32step 실제 렌더를 통과했다. [학습 검사](output_etc/check/coupled_individual_goal/training_validation.json), [장면 검사](output_etc/check/coupled_individual_goal/scene/physics_validation.json), [목표 표시](output_etc/check/coupled_individual_goal/viewer/frame_32.png).
- GPU6·32환경·seed0·1200step 평가를 완료하고 개인/그룹 동시 성공의 ever/final 집계를 연결했다. CPU 재채점·JSON 건수 일치, float32/64 개인/BEFORE 지표 차이0·상태 finite·입력 checkpoint 해시 보존을 확인했다. [재채점](output_etc/check/coupled_individual_goal/eval/general_seed0/rescore.json). config/structure/USAGE/README·상세 설계를 갱신했다. 본학습은 시작하지 않았으며 기존 학습 프로세스는 유지했다.

### PUSH·CARRY AT 목표구 반지름20cm 통일

- 요청에 따라 모든 설정의 AT 목표구를 생성 시 반지름20cm로 통일하고 PUSH 와이어 구를25cm에서20cm로 줄였다. 바닥 링은35cm로 유지하며 reset 중 scale 변경은 추가하지 않았다.
- URDF 시각 반지름0.1m×생성 scale2.0=0.2m·Python 구문·diff 검사를 통과했다. 이번 크기 조정의 viewer 실행 검증은 하지 않았다.

### PUSH viewer 표시 롤백 및 선 표시 재구현

- 사용자가 viewer에서 물체가 날아가는 비정상 동작을 보고해 직전 PUSH 표시 이식 전체를 롤백했다. reset 시 marker scale을 변경하는 코드·표시 모듈·호출부를 제거하고 기존 목표구 표시와 문서를 복구했다. 짧은 관측/렌더 검사만으로 정상 동작을 판단한 기존 검증은 불충분했다. 기존 학습·컨피그 변경은 보존했다.
- 이후 기존 목표구·actor·scale·reset을 유지하고 `add_lines`로 PUSH 목표의 바닥 링·세로선·외곽 와이어 구만 덧그리도록 다시 구현했다. 공유 목표 선은 노란색이며 일반 AT에는 추가 선이 없다. 공통 reset 파일은 변경 전과 동일하다.
- GPU5의 실제 viewer에서5500 정책·4환경·동일 seed의 기존/선 표시 실행을 각각1200step 비교했다. 전체 actor root state·정책 관측·행동이 bitwise 동일하고 최대 차이0·finite다. [궤적 비교](output_etc/check/push_goal_viewer_safe/trajectory_comparison.json). 일반 AT·독립 PUSH·2명 두 그룹·4명 공동의 선 표시만 호출하는 것도 확인했다. [표시 검사](output_etc/check/push_goal_viewer_safe/draw_only_verification.json).

### Coupled goal5 5500 간단 평가

- 최신 정기5500 저장본을 GPU5·32환경·seed0·1200step 첫 episode1회로 평가했다. 공동 PUSH의 실제 상자 목표 도달은2명6/24(25.0%)·3명1/7(14.3%)·4명1/9(11.1%)다. 혼합 장면의 독립 역할은 SIT1/1·CLIMB2/2·AT5/6·ON_TOP5/6·PUSH1/8이며 점/영역 결과는 같다. 사람 번호별 공동 참여 성공은 별도 JSON에 저장했다. [상세](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/breakdown.json).
- CPU 재채점과 GPU·JSON 건수가 일치하고 float32/64 성공 차이0·상태 finite·입력 checkpoint 해시 보존을 확인했다. 낙상1/128명이다. 소표본 검사이며 본학습 프로세스는 유지했다. [검증](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/verification.json).
- 같은 궤적의1200step final은 공동2명4/24·3명0/7·4명0/9, 독립 AT4/6이며 나머지 독립 행동은 ever와 같다. 이전 낙상자를 제외하고 final이 ever에 포함됨을 확인했다. [final 상세](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/final_breakdown.json).

## 2026-10-30

### 공동 PUSH goal5 컨피그 추가

- `stage2_coupled_push_goal5` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. 공동 PUSH는 사람→실제 상자 중심2–10m·상자→목표2–5m, 독립 PUSH는 두 거리2–10m다. 기존 coupled YAML은 원래 설정으로 복구했으며 사람 초기 배치·그 외 환경/학습 값은 유지했다. 새 실험의 기본 전이는 제공26k이고 출력·저장본 계약은 분리했다.
- CPU28개 통과(공통 모델26·새 설정 로딩2, 기본21k 파일이 필요한 기존 전이1개 제외). GPU5의1024환경 CHECK가 epoch2/frame262144 저장 후 정상 종료했다. 모델/정규화/optimizer finite·actor encoder73 tensor와 입력 RMS 동결·action head6 tensor 갱신을 확인했다. [학습 검사](output_etc/check/coupled_goal5/training_validation.json).
- 전용 test wrapper로 저장본을4환경·32step·4명 공동 PUSH로 headless 재로드했다. 셸/Python 구문·새 YAML/문서 경로를 확인하고 config/structure/USAGE/README·공동 PUSH 상세를 갱신했다. 장기 본학습·수렴 성능·실제 GUI/VNC 화면은 검증하지 않았다.

### 모션 데이터 연결 복구

- loco·sit·climb·carry 모션은 `/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/`의 각 원본으로, PUSH 모션은 `/home/hwanhee/ksh/stage1_2_jointpush/tokenhsi/data/jointpush_motions`로 심링크를 연결했다. 외부 원본은 수정하지 않았다.
- 현재 PUSH 모션 YAML의 파일 참조161개가 모두 존재함을 확인했다. 학습은 실행하지 않았다.

### 공동 PUSH 거리2–5m 분석

- 공동 PUSH의 두 XY 거리를2–5m로 줄이는 안을 검사했다.32환경의41개 공동 그룹에 각각104만 후보를 검사하고, 실패4환경은 위치·회전 구간 검사를 통해 고정된 사람·상자 크기에서 거리/뒤쪽/여유 조건을 동시에 만족할 수 없음을 확인했다. [분석](output_etc/check/coupled_distance_analysis/summary.json).
- 공통 Stage2 CPU23개 통과(기본21k 파일이 필요한 전이1개 제외). 제공26k의165 tensor 전이와 입력 RMS 로드를 확인했다. 이후 사람→상자2–10m·상자→목표2–5m 조합은32환경의 혼합/4명 공동/독립 PUSH reset·짧은 시뮬레이션을 통과했다. 사람 초기 배치는 변경하지 않았고 기존 coupled YAML은 두 거리2–10m로 복구했다.
- 최초 검증은 GPU1에서 실행 후 종료했다. 이후 사용자 지시에 따라 GPU5·6만 사용하며 GPU5에서 같은 배치 오류를 재현했다. [로그](output_etc/check/coupled_distance_2to5/scene_gpu5.log), [초기 배치](output_etc/check/coupled_distance_2to5/scene/reset_failure.json).

## 2026-10-29

### PUSH viewer 목표구 확대

- PUSH 포함 설정의 viewer/VNC/영상 목표구를 반지름3cm에서20cm로 확대했다(지름 약6.7배). 공통 marker 생성부의 표시 scale만 변경했으며 목표 좌표·성공 판정·학습 설정과 비PUSH 설정의 marker 크기는 유지했다.
- coupled1환경의 동일 카메라에서 기존/확대 목표구를 offscreen 렌더링해 가시성을 확인했다. marker scale 변경 전후 정책 관측이 bitwise 동일하고 짧은 시뮬레이션 관측/보상 finite를 확인했다. [확대 화면](output_etc/check/push_goal_marker/render/after.png), [검증](output_etc/check/push_goal_marker/render/verification.json). USAGE에 표시 크기를 기록했다.


### PUSH26k 평가와 Stage2 제로샷 검증

- `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`를 검사했다.5과제/4family·165 tensor·모델/RMS finite를 확인했고 두 입력 PTH의 실행 전후 SHA256은 동일하다. 제공78k는 기존 기준78k와 해시가 같고 새 pth 경로의4환경32step headless 재로드를 완료했다.
- 26k Stage1 PUSH는512환경×3seed·900step 일반3회/1명 행동별 단독15회를 완료했다. 점/영역5과제 macro는 일반 **84.55/88.32%**, 단독 **82.20/86.95%**다. PUSH 점 성공률은 일반87.12%·단독88.61%로21k보다 높지만 다른 과제의 하락으로 전체 점 macro는 낮았다.21k와18개 초기 장면의9개 필드가 bitwise 일치한다.
- near50·coupled는26k tensor165개를 그대로 복사하고 공통 coordination identity/Linear bias0인 제로샷 저장본을 각각 만들었다. 추가 학습0회·두 모델 tensor 동일을 확인했다. 각16환경·4명·1200step 실행과4환경32step headless viewer 재로드를 완료했다. BEFORE ON_TOP 쌍/4명 공동 PUSH viewer 명령을 USAGE에 추가했다. 관찰용 파일에는 optimizer가 없고 기본 YAML 전이21k는 변경하지 않았다.
- 총20개 trace의 CPU/GPU·JSON 건수 일치, float32/64 ever/simultaneous 차이0·finite를 확인했다. [리포트](output_etc/check/push_26k/report.md), [검증](output_etc/check/push_26k/verification.json). Stage2 결과는 작은 제로샷 실행 검사이며 학습된 Stage2 성능·실제 GUI 화면 검사는 아니다. 기존 원본/프로세스는 보존했다.


### BEFORE PUSH near50 거리 변형 추가

- `stage2_before_push_near50` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. `stage2_before_push`의 mix226 대신 참고 설정의 `relative_sum` near50 거리만 적용했다. 기존 모든 YAML은 보존하고 공통 모델·PUSH/독립 행동 비율·4명/1024환경/8토큰/1200step·loco·21k 전이는 유지했다. RSI433과 attention mask는 이식하지 않았다.
- near 모드는 BEFORE 쌍의50%이며 SIT/CLIMB는 후행→받침 거리가 선행→받침보다 작고, ON_TOP은 후행→받침+후행→payload 거리 합이 선행→받침보다 작다. 나머지50%는 original 샘플링이다. validator·viewer/eval의 새 이름/모션/task 경로·checkpoint 혼용 거부를 연결했다.
- CPU313개 통과.1024환경 CHECK epoch2/frame262144 저장·모델/RMS/optimizer finite·actor encoder73 tensor/입력 RMS 동결을 확인했다.32환경5preset에서 near 부등식·PUSH 경로/물리·alias 관측·부분 reset·가중치 불변을 확인했고4환경32step headless viewer와16환경×seed0·1200step 일반 평가를 완료했다. CPU/GPU·JSON 건수 일치, float32/64 ever 차이0·finite다. [검증](output_etc/check/before_push_near50/verification.json).
- config/structure/USAGE/README·BEFORE PUSH 상세에 설정과 명령을 갱신했다. 장기 본학습·512환경×3seed 성능·GUI/VNC 실화면은 실행하지 않았다.


### 공통 Stage2 alias 전처리와 coordination 초기화 수정

- 공통 `Stage2PushBuilder`의 alias 처리를 config 플래그 대신 task packet의 BEFORE 의존 관계로 결정하도록 통일했다. encoder용 복사본만 변환하고 복제 물체 token feature/pose·원본 packet·coordination relation은 보존한다. coupled·독립 packet은 그대로 통과한다. 기존4과제 BEFORE 구현은 유지했다.
- 새 coordination의 모든 Linear bias를 원본처럼0으로 초기화했다. 초기 출력 identity와 Stage1 PUSH165 tensor 전이는 유지하며 기존 학습 PTH 로드 시 저장된 bias를 덮어쓰지 않는다. coupled 제로샷 PTH를 같은 경로에 새 초기화로 재생성하고 이전 관찰용 저장본은 `output_etc/check/stage2_common_fix/previous_zeroshot.pth`에 보존했다.
- CPU305개 통과. BEFORE SIT/CLIMB/ON_TOP·두 쌍·coupled·solo의 같은 가중치/입력에서 config에 무관하게 actor/critic 출력 bitwise 일치, 기존 encoder 동작·복제 feature/pose·원본 packet 보존을 확인했다. 실제 GPU BEFORE 관측과 기존 학습 가중치의 교차 실행도 action/critic 최대 차이0이다.
- 두 설정 모두4명·1024환경 CHECK epoch2/frame262144 저장, 모델/RMS/optimizer finite·actor encoder73 tensor/입력 RMS 동결을 확인했다. BEFORE32환경5preset의 alias 관측/부분 reset/finite 검사와 재생성 제로샷4환경·32step headless 재로드를 완료했다. 이전/새 제로샷의 coupled·solo 출력은 동일하다. [검증](output_etc/check/stage2_common_fix/verification.json). 구조/설정 상세 문서를 갱신했고 장기 성능·실제 GUI 화면은 검증하지 않았다.


### Coupled 제로샷 viewer 저장본

- 제공 Stage1 PUSH21k의 기존165 tensor를 그대로 복사하고 공통 Stage2 coordination을 identity로 초기화한 `output_etc/check/coupled_zeroshot/Stage2CoupledPush_ZeroShot_Stage1_21000.pth`를 만들었다. 추가 optimizer step0·tensor 일치·identity 초기화를 확인했다. 원본 PTH는 보존했다.
- `TASK_GRAPH=all_coupled`로4환경·32step headless test wrapper 재로드를 완료했다. 실제 GUI 화면은 실행하지 않았다. 제로샷 viewer 명령을 USAGE에 추가했고 전이 기록은 `output_etc/check/coupled_zeroshot/transfer.json`에 있다.


### 공통 Stage2 PUSH 모델과 coupled 이식

- `stage2_coupled_push` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. 원격 `stage1_2_jointpush_st`의 coupled straight·밀도3·인원별 공동 상자/구역·그룹 gate·goal saturation을 이식했다. 독립 행동과 AMP 혼합은 현재 Stage1 PUSH 설정(AT25/ON_TOP20/SIT10/CLIMB15/PUSH30)을 읽도록 바꾸고 미사용 Stage2 RSI/adapter 스택은 가져오지 않았다.
- `stage2_before_push`와 coupled는 동일 `Stage2PushBuilder`·관측1272/AMP1330·5과제·NONE/SELF/BEFORE/COUPLED4종 사람 relation·6N task/group suffix를 사용한다. tensor 이름·크기가 같고 Stage1 PUSH21k의165 tensor를 그대로 전이한다. 기존 BEFORE PUSH의5N 관측 검사 저장본은 새 모델에 resume하지 않는다.
- 사용자 기준대로 Stage1은900step을 유지하고 모든 Stage2 학습/일반 평가/viewer 기본 시간을1200step으로 변경했다. 기존4과제 BEFORE 모델은 보존했다. coupled 평가의 PUSH 성공은 가상 구역이 아닌 실제 전체 상자 중심에서 계산한다.
- CPU292개 통과. 두 Stage2 모두4명·1024환경 CHECK가 epoch2/frame262144 저장 후 종료했고 coupled resume은 epoch3/frame393216까지 완료했다. actor encoder73 tensor/입력 RMS 동결·action head6 tensor 갱신·모델/RMS/optimizer finite를 확인했다.
- coupled32환경8preset·BEFORE32환경5preset의 배치/구역/AMP/부분 reset/가중치 불변을 확인했다. 두 설정 모두 일반32환경×seed0·1200step 및 headless viewer4환경·32step 재로드를 완료했다. CPU/GPU·JSON 건수 일치, float32/64 ever/simultaneous 차이0·finite다. [검증 보고](output_etc/check/coupled_integration/report.md), [전체 결과](output_etc/check/coupled_integration/verification.json).
- config/structure/USAGE/README·BEFORE 상세와 [공통 Stage2/coupled 상세](markdowns/reference/stage2_coupled_push.md)를 갱신했다. Stage1 YAML·원본 checkout·외부 PTH·기존 프로세스는 보존했다. 장기 본학습·정규512환경×3seed 성능·GUI/VNC 실화면은 검증하지 않았다.


### Stage2 BEFORE solo PUSH 추가

- `stage2_before_push` 환경/학습 YAML과 전용 train/test/VNC/일반 eval을 추가했다. 독립 역할만 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%를 사용한다. 기존 0.8 쌍 구성·선행 AT·크기 호환 후행 SIT/CLIMB/ON_TOP 균등 선택·mix226·공유 alias·AT gate/progress는 유지한다. 기본4명·1024환경·8토큰·900step·모두 loco다.
- PUSH 전용 큰 상자 bank를 추가해 기본20개 물리 후보에서 배정하며 일반 distractor/BEFORE 받침으로 섞이지 않게 했다. Stage1 PUSH의 밀도3·167종·cone90·거리·접촉 가중치를 적용하고 다중 사람 경로/활성 장애물 검사를 추가했다. 정책5과제·AMP4family를 기존 BEFORE coordination에 연결해 기본21k PUSH에서 tensor165개를 그대로 전이한다. 기존5개 환경/학습 YAML은 변경하지 않았다.
- CPU280개 통과. 최종 CHECK1024환경은 epoch2/frame262144, resume은 epoch3/frame393216 저장을 완료했다. actor encoder73 tensor·입력 RMS 동결, action head6 tensor 갱신, 모델/정규화/optimizer finite를 확인했다. [학습·재개](output_etc/check/before_push/training_validation.json).
- 32환경7개 preset에서 물체 alias/숨김·거리·밀도/마찰·PUSH 경로·AT progress·AMP·부분 reset·가중치 불변을 확인했다. 최종 저장본 일반32환경×seed0·900step과 headless viewer4환경·32step 재로드를 완료했다. 평가 CPU/GPU/JSON 건수 일치·float32/64 ever 차이0·finite를 확인했다. [검증 보고](output_etc/check/before_push/report.md), [전체 검증](output_etc/check/before_push/verification.json). 이는 실행 검사이며 장기 학습 성능·512환경×3seed 평가·GUI/VNC 실화면은 검증하지 않았다.
- config/structure/USAGE/README와 [설정 상세](markdowns/reference/stage2_before_push.md)를 갱신했다. 원본 checkout·외부 checkpoint·기존 실행 프로세스는 보존했다.


### Stage1 PUSH 행동 비율 변경

- 사용자 지정대로 `stage1_push`의 새 학습 크기/행동 prior를 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%로 변경했다. 검증기의 허용 계약·샘플링 테스트·현재 PUSH/USAGE/평가 문서를 갱신했다. 과거 저장본의 분포와 평가 결과는 보존하며21k PTH의 기본 viewer/일반 평가는 저장된PUSH50%를 복원한다.
- CPU35개 통과:8만 사람 슬롯의 크기 조건부 샘플링 빈도·합계1·설정 검증·이전 저장본 복원/호환. [검증](output_etc/check/push_sampling_30/tests.log), [설정](output_etc/check/push_sampling_30/config_check.json). 비율 변경 검증에 GPU 학습/평가는 실행하지 않았다.


### Stage1 PUSH 통합과900step 통일

- `stage1_push` 환경/학습 YAML·train/test/VNC·일반/1명 단독 eval을 추가했다. scratch prior는 SIT7.5/CLIMB18.75/AT24.375/ON_TOP24.375/PUSH25%로 기존 네 과제 상대 비율을 유지한다. 정책은5과제·AMP4family로 확장하고 기존4과제/3family 기본값은 보존했다. PUSH167종·밀도3·cone90·접촉 RSI bank/모션을 이식했으며 모션21개와21k PTH는 외부 원본에 심링크로 연결했다.
- 학습·일반/단독 평가·viewer 기본 시간을900step으로 통일하고 시간 종료의 `-1` 경계를 수정했다. 새5과제 macro와 기존4과제 macro를 구분한다. PTH 옆 sidecar가 없으면 내부 저장 env/sim을 읽는다. 제공21k 일반 평가는 학습 당시 PUSH50% 분포를 복원하고 새 scratch는PUSH25%다. 서로 다른 학습 계약의 resume는 거부한다.1명 PUSH와1명 ON_TOP의 물체/받침 인덱스, 여섯 번째 RSI 통계 축을 일반화했다.
- 제공 `Stage1JointPushDensity3Cone90Finetune_00021000.pth`로512환경×3seed 일반3회·단독15회, 모두900step을 완료했다.5과제 macro 점/영역은 일반 **85.84±1.07 / 88.80±0.81%**, 단독 **83.32±0.51 / 88.78±0.32%**다. PUSH는 일반 **81.78±1.93%**, 단독 **81.51±1.47%**다.18개 trace CPU/GPU·JSON 건수 일치, float32/64 ever 차이0, 상태 finite를 확인했다. [리포트](output_etc/check/push_900/report.md), [검증](output_etc/check/push_900/verification.json).
- 이전78k1200step trace의 앞900frame을 별도로 재채점해 같은 제한 시간의 비교를 추가했다. 기존 원본 trace/보고서는 보존했다. 기존 네 과제의 단독 평균은21k에서 낮아졌으며 수치와 비교 조건은 리포트에 기록했다.
- CPU259개 통과. 실제78k→PUSH 전이 후 기존 네 과제 actor/critic 출력 보존, 원본 브랜치 정책·task encoder·clean scene 관측 코드 일치를 확인했다. Stage1/BEFORE 기존 PTH4환경·32step 재로드도 통과했다. PUSH scratch CHECK는2048환경으로 RSI5517 profile 준비 후 epoch2/frame262144를 저장했고 resume은 epoch3/frame393216까지 진행했다. 모델/RMS finite, scratch 시작·PUSH25%·900step 계약을 확인했다. [학습/재개](output_etc/check/push_900/training_validation.json). 원본 PTH/코드/기존 프로세스는 보존했으며 장기 본학습·GUI/VNC 실화면은 실행하지 않았다.


### 일반 평가와 Stage1 전용 단독 평가 분리

- 일반 `*_eval.sh`는 checkpoint 저장본의 학습 인원·물체·크기·과제/BEFORE 배치를 유지한다. Stage1만 `stage1_single_eval.sh`로1명·행동별 독립 평가를 제공한다. 단독 AT만 기존9종 크기, 나머지는 학습 크기/동적 물리를 따른다. viewer native 성공 진단은 `viewer_diagnostics.py`로 이름과 결과 태그를 분리했다.
- 정식 평가는 loco·정확히1200step·독립 seed0/1/2×512환경 첫 episode, 같은 궤적의 점/영역 ever·seed 평균/표본 std·macro다. 개인 낙상 frame까지 채점하고 해당 관절 drive를 해제하며 다른 사람은 계속 수행한다. BEFORE의 개인/follower 개인·둘 다 ever·동시 성공을 별도 집계한다. 학습 reward/종료 조건은 유지했다. viewer 기본 horizon도1200으로 맞췄다.
- Stage1 원본78000으로 일반3회(2명·4물체), 단독12회(행동별1명)를 완료했다. macro 점/영역은 일반 **95.17±0.84 / 97.17±0.18%**, 단독 **96.58±0.26 / 98.83±0.40%**다. [리포트](output_etc/check/eval_unified/stage1_78k_report.md). 15개 trace CPU float64/GPU·JSON 건수 및 float32/64 ever가 전부 일치했고 상태 finite·원본 checkpoint 해시 보존을 확인했다.
- CPU242개 통과. BEFORE 세 설정은 검사 저장본으로16환경·4명·1200step 실행/재채점(낙상8/10/15명, 각각24pair), viewer는4환경·32step의 새 진단 경로를 확인했다. Stage2 본학습 성능과 GUI/VNC 화면은 평가하지 않았다. [검증](output_etc/check/eval_unified/verification.json), [BEFORE](output_etc/check/eval_unified/before_validation.json). 새 규약·명령·코드 위치를 evaluation_single/config/structure/USAGE에 갱신하고 이전 단독 규약은 legacy에 보존했다.


### 네 실험 이름 통일

- 환경/학습 YAML·전용 train/test/VNC12개·실험/출력 이름·테스트·현재 문서를 `stage1`, `stage2_before_nonemask_rsi433_near50`, `stage2_before`, `stage2_before_rsi433`로 통일했다. Stage1 보조 구현은 `stage1_config.py`, `stage1_geometry.py`로 바꿨으며 기본 전이 링크는 `checkpoints/Stage1_00078000.pth`다. 외부 원본 파일과 과거 결과 경로는 보존했다.
- 기존78000 및 이전 BEFORE checkpoint의 reward 태그·RSI 경로를 읽기 경계에서 변환한다. 보상·관측·모델 tensor·실험 간 혼용 거부는 유지한다. 기존 네 저장본을 새 test wrapper로4환경·32step headless 재로드해 정상 종료했고 네 모델의 전체 tensor·actor/critic 출력은 변경 전과 bitwise 동일했다. [재로드](output_etc/check/config_rename/smoke.json), [모델](output_etc/check/config_rename/model_equivalence.log). CPU224개 통과. 셸/Python/YAML·문서 링크·이전 실행 이름 잔여 검사 결과는 `output_etc/check/config_rename/`에 있다. GUI/VNC 실화면과 본학습은 실행하지 않았다.

### 네 컨피그와 학습·추론 경로 정리

- `stage1_v2_fixed`, `stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200`, `stage2_before_ground_mix226_atprogress_objalias_1200`, `stage2_before_ground_mix226_atprogress_objalias_rsi433_1200`만 유지했다. 나머지21쌍 YAML·63개 wrapper와 공동 운반·공유 받침·adapter·full RSI·carry50·단일인 태스크·전처리·미사용 에셋/모듈을 제거했다. 기존 파일322개를 삭제했으며 과거 문서는 legacy에 보존했다. [변경 목록](output_etc/check/four_config_cleanup/change_manifest.json).
- BEFORE의 공통 기하·배치·물리 검사·native 후행 RSI를 분리하고 모델을 공통 정책의 직접 확장으로 정리했다. 공통 환경의 관측·reset·모션/AMP를 별도 mixin으로 분리했다. 파라미터 이름·관측·보상·거리·초기화·동결/학습 범위를 유지했다.
- 전용12개 wrapper는 공통 `launch.sh`를 사용한다. conda 기본값은 모두 `tokenhsi`, seed는42, test 기본은 viewer다. GPU는 한 번 지정하며 기존 MPS만 연결한다. 설정 로더는 새 학습의 native YAML과 추론/resume의 저장 env/sim을 같은 경로에서 처리한다. `ENV_CONFIG`로 명시적 대체가 가능하며 fixed 임시 YAML 생성·실험 이름별 중복 분기를 제거했다.
- Stage1 78000은 `checkpoints/Stage1V2Fixed_00078000.pth` 심링크로 연결했다. 네 모션 데이터 디렉터리도 공유 원본에 연결했다. 원본 checkout·가중치·기존 학습 PID3432492는 보존했다. 확인 산출물·RSI 캐시·pytest 임시 파일은 `output_etc/check/`에 둔다.
- CPU214개 통과: 네 설정/계약 혼용 거부·실제78000 전이·설정 복원·wrapper/conda 선택·GTA·보상·거리·mask·RSI/fallback·재로드. [CPU](output_etc/check/four_config_cleanup/cpu.log). 네 모델의 전체 tensor·actor/critic 출력·계약이 정리 전과 bitwise 동일했다. [모델](output_etc/check/four_config_cleanup/model_equivalence.log). 네 설정의 전체 reset2회·부분 reset1회에서 관측·사람/상자 상태·크기·목표·물체 연결의 최대 차이0을 확인했다. [reset](output_etc/check/four_config_cleanup/reset_equivalence.json). 후행 RSI sampler의 출력·RNG도 동일했다.
- GPU0/MPS에서 fixed2명·2048환경과 BEFORE4명·1024환경의 CHECK가 각각 epoch2/frame262144 저장 후 종료했다. fixed native RSI6413 profile을 생성했다. BEFORE 세 설정은 source78000·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/scalar finite를 확인했다. mix226 resume은 epoch3 저장 후 종료했다. [학습/재개](output_etc/check/four_config_cleanup/training_validation.json).
- 실제78000 fixed headless 추론과 BEFORE 전용 test의4환경·32step 저장본 재로드를 확인했다. BEFORE32환경의 loco/mix·ON_TOP pickup/carry·단독 등14개 구성에서 alias 관측/pose·활성 물리 수·거리·AT progress·AMP·부분 reset·가중치 불변을 확인했다. [물리 검사 진행 결과](output_etc/check/four_config_cleanup/physics.log), [wrapper](output_etc/check/four_config_cleanup/wrapper_validation.json). 셸/Python/YAML·현재 문서 링크·4쌍 YAML/12개 wrapper·미사용 import/실험 참조 제거를 검사했다. VNC 실행 의존성이 없어 실제 GUI/VNC 화면과 장기 성능은 검증하지 않았다. 본학습은 시작하지 않았다.

## 2026-10-28

- 코드 정리용 독립 복사본을 만들었다. 원본과 복사 범위·초기 검증은 `code_copy_manifest.json`과 [당시 changelog](markdowns/legacy/pre_four_configs/changelog.md)에 보존한다.
