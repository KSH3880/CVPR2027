# Changelog

최신 변경부터 기록한다. 현재 실행법은 [config](markdowns/config.md), 코드 위치는 [structure](markdowns/structure.md)를 따른다. 과거 변경·검증 전체는 [보관 changelog](markdowns/legacy/changelog_before_cleanup.md)에 있다.

## 2026-10-29

### near50 objalias loco-only·SELF/NONE bias0

- `stage2_before_ground_atprogress_bias0_near50_objalias_1200` env/train YAML·전용 train/test/VNC를 추가했다. mix226 objalias 기반에서 거리만 original/relative-sum50/50으로 바꾸고 actor/critic은 `before_only`로 설정했다. SELF/NONE bias는 파라미터 없이0, BEFORE bias만 학습하며 NONE attention은 열어둔다. 전원 loco·역할별 물체 토큰·AT progress·1200step·2e-5·fixed78000 새 전이·encoder/RMS 동결·action head 학습을 유지한다. relative 거리의 RSI 필수 조건을 해제하고 기존 masked RSI433 near50와 새 loco 설정의 train/env·checkpoint 계약을 분리했다. [설계](markdowns/reference/stage2_before.md#near50-역할별-object-토큰과-selfnone-bias0), [명령](markdowns/config.md).
- CPU 기존42개와 신규3개 통과: 거리/초기화 독립성·양쪽 고정0/BEFORE 및 QKV gradient·NONE 연결 유지·alias·전이/동결/재로드·checkpoint 분리. 신규 검사의 첫 step gradient 가정은 기존 coordination의 zero-init output projection에 맞춰 수정 후 통과했다. [회귀](output_etc/check/before_bias0_near50_objalias_20261029/cpu.log), [신규 최종](output_etc/check/before_bias0_near50_objalias_20261029/cpu_new.log), [train/env13조합](output_etc/check/before_bias0_near50_objalias_20261029/config_contract.json).
- GPU0/MPS·4명/1024환경/8토큰 CHECK epoch2/frame262144 정상 종료. source78000·actor encoder73 tensor/RMS 불변·head6 및 BEFORE bias8 tensor(2×1) 갱신·모델/scalar542개 finite·기록된 물리 최종 실패0을 확인했다. [학습 검증](output_etc/check/before_bias0_near50_objalias_20261029/training_validation.json).
- 32환경 random/ON_TOP 두 쌍/한 쌍/단독 및 별도 SIT/CLIMB에서 near50 조건·전원 loco·관측/pose 공유·활성 물리 수·BEFORE/AT progress·AMP·부분 reset·가중치 불변을 확인했다. SIT/CLIMB 최초 검사는 결과 폴더 누락으로 저장 단계에서 실패해 폴더 생성 후 재검사했다. [기본4구성](output_etc/check/before_bias0_near50_objalias_20261029/eval/physics_validation.json), [SIT](output_etc/check/before_bias0_near50_objalias_20261029/sit/physics_validation.json), [CLIMB](output_etc/check/before_bias0_near50_objalias_20261029/climb/physics_validation.json). 전용 test의4환경·32step·near100% headless 재로드, 셸/Python/YAML·문서 경로/순서·diff 확인. [wrapper](output_etc/check/before_bias0_near50_objalias_20261029/wrapper.log). 새 본학습·GUI/VNC 실화면·장기 성능은 미검증이며 기존 학습은 유지했다.

### near50 objalias9000 원래 배치4명 평가

- 최신 고정 저장본9000(epoch9000/frame1179648000)을 원래 near50 배치·4명/8obj 슬롯·행동별32환경·seed42·loco·1200step·결정적 action으로 BEFORE3종/단독4종 평가했다. 세트 분리 hook 없이 기본64회 reset 예산을 사용했다. checkpoint finite·7개 exit0·유한 action·loco/분모·7000 평가와 초기 사람/상자 상태1e-4 이내 일치·셸/Python/diff를 확인했다. 운영 코드/config와 기존 학습은 유지했다.
- 한 번 이상 성공률은 BEFORE SIT21/64(32.81%), CLIMB17/64(26.56%), ON_TOP4/64(6.25%); 단독 SIT118/128(92.19%), CLIMB122/128(95.31%), AT89/128(69.53%), ON_TOP95/128(74.22%)였다. BEFORE ON_TOP 후행 집기60/64(93.75%)·선행 AT41/64(64.06%)·gate 제외 쌓기5/64(7.81%)다. 같은7000 표본보다 단독 AT는 상승, 단독 ON_TOP은 하락, BEFORE ON_TOP 완수는 동일했으며 단일 seed/32장면 결과다. [조건·7000 비교·상세 결과](output_etc/check/near50_objalias9000_n4_eval32_20261029/summary.json), 재현은 같은 폴더의 case `.sh`/`run_all.py`다.

### 세트 간 거리 분리 평가: 충돌 원인 분리의 한계

- 같은 near50 objalias7000·4/6/8명·행동별32환경·loco·1200step·seed42로 BEFORE3종/단독4종 총21구성을 다시 평가했다. 초기 샘플을 그대로 만든 뒤 단독 또는 BEFORE 연결 성분의 사람·과제 물체·목표를 함께 평행이동해 세트별 초기 bounding box 사이에 최소10m 여유를 두었다. distractor는 원래 가장 가까운 사람의 세트에 배정했다. 평가용 hook만 사용하며 물리 검사·초기 재시도 예산은 앞 평가와 동일하다.
- 원래 초기 사람/상자 상태가 앞 평가와1e-4 이내로 일치하고, 세트 내부 위치·회전·속도 및 native 과제 거리/progress/높이/영역 판정의 보존·물리 유효성을 검사했다. 모든 실행에서 서로 다른 세트 사람의 root 거리0.7m 미만 관측은0이었다. 이는 근접 지표이며 실제 contact pair 측정은 아니다. [조건·대응 비교·검증](output_etc/check/near50_objalias7000_separated_eval32_20261029/comparison.json), 재현은 같은 폴더의 case `.sh`/`run_all.py`다.
- 분리 후 성공률(4/6/8명 순): BEFORE SIT20.31/9.38/3.12%, CLIMB6.25/1.04/0%, ON_TOP3.12/0/0%; 단독 SIT81.25/28.12/3.12%, CLIMB86.72/6.77/0%, AT35.94/0/0%, ON_TOP66.41/1.04/0%였다. 전반적으로 조기 종료가 증가했고8명 단독은 모든32장면이 조기 종료했다. 거리를 벌리면 관측상의 세트 간 기하도 바뀌므로 이 결과는 충돌만 차단한 대조군이 아니며, 원래 인원 증가 성능 저하의 원인을 Stage1/Stage2/충돌 중 하나로 확정하지 않는다. 배치를 유지한 세트 간 접촉 차단 대조는 미실시다.
- 21개 exit0·유한 action·초기 샘플 대응·기하/물리 검사·문서 diff 확인. 초기8명 SIT 예비 실행은 `pilot_before_sit_n8/`에 보존하고 최종21구성과 구분했다. 운영 코드/config와 기존 학습은 변경하지 않았다.

### 4/6/8명 BEFORE·단독 행동별 일반화 평가

- near50 objalias7000을 동일 저장본·seed42·loco·1200step·행동별32환경으로4/6/8명 각각 평가했다. BEFORE SIT/CLIMB/ON_TOP과 단독 SIT/CLIMB/AT/ON_TOP 총21구성을 완료했다. obj 슬롯은2N이며 BEFORE는 N/2쌍, 단독은 N개 과제다. 초기 배치는 인원마다 달라지며 조기 장면 종료도 성공률 분모에 포함했다. [조건·지표·scene bootstrap 구간](output_etc/check/near50_objalias7000_scaling_eval32_20261029/summary.json), 같은 폴더의 각 case `.sh` 및 `run_all.py`로 재현/완료 결과 재사용이 가능하다.
- 한 번 이상 성공률(4/6/8명 순): BEFORE SIT29.69/34.38/11.72%, CLIMB25.00/6.25/7.03%, ON_TOP6.25/6.25/2.34%; 단독 SIT89.06/90.62/81.25%, CLIMB95.31/95.31/92.97%, AT62.50/54.17/46.09%, ON_TOP82.03/66.67/55.08%다. BEFORE는 선행 AT와 후행 목표 동시 성립이며, raw·집기·종료 시 성공·장면 조기 종료는 상세 JSON에 분리했다. 단일 seed/32장면 결과이며, 인원 증가에 따른 모델 입력·장면 배치·조기 종료 영향을 분리한 실험은 아니다.
- 8명 단독 AT는 초기32환경 중5개가 기본64회 물리 배치 한도를 소진해 첫 실행이 중단됐다. 해당 실패 로그를 보존하고 평가 프로세스에서만 원래 graph·후보분포·물리 검사 조건을 유지한 채 재시도 상한을1024로 확장했다. 8명 단독 AT 재실행 및 남은4/6/8명 단독 ON_TOP에만 적용했다. 최종21개 모두 exit0·loco 확인·finite action·인원별 분모·명령/결과 경로·셸/Python 문법·diff를 확인했다. 원본 학습 코드/config와 기존 학습 프로세스는 변경하지 않았다.

### near50 objalias 가변 인원 viewer 명령

- USAGE에7000 저장본·8명/1환경/16obj 슬롯·random_scenario·loco·near50 기본 혼합의 로컬 viewer와 VNC 전환 예시를 추가하고 config 추론 옵션에 연결했다. checkpoint/스크립트 경로·셸 문법·문서 diff를 확인했다. 명령 안내만 수행했으며8명 GUI 실행은 하지 않았다.

### Stage2 사람 attention relation별 QK/bias 진단

- near50 objalias7000을 SIT/CLIMB/ON_TOP 각각32환경·loco·1200step·seed42로 평가하며10step마다 actor/critic 사람 SA4×2head를 수집했다. 실제 layer 입력과 projection으로 `QKᵀ/√d`, bias 추가, NONE mask 전후 softmax를 각각 계산했다. relation별 softmax 확률을 먼저 구한 뒤 합산/평균했으며 후행·선행 query와 head/layer/행동을 분리했다. 임시 hook은 check 폴더에만 두었다.
- 후행 actor의 SELF/NONE/BEFORE 확률 질량은 QK만27.38/50.46/22.16%, bias 추가27.22/50.10/22.69%, mask만56.85/0/43.15%, 실제 bias+mask56.27/0/43.73%였다. NONE은 두 key의 합이다. critic 실제 확률은48.80/0/51.20%다. 선행은 BEFORE key가 없어 실제 SELF100%다. 동일 입력에서 bias 효과를 계산한 진단이며 mask/bias 제거 정책을 별도로 rollout한 결과는 아니다.
- PyTorch 실제 attention 가중치와 재구성 최대 오차1.73e-6·모델 가중치 불변·NONE 확률0을 확인했다. [상세 지표·layer/head bias](output_etc/check/stage2_attention7000_20261029/summary.json), 재현은 같은 폴더의 `sit.sh`·`climb.sh`·`ontop.sh`다. 기존 학습과 운영 코드/config는 유지했다.

### near50 objalias6500 행동별 loco 평가

- 최신 고정 저장본6500(epoch6500/frame851968000)을 GPU0·4명/8토큰·행동별32환경/64 BEFORE쌍·seed42·1200step·결정적 action으로 평가했다. SIT/CLIMB/ON_TOP을 각각 크기 조건에 맞는 별도 환경으로 생성하고 전원 loco·RSI 비활성을 확인했다. 저장된 near50 거리 혼합을 유지했으며 기존 학습/운영 코드는 변경하지 않았다.
- 한 번 이상 성공한 수는 SIT: 선행 집기58/64·AT38/64·후행 raw28/64·BEFORE20/64, CLIMB:56/64·32/64·22/64·16/64, ON_TOP:60/64·41/64·6/64·4/64(후행 집기59/64)였다. 종료 시 BEFORE 성공은 각각15/64·15/64·3/64다. HOLDING은 기존 손 거리 판정, raw는 선행 AT gate 제외, BEFORE는 선행 AT와 동시 성립이다. 단일 seed 소표본이며 조기 종료도 분모에 포함한다. [조건·결과](output_etc/check/near50_objalias_actions_loco_eval32_20261029/summary.json), 재현 명령은 같은 폴더의 `sit.sh`·`climb.sh`·`ontop.sh`에 있다.

## 2026-10-28

### 500epoch 후행 집기·목표 평가 비교

- 사용자 지정 기존 `stage1_stage2_before` mix226 AT-progress500과 현재 near50/nonemask/RSI433/objalias500을 GPU0·32환경·4명·BEFORE ON_TOP64쌍·seed42·loco·1200step·결정적 action으로 평가했다. 두 저장본 모두 epoch500/frame65536000·Stage1 source78000이다. 기존 학습·viewer와 원본 checkpoint는 유지했다.
- 에피소드 중 한 번 이상 성공 기준, 현재/기존 순서로 선행 HOLDING52/64·53/64, 후행 HOLDING54/64·2/64, 선행 AT34/64·40/64, 선행 AT gate를 포함한 후행 ON_TOP2/64·0/64였다. gate 제외 ON_TOP 기하 성공은15/64·0/64, 종료 시 gated ON_TOP은1/64·0/64다. HOLDING은 기존 손 거리 판정이며 추가 들기 높이/유지시간 기준은 없다. 거리 분포·사람 attention·학습 RSI도 달라 토큰 분리 단독 효과로 해석하지 않는다. [조건·지표·결과](output_etc/check/near50_alias_epoch500_eval32_20261028/comparison.json), [재현 명령](output_etc/check/near50_alias_epoch500_eval32_20261028/rerun.sh). 임시 평가 코드는 check 폴더에만 두며 운영 코드·config는 변경하지 않았다.

### mix226 역할별 object 토큰 후행 RSI433 ablation

- `stage2_before_ground_mix226_atprogress_objalias_rsi433_1200` env/train YAML·전용 train/test/VNC를 추가했다. mix226 objalias에서 BEFORE ON_TOP 후행만 loco40%/pickUp30%/carryWith30%를 요청한다. 선행·단독·SIT/CLIMB는 loco이며 native 유효 프레임·8회 배치 재시도·loco fallback은 기존 구현을 사용한다. 거리20/20/60·토큰 분리·전체 SELF/NONE/BEFORE bias·AT progress·1200step·2e-5·fixed78000 전이를 유지한다. [설계](markdowns/reference/stage2_before.md#at-progress-mix226-object-토큰-rsi433), [명령](markdowns/config.md).
- CPU63개 통과(기존30500 파일 의존 전이1개 제외; 자체 fixture 및 실제78000 전이 통과). 초기화만 바뀌는 설정·분포/대상 제한·물리 ID/encoder 분리·동결/갱신·재로드·checkpoint 혼용 거부와 기존 회귀를 확인했다. [CPU](output_etc/check/before_mix226_objalias_rsi433_20261028/cpu.log), [train/env7조합](output_etc/check/before_mix226_objalias_rsi433_20261028/config_contract.json). GPU0/MPS·4명/1024환경/8토큰 CHECK epoch2/frame262144에서 source78000·encoder73 tensor/RMS 불변·head6 및 전체 관계 bias8 tensor 갱신·모델/scalar572개 finite·기록된 물리 최종 실패0을 확인했다. [학습](output_etc/check/before_mix226_objalias_rsi433_20261028/training_validation.json).
- 32환경 random loco/mix·ON_TOP pickup/carry·단독 및 별도 SIT/CLIMB에서 mix226 거리·RSI 상대 자세/적용 대상·관측/pose 공유·활성 물리 수·BEFORE/AT progress·AMP·부분 reset·가중치 불변을 확인했다. ON_TOP pickup은64/64 적용, carry는56/64 적용·8개 loco fallback이었다. [기본5구성](output_etc/check/before_mix226_objalias_rsi433_20261028/eval/physics_validation.json), [SIT](output_etc/check/before_mix226_objalias_rsi433_20261028/sit/physics_validation.json), [CLIMB](output_etc/check/before_mix226_objalias_rsi433_20261028/climb/physics_validation.json). 전용 test의4환경·32step·short/mix headless 재로드, 셸/Python/YAML·문서 경로/목록 순서·diff 확인. [wrapper](output_etc/check/before_mix226_objalias_rsi433_20261028/wrapper.log). 기존 학습을 유지했으며 새 본학습과 GUI/VNC 실화면·장기 성능은 실행/검증하지 않았다.

### 기본 AT progress mix226 역할별 object 토큰

- `stage2_before_ground_mix226_atprogress_objalias_1200` env/train YAML·전용 train/test/VNC를 추가했다. 기본 `stage2_before_ground_mix226_atprogress_1200`에 `before_object_alias: replace_distractor_v1`만 추가하며 거리20/20/60·loco 초기화·AT progress·ground-only·1200step·2e-5·SELF/NONE/BEFORE 전체 bias 학습을 유지한다. 같은 fixed78000에서 새 전이하고, 고정 obj 토큰 내 distractor 대체·물리 공유·encoder 후행 target 분리·원래 사람 SA/보상 관계를 기존 objalias 구현으로 처리한다. near50/nonemask/RSI433는 포함하지 않으며 기존 AT progress 및 near50 objalias와 train/env/checkpoint 혼용을 거부한다. [설계](markdowns/reference/stage2_before.md#at-progress-mix226-역할별-object-토큰), [명령](markdowns/config.md).
- CPU60개 통과(기존30500 파일 의존 전이1개 제외; 자체 fixture 및 실제78000 전이 통과). 두 기반별 config 차이·단독 출력 동일·actor/critic 분리/전이/동결/갱신/재로드와 기존 BEFORE 회귀를 확인했다. [CPU](output_etc/check/before_mix226_objalias_20261028/cpu.log), [train/env 계약](output_etc/check/before_mix226_objalias_20261028/config_contract.json). GPU0/MPS·4명/1024환경/8토큰 CHECK epoch2/frame262144에서 source78000·encoder73 tensor/RMS 불변·head6 tensor 및 전체 관계 bias8 tensor 갱신·모델/scalar548개 finite·기록된 물리 최종 실패0을 확인했다. [학습](output_etc/check/before_mix226_objalias_20261028/training_validation.json).
- 32환경 random/ON_TOP 두 쌍/한 쌍/단독 및 전용 크기 SIT/CLIMB에서 원본 mix226 접근·운반·목표 거리 범위·loco·동일 관측/pose·8토큰/물리8·7·6개·BEFORE/AT progress·AMP·부분 reset·가중치 불변을 확인했다. [기본4구성](output_etc/check/before_mix226_objalias_20261028/eval/physics_validation.json), [SIT](output_etc/check/before_mix226_objalias_20261028/sit/physics_validation.json), [CLIMB](output_etc/check/before_mix226_objalias_20261028/climb/physics_validation.json). 전용 test의 4환경·32step·short 모드 headless 재로드와 셸/Python/YAML·문서 경로/순서·diff 확인. [wrapper](output_etc/check/before_mix226_objalias_20261028/wrapper.log). 새 본학습과 GUI/VNC 실화면·장기 성능은 실행/검증하지 않았다.

### BEFORE 공유 상자의 역할별 object 토큰

- `stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200` env/train YAML·전용 train/test/VNC를 추가했다. 같은 물리 상자의 선행 payload/후행 target 표현을 분리하기 위해 후행 actor마다 distractor 슬롯 하나를 공유 상자 토큰으로 대체한다. 토큰 수는 고정하고 대체된 distractor는 물리 장면에서도 제외한다. actor/critic encoder용 packet에서만 후행 target을 변경하며 사람 SA의 BEFORE·보상·gate·RSI는 원래 packet/물리 ID를 사용한다. 새 Stage1 mask/파라미터 없이 near50·RSI433·AT progress·1200step·2e-5·fixed78000 전이/동결 정책을 유지한다. `objects >= 2*agents`와 별도 checkpoint/train-env 계약을 검사한다. [설계](markdowns/reference/stage2_before.md#before-역할별-object-토큰), [명령](markdowns/config.md).
- CPU57개 통과(기존30500 파일 의존 전이 검사1개 제외; 자체 fixture 및 실제78000 전이 별도 통과): 역할별 정확한 task 연결·기존 BEFORE 보존·8토큰/6물체·1/2/3/4/6명 예산·순서 변경·단독 출력 동일·actor/critic 동결/갱신·재로드·기존 BEFORE/near50/nonemask/RSI433 회귀. [CPU](output_etc/check/before_objalias_20261028/cpu.log). train/env 혼용 양방향 거부도 확인했다. [설정 계약](output_etc/check/before_objalias_20261028/config_contract.json).
- GPU0/MPS·4명/1024환경/8토큰 CHECK가 epoch2/frame262144 저장 후 정상 종료했다. source78000·encoder73 tensor/RMS 불변·head6 tensor 및 양쪽 BEFORE bias8 tensor 갱신·모델/scalar566개 finite·기록된 물리 최종 실패0을 확인했다. [학습 검증](output_etc/check/before_objalias_20261028/training_validation.json). 기존 검사 캐시를 확인 폴더로 복사해 크기별 native screening 결과를 재사용했다.
- 32환경의 random loco/mix·ON_TOP pickup/carry·단독과 전용 크기로 생성한 SIT/CLIMB에서 물리 활성 수·동일 관측/pose·BEFORE/AT progress·RSI 상대 자세·AMP·부분 reset·가중치 불변을 확인했다. [기본5구성](output_etc/check/before_objalias_20261028/eval/physics_validation.json), [SIT](output_etc/check/before_objalias_20261028/sit/physics_validation.json), [CLIMB](output_etc/check/before_objalias_20261028/climb/physics_validation.json). 최초 일반 크기 환경에서 강제 SIT preset 검사는 크기 부적합으로 중단되어, 각 preset에 맞춘 별도 생성 환경으로 검증했다. 전용 test의 4환경·32step·near100%/mix headless 재로드와 셸/Python/YAML·문서 경로/목록 순서·diff 확인. [wrapper](output_etc/check/before_objalias_20261028/wrapper.log). 기존 학습을 유지했으며 본학습·GUI/VNC 실화면·장기 성능은 실행/검증하지 않았다.

### near50 현재 서버 실행 경로 확인

- `stage2_before_ground_atprogress_nonemask_rsi433_near50_1200`의 train/YAML 기본 source를 존재하는 형제 `stage1_message`의 fixed78000으로 수정했다. 현재 서버에는 GPU0만 있으므로 config의 해당 학습·로컬·VNC 명령과 USAGE 예시를 GPU0으로 맞췄다.
- GPU0/MPS·4명/1024환경/8물체·CHECK 실행이 epoch2/frame262144 저장 후 정상 종료했다. source epoch78000·actor encoder73 tensor/RMS 불변·head6 tensor 갱신·모델과 scalar566개 finite를 확인했다. 관련 CPU8개·세 wrapper 셸 문법·diff 확인. [학습 로그](output_etc/check/near50_local78000_20261028/train.log), [저장본 검증](output_etc/check/near50_local78000_20261028/validation.json), [CPU](output_etc/check/near50_local78000_20261028/cpu.log). 기존 학습 두 개와 원본 checkout은 보존했으며 본학습은 시작하지 않았다.

### RSI433 원래/후행 상대 거리 합 50/50

- `stage2_before_ground_atprogress_nonemask_rsi433_near50_1200` env/train YAML·전용 train/test/VNC를 추가했다. RSI433의 nonemask·AT progress·후행 ON_TOP loco/pickUp/carryWith40/30/30·1200step·2e-5·fixed78000 새 전이는 유지하고 거리20/20/60을 기존 BEFORE 제안 분포의 원래/후행 가까움50/50으로 교체했다. 기존 목표 샘플러를 사용하며 추가 단거리 상한은 두지 않는다.
- 가까운 쌍은 XY에서 선행 사람→선행 상자 거리보다 SIT/CLIMB 후행 사람→받침 상자, ON_TOP 후행 사람→들 상자 + 후행 사람→받침 상자의 합이 작아야 한다. 위반 시 해당 쌍의 거리 후보를 재추첨하며 그래프·모드·RSI 요청을 유지한다. 원래 모드·단독 배치는 부등식을 강제하지 않는다. RSI 사람·상자는 같은 변환으로 이동하고 최종 물리 검사에서도 합 조건을 확인한다. 평가 `BEFORE_FOLLOWER_NEAR=1`은100% 상대 가까움이며2/short는 지원하지 않는다. [설계](markdowns/reference/stage2_before.md#rsi433-상대-거리-합-near50), [명령](markdowns/config.md), [옵션](markdowns/USAGE.md#stage2-before-rsi433-relative-near50).
- CPU42개 통과:50/50·RSI 독립성·합 조건·원래 모드 범위·RSI 상대 자세·checkpoint 분리와 기존 BEFORE 회귀. [CPU](output_etc/check/before_relative50_20261028/cpu.log), [회귀](output_etc/check/before_relative50_20261028/regression.log). GPU5/MPS·4명/1024환경/8물체 CHECK epoch2/frame262144에서 source78000·encoder73 tensor/RMS 동결·head6 tensor와 양쪽 BEFORE bias 갱신·adapter 없음·모델/scalar finite를 확인했다. 초기 가까움 비율50.69%였다. [학습](output_etc/check/before_relative50_20261028/training_validation.json). 물리 배치 최종 실패0을 확인했다. 32환경 loco/mix/pickup/carry·단독5개 구성에서 최종 합 조건·원래 접근 범위·RSI 상대 자세·AT progress·AMP·부분 reset·가중치 불변을 확인했다. [물리/재로드](output_etc/check/before_relative50_20261028/eval/physics_validation.json). 전용 test로100% 가까움·BEFORE ON_TOP·mix RSI·4환경·32step headless 재로드도 통과했다. [wrapper](output_etc/check/before_relative50_20261028/eval.log). 셸/Python/YAML 문법·새 문서 경로/명령 순서·diff를 확인했다. 본학습은 시작하지 않았으며 기존 학습을 유지했다. GUI/VNC 실화면·장기 성능은 미검증이다.


### Nonemask 후행 ON_TOP loco/pickUp/carryWith 40/30/30

- `stage2_before_ground_mix226_atprogress_nonemask_rsi433_1200` env/train YAML·전용 train/test/VNC를 추가했다. 집기와 후반 운반 학습을 함께 늘리기 위해 BEFORE ON_TOP 후행만 loco40%/pickUp30%/carryWith30%를 요청하며 선행·단독·SIT/CLIMB는 loco다. 현재 서버의 fixed78000 절대경로에서 새 전이한다. nonemask·AT progress·거리20/20/60·1200step·2e-5·encoder/RMS 동결·action head 학습은 유지한다.
- carry50 전용 캐시에 pickUp을 추가했다. 원본 상대 자세를 유지하는 크기/몸 관통/native 물리 유효 프레임을 사용하며 carryWith는 HOLDING·바닥8cm 이상을 추가 검사한다. 유효 프레임 부재 또는8회 배치 재시도 소진 시 후행만 loco로 대체한다. 요청·적용·fallback 통계와 checkpoint 계약, 평가 전용 loco/mix/pickup/carry 옵션을 분리했다. [설계](markdowns/reference/stage2_before.md#nonemask-후행-on_top-rsi433), [명령](markdowns/config.md), [평가 옵션](markdowns/USAGE.md#stage2-before-nonemask-rsi433).
- CPU34개 통과: 분포·대상 제한·재정규화 없는 fallback·배치 소진/부분 환경 보존·기존 config/checkpoint 분리·nonemask/full RSI/bias 회귀. [신규·관련](output_etc/check/before_nonemask_rsi433_20261028/cpu.log), [회귀](output_etc/check/before_nonemask_rsi433_20261028/regression.log). GPU5/MPS·4명/1024환경/8물체 CHECK epoch2/frame262144 저장에서 fixed78000 전이·encoder73 tensor/RMS 동결·head6 tensor 갱신·adapter 없음·양쪽 BEFORE bias 갱신·모델/scalar finite·물리 배치 최종 실패0을 확인했다. 첫 집계 실제 loco/pickUp/carryWith는45.6/29.1/25.2%였다(carryWith 요청563개 중77개 fallback). [학습](output_etc/check/before_nonemask_rsi433_20261028/training_validation.json).
- 32환경에서 loco/mix/pickup/carry·단독5개 구성의 상대 자세·carry HOLDING/높이·AT progress 연결·AMP·부분 reset·가중치 불변을 확인했다. [물리/재로드](output_etc/check/before_nonemask_rsi433_20261028/eval/physics_validation.json). 전용 test wrapper의 BEFORE ON_TOP·mix·4환경·32step headless 재로드와 셸/Python 문법·문서 경로/순서·diff도 확인했다. [wrapper 평가](output_etc/check/before_nonemask_rsi433_20261028/eval.log). 본학습과 GUI/VNC 실화면·장기 성능은 미검증이며 기존 학습 프로세스는 유지했다.


### AT progress SELF bias0·NONE 마스킹

- `stage2_before_ground_mix226_atprogress_nonemask_1200` env/train YAML·전용 train/test/VNC와 실행 계약을 추가했다. actor·critic 사람 SA4의 `before_only_mask_none` 모드는 SELF bias 상수0·NONE 완전 차단·BEFORE bias만 학습한다. adapter 없는 AT progress의 fixed78000 새 전이·encoder/RMS 동결·action head 학습·거리20/20/60·loco·보상/gate·1200step·학습률2e-5를 유지한다. 기존 bias0와 같은 파라미터 모양이라도 checkpoint 계약으로 혼용을 차단한다. [설계](markdowns/reference/stage2_before.md#at-progress-self-bias-0none-마스킹), [명령](markdowns/config.md).
- 관련 CPU31개 통과. NONE 확률0·입력/점수 경로 차단·SELF-only QK gradient0·BEFORE/QK gradient·사람 순서 변경·actor/critic 명시적 마스크와 동등성·전이 초기 출력·head 학습/encoder 동결·재로드를 검사했다. GPU0/MPS·4명/1024환경/8물체 CHECK 학습 epoch2/frame262144에서 encoder73 tensor/RMS 동결·head6 tensor 갱신·adapter 없음·양쪽 SA/BEFORE bias 갱신·모델/scalar finite·물리 배치 최종 실패0을 확인했다. 저장본의 양쪽4레이어에서 NONE 확률 정확한0·SELF bias0도 검사했다. [CPU](output_etc/check/before_nonemask_20261028/cpu.log), [학습 검증](output_etc/check/before_nonemask_20261028/training_validation.json).

- 전용 test로 BEFORE ON_TOP·더 가까움·4환경·32step headless 재로드를 확인했다. 셸/Python 문법·문서 경로/목록 순서·diff 확인. [재로드 로그](output_etc/check/before_nonemask_20261028/eval.log). 본학습은 시작하지 않았으며 기존 두 학습을 유지했다. GUI/VNC 실화면·장기 성능은 미검증이다.

### Adapter 없는 AT progress actor·critic SELF/NONE bias0

- `stage2_before_ground_mix226_atprogress_bias0_1200` env/train YAML·전용 train/test/VNC와 실행 계약을 추가했다. 기본 AT progress에서 actor·critic 사람 SA4의 SELF/NONE bias만 상수0으로 고정하고 BEFORE bias는 학습한다. adapter 없이 action head 학습·encoder/RMS 동결·fixed78000 새 전이·거리20/20/60·loco·AT progress·1200step·학습률2e-5를 유지한다. [설계](markdowns/reference/stage2_before.md#at-progress-actorcritic-selfnone-bias-0), [명령](markdowns/config.md).
- 관련 CPU27개 통과: adapter 유무별 초기 출력 동등성·양쪽 고정0·BEFORE/QK gradient·head 학습/동결·Stage1 전이·재로드·checkpoint 혼용 거부 확인. GPU0/MPS·4명/1024환경/8물체 CHECK 학습 epoch2/frame262144에서 encoder73 tensor/RMS 동결·head6 tensor 갱신·adapter 없음·actor/critic 모든 layer/head SELF/NONE 정확한0·BEFORE/SA/critic head 갱신·finite·물리 배치 최종 실패0을 확인했다. [CPU](output_etc/check/before_atprogress_bias0_20261028/cpu.log), [학습 검증](output_etc/check/before_atprogress_bias0_20261028/training_validation.json).

- 전용 test에서 BEFORE ON_TOP·더 가까움·4환경·32step headless 재로드를 확인했다. 셸/Python 문법·문서 경로/목록 순서·diff 확인. [재로드 로그](output_etc/check/before_atprogress_bias0_20261028/eval.log). 본학습은 시작하지 않았으며 기존 두 본학습을 유지했다. GUI/VNC 실화면·장기 성능은 미검증이다.

## 2026-10-27

### Adapter actor·critic SELF/NONE bias0

- `stage2_before_ground_mix226_atprogress_adapter_bias0_1200` env/train YAML·전용 train/test/VNC를 추가했다. 기존 adapter 설정에서 actor·critic 사람 SA4 모두 SELF/NONE bias는 상수0, BEFORE bias만 학습한다. critic 모드 옵션과 checkpoint 계약을 추가했으며 기존 actor-only 실험과 기본 adapter의 동작은 유지한다. fixed78000 새 전이·공유 adapter·encoder/RMS/head 동결·거리20/20/60·loco·AT progress·1200step·학습률2e-5는 기반과 같다. [설계](markdowns/reference/stage2_before.md#adapter-actorcritic-selfnone-bias-0), [명령](markdowns/config.md).
- CPU73개 통과(기존30500 source 전이 테스트2개 제외; 새 전이 테스트 및 실제78000 전이 통과). 초기 actor/critic 출력 동등성·양쪽 BEFORE/QK gradient·고정0·adapter gradient·head 동결·checkpoint 혼용 거부를 검사했다. GPU0/MPS·4명/1024환경/8물체 CHECK 학습 epoch2/frame262144에서 encoder73/head6 tensor·RMS 동결, actor/critic 모든 layer/head의 SELF/NONE 정확한0·BEFORE bias/SA/adapter 갱신, finite·물리 배치 최종 실패0을 확인했다. [CPU](output_etc/check/before_adapter_bias0_20261027/cpu.log), [학습](output_etc/check/before_adapter_bias0_20261027/training_validation.json).
- 전용 test로 BEFORE ON_TOP·더 가까움·4환경·32step headless 재로드, 전용 train으로1024환경 resume epoch3/frame393216을 확인했다. 재개 후에도 양쪽 SELF/NONE0·encoder/head/RMS 동결·adapter optimizer step96→144·finite·물리 실패0을 검증했다. [재개](output_etc/check/before_adapter_bias0_20261027/resume_validation.json). 셸/Python 문법·문서 경로/순서·diff 확인. 본학습은 실행하지 않았고 기존 두 학습 프로세스를 유지했다. GUI/VNC 실화면·장기 성능은 미검증이다.

### AT progress 공유 action head adapter

- `stage2_before_ground_mix226_atprogress_adapter_1200` env/train YAML·전용 train/test/VNC를 추가했다. AT progress226의20/20/60·loco·1200step·보상/gate·전체 관계 bias·학습률2e-5를 유지하고 fixed78000에서 새로 전이한다. actor encoder/RMS와 head3층을 동결하며 SA4·공유 hidden Linear adapter2개를 학습한다. adapter는 weight/bias0에서 시작해 기존 Linear 출력과 더한 뒤 ReLU를 적용한다. critic/AMP 정책은 유지하며 checkpoint 계약을 분리했다. [설계](markdowns/reference/stage2_before.md#at-progress-공유-action-head-adapter), [명령](markdowns/config.md).
- CPU95개 통과(로컬30500 source를 사용하는 기존 전이 테스트2개 제외; 새 전이 테스트와 실제78000 전이 별도 통과). GPU0/MPS·4명/1024환경/8물체 CHECK 학습 epoch2/frame262144에서 actor encoder73 tensor·head6 tensor·RMS가 source와 정확히 동일하고 adapter4 tensor·SA·critic head 갱신, 모델/scalar finite·physical reset 실패0을 확인했다. [CPU](output_etc/check/before_adapter_20261027/cpu.log), [회귀](output_etc/check/before_adapter_20261027/cpu_regression.log), [학습 검증](output_etc/check/before_adapter_20261027/training_validation.json).
- 전용 test의 BEFORE ON_TOP·더 가까움·4환경·32step headless 재로드, 전용 train의1024환경 resume epoch3/frame393216을 확인했다. 재개 후 encoder/head/RMS 동결·adapter optimizer step96→144·갱신·finite·물리 실패0을 확인했다. [재개 검증](output_etc/check/before_adapter_20261027/resume_validation.json). 셸/Python 문법·문서 경로·diff 확인. 본학습은 시작하지 않았고 기존 두 학습 프로세스를 유지했다. GUI/VNC 실화면과 장기 성능은 미검증이다.

- carry50 기존2500 epoch 재개 명령을 [USAGE](markdowns/USAGE.md#stage2-before-mix226-at-progress-carry50)에 추가했다. 실제 체크포인트 존재와 train wrapper의 `RESUME_CHECKPOINT → --checkpoint/--resume 1` 연결을 확인했으며 학습은 실행하지 않았다.

### AT progress actor 피처·행동 출력 변화 진단

- AT progress19500·seed6101·32환경 random_scenario·loco·최초 에피소드1200step 상한에서10step마다 actor encoder/사람 SA4 출력과 head 대입 출력을 수집했다. encoder73 tensor/RMS가 source78000과 정확히 동일하며 검사 중 모델 가중치·학습 프로세스는 유지했다. 같은 관측의 encoder→SA4 상대 L2 변화(사람별 평균)는 단독 ON_TOP17.45%, BEFORE ON_TOP30.86%, 후행 SIT23.85%, CLIMB24.33%였다. BEFORE ON_TOP의 SA1~4 누적 변화는10.53/17.64/25.50/30.86%, 최종 cosine0.9493이었다.
- BEFORE ON_TOP 중 자기 상자1m 이내·HOLDING 미충족 구간의 피처 변화32.75%, HOLDING 충족 구간18.89%였다. 같은 관측에서 source head(encoder 출력) 대비 현재 head(SA4 출력)의 clip 후 action 상대 RMS 차이는 단독 ON_TOP24.96%, BEFORE ON_TOP67.54%였다. 현재 head에 encoder 출력을 직접 넣었을 때도 source head 대비43.57% 차이가 났다. 이는 피처/출력 변화량이며 실패 원인이나 성능 하락률로 해석하지 않는다. [결과·정의](output_etc/check/atprogress_features_20261027/summary.json).

### BEFORE ON_TOP 운반·내려놓기 정책 분리 진단

- AT progress19500/carry50 2500/full RSI4500을 공통 loco sampler·seed6101·32환경(before_stack64쌍)·1200step 상한으로 비교했다. 체크포인트별 원래 계약 검증 후 RSI 초기화 외 환경 설정의 동등성, 실제 초기 root/dof/상자/목표/graph/크기 동일성을 확인했다. HOLDING+상자 바닥8cm 이상을 연속0.1초 만족한 집기는14/6/3명, 같은 조건3초 유지는14/6/1명, raw/gated ON_TOP은 모두0명이었다. finite·가중치 불변을 확인했다. [결과와 판정 기준](output_etc/check/three_loco_ontop_20261027/summary.json).
- full RSI4000·seed42·32환경·before_stack에서 동일 초기 장면의 후행만 현재 정책/coordination 우회/Stage1 encoder+head로 비교했다. carry 시작40명의 ON_TOP은 모두0명이며, 종료 시 선행 목표 거리 중앙값은18.50/5.73/2.75m였다. 원래 Stage1 제어도 해당 BEFORE 배치에서는 파지를 오래 유지하지 못해 attention 한 부분만의 문제로 확정하지 않았다. 받침40개 중30개는 Stage1 SUPPORT 크기 범위 밖이었다. [비교](output_etc/check/ontop_policy_diagnosis_20261027/comparison.json).
- 검사 전용으로 선행을 putDown 후반·후행을 목표 근처로 배치한 동일 장면 비교에서 현재 정책은 carry 후행52명 중1명 ON_TOP 성공(연속24.6초), Stage1 후행 대입은 gated 성공0명이었다. 이 배치는 선행 phase·목표·후행 거리를 함께 바꾸며, 선행 프레임은 기존 기하 검사를 거치지만 별도 native 물리 캐시를 만들지 않았다. 학습/공용 캐시/보상·gate는 변경하지 않았다. [종합](output_etc/check/ontop_policy_diagnosis_20261027/summary.json).
- 선행 목표/현재 상자를 각각 XY1m 옮긴 관측 진단에서 후행 action 변화 RMS 평균은0.0034/0.0494였다. 목표 정보에 반응은 있으나 올바른 활용을 증명하지 않는다. carry50 과거1500 평가도 carry 유지52/57(연속3초)·ON_TOP0/57로 같은 후반 구간 실패가 있었음을 재확인했다. 작은 표본이며 캐시 불안정·크기 분포·목표 정보 활용·보상 신호 중 단일 원인은 아직 확정하지 않았다.

### Full RSI 집기·물리 진단과 선후행 carryWith 뷰어

- full RSI test/VNC에 평가 전용 `BEFORE_FULL_RSI=both_carry`를 연결했다. BEFORE ON_TOP 쌍의 선행 AT/후행 ON_TOP 모두 carryWith를 요청하고, 다른 사람과 크기/배치 fallback은 loco다. 콘솔에 실제 적용 쌍 수를 출력한다. 학습 확률·현재 프로세스·공용 캐시·checkpoint 계약은 유지했다. [사용법](markdowns/USAGE.md#stage2-before-mix226-at-progress-full-rsi).
- CPU7개 통과. 전용 test wrapper에서 seed42·before_stack·더 가까움·4명/1환경·32step headless 재로드 및 두 쌍 모두 carryWith 적용을 확인했다. GUI 실화면은 미검증이다. [로그](output_etc/check/fullrsi_pickup_diagnosis_20261027/viewer_smoke.log). 셸/Python 문법·diff 확인.
- full RSI4000을 32환경·before_stack·seed42·최대1200step에서 mix/both_carry로 진단했다. mix 후행 carryWith19/19는 HOLDING 조건을 연속3초 이상 유지했으나 ON_TOP은 모든 초기화에서0/64였다. loco 후행27/30은 자기 상자1m 이내에 접근했지만 HOLDING은7/30이었다. actor encoder73 tensor/RMS는 source78000과 동일하고 모델·rollout은 finite였다. [결과](output_etc/check/fullrsi_pickup_diagnosis_20261027/summary.json).
- 물리 안정성은 완전 통과로 판단하지 않았다. 별도0.1초 PD 유지에서 mix128명 중 carryWith4명, both_carry128명 중 carryWith9명이 native 변위/속도 기준을 초과했다. mix 실패4명은 주변 물체/사람을 분리해도 실패했고, 기존 유효 캐시 프레임4개 중3개는 native 물리 재검사에서도 탈락했다. 캐시와 실제 배치의 안정성 차이는 미해결이며 학습 캐시를 덮어쓰지 않았다. 이는 모든 ON_TOP 실패의 원인으로 확정한 결과는 아니다. [분리 검사](output_etc/check/fullrsi_pickup_diagnosis_20261027/mix/isolation.json), [재검사](output_etc/check/fullrsi_pickup_diagnosis_20261027/mix/native_rescreen.json).

## 2026-10-26

### AT progress mix226 단독·BEFORE 전체 Stage1 RSI

- 본학습 실행 확인 중 env YAML의 표시용 `experiment.name`이 기반 AT progress 이름으로 남은 것을 fullrsi 이름으로 정정했다. 실행 이름은 train config/CLI를 사용하므로 기존 실행·output·학습 동작에 영향은 없으며, 실행 중 설정/프로세스는 변경하지 않았다. YAML 전후 env/sim/전이 설정 동일·diff를 확인했다.

- `stage2_before_ground_mix226_atprogress_fullrsi_1200` env/train YAML·전용 train/test/VNC를 추가했다. 단독/선행/후행에 Stage1 과제별 RSI와70% 후반 프레임 선택·크기 fallback을 적용하며, BEFORE 공유 상자의 XY/yaw 배치·합성 재시도/fallback을 분리했다. 관계/거리 모드 고정·ground-only·현재 gate/AT progress·1200스텝·fixed78000 전이/동결을 유지한다. [설계](markdowns/reference/stage2_before.md#단독before-전체-stage1-rsi), [명령](markdowns/config.md), [평가 옵션](markdowns/USAGE.md#stage2-before-mix226-at-progress-full-rsi).
- CPU75개 통과(기존30500 fixture 부재 테스트1개 제외). GPU0·4명/1024환경/8물체 전용 train CHECK epoch2/frame262144 저장, actor encoder73 tensor/RMS 동결·action head 갱신·모델/scalar finite·physical reset 실패0 확인. 32환경·8구성×16step에서 단독/BEFORE·loco/mix·클립 상대 자세·자유 거리·AT progress·AMP·부분 reset 보존·가중치 불변을 검증했다. [CPU](output_etc/check/before_fullrsi_20261026/cpu.log), [학습](output_etc/check/before_fullrsi_20261026/training_validation.json), [물리](output_etc/check/before_fullrsi_20261026/physics/validation.json).
- 실제 요청/실행·크기/합성 fallback을 역할·과제별로 기록한다. 확인 학습의 후행 SIT/CLIMB RSI 실현은23.3/25.8%, ON_TOP putDown은5.4%였다(요청50/50/10%); 들린/기울어진 공유 받침과 장면 충돌 때문에 감소한다. 크기 캐시19,819조건 중 carryWith33/727·putDownOnTop6/16,180만 프레임이 없었다. 검증한 캐시를 공식 공용 경로 `output/rsi_cache_before_full`에 준비했다. [캐시 설치](output_etc/check/before_fullrsi_20261026/cache_installed.json). 전용 test의 loco/mix 각각4환경·32step 재로드 완료. 셸/Python 문법·문서 경로·diff를 확인했다. 장기 수렴과 GUI/VNC 실화면은 미검증이며 기존 본학습·원본 checkout은 유지했다.

### AT progress mix226 BEFORE ON_TOP 후행 carryWith RSI

- `stage2_before_ground_mix226_atprogress_carry50_1200` env/train YAML·전용 train/test/VNC를 추가했다. AT progress 226을 기반으로 거리20/20/60 추첨과 독립적으로 BEFORE ON_TOP 후행만 loco/carryWith50/50을 요청한다. RSI 사람·자기 payload를 함께 이동/회전하고 RSI payload의 바닥 거리 재샘플링은 생략한다. 선행·단독·SIT/CLIMB·loco 배치·1200스텝·gate/보상·전체 관계 bias는 유지하며 fixed78000에서 새 Stage2로 시작한다. 기존 체크포인트와 계약을 분리했다. [설계](markdowns/reference/stage2_before.md#at-progress-후행-carrywith-rsi-5050), [명령](markdowns/config.md), [추론 옵션](markdowns/USAGE.md#stage2-before-mix226-at-progress-carry50).
- 실제 크기별 carryWith 전용 물리/기하 캐시와 사용 불가 시 loco fallback을 추가했다. 평가 기본은 loco이며 `BEFORE_CARRY_RSI=mix/carry`만 후행 RSI를 켠다. 평가 skill 목록과 독립적인 carry pool·AMP history 초기화를 검증했다. 확인 학습에서 유효 슬롯3479/4096, 요청979/1912(51.2%)·적용826/1912(43.2%)·fallback153이었다. 50%는 요청 확률이며 상자 크기 분포를 바꾸지 않아 실현 비율은 낮을 수 있다.
- CPU69개 통과(기존30500 fixture 부재 테스트1개 제외). GPU0·4명/1024env/8물체 CHECK 학습 epoch2/frame262144 저장, Stage1 actor encoder73 tensor/RMS 동결·action head 갱신·finite·배치 실패0 확인. 32env·6구성의 거리/RSI 상대 자세·대상 제한·기본 loco·AMP/추론·부분 reset 보존 검사와 전용 test의 loco/carry 재로드(각4env·32step) 통과. [CPU](output_etc/check/before_carry50_20261026/cpu.log), [학습](output_etc/check/before_carry50_20261026/training_validation.json), [물리](output_etc/check/before_carry50_20261026/physics_final/validation.json). 장기 수렴·GUI/VNC 실화면은 미검증이며 본학습은 실행하지 않았다. 기존 학습·원본 checkout은 보존했다.

### mix226 두 설정의 actor SELF/NONE bias 0 비교 실험

- `stage2_before_ground_mix226_actorbias0_1200`·`stage2_before_ground_mix226_atprogress_actorbias0_1200` env/train YAML·전용 train/test/VNC를 추가했다. actor human SA4의 모든 head에서 SELF/NONE은 상수0, BEFORE만 학습 파라미터로 저장한다. critic의 세 관계 bias·기반별 보상/샘플링·20/20/60·1200스텝·PPO·Stage1 동결은 유지하고 두 실험 모두 fixed78000에서 새로 시작한다. 기존/신규 및 서로 다른 progress checkpoint의 혼용을 차단했다. [설계](markdowns/reference/stage2_before.md#actor-selfnone-bias-0), [명령](markdowns/config.md).
- CPU65개 통과(기존30500 fixture 부재 전이 테스트1개 제외). 두 전용 train 모두 GPU0·4명/1024환경/8물체·CHECK=1로 epoch2/frame262144 저장 완료. actor SELF/NONE 상수0·BEFORE 갱신, critic 세 관계 갱신, actor encoder73 tensor/RMS 동결·action head 갱신, 모델/scalar finite·물리 배치 실패0 확인. 전용 test로 각각4환경·32step·강제 더 가까운 모드 재로드 완료. [CPU](output_etc/check/before_actor_bias0_20261026/cpu.log), [저장본 검증](output_etc/check/before_actor_bias0_20261026/training_validation.json). GUI/VNC 실화면·장기 수렴은 미검증이며 본학습은 시작하지 않았다. 기존 두 본학습과 원본 checkout은 보존했다.

## 2026-10-25

### mix226 BEFORE 후행 progress를 선행 AT 목표로 변경

- `stage2_before_ground_mix226_atprogress_1200` env/train YAML·전용 train/test/VNC를 추가했다. 현재 mix226의 과제/크기 추첨·20/20/60 거리 혼합·1200스텝·PPO·모델/관측을 유지하고, BEFORE 후행 SIT/CLIMB의 사람 XY 및 ON_TOP 자기 payload XY의 progress 목적지만 선행 AT 목표로 바꾼다. gate가 열려도 목표 기준을 유지하며 기존 상자 크기/buffer·가중치·성공 포화·state/success gate·HOLDING·단독 과제는 보존한다. 새 실험만 BEFORE 과제별 gate/raw_success/success 스텝 지표를 추가했다. [설계](markdowns/reference/stage2_before.md#선행-at-목표-progress-stage2_before_ground_mix226_atprogress_1200), [명령](markdowns/config.md).
- 사용자 선택대로 현재 mix226과 같은 fixed78000에서 새로 전이하도록 기본값을 설정했다. 기존 실험과 output·환경 옵션·checkpoint 계약을 분리했다. 사용자 지정 mix226 저장 config와 비교해 런타임 추가 필드를 제외한 환경 차이는 `before.progress_target` 하나임을 확인했다. 기존 본학습과 원본 checkout은 변경/재시작하지 않았다.
- CPU59개 통과(기존30500 fixture가 없는 전이 테스트1개 제외). GPU0에서32환경·4구성·각16step으로 후행3과제×3거리모드, 실제 목표 연결, 나머지 기하/보상 입력 보존, partial reset·AMP/추론 finite·가중치 불변을 확인했다. [CPU](output_etc/check/stage2_before_ground_mix226_atprogress_1200/cpu.log), [물리 검사](output_etc/check/stage2_before_ground_mix226_atprogress_1200/physics/validation.json).
- 표준4명·1024환경·8물체 CHECK=1 학습/저장(epoch2/frame262144) 완료. encoder73 tensor/RMS 동결·action head 갱신·274종 scalar/model finite·물리 배치 실패0, 모드20.62/20.33/59.05%를 확인했다. 전용 test로 저장본4환경·32step·강제 더 가까운 모드 재로드 정상 종료. 셸/Python 문법·문서 경로·diff 확인. [학습 검사](output_etc/check/stage2_before_ground_mix226_atprogress_1200/training_validation.json), [재로드](output_etc/check/stage2_before_ground_mix226_atprogress_1200/reload.log). GUI/VNC 실화면·장기 수렴은 미검증이며 본학습은 시작하지 않았다.

## 2026-10-24

### BEFORE 20/20/60·상자 기준 AT 운반 거리·1200스텝

- `stage2_before_ground_mix226_1200` env/train YAML과 전용 train/test/VNC를 추가했다. BEFORE 쌍별20/20/60, 선행 상자→AT 목표1~10/1~10/1~5m, 후행 ON_TOP의 선행 상자 거리·자기 payload 거리1~5/1~2.5/1~1.25m를 적용한다. 나머지 거리는 합의 표를 따르며 단독 목표·그룹 추첨·loco·보상·Stage1 전이/동결·PPO·1200스텝은 유지한다. 기존 실험과 output/checkpoint 계약을 분리했다. [배치 표](markdowns/reference/stage2_before.md#바닥-before-거리-혼합-stage2_before_ground_mix226_1200), [명령](markdowns/config.md).
- CPU48개 통과(없는30500 fixture에 의존하는 기존 전이 테스트1개 제외). GPU0에서32환경·4구성·각16step으로 후행3과제×3모드 거리, AT 운반 거리/사람 제외 영역, 단독 목표, 초기 gate·부분 reset·AMP/추론 finite·가중치 불변을 확인했다.8환경·동일4구성에서 viewer 옵션2의 강제 더 가까운 배치도 통과했다. [물리 검사](output_etc/check/stage2_before_ground_mix226_1200/physics/validation.json), [강제 모드 검사](output_etc/check/stage2_before_ground_mix226_1200/short_viewer/validation.json).
- 표준4명·1024환경·8물체 CHECK=1 학습/저장(epoch2/frame262144) 완료. 실제 모드20.43/20.28/59.29%, actor encoder73 tensor·관측 RMS 동결, action head 갱신,265종 scalar/model finite, 물리 배치 실패0을 확인했다. 새 test wrapper로 저장본4환경·32step·옵션2 재로드 정상 종료. [학습 검사](output_etc/check/stage2_before_ground_mix226_1200/training_validation.json), [재로드 로그](output_etc/check/stage2_before_ground_mix226_1200/reload.log). 셸/Python 문법·문서 링크·diff 확인. GUI/VNC 실화면·장기 수렴은 미검증이며 본학습은 시작하지 않았다. 원본 checkout·기존 학습 프로세스는 보존했다.

## 2026-10-23

### mix334 viewer 더 가까운 모드 선택

- mix334 전용 test/VNC의 `BEFORE_FOLLOWER_NEAR=2`를 추가했다. 평가에서 BEFORE 쌍에 더 가까운 모드를 고정해 선행1~5m·후행1~2.5m·ON_TOP 자기 상자1~2.5m를 사용한다. 기본0의30/30/40·기존1의가까운 모드·단독 배치·학습·저장 계약은 유지한다. 직접 실행은 `--before_follower_short`이며 학습/비호환 혼합/근접 옵션 동시 사용은 거부한다. [명령](markdowns/USAGE.md#stage2-before-mix334-1200).
- CPU43개 통과(기존30500 파일 의존 전이 테스트1개 제외). GPU0의8환경·4구성·각16step에서 전 BEFORE 쌍 mode2·거리·부분 reset·AMP/추론 finite·가중치 불변을 확인했다. 실제 mix33412500 저장본을 전용 test·옵션2·4환경·32step으로 재로드해 정상 종료했다. [물리 검사](output_etc/check/mix334_short_viewer_20261023/physics/validation.json), [재로드](output_etc/check/mix334_short_viewer_20261023/reload.log). 셸 문법·문서 경로·diff 확인. 실제 GUI/VNC 화면은 검사하지 않았으며 기존 학습/viewer는 유지했다.

## 2026-10-22

### BEFORE 30/30/40 거리 혼합·1200스텝

- `stage2_before_ground_mix334_1200` env/train YAML과 GPU0 기본 train/test/VNC를 추가했다. BEFORE 쌍별 기존30%·가까움30%·더 가까움40%이며, 마지막 모드에서 선행 접근1~5m·후행 및 ON_TOP 자기 상자1~2.5m를 사용한다. 최소1m·독립 방향/거리·물리 재시도 중 모드 유지, 단독 배치·AT 목표 샘플러·보상·Stage1 전이를 보존하고 장면 에피소드 상한을1200스텝으로 늘렸다. 기존 ground/mix50와 checkpoint/output을 분리했다. [배치 계약](markdowns/reference/stage2_before.md#바닥-before-거리-혼합-stage2_before_ground_mix334_1200), [명령](markdowns/config.md).
- CPU42개 통과(기존30500 파일 의존 전이 테스트1개 제외). 초기 GPU 검사에서 발견한 선행 rigid-body 이동 인덱싱 오류를 수정한 뒤,32환경·4구성·각16step에서 모든 후행 과제×3모드의 거리·초기 gate·부분 reset·추론/AMP finite·가중치 불변을 확인했다. [물리 검사](output_etc/check/stage2_before_ground_mix334_1200/physics/validation.json).
- 표준 train의4명·1024환경·8물체 CHECK=1 학습/저장(epoch2/frame262144)을 완료했다. 저장 상한1200·모드30.78/29.79/39.43%·actor encoder73 tensor/RMS 동결·action head 갱신·265종 scalar finite·물리 실패0을 확인했다. 전용 test로 저장본4환경·32step headless 재로드 정상 종료. [학습 검사](output_etc/check/stage2_before_ground_mix334_1200/training_validation.json), [재로드](output_etc/check/stage2_before_ground_mix334_1200/reload.log). 셸 문법·새 문서 링크·diff 확인. 실제 VNC 화면·1200step 완주·장기 수렴은 미검증이며 본학습은 시작하지 않았다. 원본 checkout과 기존 학습 프로세스는 변경하지 않았다.

## 2026-10-21

### mix50 viewer 가까운 배치 선택

- mix50 test/VNC에 `BEFORE_FOLLOWER_NEAR=1`을 연결했다. 평가에서만 BEFORE 쌍의 가까운 모드를100% 선택하며 선행1~10m·후행1~5m·ON_TOP 자기 상자1~5m, 독립 방향·거리 순서 미강제를 유지한다. 기준은 선행 상자의 초기 위치다. 기본50/50·단독·저장 checkpoint 계약·기존 non-mix 근접 viewer·학습은 유지한다. [명령](markdowns/USAGE.md#stage2-before-5050-혼합-배치).
- CPU36개 통과(기존30500 파일 의존 테스트1개 제외). GPU0에서4환경·4구성·각16step의 전 BEFORE 쌍 가까운 모드·거리·부분 reset·finite·가중치 불변을 확인했다. mix50의10000 저장본을 전용 test로4환경·seed100·32step headless 재로드해 정상 종료했다. [물리 검사](output_etc/check/mix50_near_viewer_20261021/physics/validation.json), [재로드 로그](output_etc/check/mix50_near_viewer_20261021/reload.log). 셸 문법·문서 링크·diff 확인. 실제 viewer 화면은 검사하지 않았으며 실행 중 학습은 유지했다.

## 2026-10-20

### BEFORE 바닥 50/50 거리 혼합 실험

- `stage2_before_ground_mix50` env/train YAML과 GPU0 기본 train/test/VNC를 추가했다. BEFORE 쌍마다 기존/가까운 배치를 50%씩 추첨하고 물리 재시도 동안 모드를 유지한다. 선행 AT는1~10m, 가까운 후행은1~5m, ON_TOP 자기 상자는 후행 기준1~5m이며 거리 순서를 강제하지 않는다. 단독·loco·보상·모델/PPO는 기존 ground와 같다. [배치 계약](markdowns/reference/stage2_before.md#바닥-before-거리-혼합-stage2_before_ground_mix50), [명령](markdowns/config.md).
- 기본 source는 형제 원본 checkout의 fixed73000이며 읽기만 한다. 새 output·checkpoint 계약을 분리하고 test/VNC도 혼합 배치를 사용한다. 혼합 비율·쌍 수 지표를 추가하고 별도 `BEFORE_FOLLOWER_NEAR` 평가 옵션과의 동시 사용은 거부한다.
- CPU35개 통과(기존30500 파일이 없는 전이 테스트1개 제외). GPU0의8환경·4구성·각16step에서 거리·초기 gate·부분 reset·AMP/추론 finite·가중치 불변을 확인했다. 표준 train으로4명·1024환경·8물체 CHECK=1 학습/저장(epoch2/frame262144), encoder73 tensor·관측 RMS 동결·head 갱신·262종 지표 finite·물리 실패0·가까운 비율50.13%(3367쌍)를 확인했다. 저장본의 표준 test4환경·32step 재로드도 통과했다. [학습 검사](output_etc/check/stage2_before_ground_mix50/training_validation.json), [물리 검사](output_etc/check/stage2_before_ground_mix50/physics/validation.json). 셸 문법·문서 링크·diff 확인. VNC 실화면·장기 수렴은 미검증이며 본학습은 시작하지 않았다. 기존 학습 프로세스는 유지했다.

### BEFORE 근접 평가 거리 확대

- 사용자 요청으로 `BEFORE_FOLLOWER_NEAR=1`의 선행 상자 중심→두 사람 root XY 거리를 1~5m로 확대하고 후행에 작은 값을 배정한다. ON_TOP 후행의 자기 payload는 후행의 최종 root XY 기준 1~3m·무작위 각도로 다시 배치한다. 높이·겹침 검사와 단독 인원·기본 학습/평가 경로는 유지한다. [평가 옵션](markdowns/USAGE.md#stage2-before-평가-옵션)을 갱신했다.
- CPU29개 통과(거리·겹침·부분 reset·seed 재현 포함). 기존 전이 테스트1개는 이 checkout의30500 체크포인트 부재로 실패해 별도 실행에서 제외했다. GPU0의73000 체크포인트 전이로4명·4환경·8물체, `before_stack`·`one_pair_stack`·`all_solo`·`random_scenario` 각16step을 검사해 새 거리·초기 gate·AMP/추론 finite·부분 reset·가중치 불변을 확인했다. 처음 시도한 고정 `before_sit`은 검사기가 생성한 무작위 상자 크기와 호환되지 않아 중단했고 최종 무작위 구성에서 SIT/CLIMB 후행도 확인했다. [검증 결과](output_etc/check/before_near_1to5_20261020/physics/validation.json), [CPU](output_etc/check/before_near_1to5_20261020/cpu_available.log). 실제 viewer 화면은 검사하지 않았으며 기존 학습·원본 레포·프로세스는 변경하지 않았다.

### 재부팅 없는 nvitop 조회 복구

- 실행 중 커널 드라이버 `580.173.02`와 설치 NVML `580.178.04` 불일치를 확인했다. Ubuntu 공식 snapshot의 구버전 NVML만 `output_etc/check/nvml_compat/`에 추출해 명령 단위 `LD_LIBRARY_PATH`로 조회하는 방법을 [USAGE](markdowns/USAGE.md#재부팅-전-nvml-불일치-우회)에 기록했다. 시스템 패키지·원본 레포·기존 학습 프로세스는 변경하지 않았다.
- `nvitop --once` 종료 코드0, GPU 메모리·사용률·원본 fixed 학습 PID2735256 표시를 확인했다. [조회 결과](output_etc/check/nvml_compat/nvitop_once.txt). 학습·평가는 실행하지 않았다.

## 2026-10-19

### Stage2 BEFORE 바닥 전용 비교 실험

- `stage2_before_ground` env/train YAML과 GPU5 기본 train/test/VNC를 추가했다. 기존 BEFORE에서 시작 받침과 AT 도착 받침을 비활성화하고 모든 초기 상자·AT 목적지를 바닥 높이로 설정했다. ON_TOP 대상 상자는 유지한다. 물리 인덱스 보존용 고정 받침은 지하 -20/-19.5m로 격리한다. 기존 실험 파일·실행 프로세스는 유지한다.
- 그룹·가변 인원/물체·물체 예산 재추첨·색상·현재 AT gate·loco/AMP·PPO·Stage1 전이는 기존 BEFORE와 같다. reset 계약으로 기존 BEFORE와 checkpoint를 구분하고 ground wrapper에 기존 환경을 넣으면 거부한다. [설정/배치](markdowns/reference/stage2_before.md#바닥-전용-비교-stage2_before_ground), [명령](markdowns/config.md).
- CPU90개 통과. GPU5에서4환경·8preset의 바닥 상자/AT 높이·받침 비활성·초기 gate·물체 연결·각16step 추론·AMP finite·부분 reset·전이 가중치 불변을 확인했다. [물리 검사](output_etc/check/stage2_before_ground/physics/validation.json).
- 표준 train으로4명·1024환경·8물체 CHECK=1 학습·저장(epoch2/frame262144)을 완료했다. 260 scalar와 모델 finite, actor/RMS 동결, 물리 배치 실패0을 확인했다. 저장본을 표준 test로6명·6물체·4환경·근접 옵션·32step 재로드 평가했다. [학습 검사](output_etc/check/stage2_before_ground/training_validation.json).
- 별도 GPU5 VNC의4명·2환경 BEFORE STACK에서 근접 배치와 선반 없는 장면을 확인했다. AT 상자를 바닥 목표에 놓고30 physics step 후4/4 AT 성공·4/4 후행 gate 개방, 최대 Z 오차 약8.54e-6m를 확인했다. 이는 제어된 도달 가능성 검사이며 학습 성공률이 아니다. [AT 검사](output_etc/check/stage2_before_ground/viewer/floor_at_validation.json). 본학습은 시작하지 않았다.

### BEFORE viewer 물체 색상 복구

- BEFORE graph 샘플링 후 누락됐던 색상 갱신을 복구했다. `stage2_coupled`의 기존 규칙처럼 유효 edge의 source/target 전체에서 사용자를 집계해 distractor는 회색, 개인 과제 물체와 ON_TOP 받침은 사람 색, 공유 물체는 노란색으로 표시한다. 논리 토큰이 같은 물리 물체를 가리키면 사용자 집합을 합친다.
- CPU64개 통과. GPU5의 별도 VNC viewer에서4명·2환경·근접 옵션으로 BEFORE STACK→단독→전원 ON_TOP→무작위 과제를 reset하며 각16step 추론·부분 reset·finite를 확인했다. [BEFORE 화면](output_etc/check/stage2_before_colors/viewer/before_stack.png), [단독 화면](output_etc/check/stage2_before_colors/viewer/all_solo.png), [개인 ON_TOP 받침](output_etc/check/stage2_before_colors/viewer/holding_ontop.png)을 직접 확인했다. 색상 getter는 실제 렌더와 달리 흰색을 반환해 검증 근거로 쓰지 않았다.
- 기존 학습·viewer는 종료하지 않았다. 현재 viewer를 같은 명령으로 다시 실행하면 적용되며 재학습은 필요 없다.

### Stage2 BEFORE 가변 인원·물체 예산

- 사용자 요청으로 학습·test·VNC의 4명·8물체 제한을 해제했다. N≥1, M≥N이며 BEFORE 그룹은 항상 2명이다. 고유 과제 물체가 M을 넘는 환경만 그룹·역할·과제를 다시 뽑고, 남는 토큰은 실제 distractor로 채운다. 물체 예산이 작으면 최종 과제/그룹 분포가 조건부로 바뀐다. 불가능한 전원 ON_TOP/pair preset은 오류를 낸다.
- 물리 bank를 max(4N,M)으로 확장하고 압축 토큰·AT 목표 인덱스를 가변 수에 맞췄다. 홀수 인원 학습의 기존 PPO 미니배치 나눗셈 오류를 발견해 장면당4096을 유지하는 4096×N으로 계산한다. 기본4명 설정과 기존 BEFORE checkpoint 계약은 유지한다.
- CPU64개 통과. GPU5에서3명·3물체/4preset, 6명·6물체·2명·11물체·1명·1물체/각2preset을4환경·각16step 검사했다. 근접 옵션, 연결·토큰 예산·부분 reset·AMP/추론 finite·가중치 불변을 확인했다. 기존4명 저장본도 표준 test로6명·6물체·4환경·근접 옵션·32step 재로드 평가를 통과했다. 산출물은 `output_etc/check/stage2_before_variable/`이다.
- GPU5 표준 train으로3명·1024환경·3물체의 CHECK=1 학습·저장을 완료했다. epoch2/frame196608, 미니배치12288, actor/RMS 동결·모델/지표 finite·물리 배치 실패0을 확인했다. [학습 검사](output_etc/check/stage2_before_variable/training_validation.json). 셸 문법·문서 링크·diff 확인. 기존 본학습은 종료·재시작하지 않았다.

### Stage2 BEFORE 2인 그룹 실험

- `stage2_before` env/train YAML과 GPU5 기본 train/test/VNC를 추가했다. 현재 full-success 기반의 fixed Stage1 전이·전원 loco·Stage1 AMP·PPO를 유지하며 4명·1024환경·8개 실제 물체 토큰을 쓴다. BEFORE 80%/단독 20%를 순차 추첨하고 BEFORE는 정확히2명이다. 큰 공유 받침·구역 토큰·COUPLED·공동운반은 없으며 일반 받침과 distractor는 유지한다.
- 선행 AT→크기 조건 후행 SIT/CLIMB/ON_TOP을 연결했다. SA4의 B query→A key만 BEFORE이며, 현재 AT 성공으로 후행 state/success/포화를 열고 HOLDING·progress는 유지한다. 조기 수행/대기 페널티는 추가하지 않았다. `BEFORE_FOLLOWER_NEAR=1`은 무작위1~2m 후행 근접 평가 옵션이다. [설계 계약](markdowns/reference/stage2_before.md), [명령](markdowns/config.md).
- CPU55개 통과. GPU5에서8환경·8구성의 물체 연결·초기 gate 닫힘·각16step 추론·AMP finite·부분 reset·165 tensor strict 전이/가중치 불변을 확인했다. 근접 옵션의8환경·4구성도 통과했다. [기본 검사](output_etc/check/stage2_before/physics/validation.json), [근접 검사](output_etc/check/stage2_before/near/validation.json).
- GPU5 표준 train으로1024환경·CHECK=1·epoch2/frame262144 학습·저장 완료. actor encoder73 tensor·관측 RMS 불변, head/critic/SA 갱신,260 scalar finite·물리 reset 실패0을 확인했다. 첫 확인 학습에서 드러난 BEFORE scalar 지표 수집 오류를 고치고 회귀 테스트와 재학습으로 확인했다. [학습 검사](output_etc/check/stage2_before/training_validation.json).
- 저장본을 표준 test로4환경·before_sit·근접 옵션·episode32 재로드 평가해 정상 종료했다. 셸 문법·문서 링크·diff 확인. 본학습·장기 성공률·실제 VNC 화면은 검사하지 않았으며 기존 실행 프로세스와 다른 레포는 변경하지 않았다.

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
