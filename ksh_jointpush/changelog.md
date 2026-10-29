# Changelog

## 2026-10-29

### CONTACT 적용 범위 정정: 단독 KSH · 공동 jhh

- 사용자 최종 지시에 따라 `members == 1`은 KSH CONTACT state·progress·성공 판정을 그대로 보존하고, `members > 1` CONTACT에만 jhh의 중점XY/1.04m·우회 접근거리·접촉 성공 조건을 적용했다. AT·gate·RSI 등은 유지한다. 기존 전체 jhh 준비 경로와 분리한 `coupled_jhh_solo_ksh_contact` workspace/output 및 새 checkpoint 계약을 사용하며 전용 래퍼는 이 설정을 가리킨다.
- CPU 테스트7개 통과: 혼합 scene에서 단독/공동 state·progress·진단·physics hit 분기, substep hit 누적, AT/gate 보존, 원본 수식·설정·checkpoint 격리. CPU4환경×3preset 각8step·부분 reset도 통과했다. 로그: `output_etc/check/coupled_jhh_solo_ksh_contact_20261029/`. 본학습 시작·재시작 및 GPU/MPS 조작 없음.

### KSH19500 CONTACT-only jhh Partial 비교 설정

- 현재 hard/balanced/safe/clear 실행의 저장 설정을 복제하고 CONTACT state(중점XY·1.04m 높이), 장애물 우회 progress, 양손 접촉 성공 조건만 이전 jhh 방식으로 선택 적용했다. AT·gate·RSI·물성·AMP·LR는 유지하고 구현 SHA를 checkpoint 계약에 포함했다. 기존 native 기본값은 유지하며 전용 train/test/VNC 래퍼와 실행 가이드를 추가했다. 현재 RSI 손 높이0.8m와 reward 목표1.04m 차이는 reference 문서에 명시했다.
- CPU 수식/격리/계약 테스트6개, CPU4환경×3preset 각8step 정책 실행·부분 reset·유한 보상 검증 통과. GPU 미사용·optimizer0회. 기존 GPU5 학습 PID3851733을 유지했고 본학습 시작·재시작은 하지 않았다. 로그는 `output_etc/check/legacy_contact_20261029/`, 상세는 `markdowns/reference/ksh_legacy_contact.md`.

### stage1_2_base coupled 사본 점검·제공 체크포인트 viewer 준비
`2979913`을ports에고정하고coupled_base에독립실행사본·원본읽기전용자료링크/로컬78k·21k검사사본을준비했다. CPU313/PreLN·4종bias gradient/PUSH bank해시통과. 사용자범위정정전확인학습2epoch·32환경8preset물리검사를수행했고추가평가중단·검증프로세스종료. 추가학습/제로샷생성금지·사용자제공Stage2체크포인트대기,viewer미실행. 기존GPU5학습/MPS보존. 상세와검증한계는coupled_base/LOCAL_VIEWER.md.

최신 변경부터 기록한다. 현재 실행법은 [config](markdowns/config.md), 코드 위치는 [structure](markdowns/structure.md)를 따른다. 과거 변경·검증 전체는 [보관 changelog](markdowns/legacy/changelog_before_cleanup.md)에 있다.

## 2026-10-27

### Stage2 목표 포화·TokenHSI 내부 adapter 비교 설정

- 사용자 요청으로 `stage2_jointpush_goal_saturation_adapter` 환경/학습 YAML·train/test/VNC를 추가했다. 일반 loco 목표 포화 설정의 보상·샘플링·PPO를 유지하면서 기존 action head 전체를 동결하고 두 hidden Linear에 0 초기화 병렬 adapter를 더한다. 사람 간 attention·critic·AMP 학습은 유지하며 기존 설정에는 adapter를 생성하지 않는다. 새 모델 계약과 별도 출력 경로를 사용하고, 문서 명령은 Stage1 14000 전이를 명시했다.
- CPU14개에서 설정 동등성·14k 165개 항목 전이·초기 출력·동결/gradient·모델 복원을 확인했다. GPU5·1024환경의 실제 학습 초기화 경로에서 head/encoder optimizer 제외·adapter/attention 포함·lr2e-5·loco 시작 및8step 정책/critic 유한값을 확인했다. 초기 action mean 최대 차이는2.44e-6이었다. 학습 루프를 검사 함수로 대체하여 PPO/optimizer 업데이트0회이며 기존 학습은 유지했다. 결과는 `output_etc/check/stage2_jointpush_goal_saturation_adapter/`에 있다. shell 구문·diff 검사 통과.

### Stage2 일반·RSI 목표 도착 시 gate 면제 비교 설정

- 사용자 승인한 `stage2_jointpush_goal_saturation`·`stage2_jointpush_rsi_goal_saturation`을 별도 환경/학습 YAML·train/test/VNC·출력 이름으로 연결했다. 목표 거리20cm·높이 오차1mm 이내에서는 접촉 gate를 면제하여 AT·CONTACT를 포화시키고, 목표 밖에서는 기존 gate를 적용한다. 영구 성공 latch는 없으며 기존 설정·샘플링·RSI/PPO와 RSI 캐시 경로는 유지한다. 새 옵션은 Stage2 checkpoint 계약에 포함한다.
- CPU12개·shell 구문·설정 경로/문서 링크·diff 검사 통과. GPU5·각1024환경에서 Stage1 10500 전이 후 일반 loco/RSI 진단 초기화 각각8step 정책 추론의 유한값, 목표 위치를 상자 중심으로 놓았을 때 CONTACT 성공 없이 과제 보상1.2, 원래 목표 복원 시 성공 해제를 확인했다. 최초 검증 harness의 feet 진단 키 오류를 수정한 뒤 두 경로 모두 통과했다. 결과는 `output_etc/check/stage2_jointpush_goal_saturation/`에 있다. PPO 업데이트·본학습은 실행하지 않았다.

### jointpush PUSH 목표 표시 개선

- 사용자 요청으로 PUSH 목표에 지름20cm 구형 테두리·반경35cm 바닥 링·목표 높이까지 세로선을 추가했다. 단독은 담당 사람 색, coupled는 노란색으로 공통 목표 슬롯당 하나만 표시한다. Stage1/Stage2 jointpush viewer에 적용하며 물리 actor 크기·관측·보상·gate는 유지한다.
- GPU5·1환경·Stage2 4500 checkpoint에서 공동/단독 PUSH와 일반 AT를 각각8step 실행하고 목표 수1/4/0·색상·기존 marker scale·유한 보상·렌더 이미지를 확인했다. 결과는 `output_etc/check/stage2_jointpush_rsi/push_goal_markers/`에 있다. PPO 업데이트 없이 기존 학습 프로세스를 유지했다.

### Stage2 jointpush RSI 진단 추론 옵션

- 사용자 요청으로 test/VNC에 `PUSH_RSI=1`(직접 CLI `--eval_push_rsi`)을 연결했다. 평가 모드를 유지하면서 단독·coupled PUSH는 접촉 RSI100%, 다른 과제는 loco100%로 시작한다. 기본 loco 평가·학습 RSI 비율·checkpoint 계약을 유지하며 기존 네 과제 RSI 캐시는 생성하지 않는다. 실행 옵션·명세·구조 문서를 갱신했다.
- CPU20개·shell 구문·diff 검사 통과. GPU5·1024환경의 Stage2 4000 checkpoint로 옵션 ON7종/기본OFF3종 reset·16step 정책 추론·부분 reset·유한값·PUSH 참조 관절/AMP를 검증했다. 혼합 ON에서 PUSH3521명만 RSI, 나머지575명은 loco였다. Stage1 11000 제로샷·1환경 VNC에서도4명 공동 RSI 렌더/PNG를 확인했다. optimizer step0회, 기존 학습 프로세스 유지. 결과는 `output_etc/check/stage2_jointpush_rsi/eval_rsi_option/`에 있다.

## 2026-10-26

### Stage2 jointpush RSI variant 연결

- `stage2_jointpush_rsi` 환경/학습 YAML·train/test/VNC를 추가했다. 단독은 Stage1 과제별 RSI·유효/후반 프레임 샘플링을 사용하고, coupled는 그룹 단위 loco50/push50·사람별 독립 프레임으로 초기화한다. 구역별 실제 뒤쪽 면에 사람만 정렬하며 물리 PUSH 상자/목표를 유지하고, 전체 상자 침투·멤버 간 충돌을 검사한다.
- 기존 네 과제의 RSI 캐시를 4명·24 actor bank에 연결하고 추가 크기 캐시는 별도 경로에 저장한다. 그룹/과제·물성·보상·AMP·PPO를 유지하며 기본 eval은 같은 loco 배치/목표 샘플러에서 loco100%다. 기존 loco 전용 Stage2는 유지한다. 명령·구조·상세 RSI 명세를 갱신했다.
- 관련 CPU20개·shell 구문·diff 검사 통과. GPU5·1024환경 실제 무작위 PUSH 크기의 단독/coupled RSI 및 평가8종 loco100%, 상자/목표 유지·참조 관절·AMP·정책16step·부분 reset을 확인했다. 기존 네 과제 RSI/혼합 및 학습 초기화 연결은 이미 screening된 Stage1 크기를 동일 물리 상자에 적용한 fixture로 검증했다. Stage1 전이·동결·lr2e-5·finite 출력을 확인했으며 학습 루프를 검사 함수로 대체해 optimizer step은0회다.
- GPU5·1환경 VNC 렌더/PNG·정책 추론 통과. 결과는 `output_etc/check/stage2_jointpush_rsi/`에 있다. 전체 무작위 크기의 새 RSI 캐시 생성·PPO 업데이트·장기 안정성은 미검증이며 최초 실행 시 누락 캐시 생성 시간이 필요할 수 있다.

### Stage2 jointpush 실행 연결·Stage1 PUSH 제로샷

- 승인한 24개 후보 bank(장면5~8개·미사용16~19개 격리), 순차80/20 그룹·단독10/20/25/20/25, 학습/평가 loco100%, X/질량 인원수 배를 환경에 연결했다. 구역 관측/GTA·구역 CONTACT·물리 중심 AT·전원 접촉 gate, 사람 고정32후보·coupled 뒤쪽180°/목표±45°·직사각형 경로 검사를 적용했다. 단독 PUSH의 기존 경로 조건과 네 단독 과제 sampler/플랫폼을 유지한다.
- 환경/학습 YAML·train/test/VNC·전이/정규화/clip/checkpoint 계약을 연결했다. Stage1 PUSH checkpoint를 평가에 직접 입력하면165개 항목과 정규화 통계를 전이하고 coordination은 identity로 초기화한다. GPU5 기본 명령과 옵션은 config, 계약은 reference/stage2_jointpush.md에 기록했다.
- CPU46개·shell 구문·diff 검사 통과. GPU5·1024환경에서 공동/단독/각 과제/혼합8종 reset, 거리·뒤쪽·목표 cone·질량20/40/60/80kg·마찰0.4, 정책16step finite/고정 목표·GTA passthrough·부분 reset·AMP4096×1330을 확인했다. 1024환경 학습 초기화 경로의 전이·actor encoder/scene RMS 동결·장면 minibatch4096·lr2e-5도 통과했다. 검증 함수가 학습 루프를 대체했으며 optimizer step은0회다.
- GPU5·1환경 VNC에서4명 공동 상자의 렌더·정책16step·PNG 캡처를 확인했고 해당 검증 viewer는 종료했다. 결과는 `output_etc/check/stage2_jointpush/{simulator,train_init,viewer}/`에 있다. 본학습/PPO 업데이트·공동 성공률·장기 안정성·후보 bank의 순수 비용 차이는 미검증이다.

### Stage2 jointpush 공통 모듈·GPU5 물리 교체 API 검증

- 사용자 요청으로 Stage2 연결을 시작하되 본학습은 실행하지 않았다. 순차80/20·2~남은 인원 균등 그룹, 단독10/20/25/20/25, 중복 없는 구역 할당, 공유 상자 운동을 반영한 토큰 상태, coupled 전원 양손18cm 뒤쪽 면 gate를 독립 모듈로 준비했다. PUSH165개 parameter 항목 전이·actor encoder 동결·identity coordination의 정책 출력 일치를 포함한 CPU5개 검사를 통과했다.
- GPU5·GPU PhysX·1024환경(상자만 생성)에서 X전용 벡터 scale 인자가 거부되고 스칼라 전체 scale만 허용됨을 확인했다. 개별 actor 삭제 API와 rigid shape의 치수 변경 필드도 없다.2step finite를 확인했고 이 검증 simulator는 종료했다. [결과](output_etc/check/stage2_jointpush/actor_probe/result.json).
- 이후 사용자가 후보 bank·미사용 격리를 승인했다. 아래 준비 단계의 미완료 사항은 새 연결 항목으로 갱신했다. 기존 Stage1·학습 프로세스는 보존했다.

### Stage1 jointpush 평가를 loco100%로 확정

- 사용자 요청으로 SIT/CLIMB/AT/ONTOP/PUSH 평가를 모두 loco100%로 변경했다. 학습 과제별 RSI·AMP는 유지하며, 평가의 과제 비율·크기/물성·loco 사람/상자/목표 샘플링·간섭 검사는 학습과 동일하다. 평가에서는 과제 중간 RSI와 기존 과제 RSI 캐시 생성을 사용하지 않는다. 아래 최초 구현의 평가 RSI 설정을 대체한다.
- CPU45개 통과. 실제 CPU Isaac Gym에서 다섯 과제 및 기본 혼합 preset 각각32회 reset·64명 모두 loco, 접촉 RSI0명을 확인했다. 각 preset에서 제공 PTH 정책16step·AMP4계열·finite 출력 및 gradient 검사를 통과했고, PUSH의2~10m 거리·20kg·마찰0.4도 유지됐다. 검증 로그/JSON은 `output_etc/check/stage1_jointpush/eval_loco_*`에 저장했다. 실행 명령은 동일하며 본학습은 실행하지 않았다.

### Stage1 jointpush 스크래치 설정·기존 PUSH PTH 평가 연결

- 사용자 확정에 따라 로컬 Stage1 네 과제와 KOO PUSH를 결합한 `stage1_jointpush` 환경/학습 YAML·train/test/VNC를 추가했다. teacher·자동 checkpoint 전이 없이 스크래치 학습하며, 기존 실험은 기존 경로를 사용한다. Stage2 coupled는 포함하지 않는다.
- 과제 prior SIT/CLIMB/AT/ONTOP/PUSH=10/20/25/20/25, PUSH 20kg·마찰0.4·167종 cuboid·밑판 없음. loco 사람→상자와 loco/접촉 RSI 상자→목표를 각각 반경2~10m·360°로 제안하고 간섭 시 다시 샘플링한다. 기존 참조 사람 위치/yaw·SUPPORT 배치 규칙을 사용하며, 평가도 과제별 RSI/AMP와 배치 설정을 유지한다. 상세는 [명세](markdowns/reference/stage1_jointpush.md), 명령은 [config](markdowns/config.md)에 기록했다.
- CPU 테스트45개 및 기존 PUSH 회귀3개 통과. 제공된 `Stage1PushFinetune_00010500.pth`의165개 모델 항목을 변환 없이 strict 로드했고, CPU Isaac Gym에서 PUSH·기존 과제·PUSH+ONTOP 장면을 각각3회×12step 평가했다. 실제32회 reset에서 두 RSI 경로·거리·참조 yaw·20kg/마찰0.4/단일 충돌 형상을 확인했다. 정책16step의 고정 목표·finite 출력, AMP4계열512×1330 입력, 새 모델158개 parameter tensor의 gradient 업데이트도 확인했다. [검증 결과](output_etc/check/stage1_jointpush/simulator/simulator_verification.json).
- 2048환경 확인 학습은 초기화 후 기존 과제 RSI 크기5441개 캐시 생성 단계에서 이 작업의 프로세스만 종료했다. 전체 PPO 반복·장시간 성능·GUI/VNC 실행은 미검증이며 본학습은 시작하지 않았다. 로그는 `output_etc/check/stage1_jointpush/`에 보존했다. shell 구문 및 diff 검사 통과.

## 2026-10-25

### RSI uniform18K 감시·조건부10–15kg/20cm/130° 전환

- 사용자 확인에 따라30분 간격 사용자 서비스 `tokenhsi-push-rsi-uniform-18k-monitor-20261025.service`를 시작했다. 기존 학습/PID1894083은 유지하며18000 저장본을 동일 초기512건·200초·평균/확률 행동으로 평가한다. 평균 성공≥115/512와10초 정체≤255/512를 모두 회복하면 유지하고, 하나라도 미달이면 기존 서비스만 종료한 뒤 새 variant를 fixed78000·새 optimizer/epoch으로 시작한다.
- `stage1_v2_fixed_push_cuboid_rsi_uniform_m10to15_r20cm_a130` 환경/학습 YAML·train/test/VNC·물성 bank를 분리했다. 같은 부피 선형 질량10–15kg·PUSH 성공/보상/평가 수평 반경20cm(기존 높이 오차1mm 유지)·밀기 면 정면±65°를 적용한다. RSI uniform2–12m·보행2–10m·KL0·2048환경을 유지한다. 기존 variant 성공 조건·물성·모션·bank는 보존한다.
- CPU74개 통과(없는60000 fixture 대신 실제78000을 읽어 전이를 검증; 별칭/원본 변경 없음). GPU2048환경에서 실제 질량10.159–14.623kg·spawn 최대64.994°·RSI/보행 거리·teacher 미생성과 성공 반경을 확인했다. epoch2→3/optimizer96→144 재개, 각271 scalar·모델/optimizer finite·1환경8step 추론을 통과했다. [검증](output_etc/check/push_light130_20261025/validation.json).
- 감시기는13K 기준 유지/14K 악화 전환 판정과 잘못된 PID·평가 물성 거부를 확인했다. 현재 `waiting_for_18000`·기존/감시 서비스 active·`Linger=yes`·보호된 Stage2 학습 유지 상태다.18K 평가·조건부 종료·다음 본학습은 아직 실행하지 않았다. [감시 상태](output_etc/check/push_uniform_18k_monitor_20261025/activation_validation.json), [전환 명세](markdowns/reference/stage1_push_light130.md).

## 2026-10-24

### KL0 본학습 서비스 재시작

- 사용자 요청으로20:05의 CUDA illegal memory access 종료를 확인하고20:16에 기존 사용자 서비스를 재시작했다. 저장 checkpoint가 없어 원본 fixed78000에서 동일 KL0·2048환경·RSI2~12m·loco2~10m로 새로 시작했다. 기존 로그/출력은 보존했다.
- 학습 PID1894083의 epoch1/frame0~epoch13·271개 scalar finite·KL loss0·새 구간 traceback 없음과 MPS 연결·`Linger=yes`를 확인했다. 다른 repo의 Stage2 PID1888896과 원본 checkpoint를 보존했다. 오류의 근본 원인은 확정하지 않았다. [검증](output_etc/check/push_uniform_kl0_restart_20261024_201610/validation.json).

### KL0·RSI 균등 거리 본학습을 fixed78000에서 새로 시작

- 사용자 요청으로 기존 cuboid 서비스/PID1525611을 중단하고, `rsi_uniform`을 원본 `Stage1V2Fixed_00078000.pth`에서 전이해 새 optimizer·epoch으로 시작했다. KL0·GPU0·2048환경·2명·4물체, RSI2~12m·loco2~10m를 적용하고 PUSH/RESUME checkpoint·확인학습 제한을 해제했다.
- 사용자 서비스 `tokenhsi-push-rsi-uniform-kl0-20261024-194136.service`·학습 PID1862738의 실행과 `Linger=yes`·MPS 연결, 원본 epoch78000 전이 보고서와 epoch1/frame0~epoch17·271개 scalar finite·KL loss0을 확인했다. 원본 checkpoint hash·다른 repo의 Stage2 학습을 보존하고 기존 cuboid 종료/GPU 해제를 검증했다. 현재 실행·상태·로그 명령을 config에 반영했다. [검증](output_etc/check/push_uniform_kl0_fresh_20261024_194136/validation.json).

### KL0 teacher 생략·RSI 시작 상한12m

- 사용자 요청으로 KL 계수0에서 teacher 생성·복제·전용 checkpoint 로드를 생략하고, 관측 마스크 처리 전에 즉시 반환한다. 원본 fixed에서 학생 모델을 처음 전이하는 데 필요한 로드는 유지한다. 기본 계수0.001과 양수 계수의 동작은 유지하며 실행 예시에 `TEACHER_KL_COEF=0`을 명시했다.
- 사용자 확인에 따라 `rsi_uniform`의 `rsi.goal_distance_max`를12m로 분리하고 loco 상한10m를 유지했다. 장면 간격은 두 상한의 최댓값을 고려하며, 새 거리 설정은 명시적 전이에서만 차이를 허용한다. 이전 checkpoint의 일반 resume는 저장된 거리를 유지한다.
- CPU66개 통과. GPU0·2048환경의 전이/재개에서 teacher 생성·복제·전용 로드0회와 KL loss0, RSI1025명2.006~11.995m·loco1023명2.004~9.999m를 확인했다. epoch2→4/optimizer step96→192·각271 scalar 및 모델/optimizer finite·1환경8step 추론을 검증했다. 기존 본학습은 유지했으며 새 본학습은 시작하지 않았다. [검증](output_etc/check/push_teacher0_rsi12_20261024/validation.json).

### Cuboid RSI 균등 거리 variant

- 사용자 명세에 따라 `stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform` config·train/test/VNC를 분리했다. 기본 false인 `rsi.uniform_goal_distance`가 참이면 RSI의 clip 진행도에 따른 상자 이동을 생략하고, epoch0부터 최종 커리큘럼을 사용한다. 보상·물성·bank·RSI 비율·1200step을 유지하며 config/structure/USAGE에 연결했다.
- cuboid의 명시적 모델 전이에서만 새 RSI flag 차이를 허용하고, 일반 resume 계약은 유지했다. 실제 checkpoint 전이에 필요했던 동일 teacher의 상대/절대경로 비교를 보정했다. 관련 CPU60개·shell 구문 검사를 통과했다.
- GPU0·2048환경에서 RSI1025명의 거리2.005~9.996m·1m 구간별10.8~13.9%와 epoch0 최종 단계를 확인했다. cuboid24500 모델/정규화 그대로 전이·fresh optimizer, epoch2 저장→epoch4/optimizer step96→192 재개, 모델/optimizer·각271 scalar finite와1환경8step 추론을 검증했다. 기존 cuboid 본학습·원본 설정/bank/보상/모션을 보존했으며 새 본학습·GUI viewer는 시작하지 않았다. [검증](output_etc/check/push_rsi_uniform_20261024/validation.json).

## 2026-10-23

### Cuboid PUSH 물성·보상 설계 문서

- 사용자 요청으로 [cuboid 전용 문서](markdowns/reference/push_cuboid_reward_design.md)에 크기·부피별 질량·본체/밑판 마찰과 접촉/AT 보상식·성공·포화·페널티·AMP 합산만 정리하고 structure에 연결했다.
- 실행 중인 환경 YAML과 현재 PUSH 설정 일치·실제 보상 호출 경로, 예시 수치·문서 링크·diff를 확인했다. 문서만 변경했으며 학습 코드/config·프로세스는 유지하고 새 학습/시뮬레이션은 실행하지 않았다.

### 정육면체 raw 중단·cuboid 백그라운드 재개

- 사용자 요청으로 정육면체 raw 서비스와 학습 PID397999만 중단하고, cuboid의 최신 epoch13000 저장본에서 모델·optimizer·epoch을 재개했다. GPU0·2048환경·2명·4물체를 유지하며 별도 출력의 `launch.sh`에 실행 명령을 보존하고 config의 상태·로그 명령을 갱신했다.
- 사용자 서비스 `tokenhsi-push-cuboid-resume-20261023-203500.service`·학습 PID1525611의 실행과 `Linger=yes`, MPS 연결, epoch13001~13008·271개 scalar finite를 확인했다. 저장된 PUSH 설정 일치·확인학습 제한 해제·raw 프로세스/GPU 점유 해제와 다른 repo의 Stage2 학습 유지를 검증했다. [검증](output_etc/check/push_cuboid_resume_20261023_203500/validation.json).

## 2026-10-22

### 실제1환경·6명·5과제 attention 시각화

- 사용자 요청으로 정육면체9000/cuboid8000을 SIT·CLIMB·carryAT·carryONTOP·PUSH-loco·PUSH-RSI의6명 장면에서 각각600step 평가했다. 운영2명 PUSH 환경/학습 코드는 유지하고 평가 전용 장면에서 H6/O7/G6 전체19×19의 QK·QK softmax·bias·QK+bias·최종 softmax를 actor/critic4개 레이어·2개 head·41시각에 저장했다. PNG·실제 장면/확대 MP4·인터랙티브 뷰어와 실행 명령은 [진단 보고서](output_etc/check/push_attention_six_20261022/report.md)에 둔다.
- 실제 simulator1환경·6명·7상자, 모델 strict 로드, 캡처 출력/원래 레이어 오차0·action 오차0, token 순서 고정 action 차이8.4e-7 이하·독립 softmax 오차3.6e-7 이하를 확인했다. 브라우저의 선택기·10개 행렬 렌더·H264 실제 재생과 최종43개 검사를 통과했다. 본학습 코드/config/checkpoint 및 두 학습 서비스는 보존했다. attention 연결과 행동 성공의 해석은 구분한다.

## 2026-10-21

### PUSH 두 학습의 MPS 연결·백그라운드 재시작

- 사용자 요청으로 기존12~20kg 서비스만 중단하고 해당 run의 최신 저장본(epoch1000)에서 모델·optimizer·epoch을 재개했다. 중단된 cuboid는 fixed78000 전이로 새로 시작했다. 원래 출력과 설정을 보존하고 별도 출력에 실행 명령·로그를 저장했으며 config의 상태/로그 명령을 갱신했다.
- GPU0·각2048환경의 사용자 서비스 `tokenhsi-push-{raw,cuboid}-mps-20261021-162424.service`를 실행하고 로그아웃 후에도 유지되도록 linger를 활성화했다. 두 Python PID397999/397998이 같은 MPS 서버388225에 연결되고 raw epoch1001~1016·cuboid epoch1~17 및 각각271개 scalar finite임을 확인했다. MPS 속도 향상은 측정하지 않았다. [검증](output_etc/check/push_mps_restart_20261021_162424/validation.json).

### GPU0 MPS 활성화

- 현재 머신에서 MPS 데몬이 없고 두 학습이 비MPS로 실행 중임을 확인한 뒤 기존 helper로 GPU0의 사용자 전용 MPS를 시작했다. 서버PID388225·CUDA 행렬 연산 클라이언트의 실제 연결·GPU PhysX3규격 통과를 확인했다. 다음 실행은 runtime_env에서 자동 연결한다.
- 기존12~20kg PID232133·직육면체 PID383531과 GPU compute mode는 유지했다. 실행 중인 학습은 자동 전환되지 않으며 속도 향상은 측정하지 않았다. config 공통 규칙에 시작/조회 명령을 기록했다. [검증](output_etc/check/mps_enable_20261021/validation.json).

### PUSH 직육면체·부피별15~20kg·1200step

- 사용자 요청으로 `stage1_v2_fixed_push_360_raw_cuboid` config·train/test/VNC·bank를 분리했다. X1.2~1.7m, Y/Z1.45~1.7m를 추출·정렬해 Y≥Z를 보장하고 부피에15~20kg을 선형 대응한다. 본체0.9·밑판0.3/1cm/전체 폭·기존 RSI/보상/커리큘럼을 유지하며 PUSH는1200step이다. 실행 명령은 config, 계산식은 raw 참고 문서에 기록했다.
- CPU57개와 GPU3규격의 질량·직육면체 관성·마찰·접지 검사를 통과했다. 기존4종 sampler의4096개 샘플과 RNG는 bitwise 동일했다. fixed78000에서2048환경 확인학습1회 업데이트(epoch2 저장)·271개 scalar/모델 finite·실제 상자2048개의 범위/질량 매핑, 저장본의1환경·8step 한도 추론을 확인했다. 기존12~20kg 학습·이전 설정/bank/원본 checkpoint를 보존했으며 새 본학습은 시작하지 않았다. [검증](output_etc/check/push_cuboid_20261021/validation.json).

### PUSH 20~25kg 실험 중단

- 사용자 요청으로 `tokenhsi-push-mass20to25-20261021-144954.service`만 중단했다. 대상 cwd·학습 config를 확인하고 wrapper/Python PID324484/324511 종료와 GPU 점유 해제를 검증했다. 기존12~20kg 학습 PID232133은 계속 실행 중이며 소스·설정·결과 파일을 보존했다. [확인](output_etc/check/push_mass20to25_20261021/stop.json).

### PUSH 20~25kg 질량 variant·백그라운드 풀 학습

- 사용자 요청으로 `stage1_v2_fixed_push_360_raw_mass20to25` 환경/학습 config·전용 train/test/VNC·물성 bank를 분리했다. 1.2/1.45/1.7m에20/22.5/25kg을 선형 대응하며 본체0.9·밑판0.3·RSI·보상·1800step을 유지한다. 실행·시각화·서비스/로그 명령을 config 맨 아래에 추가했다.
- CPU31개와 GPU9개 물성 비교를 통과했다. 새 실제 실행의 상자2048개 질량 매핑·fixed78000 전이·2048환경·5epoch/271개 scalar finite를 확인했다. `tokenhsi-push-mass20to25-20261021-144954.service`로 GPU0 백그라운드 풀 학습 중이며 기존12~20kg 학습과 원본 설정·bank·checkpoint·보상/RSI 코드를 보존했다. [검증](output_etc/check/push_mass20to25_20261021/validation.json).

### PUSH 밑판 전체 폭·백그라운드 풀 학습

- 사용자 요청으로 밑판의 가로/세로를 상자와 같게 변경했다(`inset: 0.0`). 두께1cm·본체0.9/판0.3·외곽 크기/질량/관성을 유지하고 이전 bank를 보존했다. `config.md`는 현재 raw PUSH의 학습·학습 시각화·추론·실행 중인 서비스/로그 명령으로 좁혔다.
- CPU31개와 GPU3크기 물리 비교를 통과했다. 판0.3의 감속 기반 실효 마찰은약0.6368이다. GPU0·2명·2048환경·4물체 풀 학습을 fixed78000에서 새 optimizer/epoch으로 시작했고9epoch·271개 로그 scalar finite·실제 밑판2048개의 가로/세로 일치를 확인했다. 대화 세션과 분리된 사용자 systemd 서비스 `tokenhsi-push-raw-full-20261021-130725.service`로 실행하며 기존 학습은 유지했다. [검증](output_etc/check/push_bottom_plate_full_20261021/validation.json).

### PUSH raw 본체0.9·안쪽 밑판0.3

- 사용자 요청으로 raw PUSH 담당 상자만 본체 마찰0.9·밑판0.3인 하나의 강체/두 충돌 형상으로 변경했다. 밑판은 두께1cm·가장자리1.5cm 안쪽이며 외곽 크기·접지 높이·질량·중심·정육면체 관성을 유지한다. 이전 bank/config·기존 과제 상자·사람·ground plane은 보존했다. fixed78000 풀 학습/4환경 학습 시각화 명령을 config에 명시했다.
- CPU48개 통과. GPU PhysX에서1.2/1.45/1.7m·12/16/20kg의 판0.3/판0.9/원래 상자를 비교했고, 판0.3의 감속 기반 실효 마찰은0.6373~0.6375였다. 실제 혼합2환경·16step에서 PUSH0.9/0.3·기존 상자1.0·발1.0·땅1.0, 질량·관측/보상 finite를 확인했다. optimizer 갱신0회이며 사용자 최종 요청에 따라 본학습·학습 시각화·백그라운드 학습은 시작하지 않았다. [검증](output_etc/check/push_bottom_plate_20261021/validation.json).

### PUSH raw 양손 공통 면 XY 접촉 보상

- 사용자 합의에 따라 raw CONTACT state와 유효 접촉에서 손 중점·기준 높이·높이 허용폭/감점을 제거했다. 같은 지정 면의 XY 투영까지 양손 거리를 계산해 더 먼 거리 `d`로 `exp(-10*d*d)`를 사용하고, 면 가장자리 초과 거리도 포함한다. 유효 접촉은 양손 거리≤18cm와 기존0.5N 법선 힘 조건을 사용한다. 면 재선택 없이 기존 면·접근/AT/AMP·RSI 배치를 유지하며 이전 checkpoint는 저장된 보상을 복원한다.
- 관련 CPU43개 통과. 중점/높이 무관성·유한 면 가장자리·더 먼 손·동일 면 유지·손별/상자 힘 조건·상자 회전/기울기·실제 환경의 보상 호출·기존 보상/전이/포화를 확인했다. GPU 시뮬레이션·학습은 실행하지 않았다. [검증](output_etc/check/push_face_xy_20261021/validation.json).

### PUSH raw 크기별12~20kg·fixed78000 전이

- 사용자 요청으로 raw 시작 checkpoint를 `output/stage1/Stage1V2Fixed_00078000.pth`로 변경하고, 한 변1.2~1.7m에 질량12~20kg을 선형 대응시켰다(1.45m→16kg). 새 물성 bank를 분리하고 학습/noVNC 명령을 갱신했다. 기존 bank·원본 checkpoint·RSI 후보와 모션은 보존했다.
- 관련 CPU24개 통과. 질량 범위·크기별 단조 증가·선형 대응·기존 독립 질량 샘플러와 고정 물성·전이 계약을 확인했다. 실제 fixed78000의 actor/critic/discriminator 전이와 기존 과제 출력 비교도 통과했다. 기존 GPU 작업을 유지하기 위해 시뮬레이션·학습은 실행하지 않았다. [검증](output_etc/check/push_mass_by_size_20261021/validation.json).

### PUSH 15kg 비교 시각화

- 질량 비교 실험 폴더에15kg noVNC 실행기를 추가했다. 기존 raw checkpoint의 상자 생성 질량만 바꾸고, 기본 접촉 RSI·100회·300step의 기존 평가 흐름을 사용한다. 명령은 USAGE와 config 링크에 기록했다.
- 11000epoch 저장본으로1환경·8step 화면 없는 추론을 완료하고 simulator 실제 질량 두 개가15kg임을 확인했다. 학습 코드/config·실행 중인 학습은 변경하지 않았고 GUI viewer는 실행하지 않았다.

### PUSH raw 테스트 시각화 명령

- `config.md`의 raw 로컬/VNC 명령에 현재 학습 run의 `RAW_CKPT`를 별도로 지정하고, 원본 RSI·접촉 자세 시작·1환경·100회·최대300step을 명시했다.
- 참조 checkpoint(epoch10500)의 raw 설정·물성·호환 계약, 실행 파일/경로·문서 링크·쉘 구문을 확인했다. 학습·viewer는 실행하지 않았다.

## 2026-10-20

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
