# Changelog

최신 변경부터 기록한다. 현재 실행법은 [config](markdowns/config.md), 코드 위치는 [structure](markdowns/structure.md)를 따른다. 과거 변경·검증 전체는 [보관 changelog](markdowns/legacy/changelog_before_cleanup.md)에 있다.

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
