# Changelog

최신 변경부터 기록한다. 현재 실행법은 Stage 1 [config.md](markdowns/config.md)·Stage 2 [config_stage2.md](markdowns/config_stage2.md), 코드 위치는 [structure.md](markdowns/structure.md)를 참조한다. 실행 중인 GPU/PID는 이 파일에 고정하지 않고 실제 프로세스로 확인한다.

## 2026-10-02

### 현재 Shared9/CPA 환경의 Stage 1 baseline 실행 안내

- Stage 2 평가 환경을 유지하고 Stage 1 epoch8000 정책만 사용하는 `STAGE1_ONLY=1`의 Shared9/CPA random viewer·VNC 명령과 비교 조건을 `config_stage2.md` 기존 평가 섹션에 추가했다. 정책/RMS 이식과 협력 action 입력 0 처리는 기존 경로를 사용한다.
- 관련 CPU 테스트 **1개 통과**(정규화된 action 동등성·학습 0·협력 입력 기여 0). CPA wrapper 인자 dry-run·문서 Bash 7블록/스크립트 4개 문법·checkpoint 경로·diff 확인. GPU 평가·viewer는 실행하지 않았다.

### Viewer 목표 색상 조건 수정·SIT/ON_TOP 모양 구분

- 목표 표시는 같은 destination 물체·같은 relation을 맡는 서로 다른 owner가 2명 이상일 때만 노란색이다. AT+CLIMB·SIT+CLIMB 등 다른 행동은 각각 owner 색을 사용한다. 물체 자체의 공유 색상은 기존 물체별 owner 기준을 유지한다. SIT는 수평 원형 링, ON_TOP은 입체 상자 테두리, CLIMB은 기존 3축 십자가로 구분한다.
- 관련 CPU **144개 통과**: Shared9 9조합/별칭·2인/4인 색상, AT+CLIMB 재현, 같은 owner 중복/무효 edge, marker 형태·기존 AT destination 표시를 확인했다. 실제 marker 좌표로 만든 CPU 미리보기의 모양·Python 문법·diff를 확인했다. GPU 시뮬레이션·GUI 화면 확인은 실행하지 않았다. 기존 viewer는 다시 실행해야 적용된다.

### Viewer AT goal 누락 수정·공유 목표 노란색 통일

- AT marker 표시 여부를 edge owner가 아닌 실제 destination goal 번호로 계산한다. H0→G1처럼 goal을 랜덤 배정할 때 목적지가 Z=20으로 숨겨지던 오류를 수정했다. 공유 SIT·CLIMB·ON_TOP 목표 십자와 물체는 같은 노란색을 사용하며, ON_TOP 받침을 포함한 유효 edge의 물체별 서로 다른 owner로 공유 여부를 판단한다.
- 관련 CPU **141개 통과**: goal/owner 불일치·부분 reset·AT 없는 과제·공유9조합/별칭·2인/4인·겹친 SIT 두 표시·독립 색상을 확인했다. Python 문법·diff 확인. GPU 학습/평가·GUI 화면 확인은 실행하지 않았으며 0.7/0.3 보상 실험은 적용하지 않았다. 기존 viewer는 다시 실행해야 표시 변경을 읽는다.

### Shared9 random_scenario의 학습 9조합 복원·범용 평가 연결

- 이전 KLClimb50 기본 전환으로 Shared9 6조합이 빠진 것을 수정했다. Shared9/CPA test·VNC 및 config의 기본 평가를 전용 범용 Shared9로 연결해 학습의9조합·2인 비율을 유지하고 1환경에서도 reset마다 다시 뽑는다. 무작위2인 그룹·홀수 잔여자·물체 부족 시 조합 재정규화·소유자 HOLDING/ON_TOP 고유 source·2M edge 계약을 검사한다. 인원 확장 checkpoint를 원래 Shared9 학습 graph와 대조하며 정적/CPA 혼용 거부를 유지했다.
- 평가의 고정 asset은 공유 받침 학습 범위와 ordinary/payload 교집합40×40×40~50cm를 사용한다. 이는 단일 viewer의 AT/공유 재샘플링을 위한 평가 크기이며 학습의 전체 크기 분포와 구분한다. 초기 body 검사도 모든 사람 쌍으로 확장했다. 학습 YAML·크기 풀·보상·기존 학습 프로세스는 유지하고 Stage 2 표·학습/평가/VNC 안내·구조 문서를 갱신했다.
- 관련 CPU **109개 통과**. 실제 Shared9 epoch1500·CPA epoch500의 **2명/4상자·1환경·32step·64회**에서 각각9조합 전부+독립을 확인했다. 각각 **4명/6상자·1환경·32step·랜덤6회+최종 독립2회** 평가·strict 로딩·finite JSON 저장 완료. 자료는 `output/shared9_default_eval_check/verification.json`이다. Bash 문법·문서 전체 섹션/명령/링크·diff 확인. GUI 화면과 장기 협력 성능은 검증하지 않았다.

### Shared9 viewer 기본 샘플러·인원 확장 수정 (이후 9조합 복원으로 대체)

- Shared9/CPA test·VNC 기본 평가를 KLClimb50으로 전환했다. 1환경의 고정 크기 풀 때문에 독립 과제만 반복되던 경로를 피하고 reset마다 협력/독립을 다시 뽑으며 O≥M≥2로 확장한다. `EVAL_SAMPLER=shared9`는 기존 2명·4상자 9조합·역할별 크기 평가를 유지한다. 평가 override는 학습에서 거부하며 학습 YAML·보상·정적/CPA 계약·기존 학습 프로세스는 유지했다. Stage 2 실험 표·본학습/평가/VNC 안내·코드 위치를 갱신했다.
- 관련 CPU **98개 통과**. 실제 Shared9 epoch1500·CPA epoch500으로 각각 **2명/4상자·4명/6상자, 1환경·32step·3회 headless 평가** 완료: strict checkpoint 로딩·reset별 과제 변경·finite JSON 저장 확인. 자료는 `output/shared9_klclimb_eval_check/verification.json`에 있다. Bash 문법·문서 링크/명령·diff 확인. GUI 화면·확장 인원 협력 성능은 검증하지 않았다.

### Shared9 사람–사람 CPA 실험 연결

- `approach_stage2_rescue_shared9_cpa` 환경/PPO config·전용 train/test/VNC·별도 output을 추가했다. Shared9 graph·크기·loco RSI·AMP·edge 보상은 유지하고 기존 사람–사람 정적 거리 항을 XY CPA로 교체한다(계수0.5·거리0.7m·제어 step 시간 할인0.99). 사람–상자 보상 항은 추가하지 않았다. 정적/CPA checkpoint 계약을 저장·검사하며 과거 정적 checkpoint는 기존 config로 로딩 가능하다. 실행 명령·코드 위치·`COLLISION_PENALTY.md`의 현재 구현/미구현 설계를 갱신했다.
- 관련 CPU **111개 통과**. Stage 1 epoch8000에서 **2048환경·MAX_ITERATIONS=1 학습/저장** 완료(저장 epoch2); CPA config·184 scalar finite·encoder75 tensor/actor RMS 고정·정적/CPA 교차 로딩 거부를 확인했다. 저장 pth의 **16환경·32step·2회 headless 평가·finite JSON 저장** 완료. 자료는 `output/approach_stage2_rescue_shared9_cpa_check/`에 있다. 문서 링크·Bash 문법·diff 확인. 기존 두 학습 프로세스는 유지했고 CPA 본학습·장기 회피 성능 검증은 실행하지 않았다.

### Stage 2 실행 가이드 정리

- `config_stage2.md`에 누적된 진단 명령·설계 설명을 분리하고 공통 규칙·현재 실험·학습·로컬 평가와 서버 VNC의 기존 구조로 정리했다. Shared9 본학습 명령을 먼저 배치하고 6개 실험의 train/test/VNC·baseline·3인 viewer 명령을 유지했다. 실측/RSI/attention 재현 명령은 `tokenhsi/docs/stage2_diagnostics.md`로 옮겨 원문을 보존하고 `structure.md`에 위치를 반영했다.
- 두 문서의 Bash 문법·링크·스크립트/설정 경로·기존 진단 명령 보존·diff를 확인했다. 문서만 수정했으며 학습/config/실행 프로세스는 변경하지 않았다.

### Rescue Stage 2 checkpoint별 협력 attention 추세

- 지정 latest를 epoch11000으로 고정하고, epoch500~11000 **22개 pth·동일 관측19,314개**의 사람/역할/head별 CA·edge 마스킹 action 변화를 비교했다. 별도 진단 subprocess의 hook만 사용하며 기존 학습 config·프로세스는 유지했다. 도구·재현 명령은 `structure.md`·`config_stage2.md`, PNG/HTML·원자료는 `output/stage2_attention_trends/`에 기록했다.
- 7개 epoch의 CLIMB/SIT/STACK 양쪽 역할·독립 대조군과 최신 CA 제거/균등 routing/동료 edge 제거, **총3,360 scene·20초 horizon** 완료. 초기 관측·graph·asset 크기가 동일했다. 최신 동시 ever는 CLIMB58/SIT52/STACK70(각96회), 동료 edge 제거는43/38/55회였다. Attention 선택성과 action 영향은 증가했으나 성공률은 단조 증가하지 않았고, temporal DAG·CA 깊이 충분성의 증명과 구분한다.
- CPU **3개 통과**, 14환경·32step smoke·전체 finite action·CA 출력 복원 최대 오차7.16e-7 이하·문법/diff·PNG 시각 확인. 개별 edge 제거는 동일 관측의 CA 경로 검사이며, 전체 CA/균등/동료 edge 제거에는 실제 rollout을 수행했다.

### Rescue Shared9 본학습 연결

- `approach_stage2_rescue_shared9` config·PPO/AMP 설정·전용 train/test/VNC를 추가했다. Stage 1 epoch8000 encoder/actor RMS를 고정하고 9조합을 AT 각15%·나머지 각7.5%·독립10%로 구성한다. 2048환경을 독립205·AT922·공유921개의 고정 크기 풀로 나누며 reset마다 풀 내 조합·역할·논리/물리 배정을 바꾼다. AT/독립은 기존 크기, 공유 받침50~65×65~70×45~50cm·source30~40×30~40×40~50cm를 독립5cm 격자로 생성한다. Double ON_TOP만 새 sampler에서 허용하며 source 고유성·owner HOLDING 연결을 검사한다.
- RSI는 모든 과제를 loco로 시작하고 기존 AMP 전문가8종·무조건부10프레임/1290차원을 유지한다. 초기 FK body 중심8cm sphere proxy·실제 크기 반경 기반 배치·초기 성공 재추첨을 연결한다. 기존 reward/plane 여백·학습 프로세스는 유지하며 Shared9 학습·평가·VNC 명령을 `config_stage2.md`, 코드/테스트 위치를 `structure.md`에 반영했다.
- 관련 CPU **87개 통과**. 실제 epoch8000으로 **2048환경·MAX_ITERATIONS=1 학습/저장** 완료: accepted reset4,103scene·9조합 배정·loco RSI·AMP demo393호출·history46slot 확인, 실패0. Encoder75tensor·actor RMS 원본 동등, 협력 head/critic/AMP 갱신·scalar363개 finite. 조합/독립 각6개씩 **60개 초기 snapshot**의 별도 CPU PhysX FK·전체 충돌 형상 검사에서 사람–상자/사람–사람 관통0. 자료는 `output/approach_stage2_rescue_shared9_check/`에 있다.
- 저장 pth와 Stage 1-only baseline의 16환경·32step·2회 headless 평가·JSON 저장을 확인했다. 이는 실행/초기 상태 검증이며 장기 협력 성능·전체 크기 조합의 완전 지지·이동 중 무충돌 보장은 아니다. 본학습은 시작하지 않았다.

### VS Code 업데이트

- 사용자 요청으로 시스템 VS Code를 Microsoft 공식 APT 패키지 1.127.0→1.140.0으로 업데이트했다. 패키지 SHA-256 일치·설치 명령 성공·`code --version`·dpkg 설치 상태·`dpkg --audit` 이상 없음을 확인했다. 실행 중인 VS Code에 적용하려면 사용자가 작업을 저장하고 다시 열어야 한다.

### 최대 크기의 단독·연합 9조합 확인

- 받침60×70×50cm·source30×30×50cm의 최대값을 Stage 1 epoch8000-only로 평가했다. 진단에 독립4과제 모드를 추가해 사람마다 별도 객체·goal을 배정하고 낙상한 사람의 이후 기록만 제외한다. 입력은 `summarize_transport_sharing.py --max-sizes`, 재현 명령은 `config_stage2.md` 기존 평가 섹션, 전체표는 `output/shared_scenario_audit/max_sizes/README.md`에 기록했다. 기존 학습 config·sampler·실행 프로세스는 변경하지 않았다.
- 독립 **96scene·과제당48개 개인 시행** 완료: AT45/48(실제 lift/이동 후 AT38/48), SIT44/48, CLIMB47/48, ON_TOP47/48 ever 성공. 연합 **216scene·조합당24회** 완료: AT+ON_TOP7, AT+CLIMB0, AT+SIT1, ON_TOP+ON_TOP8, ON_TOP+CLIMB12, ON_TOP+SIT8, CLIMB+CLIMB13, CLIMB+SIT9, SIT+SIT14회 동시 ever 성공. 1초 유지·전체 projection 지지를 별도 기록하며 본학습 성공률과 구분한다.
- 독립 graph4종 validator 통과·실제 실행 중 finite action·진단 Python 문법 확인. 최대 크기만으로 AT 순서 협력이 완성된 것은 아니며, 9조합 본학습 config는 아직 생성하지 않았다.

### 독립 운반 범위와 공유 6조합 교집합 측정

- Stage 1 epoch 8000·학습 0회의 독립 HOLDING+AT로 **XYZ 515점·12,360 scene·24,720개 개인 시행**을 측정했다. 실제 lift·XY 이동 이후 AT를 delivery로 기록하고 한 개인의 낙상으로 상대 시행이 잘리지 않게 진단 프로세스에서만 timeout reset·낙상 이후 기록 제외를 적용했다. 본학습 config·성공 조건·sampler·기존 실행 프로세스는 변경하지 않았다.
- 운반 결과 안에서 고유 받침 **147점**의 공유 SIT/CLIMB/ON_TOP 6조합을 source XY·Z와 0/4cm 여유로 측정했다. 맞닿음과 관통의 FCL 처리를 분리하고 접촉/1cm 겹침 대조를 확인했다. 별도 CPU PhysX의 대표 최종 배치 **136개 교차 접촉 0건·겹침 대조 136건 검출**, 실제 공유 **30크기 설정·4,320 scene rollout** 완료.
- 받침 X50/55/60·Y65/70·Z45/50cm의 12점은 source30×30×45cm에서 운반 81.25~100%와 6조합 배치 존재를 확인했다. 출발 후보는 받침55×65×50cm·source30×30×45cm이며 더 작은35×65×50cm도 운반38/48·6조합 배치 및 동시 성공 사례를 확인했다. 공간 존재·한 번 성공·지속 성공을 구분하고 학습이 가장자리 자리를 선택하는 기준으로 정리했다.
- `summarize_transport_sharing.py`로 재현 입력·교집합 JSON/CSV를 생성한다. 최신 결과는 `output/shared_scenario_audit/transport_grid/README.md`, 재현 명령은 `config_stage2.md` 기존 평가 섹션, 도구 위치는 `structure.md`에 반영했다. 진단 Python 문법 확인·실제 실행 중 finite action 확인. 새 9조합 본학습 config는 아직 생성하지 않았다.

### 운반 가능한 공유 받침 추가 실측

- 진단 도구에 정사각 65~80cm·직사각 60×70/75/80cm, Z45/50cm, source30/35/40cm 및 all-loco·무편향 reset snapshot 옵션을 추가했다. `measure_shared_rectangles.py`는 실제 CPU PhysX FK·MJCF 형상으로 직사각 최종 배치를 측정한다. 본학습 config·sampler·기존 실행 프로세스는 변경하지 않았다. 재현 명령은 `config_stage2.md` 기존 평가 섹션, 추천·조건별 범위는 `output/shared_scenario_audit/carryable/README.md`에 기록했다.
- Stage 1 epoch 8000·학습 0회로 **43크기 설정·9,288 scene rollout**, all-loco **5,040+9,288 reset** 완료. AT ever는 Z/source 합산 65cm 정사각 53.2%, 80cm 정사각 5.3%, 60×75cm 직사각 53.5%였다. 관계 성공과 실제 lift/이동·전체 projection 지지·연속 성공을 별도 저장했다. 추천 후보는 받침 60×75×50cm·source35×35×30cm이며 두 ON_TOP source 합폭+4cm 조건을 적용한다.
- 위험 여부와 무관하게 첫 reset **360개 전부** CPU PhysX 실제 FK를 확인했다. current/65cm/source30 및 60×75/source30·35 각각 0/72, 60×80/source35는 사람–상자 1/72개에서 2cm 초과 관통을 확인했다. 다른 reset의 추천 크기에도 최대 25.8cm 관통이 남아 all-loco와 초기 형상 충돌 재샘플링을 함께 권장한다. 위험 후보 71개 별도 확인: 사람–상자 61개·사람–사람 7개이며 전체 충돌률이 아니다.
- 관련 CPU **55개 통과**, 진단 Python 문법 확인. 새 9조합 본학습 config는 아직 생성하지 않았고 정책·정적 배치 결과를 본학습 성능·무충돌 보장으로 해석하지 않는다.

### 공유 9조합 RSI·역할별 상자 크기 진단

- `audit_shared_scenarios.py`로 제안 9조합의 기존 reset·Stage 1 epoch 8000-only rollout을 진단했다. graph/크기 override는 진단 프로세스에만 적용하며 본학습 sampler·기존 보상·학습 YAML·학습 프로세스는 변경하지 않았다. 정적 측정에 source 크기/높이 옵션을 추가하고 최초 실행의 plot key 오류·성공 후 포화 설명을 수정했다. 재현 명령은 `config_stage2.md` 기존 평가 섹션, 도구 위치는 `structure.md`, 상세 자료는 `output/shared_scenario_audit/README.md`에 기록했다.
- **5,040개 reset×현재/loco 대안**, epoch 8000·20초 **1,512+1,080개 scene rollout** 완료. 대표 accepted reset 60개의 별도 CPU PhysX 실제 FK에서 2cm 초과 사람–상자 교차 57개·사람–사람 교차 2개를 확인했다. 위험 후보 선택 결과이며 전체 충돌률이 아니다. shared SIT/CLIMB loco-only는 reference FK의 자기 HOLDING 외 상자 관통 후보를 400→42개로 줄였지만 운반 RSI 관통은 남았다. GPU/CPU sim 혼용 진단의 native crash 이후 CPU 확인을 별도 프로세스로 분리했다.
- 정적 최종 배치 **89건 교차 접촉 0건·대조 89건 모두 검출**. 60cm 받침의 source 20/25/30/40cm ON_TOP ever는 0/10.8/37.5/60.8%, 85cm 이상 받침의 AT ever는 검사 조건에서 0%였다. 60cm 이하 SIT+SIT의 두 plane 성공과 XY 70cm 거리 패널티 0은 동시에 불가능함을 실제 kernel로 확인했다. source 축소·받침 일괄 확대 대신 역할/조합별 크기를 권장하며 새 9조합 본학습은 아직 구현하지 않았다.
- 관련 CPU **55개 통과**, 진단 Python 문법·문서 명령 문법·링크·diff 확인. 정책 결과는 Stage 2 학습 성능·완전 지지·지속 안정성 보장과 구분한다.

### Rescue Stage 2 범용 인원·물체 랜덤 평가

- 2명·4물체 이외의 Rescue 평가를 M≥2·O≥M 범용 sampler로 자동 확장한다. Reset마다 협력 그룹·운반자·동료 과제·상자·goal을 뽑으며 홀수 잔여자도 독립 과제를 받는다. 물체 예산에 맞춰 독립 ON_TOP을 제한하고 공유 받침당 ON_TOP 1명·고유 HOLDING·goal·최대 2M edge 계약을 유지한다. 초기 상자와 AT 목표는 실제 상자 반경·여유 공간을 고려해 배치한다. 기존 2명·4물체 학습·explicit 평가·checkpoint 계약을 유지하고 `config_stage2.md`의 표·학습·평가·VNC와 `structure.md`를 갱신했다.
- 관련 CPU **63개 통과**: 6개 인원/물체 조합·5개 preset·홀수 공유 그룹·자원 제약·재현성·평가 전용 제한·확장 모델/RMS strict 로딩·활성 협력 head의 edge 순열 동등성·초기 배치 여유 공간을 확인했다. 실제 Stage 2 pth로 **3/3·4/5·5/5·5/8·8/16(사람/물체), 각각 16환경·32step·2회 headless 평가**와 finite JSON 저장·reset 배정 변경을 확인했다. Stage 1 epoch 8000 baseline도 4/5·16환경·32step 평가·협력 입력 가중치 0을 확인했다. 자료는 `output/approach_stage2_rescue_klclimb50_general_check/`에 있다. 문서 링크·Bash 문법·diff 확인. 본학습은 실행하지 않았으며 GUI 행동·확장 인원 협력 성능은 확인하지 않았다.

### Rescue Stage 2 3인 운반·공유 CLIMB viewer

- H0의 HOLDING O0·O0 AT G0와 H1/H2의 CLIMB O0를 고정하는 평가 graph를 추가했다. Rescue의 explicit graph는 평가에서만 허용하며 무조건부 AMP 10프레임·기존 reward/checkpoint 검사는 유지한다. 기존 2인 학습 설정·실행 프로세스는 유지했다. `config_stage2.md`의 실험 표·기존 평가/VNC 섹션에 Stage 2 epoch 7500·3인·1환경·4상자·60×60×50cm 평가 명령을 추가했다.
- 관련 CPU **24개 통과**: 공유 endpoint·owner·required 연결, 평가 전용 허용·훈련/AMP 불일치 거부, 2→3인 모델/RMS strict 로딩·882차원 관측·3인 action을 확인했다. 실제 epoch 7500으로 **3인·16환경·32step headless 평가·JSON 저장 완료**, H0/H1/H2의 담당 상자가 같은 물리 slot임을 로그에서 확인했다. 자료는 `output/approach_stage2_rescue_klclimb50_three_agent_two_climb_check/`에 있다. 이는 실행 검증이며 GUI 화면·3인 협력 성능은 확인하지 않았다.

## 2026-10-01

### 공유 받침 ON_TOP/SIT/CLIMB 6조합 공간 측정

- 별도 CPU PhysX의 실제 humanoid 자세·MJCF sphere/capsule·발 mesh convex hull로 정사각 받침 X=Y 0.40~1.50m, Z 0.30~0.60m를 5cm 격자로 측정하는 `measure_shared_support.py`를 추가했다. SIT 20·CLIMB 7모션의 79자세·12배치를 사용하며 CLIMB은 직립 root·다리로 정렬하고 상체 관절을 유지한다. ON_TOP은 올리는 상자 0.40×0.40×0.30 / 0.60×0.60×0.50m, 운반자를 제외한 최종 배치만 검사한다. 접촉 범위 2cm/형상을 고려해 점유자 사이 4cm 여유·완전한 상자/발 지지·바닥/받침 간섭을 확인했다.
- 크기별 실제 통과 비율을 저장해 큰 상자에서 항상 통과한다는 단조성 가정을 피했다. 정책 rollout·운반/접근 경로·중력하 안정성 결과와 구분하며, SIT 자세 전체에 대한 무충돌 최소 크기는 확인하지 못했다. `output/shared_support_measurement/`의 JSON·CSV·보고서·그림에 조건과 한계를 기록한다. `config_stage2.md`의 기존 평가 섹션에 재현 명령, `structure.md`에 진단 도구 위치를 추가했다.
- 대표 최종 배치 **34건의 PhysX 교차 접촉 0건**, 겹친 대조 배치 **34건 모두 접촉 검출**. Python/문서 Bash 문법·diff 확인. 추가 라이브러리는 격리 경로에 설치했으며 학습 config·기존 학습 프로세스는 유지했다.

### Stage 2 human·edge·action 번호 연결 회귀 검증

- 과거 순서 오류 우려에 대응해 Rescue CPU 테스트에 협력 입력 가중치를 0이 아닌 값으로 활성화한 검증 9건을 추가했다. 네 과제·양쪽 협력 역할에서 토큰 원복 후 human readout, edge source/target 특징 조회, human Query 순서, edge 순열의 action·sigma, 토큰 순열의 협력/action head gradient 동등성을 확인한다. Human·물체·goal·edge 번호를 함께 바꾸면 action도 같은 human 순서로 바뀌는지 검증한다.
- 관련 CPU **24개 통과**, diff 확인. 관측에서 물리 human 순서가 보존되고 네트워크의 `(env, human)` 출력 순서와 물리 action 입력 순서가 일치함을 코드로 확인했다. 실행 코드는 변경하지 않았고 실행 중인 학습은 유지했다.

### Stage별 실행 문서 갱신 의무 명확화

- `AGENTS.md`와 Stage 1/2 실행 가이드에 config·checkpoint·실행 옵션 변경 시 해당 Stage 문서의 실험 표·학습·평가·VNC 갱신을 필수로 명시했다. 채팅에만 명령을 남긴 채 완료하지 않도록 했다. `config_stage2.md`의 기존 학습 섹션 첫 블록에 현재 Rescue epoch 8000 본학습 명령·상자 범위·2048환경·기본 종료 조건을 정리하고 중복 명령을 제거했다. Rescue 평가·VNC 예제는 사용 중인 random_scenario로 맞췄다.
- 문서 링크·명령 경로·Bash 예제 문법·diff 확인. 문서만 변경했으며 실행 중인 학습은 종료·재시작하지 않았다.

### Rescue 학습 상자 크기를 평가한 범위로 변경

- 사용자가 epoch 8000 baseline viewer에서 확인한 X/Y 0.40~0.60m·Z 0.30~0.50m를 Rescue Stage 2 기본 학습·평가 YAML에 적용했다. 기본 크기 0.4m에 scale 간격 0.125를 유지해 각 축 0.05m 간격·125개 조합을 독립 샘플링한다. 미사용 `testSizes`도 같은 범위로 맞추고 실행 가이드의 상자 설명을 갱신했다. 기본 범위와 중복되는 임시 평가 명령은 제거했다.
- 관련 CPU **4개 통과**, YAML의 실제 미터 단위 격자·2048환경·epoch 8000 경로·셸 문법·diff를 확인했다. 같은 범위의 시뮬레이션 검증은 직전 임시 평가 기록을 따른다. 본학습은 실행하지 않았다.

### Rescue 상자 크기 범위 임시 평가

- Rescue test/VNC에 평가 전용 `BOX_SIZE_RANGE=xy최소,xy최대,z최소,z최대`(미터)를 추가했다. X/Y 40~60cm·Z 30~50cm를 5cm 간격으로 시험할 수 있으며 기본 학습 YAML은 유지한다. Stage 1 baseline import JSON에 실제 생성된 상자 크기를 기록한다. 실행 명령은 `config_stage2.md`의 기존 평가·VNC 섹션에 추가했다.
- epoch 8000·학습 0회·random_scenario 16환경·32step headless 평가·JSON 저장 완료. 실제 상자 64개의 범위·5cm 격자와 기본 학습 범위 유지, 잘못된 범위·학습/resume 조합 거부 6건, 셸/Python 문법·문서 링크·diff를 확인했다. 자료는 `output/approach_stage2_rescue_klclimb50_box_range_check/`에 있다. GUI 행동·장기 성능은 확인하지 않았다.

### Rescue Stage 1 출발 checkpoint를 epoch 8000으로 변경

- Rescue Stage 2 train wrapper의 기본 pth와 `config_stage2.md`의 실험 표·학습·Stage 1 baseline test/VNC 명령을 `stage1/ApproachScenarioStage1RescueAtKLClimb50_00008000.pth`로 갱신했다.
- 파일 존재·셸 문법·문서 링크·현재 명령 경로·diff를 확인했다. 실제 epoch 8000의 weight·actor/AMP RMS 전이와 학습 0회·협력 입력 가중치 0을 확인하고, random_scenario 16환경·32step headless 평가·JSON 저장을 완료했다. 검증 자료는 `output/approach_stage2_rescue_klclimb50_epoch8000_check/`에 있다. 본학습은 실행하지 않았다.

### Stage 1 정책만으로 학습 전 Stage 2 평가

- Rescue test/VNC에 `STAGE1_ONLY=1`을 추가했다. Stage 1 pth·actor/AMP 관측 통계를 직접 전이하고 확장 action head의 협력 입력 가중치를 0으로 유지해 학습 없이 Stage 2 환경·성공 기준으로 평가한다. 결과는 기본 `output/approach_stage2_rescue_klclimb50/stage1_only/`에 분리하며 import JSON에 source epoch·Stage 2 학습 0회·협력 입력 가중치 0을 기록한다. 일반 Stage 2 pth 평가는 기존 경로를 사용한다.
- 관련 CPU **15개 통과**: 정규화 후 Stage 1 action/sigma 동등성·모델/RMS 불변·필수 통계 누락 거부를 확인했다. CLI의 학습·resume·checkpoint 미지정·Stage 1 환경 조합 거부, Python/셸 문법·문서 경로·diff를 확인했다. 실제 rescue epoch 7000으로 네 preset 각각 16환경·600step 학습 0회 headless 평가·finite 성공률 JSON 저장을 완료했다. 기존 Stage 2 pth의 16환경·32step 로딩·평가도 통과했다. 검증 자료는 `output/approach_stage2_rescue_klclimb50_stage1_only_check/`에 있다. GUI 화면은 직접 확인하지 않았다.

### Rescue KLClimb50 기반 4물체 Stage 2

- 지정된 `stage1/ApproachScenarioStage1RescueAtKLClimb50_00007000.pth`를 기본 전이하는 `approach_stage2_rescue_klclimb50` config·학습 설정·train/test/VNC·별도 output을 추가했다. 3→4물체에서 관측은 607→644, AMP는 무조건부 1290차원을 유지하며 source variant를 엄격히 검사한다. 협력 scene은 30/30/30%, 독립 scene은 10%이며 독립 과제는 source의 5/5/50/30/10%·상자·RSI를 계승한다. Stage 2 보상을 사용하고 teacher KL·rescue 전용 shaping은 이어가지 않는다.
- 관련 CPU **20개 통과**, 셸 문법·문서 경로·diff 확인. 실제 pth로 2048환경·`MAX_ITERATIONS=1` 학습·저장 완료: 169개 tensor 전이·8개 신규 초기화, actor encoder 75개 tensor·actor RMS 고정, 협력 head 갱신, scalar 177개 finite·물리 reset 실패 0. 전이 직후 action 최대 차이 2.69e-7·sigma 차이 0을 확인했다.
- 저장 checkpoint로 place_climb/place_sit/place_stack/independent 각각 16환경·32step headless 평가·JSON 저장을 완료했다. 자료는 `output/approach_stage2_rescue_klclimb50_4o_check/`에 있으며, 호환·실행 검증으로 장기 협력 성능은 확인하지 않았다. 실행법은 `markdowns/config_stage2.md`의 기존 섹션에만 추가했다.

### 실행 가이드 Stage 1/2 분리·문서 형식 통일

- `config.md`의 실험별 상세 섹션·중복 명령을 제거하고, Stage 1 명령을 학습·로컬 평가·서버 VNC 목록에 모았다. Stage 2 목록·checkpoint 규칙·명령은 `markdowns/config_stage2.md`로 분리했다. 두 문서와 `AGENTS.md`에 새 실험의 별도 섹션·장황한 설명 금지 및 Stage 2 전용 문서 작성 규칙을 명시했다.
- 문서 링크·명령 경로·실험별 train/test/VNC 누락·Bash 예제 문법·diff를 확인했다. 문서만 변경했으며 학습·시뮬레이션은 실행하지 않았다.

### 일반 Stage 1 unified 기반 Stage 2 전이

- 기존 owner-HOLDING Stage 2는 일반 unified의 5필드 관측과 호환되지 않아, `approach_stage2_unified` config·학습 설정·train/test/VNC·별도 output을 추가했다. 일반 unified의 2H/4O·644차원 관측, 토큰 순열·canonical binding·RSI·조건부 AMP·보상·성공·상자 설정을 계승하며 협력 30/30/30%·독립 10%를 유지한다. 새 전이는 일반 unified variant만 받고 actor encoder·관측 RMS를 고정한다.
- 관련 CPU **17개 통과**, 셸 문법·문서 링크·diff 확인. `/home/cvlab/Desktop/CVPR2027/stage1/output/`의 unified epoch 7500 pth로 2048환경·`MAX_ITERATIONS=1` 별도 학습·저장 완료: 169개 tensor 전이·8개 협력 tensor 초기화, scalar 177개 finite, 물리 reset 실패 0. Actor encoder 75개 tensor·actor RMS가 그대로이고 협력 head weight가 갱신됨을 확인했다. 실제 Stage 1 pth·RMS를 사용한 전이 직후 action 최대 차이는 2.09e-7, sigma 차이는 0이었다.
- 저장된 Stage 2 pth로 place_climb/place_sit/place_stack/independent 각각 16환경·32step headless 평가와 JSON 저장을 완료했다. 검증 자료는 `output/approach_stage2_unified_check/`에 있다. 이는 호환·실행 경로 검증이며 협력 성능·장기 수렴 결과는 아니다.

## 2026-09-30

### Stage 1 unified owner HOLDING 기반 Stage 2 협력 실험

- Stage 1의 2H/4O canonical binding, 토큰 순열, owner-HOLDING edge packet, 과제 조건 AMP, RSI, 상자·goal·거리 설정, 보상·포화를 계승한 별도 Stage 2 config·학습 설정·train/test/VNC를 추가했다. 협력 place_climb/place_sit/place_stack은 전체 scene의 30/30/30%, 독립 scene 10%의 agent별 5/5/20/35/35 샘플링을 적용한다. Stage 1 actor encoder를 고정 전이하고 owner-state packet을 grounded 협력 attention에 연결한다. 기존 Stage 2 실험과 현재 Stage 1 학습은 변경하지 않았다.
- 관련 CPU 테스트 **18개 통과**. 2048환경·`MAX_ITERATIONS=1` 별도 output 학습에서 Stage 1 epoch 2500 checkpoint의 177개 tensor 전이·8개 협력 tensor 신규 초기화, scalar 177개 finite, 물리 reset 실패 0을 확인했다. 저장 checkpoint로 1환경·32step `place_stack` headless 평가를 완료했다. 이는 실행 경로 검증이며 학습 성과는 아직 확인하지 않았다.

### AT goal marker 담당자 색 표시

- viewer에서 빨간색으로 통일됐던 AT goal marker를 실제 graph edge owner의 에이전트 색으로, 미할당 goal은 회색으로 표시한다. 물체 색과 같은 reset 시점에 갱신하며 headless 무영상 학습에는 색 API 작업을 추가하지 않는다. Stage 2 협력·독립 graph에 대해 marker 호출을 확인했다. 현재 열린 viewer에는 재시작 후 적용된다.


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
