# Changelog

현재 실행법은 [config.md](markdowns/config.md), 코드 위치는 [structure.md](markdowns/structure.md)를 따른다.

## 2026-10-16

### 2명 네 설정에도 actor 특징 캐시 활성화

- `stage2`, `stage2_grasp`, `stage2_at20`, `stage2_grasp_at20` train YAML에 `cache_frozen_actor: true`를 추가했다. 이제 여섯 설정 모두 활성화한다. 2명 actor encoder 동결은 기존부터 적용돼 있었으며 이번 변경은 계산 결과 재사용만 켠다. 미사용 `next_obses` 제거는 이미 모든 MA 설정에 적용돼 있다.
- CPU45개와 여섯 YAML 활성화 확인 통과. GPU6/MPS에서 대표 기본형 `stage2`와 Grasp형 `stage2_grasp_at20`을 각각2048환경·epoch2까지 학습·저장했다. 캐시/재계산 최대차이2.39e-6 이하, 미사용 버퍼 부재, 저장본 strict 재로드·actor encoder/scene RMS 동결·모델 전체 finite·scalar133종 finite를 확인했다. 2명 전후 속도·전체 VRAM 절감량은 별도 측정하지 않았다.
- 실행 중인 학습과 다른 저장소는 수정하지 않았다. 새 실행/재개부터 적용하며 명령과 checkpoint 형식은 유지한다. 검증 기록은 `output/two_agent_cache_check/validation.json`에 있다.

### 4명 학습의 동결 encoder 캐시·미사용 버퍼 제거

- 4명 두 train YAML에 `cache_frozen_actor: true`를 추가했다. 정규화한 scene별 동결 actor 특징을 rollout마다 계산하고 PPO 6회 업데이트에서 같은 minibatch 인덱스로 재사용한다. SA/head·critic 학습, human 셔플, RSI·AMP·보상·2048환경은 유지한다. 기존2명 설정은 캐시 기본 비활성이다.
- MA의 미사용 `next_obses` 저장·flatten 복사를 제거하고 `next_values` bootstrap은 유지했다. checkpoint tensor/optimizer 형식은 그대로이며 기존 저장본으로 재개한다. 실행 중인 본학습은 중단·재시작하지 않았다.
- CPU45개 통과: 2/4명·GraspView별 순열 변경 및 6회 업데이트의 출력·gradient 비교. 실제 Grasp 관측64개와 저장 모델의 GPU 비교에서 행동 최대차이7.46e-7, gradient 상대 L2차이6.86e-6을 확인했다. 기존 계산끼리 순열을 바꾼 대조군은3.86e-6이었다. 캐시는 FP32 허용오차 내 동등하며 전체 학습 궤적의 비트 단위 일치를 의미하지 않는다.
- GPU6·MPS·2048환경·같은 seed의 Grasp 변경 전/후 각4 epoch 비교. warmup 제외 epoch2~4 평균 업데이트27.33→23.70초(13.3% 감소), 전체38.40→35.12초(8.5% 감소), PyTorch 최대 allocated29.50→27.09GiB. 공유 GPU의 짧은 측정이며 전체 VRAM 또는 장기 성능 수치와 구분한다.
- 최적화 전 epoch4 checkpoint를 새 코드로 epoch5까지 재개·저장했다. Grasp 저장본 strict 재로드, actor encoder69개·scene RMS 불변, critic/SA/head 갱신 및 scalar194종 finite 확인. 검증 산출물은 `output/optimization_check/`에 분리했다.
- 기본4명 설정도 GPU6·2048환경에서 epoch2 학습·저장을 완료했다. 두 설정의 실제 캐시와 재계산 특징 비교, 기본/재개 Grasp 저장본 strict 재로드 및 동결·갱신 검사 통과. 기존 본학습은 checkpoint 저장 전에 사용자가 직접 중단했다고 확인했다.

### 4명 랜덤 그룹 AT20 정규 경로

- `stage2_at20_4agent`, `stage2_grasp_at20_4agent`의 env/train YAML·전용 train/test/VNC를 추가했다. 기존4개 2명 설정은 유지한다. 새 두 설정은 GraspView만 다르며 같은 Stage1 PTH로 시작한다.
- episode별 순차 pairing p=0.8·그룹별 AT/ON_TOP50:50을 적용했다. 단독용8개/공동용4개 물리 상자를 생성하고 활성 물체를8개 논리 슬롯에 매핑한다. 비활성 상자는 플레이 영역 밖 서로 떨어진 지면에 배치하며 관측·GTA attention에서 제외한다. 실제 NONE 사람은 mask하지 않는다.
- 4명 전용 reset에서 공동팀별 독립 RSI frame·무작위 원본 역할, 단독 size RSI 캐시, 사람별 AMP history를 연결했다. 원본/후진/옆걸음 AMP80/10/10과 공동100/129/135프레임을 유지했다. 캐시 생성기는4명 단독 후보 크기까지 검사한다.
- 팀별 반대 앵커 gate와 pair별 CPA release를 적용했다. GraspView는 실제 actor ID로 소유 payload를 찾아 episode 고정 앵커와 rigid-point 속도를 계산한다. actor/critic/player/RMS에4개 view를 연결하고 기존2명 PTH tensor 구조는 유지했다.
- `envSpacing=10`, 그룹별 무작위 yaw/중심, 사람·상자·목표 clearance를 적용했다. 공동 loco 받침2개는 높이를 먼저 적용한 뒤 배치 검사·강체변환한다. 사용하지 않는 상자는 겹치지 않게 격리한다. 부분 reset에서 다른 환경의 상태/graph/AMP 배정을 유지한다.
- 그룹/사람 분모를 나눈 solo/joint×AT/ON_TOP 성공·gate·초기 RSI holding·loco pickup·reset skill 지표를 추가했다. 4명 sampler/물체 수/AT20/GraspView/AMP/RSI checkpoint 계약을 구분한다.
- CPU41개 통과. 기존2명 geometry·보상·reset·CPA 결과와 변경 전 HEAD가 정확히 일치했다. Stage1 전이 두 모델 및 기존 SA epoch8000 PTH의 행동·가치·sigma도 변경 전 HEAD와 최대오차0이었다. 기존2명 SA PTH의16환경·32-step 평가도 정상 종료했다.
- GPU5/6에서 각각2048환경·epoch2 짧은 학습과 저장 완료. 캐시10,212프로필 재사용, actor encoder69개와 scene RMS 불변, critic encoder69개·action/value head각6개·actor/critic SA bias각4개 갱신 및 저장 모델 전체 finite를 확인했다. 이는 기능 검증이며 장기 학습 성능 평가는 아니다.
- GPU6 실제 물리 검사26개 조합: 2pair/1pair+2solo/4solo×AT/ON_TOP 및 mixed, 허용 RSI별16-step·부분 reset·원본 역할/DOF/속도 매핑·공동 AMP history·받침2개·랜덤128-step·비활성 상자 격리 통과. source clip/frame 선택을 배치 재시도로 바꾸지 않는다.
- 두4명 저장본의 headless 평가와 두 전용 VNC 경로의 실제 viewer 평가가 정상 종료했다. VNC는 GPU5에서 기본 모델1환경32-step, GraspView1환경64-step을 실행했다. 로컬 데스크톱 화면은 별도로 확인하지 않았다.
- 누락된 `gui_gpu_env.sh`를 복구해 VNC wrapper의 CUDA/Vulkan GPU 전달을 연결했다. 실행 명령과 파일 역할은 `markdowns/config.md`, `markdowns/structure.md`에 갱신했다. 최종 검증은 GPU5/6만 사용하며 본학습은 시작하지 않았다.
- 검증 산출물은 `output/four_agent_check/`: `release/`, `cpu.log`, `checkpoints.json`, `physics_release/physics.json`, 학습/평가/VNC 로그. 다른 저장소는 읽기 전용으로 참조했다.

### 짧은 실험명과 AT20 비교 설정

- 기존 두 설정의 env/train YAML·train/test/VNC·실험명·기본 output을 `stage2`, `stage2_grasp`으로 줄였다. 기존 성공조건과 PTH 계약은 유지한다.
- `stage2_at20`, `stage2_grasp_at20`과 각각의 전용 실행 파일을 추가했다. 공동·단독 AT를 박스 중심↔목표점 3차원 거리≤0.2m AND 높이 차이 절댓값≤0.001m로 판정한다. HOLDING gate·다른 관계·연속 state/progress 보상은 유지한다.
- AT20 성공반경을 checkpoint reward 계약에 포함해 기존 조건과의 재개·평가 혼용을 거부한다. AT20은 Stage1 전이로 시작한다.
- 단독 RSI skill 선택에서 누락된 `else`를 복구했다. 학습은 template/cache로 결정한 skill을 유지하고, 평가는 지정한 skill 분포에서 선택한다. 두 경로의 CPU 회귀 테스트를 추가했다.
- CPU26개 통과: 거리·높이 경계, 3차원 거리, 공동/단독 gate, reset/step, 기존 보상 유지와 checkpoint 계약 분리 확인. 기존 설정의 원본 geometry·보상 결과는 정확히 일치하며 기존 PTH forward 최대오차2.39e-7/7.16e-7(행동/가치), sigma0을 유지했다.
- RSI 분기 수정 후 GPU3·2048환경에서 AT20 두 설정 모두 epoch3 학습·저장 완료, 기존 단독 RSI 캐시 재사용 확인. 두 저장본의 20cm 계약·전체 모델 텐서 finite와16환경·32-step 공동 AT/단독 AT headless 평가 정상 종료 확인. Python3.8·셸 문법, 네 YAML 쌍·12개 실행 경로·문서 링크 검사 통과.
- 이름을 줄인 기존 두 설정도 이전 SA/GraspView PTH로16환경·32-step headless 평가 정상 종료했다. 실제 VNC 화면은 확인하지 않았다.
- 검증 산출물: `output/at20_check/validation.json`과 학습·평가 로그. 학습·평가·VNC 명령은 `markdowns/config.md`에 기재했다. 쓰기 작업은 이 저장소 안에서만 수행했다.

### 두 SA 정규 학습 경로로 정리

- Human SA4와 SA4 GraspView의 env/train YAML 각2개·train/test/VNC6개를 남겼다. 과거 실험 실행·설정·토이·분석·문서와 CA/geometry/NONE/BEFORE/teacher 분기를 제거했다.
- Stage1 task embedding/GTA encoder·64입력 head 및 checkpoint 파라미터 이름/크기는 유지했다. 정규 계약 `jointCarryTraining`/`joint_carry_training`으로 저장하고 기존 SA checkpoint 계약은 읽기 호환한다.
- 데이터 준비와 해시 검증, 단독 size RSI 캐시, paired FullPickup RSI reset, AMP80/10/10 reference를 두 모델의 공통 경로로 연결했다. 캐시 mask를 생성 시 적용하고 임시 subclass를 제거했다. 현재 입력18파일만 manifest에 남겼다.
- CPU14개 통과. 원본 Stage1 CarryDistill30000 전이 두 모델 및 SA4 epoch8000 PTH의 모델/RMS 재로드 비교: 행동 최대오차2.39e-7, 가치7.16e-7, sigma0; 파라미터 이름·크기 일치. 원본 대비 AT/ON_TOP/단독 geometry·HOLDING gate·보상·reset·CPA release 결과는 정확히 일치했다.
- GPU3·2048환경에서 두 모델 모두 epoch3 학습·저장을 완료했다. 단독 RSI 캐시2266프로필을 생성한 뒤 두 모델에서 같은 캐시 재사용을 확인했다. Actor encoder69개/scene RMS 불변, Critic69개·action/value head각6개·SA bias8개 block 갱신 및 전체 모델 텐서 finite 확인.
- 두 저장본 strict 재로드와 정규화 후 Actor/Value forward finite·TensorBoard133종 scalar 전체 finite 확인. 기존 SA epoch8000 PTH와 새 GraspView 저장본은 전용 test로16환경·32-step headless 평가 정상 종료했다. Python3.8/셸 문법·문서 링크·diff 검사 통과. 실제 VNC 화면은 확인하지 않았다.
- 검증 산출물: `output/cleanup_check/{saved_check.json,cache_check.json,model_reference.json,environment_parity.log,cpu.log}`와 전용 학습/평가 로그. 다른 저장소의 추적 파일 변경은 없다.
