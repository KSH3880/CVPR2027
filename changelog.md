# Changelog

현재 실행법은 [config.md](markdowns/config.md), 코드 위치는 [structure.md](markdowns/structure.md)를 따른다.

## 2026-10-16

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
