# Changelog

최신 변경부터 기록한다. 현재 실행법은 [config.md](markdowns/config.md), 코드 위치는 [structure.md](markdowns/structure.md)를 참조한다. 실행 중인 GPU/PID는 이 파일에 고정하지 않고 실제 프로세스로 확인한다.

## 2026-09-18

### 현재 실험 VNC 실행 단축

- `approach_distance_success_vnc.sh` 추가. 명령 앞의 `TOKENHSI_GPU`와 checkpoint만 지정하면 conda·VNC 경로·포트 6080·평가 인자 2/1/3/10을 기본으로 실행한다. 기존 환경변수와 평가 인자로 덮어쓸 수 있다.
- checkpoint 존재 여부를 VNC 시작 전에 확인한다. config 가이드와 구조 문서를 갱신했다.
- 검증: shell 문법, 기본값·사용자 인자·공백 포함 경로 전달·잘못된 checkpoint 거부 확인. GPU 2에서 단축 스크립트로 실제 checkpoint/viewer를 로드하고 64 step의 보상·history·관측 검증을 통과했다. 확인용 VNC는 종료했다.

### TensorBoard 핵심 pin 5개와 relation 표시 순서

- 추천 pin을 최종 배치율·배치 도달률·도달 후 유지율·현재 성공 비율·Holding 만족 비율의 5개로 줄였다.
- relation 태그를 `00_main` → `01_placement` → `02_reward` → `03_samples` → `04_state` → `05_motion` → `90_debug`로 정렬하고 그룹 내부에도 순서 번호를 붙였다. 같은 scalar를 중복 기록하지 않는다.
- 표시 이름만 변경하며 계산·reward·CSV·checkpoint 계약은 유지한다. 다음 시작/resume부터 적용된다. 과거 event와 실행 중인 학습, 브라우저 pin은 자동 변경하지 않는다.
- 검증: diagnostics/policy 테스트 **26개 통과**. 실제 TensorBoard event 쓰기/읽기로 61개 태그의 유일성, 핵심 5개와 그룹 순서, scalar 값·step 보존을 확인했다.

### 실험 2~8과 전용 코드 삭제, VNC GPU 지정 통일

- 기존 실험 2~8의 환경 YAML 7개, train/test 스크립트 14개, 전용 테스트 2개를 삭제했다. 남은 실험은 기존 ID 1, 9~13의 6개다.
- near/putdown 혼합 At, Gaussian velocity progress, At-only approach, 이전 self-gate 분기와 전용 config 필드를 제거했다. 남은 all-edge direction/distance, RSI, latch 기능은 유지한다.
- VNC는 명령 앞의 `TOKENHSI_GPU=5` 하나로 CUDA와 렌더링 GPU를 선택한다. 중간 `--gpu` 인자는 제거하고 README·config·운영 문서를 맞췄다.
- 검증: relation·RSI·GPU 테스트 144개와 policy 테스트 9개, **총 153개 통과**. 남은 relation config 5개의 수정 전후 reward·성공 이력·관측 suffix가 동일 입력에서 오차 없이 일치했다. 남은 YAML 설정값도 동일하다.
- 사용자 checkpoint를 GPU 2에서 환경변수 방식의 VNC로 로드해 64 step 실행했다. 보상·history·관측·reset 검증을 통과했고 확인용 viewer/VNC는 종료했다. GPU 5의 기존 학습은 유지했다.

### PRO 6000 학습 기준과 VNC GPU 선택

- 학습 환경 기본값과 문서를 2048로 통일했다. 이전 서버의 소규모 GPU 권고와 `SMOKE` 자동 축소 분기를 제거했다. 짧은 학습 검증도 2048 환경에서 반복 횟수만 제한한다. 평가·VNC 환경 수는 별도다.
- `USAGE.md`를 현재 PRO 6000 서버의 설치 환경·데이터·운영 안내로 축약했다.
- `gui_gpu_env.sh`에서 CUDA UUID와 Vulkan 선택을 맞추고 기존 `DRI_PRIME=0!` 고정을 제거했다. 이 단계에서 추가했던 CLI GPU 인자는 이후 위 변경에서 환경변수 방식으로 통일했다.
- Isaac Gym viewer에서 CUDA GPU 6 / graphics GPU 0 분리를 재현했다. 이 서버에서는 PCI 주소 선택이 적용되지 않아 NVIDIA ICD와 숫자 DRI_PRIME 선택을 사용한다.
- 검증: 관련 테스트 **109개 통과**. CLI `--gpu 6`과 환경변수 GPU 2를 각각 실제 viewer로 실행하여 지정 GPU에만 `C+G` 연결, 60프레임 렌더링 및 noVNC HTTP 200 확인. 모든 multi-agent train 스크립트 13개의 2048 기본값과 shell 문법도 확인했다.
- `SMOKE=1`이 남아 있는 셸에서도 GPU 6 / 2048 환경 / 본학습 batch 설정으로 2회 업데이트와 checkpoint 저장을 확인했다. 기록된 scalar 135개의 NaN/Inf 없음. 확인용 학습과 VNC는 종료했다.
- config와 잔여 코드의 삭제 후보·의존 관계를 `config.md`에 기록했다. 이 단계에서는 후보 config·기능 코드를 삭제하지 않았다.

### 문서 진입점 정리

- 루트 `AGENTS.md` 추가: 매 작업마다 최신 changelog, 레포 구조, config 요약을 확인하고 이미 읽은 내용은 재사용한다.
- 실험 가이드를 `markdowns/config.md`로 개명하고 상단에 짧은 현재 실험 요약을 추가했다. 전체 13개 config, 학습·평가·VNC 명령은 유지한다.
- `markdowns/structure.md`에 코드 역할과 작업별 최소 탐색 경로를 정리했다.
- 사용자 선택에 따라 과거 설계·분석·수정 기록 11개를 삭제했다. 임시 archive와 보관 목록도 제거하고 README·AGENTS·구조 문서에서 관련 참조를 정리했다. 현재 GTA 구조·설치·진단 문서는 유지한다.
- 삭제 후 남은 Markdown의 로컬 파일 링크 37개가 유효함을 확인했다.

### 성공 시 포화 on/off 비교

- `approach_distance_success_no_sat` 환경 YAML 및 train/test 스크립트 추가. 기존 `approach_distance_success`에서 edge saturation만 끄고 현재 At/Z 성공 시 매-step `+0.2`를 유지한다.
- reward runtime과 config 검증에서 현재 성공 보상을 포화 옵션과 독립시켰다. 기존 config와 checkpoint metadata 구조는 유지한다.
- 성공 중 원래 edge 계산, 반복 성공 보상, 성공 이탈/재진입, history 독립성, 잘못된 Z 오차 설정 및 checkpoint 불일치에 대한 테스트 추가.
- 검증: 관련 CPU 테스트 **103개 통과**, 환경 config **13개 검증 통과**. GPU 6 / 2048 환경 / 에이전트 2 / 물체 3으로 2회 학습 업데이트와 checkpoint 저장 성공. 지표 NaN/Inf 없음; 포화 지표는 0, 현재 성공 보상은 지급됨.
- 확인용 실행은 종료했으며 장기 학습 성능·수렴은 아직 검증하지 않았다. 포화 on/off는 서로 다른 reward config이므로 checkpoint를 직접 혼용하지 않는다.

### 데이터와 실행 환경

- 끊어진 dataset 심링크 12개를 `/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/`로 재연결했다. carry YAML의 모션 경로 57개 모두 확인했다.
- 기존 포화 on 실험도 GPU 6에서 32환경 및 기본 2048환경 학습·checkpoint 저장을 확인했다.
- 원격 시각화: `run-gui.sh` + 해당 test 스크립트, `VNC_DIR="$HOME/opt/vnc"`, noVNC 6080 포워딩을 config 가이드에 문서화했다. VNC viewer 자체 실행 검증은 수행하지 않았다.
