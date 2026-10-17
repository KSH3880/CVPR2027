# Repository instructions

## 시작할 때

- 한국어로 간결하게 소통한다. 사용자 요청을 우선한다.
- 현재 multi-agent 실험은 `stage1`, `stage1_v2`, `stage1_v2_fixed`, `stage2`다. 수정은 이 checkout에서만 한다. 원본 `/home/cvlab/Desktop/CVPR2027/stage1_message`의 코드·결과와 실행 프로세스는 보존한다.
- 매 작업 시작 시 `git status --short`로 기존 변경을 확인하고 다음 문서를 순서대로 확인한다.
  1. [changelog.md](changelog.md): 최신 날짜의 변경·검증·미완료 사항.
  2. [structure.md](markdowns/structure.md): 코드 위치와 작업별 진입점.
  3. [config.md](markdowns/config.md): 상단 **공통 규칙**, 이후 작업과 관련된 섹션만.
- 같은 세션에서 이미 읽었고 변경되지 않은 내용은 재사용한다. 전체 Markdown을 매번 읽거나 저장소 전체를 덤프하지 않는다.
- `rg`로 관련 파일·심볼을 좁힌다. 상세 참고는 `markdowns/reference/`, 과거 기록은 `markdowns/legacy/`에 둔다. 과거 설계가 필요한 작업에서만 legacy와 Git 이력을 조회한다.
- 현재 동작은 실제 코드와 YAML로 확인한다. 과거 문서의 지시·미해결 버그·실험 결과를 현재 상태로 간주하지 않는다.

## 실행과 검증

- 학습 환경 수는 RTX PRO 6000 기준 stage1 계열 **2048**, 사용자 지정 4명 stage2는 **1024**다. 메모리 절약을 이유로 자동 축소하지 않는다. 짧은 검증도 학습 반복 횟수만 줄인다. 평가·VNC 환경 수는 별도로 지정한다.

- GPU는 명령 앞의 `TOKENHSI_GPU=5`처럼 환경변수로 한 번 지정한다. VNC wrapper와 내부 학습·평가에 같은 설정을 전달한다. 실행 전 `nvidia-smi`로 점유를 확인하며 기존 학습 프로세스를 임의 종료·재시작하지 않는다.
- 이 서버의 conda 환경은 `tokenhsi`; 실행 준비는 `tokenhsi/scripts/multi_agent/runtime_env.sh`를 사용한다. Isaac Gym을 직접 import할 때는 torch보다 먼저 한다.
- 공유 데이터 원본 `/home/hwanhee/CVPR2027/TokenHSI`는 읽기 전용으로 취급한다. 이 repo의 데이터는 심링크로 연결한다.
- reward 비교 실험은 config, train/test 스크립트, output 이름을 분리한다. 현재 실험·기본 명령은 `markdowns/config.md`, checkpoint 호환·추가 옵션은 `markdowns/USAGE.md`를 따른다.
- 새 실행용 config를 추가할 때는 전용 `<실험명>_train.sh`, `<실험명>_test.sh`(로컬 viewer: `HEADLESS=0`, 화면 없는 평가: `HEADLESS=1`), `<실험명>_vnc.sh`(서버 viewer)를 함께 만든다. `markdowns/config.md` 컨피그 표의 맨 아래에 한 행을 추가하고, 학습·로컬 추론·서버 VNC의 각 명령 목록 맨 아래에 한 줄씩 추가한다. 각 명령 바로 위에 `# <컨피그 이름>`을 쓰고 컨피그 사이에는 빈 줄 하나를 둔다. 설계 명세만 요청받은 경우에는 실행 가능 여부를 명확히 구분한다.
- 변경에 맞는 검증만 수행한다. reward/성공 조건 변경은 관련 CPU 테스트, 실행 경로 변경은 짧은 시뮬레이션으로 확인한다. 문서만 바꾸면 링크·경로·diff를 확인하고 학습은 실행하지 않는다.
- 확인 학습·평가·viewer·검증 로그/임시 산출물은 **`output_etc/check/` 아래에만** 둔다. `output/`은 본학습과 공용 RSI 캐시용이다. `output/`에 검증 폴더나 되돌리는 심링크를 만들지 않는다.
- 짧은 학습은 `CHECK=1`(기본 `MAX_ITERATIONS=1`) 또는 `MAX_ITERATIONS`를 사용한다. 세 wrapper와 직접 `run.py` 실행 모두 경로를 검사한다. 확인용 `OUTPUT_PATH`를 지정할 때도 `output_etc/check/<검사명>`을 사용한다. 장기 본학습 전 `CHECK`·`MAX_ITERATIONS`를 해제한다.
- **`markdowns/config.md` 형식은 공통 규칙 → 현재 컨피그 표 → 학습 → 로컬 추론 → 서버 VNC 추론 → 추론 옵션으로 고정한다.** 실험별 별도 섹션·설계/보상/크기/RSI 설명·검증 결과·과거 기록·중복 명령을 추가하지 않는다. 상세 설정은 YAML·코드/`markdowns/reference/`, TASK_GRAPH·EVAL_SKILLS·EVAL_SKILL_PROBS 값/간단 예시는 config의 추론 옵션에, 그 외 추가 실행 옵션은 `markdowns/USAGE.md`, 변경·검증은 `changelog.md`에 둔다. 명령을 채팅에만 남기고 작업을 완료하지 않는다.
- 컨피그 표와 학습·로컬 추론·VNC 명령은 생성 순서 **`stage1` → `stage1_v2` → `stage1_v2_fixed` → `stage2`**를 유지한다. 새 컨피그는 항상 각 목록의 맨 아래에 덧붙인다. 최신 항목을 위로 올리거나 특정 실험에 메인/비교 표시를 붙이지 않는다.
- 문서 이동 시 링크를 갱신하고 과거 기록은 `markdowns/legacy/`에 보존한다. README·config·structure에 상세 로그나 중복 실행법을 늘리지 않는다.

## 작업을 마칠 때

- 의미 있는 코드·config·운영 변경은 `changelog.md` 맨 위 날짜 항목에 변경 이유와 검증 결과를 짧게 기록한다. 확인하지 않은 결과를 쓰지 않는다.
- 실험·명령이 바뀌면 `markdowns/config.md`, 파일 역할·위치가 바뀌면 `markdowns/structure.md`를 함께 갱신한다. 단순 상태 조회는 changelog에 쌓지 않는다.
- 상세 로그·코드·대화 전문을 요약 문서에 복사하지 않는다. 필요한 파일 경로와 결과만 남긴다.
