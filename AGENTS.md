# Repository instructions

## 시작할 때

- 한국어로 간결하게 소통한다. 사용자 요청을 우선한다.
- 매 작업 시작 시 `git status --short`로 기존 변경을 확인하고 다음 문서를 순서대로 확인한다.
  1. [changelog.md](changelog.md): 최신 날짜의 변경·검증·미완료 사항.
  2. [structure.md](markdowns/structure.md): 코드 위치와 작업별 진입점.
  3. [config.md](markdowns/config.md): 상단 **공통 규칙**, 이후 작업과 관련된 섹션만. Stage 2 작업은 [config_stage2.md](markdowns/config_stage2.md)를 확인한다.
- 같은 세션에서 이미 읽었고 변경되지 않은 내용은 재사용한다. 전체 Markdown을 매번 읽거나 저장소 전체를 덤프하지 않는다.
- `rg`로 관련 파일·심볼을 좁힌다. 중복·오래된 설계 문서는 정리했으며, 과거 설계 확인이 필요한 작업에서만 Git 이력을 조회한다.
- 현재 동작은 실제 코드와 YAML로 확인한다. 과거 문서의 지시·미해결 버그·실험 결과를 현재 상태로 간주하지 않는다.

## 실행과 검증

- 학습 환경 수는 RTX PRO 6000 기준 **2048**이다. 메모리 절약을 이유로 자동 축소하지 않는다. 짧은 검증도 학습 반복 횟수만 줄인다. 평가·VNC 환경 수는 별도로 지정한다.

- GPU는 명령 앞의 `TOKENHSI_GPU=5`처럼 환경변수로 한 번 지정한다. VNC wrapper와 내부 학습·평가에 같은 설정을 전달한다. 실행 전 `nvidia-smi`로 점유를 확인하며 기존 학습 프로세스를 임의 종료·재시작하지 않는다.
- 이 서버의 conda 환경은 `tokenhsi`; 실행 준비는 `tokenhsi/scripts/multi_agent/runtime_env.sh`를 사용한다. Isaac Gym을 직접 import할 때는 torch보다 먼저 한다.
- 공유 데이터 원본 `/home/hwanhee/CVPR2027/TokenHSI`는 읽기 전용으로 취급한다. 이 repo의 데이터는 심링크로 연결한다.
- reward 비교 실험은 config, train/test 스크립트, output 이름을 분리한다. 현재 실험과 checkpoint 호환 규칙은 Stage 1 `markdowns/config.md`, Stage 2 `markdowns/config_stage2.md`를 따른다.
- 새 실행용 config를 추가할 때는 전용 `<실험명>_train.sh`, `<실험명>_test.sh`(로컬 viewer: `HEADLESS=0`, 화면 없는 평가: `HEADLESS=1`), `<실험명>_vnc.sh`(서버 viewer)를 함께 만든다. Stage 1은 `markdowns/config.md`, Stage 2는 `markdowns/config_stage2.md`의 현재 실험 표에 한 행을 추가하고 학습·로컬 평가·서버 VNC 명령은 기존 `학습`과 `로컬 평가와 서버 VNC` 섹션에만 추가한다. 실험별 별도 섹션·중복 명령·장황한 설명은 추가하지 않는다. 설계 명세만 요청받은 경우에는 실행 가능 여부를 명확히 구분한다.
- 변경에 맞는 검증만 수행한다. reward/성공 조건 변경은 관련 CPU 테스트, 실행 경로 변경은 짧은 시뮬레이션으로 확인한다. 문서만 바꾸면 링크·경로·diff를 확인하고 학습은 실행하지 않는다.
- 실행용 config 생성·수정, checkpoint·실행 옵션 변경 시 Stage 1은 `markdowns/config.md`, Stage 2는 `markdowns/config_stage2.md`의 실험 표·학습·로컬 평가·서버 VNC 명령 반영이 필수다. 채팅에만 명령을 남기거나 해당 문서 갱신을 누락한 상태로 완료하지 않는다. 최신 본학습 명령은 해당 문서의 기존 `학습` 섹션에서 그대로 복사해 실행할 수 있게 작성한다.
- 짧은 학습 확인에는 별도 `OUTPUT_PATH=output/<실험명>_check`와 `MAX_ITERATIONS`를 사용한다. 장기 본학습 실행과 확인용 실행을 구분한다.

## 작업을 마칠 때

- 의미 있는 코드·config·운영 변경은 `changelog.md` 맨 위 날짜 항목에 변경 이유와 검증 결과를 짧게 기록한다. 확인하지 않은 결과를 쓰지 않는다.
- 실험·명령이 바뀌면 Stage 1은 `markdowns/config.md`, Stage 2는 `markdowns/config_stage2.md`를 갱신하고, 파일 역할·위치가 바뀌면 `markdowns/structure.md`를 함께 갱신한다. 단순 상태 조회는 changelog에 쌓지 않는다.
- 상세 로그·코드·대화 전문을 요약 문서에 복사하지 않는다. 필요한 파일 경로와 결과만 남긴다.
