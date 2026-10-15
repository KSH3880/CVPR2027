# Changelog

최신 변경부터 기록한다. 현재 실행법은 [config](markdowns/config.md), 코드 위치는 [structure](markdowns/structure.md)를 따른다. 과거 변경·검증 전체는 [보관 changelog](markdowns/legacy/changelog_before_cleanup.md)에 있다.

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
