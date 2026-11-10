# SCT + LLM Scheduler

- 상위 `docs/OPERATIONS.md`를 따른다. 이 프로젝트는 독립 baseline이다.
- `/home/hwanhee/ksh/`는 읽기 전용이다. 그 아래 코드·캐시·출력도 쓰지 않는다.
- GPU 작업은 물리 GPU 6 또는 7만 허용한다. CPU 검증을 우선한다.
- MPS 서버 제어, GPU reset, 프로세스 signal 종료를 수행하지 않는다.
- 초기 1회 LLM 호출, 초기 freeze → 영구 release만 허용한다.
- 합성 테스트 데이터를 실제 demonstration으로 사용하지 않는다.
