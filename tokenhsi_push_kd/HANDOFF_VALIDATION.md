# 이관 검증 — 2026-11-01

- GitHub 브랜치 HEAD와 로컬 원본 커밋 `0d6d488639429a97611c6bb379603a315e2d2867` 일치. 원본 checkout은 clean이며 수정하지 않았다.
- CPU 테스트 `12 passed`(PyTorch 기존 경고 2건). 실제 checkpoint의 기존 정책 보존·loss routing 검사를 포함한다.
- 원본 preflight 통과: runtime manifest 130개 항목, PUSH 21개·보행 4개, RSI 949후보·167 profile, PUSH asset/mesh SHA256 일치.
- 세 runtime manifest의 motion/object 참조 458건 모두 존재하고 이관 폴더 내부로 연결됨을 확인했다.
- teacher SHA256이 원본 SOURCE.json과 일치. teacher와 초기 student SHA256은 HANDOFF_SOURCE.json에 기록했다.
- 외부 심링크 없음. SHA256 목록 2067개 파일 생성. 별도 위치로 복사 후 prepare_paths 재실행·preflight·무결성 검사 `BUNDLE_OK: 2067 files` 통과.
- train/eval shell 구문 검사 통과. 짧은 실행의 `OUTPUT_PATH=output/invalid`가 GPU 실행 전에 거부되는 것을 확인했다.
- 검증 임시 복사와 출력 경로 검사 로그는 상위 저장소 `output_etc/check/tokenhsi_push_handoff/`에 두고 전달본에서는 제외했다.

GPU 시뮬레이션·PPO 통합·장기 본학습·성공률은 이관 작업에서 실행하거나 검증하지 않았다. Isaac Gym SDK와 conda 환경 설치 자체도 새 머신에서 검증하지 않았다. CPU 물리 연결 검증의 과거 이력은 원본 CHANGELOG.md를 참고한다.

## 실험 요청 후속 갱신

목표·distillation 탐색 범위·논문용 검증/제출 조건과 MESSAGE.md를 추가했다. 5%는 각 과제의 teacher 대비 상대 하락률로 명시하고 실험 전 평가 정의를 확정하도록 했다. 문서 링크와 SHA256 목록을 갱신했으며 코드/학습 설정은 변경하지 않았다.

## 평가 요구 갱신

기존 세 과제는 TokenHSI 원본 eval, PUSH는 우리 목표 영역 final85% 초과·ever 발표용 보조 지표로 정했다. 별도 연속 유지 조건은 없다. EVALUATION.md에 baseline native eval과 목표 영역 eval의 차이 및 미구현 정렬 작업을 명시했다. 문서만 변경했으며 학습/평가 실행은 하지 않았다. 링크 검사와 무결성 목록을 갱신했다.
