# CVPR2027 — Codex 진입점


## 작업별 읽기 순서

- 전체 연구 맥락: `METHOD_V2.md` → `REPOS.md` → `plan/PLAN.md`
- steer: `plan/PLAN_steer.md` → 자동 결과 → `experiments/EXPERIMENTS_steer.md` → `TokenHSI-steer/CHANGELOG.md`
- ma: `plan/PLAN_ma.md` → 자동 결과 → `experiments/EXPERIMENTS_ma.md` → `TokenHSI-ma/CHANGELOG.md`
- 실험 실행·평가: 위 문서에 더해 `docs/PITFALLS.md`를 먼저 읽는다.

## 즉시 지킬 것

- `TokenHSI/` baseline과 외부 소스·빌드 트리는 수정하거나 옮기지 않는다.
- 실행 중인 학습·평가·watcher·큐 데몬은 사용자 승인 없이 종료하지 않는다.
- 역할이 끝난 구 경로를 다시 만들거나 활성 기준으로 사용하지 않는다.
- 답변은 한국어로 한다.
