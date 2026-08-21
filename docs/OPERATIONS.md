# CVPR2027 공통 운영 규칙

Claude와 Codex가 함께 사용하는 안정적인 운영 계약이다.
자주 바뀌는 실험 설계와 GPU 배치 정책은 `plan/PLAN.md`에서 관리한다.
에이전트별 진입 파일과 다른 문서에 아래 규칙을 반복하지 않는다.

## 문서 역할

| 문서 | 기준으로 삼는 내용 |
|---|---|
| `METHOD_V2.md` | 연구 목표와 방법론 |
| `REPOS.md` | 저장소별 역할, 코드 위치, 통합 관계 |
| `plan/PLAN.md` | 공통 실험 설계·GPU 배치 + 트랙 큐 미러·GPU·주의 대시보드 |
| `plan/PLAN_<track>.md` | 사용자가 만드는 활성 큐, 트랙 판정 기준, 다음 분기 |
| `experiments/AUTO_RESULTS_<track>.md` | 완료·실패 실험의 자동 요약. 직접 수정 금지 |
| `experiments/EXPERIMENTS_<track>.md` | 상단 최신순 자동 결과 + 그 아래 비교 해석과 확정 결론 |
| `experiments/EXPERIMENTS.md` | 트랙 간 확정 결론과 통합 인덱스 |
| `<repo>/CHANGELOG.md` | 코드에서 무엇을 왜 바꿨는지와 검증 기록 |
| `docs/PITFALLS.md` | 실제 측정 사고와 실험 전 체크리스트 |
| `docs/MULTIAGENT_PORT.md` | 단일→다중 에이전트 이식 안내. 태스크 추가·통합 때 참조 |
| `runs/queue/results.tsv` | 자동 평가의 원시 결과. append-only |
| `AGENTS.md`, `CLAUDE.md` | 에이전트 진입점과 읽기 순서. 프로젝트 규칙을 복제하지 않음 |
| `docs/ENV_SCALING.md`, `TOKENHSI_*.md` | 기술 참고자료. 현재 실행 상태를 기록하지 않음 |

규칙이 충돌하면 이 문서가 우선한다. 실제 자동화 설정은 트랙 JSON, 현재 상태·설계·배치는
PLAN, 자동 수치는 AUTO_RESULTS, 검토된 해석은 EXPERIMENTS를 따른다.

## 세션 시작 순서

1. 이 문서를 읽는다.
2. 실험이나 평가를 다루면 `docs/PITFALLS.md`를 읽는다.
3. 연구 목적이 필요하면 `METHOD_V2.md`, 코드 위치가 필요하면 `REPOS.md`를 읽는다.
4. `plan/PLAN.md`와 작업 트랙 PLAN을 읽는다.
5. 결과가 필요한 경우 해당 트랙 결과 원장과 CHANGELOG를 읽는다.

## 프로세스 안전

- 실행 중인 학습·평가·watcher·큐 데몬을 임의로 종료하지 않는다.
- 큐 데몬을 멈춰도 이미 시작된 학습과 평가는 detached 상태로 계속 돈다.
- PLAN의 이동·대규모 편집은 큐 데몬을 멈춘 뒤 한다. 자동화는 마커 내부만 원자적으로 교체한다.
- 큐 데몬을 재시작해도 detached 학습·평가·watcher는 유지한다.
- 프로세스 상태와 경로는 실행 직전에 다시 확인한다. 문서에 적힌 PID를 현재값으로 가정하지 않는다.

## 변경되는 연구 정책

- 공통 실험 설계와 GPU 배치는 `plan/PLAN.md`를 따른다.
- 트랙별 판정 기준과 다음 분기는 `plan/PLAN_<track>.md`를 따른다.
- GPU 허용 목록·상한·정착 시간과 env 검증값은 `runs/queue/tracks/<track>.json`이 실제 설정이다.
- 실행 중인 bash 스크립트는 in-place로 수정하지 않는다.

## 실행 큐 규약

- 실제 큐 파일은 `runs/queue/tracks/<track>.json`의 `queue` 값이 정한다.
- `<!-- QUEUE -->`와 `<!-- /QUEUE -->` 마커를 유지한다.
- 표는 tag, 질문·판정 기준, 상태/GPU, 환경변수의 네 열 순서를 유지한다.
- 사용자는 1·2·4열과 행 순서를 편집한다. 3열 상태/GPU와 완료 행 제거는 자동화가 관리한다.
- 환경변수 열에는 트랙 JSON의 `base_env`와 다른 값만 적는다. 기본값은 반복하지 않는다.
- 트랙 PLAN의 큐만 수정한다. `plan/PLAN.md`의 큐 요약은 질문·상태·환경변수까지 자동 미러링된다.
- tag는 큐 파일 안에서 유일해야 하고 환경변수는 `NAME=value` 형태여야 한다.
- 각 트랙 JSON의 `allowed_env` 밖 변수와 `forbidden_env` 변수는 실행 전에 거부한다.
- `seed_env`를 제외한 환경변수가 같은 행을 한 설정으로 보고 `max_seeds`를 넘기지 않는다.
- 큐 행과 트랙 JSON은 매 polling 주기에 다시 읽는다. 큐 편집 때문에 데몬을 재시작하지 않는다.
- steer 학습 큐에서 `STEER_SPACING`을 사용하지 않는다. 환경 간격은 `STEER_ENVSPACING`을 사용한다.
- 큐 파일·마커·환경변수 검증이 실패하면 새 잡을 띄우지 않고 로그에 오류를 남긴 뒤 다음 주기에 다시 검사한다.
- 자동화는 큐에 없는 실험이나 다음 분기를 만들지 않는다.
- 의도적으로 비정상 종료를 측정하는 행만 `QUEUE_EXPECT_RC=<코드>`를 명시한다. 기본 예상값은 0이다.
- 완료 순서는 `원시 결과/종료 기록 → AUTO_RESULTS → 큐 행 제거`로 고정한다. 앞 단계가 실패하면 행을 보존한다.
- 실패 행은 자동 결과에 남기고 활성 큐에서는 제거하되 `plan/PLAN.md`의 `NEEDS ATTENTION`에 표시한다.

## 결과와 기록

결과 흐름은 `학습/프로브 종료 → 필요한 최종 평가 → 원시 결과·종료 표식 →
AUTO_RESULTS_<track>.md → EXPERIMENTS_<track>.md 해석` 순서다. steer 원시 평가는
`results.tsv`, ma 최종 평가는 `eval_<tag>.log`의 `MA_EVAL_SUMMARY`, lab은 `LAB` 종료
요약을 기준으로 한다.

- 원시 결과와 실패한 실험을 지우지 않는다.
- 결과를 인용할 때 평가 반복 산포와 학습 시드 산포를 구분한다.
- 자동 결과 원장은 실행 사실과 수치만 기록한다. 비교·채택·기각·다음 실험 결정은 큐 자동화가 하지 않는다.
- 트랙 EXPERIMENTS의 `AUTO_RESULTS` 마커 내부는 자동 갱신하며 최신 결과를 위에 둔다.
  수동 해석도 새 항목을 기존 기록 위에 추가해 최신순을 유지한다.
- 의미 있는 코드 작업이 끝나면 해당 CHANGELOG 날짜 아래에 `### 제목`과 이유·검증 결과를 남긴다.
- Claude의 Edit/Write 훅 외의 변경, Codex 변경, 셸로 한 변경은 자동 기록된다고 가정하지 않는다.

## 파일과 저장소 경계

- `TokenHSI/`는 원본 baseline이다. 수정하지 않는다.
- `isaacgym/`, `pytorch-src/`, `pytorch3d-src/`는 빌드용 외부 소스다. 탐색·이동·정리 대상이 아니다.
- 새 실험 산출물은 `runs/` 아래에 둔다.
- `ddp_shim/`과 설치·빌드 경로는 옮기지 않는다.
- 역할이 끝난 구 문서·백업을 다시 활성 기준으로 만들지 않는다.

## 운영 명령

```bash
bash scripts/gpumap.sh
bash scripts/queue_daemons.sh status
python3 scripts/check_docs.py
```

큐 데몬의 시작·중지·재시작은 `scripts/queue_daemons.sh`로만 한다.
