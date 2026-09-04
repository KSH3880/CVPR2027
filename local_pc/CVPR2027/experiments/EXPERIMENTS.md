# 통합 실험 결론

트랙별 상세 비교를 복사하지 않고, 트랙 간 연결에 필요한 확정 결론만 모은다.

| 트랙 | 자동 결과 | 검토된 결과 |
|---|---|---|
| steer | [AUTO](AUTO_RESULTS_steer.md) | [EXPERIMENTS](EXPERIMENTS_steer.md) |
| ma | [AUTO](AUTO_RESULTS_ma.md) | [EXPERIMENTS](EXPERIMENTS_ma.md) |
| lab | [AUTO](AUTO_RESULTS_lab.md) | [EXPERIMENTS](EXPERIMENTS_lab.md) |

## 현재 크로스트랙 결론

- steer와 ma는 각각의 기준 설계가 확정될 때까지 독립적으로 진행한다.
- 두 트랙 모두 별도 `new_extra` 입력을 사용하므로 통합 시 builder의 단일 extra 가정을 확장한다.
- lab에서 확정한 실행 레시피는 전체 PLAN으로 승격하고, 연구적 근거는 lab 결과에 보존한다.

## 통합 게이트

1. steer 단독 회귀
2. ma 단독 회귀
3. 두 토큰 동시, 각각 frozen
4. 두 토큰 동시 fine-tune
