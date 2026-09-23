# Legacy 문서

현재 실행법은 [config.md](../config.md), 코드 위치는 [structure.md](../structure.md), Stage 1 설계는 [edge_context_stage1.md](../edge_context_stage1.md)를 기준으로 한다.

## 과거 실험 설명

- [Holding k=10 reward](experiments/approach_distance_success_holding_k10_reward.md)
- [OnTop mixed](experiments/ontop_mixed_config.md)
- [Edge context success](experiments/edge_context_success.md)
- [Sampled OnTop](experiments/edge_context_ontop.md)
- [SIT/CLIMB interaction](experiments/edge_context_interaction.md)

## 구현 전 설계 명세

- [Scalar PRE/TERM context](specs/CODEX_approach_clean_edge_context_success_spec.md)
- [OnTop edge sampling extension](specs/CODEX_ontop_edge_sampling_extension_spec.md)
- [SIT/CLIMB extension](specs/CODEX_sit_climb_extension_from_ontop.md)
- [Context-free scenario without CLIMB](specs/CODEX_scenario_training_no_climb.md)

이 문서들은 과거 설계와 실험 해석용이다. 현재 동작과 명령은 실제 코드·YAML 및 상위의 현재 문서를 우선한다.
