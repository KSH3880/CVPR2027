# Multi-Agent Carry Stop–Wait–Resume Stage2 — 통합 문서 안내

이 문서의 기존 방법 설계·실험 아카이브는 2026-08-27에 다음 HANDOFF 문서로
병합했다.

- [MASteer pickup → steering → 교차 양보 통합 인수인계](HANDOFF_MASTEER_PICKUP_YIELD_2026-08-25.md)

최신 기준은 위 문서 하나다. 다음 내용도 모두 그 문서에서 관리한다.

- strict pickup과 anti-kicking 정의
- corrected pickup 2×2의 3k/6k 결과
- `ms12_base4L_s0`, `f22b_long_c20_e25000`과 현재 A의 차이
- MRAND4·CLIP1 pickup A 두 arm의 실행 계약
- carry-aware mastop B/C/D 상태기계와 평가 관문
- 현재 D heuristic과 learned planner/WM의 경계
- Vulkan UUID fail-closed GUI 절차
- 다음 실험 순서와 산출물 경로

과거 링크가 끊기지 않도록 이 파일은 안내 경로로만 유지한다. 상충하는 과거 문구를
이 파일에 다시 복사하지 않는다.
