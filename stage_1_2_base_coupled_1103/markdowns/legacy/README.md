# 과거 기록 보관

이 폴더는 과거 분석·설계·서버 이관 자료다. 문서 속 실행 명령·실험 목록·검증 경로는 작성 당시 기준이며 현재 실행 지침으로 사용하지 않는다. 현재 설정은 [config](../config.md), 구현 위치는 [structure](../structure.md)를 따른다.

| 문서 | 보관 이유 |
| --- | --- |
| [이전 changelog](changelog_before_cleanup.md) | 과거 변경·실험·검증 이력 전체 보존 |
| [원본 Carry/Sit/Climb 분석](tokenhsi_original_carry_sit_climb.md) | 원본 태스크의 상세 분석 |
| [TokenHSI·v2 학습/평가 비교](tokenhsi_vs_stage1_v2_train_eval.md) | v2 도입 당시 설계·검증 근거 |
| [다른 서버 MPS 설치](mps/CODEX_SETUP.md) | 다른 머신으로 가져가는 설치 요청/스크립트 참고 |
| [MPS GPU4·6 분리 검사](mps/gpu4_gpu6_test.md) | 과거 격리 실험과 수동 재현 명령 |

이번 checkout의 확인 산출물은 `output_etc/check/`로 이동했다. 이전 `output/...cleanup...` 경로는 `output_etc/check/stage1_cleanup_20261015/path_migration.json`에 대응 관계와 이동 시 파일 hash가 있다. 원본 레포의 결과 경로는 변경하지 않았다.

네 컨피그 정리 전 문서·설계·changelog는 [pre_four_configs](pre_four_configs/README.md)에 보존한다.
