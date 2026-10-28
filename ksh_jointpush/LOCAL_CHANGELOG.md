# 로컬 실행 변경

## 2026-10-28

### 최상위 프로젝트 분리 및 결과 경로 통합

사용자 요청으로 frozen source·전용 adapter/scripts/tests를 `CVPR2027/ksh_jointpush/`에 배치했다. 기존 실행 경로는 원자적 교환 및 호환 링크로 유지하고 현재 학습 결과는 `output/`, 확인 결과는 `output_etc/check/`에서 접근한다. 원본554파일 해시·기존balanced 설정 일치·CPU19tests·이동안전성4tests 통과. 새 경로의2환경 CPU reset/정책/부분reset도rc0·optimizer0으로 통과했다. 학습/supervisor/TensorBoard를 중단·재시작하지 않았다. [실행 안내](EXECUTION.md), [전체 이동 기록](../docs/OUTPUT_LAYOUT.md).
