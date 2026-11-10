# 변경 기록

## 2026-11-10

### SCT + LLM Scheduler 독립 base 생성

- 제공된 프롬프트와 JSON 메시지 계약, 5개 데모 검증, 일회 호출 및 추론 audit 구현.
- enabling/coupled 혼합 제약을 정수 centisecond difference constraints로 검증. 잘못된 응답 자동 보정·재시도·fallback 없음.
- 원본 FSM의 projection 방식을 로컬 adapter로 이식. 초기 시간 기반 해제만 사용하고 release 후 추가 gate 없음.
- 실제 event recorder, censored label, bounded episode callback 연결, GPU6/7 UUID guard 추가.
- CPU unittest 24개 통과. 실제 Isaac Gym·LLM 호출·성공률은 미검증.
- coupled/enabling 오프라인 CLI 모두 exit 0. 각각 A/B release 2/0초, 0/2초 확인. 11개 Python 파일 구문 검사 통과. 로그는 `output_etc/check/offline_{coupled,enabling}/`.
- ksh 코드·기존 프로세스·MPS·큐·baseline 변경 없음. 합성 데이터는 실제 calibration용이 아님.
