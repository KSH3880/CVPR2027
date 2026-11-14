# 변경 기록

## 2026-11-14

### V4 reward 독립 확인 환경

- CVPR2027에 전용 구현이 없어 원본 코드·두 run 전체·PUSH 모션·Stage1 초기 가중치를 복사했다.
- 원본 접근은 읽기로 제한했다. ksh를 가리키는 데이터 심링크는 실제 파일로 복사했다.
- SHA256 import manifest, checkpoint 확인·viewer 진입점, 로컬 사용 안내를 추가했다.
- 기존 reward CPU 회귀 테스트 4개 통과. 두 run의 마지막 numbered checkpoint(6,000/16,500)를 CPU 로드하고 저장 설정·가중치 finite를 확인했다.
- viewer 명령 dry-run과 실행기 shell 구문 검사를 통과했다. 실제 GPU viewer·학습·정식 평가는 실행하지 않았다. 확인 기록은 `output_etc/check/inspection/`에 둔다.
