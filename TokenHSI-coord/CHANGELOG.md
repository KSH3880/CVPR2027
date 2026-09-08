# TokenHSI-coord — juan 변경 기록

## 2026-09-08

### Sequential-stack 전용 실행 bridge

- `sequential_bridge.py`에 strict loader, phase별 경로 적용, 비활성 agent의 정지 근사를
  이용한 후보 선택, 동일 공간 호길이상의 XY/속도 보간을 추가했다.
- traj에서 가져온 모델·planner·checkpoint 코드는 수정하지 않았다.
- 기존 75개 + 새 bridge/runtime 17개, 총 92개 CPU 테스트 통과.
  C13 architecture 검사는 임시 가중치로 수행했으며 실제 학습 PTH는 이 머신에 없다.

### traj coordinator 독립 반입

- 출처: `traj` commit `5f39199f7ea9396532fe251c46aa634734ee9f10`.
- `coordinator/` 전체 26개 파일을 원본 그대로 추가했다.
- 기존 기능에 영향을 주지 않도록 simulator와 sequential-stack 연결은 추가하지 않았다.
- 검증: CPU 자체 테스트 75개 통과(모델 출력, gradient, planner, checkpoint round-trip).
- 26개 원본 파일의 Git blob hash 일치 확인. 기존 tracked 파일 diff 없음.
