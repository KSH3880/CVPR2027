# TokenHSI-coord — juan 변경 기록

## 2026-09-08

### traj coordinator 독립 반입

- 출처: `traj` commit `5f39199f7ea9396532fe251c46aa634734ee9f10`.
- `coordinator/` 전체 26개 파일을 원본 그대로 추가했다.
- 기존 기능에 영향을 주지 않도록 simulator와 sequential-stack 연결은 추가하지 않았다.
- 검증: CPU 자체 테스트 75개 통과(모델 출력, gradient, planner, checkpoint round-trip).
- 26개 원본 파일의 Git blob hash 일치 확인. 기존 tracked 파일 diff 없음.
