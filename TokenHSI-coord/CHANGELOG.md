# TokenHSI-coord — juan 변경 기록

## 2026-09-08

### Sequential-stack 전용 실행 bridge

- `sequential_bridge.py`에 strict loader, phase별 경로 적용, 비활성 agent의 정지 근사를
  이용한 후보 선택, 동일 공간 호길이상의 XY/속도 보간을 추가했다.
- traj에서 가져온 모델·planner·checkpoint 코드는 수정하지 않았다.
- 기존 75개 + 새 bridge/runtime 17개, 총 92개 CPU 테스트 통과.
  `output/c13.pth`의 C2 schema, path30 설정, step 300과 random-priority 계약을 확인했다.
- 실제 masteer sequential-stack 2-env/60-step GPU smoke에서 C13 경로를 75 agent-row에
  설치했고 invalid/fallback 0으로 종료했다(unsafe 19; 짧은 smoke라 성능 판정 제외).

### traj coordinator 독립 반입

- 출처: `traj` commit `5f39199f7ea9396532fe251c46aa634734ee9f10`.
- `coordinator/` 전체 26개 파일을 원본 그대로 추가했다.
- 기존 기능에 영향을 주지 않도록 simulator와 sequential-stack 연결은 추가하지 않았다.
- 검증: CPU 자체 테스트 75개 통과(모델 출력, gradient, planner, checkpoint round-trip).
- 26개 원본 파일의 Git blob hash 일치 확인. 기존 tracked 파일 diff 없음.
