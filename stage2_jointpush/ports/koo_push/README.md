# Push Stage1 격리 복사본

## Push Stage1 원본 재현 (2026-10-12)

- 코드: `ports/koo_push/`는 `/home/hwanhee/koo_cvpr/stage1_fix`의 tokenhsi/lpanlib/mps 작업 파일 복사본이다. 공동운반 core는 유지한다. 데이터 심링크 12개는 원본에서 해석한 절대 경로로 연결했다. 원본 폴더는 변경하지 않았다.
- 모델: 프로젝트 `runs/jointpush/push_stage1_repro_20261012/checkpoints/`의 고정 epoch29000. GPU는 두 진입점 모두 6으로 고정한다.
- 프로젝트 루트에서 평가: `bash stage2_jointpush/scripts/push_stage1_test.sh` (2명·4환경·각자 Push·600step·1회).
- 프로젝트 루트에서 뷰어: `bash stage2_jointpush/scripts/push_stage1_view.sh`. 기존 뷰어 실행 중에는 중복 실행하지 않는다. noVNC 포트6081을 포워딩하고 `http://localhost:6081/vnc.html?autoconnect=1&resize=remote`에 접속한다. 현재 카메라는 한 사람/상자에 집중한다.
- 원래 OPEN/PUSH 혼합 장면은 `TASK_GRAPH=random_open_push`로 선택한다. 기본 push는 두 사람이 각자의 상자를 미는 Stage1이며 공동 Push가 아니다.
- 로그·metrics·화면 캡처·복사 manifest는 프로젝트 `runs/jointpush/push_stage1_repro_20261012/`에 둔다. 120step smoke와600step4환경 평가가 rc=0으로 완료했고 실제 Isaac Gym 렌더 화면을 확인했다. 작은 표본의 재현 검사이며 장기 성능 판정은 아니다.

