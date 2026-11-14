# 로컬 변경 기록

## 2026-11-15

### enable_nobias 독립 학습 프로젝트 이식

- `ksh/stage_1_2_base_1031`의 `enable_nobias` 커밋 `16f8b2c664ee94df482f72e87a961086950bce74`를 Git archive로 읽어 코드·설정·실행기를 복사했다. 원본 추적 파일의 내용은 유지했으며 이전 `.git`·원본 운영 지침·출력은 가져오지 않았다.
- Stage1 26k 초기 가중치와 PUSH 모션은 실제 복사했다. 공통 모션 네 경로만 읽기 전용 공유 데이터에 연결했다. 데이터·가중치·실행 결과는 Git에서 제외한다.
- `local.py`에 데이터 준비 및 해시·모션 경로 검사를 추가하고 README에 학습 명령·외부 자산 준비법을 기록했다. 출처와 503개 파일의 SHA256은 `IMPORT_MANIFEST.json`에 있다. 가져온 원본 변경 기록은 `changelog.md`에 보존한다.
- CPU 검증: 가져온 503개 파일의 해시와 학습 모션 161개 경로 확인. `test_before_total_at1mm_nobias.py`, `test_before_relation_fusion.py` **19 passed**. 원본 작업 트리는 clean이며 `ksh`로 연결되는 실행 의존성은 없다.
- Python 197개·셸 165개 파일 구문 검사와 변경 문서 링크 검사 통과. 공통 `scripts/check_docs.py`는 기존 하드코딩 경로 `/home/hwanhee/CVPR2027`를 검사하다 외부 steer 진입 문서 두 개의 운영 링크 누락으로 실패했다. 이번 복사본 경로를 검사한 결과가 아니며 외부 문서는 수정하지 않았다.
- 가져온 원본의 공백 경고와 상위 REPOS의 기존 `ksh_jointpush/EXECUTION.md` 누락은 별도 확인했다. 원본 내용 보존을 위해 공백 정리는 하지 않았고, 이번에 작성한 파일과 새 링크만 추가 검증했다.
- GPU·학습·평가·viewer는 실행하지 않았다. 시뮬레이터 및 GPU 동작은 이번 이식의 검증 범위에 포함하지 않는다.
