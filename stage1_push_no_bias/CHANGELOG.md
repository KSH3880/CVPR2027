# 로컬 변경 기록

## 2026-11-15

### sct_nobias의 stage1_push_no_bias 독립 이식

- 원격 `sct_nobias` 커밋 `c385e3159bb7abcfa84efc160308ee5d2b6e566a`를 현재 저장소의 원격 추적 ref로 받은 뒤 Git archive로 복사했다. `ksh`의 checkout·파일은 변경하지 않았으며 GPU·학습·평가·viewer를 실행하지 않았다.
- 원본 코드·설정·실행기를 그대로 보존했다. task attention bias를 제거하는 scratch 설정이며 기본2명·2048환경·4물체·900step·seed42, 초기 checkpoint가 필요 없다. 이전 Stage2 사람 attention no-bias와 별개다.
- PUSH 모션은 기존 로컬 복사본에서 실제 복사하고, 공통 네 모션 디렉터리는 읽기 전용 공유 데이터에 연결했다. 실행 의존성에 `ksh` 경로는 없다. 코드·설정·모션 YAML만 Git에 포함하고 모션 배열·checkpoint·출력은 제외한다.
- `local.py`의 외부 모션 준비·무결성 검사와 README의 학습·일반/단독 평가 명령을 추가했다. `IMPORT_MANIFEST.json`에 원본 및 모션311파일 SHA256과 공유 데이터 링크를 기록했다. 가져온 원본 이력은 `changelog.md`에 보존한다.
- 검증: 가져온311파일 해시·학습 모션161경로 확인. no-bias/PUSH 관련 CPU 테스트42개 통과, 외부21k/78k checkpoint가 필요한2개 skip. 첫 실행의 tmp fixture 상위 폴더 누락은 `output_etc/check` 생성 후 해당 테스트1개를 재실행해 통과했으며 README에도 준비 명령을 포함했다. Python·셸 구문과 새 문서 링크를 검사했다.
- 가져온 원본의 공백은 SHA256 보존을 위해 정리하지 않았다. 이번에 작성한 파일에 대해 diff 공백 검사를 수행했다. GPU/시뮬레이터에서의 실행은 이번 검증 범위에 포함하지 않는다.
