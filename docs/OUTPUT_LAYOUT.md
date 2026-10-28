# 학습 결과 경로

2026-10-28 사용자 요청으로 본학습 checkpoint·TensorBoard·진단을 프로젝트별 `output/`으로 모았다. 검증/평가와 source snapshot·상태·공통 캐시는 이번 본학습 결과 이동에서 분리해 보존했다.

| 소속 | 본학습 결과 |
| --- | --- |
| KSH JointPush | `ksh_jointpush/output/<실험>/` |
| 다른 JointPush 실험 | `jointpush/output/<실험>/` |
| Carry/Stack planner, Task allocation | `TokenHSI-coord/output/{carry_planner,stack_planner,task_allocation}/` |
| Stage1 Chair | `stage1_combined/output/<실험>/` |
| Push/Door | `stage1_fix/output/push_door/<실험>/` |
| Table Carry | `stage2_jointcarry/output/table_carry/<실험>/` |

이미 각 프로젝트의 `output/`에 있던 결과는 유지했다. `TokenHSI/`와 외부 저장소의 결과·원본 데이터는 이동 대상이 아니다.

## 실행 중인 결과와 호환 경로

48개 경로를 이동하고 checkpoint 931개의 inode·크기 보존을 확인했다. 이동 전후 경로는 같은 파일을 가리킨다. 구 경로는 심링크로 유지하므로 과거 로그·환경 sidecar·checkpoint 참조를 그대로 사용할 수 있다.

실행 중인 KSH 본학습과 TensorBoard가 사용하는 5개 저장 경로는 물리적으로 이동하지 않았다. 새 `output/` 경로가 기존 저장소를 가리킨다. 당시 `ksh_jointpush/output/`·`output_etc/`도 기존 저장소를 가리켰으나, 아래 후속 정리에서 학습 중단 후 실제 디렉터리로 전환했다.

`ksh_jointpush/`는 최상위 독립 프로젝트다. frozen source·전용 adapter·scripts·tests·[실행 안내](../ksh_jointpush/EXECUTION.md)를 함께 배치했다. 예전 `jointpush/ports/ksh_jointpush`와 전용 실행 파일 경로는 호환 링크로 유지한다. 원본 source manifest의 554개 파일은 해시를 그대로 유지한다.

공통 `runs/queue`, 생성 cfg, 캐시, 실행 상태, 고정 source snapshot, 평가·확인 기록은 기존 위치에 남긴다. 과거 기록에 적힌 launch-time 경로를 일괄 치환하지 않는다. 새 본학습 launcher와 재개·평가·viewer의 결과 참조는 새 위치를 사용한다.

## 확인과 이동 기록

- `runs/training_output_migration_20261028.json`: 53개 경로의 원본/목적지, 실제 이동과 사용 중 경로 연결 구분.
- `runs/ksh_project_promotion_20261028.json`: KSH 최상위 분리와 전용 파일의 호환 링크.
- `python3 scripts/organize_training_outputs.py`: 추가 이동 대상 확인. 기본은 읽기 전용이며 현재 추가 이동은 0개다.
- `python3 scripts/organize_training_outputs.py --apply`: 명시적으로 계획한 기존 결과만 정리한다. 사용 중인 경로를 다시 검사하고 목적지가 있으면 덮어쓰지 않는다. 큐·새 실험은 만들지 않는다.

KSH의 새 실행은 [EXECUTION.md](../ksh_jointpush/EXECUTION.md), 로컬 변경은 [LOCAL_CHANGELOG.md](../ksh_jointpush/LOCAL_CHANGELOG.md)를 따른다. frozen source 안의 원본 문서는 provenance 보존을 위해 수정하지 않았다.

검증: 이동 안전성 CPU4tests, KSH CPU19tests, 새 경로의2환경 CPU reset/정책/부분reset(optimizer0, rc0), 원본554파일 해시,53쌍 경로 접근,46개 실행 파일 구문·diff 검사를 통과했다. 기본 `scripts/check_docs.py`는 이 checkout 대신 `/home/hwanhee/CVPR2027`을 검사하도록 기존부터 하드코딩되어 외부 steer 진입 문서2개의 링크 누락을 보고한다. 외부 문서는 수정하지 않았다.

## 2026-10-28 runs 로그·과거 실험 정리

현재 실행·코드·심링크 의존성을 검사해 완료 실험 폴더81개와 미사용 로그/실행 파일140개를 압축 보관한 뒤 `runs/` 원본을 제거했다. 보관 위치는 각 프로젝트 `output_etc/archive/runs/`, 공통 기록은 `runs/archive/cleanup_20261028/`다. 원본13,672,886,443바이트의 실험 폴더를10,840,947,819바이트로 압축했고 개별 로그도 별도 보관했다. `runs/` 자체 용량은 약23GB에서9.9GB로 줄었다. 이 감소에는 프로젝트별 보관 위치로 이동한 양이 포함된다.

압축마다 tar 원본 대조·SHA-256 기록을 수행하고 제거 직전에 열린 파일·실행 인자·cwd를 재확인했다. 원시 결과·실패 근거를 압축본에 보존했으며 checkpoint·큐 원장·설정·설치/빌드 경로는 유지했다. 현재 학습/supervisor/TensorBoard/proxy/viewer 생존과 학습 event 갱신을 확인했다.

[보관 인덱스 및 복원 방법](../runs/archive/cleanup_20261028/INDEX.md), [상세 목록·압축본 해시·유지 이유](../runs/archive/cleanup_20261028/manifest.json).

미사용 상위 실험 폴더 안에 있던 이전 호환 링크도 압축되었으므로, 해당 구 경로는 보관 인덱스에서 찾는다. 현재 프로젝트 `output/` 결과와 실행 중인 경로는 유지한다. 과거 launch-time 경로가 담긴 문서는 역사 기록으로 보존했다.

## 2026-10-28 KSH 학습 중단 후 출력 경로 확정

사용자 요청으로 balanced 학습을 공식 MPS client 종료 후 중단했다. `runs/jointpush/ksh_jointpush_rsi/{output,output_etc}`를 `ksh_jointpush/{output,output_etc}`로 이동하고 프로젝트 쪽 역방향 링크를 제거했다. 구 runs 출력 경로의 호환 링크는 생성하지 않았다. 생성 설정·launch 기록은 runs에 보존한다.

weight 28개의 inode·크기를 이동 전후 대조했고, 학습/평가 bootstrap의 출력 검증 root도 `ksh_jointpush/`로 변경했다. TensorBoard 6016만 새 경로로 재연결했으며 다른 학습과 MPS 서버는 유지했다. 기록: `runs/ksh_output_relocation_20261028.json`.
