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

실행 중인 KSH 본학습과 TensorBoard가 사용하는 5개 저장 경로는 물리적으로 이동하지 않았다. 새 `output/` 경로가 기존 저장소를 가리킨다. 특히 `ksh_jointpush/output/`·`output_etc/`는 기존 `runs/jointpush/ksh_jointpush_rsi/`를 계속 사용한다. 이 링크는 현재 실행을 보존하기 위한 예외이며 중단·재시작하지 않았다.

`ksh_jointpush/`는 최상위 독립 프로젝트다. frozen source·전용 adapter·scripts·tests·[실행 안내](../ksh_jointpush/EXECUTION.md)를 함께 배치했다. 예전 `jointpush/ports/ksh_jointpush`와 전용 실행 파일 경로는 호환 링크로 유지한다. 원본 source manifest의 554개 파일은 해시를 그대로 유지한다.

공통 `runs/queue`, 생성 cfg, 캐시, 실행 상태, 고정 source snapshot, 평가·확인 기록은 기존 위치에 남긴다. 과거 기록에 적힌 launch-time 경로를 일괄 치환하지 않는다. 새 본학습 launcher와 재개·평가·viewer의 결과 참조는 새 위치를 사용한다.

## 확인과 이동 기록

- `runs/training_output_migration_20261028.json`: 53개 경로의 원본/목적지, 실제 이동과 사용 중 경로 연결 구분.
- `runs/ksh_project_promotion_20261028.json`: KSH 최상위 분리와 전용 파일의 호환 링크.
- `python3 scripts/organize_training_outputs.py`: 추가 이동 대상 확인. 기본은 읽기 전용이며 현재 추가 이동은 0개다.
- `python3 scripts/organize_training_outputs.py --apply`: 명시적으로 계획한 기존 결과만 정리한다. 사용 중인 경로를 다시 검사하고 목적지가 있으면 덮어쓰지 않는다. 큐·새 실험은 만들지 않는다.

KSH의 새 실행은 [EXECUTION.md](../ksh_jointpush/EXECUTION.md), 로컬 변경은 [LOCAL_CHANGELOG.md](../ksh_jointpush/LOCAL_CHANGELOG.md)를 따른다. frozen source 안의 원본 문서는 provenance 보존을 위해 수정하지 않았다.

검증: 이동 안전성 CPU4tests, KSH CPU19tests, 새 경로의2환경 CPU reset/정책/부분reset(optimizer0, rc0), 원본554파일 해시,53쌍 경로 접근,46개 실행 파일 구문·diff 검사를 통과했다. 기본 `scripts/check_docs.py`는 이 checkout 대신 `/home/hwanhee/CVPR2027`을 검사하도록 기존부터 하드코딩되어 외부 steer 진입 문서2개의 링크 누락을 보고한다. 외부 문서는 수정하지 않았다.
