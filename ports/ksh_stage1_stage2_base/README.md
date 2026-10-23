# Stage 1·2 원본 보관과 PUSH 작업 사본

원본: [KSH3880/CVPR2027 · stage1_stage2_base](https://github.com/KSH3880/CVPR2027/tree/28f02b3278f0a3e703cfa22a1e8cc376bb8ec512).
고정 commit: `28f02b3278f0a3e703cfa22a1e8cc376bb8ec512` (2026-10-19 가져옴).

| 경로 | 용도 |
| --- | --- |
| `upstream/` | Git 원본387개 파일 그대로 보관. 일반 파일·디렉터리 쓰기 권한 제거. 수정하거나 실행하지 않는다. |
| `push/` | 독립 복사본. 향후 PUSH 통합 코드·config 수정은 이 디렉터리에서만 진행한다. |
| `UPSTREAM_MANIFEST.json` | 원본 commit/tree, 파일별 Git blob·SHA256·mode 기록. |
| `verify_upstream.py` | 보관본의 내용·심링크 문자열·실행권한·읽기 전용 상태 검증. |

`upstream/`은 원본과 동일하며, `push/`에는 별도 Stage1 PUSH 실험을 추가했다. 기존 루트 `tokenhsi/`의 코드와 실행 중인 학습은 그대로 유지한다. 복사는 hardlink가 아니므로 `push/`의 일반 파일을 수정해도 `upstream/`은 바뀌지 않는다. 원본 업데이트는 별도 버전 경로에 가져오고 이 보관본은 덮어쓰지 않는다.

기준 config는 각 사본의 `tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml`, `stage2_shared_full_success.yaml`이며, 대응 학습 YAML은 `tokenhsi/data/cfg/train/rlg/`에 있다.

저장소 루트에서 보관본 확인:

```bash
python3 ports/ksh_stage1_stage2_base/verify_upstream.py
# 최초 작업 사본까지 원본과 동일한지 확인할 때만 사용. PUSH 수정 후에는 차이가 나는 것이 정상이다.
python3 ports/ksh_stage1_stage2_base/verify_upstream.py --check-push
```

## 실행 준비 상태

Stage1 PUSH의 데이터 연결·실행 경로·짧은 학습/평가를 연결했다. [전용 명세](push/markdowns/reference/stage1_push.md), [학습·로컬·VNC 명령](push/markdowns/config.md)을 따른다. 원본12개 데이터 링크는 `upstream/`에 그대로 보존하며 `push/`의 loco 링크와PUSH 데이터 링크만 이 서버에 연결했다. SIT/CARRY/CLIMB 데이터 링크를 이 서버에 연결했다. 기존 네 과제를 유지하는 별도 다섯 과제 Stage1/Stage2와 PUSH_AT 전용 coupled 코드는 [추가 명세](push/markdowns/reference/five_task_push_at.md)를 따른다. 사용자 지시에 따라 최종 버전은 코드/CPU 검사까지만 수행하고 학습·RSI·시뮬레이션은 추가 실행하지 않았다.

PUSH wrapper는 working directory와 `PYTHONPATH`를 사본에 고정하고 output/검증 경로와 config·train/test/VNC를 분리한다. 사본 내부 `AGENTS.md`의 Stage1 2048환경·Stage2 1024환경 및 검증 출력 규칙을 따른다.
