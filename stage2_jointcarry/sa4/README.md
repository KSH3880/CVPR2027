# 전달받은 SA4 모델 (2026-10-16)

기본 SA4와 앵커 입력(grasp-view) SA4를 독립적으로 가져왔다. 출처·고정 checkpoint·SHA256·실행 설정은 [SA4_SOURCE.json](../SA4_SOURCE.json), 학습 당시 설정 사본은 `../../runs/jointcarry_import_20261016/`에 있다.

| 선택 | 입력 | 가져온 epoch |
| --- | --- | --- |
| `basic` | 기존 물체 중심 관측 + 사람 간 SA4 | 10000 |
| `anchor` | 사람별 opposite holding anchor 관측 + 사람 간 SA4 | 2500 |

앵커는 reset 때 가까운 반대편으로 배정하고 episode 동안 유지한다. 관측의 payload 위치와 rigid-point 속도를 바꾸며 실제 물체 크기·물리·보상은 유지한다. 기본/앵커/기존 CA4 checkpoint는 서로 혼용할 수 없다. 학습 중 덮어쓰는 이름 없는 checkpoint 대신 가져오는 시점의 최신 번호 저장본을 복사했다.

`stage2_jointcarry`에서 실행한다. GPU는 사용자가 점유를 확인해 지정한다.

```bash
# 기본 모델: 로컬 viewer (2명, 1환경, 4물체, 1회)
TOKENHSI_GPU=5 bash sa4/run.sh basic test 2 1 4 1
# 앵커 모델: 로컬 viewer
TOKENHSI_GPU=5 bash sa4/run.sh anchor test 2 1 4 1
# 화면 없는 평가
TOKENHSI_GPU=5 HEADLESS=1 bash sa4/run.sh anchor test 2 16 4 3
# 서버 VNC
TOKENHSI_GPU=5 PORT=6087 bash sa4/run.sh anchor vnc 2 1 4 10
```

기본 선택은 `anchor test`다. 출력은 `../runs/jointcarry_sa4/`에 생성한다. 기존 zero-shot CA4 기본값은 보존했다.

각 실험의 새 학습은 `SA4_SOURCE.json`의 `test_script`와 같은 접두사의 `_train.sh`를 사용한다. 예:

```bash
STAGE1_CHECKPOINT=/absolute/path/stage1.pth TOKENHSI_GPU=5 OUTPUT_PATH=../runs/jointcarry_sa4/anchor_train bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_graspview_task_embedding_train.sh 2 2048 4
STAGE1_CHECKPOINT=/absolute/path/stage1.pth TOKENHSI_GPU=5 OUTPUT_PATH=../runs/jointcarry_sa4/basic_train bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_train.sh 2 2048 4
```

검증: CPU 회귀19개 통과, 두 실제 checkpoint 각각 CPU PhysX 1환경·8프레임 추론 정상 종료 및 metrics 저장. 학습 당시 핵심 설정 일치·checkpoint SHA256·Python/셸 문법 확인. GPU/VNC 화면과 장기 성능은 검증하지 않았다. 검증 로그는 `../../runs/jointcarry_import_20261016/`에 있다.

원본 config.md의 기본 SA4 VNC 예시와 같은 시작/graph는 `TOKENHSI_GPU=6 PORT=6087 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith bash sa4/run.sh basic vnc 2 1 4 10`이다. 기존 기본값인 loco/joint_carry_at와 다르다. 이 서버에서 확인한 GPU6 직접 컨텍스트 실행 인자와 환경은 `../../runs/jointcarry_sa4/current_carrywith_view.json`이 가리키는 `launch.json` 참조.

## 원본 크기 직사각형 테이블

SA4에도 기존 native payload 교체를 연결했다. 기본은 **앵커 epoch2500 + TeamHOI 기본 직사각형2.0×1.2×0.86m·60.5184kg**, GPU6 PhysX/pipeline/policy다. 원본 mesh 크기·질량을 유지하며 O0만 교체하고 실제 bbox를 사람별 앵커 관측에 반영한다. 기존 bbox 반대면 앵커는 mesh 표면 파지점과 다를 수 있다.

원본 크기 가구에는 박스용 carryWith/pickUp/putDown RSI가 맞지 않아 **loco 접근 시작·joint_carry_at만** 지원한다. 든 상태 테이블 시작은 별도 자세 정합이 필요하며 이 경로에서는 거부한다. 모델 학습이나 reward 변경은 없다.

```bash
# 앵커 모델 직사각형 테이블 VNC (6088)
PORT=6088 bash sa4/native_vnc.sh teamhoi_default_rectangle anchor
# 기본 SA4 비교: 별도 포트 사용
PORT=6089 bash sa4/native_vnc.sh teamhoi_default_rectangle basic
# 화면 없는 짧은 GPU6 추론
EPISODE_LENGTH=32 EVAL_REPEATS=1 bash sa4/native_test.sh teamhoi_default_rectangle anchor
# 원래 박스 대조군
EPISODE_LENGTH=32 bash sa4/native_test.sh box anchor
```

출력은 `../runs/jointcarry_sa4/`이며 `native_run.json`에 모델/물체/크기/질량, `gpu_context.json`에 직접 GPU6 컨텍스트를 기록한다. native 모델은 `SA4_SOURCE.json`의 checkpoint를 사용하며, 이전 CA4용 `zeroshot/native_*.sh`와 구분한다. 지원 asset 이름은 `../runs/jointcarry_zeroshot/native_assets/catalog.json` 참조.
