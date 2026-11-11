# 실제 trajectory 보존 기록

사용자가 지정한 `Stage2Distill900V3_00002000.pth`를 그대로 실행한다. 원본 Stage1 SCT로 가중치를 교체하지 않는다. **이 데모는 Stage2 정책의 실측 기록이며, 원본 SCT 정책의 timing evidence로 혼용하면 안 된다.**

- 원본: `/home/hwanhee/ksh/stage1_2_base_teacher/output/stage2_distill_900_v3/Stage2Distill900V3_09-20-41-11/nn/Stage2Distill900V3_00002000.pth`
- SHA256: `c2083707d13de3f32e9244bf0184a7a5b7e8a4d580cbb65a40308d387dd1705b`
- 실행 코드·checkpoint·sidecar는 `teacher_runtime/`에 복사했다. 원본 코드/출력은 쓰지 않는다. 대형 motion 데이터는 읽기 전용 참조다.
- 기존 GPU7 MPS에 연결하며 서버 제어·프로세스 signal을 사용하지 않는다. 평가 900step 종료 후 CUDA synchronize한다.
- 외부 FSM/LLM은 붙이지 않고 모든 agent의 initial release는 0초다. 체크포인트 내부의 학습된 coordination은 유지된다.

## 수집과 선별

| 최종 예제 | 개수 |
|---|---:|
| AT → SIT | 2 |
| AT → CLIMB | 1 |
| AT → ON_TOP | 2 |
| 공동 PUSH 2인 | 2 |
| 공동 PUSH 3인 | 1 |
| 공동 PUSH 4인 | 2 |

한 환경에 하나의 interaction group을 배치한다. 기존 `table2_sampling`의 크기·거리 조건과 XY100 formal success 정의를 사용하고 distractor 물리는 유지한다. 원래 formal evaluation trace/result와 추가 dense trace를 함께 기록한다.

`export_demonstrations.py`는 native group success와 dense trace의 판정을 대조한다. 성공 시점까지 모든 참여자가 유효한 장면만 채택하고, coupled는 모든 참여자의 readiness도 도달했는지 검사한다. 같은 setting의 여러 장면은 초기 root/goal 상대 배치 차이를 기준으로 고른다. 실패 장면은 raw 파일에 보존하며 선별 기록에서만 제외한다. 따라서 선별 데이터로 일반적인 성공률·실패 확률을 추정하면 안 된다.

## 이벤트 의미

**Enabling:** 선행 AT enabling은 실제 물체 중심이 고정 목표와 3D 거리 0.20m 이내가 된 최초 valid control frame이다. 후행의 dependency-sensitive 이벤트는 이번 기록에서는 **명시적으로 기존 formal task completion**이다. 별도 approach/readiness detector를 정의한 것이 아니다. SIT/CLIMB는 root의 full support XY footprint 및 높이 오차 0.20m 조건, ON_TOP은 payload 중심의 해당 조건을 사용한다. 각 JSON에 정확한 오프셋과 proxy 의미를 적었다. 후행의 실제 완료 시간을 접근 준비 시간으로 이름만 바꿔 사용하면 안 된다.

**Coupled:** readiness는 두 손 중심 모두 assigned physical box의 oriented surface와 absolute signed 3D distance가 0.12m 미만인 최초 valid frame이다. 지정 손 목표점 거리나 contact-force 판정이 아니다. 완료는 shared box center가 shared goal과 3D 거리 0.20m 이내인 기존 group success다.

모든 event annotation은 control frame 해상도의 실측 label이며 sparse trajectory 사이를 보간해 만들지 않는다. prompt용 궤적은 30 control frames마다(약 1초) 샘플링하고 각 이벤트 직전/해당/직후 frame을 추가한다. initial state는 reset 직후 0초에서 기록한다. 최종 JSON은 최초 group success까지만 포함하며 원시 파일에는 이후를 포함한 전체 rollout이 남는다.

## 파일과 명령

원시 기록은 `teacher_runtime/output_etc/check/demo_probe_20261110/`, `teacher_runtime/output_etc/check/demo_diverse_20261110/`에 보관한다. 각 setting 폴더에는 `launch.json`, `recording.json`, `record_initial.pt`, `dense_trajectories.npz`, 기존 `initial.pt`·`trace.npz`·`result.json`, `run.log`, `exit.json`이 있다.

최종 calibration은 `output_etc/check/calibration_20261110/`에 저장한다. `demonstrations.json`은 실측 예제 10개 배열, `environment_spec.json`은 단위·event 관례·선별 한계, `selection_report.json`은 후보/성공 수, `verification.json`은 원시 데이터 재대조 결과다. 개별 예제 JSON도 함께 저장한다.

재수집은 새로운 `--tag`를 사용한다. 아래는 실행 형식이며 GPU 점유를 확인한 후 실행한다.

```bash
python3 collect_trajectories.py --gpu 7 --settings enable_sit \
  --envs 8 --seed 11 --tag another_unique_recording
```

내보내기·검증은 CPU에서 수행한다. torch/numpy가 설치된 환경을 쓴다.

```bash
python export_demonstrations.py \
  teacher_runtime/output_etc/check/demo_probe_20261110 \
  teacher_runtime/output_etc/check/demo_diverse_20261110 \
  --output output_etc/check/calibration_20261110
python verify_demonstrations.py output_etc/check/calibration_20261110
```

현재 API는 `environment_spec`과 `current_scene`만 받는다. 과거 기록을 입력에
추가하는 CLI 옵션은 제거했다. 위 데이터와 도구는 오프라인 분석용으로 보존한다.

현재 성공-only 데모는 모두 release=0의 공동 rollout이다. 출발 시각을 바꿨을 때의 물체 이동·에이전트 간섭 변화는 측정하지 않았으며 이 점은 environment_spec에도 명시한다.
