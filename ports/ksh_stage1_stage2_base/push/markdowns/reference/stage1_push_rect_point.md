# Stage1 PUSH 직육면체·중점 고정점

`stage1_push_rect_point.yaml`은 복사한 Stage1 구조에 XYZ가 독립인 직육면체와 양손 중점의 고정점 목표를 연결하는 별도 실험이다. 기존 `stage1_push`·`stage1_push_no_force`·원본 보관본은 유지한다. 실행 명령 전체는 [config](../config.md)의 마지막 항목을 따른다.

## 실제 설정

| 항목 | 값 |
| --- | --- |
| 사람/물체/목표 | 2명·실제 상자4개·목표2개; 8토큰·634D·물체당30D 유지 |
| 크기 | XYZ 각각1.20~1.50m, 0.05m 간격7값에서 독립 추출(343종) |
| 크기 추출 시점 | 물리 자산 생성 시 각 환경의 각 상자에 독립 적용; episode reset에서도 같은 물리 크기 유지 |
| 질량/밀도 | 모든 크기20kg, 밀도=20/(XYZ); 실제 shape 질량 검사 |
| 마찰 | 상자0.6·바닥1.0, 반발0 |
| 목표 위치 | 초기 상자 로컬+Y 방향으로1~5m 균등 추출; episode 중 고정, 목표Z는 각 상자 초기 중심 높이 |
| 손 중점 목표 | 초기 바닥에서0.65m, 초기 뒷면(-Y)의 중앙선. 로컬점 `(0,-Y/2,0.65-Z/2)`이며 상자 이동/회전에 따라 움직이는 재료점 |
| CONTACT 성공 | 양손 각각 유한 뒷면까지≤0.18m, 양손 중점의 목표XY 거리≤0.18m, 중점Z 오차≤0.03m |
| CONTACT 상태 | `exp(-10*max(중점XY거리, 양손면거리최대)^2 - (중점Z오차/0.04)^2)`; 예전 Z 무감점구간 제거 |
| CONTACT 진행/성공 가중치 | 이전 사용자 CONTACT의 접근 진행 보상과 성공 구조 유지, 각 state/progress/success 0.2 |
| 힘 gate | 없음. 양손 힘/상자 힘0.5N을 요구하지 않음 |
| 손 간격/개별 높이 | 고정 목표 없음. 양손의 중점이 기준이며 각 손이 같은 점에 모일 필요 없음 |
| AT/보상 | 가져온 Stage1의 AT 거리0.2m·Z1mm·보상/포화/네트워크 유지 |
| 학습/AMP | PUSH100%, 2048환경, loco/PUSH expert33/67, PUSH21모션 유지 |
| 기울기 | 새 성공/보상 제한을 추가하지 않음. 2°는 RSI/물리검사 기준이며 `push/box_tilt_deg`, `push/box_tilt_max_deg`, `push/box_upright_2deg` 진단 제공 |

`pushBox`가 이 전용 클래스의 물성 기준이다. 공통 `box.build`의 기본크기/테스트표는 이 클래스의 자산 생성에 사용하지 않는다. 기존 체크포인트는 새 크기·높이·RSI bank 계약과 다르므로 바로 resume/test하지 않는다. 새 bank는 SHA256을 YAML에 고정하며 내용이 바뀌면 로딩을 거부한다.

## RSI 선별

기존11개 후보를 새 중점 높이로 어깨 보정한다. 소스 양손 높이 차이는 가능한 범위에서 유지하고, 양손 중점만0.65m를 향하게 한다. 팔 길이·몸통·발 높이를 보존한다. 원본 AMP expert 자체는 변형하지 않는다. 초기 AMP history도 같은 보정을 적용한다.

GPU6에서343크기×11후보×2독립yaw/사람 = **7546개** 초기화를 평가했다. 초기/첫 물리 step의 중점Z≤3cm·중점XY/양손면거리≤18cm, 초기 관절 한계 위반 없음·손 이외 body point가 자기 상자 내부에 없음, 12 control step(0.4초) 동안 root≥0.65m·root drift≤0.45m·상자변위≤0.3m·기울기≤2°·발 지지 관측·reset없음을 요구했다.

모든 크기와 두 사람에서 통과한 소스 후보 **0/9/10의3개**만 `data/stage1_push_rect_point_bank.json`에 채택했다(모션0/19/20). 채택 후보2058개 초기화의 최대각0.0506° 미만, 첫 물리 step 중점Z 오차8.5mm 미만. interaction RSI50%·late70%/late구간30% 법칙은 유지하되 선택 가능한 자세 집합이 달라진다. 각 모션의 후보가1개여서 그 모션 안에서 late/non-late가 같은 후보를 선택할 수 있다.

검사 결과: `output_etc/check/stage1_push_rect_point_rsi/{audit.json,cases.csv,stage1_push_rect_point_bank.json}`. 이는 짧은 초기화 검사이며 1~5m 장거리에서 안 기울고 잘 미는 정책을 보장하지 않는다. [앞선 질량/높이 분석](push_rect_stability.md)과 구분한다.

재검사는 사본 루트에서 실행한다. 출력은 새 경로로 지정한다. 재검사는 후보 bank를 결과 폴더에 생성하며 활성 설정이나 bank를 자동 덮어쓰지 않는다.

```bash
TOKENHSI_GPU=6 bash -c '
source tokenhsi/scripts/multi_agent/runtime_env.sh
export PYTHONPATH="$TOKENHSI_ROOT/tokenhsi:$TOKENHSI_ROOT" OMP_NUM_THREADS=1
python tokenhsi/tests/check_stage1_push_rect_rsi.py \
  --task HumanoidMAStage1Push \
  --cfg_env tokenhsi/data/cfg/multi_agent/stage1_push_rect_point.yaml \
  --cfg_train tokenhsi/data/cfg/train/rlg/stage1_push_rect_point.yaml \
  --motion_file tokenhsi/data/dataset_stage1_push.yaml \
  --num_agents 2 --num_envs 3773 --num_objects 4 --task_graph random_scenario \
  --test --eval --headless --no_video \
  --output_path output_etc/check/stage1_push_rect_point_rsi_repeat
'
```

## 실행 확인

CPU 보상/계약 회귀17개 통과. GPU6·64환경4096 agent reset에서4상자 물성·634D 관측 원본 동일·180개 XYZ 조합·각축1.2~1.5m·타깃높이0.65m·부분reset·고정목표·12step finite를 확인했다(`output_etc/check/stage1_push_rect_point_reset/`). 장기학습이나 GUI 검증과 구분한다.

GPU6·2048환경 짧은 학습2epoch/frame262144·checkpoint저장 rc0,669tensorfinite/모델165tensor를 확인했다(`output_etc/check/stage1_push_rect_point_train/`). 전용 test wrapper에서2환경·12step×2회·loco/PUSH50/50 초기화로checkpoint재로딩 rc0(`output_etc/check/stage1_push_rect_point_eval/`). 이는 실행 연결 검증이며 정책의 성공률 검증이 아니다. 본학습은 시작하지 않았다.
