# 기존 Stage1 구조에 추가한 PUSH 전용 설정

힘 조건만 제거한 비교 설정은 `stage1_push_no_force.yaml`이다. `pushTask.revision=2`, `require_contact_force=false`이며 손/상자 힘 임계값은 삭제했다. 기존 양손 지정면거리≤0.18m·중점XY≤0.18m·높이0.585~0.650m 조건과 상태/접근 보상·AT·물성·RSI·AMP·관측은 유지한다. 따라서 실제 반력이 없어도 기하상 근접하면 CONTACT를 인정한다. 기존 힘 조건 버전 `stage1_push`와 config·학습 YAML·train/test/VNC·output을 분리했고 checkpoint는 서로 직접resume/test할 수 없다. 실행은 [config.md](../config.md)의 `stage1_push_no_force` 명령을 따른다.

`stage1_push.yaml`·`HumanoidMAStage1Push`는 가져온 `stage1_v2_fixed`의 네트워크와 AT를 유지하는 별도 실험이다. 기존 HOLDING/SIT/CLIMB/운반/ON_TOP config와 구현은 계속 남겨두며 이 실험의 과제 샘플링만 PUSH100%다. 수정은 이 `push/` 사본에 한정하며 `../upstream/`은 읽기 전용 원본이다.

| 항목 | 설정 |
| --- | --- |
| 학습 | 2명·2048환경·600step |
| 관측 | 사람2 + 물체4 + 목표2 = 8토큰, 총634D |
| 상자 입력 | 각30D·중심pose/GTA 유지, 접촉점 입력으로 치환하지 않음 |
| 네트워크 | 원본 `stage1_v2_fixed` 학습 YAML의 network와 동일 |
| 상자 | 4개 모두1.3³m·20kg·마찰0.6, 담당0/1·미할당2/3 모두 관측 |
| 목표 | 각 담당 상자 초기중심에서 초기 로컬+Y 방향으로1~5m 균등·Z0.65m, 에피소드 중 고정 |
| 초기 작업영역 | 네 상자를 중심(±8,±8)m에±0.5m jitter로 분리·초기yaw무작위, envSpacing24 |
| 사람 접근 시작 | 상자 뒤1~2m, 초기 모션 body penetration은 해당 범위 안에서 재추첨 |
| RSI | 50% loco / 50% 기존 높이선별11자세·후반요청70%, 최초목표까지 진행률 적용 |
| RSI 잔여거리 | 모든 donor후보에서 최소0.21m 이상 확인, 이미 AT성공이면 안 됨; 후보/목표 분포를 몰래 변경하지 않고 실패 시 중단 |
| AMP | 원본25개 모션(loco4/PUSH21), expert33/67%; 기존 Stage1의132D/frame·3-family 형식을 유지하고 별도 PUSH 실험은 manipulation family 사용 |

## CONTACT와 AT

CONTACT는 사용자의 기존 PUSH 코드에서 복사한 `push_point.py`의 실제 함수로 계산한다. 초기 목표 반대면(로컬−Y) 중앙에 고정한 material point를 사용하며, 중점 높이는0.585~0.650m 무감점, 밖거리 sigma0.04m, 중점XY·양손 면거리0.18m, 각손 법선반력0.5N·상자 안쪽힘0.5N을 요구한다. 각 제어step의 물리tick 접촉을 OR 누적한다. 접근 점수는 기존 PUSH처럼 `1/(1+max(root_xy_distance-0.5,0))`다.

AT는 원본 `edge_ontop_reward.py`·`interaction_own_success`를 그대로 사용한다. 상태는 `exp(-10*distance_xyz²)`, 성공은 거리0.2m에 대응하는 점수와 Z오차≤0.001m, 접근은XY거리·buffer0.1m·sigma1이다. state/progress/success 가중치는각0.2, 자기edge합산·공유0이며, 현재 AT 성공 시 같은 사람의 접촉 쪽 보상을 포화한다. power와CPA는원본,상자속도페널티는기존PUSH처럼비활성이다.

그래프 토폴로지와 task packet/category는 원본의 HOLDING+AT 두edge 및 carry_at 슬롯을 재사용한다. CONTACT의 점수·힘 성공만 별도 클래스에서 바꾼다. 원본 `HumanoidMACarry`의 HOLDING을 전역 교체하거나 기존 task embedding을 삭제한 것이 아니다. 다중과제를 한 정책에서 함께 학습하는 설정은 이번 구현 범위가 아니다.

## 데이터와 checkpoint

`dataset_stage1_push.yaml`·전용bank·`door_reference.py`/`push_point.py`/`open_push_rsi.py`를 이 사본에 분리했다. 모션 자체는 읽기용 링크이며 bank의21개 모션SHA256을 검사한다. 기존 높이선별11자세를 동일하게 재사용했다. 새4상자 장면의 reset/짧은물리는검사하지만 전후보PD screen을 다시 수행한 것은 아니다.

원본Stage1 checkpoint와 기존루트PUSH checkpoint는 직접resume/test하지 않는다. 새PUSH계약이 맞는 저장본만 허용한다. Stage2 공동PUSH·전이 연결은 아직 구현하지 않았으며 기존Stage2 config는 그대로다.

실행 명령은 [config.md](../config.md)에 있다. 이 사본 루트에서 실행하거나 상위레포에서 `bash ports/ksh_stage1_stage2_base/push/tokenhsi/scripts/multi_agent/stage1_push_train.sh`를 호출하면 wrapper가 이 사본으로 이동한다. 새장기학습은 자동 실행하지 않는다.

힘 조건 버전 검증 경로: `output_etc/check/stage1_push_reset/audit.json`, `output_etc/check/stage1_push_train_final/`, `output_etc/check/stage1_push_eval_final/`. 힘 제거 버전은 CPU44개와 `output_etc/check/stage1_push_no_force_reset/audit.json`의64환경·부분reset·12step 물리 검사로 확인했다. 힘 제거 버전의 PPO본학습·checkpoint재로딩·GUI·장기성공률은 미실행이다.
