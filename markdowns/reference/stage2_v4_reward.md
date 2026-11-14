# V4 teacher 보상으로 Stage2 PPO 학습

`stage2_900_v4_reward`는 `stage2_900`을 기준으로 Stage1 PUSH26k에서 새로 전이하는 실행용 PPO/AMP 컨피그다. Distillation이나 teacher 정책 추론은 사용하지 않는다. 네트워크·장면 샘플링·물리·학습률2e-5·task/AMP 가중치0.5/0.5는 `stage2_900`을 유지한다. 기본4명·1024환경·8토큰·900step이며 wrapper 기본 GPU는5다. [실행 명령](../config.md).

## 보상 출처

V4 실행의 `output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/teachers.json`에 연결된 두 teacher의 저장 `relation_config.yaml`과 원본 코드를 기준으로 한다.

| 담당 | 원본 checkpoint |
| --- | --- |
| Enable 선행·후행과 모든 solo | `/home/hwanhee/ksh/stage_1_2_base_1031/output/stage2_before_total_at1mm/Stage2BeforeTotalAt1mm_08-14-00-22/nn/Stage2BeforeTotalAt1mm_00034000.pth` |
| 공동 PUSH 그룹원 | `/home/hwanhee/ksh/stage1_2_base_jointpush/output/stage2_coupled_contact/Stage2CoupledContact_10-00-20-39/nn/Stage2CoupledContact_00030000.pth` |

- Enable/solo AT의 성공·선행 gate 해제는 Z20cm이며, 보상 포화·성공 보너스와 paired placement 포화는 Z1mm를 요구한다. 공동 PUSH AT 성공은 기존 Z1mm다. HOLDING 등의 다른 성공 조건은 유지한다.
- Enable 후행 대기 감점은 지지 물체 반대각 길이+1.5m 안에 들어온 뒤 선행 AT에 한 번 성공할 때까지 사람의 XYZ 속도 norm을 계수1로 감점한다. ON_TOP의 진입 거리는 운반 물체 XY, SIT/CLIMB는 사람 XY를 사용한다. 해제는 latch되며 환경 reset 시 초기화한다.
- 공동 PUSH 접촉 준비 전 이동 감점은 Coupled30k의 원본 `motion_penalty` 수식을 사용한다. 그룹원 전원이 두 손 근접 상태가 되기 전에 상자 XY 속도가0.05m/s를 넘으면, 이동 방향의 양의 손 반력 기여율에 따라 `-0.5 × (속도-0.05) × 기여율`을 적용한다. 힘 분모 epsilon은1e-6이다. 물리 substep별 감점을 평균하며 solo에는 적용하지 않는다.
- 두 teacher에 공통인 CONTACT·관계 보상 가중치·power/box speed 등은 기존 통합 환경과 같아 유지한다. Teacher별 배치 거리나 정책 가중치를 가져오는 설정은 아니다.

## 구현과 저장

환경 YAML의 `stage2.reward_source: v4_teachers`와 `before.reward_at_z_tolerance`, `before.waiting_penalty`, `stage2JointPush.premature_motion_penalty`가 보상 분기를 지정한다. 이전 컨피그는 이 옵션을 추가하지 않아 기존 동작을 유지한다.

Stage1 전이 경로는 `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`, 본학습 출력은 `output/stage2_900_v4_reward/Stage2_900V4Reward_*`다. 전용 train/test/VNC/eval wrapper가 있으며, 새 checkpoint의 저장 환경을 사용해 추론·resume한다. 다른 보상 설정 checkpoint와의 resume 혼합은 계약 검사에서 거절한다.
