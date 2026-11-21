# Follow-cap 5cm 학습 비교

세 실행 설정은 `stage2_sensor_900_bridge_before_fixed_layout_cache`에서 파생한다. [실행 명령](../config.md)은 동일 이름의 train/test/VNC wrapper를 사용한다.

| 설정 이름의 `stage2_sensor_900_bridge_` 뒤 부분 | 선행 중간 목표 | 보상 |
| --- | --- | --- |
| `follow_cap` | 없음 | 기존 전체 보상 |
| `follow_cap_waypoint_single_pass` | 있음 | 기존 전체 보상 |
| `follow_cap_waypoint_progress_single_pass` | 있음 | 횡단 선행의 AT progress·중앙 보조 보상만 단계별 progress로 교체 |

기존 `_waypoint`·`_waypoint_progress`는 두 번 계산하던 실행·체크포인트 재현용으로 보존한다. 새 가상목표 학습은 `_single_pass` 설정을 사용한다.

## 공통 조건

- 원본 `/home/hwanhee/ksh/stage1_2_base_teacher/output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth`에서 새 전이. actor encoder·coordination·sigma와 입력 RMS 동결, 기존 센서 학습 방식 유지. critic·행동 head·sensor·discriminator는 기존대로 학습한다.
- 4명·1024환경·8개 관측 물체 토큰·900step, 학습률 2e-5. 배치 캐시는 환경당 36개, bucket당 2개, refresh 0, 거리 alpha 1이다.
- 같은 플랫폼 BEFORE 20%, 횡단 BEFORE 80%. 같은 플랫폼에서는 세 설정 모두 선행 중간 목표·단계별 보상을 사용하지 않는다. 후행 망각 방지용 원래 과제를 유지한다.
- present/near/insert는 20/40/40%. 기존 사람·상자·최종 목표 샘플러를 유지한다. 원래 present 정렬과 near 50% 정렬은 남으며, 토이 viewer의 강제 횡단 중심선 배치를 사용하지 않는다.
- 동적 다리 본체와 별도 고정 5mm 상면 캡. 캡을 본체 상면보다 5cm 높여 매 물리 step 본체 위치·회전을 추종한다. 본체↔캡 충돌 제외, 캡에 가한 힘은 본체로 전달되지 않는다. GPU 물리에서 한 스텝 추종 지연이 있다. 본체 밀도 3·마찰 0.4, 캡 마찰 1.0 유지.
- 캡은 별도 관측 물체 토큰이 아니다. 센서·상자 크기·GTA의 기하 기준은 원래 본체다. 캡 외형은 기존 청록색 표시를 유지한다.
- 후행의 BEFORE 가상 받침 상자와 실제 최종 목표는 원래 위치를 유지한다. 후행 대기 반경 여유는 기존 1.5m다.

## 선행 중간 목표

횡단 선행의 GTA Target 위치를 현재 움직이는 다리 중심으로 옮긴다. Z는 회전된 본체 상면에 운반 상자 높이의 절반을 더한 위치다. 선행 운반 상자 중심 XY가 `R = sqrt(sx²+sy²)/2 + 1.0m` 이내에 들어오면 원래 최종 목표로 복귀하고, 에피소드 중 다시 중간 목표로 돌아가지 않는다. 처음부터 반경 안이면 즉시 최종 목표를 사용한다.

원본 scene에서 후행 가상 받침 상자를 **원래 최종 목표 위치**에 먼저 배치한다. 그 뒤 활성 환경의 선행 GTA Target XYZ만 다리 중앙으로 바꾸고, 완성된 scene을 actor·critic 각각 encoder → coordination으로 **한 번씩** 계산한다. 네 사람 모두 이 계산의 출력을 사용한다. 다른 사람의 목표 위치는 유지하지만 coordination에서 바뀐 선행 정보를 참고하므로 행동은 달라질 수 있다. 실제 목표 tensor·보상 기하는 바꾸지 않는다. `follow_cap_waypoint_single_pass`의 기존 AT 보상도 원래 최종 목표를 기준으로 한다.

`leader_waypoint.policy_input: shared_gta_single_pass_v1`을 환경 계약에 저장한다. 새 네트워크 builder는 `amp_multi_agent_bridge_waypoint_single_pass`다. 후행 alias를 해결한 뒤 선행 목표를 바꾸며, encoder 직전에 alias를 다시 계산해 다리 중앙으로 덮어쓰지 않는다. 기존 두 번 계산하는 경로와 가중치 형태는 같지만 행동 규칙이 다르므로 서로 resume/test하지 못하도록 구분한다.

PPO 관측 끝에 `[선행 인덱스, 중간 목표 활성 여부, 환경 좌표계 목표 XYZ]` 5개를 저장한다. 정규화 없이 rollout·minibatch·추론에서 같은 입력으로 사용하며 신규 학습 파라미터는 추가하지 않는다. 관측 크기는 10493이다. 전환 latch와 보상 기준값은 에피소드 상태이며, 부분 리셋·캐시 복원 시 재설정한다.

## 단계별 progress

기존 횡단 선행의 최종 AT progress 최대 0.2와 다리 중앙 보조 progress 최대 0.2를 교체한다. HOLDING·state·success·후행·push·페널티와 최종 성공 판정은 유지한다.

- 전환 전: `r = 0.2 / (1 + max(d_bridge - R, 0) / 1m)`.
- 전환 순간: 중간 progress를 1로 고정하고, 원래 최종 AT progress `p*`를 저장한다. 새 최종 progress는 0에서 시작하므로 보상은 0.2다.
- 전환 후: `r = 0.2 + 0.2 × clip((p_goal - p*) / (1 - p*), 0, 1)`.

`p_goal`은 기존 성공 포화를 포함한 AT progress다. 이미 `p* ≈ 1`인 경우 0으로 나누지 않으며 현재 progress가 1이면 최종 항도 1이다. 최종 목표에서 멀어지면 새 최종 항은 줄어들지만 중간 항 0.2는 유지한다. 매 step 지급하는 거리 보상이며 거리 차분 보상이 아니다. 합의 최대는 기존과 같은 0.4다. 다른 보상·충돌 페널티에 의한 전체 보상의 하락까지 막지는 않는다.

## 저장과 추론

세 실험의 config·run·출력 이름을 분리한다. 체크포인트의 환경 계약에 캡·중간 목표·보상 모드를 포함하므로 서로 다른 설정의 재개를 허용하지 않는다. 새 학습은 원본 2k 전이, resume/test/VNC는 해당 실험에서 저장한 센서 체크포인트를 사용한다. 단일 계산 수정본은 원본 V4 2k에서 새 학습한다. 기존 GPU5/7 프로세스의 동작은 파일 수정만으로 바뀌지 않는다. 단일 계산 검증 산출물은 `output_etc/check/bridge_single_pass_20261118/`, 이전 구현 검증은 `output_etc/check/bridge_follow_cap_training_20261118/`에 둔다.

## Insert 횡단 평가

현재 평가와 eval VNC는 `coupled_assignment=random`을 사용한다. 저장된 초기 fixture도 물리 상태 복원 후 담당 영역·개별 목표를 학습 배치 기준으로 복원한다. 수정 후 결과는 각 평가 디렉터리의 `training_assignment/`에 저장하고, 기존 상위 디렉터리의 ordered 결과·fixture는 보존한다.

5000 평가는 [결과표](../../output_etc/check/bridge_crossing_eval_5000_contact_20261118/index.html)와 [접촉·대기 집계](../../output_etc/check/bridge_crossing_eval_5000_contact_20261118/contact_analysis.json)에 저장한다. 3500·4500과 동일 초기 장면이며 없음/단계별 모델의 전체 씬 ever SR은 37.83/59.70%, final SR은 34.77/55.27%다.

4500 추가 평가는 `output_etc/check/bridge_crossing_eval_4500_20261118/`에 저장했다. GPU당 512환경×3seed 병렬·전체 메모리 85000MiB 상한으로 실행하며, 아래 3500 평가의 초기 상태를 그대로 복원했다. 모든 실행의 초기 tensor 75개와 전체 관측이 3500과 exact다. 없음/단계별 모델의 전체 씬 ever SR은 50.72/58.27%, final SR은 47.85/54.49%이며 3500의 ever SR 57.49/66.08%보다 낮았다. [행동별 결과](../../output_etc/check/bridge_crossing_eval_4500_20261118/index.html)·[3500 대비](../../output_etc/check/bridge_crossing_eval_4500_20261118/versus_3500.csv)·[초기 상태 검증](../../output_etc/check/bridge_crossing_eval_4500_20261118/cross_version_verification.json).

`follow_cap`과 `follow_cap_waypoint_progress_single_pass`의 3500 checkpoint를 각각 GPU5·7에서 평가한다. 모델당 512환경×seed 0·1·2, 환경당 첫 에피소드 900step이며 결정론적 평균 행동을 사용한다. `insert`·횡단 필요 샘플을 각각 100%로 제한하고 기존 배치 제약·밀기 2명·후행 세 과제·cap 5cm를 유지한다. 평가 전용 프로세스에서 같은 플랫폼 배치를 제외하며 학습 YAML과 실행 중 학습은 바꾸지 않는다.

학습 cache 대신 새 장면을 생성한다. GPU5 초기 상태를 파일에 저장하고 GPU7에 복원하며, 물체·플랫폼 asset 크기와 복원한 상태 및 가상목표를 제외한 관측을 비교한다. 쓰러진 사람은 기존 `formal_eval.freeze_fallen` 방식으로 모터를 끄고 나머지 사람은 계속 실행한다. 자동 reset은 하지 않는다.

- 사람 횡단: 살아 있는 선행 pelvis의 XY가 목적 플랫폼 영역 안이고 Z가 플랫폼 상면+0.6m보다 높다.
- 상자 횡단: 선행이 살아 있고 운반 상자 중심 XY가 목적 플랫폼 안이며, 회전을 고려한 상자 바닥이 플랫폼 상면−0.1m 이상이다. 사람·상자의 동시 도착도 집계한다.
- 행동 SR: 기존 BEFORE 선행 조건을 포함한 native 필수 목표 성공이며, 쓰러진 목표 소유자는 제외한다. SIT·CLIMB·ON_TOP은 해당 과제 장면 수, PUSH는 담당 사람 수가 분모다.
- Enabling SR: 선행과 후행의 목표 동시 성공. 전체 씬 SR은 여기에 두 밀기 담당의 필수 목표도 동시에 성공해야 한다. 후행 prerequisite 활성화율은 별도다.
- 모든 성공률은 한 번이라도 성공(`ever`)과 900step 종료 시 성공(`final`)을 구분한다. 독립적인 기존 point/region 공간 채점도 각 seed JSON에 보조 지표로 저장한다. 서로 다른 보상 총합은 비교하지 않는다.

원본 초기 상태·trajectory·seed별 결과 및 실행/집계 스크립트는 `output_etc/check/bridge_crossing_eval_3500_20261118/`에 둔다. 이 조건은 insert 전용 평가이므로 present/near를 섞은 학습 중 지표와 직접 비교하지 않는다.

[결과표](../../output_etc/check/bridge_crossing_eval_3500_20261118/index.html)·[비교 CSV](../../output_etc/check/bridge_crossing_eval_3500_20261118/comparison.csv)·[장면별 CSV](../../output_etc/check/bridge_crossing_eval_3500_20261118/episodes.csv)·[전체 JSON](../../output_etc/check/bridge_crossing_eval_3500_20261118/summary.json).

## 6000 최종 배치 후 좌우 정렬 평가

[결과표](../../output_etc/check/bridge_crossing_eval_6000_ordered_20261118/index.html)·[JSON](../../output_etc/check/bridge_crossing_eval_6000_ordered_20261118/summary.json). 두 기존 follow-cap 모델의 6k를 동일 512환경×3seed, insert·횡단 100%, cap 5cm, 900step으로 평가했다. 모든 사람/상자의 배치를 복원한 뒤 실제 상자 좌표계로 담당 영역과 개인 목표를 정렬했다. 새 rear90 모델 평가는 아니다. 세 시드에서 물리 상태와 공통 관측이 두 모델 간 일치하며, 초기 상태는 기존 ordered fixture와도 exact다. `training_assignment/`의 반대편 배정 결과와 구분한다.

가상목표 없음/단계별 progress 모델의 전체 씬 ever SR은 924/1536(60.16%), 944/1536(61.46%), final SR은 55.73/58.66%다. 사람+운반 상자 동시 횡단 ever는 74.22/75.20%, 선행 낙상은 22.01/25.72%다. 행동별 SR·시드별 분산·paired 결과는 JSON에 저장했다. GPU5/7 최대 점유는 76491/79520MiB다.

가상목표 없는 6k를 기존 학습 담당 배정으로 다시 평가한 [비교표](../../output_etc/check/bridge_crossing_eval_6000_training_assignment_20261118/index.html)에서 전체 씬 ever/final SR은 75.46/70.77%였다. 같은 checkpoint·같은 1536개 물리 장면이며 담당 영역과 개인 목표만 변경했다. 좌우 정렬 대비 ever +15.30%p, 사람+상자 횡단 74.22→91.60%, 공동밀기 개인 81.28→99.48%, 선행 낙상 22.01→9.51%다. 기존 학습 배정 결과이며 올바른 좌우 배정으로 학습한 새 rear90의 결과는 아니다.

## 5000 발 접촉·대기 진단

평가 프로세스에만 관찰 함수를 연결해 초기 상태 포함 물리 1801프레임(60Hz), control 900프레임(30Hz)을 기록한다. 캡 동기화 직전에 실제 발 자세·순접촉력과 그 물리 step에 사용한 캡 자세를 저장한다. 진단은 상태·난수를 변경하지 않는다. 물리 기록의 control 시점 root/bridge 자세는 기존 trajectory와 exact 비교한다.

- 접근: 선행 pelvis가 출발 플랫폼 끝 0.8m 안쪽으로 진입.
- 발 내딛기: 실제 발 mesh의 최저점+2.5cm 이내 정점들 중심이 플랫폼 끝을 5cm 넘어 틈 안에 위치. 발 높이는 플랫폼 상면−15cm 이상, pelvis 높이는 상면+40cm 이상이며 낙상 전이다. 첫 발끝 돌출이나 실제 접촉 시각과는 다르다.
- 캡 접촉 추정: 틈 안의 발바닥 정점이 캡 XY 영역(2cm 허용)과 상면 ±3cm 안에 있고 캡 법선 방향 발 순접촉력이 10N 초과인 상태가 2물리 프레임 연속. 6프레임 연속도 별도로 저장한다. 순접촉력은 물체 쌍별 힘이 아니므로 캡과의 접촉을 확정하는 판정은 아니다.
- 밀기 완료: native 기준인 목표 중심 3D 거리 <20cm, 상대 yaw의 절댓값 sin <0.1. 추가 안정 조건은 완료·속도 <0.1m/s·기울기 <5°를 직전 0.5초 연속 만족. 진행률은 삽입축에서 초기 남은 거리 대비 이동분이며 0~1로 자르지 않는다.
- 대기 후보: 상자를 바닥 기준 15cm 이상 들고 native HOLDING을 만족하며, 플랫폼 끝 1m 이내·발 내딛기 전·다리 미완료 상태에서 1초 연속 관찰. 그동안 pelvis 순이동 ≤25cm, XY 범위 대각 ≤35cm. 의도적 대기나 정체를 구분하지 못한다. 대기 종료 후 2초 안에 다리 최초 완료, 그 뒤 3초 안에 발 내딛기를 한 경우도 별도 집계한다.

각 시점의 거리·진행률·완료·안정 여부·속도·기울기는 [장면별 CSV](../../output_etc/check/bridge_crossing_eval_5000_contact_20261118/contact_episodes.csv)에 남긴다. 낙상 위치는 native 낙상 검출 순간 pelvis XY이며 접촉 위치와 동일하다고 해석하지 않는다.

[5000 타이밍 결과표](../../output_etc/check/bridge_crossing_eval_5000_contact_20261118/contact_report.html): 없음/단계별 모델에서 미완료 내딛기 뒤 횡단 전 틈 낙상은 563/600·299/387, 완료 내딛기에서는 28/747·65/1133이다. 추가 안정 조건을 만족하면 11/628·29/899이다. 완료에서 내딛고 낙상까지 모든 30Hz 표본에서 완료를 유지한 낙상도 25·61개 남아 있어 타이밍만으로 모든 실패를 설명하지 못한다. 발 내딛기 때 미완료였으나 낙상 시 완료인 11·81개는 완료 상태에서 출발한 실패로 분류하지 않는다. 대기 후보는 둘 다 0개이며 기준보다 짧은 멈춤이나 감속은 이 결과에 포함하지 않는다.

## rear90 정렬/랜덤 학습 1k 비교

[결과표](../../output_etc/check/bridge_rear90_eval_1000_ordered_20261118/index.html). 두 모델 모두 1k checkpoint, 각 512환경×3seed·900step·insert/횡단 100%·cap5cm·가상목표 없음이다. 공동밀기 사람은 후방±45°·상자 외곽0.5~5m로 새로 배치했다. 물리 상태·담당 영역·전체 관측은 모델 간 동일하며 평가 배정은 모두 최종 좌우 정렬이다. 이전 6k의 옛 사람 배치와 구분한다.

rear90/random 순서로 전체 씬 ever/final SR은 61.33/56.84%, 61.00/55.99%다. 사람+상자 횡단 ever는 78.91/80.08%, 공동밀기 개인 ever는96.22/97.23%, 선행 낙상은21.22/19.08%다. 성공률 차이가 작으며 우열을 단정하지 않는다.
