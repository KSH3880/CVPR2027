## 현재 Partial 구현

현재 RawCuboid Partial의 코드·보상·실행 명령은 `../../jointpush/partial/`, `../../jointpush/scripts/partial_*`와 [안내](../../jointpush/README.md)를 따른다. 새 partial은 여기의 KSH/기존 Koo port에 의존하지 않는다. 아래 경로는 기존 viewer·분석용이다.

## KSH Depth180 9000 Partial viewer

- `scripts/ksh_partition_prepare.py`: 원본 체크포인트의 saved config와 현재 partial 설정으로 runs 내부 설정 생성, Y1.8고정, source 읽기 전용 연결.
- `scripts/ksh_partition_scene.py`, `ksh_partition_helpers.py`: KSH FiveTask 표현을 유지하는 partial 장면·배정·보상 adapter.
- `scripts/ksh_partition_run.py`: source bootstrap, Stage1 transfer, test/eval 강제, 추론 및 가중치 불변 검증.
- `scripts/ksh_partition_view.sh`, `ksh_partition_view_run.sh`: GPU6 noVNC 및 test/eval 진입점. 학습 launcher 아님.

## Native4 attention 진단

- `scripts/stage1_native2_attention.sh`: 원래 Koo2인 OPEN/PUSH 환경에서 actor 층별/head별 attention PNG와 원본 수치를 저장한다.
- `scripts/stage1_native4_attention.sh`: 4명·문4·상자4 OPEN/PUSH 혼합 환경의16×16 actor attention PNG 저장.
- `ports/koo_push/tokenhsi/scripts/multi_agent/native4_zeroshot/mixed4.py`: attention 수집에 opt-in된 4인 혼합 과제의 배정/그래프 및 원래2인 보상 함수를 재사용하는 agent쌍별 실행.
- `ports/koo_push/tokenhsi/scripts/multi_agent/native4_zeroshot/attention_capture.py`: 실제 QKV hook과 GTA/bias를 사용해 softmax를 수집하고 토큰 순열을 복원한다. PNG·NPZ·상태/배정 메타데이터를 저장한다.
- `ports/koo_push/tokenhsi/tests/test_attention_capture.py`: softmax 일치 및 GTA head/token축 정렬 검사.

- `scripts/stage1_partition_box_ablation.sh`: 문 없는4인 Partition 환경에서 Stage1 원본으로 작업토큰4개를 유지하고 일반상자4→0개를 순차 축소하는 zero-shot 낙상 진단.
- `ports/koo_push/tokenhsi/scripts/multi_agent/native4_zeroshot/partition_stage1_setup.py`: Partition 환경과 원래 source network 설정·checkpoint 해시를 별도 출력에 기록한다.
- `ports/koo_push/tokenhsi/scripts/multi_agent/native4_zeroshot/partition_stage1_launch.py`: 의도된 환경 개입을 검증하며 원래 Stage1을 head 확장 없이 strict 복원하고 공통 진단 루프에 연결한다.

- `scripts/stage1_native2_no_doors_ablation.sh`: 원래 2인/문2/상자2 환경에서 actor attention8→6토큰 비교. 설정 복원은 공통 setup의 `--agents 2`를 사용한다.
- `ports/koo_push/tokenhsi/scripts/multi_agent/native4_zeroshot/native2_launch.py`: native4 count/graph/reward 확장 없이 원래 HumanoidMAOpenPush에서 공통 낙상/가중치 불변 진단을 실행한다.

- `scripts/stage1_native4_attention_ablation.sh`: 동일 초기 물리 상태의 full/pair(기본), none_zero, hide_humans, hide_doors, hide_boxes, hide_goals, hide_boxes_goals, hide_objects_goals, hide_box_first, hide_box_second, remove_doors 조건을 기준/full과 순차 실행하고 결과를 비교한다.
- `ports/koo_push/tokenhsi/scripts/multi_agent/native4_zeroshot/attention_probe.py`: 물리 장면별 actor mask 또는 NONE bias만 0으로 만드는 개입, 동일 관측 action 비교, 상태/가중치 해시와 첫 낙상 기록.
- `ports/koo_push/tokenhsi/scripts/multi_agent/native4_zeroshot/launch.py`: `NATIVE4_ATTENTION_SCOPE=full|pair`가 지정된 headless 진단에서만 위 실행 루프로 연결한다.
- `ports/koo_push/tokenhsi/tests/test_native4_attention_probe.py`: 독립 물리 배정과 토큰 순열 아래 모든 actor 층의 mask 범위를 검사한다.

## 서버·로컬 경로 처리

- `ports/koo_push/tokenhsi/utils/partition_push_paths.py`: CLI/환경변수/기기별 YAML/기본 후보의 Stage1 경로 해석. 학습 CLI와 checkpoint 모델 테스트가 공유한다.
- `paths.local.example.yaml`: 버전 관리하는 설정 예시. 복사한 `paths.local.yaml`은 Git 제외.
- `ports/koo_push/tokenhsi/tests/test_partition_push_paths.py`: 로컬·서버 후보 우선순위, YAML 상대경로, 명시 경로 실패, 디렉터리 최신/모호한 run 검증.

## 로컬 Partition Push 모션 데이터 (2026-10-19)

- `scripts/export_partition_reference_videos.sh`: 학습 저장 환경에서 RSI/AMP 원본 모션의 1080p MP4를 생성하는 진입점.
- `ports/koo_push/tokenhsi/scripts/multi_agent/export_partition_reference_videos.py`: 학습 MotionLib·humanoid asset을 재사용한 kinematic 재생, 프레임별 적용 포즈 검사, 개별 영상/합본/원본 해시 목록 생성.

- `ports/koo_push/tokenhsi/utils/partition_push.py`: 그룹 샘플링·구역 관측·360° 위치/heading 샘플링과 최소 사람 간격 검사.

- `scripts/partition_push_view.sh`: 로컬 화면의 Stage2 checkpoint viewer. 기본 1환경, Isaac Gym 기본 viewer와 기존 test 복원 경로를 사용한다.
- `motiondata/dataset_bones_push180/`: 변환된 Push180 모션 19개와 변환 보고서. 실행 포트의 `ports/koo_push/tokenhsi/data/dataset_bones_push180`에서 상대 심링크로 참조한다.
- `motiondata/README.md`: 데이터 위치·목록·사용 범위.
- `ports/koo_push/tokenhsi/scripts/multi_agent/partition_push_launch.py`: Isaac Gym 선행 import와 RTX 5090 TorchScript 호환 실행.
- Stage1 checkpoint는 `output/OpenPushPush180ContactAtWarmstart_00029000.pth`에 유지한다.

## 4명·100kg 추론 (2026-10-16)

`scripts/joint_push_four_agent_100kg_{test,view}.sh`: 4명/3×1.5×1.5m/100kg/GPU6 전용 preview 진입점. `open_push_spec.py`는 3·4명 shared graph의 인덱스를 인원으로 계산한다.

## 랜덤 모델 3명 추론 (2026-10-16)

`scripts/joint_push_random_shape_three_agent_{test,view}.sh`: GPU6 기본, 기존 랜덤 모델을 3명으로 추론. `joint_push_spawn.py`는 전원 쌍별 간격을 검사한다.

## 원본 Push Stage1 설명 추가 (2026-10-15)

[JOINT_PUSH_SETTINGS.md](JOINT_PUSH_SETTINGS.md)의15절에실제source29000checkpoint기준OPEN/PUSH원본환경·1.3m/20kg·독립task·warmstart/RSI·contact+AT·embedding/AMP·Stage2차이를추가했다. 평가결과없음.

## GPU0 랜덤 형상·질량 학습 (2026-10-15)

[JOINT_PUSH_RANDOM_SHAPE_SETTINGS.md](JOINT_PUSH_RANDOM_SHAPE_SETTINGS.md): 폭X2–3m/깊이Y1–2.6m/높이Z1–1.5m, 질량40–60kg 독립 격자 샘플링. 실제asset/bbox/reset/손간격/spawn반경연결. GPU0/2048환경/새Stage2,tmux `jointpush_randomshape_gpu0`. `bash stage2_jointpush/scripts/joint_push_random_shape_train.sh`,test/view도같은prefix. 기존GPU6학습유지.

## Joint Push 상세 설정 문서 (2026-10-15)

[JOINT_PUSH_SETTINGS.md](JOINT_PUSH_SETTINGS.md):현재60kg·360°시작·직선·softgate환경/초기화/모델/보상/AMP/PPO/실행법을정리한결과없는설정문서.

## 12000iter60kg/40kg대응평가 (2026-10-15)

동일60kg학습정책/seed3×64=192episode/600step.60kg엄격ever성공0%,40kg49.48%(95/192);종료시유지0%/23.96%(46/192).평균잔여거리2.583/0.645m,상자순변위1.076/3.086m.초기상태정확일치·공유상자질량만개입·미사용상자60kg유지·공식metrics대조통과. [기록](../../runs/jointpush/mass_comparison_20261015_final/RESULTS.md).별도40kg학습모델과의비교가아니다.

## 질량 진단 적용 시점 보완 (2026-10-15)

40kg진단은GPU sim 준비 전 asset밀도로적용한다. 실행중rigid-body setter방식에서root/reset이상발견으로이전진단결과는신뢰하지않는다. strictcheckpoint계약검사는유지하며복원후실제생성질량만검증한다. 기존mass40_view명령은그대로사용가능하다. 질량에맞는관성은같은boxgeometry에서자동계산된다.

## softgate 모델40kg질량진단 (2026-10-14)

`PORT=6086 bash stage2_jointpush/scripts/joint_push_softgate_mass40_view.sh /absolute/path/checkpoint.pth`. GPU6. 학습은원래60kg그대로유지하고뷰어만공유상자slot2의질량40kg/관성2/3배로개입한다. 모든checkpoint메타데이터/물리계약을먼저정상검증한뒤player에서적용한다. 설정·관측·목표·보상·마찰은변경없음. 이영상은60kg평가결과로보고하지않는다. Base/학습/3agentpreview에는적용불가.

기본산출물 `runs/jointpush/softgate_mass40_view/`; `mass_preview.json`에원래checkpoint질량과실측질량저장. 명시적평가옵션 `JOINT_PUSH_VIEW_MASS=40`으로headless test에서도같은진단가능. 4500iter모델GPU6 4환경60step검증rc=0,실측40kg확인: `runs/jointpush/mass40_check_20261014/`.

## 3명 추론 뷰어 (2026-10-14)

`scripts/joint_push_three_agent_{test,view}.sh`: GPU6 전용 9000iter/3명/60kg 추론. `joint_push_checkpoint_env.py --three-agent-preview`가 원본 저장환경의 질량·인원만 바꾼 사본을 생성한다. `HumanoidMAJointPush`의 명시적 preview 모드, 가변 OPEN/PUSH graph/runtime, 인원 기반 물리 tensor offset을 사용한다. 학습은 거부하며 기존2명 경로를 유지한다. `test_joint_push_three_agent_preview.py`는 인원 순열과 공유 endpoint를 검증한다.

## 촬영뷰어 숨김·카메라 보완 검증 (2026-10-14)

visual이없을때collision이대신렌더링되는경로를피하도록미사용asset에크기1e-6m의visual을명시한다. 기본_init_camera도카메라를재배치하지않도록override했다. 9000모델실제screen.png에서회색물체숨김과옆시점확인,reset전후카메라position/quaternion정확일치검증. `runs/jointpush/clean_scene_visual_check_20261014/`. 기존뷰어를종료하고재실행해야반영된다.

## 옆쪽 카메라·reset시점보존 (2026-10-14)

공동Push뷰어는처음에만미는방향의옆쪽에서시작한다. 높이3m/거리최소6m,agent·상자·목표중심을바라본다. agentfocus해제상태이며에피소드reset때카메라를재배치하지않아사용자가조절한위치가유지된다. 다음뷰어실행부터적용. 이전뒤쪽/reset재배치안내보다이설정이우선한다.

## 공동 Push 촬영용 배경 물체 숨김 (2026-10-14)

뷰어에서 미사용door슬롯0/1과box슬롯3의visual을제거한asset을사용한다. 슬롯2실제공유상자는표시한다. 관측슬롯/미사용물체의상태·collision및door관절구조는유지하며외부asset원본은수정하지않는다. headless학습asset은기존그대로다. 뷰어재실행후녹화해야적용된다.

## 공동 Push 뷰어 카메라 (2026-10-14)

reset마다 두agent중앙 뒤쪽에서 초기box진행방향을향해카메라를배치한다. 높이4m/뒤쪽거리최소6m,agent·상자·목표를포함하는중심을본다. 이후agentfocus는해제된자유카메라이며카메라는자동추적하지않는다. F키로수동추적전환가능. 다음뷰어실행부터적용.

## 뷰어 노란 목표 표시 제거 (2026-10-14)

사용자 요청으로 노란 목표 footprint/십자 overlay를 제거했다. 초기/현재 방향 화살표는 유지한다. 기존 작은 goal marker 및 실제 목표/관측/보상은 그대로다. 다음 뷰어 실행부터 적용된다.

## AT 연속 gate 비교 (2026-10-14)

별도 `joint_push_aligned_surround60_softgate_ca4.yaml`. 60kg/agent위치·heading360°/직선/동일면고정/목표거리2.2–5m는surround60과동일하다. `reward.at_gate_mode: soft_min`일때 AT state/progress에 `min(phi_push_agent0,phi_push_agent1) × opposite_side`를곱한다. phi_push에는손중점면거리·높이·면범위·팀대칭·간격연속점수가이미포함된다. 양쪽점수가0.9를살짝밑돌아도밀기보상을연속적으로유지한다. 두사람같은쪽이면0. 최종result/done는기존0.9양쪽손점수·반대편·간격·목표조건그대로이며PUSH보상도변경없다. 실제이동량보상은아직추가하지않는다. 기존설정은옵션없을때hard방식그대로다.

```bash
bash stage2_jointpush/scripts/joint_push_softgate_train.sh
bash stage2_jointpush/scripts/joint_push_softgate_test.sh /absolute/path/checkpoint.pth
HEADLESS=0 bash stage2_jointpush/scripts/joint_push_softgate_test.sh /absolute/path/checkpoint.pth
PORT=6089 bash stage2_jointpush/scripts/joint_push_softgate_view.sh /absolute/path/checkpoint.pth
```

Stage1초기값에서새Stage2로시작,동일softgate프로필끼리resume. 산출물 `runs/jointpush/aligned_surround60_softgate_training/`. GPU6/2048환경. 사용자요청으로기존rear-only60kg학습을종료하고tmux `jointpush_softgate`에서새Stage2학습을실행했다. run: `runs/jointpush/aligned_surround60_softgate_training/JointPushAlignedSurround60SoftGateCa4_14-11-37-12/`. 효과비교는같은surround60 hard-gate 설정과같은초기분포에서한다. 현재rear-only7000모델과는시작분포/목표거리도달라softgate만의효과로직접비교하지않는다.

## 60kg·상자 주변360° 시작 (2026-10-14)

`joint_push_aligned_surround60_ca4.yaml`: 앞면·옆면·뒷면 모두 허용하는 surround_independent 위치와 독립360° heading. 각 root는 box XY 외접원 반경(2.6m cube에서약1.84m)에1–2m를 더한 반경으로 배치하고 방위각을 균등 샘플링한다. 따라서 상자와의 최소 XY 간격은1m 이상이나 모든 면까지의 거리가 정확히1–2m인 것은 아니다. 두 root XY 최소간격0.8m. 상자는60kg/2.6m cube/목표local+Y정렬/reset고정면 유지. 목표까지 미는 거리는 reset마다2.2–5m균등랜덤.

```bash
bash stage2_jointpush/scripts/joint_push_surround_train.sh
bash stage2_jointpush/scripts/joint_push_surround_test.sh /absolute/path/checkpoint.pth
HEADLESS=0 bash stage2_jointpush/scripts/joint_push_surround_test.sh /absolute/path/checkpoint.pth
PORT=6088 bash stage2_jointpush/scripts/joint_push_surround_view.sh /absolute/path/checkpoint.pth
```

새 학습 산출물 `runs/jointpush/aligned_surround60_training/`. Stage1 전이에서 새 Stage2로 시작하고 동일surround60프로필끼리만resume한다. 현재 실행 중인 rear-only60kg학습을 자동 변경하지 않으며 새 본학습은 실행하지 않았다. 전용뷰어는 과거rear-only모델을 거부해 학습분포가360°였다고 오인하지 않게 한다. 다음curve설계에도360°위치·heading을 반영했다.

## 다음 직선 Push 실험 설계 (2026-10-14)

[확정 설계](joint_push_curve_plan.md): 곡선 제외,60kg·직선·agent위치/방향360°랜덤·동일면고정. 미는거리reset마다2.2–5m균등랜덤. 실행프로필surround60 준비 완료, 새본학습실행없음.

## 공동 Push 뷰어 표시 (2026-10-14)

공유 목표의 바닥 footprint(실제 box XY 크기)를 노란 윤곽/십자로 표시한다. 회색 화살표는 reset 시 agent 위치/heading으로 고정하고, 청록/분홍 화살표는 현재 heading을 따라간다. 랜덤 시작과 정책이 이후 상자 쪽으로 돌아서는 것을 구분할 수 있다. 표시만 추가하며 관측·초기화·보상·물리 계약은 바꾸지 않는다. 랜덤 시작 test/view wrapper는60kg/rear_independent checkpoint를 확인해 다른 모델을 거부한다. 기존뷰어를 재실행해야 새 표시가 적용된다.

6500iter60kg모델·GPU6·4환경60step 실제VNC 검증 rc=0. reset heading [-118.65°,128.47°] 확인. 랜덤 위치/heading·부분reset·고정면·60kg runtime 감사 통과. CPU 표시/배치2개 통과. 결과 `runs/jointpush/view_fix_check_20261014/`.

## 60kg·정렬 상자·독립 랜덤 agent 시작 (2026-10-13)

별도 `joint_push_aligned_random_spawn60_ca4.yaml` 프로필이다. 상자2.6m cube/60kg, 초기 목표 box local+Y 정렬, 접촉면 reset_fixed를 유지한다. loco-only RSI에서 두 agent의 초기 뒷면 거리1–2m·lateral[-1.8,1.8]m·상자 전진방향 대비 heading[-π,π]를 독립 샘플링한다. 좌우 역할을 미리 지정하지 않으며 같은 쪽에서 시작할 수도 있다. 두 root의 XY 간격0.8m 미만 후보는 다시 샘플링한다. 상자 주변360도 위치 샘플링은 아니다.

```bash
# 신규 학습: GPU6/2048환경, Stage1 전이에서 시작
bash stage2_jointpush/scripts/joint_push_random_spawn_train.sh
# 이 프로필에서 저장한 checkpoint 평가 / 로컬 viewer / 서버 VNC
bash stage2_jointpush/scripts/joint_push_random_spawn_test.sh /absolute/path/checkpoint.pth
HEADLESS=0 bash stage2_jointpush/scripts/joint_push_random_spawn_test.sh /absolute/path/checkpoint.pth
PORT=6087 bash stage2_jointpush/scripts/joint_push_random_spawn_view.sh /absolute/path/checkpoint.pth
```

학습 산출물은 `runs/jointpush/aligned_random_spawn60_training/`. 기존40kg run은 변경·중단하지 않는다. 사용자 요청으로 tmux `jointpush60`에서 GPU6/2048환경 신규 Stage2 학습을 시작했다. 기존 Stage2 checkpoint/optimizer/counter를 이어받지 않으며 Stage1 초기값 전이는 유지한다. 같은 프로필끼리만 RESUME_CHECKPOINT로 재개한다. 기존40kg 모델을 이60kg환경으로 전환하는 resume는 지원하지 않는다. test/view는 checkpoint 저장 설정을 복원하므로 과거 모델을 넣으면 과거 환경으로 실행된다.

## 공동 Push 접촉면 고정 (2026-10-13)

새 loco/koo RSI 및 정면 정렬 Stage2 설정은 `jointPush.face_mode: reset_fixed`를 사용한다. reset에서 목표 방향에 가장 가까운 뒷면을 한 번 선택해 상자 로컬 법선을 저장한다. 두 agent는 같은 면을 공유하고, 이후 목표 방향이 경계를 넘어도 재선택하지 않는다. 상자의 이동·yaw 회전에 따라 접촉면 중심과 법선은 갱신된다. 부분 reset은 해당 환경의 면만 갱신한다. 손의 lateral 대칭·간격·높이 보상은 유지한다.

기존 checkpoint에서 복원한 설정에 face_mode가 없으면 이전 `goal_dynamic` 규칙을 유지한다. 이는 저장 모델의 보상 계약 보존을 위한 것이며, 면 고정 학습은 새 설정으로 시작해야 한다. 곡선 Stage1 비교는 기존처럼 local -Y 고정면을 사용한다. 정면 정렬 새 설정의 질량은40kg이다.

## 정면 정렬 공동 Push (2026-10-13)

상자 yaw는 랜덤으로 유지하되 초기 목표를 box local +Y에 정렬하는 `goal_mode: box_forward` 설정을 추가했다. 기존 목표 방향 독립 랜덤 설정은 보존한다. 새 설정의 상자는 2.6×2.6×2.6m, 40kg이다. 기존 본학습 checkpoint는 40kg이며 자동 변경되지 않는다. 보상·AMP·모델 구조는 기존 설정을 따른다.

```bash
# 기본 loco-only RSI, GPU6
bash stage2_jointpush/scripts/joint_push_aligned_train.sh
# koo Push/loco 혼합 RSI
RSI=koo bash stage2_jointpush/scripts/joint_push_aligned_train.sh
# 해당 정렬 설정에서 저장한 checkpoint 평가 / 로컬 viewer / 서버 VNC
bash stage2_jointpush/scripts/joint_push_aligned_test.sh /absolute/path/checkpoint.pth
HEADLESS=0 bash stage2_jointpush/scripts/joint_push_aligned_test.sh /absolute/path/checkpoint.pth
PORT=6086 bash stage2_jointpush/scripts/joint_push_aligned_view.sh /absolute/path/checkpoint.pth
```

koo 모델 평가·뷰어에는 `RSI=koo`를 지정한다. 기존 goal360 checkpoint의 정렬 학습 재개 및 정렬 전용 평가를 거부한다. 새 학습은 기존 Stage1 전이에서 시작하며 동일 정렬 모델의 `RESUME_CHECKPOINT`만 재개한다. 새 본학습은 자동 실행하지 않았다.

질량 변경 전 60kg 설정에서 CPU 관련 테스트11개 통과. GPU6 4환경120step Base 정책 검증 rc=0; 초기 목표/면 법선 cosine=1.0, 실측 질량59.999996kg, reset/AMP 유한성 확인. 이는 환경 검증이며 학습 성능 검증은 아니다. 산출물: `runs/jointpush/aligned_goal_check_20261013/`.

## koo Push RSI 사용 (2026-10-13)

별도 `HumanoidMAJointPushKooRSI` / `joint_push_koo_rsi_ca4.yaml`을 추가했다. 실행 중인 loco-only 학습은 바꾸지 않는다. koo `open_push_push180_contact_at_warmstart.yaml`의 값을 그대로 따른다.

| 설정 | 값 |
|---|---|
| interaction/loco | 0.5 / 0.5 |
| late 요청 비율 | 0.7 |
| motion별 late 구간 | 후반0.3 |
| 손 stand-off | 0.06m |
| loco stateInit | Start |

원본은 사람별 독립 RSI지만 공동 Push는 환경별로 두 사람의 interaction 여부·Push motion/frame·상자 진행률을 공유한다. 그렇지 않으면 한 상자를 서로 다른 진행률로 덮어쓴다. 손은 큰 상자 뒷면의 lateral±면반폭0.5에 각각 배치하고 koo의 `lower_push_arms`·`align_push_pose`를 재사용한다. 손별 lateral offset/height는 기존 koo push 명세를 따른다. source bank의 `box_progress`만큼 실제 공유 상자를 한 번 이동하며 최종 목표는 원래 위치에 고정한다. 초깃값은 실제 reset 상태를 반복한 AMP history로 유지하고, AMP expert 혼합0.33loco/0.67Push는 변경하지 않는다.

복사본의 bank1048개 Push 후보와 PD_12_steps/source물리·reward·motion hash 계약을 검증한다. 이 bank는 작은 단독 상자의 검증 결과이며 큰 공유 상자의 전체 후보를 새로 PD screen한 bank로 주장하지 않는다. 공유 환경에서 실제 reset·손6cm 정렬·finite·학습/eval을 별도 검증한다.

```bash
# 신규 RSI 학습: 기본60kg, Stage1 epoch29000에서 전이
bash stage2_jointpush/scripts/joint_push_rsi_train.sh
# 저장 모델에 대해 RSI만 바꾸는 평가/뷰어: 물리·보상은 모델 설정 복원
bash stage2_jointpush/scripts/joint_push_rsi_test.sh /absolute/path/checkpoint.pth
PORT=6085 bash stage2_jointpush/scripts/joint_push_rsi_view.sh /absolute/path/checkpoint.pth
```

모든진입점GPU6고정. 짧은검증은MAX_ITERATIONS를양수로지정한다.0은CLI에서상한변경을하지않으므로새RSIwrapper에서거부한다. `OUTPUT_PATH`, `MAX_ITERATIONS`(학습), `SEED`, `NUM_ENVS`/`REPEATS`(평가), `STAGE1_CHECKPOINT`(신규학습)을 지원한다. RSI 모델끼리의 `RESUME_CHECKPOINT`는 동일 RSI 계약으로 재개한다. 기존 loco-only 모델의 optimizer를 그대로 RSI 학습으로 전환하는 재개는 지원하지 않으며 테스트/뷰어에서 RSI-only 진단만 허용한다. 진단은 기존40kg 모델이면40kg으로 수행하고 기본60kg과 구분한다. 엄격한 physics/reward 검증은 유지하며 초기화 항목만 명시적으로 허용한다.

16환경600step·9000iter40kg모델 평가 rc=0, 초기9개 환경이Push RSI로시작, 손중점깊이약-0.06m 확인. 목표성공0/16·scene종료6.25%, 최종PUSH edge성공각18.75%. 초기위치/phase가달라loco-only평가와성과를직접비교하지않는다. 결과 `runs/jointpush/koo_rsi_check_20261013/`에기록했다. 2048환경60kg신규학습reset에서1040환경interaction·738late요청·손depth[-0.06000020,-0.05999976] 확인;검증산출물 `runs/jointpush/koo_rsi_train_check_20261013/`.

## 60kg 곡선 Push 무학습 비교 (2026-10-12)

`joint_push_curve_test.sh` / `joint_push_curve_view.sh`는 실제2.6m cube60kg, 기존 koo Stage1, 뒷면·바닥 정렬1.3m 가상 cube 두 개를 사용한다. 목표는 초기 box local+Y 방향에서 좌우20–30° 어긋나게 배치하고 거리2.2–2.8m를 유지한다. 사람이 사용하는 면은 local-Y로 고정하며 box 회전에 따라 움직인다. 곡선 경로를 강제로 추종하는 것이 아니라, 방향 오차가 있는 상황에서 같은 면을 밀어 조향하는지 보는 비교다. 추가 학습은 없다.

agent별 초기 lateral 위치를 자기 가상 cube와 일치하도록 수정했다. 기존 carry tangent 방향을 그대로 썼을 때 일부방향의agent가자기cube반대쪽에서시작할수있던문제였다. 실제 reset 감사에서 lane오차<1e-4·뒷면뒤1–2m를 검증한다. 곡선 비교의 접촉 보상도 목표 방향 변화에 따라 옆면으로 바뀌지 않게 고정한다.

```bash
bash stage2_jointpush/scripts/joint_push_curve_test.sh
PORT=6084 bash stage2_jointpush/scripts/joint_push_curve_view.sh
```

GPU6 고정. 뷰어는6084포트포워딩. `OUTPUT_PATH`, `NUM_ENVS`, `REPEATS`, `STAGE1_CHECKPOINT`, `SEED` 사용 가능. CPU14개 통과, GPU6 16환경600step 평가 rc=0. 실제질량59.999996kg·초기20–30°·lane최대오차2.4e-7m 확인. 공동목표성공0/16,scene종료43.75%. trace30→570step 평균실제목표거리2.325→1.548m. 단일시드작은표본으로곡선제어성공을입증하지않는다. 산출물은 `runs/jointpush/virtual_curve_aligned_check_20261012/`다. 수정전초기배치검사는 `virtual_curve_check_20261012/`에보존했다.

Stage2 재개도 checkpoint 저장 env를 자동 복원하도록 변경했다. 따라서40kg 원본모델은40kg 그대로재개한다. 사용자실수로종료된본학습은2000itercheckpoint에서 `runs/jointpush/training_resume_20261012/`에재개했다. 곡선60kg비교와분리한다.

## Stage2 평가 설정 복원 (2026-10-12)

`joint_push_test.sh`와 `joint_push_view.sh`는 이제 checkpoint의 저장 env를 runs 아래 YAML로 복원한다. 따라서 기본 질량60kg과 달라도 기존40kg 모델을 원래40kg으로 평가한다. 엄격한 physics/reference 검증은 유지하며 무학습 가상 box 뷰어는별도 `joint_push_virtual_view.sh`로 실행한다. 학습 재개 스크립트도 checkpoint 저장 환경을 자동 복원한다.

## 현재 무학습 비교 관측: 뒷면 정렬1.3m cube (2026-10-12)

이전 가로 폭만 절반인1.3×2.6×2.6m 관측은 잘못된 크기 해석이었다. 현재는 agent별 가상 box를 Stage1 학습 크기인1.3×1.3×1.3m로 수정했다. 실제2.6m cube의 미는 뒷면과 바닥에 정확히 맞추고 좌우로 배치한다. local offset은 가로±0.65m, 목표를 향하는 법선 반대 방향0.65m, 아래로0.65m이며 가상 중심 높이는 바닥에서0.65m다. 목표에도 같은 offset을 적용해 공유 실제 box와 동일한 이동량을 요구한다. 회전/속도/corners/goal pose를 함께 갱신한다.

현재 config의 실제 상자는60kg이다. 실행 중인 기존 본학습은40kg으로 계속 실행되며 변경하거나 중단하지 않았다. 비교는 기존 Stage1 추론만 사용한다. 테스트/뷰어 명령은 기존 `joint_push_virtual_test.sh`/`joint_push_virtual_view.sh`를 그대로 사용한다. 뷰어는 자동 재실행하지 않았다. 새 비교 기본 산출물은 `runs/jointpush/virtual_rear_cubes_stage1/`이다.

CPU5개 통과:4방향 뒷면·바닥 일치,1.3m 크기,목표 이동량,물리 tensor 비변경,회전 및 offset velocity. 이전 half-width 평가 수치는 이전 관측에서 얻은 결과이며 현재 비교 결과와 구분한다. GPU6·실제60kg·16환경600step 평가 rc=0, 공동 목표 성공0/16, scene termination56.25%. trace의30→570step 평균 실제 goal 거리2.453→1.906m. 이전40kg 실험과 질량도 달라 직접 성능 비교하지 않는다.

## 상자 질량 변경 (2026-10-12)

공동 Push 및 무학습 반쪽 관측 config의 상자 질량을60kg으로 변경했다. 실행 중인 학습·뷰어는 중지/재시작하지 않았으므로40kg을 유지한다. 다음 신규 실행부터60kg을 적용한다. 아래40kg 검증은 변경 전 결과다. 기존40kg Stage2 모델을60kg 설정으로 평가/재개하면 물리 계약 검증이 거부하므로 기존 run의40kg 설정을 사용해야 한다.

## 무학습 Stage1 반쪽 관측 비교 (2026-10-12)

- `HumanoidMAJointPushVirtual`은 동일한 실제2.6m 정육면체40kg를 두 사람이 밀되, 기존 koo Stage1 epoch29000을 strict restore해 그대로 추론한다. CA4/확장 head/optimizer/추가 학습은 없다. 기존 Stage2 본학습 config는 바꾸지 않는다.
- 각 사람은 graph에서 자기 가상 상자/목표를 맡는다. 하나의 Stage1 scene policy가 두 사람의 action을 출력하며, 원래 Stage1의 사람·장면 attention은 유지한다. 두 사람의 관측을 완전히 독립된 scene으로 잘라 각각 별도 forward하는 비교는 아니다.
- 가상 상자는 미는 면의 가로 축을 반씩 분할한 크기다. 폭1.3m, 깊이/높이2.6m이며 중심은 실제 box 중심에서 lateral±0.65m. XYZ를 모두 절반으로 축소하지 않는다. 초기 목표 방향으로 lateral 축을 정하고 에피소드 동안 고정한다. box가 회전하면 중심 offset도 회전하고 rigid-body offset velocity를 반영한다.
- 가상 목표도 동일한 offset을 더해 두 사람 모두 실제 box→공유 목표와 같은 이동량을 요구받는다. object feature의8corners·pose·velocity와 goal pose를 함께 바꾸며, 물리 tensor/실제 collider/질량/힘/보상/성공 판정은 그대로 둔다. 두 사람의 실제 손 위치가 자동으로 대칭이 될 것을 보장하지는 않는다.
- Stage1 모델 전용 inference 계약 검증으로 schema12/semantic/PUSH relation/관측 차원/network 일치를 요구하며 Stage2 모델은 거부한다. 비교 config의 물리·RSI 차이는 의도적으로 허용하고 strict model/RMS 복원을 수행한다.

```bash
bash stage2_jointpush/scripts/joint_push_virtual_test.sh
PORT=6083 bash stage2_jointpush/scripts/joint_push_virtual_view.sh
```

GPU6 고정. 기본 모델은 로컬 Stage1 epoch29000이며 `STAGE1_CHECKPOINT`로 바꿀 수 있다. `OUTPUT_PATH`, `NUM_ENVS`, `REPEATS`, `SEED`를 지원한다. noVNC6083 포트 포워딩 후 접속한다.

검증: 기하 CPU2개 통과(half-box tiling/목표 이동량/회전/offset velocity/물리 tensor 비변경). GPU6 16환경600step1회 평가 rc=0, 공동 목표 성공0/16·scene termination81.25%. trace의30→570step 평균 실제 목표 거리2.456→2.081m, 손 중점 간격1.090→2.061m, 공동 접촉 gate는0. 이 작은 표본은 유효한 비교 경로 확인이며 학습된 Stage2와 성능 우열을 판정하지 않는다. 결과: `runs/jointpush/virtual_half_stage1_check_20261012/`; 실제 물리 trace는 `virtual_push_trace.jsonl`, 공식 성공 지표는 `metrics/edge_context_*.json`.

## 공동 Push 학습 연결 (2026-10-12)

현재 실행 경로는 `ports/koo_push/`의 `HumanoidMAJointPush`와 `joint_push_loco_rsi_ca4.yaml`이다. 아래의 함수만 구현했던 기록보다 이 연결 설정을 우선한다.

- 두 사람이 같은 박스와 목표를 사용하는 공동 Push 100%. 기존 Mixed80의 단독 20% 분할은 적용하지 않았다. Stage1 관측/checkpoint 호환을 위해 4물체 슬롯을 유지하며, 문 2개와 여분 박스 1개는 장면 밖에 배치한다.
- 공유 박스: 2.6×2.6×2.6m, 40kg, 마찰0.6. 목표는 360° 방향, 거리2.2–2.8m. 양손 중점의 팀 대칭·면 접촉·최소 간격으로 PUSH를 평가하고 현재 공동 접촉 gate로 AT를 활성화한다.
- RSI만 loco 100%: loco motion의 임의 시점 자세를 샘플링하고, 두 사람을 미는 면 뒤 1–2m에 배치해 방향을 맞춘다. Push RSI/paired bank는 사용하지 않는다. 초기 AMP history는 실제 reset 상태를 반복한다.
- AMP는 koo Push Stage1의 task-conditioned expert 구성을 유지한다: loco0.33 + Push0.67, 모든 샘플의 task label은 PUSH. 따라서 RSI의 loco-only 변경이 AMP를 loco-only로 바꾸지는 않는다.
- Stage1 encoder와 관측 RMS를 불러와 동결한다. PUSH+AT 관계 feature를 함께 사용하는 CA4(64D, 2head, FFN128)를 추가하고 action head 입력을 `[human64, context64]`로 확장해 `[W,0]`으로 초기화한다. CA4·head·critic·discriminator는 학습한다. critic/discriminator와 AMP RMS도 Stage1에서 초기화하며 새 학습의 optimizer/epoch는 초기화한다.
- 공동 운반 기본 유지: state/progress/success 각0.2, self1/teammate0, power0.0005, CPA coefficient0.5·distance0.7·discount0.99, task/AMP0.5/0.5. PPO lr2e-5, horizon32, minibatch16384, mini_epochs6, 2048환경, 600step.
- checkpoint 전이는 소스 schema·network·tensor shape를 검증한다. 저장한 Stage2 모델은 동일 설정으로 평가/재개하며 보상·RSI 계약이 다르면 거부한다.

프로젝트 루트에서 실행하며 아래 스크립트는 GPU 6을 강제한다.

```bash
bash stage2_jointpush/scripts/joint_push_train.sh
bash stage2_jointpush/scripts/joint_push_test.sh /absolute/path/checkpoint.pth
bash stage2_jointpush/scripts/joint_push_view.sh /absolute/path/checkpoint.pth
RESUME_CHECKPOINT=/absolute/path/checkpoint.pth bash stage2_jointpush/scripts/joint_push_train.sh
```

`STAGE1_CHECKPOINT`로 전이 소스를 바꿀 수 있다. 기본은 로컬에 복사한 koo epoch29000이다. `OUTPUT_PATH`, `MAX_ITERATIONS`, `SEED`를 지원한다. viewer noVNC 포트는6081이며 기존 viewer와 중복 실행하지 않는다.

검증: CPU10개 통과, GPU6에서2048환경 짧은 학습·저장 모델 재개와16환경600step 평가 rc=0. 런타임 감사에서 loco-only RSI·공유 graph/goal·실제40kg·초기 AMP history·partial reset 보존·AMP finite를 확인했다. 저장 모델의 encoder75개 tensor와 관측 RMS는 원본과 완전히 동일했고 action head와 CA4 네 층은 업데이트됐다. 로그는 `runs/jointpush/push_stage1_repro_20261012/connected_final_train.log`, `connected_final_eval.log`, 감사는 `runs/jointpush/connected_final_check_20261012/`에 있다. 짧은 검증 모델의 공동 목표 성공률은0이며 성능 학습은 아직 필요하다. 본학습은 실행하지 않았다.

## 2인 공동 Push 보상 명세 (2026-10-12)

- 구현: `ports/koo_push/tokenhsi/env/tasks/multi_agent/joint_push_reward.py`; 보상 명세: `ports/koo_push/tokenhsi/data/cfg/multi_agent/joint_push_reward.yaml`; 공유 박스 명세: `ports/koo_push/tokenhsi/data/cfg/multi_agent/joint_push_scene.yaml`. 공동 물체 생성·reset·학습 루프 연결은 위의 현재 학습 설정을 따른다. 기존 Stage1 평가 설정과 원본은 유지한다.
- 공유 박스는 한 변2.6m 정육면체,40kg,마찰0.6. 크기를 중심으로 협동 필요성을 검토하기 위해 깊이/높이도2.6m로 변경했다. 질량은40kg유지하므로 직전직육면체보다밀도는1/4이다. 크기만으로단독Push불가능을보장하지않으며단독/공동평가가필요하다.
- 고정 좌우 앵커를 제거했다. 각 agent의 양손 평균 m_i를 사용해 두 평균의 중점이 실제 뒷면 중심c에 가까울수록 symmetry=exp(-10×||mean_i(m_i)-c||²). 미는 면까지의 깊이오차·면가로범위이탈·높이0.8m오차로 개인 surface score를 계산한다.
- lateral 중점 간격은 최소 면폭×0.25(기본2.6m면0.65m)까지 연속 점수min(separation/minimum,1). 개인 PUSH phi=surface_phi×symmetry_phi×separation_phi. 서로 반대쪽·간격기준충족·두 개인phi≥0.9이면 AT gate가 열린다. 좌우 슬롯/정확한접촉위치는 지정하지 않는다. 이 함수는 현재2명 전용이며 추가 agent 확장은 별도 분산/간격 기준이 필요하다.
- Goal360 joint carry 기본을 따라 PUSH+AT의 state/progress/success 각0.2를 개인별 합산(self1/teammate0), 현재 gate 재폐쇄·접촉 비포화를 유지한다. power/CPA/AMP는 연결된 공동 Push 환경에서 적용한다. PUSH progress=root→상자XY의반대각선+0.2m pinning,AT progress=상자→목표XY의0.1m pinning,1/(1+d);AT state=exp(-10×3D거리²),성공=state≥0.9·Z오차≤0.001m·공동gate.
- 목표방향을 box yaw로 바꿔 가장 잘 향하는 면의 뒷면을 선택한다. 목표XY가중심이면 local+Y 사용. 면선택경계불연속·upright/yaw기하·반력미검증은 기존명세와 동일하다. 손 목표를 policy 입력에 추가하지 않는다.
- 입력 roots[N,2,13],hands[N,2,2,3],shared box[N,13],shared goal[N,3],실제size[3] 또는[N,3]. edge tensor 축[N,agent,(PUSH,AT)].
- CPU10개 통과(보상7+모델3): 여러대칭간격동일점수·중앙몰림/좁은간격/비대칭/면이탈차단·사람교환·gate재폐쇄·좌표회전/이동·성분합·Z오차·박스크기/질량/밀도·폭별간격·모델회귀. 프로젝트로그 `runs/jointpush/push_stage1_repro_20261012/symmetric_reward_tests_retry.log`. GPU/PhysX/MPS는 사용하지 않았다.

## PUSH+AT CA4 모델 연결 (2026-10-12)

- 격리 복사본 `ports/koo_push/tokenhsi/learning/multi_agent/push_at_coordination.py`에 구현했다. 환경·보상·Stage1 관측/edge packet은 유지한다. 공동 Push 물리 환경·학습 CLI는 위의 현재 설정에 연결했다.
- Stage1은 `[source type embedding16, relation embedding32, target type embedding16] → Edge MLP64 → layer/head projection → attention bias`를 사용한다. 새 모듈은 projection 전 PUSH64·AT64 관계 feature를 동결 재사용한다.
- `[actor64, payload64, target64, PUSH64, AT64] → Grounding MLP(320→128→64)`로 사람별 작업 토큰을 만든다. PUSH edge의 owner/payload와 일치하는 AT를 요구하며 같은 payload/goal을 가진 두 작업은 공동 관계다. CA+FFN4층·64D·2head·FFN128·층별 자기/공동/기타 bias를 적용한다.
- 사용: 기존 Stage1 모델을 먼저 strict load한 뒤 `attach_push_at_coordination(model.a2c_network)`를 호출한다. action head를 `[W,0]`로 확장하고 actor encoder를 동결한다. 함수는 새 파라미터를 만들므로 optimizer 생성 전에 호출해야 한다. 관측 RMS는 기존 통계를 복원하고 별도로 eval/동결해야 한다. 이 helper는 agent/RMS/optimizer를 수정하지 않는다.
- Stage2 저장 모델 복원은 동일 Stage1 구조를 만든 뒤 helper로 확장하고 Stage2 state_dict를 strict load한다. 기존 Stage1 평가 명령은 helper를 호출하지 않아 그대로 동작한다. Stage2 checkpoint 계약·CLI resume는 현재 전용 agent/스크립트에 연결했다.
- 테스트: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=tokenhsi /home/hwanhee/anaconda3/envs/tokenhsi/bin/python -m pytest tokenhsi/tests/test_push_at_coordination.py -q` (ports/koo_push에서 실행). CPU3개 통과: 두 관계 기여·edge 순열·잘못된 bundle 거부·빈 graph finite·사람 교환·CA4 gradient·실제29000 checkpoint 초기 action 보존·encoder 동결·head 업데이트 후 grounding gradient. 결과 로그는 프로젝트 `runs/jointpush/push_stage1_repro_20261012/model_tests.log`다. 초기 context head가0이므로 첫 backward의 CA gradient는0이며 head가 업데이트된 이후 CA gradient가 열린다.

## Push Stage1 원본 재현 (2026-10-12)

- 코드: `ports/koo_push/`는 `/home/hwanhee/koo_cvpr/stage1_fix`의 tokenhsi/lpanlib/mps 작업 파일 복사본이다. 공동운반 core는 유지한다. 데이터 심링크 12개는 원본에서 해석한 절대 경로로 연결했다. 원본 폴더는 변경하지 않았다.
- 모델: 프로젝트 `runs/jointpush/push_stage1_repro_20261012/checkpoints/`의 고정 epoch29000. GPU는 두 진입점 모두 6으로 고정한다.
- 프로젝트 루트에서 평가: `bash stage2_jointpush/scripts/push_stage1_test.sh` (2명·4환경·각자 Push·600step·1회).
- 프로젝트 루트에서 뷰어: `bash stage2_jointpush/scripts/push_stage1_view.sh`. 기존 뷰어 실행 중에는 중복 실행하지 않는다. noVNC 포트6081을 포워딩하고 `http://localhost:6081/vnc.html?autoconnect=1&resize=remote`에 접속한다. 현재 카메라는 한 사람/상자에 집중한다.
- 원래 OPEN/PUSH 혼합 장면은 `TASK_GRAPH=random_open_push`로 선택한다. 기본 push는 두 사람이 각자의 상자를 미는 Stage1이며 공동 Push가 아니다.
- 로그·metrics·화면 캡처·복사 manifest는 프로젝트 `runs/jointpush/push_stage1_repro_20261012/`에 둔다. 120step smoke와600step4환경 평가가 rc=0으로 완료했고 실제 Isaac Gym 렌더 화면을 확인했다. 작은 표본의 재현 검사이며 장기 성능 판정은 아니다.

# 레포 구조

현재 실행 실험은 원본 multi-agent carry(1), 독립 Stage 1(27·28·30·31·32·33·34·35·36·37 및 paired AT/placement·unified 독립 5과제), 협력 Stage 2(29 및 SIT plane 변형)다. 실행 명령과 checkpoint 규칙은 [config.md](config.md)에 있다.

## 진입점

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/scripts/prepare_joint_carry.py`, `joint_carry/runtime_manifest.json` | Clone 후 기존 데이터 링크·필수 파일·현재 AMP/RSI 해시 검사; 두 LocoAMP train/test 및 VNC 경로에서 실행 |
| `tokenhsi/run.py` | 학습·평가 태스크와 알고리즘 등록 |
| `tokenhsi/tests/toy_joint_carry_amp.py`, `tokenhsi/scripts/multi_agent/joint_carry_amp_toy.sh` | Align checkpoint의 공동 AT·carryWith 직선 운반 AMP 유지/제거 진단; 2048환경 짧은 학습·64환경 평가, 본학습 설정은 유지 |
| `tokenhsi/tests/toy_joint_carry_amp_report.py` | 위 진단의 동일 초기 상태 검사·거리 그래프·시뮬 관절 궤적 비교 영상 → `output_etc/align_amp_toy_20261010/` |
| `tokenhsi/tests/toy_joint_carry_goal_reward{,_report}.py` | 같은 물리 초기 상태에서 목표를 기존 이동 방향으로 변경한 정책 평가·기존 action 재생; 보상 성분/공동 지급/거리 단조성 검사 → `output_etc/align_goal_reward_20261010/` |
| `tokenhsi/tests/toy_joint_carry_permutation.py` | 2명·4물체 Mixed80의 사람 상태/goal/graph 교환 및 SA/CA 순열에 대한 관측·RMS·encoder·action 검사 → `output_etc/joint_carry_permutation_20261011/`; 별도 물리 rollout 교환 검사는 아님 |
| `tokenhsi/box_cleanup_demo.py` | 지정 rescue checkpoint의 4명·16상자 정리 데모 평가 전용 진입점 |
| `tokenhsi/scripts/multi_agent/` | 현재 실험별 train/test/VNC, `runtime_env.sh`, `run-gui.sh` |
| `tokenhsi/data/cfg/multi_agent/` | 현재 실험 YAML과 평가 graph 예제 |
| `tokenhsi/data/cfg/train/rlg/` | PPO/AMP/Transformer 학습 설정 |
| `output/<실험>/<run>/` | checkpoint `nn/`, TensorBoard `summaries/`, 진단 `diagnostics/` |
| `output_etc/` | 비활성 검증·분석·이전 영상 보관; `archive_20261009_cleanup.json`에 이동 전/후 경로 기록. `output/`은 본학습4개·RSI cache·현재 전체 보정 영상 유지 |
| `joint_carry/scripts/`, `joint_carry/HANDOFF.md` | 공동운반 RSI 합성·팔 IK·CPU PhysX 검사·인계 안내; 재현 결과는 `output/joint_carry_rsi_rebuild_20261007/`의 `verification.json`, `corrected/before_after/`, `pd_videos/` |
| `joint_carry/teamhoi_reference/`, `joint_carry/scripts/preview_teamhoi_reference.py` | 공식 TeamHOI 후진9개·옆걸음3개와 출처/해시; CPU 원본 FK 영상 생성 → `output_etc/teamhoi_reference_preview/` (AMP 미연결) |
| `joint_carry/teamhoi_retarget_pilot/`, `joint_carry/scripts/{retarget_teamhoi_reference,check_teamhoi_retarget,preview_teamhoi_retarget}.py` | 후진·옆걸음 각1개를 현재 asset에 맞춘 IK/접지·속도 보정, AMP/2048환경 GPU 물리 검사 및 비교 영상 → `output_etc/teamhoi_retarget_pilot_check/`; 학습 미연결 |
| `joint_carry/teamhoi_retarget_all/` | 과도한 무릎 굽힘이 있는 이전12개 보정본(현재 학습 미사용); 위 스크립트의 `--all` → `output_etc/teamhoi_retarget_all_check/` |
| `joint_carry/teamhoi_retarget_posture_pilot/`, `joint_carry/scripts/compare_teamhoi_gait.py` | 원본 무릎 자세를 보존하는 재보정2개·기존 걷기5개 대조; 위3개 스크립트 `--posture-first`, 결과/4열 영상 → `output_etc/teamhoi_retarget_posture_pilot_check/`; 학습 미연결 |
| `joint_carry/teamhoi_retarget_posture_all/` | 현재 LocoAMP용 자세 보존12개·해시·AMP9/3개 목록; 보정/검사/비교 영상 `--all --posture-first`, 모아보기 추가 `--montage`, 결과 → `output/teamhoi_retarget_posture_all_check/` |
| `output/joint_carry_gpu5_check_20261007/check_gpu.py`, `report.json`, `per_env.npz` | GPU PhysX·2048환경의 paired RSI 319개 초기 충격·공통 yaw·부분 reset 검사; 학습 환경 연결·AMP 검증과 구분 |

## 현재 실험이 사용하는 코드

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py` | 물체 배정·RSI·물리 환경·관측 통합 |
| `tokenhsi/env/tasks/multi_agent/scene_features.py` | clean-scene 관측 보조·공유 물체/목표 소유자·노란색·ON_TOP 큐브/SIT 원/CLIMB 십자 마커 |
| `tokenhsi/env/tasks/multi_agent/box_cleanup_demo.py`, `tokenhsi/learning/multi_agent/box_cleanup_player.py`, `tokenhsi/utils/box_cleanup_spec.py` | 중앙 16상자 크기 혼합·불규칙 2단 더미·윗단 두 라운드 운반·회색 표시·바닥 안정 판정·옛 rescue checkpoint 평가 로드·기본 10회/1500 step·VNC 결과 저장 |
| `tokenhsi/tests/test_box_cleanup_demo.py` | 8개 상자 배정·100개 혼합 크기 배치의 초기 겹침·목적지 높이·checkpoint 계약·네 명 완료 barrier와 물리 상태 유지 검증 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_task.py`, `edge_context_task.py` | graph reset, 가까운 시작, 물리 타당성·진단 |
| `tokenhsi/env/tasks/multi_agent/edge_stage1_reward.py` | Stage 1 edge 보상·포화·팀 공유 |
| `tokenhsi/env/tasks/multi_agent/edge_ontop_reward.py`, `edge_interaction_reward.py` | AT·ON_TOP·SIT·CLIMB state/progress와 plane 성공 |
| `tokenhsi/utils/edge_scenario_spec.py` | 독립 물체·goal binding, 3물체·4물체 같은 과제 쌍 및 canonical 독립 5과제 sampler; size RSI의 GPU 배치 graph 생성 |
| `tokenhsi/utils/size_rsi.py`, `tokenhsi/env/tasks/multi_agent/size_rsi_cache.py` | 행동별 실제 asset 크기·조건부 과제 배정·프레임별 초기 충격 캐시·후반 RSI 선택 |
| `tokenhsi/utils/unified_training.py` | unified 과제별 AMP 전문가 분포·라벨·리플레이 매칭·설정 검증 |
| `tokenhsi/utils/edge_stage1_spec.py` | Stage 1 variant·semantic/owner-HOLDING packet·후반 CLIMB/ON_TOP putDown RSI 검증 |
| `tokenhsi/utils/edge_stage2_spec.py` | Stage 2 협력 graph·평가 preset |
| `tokenhsi/utils/relation_task_spec.py` | reward config·checkpoint 계약 검증 |
| `tokenhsi/learning/multi_agent/amp_network_builder_ma.py`, `edge_context_encoder.py`, `ma_agent.py` | GTA actor/critic, edge bias·HOLDING 상태 융합, AMP·PPO 학습 |
| `tokenhsi/learning/multi_agent/coordination_head.py`, `stage2_transfer.py` | 협력 attention·Stage 1 weight 이식 |
| `tokenhsi/utils/joint_carry_spec.py`, `tokenhsi/learning/multi_agent/task_coordination.py`, `tokenhsi/tests/test_joint_carry.py` | Joint AT/ON_TOP sampler·config 계약·task CA·NONE/SELF/COUPLED bias·전이·순열 검증 |
| `tokenhsi/tests/{test_joint_carry_teamhold,check_joint_carry_teamhold_runtime}.py`, `approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_teamhold_task_embedding` env/train·train/test/VNC | Goal360 기반 공동 HOLDING state 교체; `joint_carry_reward.py` 점수/보상과 `edge_ontop_task.py` 부분 reset 전달, reward metadata 격리·개인 success/progress/단독 불변 검사 |
| `tokenhsi/utils/joint_carry_goals.py`, `tokenhsi/tests/{test_joint_carry_goal360,check_joint_carry_goal360_runtime}.py`, `approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding` env/train·train/test/VNC | CA4 공동 목표360°·거리 샘플링·받침 겹침 검사; `joint_carry_reset.py` 선택 분기와 보정 putDown 끝점, 별도 checkpoint 계약·RSI/AMP/부분 reset 검증 |
| `tokenhsi/tests/test_joint_carry_ca4.py`, `approach_stage2_joint_carry_mixed80_locoamp_ca4_task_embedding` env/train·train/test/VNC | 기본 LocoAMP 기반4층 CA+FFN+LayerNorm+residual; `task_coordination.py`의 `CoordinationBlock`, 층별 bias·공유 memory·순열/gradient·전이·계약 격리 검사 |
| `tokenhsi/tests/{test_joint_carry_move,check_joint_carry_move_runtime}.py`, `approach_stage2_joint_carry_mixed80_locoamp_ca4_move_task_embedding` env/train·train/test/VNC | CA4 기반 공동 XY 이동 방향 min 보상·pinning; `joint_carry_reward.py`의 `shared_alignment_reward` 이동 모드, `edge_stage2_spec.py` reward 계약; 방향/속도 크기·순열·좌표 변환·gate·실제 보상 합 검증 |
| `tokenhsi/tests/test_joint_carry_alignment.py`, `approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding` env/train·train/test/VNC | 공동 HOLDING 중 farthest-human heading 공유 보상·포화 거리·좌표/순열·checkpoint 계약 검사; 구현은 `joint_carry_reward.py`, 통합은 `edge_context_task.py` |
| `tokenhsi/utils/joint_carry_amp.py`, `tokenhsi/tests/{test_joint_carry_amp_ablation,check_joint_carry_amp_runtime}.py` | AMP 전용 후진/옆걸음 라이브러리·80/10/10·데이터 해시·CA/head-only checkpoint 격리; CPU 및 실제 Mixed80 reset/AMP 검증 |
| `tokenhsi/utils/joint_carry_rsi.py`, `tokenhsi/env/tasks/multi_agent/joint_carry_reset.py`, `joint_carry_reward.py`, `tokenhsi/tests/test_joint_carry_runtime.py` | 보정 paired snapshot 로드·공통 좌표 변환·공동/부분 reset·AMP 초기 이력·반대 앵커 placement gate·agent CPA 보상 검증 |
| `tokenhsi/tests/test_scenario_independent_with_climb.py`, `test_scenario_stage1_plane.py`, `test_scenario_stage1_sit_plane_curriculum.py`, `test_stage2_coordination.py` | 현재 실험 CPU 검증 |
| `tokenhsi/tests/test_size_rsi.py` | size RSI config·checkpoint 격리·크기/과제 분포·progress·프레임 pool·대체 skill·배치 graph/부분 reset 동등성 검증 |
| `tokenhsi/tests/test_stage1_unified.py` | 세 unified config 계약·공유 edge gradient·독립 분포·goal 회전·AMP family 매칭·토큰 순열의 actor/value 및 gradient 검증 |
| `tokenhsi/tests/test_typed_edge_bias.py` | Size RSI typed-bias 계약·전체 과제 쌍/goal slot/edge 방향·토큰/edge 순열의 출력/gradient·독립 표 업데이트·checkpoint 복원 검증 |
| `tokenhsi/tests/test_typed_edge_message.py` | 관계 메시지 설정·GTA 뒤 가중합·head 분할·NONE/SELF masking·순열 출력/gradient·독립 업데이트·checkpoint 복원 검증 |
| `tokenhsi/tests/test_graph_box_speed_penalty.py` | graph 담당 물체의 속도 패널티·부분 reset history·기존 할당 경로 회귀 검증 |

`edge_context_spec.py`, `edge_ontop_spec.py`, `edge_interaction_spec.py`와 관련 reward 함수는 과거에 도입됐지만 현재 Stage 1·2의 graph·관찰·보상 계산에서도 호출된다. 파일명만 보고 삭제하면 현재 실험이 깨진다.

Size RSI typed-bias의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_typed_bias` 이름을 쓴다. `amp_network_builder_ma.py`의 `TypedEdgeBias`와 `edge_context_encoder.py`의 `PackedEdgeTypedBiasFusion`이 타입별 bias 표를 graph에 연결한다.

Typed bias + relation message의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_typed_bias_message` 이름을 쓴다. `edge_context_encoder.py`의 `PackedEdgeRelationMessage`가 layer 공유 관계 표를 조회하고, `RelationTransformerLayer`가 GTA 복귀 후 같은 attention으로 가중한 관계 메시지를 더한다.

Task message의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_task_message` 이름을 쓴다. `utils/task_role_spec.py`가 primitive graph에서 task·역할 packet과 정책 3연결을 만들고, `learning/multi_agent/task_role_encoder.py`의 `TaskRoleFusion`이 역할별 bias/message 표를 조회한다. `tests/test_task_role_message.py`는 16개 과제 쌍·물체/goal/owner 재배정·edge/task/token 순열·gradient·보상·크기/RSI/AMP binding·checkpoint를 검증한다.

Task MLP의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_task_mlp` 이름을 쓴다. Task message와 `task_role_spec.py`의 task packet·3연결·RSI/AMP 경로를 공유하고, `task_role_encoder.py`의 `TaskRoleMLPFusion`은 메시지 없이 task·출발 역할·도착 역할을 MLP bias로 인코딩한다. `tests/test_task_role_mlp.py`는 env 설정 동일성·메시지 부재·MLP 출력/gradient·순열·보상·checkpoint 분리를 검증한다.

Task MLP split의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_task_mlp_split` 이름을 쓴다. `TaskRoleMLPFusion(split_tasks=True)`는 역할 embedding을 공유하고 task별 네 MLP·projection을 적용한다. `tests/test_task_role_mlp_split.py`는 task별 weight/gradient 독립성·공유 weight 복제 시 출력/gradient 동등성·순열·RSI/AMP/보상 동일성·checkpoint 격리를 검증한다.

Task embedding의 전용 env/train YAML과 train/test/VNC는 `approach_scenario_stage1_unified_size_rsi_task_embedding` 이름을 쓴다. `task_role_encoder.py`의 `TaskTypeEmbeddingBias`는 task/NONE/SELF의 6개 embedding을 H/O/G 타입 쌍·layer/head별 공유 projection으로 변환한다. `TaskTypeEmbeddingFusion`은 canonical task 연결에 bias를 적용하며 이후 공통 토큰/GTA 순열 경로를 따른다. `tests/test_task_type_embedding.py`는 카테고리·타입 공유, 배경/역방향/padding, endpoint 재매핑·출력/gradient 순열, RSI/AMP/보상 동일성·checkpoint 격리를 검증한다.

Carry-only distillation은 `approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill` env/train YAML과 train/test/VNC를 쓴다. `task_role_spec.py`의 carry-only 조건부 확률과 `humanoid_ma_carry.py`의 carry 크기·AMP family 분기를 사용한다. `learning/multi_agent/stage1_unified_teacher.py`는 graph를 원본 teacher 관측으로 변환하고, `distillation.py`는 Gaussian KL·gradient 검사를 제공한다. `ma_agent.py`는 rollout label·minibatch·loss를 연결한다. `tests/test_carry_distillation.py`가 carry 샘플링·RSI/보상 유지·AMP family·목표/물체/goal binding·label 정렬·teacher 동결·학생 gradient·checkpoint 분리를 검증한다.

네 과제 전체 distillation은 `approach_scenario_stage1_unified_size_rsi_task_embedding_distill` env/train YAML·train/test/VNC를 쓴다. 원래 task embedding의 과제·크기·RSI·AMP를 유지하고 위 teacher/KL 경로를 공유한다. `tests/test_task_embedding_distillation.py`가 원본 설정 동일성·checkpoint 분리·16개 과제 쌍의 teacher 라우팅/목표·SIT 방향·네 task gradient·AMP family를 검증한다.

## 작업별 확인

- **Stage 1 graph·보상:** 해당 YAML → `edge_scenario_spec.py` → `edge_ontop_task.py` → `edge_interaction_reward.py`·`edge_stage1_reward.py`; 위 Stage 1 테스트로 검증한다.
- **Stage 2 협력·전이:** `approach_stage2_coordination.yaml` → `edge_stage2_spec.py` → `coordination_head.py`·`stage2_transfer.py`; `test_stage2_coordination.py`로 검증한다.
- **실행·GPU:** [config.md](config.md), `runtime_env.sh`, `mps/`, `run-gui.sh`를 확인한다.
- **지표 해석:** [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 확인한다.

Joint carry Stage 2는 `approach_stage2_joint_carry_task_embedding` env YAML·train/test/VNC와 `amp_ma_stage2_joint_carry_task_embedding` train YAML을 쓴다. `humanoid_ma_carry.py`가 joint 전용 reset·reward·carry AMP를 연결하고 `stage2_transfer.py`가 Stage 1 actor/RMS 동결과 head 확장을 담당한다. GPU 연결 검사는 `output/approach_stage2_joint_carry_task_embedding_check/`의 `check_reset.py`, `reset_check.json`, `check_saved.py`, `saved_check.json`에 있다.

Mixed80 공동/단독 운반은 `approach_stage2_joint_carry_mixed80_task_embedding` env YAML·train/test/VNC와 `amp_ma_stage2_joint_carry_mixed80_task_embedding` train YAML을 쓴다. `utils/mixed_carry_spec.py`는 고정 환경 분할·독립 graph를, `env/tasks/multi_agent/mixed_carry_reset.py`는 공동/단독 reset의 단일 commit과 Stage 1 크기별 RSI cache 재사용을 담당한다. `joint_carry_reward.py`가 공유 여부에 따라 반대 앵커 또는 자기 HOLDING gate를 적용한다. CPU 검사는 `tests/test_mixed_carry.py`, GPU 결과는 `output/approach_stage2_joint_carry_mixed80_task_embedding_check/`에 있다.

BEFORE Stage 2는 `approach_stage2_before_task_embedding` env YAML·train/test/VNC와 `amp_ma_stage2_before_task_embedding` train YAML을 쓴다. `utils/before_spec.py`는 공통 크기·고정 80/20 환경·세 dependency graph·AMP family 분포를, `env/tasks/multi_agent/before_reward.py`는 현재 AT prerequisite과 gated success/포화를 담당한다. `before_reset.py`는 공유 물체 reference 정렬·속도/AMP history 변환·단일 물리 commit을, `before_rsi_cache.py`는 실제 source/support 크기의 물리 RSI 선별을 담당한다. 기존 `task_coordination.py`의 별도 BEFORE 모드에 비대칭 bias를 추가하며 joint checkpoint 형식은 유지한다. CPU 검증은 `tests/test_before.py`, GPU 초기 물리/부분 reset·AMP 검사는 `tests/check_before_runtime.py`이며 최종 결과는 `output/approach_stage2_before_task_embedding_check_final/`에 저장한다. 물리 안정성 진단은 pass 수를 별도로 기록하며 모든 장면 통과를 의미하지 않는다.

### 비정렬 자유 표면 Contact

- `ports/koo_push/tokenhsi/env/tasks/multi_agent/joint_push_reward.py`: `directional_surface_contact`, 실제 옆면·개별 손·목표 방향·간격 기반 Contact와 기존 fixed-face 분기.
- `ports/koo_push/tokenhsi/data/cfg/{multi_agent,train/rlg}/joint_push_directional_surface_ca4.yaml`: 독립 목표 방향과 새 Contact 전용 설정.
- `scripts/joint_push_directional_surface_{train,test,view}.sh`: 전용 실행 진입점. port 내부 동명 `ca4_{train,test,vnc}.sh`로 연결.
- `ports/koo_push/tokenhsi/tests/test_directional_surface_push.py`: 인접 면 접촉, 손 중점 악용, 잘못된 방향, 간격·높이, yaw 불변성, AT gate 회귀.

### 각도별 고정 면·손 목표점 실험

- `ports/koo_push/tokenhsi/env/tasks/multi_agent/joint_push_assigned_contact.py`: reset-time 각도별 면 선택·최소 접근거리 agent 배정·로컬 손/접근 target·실제 box pose 변환.
- `humanoid_ma_joint_push.py`: reset subset 배정 버퍼, step마다 box pose로 목표 변환. `joint_push_reward.py`: `assigned_faces` Contact/AT gate 분기.
- `ports/koo_push/tokenhsi/data/cfg/{multi_agent,train/rlg}/joint_push_assigned_faces_ca4.yaml`: 높이1.6~2.6m/각도25° 경계 전용 프로필.
- `scripts/joint_push_assigned_faces_{train,test,view}.sh`: train/test/noVNC 진입점. 내부 `joint_push_assigned_faces_ca4_{train,test,vnc}.sh` 연결.
- `ports/koo_push/tokenhsi/tests/test_assigned_faces_push.py`: 각도 경계·인접 면·교환 불변성·box pose 추종·잘못된 접근·개별 손·부분reset·높이 격자 검사.

## 공유 상자 분할 Push

- `ports/koo_push/tokenhsi/env/tasks/multi_agent/humanoid_ma_partition_push.py`: 순차coupled80%/NONE20% 가변그룹, 인원비례상자/n분할+랜덤일반4개로8토큰, reset별 일대일 고정 구역·손point·그룹별보상.
- `ports/koo_push/tokenhsi/utils/partition_push.py`: CPU 그룹샘플러·24후보asset크기/질량·가변분할기하·그룹 내 최소 거리 합 일대일할당·그룹보상.
- `ports/koo_push/tokenhsi/data/cfg/{multi_agent,train/rlg}/partition_push_ca4.yaml`: 환경/학습 프로필.
- `ports/koo_push/tokenhsi/data/partition_push_motions.yaml`: raw Push+loco AMP.
- `ports/koo_push/tokenhsi/scripts/multi_agent/partition_push_setup.py`: source 경로 선택, checkpoint 환경 복원·추론 ablation.
- `scripts/partition_push_{train,test,vnc}.sh`: 로컬 진입점.
- [PARTITION_PUSH.md](PARTITION_PUSH.md): 설정·로컬 데이터 준비·실행법.

### 랜덤 출발·CPA0.5 전용 실험

- `ports/koo_push/tokenhsi/data/cfg/multi_agent/partition_push_surround_cpa05.yaml`: 기본 Partition Push를 보존하는 surround/CPA0.5 설정.
- `scripts/partition_push_surround_cpa05_{train,test,vnc}.sh`: 전용 env·산출물로 분리한 학습/평가/기존 VNC 실행. test/VNC는 checkpoint env를 복원한다.
