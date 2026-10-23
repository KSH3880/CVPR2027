# 단독 행동 점/영역 평가와 PUSH 평가

[평가 명세](../../markdowns/evaluation_single.md)의 `single_match_region_v1`을 기존 코드와 분리해 구현했다. 실행 코드는 이 폴더에만 추가했으며, 기존 학습 코드·YAML·체크포인트·원본 JHH/KSH 레포는 수정하지 않는다. `backend.py`의 평가 전용 subclass를 별도 프로세스에서 사용한다. 이식 사본의 코드/asset/모션을 읽고 결과만 사본 `output_etc/check/` 아래 새 폴더에 저장한다. 기존 출력 폴더는 덮어쓰지 않는다.

## 점과 plane의 의미

둘은 정책 이름이 아니라 **같은 궤적의 성공 판정 기준**이다.

| 기준 | 성공 범위 |
| --- | --- |
| 점 | 검사점과 목표점의 3D 거리 ≤0.20m |
| 영역(plane) | 목표 상판의 yaw 좌표에서 각 축 내부80% AND 목표 높이와 world-Z 오차 ≤0.20m |

plane은 무한 평면이나 표면까지의 최단거리가 아니다. 내부80%는 `abs(local_x)≤0.4*Lx`, `abs(local_y)≤0.4*Ly`이므로 면적은64%다. 상자가 작으면 영역 기준이 더 엄격할 수도 있다. 몸/상자 전체가 아니라 골반/상자 중심점을 검사한다. 접촉·발 높이·손 놓기·속도·유지시간 조건은 성공식에 추가하지 않는다.

## 기존 네 과제: 두 모델 비교

모델은 가져온 Stage1V2Fixed **73000회**, 기존 과제+PUSH fine-tuning **22500회**다. 공개 TokenHSI 정책을 실행하는 bridge는 이번 구현 범위에 포함하지 않는다. 'TokenHSI식 점'은 성공식의 명칭이며 공개 논문의 원래 benchmark 전체 재현을 뜻하지 않는다.

| 과제 | 검사점 | 목표 높이 | 영역 기준 |
| --- | --- | --- | --- |
| AT | payload 중심 | 샘플링한 목표 중심 Z | 점 기준과 동일 |
| ON_TOP | payload 중심 | 현재 받침의 회전 bbox 최고점+payload 직립 반높이 | 현재 받침 내부80%+Z±20cm |
| SIT | 골반/root | 현재 상판+10cm | 상판 내부80%+Z±20cm |
| CLIMB | 골반/root | 현재 상판+94cm | 상판 내부80%+Z±20cm |

각 과제 환경당1명, AT/SIT/CLIMB 물체1개·ON_TOP 물체2개다. 보조 선반 actor는 물체 수에 포함하지 않는다. loco100%, 512환경×seed0/1/2, 900step(1/30초 제어, 종료 경계899step)이며 첫 episode의 terminating frame까지 ever를 집계한다. 낙상 종료는 기존 humanoid 구현을 따른다. 성공 후에도 episode를 끝내지 않는다. 종료된 환경은 재초기화하지 않고 이후 frame은 채점에서 제외한다.

크기는 AT9종을 최소1개씩 포함한 균등 복원 추출 후 shuffle, 나머지는 문서의 XYZ 독립5cm격자 범위를 사용한다. 사람→출발 상자 중심 XY 거리는 SIT1~5m/나머지1~10m 반경 균등이다. AT 목표·ON_TOP 받침 XY는 환경 origin±4.5m 사각형에서 사람 및 payload로부터 각각 최소1m를 만족하도록 rejection sampling한다. ON_TOP을 일렬로 만들거나 표면 거리로 바꾸지 않는다. 두 물체의 초기 yaw도 무작위다.

운반 출발/AT 도착 높이는 원본 clamp/50%높임/출발5cm 여유와 고정 선반 warmup을 유지한다. SIT/CLIMB는 바닥 배치·고정 상자, ON_TOP 받침은 동적이다. 동적 운반 물체는 밀도100kg/m³, 바닥마찰1.0과 저장된 sim 조건을 사용한다. 학습용 추가 bbox/RSI scene rejection은 평가 subclass에서 적용하지 않으며 문서의 중심 거리 rejection을 사용한다.

같은 seed의 두 모델은 root/모든 DOF 위치·속도/상자/목표/크기/binding/질량의 **초기 배열을 정확 비교**한다. 정책 행동은 이후 달라진다. 입력은 해당 모델의 scene RMS를 고정하고 모든 가중치는 strict load, deterministic actor mean으로 추론한다. 학습된 2명/4물체 모델을 네트워크가 지원하는1명/1~2물체로 실행하며 가중치를 변환·미세조정하지 않는다.

## PUSH: 승인한 별도 프로토콜

ID: `single_push_point_region_v1`. 기존 문서에는 PUSH가 없으므로 별도 프로토콜로 보고한다. 73000회 모델에는 PUSH token이 없어 이 과제는 **22500회 모델만** 실행한다. CARRY를 PUSH라고 바꿔 baseline으로 보고하지 않는다.

| 항목 | 조건 |
| --- | --- |
| 환경·시간·통계 | 1명·동적상자1개, 최대30초/900step,512환경×3seed,첫 episode ever,seed별 mean±표본std |
| 시작 | 주 평가 loco100%, 보조 평가 raw PUSH RSI100%; 결과 분리 |
| 목표 거리 | 실제 초기 상자 중심에서2~5m /5~10m, 별도 표. 긴 구간을 자동 OOD라 부르지 않음 |
| 방향 | 초기 상자 로컬+Y 면 법선, yaw는0~2π 균등 |
| 점 기준 | 상자 중심→목표 중심 3D거리≤20cm |
| 영역 기준 | **고정1m×1m 목표 사각형 내부80%**, 목표 중심Z±20cm |
| 목표 구역 | 상자 크기와 독립; 초기 상자 yaw를 따른 고정 구역. 상자 이동/회전으로 목표 구역이 움직이지 않음 |
| 목표 Z | 직립 상자 반높이. 현재 상자 Z를 따라 움직이지 않음 |
| 상자 물성 | 가져온 설정 그대로: X1.2~1.7/Y1.45~1.7/Z1.45~1.7m,Y≥Z,질량15~20kg 독립균등,상자마찰0.3/바닥1.0 |
| loco 시작 | root→상자 중심 반경1.65~6m·360°균등, OBB까지 최소0.45m를 rejection, 상자 방향으로 root yaw 정렬 |
| RSI 시작 | 가져온 raw bank/클립 가중치·후반확률0.7·후반pool30%·원본관절/속도 유지, yaw/XY 정렬과 원래 collider 후퇴 적용 |
| RSI 목표 | RSI의 box_progress로 상자를 미리 옮기지 않음. **실제 초기 상자에서** 목표거리2~5/5~10m 보장 |
| 보조지표 | 첫 도달 시간(성공 episode만 평균), 마지막/최선 목표거리 감소, 최종 남은거리,상자기울기45°초과 ever 전도율 |

영역 기준은 상자 중심만 검사하며 상자 전체가 목표 구역에 들어가야 하는 조건이 아니다. 전도·접촉·손 놓기·정지·유지시간은 주 성공식의 gate가 아니다. 특히 쓰러지며 통과할 수도 있으므로 전도율을 함께 읽는다. 두 기준의 첫 도달시간은 각각 저장한다.

현재 레포의 RSI손중점≤1m/실제밀기구간 재선별 기준과 달리, 이 평가는 **원본 raw bank를 유지한 외부 설정**이다. 새 RSI 선별/팔 보정/전수 PD 안정성 주장을 하지 않는다. 원본 bank SHA와 모션 hash는 검사한다. 원본과 달리1인 평가이고 구역/거리 프로토콜을 통일했으므로 기존2인 PUSH viewer 성공률과 같은 실험으로 취급하지 않는다.

PUSH에서도 물리 환경은 평가 subclass를 사용한다. 내부 reset/reward scaffolding은 기존 AT graph를 재사용하되 **정책에 전달하는 packet은 실제 PUSH token4**다. 실제 PUSH 상자·물성·raw RSI·목표·관측을 사용하고, reward/학습 성공flag는 채점에 사용하지 않는다. 단독 네트워크는 strict load한5-task/4-family 구조를 유지한다.

## 명령

저장소 루트에서 실행한다. 기본 conda `tokenhsi`, 기본 GPU6. 실행 전 `nvidia-smi`로 점유를 확인한다. **아래 전체 평가는 준비만 했으며 자동 실행하지 않았다.** `output`은 사본 `output_etc/check/` 아래 존재하지 않는 경로를 지정한다.

```bash
# 두 모델×네 과제 비교 + PUSH loco/RSI×두 거리, 각각512환경×3seed
TOKENHSI_GPU=6 /home/hwanhee/anaconda3/envs/tokenhsi/bin/python evaluation/single_match/batch.py --suite all --output ports/jhh_stage1_push_raw_cuboid/output_etc/check/single_eval_full

# 기존 네 과제만 두 모델 비교
TOKENHSI_GPU=6 /home/hwanhee/anaconda3/envs/tokenhsi/bin/python evaluation/single_match/batch.py --suite retention --output ports/jhh_stage1_push_raw_cuboid/output_etc/check/single_retention_full

# PUSH만 loco/RSI 및2~5/5~10m 분리
TOKENHSI_GPU=6 /home/hwanhee/anaconda3/envs/tokenhsi/bin/python evaluation/single_match/batch.py --suite push --output ports/jhh_stage1_push_raw_cuboid/output_etc/check/single_push_full

# 짧은 실행경로 확인: 각4평가환경·12step·seed0, 성능평가 아님
TOKENHSI_GPU=6 /home/hwanhee/anaconda3/envs/tokenhsi/bin/python evaluation/single_match/batch.py --suite all --smoke --seeds 0 --output ports/jhh_stage1_push_raw_cuboid/output_etc/check/single_eval_smoke_new

# 개별 headless 실행:22500회 PUSH·loco·2~5m·512환경·seed0
TOKENHSI_GPU=6 bash evaluation/single_match/run.sh --checkpoint ports/jhh_stage1_push_raw_cuboid/output/stage1_v2_fixed_push_360_raw_cuboid/Stage1V2FixedPush360RawCuboid_21-16-00-23/nn/Stage1V2FixedPush360RawCuboid_00022500.pth --policy push_22500 --task push --start loco --seed 0 --output ports/jhh_stage1_push_raw_cuboid/output_etc/check/push_loco_seed0

# 로컬 viewer:1평가환경·900step, 표에 포함하지 않는 별도 시각화
TOKENHSI_GPU=6 bash evaluation/single_match/run.sh --checkpoint ports/jhh_stage1_push_raw_cuboid/output/stage1_v2_fixed_push_360_raw_cuboid/Stage1V2FixedPush360RawCuboid_21-16-00-23/nn/Stage1V2FixedPush360RawCuboid_00022500.pth --task push --start loco --num-envs 1 --steps 900 --smoke --viewer --output ports/jhh_stage1_push_raw_cuboid/output_etc/check/push_local_view

# 서버 VNC:1평가환경·900step·RSI, 표에 포함하지 않는 별도 시각화
TOKENHSI_GPU=6 PORT=6127 bash evaluation/single_match/vnc.sh --checkpoint ports/jhh_stage1_push_raw_cuboid/output/stage1_v2_fixed_push_360_raw_cuboid/Stage1V2FixedPush360RawCuboid_21-16-00-23/nn/Stage1V2FixedPush360RawCuboid_00022500.pth --task push --start rsi --num-envs 1 --steps 900 --smoke --output ports/jhh_stage1_push_raw_cuboid/output_etc/check/push_vnc_view

# 저장된 개별 run의 궤적 독립 재채점(GPU 불필요)
/home/hwanhee/anaconda3/envs/tokenhsi/bin/python evaluation/single_match/rescore.py ports/jhh_stage1_push_raw_cuboid/output_etc/check/push_loco_seed0

# CPU 경계/회전/ever/통계 검사
/home/hwanhee/anaconda3/envs/tokenhsi/bin/python -m pytest -q evaluation/single_match/test_metrics.py
```

개별 실행의 `--task`는 `at/ontop/sit/climb/push`, PUSH는 `--start loco/rsi`, `--distance-min 5 --distance-max 10`으로 긴 구간을 선택한다. batch는 두 체크포인트 기본 경로를 내장하고 `--stage1-checkpoint`/`--push-checkpoint`로 명시적으로 대체할 수 있다. 모델은 원본 학습을 따라 자동 갱신하지 않는다. VNC는 SSH 포워딩 후 `http://localhost:6127/vnc.html`이다. 화면에 표시하는 원래 goal marker는 중심점이며 영역 판정식은 별도 scorer를 따른다.

## 산출물과 검증 범위

- run별 `initial_0.npz`: root·DOF위치/속도·상자·크기·목표·binding·질량·origin. PUSH는 고정 구역 yaw 포함.
- `trace_0.npz`: 매 action 후 root/상자/목표/active/done/terminate. 첫 episode 종료 frame 포함. 학습 성공flag를 결과로 복사하지 않는다.
- `resolved_env.yaml`, `resolved_train.yaml`, `manifest.json`, `completed.json`: 실제 설정·epoch/checkpoint SHA·평가 코드 SHA·산출물 SHA·검사 결과.
- `results.json`: 각 기준 환경별 ever/성공 수/첫 도달시간, PUSH 보조지표.
- batch의 `summary.json`/`REPORT.md`: seed별 성공률의 mean±**표본std**. 네 과제 평균은 seed 안에서 네 과제를 같은 가중치로 평균한 뒤 seed 간 mean/std를 계산한다. PUSH 시작/거리 조건을 한 숫자로 섞지 않는다. 1seed smoke는 std를 만들지 않는다.

검증은 CPU6개, GPU6의 두 모델×네 과제 및 PUSH2시작×2거리(각4환경·12step·seed0) 정상 종료, 네 과제 정책 간 전체 초기 DOF 포함 배열 일치를 확인했다. 12개 저장 궤적의 독립 재채점·PUSH 실제 초기거리/목표고정 검사도 통과했다. 추가4환경 RSI의 실제 mass/모든 shape마찰 조회·설정 대조 및 VNC6127의1환경 RSI·12step 정상 종료를 확인했다. 저장한 궤적을 float64로 채점하고 float32와 환경별 ever가 일치하는지 검사한다. 결과는 사본 `output_etc/check/single_match_20261023/`에 있다. full512×3seed 성공률은 아직 측정하지 않았다. 짧은 확인은 장기 PUSH 안정성/성공률 검증이 아니다.
