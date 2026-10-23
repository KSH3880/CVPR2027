# 39000 distill 객체 종류 제로샷

추가 학습·가중치 전이 없이 지정한 KSH `ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingDistill_00039000.pth`를 사용한다. 앞선 native 객체 전체 진단은 사용자 요청으로 중단했으며, 후속 요청의 **SIT·의자 CLIMB 좌판 입력 보정 비교**는 아래 별도 절을 따른다. viewer는 자동 실행하지 않는다. KSH 원본 코드·YAML·checkpoint는 읽기 전용으로 사용한다. 현재 레포의 `evaluate_object_zero_shot.py`가 자식 프로세스 안에서만 객체 생성·고정 task 샘플링·판정 및 선택적 정책 입력을 확장한다. bytecode 저장을 끄고, 결과·asset 사본은 현재 레포의 새 output 아래에 저장한다. 실행 전후 원본 코드/설정 및 checkpoint SHA256을 비교한다. 별도 학습용 YAML/train 스크립트는 없으며 학습에는 사용할 수 없다.

## 사용자용 객체별 시각화

### 상자 마찰0.6·세 과제 한 장면씩 자동 녹화

사용자가 직접 실행한다. 지정39000 checkpoint·원래 상자 크기/질량·바닥1.0을 유지하며 `--box-friction 0.6`은 평가 프로세스의 실제 상자 shape에만 적용하고 `friction_override.json`에 재조회값을 남긴다. 옵션을 생략하면 원본 마찰을 유지한다.

```bash
TOKENHSI_GPU=6 PORT=6137 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh \
  --cases baseline_carry baseline_sit baseline_climb --box-friction 0.6 --envs 1 \
  --scene-mode loop --viewer-episodes 1 --episode-length 600 --record-video \
  --output output/box_three_tasks_friction06_record
```

CARRY→SIT→CLIMB 순서로 각 한 장면을 최대600step(약20초), 종료 조건이면 그때까지 녹화하고 전체 종료한다. `--record-video`는 실제 viewer 카메라/UI를 PNG로 읽어 streaming MP4로 저장하며 각 case 하위 `scene_001.mp4`를 만든다. Space로 일시정지한 시간은 영상에서 제외된다. 정상 종료/예외에는 writer를 마감하며 강제종료는 보장하지 않는다. 기존 output은 새 run 폴더로 분리한다. 시뮬레이션 시간 FPS를 사용하므로 캡처 지연으로 실제 실행은 길어질 수 있다.

검증: 관련 CPU17개·Python 문법 통과. 녹화 GPU/GUI 실행은 사용자 요청에 따라 하지 않았으므로 실제 MP4 생성/재생은 미검증이다. 마찰 옵션을 사용한 앞선 CARRY viewer는 정책 step90 이상 실행 로그를 확인했으며 사용자 요청으로 종료했다.


기본 `--viewer-style native --viewer-camera stack`은 KSH 원본 Isaac Gym renderer·UI·카메라·F/V/K/L 단축키를 그대로 사용한다. 추가 키만 proxy에서 처리하며 KSH 파일은 수정하지 않는다. 이전에 제공한 근접 카메라는 `--viewer-style focus`로 선택한다. `N`/`R` 입력 시 터미널에 `reset requested`가 찍히고, 명시적인 환경 row를 reset한 뒤 `scene=N reset complete`를 출력한다. N/R은 **다른 객체 모델로 교체하는 키가 아니라 같은 객체의 배치 재샘플링**이다. 로그가 없다면 noVNC의 실제 시뮬레이터 화면을 클릭해 키보드 포커스를 준 뒤 다시 누른다.

키에 의존하지 않는 기존 자동 반복 방식:

```bash
TOKENHSI_GPU=1 PORT=6123 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh --cases sit_chair --envs 1 --viewer-style native --viewer-camera stack --scene-mode loop --episode-length 600 --output output/obj_view_sit_chair
```

이 명령은20초 상한 또는 넘어짐/종료 시 자동으로 같은 객체의 새 배치를 재생한다. native viewer의 이벤트 전달·N/R press/release 유지 CPU 회귀 포함9개와 문법 검사를 통과했다. GUI는 사용자 요청대로 자동 실행하지 않았다.

CLIMB를 SIT용 StraightChair_Normal에 제로샷으로 적용하려면 다음 명령을 사용한다. 입력은 의자 전체 bbox이며 기존 CLIMB의 bbox 최상단(등받이 높이) 기준을 그대로 사용한다. 좌판 높이를 별도로 정책에 알려주거나 학습하지 않는다.

```bash
TOKENHSI_GPU=1 PORT=6123 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh --cases climb_chair --envs 1 --viewer-style native --viewer-camera stack --scene-mode loop --episode-length 600 --output output/obj_view_climb_chair
```

각 명령을 별도 터미널에서 실행한다. GPU1·1환경이며 지정39000을 자동 사용한다. VNC wrapper는 `--visualize`를 붙여 성공률 집계 없이 실시간 재생만 한다. 기본 `--scene-mode manual`은 **한 장면을18000step(약10분) 재생한 뒤 화면을 유지**한다. 넘어지거나 성공해도 자동 재시작하지 않는다. `Space`는 일시정지/재개, `N`/`R`은 새 배치, `F`는 카메라 추적 토글, `Esc`/터미널 `Ctrl+C`는 종료다. 자동 반복이 필요할 때만 `--scene-mode loop`를 지정한다. 기본 stack 카메라는 기존처럼 전체 장면을 보여주며 `--viewer-camera agent`는 원래 agent 추적을 사용한다.

```bash
cd /home/hwanhee/koo_cvpr/stage1_fix
# SIT: StraightChair_Normal
TOKENHSI_GPU=1 PORT=6123 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh --output output/obj_view_sit_chair --cases sit_chair --envs 1 --scene-mode manual --episode-length 18000
# CLIMB: Box / Cabinet / Table_Square
TOKENHSI_GPU=1 PORT=6124 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh --output output/obj_view_climb_box --cases baseline_climb --envs 1
TOKENHSI_GPU=1 PORT=6125 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh --output output/obj_view_climb_cabinet --cases climb_cabinet --envs 1
TOKENHSI_GPU=1 PORT=6126 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh --output output/obj_view_climb_table --cases climb_table --envs 1
# CARRY: Box / ArmChair_Normal / Table_Circle
TOKENHSI_GPU=1 PORT=6127 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh --output output/obj_view_carry_box --cases baseline_carry --envs 1
TOKENHSI_GPU=1 PORT=6128 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh --output output/obj_view_carry_chair --cases carry_chair --envs 1
TOKENHSI_GPU=1 PORT=6129 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh --output output/obj_view_carry_table --cases carry_table --envs 1
```

VS Code PORTS에서 해당 포트를 포워딩한 뒤 `http://localhost:6123/vnc.html?autoconnect=1&resize=remote`처럼 접속한다. wrapper는 localhost에만 바인딩한다. 기본은 ID 정렬 첫 객체이며 다른 instance는 `--asset-index 1` 등을 붙인다. asset는 scene 생성 시 확정되므로 `N`은 같은 객체에서 배치/동작을 다시 샘플링한다. 시각화 output이 이미 있으면 `_run001`, `_run002`처럼 새 폴더를 자동 선택해 기존 기록을 보존하므로 같은 명령을 재사용할 수 있다. 정량 평가는 여전히 새 output을 요구한다. Box 대조군은 학습 분포의 기존 박스 생성 경로를 사용한다. `--scene-mode loop --viewer-episodes 1`은 한 에피소드 후 종료한다. 로컬 viewer는 같은 인자에 `HEADLESS=0 ...object_zero_shot_test.sh --visualize`를 사용한다. 실제 viewer 실행 검증은 사용자 요청으로 하지 않았다.

사용자가 제공한 viewer 로그의 실제 실패는 `env_step`이 반환한 Tensor를 다음 step에서 CommonPlayer의 dict 입력으로 넘기지 않아 발생한 `IndexError`였다. 재생 모드는 reset/step 관측을 모두 `{'obs': Tensor}`로 맞춘다. CPU7개(관측 계약·재실행 기록 보존 포함)와 문법 검사를 통과했으며 수정 후 GPU/GUI 실행은 하지 않았다.

## 입력과 물리

- 원래 네트워크·정규화·metadata 검사와 2명/4물체/634D 관측을 유지한다. 물체당 선속도3·각속도3·실제 전체 bbox8점24 = 30D이며 위치/회전은 기존 GTA pose에 들어간다. 메시 정점·카테고리·좌판 위치를 정책에 추가하지 않는다.
- source 슬롯0/1을 실제 URDF 메시·VHACD 충돌체로 교체한다. 나머지2개는 원래 생성된 박스다. 모든 객체는 원래 multi-agent 박스처럼 움직일 수 있으며 단일 rigid body/관절0을 요구한다. 원본 데이터 asset는 수정하지 않는다.
- 스케일1·원래 metadata bbox와 URDF 질량을 사용한다. 크기 sweep/정규화나 재학습은 하지 않는다. 따라서 카테고리 변경에는 데이터 원래 크기·질량 차이도 포함된다. 형상만의 효과를 고립시킨 실험은 아니다.
- 보행(loco) 시작·seed42·600step을 기본으로 한다. 학습 크기 범위 밖 객체도 고정 task로 평가하도록, 자식 프로세스 안에서 학습용 크기별 task eligibility만 해제한다. 원본 크기를 학습 범위에 맞춰 축소하지 않는다.
- 첫 rollout 전에 mesh actor를 공중으로 이동해2 physics tick 동안 이전 접촉을 비운 뒤, 사람/물체 전체 root와 DOF 위치·속도를 원래 시작 상태로 복원한다. 이2tick은 정책 rollout/episode 시간에 포함하지 않는다. 비영 COM을 가진 convex compound를 GPU PhysX에서 teleport한 직후 생기는 초기 접촉 충격을 막기 위한 평가 초기화다. 부분 종료된 장면은 원래 평가기처럼 reset하지만 해당 장면의 후속 결과는 집계하지 않는다.

## 대상

| case | task | `tokenhsi/data/` 아래 test 객체 | 개수 |
| --- | --- | --- | ---: |
| `sit_chair` | sit | `dataset_sit/objects/test/StraightChair_Normal` | 26 |
| `climb_cabinet` | climb | `dataset_amass_climb/objects/test/Cabinet` | 9 |
| `climb_table` | climb | `dataset_amass_climb/objects/test/Table_Square` | 8 |
| `carry_chair` | carry_at | `dataset_dynamic_objects/test/ArmChair_Normal` | 27 |
| `carry_table` | carry_at | `dataset_dynamic_objects/test/Table_Circle` | 9 |

기본은 각 카테고리의 ID 정렬 첫 객체1개를 사용한 pilot이다. `--assets-per-category 0`은 전체 test 객체를 사용하며, 두 source 슬롯에 순환 배정한다. 모든 객체를 최소 한 번 포함하려면 `2 × envs ≥ 선택 객체 수`여야 한다. 배정 ID는 `trials.json`에 기록한다. 객체별 표본 수가 완전히 같지는 않으므로 전체 평균만으로 객체별 성능을 판단하지 않는다. 선택적 `climb_box`는 데이터의 Box test URDF를 사용한다.

## 성공 판정

SIT는 정책 bbox를 유지하고 판정에만 좌판 높이를 적용한다. 데이터 생성기의 `tarSitPos`에는 좌판 높이+0.10m가 들어 있으므로 0.10m를 빼고 학습 때의 pelvis clearance0.12m를 적용한다. XY 영역은 기존 전체 bbox의 inner margin을 유지한다. 정확한 좌판 polygon/접촉 검사는 아니며, 등받이 높이를 목표로 삼는 오류만 방지한다. 원래 전체 bbox 최상단 판정도 `bbox_rule_*`로 병기한다. CLIMB와 CARRY_AT는 기존 성공 조건을 그대로 사용한다.

CARRY_AT의 기존 성공은 목표 도달 판정이며 실제 운반을 보장하지 않는다. `max_bbox_bottom_m`, `elevated_xy_travel_m`으로 바닥에서5cm 이상 떨어진 적이 있는지와 그 상태에서 이동한 XY 길이를 별도 기록한다. 튀어 오른 경우도 포함될 수 있으므로 이것만으로 손으로 들었다고 판정하지 않는다.

## 명령

저장소 루트에서 실행한다. 사용자 지정 GPU는1이며 기존 학습을 종료/재시작하지 않는다. output은 매번 새 경로를 지정한다.

```bash
nvidia-smi -i 1
# 대표 객체5종, 짧은 실행 경로 확인: 성공률 benchmark 아님
TOKENHSI_GPU=1 HEADLESS=1 bash tokenhsi/scripts/multi_agent/object_zero_shot_test.sh --output output/object_zero_shot_smoke_new --envs 2 --episode-length 12
# 기본 대표 객체1개/카테고리, 32scene × 2명 × 3반복
TOKENHSI_GPU=1 HEADLESS=1 bash tokenhsi/scripts/multi_agent/object_zero_shot_test.sh --output output/object_zero_shot_pilot_new --envs 32 --repeats 3
# 전체 test 객체79개, 32scene × 2명/카테고리
TOKENHSI_GPU=1 HEADLESS=1 bash tokenhsi/scripts/multi_agent/object_zero_shot_test.sh --output output/object_zero_shot_full_new --assets-per-category 0 --envs 32
# 로컬 viewer / 서버 VNC: 한 카테고리 대표 객체
TOKENHSI_GPU=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/object_zero_shot_test.sh --output output/object_zero_shot_chair_local_new --cases sit_chair --envs 1 --visualize
TOKENHSI_GPU=1 PORT=6123 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh --output output/object_zero_shot_chair_vnc_new --cases sit_chair --envs 1
```

`manifest.json`에 원본 해시·크기·질량·배정 대상·프로토콜, `runtime_audit.json`에 실제 관측 차원·충돌체 수·물성, `initial_state.json`/`first_steps_repeat_*.json`에 초기 상태·접촉 반력, `trials.json`에 개별 결과, `summary.json`에 집계를 저장한다. `integrity.json`은 원본 무변경 검사 결과다. 실패하면 해당 case의 `.log`와 `statuses.json`을 확인한다. 실행 시 코드를 읽으므로 `/home/hwanhee/ksh/stage1_task`가 필요하다.

## 좌판 입력 보정 SIT·CLIMB

2026-10-15 후속 요청으로 SIT와 의자 CLIMB의 **비학습 보정 비교**를 추가했다. 앞선 정량 평가 중단 기록과 별도의 새 실험이다. 지정39000 checkpoint·정규화·실제 의자 mesh/충돌체·질량·관성·물리 위치는 그대로 사용한다. KSH 원본을 수정하지 않는다. 별도 학습 config/train script는 없다.

`--policy-bbox full`(기존 기본), `seat-height`, `seat-inset`을 지원한다. 좌판 높이는 `tarSitPos.z - 0.10`이며 가상 상자는 원래 bbox 바닥부터 좌판까지다. metadata XY 중심도 사용한다. 물체30D의 bbox/가상 중심 선속도와 GTA origin을 함께 보정해 좌표계를 일치시킨다. 관측634D·RMS는 그대로이며 물리 state/asset는 바꾸지 않는다. 실제 등받이·팔걸이는 충돌체로 남지만 보정 bbox에서 보이지 않는다는 한계가 있다.

`seat-inset`은 XY를 기본0.75배(`--seat-xy-fraction`)로 줄이는 **근사 실험**이다. 정확한 좌판 폭·등받이 경계 metadata가 아니다. `--climb-surface seat`는 CLIMB 평가 목표를 등받이 최상단 대신 좌판으로 옮긴다. 기존 `full` 판정과 다른 과제 정의이며 새 비교의 세 조건에 모두 동일 적용한다. SIT·CLIMB 판정 XY는 전체 bbox 폭 기반으로 유지하므로 정확한 좌판 polygon/접촉 성공으로 해석하지 않는다. 기존 옵션 없는 시각화는 전체 bbox/기존 CLIMB 최상단 판정을 유지한다.

```bash
# 의자26종·32scene/64agent·600step·seed42, 세 입력 조건×두 과제 (GPU6)
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi PYTHONDONTWRITEBYTECODE=1 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
python tokenhsi/scripts/multi_agent/compare_seat_proxy.py --output output/seat_proxy_comparison_gpu6_new --envs 32 --assets-per-category 0 --episode-length 600'

# 한 조건만 headless 평가
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/object_zero_shot_test.sh --cases sit_chair climb_chair --policy-bbox seat-height --climb-surface seat --envs 32 --output output/seat_height_eval_new

# 로컬 viewer (원하는 조건으로 --policy-bbox를 바꾼다)
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/object_zero_shot_test.sh --cases sit_chair --policy-bbox seat-height --envs 1 --visualize --scene-mode loop --episode-length 600 --output output/seat_sit_local
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/object_zero_shot_test.sh --cases climb_chair --policy-bbox seat-height --climb-surface seat --envs 1 --visualize --scene-mode loop --episode-length 600 --output output/seat_climb_local

# 서버 VNC (사용자가 실행, 기존 viewer/process는 종료하지 않음)
TOKENHSI_GPU=6 PORT=6124 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh --cases sit_chair --policy-bbox seat-height --envs 1 --scene-mode loop --episode-length 600 --output output/seat_sit_vnc
TOKENHSI_GPU=6 PORT=6125 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh --cases climb_chair --policy-bbox seat-height --climb-surface seat --envs 1 --scene-mode loop --episode-length 600 --output output/seat_climb_vnc
```

`comparison.json`은 전체/객체별 final·ever 성공률과 종료율을 저장하고, 세 조건의 초기 root/물체/goal 및 물성 감사가 같은지 검사한다. `foot_above_seat_15cm_fraction`은 물체 local XY bbox 안에서 한쪽 이상 발이 좌판보다15cm 높았던 생존 step 비율이다. 정상적인 다리 들기도 포함하므로 등받이 **접촉** 비율이 아니다. 각 조건의 `policy_proxy_audit.json`에 실제 보정 열·크기·가상 pose를 기록한다. 박스 원래 판정(`bbox_rule_*`)과 좌판 판정을 구분한다. 한 seed·26종/64agent의 초기 비교이며 객체마다2~3표본이므로 per-asset 결과를 안정적인 순위로 취급하지 않는다.

2026-10-15 실행 완료: [비교 JSON](../output/seat_proxy_comparison_gpu6/comparison.json). GPU6/tokenhsi·위32scene/64agent/600step/seed42 조건이며 재학습하지 않았다.

| 정책 입력 | SIT 종료 / 도중 한 번 이상 | CLIMB 종료 / 도중 한 번 이상 |
| --- | --- | --- |
| 원래 전체 bbox | 7.81% / 28.13% | 1.56% / 46.88% |
| 좌판 높이 보정 | 40.63% / 59.38% | 12.50% / 45.31% |
| 좌판 높이·XY0.75배 | 48.44% / 64.06% | 10.94% / 40.63% |

세 조건의 초기상태·실제 물성/충돌체 감사가 완전일치했고 원본 코드/YAML·checkpoint hash도 불변이다. CPU11개 및 앞선2환경12step 보정 추론도 통과했다. SIT는 입력 보정으로 개선됐지만, CLIMB는 좌판에서 종료 시 성공 유지가 여전히 낮다. CLIMB 두 보정의 차이는64표본중1개뿐이므로 우열을 확정하지 않는다. 높은 발 비율은 CLIMB4.50→3.89→3.17%였으나 등받이 접촉을 직접 식별한 결과는 아니다. GUI 동작은 아직 검증하지 않았다. SIT 최고 조건을 보려면 위 viewer 명령의 `seat-height`를 `seat-inset`으로 바꾼다. 최종 목표는 실제 mesh에서 좌판 지지·안정적 자세이며, 이 근사 판정 성공률만으로 등받이 문제가 해결됐다고 결론내리지 않는다.

## CLIMB 올라가기 사건 평가

사용자 후속 요청에 따라 `--climb-ascent`를 추가했다. **주지표 `ascent_ever_rate`는 에피소드 중 한 번이라도 아래 조건을 동시에 만족한 agent 비율**이다. 중앙 root 목표 위치·정확한 목표 root 높이·종료까지 유지하는 조건을 요구하지 않는다. 기존 `final_success_rate`와 `ever_success_rate`도 그대로 저장해 비교한다. 정책 입력·보상·reset/종료 조건은 변경하지 않으며 학습하지 않는다.

- 양발 body origin이 지정 물체의 local bbox 전체 XY 안에 있고, 두 발 모두 native 윗면 높이에서±7cm 이내다.
- 골반이 같은 윗면보다45cm 이상 높고, 물체 local Z와 world Z의 내적이0.9 이상이다. 넘어진 물체의 옆면에 올라간 경우와 누운 자세를 제외한다.
- 양발 net contact force의 양의 world-Z 성분 합이20N 이상이다. 이는 윗면 geometry와 net foot contact를 결합한 판정이며 접촉 상대 actor ID를 직접 식별하지는 않는다.

`ascent_geometry_ever_rate`는 접촉력 조건을 뺀 geometry 사건 비율, `ascent_3steps_rate`는 위 접촉 포함 조건을 연속3정책step(약0.1초) 만족한 비율이다. `trials.json`에 최초 성공 step과 최대 연속 step도 저장한다. 기존 성공 조건과 발 XY/실제 접촉 요구가 달라 새 지표가 반드시 기존 ever 성공률보다 높아야 하는 것은 아니다. bbox 윗면을 대표 지지면으로 쓸 수 있는 Box/Cabinet/Table_Square를 대상으로 하며, 복잡한 메시의 정확한 지지면 annotation 판정은 아니다.

```bash
# 39000·보정 없는 전체 bbox 입력·원래 크기/질량·보행 시작
# Box18종/Cabinet9종/Table_Square8종, 종류별32scene/64agent·600step·seed42
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/object_zero_shot_test.sh --cases climb_box climb_cabinet climb_table --climb-ascent --envs 32 --assets-per-category 0 --episode-length 600 --output output/climb_ascent_39000_gpu6_new

# 로컬 / 서버 viewer (이 모드는 성공률 집계 없는 시각화)
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/object_zero_shot_test.sh --cases climb_box --envs 1 --visualize --scene-mode loop --episode-length 600 --output output/climb_ascent_box_local
TOKENHSI_GPU=6 PORT=6126 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh --cases climb_box --envs 1 --scene-mode loop --episode-length 600 --output output/climb_ascent_box_vnc
```

## CLIMB 학습 크기 범위 내 비교

`compare_climb_size_range.py`는 저장된 학습 YAML의 `env.box.build.taskSizeRanges.CLIMB`를 읽어 X/Y0.4~0.8m·Z0.25~0.5m에 **원래 크기가 들어가는 물체만** 평가한다. 범위는 축별 포함 경계이며 학습의5cm 샘플 grid와 정확히 일치하는 크기만 고르는 것은 아니다. 해당 native test 객체는 Box6종·Cabinet2종·Table_Square3종이다. 크기 변형/리스케일·학습·bbox 입력 보정 없이39000 모델과 앞 절의 올라가기 사건 판정을 사용한다.

대조군 `matched-box`는 같은 객체 ID/배정/실제 bbox 크기·URDF 질량으로 단순 상자를 만든다. 상자 density는 `native_mass / bbox_volume`이며 geometry에 따른 inertia/COM은 같게 강제하지 않는다. 따라서 bbox/질량/마찰을 통제한 형상·관성 차이 비교다. 각 arm의 `physical_sources_by_asset`에 모든 선택 객체의 실제 mass/friction/COM/inertia를 기록하고, `comparison.json`에서 초기 root/물체/goal·bbox·질량·마찰 및 원본 무변경을 검사한다. 대상 subset의 객체 identity가 제한되어 있어 전체 category나 크기 단독 인과효과를 대표하지 않는다.

```bash
# GPU6/tokenhsi·64scene/128agent×3종류×2geometry·600step·seed42
TOKENHSI_GPU=6 TOKENHSI_CONDA_ENV=tokenhsi PYTHONDONTWRITEBYTECODE=1 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh
python tokenhsi/scripts/multi_agent/compare_climb_size_range.py --output output/climb_training_range_matched_gpu6_new --envs 64 --episode-length 600'

# native 범위 내만 평가 / --object-geometry matched-box로 대조군 선택 가능
TOKENHSI_GPU=6 HEADLESS=1 bash tokenhsi/scripts/multi_agent/object_zero_shot_test.sh --cases climb_box climb_cabinet climb_table --climb-training-range --climb-ascent --assets-per-category 0 --envs 64 --episode-length 600 --output output/climb_in_range_native_new

# 로컬 / VNC 시각화: 범위 내 subset의 첫 객체, 크기는 원본 그대로
TOKENHSI_GPU=6 HEADLESS=0 bash tokenhsi/scripts/multi_agent/object_zero_shot_test.sh --cases climb_cabinet --climb-training-range --envs 1 --visualize --scene-mode loop --episode-length 600 --output output/climb_in_range_local
TOKENHSI_GPU=6 PORT=6127 bash tokenhsi/scripts/multi_agent/object_zero_shot_vnc.sh --cases climb_cabinet --climb-training-range --envs 1 --scene-mode loop --episode-length 600 --output output/climb_in_range_vnc
```

2026-10-15 GPU6/tokenhsi 실행 완료: [분석 보고서](../output/climb_training_range_matched_gpu6/REPORT.md)·[수치/대조 검사](../output/climb_training_range_matched_gpu6/comparison.json).

| 물체 | 범위 내 native 올라가기 | 동일 크기·질량 상자 | Native 연속3step |
| --- | --- | --- | --- |
| Box6종 | 99.22% | 99.22% | 99.22% |
| Cabinet2종 | 99.22% | 97.66% | 99.22% |
| Table_Square3종 | 81.25% | 100% | 64.84% |

각128agent·seed42·600step. 크기/질량/마찰·초기상태 대조 전부통과, 원본 코드/YAML/checkpoint 불변·CPU17개·6worker rc0. Table은 native COM이 중심보다13~16.6cm 위이며 상자보다 수평 이동도 컸지만, 형상/COM/관성의 원인을 각각 분리하지는 않았다. 범위 내 subset에서도 모든 형상이 반드시 잘되는 것은 아니다. 단일 seed·제한된 객체 identity의 결과다.

## 기존 박스 평가

KSH의 `output/four_task_distill_39000_eval_20261012_gpu6/`에 39000 기준 결과가 있다. loco·seed42·600step·32scene/2명/1반복의 종료 시 goal 성공률은 SIT92.19%·CLIMB98.44%·CARRY_AT67.19%·CARRY_ONTOP92.19%다. 학습 분포의 크기가 다른 박스들에 대한 과거 평가이며, 새 객체와 동일 geometry/물성으로 맞춘 대조군은 아니다.

학습은 고정 크기가 아니다. `size_rsi.sample_sizes`가 5cm 간격의 과제별 크기를 환경 생성 시 샘플링한다. SIT XY0.25~0.80/Z0.35~0.50m, CLIMB XY0.40~0.80/Z0.25~0.50m, CARRY 세 축0.20~0.60m다. 각 환경의 asset 크기는 생성 후 고정된다.

검증: CPU5개 통과(좌판 판정과 입력 tensor 분리·회전·잘못된 metadata/다중 body/외부 output 거부). GPU1·2평가환경·대표 객체5종 각각12step에서 실제39000 로드·유한 관측/행동·634D/30D·충돌체 생성·정상 종료, KSH 코드/설정 및 checkpoint SHA256 불변 확인(`output/object_zero_shot_smoke_gpu1_v2/`). 원형 테이블은 접촉 캐시 초기화 후 같은12step에서 튀는 현상과5cm 들림이 사라졌고 원본 해시도 유지됐다(`output/object_zero_shot_cold_reset_gpu1/`). 사람 없는 생성/teleport 대조검사는 `check_object_zero_shot_physics.py`와 `output/object_zero_shot_carry_physics_gpu1/no_human_matching_physics.json`에 있다. 짧은 검사는 성공률 해석에 사용하지 않는다. 전체79객체·카테고리별32scene/64agent·600step/seed42 1회 결과와 원본 불변 검사는 `output/object_zero_shot_39000_full_gpu1/`에 보존했다. 이후 박스 대조/CARRY 진단 실행은 사용자 요청으로 중단했으며 완성된 benchmark로 취급하지 않는다. VNC wrapper와 재생 전용 모드는 사용자 요청으로 실행하지 않았다.
