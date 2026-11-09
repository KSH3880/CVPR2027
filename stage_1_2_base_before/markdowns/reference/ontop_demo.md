# ON_TOP 대기 성공 사례 저장과 재생

`ONTOP_DEMO_SEARCH=1 TASK_GRAPH=before_stack HEADLESS=1`을 전용 test wrapper에 전달하면 `MAPlayerContinuous.run_eval`에서 사례 탐색을 실행한다. 모델·보상·배치·행동은 변경하지 않는다. 기존 4명·8토큰 설정에서 두 BEFORE ON_TOP 쌍을 샘플링하며, 성공률 보고가 아닌 영상용 사례 선택이다.

사용자가 허용한 fallback은 `ONTOP_DEMO_RADIUS=3`이다. 각 사례에서 2m 구간을 먼저 찾고 없으면 3m를 사용하며, 다른 조건은 유지한다. 원래 2m 탐색과 별도 출력 폴더를 사용한다. 초기 탐색의 native HOLDING≥0.9(손 중심→상자 중심 약0.10m) 필터는 실제 운반을 과도하게 제외할 수 있어 제거했다. 사용자의 승인으로 초기 두 탐색을 종료하고, `output_etc/check/ontop_demo_physical/`에서 실제 높이·손 위치 기준으로 다시 탐색한다.

사용자가 맨손 대기 후 집어 올리는 흐름도 허용했다. 기본 판정은 후행 root와 선행 AT 최종 목표의 수평거리 2m 이내, root XY 속력 0.2m/s 이하가 연속 1초 이상이고, 그 구간 이후에 선행 AT가 **처음** 성공하는 경우다. 초기부터 선행이 성공한 경우는 제외한다. 그 뒤 native gated ON_TOP·AT 성공과 독립 spatial region 판정이 동시에 0.5초 이상 유지되어야 한다. 에피소드 종료·낙상 이후 상태는 인정하지 않는다. 저장 후 내려놓기와 화면도 확인한다.

`output_etc/check/ontop_demo_search/`의 각 `case_*`에 다음을 저장한다.

- `metadata.json`: checkpoint 경로·SHA256, seed, batch/환경/사람 ID, 제어 dt, 대기·선행 최초 성공·후행 성공 시점.
- `initial.pt`: 초기 전체 actor root·DOF, 크기, 논리 물체 매핑, task packet, 목표·환경 원점.
- `trajectory.npz`: 매 제어 프레임의 전체 root·DOF·사람 rigid body 상태, 선택한 쌍의 거리·속력·holding·gate·success·valid 판정.

`bash tokenhsi/scripts/multi_agent/replay_ontop_demo.sh <case_directory>`로 로컬 창을 연다. 선행은 파랑, 후행은 주황이며 해당 두 사람과 두 상자만 렌더링한다. 원래 시뮬레이션에서 다른 에이전트나 물체를 제거하지 않는다. 재생기는 저장된 상태를 표시하고 **물리 스텝이나 정책 추론을 새로 하지 않으므로**, 숨김이 원래 행동이나 접촉을 바꾸지 않는다. 저장된 시퀀스 끝에서 반복한다.

Space 일시정지/재개, R 처음부터, 좌우 화살표 프레임 이동, ESC 종료. `--speed 0.5`는 절반 속도다. `L` 키로 현재 카메라 화면의 녹화를 시작하고 다시 누르면 `<case_directory>/recordings/<시각>/frame_000000.png` 형식의 개별 PNG 이미지를 보존한다. MP4 변환이나 원본 이미지 삭제는 하지 않는다. ESC/창 닫기로 종료해도 저장한다. `R`로 처음부터 재생한 뒤 녹화할 수 있고 `--record`는 시작부터 녹화한다. 프레임 수와 재생 FPS는 `recording.json`에 기록하며 캡처 중 창의 실시간 속도가 느려질 수 있다. 이는 같은 초기 상태로 정책을 새로 추론하는 재현 실험과 구분한다.

선택 완료 사례는 `output_etc/check/ontop_demo_selected/case_b0_e38_f0`이며 `bash view_ontop_success.sh`로 바로 재생한다. 1.54~1.58m 거리에서5.7초 대기, 선행 최초성공6.6초, ON_TOP 최초성공9.73초 및0.5초 지속확인10.2초. 대기 중 맨손이며 선행 완료 후 집어 올려 쌓는다. `validation.json`에 CPU 재채점 결과가 있다. 렌더러 초기화 시 기록 적용 전1step만 수행하고 재생 루프는 물리스텝 없이 동작한다.

상자를 든 채 대기하는 별도 탐색은 `ONTOP_DEMO_REQUIRE_CARRY=1`로 요청한다. 이때만 상자 바닥>0.15m, 손 중심→상자 중심<0.55m를 추가로 요구하며, 영상에서 실제 파지를 확인해야 한다. `validate_ontop_demo.py`는 저장된 몸체/상자 상태로 거리·속력·파지 기하·독립spatial 판정을 CPU에서 다시 계산한다.

## CLIMB 동일 기준 사례

동일 체크포인트에서 `TASK_GRAPH=before_climb ONTOP_DEMO_SEARCH=1 ONTOP_DEMO_REQUIRE_CARRY=0`으로 탐색한다. 환경변수 이름은 기존 탐색 도구와 공용이다. CLIMB relation10과 선행 AT를 판정하며 대기 거리·속도·시간·동시 성공0.5초 기준은 동일하다. 재생기는 필요한 두 사람과 공유 상자 하나만 표시한다. CPU 재검증에서 선행 최초완료 이전의 CLIMB 공간 성공이 없어야 한다.

선택 사례 `output_etc/check/climb_demo_search/case_b0_e45_f0`: root 목표거리 최대1.62m에서5.63초 연속대기, 선행6.73초 완료, CLIMB 공간성공8.47초/연속성공 확인8.93초. `bash view_climb_success.sh`로 원래 기본 view 설정에서 재생한다. `L`은 사례 폴더 `recordings/<시각>/frame_*.png`로 이미지 저장, `R`은 처음부터, Space는 일시정지다. 탐색 실행 명령은 `output_etc/check/climb_demo_search/search.sh`, CPU 검증은 각 사례의 `validation.json`에 보존했다.

## 8명 랜덤 과제에서 BEFORE 전원 성공 사례

`RANDOM_DEMO_SEARCH=1 TASK_GRAPH=random_scenario`는 원래 모델·환경에서8명 랜덤 장면을 기록한다. 최소 한 BEFORE 쌍이 있어야 하며, 모든 쌍의 선행/후행이 native gated 판정과 독립 공간 판정을 함께0.5초 유지해야 한다. 선행 최초완료 전 후행 공간성공이나 초기 선행완료는 제외한다. 같은 시점 전체6/8 이상 연속성공을 후보로 저장하며7/8 이상을 우선한다. 낙상·종료 후 성공은 인정하지 않는다. 이번 혼합장면 선택은 근거리 대기를 추가 필수로 강제하지 않는다.

`learning/multi_agent/random_demo.py`에서 탐색하고 `validate_random_demo.py`가 저장 프레임의 공간판정 및 시간조건을 CPU로 재검증한다. 성공률은 선택한 장면에서의 동시 성공 인원 비율이며 모델 일반 성공률 추정치가 아니다. 표시만 불필요한 물체를 숨기며8명 전원과 task에 필요한 상자를 재생한다. `L`은 기존과 같이 PNG만 보존한다.

선택 결과는 `output_etc/check/random8_demo_search/`에 있다. `case_b0_e14`는8/8 동시성공22.83초, BEFORE3→5(AT→ON_TOP) 한 쌍. `case_b0_e15`는7/8 동시성공19.20초, BEFORE0→2(AT→CLIMB)·3→7(AT→ON_TOP) 두 쌍 모두 성공이며 개인5번 SIT는 해당 시점 실패다. 각각 `view_random8_success.sh`, `view_random8_before2_success.sh`로 재생한다. 이 장면들은 성공 상태가 확인된 시점까지 잘라 저장했으며 그 이후 지속성공을 보장하는 영상은 아니다. 각 `validation.json`에 과제별 성공과 BEFORE 매핑을 기록했다.

추가 탐색은 `RANDOM_DEMO_MIN_PAIRS=2 RANDOM_DEMO_TARGET_CASES=3`으로 BEFORE 최소 두 쌍·여러 후보를 수집했다. 사용자 정정에 따라 완료 시간 상한은 적용하지 않는다. seed43·32환경×6배치에서 전원성공 두 후보를 저장했다. `output_etc/check/random8_demo_alternatives_seed43/case_b4_e16`은 BEFORE4→1(SIT)·2→3(CLIMB),8/8 성공13.13초이며 `view_random8_alternative.sh`로 재생한다. 다른 과제 사람 간 root XY 최소1.27m(0.6m 이내 구간0)이지만 이는 충돌력 측정이 아닌 근접 대리 지표다. 근거리 대기2~3m 조건은 충족하지 않는다. `case_b4_e17`도8/8 성공이며38.47초, BEFORE SIT·ON_TOP 구성이다. 두 후보 CPU 검증 및 비교는 `comparison.json`에 보존한다.

근거리 대기가 필요한8명 탐색에는 `RANDOM_DEMO_WAIT_RADIUS=3`을 추가한다. 모든 BEFORE 후행의 root와 선행 AT 최종목표의 XY거리·속력으로 기존1초/0.2m/s 조건을 검사하고, 대기 시작부터 선행 최초완료까지 지정 반경을 벗어난 후보는 제외한다. 두 후행 모두2m 대기인 장면을 우선한다. 사용자 정정 이후 최소 쌍수는 `RANDOM_DEMO_MIN_PAIRS=1`이며, 2쌍 이상 제한이나 완료 시간 상한은 없다. `near_waits`가 메타데이터와 CPU 검증JSON에 기록된다. 이 기준은 이후 근거리 탐색에 적용하며 과거8명 성공 장면은 자동으로 적합 사례가 되지 않는다.

근거리 선택 사례: `output_etc/check/random8_demo_near_seed44/case_b2_e27`, seed44·64환경×3배치에서 발견. BEFORE2→1,6→4,0→7(전부ON_TOP), 후행 대기 최대2.663/1.769/1.284m, 대기 시작부터 gate까지 최대2.669/1.915/1.296m. 최소1초/속력0.2m/s 검사 및 CPU 재검증 통과. 약15.03초에8/8 동시성공. 3m fallback이며 세 쌍 전원이2m 조건을 충족한 것은 아니다. `view_random8_near_success.sh`로 재생하며 기존 view·PNG 저장 유지. 서로 다른 과제 사람 간 root XY 최소0.980m,0.6m 이내 구간0(충돌력 직접 검증은 아님). 다른7/8 후보e28은 근접 구간이 있어 우선 선택하지 않았다.

사용자 요청에 따라 최소1쌍으로 완화한 추가 seed45·192장면 탐색은 근거리 조건 만족 새 후보0개로 정상완료했다. 기록: `output_etc/check/random8_demo_near_seed45/search_summary.json`. 기존 seed44 선택 사례는 보존한다.

## 걸어와서 대기하는 ON_TOP

`ONTOP_DEMO_REQUIRE_WALK=1 ONTOP_DEMO_RADIUS=4`로 처음부터 제자리 대기인 사례를 제외한다. 대기 시작 전에 root가 초기위치에서1m 이상 이동하고 최종목표와 거리가0.75m 이상 줄어야 하며, 기존1초/0.2m/s/선행 최초완료 전 대기 및 ON_TOP 성공 판정을 유지한다. 이는 원래4명 환경 실제 rollout에서 선택하고 view에 두 사람·두 상자만 표시한다.

선택: `output_etc/check/ontop_demo_walk_seed46/case_b0_e44_f3`. 초기목표거리 약4.9m에서1.25m 이동/1.20m 접근 후6.87~7.83초 동안 최대3.705m에서대기, 선행8.57초, 성공확인13.97초. 초반4.27m 부근에서 잠시 멈춘 뒤 다시 접근하므로 처음부터 끝까지 연속 보행인 장면은 아니다. CPU 거리/속력/이동/공간성공 재검증 및 선행 이전 후행 공간성공 없음 확인. `bash view_ontop_walk_success.sh`, L PNG 저장은 해당사례 `recordings/` 아래다.

## 상자를 든 채 짧게 대기하는 탐색

사용자가 짧은 정지도 허용해 `ONTOP_DEMO_WAIT_SECONDS=0.5`를 지원한다(기본1초). `ONTOP_DEMO_REQUIRE_CARRY=1`이면 기본적으로 `ONTOP_DEMO_STABLE_CARRY=1`도 적용한다. 기존 신체 속력0.2m/s에 더해 상자 선속력0.25m/s 이하를 대기 구간에 요구하고, 대기 시작부터 선행 최초완료까지 실제 holding 판정이 연속 유지되어야 한다. 단순히 몸을 멈추고 상자를 바닥으로 내리는 동작을 대기로 선택하지 않기 위한 기준이다. 거리4m까지 허용하며 이번 탐색에는 보행 필수를 적용하지 않는다.

seed47·128환경×8배치의1초 운반 대기 탐색은 적합 성공0개였다. 짧은 기준으로 재실행한 `ontop_demo_carry_short_seed47/case_b0_e121_f3`은 몸만 약0.63초 멈춘 채 상자를 내리고 선행 완료 전에 바닥에 두므로 기각했다. `rejected.json`에 사유를 보존하며 재생용 바로가기에는 등록하지 않는다. 강화된 기준의 실행 명령과 배치별 진단은 `output_etc/check/ontop_demo_stable_carry_seed48/`에 보존한다. CPU 검증도 상자 속력 및 gate까지 holding 연속성을 재검사한다.

seed48 강화 기준8배치/2048쌍 정상완료: 적합 후보0개. `search_summary.json`에 결과를 보존했다. 해당 성공 rollout에서 gate까지 연속 운반하면서 몸과 상자가 동시에 정지하는 더 짧은 구간도 발견하지 못했다. 새로운 운반 대기 성공 영상은 저장되지 않았다.


사용자의 대기 의미 정정(완전 정지 불필요, 큰 이동만 없으면 됨)에 따라 `ONTOP_DEMO_WAIT_EXTENT=0.3`을 추가했다. 지정 시 순간 root/상자 속력 제한 대신 대기 구간 root XY 위치들의 최대 쌍별거리≤0.3m를 검사한다. 이번 최소 구간은0.5초이며, 상자 holding은 대기부터 선행 최초완료까지 계속 유지한다. 상자를 한 번 내려놨다가 다시 든 경우에는 최초 gate까지 이어지는 마지막 holding 구간에서 대기를 찾도록 해, 앞선 대기 구간 때문에 뒤의 유효 구간을 놓치지 않는다. 기존 옵션 미지정 시 속력 기반 기준을 보존한다. 새 탐색은 `output_etc/check/ontop_demo_extent_carry_seed49/search.sh`에 보존한다.


이동 범위 기준 선택 결과: seed49·8배치/2048쌍에서 `case_b7_e88_f2` 한 후보 저장. 1.67~2.13초의15제어프레임(0.5초) 동안 root XY 직경0.292m, 목표거리 최대3.937m. 상자를 든 채 선행 최초완료4.83초까지 운반을 유지하며 후행 공간성공6.40초/연속성공확인7.43초. CPU 검증·실제 대기와 성공 화면 확인. **들어 올린 직후 짧게 자세를 잡고 이동하는 장면으로, 오래 제자리에서 기다리는 장면은 아니다.** `bash view_ontop_carry_wait.sh`로 확인할 수 있으며 원래 view와 L PNG 저장을 유지한다. 선행 완료 이전 후행 공간성공 없음도 확인했다.


## 완화된 중심 기준의 ON_TOP 후보

사용자 요청으로 `ONTOP_DEMO_SUCCESS=center_above`를 지원한다. 후행 payload 중심의 XY가 support의 yaw 기준 전체 footprint 안에 있고(안쪽 margin 없음), 중심 Z가 support 윗면 이상이면 기하 후보로 인정한다. native ON_TOP 성공·상자 바닥과 윗면 사이 간격·접촉·놓기 완료는 이 판정에 포함하지 않는다. 따라서 운반 중 상자를 위에 들고 있는 상태도 통과할 수 있어 실제 내려놓기는 영상을 별도로 확인하고 결과 설명에 구분한다. 선행은 기존 native AT+독립 공간 AT 판정을 유지하고, 선행 최초완료 이후0.5초 연속 기하 조건을 요구한다. 선행 이전 기존 후행 공간성공은 제외한다. 학습/보상/정식평가 코드는 변경하지 않는다. 기본값은 기존 `strict`이며 저장된 metadata의 `success_mode`와 CPU validation에 두 기준을 구분해 기록한다.

실행 명령: `output_etc/check/ontop_demo_center_carry_seed50/search.sh`. 기존 대기 후보를 새 기하로 재평가하기 위해 동일 시드의 실제 rollout을 별도 경로에서 다시 실행한다(기존 탈락 후보에는 전체 궤적이 저장되지 않았음).

완화 기준 결과: seed50·512쌍 중2후보. `output_etc/check/ontop_demo_center_full_seed50/case_b1_e25_f0`는 목표0.46m 부근에서 짧게 머문 뒤 다시 이동하며 선행21.27초/중심조건27.67초 통과. `case_b1_e64_f2`는3.54m 부근 짧게 머물고 선행18.03초/중심조건18.53초 통과. 전체 유효궤적을 재저장해 확인한 결과 두 사례 모두 상자를 놓은 상태로 중심조건을 유지한 프레임은 없다. 첫 사례는 계속 들고 있고 두 번째는 옆으로 떨어지므로 실제 쌓기 완료 사례로 추천하지 않는다. 원시 완화 기하 후보로 보존하며 `bash view_ontop_center_candidates.sh 25` 또는 `64`로 재생한다. `validation.json`, `followthrough.json`에 판정과 후속 결과를 분리 기록했다.

## 걸어와 주변에서 기다린 뒤 CLIMB

`output_etc/check/climb_demo_walk_seed51/case_b0_e62_f1`, seed51·64환경 첫배치128쌍에서 선택. 초기위치 대비1.01m/목표접근1.01m 후 목표2.23~2.28m 부근에서 약4.2초 머물며 해당구간 root XY 직경0.057m. 선행 최초완료6.43초, 후행 첫 공간성공8.63초, native+독립공간 연속성공 확인9.53초. 선행 이전 후행 공간성공 없음. 순간속력 제한 없이 이동범위로 선별했고 CPU 검증과 실제 대기/완료 화면 확인. 사용자는 대기를 더 느슨하게 허용했으나 첫배치에서 기존 이동범위1초/30cm 기준도 만족하는 후보를 찾았으므로 추가 완화 탐색은 불필요했다. `bash view_climb_walk_success.sh`로 두 사람·상자1개만 원래 view로 재생한다. L PNG 저장 위치는 해당사례 `recordings/`. `extended_wait.json`에 gate까지 실제 머문 구간을 기록했다.

## 16명 랜덤 과제

동일 checkpoint에서16명·32개 논리물체·random_scenario를 사용한다. `RANDOM_DEMO_MIN_PAIRS=0`으로 쌍 수를 제한하지 않고 모든 존재하는 BEFORE 쌍은 native+공간 성공0.5초를 충족해야 한다. 이번에는 대기 시간/거리 추가 필터 없이 `RANDOM_DEMO_MIN_SUCCESS=12 RANDOM_DEMO_HIGH_SUCCESS=14`로 전체 성공 인원이 많은 장면을 선택한다. 성공 인원이 같으면 쌍 개수 대신 확인 시점을 우선한다. 저장된 장면의 동시 성공 인원이며 일반 성공률 추정값이 아니다. `random_demo.py`/`validate_random_demo.py` 공용, renderer는16명 색상과 RANDOM16 목표를 지원한다. 초기 명령은 `output_etc/check/random16_demo_seed52/search.sh`.

16명 초기 배치 병목 대응: 사용자 요청으로 `RANDOM_DEMO_TASK_BOXES_ONLY=1`일 때 task/alias의 실제 물체만 활성화해 배치한다. 나머지 물리 actor 슬롯은 텐서 인덱스 호환을 위해 유지하되 장면 밖으로 보내고 경로·충돌 검사에서 제외한다. `RANDOM_DEMO_BOUNDED_RESET=1`은 유효 배치 조건을 완화하지 않고 시도 횟수만 줄여 어려운 배치를 건너뛴다. 두 옵션 모두 명시적 headless 데모 탐색에서만 동작하고 학습/정식평가에 적용하지 않는다. seed52·53 초기 배치 실행은 사용자 승인으로 종료, 수정본은 `random16_demo_taskboxes_seed54/search.sh`.

## 실제2명 BEFORE SIT

`output_etc/check/sit2_demo_walk_seed60/case_b1_e41_f0`: 실제2명·BEFORE 한 쌍, seed60·64환경×2배치 탐색. 초기1m 이동/0.92m 접근 후 목표3.59m 이내에서 대기, 선행8.43초, SIT 첫 공간성공11.87초 및 연속성공확인12.33초. 선행 이전 앉기 공간성공 없음, native+공간 기준 CPU 검증 및 화면 확인. `bash view_sit_walk_success.sh`로 재생하며 두 사람·상자1개만 표시. L PNG는 사례 폴더의 `recordings/` 아래. 먼저 실행한4명 seed58 사례는 실제2명 결과로 제공하지 않는다.

## 추가 실제2명 SIT (seed62)

`output_etc/check/sit2_demo_walk_seed62/case_b0_e58_f0`: 초기1.01m 이동 후5.50–7.87초 목표2.64–2.67m 주변에 머무름(XY 이동범위0.050m). 선행7.90초, 첫 SIT 공간성공14.57초, 연속 native+공간 성공15.90초. CPU validation.json 통과. `bash view_sit_samples.sh 2`로 재생하며1은 기존 seed60 장면. L로 사례 폴더 recordings 아래 PNG 저장.
