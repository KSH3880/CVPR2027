# Changelog

최신 변경부터 기록한다. 현재 실행법은 [config.md](markdowns/config.md), 코드 위치는 [structure.md](markdowns/structure.md)를 참조한다. 실행 중인 GPU/PID는 이 파일에 고정하지 않고 실제 프로세스로 확인한다.

## 2026-10-08

### TaskRSI PUSH 목표 거리 확대

- 짧은 이동거리 개선 요청으로 현재 task_rsi의 일반 시작 목표거리를1.5~2.0m, RSI 남은 거리를0.5~1.2m로 늘렸다. interaction과 startRandomization 거리 설정을 함께 맞췄다. 기존 문 반대 방향 및 각도 랜덤화는 유지한다.
- CPU26테스트와 diff검사 통과. GPU학습은 실행하지 않았고 현재학습/MPS는 변경하지 않았다. 새 실행부터 적용되며 이전 checkpoint view는 run의 저장 설정을 사용한다.


### 크기 변경 전 TaskRSI checkpoint viewer 설정 override

- task_rsi_test에 `CFG_ENV` override를 추가해 VNC에서도 학습 당시 `relation_config.yaml`을 사용할 수 있게 했다. 현재1.3m 설정과 계약 검사는 유지한다.
- 08-12-17-28의1000 checkpoint와 저장 설정(1.1m)의 계약 정확 일치를 CPU에서 확인했다. 셸문법/diff 검사 통과. GPU viewer와 기존학습/MPS는 실행·변경하지 않았다.


### TaskRSI PUSH 박스 1.3m 및 실행 시 크기 지정

- 기본 박스를1.3m 정육면체로 늘리고 밀도로30kg을 유지한다. 초기 push 거리1.2~1.45m로 조정했다. `--push_box_size`와 train/test/preview의 `BOX_SIZE`로 크기를 지정하고 질량·최소 접근 간격을 자동 보정한다. VNC는 환경변수를 그대로 전달한다.
- CPU26테스트 통과(질량·배치간격·크기override·잘못된 입력 포함). 셸문법/Python문법/diff 검사 통과. GPU학습·viewer는 실행하지 않았고 기존학습/MPS는 유지했다. 크기가 다른 checkpoint는 기존 계약 검사를 따른다.


### 학습 로그에 iteration 명시

- iteration이 잘 보이지 않는다는 사용자 요청으로 MAAgent의 iteration 시작에 `[train] iter=... frames=... (starting)`을 flush 출력한다. checkpoint/resume의 epoch_num을 그대로 사용하고 print_stats/rank0 조건을 따른다.
- Python 문법 및 diff 검사 통과. GPU학습은 실행하지 않았으며 실행 중 학습/MPS는 변경하지 않았다. 새프로세스부터 반영된다.


### 잘못 제거한 TaskRSI PUSH 손 위치 보상 복구

- 사용자의 "다시 바꾸기"는 보상크기를 의미했는데 손보상까지 제거한 해석을 수정했다. 현재task_rsi에 양손표면품질 기반 추가보상0.15를 복구했다. 기존PUSH progress감쇠는 추가하지 않고 손보상만 적용한다. DOOR 접근0.15 등 기존조정은 유지한다.
- 손높이 이탈·반대면·한손이탈 회귀검사를 포함한 CPU테스트 통과. 현재학습/GPU/MPS는 변경하지 않았으며 새실행부터 반영된다. GPU동작 개선은 미검증이다.


### TaskRSI DOOR 접근 보상0.1→0.15

- 접근reward를 조금 올려보자는 사용자 요청으로 현재 task_rsi의 approach_weight만0.15로 조정했다. 손거리 최고기록·접촉 전 gate·문각변화 gate는 유지한다. 다른보상/AMP/PUSH 및 fixed/random_start 설정은 변경하지 않았다.
- YAML을 읽어 단일계수 외 동일함과 interaction검증 통과를 확인했다. GPU학습/평가/테스트는 실행하지 않았고 현재학습/MPS는 변경하지 않았다. 새실행부터 적용되며 checkpoint계약 계수도 달라진다.


### 현재 TaskRSI checkpoint 정책 VNC wrapper

- 새버전 view 요청으로 task_rsi 전용 VNC wrapper를 추가해 현재 env/train YAML과 checkpoint로 정책을 실행한다. 기존 runtime/GUI/MPS 연결을 재사용하며 frozen RSI preview와 구checkpoint viewer는 유지한다. 실행/구조 문서를 갱신했다.
- 현재 run의500 checkpoint와현재 config metadata 정확한 일치를 CPU에서 확인했다. 셸문법검사 통과. GPU viewer는 실행하지 않았고 학습/MPS는 변경하지 않았다.


### DOOR 원거리 접근 loco와 근거리 손별 개방 AMP 전환

- 사용자 요청으로 공통3설정에 approach_loco를 추가했다. root-앞/뒤손잡이 최소XY거리0.8m 진입·1.0m 복귀 히스테리시스로 원거리loco/근거리손별doorOpen을 전환한다. 접촉·개방유지 phase는DOOR우선이다. reset/물리step에서 상태를 갱신한다.
- 기존5종에 접근loco family5를 추가해6종/1350-D AMP를 사용한다. 실제 전문가source는loco phase0~1이며 demo/replay/정규화/history가6종으로 매칭된다. hold_source=door_tail·보상·정책관측/action·좌우손라벨은 유지한다. approach 설정이 없는 구viewer는3/5종을 유지한다. task RSI계약v8 및 실행문서를 갱신했다.
- CPU31테스트 통과(거리 경계·히스테리시스·접촉/유지 우선·6종/loco매칭·라벨정규화·기존3종회귀), 수정 Python문법/diff검사 통과. GPU/전체물리·학습은 실행하지 않았고 기존학습/MPS는 변경하지 않았다. 실제 접근/개방 성능은 미검증이다.


### 사용자 확인 좌우 손 라벨로 RSI·AMP 참조/replay 매칭

- 사용자 확인 `right_side`미러=왼손, `left_side`원본=오른손 라벨을 추가해 RSI reach 추정을 대체했다. 알려지지 않은 경로는 명시적으로 거부한다. 정면/높이/reach/문틀 조건은 유지하며 RSI124frame·양손 종류가 남았다. task RSI계약v7.
- 공통3설정에 AMP user_hands_v1을 추가해 PUSH/오른손개방/왼손개방/오른손유지/왼손유지5-family(1340-D)를 사용한다. 전문가 clip은 사용자 손 라벨로 샘플하고 demo/replay는5-family로 매칭한다. 정규화/물리history label도5종을 지원하며 다른task/구viewer의3종 기본과 오류 메시지는 보존했다.
- RSI는 모션 작업 손을 초기 AMP손으로 지정하고 접근 시작은 랜덤 좌우로 지정한다. 실제손-손잡이 거리와 양쪽 접촉힘 조건을 만족하면 현재손으로 갱신하며 두손접촉은 더 가까운 손을 따른다. 보상/정책관측/action/문방향 필터는 유지한다. 구checkpoint 현재계약 혼용은 거부하므로 새학습이 필요하다.
- CPU29테스트 통과: 명시라벨·미지원경로 거부·5종demo/replay와 정규화·fallback·기존Unified3종 회귀. 실제8모션/양손 각2048 expert샘플·phasewindow·RSI124frame 검증. 보고서 runs/rsi_checks/door_user_hands_amp.json. GPU학습/전체물리·손전환 동작은 실행하지 않았고 MPS/기존학습은 변경하지 않았다.


### DOOR 접촉RSI와 떨어진 접근 시작50:50 혼합

- 문에서 걸어가서 여는 시작도 필요하다는 사용자 요청으로 task_rsi 실험에 DOOR RSI확률0.5를 추가했다. 나머지0.5는 loco 자세·닫힌 문·문plane 앞1.2~1.8m 시작이다. PUSH RSI0.8·정면/참조 손 필터·보상·AMP는 유지한다. 구설정은 단일확률 동작을 유지한다. 초기화 계약v6으로 변경했다.
- Preview 기본100%강제를 제거해 현재 학습 분포를 그대로 본다. 명시적 RSI_PREVIEW_PROBABILITY=0/1로 접근/접촉만 볼 수 있으며 문서를 갱신했다.
- CPU30테스트 통과: 20,000환경의 PUSH0.8/DOOR0.5 혼합비·거리1.2~1.8m·lane분포·legacy확률 및 기존 회귀. GPU 학습·viewer·전체 RSI물리는 실행하지 않았고 기존 학습/MPS는 변경하지 않았다. 실제 접근 후 문 개방 학습 성능은 미검증이다.


### DOOR RSI 등진 자세 제외·모션 작업 손 고정

- 사용자 관측에 따라 DOOR RSI에서 골반/torso 정면이 손잡이 방향60° 밖인 후보를 제외한다. 참조 회전을 정렬까지 전달하고 문틀/손 높이/reach 검사를 함께 적용한다. 몸을 등진 채 손만 붙는 후보는 제외한다.
- 실제 asset FK로 복원한 clip의 RSI phase 양손 reach 평균을 비교하여 작업 손을 고정한다. 차이6cm 미만은 모호한 clip으로 RSI 제외, 선택된 모션의 작업 손만 손잡이에 정렬하며 높이 때문에 반대 손으로 바꾸지 않는다. 원본 contact annotation이 없어 손 정보는 자세 기반 추정이다. left_side/right_side 파일명은 접촉 손 라벨이 아니다. preview에 손 인덱스 로그를 추가하고 초기화 계약v5/문서를 갱신했다. AMP3-family 좌우혼합은 유지하고 손별 AMP 조건화는 하지 않았다.
- CPU29테스트 통과: 좌우미러/모호 clip·골반/torso 등짐·강제 손 높이 불일치·기존 reward/geometry 회귀. 실제 DOOR8모션에서104 RSI프레임·양손 종류·모든 frame의 유효 후보를 확인하고 대표 자세 그림을 검토했다. 보고서/그림은 `runs/rsi_checks/door_facing_hands.{json,png}`. CPU 정렬 손잡이 최대 오차0.1012m(높이 tolerance10cm+gap3cm). GPU viewer/학습 및 전체 물리 RSI는 실행하지 않았고 MPS·학습 프로세스는 변경하지 않았다.


### 직전 보상 강화 원복·손잡이10cm 낮춤

- 사용자 요청으로 PUSH 양손 표면 보상·진행 감쇠, DOOR opening_weight0.8·angle_weight0.2와 관련 helper/로그/테스트를 제거했다. 기존0.4 진행 및 handle_gated_v2는 유지한다. 문 반대 PUSH 동선은 유지한다.
- 공통3개 설정에 door.handle_height0.95m를 추가했다(기존1.05m). 명치 부근으로 낮추려는 첫 조정이며 자세별 높이 일치는 아직 미검증이다. 실제 asset을 spec별 `runs/door_assets/`에서 원자적으로 생성해 물리/관측/RSI를 일치시켰다. 기본 DoorSpec/원본URDF는1.05m로 유지해 구 checkpoint viewer를 보존한다. 높이는 interaction 계약에 포함된다.
- CPU27테스트 통과, 수정 Python 문법 검증 통과. CPU PhysX1-step에서 앞/뒤 손잡이 actual z=0.94999999m 확인. CPU 실행의 libcuda preload 경고가 있었으나 geometry 검증은 완료했다. 전체 agent RSI·GPU 학습은 실행하지 않았으며 진행 중 학습·MPS는 변경하지 않았다.


### 로컬 TaskRSI 5500 viewer 계약 복원

- 구3500 viewer 설정과 다른 로컬 TaskRSI5500 checkpoint의 interaction·시작 분포·충돌 설정을 CPU에서 복원해 `runs/checkpoint_views/push_door_task_rsi_from_local_5500.yaml`을 생성했다. 원래 RSI와 구 보상으로 실행하며 현재 학습의 방향/보상은 유지한다.
- interaction 검증·checkpoint metadata 정확한 일치·TaskRSI/random_start network 설정 동등성 확인. GPU viewer 실행·학습·MPS 변경 없음.


### PUSH 양손 이탈 억제와 DOOR 접촉 개방 보상 강화

- 상자가 틀어지면서 손이 올라가고 DOOR가 옆으로 피하는 관측에 따라 fixed/random_start/task_rsi 공통 보상을 수정했다. PUSH는 상자 로컬 좌표의 목표 반대 측면·손 반경·높이 범위로 양손 오차를 계산하고 최소 Gaussian 품질로 진행 보상을 감쇠한다. 품질 기반 손 보상0.15는 진행/정착 때만 지급해 손만 붙이고 정지하는 보상을 막는다. 실제 접촉력 판정은 아니며 기존 상자 바닥/upright·정착/성공 조건은 유지한다.
- DOOR는 접촉 개방 진행 계수0.4→0.8 및 접촉 조건부 개방각 보상0.2를 추가했다. 기존 손잡이 gate·기록 소비·닫힘 패널티는 유지하며 통과 목표는 추가하지 않았다. 보상/진단 로그와 실행 문서를 갱신했다. 새 키는 interaction 계약에 포함되어 구 checkpoint 혼용을 거부한다. 키가 없는 구 viewer config는 기존 보상을 유지한다.
- CPU18테스트 통과(양손·높이 이탈·반대 면·한 손 이탈·목표 도착·기존 reward 회귀·계약/설정). GPU 학습·시뮬레이션은 실행하지 않았으며 사용자 학습·MPS는 변경하지 않았다. 실제 동작 개선은 새 학습에서 확인해야 한다.


### 기존 checkpoint의 원래 환경으로 viewer 실행

- 사용자 요청으로 epoch3500 checkpoint의 interaction·시작 랜덤화·충돌 설정을 CPU에서 읽어 `runs/checkpoint_views/push_door_random_start_08-01-39-16_3500.yaml`로 복원했다. 현재 학습 config의 away_from_door는 유지한다.
- random_start 평가 스크립트에 명령별 `CFG_ENV` override를 추가해 view/VNC에서 별도 환경을 선택한다. 기본값과 학습 스크립트는 유지하며 checkpoint 계약 검증을 우회하지 않는다. 실행 가이드에 구 checkpoint VNC 명령을 기재했다.
- 복원 YAML의 metadata와 checkpoint 계약 정확히 일치·interaction 검증·현재 학습 방향 유지·셸 문법 검사 통과. GPU viewer는 실행하지 않았으며 MPS·학습 프로세스는 변경하지 않았다.


### PUSH 목표와 시작 자세를 문 반대 방향으로 전환

- 문 쪽으로 밀던 동선에 대한 사용자 요청으로 fixed/random_start/task_rsi에 `interaction.push.direction: away_from_door`를 적용했다. 공통 reset에서 PUSH의 목표·사람 위치를 상자 중심으로 180도 돌리고 상자·사람 orientation도 함께 회전한다. DOOR는 유지한다. RSI는 회전된 상자 방향에서 정렬·목표 재생성을 수행하므로 같은 방향을 유지한다.
- 방향 설정은 기존 interaction checkpoint 계약에 포함된다. 이전 방향 checkpoint 혼용은 거부하며 새 설정은 scratch 학습한다. 진행 중 학습·MPS는 변경하지 않았다. 실행·구조 문서를 갱신했다.
- CPU 테스트 16개 통과: 혼합 task 8192 agent 샘플의 목표 -X·사람 뒤쪽 배치·거리 보존·DOOR 불변·RSI 기준 상자 방향을 검사했다. 수정 Python 3개 문법 검사 통과. 테스트 실행은 Isaac Gym 선행 import와 conda 라이브러리 경로를 사용했다. GPU 시뮬레이션·학습은 실행하지 않았다.


### DOOR 손잡이 접촉 기반 보상

- 공통 task/helpers와3개 config에 handle_gated_v2 적용. 개방 진행 보상을 손잡이 접촉으로 제한하고 접촉 없는 최고 기록도 소비해 지연 보상/닫기-재개방 악용을 막았다. 최고 접근 진행 보상0.1, 닫힘 패널티-0.2 추가. 문 이동으로 접근 보상을 얻는 것을 제한하고 RSI 리셋에서 거리/직전 각도 기준을 초기화했다. 기존 유지·성공·PUSH·AMP 혼합비 유지, TensorBoard 신규항목 기록, checkpoint 계약 변경.
- CPU15테스트 통과: 몸으로 개방, 뒤늦은 접촉, 접근 왕복, 닫힘/작은 진동/RSI 초기 기준 검사. CPU PhysX16환경×12리셋384샘플/부분 reset/finite/초기 진행 보상 검증 통과: output/door_reward_v2_reset_check/rsi_report.json. GPU 학습은 다른 진행 중 학습을 유지하기 위해 실행하지 않았다. 서버에는 추적 공통 코드/config를 반영하면 되며 기존 보상 checkpoint와 새 설정은 호환되지 않는다.


### PUSH 상자 질량30kg 통일

- 사용자 요청으로 fixed/random_start/task_rsi 공통 상자를30kg으로 통일했다(기존15.75/34.61kg). 1.1m 정육면체와 friction0.1 유지, density22.539444027047325. 세 설정 질량을 검증하도록 기존 테스트를 갱신했다. 변경은 다음 실행부터 적용되고 checkpoint 계약의 density가 바뀐다. 진행 중 학습은 종료하지 않았다.


### PUSH 상자 마찰 감소

- 사용자 요청으로 공통3개 PUSH/DOOR config의 상자 friction을0.6→0.1로 낮췄다. 바닥 마찰/질량/정책은 유지한다. CPU PhysX에서 두 질량15.75/34.61kg, 접촉 높이1.0/1.08m, 수평 힘60~200N, 마찰5개 후보를 비교했다. 작은 감소는 효과가 제한적이었고0.1에서도 강한 힘의 전도는 남았다. 결과 output/push_box_friction_probe/report.json. 서버 공유 설정이며 현재 학습 프로세스는 재시작 전 기존 설정을 사용한다. checkpoint 계약에 마찰이 포함된다.


### PUSH RSI 양손 표면 정렬과 리셋 PD 목표

- 한 손의 최대 reach를 기준으로 하던 방향 정렬을 양손 깊이가 같아지는 방향으로 변경했다. 실제 asset 골격으로 위치를 계산하고 손 반경4cm+표면 간격1mm를 반영한다. RSI 리셋에서 해당 사람만 초기 자세 PD 목표를 설정해 preview commit step 중 손이 이전 목표로 되돌아가는 문제를 수정했다. 공통 task/helper/config에 적용해 서버도 동일하게 사용할 수 있다. DOOR 작은 각도 분포와 문틀 여유는 유지한다. checkpoint 초기화 계약v4.
- CPU PhysX16환경×12리셋384샘플 검증 통과. 물리1step 후 양손 표면 간격0.729~1.267mm(95백분위1.187mm), 부분 reset/finite/초기 진행보상/DOOR 작은 각도 검사 통과. 결과 output/task_rsi_push_contact_cpu_check/rsi_report.json. GPU 확인은 기존 학습과 메모리 경합으로 중단했고 사용자 학습은 종료하지 않았다. 새 정렬 테스트 통과, 기존 질량 테스트는 별도로 변경된 random_start 밀도26과 기존15.75kg 기대값 불일치로 실패했다. 해당 사용자 설정은 수정하지 않았다.

## 2026-10-07

### 과제 RSI 서버 공통화와 작은 각도 시작 확대

- 로컬 PUSH/DOOR 참조 RSI와 문틀 여유3cm 검사를 공통 utils/task_rsi.py와 선택적 task reset에 옮겼다. 전용 env/train config, 서버 GPU6/MPS runtime을 사용하는 train/test/view 스크립트와 preview/검증 도구를 추가했다. 로컬 GPU/NVRTC/MPS 우회는 계속 Git 제외다.
- DOOR RSI의75%를5~25°에서 시작하도록 작은 각도 호환 자세부터 샘플한다. 나머지는5~79° 안전 각도, 후반 phase 비중70%, 과제 RSI 확률80%는 유지한다. 작은 각도는 합성 초기 문 상태이며 실제 grasp/장기 학습 성능은 미검증이다. 초기화v3 계약으로 기존v2 checkpoint와 혼용하지 않는다.
- CPU13테스트 통과. 로컬16환경/384 agent 리셋 검증에서 DOOR RSI166개 중80.12%가25° 이하, 문틀 여유/손잡이 정렬/부분 reset/finite/초기 진행보상 검증 통과. 결과 output/task_rsi_early_door_check/rsi_report.json. 공통 서버용 train/test 스크립트를 로컬 GPU 래퍼로 검증해2048환경 짧은 학습(epoch2/frame262144 checkpoint 저장) 및16환경/32-step/2회 평가 정상 종료했다. 결과 output/task_rsi_shared_early_train_check/PushDoorStage1TaskRSI_07-21-48-46/ 및 output/task_rsi_shared_early_eval_check/. 서버 GPU에서는 실행하지 않았다.


### DOOR 왼쪽 개방 모션 제한

- 사용자 요청으로 공통 Push/Door config에 `door.motion_direction: left_open`을 추가했다. BONES left_side 원본/right_side 미러만 로드하고 반대 조합을 제외한다. MotionLib의 선택적 필터를 task에서 전달해 AMP와 로컬 과제 RSI가 같은8개 참조를 사용한다. 대표 A512 원본/미러의 손 궤적에서 좌측 개방(+Y) 방향을 확인했다. 원본/미러 데이터 파일과 기존 weight=0은 유지한다.
- CPU11개 통과. GPU0/2048환경/MAX_ITERATIONS=1 과제 RSI 확인 학습 정상 종료(epoch2/frame262144 checkpoint 저장), doorOpen8개/36.233초 로드 확인. 결과 `output/task_rsi_left_open_check/PushDoorStage1TaskRSI_07-21-33-29/`, 로그 `/tmp/task-rsi-left-open-check.log`. 공통 config/코드는 서버 반영 대상, .local 구현/환경은 Git 제외다. 장기 성능은 미검증이다.

### PUSH 상자 공통 1.1m 정육면체

- 사용자 요청으로 고정/랜덤 시작 공통 env YAML의 PUSH 상자를 1.1×1.1×1.1m로 변경했다. 밀도 약11.833kg/m³로 기존 질량15.75kg을 유지한다. 상자 XY 반경·최대 목표 거리·전방 jitter를 고려해 미사용 문과20cm 여유를 확보하고, 랜덤 시작 사람 간격은1.1~1.35m로 늘렸다. 학습/평가 실제 asset와 bbox에 모두 적용된다. 이 변경은 추적 파일이며 서버 반영 대상이다.
- CPU10개 통과(크기·질량·초기 사람 간격·목표 지점 문 여유 포함). GPU0/2048환경/MAX_ITERATIONS=1 확인 학습 정상 종료(epoch2/frame262144), checkpoint의 정육면체 계약·82개 scalar finite 확인. 결과 `output/push_door_cube_check/PushDoorStage1RandomStart_07-21-27-15/`. 저장 checkpoint16환경/32-step/2회 평가 정상 종료(`output/push_door_cube_eval_check/`). diff 검사 통과.
- Git 제외된 로컬 과제 RSI도 같은 크기/질량으로 맞췄으며16환경×12 reset의 손 정렬·부분 reset·초기 진행보상 검증 통과(`output/task_rsi_cube_geometry_check/`). 기존 크기 checkpoint와 혼용하지 않는다. 장기 수렴은 미검증이다.

### 랜덤 시작 Push/Door에 기존 Unified 충돌 패널티 적용

- 기존 세팅에 맞추라는 요청으로 랜덤 시작 YAML에 agentCollisionPenalty=true·계수 0.5·거리 0.7m를 추가하고 Carry/Unified의 같은 root XY 거리 패널티 함수를 연결했다. 활성 설정만 checkpoint 계약에 추가하여 변경 전 보상 checkpoint 혼용을 거부한다. 기존 고정 시작 실험은 패널티 없이 유지한다.
- 초기 lane 3.2m·랜덤 시작 분포·door_tail AMP·2048환경·GPU 6은 유지한다. 기존 Unified도 초기 반경 2.5m의 분리 배치를 사용하므로 임의로 문 간격을 축소하지 않았다. 기존 reward term 크기를 유지하고 전체 보상에 패널티를 반영하며 extras로 별도 기록한다.
- 사용자 요청으로 테스트·문법 검사·GPU 실행은 수행하지 않았고 MPS·기존 프로세스도 변경하지 않았다.

### 랜덤 시작 실험 AMP 원복

- 사용자 비교 실험 요청으로 `push_door_stage1_random_start.yaml`의 유지 AMP를 `loco`에서 기존 `door_tail`로 되돌렸다. 랜덤 시작 분포와 전용 실행/output은 유지하여 기존 실험과 AMP 설정을 맞췄다. 현재 설정 문서를 갱신했다.
- 사용자 요청대로 테스트·문법 검사·학습·평가·viewer는 실행하지 않았으며 MPS와 기존 학습 프로세스도 변경하지 않았다.

### ZIP 기반 Push/Door 랜덤 시작 서버 반영

- `push_door_random_start_server.zip`의 task reset·CPU 샘플러·선택적 checkpoint 분포 계약·테스트 소스와 전용 env/train YAML·train/test/view/VNC 스크립트를 반영했다. 이전 미완료 랜덤 reset 변경을 제거하여 기존 `push_door_stage1`은 고정 시작과 기존 checkpoint 계약을 유지한다.
- 새 실험은 `push_door_stage1_random_start`, 학습 2048환경·별도 output이며, ZIP 설정대로 유지 AMP는 `loco`다. 보상·성공 조건·모션 manifest·모델 크기는 유지한다. 고정/랜덤 시작 checkpoint 혼용은 거부한다.
- 서버 wrapper의 GPU 기본값은 6으로 맞추고 기존 runtime/MPS 연결을 유지했다. ZIP 로컬 viewer의 MPS 강제 off 설정은 제거했다. MPS 설정/데몬과 기존 학습 프로세스는 변경하지 않았다.
- 사용자 최종 요청으로 **CPU 테스트·Python/셸 문법·diff 검사 모두 실행하지 않았다. GPU 학습·평가·viewer·GPU 테스트도 실행하지 않았다.** ZIP 안내의 로컬 CPU 9개·2048환경 학습·16환경 평가 통과는 제공된 결과이며 서버 검증 결과가 아니다.


### PUSH + 문 개방·유지 Stage 1 학습 연결

- 사용자 요청으로 `HumanoidMAPushDoor`와 전용 env/train YAML·motion manifest·train/test/VNC를 추가했다. 2명·상자 2개·왼쪽 힌지/오른쪽 손잡이 문 2개, agent별 PUSH/DOOR 독립 50/50이다. Unified clean-scene Transformer/GTA·실제 대상에 따른 semantic edge MLP bias, 656-D 정책 관측·agent당 32 actions를 사용한다. 기본 GPU는 사용자 지정 **6**, 학습 2048환경이다.
- 최고 진행도 보상 0.4·현재 유지 조건 보상 0.6·최초 연속 성공 0.2를 적용했다. PUSH는 바닥/upright 조건에서 목표 XY 진행·15cm/0.15m/s/0.5초 정착, DOOR는 80°까지 진행·손과 손잡이 양쪽 접촉/거리/각속도 조건을 2초 유지한다. 닫았다 열기·공중 운반 보상 악용을 막고, 최고 진행도와 유지 시간을 관측에 포함한다. 유지 성공 이후에도 유지 보상을 계속 준다.
- AMP는 push/door 개방/door 후반 유지 3-family·1320-D이며, 기본 reference phase는 door 0~0.7/0.7~1.0이다. 실제 각도 80°/65° hysteresis·물리 history 보존/현재 phase label·expert/replay family 매칭을 사용한다. 유지 reference의 loco 전환은 config로 선택 가능하다. BONES에 object annotation이 없어 초기화는 loco 자세 정렬·속도 0·닫힌 문/바닥 상자로 구성하며, 물체 RSI를 생성하지 않았다. 별도 checkpoint 계약과 resolved 설정을 저장한다.
- 공통 humanoid tensor/PD target에서 추가 문 DOF를 분리했다. 새 task의 Gym env-local reset·관측을 실제 handle body와 비교했다. CPU **53개 통과**(새 보상/접촉/phase/AMP/checkpoint/model 순열·gradient 및 기존 모델/AMP 회귀). GPU 6에서 기존 문 없는 `HumanoidMACarry` 16환경·5-step finite 제어도 확인했다.
- **GPU 6/2048환경**, `MAX_ITERATIONS=1` 확인 학습 완료(epoch 2·frame 262144). TensorBoard 41종/82 scalar finite·checkpoint 계약을 확인했다. 최종 run: `output/push_door_stage1_check/PushDoorStage1_07-18-09-26/`, `train_check_report.json`. 16환경 실제 물리 검증에서 부분 reset·모든 task 쌍·손잡이 오차 <1µm·공중 상자 보상 0·각도만으로 유지 성공 불가·AMP 전환/매칭·32-step finite를 확인했다(`output/push_door_stage1_sim_check/`).
- 전용 headless 16환경 평가와 서버 VNC 1환경을 GPU 6에서 확인하고 JSON/PNG를 `_eval_check/`, `_vnc_check/`에 저장했다. 짧은 checkpoint의 task 성공은 확인되지 않았으며 장기 본학습/수렴은 아직 수행하지 않았다. Python/셸 문법·문서/데이터 경로·diff 검사 통과.

### 문 자동 닫힘 힘 강화

- 손으로 개방을 유지할 때 힘이 필요하도록 기본 stiffness를 1→6N·m/rad, damping을 0.8→3N·m·s/rad로 높였다. DoorSpec·CLI·wrapper 기본값과 현재 설정 문서를 함께 수정했다. 90°에서 복원 토크 9.42N·m, 손잡이 환산 약 11.59N이다.
- GPU 5/4개 평가 환경에서 접촉 개방 약 93.39°·부분 reset·자유 복귀를 확인했다. 90°에서 0.2초 후 86.17°, 접촉 제거 12초 후 <0.001°다. 결과: `output/door_environment_stronger_check/physics_report.json`. CPU 테스트 9개·셸 문법·diff 검사 통과. 사람 정책의 잡기 행동은 아직 검증하지 않았다.

### 왼쪽 힌지·오른쪽 손잡이 약한 자동 닫힘 문

- 사용자 요청으로 고정 문틀·0.9×2.1×0.045m 문짝·오른쪽 앞/뒤 손잡이·왼쪽 Z 힌지(0~110°) URDF와 재사용 DoorFixture를 추가했다. 정면 관측자가 -X에서 +X를 볼 때 +Y가 왼쪽이며, 밀면 안쪽·왼쪽으로 열린다. 관측은 hinge angle/velocity·문짝 pose·손잡이 위치·base pose, 부분 reset은 소유한 문 DOF만 갱신한다.
- 손을 놓으면 돌아오도록 target=0·stiffness=1.0·damping=0.8의 약한 compliant drive를 적용했다. 90° 정지에서 초기 복원 토크 약 1.57N·m(손잡이 환산 약 1.93N)다. 충돌 probe·reset·자유 복귀를 검사하는 독립 물리 환경과 headless/로컬/VNC wrapper를 추가했다. 새 RL task·reward·AMP·학습 config는 아직 연결하지 않았다.
- CPU **9개 통과**. tokenhsi·**GPU 5/4개 평가 환경**에서 서로 다른 yaw의 접촉 개방(최대 약 110°)·손잡이 좌표 오차 <1µm·부분 reset·finite 상태를 확인했다. 접촉 제거 후 12초에 각도 <0.001°, 90° 초기화 후 0.2초에 약 89.32°로 약한 복귀를 확인했다. 결과는 `output/door_environment_check/physics_report.json`이다.
- GPU 5/1환경 서버 VNC에서 오른쪽 손잡이·왼쪽 개방·자동 닫힘을 확인하고 닫힘/개방/복귀 PNG를 `output/door_environment_viewer_check/`에 저장했다. Python/셸 문법·문서 링크·diff 검사 통과. 기존 학습 프로세스는 변경하지 않았다.

### BONES door·pushing 원본의 AMP 형식 변환

- 사용자 데이터 폴더의 door 좌우·미러 16개와 `push_obstacle_180` 원본·미러 22개를 `motions/<clip>/phys_humanoid_v3/ref_motion.npy`로 변환했다. 동료의 SOMA BVH 파서·CPU IK를 기반으로 전체 clip 30FPS, 절대 Hips 위치·Z-up/m·타깃 크기·관절 제한·12rad/s 프레임 연속성·발 mesh 최저 5mm를 적용한다. 코드/원본 SHA256·손발 오차 보고서와 별도 `doorOpen`/`push` manifest를 추가했다.
- 관절이 튀던 3개를 연속성 제약으로 보정했다. 최대 손발 오차 >5cm인 오른쪽 door A514 원본과 push 102 A343 원본/미러는 파일을 보관하고 manifest weight=0으로 제외했다. 기본 샘플링은 door 15개·push 20개다. 생성 산출물은 Git 제외, 원본·기존 학습 실험은 유지한다.
- BVH CPU 회귀 **4개 통과**. 38개·12,496프레임 전체의 실제 MotionLib 로드·32 DOF/관절 제한·quaternion·고정 손목·발 mesh 바닥·관절 연속성·프레임/중간시간 보간·multi-agent AMP 129-D 및 10-step 1290-D finite·weight=0 샘플링 제외를 CPU 검증했다. 코드 버전·원본 해시·manifest 경로·문서 링크·diff 검사 통과. 결과·38개 offline 재생·대표 모션 그림은 `output/bones_amp_conversion_check/`다.
- 문/상자 pose·접촉을 원본이 제공하지 않아 물체 상태를 생성하지 않았다. PUSH/OPEN task·보상·AMP family·RSI object binding 통합과 시뮬레이터 접촉 검증·학습은 아직 수행하지 않았다.

### AT 목표 높이·위치 VNC 데모

- 사용자 요청으로 rescue `distill_pth` checkpoint를 그대로 불러오는 평가 전용 task/player·test/VNC 스크립트를 추가했다. 2명 모두 HOLDING+AT를 수행하며 목표 상자 중심 z=0~2m를 0.2m 간격으로, 초기 상자 바로 위/사람과 상자 사이/옆으로 1m의 세 XY 위치와 조합해 총 33장면을 순회한다. 같은 반복에서 초기 seed·배치를 맞추고, 각 장면은 기본 최대 600 step이다. z=0도 자동 보정하지 않는다.
- 높이·XY 선택 환경변수, 2m 목표를 포함하는 카메라·확대한 담당자 색 마커, 장면별 JSON·PNG 기록을 제공한다. 새 YAML·학습 없이 기존 checkpoint 계약 검사와 평가용 rescue 호환 처리를 사용한다.
- GPU 0·1환경에서 33조합×32-step headless 실행을 완료했다. 전체 높이 격자·세 XY 계산·두 AT owner·동일 초기 배치·finite 물리 상태와 오차를 확인했다. VNC에서 높이 0/2m×세 XY의 6조합×90-step 실행·PNG 저장과 2m 마커 표시를 확인했다. Python/셸 문법·문서 링크·diff 검사 통과. 결과는 `output/at_goal_demo_check/`, `output/at_goal_demo_vnc_check/`이며 장시간 높이 도달 성능은 미검증이다.

## 2026-10-05

### 복합 task·역할별 bias/message 실험

- 사용자 요청으로 `approach_scenario_stage1_unified_size_rsi_task_message` env/train YAML·train/test/VNC·output을 분리했다. 단독 HOLDING 없이 sit/climb/carry_at/carry_ontop 10/25/32.5/32.5%, 자기 primitive 보상 합계만 사용한다(`self=1, teammate=0`). Carry 보상은 2로 나누지 않는다.
- 현재 primitive graph의 owner/endpoint에서 `[valid,task,actor,payload,target]` 정책 packet을 만든다(634-D 관측). Carry는 actor→payload/actor→target/payload→target, sit/climb은 actor→target이다. Task·역할별 bias `[4,3,3,4,2]`, message `[4,3,3,64]`, NONE/SELF 배경 표를 actor/critic 각각 학습한다. Message alpha=1·std=.02·GTA 위치는 기존 방식이다. 보상·RSI·AMP는 동일 primitive graph를 계속 사용해 새 연결의 보상 중복을 막는다. Packet v5로 기존 checkpoint 혼용을 거부한다.
- CPU **132개 통과**: 새 16가지 task 쌍/물체·goal·owner 재매핑/edge·task·token 순열의 출력·gradient, carry_ontop 위아래·AT goal, 자기 보상/동료 비의존, 크기 분포·RSI/AMP binding·strict checkpoint와 기존 정책/보상 회귀. Python/셸 문법·새 문서 링크·diff 검사 통과.
- **GPU 5**, 2048환경·`MAX_ITERATIONS=1` 확인 학습 완료(저장 epoch 2, frame 262144). 기존 RSI 캐시 hit, 257종 scalar/514값 모두 finite, 물리 reset 실패 0, 단독 HOLDING 샘플 0. Actor/critic 각각 사용되는 8개 task-role 행의 bias와 메시지 optimizer 업데이트를 확인했다. 진단 CSV의 물리 binding·자기/동료 보상 대응 오류 0.
- 저장 checkpoint로 16환경·32-step headless 평가를 random 및 `carry_at/carry_ontop + carryWith` 세 설정에서 완료했다. 검증 output은 `output/approach_scenario_stage1_unified_size_rsi_task_message_check/`, `_check_eval/`, `_check_carry_at/`, `_check_carry_ontop/`이다. 본학습을 시작하거나 기존 학습을 중단하지 않았다. 장기 수렴·성능 개선은 미확인이다.


### Size RSI typed bias + 관계 메시지 실험

- 사용자 요청으로 `approach_scenario_stage1_unified_size_rsi_typed_bias_message` env/train YAML·train/test/VNC·output·checkpoint variant를 분리했다. 기존 typed bias와 Size RSI 보상·성공·크기·RSI·캐시는 유지한다.
- Actor/critic 각각 `[3,11,3,64]` 관계 표를 추가했다. `alpha=1.0`, 초기 표준편차 `0.02`, layer 공유·head별 32차원이며 NONE/SELF·비과제 메시지는 0이다. 같은 attention으로 유효 edge 메시지를 source에 합산하고, GTA 복귀 뒤 output projection 전에 더한다. Sparse gather/scatter와 토큰 순열의 endpoint 재매핑을 사용한다. 메시지 RMS와 노드 대비 비율을 TensorBoard에 추가했다.
- CPU **123개 통과**: 새 메시지 수식·GTA 위치·gradient, 전체 과제 쌍/goal slot·토큰/edge 순열의 action/value/gradient, alpha=0 기존 출력 일치, 독립 업데이트·strict checkpoint 및 기존 정책·graph/reward 회귀. Python/셸 문법·문서 링크·diff 검사 통과.
- 사용자 GPU 지정에 따라 **GPU 5**에서 2048환경·`MAX_ITERATIONS=1` 확인 학습(저장 epoch 2, frame 262144)과 16환경·32-step headless checkpoint 평가를 완료했다. 기존 RSI 캐시 hit, scalar **257개/514값 모두 finite**, 물리 reset 실패 **0**. 양쪽 관계 표의 다섯 과제 행 모두 optimizer 업데이트를 확인했고 NONE/SELF 값은 0이다. 검증 output은 `output/approach_scenario_stage1_unified_size_rsi_typed_bias_message_check/`, `_check_eval/`이다. 이는 실행·업데이트 검증이며 장기 성능 개선은 미확인이다.

## 2026-10-04

### Size RSI 기반 독립 typed-bias 표 실험

- 사용자 요청에 따라 `approach_scenario_stage1_unified_size_rsi_typed_bias` env/train config와 전용 train/test/VNC·output을 추가했다. 과제·크기·RSI·물리 캐시·AMP·보상·성공·`r_valid`는 기존 Size RSI와 같고 variant/네트워크 계약만 분리한다.
- Semantic edge MLP·projection 대신 0 초기화한 `(source type, relation, target type, layer, head)` 표 `[3,11,3,4,2]`를 사용한다. Actor·critic은 각각 792개 파라미터의 별도 표를 갖는다. Directed edge·NONE/SELF·토큰/GTA/bias 동시 순열·human readout 복원은 유지한다. 잘못된 env/train mode 조합·표 공유·bias 비활성화는 실행 전에 거부한다.
- CPU **96개 통과**: 전체 25개 과제 쌍·goal 슬롯 교환·배경/방향·MLP bias 표현 동등성·토큰/edge 순열의 action/value/gradient·독립 표 업데이트·strict 저장/복원·기존 graph/reward/policy 회귀. Python/셸 문법·문서 링크·diff 확인.
- GPU 1에서 2048환경·`MAX_ITERATIONS=1` 확인 학습(저장 epoch 2)과 checkpoint 저장을 완료했다. 기존 RSI 캐시 hit, scalar **233개/466값 모두 finite**, 물리 reset 실패 **0**. Actor/critic 표가 각각 0에서 서로 다른 finite 값으로 갱신됐다. GPU 5의 전용 test wrapper로 해당 checkpoint를 로드해 16환경·32-step headless 평가와 JSON 저장을 완료했다. 검증 output은 `output/approach_scenario_stage1_unified_size_rsi_typed_bias_check/`, `_check_eval/`이며 장기 수렴은 미확인이다.

## 2026-10-03

### 크기 혼합·불규칙 더미와 10회 반복

- 사용자 요청에 따라 정리 데모 기본값을 10회·1500 control step(한 회 최대 시뮬레이션 시간 50초)으로 바꿨다. 상자는 가로·세로 0.40/0.45/0.50m, 높이 0.30/0.35/0.40m를 섞고 매 반복 위치·수평 회전(±15°)을 다시 샘플한다. 실제 크기에 맞춰 받침·윗단 중심과 바닥 목적지 높이를 계산한다. 전체 장면 입력·회색 표시·두 라운드 재배정은 유지한다.
- checkpoint에 저장된 학습 크기 범위(가로·세로 0.40~0.65m, 높이 0.25~0.55m)를 확인했다. CPU **5개 통과**(100개 혼합 크기 배치의 초기 겹침·높이 포함), Python/셸 문법·diff 확인. GPU 3에서 2회×90-step 설정의 짧은 평가가 각각 89 step 종료되고 크기 범위·물리 상태 finite·반복 초기화를 확인했다(`output/box_cleanup_irregular_smoke/cleanup_results.json`). VNC의 혼합 크기·회색 화면과 10회/1500-step 실행을 확인했다. 새 더미의 8개 완주는 아직 미확인이다.

### 중앙 더미의 윗단 운반·상자 색 통일

- 사용자 요청에 따라 주변 바닥 대상 상자를 없애고 16개 모두 중앙의 4×2×2단 더미로 배치했다. 두 라운드의 대상 8개는 모두 윗단이며, 이 데모의 상자는 배정·라운드와 관계없이 같은 회색이다. 상자 사이 18cm 간격을 확보했고 기존 재배정·바닥 안정 판정은 유지했다.
- 관련 CPU **5개 통과**, Python 문법·diff 확인. GPU 3의 headless와 VNC에서 첫 네 개 운반·두 번째 배정·회색 화면을 확인했다. 현재 배치의 seed 42 검사에서는 **7개 운반 후 마지막 집기 정체로 시간 초과**했다(`output/box_cleanup_pile_gap_check/cleanup_results.json`). 8개 완주는 미확인이다. 더 밀집한 배치·다른 간격·낮은 상자도 정체를 보였다. 사용자 선택에 따라 4명·16상자 전체 장면 입력을 유지하며 주변 관측 추론은 적용하지 않는다.

### 4명·16상자 공동 정리 VNC 데모

- 지정된 `ApproachScenarioStage1RescueAtKLClimb50_00008000.pth`를 새 학습 없이 사용하는 평가 전용 task/player와 test/VNC wrapper를 추가했다. 네 명을 십자로 배치하고 바깥 바닥 상자 8개·중앙 2단 상자 8개를 만든다. 네 목표 상자가 목적지 바닥에서 0.5초 안정되면 graph/goal만 교체하여 두 번째 운반을 시작한다. 두 라운드 사이 사람·물체의 물리 상태는 유지한다.
- 옛 rescue 설정은 데모 로더에서만 현재 self-sum runtime으로 대응시킨다. 관측 schema·semantic packet·나머지 계약과 weight/정규화 로드를 검사한다. 기존 checkpoint·일반 학습/평가 로더는 유지한다. 완료 판정은 바닥 안정이며 손 접촉 해제는 요구하지 않는다. 중앙 더미도 실제 물리 물체다.
- 관련 CPU **32개 통과**, Python/셸 문법·문서 경로·diff 확인. GPU 3·seed 42·1환경의 headless와 서버 VNC 모두 **996 control step(시뮬레이션 시간 33.2초)에 두 라운드·8개 운반 완료**를 확인했다. VNC 화면과 두 번째 운반의 카메라 구도를 확인했다. 결과는 `output/box_cleanup_demo_check_final/cleanup_results.json`, `output/box_cleanup_demo/cleanup_results.json`에 저장한다. 이는 해당 scene의 검증이며 모든 반복/seed의 성공을 보장하는 결과는 아니다.

## 2026-10-02

### Unified size RSI reset 성능 최적화

- 새 size RSI 실행 경로만 canonical graph를 GPU에서 배치 생성하고, 고정 asset의 크기별 과제 배정 확률을 최초 한 번 계산하도록 바꿨다. 캐시 RSI 전에 기존 motion/time을 뽑아 버리던 중복 작업도 제거했다. 과제·크기·프레임 분포·보상·AMP 연결·물리 검사 캐시는 유지하며 기존 unified 실행 경로는 변경하지 않았다. 같은 seed의 전체 난수 궤적은 달라질 수 있다.
- 모든 25개 과제 쌍·edge 순열·고정 preset·부분 reset의 graph 동등성 포함 CPU **59개 통과**. GPU 5의 실제 크기 조건부 graph 샘플링은 64환경 **120.65→1.05ms**, 128환경 **351.74→1.06ms**였고 기존 graph의 모든 tensor와 동일했다. 이는 해당 블록의 측정이며 전체 학습 가속 배율은 아니다.
- `output/approach_scenario_stage1_unified_size_rsi_perf_check/`에서 2048환경·`MAX_ITERATIONS=2` 학습과 checkpoint 저장을 완료했다. 기존 물리 캐시 hit, scalar **233개/699값 모두 finite**, 물리 reset 실패·공유 source **0**. 3개 기록 iteration의 환경 진행 시간은 5.02/6.91/7.05초였다. GPU 공유·초기 정책 차이가 있어 이전 학습과 통제된 속도 비교는 아니다. 기존 학습 프로세스는 종료·재시작하지 않았다.

### Unified 행동별 상자 크기·물리 검사 후반 RSI

- `approach_scenario_stage1_unified_size_rsi` config와 전용 train/test/VNC를 추가했다. 과제 비율 5/10/25/30/30, AT·ON_TOP RSI 40/10/40/10, 행동별 독립 XYZ 5cm 격자·고정 밀도 100kg/m³, 실제 bbox 반대각선과 edge별 buffer의 progress를 연결했다. 기존 unified의 state·성공 조건·보상 가중치는 유지한다.
- 실제 asset 크기로 허용 과제를 조건부 샘플링한다. 물리 randomAssignment를 끄고 source 0/1·support 2/3을 고정해 RSI·edge·AMP가 같은 배정을 사용한다. 토큰/edge 순열과 조건부 AMP expert 분포는 유지하며 family 비율만 65/10/25로 맞춘다. 별도 reward 계약으로 기존 checkpoint 재개를 차단한다.
- 실제 크기별 전체 clip 프레임의 0.1초 초기 충격 검사 캐시와 70% 후반/30% 전체 유효 프레임 샘플링을 추가했다. ON_TOP putDown은 source·받침 크기 쌍을 함께 검사한다. 유효 skill이 없으면 같은 과제의 다른 skill/loco로 대체하고 실제 clip·시간으로 pose·속도·AMP 이력을 초기화한다. Reset 시 외부 물체의 깊은 몸 침범을 재검사하며, 요청 대체율·실제 skill 비율·reference/첫 물리 step 성공 지표를 기록한다.
- CPU **46개 통과**, Python/셸 문법·문서 경로·diff 확인. GPU 6에서 **6,224개 크기/skill 조합·4,781,460개 초기 상태**를 검사해 캐시를 생성했다. 최종 설정의 2048환경·1 iteration 학습과 checkpoint 저장, 캐시 재사용을 확인했다. 물리 reset 실패 **0**, 공유 source **0**, TensorBoard scalar **233개 모두 finite**다. 혼합 16환경 및 SIT/CLIMB 각 1환경의 32-step headless 평가와 JSON 저장도 완료했다.
- 최종 검증 output은 `output/approach_scenario_stage1_unified_size_rsi_check_final/`이다. 기본 학습 seed 42로 같은 asset pool 캐시를 재사용하며, 새 seed의 누락 크기는 추가 검사한다. 첫 전체 검사는 약 25분 걸렸다. 이는 초기 충격·실행 경로 검증이며 장기 정책 성공이나 모든 RSI의 초기 own_success를 보장하지 않는다.

## 2026-09-30

### AT goal 마커의 edge 담당 색·슬롯 표시

- Viewer/VNC의 AT goal 점을 고정 빨강 대신 현재 graph의 AT edge owner 색으로 표시한다. marker 표시 여부도 owner 번호가 아닌 실제 AT destination goal 슬롯을 따른다. 레거시 marker는 agent 슬롯 색을 사용한다.
- `config.md` 상단의 학습·로컬 평가·서버 VNC 전체 명령 목록에 unified semantic/shared edge/owner-HOLDING 세 실험을 기재했다.
- Goal 슬롯이 owner 번호와 바뀐 graph를 포함한 관련 CPU 테스트 **6개 통과**, Python 문법·diff 확인. GPU 5의 1환경·16step AT 서버 VNC 평가가 종료되고 결과 JSON을 저장했다. 실행 중이던 학습은 재시작하지 않았다.

### Unified semantic actor·critic EdgeEncoder 공유 실험

- 기존 unified semantic의 과제·관측·보상은 유지하고, actor·critic의 semantic EdgeEncoder 임베딩·MLP·bias projection만 공유하는 별도 config·train/test/VNC·output·checkpoint variant를 추가했다. owner-HOLDING 경로는 변경하지 않았다.
- CPU unified 테스트 **7개 통과**: 두 encoder의 모듈 동일성·모델 파라미터 열거 시 중복 없음·양쪽 gradient·기존 checkpoint 계약 분리를 확인했다. 셸 문법·diff 검사도 통과했다.
- GPU 5에서 2048환경·`MAX_ITERATIONS=1` 별도 `_check` 학습과 checkpoint 저장을 완료했다. TensorBoard scalar **177개/354값 모두 finite**, 저장 weight의 actor/critic edge tensor 8개가 동일했다. 저장 checkpoint로 1환경·32step headless 평가도 완료했다. 이는 실행 경로 확인이며 장기 수렴 검증은 아니다.

### Unified 독립 5과제 · semantic/owner-HOLDING 두 실험

- 두 agent가 HOLDING/SIT/CLIMB/HOLDING+AT/HOLDING+ON_TOP을 5/5/20/35/35로 독립 샘플링하는 4물체 config·train/test/VNC·별도 output을 추가했다. 두 실험은 edge의 담당 HOLDING φ 입력 유무만 다르다. 기존 실험의 설정·실행 중 학습은 유지했다.
- Hᵢ/Oᵢ/Gᵢ 논리 할당과 agent별 별도 ON_TOP 받침을 고정하고, 타입 인코딩 뒤 토큰·GTA·bias를 함께 순열화한 다음 human readout 전에 복원한다. 새 실험에서만 goal quaternion을 identity로, 상자를 0.40×0.30×Z(0.30~0.50m)로, loco 담당 상자를 owner 주변 1~2m로 설정했다. placement 가까운 시작은 0이다.
- 원본 unified 기준 RSI와 carry/sit/climb 조건부 단일 AMP를 연결했다. AMP 10프레임에 과제 one-hot을 각각 추가하고 전문가·리플레이 family를 rollout과 행별 매칭한다. RSI history·부분 reset·정규화에서도 라벨을 유지한다. CLIMB 후반 강제 RSI는 끄고 ON_TOP putDown은 사용하지 않는다. 보상식·TERM 미사용은 유지한다.
- CPU 전체 **139개 통과**. 각각 2048환경·`MAX_ITERATIONS=2`(기존 runner 기준 저장 epoch 3) scratch 학습·저장 완료, scalar **177개 모두 finite**, 물리 reset 실패·공유 담당 물체 **0**. Owner actor/critic 상태 분기 weight 갱신을 확인했다. 토큰 순열의 action/value·gradient 동등성과 AMP family 매칭을 테스트했다.
- 각 checkpoint의 128환경 training-RSI 검사(전체 reset 4회·부분 reset·64step)에서 goal 회전·크기·loco 1~2m·graph/RSI·관측 packet·AMP 이력/정규화 연결을 확인했다. 전용 test wrapper의 64환경·32step headless 평가도 두 실험 모두 완료했다. 검증 자료는 `output/unified_validation/`에 보관했다. 이는 실행 검증이며 장기 수렴이나 모든 자세의 무충돌을 보장하는 결과는 아니다.

### Scenario graph 기준 상자 속도 패널티 수정

- 랜덤 object binding과 무관한 기존 `_agent_box_assignment`로 속도 패널티를 계산하던 오류를 수정했다. 현재 위치·pre-step history·부분 reset history 모두 graph의 담당 HOLDING/SIT/CLIMB 물체를 동일하게 조회한다. 기존 비-scenario 경로는 기존 할당을 유지한다.
- 소유자/물리 slot/edge 순서 변경·부분 reset·기존 경로 회귀를 포함한 관련 CPU 테스트 **19개 통과**. 저장 owner checkpoint로 128환경·64step 시뮬레이션에서 실제 패널티와 graph 기준 재계산값의 오차 **0**을 확인했다. 현재 실행 중인 학습은 재시작하지 않았다.
- 추가 분석은 `output/placement_code_audit/`에 보관했다. 기본 loco source 배치는 과거 owner 주변 1~2m와 현재 arena 반경 4.5m로 다르며, GTA goal 회전은 여전히 같은 번호 human heading을 사용한다. 상자 크기·spawn·goal 회전은 이번에 변경하지 않았다.

## 2026-09-29

### Paired placement · 담당자 HOLDING 상태 edge 입력

- 기존 4물체 paired placement의 과제·보상·RSI·AMP·가까운 시작 없음 설정을 유지한 별도 config·train/test/VNC·output을 추가했다. 새 관측 패킷은 edge당 `valid/src/dst/relation/owner/owner_holding_state` 6개 값이며, AT/ON_TOP에는 동일 owner·source의 선행 HOLDING `φ`, 다른 유효 edge에는 1을 넣는다. actor·critic의 기존 semantic 64차원 뒤에서 placement edge에만 상태 잔차를 더하고 checkpoint 계약을 분리했다.
- 관련 CPU 테스트 **32개 통과**, Python·셸 문법과 diff 확인. GPU 0의 기존 학습은 유지한 채 2048환경·1 iteration 별도 output 학습과 checkpoint 저장을 완료했다. TensorBoard scalar **137개 모두 finite**, AT+AT/ON_TOP+ON_TOP 샘플 비율 **0.501/0.499**였고 actor·critic 상태 분기 weight가 모두 0에서 갱신됐다. 저장 checkpoint로 각 1환경·32-step headless AT/ON_TOP 평가를 완료했다. 이는 실행 경로 검증이며 장기 수렴 결과는 아니다.

### 같은 과제 쌍 · 가까운 시작 없는 Stage 1 비교

- 기존 3물체 sampler에서 두 agent 모두 AT만 받는 config와, 새 4물체 sampler에서 scene별 AT+AT/ON_TOP+ON_TOP을 50/50으로 뽑는 config를 분리했다. 두 실험의 가까운 시작 확률은 0이며 34번의 RSI·AMP·보상식은 유지한다. 4물체 ON_TOP은 서로 다른 source 2개와 support 2개를 사용한다. 각 config에 train/test/VNC와 별도 output·checkpoint variant를 연결했다.
- 관련 CPU 테스트 **53개 통과**, 스크립트 문법·diff 확인. GPU 0에서 각각 2048환경·1 iteration scratch 학습과 checkpoint 저장을 완료했다. 물리 reset 실패는 둘 다 **0**, paired placement의 AT/ON_TOP 비율은 **0.509/0.491**이었다. 저장 checkpoint로 각각 1환경·32-step headless AT/ON_TOP 평가를 마쳤고, ON_TOP 평가에서 물리 상자 4개가 서로 다른 edge 역할에 연결됨을 확인했다. 장기 정책 학습 결과는 아직 확인하지 않았다.

## 2026-09-28

### 37번 ON_TOP putDown RSI 실험

- 36번을 유지하고 별도 config·train/test/VNC·output·checkpoint variant에서 ON_TOP `putDown` RSI 10%를 추가했다. 원본 후기 모션은 받침 상자와 겹치므로 45~70% 구간에서 시작하며, 받침을 모션 최종 XY에 배치하고 상자·사람 침투 reset을 재추첨한다. AT `putDown`과 36번 과제·보상 분포는 유지한다.
- 관련 CPU 테스트 31개, 셸·Python 문법·diff 검사 통과. GPU 0에서 2048환경·1 iteration 학습과 checkpoint 저장 완료: ON_TOP `putDown` RSI 비율 9.46%, 물리 reset 실패 0·재시도 98, scalar 138개 모두 finite. 저장 checkpoint의 1환경·32-step ON_TOP headless 평가와 JSON 저장도 완료했다. 이는 실행 경로 확인이며 장기 수렴 검증은 아니다.

### 36번 AT/ON_TOP 집중 학습 실험

- 34번의 reward·RSI·가까운 시작을 유지하고 HOLDING/SIT/CLIMB/HOLDING+AT/HOLDING+ON_TOP 샘플 비율만 0/0/0/50/50으로 바꾼 별도 Stage 1 config·train/test/VNC·output을 추가했다. 새 variant로 checkpoint 보상 계약을 분리했다.
- 관련 CPU 테스트 18개, 스크립트 문법·문서 링크·diff 검사 통과. GPU 0에서 2048환경·1 iteration 확인과 checkpoint 저장을 완료했다. AT/ON_TOP 샘플 비율 각각 0.5, 물리 reset 실패 0, scalar 137개 모두 finite였다. 저장 checkpoint의 1환경·32-step headless 평가와 JSON 저장도 완료했다. 이는 실행 경로 검증이며 장기 수렴 검증은 아니다.

### 뷰어 상자 색을 과제 배정에 연결

- Stage 1·2 로컬/VNC 뷰어에서 graph의 실제 대상 상자를 에이전트 색으로 칠한다. 공동 대상은 노란색, 비대상은 회색이다. 물리 상자 순서와 graph 배정이 달라도 색이 맞도록 변경했다.
- Stage 1 ON_TOP, Stage 2 place_climb/place_stack graph를 사용한 색 지정 호출 검증과 Python 문법 확인을 마쳤다. 실제 뷰어 화면 확인은 아직 하지 않았다.

## 2026-09-27

### Stage 2 SIT plane·34번 보상/독립 과제 변형

- 기존 29번과 분리한 `approach_stage2_coordination_sit_plane` config·train/test/VNC·output을 추가했다. SIT도 상판 안쪽 10%와 높이 오차 7cm로 판정하고, 자기 edge 합계·동료 edge 평균에 0.9/0.1 팀 보상을 적용한다. 에피소드는 place_climb/place_sit/place_stack/독립 20/20/40/20이며, 독립 과제 내부는 34번의 10/10/10/35/35와 RSI·가까운 시작 일정을 따른다. 협력 과제의 RSI는 기존 Stage 2 규칙을 유지한다.
- Stage 2 checkpoint 로드에서 reward config 불일치를 검사하도록 수정했다. 34번 epoch 500 checkpoint로 GPU 6에서 2048환경·1 iteration 이식 학습과 1환경 place_sit 평가를 완료했다. 물리 reset 실패 0, scalar 177개 모두 finite, 관찰·AMP 통계 이식 확인. 관련 CPU 테스트 22개 통과. 장기 수렴은 아직 확인하지 않았다.

### output_etc Git 제외

- `.gitignore`에 `/output_etc/`를 추가했다. `git check-ignore`로 내부 파일 제외와 `git status`에서 해당 폴더가 사라진 것을 확인했다.

### VNC 평가 결과 기본 경로 통일

- 현재 실험 9개의 VNC wrapper가 기본 `OUTPUT_PATH`를 각 실험의 `output/<실험명>`으로 전달하도록 바꿨다. 로컬 test와 같은 위치에 `metrics/`, `diagnostics/`가 생기며 학습 checkpoint의 run 디렉터리와 분리된다. 기존 `_vnc` 폴더와 실행 중인 학습은 변경하지 않았다. 셸 문법과 기본 경로를 확인했다.

### 34·35번 자기 보상 합계와 가까운 시작 일정 비교

- 33번의 AT reset 수정·state/progress/성공 식을 유지하고, 34번은 자기 edge 보상 합계·동료 edge 평균과 HOLDING/SIT/CLIMB/HOLDING+AT/HOLDING+ON_TOP 10/10/10/35/35 샘플링을 적용했다. 35번은 34번에서 가까운 시작 80%를 60만 per-env step까지 유지하고 120만 step까지 30%로 감소시킨다. 두 실험은 별도 config·train/test/VNC·output에서 scratch로 시작하며 33번과 기존 학습 프로세스는 변경하지 않았다.
- 관련 Stage 1·2 CPU 테스트 **30개 통과**, 스크립트 문법·diff 확인. GPU 6에서 각각 2048환경·1 iteration 학습과 checkpoint 저장을 완료했다. 두 실험 모두 물리 reset 실패 **0**, 공유 주 물체 **0**, scalar **177개 모두 finite**였고, 샘플 비율은 목표값에 근접했다. 짧은 진단 표본에서 AT 두 번째 목표 slot 높이도 정상이다. 저장 checkpoint의 1환경·32-step AT headless 평가와 JSON 생성도 각각 완료했다. 이는 실행 경로 검증이며 장기 수렴 검증은 아니다.

### 33번 AT 목표 reset 수정

- 독립 graph의 AT가 두 번째 목표 slot을 지정해도 reset이 첫 번째 slot에 목표를 배치하던 선택 오류를 수정했다. 32번의 보상·샘플링·RSI를 유지한 33번 config와 전용 train/test/VNC·output을 추가했다. 공통 reset 수정은 이후 새로 시작하는 27~32번에도 적용되며, 실행 중인 프로세스는 변경하지 않았다.
- 관련 CPU 테스트 **23개 통과**. GPU 6에서 2048환경·1 iteration scratch 학습과 checkpoint 저장 완료. 진단 CSV의 AT 목표 높이는 0보다 컸고, TensorBoard scalar **177개 모두 finite**, 공유 주 물체 **0**이었다. 이는 AT 목표 배치와 실행 경로 검증이며 정책 수렴 검증은 아니다.

## 2026-09-25

### 과거 실험 진입점 정리

- 실행 중인 27·28·30·32번과 Stage 2 29번, 비교용 31번 및 원본 1번을 남겼다. Git에 있던 9~26번 config 18개·전용 실행 스크립트 50개·전용 테스트 11개·오래된 설계 문서 10개·미사용 graph 예제 2개·미사용 학습 smoke config 1개를 제거했다. 현재 문서를 다시 작성하고 25·26번 전용 sampler 분기를 제거했다. 공유 graph·reward 함수는 현재 실험도 사용하므로 유지했다.
- 기존 학습 프로세스와 output은 변경하지 않았다. 삭제된 config·스크립트는 Git 이력에서 확인할 수 있다.
- 남긴 테스트 전체 **109개 통과**, 실행 스크립트 문법·Markdown 로컬 링크·diff 확인. 새 코드로 32번 2048환경·1 iteration 확인에서 checkpoint 저장, 물리 reset 실패 **0**·공유 주 물체 **0**, scalar **177개 모두 finite**였다. 이는 실행 경로 검증이며 장기 수렴 검증은 아니다.

### 32번 보상식 유지 커리큘럼

- 31번의 가까운 AT/ON_TOP 시작·후반 CLIMB RSI·과제/RSI 비율을 유지하고, AT/ON_TOP progress와 CLIMB state만 30번 식으로 되돌린 별도 schema 9 variant와 train/test/VNC를 추가했다. 31번 설정과 checkpoint는 그대로 두며 32번은 scratch로 시작한다.
- 관련 CPU 테스트 **17개 통과**. 2048환경·`MAX_ITERATIONS=1` GPU 1 확인에서 checkpoint 저장, 물리 reset 실패 **0**·재시도 **135회**, scalar **177개 모두 finite**, 템플릿 비율 약 **0.095/0.105/0.298/0.255/0.247**을 확인했다. 저장 checkpoint의 1환경·20-step ON_TOP headless 평가와 JSON 생성도 완료했다. 이는 실행 경로 검증이며 정책 수렴 검증은 아니다.

### 30번 SIT plane 정규화·31번 hard-skill 시작 상태 실험

- 28번을 보존하고 새 schema 9 variant 두 개를 분리했다. 30번은 SIT 성공을 root XY의 상자 윗면 안쪽 10%·목표 Z ±7cm로 바꾸고 agent별 active-edge 보상을 평균내 단일/두 edge 최대 task reward를 0.6으로 맞춘다. 31번은 AT·ON_TOP 가까운 시작 비율을 환경 step 0→300,000에서 80→30%로 낮추고, 장거리 progress·CLIMB root/발 높이 state·hard-skill sampling을 추가한다. 두 실험 모두 scratch이며 팀 공유 0.9/0.1은 유지한다.
- 기존 plane CSV 표본에서 AT/ON_TOP은 에피소드 첫 관측 3.33/4.27m에서 마지막 관측 5.70/6.36m로 목표에서 멀어졌다. CLIMB reference 7개·505프레임을 상자 높이 0.4m·XY 0.5m로 실측한 결과 성공 가능 29프레임 전부 원본 RSI 제외 구간에 있었다. 31번은 CLIMB RSI의 70%를 후반 65~98% 구간에 배정하며 기존 데이터셋과 27·28번 학습을 변경하지 않는다.
- 관련 CPU **34개 통과**, 스크립트 문법·diff 확인. GPU 1에서 두 실험 각각 2048환경·1 iteration scratch와 checkpoint 저장, 저장 checkpoint의 1환경·32-step headless 평가를 완료했다. 30번 물리 리셋 실패 0·재시도 121회, 31번 실패 0·재시도 156회, 각 scalar 177개 finite였다. 31번 CLIMB root/발 조건 동시 통과율은 후반 RSI 이전 짧은 확인 0%에서 1.26%로 바뀌었다. 이는 초기 상태·실행 경로 확인이며 장기 정책 수렴 검증은 아니다.

### 29번 Stage 2 coordination 전이 실험

- 28번 plane 성공 조건과 사용자 지정 전역 팀 보상 0.9/0.1을 유지하고, `place_climb/place_sit/place_stack` 협력 90%와 독립 bundle 10%를 샘플하는 schema 10 config·train/test/VNC를 추가했다. Stage 1 schema 9 semantic `.pth`를 `STAGE1_CHECKPOINT`로 선택해 actor/critic/AMP weight와 actor·AMP 관측 통계를 엄격히 복사하고, actor encoder·관측 RMS를 고정했다. 새 grounded-edge cross-attention과 Stage 1 head의 `[W,0]` 확장으로 협력 입력을 연결했다. Stage 2 resume는 별도 `RESUME_CHECKPOINT`로 분리했다.
- 공유 물체 SIT/CLIMB agent는 loco에서 시작하고, downstream 성공 또는 전체 graph 성공으로 시작한 reset은 재시도한다. 3인 공유 물체·4인 독립 쌍의 explicit 평가 graph와 가변 agent/edge 입력을 추가했다. Stage 2 taxonomy의 CLIMB 누락과 평가 시 학습 graph 동일성 검사를 수정했다.
- CPU 전체 **278개 통과**, 2048환경·1 iteration GPU 학습 및 checkpoint 저장 완료. 저장 checkpoint의 첫 head 협력 입력 weight norm은 **0.0556**, TensorBoard scalar **162개 모두 finite**, 물리 reset 실패 **0**이었다. 실제 Stage 1 plane checkpoint에서 전이 직후 동일 입력의 action mean·sigma 최대 차이는 각각 **0**이었다. 같은 Stage 2 checkpoint로 2·3·4인 1환경·20-step headless 평가와 JSON 저장을 완료했다. 2048환경 Stage 2 resume 뒤 협력 입력 weight norm은 **0.0817**로 변했고 attention weight도 갱신됐으며 actor encoder 75개 tensor와 actor RMS는 동일했다. Resume scalar **174개 모두 finite**, 물리 reset 실패 **0**이었다. 이는 실행·가변 입력 경로 검증이며 협력 성공률·장기 수렴 검증은 아니다.

## 2026-09-24

### 28번 stage1_plane 독립 시나리오 실험

- 27번의 독립 물체·goal 배정, 다섯 행동 각 0.2, RSI·AMP와 팀 보상 0.9/0.1을 유지했다. CLIMB·ON_TOP의 current-success만 원격 `ontop_climb_plane` 방식으로 변경했다. ON_TOP은 source 중심 XY의 support 윗면 안쪽 10% 영역 및 면 높이 오차 1mm, CLIMB은 root XY의 안쪽 5% 영역 및 root 높이 오차 20cm·평균 발 높이 오차 7cm를 요구한다. 연속 state·progress 보상은 기존 중심점 기준이며, 새 reward variant·config·train/test/VNC·output과 checkpoint 계약을 분리했다.
- CPU 전체 **274개 통과**, shell 문법·diff 공백 확인. GPU 3·2048환경·1 iteration scratch에서 다섯 template 비율 **0.194~0.205**, shared primary object **0**, 물리 reset 실패 **0**, TensorBoard scalar **174개 모두 finite**, checkpoint 저장을 확인했다. 저장 checkpoint로 1환경·32-step ON_TOP headless 평가와 JSON 저장도 종료했다. 이는 실행 경로 확인이며 장기 학습 수렴 검증은 아니다.
- GPU 3 본학습 로그 점검에서 CSV의 `self_task_contribution`/`other_task_contribution`이 Stage 1 보상을 자기 100%로 잘못 표시하는 문제를 발견해 config의 0.9/0.1 비율을 따르도록 수정했다. 실제 학습 reward와 TensorBoard 공유 지표는 이미 0.9/0.1로 정상이며, 실행 중 프로세스는 재시작하지 않아 해당 프로세스의 CSV 표시는 이전 방식으로 남는다. 관련 CPU 테스트 **17개 통과**·diff 공백 확인.

### 27번 독립 물체 binding + standalone CLIMB scratch 실험

- 26번의 5-template reward·0.9/0.1 팀 보상·AMP·template RSI를 유지하고, 물체 3개를 reset마다 무작위 순열로 두 agent의 서로 다른 primary object와 남은 ON_TOP support에 배정한다. AT goal도 비공유 무작위 배정하며 동시 ON_TOP은 제외한다. 유효한 template pair의 각 agent 주변 확률은 기존 0.2를 유지한다. 새 sampler/config/train/test/VNC/output과 별도 checkpoint variant를 연결했다.
- CPU 전체 **270개 통과**. GPU 2·2048환경·1 iteration 확인에서 5-template 비율 **0.196~0.203**, shared primary object **0**, 물리 reset 실패 **0**, TensorBoard scalar **172개 모두 finite**, GTA position p95 **4.4012m**, checkpoint 저장을 확인했다. 저장 checkpoint로 1환경·32-step ON_TOP headless 평가와 JSON 저장도 종료했다. 이는 실행 경로 검증이며 장기 수렴·물체 수 확대 추론은 미검증이다.
