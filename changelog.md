# Changelog

현재 실행법은 [config](markdowns/config.md), 코드 위치는 [structure](markdowns/structure.md)를 따른다. 정리 전 문서·변경 기록은 [보관 자료](markdowns/legacy/pre_four_configs/README.md)에 있다.

## 2026-11-18

### Box-only PUSH 영역을 최종 스폰에 매칭

- Box-only 학습의 2인 PUSH에 최종 배치·캐시 복원 후 상자 로컬 X축으로 영역을 확정하는 경로를 추가했다. 개인 목표·GTA 토큰·접촉 판정이 이 ID를 사용한다.
- CPU 테스트 53개 통과. 500epoch·65,536,000프레임 저장본의 optimizer와 재개 설정을 확인했다. 기존 GPU6 MPS client4061923은 실행 상태로 두었다. [재개 명령](markdowns/USAGE.md#500epoch-재개)·[설정 검증](output_etc/check/warehouse_boxonly_ordered_spawn/resume_config.log).

### DistillV4 2000 가상상자 전용 제로샷 녹화

- 제로샷 viewer의 `--box-only`를 가상상자 전용 환경에 연결했다. 원본 DistillV4 2000 모델과 RMS로 행동을 계산하고, 학습 환경의 1.2m 가상상자 전환 상태와 실제 출구 통과 측정을 재사용한다.
- CPU 테스트 59개 통과. GPU6 MPS에서 PUSH 2명 + BEFORE 1쌍의 ON_TOP 후행을 고정하고 seed47·49·51을 녹화했다. Seed49는 선행과 후행 모두 출구를 통과했고 후행은 694step에 ON_TOP을 완료했다. MP4 세 편의 1600×900·30fps와 전체 디코딩을 확인했다. [영상과 검증 결과](output_etc/check/warehouse_boxonly_distillv4_2000/verification.json).

### Warehouse 가상상자 전용 변형

- `stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint_ontop_boxonly`을 추가했다. DistillV4 2000에서 전이하며 선행과 ON_TOP 후행의 1.2m 가상상자 접근과 원래 목표를 사용한다. BEFORE 2쌍 비율은 20%다. 선행 입구 보상은 문 통과 또는 첫 HOLDING에 포화하고 집기 포화 상태는 부분 리셋에서 초기화한다.
- GPU 6·7번 전용 실행 명령과 역할별 집기·실제 출구 통과·목표 성공 평가를 추가했다. CPU 테스트 75개 통과. GPU6 MPS에서 동일한 1024환경·seed42로 확인 학습을 실행했다. `fps total`은 새 변형 14.5k–16.2k, 기존 1.2m 구성 14.5k–15.9k였다. 고정 조건 캐시 리셋 중앙값은 각각 27.01ms·27.03ms였다.
- 새 저장본을 16환경·seed0에서 재로딩해 900step을 완료했다. 선행 14명 중 9명이 집었고 1명이 출구 통과 및 목표에 성공했다. ON_TOP 후행 4명 중 1명이 집었고 출구 통과는 0건이었다. [검증 결과](output_etc/check/warehouse_waypoint_boxonly/verification.json).

### Warehouse 1.2m 선행·ON_TOP 후행 waypoint 센서 학습

- 기존 선행 전용 실험에서 분리한 `stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint_ontop`을 추가했다. DistillV4 2000에서 새로 전이하며 선행·ON_TOP 후행이 공유 입구 좌표와 독립 전환 상태를 사용한다. 진입·목표 복귀 반경은 1.2m다. BEFORE 2쌍 20%에는 양쪽 입구 유도를 끈다.
- 후행 ID·단계를 PPO 관측에 저장하고 활성 역할 장면을 한 배치에서 인코딩한다. 초기 배치·AMP 이력 캐시와 PPO 동결 actor 특징 재사용을 이어 쓴다. 제로샷 ON_TOP viewer를 1.2m로 맞추고 새 학습·테스트·VNC·평가 명령을 추가했다.
- CPU 테스트 64개 통과. GPU6 MPS의 4명·1024환경·8토큰·seed42 확인 학습에서 BEFORE 2쌍 초기 샘플은 614/3072였고, `fps total`은 새 구성 14.9k–15.9k, 선행 전용 재실행 14.2k–14.9k였다. 1024환경 고정 조건에서 상태·AMP 및 비리셋 행 보존을 확인했고 리셋 중앙값은 캐시 복원 27ms, 새 배치 563ms였다.
- 새 체크포인트를 8환경 평가에서 재로딩해 900step을 마쳤다. DistillV4 2000 별도 추론에서는 1.2m로 ON_TOP 후행 목표 전환 및 복귀가 각각 3건이었다. [검증 결과](output_etc/check/warehouse_waypoint_ontop/verification.json).

### ON_TOP 후행 출구 정체 추론 진단

- GPU6 MPS에서 DistillV4 2000의 새 단일 환경 seed44–51을 검증했다. 방 진입 5건, 후행 집기 3건, 입구 목표 복귀 및 쌓기 성공 2건이었다. [추론과 좌표 진단](output_etc/check/warehouse_ontop_exit_samples/summary.json).
- Seed43은 문 옆 정면 벽, seed49는 방 측벽에 상자가 접근해 정체했다. 집기 이후 입구까지의 최소 상자 거리는 각각 2.623m, 3.393m였다. Seed47은 479step에 원래 목표로 복귀하고 537step에 출구를 통과해 619step에 쌓기에 성공했다. Seed47·49의 추론과 녹화 이벤트 일치, 900프레임 MP4 전체 디코딩을 확인했다. [궤적 비교](output_etc/check/warehouse_ontop_exit_samples/exit_paths.png).

### ON_TOP 후행 운반 목표를 입구로 유도

- DistillV4 2000 추론에서 ON_TOP 후행이 실제 상자를 들면 GTA의 쌓기 목표 alias를 선행과 같은 입구 좌표로 옮긴다. 상자가 입구 1.3m 이내에 도달하면 원래 쌓기 목표로 복귀한다. USAGE·structure에 전환 조건과 영상 표시를 기록했다.
- CPU 45 passed. GPU6 MPS 16환경 추론에서 후행 입구 목표 전환 5건, 원래 목표 복귀 3건, ON_TOP 성공 2건을 확인했다. [16환경 결과](output_etc/check/warehouse_ontop_door_goal/screen/results.json).
- 단일 환경 seed43은 420step에 후행 목표가 입구로 전환됐으며 900step까지 복귀하지 않았다. seed42는 후행이 입구 도달 전에 낙상했다. 두 MP4의 1600×900·30fps와 전체 디코딩을 확인했다. [seed43](output_etc/check/warehouse_ontop_door_goal/seed43/warehouse.mp4)·[seed42](output_etc/check/warehouse_ontop_door_goal/seed42/warehouse.mp4).

### GPU6 학습 중단 및 ON_TOP 후행 입구 유도 제로샷 추론

- 사용자 요청으로 GPU6 waypoint(2518467)와 doorprogress(2506636)를 MPS에서 각각 분리해 중단했다. 두 서비스 inactive, 저장본 waypoint4000·doorprogress5000 SHA 일치, GPU6 MPS 서버와 다른 GPU 학습 PID 유지 확인. [중단 기록](output_etc/check/warehouse_ontop_waypoint/stopped_training.json).
- DistillV4 2000 viewer에 ON_TOP 후행 전용 입구 유도를 추가했다. 선행과 같은 입구 좌표를 후행 payload GTA에 넣고 1.3m 진입 시 실제 상자로 전환한다. 목표는 기존 쌓기 위치 alias이며, 선행과 후행의 장면을 분리해 역할별 출력을 선택한다. 시각 확인용 ON_TOP 고정과 MP4 저장 옵션을 추가했다. config·USAGE·structure를 갱신했다.
- CPU 44 passed. GPU6 MPS·seed42·16환경 추론에서 후행 실제 상자 전환14, 방 진입6, HOLDING4, ON_TOP 성공0이었다. 단일 환경 녹화 seed43은 전환72·방 진입159·HOLDING420step 후 900step을 마쳤다. seed42는 진입 전 361step에 낙상했다. MP4 2개의 1600×900·30fps 및 전체 디코딩을 확인했다. [16환경 결과](output_etc/check/warehouse_ontop_waypoint/screen/results.json)·[seed43](output_etc/check/warehouse_ontop_waypoint/seed43/warehouse.mp4)·[seed42](output_etc/check/warehouse_ontop_waypoint/seed42/warehouse.mp4).

### GPU6 MPS waypoint 백그라운드 본학습 시작

- 사용자 지정 waypoint train4/1024/8 명령을 `tokenhsi-warehouse-waypoint-mps-gpu6-20261118-020124.service`·learner2518467로 시작했다. user systemd app.slice·Restart=no·Linger=yes로 대화/터미널 종료와 독립적으로 실행한다. GPU6 MPS·conda tokenhsi·900step/seed42·DistillV4 2000 fresh 전이, 역할별 GTA·고정1.3m·BEFORE 2쌍20%를 유지하며 CHECK/MAX_ITERATIONS/resume/평가 override는 해제했다. 기존 doorprogress MPS client2506636의 시작 tick은 보존했다. config·USAGE에 현재 본학습과 운영 경로를 갱신했다.
- 실제 argv·GPU6 MPS client 연결·Linger·저장 config/코드 hash·전이 source SHA를 확인했다. epoch1–2/frame0→131072의 scalar778종 finite, 초기 fps total11572–12173를 확인했다. 초기2회이며 장기 평균/성공률 검증은 아니다. [실행 정보](output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint/launch_mps_gpu6_20261118-020124/launch.json)·[시작 검증](output_etc/check/waypoint_background_launch/verification.json).

### GPU6 기존 doorprogress 직접 CUDA 학습을 MPS로 재개

- 사용자 요청으로 기존 서비스 `tokenhsi-warehouse-doorprogress-gatefix-gpu6-20261117-201917.service`·learner1717136만 중지하고, 마지막 정기 저장1500epoch/frame196608000에서 모델·optimizer·통계를 복원했다. 종료 시점 약1636epoch 이후의 미저장 진행은 포함하지 않는다. 코드/config hash6개와 checkpoint SHA 보존을 확인했다. 기존 cwd `TokenHSI_stage12_warehouse_fix_reward`·conda `tokenhsi_jhh`·4명/1024환경/8토큰/900step/seed42와 보상·배치 설정은 유지한다.
- 서비스 `tokenhsi-warehouse-doorprogress-resume-mps-gpu6-20261118-015507.service`·learner2506636을 app.slice·Restart=no·Linger=yes로 시작했다. GPU6 MPS server2420795의 실제 client 연결·argv resume/원본 checkpoint·저장 환경의 일치를 확인했다. 첫 관측 epoch1501–1505의 frame 증가와 scalar772종 finite를 확인했다. 초기 FPS는 약16607–18077이며 속도 개선 대조가 아니다. config·USAGE 운영 경로를 갱신했다. [실행 정보](output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress/launch_resume_mps_gpu6_20261118-015507/launch.json)·[전환 검증](output_etc/check/gpu6_mps_resume/verification.json).

### Warehouse waypoint 역할별 GTA 분리·PPO 동결 특징 재사용

- 사용자 확인에 따라 공통 가상 장면 제안을 제거하고 실제/선행용 GTA를 독립 batch 행으로 묶었다. PUSH 두 명·후행은 기존 장면 출력, 활성 선행만 입구 가상 장면 출력을 선택한다. 후행의 최종 목표 alias와 선행의 입구 payload/Target은 별도 토큰이며 원래 최종 목표 기준 alias 선처리를 유지한다. encoder/coordination 호출은 각 branch당1회지만 장면별 계산은 유지한다. 기존 실행 명령·checkpoint 형식·반경1.3m·BEFORE 2쌍20%·보상·동결 범위는 같다.
- 동결 actor 특징을 PPO dataset에 저장해 6회 mini-epoch에서 재사용한다. 학습 가능한 센서/head/critic/AMP는 계속 갱신한다. 배치 캐시 키에서 역할 목록의 내부 나열 순서만 제외하고 실제 actor/predecessor/near/region·물체 크기·기하 조건은 유지했다. 실행 중인 프로세스에는 자동 적용되지 않는다. USAGE·명세·structure를 갱신했다.
- CPU 관련75 passed. GPU6/MPS·1024환경 입력·동일 저장 가중치의 actor+critic forward를3회 측정해 중앙값40.19→31.43ms(약21.8% 단축), 최대 actor 차이5.7e-7·critic 차이2.5e-6, 캐시 특징으로6회 역전파 finite를 확인했다. 이는 전체 학습 FPS 비교가 아니다. [측정](output_etc/check/warehouse_waypoint_roles/benchmark.json)·[CPU](output_etc/check/warehouse_waypoint_roles/cpu.log).
- GPU6/MPS·4명/1024환경/8토큰 확인 학습2회 업데이트·저장이 exit0이었다. scalar780종 finite, 원본 actor encoder/coordination/sigma·입력 RMS의 exact 보존과 head/critic/AMP 갱신을 확인했다. 짧은 구간 fps total은8932/8223이며 장기 또는 동일 조건 전체 FPS 개선율로 해석하지 않는다. 초기 PUSH+BEFORE 캐시 적중은16.0%였다(이전 본학습 로그의 초기3.7% 대비, 동일 난수 대조는 아님). 확인 프로세스는 종료했고 본학습은 시작하지 않았다. [학습 검증](output_etc/check/warehouse_waypoint_roles/training_verification.json).

### Warehouse waypoint 센서 학습·GPU 배치 캐시 추가

- 별도 `stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint` YAML·train/train_vnc/test/vnc/eval을 추가했다. 80% PUSH2+BEFORE 선행에 고정1.3m의 진입/집기/입구 운반/최종 목표 전환을 적용하고, 20% BEFORE 2쌍은 기존 입력/배치를 유지한다. 기존 보상·gate·동결 범위를 유지하고 전환 상태를 PPO 관측에 저장한다. GPU 배치 캐시를 두 변형으로 분리하고 중복 작업자 배치·사후 벽 간격 탈락·전체 상태 복사를 줄였다. config·USAGE·structure·명세를 갱신했다.
- CPU 관련90 passed. GPU6 직접 CUDA·1024환경에서4회 업데이트·저장 checkpoint의 scalar780종 finite, actor encoder/coordination/sigma·입력 RMS의 원본 일치를 확인했다. GPU7 최초 확인은 사용자 GPU6 한정 지시 전에 시작했으며 이후 확인 실행을 종료했다. GPU6 속도 대조는 사용자가 직접 실행하기로 하여 중단했다. 리셋 속도 향상 수치·장기 성능·실제 resume/VNC·별도 전체 기하 검증은 아직 확정하지 않았다. 마지막 캐시 준비 관측 생략 최적화는 별도 GPU 검증 전이다. [검증](output_etc/check/warehouse_waypoint_training/training_verification.json).
- 사용자 요청으로 GPU6의 종료 PID111986·활성 소켓 부재를 확인하고 오래된 pipe/log를 `/tmp/mps-test-1003/gpu6/stale.bH32zI7Z`에 보관한 뒤 MPS 서버2420795를 시작했다. 기존 학습은 유지했다. 새 wrapper는 GPU6만 허용하며 기본은 MPS 자동 연결이다. 이번 작업의 확인 학습/속도 대조는 종료했고 본학습은 시작하지 않았다.

### Warehouse 제로샷 탑뷰·선행 상자 벽 간격 확대

- 사용자 요청으로 전용 viewer 기본 카메라를 탑뷰로 바꾸고, 선행 상자 외곽과 앞·뒤·측벽의 최소 간격이 1.2m인 초기 배치만 허용했다. 부족하면 기존 배치 전체를 재샘플링해 다른 물체·사람 여유와 near 조건을 함께 유지한다. 학습 YAML과 정책·목표 전환 반경은 그대로다. 처음 확인한 0.7m 조건은 seed42 배치를 바꾸지 못해 1.2m로 확정했다. USAGE·structure를 갱신했다.
- GPU7·32환경×3 reset에서 최소 벽 간격 1.2026m 이상, 기존 사람/상자/near 조건, 원본 가중치/RMS·비선행 행동 보존과 물리 step finite를 확인했다. [검증](output_etc/check/warehouse_waypoint/wall_clearance12/geometry/verification.json).
- GPU7·6082·seed42·switch-margin1.0·1환경900step을 탑뷰 1600×900·30fps·30초 MP4로 녹화했다. 진입 전환54·집기 목표 활성166·최종 목표 복귀716step, 마지막 선행 AT 성공·낙상 종료0을 확인했다. 이는 한 장면 결과이며 성공률 비교가 아니다. 900프레임 전체 디코딩과 화면을 확인했다. [영상](output_etc/check/warehouse_waypoint/topview_clearance12_seed42/warehouse_seed42.mp4)·[결과](output_etc/check/warehouse_waypoint/topview_clearance12_seed42/results.json).

## 2026-11-17

### Warehouse 제로샷 집기 후 입구 가상 목표 추가

- 사용자 승인으로 기존 진입 가상 상자 다음에 운반 가상 목표 단계를 추가했다. 선행의 기존 HOLDING 성공 시 GTA Target XYZ만 입구로 교체하고, 운반 상자 중심 XY가 기존 `payload XY 반대각 + --switch-margin` 안에 도달하면 원래 최종 Target으로 복귀한다. 전환은 reset 전까지 유지한다. 원래 목표 기준 후행 alias 선처리·선행만 행동 교체·실제 목표/보상 보존은 동일하다. 입구 목표 활성은 주황색이며 집기/복귀 step을 결과에 기록한다. USAGE·config·structure를 갱신했다.
- CPU 19 passed. GPU7·32환경×3 reset에서 가상 상자/목표 혼합 입력의 비선행 행동·비활성 선행 행동 보존, HOLDING 선택, 원본 가중치/RMS·실제 목표 보존과 물리 step finite를 확인했다. GPU7 VNC·seed42·900step은 exit0, 진입 전환118·집기 목표 활성184·반경1.407m였다. 이 에피소드에서는 상자가 입구 반경에 도달하지 않아 최종 목표 복귀와 최종 AT 성공은 발생하지 않았다(낙상 종료0). 거리 경계·복귀 래치·reset은 CPU 검사로 확인했으며 운반 성공률 개선은 검증하지 않았다. 기존6081·학습 프로세스는 유지하고 검증은6082에서 종료했다. [검증](output_etc/check/warehouse_waypoint/outbound/geometry/verification.json)·[VNC 결과](output_etc/check/warehouse_waypoint/outbound/render/results.json).

### DistillV4 Warehouse 입구 가상 상자 제로샷 viewer

- 사용자 승인으로 추론 전용 `qualitative/warehouse_waypoint{,_policy,_task,_viewer}.py`·test/VNC wrapper를 추가했다. 원본 Distill900V4 2000의 가중치·입력 RMS를 로드하고 센서는 입력하지 않는다. 원래 목표의 후행 alias 처리 후 선행 payload GTA XYZ만 문 입구로 교체하며 선행 행동만 선택한다. 선행 root가 payload XY 반대각+`--switch-margin` 안으로 들어오면 실제 상자로 복귀하고 reset 전까지 유지한다. 실제 물체·목표·보상·성공 판정과 학습 코드는 유지했다.
- 장면은 Default/closed·PUSH2+BEFORE1쌍으로 고정했다. 선행은 앞벽 기준 x=[-3.5,-2.5]m·|y|=[2.4,3.2]m의 실제 상자 쪽, 후행은 기존 영역에서 다시 배치한다. 기존 사람/상자 여유·접근 거리·near 경로 조건을 검사한다. GPU5/6/7 선택, 검증된 MPS 또는 직접 CUDA, 기존 프로세스 보존을 지원한다. 누락된 모션 데이터 4개 디렉터리를 공유 원본의 읽기 전용 사용을 위한 심링크로 연결했다. config·USAGE·structure에 실행법과 파일 위치를 기록했다.
- 검증: CPU 15 passed. GPU7·32환경×3 reset에서 원본 가중치/RMS·비선행 및 비활성 선행 행동 일치, 센서 미사용, 실제 물체·목표 보존, 스폰/near 조건과 세 후행 종류, 물리 3step finite를 확인했다. GPU7 VNC 120step에서 선행은 118step에 전환했고 네 사람의 종료는 없었다. GPU5 VNC 3step과 GPU6 직접 CUDA headless 3step도 exit0이었다. GPU5/7 화면 저장과 GPU6의 기존 MPS 기록 보존 경로를 확인했다. 정식 진입·운반 성공률은 측정하지 않았다. [GPU 검증](output_etc/check/warehouse_waypoint/geometry/verification.json)·[VNC 결과](output_etc/check/warehouse_waypoint/render/results.json)·[로그](output_etc/check/warehouse_waypoint/implementation/).

### GPU6 일반 hidebox 중단·AT 높이 수정본 새 백그라운드 학습

- 사용자 요청으로 GPU6의 dev 일반 hidebox 서비스 `tokenhsi-two-hidebox-direct-resume-20261116-234856.service`·learner3612342를 중지했다. 실제 실행은 직접 CUDA였으므로 MPS 제어 명령은 보내지 않았다. 6500epoch 저장본의 SHA를 보존했고 서비스 inactive/dead·대상 PID 부재를 확인했다. GPU6 `2only` 운반 집중3568839, GPU5 Bridge858119, GPU7 기존 doorprogress824377과 GPU5/7 MPS server의 시작 tick은 보존했다.
- 최신 AT 높이 마스크·좌표·공유 상자 수정본을 원래 `Stage2Distill900V4_00002000.pth`에서 새 전이했다. 서비스 `tokenhsi-warehouse-doorprogress-gatefix-gpu6-20261117-201917.service`·learner1717136, run `Stage2Sensor900WarehouseFixedRolesV4BeforePairsDoorProgress_17-20-20-21`, GPU6 직접 CUDA·4명/1024환경/8토큰/900step/seed42·Default/closed·BEFORE 2팀20%·캐시 준비32회다. GPU6의 오래된 MPS 기록은 그대로 두고 공통 launch.sh·빈 MPS pipe·미생성 전용 MPS root를 사용한다. CHECK/MAX_ITERATIONS/resume은 해제했다. config 공통 안내와 USAGE의 실제 실행·로그·종료 경로를 갱신했다.
- 시작 검증: 프레임0부터 4회 업데이트·scalar772종 finite, 실제 argv·저장 config·수정 코드 SHA 일치를 확인했다. 캐시 준비55.7초·적중률56.4%, 초반4회 fps total 평균8721이었다(GPU6의 운반 집중 학습과 공유). [실행 정보](output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress/launch_fresh_gatefix_gpu6_20261117-201917/launch.json)·[중단 확인](output_etc/check/doorprogress_gpu6_launch_20261117-201917/stop.json)·[시작 검증](output_etc/check/doorprogress_gpu6_launch_20261117-201917/startup_verification.json).

### 센서 V4 AT 높이 완화 마스크 갱신 누락 수정

- 센서 과제 샘플러가 기본 Stage2와 같이 `relaxed_at_owners`를 현재 `_members == 1`로 갱신하도록 두 줄을 추가했다. 일반 PUSH2+BEFORE와 BEFORE 2팀에서 선행 AT의 후행 해제용 높이 허용오차 20cm가 적용되며, 정밀 보상은 기존 1mm 조건을 유지한다. config·좌표·캐시 정책은 변경하지 않았다.
- 관련 CPU 59 passed. 실제 샘플러와 runtime을 연결한 신규 회귀 12건 중 V4 6건은 수정 전에 실패하고 수정 후 통과했으며, 비 V4 6건도 유지됐다. SIT/CLIMB/ON_TOP·비연속 부분 갱신·20cm 경계 밖 실패·정밀 보상 분리를 검사한다. 구조 문서에 검사 위치를 추가했다.
- GPU7·4명·1024환경·Default·기본 32회 캐시 준비에서 마스크 2472개가 기대값과 일치했다. 실제 그래프에 5cm 높이 오차를 준 대조의 후행 게이트 1236개가 모두 열렸고 정밀 AT 보상은 포화되지 않았다. 캐시 212개 전량 적중의 상태/AMP 복원 오차 0, 비reset 상태 보존·부분 reset 뒤 마스크/문 통과 래치·3step finite를 확인했다. [검증 결과](output_etc/check/warehouse_at_gate_fix_20261117/verification.json).
- 기존 본학습 PID824377은 유지했다. 실행 중 프로세스에는 소스 수정이 자동 반영되지 않으므로 새 실행 또는 재개부터 적용된다.

### doorprogress 수정본을 원래 전이 모델에서 새 백그라운드 학습

- 사용자 요청으로 기존 learner PID 59926을 GPU7 MPS `terminate_client 112102 59926` 성공 응답 0 뒤 중단했다. MPS server·다른 학습과 기존 3500epoch 저장본은 보존했다. 사용자가 원래 전이 모델에서 새 학습을 선택해 `Stage2Distill900V4_00002000.pth`에서 프레임 0으로 시작했다.
- `tokenhsi-warehouse-doorprogress-fresh-targetfix-20261117-131057.service`(app.slice·Restart=no)를 실행했다. learner PID824377, GPU7/MPS·4명·1024환경·8토큰·900step·seed42·headless, RSI 없는 Default/closed·BEFORE 2팀 20%·캐시 준비 32회다. 서비스 conda 초기화의 미정의 `LD_LIBRARY_PATH`를 시작 스크립트에서 빈 값으로 정의해 보완했고, 사용자 `Linger=yes`로 터미널/로그아웃과 독립적으로 유지한다.
- 시작 검증: 프레임0부터 13회 학습 기록·scalar 772개 모두 finite, 실제 argv의 전이 체크포인트/무resume·환경 수·GPU와 저장 config·수정 코드 hash 일치를 확인했다. 초기 캐시 준비 25.0초, 최근 5회 `fps total` 평균 15283이었다(시작 구간 측정). config의 현재 학습 안내와 USAGE 실행/로그 경로를 갱신했다. [실행 정보](output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress/launch_fresh_targetfix_20261117-131057/launch.json)·[시작 검증](output_etc/check/doorprogress_relaunch_20261117/startup_verification.json)·[기존 중단 확인](output_etc/check/doorprogress_relaunch_20261117/stop.json).

### BEFORE Target 좌표·Default 공유 상자 reset·doorprogress 평가 경로 수정

- 사용자 승인으로 Default의 BEFORE 2팀 apron 배치 뒤 후행의 비활성 Target을 자기 초기 root 위치로 교정했다. 활성 AT 목표·사람/상자·RSI·보상·동결·학습 비율은 유지하며 Random/Start/Hybrid에는 이 중립 Target 치환을 수행하지 않는다. 캐시는 교정 후 상태를 저장한다. Stage2 Default 초기화에도 기존 loco와 같은 공유 상자 소유자 필터를 적용해 후행이 선행의 payload/공유 받침을 재샘플링하지 않게 하고, 전체 Default 목록은 AMP 초기화 전에 복원한다. doorprogress 평가 driver의 누락된 5개 실험 분기도 복구했다. 명세·운영·검사 위치를 갱신했다.
- 검증: 관련 CPU 70 passed. 신규 좌표/평가 회귀 중 5개는 수정 전 코드에서 실패하고 수정 후 통과했으며, 기존 Random 입력 보존 대조 3개도 통과했다. GPU7·4명·1024환경·8토큰에서 기본 32회 캐시 준비를 포함한 reset 6회(6144환경, BEFORE 변형 1288개)를 검증했다. 중립 Target·활성 AT/apron·공유 source 거리·역할·물리 배치를 확인했고 관측/보상은 finite였다. 별도 1024환경에서 212개 apron의 전량 캐시 적중 복원·AMP 이력·비reset 상태 보존도 통과했다. 큰 월드 좌표의 GPU 업로드 반올림(최대 약 0.5mm)을 고려해 live Target 비교 허용오차는 1mm다.
- 실제 `*_eval.sh`로 3000 체크포인트를 BEFORE 2팀만 64환경·seed0·900step 평가해 exit0이었다. 선행 AT 83/128(64.8%), 후행 SIT 3/37(8.1%)·CLIMB 2/47(4.3%)·ON_TOP 4/44(9.1%), BEFORE 두 명 모두 ever 9/128(7.0%), 낙상 26/256(10.2%)이었다. 수정 전의 전원 초반 낙상은 재현되지 않았으나 후행 과제 성능 자체를 해결한 것은 아니다. 기존 본학습 PID 59926은 유지했고 실행 중 프로세스에는 소스 변경이 자동 반영되지 않는다. [검증 보고서](output_etc/check/warehouse_before_target_fix_20261117/report.md)·[결과](output_etc/check/warehouse_before_target_fix_20261117/verification.json).

### BEFORE 2팀 전용 viewer 선택·Default 중립 Target 좌표 오류 진단

- 사용자 요청으로 평가 전용 `WAREHOUSE_VARIANT=mixed|push_before|before_pairs`를 추가했다. test/VNC에서 BEFORE 2팀만 또는 기본 2 PUSH + BEFORE 1쌍만 선택하며, 학습·지원하지 않는 config에서는 거부한다. 기본 학습 비율은 그대로다. 관련 CPU 7 passed, GPU7·64환경 선택과 렌더를 확인했고 config/USAGE 옵션을 기록했다.
- 3000 체크포인트의 BEFORE 2팀 초기 종료 원인은 리셋 뒤 후행의 비활성 Target 좌표가 489–1693m(중앙값 1130.55m)까지 벗어나 GTA 입력에 들어가는 것이었다. 사람·상자·활성 AT 목표·Default 자세·RSI 비활성화를 유지한 64환경×4명×120step 대조에서 해당 좌표만 교정하자 종료 254/256명(terrain 낙상 252)→0/256명이었다. actor encoder/coordination 동결 범위는 정상이며 action head/센서 파라미터는 갱신됐다. 초기 자세만 원인이라는 가설은 동일 배치 대조로 배제했다. 전체 에피소드 성공률을 재측정한 결과는 아니다.
- 원본/교정 진단의 실제 6초·0.5배속 영상을 렌더했다. 교정 영상의 env0 네 사람은 180step 동안 종료되지 않고 상자를 집는 동작이 나타났다. 첫 영상의 바닥 소실처럼 보인 현상은 그래픽 동기화·조명 보완으로 바로잡았다. 좌표 교정은 진단 프로세스에만 적용했고 실제 reset 코드의 최소 수정안은 미적용 상태로 남겼다. 기존 본학습은 유지했다. [분석·대조 실험](output_etc/check/doorprogress_beforepairs_cause_20261117/report.md)·[비교 영상](output_etc/check/doorprogress_beforepairs_cause_20261117/before_after.mp4)·[수정안](output_etc/check/doorprogress_beforepairs_cause_20261117/proposed_target_fix.diff).

### BEFORE 배치 후보 일괄 검사·초기 캐시 준비·constant LR 동기화 제거

- 사용자 승인으로 apron의 기존 16개 강체 배치 후보를 한 번에 검사하고, 어느 회전에서도 들어갈 수 없는 그룹은 충돌 검사 전에 탈락시킨다. Default 초기화 학습은 첫 rollout 전에 Warehouse 환경을 BEFORE 2팀으로 32회 미리 배치해 캐시를 채운다(`WAREHOUSE_BEFORE_CACHE_WARMUP=0`으로 생략). 준비 후 변형 비율·샘플링/에피소드 계수·난수 상태를 복원한다. 기존 여유/유효성 조건·BEFORE 20%·캐시 25% 갱신·보상·PPO 설정은 유지한다. constant LR에서는 사용하지 않는 KL `.item()`·LR 재설정만 생략한다. 운영 옵션·명세·구조를 갱신했다.
- 검증: 관련 CPU 23 passed. GPU7/MPS·4명·1024환경·8토큰·seed42로 수정 전/후 각 6회 CHECK 상당의 짧은 학습이 exit0이었다. 두 실행 모두 초기 전체 환경의 기하/BEFORE 유효성·기본 자세/AMP 이력·모션 RSI 미호출을 확인했고 scalar 772개가 모두 finite였다. `fps total` 평균은 9717.8→13166.5(+35.5%), 첫 회 제외 8792.5→12500.9였다. rollout 평균 10.55→6.68초, update 3.44→3.41초, BEFORE 캐시 적중률 1.6→60.7%였다. 초기 캐시 준비는 54.94초로 위 FPS에서 제외되며, 1.5만~2만 FPS에는 도달하지 못했다. 초반 6회 비교로 장기 평균을 보장하지 않는다. [측정](output_etc/check/doorprogress_speedup_20261117/comparison.json)·[diff](output_etc/check/doorprogress_speedup_20261117/changes.diff). 기존 본학습은 작업 시작 시 이미 종료된 상태였고 새 본학습은 시작하지 않았다.

### RSI 없는 doorprogress 본학습 FPS 병목 재확인

- 사용자 요청으로 본학습(PID 6468)을 유지하며 로그와 별도 reset 진단을 확인했다. 41회 기록 중 최근 5회 평균은 10267 FPS, rollout 9.33초/update 3.51초이고, BEFORE 캐시는 245요청 중 34적중(13.9%)이었다. MPS thread 제한은 100%다.
- 별도 GPU7·1024환경에서 캐시 미적중 경로의 32×16환경 reset을 계측했다. 512개 중 apron 89개가 배치에 3.36초, 일반 Warehouse 423개가 0.54초를 썼다. 그룹 배치 426회·내부 accept 3982회로 재시도가 핵심이었다. 일반 Stage2 PUSH/distractor/검사 결과를 버리는 계산도 확인했지만, 이를 생략한 진단은 난수 배치와 동시 GPU 사용 조건이 달라 전체 속도 개선 근거로 삼지 않았다. 추가 후보는 constant LR에서도 48 PPO minibatch마다 수행하는 CUDA KL `.item()` 동기화와 동결 actor 특징의 반복 계산이다. 학습 코드는 수정하지 않았다. [진단](output_etc/check/doorprogress_bottleneck_20261117/diagnosis.json).

### doorprogress reset RSI 비활성화

- 사용자 요청으로 doorprogress 환경 YAML을 `stateInit: Default`, 문 시작 확률 `[1.0, 0.0, 0.0]`으로 바꿨다. 일반 2 PUSH + BEFORE 장면은 항상 닫힘이며 사람 자세·관절/속도도 기본값으로 시작한다. BEFORE 2팀 20%·배치 캐시·문 progress는 유지한다. 캐시는 새 프로세스에서 기본 자세 배치로 채워진다. AMP 학습용 참조 모션은 유지한다.
- Default 초기화의 배치 재시도 후에도 새로 reset한 모든 사람의 AMP 이력이 초기화되도록 `_reset_default_slots`를 복원했다. 관련 CPU 6 passed. GPU7·4명·1024환경에서 reset 2회 중 모션 RSI 호출 0, 일반 장면 닫힘 100%, 기본 관절/속도·AMP 이력 일치를 확인했고 물리 1step의 관측·보상도 finite였다. [검증](output_etc/check/doorprogress_no_rsi_20261117/live/reset_check.json). 확인 중 기존 본학습 프로세스가 종료된 상태였으며 새 본학습은 시작하지 않았다. 다음에 기존 doorprogress 학습 명령을 실행하면 이 설정이 적용된다.

### doorprogress 학습 reset 스크린샷 10장·Bridge 후속 보상 메모

- 사용자 요청으로 doorprogress 학습 모드(4명·1024환경·8토큰·seed42)의 별도 reset에서 스크린샷 10장을 GPU7로 캡처했다. 전체 분포는 기본 812/BEFORE 2팀 212개, 기본의 문 시작은 닫힘 438/부분 177/열림 197개다. 확인용 이미지는 닫힘4·부분2·열림2·BEFORE 2팀2를 골랐고, 역할·과제·선후행 연결을 함께 표시했다. 렌더 동기화를 위한 물리 1step만 수행했으며 정책 행동·학습은 실행하지 않았다. 기존 본학습(PID 4160482)은 유지했다. 사용자 요청에 따라 원본 이미지와 [이미지 모음](output_etc/check/doorprogress_reset_screenshots_20261117/index.html)을 제공한다.
- 실제 다리의 입구·출구/중심선, 운반 상태, 다리 영역·발 지지로 접근/횡단/완주를 보상하는 후속 아이디어를 `markdowns/reference/stage2_sensor_900.md` 마지막에 미구현 계획으로 기록했다.

### BEFORE 선행의 문 입구 root XY progress 추가

- 사용자 요청으로 `stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress`(환경/학습 YAML·5개 wrapper·등록)를 추가했다. beforepairs에 `warehouse.leader_door_progress: true`만 더한 별도 보상 비교 실험이다. 기본 2 PUSH + BEFORE 1쌍의 AT 선행에게만 actor root XY와 문 입구 중앙 사이의 `1/(1+d/sigma)`를 추가한다. 기존 sigma 1.0·progress 가중치 0.2를 재사용한다. 문 폭 안에서 앞벽 중앙선을 방 안쪽으로 넘으면 에피소드 동안 raw 1.0으로 유지하고 reset 시 해당 환경의 기록만 지운다. PUSH·후행·BEFORE 2팀에는 0이며 기존 성공/종료 조건은 같다. `reward_terms/door_progress`에 별도 기록한다. config 명령·구조·명세를 갱신했다.
- 검증: 신규 대상/XY/root/회전/통과 후 포화 검사와 관련 기존 보상 검사 19 passed. GPU7/MPS·4명·1024환경·8토큰에서 CHECK(`MAX_ITERATIONS=1`) exit0, scalar 771개 모두 finite였다. 추가 항 평균은 0.0114/0.0131이고 보상 항 합계와 total 차이는 9e-9 이하다. 새 wrapper 5개와 launch의 shell 문법도 확인했다. 본학습은 시작하지 않았다. [검증](output_etc/check/warehouse_door_progress_20261117/check.json).

### BEFORE apron 검증 배치 캐시

- 사용자 승인으로 학습 전용 `warehouse_before_cache.py`를 추가했다. 같은 두 팀의 후행 과제·near/far 조합, 환경 기하·상자 크기에 대해 검증된 시작 상태를 재사용하고, 역할·물체 매핑·그래프·자세/관절/속도·목표·AMP 이력까지 복원한다. 환경당 32조건·조건당 2배치로 제한하며, 적중 가능한 reset의 25%는 새로 생성해 갱신한다. BEFORE 변형이 있는 학습에서 기본 활성이고 `WAREHOUSE_BEFORE_CACHE=0`으로 끈다. 평가에서는 꺼진다. YAML·모델·보상 계약은 그대로이며 운영·명세·구조·config 링크를 갱신했다.
- 검증: 관련 CPU 134 passed. GPU7/MPS·4명·1024환경에서 동일 조건 apron 212환경 reset 평균은 생성 0.859초/캐시 0.0195초였다. 상태·AMP·그래프 복원, 비reset 환경 보존, 충돌/BEFORE 검사 및 30step finite를 확인했다. 정상 확인 학습은 사용자 요청으로 10회 후 중지했다. `fps total` 평균 12235.98, 기록된 scalar는 모두 finite다. 초기 캐시 적중은 6/1090(0.55%)로, 위 reset 가속률을 전체 학습 FPS 개선율로 해석할 수 없다. GPU7 client 종료를 확인했다. [검증](output_etc/check/beforepairs_layout_cache_20261117/live/cache_check.json)·[학습](output_etc/check/beforepairs_layout_cache_20261117/training_check.json).

### beforepairs RSI 제외 FPS·reset 병목 측정

- 사용자 요청으로 GPU7/MPS·4명·1024환경·8토큰에서 문 닫힘 100% + `stateInit: Default`인 임시 설정을 측정했다. BEFORE 2팀 20%와 학습 조건은 유지했다. 초반 20회 `fps total` 산술평균은 9455.5(7506.8–13327.5)다. 기존 실행 config·학습 코드는 수정하지 않았다.
- 원본/RSI 제외를 각각 6iteration 계측했다. 초기 1회와 cProfile 적용 6회를 제외한 2–5회 평균 reset은 4.68/7.38초, 나머지 rollout은 3.07/2.97초, update는 3.48/3.48초다. 물리 배치 재시도는 78.75/263.75회로 증가했고, BEFORE apron 그룹 배치 반복이 주요 비용이었다. 모두 exit0이고 GPU7 client는 종료됐다. [20개 값](output_etc/check/beforepairs_no_rsi_fps_20261117_b6p4sx/fps20.csv)·[계측](output_etc/check/beforepairs_no_rsi_fps_20261117_b6p4sx/profile_summary.json)·[재현 스크립트](output_etc/check/beforepairs_no_rsi_fps_20261117_b6p4sx/run_profile.sh).

### leaderdone 본학습 중지

- 사용자 요청으로 이 repo의 학습을 모두 중지했다. 대상은 `tokenhsi-warehouse-leaderdone-20261116-204212.service`(PID 3030128) 하나였고, 마지막 저장은 `Stage2Sensor900WarehouseFixedRolesV4LeaderDoneWallBox_00001500.pth`다.
- 종료는 GPU7 MPS `terminate_client 112102 3030128`(반환 0) 후 서비스 stop 순서로 했다. MPS server 112102는 유지됐고 client는 남지 않았다. 다른 repo 서비스는 건드리지 않았다.

## 2026-11-16

### beforepairs 2500epoch 재개

- 사용자 요청으로 leaderdone 교체 때 중지한 beforepairs를 2500epoch에서 이어서 학습한다. 중지 시점은 약 2900 iteration이고, 500epoch마다 저장하므로 그 이후분은 체크포인트가 없다.
- `RESUME_CHECKPOINT=..._00002500.pth`로 `tokenhsi-warehouse-beforepairs-resume-20261116-205455.service`(GPU7/MPS, 4명·1024환경·8토큰)를 시작했다. config는 바꾸지 않았고, 이번에 추가한 Warehouse 코드는 키가 없으므로 실행되지 않는다.
- 확인: checkpoint를 로드한 뒤 epoch 2501부터 기록했고, traceback 없이 진행 중이다. GPU7 MPS client는 이 학습, leaderdone 학습, 다른 repo 학습 하나다.
- 이후 사용자 요청으로 중지했다. 재개 run `..._16-20-55-06`의 마지막 저장은 `_00003000.pth`다. 종료는 GPU7 MPS `terminate_client 112102 3078787`(반환 0) 후 서비스 stop 순서로 했고, MPS server와 leaderdone client(3030128)는 유지됐다.

### Warehouse 선행 완료 시작 + 벽 박스 CPA 실험 추가·본학습 교체

- 배경: beforepairs는 장애물 없는 apron에서 "상자로 직진"을 강화했다. 그 결과 본 시나리오의 선행 방 진입이 0.454에서 0.055로, AT 성공이 0.295에서 0.002로 무너졌다(2500 대 2500). 후행이 망가진 원인은 선행 완료 뒤 행동을 시작하는 경험이 없는 데 있다고 보고, 본 시나리오 안에서 그 상태를 직접 공급한다.
- 사용자 요청으로 `stage2_sensor_900_warehouse_fixedroles_v4_leaderdone_wallbox`(YAML·5개 wrapper·등록)를 추가했다. fixedroles_v4에 `warehouse.cooperative.leader_done_probability: 0.3`, `warehouse.wall_cpa: box`만 더했다.
  - 학습 reset의 30%는 문이 열려 있고, 선행 상자가 apron의 AT 목표 위에 있으며, 선행이 그 옆에 선 상태에서 시작한다. 후행·PUSH·다른 물체는 기존 배치 그대로다.
  - 사람 자세는 기존 loco 그대로라 RSI 캐시는 필요 없다. 평가와 viewer에서는 꺼진다.
  - 새 모듈 `warehouse_done.py`를 추가했다. 기존 파일에는 확률이 0이 아닐 때만 실행되는 분기만 넣었다. 배치 함수·보상·beforepairs 코드는 바꾸지 않았다.
- 검증:
  - CPU: 신규 9개를 포함해 Warehouse 계열 124 passed. sensor/stage2 계열은 파일별로 실행해 모두 통과했고, 실패 2건은 checkpoint 파일이 없는 기존 요인이다. 전체를 한 프로세스로 돌리면 isaacgym import 순서 때문에 수집 오류가 난다.
  - GPU7 MPS 1024환경 reset 6회: 완료 시작 30.7%, 문 상자와 선행 상자 위치 오차 0.5mm, 선행과 상자 면 거리 0.45–1.0m, 앞벽 바깥 면에서 0.40m 이상, 사람 간 0.85m 이상, 재시도 0.
  - 1 step 후 선행 성공과 후행 대기 해제가 모두 100%였다(본 시나리오 후행 해제는 0%). 선행 보상은 step 1/2/10/30에서 1.06/1.10/1.18/1.07로, open 시작 PUSH 담당자(1.08/1.12/1.19/1.02)와 같은 성공 유지 보상이었다.
  - zero-action 30step 종료율은 완료 시작 후행 0.31, 본 시나리오 작업자 0.25였다.
  - CHECK 3iter exit 0(scalar 765개 모두 유한값, 새 진단 기록 확인). headless test exit 0. 렌더 이미지를 확인했다. 산출물: `output_etc/check/warehouse_leaderdone_20261116/`.
- 운영:
  - beforepairs 서비스(2500epoch)는 GPU7 MPS `terminate_client 112102 1720488`(반환 0)로 분리한 뒤 중지했다.
  - `tokenhsi-warehouse-leaderdone-20261116-204212.service`(GPU7/MPS, 4명·1024환경·8토큰)를 시작했다.
  - 앞 항목의 `wall_cpa: box` 옵션을 이 실험에서 처음 켰다.

### Warehouse 벽 CPA 박스 옵션 추가

- 배경: 벽 CPA 선분이 벽 안쪽 면에만 있어 방 밖에서는 벽이 0.4m 뒤에 있는 것처럼 계산됐다. 바깥 모서리와 문틀 옆면도 빠져 있었다. 실제 함수로 계산하면 1m/s 정면 접근에서 방 밖 감점은 방 안의 약 0.5배, 45°는 0.1–0.3배였다. fixedroles/beforepairs 선행은 방 밖에서 앞벽 쪽으로 비스듬히 다가가므로, 감점이 가장 약한 경우에 해당한다.
- 변경: 사용자 요청으로 `sensorScene.warehouse.wall_cpa: box` 옵션을 추가했다. 물리 벽 static 박스 5개에 기존 `compute_box_cpa_collision_penalty`를 정지 상자로 적용한다(`warehouse_sensor.wall_box_cpa`). 값 검증은 `utils/warehouse_sensor.validate_config`에서 한다. 키가 없는 기존 config는 이전 선분 계산과 같다. 아직 이 옵션을 켠 실험 config는 없다.
- 검증: 신규 `tests/test_warehouse_wall_box_cpa.py` 6 passed. 방 안 무작위 1024명은 선분 결과와 같았다. 방 밖 정면 접근과 문틀 옆 접근은 방 안 정면 접근과 같은 값이었다. 잘못된 값은 거부한다. 기존 Warehouse 계열 테스트 115 passed(저장소 루트에서 실행).
- 분석: beforepairs 2500 대 fixedroles 2500 정식 평가 결과는 `output_etc/check/beforepairs_eval2500_20261116/report.md`에 있다. beforepairs에서는 선행 AT가 0.002, 방 진입이 0.055로 떨어졌다. fixedroles는 AT 0.295, 진입 0.454다.

### Warehouse BEFORE 2팀 20% 변형 추가·본학습 교체

- 배경: fixedroles_v4 2000epoch 정식 평가(512환경×3 seed)에서 PUSH 0.96, 선행 AT 0.22, 후행 SIT/CLIMB/ON_TOP 약 0이었다. 선행의 약 절반이 방에 들어가지 못했고, 후행은 대기 반경 경계에서 멈춘 뒤 선행이 성공해도 움직이지 않았다. [분석](output_etc/check/fixedroles_v4_eval2000_20261116/report.md).
- 사용자 요청으로 `stage2_sensor_900_warehouse_fixedroles_v4_beforepairs`(YAML·5개 wrapper·등록)를 추가했다. fixedroles_v4에 `warehouse.cooperative.before_pairs_probability: 0.2`만 더했다. Warehouse reset의 20%는 PUSH 없는 BEFORE 2팀(AT 선행 + SIT/CLIMB/ON_TOP 후행)이다. 일반 Stage2 BEFORE 기본 배치를 그룹 단위로 방 앞 apron에 놓고(`_place_group`의 `region` 인자), 문 상자는 비활성으로 env 밖에 둔다. 남는 상자는 apron 밖으로 치운다. 구현은 신규 `warehouse_before.py`에 있고, 기존 파일에는 키가 0이 아닐 때만 실행되는 분기만 추가했다. 변형 env는 기존 문 개방/PUSH 상자 진단에서 제외하고 `warehouse/variant/*`에 변형별 선행·후행 성공을 기록한다. 사람은 기존 Warehouse 작업자처럼 바닥 시작으로 판정한다. [명세](markdowns/reference/stage2_sensor_900.md#warehouse-역할-고정-v4--before-2팀-stage2_sensor_900_warehouse_fixedroles_v4_beforepairs).
- 검증: CPU 신규 4 passed. 기존 fixedroles/coop/room_warehouse 계열 테스트는 모두 통과했다. 전체 실패 18건은 기존 환경 요인(MPS 없는 wrapper 테스트 포함)이다. GPU7/MPS 1024환경 reset 6회에서 변형 19.2%, 후행 SIT/CLIMB/ON_TOP 809/756/789였다. 사람·작업 상자·AT 목표는 모두 apron 위·apron 높이에 있었고, 문 상자는 치워진 상태, 나머지 80%는 PUSH 2 + BEFORE 1이었다. zero-action 30step 종료는 변형 28%·기존 작업자 25%, 벽 접촉 종료는 0.3%·3.8%다. `CHECK=1` 3iter는 exit0이고 scalar 765개가 finite, 물리 배치 실패는 0이다. headless test(4환경) exit0, Xvfb 렌더 이미지도 확인했다. [로그·이미지](output_etc/check/warehouse_beforepairs_20261116/).
- 운영: 사용자 결정으로 fixedroles_v4 서비스(2500epoch)를 GPU7 MPS `terminate_client 112102 869038`(반환 0)로 분리한 뒤 중지했다. 다른 repo client 831057과 server는 유지됐다. `tokenhsi-warehouse-beforepairs-20261116-113624.service`(GPU7/MPS·4명/1024환경/8토큰, Distill900V4 2000 새 전이)를 시작했고, 3분 안에 10 iteration이 진행됐으며 traceback은 없었다. [운영](markdowns/USAGE.md#warehouse-역할-고정-v4--before-2팀-본학습). 학습 성능은 아직 측정하지 않았다.
- 지침: 사용자 요청으로 AGENTS.md에 "현재 머신에서는 GPU 6, 7번만 사용"을 추가했다. 이번 중간 분석의 정식 평가는 지침 추가 전에 GPU1에서 돌렸다.

### Warehouse 역할 고정 V4 실험 추가·본학습 교체

- 사용자 요청으로 `stage2_sensor_900_warehouse_fixedroles_v4`(환경/학습 YAML·5개 wrapper)를 추가했다. V4 보상 config 대비 Warehouse 키 6개만 다르다: Warehouse 100%, 공동 PUSH 2명, 방 안쪽 방향, 5m, 방 8×8m, worker pairing 1.0. 역할은 PUSH 2명·AT 선행·SIT/CLIMB/ON_TOP 후행으로 고정된다. 기존 옵션으로 표현되므로 코드는 등록만 바꿨다. 사용자 결정으로 Distill900V4 2000에서 새로 전이하고 시작 분포 50/25/25를 유지했다. [명세](markdowns/reference/stage2_sensor_900.md#warehouse-역할-고정-v4-stage2_sensor_900_warehouse_fixedroles_v4).
- 검증: CPU 5 passed, `test_warehouse_coop` 51 passed. GPU7/MPS 1024환경 CHECK(3iter)는 exit0이고 scalar 757개가 finite다. 분포는 Warehouse 100%, pushers2/in 100%, BEFORE 쌍 1.0/episode, 템플릿 PUSH 0.5·AT 0.25·SIT/CLIMB/ON_TOP 각 0.08, 시작 분포 1559/798/787이다. physical retry와 unreachable은 0이다. [로그](output_etc/check/warehouse_fixedroles_v4_20261116/).
- 운영: 사용자 요청으로 tmux에서 실행 중이던 boxterm 본학습(PID 2446226, 최신 checkpoint 16000epoch)을 GPU7 MPS `terminate_client 112102 2446226`(반환 0)으로 분리해 종료했다. 같은 MPS의 다른 repo client 831057과 server 112102는 유지되었고 학습도 계속 진행되었다. 이후 `tokenhsi-warehouse-fixedroles-v4-20261116-040900.service`(GPU7/MPS·4명/1024환경/8토큰·app.slice·Restart=no)로 본학습을 시작했다. 시작 5분에 26 iteration이 진행되었고 traceback은 없었다. 계정이 `Linger=no`이므로 모든 세션이 끝나면 서비스도 함께 종료될 수 있다. [운영](markdowns/USAGE.md#warehouse-역할-고정-v4-본학습). 학습 성능은 아직 측정하지 않았다.

### boxterm 기반 V4 보상 실험 추가

- 사용자 요청으로 dev checkout의 `stage2_sensor_900_cliff_bridge_four_transfer_v4_reward` 보상 체계를 이식해 `stage2_sensor_900_room_warehouse_wallterm_boxterm_v4_reward`(환경/학습 YAML·train/train_vnc/test/vnc/eval)를 추가했다. boxterm 대비 `stage2.reward_source: v4_teachers`, `before.reward_at_z_tolerance: 0.001`, 대기 감점 1.5m/payload 기준, `stage2JointPush.premature_motion_penalty`와 Run 이름만 다르다. 사용자 결정에 따라 벽/PUSH 상자 CPA·접촉 종료는 유지했다. 코드(`before_reward`·`stage2_reward`·`stage2_wait`·`humanoid_ma_stage2`·`humanoid_ma_coupled_push`·`humanoid_ma_carry`·`edge_context_task`·`coupled_push_contact`의 `near`·신규 `utils/premature_push.py`·`stage2_spec`)는 dev의 V4 분기와 같고, flag가 없는 기존 실험의 동작은 바꾸지 않았다. dev의 외력 컨트롤러 CONTACT 연결은 Bridge 전용이어서 이식하지 않았다. [명세](markdowns/reference/stage2_sensor_900.md#v4-보상-stage2_sensor_900_room_warehouse_wallterm_boxterm_v4_reward).
- 신규 CPU 9개 passed. room_warehouse/wallterm/boxterm 테스트는 통과했다. 전체 CPU 테스트의 실패 16건은 이번 변경과 무관한 기존 환경 요인이다(MPS 데몬 없음, GPU 제한, checkpoint 파일 없음, 기존 stub에 `cfg` 없음). GPU7/MPS 1024환경 CHECK(3iter)는 exit0이고 scalar 757개가 finite다. `premature_motion_penalty`는 −0.002~−0.003, `v4/waiting_armed`는 0.21–0.24다. 4환경 HEADLESS=1 test exit0. 실행 중인 boxterm 본학습(PID 2446226)은 건드리지 않았다. 학습 성능은 측정하지 않았다. [로그](output_etc/check/room_warehouse_v4_reward_20261116/).

## 2026-11-13

### Warehouse PUSH 상자 CPA·접촉 종료 비교 실험 추가

- 사용자 요청으로 `stage2_sensor_900_room_warehouse_wallterm_boxterm`(환경/학습 YAML·train/train_vnc/test/vnc/eval)을 추가했다. wallterm 대비 `push_box_collision_coefficient: 0.7`·`push_box_contact_termination: 0.3`과 Run 이름만 다르다. Warehouse에서 PUSH 담당자가 아닌 사람에게만 PUSH 상자를 장애물로 적용한다. CPA는 상자 표면 최근접점과 상자 속도를 쓰고 사람/벽과 max로 합친다. 종료는 벽과 같은 형상 3cm·접촉력이 0.3초 연속일 때이며 카운터는 따로다. PUSH 상자 접촉 로깅은 키 없는 실험에도 동작한다. 기존 wall 판정 함수 이름을 `body_box_contact`로 바꿨고, `test_room_warehouse`의 가짜 task에 새 속성을 추가했다. [명세](markdowns/reference/stage2_sensor_900.md#warehouse-push-상자-cpa접촉-종료-stage2_sensor_900_room_warehouse_wallterm_boxterm).
- CPU186 passed/3 skipped. GPU7 1024환경 CHECK(3iter) exit0·751 scalar finite다. 비 PUSH 담당자의 PUSH 상자 접촉은 스텝의3.0–4.9%, 상자 종료는0.18–0.37회/warehouse 에피소드이고, warehouse 에피소드 길이는54–63step(wallterm70–73)이다. test(4환경 HEADLESS=1) exit0. 사용자가 실행 중인 wallterm 본학습(PID 2298553)은 건드리지 않았다. [로그](output_etc/check/boxterm_20261113/).

### 벽 접촉 0.3초 종료 비교 실험 추가

- 사용자 승인으로 `stage2_sensor_900_room_warehouse_wallterm`(환경/학습 YAML·train/train_vnc/test/vnc/eval)을 추가했다. room_warehouse 대비 `sensorScene.wall_contact_termination: 0.3`과 Run 이름만 다르다. 벽 CPA는 붙어 있거나 평행하게 미끄러질 때 0이므로, 손·발을 뺀 11개 몸체의 실제 벽 접촉이 9스텝 연속되면 해당 사람을 종료한다. 넘어짐과 같은 씬 reset·정식 평가 무reset 규칙을 따른다. [명세](markdowns/reference/stage2_sensor_900.md#벽-접촉-종료-stage2_sensor_900_room_warehouse_wallterm).
- 처음의 몸체 원점 0.2m 판정은 진단에서 오탐이 많았다. 자기충돌 때문에 허벅지에 전체 스텝의 약33% 동안 접촉력이 있고, room 몸통·전완은 상자/앉기 힘이 벽에서 먼 곳에서 더 컸다. 그래서 MJCF 구·캡슐 형상이 벽 OBB에서 3cm 이내이면서 그 몸체에 접촉력이 있을 때로 바꿨다. 형상이 벽에 닿을 때 접촉력 비율은 허벅지 82–85%, 머리 55–60%다. 감지·로깅(`scene/*/wall_contact_fraction`·`wall_terminations_per_episode`)은 기존 sensor 실험에도 동작하며 종료는 새 키가 있을 때만 한다. 이 과정에서 V4 전이 변경으로 깨진 `test_room_warehouse` 동일성 검사도 전이 경로를 반영하도록 고쳤다.
- CPU174 passed/3 skipped. GPU7 1024환경 CHECK(3iter) wallterm·기존 room_warehouse 모두 exit0, 749 scalar finite다. wallterm 초반 정책은 벽 종료가 room0.92–0.98·warehouse0.74–0.95회/에피소드이고 에피소드 길이는 room206–263·warehouse70–73step이다. 기존은 벽 종료0, 접촉 비율 room4–7%·warehouse14–17%다. wallterm test(4환경 HEADLESS=1)도 exit0. [로그](output_etc/check/wallterm_20261113/). 본학습은 시작하지 않았다.

### 이 서버(hwanhee) 실행 환경 세팅·전이 Distill900V4 2000

- 깨진 데이터 심링크를 `TokenHSI_stage12_teacher_sensor`와 같은 서버 경로로 교체했다: `dataset_{carry,sit,amass_climb,amass_loco}/motions`→`/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/*/motions`, `jointpush_motions`→`/home/hwanhee/ksh/stage1_2_jointpush/...`, `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`→`/home/hwanhee/ksh/stage1_2_base_jointpush/pth/`. `checkpoints/Stage2BeforePush...RelFusionZeroBg_00021000.pth`(stage2_distill teacher)는 원본이 없어 깨진 채 남겼다.
- `runtime_env.sh`·`run-gui.sh` 기본 conda를 `tokenhsi_jhh`로 바꾸고 README/config/USAGE/명세의 현재 설명을 맞췄다. config의 room_warehouse 명령은 `TOKENHSI_GPU=7`로 바꿨다.
- 참조 repo의 `Stage2Distill900V4_00002000.pth`를 `output/stage2/`에 복사(SHA256 `a1be127a…290f055a` 일치)하고 `stage2_sensor_900_room_warehouse`의 `experiment.transfer.checkpoint`를 이 파일로 바꿨다. 다른 sensor config는 Distill1500 경로 그대로이며 이 checkout에는 해당 파일이 없다.
- GPU7(기존 MPS)에서 room_warehouse `CHECK=1` 학습(1024환경, V4 전이 경로 저장·checkpoint 저장, exit0), `HEADLESS=1` test(4환경, 진단 저장, exit0), test VNC(noVNC6081 HTTP200·Vulkan GPU7·모델 로드·진단 저장 후 정상 정리)를 확인했다. train_vnc는 실행하지 않았다. [로그](output_etc/check/setup_20261113/).

### Warehouse 5명 전환·6500epoch 평가 탐색

- 사용자 승인으로 2 PUSH·BEFORE ON_TOP1쌍·독립 AT의 5명/10토큰 평가를 `output_etc/check/warehouse_five_before_ontop_6000_20261113/`에서 진행한다. 실제 5명 관측/행동·41개 물리 상자·strict 복원과 짧은 finite 물리 실행을 확인했다. 원본 정책 평균 행동을 사용하며 본학습 코드/설정·기존 채택 영상은 변경하지 않는다.
- 새6500epoch의 epoch·모델 finite·SHA(`8de7d1869c60cdce678b196518eb0537690ecdccb8abfb750020fe8461ccc1ff`)를 확인하고 탐색을 전환했다. CPU PhysX/GPU 정책 추론의 환경32개 실행에서 GPU allocator peak는243,269,632bytes다. 수행 중10m·초기 접근/운반5m 검사와 완료 후 거리 예외를 유지한다.
- 평가 reset의 방향·간격·반출 경로·학습 크기 범위 내 상자 형태를 조건별로 기록한다. 과제 성공 후보도 사람/벽/상자 기하 검사에서 접촉하면 탈락시킨다. 아직 검증 완료 영상은0개이며, 3개 확보는 진행 중이다. 탐색·선별·Blender 저장/렌더 서비스는 대화 종료 후에도 실행한다.

### Warehouse 7명 중간 후보·6000epoch 재탐색

- 7명/14토큰·2 PUSH·BEFORE ON_TOP2쌍·독립 AT의 CPU 평가 helper와 중간 후보를 `output_etc/check/warehouse_seven_before_ontop_20261113/`에 저장했다. 센서 인원 검사만 평가 wrapper에서 확장했고 실제 7명 tensors·strict checkpoint 복원·12물리step finite를 확인했다. 본학습 코드/설정 변경은 없다.
- [중간 후보](output_etc/check/warehouse_seven_before_ontop_20261113/previews/candidate_01/index.html)는 6명의 짧은 native 성공 이력이 있지만 사람/벽 충돌·후행 대기가 있어 최종 선별 탈락이다. 후행 초기 자기 상자 거리가 약7.2m로 기본 sampler의 접근 5m 범위를 벗어난 점을 확인해 후속 탐색은 native sampler와 접근/운반 5m 검사를 사용한다. 완주 장면 3개 확보는 미완료다.
- 미리보기 720p/10fps/50초의 500frame 전체 디코딩, Blender 재개방·전체1500 keyframe와12개 평가 frame의 7명/6상자 위치 오차1.07e-6m 이하를 확인했다. 원본 관절/장면은30fps이며 자세/행동 보정은 없다. 기존 채택 영상 등166개 파일과 본학습 source20개 SHA는 보존했다.
- 사용자 승인으로 완료 후의 거리 초과·선행 이탈은 허용하고, reset/수행 중에는10m 제한을 유지한다. 완료는 native 성공과 저속·수평·실제 바닥/받침 높이의6연속frame을 함께 확인한다(합성6조건 검사 통과). 새6000epoch PTH(SHA `beb3a250d199122092ef4439a2b10d7621e101778e594f2ce68bdd2c6dbe5220`) finite를 확인하고 CPU 기본 배치 재탐색을 시작했다. 인원5명 전환 여부는 사용자 확인 대기다.

### Warehouse deterministic·선후행 외부 출발 장면5편

- 사용자 승인4500epoch를 고정해4,000episode를 탐색하고2 PUSH·실내 상자 반출 선행·BEFORE CLIMB 후행의5장면을 추가 저장했다. 기존5편은86개 파일 SHA exact로 보존했다. 평가 reset만 조절했으며 선택은 완전 닫힘2개·부분 열림 RSI3개, 선후행 모두 외부 출발이다. 정책 평균 action을 사용하고 자세/관절/행동을 보정하지 않았다.
- 후행2.13–4.02m 보행→2.50–5.63초 대기→CLIMB를 확인했다. 기록30Hz+보간 중간 자세에서 사람 간 최소0.137m·선후행/입구 상자0.057m·벽0.78mm의 비겹침, 운반 상자와 PUSH 담당자/선행 완료 전 후행/입구 상자의 분리를 검사했다. 이는 선택 구간의 기하 검사이며 물리 substep 전수 검증이나 학습 분포 성공률이 아니다.
- 네 명 모두 native 성공 이력이 있으나 끝프레임의 원래 CLIMB done은5개 모두false다. 마지막 발목 기준 높이가7cm보다1.3–2.6mm 크므로, 후행 완료 유지에는 발 지지력·발판 내부 양발·저속/수평 상자·30연속 프레임 지지를 별도 확인했다. 원래 플래그를 보존하고 추가 `physical_climb_success`와 판정 차이를 보고서/모아보기에 명시했다.
- 관절32개/속도·몸체/상자/행동·역할/목표/기하·전체 원본 episode·Blender5개·1080p/30fps MP4를 저장했다. 영상 전체 디코딩, Blender 재개방2,105프레임의 몸체/상자 위치 오차5.33e-7m 이하, finite·중복·원본 모션 exact·화면 여백을 확인했다. [모아보기](output_etc/check/warehouse_outside_det4500_20261113/index.html), [조건·재사용](output_etc/check/warehouse_outside_det4500_20261113/report.md), [전체 묶음](output_etc/check/warehouse_outside_det4500_20261113/warehouse_outside_4500_bundle.zip).
- 자체 GPU 탐색은 VRAM 여유 감소 시 정상 종료했고 최종 선별은 CPU PhysX/추론, 렌더는 한 번에1개(약1.3GiB)로 실행 후 종료했다. 학습3132517/1783512/3343348·MPS374777/374795의 동일 신원, 본학습 source20개 및4500 PTH SHA를 보존했다.

### Warehouse 2 PUSH·실내 상자 반출·후행 CLIMB 영상5편

- 사용자 요청으로 RoomWarehouse의 검색 시작 시 최신4000epoch를 고정하고 닫힌 입구·2명 공동 PUSH·실내 상자 반출 선행1명·BEFORE CLIMB 후행1명을5504episode 탐색했다. 평가 reset에서만 선행/상자 실내 배치·후행 목표 방향·간격·성공 배치 주변 반전/미세 이동을 적용했다. 선택5편은 모두 안쪽 PUSH이며 실제 정책 관절/행동을 보정하지 않았다.
- 선택 구간10.87–11.47초에서 후행1.41–2.46m 보행→선행 완료까지1.40–2.67초 대기→CLIMB, 마지막 네 명 동시 성공을 확인했다. 기록30Hz와 보간 중간 자세의 기하 검사에서 사람 간 최소0.185m·선후행/벽0.118m·선후행/입구 상자0.102m를 확보했다. 이는 선별 장면의 검사이며 전체 학습 분포 성공률은 아니다.
- 32관절/속도·몸체/상자 상태·행동·역할/목표/기하·전체 원본 episode·재사용 `.blend`5개·1080p/15fps MP4를 저장했다. 영상 전체 디코딩, Blender 독립 재개방1678프레임의 몸체/상자 위치 오차1.39e-7m 이하, ZIP CRC·링크·중복/finite를 확인했다. [모아보기](output_etc/check/warehouse_push2_carry_climb_20261113/index.html), [조건·재사용](output_etc/check/warehouse_push2_carry_climb_20261113/report.md), [전체 묶음](output_etc/check/warehouse_push2_carry_climb_20261113/warehouse_carry_climb_bundle.zip).
- 검색/렌더는 정상 종료했고 본학습3132517·다른 학습1783512/3343348·MPS374777/374795의 PID/start time, 본학습 source20개 및 원본 PTH SHA를 보존했다. 학습/공유 MPS에 종료 명령을 보내지 않았다.

## 2026-11-12

### Room·Warehouse 전용 fresh 학습·벽 CPA0.7·MPS 안전 교체

- 사용자 승인으로 `stage2_sensor_900_room_warehouse` 환경/학습 YAML·train/train_vnc/test/vnc/eval을 추가했다. Cliff/Bridge0·Room/Warehouse50:50, Distill1500 fresh 전이와 벽 CPA0.7만 적용한다. 사람 CPA0.5·거리0.7m·기존 max 합산·BEFORE 대기 보상·freeze·PPO·역할/크기/RSI를 유지하고 기존 실험은 새 키가 없으면 기존 벽 계산을 유지한다. [명세](markdowns/reference/stage2_sensor_900.md#roomwarehouse-전용-stage2_sensor_900_room_warehouse).
- CPU95 passed;1024환경4096reset에서 Room2046/Warehouse2050·Cliff/Bridge0, 실제 벽/센서/CPA 기하·공유 상자·NumPy 센서·부분 reset·16물리step을 확인했다. CHECK 종료4epoch/frame524288:동결131텐서/입력 RMS exact, 센서·action head·critic·AMP 갱신, optimizer162개/step192·741 scalar finite. [검증](output_etc/check/room_warehouse_20261112/report.md).
- 기존 CoopCliffRoom Python2797534의 신원·MPS mapping을 확인하고 `terminate_client 374795 2797534` 성공0 이후 해당 서비스만 종료했다. inactive/dead/MainPID0·PID/GPU 부재·원본9000epoch SHA 보존·다른 학습1783512/3109910과 MPS374777/374795 생존을 확인했다.
- 사용자 systemd `tokenhsi-room-warehouse-20261112-163455.service`·GPU0/MPS·4명/1024환경/8토큰/900step·seed42·Linger=yes·Restart=no로 새 본학습을 시작했다. Python3132517·epoch3·741 scalar finite·실제 Room1594/Warehouse1581/나머지0 샘플·fresh 전이/검사 cap 해제·source20개 SHA 일치·기존 학습/MPS 동일 PID/start time을 확인했다. [실행 정보](output/stage2_sensor_900_room_warehouse/launch_20261112-163455/launch.json), [시작 확인](output_etc/check/room_warehouse_20261112/startup_verification.json), [운영·안전 종료](markdowns/USAGE.md#room-warehouse-전용-본학습). 장기 성능은 미검증이다.

### Warehouse 후행 비접촉 보행 재선별·영상5편

- 사용자 지적으로 이전 추가 영상04의 이동에 입구 상자 밀림이 포함됐음을 정정하고 기존 모아보기·보고서에서 자발 보행 후보 제외를 표시했다. 몸체별 MJCF 포괄 구와 실제 회전 상자 OBB로 episode 시작부터 보행 구간 끝까지 여유 하한0.35m 초과를 요구하며, 다른 사람/물체/벽과의 분리·양발 교대·발 높이·이격을 함께 검사했다.
- 같은5500epoch·학습 변경 없이 닫힘 시작576episode(평가64환경, PUSH1/2 균등, BEFORE pairing1.0)를 추가 탐색해8후보를 확보했다. 서로 다른5편을 선택했고 선택 결과는 모두2명 PUSH·in 방향이다. 확인 구간에서 이동1.83–3.49m·입구 상자 여유 하한0.84–1.71m, ①은 선행 AT/후행 CLIMB 완수·나머지는 미완수 사례다. [모아보기](output_etc/check/warehouse_self_walk_20261112/index.html), [조건·검증](output_etc/check/warehouse_self_walk_20261112/report.md).
- 각각30초1080p/15fps 영상·보행 clip·구간 모음, 관절/장면/정책 출력/접촉력 크기 및 `.blend`를 저장했다. 전 영상 전체 디코딩·기하/보행 재검사·finite·모션 SHA 중복 검증 및 Blender5파일 재개방 위치 오차2.44e-7m 이하를 확인했다. 새 검색/렌더는 정상 종료했고 학습·MPS·다른 CUDA client에 종료 명령을 보내지 않았다. 초기 render queue의 system Python NumPy 누락은 절대 conda interpreter로 재실행해 해결했다.

### Warehouse 최신 checkpoint 시퀀스·Blender 영상 추출

- 사용자 요청으로 검색 시작 시 최신5500epoch를 고정하고 Warehouse·1/2명 PUSH·나머지 BEFORE pairing1.0을 평가 프로세스에만 적용했다. 네 방향·닫힘/부분 개방·무작위 담당자/과제·deterministic/stochastic을1152episode 탐색했다. 동일 seed 재실행128개를 제외한1024경우에서 관절·루트가 다른 후보113개를 보존했다.
- 닫힌 입구에서1명 PUSH→AT/SIT 및 추가AT, 2명 PUSH→AT/SIT, 2명 PUSH→AT/CLIMB의 서로 다른 세 시퀀스를 선택했다. 각 영상에서 비 PUSH 전원 개인 과제 ever 충족을 확인했다. 1080p/15fps·30.0/26.7/23.3초 H.264의 frame 수·전체 디코딩, 관절32개·finite·루트/몸체 정렬·quaternion·시퀀스 중복을 검사했다. 관절/속도·몸체·물체·기하/역할 JSON·MJCF/발 mesh·재사용 `.blend`와 영상 묶음을 저장했다. [영상 모아보기](output_etc/check/warehouse_showcase_20261112/index.html), [조건·검증](output_etc/check/warehouse_showcase_20261112/report.md).
- 정책 동작/학습 설정·보상·동결·원본 PTH는 유지했고 렌더링의 앞쪽 벽만 반투명 표시했다. 검사용 검색/렌더 서비스는 종료했으며 본학습2797534·MPS374795 및 기존 GPU 작업의 PID/start time을 보존했다. 첫 렌더 interpreter 경로와 중간 검색 wrapper 종료 오류는 로그에 보존하고 정상 실행/산출물을 별도로 검증했다.
- 후속 요청으로 저장113후보의 선행 상자20cm 이내 첫 도착 전 후행 이동을 비교했다. 후행 과제 성공6개에서는1Hz 누적 이동 최대1.25m였다. 상자 도착28.63초 전 후행10.39m 이동하지만 SIT 미완수인 사례를 추가 영상04와 별도 관절/장면/Blender 묶음으로 제공했다. 1080p·450frame 전체 디코딩 및 Blender 재개방 위치 오차2.93e-7m 이하 확인. [활동량 비교](output_etc/check/warehouse_showcase_20261112/follower_activity/report.md). 새 정책 평가·학습 변경은 없다.

## 2026-11-11

### Room 기하 참조 복원·혼잡 샘플링 보완·5500epoch 재개

- 사용자 요청으로11/11 22:27:34의 Room 배치 소진을 조사했다. 창고 기하 사전 생성 후 `_room_half`/`_wall_local` 복원 누락을 수정했다. CPU 기하 재현1024개 중1020개 불일치가0이 됐고 실제 방/센서/벽 CPA 참조를 맞췄다. 물리 기하·창고 변형과 사전 생성 RNG는 exact 보존한다.
- 혼잡7×7m Room의 후반 재시도에만 그룹 순서 섞기·후보16→64를 적용했다. 정상 경로16후보·전체256회·역할/BEFORE 우선순위·거리/충돌 조건·방/상자 크기·보상식/계수·동결·YAML을 유지한다. CPU106 passed; 문제 두 역할 조합16384 Room reset 및4096 혼합reset·센서/기하·16물리step·부분reset 통과. [변경·검증](output_etc/check/room_sampling_fix_20261111/report.md).
- 원본5500epoch/frame720896000·optimizer162개/step264000·curriculum anchor0의 계약/finite·복원 exact를 확인하고 본학습을 재개했다. 사용자 systemd `tokenhsi-sensor-warehouse-cliffroom-roomfix-20261111-232822.service`·GPU0/MPS·4명/1024환경/8토큰/900step·seed42·Linger=yes·Restart=no. Python2797534·첫5501epoch·확인 시5508epoch까지741 scalar finite·cap/검사 override 없음·MPS/source SHA/기존 GPU 작업 PID/start time 보존을 확인했다. [실행 정보](output/stage2_sensor_900_warehouse_coop_cliffroom/resume_roomfix_20261111-232822/launch.json), [시작 확인](output_etc/check/room_sampling_fix_20261111/startup_verification.json), [운영](markdowns/USAGE.md#warehouse-cliffroom-5500epoch-재개).

### Warehouse CliffRoom 1000epoch 백그라운드 재개

- 사용자 요청으로 `stage2_sensor_900_warehouse_coop_cliffroom`의 최신 저장1000epoch/frame131072000에서 재개했다. 이전 run은11/10 21:45:33 Room 배치 재시도 소진으로 종료됐고 마지막 완료 로그1483epoch였다. 이번 작업은 기존 코드·YAML·배치/보상/동결을 유지한 운영 재개이며 원본 PTH(SHA `f3d1bb8603558c35a103ce62ec2907c66efc4d116fec439049878e3c05c2e4d1`)·이전 로그를 보존한다.
- CPU checkpoint 검사:동일 환경/학습 계약·모델 finite·optimizer162개/step48000·curriculum anchor0. GPU0/MPS·4명/1024환경/8토큰/900step·seed42·사용자 systemd `tokenhsi-sensor-warehouse-coop-cliffroom-resume-20261111-020937.service`·app.slice·Linger=yes·Restart=no로 실행했다. Python2441521·첫1001epoch/frame131072000·curriculum update1000·1002epoch까지741 scalar finite·saved config의 resume/샘플러/1000000epoch 상한·실제 MPS·source SHA를 확인했다. 다른 GPU 작업과 MPS의 동일 PID/start time을 보존했다.
- [실행 정보](output/stage2_sensor_900_warehouse_coop_cliffroom/resume_20261111-020937/launch.json), [시작 확인](output_etc/check/warehouse_cliffroom_resume_20261111/startup_verification.json), [운영·안전 종료](markdowns/USAGE.md#warehouse-cliffroom-1000epoch-재개). 이전 Room 배치 오류를 수정하거나 장기 안정성을 검증한 작업은 아니다.

## 2026-11-10

### Warehouse 공동 PUSH Cliff·Room 조정 실험 추가

- coop 3000epoch 중간 분석(Cliff 낙상0.92/에피소드·Room 과제 정체·전체 성공 epoch1000 이후 정체)에 따라 사용자 승인으로 `stage2_sensor_900_warehouse_coop_cliffroom` 환경/학습 YAML·train/train_vnc/test/vnc/eval을 추가했다. coop 대비 장면20/30/17/33%, Cliff 전원 무대 위(`all_edge_probability`1.0), Room7–14m만 다르다. 코드는 실험 이름 등록(experiments/launch/evaluate)만 바꿨다. [명세](markdowns/reference/stage2_sensor_900.md#warehouse-공동-push-cliffroom-조정-stage2_sensor_900_warehouse_coop_cliffroom).
- 처음 승인한 Room6–14m는 1024환경 짧은 학습 첫 epoch에서 `layout exhausted`로 종료됐다. Room100%·정사각형 고정 크기 reset 검사(1024환경×8회)에서 6m는 첫 reset부터 소진, 7·8·9m는 통과하여 사용자 선택으로7–14m로 바꿨다. 검사용 runner만 계약 비교를 건너뛰었고 repo 코드는 그대로다. [로그](output_etc/check/warehouse_coop_cliffroom_20261110/room_size/).
- CPU: validate_experiment·coop와의 차이 3개+run 이름·계약 분리·Cliff PUSH/바닥 시작0과 unreachable4.0명 확인, 관련 테스트152 passed/3 skipped. [확인](output_etc/check/warehouse_coop_cliffroom_20261110/cpu_config_check.txt). GPU:1024환경 `CHECK=1`3epoch 정상 종료, Distill1500 전이·동결131텐서/입력 RMS exact·action/value head 갱신·scalar741 finite·Cliff4.0명/PUSH0·배치 실패0. [요약](output_etc/check/warehouse_coop_cliffroom_20261110/train_check_summary.txt).
- 사용자가 기존 coop 서비스를 안전 종료 스크립트로 중단했다(inactive/dead, 3000epoch PTH 보존). 사용자 systemd `tokenhsi-sensor-warehouse-coop-cliffroom-20261110-144456.service`·GPU0/MPS·4명/1024환경/8토큰/900step·seed42·Distill1500 fresh 본학습을 시작했다. Python1966251·MPS client·epoch5·scalar741 finite·검사 override 없음·저장 config의 비율/Room7–14m/전이 경로를 확인했고 다른 GPU 작업과 MPS 서버의 PID/start time을 보존했다. [실행 정보](output/stage2_sensor_900_warehouse_coop_cliffroom/launch_20261110-144456/launch.json), [시작 확인](output_etc/check/warehouse_coop_cliffroom_20261110/startup_verification.json), [운영](markdowns/USAGE.md#warehouse-공동-push-cliffroom-본학습-서비스). 장기 성능은 미검증이다.

### Warehouse 공동 PUSH 본학습 백그라운드 시작

- 사용자 요청으로 `stage2_sensor_900_warehouse_coop`·GPU0/MPS·4명/1024환경/8토큰/900step·seed42·Distill1500 fresh 본학습을 시작했다. 기존 구현·설정은 유지하고 CHECK/MAX_ITERATIONS·resume·검사/시나리오 override를 해제했다.
- 사용자 systemd `tokenhsi-sensor-warehouse-coop-20261110-003205.service`·app.slice·Linger=yes·Restart=no로 대화/SSH 종료와 독립적으로 실행한다. Python1534227·active/running·첫epoch1/frame0·curriculum update0·epoch3까지741 scalar finite·saved config의 fresh 전이/새 sampler/1000000epoch 상한·실제 MPS 접속·source SHA를 확인했다. 공유 MPS374777/374795·다른 GPU 작업1254579/1433539의 PID/start time을 보존했다.
- [실행 정보](output/stage2_sensor_900_warehouse_coop/launch_20261110-003205/launch.json), [시작 확인](output_etc/check/warehouse_coop_start_20261110/startup_verification.json), [운영·안전 종료](markdowns/USAGE.md#warehouse-공동-push-본학습-서비스). 결과는 `output/stage2_sensor_900_warehouse_coop/Stage2Sensor900WarehouseCoop_10-00-32-07/`에 저장한다. 장기 성능은 미검증이다.

### Warehouse1~3명 공동 PUSH·좌우/안팎 배치

- 사용자 시제품 승인 후 `stage2_sensor_900_warehouse_coop` 환경/학습 YAML·train/train_vnc/test/vnc/eval을 추가했다. 기존 Safety의 장면 비율·Bridge/Cliff/Room·모델/PPO/보상/동결·native 밀도/마찰을 유지하고 Warehouse만 PUSH1/2/3명·네 방향 균등으로 확장했다. 기존 설정과500epoch PTH는 보존한다.
- 실제 상자 하나를 공유하는 COUPLED 그룹·인원별 물리 bank·방향별 입구·구역 정렬·개인 목표·전체 상자 센서를 연결했다. 방6×8m와 같은 높이의9m apron, 좌우2.6m/안팎3m, 바깥 PUSH의 실내 시작,50/25/25 RSI를 적용했다. 작업자 수3/2/1명에 맞춰 진입률 분모 및 인원/방향/RSI별 진단을 추가했다. 초기 혼잡 조합 배치 소진을 거리 조건부 source 샘플링과 외부 공간 확대로 수정했다.
- CPU103 passed;1024환경×4reset(Cliff173/Bridge1818/Room724/Warehouse1381)·36조합·공유 bank/구역·입구/바닥/센서·부분 reset·실제 화면12개,33회 물리 step을 확인했다. PPO epoch3/frame393216:동결131텐서/입력 RMS exact, critic/head/sensor/AMP·value/AMP 통계 갱신, optimizer162개·741 scalar finite. 새 PTH closed16환경16step 재로딩 완료. 기존 Warehouse 배치/RNG exact 및 기존 Safety/agent/network SHA 보존. [검증](output_etc/check/warehouse_coop_20261110/report.md).
- 본학습은 재시작하지 않았다. 기존 Safety service inactive/dead/MainPID0·500epoch SHA·공유 MPS 및 외부 GPU 작업의 동일 PID/start time을 확인했다. 새 실험은 기본 Distill1500 fresh 전이이며 이전 PTH와의 계약 예외를 추가하지 않았다. 장기 성능·새 noVNC 접속은 미검증이다.

## 2026-11-09

### Rollout 경계 reset 수정·500epoch 보존·학습 중단 유지

- 사용자 요청으로 `MAAgent.play_steps()`의 첫 reset 목록을 직전 self.dones에서 환경별 대표 agent-row로 구성하도록 한 줄 수정했다. 현재3시나리오 repo와 동일한 수정이다. 샘플링/보상/동결/학습 YAML은 유지한다. 마지막 rollout step에서 종료된 환경을 다음 행동 전에 초기화한다.
- `test_ma_rollout_reset.py`를 추가했다. 실제 play_steps/wrapper의1/4명·무종료/단일/복수 환경 종료·초기 done·에피소드 보상 중복·개인 terminate bootstrap을 검증한다. 관련 CPU52 passed, 두 repo의 수정된 ma_agent/test가 동일하다.500epoch/frame65536000 PTH는 SHA 보존·모델 strict 로딩·optimizer162개 로딩·모델/optimizer/입력/value/AMP RMS finite·동일 환경 resume 계약 호환을 CPU로 확인했다. GPU 학습/평가 및 수정 후 동적 resume는 실행하지 않았다. [검증](output_etc/check/rollout_reset_fix_20261109/report.md).
- Roomopen Python1312743의 cwd/config/cgroup·MPS mapping을 확인하고 `terminate_client 374795 1312743` 응답0 후 해당 서비스만 중단했다. inactive/dead·MainPID0·Restart=no·대상PID/GPU 부재를 확인했다. 현재3시나리오1433539·TeamHOI1254579·공유 MPS374777/374795의 PID/start time과 작업 진행을 보존했다. 종료 직전 로그577epoch·원본500 PTH/기존 로그는 보존한다.
- **사용자 요청에 따라 재시작하지 않고 중단 상태를 유지한다.** 향후 재개 대상은 `Stage2Sensor900WarehouseSafety_00000500.pth`이며 현재 새 학습 서비스는 등록/시작하지 않았다. [중단 상태·재개 참고](markdowns/USAGE.md#rollout-reset-수정과-500epoch-보존).

### Warehouse safety Distill1500 fresh 전이로 교체

- 사용자 요청으로6000epoch resume 실험을 종료하고 같은 개선안 YAML을 `output/stage2/Stage2Distill900_00001500.pth`에서 새로 전이했다. 기존 정책/critic·입력/AMP 통계를 전이하며 센서·optimizer·value RMS·학습 epoch/frame/curriculum은 새로 시작한다. Source1500/SHA `fcadf2297bedf8e3a8d010fd93d045f63dab412f0c85e65d7b89de4cc15a8a1e`를 확인했다. 샘플링·보상·동결 범위는 변경하지 않았다.
- Roomopen Python1267295·cwd/config/cgroup/MPS client 확인 후 `terminate_client 374795 1267295` 성공0→해당 서비스만 중단했다. inactive/dead·MainPID0·대상PID 부재를 확인했고 공유 MPS374777/374795·현재3시나리오1256400·TeamHOI1254579의 동일 start time을 보존했다. 이전 run의 마지막 로그6171epoch와 기존 PTH/로그는 보존했다.
- 사용자 systemd `tokenhsi-sensor-warehouse-safety-fresh-20261109-204146.service`·GPU0/MPS·4명/1024환경/8토큰/900step·seed42·Linger=yes·Restart=no로 본학습 시작. Python1312743·첫epoch1/frame0·curriculum update0·521개 scalar finite·resume/검사 cap 없음·saved config의 load_checkpoint=false·전이 경로·실제 MPS 접속을 확인했다. [검증](output_etc/check/warehouse_safety_fresh_20261109/report.md), [운영](markdowns/USAGE.md#warehouse-safety-fresh-본학습-서비스). 짧은 시작 확인이며 장기 성능은 미검증이다.

### Warehouse safety 분포 조정·6000epoch resume

- 사용자 승인으로 `stage2_sensor_900_warehouse_safety` 전용 YAML·train/train_vnc/test/vnc/eval을 추가했다. 장면5/43/17/35%, Bridge90% pusher1+crosser3 횡단 curriculum+10% 기존 협력, Cliff40% 전원 안전/60% 안전2+coupled2, Room BEFORE/COUPLED/MIXED30/40/30·BEFORE 두 쌍·독립 PUSH 제외·벽 경로 요청75%, Warehouse worker pairing0.8을 적용했다. 입구 PUSH는1명·상자 크기/기하·50/25/25 RSI 유지. [명세·기대 비율](markdowns/reference/stage2_sensor_900.md#warehouse-safety-샘플링).
- 모델·PPO·보상·센서·동결 범위는 유지한다. Actor encoder/coordination/sigma·입력 RMS 동결, critic 전체·head·sensor·AMP 및 AMP/value 정규화 학습. 기존 RSI→safety만 엄격한 전환 계약 예외를 허용하며 모델/optimizer/RMS/global epoch/frame를 복원한다. 새 curriculum anchor6000을 저장하여 첫6001epoch의 schedule을0에서 시작하고 이후 resume 시 유지한다.
- 현재3시나리오 repo에서 확인된 reset 배치64회 소진 문제도 예방하도록 같은 역할/그룹/장면의 RSI를64회마다 재추출하며 최대256회 시도한다. 첫 GPU 검증에서 발견한 curriculum 플래그 초기화 순서 오류를 수정했다. 모델/리워드/물리 파일의 동작은 변경하지 않았다.
- CPU44 passed;1024환경4096 reset(Cliff203/Bridge1781/Room707/Warehouse1405)·BEFORE/COUPLED/실내외 역할·Bridge 다중 crosser/발 지지·센서 NumPy 경계 허용검사·부분 reset 통과. 기존6000 상태 exact 복원→PPO6001, 동결131텐서·입력 RMS exact, critic/head/sensor/AMP·value 통계 갱신, optimizer288000→288048, scalar521개 finite. 새 checkpoint8환경16step headless 재로딩 완료. 새 noVNC 화면·장기 성능은 미검증이다. [검증](output_etc/check/warehouse_safety_20261109/report.md).
- 기존 Roomopen 학습은 이미 종료되어 중단할 learner가 없었다. 기존6000epoch 원본 PTH에서 사용자 systemd `tokenhsi-sensor-warehouse-safety-20261109-195459.service`·GPU0/MPS·4명/1024환경/8토큰/900step·seed42로 본학습 재개. Python1267295·첫6001epoch·신규 curriculum update0·521개 scalar finite·실제 MPS·Linger=yes·검사용 cap 해제를 확인했다. 현재3시나리오 학습과 외부 GPU 작업은 동일 PID/start time을 보존하고 계속 진행 중이다. [운영·안전 종료](markdowns/USAGE.md#warehouse-safety-본학습-서비스), [시작 확인](output_etc/check/warehouse_safety_20261109/startup_verification.json).

## 2026-11-08

### Warehouse RSI 50/25/25·freeze 확인·MPS 안전 교체

- 사용자 승인으로 `stage2_sensor_900_warehouse_bridge_init_rsi`와 전용 train/train_vnc/test/vnc/eval을 추가했다. 최신 장면5/40/25/30%·Bridge20/40/40%·900step·모델/PPO를 유지하고 Warehouse만 닫힘50%·부분 개방25%·목표 개방25%로 초기화한다. 상자/PUSH 담당자를 함께 옮기며 작업자3명은 모두 외부 시작한다. 재시도 중 모드를 유지하고 거리/near·충돌을 검사한다. 평가는 기본 closed, 추론 전용 WAREHOUSE_START override·모드별 진단을 추가했다.
- 다른 세션의 freeze 코드가 기준 repo와 SHA 일치함을 확인했고 수정하지 않았다. CPU120 passed, 기존 Warehouse 배치/RNG exact;1024환경×4회 reset에서 Warehouse1246개=closed626/partial315/open305·센서/부분 reset·고정 actor·실제 렌더링을 확인했다. PPO epoch2는 동결131텐서·입력 RMS exact, action head/critic 전체/센서/AMP 및 AMP/value 통계 갱신, optimizer162개·모델 finite다. 새 PTH의 closed/mixed16환경16step 추론도 완료했다. [검증](output_etc/check/warehouse_rsi_20261108/report.md).
- 기존 service/Python406675·cwd/config·MPS client/server를 대조하고 NVIDIA `terminate_client 374795 406675` 성공0 후 해당 서비스만 중단했다. inactive/dead·대상PID/GPU 해제, 공유MPS374795·다른 학습421090 동일PID 보존 및 epoch1133→1155 진행을 확인했다. 기존1k PTH/SHA·로그를 보존했다.
- 사용자 systemd `tokenhsi-sensor-warehouse-rsi-20261108-213952.service`·Linger=yes·app.slice·Restart=no로 GPU0/MPS·4명/1024환경/8토큰·seed42·Distill1.5k fresh 본학습을 시작했다. Python570297·epoch7·396개 scalar finite·실제 MPS 연결·검사용 override 없음·세 모드 샘플링을 확인했다. [실행 정보](output/stage2_sensor_900_warehouse_bridge_init_rsi/launch_20261108-213952/launch.json), [시작 확인](output_etc/check/warehouse_rsi_20261108/startup_verification.json), [운영](markdowns/USAGE.md#warehouse-rsi-본학습-서비스). 장기 성능 개선을 입증한 결과는 아니다.


### Warehouse Bridge 초기화·샘플링 변경과 MPS 본학습

- 승인한 별도 실험 `stage2_sensor_900_warehouse_bridge_init`과 train/train_vnc/test/vnc/eval을 추가했다. cliff5% / bridge40% / room25% / warehouse30%; Bridge 내부 완료20% / 근접40% / 선두1.5cm 삽입40%다. 완료 분기는 crosser 입구0.8–2m 정렬, 해당 출발 무대 운반 상자 뒤1–2m 배치를 적용한다. Warehouse 전용 배치·물리 bank·모델·PPO·보상·동결 범위는 유지했다. [명세](markdowns/reference/stage2_sensor_900.md#warehouse-bridge-초기화-실험).
- CPU35 passed;1024환경×4 reset·센서/부분 reset·실제 화면10개, PPO epoch2의 동결131텐서·입력 RMS exact 및 critic/head/센서/AMP·value 통계 갱신, 기존4.5k strict 로딩, 새 PTH의 Warehouse8환경16step 및 혼합8환경900step 실행을 확인했다. 기존 config의 기하 함수/RNG exact·Warehouse/network 파일 SHA 불변이다. 장기 횡단 성능·완전한 물리 안정성·새 noVNC 화면을 검증한 결과는 아니다. [검증 보고서](output_etc/check/warehouse_bridge_init_20261108/report.md).
- 기존 roomopen 서비스만 신원 확인 후 중단하고4.5k PTH·로그를 보존했다. GPU0·tokenhsi_sm120·4명/1024환경/8토큰/900step·seed42·Distill1.5k fresh 본학습을 사용자 systemd `tokenhsi-sensor-warehouse-bridge-init-20261108-174320.service`로 시작했다. 공유 MPS·별도3장면 학습은 유지했다. 새 train/train_vnc는 MPS가 없으면 실행을 거부한다.
- 본학습 Python406675·epoch4·381개 scalar finite·실제 MPS client·검사용 override 없음·Linger=yes를 확인했다. [시작 확인](output_etc/check/warehouse_bridge_init_20261108/startup_verification.json), [학습 로그](output/stage2_sensor_900_warehouse_bridge_init/launch_20261108-174320/train.log), [운영 명령](markdowns/USAGE.md#warehouse-bridge-초기화-본학습-서비스).

## 2026-11-07

### Warehouse sensor 동결 수정·본학습 새 시작

- 사용자 요청으로 `TokenHSI_stage12_teacher_sensor`의 수정된 학습 범위를 동일하게 적용했다. Actor encoder·actor Stage2 coordination·sigma·입력 RMS만 동결하고 action head·critic 전체·센서·AMP discriminator 및 AMP/value RMS를 학습한다. Value RMS는 fresh 초기화한다. 공통 sensor agent/network는 기준 repo와 파일 내용이 동일하며 이 checkout의 두 센서 실험에 적용된다.
- 네 장면 환경 YAML은 기존 SHA256 `66df2951ac87bc5dbfb856e15858435815e3daf05bc45cd4b77e938557b0af9f` 그대로다. 시나리오/샘플링/센서 위치/보상/증류 구현은 유지했다. Warehouse run 이름을 `Stage2Sensor900WarehouseSquare`로 분리하고 새 학습 계약 `square_sensor_frozen_actor_coordination_v1`을 저장한다. 이전 전체 동결 checkpoint는 테스트만 가능하며 새 resume으로는 거부한다.
- CPU20 passed;네 장면1024환경 PPO2epoch 완료. 동결131텐서·입력 RMS exact, action head/critic encoder·coordination·value head/AMP 업데이트, AMP/value 통계 갱신, optimizer162파라미터·모델 finite를 확인했다. [검증](output_etc/check/warehouse_square_training_20261107/train_verification.json). 장기 성능 개선을 검증한 결과는 아니다.
- 기존 warehouse 학습은 작업 시작 시 실행 중이 아니었다. 다른 repo의3장면 서비스/Python PID10208과 MPS PID13960을 종료하지 않았다. GPU0·tokenhsi_sm120·4명/1024환경/8토큰/900step·seed42·네 장면 각25%로 Distill9001.5k에서 새 본학습을 시작했다. 사용자 systemd `tokenhsi-sensor-warehouse-square-20261107-192146.service`·app.slice·Linger=yes·Restart=no를 사용한다. [실행 정보](output/stage2_sensor_900_warehouse/launch_square_20261107-192146/launch.json), [학습 로그](output/stage2_sensor_900_warehouse/launch_square_20261107-192146/train.log), [운영 명령](markdowns/USAGE.md#창고-포함-본학습-백그라운드-서비스).
- 본학습 서비스 active/running·Python PID96971의 cwd/config·물리GPU0·1epoch 업데이트·기록된289개 scalar finite·네 장면 샘플 발생·확인용 override 없음·별도3장면 서비스 active를 확인했다. [시작 검증](output_etc/check/warehouse_square_training_20261107/startup_verification.json). 결과는 `output/stage2_sensor_900_warehouse/Stage2Sensor900WarehouseSquare_07-19-25-17/`이다.

## 2026-11-06

### Stage2 sensor 900 warehouse 본학습 백그라운드 시작

- 사용자 요청으로 이 checkout에서 GPU0·tokenhsi_sm120·4명/1024환경/8토큰/900step·headless·네 장면 각25% 본학습을 Distill9001.5k fresh 전이로 시작했다. CHECK/MAX_ITERATIONS/RESUME_CHECKPOINT·전이 override·SCENARIO/SENSOR_MASK/SENSOR_VIS를 해제했고 YAML은 변경하지 않았다.
- 사용자 systemd `tokenhsi-stage2-sensor-900-warehouse-20261106-221009.service`·Linger=yes·app.slice·Restart=no로 대화/SSH 종료와 독립적으로 실행한다. 서비스 active/running·Python PID2232845의 cwd/실제 config/GPU0 UUID/MPS 연결·epoch2·TB334태그 유한값을 확인했다. 기존3장면 학습 PID2196991과 MPS PID388225를 보존했다.
- [실행 정보](output/stage2_sensor_900_warehouse/launch_20261106-221009/launch.json), [초기 확인](output_etc/check/stage2_sensor_900_warehouse_production_start_20261106/startup_verification.json), [학습 로그](output/stage2_sensor_900_warehouse/launch_20261106-221009/train.log). 결과·TB는 `output/stage2_sensor_900_warehouse/Stage2Sensor900Warehouse_06-22-10-11/`, 상태·로그·종료 명령은 [USAGE](markdowns/USAGE.md#창고-포함-본학습-백그라운드-서비스)에 기록했다.


### Stage2 sensor 900 warehouse 별도 실험

- 승인한 `stage2_sensor_900_warehouse` 환경/학습 YAML·train/train_vnc/test/vnc/eval을 추가했다. cliff/bridge/room/warehouse 각25%, 같은 Distill9001.5k 전이·원본 전체/RMS 동결·센서 분기만 PPO·4명/1024환경/8토큰/900step을 유지한다. 기존3장면 config와 실행 중 학습은 보존했다.
- 참고 repo의4–6m 중앙 입구·전원 외부 시작·측면 PUSH·조건부 GPU 배치를 현재 통합 Stage2 그래프/BEFORE alias·고정1인용 bridge bank에 연결했다. 장면 수·벽 CPA·발밑 지형·열림/진입 진단·정식 평가 집계를 확장했다. [명세](markdowns/reference/stage2_sensor_900.md#창고-확장-stage2_sensor_900_warehouse).
- CPU159 passed, 실제1024환경×20회 reset(창고5145개), 부분 reset/고정 actor·센서 검사 통과. PPO epoch2→resume4에서 원본279텐서·세 정규화 bitwise 동일, 센서14텐서만 갱신됐다. 혼합/창고 고정 각32환경·seed0·900step 평가와 CPU 재채점 exact, 기존3장면 checkpoint 추론·닫힘/열림 화면5개·새 VNC 실제 화면/HTTP200/RFB 확인. 확인 프로세스·6084 종료, 기존 학습/MPS 보존, 새 본학습 미실행.
- 추가6step 물리 smoke의 창고 PUSH 상자 이동은 첫 step 최대6.06cm·6step21.73cm다. finite·기하 검사는 통과했으며 완전한 초기 동적 안정성·정책 성능 입증과 구분한다. [전체 검증·제한·로그](output_etc/check/stage2_sensor_900_warehouse/report.md).


### Stage2 sensor 900 본학습 백그라운드 시작

- 사용자 요청으로 GPU0·`tokenhsi_sm120`·4명·1024환경·8토큰·900step·headless 본학습을 fresh Distill9001.5k 전이로 시작했다. 확인용 CHECK/MAX_ITERATIONS/RESUME_CHECKPOINT와 SCENARIO/SENSOR_MASK를 해제했고 YAML 학습 구성은 변경하지 않았다.
- 사용자 systemd `tokenhsi-stage2-sensor-900-20261106-202813.service`, `Linger=yes`, 별도 app.slice에서 대화/SSH 종료와 독립적으로 실행한다. 서비스 active/running·Python PID2196991의 cwd/실제 config/MPS 연결·6epoch·322개 scalar의 유한값을 확인했다. 기존 MPS PID388225는 보존했다.
- [실행 정보](output/stage2_sensor_900/launch_20261106-202813/launch.json), [초기 진행 확인](output/stage2_sensor_900/launch_20261106-202813/startup_verification.json), [학습 로그](output/stage2_sensor_900/launch_20261106-202813/train.log). 결과는 `output/stage2_sensor_900/Stage2Sensor900_06-20-28-15/`; 상태·로그·종료 명령은 [USAGE](markdowns/USAGE.md#본학습-백그라운드-서비스)에 기록했다.


### Stage2 sensor 900: 원본 전체 동결·세 시나리오

- `stage2_sensor_900`을 추가했다. Distill9001.5k를 전이하고 기존 actor/critic·coordination·head·sigma·AMP·정규화를 동결하여 새 actor/critic 센서 MLP·0초기화 투영만 PPO로 학습한다. 기존 distillation 코드는 유지한다.
- 기준 repo의48×48 센서와 cliff/bridge/room을 통합 Stage2에 연결했다. Bridge pusher1/2/3명 균등·별도 crosser·고정 크기 자산·X+0.2 간격, BEFORE 선행자 공통 X/Y0.40–0.60·Z0.35–0.50 전용 bank를 적용했다. 정식 평가의 unreachable 분모 제외와 센서 suffix 뒤 group 파싱을 연결했다.
- train/train_vnc/test/vnc/eval·SCENARIO/SENSOR_MASK/SENSOR_VIS와 문서를 추가했다. 동시에 시작한 VNC들이 같은 X display를 고르던 경쟁은 display별 flock으로 수정했다.
- CPU115 passed;1024환경4096 reset에서 배치/크기/센서·고정 actor/부분 reset 검사 통과. 최종 PPO2epoch와 noVNC resume2epoch 후 원본279텐서·세 정규화가 bitwise 동일하고 센서14텐서만 갱신됐다. 혼합8환경·cliff16환경 각각900step과 CPU 재채점 exact, cliff unreachable12명 제외 분모52를 확인했다. 학습·세 시나리오 noVNC 실제 화면/HTTP 확인, 본학습 미실행, 확인 프로세스·포트 종료, 기존MPS 보존. [검증 범위·로그·화면](output_etc/check/stage2_sensor_900/report.md), [명세](markdowns/reference/stage2_sensor_900.md).


### 현재 서버 stage2_distill_900 런타임·데이터·noVNC

- 기준 repo `stage_1_2_base_fuv1`의 `tokenhsi_sm120`·사용자 설치 Xvfb 탐색을 적용했다. Distill900의 이전 서버 GPU 제한을 해제하고 GPU0·숫자/UUID를 사용한다. 학습의 `HEADLESS=0`과 `stage2_distill_900_train_vnc.sh`를 추가했으며 학습 noVNC6080·테스트6081, 4명·1024환경·8토큰 기본값을 유지한다.
- 모션297개 참조와 Stage1 26k·BEFORE21k 심링크를 확인했다. 두 teacher 경로는 `checkpoints/`로 옮겼다. 사용자 지시에 따라 아직 없는 coupled17k를 대체하지 않았고 실제 증류 학습은 미완료다.
- 관련 CPU81 passed·shell syntax·CUDA/PhysX를 확인했다. 정식 train/train_vnc는1024환경·student 전이·BEFORE 로딩 후 coupled 파일 부재에서 종료했다. 별도 초기화 전용 student로1024환경16step·1600×900 렌더링·headless1환경8step·테스트noVNC HTTP200/WebSocket101/RFB3.008·화면을 확인했다. optimizer 업데이트0회이며 학습/성능/resume 성공을 의미하지 않는다. 확인 프로세스·6080/6081/6082는 종료했고 기존MPS는 보존했다. [범위·로그·화면](output_etc/check/stage2_distill_900_setup/report.md), [실행법](markdowns/config.md).

### Distill900 0.5k·1k·1.5k 평가와 interaction group 성공률

- 기존 모든 지표를 보존하고 풀네임의 Independent Interaction Success Rate·Enabling Interaction Group Success Rate·Coupled Interaction Group Success Rate·Mixed Interaction Group Success Rate를 point/region별로 추가했다. Enable은 선행/후행 각각 ever, coupled는 실제 공유 상자 중심 ever, mixed는 seed별 세 유형 동일 가중 평균이다. Enable 선행자를 solo에 중복 집계하지 않고 공동 그룹은 인원과 관계없이1개로 센다. Seed 평균/표본std와 CPU trace 재채점도 연결했다. [정의](markdowns/reference/evaluation.md#통합-stage2의-interaction-group-성공률).
- 사용자 지정0.5k/1k/1.5k PTH를4명·8토큰·512환경×seed0/1/2·ordered·native random_scenario·900step으로 GPU6/7/2에서 평가했다. CPU21 passed,9회 모두 GPU/CPU 기존·새 지표/건수·finite 일치,3모델×각seed 초기 상태 전체필드 exact 일치,epoch/frame/SHA256·관계 분모 분할을 확인했다. 기존 학습 PID265779는 보존했고 평가 프로세스는 모두 종료했다. [보고서](output_etc/check/stage2_distill_900_eval_0500_1000_1500_20261106/report.md), [검증](output_etc/check/stage2_distill_900_eval_0500_1000_1500_20261106/verification.json).
- 영역 Mixed Interaction Group Success Rate는0.5k/1k/1.5k=75.79±0.14/75.73±0.47/76.32±0.21%다. Enabling Interaction Group Success Rate는41.99±1.04/40.67±0.56/42.66±1.24%다. 이번 Independent 지표는 혼합 샘플 안 solo이며 독립 전용 장면 또는2/8/16명 확장 평가 결과가 아니다.

### 900step VNC의 GPU UUID 거부 수정

- VNC가 숫자 GPU를 UUID로 바꾼 뒤 자식 launcher에 전달할 때900 버전 GPU 제한이 이를 거부하던 오류를 수정했다. `nvidia-smi`로 UUID/숫자를 물리 인덱스로 확인한 뒤 기존2/6/7 제한을 적용한다. 실행 명령과 실행 중 학습은 변경하지 않았다.
- GUI→자식 launcher의 숫자/UUID 허용·제외 GPU 거부를 포함해 CPU12 passed. GPU6의 실제 runtime UUID를 그대로 자식 test wrapper에 전달해500iteration student PTH·4명·혼합·1환경·2step 화면 없는 추론을 완료했다. 검증 프로세스는 종료했으며 VNC 화면은 새로 실행하지 않았다. [CPU](output_etc/check/stage2_900_gui_uuid_20261106/cpu.log), [추론](output_etc/check/stage2_900_gui_uuid_20261106/rollout.log).

### Distill900 teacher 사용자 지정21k·17k

- `stage2_distill_900` 학습 YAML의 teacher 경로만 사용자 지정 enable21k(11월5일22:58 run)·coupled17k로 변경했다. 1200step `stage2_distill`의 teacher와 공용 심링크는 유지한다. 실행 명령은 기존900 버전 train wrapper 그대로다.
- CPU에서 두 PTH의 epoch·원래 fusion·모델/RMS strict 로드·동결·사람별 행동 선택·finite를 확인했다. 학습과 시뮬레이션은 실행하지 않았다. [검증](output_etc/check/stage2_distill_900_teachers_21k17k_20261106/verification.json).

### 통합 RL·distillation 900step 컨피그

- 사용자 요청으로 `stage2_900`·`stage2_distill_900` 환경/학습 YAML과 전용 train/test/VNC/eval wrapper를 추가했다. 기존1200step 설정에서 에피소드 제한과 실험/저장 이름만 분리했으며 모델·샘플링·보상·teacher·수집/업데이트 설정은 같다. 학습 검증·viewer·정식 평가도900step으로 연결했다. 새 wrapper의 GPU는2/6/7만 허용하며 기본6이다. [명령](markdowns/config.md), [추가 옵션](markdowns/USAGE.md#통합-stage2-900step-버전).
- CPU203 passed/3 skipped, 원본과 YAML 차이·1024환경 학습 CLI 설정·잘못된1200step 입력 거부·wrapper/Python 구문·새 문서 링크를 확인했다. GPU6/7에서 각각16환경×6장면의 finite·alias·부분 reset·가중치 보존과899step 미종료/900step 종료·reset을 확인했다. Distill900 환경에서 기존1200step teacher 두 개의 로드/행동/동결도 확인했다. 학습 업데이트와 viewer는 실행하지 않았다. [CPU](output_etc/check/stage2_900_20261106/cpu.log), [RL 물리](output_etc/check/stage2_900_20261106/physics_rl/physics_validation.json), [distill 물리](output_etc/check/stage2_900_20261106/physics_distill/physics_validation.json).

- GPU2/6에서 기존 확인용 PTH로 각각16환경×seed0·ordered·900step 평가를 완료하고 CPU 재채점/성공 건수 exact 일치·finite를 확인했다. 이는 실행 경로 검증이며 새 모델 학습/성능 실험이 아니다. 검증 시뮬레이션은 모두 종료했다. [검증 요약](output_etc/check/stage2_900_20261106/verification.json).

### 통합 M≥N 토큰 예산 · ordered eval · Stage2 distillation

- 사용자 요청으로 통합 `stage2`의 M≥2N 제한을 M≥N으로 완화했다. 관계/행동을 뽑은 뒤 실제 작업 슬롯·ON_TOP 받침·enable alias를 합산하고 부족한 환경 행만 재샘플링한다. M≥2N은 기존 샘플/RNG를 보존한다. 불가능한 고정 프리셋과 M=N 고정mixed의 크기 비호환은 명확한 오류로 처리한다. [명세](markdowns/reference/stage2.md#물체와-관측).
- 통합 `stage2`·`stage2_distill` 및 두 teacher의4명/확장 viewer·eval 기본 배정을 ordered로 맞췄다. 단일 모델 eval에도 인원/토큰/5m 확장 인자를 연결하고 manifest/result에 배정 방식을 기록한다. 학습의 랜덤 구역 배정은 유지했다.
- `stage2_distill` 환경/학습 YAML·train/test/VNC/eval wrapper와 실행 가능한 teacher 모방 루프를 추가했다. Stage1 26k에서 시작해 actor encoder/RMS·critic/AMP를 고정하고 actor coordination/fusion/action head만 평균 행동 MSE로 학습한다. Enable/solo는 enable13.5k, 공동 PUSH는 coupled25.5k teacher 정답을 사용한다. Teacher 실행20iteration 후180iteration 선형 전환·65,536scene replay·optimizer/replay/epoch 복원을 지원한다. [실행](markdowns/USAGE.md#stage2-distillation), [학습 명세](markdowns/reference/stage2_distill.md).
- CPU265 passed/3 skipped/1 deselected. GPU5/6에서1024환경·기본1iteration 및 이어학습2iteration, 별도 설정의4명4토큰·student-only 수집1iteration을 확인했다. 검증 배치 MSE는 기본0.00963→0.00313, 이어학습0.00313→0.00209, student-only0.00778→0.00319로 감소했다. Stage1 원본의 action head 외159개 tensor와 입력RMS exact 일치·모델 finite·replay/optimizer 복원을 확인했다. [CPU](output_etc/check/stage2_distill_20261106/cpu_full.log), [체크포인트](output_etc/check/stage2_distill_20261106/checkpoint_verification.json).
- 두 teacher8명8토큰, student4명4토큰·8명8토큰을각16환경×seed0·ordered·1200step 실행하고 CPU 재채점/건수 exact 일치·finite를 확인했다. Student eval은 teacher를 로드하지 않는다. [teacher](output_etc/check/stage2_distill_20261106/teachers_eval8_m8/rescore.json), [student4](output_etc/check/stage2_distill_20261106/student_eval4_m4/general_seed0/rescore.json), [student8](output_etc/check/stage2_distill_20261106/student_eval8_m8/general_seed0/rescore.json). 이는 짧은 실행 검증이며 장기 성능 결과가 아니다. Viewer와 장기 학습은 실행하지 않았다.

## 2026-11-05

### 두 teacher 제로샷 viewer · 5명 이상 통합 그룹 배치

- enable/coupled PTH를 각각 원래 모델·전처리·RMS로 복원하고 공동 PUSH 그룹원만 coupled, enable 선행/후행·solo는 enable의 평균 행동으로 제어하는 test/VNC/eval wrapper를 추가했다. enable13.5k/coupled25.5k 정기 저장본의 읽기 전용 링크를 연결했다. 학습·가중치 병합은 없으며 경로/epoch/frame/해시·담당 규칙을 `teachers.json`에 기록한다. [실행법](markdowns/USAGE.md#두-teacher-제로샷-viewer).
- 5명 이상 추론은 solo/enable/coupled를 그룹 단위1:1:1로 순차 선택하고 1/2/2~4명을 할당한다. 초기 그룹 중심5m·BEFORE 배치 후 쌍 전체 중심 복원·불필요한 물체 물리/관측 제외·필수 alias/받침 보존·ordered 구역 배정을 viewer/eval에 공통 적용했다. 4명 학습은 유지한다. distillation 학습은 후속 작업이다. [설계](markdowns/reference/stage2.md).
- GPU5/6만 사용했다. CPU209 passed/3 skipped/1 deselected, 두 teacher 모델 각279개 tensor·RMS exact 복원/동결·행동 선택·제외 물체 영향 차단을 확인했다. 5/8/16명 각32환경×6장면에서5m·최대4명·관계/alias·부분 reset·finite·가중치 보존을 확인했다. 8명16환경×seed0의1200step 공간 채점과 CPU 재채점/성공 건수가 exact 일치했고 실제8명 VNC 렌더링을 확인했다. 기존 coupled 테스트 fixture와의 호환 오류를 수정 후 재검증했다. [CPU](output_etc/check/stage2_teachers_20261105/cpu_verified.log), [5명](output_etc/check/stage2_teachers_20261105/ordered5/physics_validation.json), [8명](output_etc/check/stage2_teachers_20261105/ordered8/physics_validation.json), [16명](output_etc/check/stage2_teachers_20261105/ordered16/physics_validation.json), [재채점](output_etc/check/stage2_teachers_20261105/eval8_ordered/rescore.json), [화면](output_etc/check/stage2_teachers_20261105/viewer_live.png).

### 통합 stage2 · BEFORE/COUPLED 관계별 RL 학습

- 사용자 승인 명세로 `stage2` 환경/학습 YAML과 train/test/VNC/eval 진입점을 추가했다. 4명 장면은 enable/coupled/혼합을 1:1:1로 샘플링하고 혼합은 BEFORE 2명+COUPLED 2명이다. 원래 관계별 물체·배치·보상을 연결하고 BEFORE AT는 20cm, solo/COUPLED AT는 3D 20cm 및 높이 1mm를 적용한다. [통합 명세](markdowns/reference/stage2.md).
- BEFORE/COUPLED 독립 임베딩과 공유 coordination/fusion을 추가했다. Stage1 26k에서 전이하고 actor encoder·입력 RMS를 동결한다. alias 물체 슬롯 충돌을 방지하고 BEFORE goalroute·대기 속도 감점을 연결했다. distillation과 5명 이상 실행은 아직 포함하지 않았다.
- CPU 검증 182 passed/3 skipped/1 deselected, wrapper 구문·문서 링크·diff 검사를 통과했다. 1024환경 확인 학습은 epoch2/frame262144까지 완료했으며 actor encoder 73개 tensor·입력 RMS의 전이 원본 exact 일치와 모델 finite를 확인했다. [검증](output_etc/check/stage2_20261105/checkpoint_verification.json), [CPU 로그](output_etc/check/stage2_20261105/cpu_tests.log). 사용자 지정 GPU5에서 64환경×6장면의 관계/alias·부분 reset·AMP/action/value/reward finite·가중치 보존을 확인했다. [물리 검사](output_etc/check/stage2_20261105/physics_gpu5/physics_validation.json). GPU5의 16환경×seed0 정식 평가 경로를 1200step 실행하고 CPU 재채점·성공 건수 exact 일치·finite를 확인했다. [재채점](output_etc/check/stage2_20261105/eval_gpu5/general_seed0/rescore.json). 이는 실행 검증이며 학습 성능 평가는 아니다.

### zerobg16k–20k ·1k 간격 정밀 평가 완료 ·17k 최종 추천

- 사용자 요청으로16k/17k/18k/19k/20k PTH/YAML을 복사 고정해 각4명·8물체 혼합512환경×seed0/1/2, 총15회를 GPU3/4/5에서 평가했다. 기존 스폰/distractor·ordered·loco·평균 액션·거리/물성·1200step을 유지하고10k–15k·19.5k의21회는 재사용했다. [전체 비교·최종 추천](output_etc/check/zerobg16k20k_eval512_gpu345_20261105/report.md), [선정 PTH](output_etc/check/zerobg16k20k_eval512_gpu345_20261105/selection.json), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 16k/17k/18k/19k/20k의 Ever 그룹 합산 Coupled SR은89.43±0.75/92.31±0.56/88.79±0.75/91.63±0.56/90.76±2.19%다. 17k의 단독 PUSH92.18±1.53%, 나머지4행동 평균94.40±1.18%, I/C/Mixed=93.08±0.72/92.31±0.56/92.70±0.62%다. 공동2/3/4명93.97±0.80/89.55±1.02/91.88±1.54%이며 공동 전체·공동4명·단독 PUSH·Mixed의 최고 평균을 근거로17k를 주 후보로 추천했다. 공동3명만의 최고는19.5k다. 학습/추론 기본 PTH나 학습 프로세스는 변경하지 않았다.
- 새15회 모두 CPU/GPU 재채점·finite·개인/그룹 건수·native gate·actor/critic zerobg/SELF·NONE0·가중치/PTH/YAML/epoch/frame·지면/거리 검증을 통과했다. 같은 seed의 초기12필드가 기존7개 모델과 exact일치했다. GPU당 평가 최대3개·최종0개이며 GPU0/1/2·기존 학습/MPS는 변경하지 않았다. [검증](output_etc/check/zerobg16k20k_eval512_gpu345_20261105/verification.json).

### 요청 시점 최신 zerobg19.5k · 정밀 평가 및15k 비교

- 사용자 요청 시점의 최신 정기 PTH와 일반 이름 PTH가 모두 epoch19500임을 확인했다. 정기19.5k(epoch19500/frame2555904000) PTH/YAML을 복사 고정해 GPU5에서 기존4명·8물체 혼합512환경×seed0/1/2를 평가했다. 스폰/distractor·ordered·loco·평균 액션·거리/물성·1200step을 유지하고10k–15k의18회를 재사용했다. [비교표](output_etc/check/zerobg19500_trend512_gpu5_20261105/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever 영역19.5k의 그룹 합산 Coupled SR91.40±0.80%, 단독 PUSH90.42±0.42%, 나머지4행동 동일 가중 평균94.04±1.53%다. 15k 대비 공동은2.99pp 높고 단독 PUSH/나머지는0.30/0.01pp 낮아 비슷한 수준이다. 공동2/3/4명91.76±0.66/92.70±1.89/88.28±2.29%로 개선은 주로2·3명에서 나왔다. 유형별 I/C/Mixed=91.96±1.34/91.40±0.80/91.68±0.88%이며 현재 비교에서 공동 전체와Mixed 평균이 가장 높다.
- 새3회 CPU/GPU 재채점·finite·개인/그룹 건수·native gate·actor/critic zerobg/SELF·NONE0·가중치/PTH/YAML/epoch/frame·지면/거리 검증을 통과했다. 같은 seed의 초기12필드가10k–15k와 exact일치했다. GPU5 평가 최대3개·최종0개이며 GPU0/1/2·기존 학습/MPS는 변경하지 않았다. [검증](output_etc/check/zerobg19500_trend512_gpu5_20261105/verification.json).

### zerobg15k · 정밀 평가 및10k–15k 비교

- 사용자 요청으로15k(epoch15000/frame1966080000) PTH/YAML을 복사 고정해 GPU5에서 기존4명·8물체 혼합512환경×seed0/1/2를 평가했다. 기존 스폰/distractor·ordered·loco·평균 액션·거리/물성·1200step을 유지하고10k–14k의15회를 재사용했다. [비교표](output_etc/check/zerobg15000_trend512_gpu5_20261105/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever 영역15k의 그룹 합산 Coupled SR88.41±1.32%, 단독 PUSH90.72±1.23%, 나머지4행동 동일 가중 평균94.05±1.25%로14k보다 세 항목 모두 높다. 공동2/3/4명88.79±3.22/88.00±3.66/88.45±3.32%다. 유형별 I/C/Mixed=92.26±0.51/88.41±1.32/90.34±0.43%이며 현재10k–15k 비교에서 공동 전체와Mixed 평균이 가장 높다.
- 새3회 CPU/GPU 재채점·finite·개인/그룹 건수·native gate·actor/critic zerobg/SELF·NONE0·가중치/PTH/YAML/epoch/frame·지면/거리 검증을 통과했다. 같은 seed의 초기12필드가10k–14k와 exact일치했다. GPU5 평가 최대3개·최종0개이며 GPU0/1/2·기존 학습/MPS는 변경하지 않았다. [검증](output_etc/check/zerobg15000_trend512_gpu5_20261105/verification.json).

## 2026-11-04

### zerobg14k · 정밀 평가 및10k–14k 비교

- 사용자 요청으로14k(epoch14000/frame1835008000) PTH/YAML을 복사 고정해 GPU5에서 기존4명·8물체 혼합512환경×seed0/1/2를 평가했다. 기존 스폰/distractor·ordered·loco·평균 액션·거리/물성·1200step을 유지하고10k–13k의12회를 재사용했다. [비교표](output_etc/check/zerobg14000_trend512_gpu5_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever 영역14k의 그룹 합산 Coupled SR87.87±1.89%, 단독 PUSH88.75±2.10%, 나머지4행동 동일 가중 평균93.18±1.41%다. 공동2/3/4명88.68±1.54/87.63±2.34/85.83±8.11%이며13k보다2명은 높고3/4명은 낮다. 유형별 I/C/Mixed=90.90±0.57/87.87±1.89/89.38±0.99%다. 13k 대비 공동 전체는 소폭 높고 단독은 소폭 낮으며4명 공동의 seed 편차가 크다.
- 새3회 CPU/GPU 재채점·finite·개인/그룹 건수·native gate·actor/critic zerobg/SELF·NONE0·가중치/PTH/YAML/epoch/frame·지면/거리 검증을 통과했다. 같은 seed의 초기12필드가10k–13k와 exact일치했다. GPU5 평가 최대3개·최종0개이며 GPU0/1/2·기존 학습/MPS는 변경하지 않았다. [검증](output_etc/check/zerobg14000_trend512_gpu5_20261104/verification.json).

### zerobg13k · 6/8명 all/keep3 공유 중심40cm 재집계

- 기존 네64환경 궤적의 공유 상자 중심→공유 목표3D 거리를 CPU float64로20/40cm 재채점했다. Ever20→40cm는6명 all21.88→26.56%, keep3 15.62→21.88%,8명 all25.00→32.81%, keep3 25.00→39.06%다. Final은 참고 파일에 보존했다. 추가 GPU 시뮬레이션·학습/환경/종료 조건 변경은 없다. [반경 비교](output_etc/check/zerobg13000_n68_all_keep3_env64_gpu35_20261104/radius_report.md).
- 기존20cm 성공 플래그/그룹 건수 exact일치·반경 포함 관계·Final→Ever·finite를 네 실행 모두 확인했다. 기존 그룹 유효성 규칙을 유지했다. [검증](output_etc/check/zerobg13000_n68_all_keep3_env64_gpu35_20261104/radius_verification.json), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).

### zerobg13k · 6/8명 전원 공동 all/keep3 ·64환경

- 사용자 요청으로 같은13k(epoch13000/frame1703936000)를 GPU3(6명)·5(8명)에서 각 all/keep3·64환경×seed0·ordered·loco·평균 액션·1200step으로 총4회 평가했다. 인원별 원래 상자 크기/질량·스폰/distractor를 유지한 전원 공동이며, 그룹 최대4명 혼합 확장과 별개다. keep3는 미접촉 우선3명을 유지하고 선택 밖 COUPLED를NONE으로 바꾸며 critic와 전체 그룹 gate는 유지한다. [결과](output_etc/check/zerobg13000_n68_all_keep3_env64_gpu35_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 공유 중심20cm Ever의 all→keep3는6명21.88%(14/64)→15.62%(10/64),8명25.00%(16/64)→25.00%(16/64)다. Final은 각각20.31→10.94%,21.88→14.06%로 참고 보존했다. 이번64환경1seed에서는 keep3의 Ever 개선이 없다.
- 네 실행 모두 CPU/GPU 재채점·finite·그룹/개인 건수·가중치/PTH/YAML/epoch/frame·native gate·actor/critic zerobg 설정 검증을 통과했다. all/keep3 초기12필드 exact일치, 매step 선택수/미접촉 우선 및 actor4레이어4800회의 COUPLED→NONE·padding없음을 확인했다. GPU당 평가 최대2개·최종0개이며 기존 학습/MPS·GPU0/1/2는 변경하지 않았다. [검증](output_etc/check/zerobg13000_n68_all_keep3_env64_gpu35_20261104/verification.json).

### zerobg13k · 정밀 평가 및10k–13k 비교

- 사용자 요청으로13k(epoch13000/frame1703936000) PTH/YAML을 복사 고정해 GPU5에서 기존4명·8물체 혼합512환경×seed0/1/2를 평가했다. 기존 스폰/distractor·ordered·loco·평균 액션·거리/물성·1200step을 유지하고10k·11k·12k의9회를 재사용했다. [비교표](output_etc/check/zerobg13000_trend512_gpu5_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever 영역13k의 그룹 합산 Coupled SR87.25±2.13%, 단독 PUSH88.93±1.83%, 나머지4행동 동일 가중 평균93.56±0.93%다. 공동2/3/4명86.29±1.68/89.52±0.89/86.51±5.35%이며10k보다 공동 평균은 높고 단독 PUSH·나머지 평균은 낮다. 유형별 I/C/Mixed=91.36±1.18/87.25±2.13/89.30±1.07%로10k Mixed89.15±0.81%와 비슷하다. 공동 우선 후보로13k를 추가하되4명 공동의 seed 편차를 함께 기록했다.
- 새3회 CPU/GPU 재채점·finite·개인/그룹 건수·native gate·actor/critic zerobg/SELF·NONE0·가중치/PTH/YAML/epoch/frame·지면/거리 검증을 통과했다. 같은 seed의 초기12필드가10k·11k·12k와 exact일치했다. GPU5 평가 최대3개·최종0개이며 GPU0/1/2·기존 학습/MPS는 변경하지 않았다. [검증](output_etc/check/zerobg13000_trend512_gpu5_20261104/verification.json).

### ground18k·19k·20k · 정밀 평가 및 zerobg10k 비교

- 사용자 요청으로 고정18k/19k/20k를 GPU3/4/5에서 각각512환경×seed0/1/2, 총9회 평가했다. 기존4명·8물체 혼합·ordered·loco·평균 액션·스폰/distractor·거리/물성·1200step 첫 episode를 유지하고 ground9.5k·16k·17k 및 zerobg10k 기존12회를 재사용했다. [비교표·판단](output_etc/check/ground18k19k20k_eval512_gpu345_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever 영역18k/19k/20k의 그룹 합산 Coupled SR은84.88±2.15/85.59±1.18/86.20±1.76%, 단독 PUSH81.62±1.93/77.50±4.26/79.94±0.87%, 나머지4행동 동일 가중 평균92.72±1.42/92.21±1.88/92.06±1.57%다. zerobg10k는 각각86.00±1.08/90.18±1.55/94.60±1.18%다. 공동4명은19k87.25±2.83%로 zerobg83.32±2.92%보다 높은 평균이지만, 공동 전체의 지속 개선과 단독 회복이 없어 현 시점 추가 ground 학습 중단을 추천했다. 비교용 PTH·학습 프로세스는 보존했다.
- 새9회 CPU/GPU 재채점·finite·개인/그룹 건수·native gate·ground 높이/거리·actor/critic 설정·가중치/PTH/YAML/epoch/frame 검증을 통과했다. 각 새 모델과 재사용4개 모델은 같은 seed의 초기12필드가 exact일치했다. GPU당 평가 최대3개·최종0개이며 GPU0/1/2·기존 학습/MPS는 변경하지 않았다. [검증](output_etc/check/ground18k19k20k_eval512_gpu345_20261104/verification.json).

### zerobg10k · 6/8/12/16명 혼합 확장 완료 · Stage1 유형별 SR 비교

- 고정 zerobg10k를 GPU3/4/5/6에서 인원6/8/12/16별512환경×seed0/1/2, 총12회 평가했다. 승인된5m 그룹 중심 배치·불필요한 distractor 물리/관측 제외·공동 최대4·ordered·loco·평균 액션·기존 거리/물성·1200step을 적용했다. 4명은 기존 스폰/distractor 결과를 재사용했다. Stage2 actor/critic coordination은 매1200step 정상 실행했다. [두 집계·모델 비교](output_etc/check/zerobg10k_group_gap5_mixed512_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 유형별 SR은 단독 개인 I와 공동 그룹 C의 성공 건수를 각각 합산하고 Mixed=(I+C)/2로 계산했다. zerobg10k의6/8/12/16명 I=89.04±0.82/87.25±0.28/78.59±1.49/69.25±0.29%, C=84.19±1.86/82.83±0.55/78.31±0.86/75.14±0.46%, Mixed=86.61±0.89/85.04±0.22/78.45±0.90/72.20±0.37%다. 기존 행동별·공동 인원별 동일 가중 평균도 별도 보존했다. Stage1 26k는 완료된4/6/8/12/16명 결과만 CPU 재집계했다. 16명 Stage1 I/C/Mixed는71.93±0.27/58.75±0.58/65.34±0.34%다.
- 새12회 모두 CPU 재채점/finite/건수·같은 인원/seed의 Stage1 초기12필드 exact일치·그룹 평행이동/거리·제거 토큰 영향 차단·가중치/PTH/YAML/epoch/frame 검증을 통과했다. 두 모델30개 seed 결과의 원본 개인/그룹 성공 건수를 별도로 대조했다. GPU당 평가 최대1개·최종0개·실패0회이며 GPU0/1/2·기존 학습/MPS는 변경하지 않았다. [검증](output_etc/check/zerobg10k_group_gap5_mixed512_20261104/verification.json), [건수 대조](output_etc/check/zerobg10k_group_gap5_mixed512_20261104/count_audit.json), [프로세스 감사](output_etc/check/zerobg10k_group_gap5_mixed512_20261104/process_audit.json).

### zerobg10k·11k·12k · 유형별 Interaction-Group SR 재집계

- 사용자 제안 정의로기존4명512환경×3seed의Ever영역결과를CPU재집계했다. I는단독전체개인건수, C는인원2·3·4명구분없이공동그룹건수로합산하고 E가없어 Mixed=(I+C)/2다. C성공은기존공유상자중심20cm를유지했다. 각seed에서비율을구한뒤3seed평균±표본표준편차를계산했다. 추가GPU실행과기존지표/결과변경은없다.
- 10k/11k/12k의I=92.29±0.76/91.14±1.25/91.53±1.14%, C=86.00±1.08/83.73±0.63/82.84±0.70%, Mixed=89.15±0.81/87.44±0.56/87.19±0.46%다. 기존CPU재채점통과와개인수2048/seed를확인했다. [정의·결과](output_etc/check/zerobg10k12k_interaction_group_macro_20261104/report.md).

### zerobg12k · 10k/11k/12k 정밀 비교

- 사용자 요청으로12k(epoch12000/frame1572864000) PTH/YAML을복사고정하여GPU5에서4명·8물체혼합512환경×seed0/1/2를완료했다. 기존스폰/distractor·ordered·loco·평균액션·물성·1200step을유지하고10k/11k의검증된6회를재사용했다. [비교표](output_etc/check/zerobg12000_trend512_gpu5_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever영역10k/11k/12k 공동2·3·4명동일가중평균85.70±0.94/82.93±1.58/82.39±1.42%, 단독5행동평균93.71±0.64/92.34±0.71/92.45±0.97%, 혼합50:50평균89.71±0.68/87.63±0.72/87.42±0.61%다. 12k단독PUSH91.46±0.87%는10k90.18±1.55%보다높고공동평균은낮았다.
- 새3회와재사용6회CPU재채점/finite/건수·초기12필드bitwise일치·actor/critic zerobg/SELF·NONE0·가중치불변·PTH/YAML해시/epoch/frame 검증을통과했다. GPU5평가peak3개·최종0개이며GPU0/1/2·기존학습/MPS는변경하지않았다. [검증](output_etc/check/zerobg12000_trend512_gpu5_20261104/verification.json).

### 4명 초과 그룹 스폰 명세 · Stage1 26k 확장 평가 완료

- 사용자 요청으로[4명 초과 그룹 스폰](markdowns/reference/group_spawn_over4.md)을작성했다. 4명이하기존유지/초과에만5m그룹중심배치·기존loco/그룹내상대위치보존·공동최대4·상자/목표거리유지·distractor물리/관측제외·ON_TOP필수받침유지와현재진단경로/검증범위를명시했다. structure에참조링크를추가했다.
- 같은조건으로Stage1 26k의6/8/12/16명각512환경×seed0/1/2, 총12회를GPU3/4/5/6에서완료했다. 원래Stage1 encoder/head/RMS·merge_shared=False를사용하고Stage2 coordination을실행하면오류를내도록차단했다. 배치재시도한도만512로늘리고사람/그룹/거리/충돌조건은유지했다. 인원별최대4/5/9/52회로모든환경이통과했다. [비교표](output_etc/check/stage1_26k_group_gap5_mixed512_20261104/report.md), [설정](output_etc/check/stage1_26k_group_gap5_mixed512_20261104/manifest.json).
- Ever영역 단독5행동평균85.14±1.04/85.00±0.56/80.19±0.18/73.76±0.67%,공동2/3/4명동일가중평균69.50±2.18/68.20±1.62/63.12±2.28/58.12±0.83%,혼합50:50평균77.32±1.35/76.60±1.09/71.66±1.20/65.94±0.71%다. 기존4명은원래스폰/distractor결과를재사용했다.
- 12회모두원래Stage1 actor출력exact일치·coordination우회·가중치/해시/epoch26000/frame3407872000·그룹내상대위치/자세/속도·제거토큰영향차단·거리범위·CPU재채점/finite/건수검증을통과했다. 초기설정검사거부와0.1mm좌표검사거부로그를보존하고reset직전한도적용/0.5mm검증으로수정했다. 실패환경제외나0%치환은없다. [검증](output_etc/check/stage1_26k_group_gap5_mixed512_20261104/verification.json). GPU0/1/2·기존학습/MPS는종료하지않았다.

### zerobg11k · 일반512환경×3회 완료

- 사용자 요청으로11k(epoch11000/frame1441792000)를복사고정하여GPU4에서512환경×seed0/1/2를완료했다. 기존4명·8물체혼합·ordered·loco·평균액션·원래스폰/distractor·1200step을유지했다. [비교표](output_etc/check/zerobg11000_trend512_gpu4_20261104/report.md), [검증](output_etc/check/zerobg11000_trend512_gpu4_20261104/verification.json).
- Ever영역10k→11k 단독5행동평균93.71±0.64→92.34±0.71%, 공동2/3/4명동일가중평균85.70±0.94→82.93±1.58%, 혼합50:50평균89.71±0.68→87.63±0.72%다. 11k 공동2/3/4명84.95/83.60/80.23%, 단독PUSH89.03%다. 새3회CPU재채점/finite/건수·초기12필드일치·actor/critic zerobg·가중치/해시/epoch/frame 검증을통과했다. GPU0/1·기존학습/뷰어는변경하지않았다.

### 5m 그룹 배치 · zerobg10k 혼합 viewer

- 사용자 요청으로진단폴더에 `view.sh`/`viewer.py`를추가했다. 4명이하는기존스폰/distractor를유지하고초과는기존초기위치에서그룹전체XY이동·그룹중심최소5m·공동최대4·불필요한distractor물리/관측제외를적용한다. ON_TOP받침과작업물체·ordered·loco·평균액션을유지하고기본10씬을연속재생한다. 카메라는사람/필수상자/목표만포함해parked distractor에의한과도한줌아웃을피한다. 표준코드는수정하지않았다. [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- GPU3/4에서4·8·16명각2씬×12step headless viewer경로를통과했다. 각24회finite action·가중치불변, 4명그대로/8·16명매씬그룹이동, 각씬제거토큰교란에대한encoder출력exact불변을확인했다. GUI실제렌더링/정식성공률은이번검증대상이아니다. [검증](output_etc/check/group_spawn_gap5_layout64_20261104/viewer_checks.json).

### 5m 그룹 원형 이동 · 8/16명 배치만 검증

- 사용자 요청으로GPU3/4에서각8/16명·64환경·seed0의배치를검증했다. 기존N명 사람원반경2.5m+loco XY와그룹내상대위치·자세·속도를유지하며그룹전원과kinematic body에같은XY이동을적용했다. 그룹중심간최소거리5m를유지하도록그룹수에따른원반경을계산하고자리를무작위배정했다. 기존혼합샘플링·공동최대4·작업거리/충돌조건을유지하고불필요한distractor만멀리옮겼다. 표준코드는변경하지않았다. [결과](output_etc/check/group_spawn_gap5_layout64_20261104/report.md).
- 8/16명모두64/64환경배치통과, 첫시도58/40환경·평균시도1.13/2.06회·최대4/26회이며생성/검증은11.1/15.4초였다. GPU검사와CPU재검증에서그룹내상대위치/자세/속도·body동시이동·5m간격·그룹상한·필수물체보존을확인했다. 정책rollout0step이며관측토큰제외/정책성공률은측정하지않았다. [검증](output_etc/check/group_spawn_gap5_layout64_20261104/verification.json), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).

### zerobg10k · 4명 distractor 제거512환경×3회

- 사용자 요청으로 GPU3(seed0/1)·4(seed2)에서 zerobg10k를4명·8물체 슬롯·512환경×3회·ordered·loco·평균 액션·1200step으로 평가했다. 기존 초기 장면을 생성한 뒤 불필요한 distractor를 물리 장면에서 멀리 옮기고 관측에서 제외했다. ON_TOP 받침과 실제 작업 상자는 유지했으며 사람 스폰·표준 코드는 변경하지 않았다. [비교표](output_etc/check/zerobg10k_nodistractors512_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever 영역 단독5행동 평균93.71±0.64→92.91±1.51%, 공동2·3·4명 동일 가중 평균85.70±0.94→82.77±0.89%, 혼합50:50 평균89.71±0.68→87.84±0.52%다. 공동2/3/4명85.73/83.88/78.68%, 단독PUSH90.93%다. 물리 장애물과 관측 토큰을 동시에 제거한 비교이며, 제거 후 새로 샘플링한 장면의 성능과 구분한다.
- 세seed의 초기12필드(상자는 제거 전 전체·제거 후 필수 물체) 일치·필수 물체와 관측 보존·제거 토큰 교란에 대한 encoder 출력 불변·CPU재채점/finite/건수·가중치 불변·PTH/YAML해시와epoch10000/frame1310720000 검증을 통과했다. GPU0/1/6·기존 학습·MPS는 종료하지 않았다. [검증](output_etc/check/zerobg10k_nodistractors512_20261104/verification.json).

### zerobg10k · 12명 확대·인접 스폰512×3 완료 /16명3seed 배치 실패

- 사용자 승인으로12/16명 사람 반경을2.5×sqrt(N/4)=4.33/5m로 넓히고 같은 공동 그룹을 원 위 연속 슬롯에 배치했다. 실제rootXY를슬롯에고정하고kinematic body를같이옮기며 RSI자세/높이/속도·그룹/행동/RNG를유지한다. 공동은2~min(4,남은 인원), 물체2N개·ordered·loco·평균 액션·1200step·원래 작업거리/충돌/물성·distractor반경4.5m를 유지했다. 표준학습/샘플러 코드는 변경하지 않았다. [최종 비교표](output_etc/check/zerobg10k_mixed_cap4_anchored_grouped512_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- GPU4에서12명512환경×seed0/1/2가각112/117/31회배치후전부통과하여평가를완료했다. Ever영역 단독5행동평균85.96±0.82%,공동2/3/4명평균81.69±0.61%,혼합50:50평균83.82±0.72%다. 공동2/3/4명82.79±0.67/80.51±0.77/81.77±2.73%,단독SIT/CLIMB/AT/ON_TOP/PUSH96.83/97.43/81.88/67.83/85.83%다. 기존4/6/8명9회결과를재사용하지만해당인원은기존반경2.5m·기존스폰이므로인원수만바꾼비교로해석하지않는다.
- 16명seed0/1/2는GPU3/2/5에서각512회한도까지설정변경없이시도했으나2/3/1환경배치미통과로모두중단됐다. 세seed모두rollout성공률은없다. 총6환경중5환경은단독PUSH후보배치가남았고1환경은최종장면검사가남았다. 배치실패를정책성공률0%로넣거나미통과환경을제외하지않았다. [실패 좌표·미배치그룹](output_etc/check/zerobg10k_mixed_cap4_anchored_grouped512_20261104/layout_failures.json).
- 완료3회의CPU재채점/건수/finite·actor/critic zerobg/SELF·NONE0·countnorm없음·가중치불변·PTH해시/epoch10000/frame1310720000·AT/ON_TOP거리1~5m·그룹최대4 검증을통과했다. 스폰슬롯순열/연속·root/body동시XY이동·자세/속도/높이/RNG불변을검증했다. PhysX반영후rootXY는1e-4m허용오차로검증하며나머지root성분은exact다. 실패3회는스폰/실패건수만검증하며정책재채점은없다. 최종신규평가프로세스0개이며GPU0/1/6·본학습·MPS를종료하지않았다. [검증](output_etc/check/zerobg10k_mixed_cap4_anchored_grouped512_20261104/verification.json).
- 중간시도는보존했다. [반경만확대한시험](output_etc/check/zerobg10k_mixed_cap4_spawnscale512_20261104/report.md)은147/93회후145/462환경정체로중단했다. [nominal슬롯시험](output_etc/check/zerobg10k_mixed_cap4_groupedspawn512_20261104/stop_reason.json)은RSI의기존XY변위가남아실제인접스폰과달라져최종방식으로수정했다. 두중간시도의정책성능수치는없다.

### Stage1 PUSH26k · 현재4명 혼합512환경×3회 비교

- 사용자 요청으로 원본26k(epoch26000/frame3407872000)를 복사 고정해 GPU2에서512환경×seed0/1/2를 평가했다. zerobg10k의4명/8물체·기존 단독/공동 혼합·ordered·loco·지면/현재 거리·물성/distractor·1200step 첫 episode를 사용했다. Stage1 encoder/action head/정규화·merge_shared=False를 복원하고 Stage2 coordination은 실행하지 않았다. [비교표](output_etc/check/stage1_26k_current_mixed4_512_gpu2_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever 영역 단독5행동 평균/공동2·3·4명 동일 가중 평균/혼합50:50 평균은 Stage1 88.52±1.81/71.64±0.59/80.08±0.62%, zerobg10k 93.71±0.64/85.70±0.94/89.71±0.68%다. 단독PUSH는95.86±0.57→90.18±1.55%, AT74.28±6.55→89.31±3.48%, ON_TOP영역80.81±3.87→90.76±2.96%다. 원래Stage1 환경/900step 평가와 구분한다.
- 세seed 모두 기존zerobg 초기12필드 bitwise일치, 첫 실제 관측에서 원래Stage1 actor와 평균/표준편차 출력 exact일치, 원본가중치 복사 exact일치·rollout가중치 불변·CPU재채점/건수/finite·PTH와환경해시 검증을 통과했다. 표준 모델·학습 코드는 변경하지 않았다. [검증](output_etc/check/stage1_26k_current_mixed4_512_gpu2_20261104/verification.json).

### zerobg10k · 혼합 인원 확장 · 6/8명 완료,12/16명 배치 실패

- 전체 환경 인원6/8/12/16명·물체2N개·각512환경×seed0/1/2를 GPU2/5/4/3에서 시작했다. 기존 혼합 샘플링을 유지하고 공동 그룹 크기만2~min(4,남은 인원)으로 제한했다. CPU검사에서4명이하 원본 그룹/RNG 일치와 확장 그룹 최대4명을 확인했다. 8/12/16명은 기본배치64회 소진으로 첫seed가 실패하여, 원래 사람/그룹·거리/충돌조건을 유지한 채 진단 harness의재시도한도만512회로 늘렸다. 실패로그와 검증 단계 설정거부로그를 보존했다. [진행/비교표](output_etc/check/zerobg10k_mixed_cap4_scaling512_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 6/8명 각3회는 CPU재채점/건수/finite·가중치불변·그룹상한 검증을 통과했다. Ever영역 단독/공동동일가중/혼합50:50은6명90.63±0.33/80.00±1.80/85.31±0.75%,8명89.85±0.44/77.03±1.90/83.44±0.77%다. 12명은512회에서도9환경 배치 실패로 중단됐고16명도512회 배치 소진으로 중단되어 두 인원의 성능수치는 없다. 실패환경을0% 처리하거나 제외하지 않았다. 실행 중인 본학습·표준 샘플러는 변경하지 않았다. [현재 상태](output_etc/check/zerobg10k_mixed_cap4_scaling512_20261104/status.json).
- 실패 원인 분해용 `diagnose_layout.py`를 추가하고 GPU4/3에서12/16명·512환경·seed0·8회 배치만 진단한다. 각회 충돌 판정은 원래 함수와 exact일치 검사를 수행한다. 첫회 실패는12명359/512,16명497/512이며 상자-사람 근접은각220/452환경이었다(실패항목 중복). 스폰 반경 확대는 아직 적용하지 않았다. [12명 진단](output_etc/check/zerobg10k_mixed_cap4_scaling512_20261104/layout_diagnostic_n12/layout_diagnostic.json), [16명 진단](output_etc/check/zerobg10k_mixed_cap4_scaling512_20261104/layout_diagnostic_n16/layout_diagnostic.json).

### ground/zerobg10k · 최초 전원 접촉 전후 이동 CPU 재집계

- 사용자 승인 기준으로 기존 두 모델의512환경×seed0/1/2 trace/gate_trace/initial을 CPU로 재집계했다. 추가 GPU 시뮬레이션은 없다. 최초 전원 on_face·유효 접촉 이전 초기 공유 상자 중심 대비XY변위≥50cm를 조기밀기로 정의했다. 전원 접촉 미달성은 전체1200step을 검사하고, 최초접촉이후 접촉이탈은 조기밀기로 세지 않는다. native gate의목표도착예외와분리했다. [비교표·정의](output_etc/check/ground_zerobg10k_gate_timing_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 모델별1456그룹(2명778·3명421·4명257) 합산 조기50cm 비율 ground43.61%(635)/zero46.02%(670), 전원접촉달성94.16/95.12%, 접촉달성그룹중추가50cm이동98.69/99.21%다. 2/3/4명 조기밀기는 ground31.62/51.07/67.70%,zero33.68/53.68/70.82%다. 접촉 전 최대변위 중앙값0.430/0.465m,90백분위1.511/1.354m다. 조기50cm없이접촉후추가이동까지달성55.84/53.64%다.
- 접촉 후 추가50cm이동에는 이미 움직이던 장면이 포함되며, 소요시간은 반응/출발 지연이 아니다. 조기50cm 없음도 완전정지율로 해석하지 않는다. 초기12필드일치·공동그룹분모·물리공유중심일치·전원접촉→nativegate·낙상재활성없음·원본재채점검증과입력해시를 확인했다. 표준 코드·학습·원본 결과는 변경하지 않았다. [검증](output_etc/check/ground_zerobg10k_gate_timing_20261104/verification.json).

### zerobg10k · 일반512환경×3회 · 공동/단독 동반 개선

- 사용자 요청으로 zerobg10000epoch/frame1310720000 PTH/YAML을 복사 고정해 GPU4에서각512환경×seed0/1/2 총3회를 평가했다. 4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode이며 zerobg8k/9k 및 ground10k 기존9회를 재사용했다. [비교표](output_etc/check/zerobg10000_trend512_gpu4_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- zerobg9k→10k 공유 중심20cm 공동 전체 Ever81.14→86.00%, 단독 PUSH87.19→90.18%, 나머지4행동 영역 평균91.88→94.60%로 세 항목이 함께 개선됐다. 공동2/3/4명86.06/87.72/83.32%이며 동일10k ground82.86/81.31/81.02%보다 높다. Final은공동84.07%,PUSH88.00%,나머지영역87.60%다.
- 새3회와재사용9회 CPU재채점·JSON건수·finite·native gate·actor/critic zerobg/SELF·NONE0·가중치 불변·PTH/YAML해시·epoch/frame·지면 높이/거리 검증을 통과했다. seed별초기12필드가모든비교PTH와bitwise일치했다. GPU4peak3개·최종0개이며GPU0·1·본학습·표준 코드는 건드리지 않았다. [검증](output_etc/check/zerobg10000_trend512_gpu4_20261104/verification.json).

### zerobg9k · 6/8명 all/keep3 · 20/40cm

- 사용자 요청으로 zerobg9k를 복사 고정하고 GPU4(all)·5(keep3)에서6명/12물체·8명/16물체 각64환경×seed0, 총4회를 평가했다. ordered·loco·평균 액션·원래 인원별 상자 크기/무게·distractor·1200step 첫 episode를 유지했다. 같은 궤적을 공유 상자 중심→공유 목표3D20/40cm로 재집계했다. [반경 비교표](output_etc/check/zerobg9000_n68_all_keep3_radius_env64_gpu45_20261104/radius_report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever20→40cm는6명 all12.50→18.75%,keep3 21.88→26.56%;8명 all34.38→37.50%,keep3 31.25→37.50%다. Final은6명 all12.50→17.19%,keep3 12.50→20.31%;8명 all26.56→34.38%,keep3 21.88→35.94%다.64환경1seed에서keep3의Ever개선은6명에만 나타났다.
- CPU재채점·finite·가중치 불변·PTH/YAML해시·epoch/frame·초기12필드 일치·native gate·매행동keep3선택/actor4층NONE전환/critic무개입 검증을 통과했다.20cm저장 성공플래그/건수 정확 일치·20cm⊆40cm·Final⊆Ever도 확인했다. GPU별평가peak1개·최종0개이며GPU0·1·본학습·표준 성공조건은 변경하지 않았다. [일반 검증](output_etc/check/zerobg9000_n68_all_keep3_radius_env64_gpu45_20261104/verification.json), [반경 검증](output_etc/check/zerobg9000_n68_all_keep3_radius_env64_gpu45_20261104/radius_verification.json).

### ground17k · 일반512환경×3회 · 후반 추세

- 요청 시점 새 정수 epoch ground17k를 GPU3에서각512환경×seed0/1/2로 평가했다. zerobg는9.5k까지 저장되어 사용자1k간격에 따라 이번 신규 대상에서 제외했다. 4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode를 유지하고 ground14k/16k·zerobg8k/9k 기존12회 결과를 재사용했다. [비교표](output_etc/check/ground17000_trend512_gpu3_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- ground16k→17k 공유 중심20cm 공동 전체 Ever85.85→86.68%, 단독 PUSH80.74→81.11%, 나머지4행동 영역 평균92.99→90.28%다.17k 공동2/3/4명86.89/85.55/88.02%이며 AT80.98%,ON_TOP영역87.12%로16k의86.83/90.85%보다낮다. Final은공동84.14%,PUSH78.56%,나머지영역83.85%다. 공동/PUSH 소폭 개선과 나머지 단독 하락이 함께 나타났다.
- 새3회 및 재사용12회 CPU재채점·finite·native gate·가중치 보존·원본/고정PTH 및 YAML 해시·epoch/frame·지면 높이/거리 검증을 통과했다. seed별 초기12필드가모든비교모델과bitwise일치했다. GPU3peak3개·최종0개이며GPU0·1·GPU6 본학습 및 표준 코드는 건드리지 않았다. [검증](output_etc/check/ground17000_trend512_gpu3_20261104/verification.json).

### zerobg8k·9k · 일반512환경×3회 · ground 동일 epoch 비교

- 사용자 확인으로8k·9k 두 PTH만 각512환경×seed0/1/2, 총6회를 새로 평가했다. GPU4의worker0/1/2는8k, GPU3의worker3/4/5는9k이며 GPU당peak3개·최종0개다. 4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode를 유지했다. 기존ground8k/9k의6회는 재사용했다. [비교표](output_etc/check/zerobg_trend512_gpu34_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- zerobg8k→9k 공유 중심20cm 공동 전체 Ever84.20→81.14%, 단독 PUSH82.44→87.19%, 나머지4행동 영역 평균92.02→91.88%다. 동일 epoch ground는 공동82.67/85.31%, PUSH76.48/79.26%, 나머지영역91.65/91.61%다. 공동Final은zero82.08/78.38%, ground78.98/83.39%다. 앞선64환경 진단과 달리512×3에서는 zero의 나머지 영역 평균 우위가 작고 단독 PUSH가 높다.
- CPU 재채점·JSON건수·finite·native gate·actor/critic zerobg/SELF·NONE0·가중치 불변·원본/고정PTH 및 YAML 해시·epoch/frame·지면 높이/거리 검증을 통과했다. 각seed 초기12필드가두zero PTH 및기존ground8k/9k와bitwise일치했다. [검증](output_etc/check/zerobg_trend512_gpu34_20261104/verification.json). GPU0·1·GPU6 본학습과 표준 코드는 변경하지 않았다.

### learnedbg5k·zerobg8k · 6/8명 all/keep3

- 사용자 요청으로 각 PTH를 복사 고정하고 6명/12물체·8명/16물체 × all/keep3, 각64환경×seed0 총8회를 GPU3·5에서 평가했다. ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode를 유지했고 GPU0·1과 본학습은 건드리지 않았다. [통합 비교표](output_etc/check/zerobg8000_n68_all_keep3_env64_gpu35_20261104/comparison.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 공유 중심20cm Ever의 all→keep3는 learnedbg5k에서 6명9.38→23.44%, 8명17.19→17.19%; zerobg8k에서 6명23.44→23.44%, 8명28.12→28.12%다. Final은 각각 6명9.38→20.31/18.75→15.62%, 8명12.50→14.06/20.31→23.44%다. 이번 표본에서 keep3의 Ever 이득은 learnedbg6명에만 나타났다. epoch가 다른 모델·64환경1seed 비교다.
- 8회 CPU 재채점·finite·가중치 보존·PTH/YAML 해시·epoch/frame·초기12필드 일치·native gate 검증을 통과했다. 매 행동의 keep3 선택수/미접촉 우선/SELF 제외 및 actor4층 COUPLED→NONE(마스킹 없음)·critic 무개입 검사를 통과했다. 두 평가 묶음 모두 종료했다. [learnedbg 검증](output_etc/check/learnedbg5000_n68_all_keep3_env64_gpu35_20261104/verification.json), [zerobg 검증](output_etc/check/zerobg8000_n68_all_keep3_env64_gpu35_20261104/verification.json).

### 선택 PTH 6/8명 · all/keep3 NONE 전환 비교

- 일반 평가 후 공동 최고 ground9.5k·회복/단독 균형 ground16k·fusion 공동 최고 zerobg3k를 선택해 각각6명/12물체·8명/16물체64환경×seed0·ordered·loco·평균 액션·원래 인원별 상자 크기/무게·distractor·1200step 첫 episode로 all/keep3 총12회를 평가했다. GPU3/2/5에서각모델을실행했고GPU0·1은사용하지않았다. ground5k의기존2회는재사용했다. [Ever/Final·접촉·이동·분모](output_etc/check/ground_relfusion_selected_n68_all_keep3_env64_gpu35_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 공유 중심20cm Ever의all→keep3는 ground9.5k에서6명7.81→25.00%,8명15.62→14.06%;ground16k에서6명10.94→17.19%,8명29.69→28.12%;zerobg3k에서6명14.06→12.50%,8명17.19→10.94%다. Final은각각6명6.25→23.44/10.94→15.62/12.50→10.94%,8명15.62→10.94/21.88→21.88/9.38→7.81%다. ground6명에서는개선됐지만8명은세후보모두Ever개선이없다.64환경1seed진단이다.
- keep3는미접촉상대를우선해3명을유지하며선택밖COUPLED를actor4층에서NONE으로전환한다. 매행동의선택수·미접촉우선·SELF제외·실제4층COUPLED/NONE수·attention padding 없음·critic개입없음4800회검사와저장선택trace검증을통과했다. 실제그룹·보상gate·물성은유지했다.
- 새12회와재사용2회CPU/GPU재채점·JSON건수·finite·native gate·가중치불변·설정·PTH/YAML해시·epoch/frame을확인했다. 같은인원의초기12필드가모든후보/all/keep3에서bitwise일치했다. GPU별peak2개·최종0개다. [검증](output_etc/check/ground_relfusion_selected_n68_all_keep3_env64_gpu35_20261104/verification.json). 폴더명의gpu35와달리실제GPU는2·3·5이며manifest/검증에기록했다. 본학습·표준코드는변경하지않았다.

### ground12k·14k·16k / zerobg8k·learnedbg5k까지 일반 추세

- 사용자 요청으로 GPU0·1을 제외하고 ground12k/14k/16k를 GPU2·3에서각512환경×seed0/1/2, zerobg4k/5k/6k/8k와 learnedbg3k/4k/5k를 GPU5에서각64환경×seed0로 평가했다.10개 정기PTH/YAML을 복사 고정했고4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode를 유지했다. 기존ground27회·fusion12회를 재사용했다. [ground 추세](output_etc/check/ground_12k14k16k_eval512_gpu23_20261104/report.md), [fusion 추세](output_etc/check/relfusion_ab_3k8k_eval64_gpu5_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- ground12k/14k/16k 공유 중심20cm 전체 Ever84.08/82.08/85.85%,단독 점 평균86.62/86.23/87.66%,PUSH76.47/76.44/80.74%다.16k는9.5k의공동86.96%에는못미치지만단독점평균85.07%보다높아균형후보로추가했다. 공동전체 Final82.64/80.36/83.45%다.
- zerobg4k/5k/6k/8k 공동전체 Ever78.125/68.75/79.6875/81.25%,learnedbg3k/4k/5k79.6875/62.5/62.5%다. zerobg3k의기존57/64(89.0625%)가fusion내공동최고다.64환경1seed로구조 우열이나512×3 ground와직접우열은판단하지않는다.
- 새16회의CPU/GPU재채점·JSON건수·finite·native gate·actor/critic모드·가중치불변·PTH/YAML해시·epoch/frame·ground높이/거리검증을통과했다. 평가규모별초기12필드가모든비교저장본에서bitwise일치했다.일반평가GPU2/3각peak3개·GPU5peak2개,모두종료했다. [ground 검증](output_etc/check/ground_12k14k16k_eval512_gpu23_20261104/verification.json), [fusion 검증](output_etc/check/relfusion_ab_3k8k_eval64_gpu5_20261104/verification.json). 본학습·표준코드는변경하지않았다.

### ground11k·11.5k / fusion A/B 최신 추세 · GPU0·1 제외

- 사용자 요청으로 GPU0·1을 제외하고 GPU2에서ground11k/frame1441792000·11.5k/frame1507328000을각512환경×seed0/1/2, GPU5에서zerobg2.5k/3k/3.5k·learnedbg2k/2.5k를각64환경×seed0로 새로 평가했다.7개 정기PTH/YAML을 복사 고정했고4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode를 유지했다. 기존ground21회·fusion7회를 재사용했다. [ground 전체 추세](output_etc/check/ground_11k11500_eval512_gpu2_20261104/report.md), [fusion 전체 추세](output_etc/check/relfusion_ab_2k3500_eval64_gpu5_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- ground11k/11.5k 공유 중심20cm 전체 Ever82.16/83.54%,4명81.79/83.58%,단독 점 평균85.29/83.88%,PUSH67.66/73.09%,AT84.89/85.58%다. 공동 전체 Final80.30/82.38%다.9.5k의공동86.96%·PUSH81.04%보다낮으며 AT는회복했다.
- zerobg2.5k/3k/3.5k 공동 전체 Ever78.1/89.1/75.0%,단독 점 평균88.4/83.8/84.7%,PUSH82.1/71.4/75.0%다. learnedbg2k/2.5k 공동전체65.6/65.6%,단독점평균82.4/83.8%,PUSH57.1/71.4%다.같은2.5k에서zerobg가공동·단독평균·PUSH에앞섰다. zerobg3k공동은57/64이며소표본1seed로512×3 ground와직접우열을판단하지않는다. 공동전체 Final은zero78.1/84.4/73.4%,learn60.9/65.6%다.
- 새11회의CPU/GPU재채점·JSON건수·finite·native gate·실제actor/critic모드·가중치불변·PTH/YAML해시·epoch/frame·ground높이/거리검증을통과했다. 기존결과검증·해시와평가규모별초기12필드bitwise일치도확인했다. GPU2peak3개·GPU5peak2개,최종평가0개다. [ground 검증](output_etc/check/ground_11k11500_eval512_gpu2_20261104/verification.json), [fusion 검증](output_etc/check/relfusion_ab_2k3500_eval64_gpu5_20261104/verification.json). GPU0·1·본학습·표준코드는변경하지않았다.

## 2026-11-03

### ground8k·10k · 512환경×3seed · 5k–10k 비교

- 사용자 요청으로 ground8000epoch/frame1048576000·10000epoch/frame1310720000 PTH/YAML을 복사 고정해 GPU3에서각512환경×seed0/1/2를 평가했다.4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode이며 worker3개가 각8k→10k를 순차 실행했다. 기존5k/6k/7k/9k/9.5k의15회는 재사용했다. [Ever/Final·편차·분모](output_etc/check/ground_8k10k_eval512_gpu3_20261103/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 공동 공유 중심20cm 전체 Ever는5k→6k→7k→8k→9k→9.5k→10k에서77.99→81.78→83.42→82.67→85.31→86.96→82.03%다.8k의2/3/4명85.16/80.40/79.05%,10k82.86/81.31/81.02%다.10k는9.5k보다 세seed 모두 공동 전체가 낮고,현재 평가 저장본 중9.5k가 공동 전체·각 인원과 단독 PUSH에서 가장 높다.
- 8k/10k 단독 점 평균86.04/84.46%,AT80.49/77.98%,ON_TOP85.86/88.11 및80.50/86.92%,PUSH76.48/79.82%다. 공동 전체 Final78.98/80.37%다.10k는9.5k보다AT·ON_TOP 점·PUSH·단독 점 평균도 낮지만SIT 점은 회복했다.
- 새6회의 CPU/GPU 재채점·JSON 건수·finite·native gate·기본attention/fusion 없음·가중치 불변·PTH/YAML 해시·epoch/frame·ground 높이/거리 검증을 통과했다. 기존15회 검증 기록과해시를 확인하고 각seed 초기12필드가 모든epoch에서bitwise 일치함을 확인했다. 평가peak3개·최종0개다. [검증](output_etc/check/ground_8k10k_eval512_gpu3_20261103/verification.json). 본학습·표준 코드·GPU0은 변경하지 않았다.

### learnedbg1.5k · zerobg 동일 epoch 추세 비교

- learnedbg1500epoch/frame196608000 PTH/YAML을 복사 고정해 GPU3에서4명·8물체 혼합64환경×seed0·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode1회를 평가했다. 이전 learnedbg0.5k/1k와 zerobg0.5k/1k/1.5k/2k 여섯 결과는 재사용했다. [Ever/Final·분모·추세](output_etc/check/relfusion_learnedbg1500_ab_eval64_gpu3_20261103/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- learnedbg1k→1.5k 공유 중심20cm 전체 Ever73.4→64.1%(41/64),2/3/4명83.3/46.2/66.7→71.4/61.5/33.3%,단독 점 평균79.9→79.2%,PUSH53.6→67.9%,AT82.8→79.3%,ON_TOP83.3/83.3→88.9/94.4%다. 같은1.5k zerobg는 공동 전체76.6%,단독 점 평균85.0%,PUSH64.3%,ON_TOP77.8/77.8%다. learnedbg1.5k 공동 전체 Final62.5%,ON_TOP77.8/88.9%다. 이번 소표본에서는 zerobg가 공동 전체와 단독 평균에서 앞서며 learnedbg는ON_TOP이 강하지만 구조 우열을 확정하지 않는다.
- CPU/GPU 재채점·JSON 건수·finite·native gate·actor/critic learnedbg 구성·가중치 불변·PTH/YAML 해시·epoch/frame·ground 높이/거리와 일곱 저장본 초기12필드bitwise 일치를 확인했다. [검증](output_etc/check/relfusion_learnedbg1500_ab_eval64_gpu3_20261103/verification.json). 본학습·표준 코드·GPU0은 변경하지 않았다.

### zerobg2k · 일반64환경 추세 추가

- zerobg2000epoch/frame262144000 PTH/YAML을 복사 고정해 GPU3에서4명·8물체 혼합64환경×seed0·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode1회를 평가했다. 이전0.5k/1k/1.5k 결과는 재사용했다. [Ever/Final·분모](output_etc/check/relfusion_zerobg2000_eval64_gpu3_20261103/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 1.5k→2k 공유 중심20cm 전체 Ever76.6→76.6%(49/64),2명81.0→85.7%,3명76.9→61.5%,4명55.6→55.6%다. 단독 AT93.1→96.6%,ON_TOP77.8/77.8→83.3/94.4%,PUSH64.3→67.9%,점 평균85.0→85.9%다.2k 공동 전체 Final76.6%,ON_TOP50.0/66.7%,PUSH64.3%다. 공동3명13그룹·4명9그룹 소표본이다.
- CPU/GPU 재채점·JSON 건수·finite·native gate·actor/critic zerobg 및 SELF/NONE0·가중치 불변·PTH/YAML 해시·epoch/frame·ground 높이/거리와 이전 세 저장본 대비 초기12필드bitwise 일치를 확인했다. [검증](output_etc/check/relfusion_zerobg2000_eval64_gpu3_20261103/verification.json). 본학습·표준 코드·GPU0은 변경하지 않았다.

### 최신 ground9.5k · zerobg1.5k · learnedbg1k 평가

- 요청 시점의 최신 정기 PTH/YAML을 복사 고정해 ground9500은 GPU3에서512환경×seed0/1/2, zerobg1500·learnedbg1000은 GPU2에서각64환경×seed0로 평가했다. 모두4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode다. 이전 ground12회와 fusion3회는 재사용했다. [ground 추세](output_etc/check/ground_latest9500_eval512_gpu3_20261103/report.md), [fusion 추세](output_etc/check/relfusion_ab_trend64_1000_1500_20261103/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- ground9k→9.5k 공유 중심20cm 전체 Ever85.31→86.96%,4명84.45→88.02%,단독 PUSH79.26→81.04%,단독 점 평균85.90→85.07%다. AT83.07→80.98%,ON_TOP84.63/86.71→82.62/85.32%로 내려갔다. 공동 전체 Final83.39→84.82%다.
- zerobg1k→1.5k 공동 전체 Ever59.4→76.6%,2/3/4명81.0/76.9/55.6%,단독 점 평균78.9→85.0%,PUSH71.4→64.3%다. learnedbg0.5k→1k 공동 전체64.1→73.4%,2/3/4명83.3/46.2/66.7%,단독 점 평균75.4→79.9%,PUSH64.3→53.6%다. 최신 공동 전체 Final75.0/73.4%다. fusion4명은9그룹 소표본이며512×3 ground와 직접 구조 우열을 판단하지 않는다.
- 새5회 CPU/GPU 재채점·JSON 건수·finite·native gate·actor/critic 모드·가중치 불변·PTH/YAML 해시·epoch/frame 및 ground 높이/거리 검증을 통과했다. 각 평가 규모 안에서 이전 결과와 초기12필드가bitwise 일치했다. [ground 검증](output_etc/check/ground_latest9500_eval512_gpu3_20261103/verification.json), [fusion 검증](output_etc/check/relfusion_ab_trend64_1000_1500_20261103/verification.json). 본학습·표준 코드·GPU0은 변경하지 않았다.

### ground 최신9k · 512환경×3seed · 5k/6k/7k 추세 확장

- 사용자 요청 시점 최신 정기 ground9000epoch/frame1179648000 PTH/YAML을 복사 고정하고 GPU3에서4명·8물체 혼합512환경×seed0/1/2·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode3회를 새로 평가했다. 기존5k·6k·7k각3회는 재사용했다. [Ever/Final·편차·분모](output_etc/check/ground_latest9k_eval512_gpu3_20261103/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 5k→6k→7k→9k 공유 중심20cm 전체 Ever77.99→81.78→83.42→85.31%,4명74.48→81.82→83.03→84.45%,단독 PUSH72.27→75.57→73.35→79.26%,단독 점 평균86.34→86.55→84.06→85.90%다.9k 공동2/3명85.64/85.31%,AT83.07%,ON_TOP84.63/86.71%,SIT85.81/98.27%이며 공동 Final전체83.39%,전원 개인 목표 동시 Ever전체81.75%다. 공동은 개선 추세,단독 PUSH/SIT는7k대비 회복했지만AT는5k이후 하락했다.
- 새3회 CPU/GPU 재채점·JSON 건수·finite·native gate·기본attention/fusion 없음·가중치 불변·원본/고정PTH와 YAML 해시·epoch/frame을 검증했다. 기존9회 검증 기록과PTH/YAML 해시를 확인했고 각seed 초기12필드는5k/6k/7k/9k 모두bitwise 일치했다. ground 운반 높이/거리도 통과했고 평가peak3개·최종0개다. [검증](output_etc/check/ground_latest9k_eval512_gpu3_20261103/verification.json). GPU0·본학습·표준 코드는 변경하지 않았다.

### 관계 fusion · zerobg1k / learnedbg0.5k · 일반64환경

- 사용자 요청으로 zerobg1000epoch/frame131072000·learnedbg500epoch/frame65536000 PTH/YAML을 복사 고정하고 GPU3에서4명·8물체 혼합64환경×seed0·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode2회를 평가했다. 기존 zerobg500 결과를 재사용했다. 세 모델 모두ground이고 초기12필드가bitwise 일치했다. [Ever/Final·분모](output_etc/check/relfusion_ab_trend64_500_1000_20261103/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- zerobg0.5k→1k 공유 중심20cm 전체 Ever59.4→59.4%,2/3/4명66.7/53.8/33.3→64.3/53.8/44.4%,단독 점 평균74.0→78.9%,PUSH75.0→71.4%다. learnedbg0.5k는 공동2/3/4명73.8/69.2/11.1%,전체64.1%,단독 점 평균75.4%,PUSH64.3%다. 공동4명은9그룹 소표본이며 구조 우열을 확정하지 않는다.
- 새2회 CPU/GPU 재채점·JSON 건수·finite·native gate·실제actor/critic fusion 및 A/B 배경 bias 구성·가중치 불변·원본/고정PTH와 YAML 해시·epoch/frame을 확인했다. ground 운반 높이/거리와 재사용500 결과/고정PTH 해시도 검증했다. [검증](output_etc/check/relfusion_ab_trend64_500_1000_20261103/verification.json). 본학습·표준 코드·GPU0은 변경하지 않았다.

### zerobg500 · 일반64환경 · 기존500 비교

- 사용자 요청으로 ground 관계 fusion B의500epoch/frame65536000 PTH와 YAML을 복사 고정하고 GPU2에서4명·8물체 혼합64환경×seed0·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode1회를 평가했다. 비교용 기존 solo5·CO norm·CO+NONE norm500의 같은 규모 결과3개는 재측정 없이 재사용했다. [Ever/Final·분모·비교](output_etc/check/relfusion_zerobg500_eval64_gpu2_20261103/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- zerobg 공유 중심20cm Ever2/3/4명은28/42·7/13·3/9(66.7/53.8/33.3%),전체38/64(59.4%)이고 Final도 같다. 단독 Ever SIT40/80·CLIMB100/100·AT82.8·ON_TOP72.2/77.8·PUSH75.0%,점 평균74.0%다. 기존 solo5/CO/CO+NONE의 공동 전체 Ever64.1/60.9/70.3%,단독 점 평균75.4/78.6/78.4%다.64환경1seed 진단이며 zerobg만ground 운반 높이/거리 설정이므로 구조만의 비교가 아니다.
- 신규 CPU/GPU 재채점·JSON 건수·finite·native gate·양쪽zerobg/SELF·NONE0·가중치 불변·원본/고정PTH와 YAML 해시·500epoch/frame을 확인했다. 기존3개의 재채점 검증 및 고정PTH 해시를 확인했고 ground AT/ON_TOP 출발 지면·목표 높이·거리1–5m도 통과했다. [검증](output_etc/check/relfusion_zerobg500_eval64_gpu2_20261103/verification.json). 본학습·표준 코드는 변경하지 않았다.

### ground 관계 fusion A/B 별도 컨피그

- 사용자 승인 설계로 ground 기반 `stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg`(A)·`_relfusion_zerobg`(B)의 env/train YAML 및 train/test/VNC/eval wrapper를 추가했다. 두 모델은 COUPLED64차원 임베딩을 레이어/헤드 bias 투영과 공유128→64→64 residual fusion에 함께 쓴다. A만 SELF/NONE lookup을 학습하고 B는0 고정한다. fusion은 attention 집계 후 output projection 전에 사람·레이어당1회 적용하며 단독 추가분은0이다. Pre-LN·환경·보상·물성·Stage1 PUSH26k 전이·actor encoder 동결은 유지한다. BEFORE는 이번 추가 범위에서 제외했다. [명령](markdowns/config.md), [설계](markdowns/reference/stage2_coupled_push.md#ground-관계-fusion-ab).
- 기존 coordination 경로·state_dict를 보존하고 새 경로를 별도 모듈에 추가했다. 실험 등록·평가·checkpoint 계약·문서를 연결했다. CPU73개와 설정/저장 환경40개를 통과했다. 첫 CPU 실행의 기존21k 전이 테스트1개는 해당 파일 부재로 실패했으며 관련 없는 이1개를 제외해 재실행했다. 새 A/B의 실제26k 전이·초기 액션 보존·공통 초기 가중치 일치·계산/gradient 동치·인원/셔플·단독0·gradient 발생·strict reload는 통과했다.
- GPU2에서 각각4명·1024환경·8물체로 MAX_ITERATIONS=2 확인 학습을 완료했다(epoch3/frame393216). 저장본의 actor encoder 원본 보존·action head 및 양쪽 fusion/bias 투영 갱신·모델/각263개 scalar tag finite를 확인했다. 각32환경×8preset×16step 재로드의 actor/critic/AMP/보상 finite·부분 reset·가중치 불변을 통과했고 실제 test wrapper도8환경×16step으로 완료했다. GPU0 및 기존 학습 프로세스는 변경하지 않았고 본학습·정규 성능 평가·GUI/VNC는 실행하지 않았다. [검증](output_etc/check/relfusion_ab_20261103/verification.json), [검증 재현](markdowns/USAGE.md#공통-stage2-push와-coupled).

### ground5k · 6k · 7k · 512환경×3seed 추세

- 사용자 요청으로 ground6k/frame786432000·7k/frame917504000 PTH/YAML을 복사 고정하고 GPU5/6에서4명·8물체 혼합512환경×seed0/1/2·ordered·loco·평균 액션·원래 물성/distractor·1200step6회를 새로 평가했다.6k3회 완료 후 사용자가 알려준7k를 같은 조건으로 이어 평가했고, 기존5k3회는 재사용했다. 기존 학습·표준 코드·설정은 유지했다. [Ever/Final·편차·분모](output_etc/check/ground_5k6k7k_eval512_gpu56_20261103/report.md), [명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 5k→6k→7k 공유 중심20cm 전체 Ever77.99→81.78→83.42%,4명74.48→81.82→83.03%,전원 개인 목표 동시 전체73.93→77.73→80.05%다. 단독5행동 점 평균86.34→86.55→84.06%,단독 PUSH72.27→75.57→73.35%,AT86.65→85.75→84.17%,ON_TOP 점85.09→85.10→84.26%다. 참고 공유 중심 전체 Final75.79→79.38→81.01%다. 각3seed 평균이며7k 단독 평균 하락은SIT91.14→89.36→80.93%의 영향이 크다.
- 새6회 CPU/GPU 재채점·JSON·finite·native gate·실제 기본attention 모드·가중치 불변·고정/원본PTH와 YAML 해시·epoch/frame을 확인했다. 지면 배치·ON_TOP 받침·AT/ON_TOP 운반거리1–5m와5k대비 각seed 초기12필드bitwise 일치도 통과했다. 재사용5k의 검증 기록·PTH/YAML 해시를 확인했고 평가 peak3개·최종0개다. [검증](output_etc/check/ground_5k6k7k_eval512_gpu56_20261103/verification.json).
- 사용자 요청으로 단독 PUSH만 기존9개trace를CPU float64로20/30/40/50cm 재집계했다.6k Ever75.57/82.42/86.88/89.80%,Final73.90/81.02/85.76/88.55%다. 각epoch698명이며20cm Ever/Final의seed별 성공 수·분모가 원래 집계와 정확히 일치하고 반경 포함 관계·Final→Ever도 통과했다.7k는 모든 반경에서6k보다 낮다. 추가 GPU/시뮬레이션 및 기본 평가·학습 보상 변경은 없다. [전체 결과·건수·CPU 명령](output_etc/check/ground_5k6k7k_eval512_gpu56_20261103/solo_push_radius/report.md).

### 기존 solo5 · nonenorm 동일10k · 512환경×3seed

- 사용자 요청으로 nonenorm10000epoch/frame1310720000 PTH/YAML을 복사 고정하고 GPU5/6에서4명·8물체 혼합512환경×seed0/1/2·ordered·loco·평균 액션·원래 물성/distractor·1200step3회를 새로 평가했다. 기존 solo5 같은10k의 같은 조건3회는 재사용했다. 기존 학습과 표준 코드는 유지했다. [Ever/Final·편차·분모·출처](output_etc/check/nonenorm_vs_solo5_epoch10000_512_gpu56_20261103/report.md), [명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 기존/nonenorm 순서 공유 중심20cm 전체 Ever85.31±1.07/77.97±1.61%,4명89.53±1.93/78.94±3.43%,전원 개인 목표 동시 전체82.43/71.40%다. 단독5행동 점 평균85.12/84.55%,단독 PUSH79.51/81.66%이고 참고 공유 중심 전체 Final83.25/75.09%다. 같은epoch·동일초기장면에서 두 저장본을 비교한3seed 평균±표본 표준편차다.
- 새3회 CPU/GPU 재채점·JSON·finite·native gate·실제actor/critic CO+NONE 보정·가중치 불변·고정/원본PTH와 YAML 해시·epoch/frame을 확인했다. 기존3회의 검증 기록과PTH/YAML 해시도 확인했고, 각seed의 초기12필드가 모델 사이 모두bitwise 일치했다. GPU 평가 peak3개(5에1개·6에2개)·최종0개다. [검증](output_etc/check/nonenorm_vs_solo5_epoch10000_512_gpu56_20261103/verification.json).

### ground5k · nonenorm13k · 6/8명 공동64환경

- 사용자 요청으로 직전 일반 평가에 고정한 ground5k·nonenorm13k PTH/YAML을 재사용해6명/12물체·8명/16물체 전원 공동64환경×seed0·ordered·loco·평균 액션·인원별 원래 크기/무게·distractor·1200step4회를 새로 평가했다. GPU5에서ground, GPU6에서nonenorm을 각각 순차 실행했고 기존 학습은 유지했다. 가까운 기존 solo5 6k·14k의 같은 조건4회를 재사용했다. [Ever/Final·건수·출처](output_etc/check/ground_nonenorm_n68_env64_gpu56_20261103/report.md), [명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 공유 중심20cm Ever의6명/8명은 기존6k9.38/7.81%,ground5k18.75/10.94%,기존14k15.62/15.62%,nonenorm13k10.94/23.44%다. 참고 Final은 각각7.81/4.69,10.94/10.94,15.62/12.50,9.38/15.62%다. 같은 인원의 초기12필드가8회 모두bitwise 일치했다. 서로 다른epoch·64환경1seed 진단이며 구조의 우열을 확정하지 않는다.
- 8회 CPU/GPU 재채점 기록·JSON 건수·finite·가중치 불변·native gate·PTH/YAML 해시와 epoch를 확인했다. 새 GPU 평가 peak2개·최종0개다. 첫 CPU 집계는 GPU 번호와runtime이 기록한UUID를 직접 비교해 멈췄으며, 실제 GPU5/6 UUID를 대조하도록 검사만 수정한 뒤 통과했다. 추가 시뮬레이션은 없다. [검증](output_etc/check/ground_nonenorm_n68_env64_gpu56_20261103/verification.json). 표준 코드·학습 설정은 변경하지 않았다.

### ground5k · nonenorm13k · GPU5 일반512환경×3seed

- 사용자 요청으로 각 최신 저장본 ground5000epoch/frame655360000·CO+NONE norm13000epoch/frame1703936000 PTH/YAML을 복사 고정했다. GPU5만 사용해4명·8물체 혼합512환경×seed0/1/2·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode6회를 새로 평가했다. 평가 worker2개가 각3회를 순차 실행했고 기존 본학습은 유지했다. [Ever/Final·편차·건수](output_etc/check/ground_nonenorm_latest512_gpu5_20261103/report.md), [고정 PTH·구성](output_etc/check/ground_nonenorm_latest512_gpu5_20261103/manifest.json), [재현 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- ground/nonenorm 순서 공유 중심20cm 전체 Ever는77.99±2.77/82.36±1.75%,4명은74.48±5.33/74.83±3.31%다. 단독 AT는86.65/81.35%,ON_TOP 점/영역은85.09/89.86 및78.73/81.90%,단독5행동 점 평균86.34/82.23%다. 참고 공유 중심 전체 Final75.79/79.42%,전원 개인 목표 동시 Ever73.93/76.38%,낙상112/113명(각6144명)이다. 평균±3seed 표본 표준편차다.
- 6회 CPU/GPU 재채점·JSON·finite·native gate·실제 actor/critic 보정 모드·가중치 보존·원본/고정 PTH/YAML 해시를 확인했다. ground의 AT/ON_TOP 원본·AT 목표 지면,ON_TOP 받침,운반 거리1–5m도 실제 저장 초기 상태에서 확인했다. GPU5 평가 peak2개·최종0개를 로그로 확인했다. [검증](output_etc/check/ground_nonenorm_latest512_gpu5_20261103/verification.json). 학습량과 높이/목표 분포가 다른 최신 상태 비교이며 동일 장면의 구조 비교가 아니다. 표준 코드·학습 설정 변경은 없다.
- 기존 비교 요청으로 같은512환경×3seed·ordered 평가 중 가장 가까운 solo5 8k·14k 결과를 보고서에 추가했다. 원본 집계와6개 seed의 기존 검증 기록을 확인해 재사용했으며 추가 GPU 실행은 없다. 기존5k·13k의 같은 규모 결과는 없으므로 학습량 차이를 명시했다. [기존 포함 집계·출처](output_etc/check/ground_nonenorm_latest512_gpu5_20261103/with_solo5_summary.json).

### carry5_ground 별도 컨피그 · 운반 선반 제거

- 사용자 요청으로 `stage2_coupled_push_individual_goal_solo5_carry5_ground` 환경/학습 YAML·train/test/VNC/eval을 추가했다. carry5에서 `platformMode=ground_only`, `randomHeight=false`, `randomHeightProb=0`와 실험 이름만 바꿨다. AT/ON_TOP 운반 원본과 AT 목표는 지면에 두고 ON_TOP의 실제 받침은 유지한다. 거리1–5m·정책·보상·물성·확률·Stage1 PUSH26k 전이는 같다. ground_only 사용 시 출발5cm clearance도 추가하지 않는다. [명령](markdowns/config.md), [설정 상세](markdowns/reference/stage2_coupled_push.md#carry5-지면-운반-변형).
- 공통 platform 검증/고정 actor/ground 플래그를 coupled에도 연결했다. 출발/목표 플랫폼은 기존 ground_only 기본값인 Z−20m/−19.5m에 격리된다. 새 이름을 실험·launcher·일반 평가에 등록하고 config 목록 끝에 명령을 추가했으며 구조/USAGE/README를 갱신했다. 기존 YAML·학습 프로세스와 다른 checkout은 변경하지 않았다.
- CPU58개 통과(요청과 무관한78k 전이3개 제외). GPU6에서4명·1024환경·8물체 CHECK 학습과26k 전이/저장(epoch2,frame262144)·finite를 확인했다.1024환경×10조건 총10,240장면의 거리·충돌/경로·지면 배치·ON_TOP 받침 보존을 통과했다. 첫 검사는 목표 플랫폼도−20m라는 검사 가정 때문에 실패했고, 실제 기존 기본값−19.5m에 맞춰 검사를 정정한 뒤 재검증했다. 실패 로그를 보존했다.
- GPU6·32환경×8preset×16step 저장본 재로드에서 actor/critic·AMP·보상 finite·가중치 보존·부분 reset을 확인했다. [검증](output_etc/check/carry5_ground_20261103/verification.json), [배치](output_etc/check/carry5_ground_20261103/stress/stress_validation.json), [정책](output_etc/check/carry5_ground_20261103/policy/physics_validation.json). 본학습·정규 성능 평가·GUI/VNC는 실행하지 않았다.

### solo5 20k · GPU5 단독 AT 선반 제거64환경 비교

- 사용자 요청으로 기존64환경 평가 중 가장 뒤의20k PTH를 고정하고 GPU5에서4명/8물체 혼합64환경×seed0·ordered·loco·평균 액션·원래 물성/distractor·1200step의 기존/AT 선반 제거2회를 순차 평가했다. AT만 원본 상자/목표를 지면 기준으로 낮추고 해당 플랫폼을Z−20m로 치웠다. 원본 상자5cm clearance와 높이 난수 소비·기존 충돌/배치 재시도·다른 과제는 유지했다. 학습 설정/표준 코드 변경은 없다. [결과·정의](output_etc/check/solo5_20k_at_ground64_gpu5_20261103/report.md), [명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 같은29명에서 AT Ever는26→25명(89.66→86.21%),참고 Final25→23명(86.21→79.31%)이며 AT 낙상0→1명이다. Ever 신규 성공2명·성공 상실3명으로 이번 소표본에서 선반 제거의 개선을 확인하지 못했다. 기존 조건 재실행은 과거20k 성공 수와 동일했다.
- 두 조건 CPU/GPU 재채점·JSON·finite·native gate·모델/원본 및 고정 PTH/YAML 보존을 확인했다. 초기12필드가 AT 관련Z 이외에 bitwise 일치하고 플랫폼도AT Z만 변경됐으며 reset후CPU/CUDA RNG도 같았다. [검증](output_etc/check/solo5_20k_at_ground64_gpu5_20261103/verification.json).64환경 중 단독AT29명·1seed의 추론 조건 비교이며 선반 없는 재학습 효과나 BEFORE와의 격차 원인을 확정하는 실험은 아니다.

### solo5 1.5k · GPU5 일반512환경×3seed · carry5 비교

- 사용자 요청으로 기존 solo5의1500epoch/frame196608000 PTH/YAML을 고정하고 GPU5에서4명·8물체 혼합512환경×seed0/1/2·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode3회를 순차 평가했다. 검증된 carry5 같은1.5k의3회를 재사용했다. [Ever/Final·거리·차이](output_etc/check/solo5_epoch1500_vs_carry5_eval512_gpu5_20261103/comparison.md), [명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- solo5→carry5 단독 AT 점 Ever는78.70→69.56%,ON_TOP 점/영역은75.54/78.57→70.46/74.03%다. 공동 공유 중심20cm 전체는73.30→65.51%,4명은76.32→54.99%다. 세 seed의 과제·그룹·상자 크기·사람 root/DOF는 같지만 상자/목표 배치가 달라 동일 장면의 정책 비교가 아니다. 실제 AT/ON_TOP 평균 운반 거리는solo5의5.445/5.681m에서carry5의3.015/3.006m가 됐다.
- 신규3회 CPU/GPU 재채점·JSON·finite·native gate·실제 actor/critic 설정·모델/고정 및 원본 PTH/YAML 해시를 확인했다. [검증](output_etc/check/solo5_epoch1500_vs_carry5_eval512_gpu5_20261103/verification.json), [재사용·초기 상태 비교](output_etc/check/solo5_epoch1500_vs_carry5_eval512_gpu5_20261103/comparison_verification.json).
- 사용자 지정 별도 checkout의 BEFORE source5 virtual 저장 설정/구현을 읽기 전용으로 비교했다. BEFORE는ground_only,solo5/carry5는 운반 원본 및AT 목표를 각각50% 확률로 높인다. 단독 보상 YAML·크기/밀도 범위는 같지만 BEFORE 선행은 항상AT이고 공동 그룹은PUSH다. [설정·근거·해석 한계](output_etc/check/solo5_epoch1500_vs_carry5_eval512_gpu5_20261103/config_comparison.md). 다른 checkout/학습은 수정하지 않았다.

## 2026-11-02

### carry5 1.5k · GPU5 일반512환경×3seed

- 사용자 요청으로 carry51500epoch/frame196608000 PTH와 YAML을 복사 고정하고 GPU5에서4명·8물체 혼합512환경×seed0/1/2·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode3회를 순차 평가했다. 기존 학습과 표준 코드는 유지했다. [Ever/Final·편차·분모](output_etc/check/carry5_epoch1500_eval512_gpu5_20261102/report.md), [실행 설정](output_etc/check/carry5_epoch1500_eval512_gpu5_20261102/manifest.json), [명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 단독 SIT/CLIMB/AT/ON_TOP/PUSH 점 Ever는84.91/98.96/69.56/70.46/68.86%, 단독5행동 점 평균78.55±1.95%다. 공동2/3/4명 공유 중심20cm Ever는68.66±0.80/66.01±2.21/54.99±7.54%(778/421/257그룹), 전체65.51±0.73%, 참고 Final62.83±1.12%다. 전원 개인 목표 동시 Ever/Final은58.51±1.09/55.51±1.93%이고 낙상145/6144명이다. 평균±3seed 표본 표준편차이며 앞선1k64환경1회와 직접적인 추세 비교가 아니다.
- 3회 CPU/GPU 재채점·JSON 건수·finite·native gate·실제 actor/critic 기본 보정·모델/원본 및 고정 PTH/YAML 보존을 확인했다. 단독 AT568명·ON_TOP439명의 초기 운반 거리가 모두1–5m 범위였다. GPU5 평가 peak1개·최종0개를 실행 로그로 확인했다. [검증](output_etc/check/carry5_epoch1500_eval512_gpu5_20261102/verification.json). ordered 진단이므로 formal_report=false다.

### carry5 1k · GPU5 일반64환경

- 사용자 요청으로 carry5의 최신 정기 저장본1000epoch/frame131072000과 YAML을 복사 고정하고 GPU5에서4명·8물체·64환경×seed0·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode를 평가했다. 학습·표준 코드 변경은 없다. [단독/공동 Ever·Final](output_etc/check/carry5_latest_eval64_gpu5_20261102/report.md), [실행 설정](output_etc/check/carry5_latest_eval64_gpu5_20261102/manifest.json), [명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 단독 SIT/CLIMB/AT/ON_TOP/PUSH 점 Ever는80.00/100.00/79.31/61.11/71.43%다. 공동2/3/4명 공유 중심20cm Ever는90.48/69.23/44.44%(42/13/9그룹), 전체79.69%, 참고 Final78.12%다. 전원 개인 목표 동시 Ever/Final은70.31/67.19%이며 낙상6/256명이다.64환경1seed 소규모 진단으로 기존 solo5와 목표 분포가 다르다.
- CPU/GPU 재채점·JSON 건수·finite·native gate·실제 actor/critic 기본 보정·모델/원본 및 복사 PTH/YAML 보존을 확인했다. 초기 AT 운반 거리1.016–4.993m(29명),ON_TOP1.146–4.945m(18명)로 저장된1–5m 범위를 확인했다. [검증](output_etc/check/carry5_latest_eval64_gpu5_20261102/verification.json). 정식512환경×3seed 평가는 아니다.

### 동일6k · 전체 인원4명→8명 · 20cm 비교표

- 사용자 요청으로40cm 표를 제외하고 같은6k의4명→8명 화살표 비교를 만들었다. 기존4명 norm/keep 결과가6.5k/5.5k로 학습량이 달라 세6k의4명·8물체64환경×seed0·공통resetseed0·ordered·1200step3회를 GPU3(기존/keep1 순차)·GPU5(nonenorm)에서 새로 평가했다. 검증된8명·16물체64환경3회를 재사용했다. 두 인원 모두 공동 최대4명·기존 확률/물리/거리·loco·평균 액션·distractor를 유지했다. [공유 중심·전원 개인 목표·단독 Ever/Final 비교표](output_etc/check/coupled_6k_mixed4to8_cap4_env64_gpu35_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 기존/nonenorm/keep1 순서 공유 중심20cm 전체 Ever는88.33→76.52/80.00→71.21/81.67→77.27%, 참고 Final은86.67→71.97/78.33→67.42/78.33→72.73%다.4명 공동2/3/4명 분모31/18/11그룹·단독96명,8명56/50/26그룹·단독146명이다. 각64장면1회로,4명과8명 장면 표본/구성비가 달라 전체 평균 차이에는 그룹 구성 차이도 포함된다. [건수·분모](output_etc/check/coupled_6k_mixed4to8_cap4_env64_gpu35_20261102/summary.json).
- 신규4명3회의 CPU/GPU 재채점·JSON·finite·native gate·실제 actor/critic 보정·keep1 선택·PTH/YAML/모델 불변을 확인했다.4명 내 모델별 초기12필드와 reset 전후 CPU/CUDA RNG가 bitwise 일치했다.8명 원본 summary 해시·기존 검증도 확인했으며 표준 코드/학습 설정 변경은 없다. [검증](output_etc/check/coupled_6k_mixed4to8_cap4_env64_gpu35_20261102/verification.json), [구성·재사용 출처](output_etc/check/coupled_6k_mixed4to8_cap4_env64_gpu35_20261102/manifest.json).

### 동일6k · 기존/nonenorm/keep1 · 8명 혼합·공동 최대4명

- 사용자 요청으로 세6k PTH를 재사용해8명·16물체·64환경×seed0·1200step·ordered·loco·평균 액션·원래 크기/무게/목표·distractor를 평가했다. 평가 프로세스에서만 기존 random_scenario의 공동 인원 균등 추출 상한을min(4, 남은 인원)으로 제한했다. 공동 확률0.5·단독 행동 확률·순서·물체 부족 시 재샘플링은 유지했다. CPU에서N≤4 기존 그룹/RNG 일치와N8 최대4명·유효 leader를 확인했다. 표준 sampler·학습 설정은 변경하지 않았다. [전체 Ever/Final·구성·재현](output_etc/check/coupled_6k_mixed8_cap4_env64_gpu356_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 초기 실행에서 기존/nonenorm의 환경10 배치가 재시도 상한에 걸려 reset 직전 Python/NumPy/CPU/CUDA seed0를 공통 적용했다. GPU6 keep1은 상자/목표 배치 차이로 비교에서 제외하고 GPU3에서 재실행했다. 최종 기존/keep1은GPU3, nonenorm은GPU5이며 초기12필드 및 reset 전후 CPU/CUDA RNG가 세 모델에서 bitwise 일치했다. 최초 실패 및 비교 제외 결과는 덮어쓰지 않고 manifest.attempts와 별도 경로/로그에 보존했다. GPU 간 배치 차이의 근본 원인은 확정하지 않았다.
- 기존/nonenorm/keep1 순서 공동 전원 개인 목표 Ever는89/86/88그룹(67.42/65.15/66.67%), Final은86/81/80그룹이다. 공유 중심20cm Ever는101/94/102그룹(76.52/71.21/77.27%),40cm는105/107/111그룹(79.55/81.06/84.09%)이다. 공동2/3/4명56/50/26그룹, 단독146명(SIT17·CLIMB19·AT33·ON_TOP25·PUSH52), 공동366명이고 전체64장면 중 단독/공동 혼합54장면·복수 공동 그룹54장면이다. 단일 seed 진단이다. [성공 수·분모·분할](output_etc/check/coupled_6k_mixed8_cap4_env64_gpu356_20261102/summary.json).
- 최종3회 CPU/GPU 재채점·JSON·finite·native gate·공유 중심20cm 저장 마스크 재현/40cm 단조성·모델 불변·원본/고정 PTH 및 YAML 해시·keep1 최대1명/동일 그룹/미접촉 우선/선택 상대 접촉 후 교체를 확인했다. [검증](output_etc/check/coupled_6k_mixed8_cap4_env64_gpu356_20261102/verification.json), [구성·시도 기록](output_etc/check/coupled_6k_mixed8_cap4_env64_gpu356_20261102/manifest.json).

### solo5 10k·15k·19k · 공유 중심20cm→40cm CPU 재채점

- 사용자 요청으로 기존512환경×seed0/1/2·4명 혼합·ordered·1200step 저장 궤적9회를 CPU에서 재집계했다. 실제 공유 상자 중심↔공유 목표의3D 직선거리 기준만20cm에서40cm로 완화하고 기존 그룹 유효성·Ever/Final 규칙을 유지했다. 시뮬레이션·학습·PTH·표준 성공 조건 코드는 변경하지 않았으며 표준 기본값은20cm다. [전체 및 인원별 Ever/Final·20cm 비교·재현](output_etc/check/solo5_shared_center40_10k15k19k_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 10k/15k/19k 순서40cm 공유 중심 전체 Ever는90.74±0.75/88.81±0.62/89.85±1.24%, 참고 Final은89.30±0.92/87.78±0.67/88.41±1.32%다. 평균±3seed 표본 표준편차다. 각1,456그룹에서 Ever1321/1293/1308그룹으로20cm 대비79/79/68그룹이 추가 성공했다.40cm에서도10k 평균이 가장 높다. [성공 수·seed별·분모](output_etc/check/solo5_shared_center40_10k15k19k_20261102/summary.json).
- 원래20cm의 저장 Ever/Final 마스크·JSON·기존 summary 건수/평균을 재현했다.9회 전체 프레임의 NumPy/Torch20cm·40cm 판정 일치와 임계값 완화의 단조성, 원본 trace SHA256 보존을 확인했다. 개인 목표 전원 동시 성공 및 단독 지표는 재집계 대상이 아니다. [검증](output_etc/check/solo5_shared_center40_10k15k19k_20261102/verification.json), [출처·원본 해시](output_etc/check/solo5_shared_center40_10k15k19k_20261102/manifest.json).

### 기존 solo5 · nonenorm · keep1 동일6k · GPU4 6/8명 공동64환경

- 사용자 요청으로 세 모델을6000epoch/frame786432000으로 통일해 PTH/YAML을 고정했다. keep1의 새6k도 저장 파일·내부 epoch/frame·원본/복사본 해시를 확인했다. GPU4에서 각6명·12물체 및8명·16물체 전원 공동64환경×seed0·1200step 첫 episode6회를 새로 실행했다. ordered·loco·원래 인원별 상자 크기/무게·평균 액션·distractor·저장된 actor/critic 보정/상대 선택을 유지했다. 기존 학습을 보존하고2개 worker가 인원별 세 모델을 순차 실행해 peak2개·최종0개다. [전체 Ever/Final·건수·재현](output_etc/check/nonenorm_keep1_n68_env64_gpu4_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 기존/nonenorm/keep1 순서 전원 개인 목표 Ever는6명3/5/3그룹,8명2/4/6그룹이고 Final은6명3/5/2,8명1/2/3그룹이다(각64그룹). 공유 중심 Ever는6명6/10/7,8명5/11/9그룹이며 Final은6명5/10/6,8명3/8/5그룹이다. 단일 seed 소규모 진단이며 개인 성공·낙상도 별도 기록했다. [성공 수·분모](output_etc/check/nonenorm_keep1_n68_env64_gpu4_20261102/summary.json).
- 6회 CPU/GPU 재채점·JSON 건수·finite·모델 불변·PTH/YAML 보존·실제 actor/critic 설정·전원 접촉→gate를 확인했다. 인원별 세 모델 초기12필드·CUDA RNG가 bitwise 일치하고 keep1만 CPU 모델 초기화 RNG 차이를 기록했다. keep1 궤적의 최대1명·동일 그룹·미접촉 우선·선택 상대 접촉 후 교체도 통과했다. 표준 코드·학습 설정 변경은 없다. [검증](output_etc/check/nonenorm_keep1_n68_env64_gpu4_20261102/verification.json), [고정 PTH·구성](output_etc/check/nonenorm_keep1_n68_env64_gpu4_20261102/manifest.json).

### nonenorm 6.5k · keep1 5.5k · GPU4 일반512환경×3seed

- 사용자 지정 일반 평가로 두 최신 정기 PTH를6500/5500epoch(frame851968000/720896000) 및 각 YAML과 함께 복사 고정했다. GPU4만 사용해4명·8물체 혼합512환경×seed0/1/2·ordered·loco·저장된 목표/거리/크기/무게·평균 액션·distractor·1200step 첫 episode6회를 실행했다. 기존 학습을 보존하고2개 worker가 각3회를 순차 실행했으며 로그상 peak2개·최종0개다. AT9종 크기 변경·6/8명 확장은 없다. [전체 표·seed 편차·재현](output_etc/check/nonenorm_keep1_eval_gpu4_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- nonenorm/keep1 공동 전체 Ever는72.48±1.20/71.32±2.43%, 공동4명은62.96±9.05/67.87±5.72%, 단독5행동 점 평균은82.37±1.85/81.80±1.91%다. 공유 중심 전체 Ever는77.96/77.16%, 참고 공동 Final은70.21/68.29%다. 같은512×3seed 기존 solo5 10k 결과를 재사용해 공동 전체82.43±1.60%, 단독 점 평균85.12±0.65%를 기준점으로 표시했다. 학습량이 다른 비교이며 평균±3seed 표본 표준편차다. [성공 수·분모](output_etc/check/nonenorm_keep1_eval_gpu4_20261102/summary.json).
- 6회 CPU/GPU 재채점·JSON 건수·finite·실제 actor/critic 보정·전원 접촉→gate·모델 불변·원본/고정 PTH 및 YAML 해시를 확인했다. 같은 seed의 초기12필드·CUDA RNG는 기존 solo5까지 bitwise 일치했고 keep1만 입력 차이에 따른 CPU RNG 차이를 기록했다. keep1 궤적의 최대1명·동일 그룹·미접촉 우선·선택 상대 접촉 후 교체를 확인했으며3회 교체734/672/739건 중 조기 교체는0건이다. 단독SIT225·CLIMB367·AT568·ON_TOP439·PUSH698명, 공동2/3/4명778/421/257그룹이다. 표준 코드·학습 설정은 변경하지 않았다. [검증](output_etc/check/nonenorm_keep1_eval_gpu4_20261102/verification.json), [고정 PTH·구성](output_etc/check/nonenorm_keep1_eval_gpu4_20261102/manifest.json).

### solo5 10k·15k·19k · 단독 AT 지정 상자9종

- 사용자 요청으로 기존 single AT의9종 크기를 물리 asset 생성 전에 적용하고 solo5의10k/15k/19k를 GPU2에서 각각4명·8물체 AT 전용512환경×seed0/1/2·1200step 첫 episode로 평가했다. 기존 고정 PTH/YAML을 사용하고 밀도100kg/m³·목표 샘플링·loco·평균 액션·distractor를 유지했다. 다른 작업을 보존하며1개 worker로9회 순차 실행했다. 기존 single1명 평가 및 혼합 행동 평가와 장면 구성이 다른 진단이다. BEFORE는 사용할 모델 선택 대기로 실행하지 않았다. [전체·크기별 Ever/Final·재현](output_etc/check/solo5_at9_10k15k19k_gpu2_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 각6,144명에서10k/15k/19k 순서 Ever는76.84±0.89/74.66±1.83/74.63±0.66%, 참고 Final은61.60±1.00/56.90±1.22/60.76±1.19%다. 평균±3seed 표본 표준편차다.0.57m 정육면체 Ever는41.15/31.29/30.55%로9종 중 가장 낮았다. 초기 상자→목표 XY 실제 범위는1.000~15.080m로 기존 생성 방식을 유지했다. [건수·seed별·거리](output_etc/check/solo5_at9_10k15k19k_gpu2_20261102/per_run.json).
- 9회 CPU/GPU 재채점·JSON·finite·전원 독립 AT·실제 대상9종 대응·점/영역 동일·모델 불변·원본/고정본 PTH 및 YAML 해시를 확인했다. 같은 seed의 초기12필드와 CPU/CUDA RNG가 세 PTH에서 bitwise 일치했다. 표준 코드·학습 설정은 변경하지 않았다. [검증](output_etc/check/solo5_at9_10k15k19k_gpu2_20261102/verification.json), [구성](output_etc/check/solo5_at9_10k15k19k_gpu2_20261102/manifest.json).

### 기존 solo5 · 15k/19k ordered512환경×3seed · 10k 기준 비교

- 사용자 요청으로15k/19k PTH·YAML을 복사 고정하고 GPU2에서 각각4명·8물체 혼합512환경×seed0/1/2·ordered·loco·원래 목표 샘플링/무게·평균 액션·distractor·1200step 첫 episode6회를 새로 평가했다. 기존 동일10k의3회 결과를 복사 재사용하고 현재 코드로 재채점해 총9회·3PTH를 비교했다.3개 worker 순차 실행 및 로그로 GPU2 peak3개·최종0개를 확인했다. 공동 Ever를 우선하고 단독 Ever를 다음 기준으로 평가하며 Final은 참고로 분리했다. 학습·표준 코드·기존 결과는 보존했다. [전체 표·seed 편차·그림·재현](output_etc/check/solo5_ordered_512_15k19k_vs10k_gpu2_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 10k/15k/19k 순서 공동 전체 Ever는82.43±1.60/80.29±1.73/80.85±1.35%, 공동4명은85.68±2.14/75.14±1.34/77.99±4.74%, 단독5행동 점 macro는85.12±0.65/83.37±0.85/83.08±0.56%다. 평균±3seed 표본 표준편차이며10k가 이번 비교의 공동 전체 및 단독 점 평균에서 가장 높았다. 공유 중심 전체 Ever는85.31/83.38/85.18%, 공동 전체 Final은79.54/76.66/77.96%다. [성공 수·분모·seed별 수치](output_etc/check/solo5_ordered_512_15k19k_vs10k_gpu2_20261102/summary.json), [그림](output_etc/check/solo5_ordered_512_15k19k_vs10k_gpu2_20261102/comparison.png).
- 9회 CPU/GPU 재채점·JSON 건수·finite·공유 중심·모델 불변·원본/고정본 PTH 및 YAML 해시를 검증했다. 동일 seed의 초기12필드·CPU/CUDA RNG가 세 PTH에서 bitwise 일치했고10k의 집계 전체가 이전 검증본과 정확히 같았다. 신규6회가 GPU2에서만 실행됐고 모든 trace의 기하 전원 접촉→native gate를 확인했다. 조건별 총단독SIT225·CLIMB367·AT568·ON_TOP439·PUSH698명, 공동2/3/4명778/421/257그룹이다. [구성·출처·PTH](output_etc/check/solo5_ordered_512_15k19k_vs10k_gpu2_20261102/manifest.json), [검증](output_etc/check/solo5_ordered_512_15k19k_vs10k_gpu2_20261102/verification.json).

### 기존 solo5 · 2k/4k/6k 및10k–20k 매1k · GPU2/4 ordered64환경 비교

- 사용자 요청으로2k/4k/6k/12k를 재평가하고 미평가11k/13k/15k/19k/20k를 새로 실행했다. 같은 조건의10k/14k/16k/17k/18k를 복사 재사용해 총14개 epoch의 단독5행동·공동2/3/4명·공유 중심 Ever/Final 표와 PNG/SVG 추세를 생성했다. 각4명·8물체 혼합64환경×seed0·ordered·loco·원래 목표 샘플링/무게·평균 액션·distractor·1200step 첫 episode다. 신규 실행은 GPU2·4만 사용했고 각3개 worker의 순차 실행·로그로 GPU당 peak3개 및 최종0개를 확인했다. [전체 비교표·그림·재현](output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 20k 저장 전 대기를 설정하고 파일 저장 후 내부20000epoch/frame2621440000·원본/복사본 해시를 확인한 뒤 평가를 완료했다. 이번64환경에서 공동 전체 Ever는14k57/64(89.1%)가 가장 높고 단독5행동 점 macro는17k87.7%가 가장 높았다.20k는 단독 점/영역 macro85.9/93.6%, 공동52/64(81.2%), 공유 중심53/64(82.8%), 단독 PUSH25/28(89.3%)다. Final은 참고로 분리했으며10k 이후 단조 개선을 확인하지 못했다. [성공 수·분모](output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/summary.json), [그림](output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/trend.png).
- 신규9회와 재사용5회 모두 현재 CPU 재채점·GPU/JSON 건수·finite를 확인했고 모델 불변·14개 PTH/원본 YAML 해시를 검증했다. 초기12필드는14개 모두 bitwise 일치하고 저장된13개 CPU/CUDA RNG도 일치한다. 과거10k는 RNG 파일이 없어 물리/과제 초기 상태로 대조했다. 재평가4k/6k/12k 성공 수는 과거와 같고2k는 ON_TOP 점 Ever12→13/18·PUSH Ever19→18/28·AT Final21→22/29 등의1명 차이를 기록했다.2k 표에는 새 값을 사용하며 원인은 확정하지 않았다. 학습·표준 코드·기존 결과는 보존했다. [실행·재사용 출처](output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/manifest.json), [고정 PTH](output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/checkpoint_manifest.json), [검증·기준점·동시 실행 상한](output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/verification.json).

### CO+NONE norm·keep1 동일4k · 6/8명 전원 공동32환경 비교

- 직전 추세 평가에서 고정한 두4000epoch/frame524288000 PTH를 사용해 GPU5·6에서6명·12물체 및8명·16물체 전원 공동32환경×seed0·ordered·loco·원래 인원별 상자 크기/무게·평균 액션·distractor·1200step 첫 episode4회를 새로 평가했다. 저장된 actor/critic 보정·keep1 선택을 유지했으며 학습 프로세스·표준 코드·PTH는 변경하지 않았다. [전체 Ever/Final·공유 중심·개별 성공·재현](output_etc/check/nonenorm_keep1_epoch4k_n68_32_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- norm/keep1의 전원 동시 Ever는6명4/32·6/32,8명3/32·3/32이고 Final은6명3/32·3/32,8명2/32·3/32다. 공유 중심 Ever는6명8/32·8/32,8명9/32·5/32이며 Final은6명6/32·8/32,8명7/32·4/32다. keep1은6명 전원 Ever가2그룹 높지만8명은 동률이며 공유 중심은 낮아 이번 소표본에서 일관된 확장성 우위를 확인하지 못했다. 낙상은 norm6/8명5/192·9/256, keep1 5/192·0/256명이다.
- 4회 CPU/GPU 재채점·JSON 건수·finite·모델 불변·원본/고정본 PTH/YAML 해시와 내부4k/frame을 확인했다. 동일 인원에서 초기12필드·CUDA RNG가 모델 간 bitwise 일치했고 keep1 추가 관측의 CPU 초기화 RNG 차이는 기록했다. keep1 두 인원에서 매 step 유효 상대·미접촉 우선·최대1명·전원 접촉 후1명 유지 조건을 GPU/trace로 확인했다. [조건·PTH](output_etc/check/nonenorm_keep1_epoch4k_n68_32_20261102/manifest.json), [성공 수·분모](output_etc/check/nonenorm_keep1_epoch4k_n68_32_20261102/summary.json), [검증](output_etc/check/nonenorm_keep1_epoch4k_n68_32_20261102/verification.json).

### CO+NONE norm1k–5k·keep1 1k–4k · ordered64환경 추세

- 사용자 요청으로 norm1k/2k/3k/4k/5k와 keep1 1k/2k/3k/4k 정기 PTH·YAML을 복사 고정하고 GPU5·6에서9회 모두 새로 평가했다. 각4명·8물체 혼합64환경×seed0·ordered·loco·평균 액션·원래 무게·distractor·1200step 첫 episode이며 저장된 목표 샘플링·actor/critic 보정·keep1 선택을 유지했다.6/8명·AT/ON_TOP 거리 변경은 추가하지 않았다. [전체 Ever/Final·점/영역·공유 중심·재현](output_etc/check/nonenorm_keep1_trend64_1k5k_20261102/report.md), [그림](output_etc/check/nonenorm_keep1_trend64_1k5k_20261102/trend.png), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- norm1k→5k 공동 전체 Ever는64.1/60.9/67.2/71.9/79.7%, keep1 1k→4k는56.2/59.4/64.1/64.1%다. norm5k의2/3/4명 전원 동시 Ever는81.0/76.9/77.8%, 단독5행동 점 macro는85.05%다. keep1 4k는71.4/46.2/55.6%, 단독 macro85.06%다. 같은4k에서 norm은 공동71.9%·단독79.60%, keep1은 공동64.1%·단독85.06%로 과제별 차이가 있다. Final은 참고로 분리했다. [성공 수·분모](output_etc/check/nonenorm_keep1_trend64_1k5k_20261102/summary.json).
- 9회 CPU/GPU 재채점·JSON 건수·finite·모델 불변·원본/복사본 PTH/YAML 해시 보존을 확인했다. 초기12필드와 CUDA RNG는9개 모두 bitwise 일치하고 CPU RNG는 설정 내 일치한다. keep1의 추가 관측에 따른 설정 간 CPU 모델 초기화 RNG 차이는 별도로 기록했다. keep1 네 PTH는 매 step 선택 상대·미접촉 우선·전원 접촉 후1명 유지 조건을 GPU와 저장 trace에서 확인했다. 단독 표본SIT10·CLIMB12·AT29·ON_TOP18·PUSH28명, 공동2/3/4명42/13/9그룹의 소규모 추세다. [PTH·9개 구성](output_etc/check/nonenorm_keep1_trend64_1k5k_20261102/manifest.json), [검증](output_etc/check/nonenorm_keep1_trend64_1k5k_20261102/verification.json).

### 기존 solo5 · AT/ON_TOP 목표1–5m · 네 PTH 32환경 진단

- 사용자 요청으로8k/10k/14k/18k 고정 PTH를 재사용해 AT·ON_TOP만 각각32환경×4명·8물체·seed0·loco·1200step으로 평가했다. 기존 환경 중심±4.5m 정사각형과 payload 기준1–5m 원형 후보 샘플러를 비교해 GPU5·6에서 총16회를 완료했다. 복원 후 첫 reset 전에 평가용 거리 키만 적용했고 저장 PTH·학습 설정·표준 코드는 보존했다. [Ever/Final·점/영역·건수·재현](output_etc/check/solo5_at_ontop_goal5_32_8k10k14k18k_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 8k/10k/14k/18k 순서 AT Ever 기존→1–5m는83.6→82.0/84.4→72.7/83.6→71.9/82.8→76.6%, ON_TOP 점 Ever는72.7→75.0/73.4→80.5/78.9→71.9/77.3→75.8%다. Final은 참고로 분리했다. 조건별 실제 목표 거리 범위는 AT 기존1.103–12.591m·변경1.017–4.942m, ON_TOP 기존1.131–12.591m·변경1.013–4.985m였다. 목표 방향/위치 분포와 충돌 재샘플링도 바뀌므로 동일 장면의 거리만 바꾼 짝비교는 아니며 단일 seed 소규모 진단이다.
- 16회 CPU/GPU 재채점·JSON 건수·finite·모델/PTH/원본 YAML 해시 보존, 과제별128명 전원 단독 구성과 실제1–5m 범위를 확인했다. 같은 행동·조건의 초기12필드·CPU/CUDA RNG가 네 PTH에서 bitwise 일치했다. [조건·PTH](output_etc/check/solo5_at_ontop_goal5_32_8k10k14k18k_20261102/manifest.json), [검증](output_etc/check/solo5_at_ontop_goal5_32_8k10k14k18k_20261102/verification.json), [성공 수·개별 거리](output_etc/check/solo5_at_ontop_goal5_32_8k10k14k18k_20261102/summary.json).

### 기존 solo5 · 8k/10k/14k/18k ordered512환경×3seed 정밀 비교

- 네 정기 PTH·relation_config를 복사 고정하고 GPU5·6에서 각각 일반4명·8물체 혼합512환경×seed0/1/2·ordered·원래 무게·평균 액션·distractor·loco 초기화·1200step·첫 episode를 새로 평가했다. 총12회이며6/8명은 추가하지 않았다. Ever만 PTH 선택에 사용하고 Final은 정지·유지 참고자료로 분리했다. 표준 코드와 학습 프로세스는 변경하지 않았다. [전체 표·재실행](output_etc/check/solo5_ordered_512_seed012_8k10k14k18k_20261102/report.md), [그림](output_etc/check/solo5_ordered_512_seed012_8k10k14k18k_20261102/comparison.png), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 8k/10k/14k/18k 순서로 단독5행동 점 macro Ever는84.53/85.12/79.63/82.03%, 공동 전체 전원 동시 Ever는80.85/82.43/78.24/78.79%, 공유 중심 Ever는83.80/85.31/82.43/83.46%다. 모두3seed 성공률의 평균이며 macro는 다섯 단독 과제의 단순 평균이다.10k가 이번 표본에서 단독 macro와 공동 전체가 모두 가장 높고,8k와의 차이는 작다. [seed 편차·성공 수·분모](output_etc/check/solo5_ordered_512_seed012_8k10k14k18k_20261102/summary.json).
- 12회 CPU/GPU 재채점·JSON 건수·finite·공유 중심 일치·모델/PTH 해시 보존·기하 전원 접촉→native gate를 확인했다. 동일 seed의 초기12필드와 CPU/CUDA RNG가 네 PTH에서 bitwise 일치했다. 단독은 총SIT225·CLIMB367·AT568·ON_TOP439·PUSH698명, 공동은2/3/4명 각각778/421/257그룹이다. [검증](output_etc/check/solo5_ordered_512_seed012_8k10k14k18k_20261102/verification.json), [PTH 고정·12개 구성](output_etc/check/solo5_ordered_512_seed012_8k10k14k18k_20261102/manifest.json).

### 기존 solo5 · 14k/16k/18k ordered64환경 추세 추가

- 14000/16000/18000epoch PTH·relation_config를 복사 고정하고 일반4명/8토큰 및6·8명 전원 공동12·16토큰을 모두64환경×seed0·ordered·원래 무게·평균 액션·distractor·loco 초기화·1200step·첫 episode로 평가했다.18k 일반은 직전 동일64환경 검증본을 재사용하고 나머지8개를 GPU5·6에서 새로 실행했다.5명은 추가하지 않았다. 과거1k~12k의13개 결과와 합친22개 구성의 표·PNG/SVG 추세를 생성했다. 표준 코드·학습 프로세스는 변경하지 않았다. [전체 ever/final·단독 점/영역·공유 중심·재실행](output_etc/check/solo5_ordered_trend64_14k18k_20261102/report.md), [그림](output_etc/check/solo5_ordered_trend64_14k18k_20261102/trend.png), [PTH·구성](output_etc/check/solo5_ordered_trend64_14k18k_20261102/manifest.json), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 14k/16k/18k 순서로 일반 공동 전원 개인 목표 ever는57/64·50/64·48/64, final은54/64·49/64·48/64다.6명 ever는7/64·12/64·6/64, final은6/64·12/64·6/64이며8명 ever는4/64·8/64·6/64, final은2/64·7/64·5/64다. 공유 중심 final은6명10/64·16/64·12/64,8명8/64·11/64·11/64다. 이번 추가 표본에서 일반 혼합은14k,6/8명은16k가 높았고 단조 증가하지 않았다.
- 9개 구성 CPU/GPU 재채점·JSON/finite·모델/PTH 보존과 기하 전원 접촉→native gate를 확인했다. 새 PTH들 사이 및 과거12k 대비 인원/과제별 초기12필드가 bitwise 일치하고 새 구성끼리 CPU/CUDA RNG도 일치했다. 과거와 relation_config 해시가 같음을 확인했으며 단독/공동·공유 중심·낙상·기존 native gate 추진 진단을 함께 보존했다. [검증](output_etc/check/solo5_ordered_trend64_14k18k_20261102/verification.json), [과거 포함 집계](output_etc/check/solo5_ordered_trend64_14k18k_20261102/combined_breakdown.json).

### 기존·CO+NONE·keep1 최신 저장본 18000/4500/3500 평가

- 요청 시점의 기본 이름 PTH와 최신 정기 PTH 내부 epoch가 같음을 확인하고 기존18000/frame2359296000·CO+NONE4500/frame589824000·keep1 3500/frame458752000의 정기 PTH·relation_config를 복사 고정했다. GPU5·6에서 각 일반4명/8토큰64환경,6·8명 전원 공동12·16토큰32환경을 seed0·ordered·원래 무게·평균 액션·distractor·1200step·첫 episode로 새로 평가했다. keep1은 저장된 actor/critic 최대1명 선택·미선택 NONE, 나머지는 저장 보정·상대 제한 없음을 유지한다. 학습량이 다른 최신 상태 비교다. 표준 코드·학습 프로세스는 변경하지 않았다. [전체 ever/final·점/영역·공유 중심·재실행](output_etc/check/coupled_three_latest_eval_20261102/report.md), [PTH 고정](output_etc/check/coupled_three_latest_eval_20261102/manifest.json), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 기존/CO+NONE/keep1 순서로 일반 공동 전원 개인 목표 final은48/64·47/64·38/64다.6명 ever4/32·6/32·5/32와 final4/32·5/32·4/32,8명 ever0/32·4/32·2/32와 final0/32·2/32·2/32다. 공유 중심 final은6명5/32·8/32·7/32,8명1/32·5/32·3/32다.6/8명 개인 낙상은 기존18/192·13/256,CO+NONE5/192·1/256,keep1 3/192·2/256이었다.
- 전원 접촉 전후 정지 후5초 재출발은6명1/5·1/10·0/12,8명0/7·1/16·1/12다. 첫1명 이상 접촉 후 첫 정지 조건의 출발 순간 기하/힘 접촉 인원도 별도 집계했다.9개 rollout 재채점·JSON/finite·모델/PTH 보존·keep1 매 step 선택,같은 조건 초기12필드·CUDA RNG bitwise 일치와384그룹·248정지 창·389이동 판정 기록의 속도/거리/지속시간·낙상 전 조건을 확인했다. 기존/CO+NONE CPU RNG는 일치하며 keep1의 추가 관측 차원에 따른 모델 초기화 소비 차이는 별도로 기록했다. [평가 검증](output_etc/check/coupled_three_latest_eval_20261102/verification.json), [정지·접촉 인원](output_etc/check/coupled_three_latest_eval_20261102/contact_count_report.md), [분석 검증](output_etc/check/coupled_three_latest_eval_20261102/contact_count_verification.json).

### GPU0 전용 MPS 정상 종료

- 사용자 요청으로 GPU0 전용 pipe `/tmp/mps-test-1003/gpu0/pipe`의 서버753996·제어753978과 연결 클라이언트0개를 재확인한 뒤 `quit`으로 정상 종료했다. CO 학습796534는 이미 종료된 상태였다. GPU reset이나 직접 kill은 실행하지 않았다.
- 서버 로그의 Exiting과 두 데몬 종료를 확인했다.5초 후 jh의 GPU0 VLLM2209653·Python2554841은 동일 PID로 실행/GPU 메모리 유지 중이며, 다른 GPU의 MPS3개와 확인한 학습6개도 모두 살아 있었다. 서비스 요청 자체의 기능 검사는 하지 않았다. [종료 전](output_etc/check/gpu0_mps_shutdown_20261102/before.json), [종료 결과](output_etc/check/gpu0_mps_shutdown_20261102/quit_result.json), [종료 후](output_etc/check/gpu0_mps_shutdown_20261102/after.json).

### keep1 2000 추가 · 네 설정 동일 epoch 비교

- keep1의 내부2000epoch/frame262144000 PTH·relation_config를 복사 고정해 GPU5에서 일반4명/8토큰64환경·6명/12토큰32환경, GPU6에서8명/16토큰32환경을 각각seed0·ordered·원래 무게·평균 액션·distractor·1200step·첫 episode로 평가했다. 저장된 actor/critic 최대1명 선택·미선택 NONE을 유지했다. 기존·CO·CO+NONE2k의9개 검증 rollout을 복사 재사용해 네 설정을 비교했다. 표준 코드·학습 프로세스는 변경하지 않았다. [전체 ever/final·점/영역·공유 중심·재실행](output_etc/check/coupled_epoch2000_keep1_compare_20261102/report.md), [PTH·출처](output_etc/check/coupled_epoch2000_keep1_compare_20261102/manifest.json), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- keep1의 공동2/3/4/6/8명 전원 개인 목표 ever는24/42·9/13·5/9·2/32·4/32, final은23/42·8/13·5/9·2/32·1/32다. 공유 중심 ever는26/42·10/13·5/9·2/32·10/32, final은24/42·10/13·5/9·2/32·5/32다. 일반 공동 전체 final은36/64이며 단독 행동별도 집계했다.
- 전원 접촉 전후 정지 후5초 재출발은6명1/11·8명0/13이다. 첫1명 이상 접촉 후 첫 정지 조건에서는6명15/26·8명14/28이 재출발했고 그중14/15·13/14는 첫 전원 접촉 이전 출발이었다. [정의·전체 비교·출발 접촉 인원](output_etc/check/coupled_epoch2000_keep1_compare_20261102/contact_count_report.md).
- 12개 rollout 재채점·JSON/finite·모델/PTH 보존과 keep1의 매 step 선택 규칙을 확인했다. 같은 조건 초기12필드·CUDA RNG는 네 설정 간 bitwise 일치한다. keep1의 CPU 모델 초기화 RNG 소비량 차이는 별도 기록했고 기존과 두 norm의 CPU RNG는 일치한다.512그룹·369정지 창·559이동 판정 기록의 속도/거리/지속시간·낙상 전 조건과 기존 분모 일치를 확인했다. [평가·선택 검증](output_etc/check/coupled_epoch2000_keep1_compare_20261102/verification.json), [분석 검증](output_etc/check/coupled_epoch2000_keep1_compare_20261102/contact_count_verification.json).

### 기존·CO·CO+NONE · 동일2000epoch 평가·접촉 인원 비교

- 세 PTH의 내부2000epoch/frame262144000과 원본/복사 해시를 확인해 고정했다. GPU5·6에서 기존·CO+NONE의 일반4명/8토큰64환경×seed0,6·8명 전원 공동12·16토큰32환경×seed0를 새로 평가하고 CO2k의 검증된 동일 조건3개 rollout은 복사 재사용했다. ordered·원래 무게·평균 액션·distractor·1200step·첫 episode·상대 제한 없음과 각 actor/critic4층 보정을 유지했다. 진단 실행기의 MPS UUID 전환 전 물리 GPU 번호로 작업을 분배하도록 수정 후 실행했다. 표준 코드·학습 프로세스는 변경하지 않았다. [전체 ever/final·점/영역·공유 중심·재실행](output_etc/check/coupled_epoch2000_compare_20261102/report.md), [PTH·재사용 출처](output_etc/check/coupled_epoch2000_compare_20261102/manifest.json), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 기존/CO/CO+NONE 순서로 일반 공동 전원 개인 목표 final은41/64·38/64·34/64다.6명 ever5/32·3/32·1/32와 final1/32·2/32·1/32,8명 ever5/32·0/32·1/32와 final4/32·0/32·0/32다. 같은2k에서 CO+NONE의 우위는 확인되지 않았다. 앞선3.5k와 동일 조건에서 CO+NONE6명 final은1/32→6/32로 증가했다.
- 전원 접촉 전후0.5초 정지 후5초 내 재출발은6명2/16·1/10·0/12,8명0/14·0/10·0/14다. 첫1명 이상 접촉 이후 첫 정지 조건도 별도로 집계하고 출발 순간 기하/힘 접촉 인원을 보존했다.9개 rollout 재채점·JSON/finite·모델 보존·같은 조건 초기12필드 및 CPU/CUDA RNG bitwise 일치,384그룹·279정지 창·422이동 판정 기록의 속도/거리/지속시간/낙상 전 조건을 확인했다. [성공률 검증](output_etc/check/coupled_epoch2000_compare_20261102/verification.json), [정지·접촉 인원](output_etc/check/coupled_epoch2000_compare_20261102/contact_count_report.md), [분석 검증](output_etc/check/coupled_epoch2000_compare_20261102/contact_count_verification.json).

### 네 coupled 설정 · 정지 후 출발 순간 접촉 인원 비교

- 기존 전원 접촉 전후 정지 표본을 네 설정 모두 비교하고, 각 재출발 순간의 기하/힘 접촉 인원과 확인 구간의 전원 유지 비율을 추가 집계했다. 기존/CO/CO+NONE/keep1 순서로6명5초 재출발은1/4·1/10·1/8·1/9,8명은3/8·0/10·1/12·0/9다. 이 표본의6/8명 재출발은 시작 순간 모두 기하 전원 접촉이었으나 계속 유지되는 조건은 아니었다. [정의·전체 표·재실행](output_etc/check/coupled_four_latest_eval_20261102/contact_count_report.md).
- 별도로 첫1명 이상 기하 접촉 이후 첫0.5초 정지를 그룹당1회 골랐다. CO+NONE의6명17/29·8명12/30이5초 안에 재출발했고, 모두 첫 전원 접촉 이전 출발이었다. 출발 시 접촉 인원은6명 장면1~4명,8명 장면2~7명이다. 나머지 설정도6/8명에서 이 조건의 출발은 모두 전원 접촉 이전이었다. 엄격한 양손 기준이므로 다른 사람이 전혀 닿지 않았다는 의미나 의도적 대기/추진의 인과 증거로 해석하지 않는다.
- CPU로12개 궤적512그룹·354정지 창·589이동 판정 기록의 정렬/속도·지속시간·거리·낙상 전 조건을 검사하고 기존 전원 표본의 분모/건수와 일치를 확인했다. 새 시뮬레이션·정책·학습 변경은 없다. [그룹별](output_etc/check/coupled_four_latest_eval_20261102/contact_count_groups.json), [검증](output_etc/check/coupled_four_latest_eval_20261102/contact_count_verification.json).

### CO+NONE 3500 · 전원 접촉 뒤 정지·재출발 궤적 분석

- 직전 네 모델12개 rollout을 CPU로만 분석했다. 첫 낙상/목표50cm 진입 전 엄격한 전원 양손 접촉을 사용하고, 상자 속도1cm/s 미만0.5초 정지와 목표 방향5cm/s 초과0.5초·같은 연속 구간10cm 이상 전진을 구분했다. CO+NONE의 첫 전원 접촉 직전/직후 정지 장면 중5초 안에 재출발한 비율은2명7/7·3명4/4·6명1/8·8명1/12이며4명은 해당 표본이 없다. 동료 접촉을 인식해 기다린다는 인과 증거로 해석하지 않는다. [정의·전체 표·사례·재실행](output_etc/check/coupled_four_latest_eval_20261102/stop_start_report.md).
- 밀다가 멈춘 경우까지 포함한 첫 후속 정지에서는 CO+NONE6명2/20·8명2/25가 남은 관측 중 재출발했다.5초 동안90% 이상 정체한 장면은 각각17/19·22/25다.512그룹의 기록 정렬/전원 접촉→gate와201정지·70재출발 판정의 원본 속도/위치·시점·낙상 전 조건을 확인했다. 새 시뮬레이션·정책/학습 변경은 없다. [그룹별](output_etc/check/coupled_four_latest_eval_20261102/stop_start_groups.json), [검증](output_etc/check/coupled_four_latest_eval_20261102/stop_start_verification.json).

### 실행 중 coupled 네 설정 · 최신 저장 PTH ordered 평가

- 실제 실행 중인 기존 solo5·CO norm·CO+NONE norm·keep1의 요청 시점 최신 정기 PTH를 각각17000/2000/3500/3000epoch로 복사 고정했다. shrink1은 실행/저장본이 없어 제외했다. GPU5·6에서 일반4명/8토큰64환경×seed0와6·8명 전원 공동12·16토큰32환경×seed0를 각각 평가했다. ordered·원래 무게·평균 액션·distractor·1200step·첫 episode를 유지하고 각 설정의 원래 actor/critic norm·상대 선택을 사용했다. 학습량이 다른 진단 비교다. [전체 ever/final·점/영역·공유 중심·재실행](output_etc/check/coupled_four_latest_eval_20261102/report.md), [PTH 고정](output_etc/check/coupled_four_latest_eval_20261102/manifest.json), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 기존/CO/CO+NONE/keep1 순서로6명 전원 개인 목표 ever는2/3/7/6그룹·final2/2/6/6그룹,8명 ever2/0/3/0·final2/0/2/0그룹(각32그룹)이다. 공유 중심 final은6명3/3/6/7·8명4/0/4/1그룹이다. 단독 행동별과 일반 공동2/3/4명, 전원 양손 접촉·낙상도 함께 집계했다.
- 12개 rollout의 CPU/GPU 재채점·JSON/finite·모델/PTH 보존과 keep1의 매 step 최대1명/미접촉 우선/불필요한 교체 없음을 확인했다. 네 모델의 같은 조건 초기12필드·CUDA RNG는 bitwise 일치했다. 기존과 두 norm의 CPU RNG도 일치하며 keep1은 추가 관측 차원에 따른 모델 초기화 소비 차이로 달랐다(입력 차원만 바꾼 재현 확인). 정책 토큰 셔플은 CUDA RNG를 사용한다. 표준 코드·학습 프로세스는 변경하지 않았다. [검증](output_etc/check/coupled_four_latest_eval_20261102/verification.json).

### 기존 solo5 500 · 두 norm 500과 동일 조건 비교

- 기존 solo5의500epoch/frame65536000 PTH·relation_config를 복사 고정해 GPU5(일반·8명)·6(6명)에서 추가 평가했다. 일반4명/8토큰64환경×seed0,6·8명 전원 공동은12·16토큰32환경×seed0이며 ordered·원래 무게·평균 액션·distractor·1200step·첫 episode·상대 제한 없음을 직전 norm 평가와 맞췄다. 두 norm의 검증된 결과는 재사용했다. [세 모델 비교표·재실행](output_etc/check/solo5_base500_vs_norm500_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 기존500의 공동2/3/4/6/8명 전원 개인 목표 ever는27/42·6/13·3/9·0/32·3/32, final은25/42·6/13·3/9·0/32·0/32다. 공유 중심 final은29/42·7/13·5/9·1/32·3/32다. 단독 행동별 ever/final·점/영역도 함께 집계했다.
- 세 모델 간 같은 조건 초기12필드·CPU/CUDA RNG가 bitwise 일치했다. 기존3개 rollout의 CPU/GPU 재채점·JSON 건수·finite·actor/critic 보정 없음·모델 불변·원본/복사 PTH 해시 보존을 확인했다. 표준 코드·학습 프로세스는 변경하지 않았다. [검증](output_etc/check/solo5_base500_vs_norm500_20261102/verification.json).

### 두 norm 500 PTH · 일반64환경·6/8명 공동32환경 ordered 평가

- COUPLED-only·COUPLED+NONE 보정의500epoch/frame65536000 PTH와 relation_config를 복사 고정했다. GPU5/6에서 각각4명/8토큰 일반 혼합64환경×seed0,6명/12토큰·8명/16토큰 전원 공동32환경×seed0를 ordered·원래 무게·평균 액션·distractor·1200step·첫 episode로 평가했다. 상대 제한은 끄고 각 저장본의 actor/critic4층 norm을 유지했다. 단독 행동별 ever/final·점/영역과 공동2/3/4/6/8명 전원 개인 목표·공유 중심을 집계했다. [전체 표·재실행](output_etc/check/solo5_norm500_ordered_eval_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- CO-only→CO+NONE의 공동2/3/4명 전원 개인 목표 final은19/42→24/42,4/13→8/13,4/9→4/9다.6명 ever는둘 다1/32·final1/32→0/32,8명 ever3/32→2/32·final둘 다1/32다. 6/8명 공유 중심 final은3/32→4/32·4/32→2/32이며6/8은각32그룹×1seed 진단이다.
- 같은 조건에서 두 PTH의 초기12개 tensor·CPU/CUDA RNG가 bitwise 일치했다.6개 rollout의 CPU/GPU 재채점·JSON 건수·finite·양쪽4층 보정 모드·모델 불변과 원본/복사 PTH 해시 보존을 확인했다. 표준 코드·학습 프로세스는 변경하지 않았다. [검증](output_etc/check/solo5_norm500_ordered_eval_20261102/verification.json), [PTH·설정](output_etc/check/solo5_norm500_ordered_eval_20261102/manifest.json).

### solo5 keep1/shrink1 학습 설정 · viewer keep3/shrink3 NONE 전환·재평가

- solo5 기반 `stage2_coupled_push_individual_goal_solo5_keep1`·`stage2_coupled_push_individual_goal_solo5_shrink1`의 env/train YAML과 train/test/VNC/eval을 추가했다. 학습·추론의 actor/critic4층에서 미접촉 공동 상대 최대1명을 선택하고 나머지는 NONE으로 바꾼다. keep1은 미접촉 상대가 없으면 접촉한1명 유지, shrink1은0명이다. 선택 ID를 관측에 저장해 PPO 미니배치에서도 재사용하며 기본 관측은1276차원이다. 보상·물성·샘플링·4명/1024환경/8토큰·Stage1 PUSH26k 전이는 유지하고 새 모드는 checkpoint 계약으로 구분한다. 학습은 실행하지 않았다. [설정·명령](markdowns/config.md), [정의](markdowns/reference/stage2_coupled_push.md#최대1명-coupled-선택-학습).
- 사용자 수정 요청대로 기존 viewer keep3/shrink3도 미선택 상대를 마스킹하는 대신 NONE으로 전환한다. 실제 그룹/보상과 기존5명 이상 적용·선택/교체 규칙은 유지하며 actor 전용이다. COUPLED0명도 다른 사람을 NONE으로 계속 본다. 과거 마스킹 평가 결과는 보존하고 현재 코드로 재실행하면 NONE 전환임을 USAGE에 명시했다.
- 관련 CPU64검사와 기본 모드8조건의 출력/입력·파라미터 gradient bitwise 일치를 확인했다. 접촉/상실·단독/혼합/13명·부분 reset·독립 RNG·양쪽4층의 실제 relation bias/유한 attention·PPO 배치 재사용/역전파·26k 전이·설정 간 혼용 거부를 검사했다. GPU6에서 새 두 설정 각각4명·4환경·8preset×16step 및8명·4환경·전원 공동16step의 실제 물리/모델/AMP·부분 reset·가중치 보존을 통과했다. wrapper6개 호출 캡처와8개 shell 구문을 확인했다. [검증](output_etc/check/solo5_keep1_shrink1_configs_20261101/verification.json), [CPU](output_etc/check/solo5_keep1_shrink1_configs_20261101/cpu_tests.log).
- 기존12000 PTH·8명·64환경×seed0·물체16개·ordered·원래 무게·평균 액션·1200step으로 NONE keep3(GPU5)/shrink3(GPU6)를 재평가했다. all은 직전 동일 조건 결과를 재사용했다. all/keep3/shrink3 전원 개인 목표 ever는4/3/3그룹, final은 모두2/64그룹이며 공유 중심 ever는8/9/4그룹, final은5/7/3그룹이다. 초기12필드/CPU·CUDA RNG·선택 규칙·CPU/GPU 재채점·JSON/finite·모델/PTH 보존을 확인했다. GPU5에서 두 모드 실제 VNC·8명·16step×2씬도 통과했다. [NONE 결과·이전 마스킹 비교](output_etc/check/solo5_peer_none_eval8_epoch12000_20261101/report.md), [재채점·선택 검증](output_etc/check/solo5_peer_none_eval8_epoch12000_20261101/verification.json), [VNC](output_etc/check/solo5_peer_none_eval8_epoch12000_20261101/viewer_verification.json).

## 2026-11-01

### solo5 12000 · 8명 ordered · all/keep3/shrink3 평가

- 최근 추세 평가의12000 PTH를 복사 고정해 GPU6(all/shrink3)·5(keep3)에서8명·64환경×seed0·물체16개·원래 무게·평균 액션·distractor·1200step·첫 episode 조건으로 세 방식을 재평가했다. 기존 공간 채점/개인 낙상 제어는 유지하고 접촉·선택을 매 step 기록했다. 전원 개인 목표 ever는 all/keep3/shrink3 각각4/2/3그룹, final은 모두2/64그룹이다. 공유 중심 ever는8/10/7그룹, final은5/8/4그룹이다. 전원 양손 기하 접촉 ever는54/55/54그룹이며64환경×1seed 진단이다. [비교표·재실행](output_etc/check/solo5_peer_eval8_epoch12000_20261101/report.md), [실행 안내](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 세 조건의 초기12개 tensor·CPU/CUDA RNG와 이전12000 기준의 초기 상태가 bitwise 일치했다. all의 성공 건수도 이전 결과와 일치했다. 3회 CPU/GPU 재채점·JSON 건수·finite·매 step 선택 수/접촉 우선/불필요한 재선택 없음·모델/PTH 해시 보존을 확인했다. keep3은 항상3명, shrink3은0~3명이며 전원 접촉 시 제거 규칙을 실제 rollout에서 검증했다. 표준 학습/평가 코드는 변경하지 않았다. [검증](output_etc/check/solo5_peer_eval8_epoch12000_20261101/verification.json).

### 개인 목표 coupled viewer · 접촉 기반 keep3/shrink3 상대 선택

- `COUPLED_PEERS=all|keep3|shrink3`를 test/VNC와 직접 CLI에 추가했다. 5명 이상 공동 그룹에서 query별 미접촉 상대 최대3명을 선택하고 필요한 슬롯만 갱신한다. keep3은 접촉한 상대를 채워3명 유지, shrink3은 접촉한 상대를 제외해 전원 접촉 시0명이다. 접촉 상실 시 다시 후보가 되며 기준은 양손 기하 `on_face`다. 선택되지 않은 COUPLED key만 actor 사람 attention4층에서 가리고 SELF/NONE/BEFORE는 유지한다. 실제 그룹·상자·목표·보상/gate·물성과 기본 all/4명 이하 동작은 유지한다. [명령](markdowns/config.md#추론-옵션), [정의·제약](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 관련 CPU42검사 통과와 기존 코드 대비 기본 모드8조건의 출력/입력·파라미터 gradient bitwise 일치를 확인했다. 독립 RNG·부분 reset·슬롯 유지/교체·전원 접촉/접촉 상실·NONE/BEFORE 보존·4층 셔플 mask·학습/resume/정식 평가 오사용 거부를 검사했다. [CPU 로그](output_etc/check/solo5_peer_selection_20261101/cpu_tests.log), [기존 동작 비교](output_etc/check/solo5_peer_selection_20261101/legacy_verification.json).
- 사용자 지정 PTH의12500epoch/frame1638400000 복사본을 고정해 GPU5에서8명·1환경·8토큰·ordered·16step×2씬의 두 모드 실제 VNC를 통과했다. 별도 keep3 실행에서32액션 finite·씬별 선택 초기화·모델 불변과 CUDA 전원 접촉 시 keep3 유지/shrink3 제거·부분 reset을 확인했다. 짧은 실행 검증이며 성능 평가는 하지 않았다. 학습은 실행하지 않았다. [검증·PTH 해시](output_etc/check/solo5_peer_selection_20261101/viewer_verification.json).

### solo5 사람 attention 개수 보정 두 설정 추가 · 학습 미실행

- solo5 기반 `stage2_coupled_push_individual_goal_solo5_countnorm_coupled`·`stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none`의 env/train YAML과 전용 train/test/VNC/eval을 추가했다. 전자는 COUPLED, 후자는 COUPLED/NONE의 query별 유효 key 수 K를 각각 세어 모든 actor/critic 사람 attention 층의 logit에 `−log K`를 적용한다. SELF/BEFORE·GTA/object attention·샘플링/보상/물성·26k 전이는 유지하고 학습 인원은4명·1024환경·8토큰이다. 기본 보정 없음과 기존 checkpoint 계약은 유지하며 새 모드는 저장 graph 계약으로 구분한다. [설정·명령](markdowns/config.md), [정의](markdowns/reference/stage2_coupled_push.md#사람-attention-관계별-개수-보정).
- CPU 검사69개 통과·기존78k 파일 부재3개 건너뜀, 추가 회귀46개 통과(21k 전이 전용1개 선택 제외). 관계 내 선호/SELF:BEFORE 비율 보존·관계별 복제에 대한 전체 비중 보정·masked/빈 관계·셔플/역전파·actor/critic4층 적용·실제26k 전이·실험 간 checkpoint 혼용 거부를 확인했다. 기존 코드와 기본 보정 없음의1/4/8/13명×혼합/전원 공동8조건 출력·입력/파라미터 gradient가 bitwise 일치했다. [검사 로그](output_etc/check/solo5_countnorm_configs_20261101/cpu_tests.log), [회귀](output_etc/check/solo5_countnorm_configs_20261101/regression_tests.log), [기존 동작 비교](output_etc/check/solo5_countnorm_configs_20261101/legacy_verification.json).
- 사용자 요청으로 학습·시뮬레이션은 실행하지 않았다. wrapper의 최종 Python 호출을 캡처하는 모의 검사로 두 실험의 train/test/eval 인자·출력 분리·ordered viewer 연결과 shell 구문만 검증했으며 본학습 출력 폴더를 만들지 않았다. 명령과 문서 링크/diff를 확인했다. [명령 검증](output_etc/check/solo5_countnorm_configs_20261101/command_verification.json).

### 개인 목표 solo5 · ordered5명 공동 PUSH 추세 추가

- 직전 추세 비교의4000/8000/12000 PTH를 복사 고정해 GPU0·3에서5명·물체10개·전원 공동·64환경×seed0·1200step으로 평가했다. ordered·원래 질량/인원별 크기 규칙·평균 액션을 유지했다. 개인 목표 전원 ever는15.6/12.5/17.2%, final은15.6/12.5/15.6%, 공유 중심 final은25.0/17.2/35.9%다. 요청한2/3/4/5/6/8명×epoch 표와 gate 진단도 저장했다.2/3/4명은 기존 일반 혼합42/13/9그룹이며5/6/8명 전원 공동64그룹과 장면 분포가 다르다. [표·재실행](output_etc/check/solo5_ordered_trend64_n5_seed0_20261101/report.md), [5·6·8명 그림](output_etc/check/solo5_ordered_trend64_n5_seed0_20261101/trend_n568.png).
- 3회 CPU/GPU 재채점·JSON 건수·finite·모델 불변·PTH epoch/원본·복사 해시 보존을 확인했다. PTH 사이 초기12개 tensor bitwise 일치·gate 기록 정렬·전원 기하 접촉→native gate를 확인했고 float32/64 ever·공유 중심 frame 판정 차이는0이다. 학습/표준 평가 코드는 수정하지 않았다. [검증](output_etc/check/solo5_ordered_trend64_n5_seed0_20261101/verification.json).

### 개인 목표 solo5 · ordered PTH 추세64환경×1회

- 사용자 지정대로 이후 공동 PUSH 평가는 ordered를 사용하도록 운영 문서에 기록했다. 정식 프로토콜의 비random 제한은 유지하고, 전용 비교 실행기로1000/2000/4000/6000/8000/10000/12000 일반4명 혼합과4000/8000/12000의6·8명 전원 공동을 GPU0·3에서 평가했다. 각64환경·seed0·1200step·원래 질량·평균 액션·물체2N이며 PTH와 relation_config를 복사 고정했다. 학습/표준 평가 코드는 수정하지 않았다. [정의·전체 ever/final·점/영역·재실행](output_etc/check/solo5_ordered_trend64_seed0_20261101/report.md), [그림](output_etc/check/solo5_ordered_trend64_seed0_20261101/trend.png).
- 일반 혼합 공동 전체 final은1000의57.8%→10000의87.5%→12000의78.1%다. 공동2/3/4명 표본은42/13/9그룹이라4명 변화는 특히 주의해 해석한다.6명 전원 개인 목표 ever/final은4000의4.7/1.6%→12000의12.5/12.5%,8명은4.7/1.6%→6.2/3.1%다. 단독 행동별·공유 중심과 gate 전 이동/정지 후 추진 진단도 함께 저장했다. [집계](output_etc/check/solo5_ordered_trend64_seed0_20261101/breakdown.json).
- 13회 CPU/GPU 재채점·JSON 건수·finite·모델 불변·7개 PTH 내부 epoch/해시 보존을 확인했다. 같은 인원/과제 설정의 초기12개 tensor는 PTH 사이 bitwise 일치하며 float32/64 ever·공유 중심 frame 판정 차이는0이다. gate/접촉 기록 정렬과 전원 기하 접촉→native gate도 확인했다. 기존 학습은 보존했다. [검증](output_etc/check/solo5_ordered_trend64_seed0_20261101/verification.json).

### 개인 목표 solo5 11500 · 일반 eval의 gate 이전 이동 진단

- 저장 궤적에서 첫 힘 접촉 후0.5초를 제외하고 지속 전진과 초기 충격 뒤 정지를 분리했다. 이후1초 이상 관찰 가능한2/3/4명 그룹의 gate 전 연속1초·추가10cm 전진은344/443·268/349·208/247개였다. 다른 사람의 힘 접촉 기록이 아직 없는4명 그룹 구간은65개 중 지속 전진2·거의 정지55개이며, 이55개 중 초기5cm 초과 이동 뒤 정지는13개다. 정지 프레임에도 힘 접촉 기록 비율 중앙값96.6%여서 의도적 대기/힘 부족은 구분하지 않는다. CPU 분석만 수행했고 구간 경계·finite·건수 검증을 통과했다. [정의·표·한계](output_etc/check/solo5_11500_gate_wait512_seed012_20261101/persistence_report.md).
- 같은11500 PTH·일반4명/8토큰 혼합·ordered·원래 질량·평균 액션·512환경×seed0/1/2·1200step을 GPU5·6에서 재평가하며 매 step native gate·기하/힘 접촉·공유 상자 속도를 추가 기록했다. 첫 gate 완료 frame을 제외하고 낙상 전까지만 관측해 초기부터/첫 접촉 이후의 최대 XY 변위·목표 거리 감소율·속도를 공동2/3/4명별로 집계했다. 표준 코드·정책·기존 학습은 유지했다. [정의·표·대표 장면·명령](output_etc/check/solo5_11500_gate_wait512_seed012_20261101/report.md).
- 초기부터 gate 전 최대 이동거리 중앙값은2/3/4명0.401/0.621/0.886m, 목표 거리 감소율 중앙값은10.8/16.5/23.4%,10cm 초과 이동 그룹은88.2/92.6/94.6%였다. 첫 접촉 이후만 보아도 이동은비슷하며 구간 속도 중앙값은0.217/0.208/0.236m/s다. goal_saturation으로 전원 기하 접촉보다 먼저 native gate를 완료한 그룹은0/16/21개였다. 초기사람/상자·DOF·과제/목표 tensor bitwise 일치·CPU/GPU 재채점·JSON 건수·finite·모델/PTH 해시 보존과 전원 기하 접촉 참이면 native gate도 참임을 확인했다. 재실행 궤적·낙상은 이전 eval과 일부 달라 필드별 비교를 기록했다. [집계](output_etc/check/solo5_11500_gate_wait512_seed012_20261101/summary.json), [그룹별](output_etc/check/solo5_11500_gate_wait512_seed012_20261101/groups.json), [검증](output_etc/check/solo5_11500_gate_wait512_seed012_20261101/verification.json).

### 개인 목표 solo5 11500 · 사람 간 attention/액션 헤드 피처 진단

- GPU5·6에서11500 PTH·ordered·원래 질량·평균 액션·32환경·seed0·1200step의4명 혼합 및4/6/8명 전원 공동을 진단했다.5step마다 actor human coordination4layer×2head의 동일 Q,K에 대한 softmax(QK/√d)·softmax(QK/√d+bias)를 관계/상태별로 집계하고64차원 head 입력·512차원 hidden·액션을 측정했다. 같은 encoder 출력의 human bias 제거 재계산은 진단에만 쓰고 실제 정책/표준 코드는 유지했다. [표·명령·그림](output_etc/check/solo5_attention_features_n468_20261101/report.md).
- 4명 혼합의 사람쌍 평균 NONE/SELF/COUPLED는22.92/29.64/24.37→20.89/30.78/26.79%이며 BEFORE 표본은 없다.4/6/8명의 공동 상대 전체 비중은73.29/82.37/86.96→75.99/84.43/88.58%다. 마지막 미접촉 상대의 공동 내 비중은41.65/26.07/19.40%(균등 대비1.25/1.30/1.36배)였다. 전원 접촉 정체 구간의4명 대비 head 입력 평균 코사인은6명0.929·8명0.912이며 RMS는 약0.88로 유지됐다. 사람 간 bias 제거의 상대 입력 변화는4/6/8명0.359/0.187/0.150%다. 접촉 전후 액션 변화 차이는 실제 물체/자세 변화가 포함된 관측 결과이며 인식 실패의 인과 증거로 해석하지 않는다. [레이어/head·사건별 상세](output_etc/check/solo5_attention_features_n468_20261101/comparison.json), [미접촉 상대 진단](output_etc/check/solo5_attention_features_n468_20261101/contact_attention.json).
- 4개 rollout CPU/GPU 재채점·JSON 건수·finite·float32/64 판정 차이0·attention 확률 합·모델/PTH 해시 보존을 확인했다. 셔플을 되돌린 수동 coordination 재구성 오차는1e-6 미만이었다.6/8명 trace는 직전11500 기본 토이실험과 필드별로 비교해 별도 기록했다. 기존 학습은 유지했다. [검증](output_etc/check/solo5_attention_features_n468_20261101/verification.json).
- 추가 요청에 따라 GPU6에서 저장 피처만 재생해 원본 QK/√32의 관계별 평균/표준편차·같은 query 내 표준편차·layer/head별 분포를 계산했다.4명 혼합의 NONE/COUPLED 전체 QK는0.424±1.155/0.490±1.236, 같은 query 내 표준편차 평균은0.531/0.389다. 두 관계를 함께 보는 query/head에서 COUPLED 평균 점수가 큰 비율은 원본50.9→bias 후69.0%다. 같은 head/relation의 상수 bias는 관계 내부 logit 차이를 줄이지 않는다는 점을 구분했다. 새 시뮬레이션 없이 기존 확률 평균과 재구성·해시 보존을 검증했다. [표·분포 그림·재실행](output_etc/check/solo5_attention_features_n468_20261101/qk_distribution/report.md).

### 개인 목표 solo5 11500 · 접촉 정체 후 단일 충격 토이실험

- GPU5·6에서6/8명·ordered·원래 질량·평균 액션·32환경·seed0·1200step의 기본/충격 조건을 비교했다. 전원 양손 접촉·속도1cm/s 미만0.5초·목표까지50cm 이상·전원 생존이면30초 이내 한 번만 상자 중심에 목표 방향 J=질량×0.3m/s를 물리1step(16.7ms) 동안 가했다. 이후 외력/액션 노이즈 없이 기존 정책을 유지했다. [설정·명령·결과](output_etc/check/solo5_single_impulse_n68_20261101/report.md).
- 충격 적용은6명17·8명22장면이다. 충격 전 기록 궤적/발동이 일치하는17/20장면에서 직후 속도 중앙값은0.174/0.170m/s였지만,1초 후부터5초 사이 추가50cm 전진은 기본0/0→충격1/0장면이었다. 전원 개인 목표 ever/final은6명4/4→4/4,8명2/2→1/1(각32그룹)이며 새 공유 중심 성공은 없었다.8명2장면은 충격 전 차이로 엄격 반응 비교에서 제외했다. 초기16필드/RNG 일치·실제 질량·body 인덱스·비영 외력 최대1회·가중치/PTH 해시 보존·4개 rollout CPU/GPU 재채점·JSON 건수·finite를 확인했다. 이번 강도의 상자 충격만으로 지속 추진 전환은 대부분 얻지 못했으며 원인을 확정하지 않는다. 표준 코드·기존 학습은 변경하지 않았다. [사건별·검증](output_etc/check/solo5_single_impulse_n68_20261101/comparison.json).

### 개인 목표 solo5 최신11500 · 일반 혼합 ordered 평가

- 요청 시 최신 정기/일반11500epoch/frame1507328000의 모델 일치를 확인하고 정기 PTH·relation_config를 복사 고정했다. GPU5(seed0)·6(seed1/2)에서4명·8토큰·512환경×3seed·1200step 일반 혼합 장면을 평가했다. 기존 비교 실행기로 공동 구역만 ordered 배정하고 원래 무게·평균 액션·distractor·첫 episode 공간 채점을 유지했다. 표준 random 배정 평가와 구분되는 조건이다. [설정·재실행·표](output_etc/check/solo5_11500_ordered_eval512_seed012_20261101/report.md).
- 공동2/3/4명 전원 개인 목표 ever는77.4/77.1/74.1%, final은74.6/73.8/68.6%, 공유 중심 final은77.9/79.0/77.5%다. 단독 PUSH ever/final은77.7/75.4%, 낙상은140/6144명이다. CPU/GPU 재채점·JSON 건수·finite·float32/64 판정 차이0·ordered 순서/목표 연결·모델/PTH 해시 보존을 확인했다. 초기 물리 장면·단독 목표는9500 일반 평가와 bitwise 동일하며 CPU 목표 재계산의 float32 좌표 차이는 최대0.061mm였다. 표준 코드·기존 학습은 변경하지 않았다. [상세](output_etc/check/solo5_11500_ordered_eval512_seed012_20261101/breakdown.json), [채점 검증](output_etc/check/solo5_11500_ordered_eval512_seed012_20261101/verification.json), [배정 검증](output_etc/check/solo5_11500_ordered_eval512_seed012_20261101/assignment_verification.json).

### 비교 viewer 질량 배수 직접 지정

- 요청대로 `view.sh <인원> mass [씬수]`에 `MASS_SCALE`을 연결했다.0.8은20%·0.6은40% 감량이고, 기존 mass80과 새 mass60 preset도 제공한다. 양의 유한값을 생성 단계의 공동 상자 질량·관성에 적용하고 형상/마찰·평균 액션·기본10씬은 유지한다. 표준 학습/평가는 변경하지 않았으며 config/USAGE 명령을 갱신했다.
- GPU6에서 명시적0.6배의 실제 VNC·8명·16step×2씬과0.8배의 headless·8명·16step을 통과했다. 실제 질량/관성·씬 전환·모델 보존·비양수 입력 거부와 초기 사람/상자·관측/목표/CPU·CUDA RNG 일치를 확인했다. GUI/headless의 표시 actor 수 차이는 물리 actor와 구분했다. [검증](output_etc/check/solo5_mass_noise_n68_20261101/viewer_mass_scale_validation.json).

### 감량/노이즈 비교 viewer 순차10씬 재생

- 요청대로 비교 `view.sh`에 세 번째 씬 수 인자(기본10)를 추가했다. 한 모델·VNC에서 장면을 새로 뽑아 순차 재생하고, 씬별 출력/진단을 분리하며 기존 평균/샘플링·물성·씬당1200step을 유지한다. 평가 중 낙상으로 꺼진 관절 stiffness/damping/driveMode를 다음 씬 전에 복원하고, 액션/step/reset wrapper와 hook도 씬마다 정리한다. config/USAGE의 명령·횟수 의미를 갱신했다.
- GPU6·실제 VNC·8명 mass80·16step×2씬의 새 장면/정상 전환·물성·pose·가중치 보존·정상 종료를 확인했다. 별도6명 noise·16step×2씬에서 첫 씬 종료 후 낙상 제어 해제를 강제로 주입하고 두 번째 씬의 원래 관절 속성 복원을 검증했다. [VNC 로그](output_etc/check/solo5_mass_noise_n68_20261101/viewer_repeat_smoke.log), [낙상 제어 복원](output_etc/check/solo5_mass_noise_n68_20261101/repeat_motor_check/motor_restore_validation.json). 기존 학습·실행 중 viewer는 종료하지 않았다.

### 개인 목표 solo5 11000 · 4/6/8명 감량20%/40% 비교

- 동일11000 PTH·ordered·전원 공동·평균 액션·32환경·seed0·1200step·물체2N에서4명20% 감량과4/6/8명40% 감량을 추가 평가했다.4명 두 조건은 GPU5,6/8명40%는 GPU6을 사용하고 기본·6/8명20%는 기존 검증 결과를 재사용했다. 감량40%는 생성 단계에서 전원 공동 actor만 질량/관성0.6배·밀도1.8로 만들며 형상/마찰은 유지한다. [설정·재실행](output_etc/check/solo5_mass20_40_n468_20261101/manifest.json).
- 20→40% 감량의 전원 개인 목표 ever는4명78.1→71.9%·6명31.3→50.0%·8명15.6→37.5%, final은75.0→62.5%·31.3→40.6%·12.5→21.9%다. 공유 중심 final은81.3→75.0%·46.9→53.1%·21.9→34.4%다.40% 감량의 낙상은4/6/8명 모두0명이며,4명은 목표 유지가 저하되어 감량 효과를 인원별로 구분해야 한다. [비교표](output_etc/check/solo5_mass20_40_n468_20261101/report.md).
- 같은 인원의 초기16필드·CPU/CUDA RNG bitwise 일치·실제 actor 질량/관성/마찰·첫 step pose 연속성·모델/PTH 해시 보존·CPU/GPU 재채점/JSON 건수·finite를 확인했다. 새 학습/표준 코드 변경 없이 확인 경로에 기록했으며32환경×1seed 결과다. [상세 검증](output_etc/check/solo5_mass20_40_n468_20261101/comparison.json).

### 개인 목표 solo5 11000 · 6/8명 질량20% 감소·액션 샘플링 비교

- 비교용 `view.sh <인원> <baseline|mass80|noise>`를 추가하고 config/USAGE에 VNC 명령을 기록했다. 같은 PTH·ordered·1환경에서 평가와 같은 첫 episode를 화면에 표시하며 출력은 매번 별도 확인 경로에 저장한다. GPU6·포트6096·실제 VNC/렌더링·8명 mass80·32step 실행과 물성/첫 step pose 연속성/가중치 보존·정상 종료를 확인했다. [실행 로그](output_etc/check/solo5_mass_noise_n68_20261101/viewer_smoke.log).
- 직전11000 PTH를 고정하고 ordered·전원 공동·32환경·seed0·1200step·물체2N에서 기본/질량·관성0.8배/원래 질량+매 step Gaussian 액션을 각각 평가했다. GPU5는6명, GPU6은8명을 실행했다. σ는 저장본의 고정 exp(−2.9)=0.055023이며 기존 clipping·model.eval을 유지했다. 두 변경을 동시에 적용하지 않았고 표준 학습/평가 코드는 유지했다. [설정](output_etc/check/solo5_mass_noise_n68_20261101/manifest.json), [명령·표](output_etc/check/solo5_mass_noise_n68_20261101/report.md).
- 기본→감량→노이즈의 전원 개인 목표 final은6명6.3→31.3→9.4%,8명3.1→12.5→6.3%다. 공유 중심 ever는6명12.5→46.9→50.0%,8명6.3→28.1→53.1%, final은9.4→46.9→15.6%·3.1→21.9→18.8%다. 낙상은6명1→6→6/192명,8명5→10→13/256명이다. 감량은 final 유지, 노이즈는 ever 도달 개선이 두드러졌으며32환경×1seed 진단 결과다. [전체 지표·접촉 진단](output_etc/check/solo5_mass_noise_n68_20261101/comparison.json).
- 기본 조건의 전원 양손 기하 접촉은6명32/32·8명30/32장면에서 한 번 이상 충족했다. 전원 접촉 중 목표 미도달 frame의 속도1cm/s 미만 비율은6명 기본87.4%·감량68.8%·노이즈17.8%,8명85.6%·70.6%·13.8%다. gate 미개방만으로 정체를 설명할 수 없으며, 손 반력은 순 추진력·마찰력 전체와 구분한다.
- 초기16개 물리/관측/CPU·CUDA RNG 필드 bitwise 일치·기본 조건의 이전 지표 재현·실제 actor 질량/관성/마찰·첫 step pose 연속성·Gaussian 표본/액션 선택·모델/PTH 해시 보존·CPU/GPU 재채점/JSON 건수·finite·float32/64 차이0을 확인했다. 실행 중 물성 setter가 첫 step에 상자를 주차 위치로 되돌린 감량 시도는 제외하고, 생성 단계부터 해당 전원 공동 actor만 밀도2.4로 만든 조건을 재평가했다. 초기 pose 재제출만으로 해결되지 않은 짧은 시도도 제외했다. [검증](output_etc/check/solo5_mass_noise_n68_20261101/verification.json). 기존 학습은 종료·재시작하지 않았다.

### 개인 목표 solo5 최신11000 ordered 비교32환경·4/6/8명

- 일반/정기 최신이11000epoch/frame1441792000임을 확인하고 정기 PTH·relation_config를 복사 고정했다. 요청대로4/6/8명 전원 공동 PUSH·물체2N·32환경·seed0·1200step에서 random/ordered를 비교했다. GPU6에서 시작하고 추가 허용 뒤6/8명 ordered는 GPU5에서 병행했다. 표준 평가/학습 코드는 유지하고 비교 전용 harness에서 기존 첫 episode 공간 채점을 사용했다. 정식512환경×3seed 결과는 아니다. [설정·명령](output_etc/check/solo5_ordered_eval32_n468_latest_20261101/manifest.json).
- 전원 개인 목표 ever의 random→ordered 성공 그룹 수는4명23→25/32·6명1→2/32·8명1→2/32, final은23→24·1→2·0→1/32다. 공유 중심 ever는25→26·6→4·2→2/32, final은25→26·3→3·0→1/32다. 낙상은4명6→1/128명·6명11→1/192명·8명10→5/256명으로, 낙상 감소와 다인원 성공률은 구분해야 한다. [표](output_etc/check/solo5_ordered_eval32_n468_latest_20261101/report.md), [개인/공동·상세 검증](output_etc/check/solo5_ordered_eval32_n468_latest_20261101/comparison.json).
- 같은 인원별 물리 상태/DOF/크기/그룹/공유 목표·CPU/CUDA RNG13개 필드 bitwise 일치, 개인 목표/구역 연결·ordered 순서·CPU/GPU 재채점/JSON 건수·finite·float32/64 차이0·모델 가중치/원본·복사본 PTH 해시 보존을 확인했다. 병행 전환 중 실행기 셸 재읽기 오류는 완성된6개 평가를 보존한 재실행으로 정상 종료를 확인했다. 기존 학습 프로세스는 유지했다.

### 공동 PUSH viewer 사람↔구역 배정 옵션

- coupled test/VNC에 `COUPLED_ASSIGNMENT=ordered|distance|optimal`(직접 실행 `--coupled_assignment`)을 추가했다. 좌우 정렬·Hungarian 총 접근 거리 최소화·교차 수→최장 거리→총 거리 최소화를 제공하며 기본 random은 유지한다. optimal은8명까지 전수 비교,9명 이상은 인원 제한 없이 ordered/distance에서 최대2N회 두 사람 교환으로 개선한다. 비용은 초기 root→상자 뒤 담당 구역 앞 직선 기준이며 장애물 우회·실제 낙상 방지를 보장하지 않는다.
- 최종 배치 직후 한 번 배정해 개인 목표·관측·접촉 구역을 함께 연결한다. 기존 random 추첨·물리 배치·RNG·4면 접촉 선택·모델/PTH 계약은 유지하고, 학습/resume·정식 eval_protocol의 비random 변경은 거부한다. config/USAGE/structure를 갱신했다.
- CPU77개 통과(신규17개). GPU6만 사용해 최신 정기11000 PTH로4명·8preset·4모드 및6/8/9명 전원 공동·4모드, 각8환경×16step의 strict 로드·정책/AMP finite·목표/관측·충돌·배정 고정·비정렬 부분 reset·가중치 보존을 확인했다. 모든 모드의 동일 seed 물리 장면·CPU/CUDA RNG가 bitwise 동일했다. ordered test wrapper의1환경32step headless 실행도 완료했다. 실제 GUI/VNC 화면·성공률/낙상 개선은 측정하지 않았다. [검증·재실행](output_etc/check/coupled_assignment_20261101/report.md). 기존 학습 프로세스는 유지했다.

### carry5 reset 배치 계산 최적화

- coupled 공용 배치 함수에서 distractor 한 개의 재시도 동안 경로·크기·사람 위치를 재사용하고, 충돌 경로 구성의 가상 구역 변환을 제거했다. 단독 PUSH 전용 검사는 실제 단독 행만 계산하며 전원 단독이면 추가 인덱싱 복사를 생략한다. 후보 생성/RNG·거리·충돌 여유·배치 순서·재시도 한도·모델 계약은 유지했다. 비PUSH 사람 장애물·공동 전체 상자·ON_TOP 받침/비정렬 환경 인덱스 회귀 테스트3개와 상세/구조 문서를 추가했다.
- GPU6만 사용해 carry5 1024환경을 기존→캐시→물리 조회→단독 선별4단계·19조건으로 비교했다. 배치/목표/관측/graph·CPU/CUDA RNG·재시도 횟수가 bitwise 동일하고, 부분 reset은 다른 환경을 보존했다.3seed 중앙값은 혼합0.595→0.461초(−22.5%), 전원 공동0.178→0.114초(−36.0%), 전원 단독 혼합1.520→1.006초(−33.8%), 전원 단독 PUSH10.383→8.466초(−18.5%),16환경 부분 reset0.116→0.081초(−30.6%)다. 공유 GPU의 reset wall time이며 학습 전체 FPS는 측정하지 않았다. [결과](output_etc/check/carry5_reset_optimizations_20261101/report.md), [실행부](output_etc/check/carry5_reset_optimizations_20261101/run_benchmark.sh).
- CPU86개 통과(기존21k 파일이 없는 전이1개 제외). GPU6·carry5 CHECK 저장본의32환경×8preset×16step 정책/AMP finite·부분 reset·가중치 보존을 확인했다. 전원 단독 PUSH seed2의18번 환경은 기존/변경 모두 후검사에서 경로 간격3.499985456m가3.5m 기준보다 약0.015mm 작게 나오는 기존 경계 사례였다. 조건을 완화하지 않고 별도 기록했으며 최적화 전후 판정도 같다. [검증](output_etc/check/carry5_reset_optimizations_20261101/verification.json), [경계 진단](output_etc/check/carry5_reset_optimizations_20261101/geometry_diagnosis.json). 기존 학습을 종료·재시작하지 않았다.

### 개인 목표 solo5 10000 distractor 제거 비교

- 실행 시 최신 정기10000epoch/frame1310720000을 복사 고정하고 GPU5·6에서 기본/제거 각각4명·8토큰·512환경×seed0/1/2·1200step을 평가했다. 같은 초기 장면에서 과제에 쓰이지 않는 여분 상자5,705개만 장면 밖으로 옮기고 기존 encoder의 object_valid로 관측을 가렸으며 ON_TOP 받침439개는 유지했다. 학습 코드·컨피그는 변경하지 않았다. [설정·재실행 명령](output_etc/check/solo5_10000_distractor_ablation512_seed012_20261101/manifest.json), [평가 전용 실행부](output_etc/check/solo5_10000_distractor_ablation512_seed012_20261101/ablation.py).
- 단독 AT 점 ever/final은81.1/70.7→84.4/72.2%, SIT 점은85.4/74.3→81.8/68.4%다. 공동2/3/4명 개인 목표 전원 동시 final은73.5/69.6/59.6→71.6/67.2/59.3%, 낙상은160→129/6144명이다. 점/영역·ever/final·공유 중심·표준편차·건수는 [보고서](output_etc/check/solo5_10000_distractor_ablation512_seed012_20261101/report.md), [비교 JSON](output_etc/check/solo5_10000_distractor_ablation512_seed012_20261101/comparison.json)에 저장했다.
- 양쪽 초기14개 tensor·유지한 물체/사람/목표의 bitwise 일치, 가린 토큰 값 변경 시 encoder 출력 차이0, 모델/PTH 해시 보존,6개 rollout CPU/GPU 재채점·건수 일치·finite·float32/64 차이0을 확인했다. 상자만 별도로 root setter에 제출한 초기 검사는 비정상 낙상으로 제외했다. 정상 전체 root/DOF 초기화를 재적용한 뒤8환경의 물리 제거 전용 검사에서 최초 사람 step이 기본과 bitwise 동일함을 확인하고 제거 조건을 전부 재실행했다. [검증](output_etc/check/solo5_10000_distractor_ablation512_seed012_20261101/verification.json). 기존 본학습 프로세스는 유지했다.

### 개인 목표 solo5 최신9500 정식 평가

- carry5의 본학습 저장본이 없어 현재 학습 중인 기존 solo5의 실행 시 최신 정기/일반9500epoch/frame1245184000을 확인하고 정기 파일을 복사 고정했다. GPU5(seed0)·GPU6(seed1/2),4명·8토큰·512환경×3seed·1200step 혼합 평가를 완료했다. AT/ON_TOP은 기존 solo5 목표 분포다. [실행 설정](output_etc/check/solo5_9500_eval512_seed012_20261101/manifest.json).
- 단독 PUSH의 ever/final seed 평균은79.39/76.00%, 공동2/3/4명 전원 개인 목표 ever는67.94/63.84/52.03%, final은65.76/62.16/49.68%다. 공유 중심 ever는70.12/68.80/61.34%, final은68.70/66.92/58.71%다.8500 대비 단독 PUSH는 상승하고 공동2/4명 전원 성공은 하락했다. 총1,536장면·6,144명·낙상181명이다. 단독 행동별·공동 인원별 ever/final·표준편차·건수는 [상세](output_etc/check/solo5_9500_eval512_seed012_20261101/breakdown.json),8500 대비는 [비교](output_etc/check/solo5_9500_eval512_seed012_20261101/comparison_8500.json)에 저장했다.
- 세 seed의 초기12개 tensor가8500 평가와 bitwise 동일하다. CPU/GPU·JSON 건수 일치·공유 중심 포함 float32/64 차이0·finite·원본/복사본 checkpoint 해시·기존 본학습 유지 확인. [검증](output_etc/check/solo5_9500_eval512_seed012_20261101/verification.json). 임시 스크립트의 checkpoint 파일명 중복을 바로잡아 재실행했고 모델·학습 설정·평가 코드는 변경하지 않았다.

### 개인 목표 solo5 carry5 별도 컨피그 추가

- 요청대로 `stage2_coupled_push_individual_goal_solo5_carry5` 환경/학습 YAML·train/test/VNC/eval을 추가했다. solo5에서 `solo_carry_goal_distance: [1.0, 5.0]`만 추가하여 단독 AT 목표점·ON_TOP 받침을 운반 상자 기준 거리 U(1,5)m·방향 U(0,2π)로 뽑는다. 사람↔목표 최소1m·높이·충돌/경로·재시도·그 외 거리/모델/보상/26k 전이는 유지하며 기존 solo5·PTH 계약·출력은 분리했다. config 명령 목록 맨 아래와 USAGE/구조/공동 상세/README를 갱신했다.
- CPU148개 통과(없는78k3개 skip·21k 전이1개 제외). GPU6·4명/1024환경 CHECK epoch2/frame262144·finite·actor encoder73 tensor와 입력 RMS 동결·action head6 tensor 갱신을 확인했다. 새 저장본의32환경8preset·16step 정책/AMP finite·개인 목표·거리·충돌·부분 reset·가중치 보존과 test wrapper의 headless4환경32step ON_TOP 재로드를 통과했다.
- GPU5·1024환경×10조건(혼합seed0/1/2·전원단독·전원독립 PUSH·AT·ON_TOP·SIT·CLIMB·전원공동), 총10,240장면 스트레스 검사에서 샘플링 소진0이었다. 모든 장면의 충돌·거리 검사를 통과했고 전원 AT/ON_TOP 각4096개 운반 거리도1–5m였다. 충돌 후보는 기존 재시도로 거부하며 거리/물리 조건을 완화하지 않았다.1024환경의 큰 월드 좌표 float32 오차를 반영해 검사 도구의 수치 허용오차만 조정했다. [스트레스](output_etc/check/coupled_solo5_carry5/stress/stress_validation.json).
- 일반 eval wrapper의8환경·seed0·1200step과 CPU 재채점에서 건수 일치·공유 중심 포함 float32/64 차이0·finite를 확인했다. [전체 검증](output_etc/check/coupled_solo5_carry5/verification.json). 기존 본학습은 유지했으며 새 본학습·실제 GUI/VNC·정규 성능 평가는 실행하지 않았다.

### 개인 목표 solo5 최신8500 기존4명 혼합 평가

- 실행 시 최신 정기/일반 PTH가 모두8500epoch/frame1114112000임을 확인하고 정기 저장본을 복사 고정했다. GPU5(seed0)·GPU6(seed1/2)에서 기존4명·8토큰 혼합 분포의512환경×3seed·1200step 첫 episode 평가를 완료했다. 확장6/8/10명 평가는 이번 실행에 포함하지 않았다. [실행 설정](output_etc/check/solo5_8500_eval512_seed012_20261101/manifest.json).
- 종료 시 점 성공률의 seed 평균은 단독 AT69.58·ON_TOP67.74·SIT65.43·CLIMB91.65·PUSH70.91%, 공동2/3/4명 전원 개인 목표69.02/63.12/56.13%·공유 중심72.36/66.24/60.39%다.2500 대비 단독 PUSH+11.11pp·SIT−12.85pp이며 공동 전원 개인 목표는 모두 개선됐다. 총1,536장면·6,144명·낙상152명이다. ever/final·표본 표준편차·성공/평가 건수는 [상세](output_etc/check/solo5_8500_eval512_seed012_20261101/breakdown.json), 이전 대비는 [비교](output_etc/check/solo5_8500_eval512_seed012_20261101/comparison_2500.json)에 저장했다.
- 세 seed의 초기 장면12개 tensor가2500 평가와 bitwise 동일하다. CPU/GPU·JSON 건수 일치·공유 중심 포함 float32/64 차이0·finite·입력 원본/복사본 해시·기존 본학습 유지 확인. [검증](output_etc/check/solo5_8500_eval512_seed012_20261101/verification.json). 코드와 학습 설정은 변경하지 않았다.

## 2026-10-31

### coupled 인원 상한·물체2배 조건 제거

- 요청대로 coupled 계열의6/8명 whitelist와4명 학습 제한을 제거하고 N≥1·M≥N으로 확장했다. 기본4명·8토큰은 유지하며, 추론은 양쪽 checkpoint 계약의 N/M/물리 후보 수만 정규화하고 resume은 해당 수를 그대로 검사한다.
- ON_TOP 받침이 M−N개를 넘는 장면만 그룹·과제를 다시 뽑고 가능한 장면은 보존한다. 받침의 논리/물리 슬롯을 압축 연결하며 M>2N의 여분 물체와 안 쓰는 물리 후보의 주차도 처리한다. 전원 ON_TOP 고정처럼 가능한 그래프가 없는 요청은 무한 재시도 없이 오류를 낸다.
- CPU81개 통과(없는21k 전이1개 제외). GPU5/6·2500 PTH로1/1·3/3·5/6·9/9·3/8 각4환경과3/4 독립8환경의 엄격 로드·16step finite·부분 reset·구역/목표/물리 매핑·가중치 보존을 확인했다.3/4에서는 압축 받침 ON_TOP도 검사했고,5명·5토큰 test wrapper의 headless64step을 완료했다. 기존4명 bank/RNG/물리 배정의 bitwise 일치·PTH 원본/복사본 해시·기존 학습 보존을 확인했다. [검증](output_etc/check/coupled_variable_counts/verification.json). 다른 인원의 본학습·실제 GUI/VNC·다수 seed 성능은 실행하지 않았다.
- config/USAGE/구조/공동 상세를 갱신하고 임의 N/M의 VNC 명령을 config에 저장했다.

### 기존4명 PTH로6명·8명 공동 PUSH 추론

- 요청대로 coupled 계열 viewer에서6명·12토큰/8명·16토큰을 허용했다. 물리 후보38/52개·상자 local X 폭/질량 N배·동일 구역/개인 목표·배치/접촉을 인원수에 연결했다. 학습은4명·8토큰을 유지하고, 추론 계약은 인원/토큰/물리 후보 수만 정규화하여 기존 모델·RMS를 그대로 로드한다. 공간 평가의 그룹별 집계도 인원수까지 확장했다.
- CPU59개 통과(없는21k 저장본 전이1개 제외). 기존4명 bank 크기/질량·RNG·물리 배정은 HEAD와 bitwise 동일하다. GPU5·2500 PTH·각4환경의6/8명 전원 공동에서 엄격 로드·16step finite·물리/거리/구역/개인 목표·부분 reset·가중치 보존 검사를 통과했다. test wrapper의 headless1환경 실행도6명1200step·8명64step 완료했고 PTH 원본/복사본 해시 보존을 확인했다. [검증](output_etc/check/coupled_6_8_inference/verification.json). 실제 GUI/VNC 화면과 다수 seed 성능은 확인하지 않았다.
- config에 GPU5/6의6/8명 VNC 명령을 저장하고 USAGE/구조/공동 상세를 갱신했다. 기존 본학습은 유지했다.

### 개인 목표 solo5 최신2500 정식512환경×3seed 평가

- 요청대로 실행 시 최신 정기2500 저장본을 복사 고정하고 GPU5(seed0)·GPU6(seed1/2)에서512환경×3회·1200step 첫 episode 정식 평가를 완료했다. 총1,536장면·6,144명이며 실행 인자·원본/복사본 SHA256은 [manifest](output_etc/check/solo5_2500_eval512_seed012_20261031/manifest.json)에 보존했다.
- 공동2/3/4명 개인 목표 전원 동시 ever의 seed 평균은62.67/57.24/49.40%, 공유 상자 중심 도달 ever는66.55/65.08/62.60%다. 그룹 수는778/421/257개이며 낙상179/6144명이다. 단독 행동별 점/영역·공동 인원별 개인/중심 ever/final의 seed 평균±표본 표준편차·총건수는 [상세](output_etc/check/solo5_2500_eval512_seed012_20261031/breakdown.json)에 저장했다.
- 세 trace의 CPU/GPU·JSON 건수 일치·공유 중심 포함 float32/64 차이0·finite·원본/복사본 checkpoint 해시 보존을 확인했다. [검증](output_etc/check/solo5_2500_eval512_seed012_20261031/verification.json). 기존 학습 프로세스는 유지했다.

### 개인 목표 solo5 1500 간단 평가

- GPU6·최신 정기1500 저장본을32환경·seed0·1200step 첫 episode1회로 평가했다. 공동2/3/4명별 개인 목표 전원 동시 성공은11/20·4/5·1/6, 실제 공유 상자 중심→공유 중심 목표 도달은12/20·4/5·4/6이다. 이번 공동 ever/final은 동일하다.
- 혼합 장면의 단독 점 ever는 AT10/12·ON_TOP8/12·SIT5/7·CLIMB3/3·PUSH11/15이며 낙상4/128명이다. 점/영역·final·개인 참여 결과는 [상세](output_etc/check/solo5_1500_eval32_seed0_20261031/breakdown.json)에 저장했다.
- CPU/GPU·JSON 건수 일치·공유 중심 포함 float32/64 차이0·finite·입력 checkpoint 해시 보존을 확인했다. [검증](output_etc/check/solo5_1500_eval32_seed0_20261031/verification.json). 기존 학습을 유지한 소표본 검사다.

### 개인 목표 eval에 공유 상자 중심 채점 추가

- 개인 목표 원본/solo5의 정식 eval에 `shared_center_scores`를 추가했다. 실제 공유 상자 중심→고정 공유 중심 목표3D 거리20cm·구성원 최소1명 유효 조건으로2/3/4명 그룹별 ever/final을 `result.json`·`summary.json`에 저장한다. 기존 개인 목표 전원 동시 성공·학습 보상은 유지한다.
- 새 trace에 실제 물리 중심·공유 목표·ever/final mask를 저장하고 CPU 재채점으로 검사한다. 이전 trace는 균등 분할 토큰/개인 목표 평균을 사용한다. 직전32환경 비교의 solo5 `result.json`·`summary.json`에도 중심 채점을 추가했으며 기존 값은 보존했다.
- CPU44개 통과. GPU6·500저장본·32환경·seed0·1200step 재실행에서 기존 trace12개 필드가 bitwise 동일하고 과제/개인 지표도 같다. 중심 ever/final은2명12/20·3명3/5·4명3/6으로 이전 재채점과 일치했다. CPU/GPU·JSON·float32/64 차이0·finite·checkpoint 해시 보존, 토큰 평균↔실제 중심 최대오차7.63e-6m를 확인했다. [검증](output_etc/check/solo5_shared_center_metric/verification.json). 평가 규약·USAGE·구조·공동 PUSH 상세를 갱신했다.

### goal5 7000·개인 목표 solo5 500 간단 비교 평가

- GPU5·6에서 각각 최신 정기7000/500 저장본을32환경·seed0·1200step 첫 episode1회로 평가했다. 혼합 장면의 단독 역할을 행동별로, 공동 PUSH를2/3/4명별로 나누고 ever/final을 저장했다. 공동은 설정 고유의 전원 동시 성공과 두 설정 공통의 실제 상자 중심→공유 중심 목표20cm 도달을 구분했다. 개인 목표 실험의 실제 중심/목표는 동일 공유 상자의 균등 분할 토큰/목표 평균으로 복원했다.
- 공동 중심 도달 ever는 goal5가2명2/24·3명1/7·4명2/9, solo5는12/20·3/5·3/6이다. solo5의 개인 목표 전원 동시 ever는11/20·1/5·3/6이다. 단독 PUSH는0/8·12/15, 낙상은3/128·4/128명이다. 다른 단독 행동의 점/영역과 final·개인 참여 결과는 [상세](output_etc/check/goal5_vs_individual_solo5_eval32_20261031/comparison.json)에 보존했다.
- CPU/GPU·JSON 건수 일치·float32/64 차이0·finite·두 입력 checkpoint SHA256 보존을 확인했다. [검증](output_etc/check/goal5_vs_individual_solo5_eval32_20261031/verification.json). 학습량·샘플링 분포가 다른 소표본 검사이며 기존 학습 프로세스는 유지했다.

### BEFORE near50 쌍 선택50·접근 최대5m 설정 추가

- `stage2_before_push_near50_pair50_source5` 환경·학습 YAML과 train/test/VNC/eval을 추가했다. 기존 near50에서 쌍 생성 확률0.5·일반 사람→상자1–5m·독립 PUSH 접근2–5m만 적용한다. near50·AT/ON_TOP 목표 배치·PUSH 이동2–10m·보상·모델·loco·4명/1024환경/8토큰/1200step·기본21k 전이는 유지하고 출력·저장본 계약을 분리했다.
- 1024환경 첫 검사에서 고정 주변 배치에 대한 PUSH 후보 소진을 확인했다. 새 설정에서 성공 환경을 보존하고 실패 환경만 기존 장면 재샘플링으로 넘기도록 연결했으며 거리·경로·충돌 조건과 기존 실험 기본값은 유지했다. 전원 독립 PUSH32환경 검사는 재샘플링에 수분이 소요됐다.
- 최종 CPU196개 통과(없는checkpoint4개 skip·21k 전이1개 제외). GPU6·1024환경 CHECK epoch2/frame262144 저장·finite·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신을 확인했다. 32환경7preset의 거리·near·alias·PUSH 경로·부분 reset과4환경32step headless 재로드를 통과했다. AT 목표 최대14.04m·BEFORE ON_TOP 운반 최대9.08m를 관측해 별도 목표 거리 상한을 추가하지 않았음을 확인했다.
- 일반8환경·seed0·1200step 검사와 CPU 재채점의 건수 일치·float32/64 차이0·finite를 확인했다. 기본21k 파일이 없어 검증과 문서 학습 명령에는26k를 명시했다. [전체 검증](output_etc/check/before_pair50_source5/verification.json). config/structure/USAGE/README·BEFORE 상세를 갱신했다. 본학습·GUI/VNC 실화면·정규512환경×3seed 성능 평가는 실행하지 않았다.

### 개인 목표 solo5를 별도 컨피그로 분리

- 요청에 따라 `stage2_coupled_push_individual_goal`의 단독 거리 상한 변경만 되돌렸다. 공동 확률0.5와 개인 목표·접촉 설정은 유지한다. 단독 접근/PUSH 이동 최대5m는 새 `stage2_coupled_push_individual_goal_solo5` 환경·학습 YAML과 train/test/VNC/eval로 분리하고 출력·저장본 계약도 구분했다. 단독 CARRY AT 목표 높이 보존 수정은 유지한다.
- CPU71개 통과(기존21k 전이1개 제외), 원본/변형 환경 차이는 단독 상한 두 값뿐임을 확인했다. GPU6에서 기존5m 검증 저장본을 새 wrapper로 재로딩하여4환경32step 실행을 완료했다. [CPU 로그](output_etc/check/coupled_individual_solo5_split/cpu.log), [실행 로그](output_etc/check/coupled_individual_solo5_split/reload.log). shell 구문·diff 검사 통과. 본학습은 실행·재시작하지 않았다.

### 개인 목표 실험의 단독 접근·PUSH 이동 최대5m

- 새 개인 목표 컨피그에 `solo_source_max_distance: 5.0`·`solo_goal_max_distance: 5.0`을 추가했다. SIT/CLIMB/CARRY AT/ON_TOP의 사람→상자는1–5m, 단독 PUSH 접근·이동은2–5m다. 공동 접근2–10m·이동2–5m와 CARRY 목표 XY 샘플링은 유지한다. 공통 샘플러의 기본값은 유지하고 현재 설정에만 상한을 전달한다.
- 개인 목표 변환을 공동 구성원에게만 적용하여 단독 CARRY AT의 목표 높이 덮어쓰기를 수정했다. 기존 검사의 목표 벡터 비교를 PUSH에 한정하고 단독 접근 범위·AT 목표/받침 높이 검사를 추가했다.
- CPU53개 통과(기존21k 전이1개 제외). GPU6·확률0.5·1024환경 CHECK epoch2/frame262144 저장, finite·encoder/RMS 동결·head 갱신 확인.32환경8preset의 거리·고정 목표·부분 reset 검사를 통과했고 holding_at의 높은 목표61개를 포함해 받침과 목표 높이 일치를 확인했다. [검증](output_etc/check/coupled_individual_solo5/verification.json). 기존 본학습은 재시작하지 않았다.

### 개인 목표 실험의 공동 선택 확률50%

- `stage2_coupled_push_individual_goal`의 `coupled_probability`를0.8→0.5로 변경했다. 그룹 선택 확률 기준이며4명 기대 인원 비율은 공동63.54%·단독36.46%다. 확률 validator와 설정 비교 검사를 갱신했고 기존 coupled/goal5는0.8을 유지한다.
- 관련 CPU44개 통과(기존21k 전이1개 제외), diff 검사 통과. [로그](output_etc/check/coupled_individual_goal/probability50_cpu.log). 아래 GPU 검증은 변경 전0.8 설정의 결과이며 이번에는 학습을 실행·재시작하지 않았다.

### 개인 목표·공통 4면 coupled 실험 연결

- `stage2_coupled_push_individual_goal` 환경/학습 YAML·train/test/VNC/eval을 추가했다. 공동2–5m 이동 벡터를 초기 분할 토큰마다 적용하고 개인 AT 연결·보상·도착 예외·현재 성공 포화를 사용한다. 공유 중심 목표는 물리 경로와 공통 외곽4면 선택에만 사용한다. 목표 방향60° 후보 중 담당 구역까지 그룹 최대 손 거리가 가장 작은 면을 선택하고 전원 gate도 같은 면으로 검사한다. 별도 `aligned` 조건은 제거했다. 기존 coupled/goal5·독립 행동은 유지하고 저장본 계약을 분리했다.
- CPU83개 통과(없는 기본21k 저장본을 요구하는 기존 전이1개 제외). GPU6·1024환경 CHECK epoch2/frame262144 저장·finite·actor encoder73 tensor/입력 RMS 동결·action head6 tensor 갱신 확인.32환경8preset의 거리/개인 목표 연결·고정 목표·부분 reset·정책/AMP finite 검사와4환경32step 실제 렌더를 통과했다. [학습 검사](output_etc/check/coupled_individual_goal/training_validation.json), [장면 검사](output_etc/check/coupled_individual_goal/scene/physics_validation.json), [목표 표시](output_etc/check/coupled_individual_goal/viewer/frame_32.png).
- GPU6·32환경·seed0·1200step 평가를 완료하고 개인/그룹 동시 성공의 ever/final 집계를 연결했다. CPU 재채점·JSON 건수 일치, float32/64 개인/BEFORE 지표 차이0·상태 finite·입력 checkpoint 해시 보존을 확인했다. [재채점](output_etc/check/coupled_individual_goal/eval/general_seed0/rescore.json). config/structure/USAGE/README·상세 설계를 갱신했다. 본학습은 시작하지 않았으며 기존 학습 프로세스는 유지했다.

### PUSH·CARRY AT 목표구 반지름20cm 통일

- 요청에 따라 모든 설정의 AT 목표구를 생성 시 반지름20cm로 통일하고 PUSH 와이어 구를25cm에서20cm로 줄였다. 바닥 링은35cm로 유지하며 reset 중 scale 변경은 추가하지 않았다.
- URDF 시각 반지름0.1m×생성 scale2.0=0.2m·Python 구문·diff 검사를 통과했다. 이번 크기 조정의 viewer 실행 검증은 하지 않았다.

### PUSH viewer 표시 롤백 및 선 표시 재구현

- 사용자가 viewer에서 물체가 날아가는 비정상 동작을 보고해 직전 PUSH 표시 이식 전체를 롤백했다. reset 시 marker scale을 변경하는 코드·표시 모듈·호출부를 제거하고 기존 목표구 표시와 문서를 복구했다. 짧은 관측/렌더 검사만으로 정상 동작을 판단한 기존 검증은 불충분했다. 기존 학습·컨피그 변경은 보존했다.
- 이후 기존 목표구·actor·scale·reset을 유지하고 `add_lines`로 PUSH 목표의 바닥 링·세로선·외곽 와이어 구만 덧그리도록 다시 구현했다. 공유 목표 선은 노란색이며 일반 AT에는 추가 선이 없다. 공통 reset 파일은 변경 전과 동일하다.
- GPU5의 실제 viewer에서5500 정책·4환경·동일 seed의 기존/선 표시 실행을 각각1200step 비교했다. 전체 actor root state·정책 관측·행동이 bitwise 동일하고 최대 차이0·finite다. [궤적 비교](output_etc/check/push_goal_viewer_safe/trajectory_comparison.json). 일반 AT·독립 PUSH·2명 두 그룹·4명 공동의 선 표시만 호출하는 것도 확인했다. [표시 검사](output_etc/check/push_goal_viewer_safe/draw_only_verification.json).

### Coupled goal5 5500 간단 평가

- 최신 정기5500 저장본을 GPU5·32환경·seed0·1200step 첫 episode1회로 평가했다. 공동 PUSH의 실제 상자 목표 도달은2명6/24(25.0%)·3명1/7(14.3%)·4명1/9(11.1%)다. 혼합 장면의 독립 역할은 SIT1/1·CLIMB2/2·AT5/6·ON_TOP5/6·PUSH1/8이며 점/영역 결과는 같다. 사람 번호별 공동 참여 성공은 별도 JSON에 저장했다. [상세](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/breakdown.json).
- CPU 재채점과 GPU·JSON 건수가 일치하고 float32/64 성공 차이0·상태 finite·입력 checkpoint 해시 보존을 확인했다. 낙상1/128명이다. 소표본 검사이며 본학습 프로세스는 유지했다. [검증](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/verification.json).
- 같은 궤적의1200step final은 공동2명4/24·3명0/7·4명0/9, 독립 AT4/6이며 나머지 독립 행동은 ever와 같다. 이전 낙상자를 제외하고 final이 ever에 포함됨을 확인했다. [final 상세](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/final_breakdown.json).

## 2026-10-30

### 공동 PUSH goal5 컨피그 추가

- `stage2_coupled_push_goal5` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. 공동 PUSH는 사람→실제 상자 중심2–10m·상자→목표2–5m, 독립 PUSH는 두 거리2–10m다. 기존 coupled YAML은 원래 설정으로 복구했으며 사람 초기 배치·그 외 환경/학습 값은 유지했다. 새 실험의 기본 전이는 제공26k이고 출력·저장본 계약은 분리했다.
- CPU28개 통과(공통 모델26·새 설정 로딩2, 기본21k 파일이 필요한 기존 전이1개 제외). GPU5의1024환경 CHECK가 epoch2/frame262144 저장 후 정상 종료했다. 모델/정규화/optimizer finite·actor encoder73 tensor와 입력 RMS 동결·action head6 tensor 갱신을 확인했다. [학습 검사](output_etc/check/coupled_goal5/training_validation.json).
- 전용 test wrapper로 저장본을4환경·32step·4명 공동 PUSH로 headless 재로드했다. 셸/Python 구문·새 YAML/문서 경로를 확인하고 config/structure/USAGE/README·공동 PUSH 상세를 갱신했다. 장기 본학습·수렴 성능·실제 GUI/VNC 화면은 검증하지 않았다.

### 모션 데이터 연결 복구

- loco·sit·climb·carry 모션은 `/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/`의 각 원본으로, PUSH 모션은 `/home/hwanhee/ksh/stage1_2_jointpush/tokenhsi/data/jointpush_motions`로 심링크를 연결했다. 외부 원본은 수정하지 않았다.
- 현재 PUSH 모션 YAML의 파일 참조161개가 모두 존재함을 확인했다. 학습은 실행하지 않았다.

### 공동 PUSH 거리2–5m 분석

- 공동 PUSH의 두 XY 거리를2–5m로 줄이는 안을 검사했다.32환경의41개 공동 그룹에 각각104만 후보를 검사하고, 실패4환경은 위치·회전 구간 검사를 통해 고정된 사람·상자 크기에서 거리/뒤쪽/여유 조건을 동시에 만족할 수 없음을 확인했다. [분석](output_etc/check/coupled_distance_analysis/summary.json).
- 공통 Stage2 CPU23개 통과(기본21k 파일이 필요한 전이1개 제외). 제공26k의165 tensor 전이와 입력 RMS 로드를 확인했다. 이후 사람→상자2–10m·상자→목표2–5m 조합은32환경의 혼합/4명 공동/독립 PUSH reset·짧은 시뮬레이션을 통과했다. 사람 초기 배치는 변경하지 않았고 기존 coupled YAML은 두 거리2–10m로 복구했다.
- 최초 검증은 GPU1에서 실행 후 종료했다. 이후 사용자 지시에 따라 GPU5·6만 사용하며 GPU5에서 같은 배치 오류를 재현했다. [로그](output_etc/check/coupled_distance_2to5/scene_gpu5.log), [초기 배치](output_etc/check/coupled_distance_2to5/scene/reset_failure.json).

## 2026-10-29

### PUSH viewer 목표구 확대

- PUSH 포함 설정의 viewer/VNC/영상 목표구를 반지름3cm에서20cm로 확대했다(지름 약6.7배). 공통 marker 생성부의 표시 scale만 변경했으며 목표 좌표·성공 판정·학습 설정과 비PUSH 설정의 marker 크기는 유지했다.
- coupled1환경의 동일 카메라에서 기존/확대 목표구를 offscreen 렌더링해 가시성을 확인했다. marker scale 변경 전후 정책 관측이 bitwise 동일하고 짧은 시뮬레이션 관측/보상 finite를 확인했다. [확대 화면](output_etc/check/push_goal_marker/render/after.png), [검증](output_etc/check/push_goal_marker/render/verification.json). USAGE에 표시 크기를 기록했다.


### PUSH26k 평가와 Stage2 제로샷 검증

- `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`를 검사했다.5과제/4family·165 tensor·모델/RMS finite를 확인했고 두 입력 PTH의 실행 전후 SHA256은 동일하다. 제공78k는 기존 기준78k와 해시가 같고 새 pth 경로의4환경32step headless 재로드를 완료했다.
- 26k Stage1 PUSH는512환경×3seed·900step 일반3회/1명 행동별 단독15회를 완료했다. 점/영역5과제 macro는 일반 **84.55/88.32%**, 단독 **82.20/86.95%**다. PUSH 점 성공률은 일반87.12%·단독88.61%로21k보다 높지만 다른 과제의 하락으로 전체 점 macro는 낮았다.21k와18개 초기 장면의9개 필드가 bitwise 일치한다.
- near50·coupled는26k tensor165개를 그대로 복사하고 공통 coordination identity/Linear bias0인 제로샷 저장본을 각각 만들었다. 추가 학습0회·두 모델 tensor 동일을 확인했다. 각16환경·4명·1200step 실행과4환경32step headless viewer 재로드를 완료했다. BEFORE ON_TOP 쌍/4명 공동 PUSH viewer 명령을 USAGE에 추가했다. 관찰용 파일에는 optimizer가 없고 기본 YAML 전이21k는 변경하지 않았다.
- 총20개 trace의 CPU/GPU·JSON 건수 일치, float32/64 ever/simultaneous 차이0·finite를 확인했다. [리포트](output_etc/check/push_26k/report.md), [검증](output_etc/check/push_26k/verification.json). Stage2 결과는 작은 제로샷 실행 검사이며 학습된 Stage2 성능·실제 GUI 화면 검사는 아니다. 기존 원본/프로세스는 보존했다.


### BEFORE PUSH near50 거리 변형 추가

- `stage2_before_push_near50` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. `stage2_before_push`의 mix226 대신 참고 설정의 `relative_sum` near50 거리만 적용했다. 기존 모든 YAML은 보존하고 공통 모델·PUSH/독립 행동 비율·4명/1024환경/8토큰/1200step·loco·21k 전이는 유지했다. RSI433과 attention mask는 이식하지 않았다.
- near 모드는 BEFORE 쌍의50%이며 SIT/CLIMB는 후행→받침 거리가 선행→받침보다 작고, ON_TOP은 후행→받침+후행→payload 거리 합이 선행→받침보다 작다. 나머지50%는 original 샘플링이다. validator·viewer/eval의 새 이름/모션/task 경로·checkpoint 혼용 거부를 연결했다.
- CPU313개 통과.1024환경 CHECK epoch2/frame262144 저장·모델/RMS/optimizer finite·actor encoder73 tensor/입력 RMS 동결을 확인했다.32환경5preset에서 near 부등식·PUSH 경로/물리·alias 관측·부분 reset·가중치 불변을 확인했고4환경32step headless viewer와16환경×seed0·1200step 일반 평가를 완료했다. CPU/GPU·JSON 건수 일치, float32/64 ever 차이0·finite다. [검증](output_etc/check/before_push_near50/verification.json).
- config/structure/USAGE/README·BEFORE PUSH 상세에 설정과 명령을 갱신했다. 장기 본학습·512환경×3seed 성능·GUI/VNC 실화면은 실행하지 않았다.


### 공통 Stage2 alias 전처리와 coordination 초기화 수정

- 공통 `Stage2PushBuilder`의 alias 처리를 config 플래그 대신 task packet의 BEFORE 의존 관계로 결정하도록 통일했다. encoder용 복사본만 변환하고 복제 물체 token feature/pose·원본 packet·coordination relation은 보존한다. coupled·독립 packet은 그대로 통과한다. 기존4과제 BEFORE 구현은 유지했다.
- 새 coordination의 모든 Linear bias를 원본처럼0으로 초기화했다. 초기 출력 identity와 Stage1 PUSH165 tensor 전이는 유지하며 기존 학습 PTH 로드 시 저장된 bias를 덮어쓰지 않는다. coupled 제로샷 PTH를 같은 경로에 새 초기화로 재생성하고 이전 관찰용 저장본은 `output_etc/check/stage2_common_fix/previous_zeroshot.pth`에 보존했다.
- CPU305개 통과. BEFORE SIT/CLIMB/ON_TOP·두 쌍·coupled·solo의 같은 가중치/입력에서 config에 무관하게 actor/critic 출력 bitwise 일치, 기존 encoder 동작·복제 feature/pose·원본 packet 보존을 확인했다. 실제 GPU BEFORE 관측과 기존 학습 가중치의 교차 실행도 action/critic 최대 차이0이다.
- 두 설정 모두4명·1024환경 CHECK epoch2/frame262144 저장, 모델/RMS/optimizer finite·actor encoder73 tensor/입력 RMS 동결을 확인했다. BEFORE32환경5preset의 alias 관측/부분 reset/finite 검사와 재생성 제로샷4환경·32step headless 재로드를 완료했다. 이전/새 제로샷의 coupled·solo 출력은 동일하다. [검증](output_etc/check/stage2_common_fix/verification.json). 구조/설정 상세 문서를 갱신했고 장기 성능·실제 GUI 화면은 검증하지 않았다.


### Coupled 제로샷 viewer 저장본

- 제공 Stage1 PUSH21k의 기존165 tensor를 그대로 복사하고 공통 Stage2 coordination을 identity로 초기화한 `output_etc/check/coupled_zeroshot/Stage2CoupledPush_ZeroShot_Stage1_21000.pth`를 만들었다. 추가 optimizer step0·tensor 일치·identity 초기화를 확인했다. 원본 PTH는 보존했다.
- `TASK_GRAPH=all_coupled`로4환경·32step headless test wrapper 재로드를 완료했다. 실제 GUI 화면은 실행하지 않았다. 제로샷 viewer 명령을 USAGE에 추가했고 전이 기록은 `output_etc/check/coupled_zeroshot/transfer.json`에 있다.


### 공통 Stage2 PUSH 모델과 coupled 이식

- `stage2_coupled_push` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. 원격 `stage1_2_jointpush_st`의 coupled straight·밀도3·인원별 공동 상자/구역·그룹 gate·goal saturation을 이식했다. 독립 행동과 AMP 혼합은 현재 Stage1 PUSH 설정(AT25/ON_TOP20/SIT10/CLIMB15/PUSH30)을 읽도록 바꾸고 미사용 Stage2 RSI/adapter 스택은 가져오지 않았다.
- `stage2_before_push`와 coupled는 동일 `Stage2PushBuilder`·관측1272/AMP1330·5과제·NONE/SELF/BEFORE/COUPLED4종 사람 relation·6N task/group suffix를 사용한다. tensor 이름·크기가 같고 Stage1 PUSH21k의165 tensor를 그대로 전이한다. 기존 BEFORE PUSH의5N 관측 검사 저장본은 새 모델에 resume하지 않는다.
- 사용자 기준대로 Stage1은900step을 유지하고 모든 Stage2 학습/일반 평가/viewer 기본 시간을1200step으로 변경했다. 기존4과제 BEFORE 모델은 보존했다. coupled 평가의 PUSH 성공은 가상 구역이 아닌 실제 전체 상자 중심에서 계산한다.
- CPU292개 통과. 두 Stage2 모두4명·1024환경 CHECK가 epoch2/frame262144 저장 후 종료했고 coupled resume은 epoch3/frame393216까지 완료했다. actor encoder73 tensor/입력 RMS 동결·action head6 tensor 갱신·모델/RMS/optimizer finite를 확인했다.
- coupled32환경8preset·BEFORE32환경5preset의 배치/구역/AMP/부분 reset/가중치 불변을 확인했다. 두 설정 모두 일반32환경×seed0·1200step 및 headless viewer4환경·32step 재로드를 완료했다. CPU/GPU·JSON 건수 일치, float32/64 ever/simultaneous 차이0·finite다. [검증 보고](output_etc/check/coupled_integration/report.md), [전체 결과](output_etc/check/coupled_integration/verification.json).
- config/structure/USAGE/README·BEFORE 상세와 [공통 Stage2/coupled 상세](markdowns/reference/stage2_coupled_push.md)를 갱신했다. Stage1 YAML·원본 checkout·외부 PTH·기존 프로세스는 보존했다. 장기 본학습·정규512환경×3seed 성능·GUI/VNC 실화면은 검증하지 않았다.


### Stage2 BEFORE solo PUSH 추가

- `stage2_before_push` 환경/학습 YAML과 전용 train/test/VNC/일반 eval을 추가했다. 독립 역할만 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%를 사용한다. 기존 0.8 쌍 구성·선행 AT·크기 호환 후행 SIT/CLIMB/ON_TOP 균등 선택·mix226·공유 alias·AT gate/progress는 유지한다. 기본4명·1024환경·8토큰·900step·모두 loco다.
- PUSH 전용 큰 상자 bank를 추가해 기본20개 물리 후보에서 배정하며 일반 distractor/BEFORE 받침으로 섞이지 않게 했다. Stage1 PUSH의 밀도3·167종·cone90·거리·접촉 가중치를 적용하고 다중 사람 경로/활성 장애물 검사를 추가했다. 정책5과제·AMP4family를 기존 BEFORE coordination에 연결해 기본21k PUSH에서 tensor165개를 그대로 전이한다. 기존5개 환경/학습 YAML은 변경하지 않았다.
- CPU280개 통과. 최종 CHECK1024환경은 epoch2/frame262144, resume은 epoch3/frame393216 저장을 완료했다. actor encoder73 tensor·입력 RMS 동결, action head6 tensor 갱신, 모델/정규화/optimizer finite를 확인했다. [학습·재개](output_etc/check/before_push/training_validation.json).
- 32환경7개 preset에서 물체 alias/숨김·거리·밀도/마찰·PUSH 경로·AT progress·AMP·부분 reset·가중치 불변을 확인했다. 최종 저장본 일반32환경×seed0·900step과 headless viewer4환경·32step 재로드를 완료했다. 평가 CPU/GPU/JSON 건수 일치·float32/64 ever 차이0·finite를 확인했다. [검증 보고](output_etc/check/before_push/report.md), [전체 검증](output_etc/check/before_push/verification.json). 이는 실행 검사이며 장기 학습 성능·512환경×3seed 평가·GUI/VNC 실화면은 검증하지 않았다.
- config/structure/USAGE/README와 [설정 상세](markdowns/reference/stage2_before_push.md)를 갱신했다. 원본 checkout·외부 checkpoint·기존 실행 프로세스는 보존했다.


### Stage1 PUSH 행동 비율 변경

- 사용자 지정대로 `stage1_push`의 새 학습 크기/행동 prior를 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%로 변경했다. 검증기의 허용 계약·샘플링 테스트·현재 PUSH/USAGE/평가 문서를 갱신했다. 과거 저장본의 분포와 평가 결과는 보존하며21k PTH의 기본 viewer/일반 평가는 저장된PUSH50%를 복원한다.
- CPU35개 통과:8만 사람 슬롯의 크기 조건부 샘플링 빈도·합계1·설정 검증·이전 저장본 복원/호환. [검증](output_etc/check/push_sampling_30/tests.log), [설정](output_etc/check/push_sampling_30/config_check.json). 비율 변경 검증에 GPU 학습/평가는 실행하지 않았다.


### Stage1 PUSH 통합과900step 통일

- `stage1_push` 환경/학습 YAML·train/test/VNC·일반/1명 단독 eval을 추가했다. scratch prior는 SIT7.5/CLIMB18.75/AT24.375/ON_TOP24.375/PUSH25%로 기존 네 과제 상대 비율을 유지한다. 정책은5과제·AMP4family로 확장하고 기존4과제/3family 기본값은 보존했다. PUSH167종·밀도3·cone90·접촉 RSI bank/모션을 이식했으며 모션21개와21k PTH는 외부 원본에 심링크로 연결했다.
- 학습·일반/단독 평가·viewer 기본 시간을900step으로 통일하고 시간 종료의 `-1` 경계를 수정했다. 새5과제 macro와 기존4과제 macro를 구분한다. PTH 옆 sidecar가 없으면 내부 저장 env/sim을 읽는다. 제공21k 일반 평가는 학습 당시 PUSH50% 분포를 복원하고 새 scratch는PUSH25%다. 서로 다른 학습 계약의 resume는 거부한다.1명 PUSH와1명 ON_TOP의 물체/받침 인덱스, 여섯 번째 RSI 통계 축을 일반화했다.
- 제공 `Stage1JointPushDensity3Cone90Finetune_00021000.pth`로512환경×3seed 일반3회·단독15회, 모두900step을 완료했다.5과제 macro 점/영역은 일반 **85.84±1.07 / 88.80±0.81%**, 단독 **83.32±0.51 / 88.78±0.32%**다. PUSH는 일반 **81.78±1.93%**, 단독 **81.51±1.47%**다.18개 trace CPU/GPU·JSON 건수 일치, float32/64 ever 차이0, 상태 finite를 확인했다. [리포트](output_etc/check/push_900/report.md), [검증](output_etc/check/push_900/verification.json).
- 이전78k1200step trace의 앞900frame을 별도로 재채점해 같은 제한 시간의 비교를 추가했다. 기존 원본 trace/보고서는 보존했다. 기존 네 과제의 단독 평균은21k에서 낮아졌으며 수치와 비교 조건은 리포트에 기록했다.
- CPU259개 통과. 실제78k→PUSH 전이 후 기존 네 과제 actor/critic 출력 보존, 원본 브랜치 정책·task encoder·clean scene 관측 코드 일치를 확인했다. Stage1/BEFORE 기존 PTH4환경·32step 재로드도 통과했다. PUSH scratch CHECK는2048환경으로 RSI5517 profile 준비 후 epoch2/frame262144를 저장했고 resume은 epoch3/frame393216까지 진행했다. 모델/RMS finite, scratch 시작·PUSH25%·900step 계약을 확인했다. [학습/재개](output_etc/check/push_900/training_validation.json). 원본 PTH/코드/기존 프로세스는 보존했으며 장기 본학습·GUI/VNC 실화면은 실행하지 않았다.


### 일반 평가와 Stage1 전용 단독 평가 분리

- 일반 `*_eval.sh`는 checkpoint 저장본의 학습 인원·물체·크기·과제/BEFORE 배치를 유지한다. Stage1만 `stage1_single_eval.sh`로1명·행동별 독립 평가를 제공한다. 단독 AT만 기존9종 크기, 나머지는 학습 크기/동적 물리를 따른다. viewer native 성공 진단은 `viewer_diagnostics.py`로 이름과 결과 태그를 분리했다.
- 정식 평가는 loco·정확히1200step·독립 seed0/1/2×512환경 첫 episode, 같은 궤적의 점/영역 ever·seed 평균/표본 std·macro다. 개인 낙상 frame까지 채점하고 해당 관절 drive를 해제하며 다른 사람은 계속 수행한다. BEFORE의 개인/follower 개인·둘 다 ever·동시 성공을 별도 집계한다. 학습 reward/종료 조건은 유지했다. viewer 기본 horizon도1200으로 맞췄다.
- Stage1 원본78000으로 일반3회(2명·4물체), 단독12회(행동별1명)를 완료했다. macro 점/영역은 일반 **95.17±0.84 / 97.17±0.18%**, 단독 **96.58±0.26 / 98.83±0.40%**다. [리포트](output_etc/check/eval_unified/stage1_78k_report.md). 15개 trace CPU float64/GPU·JSON 건수 및 float32/64 ever가 전부 일치했고 상태 finite·원본 checkpoint 해시 보존을 확인했다.
- CPU242개 통과. BEFORE 세 설정은 검사 저장본으로16환경·4명·1200step 실행/재채점(낙상8/10/15명, 각각24pair), viewer는4환경·32step의 새 진단 경로를 확인했다. Stage2 본학습 성능과 GUI/VNC 화면은 평가하지 않았다. [검증](output_etc/check/eval_unified/verification.json), [BEFORE](output_etc/check/eval_unified/before_validation.json). 새 규약·명령·코드 위치를 evaluation_single/config/structure/USAGE에 갱신하고 이전 단독 규약은 legacy에 보존했다.


### 네 실험 이름 통일

- 환경/학습 YAML·전용 train/test/VNC12개·실험/출력 이름·테스트·현재 문서를 `stage1`, `stage2_before_nonemask_rsi433_near50`, `stage2_before`, `stage2_before_rsi433`로 통일했다. Stage1 보조 구현은 `stage1_config.py`, `stage1_geometry.py`로 바꿨으며 기본 전이 링크는 `checkpoints/Stage1_00078000.pth`다. 외부 원본 파일과 과거 결과 경로는 보존했다.
- 기존78000 및 이전 BEFORE checkpoint의 reward 태그·RSI 경로를 읽기 경계에서 변환한다. 보상·관측·모델 tensor·실험 간 혼용 거부는 유지한다. 기존 네 저장본을 새 test wrapper로4환경·32step headless 재로드해 정상 종료했고 네 모델의 전체 tensor·actor/critic 출력은 변경 전과 bitwise 동일했다. [재로드](output_etc/check/config_rename/smoke.json), [모델](output_etc/check/config_rename/model_equivalence.log). CPU224개 통과. 셸/Python/YAML·문서 링크·이전 실행 이름 잔여 검사 결과는 `output_etc/check/config_rename/`에 있다. GUI/VNC 실화면과 본학습은 실행하지 않았다.

### 네 컨피그와 학습·추론 경로 정리

- `stage1_v2_fixed`, `stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200`, `stage2_before_ground_mix226_atprogress_objalias_1200`, `stage2_before_ground_mix226_atprogress_objalias_rsi433_1200`만 유지했다. 나머지21쌍 YAML·63개 wrapper와 공동 운반·공유 받침·adapter·full RSI·carry50·단일인 태스크·전처리·미사용 에셋/모듈을 제거했다. 기존 파일322개를 삭제했으며 과거 문서는 legacy에 보존했다. [변경 목록](output_etc/check/four_config_cleanup/change_manifest.json).
- BEFORE의 공통 기하·배치·물리 검사·native 후행 RSI를 분리하고 모델을 공통 정책의 직접 확장으로 정리했다. 공통 환경의 관측·reset·모션/AMP를 별도 mixin으로 분리했다. 파라미터 이름·관측·보상·거리·초기화·동결/학습 범위를 유지했다.
- 전용12개 wrapper는 공통 `launch.sh`를 사용한다. conda 기본값은 모두 `tokenhsi`, seed는42, test 기본은 viewer다. GPU는 한 번 지정하며 기존 MPS만 연결한다. 설정 로더는 새 학습의 native YAML과 추론/resume의 저장 env/sim을 같은 경로에서 처리한다. `ENV_CONFIG`로 명시적 대체가 가능하며 fixed 임시 YAML 생성·실험 이름별 중복 분기를 제거했다.
- Stage1 78000은 `checkpoints/Stage1V2Fixed_00078000.pth` 심링크로 연결했다. 네 모션 데이터 디렉터리도 공유 원본에 연결했다. 원본 checkout·가중치·기존 학습 PID3432492는 보존했다. 확인 산출물·RSI 캐시·pytest 임시 파일은 `output_etc/check/`에 둔다.
- CPU214개 통과: 네 설정/계약 혼용 거부·실제78000 전이·설정 복원·wrapper/conda 선택·GTA·보상·거리·mask·RSI/fallback·재로드. [CPU](output_etc/check/four_config_cleanup/cpu.log). 네 모델의 전체 tensor·actor/critic 출력·계약이 정리 전과 bitwise 동일했다. [모델](output_etc/check/four_config_cleanup/model_equivalence.log). 네 설정의 전체 reset2회·부분 reset1회에서 관측·사람/상자 상태·크기·목표·물체 연결의 최대 차이0을 확인했다. [reset](output_etc/check/four_config_cleanup/reset_equivalence.json). 후행 RSI sampler의 출력·RNG도 동일했다.
- GPU0/MPS에서 fixed2명·2048환경과 BEFORE4명·1024환경의 CHECK가 각각 epoch2/frame262144 저장 후 종료했다. fixed native RSI6413 profile을 생성했다. BEFORE 세 설정은 source78000·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/scalar finite를 확인했다. mix226 resume은 epoch3 저장 후 종료했다. [학습/재개](output_etc/check/four_config_cleanup/training_validation.json).
- 실제78000 fixed headless 추론과 BEFORE 전용 test의4환경·32step 저장본 재로드를 확인했다. BEFORE32환경의 loco/mix·ON_TOP pickup/carry·단독 등14개 구성에서 alias 관측/pose·활성 물리 수·거리·AT progress·AMP·부분 reset·가중치 불변을 확인했다. [물리 검사 진행 결과](output_etc/check/four_config_cleanup/physics.log), [wrapper](output_etc/check/four_config_cleanup/wrapper_validation.json). 셸/Python/YAML·현재 문서 링크·4쌍 YAML/12개 wrapper·미사용 import/실험 참조 제거를 검사했다. VNC 실행 의존성이 없어 실제 GUI/VNC 화면과 장기 성능은 검증하지 않았다. 본학습은 시작하지 않았다.

## 2026-10-28

- 코드 정리용 독립 복사본을 만들었다. 원본과 복사 범위·초기 검증은 `code_copy_manifest.json`과 [당시 changelog](markdowns/legacy/pre_four_configs/changelog.md)에 보존한다.
