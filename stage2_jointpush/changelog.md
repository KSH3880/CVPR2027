# Changelog

최신 변경부터 기록한다. 현재 실행법은 [config.md](markdowns/config.md), 코드 위치는 [structure.md](markdowns/structure.md)를 참조한다. 실행 중인 GPU/PID는 이 파일에 고정하지 않고 실제 프로세스로 확인한다.

## 2026-10-28

### 실험 폴더를 소속 output/runs로 추가 통합

사용자 요청으로 기존 runs 실험·평가·진단을 이 프로젝트 output/runs에서 접근하도록 정리했다. 사용 중 경로는 이동하지 않고 새 위치에서 연결했으며 완료 경로는 역방향 호환 링크를 유지했다. 전체81경로(74이동·7활성연결) 및 일반파일20,632개 inode/크기/mtime 확인. 기존 프로세스 생존 확인. 이전 cleanup 보관 폴더 부재로9개 링크 미해결. 정확한 목록·현재 기준은 ../docs/OUTPUT_LAYOUT.md 및 ../runs/project_output_migration_final_20261028.json.

### JointPush 결과 참조 및 본학습 기본 경로 통일

koo_push 전용 학습 wrapper와 viewer의 결과 참조를 ../jointpush/output으로 통일하고 config/structure의 경로를 갱신했다. 원래 runs 경로는 호환 링크로 접근 가능하다. shell 구문·실제 결과 경로 검증. 실행 중인 작업과 원본 KSH 코드는 유지했다.

## 2026-10-22

### koo_1021 Stage1BasePush 기반 Partial Joint Push Stage2 실행

- 요청을 Stage1 resume로 잘못 해석해 실행한 자체 프로세스만 사용자 정정에 따라 종료했다. 업로드7500회 Stage1의165tensor/RMS를 복사하고 새 coordination104tensor를 identity 초기화하는 전용 Partial Stage2를 추가했다. actor encoder/RMS 동결, 새 optimizer/epoch0,4명·2048환경·8object token·24physics bank, 기존 partial 분할·보상/CPA0.5·Y1.8·loco RSI·원본 conditional AMP(PUSH family/loco50+push50)를 유지한다. source는 process-local adapter로 연결하며 원본 파일은 수정하지 않는다.
- 첫2048환경 시작에서 다른 상자에 더 가까운 spawn이 검증에 걸려 optimizer 전에 종료됐다. 그룹 배정을 유지하며 모호한 surround 후보만 재추출하도록 local adapter를 보완했다. CPU128환경24step finite 통과 후 GPU5 재실행, coordination gradient229.81·2epoch 후 encoder/RMS불변/coordination변화·startup checkpoint 저장을 확인했다. 기존 GPU5 학습/MPS PID 생존, 총사용약57GiB/여유40GiB. 현재 실제 실행은 runs/jointpush/koo_partial_stage2_20261022/latest_training.json 참조. viewer GUI는 미검증이다.

### Stage1BasePush 7500회 GPU5 resume 실행

- 업로드된 Stage1BasePush_00007500.pth의 CPU 학습 계약 통과 후 GPU5 공유 MPS 실행을 시작했다. 로컬 runtime의 명시적 공유 opt-in은 여유32GiB·기존 Python1개/MPS server1개·MPS pipe 확인을 모두 요구하며 기본 점유GPU 차단은 유지한다. 외부 source/기존 학습/MPS daemon은 변경하지 않는다.
- prepare의 상대 checkpoint 경로를 source 검사에서 cwd가 바뀌기 전에 절대경로로 고정했다. 전용 tmux·로그·실행설정은 runs/jointpush/koo_1021/latest_training.json 참조. TensorBoard info/epochs=7506과 finite 학습 지표를 확인했으며 GPU5 약43GiB/여유54GiB, 기존 학습 PID 생존을 확인했다.

### koo_1021 Stage1BasePush 독립 실행 준비

- 사용자 확인에 따라 koo_1021/ec09d71의 Stage1BasePush를 `jointpush/ports/koo_1021`에 별도로 준비했다. 기존 Native Y2m과 KSH partial adapter 및 외부 원본은 유지한다. 업로드는 `jointpush/checkpoints/koo_1021/Stage1BasePush.pth`, 전용 prepare/train/test/VNC와 CPU contract 검사를 제공한다.
- 원본 학습 점유GPU 차단·2048환경·MPS 탐색을 유지하고 로컬 wrapper만 output을 runs/jointpush/koo_1021로 분리한다. CPU 원본 회귀11개·991개 source 항목/634D·1330D·949후보 bank 계약·셸 문법·diff 검사 통과, 누락 checkpoint의 GPU 실행 전 거부를 확인했다. checkpoint 업로드 전이며 strict restore/학습/GPU/viewer는 실행하지 않았다. 세부 검증 보고서는 runs/jointpush/koo_1021/preparation.json과 jointpush/KOO_1021.md를 참조한다.

### KSH Partial 초기 거리 원본 정렬

- 사용자 요청으로 전용 prepare의 spawn_distance를1~5m로 변경했다. 구역별 양손 point 접촉 보상과 goal2~6m는 유지한다. 설정 생성 후 값 확인. 실행 중6093 viewer는 변경 전 초기화 설정을 메모리에 유지하며 다음 실행부터 적용된다.

### Partial Joint Push 할당 관계 표시

- 사람별 색상으로 담당 분할 구역의 상단 테두리와 사람→구역 연결선을 표시한다. 실제 물리 box pose와 reset 시 확정된 구역 배정을 사용하며 관측·정책·보상은 바꾸지 않는다. GPU6/6093 표시용 viewer에서 추론·실제 구역 테두리/연결선 렌더링을 확인했다. 사용자 지시에 따라 기존6092 viewer와 전용VNC를 종료하고6093만 유지한다. Python 구문·diff 검사 통과, optimizer0.

### KSH Depth180 9000 체크포인트 Partial Joint Push viewer

- 지정 FiveTaskDepth18020kg_00009000 체크포인트를 별도 읽기 전용 KSH source adapter로 연결했다. 기존 Koo OPEN/PUSH 모델과 task/object/AMP 관측 규격이 달라 단순 경로 교체 대신 KSH 원래 표현과 normalizer를 유지한다. Stage1 tensor165개를 복사하고 새 coordination tensor104개는 identity 초기화한다.
- 기존 partial surround/CPA0.5 설정에서 Y만1.8m로 고정한다. 4인, coupled80%/NONE20%, 그룹 n에 따른 X=n×U(1,1.3), Z1.2~1.5m, aligned goal2~6m를 유지한다. 외부 source/checkpoint 수정은 없다. 이번 진입점은 test/eval 전용이며 학습 실행은 거부한다.
- GPU6에서2환경16step 추론 통과, finite/관측1272/8토큰/24actor/Y1.8/가중치 불변/optimizer0 확인. GPU6의1환경 viewer도 렌더링·추론·화면 캡처 확인. 첫 shim 없는 실행은 CUDA context 오류로 실패했으며 runtime shim 적용 후 해결했다. 기록: `runs/jointpush/ksh_partition_depth180_9000_20261022/`.

## 2026-10-21

### 원본 Koo 4인 OPEN/PUSH 혼합 attention PNG

- 사용자 요청으로 사람/문/상자/목표 각각4개인 native4 장면의16×16 actor attention 수집을 추가했다. 이전2인 PNG와 같은 OPEN/PUSH 혼합을 사용하며 mask나 CA4·추가학습은 없다. attention 수집 opt-in에서만4인 과제 샘플링·자기 문/상자/목표 그래프를 확장하고 원래2인 보상 함수를 두 agent쌍에 적용한다. 기존 PUSH-only ablation 기본값은 유지한다.
- CPU10개(혼합16조합 그래프·원래2인 보상 일치·attention/GTA·기존native4설정)·셸 문법·diff 검사 통과. GPU5 기존 MPS/seed123/4장면300step에서 step0/15/30/60/120/180/299 수집, 유효 장면17PNG 저장. raw shape[7,4,4,2,16,16], row합1/finite·수집기 없는 actor와 첫action차0·모델/normalizer 불변 확인. scene0/1/2는56/34/111step 첫낙상, scene3은300step까지 조기종료없음. 첫PNG 육안검사로16토큰·4층2head·과제표시 확인. `runs/jointpush/koo_native4_attention_20261021_gpu5/attention_png/`. 실행 종료·메모리반환·MPS 유지.

### 원본 Koo 2인 actor attention PNG 수집

- 사용자 요청으로 원본2명·문2·상자2/OPEN-PUSH 혼합 환경에서 actor4층×2head attention을 PNG로 저장하는 전용 진단을 추가했다. 실제 layer QKV 출력·GTA·관계bias로 softmax를 계산하며 토큰 순열을 canonical H0/H1/D0/D1/B0/B1/G0/G1 순으로 복원한다. task/물리배정 상태와 QK/bias/attention 원본도 별도 저장한다. 학습·mask·외부 원본 수정은 없다.
- CPU softmax/GTA2개 및 셸 문법 통과. GPU5 기존 MPS/seed123/4장면300step에서 step0/30/60/120/180/299 총24PNG 저장. shape[6,4,4,2,8,8], attention행합1·finite 검사, 수집기 없는 원래 actor와 첫 action차0, 정책/normalizer 불변을 확인했다. 첫 이미지 육안검사로 축·head/layer·토큰/과제 표기를 확인했다. 4장면 모두300step내 조기종료 없음. `runs/jointpush/koo_native2_attention_20261021_gpu5/attention_png/`. 실행 종료·메모리 반환·MPS 유지.

### 문 없는 Partition Push Stage1 zero-shot 박스 토큰 축소

- 사용자 요청을 4인 Partition Push로 해석해 전용 진단을 추가했다. 기존 surround_cpa05 환경에서 원래 Stage1 network/가중치/normalizer를 strict 복원하며 CA4·확장 head·전이학습은 사용하지 않는다. 물리 문은 없고 작업구역4토큰을 유지한 채 일반상자4→0개를 actor attention 모든 층의 QKV/GTA/bias에서 제외한다. 물리 박스·목표·critic은 유지한다. 문없는 Partiton 환경 특유의 크기·질량·그룹·랜덤시작/초기화 분포는 원래 Stage1과 다르다.
- CPU16개·셸 문법 통과. GPU5 기존 MPS/seed123/64장면300step 각조건 초기 물리 상태 해시 일치·정책/normalizer 불변. 총box토큰8/7/6/5/4 모두64/64장면낙상. 첫 종료 평균29.20/30.44/32.13/35.41/45.00step, 자기상자0.2m근접전낙상 포함장면58/55/56/52/49. 일반box정보를 줄여 낙상시점은 늦어졌지만 이 환경의zero-shot낙상은 해결되지 않았다. 시드반복이나 과제성공률 실험은 아니다. `runs/jointpush/partition_stage1_boxes_20261021_gpu5/summary.json`. 실행 종료·메모리 반환·MPS 유지 확인.

### 원래 2인 정책의 문 토큰 제거

- 사용자 질문에 따라 native4 확장 없이 원래 2명·문2·상자2 환경에서 문 토큰 제거를 비교하는 wrapper/launcher를 추가했다. 공통 source 설정 복원에 `--agents 2`를 지원하고 actor 모든 층의 Q/K/V·GTA·bias가8→6토큰으로 정렬되도록 일반화했다. 기존4인 기본값과 학습 가중치·외부 원본은 유지한다. 물리 문과 critic은 남는다.
- CPU12개·셸 문법·diff 검사 통과. GPU5 기존 MPS/seed123/64장면300step 원본0/64낙상, 문토큰제거0/64, 초기 물리 상태 해시 일치·정책/normalizer 불변. 첫 action차 RMS0.00003767/max0.001078. 낙상 안정성 검사이며 Push 성공률·학습속도·물리 문 제거 실험은 아니다. `runs/jointpush/native2_remove_doors_20261021_gpu5/comparison.json`. 실행 종료·메모리 반환·MPS 유지.

### Native4 모든 문 토큰 제거

- 사용자 요청으로 remove_doors 진단을 추가했다. 문4개를 actor 모든 층의 Q/K/V·GTA·bias에서 실제 제외해16→12토큰으로 attention한다. 정규화·원래 tokenizer·readout 슬롯·정책 가중치를 보존하며 물리 문과 critic은 유지한다. 이전 hide_doors(다른 장면 문 key만차단)와 구분한다.
- CPU5개 및 셸 문법 통과. GPU5 기존 MPS/seed123/64장면300step 원본50/64낙상 대비 문토큰제거54/64(84.375%), 자기 상자0.2m 근접 전 낙상 포함 장면14→17. 초기 상태 해시 일치·정책/normalizer 불변. 문토큰 전체 제거만으로 안정성이 개선되지는 않았다. `runs/jointpush/native4_remove_doors_20261021_gpu5/comparison.json`. 실행 종료·메모리 반환·MPS 유지.

### Native4 상자 개수 점진 차단

- 사용자 요청으로 다른 장면 상자2개 중 하나만 가리는 hide_box_first/hide_box_second를 추가했다. 물리 box rank로 선택하여 소유권·토큰 순열에 독립적이며 모든 query에서 정확히1개 key만 차단되는지 검사한다. 같은 장면 상자2개, 문·목표·사람은 유지한다.
- CPU mask4개·셸 문법 통과. GPU5 기존 MPS/seed123/64장면300step: 참조가능 상자4개 원본50/64낙상, 3개(첫 상자차단)46/64, 3개(둘째상자차단)48/64, 2개29/64. 자기 상자0.2m 근접 전 낙상 포함 장면은14/11/15/3. 4개/2개 조건은 이전 hide_boxes 비교에서 재사용했고 초기 물리 상태 및 checkpoint SHA 일치를 검사했다. 장면 수와 학습 시드 수는 구분하며, 물체 개수와 내용·거리 효과를 분리한 인과 결론은 아니다. `runs/jointpush/native4_box_count_20261020_gpu5/summary.json`. 실행 종료·기존 MPS 유지.

## 2026-10-20

### Native4 목표만 차단, 문·상자 유지

- 사용자 요청으로 다른 장면의 목표 key만 차단하는 hide_goals를 추가했다. 다른 종류 및 같은 장면 목표는 유지한다.
- CPU mask3개·셸 문법 통과. GPU5 기존 MPS/seed123/64장면300step 비교 완료: 원본50/64낙상, 목표만차단51/64(79.6875%). 자기 상자0.2m 근접 전 낙상 포함 장면14→12. 초기 상태 해시 일치·정책/normalizer 불변. 목표 단독 차단은 안정성을 회복하지 않았다. `runs/jointpush/native4_goals_only_20261020_gpu5/comparison.json`. 실행 종료·MPS 유지 확인.

### Native4 상자만 차단, 문·목표 유지

- 사용자 요청으로 다른 장면의 상자 key만 차단하는 hide_boxes를 추가했다. 문·목표·사람 및 같은 장면 상자를 모든 actor 층에서 유지한다. 기존 개입은 변경하지 않았다.
- CPU mask3개·셸 문법·diff 검사 통과. GPU5 기존 MPS/seed123/64장면300step 원본50/64낙상 대비 상자만차단29/64(45.3125%), 자기 상자0.2m 근접 전 낙상 포함 장면14→3. 초기 상태 해시 일치·정책/normalizer 불변. 이전 상자+목표14/64, 문+상자+목표1/64와 비교해 추가 상자 key 직접유입 차단만으로도 일부 개선되지만 전체 회복은 아니다. `runs/jointpush/native4_boxes_only_20261020_gpu5/comparison.json`.

### Native4 문·상자·목표 동시 차단

- 사용자 요청으로 상자·목표 차단에 문 차단을 더한 hide_objects_goals 조건을 추가했다. 다른 장면의 사람 및 같은 장면의 모든 토큰은 유지한다. 모든 actor 층에 적용하며 물리 환경·가중치·QKV 파라미터는 변경하지 않는다.
- CPU mask3개·셸 문법 통과. GPU5 기존 MPS/seed123/64장면300step 순차 비교 완료: 원본50/64낙상, 문+상자+목표차단1/64, 자기 상자0.2m 근접 전 낙상 포함 장면14→0. 초기 상태 해시 일치, 정책/normalizer 불변. 이전 상자+목표차단14/64에서 문까지 차단했을 때 추가 개선됐다. task 성공률이나 단일 attention 기전을 확정한 결과는 아니다. `runs/jointpush/native4_objects_goals_20261020_gpu5/comparison.json`. 실행 종료·GPU메모리 반환·MPS 유지 확인.

### Native4 추가 토큰 종류별 attention 차단

- 사용자 요청으로 다른 복제 장면의 사람/문/상자+목표 key를 각각 차단하는 hide_humans/hide_doors/hide_boxes_goals를 추가했다. 모든 actor 층에 적용하며 query와 같은 장면의 key 및 다른 종류의 key는 유지한다. 미차단 토큰을 통한 층간 간접 정보 전달은 남는다.
- CPU3개·셸 문법·diff 검사 통과. GPU5 기존 MPS에서 seed123/64장면300step으로 종류별 full/개입을 순차 실행했다. 6회 초기 상태 해시 일치 및 정책/normalizer 불변. 원본50/64낙상, 사람차단51/64, 문차단47/64, 상자+목표차단14/64. 자기 상자0.2m 근접 전 낙상 포함 장면은 원본14, 사람14, 문15, 상자+목표1. 상자+목표 정보의 직접 유입 차단이 가장 큰 안정성 개선을 보였으나 거리/관계bias/QKV 중 원인이나 단일 원인을 확정하지 않는다. `runs/jointpush/native4_types_20261020_gpu5/summary.json`. 실행 종료, 기존 학습/MPS는 건드리지 않았다.

### Native4 NONE bias 제거 비교

- 사용자 요청으로 모든 토큰/QKV/attention 연결은 유지하고 task edge로 연결되지 않은 off-diagonal NONE bias만 0으로 만드는 `none_zero` 진단을 추가했다. task/SELF bias와 critic은 보존한다. 기존 native4/pair 기본값은 유지한다.
- CPU mask·NONE/task/SELF 보존 테스트2개 및 셸 문법 통과. GPU5 기존 MPS에서 seed123/64장면300step full/none_zero 비교 완료, 초기 상태 해시 일치·정책/normalizer 불변. full50/64(78.125%), none_zero64/64(100%) 장면 낙상. 자기 상자 0.2m 근접 전 낙상을 포함한 장면14→33, 첫 action RMS차0.20406. NONE bias 제거로 안정성이 회복되지 않았으며 오히려 악화됐다. bias 전체의 원인 여부나 다른 기전을 배제하지 않는다. 결과: `runs/jointpush/native4_none_bias_20261020_gpu5/comparison.json`.

### Native4 Stage1 attention 차단 비교

- 사용자 요청으로 native4 물리 장면은 유지하면서 actor 모든 층에서 서로 다른 복제 장면 사이 attention을 차단하는 진단을 추가했다. 사람/목표는 배정 상자의 원래 2개 단위 장면, 문은 실제 door lane에 따라 그룹화한다. 토큰 순열에도 mask를 함께 적용하며 그룹당 8토큰을 검사한다. 기존 native4 기본 동작과 외부 원본은 변경하지 않았다.
- 같은 관측에서 full/pair action을 비교하고 RNG를 복원해 rollout 간 초기조건을 보존한다. 초기 상태 해시·정책/normalizer 불변·장면별 첫 종료를 기록한다. 상자 근접 지표는 자기 상자와 신체점 거리 0.2m 기준이며 실제 접촉 센서 판정이 아니다.
- CPU5개 통과. GPU5 기존 MPS에서 2장면4step 검사 후 seed123/64장면300step 비교 완료: full50/64장면 낙상, pair0/64, 초기 상태 해시 일치. 첫 action 차이 RMS0.11316/max1.13985. 추가 장면 정보 유입에 대한 민감성을 확인한 결과이며 거리/QKV/관계bias별 원인을 분리하거나 과제 성공률을 검증한 결과는 아니다. 산출물: `runs/jointpush/native4_attention_20261020_gpu5/`. 기존 학습/MPS는 종료하지 않았다.

### Partition Push 구역 독점 배정 수정

- 매 step 각 사람의 최근접 구역을 독립 선택하여 같은 구역을 중복 배정하던 오류를 수정했다. reset에서 같은 그룹 내 거리 합을 최소화하는 일대일 배정을 선택하고 에피소드 동안 고정한다. 관측 graph·손 목표·보상이 같은 배정을 사용하며 부분 reset은 다른 환경을 보존한다.
- 새 checkpoint에 배정 방식 버전을 저장하고 알 수 없는 방식은 거부한다. 기존 checkpoint는 새 배정 방식 적용을 로그에 알리며 로딩한다. 실행 중인 학습·뷰어는 종료하거나 재시작하지 않았다.
- CPU 테스트 **29개 통과**. 실제 GPU/Isaac Gym 2환경에서 기존 rear_lanes·랜덤 surround_independent, 5종 그룹 구성, 반복/부분 reset과 **64 physics step**을 검증했다. 구역 중복 없음·그룹 소유권·graph/손 목표 일치·step 중 배정 고정·finite 관측/보상을 확인했다. 로그: `runs/jointpush/partition_exclusive_regions_check_20261020/smoke.log`. 장기 정책 수렴 검증은 아니다.

### 로컬 반영 후 Stage1 경로 분리와 테스트 충돌 정리

- Partition Push 신규 학습의 없는 로컬 기본 경로와 모델 테스트의 옛 재현 경로를 공통 resolver로 통합했다. 명시 CLI/STAGE1_CHECKPOINT → 기기 전용 YAML → 로컬output/서버stage1_sources/옛재현 후보를 순서대로 사용한다. 잘못된 명시 경로·모호한 최신 run은 실패한다. 기존 Stage2 resume/test 복원은 유지한다.
- Git 제외 `paths.local.yaml`과 예시를 추가했다. 기본 프로필은 변환된.npy를 사용하므로 raw AMP 강제 테스트를 현재 기본 manifest 검증으로 수정했다. raw 모듈/원본 motion과 사용자 삭제 내용은 복구하거나 삭제하지 않았다.
- CPU29개 통과(실제 Stage1 로딩·action 보존·CA gradient 포함). 실제 checkpoint CLI가 서버stage1_sources29000을 찾는 것 확인, 셸 문법·diff 검사 통과. GPU 실행이나 기존 작업 종료는 하지 않았다.

### 기존 뒷면 정렬 3,000iter 학습 재개

- 사용자 승인으로 랜덤 출발 학습만 SIGTERM 종료했다. 해당 뷰어는 종료 직전 확인에서 이미 종료된 상태였다. 기존 랜덤 출발 결과·로그는 보존했다.
- `partition_push_train/PartitionPushCa4_19-16-41-15/nn/PartitionPushCa4_00003000.pth`에서 GPU0/1024환경으로 resume했다. checkpoint epoch3000/frame393216000/optimizer를 확인했고, 생성 env가 저장 env와 정확히 동일하며 rear_lanes/모션 manifest SHA가 일치함을 검사했다.
- tmux `partition_rear_resume_20261020`, 출력 `runs/jointpush/partition_push_rear_resume_20261020/`. launch.sh/launch.json에 재개 명령과 원본을 기록했다. 실제 epoch3001~3003 업데이트, finite 보상, 약 2.1만 frames/s 진행을 확인했다. 다음 정기 저장은 3500iter다.

## 2026-10-19

### Partition Push 신규 학습의 360° 독립 접근 시작

- 사용자 요청으로 신규 기본 spawn을 `surround_independent`로 변경했다. 각 사람은 배정 상자의 XY 외접원 밖 1~2m에서 독립 360° 위치·독립 360° heading으로 시작하고 root 간 최소 1.2m를 만족하도록 재추첨한다. 그룹·4명 인원·목표·contact point·보상·기존 CPA 회피 항은 유지한다.
- `spawn_mode`가 없는 기존 checkpoint는 `rear_lanes`를 기본으로 유지하여 재개/평가의 초기화 분포를 보존한다. 현재 학습·뷰어는 종료하거나 재시작하지 않았다. 새 YAML은 신규 학습에 적용되며 기존 resume는 저장된 spawn 설정을 복원한다.
- CPU 20개 통과: 기존/신규 5종 그룹 실제 reset·부분 reset·할당/8토큰 보존, 1만 환경 360° 위치·heading 독립성·상자 clearance·사람 최소 간격 검사. GPU0/2환경 64 physics step 및 반복/부분 reset, finite obs/reward, 모든 그룹 구성 검증 rc=0. 별도 진단 CLI의 필수 인자 누락 로그를 보존 후 보완해 통과했다. 로그: `runs/jointpush/partition_surround_check_20261019/smoke_final.log`.

### Isaac Gym 네이티브 viewer 복원과 Vulkan 드라이버 선택 수정

- 사용자 요청대로 센서/Tk 우회 창과 관련 분기·모듈을 제거하고 `partition_push_view.sh`를 기본 Isaac Gym viewer로 되돌렸다.
- 동일한 단일 상자 최소 예제의 native framebuffer는 Vulkan ICD 자동 탐색에서 전부 0, NVIDIA ICD 지정 시 정상 RGB/불투명 alpha였다. 카메라 조정·visual mesh 전환·OpenGL dispatcher 변경으로는 해결되지 않았다.
- view wrapper에서 기존 Vulkan 드라이버 지정이 없고 NVIDIA ICD manifest가 존재하면 `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`을 적용한다. 전역 환경·설치 파일·학습 프로세스는 변경하지 않는다.
- 실제 1000iter Partition Push checkpoint/1환경/30step 네이티브 평가 rc=0. 센서 생성이나 별도 창 없이 기본 1600×900 viewer에서 사람·상자·바닥이 정상 출력되는 원본 프레임을 확인했다: `runs/jointpush/partition_view_check_20261019/native_partition_nvidia_icd.png`. CPU 회귀 14개·셸 구문·diff 검사 통과.

### 로컬 검은 네이티브 창을 센서 영상 viewer로 대체

- 실행 중인 Isaac Gym 창의 X11 framebuffer가 검은색임을 직접 확인했다. 같은 simulation의 camera sensor에서는 사람·상자·바닥이 정상 렌더링되어, 이전 카메라 구도 변경만으로는 해결되지 않았음을 확인했다.
- 로컬 `partition_push_view.sh`의 기본 backend를 sensor로 변경했다. Gym 센서 RGBA 영상을 1280×720 Tk 창에 실시간 표시하며 왼쪽 드래그 회전·휠 확대·Esc/닫기를 제공한다. 네이티브 graphics context는 유지하고 해당 PID의 검은 창만 최소화한다. 다른 실행의 창은 건드리지 않는다. native backend를 명시적으로 선택할 수도 있다.
- 네이티브 viewer 해제 후 센서 렌더링은 rc=139로 실패하여 해제 방식을 제거했다. 실패 로그는 보존했다. 수정 후 실제 view 스크립트/1000iter checkpoint/1환경에서 사람 4명과 상자가 표시되는 Tk 창을 직접 캡처했다: `runs/jointpush/partition_view_check_20261019/sensor_window.png`. 1환경/240step 실행 rc=0, CPU 회귀 14개 및 창 최소화 대상 PID 검사 통과. 학습은 유지했다.

### Partition Push 뷰어 카메라 구도 보완 (빈 화면 해결 아님)

- 렌더링은 기존 물리 스텝에서 수행됨을 확인했다. Partition Push는 기존 근접 agent 카메라를 상속하고 있었으며 전용 장면 카메라를 추가했다.
- Partition Push reset마다 첫 환경의 사람·배정 상자·목표 범위로 overview 카메라를 배치한다. 격리 bank actor와 먼 일반 상자는 범위에서 제외하며 기본 추적을 해제했다. 새 episode에서 그룹 구도가 바뀌어도 화면 안에 배치한다.
- CPU 회귀 14개와 공동/독립 그룹 카메라 배치·parked actor 제외 검사 통과. GPU0/1000iter checkpoint/1환경/60step/2회 GUI 평가 rc=0 및 렌더 호출 확인. 후속 데스크톱 캡처는 Wayland/XCB 제약으로 실패하여 실제 화면 이미지는 검증하지 못했다. 기존 학습은 유지했다. 검증 로그: `runs/jointpush/partition_view_check_20261019/`.

### Partition Push 로컬 viewer 준비

- `scripts/partition_push_view.sh`를 추가했다. checkpoint 경로를 받아 기존 test 복원 경로에 연결하며 HEADLESS=0/기본 1환경/기본 100만 회 반복으로 실행한다. `NUM_ENVS`, `REPEATS`, `TOKENHSI_GPU`를 지정할 수 있고 출력은 `runs/jointpush/partition_push_view`로 분리한다. 상대 checkpoint 경로도 실행 전 절대경로로 해석한다.
- 셸 구문·인자 누락 시 사용법 출력·diff 검사를 확인했다. GUI는 실행하지 않았다.

### 로컬 Push180 모션 정리와 RTX 5090 실행 검증

- 받은 `output/dataset_bones_push180`을 `motiondata/dataset_bones_push180`으로 이동하고 57개 파일의 SHA-256 일치를 확인했다. 실행 포트에 상대 심링크를 연결하고 loco는 기존 로컬 데이터로 연결했다. raw NPZ와 Stage1 checkpoint는 output에 보관한다.
- 사용자 요청에 따라 변환된 Push180 19개만 사용한다. 기존 AMP 목록의 별도 dataset_bones_push 2개는 제외했으므로 이전 21개 목록과 분포가 다르다. 데이터 본체는 Git에서 제외하고 위치·사용 범위를 문서화했다.
- Partition Push의 기본 Stage1을 받은 로컬 29000 파일로 변경했다. conda hook의 nounset 오류, conda compiler의 crypt.h 빌드 실패를 수정하고 RTX 5090용 legacy TorchScript fusion 비활성화 및 EPISODE_LENGTH를 연결했다. 첫 실패 로그는 보존했다.
- CPU 테스트 14개, manifest 전체 경로·NPY 로드·rotation/root translation 유한값, 셸 문법·diff 검사 통과. RTX 5090/GPU0/1024환경/MAX_ITERATIONS=2 전이·학습·저장 rc=0, 저장 모델 16환경/30step/2회 복원 평가 rc=0. 단기 실행 검증이며 성공 성능 검증은 아니다. 산출물: `runs/jointpush/partition_local_check_20261019/`. 장기 학습은 시작하지 않았다.

### 직선 Push 목표 거리 2~6m

- 사용자 요청으로 `partitionPush.goal_distance`를2.2~5m에서2~6m 균등 샘플링으로 변경하고 설정 문서를 갱신했다. YAML 값과 diff를 확인했으며 학습·GPU 실행은 하지 않았다.

### 가변 Push 그룹과1인 기준 랜덤 일반 상자

- 사용자 확인에 따라사람4명고정·순차coupled80%/NONE20% 그룹샘플링을연결했다. 각그룹n명에X=n×U(1,1.3)m, 질량=n×U(20,30)kg를적용하고X를n등분한다. Y=U(1.8,2.2)m/Z=U(1.2,1.5)m는공통이다. 남는일반4개도1인기준크기/질량에서독립추첨한다.
- 과제구역합계4+일반4로8토큰을유지하고, 물리상자는그룹수+4개다. 형상/질량setter변경없이reset에서그룹을바꾸도록24actor후보bank를사용한다. canonicalgoal로동일그룹coupled를식별하고다른그룹의접촉이자기보상gate에들어가지않도록유지한다.
- 프로필version2·저장설정복원·1인기준추론ablation및[문서](markdowns/PARTITION_PUSH.md)를갱신했다. 이전4인고정version1은직접resume하지않는다.
- CPU19개통과: 그룹5종실제reset메서드/부분reset/8토큰·일반4개보존/인원비례크기·질량/샘플링통계/CA4그룹구분과기존shape회귀. GPU·학습·시뮬레이션은실행하지않았다.

### 공유 상자 4구역·일반 상자 대체 Push 로컬 프로필

- 사용자 지정 Stage1 contact-at warmstart를 전이하는4-agent `partition_push_ca4`를 추가했다. 초기 일반 상자8개 중4개를 격리하고, 큰 공유 물체1개의 X4분할 토큰4개+일반4개로8토큰을 유지한다. 잘못 해석한 전 상자4분할/32토큰 방식은 적용하지 않는다.
- 공유 크기X4~5.2/Y1.8~2.2/Z1.2~1.5m·총80~120kg, 손 높이Z×0.5 point, 최근접 구역·같은 물리 상자 coupled, aligned goal, loco-only, raw Push+loco AMP. 기존 Push CA4와 entity RMS를 재사용하며 기존 실험은 유지한다.
- 로컬 train/test/VNC-desktop 진입점·source 디렉터리 해석·저장 설정 복원·추론 크기/총질량 ablation과 [실행 문서](markdowns/PARTITION_PUSH.md)를 추가했다. 외부 `/home/hwanhee/ksh`, `/home/hwanhee/koo_cvpr`는 읽기만 했다.
- CPU10개 테스트 및 Python10파일/셸6파일 구문 검사 통과. 부분reset 검사에서 발견한 parked actor 좌표 dtype 오류를 수정 후 재통과했다. source29000의 network 설정 일치를 CPU로 확인했다. GPU·학습·시뮬레이션·실제 전이 forward는 사용자 요청에 따라 실행하지 않았다.

### 각도별 면·손 목표점 고정 및 높은 박스 프로필

- 사용자 요청으로 `assigned_faces` Contact를 분리했다. reset 시 가장 가까운 box 수평축 대비 목표각도가25°이하면 한 면 좌우, 초과면 인접 두 면 중심에 배치한다. 초기 총 접근거리가 짧은 agent 배정을 선택하고, 각 오른손·왼손 target(간격0.3m/바닥기준0.8m)과 면 바깥 접근점(0.8m)을 box 로컬좌표에 고정한다. box pose를 따라가며 면·역할은 reset 전 바꾸지 않는다. 부분reset은 해당 env 버퍼만 갱신한다.
- Contact State/Success는 배정된 두 손 중 나쁜 쪽의3D오차, Progress는 자기 접근점까지 XY거리로 계산한다. AT 거리식·가중치는 유지하고 새 Contact gate에 연결했다. 실제 손/접근점과 한면·두면 모드를 extras로 노출했다. 손바닥/힘 방향 검증은 추가하지 않았다.
- 별도 `joint_push_assigned_faces_ca4` config/train/test/view에 높이1.6~2.6m(step0.2) 적용. 가로2~3m/깊이1~2.6m/질량40~60kg 유지,1620개격자. 이전 실험 설정·checkpoint는 보존하며 이전 프로필 resume/test를 거부한다.
- CPU 회귀39개 및 셸 문법 통과. GPU6/2048환경/MAX_ITERATIONS=2 학습·저장 rc=0, 저장 checkpoint의16환경/30step/2회 복원·reset 평가 rc=0. 검증 산출물 `runs/jointpush/assigned_faces_check_20261019/`. 장기 본학습은 시작하지 않았다.

## 2026-10-17

### 비정렬 자유 표면 Contact 본학습 시작

- 사용자 요청으로 기존 28000 랜덤 박스 viewer만 종료하고 GPU6/2048환경/2명/seed6397로 전용 본학습을 시작했다. 기존 Stage1 원본29000에서 새 Stage2 초기화이며 aligned Stage2 resume가 아니다. tmux `jointpush_directional_gpu6`, 산출물 `runs/jointpush/directional_surface_training/`. MPS와 별도 Stage1 학습은 유지했다.

### 비정렬 목표·자유 표면 Contact 프로필

- 기존 직선/대칭 프로필을 보존하고 `directional_surface` Contact를 추가했다. 각 손의 실제 유한 옆면 거리·높이·목표 방향에 유효한 안쪽 법선·agent 손중점 XY 간격을 평가한다. 손 중점만 맞추는 우회와 동일 면 배정·중심 대칭 조건을 제거했다. Direction은 내적/0.7을 포화시켜 대각선의 인접 두 면도 Success를 달성할 수 있다.
- Contact State/Progress/Success 가중치0.2 유지, Progress는 회전된 실제 XY 외곽까지 거리다. AT 거리식·가중치를 유지하고 새 min-φ soft gate와 모두 Contact Success인 최종 gate로 연결했다. 상자와 목표 방향을 독립 360° 샘플링하는 랜덤 크기·질량 전용 config/train/test/view를 분리했다. 기존 aligned checkpoint의 실험 간 resume/test는 거부한다.
- CPU 회귀23개 통과(import 경로 수정 후), 셸 문법 통과. GPU6/2048환경/MAX_ITERATIONS=2의 초기화·업데이트·checkpoint 저장 rc=0, 저장 checkpoint의 2환경/30step 복원·평가 rc=0. 검증 산출물 `runs/jointpush/directional_surface_check_20261017/`. 장기 학습과 기존 viewer는 변경하지 않았다.

### 2인 랜덤 박스 viewer 학습 격자 선택

- test 전용 `JOINT_PUSH_VIEW_GRID_INDEX`로 checkpoint 학습 격자의 특정 크기·질량을 선택한다. 범위 검사 및 학습 모드 거부를 적용했다. 기본 랜덤화는 유지한다. 사용자 요청으로 GPU6 viewer를 최대 격자(3×2.6×1.5m/60kg)로 재실행했다. Python 문법 검사 및 실제 GPU6 viewer 선택 로그·6086 포트 확인 통과.

## 2026-10-16

### 4명 viewer 박스 깊이 1.5m

- 사용자 요청으로 전용 test/view 박스 깊이를3m에서1.5m로 변경했다. 가로3m/높이1.5m/질량100kg 유지. 현재 설정 문서 갱신 및 셸 문법·diff 검사 통과. viewer 재실행 시 적용된다.

### 4명 viewer 박스 가로 3m

- 사용자 요청으로 전용 test/view 박스를 4×3×1.5m에서 3×3×1.5m로 변경했다. 질량100kg 유지. 현재 설정 문서 갱신 및 셸 문법·diff 검사 통과. 실행 중 viewer는 재시작 시 적용된다.

### 4명·대형100kg 박스 추론

- 사용자 요청으로 랜덤 모델의 공유 tokenizer/Transformer를 4명 추론에 확장했다. 팀 그래프를 인원 기반 payload/goal 인덱스로 구성하고 8개edge를 사용한다. 대형payload override는 명시적 test preview에서만 허용하며 원래 checkpoint 계약을 먼저 검증한다. 전용 four_agent_100kg test/view는 4×3×1.5m/100kg, GPU6 기본이다. 학습 분포와 실행 중 학습은 변경하지 않는다.
- CPU 회귀5개·셸문법·diff 검사 통과. GPU6/10000 checkpoint/2환경/4명/30step rc=0. 실제100kg,bbox4×3×1.5m,관성오차<0.00001,부분reset보존,8edge 확인. 첫 그래프 인원제한 실패로그 보존 후 수정·재검증했다. 성공 성능 검증은 아니다. runs/jointpush/four_agent_100kg_check/.

### 랜덤 크기·질량 체크포인트 3명 추론

- 공유 tokenizer/Transformer 가중치를 유지하고 test 전용 3명 경로를 확장했다. 랜덤 shape/mass 계약은 유지하며 기존 고정40→60kg preview는 보존한다. surround spawn을 N명 쌍별 거리 검사로 확장하고 2명 기본값을 유지한다. 전용 test/view wrapper와 실행 문서를 추가했다.
- CPU shape/spawn 회귀 3개 및 셸 문법 검사 통과. GPU6에서 10000 checkpoint/2환경/3명/30step 추론 rc=0,6개 관계edge와 checkpoint 복원 확인. 학습은 변경하지 않았다. 성공 성능 검증은 아니다.

### 랜덤 크기·질량 viewer GPU 지정

- 사용자 요청으로 랜덤 모델 전용 test/VNC의 GPU0 고정을 제거하고 GPU6 기본값과 TOKENHSI_GPU override를 적용했다. 기존 checkpoint 분포 복원과 실제 형상·질량 랜덤 샘플링을 유지한다.
- 셸 문법 검사 통과. GPU viewer는 실행하지 않았다.

## 2026-10-15

### 랜덤 형상·질량 학습 GPU6 재개

- 사용자 요청으로 고정 크기 JointPush를 종료하고, 기존 랜덤 형상·질량 2500 checkpoint에서 2048환경/seed6397로 GPU6 MPS 재개. train GPU는 TOKENHSI_GPU로 지정 가능하게 변경했다.
- 산출물: runs/jointpush/random_shape_resume_gpu6_20261015_223106/. 체크포인트 복원 후 6회 이상 학습 업데이트 진행과 finite fps(약2만 frames/s)를 확인했다.

### 원본 Push Stage1 설정 설명 추가

- JOINT_PUSH_SETTINGS.md15절에source29000checkpoint저장설정(seed42/1.3m20kg/OPEN-PUSH혼합/독립물체/목표2m)과RSI/warmstart/embedding/조건부AMP/보상식/Stage2차이추가. 원본contact_at는각손뒷면XY거리/force조건을쓰며현재JointPush중점대칭/높이보상과구분한다.
- 외부원본은읽기만하고실제전이checkpoint메타데이터와대조. 결과수치제외/문서링크/diff검사완료. 학습변경없음.

### GPU0 독립 랜덤 형상·질량 공동 Push

- 새RandomShape task/profile/전용train/test/view추가. X2–3(step.2)/Y1–2.6(step.2)/Z1–1.5(step.25)m,질량40–60(step5)kg,810조합을env생성시독립배정하고reset간유지한다. 사용자추가지시로GPU1에서GPU0으로전환. 기존GPU6/MPS유지.
- 실제asset밀도/관성·관측bbox·reset높이·고정면·손간격·접근보상·360°spawn반경을실제크기에연결했다. 이전고정크기/3agent/mass진단경로는보존.랜덤형상은단일mass/3agent진단과혼용거부.
- CPU13개통과,GPU1 64환경120step후자연종료.GPU0 2048환경2iter학습/저장/16환경60step복원rc=0.740조합배정/실제질량·관성오차<.001/bbox·부분reset일치확인. 첫2048검증의audit초기화순서는수정하고실패로그보존.
- GPU0본학습tmux jointpush_randomshape_gpu0에서새Stage2로시작.실행법/범위는markdowns/JOINT_PUSH_RANDOM_SHAPE_SETTINGS.md. 산출물runs/jointpush/random_shape_mass_softgate_training/.

### 결과를 제외한 Joint Push 상세 설정 문서

- markdowns/JOINT_PUSH_SETTINGS.md 추가. 저장된relation_config와실행코드기준으로60kg물리·마찰·밀도·360°초기화·RSI·관측·그래프·Stage1전이/동결·CA4·softgate보상식·AMP·PPO·종료/저장·실행법을정리. 평가성과수치는제외.
- 저장설정/seed/motion수·문서링크·결과수치미포함/diff검사완료. 문서만변경하고새실행/학습변경없음.

### 질량 진단 적용 시점 수정 및60kg/40kg대응평가

- 실행중rigid-body setter진단에서root/reset이상을발견해이전readback만의검증결과를대체했다. 실제질량을GPU prepare_sim이전asset밀도로적용하고관성을자동계산,미사용box3은기존60kg보존. strictcheckpoint계약검사는유지한다. 기존mass40_view명령에자동적용.
- 동일12000iter정책/GPU6/3seed×64=192episode/600step,초기human/box/goal SHA정확일치.공식metrics대조:60kg ever0/192/final0/192,40kg ever95/192(49.48%)/final46/192(23.96%).평균잔여거리2.583/0.645m,순변위1.076/3.086m. 결과 runs/jointpush/mass_comparison_20261015_final/summary.json 및 RESULTS.md.
- 초기진단기록무효/대체사유보존. 모든최종trial에서asset단계질량/초기상태대응/공식metrics와수동집계일치검증. 학습/MPS유지. 정지마찰단독원인을확정하는실험은아니다.

## 2026-10-14

### 평가에서만40kg질량진단

- strictcheckpoint복원후player의명시적옵션으로공유box2질량40kg/관성비례조정. 학습/모델/원래저장설정/마찰/보상/관측유지. 전용softgate_mass40_view및실측report추가.
- 4500모델GPU6 4환경60step진단rc=0/실측40kg확인. runs/jointpush/mass40_check_20261014/.

### 9000iter 2명 정책의 3명·60kg 추론 뷰어

- 사용자 요청으로 기존40kg/2명9000iter정책을 그대로 로드하는 test전용3명 preview 추가. 상자2.6m cube 유지, 실제질량59.999996kg·3명·6개공유PUSH/AT edge assert확인. 시작좌/중/우,공유상자/목표,가변인원 attention/RMS사용. 학습모드거부,원본모델·실행학습보존.
- OPEN/PUSH runtime/물리tensor offset/CPA를인원수기반으로확장하고기존2명경로보존. preview는원본40kg저장계약에서질량만60으로전환함을검증한다. 전용test/view명령과문서추가;3명학습프로필없음.
- CPU보상·표시회귀10개와3명순열/graph2개통과. 3명30step headless rc=0,기존2명30step GPU6회귀rc=0. 최종GPU6 VNC6090실행및실제화면확인. 최초GPU1검증/뷰어는사용자GPU6전용지시후종료했으며전용명령도GPU6고정. 산출물 runs/jointpush/three_agent_60kg_preview/ 및 three_agent_60kg_gpu6_view.log. 성능성공률검증은아님.

### 숨김 mesh fallback 및 기본 카메라 초기화 보완

- 앞선실행완료검증만으로시각결과를충분히확인하지못한점보완. 미사용asset에1e-6mvisual을명시해collisionvisualfallback을피하고기본_init_camera재배치도차단.
- GPU6 9000모델1환경2에피소드×30step VNCrc=0,실제screen.png에서회색배경물체없음/옆시점확인. reset전후camera위치·회전정확일치assert통과. runs/jointpush/clean_scene_visual_check_20261014/.

### 옆쪽 촬영 카메라와 에피소드 reset 위치 보존

- 초기camera를미는방향옆쪽/높이3m로배치하고한번만호출한다. 이후reset에서카메라/focus를덮어쓰지않는다. 사용자자유카메라조정유지.
- 문법/diff검사와GPU6 9000iter모델1환경2에피소드×20step VNCrc=0. reset2회·카메라배치1회로그확인. runs/jointpush/side_camera_check_20261014/.

### 촬영용 미사용 배경 물체 visual 제거

- viewer에서door0/1및box3을visual없는임시URDF로생성. 공유box2는유지. door원본visual만제거하고collision/inertial/joint유지,미사용cube는같은크기/질량/inertia/collision로생성. 원본asset/관측슬롯/headless학습변경없음.
- 문법/diff검사 및 GPU6 9000iter모델1환경30step VNC검증rc=0,doorbody/dof/index계약확인. runs/jointpush/clean_scene_check_20261014/.

### 공동 Push 뒤쪽 자유 카메라

- reset마다 초기box미는방향 기준 두agent뒤쪽에서agent/상자/목표를바라보도록배치. 이후cameraFollow해제해focus없는고정시점유지. 학습환경/보상변경없음.
- 문법/diff검사 및 GPU6 9000iter모델1환경30step 실제VNC검증rc=0. runs/jointpush/rear_camera_check_20261014/.

### 노란 목표 영역 overlay 제거

- 사용자 요청으로 목표 footprint/십자선만 제거하고 방향화살표는 유지. 학습/물리/보상 변경 없음. 기존표시테스트를화살표만검증하도록갱신.

### 기존60kg 종료 후 softgate 새 학습 실행

- 사용자요청으로기존rear-only60kg학습PID4045659만종료. tmux jointpush_softgate/GPU6/2048환경에서Stage1초기값전이로새Stage2시작. 다른학습/MPS유지.
- 저장설정60kg·surround_independent·goal_distance[2.2,5.0]·soft_min확인,TensorBoard iter2 기록확인. run: runs/jointpush/aligned_surround60_softgate_training/JointPushAlignedSurround60SoftGateCa4_14-11-37-12/.

### AT state/progress soft-min 접촉 gate 비교

- softgate별도프로필에ATdense보상=min(두사람phi_push)×반대편조건 적용. 기존PUSH와strictresult/done유지,실제이동량보상추가없음. hard기본계약보존하고새보상metadata로혼용방지. 같은surround60/2.2–5m조건과비교하도록문서화.
- CPU11개통과(약한접촉/연속성/엄격성공/같은쪽차단), GPU6 16환경120step Base검사rc=0(60kg·360°spawn·고정면·부분reset·finite). runs/jointpush/softgate_check_20261014/. train/test/view제공,현재학습종료/새본학습실행없음.

### 다음360°시작실험 목표거리 확대

- surround60프로필의goal_distance를[2.2,5.0]m으로변경. reset별균등샘플링과초기정렬은유지. 기존실행학습/과거checkpoint설정은변경하지않았다. 설정값확인및diff검사완료.

### 다음 실험 곡선 제외 및 랜덤 미는 거리 확인

- 사용자 결정으로 곡선혼합설계를60kg직선/360°agent시작으로갱신. 목표거리는이미환경별reset마다2.2–2.8m균등샘플링하므로기능변경없이적용범위를확인하고문서화했다. 기존학습변경없음.

### 앞/옆/뒤 모두 허용하는360°agent시작

- 별도surround60프로필에 방위각360°·heading360°독립샘플링 추가. box XY외접원+1–2m반경 및 root최소간격0.8m로 실제상자/사람겹침을피한다. 미는면reset_fixed/60kg/초기box목표정렬은유지. 기존rear-only학습은변경하지않았다.
- 전용train/test/view와동일프로필resume검사,다음curve설계360°시작갱신. CPU10개 및 GPU6 16환경120step Base검증rc=0. 실제spawn위치/heading·고정면·부분reset·60kg확인. runs/jointpush/surround60_check_20261014/. 새본학습실행없음.

### 다음 직선·곡선 혼합 Push 실험 설계

- 60kg/동일agent랜덤시작에 직선50%·좌/우±10–30°각25% 혼합 제안. 초기 면 선택도 목표방향에 의존하지 않고 local−Y로 고정한다. 상자 회전 시 같은 물리면을 따라가며 world 위치 고정으로 오인하지 않는다.
- Stage1전이에서 새 Stage2 학습, 모델/보상 유지, 조건별 도달·접촉·실제이동/yaw 측정 계획과 검증 요건 기록. markdowns/joint_push_curve_plan.md. 설계만 요청되어 학습실행/실행용config생성/현재학습변경/큐편집 없음. 문서diff검사 완료.

### 공동 Push 초기 방향/목표 뷰어 표시

- 공유 목표 marker가 지름 약6cm이며 두 marker가 겹쳐 잘 안 보이는 문제를 목표 footprint와 십자 표시로 보완. reset 초기 방향 회색 화살표와 현재 방향 청록/분홍 화살표를 분리했다. 방향 랜덤화 자체는 이미 적용되어 있었으며 실제 reset 두 방향이 다름을 확인했다.
- 랜덤 시작 전용 test/view에60kg/rear_independent 모델 검증 추가. CPU2개 및 실제6500iter VNC4환경60step rc=0, runtime 초기방향/부분reset/고정면/60kg 확인. runs/jointpush/view_fix_check_20261014/. 기존 학습은 변경·종료하지 않았다.

## 2026-10-13

### 60kg 랜덤 시작 Stage2 신규 학습 실행

- 사용자 요청으로 tmux jointpush60/GPU6/2048환경에서 신규 Stage2 실행. 기존 Stage2 checkpoint·optimizer·counter 없이 Stage1 29000 초기값과 fresh CA4로 시작한다. 기존40kg학습은 유지한다.
- 초기화 보고서 및 TensorBoard iter6 기록 확인. run: runs/jointpush/aligned_random_spawn60_training/JointPushAlignedRandomSpawn60Ca4_13-16-53-40/. 로컬 Stage1 복사본 경로가 삭제되어 원본을 읽기 전용으로 runs/jointpush/stage1_sources에 복사하고 신규 wrapper 기본 경로를 갱신했다.

### 60kg 상자와 독립 랜덤 agent 시작 프로필

- 별도 aligned_random_spawn60 설정 추가. 상자/목표 초기 정렬 및 reset_fixed 유지, 두 agent의 뒤쪽 거리1–2m·lateral±1.8m·heading360도를 독립 샘플링한다. 최소 root 간격0.8m 후보 재샘플링. 기존40kg학습은 유지하고 새 본학습은 자동 실행하지 않았다.
- 전용 train/test/view 및 동일60kg프로필만 resume 허용. CPU11개 통과, GPU6 16환경120step Base 검증 rc=0. 실측60kg·실제 독립 위치/heading 일치·고정면 목표 변경 불변·부분reset 보존 확인. 결과 runs/jointpush/random_spawn60_check_20261013/.

### 공동 Push 에피소드 접촉면 고정

- 목표 방향을 매 프레임 따라가며45도 경계에서 면이 바뀌던 새 Stage2 설정을 reset_fixed로 변경했다. reset에서 저장한 상자 로컬 법선 하나를 두 사람이 공유한다. 상자 회전/이동을 따르고 부분 reset은 해당 환경만 갱신한다. 이전 checkpoint는 저장된 goal_dynamic 계약을 보존한다.
- CPU12개 통과: 목표 반전/각도 경계/상자 회전 및 두 사람 접촉 보상 확인. GPU6 4환경120step Base 정책 검사 rc=0,40kg 실측 및 목표 반전 시 고정 면 불변/부분reset 보존 확인. 산출물 runs/jointpush/fixed_face_check_20261013/. 새 학습은 실행하지 않았다.

### 정렬 Push 상자 질량 40kg으로 변경

- 사용자 요청으로 새 loco/koo 정렬 프로필의 mass를 모두40kg으로 변경했다. 크기2.6m cube 및 정렬/RSI/보상은 유지한다. 실행 중인 프로세스는 변경하지 않았다.
- 두 YAML 질량40 및 goal_mode box_forward 확인, diff 공백 검사 통과. 이전60kg 시뮬레이션 검증은 당시 설정의 기록으로 보존한다.

### 정면 정렬 60kg 학습 프로필

- 초기 목표를 box local +Y에 정렬하는 loco/koo RSI 프로필과 전용 train/test/view 진입점 추가. yaw 랜덤과 기존 보상·AMP·모델 유지. 기존 goal360 checkpoint 혼용 거부. 기존 본학습40kg 및 MPS는 변경하지 않았다.
- CPU11개 통과, GPU6 4환경120step 환경 검증 rc=0. 목표/면 cosine1.0 및 실제 질량59.999996kg 확인. 검증은 Base 정책이며 성능을 입증하지 않는다. 실행법은 config.md 상단.

### 로컬 VNC 포트 선택 보완

- X display가비어있어도해당VNC포트를다른서비스가사용하면후보다음display를선택하도록수정했다. 다른서비스는종료하지않는다.


### koo screened Push RSI를 공동 환경에 연결

- 검증에서 MAX_ITERATIONS=0이 상한 변경을 하지 않는 기존 CLI 동작을 발견했다. 새 RSI train wrapper는0을거부하도록수정했다. 검증의실제반복은TensorBoard를따르며1epoch종료로주장하지않는다. 검증산출물은삭제하지않는다.

- 원본warmstart RSI와일치하는0.5interaction/0.7late/0.3late-tail/0.06mstand-off/Start 설정을 별도환경에추가했다. 두사람의scene/frame/box_progress를공유하고최종목표는고정한다. koo shoulder/양손정렬을재사용하고실제공유상자면의좌우에배치한다. AMP혼합및기존loco학습은유지한다.
- sourcebank1048Push후보·PD12·원본physics/reward/motionhash 계약일치 확인. 큰상자의전체후보새PDscreen은아니다. 실제reset에서6cm정렬/finite를검증한다.
- 9000iter 모델의40kg RSI-only평가16환경600step rc=0,목표0/16·scene종료6.25%·최종개인PUSH edge각18.75%. 60kg2048환경신규학습reset에서1040interaction/738late요청·6cm정렬확인. 로그 runs/jointpush/koo_rsi_check_20261013 및 koo_rsi_train_check_20261013.
- 전용train/test/viewwrapper와RSI-only진단계약추가. 기존학습·MPS·원본koo/ksh저장소는중지/수정하지않았다. 실행법은markdowns/config.md상단.

## 2026-10-12

### 60kg 곡선 Push 비교 및 중단된40kg 학습 재개

- local+Y 정면에서±20–30° 목표를 생성하고 같은local-Y면을 유지하는60kg 무학습 비교를 추가했다. 가상box는뒷면·바닥에붙은1.3m cube. agent별spawn을자기virtualcube의lane과일치시키고실제reset감사를추가했다.
- CPU14개 통과. GPU6 16환경600step rc=0, 실제60kg, lane오차2.4e-7m, 목표성공0/16·scene종료43.75%, trace평균goal거리2.325→1.548m(30→570step). 결과 runs/jointpush/virtual_curve_aligned_check_20261012. 초기spawn수정전결과도보존한다.
- 재개 wrapper가 checkpoint env를 복원하도록 수정해 기본60kg과기존40kg의충돌을방지했다. 사용자가실수로종료한본학습의최신2000iter모델에서원래40kg·optimizer로GPU6재개. 산출물 runs/jointpush/training_resume_20261012. MPS/원본저장소는변경하지않았다.


### Stage2 뷰어·평가에 checkpoint 환경 복원

- 기본 box 질량60kg 변경 후 기존40kg 모델의 평가가 physics/reference 계약에서 거부되는 문제를 수정했다. test/view는 checkpoint의 relation_experiment_config.env를 runs 아래 임시 YAML로 복원해 동일 물리·보상·RSI로 평가한다. 계약 검사를 우회하지 않으며 저장 config와 box 계약 일치를 먼저 검증한다.
- 실행 중인 학습·MPS는 변경하지 않았다. 검증 기록 runs/jointpush/checkpoint_env_fix_20261012.


### 무학습 가상 box를 Stage1 크기·실제 뒷면에 정렬

- 사용자 지적에 따라 가로 폭만 절반인1.3×2.6×2.6m 관측을1.3m 정육면체로 수정했다. 실제 박스의 뒷면·바닥에 맞춰 가로±0.65m/뒷방향0.65m/아래0.65m 배치하고 목표·corners·offset velocity를 함께 수정한다. 네 방향의 면 정렬과 물리 tensor 비변경 등 CPU5개 통과.
- 기존 Stage1 추론만 사용하며 실제 박스 크기·힘은 변경하지 않는다. config 질량은앞선변경60kg, 실행 중인 기존 학습은40kg 유지. 검증 산출물은 runs/jointpush/virtual_rear_cubes_check_20261012. GPU6·60kg·16환경600step 평가 rc=0, 목표 성공0/16, scene termination56.25%, 평균 goal 거리2.453→1.906m(30→570step). 기존 뷰어는 재실행하지 않았다.


### 공유 상자 질량60kg 설정

- 사용자 요청으로 공동 Push·무학습 반쪽 관측·공유 박스 명세의 질량을40→60kg으로 변경했다. 크기2.6m 정육면체·마찰0.6은 유지한다.
- 런타임 질량 감사는 config의 목표 질량과 비교하도록 바꿨다. YAML 질량 일치 및 관련 CPU 보상 테스트를 확인했다. 실행 중인 학습·뷰어·MPS는 중지/재시작하지 않았으며 현재 프로세스는40kg을 유지한다. 새 실행부터60kg.


### 무학습 Stage1·가상 반쪽 box 공동 Push 비교

- 사용자 제안대로 실제 공유 박스는 유지하고 가로 폭만 반씩 분할한 가상 box/goal graph를 관측에 넣는 inference-only 환경을 추가했다. Stage1 epoch29000과64입력 action head를 strict 복원하며 CA4·추가 학습은 없다. 원래 Stage1 scene attention은 유지한다.
- local corner·pose·offset velocity·goal pose를 함께 갱신하고 실제 물리 상태는 변경하지 않는다. source schema/network 검증을 갖춘 명시적 inference 전용 복원 분기를 추가했으며 기존 Stage2 계약은 유지한다.
- CPU기하2개 통과, GPU6 16환경600step 평가 rc=0. 공동 목표 성공0/16, scene termination81.25%; 평균 물리 goal 거리2.456→2.081m(30→570step), 손 중점 간격1.090→2.061m. 결과 runs/jointpush/virtual_half_stage1_check_20261012. viewer6083과 전용 test/view wrapper를 추가했다. 본학습·MPS·원본 저장소는 수정하지 않았다.


### 루트 Git 편입 후 실행 산출물 ignore 보완

- 내부 Git 경계 제거 후 복사본 코드/설정이 루트 저장소에 나타나는 것은 정상이다. 코드·로봇 asset·portable reference는 유지하고 중첩 output/output_etc, logs/summaries/metrics, 화면 산출물, TensorBoard event, PID/lock과 align_tmp를 제외했다.
- git check-ignore로 제외 대상과 주요 코드/config의 노출을 확인했다. 실행 중인 학습·MPS와 산출물 파일은 변경하지 않았다.


### 공동 Push 환경·Stage2 학습 연결 및 loco-only RSI

- koo 격리 복사본에 공유 박스/공유 목표, 두 사람의 PUSH+AT graph, reset·보상·CPA·power를 연결했다. 박스는2.6m 정육면체40kg. 현재 공동100%이며 Mixed80 분할은 사용하지 않는다.
- 공동 운반 Goal360의 PPO·task/AMP 혼합·보상 가중치를 유지했다. RSI는 loco100%의 임의 시점에서만 초기화하고 AMP expert는 koo Push의 loco0.33/Push0.67을 유지한다. reset history 반복과 partial reset 보존을 구현했다.
- Stage1 checkpoint 계약을 검증해 encoder/관측 RMS를 동결하고 CA4·확장 head를 학습한다. critic/discriminator/AMP RMS 초기화, 저장·평가·재개 계약과 GPU6 고정 스크립트를 추가했다.
- CPU10개 통과. GPU6 실제2048환경 짧은 학습·저장 모델 재개·16환경600step 평가 rc=0. runtime audit에서 실제 질량·공유 binding·loco RSI·AMP history·partial reset·finite를 확인했고 저장 모델 감사에서 encoder75개/RMS 원본 일치와 CA4/head 업데이트를 확인했다. 짧은 모델 성공률0으로 장기 성능을 확인한 것은 아니다. 원본 ksh/koo와 MPS는 수정하지 않았다.
- 현재 설정 및 실행 방법: markdowns/config.md 상단. 검증 로그/모델: 프로젝트 runs/jointpush/connected_final_check_20261012 및 push_stage1_repro_20261012.


### 공동 Push 박스 정육면체 명세

- 크기 중심의 협동 설정을 위해 박스를2.6×2.6×2.6m로 변경하고 질량40kg·마찰0.6을 유지했다. 기존 Stage1재현환경과원본은변경하지않았다. 실제공유환경연결은아직전이다.
- 해당크기에서대칭/면접촉/성공gate CPU7개통과, GPU/MPS미사용. 로그는프로젝트 runs/jointpush/push_stage1_repro_20261012/cube_reward_tests.log.

### 고정 앵커 제거·중점 대칭 보상·넓은 공유 박스 명세

- 양손 중점의 팀 중심 대칭/면 접촉/최소 간격으로 보상을 변경했다. 간격은면폭25%,좌우고정없음. 공유 박스 명세를2.6×1.3×1.3m·40kg으로변경해밀도유지. 이전고정±앵커기록은현재보상에적용하지않는다.
- CPU10개통과(보상7+모델3),diff검사통과. 물리환경미연결·GPU/MPS미사용. 설정/코드경로는markdowns/config.md를따른다.

### joint carry 기반 대칭 공동 Push 보상

- 구현: `ports/koo_push/tokenhsi/env/tasks/multi_agent/joint_push_reward.py`; 명세: `ports/koo_push/tokenhsi/data/cfg/multi_agent/joint_push_reward.yaml`. 단독 실행 env YAML이 아니라 보상 함수/설정이다. 공유 물체 환경·reset·학습 루프 연결은 아직 없다. 기존 Stage1 평가 보상은 바꾸지 않았다.
- Goal360 joint carry 기본의 개인 접촉 state/progress/success + 공동 gate된 AT state/progress/success를 각 사람에게 합산한다. 가중치는 각0.2, self1/teammate0이며 현재 상태 gate다. TeamHold·Move/Align·성공 후 접촉 포화는 사용하지 않는다. power·CPA collision·AMP 혼합은 보상 함수 밖의 환경 단계에서 별도 적용할 항목이다.
- 목표 방향을 상자 yaw 좌표로 변환해 그 방향을 가장 잘 향하는 면의 반대편(뒷면)을 고른다. 실제 뒷면 중앙에 대칭인 두 손 중심 목표를 둔다. offset은 면 반폭×0.5: 1.3m 상자는±0.325m. A/B 좌우를 고정하지 않고 양손 평균 위치를 사용하며 어느 조합이든 두 사람이 서로 다른 앵커 score≥0.9일 때 AT를 활성화한다. 손 중심 score는 exp(-10×3D거리²), 손 높이는 바닥 기준0.8m다. 손별 위치/실제 반력 판정은 추가하지 않아 carry의 기하 gate 의미를 유지한다.
- PUSH progress는 root→상자 XY 거리에서 상자 반대각선+0.2m를 뺀 pinning 거리, AT progress는 상자→목표 XY 거리에서0.1m를 뺀 값으로1/(1+d)를 계산한다. AT state는 exp(-10×3D거리²), 성공은 state≥0.9·Z오차≤0.001m·공동 gate. 두 edge의 state/progress/success를 단순 합산하며 reset/partial-reset history가 필요 없는 현재 상태 함수다.
- 목표 XY가 현재 상자 중심과 같으면 local+Y를 방향 기준으로 사용한다. X/Y 면 선택 경계에서는 앵커가 불연속적으로 바뀔 수 있다. 현재 기하는 upright/yaw 기준이며 전도 상태 처리는 공유 환경 종료 조건 연결 시 확인한다. 손 목표를 정책 입력에 추가하지 않는다.
- 입력은 roots[N,2,13], hands[N,2,2,3], ONE shared box[N,13], ONE shared goal[N,3], 실제size[3] 또는[N,3]. 반환 edge tensor 축은[N,agent,(PUSH,AT)]. caller가 실제 공유 box/goal binding을 보장해야 한다.
- CPU 검증8개 통과(보상5+모델3): 대칭/실제면·사람 교환·같은 앵커 gate-off·gate 재폐쇄·도착해도 접촉 비포화·성분합/공동AT 동일 지급·Z오차·좌표 회전/이동·목표방향별면·잘못된입력/YAML일치·기존모델 회귀. 프로젝트 로그: `runs/jointpush/push_stage1_repro_20261012/reward_model_tests.log`. GPU/PhysX/학습은 실행하지 않았다.


### PUSH+AT bundle CA4 모델 모듈

- 격리 복사본 `ports/koo_push/tokenhsi/learning/multi_agent/push_at_coordination.py`에 구현했다. 환경·보상·Stage1 관측/edge packet은 유지한다. 공동 Push 물리 환경·학습 CLI 연결은 아직 구현하지 않았다.
- Stage1은 `[source type embedding16, relation embedding32, target type embedding16] → Edge MLP64 → layer/head projection → attention bias`를 사용한다. 새 모듈은 projection 전 PUSH64·AT64 관계 feature를 동결 재사용한다.
- `[actor64, payload64, target64, PUSH64, AT64] → Grounding MLP(320→128→64)`로 사람별 작업 토큰을 만든다. PUSH edge의 owner/payload와 일치하는 AT를 요구하며 같은 payload/goal을 가진 두 작업은 공동 관계다. CA+FFN4층·64D·2head·FFN128·층별 자기/공동/기타 bias를 적용한다.
- 사용: 기존 Stage1 모델을 먼저 strict load한 뒤 `attach_push_at_coordination(model.a2c_network)`를 호출한다. action head를 `[W,0]`로 확장하고 actor encoder를 동결한다. 함수는 새 파라미터를 만들므로 optimizer 생성 전에 호출해야 한다. 관측 RMS는 기존 통계를 복원하고 별도로 eval/동결해야 한다. 이 helper는 agent/RMS/optimizer를 수정하지 않는다.
- Stage2 저장 모델 복원은 동일 Stage1 구조를 만든 뒤 helper로 확장하고 Stage2 state_dict를 strict load한다. 기존 Stage1 평가 명령은 helper를 호출하지 않아 그대로 동작한다. Stage2 checkpoint 계약·CLI resume는 별도 연결이 필요하다.
- 테스트: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=tokenhsi /home/hwanhee/anaconda3/envs/tokenhsi/bin/python -m pytest tokenhsi/tests/test_push_at_coordination.py -q` (ports/koo_push에서 실행). CPU3개 통과: 두 관계 기여·edge 순열·잘못된 bundle 거부·빈 graph finite·사람 교환·CA4 gradient·실제29000 checkpoint 초기 action 보존·encoder 동결·head 업데이트 후 grounding gradient. 결과 로그는 프로젝트 `runs/jointpush/push_stage1_repro_20261012/model_tests.log`다. 초기 context head가0이므로 첫 backward의 CA gradient는0이며 head가 업데이트된 이후 CA gradient가 열린다.


### Push Stage1 격리 복사·GPU6 추론 및 뷰어 재현

- 코드: `ports/koo_push/`는 `/home/hwanhee/koo_cvpr/stage1_fix`의 tokenhsi/lpanlib/mps 작업 파일 복사본이다. 공동운반 core는 유지한다. 데이터 심링크 12개는 원본에서 해석한 절대 경로로 연결했다. 원본 폴더는 변경하지 않았다.
- 모델: 프로젝트 `runs/jointpush/push_stage1_repro_20261012/checkpoints/`의 고정 epoch29000. GPU는 두 진입점 모두 6으로 고정한다.
- 프로젝트 루트에서 평가: `bash stage2_jointpush/scripts/push_stage1_test.sh` (2명·4환경·각자 Push·600step·1회).
- 프로젝트 루트에서 뷰어: `bash stage2_jointpush/scripts/push_stage1_view.sh`. 기존 뷰어 실행 중에는 중복 실행하지 않는다. noVNC 포트6081을 포워딩하고 `http://localhost:6081/vnc.html?autoconnect=1&resize=remote`에 접속한다. 현재 카메라는 한 사람/상자에 집중한다.
- 원래 OPEN/PUSH 혼합 장면은 `TASK_GRAPH=random_open_push`로 선택한다. 기본 push는 두 사람이 각자의 상자를 미는 Stage1이며 공동 Push가 아니다.
- 로그·metrics·화면 캡처·복사 manifest는 프로젝트 `runs/jointpush/push_stage1_repro_20261012/`에 둔다. 120step smoke와600step4환경 평가가 rc=0으로 완료했고 실제 Isaac Gym 렌더 화면을 확인했다. 작은 표본의 재현 검사이며 장기 성능 판정은 아니다.

## 2026-10-11

### CA4 Goal360 TeamHold 공동 HOLDING state 교체

- `approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_teamhold_task_embedding` env/train·train/test/VNC를 추가했다. 공동 환경의 개인 HOLDING state만 반대 앵커 조합별 최소 점수 중 최댓값 `c`의 `0.2*c`로 교체한다. 개인 progress/success·운반 gate·AT/ON_TOP·단독 보상은 유지하며 Move/Align은 없다. Reward metadata로 기존 Goal360 checkpoint와 혼용을 차단한다.
- CPU52개 통과·공동 ON_TOP putDown1개 skip, Python/셸 문법·문서 링크·diff 검사를 통과했다. GPU6·64환경32-step에서 공동 AT29/ON_TOP22·단독13의 state 교체/동일 지급, 개인 progress/success·placement gate·단독 보상 불변, gate-off451 scene-step의 연속 공동 점수와 부분 reset cache를 확인했다.
- GPU6·2048환경에서 Stage1 epoch30000·seed8429 전이→epoch2 저장→epoch3 재개,16환경32-step headless test wrapper 평가를 통과했다. Actor69개 tensor/RMS 불변·CA4 bias/context head 갱신·모델/scalar133종 finite를 확인했다. 공동1638/단독410, AMP30000개80.09/10.24/9.67%, RSI 라이브러리/초기 이력·부분 reset1024/32/32·8-step finite와 AT4/ON_TOP3 시작 단계별 Goal360 목표/pose/velocity 검사를 통과했다.
- 결과는 `output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_teamhold_task_embedding_check/{teamhold_runtime_check,reset_check,goal360_runtime_check,saved_check}.json` 및 `*_resume_check/`, `*_eval_check/`다. 본학습·실제 VNC 화면·장기 공동집기 개선은 미검증이다.

### CA4 Goal360 공동 목표·putDown 끝점 초기화 분리

- `approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding` env/train·train/test/VNC를 추가했다. 공동 loco/pickUp/carryWith는 상자 기준 독립 각도0~360°·거리2.2~2.8m로 AT 목표/ON_TOP 받침을 배치하고 물체·사람 근접/받침 bbox 겹침을 재샘플링한다. 공동 AT putDown은 보정 reference 마지막 XY에 현재 프레임과 동일한 중심 이동·yaw/translation을 적용한다.
- PutDown135개 중30개의 끝점이 기존 미사용 상자와 겹쳐, 해당 장면의 미사용3개만 목표 반대쪽 반경4m에 분산 배치했다. 전135프레임 목표·상자/상자·상자/사람 clearance와 환경 범위를 검사했다. RSI 사람/운반 상자·속도·프레임 선택·단독·AMP80/10/10·보상·SA/CA 셔플을 유지하고 Move/Align은 추가하지 않았다. 기존 config 분기는 유지하며 `joint_carry_goal_sampling` 저장/로드 계약으로 실험 혼용을 차단한다.
- CPU35개 통과·공동 ON_TOP putDown1개 skip 및 최종 추가 clearance 검사 통과. GPU6·2048환경에서 최종 Stage1 epoch30000·seed2899 전이→epoch2 저장→epoch3 재개,16환경32-step headless test wrapper 평가를 통과했다. Actor69개 tensor/RMS 불변·CA4 bias/context head 갱신·모델/scalar133종 finite를 확인했다.
- 실제 공동1638/단독410, AMP30000개80.09/10.24/9.67%, 초기 이력·부분 reset1024/32/32 및 AT4/ON_TOP3 시작 단계별1638개 RSI/목표 검사와 각4-step finite를 통과했다. 상대 목표8방향·2.2~2.8m·받침/putDown 끝점 좌표 일치와 초기 clearance를 확인했다. PutDown 목표 거리는0~3.74m다. 결과는 `output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding_final_check/{reset_check,goal360_runtime_check,saved_check}.json` 및 `*_final_{resume,eval}_check/`다. 본학습·실제 VNC 화면·장기 성능은 미검증이다.

### Mixed80 사람 번호·SA/CA 순열 토이 검사

- `tests/toy_joint_carry_permutation.py`로 기존 Mixed80 epoch15000을 GPU6·64환경(공동51/단독13)·carryWith 시작180-step에서 검사했다. 832개 장면의 사람 상태/goal/graph 참조를 함께 교환해 실제 관측 재생성→공유 RMS→encoder→CA→최종 action을 비교했다. 물리 step 전 상태를 복원하며 학습 설정/가중치는 변경하지 않는다.
- 재생성 관측·정규화 오차0, 최종 action 교환 최대 오차1.03e-5·RMSE1.39e-7 미만, SA/CA 셔플 최대7.08e-6 미만, task 행 순서 오차0이었다. Encoder 내부 최대8.57e-5로 진단 기준5e-5를 넘은 항목은 결과에 그대로 남겼다. RMS 불변·모델/비교값 finite를 확인했다.
- 결과는 `output_etc/joint_carry_permutation_20261011/permutation_check.json`, 재현 명령은 config에 있다. 동일 시뮬 상태의 정책 질의 검사이며 사람을 교환한 별도 물리 rollout·다른 모델·모든 목표 방향의 일반화 검증은 아니다. 현재 reset의 공동 회전은 사람/상자/목표 상대 배치를 다양화하지 않으므로 번호 독립성과 협동 운반 성능을 구분한다.

### CA4 공동 이동 방향 min 보상·pinning 연결

- `approach_stage2_joint_carry_mixed80_locoamp_ca4_move_task_embedding` env/train·train/test/VNC를 분리했다. 공동 HOLDING 중 두 사람의 실제 골반 world XY 이동 방향과 상자→목표 방향의 양수 cosine 최솟값×0.2를 각자 한 번 지급한다. AT0.8m·ON_TOP 목표 반대각선+0.8m 안에서는 점수1이며 단독/gate-off는0이다. 기존 heading 모드는 유지하고 새 실험에는 추가하지 않는다.
- CA4 구조·Stage1 actor/RMS 동결·head `[W,0]`·기존 reward/CPA·공동80/단독20·AMP80/10/10·paired/단독 RSI·SA/CA 셔플을 유지했다. Stage1 epoch30000·seed2899에서 Stage2 재개 없이 시작하는 명령을 문서화했다. 새 reward 계약으로 기존 CA4/Align의 직접 resume/eval을 차단한다.
- CPU63개(방향/속도 크기·heading 독립성·정지/반대 이동·pinning 경계·물체/goal 재배정·사람/edge 순열·좌표 변환·기존 회귀) 및 Python/셸 문법·diff 검사를 통과했다. GPU5·2048환경 전이→epoch2 저장→epoch3 재개,16환경32-step test wrapper 평가를 통과했다. Encoder69개 tensor/RMS 불변·각 층 bias/context head 갱신·모델/scalar136종 finite를 확인했다.
- GPU64환경32-step에서 공동 AT25/ON_TOP26·단독13의 새 보상 공유·한 번만 합산·성분 합 일치·단독/gate-off 0을 확인했다(양수 보상192 scene-step). 별도2048환경 AMP30000개 비율79.68/10.25/10.07%, RSI 라이브러리/초기 이력·부분 reset1024/32/32·8-step finite도 통과했다. 결과는 `output/approach_stage2_joint_carry_mixed80_locoamp_ca4_move_task_embedding_check/{saved_check,runtime_check,reset_check}.json` 및 `*_resume_check/`, `*_eval_check/`다. 본학습·실제 VNC 화면·장기 운반 개선은 미검증이다.

## 2026-10-10

### Mixed80 LocoAMP 4층 task CA 연결

- 기본 `approach_stage2_joint_carry_mixed80_locoamp_task_embedding`에서 CA+FFN(64→128→64)+pre-LayerNorm+residual 4층·64차원·2head 실험을 `locoamp_ca4_task_embedding` env/train·train/test/VNC로 분리했다. 각 층 bias/weight는 독립이며, grounding·관계·task 셔플·padding은 한 번 계산해 재사용한다. 보상(Align 없음)·AMP80/10/10·RSI·공동80/단독20·SA 셔플·Stage1 actor/RMS 동결·head `[W,0]` 전이를 유지한다.
- 정책 계약 `task_ca4`로 기존1층/head-only checkpoint와 격리했다. 전체 파라미터4,371,226(+117,266; +2.76%), 학습 파라미터3,844,730이다. 관련 CPU40개 및 추가된 학습 후 SA/CA 셔플 재검사, 셸 문법·문서 링크·diff 검사를 통과했다.
- GPU5·2048환경에서 지정 Stage1 epoch23000 전이→epoch2 저장→epoch3 재개, 저장 모델16환경32-step 혼합 평가를 통과했다. Actor encoder69개 tensor/RMS 불변·각 층 bias와 context head 갱신·모델/scalar133종 finite를 확인했다. 실제 AMP30000개 비율79.68/10.25/10.07%, 공동1638/단독410·RSI 라이브러리/초기 이력·부분 reset1024/32/32·8-step finite도 통과했다.
- 결과는 `output/approach_stage2_joint_carry_mixed80_locoamp_ca4_task_embedding_check/{saved_check,reset_check}.json` 및 `*_resume_check/`, `*_eval_check/`다. 본학습·실제 VNC 화면·장기 협동 성능·속도 비교는 미검증이다.

### 실제 이동 방향으로 목표를 옮긴 보상 검사

- 사용자 요청으로 AMP 유지100회 학습본(`amp05`, epoch8100)을 GPU5·64환경에서 평가했다. 동일 초기 물리 상태에서 기존 목표/기존2초 이동 방향의1.5m 목표를 비교하고, 별도로 기존 action을 그대로 재생해 목표 위치만 바꿨다. 학습·본 보상식은 변경하지 않았다.
- 목표를 이동 방향에 놓아도 정책은 움직임을 바꿔 평균 최종 거리2.87m로 멀어졌다. 고정 장면1에서는 A 전진/B 후진이 A 후진/B 전진 쪽으로 바뀌었다. 목표 입력 변화에 행동이 반응한다는 관찰이며, 정확한 원인이나 접촉 힘의 주도자를 확정하지 않는다.
- Action 재생은 원래 run 대비 body/box/action/AMP/HOLDING 보상 차이0이었다. 고정 장면1의 접근 구간에서 거리1.48→0.17m·AT0.084→0.186, 같은 시점/물리 상태의 목표 위치만 비교하면 각 사람의 task 보상+0.134·혼합 보상+0.067이었다. 세 조건의 공동 HOLDING 접근9420 step에서 AT 거리 항 감소0, 식과 실제 값 오차≤1.5e-8, 두 사람 AT 동일·보상 성분 합 일치·finite를 확인했다. 모든 총보상의 시간상 단조 증가나 학습 유도의 충분성을 뜻하지 않는다.
- 재현은 기존 toy wrapper의 `reward_eval`, `TOY_DIRECTION_FILE`, `TOY_REPLAY_FILE` 및 `toy_joint_carry_goal_reward{,_report}.py`다. 결과/그래프/시뮬 관절 영상2개는 `output_etc/align_goal_reward_20261010/`에 저장했다. Python/셸 문법·diff·MP4 전체 decode를 확인했다.

### Align epoch8000의 AMP 유지/제거 직선 운반 토이실험

- 사용자 제공 `align_tmp/..._00008000.pth`에서 공동 AT·이미 잡은 carryWith·A 초기 전방/후방 1.5m 목표만 사용하는 격리 진단 실행기를 추가했다. GPU5·2048환경에서 task 가중치0.5를 유지하고 AMP 가중치0.5/0으로 각각100회 학습했다. 기존 실행 YAML·보상식·RSI/AMP 데이터는 변경하지 않았다.
- 두 조건 모두 epoch8100 정상 종료, source 대비 actor encoder69개 tensor/RMS 불변·head/CA 갱신·모델과 기록 scalar111종×100회 finite를 확인했다. 동일 초기 상태64개(방향별32개)의 최대6초 평가에서 평균 목표 거리 감소는 학습 전−0.90/−1.52m, AMP 유지−0.80/−2.66m, 제거−0.36/−1.39m다. 음수는 목표에서 멀어진 것이다. 제거 쪽이 유지보다 낫지만 평균적인 목표 접근은 아직 실패하며,100회·단일 seed로 AMP를 주원인이라 단정하지 않는다.
- 실행/집계는 `tokenhsi/scripts/multi_agent/joint_carry_amp_toy.sh`, `tokenhsi/tests/toy_joint_carry_amp{,_report}.py`다. 결과는 `output_etc/align_amp_toy_20261010/{training_verification,summary}.json`, 거리 그래프·실제 시뮬 관절 궤적 영상이다. 구문·diff 검사와 두 MP4 전체 decode를 확인했다.

## 2026-10-09

### Mixed80 LocoAMP 공동 방향 보상 연결

- 사용자 합의대로 `approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding` env/train 및 train/test/VNC를 분리했다. 공동 HOLDING 중 목표 XY에서 가장 먼 사람의 heading 점수×0.1을 두 사람에게 각각 한 번 공유한다. 방향 포화 거리는 목표 XY 반대각선+0.8m(AT0.8m·ON_TOP약1.277m)이며 단독 환경에는 적용하지 않는다. 기존 placement progress buffer0.1·AMP80/10/10·RSI·CPA·CA/SA 셔플·동결 및 head 전이는 유지한다.
- 방향 reward 계약을 checkpoint metadata로 분리하고 `reward_terms/joint_alignment`와 공동 점수/포화 비율을 기록한다. 관련 CPU28개·Python/셸 문법·diff 검사를 통과했다. GPU5·2048환경에서 Stage1 epoch23000 전이→epoch2 저장→epoch3 재개를 확인했고 encoder69개 tensor/RMS 동결·모델 및 scalar136종 finite를 검사했다.
- 저장 모델의64환경32-step 혼합 평가(공동 AT27·ON_TOP24·단독13)에서 두 사람의 동일 방향 보상·task total에 한 번만 추가·단독/gate-off 보상0·성분 합과 total 일치를 확인했다. 전용 test wrapper의16환경32-step headless 평가도 통과했다. 실제 VNC 화면·장기 학습 개선은 미검증이다. 결과는 `output/approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding_check/saved_check.json` 및 `output_etc/joint_alignment_check/{runtime_check,resume_check}.json`이다.

### 공동운반 clone 배포 준비·외부 데이터 자동 연결

- 현재 보정 AMP12개·검사용 target·TeamHOI 원본12개·paired RSI6개를 `.gitignore` 예외로 지정했다. 영상·이전 보정 binary·checkpoint는 제외한다. `joint_carry/README.md`의 연결 전 전달 안내를 현재 clone 실행법으로 교체하고 config·structure·포팅 문서를 갱신했다.
- `prepare_joint_carry.py`가 `TOKENHSI_DATA_ROOT` 또는 `--data-root`의 기존 TokenHSI 데이터를 없거나 끊어진 링크에 연결한다. 유효한 기존 데이터는 덮어쓰지 않는다. 두 LocoAMP train/test(VNC 포함)에 원본 파일100개 존재·현재 AMP/RSI18개 SHA256 검사를 연결했다. 보상·모델·분포·기존 학습 프로세스는 변경하지 않았다.
- 별도 임시 Git index로 배포 대상만 archive하여 공백 포함 새 경로에 복원했다. AMP12개·RSI6개 포함, 공동 RSI319개 로드·원본12개 해시·경로/파일 보존/오염 검출 CPU4개·셸 문법·diff 검사를 통과했다. 실제 index는 변경하지 않았으며 commit/push는 수행하지 않았다.
- 새 경로·GPU5·2048환경에서 Stage1 epoch23000으로 두 모델의 짧은 학습을 완료했다(`MAX_ITERATIONS=1`, 실제 저장 epoch2). CA는 RSI 캐시2279 profile을 새로 생성했고 head-only는 데이터 경로 재지정 없이 링크·캐시를 재사용했다. 두 모델 모두 actor69개 tensor 원본 유지·모델/기록 scalar133종 finite·정상 종료를 확인했다. 결과: `output/clone_portability_{ca,head_only}_check/portability_check.json`. 동일 서버의 경로 이관 검사이며 다른 서버의 드라이버/환경까지 검증한 것은 아니다.

### output 검증 산출물 보관 정리

- 사용자 요청으로 비활성 검증·분석·이전 영상 디렉터리19개를 `output_etc/`로 이동했다. `output/`에는 본학습 결과4개·`rsi_cache`·현재 전체 보정 영상/검사만 남겼다. 이동 목록은 `output_etc/archive_20261009_cleanup.json`이다.
- 같은 파일시스템의 rename으로 디렉터리 inode를 보존했고 실행 중 프로세스가 이동 경로를 사용하지 않는지 확인했다. 문서·이전 데이터 검증 report 경로·보조 영상/검사 스크립트 출력 경로를 갱신했다. 현재 학습 config·AMP 데이터·캐시·checkpoint는 변경하지 않았다.
- 이동19개·문서 링크·데이터 검증 report4개·보조 스크립트 Python 문법·diff 확인 통과. 두 본학습 프로세스는 이동 후에도 실행 중이다.

### 자세 보존 보행 전체12개·LocoAMP CA/head-only 연결

- 승인받은 자세 보존 방식을 후진9개·옆걸음3개에 적용해 `joint_carry/teamhoi_retarget_posture_all/`로 분리했다. IK 범위 안쪽0.001rad 여유로 quaternion 보간의 작은 한계 초과를 제거했다. 무릎 중앙값의 원본 대비 변화는 최대3.1°이며 원본·이전 보정본 해시는 보존했다. Male2 B9/B11/B13/B15만 중간 샘플을 추가해60fps, 총4458프레임이다.
- GPU5·2048환경에서4458/4458프레임의0.1초 초기 물리 검사·프레임/중간 시점 관절 범위·발 접지·FK 검사를 통과했다. 시뮬 body/FK 차이 최대0.12mm 미만, CPU/GPU 실제 AMP3200×1320 finite. 개별4열 비교12개·후진/옆걸음 모아보기2개 전체 MP4 decode 통과. 결과는 `output/teamhoi_retarget_posture_all_check/`다. 연속 보행 안정성·미끄러짐0·학습 개선을 보장하는 검사는 아니다.
- 두 `approach_stage2_joint_carry_mixed80_locoamp_{task_embedding,head_only}` env/train·train/test/VNC를 새 데이터 해시로 연결했다. AMP만 기존80%·후진10%·옆걸음10%로 보강하며 사람별 방향 배정은 없다. 공동80/단독20, 기존 reward·CPA·RSI·SA 셔플은 유지한다. Stage1 actor/RMS 동결, CA는128입력 `[W,0]`, head-only는CA 모듈 없이64입력 `W`로 초기화하고 action head를 학습한다. Critic·AMP 판별자는 학습하며 이전 보정 데이터 checkpoint는 계약 불일치로 구분한다.
- 관련 CPU18개 및 최종 데이터 계약4개 재검사, Python/전용 셸6개 문법·diff·문서 링크 검사를 통과했다. GPU5·2048환경에서 두 모델 각각 epoch2 저장, epoch3 재개, 16환경32-step headless 평가 로드를 확인했다. Actor69개 tensor·관찰 RMS 불변, head64/128·CA 유무·head/critic/discriminator 갱신과 scalar finite를 검사했다. 확인용 결과는 각 `output_etc/approach_stage2_joint_carry_mixed80_locoamp_*_posture_{check,resume_check,eval_check}/`다.
- 실제 AMP30000개 샘플의 원래/후진/옆걸음 비율은79.68/10.25/10.07%, 두 새 라이브러리와 원래 RSI 라이브러리 분리·carry label·시간 범위·공동1638/단독410·부분 reset1024/32/32의 미선택 상태 보존·8 control step finite를 확인했다. `head_only_posture_check/reset_check.json` 참조. 본학습은 시작하지 않았다.

### TeamHOI 과도한 무릎 굽힘 수정 pilot

- 사용자 영상 검토로 이전 IK의 자세 왜곡을 확인했다. 골반 고정 상태에서 발 목표를 맞춘 뒤 접지하던 순서가 원본보다 과도한 굽힘을 만들었다. 이전 전체12개 보정본과 연결된 LocoAMP 본학습은 사용 보류로 문서화했다. 기존 학습 프로세스·AMP/RSI/config 데이터 해시는 바꾸지 않았다.
- `--posture-first`로 별도2개를 생성한다. 골반과 발 목표의 공통 높이를 IK 전에 정렬하고 원본 무릎 자세 유지 항을 강화했다. 기존 loco5개를 대조했으며 무릎 중앙값은 후진 원본9.5°→이전60.5°→새11.0°, 옆걸음22.3°→56.8°→20.3°다. XY 경로·root 방향·원본 해시를 유지한다.
- GPU5·2048환경의0.1초 초기 충격 검사는253/253·530/530, CPU/GPU 실제 AMP640×1320 finite, 관절 범위·프레임/중간 시점 발 접지·FK/시뮬 body 일치 검사를 통과했다. 실제 FK 차이0.12mm 미만, 최저 발 여유 약0.4–1.2cm. Python 문법·diff·두 MP4 전체 decode 검사도 통과했다.
- **한계:** 후진의 바닥3.5cm 이내 발 표면 속도 proxy 평균은0.42m/s(이전0.22), 옆걸음0.11m/s다. 이는 접촉 미끄러짐 측정이 아니며 자연스러움·미끄러짐0·폐루프 보행·학습 효과는 미검증이다. 발 목표 오차는 최대3.1cm를 허용한 자세 보존 결과다. 새 데이터는 학습 미연결이며 나머지10개에 아직 적용하지 않았다.
- 결과: `output_etc/teamhoi_retarget_posture_pilot_check/{report,gait_comparison}.json`, `backward.mp4`, `sideways.mp4`. 영상은 원본/이전/새 보정/기존 걷기의4열 기구학 재생이며 기존 걷기는 별도 클립 반복이다. 재현 스크립트는 기존3개에 `--posture-first`, 자세 비교는 `joint_carry/scripts/compare_teamhoi_gait.py`다.

### TeamHOI 후진·옆걸음 각1개 보정 및 GPU 검증

- 사용자 승인으로 Male1 B10 후진·CMU141_33 옆걸음을 `teamhoi_retarget_pilot/`에 별도 변환했다. 현재 asset의 관절 범위 내 다리 IK, 이동하는 발의 지면 여유, 부드러운 지지 발 높이 기반 root Z 보정을 적용했다. Root XY/방향은 float32 정밀도 내 보존하며 실제 체형으로 FK·선속도·각속도를 재계산했다. 원본 SHA256은 불변이다.
- 실제 10-frame carry AMP 경로 CPU/GPU·1320차원·finite, 프레임/중간 시점의 관절 범위·발 mesh 바닥 관통·속도와 경로 정합성을 검사했다. GPU5·2048환경·GPU PhysX에서 후진253/253·옆걸음530/530프레임의0.1초 초기 충격 검사 통과, 실제 simulator body와 FK 최대 차이0.12mm 미만. 낮은 쪽 발의 바닥 여유는 약0.4–1.3cm다.
- **한계:** 바닥 근처 발 표면 속도 proxy 평균은 후진0.22m/s·옆걸음0.10m/s로 미끄러짐0을 보장하지 않는다. 8초 자유 root PD는 기존 모션 대조군도 넘어지므로 학습 가능성 판정에서 제외한다. AMP 학습 개선·폐루프 모방·joint RSI는 검증하지 않았다. 나머지10개·학습 config·실행 중 학습은 변경하지 않았다.
- 재현: `joint_carry/scripts/{retarget_teamhoi_reference,check_teamhoi_retarget,preview_teamhoi_retarget}.py`. 결과와 원본/보정 기구학 비교 영상은 `output_etc/teamhoi_retarget_pilot_check/`의 `report.json`, `backward.mp4`, `sideways.mp4`. 보정본은 AMP에 아직 연결하지 않았다.

### TeamHOI reference 원본 호환성·GPU 물리 검사

- 후진9개·옆걸음3개 원본은 기존 MotionLib/32-DOF 및 실제 10-frame carry AMP 생성 경로에서 로드됐다(384×1320, finite). 다만 원본 다리 skeleton offset이 현재 asset과 달라 동일 DOF의 발 위치가 클립 평균5.4–7.6cm 어긋나고 실제 발 mesh 기준 최대10.2–13.6cm 바닥 관통이 계산됐다. 기존 reference 대조군은 발 위치 평균 오차0.01cm 미만이었다.
- GPU5·2048환경·실제 phys_humanoid_v3·GPU PhysX에서 모든3721프레임을 0.1초 검사했다. root 속도 변화≤3m/s·변위≤0.3m 기준2064/3721(55.5%) 통과, 기존 모션 대조군238/238 통과. 8초 자유 root PD 재생은 대조군도 넘어져 AMP 사용성 판정 지표에서 제외했다. 최종 root/DOF는 모두 finite였다.
- 결과/재현 스크립트: `output_etc/teamhoi_reference_compatibility_20261009/`의 `report.json`, `amp_history.json`, `check.py`. 원본 즉시 투입은 권장하지 않으며 현재 체형으로 재타게팅·접지/속도 정합성 보정 후 재검증이 필요하다. 학습 config·RSI·실행 중 학습은 변경하지 않았고 변환 및 학습 성능 검증은 미수행이다.

### TeamHOI 후진·옆걸음 reference 다운로드·미리보기

- 사용자 요청으로 공식 TeamHOI commit `6fdc9885f8a9c82854be99adca6f2570e67ac63a`의 `near_table.yaml`에 포함된 후진9개·옆걸음3개를 `joint_carry/teamhoi_reference/`에 다운로드했다. 원본 경로·SHA256·프레임 정보는 `manifest.json`, 출처 설정·라이선스도 함께 보존했다.
- 12개 모션의 finite 배열·단위 quaternion·SHA256을 확인하고, 원본 FK·몸 방향/이동 방향 화살표를 표시하는 CPU 재생 스크립트와 두 MP4를 만들었다. 이는 reference skeleton 미리보기이며 물리 재생·현재 32-DOF로의 전환·AMP 학습 연결은 수행하지 않았다.

### 공유 목표 색상·과제별 마커 통일

- 사용자 지정 `stage2_team` shared9 시각화를 참고해 공유 물체·받침·AT 목표와 공동 과제 마커를 노란색으로 통일했다. AT 목표를 마지막 owner 색으로 덮어쓰던 경로를 제거하고 ON_TOP 와이어 큐브·SIT 수평 원·CLIMB 3축 십자를 적용했다.
- 실제 joint AT/ON_TOP graph와 렌더 API 대역을 사용한 CPU 색상 검사, 단일 graph 소유자 집계·마커 정점·Python 문법·diff 검사를 통과했다. 실제 VNC 화면은 미확인이고, 실행 중 뷰어에는 재시작 후 반영된다.

## 2026-10-08

### BEFORE 80%·독립 20% task CA 연결

- GPU5 기본의 `approach_stage2_before_task_embedding` env/train YAML·train/test/VNC를 추가했다. 현재 AT prerequisite으로 B state/success·포화를 gate하고 progress/HOLDING은 유지한다. 공통27개 크기·세 BEFORE 각546/독립410환경·Stage 1 RSI/AMP·SA/CA token 셔플을 연결했다. Carry-only source/기존 joint Stage 2 checkpoint와 구분한다.
- 공유 상자 상태는 A가 한 번만 결정하며 B reference·속도·AMP history를 정렬한다. 실제 source/support 크기로 RSI를 조회하고 geometry 재시도에서 skill을 유지한다. 부분 reset의 상태 보존과 평가 skill별 과제 적합성도 확인했다.
- CPU94개·셸/diff 검사, GPU5·2048환경 짧은 학습/epoch2 저장·epoch3 재개 통과. Actor69개 tensor·관찰 RMS6개 불변, NONE/SELF/BEFORE 갱신·COUPLED0, scalar182종/364값 finite. AMP 대응·1024개 혼합 및 각32개 그룹 부분 reset·16환경32-step headless 평가 경로를 확인했다. 결과: `output/approach_stage2_before_task_embedding_check_final/{saved_check,reset_check,resume_check}.json`.
- **미완료:** 초기 자세0.1초 물리 진단은1575/2048 통과·473개 미통과다. geometry reset 실패0과 별개이며 합성 RSI 물리 안정성은 보완이 필요하다. 강화 캐시는 fallback 증가와 잔여 불안정 때문에 적용하지 않았고 기존 Stage 1 비율/선별 기준을 유지했다. 본학습·장기 성능·VNC 화면은 실행/검증하지 않았다.


### 공동 80%·단독 20% Mixed80 실험 분리

- 사용자 승인으로 `approach_stage2_joint_carry_mixed80_task_embedding` env/train config·train/test/VNC를 추가했다. 2048환경 중 공동1638·단독410을 생성 시 고정하며 공동 AT/ON_TOP은 scene별50/50, 단독은 agent별 독립50/50이다. 기존100% 공동운반 실행은 유지했다.
- 단독은 Stage 1 carry 상자/받침 크기·크기별 물리 선별 RSI(40/10/40/10)·reference AMP history를 재사용한다. 공동 paired RSI·반복 AMP history와 reset을 단일 물리 commit으로 결합했다. 공동 SELF/COUPLED와 단독 SELF/NONE을 기존 task CA에 연결하고, 단독 placement는 자기 현재 HOLDING으로 gate한다. 새 reward 계약으로 기존 Stage 2 checkpoint와 구분한다.
- 관련 CPU **64개**, Python/셸 문법·새 문서 경로·diff 검사 통과. GPU5·2048환경·MAX_ITERATIONS=1 학습은 epoch2 저장, scalar133종/266값 finite, 물리 reset 실패0이었다. 지정 Stage 1 epoch11000 대비 actor69개 tensor·관찰 RMS 불변, 세 CA bias 갱신을 확인했다.
- 실제1638/410 분할·서로 다른 payload/goal의 SELF/NONE·공동/단독 AMP 이력·carry demo label·1024환경 혼합 및 각32환경 단일그룹 부분 reset에서 미선택 상태/관측/AMP 보존을 확인했다. 11 control step finite. 결과는 `output/approach_stage2_joint_carry_mixed80_task_embedding_check/{saved_check,reset_check}.json`이다.
- 저장 모델의 단독 AT/ON_TOP 혼합과 공동 ON_TOP을 각각16환경·32-step headless 평가해 전용 test 실행/로드 경로를 확인했다. 본학습·단독 능력 유지 및 공동운반 장기 성능·VNC 화면은 검증하지 않았다.

### 지정 carry distill epoch11000의 Stage 2 전이 허용

- 사용자 지정 `stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00011000.pth`를 확인하고, 호환되는 carry-only distill variant를 Stage 2 source 허용 목록에 추가했다. Tensor shape·키 검증은 유지하고 실행 가이드의 source를 갱신했다.
- 실제 checkpoint epoch11000·157개 tensor 전이·69개 actor 파라미터 동결·finite action을 CPU에서 확인했다. 관련 CPU 10개와 diff 검사 통과. 학습은 실행하지 않았다.

## 2026-10-07

### Joint carry collision을 기존 shared9 CPA로 변경

- 사용자 지정 `stage2_team/output/approach_stage2_rescue_shared9_cpa_team03`의 저장 config와 `collision_reward.py`를 대조해, root XY 접근 방향 × CPA 거리 위험도 × `0.99^(t/control_dt)`를 적용했다. 기존 계수0.5·거리0.7m를 유지하며 정지/동일속도/멀어지는 쌍에는0, static 거리 항은 중복 가산하지 않는다. 공동운반 외 기존 실험 경로는 유지했다.
- CPA mode/discount를 joint reward 계약에 기록했다. 이전 distance Stage 2 checkpoint의 직접 resume/eval은 거부하며 Stage 1 source 전이는 유지한다. 실행 가이드·코드 역할 문서를 갱신했다.
- 관련 CPU 24개 통과(접근/이탈/스침/정지·제어 dt 감쇠·순열·checkpoint 및 기존 Stage 2 회귀). GPU 5·2048환경에서 실제 reward 경로의 네 사례와 8 control step finite/bounds, 초기 물리·AMP·부분 reset을 확인했다. 원본 CPA와 1024개 4-agent 무작위 배치의 최대 차이는1.35e-7 미만이다. 결과는 `output/approach_stage2_joint_carry_cpa_check/reset_check.json`; 본학습은 실행하지 않았다.

### Joint carry ON_TOP 받침 크기 축소

- 사용자 요청으로 받침을 72×100×30cm에서 운반 상자와 가로·세로가 같은 **52×80×30cm**로 변경했다. Env YAML·크기 계약·실행 가이드를 동기화했다.
- 관련 CPU 6개 통과. GPU 5·실제 Stage 2 2048환경에서 새 asset 크기·0.1초 초기 물리 검사·AMP 초기 이력·1024환경 부분 reset을 확인했다. 결과는 `output/approach_stage2_joint_carry_support_size_check/reset_check.json`이다. 실제 올려놓기 성능은 이번 검사에 포함하지 않는다.

### Joint carry Stage 2 실행 연결

- 사용자 승인으로 `approach_stage2_joint_carry_task_embedding` env/train YAML·train/test/VNC를 연결했다. 원본 task embedding 기반 actor/RMS 동결, scene별 joint AT/ON_TOP 50/50, 2명·4물체, 52×80×40cm payload·72×100×30cm 받침을 쓴다. 기존 Stage 1·29번과 config/output/checkpoint를 분리했다.
- 양손 중점의 object-local Y ±25cm 앵커 중 자유 선택과 반대 앵커 동시 HOLDING gate를 구현했다. 자기 HOLDING은 항상 보상하고 placement state/progress/success만 gate한다. 포화·성공 latch는 제거했다. CA의 task packet/KV/relation bias를 함께 셔플하고 SA 복원 순서와 action 대응을 유지한다.
- 보정 snapshot의 두 사람·공유 상자·root/DOF 속도를 같은 행에서 공동 복원하며 부분 reset은 단일 root commit을 쓴다. RSI는 AT 40/10/40/10, ON_TOP 50/10/40/0(loco/pickUp/carryWith/putDown)이다. AMP는 기존 single-human carry 전문가만 유지하고 reset 상태 반복 이력·carry label·demo/replay matching을 연결했다. 평가 skill 축소가 expert 인덱스를 바꾸지 않게 했다.
- 관련 CPU **59개 통과**: 보상 gate/비포화·앵커 대칭/회전·RSI 원본 대응/속도·config/checkpoint 격리·SA/CA 순열 출력 및 gradient·reload·기존 Stage 1/2 회귀. GPU 5·2048환경·MAX_ITERATIONS=1 학습은 epoch2 저장, 131종 scalar/262값 finite. Actor69개 tensor/RMS 불변, CA bias·확장 head·critic·AMP 갱신을 확인했다.
- 실제 Stage 2 reset의 2048환경 0.1초 물리 검사(loco905/pickUp217/carryWith832/putDown94) 모두 통과. 공유 binding·속도·AMP 초기 이력·carry demo label·1024개 부분 reset의 미선택 상태/관측/history 보존을 확인했다. 학습 재개와 4가지 16환경·32-step headless 평가도 완료했다.
- 결과는 `output/approach_stage2_joint_carry_task_embedding_check/`의 `saved_check.json`, `reset_check.json`, `resume_check.json`과 `_check_joint_carry_*` 평가 폴더에 있다. 임시 미학습 source의 연결 검사로, 네 평가의 운반 완수율은0이다. 본학습·VNC 화면·장기 공동운반·실제 하중 분담·완료 후 손 놓기는 검증하지 않았다.


### Joint carry GPU 5 RSI 검사·TeamHOI AMP 확인

- 사용자 요청으로 `output/joint_carry_gpu5_check_20261007/check_gpu.py`에서 GPU 5·GPU PhysX/GPU pipeline·2048환경을 검사했다. 52×80×40cm·밀도100의 공유 상자와 두 사람을 같은 snapshot에서 복원하고 공통 yaw 0/90/180/270°를 적용했다. 기존 보정본은 변경하지 않았다.
- pickUp/carryWith/putDown 55/129/135개 snapshot 전부 포함, 반복 배치 385/853/810환경 모두 0.1초 초기 충격 기준 통과. 부분 reset 1024환경도 통과, root/DOF 위치 복사 오차0·미선택 root/DOF tensor 보존을 확인했다. 최대 root 속도 변화1.097m/s·변위0.195m, 상자 속도1.913m/s·변위0.172m 미만이다. 결과는 `report.json`, 환경별 값은 `per_env.npz`, 실행 로그는 `run.log`에 있다.
- 이는 별도 2인/1상자 물리 검사이며 Stage 2 reset/AMP 연결·장기 공동 운반 검증은 아니다. 손 접촉 진단은 손-상자 근접+net force의 대용 지표로, 물체별 접촉력 판정과 구분한다.
- TeamHOI 공식 코드와 로컬 사본에서 single-human locomotion/pickup reference, 근거리 팔/손 AMP masking·두 discriminator 보상 혼합, 전원 양손 근접 조건의 운반 보상 gate를 확인했다. 공동 모션을 AMP expert로 즉시 넣지 않고 기존 carry expert를 우선 사용하는 안을 제안했으며, AMP 설정 자체는 변경하지 않았다.

### Joint carry RSI 재생성·물리 재검증

- 사용자 요청으로 원본 B19/B20/B21에서 복제·32-DOF 팔 IK·CPU PhysX 검사를 다시 실행했다. 결과는 `output/joint_carry_rsi_rebuild_20261007/`에 분리했다. 보정 전 통과 수 29/0/0개에서 보정 후 pickUp/carryWith/putDown 55/129/135개로, 총 319개 snapshot의 모든 배열이 기존 인계본과 동일하다.
- 입력 해시 12개·snapshot shape/finite/quaternion/reference 대응·선별 조건·ZIP 무결성을 확인했다. 기존 인계본 79개 파일·의존 파일 3개도 검사 통과했다. 근거는 `verification.json`과 `physics_after/report.json`이다.
- `corrected/before_after/`에 보정 전후 영상, `pd_videos/`에 reference 대 실제 PD 추종 영상을 생성했다. 0.1초 초기 상태 검사는 통과했으나 연속 PD 추종은 균형을 잃었다. 학습 reset·AMP 연결과 GPU 학습 검증은 수행하지 않았다.

### Joint carry task CA 연결 준비

- `joint_carry_spec.py`에 2명·4물체의 joint AT/ON_TOP scene별 50/50 sampler를 추가했다. `task_coordination.py`는 동결 task embedding과 최종 actor/payload/target 토큰을 256→128→64로 결합하고, 2-head CA에 NONE/SELF/COUPLED zero-init bias를 적용한다. 공유 물체→목표의 중복 Stage-1 bias는 Stage 2에서 한 번만 반영한다.
- Task embedding 전이에 원본·네 과제 distill source를 허용하고 기존 actor 동결·action head `[W,0]` 확장을 연결했다. 지정한 외부 distill epoch1000 checkpoint의 157개 tensor 전이·69개 actor 파라미터 동결·finite action 출력을 CPU에서 확인했다. 관련 CPU 22개 통과, diff 검사 통과.
- **미완료:** RSI 초기화·팀 보상 선택 답변 대기. 전용 env/train config·실행 스크립트·GPU 5 시뮬레이션 검증은 아직 완료하지 않았으며 실행 가능한 실험으로 등록하지 않았다.

### 네 과제 task embedding unified teacher distillation

- 사용자 요청으로 `approach_scenario_stage1_unified_size_rsi_task_embedding_distill` env/train YAML·train/test/VNC·output을 추가했다. 원래 네 과제 sit/climb/carry_at/carry_ontop 10/25/32.5/32.5%와 크기·RSI·AMP·보상·task embedding/type projection을 유지하고, 기존 unified teacher KL 경로를 연결했다. SIT→Sit, CLIMB→Climb, 두 carry→Carry로 지도하며 기본 계수는 0.001이다. Carry-only와 원본 config·checkpoint를 별도로 유지한다.
- 관련 CPU **56개 통과**: 원본 설정 동일성·checkpoint 격리·16개 task 쌍의 teacher 라우팅/goal/물체 binding·회전된 SIT 방향·네 task embedding gradient·teacher 동결·critic 비의존·AMP family 분포 및 carry/task embedding/RSI/unified 회귀. Python/셸 문법·문서 링크/명령 경로·diff 검사 통과.
- **GPU 5·MPS·2048환경·MAX_ITERATIONS=1** 검증 학습을 완료했다(epoch 2, frame 262144). 공통 RSI 캐시에 누락 profile 3,622개를 추가해 총 10,262개를 확보했다. Scalar **252종/504값 모두 finite**, 물리 reset 실패·단독 HOLDING label·공유 primary 물체 0, 진단 CSV 16행의 owner binding·자기/동료 보상 오류 0이다. 실제 네 task teacher label·KL actor/embedding gradient·teacher 동결·critic KL gradient 없음을 확인했다.
- 저장 학생으로 sit/sit RSI, climb/climb RSI, carry_at/carryWith, carry_ontop/carryWith 각각 **16환경·32-step headless 평가**와 JSON 저장을 완료했다. Output은 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_check/`와 `_check_<task>/`다. 연결 검증이며 장기 성공률·VNC 화면은 미검증이다. 새 본학습을 시작하지 않았고 기존 carry 본학습은 유지했다.

### Carry-only task embedding unified teacher distillation

- 사용자 요청으로 `approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill` env/train YAML·train/test/VNC·output을 분리했다. Task/NONE/SELF embedding과 H/O/G 타입 쌍 projection은 유지하고 carry_at/carry_ontop만 agent별 50/50 독립 샘플링한다. 실제 source 크기도 carry 범위로 제한하고 크기별 과제 확률·AMP carry family 100%를 연결했다. RSI 40/10/40/10·물리 검사·자기 보상 합계·634-D packet은 유지한다.
- 보존된 `distill/`의 원본 unified teacher adapter와 rollout label·Gaussian KL 경로를 현재 PPO/AMP에 이식했다. 원본 `ckpt_stage1.pth`의 Carry 행동 분포를 동결해 계수 0.001로 지도하며 AT graph goal·ON_TOP 회전 bbox 목표·reset body·scene/agent minibatch 정렬을 따른다. 기존 AMP family matching·정규화와 네 과제 config는 유지했다. 새 variant로 checkpoint 혼용을 차단하며 학생 평가는 teacher 없이 실행한다.
- 관련 CPU **63개 통과**: 원본 teacher strict 로드, 네 carry 조합·goal/물체 binding·회전 높이·reset·label shuffle, KL 공식·teacher 동결·학생 embedding gradient·critic 비의존, carry 크기/AMP·기존 task embedding/Size RSI/unified 회귀. Python/셸 문법·새 문서 링크/명령 경로·diff 검사 통과.
- 지정한 **GPU 5·2048환경·MAX_ITERATIONS=1** 확인 학습을 완료했다(epoch 2, frame 262144). 최초 RSI 캐시 **6,640개 profile**을 생성했고 scalar **212종/424값 모두 finite**, 물리 reset 실패·단독 HOLDING/SIT/CLIMB label·공유 primary 물체는 0이다. 초기 두 update의 KL/PPO actor gradient 비율은 0.345/0.228이며 teacher 동결·critic KL gradient 없음·actor/projection/embedding 갱신을 확인했다. 진단 CSV 16행의 owner binding·자기/동료 보상 합산 오류는 0이다.
- 저장 학생으로 carry_at/carry_ontop + carryWith 각각 **16환경·32-step headless 평가**와 JSON 저장을 완료했다. Output은 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_check/` 및 `_check_carry_at/`, `_check_carry_ontop/`이다. 두 짧은 평가의 운반 완수율은 0으로, 실행 경로만 확인한 결과다. 본학습·장기 성능·VNC 화면은 검증하지 않았고 기존 GPU 프로세스와 `distill/` 원본은 변경하지 않았다.

### Task embedding·물리 타입 쌍별 projection 실험

- 사용자 합의대로 `approach_scenario_stage1_unified_size_rsi_task_embedding` 전용 env/train config와 train/test/VNC·output을 연결했다. Actor/critic 각각 task 4개+NONE+SELF의 `Embedding(6,64)`와 `[src H/O/G,dst H/O/G,layer,head,64]` projection을 사용한다. 카테고리 간 projection은 공유하며 역할 입력·중간 MLP·관계 message는 없다. ON_TOP의 H→운반 상자와 H→받침은 같은 bias를 받는다.
- 공유 task MLP와 env의 variant·train의 mode만 다르다. 보상·과제·RSI·AMP·634-D packet·캐시·GTA·순열 경로를 유지하고 전용 checkpoint 계약으로 혼용을 거부한다. Bias encoder는 각 4,992개, 전체 모델은 4,130,626개 파라미터다.
- 관련 CPU **91개 통과**: 새 수식·타입/카테고리 공유·NONE/SELF·역방향/padding·endpoint 재매핑·토큰/edge/task 순열의 출력과 gradient·6개 embedding 업데이트·RSI/AMP/보상·checkpoint 및 기존 실험 회귀. Python/셸 문법·문서 링크/명령 경로·diff 검사도 통과했다.
- **GPU 0·2048환경·MAX_ITERATIONS=1** 확인 학습 완료(저장 epoch 2, frame 262144). 기존 RSI 캐시 hit, scalar **233종/466값 모두 finite**. 물리 reset 실패·단독 HOLDING·공유 primary 물체·동료 보상 기여는 0이고 진단 CSV 16행의 binding·보상 합산 오류도 0이다. 양쪽 branch의 6개 category gradient와 9개 타입 쌍 projection 업데이트를 확인했다.
- 저장 모델로 carry_at/carry_ontop + carryWith 각각 **16환경·32-step headless 평가**와 JSON 저장을 완료했다. Output은 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_check/`, `_check_carry_at/`, `_check_carry_ontop/`이다. 기존 학습 세 개는 유지했다. 새 본학습·장기 성능·VNC 화면은 검증하지 않았다.

### Task 3종 실행 가이드 정리

- `markdowns/config.md` 상단에 빠른 확인을 추가하고 message·공유 MLP·split의 구조 차이, 네 task와 역할별 edge, 보상·RSI·AMP·평가 기본값 및 bias/성공률 해석을 정리했다. 실제 로컬 데이터 경로와 유효한 12개 심링크를 확인해 반영했다.
- 2026-10-06 23:10 KST에 확인한 본학습 성공률과 bias checkpoint 기준을 과거 확인 기록으로 명시했다. 실시간 상태 및 별도 평가 성공률과 구분하고 전체 분석 산출물을 연결했다.
- 실제 코드·YAML·wrapper와 대조했고 문서 링크 34개·실행 script/config 경로 99개, 내부 anchor·코드 블록·`git diff --check`를 확인했다. 문서만 수정했으며 학습·평가를 추가 실행하지 않았다.

## 2026-10-06

### Task별 독립 MLP·projection 비교 실험

- 사용자 요청으로 `approach_scenario_stage1_unified_size_rsi_task_mlp_split` env/train YAML·train/test/VNC·output을 추가했다. 기존 task MLP의 역할 embedding·task embedding 표는 유지하고 sit/climb/carry_at/carry_ontop별 MLP와 layer/head projection을 분리했다. 한 task 내부 역할 연결은 같은 전용 MLP를 쓰며 NONE/SELF 배경은 기존 별도 MLP다. Actor/critic은 독립이고 관계 메시지는 없다.
- 과제·보상(`self=1, teammate=0`)·RSI·AMP·크기·634-D packet은 기존 task MLP와 동일하다. 총 36개 task/역할 조합만 인코딩한다. Task encoder는 각 35,552개, 전체 추가 파라미터는 52,992개(+1.27%)다. 전용 variant/mode/fusion과 weight shape로 기존 checkpoint 혼용을 거부한다.
- 관련 CPU **84개 통과**: task별 weight/gradient 독립성, 공유 weight 복제 시 출력 및 gradient 합 동등성, 16개 과제 쌍·순열·역할/물체/goal binding·자기 보상·RSI/AMP·checkpoint 격리·기존 실험 회귀. Python/셸 문법·문서 링크와 명령 경로·diff 검사 통과.
- **GPU 0**, 2048환경·`MAX_ITERATIONS=1` 확인 학습 완료(저장 epoch 2, frame 262144). 기존 RSI 캐시 hit, scalar **233종/466값 모두 finite**, 물리 reset 실패·단독 HOLDING 샘플·동료 보상 기여 **0**. Actor/critic의 네 task MLP·projection 모두 업데이트됐고 진단 CSV의 binding·보상 합산 오류는 0이다.
- 저장 checkpoint로 carry_at/carry_ontop + carryWith 각각 **16환경·32-step headless 평가**를 완료했다. 검증 output은 `output/approach_scenario_stage1_unified_size_rsi_task_mlp_split_check/`, `_check_carry_at/`, `_check_carry_ontop/`이다. 기존 본학습 두 개를 유지했다. 새 본학습·장기 성능·VNC 화면·공유 대비 통제된 속도 비교는 수행하지 않았다.

## 2026-10-05

### 통합 task·역할 MLP bias 실험

- 사용자 요청으로 `approach_scenario_stage1_unified_size_rsi_task_mlp` env/train YAML·train/test/VNC·output을 분리했다. Env는 task message와 variant만 다르며, sit/climb/carry_at/carry_ontop 분포·3연결·자기 보상 합계(`self=1, teammate=0`)·크기·RSI·AMP·634-D packet을 공유한다.
- Task·출발/도착 역할 임베딩을 `64→64→64` MLP와 layer/head projection으로 attention bias에 연결했다. NONE/SELF 배경도 기본 Size RSI의 MLP 방식이며 actor/critic은 독립이다. 관계 메시지 파라미터·가산 경로는 없고 GTA는 유지한다. 전용 variant/fusion 계약으로 기존 checkpoint 혼용을 거부한다.
- 관련 CPU **70개 통과**: 새 설정 동일성·역할별 bias·메시지 부재·순열 출력/gradient·optimizer 업데이트·RSI/AMP binding·자기 보상·checkpoint 분리 및 기존 task-message/typed-bias/Size RSI 회귀. Python/셸 문법·문서 링크/실행 경로·diff 검사 통과.
- **GPU 0**, 2048환경·`MAX_ITERATIONS=1` 확인 학습 완료(저장 epoch 2, frame 262144). 내려받은 RSI 캐시 hit, scalar **233종/466값 모두 finite**, 물리 reset 실패 **0**. 저장 모델에 관계 메시지 파라미터가 없고 actor/critic task MLP bias projection 업데이트를 확인했다.
- 저장 checkpoint의 carry_at/carry_ontop + carryWith **16환경·32-step headless 평가**를 각각 완료했다. 결과는 `output/approach_scenario_stage1_unified_size_rsi_task_mlp_check/`, `_check_carry_at/`, `_check_carry_ontop/`이다. 본학습·장기 성능·VNC 화면은 검증하지 않았고 기존 학습은 중단하지 않았다.

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
