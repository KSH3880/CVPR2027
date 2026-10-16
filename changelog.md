# Changelog

최신 변경부터 기록한다. 현재 실행법은 [config.md](markdowns/config.md), 코드 위치는 [structure.md](markdowns/structure.md)를 참조한다. 실행 중인 GPU/PID는 이 파일에 고정하지 않고 실제 프로세스로 확인한다.

## 2026-10-16

### SA4 LocoPlatform GraspView · Stage2 학습 연결

- `_locoplatform_graspview_task_embedding` env/train YAML·전용 train/test/VNC를 추가했다. 기존 human SA4 LocoPlatform에서 공동 장면만 reset 시 위치 기반 반대 HOLDING 앵커를 배정하고 episode 동안 유지한다. 매 step 해당 점의 pose 위치/선속도를 반영한 원본634차원 관측 두 벌을1268차원으로 저장하며, 각자의 human64를 추출해 기존 SA4에 넣는다. 단독 관측은 동일하고 encoder 중복 실행을 제거했다. Actor/Critic 구조·파라미터4,398,450개·64입력 head·Actor encoder/RMS 동결·Critic 전체 학습 및 보상/AMP/RSI/받침/Goal360/CPA는 유지했다.
- view별 Stage1 RMS와 checkpoint `joint_carry_grasp_view` 계약을 연결했다. 기존 SA4 checkpoint와 혼용을 막으며 Stage1부터 새 학습한다. CPU16개 통과: 토이 관측과 일치·단독 불변·packet/human/token 셔플·정규화·초기 Actor/Value 전이·최적화 전후 출력/gradient·동결·파라미터 동일·기존 SA 회귀.
- GPU5·2048환경 epoch3 저장본에서 Actor encoder69개/scene RMS 불변, Critic69개·action head6개 및 Actor/Critic8개 SA bias 갱신·finite 확인. 공동1638/단독410·AMP79.83/10.03/10.14%·full pickup100프레임·7개 RSI/Goal360 조건·반대 앵커 관측·부분 reset/배정 유지·8-step physics 검사를 통과했다. 전용 test로16환경·loco/carryWith·64-step 설정 재로드 평가 정상 종료. 셸/YAML·diff 검사 통과. 실제 VNC 화면과 장기 성능은 이번 변경에서 미검증이다.
- 결과 `output/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_graspview_task_embedding_check/`: `saved_check.json`, `reset_check.json`, `goal360_runtime_check.json`, `grasp_view_runtime.json`, `eval/metrics/`. 실행 명령은 `markdowns/config.md#stage2-sa4-graspview`.

### Stage1 공유물체 파지 중심 이동 · 단독 토이 수정

- 사용자 의도에 맞춰 `stage1_grasp_shift_{test,vnc}.sh`를 공유물체 토이로 수정했다. 기존 LocoPlatform 환경에 `joint_carry_at/ontop`만 허용하고 원본 Stage1 네트워크/30000 checkpoint를 사용한다. 전용 inference 진입점에서 정상 환경/config 검증 후 네트워크 설정만 Stage1로 선택, source checkpoint metadata와 strict state_dict를 검사한다. 학습 진입은 거부한다.
- 시작 위치 기준 반대 앵커를 배정해 유지하고, 같은 장면의 A/B 관측 두 벌을 배치 추론해 A-view의 A행동/B-view의 B행동만 공유 물리에 적용한다. `SHIFT=0` 실제 중심, `SHIFT=1` 기존 inset0.15의 반대 앵커(현재 ±0.25m)이며 이전 단독 토이의 비율 정의를 대체한다. 실제 물체·공유 목표·보상·RSI·AMP는 유지한다.
- CPU6개 통과(기존4개+공유 관측/반대 배정/동일 목표·물리 입력 불변/배치/단독 거부). GPU5·loco·SEED42에서 baseline/shift 초기 관측 해시 일치, 두 task 모두 payload2·target6, 관측 offset 각각0.25m·합0을 확인했다. 두-view zero-shift 행동과 원본 forward 최대 오차1.19e-7. shift 실제VNC/스크린샷 확인, NaN/실행 오류 없음. 첫 장면은 baseline101/shift87스텝에 terminate되어 공동집기 개선은 미검증이다. 결과 `output_etc/stage1_shared_grasp_shift_check_20261016/`. 셸 문법·diff 검사 통과.

### Stage1 파지 중심 이동 · 최초 단독 토이 (위 공유물체 토이로 대체)

- `stage1_grasp_shift.py`·전용 player/관측 변환·test/VNC를 추가했다. Stage1 CarryDistill30000·기존 YAML·독립 단독2명/loco에서 관측 payload 중심만 로컬 Y 반폭×SHIFT로 이동하고 회전점 선속도를 보정한다. 실제 물체/bbox/목표/보상/AMP/RSI와 Stage1 토큰 셔플은 유지하며 Stage2 checkpoint를 거부한다. 학습 경로는 추가하지 않았다.
- CPU4개 통과: zero identity/입력 불변, 회전/각속도 보정·payload 이외 불변, packet 순서/중복 payload, 물체 순열. GPU5·SEED42·1환경·원본/±0.5 각각299-step 정상 종료, NaN/조기 terminate 없음·초기 관측 SHA256 동일. −0.5는 실제 VNC 실행·스크린샷 확인; +0.5/원본은 headless. 세 경우 행동 출력은 달랐으며 원하는 파지 위치 확보·공동집기 효과는 아직 판정하지 않았다.
- 결과 `output_etc/stage1_grasp_shift_check_20261016/`; 셸 문법·diff 검사 통과. 명령/적용 범위는 `markdowns/config.md#stage1-grasp-shift`.

## 2026-10-15

### LocoPlatform human SA4 · task CA 교체

- `approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding` env/train YAML·전용 train/test/VNC를 추가했다. task grounding MLP 없이 Stage1 최종 human64끼리 SA+residual+FFN+LayerNorm4층·2head·FFN128을 수행한다. 매 층 갱신된 human을 Q/K/V로 사용하며 packet 기반 SELF/COUPLED/NONE bias 모두 학습한다. human 순서와 관계 행·열을 함께 셔플하고 출력 순서를 복원한다. Actor encoder 동결·64입력 head/Stage1 전이·Critic 전체 학습 및 LocoPlatform 환경/보상/AMP/RSI는 유지했다.
- `human_sa4_actor_critic` 계약으로 기존 CA checkpoint와 혼용을 막고6명 demo가 새 YAML을 자동 선택하도록 연결했다. CPU10개 통과(관계/6명 순열·padding·매 층 QKV 수식 대조·모든 bias gradient·전이 초기 행동/가치 보존·동결/학습·기존 CA 회귀). GPU5·2048환경 epoch3 저장본에서 Actor encoder69개/scene RMS 불변, Critic69개·head 각6개 및 Actor/Critic8개 block의 세 bias 갱신, task MLP 없음·64입력·133개 scalar finite를 확인했다.
- AMP 실측79.83/10.03/10.14%, 공동1638/단독410·paired pickup100/carryWith129/putDown135·Goal360·부분 reset·짧은 physics 검사를 통과했다. 전용test2명16환경64-step 설정·공통demo6명1환경32-step 설정 headless 재로드/추론 정상 종료(6명31-step timeout). 셸 문법·diff·새 경로 검사 통과. 결과 `output/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_check/`. 본학습 성능·실제 VNC 화면은 미검증이다.

### LocoPlatform 공동 query의 단독 task 차단 · 비교 뷰어

- 추론 전용 `locoplatform_ca_solo_mask_{test,vnc}.sh`를 추가했다. 기본 Platform22000·GPU5·6명·10초, `SOLO_MASK=0/1` baseline/공동 query→단독 task 차단을 선택한다. relation의 COUPLED 여부로 마스크를 만들어 순서 셔플을 유지하며, 다른 공동 팀 task·단독 query·학습 경로는 유지한다.
- CPU4개 통과(선행 토이 마스크와 동등·단독 query 불변·query/task 순열·가중치 불변·기존 NONE 회귀). GPU5 baseline headless/차단 실제VNC 각각32-step 설정에서31-step timeout 정상 종료, 첫 초기 장면 동일·마스크 metadata·6명 시작 화면 확인. 셸 문법·diff 검사 통과. 결과 `output_etc/locoplatform_solo_mask_viewer_check_20261015/`.

### 혼합 6명 뷰어 · Geo/NONE checkpoint 자동 선택

- 기존 test/VNC 공통 경로가 checkpoint의 policy metadata로 Geo/NONE-mask YAML까지 선택하도록 연결했다. PTH만 교체해 실행하며 공동 두 팀 carryWith·단독 AT/ON_TOP loco 장면과 기존 호환성 검사를 유지한다.
- GPU5에서 Platform/Geo/NONE 최신 기본 PTH 각각6명·1환경·32-step 설정 headless 추론을 실행해 올바른 config 선택·로드·finite·31-step timeout 종료를 확인했다. 셸 문법·diff 검사 통과. 결과는 `output_etc/demo_three_config_check_20261015/`; 이번 변경 후 실제 VNC 화면은 별도로 확인하지 않았다.

### LocoPlatform NONE mask · 새 Stage2 학습

- `approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_nonemask_task_embedding` env/train YAML과 전용train/test/VNC를 추가했다. Stage1 CarryDistill30000 전이·새 CA 초기화, Actor/Critic CA4에서 학습/추론 모두 NONE만 −∞ 처리한다. 원래 SA feature task256입력·CA-only64 head·Actor 동결·Critic 전체 학습·AMP/RSI/Goal360/CPA release/받침/보상을 유지한다. mask는 packet relation 기준으로 한 번 계산하며 유효한 관련 task가 없는 query는 residual 입력을 유지한다. `task_ca4_actor_critic_none_mask` 계약으로 기존 Platform/Geo checkpoint 혼용을 차단한다.
- CPU12개(설정 동일성·checkpoint 격리·순열·추론 토이와 동등·NONE gradient0·전이/학습 및 기존회귀) 통과. GPU5·2048환경 epoch3 저장, Actor encoder69개/scene RMS Stage1과 동일·Critic69개/action/value 각6개 갱신·양쪽 task MLP256입력·NONE bias0·399개 scalar finite 확인. AMP 실측79.83/10.03/10.14%, 공동1638/단독410·paired full pickup/Goal360·부분 reset 검사 통과. 저장본 전용test로16환경64-step 설정 추론 정상 종료. 재로드한 Actor/Critic8개 block의 NONE attention0·단독 SELF100%·가중치 불변·계측 행동 오차≤2.2e−7 확인. Python/셸 문법·diff·새 문서 링크 검사 통과. 결과는 `output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_nonemask_task_embedding_check/`. 본학습·운반 성능·실제 VNC 화면은 미검증이다.

### LocoPlatform NONE hard mask · 추론 전용

- `data/cfg/inference/locoplatform_ca_none_mask.yaml`과 전용 test/VNC를 추가했다. 기본GPU5·Platform12500·6명 혼합 데모이며 `SCENE=two_agents`/`NONE_MASK=0`으로 원래2명 평가/baseline을 선택한다. Actor CA4에서 NONE만 −∞로 차단하고 SELF/COUPLED 점수·Stage1 SA·Critic·residual·checkpoint·학습 경로는 유지한다. 일반/데모 player에 명시적 추론 config를 연결하고 적용 metadata를 저장한다.
- CPU10개 통과(정확한 NONE mask·관련 토큰 유지·NONE feature 교란/순열 불변·가중치 불변·2명 공동 동일·빈 query finite 포함). GPU5에서2명16환경/6명1환경 각32-step의 NONE attention0·단독 SELF100%·model/RMS 불변을 확인했다. 2명 공동 action 차이≤2.6e−7, 계측 오차≤3.6e−7. 새6명 wrapper의mask/baseline 각각2회×63-step reset/finite 정상 종료·두 초기 장면 일치,2명16환경 전용 wrapper도64-step 설정 평가 정상 종료. Python/셸 문법·diff·추가 문서 링크 검사 통과. 결과는 `output_etc/locoplatform_none_mask_check_20261015/`이며 운반 개선·장기 평가·실제 VNC 화면은 미검증이다.

## 2026-10-14

### LocoPlatform Geometry task CA · 새 Stage2 학습

- LocoPlatform 기반 `_locoplatform_geometry_task_embedding` env/train·train/test/VNC를 추가했다. CA task 입력을 행동 embedding64+담당 사람 환경 pose7+사람 기준 payload7/target7의85차원으로 바꾼다. 기존 GTA pose를 재사용하므로 사람은 root 위치/yaw, 물체는 전체 quaternion이며 AT G/ON_TOP 받침을 packet ID로 조회한다. Actor/Critic geometry MLP85→128→64, 기존 human query·CA4·CA-only head·identity 초기화·Stage1 전이/Actor 동결을 유지한다.
- AMP80/10/10·공동80/단독20·전체 paired RSI·Goal360·CPA release·loco 받침·보상·셔플은 원본 LocoPlatform과 동일하다. `task_ca4_actor_critic_geometry` checkpoint 계약으로 기존 feature-task 실험과 직접 혼용을 차단한다. quaternion 부호·회전행렬 oracle·padding·task/entity 순열·gradient·초기 행동/가치 보존 및 기존 회귀 CPU18개 통과; 최종 wrapper 경로 검사 포함 geometry4개 재확인, Python/셸 문법·diff 검사 통과.
- GPU5·2048환경 Stage1 CarryDistill30000 전이→epoch3 저장, Actor encoder69개/scene RMS 동결·Critic/head 갱신·399개 scalar finite를 확인했다. 같은 설정의 AMP 실측79.83/10.03/10.14%, 공동1638/단독410·부분 reset·paired pickup100/carryWith129/putDown135 및 Goal360/짧은 physics를 확인했다. 저장본16환경64-step 설정 carryWith headless 재로드/추론 정상 종료. 기록은 `output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_geometry_task_embedding_check/`. 본학습·장기 성능·실제 VNC 화면은 미검증이다.

### Platform 전원 loco Stage1 SA·Stage2 CA 비교

- 같은16장면·Platform12500의 frozen encoder69개 텐서와 scene RMS가 Stage1 CarryDistill30000과 정확히 일치했다. SA human query의 자기 human/payload/target 합계95.6~98.3%, 공동 query의 단독 관련6entity 합계0.4~0.9%였다. 같은 장면2명/6명 encoder 출력 cosine은H0/H1=0.99984/0.99997, payload/goal≈0.977이었다.
- SA는 bias 전 QK만으로 관계를 선별하지 않지만 semantic bias 추가 후 자기 과제에 집중했다. human row의 head별 평균 bias 범위7.80~18.97로 CA 관계 bias0.30~0.47과 차이가 있었다. 초기 장면의 CA 쏠림을 앞단 human SA에서 같은 현상이 전파됐다고 설명할 근거는 없으며, 전체 실패 원인을 확정한 실험은 아니다.
- `output_etc/platform_attention_check_20261014/sa_check/`에4층 QK/softmax/실제 SA 그림·NPZ·분석/검증을 저장했다. GPU5·이전 all-loco map 재현 오차0·model/RMS 불변·행동 오차≤2.4e−7·softmax 재구성 오차≤1.9e−7 확인. 기존 모델/학습은 변경하지 않았다.

### Platform 6명 전원 loco CA QK 확인

- Platform12500·GPU5·16초기 장면에서 기존 공동 carryWith/단독 loco 데모의 공동 두 팀만 원래 공동 loco 초기화(상자 양쪽1~2m)로 교체했다. 공동 payload는 바닥에 두고 받침 없이 측정하며 단독 상태·목표·다른 물체는 유지했다. 전체4-layer/2-head 평균 공동4명의 단독 E/F task attention은63.9→72.2%, NONE 합계는80.7→85.2%였다. 시작 skill 차이만으로 쏠림을 설명할 수 없으며 배치/거리까지 통제한 실험은 아니다.
- 원본 장면 해시 일치 후 초기화 교체, model/RMS 불변, 셔플/계측 행동 오차≤2.4e−7, attention finite/행합 오차≤1.2e−7를 확인했다. `output_etc/platform_attention_check_20261014/all_loco/`의 실행 코드·maps·qk에 bias 전 QK/softmax와 원래 attention을 저장했다. 물리 rollout·학습 변경 없이 초기 forward만 비교했으며 층별 bias 제거는 같은 Q/K의 재채점이다.

### 혼합 장면4명 옵션 · 공동 한 팀 제외

- 기존 test/VNC에 `DEMO_AGENTS=4`를 추가했다. 공동 H0/H1 carryWith·단독 H2 AT/H3 ON_TOP loco, 전역4명 추론·8물체를 유지하며 제거한 팀의O1은 주변 상자로 둔다. Graph/task ID·reset·goal·marker·로드 로그를4명에 맞췄다. 기본6명·softmax·checkpoint 계약은 유지하며4명 기본 output은 `output/joint_carry_mixed_scene_demo_4agents`다.
- CPU8개·Python/셸 문법·diff 검사 통과. GPU5 사용자 LocoPlatform checkpoint로 headless2회×63-step finite/reset과4명·8물체·4목표·carryWith1쌍을 확인했다. 결과는 `output_etc/joint_carry_mixed_scene_demo_4agents_check_20261014/headless/`. 실제 VNC6081의4명 시작 화면을 확인하고10회 재생을 시작했다. 화면/로그는4명 output에 있으며 운반 완수율은 검증하지 않았다.

### 단독 AT·ON_TOP 2명 attention map

- GPU4·CPA Release1 epoch19000·동일 초기16장면의H4/H5(loco)를2명·O2/O3/O5/O4·G4/G5 입력으로 재매핑해 softmax/CA entmax를 비교했다. `MAP_VIEW=solo` 수집·`solo_map_plot.py` 시각화·`solo_maps/` 결과이며 별도2명 물리 rollout은 아니다. 기존 전체6명 map을 정확히 재현하고 model/RMS 불변·계측 행동 오차≤2.4e−7을 확인했다.
- 전체 CA layer/head 평균 자기/다른 단독 task는 softmax48.1/51.9%, entmax47.4/52.6%였다. 6명 H0의 마지막head2에서파트너/T4/T5 QK 점수0.932/2.623/2.091에 관계bias+0.176/−0.103/−0.103이 더해져 단독task 우세를 유지한다. 관계분류는정상이며 점수가 커진 학습 원인·실패 인과는 검증하지 않았다. 원시확률·점수·관계·분해는 `solo_maps/{maps.npz,maps_summary.json,analysis.json}`이다.

### 혼합 장면 데모 LocoPlatform checkpoint 설정 자동 선택

- 데모 test wrapper의 일반 CPA Release1 YAML 고정으로 LocoPlatform checkpoint가 거부되던 문제를 수정했다. `joint_carry_loco_platform` metadata 유무로 기존 두 실험의 env/train YAML을 선택하며 로컬/headless/VNC·CA Entmax 경로에 공통 적용한다. 기존 checkpoint 계약 검사·학습 설정·6명 장면 초기화는 유지한다.
- 관련 CPU11개·셸 문법·diff 검사를 통과했다. GPU5에서 사용자 LocoPlatform checkpoint의 headless/VNC와 기본 CPA Release1 epoch19000 CA Entmax를 각각2회·회당63-step(episode length64) 실행해 로드/reset/finite를 확인했다. VNC 화면4장과 결과/로그는 `output_etc/joint_carry_demo_checkpoint_config_check_20261014/`에 있다. 장기 운반 완수율은 검사하지 않았다.

### 동일 초기 상태2명/6명 CA·SA attention map

- H0/H1 query만 표시한 기존 그림에 전체 query 그림(`ca_last_layer_full_scene0.png`, `sa_last_layer_full_scene0.png`)을 추가했다. 원시6명 CA는6×6, SA는 사람6·물체8·목표6의20×20이며 표시만 잘랐던 것을 확인했다.
- GPU4·epoch19000·기존 토이와 해시가 같은16개 초기 장면에서 H0/H1 공동팀의2명 관측,6명 softmax,6명 CA entmax의 Actor4-layer×2-head map을 저장했다. 2명은 같은 장면의H0/H1·O0/O5/O7/O6·G0/G1을 재매핑한 관측이며 별도2명 rollout과 혼동하지 않는다. `output_etc/entmax_inference_toy_20261014/{map_collect,map_plot}.py`, `maps/{maps.npz,maps_summary.json,*scene0.png,ca_categories_16scenes.png}`.
- 해당 팀·16초기 장면·전체 CA layer/head 평균은 자기/파트너/다른 task가2명31.3/68.7/0%,6명 softmax3.4/12.7/83.9%,6명 CA entmax0.7/13.2/86.0%였다. Scene0 마지막 CA head2의H0→파트너는95.1→12.6→1.4%, entmax에서 단독 AT/ON_TOP에67.6/31.0%였다. Sparse한 대상이 자기 팀으로 제한되는 것은 아니며 기존 전체 사람/재생 시간 평균과 집계 범위를 구분한다.
- Map 인덱스 순서 고정/계측의 행동 오차≤2.4e−7·확률 row 합 오차≤4.8e−7·6명 두 조건 SA map 동일(오차0)·model/scene RMS 불변·초기 상태 해시 일치를 확인했다. 물리 step 없이 같은 관측의 실제 각 forward를 비교했으며 이후 CA layer는 이전 layer 변경의 영향도 포함한다. Python 문법·추가 링크·diff 검사를 통과했다.

### 6명 혼합 장면 CA Entmax VNC

- `joint_carry_mixed_scene_demo_ca_entmax_{test,vnc}.sh`를 추가했다. 기존6명·8물체·1환경에서 `DEMO_CA_ENTMAX=1`로 Actor CA4만 entmax1.5를 적용하며 SA/Critic·checkpoint·기존 wrapper 기본값은 유지한다. 기본 CPA Release1 epoch19000·GPU4·별도 `output/joint_carry_mixed_scene_demo_ca_entmax`, 촬영 기록에 attention 종류를 남긴다. 추론 구현은 `learning/multi_agent/ca_entmax_inference.py`로 토이 폴더와 분리했다.
- CPU7개(독립 수치해·마스크·순열·균등 attention 원본 출력·가중치 불변·기존6명 graph 계약)·토이 CA 출력과 최대오차0·GPU4 실제 VNC2회64-step finite/reset을 확인했다. Python/셸 문법·명령 경로·diff 검사 통과. 결과/화면은 `output_etc/joint_carry_mixed_scene_demo_ca_entmax_check_20261014/`이며 새 학습이나 운반 완수 검증은 수행하지 않았다.

### 추론 전용 Entmax1.5 attention 토이 · GPU4

- CPA Release1 epoch19000을 동결하고2명 공동 AT carryWith·6명 혼합 데모 각각16장면20초를 동일 초기 상태로 비교했다. Actor SA만/CA만/둘 다 entmax1.5로 바꾸는 격리된 실행이다. 학습·기존 viewer·config/checkpoint는 변경하지 않았다. 코드/원시 궤적/분석/그래프는 `output_etc/entmax_inference_toy_20261014/`다.
- 6명 넘어짐은 softmax11/16·SA5/16·CA2/16·둘 다7/16, 공동 상자의 마지막 유효 XY 오차는1.345/1.495/0.640/1.589m였다. CA의9개 개선/0개 악화는 대응 McNemar 보정 p=0.0117(6명3개 조건 비교). 같은 경과 시간의 CA 오차 차이−0.344m bootstrap95% CI[−0.780,+0.049]는0을 포함한다. 공동 배치 조건은 전부0/32이며 손 해제/완수 개선을 확인한 것은 아니다.
- CA만 바꾸면 CA 정확한0 비율53%지만 NONE task mass는78.8→79.7%, 같은 상태2명/6명 공동 행동 차이는0.0346→0.0382로 줄지 않았다. 따라서 안정성 개선을 다른 팀 정보 차단/인원 불변성 회복으로 단정하지 않는다. 2명에서는 CA 넘어짐0→1/16·오차0.109→0.435m로 악화했다.
- Entmax 독립 이분법 최대오차8.9e−16·계측용 softmax 전체 행동 오차≤3.0e−7·초기 상태 해시 일치·model/scene RMS/AMP RMS 불변·finite를 확인했다. 새 프로세스의 softmax 반복은 물리 root/box 궤적 오차0 및 넘어짐/배치 결과 동일;6명2장면의 종료 검출은1-step 차이가 있었다. Python/셸 문법·diff 검사 통과.

### 6명·8물체 공동/단독 혼합 VNC 촬영 데모

- CPA Release1 FullPickup 기존 env/train YAML·checkpoint를 재사용하는 `joint_carry_demo.py`와 `joint_carry_mixed_scene_demo_{test,vnc}.sh`를 추가했다. 한 환경의6명을 전역 관측으로 추론하며 공동 AT 두 팀은 독립 paired carryWith, 단독 AT/ON_TOP은 loco로 시작한다. 네 과제는2.8m 간격·목표+X2.5m, 주변3상자를 포함해8물체다. 초기 시점 이후 자유 카메라·반복 reset·넘어짐 재시작을 제공하고 보상/성공률 계산은 생략한다.
- 원본2명 학습·checkpoint 계약을 유지하고 demo에서 source metadata·AMP/RSI/Goal360와 model/RMS를 그대로 로드한다. CPU12개·GPU5 epoch19000의2회64-step headless reset/finite 추론 및 실제 VNC3회 실행을 확인했다. VNC는191/599/54-step에서 넘어짐/시간 만료/넘어짐으로 종료했으며6명 확장 정책의 완수율을 검증한 것은 아니다. 결과/화면은 `output_etc/joint_carry_mixed_scene_demo_{check,vnc_check}_20261014/`에 있다.
- 자유 카메라·평가 계산 생략을 적용한 최종32-step VNC 실행과 Python/셸 문법·diff 검사도 통과했다. 최종 화면/기록은 위 VNC 결과 폴더의 `final/`이다.

### Platform 손·팔 Masked AMP 오프라인 토이

- 집기 이전 구간 추가: 동일 Platform8000 loco 궤적에 손/몸통/골반 높이를 기록하고 기존 토이 판별기를 재사용했다. 상자 근처(1.2m 이내), 첫 HOLDING 전, 손8cm·몸통/골반4cm 이상 하강 및 손–앵커8cm 이상 접근한 held-out 시도는 바닥3건/받침5건이었다. 기존 판별기의 최종 AMP 기여 감소는−0.267/−0.134. 동일 데이터 재학습 일반 판별기는+1.394/+0.527, 마스크는+1.161/+0.428로, 재학습에 따른 점수 변화와 마스킹 효과를 구분해야 한다. 작은 표본이며 마스킹 우위·정책 개선 근거로 해석하지 않는다. `output_etc/masked_amp_toy_20261014/reach_phase/{analysis.json,pre_grasp_amp.png}`.
- TeamHOI `trans_agent.py`의 양팔 자세/속도·양손 위치 마스크를 현재132차원 AMP에 매핑했다(프레임별25:49,91:99,117:123;10-frame). Platform8000 정책을 동결해 GPU5·loco AT/ON_TOP 각128장면20초를 수집하고 기존80/10/10 reference32768개로 일반/마스크 판별기만 같은 조건에서600회씩·2개 시드로 재학습했다. 학습 코드·config·정책은 변경하지 않았다.
- 장면 단위 분리 평가25034개, 첫 HOLDING14개 사람 이벤트/12장면에서 최종 보상 기준 AMP 기여는 일반0.431→0.333(−0.098), 마스크0.557→0.394(−0.164)였다. 마스크가 잡은 상태 점수는 높였지만 전후 하락 감소는 확인되지 않았다. 바닥의 실제 집기 표본은 없어 바닥 집기 효과로 일반화하지 않는다.
- 모델/actor RMS/AMP RMS 불변, half 저장 후 원래 점수 최대오차0.000208, 마스크 차원 교란 출력변화0을 확인했다. `output_etc/masked_amp_toy_20261014/`의 `analysis.json`·`holding_events.json`·`amp_comparison.png`·재현 코드/명령을 참조한다. 기존 판별기 입력만0으로 한 결과는 OOD 민감도이며 학습된 Masked AMP 결과로 해석하지 않는다. PPO 재학습·TeamHOI 거리 blending·별도 near reference는 미검증이다.

## 2026-10-13

### 접촉점·마찰·힘 센서 API 실측 확인

- `output_etc/contact_api_probe_20261013/`에서 GPU5 단독 상자/실제 humanoid 토이를 실행했다. GPU pipeline의 접촉점 목록 API 2개는 실행 후 사용 불가 오류를 반환했다. CPU 환경별 접촉점은 읽혔지만 미끄러짐 중 마찰 필드는0이었다.
- 별도 GPU constraint-only force sensor는 상자 수평 마찰을 읽었다(가속도 기준−63.21N, 센서−62.20N, 기존 contact tensor약0N). 따라서 기존 텐서의 누락과 센서 기능을 구분한다. 같은 상자에 위치가 다른 두 센서는 동일 합력을 읽어 사람별 분리가 되지 않는다.
- 물체/바닥 없이 실제 humanoid 손에 외력10N을 가한 검사에서 접촉 텐서는0인데 손 constraint sensor는 수평약−4.61N을 읽어 관절 반작용 혼입을 확인했다. 현재 API만으로 사람–payload별 전체 힘을 바로 읽는 경로는 확보하지 못했다. 결과/한계/재현 명령은 `summary.json`; 학습 코드·보상은 변경하지 않았고 학습 속도 벤치마크는 수행하지 않았다.

### CPA Release1 carryWith 접촉력·운반 궤적 진단

- GPU5·epoch9500 고정 정책으로 공동 AT/ON_TOP 각64장면 최대600-step(20초)을 실행했다. 기본 Goal360·carryWith RSI/초기 속도 유지, 첫 종료 이후는 제외. 각 사람의 전 부위 접촉력·root/body·상자·HOLDING/배치 조건을 기록하고 모델/RMS 불변·finite를 확인했다. AT24/64·ON_TOP48/64가 기존 배치 조건을 한 번 이상 만족했다.
- 손만의 힘은 상자 접촉력과 맞지 않아 상체 합/장애물·다른 사람 거리/공중 지지/반작용 합 검사로 귀속 구간을 제한했다. 두 physics step을 평균해도 상자 가속도 기반 총힘과 달랐다. 별도 정지/미끄러짐 토이에서 수평 감속 힘−63.43N 대비 텐서약0N으로 마찰 누락을 확인했다. 따라서 결과는 **법선 접촉 성분 추정**이며 추진·제동·개별 하중 분담으로 단정하지 않는다.
- `output_etc/cparel_carrywith_forces_20261013/`에 raw NPZ·`rollout.json`·`analysis.json`·`case_phases.json`·`calibration.json` 및 대표5장면의 힘/움직임 그래프·정책 궤적 재생 영상을 저장했다. 뒤쪽 목표에서 반대로 가다 양쪽 HOLDING을 유지한 채 정지하는 실패, 이동 중 전후진 역할 교환, 목표 배치 후 손을 유지하는 성공을 관찰했다. 학습 config/프로세스는 변경하지 않았다. 대표5개 MP4 각599프레임 전체 디코딩·Python/셸 문법·diff 검사를 통과했다.

### 단독 원본·현재 공동 RSI 동기 비교 영상

- `output_etc/solo_joint_rsi_comparison_20261013/`에 pickup/carryWith/putDown의 단독 B19/B20/B21 원본32-DOF 자세와 현재 공동 보정본을 같은 프레임으로 재생하는 비교 영상을 만들었다. 사선/상단 시점·기본/0.5배속·전체 모아보기를 제공한다. 학습 config·데이터는 변경하지 않았다.
- 공동 snapshot root/DOF/상자 위치 일치, pickup100/carryWith129/putDown135프레임과30fps 동기화를 확인했다. 기본 영상 총364프레임, ffprobe 규격·모아보기 전체 디코딩 및 pickup 중간 화면 검사를 통과했다. 순차 기구학 재생이며 단독 크기별 RSI 선별·정책/물리 rollout 영상은 아니다. 출처/해시는 `report.json`, 재생성은 `render_comparison.py`다.

### CPA Release1 LocoPlatform · 공동 loco payload 받침 연결

- CPA Release1 기반 `approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding` env/train·train/test/VNC를 분리했다. 공동 AT/ON_TOP loco의50%만 payload 아래40×40×2cm 받침을 둔다. 높이는 `min(max(U(0,1),0.02),1.2−payload높이)`이며 초기5cm 낙하 간격을 유지한다. 기존 고질량·중력 비활성 actor를 물체/marker 뒤에 추가하고 reset은 한 번에 commit한다.
- 단독·paired pickup/carryWith/putDown·목표·다른 물체·RSI 단계 비율·AMP80/10/10·CA4 CA-only Actor/Critic·CPA Release1 보상·셔플을 유지한다. 받침 token은 추가하지 않는다. `joint_carry_loco_platform` metadata로 다른 설정과 직접 resume/eval 혼용을 차단한다.
- CPU25개 통과·기존 ON_TOP putDown1개 skip, Python/셸 문법·diff 검사 통과. GPU5·2048환경 reset에서 공동1638/단독410, loco 받침50.73%·2~80cm·기타 RSI/목표/AMP 불변·부분 reset을 확인했다. AT832/ON_TOP816장면2초 지지와 받침 제거 후 낙하 검사를 통과했다. 최대 payload XY drift0.062mm, 받침 위치 drift0.011mm였다.
- Stage1 epoch30000 전이→epoch2 저장→epoch3 재개, actor69개 tensor/RMS 동결·critic/양쪽 CA4/head 갱신·133종 scalar finite를 확인했다. 저장본16환경32-step loco VNC viewer 실행·pickup headless 평가를 통과했다. 결과는 새 `*_check/{platform_runtime_check,reset_check,goal360_runtime_check,saved_check}.json`, `*_resume_check/`, `*_eval_check/`다. 본학습·공동집기 개선·바닥 집기로 전이는 미검증이다.

### 공동 loco 시작 payload 받침 물리 토이

- 추가 높이 검사: `output_etc/payload_platform_height_sweep_20261013/`에서 GPU5·평가128환경·AT/ON_TOP의0~80cm를 검사했다. 최대80cm84장면과2cm 이상 랜덤140장면은 지지됐으나, 2cm 미만16장면은 두께2cm 판과 바닥이 겹쳐 받침이 밀려 올라갔다(최대 높이 오차15.25cm). 높은 분기만 최소2cm로 제한한 별도 토이는256/256 지지·목표 높이 검사 통과, 부분 reset·받침 제거 후 낙하·10초 정책 추론 finite·model/RMS 불변을 확인했다. 학습 설정에는 미연결이며 `report.json`은 원본 높이, `report_min2cm.json`은 하한 보정 결과다.
- 원본 TokenHSI의40×40×2cm 받침을 참고해, 실행 중에만 actor를 추가하는 `output_etc/payload_platform_toy_20261013/` 검사를 만들었다. 기존 학습 config·reward·AMP·RSI 데이터는 변경하지 않았다. GPU PhysX에서는 기존 repo의 고질량·중력 비활성 받침을 재사용하고 사람/상자/받침 actor를 한 번에 reset한다. 첫 검사에서 받침을 별도 reset 호출로 쓰면 이후 호출에 반영되지 않는 문제가 확인돼 검사 코드를 수정했다.
- GPU5·평가64환경에서 공동 AT/ON_TOP 각각 바닥·20/40/60cm 높이를 검사했다. 높은 받침96장면 모두2초 정지 지지 검사 통과, 받침을 치우면 상자가 내려가는 실제 충돌 위치 갱신 확인. AT4/ON_TOP3 초기 단계에서 payload 높이 외 사람·목표·다른 상자·AMP 이력 불변을 확인했다.
- CPA Release1 epoch7500의 각 과제64장면10초 추론은 finite·model/RMS 불변. 공동 HOLDING은0/128, 개인 HOLDING은ON_TOP60cm1/16에서 관찰됐다. 물리 구현 가능성 검사이며 받침을 사용한 재학습·바닥 공동집기로의 전이는 미검증이다. 상세 결과와 재현 명령은 `report.json`, `run.sh`다.

### Output 검증·분석 산출물 보관 정리

- 완료된 검증·분석·영상15개 폴더(268파일, 약535MiB)를 `output_etc/`로 이동했다. 모든 본학습 결과11개와 `output/rsi_cache`는 유지했다. 실행 중인 FullPickup·CPA Release1·PickupFocus 경로와 공동운반 학습 데이터는 그대로다.
- 이동 대상의 쓰기 열린 파일·목적지 충돌이 없음을 확인하고, 이동 전후 디렉터리 inode·파일 수·총 byte가 일치하는지 검사했다. 삭제·덮어쓰기는 없으며 config/structure/README의 해당 결과 링크를 수정했다. 이동 목록은 `output_etc/archive_20261013_cleanup.json`이다.

### CPA Release1 PickupFocus · 공동 집기/들기 RSI 집중

- `approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding` env/train·train/test/VNC를 분리했다. 공동 pickup 선택 후25~39번35%·55~80번35%·전체0~99번30%로 샘플링한다. 유효100프레임을 유지하며 단계 확률·단독 RSI·AMP·보상·CA4/Actor/Critic·Goal360·셔플은 기준 CPA Release1과 같다. 가중치는 로드 시 계산한다. 별도 checkpoint sampling 계약으로 기존 실험과 resume/eval 혼용을 막는다.
- CPU25개 통과·기존 공동 ON_TOP putDown1개 skip. 혼합 확률/모든 프레임 지원·source frame 순열·누락 데이터 거부·기존 config/전이/SA·CA 순열 회귀와 Python/셸 문법·diff를 확인했다.
- GPU5·2048환경 실제 reset에서 공동1638/단독410, AT pickup 집기/들기/기타39.56/42.61/17.83%·ON_TOP39.26/44.63/16.12%, AMP80.09/10.24/9.67%를 확인했다. Paired 자세/속도·Goal360/clearance·AMP 이력·부분 reset1024/32/32·finite 물리 step을 통과했다.
- Stage1 epoch30000 전이→epoch2 저장→epoch3 재개와 저장본16환경32-step pickup headless 평가를 통과했다. Actor69개 tensor/RMS 불변·Critic encoder/양쪽 CA4/head 업데이트·학습 scalar133종 finite. 결과는 새 `*_check/{saved_check,reset_check,goal360_runtime_check}.json`, `*_resume_check/`, `*_eval_check/`. 본학습·실제 VNC 화면·공동집기 개선은 미검증이다.

## 2026-10-12

### FullPickup 공동 상대 CPA 근접 해제(+1m)

- `approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding` env/train·train/test/VNC를 분리했다. 동일 carry task/payload/target의 두 사람이 모두 payload XY 반대각선+1m(현재1.477m) 안이면 CPA만0으로 한다. 매 step 재판정하고 단독·다른 task/goal·영역 밖은 기존 CPA를 유지한다. 물리 충돌·HOLDING progress와 FullPickup 구조/전이/RSI/AMP/셔플은 유지했다. Reward metadata로 기존 FullPickup checkpoint와 직접 resume/eval 혼용을 차단한다.
- CPU43개·Python/셸 문법 검사를 통과했다. 경계 포함·두 사람 모두 조건·해제 후 재적용·payload 크기/재배정·좌표/사람/edge 순열·다른 goal/task/단독 제외·기존 보상/구조 회귀를 확인했다.
- GPU5·2048환경에서 Stage1 epoch30000 전이→epoch2 저장→epoch3 재개,16환경32-step pickup 시작 headless 평가를 통과했다. Actor69개 tensor/RMS 불변·Critic encoder/양쪽 CA4/head 갱신·모델/scalar133종 finite를 확인했다. 별도64환경64-step 검사에서 공동 AT839/ON_TOP843 scene-step의 비영 CPA 제거·나머지 CPA409 scene-step 유지·단독832 scene-step CPA 불변과 기존 task 보상 성분 일치를 확인했다.
- 결과는 새 `*_check/{cpa_release_runtime_check,saved_check}.json`, `*_resume_check/`, `*_eval_check/`다. 본학습·실제 VNC 화면·공동집기 개선은 아직 검증하지 않았다.

### Goal360 loco 공동집기 CPA 보상 진단

- GPU5·Goal360 epoch13000 고정 정책으로 공동 AT/ON_TOP 각128환경·loco 시작600-step(20초)을 실행했다. 실제 CPA를 코드로 재계산해 지급된 항과 일치하는지 확인하고, 같은 상태/행동에서 CPA 항만 제거한 task/AMP 혼합 보상을 계산했다. 학습 설정·checkpoint·기존 실행은 변경하지 않았다.
- 미집기 접근 구간의 CPA 평균 크기는 AT0.0146/ON_TOP0.0179, 최대0.372/0.403이며 개인 HOLDING state+progress를 초과한 agent-step은5.01/5.67%였다. 두 과제 모두 공동 HOLDING0/128, 개인 HOLDING 달성 장면0/128이었다. 근처 정지 구간은 progress약0.197~0.199·state약0.0005·CPA약0.0015~0.0017로, 접근의 일부 큰 벌점과 손을 뻗지 않고 머무는 현상이 함께 관찰됐다.
- `toy_joint_carry_cpa.py`의 실제 reward 성분 합/CPA/finite·모델/RMS 불변 검사를 통과했다. 결과는 `output/joint_carry_cpa_toy_20261012/{report.json,joint_carry_at.npz,joint_carry_ontop.npz}`. 첫 종료 이후 표본은 제외했다. 같은 궤적의 보상 비교이며 CPA 제거 재학습·인과적 개선 검증은 수행하지 않았다.

### Goal360 FullPickup · Actor/Critic CA-only 연결

- `approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_task_embedding` env/train·train/test/VNC를 추가했다. Actor/Critic 각각 독립 CA4 출력64차원만 action/value head에 전달한다. Stage1 두 head를 그대로 전이하고 CA residual branch를0으로 초기화해 초기 네트워크 출력을 보존한다. Actor encoder/RMS만 동결하며 Critic encoder·task embedding·CA·head는 전체 학습한다. 기존 config/checkpoint 경로는 유지한다.
- 공동 pickup만 검증된100프레임으로 교체했다. `joint_carry/carry_joint_full_pickup/`의 pickup100/carryWith129/putDown135개를 별도 SHA256 계약으로 분리했다. CarryWith/putDown·단독 RSI·단계 확률·AMP80/10/10·Goal360·보상/CPA·SA/CA 셔플은 유지하며 Move/Align/TeamHold는 없다.
- CPU48개 통과·기존 공동 ON_TOP putDown1개 skip. 초기 actor/value 일치, 학습 후 SA/CA·task/entity 순열, no-task residual, Critic embedding gradient·전이/정책/데이터 계약과 기존 경로를 검사했다.
- GPU6·2048환경에서 Stage1 epoch30000 전이→epoch3 저장→epoch4 재개를 통과했다. Actor69개 tensor/RMS 불변·Critic encoder69개 및 두 head·양쪽 CA4 전층 갱신·모델/scalar133종 finite를 확인했다. 전체4,480,754개·학습3,954,258개 파라미터다.
- 실제 공동1638/단독410, AMP30000개80.09/10.24/9.67%·RSI 분리/초기 이력·부분 reset1024/32/32 및 Goal360 AT4/ON_TOP3 단계 검사를 통과했다. AT/ON_TOP 모두 pickup100프레임 전체가 선택되고 pose/velocity·360° 목표/clearance가 보존됐다. 저장본16환경32-step pickup 시작 혼합 평가도 정상 종료했다. 결과는 새 `*_check/{saved_check,reset_check,goal360_runtime_check}.json`, `*_resume_check/`, `*_eval_check/`다. 본학습·실제 VNC 화면·장기 공동집기/운반 개선은 미검증이다.

### 공동 pickup RSI 전체 100프레임 보정 후보·검토 영상

- 제외됐던 pickup 중간 자세의 초기 충격을 조사했다. 동일 CPU 검사에서 원본 단독35~69·공동36~68프레임이 실패했고, 공동에서 한 사람만 남겨도 동일했다. 진단용 asset의 무릎 X/Z 제한만 넓히면 선택한 실패 자세의 충격이 크게 감소했다. 실제 asset·학습 설정은 변경하지 않았다.
- `retarget_joint_pickup.py`로 현재 관절 범위 안에서 다리 IK·접지·팔 재보정을 수행했다. 원래100프레임·30fps·상자52×80×40cm·상자 궤적을 유지했다. 발 위치 변화 최대2.45mm·손14.58mm, 관절 범위 초과0이다.
- `output/joint_pickup_full_20261012/` 후보는 CPU 초기 geometry/0.1s shock100/100, GPU6·2048환경·4개 yaw2048/2048 및 부분 reset1024/1024 통과. 실제 RSI 로더는 pickup100/carryWith129/putDown135개를 읽고 finite를 확인했다. 기존 학습 데이터와 config에는 미연결이며 전체 Stage2 reset/policy rollout은 미검증이다.
- `preview_joint_pickup.py`로 전체 프레임의 보정 전후 기구학 영상과 각100프레임에서 독립적으로0.1s 물리를 실행한 비교 영상을 생성했다. 연속 native-PD 추종은 여전히 균형을 잃으므로 연속 동작 성공으로 해석하지 않는다. 결과는 `validation.json`, `pose_changes.json`, `comparison{,_half_speed}.mp4`, `physics_all_frames.mp4`다.

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
