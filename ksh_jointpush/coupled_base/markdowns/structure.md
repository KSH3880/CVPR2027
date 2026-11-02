# 레포 구조

유지하는 컨피그는 `stage1`, `stage2_before_nonemask_rsi433_near50`, `stage2_before`, `stage2_before_rsi433`, `stage1_push`, `stage2_before_push`, `stage2_coupled_push`, `stage2_before_push_near50`, `stage2_coupled_push_region_goals`다. 실행 명령은 [config](config.md), 설치·추가 옵션은 [USAGE](USAGE.md)에 있다.

## 실행과 설정

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/run.py`, `utils/parse_task.py` | MA 학습/평가·Stage1/BEFORE 모델 등록과 환경 생성 |
| `tokenhsi/utils/config.py`, `experiments.py` | CLI, 아홉 실험 계약, 학습/추론/resume env/sim 로딩 |
| `tokenhsi/data/cfg/{multi_agent,train/rlg}/` | 환경·학습 YAML 각 아홉 개 |
| `tokenhsi/scripts/multi_agent/*_{train,test,vnc}.sh` | 실험별 학습·viewer·VNC 진입점 27개 |
| `tokenhsi/scripts/multi_agent/*_eval.sh`, `evaluate.sh` | 일반 정식 평가9개, Stage1/Stage1 PUSH 전용 1명 단독 평가2개·공통 runtime |
| `tokenhsi/evaluate.py`, `rescore.py` | 독립 seed 실행·통계, 저장 trace CPU 재채점 |
| `tokenhsi/utils/evaluation_protocol.py`, `learning/multi_agent/formal_eval.py` | 학습 인원/단독 구분·점/영역/BEFORE 지표·개인 낙상 제어 |
| `tokenhsi/scripts/multi_agent/launch.sh`, `runtime_env.sh` | 공통 인자·출력 경로·conda·GPU |
| `ksh_jointpush/scripts/coupled_base_run.py` (CVPR2027 루트 기준) | 로컬 PyTorch2.0/CUDA11.8 호환 entry. 구역별goal train wrapper에서 선택 |
| `tokenhsi/scripts/multi_agent/{gui_gpu_env,run-gui,mps_auto_env}.sh` | viewer GPU, VNC, 기존 MPS 연결 |
| `tokenhsi/utils/checkpoint_compat.py` | 기존 checkpoint의 이름·RSI 경로를 현재 표기로 변환 |
| `checkpoints/` | 외부 checkpoint 심링크 |
| `output/`, `output_etc/check/` | 본학습·공용 캐시, 확인 학습·평가·viewer |

## 환경

아래 경로는 `tokenhsi/env/tasks/multi_agent/` 기준이다.

| 파일 | 역할 |
| --- | --- |
| `humanoid_ma.py`, `humanoid_ma_carry.py` | 다중 사람·제어·상태와 공통 환경·물체 생성 |
| `scene_observation.py`, `scene_reset.py`, `reference_motion.py` | 관측, reset/배치, 모션·RSI·AMP |
| `edge_ontop_task.py`, `edge_context_task.py`, `relation_diagnostics.py` | 과제 그래프·물리 재시도·보상·진단, CONTACT 직접 집계와 AT/CONTACT TensorBoard 별칭 |
| `humanoid_ma_push.py`, `jointpush_reward.py` | 1/2명 PUSH 자산·배치·접촉 RSI·CONTACT+AT 보상 |
| `humanoid_ma_before_push.py`, `before_push_reward.py` | BEFORE solo PUSH 자산·배치·접촉 보상, 기존 BEFORE gate 유지 |
| `humanoid_ma_coupled_push.py`, `coupled_push_reward.py` | 공동 상자·가상 구역·그룹 gate·goal saturation |
| `utils/premature_push.py` (tokenhsi 기준) | 전원 준비 전 수평 이동 벌점·양의 손 힘 비율 배분, 선택적 설정 검증 |
| `humanoid_ma_before.py` | BEFORE 역할·공유 물체·AT progress·후행 초기화 |
| `scene_layout.py`, `scene_validation.py` | 활성 물체·distractor·loco 참조·물리 유효성 |
| `before_rsi_cache.py`, `native_rsi_cache.py` | 후행 pickUp/carryWith native 크기별 screening |
| `before_pickup_carry_rsi.py`, `before_relative_sampling.py` | RSI433 적용/fallback, near50 배치 |
| `size_rsi_cache.py` | Stage1 크기별 RSI 캐시 |
| `*_reward.py` | 과제 보상·BEFORE gate·CPA·물체 속도 |

## 정책과 과제 정의

- `tokenhsi/learning/multi_agent/amp_network_builder_ma.py`, `task_role_encoder.py`: 공통 GTA·task bias·사람 action MLP·actor/critic.
- 같은 디렉터리의 `before_network.py`, `human_coordination.py`: BEFORE 사람 attention·object packet·Stage1 전이·actor encoder 동결.
- `stage2_push_network.py`: BEFORE PUSH와 coupled의 동일 5과제/4family 정책·6N task/group suffix·4종 사람 relation·packet 기반 alias 전처리·Stage1 PUSH 전이. `before_push_network.py`는 같은 builder의 호환 import다.
- `viewer_diagnostics.py`: viewer 전용 native 보상 성공 진단, 정식 평가와 별도.
- `ma_agent.py`, `ma_players.py`, `scene_normalizer.py`: 장면 단위 PPO/AMP·저장/복원·평가·정규화.
- `tokenhsi/utils/{edge_*,task_role_spec,before_spec}.py`: 그래프·역할·packet·checkpoint 계약.
- `tokenhsi/utils/{coupled_push_spec,coupled_push_scene}.py`: 공동 그룹·인원별 물리 bank·직선 목표·충돌 검사. 공동 전용 source/goal 거리 설정과 단독 PUSH 배치를 분리한다. `box_first_people`는 상자·goal 기준 공동 사람 배치를 만들고 `translate_people`는 root/kinematic body XY를 함께 이동한다.
- `tokenhsi/utils/{before_push_spec,before_push_layout}.py`: solo PUSH 샘플링·계약·다중 사람 경로 배치.
- `tokenhsi/utils/{before_sampling,before_relative_sampling,before_object_alias}.py`: mix226/near50·토큰 alias.
- `tokenhsi/learning/multi_agent/stage1_push_network.py`: 5과제 정책·4 family AMP·선택적78k 전이.
- `tokenhsi/utils/{stage1_jointpush,jointpush_scene,jointpush_geometry,jointpush_rsi,jointpush_rsi_geometry}.py`: PUSH 계약·1/2명 장면·bank/모션/기하 검증.
- `tokenhsi/utils/{stage1_geometry,stage1_config,stage1_platforms,size_rsi}.py`: Stage1/BEFORE 공통 기하·RSI·물리 설정.
- `lpanlib/poselib/`: 기존 모션 배열 로딩 라이브러리.

## 검증과 문서

`tokenhsi/tests/`는 보상·배치·GTA·mask·RSI·전이·재로드·출력 경로를 검사한다. `before_test_utils.py`, `stage1_test_utils.py`는 CPU fixture다. `check_before_object_alias.py`·`check_before_push.py`·`check_coupled_push.py`는 실제 물리·부분 reset·추론 검사 진입점이다.

[PUSH](reference/stage1_push.md), [BEFORE PUSH](reference/stage2_before_push.md), [공통 Stage2/coupled](reference/stage2_coupled_push.md), [평가 규약](reference/evaluation.md), [Stage1 단독](../evaluation_single.md), [공통 정책](reference/ma_clean_scene_gta.md), [BEFORE](reference/stage2_before.md), [지표](reference/relation_diagnostics.md), [이관](reference/PORTING_GUIDE.md)을 참고한다. 이전 실험은 [legacy](legacy/README.md)에 보존한다.

`stage2_coupled_push_region_goals`는 공동 PUSH의 구역별 고정 목표 변형이다. 같은 환경 클래스와 모델을 사용하며, `coupled_push_spec.region_goal_offsets`·`HumanoidMACoupledPush._task_goal_positions`가 관측/AT보상/marker/평가의 구역 목표를 연결한다. 기하·계약 검사는 `test_coupled_region_goals.py`, 실제 CPU 추론 검사는 `check_coupled_region_goals.py`다.

## 로컬 Stage2 attention 관찰 도구

- [ksh_stage2_attention.py](../../ksh_stage2_attention.py): 모델을복사해agent순서를복원한QK/bias/attention을추출하고PyTorch MHA와대조한다.
- [수집entry](../../scripts/coupled_stage2_attention_capture.py): 기존player입력을읽기전용hook으로수집한다. `STAGE2_ATTN_PATTERNS`로1+3·2+2·4명공동PUSH를선택하며학습프로세스를수정하지않는다.
- [그림생성](../../scripts/render_coupled_stage2_attention.py): 원본GPU출력과재생출력을대조한뒤PNG·시간변화·interactive HTML을만든다. 실행은[USAGE](USAGE.md#stage2-actor-attention-지도)를따른다.

- [세 배치 비교 렌더러](../../scripts/render_coupled_stage2_comparison.py)와 [HTML template](../../scripts/stage2_attention_comparison.html): 같은모델의세배치에대해bias·raw QK·최종softmax를한줄에표시한다. 원본출력과재생출력/MHA/softmax를대조하고공통색상척도의PNG·시점별HTML을생성한다.

## 로컬 정식 평가의 종료 거리

- [공식 평가 wrapper](../../scripts/coupled_region_goals_eval.sh): `evaluate.py`의 `TOKENHSI_RUN_ENTRY`로 로컬 호환 진입점을 전달한다. 공식 general 프로토콜과 task별 집계는 유지한다.
- [PUSH 종료 거리 집계](../../ksh_push_terminal_eval.py): 완전한 첫 에피소드 trace에서 환경 종료와 개인 낙상 시점을 구분하고 구역/물리 상자별 거리를 CSV·JSON·보고서로 저장한다. [실행법](USAGE.md#로컬-정식-평가와-push-종료-거리).

- [PUSH 부호 거리 집계](../../ksh_push_signed_eval.py): 원래 reset snapshot을 기준으로 목표선 통과 시+, 미통과 시−인 목표거리와 진행방향 투영 오차를 계산한다. [명령·정의](USAGE.md#목표-통과-방향을-반영한-부호-거리).

- [종료·연속 유지 성공률](../../ksh_terminal_success_eval.py): 공식 점/영역 성공을 전체 trace에서 재채점해 종료 순간 성공과 첫 성공 이후 매 프레임 유지 성공을 구분한다. 낙상은 실패로 포함하며 PUSH 개인/그룹 전원을 분리한다. [명령·정의](USAGE.md#종료-성공률과-첫-성공-이후-연속-유지율).

별도 보상 변형: `stage2_coupled_push_premature_motion.yaml`과 같은 이름의 train/test/vnc wrapper. 기본 구역goal 정책/학습 YAML을 재사용하고 출력은 분리한다. CPU 검증은 `tokenhsi/tests/test_premature_push.py`다.

- 로컬 출발 대기 평가: `ksh_jointpush/scripts/coupled_departure_capture.py`는 정식 평가의 초기/제어 시점 접촉·물리 상자 상태를 관찰하고, `ksh_jointpush/ksh_departure_eval.py`는 물리 그룹별 최초 출발/준비 지표를 집계한다(CVPR2027 루트 기준).
