# 설치와 추가 실행 옵션

기본 명령은 [config](config.md), 구현 위치는 [structure](structure.md)를 따른다.

## 실행 환경

모든 wrapper는 `runtime_env.sh`로 기본 conda `tokenhsi`를 활성화하고 저장소 루트에서 실행한다. 다른 환경은 `TOKENHSI_CONDA_ENV=<이름>`, conda 설치 위치는 `CONDA_BASE`로 명시한다.

`TOKENHSI_GPU=0`처럼 물리 GPU를 한 번 지정한다. CUDA·기존 MPS·viewer Vulkan에 같은 GPU를 전달한다. `mps/`는 기존 daemon을 검사하며 실행 wrapper가 daemon을 시작하거나 종료하지 않는다.

PyTorch는 서버 GPU/CUDA 호환 설치본을 사용하며 Isaac Gym은 torch보다 먼저 import한다. `requirements.txt`는 남은 의존성이다. 현재 서버의 conda에는 Isaac Gym과 CUDA 라이브러리 활성화 설정이 있다.

## 데이터와 체크포인트

`tokenhsi/data/dataset_loco_sit_carry_climb.yaml`이 참조하는 `dataset_{amass_loco,amass_climb,sit,carry}/motions`를 공유 데이터에 심링크로 연결한다. 외부 데이터와 원본 checkout은 읽기 전용이다.

현재 네 모션 링크는 `/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/` 아래 각 데이터셋을 가리킨다. `tokenhsi/data/jointpush_motions`는 `/home/hwanhee/ksh/stage1_2_jointpush/tokenhsi/data/jointpush_motions`에 연결되어 있다.

기본 전이 가중치는 `checkpoints/Stage1_00078000.pth`다. 현재 링크 대상은 `/home/cvlab/Desktop/CVPR2027/stage1_message/output/stage1_v2_fixed/Stage1V2Fixed_15-11-38-22/nn/Stage1V2Fixed_00078000.pth`다. 다른 머신은 링크를 연결하거나 `STAGE1_CHECKPOINT=<절대경로>`로 BEFORE 새 학습의 source를 지정한다.

BEFORE checkpoint는 해당 실험 학습으로 생성한다. fixed 추론에는 기존 run의 `relation_config.yaml`도 필요하다. 확인 학습 저장본은 장기 성능 검증을 의미하지 않는다.

## 설정과 resume

- 새 학습은 선택한 환경·학습 YAML을 읽는다.
- 추론과 `RESUME_CHECKPOINT=<pth>`는 checkpoint 옆 `relation_config.yaml`의 env/sim을 읽는다. 학습 YAML은 해당 실험 것을 사용한다.
- `ENV_CONFIG=<yaml>`은 명시적인 환경 설정이다. fixed/BEFORE 모두 같은 옵션을 쓴다. 저장 설정이 없으면 현재 YAML로 조용히 대체하지 않는다.
- 기존 fixed snapshot에 sim 전체가 없으면 native fixed sim을 보완한다. 부분 sim과 rest_offset 불일치는 거부한다.
- 인원·환경 수·토큰 수·episode 길이·평가 초기화·영상 CLI를 마지막으로 적용하며 실험 계약을 검사한다.
- 모든 wrapper는 기본 `SEED=42`를 쓴다. `EPISODE_LENGTH`는 추론 episode 길이다.

## 확인 실행과 캐시

`CHECK=1`은 기본 `MAX_ITERATIONS=1`의 확인 학습이다. 환경 수는 fixed2048, BEFORE1024를 유지한다. 확인용 `OUTPUT_PATH`는 `output_etc/check/<검사명>` 아래를 지정한다. wrapper와 직접 `run.py` 모두 경로를 검사한다.

본학습 checkpoint·공용 캐시는 `output/`, 확인 작업과 그 RSI 캐시는 `output_etc/check/`에 둔다. screening 코드 변경 시 캐시 키가 바뀌며 물리 검사를 다시 수행할 수 있다. 원본 캐시는 수정하지 않는다.

## BEFORE 추론 초기화와 거리

- `BEFORE_PICKUP_CARRY_RSI=loco`가 기본이다. RSI433 두 설정은 `mix`, `pickup`, `carry`를 지원한다. `mix`는 후행 ON_TOP의40/30/30 요청 분포이며 유효 프레임/배치 실패는 loco로 fallback한다. mix226 loco 설정은 loco만 지원한다.
- `BEFORE_FOLLOWER_NEAR=0`은 저장된 혼합을 쓴다.1은 near50의 상대 가까움 또는 mix226 중간 모드다.2는 mix226의 가장 짧은 모드이며 near50에서는 거부한다.
- `TASK_ROLE_SWAP=1`은 fixed preset의 역할 교환이다. 과제·스킬 값은 [추론 옵션](config.md#추론-옵션)을 따른다.

## 서버 viewer

로컬 viewer는 화면 세션이 필요하다. VNC는 Xvfb·x11vnc·noVNC·websockify가 필요하며 `PORT`, `RESOLUTION`, `VNC_DIR`, `X11VNC`, `NOVNC_DIR`, `WEBSOCKIFY`로 위치·화면을 지정한다. GUI 로그는 `output_etc/check/gui/`다. 종료 시 이번 wrapper가 띄운 VNC 서비스만 종료한다.

## 이전 이름의 저장본

기존 Stage1 78000과 정리 전 BEFORE 저장본은 그대로 읽을 수 있다. `checkpoint_compat.py`가 저장 설정의 reward 태그·RSI 경로·로컬 전이 링크만 현재 이름으로 변환한다. 모델 tensor와 보상 수치·실험별 계약 검사는 유지한다. 새 학습의 출력 경로와 실험 이름은 현재 컨피그 이름을 따른다.

## 정식 평가와 viewer 구분

`*_test.sh`/`*_vnc.sh`는 인원·RSI를 바꿔 관찰하는 viewer/진단이다. `HEADLESS=1`만 붙여도 정식 프로토콜로 바뀌지는 않는다. 일반 정식 평가는 `*_eval.sh`로 학습 인원을 유지하고, Stage1 전용1명 비교는 `stage1_single_eval.sh`로 실행한다. 시간은 정식 평가·viewer 모두 기본 Stage1 900step, Stage2 1200step이다. [실행 명령](config.md), [평가 규약](reference/evaluation.md)을 따른다.


## Stage1 PUSH

`stage1_push`는 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%의 사용자 지정 비율을 사용하는 스크래치 학습이다. 기본2명·2048환경·4물체·900step, 같은 `tokenhsi` runtime을 사용한다. 주어진21k finetune PTH의 일반 평가는 저장된 PUSH50% 분포를 복원하고 시간만900으로 맞춘다. 옆에 `relation_config.yaml`이 없으면 PTH의 `relation_experiment_config`를 읽는다. 서로 다른 분포/horizon의 PTH를 resume하지 않는다.

`stage1_push_single_eval.sh`는 실제1명·행동별 평가이며 PUSH도 포함한다. AT만9종 크기를 쓰고 PUSH는 학습167종 크기를 쓴다. 기존4과제 macro와 전체5과제 macro를 별도로 보존한다. [실행 명령](config.md), [구현 상세](reference/stage1_push.md)를 따른다.

PUSH 모션21개는 `tokenhsi/data/jointpush_motions/` 아래 읽기 전용 원본으로 연결한다. 기본 원본은 `/home/cvlab/Downloads/push_amp_raw_rsi_environment_final_20261024/data/motions/`이고 bank의 파일 해시를 검증한다. `checkpoints/Stage1PushFinetune_00021000.pth`는 Downloads의 원본 PTH를 가리킨다. 원격 파일 이름의 jointpush는 여기서는 독립 과제 통합이며 Stage2 공동 PUSH를 포함하지 않는다.

## BEFORE PUSH

`stage2_before_push`는 `stage2_before`의 독립 역할에 PUSH를 추가한다. 학습 진입점과 viewer/VNC/일반 평가 명령은 [config](config.md), 샘플링·물리·전이 설정은 [BEFORE PUSH](reference/stage2_before_push.md)에 있다.

새 학습은 기본 `checkpoints/Stage1PushFinetune_00021000.pth`에서 전이한다. `STAGE1_CHECKPOINT=/absolute/path/to/stage1_push.pth`로 다른 Stage1 PUSH 학습본을 선택할 수 있다. 기존 4과제 Stage1 가중치는 사용할 수 없다. resume는 `RESUME_CHECKPOINT`에 같은 BEFORE PUSH 실험의 PTH를 지정한다. viewer와 일반 평가에도 **Stage2 BEFORE PUSH PTH**를 사용한다. 전이 원본인 Stage1 PUSH PTH를 Stage2 저장본으로 대신 사용하지 않는다.

`TASK_GRAPH=push`는 4명 모두 독립 PUSH, `all_solo`는 다섯 행동 독립 샘플링이다. `before_sit`·`before_climb`·`before_stack`은 기존처럼 BEFORE 쌍을 고정한다. 일반 정식 평가는 저장된 학습 설정의 `random_scenario`를 사용하며, Stage2용 1명 단독 평가 wrapper는 제공하지 않는다.

## 공통 Stage2 PUSH와 coupled

`stage2_coupled_push_individual_goal_solo5_keep1`·`stage2_coupled_push_individual_goal_solo5_shrink1`은 solo5 기반 새 학습 설정이다. 학습·추론 모두 각 사람의 COUPLED 상대를 최대1명으로 선택하고 나머지 공동 상대는 NONE으로 바꾼다. keep1은 미접촉 상대가 없으면 접촉한 상대1명을 유지하고, shrink1은0명으로 줄인다. 선택은 접촉 완료/상실에 따라 갱신하며 actor/critic의4층에 같은 ID를 적용한다. 관측에 사람당 선택 ID1개(없으면−1)를 기록해 PPO 미니배치에서도 당시 선택을 재사용한다. 기본4명/8토큰 관측은1276차원이며 마지막 그룹 N개는 유지한다. 기존 보상·물성·거리·4명/1024환경·Stage1 PUSH26k 전이는 그대로다. 두 설정과 기존 solo5 PTH는 혼용하지 않으며 `COUPLED_PEERS` 뷰어 override와 함께 쓸 수 없다. 0명은 COUPLED가0명이라는 뜻이며 다른 사람은 NONE으로 계속 보인다. [상세](reference/stage2_coupled_push.md#최대1명-coupled-선택-학습), [학습·추론 명령](config.md).

개인 목표 coupled test/VNC는 `COUPLED_PEERS=all|keep3|shrink3`(직접 실행 `--coupled_peers`)으로 접촉 상태에 따른 사람 attention 상대 선택을 비교한다. 기본 `all`과 4명 이하 공동 그룹은 기존 계산이다. 5명 이상 그룹에서는 각 query가 자신을 제외한 미접촉 상대 중 최대3명을 무작위 선택하고, 유지 가능한 슬롯은 그대로 둔다. 선택된 상대가 접촉하거나 누군가 접촉을 잃어 미접촉 상대를 우선해야 할 때만 필요한 슬롯을 갱신한다. 별도 RNG를 사용하며 씬/부분 reset에서 해당 선택을 초기화한다.

`keep3`은 미접촉 상대가 3명 미만이면 접촉한 상대를 포함해3명을 유지한다. 전원 접촉 후에도 그3명을 유지한다. `shrink3`은 접촉한 상대를 제외해 미접촉 인원이2/1/0명이면 선택도2/1/0명이다. 접촉을 잃은 상대는 다시 후보가 된다. 접촉 기준은 기존 개인 구역의 양손 기하 판정 `on_face`이며 힘 접촉이나 목표 도착을 뜻하지 않는다. 별도의 시간 완충은 없어 접촉 경계에서는 선택이 다시 바뀔 수 있다.

선택되지 않은 COUPLED 상대를 actor의4개 사람 attention 층에서 NONE으로 바꾼다. attention에서 제외하지 않으며 SELF/BEFORE·물체 attention은 유지한다. 실제 공동 그룹·공유 상자·개인 목표·보상/성공 gate·물성은 그대로다. 환경의 기하 접촉 정보를 사용하는 추론 개입이므로 학습된 협응 자체의 증거로 해석하지 않는다. 학습/resume·정식 `--eval_protocol`에는 적용하지 않는다. [비교 명령](config.md#추론-옵션), [선택 구현](../tokenhsi/utils/coupled_push_peer_selection.py).

8명 선택 방식의 현재 NONE 전환 비교는 [전용 실행기](../output_etc/check/solo5_peer_none_eval8_epoch12000_20261101/run_worker.sh)로 수행한다. `TOKENHSI_GPU=5 bash output_etc/check/solo5_peer_none_eval8_epoch12000_20261101/run_worker.sh keep3`와 `TOKENHSI_GPU=6 bash output_etc/check/solo5_peer_none_eval8_epoch12000_20261101/run_worker.sh shrink3`를 사용한다. 기존12000 PTH·64환경×seed0·물체16개·ordered·원래 질량·평균 액션·1200step·첫 episode 조건이며 all은 직전 결과를 재사용한다. [NONE 전환 결과](../output_etc/check/solo5_peer_none_eval8_epoch12000_20261101/report.md). [이전 마스킹 결과](../output_etc/check/solo5_peer_eval8_epoch12000_20261101/report.md)는 다른 조건의 과거 기록이다. 그 실행기를 현재 코드로 다시 실행하면 NONE 방식이 적용된다.

공동 PUSH의 test/VNC에서 `COUPLED_ASSIGNMENT=random|ordered|distance|optimal`을 선택한다(직접 실행은 `--coupled_assignment`). 기본 `random`은 기존 배정이다. `ordered`는 각 공유 상자의 local X 방향으로 사람 root를 정렬해 왼쪽부터 담당 구역을 배정하고, `distance`는 Hungarian으로 총 접근 거리를 최소화한다. `optimal`은 예상 접근 경로의 교차 수 → 최장 접근 거리 → 총 접근 거리 순서로 최소화한다. 8명까지 전수 비교하고, 9명 이상은 ordered/distance 중 좋은 배정에서 최대2N회 최선의 두 사람 교환으로 개선한다. 큰 그룹의 전역 최적해는 보장하지 않으며 인원 상한은 없다.

접근 비용·교차는 초기 사람 root에서 상자 뒤쪽(local −Y) 담당 구역 앞 `human_clearance` 지점까지의 직선으로 추정한다. 장애물 우회·실제 정책 궤적·낙상 방지를 보장하는 경로 계획은 아니다. 최종 장면 배치 후 reset마다 한 번 정하고 개인 목표·관측 토큰·접촉 구역에 같은 번호를 사용한다. 에피소드 중 배정을 바꾸지 않으며 기존 4면 접촉 선택은 계속 동작한다. 기존 random 추첨도 소비해 동일 seed의 물리 장면/RNG를 보존한다. 학습/resume·정식 `--eval_protocol`에는 비random 배정을 허용하지 않는다. [명령](config.md#추론-옵션).

사용자 요청에 따라 이후 공동 PUSH 평가는 `ordered`를 사용한다. 현재 정식 프로토콜의 비random 제한은 유지하므로 [ordered 평가 전용 실행부](../output_etc/check/solo5_ordered_trend64_seed0_20261101/monitor.py)로 첫 episode 채점·gate/접촉 기록을 수행하고 `formal_report=false`로 구분한다. [추세 실행기](../output_etc/check/solo5_ordered_trend64_seed0_20261101/run_worker.sh)는 `TOKENHSI_GPU=0` 또는 `3`으로 실행하며, 고정한7개 PTH의 일반 혼합 및4k/8k/12k의6·8명 전원 공동을64환경·seed0·원래 무게·평균 액션·1200step으로 평가한다. 같은 seed의 초기 장면을 비교하며 random 배정 결과와 직접 합산하지 않는다.

같은4k/8k/12k의5명·물체10개 전원 공동 추가 평가는 [5명 실행기](../output_etc/check/solo5_ordered_trend64_n5_seed0_20261101/run_worker.sh)를 `TOKENHSI_GPU=0` 및 `3`으로 실행한다. 나머지 조건은 위64환경 추세 평가와 같다.

`stage2_coupled_push_individual_goal_solo5_countnorm_coupled`와 `stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none`은 solo5 기반의 새 학습 설정이다. `relationGraph.human_attention_count_normalization`은 각각 `coupled`·`coupled_none`이며 actor/critic의4개 사람 attention 층에서 query별 해당 관계의 유효 상대 수 K에 대해 logit에 `−log K`를 더한다. 전자는 COUPLED만, 후자는 COUPLED와 NONE을 각각 보정하고 SELF/BEFORE는 유지한다. 기본값 `none`인 기존 설정은 계산을 유지한다. 모드는 저장 env/체크포인트 계약에 포함되며 두 실험과 기존 solo5 저장본을 resume/추론에 혼용하지 않는다. 둘 다 기존 Stage1 PUSH26k에서 새로 시작하고 학습 기본은4명·1024환경·8토큰이다. [정의](reference/stage2_coupled_push.md#사람-attention-관계별-개수-보정), [명령](config.md).

두 norm의500 PTH 비교는 [500 평가 실행기](../output_etc/check/solo5_norm500_ordered_eval_20261102/run_worker.sh)에 `coupled` 또는 `coupled_none`을 전달한다. `TOKENHSI_GPU=5 bash output_etc/check/solo5_norm500_ordered_eval_20261102/run_worker.sh coupled`, `TOKENHSI_GPU=6 bash output_etc/check/solo5_norm500_ordered_eval_20261102/run_worker.sh coupled_none`으로 실행한다. 각 PTH를 복사 고정하며4명/8토큰 일반 혼합64환경×seed0와6명/12토큰·8명/16토큰 전원 공동 각각32환경×seed0를 ordered·원래 무게·평균 액션·1200step으로 채점한다. `COUPLED_PEERS=all`로 상대 제한 없이 저장된 norm만 적용하며 정식 프로토콜과 구분되는 첫 episode 진단이다. [단독·공동 ever/final·검증](../output_etc/check/solo5_norm500_ordered_eval_20261102/report.md).

기존 solo5의500 PTH도 [기준 평가 실행기](../output_etc/check/solo5_base500_vs_norm500_20261102/run_worker.sh)로 같은 조건에 맞춰 추가한다. `TOKENHSI_GPU=5 bash output_etc/check/solo5_base500_vs_norm500_20261102/run_worker.sh general coupled8`, `TOKENHSI_GPU=6 bash output_etc/check/solo5_base500_vs_norm500_20261102/run_worker.sh coupled6`를 사용한다. norm 두 모델의 직전 결과를 재사용하며 초기 상태/RNG를 대조한다. [기존·CO·CO+NONE의500 비교표](../output_etc/check/solo5_base500_vs_norm500_20261102/report.md).

실행 중인 기존 solo5·CO norm·CO+NONE norm·keep1 네 설정의 최신 정기 PTH 비교는 [네 모델 실행기](../output_etc/check/coupled_four_latest_eval_20261102/run_worker.sh)에 `solo5`, `norm_co`, `norm_co_none`, `keep1`을 각각 전달한다. GPU5에서 `solo5`/`norm_co_none`, GPU6에서 `norm_co`/`keep1`을 실행했다. 일반64환경·6/8명 각각32환경×seed0·ordered·1200step이며 각 저장 설정을 유지한다.17k/2k/3.5k/3k snapshot 비교로 학습량이 같지 않고, 실행/PTH가 없는 shrink1은 포함하지 않는다. [PTH·결과·검증](../output_etc/check/coupled_four_latest_eval_20261102/report.md).

같은 저장 궤적의 접촉 전후 정지·재출발 CPU 분석은 `python output_etc/check/coupled_four_latest_eval_20261102/stop_start_analysis.py`로 재실행한다. [정의·CO+NONE 인원별·네 설정 비교](../output_etc/check/coupled_four_latest_eval_20261102/stop_start_report.md).

전원 조건을 푼 첫 정지와 재출발 순간의 접촉 인원은 `python output_etc/check/coupled_four_latest_eval_20261102/contact_count_analysis.py`로 분석한다. [네 설정 비교·기하/힘 접촉 인원 분포](../output_etc/check/coupled_four_latest_eval_20261102/contact_count_report.md).

기존 solo5·CO norm·CO+NONE norm의 동일2000epoch 비교는 [실행기](../output_etc/check/coupled_epoch2000_compare_20261102/run_worker.sh)에 `solo5` 또는 `norm_co_none`을 전달하고 GPU5·6에서 각각 실행한다. 일반64환경·6/8명 각각32환경×seed0·ordered·원래 무게·평균 액션·1200step을 유지하며 CO2k는 직전 검증된 세 rollout을 재사용한다. PTH는 [manifest](../output_etc/check/coupled_epoch2000_compare_20261102/manifest.json)에 고정했다. [성공률](../output_etc/check/coupled_epoch2000_compare_20261102/report.md), [정지 후 출발·접촉 인원](../output_etc/check/coupled_epoch2000_compare_20261102/contact_count_report.md).

keep1을 추가한 동일2000epoch 네 모델 비교는 [실행기](../output_etc/check/coupled_epoch2000_keep1_compare_20261102/run_worker.sh)에 `keep1 keep1_general`, `keep1 keep1_coupled6`은 GPU5, `keep1 keep1_coupled8`은 GPU6으로 전달한다. keep1의 저장된 actor/critic 최대1명 선택·미선택 NONE과 같은 평가 조건을 유지하며 나머지9개 rollout은 재사용한다. [성공률·PTH·재실행](../output_etc/check/coupled_epoch2000_keep1_compare_20261102/report.md), [정지 후 출발·접촉 인원](../output_etc/check/coupled_epoch2000_keep1_compare_20261102/contact_count_report.md).

기존18000·CO+NONE4500·keep1 3500epoch의 최신 저장본 비교는 [실행기](../output_etc/check/coupled_three_latest_eval_20261102/run_worker.sh)에 `solo5`, `norm_co_none`, `keep1`을 전달하고 GPU5·6에서 각각 실행한다. 각 설정의 저장 동작·일반64환경·6/8명 각각32환경×seed0·ordered·원래 무게·평균 액션·1200step을 유지한다. 학습량이 다른 최신 상태 비교이며 [PTH 고정](../output_etc/check/coupled_three_latest_eval_20261102/manifest.json), [성공률·재실행](../output_etc/check/coupled_three_latest_eval_20261102/report.md), [정지 후 출발·접촉 인원](../output_etc/check/coupled_three_latest_eval_20261102/contact_count_report.md)에 기록한다.

기존 solo5의14k/16k/18k 추세 추가는 [실행기](../output_etc/check/solo5_ordered_trend64_14k18k_20261102/run_worker.sh)에 GPU5는 `0`, `2`, GPU6은 `1`, `3`을 각각 전달한다. 일반4명 혼합 및6/8명 전원 공동을 모두64환경×seed0·ordered·원래 무게·평균 액션·1200step으로 평가하고5명은 추가하지 않는다.18k 일반은 직전 동일64환경 결과를 재사용한다. [과거1k~12k와 합친 표·그림·재실행](../output_etc/check/solo5_ordered_trend64_14k18k_20261102/report.md), [PTH·9개 구성](../output_etc/check/solo5_ordered_trend64_14k18k_20261102/manifest.json).

기존 solo5의8k/10k/14k/18k 정밀 비교는 [실행기](../output_etc/check/solo5_ordered_512_seed012_8k10k14k18k_20261102/run_worker.sh)에 GPU5는 `0`, `2`, GPU6은 `1`, `3`을 전달한다. 각 PTH를 일반4명·8물체 혼합512환경×seed0/1/2·ordered·원래 무게·평균 액션·distractor·loco 초기화·1200step으로 평가하고6/8명은 추가하지 않는다. 단독5행동·공동2/3/4명·공유 중심 Ever로 선택하며 Final은 정지·유지 참고자료로만 사용한다. [PTH 고정](../output_etc/check/solo5_ordered_512_seed012_8k10k14k18k_20261102/manifest.json), [seed 평균±표준편차·재실행·집계](../output_etc/check/solo5_ordered_512_seed012_8k10k14k18k_20261102/report.md).

같은 네 PTH의 단독AT·ON_TOP 목표 거리 진단은 [실행기](../output_etc/check/solo5_at_ontop_goal5_32_8k10k14k18k_20261102/run_worker.sh)에 GPU5는 `0`, `2`, GPU6은 `1`, `3`을 전달한다. 각 행동·조건32환경×4명·8물체·seed0·1200step이며 기존 환경 중심 정사각형 목표와 payload→목표1–5m를 비교한다. 복원 후 첫 reset 전에 평가용 `solo_carry_goal_distance`만 적용하고 PTH·학습 설정은 보존한다. [조건·PTH](../output_etc/check/solo5_at_ontop_goal5_32_8k10k14k18k_20261102/manifest.json), [Ever/Final·실제 거리·재현](../output_etc/check/solo5_at_ontop_goal5_32_8k10k14k18k_20261102/report.md).

CO+NONE norm1k/2k/3k/4k/5k와 keep1 1k/2k/3k/4k 추세는 [실행기](../output_etc/check/nonenorm_keep1_trend64_1k5k_20261102/run_worker.sh)에 GPU5는 `0`, `2`, GPU6은 `1`, `3`을 전달한다. 각4명·8물체 혼합64환경×seed0·ordered·원래 저장 목표 샘플링·loco·평균 액션·원래 무게·distractor·1200step으로 평가한다. 저장된 actor/critic 보정과 keep1 선택을 유지하고6/8명·AT/ON_TOP 거리 변경은 추가하지 않는다. [PTH·구성](../output_etc/check/nonenorm_keep1_trend64_1k5k_20261102/manifest.json), [단독/공동/공유 중심 Ever·Final·그림·재실행](../output_etc/check/nonenorm_keep1_trend64_1k5k_20261102/report.md).

두 모델의 동일4k·6/8명 확장 비교는 [실행기](../output_etc/check/nonenorm_keep1_epoch4k_n68_32_20261102/run_worker.sh)에 GPU5는 `0`, `2`, GPU6은 `1`, `3`을 전달한다. 각 전원 공동6명·12물체/8명·16물체32환경×seed0·ordered·원래 인원별 상자 크기/무게·평균 액션·loco·distractor·1200step이며 저장된 norm/keep1 설정을 유지한다. [PTH·4개 구성](../output_etc/check/nonenorm_keep1_epoch4k_n68_32_20261102/manifest.json), [Ever/Final·공유 중심·개별 성공·재현](../output_etc/check/nonenorm_keep1_epoch4k_n68_32_20261102/report.md).

기존 solo5의2k/4k/6k 및10k–20k 매1k 비교는 [실행기](../output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/run_worker.sh)에 GPU2는 `0`, `2`, `4`, GPU4는 `1`, `3`, `5`를 전달한다. 각 worker는1개씩 순차 실행해 GPU당 최대3개를 유지한다.4명·8물체 혼합64환경×seed0·ordered·원래 목표 샘플링·loco·평균 액션·원래 무게·distractor·1200step이며2k/4k/6k/12k 재평가와 미평가11k/13k/15k/19k/20k를 새로 실행하고 나머지5개를 재사용한다.20k는 저장 완료·내부 epoch/frame·해시 확인 후 고정 복사본으로 평가한다. [구성·출처](../output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/manifest.json), [전체 비교표·그림·재현](../output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/report.md).

기존 solo5의15k/19k 정밀 평가는 [실행기](../output_etc/check/solo5_ordered_512_15k19k_vs10k_gpu2_20261102/run_worker.sh)에 GPU2로 `0`, `1`, `2`를 전달한다. 각 worker가 같은 seed의15k→19k를 순차 실행해 최대3개를 유지한다.4명·8물체 혼합512환경×seed0/1/2·ordered·원래 목표 샘플링·loco·평균 액션·원래 무게·distractor·1200step이며 기존10k의 동일3seed 결과를 재사용한다. 공동 Ever를 우선하고 단독 Ever를 다음 기준으로 비교하며 Final은 참고자료다. [PTH·구성·출처](../output_etc/check/solo5_ordered_512_15k19k_vs10k_gpu2_20261102/manifest.json), [평균±표준편차·전체 표·그림·재현](../output_etc/check/solo5_ordered_512_15k19k_vs10k_gpu2_20261102/report.md).

solo5의10k/15k/19k AT 지정 상자9종 진단은 `TOKENHSI_GPU=2 bash output_etc/check/solo5_at9_10k15k19k_gpu2_20261102/run_worker.sh 0`으로 순차 실행한다.4명·8물체 AT 전용512환경×seed0/1/2·1200step이며 물리 asset 생성 전에 single AT9종 크기를 적용하고 밀도100kg/m³를 유지한다. 목표 샘플링·loco·평균 액션·distractor는 기존 설정이다. single의1명 평가나 이전 혼합 평가와 장면 구성이 다르다. [고정 PTH·구성](../output_etc/check/solo5_at9_10k15k19k_gpu2_20261102/manifest.json), [전체 및 크기별 Ever/Final·재현](../output_etc/check/solo5_at9_10k15k19k_gpu2_20261102/report.md).

nonenorm6.5k·keep1 5.5k의 일반 정밀 평가는 `TOKENHSI_GPU=4 bash output_etc/check/nonenorm_keep1_eval_gpu4_20261102/run_worker.sh 0`과 같은 명령의 worker `1`로 실행한다. GPU4만 사용하며 최대2개 평가가 각 seed0/1/2를 순차 처리한다.4명·8물체 혼합512환경·ordered·loco·원래 목표/크기/무게·평균 액션·distractor·1200step이고 저장된 actor/critic 보정 및 keep1 상대 선택을 유지한다. 기존 solo5 10k의 동일3seed 결과는 재사용하는 기준점이며 학습량이 다른 비교다. [고정 PTH·구성](../output_etc/check/nonenorm_keep1_eval_gpu4_20261102/manifest.json), [단독5행동·공동2/3/4명·공유 중심 Ever/Final·재현](../output_etc/check/nonenorm_keep1_eval_gpu4_20261102/report.md).

기존 solo5·nonenorm·keep1을 모두6k로 맞춘6/8명 공동 비교는 `TOKENHSI_GPU=4 bash output_etc/check/nonenorm_keep1_n68_env64_gpu4_20261102/run_worker.sh 0`과 같은 명령의 worker `1`로 실행한다. 각 worker가6명/8명을 담당하고 세 모델을 순차 처리해 GPU4에서 최대2개 평가를 유지한다. 각64환경×seed0·6명/12물체 및8명/16물체·전원 공동·ordered·loco·인원별 원래 크기/무게·평균 액션·distractor·1200step이며 keep1도6000epoch 저장 완료·frame/해시 확인 후 고정했다. [세6k PTH·구성](../output_etc/check/nonenorm_keep1_n68_env64_gpu4_20261102/manifest.json), [전원/공유 중심/개인 Ever·Final·재현](../output_etc/check/nonenorm_keep1_n68_env64_gpu4_20261102/report.md).

solo5 10k/15k/19k의 공유 중심40cm 재집계는 tokenhsi 환경에서 `PYTHONPATH=tokenhsi python output_etc/check/solo5_shared_center40_10k15k19k_20261102/rescore.py`로 실행한다. 기존512환경×3seed 저장 궤적9회의 실제 중심↔공유 목표3D 거리를 CPU에서 다시 계산하며 시뮬레이션을 실행하지 않는다.20cm 대조군 재현·NumPy/Torch 프레임별 일치·원본 trace 해시를 확인하고 표준 평가 기본값20cm는 유지한다. [Ever/Final·인원별·20cm 비교](../output_etc/check/solo5_shared_center40_10k15k19k_20261102/report.md), [출처·원본 해시](../output_etc/check/solo5_shared_center40_10k15k19k_20261102/manifest.json).

세6k의8명 혼합·공동 최대4명 진단은 `TOKENHSI_GPU=3 bash output_etc/check/coupled_6k_mixed8_cap4_env64_gpu356_20261102/run_worker.sh solo5`, 같은 GPU의 `keep1`, GPU5의 `norm_co_none`으로 실행한다. GPU별 lock이 평가 동시 실행을1개로 제한한다.64환경×seed0·8명/16물체·random_scenario·ordered·1200step이고 공동 인원만2~min(4,남은 인원)으로 제한하며 reset 직전 난수 seed0를 공통 적용한다. 기존 규칙과 표준 학습 코드는 유지한다. 초기 배치 실패 및 GPU6 keep1 배치 차이는 비교에서 제외·보존했으며 최종 세 모델 초기 상태/RNG 일치를 확인했다. [전체 Ever/Final·20→40cm·단독·분할·재현](../output_etc/check/coupled_6k_mixed8_cap4_env64_gpu356_20261102/report.md), [구성·시도 내역](../output_etc/check/coupled_6k_mixed8_cap4_env64_gpu356_20261102/manifest.json).

동일6k·전체 인원4명→8명 비교는 `TOKENHSI_GPU=3 bash output_etc/check/coupled_6k_mixed4to8_cap4_env64_gpu35_20261102/run_worker.sh solo5`, 같은 GPU의 `keep1`, GPU5의 `norm_co_none`으로4명·8물체64환경을 보충하고 검증된8명·16물체64환경 결과를 재사용한다. 두 조건 모두seed0·reset 직전 공통seed0·ordered·공동 최대4명·1200step이다. 각 인원 내 모델별 초기 상태/RNG는 같고4명과8명의 장면/그룹 분모는 다르다. tokenhsi 환경에서 `PYTHONPATH=tokenhsi python output_etc/check/coupled_6k_mixed4to8_cap4_env64_gpu35_20261102/compare.py`로20cm 공유 중심·전원 개인 목표·단독 Ever/Final 화살표 표를 생성한다. [비교표·분모·재현](../output_etc/check/coupled_6k_mixed4to8_cap4_env64_gpu35_20261102/report.md), [구성·재사용 출처](../output_etc/check/coupled_6k_mixed4to8_cap4_env64_gpu35_20261102/manifest.json).

`stage2_before_push`와 `stage2_coupled_push`는 동일한 5과제·4family·4종 사람 relation 모델을 사용한다. 관측은1272차원(4명·8토큰), AMP1330차원이며 두 Stage2의 제한 시간은1200step이다. 기존 BEFORE PUSH의5N suffix 저장본은 새 모델과 호환되지 않는다. Stage1 PUSH21k에서 각각 새 전이를 시작한다. 전이 원본은 `STAGE1_CHECKPOINT`, 같은 실험 resume은 `RESUME_CHECKPOINT`로 지정한다. 두 Stage2의 학습/추론/평가 명령은 [config](config.md)에 있다.

coupled의 인원은 N≥1, 물체 토큰은 M≥N이며 인원 상한은 없다. `TASK_GRAPH=all_coupled`는 지정 인원 전원 공동 PUSH, `all_solo`는 독립5행동 혼합, `push`는 전원 독립 PUSH다. 기존 PTH의 가중치·정규화 통계를 그대로 사용하며 추론에서는 저장 당시와 다른 N/M을 지정할 수 있다. 본학습 기본은4명·8토큰·random_scenario이고, resume은 저장된 N/M과 일치해야 한다. [공통 구조·coupled 상세](reference/stage2_coupled_push.md), [임의 인원 관찰 명령](config.md#추론-옵션).

`stage2_coupled_push_goal5`는 공동 PUSH의 사람→상자2–10m·상자→목표2–5m를 사용한다. 독립 PUSH는 두 거리 모두2–10m이며 기본26k 전이 원본은 `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`다. 기본 coupled와 별도 학습·출력·저장본을 사용한다. 실행 명령은 [config](config.md), 상세는 [공동 PUSH](reference/stage2_coupled_push.md)를 따른다.

`stage2_coupled_push_individual_goal`은 goal5의 공동 거리·26k 전이를 유지하고 개인 분할 목표/AT 보상과 그룹 공통 4면 CONTACT를 사용한다. 단독 거리 설정은 기존 범위를 사용한다. 이전 거리/확률 설정 및 다른 coupled 저장본과 계약이 달라 resume/추론을 혼용하지 않는다. 각 실험의 전용 wrapper를 사용한다. 새 평가의 `result.json`·`summary.json`에는 2/3/4명별 개인 ever/final 및 전원 동시 도착 ever/final `coupled_scores`를 추가한다. 개인 목표 원본과 solo5는 실제 공유 상자 중심→공유 중심 목표20cm 도달의 그룹별 ever/final `shared_center_scores`도 함께 저장한다. 최소1명 유효 조건이며 개인 전원 동시 성공과 구분한다. [설계](reference/stage2_coupled_push.md), [명령](config.md).

`stage2_coupled_push_individual_goal_solo5`는 위 개인 목표 설정에서 단독 접근과 단독 PUSH 이동의 최대 거리만5m로 제한한다. 공동 확률0.5·공동 접근2–10m/이동2–5m·26k 전이를 유지한다. 원본과 전용 YAML·학습 이름·출력·train/test/VNC/eval wrapper를 분리한다. [거리 설정](reference/stage2_coupled_push.md), [명령](config.md).

`stage2_coupled_push_individual_goal_solo5_carry5`는 solo5에서 단독 CARRY AT·ON_TOP 운반 목표만 상자 기준 반경1–5m·전 방향 균등 후보로 변경한다. 사람↔목표 최소1m·물리 충돌/배치 검사는 유지한다. 새 YAML·학습 이름·출력·train/test/VNC/eval을 사용하며 기존 solo5 저장본과 resume/추론 계약을 혼용하지 않는다. 기본 전이는 같은 Stage1 PUSH26k다. [샘플링 상세](reference/stage2_coupled_push.md), [명령](config.md).

`stage2_coupled_push_individual_goal_solo5_carry5_ground`는 위 carry5의 운반 상자·AT 목표를 지면에 두는 별도 설정이다. AT/ON_TOP 출발 선반과 AT 목표 선반은 각각 Z−20m/−19.5m에 두고 ON_TOP의 목표 받침 상자는 유지한다. 거리·물성·보상·Stage1 PUSH26k 전이는 carry5와 같고 기존 carry5 PTH와는 높이 계약이 다르다. [설정 상세](reference/stage2_coupled_push.md#carry5-지면-운반-변형), [학습·추론 명령](config.md).

ground5k·nonenorm13k 최신 저장본의 일반512환경×seed0/1/2 평가는 `TOKENHSI_GPU=5 bash output_etc/check/ground_nonenorm_latest512_gpu5_20261103/run_worker.sh 0`과 같은 명령의 `1`로 재현한다. 각 worker가 한 모델의3회를 순차 실행하며 GPU5 평가 peak2개다.4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step과 저장된 보정/높이/목표를 유지한다. 학습량·목표 분포가 다른 최신 상태 비교이며 동일 장면의 구조 비교는 아니다. [Ever/Final·평균±seed 편차·분모](../output_etc/check/ground_nonenorm_latest512_gpu5_20261103/report.md), [PTH·구성](../output_etc/check/ground_nonenorm_latest512_gpu5_20261103/manifest.json), [검증](../output_etc/check/ground_nonenorm_latest512_gpu5_20261103/verification.json). 보고서에는 같은 규모·ordered의 기존 solo5 8k·14k 평가도 재사용했다. `python output_etc/check/ground_nonenorm_latest512_gpu5_20261103/report.py`로 기존 포함 표와 [집계·출처](../output_etc/check/ground_nonenorm_latest512_gpu5_20261103/with_solo5_summary.json)를 재생성한다.

ground5k·nonenorm13k의6/8명 전원 공동64환경×seed0 진단은 `TOKENHSI_GPU=5 bash output_etc/check/ground_nonenorm_n68_env64_gpu56_20261103/run_worker.sh 0`(ground)와 `TOKENHSI_GPU=6 bash output_etc/check/ground_nonenorm_n68_env64_gpu56_20261103/run_worker.sh 1`(nonenorm)로 재현한다. 각 worker가6명/12물체·8명/16물체를 순차 평가하며 ordered·loco·평균 액션·원래 물성/distractor·1200step을 유지한다. 기존 solo5 6k·14k의 같은 규모4회는 재사용했다. [공유 중심20cm·전원 개인 목표 Ever/Final·건수](../output_etc/check/ground_nonenorm_n68_env64_gpu56_20261103/report.md), [PTH·재사용 출처](../output_etc/check/ground_nonenorm_n68_env64_gpu56_20261103/manifest.json), [검증](../output_etc/check/ground_nonenorm_n68_env64_gpu56_20261103/verification.json).

기존 solo5와nonenorm 동일10k/frame1310720000의512환경×seed0/1/2 비교는 기존solo5 평가3회를 재사용하고 nonenorm만 새로 평가한다. 재현은 `TOKENHSI_GPU=5 bash output_etc/check/nonenorm_vs_solo5_epoch10000_512_gpu56_20261103/run_worker.sh 0`, `TOKENHSI_GPU=6 bash output_etc/check/nonenorm_vs_solo5_epoch10000_512_gpu56_20261103/run_worker.sh 1` 및 같은GPU6 명령의 `2`다.4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step이며 초기12필드가seed별로 일치한다. [Ever/Final·편차·분모](../output_etc/check/nonenorm_vs_solo5_epoch10000_512_gpu56_20261103/report.md), [고정PTH·기존 출처](../output_etc/check/nonenorm_vs_solo5_epoch10000_512_gpu56_20261103/manifest.json), [검증](../output_etc/check/nonenorm_vs_solo5_epoch10000_512_gpu56_20261103/verification.json).

ground5k·6k·7k의512환경×seed0/1/2 추세는5k3회를 재사용하고6k·7k각3회를 새로 평가했다. 재현은 `TOKENHSI_GPU=5 bash output_etc/check/ground_5k6k7k_eval512_gpu56_20261103/run_worker.sh 0`, `TOKENHSI_GPU=6 bash output_etc/check/ground_5k6k7k_eval512_gpu56_20261103/run_worker.sh 1` 및 같은GPU6 명령의 `2`다.4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step이며5k대비 초기12필드가seed별로 일치한다. [Ever/Final·편차·분모](../output_etc/check/ground_5k6k7k_eval512_gpu56_20261103/report.md), [고정PTH·구성](../output_etc/check/ground_5k6k7k_eval512_gpu56_20261103/manifest.json), [검증](../output_etc/check/ground_5k6k7k_eval512_gpu56_20261103/verification.json).

위ground5k·6k·7k의 단독 PUSH20/30/40/50cm 재집계는 `python output_etc/check/ground_5k6k7k_eval512_gpu56_20261103/solo_push_radius/rescore.py`로 실행한다. 기존9개trace의3D 중심 거리와 유효 프레임을CPU float64로 재채점하며 추가 시뮬레이션은 없다. [Ever/Final·편차·건수·검증](../output_etc/check/ground_5k6k7k_eval512_gpu56_20261103/solo_push_radius/report.md).

carry5의1k 저장본 일반 진단은 `TOKENHSI_GPU=5 bash output_etc/check/carry5_latest_eval64_gpu5_20261102/run.sh`로 재현한다. PTH/YAML을 복사 고정하고4명·8물체·64환경×seed0·ordered·loco·평균 액션·distractor·1200step 첫 episode를 사용한다. 정식512환경×3seed 결과와 구분하며, [단독 점/영역·공동 공유 중심20cm/전원 개인 목표 Ever/Final](../output_etc/check/carry5_latest_eval64_gpu5_20261102/report.md)과 [검증](../output_etc/check/carry5_latest_eval64_gpu5_20261102/verification.json)을 보존한다.

carry5의1.5k·512환경×seed0/1/2 평가는 `TOKENHSI_GPU=5 bash output_etc/check/carry5_epoch1500_eval512_gpu5_20261102/run_worker.sh 0`으로 GPU5에서3회를 순차 실행한다. PTH/YAML을 복사 고정하고4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step을 유지한다. [평균±seed 표준편차·Ever/Final·분모](../output_etc/check/carry5_epoch1500_eval512_gpu5_20261102/report.md), [실행 설정](../output_etc/check/carry5_epoch1500_eval512_gpu5_20261102/manifest.json), [검증](../output_etc/check/carry5_epoch1500_eval512_gpu5_20261102/verification.json)을 보존하며 ordered 진단은 formal_report=false로 구분한다.

기존 solo5의 같은1.5k·512환경×3seed 추가 평가는 `TOKENHSI_GPU=5 bash output_etc/check/solo5_epoch1500_vs_carry5_eval512_gpu5_20261103/run_worker.sh 0`으로 재현한다. GPU5에서3회를 순차 실행하고 위 carry5 결과를 재사용했다. 두 모델의 저장된 목표 분포를 각각 유지하므로 같은 장면의 정책 비교는 아니다. [단독/공동 Ever·Final 비교](../output_etc/check/solo5_epoch1500_vs_carry5_eval512_gpu5_20261103/comparison.md), [BEFORE virtual과의 높이/과제 구성 차이](../output_etc/check/solo5_epoch1500_vs_carry5_eval512_gpu5_20261103/config_comparison.md), [검증](../output_etc/check/solo5_epoch1500_vs_carry5_eval512_gpu5_20261103/verification.json)을 보존한다.

solo5의20k·단독 AT 선반 제거 비교는 `TOKENHSI_GPU=5 bash output_etc/check/solo5_20k_at_ground64_gpu5_20261103/run.sh`로 재현한다. 같은 PTH·4명/8물체 혼합64환경×seed0·ordered·loco·평균 액션·1200step에서 기존 조건과AT만 지면에 놓은 조건을 순차 실행한다. 평가 프로세스에서 원래 높이 난수를 소비한 뒤 AT 원본 상자/목표만 지면 기준으로 낮추고 해당 플랫폼을Z−20m로 치운다. 출발 상자의5cm clearance·XY·물성·다른 과제·충돌/재시도는 유지하며 학습 YAML/표준 코드는 변경하지 않는다. [비교·정의](../output_etc/check/solo5_20k_at_ground64_gpu5_20261103/report.md), [초기 상태·RNG 검증](../output_etc/check/solo5_20k_at_ground64_gpu5_20261103/verification.json).

11000 PTH 감량/액션 노이즈 비교를 화면으로 볼 때는 `output_etc/check/solo5_mass_noise_n68_20261101/view.sh <인원> <baseline|mass|mass80|mass60|noise> [씬수]`를 사용한다. `mass`는 평균 액션을 사용하며 `MASS_SCALE`로 원래 질량에 대한 배수를 지정한다(기본0.8).0.8은20% 감량,0.6은40% 감량,1.0은원래 무게다. 크기·마찰은 유지하고 생성 단계의 질량·관성만 함께 조절한다. 양의 유한값만 허용하며 명시적인 `MASS_SCALE`이 mass 계열 preset보다 우선한다. 직접 Python 실행은 `--box_mass_scale`을 사용한다. 기존 mass80은0.8배, mass60은0.6배 preset이며, noise는 원래 질량+σ=exp(−2.9) Gaussian 샘플링이다. ordered·물체2N·기본1환경·seed0·씬당1200step이고 기본10씬을 순서대로 재생한다. 각 씬 안에서는 평가처럼 낙상한 사람의 제어를 끄고, 다음 씬 시작 전에 원래 관절 제어를 복구한다. 같은 모델/VNC를 유지하고 새 장면을 뽑으며 `episode_01` 등으로 산출물을 분리한다. `SEED`·`EVAL_ENVS`·`EVAL_STEPS`로 시작 장면·동시 환경 수·씬당 길이를 지정한다. 실행마다 확인 경로를 새로 만들며 기존 평가 산출물을 덮어쓰지 않는다. 1환경 장면은 배치 크기가 달라32환경 평가의 특정 장면과 같지는 않다. [VNC 명령](config.md#추론-옵션).

### Coupled 제로샷 viewer

Stage1 PUSH21k의 기존165 tensor를 그대로 복사하고 공통 Stage2 coordination을 identity로 붙인 관찰용 저장본이다. 추가 optimizer step은0이며, 학습 재개용 optimizer는 포함하지 않는다. 기본1200step이고 아래 명령은4명 모두 같은 상자를 미는 장면을 표시한다.

```bash
# stage2_coupled_push
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=all_coupled bash tokenhsi/scripts/multi_agent/stage2_coupled_push_test.sh output_etc/check/coupled_zeroshot/Stage2CoupledPush_ZeroShot_Stage1_21000.pth 4 1 8 10
```

독립 행동과2~4명 공동 그룹이 섞인 학습 분포를 보려면 `TASK_GRAPH=random_scenario`로 바꾼다. 전이 기록은 `output_etc/check/coupled_zeroshot/transfer.json`에 있다.

### BEFORE PUSH near50

`stage2_before_push_near50`은 `stage2_before_push`의 거리 변형이며 기본4명·1024환경·8토큰·1200step·loco다. `STAGE1_CHECKPOINT`로 Stage1 PUSH 전이 원본을 지정할 수 있고 viewer/평가는 near50 실험의 저장본을 사용한다. mix226 저장본과 혼용하지 않는다. near50의 정확한 거리 부등식은 [BEFORE PUSH](reference/stage2_before_push.md#near50-거리-변형), 실행 명령은 [config](config.md)를 따른다.

`stage2_before_push_near50_pair50_source5`은 near50의 쌍 생성 확률0.5·사람→상자 최대5m 변형이다. 단독 PUSH 접근은2–5m이며 이동은2–10m를 유지한다. AT/ON_TOP 목표 배치와 near50 조건은 유지한다. 기본21k 전이 경로가 없으면 `STAGE1_CHECKPOINT`를 지정한다. 현재 config의 학습 명령은 존재하는26k 파일을 명시하며, resume/viewer/평가는 이 실험의 저장본을 사용한다. [거리 상세](reference/stage2_before_push.md#쌍-선택50접근-최대5m-변형), [명령](config.md).

### 26k PUSH 평가와 Stage2 제로샷 관찰

사용자가 제공한 가중치는 `pth/Stage1V2Fixed_00078000.pth`(기존4과제), `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`(PUSH 포함)다.26k의 일반 Stage1 평가는 저장된 PUSH50% 분포를 복원하고900step을 사용한다. 현재 YAML의 새 학습 PUSH30%와 구분한다.

```bash
# stage1_push 일반 평가
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_push_eval.sh pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth --output output_etc/check/push_26k/general

# stage1_push 1명 행동별 단독 평가
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage1_push_single_eval.sh pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth --output output_etc/check/push_26k/single

# stage2_before_push_near50 제로샷 viewer
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/stage2_before_push_near50_test.sh output_etc/check/push_26k/zeroshot/stage2_before_push_near50/ZeroShot_Stage1Push_26000.pth 4 1 8 10

# stage2_coupled_push 제로샷 viewer
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=all_coupled bash tokenhsi/scripts/multi_agent/stage2_coupled_push_test.sh output_etc/check/push_26k/zeroshot/stage2_coupled_push/ZeroShot_Stage1Push_26000.pth 4 1 8 10
```

두 Stage2 저장본은26k의 기존165 tensor를 그대로 복사하고 coordination을 identity로 초기화했다. 추가 학습0회이며 전이 원본·계약·1200step 장면 설정을 각각 저장한다. near50에서 BEFORE ON_TOP 쌍만 보려면 `TASK_GRAPH=before_stack`, coupled에서 독립/2~4명 그룹 혼합을 보려면 `TASK_GRAPH=random_scenario`를 사용한다. 관찰용 저장본에는 optimizer가 없다. 이 문서의26k 검증/관찰 명령은 기존 YAML의 기본21k 전이 경로를 변경하지 않는다.

모든 설정의 PUSH·CARRY AT 목표구는 viewer/VNC/영상에서 반지름20cm(지름40cm)로 표시한다. 크기는 actor 생성 시 지정한다. 목표 좌표와 성공 거리 판정은 그대로이며 기존 실행 명령을 다시 실행하면 적용된다.

PUSH AT의 viewer/VNC에는 바닥 링(반지름35cm)·세로선·와이어 구(반지름20cm)를 선으로 덧그린다. 공동 PUSH 선은 노란색, 독립 PUSH 선은 담당자 색이다. 선 표시 함수는 actor scale·물리 상태·reset을 수정하지 않는다.
