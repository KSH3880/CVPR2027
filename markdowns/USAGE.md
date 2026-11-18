# 설치와 추가 실행 옵션

기본 명령은 [config](config.md), 구현 위치는 [structure](structure.md)를 따른다.

## 실행 환경

모든 wrapper는 `runtime_env.sh`로 기본 conda `tokenhsi`를 활성화하고 저장소 루트에서 실행한다. 다른 환경은 `TOKENHSI_CONDA_ENV=<이름>`, conda 설치 위치는 `CONDA_BASE`로 명시한다.

`TOKENHSI_GPU=0`처럼 물리 GPU 번호나 UUID를 한 번 지정한다. CUDA·기존 MPS·viewer Vulkan에 같은 GPU를 전달한다. 기존 MPS가 없으면 직접 CUDA로 실행한다. 실행 wrapper는 MPS daemon을 시작하거나 종료하지 않는다.

PyTorch는 서버 GPU/CUDA 호환 설치본을 사용하며 Isaac Gym은 torch보다 먼저 import한다. `requirements.txt`는 남은 의존성이다. 현재 서버의 conda에는 Isaac Gym과 CUDA 라이브러리 활성화 설정이 있다.

## 데이터와 체크포인트

`tokenhsi/data/dataset_loco_sit_carry_climb.yaml`과 `dataset_stage1_jointpush.yaml`은 같은 디렉터리 아래 모션 파일을 참조한다. 복사본에는 두 매니페스트가 요구하는 모션 121개를 두었다. 외부 데이터와 원본 checkout은 읽기 전용이다.

기본 전이용 Stage1 78k와 Distill900 1.5k·V4 2k는 `checkpoints/`에 있다. 기존 Distill900의 `output/stage2/` 경로에 파일이 있을 때는 그 파일을 우선한다. 공동 PUSH 기반 설정의 Stage1 26k는 `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`를 사용한다.

새 clone으로 파일을 전달하는 방식은 결정 대기 중이다. 기존 PUSH Stage2 전이용 Stage1 21k와 `stage2_distill`·`stage2_distill_900`의 teacher 세 개도 확보가 필요하다.

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

로컬 viewer는 화면 세션이 필요하다. VNC는 Xvfb·x11vnc·noVNC·websockify가 필요하며 `PORT`, `RESOLUTION`, `VNC_DIR`, `XVFB`, `X11VNC`, `NOVNC_DIR`, `WEBSOCKIFY`로 위치·화면을 지정한다. 현재 서버는 `~/opt/vnc/usr/bin/`·`~/opt/noVNC`·conda의 websockify를 사용한다. `stage2_distill_900_train_vnc.sh`는1024환경 학습을 기본6080, `stage2_distill_900_vnc.sh`는 테스트를6081에서 제공한다. VS Code에서 해당 포트를 전달하고 `http://localhost:6080/vnc.html?autoconnect=1&resize=remote` 또는6081을 연다. GUI 로그는 `output_etc/check/gui/`다. 종료 시 이번 wrapper가 띄운 VNC 서비스만 종료한다.

## 이전 이름의 저장본

기존 Stage1 78000과 정리 전 BEFORE 저장본은 그대로 읽을 수 있다. `checkpoint_compat.py`가 저장 설정의 reward 태그·RSI 경로·로컬 전이 링크만 현재 이름으로 변환한다. 모델 tensor와 보상 수치·실험별 계약 검사는 유지한다. 새 학습의 출력 경로와 실험 이름은 현재 컨피그 이름을 따른다.

## 정식 평가와 viewer 구분

`*_test.sh`/`*_vnc.sh`는 인원·RSI를 바꿔 관찰하는 viewer/진단이다. `HEADLESS=1`만 붙여도 정식 프로토콜로 바뀌지는 않는다. 일반 정식 평가는 `*_eval.sh`로 학습 인원을 유지하고, Stage1 전용1명 비교는 `stage1_single_eval.sh`로 실행한다. 시간은 정식 평가·viewer 모두 기본 Stage1 900step, Stage2 1200step이다. [실행 명령](config.md), [평가 규약](reference/evaluation.md)을 따른다.


## Stage1 PUSH

`stage1_push`는 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%의 사용자 지정 비율을 사용하는 스크래치 학습이다. 기본2명·2048환경·4물체·900step, 같은 `tokenhsi` 런타임을 사용한다. 주어진21k finetune PTH의 일반 평가는 저장된 PUSH50% 분포를 복원하고 시간만900으로 맞춘다. 옆에 `relation_config.yaml`이 없으면 PTH의 `relation_experiment_config`를 읽는다. 서로 다른 분포/horizon의 PTH를 resume하지 않는다.

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

ground의 `_relfusion_learnedbg`(A)·`_relfusion_zerobg`(B)는 Stage1 PUSH26k에서 새로 전이하는 별도 모델이다. 같은 COUPLED 64차원 임베딩을 bias와 projection 전 fusion에 공유하며 A만 SELF/NONE lookup을 학습한다. 기존 ground 및 A/B 사이 resume은 호환되지 않는다. 각 저장본은 해당 wrapper로 실행하며 `CHECK=1`로 짧은 학습을 검증한다. [설계·호환 범위](reference/stage2_coupled_push.md#ground-관계-fusion-ab), [학습·로컬·VNC 명령](config.md).

관계 fusion 검증 재현은 `TOKENHSI_GPU=2 bash output_etc/check/relfusion_ab_20261103/run_train.sh learnedbg`와 같은 명령의 `zerobg`다. 각각4명·1024환경·8물체·MAX_ITERATIONS=2다. 이후 `TOKENHSI_GPU=2 bash output_etc/check/relfusion_ab_20261103/run_policy.sh learnedbg` 및 `zerobg`로32환경8preset 재로드와8환경16step test wrapper를 검사한다. 저장 결과·실제 업데이트·검증 범위는 [verification.json](../output_etc/check/relfusion_ab_20261103/verification.json)에 있다.

zerobg500의 일반64환경×seed0 평가는 `TOKENHSI_GPU=2 bash output_etc/check/relfusion_zerobg500_eval64_gpu2_20261103/run.sh`로 재현한다.4명·8물체 혼합·ordered·loco·평균 액션·1200step이며 기존 solo5·CO norm·CO+NONE norm500의 결과는 재사용한다. runtime 적용 후 `PYTHONPATH=tokenhsi python output_etc/check/relfusion_zerobg500_eval64_gpu2_20261103/analyze.py`로 집계한다. [비교표·조건 차이](../output_etc/check/relfusion_zerobg500_eval64_gpu2_20261103/report.md), [검증](../output_etc/check/relfusion_zerobg500_eval64_gpu2_20261103/verification.json).

관계 fusion zerobg1k·learnedbg0.5k의 일반64환경×seed0 평가는 `TOKENHSI_GPU=3 bash output_etc/check/relfusion_ab_trend64_500_1000_20261103/run.sh zerobg_1000` 및 같은 명령의 `learnedbg_500`으로 재현한다.4명·8물체 혼합·ordered·loco·평균 액션·1200step이며 zerobg0.5k 결과를 재사용한다. runtime 적용 후 `PYTHONPATH=tokenhsi python output_etc/check/relfusion_ab_trend64_500_1000_20261103/analyze.py`로 집계한다. [비교표](../output_etc/check/relfusion_ab_trend64_500_1000_20261103/report.md), [검증](../output_etc/check/relfusion_ab_trend64_500_1000_20261103/verification.json).

ground9k의512환경×seed0/1/2 평가는 `TOKENHSI_GPU=3 bash output_etc/check/ground_latest9k_eval512_gpu3_20261103/run_worker.sh 0` 및 같은 명령의 `1`·`2`로 재현한다. GPU3 최대3개 worker이며4명·8물체 혼합·ordered·loco·평균 액션·1200step,5k/6k/7k 결과를 재사용한다. runtime 적용 후 `PYTHONPATH=tokenhsi python output_etc/check/ground_latest9k_eval512_gpu3_20261103/analyze.py`와 같은 폴더의 `report.py`로 집계한다. [추세표](../output_etc/check/ground_latest9k_eval512_gpu3_20261103/report.md), [검증](../output_etc/check/ground_latest9k_eval512_gpu3_20261103/verification.json).

ground9.5k의512환경×seed0/1/2 평가는 `TOKENHSI_GPU=3 bash output_etc/check/ground_latest9500_eval512_gpu3_20261103/run_worker.sh 0` 및 같은 명령의 `1`·`2`로 재현한다. 위와 같은 조건이며5k/6k/7k/9k 결과를 재사용한다. runtime 적용 후 같은 폴더의 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로 실행하고 `report.py`를 실행한다. [추세표](../output_etc/check/ground_latest9500_eval512_gpu3_20261103/report.md), [검증](../output_etc/check/ground_latest9500_eval512_gpu3_20261103/verification.json).

관계 fusion zerobg1.5k·learnedbg1k의64환경×seed0 평가는 `TOKENHSI_GPU=2 bash output_etc/check/relfusion_ab_trend64_1000_1500_20261103/run.sh zerobg_1500` 및 같은 명령의 `learnedbg_1000`으로 재현한다.4명·8물체 혼합·ordered·loco·평균 액션·1200step이며 이전 fusion3회를 재사용한다. runtime 적용 후 `PYTHONPATH=tokenhsi python output_etc/check/relfusion_ab_trend64_1000_1500_20261103/analyze.py`로 집계한다. [비교표](../output_etc/check/relfusion_ab_trend64_1000_1500_20261103/report.md), [검증](../output_etc/check/relfusion_ab_trend64_1000_1500_20261103/verification.json).

zerobg2k의 동일64환경×seed0 평가는 `TOKENHSI_GPU=3 bash output_etc/check/relfusion_zerobg2000_eval64_gpu3_20261103/run.sh zerobg_2000`으로 재현한다. runtime 적용 후 `PYTHONPATH=tokenhsi python output_etc/check/relfusion_zerobg2000_eval64_gpu3_20261103/analyze.py`로 이전0.5k/1k/1.5k를 재사용해 집계한다. [추세표](../output_etc/check/relfusion_zerobg2000_eval64_gpu3_20261103/report.md), [검증](../output_etc/check/relfusion_zerobg2000_eval64_gpu3_20261103/verification.json).

learnedbg1.5k의 동일64환경×seed0 평가는 `TOKENHSI_GPU=3 bash output_etc/check/relfusion_learnedbg1500_ab_eval64_gpu3_20261103/run.sh learnedbg_1500`으로 재현한다. runtime 적용 후 `PYTHONPATH=tokenhsi python output_etc/check/relfusion_learnedbg1500_ab_eval64_gpu3_20261103/analyze.py`로 기존 learnedbg0.5k/1k·zerobg0.5k/1k/1.5k/2k를 재사용해 비교한다. [추세표](../output_etc/check/relfusion_learnedbg1500_ab_eval64_gpu3_20261103/report.md), [검증](../output_etc/check/relfusion_learnedbg1500_ab_eval64_gpu3_20261103/verification.json).

ground8k·10k의각512환경×seed0/1/2 평가는 `TOKENHSI_GPU=3 bash output_etc/check/ground_8k10k_eval512_gpu3_20261103/run_worker.sh 0` 및 같은 명령의 `1`·`2`로 재현한다. 최대3개 worker가각8k→10k를 순차 실행하며4명·8물체 혼합·ordered·loco·평균 액션·1200step이다. runtime 적용 후 `PYTHONPATH=tokenhsi python output_etc/check/ground_8k10k_eval512_gpu3_20261103/analyze.py`와 같은 폴더의 `report.py`로 기존5k/6k/7k/9k/9.5k를 재사용해 비교한다. [추세표](../output_etc/check/ground_8k10k_eval512_gpu3_20261103/report.md), [검증](../output_etc/check/ground_8k10k_eval512_gpu3_20261103/verification.json).

ground11k·11.5k의각512환경×seed0/1/2 평가는 `TOKENHSI_GPU=2 bash output_etc/check/ground_11k11500_eval512_gpu2_20261104/run_worker.sh 0` 및 같은 명령의 `1`·`2`로 재현한다. 최대3개 worker가각11k→11.5k를 순차 실행한다. runtime 적용 후 같은 폴더의 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로 실행하고 `report.py`로 기존5k–10k와 비교한다. [추세표](../output_etc/check/ground_11k11500_eval512_gpu2_20261104/report.md), [검증](../output_etc/check/ground_11k11500_eval512_gpu2_20261104/verification.json).

zerobg2.5k/3k/3.5k·learnedbg2k/2.5k의각64환경×seed0 평가는 `TOKENHSI_GPU=5 bash output_etc/check/relfusion_ab_2k3500_eval64_gpu5_20261104/run_worker.sh 0` 및 같은 명령의 `1`로 재현한다. 두 worker는각zero3개·learn2개를 순차 실행한다. runtime 적용 후 `PYTHONPATH=tokenhsi python output_etc/check/relfusion_ab_2k3500_eval64_gpu5_20261104/analyze.py`로 이전7회를 재사용해 비교한다. [추세표](../output_etc/check/relfusion_ab_2k3500_eval64_gpu5_20261104/report.md), [검증](../output_etc/check/relfusion_ab_2k3500_eval64_gpu5_20261104/verification.json). 두 평가 모두4명·8물체 혼합·ordered·loco·평균 액션·1200step이며GPU0·1은사용하지않는다.

ground12k/14k/16k의각512환경×seed0/1/2 평가는 `TOKENHSI_GPU=2 bash output_etc/check/ground_12k14k16k_eval512_gpu23_20261104/run_worker.sh 0`과worker1/2, `TOKENHSI_GPU=3`의같은명령worker3/4/5로재현한다. GPU당최대3개이며2에서는12k→16k,3에서는14k를실행한다. runtime 적용후같은폴더의 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로실행하고 `report.py`로집계한다. [추세표](../output_etc/check/ground_12k14k16k_eval512_gpu23_20261104/report.md).

zerobg4k/5k/6k/8k·learnedbg3k/4k/5k의각64환경×seed0 평가는 `TOKENHSI_GPU=5 bash output_etc/check/relfusion_ab_3k8k_eval64_gpu5_20261104/run_worker.sh 0` 및같은명령의 `1`로재현한다. runtime 적용후 `PYTHONPATH=tokenhsi python output_etc/check/relfusion_ab_3k8k_eval64_gpu5_20261104/analyze.py`로이전12회를재사용해비교한다. 모두4명·8물체혼합·ordered·loco·평균액션·1200step이다. [추세표](../output_etc/check/relfusion_ab_3k8k_eval64_gpu5_20261104/report.md).

선택 ground9.5k·16k·zerobg3k의6/8명 all/keep3 비교는 `TOKENHSI_GPU=3 bash output_etc/check/ground_relfusion_selected_n68_all_keep3_env64_gpu35_20261104/run_worker.sh 0` 및같은GPU의worker1, GPU5의worker2/3, GPU2의worker4/5로재현한다. 각각ground9.5k·zero3k·ground16k이며짝수worker는all,홀수는keep3로6명→8명을순차평가한다.각64환경×seed0·ordered·원래물성·평균액션·1200step이며keep3는선택밖COUPLED를NONE으로전환한다. runtime 적용후같은폴더의 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로집계한다. [비교표](../output_etc/check/ground_relfusion_selected_n68_all_keep3_env64_gpu35_20261104/report.md), [검증](../output_etc/check/ground_relfusion_selected_n68_all_keep3_env64_gpu35_20261104/verification.json). 실제GPU는폴더이름과달리2·3·5이며GPU0·1은사용하지않는다.



learnedbg5k의 6/8명 all/keep3 비교는 `TOKENHSI_GPU=3 bash output_etc/check/learnedbg5000_n68_all_keep3_env64_gpu35_20261104/run_worker.sh 0`(all), `TOKENHSI_GPU=5 bash output_etc/check/learnedbg5000_n68_all_keep3_env64_gpu35_20261104/run_worker.sh 1`(keep3)로 재현한다. 각 worker가 6명→8명을 64환경×seed0·ordered·원래 물성·평균 액션·1200step으로 평가한다. runtime 적용 후 같은 폴더의 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로 집계한다. [비교표](../output_etc/check/learnedbg5000_n68_all_keep3_env64_gpu35_20261104/report.md), [검증](../output_etc/check/learnedbg5000_n68_all_keep3_env64_gpu35_20261104/verification.json).

ground/zerobg10k의 최초 전원 접촉 전후 이동은 기존512환경×seed0/1/2 궤적을 `/home/hwanhee/anaconda3/envs/tokenhsi/bin/python output_etc/check/ground_zerobg10k_gate_timing_20261104/analyze.py`로 CPU 재집계한다. 공동2/3/4명 그룹별로 최초 전원 양손면 접촉 전 초기 상자XY변위50cm, 전원접촉 달성, 접촉 후 추가50cm를 측정한다. 목표도착 예외가 있는 native gate 자체를 준비 기준으로 쓰지 않는다. [비교표·정의](../output_etc/check/ground_zerobg10k_gate_timing_20261104/report.md), [검증](../output_etc/check/ground_zerobg10k_gate_timing_20261104/verification.json). 추가 GPU 평가 없이 재현한다.

zerobg10k의 일반512환경×seed0/1/2 평가는 `TOKENHSI_GPU=4 bash output_etc/check/zerobg10000_trend512_gpu4_20261104/run_worker.sh 0` 및 worker1/2로 재현한다. 4명·8물체 혼합·ordered·loco·평균 액션·원래 물성·1200step을 유지한다. runtime 적용 후 같은 폴더의 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로 실행하고 `report.py`로 기존 zerobg8k·9k와 ground10k를 비교한다. [비교표](../output_etc/check/zerobg10000_trend512_gpu4_20261104/report.md), [검증](../output_etc/check/zerobg10000_trend512_gpu4_20261104/verification.json).

같은zerobg10k 초기장면의distractor 제거 비교는 `TOKENHSI_GPU=3 bash output_etc/check/zerobg10k_nodistractors512_20261104/run_worker.sh 0` 및GPU3/worker1·GPU4/worker2로 실행한다. 각512환경×seed0/1/2이며 불필요한 상자의 물리 장면과 관측 토큰만 제외하고ON_TOP 받침·작업 상자·사람 스폰·나머지 평가조건을 유지한다. runtime 적용 후 `PYTHONPATH=tokenhsi python output_etc/check/zerobg10k_nodistractors512_20261104/analyze.py`로 기존결과와집계·초기배치보존검증을수행한다. [Ever 영역 비교표](../output_etc/check/zerobg10k_nodistractors512_20261104/report.md), [검증](../output_etc/check/zerobg10k_nodistractors512_20261104/verification.json).

5m 그룹 중심 간격 배치 검증은 `TOKENHSI_GPU=3 bash output_etc/check/group_spawn_gap5_layout64_20261104/run.sh 8`과 `TOKENHSI_GPU=4 bash output_etc/check/group_spawn_gap5_layout64_20261104/run.sh 16`으로 재현한다. 각64환경·seed0이며 기존사람배치후그룹전체XY평행이동·공동최대4·불필요한distractor물리제거·필수받침유지·기존거리/충돌검사를사용한다. 정책rollout과관측토큰제외검사는실행하지않는다. [배치성공/재시도](../output_etc/check/group_spawn_gap5_layout64_20261104/report.md), [검증](../output_etc/check/group_spawn_gap5_layout64_20261104/verification.json).

같은배치의zerobg10k VNC viewer는 `TOKENHSI_GPU=3 PORT=6080 HEADLESS=0 bash output_etc/check/group_spawn_gap5_layout64_20261104/view.sh 8 10`으로실행한다. 인자는 `[사람수] [씬수]`이며16명은첫인자를16으로바꾼다. 1환경·혼합·공동최대4·ordered·loco·평균액션·씬당1200step이며4명이하는기존배치를유지한다. 4명초과에만5m그룹이동과불필요한distractor물리/관측제외를적용하고카메라는실제작업영역을본다. `CHECKPOINT`로PTH변경가능하다. 확인실행은 `TOKENHSI_GPU=3 HEADLESS=1 EPISODE_LENGTH=12 VERIFY_VIEWER=1 bash output_etc/check/group_spawn_gap5_layout64_20261104/view.sh 8 2`로재현한다. [4/8/16명2씬검증](../output_etc/check/group_spawn_gap5_layout64_20261104/viewer_checks.json).

Stage1 26k의같은5m 혼합확장평가는 `TOKENHSI_GPU=3 bash output_etc/check/stage1_26k_group_gap5_mixed512_20261104/run_worker.sh 6`으로실행한다. 나머지는GPU4/인원8·GPU5/인원12·GPU6/인원16을지정하며각worker가512환경×seed0/1/2를순차평가한다. 원래Stage1 encoder/head/RMS와merge_shared=False를유지하고Stage2 coordination을우회한다. runtime 적용후같은폴더 `analyze.py`를실행해완료인원을집계한다. [샘플링명세](reference/group_spawn_over4.md), [진행/비교표](../output_etc/check/stage1_26k_group_gap5_mixed512_20261104/report.md), [설정](../output_etc/check/stage1_26k_group_gap5_mixed512_20261104/manifest.json).

zerobg10k의 같은 5m 혼합 확장 평가는 `TOKENHSI_GPU=3 bash output_etc/check/zerobg10k_group_gap5_mixed512_20261104/run_worker.sh 6`으로 실행한다. 나머지는 GPU4/인원8·GPU5/인원12·GPU6/인원16이며 각512환경×seed0/1/2다. Stage2 coordination을 정상 실행하고 같은 인원·seed의 Stage1 초기 장면과 일치를 검사한다. runtime 적용 후 `PYTHONPATH=tokenhsi python output_etc/check/zerobg10k_group_gap5_mixed512_20261104/analyze.py`로 기존 행동별·공동 인원별 평균과 유형별 I/C/Mixed SR을 함께 집계한다. Stage1은 기존 결과를 CPU 재집계하며 Mixed=(I+C)/2다. [두 집계·모델 비교](../output_etc/check/zerobg10k_group_gap5_mixed512_20261104/report.md), [검증](../output_etc/check/zerobg10k_group_gap5_mixed512_20261104/verification.json).

zerobg11k의기존4명512환경×3회평가는 `TOKENHSI_GPU=4 bash output_etc/check/zerobg11000_trend512_gpu4_20261104/run_worker.sh 0` 및worker1/2로재현한다. 기존사람스폰/distractor·ordered·loco·평균액션·1200step을유지하며runtime 적용후같은폴더 `analyze.py`, `report.py`로기존8k/9k/10k와비교한다. [Ever영역비교표](../output_etc/check/zerobg11000_trend512_gpu4_20261104/report.md), [검증](../output_etc/check/zerobg11000_trend512_gpu4_20261104/verification.json).

zerobg12k의동일4명512환경×3회평가는 `TOKENHSI_GPU=5 bash output_etc/check/zerobg12000_trend512_gpu5_20261104/run_worker.sh 0` 및worker1/2로재현한다. 기존스폰/distractor·ordered·loco·평균액션·원래물성·1200step을유지하고12k만새로평가하며10k/11k 결과를재사용한다. runtime 적용후같은폴더 `analyze.py`, `report.py`로초기배치일치/CPU검증과비교표를생성한다. [10k·11k·12k Ever영역비교표](../output_etc/check/zerobg12000_trend512_gpu5_20261104/report.md), [검증](../output_etc/check/zerobg12000_trend512_gpu5_20261104/verification.json).

zerobg13k의 같은4명512환경×seed0/1/2 평가는 `TOKENHSI_GPU=5 bash output_etc/check/zerobg13000_trend512_gpu5_20261104/run_worker.sh 0` 및 worker1/2로 재현한다. GPU5 최대3개이며 기존 스폰/distractor·ordered·loco·평균 액션·거리/물성·1200step을 유지한다. runtime 적용 후 같은 폴더의 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로 실행하고 `report.py`로10k·11k·12k 기존 결과와 비교한다. 공동 그룹 합산 SR·단독 PUSH·나머지4행동 영역 평균을 주로 보고 I/C/Mixed SR과 기존 동일 가중 평균도 보존한다. [10k–13k 비교](../output_etc/check/zerobg13000_trend512_gpu5_20261104/report.md), [검증](../output_etc/check/zerobg13000_trend512_gpu5_20261104/verification.json).

zerobg14k의 같은4명512환경×seed0/1/2 평가는 `TOKENHSI_GPU=5 bash output_etc/check/zerobg14000_trend512_gpu5_20261104/run_worker.sh 0` 및 worker1/2로 재현한다. 기존 스폰/distractor·ordered·loco·평균 액션·거리/물성·1200step을 유지하며 GPU5 최대3개다. runtime 적용 후 같은 폴더 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로 실행하고 `report.py`로10k–13k 기존 결과와 공동 그룹 합산 SR·단독 PUSH·나머지4행동 평균 및 I/C/Mixed SR을 비교한다. [10k–14k 비교](../output_etc/check/zerobg14000_trend512_gpu5_20261104/report.md), [검증](../output_etc/check/zerobg14000_trend512_gpu5_20261104/verification.json).

zerobg15k의 같은4명512환경×seed0/1/2 평가는 `TOKENHSI_GPU=5 bash output_etc/check/zerobg15000_trend512_gpu5_20261105/run_worker.sh 0` 및 worker1/2로 재현한다. GPU5 최대3개로 기존 스폰/distractor·ordered·loco·평균 액션·거리/물성·1200step을 유지한다. runtime 적용 후 같은 폴더 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로 실행하고 `report.py`로10k–14k 기존 결과와 공동 그룹 합산 SR·단독 PUSH·나머지4행동 평균 및 I/C/Mixed SR을 비교한다. [10k–15k 비교](../output_etc/check/zerobg15000_trend512_gpu5_20261105/report.md), [검증](../output_etc/check/zerobg15000_trend512_gpu5_20261105/verification.json).

요청 시점 최신 zerobg19.5k의 같은4명512환경×seed0/1/2 평가는 `TOKENHSI_GPU=5 bash output_etc/check/zerobg19500_trend512_gpu5_20261105/run_worker.sh 0` 및 worker1/2로 재현한다. 일반 이름 PTH도 epoch19500임을 확인한 뒤 정기 PTH/YAML을 복사 고정했다. 기존 스폰/distractor·ordered·loco·평균 액션·거리/물성·1200step과 GPU5 최대3개를 유지한다. runtime 적용 후 같은 폴더 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로 실행하고 `report.py`로10k–15k 기존 결과와 비교한다. [SR·단독 PUSH·나머지 평균](../output_etc/check/zerobg19500_trend512_gpu5_20261105/report.md), [검증](../output_etc/check/zerobg19500_trend512_gpu5_20261105/verification.json).

zerobg16k–20k의1k 간격 정밀 평가는 `TOKENHSI_GPU=3 bash output_etc/check/zerobg16k20k_eval512_gpu345_20261105/run_worker.sh 0` 및 GPU3/worker1·2(각 seed의16k→19k), GPU4/worker3·4·5(17k→20k), GPU5/worker6·7·8(18k)로 재현한다. 각4명·8물체 혼합512환경×seed0/1/2·ordered·loco·평균 액션·기존 스폰/distractor·거리/물성·1200step이며 GPU당 최대3개다. runtime 적용 후 같은 폴더의 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로 실행하고 `report.py`로 기존10k–15k·19.5k와 비교한다. 공동 그룹 합산 SR을 우선하고 공동4명·단독 PUSH·나머지4행동 평균·I/C/Mixed를 함께 본다. [최종 후보 비교](../output_etc/check/zerobg16k20k_eval512_gpu345_20261105/report.md), [검증](../output_etc/check/zerobg16k20k_eval512_gpu345_20261105/verification.json).

유형별Interaction-Group SR 대안집계는 `python output_etc/check/zerobg10k12k_interaction_group_macro_20261104/aggregate.py`로CPU재현한다. I는단독개인전체합산, C는공동2·3·4명그룹전체합산, Mixed=(I+C)/2이며seed별비율의3seed평균±표준편차다. 기존행동별/공동인원별macro집계와별도이며시뮬레이션을추가하지않는다. [정의·비교표](../output_etc/check/zerobg10k12k_interaction_group_macro_20261104/report.md).

zerobg10k의 혼합 환경 인원 확장은 `TOKENHSI_GPU=2 bash output_etc/check/zerobg10k_mixed_cap4_scaling512_20261104/run_worker.sh 0`(6명), GPU5/worker1(8명), GPU4/worker2(12명), GPU3/worker3(16명)으로 실행한다. 각512환경×seed0/1/2·물체2N개이며 공동 그룹만2~min(4,남은 인원)으로 제한한다. 나머지 샘플링/거리/충돌/물성·ordered·loco·1200step은 유지한다. 8/12/16명은 배치64회 소진 후512회로 재시도하며 실패 로그를 보존한다. 같은 폴더의 `analyze.py`를 tokenhsi Python으로 실행해 [진행 상태](../output_etc/check/zerobg10k_mixed_cap4_scaling512_20261104/status.json)와 [비교표](../output_etc/check/zerobg10k_mixed_cap4_scaling512_20261104/report.md)를 갱신한다. 공동 평균은2/3/4명 동일 가중, 단독은5행동 동일 가중, 혼합은두 평균의50:50이다.

Stage1 PUSH26k를 현재4명 혼합 조건에서 비교하려면 `TOKENHSI_GPU=2 bash output_etc/check/stage1_26k_current_mixed4_512_gpu2_20261104/run_worker.sh`를 실행한다. Stage1 원래 encoder/action head/정규화·merge_shared=False를 복원하고 Stage2 coordination을 실행하지 않는다. 현재 zerobg10k 환경·4명/8물체·512환경×seed0/1/2·ordered·loco·1200step을 적용하며 Stage1 원래900step 일반 평가와 구분한다. 같은 폴더의 `analyze.py`를 tokenhsi Python으로 실행해 원래 actor 출력·기존 zerobg와 초기12필드 일치·CPU 재채점·원본 PTH 해시를 검증하고 [비교표](../output_etc/check/stage1_26k_current_mixed4_512_gpu2_20261104/report.md)를 갱신한다.

12/16명 사람 반경만2.5×sqrt(N/4)로 확대한 시험은 [별도 기록](../output_etc/check/zerobg10k_mixed_cap4_spawnscale512_20261104/report.md)에 보존한다. 첫512환경 배치 정체로중단해 성능수치는없다. `check_human_geometry.py`를 runtime적용후GPU3·`PYTHONPATH=tokenhsi python`으로 실행하면 동일그룹/동일후보의기존/확대/인접배치제안 기하검사를 재현하며 `analyze_trial.py`는CPU에서설정·쌍별그룹/행동/크기·PTH해시를 검증한다. 기하검사는 전체장면/정책 평가와 구분한다.

사용자 승인한12/16명 확대+공동그룹인접스폰 평가는 `TOKENHSI_GPU=4 bash output_etc/check/zerobg10k_mixed_cap4_anchored_grouped512_20261104/run_worker.sh 0`(12명), `TOKENHSI_GPU=3 bash output_etc/check/zerobg10k_mixed_cap4_anchored_grouped512_20261104/run_worker.sh 1`(16명seed0), GPU2/worker2(16명seed1), GPU5/worker3(16명seed2)로 실행한다. 각512환경×seed0/1/2·물체2N개·ordered·loco·1200step이며 같은그룹만원 위 연속슬롯에실제rootXY를정확히놓는다. RSI모션의기존XY변위를제거하되자세/높이/속도는유지한다. 사람반경4.33/5m외거리/물성/distractor반경4.5m/충돌판정을 유지한다. root/kinematic bodyXY동시이동·자세/속도/RNG/그룹 불변과 슬롯연속을 검증하고 `spawn_layout.pt`·`layout_progress.json`을 저장한다. 같은폴더 `analyze.py`를 tokenhsi Python으로 실행해 [진행](../output_etc/check/zerobg10k_mixed_cap4_anchored_grouped512_20261104/status.json)·[검증](../output_etc/check/zerobg10k_mixed_cap4_anchored_grouped512_20261104/verification.json)·[비교표](../output_etc/check/zerobg10k_mixed_cap4_anchored_grouped512_20261104/report.md)를 갱신한다. 기존4/6/8명은사람반경2.5m·기존자리배정이므로배치차이를명시한다.

ground17k의 일반512환경×seed0/1/2 평가는 `TOKENHSI_GPU=3 bash output_etc/check/ground17000_trend512_gpu3_20261104/run_worker.sh 0` 및 worker1/2로 재현한다. 같은 폴더의 `analyze.py`를 runtime 적용 후 `PYTHONPATH=tokenhsi python`으로 실행하고 `report.py`로 기존 ground14k·16k와 zerobg8k·9k를 비교한다. [추세표](../output_etc/check/ground17000_trend512_gpu3_20261104/report.md), [검증](../output_etc/check/ground17000_trend512_gpu3_20261104/verification.json).

ground18k·19k·20k의 일반512환경×seed0/1/2 평가는 `TOKENHSI_GPU=3 bash output_etc/check/ground18k19k20k_eval512_gpu345_20261104/run_worker.sh 0` 및 GPU3/worker1·2(18k), GPU4/worker3·4·5(19k), GPU5/worker6·7·8(20k)로 재현한다. GPU당 최대3개이며 기존4명·8물체 혼합·ordered·loco·평균 액션·스폰/distractor·거리/물성·1200step을 유지한다. runtime 적용 후 같은 폴더의 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로 실행하고 `report.py`로 ground9.5k·16k·17k와 zerobg10k를 비교한다. Independent는 단독 개인, Coupled는 공동 그룹 단위 합산이며 Mixed=(I+C)/2다. 기존 행동별·공동 인원별 동일 가중 평균도 별도로 보존한다. [SR·행동별·추세 비교](../output_etc/check/ground18k19k20k_eval512_gpu345_20261104/report.md), [검증](../output_etc/check/ground18k19k20k_eval512_gpu345_20261104/verification.json).

zerobg8k·9k의 일반 평가는 각512환경×seed0/1/2로 실행한다. `TOKENHSI_GPU=4 bash output_etc/check/zerobg_trend512_gpu34_20261104/run_worker.sh 0` 및 worker1/2가8k, `TOKENHSI_GPU=3`의 같은 명령 worker3/4/5가9k다. GPU당 최대3개, 4명·8물체 혼합·ordered·loco·평균 액션·원래 물성·1200step이다. runtime 적용 후 같은 폴더의 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로 실행하고 `report.py`로 기존 ground8k·9k와 비교한다. [비교표](../output_etc/check/zerobg_trend512_gpu34_20261104/report.md), [검증](../output_etc/check/zerobg_trend512_gpu34_20261104/verification.json).

zerobg9k의6/8명 all/keep3·64환경×seed0 평가는 `TOKENHSI_GPU=4 bash output_etc/check/zerobg9000_n68_all_keep3_radius_env64_gpu45_20261104/run_worker.sh 0`(all), `TOKENHSI_GPU=5`의같은명령 worker1(keep3)로 실행한다. ordered·원래 물성·평균 액션·1200step을 유지한다. runtime 적용 후 같은 폴더의 `analyze.py`, `rescore_radius.py`를 각각 `PYTHONPATH=tokenhsi python`으로 실행하면 같은 궤적에서 공유 중심20/40cm Ever/Final을 비교한다. [반경 비교표](../output_etc/check/zerobg9000_n68_all_keep3_radius_env64_gpu45_20261104/radius_report.md), [검증](../output_etc/check/zerobg9000_n68_all_keep3_radius_env64_gpu45_20261104/verification.json).

zerobg13k의6/8명 전원 공동 all/keep3 비교는 `TOKENHSI_GPU=3 bash output_etc/check/zerobg13000_n68_all_keep3_env64_gpu35_20261104/run_worker.sh 0` 및 GPU3/worker1(6명 keep3), GPU5/worker2·3(8명 all·keep3)로 재현한다. 각64환경×seed0·ordered·loco·원래 인원별 상자 크기/질량·스폰/distractor·평균 액션·1200step이며, 그룹 최대4명 혼합 확장과 별개다. keep3는 미접촉 상대를 우선해3명을 유지하고 선택 밖 COUPLED를NONE으로 바꾸며 critic 관계와 전체 그룹 gate는 유지한다. runtime 적용 후 같은 폴더의 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로 실행한다. [Ever/Final·접촉·이동](../output_etc/check/zerobg13000_n68_all_keep3_env64_gpu35_20261104/report.md), [검증](../output_etc/check/zerobg13000_n68_all_keep3_env64_gpu35_20261104/verification.json).

위13k 결과의 공유 중심20/40cm 비교는 `PYTHONPATH=tokenhsi /home/hwanhee/anaconda3/envs/tokenhsi/bin/python output_etc/check/zerobg13000_n68_all_keep3_env64_gpu35_20261104/rescore_radius.py`로 CPU 재현한다. 기존 궤적의3D 거리·그룹 유효성 규칙을 유지하며 추가 시뮬레이션은 없다. [반경 비교표](../output_etc/check/zerobg13000_n68_all_keep3_env64_gpu35_20261104/radius_report.md), [반경 검증](../output_etc/check/zerobg13000_n68_all_keep3_env64_gpu35_20261104/radius_verification.json).

zerobg17k의 같은6/8명 전원 공동 all/keep3 비교는 `TOKENHSI_GPU=3 bash output_etc/check/zerobg17000_n68_all_keep3_env64_gpu35_20261105/run_worker.sh 0` 및 GPU3/worker1(6명 keep3), GPU5/worker2·3(8명 all·keep3)로 재현한다. 각64환경×seed0·ordered·loco·원래 인원별 크기/질량·스폰/distractor·평균 액션·1200step이며 keep3는 선택 밖 COUPLED를NONE으로 전환한다. runtime 적용 후 같은 폴더 `analyze.py`를 `PYTHONPATH=tokenhsi python`으로 실행하고 `rescore_radius.py`로 같은 궤적의 공유 중심20/40cm를 CPU 재집계한다. [20cm·접촉·이동](../output_etc/check/zerobg17000_n68_all_keep3_env64_gpu35_20261105/report.md), [20/40cm 비교](../output_etc/check/zerobg17000_n68_all_keep3_env64_gpu35_20261105/radius_report.md), [검증](../output_etc/check/zerobg17000_n68_all_keep3_env64_gpu35_20261105/verification.json).

zerobg8k의 같은 비교는 `TOKENHSI_GPU=3 bash output_etc/check/zerobg8000_n68_all_keep3_env64_gpu35_20261104/run_worker.sh 0`(all), `TOKENHSI_GPU=5 bash output_etc/check/zerobg8000_n68_all_keep3_env64_gpu35_20261104/run_worker.sh 1`(keep3)로 재현한다. 같은 폴더의 `analyze.py`로 검증·집계하고 `compare.py`로 learnbg5k와 기존 zerobg3k를 함께 정리한다. [통합 비교표](../output_etc/check/zerobg8000_n68_all_keep3_env64_gpu35_20261104/comparison.md), [검증](../output_etc/check/zerobg8000_n68_all_keep3_env64_gpu35_20261104/verification.json).

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

## 통합 Stage2

`stage2`는 4명·1024환경·8토큰의 RL 학습 설정이다. 실행 명령은 [config](config.md), 관계별 샘플링·물체·보상은 [통합 명세](reference/stage2.md)를 따른다. `TASK_GRAPH=before`, `coupled`, `mixed`로 viewer 장면을 선택할 수 있다. 단일 통합 모델의 정식 평가는 `TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_eval.sh <checkpoint.pth> --output output_etc/check/stage2_eval`로 실행한다. 통합 viewer/eval은4명도 ordered가 기본이다. 별도 teacher 모방 학습은 `stage2_distill`이다.

### 두 teacher 제로샷 viewer

두 PTH를 각각 복원하고 enable 선행/후행·solo는 enable, 공동 PUSH 그룹원은 coupled의 평균 행동으로 제어한다. 두 teacher 모두 장면 전체를 관측하며 추가 학습이나 PTH 병합은 없다. 아래 두 경로는 이 checkout에 연결한 enable 13.5k/coupled 25.5k 정기 저장본이다. 다른 저장본은 해당 변수만 바꾼다. 원본 파일은 수정하지 않는다.

```bash
ENABLE_PTH=pth/stage2_teacher_enable_13500.pth
COUPLED_PTH=pth/stage2_teacher_coupled_25500.pth

# stage2: 두 teacher, 8명·16토큰·두 관계 포함·서버 VNC
TOKENHSI_GPU=6 PORT=6081 TASK_GRAPH=mixed bash tokenhsi/scripts/multi_agent/stage2_teachers_vnc.sh "$ENABLE_PTH" "$COUPLED_PTH" 8 1 16 10

# stage2: 같은 장면 구성의 로컬 viewer
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=mixed bash tokenhsi/scripts/multi_agent/stage2_teachers_test.sh "$ENABLE_PTH" "$COUPLED_PTH" 8 1 16 10

# stage2: 8명 혼합 확장 공간 채점, seed0·512환경·1200step
TOKENHSI_GPU=5 SEED=0 OUTPUT_PATH=output_etc/check/stage2_teachers_eval8_seed0 bash tokenhsi/scripts/multi_agent/stage2_teachers_eval.sh "$ENABLE_PTH" "$COUPLED_PTH" 8 512 16
```

인자는 `<enable.pth> <coupled.pth> [agents=4] [envs=1] [objects=2N] [repeats=10]`다. eval wrapper는 envs 기본512·반복1이며 `SEED=0/1/2`와 출력 폴더를 바꿔 독립 실행한다. GPU는5/6만 허용하고 생략하면5다. VNC는 실행 로그에 나온 포트를 VS Code에서 포워딩하여 접속한다. 출력은 `output_etc/check/` 아래이며 `teachers.json`에 두 원본 경로/epoch/frame/해시와 담당 규칙을 저장한다. 단일 통합 모델의 성능과 구분한다.

5명 이상은 `GROUP_SPAWN_OVER4=1`을 자동 선택한다. `random_scenario`는 solo/enable/coupled 순차 선택, `mixed`는 두 관계 포함을 보장한다. 공동 최대4명·초기 그룹 중심 간격5m·불필요한 distractor 물리/관측 제외·ordered 구역 배정을 viewer와 eval에서 공통 사용한다. 4명은 원래 장면 샘플링/배치를 유지하고 viewer/eval 구역 배정은 ordered다. N≥4·M≥N을 지원하며 토큰이 부족한 장면만 다시 뽑는다. M=N에서는 ON_TOP 포함 구성이 제외된다. 고정 프리셋의 예산/크기 조건은 [명세](reference/stage2.md#물체와-관측)를 따른다. `EPISODE_LENGTH`는 viewer의 재생 길이만 바꾸며 공간 채점은1200step을 유지한다.

단일 통합 PTH의5명 이상 viewer도 `GROUP_SPAWN_OVER4=1`을 붙인다. 직접 `run.py` 실행은 `--group_spawn_over4 --coupled_assignment ordered --num_agents N --num_objects 2N`을 사용한다. 두 teacher 직접 실행은 여기에 `--teacher_before <enable.pth> --teacher_coupled <coupled.pth>`를 지정하고 `--checkpoint`는 생략한다.

```bash
# stage2: 단일 통합 PTH의 8명·5m 그룹 배치
TOKENHSI_GPU=6 GROUP_SPAWN_OVER4=1 TASK_GRAPH=mixed bash tokenhsi/scripts/multi_agent/stage2_vnc.sh <stage2.pth> 8 1 16 10
```

### 통합 Stage2 900step 버전

`stage2_900`(일반 RL)·`stage2_distill_900`(두 teacher 모방)은 기존1200step 버전과 별도로 관리한다. 모델·샘플링·보상·학습 방식은 같고 에피소드 제한과 실험/저장 이름을 분리했다. 전용 train/test/VNC 명령은 [config](config.md) 각 목록의 맨 아래에 있다. 학습·viewer·정식 평가 기본 제한은 모두900step이다.

Distill teacher PTH는 `tokenhsi/data/cfg/train/rlg/stage2_distill_900.yaml`의 `params.config.distillation.teacher_before`·`teacher_coupled`에서 각각 지정한다. 현재900 버전은 사용자 지정 enable21k(11월5일22:58 run)·coupled17k를 사용하며 기존1200step teacher를 그대로 불러온다. 1200step `stage2_distill`의 teacher 선택은 유지한다. `rollout_steps=32`와 teacher20+180iteration 전환은 유지한다.

두 경로는 저장소의 `checkpoints/` 아래 PTH를 사용한다. BEFORE21k는 기준 repo의 같은 이름 파일에 연결했다. **coupled17k는 아직 없으므로 실제 증류 학습은 미완료**다. 파일이 준비되면 `checkpoints/Stage2CoupledPushIndividualGoalSolo5Carry5GroundRelFusionZeroBg_00017000.pth`에 넣거나 같은 이름의 심링크를 만든다. 준비 후 `TOKENHSI_GPU=0 CHECK=1 bash tokenhsi/scripts/multi_agent/stage2_distill_900_train.sh 4 1024 8`로1iteration을 확인하며, 본학습 전에는 `CHECK`·`MAX_ITERATIONS`를 해제한다.

테스트는 학습된 student PTH 하나와 저장 설정으로 실행하며 teacher PTH는 필요하지 않다. 화면 없는 테스트는 [config](config.md)의 test 명령에 `HEADLESS=1`을 사용한다. 현재 확인용 `InitializedStudent.pth`는 Stage1 전이 직후 저장한 초기화 전용 파일이며 학습 결과나 증류 resume용 checkpoint가 아니다. [확인 범위·로그](../output_etc/check/stage2_distill_900_setup/report.md).

```bash
# stage2_900: checkpoint 공간 평가
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_900_eval.sh "$CKPT" --output output_etc/check/stage2_900_eval

# stage2_distill_900: student PTH 하나로 공간 평가
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_distill_900_eval.sh "$CKPT" --output output_etc/check/stage2_distill_900_eval
```

같은512환경 rollout에서 기존 지표와 네 가지 interaction group 성공률을 함께 기록한다. Enable은선행/후행 각각 한 번 이상 성공, coupled는 실제 공유 상자 중심 도달 기준이다. `summary.json`의 `interaction_group_scores`에 풀네임별 seed 평균/표본std와 건수를 저장한다. [정의](reference/evaluation.md#통합-stage2의-interaction-group-성공률).

### Warehouse Bridge 초기화 실험

`stage2_sensor_900_warehouse_bridge_init`은 cliff5% / bridge40% / room25% / warehouse30%, Bridge 내부 연결 완료20% / 기존 근접40% / 선두1.5cm 삽입40%로 학습한다. 완료 분기는 crosser 입구0.8–2m 및 출발 무대 운반 상자1–2m 근접 배치를 적용한다. 모델·PPO·보상·동결·Distill1.5k 출발점과 Warehouse 전용 규칙은 유지한다. [명세](reference/stage2_sensor_900.md#warehouse-bridge-초기화-실험).

```bash
# 공유 MPS 확인: 다른 학습이 연결되어 있으므로 중단/재시작하지 않는다
source mps/shell.sh
mps_status 0

# 4명·1024환경 짧은 학습
TOKENHSI_GPU=0 CHECK=1 MAX_ITERATIONS=1 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_bridge_init_train.sh 4 1024 8

# 새 실험의 PTH로 Warehouse 재로딩 / 혼합 평가 경로 확인
TOKENHSI_GPU=0 SCENARIO=warehouse HEADLESS=1 EPISODE_LENGTH=16 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_bridge_init_test.sh "$CKPT" 4 8 8 1
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_bridge_init_eval.sh "$CKPT" --envs 8 --seeds 0 --output output_etc/check/warehouse_bridge_init_eval

# 같은 실험의 PTH에서 재개
TOKENHSI_GPU=0 RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_bridge_init_train.sh 4 1024 8
```

본학습은 CHECK/MAX_ITERATIONS/OUTPUT_PATH/RESUME_CHECKPOINT/SCENARIO/SENSOR_MASK 및 진입 형상 override를 해제한다. 새 초기화는 YAML과 checkpoint 옆 saved config로 보존한다. 검사 산출물은 [검증 보고서](../output_etc/check/warehouse_bridge_init_20261108/report.md)에 정리한다.

#### Warehouse Bridge 초기화 본학습 서비스

새 서비스는 `tokenhsi-sensor-warehouse-bridge-init-20261108-174320.service`이며 실행 정보는 [launch.json](../output/stage2_sensor_900_warehouse_bridge_init/launch_20261108-174320/launch.json)에 저장한다. GPU0·tokenhsi_sm120·4명·1024환경·8토큰·900step·seed42로 Distill1.5k에서 fresh 시작한다.

이미 실행 중인 `tokenhsi-mps-gpu0.service`를 Requires/After로 공유한다. MPS 및 다른 repo의3시나리오 학습은 재시작하지 않는다. 기존 roomopen 서비스 `tokenhsi-sensor-warehouse-square-20261107-192146.service`만 중단하고 마지막4.5k PTH와 로그를 보존한다. 사용자 systemd·app.slice·Linger=yes·Restart=no로 대화/SSH 종료와 독립적으로 실행한다.

```bash
# 새 학습과 실제 MPS 연결 확인
systemctl --user status tokenhsi-sensor-warehouse-bridge-init-20261108-174320.service --no-pager
source mps/shell.sh
mps_status 0

# 로그
tail -f output/stage2_sensor_900_warehouse_bridge_init/launch_20261108-174320/train.log

# 이 새 roomopen 학습만 중단 (공유 MPS 및 다른 repo 학습은 유지)
systemctl --user stop tokenhsi-sensor-warehouse-bridge-init-20261108-174320.service
```

start.sh는 fresh 실행용이다. 서비스 재시작은 checkpoint 재개를 뜻하지 않으며 재개하려면 같은 실험의 PTH를 RESUME_CHECKPOINT에 지정한다.

### Stage2 sensor 900

`stage2_sensor_900`은 `output/stage2/Stage2Distill900_00001500.pth`를 전이하는 별도 PPO 실험이다. 기존 distillation 구현은 유지하며 이 실험은 teacher 파일을 사용하지 않는다. 두 센서 실험의 공통 agent는 actor encoder·actor Stage2 coordination·sigma·입력 RMS를 동결하고 센서·action head·critic 전체·AMP discriminator 및 AMP/value RMS를 학습한다. Value RMS는 fresh 초기화한다. 새 학습 계약의 checkpoint만 resume할 수 있으며 이전 전체 동결 PTH는 테스트만 가능하다. [샘플링·크기·동결 명세](reference/stage2_sensor_900.md).

학습·테스트·VNC 명령은 [config](config.md) 목록 끝에 있다. 기본 `tokenhsi`·GPU0·4명·1024환경·8토큰·900step이며 확인 학습은 환경 수를 유지한다. `CHECK=1 MAX_ITERATIONS=1`은 기존 학습 루프 조건상 epoch2까지 수행한다. 본학습 전 확인용 변수를 해제한다.

```bash
# 짧은 학습 (별도 확인 경로)
TOKENHSI_GPU=0 CHECK=1 MAX_ITERATIONS=1 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_train.sh 4 1024 8

# 센서 checkpoint에서 resume; 본학습용 경로는 기본 output/ 사용
TOKENHSI_GPU=0 RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/stage2_sensor_900_train.sh 4 1024 8

# 시나리오 고정과 센서 격자 표시 (테스트 전용; 학습은 세 시나리오 혼합)
TOKENHSI_GPU=0 SCENARIO=bridge SENSOR_VIS=1 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_vnc.sh "$CKPT" 4 1 8 10

# 화면 없는 짧은 추론
TOKENHSI_GPU=0 HEADLESS=1 EPISODE_LENGTH=16 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_test.sh "$CKPT" 4 1 8 1

# 평가 경로 smoke; 보고용 표본 수가 아님
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_eval.sh "$CKPT" --envs 8 --seeds 0 --output output_etc/check/stage2_sensor_900_eval
```

`SCENARIO=cliff|bridge|room`은 테스트/평가에서만 사용한다. `SENSOR_MASK=1`은 테스트에서 센서 투영 출력을 0으로 만드는 ablation이며 `SENSOR_VIS=1`은 viewer 표시만 켠다. 정식 평가의 cliff unreachable 사람은 과제·interaction group 성공률 분모에서 제외하고 별도 건수를 저장한다. 낙상률은 전체 사람 기준이다. 기본 정식 평가는 512환경×seed0/1/2이며 900step이다.

학습 noVNC는6080, 테스트 noVNC는6081이다. VS Code 포트 전달 후 `http://localhost:6080/vnc.html?autoconnect=1&resize=remote` 또는6081을 연다. 원래 `Stage2Distill900` PTH는 새 센서 네트워크 테스트용 checkpoint가 아니므로 먼저 센서 학습으로 저장한 PTH를 사용한다.

#### Stage2 sensor 900 warehouse

`stage2_sensor_900_warehouse`는 네 장면을 각25%로 섞는 별도 실험이다. 기존과 같은 Distill9001.5k에서 센서 분기를 새로 초기화하고 위의 부분 동결 학습 범위를 적용한다. 새 run 이름은 `Stage2Sensor900WarehouseSquare`다. [창고 명세](reference/stage2_sensor_900.md#창고-확장-stage2_sensor_900_warehouse), [기본 명령](config.md)을 따른다. `CKPT`는 이 네 장면 실험에서 저장한 checkpoint여야 하며 resume에는 새 학습 계약이 필요하다.

```bash
# 확인 학습 (4명·1024환경·8토큰)
TOKENHSI_GPU=0 CHECK=1 MAX_ITERATIONS=1 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_train.sh 4 1024 8

# 이어학습
TOKENHSI_GPU=0 RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_train.sh 4 1024 8

# 창고 고정 로컬 viewer / 서버 noVNC (테스트 기본6081)
TOKENHSI_GPU=0 SCENARIO=warehouse HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_test.sh "$CKPT" 4 1 8 10
TOKENHSI_GPU=0 SCENARIO=warehouse SENSOR_VIS=1 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_vnc.sh "$CKPT" 4 1 8 10

# 화면 없는 창고 추론
TOKENHSI_GPU=0 SCENARIO=warehouse HEADLESS=1 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_test.sh "$CKPT" 4 16 8 1

# 혼합 정식 평가 (창고 고정 검사는 SCENARIO=warehouse 추가)
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_eval.sh "$CKPT" --envs 512 --seeds 0 1 2 --output output_etc/check/stage2_sensor_900_warehouse_eval
```

`SCENARIO`와 `SENSOR_MASK`는 평가/viewer 전용이며 본학습에는 해제한다. `SENSOR_VIS=1`은 viewer 높이 격자 표시다. 세 장면 checkpoint의 테스트/재개는 기존 `stage2_sensor_900` wrapper를 사용한다.

### 본학습 백그라운드 서비스

2026-11-06 본학습은 `tokenhsi-stage2-sensor-900-20261106-202813.service`로 실행했다. 사용자 systemd 서비스이며 `Linger=yes`를 확인하여 대화/SSH 세션 종료와 독립적으로 실행한다. GPU0·4명·1024환경·8토큰·900step·headless, Distill9001.5k에서 fresh 전이하며 확인용 제한·checkpoint·시나리오 고정을 해제했다. YAML의 최대 epoch1000000·중간 checkpoint500epoch 간격을 사용한다.

```bash
# 상태
systemctl --user status tokenhsi-stage2-sensor-900-20261106-202813.service --no-pager

# 실시간 로그
tail -f output/stage2_sensor_900/launch_20261106-202813/train.log

# 필요할 때 이 학습만 종료
systemctl --user stop tokenhsi-stage2-sensor-900-20261106-202813.service
```

결과와 TensorBoard는 `output/stage2_sensor_900/Stage2Sensor900_*/`, 실행 정보는 `output/stage2_sensor_900/launch_20261106-202813/launch.json`에 저장한다. 오류 발생 시 자동으로 새 학습을 시작하지 않도록 서비스 `Restart=no`를 사용한다. 재개할 때는 저장된 센서 checkpoint와 기존 resume 명령을 사용한다.

#### 창고 포함 본학습 백그라운드 서비스

2026-11-07 부분 동결 수정본은 `tokenhsi-sensor-warehouse-square-20261107-192146.service`를 사용한다. `Linger=yes`·app.slice·Restart=no로 대화/SSH 세션과 독립적으로 실행한다. GPU0·tokenhsi_sm120·4명/1024환경/8토큰/900step·headless, 네 장면 각25%, Distill9001.5k fresh 전이다. CHECK/MAX_ITERATIONS/RESUME_CHECKPOINT·전이 override·SCENARIO/SENSOR_MASK/SENSOR_VIS를 해제한다. 다른 repo의3장면 학습은 보존한다. 이전 전체 동결 warehouse 서비스는 재시작 작업 당시 실행 중이 아니었으며 기존 결과는 보존한다.

```bash
# 창고 포함 학습 상태
systemctl --user status tokenhsi-sensor-warehouse-square-20261107-192146.service --no-pager

# 실시간 로그
tail -f output/stage2_sensor_900_warehouse/launch_square_20261107-192146/train.log

# 필요할 때 이 학습만 종료
systemctl --user stop tokenhsi-sensor-warehouse-square-20261107-192146.service
```

결과·TensorBoard는 `output/stage2_sensor_900_warehouse/Stage2Sensor900WarehouseSquare_*/`, 실행 설정·로그는 [launch.json](../output/stage2_sensor_900_warehouse/launch_square_20261107-192146/launch.json)에 기록한다. 최대 epoch1000000·checkpoint500epoch 간격은 YAML 기본값이다. 재부팅 후 자동 시작을 등록한 서비스는 아니며 오류 발생 시 자동 재시작하지 않는다.

### Stage2 distillation

학습·로컬 viewer·VNC 기본 명령은 [config](config.md) 맨 아래의 `stage2_distill` 항목을 따른다. 아래 teacher 경로와 실행 전환/replay 설정은 `tokenhsi/data/cfg/train/rlg/stage2_distill.yaml`의 `params.config.distillation`에 있다. 기본 teacher는 enable13.5k/coupled25.5k, solo는 enable 담당이다. [학습/동결/저장 명세](reference/stage2_distill.md).

```bash
# stage2_distill: 4명·1024환경·8토큰 본학습
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_distill_train.sh 4 1024 8

# stage2_distill: 같은 환경 수에서1iteration 확인 학습
TOKENHSI_GPU=6 CHECK=1 OUTPUT_PATH=output_etc/check/stage2_distill_check bash tokenhsi/scripts/multi_agent/stage2_distill_train.sh 4 1024 8

# stage2_distill: 같은 teacher/설정으로 optimizer·replay를 이어학습
TOKENHSI_GPU=6 RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/stage2_distill_train.sh 4 1024 8

# stage2_distill: student PTH 하나로4명 기본 ordered 공간 평가
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_distill_eval.sh "$CKPT" --output output_etc/check/stage2_distill_eval4

# stage2_distill: student PTH 하나로8명·8토큰·5m·ordered 확장 평가
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage2_distill_eval.sh "$CKPT" --agents 8 --objects 8 --group_spawn_over4 --output output_etc/check/stage2_distill_eval8_m8

# stage2: 같은 확장 평가 인자 지원
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage2_eval.sh "$CKPT" --agents 8 --objects 8 --group_spawn_over4 --output output_etc/check/stage2_eval8_m8
```

본학습 전 `CHECK`·`MAX_ITERATIONS`·`RESUME_CHECKPOINT`·`OUTPUT_PATH` 잔여 설정을 해제한다. `--coupled_assignment random|ordered|distance|optimal`로 eval 배정 방식을 지정할 수 있고 기본은 ordered다. `--agents`/`--objects`는 통합 `stage2`·`stage2_distill` eval에서만 받으며 N>4는 `--group_spawn_over4`가 필요하다. 평가 seed 기본0/1/2·환경 수512·1200step이며 인원/물체 수를 바꾼 결과는 기본 학습 분포 평가와 구분한다.

### Warehouse RSI

`stage2_sensor_900_warehouse_bridge_init_rsi`는50% 닫힘·25% 부분 개방·25% 완전 개방 상태를 혼합해 학습한다. 최신 Bridge 초기화·장면 비율·부분 동결은 유지한다. [명세](reference/stage2_sensor_900.md#warehouse-진행-상태-rsi)를 따른다.

```bash
# 기본 closed 정식 평가
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_bridge_init_rsi_eval.sh "$CKPT" --output output_etc/check/warehouse_rsi_eval

# open 상태만 확인하는 서버 viewer
TOKENHSI_GPU=0 SCENARIO=warehouse WAREHOUSE_START=open bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_bridge_init_rsi_vnc.sh "$CKPT" 4 1 8 10

# 세 시작 상태 혼합의 화면 없는 기능 검사
TOKENHSI_GPU=0 SCENARIO=warehouse WAREHOUSE_START=mixed HEADLESS=1 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_bridge_init_rsi_test.sh "$CKPT" 4 32 8 1
```

학습은 `WAREHOUSE_START`를 해제한다. 기존 학습 종료 시 공유 MPS daemon과 다른 repo의 작업을 보존하며, MPS client의 GPU 작업 종료가 확인된 뒤 해당 학습 서비스만 중단한다.

#### Warehouse RSI 본학습 서비스

2026-11-08 21:39 KST에 Distill9001.5k에서 fresh 본학습을 시작했다. 사용자 systemd `tokenhsi-sensor-warehouse-rsi-20261108-213952.service`·app.slice·Linger=yes·Restart=no, GPU0·MPS·tokenhsi_sm120·4명/1024환경/8토큰/900step이다. CHECK/MAX_ITERATIONS/RESUME_CHECKPOINT·WAREHOUSE_START/SCENARIO/SENSOR_MASK/SENSOR_VIS를 해제한다. 장면5/40/25/30%와 Bridge20/40/40%는 유지한다. 기존 roomopen Bridge-init 서비스는 종료했고1k PTH·로그는 보존했다.

```bash
# 현재 상태
systemctl --user status tokenhsi-sensor-warehouse-rsi-20261108-213952.service --no-pager

# 실시간 로그
tail -f output/stage2_sensor_900_warehouse_bridge_init_rsi/launch_20261108-213952/train.log

# MPS client 목록을 먼저 확인하는 읽기 전용 명령
TOKENHSI_GPU=0 bash -c 'source mps/shell.sh; mps_use 0; printf "ps\n" | nvidia-cuda-mps-control'
```

공유 MPS에서는 바로 signal로 종료하지 않는다. 대상 service·PID·cwd/config·MPS server/client 대응을 확인하고, 같은 MPS pipe의 `terminate_client <server PID> <대상 client PID>`가0을 반환한 뒤 해당 systemd service만 중단한다. 이 교체에서는 `terminate_client 374795 406675` 성공 후 기존 서비스만 종료했으며 다른 client421090과 MPS server374795 및 다른 학습의 epoch 진행을 확인했다. [NVIDIA 절차](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html#client-early-termination).

결과·TB는 `output/stage2_sensor_900_warehouse_bridge_init_rsi/Stage2Sensor900WarehouseBridgeInitRsi_08-21-39-54/`, [실행 정보](../output/stage2_sensor_900_warehouse_bridge_init_rsi/launch_20261108-213952/launch.json), [시작 확인](../output_etc/check/warehouse_rsi_20261108/startup_verification.json), [검증 보고서](../output_etc/check/warehouse_rsi_20261108/report.md)를 따른다. 세션 종료와 독립적으로 실행되며 재부팅 자동 시작/오류 자동 재시작은 등록하지 않았다.

### Warehouse safety

사용자 승인한 새 실험은 `stage2_sensor_900_warehouse_safety`다. 장면5/43/17/35%·Bridge90% 다중 횡단·Cliff40/60·Room30/40/30·Warehouse single PUSH1명/작업자 pairing0.8을 사용한다. [역할 비율·커리큘럼·동결·전환 계약](reference/stage2_sensor_900.md#warehouse-safety-샘플링), [검증](../output_etc/check/warehouse_safety_20261109/report.md).

```bash
# 기존 Warehouse RSI 6000epoch에서 상태를 그대로 이어서 학습
TOKENHSI_GPU=0 RESUME_CHECKPOINT='output/stage2_sensor_900_warehouse_bridge_init_rsi/Stage2Sensor900WarehouseBridgeInitRsi_08-21-39-54/nn/Stage2Sensor900WarehouseBridgeInitRsi_00006000.pth' bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_safety_train.sh 4 1024 8

# 새 safety checkpoint를 넣어 사용. 기본 Bridge 평가는 기존 협력 분기
TOKENHSI_GPU=0 SCENARIO=bridge BRIDGE_CURRICULUM_EVAL=walk_entry bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_safety_vnc.sh "$CKPT" 4 1 8 10

TOKENHSI_GPU=0 SCENARIO=warehouse WAREHOUSE_START=mixed HEADLESS=1 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_safety_test.sh "$CKPT" 4 16 8 1

TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_safety_eval.sh "$CKPT" --output output_etc/check/warehouse_safety_eval
```

위 train 명령은 수동 재개용이며 현재 실행 중인 서비스와 중복 실행하지 않는다. 새 train/train_vnc wrapper는 ENV_CONFIG를 명시하여 기존 checkpoint 옆 설정이 새 분포를 덮어쓰지 않게 한다. 모델/optimizer/입력·AMP/value RMS/epoch/frame는 복원하고 새 curriculum anchor만6000으로 시작한다. 이후 새 checkpoint에서 재개할 때는 저장된 anchor를 유지한다.

#### Warehouse safety 본학습 서비스

아래6000epoch resume 서비스는2026-11-09 20:41 사용자 요청으로 중단했다. 현재 실행은 [fresh 서비스](#warehouse-safety-fresh-본학습-서비스)를 따른다.

2026-11-09 사용자 systemd·Linger=yes·app.slice·Restart=no·기존 GPU0 MPS Requires/After로 시작했다. 대화 세션 종료와 독립적이다. 기존 Roomopen learner는 작업 시작 시 이미 종료되어 있었고 중단할 프로세스가 없었다. 기존6000epoch PTH와 로그는 보존했다. 오류 자동 재시작·재부팅 자동 시작은 등록하지 않았다.

```bash
systemctl --user status tokenhsi-sensor-warehouse-safety-20261109-195459.service

tail -f output/stage2_sensor_900_warehouse_safety/resume_20261109-195459/train.log

# 공유 MPS를 유지하고 이 learner만 종료
python3 output/stage2_sensor_900_warehouse_safety/resume_20261109-195459/stop_safely.py
```

[실행 정보](../output/stage2_sensor_900_warehouse_safety/resume_20261109-195459/launch.json), [시작 검증](../output_etc/check/warehouse_safety_20261109/startup_verification.json). Safe stop은 cwd/config/PID start time/cgroup/MPS client를 확인하고 `terminate_client` 성공0 이후 해당 서비스만 중단한다. 공유 MPS daemon/server를 직접 종료하지 않는다.

#### Warehouse safety fresh 본학습 서비스

현재 이 서비스는 사용자 요청으로 중단했다. [reset 수정·500epoch 보존 상태](#rollout-reset-수정과-500epoch-보존)를 따른다.

현재 실행은6000epoch resume이 아니라 기존 방식의 Distill9001500 전이다. 개선안 YAML은 동일하다. 센서·optimizer·value RMS·epoch/frame·curriculum을 새로 시작하며 actor encoder/coordination/sigma·입력 RMS는 동결, critic 전체·head·센서·AMP와 value/AMP 통계는 학습한다. 아래 전경 명령을 실행 중인 서비스와 중복 실행하지 않는다.

```bash
# 수동 fresh 실행 시 이전 resume override를 해제
unset RESUME_CHECKPOINT CHECK MAX_ITERATIONS OUTPUT_PATH ENV_CONFIG STAGE1_CHECKPOINT
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_safety_train.sh 4 1024 8

# 현재 대화 세션과 독립된 서비스
systemctl --user status tokenhsi-sensor-warehouse-safety-fresh-20261109-204146.service

tail -f output/stage2_sensor_900_warehouse_safety/fresh_20261109-204146/train.log

# 이 learner만 MPS 안전 종료
python3 output/stage2_sensor_900_warehouse_safety/fresh_20261109-204146/stop_safely.py
```

[실행 정보](../output/stage2_sensor_900_warehouse_safety/fresh_20261109-204146/launch.json), [시작 검증](../output_etc/check/warehouse_safety_fresh_20261109/startup_verification.json), [교체·검증 보고서](../output_etc/check/warehouse_safety_fresh_20261109/report.md). Linger=yes·app.slice·Restart=no로 대화 종료와 독립적이며 공유 MPS daemon/server는 종료하지 않는다. 이전6000resume run은 비교/추적용으로 보존했다.

### Rollout reset 수정과 500epoch 보존

사용자 요청으로 Roomopen fresh 학습을 중단하고 rollout 마지막 step의 done을 다음 rollout 첫 행동 전에 처리하도록 수정했다. **현재 중단 상태이며 재시작하지 않았다.** 기존 서비스 `tokenhsi-sensor-warehouse-safety-fresh-20261109-204146.service`는 inactive/dead·MainPID0·Restart=no다. 공유 MPS와 다른 학습은 유지했다.

보존한500epoch PTH는 [Stage2Sensor900WarehouseSafety_00000500.pth](../output/stage2_sensor_900_warehouse_safety/Stage2Sensor900WarehouseSafety_09-20-41-48/nn/Stage2Sensor900WarehouseSafety_00000500.pth)다. 모델·optimizer·입력/value/AMP RMS와 curriculum anchor0을 포함한다. SHA256은 `c33eff7da244d16071d6d5795c2e5c639dc0a53536b7c40b60d723df431727e7`다. 종료 직전 로그577epoch와 이전 run 전체도 보존했다.

```bash
# 현재 중단 상태 확인
systemctl --user status tokenhsi-sensor-warehouse-safety-fresh-20261109-204146.service

# 향후 사용자가 재개를 지시한 경우의 명령 예시. 이번 작업에서는 실행하지 않았다.
TOKENHSI_GPU=0 RESUME_CHECKPOINT='output/stage2_sensor_900_warehouse_safety/Stage2Sensor900WarehouseSafety_09-20-41-48/nn/Stage2Sensor900WarehouseSafety_00000500.pth' bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_safety_train.sh 4 1024 8
```

재개 시 새 전이학습이 아니라500epoch의 모델/optimizer/정규화/진행도를 복원한다. 지속 백그라운드 서비스는 이번에 새로 등록하지 않았다. [수정·CPU 검증](../output_etc/check/rollout_reset_fix_20261109/report.md), [체크포인트 검사](../output_etc/check/rollout_reset_fix_20261109/checkpoint_verification.json), [MPS 종료 확인](../output_etc/check/rollout_reset_fix_20261109/after_stop.json).

## Warehouse 공동 PUSH

`stage2_sensor_900_warehouse_coop`의 기본 train/test/VNC 명령은 [config](config.md)를 따른다. 기본4명·1024환경·8토큰·900step·GPU0/MPS이며 Distill1500에서 새로 전이한다. 기존 Safety 학습은 중단 상태를 유지한다.

시험/VNC의 `WAREHOUSE_START=closed|partial|open|mixed`, `SCENARIO=warehouse`를 사용할 수 있다. 기본 시험은 closed이며 PUSH 인원과 방향은 새 YAML의 확률을 따른다. 새 checkpoint에는 새 환경 계약이 저장되며 이전 Safety checkpoint를 이 설정으로 직접 재개/추론하는 것은 허용하지 않는다. 새 실험 자체의 checkpoint 재개는 기존 `RESUME_CHECKPOINT`를 사용한다.

추가 wrapper `stage2_sensor_900_warehouse_coop_train_vnc.sh`와 `stage2_sensor_900_warehouse_coop_eval.sh`도 제공한다. 구현 상세는 [명세](reference/stage2_sensor_900.md#warehouse-공동-push), 검증 산출물은 `output_etc/check/warehouse_coop_20261110/`에 둔다.

### Warehouse 공동 PUSH 본학습 서비스

2026-11-10 사용자 요청으로 `stage2_sensor_900_warehouse_coop`을 Distill1500에서 새로 전이해 GPU0/MPS·4명·1024환경·8토큰·900step·seed42로 시작했다. CHECK/MAX_ITERATIONS·resume·검사/시나리오 override를 해제했다. 사용자 systemd의 app.slice·Linger=yes·Restart=no로 대화/SSH 종료와 독립적으로 실행한다. 기존 Safety 서비스와500epoch PTH, 공유 MPS 및 다른 GPU 작업은 유지한다.

```bash
# 서비스 상태
systemctl --user status tokenhsi-sensor-warehouse-coop-20261110-003205.service

# 본학습 로그
tail -f output/stage2_sensor_900_warehouse_coop/launch_20261110-003205/train.log

# 대상 learner의 신원을 검증하고 MPS에서 분리한 뒤 해당 서비스만 종료
python3 output/stage2_sensor_900_warehouse_coop/launch_20261110-003205/stop_safely.py
```

[실행 정보](../output/stage2_sensor_900_warehouse_coop/launch_20261110-003205/launch.json), [시작 확인](../output_etc/check/warehouse_coop_start_20261110/startup_verification.json). 본학습 결과는 `output/stage2_sensor_900_warehouse_coop/Stage2Sensor900WarehouseCoop_*/`의 nn·summaries에 저장한다.

## Warehouse 공동 PUSH Cliff·Room 조정

`stage2_sensor_900_warehouse_coop_cliffroom`의 기본 train/test/VNC 명령은 [config](config.md)를 따른다. coop와 같은 4명·1024환경·8토큰·900step·GPU0/MPS·Distill1500 fresh 전이이며 차이는 [명세](reference/stage2_sensor_900.md#warehouse-공동-push-cliffroom-조정-stage2_sensor_900_warehouse_coop_cliffroom)에 있다. 시험/VNC 옵션(`SCENARIO`, `WAREHOUSE_START`)과 `RESUME_CHECKPOINT` 재개는 coop와 같고, coop checkpoint를 이 설정으로 재개하는 계약 예외는 없다.

### Warehouse 공동 PUSH CliffRoom 본학습 서비스

2026-11-10 사용자 요청으로 coop 서비스를 중단하고 이 실험을 seed42로 시작했다. 사용자 systemd의 app.slice·Linger=yes·Restart=no로 실행한다. coop 3000epoch PTH, Safety PTH, 공유 MPS 및 다른 GPU 작업은 유지한다.

```bash
# 서비스 상태
systemctl --user status tokenhsi-sensor-warehouse-coop-cliffroom-20261110-144456.service

# 본학습 로그
tail -f output/stage2_sensor_900_warehouse_coop_cliffroom/launch_20261110-144456/train.log

# 대상 learner의 신원을 검증하고 MPS에서 분리한 뒤 해당 서비스만 종료
python3 output/stage2_sensor_900_warehouse_coop_cliffroom/launch_20261110-144456/stop_safely.py
```

[실행 정보](../output/stage2_sensor_900_warehouse_coop_cliffroom/launch_20261110-144456/launch.json), [시작 확인](../output_etc/check/warehouse_coop_cliffroom_20261110/startup_verification.json). 본학습 결과는 `output/stage2_sensor_900_warehouse_coop_cliffroom/Stage2Sensor900WarehouseCoopCliffRoom_*/`의 nn·summaries에 저장한다.

### Warehouse CliffRoom 1000epoch 재개

2026-11-11 사용자 요청으로 최신 `Stage2Sensor900WarehouseCoopCliffRoom_00001000.pth`에서 모델·optimizer·정규화·curriculum을 복원해 재개했다. 이전 run은 2026-11-10 21:45:33(KST)에 Room 배치 재시도 소진으로 종료됐고 마지막 완료 로그는1483epoch였다. 이번 작업은 기존 코드·YAML·샘플링을 유지한 재개다.

GPU0/MPS·4명·1024환경·8토큰·900step·seed42를 유지한다. 사용자 systemd app.slice·Linger=yes·Restart=no로 대화/SSH 종료와 독립적으로 실행하며 CHECK/MAX_ITERATIONS·평가 override는 해제했다.

```bash
# 현재 재개 서비스
systemctl --user status tokenhsi-sensor-warehouse-coop-cliffroom-resume-20261111-020937.service

# 재개 로그
tail -f output/stage2_sensor_900_warehouse_coop_cliffroom/resume_20261111-020937/train.log

# 대상 learner 신원과 MPS client를 확인한 뒤 해당 서비스만 안전 종료
python3 output/stage2_sensor_900_warehouse_coop_cliffroom/resume_20261111-020937/stop_safely.py
```

[실행 정보](../output/stage2_sensor_900_warehouse_coop_cliffroom/resume_20261111-020937/launch.json), [checkpoint 검사](../output_etc/check/warehouse_cliffroom_resume_20261111/checkpoint.json), [시작 확인](../output_etc/check/warehouse_cliffroom_resume_20261111/startup_verification.json). 원본1000epoch PTH와 이전 로그는 보존한다.

### Warehouse CliffRoom 5500epoch 재개

2026-11-12 사용자 요청으로 이 서비스를 안전 종료했다. 마지막 저장9000epoch와 기존 결과는 보존하며, 현재 학습은 아래 Room Warehouse 전용 서비스를 따른다.

2026-11-11 22:27:34(KST) Room 배치 재시도 소진으로 종료된 실행을5500epoch/frame720896000에서 이어간다. 창고 기하 사전 생성 후 Room 크기·벽 참조를 복원하고,64회 실패한 Room의 후반 배치에서 순서를 섞고 후보를16→64개로 늘렸다. 방/상자 크기·역할·거리/충돌 조건·보상식/계수·동결·YAML과 전체256회 상한은 유지한다.

문제 역할 조합16회×1024환경과 일반 혼합4회×1024환경·물리16step·부분 reset 확인 후 원본5500epoch의 모델·optimizer·정규화·curriculum에서 재개한다. GPU0/MPS·4명·1024환경·8토큰·900step·seed42·사용자 systemd app.slice·Linger=yes·Restart=no이며 대화/SSH 종료와 독립적이다.

```bash
# 현재 재개 서비스
systemctl --user status tokenhsi-sensor-warehouse-cliffroom-roomfix-20261111-232822.service

# 학습 로그
tail -f output/stage2_sensor_900_warehouse_coop_cliffroom/resume_roomfix_20261111-232822/train.log

# 해당 learner만 MPS에서 분리한 뒤 서비스 안전 종료
python3 output/stage2_sensor_900_warehouse_coop_cliffroom/resume_roomfix_20261111-232822/stop_safely.py
```

[실행 정보](../output/stage2_sensor_900_warehouse_coop_cliffroom/resume_roomfix_20261111-232822/launch.json), [검증](../output_etc/check/room_sampling_fix_20261111/report.md), [시작 확인](../output_etc/check/room_sampling_fix_20261111/startup_verification.json).

### Room Warehouse 전용 본학습

`stage2_sensor_900_room_warehouse`는 Distill1500에서 새로 전이하여 Room/Warehouse를50:50으로 학습한다. 벽 CPA 계수만0.7이며 사람 간 CPA0.5·BEFORE 대기 보상·기존 freeze를 유지한다. 상세 설정은 [명세](reference/stage2_sensor_900.md#roomwarehouse-전용-stage2_sensor_900_room_warehouse)를 따른다.

GPU0/MPS·4명·1024환경·8토큰·900step·seed42, 사용자 systemd app.slice·Linger=yes·Restart=no로 대화/SSH 종료 후에도 실행한다. `CHECK`·`MAX_ITERATIONS`·resume·평가 override를 해제하고 학습한다.

```bash
# 서비스 상태
systemctl --user status tokenhsi-room-warehouse-20261112-163455.service

# 학습 로그
tail -f output/stage2_sensor_900_room_warehouse/launch_20261112-163455/train.log

# 해당 learner만 MPS에서 분리한 뒤 서비스 안전 종료
python3 output/stage2_sensor_900_room_warehouse/launch_20261112-163455/stop_safely.py
```

공유 MPS에서는 직접 signal을 보내지 않는다. 안전 종료 스크립트는 PID/start time·cwd/config·cgroup·MPS client를 확인한 뒤 `terminate_client` 성공0 응답을 받아 해당 서비스만 중단한다.

[실행 정보](../output/stage2_sensor_900_room_warehouse/launch_20261112-163455/launch.json), [시작 확인](../output_etc/check/room_warehouse_20261112/startup_verification.json), [검증](../output_etc/check/room_warehouse_20261112/report.md).

### Warehouse 역할 고정 V4 본학습

`stage2_sensor_900_warehouse_fixedroles_v4`는 Warehouse만 사용하며 역할을 공동 PUSH 2명(방 안쪽으로 5m)·BEFORE AT 선행 1명·SIT/CLIMB/ON_TOP 후행 1명으로 고정한다. 보상은 V4이고, Distill900V4 2000에서 새로 전이한다. [명세](reference/stage2_sensor_900.md#warehouse-역할-고정-v4-stage2_sensor_900_warehouse_fixedroles_v4).

2026-11-16 04:09에 `tokenhsi-warehouse-fixedroles-v4-20261116-040900.service`로 시작했다. 설정은 GPU7/MPS, 4명·1024환경·8토큰·900step·seed42, 사용자 systemd app.slice·Restart=no다. 계정이 `Linger=no`이므로 모든 로그인 세션(tmux 포함)이 끝나면 서비스도 함께 종료될 수 있다.

```bash
# 서비스 상태
systemctl --user status tokenhsi-warehouse-fixedroles-v4-20261116-040900.service

# 학습 로그
tail -f output/stage2_sensor_900_warehouse_fixedroles_v4/launch_20261116-040900/train.log
```

공유 MPS에서 안전하게 종료하는 절차:
1. 대상 learner의 PID·시작 시각·cwd·config를 확인한다.
2. `CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-test-1003/gpu7/pipe`에서 `get_server_list`/`get_client_list`로 server와 client가 맞는지 확인한다.
3. `terminate_client <server> <learner PID>`가 0을 반환하는지 확인한다.
4. 그 뒤에 해당 서비스만 `systemctl --user stop`으로 중단한다.

MPS server와 다른 client에는 signal을 보내지 않는다. [실행 정보](../output/stage2_sensor_900_warehouse_fixedroles_v4/launch_20261116-040900/launch.json).

### Warehouse 역할 고정 V4 + BEFORE 2팀 본학습

`stage2_sensor_900_warehouse_fixedroles_v4_beforepairs`는 fixedroles_v4의 Warehouse reset 중 20%를 apron 위 BEFORE 2팀으로 바꾼 실험이다. Distill900V4 2000에서 새로 전이한다. [명세](reference/stage2_sensor_900.md#warehouse-역할-고정-v4--before-2팀-stage2_sensor_900_warehouse_fixedroles_v4_beforepairs).

학습에서는 검증된 BEFORE apron 배치를 메모리에 보관해 재사용한다. 두 팀의 후행 과제·near/far 조합과 환경 기하·상자 크기가 맞아야 적중하며, 역할/물체 매핑·과제 그래프·사람 자세·AMP 이력까지 함께 복원한다. 조건당 최대 2개 배치, 환경당 최대 32개 조건을 보관하고, 적중 가능한 reset의 25%는 새로 생성해 갱신한다. `warehouse/before_cache/*`에서 요청·적중·갱신·저장 수를 확인한다. `WAREHOUSE_BEFORE_CACHE=0`을 기존 학습 명령 앞에 붙이면 끈다. 평가·viewer에서는 꺼지고 checkpoint에는 저장하지 않는다.

`stateInit: Default` 학습은 첫 rollout 전에 Warehouse 환경을 BEFORE 2팀으로 32회 미리 배치해 캐시를 채운다. 시작할 때 준비 시간이 추가되며 `warehouse/before_cache/warmup_seconds`에 기록한다. 준비 중 변형 비율·에피소드/샘플링 계수·난수 상태는 복원한다. 실제 학습의 BEFORE 비율과 25% 갱신은 유지한다. `WAREHOUSE_BEFORE_CACHE_WARMUP=0`으로 준비를 생략하거나 횟수를 지정한다. RSI 초기화 실험의 기본값은 0이다. 변경된 코드는 다음 학습 실행부터 적용된다.

test/VNC의 `WAREHOUSE_VARIANT=before_pairs`는 PUSH 없이 BEFORE 2팀만, `push_before`는 2 PUSH + BEFORE 1쌍만 선택한다. 생략하거나 `mixed`이면 저장된 학습 비율을 사용한다. 이 옵션은 BEFORE 2팀이 있는 Warehouse config의 평가 전용이며 학습에서 거부한다. 혼합 장면 config에서는 `SCENARIO=warehouse`도 지정한다. 구성을 고정한 진단 결과는 기본 분포 평가와 구분한다.

Default 초기화의 BEFORE 2팀에서 후행의 비활성 Target 좌표가 잘못 커지던 문제를 수정했다. 기존 체크포인트도 수정된 코드로 test/VNC를 새로 실행하면 교정된 입력을 사용한다. 이미 실행 중인 학습에는 파일 수정이 자동 반영되지 않으며, 새 프로세스에서 배치 캐시도 다시 생성된다. doorprogress의 `*_eval.sh`도 올바른 coupled 환경·JointPush 데이터·PUSH 포함 평가 경로를 사용한다.

```bash
CKPT=output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress/Stage2Sensor900WarehouseFixedRolesV4BeforePairsDoorProgress_17-04-03-34/nn/Stage2Sensor900WarehouseFixedRolesV4BeforePairsDoorProgress_00003000.pth

# stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress: BEFORE 2팀만 서버 VNC
TOKENHSI_GPU=7 WAREHOUSE_VARIANT=before_pairs bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress_vnc.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress: BEFORE 2팀만 로컬 viewer
TOKENHSI_GPU=7 WAREHOUSE_VARIANT=before_pairs HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress_test.sh "$CKPT" 4 1 8 10

# stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress: BEFORE 2팀만 64환경 평가
TOKENHSI_GPU=7 WAREHOUSE_VARIANT=before_pairs bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress_eval.sh "$CKPT" --envs 64 --seeds 0 --output output_etc/check/warehouse_before_target_fix_20261117/eval_before_pairs64
```

2026-11-16 11:36에 `tokenhsi-warehouse-beforepairs-20261116-113624.service`로 시작했다. 설정은 GPU7/MPS, 4명·1024환경·8토큰·900step·seed42, app.slice·Restart=no다. 이전 fixedroles_v4 서비스(2500epoch)는 MPS `terminate_client`(반환 0)로 분리한 뒤 중지했다. 계정이 `Linger=no`이므로 모든 로그인 세션이 끝나면 서비스도 함께 종료될 수 있다.

```bash
# 서비스 상태
systemctl --user status tokenhsi-warehouse-beforepairs-20261116-113624.service

# 학습 로그
tail -f output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs/launch_20261116-113624/train.log
```

안전 종료는 [역할 고정 V4 본학습](#warehouse-역할-고정-v4-본학습)과 같은 MPS 절차를 따른다. [실행 정보](../output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs/launch_20261116-113624/launch.json).

2026-11-16 20:40에 leaderdone 실험으로 교체하면서 이 서비스를 MPS 절차로 중지했다. 약 2900 iteration까지 돌았지만 마지막 저장은 2500epoch다. 사용자 요청으로 20:54에 2500epoch(`_00002500.pth`)에서 `RESUME_CHECKPOINT`로 이어서 학습한다. leaderdone 학습과 같은 GPU7 MPS를 함께 쓴다. 새 run 폴더는 `Stage2Sensor900WarehouseFixedRolesV4BeforePairs_16-20-55-06`이고 epoch 2501부터 기록한다.

```bash
systemctl --user status tokenhsi-warehouse-beforepairs-resume-20261116-205455.service
tail -f output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs/resume_20261116-205455/train.log
```

[재개 실행 정보](../output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs/resume_20261116-205455/launch.json).

### Warehouse 역할 고정 V4 + 선행 완료 시작 + 벽 박스 CPA 본학습

`stage2_sensor_900_warehouse_fixedroles_v4_leaderdone_wallbox`는 fixedroles_v4에 키 두 개를 더한 실험이다. 학습 reset의 30%를 선행 AT 완료 상태에서 시작하고, 벽 CPA를 실제 벽 박스 기준으로 계산한다. Distill900V4 2000에서 새로 전이한다. [명세](reference/stage2_sensor_900.md#warehouse-역할-고정-v4--선행-완료-시작--벽-박스-cpa-stage2_sensor_900_warehouse_fixedroles_v4_leaderdone_wallbox).

2026-11-16 20:42에 `tokenhsi-warehouse-leaderdone-20261116-204212.service`로 시작했다. 설정은 GPU7/MPS, 4명·1024환경·8토큰·900step·seed42, app.slice·Restart=no다. 이전 beforepairs 서비스(2500epoch)는 MPS `terminate_client 112102 1720488`(반환 0)로 분리한 뒤 중지했다. 계정이 `Linger=no`이므로 모든 로그인 세션이 끝나면 서비스도 함께 종료될 수 있다.

```bash
# 서비스 상태
systemctl --user status tokenhsi-warehouse-leaderdone-20261116-204212.service

# 학습 로그
tail -f output/stage2_sensor_900_warehouse_fixedroles_v4_leaderdone_wallbox/launch_20261116-204212/train.log
```

안전 종료는 [역할 고정 V4 본학습](#warehouse-역할-고정-v4-본학습)과 같은 MPS 절차를 따른다. [실행 정보](../output/stage2_sensor_900_warehouse_fixedroles_v4_leaderdone_wallbox/launch_20261116-204212/launch.json).

### doorprogress 좌표 수정본 새 본학습

2026-11-18 GPU6 MPS doorprogress 서비스는 중단했다. 재개에 쓸 마지막 정기 저장본은 **5000epoch**다. 아래는 기존 실행 기록이다. [중단·저장본 확인](../output_etc/check/warehouse_ontop_waypoint/stopped_training.json).

GPU6의 좌표·공유 상자·AT 높이 마스크 수정본은 기존 fresh 학습의 **1500epoch/frame196608000 저장본에서 MPS로 재개**한다. 4명·1024환경·8토큰·900step·seed42·headless, Default/closed·BEFORE 2팀20%·캐시 준비32회·정밀 보상1mm를 유지한다. 기존 doorprogress 실험을 유지한다.

사용자 요청으로 이전 직접 CUDA 서비스 `tokenhsi-warehouse-doorprogress-gatefix-gpu6-20261117-201917.service`·learner1717136을 중지했다. 종료 시점 약1636epoch의 미저장 진행은 복원되지 않으며, 마지막 정기 저장본1500의 모델·optimizer·정규화 통계·epoch/frame을 복원한다. 원본 checkpoint와 이전 실행 결과는 보존한다.

재개에 사용한 서비스는 `tokenhsi-warehouse-doorprogress-resume-mps-gpu6-20261118-015507.service`다. 실제 학습 cwd는 `/home/hwanhee/jhh/TokenHSI_stage12_warehouse_fix_reward`, conda는 기존 실행과 같은 `tokenhsi_jhh`다. GPU6의 검증된 MPS pipe `/tmp/mps-test-1003/gpu6/pipe`에 연결한다. 운영 스크립트·로그는 이 checkout에 두고, 학습 결과는 기존 cwd의 동일 output 루트 아래 새 run에 저장한다. 기존 코드·config hash6개는 원래 시작 기록과 일치한다.

```bash
# 서비스 상태
systemctl --user status tokenhsi-warehouse-doorprogress-resume-mps-gpu6-20261118-015507.service

# 학습 로그 (이 checkout의 저장소 루트)
tail -f output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress/launch_resume_mps_gpu6_20261118-015507/train.log
```

[실행 정보](../output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress/launch_resume_mps_gpu6_20261118-015507/launch.json)·[재개 스크립트](../output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress/launch_resume_mps_gpu6_20261118-015507/start.sh)·[전환 검증](../output_etc/check/gpu6_mps_resume/verification.json). 재실행은 현재 서비스가 종료된 뒤 같은 `start.sh`를 사용한다. 안전 종료는 PID·cwd·cgroup과 MPS client를 확인한 후 대상 client만 `terminate_client`로 분리하고 해당 서비스만 중지한다. MPS 서버나 다른 client를 종료하지 않는다.

GPU7의 이전 좌표 수정본은 `tokenhsi-warehouse-doorprogress-fresh-targetfix-20261117-131057.service`·PID824377·run `Stage2Sensor900WarehouseFixedRolesV4BeforePairsDoorProgress_17-13-12-17`으로 유지한다. 실행 중인 이 프로세스에는 이후 AT 높이 마스크 수정이 자동 반영되지 않는다. [GPU7 실행 정보](../output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress/launch_fresh_targetfix_20261117-131057/launch.json)·[시작 검증](../output_etc/check/doorprogress_relaunch_20261117/startup_verification.json). GPU7의 종료는 해당 PID·cwd·cgroup을 확인한 뒤 [공통 MPS 절차](#warehouse-역할-고정-v4-본학습)를 따른다. GPU5 Bridge 학습과 이전 저장본·결과는 보존한다.

## DistillV4 Warehouse 가상 상자 제로샷 viewer

추론 전용 실험이며 새 학습 config는 만들지 않는다. 기반 환경은 `stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress.yaml`이고 전용 viewer에서 `WAREHOUSE_VARIANT=push_before`, `WAREHOUSE_START=closed`를 고정한다. 4명·8토큰·Default 자세·공동 PUSH 2명+BEFORE 1쌍이며 평가 경로라 배치 캐시는 준비하지 않는다. 기존 보상·성공 판정·실제 물체·최종 목표는 유지한다.

원본 `/home/hwanhee/jhh/TokenHSI_stage12_bridge_fix_reward/output/stage2/Stage2Distill900V4_00002000.pth`의 네트워크·가중치·입력 RMS를 그대로 로드한다. `--checkpoint`로 다른 위치의 원본 Distill900V4를 지정할 수 있다. 센서는 환경에서 계산하지만 정책에는 기존 1272차원만 입력한다. 후행 alias를 원래 최종 목표로 처리한 뒤 선행 payload의 GTA XYZ만 문 입구 중앙으로 옮긴다. 높이는 apron 바닥+실제 payload 반높이이며 크기·회전은 유지한다. 원래 관측과 수정 관측으로 각각 행동을 계산하고 선행 행동만 수정 출력으로 교체한다. 후행과 PUSH 두 명은 원래 관측의 출력을 사용한다.

진입 단계에서는 선행 root XY와 입구 중심의 거리가 `실제 payload XY 반대각 + --switch-margin` 이하가 되면 실제 상자 관측으로 복귀한다. 이후 기존 HOLDING 성공 판정이 성립하면 선행의 GTA Target XYZ만 같은 입구 좌표로 바꾼다. 복귀 단계에서는 **운반 상자 중심 XY**가 같은 반경 안에 들어오면 원래 최종 Target으로 복귀한다. HOLDING은 기존 관계 성공 기준이며 별도 상자 상승 높이 조건을 추가하지 않는다. 두 전환은 각각 에피소드 동안 유지하며, 놓치더라도 목표 단계를 되돌리지 않고 reset 때 모두 초기화한다. 집는 순간 이미 반경 안이면 즉시 최종 Target을 사용한다. 두 단계 모두 원래 최종 목표 기준 후행 alias를 먼저 처리하고 선행 행동만 교체한다. `--switch-margin` 기본 1m, 0 이상 0.5m 단위이며 상한은 없다. 큰 값에서는 초기부터 전환될 수 있고, 문 통과보다 먼저 전환될 수도 있다. 후행 waiting penalty의 1.5m와 별개다.

스폰은 앞벽 안쪽 면을 x=0, 실내를 +x로 한 방 좌표 기준이다. PUSH 두 명은 기존 x=-1.85m·y=±0.9m를 유지한다. 선행은 x=[-3.5,-2.5]m·|y|=[2.4,3.2]m에서 실제 payload가 있는 쪽을 택한다. 후행은 기존 x=[-3.5,-0.9]m·y=[-3.6,3.6]m에서 다시 샘플링한다. 사람 사이 0.85m·상자 외곽 0.5m 여유와 기존 접근 거리·near 경로 조건을 검사하며, 실패하면 배치를 재시도한다. 선행 상자는 외곽과 방의 앞·뒤·측벽 사이에 최소 1.2m 여유가 있는 배치만 사용하며, 부족하면 전체 배치를 다시 샘플링한다(기반 YAML의 측벽 최소 여유는 0.2m). 최종 분포는 이 조건들에 의해 제한된다. 후행 과제는 크기가 허용하는 SIT/CLIMB/ON_TOP 중 균등 샘플링한다.

저장소 루트에서 실행하며 기본 conda는 `tokenhsi`, 1환경·900step·100episode·seed42다. 실행 전 `nvidia-smi`로 점유를 확인한다. 사용자 지정으로 이 viewer는 GPU5/6/7을 허용하며 다른 GPU는 거부한다.

```bash
# stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress: 원본 V4 가상 상자 VNC
TOKENHSI_GPU=7 PORT=6081 bash tokenhsi/scripts/multi_agent/warehouse_waypoint_vnc.sh --switch-margin 1.0

# stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress: 로컬 viewer
TOKENHSI_GPU=7 HEADLESS=0 bash tokenhsi/scripts/multi_agent/warehouse_waypoint_test.sh --switch-margin 0.5

# stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress: 짧은 headless 확인
TOKENHSI_GPU=7 HEADLESS=1 bash tokenhsi/scripts/multi_agent/warehouse_waypoint_test.sh --envs 8 --episodes 1 --steps 120 --switch-margin 1.0
```

`TOKENHSI_GPU=7`을 5 또는 6으로 바꾸면 CUDA와 VNC 렌더링 모두 같은 GPU를 선택한다. 기본 `WAYPOINT_CUDA=auto`는 기존 MPS 연결을 검증하고, 연결할 수 없으면 전용 미생성 MPS 경로로 직접 CUDA를 사용한다. MPS 서버나 기존 프로세스는 시작·중단·수정하지 않는다. `WAYPOINT_CUDA=mps`는 검증 실패 시 종료하고 `WAYPOINT_CUDA=direct`는 직접 CUDA를 지정한다. GPU6의 오래된 MPS 기록이 있는 현재 머신에서도 직접 CUDA로 실행할 수 있다.

VS Code에서 6081 포트를 전달하고 `http://localhost:6081/vnc.html?autoconnect=1&resize=remote`를 연다. 기본 카메라는 입구와 실내를 위에서 내려다보는 탑뷰다. 입구 윤곽·원은 진입 가상 상자 단계에서 자홍색, 집은 뒤 입구 가상 목표 단계에서 주황색, 원래 관측 사용 중에는 회색이다. 청록색 십자는 실제 payload, 초록색 십자는 최종 목표다. 닫힌 상태에서는 큰 문 상자가 가상 상자를 가릴 수 있다. Ctrl+C 또는 viewer 창 종료로 멈춘다.

`--seed`, `--envs`, `--episodes`, `--steps`로 실행 크기를 바꾸고 `--capture`로 시작·전환 화면을 저장한다. `--out`은 `output_etc/check/` 아래만 허용한다. 기본 저장 위치는 `output_etc/check/warehouse_waypoint/<실행시각>/`이며 `results.json`에 진입 전환 `switch_step`·집기 `pickup_step`·최종 목표 복귀 `return_step`(미발생 -1)·반경·종료·마지막 AT 상태, `resolved_env.yaml`에 기반 환경 설정을 저장한다. 강제 장면 선택은 결과의 `variant=push_before`, `start=closed`에 기록한다. 이 viewer 결과는 정식 성공률 평가가 아니다.

## Warehouse waypoint 센서 학습

[학습·관측·캐시 명세](reference/warehouse_waypoint_training.md).

`stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint`는 기존 doorprogress의 보상과 센서 학습 범위를 유지하면서, 80% PUSH 2명+BEFORE 1쌍의 선행에 입구 가상 상자→실제 상자→입구 가상 목표→최종 목표를 적용한다. 20% BEFORE 2쌍은 기존 apron 배치·실제 입력을 사용한다. 반경은 진입(root XY)·복귀(상자 중심 XY) 모두 **고정 1.3m**이며 상자 크기를 더하지 않는다. HOLDING 성공 뒤 목표가 바뀌고, 전환은 부분 reset 대상에 대해서만 초기화한다. 최종 목표 기준 후행 alias를 먼저 처리하며 실제 목표·보상·게이트는 그대로다.

전이는 `checkpoints/Stage2Distill900V4_00002000.pth`에서 시작한다. 4명/1024환경/8토큰/900step·seed42·Default/closed다. Actor encoder/coordination/sigma·입력 RMS는 동결하며 head·critic·센서·AMP discriminator를 학습한다. 관측10493차원은 기존10488차원 뒤에 선행ID·단계·입구 로컬XYZ를 붙인다. 추가5차원은 정규화하지 않으며 PPO 수집·재계산·평가 모두 저장된 값을 사용한다. 4명 모두 정책 업데이트의 영향을 받는다.

새 실험 wrapper는 **GPU6만 허용**하고 기본 conda는 `tokenhsi`다. 기존 MPS가 검증되면 연결하고, 없으면 직접 CUDA로 실행한다. 명시적 `WAYPOINT_CUDA=direct`도 지원한다. MPS 서버·기존 학습은 변경하지 않는다. 기본 학습/추론/VNC 명령은 [config](config.md) 맨 아래를 따른다. 확인 학습 예:

```bash
TOKENHSI_GPU=6 CHECK=1 MAX_ITERATIONS=3 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint_train.sh
```

새 checkpoint로 resume하려면 같은 train wrapper에 `RESUME_CHECKPOINT="$CKPT"`를 설정한다. 학습·평가 환경은 checkpoint의 저장 계약을 사용하며 기존 센서 checkpoint는 새 실험 resume/test에 혼용하지 않는다. 정식 평가는 전용 `*_eval.sh`를 사용한다. `WAREHOUSE_VARIANT=mixed|push_before|before_pairs`는 평가 전용이며 기본 mixed다. 가상 유도는 before_pairs에서 항상 꺼진다.

학습 배치 캐시는 두 변형별 환경당32개 조건·조건당2개 배치를 저장하고, 적중 시25% 확률로 새 배치를 만든다. 과제/역할·near·상자 크기·기하·시작 조건을 확인하고 물리 상태·그래프·AMP 초기 이력을 함께 복원한다. 배치 재사용에 따라 초기 장면 반복은 늘어난다. 첫 학습 전에 두 변형 각각32회 준비하며 RNG·샘플링 계수를 복원한다. 평가에서는 캐시를 사용하지 않는다. `warehouse/layout_cache/{push_before,before_pairs}/`에 요청·적중·갱신, 공통 bytes·warmup_seconds와 `warehouse/waypoint/` 단계 비율을 기록한다.

역할별 GTA 입력은 분리된다. PUSH 두 명과 후행은 기존 장면의 출력, 선행만 입구 가상 좌표를 적용한 장면의 출력을 사용한다. 후행용 alias는 원래 최종 목표에 있고 선행의 가상 상자/목표와 별도 토큰이다. 실제/선행용 장면을 별도 batch 행으로 묶어 encoder 호출을 한 번으로 합치므로, 선행의 가상 좌표가 PUSH 출력에 들어가지 않는다. 두 장면의 연산량 자체가 사라지는 것은 아니다. 동결 actor 특징은 PPO mini-epoch 간 재사용하여 반복 transformer 계산을 줄인다. 기존 명령·checkpoint를 그대로 사용하며 적용은 새 프로세스부터다. 실행 중 학습은 자동으로 바뀌지 않는다.

`WAYPOINT_RESET_BASELINE=1`은 성능 진단 전용이다. 같은 가상 입력 학습을 기존 viewer의 중복 배치·기존 BEFORE 전용 캐시·전체 상태 복사로 실행해 비교한다. 정상 학습에서는 해제한다. 검증은 `output_etc/check/warehouse_waypoint_training/`에 저장한다. 기존 제로샷 viewer의 `--switch-margin` 공식과 CLI는 변경하지 않았다.

### Waypoint GPU6 백그라운드 본학습

2026-11-18 GPU6 MPS waypoint 서비스는 중단했다. 재개에 쓸 마지막 정기 저장본은 **4000epoch**다. 아래는 기존 실행 기록이다. [중단·저장본 확인](../output_etc/check/warehouse_ontop_waypoint/stopped_training.json).

사용자 요청으로 `tokenhsi-warehouse-waypoint-mps-gpu6-20261118-020124.service`를 user systemd `app.slice`·`Restart=no`·`Linger=yes`에서 실행한다. 대화·터미널·로그아웃과 독립적으로 계속 실행한다. GPU6 MPS·conda `tokenhsi`·4명/1024환경/8토큰/900step/seed42·headless, Distill900V4 2000 fresh 전이다. CHECK/MAX_ITERATIONS/RESUME_CHECKPOINT·전이/환경 override와 평가용 환경변수는 해제했다. 기존 GPU6 doorprogress MPS 서비스는 유지한다.

```bash
# 서비스 상태
systemctl --user status tokenhsi-warehouse-waypoint-mps-gpu6-20261118-020124.service

# 학습 로그
tail -f output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint/launch_mps_gpu6_20261118-020124/train.log
```

시작 확인 당시 learner PID는2518467이었다. 운영 스크립트는 [start.sh](../output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint/launch_mps_gpu6_20261118-020124/start.sh), 실제 argv·PID·코드 hash는 [실행 정보](../output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint/launch_mps_gpu6_20261118-020124/launch.json)에 기록한다. 학습 결과·checkpoint는 같은 output 루트 아래 시간별 새 run에 저장한다. 상태 검증은 `output_etc/check/waypoint_background_launch/`다.

안전 종료는 서비스의 learner PID·cwd·cgroup과 GPU6 MPS client를 확인한 뒤 대상 client만 `terminate_client`로 분리하고 해당 서비스만 중지한다. 기존 doorprogress client와 MPS 서버는 유지한다. 이 `start.sh`는 fresh 학습 명령이므로 중단된 학습의 이어하기에는 저장 checkpoint를 지정한 별도 resume 실행을 사용한다.

## Warehouse ONTOP 후행 waypoint 센서 학습

`stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint_ontop`은 기본 `tokenhsi` 환경에서 DistillV4 2000으로 새로 전이한다. PUSH 2명 + BEFORE 1쌍의 선행과 ON_TOP 후행이 각자의 GTA 장면에서 같은 입구 좌표를 사용한다. 진입과 운반 목표 복귀 반경은 모두 **1.2m**다. Warehouse의 BEFORE 2쌍 샘플링은 20%이며 두 역할의 입구 유도를 끈다. 기본 학습·로컬 테스트·VNC 명령은 [config](config.md)를, 관측 및 캐시 계약은 [학습 명세](reference/warehouse_waypoint_training.md#12m-선행on_top-후행)를 따른다.

짧은 확인 학습 및 정식 평가 명령은 저장소 루트에서 실행한다.

```bash
TOKENHSI_GPU=0 WAYPOINT_CUDA=mps CHECK=1 MAX_ITERATIONS=3 \
  bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint_ontop_train.sh 4 1024 8

TOKENHSI_GPU=0 WAYPOINT_CUDA=mps \
  bash tokenhsi/scripts/multi_agent/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint_ontop_eval.sh "$CKPT" \
    --output output_etc/check/warehouse_waypoint_ontop_eval
```

테스트의 `CKPT`와 재개의 `RESUME_CHECKPOINT`에는 이 실험에서 저장한 체크포인트를 지정한다. 역할별 단계 비율은 `warehouse/waypoint/{leader,follower}/`, 초기 배치 캐시는 `warehouse/layout_cache/` 아래에 기록한다. 평가의 `WAREHOUSE_VARIANT` 기본값은 `mixed`이며 `push_before` 또는 `before_pairs`로 고정할 수 있다.

### Epoch 500 백그라운드 재개

GPU0/MPS·4명·1024환경·8토큰·seed42로 epoch 500의 모델·옵티마이저를 복원했다. 서비스는 user systemd의 `app.slice`에서 실행하며 `Linger=yes`를 사용한다. [학습 로그](../output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint_ontop/resume500_final_regions_20261118-161139/train.log).

```bash
systemctl --user status tokenhsi-warehouse-ontop-resume500-20261118-161139.service
tail -f output/stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_waypoint_ontop/resume500_final_regions_20261118-161139/train.log
```

종료 시 서비스 cgroup과 learner의 MPS client를 확인하고, 해당 client를 분리한 뒤 서비스를 중단한다.

### ONTOP 후행 입구 유도 추론

DistillV4 2000의 PUSH 2명 + BEFORE 1쌍 장면에서 `--follower-ontop-waypoint`를 켜면 ON_TOP 후행의 GTA 상자 위치가 선행과 같은 입구 좌표에서 시작한다. 후행 root가 입구 중심의 XY 반경 1.2m에 닿으면 실제 상자로 전환한다. 후행이 실제 상자를 들면 GTA의 ON_TOP 목표 alias를 입구로 옮기고, 들고 가는 상자가 입구 1.2m 이내에 닿으면 원래 쌓기 목표로 복귀한다. 이 모드에서 선행과 후행은 별도 GTA 장면 행과 같은 전환 반경을 사용한다. `--ontop-only`는 시각 확인용으로 후행 과제만 ON_TOP으로 고정한다.

`--record`는 viewer에서 단일 환경의 한 에피소드를 MP4로 저장한다. 다음은 저장소 루트의 Xvfb 렌더 예시다.

```bash
TOKENHSI_GPU=6 WAYPOINT_CUDA=mps HEADLESS=0 xvfb-run -a -s '-screen 0 1600x900x24 -nolisten tcp' \
  bash -c 'source tokenhsi/scripts/multi_agent/gui_gpu_env.sh && exec bash tokenhsi/scripts/multi_agent/warehouse_waypoint_test.sh "$@"' render \
  --follower-ontop-waypoint --ontop-only --seed 42 --envs 1 \
  --episodes 1 --steps 900 --record --out output_etc/check/warehouse_waypoint_ontop/zeroshot/seed42
```

`results.json`에서 후행의 실제 상자 전환, 방 진입, HOLDING, 가상 목표 전환과 복귀, ON_TOP 성공의 첫 step을 확인할 수 있다. 영상의 자홍 원은 선행, 파란 원은 ON_TOP 후행, 노란 입구 원은 후행의 가상 목표, 초록 십자는 원래 쌓기 목표다.
