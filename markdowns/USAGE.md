# 설치와 추가 실행 옵션

기본 명령은 [config](config.md), 구현 위치는 [structure](structure.md)를 따른다.

## 실행 환경

모든 wrapper는 `runtime_env.sh`로 기본 conda `tokenhsi_jhh`을 활성화하고 저장소 루트에서 실행한다. 다른 환경은 `TOKENHSI_CONDA_ENV=<이름>`, conda 설치 위치는 `CONDA_BASE`로 명시한다.

`TOKENHSI_GPU=0`처럼 물리 GPU를 한 번 지정한다. CUDA·기존 MPS·viewer Vulkan에 같은 GPU를 전달한다. `mps/`는 기존 daemon을 검사하며 실행 wrapper가 daemon을 시작하거나 종료하지 않는다.

PyTorch는 서버 GPU/CUDA 호환 설치본을 사용하며 Isaac Gym은 torch보다 먼저 import한다. `requirements.txt`는 남은 의존성이다. 현재 서버의 conda에는 Isaac Gym과 CUDA 라이브러리 활성화 설정이 있다.

## 데이터와 체크포인트

`tokenhsi/data/dataset_loco_sit_carry_climb.yaml`이 참조하는 `dataset_{amass_loco,amass_climb,sit,carry}/motions`를 공유 데이터에 심링크로 연결한다. 외부 데이터와 원본 checkout은 읽기 전용이다.

현재 네 모션 링크는 `/home/user/jhh/Projects/TokenHSI/tokenhsi/data/` 아래 각 데이터셋을 가리킨다. `tokenhsi/data/jointpush_motions`는 기준 repo `/home/user/jhh/Projects/stage_1_2_base_fuv1/tokenhsi/data/jointpush_motions`에 연결되어 있다. 해당 디렉터리의 PUSH 모션 링크는 `TokenHSI_stage1_push`의 데이터를 읽는다.

현재 사용 대상 `stage2_distill_900`의 전이 가중치는 `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`이며 `TokenHSI_stage2_before_et_rsi/output/stage1/`의26k PTH에 연결되어 있다. 다른 BEFORE 설정의78k 전이 경로는 이 checkout에서 준비하지 않았다.

BEFORE checkpoint는 해당 실험 학습으로 생성한다. fixed 추론에는 기존 run의 `relation_config.yaml`도 필요하다. 확인 학습 저장본은 장기 성능 검증을 의미하지 않는다.

모션 로더는 프레임 전체의 관절 속도를 일괄 계산하고 필요한8개 텐서만 GPU에 캐시한다. 모든 속성을 조회하던 캐시를 교체하여 `Error for key= global_root_yaw_rotation` 등의 사용하지 않는 파생 속성 오류도 출력하지 않는다. 모션 목록·가중치·RSI 제외 구간·보간·AMP 입력은 유지한다. 별도 옵션 없이 다음 viewer/test/학습 프로세스부터 적용되며 실행 중인 프로세스에는 소급 적용되지 않는다.

## 설정과 resume

- 새 학습은 선택한 환경·학습 YAML을 읽는다.
- 추론과 `RESUME_CHECKPOINT=<pth>`는 checkpoint 옆 `relation_config.yaml`의 env/sim을 읽는다. 학습 YAML은 해당 실험 것을 사용한다.
- `ENV_CONFIG=<yaml>`은 명시적인 환경 설정이다. fixed/BEFORE 모두 같은 옵션을 쓴다. 저장 설정이 없으면 현재 YAML로 조용히 대체하지 않는다.
- 기존 fixed snapshot에 sim 전체가 없으면 native fixed sim을 보완한다. 부분 sim과 rest_offset 불일치는 거부한다.
- 인원·환경 수·토큰 수·episode 길이·평가 초기화·영상 CLI를 마지막으로 적용하며 실험 계약을 검사한다.
- 모든 wrapper는 기본 `SEED=42`를 쓴다. `EPISODE_LENGTH`는 추론 episode 길이다.

`stage2_sensor_900_bridge_before_fixed_layout_cache` 새 학습은 [config](config.md)의 `ENV_CONFIG` 지정 명령으로 현재 native 설정을 사용한다. 평가와 재개는 checkpoint의 거리·refresh 설정을 유지한다. train/train-VNC wrapper는 `RESUME_CHECKPOINT`가 있고 `ENV_CONFIG`가 없으면 저장 설정을 사용한다. 명시적으로 새 native 설정을 지정해 기존 checkpoint와 다른 학습 분포로 재개하는 것은 checkpoint 계약 검사에서 거부한다.

```bash
env -u ENV_CONFIG TOKENHSI_GPU=5 RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_before_fixed_layout_cache_train.sh 4 1024 8
```

Default bridge 배치 후에는 비활성 후행 Target만 자기 root로 보정한다. 새 실행·재개에서 RAM 캐시도 교정된 값으로 다시 준비한다. 실행 중인 학습에는 소급 적용되지 않으므로 수정 효과를 받으려면 재시작·재개가 필요하다. 배치·캐시 동작은 [캐시 명세](reference/stage2_sensor_900_bridge_before_fixed_layout_cache.md)를 따른다.

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

`stage1_push`는 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%의 사용자 지정 비율을 사용하는 스크래치 학습이다. 기본2명·2048환경·4물체·900step, 같은 `tokenhsi_sm120` runtime을 사용한다. 주어진21k finetune PTH의 일반 평가는 저장된 PUSH50% 분포를 복원하고 시간만900으로 맞춘다. 옆에 `relation_config.yaml`이 없으면 PTH의 `relation_experiment_config`를 읽는다. 서로 다른 분포/horizon의 PTH를 resume하지 않는다.

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

`stage2_900`(일반 RL)·`stage2_distill_900`(두 teacher 모방)은 기존1200step 버전과 별도로 관리한다. 모델·샘플링·보상·학습 방식은 같고 에피소드 제한과 실험/저장 이름을 분리했다. 전용 train/test/VNC 명령은 [config](config.md) 각 목록의 맨 아래에 있다. 현재 사용 대상 `stage2_distill_900`은 기본GPU0이며 숫자/UUID를 지정할 수 있다. `stage2_900`의 기존GPU 제한은 유지한다. 학습·viewer·정식 평가 기본 제한은 모두900step이다.

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

### Stage2 sensor 900

`stage2_sensor_900`은 `output/stage2/Stage2Distill900_00001500.pth`를 전이하는 별도 PPO 실험이다. 기존 distillation 구현은 유지하며 이 실험은 teacher 파일을 사용하지 않는다. Actor encoder·actor Stage2 coordination·sigma·입력 RMS를 동결하고 센서·action head·critic 전체·AMP discriminator 및 AMP/value RMS를 학습한다. Value RMS는 fresh 시작 시 초기화한다. 새 run 이름은 `Stage2Sensor900Square`이며 resume은 같은 학습 계약의 checkpoint만 허용한다. 과거 전체 동결 PTH는 테스트만 가능하다. [샘플링·크기·동결 명세](reference/stage2_sensor_900.md).

학습·테스트·VNC 명령은 [config](config.md) 목록 끝에 있다. 기본 `tokenhsi_jhh`·GPU0·4명·1024환경·8토큰·900step이며 확인 학습은 환경 수를 유지한다. `CHECK=1 MAX_ITERATIONS=1`은 기존 학습 루프 조건상 epoch2까지 수행한다. 본학습 전 확인용 변수를 해제한다.

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

`BRIDGE_ENTRY_CHAMFER=1`은 추론 전용 입구 모따기다. 두 플랫폼의 통로 쪽 양 끝 모서리를 X20cm·Y40cm 삼각형만큼 깎은 convex mesh를 사용하며 최종 배치 구간의 gap은 그대로다. 센서 높이맵·발밑 지형·격자 표시도 같은 모따기를 반영한다. 기본값0이며 학습에서1은 거부한다. 기존 checkpoint로 추론할 수 있고 생성 mesh/URDF는 해당 확인용 출력 경로의 `chamfer_assets/`에 저장한다.

```bash
# 기존 체크포인트로 입구 모따기 VNC 추론
TOKENHSI_GPU=0 SCENARIO=bridge BRIDGE_ENTRY_CHAMFER=1 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_vnc.sh "$CKPT" 4 1 8 10
```

`BRIDGE_ENTRY_INSERT_CM=1.5`은 원본 직각 플랫폼을 쓰는 별도 추론 옵션이다. 로컬 +Y 진입 끝선이 같도록 두 플랫폼을 배치하고 상자 선두가 그 선을1.5cm 지난 상태에서 시작한다. 상자는 gap 중앙에 정렬하고 기존 pusher 간격·heading 규칙과 최종 gap을 유지한다. 기본0, 허용값은1–2cm, 학습에서는 사용할 수 없으며 모따기와 동시 사용도 거부한다.

```bash
# 원본 형상·입구 끝선 정렬·상자 선두1.5cm 삽입
TOKENHSI_GPU=0 SCENARIO=bridge BRIDGE_ENTRY_CHAMFER=0 BRIDGE_ENTRY_INSERT_CM=1.5 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_vnc.sh "$CKPT" 4 1 8 10
```

다리를 목표 위치·각도에 미리 놓은 crosser 정렬/무작위 비교는 [별도 진단](../output_etc/check/bridge_crosser_compare_20261108/report.md)으로 재현한다. 고정4.5k PTH·seed0·64환경에서 목표와 시작 거리를 맞춘5쌍을900step 녹화한다. 다리는 기존 동적 물체이며 좌우10cm 간격을 유지한다. `capture.py`·`analyze.py`·`package.py`와 영상·초기 상태·궤적은 해당 확인 폴더에 있고 학습 YAML은 바꾸지 않는다.

```bash
# 다리 선배치: 무작위/정렬 시작 비교·영상·좌표 검증
bash output_etc/check/bridge_crosser_compare_20261108/run_capture.sh

# tokenhsi_sm120 활성화 후 결과 분석과 좌우 비교 영상 생성
python output_etc/check/bridge_crosser_compare_20261108/analyze.py
python output_etc/check/bridge_crosser_compare_20261108/package.py
```

학습 noVNC는6080, 테스트 noVNC는6081이다. VS Code 포트 전달 후 `http://localhost:6080/vnc.html?autoconnect=1&resize=remote` 또는6081을 연다. 원래 `Stage2Distill900` PTH는 새 센서 네트워크 테스트용 checkpoint가 아니므로 먼저 센서 학습으로 저장한 PTH를 사용한다.

#### Bridge 시작 분포 실험

`stage2_sensor_900_bridge_init`은 cliff5% / bridge55% / room40%, Bridge 내부 연결 완료20%·기존 근접40%·선두1.5cm 삽입40%로 학습한다. 연결 완료 분기에는 crosser의 입구 근접과 출발 무대 운반 상자 근접도 적용한다. 시작 분포·진단 정의는 [명세](reference/stage2_sensor_900.md#bridge-시작-분포-실험)를 따른다. 동결·PPO·보상·Distill1.5k 출발점은 유지한다.

```bash
# GPU0 점유 확인 후 MPS 준비 (새 train/train_vnc는 MPS 필수)
nvidia-smi
source mps/shell.sh
mps_start 0
mps_status 0

# 짧은 학습: 4명·1024환경 유지
TOKENHSI_GPU=0 CHECK=1 MAX_ITERATIONS=1 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_init_train.sh 4 1024 8

# 테스트 / 평가: 이 실험이 저장한 checkpoint 사용
TOKENHSI_GPU=0 HEADLESS=1 EPISODE_LENGTH=16 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_init_test.sh "$CKPT" 4 8 8 1
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_init_eval.sh "$CKPT" --envs 8 --seeds 0 --output output_etc/check/bridge_init_eval

# 같은 실험의 체크포인트에서 재개
TOKENHSI_GPU=0 RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_init_train.sh 4 1024 8
```

새 본학습에서는 CHECK/MAX_ITERATIONS/RESUME_CHECKPOINT/SCENARIO/SENSOR_MASK/BRIDGE_ENTRY_INSERT_CM/BRIDGE_ENTRY_CHAMFER를 해제한다. 새 설정은 YAML로 저장되므로 추론 override를 추가하지 않는다.

#### Bridge 시작 분포 본학습 서비스

이전5/55/40 본학습은 `tokenhsi-stage2-sensor-bridge-mixture-20261108-181129.service`로 실행했으며 curriculum 교체 시 중단했다. GPU0·tokenhsi_sm120·4명·1024환경·8토큰·900step·seed42·Distill1.5k fresh 전이이며 reset 확률은5/55/40이다. 실행 정보는 [launch.json](../output/stage2_sensor_900_bridge_init/launch_mixture_20261108-181129/launch.json), 검증은 [보고서](../output_etc/check/bridge_mixture_05540_20261108/report.md)에 둔다.

`tokenhsi-mps-gpu0.service`가 MPS를 별도로 유지하고, 학습 서비스는 이를 Requires/After로 참조한다. 사용자 systemd·Linger=yes로 대화/SSH 종료와 독립적이다. 이전 균등 샘플링 서비스 `tokenhsi-stage2-sensor-bridge-init-20261108-171737.service`는 MPS `terminate_client` 성공 후 중단했으며 로그·saved config를 보존했다. 해당 run은500epoch 정기 저장 전이어서 PTH가 없었다. 별도 roomopen·다른 GPU 작업과 공유 MPS는 유지했다. 더 이전 square 학습의 정기5k PTH도 보존한다.

```bash
# 상태 및 실제 MPS 접속
systemctl --user status tokenhsi-stage2-sensor-bridge-mixture-20261108-181129.service --no-pager
systemctl --user status tokenhsi-mps-gpu0.service --no-pager
source mps/shell.sh
mps_status 0

# 로그
tail -f output/stage2_sensor_900_bridge_init/launch_mixture_20261108-181129/train.log

# 이 새 학습만 중단: 신원 확인 → MPS 클라이언트 GPU 작업 해제 성공 → 서비스 중단
python output/stage2_sensor_900_bridge_init/launch_mixture_20261108-181129/stop_safely.py
```

Restart=no다. 서비스의 start.sh는 fresh 실행용이므로 서비스 재시작이 checkpoint 재개를 뜻하지 않는다. 재개는 위 RESUME_CHECKPOINT 명령에 같은 실험의 PTH를 지정한다.

#### 본학습 백그라운드 서비스

2026-11-07 수정 본학습은 `tokenhsi-stage2-sensor-square-20261107-170827.service`로 실행한다. 사용자 systemd 서비스이며 `Linger=yes`를 확인하여 대화/SSH 세션 종료와 독립적으로 실행한다. GPU0·4명·1024환경·8토큰·900step·headless, Distill9001.5k에서 fresh 전이하며 확인용 제한·checkpoint·시나리오 고정을 해제했다. YAML의 최대 epoch1000000·중간 checkpoint500epoch 간격을 사용한다. 이전 전체 동결 학습은 중단했으며 결과는 기존 `Stage2Sensor900_*` 경로에 보존한다.

```bash
# 상태
systemctl --user status tokenhsi-stage2-sensor-square-20261107-170827.service --no-pager

# 실시간 로그
tail -f output/stage2_sensor_900/launch_square_20261107-170827/train.log

# 필요할 때 이 학습만 종료
systemctl --user stop tokenhsi-stage2-sensor-square-20261107-170827.service
```

결과와 TensorBoard는 `output/stage2_sensor_900/Stage2Sensor900Square_*/`, 실행 정보는 `output/stage2_sensor_900/launch_square_20261107-170827/launch.json`에 저장한다. 오류 발생 시 자동으로 새 학습을 시작하지 않도록 서비스 `Restart=no`를 사용한다. 재개할 때는 새 학습 계약으로 저장된 센서 checkpoint와 기존 resume 명령을 사용한다. 서버 재부팅 뒤 자동 재개하는 서비스는 아니다.

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

본학습 전 `CHECK`·`MAX_ITERATIONS`·`RESUME_CHECKPOINT`·`OUTPUT_PATH` 잔여 설정을 해제한다. `stage2_distill` wrapper는 GPU5/6만 허용하며 생략 시6이다. `--coupled_assignment random|ordered|distance|optimal`로 eval 배정 방식을 지정할 수 있고 기본은 ordered다. `--agents`/`--objects`는 통합 `stage2`·`stage2_distill` eval에서만 받으며 N>4는 `--group_spawn_over4`가 필요하다. 평가 seed 기본0/1/2·환경 수512·1200step이며 인원/물체 수를 바꾼 결과는 기본 학습 분포 평가와 구분한다.


### Bridge 횡단 curriculum

`stage2_sensor_900_bridge_curriculum`은 Bridge70%를 PUSH1명/crosser3명 횡단 연습,30%를 기존 협력 분기로 사용한다. 시나리오는5/55/40, 모델·센서·보상·동결은 유지한다. [명세](reference/stage2_sensor_900.md#bridge-횡단-curriculum), [검증](../output_etc/check/bridge_curriculum_20261109/report.md).

```bash
# MPS 연결을 확인한4명·1024환경 짧은 학습
TOKENHSI_GPU=0 CHECK=1 MAX_ITERATIONS=1 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_curriculum_train.sh 4 1024 8

# 기본 협력 분기 추론 / 횡단 연습의 입구 시작 추론
TOKENHSI_GPU=0 HEADLESS=1 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_curriculum_test.sh "$CKPT" 4 8 8 1
TOKENHSI_GPU=0 HEADLESS=1 SCENARIO=bridge BRIDGE_CURRICULUM_EVAL=walk_entry bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_curriculum_test.sh "$CKPT" 4 8 8 1

# 정식 평가 / 같은 실험 checkpoint 재개
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_curriculum_eval.sh "$CKPT" --envs 8 --seeds 0 --output output_etc/check/bridge_curriculum_eval
TOKENHSI_GPU=0 RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_curriculum_train.sh 4 1024 8
```

`BRIDGE_CURRICULUM_EVAL`은 추론 전용이며 기본 cooperative다. walk_entry/training의 길이는 각1/3이며 training만 중간 초기화를 포함한다. 새 본학습에서는 CHECK/MAX_ITERATIONS/RESUME_CHECKPOINT/OUTPUT_PATH/SCENARIO/SENSOR_MASK/BRIDGE_CURRICULUM_EVAL을 해제한다.

#### Curriculum 본학습 서비스

`tokenhsi-stage2-sensor-bridge-curriculum-resume-20261109-151548.service`는 아래 Room BEFORE 실험으로 교체되어 중단했다. 해당 서비스는 사용자 systemd·app.slice·Linger=yes·Restart=no로 실행한다. GPU0·tokenhsi_sm120·4명/1024환경/8토큰/900step·seed42이며 같은 실험의500epoch checkpoint에서 모델·optimizer·정규화를 재개한다. 대화/SSH 종료와 독립적이다. [재개 정보](../output/stage2_sensor_900_bridge_curriculum/resume_20261109-151548/launch.json), [재개 검증](../output_etc/check/bridge_curriculum_resume_20261109-151548/report.md). 최초 fresh run과 로그는 기존 경로에 보존한다.

```bash
# 현재 서비스 / 로그
systemctl --user status tokenhsi-stage2-sensor-bridge-curriculum-resume-20261109-151548.service --no-pager
tail -f output/stage2_sensor_900_bridge_curriculum/resume_20261109-151548/train.log

# 실제 공유 MPS 클라이언트
source mps/shell.sh
mps_status 0

# 이 실행만 안전 종료: PID/start time/cwd/config/cgroup 확인 후 MPS 해제 성공 시에만 서비스 중단
python output/stage2_sensor_900_bridge_curriculum/resume_20261109-151548/stop_safely.py
```

공유 MPS와 다른 repo 작업은 유지한다. 현재 resume 서비스의 start.sh는500epoch를 지정하므로 서비스 재시작도500epoch부터다. 다음 재개는 최신 저장 checkpoint를 확인해 위 RESUME_CHECKPOINT에 지정한다. 모델·optimizer·epoch/frame·정규화는 복원하며 시뮬레이션 에피소드와 replay는 초기화될 수 있어 비트 단위 연속 실행은 아니다.

### Room BEFORE 균형과 1000epoch 재개

`stage2_sensor_900_room_before`는 Room BEFORE 생성 분기를 두 쌍으로 만들고 독립 PUSH만 내부의 비PUSH 과제로 전환한다. Room 외부 COUPLED2/3/4명, Bridge 횡단70%/기존 협력30%, 시나리오5/55/40과 모델·PPO·동결은 유지한다. [명세](reference/stage2_sensor_900.md#room-before-균형), [검증](../output_etc/check/room_before_20261109/report.md).

```bash
# 기존 curriculum 또는 이 실험의 checkpoint를 모델/optimizer/통계와 함께 재개
TOKENHSI_GPU=0 RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/stage2_sensor_900_room_before_train.sh 4 1024 8

# 확인 학습: MAX_ITERATIONS는 추가 횟수가 아닌 절대 epoch 상한
TOKENHSI_GPU=0 CHECK=1 MAX_ITERATIONS=1000 RESUME_CHECKPOINT="$CKPT_1000" bash tokenhsi/scripts/multi_agent/stage2_sensor_900_room_before_train.sh 4 1024 8

# 학습 시각화 / 정식 평가
TOKENHSI_GPU=0 RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/stage2_sensor_900_room_before_train_vnc.sh 4 1024 8
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_room_before_eval.sh "$CKPT" --envs 8 --seeds 0 --output output_etc/check/room_before_eval
```

Train/train_vnc는 새 환경 YAML을 명시하며 기존 curriculum에서 Room sampling 두 설정만 달라지는 resume을 허용한다. 새 checkpoint를 이용한 추론은 새 sidecar 설정을 따른다. 이전 checkpoint와 로그는 보존하며 시뮬레이션 에피소드·replay는 재초기화될 수 있다.

#### Room BEFORE 본학습 서비스

`tokenhsi-stage2-sensor-room-before-20261109-163842.service`는 GPU0·tokenhsi_sm120·4명/1024환경/8토큰/900step·seed42, 사용자 systemd·app.slice·Linger=yes·Restart=no로 실행한다. 원래 curriculum의1000epoch/frame131,072,000에서 재개했고 CHECK/MAX_ITERATIONS는 없다. 공유 MPS daemon374777/server374795를 유지한다. [실행 정보](../output/stage2_sensor_900_room_before/resume_20261109-163842/launch.json).

```bash
# 상태 / 로그
systemctl --user status tokenhsi-stage2-sensor-room-before-20261109-163842.service --no-pager
tail -f output/stage2_sensor_900_room_before/resume_20261109-163842/train.log

# 실제 공유 MPS 클라이언트
source mps/shell.sh
mps_status 0

# 이 실행만 종료: PID/cwd/config/cgroup/MPS 신원 확인 후 해당 클라이언트 해제
python output/stage2_sensor_900_room_before/resume_20261109-163842/stop_safely.py
```

이 서비스의 start.sh는1000epoch를 지정하므로 재시작도 같은 checkpoint에서 시작한다. 이후 재개는 최신 저장본을 확인해 위 RESUME_CHECKPOINT로 지정한다.

### Room BEFORE 복구 재개 서비스

이 서비스는 rollout 경계 reset 수정 후 중단했다. 현재는 [2500epoch 재개 서비스](#rollout-reset-수정-2500epoch-재개-서비스)를 따른다.

2026-11-09 19:15 배치 재시도 소진으로 종료한 run을 최신2000epoch에서 재개했다. `stage2_sensor_900_room_before`의 분포/보상/동결은 동일하다. 공유 MPS daemon/server를 종료하지 않는다.

```bash
systemctl --user status tokenhsi-stage2-sensor-room-before-resume-20261109-194501.service

tail -f output/stage2_sensor_900_room_before/resume_recovery_20261109-194501/train.log

# 이 서비스의 cwd/config/PID와 MPS client를 확인한 후 해당 learner만 종료
python3 output/stage2_sensor_900_room_before/resume_recovery_20261109-194501/stop_safely.py
```

[실행 정보](../output/stage2_sensor_900_room_before/resume_recovery_20261109-194501/launch.json), [상태 복원·배치 복구 검증](../output_etc/check/sensor_resume_recovery_20261109/report.md). 사용자 systemd·Linger=yes로 대화 종료와 독립적이며, 오류 자동 재시작/재부팅 자동 시작은 등록하지 않았다.

### Rollout reset 수정 2500epoch 재개 서비스

현재3시나리오 `stage2_sensor_900_room_before`는 rollout 마지막 step의 done을 다음 rollout 시작에 처리하도록 수정하고, 지정한2500epoch PTH에서 모델·optimizer·입력/value/AMP RMS를 복원했다. 기존 샘플링·보상·동결·900step·curriculum 진행을 유지한다. Roomopen은 이번 수정 대상이 아니며 실행 중인 학습을 보존했다.

```bash
systemctl --user status tokenhsi-stage2-sensor-room-before-reset-20261109-230256.service

tail -f output/stage2_sensor_900_room_before/resume_rollout_reset_20261109-230256/train.log

# cwd/config/PID start time/MPS client 확인 후 현재 learner만 종료
python3 output/stage2_sensor_900_room_before/resume_rollout_reset_20261109-230256/stop_safely.py

# 수동 재개용: 현재 서비스와 중복 실행하지 않는다
TOKENHSI_GPU=0 RESUME_CHECKPOINT='output/stage2_sensor_900_room_before/Stage2Sensor900RoomBefore_09-19-45-10/nn/Stage2Sensor900RoomBefore_00002500.pth' bash tokenhsi/scripts/multi_agent/stage2_sensor_900_room_before_train.sh 4 1024 8
```

[실행 정보](../output/stage2_sensor_900_room_before/resume_rollout_reset_20261109-230256/launch.json), [시작 검증](../output_etc/check/rollout_reset_fix_20261109/startup_verification.json), [수정·검증](../output_etc/check/rollout_reset_fix_20261109/report.md). 사용자 systemd·Linger=yes·app.slice·기존 MPS Requires/After로 대화 종료와 독립적이다. 오류 자동 재시작·재부팅 자동 시작은 등록하지 않았다. 공유 MPS daemon/server를 직접 종료하지 않는다.


## Cliff/Bridge 전용 variant

`stage2_sensor_900_cliff_bridge`는 Cliff40%/Bridge60%와 외력4명 횡단·BEFORE를 사용하는 별도 실험이다. [상세 분포·배치·지표](reference/stage2_sensor_900_cliff_bridge.md). 기존 실험의 코드·체크포인트는 유지한다. 현재 본학습 서비스는 [Cliff/Bridge 본학습](#cliffbridge-본학습-서비스)을 따른다.

기본 전이는 `output/stage2/Stage2Distill900_00001500.pth`다. 학습 시작 epoch는0이며 원본1500epoch는 전이 출처다. critic/value 정규화 학습·actor encoder/coordination 동결 규칙은 기존 센서 실험과 같다.

```bash
# 짧은 학습:1024환경 유지, GPU 여유 확인 후 실행
TOKENHSI_GPU=0 CHECK=1 MAX_ITERATIONS=2 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_train.sh 4 1024 8

# 본학습을 대화/SSH 세션과 분리하여 시작할 때 사용
systemd-run --user --unit=tokenhsi-stage2-cliff-bridge --slice=app.slice \
  --property=Restart=no --working-directory="$PWD" --setenv=TOKENHSI_GPU=0 \
  bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_train.sh 4 1024 8

# 시작 상태·로그 확인
systemctl --user status tokenhsi-stage2-cliff-bridge.service --no-pager
journalctl --user -u tokenhsi-stage2-cliff-bridge.service -f

# 같은 variant의 체크포인트 재개: optimizer/정규화/진행도 복원
TOKENHSI_GPU=0 RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_train.sh 4 1024 8

# 외력4명 횡단 VNC / Cliff 화면 없는 추론
TOKENHSI_GPU=0 SCENARIO=bridge BRIDGE_MODE=controlled bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_vnc.sh "$CKPT" 4 8 8 1
TOKENHSI_GPU=0 HEADLESS=1 SCENARIO=cliff bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_test.sh "$CKPT" 4 8 8 1

# 정식 평가: 필요할 때 별도 실행
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_eval.sh --checkpoint "$CKPT" --envs 8 --seeds 0
```

본학습 전 CHECK/MAX_ITERATIONS/OUTPUT_PATH/RESUME_CHECKPOINT의 잔여 override를 해제하고, 사용자 `Linger=yes`와 공유MPS 가동을 확인한다. train/train-VNC는 기존MPS에 연결하며 daemon을 종료하지 않는다. 실행 중인 학습을 바꿀 때는 대상 서비스·Python cwd/config·MPS client를 확인하여 해당 client만 종료한다. 공유MPS나 이름이 같은 다른 Python을 일괄 종료하지 않는다.

추론의 `BRIDGE_MODE` 기본값은 training(40/30/30)이며 controlled/free/cooperative를 강제할 수 있다. SCENARIO는 cliff/bridge만 허용한다. cooperative 다리 시작은 `BRIDGE_START=present|near|insert`(기본 training: 20/40/40)로, pusher 수(=다리 길이 1.6m 단위)는 `BRIDGE_PUSHERS=1|2|3`(기본 균등)으로 고정할 수 있다. 둘 다 추론 전용이며 cooperative 분기에만 적용된다. 예: `TOKENHSI_GPU=0 SCENARIO=bridge BRIDGE_MODE=cooperative BRIDGE_START=insert BRIDGE_PUSHERS=2 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_compact_vnc.sh "$CKPT" 4 8 8 1`. 이전 실험의 PTH를 새 variant로 resume/evaluate하지 않는다. `output_etc/check/`의 `initialization_only.pth`는 학습 전 초기 모델의 추론 검사 전용이며 optimizer가 없어 학습 재개에 쓰지 않는다.

정성 장면 탐색(추론 전용, 학습 코드 미변경): `tokenhsi/qualitative/bridge_before_search.py`는 compact 체크포인트로 pusher 3명 + 선행 AT(출발 무대 상자 집고 다리 건넘) + 후행(도착 무대에서 목표 4–6m에서 출발해 근방 대기 후 CLIMB/SIT) 5명 Bridge 장면을 런타임 패치로 만들어 반복 실행하고, 무낙상·연결·건넘·AT·후행 근접 대기·행동 유지 조건을 통과한 에피소드를 `--out/pool`에 저장한다. `--follower_task climb|sit`, `--agents 4|5`, `--pushers 1|2|3`, `--follower_distance`, `--near_radius`, `--min_wait_frames`, `--hold_frames`로 조절한다. `tokenhsi/qualitative/render_candidates.py`가 pool을 순위화해 `good_XX_batchBBBB_envEEE`(trajectory.npz·metadata.json·contact_sheet.jpg·preview.mp4)를 만든다. 출력은 `output_etc/check/` 아래만 허용한다. 예: `TOKENHSI_GPU=7 bash tokenhsi/scripts/multi_agent/bridge_before_qual.sh --checkpoint "$CKPT" --out output_etc/check/qual_climb/search --num_envs 512 --rounds 10 --seed 11`


## Bridge 공동 PUSH 2명·AT→SIT 영상 내보내기

CliffBridge5000epoch의 추론 전용 배치 탐색과 실제 관절/강체·장면·Blender/MP4 산출물은 [영상·데이터 사용법](../output_etc/check/bridge_enable_push2_20261111/README.md)을 따른다. 다리는 입구1.5cm 삽입, AT/payload와 SIT는 반대 무대에서 시작하며 학습 샘플러와 기존 학습은 유지한다. 선택한 성공 사례이며 정식 평가 성공률은 아니다.

## 5명 Bridge 공동 PUSH3·AT→SIT 영상 내보내기

5명·공동 PUSH3명·AT→SIT의 추론 전용 내보내기는 [상면 마찰1.0](../output_etc/check/bridge_enable_push3_cap_20261111/README.md), [상면/본체 마찰0.4](../output_etc/check/bridge_enable_push3_cap04_20261111/README.md)를 따른다. 각3개1080p30fps 영상·관절·장면·Blender·ZIP을 저장했다. 학습의3인 다리 규격을 유지하고, SIT의 접근·대기·착석과 선회 없는 횡단을 선별한다. 기존 학습·일반4명 wrapper는 유지한다.

후속 [보행 안정성·마찰 비교](../output_etc/check/bridge_push3_stability_20261111/friction_report.md)는 같은 초기1024개에서 계수10종을 각자 새 프로세스/시뮬레이터의 생성 시 고정했다. GPU 런타임 갱신은 실제 물리에 반영되지 않아 사용하지 않는다. 선택한 상면0.2·본체0.4에서 AT→SIT까지 완료한 [추가10개 전체/확대 영상·관절·scene·Blender](../output_etc/check/bridge_push3_stability_20261111/README.md)와 [ZIP](../output_etc/check/bridge_push3_stability_20261111/blender_bundle.zip)을 저장했다. 재사용·렌더 명령은 해당 README를 따른다. 학습 코드와 일반4명 wrapper는 유지한다.

## Bridge 고정 높이·상면 마찰 variant

`stage2_sensor_900_cliff_bridge_surface`는 Bridge 신체 관측의 기준 높이를 시작 무대/바닥으로 고정하고, 단일 다리 강체의 윗5mm만 마찰1.0으로 분리한다. 본체/바닥은0.4이며 기존 다리 크기·질량·센서·낙상 판정·샘플링을 유지한다. [설정·호환·검증](reference/stage2_sensor_900_cliff_bridge_surface.md), [기본 명령](config.md)을 따른다. 새 본학습 운영은 [Bridge surface 서비스](#bridge-surface-본학습-서비스)를 따른다.

기본 시작점은 기존과 같은 Distill1500 전이이며 구형 CliffBridge checkpoint의 자동 resume은 거부한다. 새 variant에서 저장한 checkpoint끼리는 `RESUME_CHECKPOINT`로 재개할 수 있다. GPU 메모리 확보 후에만 실행한다.

```bash
# 학습 시각화: 새 variant, noVNC6080
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_surface_train_vnc.sh 4 1024 8

# 화면 없는 추론: 새 variant에서 저장한 checkpoint
TOKENHSI_GPU=0 HEADLESS=1 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_surface_test.sh "$CKPT" 4 8 8 1

# 정식 평가 진입점: 새 variant에서 저장한 checkpoint
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_cliff_bridge_surface_eval.sh "$CKPT" --output output_etc/check/bridge_surface_eval
```

## Bridge surface 본학습 서비스

2026-11-12 사용자 요청으로 기존 CliffBridge Python1921823의 cwd/config/cgroup/start time과 MPS 연결을 확인한 뒤 `terminate_client 374795 1921823` 응답0→해당 서비스만 중단했다. 기존10500epoch checkpoint·로그를 보존했으며, Roomopen2797534·TeamHOI1783512·MPS374777/374795를 유지했다.

새 실험은 `stage2_sensor_900_cliff_bridge_surface`, Run은 `Stage2Sensor900CliffBridgeSurface_12-16-07-28`이다. GPU0/MPS·tokenhsi_sm120·4명/1024환경/8토큰/900step·seed42, 기존 샘플링과 동결 규칙을 유지한다. 새 관측 기준·상면1.0/본체0.4가 학습과 저장 환경 계약에 포함된다.

- 서비스: `tokenhsi-stage2-bridge-surface-fresh-20261112-160350.service`, Python3109910. 사용자 systemd app.slice·Linger=yes·Restart=no로 대화/SSH 종료와 독립적으로 실행한다.
- 전이: 원본 Distill1500과 SHA가 같은 [보존본](../output/stage2_sensor_900_cliff_bridge_surface/launch_20261112-160350/original_Distill1500.pth). 구형10500epoch나 검증용2epoch 모델을 resume하지 않는다.
- [로그](../output/stage2_sensor_900_cliff_bridge_surface/launch_20261112-160350/train.log), [실행 정보](../output/stage2_sensor_900_cliff_bridge_surface/launch_20261112-160350/launch.json), [1024환경 검사](../output_etc/check/bridge_surface_launch_20261112-160350/smoke_verification.json), [시작 검사](../output_etc/check/bridge_surface_launch_20261112-160350/startup_verification.json).

```bash
systemctl --user status tokenhsi-stage2-bridge-surface-fresh-20261112-160350.service --no-pager
tail -f output/stage2_sensor_900_cliff_bridge_surface/launch_20261112-160350/train.log

# 이 학습의 client만 MPS 검증 후 종료
python output/stage2_sensor_900_cliff_bridge_surface/launch_20261112-160350/stop_safely.py
```

1024환경2update에서 동결131개·입력RMS exact, critic 전체/action head/센서/AMP 갱신·optimizer96step·value RMS count524289와 모델/정규화/scalar517개 finite를 확인했다. 본학습 스크립트는 CHECK/MAX_ITERATIONS/resume/시나리오 강제를 해제하고 원본 Distill1500에서 새로 시작한다. 새 학습의 장기 횡단 성능은 아직 확인하지 않았다.

## Cliff/Bridge 본학습 서비스

이전 실행 기록이다. 2026-11-12 사용자 요청으로 이 서비스를 중단하고 위 Bridge surface 서비스로 교체했다. 2026-11-10에는 이전3시나리오 학습만 종료하고 `stage2_sensor_900_cliff_bridge` 본학습을 시작했다. GPU0/MPS·tokenhsi_sm120·4명/1024환경/8토큰/900step·seed42, Cliff40%/Bridge60%이며 Bridge 내부 controlled/free/cooperative40/30/30이다. BEFORE 참여의 agent-reset 기대값은24%다.

서비스는 `tokenhsi-stage2-cliff-bridge-fresh-20261110-140027.service`, Python PID1921823이며 사용자 systemd app.slice·Linger=yes·Restart=no다. 대화/SSH 종료와 독립적이며 재부팅 자동 시작은 등록하지 않았다.

- Run: `output/stage2_sensor_900_cliff_bridge/Stage2Sensor900CliffBridge_10-14-02-02`
- 전이: `output/stage2/Stage2Distill900_00001500.pth`(원본epoch1500,SHA256 `fcadf2297bedf8e3a8d010fd93d045f63dab412f0c85e65d7b89de4cc15a8a1e`). 새학습 epoch0에서 시작하며 검증용3epoch PTH는 사용하지 않는다.
- [로그](../output/stage2_sensor_900_cliff_bridge/launch_20261110-140027/train.log), [실행 정보](../output/stage2_sensor_900_cliff_bridge/launch_20261110-140027/launch.json), [시작 검증](../output_etc/check/cliff_bridge_launch_20261110/startup_verification.json), [1024환경 기능 검증](../output_etc/check/cliff_bridge_launch_20261110/report.md).

```bash
# 상태와 최근 로그
systemctl --user status tokenhsi-stage2-cliff-bridge-fresh-20261110-140027.service --no-pager
tail -f output/stage2_sensor_900_cliff_bridge/launch_20261110-140027/train.log

# 이 학습만 안전 종료:cwd/config/cgroup/start time/MPS 확인 후 대상 client만 종료
python output/stage2_sensor_900_cliff_bridge/launch_20261110-140027/stop_safely.py
```

actor encoder/coordination/sigma와 입력 정규화는 동결하고 센서/action head/critic 전체/AMP와 value·AMP 정규화는 학습한다.1024환경3update에서 동결131개·입력RMS exact, critic/head/센서 갱신·optimizer144step·value 정규화 갱신 및 scalar517개 finite를 확인했다. 짧은 실행은 기존 루프의 `epoch > max_epochs` 종료 조건으로 MAX_ITERATIONS=2에서3update가 수행됐다. 본학습에는 CHECK/MAX_ITERATIONS/resume/시나리오 강제 설정이 없고 최대epoch1000000·checkpoint500epoch 간격이다.

이전 `tokenhsi-stage2-sensor-room-before-reset-20261109-230256.service`는 중단했다. Python1433539만 `terminate_client 374795 1433539` 응답0 후 서비스 종료했고5500epoch PTH와 로그를 보존했다. Roomopen1534227·TeamHOI1783512·공유MPS374777/374795는 동일 PID/start time을 유지하며 다른 학습 진행도 확인했다. 공유MPS daemon/server나 여러Python을 일괄 종료하지 않는다.

## Bridge 고정 운반 경계 RSI 캐시

`stage2_sensor_900_bridge_before_fixed_carry_rsi` 학습은 `output/rsi_cache_bridge_carry_v1/<signature>.pt`가 있어야 시작한다. 평가·viewer(`--eval`)는 운반 RSI를 끄므로 캐시가 필요 없다. 현재 파일 `803bbe4ba847e26b11adf0cf.pt`는 참고 repo(`TokenHSI_stage12_teacher_sensor_2only`)에서 복사했다. 로드 시 물리·모션·선별 코드 signature를 검사하므로, `bridge_carry_rsi_cache.py`·`bridge_carry_rsi_probe.py`를 바꾸면 다시 생성해야 한다.

```bash
# 캐시 재생성(확인용은 output_etc/check 아래, 본학습용은 output/rsi_cache_bridge_carry_v1)
TOKENHSI_GPU=0 bash -c '. tokenhsi/scripts/multi_agent/runtime_env.sh && python tokenhsi/scripts/multi_agent/prepare_bridge_carry_rsi.py --cache-output output/rsi_cache_bridge_carry_v1'
# 다른 캐시 폴더를 쓸 때
CARRY_RSI_CACHE_DIR=output_etc/check/<검사명>/carry_rsi_cache TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_before_fixed_carry_rsi_train.sh 4 1024 8
```

## DistillV4 Bridge 중간 목표 제로샷 viewer

추론 전용 토이 실험이다. 원본 `Stage2Distill900V4_00002000.pth`의 네트워크·가중치·입력 RMS를 직접 사용하며 센서 입력과 학습은 사용하지 않는다. 장면은 layout-cache YAML의 물리 설정을 기반으로 4명·8토큰·Default 자세·crossing 100%·다리 present로 만든다. 평가 reset이므로 배치 캐시는 준비하지 않는다. 선행 사람·운반 상자·건너편 실제 목표는 다리 횡단 중심선에 정렬한다. pusher 두 명과 후행 SIT/CLIMB/ON_TOP 균등 과제는 유지한다. 다리는 기존처럼 동적 물체다.

선행에게만 GTA Target을 현재 다리 중앙의 상면 높이+운반 상자 반높이로 보여준다. 후행 alias를 원래 목표로 변환한 뒤 선행 Target pose만 수정한다. 나머지 세 명의 행동은 원래 관측으로 계산한 출력을 사용한다. 실제 목표·보상·성공 판정은 건너편 목표를 유지한다.

운반 상자 중심의 XY 거리가 `상자 XY 반대각 + --switch-margin` 이하가 되면 원래 목표로 한 번 전환하고 에피소드 끝까지 유지한다. `--switch-margin` 기본 1m, 0 이상 0.5m 단위이며 상한은 없다. 예: `0.5`, `1`, `1.5`, `2`, `3`, `5`. 큰 반경에서는 reset 직후 전환될 수 있으며 로그의 `switch step=0`으로 확인한다. 기존 후행 waiting penalty의 1.5m 값은 바꾸지 않는다.

`--no-waypoint`를 추가하면 다리 중앙 가상목표와 반경 전환을 끄고 처음부터 네 사람 모두 원래 목표의 V4 정책 출력을 사용한다. 이때 `--switch-margin`은 동작에 영향을 주지 않고, 자홍색 가상목표·반경 표시도 숨긴다. 원래 목표의 초록색 십자만 표시하며 `results.json`은 `no_waypoint: true`, `switch_step: -1`, `radius: null`로 기록한다. 선행/상자/원래 목표의 중심선 배치·다리 물성·캡 높이·시작 모드는 그대로 유지한다. 인자를 생략하면 기존 가상목표 경로다.

저장소 루트에서 실행한다. 기본 conda는 `tokenhsi`, 기본 viewer는 1환경·900step·100episode이며 Ctrl+C 또는 viewer 종료로 멈춘다. 실행 전 `nvidia-smi`로 GPU 점유를 확인한다.

```bash
CKPT=/home/hwanhee/ksh/stage1_2_base_teacher/output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth

# stage2_sensor_900_bridge_before_fixed_layout_cache: DistillV4 중간 목표 VNC
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/bridge_waypoint_vnc.sh --checkpoint "$CKPT" --switch-margin 1.0

# stage2_sensor_900_bridge_before_fixed_layout_cache: DistillV4 중간 목표 로컬 viewer
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/bridge_waypoint_test.sh --checkpoint "$CKPT" --switch-margin 2.5

# stage2_sensor_900_bridge_before_fixed_layout_cache: 화면 없는 짧은 확인
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/bridge_waypoint_test.sh --checkpoint "$CKPT" --switch-margin 3.0 --headless --envs 8 --episodes 1 --steps 120

# stage2_sensor_900_bridge_before_fixed_layout_cache: 가상목표 없이 원래 목표·추종 캡 5cm·near
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/bridge_waypoint_vnc.sh --checkpoint "$CKPT" --bridge-physics follow_cap --cap-lift 0.05 --bridge-start near --no-waypoint
```

`TOKENHSI_GPU=5`를 6 또는 7로 바꾸면 시뮬레이션과 VNC 렌더링이 함께 해당 GPU를 사용한다. VS Code에서 6081 포트를 전달하고 `http://localhost:6081/vnc.html?autoconnect=1&resize=remote`를 연다. 자홍색 십자·원은 다리 중간 목표·전환 반경, 초록색 십자는 원래 목표이며 전환 후 중간 목표 표시는 회색이다. `--seed 42`, `--episodes 10` 등으로 장면·반복을 지정한다. `--capture`는 viewer의 시작·전환 화면을 저장한다. 결과와 resolved 환경 설정은 기본 `output_etc/check/bridge_waypoint/<실행시각>/`에 저장하며 `--out output_etc/check/<검사명>`으로 변경할 수 있다.

`--bridge-physics`로 다리 접촉 어블레이션을 선택한다. 다리 크기·중심선 배치·V4 checkpoint·목표 전환 규칙은 유지하며, 시작 배치는 `--bridge-start`로 별도 선택한다(기본 present).

| 값 | 고정 여부 | 물성 | 충돌 형상 |
| --- | --- | --- | --- |
| `baseline` (기본) | 동적 | 기존 밀도 3, 본체 마찰 0.4·상면 1.0 | 본체+5mm 상면 캡 |
| `fixed` | 고정 | 기존 물성 | 본체+5mm 상면 캡 |
| `carry` | 동적 | 운반 상자 밀도 100·마찰 1.0 | 본체+5mm 상면 캡 |
| `fixed_carry` | 고정 | 운반 상자 밀도 100·마찰 1.0 | 본체+5mm 상면 캡 |
| `platform` | 고정 | 현재 지형 무대와 같은 생성 옵션·마찰 | 다리 크기의 단일 box |
| `baseline_nocap` | 동적 | 기존 본체 밀도 3·마찰 0.4 | 다리 크기의 단일 box, 캡 제거 |
| `carry_nocap` | 동적 | 운반 상자 밀도 100·마찰 1.0 | 다리 크기의 단일 box, 캡 제거 |
| `follow_cap` | 본체 동적·캡 고정 | 기존 본체 밀도 3·마찰 0.4, 캡 마찰 1.0 | 본체와 상면 5mm 캡을 별도 강체로 분리, 캡이 위치·회전 추종 |

carry 계열은 운반 상자와 **밀도·마찰을 일치**시킨다. 총질량은 다리 부피에 비례한다. 예를 들어 3.2×2.2×1.65m 다리는 기존 34.848kg에서 1161.6kg이 된다. fixed 계열은 `fix_base_link=True`로 만든 강체를 사용하며 매 프레임 자세를 덮어써 고정하는 방식이 아니다. 첫 reset 전 기존 물리 초기화를 거친 뒤 배치하고, 에피소드마다 바뀌는 다리 위치에도 적용한다. platform 조건은 고정·물성·충돌 형상을 함께 바꾸므로 순수한 고정 효과는 baseline↔fixed로 비교한다. 실제 질량·밀도·각 충돌 형상의 접촉 속성은 시작 로그와 `results.json`의 `physics`에 저장한다.

```bash
# stage2_sensor_900_bridge_before_fixed_layout_cache: 고정 다리 제로샷 VNC
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/bridge_waypoint_vnc.sh --checkpoint "$CKPT" --switch-margin 1.0 --bridge-physics fixed
```

마지막 `fixed`를 표의 값으로 바꿔 실행한다. 같은 `--seed`·`--envs`·`--switch-margin`을 사용한다. 같은 VNC 포트에서 조건을 바꿀 때는 기존 viewer를 Ctrl+C로 종료하고 다시 실행한다. 기존 `jhh/TokenHSI_stage12_bridge_fix_reward/output/stage2/Stage2Distill900V4_00002000.pth`와 위 원본 checkpoint는 SHA-256이 동일함을 확인했다. [개별·동시 비교 영상](../output_etc/check/bridge_contact_20261117/index.html)·[접촉 진단과 비교 결과](../output_etc/check/bridge_contact_20261117/report.json).

캡 제거는 다리 크기·질량을 줄이지 않고 전체 크기의 단일 충돌 상자로 교체한다. `baseline_nocap`은 기존 본체 마찰 0.4를 모든 면에 사용하므로 상면 마찰도 1.0→0.4로 바뀐다. 밀도·마찰을 유지하면서 충돌 형상만 비교하려면 `carry`↔`carry_nocap`을 사용한다.

```bash
# stage2_sensor_900_bridge_before_fixed_layout_cache: 캡 제거·기존 본체 물성
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/bridge_waypoint_vnc.sh --checkpoint "$CKPT" --switch-margin 1.0 --bridge-physics baseline_nocap

# stage2_sensor_900_bridge_before_fixed_layout_cache: 캡 제거·carry 물성
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/bridge_waypoint_vnc.sh --checkpoint "$CKPT" --switch-margin 1.0 --bridge-physics carry_nocap
```

`follow_cap`은 상면 캡을 별도 `fix_base_link=True` 강체로 만들고, 매 물리 step 후 본체의 최신 위치·회전으로 갱신해 다음 step의 접촉에 사용한다. 본체의 질량·관성·하단 충돌 형상은 유지한다. 캡↔본체 충돌을 제외하며 캡에 가해진 힘은 본체로 전달되지 않는다. 본체에 직접 닿는 접촉은 계속 동적으로 처리한다. 물성만 바꾸는 실험이 아니라 인위적인 지지면 분리 실험이며, 실제 관절로 연결하거나 PhysX kinematic actor를 사용하는 방식은 아니다. 캡은 상자 아래가 아니라 원래 상자 내부의 최상단 5mm이며, 이 조건에서는 청록색으로 표시한다.

```bash
# stage2_sensor_900_bridge_before_fixed_layout_cache: 동적 본체를 따라가는 고정 상면 캡
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/bridge_waypoint_vnc.sh --checkpoint "$CKPT" --switch-margin 1.0 --bridge-physics follow_cap
```

`--cap-lift`는 follow_cap의 캡만 상자 로컬 상방으로 올리는 거리(m)다. 기본 `0`은 원래 높이, `0.01`은 1cm, `0.02`는 2cm다. 본체·관측용 상자 크기·GTA 목표는 유지한다. 다른 물성 조건에서 0이 아닌 값을 지정하면 오류로 처리한다.

`--bridge-start present|near|insert`는 기존 cooperative 시작 배치를 선택한다. present는 다리가 landing에 있는 상태, near는 landing에서 삽입 방향으로 0.1–1.5m 떨어진 상태, insert는 채널 입구에 다리 앞부분이 1.5cm 들어간 상태다. 선행 사람·운반 상자·원래 목표는 계속 최종 횡단 중심선에 배치하며, 선행의 중간 GTA 목표는 실제 다리 위치를 따라간다. 시작 모드는 모든 `--bridge-physics` 조건에 적용되지만 고정 다리는 near/insert에서 밀려 들어가지 않는다.

```bash
# stage2_sensor_900_bridge_before_fixed_layout_cache: 고정 캡 2cm 상승·present
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/bridge_waypoint_vnc.sh --checkpoint "$CKPT" --switch-margin 1.0 --bridge-physics follow_cap --cap-lift 0.02 --bridge-start present

# stage2_sensor_900_bridge_before_fixed_layout_cache: 고정 캡 2cm 상승·near
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/bridge_waypoint_vnc.sh --checkpoint "$CKPT" --switch-margin 1.0 --bridge-physics follow_cap --cap-lift 0.02 --bridge-start near

# stage2_sensor_900_bridge_before_fixed_layout_cache: 고정 캡 2cm 상승·insert
TOKENHSI_GPU=5 PORT=6081 bash tokenhsi/scripts/multi_agent/bridge_waypoint_vnc.sh --checkpoint "$CKPT" --switch-margin 1.0 --bridge-physics follow_cap --cap-lift 0.02 --bridge-start insert
```

사용자 확인용 영상은 [video/bridge_ablation_20261117/index.html](../video/bridge_ablation_20261117/index.html)에 모았다. `비교영상/`, `개별영상/`, `진단용/`으로 분류하며 원본 로그·프레임·검증 자료는 `output_etc/check/bridge_contact_20261117/`에 둔다.

새 실험 영상은 기존 영상 폴더에 추가하지 않고 `video/` 아래 새 폴더에 개별 MP4로 저장한다. ZIP은 만들지 않는다. 전체 목록은 [video/index.html](../video/index.html)을 따른다.

가상목표를 끈 0·1·2·5cm의 동일 장면 96회 비교는 [높이 비교 영상·CSV](../video/bridge_cap_height_compare_20261118/index.html)에 있다. present/near/insert 각각 8환경을 비교하며, 영상은 해당 batch의 scene 0이다. 재현은 `TOKENHSI_GPU=7 PORT=6082 bash output_etc/check/bridge_cap_height_compare_20261118/worker.sh 0.00 0.02` 및 `TOKENHSI_GPU=5 PORT=6084 bash output_etc/check/bridge_cap_height_compare_20261118/worker.sh 0.01 0.05`로 실행했다. 이 명령은 같은 진단 폴더에 기록하므로 새 비교에는 별도 출력 폴더를 사용한다.

## Follow-cap 5cm 학습 비교

random15k의 밀기4명+Enable2세트(총8명) 평가는 `tokenhsi` 환경에서 `python output_etc/check/bridge_random15k_push4_enable2_eval_20261120/launch.py`로 실행한다. GPU5/6/7에 seed0/1/2를 각각 배정하며 512환경×3seed·insert/횡단100%·폭1.6~2.2m 프로파일 샘플링·follow_cap5cm·가상목표 없음·최종 배치 후 좌우 정렬이다. 같은 실행의 900/1500step 누적 SR을 저장한다. 공동밀기4명 동시 성공·각 Enable의 선행/후행 동시 성공을 세 subtask로 집계하고 평균을 Subtask SR, 8명 동시 성공을 Full-scenario SR로 정의한다. launcher는 GPU별 총점유85000MiB 상한을 감시하며 기존 학습은 보존한다. `summarize.py`는 저장 trace와 집계를 대조한 후 `summary.{json,csv,md}`를 생성한다. 완료 결과는 재사용하며 새 평가에는 별도 폴더를 쓴다. [평가 폴더](../output_etc/check/bridge_random15k_push4_enable2_eval_20261120/).

random 13k~18k 평가는 `tokenhsi` 환경에서 `python output_etc/check/bridge_random_13000_18000_ordered_20261120/launch_eval.py`로 실행한다. GPU7만 사용하며 checkpoint당 512환경×3seed·900step·insert/횡단 100%·배치 후 좌우 정렬·동일 초기 fixture를 유지한다. 총 18개를 최대 8개 병렬·GPU 전체 85000MiB 상한으로 실행하고 검증 후 1k 간격 표를 생성한다. 완료 결과는 재사용하며 새 평가에는 별도 폴더를 쓴다. [결과 폴더](../output_etc/check/bridge_random_13000_18000_ordered_20261120/).

rear90 13k 평가는 `tokenhsi` 환경에서 `python output_etc/check/bridge_rear90_13000_ordered_20261119/launch_eval.py`로 실행한다. seed0/2=GPU5, seed1=GPU6이며 512환경×3seed·900step·insert/횡단 100%·배치 후 좌우 정렬·동일 초기 fixture를 유지한다. GPU 총점유 85000MiB 상한을 감시하고 12k는 기존 결과를 재사용해 비교한다. 완료 결과는 재사용하며 새 평가에는 별도 폴더를 사용한다. [결과 폴더](../output_etc/check/bridge_rear90_13000_ordered_20261119/).

동일 12k 평가는 `tokenhsi` 환경에서 `python output_etc/check/bridge_rear90_random_12000_ordered_20261119/launch_eval.py`로 실행한다. rear90=GPU5, random=GPU6이며 지정 checkpoint가 없으면 저장 완료·epoch 확인까지 기다린다. 기존 4명·512환경×3seed·900step·insert/횡단 100%·최종 좌우 정렬·동일 fixture를 유지하고 총점유 85000MiB 상한을 감시한다. 완료 결과는 재사용하며 새 평가에는 별도 폴더를 쓴다. 완료 시 [1k 단위 표](../output_etc/check/bridge_rear90_random_12000_ordered_20261119/history_7k_12k.md)·[전체 지표](../output_etc/check/bridge_rear90_random_12000_ordered_20261119/index.html)를 자동 생성한다.

미평가 rear90 8000·9000 / random 11000 보충 평가는 `tokenhsi` 환경에서 `python output_etc/check/bridge_missing_8k9k11k_ordered_20261119/launch_eval.py`로 실행한다. 기존 4명 구성·512환경×3seed·900step·insert/횡단 100%·최종 좌우 정렬·동일 초기 fixture이며 GPU5/6에 분산한다. GPU당 최대 3개 병렬·총점유 85000MiB 상한을 적용하며 완료 결과는 재사용한다. [결과 폴더](../output_etc/check/bridge_missing_8k9k11k_ordered_20261119/).

공동밀기 인원·선행/후행 쌍 수·보행로 폭을 조절하는 추론 전용 viewer:

```bash
TOKENHSI_GPU=6 PORT=6082 \
bash output_etc/check/bridge_push3_follow_cap_view_20261119/view.sh \
  --pushers 4 --before-pairs 2 --bridge-width 2.0 \
  --bridge-start insert --seed 42
```

`--pushers`는 2/3/4명(기본 3), `--before-pairs`는 선행·후행 쌍 수(양의 정수, 기본 1)다. 총인원은 `pushers + 2*before_pairs`다. 모든 선행은 같은 출발 플랫폼에서 같은 방향으로 건너며, 각 후행은 건너편에서 자기 선행의 상자를 사용한다. 쌍마다 상자·목표·AT 선행 조건을 따로 배정한다. `--bridge-width <m>`는 실제 보행로 폭을 양의 실수로 지정하며 본체·follow_cap·충돌 및 센서 크기에 함께 적용한다. 생략하면 환경 생성 시 기존 프로파일의 1.6~2.2m 폭을 샘플링하고 해당 환경의 reset 간에는 유지한다.

다리 횡단 길이는 밀기 인원×1.6m, 플랫폼 간격은 그 길이+0.1m다. 플랫폼은 기존 인원별 크기 샘플링과 4인 이상 구간의 7~8m 상한을 유지하므로 쌍 수가 많으면 초기 배치 재시도가 늘거나 배치가 불가능할 수 있다. 밀기 담당은 후방90°·외곽0.5~5m로 샘플링하고 최종 배치 후 담당 영역을 좌우 정렬한다. 후행 과제는 SIT/CLIMB/ONTOP이다.

기본 rear90 10000 checkpoint를 추가 학습 없이 사용한다. `--checkpoint`는 rear90/rear90_random을 모두 지원하며 저장된 샘플링 설정으로 실행 config를 자동 선택한다. 두 모델 모두 `--coupled_assignment ordered`와 `--pusher_assignment ordered`를 내부에 지정해 최종 배치 후 좌우 정렬을 유지한다. **1환경**, 10에피소드×900step, crossing 100%, 가상목표 없음, follow_cap 5cm다. 기존 명령은 기본값인 밀기3명·1쌍으로 동작한다. `--bridge-start present|near|insert`, `--checkpoint <rear90.pth>`, `--seed`, `--episodes`, `--steps`, `--envs`를 지원한다. 화면 없는 검사는 `HEADLESS=1`. 학습 코드/config는 변경하지 않고 viewer 프로세스에서만 인원·기하를 확장한다. 결과는 같은 폴더의 `run_<시각>/verification.json`에 저장한다.

rear90 11000·random 10000 평가는 `tokenhsi` 환경에서 `python output_etc/check/bridge_rear90_11000_random10000_ordered_20261119/launch_eval.py`로 실행한다. 두 모델의 seed0/1/2를 각각 GPU5/6/7에 배정하며 GPU당 2개 병렬·총점유 85000MiB 상한이다. 이전과 같은 512환경×3seed·900step·insert/횡단 100%·최종 좌우 정렬·초기 fixture를 사용한다. 완료 결과는 재사용하며 새 평가에는 별도 출력 폴더를 사용한다. [결과 폴더](../output_etc/check/bridge_rear90_11000_random10000_ordered_20261119/).

rear90 10000·random 9000 평가는 `tokenhsi` 환경에서 `python output_etc/check/bridge_rear90_10000_random9000_ordered_20261119/launch_eval.py`로 실행한다. 두 모델의 seed0/1/2를 각각 GPU5/6/7에 배정하며 GPU당 2개 병렬·총점유 85000MiB 상한이다. 이전과 같은 512환경×3seed·900step·insert/횡단 100%·최종 좌우 정렬·초기 fixture를 사용한다. 완료 결과는 재사용하며 새 평가에는 별도 출력 폴더를 사용한다. [결과 폴더](../output_etc/check/bridge_rear90_10000_random9000_ordered_20261119/).

최신 rear90 9500(GPU6)·random 8000(GPU7)의 동일 초기 장면 평가는 `tokenhsi` 환경에서 `python output_etc/check/bridge_rear90_9500_random8000_ordered_20261119/launch_eval.py`로 실행한다. 모델당 512환경×3seed·900step·insert/횡단 100%·최종 좌우 정렬이며 GPU별 3개 병렬·총점유 85000MiB 상한을 적용한다. 완료 결과는 재사용하며 새로운 평가에는 별도 출력 폴더를 사용한다. [결과 폴더](../output_etc/check/bridge_rear90_9500_random8000_ordered_20261119/).

동일 7k의 후행 동작 진단은 `python output_etc/check/bridge_follower_diagnostics_7000_20261119/launch_eval.py`로 GPU5/7·각 512환경×3seed를 재생했다. 기존 fixture·정렬·insert/횡단 100% 조건이며 후행 궤적과 native 오차를 추가 기록한다. `tokenhsi` 환경에서 `python output_etc/check/bridge_follower_diagnostics_7000_20261119/analyze_followers.py`로 활성 이후 동작·성공 경계까지의 추가 오차를 집계한다. 재생 편차가 있으므로 원 평가 실패 건수와 구분하며, 완료 폴더는 보존하고 새 실행에는 별도 진단 폴더를 사용한다. [진단 결과](../output_etc/check/bridge_follower_diagnostics_7000_20261119/follower_analysis.json).

후행 실패 장면 VNC는 `view_failure.sh <rear90|random> <seed 0..2> <env 0..511>`을 사용한다. 7k checkpoint·insert/횡단 100%·최종 좌우 정렬을 유지한다. 마지막 번호는 원 평가의 장면 번호이며, 해당 장면의 상자·플랫폼 크기와 초기 상태만 가져와 **환경 1개**로 실행한다. 월드 좌표를 원점 근처로 옮기고 도착 플랫폼에 카메라를 맞춘다. 후행은 밝은 초록색으로 표시한다. 900step 종료 후 마지막 화면을 유지하며 창 닫기 또는 Ctrl+C로 끝낸다. 저장된 동영상 재생이 아니라 1환경 재시뮬레이션이므로 성공/실패도 달라질 수 있다. headless 900step 확인에서는 rear90 109는 후행 실패를 유지했으나 random 58은 성공으로 바뀌었다. 실행별 출력은 `output_etc/check/bridge_failure_viewer/`에 분리한다.

```bash
# rear90: 상자를 들어도 수평 정렬까지 못 간 ONTOP 실패
TOKENHSI_GPU=6 PORT=6082 bash output_etc/check/bridge_follower_diagnostics_7000_20261119/view_failure.sh rear90 0 109

# rear90: 높이 오차가 크게 남은 ONTOP 실패
TOKENHSI_GPU=6 PORT=6082 bash output_etc/check/bridge_follower_diagnostics_7000_20261119/view_failure.sh rear90 0 95

# rear90: 활성화 뒤 움직임이 작은 CLIMB 실패
TOKENHSI_GPU=6 PORT=6082 bash output_etc/check/bridge_follower_diagnostics_7000_20261119/view_failure.sh rear90 0 191

# random: 상자를 들어도 수평 정렬까지 못 간 ONTOP 실패
TOKENHSI_GPU=6 PORT=6082 bash output_etc/check/bridge_follower_diagnostics_7000_20261119/view_failure.sh random 0 58
```

`stage2_sensor_900_bridge_follow_cap_rear90_random`은 `_rear90`과 같은 사람·상자 샘플링에서 **담당 영역만 무작위 배정**하는 별도 설정이다. 최종 배치 후 공동밀기 두 사람에게 영역 0/1을 중복 없이 균등 추첨하고, 가상 분할 상자와 개인 목표에 함께 적용한다. 캐시 배치 재사용 시에도 에피소드마다 새로 추첨하며 에피소드 중에는 고정한다. 따라서 좌우 정렬과 일치/반대일 확률이 각각 50%이며, seed가 같아도 랜덤 추첨이 추가되어 두 설정의 전체 초기 배치가 동일하다는 뜻은 아니다. 가상목표 없음·cap 5cm·후방 90도·외곽 거리 0.5~5m·모드 혼합·같은 플랫폼 20%·보상은 `_rear90`과 같다. 원본 V4 2k에서 새로 학습하고 test/VNC는 이 설정으로 저장된 checkpoint를 사용한다. [실행 명령](config.md).

새 `_rear90` 학습 배치에서 **원본 DistillV4 2k 정책**을 제로샷으로 보려면 전용 실행기를 사용한다. 새 config의 배치 캐시·모드 혼합·같은 플랫폼 20%·cap 5cm·담당 정렬을 그대로 사용하며, V4 원본 정책/RMS를 strict 로드해 기본 장면 관측으로 평균 행동을 계산한다. 센서 정책으로의 추가 학습이나 확인 학습 checkpoint는 사용하지 않는다.

```bash
TOKENHSI_GPU=7 PORT=6082 \
bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_rear90_zeroshot_vnc.sh \
  --checkpoint "/home/hwanhee/ksh/stage1_2_base_teacher/output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth" \
  --seed 42
```

기본 1환경·100에피소드·에피소드당 최대 900step이며 `--envs`, `--episodes`, `--steps`로 조절한다. `--bridge-start mixed`(기본)는 학습처럼 present/near/insert 20/40/40%를 사용한다. 한 모드만 보려면 `--bridge-start insert`(또는 `present`, `near`)를 붙인다. 모드를 고정해도 같은 플랫폼 20%는 유지한다. 초기 로그에 모드와 같은 플랫폼 여부가 표시된다. 화면 없는 확인은 `HEADLESS=1`과 `_zeroshot_test.sh`를 사용한다. 실행별 초기 배치·해결된 config·원본 경로/epoch는 `output_etc/check/bridge_rear90_zeroshot/`에 저장한다.

`stage2_sensor_900_bridge_follow_cap_rear90`은 가상목표 없는 별도 학습 설정이다. 다리 위치 확정 후 상자 정후방 ±45°·root XY에서 직사각형 외곽까지의 최단 거리 0.5~5m로 밀기 담당을 각각 샘플링한다. 각도·거리는 균등 후보를 뽑고 플랫폼/상자 외곽 0.5m, 다른 사람 root와 1m 여유를 만족하는 후보를 채택한다. 따라서 충돌 제외 후 분포는 균등하지 않을 수 있다. 같은 쪽에 두 사람이 뽑히는 것은 허용한다. 최종 상자 좌표계의 X 순서로 담당 영역을 정하고 에피소드 중 유지한다. 캐시 복원도 관측·보상 이력 초기화 전에 같은 배정을 적용한다.

present/near/insert 모두 이 규칙을 사용하며 비율 20/40/40%, 선행·후행 같은 플랫폼 20%, cap 5cm, 가상목표 없음과 기존 보상은 유지한다. 원본 DistillV4 2k에서 새로 시작한다. 기존 5k 학습의 resume 설정이 아니며 새 실행/출력 이름을 사용한다. [학습·test·VNC 명령](config.md). test/VNC의 `$CKPT`는 새 `_rear90` 학습이 저장한 체크포인트로 지정한다. 기존 5000 `view_eval.sh`는 저장된 옛 배치 재생용이므로 새 샘플링을 재현하지 않는다.

현재 비교 세트는 `follow_cap`(가상목표 없음), `follow_cap_waypoint_single_pass`(가상목표·기존 보상), `follow_cap_waypoint_progress_single_pass`(가상목표·단계별 보상)다. 접두사는 `stage2_sensor_900_bridge_`이며 새 두 설정은 GTA 위치를 수정한 scene으로 actor·critic 각각 한 번 계산한다. `_single_pass` 없는 가상목표 설정은 기존 두 번 계산한 학습 재현용이다. 새 설정은 원본 V4 2k에서 시작하며 기존 두 번 계산한 체크포인트의 resume을 받지 않는다.

세 설정의 차이·보상식·초기화·전이 계약은 [학습 명세](reference/bridge_follow_cap_training.md), 기본 학습/test/VNC 명령은 [config](config.md)를 따른다. `stage2_sensor_900_bridge_follow_cap*` 전용 실행기는 기본 conda `tokenhsi`를 사용한다. 추론 전용 `bridge_waypoint_vnc.sh`의 중심선 토이 배치와 별개이며, 학습은 원래 무작위 배치·같은 플랫폼 20%를 유지한다.

3500 checkpoint의 insert·횡단 전용 512환경×3seed 비교는 [평가 규약](reference/bridge_follow_cap_training.md#insert-횡단-평가)을 따른다. 실행한 명령은 `python output_etc/check/bridge_crossing_eval_3500_20261118/launch_eval.py`이며, GPU5·7을 내부에서 각각 지정하고 전체 GPU 메모리 80000MiB 초과 시 평가 프로세스만 종료한다. 집계는 `TOKENHSI_GPU=5 TOKENHSI_CONDA_ENV=tokenhsi bash -c '. tokenhsi/scripts/multi_agent/runtime_env.sh; python output_etc/check/bridge_crossing_eval_3500_20261118/summarize_eval.py'`다. 실행기는 해당 폴더를 사용하므로 새 평가에는 스크립트를 새 진단 폴더에 복사하고 비교 checkpoint를 지정한다.

최신 4500 비교 평가는 `python output_etc/check/bridge_crossing_eval_4500_20261118/launch_eval.py`로 실행한다. GPU당 512환경×3seed를 동시에 실행하고 전체 메모리 85000MiB를 상한으로 감시한다. 3500 평가의 초기 상태를 재사용한다. 집계는 `TOKENHSI_GPU=5 TOKENHSI_CONDA_ENV=tokenhsi bash -c '. tokenhsi/scripts/multi_agent/runtime_env.sh; python output_etc/check/bridge_crossing_eval_4500_20261118/summarize_eval.py'`다. 기존 결과 보호를 위해 실행 폴더가 있으면 재실행을 거부하므로, 추가 실행은 새 폴더에 복사해 경로·checkpoint를 지정한다.

5000 접촉·대기 진단은 `python output_etc/check/bridge_crossing_eval_5000_contact_20261118/launch_eval.py`로 실행했다. 같은 초기 상태·GPU당 3seed 병렬·85000MiB 상한을 사용한다. runtime 적용 후 해당 폴더의 `summarize_eval.py`, `analyze_contact.py`를 차례로 실행하면 행동/씬 SR과 발 내딛기·접촉 추정·낙상 시점별 밀기 상태 및 대기 후보가 집계된다. 기존 실행 폴더는 재사용하지 않는다. [판정 규약](reference/bridge_follow_cap_training.md#5000-발-접촉대기-진단).

rear90와 rear90_random의 test/VNC는 `PUSHER_ASSIGNMENT=ordered|random`으로 최종 배치 후 담당 영역을 선택한다. 생략하면 checkpoint config를 따른다. 저장된 cfg는 바꾸지 않고 runtime 샘플러의 배정 설정만 복사해 덮어쓰므로 checkpoint 계약 검증은 유지된다. 직접 Python 실행은 `--pusher_assignment ordered|random`을 사용한다. 이 옵션은 학습/resume에서 허용하지 않는다. 기존 `COUPLED_ASSIGNMENT` 대신 rear90의 최종 reset 배정을 제어하는 옵션이다.

두 실행기의 다섯 위치 인자 뒤에 `--pusher-gaps 1.5 3.0`을 붙이면 밀기 담당 에이전트 번호 오름차순으로 초기 거리를 따로 지정한다. 단위는 m, 기준은 **상자 뒷면–사람 root의 삽입축 거리**다. 상자 중심 거리나 외곽까지의 최단 거리가 아니다. 매 reset마다 기존 횡방향 위치·자세를 유지하고 앞뒤 위치만 이동한 뒤 담당 영역을 배정한다. 환경 0의 에이전트 번호와 거리 대응은 로그에 출력한다. present/near/insert 모두 적용되며 생략하면 원래 rear90 샘플링이다. 고정 거리 이동 후 원래 후보의 각도·장애물 여유를 재샘플링하지 않으므로, 이 옵션을 지정한 장면은 학습 샘플링과 구분한다. 양의 실수를 받으며 `PUSHER_GAPS="1.5 3.0"`도 지원한다. 직접 Python은 `--pusher_gaps "1.5 3.0"`, 학습/resume에서는 거부한다.

```bash
# stage2_sensor_900_bridge_follow_cap_rear90: 밀기 담당별 1.5m / 3m, 최종 좌우 정렬
TOKENHSI_GPU=6 PORT=6082 PUSHER_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_rear90_vnc.sh "output/stage2_sensor_900_bridge_follow_cap_rear90/Stage2Sensor900BridgeFollowCapRear90_18-15-42-35/nn/Stage2Sensor900BridgeFollowCapRear90.pth" 4 1 8 10 --pusher-gaps 1.5 3.0

# stage2_sensor_900_bridge_follow_cap_rear90_random: 동일 거리·최종 좌우 정렬
TOKENHSI_GPU=6 PORT=6082 PUSHER_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_rear90_random_vnc.sh "output/stage2_sensor_900_bridge_follow_cap_rear90_random/Stage2Sensor900BridgeFollowCapRear90Random_18-16-39-21/nn/Stage2Sensor900BridgeFollowCapRear90Random.pth" 4 1 8 10 --pusher-gaps 1.5 3.0
```

```bash
# rear90: 최종 좌우 정렬
TOKENHSI_GPU=6 PORT=6082 PUSHER_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_rear90_vnc.sh "output/stage2_sensor_900_bridge_follow_cap_rear90/Stage2Sensor900BridgeFollowCapRear90_18-15-42-35/nn/Stage2Sensor900BridgeFollowCapRear90.pth" 4 1 8 10

# rear90_random: 평가/시각화에서 최종 좌우 정렬
TOKENHSI_GPU=6 PORT=6082 PUSHER_ASSIGNMENT=ordered bash tokenhsi/scripts/multi_agent/stage2_sensor_900_bridge_follow_cap_rear90_random_vnc.sh "output/stage2_sensor_900_bridge_follow_cap_rear90_random/Stage2Sensor900BridgeFollowCapRear90Random_18-16-39-21/nn/Stage2Sensor900BridgeFollowCapRear90Random.pth" 4 1 8 10
```

rear90 정렬 학습과 random 학습의 **동일 1k·정렬 평가**는 `python output_etc/check/bridge_rear90_eval_1000_ordered_20261118/launch_eval.py`다. GPU5/7에서 모델당 512환경×seed 0·1·2, insert·횡단 100%, cap 5cm, 900step으로 실행한다. rear90의 후방 ±45°·상자 외곽 0.5~5m 샘플러로 새 fixture를 만들고 두 모델이 공유한다. random checkpoint의 배정 설정은 평가 프로세스에서만 ordered로 덮어써 초기 배치·캐시 복원 후에도 정렬을 유지한다. 기하/담당 배정은 각 seed의 `sampling_verification.json`, `assignment_verification.json`에 검증 기록을 남긴다. runtime에서 같은 폴더의 `summarize_eval.py`로 행동/씬 SR을 집계한다. 기존 학습은 변경하거나 종료하지 않는다.

rear90 **3500**과 random **2500**의 최신 비교는 `python output_etc/check/bridge_rear90_eval_3500_random2500_ordered_20261118/launch_eval.py`다. GPU5만 사용하며 기존 학습 포함 85000MiB 상한, 평가 최대 4개 병렬로 512환경×3seed를 실행한다. 위 1k 평가의 초기 fixture를 재사용하고 두 모델 모두 배치 후 최종 좌우 정렬을 적용한다. insert·횡단 100%, cap 5cm, 가상목표 없음, 900step이며 완료 후 자동 집계한다. 두 모델의 학습 횟수는 서로 다르다. 기존 결과 폴더 재사용은 거부한다.

rear90 **4500** / random **3500** 평가는 `python output_etc/check/bridge_rear90_eval_4500_random3500_ordered_20261119/launch_eval.py`다. rear90은 GPU5에서 완료했고, 사용자 변경 지시에 따라 random seed 0/1/2는 GPU5/6/7에 하나씩 실행한다. GPU별 총점유 85000MiB 상한이며, 아직 없는 checkpoint는 저장 크기/수정시각 안정화와 CPU 로드/epoch 검증 후 자동 실행한다. 완료 결과는 재사용한다. 위와 같은 512환경×3seed fixture·최종 좌우 정렬·900step 조건이며 두 모델 완료 후 자동 집계한다. **현재 상태: 사용자 요청으로 random 평가 대기는 취소됐고 자동 실행되지 않는다. rear90 4500 결과만 완료됐다.**

rear90/random **동일 7000** 평가는 `python output_etc/check/bridge_rear90_eval_7000_ordered_20261119/launch_eval.py`다. rear90=GPU5, random=GPU7로 모델당 512환경×3seed를 실행한다. 이전 fixture·insert/횡단 100%·cap 5cm·가상목표 없음·최종 배치 후 좌우 정렬·900step을 유지한다. GPU별 최대 3개 병렬이며 초기화 중인 평가의 미할당 메모리도 예약해 실행을 조절하고 총점유 85000MiB를 감시한다. 완료 후 자동 집계하며 학습 프로세스는 보존한다.

가상목표 없는 follow-cap 6k의 **기존 학습 담당 배정** 재평가는 `python output_etc/check/bridge_crossing_eval_6000_training_assignment_20261118/launch_eval.py`다. GPU5에서 동일 fixture·512환경×3seed·900step, insert·횡단 100%, cap 5cm·원래 사람 거리를 사용한다. 좌우 정렬 평가와 담당 영역·개인 목표만 바꾸며, 새 rear90 학습 평가는 아니다. 완료 후 runtime 환경에서 같은 폴더의 `summarize_eval.py`를 실행하면 두 배정의 행동/씬 SR 비교와 초기 상태 검증을 저장한다. 결과 폴더 재사용은 거부한다. 학습 프로세스는 종료하지 않는다.

6000의 **최종 배치 후 좌우 정렬** 평가는 `python output_etc/check/bridge_crossing_eval_6000_ordered_20261118/launch_eval.py`로 실행한다. GPU5는 가상목표 없음, GPU7은 가상목표+단계별 progress이며, 6k checkpoint 저장이 안정된 뒤 각각 512환경×3seed를 병렬 평가한다. 기존 fixture의 사람·상자를 복원한 다음 실제 상자 좌표계에서 담당 영역과 개인 목표를 정렬하고, 관측 초기화 후 다시 검증한다. insert·횡단 100%, cap 5cm, 900step이다. GPU 전체 메모리 상한은 85000MiB이며 초과 시 이 실행기의 평가 자식만 종료한다. 결과 폴더를 재사용하지 않는다. runtime 적용 후 같은 폴더의 `summarize_eval.py`, `analyze_contact.py`로 집계한다. 기존 `training_assignment` 결과와 배정 조건이 다르다.

5000 평가 초기 장면을 VNC에서 다시 실행하려면 아래 명령을 사용한다. 첫 인자는 `baseline`(가상목표 없음) 또는 `waypoint`(가상목표+단계별 progress), 두 번째는 평가 seed `0|1|2`다. 같은 seed의 512환경·asset 크기·초기 물리 상태를 복원하고, 거리·삽입 깊이를 적용한 최종 배치에서 상자 좌표계의 사람 좌우 순서로 담당 영역·개별 목표를 정렬한다(`ordered`). 기존 학습의 반대편 배정과 다르므로 학습 조건 재현용 결과와 구분한다. 기본 카메라는 환경 0을 보여주며 900step 후 종료한다. 실행마다 별도 `output_etc/check/bridge_eval_viewer/` 하위 폴더를 만들고 기존 평가 결과는 보존한다. 초기 장면은 동일하지만 GPU 시뮬레이션 궤적의 비트 단위 재현을 보장하지 않는다.

```bash
TOKENHSI_GPU=7 PORT=6081 bash output_etc/check/bridge_crossing_eval_5000_contact_20261118/view_eval.sh waypoint 0
TOKENHSI_GPU=5 PORT=6082 bash output_etc/check/bridge_crossing_eval_5000_contact_20261118/view_eval.sh baseline 0
```

시각화에서 두 밀기 담당의 초기 거리는 `--pusher-gap 1.0`으로 조절한다. 단위는 m이며 **상자 뒷면에서 사람 root 중심까지**의 삽입축 거리다. 두 사람 모두 같은 거리로 배치하며 좌우 간격 1.6m·자세·다리·목표는 유지한다. 양의 실수(예: `0.5`, `0.75`, `1.0`, `2.0`)를 받는다. 생략하면 기존 eval의 0.5~0.8m 배치를 그대로 복원한다. 지정한 실행은 원래 평가 배치와 다르며 manifest에 별도 표시한다.

```bash
TOKENHSI_GPU=7 PORT=6081 bash output_etc/check/bridge_crossing_eval_5000_contact_20261118/view_eval.sh waypoint 0 --pusher-gap 1.0
```

두 사람을 서로 다른 거리로 배치하려면 `--pusher-gaps 0.5 1.0`을 사용한다. 각 장면에서 **밀기 담당 에이전트 번호가 작은 사람, 큰 사람 순서**이며 화면상의 좌우나 담당 영역 번호 기준이 아니다. 환경 0의 실제 에이전트 번호와 적용 거리는 시작 로그와 manifest에 표시한다. 거리 기준은 동일하게 상자 뒷면에서 사람 root 중심까지의 삽입축 거리(m)다. `--pusher-gap`과 동시에 지정할 수 없으며 `--insert-depth`와 함께 사용할 수 있다.

```bash
TOKENHSI_GPU=7 PORT=6081 bash output_etc/check/bridge_crossing_eval_5000_contact_20261118/view_eval.sh waypoint 0 --pusher-gaps 0.5 1.0
```

다리의 초기 삽입 깊이는 `--insert-depth 0.10`처럼 m 단위로 지정한다. 기준은 **플랫폼 입구 끝선에서 다리 앞쪽 면까지 들어간 거리**이며, `0`은 입구에 맞닿음, `0.015`는 기존 1.5cm, `0.10`은 10cm다. 0 이상의 실수를 받는다. 다리와 두 밀기 담당을 함께 옮겨 기존 사람–상자 간격을 유지하며, `--pusher-gap`을 함께 지정하면 해당 간격으로 다시 맞춘다. 다리 최종 목표와 플랫폼·선행·후행 배치는 유지한다. 두 플랫폼은 기존과 같이 삽입 입구 끝선이 정렬되어 있다.

```bash
TOKENHSI_GPU=7 PORT=6081 bash output_etc/check/bridge_crossing_eval_5000_contact_20261118/view_eval.sh waypoint 0 --insert-depth 0.10 --pusher-gap 1.0
```

현재 follow-cap test/VNC·직접 추론의 기본 담당 영역은 학습과 같은 `coupled_assignment=random`이다. 3500·4500·5000 headless 평가 실행기는 기존 fixture의 물리 배치를 복원한 다음 담당 영역·개별 목표를 학습 방식으로 맞춘다. 이는 현재 학습의 반대편 배정을 재현하는 조건이다. `manifest.json`에 `fixture_assignment=training_placement`와 변경된 장면 수를 기록한다. `view_eval.sh`는 이와 달리 `ordered`를 명시하고 모든 위치 조절 이후 실제 상자 좌표계에서 담당 영역을 정렬한다. 이 실행은 `fixture_assignment=final_box_ordered`로 기록한다. 실행 중인 학습의 배정 방식은 변경하지 않는다.

세 평가 실행기의 새 결과와 집계·접촉 분석은 각각의 `training_assignment/` 하위 폴더를 사용한다. 해당 폴더가 이미 있으면 실행을 거부한다. 기존 `baseline/`, `waypoint/`, fixture와 보고서는 변경하지 않으며 이전 `ordered` 결과로 남는다. 원인 분석 근거는 `output_etc/check/bridge_training_view_20261118/assignment_comparison.json`이다. 위 5000 평가 명령을 실행하면 GPU5·7에서 수정된 담당 영역으로 512환경×3seed를 평가한다.

### 학습 규칙 그대로 보는 viewer

`output_etc/check/bridge_training_view_20261118/view_training.sh`는 5000 checkpoint로 학습 환경 경로를 실행한다. `--test`만 사용하고 `--eval`은 사용하지 않는다. present/near/insert 20/40/40%, 같은 플랫폼 20%, 환경별 36개 배치 캐시, `coupled_assignment=random`, 확률적 행동, 한 사람이라도 낙상하면 장면 리셋하는 학습 규칙을 사용한다. optimizer는 실행하지 않는다. 실행 중인 학습 캐시를 복사하는 것이 아니라 새 캐시를 만들므로 동일한 학습 장면 재생은 아니다.

```bash
# 가상목표 없음
TOKENHSI_GPU=5 PORT=6084 bash output_etc/check/bridge_training_view_20261118/view_training.sh baseline 42

# 가상목표 + 단계별 progress
TOKENHSI_GPU=7 PORT=6083 bash output_etc/check/bridge_training_view_20261118/view_training.sh waypoint 42
```

seed 뒤에 `--bridge-start present|near|insert|mixed`(기본 mixed)와 `--pusher-gap 3.0`을 붙일 수 있다. gap은 두 밀기 사람 root와 상자 뒷면 사이의 삽입축 거리(m)이며 추가 이동량이 아니다. 매 캐시 복원에서 공동밀기 사람만 앞뒤로 이동하고 좌우 위치·자세·담당 영역·상자·목표는 유지한다. 옵션이 없으면 기존 거리를 쓴다. 모드를 고정해도 선행·후행 같은 플랫폼 20%는 유지한다.

```bash
TOKENHSI_GPU=7 PORT=6082 bash output_etc/check/bridge_training_view_20261118/view_training.sh baseline 42 --bridge-start insert --pusher-gap 3.0
```

마지막 위치 인자는 seed다. `VIEW_ENVS=128`, `VIEW_STEPS=18000`이 기본이며 환경 0을 보여준다. 초기 캐시 준비 시간이 필요하다. 화면 없는 확인은 `HEADLESS=1`을 앞에 붙인다. 실행별 설정·로그용 지표는 `output_etc/check/bridge_training_viewer/`의 새 폴더에 저장한다. `native_rollout.json`은 종료된 에피소드만의 진단 집계이며 고정 512환경×3seed 평가 SR과 구분한다. 뷰어는 사용자가 명령으로 직접 실행한다.

### 학습 모델의 위치 지정 viewer

`bridge_wait_vnc.sh`는 출발 플랫폼 A에서 건너편 B를 정면으로 바라보는 카메라를 사용한다. 기본 배치는 선행·운반 상자가 왼쪽, 다리·공동밀기 두 사람이 오른쪽 삽입통로에서 들어오는 형태다. 선행은 서 있는 상태에서 바로 앞의 상자를 직접 집는다. 횡단 필요 100%이며 대기 행동을 강제하지 않는다. 후행 사람은 기존 샘플러로 배치하고 후행 과제·장면 크기는 기존 설정을 따른다.

학습된 `stage2_sensor_900_bridge_follow_cap*` 체크포인트와 같은 실행 폴더의 `relation_config.yaml`이 필요하다. 캡 높이·가상목표·보상은 체크포인트의 실험 설정을 사용한다. 아래 모델은 캡 5cm·가상목표+단계별 progress single-pass 3500이다.

```bash
TOKENHSI_GPU=7 PORT=6081 \
bash tokenhsi/scripts/multi_agent/bridge_wait_vnc.sh \
  --checkpoint "/home/hwanhee/ksh/bridge_fixed/output/stage2_sensor_900_bridge_follow_cap_waypoint_progress_single_pass/Stage2Sensor900BridgeFollowCapWaypointProgressSinglePass_18-01-42-56/nn/Stage2Sensor900BridgeFollowCapWaypointProgressSinglePass_00003500.pth" \
  --leader-lateral -1 --leader-depth 2.2 --pickup-distance 1.1 \
  --bridge-entry-side right --bridge-distance 6 \
  --goal-lateral 0 --goal-depth 2 --seed 42
```

| 인자 | 의미·기본값 |
|---|---|
| `--leader-lateral` | 선행·상자의 좌우 위치. `-1` 왼쪽 안전 경계, `0` 중앙, `1` 오른쪽 안전 경계. 기본 `-1`, 중간 실수 가능 |
| `--leader-depth` | 출발 플랫폼의 절벽 가장자리에서 선행까지 안쪽 거리(m). 기본 `2.2` |
| `--pickup-distance` | 선행 바로 앞, 건너편 방향으로 상자 중심까지 거리(m). 기본 `1.1` |
| `--bridge-entry-side` | 공동밀기 삽입 방향 `left/right`. 기본 `right` |
| `--bridge-distance` | 삽입 방향을 따라 다리 중심에서 최종 다리 중심까지 거리(m). 생략하면 기존 insert 시작 위치 |
| `--goal-lateral` | 건너편 원래 목표의 좌우 위치 `[-1,1]`. 기본 `0` |
| `--goal-depth` | 건너편 플랫폼 절벽 가장자리에서 목표까지 안쪽 거리(m). 기본 `2` |

위치는 사람·상자 크기와 플랫폼 여유를 포함한다. 조합이 플랫폼을 벗어나거나 사람과 상자가 겹치면 오류로 알린다. 좌우 반전은 `--leader-lateral 1 --bridge-entry-side left`다. 가상목표 없는 모델은 checkpoint를 `output/stage2_sensor_900_bridge_follow_cap/Stage2Sensor900BridgeFollowCap_18-01-22-41/nn/Stage2Sensor900BridgeFollowCap_00003500.pth`로 바꾸면 된다.

로컬 화면은 같은 인자로 `bridge_wait_test.sh`, 화면 없는 확인은 여기에 `--headless --envs 8 --episodes 1 --steps 40`을 추가한다. 기본 1환경·100에피소드·최대 900step이며 종료/낙상 시 다음 배치로 넘어간다. VS Code에서 6081 포트를 전달하고 `http://localhost:6081/vnc.html?autoconnect=1&resize=remote`로 접속한다. 초기 배치 JSON/PT·결과는 `output_etc/check/bridge_wait_viewer/<시각>/`에 저장한다. `--capture`는 viewer의 지정 step 화면을 PNG로 저장하며 `--out output_etc/check/<검사명>`으로 경로를 바꾼다.

`CHECK=1`은 1024환경을 유지한 최소 반복 확인 학습이고(기존 공통 루프는 epoch 2에서 저장), 새 학습은 원본 DistillV4 2000에서 시작한다. 이어 학습할 때는 동일한 실험의 `RESUME_CHECKPOINT=/경로/해당모델.pth`를 지정한다. test/VNC에는 **해당 설정으로 저장된 센서 모델 checkpoint**를 전달한다. 원본 DistillV4 2000을 이 test/VNC에 직접 넣는 것은 전이가 아니므로 호환되지 않는다.
