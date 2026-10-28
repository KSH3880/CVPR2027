# KSH stage1_2_jointpush 실행 준비

2026-10-27 사용자 승인으로 기존 Partial 학습만 종료하고 이 RSI+soft gate 본학습을 GPU6에서 시작했다.
원본 `/home/hwanhee/ksh/stage1_2_jointpush`는 읽기만 한다.

- 독립 snapshot: `ksh_jointpush`, commit `5e30a72c00bbec428a70c3e45bb4f3527628ff41`.
- provenance: `ksh_jointpush/source_manifest.json`. 실행 전에 소스554파일·데이터 링크·고정checkpoint 해시를 검사한다.
- 설정: `runs/jointpush/ksh_jointpush_rsi/workspace/configs/{env,train,motions}.yaml`.
- 베이스: `jointpush/checkpoints/ksh_jointpush/Stage1PushFinetune_00014000.pth`. 원본 config의10500 대신 현재 제공된 최신14000을 고정했다.10500도 회귀검사용으로 보존한다.
- snapshot의 코드·YAML은 원문을 보존한다. 생성config만 transfer path와 motion path를 변경하고 bootstrap은 기존 실행의 출력 계약을 유지한다. 실제 모션 내용·RSI bank는 변경하지 않는다.
- 본학습은 `ksh_jointpush/output/`, 검증·viewer는 `ksh_jointpush/output_etc/check/`에서 찾는다. 실행 중인 학습 보존을 위해 기존 `runs/jointpush/ksh_jointpush_rsi` 저장소 연결을 유지한다. RSI캐시는 원본과 공유하지 않고 복사한다. 모션 payload만 읽기용 링크다.

## 실험 설정

`stage2_jointpush_rsi`:1024환경×4명,관측object8개/물리후보24개,1200step.
순차coupled0.8/solo0.2이며 solo는SIT/CLIMB/AT/ONTOP/PUSH 혼합이다. coupled는공동PUSH다.
상자167profiles,X/Y1.6~2.2m,Z1.45~1.7m,인당20kg,본체마찰0.4/밑판없음.
공동인원수만큼X와질량을확대한다. 구역CONTACT/실제공동중심AT를 사용한다. AT state/progress는 그룹 평균 양손 면거리 quality soft gate, AT 최종성공은 상자 목표위치 조건만 사용하고 도달시 CONTACT/AT를 gate없이 포화한다.
학습coupled는그룹단위loco50%/pushRSI50%,단독은원본Stage1 과제별RSI다. 기본view는loco100%다.
Stage1 actor encoder/scene RMS동결,identity coordination/action head/critic/discriminator학습,새optimizer/epoch0,lr2e-5,500epoch저장이다.

## 명령

프로젝트 루트 `/home/hwanhee/juan/CVPR2027`에서 실행한다.
GPU6 이외 장치를 거부한다. 학습은 검증된GPU6 MPS pipe만 사용하며 서버를 시작·종료하지 않는다.
사용자 요청으로 메모리 여유·기존 Python 작업 수에 따른 실행 차단은 제거했다. 자원 사용 여부는 사용자가 판단한다. GPU6 제한과 검증된 MPS 연결 조건은 유지한다.
기존학습종료는별도사용자승인후공식MPS client종료절차를사용한다.

```bash
# CPU 경로/데이터/immutable source 검사; GPU 사용 없음
CUDA_VISIBLE_DEVICES='' /home/hwanhee/anaconda3/envs/tokenhsi_juan/bin/python \
  ksh_jointpush/scripts/ksh_jointpush_prepare.py

# 새 RSI Stage2 본학습 명령 (현재 실행 중이므로 중복 실행하지 않음)
TOKENHSI_GPU=6 bash ksh_jointpush/scripts/ksh_jointpush_train.sh

# Stage1 베이스 제로샷 view, 4명/1환경; GPU6 native, MPS 우회
TOKENHSI_GPU=6 PORT=6086 bash ksh_jointpush/scripts/ksh_jointpush_view.sh

# 이후 생성된 Stage2 checkpoint view
TOKENHSI_GPU=6 PORT=6086 bash ksh_jointpush/scripts/ksh_jointpush_view.sh /absolute/Stage2JointPushRsi_00000500.pth

# 새 Stage2 실험 자체의 resume (기존 Partial pth와 계약이 다름)
TOKENHSI_GPU=6 RESUME_CHECKPOINT=/absolute/new_stage2_checkpoint.pth \
  bash ksh_jointpush/scripts/ksh_jointpush_train.sh

# 실제 CPU PUSH-RSI 초기화/물리/정책 확인 (학습 없음)
TOKENHSI_GPU=6 NUM_ENVS=2 RSI_CHECK_PRESETS=all_coupled,push \
  bash ksh_jointpush/scripts/ksh_jointpush_launch.sh check
```

실험로그/weight:`ksh_jointpush/output/train_<시각>/`.
view/CPU확인:`ksh_jointpush/output_etc/check/`.
모든명령은실행conda환경을tokenhsi_juan으로고정한다.

## 검증과 한계

- CPU20tests 통과: 그룹/구역/관측/gate/RSI확률/과제/기존전이.
- 실제CPU2환경에서14000 pth를복원하고all_coupled/push의pushRSI100%검수각16step과부분reset을통과했다. fixture크기대체없이실행했다. 결과:`output_etc/check/cpu_rsi_setup/verification.json`.
- 최신14000 strict165tensor전이/104coord추가·coordination gradient/1회CPU probe업데이트와actor동결을확인했다. probe모델은저장하지않고본학습과분리했다. 결과:`gradient_verification.json`.
- GPU 본학습을 시작했고 2epoch PPO 업데이트·actor/RMS 동결·coordination 업데이트 및 startup저장을 검증했다. 새GUI는 아직 실행하지 않았다. 전체 무작위solo과제 RSI는 최초실행시누락크기를 screening하므로 추가시간이 걸릴수있다. 위CPU검수는PUSH RSI/물리경로검증이며 전체mixed훈련초기화 완료를 뜻하지 않는다.
- 기존 Partial viewer의N키와질량override는 새snapshot에 포함하지 않았다.

## 로컬 soft gate 변경

`ksh_jointpush/ksh_soft_gate.py`를 bootstrap에서만 적용하며 ksh 원본/snapshot은 수정하지 않는다. 각 사람의 두 손 중 더 먼 손의 유한면 거리 d로 q=exp(-(d/0.18)^2)를 구하고, 같은 상자 그룹의 q를 평균해 g를 만든다. 뒤쪽 위치·목표방향 조건을 유지한다. AT state/progress에만 g를 곱하고 최종 AT 성공/성공보너스는 원래 전원양손 hard조건으로 결정한다. 단독과CONTACT는 유지한다. 최종성공시 state/progress 포화도 유지한다.4명 중1명만정확히접근하면 g≈0.25이므로 다른구성원미접촉으로 shaping이완전히0이되는것을피한다.

생성config `stage2JointPush.reward_gate`에 방식·폭·코드SHA를 넣어 원래hard-gate Stage2와 직접resume되지 않게 한다. CPU보상회귀2개, 실제CPU2환경 공동/단독PUSH RSI·각16step·부분reset 검증을통과했다. 기록은 `soft_gate_tests.log`, `output_etc/check/cpu_soft_gate/verification.json`에 있다.

현재실행은 `runs/jointpush/latest_partial_training.json`, TensorBoard run은 `current_ksh_jointpush_rsi_soft`를 따른다. 기존학습곡선은 별도run으로보존한다.

## Stage1/단독 PUSH 목표 ±45도 수정

2026-10-27 사용자 요청으로 `ksh_goal_cone.py`에서 Stage1 sample_layout와 Stage2 solo sample_push의 후보 승인 조건을 보완했다. 상자 로컬+Y와 실제 상자→goal의 수평방향 차이가45도보다 크면 기존 pending 루프로 재샘플링한다. 공동PUSH의 기존±45도/agent배치는 유지한다. 크기·거리2~10m·마찰·RSI는 변경하지 않는다. 원본과 snapshot 코드는 수정하지 않고 bootstrap에서 로컬 sampler를 연결한다.

새실행 기본workspace는 `runs/jointpush/ksh_jointpush_rsi/workspace_goal45_goal_success`다. 이전workspace는보존한다. Stage1수정설정은 `configs/stage1_env.yaml`, Stage2 RSI설정은 `configs/env.yaml`이다. Stage1도 같은 `ksh_jointpush_run.py` bootstrap을 사용해야 수정 sampler가적용된다. scene에각도와adapter SHA를기록해구 Stage2 resume와 계약을분리한다.

CPU24tests(±45도경계/회전/Stage1 64환경샘플/거리/기존RSI·soft gate)와 실제CPU Stage2 단독·공동PUSH RSI 각16step/부분reset 후goal각도검사를통과했다. 기록은 `goal45_tests.log`, `output_etc/check/cpu_goal45/verification.json`이다. 학습은중단상태이며재시작하지않았다.

PUSH RSI 검사용view는 `TOKENHSI_GPU=6 TASK_GRAPH=all_coupled PUSH_RSI=1 PUSH_RSI_MIX=0.5 bash ksh_jointpush/scripts/ksh_jointpush_view.sh`로띄운다. `PUSH_RSI=1`만지정하면push참조100%,mix를지정하면그룹단위50/50초기화를에피소드마다뽑는다. 학습확률은변경하지않고eval에서만적용한다.

초기자세고정검수는 `KSH_POSE_ONLY=1 PUSH_RSI=1 PUSH_RSI_MIX=.5 TASK_GRAPH=all_coupled TOKENHSI_GPU=6 bash ksh_jointpush/scripts/ksh_jointpush_view.sh`로실행한다. reset후그래픽변환을1마이크로초/무중력/속도0으로한번동기화한뒤물리/정책진행없이렌더링하며N=혼합새자세,R=pushRSI,L=loco,Esc=종료다. 초기5렌더frame의root/DOF불변검증과PNG를출력한다.

## 목표 도달 성공/포화에서 hand gate 제거

현재설정은 `workspace_goal45_goal_success`다. 성공판정은기존AT위치기준(20cm·Z오차1mm)을유지하되hand hard gate를제거했다. 도착전AT state/progress에만soft gate를곱하고,목표도달시pairedCONTACT와AT의state/progress/success를전부최대값으로한다. 표시용유효group_gate도1이며원래손quality는별도진단값으로남긴다. 목표를벗어나면현재상태기준성공이해제되고soft shaping으로돌아간다. power/collision은변경하지않는다.

CPU26tests 및실제CPU RSI/reset/정책검증을통과했다. 새학습/뷰어는실행하지않았다. 과거hard-success Stage2와직접resume가되지않도록reward gate계약과코드SHA가분리된다.

현재 새실행 기본설정은 `workspace_push_only_aligned`다. 목표각도0도로 상자+Y에정렬한다. 기존실행중인workspace_push_only학습에는반영하지않으며재시작한실험부터적용한다. PUSH-only·면거리soft gate·목표도달포화·RSI는유지한다.


## 단독 PUSH ±45° / 공동 PUSH 직선

새 기본 workspace는 `workspace_push_solo45_coupled0`이다. `jointPush.scene.solo_goal_half_angle_degrees: 45`는 Stage1/Stage2 단독 PUSH에 적용하고, `goal_half_angle_degrees: 0`과 Stage2 공동 목표0°는 유지한다. 거리2~10m·인당20kg·보상·RSI·그룹 샘플링은 유지한다. 이전 aligned workspace/중단한 학습 결과는 보존한다. 새 설정은 생성된 scene 계약과 adapter SHA로 구분되며 구 checkpoint를 새 설정에 그대로 resume하는 것은 허용하지 않는다.

그룹 샘플러의 `coupled_probability: 0.8`은 남은 사람이 둘 이상일 때 공동 그룹을 뽑는 확률이다. 그룹 크기를 순차로 정하므로 명목 단독 비율은 사람 기준17.7333%, 단독이 한 명 이상 있는 장면52%, 전원단독 장면0.8%다. 배치 제약에 따른 재샘플링 전 확률이며, 학습 중 실측 혼합 비율과는 구분한다. viewer의 `TASK_GRAPH=all_solo`는 단독100%다.


## PUSH RSI 선택: 기존 group / 독립 agent

기본값 `KSH_RSI_MODE=group`은 기존과 동일하게 같은 상자 그룹 전체를 push/loco 50:50으로 초기화한다. `KSH_RSI_MODE=agent`는 각 agent를 독립적으로 push/loco 50:50으로 초기화하며, 같은 상자 그룹 안에서도 자세가 섞인다. 상자 위치·회전·질량은 그룹이 공유하고 push agent만 자신의 담당 면에 참조 자세를 정렬한다. agent 모드는 push–loco 혼합 쌍의 충돌도 양쪽 인덱스 순서와 무관하게 검사한다.

```bash
# 기존 그룹 단위 RSI 학습 (기본값)
KSH_RSI_MODE=group TOKENHSI_GPU=6 bash ksh_jointpush/scripts/ksh_jointpush_launch.sh train

# agent별 독립 RSI 학습
KSH_RSI_MODE=agent TOKENHSI_GPU=6 bash ksh_jointpush/scripts/ksh_jointpush_launch.sh train

# 독립 RSI 모델 viewer; 실제 혼합 자세로 시작하려면 PUSH_RSI_MIX 지정
KSH_RSI_MODE=agent PUSH_RSI=1 PUSH_RSI_MIX=.5 TOKENHSI_GPU=6 PORT=6086 bash ksh_jointpush/scripts/ksh_jointpush_view.sh /absolute/Stage2JointPushRsi.pth
```

기존 모드는 `workspace_push_solo45_coupled0`, agent 모드는 `workspace_push_solo45_coupled0_agent_rsi`에 분리된다. 두 모드 모두 단독 목표±45°/공동 목표0°·인당20kg·PUSH-only 그룹분포·보상·Stage1 14k 전이 설정은 같다. agent 모드는 별도의 RSI 계약과 adapter SHA를 저장하므로 group checkpoint와 직접 resume하지 않는다. viewer는 checkpoint와 같은 모드를 선택한다. `PUSH_RSI`를 지정하지 않은 viewer는 두 모드 모두 기존처럼 loco100%이며, `PUSH_RSI=1`만 지정하면 push100%다. 학습 확률은 viewer 옵션의 영향을 받지 않는다.


## 그룹 RSI 세 상태 균등 샘플링

`KSH_RSI_MODE=balanced`는 그룹마다 전원push / 혼합 / 전원loco를 각각1/3 확률로 고른다. 혼합은 가능한 모든 비어 있지 않은·전원도 아닌 push 구성 중 균등 추출하므로 push와loco가 반드시 한 명 이상씩 포함된다. 1인 그룹에는 혼합이 없으므로 push/loco50:50으로 초기화한다. 확률은 배치 유효성 rejection 전 명목값이다. 박스의 위치·속도 초기화는 기존과 동일하다.

```bash
# 전원push / 혼합 / 전원loco 각1/3
KSH_RSI_MODE=balanced TOKENHSI_GPU=6 bash ksh_jointpush/scripts/ksh_jointpush_launch.sh train
```

기존 `group`, `agent` 모드도 유지한다. 새 설정은 `workspace_push_solo45_coupled0_balanced_rsi`에 저장하며 별도 RSI 계약/SHA로 구분한다. 실행 중인 agent50:50 학습에는 소급 적용하지 않는다. balanced viewer에서 세 상태 분포를 보려면 `KSH_RSI_MODE=balanced PUSH_RSI=1 PUSH_RSI_MIX=.5`를 사용한다.


## 복사본 config.md 기반 GPU5 viewer

`ports/ksh_jointpush/markdowns/config.md`의 `stage2_jointpush_rsi_vnc.sh → stage2_jointpush_rsi_test.sh` 절차를 현재 학습 모델에 연결하려면 다음 wrapper를 사용한다. 매 실행 시 최신 숫자 checkpoint와 저장된 학습 설정을 독립 실행 폴더에 고정한다. 원본 VNC/test 스크립트는 바꾸지 않고, `JOINTPUSH_ENTRYPOINT`에서 현재 학습 adapter·설정을 복원한다. GUI 서비스 부분만 검증된 GPU5 native 경로를 사용한다. KSH 원본과 snapshot은 수정하지 않는다.

```bash
# config.md 기본 추론: GPU5, loco100%, random_scenario, 4명·1환경·8 object 토큰
PORT=6095 bash ksh_jointpush/scripts/ksh_config_vnc.sh

# config.md의 PUSH RSI 진단: 전원push100%, 4명 공동 PUSH
PORT=6095 PUSH_RSI=1 TASK_GRAPH=all_coupled bash ksh_jointpush/scripts/ksh_config_vnc.sh
```

기본100회 반복이며 `REPEATS=10` 등으로 지정할 수 있다. checkpoint를 고정하려면 wrapper의 첫 인자에 pth 절대경로를 전달한다. 이 경로는 문서의 초기화 옵션을 그대로 사용하므로 이전 balanced 혼합 viewer와 다르다. `PUSH_RSI=0`(기본)은loco100%, `PUSH_RSI=1`은push100%이며 `PUSH_RSI_MIX`는 해제한다. 기본6095포트를 포워딩해 `http://localhost:6095/vnc.html?autoconnect=1&resize=remote`로 접속한다. `PREPARE_ONLY=1`은 GPU/뷰어 실행 없이 사본만 준비한다.


## Viewer 다음 episode

일반 KSH rollout viewer(기존GPU5 wrapper와config.md 기반wrapper 모두)는 창을클릭한후 **N**을누르면scene0의다음episode로넘어간다. 네agent와공유상자는player의정상done/reset경로에서함께새로배치된다. 수동skip은성공으로표시하지않으며다른키(Esc/카메라/동기화)는유지한다. headless평가와학습에는적용하지않는다. 이미실행중인viewer에는hot reload되지않으므로다시실행해야한다. 정지자세검수 KSH_POSE_ONLY의N/R/L은기존동작을유지한다.
