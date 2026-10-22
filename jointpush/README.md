> **Partial Joint Push Stage2**는 [PARTIAL_KOO_1021.md](PARTIAL_KOO_1021.md)와 `scripts/koo_partial_train.sh`를 사용한다. `koo_1021_train.sh`는 Stage1 재개 전용이다.

> 최신 `koo_1021`의 **Stage1BasePush(2명·4상자)** 준비는 [KOO_1021.md](KOO_1021.md)를 참조한다. 아래 NativePhaseY2m과 checkpoint 계약이 다르다.

# PushNativePhaseY2m 재현

`juan` 브랜치에서 사용하는 Koo Stage1 Native PUSH Y2m 재현 패키지다. 소스는 `koo_1021`의 커밋 `c3b71cf9149db5304dc2ae74bf30ace01eb61e93`에서 가져왔다. `ports/koo_native_y2m/`의 소스·YAML·RSI bank·asset은 원본 그대로 보존한다. `source_manifest.json`으로 원본 768개 파일의 무결성을 확인한다. 외부 KSH/Koo 디렉터리는 수정하지 않는다.

두 에이전트가 각자의 상자 한 개를 민다. 문 없이 상자 2개·목표 2개, 총 6토큰/570차원 관측이다. `edge_mlp` relation bias, GTA, 64차원/4층/2head 구조다. 상자는 로컬 X=1.0~1.5m(0.1m 간격), Y=2m, Z=1.3m, 20kg, 마찰0.6이다. 목표는 초기 상자의 로컬 +Y 정면 2~5m, loco 출발 거리는1~2m다. RSI는 loco/native PUSH=50/50, PUSH 후보949개(clip7 제외), late 확률70%·후반30% 조건을 유지한다. AMP reference는 loco4개/PUSH21개를 사용한다. 원본에서 유지하는 door reference1개도 파일 연결이 필요하다.

## 체크포인트 업로드

기본 위치는 아래다. 가중치는 Git에서 제외하며 준비 과정에서 다운로드하거나 복사하지 않는다.

```text
jointpush/checkpoints/PushNativePhaseY2m_00006000.pth
```

다른 경로도 스크립트의 첫 번째 인자 또는 `JOINTPUSH_CHECKPOINT`로 지정할 수 있다. Y2m용으로 저장된 체크포인트를 사용한다. 이전 XY5000 원본은 새 환경 계약과 다르므로 직접 사용할 수 없다. 업로드된 파일의 환경·학습 설정·모션/RSI bank 해시는 사전 검사하고, 실제 실행에서는 원본의 전체 physics/reference 계약 검사와 strict 네트워크 복원도 수행한다.

## 준비와 CPU 검사

아래 명령은 `/home/hwanhee/juan/CVPR2027`에서 실행한다. 기본 Python 환경은 `tokenhsi_juan`이다. 다른 설치에서는 `TOKENHSI_CONDA_ENV`와 필요하면 `JOINTPUSH_PYTHON`을 지정한다.

```bash
# 데이터 경로 연결 + 원본 소스/asset/mesh/PUSH 모션/RSI 계약 해시 확인. GPU 미사용.
bash jointpush/scripts/prepare.sh

# 다른 머신의 데이터 위치: 원래 tokenhsi/data 디렉터리를 지정한다.
bash jointpush/scripts/prepare.sh --data-root /path/to/tokenhsi/data

# 업로드 후 체크포인트 사전 검사
bash jointpush/scripts/prepare.sh --checkpoint jointpush/checkpoints/PushNativePhaseY2m_00006000.pth

# 체크포인트 없이 CPU reset/RSI/AMP/24 physics step/네트워크 shape 검사
bash jointpush/scripts/check_cpu.sh

# 업로드 후 CPU physics 계약 + Actor/Critic/Discriminator strict 복원 검사
bash jointpush/scripts/check_cpu.sh jointpush/checkpoints/PushNativePhaseY2m_00006000.pth
```

데이터는 기존 프로젝트의 Koo port 또는 `/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data`에서 읽기 전용으로 재사용한다. 외부 위치는 `--data-root`로 바꿀 수 있다. 필요한 26개 motion 파일만 원본 manifest의 상대 경로에 심링크로 연결하므로 manifest와 bank 해시를 바꾸지 않는다. 검증 보고서는 `runs/jointpush/y2m/` 아래에 저장한다.

## 평가·view·학습 재개

체크포인트 업로드 후 사용한다. GPU는 반드시 명시하며 아래5번은 이 서버에서 사용할 때의 예시다. 가중치가 없으면 GPU 초기화 전에 종료한다.

```bash
# 화면 없는 평가: 기본 4환경, 3 repeats
TOKENHSI_GPU=5 SEED=2141 bash jointpush/scripts/eval.sh

# 물리 viewer: 1환경, 4명 확장 없이 원래2명. noVNC6086 / 1280x720
TOKENHSI_GPU=5 SEED=2141 PORT=6086 bash jointpush/scripts/view.sh

# 파일 경로를 직접 지정할 수도 있다.
TOKENHSI_GPU=5 bash jointpush/scripts/view.sh /absolute/path/PushNativePhaseY2m_00006000.pth

# 이어서 학습: 원본2048환경, checkpoint의 model/RMS/optimizer/epoch/frame 사용
TOKENHSI_GPU=5 SEED=2141 bash jointpush/scripts/train.sh

# 6000회 파일에서 짧은 학습 확인: MAX_ITERATIONS는 추가 횟수가 아닌 총epoch 상한
TOKENHSI_GPU=5 SEED=2141 MAX_ITERATIONS=6002 bash jointpush/scripts/train.sh
```

기본 checkpoint 경로는 모든 GPU 명령에서 동일하다. `NUM_ENVS`/`REPEATS`는 평가에서 조절할 수 있고 viewer는1환경, 학습은2048환경을 유지한다. 저장 간격은500 epoch, learning rate는2e-5다. 결과는 `runs/jointpush/y2m/{evaluation,view,training}_<timestamp>/`에 저장한다. view에서는 `http://localhost:6086/vnc.html`에 접속한다. 원격 환경은 포트 포워딩이 필요하다.

runtime은 지정 GPU의 기존 MPS를 확인하며 데몬이나 다른 작업을 종료하지 않는다. viewer는 원본 native CUDA/Vulkan 경로를 사용한다. 원본 학습 wrapper는 지정 GPU에 Python CUDA client가 있으면 시작을 거부한다. 공유 GPU에서는 실행 전 점유와 MPS 연결을 확인해야 한다. 이 패키지 준비 과정에서는 GPU·viewer·학습을 실행하지 않았다.

원래 생성 과정은 Native XY5000 → `transfer_push_native_y2m_checkpoint.py`로 Y2m 계약 전이 → resume다. 지금 준비한 경로는 사용자가 제공하는 **Y2m 체크포인트를 복원/평가/재개**하는 경로다. 원래5000 가중치와 audit가 없으므로 그 이전 학습 이력의 동일 재생은 포함하지 않는다. simulator/rollout/replay의 live 상태는 checkpoint에서 복원하지 않으므로 bitwise 동일한 학습 궤적을 보장하지 않는다.

## 현재 검증

- 소스768파일 무결성, motion26개 연결, asset/발mesh/PUSH21개 hash 및 native bank 계약 일치.
- CPU1000 agent reset 표본, 양쪽 담당 배정/부분 reset, AMP finite,24 physics step 통과.
- CPU 네트워크570D/6토큰, actor `[4,32]`×2, critic `[4,1]`, discriminator `[128,1]` finite 확인.
- 체크포인트 미업로드 경로의 조기 종료·shell/Python 문법 검사 통과.
- 업로드한 checkpoint의 strict 복원과 GPU/학습/viewer는 업로드 후 검증 대상이다.

## 업로드 후 2명 검증 (2026-10-21)

업로드6000 checkpoint의 CPU strict 복원 및 GPU5/seed2141/loco 시작64장면·최대600step 평가를 완료했다. 장면 첫낙상2/64(3.125%), agent목표도달98/128(76.5625%), 양쪽목표도달39/64, 평균종료목표거리0.641m. 모델/RMS는 불변이며 GPU 작업은 종료했다. 낙상0 조건을 만족하지 못해4명은 실행하지 않았다. 결과: `runs/jointpush/y2m/uploaded_6000_2agent_seed2141_20261021/result.json`.

```bash
TOKENHSI_GPU=5 SEED=2141 NUM_ENVS=64 PROBE_STEPS=600 bash jointpush/scripts/scaling_eval.sh
```

`scaling_eval`은 원본 낙상 종료·CONTACT+AT 성공을 쓰고, 장면 첫 종료까지 각 agent의 ever/final 도달과 접촉·거리를 기록한다. 조기종료한 장면의 다른 agent는 검열되므로 총128명은 독립 trial이 아니다. 단일 평가 seed다.

Viewer GUI wrapper는 `scripts/run_gui.sh`를 사용한다. X socket/lock뿐 아니라 대응 내부 VNC 포트도 비어 있는 번호를 골라 기존6000/6001/6003 등의 서비스를 유지한다. 외부 웹 포트는 `PORT`(기본6086)다.

## 4명·상자4개 zero-shot 확장

`scripts/native4_y2m.py`는 원본 파일 수정 없이 평가 프로세스 안에서만4명용 task/graph/runtime을 설치한다. 하나의 PhysX 환경에 사람4명·물리상자4개·목표4개를 만들며 기존2명 sampler 두개를 X축 -10/+10m로 옮긴다. 각 상자 크기6종/20kg/마찰0.6·목표정면2~5m·loco 시작을 유지하고, 각 source pair 안에서만 원래50% 담당자교환을 수행한다. 중복할당은 없다. `push_only` 기반 클래스의 물체수2와 packet폭20 고정을4/40으로 확장해 전체1140D/12토큰 관측을 만든다. 원본 가중치/RMS/행동head를 strict 로드하고 모든 attention층에12토큰을 함께 넣는다. token mask나 추가학습은 없다.

4명은 평가전용이며 interaction RSI 초기화는 지원하지 않는다. CPU검사에서만 그룹간attention을 막아 같은 상태의 원본2명 encoder 두개와 feature/action readout이 동일한지 확인한다. 실제 GPU평가에서는 mask가 없다. scalar reward는 원래2인 함수를 두 pair에 그대로 적용한다. 계약 검사에서 명시적 인원/토큰수와 layout 변경만 허용하고 다른 physics/reference 계약은 보존한다. 그룹거리가 달라지므로 token수만의 인과실험으로 해석하면 안 된다.

```bash
# CPU: 1000agent reset/중복배정/부분reset/질량/physics24step/strict복원/readout동일성
bash jointpush/scripts/scaling4_cpu.sh

# GPU: 원본 bias와12토큰 attention 유지,64장면/최대600step
TOKENHSI_GPU=5 PROBE_AGENTS=4 NUM_ENVS=64 PROBE_STEPS=600 SEED=2141 bash jointpush/scripts/scaling_eval.sh

# 4명 viewer 진입점. 이번 검증에서는 GUI를 실행하지 않았다.
TOKENHSI_GPU=5 SEED=2141 PORT=6086 bash jointpush/scripts/view4.sh
```

6000회/seed2141/64장면/최대600step 결과:2명 장면낙상2/64(3.125%), 목표도달98/128(76.5625%);4명 장면낙상63/64(98.4375%), 목표도달1/256(0.390625%). 장면 첫 종료에서 동료를 검열하며 단일seed 측정이다. 4명 낙상 중59장면은 넘어진 agent의 이전 손접촉 기록이 없다. CPU 동일상태검사에서 pair-mask4명과 원본2명×2의 feature/action 최대차0. 보고서: `runs/jointpush/y2m/uploaded_6000_scaling_comparison_20261021/REPORT.md`.

## 4명·박스8개 view

```bash
TOKENHSI_GPU=5 SEED=2141 PORT=6086 bash jointpush/scripts/view4_boxes8.sh
```

대상박스4개+비대상박스4개, 목표4개다. 추가 박스는 각 대상에서 그룹 바깥쪽 X방향4m에 배치하며 동일 크기 profile/20kg/마찰0.6을 사용한다. 16토큰(사람4+박스8+목표4)이 전부 attention에 들어간다. 대상별 PUSH/AT 8개 edge는 유지하며 비대상 박스는 명시적 task edge가 없다. 총 관측1288D. CPU strict 복원·1000agent reset·부분리셋 격리·24step 유한값을 확인했다. GPU GUI는 실행하지 않았다. 기존4박스 기본값은 유지한다.

CPU 검증: `JOINTPUSH_BOXES=8 bash jointpush/scripts/scaling4_cpu.sh`. 8박스에서는 원본2인×2 readout 동일성 검사는 적용하지 않는다.

## 비대상 박스 masking ablation

`attention_ablation.py`는 actor encoder의 원래 token shuffle을 forward당1회 유지하고 선택된 비대상 박스 key열을 모든층·head·query에서 -inf로 가린다. 물리 박스와 담당/목표 edge, critic, 가중치/RMS는 유지한다. 저장할 때 token 순서를 canonical로 복원한다. 추가박스 physical4..7=그림X0..3=token8..11이다. 담당agent가 있는 박스는 이 실험에서 가리지 않는다.

`JOINTPUSH_ATTENTION_ABLATION=1 JOINTPUSH_MASK_BOXES=4,5 JOINTPUSH_BOXES=8 PROBE_AGENTS=4 TOKENHSI_GPU=5 bash jointpush/scripts/scaling_eval.sh`로 X0/X1을 가린다. 빈 JOINTPUSH_MASK_BOXES는 mask 없는 대조군이다. 결과폴더에 first-episode 평가와 첫2장면의 여러 시점 attention_raw.npz/states.json/capture.json을 함께 저장한다. masked key 확률0, 각층 출력 재구성 오차, 초기physics/obs hash를 검사한다.

이번 사용자 요청의 전체16개 subset 비교는 `runs/jointpush/attn_view/native_y2m_boxes8_mask_ablation_20261021/`에 저장한다. `attention_render.py <조건폴더> [공통logit색상상한]`은 실제mask가 반영된 PNG를 만든다. raw QK와 원래bias를 분리하고 actualattention에서 막힌열을 빗금 처리한다. `render_mask_ablation.py <전체폴더>`는 전체갤러리와 진행비교PNG/REPORT를 생성한다. 기존 출력 디렉터리는 덮어쓰지 않는다.

## 같은 관측의2명·4명 feature 비교

`bash jointpush/scripts/feature_compare_cpu.sh`는4명/4박스/4목표의CPU궤적에서동일관측을2인pair×2로추출하고graph index만remap해같은가중치/RMS로계산한다. tokenizer입력과각층64Dhuman feature,같은action head출력을저장하며4인pair-mask가2인계산과일치하는지검증한다. 비대상박스는없다. 기본32장면/최대300step, 첫scene종료전샘플만사용한다. `render_feature_compare.py <출력폴더>`로PNG/REPORT/index를만든다. 이번결과는 `runs/jointpush/attn_view/native_y2m_feature_2vs4_20261021/`이다.

## KSH Sit/Climb과 feature 안정성 비교

`bash jointpush/scripts/feature_compare_sitclimb_cpu.sh`는KSH stage1_task 39000모델을원본수정없이CPU에서실행한다. 4명/박스8개/Sit-Climb교대로기존재현과같이생성하며각pair의관측을2명/박스4개로추출해원래학습비율을맞춘다. extra박스의pair분할은logical4/5와6/7이다. `render_feature_compare.py`로단일모델PNG, `compare_push_sitclimb_features.py <출력> <Push원자료> <KSH원자료>`로task별비교PNG와보고서를만든다. 이번출력은 runs/jointpush/attn_view/push_vs_sitclimb_feature_comparison_20261021이다.

## 구버전HOLDING+AT feature 비교

`bash jointpush/scripts/feature_compare_holding_at_cpu.sh`는KSH stage1_fix unified6000을기존로컬재현config로read-only 실행한다. 같은4인관측에서2인pair별관측과owner별semantic edge를추출한다. 원자료는 runs/jointpush/attn_view/ksh_holding_at_feature_2vs4_20261021. `compare_push_sitclimb_features.py`의마지막인자로HOLDING+AT원자료폴더를추가하면4개태스크를함께그린다. 이번통합결과는 runs/jointpush/attn_view/push_sit_climb_holding_at_feature_comparison_20261021이다.

## 5과제500iter 체크포인트 낙상 확인

원본은 `/home/hwanhee/koo_cvpr/stage1_fix/ports/ksh_stage1_stage2_base/push`이며코드/데이터/checkpoint는읽기만한다. `five_task_fall_prepare.py`로저장run설정을local workspace로복사하고 `TOKENHSI_GPU=5 MODEL=depth180 PROBE_AGENTS=2 TASK_GRAPH=random_scenario bash jointpush/scripts/five_task_fall.sh`로원래2인혼합과제를평가한다. 두번째모델은 MODEL=depth200_friction04다. 평가결과는 runs/jointpush/five_task_500_fall_20261021/workspace/output_etc/check에저장한다. 기본64장면/600step/loco-only초기화이며train RSI혼합과구분한다. `TASK_GRAPH=push_at`는PUSH만수행하는별도평가다.

4인은원본Stage1지원범위를넘는평가전용로컬확장으로PUSH_AT만검증했다. 두500모델모두원래2인혼합과제에서64/64장면낙상으로baseline이안정적이지않아4인확장성해석은보류한다. 원자료/설정/검증/상세해석은위결과폴더의REPORT.md와all_results.json에있다.

## 후반층 owner bias 보정

`JOINTPUSH_BIAS_CORRECTION=1 JOINTPUSH_BIAS_TARGET=boxes JOINTPUSH_BIAS_LAMBDA=4 PROBE_AGENTS=4 JOINTPUSH_BOXES=4 TOKENHSI_GPU=5 bash jointpush/scripts/scaling_eval.sh`는 actor 3·4층에서 Hi→다른 owner의 박스 key에만 -4를 더한다. target은 humans/boxes/goals/none, λ는0/2/4/6을 지원한다. 2명과4명, 박스 수=인원수 조건만 지원하며 원본 가중치/RMS와 critic은 유지한다. 원래 token shuffle을 유지하고 canonical 순서로 raw QK, 원래 bias, 실제 적용 bias, attention을 저장한다. 보정 없음의 CPU4인 결과·궤적이 기존 구현과 완전히 일치하는 것을 확인했다.

`run_bias_correction.py`는 GPU5에서 두 인원수의 기준조건과 3종류×3강도를 순차 평가하며 기존 결과 디렉터리 덮어쓰기를 거부한다. `report_bias_correction.py`는 전체 완료 후 CSV/비교PNG/조건별 실제 attention PNG/REPORT를 만든다. 출력은 `runs/jointpush/attn_view/native_y2m_bias_correction_20261021/`이다.

모든 query에 같은 owner 제외 규칙을 적용하려면 `JOINTPUSH_BIAS_TARGET=all_foreign JOINTPUSH_BIAS_SCOPE=all_owner`를 지정한다. H_i/B_i/G_i query 모두 다른 owner의 H/B/G key에 -λ를 한 번 더하며 같은 owner의 기존 bias는 그대로 둔다. `JOINTPUSH_BIAS_LAYERS=last2`는 actor3·4층, `all`은1~4층이다. 가중치·critic은 유지한다. 기본 scope=human은 기존 human-query 실험을 보존한다.

`run_allforeign_bias.py`는 사용자가 확장한 all_owner 범위에서 두 층 범위×세 강도×2명/4명과 대조군, 총14조건을 GPU5에서 순차 평가한다. 출력은 `runs/jointpush/attn_view/native_y2m_allowner_bias_20261021/`, 집계/전체층 bias·attention 이미지는 `report_allowner_bias.py`로 만든다. 중단한 human-only 배치는 `native_y2m_allforeign_bias_20261021/interrupted.json`에 별도 표시했다.

층별 비교는 `JOINTPUSH_BIAS_LAYERS=first2|last2|all|layer1|layer2|layer3|layer4` 또는1부터4까지의 쉼표 목록(예: `1,3`)을 지원한다. λ는0/2/4/6/8/12다. `run_layer_bias_ablation.py`는 GPU5에서 기준조건, 초반2층4/6/8/12, 각 단일층4, 후반2층4, 전체층4를2명/4명으로 순차 평가한다. 총22조건 출력은 `runs/jointpush/attn_view/native_y2m_layer_bias_ablation_20261021/`이며 `report_layer_bias_ablation.py`가 전체 비교PNG/CSV/조건별실제attention PNG를 만든다. 단일층 실험은 그 층만 보정했을 때의 효과이며 leave-one-out 필요성 검사는 아니다.

층별 배치 도중 사용자 요청으로 남은2명 평가를 생략했다. `resume_layer_bias_four_only.py`는 기존4명 trial이 있으면 종료를 기다려 회수한 뒤 나머지4명만 수행하며 중복 실행하지 않는다. 최종 층별 보고서는4명11조건만 비교한다.

`JOINTPUSH_BIAS_HARD_MASK=1 JOINTPUSH_BIAS_SCOPE=all_owner JOINTPUSH_BIAS_TARGET=all_foreign JOINTPUSH_BIAS_LAYERS=all`은 모든층의 cross-owner항목을-inf로 가린다. finite강도는사용하지않고metadata.strength=null/hard_mask=true로구분한다. `report_owner_hardmask.py`는직전층별대조군과초기hash를맞춰전체층-4및hard mask를비교하고실제차단attention이0인지검증한다. 출력은 runs/jointpush/attn_view/native_y2m_allowner_hardmask_20261021.

전체층 -4를 실제 viewer에 적용하려면 `TOKENHSI_GPU=5 PORT=6086 bash jointpush/scripts/view4_bias.sh`를 사용한다. Native Y2m6000/4명/4박스/4목표/loco 시작이며 원본 viewer loop를 유지한다. 모든 H/B/G query에서 다른 owner key만 -4이고 hard mask는 아니다. 첫 action의 실제 층 출력과 보정 항목을 검사해 출력 폴더에 viewer_bias_verification.json을 남긴다. 이후에도 hook을 유지하며, 보정을 제거하는 finish()는 호출하지 않는다. 기본 view4.sh는 환경변수 JOINTPUSH_VIEW_BIAS=1이 없으면 기존처럼 보정 없이 동작한다.

보정 viewer는 첫 실제 reset 후 전체4인·박스가 보이는 overview를 한 번 설정하고 자동 agent-follow를 해제한다. 이후 카메라는 사용자가 조작할 수 있다.

KSH Sit/Climb의2명·2박스 raw attention은 `OUTPUT_PATH=<새경로> NUM_ENVS=4 PROBE_STEPS=300 SEED=42 bash jointpush/scripts/ksh_sitclimb_attention_cpu.sh`로 추출한다. source는read-only stage1_task/Distill39000, H0=SIT/H1=CLIMB이며여분box가없다. `render_ksh_sitclimb2.py <출력>`이원래bias를유지한rawQK/bias/실제attention과초기·120step QK전용PNG를만든다. 이번출력은 runs/jointpush/attn_view/ksh_sitclimb2_boxes2_raw_20261021.

NONE비교는 `analyze_none_relations.py`로수행한다. 출력은 runs/jointpush/attn_view/none_relation_comparison_20261021이며각모델의2명/4명에서box수=agent수·32초기장면·seed2141을사용한다. Koo4기존자료와새CPU Koo2/KSH2/KSH4를합쳐task/SELF제외NONE의type/layer/head평균과humanquery foreign총량을계산한다. KSH CPU추출은 PROBE_AGENTS/JOINTPUSH_BOXES로2또는4개를선택할수있다.

PUSH100 7000iter의4인무개입평가는 `TOKENHSI_GPU=5 PROBE_AGENTS=4 NUM_ENVS=64 PROBE_STEPS=600 OUTPUT_PATH=<새workspace/output_etc/check/경로> bash jointpush/scripts/five_task_scaling7000.sh`로실행한다. 전용저장설정과manifest는 runs/jointpush/five_task_push7000_scaling_20261022/workspace,결과/보고서는상위REPORT.md/summary.json이다. `report_five_task_scaling7000.py`는낙상/접촉/AT/이동거리와원본무결성을검증한다. 현4인결과는낙상9/64·목표0/256이며2인baseline은사용자선호로생략했다.

7000iter모델의4명·8박스view: `TOKENHSI_GPU=5 PORT=6086 bash jointpush/scripts/view_push7000_4.sh`. 저장설정과검증된4인확장을재사용하며추가bias/masking이없다. noVNC는localhost:6086/vnc.html이며원본폴더에는쓰지않는다.
