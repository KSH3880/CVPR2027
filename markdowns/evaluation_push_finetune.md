# Stage1 PUSH fine-tune 9000회 평가

**정정:** 이 문서/기존 출력은 요청된1인MD평가가아닌2인환경의별도개인별평가다. 원본78000의1인결과와직접비교하지않는다. 요청에맞춘수정평가는 [evaluation_single_push_compare.md](evaluation_single_push_compare.md)를따른다. 기존2인수치는보존하되1인평가결과로보고하지않는다.

## PUSH 1800step 재평가

사용자 요청으로 같은9000회 모델의 PUSH만1800step(명목60초, 시간종료1799step)·512환경×seed0/1/2로 GPU0에 별도실행했다. loco시작·2인/4상자·상자/목표분포·성공식·거리1m구간 집계는1200step평가와같다. 출력은 `output/stage1_base_push_finetune_eval9000_1800_20261026`, tmux는 `koo_eval_push9000_1800`이다. 기존4과제평가와1200step결과는보존한다. 실행상태는해당폴더 `run_manifest.json`/`batch.log`, 완료결과는 `RESULTS.md`/`summary.json`을따른다.

```bash
TOKENHSI_GPU=0 TOKENHSI_EVAL_TASKS=push_at TOKENHSI_EVAL_PUSH_EPISODE_LENGTH=1800 \
bash tokenhsi/scripts/multi_agent/evaluate_stage1_push_finetune.sh \
output/stage1_base_push_finetune_stable/Stage1PushFinetune_24-01-18-44/nn/Stage1PushFinetune_00009000.pth \
output/stage1_base_push_finetune_eval9000_1800_20261026
```

현재실행중중복실행하지않는다. 기존eval이GPU0에서실행중이면명시적 `TOKENHSI_EVAL_CONCURRENT=1`로동시실행을허용할수있지만, wrapper는기존Python client가이평가기인지및VRAM4GiB이상여유를확인하고학습/알수없는client와의동시실행은거부한다. 1800step배치는3run만계획하며기존4과제의900step설정은바꾸지않는다.

3seed모두exit0·저장궤적재채점일치로완료했다. 성공은1754/3072=57.10%,seed간표본표준편차2.11%p로1200step의51.66%대비+5.44%p다. 3seed초기root/상자/크기/목표/binding/DOF와앞1199step의agent별ever가모두이전평가와완전일치했다. 대조결과는1800출력폴더 `initial_match_1200_all_seeds.json`,거리별비교/시간별누적성공/실패정체분석은 `PUSH_COMPARISON.{md,json,png}`다. 완료출력을덮어재실행하지말고새출력폴더를사용한다.

기준: [evaluation_single.md](evaluation_single.md). 프로토콜 `push_finetune_two_agent_position_v1`.

사용자 지정대로 기존 AT·ON_TOP·SIT·CLIMB는900step(명목30초), **PUSH_AT만1200step(명목40초)**다. 각 과제 seed0·1·2, seed별512환경의 첫 에피소드만 채점한다. 평가 중 학습하지 않는다. 체크포인트는 시각화했던 `output/stage1_base_push_finetune_stable/Stage1PushFinetune_24-01-18-44/nn/Stage1PushFinetune_00009000.pth`이며 SHA256은 실행manifest에 고정한다.

## 기준 문서와의 차이

기준 문서는1인 정책/환경이다. 현재 checkpoint의 계약은 **2인·4상자·634차원/8토큰**으로 고정되어1인으로 바꾸지 않는다. 두 사람 모두 같은 과제를 독립 수행하고 각각 채점한다. 따라서 seed별512scene/1024agent, 과제별3seed에서1536scene/3072agent다. 같은scene의 두agent는 상관된 표본이며 한 사람의 낙상으로scene이 끝나면 다른 사람도 그때까지 채점한다. 사람별/scene별구분은원자료에남긴다. `summary.json`은두사람pooled와owner별결과를모두제공한다. **1인 single_match 표와 직접동등비교하지않는다.** 초기사람offset도2인환경의(-4,-4)/(-4,4)를유지하므로기준문서와동일초기배치라고주장하지않는다.

시간종료는 기존코드 `progress>=episodeLength-1`이므로일반trace는899/1199제어step이다. 제어간격은1/30초다. 낙상조기종료를유지하며성공후즉시종료하지않는다.

## 초기화·상자·물리

- 모든과제는deterministic정책과loco초기화만사용한다. PUSH상호작용RSI를사용하지않고실제초기root/관절위치·속도를archive에저장한다.
- ATpayload는기준문서9종을owner마다최소한번포함하고나머지를균등복원추출한뒤shuffle한다. 나머지기존과제크기는원본범위그대로다: SIT XY.25~.80/Z.35~.50, CLIMB XY.40~.80/Z.20~.70, ON_TOPpayload XYZ.20~.60/받침XY.40~.80/Z.25~.50,각.05m격자독립추출.
- SIT/CLIMB상자는평가프로세스에서만fix_base_link=True로생성하고첫episode동안pose불변을검사한다. ON_TOPpayload/받침과ATpayload는동적이다. 기존작은상자밀도100kg/m³를유지한다.
- PUSH는167크기(X/Y1.6~2.2m각.1m,Z1.45~1.7m각.05m,min(X,Y)≥1.1Z),상자당20kg/마찰.4를유지한다. 여분상자까지4개를그대로생성한다.
- 기존과제배치·운반선반은checkpoint의원본Stage1경로를사용한다. 사람→물체는SIT1~5m/그외1~10m반경추출,도착목표는기존square/rejection경로다. source/env/sim설정과실제초기값을각run에저장한다.
- PUSH상자→목표는최소1m,환경중심XY±4.5m,상자정면±45°/경로여유를만족하는거부샘플링이다. 별도최대거리제한은없다. 사람→상자접근거리1~10m와분리한다.
- GPU0의native CUDA/GPU PhysX/headless로실행한다. GPU1학습의MPS에연결하지않고학습을중단하지않는다. 새config/YAML이나학습보상은변경하지않으며평가프로세스안에서만AT규격/고정asset을적용한다.

## 성공 판정

첫action적용후첫frame부터종료frame까지같은궤적을두기준으로채점한다. 초기상태를성공으로미리세지않고,종료후reset된episode를추가하지않는다. AT/PUSH는payload중심,ON_TOP은payload중심,SIT/CLIMB는골반/root를검사한다.

| 과제 | 목표점 | 점 frame 성공 | 영역 frame 성공 |
|---|---|---|---|
| AT/PUSH_AT | 고정된상자목표중심 | 3D거리≤.20m | 점과동일 |
| ON_TOP | 현재받침상판중심+payload직립높이/2 | 3D거리≤.20m | 받침yaw로컬XY내부80% AND 목표Z오차≤.20m |
| SIT | 현재상판중심+.10m | 3D거리≤.20m | 위와동일한XY/Z영역식 |
| CLIMB | 현재상판중심+.94m | 3D거리≤.20m | 위와동일한XY/Z영역식 |

상판높이는받침회전bbox의worldZ최고점이다. 내부80%는각축가장자리10%제외(`|x_local|≤.4Lx`, `|y_local|≤.4Ly`)이며면적64%다. 접촉·손놓기·발높이·속도·유지시간을성공조건에추가하지않는다. 특히PUSH는**위치도달**평가이며접촉이력으로인한운동을인과적으로증명하는판정은아니다.

`point_ever`/`region_ever`가주결과이고종료순간`point_final`/`region_final`은진단이다. 원래학습의엄격한성공식은`strict_ever`로별도저장하며주결과로복사하지않는다. 성공후낙상해도ever는유지한다. float32/float64의환경별ever일치를검사한다.

## PUSH 거리별 채점

거리분류는**실제reset후상자중심→고정목표XY거리**다. 사람접근거리나상자표면간거리가아니다. 목표분포는변경하지않는다.

- `<1m`, `[1,2)m`, `[2,3)m`, …, `[19,20)m`, `20m이상`으로집계한다. 학습설정상첫구간은일반적으로0표본이고,거부샘플링/부동소수점차이를숨기지않기위해유지한다.
- 각구간의agent시도수/점ever성공수/성공률,종료잔여거리,평균진행률/최대진행률,성공한시도의평균최초도달시간을JSON에저장한다. 표본0인구간은성공률null로두고0%로표기하지않는다.
- 진행률은`clamp((초기거리-현재거리)/초기거리,0,1)`,최대진행률은첫episode동안의최대값이다. 초기거리와종료거리는각agent별로대응한다.
- 거리구간별표본수가다르므로전체pooled구간성공률은성공수합/시도수합이다. seed별구간결과도함께저장한다. 전체과제평균±std는3개seed성공률의동일가중평균/표본표준편차(ddof1)다.

## 실행·저장

```bash
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/evaluate_stage1_push_finetune.sh
```

스크립트의첫인자는다른checkpoint,둘째인자는새출력폴더다. 기본출력은 `output/stage1_base_push_finetune_eval9000_20261026`이다. 같은완료출력에덮어쓰지않는다. PUSHseed0→1→2를먼저실행한뒤AT/ON_TOP/SIT/CLIMB를각3seed씩별도프로세스로직렬실행한다. 실패하면중단하고GPU옵션을바꿔자동재시도하지않는다.

| 파일 | 내용 |
|---|---|
| `run_manifest.json` | checkpoint SHA/GPU/진행상태/각run명령/종료코드 |
| `RESULTS.md`, `summary.json` | 완료run마다갱신하는전체결과·PUSH거리구간표 |
| `<task>_seed<seed>/eval.log` | 해당프로세스로그 |
| `manifest.json`, `resolved_env.yaml` | checkpoint/source SHA·기본설정·평가override |
| `initial.npz` | root/관절위치속도/상자·크기·binding·목표 |
| `trace.npz` | 매steproot/양발/상자/목표·active/done/terminate·두성공flag·학습성공flag·dt |
| `progress.json`, `results.json` | 진행상태·완료된agent별결과·거리구간·artifact SHA |

`active`가true인첫episode frame만재채점해야한다. 종료된scene의후속reset은물리진행용이고평가분모/ever에추가하지않는다. 초기배치archive는이평가프로토콜내재현자료이며기존single_match archive와동일배치라고주장하지않는다.

## 검증 및 실행 상태

CPU성공식/거리경계/빈구간및에피소드기록6회귀통과. 5과제각2환경의실제CPU7step정책에서checkpoint복원·전체기록·고정상자pose·종료집계를확인했고저장궤적독립재채점도전부일치했다. GPU0 본평가는2026-10-26 tmux `koo_eval_push9000`에서시작했다. 로그는상위출력폴더 `batch.log`,접속은 `tmux attach -t koo_eval_push9000`다. 최종완료를아직주장하지않으며위출력폴더의상태/결과파일을따른다.
