# Table CLIMB 공통 40초 paired 평가

## Table carry 40초 비교

공개 TokenHSI Table carry(배포pth 내부epoch40000)와 우리 encoder고정+action residual10k를 Table_Circle test9·512환경·seed0·40초로 평가한다. 사람1명/동적Table1개, loco시작, 기존Table의거리1~5m·랜덤yaw·목표XY±4.5m/바닥높이를 유지한다. 우리AT영역식은점식과같은 물체중심–목표3D거리≤20cm이며 직립·파지·접촉·정지·유지시간을 성공조건에 추가하지 않는다.

```bash
PYTHONPATH=stage1_combined /home/injesus1010/anaconda3/envs/tokenhsi118/bin/python \
  -m evaluation.climb_paired.run_table_carry --seed 0 --gpu 0
```

기본우리checkpoint는 `output/hold_at_adaptation/table/table_adaptation_compare_20261027/residual/train/TableAdaptResidual_27-17-21-38/nn/TableAdaptResidual_00010000.pth`이며 `--checkpoint` 또는 `--checkpoint10000`으로 지정할 수 있다. 공개정책 기본은 배포캐시의 `ckpt_stage2_objShape_table.pth`; `--public-checkpoint`와 `--public-base-checkpoint`를 지원한다. 같은우리checkpoint를장면생성에쓰고행동정책만선택한다. 우리모델171tensor/scene RMS,공개모델102tensor/265D RMS를 strict복원하고평가전후불변을검증한다.

두정책을순차실행하며각각원본제어루프에서시드재설정없이연속3회평가한다. 첫episode의action후부터종료frame까지ever,명목1200step/시간종료1199action이다. 대표repeat1/2 평균±반복표본std와repeat0포함3회전체통계를함께저장한다. 시드간std가아니다. 첫반복의root/DOF/물체/목표/크기/ID전배열은정책사이exact검증하고후속반복은부분reset/RNG소비에따른차이를기록한다.

결과는 `output_etc/check/hold_at_eval/table_compare_40s_<시각>/`의REPORT/summary/repeats.csv와정책별repeat하위initial/trace/독립재채점으로남는다. 진행은저장소루트 `runs/table_carry_40s_active.json` 및출력state.json/정책별progress.json을확인한다. `--prepare-only`는고정사본만준비하고`--smoke`는4환경의같은40초×3repeat연결검증이다. 앞서측정한원본20초+직립평가와구분한다.

## Chair HOLD+AT 40초 비교

공개 TokenHSI **chair carry adaptation10k**와 우리 **encoder고정+action residual8500/10000**을 같은 동적 ArmChair test27·512환경·seed0·40초로 비교한다. SIT 의자26종 평가와 별도다. 우리 AT 기준은 의자 중심–바닥 목표점의 3D 거리≤20cm이며 점/영역이 같다. 직립·파지·접촉·손놓기·속도·유지시간 조건은 추가하지 않는다. 초기 frame은 제외하고 첫 episode의 종료 action까지 ever로 집계한다.

```bash
PYTHONPATH=stage1_combined /home/injesus1010/anaconda3/envs/tokenhsi118/bin/python \
  -m evaluation.climb_paired.run_chair_carry --seed 0 --gpu 0
```

기본 checkpoint는 `output/hold_at_adaptation/chair/chair_residual_after_table_20261028_recovery/train/ChairAdaptResidual_28-11-08-36/nn/`의 실제 epoch8500/10000이며 `--checkpoint8500`/`--checkpoint10000`으로 지정할 수 있다. 공개정책 기본은 로컬 배포 캐시의 `ckpt_stage2_objShape_chair.pth`이고 `--public-checkpoint`와 `--public-base-checkpoint`로 바꿀 수 있다. 원본265D human+carry 관측/102tensor/RMS를 strict 복원하며, 공개정책 생성 중 RNG를 보존한다.

세 정책을 순차 실행하고 각 정책은 시드 재설정 없이 원본 루프에서 연속3회 평가한다. 대표값은 repeat1/2 평균±반복표본std이며 repeat0 및 3회 전체 통계도 보존한다. 시드간 std가 아니다. 사람–의자1~5m/metadata front정렬/목표XY±4.5m의 기존 carry 초기분포를 유지한다. 첫 반복의 root/DOF/의자/목표/크기/ID 전배열은 정책 사이 exact 비교하고, 후속 반복은 부분reset의 RNG소비로 달라질 수 있어 실측 일치 여부를 저장한다.

기본 출력은 `output_etc/check/hold_at_eval/chair_compare_40s_<시각>/`이다. `REPORT.md`, `summary.json`, `repeats.csv`와 각 정책/repeat의 initial/trace/독립재채점을 보존한다. 진행은 `runs/chair_carry_40s_active.json`(저장소 루트)과 결과폴더 `state.json`, 각 정책의 `progress.json`에서 확인한다. `--prepare-only`는 고정사본/해시/설정만 준비하고, `--smoke`는 4환경×40초×연속3회의 구현 검증이다. 논문 원본환경/시간 전체를 재현하는 평가는 아니다.

## Chair SIT 단일 시드 평가

Chair SIT는기본으로우리모델을시드0하나로측정한다. `--policy tokenhsi`를붙이면같은장면생성용checkpoint를유지하면서행동만공개TokenHSI Stage1 40k의원본354D 정책으로계산한다. 공개정책은별도고정사본을만들고실제policy epoch와장면checkpoint epoch를구분해기록한다. 서버에서학습한pth를로컬에서접근가능한경로에놓은후저장소루트에서실행한다. 부모폴더의이름으로학습횟수를추정하지않고실제epoch와30D/35D구조를기록한다. 모델/RMS를strict복원하며가중치변환은하지않는다. 특정장면checkpoint epoch가필요하면`--expected-epoch 20000`처럼명시할수있다. 사용자지정`ChairBBox35TokenizerFT_00002000.pth`의실제epoch는2000이다.

95%대기록의SIT전용FullFT20k는현재`output/sit_adaptation/tokenhsi_sit_full_ft/tokenhsi_sit_78k_20261024_130732/TokenHSISitFullFT_24-13-07-36/nn/TokenHSISitFullFT_00020000.pth`다. 내부epoch20000/source78k/trainable=all_layers/SIT100%를확인했다. 이파일을`CKPT`로지정하고`--expected-epoch 20000`을붙여평가한다.

```bash
PYTHONPATH=stage1_combined /home/injesus1010/anaconda3/envs/tokenhsi118/bin/python \
  -m evaluation.climb_paired.run_chair_sit --checkpoint "$CKPT" --seed 0 --seconds 40 --gpu 0
```

공개TokenHSI SIT도동일기준으로측정:

```bash
PYTHONPATH=stage1_combined /home/injesus1010/anaconda3/envs/tokenhsi118/bin/python \
  -m evaluation.climb_paired.run_chair_sit --checkpoint "$CKPT" --policy tokenhsi --seed 0 --seconds 40 --gpu 0
```

`CKPT`는위FullFT20k처럼비교할우리모델의장면생성용파일이다. 공개정책파일은기본로컬배포캐시를사용하며`--public-checkpoint`로지정할수있다. 공개정책결과폴더는`tokenhsi_sit_epoch<policy epoch>_40s_continuous3_<시각>/`다. 원본논문조건전체의재현이아닌우리영역/시간기준평가다. 첫반복의초기배열은비교가능하지만이후반복은정책별부분reset으로달라질수있다.

Chair test26·512환경·1명/고정의자1개·loco시작·40초·연속3회다. 우리영역성공은골반의의자좌판로컬XY내부80%및좌판+10cm목표Z±20cm이며첫episode초기/종료frame포함ever다. 시드재설정없이원본반복루프를사용한다. 첫회repeat0을보존하고대표값은repeat1/2평균±반복표본std다. 3회전체통계와점20cm보조점수도저장한다. 시드1개이므로시드간std를계산하지않는다.

결과는`output_etc/check/chair_sit_eval/chair_epoch<실제epoch>_40s_continuous3_<시각>/`의REPORT.md/summary.json/results.json/repeats.csv에기록하며각repeat폴더에initial/trace/독립재채점을보존한다. state.json과progress.json에서진행을확인한다. `--prepare-only`는실행전검사·고정사본생성만수행한다. `--smoke --seconds 1`은4환경의성능이아닌연결검증이다. 지정pth가없으면실행하지않고경로오류를출력한다.

## Table CLIMB 비교

현재사용자요청은 **시드마다한프로세스에서연속3회**다. 아래명령은공개TokenHSI40k와우리원본30D2k를Table test8/512환경/40초에서seed0~9각각3repeat로평가한다. `native_loop.py`가읽기전용원본Transformer player의run_eval제어흐름을추출하고`continuous.py`가공통점·영역성공과trace기록을추가한다. 시드는프로세스시작시에만설정하며반복사이재설정하지않는다. 종료된환경을루프안에서리셋하는원본동작도유지한다.

```bash
PYTHONPATH=stage1_combined /home/injesus1010/anaconda3/envs/tokenhsi118/bin/python \
  -m evaluation.climb_paired.run_continuous --seconds 40 --seeds 0 1 2 3 4 5 6 7 8 9 --gpu 0
```

원본운영집계규칙에따라repeat0도저장하되대표평균에서는제외하고 **repeat1·2평균±반복표본std**를시드별로기록한다. `results.json`에3회전체평균·std도함께보존한다. 전체`summary.json`에는각시드대표평균들의평균과 **시드간표본std**를기록하며두std를섞지않는다. `repeats.csv`는각시드·정책·반복의원시점수,`seeds.csv`는시드별대표값과반복std다.

첫반복의초기root/DOF/object/model ID는모델간exact검증한다. 그뒤에는정책별조기종료와부분reset횟수차이로RNG소비가달라질수있으므로repeat1·2의초기장면이동일하다고가정하지않으며실제배열비교를기록한다. 원본의연속reset규칙을유지한공통장면·40초평가이며논문환경/시간전체를재현한평가는아니다. 리셋된환경의TokenHSI관측에는해당환경의새kinematic body를사용한다.

기본출력은`output_etc/check/climb_eval/table_30d_2k_continuous3_<시각>/`이며기존1회평가결과는보존한다. `state.json`은현재seed·정책·완료seed쌍수,각case의`progress.json`은현재repeat/프레임을표시한다. `--smoke --seconds 1`은4환경의짧은3회흐름검증이며성능결과가아니다.

아래는이전의 **시드별1회**평가명령이며새연속3회결과와구분한다.

공개 TokenHSI Stage1 40k와 지정한 원본30D FT checkpoint를 동일 장면에서 비교한다. 기존 `TokenHSIClimbFullFT_00020000.pth`는35D이므로 이 평가의 대상에서 제외한다. Table_Square test8, 1명/고정물체1개, 512환경, loco 시작, deterministic action, 첫 episode 초기·종료 frame 포함 ever 기준이다. 40초는 30Hz에서 episodeLength1200이며 기존 timeout convention으로 마지막 action은1199일 수 있다.

대표 지표는 골반이 물체 로컬 XY 내부80%와 metadata+.94m 높이±20cm에 들어온 영역 성공이다. 골반과 목표의 3D 거리20cm 점 성공도 함께 기록한다. 접촉·발·유지시간 조건은 없다. 원본 논문의 평가 조건 재현과 구분한다.

저장소 루트에서 실행:

```bash
PYTHONPATH=stage1_combined /home/injesus1010/anaconda3/envs/tokenhsi118/bin/python \
  -m evaluation.climb_paired.run_eval --checkpoint "$CKPT" --seconds 40 --seeds 0 1 2 3 4 5 6 7 8 9 --gpu 0
```

사용자요청의30D 2k/3k 비교 후 시드평가는 아래 명령이다. 먼저40초·Table8·seed0/1/2에서3k와TokenHSI를평가하고, 같은TokenHSI결과를검증해재사용하여2k를평가한다. 영역평균이3k에서엄격히높으면3k를선택하고동률/하락이면기존2k를유지한다. 선택된checkpoint는seed3~9도전부평가하고선택된모델의0~9전체를통합한다. checkpoint선택에쓴pilot과추가seed를보고서에서구분한다.

```bash
PYTHONPATH=stage1_combined /home/injesus1010/anaconda3/envs/tokenhsi118/bin/python \
  -m evaluation.climb_paired.compare_checkpoints --gpu 0
```

과거30초Mixed(Box18/Cabinet9/Table8) 결과는2k99.80±.34%,3k99.09±.49%였다. 이번Table단독40초결과와직접같은조건으로간주하지않는다. `--public-only`는공개TokenHSI만평가하며기본78k checkpoint는장면생성에만쓴다. `--reference-dir`은동일protocol/config/공개pth와검증된trace를재사용한다. 이후우리정책의초기전배열이정확히일치해야paired결과로인정한다.

`--smoke`는4환경/첫seed/전체40초 검증이며 성능 결과로 사용하지 않는다. `--prepare-only`는 checkpoint/config/code 사본만 만든다. `--output`은 `stage1_combined/output_etc/check/` 내부 새 경로만 허용한다. 기본 출력은 그 아래 `climb_eval/table_paired_40s_<시각>/`이다. GPU 여유6500MiB 미만이면 기존 프로세스를 유지하며 대기한다.

시드 목록은 실행 전에 고정하고 전부 평가하며, 이기는 시드를 찾았다고 조기 종료하지 않는다. `seeds.csv`에 모든 시드의 두 모델 점수와 차이, `summary.json`에 전체 평균·평가seed 표본std(ddof1) 및 승리/동률/패배 시드를 남긴다. 승리 시드는 탐색용 진단이며 대표 성능으로 선택하지 않는다. 학습seed 산포와 구분한다.

각 시드에서 TokenHSI를 먼저 평가한 뒤 우리정책을 평가한다. 같은30D checkpoint는 양쪽 장면 생성에 사용하며 TokenHSI 행동은 원본354D 정책만 계산한다. public network 생성 시 RNG를 보존하고 초기root/DOF/object/model ID를 bitwise 비교한다. 추가입력/파라미터확장없이원본30D 가중치를strict복원한다. 원본 bbox와 metadata 목표는 보존한다. RTX5090의 기존 fused attention 오류를 피하도록 두 정책 모두 math attention을 사용한다.

각 case의 `initial.npz`, `trace.npz`, `results.json`, `independent_validation.json`, `run.log`를 저장한다. SciPy rotation으로 전체 trace를 독립 재채점하고 첫episode/종료frame·시간·finite·가중치/RMS 불변·checkpoint hash를 검증한다. 소스 변경 감지 시 이후 평가를 중단하며 원시 기록은 보존한다.
