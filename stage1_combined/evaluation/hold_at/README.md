# Chair HOLDING+AT 단독 평가

기준은 [evaluation_single.md](../../../docs/evaluation_single.md)의 AT 성공식·시간·집계다. 프로토콜은 `native_chair_hold_at_30s_v1`이며, 일반 박스 `single_match_region_v1`과 초기 배치 분포를 구분한다.

| 항목 | 설정 |
|---|---|
| 장면 | 환경당 사람1명·동적 ArmChair1개, distractor 없음 |
| 물체 | 원본 ArmChair_Normal test27종, 환경별 균등 복원추출·native 크기/질량/mesh |
| 표본 | 512환경 × 독립 seed0/1/2 별도 프로세스 |
| 초기화 | loco100%, reference motion의 heading 유지, cuboid RSI cache 없음 |
| 배치 | 사람–의자 XY 반경 U(1,5)m, 의자 metadata front가 사람 위치를 향함 |
| 목표 | 환경 origin 기준 XY±4.5m, 사람/의자와 squared XY distance≥2m², 바닥 중심높이 |
| 시간 | episode_length900·dt1/30초, 일반 시간종료899step |
| 추론 | 저장한 모델/정규화 고정·deterministic, optimizer 없음 |
| 성공 | **의자 중심–목표 XYZ의3D 거리≤20cm**. AT 점/영역 동일 |
| ever | 첫 action 이후부터 첫 episode 종료 frame까지. 초기 frame 제외, 후속reset 없음 |
| 종료 | 기존 낙상 종료 유지, 성공 즉시 종료하지 않음 |
| 통계 | seed별 성공률의 평균±표본표준편차(ddof=1) |

HOLDING+AT는 학습 task 이름이다. 이 평가에서는 문서에 따라 **AT 위치 도달**을 채점한다. 파지·접촉·손 놓기·직립·속도·유지시간·학습 own_success를 추가하지 않는다. 안정적으로 들고 유지하는 성공률이라고 해석하지 않는다. 일반 박스 AT의 U(1,10)m·랜덤yaw·선반높이와 달리, 여기서는 현재 Chair 파인튜닝의 native 배치·바닥목표를 유지한다.

## 실행

프로젝트 루트에서 실행한다. checkpoint에 저장된 env/sim/train 설정을 복원하며, 위 평가 인원·개수·test split·시간·loco·환경 수만 바꾼다. 30D object tokenizer를 그대로 사용한다. 출력은 새로운 `runs/` 디렉터리여야 한다.

```bash
# 설정·checkpoint·test27 에셋·코드 사본만 준비. GPU 실행 없음.
python3 stage1_combined/evaluation/hold_at/run_eval.py --checkpoint "$CKPT" --prepare-only

# 정식 평가: 기본 GPU5, 순차 실행 512환경×seed0/1/2
TOKENHSI_GPU=5 bash stage1_combined/tokenhsi/scripts/multi_agent/chair_hold_at_full_ft_eval.sh "$CKPT"

# 원본78k zero-shot: template는 Chair 환경설정만 제공, 가중치/정규화는 원본 그대로
python3 stage1_combined/evaluation/hold_at/run_eval.py --checkpoint stage1_combined/checkpoints/Stage1V2Fixed_00078000.pth --template-checkpoint "$CHAIR_FT_CKPT" --gpu 5

# 실행 검증: 4환경×seed0, 시간은900step. 정식 성능 수치 아님.
python3 stage1_combined/evaluation/hold_at/run_eval.py --checkpoint "$CKPT" --gpu 5 --smoke

# 저장 trace의 독립 재채점
python3 stage1_combined/evaluation/hold_at/metrics.py "$RUN/source/output_etc/check/hold_at_seed0"
```

현재 GPU5의두학습과평가는동시에추가하지않는다. 현재 실행은GPU5만허용하며`--gpu 5`/`TOKENHSI_GPU=5`로명시한다.기존프로세스나MPS를중단하지않는다. runtime은이미검증된MPS에만연결하고worker는CUDA동기화후자연종료한다. ksh/공유원본/실행중인학습사본은수정하지않는다.

## 산출물과 검증

배치별 `checkpoint.pth`, `env.yaml`, `train.yaml`, `preparation.json`, `asset_sha256.json`, `run_manifest.json`, `seed*.log`, `records.json`, `summary.json`, `completed.json`을 보존한다. seed별로 `resolved_env.yaml`, `manifest.json`, `initial_0.npz`(root·DOF·물체·크기·모델ID·목표), `trace_0.npz`, `results.json`, `completed.json`, `rescore_validation.json`을 남긴다.

`metrics.validate`는 위치에서성공식을다시계산하고첫episode mask·종료frame·초기분포·시간종료899step·float32/64 ever일치·artifact/checkpoint해시를검사한다. CPU테스트는20cm3D경계,종료frame성공,후속episode제외,불완전episode거부,ddof1통계,실제rollout adapter의reset1회와재채점을확인한다.

기존 `chair_hold_at_full_ft_test.sh`는16환경 viewer/기존player진단용이다. 정식 지표에는 이 디렉터리의 전용 `*_eval.sh`를 사용한다.

## 실행 검증 기록

2026-10-26 CPU5개통과. GPU0에서4환경×seed0·900step설정의실제모델추론/물리/정상종료rc0와전trace독립재채점통과. `runs/chair_hold_at_eval_smoke_20261026_152916`에초기DOF/trace/해시/결과보존. 시간종료899step과낙상조기종료406step을확인했다. 검증용1회업데이트checkpoint를사용했으며성능평가결과로사용하지않는다. 이 검증 이후의 원본78k 정식512×3평가는 아래 기록을 따른다. GPU5의기존두학습과MPS는유지했다.


## 원본78k zero-shot 정식 결과

2026-10-26 GPU0,원본78k가중치/정규화직접로딩·Chair설정템플릿만사용. test27·512×독립seed0/1/2·30초·위native프로토콜. 점=영역ever **31.64±1.86%**, 성공171/163/152명. 전trace재채점·각seed27종포함·초기root다름·원본checkpointbyte동일·통계/해시검산통과. `runs/chair_hold_at_78k_zeroshot_20261026_153255/REPORT.md`와`validation.json`참조. 파인튜닝checkpoint의정식평가는별도다.


## 원본 TokenHSI Chair adaptation 정책

`--public-chair-checkpoint /home/hwanhee/CVPR2027/TokenHSI/output/tokenhsi/ckpt_stage2_objShape_chair.pth`를기존78k scene/template 인자에추가한다. scene checkpoint는공통장면/player생성만하며행동은원본Chair10000epoch정책만계산한다. 원본Stage1은40k보조checkpoint로복원하고최종Chair102개tensor·265D RMS를strict load한다. 관측은원본223D human+42D carry,deterministic action과rescale도원본과같다. 원본network4개파일해시를baseline과대조하며관측함수AST를독립사본에서로드한다. 추가정책구성시CPU/CUDA RNG를fork/복원해공통장면초기화를보존한다.

준비사본 `runs/tokenhsi_chair_hold_at_20261026_prepared`의`public_cpu_validation.json`에서102개tensor/정규화bitwise동일·첫/후속관측forward·가중치동결을확인했다. GPU5에서512×3평가는기존두학습유지하에사용자가이번평가에한해세번째잡을명시승인했다. 이예외를다음실험에재사용하지않는다.


공개Chair정식결과(2026-10-26,GPU5): **98.31±0.60%**,성공500/506/504명. 원본265D/102tensor모델동결·각512환경·독립seed0/1/2·30초·위native AT평가. 기존78kzero31.64±1.86%와세seed의초기root/DOF/물체/modelID/목표등전배열exact일치. 전seedrc0/trace재채점/통계해시검산통과. `runs/tokenhsi_chair_hold_at_20261026_prepared/REPORT.md`와`validation.json`참조. GPU5평가client자연종료후기존두학습client만유지확인.


## 원본 TokenHSI Eval 결과 (공통 AT와 별도)

후속원본평가요청은원본`HumanoidAdaptCarryBox2Objs`환경·YAML·player루프의독립사본으로실행했다. GPU5/512환경/seed0/연속3repeat/20초/위치20cm+직립updot≥.86615. 성공501/503/499,0회제외1·2평균 **97.85%**(반복sample std0.55%p). `runs/tokenhsi_chair_official_20261026_160009/REPORT.md`및`validation.json`참조. 공통30초위치-only AT98.31±0.60%와프로토콜을구분한다.


**논문 수치와 구분:** 위공개코드측정97.85%는논문Table3의Chair88.8±3.1%재현결과가아니다. 논문은randomorientation/10×10m목표,공개코드는front사람정렬/9×9m목표다. 이차이가수치차이원인인지는미확정이다. `runs/tokenhsi_chair_official_20261026_160009/PAPER_AUDIT.md`참조.
