# 평가 보조 작업

평가 확인·비교·후처리용 보조 스크립트는 이 폴더에서 만든다.

2026-11-10 V3 2k·seed1/6/7의 Table1/2 실행은 [전용 큐](../output_etc/check/v3_2000_tables12_seeds167_20261110/run_suite.py)를 사용한다. SCT26k·SCT+FSM을 같은 초기 장면으로 비교하며, 기존59회 재사용·신규67회로 총126회를 검증한다. Table2의 seed1/6은 `run_table2_extra.py 6`·`run_table2_extra.py 7`에서 먼저 실행하고 본 큐가 완료 검증을 재사용한다. 기존 V2 진입점·학습 코드는 변경하지 않는다. [실행 상태](../output_etc/check/v3_2000_tables12_seeds167_20261110/status.json), [GPU6 추가 큐](../output_etc/check/v3_2000_tables12_seeds167_20261110/extra_status_gpu6.json), [GPU7 추가 큐](../output_etc/check/v3_2000_tables12_seeds167_20261110/extra_status_gpu7.json), [비교표](../output_etc/check/v3_2000_tables12_seeds167_20261110/report.md).

```bash
/home/hwanhee/anaconda3/envs/tokenhsi/bin/python output_etc/check/v3_2000_tables12_seeds167_20261110/run_suite.py
```

완료된 seed 검증은 재사용한다. `supervisor.pid`의 큐가 종료된 뒤 이 명령으로 이어서 실행하며, 현재 진행 상황만 집계하려면 같은 폴더의 `summarize.py`를 실행한다. 최종 큐는 원본/checkpoint SHA·초기 장면·CPU 재채점을 확인한 뒤 `summary.json`·`report.md`를 생성한다.

- 메인 평가 진입점은 기존 [evaluate.py](../tokenhsi/evaluate.py)를 유지한다.
- 성공 판정·낙상 처리·집계는 기존 함수를 재사용한다.
- 로그·결과·trace·임시 산출물은 `output_etc/check/`에 저장한다.
- 평가 조건은 [논문 프로토콜](../cvpr2027_evaluation_protocol.md)을 따른다.
- Table 2는 `run_table2.py`·`table2_eval.py`로 별도 실행한다. 메인 eval의 성공 판정·낙상을 재사용하며 Enabling Time·Coupled t_coop만 보조 경로에 추가한다.

`table2_sampling.py`는 2명 Enable 한 쌍(AT→SIT/CLIMB/ON_TOP 각각) 또는 2/3/4명 공동 PUSH 한 그룹만 만든다. M=2N·공통 교집합·거리·ordered·distractor 관측 유지/물리 제거와 동일 초기 snapshot 검증을 사용한다. Stage1은 원본26k 정책 입력 변환을 재사용하고 Stage2는 V2 2k다. 각 GPU에서 Stage1·Stage2를 순서대로 실행하면 GPU당18회이며 reference 생성 후512환경×인자로 지정한 세 seed를 평가한다. 두 GPU에 서로 다른 설정을 배정해 reference 경합을 피한다.

```bash
python eval/run_table2.py --output output_etc/check/table2_20261109 --model stage1 --gpu 6 --settings coupled_4 enable_sit coupled_2
python eval/run_table2.py --output output_etc/check/table2_20261109 --model stage2 --gpu 6 --settings coupled_4 enable_sit coupled_2

python eval/run_table2.py --output output_etc/check/table2_20261109 --model stage1 --gpu 7 --settings enable_climb enable_ontop coupled_3
python eval/run_table2.py --output output_etc/check/table2_20261109 --model stage2 --gpu 7 --settings enable_climb enable_ontop coupled_3

python eval/summarize_table2.py output_etc/check/table2_20261109 --seeds 1 3 6
```

각 GPU에서는 위의 Stage1 명령 완료 후 Stage2 명령을 실행한다. seed 기본값1/3/6은 이번 임시 Table 2 실행용이며 공통 프로토콜의 확정 seed가 아니다. `--seeds`·`--envs`로 검사를 분리한다. Table 2는 XY100%를 보조 채점에 적용하고 학습 보상·기본 메인 기준은 유지한다.

`table2_metrics.py`는 Enabling 최초 동시 성공 Time(성공만), Coupled 최초 수평 변위20cm부터 최초 성공까지의 전원 양손 접촉 비율을 계산한다. 실패는 기록된 에피소드 끝까지 포함, 미시작0%, 낙상 이후 접촉0, 시작/끝 프레임 포함·30Hz다. 손 중심과 회전한 실제 공유 상자 표면의3D 거리6cm 미만이며 힘 조건은 없다. 8/12cm도 같은 raw 기록에서 산출한다. 각 trial 평균 후 행동/인원별 동일 가중 평균·seed별 평균±표본std를 계산하며 성공이 없는 Time은 N/A다. Table 2에 Error 열은 넣지 않는다.

Coupled의 `cooperation.npz`에는900프레임 전체의 양손 위치·실제 상자 위치/회전·크기를 저장한다. `trace.npz`의 유효 상태·목표와 결합해 문턱 변경을 CPU로 재집계한다. `table2_trials.npz`는 최초 성공·시작·분모/분자 프레임 수이며 GPU 기록과 CPU 재계산의 exact 일치를 확인한다. 저장 용량·압축 소요 시간은 `table2.json`에 기록한다.

```bash
python eval/rescore_table2.py output_etc/check/table2_20261109/stage2/coupled_3/general_seed1 --distance-cm 6 8 12
python eval/rescore_table2.py output_etc/check/table2_20261109/stage2/coupled_3/general_seed1 --distance-cm 6 8 12 --start-displacement-cm 40
python eval/rescore_table2.py output_etc/check/table2_20261109/stage2/coupled_3/general_seed1 --distance-cm 12 --start-displacement-cm 20 --start-hand-distance-cm 16
```

이 명령은 기본적으로 원본을 읽기만 한다. `--start-displacement-cm`은 밀기 구간의 시작 변위이며 기본20cm다. 시작 변위를 바꾸면 시작/분모/분자만 재계산하고 최초 성공 프레임·SR은 원본과 검증한다. 별도 결과가 필요하면 `--output output_etc/check/<결과>.json`을 지정한다.

`--start-hand-distance-cm 16`은 같은 프레임에 유효 참여자의 한 손 이상이 상자 표면16cm 미만인 조건을 시작 변위 조건과 함께 요구한다. 한 번 시작하면 유지하며 높이/접촉력 조건은 없다. 옵션 생략 시 기존 이동량만 사용하는 시작 조건이다. TeamHOI를 참고한 PUSH용 비교 옵션이며 기본 프로토콜은 변경하지 않는다.

시작20cm+손16cm 조건으로 저장 궤적을 재집계한 비교·seed별 결과는 [손 접근 시작 조건 비교](../output_etc/check/table2_handstart16_20261109/report.md)에 있다.

속도·지속 시간으로 시작을 정하는 비교는 다음 명령으로 CPU 재집계한다. 기본9조합은 XY 속도0.05/0.1/0.2m/s × 0.1/0.3/0.5초다. 초기 물리 상자 중심부터 프레임 간 XY 변위를 control dt로 나눈 속도를 쓰며 평활화하지 않는다. 연속 구간의 매 프레임에 유효 참여자 한 손 이상이 표면16cm 미만이어야 한다. 최초 성공까지 조건이 확인되면 시작을 연속 구간 첫 프레임으로 소급한다. 변위20cm 조건은 대체하며 공동 접촉은 전원 양손12cm 미만·실패 포함·미시작0%다. 원본 최초 성공/12cm 접촉 프레임 수를 exact 검증하고 기본 프로토콜은 변경하지 않는다.

```bash
python eval/rescore_table2_speed.py --output output_etc/check/table2_speed_start_20261109 --speeds 0.05 0.1 0.2 --durations 0.1 0.3 0.5
python eval/rescore_table2_speed.py --output output_etc/check/table2_speed_02_10_1s_nohand_20261109 --speeds 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --durations 1 --no-start-hand
```

`summary.json`·`report.md`에 인원별/seed별 t_coop와 시작률, 실행별 JSON·NPZ에 조합별 시작/분모/분자 프레임 수를 저장한다. [속도·지속 시간 비교 결과](../output_etc/check/table2_speed_start_20261109/report.md).

`--no-start-hand`는 시작의 손 접근 조건만 제거하고 XY 속도·지속 시간만 사용한다. 전원 양손 접촉12cm 판정은 유지한다. `--start-hand-cm`과 동시 지정할 수 없다. [손 접근 조건 제거 비교](../output_etc/check/table2_speed_02_10_1s_nohand_20261109/comparison.md).

이번 Stage1 26k·Stage2 V2 2k의 Table 2 36회 결과는 [report.md](../output_etc/check/table2_20261109/report.md), 초기 장면·CPU 재집계·기록 검증은 [verification.json](../output_etc/check/table2_20261109/verification.json)에 있다.

시작 변위만40cm로 바꾼 CPU 재집계의20cm 비교·seed별 수치는 [시작 변위 비교](../output_etc/check/table2_start40_20261109/report.md)에 있다. 기본 프로토콜은20cm로 유지한다.

일반 `*_test.sh`·`*_vnc.sh`는 학습 장면 샘플러와 보상 기반 viewer 진단을 사용한다. `TASK_GRAPH=random_scenario`만으로 논문 프로토콜의 공통 교집합·균등 관계/행동·near50 해제·distractor 물리 제거가 적용되지는 않는다. Viewer의 성공률/낙상 처리도 formal SR과 다르며, `EVAL_PROTOCOL=general`은 headless 전용이라 일반 VNC에 붙이는 방식으로 전환할 수 없다.

`compare_region_extent.py`는 저장 궤적의 XY 중앙80%·전체100% 성공률을 CPU에서 비교한다. 기존80%의 SR·ever/simultaneous 재현과100% 성공의 포함 관계를 검증하며, 메인 기준·정책·원본 결과·Error는 변경하지 않는다.

`run_stage2_scalability.py`는 `stage2_scalability_eval.py`로 Independent 전용·Mixed를 분리해 2/4/8/16명 평가를 실행한다. 초기 snapshot·매 frame 물리/관측 분리·CPU 재채점을 검증하며, 출력은 `output_etc/check/`에 둔다. 2명 제한 완화와 5명 이상 그룹별 distractor 배분·관측 보존은 보조 경로에만 적용한다. `stage2_xy100.py`는 채점·Error의 XY만100%로 바꾸고 학습 보상·관측 계산은 유지한다. `summarize_stage2_scalability.py`는 기존4H Mixed XY100% 결과를 재사용해 인원수별 표를 만든다. seed는 실행 인자로 지정하며 프로토콜 기본값을 변경하지 않는다.

5명 이상 초기 공동 상자는 그룹 대표만 쓴다. 여러 참가자가 같은 물리 상자에 동시에 다른 초기 상태를 쓰던 중복을 막으며, 최종 공동 PUSH 후보·거리·ordered 배정은 기존 함수를 사용한다. `STAGE2_SAMPLING_AUDIT=1`은 초기 배치 단계별 상태/RNG 비교 파일을 추가하며 기본은 꺼져 있다.

`stage2_full_observation.py`는 관측용 distractor를 encoder attention에도 포함시킨다. 기존 `MAPlayerContinuous`의 5명 이상 task-slot mask는 이 보조 경로에서 해제하며 encoder 호출마다 확인한다. 이 수정 이전 8·16명 결과는 distractor attention 조건이 다르므로 재평가 결과와 구분한다.

Stage1 비교는 같은 runner에 `--stage1-checkpoint <Stage1_00026000.pth>`와 `--reference-evaluations <기존Full평가폴더>`를 추가한다. `--checkpoint`는 동일 초기 장면 생성을 위한 Full checkpoint이며, 첫 행동 전에 Stage1 가중치·정규화를 복원한다. `stage1_protocol_policy.py`는 enable의 목표 정렬 가상 받침과 coupled의 개인별 PUSH 구역을 입력하고 Stage2 coordination을 우회한다. 원본 Stage1 network와 행동 평균/표준편차 exact 일치·가중치 불변·coordination 호출0·초기 물리/관측 일치를 검증한다. Stage1과 Full 모두 모든 관측 토큰을 attention에 포함한다.

`compare_stage1_scalability.py <Stage1평가폴더> <Full재평가폴더> <기존Full평가폴더>`는 Stage1 전체·Full8/16H의 검증 완료를 확인하고 기존 Full2/4H를 읽기만 해 비교표와 seed별 SR·그룹 수를 만든다. 원본 결과는 변경하지 않는다.

```bash
python eval/run_stage2_scalability.py --checkpoint <checkpoint.pth> \
  --output output_etc/check/<평가명> --gpu 6 --seeds <seed1> <seed2> <seed3> \
  --jobs independent:2 independent:4 mixed:2 independent:16
python eval/run_stage2_scalability.py --checkpoint <checkpoint.pth> \
  --output output_etc/check/<평가명> --gpu 7 --seeds <seed1> <seed2> <seed3> \
  --jobs mixed:16 independent:8 mixed:8
python eval/summarize_stage2_scalability.py output_etc/check/<평가명> \
  --seeds <seed1> <seed2> <seed3> --mixed4-cache <기존XY100비교/comparison.json>
```

```bash
/home/hwanhee/anaconda3/envs/tokenhsi/bin/python eval/compare_region_extent.py \
  output_etc/check/stage2_distill_900_v2_2000_new_protocol_seeds678_20261108 \
  output_etc/check/stage2_v2_2000_xy100_seeds678_20261108
```

`stage2_protocol_eval.py`는 기존 검증된 새 프로토콜 샘플러·distractor 관측/물리 분리 hook을 재사용하는 **4명·512환경 전용** 진입점이다. `stage2_distractor_physics.py`는 기준 snapshot 경로를 제외하면 기존 검증 hook과 같다. 메인 채점·Error·CPU 재채점은 공통 코드를 사용한다.

기준 경로 기본값은 `output_etc/check/stage2_new_protocol_1500_env512_20261108/reference/`다. 새 seed는 `STAGE2_REFERENCE_ROOT=output_etc/check/<평가명>/reference`를 지정하고, 아래 명령을 `REFERENCE_ONLY=1`·출력 `<기준경로>/general_seed<seed>`로 먼저 실행해 초기 snapshot·관측만 저장한다. 이후 `REFERENCE_ONLY=0`·출력 `<평가명>/general_seed<seed>`로 같은 checkpoint·설정의 실제 평가를 실행한다. 기준 생성은 rollout·성공 집계를 하지 않는다.

```bash
TOKENHSI_GPU=6 bash -c '
source tokenhsi/scripts/multi_agent/runtime_env.sh
python eval/stage2_protocol_eval.py \
  --task HumanoidMACoupledPush --cfg_train tokenhsi/data/cfg/train/rlg/stage2_distill_900_v2.yaml \
  --cfg_env "" --motion_file tokenhsi/data/dataset_stage1_jointpush.yaml \
  --checkpoint "$1" --output_path "$2" --seed "$3" --num_envs 512 \
  --test --eval --headless --no_video --eval_protocol general --eval_repeats 1 \
  --task_graph random_scenario --enable_size_mode per_action_intersection --coupled_assignment ordered
' _ <checkpoint.pth> output_etc/check/<평가명>/general_seed0 0
```

크기 CLI는 기존 hook 진입용이며 실제 공유 상자는 전체 공통 교집합이다. 결과에는 `common_intersection`·`cvpr2027_mixed_v1`·동시 성공 기준·두 Error가 기록된다. 지정한 seed를 각각 실행한 뒤 다음 보조 코드로 초기 상태·CPU 재채점·seed 통계를 확인한다. 평가 폴더에는 checkpoint·reference·seeds를 기록한 manifest가 필요하다.

```bash
PYTHONPATH=tokenhsi /home/hwanhee/anaconda3/envs/tokenhsi/bin/python \
  eval/summarize_stage2_protocol.py output_etc/check/<평가명>
```
