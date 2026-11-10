# Changelog

현재 실행법은 [config](markdowns/config.md), 코드 위치는 [structure](markdowns/structure.md)를 따른다. 정리 전 문서·변경 기록은 [보관 자료](markdowns/legacy/pre_four_configs/README.md)에 있다.

## 2026-11-10

### V3 2k ·seed1/6/7 Table1/2 평가 완료

- GPU6/7에서512환경·900step·XY100%의 V3 2k/SCT26k/SCT+FSM Table1/2 총126회(신규67·재사용59)를 모두 완료했다. 최종 실패0·모든 큐 종료, 초기 상태/전체 관측 exact·CPU SR/건수/Error 검증·모델별 그룹 수 일치·seed 평균/표본표준편차 일치·원본/checkpoint SHA 불변을 확인했다. Table2 신규 결과의 CPU Time/접촉 재집계와 FSM replay도 통과했다. 기존 학습은 보존했다.
- V3 Table1 Independent2/4/8/16H는97.88±0.54/97.59±0.28/95.97±0.07/80.43±0.12%, Mixed는90.27±0.77/86.97±0.87/84.66±0.87/71.62±0.64%다. Table2 Enable 평균 SR77.63±1.05%·성공 Time11.24±0.23s, Coupled 평균 SR91.41±1.09%·20cm 시작/양손6cm 접촉 t_coop40.21±0.93%다. [전체 비교·seed별 결과](output_etc/check/v3_2000_tables12_seeds167_20261110/report.md), [완료 검증](output_etc/check/v3_2000_tables12_seeds167_20261110/completion_verification.json).

### V3 2k ·seed1/6/7 Table1/2 평가 시작

- 사용자 요청으로 GPU6/7에 V3 2k·SCT26k·SCT+FSM의512환경·900step·XY100%·M=2N 평가 큐를 시작했다. Table1 Independent/Mixed2/4/8/16H와 Table2 Enable3종/Coupled2/3/4H의126회 중 기존59회를 재사용하고67회를 추가한다. 기존 seed1/6과 V3의4H Mixed 초기 reference exact 및 대표5개 재사용 경로의 검증을 확인했다. Table2 seed1/6은 GPU별2개 추가 worker로 먼저 실행하며, 완료한 검증은 본 큐가 재사용한다.
- 실행은 [전용 큐](output_etc/check/v3_2000_tables12_seeds167_20261110/run_suite.py)·추가 Table2 큐와 기존 보조 평가/CPU 검증을 사용한다. 메인 코드·학습·V2 실행기·공통 MD seed는 변경하지 않았다. 재사용 symlink의 검증 파일 경로 오류를 수정하면서 자체 큐 부모만 교체했고 진행 중이던4개 평가 자식은 보존·인계했다. 최초 오류 기록은 `first_attempt_status.json`에 보존했다. 구문·대표 재사용 검증·부분 집계를 확인했으며 전체 성능은 아직 완료 결과가 아니다. [상태](output_etc/check/v3_2000_tables12_seeds167_20261110/status.json), [집계표](output_etc/check/v3_2000_tables12_seeds167_20261110/report.md). 실행 안내는 `eval/README.md`에 기록했다.

### V3 2.5k ·512환경 seed1/6/7/8 평가

- 사용자 요청으로 V3 2.5k를4명 Mixed·8토큰·512환경·900step·XY100%에서 GPU6(seed1/6)·GPU7(seed7/8)로 평가했다. 같은 seed의2k 초기 snapshot을 재사용했고4개 worker·최종 집계 모두 exit0다. 기존 학습·메인 코드·공통 MD seed는 변경하지 않았다.
- Independent/Enabling/Coupled/Mixed는97.18±0.97/71.09±3.17/91.83±1.00/86.70±1.27%다. 동일4seed의2k 대비 각각-0.21/-1.96/+1.53/-0.22pp이며 Independent는 Mixed 안 solo다. [seed별·2k 비교](output_etc/check/stage2_distill_900_v3_2500_mixed_env512_seeds1678_20261110/report.md).
- 4회 초기 상태/전체 관측 exact·각901회 물리 분리·attention 유지·CPU SR/건수 exact·Error 일치,2k와 초기 상태 모든 필드/그룹 수 exact·checkpoint/기존 산출물 SHA 불변을 확인했다. [검증](output_etc/check/stage2_distill_900_v3_2500_mixed_env512_seeds1678_20261110/verification.json).

### V3 2k ·512환경 seed9 /2.5k ·64환경 seed0–8 평가

- 사용자 요청으로 같은 V3 run의2k512환경 seed9와 새2.5k(epoch2500/frame327680000·가중치 finite)의64환경 seed0–8을 GPU6/7에서 평가했다.4명 Mixed·8토큰·900step·XY100%를 유지했다.2.5k는1.5k/2k의 seed별 초기 reference를 재사용했고6개 worker와 두 집계 모두 exit0다.11월9일 시작한 산출물 경로를 유지한다.
- 2k512환경 seed9의 Independent/Enabling/Coupled/Mixed는98.38/70.79/91.10/86.76%다.2.5k64환경9seed는97.73±1.71/74.19±6.87/91.64±4.44/87.85±3.03%이며2k 대비 Mixed +1.61pp다. Independent는 Mixed 안 solo다. [512환경 seed9](output_etc/check/stage2_distill_900_v3_2000_mixed_env512_seed9_20261109/report.md), [2.5k seed별·세 checkpoint 비교](output_etc/check/stage2_distill_900_v3_2500_mixed_env64_seeds0to8_20261109/report.md).
- 신규10회 초기 상태/전체 관측 exact·각901회 물리 분리·attention 유지·CPU SR/건수 exact·Error 일치,2.5k/2k 초기 상태 모든 필드 exact·그룹 수 커버리지·checkpoint/기존 결과 SHA 불변을 확인했다. 기존 학습·메인 코드·공통 MD seed는 변경하지 않았다. [2.5k 검증](output_etc/check/stage2_distill_900_v3_2500_mixed_env64_seeds0to8_20261109/verification.json), [512환경 검증](output_etc/check/stage2_distill_900_v3_2000_mixed_env512_seed9_20261109/verification.json).

## 2026-11-09

### V3 2k ·512환경 seed7 /64환경 seed9–11 추가 평가

- 사용자 요청으로 같은 V3 2k·4명 Mixed·8토큰·900step·XY100%의512환경 seed7과64환경 seed9/10/11을 GPU6/7에서 병렬 평가했다.4개 worker와 집계 모두 exit0이며 각 초기 상태/전체 관측 exact·901회 물리 분리·attention 유지·CPU SR/건수 exact·Error 일치·그룹 수 커버리지·checkpoint/기존 결과 SHA 불변을 확인했다. 메인 코드·기존 학습·공통 MD seed는 변경하지 않았다.
- Independent/Enabling/Coupled/Mixed는512환경 seed7에서97.26/72.16/89.81/86.41%,64환경 seed9는98.57/74.42/94.59/89.19%, seed10은98.44/58.82/88.24/81.83%, seed11은98.41/66.67/87.88/84.32%다. Independent는 Mixed 안 solo다. [추가4개·현재512환경 결과](output_etc/check/stage2_distill_900_v3_2000_mixed_env64_seeds91011_20261109/all_results.md), [512환경 검증](output_etc/check/stage2_distill_900_v3_2000_mixed_env512_seed7_20261109/verification.json), [64환경 검증](output_etc/check/stage2_distill_900_v3_2000_mixed_env64_seeds91011_20261109/verification.json).

### V3 2k ·512환경 Mixed seed1/6/8 평가

- 사용자 지정 seed1/6/8로 같은 V3 2k를4명·8토큰·512환경·900step·XY100%에서 평가했다. GPU6(seed1/6)·GPU7(seed8) 세 worker 모두 exit0이며512환경 초기 reference를 새로 생성했다. 기존 학습·64환경 결과와 공통 MD의 seed 기본값은 유지했다.
- Independent/Enabling/Coupled/Mixed는97.44±0.99/73.35±3.09/90.46±0.32/87.09±0.78%다. Independent는 Mixed 안 solo다. 초기 상태/전체 관측 exact·각901회 물리 분리·attention 유지·CPU SR/건수 exact·Error 일치·그룹 수 커버리지·checkpoint/기존64환경 산출물 SHA 불변을 확인했다. 메인 코드는 변경하지 않았다. [seed별 결과·64환경 비교](output_etc/check/stage2_distill_900_v3_2000_mixed_env512_seeds168_20261109/report.md), [검증](output_etc/check/stage2_distill_900_v3_2000_mixed_env512_seeds168_20261109/verification.json).

### V3 2k ·64환경 Mixed seed0–8 평가

- 사용자 요청으로 같은 V3 run의 `00002000.pth`(epoch2000/frame262144000·가중치 finite)를4명·8토큰·64환경·900step·XY100%에서9seed 평가했다. GPU6/7 각 최대3개 worker를 사용했고6개 worker 모두 exit0다.1.5k의 seed별 초기 reference를 재사용했으며 기존 학습·원본 결과는 보존했다.
- Independent/Enabling/Coupled/Mixed는97.61±2.42/70.04±9.79/91.05±4.07/86.24±3.46%다.1.5k 대비 평균 차이는-0.32/-0.62/+1.54/+0.20pp이며 Independent는 Mixed 안 solo다.9회 초기 상태/전체 관측 exact·각901회 물리 분리·attention 유지·CPU SR/건수 exact·Error 일치,1.5k와 초기 상태 모든 필드 exact·그룹 수 커버리지·두 checkpoint/기존 산출물 SHA 불변을 확인했다. 메인 코드는 변경하지 않았다. [seed별·checkpoint별 비교](output_etc/check/stage2_distill_900_v3_2000_mixed_env64_seeds0to8_20261109/report.md), [검증](output_etc/check/stage2_distill_900_v3_2000_mixed_env64_seeds0to8_20261109/verification.json).

### V3 1.5k ·64환경 Mixed seed3–8 추가 확인

- 같은 V3 1.5k·4명/8토큰·64환경·900step·XY100%에서 seed3/4/5는 GPU6, seed6/7/8은 GPU7로 추가 평가했다. 여섯 worker exit0, 각 초기 상태/관측 exact·901회 물리 분리·attention 유지·CPU SR/건수 exact·Error 일치를 확인했다. 기존 학습과 seed0/1/2 결과는 보존했다.
- 추가6seed Independent/Enabling/Coupled/Mixed는98.06±1.39/68.11±8.97/89.40±5.03/85.19±3.95%, 기존0–2를 합친 전체9seed는97.93±1.20/70.66±8.19/89.51±4.42/86.03±3.54%다. Independent는 Mixed 안 solo다. 기존 산출물/checkpoint SHA 불변·기존3seed 집계 exact·그룹 수 커버리지를 확인했다. [seed별·전체 결과](output_etc/check/stage2_distill_900_v3_1500_mixed_env64_seeds345678_20261109/report.md), [검증](output_etc/check/stage2_distill_900_v3_1500_mixed_env64_seeds345678_20261109/verification.json).

### V3 1.5k ·64환경 Mixed seed0/1/2 확인

- 사용자 지정 V3 `00001500.pth`(epoch1500/frame196608000·가중치 finite)을4명·8토큰·64환경·900step·XY100%의 현재 Mixed 프로토콜로 평가했다. GPU6(seed0/1)·GPU7(seed2) 두 worker exit0이며 기존 학습은 유지했다. 전용 V3 학습 YAML과 새64환경 초기 snapshot을 사용했다.
- Independent/Enabling/Coupled/Mixed Interaction Group 지표는97.67±0.88/75.76±2.88/89.72±3.85/87.72±2.18%다. Independent는 Mixed 안 solo다. 세 실행 초기 상태/전체 관측 exact·각901회 물리/관측 분리·attention 유지·CPU/GPU SR/건수 exact·Error 일치·checkpoint SHA 불변을 확인했다. 메인 평가 코드는 변경하지 않았다. [seed별 SR](output_etc/check/stage2_distill_900_v3_1500_mixed_env64_20261109/report.md), [검증](output_etc/check/stage2_distill_900_v3_1500_mixed_env64_20261109/verification.json).

### Stage2 Distill900 V3 ·enable Total27k

- 사용자 요청으로 V2 환경/학습 YAML을 복사해 `stage2_distill_900_v3`를 추가했다. Enable teacher만 같은 Total run의 `Stage2BeforeTotal_00027000.pth`로 바꾸고 실험·저장 이름을 분리했다. Coupled17k·Stage1 26k 초기화·solo의 enable 담당·900step·나머지 설정은 유지했다.
- train/test/VNC/eval wrapper, 실험·900step·distillation 검사·공용 eval 등록과 config/USAGE/structure를 갱신했다. V2와의 YAML 차이가 이름·enable 경로뿐임, 두 teacher/전이 파일 존재, 학습/평가 계약·wrapper 동일성·Python/shell 구문·eval CLI 등록·diff 검사를 확인했다. Teacher 가중치 로드·GPU 실행·학습은 하지 않았다. [정적 확인](output_etc/check/stage2_distill_900_v3_config_20261109/verification.json).

### SCT + FSM Table1/2 평가 완료

- GPU6/7에서512환경·900step·임시 seed1/3/6의14설정42회를 모두 완료했다. 42회 초기 상태/관측 exact·원본 Stage1 정책 불변·distractor901회·CPU SR/Error·FSM replay·고정 root 변위0과 Table2 18회 CPU 재집계를 확인했다. 14개 manifest/최종 summary 완료·seed 평균/표본표준편차 exact·두 checkpoint SHA 불변을 확인했고 평가 프로세스는 모두 종료됐다.
- Table1 Independent2/4/8/16H는90.62±0.68/89.68±1.16/87.40±0.51/70.84±0.68%, Mixed는71.58±0.51/68.15±0.96/64.26±0.66/52.54±0.57%다. Table2 Enabling/Coupled 평균 Group SR은39.37±1.07/75.61±1.23%, 성공 Enabling Time은14.76±0.20s다. t_coop는 미확정이므로20cm 시작·6/8/12cm 접촉 참고값과 원시 궤적으로 보존했다. [최종 결과](output_etc/check/sct_fsm_eval_v2_20261109/report.md), [완료 검증](output_etc/check/sct_fsm_eval_v2_20261109/completion_verification.json).

### SCT + FSM 평가 GPU당3개 병렬 실행

- 추가 병렬화 요청으로 단일 seed worker·seed별 잠금·전체 canonical seed 완료 시에만 설정 완료 집계를 추가했다. 대기 중 Enable CLIMB seed3/6·Coupled4H seed3/6의4개 실행을 추가해 남은 seed를 모두 동시에 처리한다. 기존 자식 rollout은 보존하고 해당 큐 부모만 종료/교체했으며 마지막 Independent2H seed6 큐는 그대로 재개했다. 구문/diff 검사와 기존 완료 rollout의 seed worker 검증을 확인했다. 37/42회 검증 완료 시점에 남은5개 모두 실제 실행 중이며 메모리는 두 GPU 모두 여유가 있었다.
- 사용자 요청으로 GPU6/7의 평가를 각3개, 총6개 worker로 늘렸다. `--worker a/b` 상태 분리·설정별 실행 잠금·manifest 원자적 저장·집계 잠금을 추가했다. 기존 계산 중인 자식 실행은 종료까지 유지한 뒤 자체 큐 부모만 교체했고 완료된 rollout/검증은 재사용했다. 다른 학습 프로세스는 건드리지 않았다.
- 구문/diff 검사를 통과했고 6개 worker의 서로 다른 설정 실행과 GPU 메모리 여유를 확인했다. 42회 중9회 검증 완료 시점에 나머지를 재배분했으며 실행/seed/환경 조건은 유지했다. [분배 계획](output_etc/check/sct_fsm_eval_v2_20261109/parallel_plan.json). 실행 명령은 `FSM/README.md`에 갱신했다.
- 29/42회 검증 완료 후 GPU7 main 큐가 끝난 자리에 `--worker c --jobs coupled_4`를 추가해 대기 중인4H PUSH를 앞당겼다. 기존 worker a의 같은 설정은 파일 잠금과 완료 결과 재사용으로 중복 시뮬레이션을 방지한다. [추가 worker](output_etc/check/sct_fsm_eval_v2_20261109/worker_gpu7_c.json).

### SCT + FSM 누적 도착 해제·Table1/2 평가 연결

- 사용자 확정으로 Coupled를 양손 표면12cm 도착 이력 누적으로 바꿨다. 마지막 도착자는 freeze하지 않고 전원 일회 해제하며, Mixed의 접촉 기준은 각 참여자의 배정 물리 상자를 사용한다. SR 기준은 유지한다. `FSM/` 안에서 기존 샘플러·Stage1 입력·메인 SR/CPU 재채점·Table2 기록을 재사용하는 실행/검증/집계를 추가했다. 프로토콜의 FSM 설명과 실행 안내를 갱신했으며 t_coop 정의 절은 보존했다.
- 첫 본평가에서 초기 freeze의 indexed setter가 대기 중 전체 reset을 덮어써 비고정 사람이 기본 위치로 돌아가는 문제를 발견했다. 처음 freeze 뒤 전체 root/DOF reset을 재제출하고 첫 물리 프레임 변위 검사도 추가했다. 기존 `sct_fsm_eval_20261109`는 자체 worker만 종료하고 전부 제외 표시했다. 앞선 viewer/소규모 확인도 초기 freeze가 있는 rollout의 성능 근거로 사용하지 않는다. 초기 smoke의 그룹 간격 검사 역시 실제 관계 그룹 기준으로 수정했다.
- CPU7 passed·구문/diff 통과. 수정 후 GPU6 Mixed2H64환경40step·GPU7 Enable SIT2환경220step/Mixed8H2환경220step에서 초기 장면/관측 exact·첫 프레임 위치 유지·원본 정책/정규화 exact·가중치 불변·coordination0·CPU SR/Error·FSM replay·고정 root 변위0을 확인했다. 이전 잘못된 초기화가 새 검사에서 거부되는 것도 확인했다. Coupled3H의 누적 도착 해제도 짧은 실행에서 확인했다. [검증](output_etc/check/fsm_reset_smoke_20261109/verification.json).
- GPU6/7에서512환경·900step·임시 seed1/3/6의 Table1 Independent/Mixed2/4/8/16H와 Table2 Enable3종/Coupled2/3/4H, 총42회 큐를 새 폴더에서 다시 시작했다. 아직 본평가 완료 결과는 아니다. 매 실행은 초기 상태·관측/물리 분리·원본 정책·CPU SR/Error/FSM을 검증하고 접촉 원시 궤적을 보존한다. [실행 상태](output_etc/check/sct_fsm_eval_v2_20261109/worker_gpu6.json), [GPU7](output_etc/check/sct_fsm_eval_v2_20261109/worker_gpu7.json).
- 첫 Mixed2H/Coupled2H seed1의512환경900step은 초기 상태 exact·첫 프레임 위치 유지·distractor901회·CPU SR/Error/FSM·고정 root 변위0 검증을 완료했고 각 worker는 seed3을 실행 중이다. CPU 검증은 압축 FSM trace를 한 번 읽도록 바꾸고 자체 worker만 재시작했다. 완료 rollout은 보존·재사용하고 중단한 seed3 준비 산출물은 `interrupted_before_verifier_reload/`에 보관했다. 전체 완료 시 `report.md`/`summary.json`을 자동 생성한다.

### Freeze 충돌 확인 viewer

- `FSM/`에 `--collision-probe`를 추가했다. 같은 초기 자세·PD 목표에서 freeze 충돌/해제 충돌/해제 비충돌을 비교하며 기존 FSM·SR 경로는 유지한다. 실행 명령은 `FSM/README.md`에 기록했다.
- GPU6/7 headless·실제 viewer에서 사람 고정 오차0과 공중 충돌 시 상자 X 속도2.998→1.717m/s를 확인했다. 정상 FSM30step의 원본 정책/정규화 exact·가중치 불변·coordination0, CPU6 passed·구문/shell 검사를 통과했다. 접촉력 최댓값에는 바닥 접촉도 포함되며 진짜 kinematic actor는 아니다. [비교·화면](output_etc/check/fsm_collision_20261109/report.md), [검증](output_etc/check/fsm_collision_20261109/verification.json).

### SCT + FSM state freeze viewer

- 사용자 승인으로 `FSM/`에 원본 SCT26k + Enable 후행 대기/선행 AT 성공 시 일회 해제·Coupled 양손 표면12cm 도착자 대기/전원 같은 프레임 접촉 시 일괄 일회 해제를 구현했다. 대기자는 파란색이며 root/DOF·속도·관측 자세를 물리 서브스텝마다 고정한다. 진짜 kinematic actor 전환은 아니고 충돌체·반력은 유지하며 상자 상태는 덮어쓰지 않는다. 코드/실행 명령은 `FSM/README.md`에 두고 메인 학습·eval 코드는 변경하지 않았다.
- CPU6 passed·구문/shell/diff 검사를 통과했다. GPU6/7에서 Enable3종·Coupled2/3/4명 각2환경·900step과 SIT/C2의2 episode 재초기화, 실제 Enable/C2 VNC900step·마지막 코드C2 220step을 확인했다. 11개 episode의 CPU SR exact/Error 일치·root/DOF 고정 오차0·원본165개 SCT 가중치/정규화/행동 exact·coordination0·checkpoint SHA 불변을 확인했다. 추가 상태 trace가 있는6개 episode는 CPU FSM replay·공통 trace의 정지 구간 root 변위0도 확인했다. 두 GPU worker·두 VNC·최종 확인·CPU 검증 exit0이며 확인 프로세스는 종료됐다.
- 일부 Coupled 장면에서는 상자가 먼저 움직여 기다리던 손에서 멀어진 뒤 전원 동시 접촉을 만족하지 못해 계속 대기했다. 합의한 barrier는 완화하지 않고 이 한계를 안내했다. config/structure/프로토콜에 FSM 경로·규칙을 반영했으며 t_coop3.4절은 보존했다. [검증](output_etc/check/fsm_checks_20261109/verification.json), [Enable 해제 화면](output_etc/check/fsm_checks_20261109/gui_enable_sit/episode_000/0182_transition.png), [Coupled 해제 화면](output_etc/check/fsm_checks_20261109/gui_coupled_2/episode_000/0156_transition.png).

### 평가 프로토콜 문서와 현재 SR 평가 동기화

- 사용자 요청으로 `cvpr2027_evaluation_protocol.md`에 공통 SR/Region Error의XY100%·모든 인원수의 순차 Mixed 샘플러·Stage1 가상 받침/개인 PUSH 입력·원본 정책 고정·distractor attention 유지를 반영했다. 일반 메인 eval의XY80% 기본값과 보조 경로의 차이, 기존80% Error와의 구분도 명시했다. seed는 확정하지 않았다.
- 문서만 수정했다. t_coop3.4절 원문 완전 보존·SR/Error 수식 일관성·Mixed 미정 항목 제거·참조 스크립트 경로·diff 검사를 확인했으며 시뮬레이션은 실행하지 않았다. [검증](output_etc/check/protocol_md_sync_20261109/verification.json).

### t_coop 속도 시작의 손 접근 조건 제거 비교

- `eval/rescore_table2_speed.py`에 `--no-start-hand`를 추가해 시작을 상자 XY 속도·연속 유지 시간만으로 정할 수 있게 했다. 공동 접촉의 전원 양손12cm·낙상 제외는 유지하며 기존 손 접근 옵션/기본 프로토콜은 보존했다.
- CPU25 passed·구문/diff 검사 통과. 두 모델·2/3/4명·seed1/3/6 저장 궤적18개에서0.2~1.0m/s(0.1간격)·1초 유지9조합을 CPU 재집계했다. 최초 성공/기존12cm 접촉 프레임 수 exact·원본 hash 불변·trial별 비율 검증과 손12cm 대비 시작 집합 확대/시작 시점 비지연을 확인했다. 82,944개 trial×속도 쌍 중 시작 변경181개·추가 시작78개로 평균 접촉 비율 차이는 작았다. [비교표](output_etc/check/table2_speed_02_10_1s_nohand_20261109/comparison.md), [검증](output_etc/check/table2_speed_02_10_1s_nohand_20261109/verification.json).

### t_coop 속도·지속 시간9조합 재집계

- `eval/rescore_table2_speed.py`로 XY 속도0.05/0.1/0.2m/s × 지속0.1/0.3/0.5초의 CPU 비교를 추가했다. 속도는 프레임 간 실제 상자 중심 XY 변위/control dt이며 초기 프레임은 초기 중심을 사용한다. 연속 구간 내 매 프레임 유효 참여자의 한 손 이상이 표면16cm 미만이어야 하고, 최초 성공 전까지 확인된 첫 연속 구간의 첫 프레임으로 시작을 소급한다. 변위 조건을 대체하며 접촉은 전원 양손12cm·실패 포함·미시작0%다. 기본 프로토콜은 유지했다.
- CPU21 passed·구문/diff 검사 통과. 저장 궤적18개·각9조합을 GPU 없이 완료했고 최초 성공 프레임/기존12cm 접촉 프레임 수 exact·원본 결과 hash 불변·저장 trial별 분모/분자 재산출 일치를 확인했다. 0.1m/s·0.3초의 Stage1/Stage2 평균은28.84±0.83/43.39±0.25%, 0.2m/s·0.5초는34.64±0.75/50.79±0.26%다. 인원/seed별 결과와 시작률·프레임 수를 보존했다. [비교표](output_etc/check/table2_speed_start_20261109/report.md), [검증](output_etc/check/table2_speed_start_20261109/verification.json).

### t_coop 손 접근 시작 조건 비교

- `eval/rescore_table2.py`에 선택적 `--start-hand-distance-cm`을 추가했다. 시작 변위와 같은 프레임에 유효 참여자의 한 손 이상이 실제 상자 표면 문턱 미만이어야 시작하며 시작 상태는 유지한다. 기본값/공통 프로토콜은 변경하지 않았다.
- CPU16 passed·기존20cm GPU 기록 프레임 수 exact 재현 후 저장 궤적18개를 시작20cm+손16cm·전원 양손 접촉6/8/12cm로 CPU 재집계했다. 최초 성공 프레임 exact·SR/완료 시간 불변·원본 결과 hash 불변을 확인했다. 접촉12cm 평균은 Stage1 33.97±1.01%, Stage2 50.77±0.13%로 손 접근 시작 조건 추가 영향은 매우 작았다. [비교·seed별 결과](output_etc/check/table2_handstart16_20261109/report.md).

### t_coop 시작 변위40cm 재집계

- 사용자 요청으로 `eval/rescore_table2.py`에 `--start-displacement-cm`을 추가했다. 기본20cm·메인 프로토콜은 유지하며 시작 변위 변경 시 최초 성공 프레임/SR을 검증하고 측정 구간만 재계산한다.
- CPU13 passed·기존20cm 프레임 수 exact 재현 후 저장 궤적18개(Stage1/Stage2·2/3/4명·seed1/3/6)를40cm 시작·접촉6/8/12cm로 재집계했다. GPU 재실행 없이 SR/완료 시간 불변·원본 결과 hash 불변을 확인했다. 인원별 동일 가중 평균은 Stage1 29.51/35.22/40.41%, Stage2 46.28/54.55/60.41%다. [비교표·seed별 결과](output_etc/check/table2_start40_20261109/report.md).

### Table 2 전용 평가 구현·36회 완료

- 사용자 승인으로 `eval/`에 단일 Enable 쌍3종·공동 PUSH2/3/4명 전용 실행·3명 환경 검사·초기 상자 대표 쓰기·Stage1 native 정책 검증을 추가했다. 메인 formal eval의 장면 상태·낙상·SR을 재사용하며 학습 코드는 변경하지 않았다.
- Enabling Time은 최초 동시 성공·성공 trial만 평균한다. Coupled t_coop는 초기 수평 변위20cm부터 최초 성공(실패는 종료)까지 전원 양손의 실제 공유 상자 표면3D 거리6cm 미만·힘 조건 없음·30Hz·양끝 포함이다. 원시 손/상자 좌표를 끝까지 저장하고8/12cm 재집계·실제 압축 용량/시간·CPU/GPU exact 검증을 추가했다. Table 2 Error 열은 제외하고 메인 Error는 유지했다.
- CPU68 passed·구문 검사 통과. GPU6의 Stage1 3명 PUSH와 GPU7 Enable3종을 각16환경·900step 검증한 뒤512환경×임시 seed1/3/6·두 모델36회를 GPU6/7에서 완료했다. 모든 실행의 초기 상태/관측 exact·각901회 물리/관측 분리·SR/Error·Table2 프레임 수 CPU/GPU exact, Stage1 원본 행동/정규화 exact·coordination0, 모델 간18개 초기 장면 묶음 exact와 checkpoint SHA 불변을 확인했다. 원시 손 좌표 finite·최초 성공 cutoff·문턱별 포함 관계, 공동 배치4,608장면/13,824명의 거리·rear·ordered·크기 확장도 확인했다. 두 worker·최종 집계 exit0다.
- Stage1/Stage2 Enabling Avg.는25.98±0.90/70.51±1.05%, Time은12.27±0.28/11.03±0.16s다. Coupled Avg.는73.81±0.94/91.12±0.82%, t_coop6cm는24.61±0.96/38.79±0.11%다. 추가 원시 기록은 총673.42MiB·저장 평균1.29초/실행이며6/8/12cm·seed별 값을 보존했다. 프로토콜/실행 안내/structure를 갱신했고 seed는 공통 MD에 고정하지 않았다. [Table 2 결과](output_etc/check/table2_20261109/report.md), [검증](output_etc/check/table2_20261109/verification.json), [공동 배치 검사](output_etc/check/table2_20261109/coupled_geometry_verification.json).

### Enable 숫자 제거 ·역할별 채도

- 사용자 요청으로 `DESIGN=1`의 숫자 표시 코드·호출을 제거했다. Enable은 같은 색상에서 선행을 밝고 낮은 채도, 후행을 높은 채도·조금 어두운 톤으로 구분한다. 개별 작업 상자·목표는 담당자 톤, 공유 상자는 두 역할의 중간 톤을 사용한다. Coupled·단독·기본 화면과 과제별 목표 표시는 유지한다.
- config/USAGE/structure에서 번호 안내를 역할별 톤으로 바꿨다. CPU58 passed·구문/diff 검사를 통과했고 GPU2의4명/16명·각2환경·6 preset에서 각12회 reset/48step의 초기 관측/정책 목표 exact·물리 분리·실제 색 지정·가중치 불변을 확인했다. 두 실행 exit0이며 실제 화면에서 숫자 제거·역할별 톤을 확인했다. [검증](output_etc/check/design_saturation_20261109/verification.json), [4명 화면](output_etc/check/design_saturation_20261109/gui4/before.png), [16명 화면](output_etc/check/design_saturation_20261109/gui16/before.png).

### DESIGN 과제별 목표 ·Enable 번호 표시

- `DESIGN=1`의 목표 구·공중 십자를 숨기고 PUSH AT 링/세로선, CARRY AT 바닥 사각 테두리, ON_TOP 윗면 모서리/아래 화살표, SIT 윗면 테두리/옅은 선 채움, CLIMB 윗면 테두리/위 화살표로 바꿨다. 공동 PUSH 공유 중심·정책 개인 목표·그룹 색·distractor 분리는 유지한다.
- Enable 사람·작업 상자에 선행1/후행2, 공유 상자에1·2를 표시한다. 카메라 방향·화면 높이 약28px·흰색/어두운 외곽선·겹침 완화를 적용하고 상자 번호는 목표 화살표 위로 띄웠다. 연결선/원형 배지는 추가하지 않았으며 config/USAGE/structure를 갱신했다.
- CPU59 passed·구문/diff 검사 통과. GPU7의4명/16명·각2환경에서6 preset·각12회 reset/48step의 관측/정책 목표 exact·물리 분리·색·가중치 불변을 확인하고 화면을 검사했다. SIT/CLIMB/ON_TOP 확대 검사와8명·12토큰 VNC/기본 옵션 off의 각2장면×4step도 exit0다. [검증](output_etc/check/design_markers_20261109/verification.json), [16명 실제 화면](output_etc/check/design_markers_20261109/gui16_final/before.png).

### Stage1 26k ·Full V2 2k 인원수별 비교 완료

- 임시 seed1/3/6·512환경·900step·XY100%로 Stage1의2/4/8/16H Independent/Mixed24회와 distractor attention 보존 Full8/16H12회를 GPU6/7에서 완료했다. Full2/4H12개 결과는 재사용했다. Stage1 Independent는91.02±0.49/89.01±1.19/87.65±0.89/71.13±0.68%, Mixed는64.33±1.05/62.42±1.64/57.83±0.88/47.04±0.39%다.
- 재평가한 Full8/16H Independent는95.57±0.19/80.96±0.20%, Mixed는84.03±0.41/71.02±0.33%다. 이전 distractor attention 제외 결과는 보존하고 새 결과로 구분했다. [모델별·seed별 SR과 그룹 수](output_etc/check/stage1_26k_protocol_20261108/report.md).
- 신규36회 초기 물리/관측 exact·각901회 distractor 물리 분리·CPU/GPU SR/건수 exact·Error 일치, Stage1 원본 행동 exact·165개 가중치/정규화 불변·coordination 호출0, checkpoint SHA 불변과 그룹 수 커버리지를 확인했다. 두 최종 worker·최종 비교 집계 exit0이다. 실패한 초기 기준/이름 충돌·조기 집계 로그는 최종 결과에서 제외하고 검증 파일에 명시했다. [검증](output_etc/check/stage1_26k_protocol_20261108/verification.json).

## 2026-11-08

### Viewer DESIGN=1

- 일반 test/VNC에 `DESIGN=1`(직접 실행 `--design`)을 추가했다. Distractor는 초기 관측을 보존하며 화면·충돌 영역 밖으로 옮기고, 공동 PUSH 목표는 정책의 개인 목표를 유지한 채 공유 중심 하나만 표시한다. Enable·Coupled의 사람·작업 상자·목표는 그룹 색으로 통일하고 기존4색 이후에는 중복 없는 색을 추가한다. 기본 실행·장면 샘플링·정식 평가 기준은 유지한다.
- 작업 물체·받침·alias 보존, 반복/부분 reset과 카메라 제외를 연결하고 config/USAGE/structure에 옵션·명령을 기록했다. 학습/resume·formal eval에는 허용하지 않는다.
- CPU54 passed. GPU6의4명/8명·각2환경에서6개 preset·각12회 reset/48step을 실행해 초기 전체 관측·정책 목표 exact 보존, 물리 분리·색 지정·공유 마커·가중치 불변을 확인했다. 실제 화면을 확인했고 일반 test wrapper의 옵션 on/off도 각2장면×4step exit0다. [검증](output_etc/check/design_20261108/verification.json), [4명 혼합 화면](output_etc/check/design_20261108/gui4_verified/mixed.png), [8명 색 확장](output_etc/check/design_20261108/gui8_verified/all_solo.png).

### Stage1 26k 비교 경로 ·distractor attention 수정

- 사용자 요청으로 Stage1 26k를 같은 초기 물리 장면·채점에서 평가하는 보조 진입점을 추가했다. Enable은 목표 정렬 가상 받침, Coupled는 개인 PUSH 구역 입력을 사용한다. 첫 행동 전에 원본165개 가중치·정규화를 복원하고 Stage2 coordination을 우회하며 native Stage1 행동 exact·가중치 불변·coordination 호출0을 검사한다. 초기 장면 생성에만 Full checkpoint를 사용한다.
- 기존5명 이상 Full 플레이어가 task-slot mask로 distractor attention을 제외하던 부분을 발견했다. 보조 평가에서 마스크를 해제하고 매 encoder 호출을 검사한다. 이전8·16H 결과는 보존하고 재평가한다. 뷰어 코드의 지역 import와 평가 관측 훅의 이름 충돌은 훅 이름만 분리해 수정했다.
- CPU62 passed·구문 검사를 통과했다. Stage1 2H Independent의 임시 seed1/3/6은 초기 물리/관측 exact·native 정책 exact·900step·CPU SR/Error 검증을 완료했고, 나머지 평가는11월9일 완료했다. 메인 학습·프로토콜 MD·기본 seed/XY 판정은 변경하지 않았다.

### V2 2.5k ·seed0/1/2 평가와 1.5k·2k 비교

- 사용자 요청으로 V2 `00002500.pth`(epoch2500/frame327680000·finite·SHA256 확인)를4H Mixed·8토큰·512환경·900step에서 GPU6(seed0)·GPU7(seed1/2)로 평가했다. 세 실행과 CPU 재채점은 exit0, 각901회 물리/관측 분리·SR/건수 exact·Error 일치와 세 checkpoint의 seed별 초기 상태 전체 필드 exact를 확인했다. 1.5k·2k는 재실행하지 않았고 원본 결과/summary hash를 보존했다.
- 기존 XY80%의 Independent/Enabling/Coupled/Mixed는2.5k에서96.99±0.46/64.46±0.74/91.63±0.37/84.36±0.48%다. 동일 궤적의 XY100%는97.53±0.37/66.77±0.46/91.63±0.37/85.31±0.35%이며 Mixed1.5k/2k/2.5k는85.63±1.18/85.52±0.57/85.31±0.35%다. XY100%의1.5k만 저장 궤적으로 추가 계산하고2k는 기존 캐시를 재사용했다. Error는 원래 XY80% 값을 유지했다. [세 checkpoint·seed별 비교](output_etc/check/stage2_distill_900_v2_2500_new_protocol_env512_20261108/checkpoint_comparison.md), [검증](output_etc/check/stage2_distill_900_v2_2500_new_protocol_env512_20261108/verification.json).
- 일반 VNC는 학습 샘플러·보상 기반 진단을 사용하며 논문 샘플링/물리 제거·formal SR/낙상 처리와 다름을 README/config에 명시했다. 실제 viewer는 실행하지 않았다. XY 비교 보조 보고서의 고정2k 문구만 checkpoint 공통 표현으로 바꿨으며 메인/학습/기본 seed·프로토콜 MD는 변경하지 않았다.

### V2 2k ·Independent/Mixed 2·4·8·16H 평가

- 임시 seed1/3/6·XY100%·512환경·900step의 Independent 전용/Mixed 분리 평가21회를 GPU6/7에서 완료하고 기존4H Mixed SR을 재사용했다. Independent2/4/8/16H는97.30±0.44/97.43±0.20/95.35±0.21/79.49±0.19%, Mixed는88.40±0.77/86.25±0.28/82.85±0.31/71.16±0.62%다. [관계별·seed별 결과와 그룹 수](output_etc/check/stage2_scalability_20261108/report.md).
- 보조 경로에2H 검사·균등 단독 행동·5명 이상 그룹별2k 잔여 토큰 distractor 배분/관측 보존을 추가했다. 16H 초기 장면 불일치는 같은 공동 상자에 참가자별 초기 상태를 중복 대입하던 GPU 쓰기에서 발생해, 5명 이상 초기 상자 쓰기를 그룹 대표로 제한했다. 최종 coupled 거리·크기·ordered 함수와 native 모션 추출은 유지했다. XY100%는 채점/Error에만 적용하며 프로토콜 MD·기본 seed·메인 채점80%·학습 설정은 변경하지 않았다.
- CPU42 passed·그룹 예산12개 구성·XY100% 경계/Error/보상 기하 불변을 확인했다. 본평가21회 모두 초기 작업 상태/전체 관측 exact·각901회 물리 분리·CPU/GPU SR/건수 exact·Error 일치를 검증했다. Coupled18,299명 초기 배치의 거리·rear/+Y 목표·ordered·native profile·최대4명도 확인했고 checkpoint/재사용 원본은 불변이다. 두 worker와 집계 exit0, 실패한 초기화 진단은 별도 보존했다. [검증](output_etc/check/stage2_scalability_20261108/verification.json).

### V2 2k ·XY100% 전체 seed0–8 정리

- 사용자 요청으로 기존 seed2/6/7/8 XY100% 결과를 재사용하고 미계산 seed0/1/3/4/5만 저장 궤적으로 CPU 재채점했다. 기존80% SR·최종 성공 flag exact 및100% 성공 포함 관계를 확인했고, 캐시 중복6/8·기존80% 전체9seed 집계·Point 불변·원본 파일 hash 불변을 검증했다. 보조 보고서의 seed 개수 표시만 실제 개수로 맞췄으며 GPU·시뮬레이션·메인 기준/프로토콜/Error 변경은 없었다.
- XY100% 전체9seed Independent/Enabling/Coupled/Mixed는97.51±0.42/69.00±1.15/91.13±1.01/85.88±0.53%다. XY80% 대비 Mixed +0.87pp다. [seed별 표·80% 비교](output_etc/check/stage2_v2_2000_xy100_allseeds_20261108/report.md), [검증](output_etc/check/stage2_v2_2000_xy100_allseeds_20261108/verification.json).

### V2 2k seed2/6/8 ·XY80%·100% 재집계

- 사용자 요청으로 저장된 V2 2k·512환경·900step 궤적의 seed2/6/8을 CPU 재집계했다. 기존80% SR·ever/simultaneous exact,100% 성공 포함 관계·Point 불변·기존6/8 비교값 exact·원본 결과 hash 불변을 확인했다. GPU·시뮬레이션·메인 코드/기준 변경은 없었다.
- Region Independent/Enabling/Coupled/Mixed는80%에서97.20±0.24/67.99±1.57/90.19±1.22/85.13±0.59%,100%에서97.56±0.26/69.47±1.67/90.19±1.22/85.74±0.73%다. [비교표](output_etc/check/stage2_v2_2000_xy100_seeds268_20261108/report.md), [검증](output_etc/check/stage2_v2_2000_xy100_seeds268_20261108/verification.json).

### V2 2k seed6/7/8 ·XY80%·100% 성공률 비교

- 사용자 질문에 따라 같은 V2 2k·512환경×seed6/7/8·900step 저장 궤적을 CPU 재채점했다. SIT/CLIMB/ON_TOP의 XY만 축별±0.4L→±0.5L로 넓히고 높이±20cm·AT/PUSH·낙상·Enable 동시 성공은 유지했다. `eval/compare_region_extent.py`에서 기존80% SR·최종 ever/simultaneous exact, Point 불변·성공 포함 관계·원본 결과 불변을 확인했다. GPU·시뮬레이션은 실행하지 않았고 메인 eval·프로토콜·Error 기준은80%로 보존했다.
- Region Independent/Enabling/Coupled/Mixed는80%의97.51±0.67/67.95±1.61/90.90±0.98/85.45±0.19%에서100%의97.93±0.46/69.91±1.27/90.90±0.98/86.25±0.16%로 변했다. Mixed +0.79pp이며 동일 정책의 판정 범위 변경 결과다. 보조 코드 안내·structure를 갱신했다. [비교표](output_etc/check/stage2_v2_2000_xy100_seeds678_20261108/report.md).

### Distill900 V2 2k ·seed3–8 GPU6/7 추가 평가

- 사용자 지정대로 같은 V2 2k·4명/8토큰·512환경·900step·ordered·새 프로토콜의 seed3/4/5는 GPU6, seed6/7/8은 GPU7에서 평가했다. 이전1.5k seed별 기준 snapshot을 재사용했고 GPU7도 초기 전체 상태·관측 exact를 확인했다. 6개 실행과 CPU 재채점 모두 exit0, 매 실행901회 물리/관측 분리·SR/건수 exact·Error 허용오차 이내 일치를 확인했다. 코드·기존 학습·원본 결과는 변경하지 않았다.
- Region Independent/Enabling/Coupled/Mixed는 seed3/4/5에서97.06±0.46/66.09±0.58/91.78±0.64/84.98±0.54%, seed6/7/8에서97.51±0.67/67.95±1.61/90.90±0.98/85.45±0.19%다. 기존2k seed0/1/2를 합친 전체9seed는97.17±0.51/66.73±1.31/91.13±1.01/85.01±0.47%이며 1.5k 전체9seed 대비+0.52/+3.10/-1.37/+0.75pp다. Independent는 Mixed 내 solo다.
- [전체9seed SR·1.5k 비교·Error](output_etc/check/stage2_distill_900_v2_2000_new_protocol_seeds678_20261108/combined_report.md), [GPU6 seed3–5](output_etc/check/stage2_distill_900_v2_2000_new_protocol_seeds345_20261108/report.md), [GPU7 seed6–8](output_etc/check/stage2_distill_900_v2_2000_new_protocol_seeds678_20261108/report.md).

### Distill900 V2 2k ·seed0/1/2 새 프로토콜 평가

- 사용자 요청으로 같은 run의 `Stage2Distill900V2_00002000.pth`(epoch2000/frame262144000·finite·SHA256 확인)를 GPU6·4명/8토큰·512환경×seed0/1/2·900step·ordered·새 프로토콜로 평가했다. 기존0/1/2 기준 snapshot을 재사용했으며 V2 1.5k와 초기 전체 상태 필드·관측 exact를 확인했다. 매 실행901회 물리/관측 분리와 CPU/GPU SR·건수 exact·Error 허용오차 이내 일치, 실행·재채점 exit0를 확인했다. 코드·기존 결과·학습 프로세스는 보존했다.
- Region Independent/Enabling/Coupled/Mixed는 96.93±0.33/66.14±0.73/90.71±1.30/84.60±0.12%다. 동일 seed의 V2 1.5k 대비 각각 +0.42/+0.43/-1.95/-0.37pp다. Independent는 Mixed 내 solo다. [seed별 SR·Error·1.5k 비교](output_etc/check/stage2_distill_900_v2_2000_new_protocol_env512_20261108/report.md), [검증](output_etc/check/stage2_distill_900_v2_2000_new_protocol_env512_20261108/verification.json), [초기 상태 비교](output_etc/check/stage2_distill_900_v2_2000_new_protocol_env512_20261108/initial_comparison.json).

### Distill900 V2 1.5k ·추가 seed6/7/8

- 사용자 요청으로 같은 V2 1.5k·GPU6·4명/8토큰·512환경·900step·ordered·새 프로토콜을 seed6/7/8에 추가 평가했다. 각 seed의 초기-only 기준 snapshot과 실제 평가의 과제/작업 상태/전체 관측 exact, 매 실행901회 관측/물리 분리, CPU/GPU SR·건수 exact·Error 허용오차 이내 일치를 확인했다. 세 실행·재채점 exit0이며 checkpoint·실행 소스 hash는 이전 평가와 같다. 코드·기존 결과·학습 프로세스는 변경하지 않았다.
- Region Independent/Enabling/Coupled/Mixed는 96.55±0.58/62.15±1.27/92.89±1.15/83.86±0.48%다. 기존0–5 Mixed84.46±1.07% 대비 -0.60pp이고 전체0–8은 84.26±0.93%다. Independent는 Mixed 내 solo이며 최종 논문 seed 선택은 별도다. [seed별 SR·Error·9seed 집계](output_etc/check/stage2_distill_900_v2_1500_new_protocol_seeds678_20261108/report.md), [검증](output_etc/check/stage2_distill_900_v2_1500_new_protocol_seeds678_20261108/verification.json).

### Distill900 V2 1.5k ·추가 seed3/4/5

- 사용자 요청으로 같은 V2 1.5k·GPU6·4명/8토큰·512환경·900step·ordered·새 프로토콜을 seed3/4/5에 추가 평가했다. 각 seed의 초기-only 기준 snapshot을 먼저 생성하고 공통 교집합·near50 해제·균등 sampler·distractor 관측 유지/물리 제거·동시 성공 SR·두 Error를 유지했다. 세 실행과 재채점은 exit0, 초기 과제/작업 물체/관측 exact·매 실행901회 물리 분리·CPU/GPU SR/건수 exact·Error 허용오차 이내 일치를 확인했다.
- Region Independent/Enabling/Coupled/Mixed는 96.89±0.43/63.03±0.30/91.95±1.94/83.96±0.58%다. 기존0/1/2의 Mixed84.96±1.32% 대비 -1.00pp이며 0–5 전체는 84.46±1.07%다. 기존 결과와 학습 프로세스는 보존했고 최종 논문 seed 선택은 확정하지 않았다. [seed별 SR·Error·6seed 집계](output_etc/check/stage2_distill_900_v2_1500_new_protocol_seeds345_20261108/report.md), [검증](output_etc/check/stage2_distill_900_v2_1500_new_protocol_seeds345_20261108/verification.json).
- Formal eval의0/1/2 제한을 비음수 seed로 완화하고 과제 macro도 실제 실행 seed로 계산한다. 보조 진입점은 기준 snapshot 생성·경로 지정과 seed별 보고를 지원한다. 물리 hook은 검증된 기존 구현에서 reference 경로만 분리해 `eval/stage2_distractor_physics.py`에 뒀다. CPU37 passed, hook의 경로 외 동일성과 실행 소스 hash를 확인하고 config/평가 규약/structure/eval 안내를 갱신했다.

### Distill900 V2 1.5k 새 프로토콜 평가

- Enable Total19.5k·Coupled17k teacher의 V2 student `Stage2Distill900V2_08-19-13-51/nn/Stage2Distill900V2_00001500.pth`(epoch1500/frame196608000·finite·SHA256 확인)를 GPU6·4명·8토큰·512환경×seed0/1/2·900step·ordered로 평가했다. 공통 교집합·near50 해제·균등 순차 I/E/C/행동·distractor 관측 유지/물리 제거와 동시 성공 SR·공통 Error를 적용했다.
- 이전 V1 평가와 세 seed 초기 상태 전체 필드·관측 exact, 매 실행901회 관측/물리 분리, CPU/GPU SR·건수 exact·Error 허용오차 이내 일치를 확인했다. 3개 실행·재채점은 정상 완료했고 기존 학습·원본 결과는 보존했다. Region Independent/Enabling/Coupled/Mixed는 96.52±0.61/65.71±3.08/92.66±2.10/84.96±1.32%다. Independent는 Mixed 내 solo이며 이전 동시 기준 Mixed84.89±1.19% 대비 +0.07pp다.
- 사용자 지정 `eval/`에 기존 검증 sampler/hook·초기 snapshot을 재사용하는 실행 진입점과 검증/요약 보조 코드를 두고 README·structure를 갱신했다. 메인 환경·학습 코드는 변경하지 않았다. [seed별 SR·Error·V1 비교](output_etc/check/stage2_distill_900_v2_1500_new_protocol_env512_20261108/report.md), [재채점 검증](output_etc/check/stage2_distill_900_v2_1500_new_protocol_env512_20261108/verification.json), [초기 상태 비교](output_etc/check/stage2_distill_900_v2_1500_new_protocol_env512_20261108/initial_comparison.json).

### 공통 Point/Region Error ·Table 2 지표 정의

- 사용자 요청으로 메인 eval·seed 요약·CPU 재채점에 Region 성공 trial의 Point/Region 최소 3D 거리(cm)를 추가했다. Enable은 동시 성공한 쌍의 선행·후행만, Coupled는 성공한 실제 공유 상자를 그룹당 한 번 집계한다. 목표 기하는 기존 SR과 공유하며 Region Error는 높이 허용오차 없는 목표 평면까지 잰다. 행동별·공동 인원별 평균/성공 표본 수와 seed 통계, N/A·유효 frame 규칙을 기록한다.
- `errors.protocol=region_success_min_distance_v1`과 manifest를 연결하고 과거 Error 없는 결과의 새 평가 재사용·혼합 요약을 막았다. 논문 프로토콜·평가 규약·config/structure·eval 안내에 반영했다. Time은 성공 trial 최초 동시 완료, t_coop는 실패 포함 밀기 구간의 전원 양손 접촉 비율로 문서화했으며 두 지표의 코드는 추후 `eval/` Table 2용으로 남겼다.
- CPU32 passed. GPU6·16환경·900step 메인 eval의 SR/건수 exact·Error CPU/GPU 0.000001cm 이내·저장 최솟값 일치를 확인했다. 기존 512환경×3 seeds의 900step 저장 trace에서도 SR·통계 exact를 보존하며 Error를 산출했다. 원본 결과·기존 학습 프로세스는 보존했다. [검증](output_etc/check/stage2_point_region_errors_20261108/verification.json), [행동별 Error](output_etc/check/stage2_point_region_errors_20261108/report.md).

### Enabling 그룹 SR을 동시 성공으로 변경

- 사용자 요청으로 논문 프로토콜·평가 규약·config 공통 안내와 메인 interaction group 집계를 `ever(AT AND follower)`로 변경했다. Mixed도 같은 Enable SR을 사용한다. 완료 시간 기준안은 첫 동시 성공 시각으로 맞췄으며 기존 과제별 각각-ever·동시 성공 보조 지표는 유지한다.
- 결과·manifest·summary에 `enabling_success=simultaneous`를 기록하고 기존 기준의 결과 재사용·혼합 집계를 거부한다. CPU 재채점은 필드 없는 과거 결과를 `both_ever`로 보존하며 보조 비교기는 두 버전 모두 읽는다. 장면 샘플링·학습 설정은 변경하지 않았다.
- CPU23 passed. 저장된 512환경×seed0/1/2의 900step trace를 별도 경로에서 새 기준으로 재채점해 point/region·seed 통계가 비교 결과와 exact 일치했고, 과거 seed0의 재채점 호환도 확인했다. 원본 결과는 보존했으며 GPU·시뮬레이션·학습은 실행하지 않았다. [검증](output_etc/check/stage2_enabling_simultaneous_code_20261108/verification.json).

### Enabling 각각-ever·동시 성공 재집계

- 사용자 요청으로 새 프로토콜 1.5k·512환경×seed0/1/2의 CPU 검증된 저장 trace를 `eval/compare_enabling_success.py`에서 재집계했다. Point/Region의 각각-ever·동시 성공 flag와 기존 행동별·그룹 건수 일치, 동시 성공의 부분집합 관계를 확인했다. 원본 결과·메인 eval·프로토콜 기준은 변경하지 않았으며 GPU·시뮬레이션은 실행하지 않았다.
- Region Enabling은 66.45±2.59→66.03±2.52%, Mixed는 85.03±1.19→84.89±1.19%다. 951쌍 중 4쌍이 동시 성공 조건에서 제외됐다. [비교·seed별 건수](output_etc/check/stage2_enabling_simultaneous_1500_env512_20261108/report.md).

### 5명 이상 distractor 배분 확정

- 사용자 결정으로 그룹 k명당 2k 토큰에서 필수 토큰을 뺀 나머지를 그룹 중심 반경 4.5m에 배치하는 관측용 distractor 규칙을 프로토콜에 반영했다. 물리 제거는 유지하고 평가 seed는 사용자 탐색 후 확정으로 남겼다. 문서 변경만 확인했으며 메인 eval·학습·실행 프로세스는 변경하지 않았다.

### 평가 보조 작업공간

- 사용자 지정 루트 `eval/`에 보조 스크립트 용도와 `output_etc/check/` 산출물 규칙을 기록하고 structure에 등록했다. 메인 eval·기존 확인용 코드·산출물은 이동하지 않았다. 문서 경로·diff만 확인했으며 학습·평가는 실행하지 않았다.

### Stage2 Distill900 V2 ·enable Total19.5k

- 사용자 요청으로 `stage2_distill_900_v2` 환경/학습 YAML과 train/test/VNC/eval wrapper를 기존900 버전에서 복사했다. Enable teacher만 `Stage2BeforeTotal_06-22-22-14/nn/Stage2BeforeTotal_00019500.pth`로 교체하고 실험·저장 이름을 분리했다. Coupled17k·Stage1 26k 초기화·900step·학습 샘플링/보상·distillation 설정은 유지했다. 새 이름의 실행 등록·900step/GPU2/6/7·공용 평가 연결과 config/USAGE/structure를 갱신했다.
- YAML 차이가 이름·enable 경로뿐임과 teacher 파일 존재·distillation 스키마·실험 등록/900step·Python/shell 구문·eval CLI help·문서 경로를 확인했다. Teacher 가중치 로드·GPU 실행·학습·시뮬레이션은 하지 않았다. [정적 확인](output_etc/check/stage2_distill_900_v2_config_20261108/verification.json).

### 새 프로토콜 Distill900 1.5k ·512환경×3 seeds

- GPU6에서 같은 1.5k student를 4명·8토큰·512환경×seed0/1/2·ordered·900step으로 평가했다. 순차 균등 I/E/C·단독/후행 균등 행동·공통 교집합·near50 해제/거리표·distractor 관측 유지/물리 제거를 적용했다. Seed별 초기-only snapshot과 초기 관측 전체·과제·사람·목표·DOF·작업 물체 상태 exact, 900step 관측/물리 분리·CPU/GPU 재채점·건수·finite를 확인했다.
- 영역 Independent/Enabling/Coupled/Mixed는 96.93±0.83/66.45±2.59/91.72±1.50/85.03±1.19%다. Independent는 Mixed 안 solo다. Seed0은 초기 좌표 반올림 precheck를 조정했으며 최종 초기값은 exact 일치했다. Rollout·trace 완료 후 검증 정보 출력 오타는 저장 trace/로그로 복구했고 성공수·궤적은 변경하지 않았다. 공용 eval·학습 코드·기존 프로세스는 보존했다. [seed별 결과·분포](output_etc/check/stage2_new_protocol_1500_env512_20261108/report.md), [검증](output_etc/check/stage2_new_protocol_1500_env512_20261108/verification.json).

### 논문 프로토콜 distractor 물리 제거 확정

- 사용자 결정으로 모든 방법·인원수에서 distractor 관측 유지·물리 제거를 [프로토콜](cvpr2027_evaluation_protocol.md)에 확정했다. 작업 물체·받침·alias·구역은 보존하고, 관측 pose·선속도·각속도는 초기 샘플 값으로 고정하며 형상/BPS·토큰은 유지한다. 미정 항목에서는 5명 이상 관측용 배치 영역만 남겼다. 문서 diff·미정 표의 일관성을 확인했으며 공용 eval 코드·학습·실행 프로세스는 변경하지 않았다.

### Distill900 1.5k distractor 물리 제거 비교

- GPU6에서 기존·새 프로토콜 각각 4명·8토큰·64환경×seed0/1/2·ordered·900step으로 distractor 물리 제거 6회를 완료했다. 이전 물리 유지 결과를 재사용했고, 작업 물체·받침·alias·구역은 보존했다. Distractor는 장면 생성 후 물리 장면 밖으로 이동하고, 관측은 같은 baseline 초기 snapshot의 위치·회전·속도·형상을 유지했다. 공용 eval·학습 코드와 기존 프로세스는 보존했다.
- 영역 Independent/Enabling/Coupled/Mixed는 기존 물리 제거 95.01±3.55/69.35±2.40/91.33±2.35/85.23±0.59%, 새 조건 물리 제거 98.13±2.12/69.62±7.92/92.16±2.60/86.64±1.36%다. 물리 유지 대비 Mixed 변화는 기존 -0.37pp, 새 조건 +0.90pp다. 각 프로토콜의 유지/제거 초기 과제·그룹·크기·사람·목표·DOF·작업 물체와 관측 물체 상태 exact, 900step 관측/물리 분리·CPU/GPU 재채점·건수·finite를 확인했다. 좌표 반올림 및 추가 초기화 경로 검증 중 제외한 시도는 별도 보관하고 최종 표에 포함하지 않았다. [비교표·조건](output_etc/check/stage2_distractor_physics_1500_env64_20261108/report.md), [검증](output_etc/check/stage2_distractor_physics_1500_env64_20261108/verification.json).

### Distill900 1.5k 기존·새 프로토콜 64환경 비교

- GPU6에서 같은 1.5k PTH를 4명·8토큰·64환경×seed0/1/2·ordered·900step으로 각 3회 평가했다. 기존은 행동별 교집합/native 샘플링, 새 조건은 공통 교집합·near50 해제/거리표·균등 행동·순차 I/E/C다. Distractor 관측·물리는 양쪽 유지했고 새 조건은 확인용 driver에만 적용했다. 공용 eval·학습 코드·기존 프로세스는 변경하지 않았다.
- 영역 Independent/Enabling/Coupled/Mixed는 기존 97.63±2.71/65.99±5.20/93.18±5.22/85.60±1.24%, 새 조건 97.23±0.13/69.52±1.82/90.46±2.77/85.74±0.99%다. Independent는 Mixed 내 solo이며, 같은 seed라도 초기 장면은 다르다. 6회 모두 CPU/GPU 재채점·건수·finite 일치, 새 크기·거리·near50 해제를 검증했다. [조건·비교](output_etc/check/stage2_protocol_compare_1500_env64_20261108/report.md), [검증](output_etc/check/stage2_protocol_compare_1500_env64_20261108/verification.json).

### CVPR2027 논문 평가 프로토콜 정리

- 사용자 요청으로 [평가 설계 문서](cvpr2027_evaluation_protocol.md)를 추가했다. Table 2는 enable 2명 한 쌍·coupled 2/3/4명 한 그룹만, M=2N, 균등 행동, 공통 공유 크기, 합일치/near50 해제, 각각-ever 성공·기존 낙상 보존을 반영했다.
- 사용자 지정 루트 경로에서 공통 규칙→enable 변경→coupled 변경→장면/테이블→미정/구현 상태로 재구성했다. 공통에 실제 크기·거리 기본값, PUSH 167개 profile 제약·물성, 행동별 공간 성공·낙상·통계를 모으고 승계안과 확정 조건을 구분했다. 이동된 파일의 내부·참조 링크도 갱신했다.
- 사용자 요청으로 중복 설명·코드 경로·과거 결과·구현 이력을 덜어내고 조건표·수치·그룹 성공·테이블 구성·미정 항목만 남겼다. 기본값의 미확정 상태는 유지했다.
- 현재 평가 규약과 structure에서 새 문서를 연결했다. 문서 링크·상대 경로·diff를 확인했으며 실행 코드·학습 설정·PDF는 변경하지 않았다. GPU 실행·학습·평가는 수행하지 않았다.

## 2026-11-06

### Distill900 2.5k 행동별 교집합 평가

- 사용자 요청으로 2.5k 저장을 기다린 뒤 GPU6에서 기존과 같은 4명·8토큰·512환경×seed0/1/2·ordered·900step·행동별 교집합 평가를 완료했다. 학습 종료·재시작은 하지 않았다. PTH epoch2500/frame327680000·hash·teacher 설정·finite, 세 seed의 이전 0.5k/1k/1.5k/2k 초기 상태 전체 필드 exact, 공유 상자 범위/격자, 기존·추가 지표 CPU/GPU 재채점 일치를 확인했다.
- 영역 Independent/Enabling/Coupled/Mixed Interaction Group Success Rate는 95.76±1.09/63.69±1.04/90.76±0.41/83.40±0.14%다. Mixed 평균은 1.5k 84.33±0.60%보다 0.93pp 낮아 현재 비교의 최고 평균은 1.5k다. [전체 지표·비교·재현](output_etc/check/stage2_distill_900_intersection_2500_20261106/report.md), [검증](output_etc/check/stage2_distill_900_intersection_2500_20261106/verification.json).

### Enable 행동별 교집합·Distill900 0.5k/1k/1.5k/2k 평가

- 사용자 선택으로 general 평가 전용 `--enable_size_mode per_action_intersection`을 추가했다. Native 관계/후행 행동을 먼저 뽑고 실제 asset 생성 전에 enable 선행자/후행 받침 공유 상자만 운반∩SIT/CLIMB/SUPPORT 크기 격자로 샘플링한다. 밀도 유지·질량 재계산·첫 episode 역할 고정으로 물리/관측/과제 연결을 일치시킨다. Solo·coupled·후행 payload 크기, 기본 학습과 native 평가는 유지했다. Manifest/result에 모드를 기록하고 학습/resume/viewer에서는 거부한다. [정의](markdowns/reference/evaluation.md#enable-행동별-크기-교집합-평가), [명령](markdowns/USAGE.md#통합-stage2-900step-버전).
- CPU111 passed,16환경900step 실행 및 학습 CLI 거부를 검증했다. 사용자 지정4개 PTH를4명·8토큰·512환경×seed0/1/2·ordered·900step으로 GPU2/6/7에서 총12회 평가했다. 모든 enable 공유 크기 범위, 체크포인트 간 동일seed 초기 상태 전체필드 exact,epoch/frame/hash/teacher 일치·finite,기존/추가 지표 CPU/GPU exact를 확인했다. 기존 학습 PID265779는 유지했고 평가 프로세스는 종료했다. [보고서](output_etc/check/stage2_distill_900_intersection_0500_1000_1500_2000_20261106/report.md), [검증](output_etc/check/stage2_distill_900_intersection_0500_1000_1500_2000_20261106/verification.json).
- 영역 Mixed Interaction Group Success Rate는0.5k/1k/1.5k/2k=83.02±1.15/83.05±1.17/84.33±0.60/82.83±1.42%, Enabling Interaction Group Success Rate는63.75±2.42/61.60±2.79/64.68±2.08/62.77±3.56%다. 이전 native ON_TOP enabling592쌍 중 행동별 받침 교집합 안에 있었던 것은65쌍이었다. 이번은 새 크기/초기 장면 분포의 별도 rollout이며 이전 trace 조건부 필터가 아니다.

### Distill900 0.5k·1k·1.5k 평가와 interaction group 성공률

- 기존 모든 지표를 보존하고 풀네임의 Independent Interaction Success Rate·Enabling Interaction Group Success Rate·Coupled Interaction Group Success Rate·Mixed Interaction Group Success Rate를 point/region별로 추가했다. Enable은 선행/후행 각각 ever, coupled는 실제 공유 상자 중심 ever, mixed는 seed별 세 유형 동일 가중 평균이다. Enable 선행자를 solo에 중복 집계하지 않고 공동 그룹은 인원과 관계없이1개로 센다. Seed 평균/표본std와 CPU trace 재채점도 연결했다. [정의](markdowns/reference/evaluation.md#통합-stage2의-interaction-group-성공률).
- 사용자 지정0.5k/1k/1.5k PTH를4명·8토큰·512환경×seed0/1/2·ordered·native random_scenario·900step으로 GPU6/7/2에서 평가했다. CPU21 passed,9회 모두 GPU/CPU 기존·새 지표/건수·finite 일치,3모델×각seed 초기 상태 전체필드 exact 일치,epoch/frame/SHA256·관계 분모 분할을 확인했다. 기존 학습 PID265779는 보존했고 평가 프로세스는 모두 종료했다. [보고서](output_etc/check/stage2_distill_900_eval_0500_1000_1500_20261106/report.md), [검증](output_etc/check/stage2_distill_900_eval_0500_1000_1500_20261106/verification.json).
- 영역 Mixed Interaction Group Success Rate는0.5k/1k/1.5k=75.79±0.14/75.73±0.47/76.32±0.21%다. Enabling Interaction Group Success Rate는41.99±1.04/40.67±0.56/42.66±1.24%다. 이번 Independent 지표는 혼합 샘플 안 solo이며 독립 전용 장면 또는2/8/16명 확장 평가 결과가 아니다.

### 900step VNC의 GPU UUID 거부 수정

- VNC가 숫자 GPU를 UUID로 바꾼 뒤 자식 launcher에 전달할 때900 버전 GPU 제한이 이를 거부하던 오류를 수정했다. `nvidia-smi`로 UUID/숫자를 물리 인덱스로 확인한 뒤 기존2/6/7 제한을 적용한다. 실행 명령과 실행 중 학습은 변경하지 않았다.
- GUI→자식 launcher의 숫자/UUID 허용·제외 GPU 거부를 포함해 CPU12 passed. GPU6의 실제 runtime UUID를 그대로 자식 test wrapper에 전달해500iteration student PTH·4명·혼합·1환경·2step 화면 없는 추론을 완료했다. 검증 프로세스는 종료했으며 VNC 화면은 새로 실행하지 않았다. [CPU](output_etc/check/stage2_900_gui_uuid_20261106/cpu.log), [추론](output_etc/check/stage2_900_gui_uuid_20261106/rollout.log).

### Distill900 teacher 사용자 지정21k·17k

- `stage2_distill_900` 학습 YAML의 teacher 경로만 사용자 지정 enable21k(11월5일22:58 run)·coupled17k로 변경했다. 1200step `stage2_distill`의 teacher와 공용 심링크는 유지한다. 실행 명령은 기존900 버전 train wrapper 그대로다.
- CPU에서 두 PTH의 epoch·원래 fusion·모델/RMS strict 로드·동결·사람별 행동 선택·finite를 확인했다. 학습과 시뮬레이션은 실행하지 않았다. [검증](output_etc/check/stage2_distill_900_teachers_21k17k_20261106/verification.json).

### 통합 RL·distillation 900step 컨피그

- 사용자 요청으로 `stage2_900`·`stage2_distill_900` 환경/학습 YAML과 전용 train/test/VNC/eval wrapper를 추가했다. 기존1200step 설정에서 에피소드 제한과 실험/저장 이름만 분리했으며 모델·샘플링·보상·teacher·수집/업데이트 설정은 같다. 학습 검증·viewer·정식 평가도900step으로 연결했다. 새 wrapper의 GPU는2/6/7만 허용하며 기본6이다. [명령](markdowns/config.md), [추가 옵션](markdowns/USAGE.md#통합-stage2-900step-버전).
- CPU203 passed/3 skipped, 원본과 YAML 차이·1024환경 학습 CLI 설정·잘못된1200step 입력 거부·wrapper/Python 구문·새 문서 링크를 확인했다. GPU6/7에서 각각16환경×6장면의 finite·alias·부분 reset·가중치 보존과899step 미종료/900step 종료·reset을 확인했다. Distill900 환경에서 기존1200step teacher 두 개의 로드/행동/동결도 확인했다. 학습 업데이트와 viewer는 실행하지 않았다. [CPU](output_etc/check/stage2_900_20261106/cpu.log), [RL 물리](output_etc/check/stage2_900_20261106/physics_rl/physics_validation.json), [distill 물리](output_etc/check/stage2_900_20261106/physics_distill/physics_validation.json).

- GPU2/6에서 기존 확인용 PTH로 각각16환경×seed0·ordered·900step 평가를 완료하고 CPU 재채점/성공 건수 exact 일치·finite를 확인했다. 이는 실행 경로 검증이며 새 모델 학습/성능 실험이 아니다. 검증 시뮬레이션은 모두 종료했다. [검증 요약](output_etc/check/stage2_900_20261106/verification.json).

### 통합 M≥N 토큰 예산 · ordered eval · Stage2 distillation

- 사용자 요청으로 통합 `stage2`의 M≥2N 제한을 M≥N으로 완화했다. 관계/행동을 뽑은 뒤 실제 작업 슬롯·ON_TOP 받침·enable alias를 합산하고 부족한 환경 행만 재샘플링한다. M≥2N은 기존 샘플/RNG를 보존한다. 불가능한 고정 프리셋과 M=N 고정mixed의 크기 비호환은 명확한 오류로 처리한다. [명세](markdowns/reference/stage2.md#물체와-관측).
- 통합 `stage2`·`stage2_distill` 및 두 teacher의4명/확장 viewer·eval 기본 배정을 ordered로 맞췄다. 단일 모델 eval에도 인원/토큰/5m 확장 인자를 연결하고 manifest/result에 배정 방식을 기록한다. 학습의 랜덤 구역 배정은 유지했다.
- `stage2_distill` 환경/학습 YAML·train/test/VNC/eval wrapper와 실행 가능한 teacher 모방 루프를 추가했다. Stage1 26k에서 시작해 actor encoder/RMS·critic/AMP를 고정하고 actor coordination/fusion/action head만 평균 행동 MSE로 학습한다. Enable/solo는 enable13.5k, 공동 PUSH는 coupled25.5k teacher 정답을 사용한다. Teacher 실행20iteration 후180iteration 선형 전환·65,536scene replay·optimizer/replay/epoch 복원을 지원한다. [실행](markdowns/USAGE.md#stage2-distillation), [학습 명세](markdowns/reference/stage2_distill.md).
- CPU265 passed/3 skipped/1 deselected. GPU5/6에서1024환경·기본1iteration 및 이어학습2iteration, 별도 설정의4명4토큰·student-only 수집1iteration을 확인했다. 검증 배치 MSE는 기본0.00963→0.00313, 이어학습0.00313→0.00209, student-only0.00778→0.00319로 감소했다. Stage1 원본의 action head 외159개 tensor와 입력RMS exact 일치·모델 finite·replay/optimizer 복원을 확인했다. [CPU](output_etc/check/stage2_distill_20261106/cpu_full.log), [체크포인트](output_etc/check/stage2_distill_20261106/checkpoint_verification.json).
- 두 teacher8명8토큰, student4명4토큰·8명8토큰을각16환경×seed0·ordered·1200step 실행하고 CPU 재채점/건수 exact 일치·finite를 확인했다. Student eval은 teacher를 로드하지 않는다. [teacher](output_etc/check/stage2_distill_20261106/teachers_eval8_m8/rescore.json), [student4](output_etc/check/stage2_distill_20261106/student_eval4_m4/general_seed0/rescore.json), [student8](output_etc/check/stage2_distill_20261106/student_eval8_m8/general_seed0/rescore.json). 이는 짧은 실행 검증이며 장기 성능 결과가 아니다. Viewer와 장기 학습은 실행하지 않았다.

## 2026-11-05

### 두 teacher 제로샷 viewer · 5명 이상 통합 그룹 배치

- enable/coupled PTH를 각각 원래 모델·전처리·RMS로 복원하고 공동 PUSH 그룹원만 coupled, enable 선행/후행·solo는 enable의 평균 행동으로 제어하는 test/VNC/eval wrapper를 추가했다. enable13.5k/coupled25.5k 정기 저장본의 읽기 전용 링크를 연결했다. 학습·가중치 병합은 없으며 경로/epoch/frame/해시·담당 규칙을 `teachers.json`에 기록한다. [실행법](markdowns/USAGE.md#두-teacher-제로샷-viewer).
- 5명 이상 추론은 solo/enable/coupled를 그룹 단위1:1:1로 순차 선택하고 1/2/2~4명을 할당한다. 초기 그룹 중심5m·BEFORE 배치 후 쌍 전체 중심 복원·불필요한 물체 물리/관측 제외·필수 alias/받침 보존·ordered 구역 배정을 viewer/eval에 공통 적용했다. 4명 학습은 유지한다. distillation 학습은 후속 작업이다. [설계](markdowns/reference/stage2.md).
- GPU5/6만 사용했다. CPU209 passed/3 skipped/1 deselected, 두 teacher 모델 각279개 tensor·RMS exact 복원/동결·행동 선택·제외 물체 영향 차단을 확인했다. 5/8/16명 각32환경×6장면에서5m·최대4명·관계/alias·부분 reset·finite·가중치 보존을 확인했다. 8명16환경×seed0의1200step 공간 채점과 CPU 재채점/성공 건수가 exact 일치했고 실제8명 VNC 렌더링을 확인했다. 기존 coupled 테스트 fixture와의 호환 오류를 수정 후 재검증했다. [CPU](output_etc/check/stage2_teachers_20261105/cpu_verified.log), [5명](output_etc/check/stage2_teachers_20261105/ordered5/physics_validation.json), [8명](output_etc/check/stage2_teachers_20261105/ordered8/physics_validation.json), [16명](output_etc/check/stage2_teachers_20261105/ordered16/physics_validation.json), [재채점](output_etc/check/stage2_teachers_20261105/eval8_ordered/rescore.json), [화면](output_etc/check/stage2_teachers_20261105/viewer_live.png).

### 통합 stage2 · BEFORE/COUPLED 관계별 RL 학습

- 사용자 승인 명세로 `stage2` 환경/학습 YAML과 train/test/VNC/eval 진입점을 추가했다. 4명 장면은 enable/coupled/혼합을 1:1:1로 샘플링하고 혼합은 BEFORE 2명+COUPLED 2명이다. 원래 관계별 물체·배치·보상을 연결하고 BEFORE AT는 20cm, solo/COUPLED AT는 3D 20cm 및 높이 1mm를 적용한다. [통합 명세](markdowns/reference/stage2.md).
- BEFORE/COUPLED 독립 임베딩과 공유 coordination/fusion을 추가했다. Stage1 26k에서 전이하고 actor encoder·입력 RMS를 동결한다. alias 물체 슬롯 충돌을 방지하고 BEFORE goalroute·대기 속도 감점을 연결했다. distillation과 5명 이상 실행은 아직 포함하지 않았다.
- CPU 검증 182 passed/3 skipped/1 deselected, wrapper 구문·문서 링크·diff 검사를 통과했다. 1024환경 확인 학습은 epoch2/frame262144까지 완료했으며 actor encoder 73개 tensor·입력 RMS의 전이 원본 exact 일치와 모델 finite를 확인했다. [검증](output_etc/check/stage2_20261105/checkpoint_verification.json), [CPU 로그](output_etc/check/stage2_20261105/cpu_tests.log). 사용자 지정 GPU5에서 64환경×6장면의 관계/alias·부분 reset·AMP/action/value/reward finite·가중치 보존을 확인했다. [물리 검사](output_etc/check/stage2_20261105/physics_gpu5/physics_validation.json). GPU5의 16환경×seed0 정식 평가 경로를 1200step 실행하고 CPU 재채점·성공 건수 exact 일치·finite를 확인했다. [재채점](output_etc/check/stage2_20261105/eval_gpu5/general_seed0/rescore.json). 이는 실행 검증이며 학습 성능 평가는 아니다.

### zerobg16k–20k ·1k 간격 정밀 평가 완료 ·17k 최종 추천

- 사용자 요청으로16k/17k/18k/19k/20k PTH/YAML을 복사 고정해 각4명·8물체 혼합512환경×seed0/1/2, 총15회를 GPU3/4/5에서 평가했다. 기존 스폰/distractor·ordered·loco·평균 액션·거리/물성·1200step을 유지하고10k–15k·19.5k의21회는 재사용했다. [전체 비교·최종 추천](output_etc/check/zerobg16k20k_eval512_gpu345_20261105/report.md), [선정 PTH](output_etc/check/zerobg16k20k_eval512_gpu345_20261105/selection.json), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 16k/17k/18k/19k/20k의 Ever 그룹 합산 Coupled SR은89.43±0.75/92.31±0.56/88.79±0.75/91.63±0.56/90.76±2.19%다. 17k의 단독 PUSH92.18±1.53%, 나머지4행동 평균94.40±1.18%, I/C/Mixed=93.08±0.72/92.31±0.56/92.70±0.62%다. 공동2/3/4명93.97±0.80/89.55±1.02/91.88±1.54%이며 공동 전체·공동4명·단독 PUSH·Mixed의 최고 평균을 근거로17k를 주 후보로 추천했다. 공동3명만의 최고는19.5k다. 학습/추론 기본 PTH나 학습 프로세스는 변경하지 않았다.
- 새15회 모두 CPU/GPU 재채점·finite·개인/그룹 건수·native gate·actor/critic zerobg/SELF·NONE0·가중치/PTH/YAML/epoch/frame·지면/거리 검증을 통과했다. 같은 seed의 초기12필드가 기존7개 모델과 exact일치했다. GPU당 평가 최대3개·최종0개이며 GPU0/1/2·기존 학습/MPS는 변경하지 않았다. [검증](output_etc/check/zerobg16k20k_eval512_gpu345_20261105/verification.json).

### 요청 시점 최신 zerobg19.5k · 정밀 평가 및15k 비교

- 사용자 요청 시점의 최신 정기 PTH와 일반 이름 PTH가 모두 epoch19500임을 확인했다. 정기19.5k(epoch19500/frame2555904000) PTH/YAML을 복사 고정해 GPU5에서 기존4명·8물체 혼합512환경×seed0/1/2를 평가했다. 스폰/distractor·ordered·loco·평균 액션·거리/물성·1200step을 유지하고10k–15k의18회를 재사용했다. [비교표](output_etc/check/zerobg19500_trend512_gpu5_20261105/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever 영역19.5k의 그룹 합산 Coupled SR91.40±0.80%, 단독 PUSH90.42±0.42%, 나머지4행동 동일 가중 평균94.04±1.53%다. 15k 대비 공동은2.99pp 높고 단독 PUSH/나머지는0.30/0.01pp 낮아 비슷한 수준이다. 공동2/3/4명91.76±0.66/92.70±1.89/88.28±2.29%로 개선은 주로2·3명에서 나왔다. 유형별 I/C/Mixed=91.96±1.34/91.40±0.80/91.68±0.88%이며 현재 비교에서 공동 전체와Mixed 평균이 가장 높다.
- 새3회 CPU/GPU 재채점·finite·개인/그룹 건수·native gate·actor/critic zerobg/SELF·NONE0·가중치/PTH/YAML/epoch/frame·지면/거리 검증을 통과했다. 같은 seed의 초기12필드가10k–15k와 exact일치했다. GPU5 평가 최대3개·최종0개이며 GPU0/1/2·기존 학습/MPS는 변경하지 않았다. [검증](output_etc/check/zerobg19500_trend512_gpu5_20261105/verification.json).

### zerobg15k · 정밀 평가 및10k–15k 비교

- 사용자 요청으로15k(epoch15000/frame1966080000) PTH/YAML을 복사 고정해 GPU5에서 기존4명·8물체 혼합512환경×seed0/1/2를 평가했다. 기존 스폰/distractor·ordered·loco·평균 액션·거리/물성·1200step을 유지하고10k–14k의15회를 재사용했다. [비교표](output_etc/check/zerobg15000_trend512_gpu5_20261105/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever 영역15k의 그룹 합산 Coupled SR88.41±1.32%, 단독 PUSH90.72±1.23%, 나머지4행동 동일 가중 평균94.05±1.25%로14k보다 세 항목 모두 높다. 공동2/3/4명88.79±3.22/88.00±3.66/88.45±3.32%다. 유형별 I/C/Mixed=92.26±0.51/88.41±1.32/90.34±0.43%이며 현재10k–15k 비교에서 공동 전체와Mixed 평균이 가장 높다.
- 새3회 CPU/GPU 재채점·finite·개인/그룹 건수·native gate·actor/critic zerobg/SELF·NONE0·가중치/PTH/YAML/epoch/frame·지면/거리 검증을 통과했다. 같은 seed의 초기12필드가10k–14k와 exact일치했다. GPU5 평가 최대3개·최종0개이며 GPU0/1/2·기존 학습/MPS는 변경하지 않았다. [검증](output_etc/check/zerobg15000_trend512_gpu5_20261105/verification.json).

## 2026-11-04

### zerobg14k · 정밀 평가 및10k–14k 비교

- 사용자 요청으로14k(epoch14000/frame1835008000) PTH/YAML을 복사 고정해 GPU5에서 기존4명·8물체 혼합512환경×seed0/1/2를 평가했다. 기존 스폰/distractor·ordered·loco·평균 액션·거리/물성·1200step을 유지하고10k–13k의12회를 재사용했다. [비교표](output_etc/check/zerobg14000_trend512_gpu5_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever 영역14k의 그룹 합산 Coupled SR87.87±1.89%, 단독 PUSH88.75±2.10%, 나머지4행동 동일 가중 평균93.18±1.41%다. 공동2/3/4명88.68±1.54/87.63±2.34/85.83±8.11%이며13k보다2명은 높고3/4명은 낮다. 유형별 I/C/Mixed=90.90±0.57/87.87±1.89/89.38±0.99%다. 13k 대비 공동 전체는 소폭 높고 단독은 소폭 낮으며4명 공동의 seed 편차가 크다.
- 새3회 CPU/GPU 재채점·finite·개인/그룹 건수·native gate·actor/critic zerobg/SELF·NONE0·가중치/PTH/YAML/epoch/frame·지면/거리 검증을 통과했다. 같은 seed의 초기12필드가10k–13k와 exact일치했다. GPU5 평가 최대3개·최종0개이며 GPU0/1/2·기존 학습/MPS는 변경하지 않았다. [검증](output_etc/check/zerobg14000_trend512_gpu5_20261104/verification.json).

### zerobg13k · 6/8명 all/keep3 공유 중심40cm 재집계

- 기존 네64환경 궤적의 공유 상자 중심→공유 목표3D 거리를 CPU float64로20/40cm 재채점했다. Ever20→40cm는6명 all21.88→26.56%, keep3 15.62→21.88%,8명 all25.00→32.81%, keep3 25.00→39.06%다. Final은 참고 파일에 보존했다. 추가 GPU 시뮬레이션·학습/환경/종료 조건 변경은 없다. [반경 비교](output_etc/check/zerobg13000_n68_all_keep3_env64_gpu35_20261104/radius_report.md).
- 기존20cm 성공 플래그/그룹 건수 exact일치·반경 포함 관계·Final→Ever·finite를 네 실행 모두 확인했다. 기존 그룹 유효성 규칙을 유지했다. [검증](output_etc/check/zerobg13000_n68_all_keep3_env64_gpu35_20261104/radius_verification.json), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).

### zerobg13k · 6/8명 전원 공동 all/keep3 ·64환경

- 사용자 요청으로 같은13k(epoch13000/frame1703936000)를 GPU3(6명)·5(8명)에서 각 all/keep3·64환경×seed0·ordered·loco·평균 액션·1200step으로 총4회 평가했다. 인원별 원래 상자 크기/질량·스폰/distractor를 유지한 전원 공동이며, 그룹 최대4명 혼합 확장과 별개다. keep3는 미접촉 우선3명을 유지하고 선택 밖 COUPLED를NONE으로 바꾸며 critic와 전체 그룹 gate는 유지한다. [결과](output_etc/check/zerobg13000_n68_all_keep3_env64_gpu35_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 공유 중심20cm Ever의 all→keep3는6명21.88%(14/64)→15.62%(10/64),8명25.00%(16/64)→25.00%(16/64)다. Final은 각각20.31→10.94%,21.88→14.06%로 참고 보존했다. 이번64환경1seed에서는 keep3의 Ever 개선이 없다.
- 네 실행 모두 CPU/GPU 재채점·finite·그룹/개인 건수·가중치/PTH/YAML/epoch/frame·native gate·actor/critic zerobg 설정 검증을 통과했다. all/keep3 초기12필드 exact일치, 매step 선택수/미접촉 우선 및 actor4레이어4800회의 COUPLED→NONE·padding없음을 확인했다. GPU당 평가 최대2개·최종0개이며 기존 학습/MPS·GPU0/1/2는 변경하지 않았다. [검증](output_etc/check/zerobg13000_n68_all_keep3_env64_gpu35_20261104/verification.json).

### zerobg13k · 정밀 평가 및10k–13k 비교

- 사용자 요청으로13k(epoch13000/frame1703936000) PTH/YAML을 복사 고정해 GPU5에서 기존4명·8물체 혼합512환경×seed0/1/2를 평가했다. 기존 스폰/distractor·ordered·loco·평균 액션·거리/물성·1200step을 유지하고10k·11k·12k의9회를 재사용했다. [비교표](output_etc/check/zerobg13000_trend512_gpu5_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever 영역13k의 그룹 합산 Coupled SR87.25±2.13%, 단독 PUSH88.93±1.83%, 나머지4행동 동일 가중 평균93.56±0.93%다. 공동2/3/4명86.29±1.68/89.52±0.89/86.51±5.35%이며10k보다 공동 평균은 높고 단독 PUSH·나머지 평균은 낮다. 유형별 I/C/Mixed=91.36±1.18/87.25±2.13/89.30±1.07%로10k Mixed89.15±0.81%와 비슷하다. 공동 우선 후보로13k를 추가하되4명 공동의 seed 편차를 함께 기록했다.
- 새3회 CPU/GPU 재채점·finite·개인/그룹 건수·native gate·actor/critic zerobg/SELF·NONE0·가중치/PTH/YAML/epoch/frame·지면/거리 검증을 통과했다. 같은 seed의 초기12필드가10k·11k·12k와 exact일치했다. GPU5 평가 최대3개·최종0개이며 GPU0/1/2·기존 학습/MPS는 변경하지 않았다. [검증](output_etc/check/zerobg13000_trend512_gpu5_20261104/verification.json).

### ground18k·19k·20k · 정밀 평가 및 zerobg10k 비교

- 사용자 요청으로 고정18k/19k/20k를 GPU3/4/5에서 각각512환경×seed0/1/2, 총9회 평가했다. 기존4명·8물체 혼합·ordered·loco·평균 액션·스폰/distractor·거리/물성·1200step 첫 episode를 유지하고 ground9.5k·16k·17k 및 zerobg10k 기존12회를 재사용했다. [비교표·판단](output_etc/check/ground18k19k20k_eval512_gpu345_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever 영역18k/19k/20k의 그룹 합산 Coupled SR은84.88±2.15/85.59±1.18/86.20±1.76%, 단독 PUSH81.62±1.93/77.50±4.26/79.94±0.87%, 나머지4행동 동일 가중 평균92.72±1.42/92.21±1.88/92.06±1.57%다. zerobg10k는 각각86.00±1.08/90.18±1.55/94.60±1.18%다. 공동4명은19k87.25±2.83%로 zerobg83.32±2.92%보다 높은 평균이지만, 공동 전체의 지속 개선과 단독 회복이 없어 현 시점 추가 ground 학습 중단을 추천했다. 비교용 PTH·학습 프로세스는 보존했다.
- 새9회 CPU/GPU 재채점·finite·개인/그룹 건수·native gate·ground 높이/거리·actor/critic 설정·가중치/PTH/YAML/epoch/frame 검증을 통과했다. 각 새 모델과 재사용4개 모델은 같은 seed의 초기12필드가 exact일치했다. GPU당 평가 최대3개·최종0개이며 GPU0/1/2·기존 학습/MPS는 변경하지 않았다. [검증](output_etc/check/ground18k19k20k_eval512_gpu345_20261104/verification.json).

### zerobg10k · 6/8/12/16명 혼합 확장 완료 · Stage1 유형별 SR 비교

- 고정 zerobg10k를 GPU3/4/5/6에서 인원6/8/12/16별512환경×seed0/1/2, 총12회 평가했다. 승인된5m 그룹 중심 배치·불필요한 distractor 물리/관측 제외·공동 최대4·ordered·loco·평균 액션·기존 거리/물성·1200step을 적용했다. 4명은 기존 스폰/distractor 결과를 재사용했다. Stage2 actor/critic coordination은 매1200step 정상 실행했다. [두 집계·모델 비교](output_etc/check/zerobg10k_group_gap5_mixed512_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 유형별 SR은 단독 개인 I와 공동 그룹 C의 성공 건수를 각각 합산하고 Mixed=(I+C)/2로 계산했다. zerobg10k의6/8/12/16명 I=89.04±0.82/87.25±0.28/78.59±1.49/69.25±0.29%, C=84.19±1.86/82.83±0.55/78.31±0.86/75.14±0.46%, Mixed=86.61±0.89/85.04±0.22/78.45±0.90/72.20±0.37%다. 기존 행동별·공동 인원별 동일 가중 평균도 별도 보존했다. Stage1 26k는 완료된4/6/8/12/16명 결과만 CPU 재집계했다. 16명 Stage1 I/C/Mixed는71.93±0.27/58.75±0.58/65.34±0.34%다.
- 새12회 모두 CPU 재채점/finite/건수·같은 인원/seed의 Stage1 초기12필드 exact일치·그룹 평행이동/거리·제거 토큰 영향 차단·가중치/PTH/YAML/epoch/frame 검증을 통과했다. 두 모델30개 seed 결과의 원본 개인/그룹 성공 건수를 별도로 대조했다. GPU당 평가 최대1개·최종0개·실패0회이며 GPU0/1/2·기존 학습/MPS는 변경하지 않았다. [검증](output_etc/check/zerobg10k_group_gap5_mixed512_20261104/verification.json), [건수 대조](output_etc/check/zerobg10k_group_gap5_mixed512_20261104/count_audit.json), [프로세스 감사](output_etc/check/zerobg10k_group_gap5_mixed512_20261104/process_audit.json).

### zerobg10k·11k·12k · 유형별 Interaction-Group SR 재집계

- 사용자 제안 정의로기존4명512환경×3seed의Ever영역결과를CPU재집계했다. I는단독전체개인건수, C는인원2·3·4명구분없이공동그룹건수로합산하고 E가없어 Mixed=(I+C)/2다. C성공은기존공유상자중심20cm를유지했다. 각seed에서비율을구한뒤3seed평균±표본표준편차를계산했다. 추가GPU실행과기존지표/결과변경은없다.
- 10k/11k/12k의I=92.29±0.76/91.14±1.25/91.53±1.14%, C=86.00±1.08/83.73±0.63/82.84±0.70%, Mixed=89.15±0.81/87.44±0.56/87.19±0.46%다. 기존CPU재채점통과와개인수2048/seed를확인했다. [정의·결과](output_etc/check/zerobg10k12k_interaction_group_macro_20261104/report.md).

### zerobg12k · 10k/11k/12k 정밀 비교

- 사용자 요청으로12k(epoch12000/frame1572864000) PTH/YAML을복사고정하여GPU5에서4명·8물체혼합512환경×seed0/1/2를완료했다. 기존스폰/distractor·ordered·loco·평균액션·물성·1200step을유지하고10k/11k의검증된6회를재사용했다. [비교표](output_etc/check/zerobg12000_trend512_gpu5_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever영역10k/11k/12k 공동2·3·4명동일가중평균85.70±0.94/82.93±1.58/82.39±1.42%, 단독5행동평균93.71±0.64/92.34±0.71/92.45±0.97%, 혼합50:50평균89.71±0.68/87.63±0.72/87.42±0.61%다. 12k단독PUSH91.46±0.87%는10k90.18±1.55%보다높고공동평균은낮았다.
- 새3회와재사용6회CPU재채점/finite/건수·초기12필드bitwise일치·actor/critic zerobg/SELF·NONE0·가중치불변·PTH/YAML해시/epoch/frame 검증을통과했다. GPU5평가peak3개·최종0개이며GPU0/1/2·기존학습/MPS는변경하지않았다. [검증](output_etc/check/zerobg12000_trend512_gpu5_20261104/verification.json).

### 4명 초과 그룹 스폰 명세 · Stage1 26k 확장 평가 완료

- 사용자 요청으로[4명 초과 그룹 스폰](markdowns/reference/group_spawn_over4.md)을작성했다. 4명이하기존유지/초과에만5m그룹중심배치·기존loco/그룹내상대위치보존·공동최대4·상자/목표거리유지·distractor물리/관측제외·ON_TOP필수받침유지와현재진단경로/검증범위를명시했다. structure에참조링크를추가했다.
- 같은조건으로Stage1 26k의6/8/12/16명각512환경×seed0/1/2, 총12회를GPU3/4/5/6에서완료했다. 원래Stage1 encoder/head/RMS·merge_shared=False를사용하고Stage2 coordination을실행하면오류를내도록차단했다. 배치재시도한도만512로늘리고사람/그룹/거리/충돌조건은유지했다. 인원별최대4/5/9/52회로모든환경이통과했다. [비교표](output_etc/check/stage1_26k_group_gap5_mixed512_20261104/report.md), [설정](output_etc/check/stage1_26k_group_gap5_mixed512_20261104/manifest.json).
- Ever영역 단독5행동평균85.14±1.04/85.00±0.56/80.19±0.18/73.76±0.67%,공동2/3/4명동일가중평균69.50±2.18/68.20±1.62/63.12±2.28/58.12±0.83%,혼합50:50평균77.32±1.35/76.60±1.09/71.66±1.20/65.94±0.71%다. 기존4명은원래스폰/distractor결과를재사용했다.
- 12회모두원래Stage1 actor출력exact일치·coordination우회·가중치/해시/epoch26000/frame3407872000·그룹내상대위치/자세/속도·제거토큰영향차단·거리범위·CPU재채점/finite/건수검증을통과했다. 초기설정검사거부와0.1mm좌표검사거부로그를보존하고reset직전한도적용/0.5mm검증으로수정했다. 실패환경제외나0%치환은없다. [검증](output_etc/check/stage1_26k_group_gap5_mixed512_20261104/verification.json). GPU0/1/2·기존학습/MPS는종료하지않았다.

### zerobg11k · 일반512환경×3회 완료

- 사용자 요청으로11k(epoch11000/frame1441792000)를복사고정하여GPU4에서512환경×seed0/1/2를완료했다. 기존4명·8물체혼합·ordered·loco·평균액션·원래스폰/distractor·1200step을유지했다. [비교표](output_etc/check/zerobg11000_trend512_gpu4_20261104/report.md), [검증](output_etc/check/zerobg11000_trend512_gpu4_20261104/verification.json).
- Ever영역10k→11k 단독5행동평균93.71±0.64→92.34±0.71%, 공동2/3/4명동일가중평균85.70±0.94→82.93±1.58%, 혼합50:50평균89.71±0.68→87.63±0.72%다. 11k 공동2/3/4명84.95/83.60/80.23%, 단독PUSH89.03%다. 새3회CPU재채점/finite/건수·초기12필드일치·actor/critic zerobg·가중치/해시/epoch/frame 검증을통과했다. GPU0/1·기존학습/뷰어는변경하지않았다.

### 5m 그룹 배치 · zerobg10k 혼합 viewer

- 사용자 요청으로진단폴더에 `view.sh`/`viewer.py`를추가했다. 4명이하는기존스폰/distractor를유지하고초과는기존초기위치에서그룹전체XY이동·그룹중심최소5m·공동최대4·불필요한distractor물리/관측제외를적용한다. ON_TOP받침과작업물체·ordered·loco·평균액션을유지하고기본10씬을연속재생한다. 카메라는사람/필수상자/목표만포함해parked distractor에의한과도한줌아웃을피한다. 표준코드는수정하지않았다. [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- GPU3/4에서4·8·16명각2씬×12step headless viewer경로를통과했다. 각24회finite action·가중치불변, 4명그대로/8·16명매씬그룹이동, 각씬제거토큰교란에대한encoder출력exact불변을확인했다. GUI실제렌더링/정식성공률은이번검증대상이아니다. [검증](output_etc/check/group_spawn_gap5_layout64_20261104/viewer_checks.json).

### 5m 그룹 원형 이동 · 8/16명 배치만 검증

- 사용자 요청으로GPU3/4에서각8/16명·64환경·seed0의배치를검증했다. 기존N명 사람원반경2.5m+loco XY와그룹내상대위치·자세·속도를유지하며그룹전원과kinematic body에같은XY이동을적용했다. 그룹중심간최소거리5m를유지하도록그룹수에따른원반경을계산하고자리를무작위배정했다. 기존혼합샘플링·공동최대4·작업거리/충돌조건을유지하고불필요한distractor만멀리옮겼다. 표준코드는변경하지않았다. [결과](output_etc/check/group_spawn_gap5_layout64_20261104/report.md).
- 8/16명모두64/64환경배치통과, 첫시도58/40환경·평균시도1.13/2.06회·최대4/26회이며생성/검증은11.1/15.4초였다. GPU검사와CPU재검증에서그룹내상대위치/자세/속도·body동시이동·5m간격·그룹상한·필수물체보존을확인했다. 정책rollout0step이며관측토큰제외/정책성공률은측정하지않았다. [검증](output_etc/check/group_spawn_gap5_layout64_20261104/verification.json), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).

### zerobg10k · 4명 distractor 제거512환경×3회

- 사용자 요청으로 GPU3(seed0/1)·4(seed2)에서 zerobg10k를4명·8물체 슬롯·512환경×3회·ordered·loco·평균 액션·1200step으로 평가했다. 기존 초기 장면을 생성한 뒤 불필요한 distractor를 물리 장면에서 멀리 옮기고 관측에서 제외했다. ON_TOP 받침과 실제 작업 상자는 유지했으며 사람 스폰·표준 코드는 변경하지 않았다. [비교표](output_etc/check/zerobg10k_nodistractors512_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever 영역 단독5행동 평균93.71±0.64→92.91±1.51%, 공동2·3·4명 동일 가중 평균85.70±0.94→82.77±0.89%, 혼합50:50 평균89.71±0.68→87.84±0.52%다. 공동2/3/4명85.73/83.88/78.68%, 단독PUSH90.93%다. 물리 장애물과 관측 토큰을 동시에 제거한 비교이며, 제거 후 새로 샘플링한 장면의 성능과 구분한다.
- 세seed의 초기12필드(상자는 제거 전 전체·제거 후 필수 물체) 일치·필수 물체와 관측 보존·제거 토큰 교란에 대한 encoder 출력 불변·CPU재채점/finite/건수·가중치 불변·PTH/YAML해시와epoch10000/frame1310720000 검증을 통과했다. GPU0/1/6·기존 학습·MPS는 종료하지 않았다. [검증](output_etc/check/zerobg10k_nodistractors512_20261104/verification.json).

### zerobg10k · 12명 확대·인접 스폰512×3 완료 /16명3seed 배치 실패

- 사용자 승인으로12/16명 사람 반경을2.5×sqrt(N/4)=4.33/5m로 넓히고 같은 공동 그룹을 원 위 연속 슬롯에 배치했다. 실제rootXY를슬롯에고정하고kinematic body를같이옮기며 RSI자세/높이/속도·그룹/행동/RNG를유지한다. 공동은2~min(4,남은 인원), 물체2N개·ordered·loco·평균 액션·1200step·원래 작업거리/충돌/물성·distractor반경4.5m를 유지했다. 표준학습/샘플러 코드는 변경하지 않았다. [최종 비교표](output_etc/check/zerobg10k_mixed_cap4_anchored_grouped512_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- GPU4에서12명512환경×seed0/1/2가각112/117/31회배치후전부통과하여평가를완료했다. Ever영역 단독5행동평균85.96±0.82%,공동2/3/4명평균81.69±0.61%,혼합50:50평균83.82±0.72%다. 공동2/3/4명82.79±0.67/80.51±0.77/81.77±2.73%,단독SIT/CLIMB/AT/ON_TOP/PUSH96.83/97.43/81.88/67.83/85.83%다. 기존4/6/8명9회결과를재사용하지만해당인원은기존반경2.5m·기존스폰이므로인원수만바꾼비교로해석하지않는다.
- 16명seed0/1/2는GPU3/2/5에서각512회한도까지설정변경없이시도했으나2/3/1환경배치미통과로모두중단됐다. 세seed모두rollout성공률은없다. 총6환경중5환경은단독PUSH후보배치가남았고1환경은최종장면검사가남았다. 배치실패를정책성공률0%로넣거나미통과환경을제외하지않았다. [실패 좌표·미배치그룹](output_etc/check/zerobg10k_mixed_cap4_anchored_grouped512_20261104/layout_failures.json).
- 완료3회의CPU재채점/건수/finite·actor/critic zerobg/SELF·NONE0·countnorm없음·가중치불변·PTH해시/epoch10000/frame1310720000·AT/ON_TOP거리1~5m·그룹최대4 검증을통과했다. 스폰슬롯순열/연속·root/body동시XY이동·자세/속도/높이/RNG불변을검증했다. PhysX반영후rootXY는1e-4m허용오차로검증하며나머지root성분은exact다. 실패3회는스폰/실패건수만검증하며정책재채점은없다. 최종신규평가프로세스0개이며GPU0/1/6·본학습·MPS를종료하지않았다. [검증](output_etc/check/zerobg10k_mixed_cap4_anchored_grouped512_20261104/verification.json).
- 중간시도는보존했다. [반경만확대한시험](output_etc/check/zerobg10k_mixed_cap4_spawnscale512_20261104/report.md)은147/93회후145/462환경정체로중단했다. [nominal슬롯시험](output_etc/check/zerobg10k_mixed_cap4_groupedspawn512_20261104/stop_reason.json)은RSI의기존XY변위가남아실제인접스폰과달라져최종방식으로수정했다. 두중간시도의정책성능수치는없다.

### Stage1 PUSH26k · 현재4명 혼합512환경×3회 비교

- 사용자 요청으로 원본26k(epoch26000/frame3407872000)를 복사 고정해 GPU2에서512환경×seed0/1/2를 평가했다. zerobg10k의4명/8물체·기존 단독/공동 혼합·ordered·loco·지면/현재 거리·물성/distractor·1200step 첫 episode를 사용했다. Stage1 encoder/action head/정규화·merge_shared=False를 복원하고 Stage2 coordination은 실행하지 않았다. [비교표](output_etc/check/stage1_26k_current_mixed4_512_gpu2_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever 영역 단독5행동 평균/공동2·3·4명 동일 가중 평균/혼합50:50 평균은 Stage1 88.52±1.81/71.64±0.59/80.08±0.62%, zerobg10k 93.71±0.64/85.70±0.94/89.71±0.68%다. 단독PUSH는95.86±0.57→90.18±1.55%, AT74.28±6.55→89.31±3.48%, ON_TOP영역80.81±3.87→90.76±2.96%다. 원래Stage1 환경/900step 평가와 구분한다.
- 세seed 모두 기존zerobg 초기12필드 bitwise일치, 첫 실제 관측에서 원래Stage1 actor와 평균/표준편차 출력 exact일치, 원본가중치 복사 exact일치·rollout가중치 불변·CPU재채점/건수/finite·PTH와환경해시 검증을 통과했다. 표준 모델·학습 코드는 변경하지 않았다. [검증](output_etc/check/stage1_26k_current_mixed4_512_gpu2_20261104/verification.json).

### zerobg10k · 혼합 인원 확장 · 6/8명 완료,12/16명 배치 실패

- 전체 환경 인원6/8/12/16명·물체2N개·각512환경×seed0/1/2를 GPU2/5/4/3에서 시작했다. 기존 혼합 샘플링을 유지하고 공동 그룹 크기만2~min(4,남은 인원)으로 제한했다. CPU검사에서4명이하 원본 그룹/RNG 일치와 확장 그룹 최대4명을 확인했다. 8/12/16명은 기본배치64회 소진으로 첫seed가 실패하여, 원래 사람/그룹·거리/충돌조건을 유지한 채 진단 harness의재시도한도만512회로 늘렸다. 실패로그와 검증 단계 설정거부로그를 보존했다. [진행/비교표](output_etc/check/zerobg10k_mixed_cap4_scaling512_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 6/8명 각3회는 CPU재채점/건수/finite·가중치불변·그룹상한 검증을 통과했다. Ever영역 단독/공동동일가중/혼합50:50은6명90.63±0.33/80.00±1.80/85.31±0.75%,8명89.85±0.44/77.03±1.90/83.44±0.77%다. 12명은512회에서도9환경 배치 실패로 중단됐고16명도512회 배치 소진으로 중단되어 두 인원의 성능수치는 없다. 실패환경을0% 처리하거나 제외하지 않았다. 실행 중인 본학습·표준 샘플러는 변경하지 않았다. [현재 상태](output_etc/check/zerobg10k_mixed_cap4_scaling512_20261104/status.json).
- 실패 원인 분해용 `diagnose_layout.py`를 추가하고 GPU4/3에서12/16명·512환경·seed0·8회 배치만 진단한다. 각회 충돌 판정은 원래 함수와 exact일치 검사를 수행한다. 첫회 실패는12명359/512,16명497/512이며 상자-사람 근접은각220/452환경이었다(실패항목 중복). 스폰 반경 확대는 아직 적용하지 않았다. [12명 진단](output_etc/check/zerobg10k_mixed_cap4_scaling512_20261104/layout_diagnostic_n12/layout_diagnostic.json), [16명 진단](output_etc/check/zerobg10k_mixed_cap4_scaling512_20261104/layout_diagnostic_n16/layout_diagnostic.json).

### ground/zerobg10k · 최초 전원 접촉 전후 이동 CPU 재집계

- 사용자 승인 기준으로 기존 두 모델의512환경×seed0/1/2 trace/gate_trace/initial을 CPU로 재집계했다. 추가 GPU 시뮬레이션은 없다. 최초 전원 on_face·유효 접촉 이전 초기 공유 상자 중심 대비XY변위≥50cm를 조기밀기로 정의했다. 전원 접촉 미달성은 전체1200step을 검사하고, 최초접촉이후 접촉이탈은 조기밀기로 세지 않는다. native gate의목표도착예외와분리했다. [비교표·정의](output_etc/check/ground_zerobg10k_gate_timing_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 모델별1456그룹(2명778·3명421·4명257) 합산 조기50cm 비율 ground43.61%(635)/zero46.02%(670), 전원접촉달성94.16/95.12%, 접촉달성그룹중추가50cm이동98.69/99.21%다. 2/3/4명 조기밀기는 ground31.62/51.07/67.70%,zero33.68/53.68/70.82%다. 접촉 전 최대변위 중앙값0.430/0.465m,90백분위1.511/1.354m다. 조기50cm없이접촉후추가이동까지달성55.84/53.64%다.
- 접촉 후 추가50cm이동에는 이미 움직이던 장면이 포함되며, 소요시간은 반응/출발 지연이 아니다. 조기50cm 없음도 완전정지율로 해석하지 않는다. 초기12필드일치·공동그룹분모·물리공유중심일치·전원접촉→nativegate·낙상재활성없음·원본재채점검증과입력해시를 확인했다. 표준 코드·학습·원본 결과는 변경하지 않았다. [검증](output_etc/check/ground_zerobg10k_gate_timing_20261104/verification.json).

### zerobg10k · 일반512환경×3회 · 공동/단독 동반 개선

- 사용자 요청으로 zerobg10000epoch/frame1310720000 PTH/YAML을 복사 고정해 GPU4에서각512환경×seed0/1/2 총3회를 평가했다. 4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode이며 zerobg8k/9k 및 ground10k 기존9회를 재사용했다. [비교표](output_etc/check/zerobg10000_trend512_gpu4_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- zerobg9k→10k 공유 중심20cm 공동 전체 Ever81.14→86.00%, 단독 PUSH87.19→90.18%, 나머지4행동 영역 평균91.88→94.60%로 세 항목이 함께 개선됐다. 공동2/3/4명86.06/87.72/83.32%이며 동일10k ground82.86/81.31/81.02%보다 높다. Final은공동84.07%,PUSH88.00%,나머지영역87.60%다.
- 새3회와재사용9회 CPU재채점·JSON건수·finite·native gate·actor/critic zerobg/SELF·NONE0·가중치 불변·PTH/YAML해시·epoch/frame·지면 높이/거리 검증을 통과했다. seed별초기12필드가모든비교PTH와bitwise일치했다. GPU4peak3개·최종0개이며GPU0·1·본학습·표준 코드는 건드리지 않았다. [검증](output_etc/check/zerobg10000_trend512_gpu4_20261104/verification.json).

### zerobg9k · 6/8명 all/keep3 · 20/40cm

- 사용자 요청으로 zerobg9k를 복사 고정하고 GPU4(all)·5(keep3)에서6명/12물체·8명/16물체 각64환경×seed0, 총4회를 평가했다. ordered·loco·평균 액션·원래 인원별 상자 크기/무게·distractor·1200step 첫 episode를 유지했다. 같은 궤적을 공유 상자 중심→공유 목표3D20/40cm로 재집계했다. [반경 비교표](output_etc/check/zerobg9000_n68_all_keep3_radius_env64_gpu45_20261104/radius_report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- Ever20→40cm는6명 all12.50→18.75%,keep3 21.88→26.56%;8명 all34.38→37.50%,keep3 31.25→37.50%다. Final은6명 all12.50→17.19%,keep3 12.50→20.31%;8명 all26.56→34.38%,keep3 21.88→35.94%다.64환경1seed에서keep3의Ever개선은6명에만 나타났다.
- CPU재채점·finite·가중치 불변·PTH/YAML해시·epoch/frame·초기12필드 일치·native gate·매행동keep3선택/actor4층NONE전환/critic무개입 검증을 통과했다.20cm저장 성공플래그/건수 정확 일치·20cm⊆40cm·Final⊆Ever도 확인했다. GPU별평가peak1개·최종0개이며GPU0·1·본학습·표준 성공조건은 변경하지 않았다. [일반 검증](output_etc/check/zerobg9000_n68_all_keep3_radius_env64_gpu45_20261104/verification.json), [반경 검증](output_etc/check/zerobg9000_n68_all_keep3_radius_env64_gpu45_20261104/radius_verification.json).

### ground17k · 일반512환경×3회 · 후반 추세

- 요청 시점 새 정수 epoch ground17k를 GPU3에서각512환경×seed0/1/2로 평가했다. zerobg는9.5k까지 저장되어 사용자1k간격에 따라 이번 신규 대상에서 제외했다. 4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode를 유지하고 ground14k/16k·zerobg8k/9k 기존12회 결과를 재사용했다. [비교표](output_etc/check/ground17000_trend512_gpu3_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- ground16k→17k 공유 중심20cm 공동 전체 Ever85.85→86.68%, 단독 PUSH80.74→81.11%, 나머지4행동 영역 평균92.99→90.28%다.17k 공동2/3/4명86.89/85.55/88.02%이며 AT80.98%,ON_TOP영역87.12%로16k의86.83/90.85%보다낮다. Final은공동84.14%,PUSH78.56%,나머지영역83.85%다. 공동/PUSH 소폭 개선과 나머지 단독 하락이 함께 나타났다.
- 새3회 및 재사용12회 CPU재채점·finite·native gate·가중치 보존·원본/고정PTH 및 YAML 해시·epoch/frame·지면 높이/거리 검증을 통과했다. seed별 초기12필드가모든비교모델과bitwise일치했다. GPU3peak3개·최종0개이며GPU0·1·GPU6 본학습 및 표준 코드는 건드리지 않았다. [검증](output_etc/check/ground17000_trend512_gpu3_20261104/verification.json).

### zerobg8k·9k · 일반512환경×3회 · ground 동일 epoch 비교

- 사용자 확인으로8k·9k 두 PTH만 각512환경×seed0/1/2, 총6회를 새로 평가했다. GPU4의worker0/1/2는8k, GPU3의worker3/4/5는9k이며 GPU당peak3개·최종0개다. 4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode를 유지했다. 기존ground8k/9k의6회는 재사용했다. [비교표](output_etc/check/zerobg_trend512_gpu34_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- zerobg8k→9k 공유 중심20cm 공동 전체 Ever84.20→81.14%, 단독 PUSH82.44→87.19%, 나머지4행동 영역 평균92.02→91.88%다. 동일 epoch ground는 공동82.67/85.31%, PUSH76.48/79.26%, 나머지영역91.65/91.61%다. 공동Final은zero82.08/78.38%, ground78.98/83.39%다. 앞선64환경 진단과 달리512×3에서는 zero의 나머지 영역 평균 우위가 작고 단독 PUSH가 높다.
- CPU 재채점·JSON건수·finite·native gate·actor/critic zerobg/SELF·NONE0·가중치 불변·원본/고정PTH 및 YAML 해시·epoch/frame·지면 높이/거리 검증을 통과했다. 각seed 초기12필드가두zero PTH 및기존ground8k/9k와bitwise일치했다. [검증](output_etc/check/zerobg_trend512_gpu34_20261104/verification.json). GPU0·1·GPU6 본학습과 표준 코드는 변경하지 않았다.

### learnedbg5k·zerobg8k · 6/8명 all/keep3

- 사용자 요청으로 각 PTH를 복사 고정하고 6명/12물체·8명/16물체 × all/keep3, 각64환경×seed0 총8회를 GPU3·5에서 평가했다. ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode를 유지했고 GPU0·1과 본학습은 건드리지 않았다. [통합 비교표](output_etc/check/zerobg8000_n68_all_keep3_env64_gpu35_20261104/comparison.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 공유 중심20cm Ever의 all→keep3는 learnedbg5k에서 6명9.38→23.44%, 8명17.19→17.19%; zerobg8k에서 6명23.44→23.44%, 8명28.12→28.12%다. Final은 각각 6명9.38→20.31/18.75→15.62%, 8명12.50→14.06/20.31→23.44%다. 이번 표본에서 keep3의 Ever 이득은 learnedbg6명에만 나타났다. epoch가 다른 모델·64환경1seed 비교다.
- 8회 CPU 재채점·finite·가중치 보존·PTH/YAML 해시·epoch/frame·초기12필드 일치·native gate 검증을 통과했다. 매 행동의 keep3 선택수/미접촉 우선/SELF 제외 및 actor4층 COUPLED→NONE(마스킹 없음)·critic 무개입 검사를 통과했다. 두 평가 묶음 모두 종료했다. [learnedbg 검증](output_etc/check/learnedbg5000_n68_all_keep3_env64_gpu35_20261104/verification.json), [zerobg 검증](output_etc/check/zerobg8000_n68_all_keep3_env64_gpu35_20261104/verification.json).

### 선택 PTH 6/8명 · all/keep3 NONE 전환 비교

- 일반 평가 후 공동 최고 ground9.5k·회복/단독 균형 ground16k·fusion 공동 최고 zerobg3k를 선택해 각각6명/12물체·8명/16물체64환경×seed0·ordered·loco·평균 액션·원래 인원별 상자 크기/무게·distractor·1200step 첫 episode로 all/keep3 총12회를 평가했다. GPU3/2/5에서각모델을실행했고GPU0·1은사용하지않았다. ground5k의기존2회는재사용했다. [Ever/Final·접촉·이동·분모](output_etc/check/ground_relfusion_selected_n68_all_keep3_env64_gpu35_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 공유 중심20cm Ever의all→keep3는 ground9.5k에서6명7.81→25.00%,8명15.62→14.06%;ground16k에서6명10.94→17.19%,8명29.69→28.12%;zerobg3k에서6명14.06→12.50%,8명17.19→10.94%다. Final은각각6명6.25→23.44/10.94→15.62/12.50→10.94%,8명15.62→10.94/21.88→21.88/9.38→7.81%다. ground6명에서는개선됐지만8명은세후보모두Ever개선이없다.64환경1seed진단이다.
- keep3는미접촉상대를우선해3명을유지하며선택밖COUPLED를actor4층에서NONE으로전환한다. 매행동의선택수·미접촉우선·SELF제외·실제4층COUPLED/NONE수·attention padding 없음·critic개입없음4800회검사와저장선택trace검증을통과했다. 실제그룹·보상gate·물성은유지했다.
- 새12회와재사용2회CPU/GPU재채점·JSON건수·finite·native gate·가중치불변·설정·PTH/YAML해시·epoch/frame을확인했다. 같은인원의초기12필드가모든후보/all/keep3에서bitwise일치했다. GPU별peak2개·최종0개다. [검증](output_etc/check/ground_relfusion_selected_n68_all_keep3_env64_gpu35_20261104/verification.json). 폴더명의gpu35와달리실제GPU는2·3·5이며manifest/검증에기록했다. 본학습·표준코드는변경하지않았다.

### ground12k·14k·16k / zerobg8k·learnedbg5k까지 일반 추세

- 사용자 요청으로 GPU0·1을 제외하고 ground12k/14k/16k를 GPU2·3에서각512환경×seed0/1/2, zerobg4k/5k/6k/8k와 learnedbg3k/4k/5k를 GPU5에서각64환경×seed0로 평가했다.10개 정기PTH/YAML을 복사 고정했고4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode를 유지했다. 기존ground27회·fusion12회를 재사용했다. [ground 추세](output_etc/check/ground_12k14k16k_eval512_gpu23_20261104/report.md), [fusion 추세](output_etc/check/relfusion_ab_3k8k_eval64_gpu5_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- ground12k/14k/16k 공유 중심20cm 전체 Ever84.08/82.08/85.85%,단독 점 평균86.62/86.23/87.66%,PUSH76.47/76.44/80.74%다.16k는9.5k의공동86.96%에는못미치지만단독점평균85.07%보다높아균형후보로추가했다. 공동전체 Final82.64/80.36/83.45%다.
- zerobg4k/5k/6k/8k 공동전체 Ever78.125/68.75/79.6875/81.25%,learnedbg3k/4k/5k79.6875/62.5/62.5%다. zerobg3k의기존57/64(89.0625%)가fusion내공동최고다.64환경1seed로구조 우열이나512×3 ground와직접우열은판단하지않는다.
- 새16회의CPU/GPU재채점·JSON건수·finite·native gate·actor/critic모드·가중치불변·PTH/YAML해시·epoch/frame·ground높이/거리검증을통과했다. 평가규모별초기12필드가모든비교저장본에서bitwise일치했다.일반평가GPU2/3각peak3개·GPU5peak2개,모두종료했다. [ground 검증](output_etc/check/ground_12k14k16k_eval512_gpu23_20261104/verification.json), [fusion 검증](output_etc/check/relfusion_ab_3k8k_eval64_gpu5_20261104/verification.json). 본학습·표준코드는변경하지않았다.

### ground11k·11.5k / fusion A/B 최신 추세 · GPU0·1 제외

- 사용자 요청으로 GPU0·1을 제외하고 GPU2에서ground11k/frame1441792000·11.5k/frame1507328000을각512환경×seed0/1/2, GPU5에서zerobg2.5k/3k/3.5k·learnedbg2k/2.5k를각64환경×seed0로 새로 평가했다.7개 정기PTH/YAML을 복사 고정했고4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode를 유지했다. 기존ground21회·fusion7회를 재사용했다. [ground 전체 추세](output_etc/check/ground_11k11500_eval512_gpu2_20261104/report.md), [fusion 전체 추세](output_etc/check/relfusion_ab_2k3500_eval64_gpu5_20261104/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- ground11k/11.5k 공유 중심20cm 전체 Ever82.16/83.54%,4명81.79/83.58%,단독 점 평균85.29/83.88%,PUSH67.66/73.09%,AT84.89/85.58%다. 공동 전체 Final80.30/82.38%다.9.5k의공동86.96%·PUSH81.04%보다낮으며 AT는회복했다.
- zerobg2.5k/3k/3.5k 공동 전체 Ever78.1/89.1/75.0%,단독 점 평균88.4/83.8/84.7%,PUSH82.1/71.4/75.0%다. learnedbg2k/2.5k 공동전체65.6/65.6%,단독점평균82.4/83.8%,PUSH57.1/71.4%다.같은2.5k에서zerobg가공동·단독평균·PUSH에앞섰다. zerobg3k공동은57/64이며소표본1seed로512×3 ground와직접우열을판단하지않는다. 공동전체 Final은zero78.1/84.4/73.4%,learn60.9/65.6%다.
- 새11회의CPU/GPU재채점·JSON건수·finite·native gate·실제actor/critic모드·가중치불변·PTH/YAML해시·epoch/frame·ground높이/거리검증을통과했다. 기존결과검증·해시와평가규모별초기12필드bitwise일치도확인했다. GPU2peak3개·GPU5peak2개,최종평가0개다. [ground 검증](output_etc/check/ground_11k11500_eval512_gpu2_20261104/verification.json), [fusion 검증](output_etc/check/relfusion_ab_2k3500_eval64_gpu5_20261104/verification.json). GPU0·1·본학습·표준코드는변경하지않았다.

## 2026-11-03

### ground8k·10k · 512환경×3seed · 5k–10k 비교

- 사용자 요청으로 ground8000epoch/frame1048576000·10000epoch/frame1310720000 PTH/YAML을 복사 고정해 GPU3에서각512환경×seed0/1/2를 평가했다.4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode이며 worker3개가 각8k→10k를 순차 실행했다. 기존5k/6k/7k/9k/9.5k의15회는 재사용했다. [Ever/Final·편차·분모](output_etc/check/ground_8k10k_eval512_gpu3_20261103/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 공동 공유 중심20cm 전체 Ever는5k→6k→7k→8k→9k→9.5k→10k에서77.99→81.78→83.42→82.67→85.31→86.96→82.03%다.8k의2/3/4명85.16/80.40/79.05%,10k82.86/81.31/81.02%다.10k는9.5k보다 세seed 모두 공동 전체가 낮고,현재 평가 저장본 중9.5k가 공동 전체·각 인원과 단독 PUSH에서 가장 높다.
- 8k/10k 단독 점 평균86.04/84.46%,AT80.49/77.98%,ON_TOP85.86/88.11 및80.50/86.92%,PUSH76.48/79.82%다. 공동 전체 Final78.98/80.37%다.10k는9.5k보다AT·ON_TOP 점·PUSH·단독 점 평균도 낮지만SIT 점은 회복했다.
- 새6회의 CPU/GPU 재채점·JSON 건수·finite·native gate·기본attention/fusion 없음·가중치 불변·PTH/YAML 해시·epoch/frame·ground 높이/거리 검증을 통과했다. 기존15회 검증 기록과해시를 확인하고 각seed 초기12필드가 모든epoch에서bitwise 일치함을 확인했다. 평가peak3개·최종0개다. [검증](output_etc/check/ground_8k10k_eval512_gpu3_20261103/verification.json). 본학습·표준 코드·GPU0은 변경하지 않았다.

### learnedbg1.5k · zerobg 동일 epoch 추세 비교

- learnedbg1500epoch/frame196608000 PTH/YAML을 복사 고정해 GPU3에서4명·8물체 혼합64환경×seed0·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode1회를 평가했다. 이전 learnedbg0.5k/1k와 zerobg0.5k/1k/1.5k/2k 여섯 결과는 재사용했다. [Ever/Final·분모·추세](output_etc/check/relfusion_learnedbg1500_ab_eval64_gpu3_20261103/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- learnedbg1k→1.5k 공유 중심20cm 전체 Ever73.4→64.1%(41/64),2/3/4명83.3/46.2/66.7→71.4/61.5/33.3%,단독 점 평균79.9→79.2%,PUSH53.6→67.9%,AT82.8→79.3%,ON_TOP83.3/83.3→88.9/94.4%다. 같은1.5k zerobg는 공동 전체76.6%,단독 점 평균85.0%,PUSH64.3%,ON_TOP77.8/77.8%다. learnedbg1.5k 공동 전체 Final62.5%,ON_TOP77.8/88.9%다. 이번 소표본에서는 zerobg가 공동 전체와 단독 평균에서 앞서며 learnedbg는ON_TOP이 강하지만 구조 우열을 확정하지 않는다.
- CPU/GPU 재채점·JSON 건수·finite·native gate·actor/critic learnedbg 구성·가중치 불변·PTH/YAML 해시·epoch/frame·ground 높이/거리와 일곱 저장본 초기12필드bitwise 일치를 확인했다. [검증](output_etc/check/relfusion_learnedbg1500_ab_eval64_gpu3_20261103/verification.json). 본학습·표준 코드·GPU0은 변경하지 않았다.

### zerobg2k · 일반64환경 추세 추가

- zerobg2000epoch/frame262144000 PTH/YAML을 복사 고정해 GPU3에서4명·8물체 혼합64환경×seed0·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode1회를 평가했다. 이전0.5k/1k/1.5k 결과는 재사용했다. [Ever/Final·분모](output_etc/check/relfusion_zerobg2000_eval64_gpu3_20261103/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 1.5k→2k 공유 중심20cm 전체 Ever76.6→76.6%(49/64),2명81.0→85.7%,3명76.9→61.5%,4명55.6→55.6%다. 단독 AT93.1→96.6%,ON_TOP77.8/77.8→83.3/94.4%,PUSH64.3→67.9%,점 평균85.0→85.9%다.2k 공동 전체 Final76.6%,ON_TOP50.0/66.7%,PUSH64.3%다. 공동3명13그룹·4명9그룹 소표본이다.
- CPU/GPU 재채점·JSON 건수·finite·native gate·actor/critic zerobg 및 SELF/NONE0·가중치 불변·PTH/YAML 해시·epoch/frame·ground 높이/거리와 이전 세 저장본 대비 초기12필드bitwise 일치를 확인했다. [검증](output_etc/check/relfusion_zerobg2000_eval64_gpu3_20261103/verification.json). 본학습·표준 코드·GPU0은 변경하지 않았다.

### 최신 ground9.5k · zerobg1.5k · learnedbg1k 평가

- 요청 시점의 최신 정기 PTH/YAML을 복사 고정해 ground9500은 GPU3에서512환경×seed0/1/2, zerobg1500·learnedbg1000은 GPU2에서각64환경×seed0로 평가했다. 모두4명·8물체 혼합·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode다. 이전 ground12회와 fusion3회는 재사용했다. [ground 추세](output_etc/check/ground_latest9500_eval512_gpu3_20261103/report.md), [fusion 추세](output_etc/check/relfusion_ab_trend64_1000_1500_20261103/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- ground9k→9.5k 공유 중심20cm 전체 Ever85.31→86.96%,4명84.45→88.02%,단독 PUSH79.26→81.04%,단독 점 평균85.90→85.07%다. AT83.07→80.98%,ON_TOP84.63/86.71→82.62/85.32%로 내려갔다. 공동 전체 Final83.39→84.82%다.
- zerobg1k→1.5k 공동 전체 Ever59.4→76.6%,2/3/4명81.0/76.9/55.6%,단독 점 평균78.9→85.0%,PUSH71.4→64.3%다. learnedbg0.5k→1k 공동 전체64.1→73.4%,2/3/4명83.3/46.2/66.7%,단독 점 평균75.4→79.9%,PUSH64.3→53.6%다. 최신 공동 전체 Final75.0/73.4%다. fusion4명은9그룹 소표본이며512×3 ground와 직접 구조 우열을 판단하지 않는다.
- 새5회 CPU/GPU 재채점·JSON 건수·finite·native gate·actor/critic 모드·가중치 불변·PTH/YAML 해시·epoch/frame 및 ground 높이/거리 검증을 통과했다. 각 평가 규모 안에서 이전 결과와 초기12필드가bitwise 일치했다. [ground 검증](output_etc/check/ground_latest9500_eval512_gpu3_20261103/verification.json), [fusion 검증](output_etc/check/relfusion_ab_trend64_1000_1500_20261103/verification.json). 본학습·표준 코드·GPU0은 변경하지 않았다.

### ground 최신9k · 512환경×3seed · 5k/6k/7k 추세 확장

- 사용자 요청 시점 최신 정기 ground9000epoch/frame1179648000 PTH/YAML을 복사 고정하고 GPU3에서4명·8물체 혼합512환경×seed0/1/2·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode3회를 새로 평가했다. 기존5k·6k·7k각3회는 재사용했다. [Ever/Final·편차·분모](output_etc/check/ground_latest9k_eval512_gpu3_20261103/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 5k→6k→7k→9k 공유 중심20cm 전체 Ever77.99→81.78→83.42→85.31%,4명74.48→81.82→83.03→84.45%,단독 PUSH72.27→75.57→73.35→79.26%,단독 점 평균86.34→86.55→84.06→85.90%다.9k 공동2/3명85.64/85.31%,AT83.07%,ON_TOP84.63/86.71%,SIT85.81/98.27%이며 공동 Final전체83.39%,전원 개인 목표 동시 Ever전체81.75%다. 공동은 개선 추세,단독 PUSH/SIT는7k대비 회복했지만AT는5k이후 하락했다.
- 새3회 CPU/GPU 재채점·JSON 건수·finite·native gate·기본attention/fusion 없음·가중치 불변·원본/고정PTH와 YAML 해시·epoch/frame을 검증했다. 기존9회 검증 기록과PTH/YAML 해시를 확인했고 각seed 초기12필드는5k/6k/7k/9k 모두bitwise 일치했다. ground 운반 높이/거리도 통과했고 평가peak3개·최종0개다. [검증](output_etc/check/ground_latest9k_eval512_gpu3_20261103/verification.json). GPU0·본학습·표준 코드는 변경하지 않았다.

### 관계 fusion · zerobg1k / learnedbg0.5k · 일반64환경

- 사용자 요청으로 zerobg1000epoch/frame131072000·learnedbg500epoch/frame65536000 PTH/YAML을 복사 고정하고 GPU3에서4명·8물체 혼합64환경×seed0·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode2회를 평가했다. 기존 zerobg500 결과를 재사용했다. 세 모델 모두ground이고 초기12필드가bitwise 일치했다. [Ever/Final·분모](output_etc/check/relfusion_ab_trend64_500_1000_20261103/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- zerobg0.5k→1k 공유 중심20cm 전체 Ever59.4→59.4%,2/3/4명66.7/53.8/33.3→64.3/53.8/44.4%,단독 점 평균74.0→78.9%,PUSH75.0→71.4%다. learnedbg0.5k는 공동2/3/4명73.8/69.2/11.1%,전체64.1%,단독 점 평균75.4%,PUSH64.3%다. 공동4명은9그룹 소표본이며 구조 우열을 확정하지 않는다.
- 새2회 CPU/GPU 재채점·JSON 건수·finite·native gate·실제actor/critic fusion 및 A/B 배경 bias 구성·가중치 불변·원본/고정PTH와 YAML 해시·epoch/frame을 확인했다. ground 운반 높이/거리와 재사용500 결과/고정PTH 해시도 검증했다. [검증](output_etc/check/relfusion_ab_trend64_500_1000_20261103/verification.json). 본학습·표준 코드·GPU0은 변경하지 않았다.

### zerobg500 · 일반64환경 · 기존500 비교

- 사용자 요청으로 ground 관계 fusion B의500epoch/frame65536000 PTH와 YAML을 복사 고정하고 GPU2에서4명·8물체 혼합64환경×seed0·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode1회를 평가했다. 비교용 기존 solo5·CO norm·CO+NONE norm500의 같은 규모 결과3개는 재측정 없이 재사용했다. [Ever/Final·분모·비교](output_etc/check/relfusion_zerobg500_eval64_gpu2_20261103/report.md), [재현](markdowns/USAGE.md#공통-stage2-push와-coupled).
- zerobg 공유 중심20cm Ever2/3/4명은28/42·7/13·3/9(66.7/53.8/33.3%),전체38/64(59.4%)이고 Final도 같다. 단독 Ever SIT40/80·CLIMB100/100·AT82.8·ON_TOP72.2/77.8·PUSH75.0%,점 평균74.0%다. 기존 solo5/CO/CO+NONE의 공동 전체 Ever64.1/60.9/70.3%,단독 점 평균75.4/78.6/78.4%다.64환경1seed 진단이며 zerobg만ground 운반 높이/거리 설정이므로 구조만의 비교가 아니다.
- 신규 CPU/GPU 재채점·JSON 건수·finite·native gate·양쪽zerobg/SELF·NONE0·가중치 불변·원본/고정PTH와 YAML 해시·500epoch/frame을 확인했다. 기존3개의 재채점 검증 및 고정PTH 해시를 확인했고 ground AT/ON_TOP 출발 지면·목표 높이·거리1–5m도 통과했다. [검증](output_etc/check/relfusion_zerobg500_eval64_gpu2_20261103/verification.json). 본학습·표준 코드는 변경하지 않았다.

### ground 관계 fusion A/B 별도 컨피그

- 사용자 승인 설계로 ground 기반 `stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg`(A)·`_relfusion_zerobg`(B)의 env/train YAML 및 train/test/VNC/eval wrapper를 추가했다. 두 모델은 COUPLED64차원 임베딩을 레이어/헤드 bias 투영과 공유128→64→64 residual fusion에 함께 쓴다. A만 SELF/NONE lookup을 학습하고 B는0 고정한다. fusion은 attention 집계 후 output projection 전에 사람·레이어당1회 적용하며 단독 추가분은0이다. Pre-LN·환경·보상·물성·Stage1 PUSH26k 전이·actor encoder 동결은 유지한다. BEFORE는 이번 추가 범위에서 제외했다. [명령](markdowns/config.md), [설계](markdowns/reference/stage2_coupled_push.md#ground-관계-fusion-ab).
- 기존 coordination 경로·state_dict를 보존하고 새 경로를 별도 모듈에 추가했다. 실험 등록·평가·checkpoint 계약·문서를 연결했다. CPU73개와 설정/저장 환경40개를 통과했다. 첫 CPU 실행의 기존21k 전이 테스트1개는 해당 파일 부재로 실패했으며 관련 없는 이1개를 제외해 재실행했다. 새 A/B의 실제26k 전이·초기 액션 보존·공통 초기 가중치 일치·계산/gradient 동치·인원/셔플·단독0·gradient 발생·strict reload는 통과했다.
- GPU2에서 각각4명·1024환경·8물체로 MAX_ITERATIONS=2 확인 학습을 완료했다(epoch3/frame393216). 저장본의 actor encoder 원본 보존·action head 및 양쪽 fusion/bias 투영 갱신·모델/각263개 scalar tag finite를 확인했다. 각32환경×8preset×16step 재로드의 actor/critic/AMP/보상 finite·부분 reset·가중치 불변을 통과했고 실제 test wrapper도8환경×16step으로 완료했다. GPU0 및 기존 학습 프로세스는 변경하지 않았고 본학습·정규 성능 평가·GUI/VNC는 실행하지 않았다. [검증](output_etc/check/relfusion_ab_20261103/verification.json), [검증 재현](markdowns/USAGE.md#공통-stage2-push와-coupled).

### ground5k · 6k · 7k · 512환경×3seed 추세

- 사용자 요청으로 ground6k/frame786432000·7k/frame917504000 PTH/YAML을 복사 고정하고 GPU5/6에서4명·8물체 혼합512환경×seed0/1/2·ordered·loco·평균 액션·원래 물성/distractor·1200step6회를 새로 평가했다.6k3회 완료 후 사용자가 알려준7k를 같은 조건으로 이어 평가했고, 기존5k3회는 재사용했다. 기존 학습·표준 코드·설정은 유지했다. [Ever/Final·편차·분모](output_etc/check/ground_5k6k7k_eval512_gpu56_20261103/report.md), [명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 5k→6k→7k 공유 중심20cm 전체 Ever77.99→81.78→83.42%,4명74.48→81.82→83.03%,전원 개인 목표 동시 전체73.93→77.73→80.05%다. 단독5행동 점 평균86.34→86.55→84.06%,단독 PUSH72.27→75.57→73.35%,AT86.65→85.75→84.17%,ON_TOP 점85.09→85.10→84.26%다. 참고 공유 중심 전체 Final75.79→79.38→81.01%다. 각3seed 평균이며7k 단독 평균 하락은SIT91.14→89.36→80.93%의 영향이 크다.
- 새6회 CPU/GPU 재채점·JSON·finite·native gate·실제 기본attention 모드·가중치 불변·고정/원본PTH와 YAML 해시·epoch/frame을 확인했다. 지면 배치·ON_TOP 받침·AT/ON_TOP 운반거리1–5m와5k대비 각seed 초기12필드bitwise 일치도 통과했다. 재사용5k의 검증 기록·PTH/YAML 해시를 확인했고 평가 peak3개·최종0개다. [검증](output_etc/check/ground_5k6k7k_eval512_gpu56_20261103/verification.json).
- 사용자 요청으로 단독 PUSH만 기존9개trace를CPU float64로20/30/40/50cm 재집계했다.6k Ever75.57/82.42/86.88/89.80%,Final73.90/81.02/85.76/88.55%다. 각epoch698명이며20cm Ever/Final의seed별 성공 수·분모가 원래 집계와 정확히 일치하고 반경 포함 관계·Final→Ever도 통과했다.7k는 모든 반경에서6k보다 낮다. 추가 GPU/시뮬레이션 및 기본 평가·학습 보상 변경은 없다. [전체 결과·건수·CPU 명령](output_etc/check/ground_5k6k7k_eval512_gpu56_20261103/solo_push_radius/report.md).

### 기존 solo5 · nonenorm 동일10k · 512환경×3seed

- 사용자 요청으로 nonenorm10000epoch/frame1310720000 PTH/YAML을 복사 고정하고 GPU5/6에서4명·8물체 혼합512환경×seed0/1/2·ordered·loco·평균 액션·원래 물성/distractor·1200step3회를 새로 평가했다. 기존 solo5 같은10k의 같은 조건3회는 재사용했다. 기존 학습과 표준 코드는 유지했다. [Ever/Final·편차·분모·출처](output_etc/check/nonenorm_vs_solo5_epoch10000_512_gpu56_20261103/report.md), [명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 기존/nonenorm 순서 공유 중심20cm 전체 Ever85.31±1.07/77.97±1.61%,4명89.53±1.93/78.94±3.43%,전원 개인 목표 동시 전체82.43/71.40%다. 단독5행동 점 평균85.12/84.55%,단독 PUSH79.51/81.66%이고 참고 공유 중심 전체 Final83.25/75.09%다. 같은epoch·동일초기장면에서 두 저장본을 비교한3seed 평균±표본 표준편차다.
- 새3회 CPU/GPU 재채점·JSON·finite·native gate·실제actor/critic CO+NONE 보정·가중치 불변·고정/원본PTH와 YAML 해시·epoch/frame을 확인했다. 기존3회의 검증 기록과PTH/YAML 해시도 확인했고, 각seed의 초기12필드가 모델 사이 모두bitwise 일치했다. GPU 평가 peak3개(5에1개·6에2개)·최종0개다. [검증](output_etc/check/nonenorm_vs_solo5_epoch10000_512_gpu56_20261103/verification.json).

### ground5k · nonenorm13k · 6/8명 공동64환경

- 사용자 요청으로 직전 일반 평가에 고정한 ground5k·nonenorm13k PTH/YAML을 재사용해6명/12물체·8명/16물체 전원 공동64환경×seed0·ordered·loco·평균 액션·인원별 원래 크기/무게·distractor·1200step4회를 새로 평가했다. GPU5에서ground, GPU6에서nonenorm을 각각 순차 실행했고 기존 학습은 유지했다. 가까운 기존 solo5 6k·14k의 같은 조건4회를 재사용했다. [Ever/Final·건수·출처](output_etc/check/ground_nonenorm_n68_env64_gpu56_20261103/report.md), [명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 공유 중심20cm Ever의6명/8명은 기존6k9.38/7.81%,ground5k18.75/10.94%,기존14k15.62/15.62%,nonenorm13k10.94/23.44%다. 참고 Final은 각각7.81/4.69,10.94/10.94,15.62/12.50,9.38/15.62%다. 같은 인원의 초기12필드가8회 모두bitwise 일치했다. 서로 다른epoch·64환경1seed 진단이며 구조의 우열을 확정하지 않는다.
- 8회 CPU/GPU 재채점 기록·JSON 건수·finite·가중치 불변·native gate·PTH/YAML 해시와 epoch를 확인했다. 새 GPU 평가 peak2개·최종0개다. 첫 CPU 집계는 GPU 번호와runtime이 기록한UUID를 직접 비교해 멈췄으며, 실제 GPU5/6 UUID를 대조하도록 검사만 수정한 뒤 통과했다. 추가 시뮬레이션은 없다. [검증](output_etc/check/ground_nonenorm_n68_env64_gpu56_20261103/verification.json). 표준 코드·학습 설정은 변경하지 않았다.

### ground5k · nonenorm13k · GPU5 일반512환경×3seed

- 사용자 요청으로 각 최신 저장본 ground5000epoch/frame655360000·CO+NONE norm13000epoch/frame1703936000 PTH/YAML을 복사 고정했다. GPU5만 사용해4명·8물체 혼합512환경×seed0/1/2·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode6회를 새로 평가했다. 평가 worker2개가 각3회를 순차 실행했고 기존 본학습은 유지했다. [Ever/Final·편차·건수](output_etc/check/ground_nonenorm_latest512_gpu5_20261103/report.md), [고정 PTH·구성](output_etc/check/ground_nonenorm_latest512_gpu5_20261103/manifest.json), [재현 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- ground/nonenorm 순서 공유 중심20cm 전체 Ever는77.99±2.77/82.36±1.75%,4명은74.48±5.33/74.83±3.31%다. 단독 AT는86.65/81.35%,ON_TOP 점/영역은85.09/89.86 및78.73/81.90%,단독5행동 점 평균86.34/82.23%다. 참고 공유 중심 전체 Final75.79/79.42%,전원 개인 목표 동시 Ever73.93/76.38%,낙상112/113명(각6144명)이다. 평균±3seed 표본 표준편차다.
- 6회 CPU/GPU 재채점·JSON·finite·native gate·실제 actor/critic 보정 모드·가중치 보존·원본/고정 PTH/YAML 해시를 확인했다. ground의 AT/ON_TOP 원본·AT 목표 지면,ON_TOP 받침,운반 거리1–5m도 실제 저장 초기 상태에서 확인했다. GPU5 평가 peak2개·최종0개를 로그로 확인했다. [검증](output_etc/check/ground_nonenorm_latest512_gpu5_20261103/verification.json). 학습량과 높이/목표 분포가 다른 최신 상태 비교이며 동일 장면의 구조 비교가 아니다. 표준 코드·학습 설정 변경은 없다.
- 기존 비교 요청으로 같은512환경×3seed·ordered 평가 중 가장 가까운 solo5 8k·14k 결과를 보고서에 추가했다. 원본 집계와6개 seed의 기존 검증 기록을 확인해 재사용했으며 추가 GPU 실행은 없다. 기존5k·13k의 같은 규모 결과는 없으므로 학습량 차이를 명시했다. [기존 포함 집계·출처](output_etc/check/ground_nonenorm_latest512_gpu5_20261103/with_solo5_summary.json).

### carry5_ground 별도 컨피그 · 운반 선반 제거

- 사용자 요청으로 `stage2_coupled_push_individual_goal_solo5_carry5_ground` 환경/학습 YAML·train/test/VNC/eval을 추가했다. carry5에서 `platformMode=ground_only`, `randomHeight=false`, `randomHeightProb=0`와 실험 이름만 바꿨다. AT/ON_TOP 운반 원본과 AT 목표는 지면에 두고 ON_TOP의 실제 받침은 유지한다. 거리1–5m·정책·보상·물성·확률·Stage1 PUSH26k 전이는 같다. ground_only 사용 시 출발5cm clearance도 추가하지 않는다. [명령](markdowns/config.md), [설정 상세](markdowns/reference/stage2_coupled_push.md#carry5-지면-운반-변형).
- 공통 platform 검증/고정 actor/ground 플래그를 coupled에도 연결했다. 출발/목표 플랫폼은 기존 ground_only 기본값인 Z−20m/−19.5m에 격리된다. 새 이름을 실험·launcher·일반 평가에 등록하고 config 목록 끝에 명령을 추가했으며 구조/USAGE/README를 갱신했다. 기존 YAML·학습 프로세스와 다른 checkout은 변경하지 않았다.
- CPU58개 통과(요청과 무관한78k 전이3개 제외). GPU6에서4명·1024환경·8물체 CHECK 학습과26k 전이/저장(epoch2,frame262144)·finite를 확인했다.1024환경×10조건 총10,240장면의 거리·충돌/경로·지면 배치·ON_TOP 받침 보존을 통과했다. 첫 검사는 목표 플랫폼도−20m라는 검사 가정 때문에 실패했고, 실제 기존 기본값−19.5m에 맞춰 검사를 정정한 뒤 재검증했다. 실패 로그를 보존했다.
- GPU6·32환경×8preset×16step 저장본 재로드에서 actor/critic·AMP·보상 finite·가중치 보존·부분 reset을 확인했다. [검증](output_etc/check/carry5_ground_20261103/verification.json), [배치](output_etc/check/carry5_ground_20261103/stress/stress_validation.json), [정책](output_etc/check/carry5_ground_20261103/policy/physics_validation.json). 본학습·정규 성능 평가·GUI/VNC는 실행하지 않았다.

### solo5 20k · GPU5 단독 AT 선반 제거64환경 비교

- 사용자 요청으로 기존64환경 평가 중 가장 뒤의20k PTH를 고정하고 GPU5에서4명/8물체 혼합64환경×seed0·ordered·loco·평균 액션·원래 물성/distractor·1200step의 기존/AT 선반 제거2회를 순차 평가했다. AT만 원본 상자/목표를 지면 기준으로 낮추고 해당 플랫폼을Z−20m로 치웠다. 원본 상자5cm clearance와 높이 난수 소비·기존 충돌/배치 재시도·다른 과제는 유지했다. 학습 설정/표준 코드 변경은 없다. [결과·정의](output_etc/check/solo5_20k_at_ground64_gpu5_20261103/report.md), [명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 같은29명에서 AT Ever는26→25명(89.66→86.21%),참고 Final25→23명(86.21→79.31%)이며 AT 낙상0→1명이다. Ever 신규 성공2명·성공 상실3명으로 이번 소표본에서 선반 제거의 개선을 확인하지 못했다. 기존 조건 재실행은 과거20k 성공 수와 동일했다.
- 두 조건 CPU/GPU 재채점·JSON·finite·native gate·모델/원본 및 고정 PTH/YAML 보존을 확인했다. 초기12필드가 AT 관련Z 이외에 bitwise 일치하고 플랫폼도AT Z만 변경됐으며 reset후CPU/CUDA RNG도 같았다. [검증](output_etc/check/solo5_20k_at_ground64_gpu5_20261103/verification.json).64환경 중 단독AT29명·1seed의 추론 조건 비교이며 선반 없는 재학습 효과나 BEFORE와의 격차 원인을 확정하는 실험은 아니다.

### solo5 1.5k · GPU5 일반512환경×3seed · carry5 비교

- 사용자 요청으로 기존 solo5의1500epoch/frame196608000 PTH/YAML을 고정하고 GPU5에서4명·8물체 혼합512환경×seed0/1/2·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode3회를 순차 평가했다. 검증된 carry5 같은1.5k의3회를 재사용했다. [Ever/Final·거리·차이](output_etc/check/solo5_epoch1500_vs_carry5_eval512_gpu5_20261103/comparison.md), [명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- solo5→carry5 단독 AT 점 Ever는78.70→69.56%,ON_TOP 점/영역은75.54/78.57→70.46/74.03%다. 공동 공유 중심20cm 전체는73.30→65.51%,4명은76.32→54.99%다. 세 seed의 과제·그룹·상자 크기·사람 root/DOF는 같지만 상자/목표 배치가 달라 동일 장면의 정책 비교가 아니다. 실제 AT/ON_TOP 평균 운반 거리는solo5의5.445/5.681m에서carry5의3.015/3.006m가 됐다.
- 신규3회 CPU/GPU 재채점·JSON·finite·native gate·실제 actor/critic 설정·모델/고정 및 원본 PTH/YAML 해시를 확인했다. [검증](output_etc/check/solo5_epoch1500_vs_carry5_eval512_gpu5_20261103/verification.json), [재사용·초기 상태 비교](output_etc/check/solo5_epoch1500_vs_carry5_eval512_gpu5_20261103/comparison_verification.json).
- 사용자 지정 별도 checkout의 BEFORE source5 virtual 저장 설정/구현을 읽기 전용으로 비교했다. BEFORE는ground_only,solo5/carry5는 운반 원본 및AT 목표를 각각50% 확률로 높인다. 단독 보상 YAML·크기/밀도 범위는 같지만 BEFORE 선행은 항상AT이고 공동 그룹은PUSH다. [설정·근거·해석 한계](output_etc/check/solo5_epoch1500_vs_carry5_eval512_gpu5_20261103/config_comparison.md). 다른 checkout/학습은 수정하지 않았다.

## 2026-11-02

### carry5 1.5k · GPU5 일반512환경×3seed

- 사용자 요청으로 carry51500epoch/frame196608000 PTH와 YAML을 복사 고정하고 GPU5에서4명·8물체 혼합512환경×seed0/1/2·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode3회를 순차 평가했다. 기존 학습과 표준 코드는 유지했다. [Ever/Final·편차·분모](output_etc/check/carry5_epoch1500_eval512_gpu5_20261102/report.md), [실행 설정](output_etc/check/carry5_epoch1500_eval512_gpu5_20261102/manifest.json), [명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 단독 SIT/CLIMB/AT/ON_TOP/PUSH 점 Ever는84.91/98.96/69.56/70.46/68.86%, 단독5행동 점 평균78.55±1.95%다. 공동2/3/4명 공유 중심20cm Ever는68.66±0.80/66.01±2.21/54.99±7.54%(778/421/257그룹), 전체65.51±0.73%, 참고 Final62.83±1.12%다. 전원 개인 목표 동시 Ever/Final은58.51±1.09/55.51±1.93%이고 낙상145/6144명이다. 평균±3seed 표본 표준편차이며 앞선1k64환경1회와 직접적인 추세 비교가 아니다.
- 3회 CPU/GPU 재채점·JSON 건수·finite·native gate·실제 actor/critic 기본 보정·모델/원본 및 고정 PTH/YAML 보존을 확인했다. 단독 AT568명·ON_TOP439명의 초기 운반 거리가 모두1–5m 범위였다. GPU5 평가 peak1개·최종0개를 실행 로그로 확인했다. [검증](output_etc/check/carry5_epoch1500_eval512_gpu5_20261102/verification.json). ordered 진단이므로 formal_report=false다.

### carry5 1k · GPU5 일반64환경

- 사용자 요청으로 carry5의 최신 정기 저장본1000epoch/frame131072000과 YAML을 복사 고정하고 GPU5에서4명·8물체·64환경×seed0·ordered·loco·평균 액션·원래 물성/distractor·1200step 첫 episode를 평가했다. 학습·표준 코드 변경은 없다. [단독/공동 Ever·Final](output_etc/check/carry5_latest_eval64_gpu5_20261102/report.md), [실행 설정](output_etc/check/carry5_latest_eval64_gpu5_20261102/manifest.json), [명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 단독 SIT/CLIMB/AT/ON_TOP/PUSH 점 Ever는80.00/100.00/79.31/61.11/71.43%다. 공동2/3/4명 공유 중심20cm Ever는90.48/69.23/44.44%(42/13/9그룹), 전체79.69%, 참고 Final78.12%다. 전원 개인 목표 동시 Ever/Final은70.31/67.19%이며 낙상6/256명이다.64환경1seed 소규모 진단으로 기존 solo5와 목표 분포가 다르다.
- CPU/GPU 재채점·JSON 건수·finite·native gate·실제 actor/critic 기본 보정·모델/원본 및 복사 PTH/YAML 보존을 확인했다. 초기 AT 운반 거리1.016–4.993m(29명),ON_TOP1.146–4.945m(18명)로 저장된1–5m 범위를 확인했다. [검증](output_etc/check/carry5_latest_eval64_gpu5_20261102/verification.json). 정식512환경×3seed 평가는 아니다.

### 동일6k · 전체 인원4명→8명 · 20cm 비교표

- 사용자 요청으로40cm 표를 제외하고 같은6k의4명→8명 화살표 비교를 만들었다. 기존4명 norm/keep 결과가6.5k/5.5k로 학습량이 달라 세6k의4명·8물체64환경×seed0·공통resetseed0·ordered·1200step3회를 GPU3(기존/keep1 순차)·GPU5(nonenorm)에서 새로 평가했다. 검증된8명·16물체64환경3회를 재사용했다. 두 인원 모두 공동 최대4명·기존 확률/물리/거리·loco·평균 액션·distractor를 유지했다. [공유 중심·전원 개인 목표·단독 Ever/Final 비교표](output_etc/check/coupled_6k_mixed4to8_cap4_env64_gpu35_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 기존/nonenorm/keep1 순서 공유 중심20cm 전체 Ever는88.33→76.52/80.00→71.21/81.67→77.27%, 참고 Final은86.67→71.97/78.33→67.42/78.33→72.73%다.4명 공동2/3/4명 분모31/18/11그룹·단독96명,8명56/50/26그룹·단독146명이다. 각64장면1회로,4명과8명 장면 표본/구성비가 달라 전체 평균 차이에는 그룹 구성 차이도 포함된다. [건수·분모](output_etc/check/coupled_6k_mixed4to8_cap4_env64_gpu35_20261102/summary.json).
- 신규4명3회의 CPU/GPU 재채점·JSON·finite·native gate·실제 actor/critic 보정·keep1 선택·PTH/YAML/모델 불변을 확인했다.4명 내 모델별 초기12필드와 reset 전후 CPU/CUDA RNG가 bitwise 일치했다.8명 원본 summary 해시·기존 검증도 확인했으며 표준 코드/학습 설정 변경은 없다. [검증](output_etc/check/coupled_6k_mixed4to8_cap4_env64_gpu35_20261102/verification.json), [구성·재사용 출처](output_etc/check/coupled_6k_mixed4to8_cap4_env64_gpu35_20261102/manifest.json).

### 동일6k · 기존/nonenorm/keep1 · 8명 혼합·공동 최대4명

- 사용자 요청으로 세6k PTH를 재사용해8명·16물체·64환경×seed0·1200step·ordered·loco·평균 액션·원래 크기/무게/목표·distractor를 평가했다. 평가 프로세스에서만 기존 random_scenario의 공동 인원 균등 추출 상한을min(4, 남은 인원)으로 제한했다. 공동 확률0.5·단독 행동 확률·순서·물체 부족 시 재샘플링은 유지했다. CPU에서N≤4 기존 그룹/RNG 일치와N8 최대4명·유효 leader를 확인했다. 표준 sampler·학습 설정은 변경하지 않았다. [전체 Ever/Final·구성·재현](output_etc/check/coupled_6k_mixed8_cap4_env64_gpu356_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 초기 실행에서 기존/nonenorm의 환경10 배치가 재시도 상한에 걸려 reset 직전 Python/NumPy/CPU/CUDA seed0를 공통 적용했다. GPU6 keep1은 상자/목표 배치 차이로 비교에서 제외하고 GPU3에서 재실행했다. 최종 기존/keep1은GPU3, nonenorm은GPU5이며 초기12필드 및 reset 전후 CPU/CUDA RNG가 세 모델에서 bitwise 일치했다. 최초 실패 및 비교 제외 결과는 덮어쓰지 않고 manifest.attempts와 별도 경로/로그에 보존했다. GPU 간 배치 차이의 근본 원인은 확정하지 않았다.
- 기존/nonenorm/keep1 순서 공동 전원 개인 목표 Ever는89/86/88그룹(67.42/65.15/66.67%), Final은86/81/80그룹이다. 공유 중심20cm Ever는101/94/102그룹(76.52/71.21/77.27%),40cm는105/107/111그룹(79.55/81.06/84.09%)이다. 공동2/3/4명56/50/26그룹, 단독146명(SIT17·CLIMB19·AT33·ON_TOP25·PUSH52), 공동366명이고 전체64장면 중 단독/공동 혼합54장면·복수 공동 그룹54장면이다. 단일 seed 진단이다. [성공 수·분모·분할](output_etc/check/coupled_6k_mixed8_cap4_env64_gpu356_20261102/summary.json).
- 최종3회 CPU/GPU 재채점·JSON·finite·native gate·공유 중심20cm 저장 마스크 재현/40cm 단조성·모델 불변·원본/고정 PTH 및 YAML 해시·keep1 최대1명/동일 그룹/미접촉 우선/선택 상대 접촉 후 교체를 확인했다. [검증](output_etc/check/coupled_6k_mixed8_cap4_env64_gpu356_20261102/verification.json), [구성·시도 기록](output_etc/check/coupled_6k_mixed8_cap4_env64_gpu356_20261102/manifest.json).

### solo5 10k·15k·19k · 공유 중심20cm→40cm CPU 재채점

- 사용자 요청으로 기존512환경×seed0/1/2·4명 혼합·ordered·1200step 저장 궤적9회를 CPU에서 재집계했다. 실제 공유 상자 중심↔공유 목표의3D 직선거리 기준만20cm에서40cm로 완화하고 기존 그룹 유효성·Ever/Final 규칙을 유지했다. 시뮬레이션·학습·PTH·표준 성공 조건 코드는 변경하지 않았으며 표준 기본값은20cm다. [전체 및 인원별 Ever/Final·20cm 비교·재현](output_etc/check/solo5_shared_center40_10k15k19k_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 10k/15k/19k 순서40cm 공유 중심 전체 Ever는90.74±0.75/88.81±0.62/89.85±1.24%, 참고 Final은89.30±0.92/87.78±0.67/88.41±1.32%다. 평균±3seed 표본 표준편차다. 각1,456그룹에서 Ever1321/1293/1308그룹으로20cm 대비79/79/68그룹이 추가 성공했다.40cm에서도10k 평균이 가장 높다. [성공 수·seed별·분모](output_etc/check/solo5_shared_center40_10k15k19k_20261102/summary.json).
- 원래20cm의 저장 Ever/Final 마스크·JSON·기존 summary 건수/평균을 재현했다.9회 전체 프레임의 NumPy/Torch20cm·40cm 판정 일치와 임계값 완화의 단조성, 원본 trace SHA256 보존을 확인했다. 개인 목표 전원 동시 성공 및 단독 지표는 재집계 대상이 아니다. [검증](output_etc/check/solo5_shared_center40_10k15k19k_20261102/verification.json), [출처·원본 해시](output_etc/check/solo5_shared_center40_10k15k19k_20261102/manifest.json).

### 기존 solo5 · nonenorm · keep1 동일6k · GPU4 6/8명 공동64환경

- 사용자 요청으로 세 모델을6000epoch/frame786432000으로 통일해 PTH/YAML을 고정했다. keep1의 새6k도 저장 파일·내부 epoch/frame·원본/복사본 해시를 확인했다. GPU4에서 각6명·12물체 및8명·16물체 전원 공동64환경×seed0·1200step 첫 episode6회를 새로 실행했다. ordered·loco·원래 인원별 상자 크기/무게·평균 액션·distractor·저장된 actor/critic 보정/상대 선택을 유지했다. 기존 학습을 보존하고2개 worker가 인원별 세 모델을 순차 실행해 peak2개·최종0개다. [전체 Ever/Final·건수·재현](output_etc/check/nonenorm_keep1_n68_env64_gpu4_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 기존/nonenorm/keep1 순서 전원 개인 목표 Ever는6명3/5/3그룹,8명2/4/6그룹이고 Final은6명3/5/2,8명1/2/3그룹이다(각64그룹). 공유 중심 Ever는6명6/10/7,8명5/11/9그룹이며 Final은6명5/10/6,8명3/8/5그룹이다. 단일 seed 소규모 진단이며 개인 성공·낙상도 별도 기록했다. [성공 수·분모](output_etc/check/nonenorm_keep1_n68_env64_gpu4_20261102/summary.json).
- 6회 CPU/GPU 재채점·JSON 건수·finite·모델 불변·PTH/YAML 보존·실제 actor/critic 설정·전원 접촉→gate를 확인했다. 인원별 세 모델 초기12필드·CUDA RNG가 bitwise 일치하고 keep1만 CPU 모델 초기화 RNG 차이를 기록했다. keep1 궤적의 최대1명·동일 그룹·미접촉 우선·선택 상대 접촉 후 교체도 통과했다. 표준 코드·학습 설정 변경은 없다. [검증](output_etc/check/nonenorm_keep1_n68_env64_gpu4_20261102/verification.json), [고정 PTH·구성](output_etc/check/nonenorm_keep1_n68_env64_gpu4_20261102/manifest.json).

### nonenorm 6.5k · keep1 5.5k · GPU4 일반512환경×3seed

- 사용자 지정 일반 평가로 두 최신 정기 PTH를6500/5500epoch(frame851968000/720896000) 및 각 YAML과 함께 복사 고정했다. GPU4만 사용해4명·8물체 혼합512환경×seed0/1/2·ordered·loco·저장된 목표/거리/크기/무게·평균 액션·distractor·1200step 첫 episode6회를 실행했다. 기존 학습을 보존하고2개 worker가 각3회를 순차 실행했으며 로그상 peak2개·최종0개다. AT9종 크기 변경·6/8명 확장은 없다. [전체 표·seed 편차·재현](output_etc/check/nonenorm_keep1_eval_gpu4_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- nonenorm/keep1 공동 전체 Ever는72.48±1.20/71.32±2.43%, 공동4명은62.96±9.05/67.87±5.72%, 단독5행동 점 평균은82.37±1.85/81.80±1.91%다. 공유 중심 전체 Ever는77.96/77.16%, 참고 공동 Final은70.21/68.29%다. 같은512×3seed 기존 solo5 10k 결과를 재사용해 공동 전체82.43±1.60%, 단독 점 평균85.12±0.65%를 기준점으로 표시했다. 학습량이 다른 비교이며 평균±3seed 표본 표준편차다. [성공 수·분모](output_etc/check/nonenorm_keep1_eval_gpu4_20261102/summary.json).
- 6회 CPU/GPU 재채점·JSON 건수·finite·실제 actor/critic 보정·전원 접촉→gate·모델 불변·원본/고정 PTH 및 YAML 해시를 확인했다. 같은 seed의 초기12필드·CUDA RNG는 기존 solo5까지 bitwise 일치했고 keep1만 입력 차이에 따른 CPU RNG 차이를 기록했다. keep1 궤적의 최대1명·동일 그룹·미접촉 우선·선택 상대 접촉 후 교체를 확인했으며3회 교체734/672/739건 중 조기 교체는0건이다. 단독SIT225·CLIMB367·AT568·ON_TOP439·PUSH698명, 공동2/3/4명778/421/257그룹이다. 표준 코드·학습 설정은 변경하지 않았다. [검증](output_etc/check/nonenorm_keep1_eval_gpu4_20261102/verification.json), [고정 PTH·구성](output_etc/check/nonenorm_keep1_eval_gpu4_20261102/manifest.json).

### solo5 10k·15k·19k · 단독 AT 지정 상자9종

- 사용자 요청으로 기존 single AT의9종 크기를 물리 asset 생성 전에 적용하고 solo5의10k/15k/19k를 GPU2에서 각각4명·8물체 AT 전용512환경×seed0/1/2·1200step 첫 episode로 평가했다. 기존 고정 PTH/YAML을 사용하고 밀도100kg/m³·목표 샘플링·loco·평균 액션·distractor를 유지했다. 다른 작업을 보존하며1개 worker로9회 순차 실행했다. 기존 single1명 평가 및 혼합 행동 평가와 장면 구성이 다른 진단이다. BEFORE는 사용할 모델 선택 대기로 실행하지 않았다. [전체·크기별 Ever/Final·재현](output_etc/check/solo5_at9_10k15k19k_gpu2_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 각6,144명에서10k/15k/19k 순서 Ever는76.84±0.89/74.66±1.83/74.63±0.66%, 참고 Final은61.60±1.00/56.90±1.22/60.76±1.19%다. 평균±3seed 표본 표준편차다.0.57m 정육면체 Ever는41.15/31.29/30.55%로9종 중 가장 낮았다. 초기 상자→목표 XY 실제 범위는1.000~15.080m로 기존 생성 방식을 유지했다. [건수·seed별·거리](output_etc/check/solo5_at9_10k15k19k_gpu2_20261102/per_run.json).
- 9회 CPU/GPU 재채점·JSON·finite·전원 독립 AT·실제 대상9종 대응·점/영역 동일·모델 불변·원본/고정본 PTH 및 YAML 해시를 확인했다. 같은 seed의 초기12필드와 CPU/CUDA RNG가 세 PTH에서 bitwise 일치했다. 표준 코드·학습 설정은 변경하지 않았다. [검증](output_etc/check/solo5_at9_10k15k19k_gpu2_20261102/verification.json), [구성](output_etc/check/solo5_at9_10k15k19k_gpu2_20261102/manifest.json).

### 기존 solo5 · 15k/19k ordered512환경×3seed · 10k 기준 비교

- 사용자 요청으로15k/19k PTH·YAML을 복사 고정하고 GPU2에서 각각4명·8물체 혼합512환경×seed0/1/2·ordered·loco·원래 목표 샘플링/무게·평균 액션·distractor·1200step 첫 episode6회를 새로 평가했다. 기존 동일10k의3회 결과를 복사 재사용하고 현재 코드로 재채점해 총9회·3PTH를 비교했다.3개 worker 순차 실행 및 로그로 GPU2 peak3개·최종0개를 확인했다. 공동 Ever를 우선하고 단독 Ever를 다음 기준으로 평가하며 Final은 참고로 분리했다. 학습·표준 코드·기존 결과는 보존했다. [전체 표·seed 편차·그림·재현](output_etc/check/solo5_ordered_512_15k19k_vs10k_gpu2_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 10k/15k/19k 순서 공동 전체 Ever는82.43±1.60/80.29±1.73/80.85±1.35%, 공동4명은85.68±2.14/75.14±1.34/77.99±4.74%, 단독5행동 점 macro는85.12±0.65/83.37±0.85/83.08±0.56%다. 평균±3seed 표본 표준편차이며10k가 이번 비교의 공동 전체 및 단독 점 평균에서 가장 높았다. 공유 중심 전체 Ever는85.31/83.38/85.18%, 공동 전체 Final은79.54/76.66/77.96%다. [성공 수·분모·seed별 수치](output_etc/check/solo5_ordered_512_15k19k_vs10k_gpu2_20261102/summary.json), [그림](output_etc/check/solo5_ordered_512_15k19k_vs10k_gpu2_20261102/comparison.png).
- 9회 CPU/GPU 재채점·JSON 건수·finite·공유 중심·모델 불변·원본/고정본 PTH 및 YAML 해시를 검증했다. 동일 seed의 초기12필드·CPU/CUDA RNG가 세 PTH에서 bitwise 일치했고10k의 집계 전체가 이전 검증본과 정확히 같았다. 신규6회가 GPU2에서만 실행됐고 모든 trace의 기하 전원 접촉→native gate를 확인했다. 조건별 총단독SIT225·CLIMB367·AT568·ON_TOP439·PUSH698명, 공동2/3/4명778/421/257그룹이다. [구성·출처·PTH](output_etc/check/solo5_ordered_512_15k19k_vs10k_gpu2_20261102/manifest.json), [검증](output_etc/check/solo5_ordered_512_15k19k_vs10k_gpu2_20261102/verification.json).

### 기존 solo5 · 2k/4k/6k 및10k–20k 매1k · GPU2/4 ordered64환경 비교

- 사용자 요청으로2k/4k/6k/12k를 재평가하고 미평가11k/13k/15k/19k/20k를 새로 실행했다. 같은 조건의10k/14k/16k/17k/18k를 복사 재사용해 총14개 epoch의 단독5행동·공동2/3/4명·공유 중심 Ever/Final 표와 PNG/SVG 추세를 생성했다. 각4명·8물체 혼합64환경×seed0·ordered·loco·원래 목표 샘플링/무게·평균 액션·distractor·1200step 첫 episode다. 신규 실행은 GPU2·4만 사용했고 각3개 worker의 순차 실행·로그로 GPU당 peak3개 및 최종0개를 확인했다. [전체 비교표·그림·재현](output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 20k 저장 전 대기를 설정하고 파일 저장 후 내부20000epoch/frame2621440000·원본/복사본 해시를 확인한 뒤 평가를 완료했다. 이번64환경에서 공동 전체 Ever는14k57/64(89.1%)가 가장 높고 단독5행동 점 macro는17k87.7%가 가장 높았다.20k는 단독 점/영역 macro85.9/93.6%, 공동52/64(81.2%), 공유 중심53/64(82.8%), 단독 PUSH25/28(89.3%)다. Final은 참고로 분리했으며10k 이후 단조 개선을 확인하지 못했다. [성공 수·분모](output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/summary.json), [그림](output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/trend.png).
- 신규9회와 재사용5회 모두 현재 CPU 재채점·GPU/JSON 건수·finite를 확인했고 모델 불변·14개 PTH/원본 YAML 해시를 검증했다. 초기12필드는14개 모두 bitwise 일치하고 저장된13개 CPU/CUDA RNG도 일치한다. 과거10k는 RNG 파일이 없어 물리/과제 초기 상태로 대조했다. 재평가4k/6k/12k 성공 수는 과거와 같고2k는 ON_TOP 점 Ever12→13/18·PUSH Ever19→18/28·AT Final21→22/29 등의1명 차이를 기록했다.2k 표에는 새 값을 사용하며 원인은 확정하지 않았다. 학습·표준 코드·기존 결과는 보존했다. [실행·재사용 출처](output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/manifest.json), [고정 PTH](output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/checkpoint_manifest.json), [검증·기준점·동시 실행 상한](output_etc/check/solo5_dense_trend64_2k20k_gpu24_20261102/verification.json).

### CO+NONE norm·keep1 동일4k · 6/8명 전원 공동32환경 비교

- 직전 추세 평가에서 고정한 두4000epoch/frame524288000 PTH를 사용해 GPU5·6에서6명·12물체 및8명·16물체 전원 공동32환경×seed0·ordered·loco·원래 인원별 상자 크기/무게·평균 액션·distractor·1200step 첫 episode4회를 새로 평가했다. 저장된 actor/critic 보정·keep1 선택을 유지했으며 학습 프로세스·표준 코드·PTH는 변경하지 않았다. [전체 Ever/Final·공유 중심·개별 성공·재현](output_etc/check/nonenorm_keep1_epoch4k_n68_32_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- norm/keep1의 전원 동시 Ever는6명4/32·6/32,8명3/32·3/32이고 Final은6명3/32·3/32,8명2/32·3/32다. 공유 중심 Ever는6명8/32·8/32,8명9/32·5/32이며 Final은6명6/32·8/32,8명7/32·4/32다. keep1은6명 전원 Ever가2그룹 높지만8명은 동률이며 공유 중심은 낮아 이번 소표본에서 일관된 확장성 우위를 확인하지 못했다. 낙상은 norm6/8명5/192·9/256, keep1 5/192·0/256명이다.
- 4회 CPU/GPU 재채점·JSON 건수·finite·모델 불변·원본/고정본 PTH/YAML 해시와 내부4k/frame을 확인했다. 동일 인원에서 초기12필드·CUDA RNG가 모델 간 bitwise 일치했고 keep1 추가 관측의 CPU 초기화 RNG 차이는 기록했다. keep1 두 인원에서 매 step 유효 상대·미접촉 우선·최대1명·전원 접촉 후1명 유지 조건을 GPU/trace로 확인했다. [조건·PTH](output_etc/check/nonenorm_keep1_epoch4k_n68_32_20261102/manifest.json), [성공 수·분모](output_etc/check/nonenorm_keep1_epoch4k_n68_32_20261102/summary.json), [검증](output_etc/check/nonenorm_keep1_epoch4k_n68_32_20261102/verification.json).

### CO+NONE norm1k–5k·keep1 1k–4k · ordered64환경 추세

- 사용자 요청으로 norm1k/2k/3k/4k/5k와 keep1 1k/2k/3k/4k 정기 PTH·YAML을 복사 고정하고 GPU5·6에서9회 모두 새로 평가했다. 각4명·8물체 혼합64환경×seed0·ordered·loco·평균 액션·원래 무게·distractor·1200step 첫 episode이며 저장된 목표 샘플링·actor/critic 보정·keep1 선택을 유지했다.6/8명·AT/ON_TOP 거리 변경은 추가하지 않았다. [전체 Ever/Final·점/영역·공유 중심·재현](output_etc/check/nonenorm_keep1_trend64_1k5k_20261102/report.md), [그림](output_etc/check/nonenorm_keep1_trend64_1k5k_20261102/trend.png), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- norm1k→5k 공동 전체 Ever는64.1/60.9/67.2/71.9/79.7%, keep1 1k→4k는56.2/59.4/64.1/64.1%다. norm5k의2/3/4명 전원 동시 Ever는81.0/76.9/77.8%, 단독5행동 점 macro는85.05%다. keep1 4k는71.4/46.2/55.6%, 단독 macro85.06%다. 같은4k에서 norm은 공동71.9%·단독79.60%, keep1은 공동64.1%·단독85.06%로 과제별 차이가 있다. Final은 참고로 분리했다. [성공 수·분모](output_etc/check/nonenorm_keep1_trend64_1k5k_20261102/summary.json).
- 9회 CPU/GPU 재채점·JSON 건수·finite·모델 불변·원본/복사본 PTH/YAML 해시 보존을 확인했다. 초기12필드와 CUDA RNG는9개 모두 bitwise 일치하고 CPU RNG는 설정 내 일치한다. keep1의 추가 관측에 따른 설정 간 CPU 모델 초기화 RNG 차이는 별도로 기록했다. keep1 네 PTH는 매 step 선택 상대·미접촉 우선·전원 접촉 후1명 유지 조건을 GPU와 저장 trace에서 확인했다. 단독 표본SIT10·CLIMB12·AT29·ON_TOP18·PUSH28명, 공동2/3/4명42/13/9그룹의 소규모 추세다. [PTH·9개 구성](output_etc/check/nonenorm_keep1_trend64_1k5k_20261102/manifest.json), [검증](output_etc/check/nonenorm_keep1_trend64_1k5k_20261102/verification.json).

### 기존 solo5 · AT/ON_TOP 목표1–5m · 네 PTH 32환경 진단

- 사용자 요청으로8k/10k/14k/18k 고정 PTH를 재사용해 AT·ON_TOP만 각각32환경×4명·8물체·seed0·loco·1200step으로 평가했다. 기존 환경 중심±4.5m 정사각형과 payload 기준1–5m 원형 후보 샘플러를 비교해 GPU5·6에서 총16회를 완료했다. 복원 후 첫 reset 전에 평가용 거리 키만 적용했고 저장 PTH·학습 설정·표준 코드는 보존했다. [Ever/Final·점/영역·건수·재현](output_etc/check/solo5_at_ontop_goal5_32_8k10k14k18k_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 8k/10k/14k/18k 순서 AT Ever 기존→1–5m는83.6→82.0/84.4→72.7/83.6→71.9/82.8→76.6%, ON_TOP 점 Ever는72.7→75.0/73.4→80.5/78.9→71.9/77.3→75.8%다. Final은 참고로 분리했다. 조건별 실제 목표 거리 범위는 AT 기존1.103–12.591m·변경1.017–4.942m, ON_TOP 기존1.131–12.591m·변경1.013–4.985m였다. 목표 방향/위치 분포와 충돌 재샘플링도 바뀌므로 동일 장면의 거리만 바꾼 짝비교는 아니며 단일 seed 소규모 진단이다.
- 16회 CPU/GPU 재채점·JSON 건수·finite·모델/PTH/원본 YAML 해시 보존, 과제별128명 전원 단독 구성과 실제1–5m 범위를 확인했다. 같은 행동·조건의 초기12필드·CPU/CUDA RNG가 네 PTH에서 bitwise 일치했다. [조건·PTH](output_etc/check/solo5_at_ontop_goal5_32_8k10k14k18k_20261102/manifest.json), [검증](output_etc/check/solo5_at_ontop_goal5_32_8k10k14k18k_20261102/verification.json), [성공 수·개별 거리](output_etc/check/solo5_at_ontop_goal5_32_8k10k14k18k_20261102/summary.json).

### 기존 solo5 · 8k/10k/14k/18k ordered512환경×3seed 정밀 비교

- 네 정기 PTH·relation_config를 복사 고정하고 GPU5·6에서 각각 일반4명·8물체 혼합512환경×seed0/1/2·ordered·원래 무게·평균 액션·distractor·loco 초기화·1200step·첫 episode를 새로 평가했다. 총12회이며6/8명은 추가하지 않았다. Ever만 PTH 선택에 사용하고 Final은 정지·유지 참고자료로 분리했다. 표준 코드와 학습 프로세스는 변경하지 않았다. [전체 표·재실행](output_etc/check/solo5_ordered_512_seed012_8k10k14k18k_20261102/report.md), [그림](output_etc/check/solo5_ordered_512_seed012_8k10k14k18k_20261102/comparison.png), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 8k/10k/14k/18k 순서로 단독5행동 점 macro Ever는84.53/85.12/79.63/82.03%, 공동 전체 전원 동시 Ever는80.85/82.43/78.24/78.79%, 공유 중심 Ever는83.80/85.31/82.43/83.46%다. 모두3seed 성공률의 평균이며 macro는 다섯 단독 과제의 단순 평균이다.10k가 이번 표본에서 단독 macro와 공동 전체가 모두 가장 높고,8k와의 차이는 작다. [seed 편차·성공 수·분모](output_etc/check/solo5_ordered_512_seed012_8k10k14k18k_20261102/summary.json).
- 12회 CPU/GPU 재채점·JSON 건수·finite·공유 중심 일치·모델/PTH 해시 보존·기하 전원 접촉→native gate를 확인했다. 동일 seed의 초기12필드와 CPU/CUDA RNG가 네 PTH에서 bitwise 일치했다. 단독은 총SIT225·CLIMB367·AT568·ON_TOP439·PUSH698명, 공동은2/3/4명 각각778/421/257그룹이다. [검증](output_etc/check/solo5_ordered_512_seed012_8k10k14k18k_20261102/verification.json), [PTH 고정·12개 구성](output_etc/check/solo5_ordered_512_seed012_8k10k14k18k_20261102/manifest.json).

### 기존 solo5 · 14k/16k/18k ordered64환경 추세 추가

- 14000/16000/18000epoch PTH·relation_config를 복사 고정하고 일반4명/8토큰 및6·8명 전원 공동12·16토큰을 모두64환경×seed0·ordered·원래 무게·평균 액션·distractor·loco 초기화·1200step·첫 episode로 평가했다.18k 일반은 직전 동일64환경 검증본을 재사용하고 나머지8개를 GPU5·6에서 새로 실행했다.5명은 추가하지 않았다. 과거1k~12k의13개 결과와 합친22개 구성의 표·PNG/SVG 추세를 생성했다. 표준 코드·학습 프로세스는 변경하지 않았다. [전체 ever/final·단독 점/영역·공유 중심·재실행](output_etc/check/solo5_ordered_trend64_14k18k_20261102/report.md), [그림](output_etc/check/solo5_ordered_trend64_14k18k_20261102/trend.png), [PTH·구성](output_etc/check/solo5_ordered_trend64_14k18k_20261102/manifest.json), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 14k/16k/18k 순서로 일반 공동 전원 개인 목표 ever는57/64·50/64·48/64, final은54/64·49/64·48/64다.6명 ever는7/64·12/64·6/64, final은6/64·12/64·6/64이며8명 ever는4/64·8/64·6/64, final은2/64·7/64·5/64다. 공유 중심 final은6명10/64·16/64·12/64,8명8/64·11/64·11/64다. 이번 추가 표본에서 일반 혼합은14k,6/8명은16k가 높았고 단조 증가하지 않았다.
- 9개 구성 CPU/GPU 재채점·JSON/finite·모델/PTH 보존과 기하 전원 접촉→native gate를 확인했다. 새 PTH들 사이 및 과거12k 대비 인원/과제별 초기12필드가 bitwise 일치하고 새 구성끼리 CPU/CUDA RNG도 일치했다. 과거와 relation_config 해시가 같음을 확인했으며 단독/공동·공유 중심·낙상·기존 native gate 추진 진단을 함께 보존했다. [검증](output_etc/check/solo5_ordered_trend64_14k18k_20261102/verification.json), [과거 포함 집계](output_etc/check/solo5_ordered_trend64_14k18k_20261102/combined_breakdown.json).

### 기존·CO+NONE·keep1 최신 저장본 18000/4500/3500 평가

- 요청 시점의 기본 이름 PTH와 최신 정기 PTH 내부 epoch가 같음을 확인하고 기존18000/frame2359296000·CO+NONE4500/frame589824000·keep1 3500/frame458752000의 정기 PTH·relation_config를 복사 고정했다. GPU5·6에서 각 일반4명/8토큰64환경,6·8명 전원 공동12·16토큰32환경을 seed0·ordered·원래 무게·평균 액션·distractor·1200step·첫 episode로 새로 평가했다. keep1은 저장된 actor/critic 최대1명 선택·미선택 NONE, 나머지는 저장 보정·상대 제한 없음을 유지한다. 학습량이 다른 최신 상태 비교다. 표준 코드·학습 프로세스는 변경하지 않았다. [전체 ever/final·점/영역·공유 중심·재실행](output_etc/check/coupled_three_latest_eval_20261102/report.md), [PTH 고정](output_etc/check/coupled_three_latest_eval_20261102/manifest.json), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 기존/CO+NONE/keep1 순서로 일반 공동 전원 개인 목표 final은48/64·47/64·38/64다.6명 ever4/32·6/32·5/32와 final4/32·5/32·4/32,8명 ever0/32·4/32·2/32와 final0/32·2/32·2/32다. 공유 중심 final은6명5/32·8/32·7/32,8명1/32·5/32·3/32다.6/8명 개인 낙상은 기존18/192·13/256,CO+NONE5/192·1/256,keep1 3/192·2/256이었다.
- 전원 접촉 전후 정지 후5초 재출발은6명1/5·1/10·0/12,8명0/7·1/16·1/12다. 첫1명 이상 접촉 후 첫 정지 조건의 출발 순간 기하/힘 접촉 인원도 별도 집계했다.9개 rollout 재채점·JSON/finite·모델/PTH 보존·keep1 매 step 선택,같은 조건 초기12필드·CUDA RNG bitwise 일치와384그룹·248정지 창·389이동 판정 기록의 속도/거리/지속시간·낙상 전 조건을 확인했다. 기존/CO+NONE CPU RNG는 일치하며 keep1의 추가 관측 차원에 따른 모델 초기화 소비 차이는 별도로 기록했다. [평가 검증](output_etc/check/coupled_three_latest_eval_20261102/verification.json), [정지·접촉 인원](output_etc/check/coupled_three_latest_eval_20261102/contact_count_report.md), [분석 검증](output_etc/check/coupled_three_latest_eval_20261102/contact_count_verification.json).

### GPU0 전용 MPS 정상 종료

- 사용자 요청으로 GPU0 전용 pipe `/tmp/mps-test-1003/gpu0/pipe`의 서버753996·제어753978과 연결 클라이언트0개를 재확인한 뒤 `quit`으로 정상 종료했다. CO 학습796534는 이미 종료된 상태였다. GPU reset이나 직접 kill은 실행하지 않았다.
- 서버 로그의 Exiting과 두 데몬 종료를 확인했다.5초 후 jh의 GPU0 VLLM2209653·Python2554841은 동일 PID로 실행/GPU 메모리 유지 중이며, 다른 GPU의 MPS3개와 확인한 학습6개도 모두 살아 있었다. 서비스 요청 자체의 기능 검사는 하지 않았다. [종료 전](output_etc/check/gpu0_mps_shutdown_20261102/before.json), [종료 결과](output_etc/check/gpu0_mps_shutdown_20261102/quit_result.json), [종료 후](output_etc/check/gpu0_mps_shutdown_20261102/after.json).

### keep1 2000 추가 · 네 설정 동일 epoch 비교

- keep1의 내부2000epoch/frame262144000 PTH·relation_config를 복사 고정해 GPU5에서 일반4명/8토큰64환경·6명/12토큰32환경, GPU6에서8명/16토큰32환경을 각각seed0·ordered·원래 무게·평균 액션·distractor·1200step·첫 episode로 평가했다. 저장된 actor/critic 최대1명 선택·미선택 NONE을 유지했다. 기존·CO·CO+NONE2k의9개 검증 rollout을 복사 재사용해 네 설정을 비교했다. 표준 코드·학습 프로세스는 변경하지 않았다. [전체 ever/final·점/영역·공유 중심·재실행](output_etc/check/coupled_epoch2000_keep1_compare_20261102/report.md), [PTH·출처](output_etc/check/coupled_epoch2000_keep1_compare_20261102/manifest.json), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- keep1의 공동2/3/4/6/8명 전원 개인 목표 ever는24/42·9/13·5/9·2/32·4/32, final은23/42·8/13·5/9·2/32·1/32다. 공유 중심 ever는26/42·10/13·5/9·2/32·10/32, final은24/42·10/13·5/9·2/32·5/32다. 일반 공동 전체 final은36/64이며 단독 행동별도 집계했다.
- 전원 접촉 전후 정지 후5초 재출발은6명1/11·8명0/13이다. 첫1명 이상 접촉 후 첫 정지 조건에서는6명15/26·8명14/28이 재출발했고 그중14/15·13/14는 첫 전원 접촉 이전 출발이었다. [정의·전체 비교·출발 접촉 인원](output_etc/check/coupled_epoch2000_keep1_compare_20261102/contact_count_report.md).
- 12개 rollout 재채점·JSON/finite·모델/PTH 보존과 keep1의 매 step 선택 규칙을 확인했다. 같은 조건 초기12필드·CUDA RNG는 네 설정 간 bitwise 일치한다. keep1의 CPU 모델 초기화 RNG 소비량 차이는 별도 기록했고 기존과 두 norm의 CPU RNG는 일치한다.512그룹·369정지 창·559이동 판정 기록의 속도/거리/지속시간·낙상 전 조건과 기존 분모 일치를 확인했다. [평가·선택 검증](output_etc/check/coupled_epoch2000_keep1_compare_20261102/verification.json), [분석 검증](output_etc/check/coupled_epoch2000_keep1_compare_20261102/contact_count_verification.json).

### 기존·CO·CO+NONE · 동일2000epoch 평가·접촉 인원 비교

- 세 PTH의 내부2000epoch/frame262144000과 원본/복사 해시를 확인해 고정했다. GPU5·6에서 기존·CO+NONE의 일반4명/8토큰64환경×seed0,6·8명 전원 공동12·16토큰32환경×seed0를 새로 평가하고 CO2k의 검증된 동일 조건3개 rollout은 복사 재사용했다. ordered·원래 무게·평균 액션·distractor·1200step·첫 episode·상대 제한 없음과 각 actor/critic4층 보정을 유지했다. 진단 실행기의 MPS UUID 전환 전 물리 GPU 번호로 작업을 분배하도록 수정 후 실행했다. 표준 코드·학습 프로세스는 변경하지 않았다. [전체 ever/final·점/영역·공유 중심·재실행](output_etc/check/coupled_epoch2000_compare_20261102/report.md), [PTH·재사용 출처](output_etc/check/coupled_epoch2000_compare_20261102/manifest.json), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 기존/CO/CO+NONE 순서로 일반 공동 전원 개인 목표 final은41/64·38/64·34/64다.6명 ever5/32·3/32·1/32와 final1/32·2/32·1/32,8명 ever5/32·0/32·1/32와 final4/32·0/32·0/32다. 같은2k에서 CO+NONE의 우위는 확인되지 않았다. 앞선3.5k와 동일 조건에서 CO+NONE6명 final은1/32→6/32로 증가했다.
- 전원 접촉 전후0.5초 정지 후5초 내 재출발은6명2/16·1/10·0/12,8명0/14·0/10·0/14다. 첫1명 이상 접촉 이후 첫 정지 조건도 별도로 집계하고 출발 순간 기하/힘 접촉 인원을 보존했다.9개 rollout 재채점·JSON/finite·모델 보존·같은 조건 초기12필드 및 CPU/CUDA RNG bitwise 일치,384그룹·279정지 창·422이동 판정 기록의 속도/거리/지속시간/낙상 전 조건을 확인했다. [성공률 검증](output_etc/check/coupled_epoch2000_compare_20261102/verification.json), [정지·접촉 인원](output_etc/check/coupled_epoch2000_compare_20261102/contact_count_report.md), [분석 검증](output_etc/check/coupled_epoch2000_compare_20261102/contact_count_verification.json).

### 네 coupled 설정 · 정지 후 출발 순간 접촉 인원 비교

- 기존 전원 접촉 전후 정지 표본을 네 설정 모두 비교하고, 각 재출발 순간의 기하/힘 접촉 인원과 확인 구간의 전원 유지 비율을 추가 집계했다. 기존/CO/CO+NONE/keep1 순서로6명5초 재출발은1/4·1/10·1/8·1/9,8명은3/8·0/10·1/12·0/9다. 이 표본의6/8명 재출발은 시작 순간 모두 기하 전원 접촉이었으나 계속 유지되는 조건은 아니었다. [정의·전체 표·재실행](output_etc/check/coupled_four_latest_eval_20261102/contact_count_report.md).
- 별도로 첫1명 이상 기하 접촉 이후 첫0.5초 정지를 그룹당1회 골랐다. CO+NONE의6명17/29·8명12/30이5초 안에 재출발했고, 모두 첫 전원 접촉 이전 출발이었다. 출발 시 접촉 인원은6명 장면1~4명,8명 장면2~7명이다. 나머지 설정도6/8명에서 이 조건의 출발은 모두 전원 접촉 이전이었다. 엄격한 양손 기준이므로 다른 사람이 전혀 닿지 않았다는 의미나 의도적 대기/추진의 인과 증거로 해석하지 않는다.
- CPU로12개 궤적512그룹·354정지 창·589이동 판정 기록의 정렬/속도·지속시간·거리·낙상 전 조건을 검사하고 기존 전원 표본의 분모/건수와 일치를 확인했다. 새 시뮬레이션·정책·학습 변경은 없다. [그룹별](output_etc/check/coupled_four_latest_eval_20261102/contact_count_groups.json), [검증](output_etc/check/coupled_four_latest_eval_20261102/contact_count_verification.json).

### CO+NONE 3500 · 전원 접촉 뒤 정지·재출발 궤적 분석

- 직전 네 모델12개 rollout을 CPU로만 분석했다. 첫 낙상/목표50cm 진입 전 엄격한 전원 양손 접촉을 사용하고, 상자 속도1cm/s 미만0.5초 정지와 목표 방향5cm/s 초과0.5초·같은 연속 구간10cm 이상 전진을 구분했다. CO+NONE의 첫 전원 접촉 직전/직후 정지 장면 중5초 안에 재출발한 비율은2명7/7·3명4/4·6명1/8·8명1/12이며4명은 해당 표본이 없다. 동료 접촉을 인식해 기다린다는 인과 증거로 해석하지 않는다. [정의·전체 표·사례·재실행](output_etc/check/coupled_four_latest_eval_20261102/stop_start_report.md).
- 밀다가 멈춘 경우까지 포함한 첫 후속 정지에서는 CO+NONE6명2/20·8명2/25가 남은 관측 중 재출발했다.5초 동안90% 이상 정체한 장면은 각각17/19·22/25다.512그룹의 기록 정렬/전원 접촉→gate와201정지·70재출발 판정의 원본 속도/위치·시점·낙상 전 조건을 확인했다. 새 시뮬레이션·정책/학습 변경은 없다. [그룹별](output_etc/check/coupled_four_latest_eval_20261102/stop_start_groups.json), [검증](output_etc/check/coupled_four_latest_eval_20261102/stop_start_verification.json).

### 실행 중 coupled 네 설정 · 최신 저장 PTH ordered 평가

- 실제 실행 중인 기존 solo5·CO norm·CO+NONE norm·keep1의 요청 시점 최신 정기 PTH를 각각17000/2000/3500/3000epoch로 복사 고정했다. shrink1은 실행/저장본이 없어 제외했다. GPU5·6에서 일반4명/8토큰64환경×seed0와6·8명 전원 공동12·16토큰32환경×seed0를 각각 평가했다. ordered·원래 무게·평균 액션·distractor·1200step·첫 episode를 유지하고 각 설정의 원래 actor/critic norm·상대 선택을 사용했다. 학습량이 다른 진단 비교다. [전체 ever/final·점/영역·공유 중심·재실행](output_etc/check/coupled_four_latest_eval_20261102/report.md), [PTH 고정](output_etc/check/coupled_four_latest_eval_20261102/manifest.json), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 기존/CO/CO+NONE/keep1 순서로6명 전원 개인 목표 ever는2/3/7/6그룹·final2/2/6/6그룹,8명 ever2/0/3/0·final2/0/2/0그룹(각32그룹)이다. 공유 중심 final은6명3/3/6/7·8명4/0/4/1그룹이다. 단독 행동별과 일반 공동2/3/4명, 전원 양손 접촉·낙상도 함께 집계했다.
- 12개 rollout의 CPU/GPU 재채점·JSON/finite·모델/PTH 보존과 keep1의 매 step 최대1명/미접촉 우선/불필요한 교체 없음을 확인했다. 네 모델의 같은 조건 초기12필드·CUDA RNG는 bitwise 일치했다. 기존과 두 norm의 CPU RNG도 일치하며 keep1은 추가 관측 차원에 따른 모델 초기화 소비 차이로 달랐다(입력 차원만 바꾼 재현 확인). 정책 토큰 셔플은 CUDA RNG를 사용한다. 표준 코드·학습 프로세스는 변경하지 않았다. [검증](output_etc/check/coupled_four_latest_eval_20261102/verification.json).

### 기존 solo5 500 · 두 norm 500과 동일 조건 비교

- 기존 solo5의500epoch/frame65536000 PTH·relation_config를 복사 고정해 GPU5(일반·8명)·6(6명)에서 추가 평가했다. 일반4명/8토큰64환경×seed0,6·8명 전원 공동은12·16토큰32환경×seed0이며 ordered·원래 무게·평균 액션·distractor·1200step·첫 episode·상대 제한 없음을 직전 norm 평가와 맞췄다. 두 norm의 검증된 결과는 재사용했다. [세 모델 비교표·재실행](output_etc/check/solo5_base500_vs_norm500_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 기존500의 공동2/3/4/6/8명 전원 개인 목표 ever는27/42·6/13·3/9·0/32·3/32, final은25/42·6/13·3/9·0/32·0/32다. 공유 중심 final은29/42·7/13·5/9·1/32·3/32다. 단독 행동별 ever/final·점/영역도 함께 집계했다.
- 세 모델 간 같은 조건 초기12필드·CPU/CUDA RNG가 bitwise 일치했다. 기존3개 rollout의 CPU/GPU 재채점·JSON 건수·finite·actor/critic 보정 없음·모델 불변·원본/복사 PTH 해시 보존을 확인했다. 표준 코드·학습 프로세스는 변경하지 않았다. [검증](output_etc/check/solo5_base500_vs_norm500_20261102/verification.json).

### 두 norm 500 PTH · 일반64환경·6/8명 공동32환경 ordered 평가

- COUPLED-only·COUPLED+NONE 보정의500epoch/frame65536000 PTH와 relation_config를 복사 고정했다. GPU5/6에서 각각4명/8토큰 일반 혼합64환경×seed0,6명/12토큰·8명/16토큰 전원 공동32환경×seed0를 ordered·원래 무게·평균 액션·distractor·1200step·첫 episode로 평가했다. 상대 제한은 끄고 각 저장본의 actor/critic4층 norm을 유지했다. 단독 행동별 ever/final·점/영역과 공동2/3/4/6/8명 전원 개인 목표·공유 중심을 집계했다. [전체 표·재실행](output_etc/check/solo5_norm500_ordered_eval_20261102/report.md), [운영 명령](markdowns/USAGE.md#공통-stage2-push와-coupled).
- CO-only→CO+NONE의 공동2/3/4명 전원 개인 목표 final은19/42→24/42,4/13→8/13,4/9→4/9다.6명 ever는둘 다1/32·final1/32→0/32,8명 ever3/32→2/32·final둘 다1/32다. 6/8명 공유 중심 final은3/32→4/32·4/32→2/32이며6/8은각32그룹×1seed 진단이다.
- 같은 조건에서 두 PTH의 초기12개 tensor·CPU/CUDA RNG가 bitwise 일치했다.6개 rollout의 CPU/GPU 재채점·JSON 건수·finite·양쪽4층 보정 모드·모델 불변과 원본/복사 PTH 해시 보존을 확인했다. 표준 코드·학습 프로세스는 변경하지 않았다. [검증](output_etc/check/solo5_norm500_ordered_eval_20261102/verification.json), [PTH·설정](output_etc/check/solo5_norm500_ordered_eval_20261102/manifest.json).

### solo5 keep1/shrink1 학습 설정 · viewer keep3/shrink3 NONE 전환·재평가

- solo5 기반 `stage2_coupled_push_individual_goal_solo5_keep1`·`stage2_coupled_push_individual_goal_solo5_shrink1`의 env/train YAML과 train/test/VNC/eval을 추가했다. 학습·추론의 actor/critic4층에서 미접촉 공동 상대 최대1명을 선택하고 나머지는 NONE으로 바꾼다. keep1은 미접촉 상대가 없으면 접촉한1명 유지, shrink1은0명이다. 선택 ID를 관측에 저장해 PPO 미니배치에서도 재사용하며 기본 관측은1276차원이다. 보상·물성·샘플링·4명/1024환경/8토큰·Stage1 PUSH26k 전이는 유지하고 새 모드는 checkpoint 계약으로 구분한다. 학습은 실행하지 않았다. [설정·명령](markdowns/config.md), [정의](markdowns/reference/stage2_coupled_push.md#최대1명-coupled-선택-학습).
- 사용자 수정 요청대로 기존 viewer keep3/shrink3도 미선택 상대를 마스킹하는 대신 NONE으로 전환한다. 실제 그룹/보상과 기존5명 이상 적용·선택/교체 규칙은 유지하며 actor 전용이다. COUPLED0명도 다른 사람을 NONE으로 계속 본다. 과거 마스킹 평가 결과는 보존하고 현재 코드로 재실행하면 NONE 전환임을 USAGE에 명시했다.
- 관련 CPU64검사와 기본 모드8조건의 출력/입력·파라미터 gradient bitwise 일치를 확인했다. 접촉/상실·단독/혼합/13명·부분 reset·독립 RNG·양쪽4층의 실제 relation bias/유한 attention·PPO 배치 재사용/역전파·26k 전이·설정 간 혼용 거부를 검사했다. GPU6에서 새 두 설정 각각4명·4환경·8preset×16step 및8명·4환경·전원 공동16step의 실제 물리/모델/AMP·부분 reset·가중치 보존을 통과했다. wrapper6개 호출 캡처와8개 shell 구문을 확인했다. [검증](output_etc/check/solo5_keep1_shrink1_configs_20261101/verification.json), [CPU](output_etc/check/solo5_keep1_shrink1_configs_20261101/cpu_tests.log).
- 기존12000 PTH·8명·64환경×seed0·물체16개·ordered·원래 무게·평균 액션·1200step으로 NONE keep3(GPU5)/shrink3(GPU6)를 재평가했다. all은 직전 동일 조건 결과를 재사용했다. all/keep3/shrink3 전원 개인 목표 ever는4/3/3그룹, final은 모두2/64그룹이며 공유 중심 ever는8/9/4그룹, final은5/7/3그룹이다. 초기12필드/CPU·CUDA RNG·선택 규칙·CPU/GPU 재채점·JSON/finite·모델/PTH 보존을 확인했다. GPU5에서 두 모드 실제 VNC·8명·16step×2씬도 통과했다. [NONE 결과·이전 마스킹 비교](output_etc/check/solo5_peer_none_eval8_epoch12000_20261101/report.md), [재채점·선택 검증](output_etc/check/solo5_peer_none_eval8_epoch12000_20261101/verification.json), [VNC](output_etc/check/solo5_peer_none_eval8_epoch12000_20261101/viewer_verification.json).

## 2026-11-01

### solo5 12000 · 8명 ordered · all/keep3/shrink3 평가

- 최근 추세 평가의12000 PTH를 복사 고정해 GPU6(all/shrink3)·5(keep3)에서8명·64환경×seed0·물체16개·원래 무게·평균 액션·distractor·1200step·첫 episode 조건으로 세 방식을 재평가했다. 기존 공간 채점/개인 낙상 제어는 유지하고 접촉·선택을 매 step 기록했다. 전원 개인 목표 ever는 all/keep3/shrink3 각각4/2/3그룹, final은 모두2/64그룹이다. 공유 중심 ever는8/10/7그룹, final은5/8/4그룹이다. 전원 양손 기하 접촉 ever는54/55/54그룹이며64환경×1seed 진단이다. [비교표·재실행](output_etc/check/solo5_peer_eval8_epoch12000_20261101/report.md), [실행 안내](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 세 조건의 초기12개 tensor·CPU/CUDA RNG와 이전12000 기준의 초기 상태가 bitwise 일치했다. all의 성공 건수도 이전 결과와 일치했다. 3회 CPU/GPU 재채점·JSON 건수·finite·매 step 선택 수/접촉 우선/불필요한 재선택 없음·모델/PTH 해시 보존을 확인했다. keep3은 항상3명, shrink3은0~3명이며 전원 접촉 시 제거 규칙을 실제 rollout에서 검증했다. 표준 학습/평가 코드는 변경하지 않았다. [검증](output_etc/check/solo5_peer_eval8_epoch12000_20261101/verification.json).

### 개인 목표 coupled viewer · 접촉 기반 keep3/shrink3 상대 선택

- `COUPLED_PEERS=all|keep3|shrink3`를 test/VNC와 직접 CLI에 추가했다. 5명 이상 공동 그룹에서 query별 미접촉 상대 최대3명을 선택하고 필요한 슬롯만 갱신한다. keep3은 접촉한 상대를 채워3명 유지, shrink3은 접촉한 상대를 제외해 전원 접촉 시0명이다. 접촉 상실 시 다시 후보가 되며 기준은 양손 기하 `on_face`다. 선택되지 않은 COUPLED key만 actor 사람 attention4층에서 가리고 SELF/NONE/BEFORE는 유지한다. 실제 그룹·상자·목표·보상/gate·물성과 기본 all/4명 이하 동작은 유지한다. [명령](markdowns/config.md#추론-옵션), [정의·제약](markdowns/USAGE.md#공통-stage2-push와-coupled).
- 관련 CPU42검사 통과와 기존 코드 대비 기본 모드8조건의 출력/입력·파라미터 gradient bitwise 일치를 확인했다. 독립 RNG·부분 reset·슬롯 유지/교체·전원 접촉/접촉 상실·NONE/BEFORE 보존·4층 셔플 mask·학습/resume/정식 평가 오사용 거부를 검사했다. [CPU 로그](output_etc/check/solo5_peer_selection_20261101/cpu_tests.log), [기존 동작 비교](output_etc/check/solo5_peer_selection_20261101/legacy_verification.json).
- 사용자 지정 PTH의12500epoch/frame1638400000 복사본을 고정해 GPU5에서8명·1환경·8토큰·ordered·16step×2씬의 두 모드 실제 VNC를 통과했다. 별도 keep3 실행에서32액션 finite·씬별 선택 초기화·모델 불변과 CUDA 전원 접촉 시 keep3 유지/shrink3 제거·부분 reset을 확인했다. 짧은 실행 검증이며 성능 평가는 하지 않았다. 학습은 실행하지 않았다. [검증·PTH 해시](output_etc/check/solo5_peer_selection_20261101/viewer_verification.json).

### solo5 사람 attention 개수 보정 두 설정 추가 · 학습 미실행

- solo5 기반 `stage2_coupled_push_individual_goal_solo5_countnorm_coupled`·`stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none`의 env/train YAML과 전용 train/test/VNC/eval을 추가했다. 전자는 COUPLED, 후자는 COUPLED/NONE의 query별 유효 key 수 K를 각각 세어 모든 actor/critic 사람 attention 층의 logit에 `−log K`를 적용한다. SELF/BEFORE·GTA/object attention·샘플링/보상/물성·26k 전이는 유지하고 학습 인원은4명·1024환경·8토큰이다. 기본 보정 없음과 기존 checkpoint 계약은 유지하며 새 모드는 저장 graph 계약으로 구분한다. [설정·명령](markdowns/config.md), [정의](markdowns/reference/stage2_coupled_push.md#사람-attention-관계별-개수-보정).
- CPU 검사69개 통과·기존78k 파일 부재3개 건너뜀, 추가 회귀46개 통과(21k 전이 전용1개 선택 제외). 관계 내 선호/SELF:BEFORE 비율 보존·관계별 복제에 대한 전체 비중 보정·masked/빈 관계·셔플/역전파·actor/critic4층 적용·실제26k 전이·실험 간 checkpoint 혼용 거부를 확인했다. 기존 코드와 기본 보정 없음의1/4/8/13명×혼합/전원 공동8조건 출력·입력/파라미터 gradient가 bitwise 일치했다. [검사 로그](output_etc/check/solo5_countnorm_configs_20261101/cpu_tests.log), [회귀](output_etc/check/solo5_countnorm_configs_20261101/regression_tests.log), [기존 동작 비교](output_etc/check/solo5_countnorm_configs_20261101/legacy_verification.json).
- 사용자 요청으로 학습·시뮬레이션은 실행하지 않았다. wrapper의 최종 Python 호출을 캡처하는 모의 검사로 두 실험의 train/test/eval 인자·출력 분리·ordered viewer 연결과 shell 구문만 검증했으며 본학습 출력 폴더를 만들지 않았다. 명령과 문서 링크/diff를 확인했다. [명령 검증](output_etc/check/solo5_countnorm_configs_20261101/command_verification.json).

### 개인 목표 solo5 · ordered5명 공동 PUSH 추세 추가

- 직전 추세 비교의4000/8000/12000 PTH를 복사 고정해 GPU0·3에서5명·물체10개·전원 공동·64환경×seed0·1200step으로 평가했다. ordered·원래 질량/인원별 크기 규칙·평균 액션을 유지했다. 개인 목표 전원 ever는15.6/12.5/17.2%, final은15.6/12.5/15.6%, 공유 중심 final은25.0/17.2/35.9%다. 요청한2/3/4/5/6/8명×epoch 표와 gate 진단도 저장했다.2/3/4명은 기존 일반 혼합42/13/9그룹이며5/6/8명 전원 공동64그룹과 장면 분포가 다르다. [표·재실행](output_etc/check/solo5_ordered_trend64_n5_seed0_20261101/report.md), [5·6·8명 그림](output_etc/check/solo5_ordered_trend64_n5_seed0_20261101/trend_n568.png).
- 3회 CPU/GPU 재채점·JSON 건수·finite·모델 불변·PTH epoch/원본·복사 해시 보존을 확인했다. PTH 사이 초기12개 tensor bitwise 일치·gate 기록 정렬·전원 기하 접촉→native gate를 확인했고 float32/64 ever·공유 중심 frame 판정 차이는0이다. 학습/표준 평가 코드는 수정하지 않았다. [검증](output_etc/check/solo5_ordered_trend64_n5_seed0_20261101/verification.json).

### 개인 목표 solo5 · ordered PTH 추세64환경×1회

- 사용자 지정대로 이후 공동 PUSH 평가는 ordered를 사용하도록 운영 문서에 기록했다. 정식 프로토콜의 비random 제한은 유지하고, 전용 비교 실행기로1000/2000/4000/6000/8000/10000/12000 일반4명 혼합과4000/8000/12000의6·8명 전원 공동을 GPU0·3에서 평가했다. 각64환경·seed0·1200step·원래 질량·평균 액션·물체2N이며 PTH와 relation_config를 복사 고정했다. 학습/표준 평가 코드는 수정하지 않았다. [정의·전체 ever/final·점/영역·재실행](output_etc/check/solo5_ordered_trend64_seed0_20261101/report.md), [그림](output_etc/check/solo5_ordered_trend64_seed0_20261101/trend.png).
- 일반 혼합 공동 전체 final은1000의57.8%→10000의87.5%→12000의78.1%다. 공동2/3/4명 표본은42/13/9그룹이라4명 변화는 특히 주의해 해석한다.6명 전원 개인 목표 ever/final은4000의4.7/1.6%→12000의12.5/12.5%,8명은4.7/1.6%→6.2/3.1%다. 단독 행동별·공유 중심과 gate 전 이동/정지 후 추진 진단도 함께 저장했다. [집계](output_etc/check/solo5_ordered_trend64_seed0_20261101/breakdown.json).
- 13회 CPU/GPU 재채점·JSON 건수·finite·모델 불변·7개 PTH 내부 epoch/해시 보존을 확인했다. 같은 인원/과제 설정의 초기12개 tensor는 PTH 사이 bitwise 일치하며 float32/64 ever·공유 중심 frame 판정 차이는0이다. gate/접촉 기록 정렬과 전원 기하 접촉→native gate도 확인했다. 기존 학습은 보존했다. [검증](output_etc/check/solo5_ordered_trend64_seed0_20261101/verification.json).

### 개인 목표 solo5 11500 · 일반 eval의 gate 이전 이동 진단

- 저장 궤적에서 첫 힘 접촉 후0.5초를 제외하고 지속 전진과 초기 충격 뒤 정지를 분리했다. 이후1초 이상 관찰 가능한2/3/4명 그룹의 gate 전 연속1초·추가10cm 전진은344/443·268/349·208/247개였다. 다른 사람의 힘 접촉 기록이 아직 없는4명 그룹 구간은65개 중 지속 전진2·거의 정지55개이며, 이55개 중 초기5cm 초과 이동 뒤 정지는13개다. 정지 프레임에도 힘 접촉 기록 비율 중앙값96.6%여서 의도적 대기/힘 부족은 구분하지 않는다. CPU 분석만 수행했고 구간 경계·finite·건수 검증을 통과했다. [정의·표·한계](output_etc/check/solo5_11500_gate_wait512_seed012_20261101/persistence_report.md).
- 같은11500 PTH·일반4명/8토큰 혼합·ordered·원래 질량·평균 액션·512환경×seed0/1/2·1200step을 GPU5·6에서 재평가하며 매 step native gate·기하/힘 접촉·공유 상자 속도를 추가 기록했다. 첫 gate 완료 frame을 제외하고 낙상 전까지만 관측해 초기부터/첫 접촉 이후의 최대 XY 변위·목표 거리 감소율·속도를 공동2/3/4명별로 집계했다. 표준 코드·정책·기존 학습은 유지했다. [정의·표·대표 장면·명령](output_etc/check/solo5_11500_gate_wait512_seed012_20261101/report.md).
- 초기부터 gate 전 최대 이동거리 중앙값은2/3/4명0.401/0.621/0.886m, 목표 거리 감소율 중앙값은10.8/16.5/23.4%,10cm 초과 이동 그룹은88.2/92.6/94.6%였다. 첫 접촉 이후만 보아도 이동은비슷하며 구간 속도 중앙값은0.217/0.208/0.236m/s다. goal_saturation으로 전원 기하 접촉보다 먼저 native gate를 완료한 그룹은0/16/21개였다. 초기사람/상자·DOF·과제/목표 tensor bitwise 일치·CPU/GPU 재채점·JSON 건수·finite·모델/PTH 해시 보존과 전원 기하 접촉 참이면 native gate도 참임을 확인했다. 재실행 궤적·낙상은 이전 eval과 일부 달라 필드별 비교를 기록했다. [집계](output_etc/check/solo5_11500_gate_wait512_seed012_20261101/summary.json), [그룹별](output_etc/check/solo5_11500_gate_wait512_seed012_20261101/groups.json), [검증](output_etc/check/solo5_11500_gate_wait512_seed012_20261101/verification.json).

### 개인 목표 solo5 11500 · 사람 간 attention/액션 헤드 피처 진단

- GPU5·6에서11500 PTH·ordered·원래 질량·평균 액션·32환경·seed0·1200step의4명 혼합 및4/6/8명 전원 공동을 진단했다.5step마다 actor human coordination4layer×2head의 동일 Q,K에 대한 softmax(QK/√d)·softmax(QK/√d+bias)를 관계/상태별로 집계하고64차원 head 입력·512차원 hidden·액션을 측정했다. 같은 encoder 출력의 human bias 제거 재계산은 진단에만 쓰고 실제 정책/표준 코드는 유지했다. [표·명령·그림](output_etc/check/solo5_attention_features_n468_20261101/report.md).
- 4명 혼합의 사람쌍 평균 NONE/SELF/COUPLED는22.92/29.64/24.37→20.89/30.78/26.79%이며 BEFORE 표본은 없다.4/6/8명의 공동 상대 전체 비중은73.29/82.37/86.96→75.99/84.43/88.58%다. 마지막 미접촉 상대의 공동 내 비중은41.65/26.07/19.40%(균등 대비1.25/1.30/1.36배)였다. 전원 접촉 정체 구간의4명 대비 head 입력 평균 코사인은6명0.929·8명0.912이며 RMS는 약0.88로 유지됐다. 사람 간 bias 제거의 상대 입력 변화는4/6/8명0.359/0.187/0.150%다. 접촉 전후 액션 변화 차이는 실제 물체/자세 변화가 포함된 관측 결과이며 인식 실패의 인과 증거로 해석하지 않는다. [레이어/head·사건별 상세](output_etc/check/solo5_attention_features_n468_20261101/comparison.json), [미접촉 상대 진단](output_etc/check/solo5_attention_features_n468_20261101/contact_attention.json).
- 4개 rollout CPU/GPU 재채점·JSON 건수·finite·float32/64 판정 차이0·attention 확률 합·모델/PTH 해시 보존을 확인했다. 셔플을 되돌린 수동 coordination 재구성 오차는1e-6 미만이었다.6/8명 trace는 직전11500 기본 토이실험과 필드별로 비교해 별도 기록했다. 기존 학습은 유지했다. [검증](output_etc/check/solo5_attention_features_n468_20261101/verification.json).
- 추가 요청에 따라 GPU6에서 저장 피처만 재생해 원본 QK/√32의 관계별 평균/표준편차·같은 query 내 표준편차·layer/head별 분포를 계산했다.4명 혼합의 NONE/COUPLED 전체 QK는0.424±1.155/0.490±1.236, 같은 query 내 표준편차 평균은0.531/0.389다. 두 관계를 함께 보는 query/head에서 COUPLED 평균 점수가 큰 비율은 원본50.9→bias 후69.0%다. 같은 head/relation의 상수 bias는 관계 내부 logit 차이를 줄이지 않는다는 점을 구분했다. 새 시뮬레이션 없이 기존 확률 평균과 재구성·해시 보존을 검증했다. [표·분포 그림·재실행](output_etc/check/solo5_attention_features_n468_20261101/qk_distribution/report.md).

### 개인 목표 solo5 11500 · 접촉 정체 후 단일 충격 토이실험

- GPU5·6에서6/8명·ordered·원래 질량·평균 액션·32환경·seed0·1200step의 기본/충격 조건을 비교했다. 전원 양손 접촉·속도1cm/s 미만0.5초·목표까지50cm 이상·전원 생존이면30초 이내 한 번만 상자 중심에 목표 방향 J=질량×0.3m/s를 물리1step(16.7ms) 동안 가했다. 이후 외력/액션 노이즈 없이 기존 정책을 유지했다. [설정·명령·결과](output_etc/check/solo5_single_impulse_n68_20261101/report.md).
- 충격 적용은6명17·8명22장면이다. 충격 전 기록 궤적/발동이 일치하는17/20장면에서 직후 속도 중앙값은0.174/0.170m/s였지만,1초 후부터5초 사이 추가50cm 전진은 기본0/0→충격1/0장면이었다. 전원 개인 목표 ever/final은6명4/4→4/4,8명2/2→1/1(각32그룹)이며 새 공유 중심 성공은 없었다.8명2장면은 충격 전 차이로 엄격 반응 비교에서 제외했다. 초기16필드/RNG 일치·실제 질량·body 인덱스·비영 외력 최대1회·가중치/PTH 해시 보존·4개 rollout CPU/GPU 재채점·JSON 건수·finite를 확인했다. 이번 강도의 상자 충격만으로 지속 추진 전환은 대부분 얻지 못했으며 원인을 확정하지 않는다. 표준 코드·기존 학습은 변경하지 않았다. [사건별·검증](output_etc/check/solo5_single_impulse_n68_20261101/comparison.json).

### 개인 목표 solo5 최신11500 · 일반 혼합 ordered 평가

- 요청 시 최신 정기/일반11500epoch/frame1507328000의 모델 일치를 확인하고 정기 PTH·relation_config를 복사 고정했다. GPU5(seed0)·6(seed1/2)에서4명·8토큰·512환경×3seed·1200step 일반 혼합 장면을 평가했다. 기존 비교 실행기로 공동 구역만 ordered 배정하고 원래 무게·평균 액션·distractor·첫 episode 공간 채점을 유지했다. 표준 random 배정 평가와 구분되는 조건이다. [설정·재실행·표](output_etc/check/solo5_11500_ordered_eval512_seed012_20261101/report.md).
- 공동2/3/4명 전원 개인 목표 ever는77.4/77.1/74.1%, final은74.6/73.8/68.6%, 공유 중심 final은77.9/79.0/77.5%다. 단독 PUSH ever/final은77.7/75.4%, 낙상은140/6144명이다. CPU/GPU 재채점·JSON 건수·finite·float32/64 판정 차이0·ordered 순서/목표 연결·모델/PTH 해시 보존을 확인했다. 초기 물리 장면·단독 목표는9500 일반 평가와 bitwise 동일하며 CPU 목표 재계산의 float32 좌표 차이는 최대0.061mm였다. 표준 코드·기존 학습은 변경하지 않았다. [상세](output_etc/check/solo5_11500_ordered_eval512_seed012_20261101/breakdown.json), [채점 검증](output_etc/check/solo5_11500_ordered_eval512_seed012_20261101/verification.json), [배정 검증](output_etc/check/solo5_11500_ordered_eval512_seed012_20261101/assignment_verification.json).

### 비교 viewer 질량 배수 직접 지정

- 요청대로 `view.sh <인원> mass [씬수]`에 `MASS_SCALE`을 연결했다.0.8은20%·0.6은40% 감량이고, 기존 mass80과 새 mass60 preset도 제공한다. 양의 유한값을 생성 단계의 공동 상자 질량·관성에 적용하고 형상/마찰·평균 액션·기본10씬은 유지한다. 표준 학습/평가는 변경하지 않았으며 config/USAGE 명령을 갱신했다.
- GPU6에서 명시적0.6배의 실제 VNC·8명·16step×2씬과0.8배의 headless·8명·16step을 통과했다. 실제 질량/관성·씬 전환·모델 보존·비양수 입력 거부와 초기 사람/상자·관측/목표/CPU·CUDA RNG 일치를 확인했다. GUI/headless의 표시 actor 수 차이는 물리 actor와 구분했다. [검증](output_etc/check/solo5_mass_noise_n68_20261101/viewer_mass_scale_validation.json).

### 감량/노이즈 비교 viewer 순차10씬 재생

- 요청대로 비교 `view.sh`에 세 번째 씬 수 인자(기본10)를 추가했다. 한 모델·VNC에서 장면을 새로 뽑아 순차 재생하고, 씬별 출력/진단을 분리하며 기존 평균/샘플링·물성·씬당1200step을 유지한다. 평가 중 낙상으로 꺼진 관절 stiffness/damping/driveMode를 다음 씬 전에 복원하고, 액션/step/reset wrapper와 hook도 씬마다 정리한다. config/USAGE의 명령·횟수 의미를 갱신했다.
- GPU6·실제 VNC·8명 mass80·16step×2씬의 새 장면/정상 전환·물성·pose·가중치 보존·정상 종료를 확인했다. 별도6명 noise·16step×2씬에서 첫 씬 종료 후 낙상 제어 해제를 강제로 주입하고 두 번째 씬의 원래 관절 속성 복원을 검증했다. [VNC 로그](output_etc/check/solo5_mass_noise_n68_20261101/viewer_repeat_smoke.log), [낙상 제어 복원](output_etc/check/solo5_mass_noise_n68_20261101/repeat_motor_check/motor_restore_validation.json). 기존 학습·실행 중 viewer는 종료하지 않았다.

### 개인 목표 solo5 11000 · 4/6/8명 감량20%/40% 비교

- 동일11000 PTH·ordered·전원 공동·평균 액션·32환경·seed0·1200step·물체2N에서4명20% 감량과4/6/8명40% 감량을 추가 평가했다.4명 두 조건은 GPU5,6/8명40%는 GPU6을 사용하고 기본·6/8명20%는 기존 검증 결과를 재사용했다. 감량40%는 생성 단계에서 전원 공동 actor만 질량/관성0.6배·밀도1.8로 만들며 형상/마찰은 유지한다. [설정·재실행](output_etc/check/solo5_mass20_40_n468_20261101/manifest.json).
- 20→40% 감량의 전원 개인 목표 ever는4명78.1→71.9%·6명31.3→50.0%·8명15.6→37.5%, final은75.0→62.5%·31.3→40.6%·12.5→21.9%다. 공유 중심 final은81.3→75.0%·46.9→53.1%·21.9→34.4%다.40% 감량의 낙상은4/6/8명 모두0명이며,4명은 목표 유지가 저하되어 감량 효과를 인원별로 구분해야 한다. [비교표](output_etc/check/solo5_mass20_40_n468_20261101/report.md).
- 같은 인원의 초기16필드·CPU/CUDA RNG bitwise 일치·실제 actor 질량/관성/마찰·첫 step pose 연속성·모델/PTH 해시 보존·CPU/GPU 재채점/JSON 건수·finite를 확인했다. 새 학습/표준 코드 변경 없이 확인 경로에 기록했으며32환경×1seed 결과다. [상세 검증](output_etc/check/solo5_mass20_40_n468_20261101/comparison.json).

### 개인 목표 solo5 11000 · 6/8명 질량20% 감소·액션 샘플링 비교

- 비교용 `view.sh <인원> <baseline|mass80|noise>`를 추가하고 config/USAGE에 VNC 명령을 기록했다. 같은 PTH·ordered·1환경에서 평가와 같은 첫 episode를 화면에 표시하며 출력은 매번 별도 확인 경로에 저장한다. GPU6·포트6096·실제 VNC/렌더링·8명 mass80·32step 실행과 물성/첫 step pose 연속성/가중치 보존·정상 종료를 확인했다. [실행 로그](output_etc/check/solo5_mass_noise_n68_20261101/viewer_smoke.log).
- 직전11000 PTH를 고정하고 ordered·전원 공동·32환경·seed0·1200step·물체2N에서 기본/질량·관성0.8배/원래 질량+매 step Gaussian 액션을 각각 평가했다. GPU5는6명, GPU6은8명을 실행했다. σ는 저장본의 고정 exp(−2.9)=0.055023이며 기존 clipping·model.eval을 유지했다. 두 변경을 동시에 적용하지 않았고 표준 학습/평가 코드는 유지했다. [설정](output_etc/check/solo5_mass_noise_n68_20261101/manifest.json), [명령·표](output_etc/check/solo5_mass_noise_n68_20261101/report.md).
- 기본→감량→노이즈의 전원 개인 목표 final은6명6.3→31.3→9.4%,8명3.1→12.5→6.3%다. 공유 중심 ever는6명12.5→46.9→50.0%,8명6.3→28.1→53.1%, final은9.4→46.9→15.6%·3.1→21.9→18.8%다. 낙상은6명1→6→6/192명,8명5→10→13/256명이다. 감량은 final 유지, 노이즈는 ever 도달 개선이 두드러졌으며32환경×1seed 진단 결과다. [전체 지표·접촉 진단](output_etc/check/solo5_mass_noise_n68_20261101/comparison.json).
- 기본 조건의 전원 양손 기하 접촉은6명32/32·8명30/32장면에서 한 번 이상 충족했다. 전원 접촉 중 목표 미도달 frame의 속도1cm/s 미만 비율은6명 기본87.4%·감량68.8%·노이즈17.8%,8명85.6%·70.6%·13.8%다. gate 미개방만으로 정체를 설명할 수 없으며, 손 반력은 순 추진력·마찰력 전체와 구분한다.
- 초기16개 물리/관측/CPU·CUDA RNG 필드 bitwise 일치·기본 조건의 이전 지표 재현·실제 actor 질량/관성/마찰·첫 step pose 연속성·Gaussian 표본/액션 선택·모델/PTH 해시 보존·CPU/GPU 재채점/JSON 건수·finite·float32/64 차이0을 확인했다. 실행 중 물성 setter가 첫 step에 상자를 주차 위치로 되돌린 감량 시도는 제외하고, 생성 단계부터 해당 전원 공동 actor만 밀도2.4로 만든 조건을 재평가했다. 초기 pose 재제출만으로 해결되지 않은 짧은 시도도 제외했다. [검증](output_etc/check/solo5_mass_noise_n68_20261101/verification.json). 기존 학습은 종료·재시작하지 않았다.

### 개인 목표 solo5 최신11000 ordered 비교32환경·4/6/8명

- 일반/정기 최신이11000epoch/frame1441792000임을 확인하고 정기 PTH·relation_config를 복사 고정했다. 요청대로4/6/8명 전원 공동 PUSH·물체2N·32환경·seed0·1200step에서 random/ordered를 비교했다. GPU6에서 시작하고 추가 허용 뒤6/8명 ordered는 GPU5에서 병행했다. 표준 평가/학습 코드는 유지하고 비교 전용 harness에서 기존 첫 episode 공간 채점을 사용했다. 정식512환경×3seed 결과는 아니다. [설정·명령](output_etc/check/solo5_ordered_eval32_n468_latest_20261101/manifest.json).
- 전원 개인 목표 ever의 random→ordered 성공 그룹 수는4명23→25/32·6명1→2/32·8명1→2/32, final은23→24·1→2·0→1/32다. 공유 중심 ever는25→26·6→4·2→2/32, final은25→26·3→3·0→1/32다. 낙상은4명6→1/128명·6명11→1/192명·8명10→5/256명으로, 낙상 감소와 다인원 성공률은 구분해야 한다. [표](output_etc/check/solo5_ordered_eval32_n468_latest_20261101/report.md), [개인/공동·상세 검증](output_etc/check/solo5_ordered_eval32_n468_latest_20261101/comparison.json).
- 같은 인원별 물리 상태/DOF/크기/그룹/공유 목표·CPU/CUDA RNG13개 필드 bitwise 일치, 개인 목표/구역 연결·ordered 순서·CPU/GPU 재채점/JSON 건수·finite·float32/64 차이0·모델 가중치/원본·복사본 PTH 해시 보존을 확인했다. 병행 전환 중 실행기 셸 재읽기 오류는 완성된6개 평가를 보존한 재실행으로 정상 종료를 확인했다. 기존 학습 프로세스는 유지했다.

### 공동 PUSH viewer 사람↔구역 배정 옵션

- coupled test/VNC에 `COUPLED_ASSIGNMENT=ordered|distance|optimal`(직접 실행 `--coupled_assignment`)을 추가했다. 좌우 정렬·Hungarian 총 접근 거리 최소화·교차 수→최장 거리→총 거리 최소화를 제공하며 기본 random은 유지한다. optimal은8명까지 전수 비교,9명 이상은 인원 제한 없이 ordered/distance에서 최대2N회 두 사람 교환으로 개선한다. 비용은 초기 root→상자 뒤 담당 구역 앞 직선 기준이며 장애물 우회·실제 낙상 방지를 보장하지 않는다.
- 최종 배치 직후 한 번 배정해 개인 목표·관측·접촉 구역을 함께 연결한다. 기존 random 추첨·물리 배치·RNG·4면 접촉 선택·모델/PTH 계약은 유지하고, 학습/resume·정식 eval_protocol의 비random 변경은 거부한다. config/USAGE/structure를 갱신했다.
- CPU77개 통과(신규17개). GPU6만 사용해 최신 정기11000 PTH로4명·8preset·4모드 및6/8/9명 전원 공동·4모드, 각8환경×16step의 strict 로드·정책/AMP finite·목표/관측·충돌·배정 고정·비정렬 부분 reset·가중치 보존을 확인했다. 모든 모드의 동일 seed 물리 장면·CPU/CUDA RNG가 bitwise 동일했다. ordered test wrapper의1환경32step headless 실행도 완료했다. 실제 GUI/VNC 화면·성공률/낙상 개선은 측정하지 않았다. [검증·재실행](output_etc/check/coupled_assignment_20261101/report.md). 기존 학습 프로세스는 유지했다.

### carry5 reset 배치 계산 최적화

- coupled 공용 배치 함수에서 distractor 한 개의 재시도 동안 경로·크기·사람 위치를 재사용하고, 충돌 경로 구성의 가상 구역 변환을 제거했다. 단독 PUSH 전용 검사는 실제 단독 행만 계산하며 전원 단독이면 추가 인덱싱 복사를 생략한다. 후보 생성/RNG·거리·충돌 여유·배치 순서·재시도 한도·모델 계약은 유지했다. 비PUSH 사람 장애물·공동 전체 상자·ON_TOP 받침/비정렬 환경 인덱스 회귀 테스트3개와 상세/구조 문서를 추가했다.
- GPU6만 사용해 carry5 1024환경을 기존→캐시→물리 조회→단독 선별4단계·19조건으로 비교했다. 배치/목표/관측/graph·CPU/CUDA RNG·재시도 횟수가 bitwise 동일하고, 부분 reset은 다른 환경을 보존했다.3seed 중앙값은 혼합0.595→0.461초(−22.5%), 전원 공동0.178→0.114초(−36.0%), 전원 단독 혼합1.520→1.006초(−33.8%), 전원 단독 PUSH10.383→8.466초(−18.5%),16환경 부분 reset0.116→0.081초(−30.6%)다. 공유 GPU의 reset wall time이며 학습 전체 FPS는 측정하지 않았다. [결과](output_etc/check/carry5_reset_optimizations_20261101/report.md), [실행부](output_etc/check/carry5_reset_optimizations_20261101/run_benchmark.sh).
- CPU86개 통과(기존21k 파일이 없는 전이1개 제외). GPU6·carry5 CHECK 저장본의32환경×8preset×16step 정책/AMP finite·부분 reset·가중치 보존을 확인했다. 전원 단독 PUSH seed2의18번 환경은 기존/변경 모두 후검사에서 경로 간격3.499985456m가3.5m 기준보다 약0.015mm 작게 나오는 기존 경계 사례였다. 조건을 완화하지 않고 별도 기록했으며 최적화 전후 판정도 같다. [검증](output_etc/check/carry5_reset_optimizations_20261101/verification.json), [경계 진단](output_etc/check/carry5_reset_optimizations_20261101/geometry_diagnosis.json). 기존 학습을 종료·재시작하지 않았다.

### 개인 목표 solo5 10000 distractor 제거 비교

- 실행 시 최신 정기10000epoch/frame1310720000을 복사 고정하고 GPU5·6에서 기본/제거 각각4명·8토큰·512환경×seed0/1/2·1200step을 평가했다. 같은 초기 장면에서 과제에 쓰이지 않는 여분 상자5,705개만 장면 밖으로 옮기고 기존 encoder의 object_valid로 관측을 가렸으며 ON_TOP 받침439개는 유지했다. 학습 코드·컨피그는 변경하지 않았다. [설정·재실행 명령](output_etc/check/solo5_10000_distractor_ablation512_seed012_20261101/manifest.json), [평가 전용 실행부](output_etc/check/solo5_10000_distractor_ablation512_seed012_20261101/ablation.py).
- 단독 AT 점 ever/final은81.1/70.7→84.4/72.2%, SIT 점은85.4/74.3→81.8/68.4%다. 공동2/3/4명 개인 목표 전원 동시 final은73.5/69.6/59.6→71.6/67.2/59.3%, 낙상은160→129/6144명이다. 점/영역·ever/final·공유 중심·표준편차·건수는 [보고서](output_etc/check/solo5_10000_distractor_ablation512_seed012_20261101/report.md), [비교 JSON](output_etc/check/solo5_10000_distractor_ablation512_seed012_20261101/comparison.json)에 저장했다.
- 양쪽 초기14개 tensor·유지한 물체/사람/목표의 bitwise 일치, 가린 토큰 값 변경 시 encoder 출력 차이0, 모델/PTH 해시 보존,6개 rollout CPU/GPU 재채점·건수 일치·finite·float32/64 차이0을 확인했다. 상자만 별도로 root setter에 제출한 초기 검사는 비정상 낙상으로 제외했다. 정상 전체 root/DOF 초기화를 재적용한 뒤8환경의 물리 제거 전용 검사에서 최초 사람 step이 기본과 bitwise 동일함을 확인하고 제거 조건을 전부 재실행했다. [검증](output_etc/check/solo5_10000_distractor_ablation512_seed012_20261101/verification.json). 기존 본학습 프로세스는 유지했다.

### 개인 목표 solo5 최신9500 정식 평가

- carry5의 본학습 저장본이 없어 현재 학습 중인 기존 solo5의 실행 시 최신 정기/일반9500epoch/frame1245184000을 확인하고 정기 파일을 복사 고정했다. GPU5(seed0)·GPU6(seed1/2),4명·8토큰·512환경×3seed·1200step 혼합 평가를 완료했다. AT/ON_TOP은 기존 solo5 목표 분포다. [실행 설정](output_etc/check/solo5_9500_eval512_seed012_20261101/manifest.json).
- 단독 PUSH의 ever/final seed 평균은79.39/76.00%, 공동2/3/4명 전원 개인 목표 ever는67.94/63.84/52.03%, final은65.76/62.16/49.68%다. 공유 중심 ever는70.12/68.80/61.34%, final은68.70/66.92/58.71%다.8500 대비 단독 PUSH는 상승하고 공동2/4명 전원 성공은 하락했다. 총1,536장면·6,144명·낙상181명이다. 단독 행동별·공동 인원별 ever/final·표준편차·건수는 [상세](output_etc/check/solo5_9500_eval512_seed012_20261101/breakdown.json),8500 대비는 [비교](output_etc/check/solo5_9500_eval512_seed012_20261101/comparison_8500.json)에 저장했다.
- 세 seed의 초기12개 tensor가8500 평가와 bitwise 동일하다. CPU/GPU·JSON 건수 일치·공유 중심 포함 float32/64 차이0·finite·원본/복사본 checkpoint 해시·기존 본학습 유지 확인. [검증](output_etc/check/solo5_9500_eval512_seed012_20261101/verification.json). 임시 스크립트의 checkpoint 파일명 중복을 바로잡아 재실행했고 모델·학습 설정·평가 코드는 변경하지 않았다.

### 개인 목표 solo5 carry5 별도 컨피그 추가

- 요청대로 `stage2_coupled_push_individual_goal_solo5_carry5` 환경/학습 YAML·train/test/VNC/eval을 추가했다. solo5에서 `solo_carry_goal_distance: [1.0, 5.0]`만 추가하여 단독 AT 목표점·ON_TOP 받침을 운반 상자 기준 거리 U(1,5)m·방향 U(0,2π)로 뽑는다. 사람↔목표 최소1m·높이·충돌/경로·재시도·그 외 거리/모델/보상/26k 전이는 유지하며 기존 solo5·PTH 계약·출력은 분리했다. config 명령 목록 맨 아래와 USAGE/구조/공동 상세/README를 갱신했다.
- CPU148개 통과(없는78k3개 skip·21k 전이1개 제외). GPU6·4명/1024환경 CHECK epoch2/frame262144·finite·actor encoder73 tensor와 입력 RMS 동결·action head6 tensor 갱신을 확인했다. 새 저장본의32환경8preset·16step 정책/AMP finite·개인 목표·거리·충돌·부분 reset·가중치 보존과 test wrapper의 headless4환경32step ON_TOP 재로드를 통과했다.
- GPU5·1024환경×10조건(혼합seed0/1/2·전원단독·전원독립 PUSH·AT·ON_TOP·SIT·CLIMB·전원공동), 총10,240장면 스트레스 검사에서 샘플링 소진0이었다. 모든 장면의 충돌·거리 검사를 통과했고 전원 AT/ON_TOP 각4096개 운반 거리도1–5m였다. 충돌 후보는 기존 재시도로 거부하며 거리/물리 조건을 완화하지 않았다.1024환경의 큰 월드 좌표 float32 오차를 반영해 검사 도구의 수치 허용오차만 조정했다. [스트레스](output_etc/check/coupled_solo5_carry5/stress/stress_validation.json).
- 일반 eval wrapper의8환경·seed0·1200step과 CPU 재채점에서 건수 일치·공유 중심 포함 float32/64 차이0·finite를 확인했다. [전체 검증](output_etc/check/coupled_solo5_carry5/verification.json). 기존 본학습은 유지했으며 새 본학습·실제 GUI/VNC·정규 성능 평가는 실행하지 않았다.

### 개인 목표 solo5 최신8500 기존4명 혼합 평가

- 실행 시 최신 정기/일반 PTH가 모두8500epoch/frame1114112000임을 확인하고 정기 저장본을 복사 고정했다. GPU5(seed0)·GPU6(seed1/2)에서 기존4명·8토큰 혼합 분포의512환경×3seed·1200step 첫 episode 평가를 완료했다. 확장6/8/10명 평가는 이번 실행에 포함하지 않았다. [실행 설정](output_etc/check/solo5_8500_eval512_seed012_20261101/manifest.json).
- 종료 시 점 성공률의 seed 평균은 단독 AT69.58·ON_TOP67.74·SIT65.43·CLIMB91.65·PUSH70.91%, 공동2/3/4명 전원 개인 목표69.02/63.12/56.13%·공유 중심72.36/66.24/60.39%다.2500 대비 단독 PUSH+11.11pp·SIT−12.85pp이며 공동 전원 개인 목표는 모두 개선됐다. 총1,536장면·6,144명·낙상152명이다. ever/final·표본 표준편차·성공/평가 건수는 [상세](output_etc/check/solo5_8500_eval512_seed012_20261101/breakdown.json), 이전 대비는 [비교](output_etc/check/solo5_8500_eval512_seed012_20261101/comparison_2500.json)에 저장했다.
- 세 seed의 초기 장면12개 tensor가2500 평가와 bitwise 동일하다. CPU/GPU·JSON 건수 일치·공유 중심 포함 float32/64 차이0·finite·입력 원본/복사본 해시·기존 본학습 유지 확인. [검증](output_etc/check/solo5_8500_eval512_seed012_20261101/verification.json). 코드와 학습 설정은 변경하지 않았다.

## 2026-10-31

### coupled 인원 상한·물체2배 조건 제거

- 요청대로 coupled 계열의6/8명 whitelist와4명 학습 제한을 제거하고 N≥1·M≥N으로 확장했다. 기본4명·8토큰은 유지하며, 추론은 양쪽 checkpoint 계약의 N/M/물리 후보 수만 정규화하고 resume은 해당 수를 그대로 검사한다.
- ON_TOP 받침이 M−N개를 넘는 장면만 그룹·과제를 다시 뽑고 가능한 장면은 보존한다. 받침의 논리/물리 슬롯을 압축 연결하며 M>2N의 여분 물체와 안 쓰는 물리 후보의 주차도 처리한다. 전원 ON_TOP 고정처럼 가능한 그래프가 없는 요청은 무한 재시도 없이 오류를 낸다.
- CPU81개 통과(없는21k 전이1개 제외). GPU5/6·2500 PTH로1/1·3/3·5/6·9/9·3/8 각4환경과3/4 독립8환경의 엄격 로드·16step finite·부분 reset·구역/목표/물리 매핑·가중치 보존을 확인했다.3/4에서는 압축 받침 ON_TOP도 검사했고,5명·5토큰 test wrapper의 headless64step을 완료했다. 기존4명 bank/RNG/물리 배정의 bitwise 일치·PTH 원본/복사본 해시·기존 학습 보존을 확인했다. [검증](output_etc/check/coupled_variable_counts/verification.json). 다른 인원의 본학습·실제 GUI/VNC·다수 seed 성능은 실행하지 않았다.
- config/USAGE/구조/공동 상세를 갱신하고 임의 N/M의 VNC 명령을 config에 저장했다.

### 기존4명 PTH로6명·8명 공동 PUSH 추론

- 요청대로 coupled 계열 viewer에서6명·12토큰/8명·16토큰을 허용했다. 물리 후보38/52개·상자 local X 폭/질량 N배·동일 구역/개인 목표·배치/접촉을 인원수에 연결했다. 학습은4명·8토큰을 유지하고, 추론 계약은 인원/토큰/물리 후보 수만 정규화하여 기존 모델·RMS를 그대로 로드한다. 공간 평가의 그룹별 집계도 인원수까지 확장했다.
- CPU59개 통과(없는21k 저장본 전이1개 제외). 기존4명 bank 크기/질량·RNG·물리 배정은 HEAD와 bitwise 동일하다. GPU5·2500 PTH·각4환경의6/8명 전원 공동에서 엄격 로드·16step finite·물리/거리/구역/개인 목표·부분 reset·가중치 보존 검사를 통과했다. test wrapper의 headless1환경 실행도6명1200step·8명64step 완료했고 PTH 원본/복사본 해시 보존을 확인했다. [검증](output_etc/check/coupled_6_8_inference/verification.json). 실제 GUI/VNC 화면과 다수 seed 성능은 확인하지 않았다.
- config에 GPU5/6의6/8명 VNC 명령을 저장하고 USAGE/구조/공동 상세를 갱신했다. 기존 본학습은 유지했다.

### 개인 목표 solo5 최신2500 정식512환경×3seed 평가

- 요청대로 실행 시 최신 정기2500 저장본을 복사 고정하고 GPU5(seed0)·GPU6(seed1/2)에서512환경×3회·1200step 첫 episode 정식 평가를 완료했다. 총1,536장면·6,144명이며 실행 인자·원본/복사본 SHA256은 [manifest](output_etc/check/solo5_2500_eval512_seed012_20261031/manifest.json)에 보존했다.
- 공동2/3/4명 개인 목표 전원 동시 ever의 seed 평균은62.67/57.24/49.40%, 공유 상자 중심 도달 ever는66.55/65.08/62.60%다. 그룹 수는778/421/257개이며 낙상179/6144명이다. 단독 행동별 점/영역·공동 인원별 개인/중심 ever/final의 seed 평균±표본 표준편차·총건수는 [상세](output_etc/check/solo5_2500_eval512_seed012_20261031/breakdown.json)에 저장했다.
- 세 trace의 CPU/GPU·JSON 건수 일치·공유 중심 포함 float32/64 차이0·finite·원본/복사본 checkpoint 해시 보존을 확인했다. [검증](output_etc/check/solo5_2500_eval512_seed012_20261031/verification.json). 기존 학습 프로세스는 유지했다.

### 개인 목표 solo5 1500 간단 평가

- GPU6·최신 정기1500 저장본을32환경·seed0·1200step 첫 episode1회로 평가했다. 공동2/3/4명별 개인 목표 전원 동시 성공은11/20·4/5·1/6, 실제 공유 상자 중심→공유 중심 목표 도달은12/20·4/5·4/6이다. 이번 공동 ever/final은 동일하다.
- 혼합 장면의 단독 점 ever는 AT10/12·ON_TOP8/12·SIT5/7·CLIMB3/3·PUSH11/15이며 낙상4/128명이다. 점/영역·final·개인 참여 결과는 [상세](output_etc/check/solo5_1500_eval32_seed0_20261031/breakdown.json)에 저장했다.
- CPU/GPU·JSON 건수 일치·공유 중심 포함 float32/64 차이0·finite·입력 checkpoint 해시 보존을 확인했다. [검증](output_etc/check/solo5_1500_eval32_seed0_20261031/verification.json). 기존 학습을 유지한 소표본 검사다.

### 개인 목표 eval에 공유 상자 중심 채점 추가

- 개인 목표 원본/solo5의 정식 eval에 `shared_center_scores`를 추가했다. 실제 공유 상자 중심→고정 공유 중심 목표3D 거리20cm·구성원 최소1명 유효 조건으로2/3/4명 그룹별 ever/final을 `result.json`·`summary.json`에 저장한다. 기존 개인 목표 전원 동시 성공·학습 보상은 유지한다.
- 새 trace에 실제 물리 중심·공유 목표·ever/final mask를 저장하고 CPU 재채점으로 검사한다. 이전 trace는 균등 분할 토큰/개인 목표 평균을 사용한다. 직전32환경 비교의 solo5 `result.json`·`summary.json`에도 중심 채점을 추가했으며 기존 값은 보존했다.
- CPU44개 통과. GPU6·500저장본·32환경·seed0·1200step 재실행에서 기존 trace12개 필드가 bitwise 동일하고 과제/개인 지표도 같다. 중심 ever/final은2명12/20·3명3/5·4명3/6으로 이전 재채점과 일치했다. CPU/GPU·JSON·float32/64 차이0·finite·checkpoint 해시 보존, 토큰 평균↔실제 중심 최대오차7.63e-6m를 확인했다. [검증](output_etc/check/solo5_shared_center_metric/verification.json). 평가 규약·USAGE·구조·공동 PUSH 상세를 갱신했다.

### goal5 7000·개인 목표 solo5 500 간단 비교 평가

- GPU5·6에서 각각 최신 정기7000/500 저장본을32환경·seed0·1200step 첫 episode1회로 평가했다. 혼합 장면의 단독 역할을 행동별로, 공동 PUSH를2/3/4명별로 나누고 ever/final을 저장했다. 공동은 설정 고유의 전원 동시 성공과 두 설정 공통의 실제 상자 중심→공유 중심 목표20cm 도달을 구분했다. 개인 목표 실험의 실제 중심/목표는 동일 공유 상자의 균등 분할 토큰/목표 평균으로 복원했다.
- 공동 중심 도달 ever는 goal5가2명2/24·3명1/7·4명2/9, solo5는12/20·3/5·3/6이다. solo5의 개인 목표 전원 동시 ever는11/20·1/5·3/6이다. 단독 PUSH는0/8·12/15, 낙상은3/128·4/128명이다. 다른 단독 행동의 점/영역과 final·개인 참여 결과는 [상세](output_etc/check/goal5_vs_individual_solo5_eval32_20261031/comparison.json)에 보존했다.
- CPU/GPU·JSON 건수 일치·float32/64 차이0·finite·두 입력 checkpoint SHA256 보존을 확인했다. [검증](output_etc/check/goal5_vs_individual_solo5_eval32_20261031/verification.json). 학습량·샘플링 분포가 다른 소표본 검사이며 기존 학습 프로세스는 유지했다.

### BEFORE near50 쌍 선택50·접근 최대5m 설정 추가

- `stage2_before_push_near50_pair50_source5` 환경·학습 YAML과 train/test/VNC/eval을 추가했다. 기존 near50에서 쌍 생성 확률0.5·일반 사람→상자1–5m·독립 PUSH 접근2–5m만 적용한다. near50·AT/ON_TOP 목표 배치·PUSH 이동2–10m·보상·모델·loco·4명/1024환경/8토큰/1200step·기본21k 전이는 유지하고 출력·저장본 계약을 분리했다.
- 1024환경 첫 검사에서 고정 주변 배치에 대한 PUSH 후보 소진을 확인했다. 새 설정에서 성공 환경을 보존하고 실패 환경만 기존 장면 재샘플링으로 넘기도록 연결했으며 거리·경로·충돌 조건과 기존 실험 기본값은 유지했다. 전원 독립 PUSH32환경 검사는 재샘플링에 수분이 소요됐다.
- 최종 CPU196개 통과(없는checkpoint4개 skip·21k 전이1개 제외). GPU6·1024환경 CHECK epoch2/frame262144 저장·finite·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신을 확인했다. 32환경7preset의 거리·near·alias·PUSH 경로·부분 reset과4환경32step headless 재로드를 통과했다. AT 목표 최대14.04m·BEFORE ON_TOP 운반 최대9.08m를 관측해 별도 목표 거리 상한을 추가하지 않았음을 확인했다.
- 일반8환경·seed0·1200step 검사와 CPU 재채점의 건수 일치·float32/64 차이0·finite를 확인했다. 기본21k 파일이 없어 검증과 문서 학습 명령에는26k를 명시했다. [전체 검증](output_etc/check/before_pair50_source5/verification.json). config/structure/USAGE/README·BEFORE 상세를 갱신했다. 본학습·GUI/VNC 실화면·정규512환경×3seed 성능 평가는 실행하지 않았다.

### 개인 목표 solo5를 별도 컨피그로 분리

- 요청에 따라 `stage2_coupled_push_individual_goal`의 단독 거리 상한 변경만 되돌렸다. 공동 확률0.5와 개인 목표·접촉 설정은 유지한다. 단독 접근/PUSH 이동 최대5m는 새 `stage2_coupled_push_individual_goal_solo5` 환경·학습 YAML과 train/test/VNC/eval로 분리하고 출력·저장본 계약도 구분했다. 단독 CARRY AT 목표 높이 보존 수정은 유지한다.
- CPU71개 통과(기존21k 전이1개 제외), 원본/변형 환경 차이는 단독 상한 두 값뿐임을 확인했다. GPU6에서 기존5m 검증 저장본을 새 wrapper로 재로딩하여4환경32step 실행을 완료했다. [CPU 로그](output_etc/check/coupled_individual_solo5_split/cpu.log), [실행 로그](output_etc/check/coupled_individual_solo5_split/reload.log). shell 구문·diff 검사 통과. 본학습은 실행·재시작하지 않았다.

### 개인 목표 실험의 단독 접근·PUSH 이동 최대5m

- 새 개인 목표 컨피그에 `solo_source_max_distance: 5.0`·`solo_goal_max_distance: 5.0`을 추가했다. SIT/CLIMB/CARRY AT/ON_TOP의 사람→상자는1–5m, 단독 PUSH 접근·이동은2–5m다. 공동 접근2–10m·이동2–5m와 CARRY 목표 XY 샘플링은 유지한다. 공통 샘플러의 기본값은 유지하고 현재 설정에만 상한을 전달한다.
- 개인 목표 변환을 공동 구성원에게만 적용하여 단독 CARRY AT의 목표 높이 덮어쓰기를 수정했다. 기존 검사의 목표 벡터 비교를 PUSH에 한정하고 단독 접근 범위·AT 목표/받침 높이 검사를 추가했다.
- CPU53개 통과(기존21k 전이1개 제외). GPU6·확률0.5·1024환경 CHECK epoch2/frame262144 저장, finite·encoder/RMS 동결·head 갱신 확인.32환경8preset의 거리·고정 목표·부분 reset 검사를 통과했고 holding_at의 높은 목표61개를 포함해 받침과 목표 높이 일치를 확인했다. [검증](output_etc/check/coupled_individual_solo5/verification.json). 기존 본학습은 재시작하지 않았다.

### 개인 목표 실험의 공동 선택 확률50%

- `stage2_coupled_push_individual_goal`의 `coupled_probability`를0.8→0.5로 변경했다. 그룹 선택 확률 기준이며4명 기대 인원 비율은 공동63.54%·단독36.46%다. 확률 validator와 설정 비교 검사를 갱신했고 기존 coupled/goal5는0.8을 유지한다.
- 관련 CPU44개 통과(기존21k 전이1개 제외), diff 검사 통과. [로그](output_etc/check/coupled_individual_goal/probability50_cpu.log). 아래 GPU 검증은 변경 전0.8 설정의 결과이며 이번에는 학습을 실행·재시작하지 않았다.

### 개인 목표·공통 4면 coupled 실험 연결

- `stage2_coupled_push_individual_goal` 환경/학습 YAML·train/test/VNC/eval을 추가했다. 공동2–5m 이동 벡터를 초기 분할 토큰마다 적용하고 개인 AT 연결·보상·도착 예외·현재 성공 포화를 사용한다. 공유 중심 목표는 물리 경로와 공통 외곽4면 선택에만 사용한다. 목표 방향60° 후보 중 담당 구역까지 그룹 최대 손 거리가 가장 작은 면을 선택하고 전원 gate도 같은 면으로 검사한다. 별도 `aligned` 조건은 제거했다. 기존 coupled/goal5·독립 행동은 유지하고 저장본 계약을 분리했다.
- CPU83개 통과(없는 기본21k 저장본을 요구하는 기존 전이1개 제외). GPU6·1024환경 CHECK epoch2/frame262144 저장·finite·actor encoder73 tensor/입력 RMS 동결·action head6 tensor 갱신 확인.32환경8preset의 거리/개인 목표 연결·고정 목표·부분 reset·정책/AMP finite 검사와4환경32step 실제 렌더를 통과했다. [학습 검사](output_etc/check/coupled_individual_goal/training_validation.json), [장면 검사](output_etc/check/coupled_individual_goal/scene/physics_validation.json), [목표 표시](output_etc/check/coupled_individual_goal/viewer/frame_32.png).
- GPU6·32환경·seed0·1200step 평가를 완료하고 개인/그룹 동시 성공의 ever/final 집계를 연결했다. CPU 재채점·JSON 건수 일치, float32/64 개인/BEFORE 지표 차이0·상태 finite·입력 checkpoint 해시 보존을 확인했다. [재채점](output_etc/check/coupled_individual_goal/eval/general_seed0/rescore.json). config/structure/USAGE/README·상세 설계를 갱신했다. 본학습은 시작하지 않았으며 기존 학습 프로세스는 유지했다.

### PUSH·CARRY AT 목표구 반지름20cm 통일

- 요청에 따라 모든 설정의 AT 목표구를 생성 시 반지름20cm로 통일하고 PUSH 와이어 구를25cm에서20cm로 줄였다. 바닥 링은35cm로 유지하며 reset 중 scale 변경은 추가하지 않았다.
- URDF 시각 반지름0.1m×생성 scale2.0=0.2m·Python 구문·diff 검사를 통과했다. 이번 크기 조정의 viewer 실행 검증은 하지 않았다.

### PUSH viewer 표시 롤백 및 선 표시 재구현

- 사용자가 viewer에서 물체가 날아가는 비정상 동작을 보고해 직전 PUSH 표시 이식 전체를 롤백했다. reset 시 marker scale을 변경하는 코드·표시 모듈·호출부를 제거하고 기존 목표구 표시와 문서를 복구했다. 짧은 관측/렌더 검사만으로 정상 동작을 판단한 기존 검증은 불충분했다. 기존 학습·컨피그 변경은 보존했다.
- 이후 기존 목표구·actor·scale·reset을 유지하고 `add_lines`로 PUSH 목표의 바닥 링·세로선·외곽 와이어 구만 덧그리도록 다시 구현했다. 공유 목표 선은 노란색이며 일반 AT에는 추가 선이 없다. 공통 reset 파일은 변경 전과 동일하다.
- GPU5의 실제 viewer에서5500 정책·4환경·동일 seed의 기존/선 표시 실행을 각각1200step 비교했다. 전체 actor root state·정책 관측·행동이 bitwise 동일하고 최대 차이0·finite다. [궤적 비교](output_etc/check/push_goal_viewer_safe/trajectory_comparison.json). 일반 AT·독립 PUSH·2명 두 그룹·4명 공동의 선 표시만 호출하는 것도 확인했다. [표시 검사](output_etc/check/push_goal_viewer_safe/draw_only_verification.json).

### Coupled goal5 5500 간단 평가

- 최신 정기5500 저장본을 GPU5·32환경·seed0·1200step 첫 episode1회로 평가했다. 공동 PUSH의 실제 상자 목표 도달은2명6/24(25.0%)·3명1/7(14.3%)·4명1/9(11.1%)다. 혼합 장면의 독립 역할은 SIT1/1·CLIMB2/2·AT5/6·ON_TOP5/6·PUSH1/8이며 점/영역 결과는 같다. 사람 번호별 공동 참여 성공은 별도 JSON에 저장했다. [상세](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/breakdown.json).
- CPU 재채점과 GPU·JSON 건수가 일치하고 float32/64 성공 차이0·상태 finite·입력 checkpoint 해시 보존을 확인했다. 낙상1/128명이다. 소표본 검사이며 본학습 프로세스는 유지했다. [검증](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/verification.json).
- 같은 궤적의1200step final은 공동2명4/24·3명0/7·4명0/9, 독립 AT4/6이며 나머지 독립 행동은 ever와 같다. 이전 낙상자를 제외하고 final이 ever에 포함됨을 확인했다. [final 상세](output_etc/check/coupled_goal5_5500_eval32_seed0_20261031/final_breakdown.json).

## 2026-10-30

### 공동 PUSH goal5 컨피그 추가

- `stage2_coupled_push_goal5` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. 공동 PUSH는 사람→실제 상자 중심2–10m·상자→목표2–5m, 독립 PUSH는 두 거리2–10m다. 기존 coupled YAML은 원래 설정으로 복구했으며 사람 초기 배치·그 외 환경/학습 값은 유지했다. 새 실험의 기본 전이는 제공26k이고 출력·저장본 계약은 분리했다.
- CPU28개 통과(공통 모델26·새 설정 로딩2, 기본21k 파일이 필요한 기존 전이1개 제외). GPU5의1024환경 CHECK가 epoch2/frame262144 저장 후 정상 종료했다. 모델/정규화/optimizer finite·actor encoder73 tensor와 입력 RMS 동결·action head6 tensor 갱신을 확인했다. [학습 검사](output_etc/check/coupled_goal5/training_validation.json).
- 전용 test wrapper로 저장본을4환경·32step·4명 공동 PUSH로 headless 재로드했다. 셸/Python 구문·새 YAML/문서 경로를 확인하고 config/structure/USAGE/README·공동 PUSH 상세를 갱신했다. 장기 본학습·수렴 성능·실제 GUI/VNC 화면은 검증하지 않았다.

### 모션 데이터 연결 복구

- loco·sit·climb·carry 모션은 `/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/`의 각 원본으로, PUSH 모션은 `/home/hwanhee/ksh/stage1_2_jointpush/tokenhsi/data/jointpush_motions`로 심링크를 연결했다. 외부 원본은 수정하지 않았다.
- 현재 PUSH 모션 YAML의 파일 참조161개가 모두 존재함을 확인했다. 학습은 실행하지 않았다.

### 공동 PUSH 거리2–5m 분석

- 공동 PUSH의 두 XY 거리를2–5m로 줄이는 안을 검사했다.32환경의41개 공동 그룹에 각각104만 후보를 검사하고, 실패4환경은 위치·회전 구간 검사를 통해 고정된 사람·상자 크기에서 거리/뒤쪽/여유 조건을 동시에 만족할 수 없음을 확인했다. [분석](output_etc/check/coupled_distance_analysis/summary.json).
- 공통 Stage2 CPU23개 통과(기본21k 파일이 필요한 전이1개 제외). 제공26k의165 tensor 전이와 입력 RMS 로드를 확인했다. 이후 사람→상자2–10m·상자→목표2–5m 조합은32환경의 혼합/4명 공동/독립 PUSH reset·짧은 시뮬레이션을 통과했다. 사람 초기 배치는 변경하지 않았고 기존 coupled YAML은 두 거리2–10m로 복구했다.
- 최초 검증은 GPU1에서 실행 후 종료했다. 이후 사용자 지시에 따라 GPU5·6만 사용하며 GPU5에서 같은 배치 오류를 재현했다. [로그](output_etc/check/coupled_distance_2to5/scene_gpu5.log), [초기 배치](output_etc/check/coupled_distance_2to5/scene/reset_failure.json).

## 2026-10-29

### PUSH viewer 목표구 확대

- PUSH 포함 설정의 viewer/VNC/영상 목표구를 반지름3cm에서20cm로 확대했다(지름 약6.7배). 공통 marker 생성부의 표시 scale만 변경했으며 목표 좌표·성공 판정·학습 설정과 비PUSH 설정의 marker 크기는 유지했다.
- coupled1환경의 동일 카메라에서 기존/확대 목표구를 offscreen 렌더링해 가시성을 확인했다. marker scale 변경 전후 정책 관측이 bitwise 동일하고 짧은 시뮬레이션 관측/보상 finite를 확인했다. [확대 화면](output_etc/check/push_goal_marker/render/after.png), [검증](output_etc/check/push_goal_marker/render/verification.json). USAGE에 표시 크기를 기록했다.


### PUSH26k 평가와 Stage2 제로샷 검증

- `pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth`를 검사했다.5과제/4family·165 tensor·모델/RMS finite를 확인했고 두 입력 PTH의 실행 전후 SHA256은 동일하다. 제공78k는 기존 기준78k와 해시가 같고 새 pth 경로의4환경32step headless 재로드를 완료했다.
- 26k Stage1 PUSH는512환경×3seed·900step 일반3회/1명 행동별 단독15회를 완료했다. 점/영역5과제 macro는 일반 **84.55/88.32%**, 단독 **82.20/86.95%**다. PUSH 점 성공률은 일반87.12%·단독88.61%로21k보다 높지만 다른 과제의 하락으로 전체 점 macro는 낮았다.21k와18개 초기 장면의9개 필드가 bitwise 일치한다.
- near50·coupled는26k tensor165개를 그대로 복사하고 공통 coordination identity/Linear bias0인 제로샷 저장본을 각각 만들었다. 추가 학습0회·두 모델 tensor 동일을 확인했다. 각16환경·4명·1200step 실행과4환경32step headless viewer 재로드를 완료했다. BEFORE ON_TOP 쌍/4명 공동 PUSH viewer 명령을 USAGE에 추가했다. 관찰용 파일에는 optimizer가 없고 기본 YAML 전이21k는 변경하지 않았다.
- 총20개 trace의 CPU/GPU·JSON 건수 일치, float32/64 ever/simultaneous 차이0·finite를 확인했다. [리포트](output_etc/check/push_26k/report.md), [검증](output_etc/check/push_26k/verification.json). Stage2 결과는 작은 제로샷 실행 검사이며 학습된 Stage2 성능·실제 GUI 화면 검사는 아니다. 기존 원본/프로세스는 보존했다.


### BEFORE PUSH near50 거리 변형 추가

- `stage2_before_push_near50` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. `stage2_before_push`의 mix226 대신 참고 설정의 `relative_sum` near50 거리만 적용했다. 기존 모든 YAML은 보존하고 공통 모델·PUSH/독립 행동 비율·4명/1024환경/8토큰/1200step·loco·21k 전이는 유지했다. RSI433과 attention mask는 이식하지 않았다.
- near 모드는 BEFORE 쌍의50%이며 SIT/CLIMB는 후행→받침 거리가 선행→받침보다 작고, ON_TOP은 후행→받침+후행→payload 거리 합이 선행→받침보다 작다. 나머지50%는 original 샘플링이다. validator·viewer/eval의 새 이름/모션/task 경로·checkpoint 혼용 거부를 연결했다.
- CPU313개 통과.1024환경 CHECK epoch2/frame262144 저장·모델/RMS/optimizer finite·actor encoder73 tensor/입력 RMS 동결을 확인했다.32환경5preset에서 near 부등식·PUSH 경로/물리·alias 관측·부분 reset·가중치 불변을 확인했고4환경32step headless viewer와16환경×seed0·1200step 일반 평가를 완료했다. CPU/GPU·JSON 건수 일치, float32/64 ever 차이0·finite다. [검증](output_etc/check/before_push_near50/verification.json).
- config/structure/USAGE/README·BEFORE PUSH 상세에 설정과 명령을 갱신했다. 장기 본학습·512환경×3seed 성능·GUI/VNC 실화면은 실행하지 않았다.


### 공통 Stage2 alias 전처리와 coordination 초기화 수정

- 공통 `Stage2PushBuilder`의 alias 처리를 config 플래그 대신 task packet의 BEFORE 의존 관계로 결정하도록 통일했다. encoder용 복사본만 변환하고 복제 물체 token feature/pose·원본 packet·coordination relation은 보존한다. coupled·독립 packet은 그대로 통과한다. 기존4과제 BEFORE 구현은 유지했다.
- 새 coordination의 모든 Linear bias를 원본처럼0으로 초기화했다. 초기 출력 identity와 Stage1 PUSH165 tensor 전이는 유지하며 기존 학습 PTH 로드 시 저장된 bias를 덮어쓰지 않는다. coupled 제로샷 PTH를 같은 경로에 새 초기화로 재생성하고 이전 관찰용 저장본은 `output_etc/check/stage2_common_fix/previous_zeroshot.pth`에 보존했다.
- CPU305개 통과. BEFORE SIT/CLIMB/ON_TOP·두 쌍·coupled·solo의 같은 가중치/입력에서 config에 무관하게 actor/critic 출력 bitwise 일치, 기존 encoder 동작·복제 feature/pose·원본 packet 보존을 확인했다. 실제 GPU BEFORE 관측과 기존 학습 가중치의 교차 실행도 action/critic 최대 차이0이다.
- 두 설정 모두4명·1024환경 CHECK epoch2/frame262144 저장, 모델/RMS/optimizer finite·actor encoder73 tensor/입력 RMS 동결을 확인했다. BEFORE32환경5preset의 alias 관측/부분 reset/finite 검사와 재생성 제로샷4환경·32step headless 재로드를 완료했다. 이전/새 제로샷의 coupled·solo 출력은 동일하다. [검증](output_etc/check/stage2_common_fix/verification.json). 구조/설정 상세 문서를 갱신했고 장기 성능·실제 GUI 화면은 검증하지 않았다.


### Coupled 제로샷 viewer 저장본

- 제공 Stage1 PUSH21k의 기존165 tensor를 그대로 복사하고 공통 Stage2 coordination을 identity로 초기화한 `output_etc/check/coupled_zeroshot/Stage2CoupledPush_ZeroShot_Stage1_21000.pth`를 만들었다. 추가 optimizer step0·tensor 일치·identity 초기화를 확인했다. 원본 PTH는 보존했다.
- `TASK_GRAPH=all_coupled`로4환경·32step headless test wrapper 재로드를 완료했다. 실제 GUI 화면은 실행하지 않았다. 제로샷 viewer 명령을 USAGE에 추가했고 전이 기록은 `output_etc/check/coupled_zeroshot/transfer.json`에 있다.


### 공통 Stage2 PUSH 모델과 coupled 이식

- `stage2_coupled_push` 환경/학습 YAML·train/test/VNC/일반 eval을 추가했다. 원격 `stage1_2_jointpush_st`의 coupled straight·밀도3·인원별 공동 상자/구역·그룹 gate·goal saturation을 이식했다. 독립 행동과 AMP 혼합은 현재 Stage1 PUSH 설정(AT25/ON_TOP20/SIT10/CLIMB15/PUSH30)을 읽도록 바꾸고 미사용 Stage2 RSI/adapter 스택은 가져오지 않았다.
- `stage2_before_push`와 coupled는 동일 `Stage2PushBuilder`·관측1272/AMP1330·5과제·NONE/SELF/BEFORE/COUPLED4종 사람 relation·6N task/group suffix를 사용한다. tensor 이름·크기가 같고 Stage1 PUSH21k의165 tensor를 그대로 전이한다. 기존 BEFORE PUSH의5N 관측 검사 저장본은 새 모델에 resume하지 않는다.
- 사용자 기준대로 Stage1은900step을 유지하고 모든 Stage2 학습/일반 평가/viewer 기본 시간을1200step으로 변경했다. 기존4과제 BEFORE 모델은 보존했다. coupled 평가의 PUSH 성공은 가상 구역이 아닌 실제 전체 상자 중심에서 계산한다.
- CPU292개 통과. 두 Stage2 모두4명·1024환경 CHECK가 epoch2/frame262144 저장 후 종료했고 coupled resume은 epoch3/frame393216까지 완료했다. actor encoder73 tensor/입력 RMS 동결·action head6 tensor 갱신·모델/RMS/optimizer finite를 확인했다.
- coupled32환경8preset·BEFORE32환경5preset의 배치/구역/AMP/부분 reset/가중치 불변을 확인했다. 두 설정 모두 일반32환경×seed0·1200step 및 headless viewer4환경·32step 재로드를 완료했다. CPU/GPU·JSON 건수 일치, float32/64 ever/simultaneous 차이0·finite다. [검증 보고](output_etc/check/coupled_integration/report.md), [전체 결과](output_etc/check/coupled_integration/verification.json).
- config/structure/USAGE/README·BEFORE 상세와 [공통 Stage2/coupled 상세](markdowns/reference/stage2_coupled_push.md)를 갱신했다. Stage1 YAML·원본 checkout·외부 PTH·기존 프로세스는 보존했다. 장기 본학습·정규512환경×3seed 성능·GUI/VNC 실화면은 검증하지 않았다.


### Stage2 BEFORE solo PUSH 추가

- `stage2_before_push` 환경/학습 YAML과 전용 train/test/VNC/일반 eval을 추가했다. 독립 역할만 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%를 사용한다. 기존 0.8 쌍 구성·선행 AT·크기 호환 후행 SIT/CLIMB/ON_TOP 균등 선택·mix226·공유 alias·AT gate/progress는 유지한다. 기본4명·1024환경·8토큰·900step·모두 loco다.
- PUSH 전용 큰 상자 bank를 추가해 기본20개 물리 후보에서 배정하며 일반 distractor/BEFORE 받침으로 섞이지 않게 했다. Stage1 PUSH의 밀도3·167종·cone90·거리·접촉 가중치를 적용하고 다중 사람 경로/활성 장애물 검사를 추가했다. 정책5과제·AMP4family를 기존 BEFORE coordination에 연결해 기본21k PUSH에서 tensor165개를 그대로 전이한다. 기존5개 환경/학습 YAML은 변경하지 않았다.
- CPU280개 통과. 최종 CHECK1024환경은 epoch2/frame262144, resume은 epoch3/frame393216 저장을 완료했다. actor encoder73 tensor·입력 RMS 동결, action head6 tensor 갱신, 모델/정규화/optimizer finite를 확인했다. [학습·재개](output_etc/check/before_push/training_validation.json).
- 32환경7개 preset에서 물체 alias/숨김·거리·밀도/마찰·PUSH 경로·AT progress·AMP·부분 reset·가중치 불변을 확인했다. 최종 저장본 일반32환경×seed0·900step과 headless viewer4환경·32step 재로드를 완료했다. 평가 CPU/GPU/JSON 건수 일치·float32/64 ever 차이0·finite를 확인했다. [검증 보고](output_etc/check/before_push/report.md), [전체 검증](output_etc/check/before_push/verification.json). 이는 실행 검사이며 장기 학습 성능·512환경×3seed 평가·GUI/VNC 실화면은 검증하지 않았다.
- config/structure/USAGE/README와 [설정 상세](markdowns/reference/stage2_before_push.md)를 갱신했다. 원본 checkout·외부 checkpoint·기존 실행 프로세스는 보존했다.


### Stage1 PUSH 행동 비율 변경

- 사용자 지정대로 `stage1_push`의 새 학습 크기/행동 prior를 AT25/ON_TOP20/SIT10/CLIMB15/PUSH30%로 변경했다. 검증기의 허용 계약·샘플링 테스트·현재 PUSH/USAGE/평가 문서를 갱신했다. 과거 저장본의 분포와 평가 결과는 보존하며21k PTH의 기본 viewer/일반 평가는 저장된PUSH50%를 복원한다.
- CPU35개 통과:8만 사람 슬롯의 크기 조건부 샘플링 빈도·합계1·설정 검증·이전 저장본 복원/호환. [검증](output_etc/check/push_sampling_30/tests.log), [설정](output_etc/check/push_sampling_30/config_check.json). 비율 변경 검증에 GPU 학습/평가는 실행하지 않았다.


### Stage1 PUSH 통합과900step 통일

- `stage1_push` 환경/학습 YAML·train/test/VNC·일반/1명 단독 eval을 추가했다. scratch prior는 SIT7.5/CLIMB18.75/AT24.375/ON_TOP24.375/PUSH25%로 기존 네 과제 상대 비율을 유지한다. 정책은5과제·AMP4family로 확장하고 기존4과제/3family 기본값은 보존했다. PUSH167종·밀도3·cone90·접촉 RSI bank/모션을 이식했으며 모션21개와21k PTH는 외부 원본에 심링크로 연결했다.
- 학습·일반/단독 평가·viewer 기본 시간을900step으로 통일하고 시간 종료의 `-1` 경계를 수정했다. 새5과제 macro와 기존4과제 macro를 구분한다. PTH 옆 sidecar가 없으면 내부 저장 env/sim을 읽는다. 제공21k 일반 평가는 학습 당시 PUSH50% 분포를 복원하고 새 scratch는PUSH25%다. 서로 다른 학습 계약의 resume는 거부한다.1명 PUSH와1명 ON_TOP의 물체/받침 인덱스, 여섯 번째 RSI 통계 축을 일반화했다.
- 제공 `Stage1JointPushDensity3Cone90Finetune_00021000.pth`로512환경×3seed 일반3회·단독15회, 모두900step을 완료했다.5과제 macro 점/영역은 일반 **85.84±1.07 / 88.80±0.81%**, 단독 **83.32±0.51 / 88.78±0.32%**다. PUSH는 일반 **81.78±1.93%**, 단독 **81.51±1.47%**다.18개 trace CPU/GPU·JSON 건수 일치, float32/64 ever 차이0, 상태 finite를 확인했다. [리포트](output_etc/check/push_900/report.md), [검증](output_etc/check/push_900/verification.json).
- 이전78k1200step trace의 앞900frame을 별도로 재채점해 같은 제한 시간의 비교를 추가했다. 기존 원본 trace/보고서는 보존했다. 기존 네 과제의 단독 평균은21k에서 낮아졌으며 수치와 비교 조건은 리포트에 기록했다.
- CPU259개 통과. 실제78k→PUSH 전이 후 기존 네 과제 actor/critic 출력 보존, 원본 브랜치 정책·task encoder·clean scene 관측 코드 일치를 확인했다. Stage1/BEFORE 기존 PTH4환경·32step 재로드도 통과했다. PUSH scratch CHECK는2048환경으로 RSI5517 profile 준비 후 epoch2/frame262144를 저장했고 resume은 epoch3/frame393216까지 진행했다. 모델/RMS finite, scratch 시작·PUSH25%·900step 계약을 확인했다. [학습/재개](output_etc/check/push_900/training_validation.json). 원본 PTH/코드/기존 프로세스는 보존했으며 장기 본학습·GUI/VNC 실화면은 실행하지 않았다.


### 일반 평가와 Stage1 전용 단독 평가 분리

- 일반 `*_eval.sh`는 checkpoint 저장본의 학습 인원·물체·크기·과제/BEFORE 배치를 유지한다. Stage1만 `stage1_single_eval.sh`로1명·행동별 독립 평가를 제공한다. 단독 AT만 기존9종 크기, 나머지는 학습 크기/동적 물리를 따른다. viewer native 성공 진단은 `viewer_diagnostics.py`로 이름과 결과 태그를 분리했다.
- 정식 평가는 loco·정확히1200step·독립 seed0/1/2×512환경 첫 episode, 같은 궤적의 점/영역 ever·seed 평균/표본 std·macro다. 개인 낙상 frame까지 채점하고 해당 관절 drive를 해제하며 다른 사람은 계속 수행한다. BEFORE의 개인/follower 개인·둘 다 ever·동시 성공을 별도 집계한다. 학습 reward/종료 조건은 유지했다. viewer 기본 horizon도1200으로 맞췄다.
- Stage1 원본78000으로 일반3회(2명·4물체), 단독12회(행동별1명)를 완료했다. macro 점/영역은 일반 **95.17±0.84 / 97.17±0.18%**, 단독 **96.58±0.26 / 98.83±0.40%**다. [리포트](output_etc/check/eval_unified/stage1_78k_report.md). 15개 trace CPU float64/GPU·JSON 건수 및 float32/64 ever가 전부 일치했고 상태 finite·원본 checkpoint 해시 보존을 확인했다.
- CPU242개 통과. BEFORE 세 설정은 검사 저장본으로16환경·4명·1200step 실행/재채점(낙상8/10/15명, 각각24pair), viewer는4환경·32step의 새 진단 경로를 확인했다. Stage2 본학습 성능과 GUI/VNC 화면은 평가하지 않았다. [검증](output_etc/check/eval_unified/verification.json), [BEFORE](output_etc/check/eval_unified/before_validation.json). 새 규약·명령·코드 위치를 evaluation_single/config/structure/USAGE에 갱신하고 이전 단독 규약은 legacy에 보존했다.


### 네 실험 이름 통일

- 환경/학습 YAML·전용 train/test/VNC12개·실험/출력 이름·테스트·현재 문서를 `stage1`, `stage2_before_nonemask_rsi433_near50`, `stage2_before`, `stage2_before_rsi433`로 통일했다. Stage1 보조 구현은 `stage1_config.py`, `stage1_geometry.py`로 바꿨으며 기본 전이 링크는 `checkpoints/Stage1_00078000.pth`다. 외부 원본 파일과 과거 결과 경로는 보존했다.
- 기존78000 및 이전 BEFORE checkpoint의 reward 태그·RSI 경로를 읽기 경계에서 변환한다. 보상·관측·모델 tensor·실험 간 혼용 거부는 유지한다. 기존 네 저장본을 새 test wrapper로4환경·32step headless 재로드해 정상 종료했고 네 모델의 전체 tensor·actor/critic 출력은 변경 전과 bitwise 동일했다. [재로드](output_etc/check/config_rename/smoke.json), [모델](output_etc/check/config_rename/model_equivalence.log). CPU224개 통과. 셸/Python/YAML·문서 링크·이전 실행 이름 잔여 검사 결과는 `output_etc/check/config_rename/`에 있다. GUI/VNC 실화면과 본학습은 실행하지 않았다.

### 네 컨피그와 학습·추론 경로 정리

- `stage1_v2_fixed`, `stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200`, `stage2_before_ground_mix226_atprogress_objalias_1200`, `stage2_before_ground_mix226_atprogress_objalias_rsi433_1200`만 유지했다. 나머지21쌍 YAML·63개 wrapper와 공동 운반·공유 받침·adapter·full RSI·carry50·단일인 태스크·전처리·미사용 에셋/모듈을 제거했다. 기존 파일322개를 삭제했으며 과거 문서는 legacy에 보존했다. [변경 목록](output_etc/check/four_config_cleanup/change_manifest.json).
- BEFORE의 공통 기하·배치·물리 검사·native 후행 RSI를 분리하고 모델을 공통 정책의 직접 확장으로 정리했다. 공통 환경의 관측·reset·모션/AMP를 별도 mixin으로 분리했다. 파라미터 이름·관측·보상·거리·초기화·동결/학습 범위를 유지했다.
- 전용12개 wrapper는 공통 `launch.sh`를 사용한다. conda 기본값은 모두 `tokenhsi`, seed는42, test 기본은 viewer다. GPU는 한 번 지정하며 기존 MPS만 연결한다. 설정 로더는 새 학습의 native YAML과 추론/resume의 저장 env/sim을 같은 경로에서 처리한다. `ENV_CONFIG`로 명시적 대체가 가능하며 fixed 임시 YAML 생성·실험 이름별 중복 분기를 제거했다.
- Stage1 78000은 `checkpoints/Stage1V2Fixed_00078000.pth` 심링크로 연결했다. 네 모션 데이터 디렉터리도 공유 원본에 연결했다. 원본 checkout·가중치·기존 학습 PID3432492는 보존했다. 확인 산출물·RSI 캐시·pytest 임시 파일은 `output_etc/check/`에 둔다.
- CPU214개 통과: 네 설정/계약 혼용 거부·실제78000 전이·설정 복원·wrapper/conda 선택·GTA·보상·거리·mask·RSI/fallback·재로드. [CPU](output_etc/check/four_config_cleanup/cpu.log). 네 모델의 전체 tensor·actor/critic 출력·계약이 정리 전과 bitwise 동일했다. [모델](output_etc/check/four_config_cleanup/model_equivalence.log). 네 설정의 전체 reset2회·부분 reset1회에서 관측·사람/상자 상태·크기·목표·물체 연결의 최대 차이0을 확인했다. [reset](output_etc/check/four_config_cleanup/reset_equivalence.json). 후행 RSI sampler의 출력·RNG도 동일했다.
- GPU0/MPS에서 fixed2명·2048환경과 BEFORE4명·1024환경의 CHECK가 각각 epoch2/frame262144 저장 후 종료했다. fixed native RSI6413 profile을 생성했다. BEFORE 세 설정은 source78000·actor encoder73 tensor/입력 RMS 동결·head6 tensor 갱신·모델/scalar finite를 확인했다. mix226 resume은 epoch3 저장 후 종료했다. [학습/재개](output_etc/check/four_config_cleanup/training_validation.json).
- 실제78000 fixed headless 추론과 BEFORE 전용 test의4환경·32step 저장본 재로드를 확인했다. BEFORE32환경의 loco/mix·ON_TOP pickup/carry·단독 등14개 구성에서 alias 관측/pose·활성 물리 수·거리·AT progress·AMP·부분 reset·가중치 불변을 확인했다. [물리 검사 진행 결과](output_etc/check/four_config_cleanup/physics.log), [wrapper](output_etc/check/four_config_cleanup/wrapper_validation.json). 셸/Python/YAML·현재 문서 링크·4쌍 YAML/12개 wrapper·미사용 import/실험 참조 제거를 검사했다. VNC 실행 의존성이 없어 실제 GUI/VNC 화면과 장기 성능은 검증하지 않았다. 본학습은 시작하지 않았다.

## 2026-10-28

- 코드 정리용 독립 복사본을 만들었다. 원본과 복사 범위·초기 검증은 `code_copy_manifest.json`과 [당시 changelog](markdowns/legacy/pre_four_configs/changelog.md)에 보존한다.
