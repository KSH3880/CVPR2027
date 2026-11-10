## 2026-11-10

### TokenHSI* Table 1·2 다인 평가 시작

사용자 지정 `TokenHSIPushKD_00031000.pth`(SHA256 `413e77b4d35d5543211d30a52114ca3f47cc8258deacebcc203a0ea333d78542`)를 공통 SCT 물리 장면에서 평가한다. `/home/hwanhee/ksh/stage1_2_base_teacher`는 읽기 전용으로 참조하고 코드 복사본·실행기·모든 결과를 `output_etc/check/tokenhsi31k_tables12_20261110/`에 둔다. Table1 Independent/Mixed × 2/4/8/16H, Table2 Enable3종/Coupled2·3·4H에 seed1/6/7·512환경·900step, 총42회다.

원본 TokenHSI397D 입력과 저장 정규화·가중치를 고정하며 Enable에는 목표 정렬 가상 받침, Coupled에는 개인별 PUSH 구역을 입력한다. 물리 장면·공통 낙상·SR·Time·t_coop 판정은 공통 평가기를 재사용한다. Native TokenHSI에는 SCT scene attention이 없으므로 자기 상태·배정된 과제 물체만 입력한다. 자기 상태 함수 AST 및 SIT/CARRY/CLIMB/PUSH packing의 원본 단일 행동 함수와 CPU exact 일치를 검증했다. 체크포인트 strict load·가중치/정규화 불변 검증도 통과했다.

GPU6·7 기존 MPS에 연결하며 데몬·기존 학습을 조작하지 않는다. 자연 종료와 CUDA 동기화를 사용하며 signal/timeout/reset은 사용하지 않는다. 각GPU의2H 선행 정식 평가가900step·초기snapshot exact·CPU/GPU SR/Error 재채점(공동PUSH는Time/contact도) 검증을 통과하여 나머지40회 자동 분배를 시작했다. 동시에 GPU당 작은평가 최대3개 또는8/16H평가1개로 제한하고 메모리 여유도 검사한다. 진행은 `status.json`, 집계는 `summary.json`·`report.md`, 원시는 `carrier/output_etc/check/tables12/`를 참조한다. 최종 수치 완료로 기록한 것은 아니다.


### 31k 추가 평가용 원본 carry testSizes

사용자 요청으로 AT/ON_TOP 운반 상자에 원본 TokenHSI9종 testSizes를 선택할 수 있는 평가 전용 옵션을 추가했다. 원본과 동일한 BoxLib 샘플러를 사용하며 물리 형상·관측 크기·BPS를 일치시킨다. 받침과1109 채점은 유지한다. PUSH/학습 기본 경로는 변경하지 않았다. 원본 config와 SHA256은 평가 결과에 보존한다. CPU scoring/scene 테스트17개 통과, BoxLib AST 동일 확인. 실제 smoke 및 정식 결과는 output_etc/check/kd31k_native_extra_20261110/에 저장한다.

## 2026-11-09 CVPR2027 저장소 복사

### 23k 체크포인트 GPU7 MPS 단독 평가

- 최종 사용자 지시에 따라 ON_TOP을 제외하고 PUSH를 우선했다. 기존 MPS 평가의 AT seed1은 정지 파일로 프레임 경계에서 동기화·정상 정리했으며 부분 궤적은 미채점 보존했다. PUSH512×seed0/1/2·900step·revision1109 정식3회를 완료하고 추가 GPU 평가 없이 종료했다. Point/Region Ever는251/512·281/512·253/512, **51.11±3.28%**, 성공 trial의 최소거리Error는7.86±0.42cm, 낙상률1.43±0.49%다. 이 수치는 final SR이 아니다.
- 앞서 완료된 SIT99.67±0.11%·CLIMB3.32±1.28%(각native Point·3seed), AT66.21%(seed0만)를 함께 보존했다. 완료10회 정규화 고정·CPU 재채점·물리/관절/접촉/거리 유한값·공통 소스 해시 일치, 체크포인트 SHA 불변을 확인했다. 완료 로그에 CUDA/에셋/메모리 오류가 없고 기존MPS PID112084/112102와 학습880633 생존·TensorBoard 갱신을 종료 후 확인했다. [보고서](output_etc/check/kd23k_eval_mps_20261109/report.md), [검증](output_etc/check/kd23k_eval_mps_20261109/verification.json), [PUSH 원시 집계](output_etc/check/kd23k_eval_mps_20261109/push_priority/formal/summary.json).
- 사용자 지정 `output/push4096_kd0p1_sharedv2/TokenHSIPushKD_00023000.pth`를 CPU에서 로드해 epoch23000·모델107 tensor finite·KD0.1/shared v2 계약을 확인했다. SHA256은 `a2a327c46d4f56bd813438026167212e6f08d385c021f78004e6d823350743cb`다.
- 이관 시 제외된 데이터는 이 checkout의 `tokenhsi/data/`에만 공유 데이터 심링크6개를 만들고, 제공 template으로 runtime manifest3개를 생성했다. 원본 공유 디렉터리는 수정하지 않았다. 사용 모션/물체 참조458건 존재, PUSH21개 계약 해시 및 연결 데이터1667개 bundle 해시 일치를 확인했다.
- 기존 scoring/scene CPU 테스트17개와 5개 단독 스킬 CPU2장면×900step smoke/재채점이 통과했다. 초기 GPU7 사전 검사는 평가 프로세스의 MPS 접속만 비활성화한 별도 CUDA 방식으로 수행했고, 이후 아래의 사용자 요청으로 기존 MPS에 접속하는 실행으로 전환했다. MPS 데몬·기존 학습은 조작하지 않았다.
- 초기 실행·검증 산출물은 `output_etc/check/kd23k_eval_20261109/`에 보존한다. CPU/GPU smoke는 성능 수치로 사용하지 않는다.
- 평가 실행기에 `--cuda-launch-blocking 0|1`을 추가했다. 기본0은 정식 평가의 일반 실행,1은 작은 사전 검사의 동기화 진단용이다. 점수·물리·정규화 설정과 기존 MPS는 변경하지 않았다.
- 이후 사용자 요청으로 GPU7의 기존 MPS 서버에 접속하는 별도 실행으로 전환했다. 기존 비MPS 사전 검사 AT185frame에서 해당 비MPS 프로세스에만 SIGINT를 전달해 기존 정리 경로로 종료했고, 원시 로그를 보존했다. MPS 데몬/기존 학습에는 signal이나 설정 변경을 하지 않았다. 새 평가 클라이언트만 SM25% 상한·일반 CUDA 실행을 사용하며 산출물은 `output_etc/check/kd23k_eval_mps_20261109/`에 분리한다.
- 평가 시작 전 root-state와 actor 인덱스 불변조건 검사를 추가하고, 단독 평가 종료 전 CUDA 동기화 및 `CVPR_EVAL_STOP_FILE`을 통한 프레임 경계 정상 중지를 지원한다. CPU의 정상/음수/범위초과/int64/비연속 인덱스 검사와 기존17개 테스트가 통과했다. 채점·물리·학습 손실은 변경하지 않았다.

- 작업 폴더의 코드·설정·문서를 `CVPR2027/tokenhsi_push_kd/`로 복사했다. 기존 실행 경로와 학습 프로세스는 유지한다.
- 체크포인트, 실행 로그/output, motion/object/SMPL 데이터와 캐시는 제외했다. 실행에 필요한 데이터·가중치는 별도로 준비해야 한다.
- 복사 전 평가 코드 검증: CPU 단위 테스트 23개 통과, 5개 스킬 900 step smoke 및 native SIT snapshot replay 확인.

# 변경 기록

## 2026-11-07

- PUSH4096/KD0.1 shared v2 학습이06:31:33(KST)에5758 ITER 완료 후 CUDA launch timeout으로 종료된 것을 확인했다. 커널 로그는 해당 PID444518에 대해 watchdog GPU locked 및 Xid8을 기록했다. 학습 로그에 OOM 메시지는 없고 GPU hang의 근본 원인은 미확정이다.
- 최신5500 ITER 체크포인트의 CPU 로드, 모델 전체 finite, optimizer state 및 KD0.1/shared v2 계약을 확인했다. 이후258 ITER는 마지막 체크포인트에 포함되지 않는다. 실행 상태 메타데이터를 `failed_cuda_timeout`으로 정정하고 `failure_report_20261107.json`을 저장했다. 재시작은 수행하지 않았다.

## 2026-11-06

### TensorBoard 본학습3개로 정리

- 사용자 추가 요청에 따라 CHECK 실행8개의 TensorBoard 이벤트 디렉터리와 `output_etc/check`의 실행 로그28개를 삭제했다. 체크포인트·검증 요약·그래프·physics trace·빌드 캐시는 유지한다. 삭제 목록은 `output/check_log_cleanup_20261106.json`에 기록했다.
- TensorBoard 표시 대상을 기존2프로세스 PUSH100/KD0.001 fresh, PUSH100/KD0.01 fresh, 현재 shared v2 PUSH4096/KD0.1 fresh의3개로 제한했다. 기타 본학습 기록은 디스크에 보존하되 표시하지 않는다. 현재 학습은 계속 유지한다.

### 잘못된 통합 실행 기록 삭제 및 TensorBoard 갱신

- 사용자 요청에 따라 수정 전 shared 실행7개의 로그·체크포인트를6개 디렉터리에서 삭제했다(약307.9MiB). 대상은 이전 PUSH4096 KD0.1/0.01/0.001, 초기 inline/shared 시도, shared4096 검증 및 shared VRAM 검증이다. 삭제 목록은 `output/log_cleanup_20261106.json`에 기록했다.
- 기존2프로세스 worker 실행12개, 수정 후 shared60 ITER 검증, 현재 학습을 보존했다. 보존된 비활성 실행 파일62개의 크기·수정 시각이 그대로임을 확인했다. `backend_parity`의 버그 진단 자료와 비교 요약은 보존한다.
- TensorBoard를 `http://localhost:6007`에서14개 실행으로 갱신했다. 현재 학습 이름은 `CURRENT_SHAREDv2_PUSH4096_KD0.1_fresh`, 기존2프로세스 실행은 `WORKER` 접두사로 구분한다. 현재 학습 PID444518은 유지한다. 삭제된 실행의 과거 경로는 아래 역사 기록에 남아 있으나 실제 파일은 삭제되었다.

### 수정된 shared simulator로 PUSH4096/KD0.1 본학습 시작

- 사용자 요청으로 KD를0.1로 설정하고 PUSH4096/native2048, sharedSimVersion2, seed42 초기 모델에서 새 optimizer/ITER1로 시작했다. 최대26000 ITER 및 기존 horizon32/minibatch16384/mini_epochs6/학습률2e-5를 유지한다.
- PID444518, `output/push4096_kd0p1_sharedv2_fresh_seed42_20261106_160919/`. 설정·소스 스냅샷과 입력 해시를 저장했다. 초기 ITER 진행, 환경 계약, TensorBoard의 KD 계수0.1 및 GPU 프로세스 생존을 확인했다. 초기 GPU 사용량26506MiB는 한 시점 측정이며 peak가 아니다.

### Shared simulator reset 누락 수정 및 동일 조건 비교

- 이전 통합 facade는 reference/native의 state setter를 같은 physics step 사이에 반복 호출했다. Isaac Gym GPU pipeline의 setter/refresh 호출 제약을 위반했고, reference reset이 실제 physics에 반영되지 않는 것을 재현했다. 초기 상태와 액션이 같은 8+8 환경에서 첫 step의 위치 차이 최대34.7597m, 관측 차이37.4515, 보상 차이0.9910이었다. 아래의 초기20 ITER 검사는 실행·VRAM 검사였으며 동등성을 검증하지 못했다.
- `SharedSim`이 양쪽 풀의 reset 인덱스·상태를 모아 setter별로 simulate 직전에 한 번 제출하도록 수정했다. 전역 refresh도 physics step별 한 번으로 제한하고 native Python/NumPy/Torch RNG를 worker처럼 분리했다. 전체 state setter와 PD 제어도 통합한다. 수정된 shared 계약에는 `sharedSimVersion: 2`를 기록하여 수정 전 체크포인트의 strict resume을 구분한다.
- 수정 후 고정 액션12 step 및 중간 부분 reset에서 reference 위치/DOF/rigid body/관측/보상이 기존 worker와 정확히 일치한다. Native도 첫 reset 전 warmup과 월드 배치 좌표를 맞춘 진단에서12 step 모두 정확히 일치했다. 이 진단 옵션은 기존 worker의 학습 코드에 적용하지 않았다. CPU 회귀 테스트23개 통과.
- PUSH2048/native2048, KD0.001, seed42, 같은 초기 체크포인트·학습률·horizon32/minibatch16384/mini_epochs6으로 worker/shared 각각60 ITER 완료. PUSH reward 평균41–60 ITER는30.3765/35.7003, 마지막 값31.6430/39.1177이다. 환경 계약 차이는 backend와 sharedSimVersion뿐이다. ITER21–60 평균 teacher KL은0.6599/0.6803, actor loss는−0.02156/−0.02299다. 두 프로세스 종료와 GPU 메모리 반환 확인. 본학습은 재개하지 않았다.
- 코드/입력 해시, 물리 trace, 학습 로그·체크포인트, 비교 JSON/그래프는 `output_etc/check/backend_parity/`에 보존한다. 수정 전 shared 실행은 KD 계수나 환경 수 효과의 유효한 비교군으로 사용할 수 없다. 이번 단기 비교로 장기 수렴이나4096환경 성능까지 보장하지 않는다.

### KD0.001 동일 구성 검증 시작

- 사용자 요청으로 KD0.01 실행을175 ITER에서 중단했다. PUSH4096/native2048/shared/seed42 초기 모델은 유지하고 KD만0.001로 낮춰200 ITER 제한 검증을 시작했다.
- PID420661, `output_etc/check/push4096_kd0p001_fresh_seed42_20261106_140658/`. 설정 스냅샷 비교 및 TensorBoard의 계수 확인. 비교 스크립트 `compare.py`와 `reward_comparison.json`을 해당 폴더에 보존한다.
- 200 ITER 정상 완료 후 자동 종료, 최종 체크포인트 저장. PUSH reward 평균21–50: −1.84, 51–100: −1.03, 101–150: −1.43, 151–200: −0.48. 이전 worker/2048환경/KD0.001의 마지막 구간 평균104.10과 차이가 남는다. KD 계수를 낮추는 것만으로 이전 상승 추세는 재현되지 않았다. 환경 수·backend·난수 순서가 달라 원인을 하나로 확정하지 않는다.
- 읽기 전용 코드 점검에서 shared simulator의 반복 state setter/refresh 호출과 Isaac Gym GPU pipeline 제약을 대조했다. 물리 상태 동등성은 미검증이며, 메모 `simulator_audit_notes.txt`에 원인 후보와 검증 한계를 기록했다. 이번 비교 실행 중 simulator 코드는 변경하지 않았다.

### 동일 환경에서 KD0.01 새 비교 학습

- 사용자 요청으로 PUSH4096/KD0.1 실행을122 ITER에서 중단하고, 환경·simulator·seed·초기 모델을 유지한 채 KD 계수만0.01로 바꿔 처음부터 시작했다. 설정 스냅샷 비교로 변경 항목을 확인했다.
- 새 PID414128, `output/push4096_kd0p01_fresh_seed42_20261106_133801/`. ITER3 및 환경 계약 확인. TensorBoard6007에 `PUSH4096_KD_0.01_from_start`를 추가하고 이전 실행들도 유지했다.

### PUSH4096/KD0.1 새 본학습 시작

- 사용자 요청으로 기본 설정을 reference4096/native2048 및 KD0.1로 변경하고 seed42 초기 모델에서 새 optimizer/ITER1로 시작했다. shared simulator, horizon32/minibatch16384/mini_epochs6, 최대26000 ITER를 사용한다.
- PID409930, 실행 폴더 `output/push4096_kd0p1_fresh_seed42_20261106_131727/`. 첫2 ITER 정상 진행 및 체크포인트 계약의 환경 수/계수 확인. 설정 스냅샷과 이전 실행 메타데이터를 실행 폴더에 보존했다. 최신 상태는 `output/active_training.json` 참조.

### PUSH 환경만 2배 검증

- `referenceNumEnvs` 옵션으로 PUSH4096/native KD2048을 구성했다. 두 풀의 서로 다른 크기에 맞춰 학습 버퍼 행 수와 shared simulator의 native collision group offset을 수정했다. 옵션 생략 시 기본2048+2048은 유지한다.
- 초기 체크포인트/seed42/KD0.01/horizon32/minibatch16384/mini_epochs6으로20 ITER 정상 종료. 학습 프로세스 VRAM peak27.25 GiB(기존20.87 대비+6.38), 평균 ITER10.01초(기존7.75초). 1초 간격 샘플이며 데스크톱 제외, 장기 peak는 미검증.
- 회귀 테스트19개 통과. 테스트 프로세스 종료 확인, 본학습은 시작하지 않았다. 설정·로그·비교표: `output_etc/check/shared_sim_push4096/`. 첫 시도의 버퍼 크기 오류 로그는 해당 폴더의 `initial_buffer_mismatch/`에 보존했다.

### 단일 simulator 통합 완료

- `SharedSim`/partition Gym facade로 reference와 native 환경을 같은 simulator에 생성한다. 전역 actor/DOF/body/sensor tensor를 풀별로 나누고 reset 인덱스와 action 버퍼를 연결하며, 공통 physics loop로 두 풀을 함께 진행한다. RSI 초기화용 전체 state setter도 전역 버퍼에 연결한다.
- `push_kd.yaml`의 기본 backend를 shared로 변경했다. `TOKENHSI_SIM_BACKEND=worker`로 이전 방식을 선택할 수 있고 실제 backend를 환경 계약에 기록한다. 현재 headless 학습만 통합하며 평가/view 경로는 유지한다.
- 동일한 4096행/KD0.01 설정으로 shared와 worker 각각 20 ITER 정상 완료: GPU 프로세스 합산 peak 24.68→20.87 GiB, 약3.81 GiB/15.4% 감소. 평균 ITER7.74→7.75초. 단기 측정이며 본학습은 시작하지 않았다. 자료: `output_etc/check/shared_sim_vram/`.
- CPU 회귀 테스트19개 통과. GPU 소규모 실제 PPO/KD 업데이트와 풀 간 reset 격리/공통 physics step 검사를 수행했다.

### 단일 프로세스 통합 시도

- 사용자 요청으로 native simulator를 main 프로세스에 직접 생성하고 tensor를 직접 전달하는 inline backend를 구현해 검사했다. CPU2환경, GPU16환경 모두 두 번째 gym.create_sim에서segmentation fault(exit139)가 발생했다.
- 모델·에셋 없는 최소 재현도 첫 simulator 생성 성공/두 번째 생성 crash를 확인했다. 현재 런타임에서는 두 simulator의 단일 프로세스 배치가 작동하지 않으므로 production 연결은 되돌렸다. 실험 소스·로그는 `output_etc/check/inline_pool/`에 보존했다.
- worker 방식 유지, 본학습 중단 상태 유지. VRAM 절감량은 측정하지 못했다. 진정한 통합에는 shared simulator 생성·global tensor slicing·통합 physics step 구현이 필요하다.

### PPO/KD gradient 크기 검사

- 사용자 요청으로 본학습은 중단 상태로 유지하고 seed42 초기 체크포인트에서 고정KD0.01·PUSH100%·원래4096행/minibatch16384 구성으로5 ITER만 실행 후 정상 종료했다.
- opt-in `PUSH_KD_GRAD_PROBE`로 shared transformer encoder에 대한 PPO actor/KD의 gradient norm, cosine, 목표30% 계수를 측정한다. autograd.grad로 .grad를 누적하지 않고 실제 업데이트는 기존 합산 loss로 수행한다. 분석용 벡터에서norm/cosine 및 최종backward 불변 확인.
- 총240 optimizer 업데이트 중15개 측정. 첫 near-zero KD 샘플을 제외한14개에서 가중KD/PPO norm 비율 평균약0.50%, 범위0.34~0.72%. 초기5 ITER 결과이며 Adam 이후의 업데이트 비율이나 장기 균형을 의미하지 않는다. 동적 계수는 적용하지 않았다.
- 결과 `output_etc/check/gradient_probe_kd0p01/{gradients.jsonl,summary.json,run.log}`.

### 사용자 의도 수정: KD0.01 처음부터 학습

- 재개가 아닌 초기 상태부터 학습하라는 사용자 정정에 따라6500 재개 프로세스를 종료했다. seed42 `weights/push_init_random_seed42.pth`로 새 optimizer/epoch에서 시작한다. PUSH100%, KD0.01 유지.
- 현재 실행 `output/push100_kd0p01_fresh_seed42_20261106_115849/`, PID395372. ITER1부터 정상 진행 확인. TensorBoard6007은 이전 KD0.001 초기 실행과 새 KD0.01 초기 실행만 비교하도록 변경했다. 잘못 재개한 실행의 로그는 보존한다.

### KD 0.01로 변경 후 재개

- 사용자 요청으로 기존 학습을6619 ITER에서 중단하고 최신 저장본6500 ITER의 모델·optimizer 상태에서 재개했다. PUSH100%, Reference1명, 총4096행을 유지하고 KD0.001→0.01로 변경했다. 미저장119 ITER는 재학습한다.
- `PUSH_KD_ALLOW_COEF_CHANGE=1`일 때에만 resume contract의 KD 계수 차이를 허용하며 나머지 차이는 계속 거부한다. 관련2테스트 통과.
- 출력 `output/push100_kd001_resume6500_20261106_114949/`, PID393565. ITER6501 이후 정상 업데이트 및 TensorBoard KD 계수0.01 확인.

## 2026-11-05

### PUSH100% 재실행 및 ITER 콘솔 로그

- 사용자 요청으로 PUSH100%·Reference1인·KD0.001 본학습을 GPU0/seed42에서 시작했다. 이전202 iteration 실행에는 저장 checkpoint가 없어 random-init checkpoint에서 새로 시작함을 고지했다.
- 콘솔 FPS 로그에 `ITER 현재/최대`를 추가했다. `ITER 1/26000`, `ITER 2/26000` 정상 출력 확인. PID308665, 출력 `output/push100_solo_random42_seed42_20261105_210745/`.

### 향후 Reference PUSH 100%

- 사용자 요청으로 Reference PUSH prior를100%로, 나머지 과제 prior를0%로 변경했다. Native KD2048명은 유지하여 전체4096개 행 중 PUSH2048명(50%)이다. validator 및 sampling 검증을 갱신했다. 학습은 중단 상태로 유지한다.

### 학습 중단 및 PPO/KD loss 비교

- 사용자 요청으로 PUSH80%·Reference1인 main PID304262와 worker305001을 종료하고 GPU 메모리 반환을 확인했다. TensorBoard202 iteration의 actor/critic/bounds/AMP/KD 값을 추출했다. KD 실반영 값은teacher KL×0.001이며 각 loss는iteration 내 minibatch 평균이다.
- 전체 값과 가중치 적용 결과: `output_etc/check/loss_scale_push80/{iterations.csv,summary.json}`. Scalar 비율은gradient 크기 비율을 뜻하지 않으며 추가 학습은 실행하지 않았다.

### PUSH80%·Reference1인 본학습 실행

- 사용자 요청으로 GPU0/tokenhsi118, seed42, reference2048×1명 +native KD2048명, 최대26000 epoch 본학습을 detached 시작했다. PUSH tokenizer random init, 고정KD0.001 및 기존 PPO/RSI 설정을 유지한다.
- RSI1236프로필 초기화 후2 epoch 완료 및 프로세스 생존 확인. 당시 학습 프로세스 GPU 메모리 합계24134MiB(23.57GiB); 장기 peak 측정치는 아니다. PID304262, 출력 `output/push80_solo_random42_seed42_20261105_202741/`.

### Reference PUSH 80%

- 사용자 요청으로 PUSH prior70→80%, SIT/CLIMB/AT/ON_TOP은2/5/6.5/6.5%로 조정했다. 1인 Reference2048 + native KD2048 구성에서 전체 PUSH 기대 비율은40%다. validator와 sampling 검증을 갱신했다. 본학습은 재시작하지 않았다.

### Reference 1인 전환

- 2인 본학습은18 epoch 완료 후 AMP replay sample의1006MiB 할당 중 CUDA OOM으로 종료됐다. 사용자 요청으로 Reference를 장면당1명/상자2개(작업·보조 각1개)로 변경했다. 기본2048 reference +2048 native KD =4096 policy rows이며 reference PUSH70%, RSI, 상자 밀도/보상 수식, tokenizer 초기화, PPO minibatch는 유지한다.
- 자산·graph·배치·reset·관측·평가·학습 버퍼의2인/3배 고정 인덱스를 인원수에 맞췄다. 2인 구성도 계속 지원한다. 두 번째 사람과 그 소유 상자가 제거되므로 사람 간 충돌/경로 분리 조건은1인에서 적용되지 않는다.
- RSI 캐시 인덱스 변경으로 새 캐시를 생성한다. 기존 캐시를 덮어쓰거나 RSI를 생략하지 않았다.
- CPU17테스트 및 모션/에셋 preflight 통과. GPU PUSH 추론4개 에피소드×1200step 완료. Reference16명+native16명, 검사 전용 minibatch512에서 원래 RSI를 사용한2 epoch 학습·체크포인트 저장·정상 종료 확인. 기본2048 환경의 장기 VRAM은 아직 재측정하지 않았고 본학습은 재시작하지 않았다.
- 검증 로그: `output_etc/check/reference_solo/{eval.log,eval.json,train.log}`.

### PUSH 70% 본학습 시작

- 사용자 요청으로 GPU 0/tokenhsi118에서 seed 42, PUSH tokenizer 랜덤 초기화, 고정 KD 계수 0.001, 최대 26000 epoch 본학습을 detached 실행했다. 기존 RSI 설정과 캐시를 사용한다.
- `scripts/train_local.sh`/`.py`에 로컬 CUDA 호환 진입점을 추가했다. 시작 후 7회 epoch FPS 출력과 main/worker 프로세스 실행을 확인했다. GPU 메모리 합계 29598MiB(28.90GiB).
- 출력/로그: `output/push70_random42_seed42_20261105_200753/train.log`. 실행 정보는 같은 폴더 `launch.json` 및 `output/active_training.json`에 기록했다.

### 현재 설정 VRAM 실측 (PUSH70%, 랜덤 tokenizer)

- 사용자 요청에 따라 RSI를 생략하지 않고 reference2048×2명 + native KD2048명, minibatch16384/horizon32/mini_epochs6으로3 epoch 실행 후 정상 종료했다. RTX5090/tokenhsi118, 기존 과제 RSI3143프로필 캐시를 최초 생성했다.
- 첫 rollout에서 추가로 확인된 teacher sigma buffer `[32]` mask 오류를 `ones_like(mus)`로 수정했다(제출 코드와 동일). rollout/optimizer routing CPU2테스트 통과.
- `scripts/profile_training_vram.py`는1초 간격 프로세스 GPU 사용량과 epoch별 PyTorch peak를 기록한다. native worker에는 opt-in `TOKENHSI_LOCAL_CUDA_COMPAT=1`로 CUDA11/Blackwell 호환 처리를 적용했다.
- 측정 peak 합계29592MiB(28.90GiB): main24688MiB + KD worker4904MiB. main PyTorch allocated peak17.20GiB/reserved peak19.19GiB. 학습 외 GUI/데스크톱 프로세스는 합계에서 제외했다.3 epoch 후 GPU 전체1871MiB로 반환 확인. 초기3 epoch의 표본 측정이며 장기 상한을 보장하지 않는다.
- 결과: `output_etc/check/vram_push70/{summary.json,allocator.jsonl,gpu_samples.json,run.log}`. 최초 실패 로그는 별도 보존했다. 본학습은 실행하지 않았다.

### PUSH 70% 및 tokenizer 랜덤 초기화

- 사용자 요청으로 reference `PUSH_AT` prior를50→70%로 변경하고 기존 과제 비중을 비례 축소했다. validator에도70% 구성을 허용했다. native KD 풀/접촉 RSI 비율은 유지한다.
- 체크포인트 확장 스크립트의 기본 PUSH tokenizer를 seed 지정 랜덤 Linear weight/zero bias로 변경했다. `--push-init carry`로 이전 초기화를 재현할 수 있다. 기존 파일을 보존하고 `weights/push_init_random_seed42.pth`를 생성하여 새 학습 config에 연결했다.
- 새 학습을 막던 고정sigma 벡터의 batch mask 오류를 `expand_as(mu)`로 수정했다.
- 실제20,000 reference 슬롯의70% sampling, 새 tokenizer만 변경됨/seed 재현, 기존 정책 출력 보존, 실제 학습 메서드의 sigma/gradient routing을 포함한 CPU9테스트 통과. 학습은 시작하지 않았다.

### 제출 모델 연속 뷰어

- 사용자 요청으로 GUI 기본값을 연속 deterministic 추론으로 변경했다. 종료된 장면만 reset하고 새 에피소드를 계속 재생한다. `VIEW_LOOP=0`은 기존 단일 평가이며 headless 기본값도 단일 평가를 유지한다.
- 검은 화면 보고 후 GUI 실행에 NVIDIA Vulkan ICD와 device-selection 환경변수를 명시했다. 재시작 후 Isaac Gym 자체 프레임 캡처에서 사람·상자·바닥의 정상 렌더링 확인: `output_etc/check/submission_inference/view.png`. 선택적인 `VIEW_CAPTURE`로 30번째 step 화면을 저장할 수 있다.

### 제출 체크포인트 로컬 추론

- `prepare_paths.py`로 세 runtime manifest를 현재 저장소 경로로 재생성했다. 130개 모션, PUSH RSI 및 에셋 해시 검증 통과.
- 제출 코드의 `push_kd_player.py`와 `native_eval_warmup.py`를 가져와 PUSH batch 크기 초기화 및 제출 평가 지표를 적용했다.
- `scripts/infer_submission.sh`/`.py` 추가: tokenhsi118, GPU 선택, headless 선택, RTX 5090/CUDA 11 fusion·3축 product 호환 처리. 보상과 체크포인트는 변경하지 않았다.
- 실제 GPU 추론 완료: seed 0, 4장면/8명, 1200-step horizon, 정상 종료. 성공 1/8, 낙상 0/8. 소규모 실행 검증이며 정식 성능 평가가 아니다. 결과: `output_etc/check/submission_inference/push_eval.json`, 로그: 같은 폴더 `run.log`. GUI는 미검증.

## 2026-10-31

### 확정 설정 문서 정리

- METHOD에 PUSH 토큰397D/전용 tokenizer, AMP21+4개·50:50·10프레임, RSI949후보·4모션·후반 선택·원본 자세 보존·잔여2~10m를 상세 기록했다. Teacher/student 초기 모델과26k 환경 기준을 구분했다.
- 기존3과제 native KD·TRAJ 제외와 프로세스 격리 후 CPU 검증 상태를 반영하고 METHOD/README의 오래된 미검증 문구를 정리했다. 문서만 변경했으며 코드·학습 설정은 유지했다.

### 격리 pool CPU 검증 완료

- native pool은spawn별도프로세스·NumPy array pipe로연결한다. 상대main경로/음수seed/공유메모리socket제약을수정했다.
- 참조2scene(4명)+native2환경의reset/397D관측/물리3step finite와실제student행동3step·native teacher KL0.0을확인했다. CPU12테스트통과. GPU/PPO본학습은미실행. 로그:output_etc/check/reference_integration/{mixed_isolated,policy_isolated}.log.


### PUSH reference scene 연결

- 사용자요청대로PUSH를26k의실제2인4상자mixed 환경에연결했다. namespace snapshot/원본sampling·support/경로거부·heading·CPA비용·RSI/reward사용, actor관측만TokenHSI397D로변환한다. Native3과제KD pool을별도유지하며TRAJ제외.
- PUSH/KD/참조비PUSH 슬롯을별도mask로구분하고6144policy row(참조2048scene×2 + native2048)로batch를맞췄다. source/환경config를resume계약에기록하고문서를현재구성으로갱신했다.
- CPU 단독참조PUSH2scene/4명 reset·3물리step finite통과. mixed reset의기존과제cache생성후같은프로세스의두번째simulator생성에서segmentation fault. 추가물리실행중단후native simulator를spawnworker/CPU IPC로격리했다. 격리후전체mixed물리연결·GPU/PPO는미검증이다. 로그:output_etc/check/reference_integration/mixed.log.


- 사용자 요청으로 TRAJ를 학습/KD/평가에서 제외. 기존 풀은 SIT/CARRY/CLIMB 균등 reset이며 PUSH50% 풀은 유지한다. 관측/tokenizer 슬롯은 체크포인트 호환을 위해 보존한다.

- TokenHSI baseline 독립 snapshot에 PUSH PPO+AMP / 기존4과제 Gaussian forward KL 분리 구현. Teacher/student 초기 가중치 준비. 원본 프로젝트 변경 없음.
- Stage1 26k의167크기·밀도3·마찰0.4·목표2~10m/±45°·RSI949후보·AMP50:50·CONTACT+AT·20cm/Z1mm 성공 연결. 1인 장면과2인 원본의 차이를 METHOD에 명시.
- CPU6테스트 통과: real checkpoint strict 로드와 기존행동 보존, KL/gradient 분리, 실제 training method의 optimizer step, reward/데이터 계약. Python compileall/셸 문법 통과.130모션 경로 존재 및 PUSH21개·asset/mesh hash 일치.
- Isaac Gym full task import는 외부 읽기 전용 JIT cache lock에서 중단. Simulator/rollout/학습 통합·장기 성공률 미검증. 본학습 실행 없음.

31k 추가 평가 완료: SIT99.61±0.39%, CLIMB4.69±0.70%, AT90.56±1.77%, ON_TOP Region89.65±2.17%. 각512×3seed, 원본9종운반상자 검증·12회CPU재채점·유한값 검사 통과. SIT 낙상1499/1536으로 ever와 유지성능을 구분해야 함. 결과 output_etc/check/kd31k_native_extra_20261110/report.md.

### CLIMB 첫 에피소드 대 동일 RNG reset 진단

`CVPR_CLIMB_RESET_DIAGNOSTIC=1`인 경우에만 native 최초 reset 직전 RNG를 보존하고, 첫900step 뒤 동일 RNG로 reset한2·3번째900step을 따로 측정한다. 낙상으로 해제한 모터 설정을 복구하고 root·관절·목표·관측·rigid-body·접촉을 기록한다. 원본 teacher의3%/99.8%는 과거 주석이며 이번 측정으로 단정하지 않는다. CPU2장면3episode 및 재채점 통과, 기존 CPU테스트17개 통과. 기본1109 결과는 첫episode만으로 유지하며 이후 결과는 진단으로 분리한다. 결과: output_etc/check/kd31k_climb_reset_diagnostic_20261110/.

CLIMB31k 진단 완료: 동일 초기 장면·관측에서 첫episode4.62±0.63%, 두번째99.80±0.20%, 세번째99.93±0.11%. 새환경에서0-action1step후동일reset도99.93±0.11%로 재현. 700step정책워밍업은 필요조건이 아님. GPU raw rigid-body/contact 초기 버퍼 차이를 관찰했으나 특정PhysX 내부 원인은 확정하지 않음.12회CPU재채점·유한값·initial동일성검증 통과. 기본1109 첫episode경로는 유지. 상세 output_etc/check/kd31k_climb_reset_diagnostic_20261110/report.md.
