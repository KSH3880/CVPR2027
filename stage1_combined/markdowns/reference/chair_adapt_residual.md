# Chair carry action residual

Table에서 검증한 `table_adapt_residual`을 ArmChair_Normal에 적용한다. 기존 Table 학습을 유지하며, 2026-10-27 사용자 요청으로 Table 정상 종료 후 Chair 자동 시작을 예약했다. 현재 상태는 `stage1_combined/chair_residual_active_run.json`을 확인한다.

- 원본 `checkpoints/Stage1V2Fixed_00078000.pth`에서 fresh 시작. Table 학습 가중치를 이식하지 않는다.
- 1명·동적 의자 1개. train63/test27, 실제 mesh/mass/bbox. 객체 입력30D(자체 로컬 속도6+bbox24), 전체 관측280D, HOLDING_AT graph/GTA 유지.
- actor/critic encoder, 원본 action head, sigma, 전체 scene RMS 동결. 정책은 `mu=base_mu+residual(latent64)`의 256→128→32 ReLU MLP만 학습하며 마지막 linear는 zero-init이다. 새 tokenizer는 추가하지 않는다.
- critic value head와 fresh AMP discriminator도 학습한다. 총 학습 tensor18개(residual6/value head6/discriminator6). scene RMS는78k 고정, AMP RMS는 fresh, value RMS는78k 복원 후 갱신.
- TokenHSI 원본 carry 보상 `.2*walk+.2*handheld+carry+.2*putdown+upperbody`; power OFF, box speed threshold2m/s, upright/height/source-departure gate 유지. combined=.5*task+.5*AMP, AMP scale3. relationReward 설정은 graph/진단에 쓰이며 실제 task reward는 위 native reward로 덮어쓴다.
- 원본 carry reset/termination 공유, 바닥 배치, source1~5m, Chair 정면이 사람을 향한다. 목표 origin±4.5m 및 사람/물체와 거리²≥2. RSI는 locomotion reference만(loco_more/loco=.5/.5), 물체 carry RSI 미사용. AMP expert는 동일 locomotion/carry 모션.
- Table과 같은4096환경, spacing5, horizon32, minibatch16384, mini_epochs6, lr2e-5 constant, seed42. 총10000회,100회마다 번호 checkpoint. Table/Chair 환경 YAML은 category/reward label/face_human만 다르고 학습 YAML은 실험명만 다르다.

원본 TokenHSI의 Table/Chair YAML 차이는 `objSpecified` 하나다. 원본 보상·종료 함수는 같은 파일에서 사용하므로 `HumanoidTableTokenHSIFT`의 구현을 공유하고 `_reset_boxes`만 Chair 정면 reset으로 바꾼다. platform 설정 검증은 정확히 일치하는 native_chair/ArmChair_Normal 조합을 추가 허용한다.

`dataset_table_hold_at.yaml`은 물체 에셋을 지정하지 않는 공용 carry 모션 목록이므로 Chair도 그대로 사용한다. 현재 checkout에는 과거 Chair 스크립트가 참조하던 `dataset_tokenhsi_hold_at.yaml`이 없다. 새 스크립트는 존재하는 목록을 명시하고, 기존 정식 Chair evaluator는 구 목록이 있을 때 그것을 유지하며 없으면 공용 목록을 사용한다.

실행 명령은 [config](../config.md). 학습 환경 수를 임의 축소하지 않는다. 동시4096 학습을 시작할 GPU 여유가 없으므로 현재 검증은 CPU 전이/gradient와 소규모 **추론**으로 제한하며4096 PPO 실행 검증과 본학습은 아직 수행하지 않는다.

정식 평가는 `chair_adapt_residual_eval.sh [CKPT]`로 native_chair_hold_at_30s_v1(test27,512×독립seed0/1/2,30초,AT20cm 첫episode ever)을 쓴다. residual network 이름과 저장된 normalizer를 그대로 복원한다. `--smoke`는4환경×seed0 동작검사이고 성능 측정이 아니다. `--prepare-only`는 실행 없이 checkpoint/에셋/코드 사본만 만든다. GPU0·tokenhsi118은 새 wrapper가 지정하며 기존 driver의GPU5 기본값은 유지한다.

검증 산출물: `runs/chair_adapt_residual_prepare_20261027/output_etc/check/`. CPU script의 `untrained_chair_residual.pth`는 원본78k+zero residual로 만든 **추론 검증 전용** 파일이며 학습·resume용 checkpoint가 아니다.

준비 검증: CPU 전이/2backward/동결/부분reset/정면·에셋 검사,1환경 native task 추론rc0,정식 evaluator4환경×seed0/900step smoke rc0 및 trace/32·64bit/hash 재채점 통과,metrics5개 통과. 4096 PPO는 미검증이다. 전체 문서 검사기는 기존 다른서버 경로 하드코딩으로 실행 실패했으며 새 문서 링크/Python·shell 구문은 별도 확인했다.


## Table 종료 후 자동 실행 예약

예약 루트: `runs/chair_residual_after_table_20261027/`. 독립 supervisor는 Table residual의 정상 종료 marker(`residual/exit_10000.json`,rc0)와10000번호 checkpoint·기존 PID 종료를 확인한 뒤 GPU0 여유≥23000MiB일 때 실행한다. 10초 간격으로 확인한다. Table 실패·취소·완료 marker 누락 때는 Chair를 시작하지 않는다.

고정 `code/` 사본·78k checkpoint/해시, seed42/4096환경/TokenHSI보상·PPO/총10k/save100을 사용한다. 먼저4096환경2회 PPO를 `output_etc/check/train/`에서 검증하고, 원본153tensor/전체scene RMS동결·학습18tensor·양의residual gradient를 확인한 뒤 본학습을 원본78k에서 fresh 시작한다. 본학습은 `train/`,로그는 `preflight.log`와 `train.log`. 실패 시 자동 재시도하지 않으며 state/failed.json에 남긴다. 프로세스 starttime과 파일 lock/launch marker로 PID 재사용·중복 시작을 막는다. 큐 데몬 및 기존 Table watcher는 변경하지 않는다.

상태 확인(프로젝트 루트):

```bash
cat stage1_combined/chair_residual_active_run.json
tail -f runs/chair_residual_after_table_20261027/scheduler.log
```

예약 시점에는 Table이 아직 실행 중이므로 실제 Chair4096 PPO 및 본학습은 대기 상태다. scheduler 조건·정상/실패분기·preflight→fresh 본학습·중복방지 모의 검증만 완료했으며 로그는 예약 루트 `output_etc/check/scheduler_test.log`에 있다.


## 2026-10-28 예약 복구

기존 예약은 Table10k 정상종료 후04:45에 실행됐으나,고정 checkout 바깥의 `output_etc/check/train`을 요청해 wrapper 경로검사에서rc2로 종료했다. 이 실행은 PPO/본학습 시작 전 실패이며 기존 폴더를 보존한다.

복구 루트는 `runs/chair_residual_after_table_20261028_recovery/`. preflight는 해당 `code/output_etc/check/chair_preflight/train`으로 이동했고 scheduler 등록 시 실제 `resolve_output_path`를 실행하여 검증/본학습 경로를 모두 검사한다. 실제 resolver를 사용하는 회귀검증으로 옛 경로 거부/새 경로 허용을 확인했다. Table최종 checkpoint/78k/환경·보상/PPO·4096환경·10k설정은 동일하다. 복구 과정에서도 다른 평가 프로세스는 종료하지 않는다. 현재 활성 경로/PID/학습 단계는 `chair_residual_active_run.json`을 따른다.

복구 검증 결과: 실제4096환경2PPO rc0/동결153·scene RMS exact/학습18tensor/양gradient 통과. 2026-10-28 11:08:33KST fresh78k 본학습을 시작했다. 현재 상태·PID는 활성 JSON을 따른다. Table은10k 정상완료,Chair가밤사이학습된것은아니다.

2026-10-28 최신저장2500회 공통30초AT평가:426/415/435명,**83.07±1.96%**(512×독립평가seed3개). test27모두포함,trace/precision/hash·동결153/scene RMS검증통과. 본학습유지. 산출물 `runs/chair_residual_eval2500_20261028/REPORT.md` 및validation.json.

2026-10-28 최신저장3100회 공통30초AT평가:426/432/437명,**84.31±1.08%**(평가seed표본std). 2500회대비+1.24%p,전seed초기전배열2500회와bitwise exact·trace/precision/hash·동결검증통과. 본학습유지. `runs/chair_residual_eval3100_20261028/REPORT.md` 및validation.json.

2026-10-28 최신저장3800회 공통30초AT평가:452/450/442명,**87.50±1.03%**(평가seed표본std). 3100회대비+3.19%p,전seed초기전배열3100회와bitwise exact·trace/precision/hash·동결검증통과. 본학습유지. `runs/chair_residual_eval3800_20261028/REPORT.md` 및validation.json.

2026-10-28 최신저장4400회 공통30초AT평가:455/460/461명,**89.58±0.63%**(평가seed표본std). 3800회대비+2.08%p,전seed초기전배열3800회와bitwise exact·trace/precision/hash·동결검증통과. 본학습유지. `runs/chair_residual_eval4400_20261028/REPORT.md` 및validation.json.


## 최신 checkpoint 자동 평가 명령

프로젝트 루트(CVPR2027)에서:

```bash
TOKENHSI_GPU=0 bash stage1_combined/tokenhsi/scripts/multi_agent/chair_adapt_residual_eval.sh
```

인자 생략 시 활성 `chair_residual_active_run.json`의 본학습 경로에서 번호가 가장 큰 Chair residual pth를 선택한다. best alias/preflight는 제외하고, 저장 중인 최신 파일은 잠시 재시도한다. 평가 시작 시 번호 저장본을 고정 복사하므로 도중 새 checkpoint가 생겨도 대상이 바뀌지 않는다.

특정 checkpoint를 평가하려면 명령 마지막에 `"$CKPT"`를 붙인다. 기본값은GPU0/tokenhsi118/PYTORCH_JIT0,512환경×독립seed0/1/2·30초·AT20cm다. seed별 성공수/성공률을 출력하고 마지막에 `Mean ± Std`(ddof1 평가seed표본std)와 결과 폴더를 출력한다. 전체episode 배열은 summary.json/trace 파일에 그대로 보존한다. `--smoke`는 성능 mean/std로 표시하지 않는다. `--prepare-only`는GPU실행없이설정/코드/에셋/모델사본만만든다.

2026-10-28 검증:최신번호선택/별칭제외/부분저장재시도/잘못된category거부/mean·std표시/smoke구분 CPU5개 통과. 실제wrapper `--prepare-only`로최신5000선택 및512×3seed설정확인. 기존4400summary를같은formatter에넣어89.58±0.63%출력확인. 이CLI변경검증에서는새GPU평가를실행하지않았다. 로그 `runs/chair_eval_cli_20261028/output_etc/check/`.


## 한 번의 rollout에서 우리/TokenHSI 조건 동시 채점

기존자동최신평가명령그대로사용한다. seed마다동일한첫episode를최대30초만실행하고,우리AT20cm/30초와TokenHSI거리20cm+upright(updot≥.86615)/처음599action(20초설정)을함께저장한다. 각각3seed평균±표본std를출력한다. TokenHSI행은공통scene에조건만적용한점수로서원본환경/연속repeat평가와구분한다.

결과폴더의`episodes.csv`에환경별두성공여부·최초성공step·성공중최소거리오차를합쳐저장하고`summary.json`에두기준통계를기록한다. 기본경로는프로젝트`output_etc/check/hold_at_eval/<YYYYMMDD_HHMMSS>/`이다. 자세한필드·검증은 [동시채점명세](../../evaluation/hold_at/README.md#같은-episode에서-두-기준-동시-기록)를확인한다.

2026-10-28 Chair본학습10000회/rc0완료. 최종10k의512×독립seed0/1/2평가:우리30초AT **95.51±0.52%**,동일episode TokenHSI20초AT+upright조건 **89.91±1.11%**(평가seed표본std). 전trace/해시/통계/CSV1536행검증통과. `output_etc/check/hold_at_eval/20261028_233319/REPORT.md` 및validation.json.
