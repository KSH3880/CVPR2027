# CLIMB FT5k에서 낮은 학습률로 추가20k

사용자최종지시: 기존FT5k에서lr5e-6로추가20000 iteration. 시작은78k원본이아니라TokenHSIClimbFullFT_00005000.pth다. 기존FT epoch5000→25000(총기준누적103k상당). 학습환경은기존tokenhsi_climb_full_ft.yaml을그대로재사용:1명/1고정물체/2048환경,train Box/Cabinet/Table각1/3,loco/CLIMB RSI50:50,300step/IET. 전체레이어/정규화학습유지.

모델·scene/AMP/value정규화·Adam모멘텀·epoch/frame을원본FT5k에서exact복원하고optimizerLR만5e-6로덮어쓴다. bootstrap 단계의78k전이는Runner.restore 전에수행되며실제학습시작weight는5k다. continuation_validation.json이실제복원원장이다. 정상resume와같이시뮬레이션및미저장AMP replay buffer는새프로세스에서초기화된다. 학습은1k마다끊지않고추가20k 연속실행한다.

검증:2048환경·epoch5000→5001 rc0;모델/정규화/Adam모멘텀exact,optimizerLR5e-6,전체층gradient확인. output_etc/check/tokenhsi_climb_low_lr_resume_20261025/ 및 /tmp/tokenhsi_climb_low_lr_smoke.log. train config는기존대비name/LR/max_epochs/save_frequency만다름. 본학습max_epochs25000,save_frequency1000. CHECK=1은resume절대epoch5001로검사한다.

현재실행원장: project runs/tokenhsi_climb_low_lr_20261025/active_run.json. 학습log/train/실제복원검증도같은run아래다. supervisor가자연종료exitcode를train_exit.json에남긴다. 기존학습/결과/baseline은보존한다.

watch_continuation.py는epoch6000~25000의20개checkpoint를각1회성공평가할때까지감시한다. GPU여유26000MiB조건을지키므로현재GPU1장에서는통상학습종료후순차평가한다. 학습을멈추지않는다. 동일실제object mix/1명1고정물체/test/loco100%/512환경/seed0/3repeat/10초/골반20cm ever,평균은repeat1·2다. 평가산출물은output_etc/check/<runname>/epoch_<epoch>_attempt_<n>/에둔다. checkpoint미완성은대기,평가실패는로그보존후30분뒤새attempt로재시도한다. 프로세스중복은flock으로거부한다.

각평가는전trace NumPy64재채점/환경config일치/finite/첫episode종료1회를검증한다. run의evaluation_status.json·EVALUATION_REPORT.md·best_checkpoint.json은원자적JSON갱신으로진척과최고checkpoint를기록한다. 시작점87.21%도best후보로보존하므로추가학습이모두나빠지면FT5k를계속선택한다. 학습종료후필수checkpoint누락시상태를명시하고종료한다. 20개평가가끝나면watcher도종료한다.
