# 원본 TokenHSI Table 설정 + 우리78k 모델

사용자 최종요청에 따라 carry RSI를 사용하지 않고 원본Table 보상·종료·PPO/AMP 설정을 이식한다. 이전 HOLDING_AT 관계보상 학습과RSI실험은 보존하며 재개하지 않는다. TokenHSI baseline은 읽기전용이다.

## 원본과 일치하는 부분

- 원본 `humanoid_adapt_carry_box2objs.py`의 보상5함수와 종료2함수를 AST동일하게 복사했다. 실제 task보상은 `.2 walk+.2 handheld+carry+.2 putdown+upperbody`이며 에너지OFF,원본carry함수 안의사람/물체속도2m/s초과패널티·들기높이gate·출발점이탈0.8·직립유도 포함. 관계 state/progress/success는실제PPO보상에합산하지않는다.
- 물체낙하/사람낙상·높이3m·속도5m/s·사람이물체에서이탈하는종료를원본함수로적용. 출발/목표XY예외는숨겨진MA받침좌표대신실제출발object/목표좌표로대응한다.
- 4096환경/spacing5/600step(20초)/1명1동적Table,train21/test9 원본질량/크기/mesh. 1~5m/무작위yaw/바닥+5cm,목표XY±4.5m·squared distance≥2. Table은원본코드에서randomHeight를제외한다. YAML의prob1/maxTop1.35도보존했고전용조건만기존stage1validator에서허용했다.
- 원본초기화 loco_more/loco=.5/.5,expert loco_more/loco/pickUp/carryWith/putDown=.5/.2/.1/.1/.1. Table carry RSI/IK bank미사용.
- PPO/AMP `params.config`는실험이름외원본YAML과전항목동일:lr2e-5/horizon32/minibatch16384/mini_epochs6/AMPscale3/task:AMP=.5:.5/save100/max_epochs1000000/save_intermediateFalse. source 원본config와CPU검증.

## 우리 모델에 대응한 부분과 한계

- 우리 Stage1V2Fixed_00078000의GTA/네트워크/30D물체토큰/목표·HOLDING_AT graph입력 유지. 원본265D/42D carry tokenizer 모델을실행하는것은아니다. 그래프는관측과보조진단에사용되지만실제학습보상은원본carry다.
- 원본adaptation은old actor의self encoder/old task encoder/transformer/composer를동결하고newcarry tokenizer를학습한다. 우리모델에는이중old/new42D carry토큰이없으므로 **actor 물체토크나이저만학습,나머지actor/sigma동결**, critic전체와fresh판별자학습으로대응한다. 추가adapter나관측차원확장없음. 원본과아키텍처가달라동결단위까지문자그대로동일할수는없다.
- 정책/critic 및scene·value normalizer는78k에서복원한다. 사람normalizer는동결하고물체normalizer는적응시킨다. discriminator와AMP normalizer는새태스크학습용으로fresh 초기화한다. 우리critic의사전가중치유지는우리모델을사용하는부분이다.
- inherited MA relation 진단은관계기준상태일뿐actual reward가아니다. TensorBoard `reward_terms/{walk,handheld,carry,putdown,upperbody,power,total,amp,combined}`를본다. power=0, speed penalty는carry에포함. combined=.5total+.5amp.

실행 config.md. 본학습은 `table_tokenhsi_ft_active_run.json`, 원시자료 `runs/table_tokenhsi_20261026` 참조. 원본save_intermediateFalse이므로기본체크포인트는 nn/TableTokenHSIFT.pth이며100회마다최신본을저장한다. 학습상한1,000,000회도원본값그대로다.

검증: CPU3개통과(7함수AST/전PPO계수/환경).4096환경1회PPO·epoch1저장·모든165tensor finite. 동결74tensor는78k와bitwise동일,학습대상91tensor변경,actor/critic물체tokenizer gradient비영,사람normalizer동결확인. checkpoint를resume하지않고본학습도원본78k부터시작한다.
