# 原본 TokenHSI Stage1 학습·평가 설정 감사 — 2026-10-24

로컬 원본 `TokenHSI/` 코드/YAML/공식 stage1_eval.sh 기준. 논문 수치를 새로 측정한 것이 아니며 baseline과 학습 프로세스는 변경하지 않았다. 전체 설정은 project runs/tokenhsi_train_eval_settings_20261024/original_env_sim.json 및 original_train_network.json에 보존한다.

## 실행 모드

| 항목 | 학습 stage1_train.sh | 공식 평가 stage1_eval.sh |
|---|---|---|
| task | HumanoidTrajSitCarryClimb | 동일 |
| 환경수 |4096|512|
| 사람수 |환경당1명|환경당1명|
| 과제 |traj10%,sit30%,carry30%,climb30%|--eval_task 하나100%|
| 물체split |train|test (--eval에서 강제)|
| 초기화 |stateInit Random, 참조모션 랜덤frame|평가skill분포로 참조모션 랜덤frame|
| episode |SIT/CLIMB/TRAJ300step, CARRY600step|동일|
| 실시간 |제어30Hz, 명목10초/20초|실제일반종료299/599step(약9.97/19.97초)|
| Interaction Early Termination |SIT/CLIMB enabled, 성공근처frame누적약60step|disabled|
| 낙상종료 |enabled|enabled|
| 정책 |PPO+AMP 확률적학습|frozen checkpoint/player추론|
| 반복/seed |훈련seed CLI기준|공식script seed0, 프로세스내3repeat|

**stage1_test.sh의 --test만으로는 공식 eval이 아니다.** 해당script는16환경 viewer이고 --eval이 없으므로 객체split 및 초기화분포가 정식평가로 전환되지 않는다. 공식평가는 --test --eval --eval_task와512환경을 함께 전달한다. 공식run_eval은각환경의첫완료episode만repeat별로집계하고, 3repeat를 ObjectSet_test_0/1/2에 각각저장한다. 독립seed0/1/2실험이나자동mean±std출력이아니다. 별도후처리의0repeat제외여부는현재baseline에후처리파일이없어이문서에서확정하지않는다.

## SIT

| 항목 | 학습 | 공식 평가 |
|---|---|---|
| 물체 |StraightChair_Normal train49종|test26종|
| 종류내모델분포 |균등복원추출|동일|
| 모델교체 |환경생성때고정|동일;repeat마다새모델세트가아님|
| 초기skill |loco_sit50% + sit50%|loco_sit100%, sit0%|
| 초기자세 |각skill의참조모션에서sample_time_rsi|loco_sit참조모션에서sample_time_rsi|
| loco_sit 의자배치 |사람root XY 주변 반경U(1,5)m, 각도균등360°|동일|
| loco_sit 의자yaw |U(0,2π)|동일|
| 정면±45° 제한 |없음|없음|
| sit skill 의자배치 |참조모션의물체XY/yaw, native bbox 바닥배치|해당skill확률0|
| 목표 |물체pose로월드변환한config.tarSitPos|동일|
| 물리 |native메시,고정물체|동일|

SIT의tarSitPos는실제좌판위골반목표로저장되어있으며, 우리의chair seat+10cm와수치상일치한다. 원본코드에서SIT목표에10cm를추가하는것이아니라이미저장된metadata를직접사용한다.

**sit50%가모든시작의절반이이미완료착석이라는뜻은아니다.** sit20개clip 중sampling weight에따라뽑고허용frame에서랜덤초기화한다. YAML의rsi_skipped_range로심한자기관통/금지구간을제외한다. 접근/이행frame도포함될수있다. loco_sit12개clip은실제걷기데이터이며, 평가에서도참조초기화자체는남는다. 평가가착석모션RSI를쓰지않는것과reference state init함수자체를쓰지않는것은다르다.

## CLIMB

| 항목 | 학습 | 공식 평가 |
|---|---|---|
| 물체 |Box18/Cabinet10/Table_Square10|Box18/Cabinet9/Table_Square8|
| 종류확률 |각1/3|각1/3|
| 종류내모델 |균등복원추출|동일|
| 초기skill |loco_climb50% + climb50%, climbNoRSI0%|loco_climb100%,나머지0%|
| 초기자세 |각skill참조모션에서sample_time_rsi|loco_climb참조모션에서sample_time_rsi|
| loco_climb 물체배치 |사람주변반경U(1,10)m,360°|동일|
| loco_climb 물체yaw |U(0,2π)|동일|
| climb skill 물체배치 |참조물체XY/yaw, native bbox바닥배치|해당skill확률0|
| 목표XY |물체중심|동일|
| 목표Z |object_root_z+config.tarSitPos.z+char_h|동일|
| char_h |phys_humanoid_v3.xml은0.94m|동일|
| 물리 |native메시,고정물체|동일|

종류확률1/3은총모델수에비례한추출이아니며,512환경을171/171/170으로강제하지도않는다. Box의test18크기는XY=(.4,.5),(.5,.3),(.6,.4)m 각각Z=.2,.3,.4,.5,.6,.7m. Chair CLIMB와Table SIT는원본기본task구성이아닌우리의교차물체확장이다.

## 성공·종료

SIT/CLIMB검사점은골반root XYZ. 성공은목표3D거리≤.2m인frame이한번이라도있는ever(`_success_buf`누적). 방향/발높이/속도/유지시간/착석접촉은성공추가조건없음. 영역성공은원본에없다. 성공precision은성공episode별최소거리이며최종frame오차와다르다.

낙상은발을제외한접촉과신체높이기준을AND, threshold height.15m, 최초2step영향제외. TRAJ에는목표와너무멀어지는failDist4m종료추가. 학습IET는근접성공frame카운터만증가시키므로연속2초유지조건이아니며,평가에선꺼진다. 평가시간은SIT/CLIMB의Short300step이우선한다. --episode_length는긴episodeLength만바꾸므로SIT시간을바꾸려면Short도확인해야한다.

## 보상(성공식과구분)

- 전체PPO 보상혼합은task.5 + AMP.5, power penalty coefficient.0005.
- SIT: 근접골반목표reward.7 + 접근속도1.5m/s reward.3. 가까운구간선속도/각속도벌점.
- CLIMB: 접근속도reward.2 + 골반목표reward.5 + 발높이reward.3, 선속도벌점. 발높이는보상에는있지만평가점성공식에는없음.

## 나머지 task 초기화

TRAJ는학습/평가모두loco100%. Carry는학습loco_carry50/pickUp10/carryWith30/putDown10/omomo0%,평가loco_carry100%. Carry상자학습XYZ.2~.6m(.05m격자),평가9개지정크기(.22,.27,.32,.37,.42,.47,.52,.57정육면체와.30×.30×.40). Carry상자는동적,출발/도착받침높이randomHeightProb.5로무작위화. 해당세부수치는전체JSON과원본YAML참조. Carry기본episode20초, TRAJ10초.

## 정책·물리 설정

정책354D=사람223+TRAJ20+SIT38+CARRY42+CLIMB27+mask4. 신체/token/task별encoder와Transformer4층,2head,64feature,feedforward512,tokenizer256/128,composer1024/512. checkpoint학습이력과기본max_epochs1e6은구분;공개ckpt_stage1.pth는epoch40000.

PPO lr2e-5 constant,gamma.99,tau.95,horizon32,minibatch16384,mini_epochs6,e_clip.2,critic_coef5,grad_norm1,entropy0. AMP obs10프레임(현재+과거9),demo/replay buffer200000,AMP batch512/minibatch4096,disc lr는같은optimizer계약. optimizer계약전체는원본train JSON참조.

환경spacing5m,dt1/60·controlFrequencyInv2→30Hz,PD제어,powerScale1,selfcollision enabled,foot contact bodies. 바닥정/동마찰1,restitution0. PhysX substeps2,solver_type1,num_position_iterations4,num_velocity_iterations0,contact_offset.02,rest_offset0. 물체VHACD resolution100000/max_hulls128/max_vertices64. nativeSIT/CLIMB물체고정.

## 코드 근거

- TokenHSI/tokenhsi/scripts/tokenhsi/stage1_train.sh 및stage1_eval.sh
- TokenHSI/tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml
- TokenHSI/tokenhsi/env/tasks/multi_task/humanoid_traj_sit_carry_climb.py: ObjectLib(47),eval override(303),_reset_task_sit(1398),_reset_task_climb(1625),_reset_task_indicator(1679),_reset_ref_state_init(1794),_compute_metrics_evaluation_v1(1215),compute_humanoid_reset(1999).
- TokenHSI/tokenhsi/learning/transformer/trans_players.py: run_eval(216).
- TokenHSI/tokenhsi/utils/motion_lib.py: sample_time_rsi(155).

## 현재우리 chair 학습/최근평가와의차이

우리학습은2명/4의자,SIT100%,정면±45° 및native cache확률50%,late/full pool사용. 원본학습은1명,정면강제없음,loco_sit/sit50:50의원본참조모션초기화. 최근우리평가는1명1의자,범용loco시작,360°,30초,독립3seed 및추가영역지표이므로원본공식eval전체와동일하지않다. 다만**360° 시작자체는원본SIT eval에서도사용**한다. 현재낮은성공률의원인을정면차이하나로확정할수없다. 이번감사는조회/문서화만수행,학습과평가코드는변경하지않음.
