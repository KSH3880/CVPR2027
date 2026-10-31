# TokenHSI + PUSH (FT + KD)

## 확정한 구성

TRAJ는 제외한다. PUSH는 지정 Stage1 26k의 실제 환경을 사용하고, 기존 SIT/CARRY/CLIMB는 원본 TokenHSI 환경에서 고정 teacher KL로 보존한다. 추론에는 student 하나만 사용한다.

**26k는 PUSH 환경의 기준이다. Student 초기 모델과 teacher는 원본 TokenHSI Stage1이며, 우리26k 정책 가중치를 student에 넣는 것은 아니다.**

| 영역 | 환경 | 학습 |
|---|---|---|
| PUSH | Stage1 26k의2인·4상자 mixed scene | PUSH 슬롯만 PPO+AMP |
| 참조 장면의 비PUSH 사람 | 같은 참조 장면에서 student로 제어 | loss 없음, 장면 상호작용 유지 |
| 기존3과제 보존 | 원본 TokenHSI SIT/CARRY/CLIMB | teacher forward KL만 |

참조 장면을 PUSH-only로 바꾸지 않는다. 26k 원래의 PUSH50%/SIT5%/CLIMB12.5%/AT16.25%/ONTOP16.25% prior와 크기 조건부 과제 선택까지 보존한다. 다른 과제인 동료가 있는 장면도 유지해야 PUSH가 경험하는 간섭 분포가 원본과 같기 때문이다. 참조 장면의 비PUSH 사람에는 native KD에서 보존한 student를 적용하며 별도의 teacher action으로 simulator를 진행하지 않는다.

## 환경 코드와 정책의 분리

`push_reference/`는 Stage1 26k checkout의 env/utils 독립 snapshot이다. Python import namespace와 RSI cache 코드 위치 참조만 바꿨다. 사람·상자·목표 샘플러, 전체 reset 거부검사, reward, 물리, RSI sampler는 원본 함수를 그대로 호출한다. 원본 파일 hash는 `push_reference/PROVENANCE.json`에 기록돼 있다.

참조 정책/관계토큰/GTA/coordination은 사용하지 않는다. 참조 환경의 물리 상태를 사람별 TokenHSI 관측397D로 투영하고 같은 student를 독립 실행한다. 한 사람의 입력에는 담당 물체와 목표를 넣으며 별도의 동료 관측을 추가하지 않는다. 원본 environment가 bookkeeping용으로 계산하는634D scene 관측은 student 입력에 사용하지 않는다.

기존 스킬은 `amp_humanoid_traj_sit_carry_climb.yaml`의 원본 환경/관측/초기화를 사용한다. TRAJ 선택 확률만0으로 두고 SIT/CARRY/CLIMB를 reset 시 균등 선택한다. Teacher도 원본 TokenHSI `ckpt_stage1.pth`이며, 예전39k 가중치는 사용하지 않는다.

## PUSH 전용 토큰과 입력

| 입력 블록 | 차원 | PUSH 수행 시 |
|---|---:|---|
| 자기 humanoid 상태 |223|사용 |
| 기존 TRAJ/SIT/CARRY/CLIMB 관측 슬롯 |127|0으로 마스킹 |
| PUSH 담당 상자와 목표 |42|사용 |
| 과제 one-hot |5|`[0,0,0,0,1]` |
| 합계 |397|사람마다 독립 입력 |

PUSH42D는 상자 선속도3·각속도3·위치3·회전6·8개 꼭짓점24·목표 위치3이다. 실제 상자 크기를 꼭짓점 값에 반영하며 TokenHSI의 작은 상자 크기로 축소하거나 제한하지 않는다. 무게는 별도 입력 채널 없이 물리 환경에서 부피×밀도3으로 적용한다.

PUSH 전용 tokenizer를 추가하고 기존 CARRY tokenizer에서 복사해 초기화한다. 기존 TokenHSI Transformer와 action head를 통해 사람별32차원 행동을 출력한다. Critic과 AMP 판별기의 추가 입력 연결도 확장한다. PUSH 토큰과 공유 student는 PPO+AMP로 학습하며 teacher KL을 PUSH 샘플에 적용하지 않는다.

동료와 여분 상자는 같은 물리 장면에 있지만 해당 정보를 별도 관측 토큰으로 student에 제공하지 않는다. 따라서 우리 모델과 입력 정보 범위가 다르며 논문에서 독립 정책 baseline임을 명시한다. 자세한 좌표 변환과 입력 설명은 [PUSH_INPUT.md](docs/PUSH_INPUT.md)를 참조한다.

## PUSH 조건

기준: `Stage1JointPushDensity3Cone90Finetune_00026000.pth`, SHA256 `ec4215803b529bab1dd1be430154604418392e24faed59bca69b5c61adafe67c`.

| 항목 | 원본과 동일하게 적용 |
|---|---|
| 장면 | 사람2명·상자4개(담당2+support2), 원본 과제 혼합 |
| 크기 |167개 profile 균등, XY1.6~2.2m/0.1m, Z1.45~1.7m/0.05m, min(X,Y)≥1.1Z |
| 질량 | 부피×밀도3,11.136~24.684kg, 생성 시 고정 |
| 마찰 | PUSH0.4, support는 원본 조건 |
| 사람/상자 | reference root에서 상자를2~10m/360°로 제안. loco 사람 heading을 상자 방향으로 강제하지 않음 |
| 목표 | 상자→목표 반경U(2,10)m 제안 후 초기+Y±45°와 전체 장면 거부검사 |
| 간섭 | 목표↔사람≥1m,3.5m 경로 분리·동료·다른 상자·support 검사, 원본 최종reset 거부/재시도 |
| AMP | 동일 밀기21개/보행4개, PUSH family 내부50:50,10frame |
| RSI | 동일949후보/4모션, loco50/접촉50, 후반30% 후보 요청70% |
| RSI 배치 | 원본 DOF/속도 유지, 원본 align_push_pose/safe_place, stand_off0.06m |
| RSI 목표 | 진행률로 상자 이동하지 않음, 잔여2~10m, 최종 배치 검사 |
| CONTACT | 목표방향60° 이내 공통 면, 최악 양손 XY거리, phi>0.9+양손/상자 법선힘≥0.5N |
| 접근 보상 | CONTACT는root→상자, AT는상자→목표, 원본 거리 함수 |
| AT 성공 | 중심3D거리≤20cm 및 높이오차≤1mm |
| 합산 | CONTACT/AT 각각 state·progress·success0.2, AT 성공 시 두 관계 포화 |
| 추가 비용 | 원본 동력−0.0005 및 CPA 사람 충돌 비용 포함 |
| 속도 보상 | 없음, 이동량 progress gate 없음 |
| 길이/종료 |1200step/40초, 원본 낙상·동료 낙상에 따른 scene reset 유지 |

거리/각도 범위는 거부 전 제안 분포다. 최종 분포는 간섭·목표최소거리 등의 거부에 조건화된다. 단순히 같은 범위에서 독립추첨하는 구현과 구분한다. 모션/asset/mesh/bank hash를 대조한다. Bank의20kg screening 출처를 보존하며 밀도3의 새 전수screening을 뜻하지 않는다.

## PUSH AMP 상세

| 항목 | 확정 설정 |
|---|---|
| PUSH expert |우리26k와 동일한 밀기21개 전체 유지 |
| 보행 expert |우리26k와 동일한 전용 보행4개 |
| 파일 검증 |PUSH21개와 보행4개 모두 SHA256 대조 |
| PUSH family 내부 선택 |보행50% / 밀기50% 확률 |
| 그룹 내부 clip 선택 |각 clip의 weight1.0, clip 균등 |
| History |10프레임 |
| PPO 보상 합산 |`0.5 × 환경 보상 + 0.5 × AMP 보상` |
| Discriminator reward scale |2 |
| 학습에 쓰는 AMP 데이터 |PUSH 슬롯의 real/replay와 PUSH expert만 |

참조환경의 family4개 라벨을 student의 task5개 라벨로 변환한다. 신체 특징과 history 값은 보존하고 PUSH에는 PUSH 라벨을 사용한다. 기존 과제의 native KD 샘플은 discriminator 학습이나 AMP replay에 넣지 않는다.

**AMP의 보행/밀기50:50과 아래 RSI의 보행/접촉50:50은 서로 다른 설정이다.** 인턴의 AMP20:80은 사용하지 않는다.

## PUSH RSI 상세

| 항목 | 확정 설정 |
|---|---|
| 초기화 선택 |보행 reference 시작50% / PUSH 접촉 RSI50% |
| Bank |`tokenhsi/data/push_rsi_bank.json`, 우리26k와 동일한949후보 |
| 후보 모션 |0-based ID0/3/4/8의4모션 |
| 모션 선택 |유효 후보가 있는 모션의 원래 clip weight 기준 |
| 후반 선택 |70% 확률로 선택 모션의 유효 후보 중 마지막30%, 나머지30%는 전체 후보 |
| 자세 |원본 DOF position/velocity 보존. 팔 높이를 별도로 보정하지 않음 |
| 정렬 |원본 `align_push_pose`와 `safe_place`, stand-off0.06m |
| 상자 진행률 |후보의 `box_progress`로 상자를 미리 이동시키지 않음 |
| 목표 잔여 거리 |RSI 배치 후에도 상자→목표2~10m 유지 |
| 최종 배치 |원본 사람·물체·경로·목표 거리 거부검사 적용 |

후반30%는 전체 원본 모션의 마지막30%가 아니라 해당 모션에서 살아남은 유효 후보의 시간순 후반이다. 949후보 전체를 먼저 균등 선택하는 방식과도 다르다. 인턴의11후보는 사용하지 않는다. 학습에서는 이 혼합을 유지하고 현재 평가 스크립트는 원본26k의 loco-only 조건을 사용한다.

## Loss와 라우팅

```text
L = L_PPO(PUSH) + c_v L_value(PUSH) + c_b L_bounds(PUSH)
    - c_e entropy(PUSH) + c_d L_disc(PUSH)
    + lambda_KD mean_native_KD KL(teacher || student)
```

KL은 Gaussian32차원 합산 후 해당native KD 샘플 평균이다. Teacher 모델/RMS는 고정한다. 기본lambda=0.001, validation 후보0.0001/0.001/0.01이다. 허용 보존 성능 저하와validation seed를 먼저 정하고 최종test로 선택하지 않는다.

PUSH mask와 KD mask는 별도다. 참조 장면의 비PUSH 슬롯은 둘 다false다. PPO advantage 정규화·critic/value RMS·AMP real/replay/demo는 PUSH만 사용한다. Native KD에서는 value/AMP/task-reward 기반RL이 없다. Feature/value distillation은 없다. Student actor RMS는 전체 관측에서 갱신한다. 같은 초기teacher에서student를 확장하되 PUSH tokenizer는carry 복사, 추가critic/판별기 입력 연결은0 초기화한다. 원본 TRAJ 슬롯은 체크포인트 호환용으로만 남긴다.

## 학습 규모와 계산 예산

기본 CLI `--num_envs 2048`은 **참조scene2048개 + native KD환경2048개**를 만든다. 참조scene은2명이라 policy row4096개, native KD row2048개, 합계6144개다. 한 rollout32step이면 총196608 actor step이다. PUSH 슬롯은 원본prior에 의해 대략 절반이므로 PUSH약65536step + native KD65536step + 비PUSH scene 제어약65536step이다. 정확한PUSH 수는 asset prior/scene sampling에 따라 달라져로그로 기록한다.

이 구성은 과거 “단일2048환경을 절반씩 고정분할”보다 메모리·물리 비용이 크다. Native KD simulator는 spawn worker 프로세스로 격리하고 CPU NumPy 배열을 pipe로 복사해 관측/행동을 전송한다. 참조simulator와Isaac Gym instance를같은프로세스에동시생성하지않는다. 이 IPC 비용은wall time에포함한다. Simulator는 분리된 두 pool이며 main learner는6144개의독립 policy row로 받는다. 추가 RL loss는PUSH에서만 계산된다. 기본26000epoch이나 Ours와비교할때는iteration 대신실제PUSH step·KD step·전체scene step·추가wall time 및pretrained 사용을보고한다.

## 평가

PUSH는 참조환경의공식loco-only 평가 조건을 사용한다(2인·4상자, PUSH preset).512scene이면1024사람 표본이다. Student를결정적평균행동으로실행해첫episode의final/ever 성공·낙상·종료잔여거리를분리한다. 현재 RSI eval override는 명시적으로거부한다. Native기존3과제는원본TokenHSI 평가규칙을유지한다. TRAJ eval은거부한다.

원본mixed scene의동료는student가제어하므로 Ours의정책과상호작용결과는달라진다. 동일초기seed만으로동일실현장면/궤적을보장하지않는다. 최종논문평가에는공통저장initial plan을사용하고,TokenHSI CARRY와우리AT/ONTOP 과제대응을명시해야한다.

## 코드 검증과 실행 상태

검증은CHANGELOG와 `output_etc/check/reference_integration/`에기록한다. 참조 sampler/reward/RSI 소스보존,2인layout의거리·cone·간섭조건, AMP family변환, 실체크포인트복원·기존행동보존, PUSH/KD/무학습slot의optimizer분리를CPU로검사한다.

본학습을시작하지않았으며GPU policy rollout/PPO 통합·장기성공률은별도검증대상이다. CPU sim검사는학습이아니고작은scene으로물리/reset/관측연결만확인한다.

CPU 테스트12개 통과. 단독 PUSH 참조 환경의2scene/4명 reset과 물리3step을 확인했다. 프로세스 격리 후에는 참조2scene/4명+native2환경의 관측 `[6,397]`, 물리3step과 관측/보상 finite를 확인했다. 실제 teacher/student 초기 checkpoint를 strict 로드한 정책 행동3step에서도 native KD의 KL=0.0을 확인했다.

로그는 `output_etc/check/reference_integration/mixed_isolated.log`와 `policy_isolated.log`다. 초기 동일 프로세스의 simulator 동시 생성 실패와 수정 이력은 CHANGELOG에 남긴다. CPU 연결 검증은 GPU PPO 통합이나 장기 성공률 검증을 뜻하지 않는다. 본학습은 미실행이다.
