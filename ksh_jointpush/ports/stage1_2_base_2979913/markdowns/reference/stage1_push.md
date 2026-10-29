# Stage1 PUSH

`stage1_push`는 현재 Stage1의 네 과제에 독립 PUSH를 추가한 스크래치 실험이다. 기본2명·2048환경·4물체·900step이며 같은 `tokenhsi` 런타임을 사용한다. Stage2의 공동 PUSH는 포함하지 않는다.

## 새 학습과 기존 PTH

새 학습의 크기 생성 prior는 HOLDING0 / SIT .10 / CLIMB .15 / AT .25 / ON_TOP .20 / PUSH .30이다. 사용자 지정 비율을 사용한다. 크기는 생성 때 고정되고 비PUSH 슬롯은 가능한 과제를 조건부로 다시 샘플링한다. 기본은 checkpoint 없이 학습하며 기존 Stage1 reward·크기·물리·RSI·PPO 설정을 유지한다. 공통 horizon은 모든 실험에서900으로 변경했다.

`checkpoints/Stage1PushFinetune_00021000.pth`는 Downloads의 원본 PTH를 가리킨다. 이 저장본은78k에서 시작한 finetune으로 PUSH50 / SIT5 / CLIMB12.5 / AT16.25 / ON_TOP16.25%를 사용했다. 일반 평가는 checkpoint의 저장 env/sim을 복원하므로 이 분포를 유지하고 시간만900으로 맞춘다. 옆에 sidecar가 없으면 PTH 내부 `relation_experiment_config`를 읽는다. 새 scratch 설정으로21k를 resume하는 것은 계약 불일치로 거부한다.

## PUSH 동작

- X/Y1.6–2.2m(.1 격자), Z1.45–1.70m(.05 격자), `min(X,Y)≥1.1Z`인167종을 사용한다. 밀도3kg/m³·마찰.4, 질량11.136–24.684kg이다. 기존 과제/받침은 밀도100이다.
- loco 사람→상자2–10m·360도, 상자→목표2–10m이다. 목표 방향은 상자 로컬+Y 기준±45도이며 경로/물체/사람 간섭을 거부한다. source는 바닥, target은 초기 payload 중심 높이로 고정한다.
- 학습 PUSH 초기화는 push_loco50%/접촉 RSI50%다. 접촉 후보949개, clip7 제외, 후반30% 요청70%다. bank는 원래20kg screening 결과를 재사용하며 `reference_mass:20`이 실제 상자 질량이라는 뜻은 아니다. reset에서 현재 기하/간섭을 검사한다.
- CONTACT 관계12 + AT 관계7을 사용한다. CONTACT는 양손의 밀기 면까지 거리·양손 반력·상자 힘을 검사한다. state/progress/success 가중치는 각.2다. AT 성공은3D20cm 및 Z1mm를 쓰고 CONTACT prerequisite gate는 없다. AT 성공 시 CONTACT도 현재 frame 기준 포화한다. 별도 AT progress 이동 gate는 설정하지 않는다.
- AMP는 기존 carry/sit/climb에 push를 추가한4 family다. PUSH 전문가 혼합은 push_loco50%/push50%다. task embedding은5×64, 판별기 입력은1330(기존1320)이고 actor/critic의 나머지 구조는 동일하다. 기본 scratch는 모든 학습 대상·RMS를 업데이트하며 teacher/encoder 동결은 없다.

## 일반·단독 평가

일반은 저장된 학습 인원·물체·분포·배치를 유지한다. 단독은 Stage1 계열 전용 진입점에서 실제1명으로 AT/ON_TOP/SIT/CLIMB/PUSH를 각각 독립 평가한다. AT만9종 크기 override이며 PUSH는 학습167종을 사용한다. 단독 AT/SIT/CLIMB/PUSH는 물체1개, ON_TOP은2개, distractor는 없다. 원격2명 고정 sampler를1/2명으로 일반화했다.

정식 평가는900step·512환경×독립 seed0/1/2·loco100%다. PUSH의 loco는 전용 push_loco 모션에서 뽑는다. 같은 궤적의 점/영역을 계산하며 PUSH는 payload 중심→고정 AT 목표의3D거리20cm로 두 기준이 같다. CONTACT·힘·유지시간은 평가 성공 조건에 추가하지 않는다. 개인 낙상 frame까지 채점하고 그 사람의 drive만 해제하며 다른 사람은 계속 수행한다.

결과는 `macro4`(기존네행동), `macro5`(PUSH포함다섯행동)를 구분한다. `macro`는 모델의 전체 행동 평균이다. 각 seed에서 행동별 성공률을 같은 비중으로 평균한 뒤 세 seed 평균/표본std를 구한다. [평가 규약](evaluation.md)과 [실행 명령](../config.md)을 따른다.

## 코드·데이터

- `humanoid_ma_push.py`: 자산/물성·접촉 RSI·1/2명 reset·physics contact 수집.
- `utils/stage1_jointpush.py`, `jointpush_scene.py`: sampler/물성/저장 계약, 경로·cone·distractor.
- `jointpush_geometry.py`, `jointpush_rsi.py`, `jointpush_rsi_geometry.py`: 손 목표·bank 해시·pose 정렬/기하 검사.
- `stage1_push_network.py`:5과제 모델 및 선택적인78k 초기화 도우미. 기존4과제/3family 모델은 기본값을 유지한다.
- `dataset_stage1_jointpush.yaml`, `stage1_jointpush_rsi_bank.json`: 원격 manifest/bank의 bytes를 유지한다. PUSH21개 raw 모션은 Downloads 패키지와 bank SHA256이 모두 일치하며 `jointpush_motions/` 아래 읽기 전용 심링크로 연결한다.

원본 구현은 [stage1_2_push_stopgate / a26d27b](https://github.com/KSH3880/CVPR2027/tree/a26d27b9b247f09e87f77ba74cae0aed860c4206)에서 필요한 Stage1 부분만 이식했다. 원격 문서의 별도 conda 환경·과거 PUSH 실험·Stage2 coupled/adapters는 가져오지 않았다. 현재 BEFORE는 기존4과제 Stage1을 전이하며 PUSH21k를 BEFORE에 직접 전이하는 기능은 추가하지 않았다.
