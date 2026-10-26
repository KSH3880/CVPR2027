# koo_1021 현재 GPU1 학습 설정

2026-10-26 기준으로 실행 중인 `stage1_push_finetune`의 CLI와 저장된 `relation_config.yaml`, 실제 코드를 대조했다. 실행 방법과 GitHub 전달 절차는 [팀 가이드](../STAGE1_PUSH_FINETUNE_GUIDE.md)를 따른다.

## 코드와 가중치의 출처

- 코드 기반: `KSH3880/CVPR2027`의 `stage1_stage2_base`, commit `28f02b3278f0a3e703cfa22a1e8cc376bb8ec512`.
- 실제 코드는 `ports/stage1_base_push/tokenhsi/`에서 실행한다.
- 초기 모델: 원본 `Stage1V2Fixed_00078000.pth`. 기존 SIT·CLIMB·carry AT·carry ONTOP에 PUSH_AT을 추가했다.
- actor/critic 과제 embedding은4→5행. PUSH 행은carry_at 행을 복사해 독립적으로 학습한다.
- AMP는carry/sit/climb에PUSH family를 추가했다. 새PUSH 입력열과정규화초기값은carry에서복사한다. 기존1320D→1330D로확장한다.
- 초기 전이에서는optimizer/epoch/frame을초기화했다. 현재실행은그이후의파인튜닝9000회저장본에서재개했다.

## 유효한 학습 값

| 항목 | 현재 값 |
| --- | --- |
| config / 실험명 | `stage1_push_finetune` / `Stage1PushFinetune` |
| 사람 / 실제 상자 / 환경 | 2 / 4 / 2048 |
| 최대 에피소드 | **1200step**: CLI `--episode_length 1200` 적용. 소스 YAML 기본은600 |
| seed | 42 |
| 설정상 과제 비율 | PUSH50%, SIT5%, CLIMB12.5%, carry AT/ONTOP 각16.25% |
| 과제 배정 | agent별 size-conditioned sampling; 실제 배정은 생성된 물리asset의 호환성을 따름 |
| 학습률 | 2e-5, constant |
| 과제/AMP 보상 비율 | 각0.5 |
| 정책 구조 | 별도actor/critic, Transformer4층·2head·64D, task-shared embedding·TaskActionFusion·GTA |
| 입력 | 정책634D·8토큰, AMP1330D |
| 정규화 | 입력과AMP 통계를 학습 중 갱신 |
| teacher | 별도 frozen teacher와KL 보존 손실 없음 |

## PUSH 상자와 여분 상자

PUSH profile은 X/Y각1.60~2.20m(0.1m격자), Z1.45~1.70m(0.05m격자), `min(X,Y)≥1.1Z`를만족하는167종이다. 질량20kg, 상자마찰0.4, 바닥마찰1.0, 선형/각감쇠0.01, 반발0이며 별도밑판은없다.

**이 원본 실험은 PUSH 담당자의 여분 상자에도 PUSH 크기·물성을 적용한다.** 코드의 `_push_asset_mask`가 `[push_owner0, push_owner1, push_owner0, push_owner1]`이고 `sample_asset_sizes()`도같은소유자의payload와여분에PUSH profile을각각뽑는다. 따라서두사람모두PUSH인장면은실제4상자모두큰PUSH상자다. 기존과제담당자의payload/여분은기존과제크기와SUPPORT규칙을사용한다.

복사본 `koo_push_1026`에적용한“PUSH는담당물체에만, 여분2개는항상원본SUPPORT” 변경은이실험에적용하지않았다. 원본학습설정을그대로공유하기위한문서다.

## 위치와 RSI 샘플링

1. 사람의보행시작점을기준으로상자중심을360°·반경1~10m에서제안하고상자yaw를무작위로뽑는다.
2. 목표는환경중심XY각±4.5m에서제안한다. 사람/상자와1m이상, 상자정면±45°, 두이동경로의여유등을검사해거부샘플링한다. 상자→목표거리를특정반경구간에서균등하게뽑는방식은아니다.
3. PUSH agent마다50%확률로보행, 50%확률로PUSH RSI를선택한다.
4. RSI이면승인bank949후보중모션/시점을뽑고, **목표를고정한채상자를진행도만큼이동**한다.

```text
box_progress = 0.85 × phase
RSI 상자 위치 = 최초 상자 위치 + box_progress × (목표 − 최초 상자 위치)
RSI 후 남은 거리 = 최초 상자→목표 거리 × (1 − box_progress)
```

이후사람은이동한상자의목표반대쪽접촉면에원본native RSI자세로정렬한다. 따라서사람→상자1~10m는보행시작에해당한다. RSI시에는사람이상자근처에서시작한다. 원본클립의실제사람이동거리(m)는위공식의상한으로사용하지않는다.

상자→목표거리는RSI전최소1m이고별도거리상한은없지만좌표와배치조건에의해제한된다. 기존CPU4096상자표본의1.000~19.088m는관측범위이며설정상한이아니다. RSI후에는1m미만이남을수도있다.

RSI는clip0/3/4/8의949후보를사용하고clip7은제외한다. 각클립후반30%의pool을70%확률로요청한다. 기존SIT/CLIMB/carry의RSI확률·모션은원본경로를유지한다. 평가기본은PUSH 보행시작이다.

## 보상

- CONTACT: 목표에대응하는유한상자면과양손의XY거리기반. 점하나에손을맞추는방식이아니다. 양손면거리중최댓값으로상태점수를만들고양손반력/상자힘조건을함께사용한다.
- 접근progress는사람→상자중심거리기반이다. CONTACT와AT의state/progress/success가각각0.2다.
- AT은원본Stage1 목표도착보상·성공판정을유지한다. 학습성공은기존20cm/Z오차1mm조건이며, 평가문서의점/영역재채점과구분한다.
- 본인과제보상100%, 동료보상0%, 기존사람CPA충돌페널티를유지한다.

상세구현은port의 `stage1_push_reward.py`, `push_contact.py`, `humanoid_ma_stage1_push.py`를따른다. `physical_push`에다른진단용상수들이남아있다고그값을실제AT성공판정으로해석하지않는다.

## koo_push_1026과 구분

| 항목 | 이 원본 `koo_1021` | 복사본 `koo_push_1026` |
| --- | --- | --- |
| config | `stage1_push_finetune` | `stage1_push_rsi_uniform` |
| RSI 상자 이동 | `box_progress`만큼목표쪽으로이동 | 이동하지않음 |
| RSI 목표 거리 | 기존배치후진행도에따라남음 | 2~10m 반경균등제안·정면±45° |
| PUSH 담당자의 여분 상자 | 큰PUSH profile·20kg·마찰0.4 | 원본SUPPORT 크기·밀도100·마찰1.0 |
| 계약/초기 checkpoint | 원본fine-tune계약 | uniform+SUPPORT계약, 서로직접resume불가 |

이문서작성에서는원본의코드·YAML·진행중학습을수정하지않았다. 비교표는두실험을섞지않기위한것이다.
