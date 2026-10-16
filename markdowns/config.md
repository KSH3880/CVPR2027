# Multi-Agent Carry 실행 가이드

현재 실행 가능한 실험은 원본 1번과 Stage 1 **27·28·30·31·32·33·34·35·36·37번 및 paired/unified 변형**, Stage 2 **29번·SIT plane·Joint carry task CA 변형**이다. 과거 9~26번 config와 전용 실행 스크립트는 정리했다. 모든 명령은 저장소 루트에서 실행한다.

## 빠른 확인

- **SA4 GraspView**: [설정·명령](#stage2-sa4-graspview). 공동에서 사람별 반대 앵커 관측→자기 human64→기존 SA4. 단독/AMP/RSI/보상 유지, Stage1 전이 후 새 학습.

- Stage1 파지 중심 이동: [뷰어 명령](#stage1-grasp-shift). 두 사람이 같은 payload·목표를 공유하고 Stage1 CarryDistill30000으로 각각 추론한다. `SHIFT=0` 실제 중심 / `SHIFT=1` 각자 반대 앵커 관측; 학습 없음.

- **LocoPlatform human SA4**: [설정·명령](#stage2-locoplatform-human-sa). task CA를 human SA4로 교체; SELF/COUPLED/NONE bias 모두 사용, Stage1 전이 후 새 Stage2 학습.

- LocoPlatform 공동 query의 단독 task 차단 토이: [뷰어/명령](#locoplatform-ca-solo-mask). 기본 Platform22000·6명·GPU5, 공동 carryWith/단독 loco; `SOLO_MASK=0` baseline, `1`은 공동→단독 task만 차단한다.

- **LocoPlatform NONE 차단 학습**: [실험/명령](#stage2-locoplatform-nonemask). Stage1 전이→새 Stage2; Actor/Critic CA4 모두 학습·추론에서 NONE 차단. 기존 추론 토이와 별도.

- LocoPlatform NONE 차단 추론 토이: [설정/실행](#locoplatform-ca-none-mask). Actor CA4에서 NONE만 −∞; 기존 checkpoint, GPU5, 기본6명 혼합 장면. `NONE_MASK=0`으로 동일 설정 baseline 비교.

- LocoPlatform geometry task CA: [실험/명령](#stage2-locoplatform-geometry). 행동64 + 사람 환경 pose7 + 사람 기준 payload7/target7 →85→128→64; Actor/Critic CA4, 기존 RSI·AMP·보상 유지.

- 같은6명 전원 loco의 Stage1 SA: [QK/softmax/실제 SA 그림](../output_etc/platform_attention_check_20261014/sa_check/sa_layer4.png)·[결과](../output_etc/platform_attention_check_20261014/sa_check/report.json). Stage1 원본 encoder/RMS와 일치, human query는 자기 human/payload/target에95.6~98.3%. CA 쏠림과 구분한 초기 forward 진단.

- Platform12500 6명 전원 loco CA: [bias 전 QK/softmax와 실제 attention](../output_etc/platform_attention_check_20261014/all_loco/qk/layer4_scene0.png)·[전체 수치](../output_etc/platform_attention_check_20261014/all_loco/qk/report.json). 공동 두 팀도 loco로 바꾼16초기 장면에서 공동 query의 단독 E/F task attention 평균72.2%; 기존 혼합 시작63.9%. 초기 forward 진단이며 별도 학습 config는 아니다.

- 추론 전용 attention 희소화: [Entmax1.5 토이](#entmax-inference-toy). GPU4·CPA Release1 epoch19000·2명/6명 각각16장면, softmax/SA/CA/둘 다 비교. 6명 CA 안정성 개선과2명 악화를 함께 기록했다.
- 촬영용 4명/6명·8물체·1환경: [공동/단독 혼합 VNC 데모](#joint-carry-mixed-scene-demo). `DEMO_AGENTS=4`는 공동 AT 한 팀 + 단독 AT·ON_TOP, 기본6명은 공동 두 팀. 공동 carryWith·단독 loco; CPA Release1/LocoPlatform checkpoint에 맞춰 YAML 자동 선택.
- 촬영용 CA만 entmax: [6명 CA Entmax 뷰어](#joint-carry-ca-entmax-viewer). Actor CA4만 α=1.5, SA/Critic 유지; 기본 checkpoint19000·별도 output.

- 손·팔 Masked AMP 토이: [비교 그래프](../output_etc/masked_amp_toy_20261014/amp_comparison.png)·[결과/해석 한계](../output_etc/masked_amp_toy_20261014/analysis.json). Platform8000 정책 동결·loco 장면의 일반/마스크 판별기만 같은 데이터로 재학습. 잡은 상태 점수는 높아졌으나 첫 HOLDING 전후 AMP 하락은 일반−0.098/마스크−0.164로 완화되지 않았다(최종 보상 기여 기준,12장면14개 이벤트). 바닥 집기 표본 없음; PPO 재학습 효과는 미검증. 재수집 인자는 `output_etc/masked_amp_toy_20261014/command.json`, 분석은 GPU5 runtime 설정 후 `python output_etc/masked_amp_toy_20261014/analyze.py`로 재현한다.
- 같은 토이의 [잡기 전 숙임·손 접근 비교](../output_etc/masked_amp_toy_20261014/reach_phase/pre_grasp_amp.png)·[수치/선정 기준](../output_etc/masked_amp_toy_20261014/reach_phase/analysis.json): 바닥3건·받침5건에서 기존 AMP 기여−0.267/−0.134. 일반 재학습도 점수가 크게 달라져 마스크만의 개선으로 해석 불가. 정책/판별기 추가 학습 없이 저장된 토이 판별기로 점수를 계산했다. 재수집 `reach_phase/command.json`, 분석 `python output_etc/masked_amp_toy_20261014/reach_phase/analyze.py`.

- 공동 loco payload 받침: [CPA Release1 LocoPlatform](#stage2-cparelease1-locoplatform). 공동 loco의50%만 받침 사용; payload 크기로 높이 상한 계산, 기존 RSI·AMP·보상 유지.

- 공동 pickup 집중 샘플링: [CPA Release1 PickupFocus](#stage2-cparelease1-pickupfocus). pickup 안에서 집기25~39번35%·들기55~80번35%·전체30%; 단계 확률은 유지.
- 공동집기 근처 CPA 해제: [FullPickup CPA Release1](#stage2-fullpickup-cparelease1). 공동 상대만, 두 사람 모두 상자 XY 반대각선+1m 안이면 CPA0. FullPickup 구조·RSI·AMP 유지.
- 최신 CA-only·전체 pickup: [Goal360 FullPickup](#stage2-goal360-fullpickup). Actor/Critic 모두 CA4→64입력 head, critic 전체 학습·actor encoder 동결, 공동 pickup100프레임.
- 공동 HOLDING state: [Goal360 TeamHold](#stage2-mixed80-locoamp-ca4-goal360-teamhold). 공동 개인 state만 반대 앵커 조합의 최소 점수로 교체; 개인 progress/success 유지, Move 없음.

- 공동 목표 360°: [Mixed80 LocoAMP CA4 Goal360](#stage2-mixed80-locoamp-ca4-goal360). 공동 목표·받침 방향을 독립 샘플링하고 AT putDown은 보정 reference 끝점을 사용한다.

- CA4 공동 이동 방향 보상: [Mixed80 LocoAMP CA4 Move](#stage2-mixed80-locoamp-ca4-move). 두 사람의 실제 이동 방향 점수의 최솟값×0.2, 목표 근처 pinning, Stage1에서 새 학습.
- 4층 task CA: [Mixed80 LocoAMP CA4](#stage2-mixed80-locoamp-ca4). 기본 LocoAMP 보상·AMP·RSI를 유지하고 CA+FFN+LayerNorm+residual 4층을 비교한다.
- 다른 서버 clone: [공동운반 데이터 준비](../joint_carry/README.md). 기존 `tokenhsi/data`를 `TOKENHSI_DATA_ROOT`로 지정하면 두 LocoAMP 실행 경로에서 자동 연결·AMP/RSI 해시 검사를 한다.

- 공동운반 방향 보상: [LocoAMP Align task CA](#stage2-mixed80-locoamp-align). 가장 먼 사람의 방향 점수를 공동 HOLDING 중 두 사람에게 공유.
- Align AMP 진단: [carryWith 직선 운반 토이실험](#align-amp-toy). 같은 checkpoint에서 AMP 보상 유지/제거를 짧게 비교한다.
- 자세 보존 보행 AMP·CA ablation: [Mixed80 LocoAMP 비교](#stage2-mixed80-locoamp). 후진/옆걸음 reference 추가, CA 유지와 Stage1 64입력 head-only를 동일 조건으로 비교한다.
- BEFORE 80%·독립 20%: [BEFORE task CA](#stage2-before-task-ca). 현재 AT prerequisite·Stage 1 과제별 RSI/AMP·공통 크기를 사용하는 별도 실험이다.
- 공동 80%·단독 20%: [Mixed80 task CA](#stage2-mixed80-task-ca). 고정 환경 분할, 단독 Stage 1 크기·RSI 재사용.
- 공동운반 Stage 2: [Joint carry task CA](#stage2-joint-carry-task-ca). Frozen task embedding·공유 상자·opposite-anchor 보상·paired RSI를 사용하는 전용 실험이다.

- Distill 실행: [carry 두 과제](#carry-task-embedding-distillation), [네 과제 전체](#four-task-embedding-distillation). 각각 별도 config/output/checkpoint를 사용한다.
- 최근 비교 대상은 **Task message / 공유 Task MLP / Task MLP split / Task embedding**이다. [네 실험 공통 설정과 차이](#task-4종-공통-설정과-비교), [최근 확인 기록](#task-3종-최근-확인-기록)을 먼저 확인한다.
- 실행 명령: [Task message](#unified-size-rsi-task-message), [공유 Task MLP](#unified-size-rsi-task-mlp), [Task MLP split](#unified-size-rsi-task-mlp-split), [Task embedding](#unified-size-rsi-task-embedding). 네 실험 모두 **2명·2048환경·4물체**, 전용 train/test/VNC와 output을 사용한다. 서로의 checkpoint를 resume/eval에 섞지 않는다.
- `split`은 **task별 concat 이후 MLP와 layer/head projection 분리**다. 역할 embedding은 공유하며, actor/critic은 독립이다. 두 MLP 실험 모두 관계 message는 없다.
- `task_embedding`은 역할 입력·중간 MLP·message 없이 task/NONE/SELF의 6개 embedding과 H/O/G 타입 쌍별 projection을 사용한다. Actor/critic은 독립이며 각 branch 안에서 모든 카테고리가 projection을 공유한다.
- 학습 중 HOLDING/AT/ON_TOP 성공률은 보상 primitive 지표다. 정책의 네 task 및 별도 평가의 운반 완수율과 구분한다. Bias는 원래 값과 같은 물리 타입의 NONE 대비 차이를 구분한다.
- 현재 GPU/PID·epoch는 문서의 과거 기록으로 판단하지 않는다. 실행 전 `nvidia-smi`, 실행 중인 `tokenhsi/run.py` 인자와 해당 run의 `relation_config.yaml`·`summaries/`·`nn/`을 확인한다.
- 완료된 검증·분석·영상은 `output_etc/`에 보관한다. `output/`에는 모든 본학습 결과와 `rsi_cache`를 유지한다. 2026-10-13 추가 이동15개(파일268개)는 [이동 목록](../output_etc/archive_20261013_cleanup.json)을 참조한다. 과거 로그/JSON 안의 경로는 당시 경로이며 이동 목록으로 대응한다.

## 공통 규칙

- 학습 인자는 `[num_agents] [num_envs] [num_objects]`; Stage 1 기본값은 `2 2048 3`이다. 짧은 확인도 환경 수 2048을 유지하고 `MAX_ITERATIONS`만 줄인다.
- 실행 전 `nvidia-smi`로 점유를 확인하고 명령 앞에 `TOKENHSI_GPU`를 지정한다. 실행 중인 프로세스는 임의 종료하지 않는다.
- 본학습 전 셸의 `MAX_ITERATIONS`, `OUTPUT_PATH`, `RESUME_CHECKPOINT` 잔여값을 확인한다.
- 학습은 기본적으로 scratch다. 기존 실험의 reward/checkpoint 계약을 섞지 않는다. Stage 2 전이는 아래의 `STAGE1_CHECKPOINT`를 사용한다.
- 평가·VNC 인자는 `<checkpoint.pth> [agents] [envs] [objects] [repeats]`다. `HEADLESS=0`은 로컬 viewer, `HEADLESS=1`은 화면 없는 평가다.
- 로컬 viewer와 서버 VNC의 상자·받침 색은 매 reset의 과제 배정을 따른다. 단독 대상은 해당 에이전트 색, 공동 대상은 노란색, 나머지는 회색이다. AT goal 점과 같은 과제·대상을 공유하는 마커도 공동이면 노란색이다. ON_TOP은 와이어 큐브, SIT은 수평 원, CLIMB는 3축 십자로 표시한다(shared9 시각화와 동일). 사람은 에이전트 색을 유지하고, 상자 정리 데모는 모든 상자를 같은 회색으로 표시한다. 실행 중인 뷰어는 다시 시작해야 반영된다.
- 공유 데이터 원본은 읽기 전용으로 취급하고 이 저장소에서는 심링크로 사용한다. 현재 로컬 링크는 `/home/cvlab/Desktop/CVPR2027/TokenHSI`로 연결된다(2026-10-07, 12개 링크 모두 유효). 서버 문서의 `/home/hwanhee/CVPR2027/TokenHSI`와 경로가 다르므로 다른 머신에서는 실제 링크 대상을 확인한다. RSI 캐시는 이 저장소의 `output/rsi_cache`다.
- 4명·16상자 정리 데모는 아래 **상자 정리 VNC 데모** 절을 따른다. 지정된 rescue checkpoint를 사용하는 평가 전용 실행이다.

## 현재 실험

**공동 loco payload 받침 선행 토이(현재 [LocoPlatform 실험](#stage2-cparelease1-locoplatform)에 연결):** GPU5·평가64환경에서 공동 AT/ON_TOP의 payload 아래에만40×40×2cm 받침을 추가했다. 바닥/20/40/60cm 높이를 비교하고, 높은 받침96장면의2초 지지 및 받침 제거 후 낙하를 확인했다. 기존 사람·목표·다른 상자·pickup/carryWith/putDown RSI·AMP 이력은 유지된다. CPA Release1 epoch7500의10초 추론에서 공동 HOLDING0/128이며, 재학습 개선 여부는 미검증이다. 학습·viewer용 실행 config가 아닌 별도 headless 검사다.

[검사 결과](../output_etc/payload_platform_toy_20261013/report.json), [검사 코드](../output_etc/payload_platform_toy_20261013/check.py). 저장소 루트에서 재현:

```bash
TOKENHSI_GPU=5 bash output_etc/payload_platform_toy_20261013/run.sh
```

추가 높이 검사에서는 원본 규칙의 최대80cm·중간 랜덤 높이는 지지됐지만, 2cm 미만은 두께2cm 판이 바닥과 겹쳐 밀려 올라갔다. 높은 분기만 최소2cm로 제한한 토이는 AT/ON_TOP256장면 모두 지지·목표 높이를 유지했다. [원본 높이 결과](../output_etc/payload_platform_height_sweep_20261013/report.json), [최소2cm 보정 결과](../output_etc/payload_platform_height_sweep_20261013/report_min2cm.json). 이 하한 보정을 새 LocoPlatform config에 적용했다.

```bash
# 원본 높이 규칙 + 바닥 근처/최대 높이 경계 검사
TOKENHSI_GPU=5 bash output_etc/payload_platform_height_sweep_20261013/run.sh
# 같은 장면에서 받침 높이 하한2cm 보정
TOY_MIN_HEIGHT=0.02 TOKENHSI_GPU=5 bash output_etc/payload_platform_height_sweep_20261013/run.sh
```

**CPA Release1 carryWith 힘·움직임 진단(2026-10-13):** epoch9500 고정 정책·GPU5·공동 AT/ON_TOP 각64장면·최대20초를 기록했다. 기존 Goal360/RSI 초기 속도를 유지하고 첫 종료 뒤 표본은 제외했다. 배치 조건 한 번 이상 만족은 AT24/64·ON_TOP48/64였으며 완전한 손 해제를 의미하지 않는다. 초록agent0/파랑agent1의 전진·후진·좌우 이동과 목표 방향별 결과는 [분석 JSON](../output_etc/cparel_carrywith_forces_20261013/analysis.json), [장면별 구간 수치](../output_etc/cparel_carrywith_forces_20261013/case_phases.json)이다.

**추가 API 검사:** [접촉점·힘 센서 토이 결과와 재현 명령](../output_etc/contact_api_probe_20261013/summary.json). GPU pipeline 접촉점 목록 API는 실행 후 사용 불가. 별도 GPU force sensor는 자유 상자의 마찰을 읽지만 몸체 전체 합력/토크이며 상대별 분리는 없다. 실제 humanoid 손 센서는 비접촉 상태에서도 관절 반작용이 섞인다. 기존 contact tensor의 마찰 누락을 모든 센서의 한계로 일반화하지 않는다. 정확한 사람별 payload 힘 보상은 아직 미구현이며 학습 속도는 측정하지 않았다.

힘은 손뿐 아니라 상체(골반/몸통/팔/손)의 접촉 벡터를 합산한 반작용 추정이며, 두 physics step을 평균한다. 상자가 공중에 있고 다른 물체/사람과의 보수적 거리 검사 및 상자 접촉력 합 일치(<5%)를 통과한 구간만 강조한다. **미끄러지는 상자 토이에서 감속에 필요한 수평 힘−63.43N인데 텐서 수평 성분은약0N이었다. 따라서 마찰이 빠진 법선 접촉 성분이며 전체 추진력·제동력·개별 지지 하중으로 해석하지 않는다.** [교정 토이](../output_etc/cparel_carrywith_forces_20261013/calibration.json). 정지 장면에서도 목표 반대 법선 성분이 클 수 있다. NVIDIA 포럼의 [마찰력 누락 관련 보고](https://forums.developer.nvidia.com/t/recovering-frictional-forces-from-gym-get-env-rigid-contacts/264682)와도 구분해, 이 실행 환경에서 직접 확인한 제한이다.

대표 기록 영상/그래프: AT 정면 접근 후 놓기 [영상](../output_etc/cparel_carrywith_forces_20261013/AT_scene0.mp4)·[그래프](../output_etc/cparel_carrywith_forces_20261013/AT_scene0.png), AT 뒤쪽 목표에서 반대로 가다 정지 [영상](../output_etc/cparel_carrywith_forces_20261013/AT_scene2.mp4)·[그래프](../output_etc/cparel_carrywith_forces_20261013/AT_scene2.png), 왼쪽 목표 [영상](../output_etc/cparel_carrywith_forces_20261013/AT_scene8.mp4), ON_TOP 배치 [영상](../output_etc/cparel_carrywith_forces_20261013/ONTOP_scene3.mp4)·[그래프](../output_etc/cparel_carrywith_forces_20261013/ONTOP_scene3.png), ON_TOP 반대 방향 이동/집기 이탈 [영상](../output_etc/cparel_carrywith_forces_20261013/ONTOP_scene46.mp4)·[그래프](../output_etc/cparel_carrywith_forces_20261013/ONTOP_scene46.png). 실제 정책의 물리 궤적을 skeleton으로 다시 그린 영상이다. 재현: `TOKENHSI_GPU=5 bash output_etc/cparel_carrywith_forces_20261013/run.sh`. 학습 모델/RMS·학습 config는 변경하지 않았다.

**단독 원본 ↔ 현재 공동 RSI 비교 영상(2026-10-13):** 왼쪽은 공동 생성에 사용한 단독 B19/B20/B21 원본의32-DOF 자세, 오른쪽은 현재 `carry_joint_full_pickup` 보정본이다. 같은 프레임·30fps로 사선/상단 시점을 비교한다. Pickup100·carryWith129·putDown135프레임 전체를 재생하며, 단독 환경의 크기별 RSI 선별 분포나 정책 rollout을 나타내지는 않는다. 공동 영상 입력은 실제 snapshot의 root/DOF/상자 위치와 일치하는지 확인했다.

| 동작 | 기본 속도 | 0.5배속 |
| --- | --- | --- |
| 전체 모아보기 | [전체](../output_etc/solo_joint_rsi_comparison_20261013/all.mp4) | [전체 느리게](../output_etc/solo_joint_rsi_comparison_20261013/all_half_speed.mp4) |
| Pickup | [비교](../output_etc/solo_joint_rsi_comparison_20261013/pickUp/comparison.mp4) | [느리게](../output_etc/solo_joint_rsi_comparison_20261013/pickUp/comparison_half_speed.mp4) |
| CarryWith | [비교](../output_etc/solo_joint_rsi_comparison_20261013/carryWith/comparison.mp4) | [느리게](../output_etc/solo_joint_rsi_comparison_20261013/carryWith/comparison_half_speed.mp4) |
| PutDown | [비교](../output_etc/solo_joint_rsi_comparison_20261013/putDown/comparison.mp4) | [느리게](../output_etc/solo_joint_rsi_comparison_20261013/putDown/comparison_half_speed.mp4) |

재생성: `python output_etc/solo_joint_rsi_comparison_20261013/render_comparison.py` (`tokenhsi` 환경). [데이터 출처·해시](../output_etc/solo_joint_rsi_comparison_20261013/report.json).

**공동 pickup 전체 프레임 보정:** `output/joint_pickup_full_20261012/pickUp/`에 무릎 제한 내 다리 IK·접지·손 재보정100프레임을 저장했다. 상자52×80×40cm와 궤적은 유지한다. CPU geometry/초기0.1s100/100, GPU6·2048환경 및 부분 reset 통과. 검증본을 `joint_carry/carry_joint_full_pickup/`에 복사해 [새 FullPickup config](#stage2-goal360-fullpickup)에 연결했다. 기존 config의 pickup55개는 유지한다. CarryWith129/putDown135개는 기존 데이터를 재사용한다.

[전체 동작 비교](../output_etc/joint_pickup_full_20261012/comparison.mp4), [느린 비교](../output_etc/joint_pickup_full_20261012/comparison_half_speed.mp4), [100프레임 각각의 물리 reset 비교](../output_etc/joint_pickup_full_20261012/physics_all_frames.mp4), [검증 범위·결과](../output_etc/joint_pickup_full_20261012/validation.json). 기구학 영상은 정책 실행이 아니며, 물리 영상도 매 source frame을 독립 초기화한0.1s 검사다. 연속 native-PD 추종 성공이나 전체 Stage2 정책 성공은 검증하지 않았다.

후보·영상 재생성(학습 실행 아님):

```bash
TOKENHSI_GPU=6 bash -c '
  source tokenhsi/scripts/multi_agent/runtime_env.sh
  python joint_carry/scripts/retarget_joint_pickup.py &&
  python joint_carry/scripts/preview_joint_pickup.py &&
  python joint_carry/scripts/preview_joint_pickup.py --physics
'
```

후보를 다시 만들면 해당 파일 해시로 CPU/GPU 물리 검사를 다시 수행한 뒤 snapshots를 생성해야 한다. 이번 검사 실행기는 결과 폴더의 `check_gpu.py`, `finalize.py`이고, CPU 검사는 기존 `validate_carry_physics.audit`를 사용했다. 실행 데이터의 SHA256은 config·checkpoint·`runtime_manifest.json`으로 검사한다.

TeamHOI 보행 reference 미리보기: `joint_carry/teamhoi_reference/`에 공식 `near_table.yaml`의 후진9개·옆걸음3개와 출처·해시를 보관한다. [후진 영상](../output_etc/teamhoi_reference_preview/backward.mp4), [옆걸음 영상](../output_etc/teamhoi_reference_preview/sideways.mp4)은 원본 skeleton 재생이며 짧은 클립은 마지막 프레임에서 멈춘다. 현재 학습 AMP에는 연결하지 않았다. 재생성은 `tokenhsi` 환경에서 `python joint_carry/scripts/preview_teamhoi_reference.py`로 한다.

원본 호환성 검사: 로더·10-frame carry AMP 입력 생성은 통과했지만 현재 asset과 다리 skeleton이 달라 발 위치 평균5.4–7.6cm 불일치가 있다. GPU5·2048환경의 0.1초 초기 충격 검사는2064/3721프레임 통과(기존 모션238/238). **원본 직접 연결은 보류**, 체형·접지·속도 보정 후 재검증한다. [물리 검사 결과](../output_etc/teamhoi_reference_compatibility_20261009/report.json), [AMP 입력 검사](../output_etc/teamhoi_reference_compatibility_20261009/amp_history.json). 자유 root PD 재생은 기존 모션도 넘어지므로 학습 가능성이나 성능 평가로 해석하지 않는다.

**보정 pilot:** 후진 Male1 B10·옆걸음 CMU141_33 각1개를 `joint_carry/teamhoi_retarget_pilot/{backward,sideways}.npy`로 별도 저장했다. 관절 범위 내 IK·접지·속도 보정 후 GPU5·2048환경의0.1초 초기 충격 검사는253/253·530/530, 실제 carry AMP1320차원 생성과 물리 body/FK 일치 검사를 통과했다. [최종 검사 결과](../output_etc/teamhoi_retarget_pilot_check/report.json), [후진 비교 영상](../output_etc/teamhoi_retarget_pilot_check/backward.mp4), [옆걸음 비교 영상](../output_etc/teamhoi_retarget_pilot_check/sideways.mp4). 영상은 현재 asset 기준 원본/보정 **기구학 재생**이다. 미끄러짐0·정책 보행·AMP 학습 개선까지 보장하지 않으며 학습 AMP/RSI에는 미연결이다. 기존 공동집기 보상도 변경하지 않았다.

**이전 전체 보정본(과도한 무릎 굽힘 확인, 사용 보류):** `joint_carry/teamhoi_retarget_all/`의 후진9개·옆걸음3개, 총3915프레임이 동일 호환성·초기 물리 검사를 통과했다. Male2 B11 급회전 클립은 원래 자세/경로를 유지하며 60fps로 중간 샘플을 추가해 발 위치 보간 오차를 줄였다(나머지30fps). [후진9개 모아보기](../output_etc/teamhoi_retarget_all_check/backward_all.mp4), [옆걸음3개 모아보기](../output_etc/teamhoi_retarget_all_check/sideways_all.mp4), [전체 검사](../output_etc/teamhoi_retarget_all_check/report.json). 클립별 원본/보정 비교 MP4도 같은 폴더에 있다. 모아보기의 노란 화살표는 몸 방향, 분홍은 이동 방향이며 짧은 클립은 마지막 프레임에서 멈춘다. 보정 스크립트·검사·클립별 영상 명령에 `--all`을 붙이면 전체 세트를 재생성한다. 이전 버전은 현재 LocoAMP config에서 사용하지 않는다.

**자세 보존 재보정 pilot:** `joint_carry/teamhoi_retarget_posture_pilot/`에 후진·옆걸음 각1개를 별도로 저장했다. 기존 loco 5개를 대조하고, 골반과 발 목표를 IK 전에 접지한 뒤 원본 무릎 자세 유지 항을 강화했다. 무릎 중앙값은 후진 원본9.5°/이전60.5°/새11.0°, 옆걸음22.3°/56.8°/20.3°다. GPU5·2048환경 초기0.1초 검사는253/253·530/530, AMP640×1320 finite와 발/FK 검사를 통과했다. 자연스러움·미끄러짐0·학습 성능을 보장하는 검사는 아니다. 후진의 바닥 근처 발 속도 proxy는 이전보다 높아 영상 검토가 필요하다. [후진 4열 비교](../output_etc/teamhoi_retarget_posture_pilot_check/backward.mp4), [옆걸음 4열 비교](../output_etc/teamhoi_retarget_posture_pilot_check/sideways.mp4): 원본→이전 보정→새 보정→기존 걷기(반복). 마지막 대조군은 다른 클립으로 보행 위상이 동기화되어 있지 않다. [자세 비교 수치](../output_etc/teamhoi_retarget_posture_pilot_check/gait_comparison.json), [물리 검사](../output_etc/teamhoi_retarget_posture_pilot_check/report.json). 새 pilot은 학습 config에 아직 연결하지 않았다.

**현재 학습용 전체 세트:** `joint_carry/teamhoi_retarget_posture_all/`에 같은 자세 보존 방식으로 후진9개·옆걸음3개를 생성했다. 보간 중 작은 관절 한계 초과를 막기 위해 IK 범위 안쪽에0.001rad 여유를 둔다. Male2 B9/B11/B13/B15는60fps, 나머지는30fps이며 원본3721프레임과 같은 길이/경로를 총4458프레임으로 표현한다. 모든4458프레임이 초기0.1초 검사 및 프레임/중간 시점 관절·접지·FK 검사를 통과했고, 실제 CPU/GPU AMP3200×1320 입력은 finite다. 무릎 중앙값의 원본 대비 변화는 클립별 최대약3.2°다. 이는 연속 보행 안정성·미끄러짐0의 보장이 아니다.

[후진9개 모아보기](../output_etc/teamhoi_retarget_posture_all_check/backward_all.mp4), [옆걸음3개 모아보기](../output_etc/teamhoi_retarget_posture_all_check/sideways_all.mp4), [물리 검사](../output_etc/teamhoi_retarget_posture_all_check/report.json), [자세 비교](../output_etc/teamhoi_retarget_posture_all_check/gait_comparison.json). 클립별4열 비교12개와 모아보기2개, MP4 총14개의 전체 decode를 확인했다. 이 세트가 아래 두 LocoAMP config에 연결되며 기존 Mixed80·Stage1 RSI는 그대로다.

두 클립 pilot만 재현하려면 아래 명령에서 `--all`을 빼고 자세 비교 스크립트도 인자 없이 실행한다.

보정·검사·영상 재현(학습 실행 아님):

```bash
TOKENHSI_GPU=5 bash -c '
  source tokenhsi/scripts/multi_agent/runtime_env.sh
  python joint_carry/scripts/retarget_teamhoi_reference.py --all --posture-first &&
  python joint_carry/scripts/check_teamhoi_retarget.py --all --posture-first &&
  python joint_carry/scripts/preview_teamhoi_retarget.py --all --posture-first &&
  python joint_carry/scripts/preview_teamhoi_retarget.py --all --posture-first --montage &&
  python joint_carry/scripts/compare_teamhoi_gait.py --all
'
```

| 번호 | Config | 역할 |
| ---: | --- | --- |
| SA4 GraspView | [approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_graspview_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_graspview_task_embedding.yaml) | LocoPlatform SA4와 동일 파라미터; 공동에서만 사람별 앵커 중심 관측, 전용 train/test/VNC |
| LocoPlatform human SA4 | [approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding.yaml) | task MLP 없이 human 간 SA4; 세 관계 bias·AMP/RSI/보상 유지 |
| Stage1 파지 중심 이동 토이 | [전용 VNC](../tokenhsi/scripts/multi_agent/stage1_grasp_shift_vnc.sh) | LocoPlatform 공동 환경·Stage1 원본30000 모델, 공유2명·loco·joint_carry_at; 사람별 관측 두 벌로 반대 앵커 비교, 학습 없음 |
| Stage 2 LocoPlatform NONE mask | [approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_nonemask_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_nonemask_task_embedding.yaml) | Actor/Critic CA4에서 학습부터 NONE 차단, 기존 SA feature task·RSI/AMP/보상 유지 |
| 추론 토이 | [locoplatform_ca_none_mask.yaml](../tokenhsi/data/cfg/inference/locoplatform_ca_none_mask.yaml) | 기존 LocoPlatform Actor CA의 NONE만 차단; 학습 없음 |
| 1 | [amp_humanoid_ma_carry.yaml](../tokenhsi/data/cfg/multi_agent/amp_humanoid_ma_carry.yaml) | 원본 multi-agent carry 기준 |
| 27 | [approach_scenario_independent_with_climb.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_independent_with_climb.yaml) | 독립 물체·goal binding, 점 기준 성공 |
| 28 | [approach_scenario_stage1_plane.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_plane.yaml) | 27번의 ON_TOP·CLIMB을 상판 영역 성공으로 변경; Stage 2 기본 출발점 |
| 29 | [approach_stage2_coordination.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_coordination.yaml) | Stage 1 checkpoint에서 협력 graph 학습 |
| Stage 2 plane | [approach_stage2_coordination_sit_plane.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_coordination_sit_plane.yaml) | 34번 기반 SIT plane·보상·독립 과제 분포 |
| Stage 2 Joint carry | [approach_stage2_joint_carry_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_task_embedding.yaml) | 동결 task embedding + task CA·COUPLED bias, 2인 공유 상자 AT/ON_TOP |
| Stage 2 Mixed80 | [approach_stage2_joint_carry_mixed80_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_task_embedding.yaml) | 공동 80%·단독 20%, 단독 Stage 1 크기·RSI와 SELF/NONE CA |
| Mixed80 LocoAMP CA | [approach_stage2_joint_carry_mixed80_locoamp_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_task_embedding.yaml) | 보행 AMP 보강, 기존 task CA·128입력 head |
| Mixed80 LocoAMP CA4 | [approach_stage2_joint_carry_mixed80_locoamp_ca4_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_task_embedding.yaml) | 같은 보상·AMP·RSI, CA+FFN+LayerNorm+residual 4층·128입력 head |
| Mixed80 LocoAMP CA4 Goal360 FullPickup CPA Release1 | [approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding.yaml) | FullPickup 기반 공동 상대 CPA만 상자 반대각선+1m 안에서 해제; 두 사람 모두 근접 필요 |
| CPA Release1 혼합 장면 데모 | CPA Release1/LocoPlatform config 자동 선택·[전용 진입점](../tokenhsi/joint_carry_demo.py) | 평가 전용4명/6명·8물체·1환경, 공동 AT 한 팀/두 팀 carryWith + 단독 AT/ON_TOP loco; 학습 스크립트 없음 |
| LocoPlatform 공동→단독 task 차단 | [전용 VNC](../tokenhsi/scripts/multi_agent/locoplatform_ca_solo_mask_vnc.sh) | 추론 전용 Platform22000·6명; 기존 모델로 baseline/공동 query의 단독 task 차단 비교, 학습 스크립트 없음 |
| CPA Release1 혼합 장면 CA Entmax | 위 CPA Release1 config 재사용·[전용 VNC](../tokenhsi/scripts/multi_agent/joint_carry_mixed_scene_demo_ca_entmax_vnc.sh) | 같은6명 장면, Actor CA4만 entmax1.5로 추론; 기본19000·학습 스크립트 없음 |
| Entmax1.5 추론 토이 | 위 CPA Release1 config 재사용·[격리된 실행](../output_etc/entmax_inference_toy_20261014/rollout.py) | GPU4·2명/6명 각각16장면, Actor SA/CA 정규화만 변경; 학습/기존 viewer 미연결 |
| LocoPlatform Geometry task CA | [approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_geometry_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_geometry_task_embedding.yaml) | LocoPlatform 기반85차원 pose task 입력; Stage1 전이 후 Stage2 새 학습 |
| CPA Release1 LocoPlatform | [approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding.yaml) | 공동 loco의50%에서 payload만 받침 사용; 크기 기반 높이 제한, CPA Release1 기반 |
| CPA Release1 PickupFocus | [approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding.yaml) | CPA Release1 기반 공동 pickup 프레임 집중35/35/30; 전체100프레임 유지 |
| Mixed80 LocoAMP CA4 Goal360 FullPickup | [approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_task_embedding.yaml) | Actor/Critic CA4→64입력 head·critic 전체 학습, 공동 pickup100프레임; 기본 Goal360 보상 |
| Mixed80 LocoAMP CA4 Goal360 TeamHold | [approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_teamhold_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_teamhold_task_embedding.yaml) | Goal360 기반, 공동 HOLDING state만 공유 bottleneck 점수로 교체; Move 없음 |
| Mixed80 LocoAMP CA4 Goal360 | [approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding.yaml) | CA4 기반 공동 목표360°·2.2~2.8m, AT putDown reference 끝점; Move/Align 없음 |
| Mixed80 LocoAMP CA4 Move | [approach_stage2_joint_carry_mixed80_locoamp_ca4_move_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_move_task_embedding.yaml) | CA4 기반, 두 사람 실제 이동 방향 min 팀 보상0.2·목표 근처 pinning |
| Mixed80 LocoAMP Align CA | [approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding.yaml) | LocoAMP CA 기반, 공동 HOLDING 중 가장 먼 사람의 방향 보상 공유 |
| Mixed80 LocoAMP head-only | [approach_stage2_joint_carry_mixed80_locoamp_head_only.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_head_only.yaml) | 동일 AMP/reward/RSI, CA 없음·Stage1 64입력 head |
| Stage 2 BEFORE | [approach_stage2_before_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_stage2_before_task_embedding.yaml) | BEFORE 80%·독립 20%, carry→climb/sit/stack·현재 AT gate·공통 크기 |
| 30 | [approach_scenario_stage1_sit_plane_normalized.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_sit_plane_normalized.yaml) | 28번 + SIT 상판 성공·유효 edge 평균 보상 |
| 31 | [approach_scenario_stage1_skill_curriculum.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_skill_curriculum.yaml) | 30번 + 가까운 시작·후반 CLIMB RSI·어려운 과제 증량·보상 shaping |
| 32 | [approach_scenario_stage1_skill_curriculum_reward_preserved.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved.yaml) | 31번의 시작 상태·RSI·샘플링을 유지하고 state/progress는 30번 식 |
| 33 | [approach_scenario_stage1_at_goal_fix.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_at_goal_fix.yaml) | 32번 + AT 목표 slot reset 오류 수정 |
| 34 | [approach_scenario_stage1_self_sum.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_self_sum.yaml) | 33번 + 자기 edge 보상 합계·동료 edge 평균, 과제 비율 10/10/10/35/35 |
| 35 | [approach_scenario_stage1_slow_near_start.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_slow_near_start.yaml) | 34번 + 가까운 시작 80%를 60만 step 유지하고 120만 step까지 30%로 감소 |
| 36 | [approach_scenario_stage1_placement_focus.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_placement_focus.yaml) | 34번 + HOLDING+AT/HOLDING+ON_TOP만 각 50% 샘플링 |
| 37 | [approach_scenario_stage1_ontop_putdown.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_ontop_putdown.yaml) | 36번 + ON_TOP에도 충돌 없는 중간 구간의 `putDown` RSI 10% 적용 |
| Paired AT | [approach_scenario_stage1_paired_at_no_near.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_paired_at_no_near.yaml) | 2-agent/3-object, 매 scene AT+AT, 가까운 시작 없음 |
| Paired placement | [approach_scenario_stage1_paired_placement_no_near.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_paired_placement_no_near.yaml) | 2-agent/4-object, scene별 AT+AT 또는 ON_TOP+ON_TOP 각 50%, 가까운 시작 없음 |
| Paired owner HOLDING | [approach_scenario_stage1_paired_placement_owner_holding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_paired_placement_owner_holding.yaml) | Paired placement + AT/ON_TOP edge에 담당자의 선행 HOLDING `φ` 입력 |
| Unified semantic | [approach_scenario_stage1_unified.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified.yaml) | 2H/4O 독립 5과제, canonical binding·토큰 순열·과제 조건 AMP·unified RSI |
| Unified shared edge | [approach_scenario_stage1_unified_shared_edge_encoder.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder.yaml) | Unified semantic의 actor·critic semantic EdgeEncoder 파라미터 공유 |
| Unified owner HOLDING | [approach_scenario_stage1_unified_owner_holding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_owner_holding.yaml) | Unified semantic + AT/ON_TOP 담당 HOLDING φ 입력 |
| Unified size RSI | [approach_scenario_stage1_unified_size_rsi.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi.yaml) | 행동별 실제 asset 크기·bbox progress·물리 검사한 후반 RSI, 과제 비율 5/10/25/30/30 |
| Unified size RSI typed bias | [approach_scenario_stage1_unified_size_rsi_typed_bias.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias.yaml) | Size RSI 유지, 타입·관계·타입별 독립 layer/head bias 표 |
| Unified size RSI typed message | [approach_scenario_stage1_unified_size_rsi_typed_bias_message.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message.yaml) | Typed bias 유지, 64-D 관계 메시지 추가 |
| Unified size RSI task message | [approach_scenario_stage1_unified_size_rsi_task_message.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message.yaml) | 복합 carry 과제·역할별 bias/message, 단방향 3연결·자기 보상만 |
| Unified size RSI task MLP | [approach_scenario_stage1_unified_size_rsi_task_mlp.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp.yaml) | Task message와 동일한 과제·RSI·AMP·자기 보상, task·역할 MLP bias만 사용 |
| Unified size RSI task MLP split | [approach_scenario_stage1_unified_size_rsi_task_mlp_split.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split.yaml) | Task MLP의 역할 embedding 공유, task별 MLP·projection 독립 |
| Unified size RSI task embedding | [approach_scenario_stage1_unified_size_rsi_task_embedding.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding.yaml) | Task/NONE/SELF embedding과 H/O/G 타입 쌍별 projection, 중간 MLP·메시지 없음 |
| Carry task embedding distill | [approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill.yaml) | Carry AT/ON_TOP 각 50%, 타입 쌍 projection 유지, 동결 unified teacher KL |
| Four-task embedding distill | [approach_scenario_stage1_unified_size_rsi_task_embedding_distill.yaml](../tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill.yaml) | Sit/climb/carry_at/carry_ontop 원래 분포 유지, 동결 unified teacher KL |

27~37번 Stage 1은 2-agent/3-object 독립 graph와 schema 9 semantic 관찰을 쓴다. ON_TOP 두 개를 동시에 샘플하지 않는다. 28~37번의 팀 보상은 자기 0.9, 동료 0.1이다. 30~33번은 유효 edge를 agent별로 평균 내어 단일 edge와 두 edge의 최대 task 보상을 0.6으로 맞춘다. 34~37번은 자기 edge를 합산하고 동료 edge만 평균 내며 state·progress·성공 식은 33번과 같다.

Paired AT/paired placement도 schema 9 semantic 관찰과 34번의 reward·RSI·AMP를 유지한다. 두 agent의 **과제 relation만 같은 타입**으로 샘플하고 실제 motor action은 각자 선택한다. Paired placement는 두 ON_TOP에 서로 다른 받침을 주기 위해 4개 물체를 쓴다. 두 기준 실험은 가까운 시작 확률이 처음부터 끝까지 0이며, 각자 별도 checkpoint/output에서 scratch로 시작한다.

Paired owner HOLDING은 paired placement의 샘플링·reward·RSI·AMP·가까운 시작 없음 설정을 유지한다. 관측만 edge당 5개에서 6개 필드로 늘리고 AT/ON_TOP에는 같은 owner의 `H→source O` HOLDING `φ`, 나머지 유효 edge에는 1을 기록한다. actor·critic의 기존 semantic 64차원 뒤에서 AT/ON_TOP에만 상태 잔차를 더한다. 기존 paired checkpoint와 관측·네트워크 형식이 달라 별도 output에서 scratch로 학습한다.

2026-09-30 공통 코드 수정 이후 scenario의 상자 속도 패널티는 현재 graph에 연결된 담당 물체를 따른다. 이전에는 legacy agent assignment를 사용해 랜덤 binding에서 다른 물체의 패널티가 들어갈 수 있었다. 모델/관측 shape는 같아서 기존 checkpoint를 로드할 수 있지만, 수정 전후 학습의 penalty 의미는 다르다. 이미 실행 중인 프로세스에는 자동 적용되지 않으며 이후 새 프로세스부터 적용된다.

33번은 32번의 보상·샘플링·RSI를 유지한다. AT graph가 두 번째 목표 slot을 가리킬 때 reset도 해당 목표를 배치하도록 공통 코드의 slot 선택을 수정했다. 수정 후 새로 시작하는 27~32번 실행에도 적용되지만, 이미 실행 중인 프로세스와 checkpoint는 바뀌지 않는다. 비교 실험은 33번 전용 output에 scratch로 시작한다.

## 학습

아래 `GPU`를 사용할 장치 번호로 바꾼다. 각 Stage 1 실험은 별도 output에 scratch로 시작한다.

```bash
GPU=5
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_independent_with_climb_train.sh 2 2048 3  # 27
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_plane_train.sh 2 2048 3  # 28
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_sit_plane_normalized_train.sh 2 2048 3  # 30
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_train.sh 2 2048 3  # 31
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_train.sh 2 2048 3  # 32
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_at_goal_fix_train.sh 2 2048 3  # 33
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_train.sh 2 2048 3  # 34
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_slow_near_start_train.sh 2 2048 3  # 35
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_train.sh 2 2048 3  # 36
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_train.sh 2 2048 3  # 37
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_train.sh 2 2048 3  # Paired AT
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_train.sh 2 2048 4  # Paired placement
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_train.sh 2 2048 4  # Paired owner HOLDING
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_train.sh 2 2048 4  # Unified semantic
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_train.sh 2 2048 4  # Unified shared edge
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_train.sh 2 2048 4  # Unified owner HOLDING
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_train.sh 2 2048 4  # Unified size RSI
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_train.sh 2 2048 4  # Unified size RSI typed bias
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_train.sh 2 2048 4  # Unified size RSI typed message
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_train.sh 2 2048 4  # Unified size RSI task message
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_train.sh 2 2048 4  # Unified size RSI task MLP
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_train.sh 2 2048 4  # Unified size RSI task MLP split
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_train.sh 2 2048 4  # Unified size RSI task embedding
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_train.sh 2 2048 4  # Carry task embedding distill
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_train.sh 2 2048 4  # Four-task embedding distill
```

한 번에 필요한 실험 **한 줄만** 실행한다. 짧은 확인은 별도 output을 쓴다.

```bash
MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_scenario_stage1_skill_curriculum_reward_preserved_check \
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_train.sh 2 2048 3
```

정기 checkpoint 저장 간격은 학습 설정에서 500 epoch다. `.pth`는 각 run의 `nn/`, TensorBoard는 `summaries/`, edge 진단은 `diagnostics/`에 저장된다.

## 로컬 평가와 서버 VNC

학습과 **같은 번호의 checkpoint와 스크립트**를 사용한다. `CKPT`를 실제 `.pth` 절대 경로로 설정한다.

```bash
GPU=5
CKPT='/absolute/path/to/checkpoint.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_independent_with_climb_test.sh "$CKPT" 2 1 3 10  # 27
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_plane_test.sh "$CKPT" 2 1 3 10  # 28
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=sit bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_sit_plane_normalized_test.sh "$CKPT" 2 1 3 10  # 30
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_test.sh "$CKPT" 2 1 3 10  # 31
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_test.sh "$CKPT" 2 1 3 10  # 32
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_at_goal_fix_test.sh "$CKPT" 2 1 3 10  # 33
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_test.sh "$CKPT" 2 1 3 10  # 34
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_slow_near_start_test.sh "$CKPT" 2 1 3 10  # 35
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_test.sh "$CKPT" 2 1 3 10  # 36
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_test.sh "$CKPT" 2 1 3 10  # 37
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_test.sh "$CKPT" 2 1 3 10  # Paired AT
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_test.sh "$CKPT" 2 1 4 10  # Paired placement
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_test.sh "$CKPT" 2 1 4 10  # Paired owner HOLDING
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_test.sh "$CKPT" 2 1 4 10  # Unified semantic
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_test.sh "$CKPT" 2 1 4 10  # Unified shared edge
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_test.sh "$CKPT" 2 1 4 10  # Unified owner HOLDING
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_test.sh "$CKPT" 2 1 4 10  # Unified size RSI
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_test.sh "$CKPT" 2 1 4 10  # Unified size RSI typed bias
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_test.sh "$CKPT" 2 1 4 10  # Unified size RSI typed message
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_test.sh "$CKPT" 2 1 4 10  # Unified size RSI task message
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_test.sh "$CKPT" 2 1 4 10  # Unified size RSI task MLP
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_test.sh "$CKPT" 2 1 4 10  # Unified size RSI task MLP split
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_test.sh "$CKPT" 2 1 4 10  # Unified size RSI task embedding
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_test.sh "$CKPT" 2 1 4 10  # Carry task embedding distill
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_test.sh "$CKPT" 2 1 4 10  # Four-task embedding distill
TOKENHSI_GPU=3 HEADLESS=0 bash tokenhsi/scripts/multi_agent/box_cleanup_demo_test.sh  # 4명·16상자 정리 데모
```

화면 없는 학습 분포 평가는 해당 test 명령에서 `HEADLESS=1 TASK_GRAPH=random_scenario`와 평가 환경 수 `64`를 지정한다. 로컬 test와 서버 VNC는 모두 기본적으로 `output/<실험명>/metrics/`와 `diagnostics/`에 평가 결과를 쓴다. 반복 평가 시 진단 CSV는 같은 파일에 이어 쓰므로 실행별 분석은 `OUTPUT_PATH`를 따로 지정한다. 학습 checkpoint는 별도 run 디렉터리의 `nn/`에 저장된다. 아래는 서버 viewer 실행 명령이다.

```bash
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_independent_with_climb_vnc.sh "$CKPT"  # 27
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_plane_vnc.sh "$CKPT"  # 28
TOKENHSI_GPU="$GPU" TASK_GRAPH=sit bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_sit_plane_normalized_vnc.sh "$CKPT"  # 30
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_vnc.sh "$CKPT"  # 31
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_skill_curriculum_reward_preserved_vnc.sh "$CKPT"  # 32
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_at_goal_fix_vnc.sh "$CKPT"  # 33
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_self_sum_vnc.sh "$CKPT"  # 34
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_slow_near_start_vnc.sh "$CKPT"  # 35
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_vnc.sh "$CKPT"  # 36
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_vnc.sh "$CKPT"  # 37
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_vnc.sh "$CKPT"  # Paired AT
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_vnc.sh "$CKPT"  # Paired placement
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_vnc.sh "$CKPT"  # Paired owner HOLDING
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_vnc.sh "$CKPT"  # Unified semantic
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_vnc.sh "$CKPT"  # Unified shared edge
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_vnc.sh "$CKPT"  # Unified owner HOLDING
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_vnc.sh "$CKPT"  # Unified size RSI
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_vnc.sh "$CKPT"  # Unified size RSI typed bias
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_vnc.sh "$CKPT"  # Unified size RSI typed message
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_vnc.sh "$CKPT"  # Unified size RSI task message
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_vnc.sh "$CKPT"  # Unified size RSI task MLP
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_vnc.sh "$CKPT"  # Unified size RSI task MLP split
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_vnc.sh "$CKPT"  # Unified size RSI task embedding
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_vnc.sh "$CKPT"  # Carry task embedding distill
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_vnc.sh "$CKPT"  # Four-task embedding distill
TOKENHSI_GPU=3 PORT=6080 bash tokenhsi/scripts/multi_agent/box_cleanup_demo_vnc.sh  # 4명·16상자 정리 데모
```

## 상자 정리 VNC 데모

별도 학습/config 없이 34번 YAML을 바탕으로 `box_cleanup_demo.py`에서 평가 전용 task/player를 등록한다. 기본 checkpoint는 `distill_pth/ApproachScenarioStage1RescueAtKLClimb50_00008000.pth`다. 사람 4명·물리 상자 16개·8개 semantic edge로 정책을 그대로 로드하며, H223/O30 정규화 통계도 재사용한다. 데모 로더만 옛 rescue variant와 추가 AT/CLIMB shaping 설정을 현재 self-sum runtime에 대응시킨다. schema·packet·나머지 reward 계약은 검사하고, 원본 checkpoint와 일반 학습/평가 로더는 변경하지 않는다. 데모의 결과는 운반 진행 기록이며 rescue 보상 비교 평가는 아니다.

- 중앙에 상자 16개를 불규칙한 2단 더미로 모아 쌓는다. 가로·세로 0.40/0.45/0.50m, 높이 0.30/0.35/0.40m를 독립적으로 섞으며 밀도는 100kg/m³다. 기본 중심 간격 70cm에서 두 줄의 가로 위치·각 상자의 위치를 흐트러뜨리고 수평 회전을 ±15° 섞는다. 받침 위 대상도 조금 어긋나게 놓으며 시작 시 물체끼리 겹치지 않는다. 크기는 실행 시작 시 고정되고 위치·회전은 매 반복 다시 샘플한다. 배정 여부와 관계없이 같은 회색이다.
- 동서남북 사람 4명이 자기 방향의 윗단 상자를 하나씩 옮기고, 모두 끝나면 남은 윗단 상자 네 개를 배정한다. 운반 대상 8개 모두 처음에는 받침 상자 위에 있으며 실제 높이에 맞춰 초기 중심·바닥 목적지 높이를 계산한다. 해당 checkpoint의 저장된 학습 설정은 가로·세로 0.40~0.65m, 높이 0.25~0.55m다. 데모 크기는 이 범위 안이며 크기별 성공률을 보장하지 않는다.
- 목적지 XY 오차 12cm 미만·회전 bbox 바닥 오차 2.5cm 미만·선속도 0.15m/s 미만·각속도 0.5rad/s 미만을 15 control step(0.5초) 유지하면 내려놓기로 기록한다. 손 접촉 해제는 요구하지 않는다. 먼저 끝난 사람은 기존 배정을 유지하며 정책이 계속 자세를 제어한다.
- 네 상자가 동시에 안정적으로 놓이면 graph와 목적지만 바꾼다. 사람·상자를 teleport/reset하지 않고 두 번째 운반을 시작한다. 두 라운드 후 종료하며 viewer는 마지막 화면을 유지한다. 중앙 상자도 실제 물리 물체여서 접촉하면 움직이거나 더미가 무너질 수 있다.
- 이전 동일 크기·정렬된 중앙 더미 버전의 seed 42 검사에서는 7개 운반 후 마지막 윗단 집기가 정체되어 시간 초과했다. 현재 불규칙·크기 혼합 더미의 8개 완주는 아직 확인하지 못했다. 사용자 선택에 따라 4명·16물체 전체 관측을 그대로 입력하며 주변 관측 추론은 적용하지 않는다.

```bash
# 서버 VNC: 기본 10회·한 회 최대 1500 step(50초), 6080 포워딩 후 noVNC 접속
TOKENHSI_GPU=3 PORT=6080 bash tokenhsi/scripts/multi_agent/box_cleanup_demo_vnc.sh
# 로컬 viewer / 화면 없는 검사
TOKENHSI_GPU=3 HEADLESS=0 bash tokenhsi/scripts/multi_agent/box_cleanup_demo_test.sh
TOKENHSI_GPU=3 HEADLESS=1 REPEATS=1 OUTPUT_PATH=output/box_cleanup_demo_check bash tokenhsi/scripts/multi_agent/box_cleanup_demo_test.sh
```

checkpoint를 바꾸려면 첫 인자로 경로를 넘긴다. `REPEATS` 기본값은 10이며 각 반복마다 두 라운드 장면을 처음부터 재생한다. `KEEP_OPEN=0`은 viewer가 마지막 반복 종료 시 자동 닫히게 한다. `EPISODE_LENGTH` 기본값은 1500 control step(시뮬레이션 시간 50초), `SEED`는 42다. 실패·시간 초과도 `OUTPUT_PATH/cleanup_results.json`에 실제 상자 크기와 함께 기록하고 다음 라운드를 강제 성공시키지 않는다. Viewer는 `frame_0/120/300/600.png`와 `final.png`를 저장한다.

## Unified 독립 5과제 · semantic / shared edge / owner-HOLDING 비교

Semantic과 owner-HOLDING은 **edge 입력 상태 유무만 다르다**. Shared edge는 semantic과 과제·관측·보상이 같고 actor·critic의 semantic EdgeEncoder 임베딩·MLP·bias projection만 공유한다. 토크나이저·GTA/Transformer·head는 분리한다. 기존 paired/focus 실험은 보존하며 각자 새 output에서 scratch로 시작한다. 새 AMP 입력은 프레임당 기존 129차원에 과제 one-hot 3차원을 더한 132차원, 10프레임 총 1320차원이다. 기존 paired checkpoint와는 AMP 및 variant 계약이 달라 이어 학습하지 않는다.

- 학습: 2명·4물체. 각 agent가 독립적으로 `HOLDING/SIT/CLIMB/HOLDING_AT/HOLDING_ON_TOP = 5/5/20/35/35%`를 샘플한다. 두 ON_TOP도 가능하며 명목 빈도는 12.25%다.
- 논리 할당(0-based): H₀→O₀→G₀, H₁→O₁→G₁. SIT/CLIMB도 각자의 Oᵢ를 사용한다. ON_TOP은 O₀→O₂, O₁→O₃. 물리 asset slot만 reset마다 섞이며 모든 관측·RSI·보상은 동일 logical-to-physical mapping을 따른다.
- H/O/G별 인코딩 뒤 transformer 입력을 모든 타입에 걸쳐 매 forward 무작위 순열로 섞는다. 한 forward의 batch는 같은 순열을 사용하며, GTA와 bias의 두 축도 함께 이동한다. 출력은 canonical 순서로 복원한 다음 human별 actor/value head에 전달한다. 순열 등가성에 따른 수치 동등성은 검증하되 추가 학습 이득을 보장하지 않는다.
- Goal GTA quaternion은 `(0,0,0,1)`로 고정한다. 상자 크기는 X=0.40m, Y=0.30m, Z=0.30~0.50m(0.05m 간격)다.
- loco RSI의 담당 상자는 **owner root XY에서 반경 1~2m**에 배치한다. AT 목표/ON_TOP 받침에 가까운 시작은 0%이며 별개 설정이다. pickUp/carryWith/putDown/sit/climb RSI의 참조 물체 위치는 유지한다. 물리 배치 부적합 시 graph를 유지한 채 reset을 다시 뽑는다.
- state/progress/success 0.2/0.2/0.2, 자기 edge 합계·동료 edge 평균의 0.9/0.1 공유, placement 성공 시 matching HOLDING 포화는 유지한다. TERM 입력은 추가하지 않는다.

| 과제 | RSI |
| --- | --- |
| HOLDING | loco 50%, pickUp 50% |
| SIT | loco 50%, sit 50% |
| CLIMB | loco 50%, climb 50%; 원래 허용 시점 사용, 후반 강제 샘플링 없음 |
| HOLDING+AT | loco 50%, pickUp 10%, carryWith 30%, putDown 10% |
| HOLDING+ON_TOP | loco 50%, pickUp 10%, carryWith 40%; putDown 없음 |

AMP는 단일 판별기에 **각 agent의 할당 과제 family**를 매 프레임 붙인다. HOLDING/AT/ON_TOP은 carry, SIT는 sit, CLIMB은 climb이다. RSI로 걷기를 뽑았어도 family는 할당 과제를 따른다. 전문가의 과제 내부 분포는 원본 unified에 맞춘다: carry=loco/OMOMO/pickUp/putDown 1/3·1/3·1/6·1/6, sit=loco/sit 1/3·2/3, climb=loco/climb/climbNoRSI 각 1/3. `carryWith`는 RSI에는 있지만 이 AMP 전문가 혼합에는 없다.

`skillDiscProb`는 리셋 시 family 비율 75/5/20으로 환산한 명목 전체 분포다. 실제 조건부 expert 생성은 `utils/unified_training.py`의 분포를 사용한다. 학습에서는 전문가·리플레이를 현재 rollout의 family와 **행별로 맞춰** 뽑아 PPO minibatch마다 family 빈도가 일치한다. 리플레이에 해당 family가 아직 없을 때만 같은 family의 현재 rollout을 사용한다. 과제 라벨은 AMP 정규화에서 통과시키며 참조 이력·부분 reset에도 유지한다. task/AMP reward 비중은 기존 0.5/0.5다.

Semantic 실험은 edge당 `[valid,src,dst,relation,owner]` 5필드이고 semantic type/relation만 bias에 인코딩한다. Owner 실험은 6번째 값에 동일 owner·source의 HOLDING φ를 넣고 AT/ON_TOP의 semantic64 뒤에서 상태 잔차를 더한다. 다른 유효 edge의 입력 값은 1이며, 상태 잔차는 placement edge에만 적용한다.

아래 학습 명령에서 필요한 실험을 선택한다. `TOKENHSI_GPU=0`은 로컬 단일 GPU 기준이다.

```bash
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_train.sh 2 2048 4
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_train.sh 2 2048 4
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_train.sh 2 2048 4
```

각 checkpoint에는 같은 실험의 test/VNC를 사용한다. 로컬 viewer는 다음과 같다.

```bash
CKPT='/absolute/path/to/checkpoint.pth'
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_test.sh "$CKPT" 2 1 4 10
```

`TASK_GRAPH=holding|sit|climb|holding_at|holding_ontop`은 두 사람 모두 해당 과제로 평가한다. 기본 평가는 loco 시작이다. placement의 들고 있는 시작을 보려면 `TASK_GRAPH=holding_at` 또는 `holding_ontop`에 `EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0`을 추가한다. `HEADLESS=1`은 화면 없는 평가다.

```bash
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_vnc.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_shared_edge_encoder_vnc.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_owner_holding_vnc.sh "$CKPT" 2 1 4 10
```

각 기본 output은 `output/approach_scenario_stage1_unified`, `output/approach_scenario_stage1_unified_shared_edge_encoder`, `output/approach_scenario_stage1_unified_owner_holding`이다. Shared edge는 별도 variant/checkpoint로 scratch 학습하며 기존 unified checkpoint를 resume하지 않는다. 짧은 검증은 별도 `_check` output과 `MAX_ITERATIONS=2`를 사용한다. 2명·4물체 sampler이며 가변 인원 평가 sampler 확장은 포함하지 않는다.

## Stage 1 AT/ON_TOP 집중 실험 · 36번

34번과 동일한 reward·RSI·가까운 시작 설정에서 `HOLDING/SIT/CLIMB/HOLDING_AT/HOLDING_ON_TOP = 0/0/0/0.5/0.5`로 샘플한다. 한 scene에 AT와 ON_TOP이 각각 하나씩 배정되고, 두 과제 모두 자기 HOLDING edge를 포함한다. 단독 HOLDING 및 SIT/CLIMB 과제는 학습 중 나오지 않는다. `templateRsi`의 AT/ON_TOP 행은 그대로 적용되고, CLIMB 후반 RSI는 CLIMB 과제가 없어 사용되지 않는다. `skillDiscProb`의 SIT/CLIMB 모션 비율은 34번과 동일하다. 새 variant·output에서 scratch로 시작한다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachScenarioStage1PlacementFocus.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_placement_focus_vnc.sh "$CKPT"
```

## Stage 1 같은 과제 쌍 · 가까운 시작 없음

**Paired AT**은 3개 물체에서 두 agent에게 각각 다른 물체·목표를 배정한다. 각 agent의 edge는 `H→O HOLDING`, `O→G AT`이며 한 scene에 AT edge가 두 개다. **Paired placement**는 4개 물체에서 scene마다 `AT+AT` 또는 `ON_TOP+ON_TOP`을 50%씩 뽑는다. ON_TOP scene의 두 source와 두 support는 모두 다른 물체다. 두 실험 모두 34번의 RSI·AMP·보상식을 유지하고 `near_start` 확률만 0으로 고정한다. `TASK_GRAPH=holding_at`/`holding_ontop`으로 paired placement 평가 장면을 각각 고를 수 있다. 다른 물체 수와 variant는 서로 checkpoint 호환되지 않는다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachScenarioStage1PairedAtNoNear.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_at_no_near_vnc.sh "$CKPT"

TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_train.sh 2 2048 4
CKPT='/absolute/path/to/ApproachScenarioStage1PairedPlacementNoNear.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_no_near_vnc.sh "$CKPT"
```

## Stage 1 Paired placement · 담당자 HOLDING 상태 입력

위 Paired placement와 동일한 2-agent/4-object·AT+AT/ON_TOP+ON_TOP·가까운 시작 없음·RSI·AMP·reward를 사용한다. AT/ON_TOP edge의 여섯 번째 관측 필드는 동일 owner와 source 물체의 직전 HOLDING edge `φ`다. HOLDING 등 다른 유효 edge에는 1을 넣고, 무효 edge는 0으로 마스킹한다. `φ`는 손–상자 중심 거리 기반의 0~1 상태이며 실제 물리적 파지 플래그는 아니다. 기존 paired checkpoint는 새 관측 폭과 맞지 않는다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_train.sh 2 2048 4
CKPT='/absolute/path/to/ApproachScenarioStage1PairedPlacementOwnerHolding.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_paired_placement_owner_holding_vnc.sh "$CKPT"
```

## Stage 1 ON_TOP putDown RSI 실험 · 37번

36번과 동일한 과제·보상 분포에서 `HOLDING_ON_TOP` RSI 중 `carryWith` 10%를 `putDown`으로 옮긴다. ON_TOP `putDown`은 원본 모션의 45~70% 구간에서만 시작하고, 받침 상자를 모션의 최종 놓기 XY에 배치한다. 이 구간은 원본 모션의 후기 상자·받침 겹침을 피한다. 받침의 물리 크기는 기존 분포 그대로이며, reset 시 상자 또는 사람과 겹치면 재추첨한다. AT `putDown` RSI 10%와 나머지 설정은 36번과 같다. 별도 variant·output에서 scratch로 시작한다.

```bash
TOKENHSI_GPU="$GPU" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachScenarioStage1OntopPutdown.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_ontop_putdown_vnc.sh "$CKPT"
```

## Stage 2 협력 실험 · 29번

공동운반 RSI의 별도 GPU 물리 검사는 `TOKENHSI_GPU=5 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; python output/joint_carry_gpu5_check_20261007/check_gpu.py'`로 재현한다. 2인/공유 상자1개·2048환경·52×80×40cm의 0.1초 검사이며, 같은 폴더의 `report.json`·`per_env.npz`를 갱신한다. 입력 보정본은 유지한다. Stage 2 학습 reset 연결 및 AMP expert 품질·장기 운반 성공 검증은 포함하지 않는다.

29번은 지정한 Stage 1 `.pth`에서 actor/critic/AMP weight와 관찰 통계를 옮겨 새 실행을 시작한다. `RESUME_CHECKPOINT`는 이미 시작한 **같은 Stage 2 변형**을 이어 학습할 때만 쓴다. 기존 29번의 SIT 성공과 보상 정규화는 **28번 정의**다.

```bash
STAGE1_CHECKPOINT='/absolute/path/to/Stage1.pth' TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_train.sh 2 2048 3
RESUME_CHECKPOINT='/absolute/path/to/ApproachStage2Coordination.pth' TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachStage2Coordination.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=place_stack \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=place_climb \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_vnc.sh "$CKPT"
```

Stage 2 plane은 34번의 SIT plane 성공·자기 edge 합계·동료 edge 평균을 적용한다. 협력 과제 비율은 `place_climb/place_sit/place_stack = 20/20/40%`, 독립 과제는 20%다. 독립 과제 내부의 HOLDING/SIT/CLIMB/HOLDING+AT/HOLDING+ON_TOP은 34번과 같이 10/10/10/35/35이며 RSI·가까운 시작 일정도 34번과 같다. 협력 과제는 공유 물체에 맞춘 Stage 2 RSI를 유지한다. 별도 config·스크립트·output을 사용한다.

```bash
STAGE1_CHECKPOINT='/absolute/path/to/ApproachScenarioStage1SelfSum.pth' TOKENHSI_GPU="$GPU" \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_train.sh 2 2048 3
CKPT='/absolute/path/to/ApproachStage2CoordinationSitPlane.pth'
TOKENHSI_GPU="$GPU" HEADLESS=0 TASK_GRAPH=place_stack \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_test.sh "$CKPT" 2 1 3 10
TOKENHSI_GPU="$GPU" TASK_GRAPH=place_stack \
  bash tokenhsi/scripts/multi_agent/approach_stage2_coordination_sit_plane_vnc.sh "$CKPT"
```

3·4인 평가 graph는 `tokenhsi/data/cfg/multi_agent/graphs/`에 있다. 입력 크기 검증용이며 성공은 별도 학습으로 확인한다.

## 원본 1번

원본 비교가 필요할 때 `ma_carry_train.sh`, `ma_carry_test.sh`, `ma_carry_watch.sh`를 사용한다. 실험별 GPU 지정과 데이터 경로는 위 공통 규칙을 따른다.

지표 정의는 [relation_diagnostics.md](../tokenhsi/docs/relation_diagnostics.md)를 참고한다. TensorBoard는 `tensorboard --logdir output --host 127.0.0.1 --port 6006`으로 실행한다.

## Unified size RSI

`approach_scenario_stage1_unified_size_rsi`는 unified semantic에서 분리한 scratch 실험이다. 기존 unified checkpoint는 reward 계약이 달라 재개할 수 없다. 2-agent/4-object canonical 배정, semantic edge 관찰, 토큰·edge 순열, state·성공 조건·보상 가중치, 가까운 시작 없음은 유지한다.

| 과제/물체 | X·Y 각각 (m) | Z (m) |
| --- | --- | --- |
| HOLDING, HOLDING_AT, HOLDING_ON_TOP source | 0.20–0.60 | 0.20–0.60 |
| SIT source | 0.25–0.80 | 0.35–0.50 |
| CLIMB source | 0.40–0.80 | 0.25–0.50 |
| ON_TOP support | 0.50–0.80 | 0.25–0.45 |

- X/Y/Z는 각각 독립적인 5cm 격자 균등분포이고 밀도는 100kg/m³다. 큰 carry 상자는 최대 21.6kg이며, 범위 안의 모든 크기에서 행동 성공이 보장되는 것은 아니다.
- 물리 asset 크기는 scene 생성 때 확정한다. 먼저 과제 비율 **HOLDING/SIT/CLIMB/AT/ON_TOP = 5/10/25/30/30**의 혼합 크기 분포를 만들고, reset에서는 해당 크기가 허용하는 과제를 `p(task)/허용 크기 조합 수`에 비례해 다시 선택한다. 이는 전체 혼합 분포에서 목표 과제 비율과 과제별 균등 크기를 보존한다. 유한한 환경 pool의 실측 비율에는 표본 오차가 있다. 물리 상자 randomAssignment를 끄고 source 0/1·support 2/3을 고정해 실제 크기와 RSI 캐시의 owner 연결을 보존한다. 토큰·edge 입력 순열은 계속 사용한다. 고정 평가 preset은 처음부터 해당 과제 크기로 asset을 생성한다.
- Progress는 `1/(1+max(dxy-r_valid,0)/1m)`다. `r_valid=목표 상자 XY 반대각선+buffer`; buffer는 HOLDING 0.20m, SIT/CLIMB 0.30m, ON_TOP 0.10m다. AT는 점 목표여서 반대각선 없이 0.10m다. ON_TOP은 받침 크기를 사용한다.
- AT/ON_TOP RSI는 모두 **loco/pickUp/carryWith/putDown=40/10/40/10**. HOLDING·SIT·CLIMB의 기존 50/50 skill 비율은 유지한다. AMP expert 내부 skill 분포는 원본 unified와 같고 family 사전분포만 carry/sit/climb=65/10/25로 맞춘다.
- 학습 첫 로딩에서 실제 asset의 크기별로 SIT/CLIMB/pickUp/carryWith/putDown **전체 프레임**을 0.1초 물리 검사한다. ON_TOP putDown은 source·받침 크기 쌍과 reference 최종 XY에 둔 접지 받침을 함께 검사한다. 이 RSI 받침 회전은 identity다. 학습 reset 중에는 시뮬레이션 검사를 하지 않는다.
- 통과 조건은 모든 검사 step에서 finite, root 속도 변화 ≤3m/s, 상자 속도 ≤5m/s, root·상자 이동 ≤0.30m다. 미세 접촉만으로 탈락시키지 않으며 own_success를 초기화 필수 조건으로 요구하지 않는다. Reset에서는 다른 owner/남는 상자가 reference 몸을 깊게 침범하면 배치를 다시 뽑는다. 이 검사는 초기 충격 검사이고 장기 성공 보장은 아니다.
- SIT/CLIMB은 reference의 상자 접근·높이로 상호작용 구간을 추정한다. 마지막 상호작용 뒤의 퇴장 프레임은 배제하고, **70%는 유효 상호작용 프레임의 마지막 30%, 30%는 전체 유효 프레임**에서 선택한다. 후반 후보가 없으면 앞의 유효 프레임을 쓰며, 불연속 유효 구간을 유지한다. 가능한 clip끼리는 원래 clip 가중치를 사용한다.
- 요청 skill에 유효 프레임이 없으면 같은 과제의 다른 허용 skill, 최종적으로 loco를 선택한다. 실제 선택한 skill·motion ID·시간으로 humanoid pose/velocity, 담당 source, AMP history를 함께 초기화한다. humanoid pose의 크기 보정·IK는 하지 않는다. SIT/CLIMB source는 reference XY/회전과 실제 높이를, carry source는 reference 궤적과 바닥 높이 clamp를 사용한다.
- 캐시는 `output/rsi_cache/*.pkl.gz`에 저장한다. motion tensor·상자 reference·asset 파일·물리 설정·검사 기준·RSI 코드가 바뀌면 별도 캐시를 만든다. 기본 학습 SEED는 42이며 `SEED=...`로 변경할 수 있다. 같은 설정에 새 크기 조합이 생기면 누락 조합만 추가 검사한다. 첫 실행은 이 과정으로 오래 걸릴 수 있다.
- Size RSI 실행 경로는 고정 asset의 과제 배정 확률을 한 번 계산하고 reset graph를 GPU에서 배치 생성한다. 캐시 RSI 전에 기존 motion/time을 뽑아 버리던 중복 작업도 제거했다. 설정·분포·graph 의미·AMP 연결·기존 물리 캐시는 유지한다. 기존 unified 실행 경로는 그대로이며 새 프로세스부터 적용된다. 중복 난수 추출 제거로 같은 seed의 학습 궤적은 달라질 수 있다.
- TensorBoard `sampling/rsi_<task>_actual_<skill>`은 최종 수락된 reset의 실제 skill 비율, `sampling/rsi_<task>_fallback_attempt_fraction`은 전체 초기화 시도의 대체 비율이다. `sampling/rsi_ontop_putdown_fraction`도 확인한다. `sampling/rsi_<relation>_reference_success`와 `first_step_success`로 reference 초기 성공과 첫 물리 step 뒤의 성공을 비교한다. 기존 `own_success`는 실제 물리 rollout의 성공 지표다.

```bash
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_train.sh 2 2048 4
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=6 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_vnc.sh "$CKPT" 2 1 4 10
```


## Unified size RSI typed bias

`approach_scenario_stage1_unified_size_rsi_typed_bias`는 위 Size RSI의 과제 분포·물리 크기·후반 RSI·캐시·AMP·보상·성공 조건·`r_valid`를 유지하고 semantic bias 생성부만 교체한 별도 scratch 실험이다.

- 전용 train config는 `amp_ma_carry_relation_unified_size_rsi_typed_bias.yaml`이며 `relation_bias_mode: typed_lookup`을 사용한다. 기존 레거시 `lookup`과 다르다. Env variant와 train mode가 맞지 않거나 actor/critic 표 공유·bias 비활성화를 지정하면 실행 전에 거부한다.
- `(source 타입, relation, target 타입)`으로 `[3,11,3,4,2]` 표에서 layer/head bias 8개를 직접 조회한다. 표는 0으로 초기화하고 PPO/critic 역전파로 학습한다. Actor·critic은 각각 792개 파라미터의 독립 표를 갖는다. 16/32/16 edge embedding·공유 edge MLP·projection은 이 실험에서 사용하지 않는다.
- 유효 edge의 방향·endpoint에 bias를 넣고, 나머지는 타입별 NONE·대각선 SELF를 사용한다. Edge 순열·토큰/GTA/bias 동시 순열·human readout 복원은 기존과 같다. Node tokenizer·GTA·attention·action/value head는 유지한다.
- 같은 타입/관계 조합은 agent·물체 슬롯과 관계없이 같은 표 값을 사용한다. 연속 관측에 따른 attention 변화는 기존 QK/GTA가 담당한다.
- Output은 `output/approach_scenario_stage1_unified_size_rsi_typed_bias`다. 기존 Size RSI checkpoint로 직접 resume/eval할 수 없다. 이 실험에서 저장한 checkpoint는 `RESUME_CHECKPOINT`로 재개한다. 기존 output과 실행 중인 학습을 변경하지 않는다.

```bash
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_train.sh 2 2048 4
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=5 TASK_GRAPH=holding_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_vnc.sh "$CKPT" 2 1 4 10
```


## Unified size RSI typed bias + relation message

`approach_scenario_stage1_unified_size_rsi_typed_bias_message`는 typed-bias 실험에 관계 메시지만 추가한 별도 scratch 실험이다. Size RSI의 과제 분포·크기·후반 RSI 캐시·AMP·보상·성공 조건·`r_valid`를 유지한다.

- 기존 scalar bias 표 `[3,11,3,4,2]`와 QK→bias→softmax는 유지한다. 새 관계 표 `[3,11,3,64]`를 `(source 타입, relation, target 타입)`으로 조회하고 두 head에 32차원씩 나눈다. 표는 layer 간 공유하고 actor/critic은 분리한다. 추가 학습 파라미터는 각 6,336개, 합계 12,672개다.
- `relation_message: {enable: true, alpha: 1.0, init_std: 0.02}`가 전용 train config에 있다. 과제 relation은 작은 정규분포로 초기화하고 NONE/SELF·비과제 메시지는 0이다. 유효한 방향성 과제 edge에만 같은 attention 가중치를 곱해 source 노드로 합산한다. 기존 V 메시지의 GTA 복귀 변환 뒤, head 결합·output projection 전에 관계 메시지를 더한다.
- 저장된 edge만 gather/scatter로 처리하며 dense `[batch,head,node,node,32]`를 만들지 않는다. 토큰 셔플 때 src/dst도 함께 바꾸며 edge 순열·goal slot 교환에 대해 action/value/gradient가 유지되는지 검사한다.
- TensorBoard `relation_attention/{actor,critic}/layer{0..3}/node_message_rms`, `relation_message_rms`, `relation_to_node_rms`에 두 메시지의 크기를 기록한다. 기존 attention 진단과 동일하게 샘플된 관측에서 측정하며 relation 값에는 alpha가 적용된다.
- Env variant와 메시지 설정을 함께 검사한다. 기존 Size RSI/typebias checkpoint를 이 실험에 직접 resume/eval할 수 없다. 새 실험에서 생성한 checkpoint로만 `RESUME_CHECKPOINT` 재개한다. Output은 `output/approach_scenario_stage1_unified_size_rsi_typed_bias_message`다.

```bash
# Scratch 학습: GPU 점유를 확인한 뒤 지정
RESUME_CHECKPOINT='' TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_train.sh 2 2048 4
# 로컬 viewer / headless 평가 / 서버 VNC
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=5 TASK_GRAPH=holding_at EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_vnc.sh "$CKPT" 2 1 4 10
# 같은 실험 checkpoint 재개
RESUME_CHECKPOINT="$CKPT" TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_typed_bias_message_train.sh 2 2048 4
```


## Task 4종 공통 설정과 비교

아래 네 실험의 환경 설정은 checkpoint 격리용 `stage1_variant`만 다르다. Task bias/message 표현을 비교하며 과제·보상·크기·RSI·AMP와 634-D 관측을 유지한다.

| 항목 | Task message | 공유 Task MLP | Task MLP split | Task embedding |
| --- | --- | --- | --- | --- |
| 실험명 접미사 | `task_message` | `task_mlp` | `task_mlp_split` | `task_embedding` |
| Train mode | `task_role_lookup` | `task_role_mlp` | `task_role_mlp_split` | `task_type_embedding` |
| Task bias | task·출발/도착 역할·layer/head별 표 | 역할/task embedding → 공유 MLP → 공유 projection | 역할/task embedding → task별 MLP → task별 projection | task embedding → 물리 타입 쌍별 projection |
| 관계 message | task·역할별 학습 벡터 표 | 없음 | 없음 | 없음 |
| NONE/SELF 배경 | 물리 타입·관계·물리 타입별 표 | 별도 EdgeEncoder MLP | 별도 EdgeEncoder MLP, task별 분리하지 않음 | NONE/SELF embedding → task와 공유하는 타입 쌍별 projection |
| Actor/critic | 독립 | 독립 | 독립 | 독립 |

네 이름 앞에는 `approach_scenario_stage1_unified_size_rsi_`가 붙는다. Env YAML은 `tokenhsi/data/cfg/multi_agent/`, train YAML은 `tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_<접미사>.yaml`, wrapper는 `tokenhsi/scripts/multi_agent/<실험명>_{train,test,vnc}.sh`다. Output은 `output/<실험명>/<run>/`이며 저장 설정은 `relation_config.yaml`, 모델은 `nn/`, TensorBoard는 `summaries/`에 있다.

### Task와 edge

H=actor(사람), O=payload(운반 상자), G=바닥 target, S=받침 target이다. Sit/climb의 O는 payload가 아닌 target 역할이다. 각 agent가 독립 배정받으며 비율은 크기 조건부 샘플링 전의 사전 확률이다.

| Task ID | 정책 task | 사전 비율 | 정책의 단방향 연결 | 보상·RSI·AMP에 사용하는 primitive |
| ---: | --- | ---: | --- | --- |
| 0 | `sit` | 10% | H→O(target) | SIT |
| 1 | `climb` | 25% | H→O(target) | CLIMB |
| 2 | `carry_at` | 32.5% | H→O, H→G, O→G | HOLDING(H→O) + AT(O→G) |
| 3 | `carry_ontop` | 32.5% | H→O, H→S, O→S | HOLDING(H→O) + ON_TOP(O→S) |

Carry의 세 연결에는 **모두 같은 `carry_at` 또는 `carry_ontop` task ID**가 들어간다. 역할 쌍 `(actor,payload)`, `(actor,target)`, `(payload,target)`이 다르므로 bias도 달라질 수 있다. Split에서도 한 task 내부 세 연결은 같은 전용 MLP·projection을 쓴다. 단독 HOLDING task는 없고 추가 H→target 연결에는 보상을 중복 부여하지 않는다.

### 보상·RSI·AMP 및 평가

- 보상은 자기 primitive 합계 ×1, 동료 보상 ×0이며 carry를 2로 나누지 않는다. PPO의 `task_reward_w=0.5`, `disc_reward_w=0.5`는 그대로다. 팀 보상 제거와 task/AMP 혼합 가중치는 다른 설정이다.
- RSI는 SIT의 loco/sit 50/50, CLIMB의 loco/climb 50/50, 두 carry의 loco/pickUp/carryWith/putDown 40/10/40/10이다. 물리 검사한 유효 프레임에서 후반 70%/전체 30% 방식으로 뽑고, 유효 skill이 없는 경우 fallback으로 실제 비율이 달라질 수 있다.
- AMP family는 carry/sit/climb 65/10/25이며 두 carry task는 같은 carry expert family를 사용한다. RSI skill 분포와 AMP expert 분포를 동일한 것으로 해석하지 않는다.
- 공유 MLP는 `16-D src role + 32-D task + 16-D dst role → 64→64→64 → layer/head bias` 경로 전체를 task 간 공유한다. Split은 embedding 표를 유지하고 뒤의 MLP·projection만 네 task로 분리한다. NONE/SELF 배경 encoder는 두 경우 모두 task encoder와 별도다.
- Task message의 메시지는 task·역할별 **상태와 무관한 학습 벡터**다. 같은 attention으로 집계하며, 현재 동작을 입력받아 행동 지시를 생성하는 별도 네트워크는 아니다.
- Test wrapper는 기본 `HEADLESS=1`, `EVAL_SKILLS=loco`, `EVAL_SKILL_PROBS=1.0`이다. 이미 든 상태를 확인할 때는 `TASK_GRAPH=carry_at` 또는 `carry_ontop`과 `EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0`을 함께 지정한다. `holding_at/holding_ontop`은 별칭이다. 로컬 viewer는 `HEADLESS=0`, 서버 viewer는 전용 VNC wrapper를 쓴다.

### 지표 읽는 법

- 성공률 비교는 `info/epochs`에 맞춰 최근 200 epoch 평균을 사용한다. 시작 시간이 다른 split은 같은 epoch 구간도 비교한다.
- `relation/edge/{holding,at,ontop,sit,climb}/own_success`는 개별 primitive 조건이다. Carry 전체 완료율과 같지 않으며 추가 정책 edge H→G/H→S의 별도 성공률은 없다. `relation/goal/final_agent_success`는 에피소드 종료 시 담당 과제 전체 성공이다. RSI를 포함한 학습 지표이므로 처음부터 수행하는 평가 결과로 해석하지 않는다.
- Bias는 **task bias 원래 값**과 **Δ = task bias − 같은 물리 타입 쌍의 NONE bias**를 함께 본다. 4 layer × 2 head의 부호 있는 평균은 head별 편차를 가린다. 큰 양의 Δ가 NONE 억제에서 나올 수 있으며 실제 attention은 QK 점수와 함께 결정된다. Bias 증가만으로 성공률 개선 원인을 확정하지 않는다.

## Task 3종 최근 확인 기록

**2026-10-06 23:10 KST 로그 확인분**을 2026-10-07 문서에 반영했다. 실시간 상태가 아니며 아래 성공률은 각 run의 최근 200 epoch 평균이다. 세 실험 모두 2명·2048환경·4물체, seed 42로 본학습을 시작했다.

| 항목 | Task message | 공유 Task MLP | Task MLP split |
| --- | ---: | ---: | ---: |
| 확인 epoch | 10,409 | 10,767 | 3,006 |
| HOLDING | 61.6% | 61.7% | 18.6% |
| AT | 5.7% | 6.0% | 6.3% |
| ON_TOP | 4.4% | 15.7% | 4.2% |
| SIT | 9.2% | 20.4% | 0.9% |
| CLIMB | 19.9% | 20.4% | 16.2% |
| 종료 시 담당 과제 전체 성공 | 8.7% | 18.2% | 6.5% |

동일한 2,801~3,000 epoch의 전체 성공은 message 5.41%, 공유 MLP 5.70%, split 6.48%였다. 해당 확인에서 scalar NaN/Inf는 없고 최근 구간의 물리 reset 실패·동료 보상 기여는 0이었다. Split의 actor/critic 각각 네 task MLP가 본학습 checkpoint 500→1,000 사이에 모두 업데이트된 것도 확인했다. 한 seed의 학습 기록이며 분리 효과나 별도 평가 성공률을 확정하지 않는다.

Bias 확인 checkpoint는 message 10,000 / 공유 MLP 10,500 / split 3,000 epoch다. 공유 MLP의 carry_at H→O는 원래 bias +2.398, NONE −16.112, Δ +18.511로 NONE 억제의 영향이 컸다. Split의 동일 연결은 원래 +0.717, Δ +4.006이었다. 전체 edge·actor/critic 값은 로컬 분석 산출물 [CSV](../output/trend_review_20261006/three_edge_vs_none_latest.csv)·[그래프](../output/trend_review_20261006/three_edge_vs_none_latest.png)를 참조한다. 이 `latest` 산출물은 재분석 때 갱신되며 Git에 포함되지 않는다.

검증 이력은 [changelog](../changelog.md)에 있다. Split 관련 CPU 84개 및 2048환경 짧은 학습, carry_at/carry_ontop 각각 16환경·32-step headless 평가가 통과했다. 장기 성능 비교와 VNC 화면 검증을 대신하는 결과는 아니다.

## Unified size RSI task message

`approach_scenario_stage1_unified_size_rsi_task_message`는 typed message 기반의 별도 scratch 실험이다. 과제를 `sit / climb / carry_at / carry_ontop`으로 입력하며, 각 agent의 사전 샘플링 비율은 **10 / 25 / 32.5 / 32.5%**다. 실제 고정 asset에는 기존 크기 조건부 샘플링을 적용한다. 단독 HOLDING 과제는 제외한다.

- 정책 suffix는 agent당 `[valid, task, actor, payload, target]` 한 레코드다. Task ID는 위 순서의 0~3, endpoint는 canonical token ID다. 2H/4O/2G에서 `568 scene + 56 pose + 10 task = 634`차원이며, suffix는 관측 정규화를 우회해 PPO sample에 그대로 저장한다.
- `carry_at(H,O,G)`와 `carry_ontop(H,O,S)`는 각각 actor→payload, actor→target, payload→target의 **단방향 3연결**을 만든다. 셋 모두 같은 task ID지만 역할 쌍으로 별도 조회한다. `carry_ontop(H,O,S)`의 O는 위에 놓을 물체, S는 받침이다. Sit/climb은 actor→target 한 연결이다. 역방향은 일반 QK attention과 NONE bias를 사용한다.
- Actor/critic 각각 bias `[4 task,3 src role,3 dst role,4 layer,2 head]`와 message `[4,3,3,64]` 표를 학습한다. 역할은 actor=0/payload=1/target=2다. NONE/SELF 배경 bias는 기존 물리 타입 기준 `[3,2,3,4,2]` 표다. 합계 각 **2,736개** 파라미터이며 각자 독립이다. Bias 0 초기화, message std=0.02·alpha=1, layer 공유·head별 32차원·GTA 복귀 뒤 가중합은 이전 message 실험과 같다.
- 정책 task packet은 현재 보상 graph의 owner와 endpoint에서 매번 구성한다. 보상 graph의 HOLDING+AT/ON_TOP은 유지하며 새 H→target 연결에는 별도 보상을 만들지 않는다. `self=1, teammate=0, edge_aggregation=self_sum`: 자기 primitive 보상을 합산한다. 최대 task 보상은 sit/climb 0.6, carry 1.2이며 carry를 2로 나누지 않는다. State/progress/success·r_valid·성공 saturation·패널티 식은 유지한다.
- 내부 template 슬롯은 기존 순서 `[HOLDING,SIT,CLIMB,HOLDING_AT,HOLDING_ON_TOP]`, 비율 `[0,.10,.25,.325,.325]`이다. RSI 프레임·물리 캐시와 AMP family/expert 연결은 기존 primitive graph를 사용한다. Carry family 총량 65%와 크기 분포가 같으므로 기존 asset 생성·RSI 캐시를 재사용한다. YAML의 HOLDING 크기/RSI 항목은 내부 공통 형식으로 남지만 단독 과제는 뽑지 않는다.
- Token 순열 시 bias 두 축·메시지 endpoint·GTA pose를 함께 바꾸고 human readout을 복원한다. Task/edge 레코드 순서에는 의존하지 않는다. `TASK_GRAPH=carry_at/carry_ontop`을 지원하고 기존 `holding_at/holding_ontop`은 별칭이다. `holding` 단독 preset은 거부한다.
- 전용 train YAML은 `amp_ma_carry_relation_unified_size_rsi_task_message.yaml`, mode는 `task_role_lookup`, output은 `output/approach_scenario_stage1_unified_size_rsi_task_message`다. Packet version 5와 별도 variant로 기존 checkpoint의 직접 resume/eval을 거부한다. 같은 새 실험 checkpoint로만 재개한다. 성공률 로그의 HOLDING/AT/ON_TOP은 계속 **보상 primitive** 기준이며 정책 task 분류와 구분한다.

```bash
# Scratch 학습
RESUME_CHECKPOINT='' TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_train.sh 2 2048 4
# 로컬 viewer / headless 평가 / carryWith 상태의 서버 VNC
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=5 TASK_GRAPH=carry_at EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_vnc.sh "$CKPT" 2 1 4 10
# 같은 실험 checkpoint 재개
RESUME_CHECKPOINT="$CKPT" TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_message_train.sh 2 2048 4
```

## Unified size RSI task MLP

`approach_scenario_stage1_unified_size_rsi_task_mlp`는 위 task message의 env 설정을 그대로 사용하는 별도 scratch 실험이다. Env YAML은 checkpoint 분리용 `stage1_variant`만 다르다. 과제 비율 10/25/32.5/32.5%, 단독 HOLDING 제외, carry의 단방향 3연결, 자기 primitive 보상 합계(`self=1, teammate=0`), 크기 분포·물리 검사 RSI·AMP family/전문가 연결·634-D task packet을 모두 공유한다. Carry의 loco/pickUp/carryWith/putDown RSI는 AT와 ON_TOP 모두 40/10/40/10이다.

- 정책은 `task_role_mlp`다. 출발 역할 16-D·task 32-D·도착 역할 16-D 임베딩을 연결한 뒤 `64→64→64` ReLU MLP와 layer/head projection으로 attention bias를 만든다. 36가지 task/역할 조합을 먼저 인코딩하고 각 연결에 조회한다. Actor/critic은 별도 파라미터를 갖는다.
- NONE/SELF 배경은 기본 Size RSI처럼 물리 타입·관계·물리 타입의 `EdgeEncoder` MLP를 사용한다. 두 bias projection은 0으로 초기화한다. GTA·토큰 순열은 유지한다.
- `relation_message.enable=false`이며 관계 메시지 표와 가산 경로는 없다. 일반 attention의 value 집계는 유지한다. 기존 task message와 네트워크 및 checkpoint 계약이 달라 직접 resume/eval을 거부한다.
- 전용 env/train YAML·train/test/VNC·output을 사용한다. RSI 캐시 경로는 `output/rsi_cache`로 같으며 내용 hash가 맞으면 재사용한다. 아래 GPU 0은 현재 로컬 서버 예시이며 실행 전 점유를 확인한다.

```bash
# Scratch 학습: 2명, 2048환경, 4물체
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_train.sh 2 2048 4
# 짧은 학습 확인: 본학습과 output 분리
TOKENHSI_GPU=0 MAX_ITERATIONS=1 RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_train.sh 2 2048 4
# CKPT에 같은 task MLP 실험의 checkpoint 절대 경로를 지정
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=carry_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=0 TASK_GRAPH=carry_ontop EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_vnc.sh "$CKPT" 2 1 4 10
# 같은 task MLP checkpoint에서 재개
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_train.sh 2 2048 4
```

## Unified size RSI task MLP split

`approach_scenario_stage1_unified_size_rsi_task_mlp_split`는 task MLP의 공유 영향을 비교하는 별도 scratch 실험이다. Env는 `stage1_variant`만 다르며 과제·3연결·자기 보상·크기·RSI·AMP·634-D packet은 task MLP 및 task message와 같다. Carry의 RSI는 AT/ON_TOP 모두 loco/pickUp/carryWith/putDown 40/10/40/10이고 기존 물리 캐시를 재사용한다.

- 출발 역할 `Embedding(3,16)`·task `Embedding(4,32)`·도착 역할 `Embedding(3,16)`을 유지한다. 출발·도착 표는 별도이며 네 task가 공유한다. Task 표는 task별 다른 행을 쓴다.
- Concat 이후 `64→64→64` MLP를 sit/climb/carry_at/carry_ontop별로 네 개 만든다. Bias projection도 `[task=4,layer=4,head=2,64]`로 분리한다. 한 task 내부의 세 역할 연결은 같은 전용 MLP·projection을 사용한다. Task 간 역할 embedding 공유의 영향은 계속 남는다.
- Task별 9개 역할 조합만 처리한 뒤 환경별로 조회한다(총 36개). Actor/critic은 전체 파라미터가 독립이다. NONE/SELF 배경은 기존 별도 MLP를 유지하며 task별로 분리하지 않는다. 관계 메시지는 없고 GTA는 유지한다.
- Task encoder는 actor/critic 각각 35,552개 파라미터다(기존 9,056개). 전체 네트워크는 4,209,730개로 52,992개(+1.27%) 증가한다. 추가 FP32 weight 약 0.20 MiB, gradient·Adam 상태까지 약 0.81 MiB이며 이는 전체 실행 메모리 증가의 실측값은 아니다.
- 전용 train mode는 `task_role_mlp_split`이다. Packet v5는 유지하지만 variant/fusion 및 weight shape가 달라 task message·공유 task MLP checkpoint를 직접 resume/eval할 수 없다. 전용 config·train/test/VNC·output에서 scratch로 시작한다.

```bash
# Scratch 학습: 2명, 2048환경, 4물체
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp_split bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_train.sh 2 2048 4
# 짧은 확인용 output
TOKENHSI_GPU=0 MAX_ITERATIONS=1 RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp_split_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_train.sh 2 2048 4
# CKPT에 같은 task MLP split 실험의 checkpoint 절대 경로를 지정
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=carry_at EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=0 TASK_GRAPH=carry_ontop EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_vnc.sh "$CKPT" 2 1 4 10
# 같은 split checkpoint 재개
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT="$CKPT" OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_mlp_split bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_mlp_split_train.sh 2 2048 4
```

## Unified size RSI task embedding

`approach_scenario_stage1_unified_size_rsi_task_embedding`은 공유 task MLP 기반의 별도 scratch 실험이다. Env YAML은 `stage1_variant`만 다르고, train YAML은 `relation_bias_mode=task_type_embedding`만 다르다. 보상·성공 조건·과제 분포·634-D task packet·RSI·AMP·GTA·토큰/edge 셔플을 유지한다.

- Actor/critic 각각 `Embedding(6,64)` 하나를 둔다. 카테고리는 `[sit, climb, carry_at, carry_ontop, NONE, SELF]`다. 역할/물리 타입/agent ID를 embedding 입력으로 넣지 않으며 중간 MLP·관계 message가 없다.
- Projection은 `[src type=3, dst type=3, layer=4, head=2, 64]`다. `b[c,s,t,l,h] = dot(E[c], W[s,t,l,h])`이며 H/O/G 방향 쌍·layer/head마다 독립이다. Task·NONE·SELF는 **같은 W를 공유**하고 카테고리 embedding만 다르다. Actor와 critic은 전체 파라미터가 독립이다.
- 같은 task·타입 쌍·layer/head라면 agent/물체 ID와 무관하게 bias가 같다. `carry_at`의 H→O/H→G/O→G는 embedding을 공유하고 projection은 다르다. `climb H→O`와 `carry_at H→O`는 projection을 공유하고 embedding은 다르다.
- 받침 S는 O 타입이다. 따라서 `carry_ontop`의 H→운반 상자와 H→받침은 같은 bias를 받는다. 운반 상자→받침은 O→O projection을 쓴다. 이전 actor/payload/target **역할 쌍별** 설계와 구분한다.
- NONE은 task 연결이 없는 방향, SELF는 대각선에 적용한다. 배경 위에 task bias를 중복 가산하지 않고 유효 task 연결을 교체한다. 역방향·다른 agent의 비연결 토큰은 NONE으로 남는다.
- Canonical endpoint의 실제 H/O/G 타입으로 bias를 만든 뒤 토큰 순열에 맞춰 bias 두 축과 GTA pose를 함께 이동한다. Human readout은 역순열로 복원한다. Task record/primitive edge 순서로 projection을 고르지 않는다.
- Embedding은 std=.02 랜덤, projection은 0으로 초기화한다. 최초 bias는 0이며 학습 후 성장 속도·성공률 개선은 보장하지 않는다. Encoder당 embedding 384 + projection 4,608 = **4,992개** 파라미터다. 전체 네트워크는 4,130,626개로 공유 task MLP보다 26,112개 적다.
- RSI는 SIT loco/sit 50/50, CLIMB loco/climb 50/50, 두 carry loco/pickUp/carryWith/putDown 40/10/40/10을 유지한다. AMP carry/sit/climb family 65/10/25 및 expert matching, size RSI 캐시 `output/rsi_cache`도 그대로다. 보상은 자기 primitive 합계 ×1·동료 ×0이고 PPO task/AMP 가중치는 0.5/0.5다.
- 전용 variant/fusion과 weight 구조로 기존 세 실험 checkpoint의 직접 resume/eval을 거부한다. 학습·평가 모두 task embedding 전용 wrapper와 checkpoint를 사용한다.

2026-10-07 검증: 관련 CPU 91개, GPU 0의 2048환경 짧은 학습(저장 epoch 2), 저장 모델의 carry_at/carry_ontop + carryWith 각각 16환경·32-step headless 평가를 완료했다. 학습 scalar 233종/466값 모두 finite, 물리 reset 실패·물체 binding/보상 합산 오류 0이며 6개 카테고리와 타입 쌍 projection의 학습 경로를 확인했다. 검증 output은 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_check/`와 `_check_carry_at/`, `_check_carry_ontop/`다. 본학습·장기 성능 및 VNC 화면은 아직 검증하지 않았다.

```bash
# Scratch 학습: 2명·2048환경·4물체. 실행 전 GPU 점유 확인
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_train.sh 2 2048 4
# 짧은 검증은 별도 output
TOKENHSI_GPU=0 MAX_ITERATIONS=1 RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_train.sh 2 2048 4
# CKPT에는 task embedding 실험의 checkpoint 경로 지정
TOKENHSI_GPU=0 HEADLESS=0 TASK_GRAPH=carry_at EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=0 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=0 TASK_GRAPH=carry_ontop EVAL_SKILLS=carryWith EVAL_SKILL_PROBS=1.0 bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_vnc.sh "$CKPT" 2 1 4 10
# 같은 실험 checkpoint 재개
TOKENHSI_GPU=0 MAX_ITERATIONS='' RESUME_CHECKPOINT="$CKPT" OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_train.sh 2 2048 4
```

## Carry task embedding distillation

`approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill`은 task embedding 모델에서 두 agent가 carry_at/carry_ontop만 각각 50/50 독립 샘플링하는 전용 실험이다. 2명·2048환경·4물체와 기존 task/NONE/SELF `Embedding(6,64)`·H/O/G 타입 쌍 projection을 유지한다. Sit/climb embedding 슬롯과 공통 설정 행은 남지만 해당 과제는 샘플링하지 않는다. 기존 네 과제 실험과 checkpoint/output을 분리한다.

- Source 크기는 carry 범위 XYZ 각각 0.20~0.60m, 0.05m 격자로 생성한다. 받침 범위와 밀도는 기존 설정을 유지한다. 고정 asset의 조건부 과제 확률도 carry 두 개만 허용한다.
- RSI는 두 carry 모두 loco/pickUp/carryWith/putDown 40/10/40/10, 물리 검사·후반 프레임 규칙을 유지한다. 공통 RSI 캐시 코드는 그대로이며 실제 크기 조합의 누락 profile은 최초 검사한다.
- AMP family는 carry 100%, 실제 전문가 skill 분포는 loco/omomo/pickUp/putDown = 1/3, 1/3, 1/6, 1/6이다. AMP one-hot·전문가/리플레이 matching은 유지한다. 보상은 자기 primitive 합계×1, 동료×0, PPO task/AMP 가중치는 0.5/0.5다.
- 동결한 원본 unified Stage 1 teacher에 현재 agent 신체·담당 상자·목표를 354-D로 변환해 전달한다. AT는 graph goal, ON_TOP은 실제 회전 bbox 높이를 반영한 받침 위 source 중심을 목표로 한다. Teacher의 Carry task만 활성화하고 원본 정규화 통계를 사용한다.
- 학생 rollout에서 teacher의 32-D Gaussian mean/sigma를 저장하고, scene/agent minibatch 정렬을 유지해 `0.001 × KL(teacher || student)`를 기존 PPO/AMP loss에 더한다. Critic에는 직접 KL gradient를 주지 않으며 teacher는 업데이트하지 않는다.
- 기본 teacher는 `/home/hwanhee/CVPR2027/TokenHSI/output/tokenhsi/ckpt_stage1.pth`다. `TEACHER_CHECKPOINT`, `TEACHER_KL_COEF`, `TEACHER_GRAD_CHECKS`로 변경한다. Train YAML에도 기본 teacher 설정이 들어 있다. 평가는 학생 checkpoint만 사용하므로 teacher 파일이 필요 없다.
- TensorBoard `distill/`에 KL·mean RMSE·teacher/학생 sigma·PPO/critic loss·template label 수를 기록한다. `TEACHER_GRAD_CHECKS=2`는 초기 두 optimizer update의 KL gradient·critic 비의존·actor 갱신과 teacher 동결을 검사한다.

```bash
# 본학습: GPU 5, 2명·2048환경·4물체. 실행 전 nvidia-smi로 점유 확인
TOKENHSI_GPU=5 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_train.sh 2 2048 4
# 짧은 확인 학습: 현재 종료 조건상 MAX_ITERATIONS=1은 epoch 2까지 진행
TOKENHSI_GPU=5 MAX_ITERATIONS=1 TEACHER_GRAD_CHECKS=2 OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_train.sh 2 2048 4
# CKPT는 이 실험에서 저장한 학생 checkpoint
CKPT='/absolute/path/to/student.pth'
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=carry_at bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_test.sh "$CKPT" 2 16 4 3
TOKENHSI_GPU=5 TASK_GRAPH=carry_ontop bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_vnc.sh "$CKPT" 2 1 4 10
```

Test wrapper는 기본 `HEADLESS=0`이며 서버의 화면 없는 평가는 `HEADLESS=1`을 명시한다. 기존 task embedding 또는 옛 34번 distill checkpoint를 직접 resume하지 않는다. 짧은 실행 확인은 장기 운반 성공률 검증과 구분한다.

2026-10-07 연결 검증: CPU 63개 통과. GPU 5·2048환경에서 epoch 2/frame 262144까지 학습·저장했고 scalar 212종/424값 모두 finite, 물리 reset 실패 0, teacher 동결·KL actor/embedding gradient·critic 비의존을 확인했다. carry_at/carry_ontop + carryWith 각각 16환경·32-step headless 평가가 정상 종료됐다. 두 짧은 평가의 운반 완수율은 0이며 장기 성능 결과가 아니다. RSI 캐시 6,640개 profile은 `output/rsi_cache/6361a69fcf429a02a66602bb.pkl.gz`, 검증 결과는 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_check/`와 `_check_carry_at/`, `_check_carry_ontop/`에 있다. 본학습은 시작하지 않았다.

## Four-task embedding distillation

`approach_scenario_stage1_unified_size_rsi_task_embedding_distill`은 원래 task embedding 실험의 sit/climb/carry_at/carry_ontop **10/25/32.5/32.5%**와 행동별 크기·RSI·AMP·보상·네트워크를 유지하고 원본 unified teacher KL을 추가한다. Env YAML은 원본과 variant만 다르다. 단독 HOLDING·별도 Traj task는 샘플링하지 않는다. Carry 전용 distill과는 독립된 scratch 실험이며 checkpoint를 직접 혼용하지 않는다.

- Actor/critic 각각 task 4개+NONE+SELF `Embedding(6,64)`와 H/O/G 타입 쌍 projection을 사용한다. 중간 MLP·관계 message는 없다.
- Teacher는 SIT→Sit, CLIMB→Climb, 두 carry→Carry로 라우팅한다. SIT 목표는 상판+pelvis clearance이며 facing은 상자의 로컬 +X다. CLIMB 목표는 상판+character height다. AT graph goal·ON_TOP 회전 bbox 목표·원본 teacher 정규화·teacher 동결·학생 rollout KL은 carry distill과 같은 구현이다.
- RSI는 SIT loco/sit 50/50, CLIMB loco/climb 50/50, 두 carry loco/pickUp/carryWith/putDown 40/10/40/10이다. AMP carry/sit/climb family는 65/10/25이며 전문가/리플레이 matching을 유지한다. 크기별 물리 RSI 캐시는 공유하되 누락 profile은 검사 후 추가한다.
- Teacher 기본 경로는 `/home/hwanhee/CVPR2027/TokenHSI/output/tokenhsi/ckpt_stage1.pth`, KL 계수는 0.001이다. `TEACHER_CHECKPOINT`, `TEACHER_KL_COEF`, `TEACHER_GRAD_CHECKS`를 지원한다. 기존 PPO·AMP 학습에 KL을 더하며 학생 평가는 teacher 파일 없이 가능하다.

```bash
# 본학습: 실행 전 GPU 점유 확인, 기존 carry 학습은 자동 중단하지 않음
TOKENHSI_GPU=5 MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_distill bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_train.sh 2 2048 4
# 짧은 연결 검증
TOKENHSI_GPU=5 MAX_ITERATIONS=1 TEACHER_GRAD_CHECKS=2 OUTPUT_PATH=output/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_check bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_train.sh 2 2048 4
CKPT='/absolute/path/to/four_task_student.pth'
# 로컬 viewer와 headless 평가; TASK_GRAPH=sit/climb/carry_at/carry_ontop/random_scenario
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=sit bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_test.sh "$CKPT" 2 1 4 10
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_test.sh "$CKPT" 2 16 4 3
# 서버 VNC
TOKENHSI_GPU=5 TASK_GRAPH=climb bash tokenhsi/scripts/multi_agent/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_vnc.sh "$CKPT" 2 1 4 10
```

2026-10-07 연결 검증: 관련 CPU 56개 통과. GPU 5·MPS·2048환경에서 epoch 2/frame 262144까지 학습·저장했고 scalar 252종/504값 모두 finite, 물리 reset 실패·binding/보상 합산 오류 0을 확인했다. 네 task 모두 teacher label로 들어가며 teacher 동결·학생 actor/embedding gradient·critic KL 비의존을 확인했다. 저장 학생으로 sit/sit RSI·climb/climb RSI·두 carry/carryWith 각각 16환경·32-step 평가가 정상 종료됐다. Output은 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_check/`와 `_check_<task>/`다. 공통 RSI 캐시는 3,622개 profile을 추가해 총 10,262개다. 이는 연결 검증이며 장기 성능 결과가 아니다. 새 본학습은 시작하지 않았으며 기존 carry 학습을 유지했다.

<a id="stage2-joint-carry-task-ca"></a>
## Stage 2 Joint carry task CA

`approach_stage2_joint_carry_task_embedding`은 원본 Stage 1 task embedding checkpoint를 전이하는 **실행 가능한** 공동운반 실험이다. 원본·네 과제 distill·carry-only distill의 호환 checkpoint를 허용하며 teacher는 연결하지 않는다. 기존 29번과 config·output·checkpoint를 구분한다.

- 2명·2048환경·4물체. Scene별 joint AT/ON_TOP 50/50이며 두 task가 같은 payload와 목표를 공유한다. Payload O0는 XYZ 52×80×40cm·밀도100, ON_TOP 받침 O2는 운반 상자와 가로·세로가 같은 52×80×30cm로 고정한다. 크기 확장은 별도 RSI 재검증이 필요하다.
- Actor encoder·task embedding·관찰 RMS를 동결한다. `[task embedding; 최종 human; payload; target]`을 256→128→64로 결합한다. Human Q, task K/V, 2-head CA와 head별 NONE/SELF/COUPLED bias(0 초기화)를 학습한다. Action head는 `[W,0]`으로 확장하며 critic·AMP discriminator도 학습한다.
- SA entity/token·GTA/bias 셔플 후 canonical 순서로 복원한다. CA는 `shuffle_task_order: true`로 task packet을 따로 섞고 K/V·relation bias를 함께 계산한다. Human action 순서는 유지한다. 현재 두 task에는 SELF/COUPLED만 등장하므로 NONE bias는 학습되지 않는다.
- 앵커는 object-local `(0, ±(L/2−0.15), 0)`, 현재 ±25cm다. 각 agent의 양손 중점에 대해 가까운 앵커 HOLDING 보상을 준다. 두 사람이 서로 다른 앵커에서 `exp(-10d²) >= 0.9`여야 AT/ON_TOP의 state/progress/success 보상을 모두 연다. 각 agent 보상은 자기 HOLDING + gated placement 합계이며 세 항의 가중치는 각각0.2다. 포화·완료 latch는 없고 한 명이 놓치면 placement는 다시0이다. 접촉력·하중 분담을 직접 판정하지 않으며 최종 손 놓기는 아직 별도 과제로 다루지 않는다.
- Collision은 `stage2_team/output/approach_stage2_rescue_shared9_cpa_team03`의 CPA 수식을 따른다. Root XY 상대 위치·속도로 접근 방향 가중치 × 최근접 거리 위험도 × `0.99^(t_CPA/control_dt)`를 계산하고 기존 계수0.5·거리0.7m를 적용한다. `relationReward.joint_carry.collision`에 mode/ttc_discount를 기록한다. 정지·동일 속도·멀어지는 쌍에는0이며 static 거리 항을 추가하지 않는다. 변경 전 distance 방식 Stage 2 checkpoint는 reward 계약 차이로 직접 resume/eval이 거부된다. Stage 1 source 전이는 유지한다.
- RSI는 scene당 같은 skill/행의 두 사람·공유 상자·속도를 함께 복원하고 공통 yaw/translation을 적용한다. AT의 loco/pickUp/carryWith/putDown은 40/10/40/10, ON_TOP은 50/10/40/0이다. 바닥용 putDown을 ON_TOP에 적용하는 평가 요청은 거부한다. Loco는 상자 양쪽에서 접근하도록 배치한다.
- AMP expert는 기존 single-human carry family만 사용한다: loco 1/3, OMOMO 1/3, pickUp 1/6, putDown 1/6. 보정 joint clip은 RSI 전용이다. Reset AMP history는 복원 상태를 반복하며 실제 rollout으로 교체한다. Family one-hot은 정규화에서 제외하고 demo/replay matching을 유지한다. 평가 RSI skill을 바꿔도 expert skill 인덱스는 원래 목록을 따른다.

```bash
# 사용자 지정 Stage 1 carry distill epoch11000 (2026-10-08 전이 확인)
SOURCE=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00011000.pth

# GPU 5 짧은 확인; 본학습과 output 분리
TOKENHSI_GPU=5 STAGE1_CHECKPOINT="$SOURCE" MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_stage2_joint_carry_task_embedding_check bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_task_embedding_train.sh 2 2048 4

# 본학습 (위 확인과 별도 실행)
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS TOKENHSI_GPU=5 STAGE1_CHECKPOINT="$SOURCE" OUTPUT_PATH=output/approach_stage2_joint_carry_task_embedding bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_task_embedding_train.sh 2 2048 4

# CKPT에는 이 Stage 2 실험에서 저장한 checkpoint 경로를 지정
TOKENHSI_GPU=5 RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_task_embedding_train.sh 2 2048 4

# 로컬 viewer / 화면 없는 평가 / 서버 VNC
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=joint_carry_at EVAL_SKILLS=carryWith bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_task_embedding_test.sh "$CKPT" 2 16 4 1
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=joint_carry_ontop EVAL_SKILLS=carryWith bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_task_embedding_test.sh "$CKPT" 2 16 4 1
TOKENHSI_GPU=5 TASK_GRAPH=joint_carry_at EVAL_SKILLS=carryWith bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_task_embedding_vnc.sh "$CKPT" 2 1 4 10
```

평가 preset은 `random_scenario`, `joint_carry_at`, `joint_carry_ontop`이다. 기본 초기화는 loco이며 `EVAL_SKILLS`, `EVAL_SKILL_PROBS`, `EPISODE_LENGTH`로 조정한다. 임시 source와 짧은 확인 checkpoint는 공동운반 성능 검증용 완성 모델이 아니다.

2026-10-07 최초 연결 검증(당시 받침 72×100×30cm): CPU 59개, GPU 5·2048환경 짧은 학습/저장/재개, 실제 reset 2048환경 초기 물리 검사 및 1024환경 부분 reset을 통과했다. `output/approach_stage2_joint_carry_task_embedding_check/`의 `saved_check.json`, `reset_check.json`, `resume_check.json`을 참조한다. AT/ON_TOP carryWith, AT putDown, ON_TOP loco를 각각 16환경·32-step으로 평가해 실행 경로를 확인했다. 운반 완수율은 모두0이며 장기 학습·VNC 화면 검증은 하지 않았다.

이후 받침을 52×80×30cm로 축소하고 관련 CPU 6개·GPU 5의 실제 Stage 2 2048환경 초기 물리/AMP 이력/부분 reset 검사를 통과했다. 새 크기와 결과는 `output/approach_stage2_joint_carry_support_size_check/reset_check.json`에 기록했다. 받침 위 실제 배치 성공을 검증한 결과는 아니다.

Collision을 shared9 CPA로 교체한 뒤 CPU 24개와 GPU 5·2048환경 실제 reward 8-step 검사를 통과했다. 원본 수식 비교·초기 물리·부분 reset 결과는 `output/approach_stage2_joint_carry_cpa_check/reset_check.json`에 있다.


<a id="stage2-mixed80-locoamp"></a>
## Mixed80 보행 AMP 보강·CA ablation

**현재 데이터:** 사용자 확인한 자세 보존 방식을 전체12개에 적용한 `teamhoi_retarget_posture_all`을 사용한다. 이전 굽힌 보정본을 사용한 확인용 checkpoint는 데이터 계약이 달라 직접 resume/eval할 수 없다. 지정 Stage1에서 새로 시작한다.

두 실험은 기존 Mixed80의 공동1638·단독410환경, 상자 크기, 공동 HOLDING/placement gate·CPA·reward 계수, paired/단독 RSI와 초기 AMP 이력을 그대로 사용한다. 새 동시집기 보상은 추가하지 않았다. 현재 본학습은 Stage1 carry distill epoch23000에서 각각 새로 시작하며 PPO optimizer/epoch는 초기화한다. 기존 Stage2를 이어 학습하는 설정이 아니다.

| 설정 | CA 유지 | Head-only |
| --- | --- | --- |
| 이름 접미사 | `locoamp_task_embedding` | `locoamp_head_only` |
| actor 입력 | `[H64, CA64]` | `H64` |
| head 초기화 | Stage1 `[W, 0]` | Stage1 `W` 그대로 |
| 학습하는 정책 부분 | CA + action head | action head만 |
| 동결 | Stage1 actor encoder·관측 RMS | 동일 |

Critic·AMP 판별자·AMP 정규화 통계는 두 실험 모두 학습한다. SA의 기존 task embedding·shared-task 처리·토큰 셔플은 같고, head-only에는 CA 모듈/파라미터 자체가 없다. 보상·RSI가 같아도 actor의 SA가 teammate 상태를 읽을 수는 있다.

AMP는 사람마다 평가하는 기존 carry family 판별자를 유지한다. Expert마다 **기존 분포80%·후진10%·옆걸음10%**를 뽑고, 새 그룹 안에서는 클립을 균등하게 고른 뒤 유효한 연속10-frame 구간의 시간을 균등하게 뽑는다. 특정 사람에게 이동 방향을 고정 배정하지 않는다. 전체 유효 비율은 기존 loco26.67%·OMOMO26.67%·pickUp13.33%·putDown13.33%·후진10%·옆걸음10%다. `skillDiscProb`는 기존 그룹 내부 비율이며, 새 비율은 `jointCarryAblation.amp`로 명시한다.

보정 보행은 `joint_carry/teamhoi_retarget_posture_all/amp_motions.yaml`을 통해 **AMP 전용 별도 MotionLib**로 로드한다. RSI는 원래 `dataset_loco_sit_carry_climb.yaml`을 계속 사용한다. 보정 데이터의 검증 상태·SHA256 및 YAML의9/3개·균등 weight를 확인한다. Checkpoint에 데이터 해시·AMP 비율·CA/head-only 계약을 저장해 구형 Mixed80 또는 서로 다른 ablation의 직접 resume/eval을 거부한다.

연결 검증: GPU5·2048환경에서 두 모델 각각 epoch2 저장→epoch3 재개, 16환경32-step headless 평가 로드를 통과했다. 지정 Stage1 대비 actor69개 tensor·관찰 RMS 불변, head64/128·CA 부재/존재, head/critic/discriminator 갱신과 scalar finite를 확인했다. 실제 AMP30000개 비율은79.68/10.25/10.07%이고 RSI 라이브러리/이력·부분 reset도 보존됐다. 결과는 각 `output_etc/approach_stage2_joint_carry_mixed80_locoamp_*_posture_check/saved_check.json`, head-only의 `reset_check.json` 및 별도 `*_posture_resume_check/`, `*_posture_eval_check/`다. 본학습과 장기 성능은 미검증이다.

새 학습(각각 별도 실행):

```bash
TOKENHSI_GPU=5 SEED=42 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00023000.pth \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_task_embedding_train.sh 2 2048 4

TOKENHSI_GPU=5 SEED=42 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00023000.pth \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_head_only_train.sh 2 2048 4
```

기본 output은 각 config와 같은 이름의 `output/approach_stage2_joint_carry_mixed80_locoamp_*`다. 짧은 확인은 `OUTPUT_PATH=output/<실험명>_check MAX_ITERATIONS=1`로 분리하고, 본학습 전 `RESUME_CHECKPOINT`·`MAX_ITERATIONS` 잔여값을 확인한다. 재개할 때는 같은 실험의 `RESUME_CHECKPOINT`를 사용한다.

로컬 viewer(각 실험 checkpoint 경로를 지정; 화면 없는 평가는 `HEADLESS=1`):

```bash
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_task_embedding_test.sh "$CA_CHECKPOINT" 2 16 4 1

TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_head_only_test.sh "$HEAD_ONLY_CHECKPOINT" 2 16 4 1
```

서버 VNC(10환경이면 공동8·단독2):

```bash
TOKENHSI_GPU=5 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_task_embedding_vnc.sh "$CA_CHECKPOINT" 2 10 4 10

TOKENHSI_GPU=5 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_head_only_vnc.sh "$HEAD_ONLY_CHECKPOINT" 2 10 4 10
```

<a id="stage2-mixed80-locoamp-ca4"></a>
## Mixed80 LocoAMP 4층 task CA

`approach_stage2_joint_carry_mixed80_locoamp_ca4_task_embedding`은 기본 `locoamp_task_embedding`의 CA 구조만 변경한 별도 실험이다. Align 방향 보상은 없다. 공동80/단독20, reward/CPA, 기존80%·후진10%·옆걸음10% AMP와 데이터 해시, paired/단독 RSI, actor encoder·관측 RMS 동결은 기반과 같다. Critic·AMP 판별자·AMP 정규화 통계는 계속 학습한다.

64차원·2head의 **CA+FFN(64→128→64)+LayerNorm+residual** 블록4개를 쌓는다. `h0=H`, 각 층에서 `u=h+CA(LN(h), I, I; bias)`, `h_next=u+FFN(LN(u))`를 계산한다. Interaction memory `I`는 Stage1 최종 토큰으로 한 번 만들고 공유하며, 각 층의 attention·FFN·두 LayerNorm·head별 NONE/SELF/COUPLED bias는 독립적으로 학습한다. 첫 층 이후 query는 앞 층에서 갱신한 피처다. 최종 action head 입력은 `[원래 H64, h4의64]`이며 Stage1 첫 weight를 `[W, 0]`으로 전이하므로 초기 action은 Stage1과 같다.

SA 기존 셔플을 유지하고, CA task packet 셔플·grounding·관계 판정·padding mask는 forward당 한 번 계산해 네 층에서 재사용한다. Attention weight 출력은 만들지 않는다. K/V projection은 층별 weight가 다르므로 각 층에서 계산한다. CA 파라미터57,798→175,064, 전체4,253,960→4,371,226(+117,266; +2.76%)이다. 층 수와 FFN 추가 효과를 함께 비교하는 실험이다.

Checkpoint의 정책 계약은 `task_ca4`다. 기존1층/head-only/Align checkpoint를 직접 resume/eval하지 않으며, 지정 Stage1에서 새로 시작한다. 본학습 명령은 `RESUME_CHECKPOINT`·`MAX_ITERATIONS`가 export되지 않은 셸 기준이다.

연결 검증: 관련 CPU40개·학습 후 SA/CA 셔플 재검사, GPU5·2048환경 전이→epoch2 저장→epoch3 재개, 16환경32-step headless 혼합 평가를 통과했다. Actor encoder69개 tensor·관측 RMS 동결, 네 층 bias/context head 갱신, 모델/scalar133종 finite를 확인했다. AMP30000개 실제 비율79.68/10.25/10.07%, 공동1638/단독410·RSI 초기 이력·부분 reset 검사도 통과했다. 결과는 `output/approach_stage2_joint_carry_mixed80_locoamp_ca4_task_embedding_check/{saved_check,reset_check}.json`이다. 실제 VNC 화면·장기 성능·속도 비교는 미검증이며 본학습은 시작하지 않았다.

```bash
TOKENHSI_GPU=5 SEED=42 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00023000.pth \
OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_task_embedding \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_task_embedding_train.sh 2 2048 4
```

재개는 같은 실험의 `RESUME_CHECKPOINT="$CA4_CHECKPOINT"`를 지정한다. 짧은 검증은 `MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_task_embedding_check`로 분리한다. 아래 `CA4_CHECKPOINT`에는 이 실험에서 저장한 `.pth` 경로를 지정한다.

로컬 viewer / 화면 없는 평가 / 서버 VNC (`random_scenario`·10환경은 공동8/단독2):

```bash
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_task_embedding_test.sh "$CA4_CHECKPOINT" 2 16 4 1

TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_task_embedding_test.sh "$CA4_CHECKPOINT" 2 16 4 1

TOKENHSI_GPU=5 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_task_embedding_vnc.sh "$CA4_CHECKPOINT" 2 10 4 10
```

<a id="stage2-mixed80-locoamp-ca4-goal360"></a>
## Mixed80 LocoAMP CA4 공동 목표360°

`approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding`은 CA4 기반의 별도 env/train·train/test/VNC 실험이다. 공동 loco/pickUp/carryWith는 운반 상자 중심에서 각도0~360°·거리2.2~2.8m를 독립 균등 샘플링한다. AT는 공유 목표, ON_TOP은 공유 받침을 해당 XY에 놓는다. 후보가 기존 물체와 겹치거나 사람 root 1m 이내이거나 받침 확장 bbox(10cm)에 body가 들어가면 재샘플링한다. 따라서 최종 분포는 이 공간 조건으로 제한된다. 64회 실패하면 중단하며 사람·운반 상자를 다시 뽑지 않는다.

공동 AT putDown은 `cooperative_rsi_candidate.npz` 마지막 `box_position`의 XY를 사용한다. 현재 snapshot 상자 중심을 뺀 뒤 사람·상자와 같은 yaw/translation을 적용하고 목표 Z는 운반 상자 반높이로 둔다. 이는 마지막 screened row를 임의 선택하는 방식이 아니다. 공동 ON_TOP putDown은 기존처럼0%다. 끝점이 기존 미사용 상자와 겹치는 putDown 장면은 해당 미사용 상자3개만 목표 반대편 반경4m에 분산 배치한다. 원본135개 중30개가 이 처리 대상이며 사람·운반 상자·목표 끝점은 보존한다.

RSI 데이터·프레임 선택·사람/운반 상자 자세와 속도·AMP history 방식·RSI 비율·단독 초기화·보상/CPA·AMP80/10/10·SA/CA 순서 셔플·Stage1 actor/RMS 동결은 유지한다. Move/Align 보상은 없다. 새 `joint_carry_goal_sampling` checkpoint 계약으로 기존 실험과 직접 resume/eval을 구분한다. Stage1 전이는 허용하며 기본 비교 출발점은 epoch30000·seed2899다.

```bash
# 새 학습: Stage2 resume 없이 Stage1에서 전이
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=6 SEED=2899 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding_train.sh 2 2048 4

# GOAL360_CHECKPOINT는 새 실험에서 저장한 .pth
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding_test.sh "$GOAL360_CHECKPOINT" 2 16 4 1

TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding_test.sh "$GOAL360_CHECKPOINT" 2 16 4 1

TOKENHSI_GPU=6 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding_vnc.sh "$GOAL360_CHECKPOINT" 2 10 4 10
```

검증(2026-10-11): 관련 CPU35개 통과(공동 ON_TOP putDown은 미사용으로1개 skip), GPU6·2048환경에서 최종 코드 Stage1 epoch30000 전이→epoch2 저장→epoch3 재개 및16환경32-step test wrapper 평가를 통과했다. Actor encoder69개 tensor/RMS 불변·네 CA 층 bias/context head 갱신·모델/scalar133종 finite를 확인했다. 공동1638/단독410, AMP30000개 비율80.09/10.24/9.67%, 부분 reset1024/32/32·초기 AMP 이력도 확인했다. 공동 AT4단계/ON_TOP3단계 각각1638개 reset에서 사람/운반 상자 RSI 자세·속도 보존, 상대 목표8방향·거리2.2~2.8m, 받침 배치/초기 clearance, putDown 끝점/목표 공간과 각4-step finite를 검사했다. PutDown의 초기 목표 거리는 원본 프레임에 따라0~3.74m다. 목표 방향 균등 후보를 공간 조건으로 거르므로 최종 각도별 빈도는 같지 않을 수 있다.

최종 결과: [reset/AMP](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding_final_check/reset_check.json), [RSI/목표](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding_final_check/goal360_runtime_check.json), [전이/재개](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding_final_check/saved_check.json). 실제 VNC 화면·장기 운반 성능은 미검증이며 본학습은 시작하지 않았다.

재개는 같은 실험의 `RESUME_CHECKPOINT="$GOAL360_CHECKPOINT"`를 train에 전달한다. 확인용 학습은 `MAX_ITERATIONS=1`·별도 `OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding_check`를 사용한다. PutDown viewer는 `TASK_GRAPH=joint_carry_at EVAL_SKILLS=putDown`으로 실행한다.

GPU reset/AMP 재검사는 아래처럼 실행한다(학습은 하지 않음).

```bash
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; PYTHONPATH=tokenhsi python tokenhsi/tests/check_joint_carry_goal360_runtime.py \
  --task HumanoidMACarry \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --transfer_checkpoint stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
  --num_envs 2048 --num_agents 2 --num_objects 4 --task_graph random_scenario \
  --output_path output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding_check --headless --no_video'
```

### Goal360 loco 공동집기 CPA 진단

`toy_joint_carry_cpa.py`는 고정된 Goal360 정책으로 공동 AT/ON_TOP을 각각 loco에서20초 실행한다. 실제 지급된 CPA와 재계산 값을 비교하고, 같은 상태·행동에서 CPA 보상만 제거한 결과를 별도 계산한다. 정책/관찰 통계·물리 충돌은 바꾸지 않으며 첫 종료 이후 표본은 제외한다. 접근 구간은 미집기 에이전트의 상자 방향 world root 속도>0.15m/s, 근처는 root–상자 XY<1.5m, 정지는 root XY 속도<0.1m/s다.

```bash
TOKENHSI_GPU=5 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; PYTHONPATH=tokenhsi python tokenhsi/tests/toy_joint_carry_cpa.py \
  --task HumanoidMACarry \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --checkpoint output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_task_embedding/ApproachStage2JointCarryMixed80LocoAmpCa4Goal360TaskEmbedding_11-14-48-50/nn/ApproachStage2JointCarryMixed80LocoAmpCa4Goal360TaskEmbedding_00013000.pth \
  --num_envs 128 --num_agents 2 --num_objects 4 --task_graph joint_carry_at \
  --eval_skills loco --eval_skill_probs 1.0 --episode_length 600 --eval_repeats 1 \
  --output_path output/joint_carry_cpa_toy_20261012 --headless --no_video --test --eval --seed 4242'
```

2026-10-12: 각128환경에서 공동/개인 HOLDING 달성0. 접근 CPA 평균 크기 AT0.0146·ON_TOP0.0179, 최대0.372·0.403이며 HOLDING state+progress 초과 비율5.01·5.67%다. 근처 정지 시 progress약0.2·state약0.0005·CPA거의0으로 머무는 현상도 관찰했다. [보고서](../output_etc/joint_carry_cpa_toy_20261012/report.json)와 과제별 NPZ에 기록했다. 계산/성분 합·모델/RMS 불변·finite를 확인했으며 **CPA 제거 재학습이나 원인 확정 실험은 아니다**.

<a id="stage2-goal360-fullpickup"></a>
## Mixed80 LocoAMP CA4 Goal360 FullPickup · CA-only Actor/Critic

`approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_task_embedding`은 기본 Goal360에서 다음 두 부분을 바꾼 별도 실험이다.

- Actor: Stage1 encoder 동결 → CA4 → 64입력 action head. Critic: Stage1 critic encoder → 독립 CA4 → 64입력 value head. Critic은 encoder/task embedding/CA/head 모두 학습한다. 두 branch는 구조만 같고 가중치는 공유하지 않는다.
- `[human, CA]` concat을 제거했다. 각 CA 블록은 human residual을 포함하며 attention output projection·FFN 마지막 층을0으로 초기화해 최초 출력은 Stage1 human과 같다. Stage1 action/value head는 확장 없이 그대로 복사한다. Stage1 critic/AMP weight·관찰 RMS도 전이하고 actor encoder/RMS만 동결한다. Reward/value 통계는 기존 Stage2처럼 새로 시작한다.
- 공동 pickup RSI만 보정100프레임으로 교체한다. `joint_carry/carry_joint_full_pickup/`에서 pickup100/carryWith129/putDown135개를 읽는다. 단독 Stage1 RSI·공동 단계별 샘플링 확률·AMP 원래80/후진10/옆걸음10은 유지한다. 보정 pickup은 RSI용이며 AMP reference를 바꾸지 않는다.

공동80/단독20·상자 크기·Goal360 샘플링과 AT putDown 끝점·보상/CPA·SA/CA 순서 셔플은 기존 Goal360과 같다. Move/Align/TeamHold는 없다. 새 구조와 RSI SHA256을 checkpoint 계약으로 분리했으므로 기존 Stage2 checkpoint를 직접 resume/eval하지 않는다.

학습은 Stage1 epoch30000에서 새 Stage2로 시작한다. GPU는 명령 앞 한 곳에서 지정한다.

```bash
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=6 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_task_embedding \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_task_embedding_train.sh 2 2048 4

# 생성된 checkpoint 경로로 교체
FULLPICKUP_CHECKPOINT="output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_task_embedding/<run>/nn/<checkpoint>.pth"

# 로컬 viewer
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_task_embedding_test.sh "$FULLPICKUP_CHECKPOINT" 2 16 4 1

# 화면 없는 평가
TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_task_embedding_test.sh "$FULLPICKUP_CHECKPOINT" 2 16 4 1

# 서버 VNC
TOKENHSI_GPU=6 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_task_embedding_vnc.sh "$FULLPICKUP_CHECKPOINT" 2 10 4 10
```

`EVAL_SKILLS=pickUp` 또는 `carryWith`로 시작 단계를 지정할 수 있다. 공동 putDown은 `TASK_GRAPH=joint_carry_at EVAL_SKILLS=putDown`이다. 같은 실험 재개만 `RESUME_CHECKPOINT="$FULLPICKUP_CHECKPOINT"`를 사용한다. 확인용 학습은 `MAX_ITERATIONS=2`·별도 `OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_task_embedding_check`로 실행한다.

검증: CPU48개 통과·기존 공동 ON_TOP putDown1개 skip. GPU6·2048환경에서 Stage1 30000 전이→epoch3 저장→epoch4 재개, Actor69개 tensor/RMS 보존·Critic encoder/양쪽 CA4/head 갱신과133종 scalar finite를 확인했다. 실제 공동1638/단독410, AMP30000개80.09/10.24/9.67%, 부분 reset1024/32/32와 Goal360 모든 지원 단계 검사를 통과했다. 공동 AT/ON_TOP 모두 pickup0~99 전체가 실제 선택되고 pose/velocity가 보존된다. 전체4,480,754개·학습3,954,258개 파라미터다.

저장본16환경32-step pickup 시작 혼합 headless 평가도 정상 종료했다. 결과: [전이/재개](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_task_embedding_check/saved_check.json), [RSI/AMP reset](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_task_embedding_check/reset_check.json), [Goal360/전체 pickup](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_task_embedding_check/goal360_runtime_check.json). 본학습·실제 VNC 화면·장기 공동집기/운반 개선은 미검증이다.

<a id="stage2-fullpickup-cparelease1"></a>
## FullPickup 공동 상대 CPA Release1

`approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding`은 FullPickup에 `relationReward.joint_carry.collision.release_buffer: 1.0`만 추가한 별도 env/train·train/test/VNC 실험이다. 같은 carry task·payload·target을 공유하는 두 사람 모두 운반 상자 중심에서 XY 반대각선+1m 안이면 두 사람 사이 CPA를0으로 한다. 현재52×80cm 상자는 반경약1.477m다. HOLDING 성공 전부터 적용하며 한 명이라도 밖으로 나가면 기존 CPA가 다시 적용된다. 단독·다른 payload/goal/task에는 적용하지 않는다.

CPA 계수0.5·예측 안전거리0.7m·할인0.99와 물리 충돌은 유지한다. 기존 HOLDING progress 반경(반대각선+0.2m)도 유지한다. Actor/Critic CA4→64입력 head·Stage1 전이/actor encoder/RMS 동결·Critic 전체 학습·pickup100/carryWith129/putDown135 RSI·AMP80/10/10·Goal360·공동80/단독20·SA/CA/edge 셔플은 FullPickup과 같다. Move/Align/TeamHold는 추가하지 않는다.

Reward metadata가 달라 기존 FullPickup checkpoint의 직접 resume/eval은 거부한다. Stage1에서 새 Stage2로 시작하고, 이후에는 이 실험 checkpoint만 재개한다.

```bash
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=5 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding_train.sh 2 2048 4

CPA_RELEASE_CHECKPOINT="output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding/<run>/nn/<checkpoint>.pth"

# 로컬 viewer
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding_test.sh "$CPA_RELEASE_CHECKPOINT" 2 16 4 1

# 화면 없는 평가
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding_test.sh "$CPA_RELEASE_CHECKPOINT" 2 16 4 1

# 서버 VNC
TOKENHSI_GPU=5 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding_vnc.sh "$CPA_RELEASE_CHECKPOINT" 2 10 4 10
```

Viewer 시작 단계는 `EVAL_SKILLS=loco`(기본)·`pickUp`·`carryWith`로 지정한다. 공동 putDown은 `TASK_GRAPH=joint_carry_at EVAL_SKILLS=putDown`이다. 같은 실험 재개는 `RESUME_CHECKPOINT="$CPA_RELEASE_CHECKPOINT"`를 train에 전달한다. 짧은 확인은 별도 `OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding_check`와 `MAX_ITERATIONS=1`을 사용한다.

검증: CPU43개·Python/셸 문법, GPU5·2048환경 Stage1 30000 전이→epoch2 저장→epoch3 재개,16환경32-step pickup 시작 headless 평가 통과. Actor69개 tensor/RMS 동결·Critic encoder/양쪽 CA4/head 갱신·133종 scalar finite를 확인했다. 별도64환경64-step에서 공동 AT839/ON_TOP843 scene-step의 비영 CPA 제거, 단독832 scene-step CPA 불변과 기존 task 보상 성분 일치를 확인했다. 결과는 [전이/재개](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding_check/saved_check.json), [CPA 실제 적용](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding_check/cpa_release_runtime_check.json)이다. 본학습·실제 VNC 화면·공동집기 개선은 미검증이다.

<a id="joint-carry-mixed-scene-demo"></a>
<a id="locoplatform-ca-solo-mask"></a>
## LocoPlatform 공동 query의 단독 task 차단 · 추론 토이

기본 Platform22000 checkpoint·GPU5·6명·seed42·최대300-step(10초)·10회다. 공동 AT 두 팀은 carryWith, 단독 AT/ON_TOP은 loco로 시작한다. `SOLO_MASK=1`은 Actor CA4에서 공동 담당 query가 단독 task를 보는 경로만 차단하고 나머지 확률을 재정규화한다. 다른 공동 팀의 NONE task는 유지하며, 단독 query·SA·Critic·학습에는 적용하지 않는다. 관계 행렬의 COUPLED 여부로 query/task를 구분하므로 순서 셔플과 무관하다. 기본 checkpoint를 바꾸려면 마지막 인자로 PTH를 지정한다.

```bash
# 서버 VNC: 기존 / 단독 task 차단 (하나씩 실행)
TOKENHSI_GPU=5 SOLO_MASK=0 bash tokenhsi/scripts/multi_agent/locoplatform_ca_solo_mask_vnc.sh
TOKENHSI_GPU=5 SOLO_MASK=1 bash tokenhsi/scripts/multi_agent/locoplatform_ca_solo_mask_vnc.sh

# 로컬 viewer / 화면 없는 짧은 확인
TOKENHSI_GPU=5 SOLO_MASK=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/locoplatform_ca_solo_mask_test.sh
TOKENHSI_GPU=5 SOLO_MASK=1 HEADLESS=1 REPEATS=1 EPISODE_LENGTH=32 bash tokenhsi/scripts/multi_agent/locoplatform_ca_solo_mask_test.sh
```

출력은 `output/locoplatform_ca_solo_mask/{baseline,masked}`로 분리한다. 전체 NONE 차단 토이/학습과 다르며 학습 스크립트는 없다. CPU4개(토이와 동일한 마스크·단독 query 불변·query/task 순열·기존 NONE 회귀), GPU5 baseline headless/차단 실제VNC 각각31-step timeout 종료·동일 초기 장면·6명 화면을 확인했다. [뷰어 검증](../output_etc/locoplatform_solo_mask_viewer_check_20261015/).

선행16장면·32팀·10초 비교에서 holding 유지 시간은62.0→94.2%, 끝 XY 거리는0.386→0.181m였지만 배치 성공 조건은 양쪽0/32였다. [토이 결과](../output_etc/platform_solo_task_mask_toy_20261015/analysis.json). 1환경 viewer는 배치16환경 토이의 개별 장면을 그대로 재생하는 것은 아니며, baseline/차단 첫 초기 장면은 동일하다.

## 공동/단독 혼합 장면 VNC 데모

CPA Release1 FullPickup의 기존 env/train YAML과 checkpoint를 재사용하는 **추론 전용** 경로다. 학습 config·보상·데이터·checkpoint는 변경하지 않는다. 기본 한 환경에6명·8물체가 있고(`DEMO_AGENTS=4`로4명), 전체 사람/8물체/각 사람의 task packet을 같은 정책에 입력한다. 두 사람씩 묶은 정책을 따로 호출하는 방식은 아니다. 기존 metadata·AMP/RSI/Goal360 계약과 model/RMS tensor를 그대로 검사·로드한다.

Checkpoint의 `joint_carry_loco_platform`과 `joint_carry_ablation.policy`로 일반 CPA Release1, LocoPlatform, Geo, NONE-mask env/train YAML을 자동 선택한다. 파일명과 무관하며 기존 호환성 검사는 유지한다. 아래 명령의 `CKPT`만 교체하면 로컬/headless/VNC 데모에서 사용할 수 있다. 장면은 기존 공동 carryWith·단독 loco 시작을 유지하며 학습의 공동 loco 받침 샘플링을 재현하지 않는다. NONE-mask 모델은 자체 config의 NONE 차단을 유지한다.

```bash
# 세 모델 각각의 최신 기본 checkpoint로 6명 VNC (하나씩 실행)
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/joint_carry_mixed_scene_demo_vnc.sh output/*locoplatform_task_embedding/*/nn/*TaskEmbedding.pth
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/joint_carry_mixed_scene_demo_vnc.sh output/*locoplatform_geometry_task_embedding/*/nn/*TaskEmbedding.pth
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/joint_carry_mixed_scene_demo_vnc.sh output/*locoplatform_nonemask_task_embedding/*/nn/*TaskEmbedding.pth
```

같은 실험에 본학습 run이 여러 개 있으면 wildcard 대신 원하는 PTH 경로를 지정한다. 로컬 viewer는 같은 PTH로 `HEADLESS=0 ..._test.sh`, 화면 없는 평가는 `HEADLESS=1 ..._test.sh`를 사용한다.

2026-10-14 자동 선택 검증: CPU11개 통과. GPU5에서 LocoPlatform checkpoint headless/VNC·일반 CPA Release1 epoch19000 CA Entmax를 각각2회·63-step 실행해 로드/reset/finite를 확인했다. [결과·화면·로그](../output_etc/joint_carry_demo_checkpoint_config_check_20261014/).

- H0/H1 → O0 → G0, H2/H3 → O1 → G2: 공동 AT 두 팀, 독립 샘플링한 paired carryWith 자세·속도로 시작.
- H4 → O2 → G4: 단독 AT loco; H5 → O3 → O4: 단독 ON_TOP loco. 단독 상자40cm 큐브·받침52×80×30cm, 상자까지1.5m 앞에서 시작.
- O5/O6/O7은 주변 회색 상자. 네 과제는2.8m 간격으로 배치하고 목표는+X로2.5m 떨어져 있다. 사람6색·공동 상자/AT 목표 노란색·단독 대상 담당 사람 색. 초기 사선 시점 이후 카메라는 자유롭게 조정할 수 있다.

기본600-step(약20초)·10회 재생하며 넘어지면 다음 장면을 시작한다. 마지막 장면은 `KEEP_OPEN=1`(기본)로 유지한다. `REPEATS`, `EPISODE_LENGTH`, `SEED`, `PORT`, `OUTPUT_PATH`를 지정할 수 있고 `SAVE_FRAMES=1`은 일부 VNC 화면 PNG를 저장한다. `demo_results.json`은 시작/끝 위치·RSI 프레임·재생 종료 사유를 남기며 성공률 평가는 하지 않는다. 새 학습용 config가 아니므로 train wrapper는 없다.

`DEMO_AGENTS=4`는 두 번째 공동 팀을 제외한다. H0/H1→O0→G0 공동 AT carryWith, H2→O2→G2 단독 AT loco, H3→O3→O4 단독 ON_TOP loco다. 물체8개·기존 경로 위치를 유지하고 O1은 바닥의 주변 상자로 남긴다. 4명 전체를 한 번에 추론하며 attention은 기존softmax다. 기본 output은 `output/joint_carry_mixed_scene_demo_4agents`; 원래6명은 옵션 생략 시 그대로다.

```bash
PLATFORM_CKPT=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding/ApproachStage2JointCarryMixed80LocoAmpCa4Goal360FullPickupCpaRelease1LocoPlatformTaskEmbedding_13-16-32-42/nn/ApproachStage2JointCarryMixed80LocoAmpCa4Goal360FullPickupCpaRelease1LocoPlatformTaskEmbedding.pth

# 4명 서버 VNC
TOKENHSI_GPU=5 DEMO_AGENTS=4 PORT=6081 REPEATS=10 \
bash tokenhsi/scripts/multi_agent/joint_carry_mixed_scene_demo_vnc.sh "$PLATFORM_CKPT"

# 4명 로컬 viewer
TOKENHSI_GPU=5 DEMO_AGENTS=4 HEADLESS=0 REPEATS=10 \
bash tokenhsi/scripts/multi_agent/joint_carry_mixed_scene_demo_test.sh "$PLATFORM_CKPT"
```

4명 검증: CPU8개·GPU5 사용자LocoPlatform checkpoint headless2회×63-step finite/reset·실제VNC 시작 화면을 확인했다. [추론 기록](../output_etc/joint_carry_mixed_scene_demo_4agents_check_20261014/headless/demo_results.json)·[4명 시작 화면](../output/joint_carry_mixed_scene_demo_4agents/repeat_0_frame_0.png). 운반 완수율은 미검증이다.

저장소 루트에서 실행한다. 아래19000은 검증에 사용한 고정 checkpoint이며 같은 CPA Release1 실험의 다른 checkpoint로 교체할 수 있다.

```bash
CKPT=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_task_embedding/ApproachStage2JointCarryMixed80LocoAmpCa4Goal360FullPickupCpaRelease1TaskEmbedding_12-20-50-28/nn/ApproachStage2JointCarryMixed80LocoAmpCa4Goal360FullPickupCpaRelease1TaskEmbedding_00019000.pth

# 서버 VNC: 접속 주소는 wrapper가 출력한다. 사용 중이면 PORT 변경.
TOKENHSI_GPU=5 PORT=6094 REPEATS=10 \
bash tokenhsi/scripts/multi_agent/joint_carry_mixed_scene_demo_vnc.sh "$CKPT"

# 로컬 viewer
TOKENHSI_GPU=5 HEADLESS=0 REPEATS=10 \
bash tokenhsi/scripts/multi_agent/joint_carry_mixed_scene_demo_test.sh "$CKPT"

# 화면 없는 짧은 추론 확인
TOKENHSI_GPU=5 HEADLESS=1 REPEATS=2 EPISODE_LENGTH=64 \
OUTPUT_PATH=output/joint_carry_mixed_scene_demo_check \
bash tokenhsi/scripts/multi_agent/joint_carry_mixed_scene_demo_test.sh "$CKPT"
```

검증: CPU12개·GPU5 epoch19000 headless64-step2회 reset/finite 추론 및 실제 VNC3회를 확인했다. VNC191/599/54-step은 넘어짐/시간 만료/넘어짐으로 끝났으며6명 운반 완수율은 미검증이다. [시작 화면](../output_etc/joint_carry_mixed_scene_demo_vnc_check_20261014/repeat_0_frame_0.png)·[재생 기록](../output_etc/joint_carry_mixed_scene_demo_vnc_check_20261014/demo_results.json).

<a id="joint-carry-ca-entmax-viewer"></a>
## 6명 공동/단독 혼합 장면 · CA Entmax 뷰어

기존 혼합 장면의6명·8물체·1환경, 공동 AT 두 팀 carryWith·단독 AT/ON_TOP loco·자유 카메라·최대20초/10회 반복을 재사용한다. 전용 wrapper는 `DEMO_CA_ENTMAX=1`로 **Actor CA4만 entmax(α=1.5)**를 사용한다. Actor SA·Critic·checkpoint/model/RMS는 기존 설정이며 기존 wrapper 기본값은 softmax다. Entmax 연산은 `learning/multi_agent/ca_entmax_inference.py`에 있어 토이 산출물 폴더에 의존하지 않는다.

Checkpoint 인자는 생략할 수 있으며 토이와 같은 CPA Release1 epoch19000을 사용한다. CPA Release1 또는 LocoPlatform checkpoint는 마지막 인자로 지정한다. 기본 output은 `output/joint_carry_mixed_scene_demo_ca_entmax`이며 촬영 이미지는 그 아래 `imgs/`, `demo_results.json`에 `actor_ca_attention: entmax15`를 기록한다. 새 학습 config가 아닌 추론 옵션이므로 train wrapper는 없다.

저장소 루트에서:

```bash
# 서버 VNC; 접속 주소는 터미널 출력
TOKENHSI_GPU=4 PORT=6095 REPEATS=10 \
bash tokenhsi/scripts/multi_agent/joint_carry_mixed_scene_demo_ca_entmax_vnc.sh

# 로컬 viewer
TOKENHSI_GPU=4 HEADLESS=0 REPEATS=10 \
bash tokenhsi/scripts/multi_agent/joint_carry_mixed_scene_demo_ca_entmax_test.sh

# 짧은 화면 없는 추론 확인
TOKENHSI_GPU=4 HEADLESS=1 REPEATS=2 EPISODE_LENGTH=64 \
OUTPUT_PATH=output/joint_carry_mixed_scene_demo_ca_entmax_check \
bash tokenhsi/scripts/multi_agent/joint_carry_mixed_scene_demo_ca_entmax_test.sh
```

검증: CPU7개·토이 CA 출력 최대오차0·GPU4 epoch19000 실제 VNC2회64-step의 entmax 적용/finite/reset을 확인했다. 결과는 `output_etc/joint_carry_mixed_scene_demo_ca_entmax_check_20261014/`이다.

<a id="entmax-inference-toy"></a>
## 추론 전용 Entmax1.5 attention 비교

[실험 코드/산출물](../output_etc/entmax_inference_toy_20261014/)은 CPA Release1 epoch19000의 Actor SA·CA 정규화만 실행 중 교체한다. Critic·모델 파라미터·scene/AMP RMS·학습/기존 viewer는 유지한다. Entmax 패키지를 설치하지 않고 α=1.5의 정렬 해법을 사용하며 독립 이분법으로 검증했다([공식 구현](https://github.com/deep-spin/entmax/blob/master/entmax/activations.py)). CA의 수동 Q/K/V 계산은 기존 softmax 전체 행동 출력과 최대3e−7 이내로 일치한다.

평가 환경은 각각16개다. 2명은 기존 공동 AT·carryWith·Goal360,6명은 기존 혼합 데모 배치를 평가 실행 안에서만 복제한다(각 장면6명·8물체, 공동 AT 두 팀 carryWith·단독 AT/ON_TOP loco). 조건별 초기 관측/root/DOF/상자/목표 해시는 동일하며 최대600-step/20초·첫 넘어짐 이후는 지표에서 제외한다. 2명/6명 사이 초기 장면 분포는 다르므로 성능은 각 설정 안에서 비교한다. 6명 두 공동 팀은 같은 장면에 묶어 대응 bootstrap을 계산한다.

6명 결과: softmax/SA만/CA만/둘 다의 넘어짐11/5/2/7개(각16장면), 공동 마지막 유효 XY 오차1.345/1.495/0.640/1.589m. CA의 넘어짐 개선은9개/악화0개, 대응 McNemar의3개 조건 보정 p=0.0117이다. 같은 경과 시간 오차 차이는−0.344m·95% CI[−0.780,+0.049]로0을 포함하며, 마지막 오차 개선에는 더 오래 살아 있는 효과가 섞인다. 기존 배치 조건 충족은 전부0/32이며 손 해제/운반 완수 평가가 아니다.

CA만 교체한6명에서 CA 정확한0 비율은53%지만 다른 task의 NONE mass는78.8→79.7%이며 같은 상태2명/6명 공동 행동 차이도0.0346→0.0382로 줄지 않았다. 다른 팀 정보 차단이 개선의 원인이라고 결론내릴 수 없다. 2명 CA는 넘어짐0→1/16·오차0.109→0.435m였으므로 기본 viewer에 일괄 적용하지 않았다. 표본16장면·단일 checkpoint·α=1.5·재학습 없음이라는 한계가 있다.

[전체 결과](../output_etc/entmax_inference_toy_20261014/analysis.json)·[재생 성능 그래프](../output_etc/entmax_inference_toy_20261014/rollout_comparison.png)·[attention 그래프](../output_etc/entmax_inference_toy_20261014/attention_comparison.png)·[대응 차이/95% CI](../output_etc/entmax_inference_toy_20261014/paired_effects.png). `results_{2,6}.json`은 동결·초기 상태·조건별 결과, `agents*_trial0_*.npz`는 원시 궤적이다. `repeat/`의 새 프로세스 softmax 반복은 root/상자 궤적 오차0·넘어짐/배치 결과 동일했고,6명2장면의 종료 검출만1-step 차이가 있었다.

저장소 루트에서 재현(기존 결과를 보존하려면 별도 `TOY_OUTPUT` 지정):

```bash
TOKENHSI_GPU=4 TOY_OUTPUT=output_etc/entmax_inference_toy_20261014_repro \
bash output_etc/entmax_inference_toy_20261014/run.sh

# 같은 조건의 softmax만 새 프로세스에서 반복
TOKENHSI_GPU=4 TOY_VARIANTS=softmax \
TOY_OUTPUT=output_etc/entmax_inference_toy_20261014_repro/repeat \
bash output_etc/entmax_inference_toy_20261014/run.sh

# 분석/그래프
TOKENHSI_GPU=4 bash -c '
source tokenhsi/scripts/multi_agent/runtime_env.sh
python output_etc/entmax_inference_toy_20261014/report.py output_etc/entmax_inference_toy_20261014_repro
'
```

`TOY_SAMPLES`(기본16), `TOY_STEPS`(기본600), `TOY_AGENTS=2` 또는`6`으로 검사 범위를 지정할 수 있다. 전용 학습 config/train/VNC wrapper가 아닌 추론 토이다.

**동일 상태 attention map:** [CA 마지막 layer/head](../output_etc/entmax_inference_toy_20261014/maps/ca_last_layer_scene0.png)·[CA 모든 layer/head](../output_etc/entmax_inference_toy_20261014/maps/ca_all_layers_scene0.png)·[SA 마지막 layer/head](../output_etc/entmax_inference_toy_20261014/maps/sa_last_layer_scene0.png)·[16장면 CA category 평균](../output_etc/entmax_inference_toy_20261014/maps/ca_categories_16scenes.png). 기존 토이와 해시가 같은 초기16장면에서 같은H0/H1 공동팀의2명 관측과6명 softmax/CA entmax를 비교한다. 2명 칼럼은H0/H1·O0/O5/O7/O6·G0/G1을 재매핑한 관측이며 별도2명 rollout은 아니다. Token/task 순서는 map 표시를 위해 canonical로 고정했고 행동 오차는≤2.4e−7,model/scene RMS 불변이다. 6명 두 조건의SA map은 동일하다.

해당 팀·16초기 장면·전체 CA layer/head 평균 자기/파트너/다른 task mass는2명31.3/68.7/0%,6명 softmax3.4/12.7/83.9%,6명 entmax0.7/13.2/86.0%다. 앞의 전체6명·유효 재생 시간 평균과 집계 범위가 다르다. Scene0 마지막 head2에서H0→파트너는95.1→12.6→1.4%, entmax의단독 AT/ON_TOP은67.6/31.0%다. Sparse해져도 자기 팀을 선택한다는 보장은 없으며 가중치는 최종 행동 기여율이 아니다. 실제 각 forward를 기록했으므로 후반CA layer에는 앞선layer 변경 효과도 포함된다.

전체 query 그림: [CA 사람6×task6](../output_etc/entmax_inference_toy_20261014/maps/ca_last_layer_full_scene0.png)·[SA 전체20×20](../output_etc/entmax_inference_toy_20261014/maps/sa_last_layer_full_scene0.png). 위의 기존 그림은 비교 대상 H0/H1 query 행만 표시했다. 실제6명 입력에는 사람 토큰6개가 있으며 SA의 나머지 토큰은 물체8개·목표6개다.

단독2명(loco AT·ON_TOP): [2명 CA 마지막layer](../output_etc/entmax_inference_toy_20261014/solo_maps/ca_two_solo_last_scene0.png)·[2명/6명 비교](../output_etc/entmax_inference_toy_20261014/solo_maps/ca_last_layer_scene0.png)·[CA 모든layer](../output_etc/entmax_inference_toy_20261014/solo_maps/ca_all_layers_scene0.png)·[SA](../output_etc/entmax_inference_toy_20261014/solo_maps/sa_last_layer_scene0.png). 동일 초기H4/H5·O2/O3/O5/O4·G4/G5 입력을2명으로 재매핑하며 별도2명 물리 rollout은 아니다. 16장면·전체CA 평균 자기/다른 단독task softmax48.1/51.9%, entmax47.4/52.6%. H0 기준T4/T5는정상NONE관계이나 마지막head2 QK 점수가파트너보다커서 유한관계bias를넘는다. 점수분해는 `solo_maps/analysis.json`. GPU4 runtime 설정 후 `MAP_VIEW=solo python output_etc/entmax_inference_toy_20261014/map_collect.py`, `python output_etc/entmax_inference_toy_20261014/solo_map_plot.py`로 재현한다.

재현(위토이16장면 결과 필요): `TOKENHSI_GPU=4`로 runtime 환경을 준비한 뒤 `python output_etc/entmax_inference_toy_20261014/map_collect.py`, `python output_etc/entmax_inference_toy_20261014/map_plot.py`를 차례로 실행한다. 원시확률/logit/초기관측은 `maps/maps.npz`, 검증/집계는 `maps/maps_summary.json`·`maps/category_means.json`이다.

<a id="stage2-locoplatform-human-sa"></a>
## LocoPlatform human SA4

`approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding`는 LocoPlatform의 Stage2 task CA만 human self-attention으로 바꾸는 별도 실험이다. Stage1 SA 최종 human64만 추출하고, Actor/Critic 각각 SA+residual+FFN+LayerNorm 4층·2head·FFN128을 통과시켜64입력 action/value head에 넣는다. 매 층 갱신된 human을 정규화해 Q/K/V 모두에 사용한다. task 토큰·grounding MLP는 없으며 packet은 SELF(자기), COUPLED(동일 carry task/payload/target의 상대), NONE(그 외) 관계 판정에만 사용한다. 세 learnable head별 bias를 모두 더하고 NONE을 차단하지 않는다.

Stage1 entity/token 셔플을 유지하며 Stage2는 human과 관계 행·열을 함께 섞고 출력 순서를 복원한다. packet 순서에는 무관하다. Actor encoder/scene RMS 동결·새 SA 및 action head 학습, Critic encoder/SA/value head 전체 학습이다. 기존 Stage1 전이 규칙으로 encoder/head를 초기화하고 새 SA residual 추가분을0으로 초기화해 초기 행동·가치를 보존한다. `head_input: ca_only`, `critic_ca: true`는 기존64입력/양쪽 coordination 설정 키이며 이 실험에서는 실제 모듈이 human SA다.

환경은 기존 LocoPlatform과 동일하다: 공동80/단독20, AMP 원본80/후진10/옆걸음10, paired full pickup RSI·Goal360·CPA release+1m·공동 loco 받침·보상·물체 크기 유지. `human_sa4_actor_critic` 정책 계약으로 기존 CA/Geo/NONE checkpoint와 resume/eval 혼용을 차단한다. 전용 test는 추론용 NONE override를 제거한다. 기본GPU5이며 장기 학습은 직접 실행한다.

```bash
# Stage1 전이로 새 Stage2 학습 (기존 Stage2 resume 아님)
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=5 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_train.sh 2 2048 4

HUMAN_SA_CHECKPOINT="output/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding/<run>/nn/<checkpoint>.pth"

# 로컬 viewer; 화면 없는 평가는 HEADLESS=1
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_test.sh "$HUMAN_SA_CHECKPOINT" 2 16 4 1

# 서버 VNC
TOKENHSI_GPU=5 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_vnc.sh "$HUMAN_SA_CHECKPOINT" 2 1 4 10
```

짧은 검증은 `OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_check MAX_ITERATIONS=2`로 분리한다. 학습2048환경, 평가 환경 수는 별도 지정한다.

검증(2026-10-15): CPU10개 통과. GPU5·2048환경 epoch3 저장본에서 Stage1 Actor encoder/scene RMS 동결·새 SA/세 bias/head/Critic 갱신을 확인했다. AMP79.83/10.03/10.14%, 공동1638/단독410·full pickup/paired RSI·Goal360·부분 reset·짧은 physics 통과. 2명16환경 및6명1환경 headless 추론 정상 종료. [저장본 검사](../output/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_check/saved_check.json)·[AMP/reset](../output/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_check/reset_check.json)·[RSI/목표](../output/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_check/goal360_runtime_check.json). 본학습 성능·실제 VNC 화면은 미검증이다.

기존6명 혼합 뷰어도 checkpoint metadata로 이 SA config를 자동 선택한다(공동 두 팀 carryWith + 단독 AT/ON_TOP loco):

```bash
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/joint_carry_mixed_scene_demo_vnc.sh "$HUMAN_SA_CHECKPOINT"
```

<a id="stage2-locoplatform-nonemask"></a>
## LocoPlatform NONE 차단 학습

`approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_nonemask_task_embedding`는 기존 LocoPlatform 기반의 **별도 학습 실험**이다. Geometry 실험이 아니며 task 입력은 원래 `[행동64; SA human64; SA payload64; SA target64]`256→128→64다. `coordination.mask_none: true`가 Actor/Critic의 CA4 전 층에서 NONE logit을 −∞로 만든다. SELF/COUPLED에는 원래 QK와 학습 bias를 사용한다. 단독은 SELF100%, 공동은 SELF/COUPLED만 볼 수 있으며 유효한 관련 task가 없는 query는 residual human을 유지한다. 마스크는 packet relation을 따라가므로 SA/edge/task 순서 셔플과 함께 재정렬된다. 마스크는 한 forward에서 한 번 만들고4층이 재사용한다. NONE bias 파라미터는 shape 호환상 남지만 gradient0이므로 초기0으로 유지된다.

Scratch는 Stage2 checkpoint resume 없이 Stage1 CarryDistill30000에서 encoder/head·기존 전이 항목을 초기화하고 새 CA를 학습한다는 뜻이다. Actor encoder 동결·CA-only64입력 action head, Critic 전체 학습, 공동80/단독20, AMP80/10/10, full pickup RSI, Goal360, CPA release+1m, loco 받침, 보상·물체 크기는 LocoPlatform과 같다. 정책 계약 `task_ca4_actor_critic_none_mask`로 기존 Platform/Geo와 resume/eval 혼용을 차단한다. **추론 토이의 `NONE_MASK=0`으로 이 학습 모델의 마스크를 끌 수 없다.** 전용 wrapper는 추론 override 환경변수를 제거하고 학습 YAML의 구조를 사용한다.

```bash
# GPU5: Stage1 전이, 새 Stage2 본학습
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=5 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_nonemask_task_embedding \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_nonemask_task_embedding_train.sh 2 2048 4

NONE_MASK_CHECKPOINT="output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_nonemask_task_embedding/<run>/nn/<checkpoint>.pth"

# 로컬 viewer: 화면 없는 평가는 HEADLESS=1
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_nonemask_task_embedding_test.sh "$NONE_MASK_CHECKPOINT" 2 16 4 1

# 서버 VNC
TOKENHSI_GPU=5 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_nonemask_task_embedding_vnc.sh "$NONE_MASK_CHECKPOINT" 2 10 4 10
```

짧은 검증은 별도 `OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_nonemask_task_embedding_check`와 `MAX_ITERATIONS=2`를 사용한다. 실제 학습은2048환경이며 viewer 환경 수는 별도 지정한다. 위 전용 VNC는 원래2명 평가 장면이며 기존6명 추론 토이 wrapper와 구분한다.

검증(2026-10-15): CPU12개·GPU5 2048환경 Stage1 전이→epoch3 저장, Actor encoder69개/scene RMS 동결·Critic/head 갱신·NONE bias0·399개 scalar finite. AMP 실측79.83/10.03/10.14%, 공동1638/단독410·paired RSI/Goal360·부분 reset 유지. 저장본16환경64-step 설정 전용test 정상 종료. 재로드 Actor/Critic8개 block에서 NONE0·단독 SELF100%, 계측 행동 오차≤2.2e−7 확인. [저장본 검사](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_nonemask_task_embedding_check/saved_check.json)·[AMP/reset](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_nonemask_task_embedding_check/reset_check.json)·[RSI/목표](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_nonemask_task_embedding_check/goal360_runtime_check.json). 본학습 성능·실제 VNC 화면은 미검증.

<a id="locoplatform-ca-none-mask"></a>
## LocoPlatform NONE 차단 · 추론 전용

[설정](../tokenhsi/data/cfg/inference/locoplatform_ca_none_mask.yaml)의 `actor_ca_none_mask: true`는 Actor CA4의 softmax 직전 NONE logit만 −∞로 만든다. SELF/COUPLED QK·bias와 residual·FFN, Stage1 SA, Critic, checkpoint, 기존 RSI/AMP 설정은 유지한다. 단독은 SELF100%, 공동은 SELF/COUPLED 안에서 기존 점수로 attention을 계산한다. 유효한 관련 task가 없는 query는 해당 block의 human 입력을 그대로 반환한다. 기존 player의 기본 동작과 학습 경로에는 적용하지 않는다. 별도 학습 config/train wrapper는 없는 추론 토이다.

기본 checkpoint는 기존 분석에 사용한 **LocoPlatform12500**이며 첫 인자로 다른 LocoPlatform checkpoint를 지정할 수 있다. Geo checkpoint는 이 wrapper 대상이 아니다. 기본 `SCENE=mixed`는 기존6명 데모(공동 AT 두 팀 carryWith + 단독 AT/ON_TOP 두 명 loco,8물체,1환경)를 재사용한다. `DEMO_AGENTS=4`는 공동 한 팀+단독 두 명이다. 기존 데모의 고정 목표 배치를 사용하므로 학습 Goal360 분포 평가와 구분한다. `SCENE=two_agents`는 원래 LocoPlatform 평가 환경/Goal360를 사용하며 기본16환경·random_scenario·carryWith다.

```bash
# 서버 VNC: NONE만 차단, GPU5, 기본6명
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/locoplatform_ca_none_mask_vnc.sh

# 같은 checkpoint/seed/장면의 기존 baseline
TOKENHSI_GPU=5 NONE_MASK=0 bash tokenhsi/scripts/multi_agent/locoplatform_ca_none_mask_vnc.sh

# 로컬 viewer / 화면 없는 짧은 확인
TOKENHSI_GPU=5 HEADLESS=0 bash tokenhsi/scripts/multi_agent/locoplatform_ca_none_mask_test.sh
TOKENHSI_GPU=5 HEADLESS=1 REPEATS=2 EPISODE_LENGTH=64 \
bash tokenhsi/scripts/multi_agent/locoplatform_ca_none_mask_test.sh

# 2명 혼합 과제: 단독 NONE 차단 + 공동 원본 동작 확인
TOKENHSI_GPU=5 SCENE=two_agents NUM_ENVS=16 EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/locoplatform_ca_none_mask_vnc.sh
```

기본 output은 `output/locoplatform_ca_none_mask/<mixed|two_agents>/<masked|baseline>`이며 `OUTPUT_PATH`로 변경 가능하다. 두 조건은 SEED42/같은 checkpoint를 사용하고 첫 초기 장면을 맞추되, 이후 물리 상태는 행동에 따라 달라진다. `inference_override.json`에 실제 적용 config/checkpoint를 남기고 혼합 데모의 `demo_results.json`에도 mask 여부를 기록한다. `DEMO_CA_ENTMAX`는 wrapper에서0으로 고정하며, 직접 두 기법을 동시에 요청하면 거부한다. 혼합 데모는 성공률 집계가 없는 viewer이며 성능 개선을 보장하지 않는다.

검증(2026-10-15): CPU10개, GPU5의2명16환경/6명1환경32-step에서 NONE0·단독 SELF100%·가중치/RMS 불변. 2명 공동 action 차이≤2.6e−7. 전용6명 mask/baseline 각각2회×63-step 정상 종료·초기 장면 일치,2명16환경 wrapper도64-step 설정 평가 정상 종료. [2명 검사](../output_etc/locoplatform_none_mask_check_20261015/native/mask_check.json)·[6명 검사](../output_etc/locoplatform_none_mask_check_20261015/mixed/mask_check.json). 실제 VNC 화면과 운반 개선 여부는 미검증.

<a id="stage2-locoplatform-geometry"></a>
## LocoPlatform Geometry task CA

`approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_geometry_task_embedding`는 LocoPlatform의 task 토큰 입력만 geometry로 교체한 별도 실험이다. `[행동 embedding64; 담당 사람 env-local pose7; 사람 기준 payload pose7; 사람 기준 target pose7]`를85→128→64 MLP로 인코딩한다. Pose는 `(x,y,z,qx,qy,qz,qw)`이며 기존 GTA 관측을 재사용한다. 사람은 root 위치와 yaw quaternion, 물체는 전체 회전이다. 상대 위치는 `R_Hᵀ(p_X−p_H)`, 상대 회전은 `q_H⁻¹⊗q_X`; quaternion을 정규화하고 w≥0 표현을 사용한다. AT는 task가 참조하는 G, ON_TOP은 받침 물체를 target으로 사용한다. 손/관절·속도·크기는 geometry 입력에 추가하지 않는다. 사람의 환경 좌표가 포함되므로 전체 회전/이동 불변성을 보장하지 않는다.

Actor query는 동결된 Stage1 SA human feature다. task K/V 입력에서만 최종 SA 사람/물체/목표 feature를 제거한다. Critic도 같은 geometry CA 구조를 사용하되 encoder·embedding·CA·value head 전체를 학습한다. CA4·2head·FFN128·CA-only64입력 head와 identity 초기화를 유지한다. SELF/COUPLED/NONE은 기존 learnable bias이며 hard mask를 추가하지 않는다. task 토큰은 한 번 만들고4층에서 재사용한다.

공동80/단독20, paired 전체 pickup RSI·Goal360, 단독 Stage1 크기/RSI, AMP original/backward/sideways80/10/10, CPA release+1m, 공동 loco50% payload 받침, 보상·SA/CA/task 순서 셔플은 LocoPlatform과 같다. `task_ca4_actor_critic_geometry` 계약으로 기존 feature-task checkpoint와 resume/eval 혼용을 차단한다. **Scratch는 Stage2 resume 없이 Stage1 encoder/head를 전이하고 새 geometry MLP·CA를 학습한다는 뜻**이다.

```bash
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=5 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_geometry_task_embedding \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_geometry_task_embedding_train.sh 2 2048 4

GEOMETRY_CHECKPOINT="output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_geometry_task_embedding/<run>/nn/<checkpoint>.pth"

# 로컬 viewer (화면 없는 평가는 HEADLESS=1)
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_geometry_task_embedding_test.sh "$GEOMETRY_CHECKPOINT" 2 16 4 1

# 서버 VNC viewer
TOKENHSI_GPU=5 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_geometry_task_embedding_vnc.sh "$GEOMETRY_CHECKPOINT" 2 10 4 10
```

Stage1 source는 LocoPlatform과 같은 CarryDistill30000을 예시로 썼으며 `STAGE1_CHECKPOINT`로 지정한다. 짧은 확인은 별도 `OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_geometry_task_embedding_check`, `MAX_ITERATIONS=2`를 사용한다. 본학습은 자동 실행하지 않는다.

검증(2026-10-14): CPU18개·최종 wrapper 경로 포함 geometry4개 재확인 통과. GPU5·2048환경 epoch3 저장,85입력 양쪽 MLP·Actor encoder/RMS 동결·Critic/head 갱신·399개 scalar finite 확인. AMP 실측79.83/10.03/10.14%, 공동1638/단독410·paired RSI/Goal360·부분 reset 유지, 저장본16환경64-step 설정 carryWith headless 추론 정상 종료. [학습/동결](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_geometry_task_embedding_check/saved_check.json)·[AMP/reset](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_geometry_task_embedding_check/reset_check.json)·[paired RSI/목표](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_geometry_task_embedding_check/goal360_runtime_check.json). 장기 학습 성능과 실제 VNC 화면은 미검증이다.

<a id="stage2-cparelease1-locoplatform"></a>
## CPA Release1 LocoPlatform · 공동 loco payload 받침

`approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding`은 CPA Release1 기반 별도 env/train·train/test/VNC 실험이다. **공동 AT/ON_TOP의 loco 초기화만** 바꾼다. Loco 중50%는 기존 바닥 시작,50%는 payload 아래40×40×2cm 받침을 둔다. 단독·pickup/carryWith/putDown·목표 ON_TOP 받침·다른 상자는 그대로다. PickupFocus·Move·TeamHold는 추가하지 않았다.

받침 윗면 높이는 `min(max(U(0,1m),0.02m),1.2m−payload 높이)`다. 현재 payload 높이40cm이면2~80cm이며, 높은 분기의 약20%가 상한80cm에 모인다. 안정화 후 payload 윗면은 최대1.2m다. 초기에는 원본처럼 받침 위5cm 간격을 두어 떨어뜨리므로 첫 순간 윗면은 최대1.25m다. 상한은 실제 payload 크기로 계산한다. 받침이 바닥과 겹치지 않도록 높은 분기만 최소2cm로 제한한다.

받침은 기존 고질량·중력 비활성 물리 actor를 재사용하고 사람/상자/marker 다음에 하나 추가한다. 과제 물체·관찰 token에는 포함하지 않는다. reset 시 사람·상자·받침·marker를 한 번에 갱신한다. 공동80/단독20·Goal360·full pickup100프레임·단계 확률·AMP80/10/10·CA4 CA-only Actor/Critic·CPA Release1·보상·SA/CA/edge 셔플은 기반과 같다.

`joint_carry_loco_platform` checkpoint 계약으로 받침 유무가 다른 실험과 직접 resume/eval 혼용을 막는다. Stage1 epoch30000에서 새로 시작하며 같은 실험 checkpoint만 재개한다.

```bash
# 학습: Stage1에서 새로 시작
TOKENHSI_GPU=5 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_train.sh 2 2048 4

LOCO_PLATFORM_CHECKPOINT="output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding/<run>/nn/<checkpoint>.pth"

# 로컬 viewer
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=random_scenario EVAL_SKILLS=loco \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_test.sh "$LOCO_PLATFORM_CHECKPOINT" 2 16 4 1

# 화면 없는 평가
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario EVAL_SKILLS=loco \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_test.sh "$LOCO_PLATFORM_CHECKPOINT" 2 16 4 1

# 서버 VNC viewer
TOKENHSI_GPU=5 TASK_GRAPH=joint_carry_at EVAL_SKILLS=loco \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_vnc.sh "$LOCO_PLATFORM_CHECKPOINT" 2 10 4 10
```

공동 ON_TOP만 보려면 `TASK_GRAPH=joint_carry_ontop`, 혼합은 `random_scenario`다. `EVAL_SKILLS=pickUp`·`carryWith` 및 AT `putDown`에는 시작 받침이 없다. 같은 실험 재개는 `RESUME_CHECKPOINT="$LOCO_PLATFORM_CHECKPOINT"`, 짧은 확인은 별도 `OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_check`와 `MAX_ITERATIONS=1`을 사용한다.

검증(2026-10-13): CPU25개 통과·기존 ON_TOP putDown1개 skip, Python/셸 문법 통과. GPU5·2048환경에서 공동1638/단독410, 공동 AT/ON_TOP loco 받침831/1638개(50.73%)·높이2~80cm, paired RSI·단독·목표·AMP 초기 이력 불변과 부분 reset을 확인했다. AT832/ON_TOP816장면의2초 지지·받침 제거 후 낙하 검사를 통과했다. Stage1 epoch30000 전이→epoch2 저장→epoch3 재개, actor69개 tensor/RMS 동결·critic/양쪽 CA4/head 갱신·133종 scalar finite를 확인했다. 저장본16환경32-step의 loco VNC viewer 실행과 pickup headless 평가도 정상 종료했다. [받침 물리/초기화](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_check/platform_runtime_check.json), [RSI/AMP](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_check/reset_check.json), [전이/재개/평가](../output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_locoplatform_task_embedding_check/saved_check.json). 본학습·공동집기 개선·바닥 시작으로의 전이는 아직 미검증이다.

<a id="stage2-cparelease1-pickupfocus"></a>
## CPA Release1 PickupFocus · 집기/들기 프레임 집중

`approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding`은 CPA Release1에서 공동 pickup 프레임 분포만 바꾼 별도 실험이다. `jointCarryRsi.pickupFrameSampling`의 inclusive source frame 범위25~39에서35%,55~80에서35%,전체0~99에서30%를 균등 샘플링한다. 전체 분기와 집중 구간이 겹치므로 실제 구간별 확률은39.5/42.8/17.7%다. 프레임 번호는0부터이며 두 사람·상자·속도를 같은 paired snapshot에서 읽는다. 가중치는 로드 시 한 번 계산하고 reset마다 한 번의 multinomial로 선택한다.

Epoch에 따른 스케줄은 없다. 공동 AT의 loco/pickUp/carryWith/putDown40/10/40/10%, ON_TOP50/10/40/0%, 단독 Stage1 RSI는 유지한다. Pickup 자체는 여전히10%다. RSI 전체100프레임·Goal360·AMP80/10/10·CA4 Actor/Critic·CPA Release1·리워드·크기·공동80/단독20·SA/CA/edge 셔플은 기준 실험과 동일하다.

새 `joint_carry_pickup_sampling` checkpoint 계약으로 기존 CPA Release1과 직접 resume/eval 혼용을 차단한다. Stage1 전이로 새로 시작하며 같은 실험 checkpoint만 재개한다. Viewer/평가에서도 `EVAL_SKILLS=pickUp`이면 위 분포가 적용된다. Loco/carryWith/putDown 프레임 분포는 기존과 같다.

```bash
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=5 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding_train.sh 2 2048 4

PICKUP_FOCUS_CHECKPOINT="output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding/<run>/nn/<checkpoint>.pth"

# 로컬 viewer
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding_test.sh "$PICKUP_FOCUS_CHECKPOINT" 2 16 4 1

# 화면 없는 평가
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding_test.sh "$PICKUP_FOCUS_CHECKPOINT" 2 16 4 1

# 서버 VNC
TOKENHSI_GPU=5 TASK_GRAPH=random_scenario \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding_vnc.sh "$PICKUP_FOCUS_CHECKPOINT" 2 10 4 10
```

공동 pickup만 보려면 `TASK_GRAPH=joint_carry_at EVAL_SKILLS=pickUp`을 지정한다. 기본 viewer 시작은 loco다. 재개는 `RESUME_CHECKPOINT="$PICKUP_FOCUS_CHECKPOINT"`, 짧은 확인은 별도 `OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding_check`와 `MAX_ITERATIONS=1`을 사용한다.

검증: CPU25개 통과·기존 ON_TOP putDown1개 skip, GPU5·2048환경 reset/AMP/Goal360·부분 reset 검사 통과. 실제 pickup 집기/들기/기타 비율은 AT39.56/42.61/17.83%, ON_TOP39.26/44.63/16.12%였다. Stage1 epoch30000 전이→epoch2 저장→epoch3 재개,16환경32-step pickup headless 평가 통과. Actor69개 tensor/RMS 동결·Critic/양쪽 CA4/head 갱신·133종 scalar finite를 확인했다. [전이/재개/평가](../output_etc/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding_check/saved_check.json), [RSI/AMP](../output_etc/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding_check/reset_check.json), [프레임 분포/Goal360](../output_etc/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_fullpickup_cparelease1_pickupfocus_task_embedding_check/goal360_runtime_check.json). 본학습·실제 VNC 화면·장기 개선은 미검증이다.

<a id="stage2-mixed80-locoamp-ca4-goal360-teamhold"></a>
## Mixed80 LocoAMP CA4 Goal360 공동 HOLDING state

`approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_teamhold_task_embedding`은 Goal360에 `relationReward.joint_carry.holding_state: opposite_anchor_min`만 추가한 별도 env/train·train/test/VNC 실험이다. 두 사람/두 앵커 점수 `s[i,k]=exp(-10*||mean(hands_i)-anchor_k||²)`에서 `c=max(min(s[A,0],s[B,1]),min(s[A,1],s[B,0]))`를 계산해, 공동 환경의 각 HOLDING state를 `0.2*c`로 교체한다. 기존 개인 state에 더하는 보상이 아니며 둘이 잡기 전에도 연속적으로 지급한다.

개인 `phi_raw=max_k(s[i,k])`·HOLDING progress/success·공동 반대 앵커 gate·AT/ON_TOP 전체 보상·페널티·단독 보상은 그대로다. 따라서 로그의 HOLDING `phi_raw`는 개인 점수이고 새 state component의 기준은 `c`다. Move/Align은 없다. Goal360 RSI/목표·putDown 끝점/미사용 상자 clearance·AMP80/10/10·CA4/SA/CA 셔플·Stage1 encoder/RMS 동결도 유지한다. Reward metadata가 달라 기존 Goal360 checkpoint의 직접 resume/eval은 거부하며 새 실험 checkpoint끼리만 재개한다.

현재 Goal360 본학습과 같은 Stage1 epoch30000·seed8429에서 시작하는 비교 명령이다. GPU는 실행 환경에 맞춰 `TOKENHSI_GPU` 한 곳에서 지정한다.

```bash
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=6 SEED=8429 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_teamhold_task_embedding \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_teamhold_task_embedding_train.sh 2 2048 4

# TEAMHOLD_CHECKPOINT는 이 실험에서 저장한 .pth
TOKENHSI_GPU=6 HEADLESS=0 TASK_GRAPH=random_scenario EVAL_SKILLS=loco \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_teamhold_task_embedding_test.sh "$TEAMHOLD_CHECKPOINT" 2 16 4 1

TOKENHSI_GPU=6 HEADLESS=1 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_teamhold_task_embedding_test.sh "$TEAMHOLD_CHECKPOINT" 2 16 4 1

TOKENHSI_GPU=6 TASK_GRAPH=random_scenario EVAL_SKILLS=loco \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_teamhold_task_embedding_vnc.sh "$TEAMHOLD_CHECKPOINT" 2 10 4 10
```

같은 실험 재개는 `RESUME_CHECKPOINT="$TEAMHOLD_CHECKPOINT"`를 train에 지정한다. 확인용은 `MAX_ITERATIONS=1`·별도 `OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_goal360_teamhold_task_embedding_check`를 사용한다.

검증: CPU52개 통과·공동 ON_TOP putDown1개 skip. GPU6·2048환경 Stage1 30000 전이→epoch2 저장→epoch3 재개 및16환경32-step headless 평가를 통과했다. 별도64환경32-step에서 두 사람 공동 state 동일 지급·교체만 적용·개인 progress/success/단독 불변·부분 reset cache를 확인했다. RSI/AMP 초기화, Goal360 단계별 목표/pose/velocity, encoder/RMS 동결과 CA4/head 갱신도 통과했다. 결과는 위 `*_check/{teamhold_runtime_check,reset_check,goal360_runtime_check,saved_check}.json`에 있다. 본학습의 공동집기 개선과 실제 VNC 화면은 아직 검증하지 않았다.

<a id="stage2-mixed80-locoamp-ca4-move"></a>
## Mixed80 LocoAMP CA4 공동 이동 방향 보상

`approach_stage2_joint_carry_mixed80_locoamp_ca4_move_task_embedding`은 CA4에 실제 이동 방향 보상만 추가한다. Stage2 재개 없이 Stage1 carry distill epoch30000에서 새로 시작하며, 비교 중인 CA4와 같은 seed2899를 사용한다. CA+FFN+LayerNorm+residual 4층, encoder/RMS 동결, head `[W,0]` 전이, 공동80/단독20, AMP80/10/10·데이터 해시, paired/단독 RSI·초기 AMP 이력, SA/CA/task 셔플은 기반과 같다. carryWith-only나 가까운 목표 curriculum으로 바꾸지 않는다.

공동 HOLDING gate `g`와 상자→목표 XY 단위 방향 `u`에 대해, 각 사람의 현재 골반 world XY 선속도 `v_i`를 정규화한 `s_i=clamp(dot(v_i/max(||v_i||,1e-6),u),0,1)`를 계산한다. 별도 위치 history를 사용하지 않아 reset 순간 위치 이동을 속도로 세지 않는다. 몸 heading·목표 속도·리더는 지정하지 않는다. 두 사람에게 각각 `0.2*g*min(s_A,s_B)`를 한 번씩 더하고, 단독은0이다. 방향은 같아도 속도 크기는 달라도 된다.

상자–목표 XY 거리가 `목표 XY 반대각선+0.8m` 이내이면 min 점수 대신1을 사용한다(AT0.8m, 현재 ON_TOP 약1.277m). 따라서 먼 곳의 정지는0, 가까운 곳의 정지는 gate가 켜졌을 때0.2다. 기존 거리 보상/성공 조건은 유지하며 farthest-human heading은 추가하지 않는다. 최종 PPO 보상은 `0.5*(기존 task+새 방향 보상+기존 음수 패널티)+0.5*AMP`이므로 새 항의 최종 기여 상한은 사람당0.1이다. 이 항만으로 속도 일치나 실제 목표 도착을 보장하지 않는다.

기존 Align 보상 연결을 사용하므로 로그명은 `reward_terms/joint_alignment`, `relation/90_debug/joint_alignment/{score,saturated}`다. Move reward 계약이 checkpoint에 저장되며, 기존 CA4/Align 모델의 직접 resume/eval은 거부한다. 아래 명령은 이 실험을 새로 시작하며 기존 학습을 종료하지 않는다.

검증: CPU63개, GPU5·2048환경 전이→epoch2 저장→epoch3 재개 및16환경32-step test wrapper 평가 통과. Encoder69개 tensor/RMS 동결·모델/scalar136종 finite, 실제64환경에서 보상 동일 공유·한 번 합산·성분 합·단독/gate-off 0을 확인했다. AMP30000개79.68/10.25/10.07%·공동1638/단독410·RSI 초기 이력/부분 reset 검사도 통과했다. 결과는 `output/approach_stage2_joint_carry_mixed80_locoamp_ca4_move_task_embedding_check/{saved_check,runtime_check,reset_check}.json`이며 본학습·실제 VNC 화면·장기 성능은 미검증이다.

```bash
env -u RESUME_CHECKPOINT -u MAX_ITERATIONS \
TOKENHSI_GPU=5 SEED=2899 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_move_task_embedding \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_move_task_embedding_train.sh 2 2048 4
```

`env -u`는 이 명령에서만 기존 재개/반복 제한 변수를 제외한다. 짧은 확인은 별도로 `MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_ca4_move_task_embedding_check`를 설정한다. 재개는 같은 Move 실험의 `RESUME_CHECKPOINT="$MOVE_CHECKPOINT"`를 지정한다. 아래 `MOVE_CHECKPOINT`에는 이 실험에서 저장한 `.pth` 경로를 넣는다.

로컬 viewer / 화면 없는 평가 / 서버 VNC:

```bash
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_move_task_embedding_test.sh "$MOVE_CHECKPOINT" 2 16 4 1

TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_move_task_embedding_test.sh "$MOVE_CHECKPOINT" 2 16 4 1

TOKENHSI_GPU=5 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_ca4_move_task_embedding_vnc.sh "$MOVE_CHECKPOINT" 2 10 4 10
```

<a id="stage2-mixed80-locoamp-align"></a>
## Mixed80 LocoAMP 공동 방향 보상

`approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding`은 LocoAMP task CA에 방향 보상만 추가한 별도 실험이다. Stage1 carry distill epoch23000에서 새로 시작한다. 공동80/단독20·상자/받침 크기·paired 및 단독 RSI·AMP80/10/10·CPA·frozen encoder/RMS·CA/SA/task 셔플·128입력 head 전이는 기반 실험과 같다.

공동 AT/ON_TOP에서 매 step 목표 XY와 골반 XY 거리가 가장 먼 human을 선택한다. 선택된 사람의 heading과 `운반 상자→목표` XY 단위 방향의 내적을 `[0,1]`로 제한한다. 방향 점수의 포화 거리는 `목표 XY 반대각선+0.8m`이며 AT 점 목표는0.8m, 현재 ON_TOP 받침은약1.277m다. 이 거리 이내에서는 방향 점수가1이다. 별도 완화식·역할 유지 상태는 없고 동률에서는 실제 human index0을 선택한다. 선택은 토큰/edge 입력 순서와 무관하다.

두 사람의 반대 앵커 HOLDING gate가 켜졌을 때 `0.1×방향 점수`를 각자에게 똑같이 더한다. 단독 환경에는 추가하지 않는다. 기존 HOLDING·AT/ON_TOP state/progress/success 및 progress buffer0.1은 그대로다. 방향 보상은 `reward_terms/joint_alignment`, 공동 장면의 점수/포화 비율은 `relation/90_debug/joint_alignment/{score,saturated}`에 기록한다. 새 reward 계약이 checkpoint에 들어가므로 기존 LocoAMP CA checkpoint의 직접 resume/eval은 거부한다. 새 실험에서 저장한 checkpoint로 train 재개·test/VNC를 실행한다.

검증: 관련 CPU28개, GPU5·2048환경의 Stage1 epoch23000 전이→epoch2 저장→epoch3 재개, encoder69개 tensor/RMS 동결·모델/scalar finite를 확인했다. 저장 모델64환경32-step에서 공동 AT/ON_TOP·단독 보상 분기 및 동일 보상의 중복 없는 지급을 검사했고 전용 test wrapper16환경32-step을 통과했다. [저장 검사](../output/approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding_check/saved_check.json), [실제 보상 검사](../output_etc/joint_alignment_check/runtime_check.json), [재개 검사](../output_etc/joint_alignment_check/resume_check.json). 실제 VNC 화면·장기 성능은 미검증이다.

학습:

```bash
TOKENHSI_GPU=5 SEED=42 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00023000.pth \
OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding_train.sh 2 2048 4
```

본학습 명령은 `RESUME_CHECKPOINT`·`MAX_ITERATIONS`가 export되지 않은 셸 기준이다. 짧은 확인은 `MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding_check`를 지정한다. 같은 실험 재개는 `RESUME_CHECKPOINT="$ALIGN_CHECKPOINT"`를 쓴다.

로컬 viewer / 화면 없는 평가 / 서버 VNC:

```bash
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding_test.sh "$ALIGN_CHECKPOINT" 2 16 4 1

TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=joint_carry_ontop EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding_test.sh "$ALIGN_CHECKPOINT" 2 16 4 1

TOKENHSI_GPU=5 TASK_GRAPH=random_scenario EVAL_SKILLS=carryWith \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding_vnc.sh "$ALIGN_CHECKPOINT" 2 10 4 10
```

<a id="align-amp-toy"></a>
### Align carryWith 직선 운반 AMP 진단

`joint_carry_amp_toy.sh`는 기존 Align config를 프로세스 안에서만 재설정하는 **화면 없는 진단 전용** 실행기다. 공동 AT 100%·동시 HOLDING carryWith 100%, A의 초기 전방/후방 1.5m 목표를 절반씩 사용한다. 초기 속도는 0이고 미사용 물체는 경로 밖으로 옮긴다. 본학습 YAML·RSI 데이터·보상식은 변경하지 않는다.

두 조건은 같은 Align checkpoint의 정책·optimizer에서 재개한다. Task 보상 가중치 0.5를 유지하고 AMP 보상 가중치만 0.5/0으로 바꾼다. AMP 판별자 학습은 양쪽 모두 유지한다. 학습은 2048환경에서 기본 100회 업데이트, 평가는 동일 seed의 64환경에서 최대 6초다. Encoder/RMS 동결·finite·초기 장면 일치를 검사한다. 진단 checkpoint는 이 실행기로 평가하며, 일반 Mixed80 성능이나 장기 학습 가능성으로 해석하지 않는다. 별도 viewer/VNC 실행기는 없다.

```bash
TOY_SOURCE=align_tmp/ApproachStage2JointCarryMixed80LocoAmpAlignTaskEmbedding_00008000.pth
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/joint_carry_amp_toy.sh train 0.5 "$TOY_SOURCE" output/align_amp_toy_20261010_amp05_check 100
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/joint_carry_amp_toy.sh train 0 "$TOY_SOURCE" output/align_amp_toy_20261010_amp0_check 100

# TOY_AMP05/TOY_AMP0에는 각 학습 run의 nn/JointCarryAmpToy.pth 경로 지정
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/joint_carry_amp_toy.sh eval 0.5 "$TOY_SOURCE" output_etc/align_amp_toy_20261010/baseline
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/joint_carry_amp_toy.sh eval 0.5 "$TOY_AMP05" output_etc/align_amp_toy_20261010/amp05
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/joint_carry_amp_toy.sh eval 0 "$TOY_AMP0" output_etc/align_amp_toy_20261010/amp0

# tokenhsi 환경: 거리 그래프·실제 시뮬 관절 궤적 비교 영상 생성
python tokenhsi/tests/toy_joint_carry_amp_report.py output_etc/align_amp_toy_20261010
```

결과는 각 output의 `training_check.json`과 [평가 비교](../output_etc/align_amp_toy_20261010/summary.json)에 기록한다. 초기 공동 HOLDING과 목표 거리 감소를 따로 보며, 영상은 시뮬레이션 관절 좌표를 그린 비교용 재생이다.

2026-10-10: 지정 Align epoch8000에서 두 조건 모두100회 학습·epoch8100 저장과 encoder69개 tensor/RMS 동결·head/CA 갱신·finite를 확인했다. 동일64환경 최대6초 평가의 목표 거리 감소(전방/후방)는 원본−0.90/−1.52m, AMP 유지−0.80/−2.66m, 제거−0.36/−1.39m다. 제거 조건이 상대적으로 낫지만 모두 평균적으로 멀어지므로 해결 확인은 아니다. [거리 그래프](../output_etc/align_amp_toy_20261010/distance.png), [고정 장면0](../output_etc/align_amp_toy_20261010/scene_0.mp4), [고정 장면1](../output_etc/align_amp_toy_20261010/scene_1.mp4). 단일 seed·짧은 학습이라 원인 단정이나 일반 과제 성공률 평가에는 쓰지 않는다.

#### 실제 이동 방향의 목표·보상 검사

`reward_eval`은 추가 학습 없이 같은64환경에서 실제 보상 성분·AMP·action·물리 궤적을 기록한다. `TOY_DIRECTION_FILE`에는 앞선 toy 평가의 `traces.npz`를 지정한다. 각 환경의 처음2초 상자 이동 방향에1.5m 목표를 두며, 모든 환경에서 이동량5cm 이상인지 확인한다. `TOY_REPLAY_FILE`을 함께 주면 새 목표에 대한 정책 대신 기존 action을 그대로 재생한다. 마지막 조건은 보상 계산을 분리하는 검사이며 목표를 따라가는 정책의 성공으로 해석하지 않는다.

```bash
REWARD_CHECKPOINT=output/align_amp_toy_20261010_amp05_check/JointCarryAmpToy_10-20-13-09/nn/JointCarryAmpToy.pth
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/joint_carry_amp_toy.sh reward_eval 0.5 "$REWARD_CHECKPOINT" output_etc/align_goal_reward_20261010/original
TOKENHSI_GPU=5 TOY_DIRECTION_FILE=output_etc/align_amp_toy_20261010/amp05/traces.npz \
bash tokenhsi/scripts/multi_agent/joint_carry_amp_toy.sh reward_eval 0.5 "$REWARD_CHECKPOINT" output_etc/align_goal_reward_20261010/along_motion
TOKENHSI_GPU=5 TOY_DIRECTION_FILE=output_etc/align_amp_toy_20261010/amp05/traces.npz \
TOY_REPLAY_FILE=output_etc/align_goal_reward_20261010/original/reward_traces.npz \
bash tokenhsi/scripts/multi_agent/joint_carry_amp_toy.sh reward_eval 0.5 "$REWARD_CHECKPOINT" output_etc/align_goal_reward_20261010/action_replay
# tokenhsi 환경
python tokenhsi/tests/toy_joint_carry_goal_reward_report.py output_etc/align_goal_reward_20261010
```

2026-10-10: 초기 물리 상태 동일·재생 body/box/action/AMP/HOLDING 보상 차이0, 두 사람 AT 보상 동일·성분 합 일치를 확인했다. 공동 HOLDING 중 접근9420 step에서 거리 항 감소0이었다. 고정 장면1에서 목표에 접근하면 AT0.084→0.186, 동일 물리 시점의 목표만 비교해도 두 사람 혼합 보상이 각각0.067 증가했다. 반면 새 목표를 본 정책은 움직임을 바꿔 다시 멀어지는 장면이 있어, 보상 배선 검사 통과와 정책의 목표 운반 성공은 구분한다. [결과](../output_etc/align_goal_reward_20261010/summary.json), [보상 그래프](../output_etc/align_goal_reward_20261010/reward_comparison.png), [장면0](../output_etc/align_goal_reward_20261010/scene_0.mp4), [장면1](../output_etc/align_goal_reward_20261010/scene_1.mp4). 영상은 기존 목표 정책/새 목표 정책/새 목표에 기존 action 재생 순서다.

<a id="stage2-mixed80-task-ca"></a>
## Stage 2 공동 80%·단독 20% task CA

`approach_stage2_joint_carry_mixed80_task_embedding`은 별도 config/output/checkpoint를 사용하는 실행 가능한 실험이다. 기존 100% 공동운반 학습은 자동 변경되지 않는다. Stage 1 checkpoint에서 새로 시작하며 기존 100% 공동운반 Stage 2 checkpoint의 직접 resume/eval은 보상 계약 차이로 거부한다.

- 2048환경을 생성할 때 공동 1638개·단독 410개로 고정한다. 공동은 reset마다 joint AT/ON_TOP 50/50, 단독은 agent마다 AT/ON_TOP을 독립적으로 50/50 샘플링한다. 에피소드 길이가 다르므로 reset 횟수 비율은 80/20과 다를 수 있다. `sampling/joint_environment_fraction`은 실제 환경 비율, `sampling/independent`는 reset 비율이다.
- 공동 payload/받침·앵커·paired RSI·CPA는 기존 공동운반 설정을 유지한다. 단독 payload O0/O1은 각 축 20~60cm, 받침 O2/O3는 XY 50~80cm·Z 25~45cm의 Stage 1 carry 크기 범위(5cm 간격)를 재사용한다. 각자 다른 payload·받침·goal을 사용하며 크기는 환경 생애 동안 고정한다.
- 단독 HOLDING은 Stage 1의 상자 중심 기준이다. 각자의 현재 HOLDING 만족도가 0.9 이상일 때 자기 placement state/progress/success만 열린다. 공동은 반대 앵커 동시 HOLDING gate를 유지한다. 두 경우 모두 포화·성공 latch는 없다.
- 단독 RSI는 agent별 loco/pickUp/carryWith/putDown 40/10/40/10이며 기존 크기별 물리 선별·후반 프레임 샘플링·대체 skill 경로를 재사용한다. 단독 ON_TOP의 putDown은 받침을 포함해 검사한 RSI다. 공동 ON_TOP은 기존대로 바닥 putDown을 쓰지 않는다. 캐시는 `output/rsi_cache`에 저장하며 새 크기 조합은 첫 실행 때 검사한다.
- AMP expert 분포는 두 그룹 모두 기존 single-human carry family다. 공동 reset의 AMP 이력은 현재 복원 상태 반복, 단독은 원래 single-human reference 이력이다. 혼합 reset은 물리 tensor를 한 번에 commit한다.
- CA K/V에는 두 task가 모두 들어간다. 공동은 SELF/COUPLED, 단독은 SELF/NONE이며 NONE은 차단 마스크가 아니다. 따라서 세 bias가 모두 학습된다. Frozen actor/RMS와 SA·CA 순서 셔플 방식은 기존과 같다.

```bash
# 새 본학습: 저장소 루트에서 실행
TOKENHSI_GPU=5 \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00011000.pth \
OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_task_embedding \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_task_embedding_train.sh 2 2048 4

# 확인용은 MAX_ITERATIONS=1과 별도 output을 사용
TOKENHSI_GPU=5 STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00011000.pth MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_task_embedding_check bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_task_embedding_train.sh 2 2048 4

# CKPT는 새 Mixed80 실험에서 저장한 Stage 2 checkpoint
TOKENHSI_GPU=5 RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_task_embedding_train.sh 2 2048 4

# 로컬 viewer / 화면 없는 단독 평가 / 서버 VNC
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=joint_carry_at EVAL_SKILLS=carryWith bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_task_embedding_test.sh "$CKPT" 2 16 4 1
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=independent_at_ontop EVAL_SKILLS=carryWith bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_task_embedding_test.sh "$CKPT" 2 16 4 1
TOKENHSI_GPU=5 TASK_GRAPH=joint_carry_ontop EVAL_SKILLS=carryWith bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_task_embedding_vnc.sh "$CKPT" 2 1 4 10
```

본학습 명령은 `RESUME_CHECKPOINT`·`MAX_ITERATIONS`를 별도로 export하지 않은 셸을 기준으로 한다. 평가 preset은 `random_scenario`, `joint_carry_at`, `joint_carry_ontop`, `independent`, `independent_at_at`, `independent_at_ontop`, `independent_ontop_at`, `independent_ontop_ontop`이다. 명시적 joint/independent preset은 모든 평가 환경을 해당 그룹의 크기로 생성한다. 평가 RSI는 기존 Stage 1처럼 지정 reference skill을 직접 쓰며 학습용 크기별 cache 선별과 구분한다.

2026-10-08 검증: CPU64개, GPU5·2048환경 짧은 학습(epoch2 저장)·1638/410 분할·AMP 이력·1024개 혼합 및 각32개 단일그룹 부분 reset을 통과했다. Actor69개 tensor/RMS 불변, NONE/SELF/COUPLED bias 갱신과 scalar133종/266값 finite를 확인했다. 저장 모델의 단독 AT/ON_TOP 혼합·공동 ON_TOP을 각각16환경·32-step headless 평가했다. 결과는 `output/approach_stage2_joint_carry_mixed80_task_embedding_check/{saved_check,reset_check}.json`에 있다. 연결 검증이며 장기 성능·단독 능력 유지·VNC 화면은 미검증이다.

### 사람 번호·SA/CA 순서 진단

기존 Mixed80 epoch15000·GPU6·carryWith 시작64환경을180-step 진행하며832개 장면에서 사람 상태/goal/graph 참조를 함께 교환했다. 재생성 관측·정규화 오차0, 최종 action 교환 최대1.03e-5·RMSE1.39e-7 미만, SA/CA 순서 변경 최대7.08e-6 미만이었다. Encoder 내부 최대8.57e-5는 진단 기준5e-5 초과로 그대로 기록했다. [결과](../output_etc/joint_carry_permutation_20261011/permutation_check.json). 동일 물리 상태에서 정책 입력을 교환한 검사이며, 모든 모델/목표 방향의 일반화나 교환한 물리 rollout의 동일성을 보장하지 않는다.

```bash
TOKENHSI_GPU=6 bash -c 'source tokenhsi/scripts/multi_agent/runtime_env.sh; PYTHONPATH=tokenhsi python tokenhsi/tests/toy_joint_carry_permutation.py \
  --task HumanoidMACarry \
  --cfg_train tokenhsi/data/cfg/train/rlg/amp_ma_stage2_joint_carry_mixed80_task_embedding.yaml \
  --cfg_env tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_task_embedding.yaml \
  --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
  --checkpoint output/approach_stage2_joint_carry_mixed80_task_embedding/ApproachStage2JointCarryMixed80TaskEmbedding_08-11-18-15/nn/ApproachStage2JointCarryMixed80TaskEmbedding_00015000.pth \
  --num_envs 64 --num_agents 2 --num_objects 4 \
  --task_graph random_scenario --eval_skills carryWith --eval_skill_probs 1.0 \
  --seed 42 --episode_length 210 \
  --output_path output_etc/joint_carry_permutation_20261011 \
  --test --eval --headless --no_video'
```

<a id="stage2-before-task-ca"></a>
## Stage 2 BEFORE 80%·독립 20% task CA

`approach_stage2_before_task_embedding`은 COUPLED를 제외한 별도 env/train config·train/test/VNC·output이다. 네 과제 Stage 1 task embedding 또는 네 과제 distill checkpoint에서 전이한다. Carry-only source와 기존 공동운반 Stage 2 checkpoint의 직접 전이/resume/eval은 거부한다. 실행 연결은 확인했으나, 합성 RSI 초기 자세의 물리 안정성 검사는 일부 미통과 상태다(아래 검증 결과 참조).

- 2명·2048환경·4물체. 환경 생성 시 BEFORE-climb/sit/stack 각546개·독립410개를 셔플 배정하고 부분 reset에도 그룹을 유지한다. 독립 과제는 agent별 sit/climb/carry_at/carry_ontop 10/25/32.5/32.5%다. 짧은 에피소드가 많은 그룹은 reset 횟수 기준 로그에서 더 자주 나타날 수 있으며 고정 환경 비율과 구분한다.
- BEFORE는 A(H0)의 `carry_at(O0,G0)` 이후 B(H1)의 `climb(O0)`, `sit(O0)`, `carry_ontop(O1,O0)`다. 사람/entity·interaction 토큰 순서를 셔플하고 대응을 복원한다. 별도 역할 교환은 하지 않는다. NONE/SELF/COUPLED/BEFORE의 head별 bias 중 COUPLED는 사용하지 않는다. Actor encoder/task embedding·관찰 RMS 동결과 `[W,0]` head 확장을 유지한다.
- A의 현재 AT geometry 성공만 prerequisite으로 사용한다. A가 손을 놓아도 AT가 성립하면 열린다. B의 dependent state/success와 성공 포화는 gate를 따르고 progress는 항상 적용한다. B의 자기 HOLDING은 항상 보상한다. Latch·강제 대기·행동 시작 순서 제약은 없다. 독립 scene의 보상은 Stage 1 자기 edge 합계다. Collision은 기존 CPA 계수0.5·거리0.7m·discount0.99를 사용한다.
- 모든 물체는 X/Y 50–60cm·Z35–45cm의 각 축5cm 간격, 총27개 크기에서 샘플링한다. 밀도100을 유지한다. 크기는 asset 생성 시 고정하며 서로 다른 역할에 같은 허용 범위를 쓴다.
- RSI는 carry AT/ON_TOP의 loco/pickUp/carryWith/putDown 40/10/40/10, SIT loco/sit 50/50, CLIMB loco/climb 50/50다. 별도 prerequisite 충족/미충족 비율이나 성공 시작 거부는 없다. 크기별 물리 선별은 실제 source/support binding을 사용하며 Stage 1의 `output/rsi_cache`와 선별 기준을 재사용한다. 단일 agent reference의 선별 통과가 합성된 두 사람 장면의 물리 안정성을 보장하지는 않는다. 공유 상자는 A 상태로 한 번만 결정하고 B reference·속도·AMP reference history를 같은 좌표 변환으로 정렬한다. 재시도 중 이미 선택한 skill을 유지하며 부족한 단일 skill pool의 fallback은 지표로 기록한다.
- AMP는 carry/sit/climb별 기존 expert와 family-matched demo/replay를 유지한다. Demo family 분포는 활성 환경 그룹에서 계산하고, gate가 닫혀도 B의 family를 변경하지 않는다. 평가 skill 목록을 줄여도 원래 expert 인덱스를 유지한다.

```bash
# 네 과제 Stage 1 checkpoint를 명시한다. 아래 epoch6500은 연결 검사에 사용한 source다.
SOURCE=/home/hwanhee/ksh/stage1_task/output/approach_scenario_stage1_unified_size_rsi_task_embedding_distill/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingDistill_07-15-20-48/nn/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingDistill_00006500.pth
# 본학습
TOKENHSI_GPU=5 STAGE1_CHECKPOINT="$SOURCE" MAX_ITERATIONS='' RESUME_CHECKPOINT='' OUTPUT_PATH=output/approach_stage2_before_task_embedding bash tokenhsi/scripts/multi_agent/approach_stage2_before_task_embedding_train.sh 2 2048 4
# 짧은 연결 검사
TOKENHSI_GPU=5 STAGE1_CHECKPOINT="$SOURCE" MAX_ITERATIONS=1 OUTPUT_PATH=output/approach_stage2_before_task_embedding_check bash tokenhsi/scripts/multi_agent/approach_stage2_before_task_embedding_train.sh 2 2048 4
# 같은 BEFORE checkpoint에서 재개
CKPT=/absolute/path/to/before_checkpoint.pth
TOKENHSI_GPU=5 RESUME_CHECKPOINT="$CKPT" bash tokenhsi/scripts/multi_agent/approach_stage2_before_task_embedding_train.sh 2 2048 4
# 로컬 viewer
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=before_climb bash tokenhsi/scripts/multi_agent/approach_stage2_before_task_embedding_test.sh "$CKPT" 2 1 4 10
# 화면 없는 평가
TOKENHSI_GPU=5 HEADLESS=1 TASK_GRAPH=before_stack bash tokenhsi/scripts/multi_agent/approach_stage2_before_task_embedding_test.sh "$CKPT" 2 16 4 3
# 서버 VNC
TOKENHSI_GPU=5 TASK_GRAPH=before_sit bash tokenhsi/scripts/multi_agent/approach_stage2_before_task_embedding_vnc.sh "$CKPT" 2 1 4 10
```

평가 preset은 `before_climb`, `before_sit`, `before_stack`, `independent`, `sit`, `climb`, `carry_at`, `carry_ontop`, `random_scenario`다. 기본은 두 사람 loco 시작이다. `EPISODE_LENGTH`와 `EVAL_SKILLS`/`EVAL_SKILL_PROBS`를 지원하며, skill 목록에서 각 agent 과제에 적합한 항목만 샘플링하며, 어느 한쪽에 가능한 항목이 없으면 오류를 낸다. GPU 기본값은 wrapper 전체에서5이며 `TOKENHSI_GPU`로 한 번 덮어쓸 수 있다.

2026-10-08 검증: 관련 CPU94개와 GPU5·2048환경 짧은 학습/epoch2 저장·epoch3 재개를 통과했다. Actor69개 tensor·관찰 RMS6개 불변, NONE/SELF/BEFORE bias 갱신·COUPLED0, scalar182종/364값 finite를 확인했다. 환경 분할은 독립410·각 BEFORE546이며 첫 reset의 RSI fallback은0이었다. AMP family/demo 대응·reference history와 1024개 혼합/각32개 그룹 부분 reset에서 미선택 상태 보존을 확인했다. 최종 모델의 BEFORE-climb에서 agent별 loco/climb reference 선택·16환경32-step headless 평가가 완료됐다. 초기 연결 모델의 나머지 sit/stack/독립 평가 경로도 각각16환경32-step으로 확인했다.

**남은 제한:** 고정 PD target으로 초기 자세를0.1초 유지하는 별도 물리 진단은1575/2048개만 통과했다(473개 미통과, 최대 box 속도14.42m/s·변위1.21m). 따라서 코드의 geometry 재시도 실패0을 물리 안정성 통과로 해석하면 안 된다. 더 엄격한 단일-agent 캐시를 시험했지만 skill fallback이 증가했고 모든 장면을 안정화하지 못해 기본 설정에 적용하지 않았다. Stage 1 RSI 비율은 유지했으며 합성 초기화의 물리 안정성 보완은 남아 있다. 결과는 `output/approach_stage2_before_task_embedding_check_final/{saved_check,reset_check,resume_check}.json`; 본학습·장기 성능·VNC 화면은 미검증이다.


<a id="stage1-grasp-shift"></a>
## Stage1 공유물체 파지 중심 이동 · 추론 토이

두 사람이 **같은 payload 하나·같은 목표 하나**를 공유한다. 환경은 기존 LocoPlatform(52×80×40cm payload·Goal360·공동 RSI·loco의50% 받침)을 사용하고, 정책은 `stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth` 원본 Stage1이다. 전용 진입점에서 환경 설정 검증 후 네트워크 설정만 원본 Stage1으로 선택하며 source metadata/strict state_dict 검사를 한다. Stage2 CA/SA나 학습은 사용하지 않는다.

시작 시 사람 위치를 상자 로컬 좌표로 변환해 반대 Y측 앵커를 배정하고 episode 동안 유지한다. 사람 번호에 고정하지 않는다. 앵커는 기존 보상과 같은 `±(Y반폭−0.15m)`이며 현재 상자에서 ±0.25m다. 매 step 같은 물리 장면의 관측 두 벌을 만들어 A용에는 A 앵커, B용에는 B 앵커가 payload 중심으로 보이게 한다. 두 벌을 같은 Stage1 모델에 배치로 넣고 A-view의 A행동/B-view의 B행동만 선택해 **동일한 물리 환경**에 적용한다.

`SHIFT=0`은 둘 다 실제 중심, `SHIFT=1`은 각각 배정된 앵커, `SHIFT=0.5`는 중심→앵커의 절반이다. `SHIFT=-1`은 두 사람의 앵커를 뒤바꾸는 진단용이며 기본 비교에는 필요 없다. 이전 독립 단독 토이의 SHIFT 정의/명령을 대체한다.

관측 위치 `p+Rδ` 및 object-local 속도 `v_local+ω_local×δ`만 변경한다. 실제 물체·크기·목표·task packet·보상/RSI/AMP는 유지하며, 관측 변화는 상자↔목표 관계에도 영향을 준다. 배정은 관측 실험용으로, 보상이 앵커를 강제 배정하는 것은 아니다. 공동집기 개선 여부는 뷰어에서 판단한다. 가상 상자는 따로 그리지 않는다.

서버 VNC — 기본GPU5·포트6080·1환경·SEED42·loco·300-step·10회. 하나씩 실행하고 다음 비교 전에 기존 viewer를 닫는다.

기존 중심을 보는 공유물체 baseline:
```bash
TOKENHSI_GPU=5 SHIFT=0 bash tokenhsi/scripts/multi_agent/stage1_grasp_shift_vnc.sh
```

각자 반대 앵커를 중심으로 보는 공유물체:
```bash
TOKENHSI_GPU=5 SHIFT=1 bash tokenhsi/scripts/multi_agent/stage1_grasp_shift_vnc.sh
```

로컬 viewer / headless:
```bash
TOKENHSI_GPU=5 SHIFT=1 HEADLESS=0 bash tokenhsi/scripts/multi_agent/stage1_grasp_shift_test.sh
TOKENHSI_GPU=5 SHIFT=1 HEADLESS=1 REPEATS=1 bash tokenhsi/scripts/multi_agent/stage1_grasp_shift_test.sh
```

`TASK_GRAPH=joint_carry_ontop`, `EVAL_SKILLS=carryWith`로 변경 가능. 단독/random graph는 거부한다. 첫 위치 인자로 호환 Stage1 checkpoint를 교체한다. `SAVE_TRACE=1`은 관측/행동/손 위치 및 zero-shift 두-view 행동 복원 오차를 기록하고, `SAVE_FRAMES=1`은 실제 viewer 스크린샷을 저장한다. 결과는 `output_etc/stage1_shared_grasp_shift_<SHIFT>/`. 비교 시 동일 시드의 첫 장면을 기준으로 보며, 정책에 따른 종료 시점이 달라 이후 reset 장면까지 일치한다고 보장하지 않는다. 여러 환경은 첫 종료 시 함께 중단하므로 성공률 평가용이 아니다. 학습 명령은 없다.


<a id="stage2-sa4-graspview"></a>
## SA4 LocoPlatform GraspView · 공동 파지 관측

기반은 `approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_task_embedding`. 새 실험 `approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_graspview_task_embedding`는 **공동80%에서만 사람별 payload 중심 관측을 변경**한다. 실제 상자·목표·bbox·물리/보상·AMP 원본80/후진10/옆걸음10·공동 RSI/full pickup·Goal360·CPA release+1m·loco 받침 및 단독20%는 유지한다.

reset 때 원본 사람/상자 pose로 로컬 Y 순서를 정하고 반대 앵커를 배정한다. 앵커는 기존 HOLDING과 같은 `±(Y반폭−0.15m)`, 현재 상자에서 ±0.25m다. 배정을 episode 동안 유지하고 매 step 상자 회전에 따라 위치/선속도만 갱신한다. 실제 회전과 크기는 그대로 둔다. 리워드에 배정을 강제하지 않는다.

관측은 원본634차원의 A-view/B-view를 묶어1268차원으로 rollout/replay에 저장한다. 정규화는 Stage1 통계로 각634차원 view에 따로 적용하고 pose/packet은 통과시킨다. 공동에서는 각 view에서 담당 human64를 선택해 `[h_A,h_B]`로 SA4에 전달하며 단독에서는 두 view가 동일하여 encoder를 한 번만 실행한다. 80/20에서 encoder 처리 장면 수는 대략1.8배, 파라미터/SA4/action64/value64/AMP1320은 그대로다. Critic도 같은 view 구성을 사용한다.

Actor Stage1 encoder·scene RMS 동결, SA4/action head 학습, Critic 전체 학습. 초기 SA 추가 출력은0이어서 초기 Actor 행동은 Stage1에 각자 앵커 관측을 넣은 행동과 일치한다. Actor/Critic encoder와 head는 Stage1 checkpoint에서 전이한다. 사람·task·token 셔플 후 canonical human ID를 따라 자기 feature를 선택한다. 단독은 기존 실제 중심이며 SELF/COUPLED/NONE bias 모두 유지한다.

`jointCarryGraspView`/checkpoint `joint_carry_grasp_view` 계약으로 기존 SA4 checkpoint와 resume/eval 혼용을 차단한다. 이 실험은 현재2명·4물체 전용이며 기존6명 demo에 넣지 않는다. 새 학습은 Stage1에서 시작한다.

```bash
# 새 Stage2 학습: 기존 Stage2 resume가 아님
TOKENHSI_GPU=5 \
RESUME_CHECKPOINT= \
STAGE1_CHECKPOINT=stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00030000.pth \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_graspview_task_embedding_train.sh 2 2048 4
```

```bash
GRASP_CHECKPOINT="output/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_graspview_task_embedding/<run>/nn/<checkpoint>.pth"

# 로컬 viewer (화면 없는 평가는 HEADLESS=1)
TOKENHSI_GPU=5 HEADLESS=0 TASK_GRAPH=joint_carry_at EVAL_SKILLS=loco \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_graspview_task_embedding_test.sh "$GRASP_CHECKPOINT" 2 1 4 10

# 서버 VNC
TOKENHSI_GPU=5 TASK_GRAPH=joint_carry_at EVAL_SKILLS=loco \
bash tokenhsi/scripts/multi_agent/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_graspview_task_embedding_vnc.sh "$GRASP_CHECKPOINT" 2 1 4 10
```

`TASK_GRAPH=joint_carry_ontop`, `EVAL_SKILLS=carryWith/pickUp`도 지원한다. 단독 확인은 `TASK_GRAPH=independent`. ON_TOP의 지상 putDown RSI는 기존처럼 지원하지 않는다. 짧은 학습은 `OUTPUT_PATH=output/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_graspview_task_embedding_check MAX_ITERATIONS=2`로 별도 경로를 사용하며2048환경은 유지한다. 본학습 시 `MAX_ITERATIONS`를 설정해 둔 셸이면 해제한다.

검증(2026-10-16): CPU16개 통과, GPU5·2048환경 epoch3 저장/재로드 확인. Actor encoder69개·scene RMS 불변, Critic69개/action head6개/8개 SA bias 갱신, 모델 파라미터4,398,450개로 기존 SA4와 동일. 공동1638/단독410·AMP79.83/10.03/10.14%·full pickup100프레임·7가지 RSI/Goal360·부분 reset·배정 유지·physics 통과. 16환경·loco/carryWith·64-step 설정 headless 추론 정상 종료. [저장본](../output/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_graspview_task_embedding_check/saved_check.json)·[AMP/reset](../output/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_graspview_task_embedding_check/reset_check.json)·[RSI/목표](../output/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_graspview_task_embedding_check/goal360_runtime_check.json)·[앵커/reset](../output/approach_stage2_joint_carry_mixed80_locoamp_sa4_goal360_fullpickup_cparelease1_locoplatform_graspview_task_embedding_check/grasp_view_runtime.json). 실제 VNC 화면·장기 성능은 미검증이다.
