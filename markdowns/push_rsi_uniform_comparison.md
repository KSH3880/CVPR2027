# GPU1 원본·JHH rsi_uniform·stage1_push_koo 비교

확인일: 2026-10-26. 설정과 실행 코드를 직접 비교했다. 본 복사본은 `stage1_fix_koo`의 미커밋 변경까지 사용자가 복사한 `stage1_fix_koo copy`를 `stage1_push_koo`로 이름 변경한 것이다. 원본 GPU1 학습은 별도 저장소에서 계속 실행하며, 이 문서의 변경은 복사본에만 적용한다.

## 최신 변경: 원본 Stage1 SUPPORT 복원

팀 확인에 따라 `stage1_push_rsi_uniform`은 실제4상자(담당2+SUPPORT2)를 유지한다. 종전복사본은PUSH담당자의여분상자까지큰PUSH asset으로만들었으나, 현재공유설정은PUSH물성을담당물체에만적용한다. SUPPORT는원본X/Y0.4~0.8·Z0.25~0.5m·밀도100·마찰1.0으로복원했다. ON_TOP에서는원본받침배치를유지하고, 사용하지않는SUPPORT는원판영역에일반distractor로배치한다. PUSH경로간섭시재배치하며숨겨진actor pool을추가하지않았다.

이변경은 `pushExtension.distractor_mode: stage1_support_v1`로구분한다. PUSH상자167종/20kg/마찰0.4·RSI2~10m/±45°·보상/모델/과제비율/학습설정은유지했다. 초기모델/RMS tensor도변경전과동일하다. 새초기본은 `Stage1PushRsiUniformSupport_from78000.pth`이며아래의과거물성/출력비교는변경이력을포함한다.

## 이번에 적용한 범위

기반은 `stage1_stage2_base` commit `28f02b3`의 독립 port와 원본 `Stage1V2Fixed_00078000.pth`에서 시작한 GPU1 PUSH fine-tune이다. JHH 브랜치의 학습기·보상·네트워크를 이식하지 않았다. JHH `uniform_goal_distance`의 원리인 **모션 phase와 무관한 남은 거리 샘플링, RSI 상자 진행도 이동 생략**만 적용했다.

| 항목 | GPU1 원본 | JHH `stage1_v2_fixed_push_360_raw_cuboid_rsi_uniform` | 이 복사본 |
| --- | --- | --- | --- |
| PUSH 보행 사람↔상자 중심 | 사람 시작점 주위360°·반경1~10m 상자 제안 | 상자 중심 주위360°·반경1.65~6m 사람 제안 | GPU1 유지 |
| 보행 상자→목표 | 환경중심XY±4.5m·최소1m·상자정면±45° 거부샘플링 | 상자정면 일직선·2~10m 거리 균등 | GPU1 유지 |
| RSI 상자→목표 | 기존 목표 고정, 상자를 `initial + box_progress*(goal-initial)`로 이동 | 2~12m 거리 균등, 상자 이동 없음 | **2~10m 거리 균등 제안·정면±45°, 상자 이동 없음** |
| RSI 사람 | 선택 클립/시점의 native pose를 이동한 상자 접촉면에 정렬 | 선택 클립/시점의 raw pose를 상자 뒷면에 정렬 | 기존 GPU1 clip/시점·접촉면 정렬 유지, 상자는 원위치 |
| RSI 확률/후반 요청 | 50%, 후반30%를70% 확률로 요청 | 50%, 후반 후보 우선70% | GPU1 유지 |
| RSI 후보 | clip0/3/4/8·949개, clip7 제외 | JHH의 별도 bank | GPU1 bank/해시 유지 |
| PUSH 과제 배정 | agent별 size-conditioned task sampler | 환경을 PUSH/기존과제로 분리; PUSH환경의 두 agent 모두 PUSH | GPU1 유지 |
| 과제 비율 | PUSH50·SIT5·CLIMB12.5·AT/ONTOP각16.25% | PUSH환경50%, 기존환경 내 SIT10·CLIMB25·AT/ONTOP각32.5% | GPU1 유지 |
| 크기 | X/Y1.6~2.2·Z1.45~1.7m, min(X,Y)≥1.1Z·167종 | X1.2~1.7·Y/Z1.45~1.7m 연속·Y≥Z | GPU1 유지 |
| 질량·마찰 | 20kg 고정·상자0.4, 별도밑판없음 | 부피선형15~20kg·본체0.9·1cm밑판0.3 | GPU1 유지 |
| 바닥/감쇠/반발 | 지면1.0, 선형/각감쇠0.01, 반발0 | 동일 | GPU1 유지 |
| CONTACT 상태 | 양손 중 최악 면 XY거리의 exp(-10 d²) | 동일 기본식 | GPU1 유지 |
| CONTACT 면 | 목표 반대쪽 유한면을 yaw 기준으로 선택 | 상자 로컬-Y면을 full quaternion으로 투영 | GPU1 유지 |
| CONTACT 성공 | 양손 면점수·힘조건, 제어tick OR | 양손면거리≤0.18m·힘조건, 제어tick OR | GPU1 유지 |
| CONTACT 접근 | 사람→상자중심 거리 d에 1/(1+max(d-0.5,0)) | 상자 우회를 고려한 접촉 접근 경로 d에 1/(1+d) | GPU1 유지 |
| AT/CONTACT 가중치 | 각각 state/progress/success=0.2 | PUSH환경도 각각0.2 | GPU1 유지 |
| 과제 보상 합산 | 본인100%, 동료0% | PUSH환경은 본인90%+동료10% | GPU1 유지 |
| 사람 충돌 | 기존 CPA 페널티 | PUSH환경 거리페널티 계산 변경 및 최종보상에서 제거 | GPU1 유지 |
| Transformer | 4층·2head·64D·TaskActionFusion·GTA | 동일한 기반 구조; wrapper가 과제embedding5행 생성 | GPU1 유지 |
| 네트워크/학습 등록명 | amp_multi_agent / ma | amp_multi_agent_push / ma_push | GPU1 유지 |
| 정책/AMP 차원 | 634D·8토큰 / 1330D | 동일 | GPU1 유지 |
| actor/critic 새PUSH embedding | carry_at 행 복사 | carry_at 행 복사 | GPU1 유지 |
| 새PUSH AMP열 | carry family 가중치·RMS 복사 | 가중치0·RMS mean0/var1 | GPU1 유지 |
| 입력/AMP 정규화 | 학습 중 갱신 | 전이한 통계 고정 | GPU1 유지 |
| 학습률 | 2e-5 | 1e-5 | GPU1 유지 |
| teacher | 별도teacher 없음 | 실행시TEACHER_KL_COEF=0이면 미생성; YAML기본0.001 | GPU1 유지 |
| 학습 규모/길이 | 2인·4상자·2048환경, 현재실행1200step | PUSH1200step·기존과제600step | 2048환경, 전용train 기본1200step |

JHH 경량판 `m10to15_r20cm_a130`은 질량10~15kg·성공반경20cm·사람spawn정면±65°의 별도 variant다. 위 표는 기본 `rsi_uniform`과 비교한 것이다.

## 복사본 reset 순서

1. GPU1 기존 장면 sampler로 상자·기본목표·여분상자를 배치한다. 사람 보행 시작점과의1~10m, 기존목표 square/cone/path 검사를 유지한다.
2. PUSH agent별로 기존50%확률로 보행/RSI를 선택한다. 기존과제 RSI는 원본 경로를 사용한다.
3. RSI agent의 목표만 다시 뽑는다. 반경은U(2,10)m, 각도는상자정면±45°의 균등 제안이다. 보행 agent의 목표와 모든 상자 좌표는 그대로 둔다.
4. 두 사람의 경로·다른 사람의 보행 시작점·이미배치된 여분상자와의 여유를 검사한다. 실패한 scene의 RSI목표만 다시 제안한다. 따라서 **최종 수용 거리분포는 장애물/경로조건에 의한 조건부분포**이며, 완벽한 구간별 동일표본수를 보장하지 않는다. 2~10m 범위는 보장한다.
5. 같은949후보에서 모션/시점을 고른다. `box_progress`는 여전히 bank에 남지만 상자/목표 좌표를 이동하는 데 사용하지 않는다. 사람은 기존align/safe_place로 담당상자에 정렬한다. DOF·속도·AMP history 연결은 기존코드다.
6. eval은 기존처럼 보행만 시작하므로 평가목표는 기존GPU1 방식이다. 학습RSI 거리변경을 eval의2~10m 전용평가로 오해하지 않는다.

예: RSI시점phase80%·최초상자→목표8m이면, GPU1 원본은bank box_progress≈0.68만큼상자를이동해약2.56m가남는다. 이복사본은phase와무관하게2~10m를새로뽑고 그거리를유지한다. 원본모션의실제사람이동거리약3.67~4.72m는두계산의거리상한이아니다.

## 실행·체크포인트

공유용 전용 이름은 `stage1_push_rsi_uniform`이다. [팀 가이드](../PUSH_RSI_UNIFORM_GUIDE.md)를 따르며, 새 출력은 `output/stage1_base_push_rsi_uniform/`이다. 아래 `stage1_push_finetune` 이름과 `stage1_base_push_finetune_rsi_uniform` 출력은 최초복사본 준비 이력이다. 공유용 YAML은 같은실험값에1200step을명시하고내부source config이름만분리했다.

[실행 명령](config.md#push-rsi-uniform-copy). 설정이 다른 기존PUSH checkpoint를 직접resume하면 계약검사로 거부한다. 초기 전이본은 같은원본78000·같은embedding/AMP열초기화·fresh optimizer/epoch0으로 준비한다. 기존9000회학습본에서변경sampler로전이하는작업은이번범위에포함하지않았다. 새본학습은시작하지않는다.

본 복사본의 기존 `output/`에는 원본학습의복사이력이있다. 저장된과거GPU1실행상태/절대경로는현재복사본의실행상태가아니다. 새출력은 `output/stage1_base_push_finetune_rsi_uniform/`이며, 복사된1200step구launcher는원본절대경로를실행하지못하도록비활성화했다.

## 코드 근거

- GPU1/복사본: `ports/stage1_base_push/tokenhsi/env/tasks/multi_agent/{humanoid_ma_stage1_push,stage1_push_reward,push_contact}.py`, `learning/multi_agent/{amp_network_builder_ma,ma_agent}.py`, `prepare_finetune.py`.
- JHH: `../stage1_push_jhh/tokenhsi/env/tasks/multi_agent/{humanoid_ma_stage1_push,push_reward}.py`, `learning/multi_agent/{stage1_push_network,stage1_push_agent}.py`, `utils/push_navigation.py` 및동명env/train YAML.
- 검증 결과: `output_etc/check/push_rsi_uniform/REPORT.md`에실제수행범위와결과를기록한다.
