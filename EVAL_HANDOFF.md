# 평가 인수인계

**기준: PDF (15) · 계산 규약 `cvpr2027-evaluation-v1.1.0` · 2026-11-12.**
작업 위치: `/home/hwanhee/ksh/stage1_2_base_teacher`

## 평가 구성

| 표 | 설정 | 지표 | 실행 수/모델 |
| --- | --- | --- | ---: |
| Table 1 | 2/4/8/16H × Independent/Mixed | Independent SR / Mixed SR(I·E·C 동일 가중) | 24 |
| Table 2 | 2명 한 쌍, AT→SIT/CLIMB/ON_TOP 각각 | Group SR·Avg.·Time(Succ./All) | 9 |
| Table 3 | 공동 PUSH 한 그룹, 2/3/4명 각각 | Group SR·Avg.·t_coop | 9 |

## 실행 전 버전 확인

- **512환경 × seeds 1/6/7 · 900step=30초 · XY100% · M=2N.**
- **8H/16H:** 그룹 중심 원 반경 **3m/7.5m**. 기존 최소5m 간격 결과는 사용하지 않는다.
- **Distractor:** 관측·attention 유지, 물리만 제거. 초기 snapshot을 모델 간 공유한다.
- **Enable:** 공유 상자 공통 교집합·near50 해제. 성공은 **같은 frame의 AT와 후행 동시 성공**.
- **Time:** Succ.는 성공만, All은 실패에30초 부여. 행동별 평균→동일 가중 평균→seed 평균±표본std.
- **Coupled:** 실제 공유 상자의 목표3D 거리≤20cm가 성공. t_coop은 **출발변위20cm·손 표면거리12cm·기록 전체 누적** 정의. 과거6cm/구간 내 비율과 혼용 금지. pooled 평균과 seed 평균±std를 구분한다.
- **Checkpoint:** TokenHSI40.5k / SCT·FSM26k / Ours **V4 2k**. Coop. FT는 현재 **Table 1=6k, Table 2=4k, Table 3=6k**. 다른 checkpoint 요청 시 manifest에 따로 명시한다.

## 평가 진행

1. [상세 인수인계](eval/tables123_handoff.md)의 **checkpoint·공통 snapshot·실행 경로**를 확인한다. 계산 정의는 [공용 규약](../evaluation_protocol/README.md)을 따른다.
2. 기존 adapter를 사용해 새 checkpoint/config/reference/출력을 명시한다. **기존 고정 큐를 그대로 실행하지 않는다**(V2 checkpoint·GPU5가 고정된 실행기 있음). 코드의 `table2/enable_*`=논문 Table 2, `table2/coupled_*`=논문 Table 3.
3. Conda `tokenhsi`, `runtime_env.sh` 사용. 새 GPU 작업은 **6/7만**, 기존 작업 보존. 출력은 `output_etc/check/<새 평가명>/`.
4. Manifest에 **규약 버전·checkpoint/config·SHA·snapshot 출처/SHA·seed·환경 수·step·실제 배치 조건**을 저장한다. 버전 이름만 같다고 기존 결과를 재사용하지 않는다.
5. 초기 상태·실제 첫 정책 입력·CPU 재채점을 검증하고 **seed별 값, 평균±표본std, report/summary/manifest/검증 파일**을 제출한다. 현재42회 기준이며 일치하는 기존 결과는 재사용한다.

**예외:** TokenHSI는 고정397D 입력 차이를 명시한다. 기존 FSM은 첫 정책 관측 불일치가 남은 참고 결과이므로 새 공식 평가는 수정·검증이 필요하다. LLM Scheduler 원시 기록은 외부에 있다. Jerk·Post-AT·Collision·ablation은 이번 필수 평가에서 제외한다.
