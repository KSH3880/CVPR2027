# Stage2 jointpush RSI

`stage2_jointpush_rsi`는 [기존 Stage2](stage2_jointpush.md)의 그룹·과제·물성·보상·AMP·PPO·전이 설정을 유지하고 학습 초기화에 RSI를 추가한다. 기존 `stage2_jointpush`는 계속 학습/평가 loco100%다. 실행 명령은 [config](../config.md)에 있다.

## 학습 초기화

| 과제 | 초기화 확률 |
| --- | --- |
| 단독 SIT | loco50 / sit50 |
| 단독 CLIMB | loco50 / climb50 |
| 단독 AT·ONTOP | loco50 / pickUp10 / carryWith30 / putDown10 |
| 단독 PUSH | loco50 / push50 |
| coupled PUSH | 그룹 단위 loco50 / push50 |

단독 기존 네 과제는 Stage1의 크기별 유효 프레임 검사·후반 프레임 우선 샘플링·유효 후보가 없을 때의 fallback을 따른다. SIT/CLIMB와 ONTOP 받침은 실제 선택된 물리 actor 크기로 조회한다. 네 사람 모두 이 경로를 사용한다.

PUSH는 Stage1의 선별 bank를 사용한다. 유효 프레임의 후반30%에서70%, 전체에서30%를 샘플링한다. coupled는 그룹 전체가 같은 초기화 종류를 선택하고, 사람별 모션/프레임은 독립적으로 뽑는다.

## PUSH reset 순서

1. 기존 Stage2 방식으로 그룹·과제·구역을 배정하고 loco 사람 배치를 만든다.
2. 사람 기준2~10m 상자와 상자 기준2~10m 목표를 샘플링한다. coupled는 모든 멤버의 뒤쪽180° 조건과 목표+Y±45° 조건을 유지한다.
3. RSI 대상자의 참조 자세를 회전·평행 이동하여 담당 구역의 실제 뒤쪽 외부 면에 붙인다. 관절 자세는 참조 프레임 그대로이며 속도도 함께 회전한다. 단독은 Stage1의 목표 방향 기준 접촉 면을 사용한다.
4. Stage1의 stand-off·바닥·침투 보정을 실제 공유 상자 전체 크기에 적용한다. coupled 멤버 간 신체 충돌도 검사하고 실패한 장면은 다시 초기화한다.
5. RSI 적용은 이미 뽑은 물리 PUSH 상자와 목표를 이동시키지 않는다. 모션 진행률만큼 상자를 당기는 동작도 없다. 이후 공유 상자의 운동은 모든 구역 토큰에 반영한다.

RSI 뒤의 사람은 접촉 거리에서 시작하므로 최종 사람→상자 거리가2~10m일 필요는 없다. 이 거리 조건은 loco 배치에 적용한다. 전원 접촉 gate는 실제 현재 상태로 계산하며 RSI라는 이유로 강제로 켜지 않는다.

## 평가와 캐시

기본 eval은 학습과 같은 그룹·과제 비율·상자 크기/물성·loco 사람/상자/목표 샘플러를 사용하고 모든 과제를 loco100%로 초기화한다. RSI bank/크기별 캐시를 생성하지 않는다. `TASK_GRAPH`로 특정 과제를 선택하면 해당 평가 분포만 달라진다.

`PUSH_RSI=1`을 test/VNC에 지정하면 진단용으로 PUSH만 접촉 RSI100%로 시작한다. coupled는 그룹 전원이 배치되며 다른 과제는 loco100%다. 직접 실행 인자는 `--eval_push_rsi`이고 이 task의 `--test --eval`에서만 허용한다. 평가 모드를 유지하고 학습 업데이트·기존 네 과제 RSI 캐시 생성은 하지 않는다. 옵션을 생략하면 기존 loco 평가다.

학습 시 기존 Stage1의 크기별 RSI 캐시를 재사용하고, 부족한 크기는 동일한 물리 screening으로 생성한다. Stage2 추가 캐시는 `env.stage2JointPush.solo_rsi_cache_directory`인 `output/rsi_cache_stage2_jointpush_rsi`에 저장한다. 기존 캐시가 포괄하지 않는 크기가 많으면 첫 학습 초기화가 오래 걸릴 수 있다. 이후 같은 크기/계약은 저장된 캐시를 재사용한다.

Stage1 PUSH checkpoint의 전이·제로샷은 기존 Stage2와 같다. RSI variant와 loco variant의 일반 resume 계약은 구분한다.

## 검증 범위

GPU5·1024환경에서 실제 무작위 PUSH 크기로 단독/coupled RSI reset, 고정 상자/목표·참조 관절·그룹별 초기화·AMP·정책 추론을 확인했다. 평가8종은 실제 무작위 크기로 모두 loco100%임을 확인했다.

기존 네 과제의 RSI 및 혼합 학습 초기화 검증은 이미 screening된 Stage1 캐시 크기를 물리 상자에도 동일하게 적용한 fixture를 사용했다. 실행용 config의 크기 분포는 변경하지 않았다. 전체 무작위 크기에 대한 새 캐시 생성과 PPO 업데이트·장기 학습 성능은 검증하지 않았다. 학습 경로 검사에서는 학습 루프를 검사 함수로 대체하여 optimizer step을 실행하지 않았다. 결과는 `output_etc/check/stage2_jointpush_rsi/`에 있다.
