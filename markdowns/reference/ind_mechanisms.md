# IND 성능 차이 분석 · SCT 26k / Ours V4 2k

512환경·seed 1/6/7·900step·평균 행동·고정 RMS·첫 episode·XY100%를 사용한다. 기존 Table 1과 결과를 합치거나 공식 성공 조건을 바꾸지 않는다. 출력은 `output_etc/check/ind_mechanisms_v4_2000/`다.

## 실험과 주장

| 실험 | 구성 | 확인할 주장 | 신규 본 평가 |
| --- | --- | --- | ---: |
| A | 기존 2H IND의 사람 A/B를 각각 1H로 추출, SCT/Ours 비교 | 단독 수행 차이와 공유 공간에서의 추가 실패 분리 | 12회 |
| B | 두 CARRY_AT 과제의 분리/90도 교차 경로 | 작업 경로가 겹칠 때 성능 감소가 작은가 | 12회 |
| C | A의 단독 기록 CPU 분석 | 목표 정밀도·연속 성공 유지가 개선되는가 | 0회 |

A는 원본 2H snapshot의 root/DOF/몸체 상태·속도·목표·작업 상자·받침·물성을 유지한다. N=1/M=2로 바꾸며 사람별 원래 primary/support 토큰을 보존한다. 관측용 distractor는 물리에서 제거하고 attention에는 유지한다. 실제 첫 관측은 원래 2H 관측에서 해당 사람·물체·목표를 추출한 값과 exact 비교한다. 원본 첫 관측 복원에 필요한 몸체 상태는 같은 seed를 초기화만 다시 실행해 보충하고, 기존 initial/observed snapshot과 exact 비교한다.

B는 접근 1–5m, 운반 3–5m를 장면별 추출하고 두 사람의 계획 경로 길이를 맞춘다. 교차는 운반 중점에서 ±90도, 분리는 평행한 경로 중심 8m 간격이다. 전체 yaw와 좌우 방향을 무작위화한다. 개인 자세·속도·상자 yaw·크기·질량·거리·관측용 distractor의 개인 기준 배치를 보존하며, 시작 사람/상자 겹침을 검사한다. 경로 교차만 의도적으로 허용하는 별도 평가 분포다. GPU world/local 좌표 변환의 XY 오차는 0.1mm 이내이며 회전·속도·DOF·목표 등은 exact로 확인한다. 두 모델 간 초기 상태와 실제 첫 관측은 exact여야 한다.

B 개발 기하는 seed101로 생성하고 정식 표에서 제외한다. 개발의 motion/asset pool은 capture seed1을 재사용하므로 완전히 독립된 개발 표본은 아니다. 정책 성능을 기준으로 장면을 선택하지 않는다. 초기 개발 장면은 `pilot/scene_generation_v1/`에 보존하고, 두 조건의 관측용 distractor 상대 배치를 통일한 revision2를 정식 배치로 고정한다.

## 지표와 해석

- 공식 개인 ever SR, 2H의 두 사람 모두 ever 성공, 성공 전 낙상, Time Succ./All(실패는 제한시간).
- C: AT/PUSH의 목표 10/20/30/50cm 성공률, 50cm 안에 들어왔지만20cm 미도달 비율. 전체 시도를 분모로 쓴다.
- 연속 성공0.5/1초 유지율은 보조지표다. 공식 ever 성공 판정은 바꾸지 않는다.
- B는 몸체 중심 간 최소거리와 몸체 중심–상대 상자 표면 최소거리를 저장한다. 사람 근접10/20/30cm, 상자 근접6/12/20cm를 최초 성공/낙상까지 집계한다. 이는 mesh 접촉·접촉 쌍 충돌률이 아니다.
- B의 이동거리는 최초 성공/낙상까지 계산한다. 각자의 계획 운반 중점 반경1.5m 진입률과 구간 내 속도/저속(<0.2m/s) 비율을 저장하며 속도·저속 비율은 진입한 trial에 조건부다.
- seed별 결과→평균/표본std. 차이의95% CI는 seed별 층화, 원본 장면 단위2,000회 bootstrap이며 같은 장면의 두 사람을 함께 뽑는다.
- A: 단독 ours−SCT, 2H ours−SCT, `(ours_2H−ours_solo)−(SCT_2H−SCT_solo)`.
- B: `(ours_cross−ours_separated)−(SCT_cross−SCT_separated)`가 양수이면 교차에 따른 감소가 ours에서 작다.

단독 조건은 토큰 수도 달라져 순수 물리 충돌 제거 실험이 아니다. B는 CARRY_AT 전용으로 Table 1 전체를 대체하지 않는다. 고정 checkpoint 비교이며 학습 seed 변동이나 협력 학습의 인과적 전이를 증명하지 않는다.

## 실행 구조

`ind_mechanisms_capture.py`는 기존 reset 재현, `ind_mechanisms_scenes.py`는 단독/교차 장면 생성, `ind_mechanisms_eval.py`는 프로세스 내부 평가 adapter다. 기존 학습 코드·checkpoint를 수정하지 않는다. SCT는 기존 native Stage1 행동/정규화 동치 검사를 재사용한다.

`run_ind_mechanisms.py`는 GPU6/7만 사용한다. 전체 사용량+진행 중 작업의 미사용 예약+신규 예약이70,000MiB 이하일 때 시작하며 GPU별 최대3개다. 최초 작업 예약은6,144MiB, OOM은 원래512환경을 유지해 더 큰 예약으로1회 재시도한다. 이는 새 작업의 시작 제한이며 다른 사용자의 메모리 증가를 강제로 제한하지 않는다.

A/B는 각각 `a_ready.json`/`b_ready.json`이 있어야 시작한다. 초기 상태·CPU 채점 실패는 새 배정을 멈추며 이미 실행 중인 작업은 보존한다. 검증 완료 결과는 재사용하고 미완료/실패 시도는 별도 폴더에 보존한다. `status.json`, `scheduler.log`, 작업별 `execution.json`에 상태·GPU·PID·시간·프로세스 피크 메모리를 기록한다.

`ind_mechanisms_develop.py`는 A의 미배정 작업이 없어질 때까지 기다린다. 이후 기존 A 작업의 시작 예약도 반영하면서 개발 seed101의 두 조건×두 모델4회를 실행하고, 모델 간 초기 상태/첫 입력 exact·CPU 검증 후에만 `b_ready.json`을 만든다. 개발 실행은24회 본 평가에 포함하지 않는다.

`ind_mechanisms_metrics.py`는 독립 CPU 재채점·시간/유지/근접 지표를 계산하고, `summarize_ind_mechanisms.py`는 모델 간 입력/초기 상태 exact 대조·통계·`report.md`·`summary.json`·정밀도 PDF를 만든다. 기존2H 분석은 원본 trace/initial을 읽기 전용 링크하고 별도 복사한 result를 사용해 원본 파일을 쓰지 않는다.

실행 명령은 [USAGE](../USAGE.md#ind-성능-차이-보조-평가), 현재 결과는 [보고서](../../output_etc/check/ind_mechanisms_v4_2000/report.md), 진행 상태는 [status](../../output_etc/check/ind_mechanisms_v4_2000/status.json)를 본다.

2026-11-14에 본 평가24회와 개발4회를 완료했다. 최종 검증은 [completion_verification.json](../../output_etc/check/ind_mechanisms_v4_2000/completion_verification.json), 초기 경로 예시는 [layout_examples.pdf](../../output_etc/check/ind_mechanisms_v4_2000/layout_examples.pdf), 단독 정밀도 곡선은 [solo_precision.pdf](../../output_etc/check/ind_mechanisms_v4_2000/solo_precision.pdf)다. 수치와 해석은 보고서·changelog에 둔다.

## 2인 SIT 교차 파일럿

`run_sit_crossing_pilot.py`는 별도64환경·seed1의 loco 초기 상태를 사용해 SCT26k/Ours V4 2k × 분리/교차4회를900step 평가한다. 출력은 `output_etc/check/sit_crossing_v4_2000_env64_seed1/`이며 기존 CARRY 실험과 합치지 않는다. GPU6=SCT/GPU7=Ours로 각 조건을 순차 실행하고, GPU 사용량+6,144MiB가70,000MiB 이하일 때 시작한다. 자식마다 해당 GPU의 MPS 연결을 새로 탐색한다.

장면마다 사람→자기 SIT 물체 중심 수평 거리D를3–5m 균등 추출하여 두 사람에게 동일하게 적용한다. 시작점과 목표는 경로 중점에서 각각D/2이며, 교차는±90도·분리는 평행 경로8m 간격이다. 전체 yaw와 물체의 상대 yaw는 무작위화한다. SIT용 물리 bank의 크기·질량과 loco 자세·속도를 보존한다. 관측용 distractor와 비활성 goal token도 각 그룹과 함께 옮긴다. 따라서 물리 간섭뿐 아니라 상대 장면 맥락도 달라지며, 작업 정보 사용의 인과 효과를 직접 측정하지 않는다.

중점 반경0.8m를 교차 구간으로 정의한다. `summarize_sit_crossing_pilot.py`는 공식SR·성공 전 낙상·1초 유지·완료 시간·양쪽 성공, 두 사람 모두 성공/낙상 전의 몸체 중심20cm 근접, 구간 진입 시점·동시 체류, 자기 물체 중심1m 접근 여부/시간을 저장한다. 근접은 실제 접촉 쌍 충돌률이 아니며, 목표1m 도달은 접근 보조지표다. 초기 상태/첫 정책 입력의 모델 간 exact 일치, scene/checkpoint 보존, CPU 재채점을 검사한다. 단일seed 파일럿으로 정식 성능 주장을 하지 않는다.
