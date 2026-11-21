# 지연 대응 평가 — 512환경 × 3회

저장일: 2026-11-15 (KST). **0초 → 환경별 랜덤 1–5초**. 숫자는 **3 seed 평균 ± 표본표준편차**다.

## 실험 설정

- Enabling: AT→SIT / CLIMB / ONTOP, 각 환경은 선행자·후행자 2명.
- Coupled: 공동 PUSH 2 / 3 / 4명. 해당 인원만 사용하며 추가 agent/padding 없음.
- 회차당 Enabling 512개, Coupled 512개. 각 종류를 유형별 170/171/171개로 나눠 실행하고 seed마다 170개 유형을 순환한다.
- Seeds 1/6/7, episode 900step(30초). 각 모델·지연 조건의 세부 유형은 3회 합계 정확히 512개.
- 5모델 × 2종류 × 2조건 × 3회 = 60개 합산 조건, 내부 실행 180개, 총 30,720 episode.
- 0초도 새로 평가했다. 초기 사람 속도0, 같은 초기 장면/첫 정책 입력, 모델 간 동일한 환경별 지연 schedule을 검증했다.
- Enabling은 선행자, Coupled는 환경번호 modulo 인원수의 한 agent를 지연한다. 지연은 Uniform[1,5)초 표본을 1/30초 단위로 반올림하며 관계 구조는 유지한다.
- 신규 학습 없이 기존 checkpoint를 사용했다. 정확한 경로·SHA는 [실험 manifest](main/manifest.json)에 있다.

## Enabling — 주요 지표

각 seed의 512개를 합쳐 집계한 뒤 3 seed 평균을 계산했다. 준비율·사전점유율은 AT 성공 표본만 포함한다.

| 모델 | Group SR(%) | 준비율(%) | 사전점유율(%) | AT SR(%) |
|---|---:|---:|---:|---:|
| TokenHSI 40.5k | 10.55 ± 0.68 → 4.43 ± 0.60 | 70.57 ± 3.47 → 70.85 ± 2.25 | 30.09 ± 1.88 → 35.05 ± 0.20 | 54.36 ± 0.69 → 51.17 ± 0.78 |
| SCT 26k | 31.71 ± 1.30 → 22.07 ± 0.68 | 55.87 ± 3.99 → 65.30 ± 2.04 | 22.93 ± 2.61 → 39.17 ± 2.54 | 76.69 ± 2.16 → 67.97 ± 2.44 |
| SCT+FSM 26k | 42.71 ± 2.99 → 42.71 ± 3.28 | 0.00 ± 0.00 → 0.00 ± 0.00 | 0.00 ± 0.00 → 0.00 ± 0.00 | 83.79 ± 1.79 → 83.01 ± 1.37 |
| SCT+CoopFT 7k | 54.82 ± 3.21 → 55.53 ± 1.08 | 13.28 ± 0.95 → 21.42 ± 1.79 | 0.05 ± 0.08 → 0.30 ± 0.15 | 97.20 ± 0.81 → 96.29 ± 0.39 |
| DCT V4 2k | 82.55 ± 1.19 → 80.47 ± 1.56 | 19.57 ± 0.25 → 29.28 ± 1.80 | 0.03 ± 0.03 → 0.59 ± 0.23 | 97.92 ± 0.11 → 95.64 ± 0.79 |

## Enabling — AT 직전 거리와 Post-AT 시간

**이 보조 표는 각 유형의 조건부 평균 → 세 유형 동일 가중 → 3 seed 평균**이다. 위 주요 표의 512개 pooled 방식과 구분한다. 각 유형별 수치와 성공 표본 수는 아래 상세 보고서에 있다.

| 모델 | AT 완료 직전 거리(m) | Post-AT Completion Time(s) | Group 성공 수 / 1,536 |
|---|---:|---:|---:|
| TokenHSI 40.5k | 1.25 ± 0.13 → 1.24 ± 0.04 | 2.44 ± 0.39 → 2.77 ± 1.09 | 162 → 68 |
| SCT 26k | 2.48 ± 0.22 → 1.82 ± 0.12 | 4.12 ± 0.42 → 3.55 ± 0.30 | 487 → 339 |
| SCT+FSM 26k | 5.71 ± 0.16 → 5.69 ± 0.19 | 7.18 ± 0.32 → 7.18 ± 0.32 | 656 → 656 |
| SCT+CoopFT 7k | 4.81 ± 0.09 → 4.25 ± 0.11 | 5.58 ± 0.25 → 4.71 ± 0.23 | 842 → 853 |
| DCT V4 2k | 4.49 ± 0.12 → 3.84 ± 0.04 | 4.95 ± 0.11 → 4.28 ± 0.04 | 1268 → 1236 |

## Coupled

각 seed의 512개를 합쳐 집계한 뒤 3 seed 평균. t_coop/Jerk는 아래 유효 표본 규약을 유지한다.

| 모델 | Group SR(%) | t_coop(%) | Jerk(m/s³) |
|---|---:|---:|---:|
| TokenHSI 40.5k | 63.80 ± 2.44 → 52.73 ± 0.90 | 48.37 ± 1.05 → 35.81 ± 0.33 | 19.57 ± 0.39 → 14.85 ± 0.53 |
| SCT 26k | 75.00 ± 1.37 → 67.64 ± 0.41 | 54.79 ± 2.12 → 43.71 ± 0.95 | 23.10 ± 1.46 → 22.53 ± 1.06 |
| SCT+FSM 26k | 75.00 ± 2.20 → 73.70 ± 0.96 | 62.42 ± 2.19 → 65.01 ± 0.60 | 31.46 ± 0.52 → 30.20 ± 0.27 |
| SCT+CoopFT 7k | 83.27 ± 1.30 → 80.21 ± 1.98 | 82.83 ± 0.62 → 72.49 ± 0.98 | 14.49 ± 0.79 → 11.42 ± 0.21 |
| DCT V4 2k | 92.45 ± 2.15 → 86.85 ± 1.19 | 86.37 ± 0.59 → 75.38 ± 1.73 | 13.83 ± 0.10 → 11.71 ± 0.21 |

## 지표 정의와 포함 표본

- **Group SR**: 전체 episode 중 최초 동시 그룹 성공 비율. Enabling은 AT와 후행 성공 조건이 같은 frame에 만족돼야 한다.
- **준비율**: 100×[d(0)−d(t_AT 직전)]/max[d(0), 1e−6]. d는 후행 root XY→선행 AT 배치 목표 XY의 거리. 음수 보존. AT 성공 전체 표본에서 계산한다.
- **사전점유율**: 초기 snapshot부터 AT 직전까지 후행 root 중심 반경0.35m 원이 예정 배치 footprint(초기 yaw, 물체 크기, 각 축 margin0.10m)와 겹친 frame 비율. AT 성공 전체 표본. 실제 접촉/방해의 직접 판정은 아니다.
- **AT 완료 직전 거리**: 최초 AT 성공 step k의 직전 k−1에서 후행 root XY→AT 배치 목표 XY 거리(m). k=1이면 초기 snapshot. 1m offset 없음. AT 실패 제외, Group 실패와 후행 낙상은 실제 위치로 포함한다.
- **Post-AT Completion Time**: (최초 동시 Group 성공 step−최초 AT 성공 step)×기록 control_dt(약1/30초). **Group 성공 episode만 포함**. 선행만 성공/미완료/실패 제외, 실패시간 대입 없음. 성공0개 유형은 N/A이며 해당 동일 가중 평균도 N/A다.
- **t_coop**: 기존 규약 유지. 모든 참여자의 양손이 상자 표면12cm 이내인 frame 수(전체 기록에서 누적)를 상자 XY 출발변위20cm 도달부터 최초 성공/종료까지의 counter 길이로 나눈다. 1 초과는 clip, 비유효값 제외. 분자·분모 시간 범위가 다르므로 해당 활동 구간 내 순수 비율로 해석하지 않는다.
- **Jerk**: 상자 둘레64점의 회전 포함 world XYZ 위치를 dt=0.0333으로3회 차분한 절댓값 평균. 기존 출발/종료 mask와0값 episode 제외 규약 유지. 실패/정지 때문에 낮아질 수 있으므로 성공률과 함께 본다.
- 준비율이 높거나 거리가 작다는 것만으로 적절한 대기를 의미하지 않는다. 사전점유율과 Group SR을 함께 해석한다.
- Post-AT 시간은 모델별로 성공 표본이 다르므로 짧은 조건부 시간을 전체 성능 우위로 해석하지 않는다.
- TokenHSI는 원본397D 자기/담당과제 입력을 유지한다. 동일한 전체 scene attention 입력 모델로 해석하지 않는다.
- 표준편차는3회 seed 평균의 표본표준편차이며 신뢰구간/유의성 검정이 아니다.

## 저장 파일

- [tables.csv](tables.csv): 모든 유형·평균·모델·조건·지표의 mean/std, 단위와 집계 방식을 명시한 긴 형식 CSV.
- [주요 지표 전체 표](main/report.md) / [JSON](main/summary.json): 유형별·512개 pooled·세 유형 동일 가중 보조값.
- [거리 전체 표](at_distance/report.md) / [JSON](at_distance/summary.json) / [episode CSV](at_distance/trials.csv).
- [Post-AT 전체 표](post_at_time/report.md) / [JSON](post_at_time/summary.json) / [episode CSV](post_at_time/trials.csv).
- `main/analysis/`:180개 조건의 검증된 JSON 및 episode 배열 NPZ. `main/schedules/`:36개 환경별 지연 schedule.
- [최종 검증](main/completion_verification.json), [표본·성공률 독립 검증](main/final_coverage_audit.json), [거리 검증](at_distance/verification.json), [Post-AT 검증](post_at_time/verification.json).
- [보관 manifest](archive_manifest.json): 복사 원본·SHA 및 보관 파일 목록. 대용량 원시 rollout은 아래 원본 경로에 남겨뒀다.

## 원본 경로

- 주요 평가: `/home/hwanhee/ksh/stage1_2_base_teacher/output_etc/check/dct_random_delay_family512_seeds167_20261115`
- 거리 재집계: `/home/hwanhee/ksh/stage1_2_base_teacher/output_etc/check/at_pre_distance_20261115/family512`
- Post-AT 재집계: `/home/hwanhee/ksh/stage1_2_base_teacher/output_etc/check/post_at_success_only_20261115/family512`

이 보관본은512환경·3회 평가만 포함한다. 앞선64환경 파일럿이나0/2/4초 결과와 혼합하지 않는다.
