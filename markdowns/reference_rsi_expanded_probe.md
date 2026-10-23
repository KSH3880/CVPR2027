# XYZ 0.20~0.65m 확장 RSI 충격 검사

2026-10-02 사용자 최종 요청으로 GPU 6 검사를 실행했다. 초기 요청의 XY만 0.2m,
Z 0.25m 하한 대신 **최종 XYZ 모두 0.20~0.65m**를 적용했다.

- 5cm 간격, 10×10×10=1000개 크기 조합.
- sit/climb/pickUp/carryWith/putDown 모두 포함. carry를 제외한 검사가 아니다.
- 원본 참조 모션 후반 65~98%, distill 정책 제어, measurement_version=2.
- 2048환경×128회 reset×20 control step, seed 0. 학습하지 않는다.
- 기존 252개/32회 검사와 비슷한 표본 밀도를 위해 reset을 약 4배로 늘렸다.
  무작위 accepted 상태 검사이며 모든 조합을 같은 수로 강제 검사하지 않는다.
- 전용 env config에서 baseSize=0.4m, scaleRangeXYZ=[0.5,1.625], interval=0.125.
- 박스 두 개 최대 합산 높이 1.30m를 허용하도록 전용 maxTopSurfaceHeight=1.30.
- 기존 학습 설정·프로세스와 ksh는 변경하지 않는다.

## 실행과 결과 경로

전용 입력/실행/원시 결과는 `../../runs/reference_rsi/impact_xyz20_65_seed0/`.
`expanded_env.yaml`, `manifest.json`, `run.sh`, `launcher.pid`, `run.log`를 보관한다.
완료 시 summary.json/agents.jsonl/scenes.jsonl/by_size_skill_phase.csv가 생성된다.
summary.json 이전에는 완료로 해석하지 않는다.

사용자 요청에 따라 이미 실행했으므로 아래 명령을 중복 실행하지 않는다.

```bash
bash /home/hwanhee/koo_cvpr/runs/reference_rsi/impact_xyz20_65_seed0/run.sh
```

최신 로그 확인:

```bash
tail -n 30 /home/hwanhee/koo_cvpr/runs/reference_rsi/impact_xyz20_65_seed0/run.log
```

`probe_reference_rsi.sh`에 `RSI_PROBE_CFG` 입력 override를 추가해 전용 config를 사용한다.
기본 학습 YAML은 수정하지 않는다. GPU 선택 wrapper는 6/7만 허용하고 이번 실행은 6이다.

## 해석 제한과 후속 분석

### 완료 분석 (2026-10-02)

정상 완료: 262,144 scenes, NaN 0, partial reset 보존 통과. 전체 분석은
[ANALYSIS.md](../../runs/reference_rsi/impact_xyz20_65_seed0/ANALYSIS.md),
구간·축·정확한 크기별 집계는 같은 디렉터리의 `analysis_*.csv`에 있다.
첫 Δv는 `max(||v1-v0||, ||v2-v1||)`이다. scene summary의 20step 전체 Δv와 구분한다.

| 행동 | agent 표본 | 첫 Δv>2m/s | 90~98%에서 Δv≤2 관측 크기 |
| --- | ---: | ---: | ---: |
| pickUp | 50,164 | 13.29% | 1000/1000 |
| carryWith | 130,412 | 24.70% | 943/1000 |
| putDown | 18,834 | 9.13% | 936/1000 |
| sit | 12,363 | 25.73% | 693/1000 |
| climb | 50,647 | 30.15% | 963/1000 |

65×65×65cm carryWith는 80~90% 구간에서 23/25개가 Δv>2였고, 90~98%에서는 4/21개였다.
최대 크기에도 낮은 충격의 후반 참조 상태가 있지만 소수 관측으로 안전을 보장하지 않는다.
최대 크기 sit의 80~98%는 미관측이며 불가능하다고 결론 내리지 않는다.

한국어 해석: 최대 크기 carryWith도 후반 시점에 따라 차이가 커서 크기를 줄이기 전에
크기별 클립·시점 선별을 시도할 근거가 있다. 작은 박스도 항상 낮은 충격은 아니다.
Y=20cm carryWith의 첫 Δv>2 비율은 21.58%, Y=40cm는 3.50%, Y=65cm는 75.68%
(다른 축과 모션 합산). Z=60/65cm의 climb은 57.09/55.40%, sit은 53.83/51.64%로,
높이를 늘리면 이 행동들이 더 쉬워진다는 가정은 이 검사에서 뒷받침되지 않는다.

새 예비 후보: `../../runs/reference_rsi/impact_candidates_xyz20_65_v1.json`.
203,375개 = preferred(Δv≤1) 137,008 + conditional(1<Δv≤2) 66,367.
원본 참조 motion ID/phase만 저장하며 teacher rollout 상태를 사용하지 않는다.
학생 반복 재검사를 거치기 전에는 학습 sampler에 넣지 않았다.

distill 정책 전달 당시 크기 범위는 XY 40~65cm / Z 25~55cm였다. 새 크기는 일부 정책
학습 범위 밖이다. 첫 순간의 물리 충격과 이후 정책 반응을 함께 기록하지만 성공률이나
학생 수렴의 한계로 단정하지 않는다.

낮은 Δv라도 작은 박스가 손에 닿지 않아 떨어질 수 있다. 손 거리 변화·박스 속도·수직 이동·
회전을 함께 확인하고, 정상 putDown 손 놓기와 불안정 carry를 구분한다. 정상 sit/climb에
carry의 손 거리 기준을 적용하지 않는다. 높이 증가 효과도 size×skill×motion×phase로
분리한다. 후보표의 1cm 기하 screen은 이 검사에서 사용하지 않는다.

1000개 크기를 기준으로 실제 관측 수와 미관측 조합을 집계했다. 기존 252크기 보고서와
구분해서 보관하며 확장 후보 학생 재검사 명령은 [재검사 가이드](reference_rsi_expanded_candidates.md)에 있다.
