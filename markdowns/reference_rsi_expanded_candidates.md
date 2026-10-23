# 확장 크기 RSI 후보와 학생 재검사

2026-10-02. 목표는 XYZ 0.20~0.65m 크기를 유지하면서 원본 참조 RSI의 쓸 만한 후반 클립·시점을 선택하는 것이다. 완료 distill 검사는 [확장 분석](../../runs/reference_rsi/impact_xyz20_65_seed0/ANALYSIS.md)에 있다.

## 이번에 준비된 것

- 예비 후보: `../runs/reference_rsi/impact_candidates_xyz20_65_v1.json`
- 203,375개: pickUp 43,498 / carryWith 98,203 / putDown 17,114 / sit 9,182 / climb 35,378.
- preferred(첫 Δv≤1m/s) 137,008개, conditional(1<Δv≤2) 66,367개.
- 실제 크기와 원본 motion ID/phase를 저장한다. teacher의 rollout 자세를 학습 초기 상태로 쓰지 않는다.
- 학생 검사 설정: `../runs/reference_rsi/impact_xyz20_65_seed0/student_expanded_env.yaml`.
- 학생의 기존 full AMP 설정에서 box.build 및 최대 스택 높이(1.30m)만 확장했다. relationReward/RSI와 checkpoint 계약은 그대로다. 새 size_progress 보상을 섞지 않는다.
- 기존 후보 및 결과는 보존했다. 확장 학생 검사는 아래 완료 분석을 참고한다.

## 학생 검사 완료 분석

2048환경×256회×20step 정상 완료. NaN 0, 부분 reset 보존 통과,
동일 student16500 SHA `1a5804219cb3351e9af4188992e9b26ea453258f455e1a1b45e5d525012b53f9`.
원시 결과·전체 크기/구간/축별 CSV는
[학생 분석](../../runs/reference_rsi/student_candidates_xyz20_65_seed1/ANALYSIS.md)에 있다.

203,375개 중 166,480개 관측, 36,895개 미관측. n≥3 후보 76,812개 중
모든 관측에서 첫 두 step Δv≤2m/s인 후보는 62,949개다.

| 행동 | 반복 저충격 후보 | 후보가 있는 크기 | 90~98% 후보가 있는 크기 |
| --- | ---: | ---: | ---: |
| pickUp | 14,063 | 997/1000 | 932/1000 |
| carryWith | 30,959 | 948/1000 | 883/1000 |
| putDown | 5,537 | 936/1000 | 712/1000 |
| sit | 2,974 | 905/1000 | 329/1000 |
| climb | 9,416 | 962/1000 | 572/1000 |

정확히 65×65×65cm의 후보는 pickUp/carryWith/putDown/sit/climb 각각
6/22/6/3/3개다. carryWith의 22개 중 손 거리 증가가 모든 관측에서 20cm 이하인 것은
20개지만 이 threshold가 실제 파지 성공을 보장하지는 않는다.
같은 최대 크기 carryWith 90~98%의 Δv>2는 7/84=8.33%, 손 거리 증가>20cm는
4/84=4.76%. 80~90%는 4회만 검사돼 모두 Δv>2였으며 전체 구간 불가능 판정이 아니다.

기존 최대 65×65×55cm carryWith 90~98%는 Δv>2=42.65%, 손 거리 증가>20cm=57.35%
(n=68). 크기 단조 증가가 위험을 설명하지 않으며 정확한 크기와 원본 시점별로 선택해야 한다.

Δv≤2인 climb 관측에서 root 높이 감소>20cm는 49.68%; Z=65cm에서는 82.11%다.
원본 동작의 정상 하강과 실패를 구분할 높이/발의 상판 상대 오차·agent 성공 유지 정보가
현재 결과에 충분하지 않다. 따라서 climb의 후반 후보는 낮은 Δv만으로 성공 유지까지
검증됐다고 보고하지 않는다. pickUp의 박스 상승과 putDown의 몸을 숙임·손 놓기도
해당 행동의 정상 동작일 수 있다.

검토용 `student_low_impact_candidates.json`을 결과 디렉터리에 저장했다.
원본 ID/phase와 모든 후보별 student 관측을 보존하며 학습 sampler에는 아직 적용하지 않았다.
파일의 source tier는 teacher 관측 기준이다. 향후 학습 가중치는 student_first2_delta_v_max
등 학생 검사 결과를 기준으로 별도로 정해야 한다.

이후 적용은 크기 우선 추첨을 유지하고 같은 크기·행동에서 후보를 선택한다.
carry/pickUp/putDown의 90~98% 편향은 후보가 있는 조합에서 시도할 수 있지만,
sit/climb은 후반의 coverage가 낮고 정상 수행 여부도 추가 확인해야 한다.
크기/행동 후보가 부족한 경우 같은 크기의 덜 늦은 후보를 먼저 고려하고,
미측정 조합은 작은 크기나 loco로 조용히 대체하지 않는다.
현재 학습/RSI와 새 progress 설정은 이 분석 때문에 변경하지 않았다.

## 학생 검사 명령

이전 v2 검사와 같은 16500 snapshot을 사용해 크기 확장과 후보 변경을 비교한다. 256회는 이전 252크기·64회 검사와 비슷한 크기당 표본 밀도를 목표로 한 설정이다. 모든 후보가 반복 측정되는 것을 보장하지 않는다.

```bash
cd /home/hwanhee/koo_cvpr/stage1_fix
TOKENHSI_GPU=6 SEED=1 \
RSI_PROBE_ENVS=2048 RSI_PROBE_ROUNDS=256 RSI_PROBE_STEPS=20 \
RSI_STUDENT_CFG=/home/hwanhee/koo_cvpr/runs/reference_rsi/impact_xyz20_65_seed0/student_expanded_env.yaml \
RSI_PROBE_OUTPUT=/home/hwanhee/koo_cvpr/runs/reference_rsi/student_candidates_xyz20_65_seed1 \
bash tokenhsi/scripts/multi_agent/probe_student_candidates.sh \
 /home/hwanhee/koo_cvpr/runs/reference_rsi/student_candidates_seed1_v2/student_snapshot.pth \
 /home/hwanhee/koo_cvpr/runs/reference_rsi/impact_candidates_xyz20_65_v1.json
```

학습·optimizer 업데이트 없이 20step 물리 검사를 반복한다. checkpoint를 출력 디렉터리에 고정하고 measurement_version=2를 사용한다. wrapper는 GPU 6/7만 허용한다. 같은 출력 폴더의 검사 중복 실행은 피한다.

`reference_bank_mesh.npz`는 이번 예비 추출에서 참조 데이터 fingerprint 용도로만 사용한다. 그 파일의 기존 252크기 기하 통과표로 새 후보를 필터링하지 않는다. replay는 후보의 실제 크기 키로 선택한다.

## 결과를 반영할 방향

1. 크기를 먼저 원래 랜덤 분포로 결정한다. 후보 수가 많은 크기가 더 자주 선택되도록 바꾸지 않는다.
2. 같은 크기·행동의 원본 참조 후보 중 학생 반복 검사에서 충격이 낮은 클립·시점을 고른다. 기존 probe replay의 preferred 가중치 2, conditional 1은 초기 비교 기준이며 학습 최적값은 아니다.
3. carryWith/pickUp/putDown은 90~98%가 평균적으로 낮은 충격이지만 크기·클립마다 예외가 있다. sit/climb은 늦을수록 전체 충격 비율이 높아 일괄 후반 편향을 적용하지 않는다.
4. carryWith의 손 거리 증가·수직 상승·회전·넘어짐을 Δv와 함께 본다. putDown의 정상 손 놓기를 carry 파지 실패 기준으로 제외하지 않는다. 어느 threshold도 아직 성공/실패의 검증된 경계는 아니다.
5. 후보가 부족하거나 미측정인 조합은 추가 검사 대상으로 표시한다. 크기를 줄이거나 loco로 조용히 대체해서 후보 범위를 넓혔다고 보고하지 않는다.

학생 반복 검사는 완료됐으나 검토용 저충격 표를 아직 학습 sampler에 적용하지 않았다.
낮은 초기 충격은 동작 성공이나 수렴 가속을 보장하지 않는다. 학습 반영 시 보상 변경 효과와
RSI 변경 효과도 각각 비교해야 한다.
