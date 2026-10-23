# PUSH 직육면체·동일 질량·기울기 검사

XYZ각1.2~1.5m에서 양손 **중점**이 뒷면 지정점을 향하도록 하는 분석이다. 크기와 질량을 독립으로 바꾸며, 동일 질량 후보는 모든 크기에 같은 실제 질량으로 적용한다. 학습 설정이나 CONTACT 보상 변경은 포함하지 않는다.

사본 루트에서 재실행한다. GPU6 전용이며 기존 프로세스를 종료하지 않는다. 평가 환경 수는 접촉체최대4185/실제humanoid2835로 학습2048과 별도다. 출력 경로가 이미 있으면 중단한다.

```bash
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/analyze_push_rect_stability.sh
```

기본 출력은 `output_etc/check/push_rect_stability_<날짜시각>/`. `OUTPUT_PATH=output_etc/check/<새이름>`으로 지정할 수 있다. 결과만 다시 집계할 때는 runtime_env로 tokenhsi 환경을 준비한 뒤 `python tokenhsi/scripts/multi_agent/report_push_rect_stability.py <검사폴더>`를 실행한다.

2026-10-19 결과는 [REPORT.md](../../output_etc/check/push_rect_stability_20261019/REPORT.md), [그래프](../../output_etc/check/push_rect_stability_20261019/stability.png), [수치/검증](../../output_etc/check/push_rect_stability_20261019/summary.json)에 있다.

- 상자0.6·바닥1.0, 10/15/20/25/30kg, 27크기 조합, 손점Z30~65%·2힘한도·3양손조건의 유한접촉체 검사. 저점30~50%Z는4050/4050회 2°이하·0.5m이상 이동, 최대0.131°. 높은점55~65%Z 일부에서전도(최대95.226°).
- 20kg·바닥기준0.65m 추가검사: 각축0.05m간격343크기×6조건=2058/2058회통과, 최대0.0873°·이동0.818~0.849m. 무구동대조343회통과. 실제humanoid균형이 없는 구체접촉체 결과다.
- 실제21개PUSH AMP선택초기자세+전신PD/중점IK의28350회 검사에서는 유효조건+2°이하+0.1m이상 이동7회(높이비율sweep0/22680, 고정0.65m7/5670). 제어기균형실패가많으며학습정책성공률로해석하지않는다. 개발중quick검사는별도보존한다.

기울기가 우선이면 20kg·뒷면중앙선바닥0.65m를 **후속 실험 시작값**으로 권고한다. 최적질량/안정정책검증완료가 아니다. 중앙COM·수평준정적힘에서 전도/미끄럼 임계힘비는 Y/(2hμ)로 질량이 상쇄되므로 무게만늘려전도를막을수없다. 1~5m장거리·연속크기전체·임의하중·학습중무기울기는미검증이다.

`test_push_stability_metrics.py`는 yaw와무관한기울기·작은각도/전도·질량상쇄측정을검사한다. 보고서는trace의각도/최종변위를재계산하고실제질량을감사한다. 데이터·원본보관본·학습YAML·checkpoint는변경하지않는다.
