# Y1.8m PUSH 손 높이 비교

`stage1_push_depth180_safe_h065_relaxed`와 `stage1_push_depth180_safe_hamp_relaxed`는 Y1.8m safe 기반의 별도 scratch 실험이다. X1.2~1.5m/5cm·Y1.8m·Z1.3m·20kg/마찰0.6, 사람2/상자4/634D·접근1~5m·목표+Y1~4m·native RSI50%/후반70%·원본 관절/AMP·기존 AT와 paired 보상 포화를 유지한다. AT 성공 기울기 제한은 추가하지 않는다.

## State와 성공

```text
state = exp(-4.682689584792 * 수평오차² - (높이오차 / 0.154039131233)²)
CONTACT 자체 성공 = 같은 물리 substep에서 state > 0.9
                   AND 양손 각각의 면 법선 반력 >= 0.5N
                   AND 상자 면 법선 반력 >= 0.5N
```

`xy_reward_scale`, `z_reward_sigma_m`은 연속 state 감쇠 계수다. 별도 수평 15cm·높이±5cm 성공 gate는 제거했다. 수평 또는 높이 오차만 있는 경우 state=0.9 경계가 각각 약15cm/5cm이며 두 오차가 함께 있으면 더 엄격해진다. 등호는 성공이 아니다. force/state 성공은 제어 step 안에서 누적한다. CONTACT 보상은 현재 AT 성공으로도 paired 포화될 수 있으나 자체 성공 로그와 구분한다.

## 손 높이와 RSI

- `h065`: 상자 뒷면의 지면 기준0.65m 목표점. RSI 팔을 해당 높이로 retarget하지 않는다.
- `hamp`: `height_mode: amp_reference`. RSI는 샘플한 원본 PUSH clip/time의 손 중점 높이를 사용한다. 원본 관절·양손 높이 차이를 보존한 rigid XY 정렬/전체 인체 관통 보정 후 높이를 목표로 고정한다. 바닥 보정은 전체 인체를 올리는 기존 safe reset을 따른다. 일반 보행 시작도 동일한 native PUSH pool에서 별도 reference를 추출해 목표 높이를 고정한다. 정책이 움직이는 손 높이를 매 step 따라가지 않는다. 상자 면 밖 높이는 clamp하지 않고 오류로 검출한다. 부분 reset은 해당 슬롯만 갱신한다.

물리·모션 계약은 동일하므로 기존 `stage1_push_depth180_safe_cache.npz`와 SHA를 재사용한다. 높이 목표와 state 감쇠는 checkpoint의 pushTask/pushPointReward에 저장돼 기존 학습 및 두 비교 버전과 직접 resume/test 혼용을 거부한다. 기존 실행 중 학습은 유지한다.

## 실행과 검증

학습·로컬·VNC 명령은 [config](../config.md)에 각 실험명으로 기재한다. 본학습2048환경, 짧은 학습도2048환경을 유지한다. 검증 산출물은 `output_etc/check/stage1_push_depth180_safe_{h065,hamp}_relaxed_reset/`에만 저장한다. CPU 성공/force/높이 계약은 `tests/test_push_depth180_relaxed.py`, GPU reset·관절 보존·부분reset·12step 검사는 `tests/check_stage1_push.py`를 사용한다. 검증 결과는 changelog에 기록한다.


## 높은 손 위치의 전도 확인

원본 AMP 높이 버전의 GPU6 4096 agent reset 전체에서 목표 높이는0.8683~1.1695m였다. 마지막 reset만 본 범위0.9475~1.1228m보다 넓다. RSI 원본관절·무관통·부분reset·12step 유한값 검사를 통과했지만, 높은 위치의 장기 밀기 안정성을 뜻하지 않는다.

20kg·Y1.8m 접촉체 검사는0.95~1.0m에서 최대0.019°미만·약0.83~0.85m 이동,1.10m에서최대3.95°,1.12m에서최대34.79°,1.15m에서42조건중21조건45°초과였다. 원본 AMP 높이를 임의로 clamp하지 않았으며 높은 reference의 위험은 남는다. 실제 humanoid PD/IK 검사도 높은 위치의 유효한 안정 손 밀기를 입증하지 못했다. [비교보고서](../../output_etc/check/push_depth180_high_20261020/REPORT.md)를 따른다.


## 손 목표 높이1.0m 고정

사용자 선택으로 `stage1_push_depth180_safe_h100_relaxed`를 추가했다. 손 목표는 상자 뒷면 지면 기준1.0m 고정이다. 상자X1.2~1.5m·Y1.8m·Z1.3m·20kg, 목표1~4m/접근1~5m·원본AMP/RSI관절보존·safe reset·연속state감쇠·state>0.9+0.5N 성공·AT/paired포화는 동일하다. 원본AMP의 손높이를1.0m로 retarget하지 않고 접촉목표점만 고정한다. 기존0.65m/AMP 버전 및20kg RSI cache는 보존한다. checkpoint는 각높이/계약과직접혼용불가이며 새scratch실행이다. 학습질량은20kg을유지하며25kg은별도비학습비교결과다.

학습·로컬·VNC는 [config](../config.md)의 `stage1_push_depth180_safe_h100_relaxed` 명령을 사용한다. GPU 검증 경로는 `output_etc/check/stage1_push_depth180_safe_h100_relaxed_reset/`다. 기존실행중학습은자동변경/재시작하지않는다.
