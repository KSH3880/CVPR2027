# Y1.8m·20kg·원본 RSI≤1m 선별 비교

| 설정 | 손 목표 높이 | AT 성공 기울기(yaw 제외) |
| --- | --- | --- |
| `stage1_push_depth180_h100_rsi1m_at5deg` | 1.0m | ≤5° |
| `stage1_push_depth180_h080_rsi1m` | 0.8m | 제한 없음 |

공통 X1.2~1.5m/5cm·Y1.8m고정·Z1.3m고정·20kg·마찰0.6·2명/4상자/634D. 사용자확인에따라상자→목표만1~4m이며사람→상자시작은1~5m를유지한다. 손목표는상자뒷면고정물질점높이이며원본RSI손높이를해당높이로수정하지않는다.

## RSI 선택

`pushTask.rsi_max_hand_midpoint_height_m: 1.0`을사용한다. 원본PUSH AMP 21모션의프레임중양손중점높이가1m이하이고,기존safe바닥강체이동후에도1m이하인프레임만기존물리유효mask와교집합으로선택한다. float32배치오차용1e-3m선별여유를사용한다. 초기화시에도실제배치높이를assert한다. 양손중점기준이며각손높이를따로clamp하거나원본관절/속도/AMP데이터를수정하지않는다.

기존56042 size-frame에서12831개(7크기×1833프레임),21/21모션이남았다. 원본중점0.867878~0.998610m,배치후중점0.867878~0.998943m다. `stage1_push_depth180_rsi_le1m_cache.npz`와SHA를두config가공유한다. 원래MotionLib모션가중치·RSI50%·후반70%(선별유효프레임의후반30%)를유지하며빈크기pool은오류로중단한다. 조건밖프레임이나loco로조용히fallback하지않는다.

`tests/filter_push_rsi_height.py`가원본safe캐시와원본MotionLib의gts/grs에서새mask만생성한다. 원본캐시/공유모션은읽기만하며원본데이터수정이나RSI자세retarget은없다. 생성감사는 `output_etc/check/push_depth180_rsi_le1m_selection_v2/selection.json`이다.

## 성공·보상

연속state는기존완화값(`xy_reward_scale`, `z_reward_sigma_m`)을사용한다. CONTACT자체성공은같은물리substep에서state>0.9와양손각각/상자면법선반력≥0.5N을요구한다. 별도수평/높이gate는없다. AT성공에따른CONTACT보상paired포화는유지한다. 1번의기울기5°gate는AT성공과그에따른paired포화에만적용하며state/progress연속감쇠·전도종료를추가하지않는다. 2번에는AT기울기제한이없다.

## 실행·호환

[config](../config.md)에각train/test/VNC명령을기재한다. GPU6에서각2048환경으로scratch학습한다. 두실험은손목표/기울기계약이다르므로checkpoint를직접혼용하지않는다. 기존unfiltered RSI checkpoint도직접resume불가다. 확인실행은 `output_etc/check/<실험명>_check/` 아래로분리하며학습환경은2048을유지한다. 장기본학습은자동실행하지않는다.


## 검증 결과

CPU84개 통과. GPU6에서 두 실험 각각64검증환경4096agent reset·원본 DOF position/velocity 보존·각 body Z의 동일 강체이동·634D·초기 무관통·부분reset·12step 유한값을 확인했다. 실제 RSI 손 중점 범위는 두 실험 모두0.867878~0.998943m였다. 각2048학습환경의 짧은 PPO/AMP 확인2epoch/frame262144 후669tensor 유한값·전용test2환경/RSI100%/12step×2 재로딩을 확인했다. 검증 체크포인트는 `output_etc/check/<실험명>_check/train/`이며 새 본학습은 자동 시작하지 않았다. 장기 정책의 성공률·무전도는 이 확인으로 보장하지 않는다.
