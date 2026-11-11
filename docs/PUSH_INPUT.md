# PUSH 학습 입력과 환경 구분

확정: PUSH의 상자 크기·무게·배치·reward·AMP·RSI는 지정 Stage1 26k 설정을 사용한다. 기존 SIT/CARRY/CLIMB의 KD 학습은 원본 TokenHSI 물체·크기·초기화·관측을 유지한다. TRAJ는 제외한다.

## PUSH student 입력

| 블록 | 차원 | PUSH에서 사용 |
|---|---:|---|
| 자기 humanoid 상태 |223|사용 |
| 기존 TRAJ/SIT/CARRY/CLIMB 관측 슬롯 |127|0으로마스킹 |
| PUSH 담당 상자와 목표 |42|사용 |
| task one-hot |5|[0,0,0,0,1] |
| 합계 |397| |

PUSH42D는 TokenHSI CARRY와 같은 관측 형식을 사용한다: 상자선속도3+각속도3+위치3+회전6+상자8개꼭짓점24+목표위치3. 실제 변환은 원본 `humanoid_carry.compute_location_observations()`를 호출한다. 상자·목표 위치/방향은 이 함수의 사람 기준 좌표 변환을 따른다.

크기를 TokenHSI의 작은 상자로 줄이거나 범위를잘라내지않는다. 우리167종의실제크기를8개꼭짓점에반영한다. 크기가달라도꼭짓점수가8개이므로입력차원은같고값만달라진다. 무게는별도입력채널이없고물리환경에서부피×밀도3으로정해진다.

TokenHSI 원본 actor 구조를유지하면서42D PUSH tokenizer만추가한다. Carry tokenizer에서초기화하지만 PUSH의큰상자에맞는행동은PPO+AMP로학습한다. Teacher는PUSH 행동을지도하지않는다.

## 여러 사람과 물체

PUSH 참조scene에는2명·4상자가있다. 각사람의397D 관측을같은student에독립적으로넣고각32D행동을해당사람에적용한다. 동료/여분상자는물리적으로존재하지만student의별도관측토큰으로추가하지않는다. 우리GTA/관계토큰입력을그대로사용하지않는다. 이관측차이는논문baseline설명에명시한다.

원래26k의장면분포를보존하려고참조scene내비PUSH동료도남긴다. 이슬롯은student로제어하되RL/KD loss는없다. 기존과제의KD용물체를우리상자로바꾼다는뜻이아니다. KD용SIT/CARRY/CLIMB는별도원본TokenHSI환경에서수집한다.
