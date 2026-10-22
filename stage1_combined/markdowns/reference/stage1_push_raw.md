# PUSH 원본 RSI·가변 정육면체

`stage1_v2_fixed_push_360_raw` 전용 설정으로 추가했다. 기존 PUSH/360 설정·checkpoint의 기본 동작은 유지한다.

- RSI는 기존7개 원본 clip·11개 후보 시각을 사용한다. 관절·손 높이·속도와 AMP 이력에 팔 높이 보정을 적용하지 않는다.
- 접촉 기준점은 상자가 수평일 때 바닥에서1.04m다. 손 중점 보상·기존 Z 허용 구간과0.5N 유효 접촉 조건은 유지한다.
- 정육면체 한 변1.2~1.7m, 질량25~30kg을 환경 생성 시 상자마다 독립적으로 연속 균등 샘플링한다. 크기/질량은 actor 수명 동안 고정하며 마찰은0.6이다.
- 상자 중심 높이·미는 면·장면 간격은 실제 크기로 계산한다. 접촉 RSI의 손 중점6cm 정렬 후, 사람의 모든 sphere/capsule/발 mesh가 지정 면 바깥에 최소5mm 떨어지도록 필요한 경우에만 전신을 수평 후퇴시킨다. 관절·높이·속도는 변경하지 않는다.
- 모든 충돌 형상이 미는 면 바깥 반공간에 있으면 상자 내부와 교차하지 않는다. 면 기준의 support 계산은 폭·높이와 무관하므로 사용 중인11개 후보의 초기 침투 방지는1.2~1.7m 연속 크기 범위에 적용된다.
- 기존/PUSH50:50·teacher KL·AMP·보행/목표 커리큘럼은 유지한다. 원본 NPY와 기존 공용 RSI cache는 변경하지 않으며 새 bank는 후보 시각과 물성 계약을 별도로 기록한다.

기존 PUSH 모델 전이는 `PUSH_CHECKPOINT`, 새 raw 모델 재개는 `RESUME_CHECKPOINT`를 사용한다. 전이는 모델/정규화를 가져오고 optimizer/epoch을 새로 시작한다. 기존 checkpoint를 새 wrapper의 재개/추론에 직접 넣으면 거부한다. 실행 명령은 [config](../config.md), 추가 옵션은 [USAGE](../USAGE.md#push-원본-rsi가변-정육면체)를 따른다.

## 검증

- 관련 CPU47개 통과. 크기/질량 범위·독립 샘플링·1.04m 기준점·최소 후퇴·전이/재개 계약을 검사했다.
- 실제 관절 FK로51개 크기(1cm 간격)×11후보×질량25/27.5/30kg=1683건과 연속 무작위365건을 검사했다. 초기 침투0/2048, 최소 면 여유 약5.00mm, 추가 후퇴 최대6.784mm였다.
- 같은2048건의 별도 GPU PhysX에서 원본 초기 속도·원래 humanoid/상자·동일 물리 파라미터로 최초 관절각을 PD 목표로 고정해1초 관측했다. 머리 접촉 충격0N·2m/s 초과0건·최대 상자 속도0.236m/s·최대 기울기0.740도였다. 질량 실측 최대 오차는약0.000006kg였다.
- 기존 PUSH11000에서 실제2048환경 짧은 학습을 실행해 epoch2/frame262144 저장, 모델·542개 로그 scalar finite와 정책/AMP 입력 정규화 불변을 확인했다. 본학습은 시작하지 않았다.
- 새 checkpoint의 실제1환경 추론64회·128쌍에서 초기 침투0, 머리 충격0N, 초기0.1초2m/s 초과0건, 최대 상자 기울기3.964도를 확인했다. 초기0.1초 최대 속도는1.803m/s였다. 원본 관절·속도·손 높이·AMP 이력, simulator 질량·마찰·접촉점 높이도 별도로 확인했다.
- 기존 checkpoint의 기본 보정 reset은 이전 root·관절·속도·상자·goal·후보 시각과 bitwise 동일했다. VNC wrapper 인자 전달과4환경 배치를 검사했으며 이번 작업에서 GUI viewer는 실행하지 않았다.

이 검증은 초기 침투와 짧은 물리 안정성 검사다. 장기 정책의 밀기 성공·전도 방지나 모든 반복 reset 이력의 무충격을 보장하는 평가는 아니다.

검증 자료: [전체 크기 검사](../../output_etc/check/push_raw_variable_20261020/sweep_summary.json), [학습](../../output_etc/check/push_raw_variable_20261020/train_validation.json), [실제 reset 반복](../../output_etc/check/push_raw_variable_20261020/native/summary.json), [원본 상태·물성](../../output_etc/check/push_raw_variable_20261020/state/validation.json).

## 직육면체 variant

`stage1_v2_fixed_push_360_raw_cuboid`는 raw 설정에서 push 상자 물성·bank·PUSH 에피소드 길이만 바꾼다. 원본 RSI·1.04m 접촉점·보상·기존 과제·전이 source는 같다.

- push 상자는 x 1.2~1.7m, y(미는 방향 깊이)·z 1.45~1.7m를 축마다 독립 균등 샘플링한 뒤 y·z 중 큰 값을 y에 둔다(`depth_at_least_height`). 질량15~20kg, 마찰0.3이다. carry/sit/climb 상자와 PUSH 환경의 여분 상자는 기본 마찰1.0을 유지한다.
- 1.04m에서 수평으로 밀 때 깊이 d가 2·μ_eff·h보다 크면 기울기 전에 미끄러진다. 땅1.0과 평균되는 실효 마찰은 0.654(CPU 실측)라 기준은 1.36m이고 최소 깊이1.45m가 이를 넘는다.
- PUSH 에피소드는 1200step이다. bank는 raw 후보 11개를 그대로 쓰고 물성 계약만 바꿨다. 면 기준 후퇴라 상자 폭·깊이·높이와 무관하다.
- 마찰이 달라 raw checkpoint의 `PUSH_CHECKPOINT` warm-start는 거부되며 Stage1 73000에서 바로 전이한다.

검증: CPU 203개 통과(2개는 서버에 없는 60000 checkpoint 로드 실패). GPU5 2048환경 짧은 학습 epoch4 저장·source_epoch 73000 전이·모델 finite를 확인했다. 그 checkpoint로 256환경을 생성해 push 상자 256개의 실제 마찰0.3·질량15.0~20.0kg·크기 범위·y≥z와 나머지 상자 768개의 마찰1.0을 확인했고, test wrapper 2환경 평가를 실행했다. 본학습과 GUI viewer는 실행하지 않았다.
