# PUSH RSI: 원본 손 높이≤1m + 미는 자세 구간 선별

앞으로의 PUSH RSI 작업은 [공통 기준](../../../../../markdowns/push_rsi_guidelines.md)을 따른다.

사용자가 허용한 loco 접근 시작은 유지한다. PUSH RSI에서 팔을 내린 준비/정리 자세를 제외한다. 손 중점≤1m는 상자 전도 우려로 유지하며, 관절·속도·손 높이를 보정하지 않는다. 보상 손 목표 0.8m/1.0m는 본 선별로 바꾸지 않는다.

## 결과와 기준

- 별도 캐시: `tokenhsi/data/stage1_push_depth180_rsi_le1m_push_phase_cache.npz`.
- 모션별 시간 구간·원본 hash·캐시 hash: `tokenhsi/data/stage1_push_depth180_rsi_le1m_push_phase_selection.json`.
- 기존 ≤1m 캐시의 부분집합이다. 좌우 각각 손이 같은 쪽 어깨보다 몸통 전방으로≥0.25m, 수직으로 어깨보다 낮은 정도≤0.25m인 자세만 허용한다. 기존 7크기 전부 통과한 연속 구간의 양끝0.2초를 잘라낸 뒤0.5초 이상 남아야 한다. 높이 기준과 무충돌/기존0.1초 PD screen을 모두 계승한다.
- 21개 중5개(clip0/3/4/7/8), 고유1196프레임×7크기=8372조합. 원본 관절/속도·AMP21개·loco4개·loco/PUSH 혼합 비율과 late70% 규칙은 그대로이며, RSI pool만 줄어든다. A344 계열 및 좌우 반전만 남아 RSI 동작 다양성이 줄었다.
- 문제 사례 clip15,t=0.066667s 제외. 허용 사례 clip8,t=3.933333s 유지.

| clip | 원본 | 선택 구간(초, 양끝 포함) | 프레임 수 |
| --- | --- | --- | --- |
| 0 | 001 A344 M | 3.2333~10.1667 | 209 |
| 3 | 002 A344 | 3.0000~11.1667 | 246 |
| 4 | 002 A344 M | 2.9333~11.1667 | 248 |
| 7 | 003 A344 | 2.7333~10.9333 | 247 |
| 8 | 003 A344 M | 2.7667~10.9333 | 246 |

## 확인과 한계

`output_etc/check/push_rsi_phase_selection_20261020/`에 전수 실제 배치 감사, 5개 구간 시작/중간/끝15장, 검수 모음 `review_01.jpg`, 원본 시간 순 RSI 자세 영상 `reselected_push_rsi_with_box.mp4`(40초/10fps)가 있다. 영상은 정책/물리 rollout이 아니라 실제 초기 배치 코드에서 추출한 연속 원본 시간의 자세다. 실제 발 메시도 함께 표시한다.

8372배치에서 원본 DOF 오차0, 몸-상자 겹침0, 목표 반대편 위반0, 손의 면 투영 이탈0. 손 중점 높이0.945716~0.972829m, 상자 뒷면 간격 약8cm, 머리-상자 최소 간격0.323m. 원본에는 상자 궤적/접촉 정답이 없으므로 이 선별은 팔 자세 기준이며 초기 힘 접촉을 보장하지 않는다. 0.8m 보상 목표에서 초기 기하 state>0.9인 후보는 없다.

실제 발 메시 전수 측정1196건에서 양발 중 더 낮은 발바닥은 바닥 위5.0~41.3mm이며, 양발 모두40mm를 넘는 것은3프레임이다. 원본 발걸음 자세를 유지하고 있으며 양발이 항상 바닥에 닿는 조건은 아니다. 손 높이를 맞추기 위해 몸을 띄우지는 않는다. 높이≤1m와 자세 선별만으로 학습 중 전도가 사라진다고 보장할 수 없다.

GPU6 별도35평가환경에서5모션×시작/중간/끝×7크기×2명=210건을 원본 초기 관절각 PD 유지로0.1초(6tick) 검사했고210/210통과, fall0·nonfinite0이었다. 장기 균형·상자 이동/전도 검사는 아니다. CPU20개 및 영상 전체 디코딩도 통과했다.

기존 학습/활성 YAML/캐시는 변경하지 않았다. 새 캐시는 현재 샘플러에서 로딩·후반 샘플링 검증을 마친 별도 데이터다. 아래 전용 두 설정에는 새 cache 파일/SHA를 연결했다. 기존 checkpoint에 파일만 바꿔 직접 resume하지 않으며 새 학습 명령을 사용한다.

## 재현

사본 루트에서 실행한다. 출력 폴더의 동명 캐시가 이미 있으면 덮어쓰지 않는다.

```bash
python3 tokenhsi/scripts/multi_agent/select_push_rsi_phase.py \
  --source tokenhsi/data/stage1_push_depth180_rsi_le1m_cache.npz \
  --manifest tokenhsi/data/dataset_stage1_push.yaml \
  --output output_etc/check/push_rsi_phase_rebuild

# tokenhsi 환경, Isaac Gym→torch 순서 포함; GPU 시뮬레이션 없음
python -m pytest tokenhsi/tests/test_push_rsi_phase_selection.py tokenhsi/tests/test_push_rsi_height_selection.py -q
```

중간에 높이 제한을 뺀 비교 캐시도 검사했으나, 사용자가1m 제한 유지를 확인한 뒤 최종본에서 제외했다. `push_rsi_native_phase_selection_20261020/`은 그 비교 산출물이며 학습에 사용하지 않는다.

## 선별본 전용 두 학습 설정

`stage1_push_depth180_h100_rsi1m_at5deg_phase`는 손 목표1.0m·AT 성공≤5°, `stage1_push_depth180_h080_rsi1m_phase`는 손 목표0.8m·AT 기울기 제한 없음이다. 둘 다 위의 새 선별 cache/SHA를 사용하며 loco50%/PUSH50%, 손 높이≤1m, 원본 관절/속도·AMP21모션을 유지한다. 커리큘럼 없이 목표1~4m·loco 시작1~5m다. 학습·로컬·서버 VNC 명령은 [실행 가이드](../config.md)에 있다. 전용 output에서 새 학습하는 명령이며 기존 checkpoint를 직접 resume하지 않는다. 기존 프로세스는 자동 종료/재시작하지 않는다.
