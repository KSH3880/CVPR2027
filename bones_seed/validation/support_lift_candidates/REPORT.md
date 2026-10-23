# Stage2 협동 승강용 모션 후보

2026-10-16. 목표: 여러 사람이 주먹형 손으로 물체 바닥을 받쳐 올리고, 그 위 탑승자를 함께 상승시키는 과제의 Stage1 참조 후보.

공식 viewer: https://seed-viewer.bones.studio/ (SEED에서 파일명 검색). BVH 4개는 공식 viewer API의 `tree` 필드를 저장했다. 원본 Y-up/cm를 기존 `bones_bvh.read_bvh`로 m 단위 FK 분석했다. 전 프레임 finite 검사 통과. 원본 물체 궤적/접촉/질량/탑승자는 없으므로 지지력·플랫폼 수평·협동 성공은 검증하지 않았다.

| 우선 | 파일명 | 확인 구간 | 결과 |
|---|---|---|---|
| 1 | `medium_big_heavy_two_hands_front_medium_to_front_high_R_001__A520` | 0.95–4.45초 | 0.95→3.3초 양손목 1.017/1.018→1.840/1.863m. 양팔 상승 후보. 마지막 선반 배치/해제 구간은 별도 취급. |
| 2 | `medium_big_heavy_two_hands_front_medium_to_front_high_R_001__A522` | 1–3.45초 | 1→2.6초 양손목 1.078/1.079→1.808/1.815m. 더 빠른 상승 후보. |
| 3 | `medium_big_heavy_two_hands_pick_up_front_low_R_002__A504` | 2.1–5.25초 상승, 5.25–12.65초 유지 | 낮은 자세에서 일어나며 상승. 메타데이터상 처음 옆면 파지 후 바닥 지지로 전환. 처음부터 바닥 지지하는 모션으로 확정하지 않는다. |
| 낮음 | `medium_big_heavy_two_hands_pick_up_front_medium_R_001__A509` | 1.27–4.6초 | 해당 구간 손목 상승 약0.14m. 큰 승강 동작 후보로 우선순위 낮음. |

[4행 자세 비교](contact_sheet.png)는 표 순서다. 빨강 오른쪽/파랑 왼쪽, 점은 손목이며 주먹 collider가 아니다. [샘플 손목 분석](wrist_analysis.json)은 고정 시점 비교이며 전체 구간 최대 상승량이 아니다.

Stage1 후보 판단: 낮은 높이 지지·양팔 상승·높은 높이 지지를 함께 확보하는 편이 목표에 맞다. A520/A522는 실제 상승을 확인했지만 상승 내내 바닥 지지가 유지되는지는 공식 viewer에서 추가 확인해야 한다. 탑승자를 포함한 하중과 주먹 collider 접촉, 플랫폼 기울기, 여러 사람 동기화는 Stage2에서 별도 물리 검증이 필요하다. 학습/retarget/reward/config 변경은 하지 않았다.
