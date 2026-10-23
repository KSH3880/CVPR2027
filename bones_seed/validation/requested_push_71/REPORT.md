# 요청 PUSH 71개 원본 검토

요청 목록은 180 계열 21개, 270 계열 24개, 360 계열 26개다. 기존 AMP 두 clip은 180 계열에 포함되므로 전체 추가 시 신규 clip은 69개다.

## 현재 과제와의 관계

| 계열 | 원본 temporal label 기준 동작 | 현재 양손 PUSH와 관계 |
|---|---|---|
| 180 | 앞에서 양손으로 밀며 전진 | 변환·물리 RSI 검증 후보 |
| 270 | 옆몸 및 한손으로 밀며 옆걸음 | 양손 근접·반력 credit과 불일치 가능 |
| 360 | 등·엉덩이로 밀며 뒤로 걸음 | 양손 credit과 불일치 |

현재 조건은 `tokenhsi/env/tasks/multi_agent/open_push_reward.py:push_hand_contact`의 두 손 모두 박스 뒤면 XY 근접·전방 반력 및 박스 반력이다. 세계 좌표 회전과 신체에 대한 밀기 방향 차이를 구분해야 한다. 원본에는 박스 pose나 접촉 기록이 없으므로 temporal label만으로 실제 접촉·RSI 통과를 확정하지 않는다.

후속 사용자 선택으로180 계열21개만 별도 AMP manifest·2m 넓은 배치 RSI bank에 연결했다. 현재 학습은 유지한다.270·360은 포함하지 않는다. 상세 실행·검증은 `markdowns/config.md`의 PUSH180 절을 참조한다.

## 원본 동작 구간

모든 event는 원본 시간(sec)이며 retarget/crop 결과가 아니다.

| clip | 시작–끝 | 설명 |
|---|---|---|
| `push_obstacle_180_001__A344_M` | 0–2.3 | A person standing upright pushes against an obstacle in front. |
| `push_obstacle_180_001__A344_M` | 2.3–10.65 | A person slowly walks forward while pushing an obstacle with both hands. |
| `push_obstacle_180_001__A344_M` | 10.65–12.45 | A person stops pushing an obstacle with both hands before standing upright. |
| `push_obstacle_180_001__A344_M` | 12.45–15.4 | A person standing upright lowers their arms by their sides. |
| `push_obstacle_180_002__A342` | 0–1.45 | A person standing upright leans forward while raising their arms and placing their hands firmly on the obstacle. |
| `push_obstacle_180_002__A342` | 1.45–2.85 | A person leaning forward with their hands on the object begins to walk ahead while steadily pushing the obstacle forward. |
| `push_obstacle_180_002__A342` | 2.85–11.65 | A person leaning forward with their hands on the object walks ahead while steadily pushing the obstacle forward. |
| `push_obstacle_180_002__A342` | 11.65–15.6 | A person rises to stand upright while lowering their arms by their side after pushing the obstacle. |
| `push_obstacle_180_002__A342_M` | 0–1.45 | A person standing upright leans forward while raising their arms and placing their hands firmly on the obstacle. |
| `push_obstacle_180_002__A342_M` | 1.45–2.85 | A person leaning forward with their hands on the object begins to walk ahead while steadily pushing the obstacle forward. |
| `push_obstacle_180_002__A342_M` | 2.85–11.65 | A person leaning forward with their hands on the object walks ahead while steadily pushing the obstacle forward. |
| `push_obstacle_180_002__A342_M` | 11.65–15.6 | A person rises to stand upright while lowering their arms by their side after pushing the obstacle. |
| `push_obstacle_180_002__A344` | 0–2.2 | A person standing upright pushes against an obstacle in front. |
| `push_obstacle_180_002__A344` | 2.2–11.7 | A person slowly walks forward while pushing an obstacle with both hands. |
| `push_obstacle_180_002__A344` | 11.7–13.35 | A person stops pushing an obstacle with both hands before standing upright. |
| `push_obstacle_180_002__A344` | 13.35–14.8 | A person standing upright lowers their arms by their sides. |
| `push_obstacle_180_002__A344_M` | 0–2.2 | A person standing upright pushes against an obstacle in front. |
| `push_obstacle_180_002__A344_M` | 2.2–11.7 | A person slowly walks forward while pushing an obstacle with both hands. |
| `push_obstacle_180_002__A344_M` | 11.7–13.35 | A person stops pushing an obstacle with both hands before standing upright. |
| `push_obstacle_180_002__A344_M` | 13.35–14.8 | A person standing upright lowers their arms by their sides. |
| `push_obstacle_180_003__A342` | 0–1.66667 | A person standing upright leans forward while raising their arms and placing their hands firmly on the obstacle. |
| `push_obstacle_180_003__A342` | 1.66667–2.9 | A person leaning forward with their hands on the object begins to walk ahead while steadily pushing the obstacle forward. |
| `push_obstacle_180_003__A342` | 2.9–11.7333 | A person leaning forward with their hands on the object walks ahead while steadily pushing the obstacle forward. |
| `push_obstacle_180_003__A342` | 11.7333–13.9 | A person rises to stand upright while lowering their arms by their side after pushing the obstacle. |
| `push_obstacle_180_003__A342_M` | 0–1.66667 | A person standing upright leans forward while raising their arms and placing their hands firmly on the obstacle. |
| `push_obstacle_180_003__A342_M` | 1.66667–2.9 | A person leaning forward with their hands on the object begins to walk ahead while steadily pushing the obstacle forward. |
| `push_obstacle_180_003__A342_M` | 2.9–11.7333 | A person leaning forward with their hands on the object walks ahead while steadily pushing the obstacle forward. |
| `push_obstacle_180_003__A342_M` | 11.7333–13.9 | A person rises to stand upright while lowering their arms by their side after pushing the obstacle. |
| `push_obstacle_180_003__A344` | 0–2.2 | The person leans forward with both arms bent, preparing to push an object. |
| `push_obstacle_180_003__A344` | 2.2–11.3 | The person pushes the object forward while slowly stepping to apply force. |
| `push_obstacle_180_003__A344` | 11.3–15.25 | The person brings their arms back off the object and begins to rise and stand upright. |
| `push_obstacle_180_003__A344_M` | 0–2.2 | The person leans forward with both arms bent, preparing to push an object. |
| `push_obstacle_180_003__A344_M` | 2.2–11.3 | The person pushes the object forward while slowly stepping to apply force. |
| `push_obstacle_180_003__A344_M` | 11.3–15.25 | The person brings their arms back off the object and begins to rise and stand upright. |
| `push_obstacle_180_101__A341` | 0–1.9 | A person standing upright pushes against an obstacle in front. |
| `push_obstacle_180_101__A341` | 1.9–12.2 | A person slowly walks forward while pushing an obstacle with both hands. |
| `push_obstacle_180_101__A341` | 12.2–13.7667 | A person stops pushing an obstacle with both hands before standing upright. |
| `push_obstacle_180_101__A341` | 13.7667–14.7667 | A person standing upright lowers their arms by their sides. |
| `push_obstacle_180_101__A341_M` | 0–1.9 | A person standing upright pushes against an obstacle in front. |
| `push_obstacle_180_101__A341_M` | 1.9–12.2 | A person slowly walks forward while pushing an obstacle with both hands. |
| `push_obstacle_180_101__A341_M` | 12.2–13.7667 | A person stops pushing an obstacle with both hands before standing upright. |
| `push_obstacle_180_101__A341_M` | 13.7667–14.7667 | A person standing upright lowers their arms by their sides. |
| `push_obstacle_180_101__A343` | 0–1.6 | The person leans forward with both arms bent, preparing to push an object. |
| `push_obstacle_180_101__A343` | 1.6–15.6 | The person pushes the object forward while slowly stepping to apply force. |
| `push_obstacle_180_101__A343` | 15.6–17.6 | The person brings their arms back off the object and begins to rise and stand upright. |
| `push_obstacle_180_101__A343_M` | 0–1.6 | The person leans forward with both arms bent, preparing to push an object. |
| `push_obstacle_180_101__A343_M` | 1.6–15.6 | The person pushes the object forward while slowly stepping to apply force. |
| `push_obstacle_180_101__A343_M` | 15.6–17.6 | The person brings their arms back off the object and begins to rise and stand upright. |
| `push_obstacle_180_102__A341` | 0–1.75 | A person standing upright pushes against an obstacle in front. |
| `push_obstacle_180_102__A341` | 1.75–11.7 | A person slowly walks forward while pushing an obstacle with both hands. |
| `push_obstacle_180_102__A341` | 11.7–13.35 | A person stops pushing an obstacle with both hands before standing upright. |
| `push_obstacle_180_102__A341` | 13.35–14.65 | A person standing upright lowers their arms by their sides. |
| `push_obstacle_180_102__A341_M` | 0–1.75 | A person standing upright pushes against an obstacle in front. |
| `push_obstacle_180_102__A341_M` | 1.75–11.7 | A person slowly walks forward while pushing an obstacle with both hands. |
| `push_obstacle_180_102__A341_M` | 11.7–13.35 | A person stops pushing an obstacle with both hands before standing upright. |
| `push_obstacle_180_102__A341_M` | 13.35–14.65 | A person standing upright lowers their arms by their sides. |
| `push_obstacle_180_102__A343` | 0–1.95 | The person leans forward with both arms bent, preparing to push an object. |
| `push_obstacle_180_102__A343` | 1.95–14.2 | The person pushes the object forward while slowly stepping to apply force. |
| `push_obstacle_180_102__A343` | 14.2–17.15 | The person brings their arms back off the object and begins to rise and stand upright. |
| `push_obstacle_180_102__A343_M` | 0–1.95 | The person leans forward with both arms bent, preparing to push an object. |
| `push_obstacle_180_102__A343_M` | 1.95–14.2 | The person pushes the object forward while slowly stepping to apply force. |
| `push_obstacle_180_102__A343_M` | 14.2–17.15 | The person brings their arms back off the object and begins to rise and stand upright. |
| `push_obstacle_180_103__A341` | 0–1.45 | A person standing upright pushes against an obstacle in front. |
| `push_obstacle_180_103__A341` | 1.45–12.85 | A person slowly walks forward while pushing an obstacle with both hands. |
| `push_obstacle_180_103__A341` | 12.85–14.1 | A person stops pushing an obstacle with both hands before standing upright. |
| `push_obstacle_180_103__A341` | 14.1–15.15 | A person standing upright lowers their arms by their sides. |
| `push_obstacle_180_103__A341_M` | 0–1.45 | A person standing upright pushes against an obstacle in front. |
| `push_obstacle_180_103__A341_M` | 1.45–12.85 | A person slowly walks forward while pushing an obstacle with both hands. |
| `push_obstacle_180_103__A341_M` | 12.85–14.1 | A person stops pushing an obstacle with both hands before standing upright. |
| `push_obstacle_180_103__A341_M` | 14.1–15.15 | A person standing upright lowers their arms by their sides. |
| `push_obstacle_180_103__A343` | 0–1.9 | The person leans forward with both arms bent, preparing to push an object. |
| `push_obstacle_180_103__A343` | 1.9–14.9 | The person pushes the object forward while slowly stepping to apply force. |
| `push_obstacle_180_103__A343` | 14.9–16.75 | The person brings their arms back off the object and begins to rise and stand upright. |
| `push_obstacle_180_103__A343_M` | 0–1.9 | The person leans forward with both arms bent, preparing to push an object. |
| `push_obstacle_180_103__A343_M` | 1.9–14.9 | The person pushes the object forward while slowly stepping to apply force. |
| `push_obstacle_180_103__A343_M` | 14.9–16.75 | The person brings their arms back off the object and begins to rise and stand upright. |
| `push_obstacle_270_001__A342` | 0–1.7 | A person steps forward and turns left. |
| `push_obstacle_270_001__A342` | 1.7–4.75 | A person takes a step forward and simultaneously turns left to push an obstacle with their side by leaning on it. |
| `push_obstacle_270_001__A342` | 4.75–16.15 | A person is pushing an obstacle forward with effort, walking sideways in the right direction, while leaning on it. |
| `push_obstacle_270_001__A342` | 16.15–19.65 | A person stops pushing an obstacle and turns right to stand upright. |
| `push_obstacle_270_001__A342` | 19.65–21.25 | A person is standing idle, gently swaying their hands. |
| `push_obstacle_270_001__A342_M` | 0–1.7 | A person steps forward and turns right. |
| `push_obstacle_270_001__A342_M` | 1.7–4.75 | A person takes a step forward and simultaneously turns right to push an obstacle with their side by leaning on it. |
| `push_obstacle_270_001__A342_M` | 4.75–16.15 | A person is pushing an obstacle forward with effort, walking sideways in the left direction, while leaning on it. |
| `push_obstacle_270_001__A342_M` | 16.15–19.65 | A person stops pushing an obstacle and turns left to stand upright. |
| `push_obstacle_270_001__A342_M` | 19.65–21.25 | A person is standing idle, gently swaying their hands. |
| `push_obstacle_270_001__A344` | 0–1.3 | A person steps forward and turns left. |
| `push_obstacle_270_001__A344` | 1.3–2.6 | A person takes a step forward and simultaneously turns left to push an obstacle with their side by leaning on it. |
| `push_obstacle_270_001__A344` | 2.6–14.7333 | A person is pushing an obstacle forward with effort, walking sideways in the right direction, while leaning on it. |
| `push_obstacle_270_001__A344` | 14.7333–18.5 | A person stops pushing an obstacle and turns right to stand upright. |
| `push_obstacle_270_001__A344` | 18.5–20.1 | A person is standing idle, gently swaying their hands. |
| `push_obstacle_270_001__A344_M` | 0–1.3 | A person steps forward and turns right. |
| `push_obstacle_270_001__A344_M` | 1.3–2.6 | A person takes a step forward and simultaneously turns right to push an obstacle with their side by leaning on it. |
| `push_obstacle_270_001__A344_M` | 2.6–14.7333 | A person is pushing an obstacle forward with effort, walking sideways in the left direction, while leaning on it. |
| `push_obstacle_270_001__A344_M` | 14.7333–18.5 | A person stops pushing an obstacle and turns left to stand upright. |
| `push_obstacle_270_001__A344_M` | 18.5–20.1 | A person is standing idle, gently swaying their hands. |
| `push_obstacle_270_002__A342` | 0–1 | A person steps forward and turns left. |
| `push_obstacle_270_002__A342` | 1–2.6 | A person takes a step forward and simultaneously turns left to push an obstacle with their side by leaning on it. |
| `push_obstacle_270_002__A342` | 2.6–16 | A person is pushing an obstacle forward with effort, walking sideways in the right direction, while leaning on it. |
| `push_obstacle_270_002__A342` | 16–20.4 | A person stops pushing an obstacle and turns right to stand upright. |
| `push_obstacle_270_002__A342` | 20.4–24.2 | A person is standing idle, gently swaying their hands. |
| `push_obstacle_270_002__A342_M` | 0–1 | A person steps forward and turns right. |
| `push_obstacle_270_002__A342_M` | 1–2.6 | A person takes a step forward and simultaneously turns right to push an obstacle with their side by leaning on it. |
| `push_obstacle_270_002__A342_M` | 2.6–16 | A person is pushing an obstacle forward with effort, walking sideways in the left direction, while leaning on it. |
| `push_obstacle_270_002__A342_M` | 16–20.4 | A person stops pushing an obstacle and turns left to stand upright. |
| `push_obstacle_270_002__A342_M` | 20.4–24.2 | A person is standing idle, gently swaying their hands. |
| `push_obstacle_270_002__A344` | 0–0.9 | A person steps forward and turns left. |
| `push_obstacle_270_002__A344` | 0.9–2.1 | A person takes a step forward and simultaneously turns left to push an obstacle with their side by leaning on it. |
| `push_obstacle_270_002__A344` | 2.1–18.45 | A person is pushing an obstacle forward with effort, walking sideways in the right direction, while leaning on it. |
| `push_obstacle_270_002__A344` | 18.45–22.25 | A person stops pushing an obstacle and turns right to stand upright. |
| `push_obstacle_270_002__A344_M` | 0–0.9 | A person steps forward and turns right. |
| `push_obstacle_270_002__A344_M` | 0.9–2.1 | A person takes a step forward and simultaneously turns right to push an obstacle with their side by leaning on it. |
| `push_obstacle_270_002__A344_M` | 2.1–18.45 | A person is pushing an obstacle forward with effort, walking sideways in the left direction, while leaning on it. |
| `push_obstacle_270_002__A344_M` | 18.45–22.25 | A person stops pushing an obstacle and turns left to stand upright. |
| `push_obstacle_270_003__A342` | 0–2 | A person steps forward and turns left. |
| `push_obstacle_270_003__A342` | 2–3.5 | A person takes a step forward and simultaneously turns left to push an obstacle with their side by leaning on it. |
| `push_obstacle_270_003__A342` | 3.5–16.3 | A person is pushing an obstacle forward with effort, walking sideways in the right direction, while leaning on it. |
| `push_obstacle_270_003__A342` | 16.3–19.55 | A person stops pushing an obstacle and turns right to stand upright. |
| `push_obstacle_270_003__A342` | 19.55–22.4 | A person is standing idle, gently swaying their hands. |
| `push_obstacle_270_003__A342_M` | 0–2 | A person steps forward and turns right. |
| `push_obstacle_270_003__A342_M` | 2–3.5 | A person takes a step forward and simultaneously turns right to push an obstacle with their side by leaning on it. |
| `push_obstacle_270_003__A342_M` | 3.5–16.3 | A person is pushing an obstacle forward with effort, walking sideways in the left direction, while leaning on it. |
| `push_obstacle_270_003__A342_M` | 16.3–19.55 | A person stops pushing an obstacle and turns left to stand upright. |
| `push_obstacle_270_003__A342_M` | 19.55–22.4 | A person is standing idle, gently swaying their hands. |
| `push_obstacle_270_003__A344` | 0–1.55 | A person steps forward and turns left. |
| `push_obstacle_270_003__A344` | 1.55–2.85 | A person takes a step forward and simultaneously turns left to push an obstacle with their side by leaning on it. |
| `push_obstacle_270_003__A344` | 2.85–15.85 | A person is pushing an obstacle forward with effort, walking sideways in the right direction, while leaning on it. |
| `push_obstacle_270_003__A344` | 15.85–19 | A person stops pushing an obstacle and turns right to stand upright. |
| `push_obstacle_270_003__A344` | 19–22.25 | A person is standing idle, gently swaying their hands. |
| `push_obstacle_270_003__A344_M` | 0–1.55 | A person steps forward and turns right. |
| `push_obstacle_270_003__A344_M` | 1.55–2.85 | A person takes a step forward and simultaneously turns right to push an obstacle with their side by leaning on it. |
| `push_obstacle_270_003__A344_M` | 2.85–15.85 | A person is pushing an obstacle forward with effort, walking sideways in the left direction, while leaning on it. |
| `push_obstacle_270_003__A344_M` | 15.85–19 | A person stops pushing an obstacle and turns left to stand upright. |
| `push_obstacle_270_003__A344_M` | 19–22.25 | A person is standing idle, gently swaying their hands. |
| `push_obstacle_270_101__A341` | 0–2 | A person standing upright, turns to the left and starting to push an obstacle. |
| `push_obstacle_270_101__A341` | 2–16.9333 | A person is pushing an obstacle using their right side and the left hand while walking to their right in a crisscross manner and coming to a stop. |
| `push_obstacle_270_101__A341` | 16.9333–19.5333 | A person turns to their right and stands upright. |
| `push_obstacle_270_101__A341_M` | 0–2 | A person standing upright, turns to the right and starting to push an obstacle. |
| `push_obstacle_270_101__A341_M` | 2–16.9333 | A person is pushing an obstacle using their left side and the right hand while walking to their left in a crisscross manner and coming to a stop. |
| `push_obstacle_270_101__A341_M` | 16.9333–19.5333 | A person turns to their left and stands upright. |
| `push_obstacle_270_101__A343` | 0–2 | A person standing is leaning the right shoulder on an obstacle placed forward. |
| `push_obstacle_270_101__A343` | 2–3.1 | A person leaning the right shoulder on an obstacle is starting to push it. |
| `push_obstacle_270_101__A343` | 3.1–15.83 | A person leaning on an obstacle with the right shoulder is pushing it towards their right. |
| `push_obstacle_270_101__A343` | 15.83–19.67 | A person leaning on an obstacle towards their right and pushing it is stopping, turning towards their right and standing. |
| `push_obstacle_270_101__A343` | 19.67–21.13 | A person standing is lifting the left leg a bit and standing in a neutral stance. |
| `push_obstacle_270_101__A343_M` | 0–2 | A person standing is leaning the left shoulder on an obstacle placed forward. |
| `push_obstacle_270_101__A343_M` | 2–3.1 | A person leaning the left shoulder on an obstacle is starting to push it. |
| `push_obstacle_270_101__A343_M` | 3.1–15.83 | A person leaning on an obstacle with the left shoulder is pushing it towards their left. |
| `push_obstacle_270_101__A343_M` | 15.83–19.67 | A person leaning on an obstacle towards their left and pushing it is stopping, turning towards their left and standing. |
| `push_obstacle_270_101__A343_M` | 19.67–21.13 | A person standing is lifting the right leg a bit and standing in a neutral stance. |
| `push_obstacle_270_102__A341` | 0–1.8 | A person standing upright, turns to the left and starting to push an obstacle. |
| `push_obstacle_270_102__A341` | 1.8–17.05 | A person is pushing an obstacle using their right side and the left hand while walking to their right in a crisscross manner and coming to a stop. |
| `push_obstacle_270_102__A341` | 17.05–19.15 | A person turns to their right and stands upright. |
| `push_obstacle_270_102__A341_M` | 0–1.8 | A person standing upright, turns to the right and starting to push an obstacle. |
| `push_obstacle_270_102__A341_M` | 1.8–17.05 | A person is pushing an obstacle using their left side and the right hand while walking to their left in a crisscross manner and coming to a stop. |
| `push_obstacle_270_102__A341_M` | 17.05–19.15 | A person turns to their left and stands upright. |
| `push_obstacle_270_102__A343` | 0–2.3 | A person standing is leaning the right shoulder on an obstacle placed forward. |
| `push_obstacle_270_102__A343` | 2.3–3.6 | A person leaning the right shoulder on an obstacle is starting to push it. |
| `push_obstacle_270_102__A343` | 3.6–13.75 | A person leaning on an obstacle with the right shoulder is pushing it towards their right. |
| `push_obstacle_270_102__A343` | 13.75–17.95 | A person leaning on an obstacle towards their right and pushing it is stopping, turning towards their right and standing. |
| `push_obstacle_270_102__A343` | 17.95–18.5 | A person standing is lifting the left leg a bit and standing in a neutral stance. |
| `push_obstacle_270_102__A343_M` | 0–2.3 | A person standing is leaning the left shoulder on an obstacle placed forward. |
| `push_obstacle_270_102__A343_M` | 2.3–3.6 | A person leaning the left shoulder on an obstacle is starting to push it. |
| `push_obstacle_270_102__A343_M` | 3.6–13.75 | A person leaning on an obstacle with the left shoulder is pushing it towards their left. |
| `push_obstacle_270_102__A343_M` | 13.75–17.95 | A person leaning on an obstacle towards their left and pushing it is stopping, turning towards their left and standing. |
| `push_obstacle_270_102__A343_M` | 17.95–18.5 | A person standing is lifting the right leg a bit and standing in a neutral stance. |
| `push_obstacle_270_103__A341` | 0–2.05 | A person standing upright, turns to the left and starting to push an obstacle. |
| `push_obstacle_270_103__A341` | 2.05–17 | A person is pushing an obstacle using their right side and the left hand while walking to their right in a crisscross manner and coming to a stop. |
| `push_obstacle_270_103__A341` | 17–19.9 | A person turns to their right and stands upright. |
| `push_obstacle_270_103__A341_M` | 0–2.05 | A person standing upright, turns to the right and starting to push an obstacle. |
| `push_obstacle_270_103__A341_M` | 2.05–17 | A person is pushing an obstacle using their left side and the right hand while walking to their left in a crisscross manner and coming to a stop. |
| `push_obstacle_270_103__A341_M` | 17–19.9 | A person turns to their left and stands upright. |
| `push_obstacle_270_103__A343` | 0–2.85 | A person standing is leaning the right shoulder on an obstacle placed forward. |
| `push_obstacle_270_103__A343` | 2.85–4.2 | A person leaning the right shoulder on an obstacle is starting to push it. |
| `push_obstacle_270_103__A343` | 4.2–20.35 | A person leaning on an obstacle with the right shoulder is pushing it towards their right. |
| `push_obstacle_270_103__A343` | 20.35–23.4 | A person leaning on an obstacle towards their right and pushing it is stopping, turning towards their right and standing. |
| `push_obstacle_270_103__A343` | 23.4–24.1 | A person standing is lifting the left leg a bit and standing in a neutral stance. |
| `push_obstacle_270_103__A343_M` | 0–2.85 | A person standing is leaning the left shoulder on an obstacle placed forward. |
| `push_obstacle_270_103__A343_M` | 2.85–4.2 | A person leaning the left shoulder on an obstacle is starting to push it. |
| `push_obstacle_270_103__A343_M` | 4.2–20.35 | A person leaning on an obstacle with the left shoulder is pushing it towards their left. |
| `push_obstacle_270_103__A343_M` | 20.35–23.4 | A person leaning on an obstacle towards their left and pushing it is stopping, turning towards their left and standing. |
| `push_obstacle_270_103__A343_M` | 23.4–24.1 | A person standing is lifting the right leg a bit and standing in a neutral stance. |
| `push_obstacle_360_001__A344` | 0–2.8 | A person turns back over their right shoulder and bends their legs. |
| `push_obstacle_360_001__A344` | 2.8–4 | A person bends their legs deeply to push an obstacle using their back. |
| `push_obstacle_360_001__A344` | 4–15.8 | A person is walking backward while taking several small steps at a slow pace in a seated-like position and pushing an obstacle using their back. |
| `push_obstacle_360_001__A344` | 15.8–24.8 | A person walking backward comes to a stop and stands in an upright posture. |
| `push_obstacle_360_001__A344` | 24.8–27.9 | A person turns backward over their left shoulder. |
| `push_obstacle_360_001__A344_M` | 0–2.8 | A person turns back over their left shoulder and bends their legs. |
| `push_obstacle_360_001__A344_M` | 2.8–4 | A person bends their legs deeply to push an obstacle using their back. |
| `push_obstacle_360_001__A344_M` | 4–15.8 | A person is walking backward while taking several small steps at a slow pace in a seated-like position and pushing an obstacle using their back. |
| `push_obstacle_360_001__A344_M` | 15.8–24.8 | A person walking backward comes to a stop and stands in an upright posture. |
| `push_obstacle_360_001__A344_M` | 24.8–27.9 | A person turns backward over their right shoulder. |
| `push_obstacle_360_002__A342` | 0–3 | A person standing upright slowly turns over their right shoulder and bends their legs. |
| `push_obstacle_360_002__A342` | 3–17.5 | A person with their legs bent slowly pushes an obstacle using their back. |
| `push_obstacle_360_002__A342` | 17.5–21.4 | A person stops pushing an obstacle with their back. |
| `push_obstacle_360_002__A342` | 21.4–24.5 | A person slowly turns over their left shoulder to face the opposite direction. |
| `push_obstacle_360_002__A342` | 24.5–25.6333 | A person stands upright while facing forward. |
| `push_obstacle_360_002__A342_M` | 0–3 | A person standing upright slowly turns over their left shoulder and bends their legs. |
| `push_obstacle_360_002__A342_M` | 3–17.5 | A person with their legs bent slowly pushes an obstacle using their back. |
| `push_obstacle_360_002__A342_M` | 17.5–21.4 | A person stops pushing an obstacle with their back. |
| `push_obstacle_360_002__A342_M` | 21.4–24.5 | A person slowly turns over their right shoulder to face the opposite direction. |
| `push_obstacle_360_002__A342_M` | 24.5–25.6333 | A person stands upright while facing forward. |
| `push_obstacle_360_002__A344` | 0–2.75 | A person turns back over their right shoulder and bends their legs. |
| `push_obstacle_360_002__A344` | 2.75–5.6 | A person bends their legs deeply to push an obstacle using their back. |
| `push_obstacle_360_002__A344` | 5.6–17.8 | A person is walking backward while taking several small steps at a slow pace in a seated-like position and pushing an obstacle using their back. |
| `push_obstacle_360_002__A344` | 17.8–22.55 | A person walking backward comes to a stop and stands in an upright posture. |
| `push_obstacle_360_002__A344` | 22.55–24.75 | A person turns backward over their left shoulder. |
| `push_obstacle_360_002__A344_M` | 0–2.75 | A person turns back over their left shoulder and bends their legs. |
| `push_obstacle_360_002__A344_M` | 2.75–5.6 | A person bends their legs deeply to push an obstacle using their back. |
| `push_obstacle_360_002__A344_M` | 5.6–17.8 | A person is walking backward while taking several small steps at a slow pace in a seated-like position and pushing an obstacle using their back. |
| `push_obstacle_360_002__A344_M` | 17.8–22.55 | A person walking backward comes to a stop and stands in an upright posture. |
| `push_obstacle_360_002__A344_M` | 22.55–24.75 | A person turns backward over their right shoulder. |
| `push_obstacle_360_003__A342` | 0–3.75 | A person standing upright slowly turns over their right shoulder and bends their legs. |
| `push_obstacle_360_003__A342` | 3.75–18.35 | A person with their legs bent slowly pushes an obstacle using their back. |
| `push_obstacle_360_003__A342` | 18.35–22.4 | A person stops pushing an obstacle with their back. |
| `push_obstacle_360_003__A342` | 22.4–26.35 | A person slowly turns over their left shoulder to face the opposite direction. |
| `push_obstacle_360_003__A342` | 26.35–28.1 | A person stands upright while facing forward. |
| `push_obstacle_360_003__A342_M` | 0–3.75 | A person standing upright slowly turns over their left shoulder and bends their legs. |
| `push_obstacle_360_003__A342_M` | 3.75–18.35 | A person with their legs bent slowly pushes an obstacle using their back. |
| `push_obstacle_360_003__A342_M` | 18.35–22.4 | A person stops pushing an obstacle with their back. |
| `push_obstacle_360_003__A342_M` | 22.4–26.35 | A person slowly turns over their right shoulder to face the opposite direction. |
| `push_obstacle_360_003__A342_M` | 26.35–28.1 | A person stands upright while facing forward. |
| `push_obstacle_360_003__A344` | 0–2.5 | A person standing upright slowly turns over their right shoulder and bends their legs. |
| `push_obstacle_360_003__A344` | 2.5–19.4 | A person with their legs bent slowly pushes an obstacle using their back. |
| `push_obstacle_360_003__A344` | 19.4–24.2 | A person stops pushing an obstacle with their back. |
| `push_obstacle_360_003__A344` | 24.2–26.45 | A person slowly turns over their left shoulder to face the opposite direction. |
| `push_obstacle_360_003__A344` | 26.45–27.95 | A person stands upright while facing forward. |
| `push_obstacle_360_003__A344_M` | 0–2.5 | A person standing upright slowly turns over their left shoulder and bends their legs. |
| `push_obstacle_360_003__A344_M` | 2.5–19.4 | A person with their legs bent slowly pushes an obstacle using their back. |
| `push_obstacle_360_003__A344_M` | 19.4–24.2 | A person stops pushing an obstacle with their back. |
| `push_obstacle_360_003__A344_M` | 24.2–26.45 | A person slowly turns over their right shoulder to face the opposite direction. |
| `push_obstacle_360_003__A344_M` | 26.45–27.95 | A person stands upright while facing forward. |
| `push_obstacle_360_005__A342` | 0–4.15 | A person standing upright slowly turns over their right shoulder and bends their legs. |
| `push_obstacle_360_005__A342` | 4.15–16.95 | A person with their legs bent slowly pushes an obstacle using their back. |
| `push_obstacle_360_005__A342` | 16.95–20.55 | A person stops pushing an obstacle with their back. |
| `push_obstacle_360_005__A342` | 20.55–22.95 | A person slowly turns over their left shoulder to face the opposite direction. |
| `push_obstacle_360_005__A342` | 22.95–25.3 | A person stands upright while facing forward. |
| `push_obstacle_360_005__A342_M` | 0–4.15 | A person standing upright slowly turns over their left shoulder and bends their legs. |
| `push_obstacle_360_005__A342_M` | 4.15–16.95 | A person with their legs bent slowly pushes an obstacle using their back. |
| `push_obstacle_360_005__A342_M` | 16.95–20.55 | A person stops pushing an obstacle with their back. |
| `push_obstacle_360_005__A342_M` | 20.55–22.95 | A person slowly turns over their right shoulder to face the opposite direction. |
| `push_obstacle_360_005__A342_M` | 22.95–25.3 | A person stands upright while facing forward. |
| `push_obstacle_360_005__A344` | 0–2.3 | A person turns back over their right shoulder and bends their legs. |
| `push_obstacle_360_005__A344` | 2.3–4.85 | A person bends their legs deeply to push an obstacle using their back. |
| `push_obstacle_360_005__A344` | 4.85–16.85 | A person is walking backward while taking several small steps at a slow pace in a seated-like position and pushing an obstacle using their back. |
| `push_obstacle_360_005__A344` | 16.85–21.35 | A person walking backward comes to a stop and stands in an upright posture. |
| `push_obstacle_360_005__A344` | 21.35–24.2 | A person turns backward over their left shoulder. |
| `push_obstacle_360_005__A344_M` | 0–2.3 | A person turns back over their left shoulder and bends their legs. |
| `push_obstacle_360_005__A344_M` | 2.3–4.85 | A person bends their legs deeply to push an obstacle using their back. |
| `push_obstacle_360_005__A344_M` | 4.85–16.85 | A person is walking backward while taking several small steps at a slow pace in a seated-like position and pushing an obstacle using their back. |
| `push_obstacle_360_005__A344_M` | 16.85–21.35 | A person walking backward comes to a stop and stands in an upright posture. |
| `push_obstacle_360_005__A344_M` | 21.35–24.2 | A person turns backward over their right shoulder. |
| `push_obstacle_360_101__A341` | 0–2.3 | A person turns back over their right shoulder and bends their legs. |
| `push_obstacle_360_101__A341` | 2.3–4.3 | A person bends their legs deeply to push an obstacle using their back. |
| `push_obstacle_360_101__A341` | 4.3–16.14 | A person is walking backward while taking several small steps at a slow pace in a seated-like position and pushing an obstacle using their back. |
| `push_obstacle_360_101__A341` | 16.14–20.86 | A person walking backward comes to a stop and stands in an upright posture. |
| `push_obstacle_360_101__A341` | 20.86–23.7 | A person turns backward over their left shoulder. |
| `push_obstacle_360_101__A341_M` | 0–2.3 | A person turns back over their left shoulder and bends their legs. |
| `push_obstacle_360_101__A341_M` | 2.3–4.3 | A person bends their legs deeply to push an obstacle using their back. |
| `push_obstacle_360_101__A341_M` | 4.3–16.14 | A person is walking backward while taking several small steps at a slow pace in a seated-like position and pushing an obstacle using their back. |
| `push_obstacle_360_101__A341_M` | 16.14–20.86 | A person walking backward comes to a stop and stands in an upright posture. |
| `push_obstacle_360_101__A341_M` | 20.86–23.7 | A person turns backward over their right shoulder. |
| `push_obstacle_360_101__A343` | 0–1 | A person is standing still. |
| `push_obstacle_360_101__A343` | 1–4 | A person turns back over their right shoulder and faces backward. |
| `push_obstacle_360_101__A343` | 4–6 | A person takes a few small steps forward while leaning the hips onto the obstacle behind them. |
| `push_obstacle_360_101__A343` | 6–28.5 | A person leaning back on the obstacle starts pushing it with their back by taking several small steps backward. |
| `push_obstacle_360_101__A343` | 28.5–31.3 | A person stops pushing the obstacle and stands upright by taking a step forward. |
| `push_obstacle_360_101__A343` | 31.3–33.9 | A person turns back over their left shoulder. |
| `push_obstacle_360_101__A343_M` | 0–1 | A person is standing still. |
| `push_obstacle_360_101__A343_M` | 1–4 | A person turns back over their left shoulder and faces backward. |
| `push_obstacle_360_101__A343_M` | 4–6 | A person takes a few small steps forward while leaning the hips onto the obstacle behind them. |
| `push_obstacle_360_101__A343_M` | 6–28.5 | A person leaning back on the obstacle starts pushing it with their back by taking several small steps backward. |
| `push_obstacle_360_101__A343_M` | 28.5–31.3 | A person stops pushing the obstacle and stands upright by taking a step forward. |
| `push_obstacle_360_101__A343_M` | 31.3–33.9 | A person turns back over their right shoulder. |
| `push_obstacle_360_102__A341` | 0–2.35 | A person turns back over their right shoulder and bends their legs. |
| `push_obstacle_360_102__A341` | 2.35–4.05 | A person bends their legs deeply to push an obstacle using their back. |
| `push_obstacle_360_102__A341` | 4.05–17.05 | A person is walking backward while taking several small steps at a slow pace in a seated-like position and pushing an obstacle using their back. |
| `push_obstacle_360_102__A341` | 17.05–20.35 | A person walking backward comes to a stop and stands in an upright posture. |
| `push_obstacle_360_102__A341` | 20.35–22.95 | A person turns backward over their left shoulder. |
| `push_obstacle_360_102__A341_M` | 0–2.35 | A person turns back over their left shoulder and bends their legs. |
| `push_obstacle_360_102__A341_M` | 2.35–4.05 | A person bends their legs deeply to push an obstacle using their back. |
| `push_obstacle_360_102__A341_M` | 4.05–17.05 | A person is walking backward while taking several small steps at a slow pace in a seated-like position and pushing an obstacle using their back. |
| `push_obstacle_360_102__A341_M` | 17.05–20.35 | A person walking backward comes to a stop and stands in an upright posture. |
| `push_obstacle_360_102__A341_M` | 20.35–22.95 | A person turns backward over their right shoulder. |
| `push_obstacle_360_102__A343` | 0.05–2.75 | A person turns back over their right shoulder and faces backward. |
| `push_obstacle_360_102__A343` | 2.75–4.45 | A person takes a few small steps forward while leaning the hips onto the obstacle behind them. |
| `push_obstacle_360_102__A343` | 4.45–21.25 | A person leaning back on the obstacle starts pushing it with their back by taking several small steps backward. |
| `push_obstacle_360_102__A343` | 21.25–24.45 | A person stops pushing the obstacle and stands upright by taking a step forward. |
| `push_obstacle_360_102__A343` | 24.45–28.05 | A person turns back over their left shoulder. |
| `push_obstacle_360_102__A343_M` | 0.05–2.75 | A person turns back over their left shoulder and faces backward. |
| `push_obstacle_360_102__A343_M` | 2.75–4.45 | A person takes a few small steps forward while leaning the hips onto the obstacle behind them. |
| `push_obstacle_360_102__A343_M` | 4.45–21.25 | A person leaning back on the obstacle starts pushing it with their back by taking several small steps backward. |
| `push_obstacle_360_102__A343_M` | 21.25–24.45 | A person stops pushing the obstacle and stands upright by taking a step forward. |
| `push_obstacle_360_102__A343_M` | 24.45–28.05 | A person turns back over their right shoulder. |
| `push_obstacle_360_103__A341` | 0–2.4 | A person turns back over their right shoulder and bends their legs. |
| `push_obstacle_360_103__A341` | 2.4–3.25 | A person bends their legs deeply to push an obstacle using their back. |
| `push_obstacle_360_103__A341` | 3.25–18.45 | A person is walking backward while taking several small steps at a slow pace in a seated-like position and pushing an obstacle using their back. |
| `push_obstacle_360_103__A341` | 18.45–21.85 | A person walking backward comes to a stop and stands in an upright posture. |
| `push_obstacle_360_103__A341` | 21.85–30.45 | A person turns backward over their left shoulder. |
| `push_obstacle_360_103__A341_M` | 0–2.4 | A person turns back over their left shoulder and bends their legs. |
| `push_obstacle_360_103__A341_M` | 2.4–3.25 | A person bends their legs deeply to push an obstacle using their back. |
| `push_obstacle_360_103__A341_M` | 3.25–18.45 | A person is walking backward while taking several small steps at a slow pace in a seated-like position and pushing an obstacle using their back. |
| `push_obstacle_360_103__A341_M` | 18.45–21.85 | A person walking backward comes to a stop and stands in an upright posture. |
| `push_obstacle_360_103__A341_M` | 21.85–30.45 | A person turns backward over their right shoulder. |
| `push_obstacle_360_103__A343` | 0.15–3 | A person turns back over their right shoulder and faces backward. |
| `push_obstacle_360_103__A343` | 3–5.35 | A person takes a few small steps forward while leaning the hips onto the obstacle behind them. |
| `push_obstacle_360_103__A343` | 5.35–24.05 | A person leaning back on the obstacle starts pushing it with their back by taking several small steps backward. |
| `push_obstacle_360_103__A343` | 24.05–26.85 | A person stops pushing the obstacle and stands upright by taking a step forward. |
| `push_obstacle_360_103__A343` | 26.85–29.6 | A person turns back over their left shoulder. |
| `push_obstacle_360_103__A343_M` | 0.15–3 | A person turns back over their left shoulder and faces backward. |
| `push_obstacle_360_103__A343_M` | 3–5.35 | A person takes a few small steps forward while leaning the hips onto the obstacle behind them. |
| `push_obstacle_360_103__A343_M` | 5.35–24.05 | A person leaning back on the obstacle starts pushing it with their back by taking several small steps backward. |
| `push_obstacle_360_103__A343_M` | 24.05–26.85 | A person stops pushing the obstacle and stands upright by taking a step forward. |
| `push_obstacle_360_103__A343_M` | 26.85–29.6 | A person turns back over their right shoulder. |

## 원본 BVH 검증

요청 71개 선별 추출 완료, 누락 0개. 기존 두 원본은 덮어쓰지 않았다. 71개 모두 parser/FK 검사에서 Root 포함 78노드·240채널·양수 frame time·정적 Root wrapper·위치/회전 유한값을 확인했다. 상세 수치는 `raw_validation.json`, 파일별 크기/SHA256은 `extraction_report.json`에 기록했다. 이 검사는 원본71개 대상이다. 후속180 계열21개의 retarget·물리 RSI 검증은 별도 완료했으며 정책 성능 검증은 실행하지 않았다.
