# Stage2 Coupled Carry · SA4

Human SA4와 SA4 GraspView 두 모델을 위한 공동운반 PPO/AMP 저장소다.
공동80%·단독20% 환경과 Stage1 CarryDistill 전이를 공유한다.

| 모델 | 관측 |
| --- | --- |
| Human SA4 | 실제 payload 중심634차원 → human SA4 |
| SA4 GraspView | 사람별 반대 파지 앵커 관측1268차원 → 자기 human → SA4 |

- [학습·로컬 평가·서버 VNC](markdowns/config.md)
- [코드 구조](markdowns/structure.md)
- [변경과 검증 결과](changelog.md)
- [데이터 준비](joint_carry/README.md)

Isaac Gym이 설치된 `tokenhsi` conda 환경과 원본 TokenHSI 데이터가 필요하다.
Stage1 CarryDistill30000 PTH는 별도로 지정한다. 데이터 원본과 다른 저장소는 수정하지 않는다.
학습은2명·2048환경·4물체다. `stage2`/`stage2_grasp`은 기존 AT 조건을 유지하고,
`stage2_at20`/`stage2_grasp_at20`은 AT 성공 거리20cm·높이 차이1mm 조건을 제공한다.
모델 또는 성공조건이 다른 Stage2 체크포인트는 서로 재개·평가하지 않는다.

기반 구현: [TokenHSI](https://github.com/liangpan99/TokenHSI).
보정 locomotion 데이터의 TeamHOI 라이선스는 [LICENSE.TeamHOI](joint_carry/teamhoi_reference/LICENSE.TeamHOI)에 있다.
원본 코드 라이선스는 [LICENSE](LICENSE)를 따른다.
