# Stage1와 BEFORE Multi-Agent TokenHSI

SIT/CLIMB/AT/ON_TOP 및 PUSH를 학습한다. Stage1·Stage1 PUSH와 역할별 object 토큰을 쓰는 BEFORE 세 설정과 solo PUSH를 추가한 `stage2_before_push`·공동 PUSH `stage2_coupled_push`, 거리 변형 `stage2_before_push_near50`·`stage2_coupled_push_goal5`, 개인 목표·공통 4면 접촉 `stage2_coupled_push_individual_goal`와 단독 거리 변형 `stage2_coupled_push_individual_goal_solo5`·운반 거리 변형 `stage2_coupled_push_individual_goal_solo5_carry5`·지면 운반 변형 `stage2_coupled_push_individual_goal_solo5_carry5_ground`, BEFORE 쌍 선택50·접근5m `stage2_before_push_near50_pair50_source5`와 공동 attention 개수 보정 두 설정, ground 관계 fusion A/B 두 설정을 유지한다. 전체 이름과 학습·추론·VNC 명령은 [config](markdowns/config.md)에 있다.

- [추가 옵션](markdowns/USAGE.md): conda·데이터·Stage1 78000·resume·추론 초기화.
- [코드 구조](markdowns/structure.md), [변경 기록](changelog.md), [작업 규칙](AGENTS.md), [과거 기록](markdowns/legacy/README.md).

현재 창고 포함 센서 학습은 `stage2_sensor_900_warehouse_coop`로 **4명·1024환경·8토큰·900step**이다. wrapper는 기본 conda `tokenhsi`와 지정한 `TOKENHSI_GPU`를 사용한다. 본학습은 `output/`, 확인·평가·viewer는 `output_etc/check/`에 저장한다. Distill9001.5k를 전이하고 actor encoder/coordination·sigma·입력 RMS를 동결하며 센서·action head·critic 전체·AMP 및 AMP/value RMS를 학습한다. 기존 세 장면 컨피그도 유지한다. [센서 설정](markdowns/reference/stage2_sensor_900.md)과 [추가 옵션](markdowns/USAGE.md#stage2-sensor-900)을 따른다.

실행에 쓰는 모션과 기본 전이 체크포인트 경로는 [데이터와 체크포인트](markdowns/USAGE.md#데이터와-체크포인트)에 정리했다. 원본 `stage1_message`의 코드·가중치·결과·프로세스는 보존한다.

## Acknowledgements

[TokenHSI](https://github.com/liangpan99/TokenHSI) by Liang Pan, Zeshi Yang, Zhiyang Dou, Wenjia Wang, Buzhen Huang, Bo Dai, Taku Komura and Jingbo Wang을 기반으로 한다. 원본은 ASE·PADL·InterScene에 기반한다. 원본 저작권과 [MIT License](LICENSE)를 유지하며 외부 라이브러리·데이터셋은 각 라이선스를 따른다.
