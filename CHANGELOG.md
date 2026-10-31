# 변경 기록

## 2026-10-31

### 확정 설정 문서 정리

- METHOD에 PUSH 토큰397D/전용 tokenizer, AMP21+4개·50:50·10프레임, RSI949후보·4모션·후반 선택·원본 자세 보존·잔여2~10m를 상세 기록했다. Teacher/student 초기 모델과26k 환경 기준을 구분했다.
- 기존3과제 native KD·TRAJ 제외와 프로세스 격리 후 CPU 검증 상태를 반영하고 METHOD/README의 오래된 미검증 문구를 정리했다. 문서만 변경했으며 코드·학습 설정은 유지했다.

### 격리 pool CPU 검증 완료

- native pool은spawn별도프로세스·NumPy array pipe로연결한다. 상대main경로/음수seed/공유메모리socket제약을수정했다.
- 참조2scene(4명)+native2환경의reset/397D관측/물리3step finite와실제student행동3step·native teacher KL0.0을확인했다. CPU12테스트통과. GPU/PPO본학습은미실행. 로그:output_etc/check/reference_integration/{mixed_isolated,policy_isolated}.log.


### PUSH reference scene 연결

- 사용자요청대로PUSH를26k의실제2인4상자mixed 환경에연결했다. namespace snapshot/원본sampling·support/경로거부·heading·CPA비용·RSI/reward사용, actor관측만TokenHSI397D로변환한다. Native3과제KD pool을별도유지하며TRAJ제외.
- PUSH/KD/참조비PUSH 슬롯을별도mask로구분하고6144policy row(참조2048scene×2 + native2048)로batch를맞췄다. source/환경config를resume계약에기록하고문서를현재구성으로갱신했다.
- CPU 단독참조PUSH2scene/4명 reset·3물리step finite통과. mixed reset의기존과제cache생성후같은프로세스의두번째simulator생성에서segmentation fault. 추가물리실행중단후native simulator를spawnworker/CPU IPC로격리했다. 격리후전체mixed물리연결·GPU/PPO는미검증이다. 로그:output_etc/check/reference_integration/mixed.log.


- 사용자 요청으로 TRAJ를 학습/KD/평가에서 제외. 기존 풀은 SIT/CARRY/CLIMB 균등 reset이며 PUSH50% 풀은 유지한다. 관측/tokenizer 슬롯은 체크포인트 호환을 위해 보존한다.

- TokenHSI baseline 독립 snapshot에 PUSH PPO+AMP / 기존4과제 Gaussian forward KL 분리 구현. Teacher/student 초기 가중치 준비. 원본 프로젝트 변경 없음.
- Stage1 26k의167크기·밀도3·마찰0.4·목표2~10m/±45°·RSI949후보·AMP50:50·CONTACT+AT·20cm/Z1mm 성공 연결. 1인 장면과2인 원본의 차이를 METHOD에 명시.
- CPU6테스트 통과: real checkpoint strict 로드와 기존행동 보존, KL/gradient 분리, 실제 training method의 optimizer step, reward/데이터 계약. Python compileall/셸 문법 통과.130모션 경로 존재 및 PUSH21개·asset/mesh hash 일치.
- Isaac Gym full task import는 외부 읽기 전용 JIT cache lock에서 중단. Simulator/rollout/학습 통합·장기 성공률 미검증. 본학습 실행 없음.
