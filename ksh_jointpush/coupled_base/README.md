# Stage1와 BEFORE Multi-Agent TokenHSI

SIT/CLIMB/AT/ON_TOP 및 PUSH를 학습한다. Stage1·Stage1 PUSH와 역할별 object 토큰을 쓰는 BEFORE 세 설정과 solo PUSH를 추가한 `stage2_before_push`·공동 PUSH `stage2_coupled_push`, BEFORE PUSH의 거리 변형 `stage2_before_push_near50`를 유지한다. 전체 이름과 학습·추론·VNC 명령은 [config](markdowns/config.md)에 있다.

- [추가 옵션](markdowns/USAGE.md): conda·데이터·Stage1 78000·resume·추론 초기화.
- [구역별 goal 공동 PUSH 로컬 실행](LOCAL_RUN.md): Git 다운로드·외부 데이터/21k 준비·새 학습·로컬 viewer.
- [코드 구조](markdowns/structure.md), [변경 기록](changelog.md), [작업 규칙](AGENTS.md), [과거 기록](markdowns/legacy/README.md).

학습은 Stage1/PUSH **2명·2048환경·4물체**, BEFORE/coupled **4명·1024환경·8토큰**이며 기본 제한 시간은 Stage1 900step, Stage2 1200step이다. wrapper는 기본 conda `tokenhsi`와 지정한 `TOKENHSI_GPU`를 사용한다. 본학습은 `output/`, 확인·평가·viewer는 `output_etc/check/`에 저장한다.

공유 모션과 Stage1 checkpoint는 외부 원본에 심링크로 연결한다. 원본 `stage1_message`의 코드·가중치·결과·프로세스는 보존한다.

## Acknowledgements

[TokenHSI](https://github.com/liangpan99/TokenHSI) by Liang Pan, Zeshi Yang, Zhiyang Dou, Wenjia Wang, Buzhen Huang, Bo Dai, Taku Komura and Jingbo Wang을 기반으로 한다. 원본은 ASE·PADL·InterScene에 기반한다. 원본 저작권과 [MIT License](LICENSE)를 유지하며 외부 라이브러리·데이터셋은 각 라이선스를 따른다.
