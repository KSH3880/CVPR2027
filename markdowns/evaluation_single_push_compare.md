# MD 기준 1인 평가: 원본78000 vs PUSH 파인튜닝9000

정의의 기준은 사용자 문서 [evaluation_single.md](evaluation_single.md)다. **기존 2인 환경에서 얻은51.7%/57.1% PUSH 및 AT/ON_TOP 결과는 이1인 평가 결과가 아니며, 원본78000 비교에서 제외한다.** 원래 문서는 수정하지 않는다.

## 실행 조건

| 항목 | 적용 |
|---|---|
| 사람 | 실제물리actor1명. 둘중한명만마스킹하거나두명결과를나누는방식아님 |
| 물체 | AT/SIT/CLIMB/PUSH는1개,ON_TOP은payload+받침2개. 보조선반은별도 |
| 기존4과제 | 각각900step(명목30초/보통899step),seed0/1/2각512환경 |
| PUSH | 사용자후속지정1800step(명목60초/보통1799step),seed0/1/2각512환경 |
| 시도수 | 과제/모델당1536개독립환경의첫episode |
| 정책 | 원본78000pth와FT9000pth를직접strict복원. 기존가중치/정규화변경없음 |
| 시작 | 기존4과제는원본문서의loco모션, PUSH는push_loco4개로보행시작 |
| 성공 | 원본문서의점/영역ever를동일궤적에적용. 초기상태성공미가산·종료frame포함 |
| 조기종료 | 본인낙상또는시간종료. 다른사람의낙상에의한동시종료없음 |
| 통계 | seed별성공률평균±표본표준편차ddof1,기존4과제는FT−원본차이도표시 |

AT9규격/최소1개포함후복원추출,기존과제의크기범위·그리드,사람→상자반경분포,목표square/rejection,출발payload와AT목표높이,ON_TOP동적받침,SIT/CLIMB고정상자,바닥/PhysX/제어간격,점20cm/영역내부80%+Z20cm는원본문서를따른다. 원본문서에없는추가bbox간격선별이나접촉/발높이/유지시간조건을더하지않는다.

## 원본과 FT의 공통 초기 상태

원본78000에저장된env/sim과기존reset/loco경로로과제·seed별초기상태를CPU에서한번생성한다. root·관절위치속도·상자/크기·목표·선반·binding·motionID/time을한파일에저장하고두모델의GPU평가가그파일을공유한다. 같은시드라는이유만으로초기조건동일을주장하지않고같은archive SHA를실행manifest에기록한다. CPU/GPU의난수backend가다르므로이전historical single_match archive와bitwise동일이라고주장하지않는다. 프로토콜의샘플링분포와조건을맞춘새공통장면비교다.

GPU고정actor는시뮬레이션준비후root tensor만바꾸어도실제collider가이동하지않는다. 따라서샘플링된초기좌표에고정상자와선반을**생성시배치**한다. 첫episode동안SIT/CLIMB실제상자pose불변을검사한다. 초기rigid-body buffer는GPU첫simulate전까지이전생성좌표일수있으므로초기거리/관절은실제reset root/kinematic archive를사용한다.

## 정책 복원

공유Transformer는개체수에따라가중치크기가바뀌지않는다. 먼저원본체크포인트의네트워크/정규화를strict복원한뒤개체수만1인으로retarget한다. 모든학습parameter가저장본과완전히같음을검사한다. actor/category/RMS값을다시학습하거나원본에PUSH행을추가하지않는다. 원본은기존4과제만평가하며PUSH는FT모델만평가한다.

1인/물체1개는3token/280D,ON_TOP은4token/317D다. 각각의scene normalizer는기존Human223D/Object30D통계를그대로공유한다. 최초reset관절을계속유지하는검사용옵션은사용하지않고정책PD action으로rollout한다.

## PUSH 확장

PUSH는원본문서에없으므로위1인규칙에사용자지정조건을추가한다. 상자167크기/20kg/마찰.4,사람→상자중심반경U(1,10)m,goal=env origin의XY±4.5m·사람/상자와각최소1m·상자로컬+Y정면±45°거부샘플링이다. 1인이므로다른사람/다른담당상자경로간격조건은없다. 목표는고정한다.

PUSH점ever는상자중심→목표3D거리20cm이내도달이며영역지표도같다. 거리구간은실제초기상자→목표XY거리1m간격이고표본수·성공수·성공률·진행률·잔여거리·도달시간을저장한다. 접촉반력/이력을성공조건에추가하지않는다.

## 명령과 결과

```bash
TOKENHSI_GPU=0 bash tokenhsi/scripts/multi_agent/evaluate_single_match_compare.sh
```

출력: `output_etc/check/single_match_78000_vs_push9000_20261026`.

- `run_manifest.json`:실제1인·모델·시드·진행상태/종료코드.
- `RESULTS.md`/`summary.json`:모델별점/영역성공률및FT−원본차이,완료run마다갱신.
- `initial_states/`:모델간공유초기상태npz/생성로그.
- `source78000/<task>_seed*/` 및 `push_ft9000/<task>_seed*/`:원시궤적·관절초기상태·manifest·결과·독립재채점검증.

기존4과제는원본/FT를동일장면에각각실행한다(24run). 이후PUSH를FT만3run실행한다. 평가중학습하지않고GPU1학습에연결하지않는다. 완료후예전2인결과와섞지않고이폴더의값으로보고한다.

## 검증 상태

원본78000의실제1인CPU strict복원/짧은rollout,FT9000의GPU0실제1인·상자1개·SIT정책120step/고정상자/독립재채점검사를통과했다. 나머지과제경로와본평가는후속확인범위만기록한다.
