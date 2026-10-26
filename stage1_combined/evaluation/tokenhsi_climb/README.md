# 原本 TokenHSI CLIMB 기준 평가

`tokenhsi_climb_full_ft_test.sh <checkpoint>`를 사용한다. 기본512환경/seed0/프로세스 내3repeat/test split/걷기100%/300step/IET off. 물체는Box18·Cabinet9·Table_Square8종, 종류확률각1/3. 학습checkpoint에저장된 env/sim을읽고 split만test로설정한다. 모델은학습된GTA/35D모델그대로다.

launch.py는평가프로세스에서만point metrics를등록하고원본trans_players.py의run_eval 루프를복사하여사용한다. 원본루프에서저장위치와검증용trace기록만추가한다. 학습코드/weight를변경하지않는다. 첫완료episode만집계, 성공=root와metadata목표XYZ거리≤.2m ever,precision=성공한episode의최소거리평균. reset에서precision=inf로초기화한다. 기존relation/영역성공은공식지표로사용하지않는다.

trace_0/1/2.npz에는step별root/target/alive/done을저장한다. numpy float64로첫episode의ever성공/최소거리/종료를재채점할수있다. 0회차를포함한3회원시수치를보존하고프로젝트관례에따른주요요약은1·2회차평균이다. 반복간산포는독립학습seed산포가아니다.

실제2026-10-25 평가: output_etc/check/tokenhsi_climb_20k_official_20261025/. contract.json, env.yaml, eval.log, metrics_*.json과trace를확인한다. 초기어댑터들여쓰기오류로그는failed_adapter.log에보존한다.

78k zero-shot대조: `prepare_zeroshot.py --source <78k> --template <FT체크포인트> --output <평가용pth>`로FT시작시와동일한가중치확장만한다. 원본모델전체key/177tensor와정규화통계는78k에서가져오며추가5입력열은0,학습update없음. template는텐서형태와평가config/contract만제공한다. 변환원장JSON보존. 생성checkpoint를같은test wrapper로평가한다. 결과 `output_etc/check/tokenhsi_climb_78k_zeroshot_20261025/REPORT.md`: zero-shot80.57%,FT80.47%;성공최소거리8.249→3.638cm. 2명/4물체조건은이번평가에포함하지않았다.

FT중간checkpoint비교는 `bash evaluation/tokenhsi_climb/run_checkpoints.sh [fresh_output]`로5k/10k/15k를같은실제혼합환경에서평가한다. 기본조건은1명1고정물체/Box·Cabinet·Table_Square각1/3/test/10초/512환경/seed0/3repeat다. `python evaluation/tokenhsi_climb/summarize_checkpoints.py <output>`는기존zero-shot·20k endpoint도전trace재채점/환경파일exact동일성검증후재사용하여5조건REPORT/summary를작성한다. endpoint경로는스크립트에명시되어있다. 실행산출물: output_etc/check/tokenhsi_climb_ft_sweep_20261025/. 새학습은실행하지않는다.

## FT5k 주변 저장본 비교

`python -m evaluation.tokenhsi_climb.sweep_near_5k --output output_etc/check/<새 경로>`는 FT4500~5500을100회 간격으로 비교한다. 기존 FT5000은 동일조건의 검증 결과를 재사용하고 나머지10개는GPU0에서2개씩 병렬 평가한다. 조건은위의512환경/seed0/3repeat/10초와동일하다. `summary.json`, `REPORT.md`, `best_checkpoint.json`에 관측 최고를기록한다. 단일seed 반복의선택이므로 근소한차이의일반적우위는확정하지않는다.


## 점·영역 동시 평가

`POINT_REGION=1 OUTPUT_PATH=output_etc/check/<새 경로> bash tokenhsi/scripts/multi_agent/tokenhsi_climb_full_ft_test.sh <checkpoint>`는 기존 점 평가 루프의 같은 궤적에 `evaluation/finetuned/backend.py::classify` 영역식을 적용한다. 영역은 물체로컬XY각축 내부80%와 목표Z±20cm이며 점은3D거리20cm다. 첫episode 종료frame 포함 ever,기본512환경/seed0/3repeat/10초 유지. trace에물체pose/bbox/metadata높이를추가해독립재채점하고point_region.json을기록한다.

원본78k30D zero-shot은30d_full_ft_test.sh에원본pth를직접전달한다. checkpoint에chairFT환경설정이없는경우30D CLIMB환경YAML만평가용으로선택하며모델state/정규화는원본pth에서직접load한다. 가중치복제·35D확장은없다. POINT_REGION=1은같은10초궤적의영역도집계한다.
