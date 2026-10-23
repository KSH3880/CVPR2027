# stage1_combined

73000 체크포인트에서 의자 SIT/CLIMB용 object tokenizer만 미세조정하는 독립 코드 폴더.

입력35D: 로컬 선속도3 + 각속도3 + bbox 꼭짓점24 + 등받이XY2 + 좌판XYZ3. MLP35→256→128→64. actor/critic 의자 토크나이저 외 기존 가중치165개와 정규화3종은 동결. 2명×2048환경, 사용 의자2개+주변box/table/chair2개. 기본20,000iteration, GPU5.

학습·추론 명령: [markdowns/config.md](markdowns/config.md).
의자/주변 테이블 에셋과 초기 체크포인트는 이 폴더에 실제 복사. 공용 motion/SMPL 데이터와 IsaacGym/conda는 기존 공유 경로를 사용. ksh 및 baseline 수정 없음.

기존 GPU5 의자 토크나이저 학습은 사용자 요청으로 종료했고 체크포인트·로그를 보존했다. [실행 기록](../runs/chair_bbox_ft_20261022/active_run.json). 이 폴더에서 중복 학습은 시작하지 않았다. 이후 실행은 이 폴더 기준.

물체 종류별 기본 평가: [Box/Chair/Cabinet/Table 독립 SIT·CLIMB 비교](evaluation/object_comparison/README.md). Box는 원본 TokenHSI 테스트18 크기를 사용한다.

현재 로컬 본학습은 `chair_table_eval_full_ft`로 GPU0에서 73000부터 전체 레이어를 학습한다. Chair CLIMB는 좌판+.94, Table CLIMB는 metadata+.94로 object_comparison 목표를 따른다. SIT는 유지한다. [실행 기록](../runs/chair_table_eval_full_ft_20261022_212639/active_run.json), [로그](../runs/chair_table_eval_full_ft_20261022_212639/train.log).
