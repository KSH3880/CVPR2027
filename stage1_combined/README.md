# stage1_combined

현재 기본 학습은 **78000 weight 기반 chair SIT 전체 모델 fine-tuning**이다.

1명·의자1개×2048환경, GPU5,20,000iteration. 원본TokenHSI SIT의360도배치·1~5m거리·걷기/SIT참조초기화50:50·보상·약10초/IET를적용한다. 정책입력은기존35Dchair/GTA규약유지. AMP는걷기12개/SIT20개모션만로딩하며CLIMB는제외한다. 기존nativeRSI캐시및정면제한은사용하지않는다.

[실행명령](markdowns/config.md) · [설정/검증](markdowns/reference/tokenhsi_sit_ft.md) · [현재실행기록](tokenhsi_sit_active_run.json). 기존2명/4의자학습과그체크포인트·로그는보존하며해당학습은사용자요청으로종료했다.

초기체크포인트/의자에셋은이폴더에있고,공용motion/SMPL·IsaacGym/conda는공유경로를사용한다. ksh와원본baseline은수정하지않는다. [물체별독립평가](evaluation/object_comparison/README.md)는별도비교용코드다.
