# Partition Push 로컬 모션 데이터

`dataset_bones_push180/`은 `output/`에 받은 변환 완료 데이터를 옮긴 디렉터리다.
19개 모션마다 `phys_humanoid_v3/ref_motion.npy`, `conversion_report.json`,
`preview_data.json`이 있다. 이동 전후 57개 파일의 SHA-256 일치를 확인했다.

실행 포트의 `../ports/koo_push/tokenhsi/data/dataset_bones_push180`은 이 디렉터리의
상대 심링크다. AMP 목록은 해당 포트의 `partition_push_motions.yaml`이다.
Push180 19개와 기존 로컬 데이터에 연결한 loco 4개를 사용한다.
이전 목록의 별도 `dataset_bones_push` 2개는 사용하지 않는다.

`../output/raw_push_amp/`의 raw NPZ는 보관만 하며 학습에서는 읽지 않는다.
Stage1 checkpoint는 `../output/OpenPushPush180ContactAtWarmstart_00029000.pth`에 있다.
데이터 본체는 Git 추적에서 제외한다.
