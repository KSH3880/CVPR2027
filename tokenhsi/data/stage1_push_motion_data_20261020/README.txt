현재 두 Y1.8m PUSH 학습의 모션 데이터 묶음입니다.

- PUSH: manifest에 지정한 ref_motion.npy 21개 (dataset_bones_push180 19개 + dataset_bones_push 2개).
- 보행: 실제 학습에서 사용하는 AMASS ref_motion.npy 4개.
- 원본 dataset_stage1_push.yaml, 손 중점≤1m RSI 선별 cache, 두 실험의 env/train YAML.
- 압축을 풀면 tokenhsi/data/... 경로를 유지합니다. 심링크 대신 실제 파일을 넣었습니다.
- FILE_SHA256.json은 실제 파일 SHA256입니다.
- 모션은 학습용 phys_humanoid_v3 SkeletonMotion NPY 포맷입니다. BVH 원본은 포함하지 않습니다.
- 실행 코드·humanoid asset·Isaac Gym·checkpoint는 별도입니다. RSI pool만≤1m로 선별하며 원본 AMP 모션 파일은 수정하지 않았습니다.
