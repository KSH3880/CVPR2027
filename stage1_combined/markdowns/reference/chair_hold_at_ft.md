# Chair HOLDING+AT 준비

사용자 실행 승인 후 GPU5에서 2048환경 1회 검증·저장·자연종료를 완료하고 본학습을 시작했다. 실행 기록은 `../../chair_hold_at_active_run.json`, 상세는 상위 `runs/chair_hold_at_78k_20261026_135444/REPORT.md`를 따른다. MPS는 기존 runtime의 검증된 연결만 사용하며 종료/재시작 명령은 없다.

- 기반: checkpoints/Stage1V2Fixed_00078000.pth, actor/critic object tokenizer 입력30D 및 전체165개 텐서 strict 전이. 원래 scene/AMP/value normalizer 복원. sit와 동일한 전체층 FT, lr2e-5, 20k회,500회 저장.
- 1명/동적의자1개/2048환경/spacing5/600step. 환경별 train ArmChair_Normal 모델 고정 균등복원추출. 원본 물리 에셋/질량 사용, fix_base_link=False. 초기거리1~5m/정면이사람향함/바닥+5cm, 목표XY±4.5m/바닥/사람과물체 squared distance>=2. loco_more/loco50:50 초기화, 가구에 cuboid RSI 캐시·추가 body-point rejection 미적용.
- HOLDING_AT100%, reward는78k stage1_v2_fixed 관계 설정/함수 그대로. 원본 TokenHSI carry 보상으로 교체하지 않음. AMP expert loco_more/loco/pickUp/carryWith/putDown=.5/.2/.1/.1/.1,carry family0·3차원라벨유지.
- object tokenizer는 속도3+각속도3+bbox24=30D. 위치/회전은기존GTA7D pose,목표는별도노드/관계packet으로전달. front/좌판/mesh 추가입력없음.
- `HOLD_AT_OBJECT_ROOT`는 `train/ArmChair_Normal` 및 `test/ArmChair_Normal`이 있는 원본 dataset_dynamic_objects 경로. 원본 공유 `TokenHSI/tokenhsi/data/dataset_dynamic_objects`에서 train63/test27 확인. 실행 에셋은 `runs/chair_hold_at_78k_20261026_135444/assets` 독립 사본이며 ksh/공유 원본은 읽기 전용으로 보존한다. train은 runtime/GPU연결전 경로없으면exit2.
- CPU strict 로딩·추론/normalizer·관계설정동일/부분reset·front·목표분포 확인. CPU와 모든 참조 모션 경로 확인 완료. GPU 검증 및 본학습 실행 기록은 해당 runs 폴더에서 관리한다.
- 실행은 config.md. 신규 산출물은 상위 runs/ 아래, 확인실행은 그 안 output_etc/check/ 경로. 학습과평가동일env저장본사용.
