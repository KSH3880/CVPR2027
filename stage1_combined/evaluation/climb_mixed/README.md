# CLIMB 원본 물체 혼합 평가

이 방식은 보존용이다. 현재 기본은 [종류별 독립 SIT/CLIMB 비교](../object_comparison/README.md)를 사용한다.

원본 `ObjectLib`와 동일한 확률: **Box / Cabinet / Table_Square 각 1/3**, 종류 내부 모델 균등 복원 추출(`torch.multinomial`). 테스트 모델은18/9/8종. 512환경의 실제 개수를 171/171/170으로 강제하지 않는다. 원본 category 순서와 모델 directory enumeration을 catalog에 기록한다. 물체 모델은 환경 생성시 한 번 선택해 고정한다.

SIT·의자 별도 평가는 변경하지 않는다. 이번 코드는 CLIMB 전용이다.

```bash
cd /home/hwanhee/juan/CVPR2027/stage1_combined
# CPU sampler 검증
/home/hwanhee/anaconda3/envs/tokenhsi_jhh/bin/python evaluation/climb_mixed/test_sampling.py
# 코드/자산 준비만
python3 evaluation/climb_mixed/run_eval.py --prepare-only
# 두 모델 각16환경×seed0, 900step 구현 검증
python3 evaluation/climb_mixed/run_eval.py --smoke
# 두 모델 각512환경×독립seed0/1/2 정식 평가, GPU5 순차 실행
python3 evaluation/climb_mixed/run_eval.py
```

공개 TokenHSI Stage1 40k와 우리73k를 고정해 같은 초기장면/관절DOF/물체 배정으로 비교한다. 공개 weight는 `/home/hwanhee/CVPR2027/TokenHSI/output/tokenhsi/ckpt_stage1.pth`, 우리 weight는 `checkpoints/Stage1V2Fixed_00073000.pth`. 정책별 학습 없음. 초기화 loco,1명1고정물체,최대30초 첫episode 종료frame 포함 ever. 점은 골반/root의 목표3D거리≤20cm, 영역은 물체XY각축내부80%와 목표Z±20cm. 목표는 원본 `object_z + config.tarSitPos.z + .94`. 실제 native mesh/bbox를 사용한다. 학습용 크기 eligibility 때문에 원본 자산을 제외하지 않도록 평가 전용 proxy만 사용한다. 실제 크기/관측/물리는 바꾸지 않는다.

실제 category/model 배정과 확률은 각 `manifest.json`의 sampling 및 `results.json`의 category_counts/per_object/per_category에 남는다. 전체 mixture의 seed별 성공률과 평균±표본표준편차가 주 결과이며, category별 평균/표준편차는 추가 진단이다. `REPORT.md`, `summary.json`, 초기장면/trace,독립재채점 및 hash 검증을 남긴다. --smoke는 정식 통계를 만들지 않는다.

평가 backend는 이전에 검증한 metadata 평가 사본 `runs/table_metadata_climb_20261022/source`를 복사해 `runs/climb_mixed_eval_20261022/source`에 준비한다. 원본 baseline/ksh는 수정하지 않는다. `--source-template`, `--asset-root`, `--workspace`, `--checkpoint`로 명시적 경로를 바꿀 수 있다. 기존 backend 코드/입력이 달라지면 새 workspace를 요구하며 덮어쓰지 않는다. 코드 변경 후에는 기존 source를 재사용하지 말고 `--workspace`에 새 경로를 지정한다. 본 코드의 원본 확률/metadata 목표 반영은 논문의 시간/초기화까지 전부 재현했다는 의미가 아니다.

검증 완료: CPU 원본 sampler weights/추출 seed0/1/2 exact, GPU5 두정책 각16env×seed0. 실제배정 Box6/Cabinet5/Table5 동일, 초기장면·관절DOF 전체배열 exact, 재채점/hash/float32float64/종료frame 검증 통과. 결과는 runs/climb_mixed_eval_20261022/eval_20261022_204549_smoke/. 소규모 결과는 정식 성능수치로 사용하지 않는다.
