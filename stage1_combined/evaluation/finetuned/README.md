# 새 tokenizer fine-tuned 정책 평가

`object_comparison`의 Chair/Table 목표·점/영역 성공식을 현재35D tokenizer checkpoint에 연결하는 평가 어댑터다. 기존73k/public40k backend 사본 없이 현재 코드와 native test 자산으로 실행한다. 학습 코드는 변경하지 않는다.

- Chair26/Table8 각각 독립 SIT/CLIMB, 종류 내 균등 복원 추출.
- 1명/1고정물체, loco 시작, 최대900step. 첫 episode의 초기 상태와 종료 frame을 포함한 ever 성공.
- root 목표거리≤20cm, 영역은 좌판/상판 로컬XY 내부80% 및 목표Z±20cm. feet/contact/유지시간 조건 없음.
- Chair 좌판+.10/.94, Table bbox상판+.10/metadata+.94. 학습 목표와 실행 중 직접 대조.
- 정식 조건별512환경×seed0/1/2. 평균과 시드 표본표준편차. smoke는4환경×seed0이며 성능 통계로 사용하지 않는다.
- checkpoint를 실행 폴더로 복사하고 SHA256/epoch를 기록. 초기장면·전체root/object/alive/done trace 보존 및 독립 float64 재채점 검증.
- 현재 fine-tuned 정책만 평가한다. 기존/public 정책과 초기장면을 exact로 맞춘 paired 비교나 논문 benchmark 재현은 아니다. 이전 서버 backend의 초기화와 동일성도 검증하지 않았다.

저장소 루트 `stage1_combined`에서 실행:

```bash
python evaluation/finetuned/run_eval.py --checkpoint <checkpoint.pth> --smoke
python evaluation/finetuned/run_eval.py --checkpoint <checkpoint.pth>
```

`--gpu` 기본0, `--output`은 새 `output_etc/check/` 실행 경로를 지정한다. `summary.json`/`REPORT.md`에 결과를 기록한다. 기존 출력 폴더는 재사용하지 않는다. `launch_case.py`는 해당 프로세스에서만 평가 task/player를 등록한다.
