# Stage2 V4 reward 로컬 확인

`/home/hwanhee/ksh/stage1_2_base_teacher/output/stage2_900_v4_reward`의 독립 복사본이다.
원본은 읽기만 했으며 코드·가중치·결과를 수정하지 않았다. 가져온 코드와 파일별 SHA256은
`IMPORT_MANIFEST.json`에 있다. 원본의 현재 코드 스냅샷이며 학습 당시 git revision까지
확정한 것은 아니다. 실행 시에는 각 checkpoint의 저장 `relation_config.yaml`을 사용한다.

## 구성

- `tokenhsi/`: reward 구현, PPO/AMP, 환경·학습 설정, test/VNC/eval 실행기.
- `output/stage2_900_v4_reward/`: 두 학습 run의 모든 checkpoint, 설정, seed, TensorBoard, 진단 CSV.
- `pth/`: Stage1 PUSH26k 초기 가중치의 실제 복사본.
- `markdowns/reference/stage2_v4_reward.md`: 보상 수식·teacher 출처 설명.
- `review.py`: checkpoint 목록, CPU 무결성·저장 설정 확인, 로컬 viewer.

공동 PUSH 모션도 실제 복사했다. 공통 SIT/CARRY/LOCO/CLIMB 모션 네 경로만 기존 baseline
데이터에 심링크로 연결되어 있다. 해당 데이터는 읽기 전용으로 사용하며 외부로 옮길 때는 별도 준비한다.
학습·viewer·평가는 이 복사본에서 실행한다. 가져온 `eval/`의 기존 고정 suite는 과거
checkpoint와 경로가 포함되므로 그대로 실행하지 않는다.

## 사용

```bash
cd /home/hwanhee/juan/CVPR2027/stage2_900_v4_reward
conda activate tokenhsi
python review.py list
python review.py inspect
python review.py view --gpu 6 --dry-run
python review.py view --gpu 6
# VNC 환경에서 보기
python review.py vnc --gpu 6
# 특정 checkpoint는 list에 나온 로컬 경로 지정
python review.py inspect --checkpoint output/stage2_900_v4_reward/Stage2_900V4Reward_12-01-51-03/nn/Stage2_900V4Reward_00006000.pth
tensorboard --logdir output/stage2_900_v4_reward --port 6006
```

기본 checkpoint는 가장 높은 번호인 두 번째 run의 16,500이며 성능으로 선택한 best가 아니다.
첫 run은 6,000까지, 두 번째 run은 6,500–16,500 checkpoint를 포함한다. 두 run의 seed와
resume 관계는 저장 설정·가중치로 확인해야 하며 독립 학습 seed 두 개로 간주하지 않는다.
CPU 확인 결과는 `output_etc/check/inspection/`, viewer 결과는 `output_etc/check/viewer/`에 쓴다.
다른 GPU가 필요하면 `--gpu`로 지정한다. 기본 viewer는 4명·1환경·8물체·10회다.

## 논문 정리 시 확인할 내용

이 모델은 teacher 보상 수식을 사용한 PPO/AMP이며 distillation 모델이 아니다.
핵심 변경은 Enable/solo AT 성공 Z20cm와 보상 Z1mm 분리, 선행 성공 전 대기 감점,
공동 PUSH 접촉 준비 전 이동 감점이다. 상세 수식은 위 reference 문서에 있다.

TensorBoard reward와 진단 CSV는 학습 진단이다. 논문 성공률·시간은 별도 고정 평가로
측정해야 한다. 저장된 학습 조건과 정식 평가 조건(512환경·seeds 1/6/7·900step)을
구분한다. 이 폴더 생성 작업에서 새 GPU 학습이나 정식 평가는 시작하지 않았다.
