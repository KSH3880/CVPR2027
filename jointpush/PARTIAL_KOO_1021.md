# koo_1021 Stage1BasePush 기반 Partial Joint Push Stage2

업로드된 `jointpush/checkpoints/koo_1021/Stage1BasePush_00007500.pth`를 **Stage1 base**로 사용한다. Stage1 optimizer/epoch를 resume하지 않고 Stage2 optimizer와 epoch를 새로 시작한다. actor encoder와 scene normalizer를 동결하고 actor/critic coordination을 identity 초기화한다. action head·critic·discriminator는 원본 Stage2 방식대로 학습한다. 기본 source의165 tensor를 strict copy하고 새 coordination104 tensor만 추가한다.

## 환경

- 사람4명·object token8개·물리 bank24개, 학습2048환경/600step.
- coupled 확률0.8, 그룹 크기1~4명. 그룹별 하나의 실제 큰 상자를 사람 수에 맞게 분할한 관측과 담당 양손 point를 사용한다. 나머지 ordinary box4개는 distractor다. bank의 나머지 actor는 멀리 주차한다.
- 그룹 인원 n에 따라 X=n×U(1.0,1.3)m, Y=1.8m, Z=U(1.2,1.5)m, 질량=n×U(20,30)kg. 상자 마찰은 Stage1 원본0.6이다.
- 목표는 상자 정면2~6m. agent는360도 독립 위치/방향, 상자 둘레 경계반경+1~5m, agent 사이 최소1.2m에서 출발한다. 담당 상자보다 다른 상자가 가까운 후보만 재추출한다. 그룹 배정은 유지한다.
- RSI는 loco만 사용하며, AMP는 원래 PUSH family3에 loco/PUSH50:50 reference와4family/1330D discriminator를 유지한다.
- 원래 partial 손 point/구역별 contact 및 AT state/progress/success 각0.2, power, CPA collision coefficient0.5/distance0.5m를 유지한다.
- 정책 관측1272D,16token, Stage1과 같은 5개 task embedding·TaskActionFusion·GTA 및 새 Stage2 coordination4층.

## 실행

```bash
# 새 Partial Stage2 학습. 새 workspace를 만들고 Stage1 base만 전이한다.
TOKENHSI_GPU=5 bash jointpush/scripts/koo_partial_train.sh \
  jointpush/checkpoints/koo_1021/Stage1BasePush_00007500.pth

# 저장된 Stage2 checkpoint를 해당 학습 workspace의 설정으로 평가
TOKENHSI_GPU=5 HEADLESS=1 \
KOO_PARTIAL_WORK=/absolute/path/to/training/workspace \
bash jointpush/scripts/koo_partial_test.sh /absolute/path/to/Stage2.pth

# 로컬 viewer: HEADLESS=0으로 같은 test.sh 사용
TOKENHSI_GPU=5 HEADLESS=0 \
KOO_PARTIAL_WORK=/absolute/path/to/training/workspace \
bash jointpush/scripts/koo_partial_test.sh /absolute/path/to/Stage2.pth

# 서버 noVNC viewer
TOKENHSI_GPU=5 PORT=6086 \
KOO_PARTIAL_WORK=/absolute/path/to/training/workspace \
bash jointpush/scripts/koo_partial_vnc.sh /absolute/path/to/Stage2.pth
```

최신 실행의 tmux·로그·workspace는 `runs/jointpush/koo_partial_stage2_20261022/latest_training.json` 참조. 모든 설정과 결과는 runs 아래에 저장한다. 기본500iter마다 numbered checkpoint,2epoch 후 encoder/RMS 불변·coordination 변화 검사와 startup checkpoint를 저장한다. MPS를 read-only 탐색하며2048환경 본학습은 여유50GiB 이상/기존 Python client 최대1개 조건에서만 시작한다. 기존 학습/MPS daemon은 변경하지 않는다.

`koo_partial_run.py`의 process-local adapter만 사용하며 원본 source와 KSH partial adapter는 수정하지 않는다. Stage1 checkpoint는 업로드 후 전용 CPU 계약 검사를 통과했다. Stage2 저장본은 partial 설정/graph/RSI 별도 계약으로 구분한다. GUI viewer의 신규 경로는 아직 실제 렌더링 미검증이다.

검증: CPU128환경24step finite/Stage1 strict transfer165개·새104개, GPU2048환경 학습 gradient 및2epoch encoder/RMS 불변·coordination 업데이트/startup 저장 확인.
