# 공개 Stage1 정책의 공통 점 평가

공개 TokenHSI 40k 가중치·354D 관측·RMS·Transformer·deterministic action을 기존 `evaluation/finetuned` 장면과 점/영역 평가에 연결한다. 기존35D checkpoint는 장면/player 생성용이며 행동에는 사용하지 않는다. 원본 self/location 관측 함수는 baseline AST에서 그대로 로드하며, 원본 network 파라미터를 strict load한다. 원본 baseline은 수정하지 않는다.

Chair26/Table8 테스트 자산 × SIT/CLIMB, 512환경 × seed0/1/2, loco 시작·첫 episode 최대30초. root 목표3D거리≤20cm, 종료frame 포함 ever. 목표는 Chair 좌판+.10/.94, Table bbox상판+.10/metadata+.94. 원본 점 성공식과 같지만 교차 물체/행동 조합·시간·초기화·집계까지 원본 논문 benchmark를 재현한 것은 아니다. 공개정책의 원본 bbox와 facing을 사용하며 Chair35D tokenizer나 Table 상판 bbox 보정은 적용하지 않는다.

```bash
. tokenhsi/scripts/multi_agent/runtime_env.sh
export PYTHONPATH="$TOKENHSI_ROOT:$TOKENHSI_ROOT/tokenhsi" PYTORCH_JIT=0
python -m evaluation.public_point.run_eval --output output_etc/check/public_point_40k_NEW
```

각 조건의 `initial.npz`, `trace.npz`, `results.json`, 상위 `manifest.json`/`summary.json`을 보존한다. 기존 평가 backend에서 저장 trace를 독립 float64 재채점하고 종료frame 포함 여부를 검사한다. checkpoint 위치와 장면 생성용 checkpoint는 `run_eval.py`에 명시돼 있다.

## 원본 CLIMB test 3종 혼합의 공통 평가

`python -m evaluation.public_point.run_eval --scene30d --categories Mixed --actions CLIMB --output output_etc/check/<새 경로>`는 공개40k Stage1의원래354D 관측/Transformer/RMS로CLIMB를실행한다. Box18/Cabinet9/Table8 test를각종류1/3로추출하고Ours30초/512환경×seed0/1/2/첫episode 점·영역을측정한다.

scene30d 옵션은장면/player생성에원래30D MA구조를쓰며그모델의행동은사용하지않는다. 행동은 PublicPolicy만계산한다. 공개모델의추가입력/파라미터확장은없고원본모든key/shape를strict load한다. 공개정책구성중의RNG를fork/복구하여같은30D 장면의기존78k 평가와초기root/DOF/물체/modelID를직접대조할수있다. 원본논문의10초/3repeat 평가와는다른공통평가결과다.
