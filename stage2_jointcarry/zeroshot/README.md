원본 크기 TokenHSI/TeamHOI 물체의 **GPU6 물리·정책 실행**은 [NATIVE_ASSETS.md](NATIVE_ASSETS.md)를 따른다. 아래는 이전 박스크기 정규화·CPU물리 probe 기록이다.

# 공동 운반 zero-shot 형상 probe

`ksh/stage2_jointcarry`를 읽어서 만든 독립 복사본. 원본은 수정하지 않는다.
출처·checkpoint SHA256은 `../ZEROSHOT_SOURCE.json`에 기록한다. 복사 시 최신 저장본은 epoch 21500이다.

현재 후보는 test split 원형 테이블이다. **원래 가구 크기 실험이 아니다.**
원래 mesh를 박스 크기 0.52×0.8×0.4 m에 축별 정규화하고 질량을 16.64 kg으로 맞췄다.
payload O0만 교체하며 나머지 물체·받침은 박스다. bbox 8점 관측과 박스 기준 파지 앵커/보상을 유지한다.
따라서 기존 성공률은 가구의 실제 파지·지지 판정이 아니라 박스 대리 판정이다.
정규화·형상 변경·충돌 decomposition이 실제 물체 일반화와 구별되어야 한다.

## 실행

이 디렉터리의 `eval.sh`는 학습을 하지 않는다. 물리는 항상 CPU PhysX/CPU pipeline이다.
물체 교체 시 GPU PhysX는 코드에서 거부한다. GPU policy는 물리와 장치가 다를 수 있도록
복사본의 player 장치 설정 및 평가 집계 텐서 장치를 수정했다.

```bash
# CPU 사전 확인
EPISODE_LENGTH=32 EVAL_REPEATS=1 bash stage2_jointcarry/zeroshot/eval.sh box
EPISODE_LENGTH=32 EVAL_REPEATS=1 bash stage2_jointcarry/zeroshot/eval.sh round_table
# GPU 6 정책 추론 + CPU 물리, 기존 per-GPU MPS 검증/연결
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ZERO_SHOT_POLICY_DEVICE=cuda:0 NUM_ENVS=16 bash stage2_jointcarry/zeroshot/eval.sh box
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ZERO_SHOT_POLICY_DEVICE=cuda:0 NUM_ENVS=16 bash stage2_jointcarry/zeroshot/eval.sh round_table
```

기본 600프레임·3회·동일 시드·joint_carry_at·loco 시작. 결과는 `CVPR2027/runs/jointcarry_zeroshot/` 아래 생성된다.
이 평가기는 `edge_context_eval.py`이며 single-agent의 ObjectSet 집계기가 아니다.
CPU 물리와 GPU 물리의 수치는 서로 직접 비교하지 않는다.

`prepare.py`는 원본 OBJ를 읽어서 로컬 OBJ/URDF로 복사·정규화한다.
`preflight.py`는 CUDA 없이 bounds와 mesh 인덱스를 검사한다.
`check_asset_cpu.py`는 GPU를 숨기고 테이블을 120스텝 낙하시킨다.
현재 1 body/0 DOF/3 convex shapes, 16.64 kg, 바닥 정착 z=0.20482 m를 확인했다.
Isaac Gym VHACD 캐시는 기본 `~/.isaacgym/vhacd`를 사용하며 torch 확장은 복사본 `.torch_extensions`로 격리한다.
기존 학습/MPS daemon을 종료·재설정하는 동작은 없다.

최종 비교 결과와 실패 실행 구분은 [RESULTS.md](RESULTS.md)를 참조한다.

## VNC viewer · 6086

`PORT=6086 bash stage2_jointcarry/zeroshot/view.sh round_table`

기본은 두 사람·테이블 mesh·carryWith 시작·1환경·10000회 반복이다. CPU PhysX와 GPU6 정책/렌더링을 사용한다.
접속: `http://localhost:6086/vnc.html?autoconnect=1&resize=remote` (SSH/VS Code에서6086포트 전달).
화면/launcher 로그: `CVPR2027/runs/jointcarry_zeroshot/view_6086/`.
