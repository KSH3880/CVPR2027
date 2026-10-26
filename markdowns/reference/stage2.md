# Stage2: fixed 기반 공동 운반

기준은 `stage1_v2_fixed`다. 모델·성공 판정·placement 시 HOLDING 보상 포화·거리·고정 선반·단독 크기/과제는 현재 stage1 구현을 사용한다. RSI는 ON_TOP에서 putDown을 제외하는 차이가 있다.

## 설정

- 4명·1024환경·장면/관측 상자8개. 물리 환경 간격은16m이며 미할당 상자의 arena 반경은 stage1과 같은4.5m를 유지한다. 순차 pairing p=0.8, 공동 AT/ON_TOP50:50. 단독은 실제 크기에 유효한 SIT/CLIMB/AT/ON_TOP 과제를 고른다.
- 공동 payload X0.2~0.6m, Y0.4~1.2m, Z0.2~0.6m, 0.05m 간격·독립 sampling. X는 한 사람이 양손으로 잡는 폭, Y는 두 사람 사이의 길이 축이다. 밀도100kg/m³ 유지.
- 미할당 상자와 ON_TOP 받침은 stage1 SUPPORT: X/Y0.4~0.8m, Z0.25~0.5m. 각 상자는 독립적으로 뽑는다.
- 단독/공동 asset bank12개 중8개를 장면에 매핑한다. 나머지4개는 격리한다. 8슬롯 안의 미할당 상자는 관측과 물리에 남는다.
- 단독/공동 운반 RSI는 AT에서 loco/pickUp/carryWith/putDown=50/10/30/10, ON_TOP에서50/10/40/0이다. 공동 두 사람은 동일 frame과 반대 source 역할로 초기화한다. AT/ON_TOP의 공동 캐시는 공유하며 받침별 RSI를 준비하지 않는다.
- GraspView는 reset 때 반대편 앵커를 배정하고 episode 동안 유지한다. 사람별 공유 payload GTA 위치를 자기 앵커로 바꾸며 실제 크기와 rigid-point 속도를 사용한다. 앵커는 로컬±Y 끝에서15cm 안쪽이다. RSI 배치·GTA·HOLDING 리워드는 `utils/grasp_geometry.py`의 같은 계산을 사용하며, 리워드도 배정된 자기 앵커를 사용한다.
- 협력 상대가 상자 근처에 있을 때 해당 pair의 CPA만 해제한다. 기본 CPA와 다른 사람 간 penalty는 stage1 기준이다.
- 최종 배치 성공에는 현재 파지 gate를 추가하지 않는다. ON_TOP/SIT/CLIMB의 안쪽 영역·높이 조건과 AT20cm/1mm 조건은 fixed 그대로다.

## 모델과 전이

현재 stage1 encoder/task residual MLP의 모든 tensor를 strict 복사한다. Actor encoder와 관측 RMS는 동결하고, actor/critic 뒤에 identity 초기화한 human SA4를 추가한다. Critic·SA·행동/value head·discriminator를 학습하는 경로다. 공유 payload→target의 중복 task bias는 한 번의 bias와 같게 평균한다.

기본 전이 파일은 `output/stage1/Stage1V2Fixed_00060000.pth`다. 더 학습한 파일은 **새 stage2 시작 전** `STAGE1_CHECKPOINT`로 지정한다. 기존 stage2 재개에는 `RESUME_CHECKPOINT`를 쓴다. 전이와 재개를 혼용하지 않는다. 기본 명령은 [config](../config.md), 저장된 환경을 명시적으로 바꾸는 옵션은 `STAGE2_ENV_CONFIG`다.

## 공동 RSI 데이터

`joint_carry/source_rsi/{pickUp,carryWith,putDown}`는 기존 `stage2_coupled`의 최종 공동 자료를 보존한 비교 기준이다. Pickup100프레임, carryWith129프레임, putDown135프레임이며 경로와 SHA256은 `joint_carry/source_manifest.json`에 있다. 외부 레포를 수정하지 않는다.

기본 생성은 `single_source.py`가 읽는 1인 B19/pickUp·B20/carryWith·B21/putDown에서 시작한다. 입력 파일은 기존 데이터 심링크를 읽기 전용으로 사용하며 SHA256은 `single_source_manifest.json`에 있다. 원본별 다리 관절 제한·발 접지 보정은 한 번 캐시한다. 각 크기에서 사람1을 Y 끝면 기준15cm 앵커에 맞춰 배치하고, 상자 중심 기준 Z180도 회전한 사람2를 생성한다. 지면 구간의 상자 바닥 높이를 보존하고 운반 구간은 원본 중심 높이를 유지하도록 전환한 뒤, 실제32-DOF 팔 IK로 파지점과 팔 침투를 맞춘다. 사람2에는 같은 로컬 관절 자세를 복사하며 속도는 보정된 전체 프레임에서 다시 계산한다.

`correction.py`와 `leg_correction.py`는 기존 제작기 기반이다. 현재 팔 IK는 원본 대비±0.85rad 제한 없이 실제 관절 범위를 사용한다. 기본 1인 생성 경로에는 이전 크기 확장 실험의 추가 몸통 yaw·폭에 따른 운반 높이 조정을 사용하지 않는다. `--source stage2`는 기존 공동 보정본에 같은 inset 변경을 적용하는 비교용 경로이며 기본 학습 데이터와 디렉터리를 분리한다.

`joint_carry/build_size_rsi.py`는 요청한 payload 크기마다 세 모션의 모든 후보 frame을 검사한다. 실제 asset geometry 침투와 GPU PhysX0.1초의 충격·finite 검사를 수행한다. CarryWith는 두 사람이 동시에 각각 최소 한 손의 접촉 조건을6스텝 중3스텝 이상 만족해야 한다. GPU 접촉은 손의 net force와 상자 표면 거리를 조합한 근사이며, 예전 CPU 검사의 정확한 contact pair 조회와는 다르다. 각 asset profile은 분리된 worker의 한 sim에서 검사하고 프로세스 종료로 PhysX 자원을 회수한다. 실패한 worker는 같은 조건으로 한 번 재시도하며 빈 결과는 재사용하지 않는다. 이는 reset 검사이며 장기 자세 유지나 학습 성능 검증은 아니다. 준비기의 과거 support 옵션은 실행 경로에서 사용하지 않는다.

- 기본 저장 위치: `joint_carry/stage2_size_rsi/inset15_single/`. 크기별 NPZ와 검사/해시 JSON.
- 입력 모션·보정/기하/물리 코드·fixed sim 설정·asset hash가 바뀌면 다시 생성한다.
- reset에서 필요한 profile이 없거나 생성 조건 hash가 다르면 `--source single --inset 0.15`로 준비기를 호출한다. 유효한15cm 캐시가 있으면 즉시 읽는다. 유효 frame이 없으면 해당 크기/skill을 명시하고 중단한다. 크기 축소나 loco 대체를 하지 않는다.
- 단독 RSI는 별도의`solo/` 캐시를 사용하고 기존 stage1 운영 캐시는 보존한다.
- 요청 JSON으로 생성 범위를 지정한다. `--source single --inset 0.15`이 기본이며 `--workers 4`는 GPU5 안에서 별도 물리 검사 프로세스4개를 실행한다. `audit_size_rsi.py`는 요청한 모든 크기의 hash·finite·관절 범위·상태 일치·접촉 조건을 검사한다. 실제 완료 범위는 최종 audit report를 확인한다.

현재15cm의 전체1377크기·4131개 NPZ [최종 audit](../../output_etc/check/stage2_anchor15/full1377_audit.json), 실제 GPU 로더와 캐시 누락 시 자동 생성·재로드 검사가 통과했다. 최종 cache 상태는 생성 report와 별도로 전체 audit를 기준으로 확인한다. [검증 요약](../../output_etc/check/stage2_anchor15/validation.json)에 실행 결과와 한계를 기록했다.

전체 1,377크기의 생성·검사 명령이다. 기존의 유효한 파일은 재사용한다. 학습은 실행하지 않는다.

```bash
export TOKENHSI_GPU=5 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
. tokenhsi/scripts/multi_agent/runtime_env.sh
mkdir -p output_etc/check/stage2_anchor15
python - <<'PY'
import itertools
import json
from pathlib import Path
sizes = itertools.product(range(4, 13), range(8, 25), range(4, 13))
requests = [dict(size=[v / 20 for v in size]) for size in sizes]
Path('output_etc/check/stage2_anchor15/requests1377.json').write_text(json.dumps(requests) + '\n')
PY
python joint_carry/build_size_rsi.py \
  --requests output_etc/check/stage2_anchor15/requests1377.json \
  --source single --inset 0.15 --workers 4 \
  --directory joint_carry/stage2_size_rsi/inset15_single \
  --report output_etc/check/stage2_anchor15/full1377_report.json \
  > output_etc/check/stage2_anchor15/full1377.log 2>&1
python joint_carry/audit_size_rsi.py \
  --directory joint_carry/stage2_size_rsi/inset15_single \
  --requests output_etc/check/stage2_anchor15/requests1377.json \
  --report output_etc/check/stage2_anchor15/full1377_audit.json \
  > output_etc/check/stage2_anchor15/full1377_audit.log 2>&1
```

## RSI 영상 확인

`joint_carry/render_size_rsi.py`는 예전 stage2 source와 대표27크기의 새 후보를 native humanoid/상자로 재생한다. 화면의 초록 표시는 실제 reset pool에 포함된 프레임, 빨강 표시는 제외된 프레임이다. 영상에는 모션 연속성을 위해 모든 후보를 표시한다. 저장된 통과 프레임과 재생 자세의 일치를 확인하며, 정책 추론이나 물리 rollout은 아니다. `--source stage2 --inset 0.15 --directory <비교 캐시>`처럼 생성 조건을 명시하면 기존 공동 자료의 비교 영상도 재생성할 수 있다.

```bash
TOKENHSI_GPU=5 bash -c '. tokenhsi/scripts/multi_agent/gui_gpu_env.sh
. tokenhsi/scripts/multi_agent/runtime_env.sh
mkdir -p output_etc/check/stage2_anchor15/videos_single
xvfb-run -a -e output_etc/check/stage2_anchor15/videos_single/xvfb.log python joint_carry/render_size_rsi.py --output output_etc/check/stage2_anchor15/videos_single'
```

## 검증

`tokenhsi/tests/test_stage2.py`는 그룹 graph/physical mapping·stage1 설정 유지·tensor 전이·identity SA·동결/gradient·RMS를 확인한다. `tokenhsi/tests/check_stage2.py`는 GPU reset/부분 reset과 동결 모델 추론을 실행하며 ON_TOP putDown 제외와 carry 초기 자기 앵커 정렬도 검사한다. 실제 검사 결과와 미검증 항목은 [changelog](../../changelog.md), 최초 연결은 `output_etc/check/stage2_integration/`, 새 범위·초기화와 영상은 `output_etc/check/stage2_rsi_video/`에 기록한다. 현재15cm·1인 원본의 비교와 전체 생성 결과는 [검토 페이지](../../output_etc/check/stage2_anchor15/index.html)에 모은다.
