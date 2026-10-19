# OMOMO lift 변환 데이터

데이터 루트: `stage1_fix/tokenhsi/data/dataset_omomo_lift/`.
원본: `stage1_fix/hhtools_omomo/` (보존).

## 파일 배치

```text
dataset_omomo_lift/
  motions/<원본 시퀀스명>/
    <object>_cleaned_simplified.obj  # 원본 local origin
    object_centered.obj             # local bbox 중심을 원점으로 이동
    full/phys_humanoid_v3/
      ref_motion.npy                # Poselib SkeletonMotion, 15 bodies, 30fps
      box_motion.npy                # (T,7), xyz + xyzw, centred bbox pose
      object_scale.npy              # (T,), mesh에 적용할 원본 scale
      conversion_report.json
    lift/phys_humanoid_v3/           # 지정 구간만 잘라 동일 구성
  conversion_summary.json
  validation.json
```

| 시퀀스 | split | 전체 프레임 | lift 프레임(0-based, 양끝 포함) | 잘린 프레임 수 |
|---|---|---:|---|---:|
| sub7_smallbox_014__train_full | train | 196 | 20–64 | 45 |
| sub16_largebox_026__test_full | test | 181 | 31–75 | 45 |
| sub8_plasticbox_017__train_full | train | 208 | 19–76 | 58 |

## 로딩 목록

`tokenhsi/data/dataset_omomo_lift_{full,lift}_{train,test,all}.yaml` 6개를 생성했다.
MotionLib skill key는 `lift`이며 `file`, `obj_file`, `weight` 계약을 따른다.
`train`에는 2개, `test`에는 1개, `all`에는 3개가 들어 있다.
`all`은 전체 확인용이며 원본 test 시퀀스를 포함한다. 기존 학습 목록은 변경하지 않았다.
`lift`라는 키를 추가했다고 환경의 새 skill/reward/AMP conditioning이 구현된 것은 아니다.

## 변환 방식과 검증 범위

- 공식 OMOMO 276-D 정의: 24개 global position, 24개 velocity, 22개 global 6D rotation. PyTorch3D row convention으로 회전을 복원한다.
- 기존 `dataset_carry/generate_motion.py`와 같은 SMPL→phys_humanoid_v3 관절 매핑·T-pose 보정을 쓰고, `project_joints_simple`로 팔꿈치를 hinge에 투영한다. 손 회전은 identity로 고정한다. 이 모델에는 손목 DOF가 없다.
- root는 packed global pelvis를 그대로 사용한다. 여기에 `trans2joint`를 다시 더하지 않는다. 원본 XY·heading과 수직 운동을 유지하며 발 mesh가 바닥을 뚫지 않도록 시퀀스 전체에 일정 Z offset을 적용한다. 상자에도 같은 offset을 적용한다.
- 상자는 원본 mesh local bbox 중심으로 pose를 옮긴다. `object_centered.obj`에 `object_scale.npy`와 `box_motion.npy`를 함께 적용하면 원본 world geometry를 보존한다. box_motion만으로 simulator 상자 치수가 자동 설정되지는 않는다.
- 실제 CPU MotionLib에서 6개 모두 load, 전 프레임·중간 시점 보간, 32 DOF, 129-D AMP 및 10-frame/1290-D history 유한값을 확인했다. quaternion 정규화, 고정 손, 원본 hash, full/lift slicing, 발 바닥 높이, 상자 world vertex 보존도 확인했다.

**형식 변환 검증은 통과했지만 물리·접촉 적합성은 미검증이다.** 원본과 체형이 달라 손 위치 평균 오차는 약 7–11cm다. 기존 carry 방식은 모든 관절 제한을 강제하지 않으며 실제 측정한 최대 초과는 전체본 약 0.197/0.299/0.327rad, lift본 약 0.194/0.126/0.220rad다. 접촉 IK나 제한을 적용한 추가 retargeting을 거치지 않은 후보 데이터다. 밑받침 자세 생성·상자 회전 고정·학습 보상·손목 feature 제외는 별도 작업이다.

## 재생성·검증

`stage1_fix/`에서 실행한다. 아래는 CPU 작업이며 학습을 시작하지 않는다.

```bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
python tokenhsi/scripts/multi_agent/convert_omomo_lift.py
python tokenhsi/scripts/multi_agent/validate_omomo_lift.py
```

이번 검증은 기존 활성 conda 환경 `tokenhsi_juan`에서 수행했다. 원본 데이터·생성된 배열·mesh·보고서는 Git 제외하고 변환기·검증기·YAML·이 문서는 추적한다.

공식 포맷 근거:
- https://github.com/jaggerShen/human-humanoid-tools/blob/main/hhtools/io/datasets/omomo.py
- https://github.com/lijiaman/omomo_release/blob/main/manip/data/hand_foot_dataset.py

## RSI 방식 사람·상자 뷰어

정책/checkpoint 없이 참조 root·32 DOF와 상자 pose/scale을 직접 재생한다. 최초 1회 준비 step 후 물리 적분을 하지 않으므로 정지 자세가 유지된다. 상자는 원본 mesh의 local bbox 크기를 적용한 cuboid다. 실제 접촉 안정성 테스트는 아니다. 세 후보를 왼쪽부터 smallbox / largebox / plasticbox로 나란히 표시한다. 기본값은 lift 구간·정지 시작이다.

```bash
# stage1_fix/에서. GPU 6만 허용. 기존 per-GPU MPS에 읽기 전용 검증 후 자동 연결.
TOKENHSI_GPU=6 PORT=6080 bash tokenhsi/scripts/multi_agent/omomo_lift_vnc.sh
# 이미 DISPLAY가 있는 로컬 창
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/omomo_lift_view.sh
# 전체 구간 또는 바로 재생
TOKENHSI_GPU=6 PORT=6081 bash tokenhsi/scripts/multi_agent/omomo_lift_vnc.sh --variant full --play
```

브라우저: `http://localhost:6080/vnc.html?autoconnect=1&resize=remote` (원격이면 6080 포트 포워딩).
Space 재생/정지, R 임의 phase에서 정지, ←/→ 기준 클립 한 프레임만큼 phase 이동, N 전체/개별 카메라 순환, Home 처음, S 스크린샷. 세 클립은 normalized phase를 공유하므로 길이가 다른 클립은 상대 시간에 맞춰 움직인다. 선택 카메라 클립의 길이를 재생 속도 기준으로 쓴다.

GPU 6에서 시작/80% phase 각각 3-frame GUI 점검과 스크린샷 확인을 마쳤다. MPS 시작·중단·재시작·설정 변경은 하지 않는다. 뷰어가 쓰는 runtime 자동 연결 경로는 기존 GPU 6 데몬의 환경과 `get_server_list`만 검증한다.
