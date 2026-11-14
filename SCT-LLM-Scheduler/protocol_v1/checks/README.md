# 공통 계산 예제

`cvpr2027-evaluation-v1.0.0`의 경계·예외·연산을 시뮬레이터 없이 확인한다. NumPy·PyTorch가 필요하다. 파일 위치를 옮겨도 실행할 수 있다.

```bash
CUDA_VISIBLE_DEVICES='' python checks/verify.py
```

공용 폴더 밖에서는 `verify.py`의 절대 경로를 지정한다. GPU·Isaac Gym·checkpoint를 사용하지 않는다. 검사 자체의 표준 출력만 생성한다.

| 파일 | 역할 |
| --- | --- |
| `cases.json` | 공간 성공·Error·Enable 동시 성공·20cm 시작·counter·NaN/clip의 고정 입력/기대값 |
| `reference_metrics.py` | 레포 독립 계산 예제. 거리 m·Time step·t_coop 비율0–1·Error cm |
| `teamhoi_reference.py` | 기존 보존본에서 함수 본문을 추출한 원본 Jerk·쿼터니언 회전. 원본 연산 순서를 유지 |
| `source.json` | 추출 함수와 원본 파일의 SHA256 |
| `verify.py` | 고정 예제와 회전/기울기·실패 제외·그룹 macro·Jerk 마스크 검사 |

다른 레포의 계산을 연결하려면 아래 함수와 같은 입출력을 갖는 작은 wrapper 모듈을 만들고 지정한다. 함수 인자는 `reference_metrics.py`를 따른다.

```bash
CUDA_VISIBLE_DEVICES='' python checks/verify.py --implementation /path/to/repo/protocol_adapter.py
```

| 함수 | 의미·반환값 |
| --- | --- |
| `region_distances(point, target, support, size)` | 받침 pose xyz/xyzw·전체 크기 입력. Point/Region 거리(m)와 XY100%+Z20cm 성공 dict |
| `support_goal(support, size, offset)` | 회전 받침의 최고Z+offset 목표점 |
| `first_enabling(success, valid)` | [T,2] 선행/후행의 유효 동시 성공 최초1-based step; 실패−1 |
| `first_coupled(centers, goal, valid)` | 실제 공유 중심 [T,3]·참여자 유효 [T,k]의 최초 성공step |
| `start_counters(centers, initial, first_success)` | 주 시작20cm의 `(s,a,b)`; 미출발s=−1 |
| `cooperative_ratio(count, a, b)` | 전체 공동 frame 수/원본 counter 차이·clip; 비율0–1 또는NaN |
| `surface_distance(hands, box, size)` | 회전 실제 전체 상자 경계까지 절대 거리(m), float32 |
| `minimum_error(distances, valid, successful)` | 성공 trial의 전체 유효 시점 최소 거리×100(cm); 실패None |
| `interaction_rates(independent, enabling, coupled)` | 유형별 trial 성공 배열의 네 SR(%). 비어 있는 유형·Mixed는None |
| `world_rim(pose, size)` | 실제 pose [T,7]의 중간 높이 둘레64점 [T,64,3], float32 |
| `mean_absolute_jerk(points, a, b)` | 원본 mask·dt0.0333·float32·축별 절댓값 평균, m/s³ |

Jerk 함수는 trial 값0을 그대로 반환한다. 여러 trial 평균에서0 제외는 호출자의 집계 단계다. t_coop NaN 제외 역시 집계 단계이며 실패 여부로 제외하지 않는다.

**통과 범위는 계산 예제뿐이다.** 실제 task ID 변환·전원 양손 판정·전체900frame 기록·초기 장면·낙상 제어·정책 입력·distractor attention·sampler 분포·seed 집계는 [공통 수락 기준](../README.md#53-레포-구현-수락-기준)에 따라 별도 검증한다.
