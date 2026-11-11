# Stage2 공유 받침 비교

실행 명령은 [config](config.md)에 있다. 두 실험은 이 checkout의 Stage2를 기반으로 하며 공동운반은 없다.

| 실험 | 공유 SIT/CLIMB/ON_TOP 성공의 XY 영역 |
| --- | --- |
| `stage2_shared_full_success` | 원본 물리 받침 전체의 내부 영역 |
| `stage2_shared_region_success` | 현재 선택한 구역의 내부 영역 |

두 YAML의 환경 차이는 `sharedSupport.success_region`뿐이다. 성공 영역의 기존 10% 여유 경계, 높이 허용오차, 보상 계수와 성공 시 state/progress 및 자기 HOLDING 포화는 유지한다. 단독 과제의 성공 기준은 두 실험에서 동일하다.

## 샘플링

학습은 4명·1024환경이다. 사람 순서를 섞은 뒤 남은 사람이 둘 이상이면 coupled 80% / NONE 20%를 추첨한다. coupled는 2명부터 남은 인원까지 균등 추첨하고, NONE은 한 명을 단독으로 소비한다. 마지막 한 명은 단독이다. 이론상 사람 기준 공유 82.27%, 환경 기준 공유 포함 99.2%다. BEFORE는 없다.

단독 SIT/CLIMB/AT/ON_TOP은 10/25/32.5/32.5%, 공유 SIT/CLIMB/ON_TOP은 각각 1/3이며 사람별로 독립 추첨한다. 공유 ON_TOP도 각자의 payload를 사용한다. 인원·과제·그룹은 reset 때 추첨하고 에피소드 동안 유지한다.

물리 asset은 기존 Stage2처럼 환경 생성 시 bank로 준비한다. 각 사람의 단독 운반/SIT/CLIMB/ON_TOP 받침 크기는 현재 Stage2 범위를 사용한다. 공유 받침은 인원과 독립적으로 X/Y 1.2~1.6m, Z 0.35~0.5m를 0.05m 간격으로 샘플링한다. 크기 자체를 매 reset 다시 만드는 방식은 아니다.

정책의 총 obj 토큰 수는 실행 인자로 지정하며 기본값은 12다. 과제용 개인 물체와 공유 구역을 먼저 빈자리 없이 모으고, 남는 자리는 미할당 일반 bank 상자 중 무작위로 선택한 실제 distractor로 채운다. distractor 하나는 한 토큰이다. 과제용 물체와 distractor 모두 관측·충돌에 포함하고, 사용하지 않는 교체용 bank actor만 장면 밖에 둔다. 일반 bank를 더 늘려야 할 때 추가 후보의 크기는 기존 `SUPPORT` 범위를 사용한다.

물리 상자는 먼저 환경 중심 반경 4.5m 원 안에 면적 균등 분포로 배치한다. 기존 중심 거리 검사(사람 1m, 상자 0.7m)를 거친 뒤 과제용 물체·사람 위치를 조정한다. 이후 과제용 물체나 먼저 확정한 distractor의 XY 외접원(+0.05m)과 겹치거나 사람 중심에서 1m 이내인 distractor만 같은 원 안에서 재배치한다. 최종 물리 크기 기반 겹침·관통·AT 목적지 방해 검사에는 distractor도 포함한다. 초기 배치의 원은 이동을 제한하는 벽이 아니며, 너무 많은 distractor로 배치가 불가능하면 재시도 한도에서 오류를 낸다.

## 관측·관계·보상

공유 그룹당 물리 받침은 하나다. 각 축을 `ceil(length / 0.8)`등분하여 전체 상판을 덮는 구역 토큰을 생성한다. 기본 범위에서는 항상 2×2구역이며 구역 한 변은 0.6~0.8m다. 원본 받침 토큰은 별도로 넣지 않는다. 분할의 로컬 좌표는 고정되고, 매 관측에서 원본 pose와 `v + omega × offset`으로 구역 pose·속도를 갱신한다. 미선택 구역도 관측에 포함한다. 모든 obj 자리를 실제 구역/물체로 채우므로 정책 입력에 padding mask가 없다. 토큰을 모은 뒤 과제의 참조 인덱스를 함께 갱신하며 원본 물리 ID는 별도로 유지한다.

구역 소유권이나 중복 없는 배정은 없다. SIT/CLIMB는 사람 root XY, ON_TOP은 자기 payload XY에 가장 가까운 구역 중심을 매 시점 선택한다. 이 구역을 task packet과 state/progress에 동일하게 연결한다. 모든 사람은 동일한 분할 장면을 본다. HOLDING은 자기 payload 중심에 대한 기존 보상이다.

성공 XY만 위 표처럼 다르고, 공유 과제의 상판 높이는 항상 실제 물리 받침의 pose·전체 크기에서 계산한다. 실제 접촉과 충돌 역시 물리 받침 하나에서 발생한다. 원본 물리 ID는 정규화하지 않는 별도 관측 metadata로 유지하며, task/구역 토큰이 달라도 같은 공유 받침이면 coupled다. 다른 사람은 NONE, 자신은 self다. 공동운반 CPA 면제는 사용하지 않는다. 동시 성공·대기·보상 공유를 추가하지 않는다.

## 초기화·전이

모든 단독/공유 과제는 loco reference에서 시작한다. 크기별 RSI screening/cache와 공동운반 reference는 사용하지 않는다. AMP는 Stage1의 과제별 expert 구성과 family 샘플링을 유지하고 Stage2의 backward/sideways 추가 소스는 제외한다.

기본 전이는 `stage1/Stage1V2Fixed_00030500.pth`이며 `STAGE1_CHECKPOINT`로 지정할 수 있다. actor encoder와 관측 RMS는 동결한다. action head, coordination, critic 및 AMP discriminator 등 나머지 학습 경로는 기존 Stage2를 따른다. 새로운 협력 파라미터는 항등 초기화한다.

공유 사람은 받침의 외곽 변과 그 변 위의 위치를 무작위로 뽑아 바깥쪽 법선 방향으로 배치한다. 외곽 거리 범위는 SIT 1~5m, CLIMB/ON_TOP 1~10m다. ON_TOP payload는 이동한 사람의 위치를 기준으로 기존 단독 loco 배치 규칙을 적용하며, payload/AT 목표의 고정 선반도 기존 경로를 사용한다. 물체·사람 겹침 시 같은 그룹/과제를 유지한 채 배치를 재추첨한다.

## 평가 옵션

전용 train 스크립트는 `[agents=4] [envs=1024] [objects=12]`, test/VNC는 `<checkpoint> [agents=4] [envs=1] [objects=12] [repeats=10]`을 받는다. objects는 총 정책 obj 토큰 수다. 4명·기본 크기 범위에서 최소 12이며, 예를 들어 16을 입력하면 과제용 토큰 외의 나머지를 distractor로 채워 항상 16개를 관측한다. 4명 공유 SIT는 4구역+8 distractor, 두 공유 ON_TOP pair는 8구역+4 payload로 각각 12토큰이다. 실제 장면 상자 수는 각각 9개와 6개다.

인원 M, 받침당 최대 구역 수 K에서 필요한 최소 토큰 수는 `2*M + floor(M/2)*max(K-2, 0)`이다. 입력이 부족하면 자동 축소나 과제 재추첨 없이 오류를 낸다. 물리 bank는 `max(4*M, objects) + floor(M/2)`개를 준비하며, 활성 물체 수와 다르다. 기본 12토큰에서는 물리 bank 18개다.

`TASK_GRAPH`는 `random_scenario`, `all_solo`, `all_shared`, `sit`, `climb`, `carry_at`, `carry_ontop`, `shared_sit`, `shared_climb`, `shared_ontop`을 지원한다. `shared_*`와 `all_shared`는 전원이 하나의 받침을 공유한다. 초기화는 평가도 loco만 지원한다.

평가는 인원 변경을 지원한다. 더 큰 받침은 저장된 환경 YAML을 복사하여 `sharedSupport.size_ranges`를 변경하고 `SHARED_SUPPORT_ENV_CONFIG`로 지정한다. 각 XY 최소값은 0.4m 이상이어야 하며 실행의 objects 인자도 위 최소값 이상으로 지정해야 한다. 추가 구역과 인원에 따른 메모리 사용량 및 학습 범위 밖 성공률은 별도로 검증해야 한다. 크기 범위 변경은 평가 checkpoint 검사에서 허용하지만 full/region 성공 방식이 다른 checkpoint는 거부한다. 학습 재개는 같은 graph/entity count를 요구한다. 관측에서 padding metadata를 제거했으므로 이전 공유 실험의 version 1 checkpoint는 직접 로드하지 않으며, 새 실험은 기존 Stage1 checkpoint에서 전이한다.

검증 결과는 [changelog](changelog.md)에 기록한다. GPU 검사는 `tokenhsi/tests/check_shared_support.py`를 사용하며 산출물은 `output_etc/check/` 아래에만 저장한다.
