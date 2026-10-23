# PUSH 시작 collider 관통 수정

## 확인한 문제

기존 일반시작은몸체중심이상자+4cm영역에들어가는지만검사했다. 발목중심은밖에있어도발끝메시는상자내부에들어갈수있었다. GPU5 설정의초기학습재현96control step에서일반시작45°이상전도47건, GPU6 Y1.8설정에서는15건이었다. 저장한8개대표사례는두설정모두발메시정점의초기상자관통이있었다. GPU5 대표깊이는5.2~11.2cm, GPU6는2.4~10.3cm였다. 이는실행중정책checkpoint의직접재생이아니라같은저장환경/seed의별도초기학습재현이다.

GPU5에서모션/관절/1~2m범위를유지하고발끝겹침만뒤로이동한대조에서는47→0건이었다. 실제새필터적용첫검사에서는47→1건, GPU6는15→0건이었다. 남은GPU5사례의발메시가초기바닥아래2.4cm에있어바닥관통보정도추가했다. 바닥포함최종결과는changelog/해당audit JSON을따른다.

## 수정 계약

| GPU | 새 실험 | 상자 | 일반 시작 root–상자중심 |
| --- | --- | --- | --- |
| 5 | stage1_push_midpoint_safe | X=Y1.20~1.50m·Z1.30m·20kg | 1~2m 유지 |
| 6 | stage1_push_depth180_safe | X만1.20~1.50m·Y1.80m/Z1.30m·20kg | 1~5m 유지 |

- 두설정모두상자→목표는+Y1~4m,손목표는−Y면중앙0.65m다. 2명·4상자·2048환경·634D/8토큰·중점CONTACT/state>0.9·원본AT/paired포화·AMP는유지한다.
- `reset_geometry_filter: collider_bounds_v1`은실제MJCF sphere/capsule반경/offset과foot mesh전체bounding corners를상자로컬좌표로변환한보수적AABB를사용한다. 실제교차보다안전한자세도거부할수있으나발끝범위를누락하지않는다.
- 상자에2cm여유를두고일반시작의겹치는거리만최대64회재추출한다. 범위를자동확장하지않으며실패하면명시적오류다. 관절/모션frame은바꾸지않는다.
- 초기모든collider의바닥아래부분은사람전체를Z로최소이동시켜2mm바닥여유를확보한다. 손/팔을별도retarget하지않고상대자세와관절위치/속도는보존한다.
- native RSI도전체사람의Z/뒤쪽배치만정리한다. 원본21모션·MotionLib가중·50%PUSH/50%loco·70%후반30%추출·원본DOF는유지한다. cache는새reset계약/asset+mesh SHA와함께분리한다.
- 기존학습에는hot patch를하지않으며기존config/output/프로세스는유지한다. 새wrapper는전용output에새학습한다. reset분포계약이다르므로기존checkpoint를직접resume/test하지않는다.
- 초기관통수정은정책이이후상자를강하게쳐서넘기는모든경우를막는장치가아니다.

## 최종 검증

CPU41개·각7크기56042 RSI조합통과. 각2048환경3epoch/frame393216·96control step 초기학습재현에서45°초과는GPU5 47→0건/GPU6 15→0건이었다. 최종초기화44748건에서collider–상자겹침0·바닥여유통과,시작범위유지,각669tensorfinite·전용test12step×2재로딩통과. 장기전도율평가가아니다. [상세보고서](../../output_etc/check/push_safe_start_report/REPORT.md).

## 실행

사본루트에서실행한다. 캐시는최초준비/물리계약변경시에만생성하고, 준비가끝났다면train만실행한다. 학습/로컬/VNC전체목록은[config](../config.md)에있다.

```bash
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/build_stage1_push_midpoint_safe_cache.sh
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/build_stage1_push_depth180_safe_cache.sh

# GPU5: 기존1~2m 유지
TOKENHSI_GPU=5 bash tokenhsi/scripts/multi_agent/stage1_push_midpoint_safe_train.sh 2 2048 4

# GPU6: Y1.8m·일반시작1~5m
TOKENHSI_GPU=6 bash tokenhsi/scripts/multi_agent/stage1_push_depth180_safe_train.sh 2 2048 4
```

전도사건수집은 `tokenhsi/tests/audit_push_topple.py`에run.py의학습인자를전달하고확인출력/2048환경/짧은반복을사용한다. `PUSH_AUDIT_CLEAR_FEET=1`은기존GPU5 1~2m설정만을위한진단용대조옵션이며제품학습설정이아니다. 발메시정점분석은 `analyze_push_topple_geometry.py <topple_audit.json>`을사용한다.

검증파일은사본 `output_etc/check/push_gpu{5,6}_safe_floor_replay/`,캐시는 `stage1_push_{midpoint,depth180}_safe_build/`에둔다. 장기성공률/전도율과GUI는미검증이다.


## CONTACT 힘 조건 복구

2026-10-20부터 `stage1_push_midpoint_safe`, `stage1_push_depth180_safe`, `stage1_push_midpoint_safe_at5deg` 기본 YAML의 `require_contact_force`는 `true`이며 `min_hand_force_n`/`min_box_force_n`은 각각 0.5N이다. 동일 물리 substep에서 양손 중점 점수>0.9, 양손 각각의 면 법선 반력≥0.5N, 상자 면 법선 반력≥0.5N을 모두 만족해야 CONTACT 자체 성공이다. 제어 step 안의 성공을 누적하며 reset 때 이전 접촉은 제거한다. 센서는 각 손 body와 상자의 순 접촉력이므로 손-상자 쌍만의 힘을 분리한 센서는 아니다. 별도 개별손 면거리 gate는 추가하지 않는다.

연속 중점 state 보상·가중치·AT 성공과 paired CONTACT 보상 포화·RSI/캐시는 유지한다. AT 성공으로 CONTACT 보상이 포화돼도 CONTACT 자체 성공 로그는 별도다. 기존 train/test/VNC 명령은 그대로 사용한다. 실행 중 프로세스는 기존 설정을 유지하며 재시작하지 않았다. 변경 전 checkpoint는 새 기본 설정으로 직접 resume 불가이고, test/VNC가 checkpoint에 저장된 기존 `relation_config.yaml`을 읽으면 기존 힘 조건 없는 판정이다. 새 조건 학습은 새 scratch 실행부터 적용한다.
