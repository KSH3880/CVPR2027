# Changelog

최신 변경부터 기록한다. 현재 실행법은 [config.md](markdowns/config.md), 코드 위치는 [structure.md](markdowns/structure.md)를 참조한다. 실행 중인 GPU/PID는 이 파일에 고정하지 않고 실제 프로세스로 확인한다.

## 2026-10-15

### Stage1 v2 fixed: 고정 선반 검증 후 별도 실행 연결

- 동적 선반의 바닥 접촉 이동을 피하도록 새 `stage1_v2_fixed` env/train·train/test/VNC를 추가했다. v2에만 `platformPhysics: fixed_warmup_v1`을 선택해 고정 선반과 첫 전체 reset 전 물리1회 초기화를 사용한다. AT 중심 목표·크기·보상·AMP·기존 stage1/v2 YAML/스크립트는 유지한다. 새 output을 분리했으며 본학습은 실행하지 않았다.
- 후보 검사를 통과한 뒤 적용했다. CPU **97개**, GPU0·2048환경 후보 epoch3/frame393216 및 resume epoch4/frame524288·actor/critic 관련12개 파라미터 업데이트·finite·물리 reset 실패0. 후보의 선반 이동0, 첫 물리 추가시간 약21ms, 기존/후보 각3epoch 속도는 비슷한 범위였다. RSI6,413 profile의 새 screening mask가 기존과 전부 같았다.
- 최종 train wrapper도 2048환경 epoch2/frame262144·514 scalar finite·reset 실패0 통과. 실제 코드의 13과제/초기 skill·부분 reset·CPA416회·거리769건, 바닥/0.005/0.02/0.25/0.75m 선반 접촉을 확인했다. 밑면 오차≤0.0043mm. 별도 후보 검사에서 마커 좌표도 검증했다. 초기 검사 코드의 연속 reset 호출·float32 XY 오차 허용 문제는 격리 harness에서 수정하고 최초 실패 로그를 보관했다.
- 새 snapshot/checkpoint에 sim도 저장하고 test에서 저장 env+sim을 복원한다. sim이 없는 구 v2 snapshot은 새 실험의 native sim을 명시적으로 보충해 기본 rest_offset 때문에 2mm 틈이 생기는 문제를 막는다. 새 checkpoint·기존 v2 8k의 표준 headless32환경32step 평가 통과. VNC 호출은 기존 NVML 버전 불일치로 GPU 선택에서 종료돼 화면 미검증이며 서버 설정은 수정하지 않았다.
- [검증 결과](output_etc/v2_fixed_validation_20261015/validation.json), config/structure에 재현 명령과 범위·제한을 기록했다. 기존 학습 프로세스·checkpoint는 변경하지 않았다. 장기 수렴·성공률 개선을 입증한 검사는 아니다.

### Stage1·v2 viewer: checkpoint 저장 환경 자동 선택

- 실행 중인 구버전 stage1 checkpoint를 최신 YAML 거리 분포로 보여주던 혼동을 해소했다. stage1/v2 test와 이를 호출하는 VNC는 기본 env로 checkpoint run의 `relation_config.yaml`을 사용한다. 전용 환경변수의 명시적 override를 우선하며, 저장 config가 없으면 최신 YAML로 조용히 대체하지 않고 경로 지정 안내 후 종료한다. 평가 인원·환경 수·graph·초기 skill 옵션은 유지한다. 실행 중인 학습/viewer는 변경하지 않았다.
- 경로 선택 10건(일반·심링크·이동 checkpoint·명시적 override·누락), 셸 문법·diff 확인. GPU 0 headless로 stage1 17k/v2 7.5k를 각각 2환경·32step 평가해 종료 코드 0·finite·`legacy_near`/`tokenhsi` 분포를 확인했다. 결과는 `output_etc/stage1_viewer_config_check_20261015/validation.json`. NVML 버전 불일치는 지속하며 GUI/VNC 화면은 이번에 검증하지 않았다. config/structure 문서를 갱신했다.

## 2026-10-14

### Action 평가: 가변 인원·물체 수 지원

- 사용자 요청으로 task-role 시나리오의 추론 graph·과제 packet·크기 샘플링을 M명·O물체로 일반화했다. `random_scenario`는 M≥1·O≥M이며 각자 주물체 하나를 갖는다. ON_TOP이 O−M개 받침보다 많으면 초과 담당자를 무작위로 골라 크기 적격 다른 과제로 재추첨하고 받침은 중복 없이 연결한다. 기존 2명·4물체 경로·학습 계약은 유지한다.
- 가변 인원 평가의 출발 간격/장면 반경을 확장했다. 첫 9명 검사에서 밀집 장면이 64회 reset에 실패한 문제를 해소했다. 1명·1물체 검사에서 발견한 `topk(2)`도 물체 수에 맞췄다. 과제 통계 분모는 실제 인원 수를 사용한다. config/structure에 실행법·검증 위치를 기록했다.
- CPU **58개 통과**(1~32명·물체 부족 재추첨·크기 적격·binding·부분 reset·기존 Action 계약). 요청한 checkpoint의 GPU 0 headless 32step 평가를 사람/물체 **1/1, 3/3, 3/4, 4/16, 9/14, 12/12**에서 각각 2~3회 완료하고 trace의 담당자·받침 연결 및 평가 결과를 검사했다. 결과/로그는 `output_etc/generic_action_eval_20261014/validation.json`과 해당 폴더에 있다. 장기 성능·GUI/VNC 화면은 미검증이며 기존 학습 프로세스는 변경하지 않았다.

### Action viewer: 한 명 SIT·한 명 CLIMB 고정

- `TASK_GRAPH=sit_climb`을 추가했다. 첫 번째 사람→물체0 SIT, 두 번째 사람→물체1 CLIMB를 일반/크기 조건부 graph 및 reset에서 유지하고, 담당 상자 크기도 각 과제 범위로 생성한다. 기존 Action test/VNC wrapper로 사용하며 config에 명령을 추가했다.
- `test_size_rsi.py` CPU 27개 통과. 요청한 Action 기본 checkpoint로 GPU 0·2명·1환경·4물체·32step headless 평가 종료 코드 0, 실제 graph의 SIT/CLIMB binding 확인. 결과는 `output/action_sit_climb_check/metrics/`다. `nvidia-smi`는 NVML 버전 불일치로 실패했으나 CUDA 시뮬레이션은 완료됐다. GUI/VNC 화면은 미검증이다.

### stage1_v2: 새 크기·plane 성공·RSI 검사 적용 및 독립 실행 연결

- 새 env/train `stage1_v2.yaml`과 train/test/VNC wrapper를 만들었다. stage1의 공유 task bias·Action MLP·CPA·TokenHSI 거리/선반·운반 RSI 50/10/30/10은 유지하며, Climb Z 0.20~0.70m·XY 내부80%, 받침/배경 XY 0.40~0.80·Z 0.25~0.50m, AT 3D20cm+높이1mm를 적용했다. 기존 stage1/Action 설정·본학습은 유지하고 checkpoint와 output을 분리했다.
- 실제 asset 생성과 크기별 과제 적격 확률에 동일한 새 범위를 전달한다. Climb 두 상자를 적층으로 잘못 더하던 검사는 v2의 실제 payload→자기 받침 binding과 Carry 적격 크기로 제한했다. 기존 variant의 범위·성공 조건은 유지한다.
- `output/rsi_cache_stage1_v2`에 6,413개 profile을 새로 물리 screening했다. 초기/진행 body point 관통·발 바닥 침범·finite·0.1초 충격·낙상 검사를 추가하고 SIT/CLIMB phase는 실제 크기로 계산한다. 모션/root/AMP 이력을 수직 retarget하지 않는다. SIT 487·Climb 715개 profile은 전부 nonempty, SIT phase 공백 0. Climb 129개는 phase가 없어 전체 valid, carryWith 35/712개는 skill fallback을 사용한다. 이를 실행 분포·장기 안정성 보장과 구분해 문서화했다.
- CPU **70개 통과**, GPU 0·**2048환경** 확인 학습 epoch2/frame262144 및 resume epoch3/frame393216, 771 scalar finite·물리 reset 실패0·actor/critic embedding/projection/Action MLP 업데이트 확인. 실제 reset23회(부분21회)의 모션10,570개·AMP 이력95,130프레임·높은 Climb RSI521개, PPO minibatch48개/786,432행의 task/demo/replay family 정렬과 정규화 후 one-hot 유지 검사를 통과했다.
- 표준 headless wrapper 32환경·32step 완료. 별도 13과제/초기 skill·416step·310reset·772거리 검사, CPA·선반 배치/재배치 접촉 통과. 첫 선반 검사의 마지막 한 프레임 접촉력 assertion은 실패해, 착지 중 최대 접촉력과 정지 높이/속도를 함께 검사하도록 테스트를 보완했다. 최종 두 높이의 밑면 오차≤7.01e-6m. VNC 실제 호출은 기존 NVML 불일치로 GPU 선택 단계에서 종료돼 화면은 미검증이다.
- [학습·평가 비교](markdowns/tokenhsi_vs_stage1_v2_train_eval.md), config/structure를 갱신했다. 실행·분석 결과는 `output_etc/stage1_v2_check_20261014/validation.json`; 확인 학습은 `output/stage1_v2_check/`, resume는 `output/stage1_v2_resume_check/`다. 새 본학습/장기 성능 평가는 실행하지 않았다.

### Action 55k: Climb 실제 높이 0.40~0.70m 정책 평가

- 사용자 요청으로 frozen Action 55k를 0.40/0.50/0.55/0.60/0.65/0.70m 실제 상자 asset에서 평가했다. 높이마다 XY 0.40~0.80m의 81조합을 균등 배정하고 486환경·2명·4물체·CLIMB 두 과제·loco·1~2m·20초·3회를 실행했다. 높이당 486 agent 시도, 총 2,916시도다. 기존 체크포인트 검사를 유지하며 평가 프로세스에서만 asset/과제 적격 높이와 Climb 전용 개별 높이 검사를 사용했다. 일반 top-2 적층 검사의 운영 코드는 수정하지 않았다.
- 제안 내부 80%·기존 root/발 높이 조건의 ever/final은 높이순 **100/99.8, 99.4/99.4, 99.0/98.6, 98.1/96.7, 96.7/91.6, 63.0/53.9%**. 추가로 상자 기울기≤15°·실제 상판 높이가 요청 높이±5cm인 상태의 final은 **99.8/99.4/98.6/96.3/90.1/43.4%**, 종료 직전 2초 연속 성공은 **99.6/99.4/98.6/95.5/88.1/38.3%**다. 높은 상자가 넘어져 낮아진 경우를 추가 진단에서 제외했다. 원래 내부 90% 및 중심점 20cm 지표도 별도 저장했다.
- 세 회차 정상 완료·finite·매 step 운영 성공과 독립 기하 일치·trace 사후 재집계·XY 균등 배정·출발 거리·config/checkpoint hash 불변·재추첨 실패 0을 확인했다. `output_etc/action_climb_height_eval_20261014/`의 `evaluate.py`, `analyze.py`, `manifest.json`, `summary.json`, `validation.json`에 재현 코드와 결과가 있다. 이번 결과는 0.7m 동작 수행 가능성을 뒷받침하며, 1~10m·새 학습 수렴·높은 상자 RSI 안정성 검증은 아니다. 운영 크기/성공 조건·받침/배경 크기와 기존 학습은 변경하지 않았다.

### stage1: Climb 높이 확대·RSI 사전 감사 (설정 미적용)

- Climb Z 0.20~0.70m·XY 내부 80% 제안에 대해 기존 캐시 6,224개 profile을 분석하고, 2048환경의 격리 실행에서 Climb 891개 크기와 대표 payload 3종×받침 245종을 검사했다. 누락 Climb 433·ON_TOP 722개만 별도 캐시에 native 0.1초 screening으로 추가했다. 기존 캐시 SHA256·운영 YAML·본학습은 유지했다.
- 독립적인 0.7m Climb 상자 둘을 1.4m 적층으로 취급하는 전역 top-2 높이 검사가 현재 1.2m 제한에 걸린다. 물리 감사에서는 두 번째 담당 상자를 0.4m로 고정해 첫 번째의 전 범위를 확인했으며, 실제 확대 적용 전 역할별 적층 검사 수정이 필요하다.
- 모든 Climb 크기에 screening 통과 프레임이 있지만, 제안 성공 자세가 하나라도 있는 XY 크기는 Z=0.50/0.55/0.60/0.65/0.70m에서 각각 81/22/0/0/0개(높이별 81개)다. 참조 root 최대 1.32052m가 0.6m 상자 성공 최소 root 1.34m보다 낮다. 초기 성공 자세 부재는 정책의 학습 불가능을 뜻하지 않으며, 캐시 재생성만으로 높은 상판의 완성 자세를 만들지는 못한다. 단순 수직 이동·고정 PD의 추가 탐색도 안정적인 보정을 입증하지 못했다.
- 기존 carryWith 34/713·ON_TOP putDown 2/3,119 profile이 비어 대체 skill 경로를 탄다. SIT 508개 profile은 valid∩phase가 없어 late pool이 전체 valid로 대체된다. 따라서 명목 RSI 비율·후반 프레임 우대를 실제 실행 비율/후반 자세 보장으로 해석하면 안 된다.
- 9개 학습식 reset batch(총 6,144장면)는 finite·재추첨 실패 0·이번 표본 fallback 0. 다만 고정 PD 0.1초 후 SIT 29/1,266·ON_TOP 1/2,205장면에 낙상 종료가 있어 안정성 전부 통과로 판정하지 않는다. Climb 기하 분석 449,955건은 운영 성공 helper와 일치했다. `output_etc/stage1_rsi_height_audit_20261014/validation.json` 및 `geometry_summary.json`에 범위·제한·결과를 기록했다. 정책 학습/성능 평가가 아니며 config/structure에 적용 전 확인점을 연결했다.

### stage1: TokenHSI 거리 분포·운반 RSI 정렬

- 사용자 요청으로 stage1 기본 `samplingMode=tokenhsi`를 추가했다. loco 접근 반경은 SIT 1~5m, CLIMB/AT/ON_TOP 1~10m, AT 목표와 ON_TOP 받침 XY는 환경 원점 기준 ±4.5m·사람/payload와 최소 1m다. putDown 참조 목표와 ON_TOP 받침 높이, 선반·물체 크기·성공/보상·AMP는 유지한다. 다중 장면의 기존 간섭 재추첨은 유지하므로 원본 최종 분포와 완전 동일하다는 뜻은 아니다.
- AT/ON_TOP 운반 RSI를 50/10/30/10으로 변경했다. 새 run은 `Stage1TokenHSISampling_*`; 실행 중인 기존 학습은 유지했다. 구버전 1~2m·40/10/40/10 config를 계속 지원하고, 서로 다른 설정의 암묵적 resume는 검사한다. train/test/VNC에 `STAGE1_ENV_CONFIG`로 저장 config를 지정할 수 있으며 평가 JSON에 분포를 기록한다.
- CPU **67개 통과**, GPU 0·**2048환경** 확인 학습 epoch 2 저장·514 scalar finite·물리 reset 실패 0. 저장 정책의 13개 과제/RSI·416 step·341 reset·초기 거리 765건, 부분 reset·ON_TOP binding·CPA·선반 접촉 통과. 신/구 config의 표준 headless test 각 16환경·32 step도 종료 코드 0·분포 표시 확인. 문법·diff·원본 문서 링크 70개 검사 통과. 검증은 `output_etc/stage1_tokenhsi_sampling_check_20261014/validation.json`, 확인 학습은 `output/stage1_tokenhsi_sampling_check/`다. 장기 성능·VNC 화면 검증은 하지 않았다.
- 원본 Sit/Climb train/test 8개 종류별 asset JSON을 재집계하고 `tokenhsi_original_carry_sit_climb.md`에 Climb Box 크기 조합·가구별 범위·Sit 좌면 대표 높이를 추가했다. 현재 물체 크기와 성공 조건은 변경하지 않았다. config/structure도 갱신했다.

### Action 55k: 원본 TokenHSI 환경의 출발 거리 1~2m 평가

- 사용자 요청으로 원본 물체·선반·목표·3D 20cm ever 조건과 앞선 ON_TOP 확장을 유지하고, 사람–담당 물체의 초기 XY 반경만 U(1,2m)로 제한했다. `evaluate_action_tokenhsi.py`에 `--approach-max-distance`를 추가하고 원본 태스크의 세 중복 XY sampler를 공통 메서드로 추출했다. 기본 Sit 1~5m·Carry/Climb 1~10m와 학습 YAML은 유지하며 AT 도착 목표·ON_TOP 받침 거리는 제한하지 않는다.
- Action 55k·1인·64환경×3회·seed 0, 과제별 192시도: AT(Carry)/ON_TOP/SIT/CLIMB ever **52.1/48.4/50.5/92.7%**, 종료 시 **45.8/46.4/44.3/7.8%**. 이전 원본 거리 ever 19.8/17.7/49.5/54.2%와 비교한다. CLIMB 순간 도달과 종료 유지의 큰 차이가 있으며, Sit은 거리 축소만으로 거의 개선되지 않았다.
- CPU sampler 6개 경우에서 원본 출력·RNG bitwise 일치, 제한 시 방향·RNG 소비 유지와 1~2m 범위 확인. 네 평가 종료 코드 0, 768시도 실제 초기 거리·finite·매 step 원본 성공/precision 일치·정적 mesh 이동 검사, trace 독립 재집계 통과. 이전 실행과 checkpoint·asset 목록/크기·시간·물리 모드 일치를 검증했다. Python 문법·문서 링크·diff 확인. 결과·검증·소스 snapshot은 `output_etc/action_tokenhsi_near2m_20261014/`에 보관하고 config/structure에 연결했다.

### Action 55k: 기존 환경에서 시작 거리만 변경한 대조 평가

- 사용자 요청으로 기존 2명·4물체·loco·600 step·seed 42·64환경×3회 평가에서 `ownerLocoDistanceRange`만 1~2/3~5/5~10m로 변경했다. 전용 `output_etc/action_distance_eval_20261014/evaluate.py`에서 표준 checkpoint/config 검사 후 평가 reset 전에 메모리로 적용하며 학습 YAML·기존 실행 코드는 변경하지 않았다.
- 구간별 384 agent 시도 합산 종료 성공 **93.2/76.8/9.9%**, 담당 물체 XY 1m 이내 접근 **100/86.2/12.2%**, 낙상 종료 장면 **1.0/14.1/88.0%**. 5~10m의 본인 낙상 169명 중 168명은 접근 전에 넘어졌고 다른 163명은 접근 전에 동료 낙상으로 중단됐다. 평균 장면 길이는 19.87/17.50/4.40초다. 거리 증가만으로 접근 안정성이 저하되는 현상을 확인했으며 내부 보행 실패 원인은 확정하지 않았다.
- 3프로세스 종료 코드 0·각 3회 완료, 실제 초기 거리 범위·finite·성공/접근/낙상 독립 재집계 검증 통과. 기준 1~2m 회차 성공 수 **117/120/121**이 기존 표준 55k와 정확히 일치했다. 같은 checkpoint SHA256·거리 외 env 설정 동일 검사, Python 문법·문서 경로·diff 확인. 개별 장면은 reset RNG 차이로 완전히 paired하지 않는다. 실행/검증/결과는 해당 폴더의 `README.md`, `summary.json`, `analyze.py`와 trace이며 config에 재현 명령을 추가했다.

### Action 55k의 원본 TokenHSI 환경 간단 평가

- 평가 전용 `tokenhsi/evaluate_action_tokenhsi.py`에서 frozen Action actor/normalizer를 원본 1인 Carry/Sit/Climb 환경에 연결했다. 원본 test mesh·시작 분포·시간·3D 20cm ever 판정을 사용하며, ON_TOP만 Carry 도착 선반을 현재 크기 규칙의 동적 받침상자로 바꾼 확장이다. 학습 설정·가중치·기존 eval은 변경하지 않았다.
- GPU 0·seed 0·각 64환경×3회(192시도): Carry/Sit/Climb/ON_TOP ever **19.8/49.5/54.2/17.7%**, 종료 시 **17.7/44.3/9.9/15.6%**. 원본 환경으로 옮긴 일반화 평가이며 기존 장면에 성공식만 바꾼 AT 95.3%와 구분한다. Action에는 full bbox만 제공하며 별도 좌판 목표점은 추가하지 않는다.
- 원본 fixed 선반의 GPU 재배치 후 collider 위치 불일치를 낙하 검사로 확인했다. Carry/ON_TOP은 중력 비활성·고질량 선반과 reset 속도 초기화를 사용한다. Sit/Climb은 원본 fixed mesh를 유지하고 전체 장면의 공통 SE(2) 변환·모델 관측 역변환으로 원본 상대 배치를 보존한다. 초기 호환 구현의 선반 발사·가구 이동/관측 좌표 문제로 나온 실행은 모두 최종 집계에서 제외했다.
- 네 평가 종료 코드 0, strict weight load·finite·매 step 원본 누적 성공/precision 일치·trace 독립 재집계·출발 거리 범위 검증 통과. Action 관측 함수와 bridge 일치, 고정 장면 역변환 관측 오차 ≤2.21e-6·거리 ≤4.77e-7m. 출발 선반 낙하 검사 및 8cm probe 상자의 원본 의자/가구 접촉 각 64/64 통과, 평가 중 mesh drift <1.2e-7m. Python 문법·문서 경로·diff 확인.
- `output_etc/action_tokenhsi_20261014/`에 승인된 네 과제 trace·초기 상태·manifest·검증·최종 실행 소스와 제외된 디버깅 결과를 구분 보관했다. 공식 512환경과 다른 소표본, 로컬 Climb test 35개, 물리 호환 처리가 있으므로 논문 표의 완전 동일 재현으로 부르지 않는다. config/structure에 재현법을 연결했다.

### TokenHSI 원본 Carry·Sit·Climb 상세 문서

- `markdowns/tokenhsi_original_carry_sit_climb.md`에 원본 세 과제의 크기·초기화·목표/선반 수식·보상·성공/종료·평가 집계·원본 명령 인터페이스를 정리하고 structure에서 연결했다. Carry의 putdown 보상(높이 1mm·XY 10cm)과 평가 성공(3D 20cm 도달)을 구분한다.
- 로컬 Sit train/test 49/26개, Climb 38/35개를 확인하고 평가 asset 61개의 치수·목표 높이·JSON 링크를 수록했다. 논문 Climb test 26개와 로컬 35개, 공식 기본 학습 4096환경과 로컬 2048환경 등 차이를 명시했다. 문서 링크·asset 집계·수치·diff 검증만 수행하며 실행 설정/데이터 변경·학습/시뮬레이션 실행 없음.

### TokenHSI 원본 성공 조건·현재 eval 비교

- 공식 GitHub Stage 1 evaluator·논문 4.1·로컬 원본 및 현재 Action/stage1 실행 경로를 대조했다. 원본 SIT/CLIMB/Carry는 root/상자 중심의 3D 20cm 도달을 episode 내 누적하며 512환경×3회다. 현재는 relation runtime의 종료 edge 성공과 전체 agent/scene의 final·ever를 저장한다. YAML `eval.successThreshold: 0.2`는 현재 relation 판정에 사용되지 않는다.
- 현재 AT는 phi≥0.9(3D 약 10.265cm)와 높이 1mm, SIT/CLIMB/ON_TOP은 상판 내부 영역과 각각의 높이 조건이다. CLIMB 발 조건은 양발 평균 높이이며 접촉·각 발의 XY·지속시간 검사가 아니다. 다중 사람 중 누구든 낙상하면 장면이 종료되며 원본과 물체/시간/시작 분포도 다르다.
- 기존 Action 55k AT trace 107건을 재판정했다. 기존 조건 final/ever **84/86**, 3D 20cm 조건 final/ever **99/102**(92.5/95.3%)다. finite·alive mask·거리/phi·runtime 성공 일치·포함관계 검사 통과. `output_etc/action_at_diagnosis_20261014/compare_success_rules.py`와 `success_rule_comparison.json`에 재현 코드·입력 hash·결과를 보관한다. 성공 수식만 바꾼 사후 분석으로 원본 benchmark 재현이 아니며 학습/평가 설정·모델 변경 없음.

### Action 55k AT 실패 단계 진단

- 표준 평가와 같은 64환경·600 step·loco 시작·seed 42·3회에서 AT의 매 step XY/높이/phi/성공·HOLDING·상자 속도/기울기를 기록했다. 회차 AT 성공률 75.6/78.8/81.8%가 직전 평가와 일치했고, 총 107시도 중 종료 84·평가 중 86 성공이다. 실패 23건 중 21건은 한 번도 성공하지 못했고 2건은 성공 후 이탈했다.
- 실패 23건의 거리 조건만 실패/높이만 실패/둘 다 실패는 4/5/14건이다. XY 오차 중앙값 10.6cm·절대 높이 오차 중앙값 7.1cm, 마지막 4초 평균 XY<30cm·속도<10cm/s 19건으로 목표 근처 마지막 배치 정체가 주로 관찰됐다. 종료 속도<5cm/s 19건·기울기>30도 18건·HOLDING 조건 만족 14건이다. HOLDING은 손 거리 조건이며 실제 grasp 판정을 뜻하지 않는다.
- 저장 trace에 높이 허용치만 1mm→1cm로 완화해 재판정해도 성공 84건 그대로, 2cm는 85건이다. 실제 rollout을 바꾼 실험이 아니며 reward/구조의 인과 원인은 확정하지 않았다. 모든 활성 frame의 성공 수식과 runtime 일치·finite·최종/최초 성공 수 재계산 통과. `output_etc/action_at_diagnosis_20261014/`에 실행/분석 코드·3회 trace·summary를 보관한다. 기존 학습·성공 조건·보상 변경 없음.

### Action 45k·50k·55k 표준 test 각 3회 평가

- 요청한 Action 45,000/50,000/55,000을 기존 `_test.sh`로 각각 64환경·2명·4물체·600 step·loco 시작·random scenario·seed 42·3회, GPU 0에서 순차 평가했다. 저장본당 192장면/384 agent 시도이며 과제별 수치는 회차 성공률의 단순 평균이다. 같은 seed 설정이 후속 회차의 동일 장면을 보장하지는 않는다.
- 종료 전체 성공 평균±회차 표준편차는 **90.4±1.2 / 91.9±3.5 / 93.2±1.6%**, SIT/CLIMB/AT/ON_TOP은 45k **97.0/95.2/74.4/98.6%**, 50k **100/98.0/80.3/96.1%**, 55k **96.3/100/78.7/98.5%**다. 이번 표본에서 55k 전체 평균이 가장 높으며, AT는 50k 평균이 높지만 회차 범위 68.3~89.7%로 55k의 75.6~81.8%보다 넓다. 장기 우열·망각의 원인은 검증하지 않았다.
- 세 프로세스 종료 코드 0·각 3회/192장면·finite·성공 범위/최초-종료 관계·평균/표준편차 재계산 검증 통과. `output_etc/action_test_45k_50k_55k_repeat3_20261014/`에 실행 코드·manifest·원본 metrics·summary·검증을 보관한다. GPU의 NVML 조회는 버전 불일치로 실패했으나 CUDA 여유 메모리 약 54GB 확인 후 실행했다. 기존 Action/stage1 본학습은 유지했으며 설정 변경 없음.

## 2026-10-13

### stage1: Action + 공유 task bias + CPA + payload 선반

- 사용자 합의대로 전용 `stage1` env/train YAML·train/test/VNC·output을 연결했다. Action MLP와 과제/보상/AMP/RSI/634-D packet을 유지하며 타입 쌍 projection을 layer/head당 하나로 공유한다. NONE/SELF=0·actor/critic 독립, 기존 CPA 충돌항을 사용하고 전용 variant/fusion·선반 설정 검사로 checkpoint를 격리했다.
- `utils/stage1_platforms.py`에서 AT/ON_TOP loco 출발 payload 및 AT loco/pickUp/carryWith 도착지에 원본 TokenHSI 50%·U(0,1m)·상자 윗면 1.2m clamp·출발 5cm 여유 규칙을 적용한다. Carry RSI 시작·putDown AT 바닥 목표·ON_TOP 받침·SIT/CLIMB·배경은 유지하고 graph binding과 부분 reset을 따른다. 기존 GPU 재배치 가능한 선반 actor를 재사용한다.
- **CPU 168개 통과**. GPU 0·**2048환경** 확인 학습 epoch 2 및 resume epoch 3 저장 완료, 771 scalar finite·물리 reset 실패 0, 양쪽 branch의 projection/embedding/MLP optimizer step 96→144 및 실제 업데이트 확인. 네 과제별 test wrapper **32환경·64 step** headless 평가 완료.
- 실제 시뮬레이터 **13개 과제/RSI 조합·416 step·319 reset 호출**에서 선반 범위/부분 reset·CPA 보상 대조 통과. 출발/도착 선반을 0.25m·0.75m로 옮겨 물리 지지·양의 접촉력·높이 오차 <1.28e-5m를 확인했다. 초기 접촉 fixture는 앞선 rollout의 쓰러진 사람 간섭으로 실패해 사람을 분리하고 재검증했다. 자료는 `output_etc/stage1_check_20261013/validation.json`과 로그·metrics다.
- Python/셸 문법·문서 경로·diff 확인. 본학습은 시작하지 않았고 기존 Action 학습은 유지했다. **VNC는 실제 호출했으나 NVML/로드된 드라이버 버전 불일치로 GPU 선택 단계에서 종료**, 화면 검증·장기 성능은 미완료다. 시스템 드라이버는 수정하지 않았다.

### Action 46k·47k 표준 test 각 5회 평가

- 기존 Action `_test.sh`로 46,000/47,000을 각각 64환경·2명·4물체·600 step·loco 시작·random scenario·seed 42·5회 평가했다. 저장본당 320장면/640 agent 시도이며 AT 높이 허용치 1mm 등 기존 성공 조건을 유지했다. 과제별 비율은 회차별 단순 평균이고 후속 회차 장면은 정확히 일치시킨 비교가 아니다.
- SIT/CLIMB/AT/ON_TOP 종료 평균은 46k **90.2/98.0/76.3/98.4%**, 47k **87.1/88.1/76.8/98.6%**. 전체 종료는 **89.7/87.8%**, 평가 중 성공은 **91.4/92.7%**, scene 조기 종료는 **2.2/6.6%**다. AT는 두 저장본 모두 45k보다 높지만 앞서 44k의 82.1%보다 낮고, 47k는 CLIMB 종료 성공 감소가 보인다. 원인/망각은 검증하지 않았다.
- 2프로세스 종료 코드 0·각 5회/320장면·finite·성공 범위·평균/표준편차 재계산 검증 통과. `output_etc/action_test_46k_47k_repeat5_20261013/`에 실행/검증 코드·manifest·원본 지표·summary를 보관한다. Action 본학습은 유지했으며 모델/학습 설정 변경 없음.

### Action 43k·44k·45k 표준 test 각 5회 평가

- 요청한 Action 43,000/44,000/45,000 저장본을 기존 `_test.sh`로 각각 64환경·2명·4물체·600 step·loco 시작·random scenario·seed 42·5회 평가했다. 저장본당 320장면/640 agent 시도, 총 15회이며 GPU 0에서 순차 실행했다. 과제별 값은 회차 성공률의 단순 평균이고, 같은 seed 설정이 모든 후속 회차의 동일 장면을 보장하지는 않는다.
- SIT/CLIMB/AT/ON_TOP 종료 평균은 43k **76.5/98.0/66.6/97.7%**, 44k **92.8/97.7/82.1/97.6%**, 45k **96.6/96.5/69.1/96.7%**. 전체 종료 평균±회차 표준편차는 **86.3±2.1 / 91.9±2.1 / 88.0±3.4%**, 평가 중 성공 평균은 **89.1/93.4/89.4%**다. 이번 표본에서는 44k 전체/AT, 45k SIT가 가장 높다. 45k AT는 회차별 55.0~75.8%로 편차가 있으며 회귀/망각의 원인은 검증하지 않았다.
- 3프로세스 종료 코드 0·각 5회/320장면·finite·성공 범위/최초-종료 관계·평균/표준편차 재계산 검증 통과. `output_etc/action_test_43k_44k_45k_repeat5_20261013/`에 실행 코드·환경 manifest·반복 원본·summary·검증을 보관한다. 기존 Action 본학습은 유지했으며 모델/학습 설정 변경 없음.

### Action 최신 3개 checkpoint의 표준 test 3회 반복 평가

- 사용자 요청으로 기존 Action `_test.sh`를 수정 없이 사용해 41,500/42,000/42,500을 각각 64환경·2명·4물체·600 step·loco 시작·random scenario·seed 42·3회 평가했다. 저장본당 192장면/384 agent 시도, 총 9회이며 GPU 0에서 순차 실행했다. 과제별 수치는 회차 성공률의 단순 평균으로 표본 수 가중 합산이 아니며, seed 설정은 같아도 모든 회차 장면을 동일하게 고정한 비교는 아니다.
- 종료 개인 목표 성공 평균은 **84.1/86.5/90.6%**, 평가 중 성공은 **87.5/89.3/93.2%**다. SIT/CLIMB/AT/ON_TOP 평균은 41,500 **79.5/94.9/69.2/92.8%**, 42,000 **81.1/98.2/66.8/96.8%**, 42,500 **87.9/100/78.6/95.5%**. 최신 SIT 회차별 63.6/100/100%로 소표본 편차가 크며, 전체 종료는 89.1/91.4/91.4%다.
- 3프로세스 종료 코드 0·각 3회/64장면·finite·지표 범위/최초-종료 성공 관계 확인. `output_etc/action_test_repeats_20261013/`에 실행/환경 manifest·반복 원본·평균/범위/표준편차 summary·검증을 보관하고 `markdowns/config.md`에 반복 평가법을 추가했다. 기존 두 본학습은 유지했다.

### Action 기반 사람 간 CPA collision 실험 연결

- Git `595688e`의 `collision_reward.py`·`approach_stage2_rescue_shared9_cpa.yaml`을 찾아 참고 CPA 식을 복원했다. Action 전용 `_cpa` env/train YAML·train/test/VNC·output을 분리하고 상대 root XY 위치/속도의 접근도×최근접 거리 위험×시간 할인(0.99)을 collision 항에 연결했다. 계수 0.5·거리 0.7m, 정지/평행/이탈 위험 0, 기존 거리 페널티 중복 없음. 모델·타입별 projection·concat MLP·AMP·RSI·과제/성공 조건·셔플은 그대로다. Bias 통합은 사용자 검토 단계로 구현하지 않았다.
- 기본 static 경로를 유지하고 checkpoint에 collision 계약을 저장한다. 과거 저장 env의 static 계약을 읽으며 static/CPA 및 파라미터가 다른 checkpoint의 직접 resume/eval을 거부한다. 기존 Action→CPA weight 전이는 추가하지 않았다. 참고 Stage 2의 다른 모델/보상 변경은 가져오지 않았다.
- **CPU 37개 통과**: CPA 접근/이탈/정지/교차·대칭/순열/공통 속도 불변·static 동등성·실제 dispatch·보상 1회 합산·checkpoint 격리·Action/상자 속도 회귀. **GPU 0·2048환경·MAX_ITERATIONS=1** 확인 학습 epoch 2 저장 완료, scalar 257종/514값 finite·물리 reset 실패 0. 저장본 **16환경·32-step headless 평가** 완료, 실제 reward 31회가 독립 CPA 계산과 차이 0(비영 위험 표본 14개). `output_etc/action_cpa_check_20261013/validation.json` 및 로그, `output/approach_scenario_stage1_unified_size_rsi_task_embedding_action_cpa_check/`에 보관한다.
- 셸 문법·문서 링크·diff 검사 통과. 기존 Action/Gate 본학습을 유지했고 CPA 본학습·장기 성능·VNC 화면은 검증하지 않았다.

### Action·Role gate 최신 간단 평가·밤사이 추세

- 사용자 표현의 goal은 앞서 비교하던 실행 중인 Role gate로 해석하고 Action 42,000 / gate 23,500을 GPU 0에서 순차 평가했다. 이전과 동일한 64환경·2명·4물체·600 step·seed 42·loco 시작·random scenario 1회이며 초기 task/binding packet이 두 모델 및 전날 평가와 일치한다.
- 종료 성공 수(SIT/CLIMB/AT/ON_TOP, 표본 11/37/45/35)는 Action **9/36/29/34**, gate **6/37/0/0**. 전날 36,000/17,500 대비 Action SIT 4→9·전체 개인 목표 성공 77.3→84.4%, gate CLIMB 0→37·SIT 2→6·전체 1.6→33.6%다. SIT 평가 중 성공은 Action 10/11, gate 7/11이며 gate 운반 목표는 평가 중 성공도 0이다.
- 최근 200회 평균 학습 지표를 전날 저녁 저장한 200회 평균과 비교했다. Action SIT 성공 상태 54.2→82.3%·종료 개인 목표 80.6→84.8%; gate SIT 29.0→51.9%·CLIMB 21.2→80.8%·종료 개인 목표 23.5→44.7%. 조회 시 최신 로그는 Action 42,451 / gate 23,668이다. 학습 RSI 혼합 지표와 별도 loco 평가를 구분하며 단일 seed 소표본이다.
- 두 평가 종료 코드 0·finite·성공 수 범위·표본 합·packet 검증 통과. TensorBoard 마지막 64MiB scalar 872,024/835,716개에서 비유한 값 0. `output_etc/rough_eval_20261013/`에 실행/검증 코드·manifest·summary·학습 추세·로그를 보관한다. NVML 조회는 기존 버전 불일치로 실패했으나 CUDA 여유 메모리 확인 후 실행했으며 두 본학습은 유지했다.

## 2026-10-12

### Action 36,500 bias·embedding·직접 주입 MLP 분석

- 최신 Action 36,500과 16,000/23,000의 actor·critic task embedding 및 타입 쌍별 bias를 추출했다. 최신 actor의 H→O bias(4층×2head) 과제 간 cosine은 0.9977~0.9998, task embedding은 0.8263~0.9651이다. AT/ON_TOP embedding cosine은 16,000의 0.9533→0.9651, SIT/CLIMB는 0.8653→0.8263이다. NONE/SELF=0인 현재 모델에서 실제 적용 타입 쌍과 미사용 표 항목을 구분한다.
- 최신 actor의 loco 초기화 64장면·0/120/300/540 step 관측(512 agent 관측)에서 직접 주입 MLP 제거 시 정책 평균 출력 상대 RMS 변화는 전체 27.2%, SIT/CLIMB/AT/ON_TOP 39.2/34.9/20.8/20.8%다. Bias만 제거하면 전체 51.6%; bias를 유지하고 MLP의 AT↔ON_TOP task 입력만 바꾸는 해당 과제 출력 변화는 5.0/5.3%다. MLP 주입량/입력 RMS는 SIT에서 층별 312/88/82/35%이며 output projection·residual·normalization 전 값이다. 대체 행동 rollout이나 성공률 기여도 측정은 아니다.
- 실제 player 출력과 CPU 재구축 출력 최대 차이 9.62e-7·정규화 차이 0·probe 복원 뒤 출력 차이 0·finite JSON 확인. `output_etc/action_components_20261012/`에 capture/분석 코드·원시 관측·actor/critic 전체 bias·유사도·probe 결과를 보관한다. 본학습·모델 설정은 변경하지 않았다.

### Action 뷰어 체감과 SIT 평가 차이 확인

- Action 최신 번호 저장본 36,000을 저녁 평가와 같은 조건으로 재실행하고 SIT 11명의 매 step XY 거리·높이 오차·안쪽 영역 이탈·성공을 기록했다. 초기 task packet 및 SIT 최초/종료 성공 **4/11**이 재현됐으며 활성 frame 전체에서 geometry 조건과 runtime 성공이 일치했다.
- 종료 실패 7명 중 1명은 골반 높이 오차 +3.4cm로 통과하지만 안쪽 영역에서 3.46cm 벗어났다. 나머지 6명은 목표 상자 중심 XY 거리 76~94cm·목표 높이 대비 +26~45cm였고 해당 성공 영역에 진입하지 못했다. 영상 자세나 뷰어의 실제 장면은 확인하지 않았으므로 시각적 앉기/뷰어 차이의 원인은 단정하지 않는다.
- 실행 종료 코드 0·trace finite·frame별 성공 조건 대조 통과. 비대상 AT 종료 성공은 재실행 시 30→29로 달랐으며 전체 rollout의 완전 재현은 주장하지 않는다. 자료는 `output_etc/action_sit_diagnosis_20261012/`; 모델·학습 설정 변경 없음.

### Action·Role gate 저녁 재평가·학습 추세

- Action **36,000** / role gate **17,500**을 직전과 동일한 64환경·2명·4물체·600 step·seed 42·loco 시작·random scenario 1회로 GPU 0에서 순차 평가했다. 초기 task/binding packet은 직전 및 두 모델 간 일치한다.
- 종료 성공 수(SIT/CLIMB/AT/ON_TOP, 표본 11/37/45/35)는 Action **4/31/30/34**, gate **2/0/0/0**다. Action은 직전 34,500 대비 AT 21→30·전체 종료 성공 75.0→77.3%; CLIMB는 평가 중 37/37 성공하지만 종료 유지 37→31로 감소했다. Gate는 SIT 종료 1→2·평가 중 성공 1→7, HOLDING 상태 만족 32.5→53.8%이며 다른 과제 목표는 여전히 성공 0이다.
- 최근 200회 학습 평균 종료 개인 성공은 직전 checkpoint 부근 대비 Action **76.5→80.6%**, gate **21.2→23.5%**다. Gate SIT 성공 상태 비율은 **17.1→29.0%**로 증가했다. 학습은 RSI 혼합 지표이며 별도 loco 평가·단일 seed 소표본과 구분한다.
- 두 평가 종료 코드 0·finite·표본 합/성공 범위·초기 packet 일치 검증 통과. TensorBoard 파일별 마지막 64MiB의 scalar 872,024/835,752개에 비유한 값 0을 확인했다. `output_etc/rough_eval_20261012_evening/`에 실행 코드·manifest·summary·학습 추세·로그를 보관한다. NVML 조회는 버전 불일치로 실패했으며 CUDA 여유 메모리 확인 후 실행했다. 기존 두 본학습은 유지했다.

### 동일 에포크·8~30토큰 attention 유지 비교

- 사용자 요청에 따라 두 모델의 **16,500 epoch/frame 2,162,688,000**을 맞추고 행동 전 loco 초기 관측 64장면에 배경을 누적 추가했다. 최종 비교는 **8/10/12/16/20/24/30토큰**, 상자만 추가 및 사람·상자·목표 동시 추가, 배치 seed 17/42/73이다. 원래 객체·과제는 그대로이며 추가 사람은 과제가 없다. 최대 혼합 장면은 7H/16O/7G로 물리 rollout 없는 관측 probe다.
- 8→30토큰의 과제 attention 합 유지율(원래 두 사람·4층·2head 평균)은 상자만 추가 시 Action **75.1%**, Entmax **87.2%**, 혼합 추가 시 **72.3%/80.9%**다. 혼합 마지막 층만 보면 **95.5%/90.2%**, 정책 평균 출력 상대 RMS 변화는 **5.2%/12.8%**이므로 전체 층 평균 희석 완화가 최종 행동 안정성 개선을 뜻하지 않는다. 같은 L1 점수에 정규화 함수만 바꾼 국소 비교도 별도 저장했다.
- `output_etc/token_scaling_20261012/`에 코드·checkpoint hash·원시 확장 관측·attention·지표·한국어 PNG 4종을 보관한다. 동일 epoch/frame·두 모델 입력 동일·누적 추가 시 기존 feature/pose 보존·기존 토큰의 L1 logit 변화 0·계측 출력 동일·finite·행합 오차 <3.5e-6·문법·PNG 시각 확인 통과. 확장 행동 성공률과 재학습은 검증하지 않았으며 본학습·모델 코드는 변경하지 않았다.

### Entmax·Action 실제 attention map 비교

- Action 35,000의 64환경·loco 시작 rollout에서 0/60/120 step의 192개 장면 관측을 추출하고, 같은 원시 관측을 Action 35,000과 role gate/Entmax 16,500의 저장 정규화·actor에 입력했다. 사람 2명·상자 8개·목표 2개를 추가한 4H/12O/4G 관측도 비교했다. 추가 사람은 과제가 없는 배경 인물이며, 확장 장면의 물리 rollout·성공률 평가 및 동일 학습량 비교는 아니다.
- 확장 관측의 원래 두 사람·4층·2head 평균: 과제 대상 attention 합 Action **80.3%** / Entmax **59.5%**, 추가 객체 합 **12.9% / 23.5%**, 정확한 0 비율 **0% / 57.1%**다. Entmax의 희소성이 더 높은 과제 집중을 뜻하지 않았으며, Action 궤적 표본·서로 다른 구조/학습량의 비교라 정규화 함수만의 인과 효과로 해석하지 않는다.
- `output_etc/attention_comparison_20261012/`에 재현 코드·원시 관측·head별 attention/logit NPZ·summary와 한국어 PNG 5종을 보관한다. 네 과제 예시·전체 map·같은 Entmax logit에 Softmax만 적용한 국소 비교를 포함한다. 실제 player 대비 정규화 차이 0·정책 평균 출력 최대 차이 8.05e-7, 계측 전후 출력 동일·token 순열 map 오차 <1e-6·행합 오차 <1.5e-6·finite 검사 및 PNG 시각 확인 통과. 본학습·모델 코드는 변경하지 않았다.

### Action·Role gate 최신 저장본 재평가 (16:12)

- Action 34,500 / role gate 16,000을 이전과 같은 64환경·2명·4물체·600 step·seed 42·loco 시작·random scenario 1회로 순차 평가했다. 초기 task/binding packet이 이전 평가 및 두 모델 간 일치한다.
- 종료 성공 수(SIT/CLIMB/AT/ON_TOP, 표본 11/37/45/35)는 action **5/37/21/33**, gate **1/0/0/0**. 직전 평가 대비 action AT는 14→21, 전체 개인 목표 성공은 70.3→75.0%; gate는 SIT 1건이 생겼고 나머지는 평가 중 성공도 0이다. Gate 종료 HOLDING 조건 만족은 32.5%로 같다. 단일 seed 소표본이며 학습량은 다르다.
- 두 실행 종료 코드 0·finite 지표·표본 합·성공 수 범위를 확인했다. 실행 코드·manifest·검증·결과는 `output_etc/rough_eval_20261012_161156/`에 보관한다. 기존 두 학습 프로세스가 계속 실행 중임을 확인했으며 모델·학습 설정은 변경하지 않았다.

### Action·Role gate 간단 평가

- 사용자 요청으로 action 32,000 / role gate 13,500 checkpoint를 GPU 0에서 순차 평가했다. 모델당 64환경·2명·4물체·600 step·seed 42·loco 시작·random scenario 1회이며 task 교체 probe는 제외했다. 초기 task/binding packet은 두 모델 및 10월 11일 평가와 일치한다.
- 종료 시 개인 목표 성공 수(SIT/CLIMB/AT/ON_TOP, 표본 11/37/45/35)는 action **5/37/14/34**, gate **0/0/0/0**. Gate의 마지막 HOLDING 조건 만족 비율은 32.5%이나 과제 목표는 평가 중 한 번도 달성하지 못했다. 학습 RSI 혼합 지표와 구분하며 단일 seed 소표본 결과다.
- 두 실행 종료 코드 0·128개 agent 표본 합·성공 수 범위·finite 지표를 확인했다. `output_etc/rough_eval_20261012/`에 실행 코드·manifest·summary·로그를 보관한다. 학습 프로세스와 설정은 변경하지 않았다.

## 2026-10-11

### Task embedding SIT 정체 진단

- 사용자 요청으로 최신 로그의 SIT 배정·RSI·보상과 embedding 35,000 / action 23,500의 SIT 전용 평가를 확인했다. 모델당 loco/sit 모션 초기화 각각 32환경·2명·600 step·seed 42 1회: 종료 성공은 embedding **0/64, 5/64**, action **28/64, 27/64**다. Sit 모션 초기화는 이미 앉은 상태를 강제한 유지 평가가 아니다.
- Embedding의 최근 SIT 접근 보상은 0.185/최대 0.2이나 성공 상태 비율은 약 10.5%다. Loco 평가 마지막 표본에서 목표 XY 거리 0.707m·높이 오차 0.367m였다. Actor 대상 edge attention은 층별 평균 0.410/0.161/0.116/0.081이며, 같은 관측의 SIT 임베딩을 CLIMB로 교체하면 정책 평균 출력 상대 RMS가 39.6% 변했다. Task 무시·대상 attention 포화 가설은 이 측정으로 지지되지 않으며, 낮은 attention의 인과 효과나 과제별 PPO gradient 충돌은 검증하지 않았다.
- 두 모델과 task MLP의 variant 제외 reward 설정은 같고, 최신 두 실행의 SIT 배정 약 10%·loco/sit RSI 약 50:50·RSI 거부/대체 0을 확인했다. 평가 4건 종료 코드 0, 동일 입력/임베딩 재실행의 출력 차이 0. 분석·코드·로그는 `output_etc/sit_diagnosis_20261011/`에 보관하며 학습 중단·학습 설정 변경은 하지 않았다.

### 실행 중인 세 모델의 소규모 평가·task 입력 민감도

- 사용자 요청으로 embedding 34,500 / action 23,000 / role gate 5,000 checkpoint를 GPU 0에서 순차 평가했다. 모델당 64환경·2명·4물체·600 step·seed 42·loco 시작·random scenario 1회이며, 초기 task/binding packet hash가 일치했다. 학습 프로세스·모델 코드는 변경하지 않았다.
- 종료 시 개인 목표 성공 수(SIT/CLIMB/AT/ON_TOP, 표본 11/37/45/35): embedding **0/37/31/31**, action **4/37/5/33**, gate **0/0/0/0**. 학습량이 다르고 단일 seed 소표본이며, 학습 RSI 혼합 성공 상태 비율과 구분한다.
- 기준 rollout의 10개 시점에서 actor task embedding을 순환 교체했을 때 정책 평균 출력의 상대 RMS 차이는 embedding **18.7~20.4%**, action **22.5~29.9%**, gate **4.2~6.3%**다. Action의 bias를 유지하고 직접 주입 경로만 교체하면 **22.5~30.3%**, 제거하면 **23.2%**다. 대체 행동으로 rollout하지 않았으므로 성공률 기여도/인과 효과를 뜻하지 않는다.
- 세 평가 종료 코드 0, 128개 agent 표본 합·성공 수 범위·probe finite를 확인했다. 스크립트·실행 인자·결과는 `output_etc/rough_eval_20261011/`에 보관한다. NVML 버전 불일치로 nvidia-smi는 실패했지만 CUDA 연산·여유 메모리 확인 후 평가를 완료했다.

## 2026-10-10

### Task/Role gate 역전파 속도 최적화

- Profiler에서 gate category 조회의 backward가 12개 행으로 gradient를 집중 합산하는 `_index_put_impl_`에 큰 시간을 쓰는 것을 확인했다. 학습 시 one-hot selector × 12×16 gate 표로 바꾸고, 4D block 확장은 조회 뒤에 수행한다. Rollout/frozen 및 FP32 TF32 활성화 경로는 compact index lookup을 사용한다. 모델 수식·파라미터·checkpoint key·AMP·RSI·보상·셔플·Entmax·실행 config는 유지한다.
- 동일 checkpoint·입력의 GPU 교차 측정에서 **8192 scene actor+critic forward/backward 170.0→87.3ms(약 49% 감소)**, 재측정 332.6→163.0ms(약 51% 감소). 공유 GPU이므로 절대 시간은 변동하며 전체 iteration 단축률을 뜻하지 않는다. 출력 최대 차이 1.19e-7, 180개 gradient tensor 최대 차이 1.11e-7. Entmax prefix GEMM 후보는 개별 연산만 빨라지고 전체 모델 추가 개선은 없어 제외했다.
- **CPU 137개 + CUDA TF32 fallback 1개 통과**. 기존 checkpoint를 optimizer와 함께 복원해 **2048환경·epoch 2→3/frame 262144→393216** 확인 학습·저장을 완료했다. Optimizer state 186개 step 96→144, gate/task embedding 갱신·321 scalar finite·물리 reset 실패 0을 확인했다.
- 분석 자료는 `output_etc/task_role_gate_lookup_opt_20261010/`, 확인 실행은 `output_etc/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax_lookup_opt_check/`에 보관한다. 실행 중인 본학습은 중단·재시작하지 않았으며 새 코드 적용에는 저장 checkpoint에서 재개가 필요하다. 장기 성능 동일성은 검증하지 않았다.


### output 검증·분석 산출물 보관 정리

- 사용자 요청으로 gate/Entmax 짧은 학습·네 과제 평가, task action 영향 분석, rough loco 평가, 날짜별 trend 분석의 **14폴더·87파일(약 53MiB)**을 기존 `output_etc/`로 이동했다. `output/`에는 본학습 6폴더와 `rsi_cache`를 유지하고 실행 중인 학습 3개는 변경하지 않았다.
- 이동 대상의 실행 인자 참조·열린 쓰기 파일·목적지 충돌이 없음을 확인하고, 동일 파일시스템 rename 뒤 87파일의 크기·SHA256 일치를 검사했다. 대응 경로와 hash는 `output_etc/archive_manifest_20261010.json`에 보관한다. 문서의 보관본 경로·링크를 갱신했으며 기존 실행 명령의 신규 output 경로와 산출물 내부의 원래 실행 기록은 유지한다. 문서 링크·diff 검사 통과; 학습·평가는 새로 실행하지 않았다.


### Task/Role GTA K/V gate + Entmax 연결

- `task_embedding` 기반 `approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax` env/train YAML·train/test/VNC·output을 분리했다. Env는 variant만 변경하고 과제·보상·크기·RSI·AMP·634-D packet·edge/token 순열을 유지한다. 새 attention mode는 additive bias/message를 끄며 전용 fusion 계약으로 기존 checkpoint 혼용을 거부한다.
- Actor/critic 독립 `Embedding(4,64)`와 layer별 `67→64→16` SiLU gate를 연결했다. Task/role 12조합을 재사용하고 4D GTA block별 같은 gate로 K/V를 함께 조절한다. GTA 뒤 sparse correction·전체 scene Entmax 1.5·GTA 복귀 순서를 유지하며, global 초기화 뒤 gate=1을 복원한다. 전체 4,164,290개 파라미터다.
- 관련 CPU **135개 통과**: Entmax bisection/finite-difference·sparse/dense 출력/gradient·GTA 교환/삽입 순서·padding·non-task 후보·human/object/goal 재배정 및 토큰/edge/task 순열·actor/critic 독립 업데이트·AMP/RSI/보상·checkpoint·기존 attention 회귀. 셸/Python 문법·문서 경로·diff 검사 통과.
- **GPU 0·2048환경·MAX_ITERATIONS=1** 학습 완료(epoch 2/frame 262144). RSI 캐시 hit, scalar **321종/642값 모두 finite**, 물리 reset 실패 0, 진단 CSV 16행/8그룹의 binding·보상 합산 오류 0. Actor/critic task embedding 네 행 및 gate MLP 전체 34개 tensor의 optimizer moment가 finite/nonzero다. Task attention mass·zero-rate·support 수·gate/embedding gradient norm을 기록한다.
- 저장 모델로 SIT/CLIMB/carry_at/carry_ontop 각각 **16환경·32-step headless 평가** 완료. Carry는 carryWith 시작, ON_TOP은 role swap을 적용했다. 검증 자료는 `output_etc/approach_scenario_stage1_unified_size_rsi_task_role_gate_entmax_check/verification.json`·`validation/` 및 네 `_check_<task>/`에 저장했다. 기존 학습 3개를 유지했으며 새 본학습·장기 성능·distractor/근접 장애물 행동 비교·확장 scene·VNC 화면·Stage 2는 검증하지 않았다.

## 2026-10-08

### Task embedding + 사람 토큰 행동 정보 주입

- 사용자 합의대로 `approach_scenario_stage1_unified_size_rsi_task_embedding_action` env/train YAML·train/test/VNC·output을 분리했다. 기존 task embedding의 과제·보상·상자 크기·RSI·AMP·캐시·634-D packet은 유지한다. Task 4개 `Embedding(4,64)`를 타입별 bias projection과 직접 주입 MLP가 공유하며 NONE/SELF bias는 항상 0이다. 관계 message는 없다.
- GTA 복귀·head concat 뒤 output projection 전에 사람 토큰에 `z += MLP([z,e_task])`를 적용한다. 128→64→64 ReLU MLP를 네 layer가 공유하고 actor/critic은 독립이다. 명시적 actor endpoint를 토큰 순열에 맞춰 옮겨 task record·edge 순서와 무관하게 주입한다. 출력층은 전역 초기화 뒤 0으로 초기화하며 초기 주입은 0이다. 전체 모델은 4,155,202개 파라미터다.
- 관련 CPU **103개 통과**: 실제 주입 위치·사람/padding 마스크·task 교체·토큰/edge/task 순열 출력 및 gradient·업데이트·SELF/NONE=0·checkpoint 격리·보상/샘플링/RSI/AMP·기존 실험 회귀. 전용 fusion metadata로 이전 실험 checkpoint 혼용을 거부한다.
- **GPU 0·2048환경·MAX_ITERATIONS=1** 확인 학습 완료(저장 epoch 2, frame 262144). 기존 RSI 캐시 hit, scalar **257종/514값 모두 finite**, 물리 reset 실패·동료 보상 기여 0. Actor/critic의 네 embedding 행·task projection·새 MLP 전체 optimizer moment가 nonzero이며 SELF/NONE bias는 학습 후에도 0이다. 진단 CSV 16행/8개 그룹의 물체 binding·자기 보상 합산 오류 0.
- 저장 모델의 carry_at/carry_ontop + carryWith **각 16환경·32-step headless 평가** 완료. ON_TOP은 `TASK_ROLE_SWAP=1`도 적용했다. 검증 산출물은 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_action_check/` 및 `_check_carry_at/`, `_check_carry_ontop/`다. 실행 중인 기존 학습은 유지했고 새 본학습·장기 성능·VNC 화면·학습된 task 입력 ablation은 검증하지 않았다.

## 2026-10-07

### 보정 공동운반 RSI 인계 정리

- 다음 Codex의 RSI 연결을 위해 공동운반 생성·보정·검사 코드를 `joint_carry/scripts/`로 모으고 보정 전 4개 산출물 폴더를 `joint_carry/archive/`로 분리했다. 보정본과 최종 물리 결과는 루트 바로 아래 유지하며 실행 경로·config·structure를 갱신했다.
- `joint_carry/HANDOFF.md`에 319개 paired snapshot의 배열 계약, 공유 상자·두 사람 동시 reset, graph/binding·AMP 이력의 구현 진입점, 검증 범위와 후속 요청을 정리했다. 무결성 확인 코드·SHA256 manifest·코드/보정본/영상을 포함한 `joint_carry_handoff.zip`을 추가했다. 학습 연결은 미구현이며 연속 전문가 모션 승인을 뜻하지 않는다.
- 이동 전후 바이너리 131개의 SHA256 보존, 전달 파일 79개·의존 파일 3개 무결성, ZIP 80개 항목 CRC 및 임시 위치 압축 해제 후 재검사를 통과했다. 이동한 검사 코드로 세 동작 각각 CPU PhysX 6-step smoke·FK 오차 <1e-6m·finite/충격 기준을 확인했다. 문서 링크·Python 문법·diff 검사 통과. 전체 물리 검사 재생성이나 학습은 실행하지 않았다.

### 공동운반 산출물 joint_carry로 이동

- 사용자 요청으로 이번 영상·관절 데이터·물리 검증·ZIP의 6개 폴더, 170파일을 `output/`에서 저장소 루트 `joint_carry/`로 이동했다. 최종 영상·데이터 링크를 `joint_carry/README.md`에 모으고 스크립트 기본 경로·실행 문서·기존 기록의 산출물 경로를 갱신했다. 생성 산출물의 기존 Git 제외 정책도 유지했다.
- 파일 163개는 SHA256이 그대로이며, 나머지 7개는 메타데이터·로그·ZIP 내부 경로만 갱신했다. 이동 누락 없음·검증 보고서의 입력 SHA256 12개·ZIP 2개 CRC·안내 링크·Python 문법·diff 검사를 확인했다. 모션 생성·물리 검사·학습은 다시 실행하지 않았다.

### 공동운반 실제 관절 보정·RSI snapshot 추출

- 사용자 요청으로 `correct_carry_candidates.py`를 추가했다. 물리 캐릭터의 1축 팔꿈치·고정 손목으로 변환한 뒤 손 접촉·팔 capsule 관통을 함께 고려하는 IK를 적용한다. 상자 크기·15cm 안쪽 잡기 위치·다리/몸통/머리 관절각은 유지하고 root를 클립별 1.4~2.6cm 올렸다. 팔 DOF 최대 변화는 pickUp 42.0°·carryWith 29.4°·putDown 36.6°다. 수정 궤적에서 속도를 재계산했다.
- 보정본을 같은 CPU PhysX 검사로 재검증했다. 원본 RSI 허용 구간 **pickUp 55/55·carryWith 129/129·putDown 135/135**가 0.1초 충격·관통 검사를 통과했다. carryWith는 두 사람이 6 step 중 최소 5 step에서 각각 손 접촉을 보였다. 운반 중 손/팔 최대 관통은 11.51/10.10cm→약 0.50/0cm, 실제 FK와 영상 자세의 최대 오차는 6.44e-7m 미만이다. 이는 단기 초기 상태 검사이며 연속 발 미끄러짐·장기 PD 균형은 해결하지 않았다.
- `joint_carry/carry_joint_corrected/`에 보정 reference, 독립 `screened_rsi_snapshots.npz` 319개, `before_after/comparison_all_skills.mp4`, `corrected_rsi_bundle.zip`을 저장했다. 물리 보고서는 `joint_carry/carry_corrected_physics/`다. Finalize는 후보 SHA256 일치·원본 제외 마스크·충격/형상 통과를 확인하고 carryWith에는 양쪽 손 접촉 조건을 추가한다. `training_loader_integrated=false`, `continuous_motion_validated=false`이며 실제 학습/graph reset에는 연결하지 않았다.
- 하체/몸통/머리 각도 보존(수치 오차 4.45e-16rad 이하)·clone 내부 관절 일치·속도 quaternion 차분·snapshot 원본 프레임 매칭·finite·ZIP 22파일 CRC를 확인했다. 영상 14개 전체 디코딩·세 동작 대표 화면·Python 문법·diff 검사 통과. 기존 학습·원본 데이터·앞선 후보는 수정하지 않았다.

### 공동운반 후보의 실제 형상·PhysX 검사

- 사용자 요청으로 `validate_carry_physics.py`를 추가하고 15cm 안쪽 합성본과 원본의 **728개 프레임**을 CPU PhysX에서 검사했다. 기존 asset/PD/마찰/중력/밀도·1/60초·2 substep으로 0.1초 RSI 충격 검사와 root를 강제 이동하지 않는 동작별 PD 추종을 실행했다. 학습 GPU는 사용하지 않았고 입력·캐시·checkpoint도 수정하지 않았다. 결과는 `joint_carry/archive/carry_physics_validation/`의 `report.json`, 프레임 CSV·마스크·실제 FK·rollout·요약 PNG다.
- 원본 RSI 허용 구간에 0.1초 충격+관통 2cm 이하 기준을 적용하면 합성 **pickUp 29/55(집기 전 0~28)·carryWith 0/129·putDown 0/135**다. 충격만 보면 55/55·129/129·128/135다. 실제 형상에서 carryWith 손/팔 최대 관통 11.51/10.10cm를 확인했다. 원본도 동일한 엄격한 형상 기준에서 carryWith는 탈락하므로 기존 충격 캐시 통과와 접촉 타당성을 구분한다. RSI 승인이나 학습 연결은 하지 않았다.
- 앞선 영상 재현 설명을 정정했다. 영상/NPZ의 full quaternion은 동일하지만 실제 팔꿈치는 Y축 1-DOF여서 기존 `MotionLib` 32-DOF 변환이 X/Z를 제외한다. 세 클립의 실제 손 위치 오차는 최대 8.45cm이고, 손을 제외한 body 원점은 전체 프레임에서 2e-5m 이내로 일치했다. 검사 형상은 **실제 시뮬레이터 FK**를 사용했다. 이후 생성기의 메타데이터/README와 실행 문서에 차이를 명시했다.
- 실제 PD 추종에서 원본·합성 모두 장기 균형/운반 유지에 실패했다. 이는 학습된 제어기 없이 관절 target만 준 결과이며 학습 불가능의 증거가 아니다. 형상 거리의 수평 회전/이동 불변성(오차 7.8e-16m), 자유낙하 0.1초(5.31cm·-0.980m/s), 입력 SHA256 보존·Python 문법·diff 검사 통과. 캡슐 표본·사람 간 겹침 하한·CPU/GPU 차이 등 검사 범위는 보고서에 기록했다.

### Carry 원본·공동운반 관절 비교 영상

- 후속 요청에 따라 `--grip-inset 0.15`로 몸 전체만 각 15cm 안쪽으로 옮기고 `--skill all`로 pickUp B19·carryWith B20·putDown B21을 모두 생성했다. `joint_carry/archive/carry_joint_preview_all_inset15/`에 동작별 MP4/NPZ·약 12.13초 모음·데이터 ZIP을 저장한다. 세 클립은 별개이며 연속 궤적 합성은 아니다. 원본 RSI 제외 마스크를 보존하고 합성 물리 승인 마스크는 모두 false다.
- 세 동작의 내부 관절각·뼈 길이 보존, 기존 영상 대비 15cm root 이동, 원본 FK 최대 오차 3.63e-7m 이하·finite·제외 마스크를 확인했다. 영상 16개 전체 디코딩과 세 동작 대표 화면을 확인했고 결과는 `validation.json`에 기록했다. 관절 데이터는 영상과 같으며, 실제 RSI 채택에는 root·상자 상태를 포함한 동시 reset과 물리 검증이 필요하다.
- 사용자 요청으로 CPU 전용 `carry_joint_preview.py`를 추가했다. B20 carryWith 30fps·129프레임의 관절 FK를 재생하고, 상자 양 끝에 원본 자세와 180° 복제 자세를 배치한다. `joint_carry/archive/carry_joint_preview/`에 비교·개별·반속 MP4, 원본/공동운반 관절·속도 NPZ, 합성 방법·진단 JSON을 저장했다. 원본 데이터·학습 설정은 변경하지 않았다.
- 원본 FK와 `SkeletonMotion`의 최대 오차 2.97e-7m, 32-DOF와 `MotionLib`의 각도 오차 1.27e-6rad를 확인했다. 속도 비교는 float32 입력의 작은 회전 차분에서 0.00234rad/s 오차가 보여 float64 입력으로 재확인했고, 전체 128구간 최대 오차는 2.37e-7rad/s다. 복제 FK·관절각 동일성·finite 검사, MP4 4개 전체 디코딩, 대표 3프레임 렌더, Python 문법·diff 검사 통과.
- **RSI 후보이며 물리 검증·IK·reset 연결은 미완료**다. 낮은 발의 수평 속도 proxy가 원본 0.25m/s 대비 복제 사람 1.58m/s로 커서 단순 복제 클립의 발 미끄러짐을 확인했다. NPZ/JSON에 `rsi_ready=false`, `physics_validated=false`를 명시했다. GPU 학습·시뮬레이션은 실행하지 않았다.

### Task embedding·물리 타입 쌍별 projection 실험

- 사용자 합의대로 `approach_scenario_stage1_unified_size_rsi_task_embedding` 전용 env/train config와 train/test/VNC·output을 연결했다. Actor/critic 각각 task 4개+NONE+SELF의 `Embedding(6,64)`와 `[src H/O/G,dst H/O/G,layer,head,64]` projection을 사용한다. 카테고리 간 projection은 공유하며 역할 입력·중간 MLP·관계 message는 없다. ON_TOP의 H→운반 상자와 H→받침은 같은 bias를 받는다.
- 공유 task MLP와 env의 variant·train의 mode만 다르다. 보상·과제·RSI·AMP·634-D packet·캐시·GTA·순열 경로를 유지하고 전용 checkpoint 계약으로 혼용을 거부한다. Bias encoder는 각 4,992개, 전체 모델은 4,130,626개 파라미터다.
- 관련 CPU **91개 통과**: 새 수식·타입/카테고리 공유·NONE/SELF·역방향/padding·endpoint 재매핑·토큰/edge/task 순열의 출력과 gradient·6개 embedding 업데이트·RSI/AMP/보상·checkpoint 및 기존 실험 회귀. Python/셸 문법·문서 링크/명령 경로·diff 검사도 통과했다.
- **GPU 0·2048환경·MAX_ITERATIONS=1** 확인 학습 완료(저장 epoch 2, frame 262144). 기존 RSI 캐시 hit, scalar **233종/466값 모두 finite**. 물리 reset 실패·단독 HOLDING·공유 primary 물체·동료 보상 기여는 0이고 진단 CSV 16행의 binding·보상 합산 오류도 0이다. 양쪽 branch의 6개 category gradient와 9개 타입 쌍 projection 업데이트를 확인했다.
- 저장 모델로 carry_at/carry_ontop + carryWith 각각 **16환경·32-step headless 평가**와 JSON 저장을 완료했다. Output은 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_check/`, `_check_carry_at/`, `_check_carry_ontop/`이다. 기존 학습 세 개는 유지했다. 새 본학습·장기 성능·VNC 화면은 검증하지 않았다.

### Task 3종 실행 가이드 정리

- `markdowns/config.md` 상단에 빠른 확인을 추가하고 message·공유 MLP·split의 구조 차이, 네 task와 역할별 edge, 보상·RSI·AMP·평가 기본값 및 bias/성공률 해석을 정리했다. 실제 로컬 데이터 경로와 유효한 12개 심링크를 확인해 반영했다.
- 2026-10-06 23:10 KST에 확인한 본학습 성공률과 bias checkpoint 기준을 과거 확인 기록으로 명시했다. 실시간 상태 및 별도 평가 성공률과 구분하고 전체 분석 산출물을 연결했다.
- 실제 코드·YAML·wrapper와 대조했고 문서 링크 34개·실행 script/config 경로 99개, 내부 anchor·코드 블록·`git diff --check`를 확인했다. 문서만 수정했으며 학습·평가를 추가 실행하지 않았다.

## 2026-10-06

### Task별 독립 MLP·projection 비교 실험

- 사용자 요청으로 `approach_scenario_stage1_unified_size_rsi_task_mlp_split` env/train YAML·train/test/VNC·output을 추가했다. 기존 task MLP의 역할 embedding·task embedding 표는 유지하고 sit/climb/carry_at/carry_ontop별 MLP와 layer/head projection을 분리했다. 한 task 내부 역할 연결은 같은 전용 MLP를 쓰며 NONE/SELF 배경은 기존 별도 MLP다. Actor/critic은 독립이고 관계 메시지는 없다.
- 과제·보상(`self=1, teammate=0`)·RSI·AMP·크기·634-D packet은 기존 task MLP와 동일하다. 총 36개 task/역할 조합만 인코딩한다. Task encoder는 각 35,552개, 전체 추가 파라미터는 52,992개(+1.27%)다. 전용 variant/mode/fusion과 weight shape로 기존 checkpoint 혼용을 거부한다.
- 관련 CPU **84개 통과**: task별 weight/gradient 독립성, 공유 weight 복제 시 출력 및 gradient 합 동등성, 16개 과제 쌍·순열·역할/물체/goal binding·자기 보상·RSI/AMP·checkpoint 격리·기존 실험 회귀. Python/셸 문법·문서 링크와 명령 경로·diff 검사 통과.
- **GPU 0**, 2048환경·`MAX_ITERATIONS=1` 확인 학습 완료(저장 epoch 2, frame 262144). 기존 RSI 캐시 hit, scalar **233종/466값 모두 finite**, 물리 reset 실패·단독 HOLDING 샘플·동료 보상 기여 **0**. Actor/critic의 네 task MLP·projection 모두 업데이트됐고 진단 CSV의 binding·보상 합산 오류는 0이다.
- 저장 checkpoint로 carry_at/carry_ontop + carryWith 각각 **16환경·32-step headless 평가**를 완료했다. 검증 output은 `output/approach_scenario_stage1_unified_size_rsi_task_mlp_split_check/`, `_check_carry_at/`, `_check_carry_ontop/`이다. 기존 본학습 두 개를 유지했다. 새 본학습·장기 성능·VNC 화면·공유 대비 통제된 속도 비교는 수행하지 않았다.

## 2026-10-05

### 통합 task·역할 MLP bias 실험

- 사용자 요청으로 `approach_scenario_stage1_unified_size_rsi_task_mlp` env/train YAML·train/test/VNC·output을 분리했다. Env는 task message와 variant만 다르며, sit/climb/carry_at/carry_ontop 분포·3연결·자기 보상 합계(`self=1, teammate=0`)·크기·RSI·AMP·634-D packet을 공유한다.
- Task·출발/도착 역할 임베딩을 `64→64→64` MLP와 layer/head projection으로 attention bias에 연결했다. NONE/SELF 배경도 기본 Size RSI의 MLP 방식이며 actor/critic은 독립이다. 관계 메시지 파라미터·가산 경로는 없고 GTA는 유지한다. 전용 variant/fusion 계약으로 기존 checkpoint 혼용을 거부한다.
- 관련 CPU **70개 통과**: 새 설정 동일성·역할별 bias·메시지 부재·순열 출력/gradient·optimizer 업데이트·RSI/AMP binding·자기 보상·checkpoint 분리 및 기존 task-message/typed-bias/Size RSI 회귀. Python/셸 문법·문서 링크/실행 경로·diff 검사 통과.
- **GPU 0**, 2048환경·`MAX_ITERATIONS=1` 확인 학습 완료(저장 epoch 2, frame 262144). 내려받은 RSI 캐시 hit, scalar **233종/466값 모두 finite**, 물리 reset 실패 **0**. 저장 모델에 관계 메시지 파라미터가 없고 actor/critic task MLP bias projection 업데이트를 확인했다.
- 저장 checkpoint의 carry_at/carry_ontop + carryWith **16환경·32-step headless 평가**를 각각 완료했다. 결과는 `output/approach_scenario_stage1_unified_size_rsi_task_mlp_check/`, `_check_carry_at/`, `_check_carry_ontop/`이다. 본학습·장기 성능·VNC 화면은 검증하지 않았고 기존 학습은 중단하지 않았다.

### 복합 task·역할별 bias/message 실험

- 사용자 요청으로 `approach_scenario_stage1_unified_size_rsi_task_message` env/train YAML·train/test/VNC·output을 분리했다. 단독 HOLDING 없이 sit/climb/carry_at/carry_ontop 10/25/32.5/32.5%, 자기 primitive 보상 합계만 사용한다(`self=1, teammate=0`). Carry 보상은 2로 나누지 않는다.
- 현재 primitive graph의 owner/endpoint에서 `[valid,task,actor,payload,target]` 정책 packet을 만든다(634-D 관측). Carry는 actor→payload/actor→target/payload→target, sit/climb은 actor→target이다. Task·역할별 bias `[4,3,3,4,2]`, message `[4,3,3,64]`, NONE/SELF 배경 표를 actor/critic 각각 학습한다. Message alpha=1·std=.02·GTA 위치는 기존 방식이다. 보상·RSI·AMP는 동일 primitive graph를 계속 사용해 새 연결의 보상 중복을 막는다. Packet v5로 기존 checkpoint 혼용을 거부한다.
- CPU **132개 통과**: 새 16가지 task 쌍/물체·goal·owner 재매핑/edge·task·token 순열의 출력·gradient, carry_ontop 위아래·AT goal, 자기 보상/동료 비의존, 크기 분포·RSI/AMP binding·strict checkpoint와 기존 정책/보상 회귀. Python/셸 문법·새 문서 링크·diff 검사 통과.
- **GPU 5**, 2048환경·`MAX_ITERATIONS=1` 확인 학습 완료(저장 epoch 2, frame 262144). 기존 RSI 캐시 hit, 257종 scalar/514값 모두 finite, 물리 reset 실패 0, 단독 HOLDING 샘플 0. Actor/critic 각각 사용되는 8개 task-role 행의 bias와 메시지 optimizer 업데이트를 확인했다. 진단 CSV의 물리 binding·자기/동료 보상 대응 오류 0.
- 저장 checkpoint로 16환경·32-step headless 평가를 random 및 `carry_at/carry_ontop + carryWith` 세 설정에서 완료했다. 검증 output은 `output/approach_scenario_stage1_unified_size_rsi_task_message_check/`, `_check_eval/`, `_check_carry_at/`, `_check_carry_ontop/`이다. 본학습을 시작하거나 기존 학습을 중단하지 않았다. 장기 수렴·성능 개선은 미확인이다.


### Size RSI typed bias + 관계 메시지 실험

- 사용자 요청으로 `approach_scenario_stage1_unified_size_rsi_typed_bias_message` env/train YAML·train/test/VNC·output·checkpoint variant를 분리했다. 기존 typed bias와 Size RSI 보상·성공·크기·RSI·캐시는 유지한다.
- Actor/critic 각각 `[3,11,3,64]` 관계 표를 추가했다. `alpha=1.0`, 초기 표준편차 `0.02`, layer 공유·head별 32차원이며 NONE/SELF·비과제 메시지는 0이다. 같은 attention으로 유효 edge 메시지를 source에 합산하고, GTA 복귀 뒤 output projection 전에 더한다. Sparse gather/scatter와 토큰 순열의 endpoint 재매핑을 사용한다. 메시지 RMS와 노드 대비 비율을 TensorBoard에 추가했다.
- CPU **123개 통과**: 새 메시지 수식·GTA 위치·gradient, 전체 과제 쌍/goal slot·토큰/edge 순열의 action/value/gradient, alpha=0 기존 출력 일치, 독립 업데이트·strict checkpoint 및 기존 정책·graph/reward 회귀. Python/셸 문법·문서 링크·diff 검사 통과.
- 사용자 GPU 지정에 따라 **GPU 5**에서 2048환경·`MAX_ITERATIONS=1` 확인 학습(저장 epoch 2, frame 262144)과 16환경·32-step headless checkpoint 평가를 완료했다. 기존 RSI 캐시 hit, scalar **257개/514값 모두 finite**, 물리 reset 실패 **0**. 양쪽 관계 표의 다섯 과제 행 모두 optimizer 업데이트를 확인했고 NONE/SELF 값은 0이다. 검증 output은 `output/approach_scenario_stage1_unified_size_rsi_typed_bias_message_check/`, `_check_eval/`이다. 이는 실행·업데이트 검증이며 장기 성능 개선은 미확인이다.

## 2026-10-04

### Size RSI 기반 독립 typed-bias 표 실험

- 사용자 요청에 따라 `approach_scenario_stage1_unified_size_rsi_typed_bias` env/train config와 전용 train/test/VNC·output을 추가했다. 과제·크기·RSI·물리 캐시·AMP·보상·성공·`r_valid`는 기존 Size RSI와 같고 variant/네트워크 계약만 분리한다.
- Semantic edge MLP·projection 대신 0 초기화한 `(source type, relation, target type, layer, head)` 표 `[3,11,3,4,2]`를 사용한다. Actor·critic은 각각 792개 파라미터의 별도 표를 갖는다. Directed edge·NONE/SELF·토큰/GTA/bias 동시 순열·human readout 복원은 유지한다. 잘못된 env/train mode 조합·표 공유·bias 비활성화는 실행 전에 거부한다.
- CPU **96개 통과**: 전체 25개 과제 쌍·goal 슬롯 교환·배경/방향·MLP bias 표현 동등성·토큰/edge 순열의 action/value/gradient·독립 표 업데이트·strict 저장/복원·기존 graph/reward/policy 회귀. Python/셸 문법·문서 링크·diff 확인.
- GPU 1에서 2048환경·`MAX_ITERATIONS=1` 확인 학습(저장 epoch 2)과 checkpoint 저장을 완료했다. 기존 RSI 캐시 hit, scalar **233개/466값 모두 finite**, 물리 reset 실패 **0**. Actor/critic 표가 각각 0에서 서로 다른 finite 값으로 갱신됐다. GPU 5의 전용 test wrapper로 해당 checkpoint를 로드해 16환경·32-step headless 평가와 JSON 저장을 완료했다. 검증 output은 `output/approach_scenario_stage1_unified_size_rsi_typed_bias_check/`, `_check_eval/`이며 장기 수렴은 미확인이다.

## 2026-10-03

### 크기 혼합·불규칙 더미와 10회 반복

- 사용자 요청에 따라 정리 데모 기본값을 10회·1500 control step(한 회 최대 시뮬레이션 시간 50초)으로 바꿨다. 상자는 가로·세로 0.40/0.45/0.50m, 높이 0.30/0.35/0.40m를 섞고 매 반복 위치·수평 회전(±15°)을 다시 샘플한다. 실제 크기에 맞춰 받침·윗단 중심과 바닥 목적지 높이를 계산한다. 전체 장면 입력·회색 표시·두 라운드 재배정은 유지한다.
- checkpoint에 저장된 학습 크기 범위(가로·세로 0.40~0.65m, 높이 0.25~0.55m)를 확인했다. CPU **5개 통과**(100개 혼합 크기 배치의 초기 겹침·높이 포함), Python/셸 문법·diff 확인. GPU 3에서 2회×90-step 설정의 짧은 평가가 각각 89 step 종료되고 크기 범위·물리 상태 finite·반복 초기화를 확인했다(`output/box_cleanup_irregular_smoke/cleanup_results.json`). VNC의 혼합 크기·회색 화면과 10회/1500-step 실행을 확인했다. 새 더미의 8개 완주는 아직 미확인이다.

### 중앙 더미의 윗단 운반·상자 색 통일

- 사용자 요청에 따라 주변 바닥 대상 상자를 없애고 16개 모두 중앙의 4×2×2단 더미로 배치했다. 두 라운드의 대상 8개는 모두 윗단이며, 이 데모의 상자는 배정·라운드와 관계없이 같은 회색이다. 상자 사이 18cm 간격을 확보했고 기존 재배정·바닥 안정 판정은 유지했다.
- 관련 CPU **5개 통과**, Python 문법·diff 확인. GPU 3의 headless와 VNC에서 첫 네 개 운반·두 번째 배정·회색 화면을 확인했다. 현재 배치의 seed 42 검사에서는 **7개 운반 후 마지막 집기 정체로 시간 초과**했다(`output/box_cleanup_pile_gap_check/cleanup_results.json`). 8개 완주는 미확인이다. 더 밀집한 배치·다른 간격·낮은 상자도 정체를 보였다. 사용자 선택에 따라 4명·16상자 전체 장면 입력을 유지하며 주변 관측 추론은 적용하지 않는다.

### 4명·16상자 공동 정리 VNC 데모

- 지정된 `ApproachScenarioStage1RescueAtKLClimb50_00008000.pth`를 새 학습 없이 사용하는 평가 전용 task/player와 test/VNC wrapper를 추가했다. 네 명을 십자로 배치하고 바깥 바닥 상자 8개·중앙 2단 상자 8개를 만든다. 네 목표 상자가 목적지 바닥에서 0.5초 안정되면 graph/goal만 교체하여 두 번째 운반을 시작한다. 두 라운드 사이 사람·물체의 물리 상태는 유지한다.
- 옛 rescue 설정은 데모 로더에서만 현재 self-sum runtime으로 대응시킨다. 관측 schema·semantic packet·나머지 계약과 weight/정규화 로드를 검사한다. 기존 checkpoint·일반 학습/평가 로더는 유지한다. 완료 판정은 바닥 안정이며 손 접촉 해제는 요구하지 않는다. 중앙 더미도 실제 물리 물체다.
- 관련 CPU **32개 통과**, Python/셸 문법·문서 경로·diff 확인. GPU 3·seed 42·1환경의 headless와 서버 VNC 모두 **996 control step(시뮬레이션 시간 33.2초)에 두 라운드·8개 운반 완료**를 확인했다. VNC 화면과 두 번째 운반의 카메라 구도를 확인했다. 결과는 `output/box_cleanup_demo_check_final/cleanup_results.json`, `output/box_cleanup_demo/cleanup_results.json`에 저장한다. 이는 해당 scene의 검증이며 모든 반복/seed의 성공을 보장하는 결과는 아니다.

## 2026-10-02

### Unified size RSI reset 성능 최적화

- 새 size RSI 실행 경로만 canonical graph를 GPU에서 배치 생성하고, 고정 asset의 크기별 과제 배정 확률을 최초 한 번 계산하도록 바꿨다. 캐시 RSI 전에 기존 motion/time을 뽑아 버리던 중복 작업도 제거했다. 과제·크기·프레임 분포·보상·AMP 연결·물리 검사 캐시는 유지하며 기존 unified 실행 경로는 변경하지 않았다. 같은 seed의 전체 난수 궤적은 달라질 수 있다.
- 모든 25개 과제 쌍·edge 순열·고정 preset·부분 reset의 graph 동등성 포함 CPU **59개 통과**. GPU 5의 실제 크기 조건부 graph 샘플링은 64환경 **120.65→1.05ms**, 128환경 **351.74→1.06ms**였고 기존 graph의 모든 tensor와 동일했다. 이는 해당 블록의 측정이며 전체 학습 가속 배율은 아니다.
- `output/approach_scenario_stage1_unified_size_rsi_perf_check/`에서 2048환경·`MAX_ITERATIONS=2` 학습과 checkpoint 저장을 완료했다. 기존 물리 캐시 hit, scalar **233개/699값 모두 finite**, 물리 reset 실패·공유 source **0**. 3개 기록 iteration의 환경 진행 시간은 5.02/6.91/7.05초였다. GPU 공유·초기 정책 차이가 있어 이전 학습과 통제된 속도 비교는 아니다. 기존 학습 프로세스는 종료·재시작하지 않았다.

### Unified 행동별 상자 크기·물리 검사 후반 RSI

- `approach_scenario_stage1_unified_size_rsi` config와 전용 train/test/VNC를 추가했다. 과제 비율 5/10/25/30/30, AT·ON_TOP RSI 40/10/40/10, 행동별 독립 XYZ 5cm 격자·고정 밀도 100kg/m³, 실제 bbox 반대각선과 edge별 buffer의 progress를 연결했다. 기존 unified의 state·성공 조건·보상 가중치는 유지한다.
- 실제 asset 크기로 허용 과제를 조건부 샘플링한다. 물리 randomAssignment를 끄고 source 0/1·support 2/3을 고정해 RSI·edge·AMP가 같은 배정을 사용한다. 토큰/edge 순열과 조건부 AMP expert 분포는 유지하며 family 비율만 65/10/25로 맞춘다. 별도 reward 계약으로 기존 checkpoint 재개를 차단한다.
- 실제 크기별 전체 clip 프레임의 0.1초 초기 충격 검사 캐시와 70% 후반/30% 전체 유효 프레임 샘플링을 추가했다. ON_TOP putDown은 source·받침 크기 쌍을 함께 검사한다. 유효 skill이 없으면 같은 과제의 다른 skill/loco로 대체하고 실제 clip·시간으로 pose·속도·AMP 이력을 초기화한다. Reset 시 외부 물체의 깊은 몸 침범을 재검사하며, 요청 대체율·실제 skill 비율·reference/첫 물리 step 성공 지표를 기록한다.
- CPU **46개 통과**, Python/셸 문법·문서 경로·diff 확인. GPU 6에서 **6,224개 크기/skill 조합·4,781,460개 초기 상태**를 검사해 캐시를 생성했다. 최종 설정의 2048환경·1 iteration 학습과 checkpoint 저장, 캐시 재사용을 확인했다. 물리 reset 실패 **0**, 공유 source **0**, TensorBoard scalar **233개 모두 finite**다. 혼합 16환경 및 SIT/CLIMB 각 1환경의 32-step headless 평가와 JSON 저장도 완료했다.
- 최종 검증 output은 `output/approach_scenario_stage1_unified_size_rsi_check_final/`이다. 기본 학습 seed 42로 같은 asset pool 캐시를 재사용하며, 새 seed의 누락 크기는 추가 검사한다. 첫 전체 검사는 약 25분 걸렸다. 이는 초기 충격·실행 경로 검증이며 장기 정책 성공이나 모든 RSI의 초기 own_success를 보장하지 않는다.

## 2026-09-30

### AT goal 마커의 edge 담당 색·슬롯 표시

- Viewer/VNC의 AT goal 점을 고정 빨강 대신 현재 graph의 AT edge owner 색으로 표시한다. marker 표시 여부도 owner 번호가 아닌 실제 AT destination goal 슬롯을 따른다. 레거시 marker는 agent 슬롯 색을 사용한다.
- `config.md` 상단의 학습·로컬 평가·서버 VNC 전체 명령 목록에 unified semantic/shared edge/owner-HOLDING 세 실험을 기재했다.
- Goal 슬롯이 owner 번호와 바뀐 graph를 포함한 관련 CPU 테스트 **6개 통과**, Python 문법·diff 확인. GPU 5의 1환경·16step AT 서버 VNC 평가가 종료되고 결과 JSON을 저장했다. 실행 중이던 학습은 재시작하지 않았다.

### Unified semantic actor·critic EdgeEncoder 공유 실험

- 기존 unified semantic의 과제·관측·보상은 유지하고, actor·critic의 semantic EdgeEncoder 임베딩·MLP·bias projection만 공유하는 별도 config·train/test/VNC·output·checkpoint variant를 추가했다. owner-HOLDING 경로는 변경하지 않았다.
- CPU unified 테스트 **7개 통과**: 두 encoder의 모듈 동일성·모델 파라미터 열거 시 중복 없음·양쪽 gradient·기존 checkpoint 계약 분리를 확인했다. 셸 문법·diff 검사도 통과했다.
- GPU 5에서 2048환경·`MAX_ITERATIONS=1` 별도 `_check` 학습과 checkpoint 저장을 완료했다. TensorBoard scalar **177개/354값 모두 finite**, 저장 weight의 actor/critic edge tensor 8개가 동일했다. 저장 checkpoint로 1환경·32step headless 평가도 완료했다. 이는 실행 경로 확인이며 장기 수렴 검증은 아니다.

### Unified 독립 5과제 · semantic/owner-HOLDING 두 실험

- 두 agent가 HOLDING/SIT/CLIMB/HOLDING+AT/HOLDING+ON_TOP을 5/5/20/35/35로 독립 샘플링하는 4물체 config·train/test/VNC·별도 output을 추가했다. 두 실험은 edge의 담당 HOLDING φ 입력 유무만 다르다. 기존 실험의 설정·실행 중 학습은 유지했다.
- Hᵢ/Oᵢ/Gᵢ 논리 할당과 agent별 별도 ON_TOP 받침을 고정하고, 타입 인코딩 뒤 토큰·GTA·bias를 함께 순열화한 다음 human readout 전에 복원한다. 새 실험에서만 goal quaternion을 identity로, 상자를 0.40×0.30×Z(0.30~0.50m)로, loco 담당 상자를 owner 주변 1~2m로 설정했다. placement 가까운 시작은 0이다.
- 원본 unified 기준 RSI와 carry/sit/climb 조건부 단일 AMP를 연결했다. AMP 10프레임에 과제 one-hot을 각각 추가하고 전문가·리플레이 family를 rollout과 행별 매칭한다. RSI history·부분 reset·정규화에서도 라벨을 유지한다. CLIMB 후반 강제 RSI는 끄고 ON_TOP putDown은 사용하지 않는다. 보상식·TERM 미사용은 유지한다.
- CPU 전체 **139개 통과**. 각각 2048환경·`MAX_ITERATIONS=2`(기존 runner 기준 저장 epoch 3) scratch 학습·저장 완료, scalar **177개 모두 finite**, 물리 reset 실패·공유 담당 물체 **0**. Owner actor/critic 상태 분기 weight 갱신을 확인했다. 토큰 순열의 action/value·gradient 동등성과 AMP family 매칭을 테스트했다.
- 각 checkpoint의 128환경 training-RSI 검사(전체 reset 4회·부분 reset·64step)에서 goal 회전·크기·loco 1~2m·graph/RSI·관측 packet·AMP 이력/정규화 연결을 확인했다. 전용 test wrapper의 64환경·32step headless 평가도 두 실험 모두 완료했다. 검증 자료는 `output/unified_validation/`에 보관했다. 이는 실행 검증이며 장기 수렴이나 모든 자세의 무충돌을 보장하는 결과는 아니다.

### Scenario graph 기준 상자 속도 패널티 수정

- 랜덤 object binding과 무관한 기존 `_agent_box_assignment`로 속도 패널티를 계산하던 오류를 수정했다. 현재 위치·pre-step history·부분 reset history 모두 graph의 담당 HOLDING/SIT/CLIMB 물체를 동일하게 조회한다. 기존 비-scenario 경로는 기존 할당을 유지한다.
- 소유자/물리 slot/edge 순서 변경·부분 reset·기존 경로 회귀를 포함한 관련 CPU 테스트 **19개 통과**. 저장 owner checkpoint로 128환경·64step 시뮬레이션에서 실제 패널티와 graph 기준 재계산값의 오차 **0**을 확인했다. 현재 실행 중인 학습은 재시작하지 않았다.
- 추가 분석은 `output/placement_code_audit/`에 보관했다. 기본 loco source 배치는 과거 owner 주변 1~2m와 현재 arena 반경 4.5m로 다르며, GTA goal 회전은 여전히 같은 번호 human heading을 사용한다. 상자 크기·spawn·goal 회전은 이번에 변경하지 않았다.

## 2026-09-29

### Paired placement · 담당자 HOLDING 상태 edge 입력

- 기존 4물체 paired placement의 과제·보상·RSI·AMP·가까운 시작 없음 설정을 유지한 별도 config·train/test/VNC·output을 추가했다. 새 관측 패킷은 edge당 `valid/src/dst/relation/owner/owner_holding_state` 6개 값이며, AT/ON_TOP에는 동일 owner·source의 선행 HOLDING `φ`, 다른 유효 edge에는 1을 넣는다. actor·critic의 기존 semantic 64차원 뒤에서 placement edge에만 상태 잔차를 더하고 checkpoint 계약을 분리했다.
- 관련 CPU 테스트 **32개 통과**, Python·셸 문법과 diff 확인. GPU 0의 기존 학습은 유지한 채 2048환경·1 iteration 별도 output 학습과 checkpoint 저장을 완료했다. TensorBoard scalar **137개 모두 finite**, AT+AT/ON_TOP+ON_TOP 샘플 비율 **0.501/0.499**였고 actor·critic 상태 분기 weight가 모두 0에서 갱신됐다. 저장 checkpoint로 각 1환경·32-step headless AT/ON_TOP 평가를 완료했다. 이는 실행 경로 검증이며 장기 수렴 결과는 아니다.

### 같은 과제 쌍 · 가까운 시작 없는 Stage 1 비교

- 기존 3물체 sampler에서 두 agent 모두 AT만 받는 config와, 새 4물체 sampler에서 scene별 AT+AT/ON_TOP+ON_TOP을 50/50으로 뽑는 config를 분리했다. 두 실험의 가까운 시작 확률은 0이며 34번의 RSI·AMP·보상식은 유지한다. 4물체 ON_TOP은 서로 다른 source 2개와 support 2개를 사용한다. 각 config에 train/test/VNC와 별도 output·checkpoint variant를 연결했다.
- 관련 CPU 테스트 **53개 통과**, 스크립트 문법·diff 확인. GPU 0에서 각각 2048환경·1 iteration scratch 학습과 checkpoint 저장을 완료했다. 물리 reset 실패는 둘 다 **0**, paired placement의 AT/ON_TOP 비율은 **0.509/0.491**이었다. 저장 checkpoint로 각각 1환경·32-step headless AT/ON_TOP 평가를 마쳤고, ON_TOP 평가에서 물리 상자 4개가 서로 다른 edge 역할에 연결됨을 확인했다. 장기 정책 학습 결과는 아직 확인하지 않았다.

## 2026-09-28

### 37번 ON_TOP putDown RSI 실험

- 36번을 유지하고 별도 config·train/test/VNC·output·checkpoint variant에서 ON_TOP `putDown` RSI 10%를 추가했다. 원본 후기 모션은 받침 상자와 겹치므로 45~70% 구간에서 시작하며, 받침을 모션 최종 XY에 배치하고 상자·사람 침투 reset을 재추첨한다. AT `putDown`과 36번 과제·보상 분포는 유지한다.
- 관련 CPU 테스트 31개, 셸·Python 문법·diff 검사 통과. GPU 0에서 2048환경·1 iteration 학습과 checkpoint 저장 완료: ON_TOP `putDown` RSI 비율 9.46%, 물리 reset 실패 0·재시도 98, scalar 138개 모두 finite. 저장 checkpoint의 1환경·32-step ON_TOP headless 평가와 JSON 저장도 완료했다. 이는 실행 경로 확인이며 장기 수렴 검증은 아니다.

### 36번 AT/ON_TOP 집중 학습 실험

- 34번의 reward·RSI·가까운 시작을 유지하고 HOLDING/SIT/CLIMB/HOLDING+AT/HOLDING+ON_TOP 샘플 비율만 0/0/0/50/50으로 바꾼 별도 Stage 1 config·train/test/VNC·output을 추가했다. 새 variant로 checkpoint 보상 계약을 분리했다.
- 관련 CPU 테스트 18개, 스크립트 문법·문서 링크·diff 검사 통과. GPU 0에서 2048환경·1 iteration 확인과 checkpoint 저장을 완료했다. AT/ON_TOP 샘플 비율 각각 0.5, 물리 reset 실패 0, scalar 137개 모두 finite였다. 저장 checkpoint의 1환경·32-step headless 평가와 JSON 저장도 완료했다. 이는 실행 경로 검증이며 장기 수렴 검증은 아니다.

### 뷰어 상자 색을 과제 배정에 연결

- Stage 1·2 로컬/VNC 뷰어에서 graph의 실제 대상 상자를 에이전트 색으로 칠한다. 공동 대상은 노란색, 비대상은 회색이다. 물리 상자 순서와 graph 배정이 달라도 색이 맞도록 변경했다.
- Stage 1 ON_TOP, Stage 2 place_climb/place_stack graph를 사용한 색 지정 호출 검증과 Python 문법 확인을 마쳤다. 실제 뷰어 화면 확인은 아직 하지 않았다.

## 2026-09-27

### Stage 2 SIT plane·34번 보상/독립 과제 변형

- 기존 29번과 분리한 `approach_stage2_coordination_sit_plane` config·train/test/VNC·output을 추가했다. SIT도 상판 안쪽 10%와 높이 오차 7cm로 판정하고, 자기 edge 합계·동료 edge 평균에 0.9/0.1 팀 보상을 적용한다. 에피소드는 place_climb/place_sit/place_stack/독립 20/20/40/20이며, 독립 과제 내부는 34번의 10/10/10/35/35와 RSI·가까운 시작 일정을 따른다. 협력 과제의 RSI는 기존 Stage 2 규칙을 유지한다.
- Stage 2 checkpoint 로드에서 reward config 불일치를 검사하도록 수정했다. 34번 epoch 500 checkpoint로 GPU 6에서 2048환경·1 iteration 이식 학습과 1환경 place_sit 평가를 완료했다. 물리 reset 실패 0, scalar 177개 모두 finite, 관찰·AMP 통계 이식 확인. 관련 CPU 테스트 22개 통과. 장기 수렴은 아직 확인하지 않았다.

### output_etc Git 제외

- `.gitignore`에 `/output_etc/`를 추가했다. `git check-ignore`로 내부 파일 제외와 `git status`에서 해당 폴더가 사라진 것을 확인했다.

### VNC 평가 결과 기본 경로 통일

- 현재 실험 9개의 VNC wrapper가 기본 `OUTPUT_PATH`를 각 실험의 `output/<실험명>`으로 전달하도록 바꿨다. 로컬 test와 같은 위치에 `metrics/`, `diagnostics/`가 생기며 학습 checkpoint의 run 디렉터리와 분리된다. 기존 `_vnc` 폴더와 실행 중인 학습은 변경하지 않았다. 셸 문법과 기본 경로를 확인했다.

### 34·35번 자기 보상 합계와 가까운 시작 일정 비교

- 33번의 AT reset 수정·state/progress/성공 식을 유지하고, 34번은 자기 edge 보상 합계·동료 edge 평균과 HOLDING/SIT/CLIMB/HOLDING+AT/HOLDING+ON_TOP 10/10/10/35/35 샘플링을 적용했다. 35번은 34번에서 가까운 시작 80%를 60만 per-env step까지 유지하고 120만 step까지 30%로 감소시킨다. 두 실험은 별도 config·train/test/VNC·output에서 scratch로 시작하며 33번과 기존 학습 프로세스는 변경하지 않았다.
- 관련 Stage 1·2 CPU 테스트 **30개 통과**, 스크립트 문법·diff 확인. GPU 6에서 각각 2048환경·1 iteration 학습과 checkpoint 저장을 완료했다. 두 실험 모두 물리 reset 실패 **0**, 공유 주 물체 **0**, scalar **177개 모두 finite**였고, 샘플 비율은 목표값에 근접했다. 짧은 진단 표본에서 AT 두 번째 목표 slot 높이도 정상이다. 저장 checkpoint의 1환경·32-step AT headless 평가와 JSON 생성도 각각 완료했다. 이는 실행 경로 검증이며 장기 수렴 검증은 아니다.

### 33번 AT 목표 reset 수정

- 독립 graph의 AT가 두 번째 목표 slot을 지정해도 reset이 첫 번째 slot에 목표를 배치하던 선택 오류를 수정했다. 32번의 보상·샘플링·RSI를 유지한 33번 config와 전용 train/test/VNC·output을 추가했다. 공통 reset 수정은 이후 새로 시작하는 27~32번에도 적용되며, 실행 중인 프로세스는 변경하지 않았다.
- 관련 CPU 테스트 **23개 통과**. GPU 6에서 2048환경·1 iteration scratch 학습과 checkpoint 저장 완료. 진단 CSV의 AT 목표 높이는 0보다 컸고, TensorBoard scalar **177개 모두 finite**, 공유 주 물체 **0**이었다. 이는 AT 목표 배치와 실행 경로 검증이며 정책 수렴 검증은 아니다.

## 2026-09-25

### 과거 실험 진입점 정리

- 실행 중인 27·28·30·32번과 Stage 2 29번, 비교용 31번 및 원본 1번을 남겼다. Git에 있던 9~26번 config 18개·전용 실행 스크립트 50개·전용 테스트 11개·오래된 설계 문서 10개·미사용 graph 예제 2개·미사용 학습 smoke config 1개를 제거했다. 현재 문서를 다시 작성하고 25·26번 전용 sampler 분기를 제거했다. 공유 graph·reward 함수는 현재 실험도 사용하므로 유지했다.
- 기존 학습 프로세스와 output은 변경하지 않았다. 삭제된 config·스크립트는 Git 이력에서 확인할 수 있다.
- 남긴 테스트 전체 **109개 통과**, 실행 스크립트 문법·Markdown 로컬 링크·diff 확인. 새 코드로 32번 2048환경·1 iteration 확인에서 checkpoint 저장, 물리 reset 실패 **0**·공유 주 물체 **0**, scalar **177개 모두 finite**였다. 이는 실행 경로 검증이며 장기 수렴 검증은 아니다.

### 32번 보상식 유지 커리큘럼

- 31번의 가까운 AT/ON_TOP 시작·후반 CLIMB RSI·과제/RSI 비율을 유지하고, AT/ON_TOP progress와 CLIMB state만 30번 식으로 되돌린 별도 schema 9 variant와 train/test/VNC를 추가했다. 31번 설정과 checkpoint는 그대로 두며 32번은 scratch로 시작한다.
- 관련 CPU 테스트 **17개 통과**. 2048환경·`MAX_ITERATIONS=1` GPU 1 확인에서 checkpoint 저장, 물리 reset 실패 **0**·재시도 **135회**, scalar **177개 모두 finite**, 템플릿 비율 약 **0.095/0.105/0.298/0.255/0.247**을 확인했다. 저장 checkpoint의 1환경·20-step ON_TOP headless 평가와 JSON 생성도 완료했다. 이는 실행 경로 검증이며 정책 수렴 검증은 아니다.

### 30번 SIT plane 정규화·31번 hard-skill 시작 상태 실험

- 28번을 보존하고 새 schema 9 variant 두 개를 분리했다. 30번은 SIT 성공을 root XY의 상자 윗면 안쪽 10%·목표 Z ±7cm로 바꾸고 agent별 active-edge 보상을 평균내 단일/두 edge 최대 task reward를 0.6으로 맞춘다. 31번은 AT·ON_TOP 가까운 시작 비율을 환경 step 0→300,000에서 80→30%로 낮추고, 장거리 progress·CLIMB root/발 높이 state·hard-skill sampling을 추가한다. 두 실험 모두 scratch이며 팀 공유 0.9/0.1은 유지한다.
- 기존 plane CSV 표본에서 AT/ON_TOP은 에피소드 첫 관측 3.33/4.27m에서 마지막 관측 5.70/6.36m로 목표에서 멀어졌다. CLIMB reference 7개·505프레임을 상자 높이 0.4m·XY 0.5m로 실측한 결과 성공 가능 29프레임 전부 원본 RSI 제외 구간에 있었다. 31번은 CLIMB RSI의 70%를 후반 65~98% 구간에 배정하며 기존 데이터셋과 27·28번 학습을 변경하지 않는다.
- 관련 CPU **34개 통과**, 스크립트 문법·diff 확인. GPU 1에서 두 실험 각각 2048환경·1 iteration scratch와 checkpoint 저장, 저장 checkpoint의 1환경·32-step headless 평가를 완료했다. 30번 물리 리셋 실패 0·재시도 121회, 31번 실패 0·재시도 156회, 각 scalar 177개 finite였다. 31번 CLIMB root/발 조건 동시 통과율은 후반 RSI 이전 짧은 확인 0%에서 1.26%로 바뀌었다. 이는 초기 상태·실행 경로 확인이며 장기 정책 수렴 검증은 아니다.

### 29번 Stage 2 coordination 전이 실험

- 28번 plane 성공 조건과 사용자 지정 전역 팀 보상 0.9/0.1을 유지하고, `place_climb/place_sit/place_stack` 협력 90%와 독립 bundle 10%를 샘플하는 schema 10 config·train/test/VNC를 추가했다. Stage 1 schema 9 semantic `.pth`를 `STAGE1_CHECKPOINT`로 선택해 actor/critic/AMP weight와 actor·AMP 관측 통계를 엄격히 복사하고, actor encoder·관측 RMS를 고정했다. 새 grounded-edge cross-attention과 Stage 1 head의 `[W,0]` 확장으로 협력 입력을 연결했다. Stage 2 resume는 별도 `RESUME_CHECKPOINT`로 분리했다.
- 공유 물체 SIT/CLIMB agent는 loco에서 시작하고, downstream 성공 또는 전체 graph 성공으로 시작한 reset은 재시도한다. 3인 공유 물체·4인 독립 쌍의 explicit 평가 graph와 가변 agent/edge 입력을 추가했다. Stage 2 taxonomy의 CLIMB 누락과 평가 시 학습 graph 동일성 검사를 수정했다.
- CPU 전체 **278개 통과**, 2048환경·1 iteration GPU 학습 및 checkpoint 저장 완료. 저장 checkpoint의 첫 head 협력 입력 weight norm은 **0.0556**, TensorBoard scalar **162개 모두 finite**, 물리 reset 실패 **0**이었다. 실제 Stage 1 plane checkpoint에서 전이 직후 동일 입력의 action mean·sigma 최대 차이는 각각 **0**이었다. 같은 Stage 2 checkpoint로 2·3·4인 1환경·20-step headless 평가와 JSON 저장을 완료했다. 2048환경 Stage 2 resume 뒤 협력 입력 weight norm은 **0.0817**로 변했고 attention weight도 갱신됐으며 actor encoder 75개 tensor와 actor RMS는 동일했다. Resume scalar **174개 모두 finite**, 물리 reset 실패 **0**이었다. 이는 실행·가변 입력 경로 검증이며 협력 성공률·장기 수렴 검증은 아니다.

## 2026-09-24

### 28번 stage1_plane 독립 시나리오 실험

- 27번의 독립 물체·goal 배정, 다섯 행동 각 0.2, RSI·AMP와 팀 보상 0.9/0.1을 유지했다. CLIMB·ON_TOP의 current-success만 원격 `ontop_climb_plane` 방식으로 변경했다. ON_TOP은 source 중심 XY의 support 윗면 안쪽 10% 영역 및 면 높이 오차 1mm, CLIMB은 root XY의 안쪽 5% 영역 및 root 높이 오차 20cm·평균 발 높이 오차 7cm를 요구한다. 연속 state·progress 보상은 기존 중심점 기준이며, 새 reward variant·config·train/test/VNC·output과 checkpoint 계약을 분리했다.
- CPU 전체 **274개 통과**, shell 문법·diff 공백 확인. GPU 3·2048환경·1 iteration scratch에서 다섯 template 비율 **0.194~0.205**, shared primary object **0**, 물리 reset 실패 **0**, TensorBoard scalar **174개 모두 finite**, checkpoint 저장을 확인했다. 저장 checkpoint로 1환경·32-step ON_TOP headless 평가와 JSON 저장도 종료했다. 이는 실행 경로 확인이며 장기 학습 수렴 검증은 아니다.
- GPU 3 본학습 로그 점검에서 CSV의 `self_task_contribution`/`other_task_contribution`이 Stage 1 보상을 자기 100%로 잘못 표시하는 문제를 발견해 config의 0.9/0.1 비율을 따르도록 수정했다. 실제 학습 reward와 TensorBoard 공유 지표는 이미 0.9/0.1로 정상이며, 실행 중 프로세스는 재시작하지 않아 해당 프로세스의 CSV 표시는 이전 방식으로 남는다. 관련 CPU 테스트 **17개 통과**·diff 공백 확인.

### 27번 독립 물체 binding + standalone CLIMB scratch 실험

- 26번의 5-template reward·0.9/0.1 팀 보상·AMP·template RSI를 유지하고, 물체 3개를 reset마다 무작위 순열로 두 agent의 서로 다른 primary object와 남은 ON_TOP support에 배정한다. AT goal도 비공유 무작위 배정하며 동시 ON_TOP은 제외한다. 유효한 template pair의 각 agent 주변 확률은 기존 0.2를 유지한다. 새 sampler/config/train/test/VNC/output과 별도 checkpoint variant를 연결했다.
- CPU 전체 **270개 통과**. GPU 2·2048환경·1 iteration 확인에서 5-template 비율 **0.196~0.203**, shared primary object **0**, 물리 reset 실패 **0**, TensorBoard scalar **172개 모두 finite**, GTA position p95 **4.4012m**, checkpoint 저장을 확인했다. 저장 checkpoint로 1환경·32-step ON_TOP headless 평가와 JSON 저장도 종료했다. 이는 실행 경로 검증이며 장기 수렴·물체 수 확대 추론은 미검증이다.
