# Changelog

최신 변경부터 기록한다. 현재 실행법은 [config.md](markdowns/config.md), 코드 위치는 [structure.md](markdowns/structure.md)를 참조한다. 실행 중인 GPU/PID는 이 파일에 고정하지 않고 실제 프로세스로 확인한다.

## 2026-10-09

### Mixed80 LocoAMP 공동 방향 보상 연결

- 사용자 합의대로 `approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding` env/train 및 train/test/VNC를 분리했다. 공동 HOLDING 중 목표 XY에서 가장 먼 사람의 heading 점수×0.1을 두 사람에게 각각 한 번 공유한다. 방향 포화 거리는 목표 XY 반대각선+0.8m(AT0.8m·ON_TOP약1.277m)이며 단독 환경에는 적용하지 않는다. 기존 placement progress buffer0.1·AMP80/10/10·RSI·CPA·CA/SA 셔플·동결 및 head 전이는 유지한다.
- 방향 reward 계약을 checkpoint metadata로 분리하고 `reward_terms/joint_alignment`와 공동 점수/포화 비율을 기록한다. 관련 CPU28개·Python/셸 문법·diff 검사를 통과했다. GPU5·2048환경에서 Stage1 epoch23000 전이→epoch2 저장→epoch3 재개를 확인했고 encoder69개 tensor/RMS 동결·모델 및 scalar136종 finite를 검사했다.
- 저장 모델의64환경32-step 혼합 평가(공동 AT27·ON_TOP24·단독13)에서 두 사람의 동일 방향 보상·task total에 한 번만 추가·단독/gate-off 보상0·성분 합과 total 일치를 확인했다. 전용 test wrapper의16환경32-step headless 평가도 통과했다. 실제 VNC 화면·장기 학습 개선은 미검증이다. 결과는 `output/approach_stage2_joint_carry_mixed80_locoamp_align_task_embedding_check/saved_check.json` 및 `output_etc/joint_alignment_check/{runtime_check,resume_check}.json`이다.

### 공동운반 clone 배포 준비·외부 데이터 자동 연결

- 현재 보정 AMP12개·검사용 target·TeamHOI 원본12개·paired RSI6개를 `.gitignore` 예외로 지정했다. 영상·이전 보정 binary·checkpoint는 제외한다. `joint_carry/README.md`의 연결 전 전달 안내를 현재 clone 실행법으로 교체하고 config·structure·포팅 문서를 갱신했다.
- `prepare_joint_carry.py`가 `TOKENHSI_DATA_ROOT` 또는 `--data-root`의 기존 TokenHSI 데이터를 없거나 끊어진 링크에 연결한다. 유효한 기존 데이터는 덮어쓰지 않는다. 두 LocoAMP train/test(VNC 포함)에 원본 파일100개 존재·현재 AMP/RSI18개 SHA256 검사를 연결했다. 보상·모델·분포·기존 학습 프로세스는 변경하지 않았다.
- 별도 임시 Git index로 배포 대상만 archive하여 공백 포함 새 경로에 복원했다. AMP12개·RSI6개 포함, 공동 RSI319개 로드·원본12개 해시·경로/파일 보존/오염 검출 CPU4개·셸 문법·diff 검사를 통과했다. 실제 index는 변경하지 않았으며 commit/push는 수행하지 않았다.
- 새 경로·GPU5·2048환경에서 Stage1 epoch23000으로 두 모델의 짧은 학습을 완료했다(`MAX_ITERATIONS=1`, 실제 저장 epoch2). CA는 RSI 캐시2279 profile을 새로 생성했고 head-only는 데이터 경로 재지정 없이 링크·캐시를 재사용했다. 두 모델 모두 actor69개 tensor 원본 유지·모델/기록 scalar133종 finite·정상 종료를 확인했다. 결과: `output/clone_portability_{ca,head_only}_check/portability_check.json`. 동일 서버의 경로 이관 검사이며 다른 서버의 드라이버/환경까지 검증한 것은 아니다.

### output 검증 산출물 보관 정리

- 사용자 요청으로 비활성 검증·분석·이전 영상 디렉터리19개를 `output_etc/`로 이동했다. `output/`에는 본학습 결과4개·`rsi_cache`·현재 전체 보정 영상/검사만 남겼다. 이동 목록은 `output_etc/archive_20261009_cleanup.json`이다.
- 같은 파일시스템의 rename으로 디렉터리 inode를 보존했고 실행 중 프로세스가 이동 경로를 사용하지 않는지 확인했다. 문서·이전 데이터 검증 report 경로·보조 영상/검사 스크립트 출력 경로를 갱신했다. 현재 학습 config·AMP 데이터·캐시·checkpoint는 변경하지 않았다.
- 이동19개·문서 링크·데이터 검증 report4개·보조 스크립트 Python 문법·diff 확인 통과. 두 본학습 프로세스는 이동 후에도 실행 중이다.

### 자세 보존 보행 전체12개·LocoAMP CA/head-only 연결

- 승인받은 자세 보존 방식을 후진9개·옆걸음3개에 적용해 `joint_carry/teamhoi_retarget_posture_all/`로 분리했다. IK 범위 안쪽0.001rad 여유로 quaternion 보간의 작은 한계 초과를 제거했다. 무릎 중앙값의 원본 대비 변화는 최대3.1°이며 원본·이전 보정본 해시는 보존했다. Male2 B9/B11/B13/B15만 중간 샘플을 추가해60fps, 총4458프레임이다.
- GPU5·2048환경에서4458/4458프레임의0.1초 초기 물리 검사·프레임/중간 시점 관절 범위·발 접지·FK 검사를 통과했다. 시뮬 body/FK 차이 최대0.12mm 미만, CPU/GPU 실제 AMP3200×1320 finite. 개별4열 비교12개·후진/옆걸음 모아보기2개 전체 MP4 decode 통과. 결과는 `output/teamhoi_retarget_posture_all_check/`다. 연속 보행 안정성·미끄러짐0·학습 개선을 보장하는 검사는 아니다.
- 두 `approach_stage2_joint_carry_mixed80_locoamp_{task_embedding,head_only}` env/train·train/test/VNC를 새 데이터 해시로 연결했다. AMP만 기존80%·후진10%·옆걸음10%로 보강하며 사람별 방향 배정은 없다. 공동80/단독20, 기존 reward·CPA·RSI·SA 셔플은 유지한다. Stage1 actor/RMS 동결, CA는128입력 `[W,0]`, head-only는CA 모듈 없이64입력 `W`로 초기화하고 action head를 학습한다. Critic·AMP 판별자는 학습하며 이전 보정 데이터 checkpoint는 계약 불일치로 구분한다.
- 관련 CPU18개 및 최종 데이터 계약4개 재검사, Python/전용 셸6개 문법·diff·문서 링크 검사를 통과했다. GPU5·2048환경에서 두 모델 각각 epoch2 저장, epoch3 재개, 16환경32-step headless 평가 로드를 확인했다. Actor69개 tensor·관찰 RMS 불변, head64/128·CA 유무·head/critic/discriminator 갱신과 scalar finite를 검사했다. 확인용 결과는 각 `output_etc/approach_stage2_joint_carry_mixed80_locoamp_*_posture_{check,resume_check,eval_check}/`다.
- 실제 AMP30000개 샘플의 원래/후진/옆걸음 비율은79.68/10.25/10.07%, 두 새 라이브러리와 원래 RSI 라이브러리 분리·carry label·시간 범위·공동1638/단독410·부분 reset1024/32/32의 미선택 상태 보존·8 control step finite를 확인했다. `head_only_posture_check/reset_check.json` 참조. 본학습은 시작하지 않았다.

### TeamHOI 과도한 무릎 굽힘 수정 pilot

- 사용자 영상 검토로 이전 IK의 자세 왜곡을 확인했다. 골반 고정 상태에서 발 목표를 맞춘 뒤 접지하던 순서가 원본보다 과도한 굽힘을 만들었다. 이전 전체12개 보정본과 연결된 LocoAMP 본학습은 사용 보류로 문서화했다. 기존 학습 프로세스·AMP/RSI/config 데이터 해시는 바꾸지 않았다.
- `--posture-first`로 별도2개를 생성한다. 골반과 발 목표의 공통 높이를 IK 전에 정렬하고 원본 무릎 자세 유지 항을 강화했다. 기존 loco5개를 대조했으며 무릎 중앙값은 후진 원본9.5°→이전60.5°→새11.0°, 옆걸음22.3°→56.8°→20.3°다. XY 경로·root 방향·원본 해시를 유지한다.
- GPU5·2048환경의0.1초 초기 충격 검사는253/253·530/530, CPU/GPU 실제 AMP640×1320 finite, 관절 범위·프레임/중간 시점 발 접지·FK/시뮬 body 일치 검사를 통과했다. 실제 FK 차이0.12mm 미만, 최저 발 여유 약0.4–1.2cm. Python 문법·diff·두 MP4 전체 decode 검사도 통과했다.
- **한계:** 후진의 바닥3.5cm 이내 발 표면 속도 proxy 평균은0.42m/s(이전0.22), 옆걸음0.11m/s다. 이는 접촉 미끄러짐 측정이 아니며 자연스러움·미끄러짐0·폐루프 보행·학습 효과는 미검증이다. 발 목표 오차는 최대3.1cm를 허용한 자세 보존 결과다. 새 데이터는 학습 미연결이며 나머지10개에 아직 적용하지 않았다.
- 결과: `output_etc/teamhoi_retarget_posture_pilot_check/{report,gait_comparison}.json`, `backward.mp4`, `sideways.mp4`. 영상은 원본/이전/새 보정/기존 걷기의4열 기구학 재생이며 기존 걷기는 별도 클립 반복이다. 재현 스크립트는 기존3개에 `--posture-first`, 자세 비교는 `joint_carry/scripts/compare_teamhoi_gait.py`다.

### TeamHOI 후진·옆걸음 각1개 보정 및 GPU 검증

- 사용자 승인으로 Male1 B10 후진·CMU141_33 옆걸음을 `teamhoi_retarget_pilot/`에 별도 변환했다. 현재 asset의 관절 범위 내 다리 IK, 이동하는 발의 지면 여유, 부드러운 지지 발 높이 기반 root Z 보정을 적용했다. Root XY/방향은 float32 정밀도 내 보존하며 실제 체형으로 FK·선속도·각속도를 재계산했다. 원본 SHA256은 불변이다.
- 실제 10-frame carry AMP 경로 CPU/GPU·1320차원·finite, 프레임/중간 시점의 관절 범위·발 mesh 바닥 관통·속도와 경로 정합성을 검사했다. GPU5·2048환경·GPU PhysX에서 후진253/253·옆걸음530/530프레임의0.1초 초기 충격 검사 통과, 실제 simulator body와 FK 최대 차이0.12mm 미만. 낮은 쪽 발의 바닥 여유는 약0.4–1.3cm다.
- **한계:** 바닥 근처 발 표면 속도 proxy 평균은 후진0.22m/s·옆걸음0.10m/s로 미끄러짐0을 보장하지 않는다. 8초 자유 root PD는 기존 모션 대조군도 넘어지므로 학습 가능성 판정에서 제외한다. AMP 학습 개선·폐루프 모방·joint RSI는 검증하지 않았다. 나머지10개·학습 config·실행 중 학습은 변경하지 않았다.
- 재현: `joint_carry/scripts/{retarget_teamhoi_reference,check_teamhoi_retarget,preview_teamhoi_retarget}.py`. 결과와 원본/보정 기구학 비교 영상은 `output_etc/teamhoi_retarget_pilot_check/`의 `report.json`, `backward.mp4`, `sideways.mp4`. 보정본은 AMP에 아직 연결하지 않았다.

### TeamHOI reference 원본 호환성·GPU 물리 검사

- 후진9개·옆걸음3개 원본은 기존 MotionLib/32-DOF 및 실제 10-frame carry AMP 생성 경로에서 로드됐다(384×1320, finite). 다만 원본 다리 skeleton offset이 현재 asset과 달라 동일 DOF의 발 위치가 클립 평균5.4–7.6cm 어긋나고 실제 발 mesh 기준 최대10.2–13.6cm 바닥 관통이 계산됐다. 기존 reference 대조군은 발 위치 평균 오차0.01cm 미만이었다.
- GPU5·2048환경·실제 phys_humanoid_v3·GPU PhysX에서 모든3721프레임을 0.1초 검사했다. root 속도 변화≤3m/s·변위≤0.3m 기준2064/3721(55.5%) 통과, 기존 모션 대조군238/238 통과. 8초 자유 root PD 재생은 대조군도 넘어져 AMP 사용성 판정 지표에서 제외했다. 최종 root/DOF는 모두 finite였다.
- 결과/재현 스크립트: `output_etc/teamhoi_reference_compatibility_20261009/`의 `report.json`, `amp_history.json`, `check.py`. 원본 즉시 투입은 권장하지 않으며 현재 체형으로 재타게팅·접지/속도 정합성 보정 후 재검증이 필요하다. 학습 config·RSI·실행 중 학습은 변경하지 않았고 변환 및 학습 성능 검증은 미수행이다.

### TeamHOI 후진·옆걸음 reference 다운로드·미리보기

- 사용자 요청으로 공식 TeamHOI commit `6fdc9885f8a9c82854be99adca6f2570e67ac63a`의 `near_table.yaml`에 포함된 후진9개·옆걸음3개를 `joint_carry/teamhoi_reference/`에 다운로드했다. 원본 경로·SHA256·프레임 정보는 `manifest.json`, 출처 설정·라이선스도 함께 보존했다.
- 12개 모션의 finite 배열·단위 quaternion·SHA256을 확인하고, 원본 FK·몸 방향/이동 방향 화살표를 표시하는 CPU 재생 스크립트와 두 MP4를 만들었다. 이는 reference skeleton 미리보기이며 물리 재생·현재 32-DOF로의 전환·AMP 학습 연결은 수행하지 않았다.

### 공유 목표 색상·과제별 마커 통일

- 사용자 지정 `stage2_team` shared9 시각화를 참고해 공유 물체·받침·AT 목표와 공동 과제 마커를 노란색으로 통일했다. AT 목표를 마지막 owner 색으로 덮어쓰던 경로를 제거하고 ON_TOP 와이어 큐브·SIT 수평 원·CLIMB 3축 십자를 적용했다.
- 실제 joint AT/ON_TOP graph와 렌더 API 대역을 사용한 CPU 색상 검사, 단일 graph 소유자 집계·마커 정점·Python 문법·diff 검사를 통과했다. 실제 VNC 화면은 미확인이고, 실행 중 뷰어에는 재시작 후 반영된다.

## 2026-10-08

### BEFORE 80%·독립 20% task CA 연결

- GPU5 기본의 `approach_stage2_before_task_embedding` env/train YAML·train/test/VNC를 추가했다. 현재 AT prerequisite으로 B state/success·포화를 gate하고 progress/HOLDING은 유지한다. 공통27개 크기·세 BEFORE 각546/독립410환경·Stage 1 RSI/AMP·SA/CA token 셔플을 연결했다. Carry-only source/기존 joint Stage 2 checkpoint와 구분한다.
- 공유 상자 상태는 A가 한 번만 결정하며 B reference·속도·AMP history를 정렬한다. 실제 source/support 크기로 RSI를 조회하고 geometry 재시도에서 skill을 유지한다. 부분 reset의 상태 보존과 평가 skill별 과제 적합성도 확인했다.
- CPU94개·셸/diff 검사, GPU5·2048환경 짧은 학습/epoch2 저장·epoch3 재개 통과. Actor69개 tensor·관찰 RMS6개 불변, NONE/SELF/BEFORE 갱신·COUPLED0, scalar182종/364값 finite. AMP 대응·1024개 혼합 및 각32개 그룹 부분 reset·16환경32-step headless 평가 경로를 확인했다. 결과: `output/approach_stage2_before_task_embedding_check_final/{saved_check,reset_check,resume_check}.json`.
- **미완료:** 초기 자세0.1초 물리 진단은1575/2048 통과·473개 미통과다. geometry reset 실패0과 별개이며 합성 RSI 물리 안정성은 보완이 필요하다. 강화 캐시는 fallback 증가와 잔여 불안정 때문에 적용하지 않았고 기존 Stage 1 비율/선별 기준을 유지했다. 본학습·장기 성능·VNC 화면은 실행/검증하지 않았다.


### 공동 80%·단독 20% Mixed80 실험 분리

- 사용자 승인으로 `approach_stage2_joint_carry_mixed80_task_embedding` env/train config·train/test/VNC를 추가했다. 2048환경 중 공동1638·단독410을 생성 시 고정하며 공동 AT/ON_TOP은 scene별50/50, 단독은 agent별 독립50/50이다. 기존100% 공동운반 실행은 유지했다.
- 단독은 Stage 1 carry 상자/받침 크기·크기별 물리 선별 RSI(40/10/40/10)·reference AMP history를 재사용한다. 공동 paired RSI·반복 AMP history와 reset을 단일 물리 commit으로 결합했다. 공동 SELF/COUPLED와 단독 SELF/NONE을 기존 task CA에 연결하고, 단독 placement는 자기 현재 HOLDING으로 gate한다. 새 reward 계약으로 기존 Stage 2 checkpoint와 구분한다.
- 관련 CPU **64개**, Python/셸 문법·새 문서 경로·diff 검사 통과. GPU5·2048환경·MAX_ITERATIONS=1 학습은 epoch2 저장, scalar133종/266값 finite, 물리 reset 실패0이었다. 지정 Stage 1 epoch11000 대비 actor69개 tensor·관찰 RMS 불변, 세 CA bias 갱신을 확인했다.
- 실제1638/410 분할·서로 다른 payload/goal의 SELF/NONE·공동/단독 AMP 이력·carry demo label·1024환경 혼합 및 각32환경 단일그룹 부분 reset에서 미선택 상태/관측/AMP 보존을 확인했다. 11 control step finite. 결과는 `output/approach_stage2_joint_carry_mixed80_task_embedding_check/{saved_check,reset_check}.json`이다.
- 저장 모델의 단독 AT/ON_TOP 혼합과 공동 ON_TOP을 각각16환경·32-step headless 평가해 전용 test 실행/로드 경로를 확인했다. 본학습·단독 능력 유지 및 공동운반 장기 성능·VNC 화면은 검증하지 않았다.

### 지정 carry distill epoch11000의 Stage 2 전이 허용

- 사용자 지정 `stage1/ApproachScenarioStage1UnifiedSizeRsiTaskEmbeddingCarryDistill_00011000.pth`를 확인하고, 호환되는 carry-only distill variant를 Stage 2 source 허용 목록에 추가했다. Tensor shape·키 검증은 유지하고 실행 가이드의 source를 갱신했다.
- 실제 checkpoint epoch11000·157개 tensor 전이·69개 actor 파라미터 동결·finite action을 CPU에서 확인했다. 관련 CPU 10개와 diff 검사 통과. 학습은 실행하지 않았다.

## 2026-10-07

### Joint carry collision을 기존 shared9 CPA로 변경

- 사용자 지정 `stage2_team/output/approach_stage2_rescue_shared9_cpa_team03`의 저장 config와 `collision_reward.py`를 대조해, root XY 접근 방향 × CPA 거리 위험도 × `0.99^(t/control_dt)`를 적용했다. 기존 계수0.5·거리0.7m를 유지하며 정지/동일속도/멀어지는 쌍에는0, static 거리 항은 중복 가산하지 않는다. 공동운반 외 기존 실험 경로는 유지했다.
- CPA mode/discount를 joint reward 계약에 기록했다. 이전 distance Stage 2 checkpoint의 직접 resume/eval은 거부하며 Stage 1 source 전이는 유지한다. 실행 가이드·코드 역할 문서를 갱신했다.
- 관련 CPU 24개 통과(접근/이탈/스침/정지·제어 dt 감쇠·순열·checkpoint 및 기존 Stage 2 회귀). GPU 5·2048환경에서 실제 reward 경로의 네 사례와 8 control step finite/bounds, 초기 물리·AMP·부분 reset을 확인했다. 원본 CPA와 1024개 4-agent 무작위 배치의 최대 차이는1.35e-7 미만이다. 결과는 `output/approach_stage2_joint_carry_cpa_check/reset_check.json`; 본학습은 실행하지 않았다.

### Joint carry ON_TOP 받침 크기 축소

- 사용자 요청으로 받침을 72×100×30cm에서 운반 상자와 가로·세로가 같은 **52×80×30cm**로 변경했다. Env YAML·크기 계약·실행 가이드를 동기화했다.
- 관련 CPU 6개 통과. GPU 5·실제 Stage 2 2048환경에서 새 asset 크기·0.1초 초기 물리 검사·AMP 초기 이력·1024환경 부분 reset을 확인했다. 결과는 `output/approach_stage2_joint_carry_support_size_check/reset_check.json`이다. 실제 올려놓기 성능은 이번 검사에 포함하지 않는다.

### Joint carry Stage 2 실행 연결

- 사용자 승인으로 `approach_stage2_joint_carry_task_embedding` env/train YAML·train/test/VNC를 연결했다. 원본 task embedding 기반 actor/RMS 동결, scene별 joint AT/ON_TOP 50/50, 2명·4물체, 52×80×40cm payload·72×100×30cm 받침을 쓴다. 기존 Stage 1·29번과 config/output/checkpoint를 분리했다.
- 양손 중점의 object-local Y ±25cm 앵커 중 자유 선택과 반대 앵커 동시 HOLDING gate를 구현했다. 자기 HOLDING은 항상 보상하고 placement state/progress/success만 gate한다. 포화·성공 latch는 제거했다. CA의 task packet/KV/relation bias를 함께 셔플하고 SA 복원 순서와 action 대응을 유지한다.
- 보정 snapshot의 두 사람·공유 상자·root/DOF 속도를 같은 행에서 공동 복원하며 부분 reset은 단일 root commit을 쓴다. RSI는 AT 40/10/40/10, ON_TOP 50/10/40/0(loco/pickUp/carryWith/putDown)이다. AMP는 기존 single-human carry 전문가만 유지하고 reset 상태 반복 이력·carry label·demo/replay matching을 연결했다. 평가 skill 축소가 expert 인덱스를 바꾸지 않게 했다.
- 관련 CPU **59개 통과**: 보상 gate/비포화·앵커 대칭/회전·RSI 원본 대응/속도·config/checkpoint 격리·SA/CA 순열 출력 및 gradient·reload·기존 Stage 1/2 회귀. GPU 5·2048환경·MAX_ITERATIONS=1 학습은 epoch2 저장, 131종 scalar/262값 finite. Actor69개 tensor/RMS 불변, CA bias·확장 head·critic·AMP 갱신을 확인했다.
- 실제 Stage 2 reset의 2048환경 0.1초 물리 검사(loco905/pickUp217/carryWith832/putDown94) 모두 통과. 공유 binding·속도·AMP 초기 이력·carry demo label·1024개 부분 reset의 미선택 상태/관측/history 보존을 확인했다. 학습 재개와 4가지 16환경·32-step headless 평가도 완료했다.
- 결과는 `output/approach_stage2_joint_carry_task_embedding_check/`의 `saved_check.json`, `reset_check.json`, `resume_check.json`과 `_check_joint_carry_*` 평가 폴더에 있다. 임시 미학습 source의 연결 검사로, 네 평가의 운반 완수율은0이다. 본학습·VNC 화면·장기 공동운반·실제 하중 분담·완료 후 손 놓기는 검증하지 않았다.


### Joint carry GPU 5 RSI 검사·TeamHOI AMP 확인

- 사용자 요청으로 `output/joint_carry_gpu5_check_20261007/check_gpu.py`에서 GPU 5·GPU PhysX/GPU pipeline·2048환경을 검사했다. 52×80×40cm·밀도100의 공유 상자와 두 사람을 같은 snapshot에서 복원하고 공통 yaw 0/90/180/270°를 적용했다. 기존 보정본은 변경하지 않았다.
- pickUp/carryWith/putDown 55/129/135개 snapshot 전부 포함, 반복 배치 385/853/810환경 모두 0.1초 초기 충격 기준 통과. 부분 reset 1024환경도 통과, root/DOF 위치 복사 오차0·미선택 root/DOF tensor 보존을 확인했다. 최대 root 속도 변화1.097m/s·변위0.195m, 상자 속도1.913m/s·변위0.172m 미만이다. 결과는 `report.json`, 환경별 값은 `per_env.npz`, 실행 로그는 `run.log`에 있다.
- 이는 별도 2인/1상자 물리 검사이며 Stage 2 reset/AMP 연결·장기 공동 운반 검증은 아니다. 손 접촉 진단은 손-상자 근접+net force의 대용 지표로, 물체별 접촉력 판정과 구분한다.
- TeamHOI 공식 코드와 로컬 사본에서 single-human locomotion/pickup reference, 근거리 팔/손 AMP masking·두 discriminator 보상 혼합, 전원 양손 근접 조건의 운반 보상 gate를 확인했다. 공동 모션을 AMP expert로 즉시 넣지 않고 기존 carry expert를 우선 사용하는 안을 제안했으며, AMP 설정 자체는 변경하지 않았다.

### Joint carry RSI 재생성·물리 재검증

- 사용자 요청으로 원본 B19/B20/B21에서 복제·32-DOF 팔 IK·CPU PhysX 검사를 다시 실행했다. 결과는 `output/joint_carry_rsi_rebuild_20261007/`에 분리했다. 보정 전 통과 수 29/0/0개에서 보정 후 pickUp/carryWith/putDown 55/129/135개로, 총 319개 snapshot의 모든 배열이 기존 인계본과 동일하다.
- 입력 해시 12개·snapshot shape/finite/quaternion/reference 대응·선별 조건·ZIP 무결성을 확인했다. 기존 인계본 79개 파일·의존 파일 3개도 검사 통과했다. 근거는 `verification.json`과 `physics_after/report.json`이다.
- `corrected/before_after/`에 보정 전후 영상, `pd_videos/`에 reference 대 실제 PD 추종 영상을 생성했다. 0.1초 초기 상태 검사는 통과했으나 연속 PD 추종은 균형을 잃었다. 학습 reset·AMP 연결과 GPU 학습 검증은 수행하지 않았다.

### Joint carry task CA 연결 준비

- `joint_carry_spec.py`에 2명·4물체의 joint AT/ON_TOP scene별 50/50 sampler를 추가했다. `task_coordination.py`는 동결 task embedding과 최종 actor/payload/target 토큰을 256→128→64로 결합하고, 2-head CA에 NONE/SELF/COUPLED zero-init bias를 적용한다. 공유 물체→목표의 중복 Stage-1 bias는 Stage 2에서 한 번만 반영한다.
- Task embedding 전이에 원본·네 과제 distill source를 허용하고 기존 actor 동결·action head `[W,0]` 확장을 연결했다. 지정한 외부 distill epoch1000 checkpoint의 157개 tensor 전이·69개 actor 파라미터 동결·finite action 출력을 CPU에서 확인했다. 관련 CPU 22개 통과, diff 검사 통과.
- **미완료:** RSI 초기화·팀 보상 선택 답변 대기. 전용 env/train config·실행 스크립트·GPU 5 시뮬레이션 검증은 아직 완료하지 않았으며 실행 가능한 실험으로 등록하지 않았다.

### 네 과제 task embedding unified teacher distillation

- 사용자 요청으로 `approach_scenario_stage1_unified_size_rsi_task_embedding_distill` env/train YAML·train/test/VNC·output을 추가했다. 원래 네 과제 sit/climb/carry_at/carry_ontop 10/25/32.5/32.5%와 크기·RSI·AMP·보상·task embedding/type projection을 유지하고, 기존 unified teacher KL 경로를 연결했다. SIT→Sit, CLIMB→Climb, 두 carry→Carry로 지도하며 기본 계수는 0.001이다. Carry-only와 원본 config·checkpoint를 별도로 유지한다.
- 관련 CPU **56개 통과**: 원본 설정 동일성·checkpoint 격리·16개 task 쌍의 teacher 라우팅/goal/물체 binding·회전된 SIT 방향·네 task embedding gradient·teacher 동결·critic 비의존·AMP family 분포 및 carry/task embedding/RSI/unified 회귀. Python/셸 문법·문서 링크/명령 경로·diff 검사 통과.
- **GPU 5·MPS·2048환경·MAX_ITERATIONS=1** 검증 학습을 완료했다(epoch 2, frame 262144). 공통 RSI 캐시에 누락 profile 3,622개를 추가해 총 10,262개를 확보했다. Scalar **252종/504값 모두 finite**, 물리 reset 실패·단독 HOLDING label·공유 primary 물체 0, 진단 CSV 16행의 owner binding·자기/동료 보상 오류 0이다. 실제 네 task teacher label·KL actor/embedding gradient·teacher 동결·critic KL gradient 없음을 확인했다.
- 저장 학생으로 sit/sit RSI, climb/climb RSI, carry_at/carryWith, carry_ontop/carryWith 각각 **16환경·32-step headless 평가**와 JSON 저장을 완료했다. Output은 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_distill_check/`와 `_check_<task>/`다. 연결 검증이며 장기 성공률·VNC 화면은 미검증이다. 새 본학습을 시작하지 않았고 기존 carry 본학습은 유지했다.

### Carry-only task embedding unified teacher distillation

- 사용자 요청으로 `approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill` env/train YAML·train/test/VNC·output을 분리했다. Task/NONE/SELF embedding과 H/O/G 타입 쌍 projection은 유지하고 carry_at/carry_ontop만 agent별 50/50 독립 샘플링한다. 실제 source 크기도 carry 범위로 제한하고 크기별 과제 확률·AMP carry family 100%를 연결했다. RSI 40/10/40/10·물리 검사·자기 보상 합계·634-D packet은 유지한다.
- 보존된 `distill/`의 원본 unified teacher adapter와 rollout label·Gaussian KL 경로를 현재 PPO/AMP에 이식했다. 원본 `ckpt_stage1.pth`의 Carry 행동 분포를 동결해 계수 0.001로 지도하며 AT graph goal·ON_TOP 회전 bbox 목표·reset body·scene/agent minibatch 정렬을 따른다. 기존 AMP family matching·정규화와 네 과제 config는 유지했다. 새 variant로 checkpoint 혼용을 차단하며 학생 평가는 teacher 없이 실행한다.
- 관련 CPU **63개 통과**: 원본 teacher strict 로드, 네 carry 조합·goal/물체 binding·회전 높이·reset·label shuffle, KL 공식·teacher 동결·학생 embedding gradient·critic 비의존, carry 크기/AMP·기존 task embedding/Size RSI/unified 회귀. Python/셸 문법·새 문서 링크/명령 경로·diff 검사 통과.
- 지정한 **GPU 5·2048환경·MAX_ITERATIONS=1** 확인 학습을 완료했다(epoch 2, frame 262144). 최초 RSI 캐시 **6,640개 profile**을 생성했고 scalar **212종/424값 모두 finite**, 물리 reset 실패·단독 HOLDING/SIT/CLIMB label·공유 primary 물체는 0이다. 초기 두 update의 KL/PPO actor gradient 비율은 0.345/0.228이며 teacher 동결·critic KL gradient 없음·actor/projection/embedding 갱신을 확인했다. 진단 CSV 16행의 owner binding·자기/동료 보상 합산 오류는 0이다.
- 저장 학생으로 carry_at/carry_ontop + carryWith 각각 **16환경·32-step headless 평가**와 JSON 저장을 완료했다. Output은 `output/approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill_check/` 및 `_check_carry_at/`, `_check_carry_ontop/`이다. 두 짧은 평가의 운반 완수율은 0으로, 실행 경로만 확인한 결과다. 본학습·장기 성능·VNC 화면은 검증하지 않았고 기존 GPU 프로세스와 `distill/` 원본은 변경하지 않았다.

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
