# wait_penalty_v2 + BEFORE relation fusion / zero background

실험 이름: `stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_wait_penalty_v2_relfusion_zerobg`.

ksh 원본에서 복사한 설정과 구현을 로컬에 등록했다. `before_relation_fusion.py`와 네트워크 연결은 원본 동작을 유지하며, 기존 로컬 attention-order 진단은 보존했다. 출처와 SHA256은 `output_etc/check/relfusion_zerobg_port/source_manifest.json`에 있다. ksh 파일 및 프로세스는 수정하지 않는다.

## 모델

- BEFORE 후행 query→선행 key의 attention bias와, 후행 attention 출력의 잔차 MLP가 같은64차원 관계 embedding을 사용한다.
- embedding/MLP는4층 공유, bias projection은층/head별이다. Actor와Critic의 fusion 모듈은 별도다.
- SELF/NONE/미사용 COUPLED bias는0이다. 후행에만 fusion을 적용하며, task 성공 이후에도 관계가 남아 있으면 적용된다.
- 초기 fusion output 및 coordination residual을0으로 초기화하여 Stage1 전이 직후 출력은 기존과 동일하다. gradient가 흐르고 후속 학습에서 변하는지 검증한다.

## 보상과 실행

보상은 원본 wait_penalty_v2 그대로다: virtual box, 경계=박스XY반대각선+3m, 진입 후 root XYZ속력 벌점계수1, 선행 최초 AT 성공 시 벌점 해제, AT 높이 허용0.2m. **soft entry/asym/연속progress/추가 접근 early term은 없다.** 기존 낙상 종료와1200step 시간제한은 유지한다.

train/test/vnc/eval 전용 wrapper 및 [config 명령](../config.md)에 등록했다. 학습은 기본 Stage1 PUSH26k 전이, seed42,4명·1024환경·8객체다. 기존 soft-entry 학습은 이 설정과 별도이며 자동 교체하지 않는다.

가져온 초기 설정은 `tokenhsi/data/cfg/imported/relfusion_zerobg/`에 보존한다. 현재 실행용 설정은 `tokenhsi/data/cfg/multi_agent/` 및 `tokenhsi/data/cfg/train/rlg/`의 실험 이름 YAML이다. 로컬에 복사한 원본 checkpoint는 `pth/imported_relfusion_zerobg/`에 있다. 서로 다른 fusion 계약 checkpoint를 일반 wait_penalty_v2 모델로 로딩하지 않는다.

## 검증

CPU: 원본 fusion 테스트(초기 항등, 단방향 BEFORE, Actor/Critic 분리, virtual 입력, gradient, shuffle, strict reload) 및 기존 Stage2 모델·대기보상 회귀.
GPU: `output_etc/check/relfusion_zerobg_port/`에 짧은2 iteration 학습/체크포인트 및 로컬 원본 checkpoint 추론 검사 기록을 둔다. 이는 실행 가능성 검증으로 성능 평가나 본학습이 아니다.
