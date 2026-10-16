# 레포 구조

Human SA4와 SA4 GraspView에 기존 AT/AT20 총 네 설정을 제공한다. 실행법은 [config.md](config.md)에 있다.

| 경로 | 역할 |
| --- | --- |
| `tokenhsi/run.py`, `utils/parse_task.py` | HumanoidMACarry와 MA PPO/AMP 등록·실행 |
| `tokenhsi/data/cfg/multi_agent/stage2*.yaml` | 네 설정의 환경 YAML |
| `tokenhsi/data/cfg/train/rlg/stage2*.yaml` | 네 설정의 학습 YAML |
| `tokenhsi/scripts/multi_agent/stage2*_{train,test,vnc}.sh` | 전용 학습·로컬 평가·서버 VNC |
| `tokenhsi/scripts/multi_agent/runtime_env.sh`, `mps_auto_env.sh`, `run-gui.sh`, `mps/` | 공통 conda/GPU/MPS/VNC 실행 준비 |
| `tokenhsi/learning/multi_agent/amp_network_builder_ma.py` | Stage1과 호환되는 task embedding/GTA encoder와64입력 head |
| `tokenhsi/learning/multi_agent/human_coordination.py` | human SA4·SELF/COUPLED/NONE 관계·human 순열 |
| `tokenhsi/learning/multi_agent/grasp_view.py` | GraspView 관측·고정 앵커 배정·자기 human 선택 |
| `tokenhsi/learning/multi_agent/stage2_transfer.py`, `scene_normalizer.py` | CarryDistill Stage1 전이·동결·view별 RMS |
| `tokenhsi/learning/multi_agent/ma_agent.py`, `ma_players.py`, `edge_context_eval.py` | scene 단위 PPO/AMP 학습과 평가 |
| `tokenhsi/env/tasks/multi_agent/humanoid_ma_carry.py`, `edge_context_task.py`, `edge_ontop_task.py` | 관측·물리·공동/단독 reset·진단 |
| `tokenhsi/env/tasks/multi_agent/joint_carry_{reward,reset,platform}.py` | 반대 앵커 보상·CPA 해제·paired RSI·받침 |
| `tokenhsi/utils/mixed_carry_spec.py`, `joint_carry_spec.py`, `relation_task_spec.py` | 80/20 graph·실험 계약·체크포인트 검증 |
| `tokenhsi/utils/size_rsi.py`, `env/tasks/multi_agent/size_rsi_cache.py` | 단독 size RSI와 초기 물리 검사 캐시 |
| `tokenhsi/utils/edge_*`, `env/tasks/multi_agent/edge_*` | 현재 공동/단독 과제가 재사용하는 graph·geometry·보상 helper |
| `tokenhsi/utils/task_role_spec.py`, `learning/multi_agent/task_role_encoder.py` | 기존 PTH의 task packet과 embedding 파라미터 |
| `tokenhsi/scripts/prepare_joint_carry.py`, `joint_carry/runtime_manifest.json` | 원본 데이터 연결·필수 입력과 해시 검증 |
| `joint_carry/teamhoi_retarget_posture_all/` | 현재 AMP 후진9개·옆걸음3개 |
| `joint_carry/carry_joint_full_pickup/`, `carry_joint_corrected/{carryWith,putDown}/` | 현재 공동 RSI·내부 상대 심링크 |
| `tokenhsi/data/assets/`, `lpanlib/` | humanoid 자산과 모션 라이브러리 |
| `tokenhsi/tests/` | SA·GraspView·AT20 성공조건 CPU 테스트와 원본/PTH 비교 실행기 |
| `output/` | 학습·평가·검증 산출물과 RSI 캐시; Git 제외 |

Stage1 이름의 공통 helper와 checkpoint metadata는 두 SA의 전이·환경 계산에 필요하다.
Stage1 학습/토이 실행 경로는 제공하지 않는다. 원본 데이터 심링크와 외부 PTH는 읽기 전용으로 사용한다.
