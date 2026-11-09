# 인턴 실험 요청서

목표는 **distillation 기반 방법 안에서 여러 방식을 검토하여, TokenHSI 기존 성능 하락을 5% 이내로 유지하면서 우리 목표 영역 기준 PUSH final 성공률이 85%를 초과하는 단일 student checkpoint(`.pth`)를 만드는 것**이다. 최종 checkpoint는 요청자가 직접 재평가하여 논문 결과 테이블에 사용할 예정이다. TRAJ는 제외하고 기존 성능은 SIT/CARRY/CLIMB를 대상으로 한다. 방법의 기준은 [METHOD.md](METHOD.md), 입력은 [PUSH_INPUT.md](docs/PUSH_INPUT.md)다.

## 목표와 평가 기준

- 이 요청서에서는 “5% 하락”을 **원본 TokenHSI teacher 대비 상대 하락률**로 해석한다. 각 과제의 성공률을 동일 조건에서 비교하여 `student ≥ 0.95 × teacher`를 만족해야 한다. 세 과제 평균으로 특정 과제의 큰 하락을 가리지 않도록 **SIT/CARRY/CLIMB 각각** 확인한다. 이는 5%p 하락과 다르며, 의도한 정의가 다르면 실험 전에 담당자와 확정하여 문서에 남긴다. teacher 성공률이 0인 과제는 상대 하락률을 계산할 수 없으므로 별도 기준을 먼저 합의한다.
- PUSH 목표는 **우리 목표 영역 기준 final 성공률 > 85%**다. final은 첫 에피소드 종료 시점에 목표 영역 안에 있는지를 판정한다. **ever 성공률은 보조 지표로 함께 제출하고 발표 자료에도 포함**한다. ever는 같은 에피소드 중 한 번이라도 목표 영역에 도달했는지다. 별도의 연속 유지 시간 조건은 추가하지 않는다. 어떤 checkpoint 하나가 기존 과제 보존과 PUSH 목표를 동시에 만족해야 한다. 과제마다 다른 checkpoint를 골라 합치지 않는다.
- 실험 전에 평가 seed·표본 수·성공 정의·집계 방식과 validation/test 분리를 확정한다. PUSH 주 지표는 final, 보조 지표는 ever로 고정하며 두 값과 낙상을 모두 보고한다. 기존 SIT/CARRY/CLIMB는 원본 TokenHSI의 각 과제 eval 기준으로 teacher와 student를 평가한다. PUSH는 우리 PUSH의 목표 영역 기준을 사용하며 구체적인 정의와 현재 이관 eval의 차이는 [평가 기준](EVALUATION.md)을 따른다. 결과를 본 뒤 유리한 지표나 seed로 바꾸지 않는다.
- 원본 teacher와 초기 student의 기존 과제 성능을 같은 프로토콜로 먼저 측정한다. 최종 선정은 validation으로 하고, 고정한 test는 마지막 평가에 사용한다. seed별 결과·표본 수·변동성을 함께 제출한다. 목표 달성은 보장된 결과가 아니며, 미달 시 실제 수치와 원인을 그대로 보고한다.

## 탐색 범위와 현재 구현 점검

기존 스킬 보존은 **distillation 방식으로 제한**한다. 현재 제공한 방식은 고정 teacher의 행동 분포에 forward KL을 적용한 나이브한 출발점이므로, 그대로 신뢰하기보다 구현과 설정의 타당성을 먼저 점검해 달라는 요청이다. 초기 KD 계수 0.001이나 기존 후보 세 개만으로 탐색을 제한하지 않는다.

Distillation 범위 안에서 loss 구성·방향, 계수와 스케줄, 과제별 가중치·샘플링, teacher 데이터 수집/replay 등 여러 대안을 검토할 수 있다. 필요하면 feature distillation 같은 보조 항도 검토하되, 추가 구성과 학습 비용을 명시한다. 최종 추론은 teacher 없이 단일 student로 수행한다. TokenHSI 구조/관측, teacher·초기 가중치, PUSH 환경·물성·reward·AMP·RSI와 성공 기준은 고정한다. 이 범위를 바꾸어야 한다면 먼저 담당자와 합의하고 별도 실험으로 구분한다.

점검에는 teacher 동결·RMS 처리·관측/토큰 매핑·KL 방향/스케일·PUSH/KD/비학습 슬롯의 loss 및 gradient 분리·AMP replay 분리·checkpoint 저장/복원 일관성을 포함한다. 초기 student가 원래 스킬을 보존하는지, 평가에 teacher 행동이 섞이지 않는지도 확인한다. 오류 수정과 성능 개선의 효과는 구분하여 기록한다.

## 논문 사용과 최종 제출

방법을 논문 본문에 설명하고 결과 테이블에 넣을 수 있도록 신중하게 진행한다. 각 방법의 수식·loss 적용 대상·teacher 사용 방식·하이퍼파라미터·선정 절차와 변경 코드를 재현 가능하게 남긴다. 외부 방법/코드를 사용하면 출처와 사용 범위를 기록하고, 제공한 나이브 KD와 최종 방법의 차이 및 비교 실험을 정리한다. 평가 누수·성공 조건 완화·일부 seed만 선택하는 방식으로 목표를 맞추지 않는다.

최종 제출물은 **목표를 동시에 만족하는 student `.pth`**, 정확히 대응하는 코드/설정/실행 명령, checkpoint SHA256, teacher 대비 과제별 평가표, PUSH 목표 영역 기준 final/ever·낙상·거리 결과 및 발표용 비교표, 전체 seed별 원본 결과/로그, 방법 설명과 탐색 이력이다. 다른 머신에서 strict load와 student-only 재평가가 가능해야 한다. 요청자가 전달받은 checkpoint를 독립적으로 점검한 뒤 테이블 반영 여부를 결정한다. 목표 미달이면 가장 나은 후보와 미달 항목을 명시하고 완료로 표시하지 않는다.

## 진행 순서

1. README 순서로 무결성·입력·CPU 검사를 통과시키고 환경 버전과 로그를 보존한다.
2. GPU 점유를 확인하고 **참조 2048scene + native KD 2048환경** 그대로 1 epoch 통합 검사를 실행한다. 메모리 부족이나 CUDA 오류가 나면 로그를 공유한다. 환경 수를 임의로 줄이거나 기존 프로세스를 종료하지 않는다.
3. 기본 KD 계수 0.001의 나이브 KD를 기준 실험으로 먼저 확인하고, 이후 distillation 대안을 비교한다. 26,000 epoch은 설정값이며 학습 완료 모델은 이관본에 없다. GPU 통합 검사 실패 시 본학습으로 진행하지 않는다.
4. 기존 3과제의 초기 student 평가를 먼저 보존한다. 이후 저장 student에 PUSH와 기존 3과제 평가를 수행한다. PUSH는 loco-only·512scene(1024명)에서 우리 목표 영역 기준의 final/ever를 기록하고, native는 TokenHSI 원본 과제별 eval 규칙을 따른다. 위 목표에 맞춰 공통 seed 목록·평가 지표·validation/test 분리를 실행 전에 확정한다.
5. KD 대안마다 환경 YAML·학습 YAML·train/eval 스크립트·출력을 분리한다. 0.0001/0.001/0.01은 초기 계수 후보이며 탐색 범위의 상한이 아니다. teacher와 초기 student, 물성·reward·AMP·RSI를 고정한다. 최종 test로 계수를 고르지 않는다.

원본 eval의 PUSH JSON은 `output_etc/check/push_eval.json`에 덮어쓴다. 각 실행 직후 checkpoint/seed별 폴더에 복사한다. 기존 과제는 평가 로그와 원본 산출물을 함께 보존한다. PUSH 성공률과 native 평가 수치는 서로 다른 과제 정의다. 논문용 직접 비교에는 공통 저장 initial plan 등 METHOD의 추가 조건이 필요하다.

## 결과 기록

각 실행마다 아래 표를 채우고 실행 명령·수정 YAML·전체 로그·checkpoint SHA256을 함께 전달한다. 실패도 결과에 포함한다.

| 항목 | 기록 |
| --- | --- |
| 코드 | HANDOFF_SOURCE.json의 commit, 추가 수정 diff |
| 실행 | 날짜, GPU/드라이버, Python/torch/Isaac Gym 버전, 명령 |
| 학습 | seed, KD 계수, epoch, 실제 PUSH/KD/전체 scene step, wall time |
| checkpoint | 경로, SHA256, 새 학습/재개 여부 |
| PUSH | seed별 표본 수, 목표 영역 final 성공률(주 지표)/ever 성공률(발표용 보조), 낙상, 종료 잔여 거리 |
| SIT/CARRY/CLIMB | 각각 teacher/초기 student/학습 후 지표, 상대 하락률 및 %p 차이 |
| 안정성 | NaN/Inf, OOM, worker/IPC 오류, 처리속도 |

이관 시점에는 GPU PPO 통합·장기 학습·성공률이 검증되지 않았다. 기존 CPU 검증 이력은 [CHANGELOG.md](CHANGELOG.md)에 있으며 새 결과와 구분한다.
