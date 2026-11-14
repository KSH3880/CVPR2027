# Few-shot: SCT Stage1 단독 행동 기록 10개

기존 baseline 프롬프트는 변경하지 않는다. 이 버전은 스케줄 정답 대신 **정책의 움직임·행동 소요 시간**을 알려주는 1인 행동 참고 자료를 사용한다.

## 구성

- AT 5개: 단독 agent가 물체를 집고 운반하여 목표 중심 20cm 이내에 도달할 때까지.
- PUSH 5개: 단독 agent가 접근하여 원본 PUSH 양손 접근·접촉력 detector를 처음 만족할 때까지.
- 에피소드당 agent는 정확히 1명. t=0부터 실행하며 scheduler 대기·freeze는 없다.
- 모델은 기존 SCT Stage1 26k native checkpoint. Stage2·FSM·LLM 제어는 사용하지 않는다.
- 성공한 원본 rollout에서 초기 사람→물체 거리의 다섯 분위 위치에 해당하는 예시를 선택한다. 전체 시도/실패도 원시 기록과 selection_audit에 남긴다. 성공률 추정을 위한 무작위 표본은 아니다.

## 입력 구조

```python
messages = [
    {"role": "system", "content": fewshot_system_prompt},
    {"role": "user", "content": json.dumps({
        "environment_spec": environment_spec,
        "demonstrations": demonstrations,  # 1인 AT 5개 + PUSH contact 5개
        "current_scene": current_scene,
    }, ensure_ascii=False, allow_nan=False)},
]
```

각 예시에는 초기 사람/물체 상태, task와 목표, 시간별 root·양손·물체 궤적, 측정 이벤트와 detector 정의, 원본 seed/env/체크포인트 정보가 들어간다. 별도 release schedule 정답은 넣지 않는다. 단위는 m/s/rad, 접촉력은 N. 예시 좌표는 환경 origin을 뺀 좌표이며 world 축 방향은 유지한다.

`environment_spec.controller.timing_calibration`은 기존 “No historical demonstrations ... supplied” 대신 다음처럼 기술한다.

```json
{"timing_calibration": "Ten single-agent Stage1 behavioral recordings are supplied: five AT completions and five PUSH first-contact trajectories. They provide empirical timing references, not coordinated release schedules."}
```

## 프롬프트 해석

- AT는 Enable **선행**의 운반/배치 시간 참고용이다. 후행 SIT·CLIMB·ON_TOP 시간을 직접 보정한 자료는 아니다.
- 1인 PUSH 접촉은 Coupled 참여자의 접근 시간 참고용이다. 다인 접촉 동기화나 공유 영역 정렬·충돌을 재현한 예시는 아니다.
- AT 완료는 물체 중심의 3D 목표 오차 ≤0.20m이며 손을 놓았거나 물체가 정지했다는 의미가 아니다.
- PUSH contact는 원본 코드의 접근 및 힘 조건이며, 12cm 표면 거리만으로 정의한 t_coop와 다르다. Current scene의 정성적 readiness와도 같다고 가정하지 않는다.
- contact hit는 물리 substep에서 발생한 결과를 control frame에 OR로 모은 것이다. 시간 해상도와 마지막 substep의 힘 표본 차이를 명시한다.
- 궤적은 0.5초 간격, 마지막1초는0.1초 간격으로 줄이고 초기·종료 프레임을 반드시 포함한다. 이벤트 시각은 full-resolution 기록에서 추출한다.
- 첫 물리 step 이전 힘은 측정 불가이므로 JSON에서 null이다.

## 실행 상태

CLI에서 `--fewshot`을 명시하면 manifest의 실제 예시10개를 붙이고 timing_calibration 설명을 갱신한다. demonstrations가 있는 요청은 fewshot/system.txt를 사용하며, 없는 요청은 기존 baseline을 사용한다. current_scene은 변경하지 않는다. 기존 응답 검증을 그대로 적용한다.

```bash
python -m sct_scheduler <저장된 request.json> --fewshot --json-mode --reasoning-effort medium --output output_etc/check/<새 출력 폴더>
```

실제 입력은 출력 폴더의 request.json/messages.json, 응답은 response.raw.txt, 검증 통과 응답은 response.validated.json에 보관한다. API 오류나 검증 실패는 failure.json에 남기고 자동 재호출하지 않는다. 물리 평가는 이 명령으로 실행되지 않는다.

재수집: `python collect_single_stage1.py` (GPU7, task당32개 독립 1인 에피소드). 추출: tokenhsi_koo Python으로 `export_single_stage1_fewshot.py <runtime 기록 경로>` 실행.

baseline_snapshot/은 최초 분리 시점의 기존 프롬프트이며 그대로 보존한다. 출력 JSON 구조와 release 규칙은 baseline과 동일하다.

## 준비된 실제 샘플

[demonstrations.json](../../output_etc/check/fewshot_examples_single_stage1_eval_geometry_20261112_154050/demonstrations.json)에 실제 예시10개가 준비되어 있다. [샘플 목록과 검증](../../output_etc/check/fewshot_examples_single_stage1_eval_geometry_20261112_154050/README.md)을 참고한다.

## Eval 범위에 맞춘 샘플 재수집

최초 단독 프로토콜 샘플은 AT 물체→목표 최대13.31m, PUSH 최대9.50m로 실제 사용 eval보다 넓었다. 기존 기록은 보존하고 새 `collect_eval_geometry_stage1.py`로 교체 자료를 수집한다.

| 조건 | AT | PUSH |
|---|---|---|
| 초기 root→물체 중심 XY 거리 | 1~5m | 2~10m |
| 물체 중심→목표 XY 거리 | 1~5m | 2~5m |
| 물체 크기 | Enable 공유물체 범위: X/Y0.40~0.60, Z0.35~0.50m,0.05m grid | native 1인 PUSH base 크기 |
| 배치 | 원본 loco 자세·heading 유지, 두 거리 구간을 독립 stratified uniform으로 생성 | 상자 뒤쪽, 목표 box-local +Y |

각 task32개 환경, 환경당1명이며 거리의 **범위**를 맞춘 독립 행동 자료다. 다인 eval과 같은 초기 장면이나 동일한 난이도 분포라고 주장하지 않는다. PUSH 물체의 인원수별 X축 확장과 다른 참여자는 포함하지 않는다. `geometry_audit.json`에 모든 환경의 실제 거리·크기·범위 검증을 기록한다.

현재 준비된 예시는 eval 거리 정렬 버전이다. 재수집은 `python collect_eval_geometry_stage1.py`, 이전 `collect_single_stage1.py`는 거리 정렬 전 버전을 재현한다.
