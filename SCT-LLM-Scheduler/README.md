# SCT + LLM Scheduler

고정된 **Stage1 SCT 26k** 정책에 LLM의 초기 출발 스케줄을 적용하는 평가 코드다.
장면 실행 전에 한 번 API를 호출하고, 에이전트를 초기 위치에서 freeze한 뒤 지정된 simulation time에 영구 release한다.
실행 중 재계획·접촉 후 재동결은 없다.

**설정부터 실행·결과 확인까지는 [USAGE.md](USAGE.md)를 따른다.**

| 필요한 작업 | 안내 |
|---|---|
| conda·API key·모델 설정 | [사용 안내 2절](USAGE.md#2-환경과-api-설정) |
| 실제 system/user 메시지·입출력 JSON | [3·4절](USAGE.md#3-api-입력과-프롬프트) |
| API 재호출·저장 응답 재평가 | [5절](USAGE.md#5-실행-작업-선택) |
| legacy/snapshot freeze·GPU 운영 | [6절](USAGE.md#6-gpu-평가와-freeze) |
| 최신 SR·Time Succ./All·t_coop·Jerk | [7절](USAGE.md#7-프로토콜검증보고서) |
| 상태·실패·파일 위치 확인 | [8·9절](USAGE.md#8-저장-위치) |
| 8H3m·16H4m 부분 평가 | [10절](USAGE.md#10-8h16h-그룹-간격-변형) |
| 현재 결과 폴더 | [11절](USAGE.md#11-실험-경로) |

```bash
cd /home/hwanhee/juan/CVPR2027/SCT-LLM-Scheduler
conda activate tokenhsi_koo
python -m unittest discover -s tests -q
```

`.env` 설정 예시는 [.env.example](.env.example), 실제 프롬프트는 [prompts/system.txt](prompts/system.txt),
정책 checkpoint 식별자는 [stage1_policy.json](stage1_policy.json)에 있다.
현재 주 평가 freeze는 **legacy**다. GPU는 **6/7만**, `/home/hwanhee/ksh`는 읽기 전용이며 MPS를 제어하지 않는다.

[최신 전체 보고서](output_etc/check/eval_protocol_v1_legacy_20261111_165027/report.md)는64환경 탐색 결과다.
실제 GPU rollout·원본 정책 일치·초기 상태·release·CPU 재채점·첫 정책 입력 검증을 완료했고,
최신 프로토콜 지표와 seed 평균±표본표준편차를 저장했다. 실행별 검증 범위는 해당 manifest·verification 파일을 확인한다.

[CHANGELOG.md](CHANGELOG.md)는 변경 이력이다.
[BATCH_EVAL.md](BATCH_EVAL.md)·[RECORDING.md](RECORDING.md)는 세부/과거 실행 참고자료이며,
현재 사용 흐름과 경로는 [USAGE.md](USAGE.md)를 우선한다.
원본 참고 코드·라이선스는 [provenance.json](provenance.json), [THIRD_PARTY_LICENSE.txt](THIRD_PARTY_LICENSE.txt)에 보존한다.
