# CVPR2027 평가 프로토콜

**다른 작업자에게 평가는 [짧은 평가 인수인계](EVAL_HANDOFF.md)를 전달한다.** Table 1=Independent/Mixed 확장 평가, Table 2=Enabling, Table 3=Coupled다.

전체 기준은 **[공용 평가 프로토콜](../evaluation_protocol/README.md)** 한 곳에서 관리한다. 버전 `cvpr2027-evaluation-v1.1.0`, 공동 PUSH 출발 변위 **20cm**. Table 1의8H/16H Independent·Mixed는 그룹 중심을 각각 **반경3m/7.5m** 원 둘레에 균등한 각 간격으로 배치한다. 반경은 그룹 간 최소 간격이 아니다.

| 내용 | 위치 |
| --- | --- |
| 공통 조건·상자·거리·성공·Error·관계별 규칙·테이블·최신 t_coop/Jerk | [전체 기준](../evaluation_protocol/README.md) |
| 레포별 진입점·실행 명령·checkpoint·지원 상태 | [구현 안내](../evaluation_protocol/implementations.md) |
| 다른 레포에서 사용할 계산 예제 | [공통 검증](../evaluation_protocol/checks/README.md) |
| 이 checkout의 실행 세부 안내 | [eval/README.md](eval/README.md) |

이전 규약은 [보관본](markdowns/legacy/cvpr2027_evaluation_protocol_20261111_before_shared.md)에 남겼다. 기존 구간 내6cm t_coop은 과거 정의이며 최신20cm 시작·12cm 표면·전체 누적 t_coop과 구분한다.
