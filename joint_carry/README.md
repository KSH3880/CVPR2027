# 공동 RSI·AMP

이 checkout의 stage2 입력과 크기별 보정/물리 검사 코드다. 원본은 읽기 전용으로 참조하고 `source_rsi/`에 복사했다. 출처·hash는 `source_manifest.json`, AMP 출처·라이선스는 `teamhoi_reference/`와 `teamhoi_retarget_posture_all/manifest.json`에 있다.

`single_source.py`는 원본 1인 B19/B20/B21에서 공통 다리 보정과 크기별 공동 배치를 생성한다. `size_candidates.py`가 실제32-DOF 팔을 보정하고, `size_screen.py`가 native asset과 GPU PhysX로 검사한다. `build_size_rsi.py`의 요청 JSON은 `[{"size": [0.4, 0.8, 0.4]}]` 형식이다. 기본 생성은 `--source single --inset 0.15`, 기본 디렉터리는 `joint_carry/stage2_size_rsi/inset15_single`이다. `--workers 4`로 같은 GPU의 독립 worker4개를 사용할 수 있다. 기존 공동 보정본 비교에는 `--source stage2`와 별도 디렉터리를 지정한다.

AT/ON_TOP은 공동 캐시를 공유하고 ON_TOP은 putDown을 사용하지 않으므로 받침별 자료는 준비하지 않는다. `audit_size_rsi.py --directory <캐시> --requests <요청 JSON> --report <검사 JSON>`으로 전체 요청 크기의 무결성과 유효 상태를 검사한다. 검사 report·로그는 `output_etc/check/`에 둔다.

`render_size_rsi.py --output output_etc/check/stage2_anchor15/videos_single`는 원본과 새 범위의 대표27크기를 native asset으로 재생한다. 물리 검사 통과/제외 프레임을 표시하며 정책 rollout은 아니다.

실행 환경은 `tokenhsi/scripts/multi_agent/runtime_env.sh`를 사용한다. 상세 계약은 [Stage2 참고](../markdowns/reference/stage2.md)를 따른다.
