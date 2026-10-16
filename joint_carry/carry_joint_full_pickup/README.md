# 공동 pickup 전체 프레임 RSI

`pickUp/`은 `output/joint_pickup_full_20261012/pickUp/`에서 검증한 보정본이다.
현재 asset의 관절 제한 안에서 다리·팔을 보정했으며, 30fps·100프레임·상자52×80×40cm와 상자 궤적을 유지한다.
CPU geometry/초기0.1s 물리100/100, GPU2048환경·부분 reset 검사를 통과했다.

`carryWith/`, `putDown/`은 기존 `carry_joint_corrected/`의 상대 심링크다.
로더 pool은 pickup100/carryWith129/putDown135, 총364프레임이다.
새 `goal360_fullpickup_task_embedding` config만 이 경로를 사용한다. AMP reference는 교체하지 않는다.

6개 로더 입력의 결합 SHA256은 `2ef45907ea7fd580eb0863d95d1fa158430cfda5444d8ac96b02b28b2345037a`다.
개별 파일은 `../runtime_manifest.json`, 실행법·검증 범위는 `../../markdowns/config.md`를 참조한다.
이는 RSI 초기화 검사이며 장기 공동집기·운반 성공을 보장하지 않는다.
