# 보정된 공동운반 RSI 인계

**다음 Codex는 [HANDOFF.md](HANDOFF.md)를 먼저 읽는다.** 준비된 것은 두 사람·한 상자의 보정된 RSI 초기 상태 319개이며 학습 reset 연결은 미구현이다.

- [최종 보정 전후 영상](carry_joint_corrected/before_after/comparison_all_skills.mp4)
- [최종 보정 전후 영상 · 0.5배속](carry_joint_corrected/before_after/comparison_half_speed_all_skills.mp4)
- [동작별 RSI snapshot 개수](carry_joint_corrected/selected_frames.json) — 아래 기준 경로에서 총 319개 로딩
- [코드·인계 문서까지 포함한 전달 ZIP](joint_carry_handoff.zip)
- [보정 후 물리 보고서](carry_corrected_physics/report.json)
- [보정 후 물리 그래프](carry_corrected_physics/physics_summary.png)

| 폴더 | 내용 |
| --- | --- |
| `HANDOFF.md` | 다음 구현 목표·배열 계약·검증 범위·코드 진입점·실행법 |
| `scripts/` | 생성·보정·물리 검사·인계 무결성 확인 코드 |
| `carry_joint_corrected/` | 최종 보정 reference·동작별 RSI snapshot·보정 전후 영상 |
| `carry_corrected_physics/` | 보정 후 실제 32-DOF FK·관통·0.1초 초기 충격·PD 추종 검사 |
| `archive/` | 보정 전 모션·영상·실패 검사. RSI 학습 입력으로 사용하지 않는다. 전달 ZIP에서는 제외한다. |
| `handoff_manifest.json` | 전달 파일·저장소 의존 파일의 SHA256 |

RSI 연결 시 읽을 기준 파일은 `carry_joint_corrected/{pickUp,carryWith,putDown}/screened_rsi_snapshots.npz`다.

최종 snapshot 319개는 **0.1초 초기 상태 검사**를 통과했다. 연속 보행의 발 미끄러짐·장기 균형은 해결하지 않았고 현재 학습 reset에는 연결하지 않았다. 팔 관절은 IK로 보정했고 다리 관절각은 유지했다.

저장소 루트에서 무결성을 확인한다(학습·시뮬레이션 실행 없음).

```bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
python joint_carry/scripts/check_handoff.py
```

재생성 명령은 [HANDOFF.md](HANDOFF.md), 공통 규칙은 [실행 가이드](../markdowns/config.md)를 따른다. 기존 TokenHSI 저장소·Isaac Gym·`tokenhsi` 환경에 의존하며 독립 실행 패키지는 아니다. 큰 데이터·영상·ZIP은 Git 추적에서 제외하므로 새 checkout/머신에는 **전달 ZIP도 함께 복사**한다.
