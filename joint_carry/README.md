# 공동운반 실행 데이터

현재 두 SA가 사용하는 입력만 포함한다.

| 경로 | 용도 |
| --- | --- |
| `teamhoi_retarget_posture_all/` | AMP 후진9개·옆걸음3개, sampling 목록·해시 |
| `carry_joint_full_pickup/pickUp/` | 보정 pickup100프레임 |
| `carry_joint_full_pickup/{carryWith,putDown}` | `carry_joint_corrected/`로 연결되는 내부 상대 심링크 |
| `carry_joint_corrected/{carryWith,putDown}/` | carryWith129·putDown135프레임 |
| `runtime_manifest.json` | 현재 AMP/RSI18파일의 SHA256 |
| `teamhoi_reference/LICENSE.TeamHOI`, `manifest.json` | 데이터 라이선스·원본 출처 |

원본 TokenHSI 데이터와 Stage1 체크포인트는 별도로 준비한다. 저장소 루트에서:

```bash
source tokenhsi/scripts/multi_agent/runtime_env.sh
python tokenhsi/scripts/prepare_joint_carry.py --data-root /path/to/TokenHSI/tokenhsi/data
```

`TOKENHSI_DATA_ROOT`로 지정해도 된다. 원본은 읽기 전용이고 없거나 끊어진 링크만 연결한다.
단독 size RSI 캐시는 첫 실행에서 `output/rsi_cache`에 생성한다.
재보정 없이 포함 데이터를 그대로 사용한다. [실행 가이드](../markdowns/config.md)를 따른다.
