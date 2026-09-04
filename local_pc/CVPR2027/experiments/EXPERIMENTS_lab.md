# lab 실험 결론

lab 큐에서 끝난 인프라·프로브 결과의 검토된 결론이다. 원시 종료 기록은
[AUTO_RESULTS](AUTO_RESULTS_lab.md), 현재 큐는 [PLAN_lab](../plan/PLAN_lab.md)을 따른다.

<!-- AUTO_RESULTS -->

## 자동 결과 (최신순)

> 자동 생성 영역이다. 최신 결과가 위, 가장 오래된 결과가 아래다.
> 비교·채택·기각 해석은 이 블록 아래의 수동 기록에 남긴다.

| 완료 시각 | 상태 | tag | 자동 요약 | 큐 질문 |
|---|---|---|---|---|
| 2026-08-18 13:51 | 완료 | `lab_b_now` | LAB tag=lab_b_now repo=steer envs=4096 cp=32M spacing=0 cell=60 rc=0 warn=0 needM=0.0 sfps=56377 | **빠져 있던 기준선.** 지금 운영 설정 그대로(spacing 0 · 32M · CELL 60)를 단독 카드에서 잰다. 이게 없으면 B·C 의 sfps 를 "지금보다 몇 % 빠른가" 로 못 읽는다 -- 통과하는 것끼리만 비교하게 된다 |
| 2026-08-18 13:51 | 완료 | `lab_b_cp1` | LAB tag=lab_b_cp1 repo=steer envs=4096 cp=1M spacing=5 cell=60 rc=0 warn=0 needM=0.0 sfps=60786 | **B-1. 최소 버퍼.** A 에서 spacing>=5 면 요구량이 1M 미만으로 떨어졌다. 지금 쓰는 32M 은 과하고, 과한 버퍼는 처리량을 깎는다 (128M 이 32M 대비 28 % 손해). 여기서부터 올린다 |
| 2026-08-18 13:51 | 완료 | `lab_b_cp4` | LAB tag=lab_b_cp4 repo=steer envs=4096 cp=4M spacing=5 cell=60 rc=0 warn=0 needM=0.0 sfps=58512 | B-2 |
| 2026-08-18 13:51 | 완료 | `lab_b_cp32` | LAB tag=lab_b_cp32 repo=steer envs=4096 cp=32M spacing=5 cell=60 rc=0 warn=0 needM=0.0 sfps=53830 | B-3. **지금 쓰는 값.** 1M 과 sfps 를 비교하면 버퍼 과다의 대가가 나온다 |
| 2026-08-18 13:51 | 완료 | `lab_c_8192` | LAB tag=lab_c_8192 repo=steer envs=8192 cp=4M spacing=5 cell=60 rc=0 warn=0 needM=0.0 sfps=58619 | **C-1. 8192 가 열리나.** steer 는 지금까지 8192 에서 죽었고 그 원인이 접촉쌍이었다. 열리면 이터당 샘플이 두 배 -- 원본 대비 24 % 라는 지금 최대 약점을 직접 때린다 |
| 2026-08-18 13:51 | 완료 | `lab_c_16384` | LAB tag=lab_c_16384 repo=steer envs=16384 cp=8M spacing=5 cell=60 rc=0 warn=0 needM=0.0 memMiB=42833 sfps=87627 | C-2. 상한. 메모리가 먼저 막는지 접촉쌍이 먼저 막는지 |
| 2026-08-18 13:51 | 완료 | `lab_r_4096` | LAB tag=lab_r_4096 repo=steer envs=4096 cp=0.25M spacing=5 cell=60 rc=0 warn=0 needM=0.0 memMiB=16833 sfps=57203 | **D-1. 요구량 자체.** cp=1M 은 통과해버려서 "1M 이하"까지만 안다. 일부러 모자라게 줘 로그의 `Capacity to N` 으로 실제 요구량을 읽는다 |
| 2026-08-18 13:51 | 완료 | `lab_r_8192` | LAB tag=lab_r_8192 repo=steer envs=8192 cp=0.25M spacing=5 cell=60 rc=0 warn=0 needM=0.0 memMiB=25577 sfps=59814 | D-2. 8192 요구량. 4096 의 2 배인지(env 선형) 그 이상인지 |
| 2026-08-18 13:51 | 완료 | `lab_r_16384` | LAB tag=lab_r_16384 repo=steer envs=16384 cp=0.25M spacing=5 cell=60 rc=1 warn=1 needM=0.3 memMiB=41171 sfps=NA | D-3. 16384 요구량. 선형이면 env 당 상수가 확정되고 그 뒤로는 계산으로 정한다. 접촉쌍을 일부러 모자라게 줘 `rc=1`이 예상 종료다 |
| 2026-08-18 13:51 | 완료 | `lab_c_4096` | LAB tag=lab_c_4096 repo=steer envs=4096 cp=4M spacing=5 cell=60 rc=0 warn=0 needM=0.0 memMiB=17881 sfps=61732 | **C-0. 기준점.** 같은 조건의 4096. C-1/C-2 를 이것과 나눠야 env 당 처리량이 나온다 |
| 2026-08-18 13:51 | 완료 | `lab_e_16384_cp1` | LAB tag=lab_e_16384_cp1 repo=steer envs=16384 cp=1M spacing=5 cell=60 rc=0 warn=0 needM=0.0 memMiB=40865 sfps=66619 | **E-1.** 16384 를 8M 으로 쟀는데 B축은 버퍼가 클수록 느리다고 했다. 요구량이 0.3M 이니 1M 로 충분 — 87.6k 가 더 오르는지 |
| 2026-08-18 13:51 | 완료 | `lab_e_32768` | LAB tag=lab_e_32768 repo=steer envs=32768 cp=2M spacing=5 cell=60 rc=0 warn=0 needM=0.0 memMiB=74413 sfps=37101 | **E-2. 메모리 상한.** 실측 `mem ≈ 8.0 GB + 0.00215×env` 로 78.5 GB 예측 = 80 GB 에 아슬. OOM 이면 16384 가 상한으로 확정 |

<!-- /AUTO_RESULTS -->

## 환경 간격·접촉 버퍼·환경 수 (2026-08-17)

### 결론

| 항목 | 확정값 | 근거 |
|---|---:|---|
| `envSpacing` | 5 | spacing 0의 22.6 M 접촉 요구가 1 M 미만으로 감소했고 15·30·60은 추가 이득 없음 |
| `STEER_CELL` | 제거 가능 | envSpacing이 캐릭터와 플랫폼을 함께 분리하므로 수동 격자는 중복 |
| contact pairs | 4 M | 16,384 env에서도 실측 요구량 약 0.3 M |
| 비교 배치 | 4,096 env | 카드당 2잡으로 시드·설정 슬롯 확보 |
| 대형 장기 학습 | 16,384 env | 공유 후보에서 후순위로 두되 메모리가 충분하면 카드당 2잡 허용 |
| 32,768 env | 사용하지 않음 | 메모리에는 들어가지만 처리량이 절반 수준으로 하락 |

### 원인

`envSpacing=0`에서는 사용하지 않는 플랫폼들이 같은 world 원점에 주차됐다.
플랫폼과 target platform 8,192개가 겹치면서 접촉쌍 요구량이 22.6 M까지 증가했다.
`envSpacing=5`는 env 원점 자체를 분리해 이 중첩을 제거한다.

### 적용 조건

- 실행 중 steer 배치의 spacing·env 수 회귀가 통과한 뒤 기본 설정에 반영한다.
- 설정 변경 뒤 원본 Carry 성공률 회귀를 다시 측정한다.
- 처리량 비교에는 GPU 동거와 메모리 조건을 함께 기록하고 같은 조건끼리 비교한다.
