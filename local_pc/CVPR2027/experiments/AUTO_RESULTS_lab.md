# lab 자동 결과 원장

> 자동 생성 파일이다. 직접 수정하지 않는다. 원시 기준은 트랙 상태 JSON과 실행 로그의 종료 요약이다.
> 연구적 비교와 확정 결론은 해당 `EXPERIMENTS.md`에 따로 기록한다.

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
