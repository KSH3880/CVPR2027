# 전체 실험 계획

이 문서는 현재 우선순위, 공통 실험 설계, GPU 배치를 관리한다.
실행 큐는 트랙 PLAN, 완료 수치는 `../experiments/AUTO_RESULTS_<track>.md`,
검토된 결론은 `../experiments/EXPERIMENTS_<track>.md`가 기준이다.

## 활성 큐 요약

트랙 PLAN의 활성 큐가 원본이다. 아래 표는 실행 상태가 갱신될 때 자동으로 복사되므로
직접 수정하지 않는다.

<!-- QUEUE_SUMMARY -->

| 트랙 | tag | 질문·판정 기준 | 상태/GPU | 환경변수 |
|---|---|---|---|---|
| [steer](PLAN_steer.md) | — | 활성 큐 없음 | — | — |
| [ma](PLAN_ma.md) | — | 활성 큐 없음 | — | — |
| [lab](PLAN_lab.md) | — | 활성 큐 없음 | — | — |

<!-- /QUEUE_SUMMARY -->

| 트랙 | 활성 큐 | 자동 결과 | 검토된 결론 |
|---|---|---|---|
| steer | [PLAN_steer.md](PLAN_steer.md) | [AUTO](../experiments/AUTO_RESULTS_steer.md) | [EXPERIMENTS](../experiments/EXPERIMENTS_steer.md) |
| ma | [PLAN_ma.md](PLAN_ma.md) | [AUTO](../experiments/AUTO_RESULTS_ma.md) | [EXPERIMENTS](../experiments/EXPERIMENTS_ma.md) |
| lab | [PLAN_lab.md](PLAN_lab.md) | [AUTO](../experiments/AUTO_RESULTS_lab.md) | [EXPERIMENTS](../experiments/EXPERIMENTS_lab.md) |

## GPU 현황

`bash scripts/gpumap.sh --write`가 아래 블록을 갱신한다. 직접 수정하지 않는다.

<!-- GPUMAP -->

`08-21 20:45` 기준 · 실행 시작/종료 때 자동 갱신

| GPU | 메모리 | 프로세스 | 트랙 | 올라가 있는 것 |
|---|---|---|---|---|
| gpu0 | 0 M | 0 | steer | `f34_rs60`(starting) |
| gpu1 | 0 M | 0 | steer | `f31_vk125`(starting) |
| gpu2 | 0 M | 0 | — | _(빈 카드)_ |
| gpu3 | 18939 M | 1 | ? | `ms12_base4L_s0` |
| gpu4 | 18937 M | 1 | ? steer | `ms11_clip_s0` `f34_rp60`(starting) |
| gpu5 | 37908 M | 2 | ? steer | `ms11_clip_s1` `ms6_m8_long_s0` `f34_rp150`(starting) |
| gpu6 | 37678 M | 2 | ? | `ms13_sep20L_s0` `ms13_sep9L_s0` |
| gpu7 | 0 M | 0 | — | _(빈 카드)_ |

| 트랙 | 대기 | 다음 |
|---|---|---|
| steer | 0 | — |
| ma | 0 | — |
| lab | 0 | — |

### NEEDS ATTENTION

| 트랙 | tag | 상태 |
|---|---|---|
| steer | `f30_t_c10` | training script rc=137 |
| steer | `f30_t_c05` | training script rc=137 |
| steer | `f31_vk060` | training script rc=137 |
| steer | `f32_stop2` | training script rc=137 |
| steer | `f22b_long_ctrl` | training script rc=137 |

<!-- /GPUMAP -->

## 실험 설계

### 실험 단계

```text
Probe → Pilot → Confirm
```

- **Probe**: 학습 없이 구현·계측·방향을 확인한다.
- **Pilot**: 학습 시드 1개로 효과 크기와 실패 모드를 본다.
- **Confirm**: 살아남은 설정만 두 번째 시드로 재현한다.
- 채택은 Confirm을 통과한 뒤에만 한다. 설정당 학습 시드는 기본 최대 2개다.

### 실행 중인 학습을 죽일 때

**먼저 사용자에게 묻는다.** 버그를 발견했더라도 마찬가지다 — 그때까지 나온 체크포인트는
그 자체로 자료이고, "다시 돌리는 게 낫다"는 판단은 사용자 것이다.
예외는 즉시 죽는 크래시 루프뿐이다.

**프로세스는 태그로 특정한다**: `--output_path output/steer/<tag>`.
`tokenhsi/run.py` 같은 공통 문자열로 grep 하면 **모든 학습이 걸린다.**

> 2026-08-19 에 세 번 어겼다. 뷰어를 정리하다 `grep tokenhsi/run.py` 로 전체를 죽여
> 장기 학습 2개가 15,000 / 17,000 iter 에서 재시작됐고, 버그를 찾을 때마다 7개씩 두 번 더
> 죽였다. 체크포인트가 살아 있어 복구는 됐지만 물어봤어야 했다.

### 큐 투입 조건

큐 행에는 다음이 정해져 있어야 한다.

1. 질문: 무엇을 확인하는가.
2. 변경축: 기준 설정에서 무엇 하나가 바뀌는가.
3. 핵심 지표와 보호 지표: 무엇을 개선하고 무엇을 지켜야 하는가.
4. 무효 조건: 어떤 경우 측정을 버리는가.
5. 다음 분기: 성공·실패·애매함에 각각 무엇을 하는가.

### 대조군

- 동일 코드·환경·평가 조건의 최근 대조군이 있으면 재사용한다.
- 코드, 환경, 평가기, 핵심 설정이 바뀌었을 때만 대조군을 다시 실행한다.
- 비교할 때 사용한 대조군 tag를 EXPERIMENTS에 명시한다.

### 지표와 판정

| 구분 | 역할 |
|---|---|
| 핵심 지표 | 실험이 직접 개선하려는 값 |
| 보호 지표 | 개선 과정에서 무너지면 안 되는 값 |
| 진단 지표 | 실패 원인을 설명하는 값 |

결과는 `채택`, `기각`, `보류`, `무효` 중 하나로 기록한다. 설정 불일치·계측 오류는
가설의 실패가 아니라 무효다. `n=1`은 잠정 결과이며 1시그마 안의 차이를 억지로
순위화하지 않는다. 자동 결과는 사실만 담고 최종 판정은 EXPERIMENTS에 적는다.

### 실행 프로필

같이 움직여야 하는 환경·버퍼 값은 하나의 프로필로 취급한다. 측정 근거는
[ENV_SCALING](../docs/ENV_SCALING.md)에 둔다.

| 프로필 | `num_envs` | `envSpacing` | contact pairs | GPU 배치 |
|---|---:|---:|---:|---|
| batch | 4,096 | 5 | 4 M | 최대 2잡 |
| long | 16,384 | 5 | 4 M | 최대 2잡, 공유 후보에서 후순위 |

- `envSpacing=0`과 32,768 env는 사용하지 않는다.
- 새 큐 실행의 기본값은 steer `4096/5/4M`, ma `2048 env × 2 agents/5/4M`,
  lab `4096/5/4M`이다. 행에 명시한 값이 기본값을 덮어쓴다.
- 이미 실행 중인 런은 시작할 때 저장한 env와 생성 cfg를 유지하며 중간에 바꾸지 않는다.

## 평가와 완료

| 트랙 | 자동 최종 평가 | 완료 기준 |
|---|---|---|
| steer | 모든 학습 행 | 평가 결과가 `results.tsv`에 원자적으로 기록됨 |
| ma | `MA_MODE=train/adapt` | 512 env 평가의 `MA_EVAL_SUMMARY rc=0` 기록 |
| lab | 없음 | 프로브 종료 요약 `LAB ... rc=<expected>` 기록 |

- 최종 평가가 필요한 행은 학습 프로세스 종료만으로 완료하지 않는다.
- 미평가 체크포인트는 같은 트랙의 새 학습보다 먼저 배치한다.
- steer 장기 런의 중간 체크포인트도 별도 watcher 없이 같은 큐 데몬이 평가한다.
- 평가는 학습 당시 env sidecar와 생성 cfg를 재사용한다. 설정을 추측하지 않는다.
- 평가가 실패하면 큐 행을 남기고 30분 뒤 재시도하며 AUTO_RESULTS로 옮기지 않는다.
- 예외적으로 평가를 끄거나 켜야 할 때만 `QUEUE_EVAL=0|1`을 쓴다. ma의 `sweep/test`
  모드는 기본적으로 최종 평가를 생략한다.

## GPU 배치

실제 제한값은 `runs/queue/tracks/<track>.json`이 기준이다.
큐와 트랙 설정은 기본 60초마다 다시 읽는다.

### 하드 제한

| 트랙 | 가능 GPU (`gpus`) | 빈 카드 정착 | GPU당 최대 | 트랙 전체 최대 |
|---|---|---:|---:|---:|
| steer | 0–7 | 5분 | 2 | 제한 없음 |
| ma | 0–7 | 5분 | 2 | 제한 없음 |
| lab | 0–7 | 5분 | 2 | 제한 없음 |

- `gpus` 목록을 바꾸면 해당 트랙이 사용할 수 있는 카드를 제한할 수 있다.
- 새 잡의 최소 메모리 여유 26,000 MiB가 없으면 잡 수가 1개여도 제외한다.
- GPU 락을 먼저 확보하고 GPU당 3잡은 자동으로 허용하지 않는다.

### 선택 순서

1. `gpus` 목록 밖 카드와 메모리가 부족한 카드를 제외한다.
2. 정착된 빈 GPU가 있으면 항상 빈 카드를 먼저 사용한다.
3. 빈 카드가 없으면 1잡 GPU 중 사용 메모리가 낮은 카드부터 두 번째 잡을 배치한다.
4. 따라서 long·대형 잡은 웬만하면 1잡으로 남지만, 다른 후보가 없고 메모리가 충분하면 2잡을 허용한다.
5. 모든 카드가 2잡이거나 메모리가 부족하면 기다린다.
6. polling 한 번에는 학습 또는 평가 하나만 시작한다.
7. 자동 최종 평가가 있는 트랙은 미평가 체크포인트를 새 학습보다 먼저 배치한다.

프로세스 수로 실제 잡 수를 세고, 메모리는 별도 상한으로 검사한다. 자동화 밖에서 시작된
잡도 GPU 프로세스 수에 포함한다. 큐 순서는 사용자가 정하며 자동화는 재정렬하지 않는다.

수동 예약은 `scripts/ma/gpu.sh hold <gpu> [count]`, 현재 배치는
`bash scripts/gpumap.sh`로 확인한다.

## 트랙 간 결정

- steer와 ma는 현재 독립적으로 진행한다. 둘 다 기준 설계가 확정된 뒤 토큰을 통합한다.
- 통합 시 steer 단독, ma 단독, 두 토큰 frozen, 두 토큰 fine-tune 순으로 회귀를 확인한다.
- lab 결과는 해당 트랙 설계나 공통 실행 레시피로 승격한 뒤 lab을 다시 비운다.

## 열린 리스크

- steer 장기 학습에서 성공률과 이탈이 반대 방향으로 움직일 수 있다.
- ma 보상 공유가 lazy-agent를 만들 수 있다.
- 두 토큰을 동시에 붙일 때 adapt builder의 단일 `new_extra` 가정을 확장해야 한다.
