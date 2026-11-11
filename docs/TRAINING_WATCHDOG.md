# CUDA timeout 자동 재개

2026-11-09 현재 학습에 `scripts/watch_training.py`를 별도 프로세스로 연결했다.

- 10초마다 원래 PID와 process start identity를 확인한다.
- 활성 학습이 바뀌지 않았고 종료 로그에 CUDA launch timeout이 있을 때만 60초 후 재개한다.
- 최신 번호 체크포인트의 epoch, optimizer 상태, method contract를 CPU에서 검증한다. 손상본은 건너뛰고, 현재 실행에 저장본이 없으면 이전 resume checkpoint를 사용한다.
- 기존 환경·학습 설정을 복사하고 ITER 상한은 무제한으로 유지한다. TensorBoard도 이전 기록에 연결한다.
- 한 시간에 최대 3회 재개한다. 초기화/복원 오류 또는 다른 종료 원인은 자동 재시도하지 않는다.
- 상태: `output/training_watchdog.json`, 로그: `output/training_watchdog.log`.
- 재부팅/전원 종료 후 자동 실행되는 서비스는 아니다. GPU hang으로 프로세스가 살아 있는 경우에도 강제로 종료하지 않는다.

## 수동 중단

학습 중단/평가/설정 변경 전 아래 명령으로 자동 재개부터 해제한다.

```bash
touch output/training_watchdog.stop
```

그런 다음 해당 학습만 정상 중단하고 `output/active_training.json`의 status를 `stopped_by_user` 등으로 기록한다. 보통 수동 종료는 CUDA timeout traceback이 없으므로 자동 재개되지 않지만, stop 파일을 두면 의도를 명확히 보존한다.

다시 감시하려면 기존 watcher 종료를 확인하고 stop 파일을 제거한 뒤, GPU/호스트 프로세스에 접근 가능한 환경에서 watcher를 실행한다. 현재 학습이 실행 중일 때만 감시를 시작할 수 있다.

검증: timeout/수동중단/다른실행/정상종료 분기 및 체크포인트 우선순위 단위 테스트 2개 통과. 현재 학습의 실제 crash를 인위적으로 발생시키지는 않았다. 재개 helper는 18,500부터 실제 복원 및 ITER 진행/TensorBoard 연결이 검증된 경로를 사용한다.
