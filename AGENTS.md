# CVPR2027 — Codex 진입점

작업 전에 반드시 [공통 운영 규칙](docs/OPERATIONS.md)을 읽고 따른다.
현재 상태와 활성 문서 경로도 그 문서를 기준으로 판단한다.

## 작업별 읽기 순서

- 전체 연구 맥락: `METHOD_V2.md` → `REPOS.md` → `plan/PLAN.md`
- steer: `plan/PLAN_steer.md` → 자동 결과 → `experiments/EXPERIMENTS_steer.md` → `TokenHSI-steer/CHANGELOG.md`
- ma: `plan/PLAN_ma.md` → 자동 결과 → `experiments/EXPERIMENTS_ma.md` → `TokenHSI-ma/CHANGELOG.md`
- 실험 실행·평가: 위 문서에 더해 `docs/PITFALLS.md`를 먼저 읽는다.
- GUI·카메라·녹화: `docs/PITFALLS.md`의 Vulkan GPU 사고 항목을 먼저 읽고
  가드가 연결된 `scripts/{steer,ma,masteer,mastop}/view.sh`, 트랙 `record.sh`,
  또는 `scripts/tokenhsi_gui.sh`만 사용한다.

## 즉시 지킬 것

- 코드 수정은 /home/hwanhee/koo_cvpr 레포 내에서만 해야 한다.
- CUDA_VISIBLE_DEVICES 6,7번만 사용할것. 다른 GPU 넘버 사용을 사전에 막을것.
- Isaac Gym GUI/Vulkan 실행에서 `CUDA_VISIBLE_DEVICES`만으로 GPU가 제한된다고
  가정하지 않는다. 실행 전에 `scripts/vulkan_gpu_guard.py <6|7>`의 UUID 검증을 통과해야 한다.
- `DRI_PRIME=pci-...!`는 금지하며 숫자 selector `DRI_PRIME=6!|7!`만 쓴다.
  `--graphics_device_id 0`은 가드가 Vulkan 장치를 한 장으로 제한한 뒤에만 허용한다.
- `TokenHSI/` baseline과 외부 소스·빌드 트리는 수정하거나 옮기지 않는다.
- 실행 중인 학습·평가·watcher·큐 데몬은 사용자 승인 없이 종료하지 않는다.
- 역할이 끝난 구 경로를 다시 만들거나 활성 기준으로 사용하지 않는다.
- 큐의 1·2·4열은 사용자 소유다. 자동화는 3열 상태/GPU와 완료 행만 관리하며 다음 실험을 만들지 않는다.
- 답변은 한국어로 한다.
