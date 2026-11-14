# 변경 기록

## 2026-11-12

### 단일 프롬프트의 Enable·Coupled 대기 규칙 분리 및 2H 재평가

- 사용자 요청으로 공통 대기 단축 목표를 제거하고 Enable에는 대기 최소화 목표 없음·불확실성/명시된 안정화 조건에 따른 추가 대기, Coupled에는 기존 T=max(a_i), release_i=T-a_i 및 추가 공통/개별 지연 금지를 명시. 단일/배치 system·단일 user template·manifest 일치. 행동 예시 지시 유지 검증.
- fewshot_2h_type_rules_w4_20261112_185737 시작. 동일 Mixed2H192장면(seed1/6/7×64), 신규 응답 worker4 후 GPU6·7 기존 MPS 자동 연결·legacy 평가/검증. 2H 이후 확장 없음.
- 실행 manifest의 worker4·mixed_2h192장면 및 최신 프롬프트 바이트 일치 확인. MPS 제어 및 기존 프로세스 종료 없음.

### Coupled88.63%·Enable28.02% 직전 Few-shot 프롬프트 복원

- 사용자 정정에 따라 fewshot_2h_enable_wait_w4_20261112_175052 저장 system prompt를 활성 fewshot에 바이트 일치 복원. 단일 user template·배치 system·manifest도 불확실성 대기 추가 전 문구로 되돌림.
- 행동 예시10개와 API worker4 유지. 저장 프롬프트 바이트 일치 및 Enable 전용 대기 허용 확인. API/eval 재실행 없음.

### Enable 불확실성 기반 추가 대기 복원 및 Few-shot 2H 재실행

- 사용자 요청으로 Enable follower에만 대기 최소화 목표 없음·예측 불확실성/명시된 안정화 조건에 근거한 추가 대기·휴리스틱 가정 명시를 복원. 단일/배치 system, 단일 user template, manifest 일치. Coupled 본문과 10개 행동 예시 지시는 직전 실행과 동일함을 검증.
- fewshot_2h_enable_uncertainty_w4_20261112_182521 시작: 같은 Mixed2H seed1/6/7×64=192장면, 신규 응답, worker4. 이후 GPU6·7 기존 MPS 자동 연결/legacy 평가·검증. 2H 이후 확장 없음.
- 실행 manifest의 mixed_2h192장면·worker4·활성 프롬프트 일치 확인. 기존 학습/평가 종료 및 MPS 제어 없음.

### Few-shot API worker 기본 4개 및 Enable 대기 문구 2H 재실행

- 사용자 요청으로 replay_fewshot_api, run_fewshot_2h_validation, run_fewshot_full의 API worker 기본값을 16→4로 변경. Enable follower 전용 대기 허용을 manifest 설명에도 반영.
- fewshot_2h_enable_wait_w4_20261112_175052 시작: 동일 Mixed2H seed1/6/7×64=192장면, 최신 프롬프트로 신규 호출, --workers 4 명시. 이후 GPU6·7 기존 MPS 자동 연결 및 legacy 평가·검증. 2H 이후 확장 없음.
- Python 구문·worker 기본값·Enable 전용 문구 CPU 확인 완료. 실행 manifest의 worker4·mixed_2h192장면·활성 system prompt 일치 확인. 기존 프로세스 및 MPS 제어 없음.

### 추가 대기 허용을 Enable 후행 에이전트로 명시 제한

- 사용자 요청으로 활성 fewshot 단일·배치 system 및 단일 user template에 근거 있는 추가 대기 허용은 Enable follower에만 적용한다고 명시. Coupled는 기존 도착 동기화 규칙, Independent는 0초 출발을 따르도록 함께 명시.
- 프롬프트 문구와 단일·배치 Coupled 본문 일치 확인. API 호출·GPU 평가 실행 없음.

### 근거 있는 대기 허용 프롬프트 Mixed2H 재평가

- 사용자 요청으로 fewshot_2h_reasoned_wait_20261112_171736 시작. 동일 Mixed2H seed1/6/7×64=192 초기장면, 수치 간격 없이 선행 이후+근거 있는 추가 대기 허용. 신규 API worker16 후 GPU6·7 legacy 평가.
- 이번 실행은2H 검증까지이며 나머지 전체 실행 없음. 이전 응답 재사용 없음, ETA 및429/timeout 분리 기록 유지.

### Enable 숫자 간격 제거·근거 있는 대기 허용

- 사용자 요청으로 활성 fewshot 및 fewshot_batch에서0.2초 조건 제거, release+q>p 유지. 장면/과제상 이유가 있을 때 추가 초기 대기를 허용하고 follower basis에 이유를 설명하도록 명시.
- Objective/본문/Output/user template/manifest/향후 batch export metadata 일치. Coupled 규칙과 zero-shot 해시 불변 확인. 이전0.2초 입력/결과 snapshot 보존, API/eval 재시작 없음.

### 0.2초 전체 확장 사용자 요청 중단

- full_gap02_20261112_165548 API 단계에서 사용자 요청으로 CPU continuation500087/supervisor541057/API541058만 중단. 명령줄 및 NVIDIA FD 없음 확인. 전체 GPU평가 시작 전; 완료된2H결과/응답 보존.
- 상태 stopped_by_user 및 저장/미응답 수 기록. 기존 학습/MPS 조작 없음.

### 1920장면 단일 요청 입력 TXT 및 배치 프롬프트 분리

- export_batch_llm_input.py로 Mixed2/4/8/16+Table2 1920장면/8640agent를 원본 initial request에서 수집. 8H R3/16H R7.5 검증, environment_spec1개와 예시10개 공통 참조.
- prompts/fewshot_batch 별도 system/user template: record_id별 독립 scheduling 및 results wrapper. Enable0.2초 유지. 기존 프롬프트/실행 중 API 수정 없음.
- output_etc/check/batch_input_1920_gap02_20261112에 실제 user_input_1920.txt(13,144,214자), system_prompt.txt, manifest/원본SHA index/사용법 저장. 1920 scene roundtrip·원본 일치 검증. 단일호출 모델 한도 미검증, API호출 없음.

### 후속 전체 API worker16 및 제한 오류 분리

- 사용자 요청으로 run_fewshot_full 기본worker16/CLI override 추가. 이미 실행 중2H worker4는 재전송 없이 유지하고 기존 continuation이 호출할 전체 단계에서16 사용(나머지1728건).
- API status에 api_workers/api_rate_limited/api_timeouts/api_transport_errors/last_api_error 추가. 429·타임아웃·전송오류는 api_alert.json/로그에도 저장, 자동재시도 없음.48테스트 통과.

### 0.2초 버전 전체 평가 자동 연결

- 사용자가 전체 실행을 재확인하여 continue_gap02_full.py 시작. 진행 중2H 완료 후 승인 대기 없이 Mixed2/4/8/16+Table2 총1920장면으로 자동 확장. 수정 프롬프트 유지,8H R3/16H R7.5.
- 기존2H192응답 재사용, 남은1728신규 호출. --reuse-eval은 동일 request/raw응답·초기reference·checkpoint·freeze 확인 후 검증된2H 물리결과를 복사하고 CPU 재검증, 나머지27 GPU평가 실행. 원시 기록 보존. 테스트48개 통과.

### Enable0.2초 프롬프트 Mixed2H 재평가 시작

- 사용자 요청으로 fewshot_2h_gap02_20261112_165256 실행. 같은192초기장면에 수정된 few-shot 프롬프트로 신규 호출(동시4), 후속 GPU6·7 legacy 물리평가. 이전 응답 재사용/보정 없음.
- ETA 포함 단일 status 저장 경로 사용. 2H 결과 검토 전 나머지 규모 자동 시작 없음.

### Few-shot Enable 최소 예측 이벤트 간격0.2초

- 사용자 요청으로 fewshot system/user template의 Enable 조건을 release+q >= p+0.2로 변경. 출발 시각 간격이 아닌 예측 상호작용/선행완료 사이 최소 간격이며 기존 더 큰 간격 축소 의무 없음.
- Objective/본문/Output 규칙을 일치시킴. zero-shot system 해시 및 Coupled 규칙 불변 확인. 호출/평가 재실행 및 기존 응답 보정 없음.

### ETA 표시 깜박임 수정

- 원인은 기존 메모리의 supervisor가 ETA 없는 status를 덮어쓰고 호환 ETA watcher가 다시 추가하는 이중 작성. 해당 CPU supervisor/watcher만 신원·NVIDIA FD 없음 확인 후 교체, GPU 평가 PID470762 및 자식은 유지.
- monitor_existing_2h.py가 상위 status의 단일 작성자로 evaluator 상태/고정 ETA 필드를 반영. 신규 실행은 이미 native ETA 저장 경로 사용. 기존 호환 watcher 재사용하지 않음.

### Few-shot 상태에 단계별 ETA 추가

- progress.StageETA: eta_seconds/eta_at/eta_scope/eta_basis/eta_updated_at 기록. API 완료율에서 재사용 응답을 제외하고, eval은 검증 완료 작업 처리율로 추정. 자료 부족·중단은 null, 완료는0. API ETA는 후속GPU평가를 포함하지 않음.
- API/eval 및 full/2h supervisor 저장 경로에 연결. 실행 중2h는 재시작 없이 별도 status ETA 감시기로 적용. 회귀48개 통과.

### 전체 실행 중단 후 Mixed2H 우선 검증

- 사용자 요청으로 full resume CPU supervisor425079/API425083 중단(NVIDIA FD 없음 확인). 완료응답69개 보존, GPU평가 미시작.
- fewshot_2h_validation_20261112_163406: 동일 프롬프트/예시로 Mixed2H seed1·6·7×64=192장면만 API 후 GPU6·7 물리평가. 3개 검증·요약 후 ready_for_review로 종료하고 나머지 자동 시작 없음. 기존69개는 후속 전체 실행에서 재사용 가능.

### Few-shot 전체 평가 동일 프롬프트로 재개

- 사용자 요청으로 fewshot_full_radius_resume_20261112_162808 시작. --reuse-api/--reuse-from으로 완료 raw응답42개를 초기입력·프롬프트·모델·reasoning 일치 확인 후 재사용, 미응답1878개만 신규 호출. 중단 기록 보존.
- few-shot 프롬프트/예시/반경/legacy freeze/동시4/API 후 GPU6·7 평가 유지. 회귀 테스트46개 통과. 완료 여부는 새 status 기준.

### Few-shot 전체 실행 사용자 요청 중단

- fewshot_full_radius_20261112_161633 API 단계에서 사용자 요청으로 중단. /proc 명령줄과 NVIDIA 장치 FD 없음 확인 후 해당 CPU supervisor392145/API392146만 종료. GPU 평가 시작 전, 학습/MPS 조작 없음.
- 응답 및 미완료 입력 보존, status를 stopped_by_user로 기록. 프롬프트 미수정. 중단 시점 수치는 해당 status 참조.

### Few-shot 전체 평가 시작 — 고정 반경8H3m/16H7.5m

- run_fewshot_full.py → replay_fewshot_api.py → evaluate_fewshot_api.py → protocol_radius_rescore.py 분리 실행. 기존 baseline 보존, 실제 AT5/PUSH5 예시로1920회 호출(동시4), legacy freeze/GPU6·7/64env×seed1·6·7/900step. Independent 전용 제외,30설정.
- 2H/4H/Table2는 eval_protocol_v1_legacy_20261111_165027,8H/16H는 mixed8_16_radius3_7p5_20261112_082344 초기 snapshot 재사용. 전체1920입력과 snapshot 일치 및 반경6묶음 확인. 최신 공용 reference_metrics 코드 동일, 프로토콜v1.1.0으로 반경/Time Succ-All/seed std 집계.
- fewshot matching_initial_request는 demonstrations와 timing_calibration 설명만 허용하고 다른 환경/초기 장면 변경을 거부. 기존 물리 evaluator 경계에 적용. 테스트46개 통과.
- 실행 output_etc/check/fewshot_full_radius_20261112_161633 (API→GPU eval 자동 연결). 실행/실패 기록과 code_snapshot 보존. 완료 여부는 status.json 기준.

### Few-shot 단일 장면 GPU 뷰어

- view_fewshot_single.py와 별도 runtime entry로 저장된 mixed4h seed1 env0를 1환경·4명으로 복원. fewshot_single_scene_20261112_test01의 검증된 응답 사용, API 재호출 없음. Stage1Policy native 실행 및 legacy 초기 freeze 유지.
- GPU6/noVNC6086 실행 viewer_fewshot_single_20261112_160748. 첫 정책 입력 최대 오차8.35e-7, 첫 step root 이동0.0418m로 검증 통과. Space 재생/일시정지, Q 정상 종료. 기존 2명 뷰어 코드 보존.

### Few-shot API 연결 및 초기 장면 1건 호출

- CLI --fewshot으로 manifest의 실제 AT5/PUSH5 자료를 붙이고 calibration 설명을 갱신. demonstrations 유무로 프롬프트를 분리하며 baseline과 current_scene 보존.
- tokenhsi_koo에서 기존 테스트44개 통과. 실제 자료10개와 baseline/fewshot 선택, 원본 current_scene 불변을 사전 확인.
- mixed_4h/general_seed1/env_0000 초기 장면으로 medium 실제 호출1회, 자동 재시도 없음. output_etc/check/fewshot_single_scene_20261112_test01에 입력/응답/usage 보관, 엄격 응답 검증 통과. A01/A02/A03/A04 release=0/0.90/1.70/0초, coupled 예측 readiness=4.60초. 물리 eval/전체 API 배치 실행 없음.

### Few-shot 샘플 거리·AT 크기를 eval 범위에 정렬

- 기존1인 샘플의 AT 운반거리 최대13.31m/PUSH목표거리9.50m 확인. 별도 behavior_eval_geometry/record_eval_geometry_entry/collect_eval_geometry_stage1 경로로 AT거리1~5m×2·Enable공통크기, PUSH거리2~10/2~5m·뒤쪽·로컬+Y 적용. 1인base PUSH크기는 유지하고 다인환경과 차이를 명시.
- GPU7,seed11214,각32개×900step, native Stage1 exact. 새10개를 fewshot_examples_single_stage1_eval_geometry_20261112_154050에 생성. 전체거리 범위·이벤트 최초프레임·AT float64 판정·baseline보존 검증. 이전자료 보존, fewshot manifest/README를 새예시로 갱신.

### 1인 Stage1 AT/PUSH few-shot 기록10개 준비

- 사용자 요청에 맞춰 fewshot을 Enable/Coupled 스케줄 예시에서 단독 AT 완료5개/PUSH 접촉5개 행동 참고로 변경. 기존 baseline·OUTPUT 규칙 보존 검증.
- collect_single_stage1.py와 record_single_stage1_entry.py로 native Stage1 26k 1agent×32env/task×900step, seed11213 GPU7 실행. freeze/API/Stage2 없음, weights 및 normalization exact.
- export_single_stage1_fewshot.py로 성공 rollout에서 거리 분위 기반5개씩 선택, full-resolution 이벤트와 sparse trajectory를 대조 검증. source single_stage1_fewshot_20261112_153316, 예시 fewshot_examples_single_stage1_fewshot_20261112_153316. 초기 힘 null, 접촉 substep 해상도와 AT 완료 의미를 명시.

### Few-shot 10개 프롬프트 독립 추가

- prompts/fewshot에 system/user template/입력 가이드 및 기준 snapshot 생성. 기존 문체·release 공식·출력 스키마 보존, 실제 성공 기록10개 활용/이벤트 라벨/대기 제외/선택 편향/FSM 접촉 freeze와의 차이 지시 추가.
- 기존 두 프롬프트 바이트와 system OUTPUT 이하 동일 확인. 현재 API는 demonstrations를 허용하지 않으므로 아직 자동 연결하지 않았으며 그 제한을 README에 명시. API/eval 실행 없음.

### LLM 입출력 명세 문서 작성

- LLM_INPUT_OUTPUT.md에 메시지 구조·모든 입력/출력 필드·실제 Mixed4 seed1 env0 입력/응답·당시 system 원문·출발 시각 해석·저장 파일을 정리. 현재 원복 프롬프트와 예시 당시 실험 프롬프트를 구분. JSON 원문 동등성과 문서 링크 확인. API 추가 호출 없음.

### 실험용 Coupled 문단 원복

- 사용자 요청으로 system.txt에서 상대 시간 비교·영역/정렬·물체 이동 고려 추가 문단 제거. 기존 프롬프트 snapshot과 일치 확인. 완료된 API/eval의 프롬프트 사본과 결과는 보존.

### SCT 단독 2H 선행 실패 viewer

- 원본 SCT mixed2 seed1에서 선행 미성공20장면 초기 root/DOF/관측/물체를 단일환경으로 복원. view_sct_single.py와 runtime entry에 freeze 없는 NoScheduler 적용. GPU6, 모델1회 로딩, N전환, API0. 원본 영상 재생이 아닌 초기조건 기반 재시뮬레이션. source env 목록은 sct_2h_view_data_20261112/source.json.

### Enable 조기 접근 실패 후보 분석

- analyze_enable_early_arrival.py로 기존 Mixed2/4 및 Table2 Enable의 release 이후 유효 root 접근과 물체 목표 미준비를 비교. 실제 시도 이벤트 부재로 실패 원인 확정 대신 후보 비율을 기록. 반경0.3/0.5/0.75m 민감도, 선행 나중 완료/미완료, 낙상을 보존. enable_early_arrival_20261112/report.md 및 raw rows에 저장. GPU/API 추가 실행 없음.

### 상대 시간 비교 프롬프트 4/8H 물리 평가 시작

- evaluate_selected_api.py로 API manifest의6개 job/384장면만 선택. 각 저장 request를 해당 source reference request와 JSON 대조 후 source별 초기 장면을 연결. 4H 기존배치,8H 최소3m간격 유지. Stage1 26k/legacy freeze/900step, GPU6·7 사용, API추가0.
- 응답383개 strict통과,1개 episode event coverage 오류이나384개 모두 executable. 기존 execution-only 기준으로 원본 release 적용. 결과 eval_coupled_relative_4h8h_20261112; 완료는 status.json 기준.


### Coupled 접촉 시차와 측방 오차 분석

- 기존 Table2 576trial을 analyze_coupled_contact_timing.py로 분석. 12cm 양손 표면 접근 대리 접촉, 미관측 검열, 목표 방향 측방/전후 오차와 낙상을 기록. summary.json/trials.json/report.md는 coupled_contact_timing_20261112에 저장. 3/4명에서 시차-측방 rho0.33이나 실패 subset 관계는 약함. 인과 및 선택 효과 한계를 보고서에 명시. GPU/API 추가 실행 없음.

### Coupled 상대 시간 비교 프롬프트와 4/8H API 재호출

- Coupled 준비 시간의 일관된 상대 비교·배정 영역/heading/회전/정렬·선도 참여자에 의한 물체 이동 가능성 문구 추가. 최소/최대 대기나 안전 마진 규칙은 추가하지 않음. 이전 프롬프트는 prompt_snapshots에 보존.
- replay_saved_api.py에 --settings, --additional-source 추가. 설정별 request를 선택하고 추가 source의 동일 경로가 우선하며 입력 해시와 출처 목록을 manifest에 보존. 전체 실행 기본값은 기존 1920개 유지.
- 4H 기존 suite와 8H 최신 3m 간격 장면, 각각64env×seed1/6/7 총384 API 제출. gpt-6-luna, reasoning medium,16병렬,재시도0, GPU eval0. 상태/원문/검증 결과는 api_coupled_relative_4h8h_20261112_1345에 별도 저장.


### Coupled N 전환 복원 오류와 불필요한 점 표시 수정

- 접촉 영역 복원에서 랜덤 `assign_regions(groups)`를 사용해 env1의 두 agent 영역이 뒤바뀌고 관측 일치 검사에서 종료되던 원인을 수정. 원본 평가와 동일하게 초기 root·box 좌표에 `ordered` 배정을 적용하고 비-PUSH member 수는 1로 복원.
- 뷰어 초기화에서 생성된 구형 goal marker를 멀리 옮겨 숨김. 정책 목표·관측은 변경하지 않는다. API 호출 없이 저장 응답 유지.
- GPU6에서 env0→1→2를 N으로 연속 전환 검증. 초기 관측 오차 최대4.77e-6으로 통과, 첫 스텝 root/rigid body 일치. env2 이미지에서 구형 점 제거 확인. 출력 `viewer_persistent_single_20261112_134012`.


### 저장 Coupled 2명 단일 환경 viewer 지원

- `--setting coupled_2`로 기존 평가 초기상태·응답을 읽는다. Coupled의 그룹 인원·접촉 영역을 복원하고 선행 없는 장면 색상 처리 지원. 환경/모델을 장면마다 재생성하지 않으며 API 호출 없음.


### 저장된 평가 장면을 단일 환경에서 연속 조회

- `view_saved_single.py`와 `teacher_runtime/view_saved_single_entry.py` 추가. 실제 1환경·2명으로 AT→SIT 저장 장면을 복원하며, N 전환에도 같은 프로세스·시뮬레이터·Stage1 모델·MotionLib를 재사용한다. API 호출 없음. Space 일시정지, N 다음, Q 정상 종료.
- source64 장면의 물체 자산을 시작 시 준비하고 현재 작업 물체만 사용한다. root/DOF/물체/관계/관측·기존 release schedule 복원. 원래 평가 영상의 정확한 재생이 아닌 단일 환경 진단용 재실행이다.
- GPU6 검증: env0→env1 전환 후 모델 1회 로딩, API0 유지. 관측 최대 복원 오차 각각 8.35e-7/1.85e-6. 첫 스텝 root 이동 0.0354/0.0381m, root와 rigid body 위치 일치. 초기 프레임 이미지 확인. 이전 64환경 viewer는 Q로 정상 종료 후 대체.
- `USAGE.md` 15절에 현재 권장 실행법과 재현 한계를 기록.

## 2026-11-11

### 16명 예시 지시문을 최신 user template에 정렬

- prompts/user_example_16_agents.txt의 지시문을 현재 user_template.txt와 동일하게 교체. 명목 속도·준비시간 가정 강제 문장을 제거하고 초기 frozen 대기 제외, 낮은confidence, 불필요한 대기 회피, 후행 release 근거 및 condition validity 설명을 최신template대로 반영.
- 장면·event/glossary 데이터 부분 byte-equivalent 유지, JSON16agent 유효 확인. 실제 API는 system.txt+request JSON을 사용하므로 실행 중 실험 프롬프트/응답에는 영향 없음. USAGE에 예시 지시문 동기화 명시.


### 폴더 사용 설명서 정리

- USAGE.md에 API 설정·messages 구성·응답 검증·raw 저장·capture/API/eval/CPU후처리 흐름·실행 명령·프로토콜/Time Succ.All/seed통계·상태 확인·GPU/MPS 안전·최신 결과 경로 정리. 전체1920장면과 간격변형384장면의 실행기 고정 가정을 명시.
- README.md를 최신 진입점으로 정리하고 과거 초기화 미해결/rollout 미검증/status ok 관련 설명을 제거. 일반 CLI와 경로고정 wrapper, legacy주평가와snapshot실험을 구분.
- 문서의 실제 CLI 옵션과 경로를 코드로 대조, 두 문서의 로컬 링크 존재 검사 통과. 문서 변경만 수행하며 실행 중인 간격 변형 작업·키·프롬프트·평가 코드는 변경하지 않음.


### Mixed8/16 그룹 간격 변형 및 Table2 Succ./All 시간

- 사용자 요청으로 SCT_GROUP_GAP_VARIANT=8h3m_16h4m 선택 시 그룹중심 원 배치의 최소간격을8H3m/16H4m로 설정. 기본값5m 보존. 샘플러·eval 최소간격 검사·metadata를 동일 helper에 연결. CPU실제 배치 함수에서 최소2.999999/3.999992m 검증, 기본5m 보존 및 기존CPU44 tests 통과.
- run_group_gap_eval.py로 8/16H만 각64env×seed1/6/7의6회384새장면 capture→API→legacyfreeze900step→CPU검증·집계 제출. 변경된장면에 이전 스케줄을 재사용하지 않음. GPU6=8H/GPU7=16H, API16병렬 재시도없음. 출력 mixed8_16_gap3_4_20261111_231041; 완료는status.json 기준. 2/4H 및Table2 GPU 재실행 없음.
- 기존Table2 first_success만으로 Time Succ./All 재집계. 성공은step/30, 실패는최대30초 부여. 행동별동일가중→seed평균±표본표준편차 보존. 18run의All 값을 성공수×Succ평균+실패수×30의 독립식으로검산. Enable통합 Succ12.62±0.78 /All23.88±1.07초.
- protocol_v1_summary.json의runs/groups/overall에completion_time_all_s 및정의기록, 두report와생성기에 Time(Succ./All)표시 반영. raw기록/기존시간값은보존, protocol_v1.json 후처리 필드만추가. ksh/MPS/기존작업 변경없음.


### Coupled 전체 Avg.의 seed 표준편차 추가

- 인원2/3/4 동일가중 평균을 seed별로 먼저 구한 뒤 seed평균·표본표준편차(ddof=1) 계산. protocol_v1_summary.json의 coupled_overall에 SR/tcoop/Jerk seed 통계·seed별 중간값·집계 정의 추가. pooled 값은 별도 유지.
- report.md/protocol_v1_report.md에 Coupled 전체 Avg. 행 추가, 생성기도 동일 반영. 현재 tcoop44.64±1.99%, Jerk22.52±0.86m/s³. 기존 실행/원시 기록 재계산 없이 저장된seed값으로 산출. 독립 합산 검산 및 두 report 내용 일치 확인.


### report.md를 최신 프로토콜 기본 보고서로 통일

- 기존 Table1/설정별 Table2 형식을 유지하면서 report.md에 최신 프로토콜 SR·Enable Time·12cm t_coop·Jerk seed평균±표본표준편차와 통합요약표 출력. 구 지표 보고서는 report_legacy_metrics.md에 보존. protocol_v1_report.md도 동일 내용으로 생성하여 경로에 따른 혼동 제거.
- 생성기와 완료 결과 파일 모두 수정. 두 최신 report 내용 exact, 새 지표키/Enable 통합값 확인. 평가·원시 기록·집계 JSON은 변경하지 않음.


### Coupled 보고서를 seed 평균±표준편차로 표시

- 사용자 요청에 따라 SR/t_coop/Jerk 모두 seed1·6·7 평균±표본표준편차(ddof=1)로 표시. pooled 평균에 seed표준편차를 붙이지 않고 기존 저장된 seed 통계를 사용. Coupled4 유효191trial로 pooled/seed 평균이 달라 t_coop34.35, Jerk21.50으로 표기. 원시 지표·pooled JSON은 보존. 재시뮬레이션 없이 보고서와 생성기 수정.


### 최신 프로토콜 보고서 Enable 표 표시 수정

- Enable 집계는 protocol_v1_summary.json에 있었으나 보고서에 JSON 한 줄로 출력하던 문제를 수정. 행동별/통합 SR·성공 완료 시간을 표로 표시하고 생성기도 동일 수정. 평가·원시 기록·집계값 변경 없음.


### 공용 cvpr2027-evaluation-v1.0.0 기준 재평가

- 사용자 지정 /home/hwanhee/ksh/evaluation_protocol/README.md와 implementations/checks를 읽기 전용 확인하고 protocol_v1/에 사본·SHA256 보존. 현재 sampler·공간 SR/Error 구현8개가 원본과 byte-exact임을 확인. 기존64환경/seed1·6·7/Independent전용 제외 범위를 보존하며 정식512환경·완전한Table1이라고 표시하지 않음.
- protocol_v1_rescore.py 추가: Coupled 전체900frame 양손12cm 근접(생존 mask 추가 없음), reset 중심20cm 최초 변위, 원본 start/end counter 차이·float32 clip/NaN 처리, 둘레64점·dt0.0333·원본 순차 차분 Jerk와0제외 집계. trial 합산 인원별평균 및 인원동일가중, seed평균±표본표준편차 분리. Enable Time은 first_success/30, 행동동일가중→seed평균±표본표준편차로 산출. 과거 table2.json의6/8/12cm 구간 지표는 legacy 기록으로 보존, 새 지표는 protocol_v1*.json/npz/report로 구분.
- SCT_PROTOCOL_VERSION 활성 시 실제 첫 정책 입력과 indexed reset 관측(대기 agent 속도0) exact 비교 및 PT/JSON 저장. 초기화/관측처리를 바꿔 수치를 보정하지 않고 검증만 추가. 새 evaluator는 초기 입력 검증 실패 시 해당 GPU lane의 후속 작업 제출 중단.
- 공용 계산 예제34개, CPU44 tests 통과. 기존Coupled2/seed1의 별도 smoke64trial로 최초 성공 exact·벡터 contact와공용surface 계산 exact 확인. 새 GPU실행 Mixed2 seed1/6 첫정책입력 오차0 확인.
- run_protocol_v1_eval.py로 기존 API_no_unnecessary_wait 응답 재사용·legacy freeze·Stage1 26k·GPU6/7·30회64env900step 재평가 시작. API 추가0, source/기존결과/ksh/MPS 변경 없음. pipeline protocol_v1_pipeline_20261111_165027, eval eval_protocol_v1_legacy_20261111_165027. 진행률은 pipeline/status.json, 최종신규지표는 eval/protocol_v1_report.md 및 protocol_v1_summary.json. 완료는 상태파일 기준.


### 실행 중 파이프라인 진행률 표시 보완

- 상위 status가 phase만 기록하던 문제를 확인. API subprocess는 정상 진행 중이며 중단/재시작 없이 mirror_pipeline_progress.py로 하위 API/평가 상태를5초마다 상위에 표시한다.
- api_done/api_total/api_progress_percent, eval_done/eval_total/eval_verified/eval_progress_percent 및 running을 추가. 실행 제어/스케줄/모델에 영향 없음. 파이프라인 완료 또는 실패 시 표시 프로세스도 정상 종료.
- 대상 wait_prompt_pipeline_20261111_151400/status.json에서 실제 카운터 표시 확인.


### 불필요한 대기 회피만 유지한 프롬프트 재실험

- 사용자 승인에 따라 system/user_template/user_example의 추가 대기 권장·불확실성 여유 허용 문구를 제거. 목표에 Successful coordination is the primary objective / Avoid unnecessarily long initial waiting 유지. earliest release 비강제 문장도 제거하되 최소 대기 최적화·고정 여유시간·불확실성 대기 금지 같은 새 방향성은 추가하지 않음. Enable strict-after 및 Coupled 식은 그대로 유지.
- CPU44 tests 통과. run_wait_prompt_experiment.py로 기존 1920장면 API 재호출(현재 .env,16병렬) 후 evaluate_saved_api.py --freeze legacy 전체30회·64env·900step·GPU6/7 자동 실행 연결. 기존 응답/평가 보존.
- 상태 output_etc/check/wait_prompt_pipeline_20261111_151400/status.json, API api_no_unnecessary_wait_20261111_151400, 평가 eval_no_unnecessary_wait_legacy_20261111_151400. 완료 여부는 상태 파일 기준.


### 原本FSM Mixed16 선행 실패 궤적 분류

- analyze_fsm_predecessor_failures.py로 원본 FSM 512환경×seed1/3/6, 선행4019명 중 region ever 실패1719명을 CPU 분석. 낙상 전/종료까지 물건 중심 상승0.15m, root-물건 XY1.2m 접근, 거리증가2m+root 이동3m 이탈 기준을 공개한다. 파지/contact 실측으로 해석하지 않음.
- 상호 배타 분류: 낙상 없이 들림 징후 없이 이탈808, 낙상 직전 사람 근접 없음286, 근접 있음40, 들림 징후 있으나 목표 실패517, 접근했으나 들림 없음59, 접근/들림 없음9. 실패 선행 중 낙상326명. 낙상 전0.5초 XY0.8m 근접은 충돌 확정이 아니며 물체/지면 충돌은 판정하지 못한다.
- 초기부터 물건이 바닥에서0.15m 이상 떠 있었던 실패 사례0건 확인. 전체 선행 낙상799명과 실패 중 낙상326명은 다른 집계이며 성공 후 낙상은 이번 분류에서 제외.
- 결과 output_etc/check/fsm_predecessor_failure_analysis_20261111/{summary.json,failures.csv}, 범주별 예시3건 포함. 원본은 읽기 전용, GPU/API 실행 없음.


### 사용자 지정 기존 freeze로 soft wait 전체 재평가

- 사용자 정정에 따라 evaluate_saved_api.py에 --freeze legacy/snapshot_resume 선택 옵션을 추가하고 기본값을 legacy로 변경. legacy 코드 자체는 수정하지 않음. 각 결과에 freeze_selection.json으로 audit 일치 검사 기록.
- 같은 api_replay_soft_wait_20261111_120506의 1,920개 새 스케줄·같은 초기 reference·Stage1 26k로 기존 legacy freeze 평가30회 재제출. GPU6/7, 64환경·900step. API 재호출0이며 snapshot 평가 결과 보존.
- CPU44 tests 및 구문 검사 통과. 실행 output_etc/check/eval_soft_wait_legacy_20261111_135137, 완료 여부는 status.json 기준. 기존 학습/MPS/ksh 변경 없음.


### Soft wait API 응답 전체 시뮬레이션 평가 제출

- evaluate_saved_api.py로 api_replay_soft_wait_20261111_120506의 1,920개 새 스케줄을 평가. 기존 reference를 새 출력 트리에서 참조하여 capture/API 반복 없이 Mixed2/4/8/16 + Table2 6설정 × seed1/6/7 총30회, 각64환경·900step을 GPU6/7에 순차 제출한다.
- Stage1 26k + snapshot_resume freeze. 실행 가능한 검증 실패3건도 원문 그대로 포함하며 스케줄 수정 없음. 기존 전체 결과는 legacy freeze이므로 프롬프트 효과만의 통제 비교라고 주장하지 않음.
- 각 실행 후 초기 상태·실제 release tick·CPU rescore·Table2 replay 및 snapshot 속도복원 오차0 검증, 전체 완료 후 summary.json/report.md 자동 집계. 소스 해시 변경 시 다음 작업 제출 거부.
- 실행 output_etc/check/eval_soft_wait_snapshot_20261111_131017. 실제 완료/실패 상태는 status.json 기준. GPU 허용6/7, ksh 쓰기/MPS 제어/기존 프로세스 종료 없음.


### 초기 장면 재사용·soft wait 프롬프트 API 재호출

- replay_saved_api.py 추가. 기존 full_mixed_table2_env64_20261111_003631의 1,920 request를 검증·그대로 재사용하고 현재 system prompt로 재호출한다. 물리 초기화/rollout/GPU 작업 없이 API 전용이며 기존 응답은 보존한다.
- gpt-6-luna, reasoning effort medium, 최대16병렬, timeout180초, 자동재시도0. 현재 prompt 원문·SHA256·장면별 request 해시·raw/validated response·실패 원인·진행 상태를 별도 저장. 호출 중 prompt 변경 시 HTTP 전송 전에 해당 호출 거부.
- 실행 output_etc/check/api_replay_soft_wait_20261111_120506. 완료 여부와 실패 수는 해당 status.json 기준이며 시뮬레이션 평가는 아직 제출하지 않음.


### FSM vs LLM 다른 16인 Mixed 장면 paired 비교

- compare_fsm_llm_episodes.py로 기존 env64 seed1/6/7 reference와 API 스케줄 재사용. 각 seed의 FSM/snapshot LLM을 같은 GPU에서 순차 실행(GPU6 seed1·7, GPU7 seed6), 각64환경·900step, 6개 실행 exit0. API 추가 호출 없음.
- 실행 불가능한 seed6 env1을 양쪽 집계에서 제외하여 191장면·선행544명을 대응. packet/predecessor/groups/goals/sizes exact 일치 검증. 원본 FSM/새 snapshot LLM/기존 LLM을 별도 집계.
- 선행 성공278/280/284명(51.10/51.47/52.21%), 선행 낙상120/176/171명(22.06/32.35/31.43%). 새 freeze의 단일 장면 낙상 개선은 전체 표본에서는 재현되지 않음. 성공은 region ever 기준, 낙상과 상호 배타적이지 않음.
- FSM과 snapshot LLM에서 선행 모두 성공207, FSM만 성공71, LLM만 성공73, 둘 다 실패193. 성공 여부 차이는 존재하지만 선행 성공률은 거의 동일하고 LLM 낙상률이 높음. 스케줄·관측 갱신·freeze 구현 차이가 함께 있으므로 특정 원인으로 단정하지 않음.
- 최초 후처리에서 excluded 레코드를 정수로 간주한 인덱스 오류를 검출하고 environment 필드를 읽도록 수정. GPU 결과는 재실행/수정하지 않고 summarize_fsm_llm_episodes.py로 재집계. 이전 집계 오류는 status.json에 보존.
- 결과 output_etc/check/fsm_llm_paired_16h_env64_20261111/comparison.json, 선행별 predecessor_pairs.csv. 모든 GPU 실행 종료, ksh/MPS/기존 학습 변경 없음.


### FSM 비의존 snapshot freeze 실험

- sct_scheduler/snapshot_freeze.py 신규 구현. FSM 상태·접촉·성공 gate 없이 초기 snapshot을 대기 중 고정하고, release 시 root/DOF/body 초기 속도까지 1회 복원한다. 물체는 투영하지 않으며 충돌은 유지한다. 실제 kinematic actor 전환은 아니다.
- batch_eval_entry.py의 SCT_FREEZE_IMPL=snapshot_resume로 선택하며 기본 legacy 실행은 보존. 같은 모델·장면·저장 스케줄로 GPU6 1환경16명900step 실행, 정상 종료. API 추가 호출 없음.
- 최초 delayed release(frame45) 이전 root 궤적이 기존과 exact 일치, frame45부터 차이 발생. 복원 오차0, 초기 root/body gap0. 낙상6→3, ON_TOP A04는 여전히 실패(해제 후 물건 중심 최대 높이0.2055m). 단일 장면 결과로 일반적 개선을 주장하지 않음.
- CPU 테스트44개 통과. 새 테스트는 batch actor 매핑, frozen 속도0, release 시 초기 속도 복원, 이후 재복원 방지, simulator 제출 actor IDs, hook 정리를 확인한다.
- 결과 output_etc/check/viewer_one_env_20261111_015804/snapshot_freeze_comparison.json 및 teacher_runtime 동명 경로/snapshot_resume. 기존 전체평가 결과는 수정하지 않음.


### 동일 16인 장면 FSM 동작 진단

- 원본 FSM eval_entry/controller/state를 teacher_runtime/fsm_diagnostic에 그대로 복사하고 ksh 쓰기 금지 audit guard로 GPU6에서 1환경·900step 실행. 초기 observed_scene exact 검증 통과, 기존 headless resolved_env 설정 일치, Stage1 native actor/RMS 검증 통과.
- GPU7에서는 fsm_schedule_diagnostic으로 원본 freeze/관측 갱신 경로에 저장 LLM release frame만 적용한 대조 실행. 운영 scheduler와 기존 전체 평가 결과는 수정하지 않음. 두 실행 모두 exit 0, API 추가 호출 없음.
- 기존 LLM / 원본 FSM / FSM freeze+동일 LLM 스케줄의 낙상은 각각 6/5/2명. ON_TOP A04는 세 실행 모두 실패. 원본 FSM은 frame634(21.13초)에 실제 선행 성공 후 해제했지만 물건 높이가 0.15m에 머물고 frame894에 낙상.
- 같은 스케줄 대조도 첫 프레임부터 root 최대 약0.00597m 차이. 관측 초기화·갱신/상태 제출 경로와 GPU 차이를 추가 분리하기 전에는 freeze 코드 결함이나 LLM 타이밍만의 원인으로 단정하지 않음. 단일 장면 진단이며 집계 성능 비교 아님.
- 결과 output_etc/check/viewer_one_env_20261111_015804/fsm_behavior_comparison.json, 재집계 compare_fsm_behavior.py. 기존 viewer/headless 동일 스케줄 재생은 trace 전체 exact 일치(viewer_headless_comparison.json).


### Viewer를 1환경으로 제한

- 사용자 요청에 따라 기존64환경 viewer에 Q 키를 보내 GPU 동기화 정상 종료. viewer 진입점에 num_envs==1 검사를 추가하고 선행/후행 색상·카메라를 입력 관계에서 선택하도록 수정.
- GPU6에서 새1환경·16명 seed1 snapshot 생성 후 현재 prompt로 API1회 호출, 저장 스케줄을 6086 viewer에 적용. 기존64환경 평가의 실패 장면과 동일하다고 주장하지 않음. 경로 output_etc/check/viewer_one_env_20261111_015804 및 teacher_runtime 동명 경로.

### GPU6 평가 장면 viewer

- 별도 view_batch_entry.py로 현재 batch evaluator를 그대로 감싸 GPU viewer 연결. 64환경/16명 seed1, 같은 reference·Stage1 26k·원본 API 스케줄·낙상 처리로 실행하며 env0의 A11→A04 실패 사례를 표시한다. 기존 1환경/GPU7 viewer는 수정하지 않음.
- GPU6 UUID 실행 확인, noVNC6086, 첫 control step 후 일시정지. 초기 roots/goals/groups/logical_box_order exact, 최대 root 이동0.091865m 및 root/body 차이0.000021579m 확인. 화면 캡처로 준비 상태 확인. 색상은 선행 A11 빨강/후행 A04 하늘색, Space 재생·정지/숫자 카메라/Q 정상 종료.
- 실행 경로 teacher_runtime/output_etc/check/viewer_gpu6_eval16_20261111_015305. API 추가 호출·기존 평가/학습 종료·MPS 변경 없음. 전체 궤적의 headless 일치 여부는 아직 검증 전이며 초기 상태/첫 step 검증만 완료.

### Mixed + Table2 전체 64환경 평가 큐

- 원본 최신 run_suite의 42개 조합(Independent/Mixed 2/4/8/16H × seed1/6/7 + Table2 6설정 × 3 seeds)을 확인. 사용자 최종 요청에 따라 Independent 전용 12회를 제외하여 30회 × 64환경 = 1,920장면/API 호출로 구성.
- Mixed 12회와 Table2 enable SIT/CLIMB/ON_TOP·coupled2/3/4의18회, 900step, Stage1 원본26k 정책. Mixed 내부 독립 agent는 0초 출발을 prompt/검증에 명시. GPU6/7 각각 한 subprocess, API 최대16병렬, 자동 재시도·출발 시각 보정 없음.
- 현재 exporter로 새 snapshot에서 request를 생성하며 명목 속도 지시 부재 검사. 실행 코드와 prompt snapshot/hash 저장, 실행 중 코드가 변하면 다음 단계 제출 중단. 과거 API 결과 재사용 없음.
- Table2까지 초기 freeze 경로 연결. 실행 불가능한 장면 제외 시 formal trace/initial과 Table2 tracker/cooperation 배열을 같은 인덱스로 필터하고 full initial 별도 보존. 원본 CPU rescore·Table2 replay 및 실제 release tick 검증 자동 수행.
- CPU 43 tests 및 30개 조합/구문 검사 통과. GPU 사전 검증은 mixed16/enable_sit/coupled3 × 2환경×30step이며 Table2는 마지막 슬롯을 제외하는 합성 검사 포함, 논문 수치에서 제외.
- 최초 사전 검증(003425)은 freeze로 0이 된 초기 속도와 원본 속도의 직접 비교 때문에 CPU 검사에서 중단(API 0건). 속도 초기화만 반영해 검사 수정 후 기존 궤적 재검증 통과. 실패 기록은 보존.
- 활성 실행 경로 output_etc/check/full_mixed_table2_env64_20261111_003631. status.json이 실제 진행 상태 기준. 원본 ksh·기존 학습·MPS 수정/종료 없음.
- Mixed16 / Enable SIT2 / Coupled3 사전 GPU 검증 모두 통과. Table2의 제외 슬롯 필터·CPU/GPU SR·Error·Time/contact replay 일치 확인. 64환경 본 캡처 시작, 첫4개 설정 캡처 완료 및 실패0 확인 후 자동 큐 유지.

## 2026-11-10

### 응답 limitations 필드 제거

- 사용자 요청에 따라 출력 schema, 필수 필드 검증, 합성 응답 및 preflight에서 limitations 제거. 출력은 predictions/schedule/predicted_episode_events 세 필드이며 가정·불확실성은 prediction basis/confidence로 설명한다.
- 입력 environment_spec의 환경 설명은 유지. 과거 응답은 batch 로드 시 메모리에서만 구 필드를 제거하며 저장된 원본은 수정하지 않음.
- tokenhsi_koo CPU 42 tests 통과. API/GPU 재실행 없음.

### 응답 status 필드와 상태 분기 제거

- 사용자 요청에 따라 system/user prompt에서 ok/infeasible 및 status 필드를 제거. 통합 출력은 predictions/schedule/predicted_episode_events/limitations 네 필드이며 완전한 스케줄을 요구한다.
- 검증기, CLI, episode/Isaac adapter, batch API/평가 경로, 합성 fixture를 새 형식에 맞춤. 빈 스케줄·제약 위반·시간 합산 오류는 계속 거부한다. 기존 배치 응답의 status=ok는 GPU batch 로드 시 메모리에서만 제거하며 저장된 과거 응답과 평가 결과는 보존한다.
- 프로젝트 tokenhsi_koo 환경에서 CPU 41 tests 통과. 기본 Python에는 torch가 없어 테스트 환경을 바로잡아 재검증함. API/GPU 재실행 없음.

### 출력 규칙 통합

- system prompt의 상태별 출력 규칙을 하나의 OUTPUT RULES 목록으로 통합하고 중복 표현을 정리. JSON schema 및 ok/infeasible 처리와 타이밍 제약은 유지. 코드 변경·API/GPU 실행 없음.

### 사용하지 않는 계획 경로 설명 삭제

- 현재 입력에 계획 경로가 없으므로 system prompt의 INFORMATION BOUNDARY에서 planned geometric path 문단 삭제. 실행 전 정보 및 장면 데이터/지시 구분 규칙은 유지. API 재호출 없음.

### TIMING ESTIMATES의 명시적 속도 지시 제거

- 사용자 요청에 따라 system prompt의 해당 절에서 nominal/walking/measured speeds 언급과 속도·준비시간 가정 지시를 제거. 참조 대상이 없어진 Adapt them 문장도 삭제.
- 관측·가정 설명, active duration에서 초기 대기 제외, 미보정 추정의 low confidence 및 수치 출력 조건은 유지. 코드·기존 결과 변경 및 API/GPU 실행 없음.

### Enabling 최소 대기 조건 제거

- 사용자 요청에 따라 system/user template/16인 예시에서 earliest release 요구, 대기 최소화 목적, safety margin 금지를 제거. 선행 release=0 및 후행 상호작용 event가 선행 enabling event 이후라는 조건만 유지하고 추가 대기를 허용한다. 고정 시간 간격은 지정하지 않는다.
- 응답 검증은 최소 해와의 동등성 대신 제출된 episode event의 실제 부등식/동기화 조건을 검사한다. 참고 solver는 합성 fixture/preflight에만 유지. 기존 API 응답 및 완료 평가 결과는 수정하지 않음.
- CPU 41 tests 통과: 추가 enabling 대기 허용, 동시 event 거부, 선행 release=0, coupled 동기화, 시간 합산 검증 포함. API/GPU 재실행 없음.

### 저장된 원본 스케줄 실행 평가 (검증 오류도 포함)

- 사용자 요청에 따라 예측 시간·제약·필드 오류를 실행 차단 조건에서 분리. 기존 strict 검증은 유지하며 명시적 execution-only 평가에서 agent별 유한·비음수 출발 시각 완비만 요구한다. 출발 시각 재계산과 API 재호출 없음.
- 1,536건 중 1,535건 포함(기존 검증 실패 67건 포함). seed7/env0102만 A04 release 누락으로 제외. 원본 API 파일은 읽기 전용 재사용하며 새 실행 manifest에 모든 오류/제외를 기록.
- 초기 샘플 재현을 위해 GPU batch는 512 슬롯 유지. 제외 슬롯은 horizon 전체 동결하고 로컬 formal_eval에서 점수 및 trace의 모든 환경 축을 필터링한다. 원래 환경 인덱스 매핑 별도 저장, 초기 snapshot은 512 슬롯 그대로 보존. 원본 ksh 파일은 수정하지 않음.
- 실행: output_etc/check/eval16_raw_20261110_214845, GPU7, Stage1 26k, seed1/6/7 순차 900step. CPU 39 tests 통과 및 전건 prepare 검사 완료. seed1 초기 snapshot 4항목 exact, 첫 step 최대 root 이동 0.099482m / root-body 차이 0.000086317m. 최종 GPU 결과는 실행 상태 파일 기준.
- 완료: seed별 512/512/511환경, 900step. 원본 CPU rescore가 세 seed 모두 GPU 성공 판정·JSON count·오차에 일치. 24,560 agent 해제 시각이 원본 schedule의 ceil(control_dt)와 전부 일치, 제외 슬롯은 해제 0회. 새 API 호출 0.
- Region 그룹 성공률(seed 평균±표준편차): Independent 76.57±0.81%, Enabling 10.63±0.67%, Coupled 53.34±2.35%, Mixed 46.84±0.69%. Enabling은 원본 simultaneous 성공 기준. summary.json / verification.json / release_verification.json에 원시 집계와 검증 보관.

### 원본 16인 설정의 Stage1 + LLM 일괄 평가 시작

- 사용자 승인: mixed 16명/32 object tokens, seed1/6/7 × 512환경 = 1,536장면, 900step. 원본 run_stage2_scalability/eval 프로토콜과 동일 V2 carrier + Stage1 26k, XY100%, distractor 관측 유지/물리 제거, 그룹간 배치와 full attention 사용. 원본 eval 핵심 파일 해시 일치 확인.
- 초기 순간 이동 수정: before_action(0)의 frozen-only indexed 제출이 pending 전체 reset을 덮어쓰지 않도록 모든 actor/DOF 재제출. 첫 관측의 rigid-body buffer도 reset kinematic 상태로 맞춤.
- CPU 36개 테스트 통과. GPU7 사전 검증(2환경×16명×30step)에서 첫 스텝 최대 root 이동 0.043434m, root/body 차이 0.000001907m, Stage1 native actor/normalization exact 및 coordination_calls=0 확인. 사전 검증은 합성 스케줄이며 본평가에 포함하지 않음.
- 초기 장면 → 장면별 API 저장(16 concurrent) → 전부 검증 후 seed별 formal eval → 원본 CPU rescore/summary 경로 추가. API 오류/비실행 응답은 보존하고 미완성 스케줄로 평가하지 않음. 후행 timing target은 completion 대신 의존 상호작용 onset으로 변경.
- 진행 기록: output_etc/check/eval16_full_20261110_202645/status.json. 이 항목 작성 시 실행 시작 상태이며 최종 결과는 아직 없음. 원본 ksh 쓰기 및 MPS 종료 명령 없음.
- 후속 상태: seed1/6/7 각각 512개 초기 요청 생성 완료, 총 1,536개. API 병렬 저장 단계 진입 및 최초 검증 응답 성공 확인. 전체 응답 검증 후 GPU 평가·CPU 재채점 자동 실행 예정이며 최종 완료와 구분한다.

### 실제 실행 정책을 Stage1 SCT로 정정

- 원본 FSM이 쓰는 `eval/stage1_protocol_policy.py` 경로를 로컬 replay/view 진입점에 연결. Stage1JointPushDensity3Cone90Finetune_00026000.pth를 로컬 복사하고 SHA256 ec4215803b529bab1dd1be430154604418392e24faed59bca69b5c61adafe67c 및 epoch26000 CPU 확인.
- Stage1 가중치·정규화 복원, goal-aligned observation 변환, native actor 일치 검사 및 Stage2 coordination 호출 거부 hook 사용. 종료 시 policy proof 기록. Stage2 checkpoint는 환경/네트워크 구성용이며 행동 정책으로 사용하지 않음.
- capture/export 메타데이터를 Stage1 controller_id로 변경하고 이전 Stage2 스케줄 실행 거부. CPU 테스트 36개 통과 및 진입점 문법 검증. GPU/API 재실행 없음; 실제 GPU policy proof와 초기 상태 동기화 오류 수정은 아직 미완료.

### Native viewer 종료 및 초기 상태 동기화 오류 확인

- 사용자 요청으로 :99 Isaac Gym 창에 Q 입력, VIEWER_NORMAL_EXIT 및 viewer/noVNC 프로세스·6087/5999 포트 종료 확인. GPU process signal/MPS 제어 명령 없이 종료.
- 기존 native_view16_20261110_200753/rollout.npz에서 t=0 root와 pelvis rigid-body 위치의 최대 불일치 8.2049m 확인. 첫 저장 구간(약 0.067초)에 일부 즉시 출발 agent의 root가 3.53~8.27m 이동. 시작 순간 상태가 일관되지 않으며, 단순 화면 갱신 문제가 아닌 초기 물리 상태 적용 오류 조사 필요.
- 앞선 초기 tensor 일치 및 frozen projection 오차 0은 전체 초기화의 정당성을 보장하지 않음. 이 rollout을 스케줄 성능·물리 성공 검증으로 사용하지 않는다. GPU 재실행하지 않음.

### 실제 GPU Isaac Gym viewer 제공

- 도식 재생 대신 native Gym viewer를 GPU7 CUDA/Vulkan 양쪽에 연결하고 noVNC localhost:6087로 제공. nvidia-smi에서 viewer PID의 GPU7 M+C+G context 확인.
- 같은 초기 장면 검증 후 Space 시작 대기. 0은 전체, 1~7은 그룹 카메라, Q는 CUDA 동기화 후 정상 종료. 저장된 strict-after 스케줄을 그대로 사용하며 API 재호출 없음. 실행 완료 후에도 화면 유지.
- 첫 기동은 formal 설정의 headless 요구로 config 단계 실패; 새 viewer 진입점에서 장면 설정만 재사용하도록 제한적으로 처리. formal 평가 점수는 생성하지 않음.
- 기동 기록: teacher_runtime/output_etc/check/native_view16_20261110_200753. 기존 학습·MPS 종료 명령 없음.

### 실제 16명 스케줄 freeze rollout 및 재생 뷰

- API strictly-after 응답을 재사용해 GPU7 기존 MPS에서 600step(20초) 정상 실행(rc0), 추가 API 호출 없음. 동일 seed reset의 root/box/크기/task packet/그룹/목표를 저장 초기 상태와 대조해 모두 일치 확인.
- InitialReleaseFreeze를 실제 Stage2 정책에 적용. 16명 전원 release 기록, frozen root/DOF의 projection 후 최대 오차 0. A12 요청 0.51초는 control frame에서 0.533333초에 해제. 전원 해제 후 재동결 없음.
- 실제 root/rigid body/물체/frozen/종료 flag를 기록하고 독립 HTML 시간·그룹·시점 재생기 및 CPU skeleton/box MP4 생성. Gym 원본 mesh 렌더가 아닌 실측 궤적 도식임을 표시. MP4 미리보기 육안 확인 및 HTML canvas mock 그룹별 렌더 검증.
- A10/A14 종료 flag 및 낮은 최종 root 높이 관찰; 성공 rollout으로 주장하지 않음. 정책은 기존 Stage2 learned coordination 포함, 추가 초기 시간 gate만 적용.
- 결과: teacher_runtime/output_etc/check/replay16_20261110_200020/{view.html,replay.mp4,rollout.npz,release_audit.json,initial_match.json}.

### Strictly-after 검증 및 실제 장면 재호출

- enabling 검증을 출력 정밀도의 엄격 부등식으로 변경. 고정 물리 지연은 추가하지 않으며 선행 release=0도 검증. coupled 정렬은 유지. 합성 예제 갱신 및 CPU 테스트 34개 통과.
- 같은 16명 초기 관측으로 gpt-6-luna / medium API 1회 호출, ok 및 검증 통과. A02 예상 완료 5.60초, A12 release 0.51초 + 예상 소요 5.10초 = 5.61초.
- 입력의 A12 이벤트는 여전히 기존 formal completion proxy이므로 최초 앉기 접촉이 선행 완료 뒤라는 검증은 아님. 실제 rollout 없음. 원문: output_etc/check/api_scene16_strict_after_20261110_195414.

### Enable 프롬프트의 엄격한 선후 관계

- 사용자 요청대로 dependency-sensitive event를 선행 enabling event보다 strictly after로 변경하고 동시 발생 금지. 고정 추가 지연 초수는 명시하지 않음.
- 기존 동시 허용 공식 대신 release + q > p를 만족하는 최소 대기 조건으로 system/user template을 정리. 이번 수정은 프롬프트에 한정; 기존 코드 검증기의 비엄격 부등식은 별도 수정 전까지 유지됨. API 호출 없음.

### 가정 기반 프롬프트로 실제 16명 장면 재호출

- 동일한 초기 관측을 재사용하고 입력의 옛 insufficient_information 지시만 교체. 기존 감사 기록은 보존하고 새 경로에 API 1회 호출.
- gpt-6-luna / reasoning medium 응답 ok. 16명 예측·출발 시각·episode event 산술 및 enabling/coupled 최단 대기 제약 검증 통과.
- Enable A02=0초, A12=2초. coupled 그룹 예상 접근 시각은 각각 9.42/11.63/12.29/4.64초. 모든 예측은 low confidence이며 속도·준비 시간은 모델 가정. 실제 rollout 없음.
- 결과: output_etc/check/api_scene16_best_effort_20261110_194522/response.validated.json.

### 정보 부족 응답 선택지 제거

- system/user 프롬프트에서 calibration 부재에도 명시적 속도·준비 시간 가정으로 숫자 예측과 전체 스케줄을 생성하도록 변경. 가정은 실측 사실과 구분하고 low confidence로 표시.
- 응답 스키마의 insufficient_information 상태 제거. infeasible은 시간 제약 충돌에만 사용하도록 명시. exporter의 정보 부족 반환 지시도 제거하고 가독용 16명 예시 갱신.
- CPU 테스트 32개 통과, 제거한 상태의 응답 거부 포함. 기존 실제 요청·응답 감사 기록은 보존. 추가 API/GPU 호출 없음.

### 실제 16명 reset 관측으로 API 1회 호출

- `capture_initial_entry.py`로 GPU7 기존 MPS에서 지정 V3 checkpoint를 로딩하고 환경 1개/16명/seed21 초기 상태를 캡처. policy inference/env_step 없이 CUDA 동기화 후 정상 종료(rc0). 첫 시도는 필수 --group_spawn_over4 누락으로 config 단계 종료; 실패 기록 보존 후 옵션을 추가한 새 경로에서 완료.
- 실제 구성: Enable A02→A12, coupled A01/A08/A15/A16 및 A03/A05/A07/A10 및 A06/A11 및 A09/A13, 독립 A04 PUSH/A14 SIT. `export_initial_scene.py`로 초기 관측만 JSON 및 user template으로 내보내고 스키마 검증.
- Coupled 접근은 수치 임계값 없는 계획상 도달 시점으로 명시. Enable 후행 이벤트는 기존 formal completion proxy이며 실제 최초 접촉 detector가 아님을 입력에 공개.
- OpenAI 설정 모델, reasoning medium으로 API 정확히 1회 호출. 응답은 insufficient_information, predictions/schedule 빈 배열: controller 속도·실측 event duration이 없어 추정 근거 부족. 응답 스키마 검증 성공, CLI rc2는 정상 non-ok 상태 표시. LLM 스케줄 적용 rollout은 수행하지 않음.
- 원시 캡처: teacher_runtime/output_etc/check/initial16_20261110_194117. 입력/template: output_etc/check/real_scene16_20261110_194117. API 감사 기록: output_etc/check/api_scene16_20261110_194117.

### 16명 혼합 장면 입력 예시

- Enable 2쌍 + coupled 2/3/3/4명 그룹으로 총 16명 가상 초기 장면 작성. `examples/hypothetical_16_agents_request.json`은 실제 API 입력 형식, `prompts/user_example_16_agents.txt`는 user template을 채운 가독용 예시.
- 위치·목표·이벤트 조건은 설명용 가정이며 실측/성공 데이터가 아님을 명시. 입력 스키마, 16명 이벤트 소유권, 그룹별 구성, 실제 메시지 직렬화 검증 완료. API/GPU 호출 없음.

### Coupled PUSH 그룹 크기 정정

- 사용자 정정에 따라 system prompt와 user template의 4명 기본 전제를 제거. 그룹은 2~4명이며 실제 구성원은 입력에서 읽도록 명시.
- 장면 전체 인원과 개별 PUSH 그룹 크기를 구분. 그룹 구성원별 초기 release 결정 방식은 유지.
- API 메시지 조립에서 수정된 프롬프트 로딩 확인. API/GPU 호출 없음.

### Enable / coupled 제어 역할을 프롬프트에 명시

- Enable은 2명 단위로 선행 release=0 고정, 후행 freeze 해제 시각만 결정. 출력 스키마 호환을 위해 선행도 schedule에 0으로 포함.
- Coupled 기본 대상은 4명 PUSH 그룹이며 전원의 개별 출발 시각과 선출발자를 결정. 명시된 다른 그룹 크기도 동일 규칙 적용. 동시 출발 허용, 출발 후 별도 PUSH gate 없음.
- system prompt와 참고 user template 갱신. API 메시지 조립에서 변경된 system prompt가 로딩되는지 CPU 확인; API/GPU 호출 없음.

### 10-shot API 경로 제거

- 사용자 요청에 따라 API 입력을 environment_spec/current_scene으로 축소. 데모 10개 필수 검증, 프롬프트의 calibration 지시, supporting_example_ids 출력 필드와 --calibration 옵션 제거.
- 프롬프트는 초기 장면의 근거와 시간 가정을 명시하며 추정할 수 없으면 insufficient_information을 반환. 초기 release 제약과 물리 동결 동작은 유지.
- 합성 예제와 CLI 테스트 갱신. 실측 trajectory·과거 실행 결과는 보존하고 기록 검증 도구는 API 입력 검증에서 분리.
- CPU unittest 31개 통과. 유료 API 호출·GPU 작업 없음.

### OpenAI 직접 호출 및 reasoning medium

- 기본 base URL을 OpenAI `/v1`로 변경. 로컬 `.env`의 기존 OpenAI URL·사용자 모델·키를 보존하고 `SCT_LLM_REASONING_EFFORT=medium` 추가.
- `.env`/환경변수/CLI reasoning effort 설정을 Chat Completions 최상위 `reasoning_effort`로 전달. 빈 값·미지정은 CLI에서 medium, 명시 CLI가 우선하며 inference audit에도 기록.
- CPU 테스트 33개 통과: 요청 본문, 빈 값 fallback, CLI override, 오타 거부 포함. 실제 유료 OpenAI 호출은 수행하지 않음.

### Stage2 실측 성공 trajectory 10-shot 수집

- 사용자 지정 V3 2000 checkpoint와 sidecar·코드를 `teacher_runtime/`에 독립 복사. ksh 쓰기 거부 audit hook 및 로컬 CUDA/extension 캐시 사용.
- GPU7 기존 MPS에 연결하여 6개 setting을 순차 실행, 모두 900step 정상 종료(rc0). 외부 FSM 대기·Stage1 가중치 교체 없음.
- SIT 7/8, CLIMB 13/16, ON_TOP 12/16, 2인 PUSH 13/16, 3인 PUSH 14/16, 4인 PUSH 15/16 native group success. 전체 88장면 중 74성공이며, 생존·readiness 추가 조건을 통과한 후보는 69개.
- 최종 SIT2/CLIMB1/ON_TOP2 + PUSH2인2/3인1/4인2, 총 10개 성공 사례 선택. 초기 상태·root·손·물체 궤적·실측 이벤트를 JSON으로 추출하고 dense raw와 실패 기록도 보존.
- Enabling dependency-sensitive event는 명시적으로 formal completion. 별도 approach/readiness로 오기하지 않음. Coupled readiness는 실제 box 표면 양손 12cm 미만.
- `--calibration`으로 실제 demonstration 묶음을 API 메시지에 로딩. CPU 테스트 31개 통과, 10개 예제 원시 데이터 대조·request 스키마 검증 통과.
- 결과: `output_etc/check/calibration_20261110/`. 실측 raw: `teacher_runtime/output_etc/check/demo_{probe,diverse}_20261110/`. API 호출 0회, 기존 작업·MPS 종료 없음. GPU7 사용 메모리는 종료 후 실행 전 42,557 MiB로 복귀.

### Baze gateway API 설정 및 호출

- 기본 base URL `https://api.baze.co/v1/gateway`에 `/chat/completions`를 붙여 호출. 전체 endpoint override 유지.
- 프로젝트 `.env`에서 키·모델 로딩, 환경변수/CLI override, 빈 설정 템플릿 추가. 로컬 `.env`는 0600 및 Git ignore 확인.
- JSON mode는 명시 옵션으로 변경. Bearer 인증·HTTP 오류 메시지 비밀값 제외 및 재시도 없음.
- CPU 테스트 30개 통과: URL 경로, 인증 헤더, payload, 설정 우선순위, 실제 CLI 설정 연결, 오류 처리 포함.
- Baze 공개 endpoint 규격을 확인하지 못해 Chat Completions 호환을 가정. API 키·모델 미입력으로 실제 인증 호출은 미실행. GPU/MPS 사용 없음.

### Few-shot 10개로 확대

- 사용자 요청에 따라 system/user prompt, 입력 검증, 합성 예제·생성기 및 README를 10개 demonstration 기준으로 변경.
- 10개 입력 허용 및 5/9/11개 입력 거부 확인. CPU unittest 24개 통과.
- API 호출과 GPU 실행 없이 검증. 기존 5-shot 실행 기록은 보존.

### SCT + LLM Scheduler 독립 base 생성

- 제공된 프롬프트와 JSON 메시지 계약, 5개 데모 검증, 일회 호출 및 추론 audit 구현.
- enabling/coupled 혼합 제약을 정수 centisecond difference constraints로 검증. 잘못된 응답 자동 보정·재시도·fallback 없음.
- 원본 FSM의 projection 방식을 로컬 adapter로 이식. 초기 시간 기반 해제만 사용하고 release 후 추가 gate 없음.
- 실제 event recorder, censored label, bounded episode callback 연결, GPU6/7 UUID guard 추가.
- CPU unittest 24개 통과. 실제 Isaac Gym·LLM 호출·성공률은 미검증.
- coupled/enabling 오프라인 CLI 모두 exit 0. 각각 A/B release 2/0초, 0/2초 확인. 11개 Python 파일 구문 검사 통과. 로그는 `output_etc/check/offline_{coupled,enabling}/`.
- ksh 코드·기존 프로세스·MPS·큐·baseline 변경 없음. 합성 데이터는 실제 calibration용이 아님.

## 2026-11-12

### 저장된 평가 전용 연속 viewer로 전환

- 사용자 정정에 따라 새장면API viewer를 실제Q입력으로정상종료(episode4, 이후API생성중단). view_saved_eval.py 추가: API transport/settings미사용, 기존eval launch/reference/request/raw schedule 그대로사용. N다음저장env/완료자동전환/Q전체종료 지원.
- 기존64batch의초기배치재현을위해 내부64환경을보존하되선택장면카메라만표시. 명시적 SCT_VIEW_SAVED_BATCH=1에서만view_batch_entry가64batch허용, 일반viewer는기존1env제한유지. 실제물리재실행이며기록영상재생이라고주장하지않음.
- viewer_saved_eval_2h_20261112_131141, GPU6/noVNC6086, Table2enable_sit seed1 env0 처음frame일시정지. 선행0초/후행1.1초 기존응답적용. 원본평가64환경첫정책입력exact·오차0 검증(replay_verification.json). API추가호출0, 원본MPS/학습/평가기록변경없음. USAGE14절저장평가viewer추가.


### 2명 연속 viewer·N 장면 전환

- view_continuous_enable.py 추가. persistent noVNC6086/GPU6/1환경2명에서 매episode 새seed·capture·API1회·legacy Stage1 viewer를 순차 실행. 기본10episode, Enable SIT/CLIMB/ON_TOP 순환, --settings/--seed/--episodes/--port 지원. 각 GPUchild 정상 종료 후 재생성하여 이전 낙상 motor상태를 복원된 새환경으로 분리.
- view_batch_entry.py에 SCT_VIEW_CONTINUOUS 선택시 N키skip/Q전체종료, 완료2초후자동next 추가. 최초episode만시작일시정지, 이후자동재생. 기존단일viewer 기본동작은유지. GPU signal/MPS조작 없음.
- 기존단일viewer는이미정상종료 상태확인. 새 viewer_continuous_2h_20261112_130349에서 실제XTest N입력으로episode1(next) 정상종료 확인, episode2는850frame 전원낙상 종료 후 episode3자동진입 확인. controls_verification.json 기록, 첫정책입력/초기상태검사통과. 구문검사통과, USAGE13절실행법추가.


### GPU6 2명 Enable 1환경 viewer

- 사용자 요청으로 view_two_agent_enable.py 추가, AT→SIT 1환경2명 seed1 fresh reset capture 후 현재prompt API1회, legacyfreeze·Stage1 26k로 view_batch_entry.py 실행. 기존64환경 특정failure를 추출한 장면이라고 주장하지 않음.
- GPU6/noVNC6086/display99, 첫frame일시정지. A02선행빨강0초/A01후행파랑8.2초. 초기state검사통과, 첫root변위0.03536m/rootbodygap0, 실제첫정책입력exact 오차0. Space재생·정지/Q GPU동기화정상종료. MPS/기존프로세스제어 없음.
- 경로 output_etc/check/viewer_enable_2h_20261112_125738 및teacher_runtime동명폴더, 상태status.json/viewer_ready.json. 화면캡처는기본Pillow XCB미지원으로미생성; 실제viewer ready 및render loop/포트검증 완료.


### 2H/4H 성능 저하 궤적 분석

- 최신 protocol_v1 legacy 평가의 Mixed2/4와 Table2Coupled2/4를 CPU 분석. small_scene_analysis_20261112/summary.json에 Enable쌍별 실패 분해·선행 예측/실측 오차·Coupled 전원양손근접 이력/도착산포 저장. 분석기 analyze_small_scene_failures.py 추가. 추가 API/GPU 없음.
- Mixed2:61쌍=성공19/선행미완료16/선행성공후행미완료22/각자성공동시불성립4. Mixed4:121쌍=41/30/45/5. 도달한선행45/91건에서 실측평균7.01/8.63초, 동일사례의 예측이 평균7.16/5.47초 더 길었음. 미도달사례는 시간오차 계산에서 제외.
- 선행미완료16/30 중 물건 들림징후없이이탈11/19, 실패중선행낙상0/2. 선행성공/후행실패22/45 중 후행낙상6/15. ON_TOP해당12/22 중 들림징후없이이탈3/14. 들림0.15m/거리증가2m/root변위3m는 행동proxy이며실측파지가아님.
- Table2Coupled 실패51/77 중 전원양손12cm유효근접 이력44/59, 동시전원근접33/30. 전원접근한 실패도 최초접근시간산포평균2.71/5.96초(성공전 전원접근 성공사례1.78/3.15초). 상자먼저이동은성공에서도빈번하므로 원인으로단정하지않음. 성공4인중23건은전원근접 없이성공하여공동상자SR이전원참여보장이아님을확인.
- 최종 JSON에 후행추가진단포함. 조건부통계는인과증거로해석하지않고 접촉력/정확한interaction onset미기록 한계를 명시.

### Mixed 8H/16H fixed-radius rerun
- Added opt-in `SCT_GROUP_RADIUS_VARIANT=8h3m_16h7p5m`: group centers use R=3/7.5m with the existing angular RNG, internal translations and follower recentering preserved. Independent and Mixed placement share this implementation.
- Radius diagnostics merge Enabling members and check actual centers against radius and equal-angle targets; gap metadata is null.
- Added `run_group_radius_eval.py`: 64 environments, seeds 1/6/7, legacy freeze, fresh schedules, GPU6 sequential execution; existing outputs preserved.
- Validation: 44 CPU tests passed; runtime capture/evaluation verification recorded under `mixed8_16_radius3_7p5_20261112_082344`. Launch supervisor PID 3551037. No changes under /home/hwanhee/ksh; no MPS control or process signals.
