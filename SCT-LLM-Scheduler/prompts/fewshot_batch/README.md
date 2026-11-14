# 다장면 단일 요청용 few-shot 프롬프트

기존 prompts/fewshot은 장면당1회 호출에 사용하며 그대로 유지한다. 이 디렉터리는 여러 current_scene을 하나의 요청으로 넣는 별도 실험용이다.

- system.txt: Enable은 선행 이후 및 근거가 있는 추가 대기 허용. Coupled 규칙 유지. 각 record_id의 에피소드를 독립 처리하고 results 배열로 반환.
- user_template.txt: 구조 설명용 placeholder. 실제 바로 입력할 파일은 output_etc/check/batch_input_1920_gap02_20261112/user_input_1920.txt.
- 생성: `python export_batch_llm_input.py --output output_etc/check/<새 폴더>`.
- 출력 폴더 system_prompt.txt는 생성 시점의 시스템 프롬프트 snapshot.
- 원본 단일 장면 API·eval 실행에는 영향 없음. 이 배치 입출력은 기존 OneShotScheduler에 바로 연결되지 않음.
- 1920개 입력의 JSON/원본 일치는 검증했지만 모델 context/output 한도와 단일 호출 처리 가능성은 검증하지 않음. API 호출 없음.

기존 batch_input_1920_gap02_20261112/system_prompt.txt는 0.2초 실험 snapshot으로 보존한다. 최신 조건은 이 디렉터리 system.txt를 사용한다.
