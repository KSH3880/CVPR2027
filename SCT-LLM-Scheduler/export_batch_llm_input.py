"""Export all 1920 initial scenes for a single multi-scene LLM request; no API."""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from run_full_suite import jobs
from sct_scheduler.fewshot import attach_references
from sct_scheduler.core import validate_request
ROOT=Path(__file__).resolve().parent

def digest(data):return hashlib.sha256(data).hexdigest()
def compact(x):return json.dumps(x,ensure_ascii=False,allow_nan=False,separators=(',',':'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    out=a.output.resolve();assert ROOT/'output_etc/check' in out.parents
    out.mkdir(parents=True,exist_ok=False)
    baseline=ROOT/'output_etc/check/eval_protocol_v1_legacy_20261111_165027'
    radius=ROOT/'output_etc/check/mixed8_16_radius3_7p5_20261112_082344'
    specs={};spec_ids={};records=[];audit=[];demonstrations=None
    for job in jobs():
        source=radius if job['people'] in (8,16) else baseline
        gpu=Path(json.loads((source/'status.json').read_text())['gpu_output'])
        ref=gpu/'reference'/job['key']
        if job['people'] in (8,16):
            layout=json.loads((ref/'scalability_protocol.json').read_text())
            assert layout['group_spawn_radius']=={8:3.,16:7.5}[job['people']] and layout['group_spawn_gap'] is None
        for i in range(64):
            path=ref/'requests'/f'env_{i:04}'/'request.json'
            original=json.loads(path.read_text());validate_request(original)
            request=attach_references(original)
            if demonstrations is None:demonstrations=request['demonstrations']
            assert request['demonstrations']==demonstrations
            key=compact(request['environment_spec'])
            if key not in spec_ids:
                spec_id=f'environment_{len(spec_ids)+1:02d}';spec_ids[key]=spec_id;specs[spec_id]=request['environment_spec']
            record_id=f"{job['key']}/env_{i:04}"
            records.append(dict(record_id=record_id,environment_spec_id=spec_ids[key],current_scene=request['current_scene']))
            audit.append(dict(record_id=record_id,source=str(path),source_sha256=digest(path.read_bytes()),scene_sha256=digest(compact(original['current_scene']).encode())))
    assert len(records)==1920 and len({r['record_id'] for r in records})==1920
    payload=dict(batch_id=out.name,environment_specs=specs,demonstrations=demonstrations,scenes=records)
    user=compact(payload)
    system=(ROOT/'prompts/fewshot_batch/system.txt').read_text()
    (out/'user_input_1920.txt').write_text(user+'\n')
    (out/'system_prompt.txt').write_text(system)
    # Round-trip and source equality prove no measured future trajectories entered current scenes.
    decoded=json.loads((out/'user_input_1920.txt').read_text())
    for row,item in zip(decoded['scenes'],audit):
        original=json.loads(Path(item['source']).read_text())
        assert row['current_scene']==original['current_scene']
        validate_request(dict(environment_spec=decoded['environment_specs'][row['environment_spec_id']],current_scene=row['current_scene']))
    manifest=dict(scene_count=len(records),agent_count=sum(len(r['current_scene']['initial_observation']['agents']) for r in records),settings=dict(Counter(r['record_id'].split('/')[1] for r in records)),environment_spec_count=len(specs),demonstration_count=len(demonstrations),user_characters=len(user),user_bytes=len(user.encode()),system_sha256=digest(system.encode()),user_sha256=digest((user+'\n').encode()),group_radius_m={'8':3,'16':7.5},enable_min_predicted_event_gap_s=None,api_calls=0,current_scenes_exactly_preserved=True,outputs_included=False,existing_prompts_unchanged=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'scene_index.json').write_text(json.dumps(audit,indent=2))
    (out/'README.md').write_text('''# 1,920장면 단일 요청용 입력

- `system_prompt.txt`: 별도 system 메시지. 기존 few-shot에서 다장면 처리/출력 래퍼만 분리한 버전.
- `user_input_1920.txt`: user 메시지에 그대로 넣는 실제 JSON. placeholder 없음.
- 공통 environment_spec은 ID로 참조하고 예시10개는 한 번만 포함. 각 current_scene은 원본과 완전히 동일.
- `record_id`는 설정/seed/env를 포함한 고유 키. agent_id/event_id/scene_id는 장면 사이 중복될 수 있으므로 record_id로 결과를 매칭.
- 30설정×64환경=1920장면. Mixed2/4/8/16+Table2, Independent 전용 제외. 8H R3m/16H R7.5m, Enable은 선행 이후, 장면/과제 근거가 있으면 추가 대기 허용.
- 반환 형식은 {"results":[{"record_id":"...","predictions":[],"schedule":[],"predicted_episode_events":[]}]}.
- 파일 생성만 수행했으며 호출은 하지 않음. 실행 중인 장면당 API 실험은 변경하지 않음.
- 기존 OneShotScheduler는 이 배치 입력/결과를 직접 처리하지 않는다. 별도 호출에서 아래 messages를 사용하고 결과를 record_id별로 분리해 검증해야 한다.

```python
import json
from pathlib import Path
folder = Path(__file__).resolve().parent  # 이 파일들이 있는 폴더
messages = [
    {"role": "system", "content": (folder / "system_prompt.txt").read_text()},
    {"role": "user", "content": (folder / "user_input_1920.txt").read_text()},
]
```

입력 전체와 1,920장면 응답이 사용 모델의 context 및 output 한도에 모두 들어가야 한다. 이 파일의 생성/JSON 검증은 단일 호출에서 처리 가능하다는 검증이 아니다. 전체가 들어가지 않으면 장면 경계로 나눠야 한다. 배치 입력은 기존 장면별 독립 호출과 다른 실험 조건으로 보고한다.
''')
    print(json.dumps(dict(output=str(out),**manifest),ensure_ascii=False))
if __name__=='__main__':main()
