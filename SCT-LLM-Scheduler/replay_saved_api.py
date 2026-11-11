"""Call the current prompt against saved initial observations, without simulation."""
import argparse,hashlib,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from sct_scheduler.config import read_settings,chat_endpoint,DEFAULT_BASE_URL,DEFAULT_REASONING_EFFORT
from sct_scheduler.client import OneShotScheduler,chat_transport,loads
from sct_scheduler.core import validate_request
from sct_scheduler.executable import validate_executable

ROOT=Path(__file__).resolve().parent

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    temporary.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve()
    assert (ROOT/'output_etc/check').resolve() in out.parents and 1<=args.workers<=16
    paths=sorted((source/'api').rglob('request.json'));assert len(paths)==1920
    work=[]
    for p in paths:
        req=loads(p.read_text());validate_request(req)
        work.append((p.relative_to(source/'api').parent,req,hashlib.sha256(p.read_bytes()).hexdigest()))
    prompt=(ROOT/'prompts/system.txt').read_text();settings=read_settings()
    effort=settings.get('SCT_LLM_REASONING_EFFORT','').strip() or DEFAULT_REASONING_EFFORT
    model=settings['SCT_LLM_MODEL'];url=settings.get('SCT_LLM_BASE_URL') or DEFAULT_BASE_URL
    raw_transport=chat_transport(chat_endpoint(url),model,settings['SCT_LLM_API_KEY'],timeout_s=180,json_mode=True,reasoning_effort=effort)
    def transport(messages):
        if messages[0]['content']!=prompt:raise RuntimeError('Prompt changed during replay; no HTTP call made')
        return raw_transport(messages)
    out.mkdir(parents=True,exist_ok=False)
    (out/'system_prompt.txt').write_text(prompt)
    save(out/'manifest.json',dict(source=str(source),scene_count=len(work),workers=args.workers,model=model,base_url=url,reasoning_effort=effort,api_only=True,gpu_jobs=0,retries=0,source_requests_unchanged=True,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),requests=[dict(scene=str(rel),sha256=sha) for rel,_,sha in work]))
    state=dict(stage='api',completed=False,total=len(work),api_done=0,api_validated=0,api_failed=0,api_nonexecutable=0,started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    save(out/'status.json',state);rows=[]
    def call(item):
        rel,req,_=item;dest=out/'api'/rel;error=None
        try:OneShotScheduler(transport).schedule(req,dest)
        except Exception as exc:error=str(exc)
        executable=True
        try:validate_executable(req,loads((dest/'response.raw.txt').read_text()))
        except Exception:executable=False
        return dict(scene=str(rel),validation_error=error,executable=executable)
    print(json.dumps(dict(output=str(out),total=len(work),model=model,workers=args.workers,reasoning_effort=effort)),flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(call,item) for item in work]):
            row=future.result();rows.append(row);state['api_done']+=1;state['api_validated']+=int(row['validation_error'] is None);state['api_failed']+=int(row['validation_error'] is not None);state['api_nonexecutable']+=int(not row['executable'])
            save(out/'status.json',state)
            if row['validation_error']:save(out/'api_failures.json',[r for r in rows if r['validation_error']])
            if state['api_done']%64==0:print(json.dumps(state),flush=True)
    save(out/'api_audit.json',rows);state.update(stage='complete',completed=True);save(out/'status.json',state);print(json.dumps(state),flush=True)

if __name__=='__main__':main()
