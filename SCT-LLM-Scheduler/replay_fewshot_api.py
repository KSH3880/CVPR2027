"""Call the current prompt against saved initial observations, without simulation."""
import argparse,hashlib,json,time,shutil
from pathlib import Path
from sct_scheduler.progress import StageETA
eta=StageETA()
from concurrent.futures import ThreadPoolExecutor,as_completed
from sct_scheduler.config import read_settings,chat_endpoint,DEFAULT_BASE_URL,DEFAULT_REASONING_EFFORT
from sct_scheduler.client import OneShotScheduler,chat_transport,loads
from sct_scheduler.core import validate_request, validate_response, build_messages
from sct_scheduler.fewshot import attach_references
from sct_scheduler.executable import validate_executable

ROOT=Path(__file__).resolve().parent

def save(path,value):
    if path.name == "status.json": eta.annotate(value)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    temporary.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--settings',nargs='+',choices=['mixed_2h','mixed_4h','mixed_8h','mixed_16h','enable_sit','enable_climb','enable_ontop','coupled_2','coupled_3','coupled_4']);parser.add_argument('--additional-source',type=Path,action='append',default=[]);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--reuse-from',type=Path)
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve()
    assert (ROOT/'output_etc/check').resolve() in out.parents and 1<=args.workers<=16
    sources=[source]+[p.resolve() for p in args.additional_source]
    selected={}
    for root in sources:
        for p in sorted((root/'api').rglob('request.json')):
            rel=p.relative_to(root/'api')
            if args.settings is None or rel.parts[1] in args.settings:selected[rel]=p
    paths=[selected[k] for k in sorted(selected)]
    expected=1920 if args.settings is None else len(set(args.settings))*192
    assert len(paths)==expected,(len(paths),expected)
    work=[]
    for p in paths:
        req=attach_references(loads(p.read_text()));validate_request(req)
        work.append((next(p.relative_to(r/'api').parent for r in reversed(sources) if r/'api' in p.parents),req,hashlib.sha256(p.read_bytes()).hexdigest()))
    prompt=(ROOT/'prompts/fewshot/system.txt').read_text();settings=read_settings()
    effort=settings.get('SCT_LLM_REASONING_EFFORT','').strip() or DEFAULT_REASONING_EFFORT
    model=settings['SCT_LLM_MODEL'];url=settings.get('SCT_LLM_BASE_URL') or DEFAULT_BASE_URL
    raw_transport=chat_transport(chat_endpoint(url),model,settings['SCT_LLM_API_KEY'],timeout_s=180,json_mode=True,reasoning_effort=effort)
    def transport(messages):
        if messages[0]['content']!=prompt:raise RuntimeError('Prompt changed during replay; no HTTP call made')
        return raw_transport(messages)
    out.mkdir(parents=True,exist_ok=False)
    save(out/'demonstrations.json', work[0][1]['demonstrations'])
    (out/'system_prompt.txt').write_text(prompt)
    save(out/'manifest.json',dict(source=str(source),additional_sources=[str(r) for r in sources[1:]],selected_settings=args.settings,scene_count=len(work),workers=args.workers,model=model,base_url=url,reasoning_effort=effort,api_only=True,gpu_jobs=0,retries=0,source_current_scenes_unchanged=True,fewshot=True,demonstrations=10,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),requests=[dict(scene=str(rel),sha256=sha) for rel,_,sha in work]))
    reused=set()
    if args.reuse_from:
        prior=args.reuse_from.resolve()
        assert loads((prior/'manifest.json').read_text())['model']==model
        assert loads((prior/'manifest.json').read_text())['reasoning_effort']==effort
        for rel,req,_ in work:
            src=prior/'api'/rel
            if not (src/'response.raw.txt').exists():continue
            assert loads((src/'request.json').read_text())==req, str(src)
            assert loads((src/'messages.json').read_text())==build_messages(req), str(src)
            shutil.copytree(src,out/'api'/rel)
            reused.add(str(rel))
        save(out/'reuse_audit.json',dict(source=str(prior),count=len(reused),scenes=sorted(reused),new_calls_planned=len(work)-len(reused)))
    state=dict(stage='api',completed=False,total=len(work),api_done=0,api_validated=0,api_failed=0,api_nonexecutable=0,api_progress_percent=0.,started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    state['api_reused']=len(reused)
    state.update(api_workers=args.workers,api_rate_limited=0,api_timeouts=0,api_transport_errors=0,last_api_error=None)
    save(out/'status.json',state);rows=[]
    def call(item):
        rel,req,_=item;dest=out/'api'/rel;error=None;error_type=None
        try:
            if str(rel) in reused:
                validate_response(req,loads((dest/'response.raw.txt').read_text()))
            else:
                OneShotScheduler(transport).schedule(req,dest)
        except Exception as exc:error=str(exc);error_type=type(exc).__name__
        executable=True
        try:validate_executable(req,loads((dest/'response.raw.txt').read_text()))
        except Exception:executable=False
        return dict(scene=str(rel),validation_error=error,error_type=error_type,executable=executable,reused=str(rel) in reused)
    print(json.dumps(dict(output=str(out),total=len(work),model=model,workers=args.workers,reasoning_effort=effort)),flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(call,item) for item in work]):
            row=future.result();rows.append(row);state['api_done']+=1;state['api_validated']+=int(row['validation_error'] is None);state['api_failed']+=int(row['validation_error'] is not None);state['api_nonexecutable']+=int(not row['executable'])
            if row['validation_error']:
                message=row['validation_error'].lower()
                limited='http 429' in message
                timeout='timeout' in str(row.get('error_type','')).lower() or 'timed out' in message
                transport_error=not (out/'api'/row['scene']/'response.raw.txt').exists()
                state['api_rate_limited']+=int(limited)
                state['api_timeouts']+=int(timeout)
                state['api_transport_errors']+=int(transport_error)
                state['last_api_error']=row
                if limited or timeout or transport_error:
                    alert=dict(scene=row['scene'],error=row['validation_error'],rate_limited=limited,timeout=timeout,workers=args.workers)
                    save(out/'api_alert.json',alert)
                    print('API_ALERT '+json.dumps(alert),flush=True)
            state['api_progress_percent']=round(100*state['api_done']/state['total'],2)
            save(out/'status.json',state)
            if row['validation_error']:save(out/'api_failures.json',[r for r in rows if r['validation_error']])
            if state['api_done']%64==0:print(json.dumps(state),flush=True)
    save(out/'api_audit.json',rows);state.update(stage='complete',completed=True);save(out/'status.json',state);print(json.dumps(state),flush=True)

if __name__=='__main__':main()
