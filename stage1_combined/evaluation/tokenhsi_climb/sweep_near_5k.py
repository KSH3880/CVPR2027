"""Evaluate saved FT4500..5500 every100, reusing verified FT5000."""
from pathlib import Path
import concurrent.futures,datetime,json,os,subprocess,argparse,threading
import numpy as np
from evaluation.tokenhsi_climb.watch_continuation import verify,BASE,ROOT
CKPT=ROOT.parent/'runs/tokenhsi_climb_78k_20261024/train/TokenHSIClimbFullFT_24-16-15-27/nn'

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 epochs=list(range(4500,5501,100));paths={e:CKPT/f'TokenHSIClimbFullFT_{e:08d}.pth' for e in epochs};assert all(x.is_file() for x in paths.values())
 state=dict(protocol='1human/1fixed_object/Box-Cabinet-Table equal category mix/test/512env/seed0/3repeat/300step/point20cm ever',epochs=epochs,workers=2,training_stopped=True,results={})
 lock=threading.Lock()
 def persist():
  rows=state['results'];best=max(rows,key=lambda k:rows[k]['success_rate']) if rows else None;state['best_epoch']=int(best) if best else None
  (out/'summary.json').write_text(json.dumps(state,indent=2))
  lines=['# FT5k 주변 체크포인트 비교','',state['protocol']+'. 0회차 제외 repeat1·2 평균±표본SD. 독립 seed 간 산포가 아님. 기존 FT5000의 동일 조건 검증 결과는 재사용.','', '| FT epoch | 점 성공률 (%) | 성공 시 최소 거리 (cm) |','|---|---:|---:|']
  for key in sorted(rows,key=int):
   r=rows[key];lines.append(f"| {key} | {100*r['success_rate']:.2f} ± {r['repeat_std_pp']:.2f} | {100*r['precision_m']:.2f} |")
  if best:lines+=['',f"현재 관측 최고: FT{best}, {100*rows[best]['success_rate']:.2f}%.",f"Weight: `{rows[best]['checkpoint']}`",'단일 seed의 평가 반복 비교이므로 근소한 차이를 일반적 우위로 확정하지 않는다.']
  (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 def evaluate(epoch):
  if epoch==5000:directory=BASE
  else:
   directory=out/f'ft_{epoch:05d}';directory.mkdir()
   env=os.environ.copy()
   for k in ['CHECK','MAX_ITERATIONS','RESUME_CHECKPOINT','EPISODE_LENGTH','TASK_GRAPH']:env.pop(k,None)
   env.update(TOKENHSI_GPU='0',TOKENHSI_CONDA_ENV='tokenhsi118',PYTORCH_JIT='0',HEADLESS='1',SEED='0',CHAIR_TEST_ENVS='512',CHAIR_REPEATS='3',OUTPUT_PATH=str(directory),OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
   print('START',epoch,flush=True)
   with (directory/'eval.log').open('w') as log:
    process=subprocess.Popen(['bash','tokenhsi/scripts/multi_agent/tokenhsi_climb_full_ft_test.sh',str(paths[epoch])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    (directory/'process.json').write_text(json.dumps(dict(pid=process.pid,checkpoint=str(paths[epoch]))));rc=process.wait()
   if rc:raise RuntimeError(f'FT{epoch} failed rc={rc}: {directory}')
  result=verify(directory);raw=result['raw'];rates=[raw[f'ObjectSet_test_{i}']['success_rate'] for i in [1,2]]
  result.update(checkpoint=str(paths[epoch]),output=str(directory),reused=epoch==5000,repeat_std_pp=float(100*np.std(rates,ddof=1)))
  with lock:state['results'][str(epoch)]=result;persist()
  print('DONE',epoch,round(result['success_rate']*100,4),flush=True)
 evaluate(5000)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(evaluate,e) for e in epochs if e!=5000]
  for f in concurrent.futures.as_completed(futures):f.result()
 state['status']='complete';persist();best=state['results'][str(state['best_epoch'])];(out/'best_checkpoint.json').write_text(json.dumps(dict(epoch=state['best_epoch'],**best),indent=2));(out/'completed.json').write_text(json.dumps(dict(rc=0,candidates=len(epochs))));print('SWEEP_COMPLETE',state['best_epoch'],best['success_rate'],flush=True)
if __name__=='__main__':main()
