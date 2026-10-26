"""Evaluate each authorized 1k checkpoint when GPU scheduling constraints allow."""
import argparse,datetime,fcntl,json,os,subprocess,time
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output_etc/check/tokenhsi_climb_ft_sweep_20261025/ft_05000'
SOURCE=ROOT.parent/'runs/tokenhsi_climb_78k_20261024/train/TokenHSIClimbFullFT_24-16-15-27/nn/TokenHSIClimbFullFT_00005000.pth'

def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def verify(directory):
    files=list(directory.glob('metrics_*.json'));assert len(files)==1
    data=json.loads(files[0].read_text())
    assert yaml.safe_load((directory/'env.yaml').read_text())==yaml.safe_load((BASE/'env.yaml').read_text())
    for i in range(3):
        t=np.load(directory/f'trace_{i}.npz');distance=np.linalg.norm(t['root'].astype('float64')-t['target'].astype('float64'),axis=-1)
        assert distance.shape[1:]==(512,) and len(distance)<=299 and np.isfinite(distance).all()
        good=((distance<=.2)&t['alive']).any(0);minimum=np.where(t['alive'],distance,np.inf).min(0)
        assert int(good.sum())==data[f'ObjectSet_test_{i}']['success_trials']
        assert abs(float(minimum[good].mean())-data[f'ObjectSet_test_{i}']['success_precision'])<1e-7
        assert ((t['done'].astype(bool)&t['alive']).sum(0)==1).all()
    return dict(success_rate=float(np.mean([data[f'ObjectSet_test_{i}']['success_rate'] for i in (1,2)])),precision_m=float(np.mean([data[f'ObjectSet_test_{i}']['success_precision'] for i in (1,2)])),raw=data,trace_verified=True)

def gpu_free(gpu):
    result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    for line in result.splitlines():
        index,free=line.split(',')
        if int(index)==gpu:return int(free)
    raise RuntimeError('GPU not found')

def report(run,state):
    baseline=state['baseline'];best=dict(epoch=5000,checkpoint=str(SOURCE),success_rate=baseline['success_rate'])
    lines=['# CLIMB low-LR continuation 평가','',f"갱신: {stamp()}",'','FT5k에서lr5e-6로추가20k(기존FT epoch6000~25000) 학습. 1k마다저장한20개체크포인트를실제Box/Cabinet/Table각1/3혼합·1명1고정물체·10초·512환경·seed0·3repeat로순차평가한다. GPU여유26000MiB이상일때만실행하며학습프로세스를멈추지않는다.','', '| 기존 FT epoch | 이번 추가 학습 | 성공률 | 성공 최소거리 | 상태 |','|---|---:|---:|---:|---|',f"| 5000 (시작점) | 0 | {baseline['success_rate']*100:.2f}% | {baseline['precision_m']*100:.2f}cm | 기존 검증 결과 |"]
    for epoch in range(6000,25001,1000):
        row=state['evaluations'].get(str(epoch))
        if row and row.get('status')=='complete':
            lines.append(f"| {epoch} | {epoch-5000} | {row['success_rate']*100:.2f}% | {row['precision_m']*100:.2f}cm | 재채점 통과 |")
            if row['success_rate']>best['success_rate']:best=dict(epoch=epoch,checkpoint=row['checkpoint'],success_rate=row['success_rate'])
        else:lines.append(f"| {epoch} | {epoch-5000} | — | — | {row.get('status','대기') if row else '대기'} |")
    state['best']=best;state['updated_at']=stamp();atomic(run/'evaluation_status.json',state);atomic(run/'best_checkpoint.json',best)
    (run/'EVALUATION_REPORT.md').write_text('\n'.join(lines)+'\n\n주요값은repeat1·2평균이며학습seed산포가아니다. 기존시작점평가도전trace재채점했다.\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir');parser.add_argument('--gpu',type=int,default=0);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        result=verify(BASE);assert result['success_rate']==0.8720703125
        print('WATCHER_SELF_TEST',result['success_rate']);return
    run=Path(args.run_dir).resolve();lock=(run/'evaluation_watcher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads((run/'evaluation_status.json').read_text()) if (run/'evaluation_status.json').exists() else dict(status='watching',baseline=verify(BASE),evaluations={})
    evalroot=ROOT/'output_etc/check'/run.name;evalroot.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(TOKENHSI_GPU=str(args.gpu),TOKENHSI_CONDA_ENV='tokenhsi118',PYTORCH_JIT='0',PYTHONUNBUFFERED='1',HEADLESS='1',SEED='0',CHAIR_TEST_ENVS='512',CHAIR_REPEATS='3')
    for key in ['EPISODE_LENGTH','TASK_GRAPH','CHECK','MAX_ITERATIONS','RESUME_CHECKPOINT']:env.pop(key,None)
    report(run,state)
    while True:
        pending=[epoch for epoch in range(6000,25001,1000) if state['evaluations'].get(str(epoch),{}).get('status')!='complete']
        if not pending:
            state['status']='complete';report(run,state);return
        epoch=pending[0];paths=list((run/'train').glob(f'*/nn/TokenHSIClimbLowLR_{epoch:08d}.pth'))
        if not paths:
            if (run/'train_exit.json').exists():
                state['status']='training_ended_with_missing_checkpoint';state['missing_epoch']=epoch;report(run,state);return
            time.sleep(30);continue
        checkpoint=paths[0]
        old=state['evaluations'].get(str(epoch),{})
        if old.get('retry_after',0)>time.time():time.sleep(30);continue
        try:
            if gpu_free(args.gpu)<26000:time.sleep(30);continue
            import torch
            weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert weights['epoch']==epoch
            del weights
        except Exception as error:
            print('checkpoint/gpu pending',repr(error),flush=True);time.sleep(30);continue
        attempt=old.get('attempt',0)+1;directory=evalroot/f'epoch_{epoch:05d}_attempt_{attempt}';directory.mkdir(exist_ok=False)
        state['evaluations'][str(epoch)]=dict(status='running',attempt=attempt,checkpoint=str(checkpoint),output=str(directory));report(run,state)
        env['OUTPUT_PATH']=str(directory)
        with (directory/'eval.log').open('w') as log:
            rc=subprocess.call(['bash','tokenhsi/scripts/multi_agent/tokenhsi_climb_low_lr_test.sh',str(checkpoint)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            assert rc==0,('eval exit',rc)
            result=verify(directory);result.update(status='complete',attempt=attempt,checkpoint=str(checkpoint),output=str(directory),completed_at=stamp())
            state['evaluations'][str(epoch)]=result
        except Exception as error:
            state['evaluations'][str(epoch)].update(status='failed_retry_pending',error=repr(error),retry_after=time.time()+1800)
        report(run,state)

if __name__=='__main__':main()
