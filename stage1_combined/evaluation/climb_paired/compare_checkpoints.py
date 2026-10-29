"""User-requested 2k/3k comparison, then a fixed seed sweep of the selected model."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluation.climb_paired.run_eval import ROOT, save_summary, sha, write_json

CHECKPOINT_DIR=ROOT/'output/climb_adaptation/tokenhsi_climb_30d_100k_20261026_121912/train/TokenHSIClimb30DFullFT_26-12-19-15/nn'


def choose_checkpoint(pilots):
    """Compare full three-seed region means. A tie retains the incumbent 2k."""
    return 3000 if pilots[3000]['metrics']['region']['ours']['mean_percent'] > pilots[2000]['metrics']['region']['ours']['mean_percent'] else 2000


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    out=(args.output or ROOT/'output_etc/check/climb_eval'/('table_30d_2k_3k_40s_'+stamp)).resolve()
    assert ROOT/'output_etc/check' in out.parents
    out.mkdir(parents=True,exist_ok=False)
    checkpoints={epoch:CHECKPOINT_DIR/f'TokenHSIClimb30DFullFT_{epoch:08d}.pth' for epoch in [2000,3000]}
    for path in checkpoints.values():assert path.is_file(),path
    write_json(out/'plan.json',dict(seconds=40,num_envs=512,category='Table',test_models=8,
        pilot_epochs=[3000,2000],pilot_seeds=[0,1,2],remaining_seeds=[3,4,5,6,7,8,9],primary_metric='region',
        selection_rule='3k only if region mean strictly higher than 2k on seeds0/1/2; otherwise 2k',
        checkpoints={str(epoch):dict(path=str(path),sha256=sha(path)) for epoch,path in checkpoints.items()},
        note='Exploratory checkpoint/seed comparison. All planned seeds retained; chosen/winning seed results are not representative test performance.'))
    write_json(out/'state.json',dict(status='starting',supervisor_pid=os.getpid()))
    env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')

    def evaluate(epoch,name,seeds,reference=None):
        target=out/name
        command=[sys.executable,'-m','evaluation.climb_paired.run_eval','--checkpoint',str(checkpoints[epoch]),
            '--seconds','40','--seeds',*[str(seed) for seed in seeds],'--gpu',args.gpu,'--output',str(target)]
        if reference:command+=['--reference-dir',str(reference)]
        with (out/(name+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write_json(out/'state.json',dict(status='running',stage=name,epoch=epoch,driver_pid=child.pid,supervisor_pid=os.getpid()))
            rc=child.wait()
        if rc:raise RuntimeError(f'{name} rc={rc}; see {target} and {out/(name+".log")}')
        completed=json.loads((target/'completed.json').read_text());assert completed['rc']==0
        return json.loads((target/'summary.json').read_text())

    try:
        pilots={}
        pilots[3000]=evaluate(3000,'pilot_3k',[0,1,2])
        pilots[2000]=evaluate(2000,'pilot_2k',[0,1,2],out/'pilot_3k')
        selected=choose_checkpoint(pilots)
        write_json(out/'checkpoint_comparison.json',dict(selected_epoch=selected,pilots={str(key):value for key,value in pilots.items()},
            selection_metric='region mean over seeds0/1/2; strict improvement required for3k'))
        print('SELECTED',selected,flush=True)
        evaluate(selected,'remaining_seeds_3_to_9',list(range(3,10)))
        initial=out/('pilot_3k' if selected==3000 else 'pilot_2k')
        rows=json.loads((initial/'records.json').read_text())+json.loads((out/'remaining_seeds_3_to_9/records.json').read_text())
        assert [row['seed'] for row in rows]==list(range(10))
        save_summary(out,rows,list(range(10)),False,selected)
        write_json(out/'completed.json',dict(rc=0,selected_epoch=selected,num_eval_seeds=10,all_traces_rescored=True))
        write_json(out/'state.json',dict(status='complete',rc=0,selected_epoch=selected))
        print('RESULT',out,flush=True)
    except Exception as exc:
        write_json(out/'state.json',dict(status='failed',error=str(exc)))
        raise


if __name__=='__main__':main()
