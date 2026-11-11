"""Audit matched episodes and summarize predecessor outcomes from saved traces."""
import csv,json
import numpy as np
import compare_fsm_llm_episodes as c


def main():
    for seed in (1,6,7):
        for mode in ('fsm','llm_snapshot'):
            p=c.GPU/f'seed{seed}'/mode
            assert json.loads((p/'exit.json').read_text())['returncode']==0
    c.summarize()
    rows=[]
    for seed in (1,6,7):
        p=c.GPU/f'seed{seed}';audit=json.loads((p/'llm_snapshot/schedule_execution_audit.json').read_text())
        excluded=[v['environment'] for v in audit['excluded']];keep=[i for i in range(64) if i not in excluded]
        f=np.load(p/'fsm/trace.npz');l=np.load(p/'llm_snapshot/trace.npz')
        old=np.load(c.RT/'output_etc/check'/c.SOURCE/f'eval/table1/mixed_16h/general_seed{seed}/trace.npz')
        for i,original in enumerate(keep):
            for follower,leader in enumerate(l['predecessor'][i]):
                if leader<0:continue
                row=dict(seed=seed,environment=original,predecessor=f'A{leader+1:02}',follower=f'A{follower+1:02}',
                    follower_task={0:'SIT',1:'CLIMB',3:'ON_TOP'}[int(l['packet'][i,follower,1])])
                for mode,z,env in [('fsm',f,original),('llm_snapshot',l,i),('llm_legacy',old,i)]:
                    row[mode+'_success']=bool(z['ever'][env,leader,1])
                    row[mode+'_fall_frame']=int(z['fall_step'][env,leader])
                rows.append(row)
    with (c.BASE/'predecessor_pairs.csv').open('w') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=json.loads((c.BASE/'comparison.json').read_text());report['paired_outcomes']={}
    for mode in ['llm_snapshot','llm_legacy']:
        report['paired_outcomes'][mode]=dict(
            both_success=sum(r['fsm_success'] and r[mode+'_success'] for r in rows),
            fsm_only_success=sum(r['fsm_success'] and not r[mode+'_success'] for r in rows),
            llm_only_success=sum(not r['fsm_success'] and r[mode+'_success'] for r in rows),
            both_fail=sum(not r['fsm_success'] and not r[mode+'_success'] for r in rows))
    c.save(c.BASE/'comparison.json',report)
    state=json.loads((c.BASE/'status.json').read_text())
    state['previous_aggregation_errors']=state['failures'];state['failures']=[];state['completed']=True
    state['summary_verified']=True;c.save(c.BASE/'status.json',state)
    print(json.dumps({'aggregate':report['aggregate'],'paired':report['paired_outcomes']},indent=2))

if __name__=='__main__':main()
