"""Separate within-seed repeat scatter from between-seed scatter."""
import numpy as np


def bodies_after_reset(physical,kinematic,ids):
    """Use kinematic bodies only for reset rows; never modify Gym tensor views."""
    values=[]
    for current,section in zip(physical,[slice(0,3),slice(3,7),slice(7,10),slice(10,13)]):
        value=current.clone()
        value[ids]=kinematic[ids,...,section]
        values.append(value)
    return tuple(values)


def repeat_statistics(records):
    assert [r['repeat'] for r in records]==[0,1,2]
    result={}
    for metric in ['point','region']:
        values=np.array([r[metric+'_ever'] for r in records])*100
        result[metric]=dict(repeat_percent=values.tolist(),
            mean_percent=float(values[1:].mean()),
            within_seed_sample_std_percentage_points=float(values[1:].std(ddof=1)),
            included_repeats=[1,2],excluded_repeats=[0],
            all_three_mean_percent=float(values.mean()),
            all_three_sample_std_percentage_points=float(values.std(ddof=1)))
    return result


def seed_statistics(rows):
    assert rows and len({r['seed'] for r in rows})==len(rows)
    result={}
    for metric in ['point','region']:
        stats={}
        rates={p:np.array([r[p]['repeat_statistics'][metric]['mean_percent'] for r in rows]) for p in ['tokenhsi','ours']}
        for policy,values in rates.items():
            stats[policy]=dict(seed_mean_percent=values.tolist(),mean_percent=float(values.mean()),
                between_seed_sample_std_percentage_points=float(values.std(ddof=1)) if len(values)>1 else None,
                within_seed_sample_std_percentage_points=[r[policy]['repeat_statistics'][metric]['within_seed_sample_std_percentage_points'] for r in rows])
        delta=rates['ours']-rates['tokenhsi']
        stats.update(delta_mean_percentage_points=float(delta.mean()),
            winning_seeds=[r['seed'] for r,d in zip(rows,delta) if d>0],
            tied_seeds=[r['seed'] for r,d in zip(rows,delta) if d==0],
            losing_seeds=[r['seed'] for r,d in zip(rows,delta) if d<0])
        result[metric]=stats
    return result
