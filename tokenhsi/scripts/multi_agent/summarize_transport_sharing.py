"""Select measured transport sizes and intersect six shared-support layouts.

All cutoffs are diagnostic filters, not changes to task success or proof of
physical impossibility. Uses only the Python standard library.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, default=ROOT/'output/shared_scenario_audit')
p.add_argument('--select', action='store_true', help='write geometry/rollout inputs after the transport grid finishes')
p.add_argument('--max-sizes', action='store_true', help='write the role-wise maximum size input without requiring a transport grid')
args = p.parse_args()
if args.max_sizes:
    folder=args.root/'max_sizes'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'cases.json'
    path.write_text(json.dumps([dict(case='support60_70_50_source30_30_50',
        support=[.6,.7,.5],payload=[.3,.3,.5])],indent=2))
    print(path)
    raise SystemExit(0)
out = args.root/'transport_grid'
summary = json.loads((out/'rollout_summary.json').read_text())
records = json.loads((out/'rollout_records.json').read_text())
sizes = {name: support for name,support,payload in summary['cases']}
rates = {}
for case,size in sizes.items():
    edges = [e for row in records if row['case']==case for e in row['edges'] if e['relation']==7]
    rates[tuple(size)] = dict(case=case,size=size,trials=len(edges),
        at_ever=sum(e['ever'] for e in edges),
        transport_delivery=sum(e['delivered_after_transport'] for e in edges),
        rate=sum(e['delivered_after_transport'] for e in edges)/len(edges))
(out/'transport_rates.json').write_text(json.dumps(list(rates.values()),indent=2))
if args.select:
    strong = [v['size'] for v in rates.values() if v['rate']>=.5]
    pools = {
        'supports_for_geometry.json':[s for s in strong if s[0]>=.4 and s[1]>=.6],
        'supports_tall_payload.json':[s for s in strong if .4<=s[0]<=.65 and .6<=s[1]<=.75 and s[2] in (.45,.5)],
        'supports_thin.json':[s for s in strong if .3<=s[0]<.4 and s[1]>=.6]}
    for name,values in pools.items():
        (out/name).write_text(json.dumps(values,indent=2))
    cases=[]
    for support in ([.5,.65,.5],[.55,.65,.5],[.6,.6,.5],[.6,.65,.5],[.65,.65,.5]):
        for height in (.3,.4,.45,.5):
            payload=[.3,.3,height]
            name='support%d_%d_%d_source%d_%d_%d'%tuple(round(v*100) for v in support+payload)
            cases.append(dict(case=name,support=support,payload=payload))
    for support in ([.55,.75,.5],[.6,.75,.5]):
        for height in (.3,.45,.5):
            payload=[.35,.35,height]
            name='support%d_%d_%d_source%d_%d_%d'%tuple(round(v*100) for v in support+payload)
            cases.append(dict(case=name,support=support,payload=payload))
    (out/'sharing_cases.json').write_text(json.dumps(cases,indent=2))
    thin_cases=[]
    for support in ([.35,.65,.5],[.35,.7,.5],[.4,.65,.5],[.45,.65,.5]):
        payload=[.3,.3,.45]
        name='support%d_%d_%d_source%d_%d_%d'%tuple(round(v*100) for v in support+payload)
        thin_cases.append(dict(case=name,support=support,payload=payload))
    (out/'sharing_thin_cases.json').write_text(json.dumps(thin_cases,indent=2))
    (out/'verify_thin_supports.json').write_text(json.dumps([[.35,.65,.5],[.35,.7,.5]],indent=2))
    print('Transport sizes:',len(rates),'geometry pools:',{k:len(v) for k,v in pools.items()})
else:
    humans={}
    payloads={}
    for folder in ('transport_geometry','transport_geometry_tall','transport_geometry_thin'):
        for row in json.loads((args.root/folder/'results.json').read_text())['results']:
            key=tuple(row['support']),row['clearance_m']
            if row['payload'] is None:
                humans.setdefault(key,{})[row['pair']]=row['coverage']
            else:
                payloads.setdefault(key+(tuple(row['payload']),),{})[row['pair']]=row['coverage']
    rows=[]
    for (support,clearance,payload),coverages in payloads.items():
        cov=dict(humans.get((support,clearance),{}),**coverages)
        if len(cov)!=6:
            raise RuntimeError('Missing shared pair: '+str((support,payload,clearance,cov)))
        support_rate=rates[support]['rate']
        payload_rate=rates[payload]['rate']
        rows.append(dict(support=list(support),payload=list(payload),clearance_m=clearance,
            support_transport_delivery_rate=support_rate,payload_transport_delivery_rate=payload_rate,
            all_six_have_layout=all(v>0 for v in cov.values()),min_pose_coverage=min(cov.values()),
            coverage=cov,operational_candidate=support_rate>=.7 and payload_rate>=.5 and all(v>0 for v in cov.values())))
    rows.sort(key=lambda r:(r['support'],r['payload'],r['clearance_m']))
    (out/'intersection.json').write_text(json.dumps(rows,indent=2))
    flat=[dict(support=r['support'],payload=r['payload'],clearance_m=r['clearance_m'],
        support_transport_delivery_rate=r['support_transport_delivery_rate'],
        payload_transport_delivery_rate=r['payload_transport_delivery_rate'],
        all_six_have_layout=r['all_six_have_layout'],operational_candidate=r['operational_candidate'],
        **r['coverage']) for r in rows]
    with (out/'intersection.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    sharing=[]
    rollout=json.loads((args.root/'sharing_after_transport/rollout_records.json').read_text())
    rollout+=json.loads((args.root/'sharing_thin/rollout_records.json').read_text())
    for case in dict.fromkeys(r['case'] for r in rollout):
        episodes=[r for r in rollout if r['case']==case]
        top=[e for r in episodes for e in r['edges'] if e['relation']==8]
        sharing.append(dict(case=case,scenes=len(episodes),ontop_trials=len(top),
            ontop_ever=sum(e['ever'] for e in top),
            ontop_fully_supported_ever=sum(e['ever_fully_supported'] for e in top),
            pairs=[dict(pair=pair,trials=sum(r['pair']==pair for r in episodes),
                joint_ever=sum(r['joint_ever'] for r in episodes if r['pair']==pair),
                joint_one_second=sum(r['max_joint_success_run_steps']>=30 for r in episodes if r['pair']==pair))
                for pair in dict.fromkeys(r['pair'] for r in episodes)]))
    (out/'sharing_rates.json').write_text(json.dumps(sharing,indent=2))
    print('Intersection entries:',len(rows),'operational 4cm candidates:',
          sum(r['operational_candidate'] and r['clearance_m']==.04 for r in rows))
