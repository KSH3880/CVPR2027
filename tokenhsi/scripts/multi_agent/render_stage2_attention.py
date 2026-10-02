"""Standalone scientific plots and an HTML report for Stage-2 CA trends."""
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FAMILIES = ['place_climb', 'place_sit', 'place_stack', 'independent']
NAMES = ['CLIMB', 'SIT', 'STACK', 'Independent']
COLORS = ['#2563eb', '#d97706', '#059669', '#7c3aed']


def interval(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [None, None, None]
    mean = float(values.mean())
    if len(values) == 1:
        return [mean, mean, mean]
    rng = np.random.RandomState(71)
    boot = values[rng.randint(len(values), size=(1000, len(values)))].mean(-1)
    return [mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def wilson(k, n):
    if n == 0:
        return [None, None, None]
    p, z = k / n, 1.96
    c = (p + z*z/(2*n))/(1+z*z/n)
    r = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return [p, c-r, c+r]


def grouped(values, indices, envs_per_condition):
    """Equal weight per episode, rather than overweighting long survivors."""
    if values.ndim > 1:
        values = np.nanmean(values, axis=tuple(range(1, values.ndim)))
    results = {}
    condition = indices[:, 2] // envs_per_condition
    for family in range(4):
        rows = condition == 6 if family == 3 else condition // 2 == family
        episode_ids = indices[:, 0] * (7*envs_per_condition) + indices[:, 2]
        means = [float(np.nanmean(values[rows & (episode_ids == episode)]))
                 for episode in np.unique(episode_ids[rows])]
        results[FAMILIES[family]] = interval(means)
    return results


def save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def render(output):
    output = Path(output)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True,
                         'grid.alpha': .18, 'figure.dpi': 110})
    metadata = [json.loads(p.read_text()) for p in sorted(output.glob('epoch_*/results.json'))]
    if not metadata:
        raise RuntimeError('No completed rollout records')
    latest = metadata[-1]
    latest_trace = np.load(output / ('epoch_%05d/traces.npz' % latest['epoch']))
    env_count = latest['envs_per_condition']
    starts = []
    for meta in metadata:
        trace = np.load(output / ('epoch_%05d/traces.npz' % meta['epoch']))
        starts.append(dict(epoch=meta['epoch'],
            initial_observation_max_abs_difference=float(np.abs(trace['initial_obs']-latest_trace['initial_obs']).max()),
            graph_equal=bool(all(np.array_equal(trace[k], latest_trace[k]) for k in ['src','dst','valid','relation','owner'])),
            asset_sizes_equal=bool(np.array_equal(trace['box_sizes'], latest_trace['box_sizes']))))
    summary = dict(latest_epoch=latest['epoch'], matched_start_checks=starts,
                   rollout=[], fixed_observations=[], method={
        'attention_axes': 'sample, head, query human, task edge',
        'routing_intervention': 'Remove one edge from CA keys/values; renormalize softmax. Encoder/query stay fixed.',
        'action_metric': 'RMS difference of 32 action means after clamp [-1,1]. No counterfactual physics rollout.',
        'weighting': 'Equal episode weight; 95% episode-bootstrap CI for attention/action metrics; Wilson for rates.',
        'events': 'First current AT success; stable AT = current AT and box speed <0.1m/s for 0.5s.',
        'scope': 'Default training box range and RSI. 2 humans, 4 objects, 7 conditions, 3 repeat seeds.',
        'caveats': 'Fixed bank comes from latest-policy trajectories; events reflect physical sharing, not a temporal DAG.'})
    for meta in metadata:
        result = {'epoch': meta['epoch'], 'families': {}}
        for family in FAMILIES:
            trials = [r for r in meta['trials'] if r['family']==family]
            result['families'][family] = {key: wilson(sum(r[key] for r in trials), len(trials))
                                         for key in ['joint_ever','joint_final','joint_1s','terminated']}
            result['families'][family]['n'] = len(trials)
            if family != 'independent':
                result['families'][family]['at_ever'] = wilson(sum(r['edge_ever'][1] for r in trials), len(trials))
                edge = 3 if family=='place_stack' else 2
                result['families'][family]['downstream_ever'] = wilson(sum(r['edge_ever'][edge] for r in trials), len(trials))
                events = [r for r in trials if r['first_edge_seconds'][edge]>=0]
                result['families'][family]['downstream_before_at'] = dict(n=len(events), count=sum(
                    r['first_edge_seconds'][1]<0 or r['first_edge_seconds'][edge]<r['first_edge_seconds'][1]
                    for r in events),
                    both_events=sum(r['first_edge_seconds'][1]>=0 for r in events),
                    before_with_at=sum(r['first_edge_seconds'][1]>=0 and
                        r['first_edge_seconds'][edge]<r['first_edge_seconds'][1] for r in events),
                    without_at=sum(r['first_edge_seconds'][1]<0 for r in events))
        summary['rollout'].append(result)

    summary['ablations'] = []
    for path in sorted(output.glob('ablations/*/epoch_*/results.json')):
        meta = json.loads(path.read_text())
        trace = np.load(path.parent / 'traces.npz')
        comparison = dict(mode=meta['intervention'], epoch=meta['epoch'],
            initial_observation_max_abs_difference=float(np.abs(trace['initial_obs']-latest_trace['initial_obs']).max()),
            graph_equal=bool(all(np.array_equal(trace[k], latest_trace[k]) for k in ['src','dst','valid','relation','owner'])),
            asset_sizes_equal=bool(np.array_equal(trace['box_sizes'],latest_trace['box_sizes'])),families={})
        for family in FAMILIES:
            trials = [r for r in meta['trials'] if r['family']==family]
            baseline = [r for r in latest['trials'] if r['family']==family]
            comparison['families'][family] = {key: wilson(sum(r[key] for r in trials),len(trials))
                                             for key in ['joint_ever','joint_1s','joint_final','terminated']}
            comparison['families'][family]['n'] = len(trials)
            comparison['families'][family]['paired_changes'] = {
                key:dict(improved=sum(not before[key] and after[key] for before,after in zip(baseline,trials)),
                         worsened=sum(before[key] and not after[key] for before,after in zip(baseline,trials)))
                for key in ['joint_ever','joint_1s','joint_final']}
        summary['ablations'].append(comparison)

    bank_path = output / 'observation_bank.npz'
    if bank_path.is_file():
        bank = np.load(bank_path)
        indices = bank['indices']
        owner = latest_trace['owner'][indices[:, 2]]
        other = owner[:, None, :] != np.arange(2)[None, :, None]
        summary['routing_by_role'] = []
        success_episode = {(r['repeat'],r['env']):r['joint_ever'] for r in latest['trials']}
        latest_success = np.array([success_episode[(int(r),int(env))] for r,_,env in indices])
        for path in sorted(output.glob('replay_*.npz')):
            data = np.load(path)
            attn = data['attention']
            teammate_mass = (attn*other[:, None]).sum(-1)
            p, q = attn[:,0], attn[:,1]
            mean = (p+q)/2
            js = .5*((p*np.log(np.maximum(p,1e-30)/np.maximum(mean,1e-30))).sum(-1)+
                      (q*np.log(np.maximum(q,1e-30)/np.maximum(mean,1e-30))).sum(-1))/np.log(2)
            family_metrics = {key: grouped(value, indices, env_count) for key, value in dict(
                entropy=data['entropy'], teammate_mass=teammate_mass,
                ca_off_delta=data['ca_off_delta'], uniform_delta=data['uniform_delta'], head_js=js).items()}
            summary['fixed_observations'].append(dict(epoch=int(path.stem.split('_')[1]), metrics=family_metrics))
            outcome_metrics = {}
            for outcome in [True,False]:
                mask = latest_success==outcome
                outcome_metrics['latest_success' if outcome else 'latest_failure'] = {
                    key:grouped(value[mask],indices[mask],env_count) for key,value in dict(
                        entropy=data['entropy'],ca_off_delta=data['ca_off_delta'],uniform_delta=data['uniform_delta']).items()}
            summary['fixed_observations'][-1]['latest_outcome_strata']=outcome_metrics
            role_metrics = dict(epoch=int(path.stem.split('_')[1]), families={})
            conditions = indices[:,2] // env_count
            episode_id = indices[:,0] * (7*env_count) + indices[:,2]
            for family_id,family in enumerate(FAMILIES[:3]):
                rows = np.where(conditions//2 == family_id)[0]
                carrier = conditions[rows] % 2
                role_metrics['families'][family] = {}
                for role,agent in [('carrier',carrier),('mate',1-carrier)]:
                    weights = attn[rows,:,agent,:]
                    entropies = data['entropy'][rows,:,agent]
                    impacts = data['edge_delta'][rows,agent,:]
                    episode_weights = np.stack([weights[episode_id[rows]==ep].mean(0) for ep in np.unique(episode_id[rows])])
                    episode_entropy = np.stack([entropies[episode_id[rows]==ep].mean(0) for ep in np.unique(episode_id[rows])])
                    active_edges = 4 if family_id==2 else 3
                    episode_impact = np.stack([impacts[episode_id[rows]==ep,:active_edges].mean(0) for ep in np.unique(episode_id[rows])])
                    edge_impact = episode_impact.mean(0)
                    role_metrics['families'][family][role] = dict(
                        attention=episode_weights.mean(0).tolist(),entropy=episode_entropy.mean(0).tolist(),
                        edge_action_delta=[float(v) if np.isfinite(v) else None for v in edge_impact]+[None]*(4-active_edges))
            summary['routing_by_role'].append(role_metrics)
        summary['bank_observations'] = len(indices)
        latest_replay = np.load(output/('replay_%05d.npz'%latest['epoch']))
        summary['latest_replay_attention_max_abs_difference'] = float(np.abs(
            latest_replay['attention'] - latest_trace['attention'][latest_trace['alive']]).max())

    # Outcome curves on the policies' own trajectories.
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, key, title in zip(axs.flat, ['joint_ever','joint_1s','joint_final','terminated'],
        ['Simultaneous scene success: ever','Simultaneous success held >= 1 second',
         'Simultaneous success at episode end','Scene termination / fall']):
        for family, name, color in zip(FAMILIES,NAMES,COLORS):
            values = np.array([r['families'][family][key] for r in summary['rollout']])
            epochs = [r['epoch'] for r in summary['rollout']]
            ax.plot(epochs, values[:,0], 'o-', label=name, color=color)
            ax.fill_between(epochs, values[:,1], values[:,2], alpha=.1, color=color)
        ax.set_title(title); ax.set_ylim(-.03, 1.03); ax.set_ylabel('Scene fraction')
    axs[0,0].legend(ncol=2, fontsize=9)
    for ax in axs[1]: ax.set_xlabel('Stage-2 epoch')
    fig.suptitle('Actual rollouts: default box range / RSI, both cooperative roles', fontsize=14)
    fig.tight_layout(); save(fig, output/'rollout_trends.png')

    if summary['ablations']:
        modes = ['none']+[x['mode'] for x in summary['ablations']]
        labels = ['Original', 'CA off', 'Peer edges removed', 'Uniform routing']
        fig,axs = plt.subplots(1,3,figsize=(14,4.5),sharey=True)
        for ax,key,title in zip(axs,['joint_ever','joint_1s','joint_final'],
                               ['Simultaneous success: ever','Success held >=1s','Success at episode end']):
            x=np.arange(4);width=.18
            for i,mode in enumerate(modes):
                values = summary['rollout'][-1]['families'] if mode=='none' else next(r['families'] for r in summary['ablations'] if r['mode']==mode)
                heights=[values[family][key][0] for family in FAMILIES]
                bars=ax.bar(x+(i-1.5)*width,heights,width,label=labels[i])
                for b,v in zip(bars,heights): ax.text(b.get_x()+b.get_width()/2,v+.02,'%.0f'%(100*v),ha='center',fontsize=8)
            ax.set_xticks(x);ax.set_xticklabels(NAMES,rotation=15);ax.set_title(title,fontsize=10);ax.set_ylim(0,1.1)
        axs[0].set_ylabel('Scene success fraction');axs[0].legend(fontsize=8)
        fig.suptitle('Latest checkpoint: actual rollout interventions with matched initial observations',fontsize=13)
        fig.tight_layout();save(fig,output/'ca_interventions.png')

    if summary['fixed_observations']:
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        keys = ['entropy','teammate_mass','ca_off_delta','uniform_delta','head_js']
        titles = ['Normalized attention entropy','Attention mass on teammate-owned edges',
                  'Action change: all CA context off','Action change: uniform CA weights',
                  'Difference between two heads (JS / log 2)']
        epochs = [r['epoch'] for r in summary['fixed_observations']]
        for ax, key, title in zip(axs.flat, keys, titles):
            for family,name,color in zip(FAMILIES,NAMES,COLORS):
                values = np.array([r['metrics'][key][family] for r in summary['fixed_observations']])
                ax.plot(epochs,values[:,0],label=name,color=color)
                ax.fill_between(epochs,values[:,1],values[:,2],alpha=.1,color=color)
            ax.set_title(title,fontsize=10); ax.set_xlabel('Stage-2 epoch')
            ax.set_ylabel('Clipped action RMS' if 'delta' in key else '0-1')
            if key in ('entropy','teammate_mass','head_js'): ax.set_ylim(-.02,1.02)
        axs[0,0].legend(fontsize=9)
        axs[1,2].axis('off')
        axs[1,2].text(0,.9,'Same observations at every epoch\nLatest-policy observation bank\nEqual weight per episode\nBands: 95% episode-bootstrap CI\n\nEntropy 1 = uniform routing\nEntropy 0 = one-edge routing\nJS 0 = identical head weights\n\nAction interventions affect CA only.',va='top')
        fig.suptitle('Checkpoint changes on a fixed observation bank',fontsize=14)
        fig.tight_layout(); save(fig,output/'fixed_state_trends.png')

        fig,axs=plt.subplots(6,2,figsize=(13,17),sharex=True,sharey=True)
        for family_id,family in enumerate(FAMILIES[:3]):
            edge_labels=['Carrier HOLDING','Carrier AT','Mate HOLDING' if family_id==2 else 'Mate '+NAMES[family_id],'Mate ON_TOP']
            for role_id,role in enumerate(['carrier','mate']):
                for head in range(2):
                    ax=axs[family_id*2+role_id,head]
                    for edge in range(4 if family_id==2 else 3):
                        values=[r['families'][family][role]['attention'][head][edge] for r in summary['routing_by_role']]
                        ax.plot(epochs,values,label=edge_labels[edge])
                    ax.set_title('%s / %s query / head %d'%(NAMES[family_id],role,head))
                    ax.set_ylim(0,1);ax.set_ylabel('Attention weight');ax.legend(fontsize=8,ncol=2)
        for ax in axs[-1]:ax.set_xlabel('Stage-2 epoch')
        fig.suptitle('Per-role, per-head edge routing on identical observations',fontsize=14)
        fig.tight_layout();save(fig,output/'routing_by_role.png')

        # Per-head / per-agent maps on identical latest-policy states.
        chosen = sorted(set([summary['fixed_observations'][0]['epoch'],4000,8000,latest['epoch']]) &
                        {r['epoch'] for r in summary['fixed_observations']})
        for family_id, family in enumerate(FAMILIES[:3]):
            family_trials = [r for r in latest['trials'] if r['family']==family]
            # Explicitly label selection: longest scene-success hold, then longest episode.
            representative = max(family_trials, key=lambda r:(r['max_joint_seconds'],r['length_steps'],-r['env']))
            repeat, env = representative['repeat'],representative['env']
            rows = (indices[:,0]==repeat)&(indices[:,2]==env)
            t = indices[rows,1]*latest['stride']*latest['dt']
            carrier,mate = representative['carrier'],1-representative['carrier']
            labels = ['Carrier HOLDING','Carrier AT', 'Mate HOLDING' if family_id==2 else 'Mate '+NAMES[family_id], 'Mate ON_TOP']
            fig, axs = plt.subplots(len(chosen)*2+2, 2, figsize=(13, 2.0*(len(chosen)*2+2)),sharex=True)
            for i,epoch in enumerate(chosen):
                attn = np.load(output/('replay_%05d.npz'%epoch))['attention'][rows]
                for role,agent in enumerate([carrier,mate]):
                    for head in range(2):
                        ax=axs[2*i+role,head]
                        image=ax.imshow(attn[:,head,agent].T,origin='lower',aspect='auto',vmin=0,vmax=1,
                            extent=[t[0],t[-1]+latest['stride']*latest['dt'],-.5,3.5],cmap='viridis')
                        ax.set_yticks(range(4)); ax.set_yticklabels(labels,fontsize=8)
                        ax.set_title('Epoch %d / %s H%d / head %d'%(epoch,'carrier' if role==0 else 'mate',agent,head))
                        ax.grid(False)
                        for event in [representative['first_edge_seconds'][1], representative['stable_at_seconds']]:
                            if event>=0: ax.axvline(event,color='white',linestyle='--',linewidth=.8)
            sample_idx = indices[rows,1]
            phi = latest_trace['phi'][repeat,sample_idx,env]
            for edge in np.where(latest_trace['valid'][env])[0]:
                axs[-2,0].plot(t,phi[:,edge],label=labels[edge])
            axs[-2,0].set_ylabel('State score');axs[-2,0].legend(fontsize=8)
            axs[-2,1].plot(t,latest_trace['goal_distance'][repeat,sample_idx,env],label='Goal distance (m)')
            axs[-2,1].plot(t,latest_trace['box_speed'][repeat,sample_idx,env],label='Box speed (m/s)')
            axs[-2,1].legend(fontsize=8)
            latest_replay=np.load(output/('replay_%05d.npz'%latest['epoch']))
            for head in range(2):
                for edge in np.where(latest_trace['valid'][env])[0]:
                    axs[-1,head].plot(t,latest_replay['edge_delta'][rows, [carrier,mate][head],edge],label=labels[edge])
                axs[-1,head].set_ylabel('Action RMS / edge masked');axs[-1,head].set_xlabel('Time on latest-policy rollout (s)')
            fig.suptitle('%s: identical states replayed across epochs\nRepresentative = longest simultaneous-success hold; repeat %d, env %d; ever=%s, hold=%.2fs'%
                (NAMES[family_id],repeat,env,representative['joint_ever'],representative['max_joint_seconds']),fontsize=12)
            fig.tight_layout(rect=[0,0,1,.96]);save(fig,output/('time_maps_%s.png'%family))

        # Event-aligned curves on each policy's own trajectories.
        fig, axs=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
        aligned=[]
        grid=np.linspace(-3,3,19)
        for col,family in enumerate(FAMILIES[:3]):
            for row,event_key in enumerate(['first_at','stable_at']):
                ax=axs[row,col]
                counts=[]
                for meta in metadata:
                    trace=np.load(output/('epoch_%05d/traces.npz'%meta['epoch']))
                    curves=[]
                    for trial in meta['trials']:
                        if trial['family']!=family:continue
                        event=trial['first_edge_seconds'][1] if event_key=='first_at' else trial['stable_at_seconds']
                        if event<0:continue
                        r,env=trial['repeat'],trial['env']
                        live=trace['alive'][r,:,env]
                        ts=np.arange(live.shape[0])*meta['stride']*meta['dt']
                        weights=trace['attention'][r,:,env,:,1-trial['carrier'],1].mean(-1)
                        if live.sum()<2:continue
                        curves.append(np.interp(grid,ts[live]-event,weights[live],left=np.nan,right=np.nan))
                    if curves:
                        curves=np.stack(curves)
                        with np.errstate(invalid='ignore'):
                            average=np.array([np.nanmean(curves[:,i]) if np.isfinite(curves[:,i]).any() else np.nan for i in range(19)])
                        ax.plot(grid,average,label='%d (n=%d)'%(meta['epoch'],len(curves)))
                    counts.append(dict(epoch=meta['epoch'],family=family,event=event_key,n=len(curves)))
                aligned.extend(counts)
                ax.axvline(0,color='black',linestyle='--',linewidth=1)
                ax.set_title('%s / %s'%(NAMES[col], 'first AT success' if row==0 else 'stable AT'))
                ax.set_ylabel('Mate attention to carrier AT');ax.set_ylim(0,.3);ax.legend(fontsize=7,ncol=2)
                if row==1:ax.set_xlabel('Time relative to event (s)')
        summary['event_counts']=aligned
        fig.suptitle('Event-aligned CA routing on actual policy rollouts (heads averaged)',fontsize=14)
        fig.tight_layout();save(fig,output/'event_aligned.png')

        # Latest-policy success/failure examples use a deterministic median-length trial.
        fig,axs=plt.subplots(6,3,figsize=(16,13),sharey='row')
        for col,family in enumerate(FAMILIES[:3]):
            downstream = 3 if col==2 else 2
            labels=['Carrier HOLDING','Carrier AT','Mate HOLDING' if col==2 else 'Mate '+NAMES[col],'Mate ON_TOP']
            for outcome_row,success in enumerate([True,False]):
                candidates=[r for r in latest['trials'] if r['family']==family and r['joint_ever']==success]
                candidates.sort(key=lambda r:(r['length_steps'],r['repeat'],r['env']))
                if not candidates:
                    for offset in range(3): axs[outcome_row*3+offset,col].axis('off')
                    continue
                trial=candidates[len(candidates)//2]
                r,env=trial['repeat'],trial['env']
                live=latest_trace['alive'][r,:,env]
                t=np.arange(len(live))[live]*latest['stride']*latest['dt']
                mate=1-trial['carrier']
                for head in range(2):
                    ax=axs[outcome_row*3+head,col]
                    weights=latest_trace['attention'][r,live,env,head,mate,:]
                    for edge in range(4 if col==2 else 3): ax.plot(t,weights[:,edge],label=labels[edge])
                    ax.set_ylim(0,1);ax.set_title('%s / %s / mate head %d'%(NAMES[col],'success' if success else 'failure',head),fontsize=10)
                    if col==0:ax.set_ylabel('Attention')
                    for event,style in [(trial['first_edge_seconds'][1],'--'),(trial['first_edge_seconds'][downstream],':')]:
                        if event>=0:ax.axvline(event,color='black',linestyle=style,linewidth=.8)
                    ax.legend(fontsize=7,ncol=2)
                ax=axs[outcome_row*3+2,col]
                phi=latest_trace['phi'][r,live,env]
                ax.plot(t,phi[:,1],label='AT state score')
                ax.plot(t,phi[:,downstream],label='Downstream state score')
                ax.set_title('repeat %d / env %d / hold %.2fs'%(r,env,trial['max_joint_seconds']),fontsize=9)
                ax.legend(fontsize=8);ax.set_xlabel('Episode time (s)')
        fig.suptitle('Latest policy: success and failure examples\nMedian episode length within each outcome; dashed = first AT, dotted = first downstream success',fontsize=13)
        fig.tight_layout(rect=[0,0,1,.95]);save(fig,output/'success_failure_examples.png')

    if summary['ablations']:
        summary['method']['action_metric'] += ' Latest-checkpoint full rollout interventions also measured separately.'
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    tables=[]
    for family,name in zip(FAMILIES,NAMES):
        rows=[]
        for result in summary['rollout']:
            values=result['families'][family]
            cells=[str(result['epoch']),str(values['n'])]+['%.1f%%'%(100*values[k][0]) for k in ['joint_ever','joint_1s','joint_final','terminated']]
            if family!='independent':
                cells+=['%.1f%%'%(100*values[k][0]) for k in ['at_ever','downstream_ever']]
                event=values['downstream_before_at'];cells+=['%d/%d'%(event['before_with_at'],event['both_events']),str(event['without_at'])]
            rows.append('<tr>'+''.join('<td>'+cell+'</td>' for cell in cells)+'</tr>')
        headers=['Epoch','N','Joint ever','Joint >=1s','Joint final','Fall']
        if family!='independent':headers+=['AT ever','Downstream ever','Downstream before AT (both reached)','Downstream, AT never reached']
        tables.append('<h2>'+name+'</h2><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table>')
    first=summary['rollout'][0];last=summary['rollout'][-1]
    conclusions=[]
    for family,name in zip(FAMILIES[:3],NAMES[:3]):
        begin,end=first['families'][family],last['families'][family]
        conclusions.append('%s 동시 ever 성공: epoch %d %.1f%% → epoch %d %.1f%%. 최신 1초 유지 %.1f%%, 종료 시 성공 %.1f%%.'%
            (name,first['epoch'],100*begin['joint_ever'][0],last['epoch'],100*end['joint_ever'][0],100*end['joint_1s'][0],100*end['joint_final'][0]))
    if summary['fixed_observations']:
        for family,name in zip(FAMILIES[:3],NAMES[:3]):
            early=summary['fixed_observations'][0]['metrics'];late=summary['fixed_observations'][-1]['metrics']
            conclusions.append('%s 동일 관측에서 entropy %.3f → %.3f, CA 전체 제거 action RMS %.4f → %.4f, uniform attention 대체 RMS %.4f → %.4f.'%
                (name,early['entropy'][family][0],late['entropy'][family][0],early['ca_off_delta'][family][0],late['ca_off_delta'][family][0],
                 early['uniform_delta'][family][0],late['uniform_delta'][family][0]))
        for family,name in zip(FAMILIES[:3],NAMES[:3]):
            roles=summary['routing_by_role'][-1]['families'][family]
            carrier_at=np.mean([head[1] for head in roles['carrier']['attention']])
            mate_at=np.mean([head[1] for head in roles['mate']['attention']])
            conclusions.append('%s 최신 동일 관측 bank의 AT edge 가중치: 운반자 query %.1f%%, 동료 query %.1f%%. 낮은 AT 가중치는 goal 정보 미사용을 뜻하지 않는다. 다른 edge의 node 특징과 Query도 장면 전체를 인코딩한 결과다.'%
                (name,100*carrier_at,100*mate_at))
    for result in summary['ablations']:
        values=result['families']
        conclusions.append('최신 %s 실제 rollout: CLIMB/SIT/STACK 동시 ever 성공 %.1f%% / %.1f%% / %.1f%%. CA 외 actor encoder 경로는 유지된다.'%
            (result['mode'],*(100*values[f]['joint_ever'][0] for f in FAMILIES[:3])))
    if summary['ablations']:
        conclusions.append('해석: CA는 동료 edge 정보를 사용하지만, 전체 CA를 끈 효과는 과제별로 다르다. 학습 성공률 상승을 CA 깊이의 충분성이나 명시적 순서 추론의 증명으로 해석하지 않는다.')
    starts_equal=all(r['initial_observation_max_abs_difference']<1e-5 and r['graph_equal'] and r['asset_sizes_equal'] for r in starts)
    images=['rollout_trends.png','fixed_state_trends.png','ca_interventions.png','routing_by_role.png','event_aligned.png','success_failure_examples.png']+['time_maps_'+family+'.png' for family in FAMILIES[:3]]
    image_html=''.join('<figure><a href="'+name+'"><img src="'+name+'"></a><figcaption>'+name+'</figcaption></figure>' for name in images if (output/name).is_file())
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Stage 2 attention trends</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;color:#18222f}h1,h2{line-height:1.3}figure{margin:30px 0}img{width:100%;border:1px solid #ddd}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#edf2f7}code{background:#edf2f7;padding:2px 5px}li{margin:8px 0}.note{padding:18px;background:#f1f5f9;border-left:4px solid #2563eb}</style>
<h1>Stage 2 cross-attention 학습 추세</h1>'''
    page+='<p>지정한 latest checkpoint를 고정한 epoch <b>%d</b> 기준. 결과는 관측·행동·실제 rollout을 분리해 해석한다.</p>'%latest['epoch']
    page+='<ul>'+''.join('<li>'+html.escape(line)+'</li>' for line in conclusions)+'</ul>'
    page+='<div class="note">기본 상자 범위·현재 RSI, 사람 2명·상자 4개. 협력 CLIMB/SIT/STACK 각각 양쪽 역할과 독립 대조군. 조건당 %d환경 × %d seed, horizon %.1f초. Attention은 %d step마다 기록.<br>Checkpoint 간 초기 관측·graph·asset 크기 일치: <b>%s</b>.<br>고정 관측 bank는 최신 정책의 방문 상태를 사용한다. 성공·실패를 모두 포함하지만 방문 상태의 범위는 최신 정책에 제한된다.</div>'%(env_count,latest['repeats'],latest['steps']*latest['dt'],latest['stride'],'PASS' if starts_equal else '차이 있음: summary.json 참조')
    page+='<p>현재 graph는 temporal DAG를 강제하지 않는다. 동료가 AT 전에 성공한 것은 곧바로 순서 위반을 뜻하지 않는다. Attention만으로 인과적 협력을 결론내리지 않으며, 개별 edge 마스킹은 encoder를 고정한 CA 경로의 순간 영향이다. 최신 checkpoint의 CA 제거·uniform routing·동료 edge 제거에는 실제 물리 rollout을 추가했다. 이 개입에서도 encoder를 통한 다른 사람/물체의 정보 전달은 유지된다.</p>'
    page+=image_html+''.join(tables)
    page+='<h2>재현·원자료</h2><p><a href="summary.json">집계 JSON</a> · <a href="replay_manifest.json">Checkpoint 해시와 추출 오차</a> · <a href="observation_bank.npz">고정 관측 bank</a>. 각 epoch 폴더에 results.json과 traces.npz가 있으며 replay 파일은 같은 bank의 attention/action 진단이다.</p>'
    page+='<p>95% CI: 성공률 Wilson, attention/action은 에피소드 평균을 단위로 bootstrap. 같은 seed를 사용한 시행들이 완전히 독립인 표본이라고 보장하지 않으며, 제한된 평가에서의 변동 범위로 해석한다.</p></html>'
    (output/'report.html').write_text(page)
    print(json.dumps(dict(latest_epoch=latest['epoch'],rollout=summary['rollout'],
                         starts_equal=starts_equal,report=str(output/'report.html')),indent=2))
