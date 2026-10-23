"""Opt-in actor attention capture, restored to canonical entity order by encoder."""
import json
from pathlib import Path
import numpy as np


class AttentionRecorder:
    def __init__(self, directory, interval=10, max_samples=32):
        if interval < 1 or max_samples < 1:
            raise ValueError('Attention interval and sample count must be positive')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.interval, self.max_samples = interval, max_samples
        self.step = self.samples = 0

    def should_capture(self):
        return self.samples < self.max_samples and self.step % self.interval == 0

    def save(self, attention, task):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        m, o = task.num_agents, task.num_objects
        objects = [f'object{i}' for i in range(o)]
        regions = getattr(task, '_agent_region_assignment', None)
        if regions is not None:
            if getattr(task, '_hidden_door_tokens', False):
                objects = [f'door{i}' for i in range(m)] + [f'region{i}' for i in range(m)]
            else:
                objects = [f'region{i}' for i in range(m)] + [f'box{i}' for i in range(o-m)]
        elif hasattr(task, '_door_body_count'):
            objects = [f'door{i}' for i in range(o//2)] + [f'box{i}' for i in range(o//2)]
        labels = [f'agent{i}' for i in range(m)] + objects + [f'goal{i}' for i in range(m)]
        weights = attention.float().cpu().numpy()  # layer, head, query, key; env0
        if weights.shape[-2:] != (len(labels), len(labels)):
            raise ValueError('Attention capture entity labels do not match token count')
        metadata = dict(step=self.step, env=0, labels=labels,
                        query_axis='rows', key_axis='columns', token_order='canonical',
                        shape=list(weights.shape), agent_region_assignment=None,
                        episode_step=int(task.progress_buf[0]), checkpoint_kind='actor')
        if regions is not None:
            metadata['agent_region_assignment'] = regions[0].detach().cpu().tolist()
        for name in ('_agent_box_assignment', '_ordinary_keep', '_logical_box_order', '_group_leaders'):
            value = getattr(task, name, None)
            if value is not None:
                metadata[name] = value[0].detach().cpu().tolist()
        prefix = self.directory/f'actor_attention_{self.step:06d}'
        np.savez_compressed(str(prefix)+'.npz', attention=weights, labels=np.asarray(labels))
        averages = weights.mean(1)[:, :m]  # average heads, preserve each layer/agent
        fig, axes = plt.subplots(len(weights), 1, figsize=(max(10,len(labels)*.65), len(weights)*2.5), squeeze=False)
        vmax = max(float(averages.max()), .01)
        for layer, ax in enumerate(axes[:, 0]):
            ax.imshow(averages[layer], vmin=0, vmax=vmax, cmap='magma', aspect='auto')
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticks(range(m)); ax.set_yticklabels(labels[:m])
            ax.set_title(f'Actor layer {layer}: query agent -> key token (head mean)')
        fig.tight_layout(); fig.savefig(str(prefix)+'.png', dpi=140); plt.close(fig)
        metadata['top3_per_layer_agent'] = [
            [[dict(token=labels[j], weight=float(row[j])) for j in row.argsort()[-3:][::-1]]
             for row in layer] for layer in averages]
        Path(str(prefix)+'.json').write_text(json.dumps(metadata, indent=2)+'\n')
        self.samples += 1
        print(f'[Attention] saved {prefix}.png ({self.samples}/{self.max_samples})', flush=True)
