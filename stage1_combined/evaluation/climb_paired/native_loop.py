"""Extract baseline's three-repeat control flow; instrument common metrics/I/O."""
import ast
import textwrap


def build_loop(source,tracker_factory):
    text=source.read_text()
    tree=ast.parse(text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run_eval')
    code=textwrap.dedent(''.join(text.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno]))
    changes={
        '    eval_res = {}':'    eval_res = {}\n    tracker = tracker_factory(self)',
        "        batch_size = self.get_batch_size(obs_dict['obs'], batch_size)":"        batch_size = self.get_batch_size(obs_dict['obs'], batch_size)\n        tracker.begin(i)",
        '            action = self.get_action(obs_dict, is_determenistic)':'            tracker.after_reset(normal_done_indices)\n            action = self.get_action(obs_dict, is_determenistic)',
        '            obs_dict, r, done, info =  self.env_step(self.env, action)':'            obs_dict, r, done, info =  self.env_step(self.env, action)\n            info = tracker.step(done, info)',
        '        success_rate = games_success / games_played':'        tracker.finish(i)\n        success_rate = games_success / games_played',
    }
    for old,new in changes.items():
        assert code.count(old)==1,old
        code=code.replace(old,new)
    # Artifact location and format are adapter-owned. The baseline loop is read-only.
    index=code.index('    # save metrics')
    code=code[:index]+'    tracker.complete(eval_res)\n    return\n'
    import torch
    namespace=dict(torch=torch,tracker_factory=tracker_factory)
    exec(compile(code,str(source)+'[common40s-repeat-trace]','exec'),namespace)
    return torch.no_grad()(namespace['run_eval'])
