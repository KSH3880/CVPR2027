"""Register an evaluation-only task/player; leave running training code untouched."""
from pathlib import Path
import sys,importlib.util,os,json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tokenhsi'),str(ROOT)]
import isaacgym
from evaluation.finetuned.backend import EvaluationTask,EvaluationPlayer
import utils.parse_task as parser
if os.environ.get('OURS_ORIGINAL30D')=='1':
    import torch
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from learning.multi_agent.ma_players import MAPlayerContinuous
    class Original30Task(EvaluationTask):
        def get_obs_size(self):return HumanoidMACarry.get_obs_size(self)
        def get_scene_entity_sizes(self):return HumanoidMACarry.get_scene_entity_sizes(self)
        def _compute_clean_scene_obs(self,env_ids=None):return HumanoidMACarry._compute_clean_scene_obs(self,env_ids)
    class Original30Player(MAPlayerContinuous):
        run_eval=EvaluationPlayer.run_eval
        def restore(self,path):
            result=super().restore(path);source=torch.load(path,map_location='cpu',weights_only=False)
            assert self.env.task.get_scene_entity_sizes()[1]==30
            assert set(self.model.state_dict())==set(source['model'])
            for k,v in self.model.state_dict().items():assert torch.equal(v.detach().cpu(),source['model'][k]),k
            for k,v in self.running_mean_std.state_dict().items():assert torch.equal(v.detach().cpu(),source['running_mean_std'][k]),k
            out=Path(self.env.task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
            (out/'original30_policy_verified.json').write_text(json.dumps(dict(epoch=source['epoch'],object_input=30,model_state_exact=True,scene_rms_exact=True,no_added_model_keys=True)))
            print('ORIGINAL30_POLICY_EXACT',source['epoch'],flush=True);return result
    EvaluationTask=Original30Task
    EvaluationPlayer=Original30Player
parser.EvaluationTask=EvaluationTask
spec=importlib.util.spec_from_file_location('evaluation_run',ROOT/'tokenhsi/run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
original=run.build_alg_runner
def build(observer):
    runner=original(observer);runner.player_factory.register_builder('climb30_ft' if os.environ.get('OURS_ORIGINAL30D')=='1' else 'chair_ft',lambda **kwargs:EvaluationPlayer(**kwargs));return runner
run.build_alg_runner=build
run.main()
