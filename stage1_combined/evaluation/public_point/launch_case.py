from pathlib import Path
import sys,importlib.util,json,os
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tokenhsi'),str(ROOT)]
import isaacgym
from evaluation.finetuned.backend import EvaluationTask,EvaluationPlayer
from evaluation.public_point.policy import PublicPolicy
import utils.parse_task as parser
if os.environ.get('PUBLIC_SCENE30')=='1':
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from learning.multi_agent.ma_players import MAPlayerContinuous
    class Scene30Task(EvaluationTask):
        def get_obs_size(self):return HumanoidMACarry.get_obs_size(self)
        def get_scene_entity_sizes(self):return HumanoidMACarry.get_scene_entity_sizes(self)
        def _compute_clean_scene_obs(self,env_ids=None):return HumanoidMACarry._compute_clean_scene_obs(self,env_ids)
    class Scene30Player(MAPlayerContinuous):
        run_eval=EvaluationPlayer.run_eval
    EvaluationTask=Scene30Task
    EvaluationPlayer=Scene30Player
class PublicPlayer(EvaluationPlayer):
    def run_eval(self):
        # The 35D wrapper only constructs the shared scene. All actions come from 40k.
        import torch
        # Keep scene initialization RNG unchanged by the additional public network.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.public=PublicPolicy(self.env.task,self,self.env.task.cfg['env']['chairFT']['public_checkpoint'])
        result=super().run_eval()
        out=Path(self.env.task.cfg['args'].output_path);p=out/'results.json';d=json.loads(p.read_text());d['policy']=self.public.metadata;p.write_text(json.dumps(d,indent=2));return result
    def get_action(self,obs_dict,is_determenistic=False):return self.public.action()
parser.EvaluationTask=EvaluationTask
spec=importlib.util.spec_from_file_location('public_eval_run',ROOT/'tokenhsi/run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run);old=run.build_alg_runner
def build(observer):
    runner=old(observer);runner.player_factory.register_builder('climb30_ft' if os.environ.get('PUBLIC_SCENE30')=='1' else 'chair_ft',lambda **kw:PublicPlayer(**kw));return runner
run.build_alg_runner=build;run.main()
