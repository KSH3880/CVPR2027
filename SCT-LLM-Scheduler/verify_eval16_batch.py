"""CPU rescore and seed summary using the original evaluator's own functions."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'teacher_runtime/tokenhsi'),str(ROOT/'teacher_runtime/eval')]
import torch
from evaluate import summarize
from rescore import rescore
import stage2_xy100
stage2_xy100.install();torch.set_num_threads(1)

def verify(base):
 status=json.loads((base/'status.json').read_text());gpu=Path(status['gpu_output']);checks=[]
 for seed in (1,6,7):
  p=gpu/'eval'/('general_seed'+str(seed));data=json.loads((p/'result.json').read_text());proof=data['stage1_policy']
  excluded=data['llm_scheduler'].get('excluded',[])
  assert (data['num_envs'],data['num_agents'],data['num_objects'],data['horizon'],data['simulated_steps'])==(512-len(excluded),16,32,900,900)
  assert proof['native_actor_exact'] and proof['normalization_exact'] and proof['weights_unchanged'] and proof['coordination_calls']==0
  physics=json.loads((p/'physics_only_verification.json').read_text());assert physics['checked_frames']==901
  assert all(json.loads((p/'llm_initial_match.json').read_text()).values())
  result=rescore(p)
  assert all(result[k] for k in ('finite','cpu_gpu_exact','json_counts_exact','errors_match_recorded'))
  checks.append(dict(seed=seed,passed=True,initial_step=data['llm_scheduler']['first_step'],stage1_coordination_calls=0))
 summary=summarize(gpu/'eval');(base/'summary.json').write_text(json.dumps(summary,indent=2));(base/'verification.json').write_text(json.dumps(checks,indent=2))
 usage=dict(calls=0,prompt_tokens=0,completion_tokens=0,total_tokens=0)
 for p in (base/'api').glob('general_seed*/env_*/inference.json'):
  row=json.loads(p.read_text());usage['calls']+=row['calls']
  for key in ('prompt_tokens','completion_tokens','total_tokens'):usage[key]+=row['usage'].get(key,0)
 (base/'api_usage.json').write_text(json.dumps(usage,indent=2))
 counts=[json.loads((gpu/'eval'/('general_seed'+str(s))/'result.json').read_text())['num_envs'] for s in (1,6,7)]
 report='# Stage1 SCT + LLM initial release · 16명 평가\n\nseed 1/6/7 평가 환경 수 '+str(counts)+', 합계 '+str(sum(counts))+'환경, 900step, XY100%, Stage2 coordination 호출 0. 원본 스케줄 실행; 검증 오류는 audit 참조.\n\n'+json.dumps(summary.get('interaction_group_scores',summary['scores']),indent=2)+'\n'
 (base/'report.md').write_text(report)
if __name__=='__main__':verify(Path(sys.argv[1]).resolve())
