"""Reject unsupported poses; ground stance feet without lifting the approach."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from rsi_native.geometry import collision_samples,NativeGeometry
from rsi_native.support import foot_bottom,project_ground,supported


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 m=json.loads((a.source/'manifest.json').read_text());shapes=collision_samples('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml',m['body_names']);report=[]
 for e in m['assets']:
  geo=NativeGeometry(e['path'],shapes);folder=a.output/e['id'];folder.mkdir()
  for skill in ['sit','climb']:
   d=dict(np.load(a.source/e['id']/(skill+'.npz')));before,_=foot_bottom(d['body_state'],shapes);source_bottom=before-d['transform'][:,6]
   grounded=(~d['phase'])&(np.abs(source_bottom)<=.06)
   r,b,delta=project_ground(d['root_state'][grounded],d['body_state'][grounded],shapes)
   d['root_state'][grounded]=r;d['body_state'][grounded]=b;d['transform'][grounded,6]+=delta;d['correction'][grounded,2]+=delta
   d['ground_projected']=grounded
   clearance,floor=geo.check(d['body_state'][...,:3],d['body_state'][...,3:7],e['bbox'][2]/2);support,gap=supported(d['body_state'],geo,e['bbox'][2]/2)
   # Re-evaluate all candidates after projection, including previously unsafe poses.
   valid=d['anchor_valid']&(d['dof_correction_max']<=.3)&(clearance>=-.015)&(floor>=-.02)&support
   d.update(clearance=clearance,floor_clearance=floor,static_pass=valid,support_pass=support,support_gap=gap)
   np.savez_compressed(folder/(skill+'.npz'),**d)
   report.append(dict(asset=e['id'],category=e['category'],skill=skill,ground_projected=int(grounded.sum()),static=int(valid.sum()),nonphase=int((valid&~d['phase']).sum()),phase=int((valid&d['phase']).sum()),unsupported_rejected=int((~support).sum())))
   print('SUPPORT_REPAIR',json.dumps(report[-1]),flush=True)
 m.update(support_mode='ground_project_stance_nonphase_v1',support_tolerance=.03,support_recipe='source-grounded feet in nonphase project native lowest foot to5mm, remove vertical support-point velocity; otherwise require actual floor or horizontal native mesh contact within3cm; reject unsupported flight',support_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(a.output/'manifest.json').write_text(json.dumps(m,indent=2));(a.output/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
