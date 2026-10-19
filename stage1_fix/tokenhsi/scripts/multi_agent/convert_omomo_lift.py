"""hhtools OMOMO -> native carry retarget, full/lift clips and object tracks.

Matches dataset_carry/generate_motion.py mapping and project_joints_simple.
Packed layout: 24*3 positions + 24*3 velocities + 22*6 global rotations.
"""
import hashlib
import json
import pickle
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'tokenhsi')]
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml
from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from lpanlib.poselib.core.rotation3d import quat_mul_norm, quat_from_angle_axis
from tokenhsi.data.data_utils import project_joints_simple

DATA = ROOT/'tokenhsi/data'
OUT = DATA/'dataset_omomo_lift'
XML = DATA/'assets/mjcf/phys_humanoid_v3.xml'
MAPPING = [0,6,12,17,19,21,16,18,20,2,5,8,1,4,7]
REFERENCES = [
 'https://github.com/jaggerShen/human-humanoid-tools/blob/main/hhtools/io/datasets/omomo.py',
 'https://github.com/lijiaman/omomo_release/blob/main/manip/data/hand_foot_dataset.py']

class NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy._core.numeric','numpy.core.numeric') and name=='_frombuffer':
            return np.core.numeric._frombuffer
        if module=='numpy' and name=='dtype': return np.dtype
        raise pickle.UnpicklingError((module,name))

def rotation_6d_to_matrix(v):
    # PyTorch3D row convention used by OMOMO.
    a,b=v[...,:3],v[...,3:]
    norm=np.linalg.norm(a,axis=-1,keepdims=True)
    if np.any(norm<1e-7): raise ValueError('Degenerate rotation')
    x=a/norm
    y=b-(x*b).sum(-1,keepdims=True)*x
    norm=np.linalg.norm(y,axis=-1,keepdims=True)
    if np.any(norm<1e-7): raise ValueError('Degenerate rotation')
    y=y/norm
    return np.stack((x,y,np.cross(x,y)),axis=-2)

def make_motion(tree,q,root):
    return SkeletonMotion.from_skeleton_state(SkeletonState.from_rotation_and_root_translation(
        tree,torch.as_tensor(q,dtype=torch.float32),torch.as_tensor(root,dtype=torch.float32),is_local=True),fps=30)

def continuous(q):
    q=np.array(q,copy=True)
    for i in range(1,len(q)):
        flip=(q[i]*q[i-1]).sum(-1)<0
        q[i,flip]*=-1
    return q

def foot_floor(motion):
    floors=[]
    for side in ('right','left'):
        mesh=DATA/'assets/smpl_foot_meshes'/f'{side}_foot.obj'
        vertices=np.array([[float(x) for x in line.split()[1:4]]
                           for line in mesh.read_text().splitlines() if line.startswith('v ')])
        j=motion.skeleton_tree.index(side+'_foot')
        r=Rotation.from_quat(motion.global_rotation[:,j].numpy()).as_matrix()
        floors.append((np.einsum('fi,vi->fv',r[:,2],vertices)
                      +motion.global_translation[:,j,2].numpy()[:,None]).min(-1))
    return np.minimum(*floors)

def convert(row,tree):
    name=row['sequence']
    source_dir=ROOT/'hhtools_omomo/OMOMO_hhtools_all_full'/name
    source=source_dir/(name+'.pkl')
    with source.open('rb') as f: raw=NumpyOnly(f).load()
    features=np.asarray(raw['motion'])
    assert features.ndim==2 and features.shape[1]==276 and np.isfinite(features).all()
    n=len(features)
    positions=features[:,:72].reshape(n,24,3)
    rotations=rotation_6d_to_matrix(features[:,144:].reshape(n,22,6))
    q=Rotation.from_matrix(rotations[:,MAPPING].reshape(-1,3,3)).as_quat().reshape(n,15,4)
    tpose=SkeletonState.zero_pose(tree)
    local=tpose.local_rotation.clone()
    for side,angle in [('left',90.),('right',-90.)]:
        local[tree.index(side+'_upper_arm')]=quat_from_angle_axis(
            torch.tensor([angle]),torch.tensor([1.,0.,0.]),degree=True)[0]
    tpose=SkeletonState.from_rotation_and_root_translation(tree,local,tpose.root_translation,is_local=True)
    aligned=quat_mul_norm(torch.tensor([-.5,-.5,-.5,.5]),tpose.global_rotation)
    target=quat_mul_norm(torch.tensor(q,dtype=torch.float32),aligned)
    # Global pelvis is already absolute. trans2joint is for SMPL transl, not this root.
    state=SkeletonState.from_rotation_and_root_translation(
        tree,target,torch.tensor(positions[:,0],dtype=torch.float32),is_local=False).local_repr()
    native=SkeletonMotion.from_skeleton_state(state,fps=30)
    motion=project_joints_simple(native)
    q=continuous(motion.local_rotation.numpy())
    root=motion.root_translation.numpy().copy()
    shift=.005-float(foot_floor(motion).min())
    root[:,2]+=shift
    motion=make_motion(tree,q,root)
    mesh_source=next(source_dir.glob('*.obj'))
    vertices=np.array([[float(x) for x in line.split()[1:4]]
                       for line in mesh_source.read_text().splitlines() if line.startswith('v ')])
    centre=(vertices.min(0)+vertices.max(0))/2
    scales=np.asarray(raw['obj_scale']).reshape(n)
    rot=np.asarray(raw['obj_rot_mat'])
    obj=np.asarray(raw['obj_trans'])+scales[:,None]*np.einsum('tij,j->ti',rot,centre)
    obj[:,2]+=shift
    obj_q=continuous(Rotation.from_matrix(rot).as_quat()[:,None])[:,0]
    box=np.concatenate((obj,obj_q),axis=-1).astype(np.float32)
    folder=OUT/'motions'/name
    folder.mkdir(parents=True,exist_ok=True)
    (folder/mesh_source.name).write_bytes(mesh_source.read_bytes())
    lines=[]
    for line in mesh_source.read_text().splitlines():
        if line.startswith('v '):
            v=np.array([float(x) for x in line.split()[1:4]])-centre
            line='v '+' '.join(f'{x:.9g}' for x in v)
        lines.append(line)
    (folder/'object_centered.obj').write_text('\n'.join(lines)+'\n')
    reports=[]
    for variant,a,b in [('full',0,n-1),('lift',row['lift_frame_start_user'],row['lift_frame_end_user'])]:
        dest=folder/variant/'phys_humanoid_v3'
        dest.mkdir(parents=True,exist_ok=True)
        crop=make_motion(tree,q[a:b+1],root[a:b+1])
        crop.to_file(str(dest/'ref_motion.npy'))
        np.save(dest/'box_motion.npy',box[a:b+1])
        np.save(dest/'object_scale.npy',scales[a:b+1].astype(np.float32))
        report=dict(sequence=name,split=row['split'],variant=variant,
            source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            frames=b-a+1,fps=30,source_frames_inclusive=[a,b],frame_index='zero-based inclusive',
            target_skeleton=tree.node_names,joint_mapping=MAPPING,
            algorithm='native carry global-rotation mapping + project_joints_simple',
            wrist_rotation='fixed identity; no wrist DOF in target',
            coordinates='Z-up metres; source heading/XY retained; constant shared vertical shift',
            shared_vertical_shift_m=shift,foot_mesh_floor_min_m=float(foot_floor(crop).min()),
            object_pose='centred local bbox; xyz+xyzw; per-frame scale separate',
            object_mesh='../../object_centered.obj',object_mesh_raw_bbox_center=centre.tolist(),
            object_local_extents_raw=np.ptp(vertices,axis=0).tolist(),
            object_center_rise_m=float(box[b,2]-box[a,2]),
            hand_fit_mean_m=np.linalg.norm(crop.global_translation[:,[5,8]].numpy()
                -positions[a:b+1,[21,20]]-np.array([0,0,shift]),axis=-1).mean(0).tolist(),
            converter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),references=REFERENCES,
            limitations=['No contact IK, underhand synthesis, rotation lock or physics validation.',
                         'Native mapping does not enforce all simulator joint limits.',
                         'Apply object_scale.npy to object_centered.obj for faithful geometry.'])
        assert torch.isfinite(crop.tensor).all() and np.isfinite(box[a:b+1]).all()
        (dest/'conversion_report.json').write_text(json.dumps(report,indent=2)+'\n')
        reports.append(report)
    print(name,'saved full/lift:',n,row['lift_frame_end_user']-row['lift_frame_start_user']+1,flush=True)
    return reports

def main():
    torch.set_num_threads(1)
    manifest=json.loads((ROOT/'hhtools_omomo/lift_manifest.json').read_text())
    tree=SkeletonTree.from_mjcf(str(XML))
    reports=[]
    for row in manifest['candidates']: reports.extend(convert(row,tree))
    for variant in ('full','lift'):
        for split in ('train','test','all'):
            entries=[]
            for r in reports:
                if r['variant']!=variant or (split!='all' and r['split']!=split): continue
                stem=Path('dataset_omomo_lift/motions')/r['sequence']/variant/'phys_humanoid_v3'
                entries.append(dict(file=str(stem/'ref_motion.npy'),obj_file=str(stem/'box_motion.npy'),weight=1.0))
            (DATA/f'dataset_omomo_lift_{variant}_{split}.yaml').write_text(
                '# Converted candidates; review before training. all includes original test split.\n'
                +yaml.safe_dump({'motions':{'lift':entries}},sort_keys=False))
    (OUT/'conversion_summary.json').write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__': main()
