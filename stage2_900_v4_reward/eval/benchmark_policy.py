import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TOKEN = ROOT.parent/'tokenhsi_push'
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT/'eval'), str(TOKEN), str(TOKEN/'scripts')]
import isaacgym
import torch

from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from learning.multi_agent.stage1_push_network import Stage1PushBuilder
from learning.multi_agent.stage2_push_network import Stage2PushBuilder
from stage1_protocol_policy import stage1_scene
from tokenhsi_table_policy import TokenHSIPolicy, project_scene

CHECKPOINTS = dict(
    tokenhsi=TOKEN/'checkpoints/TokenHSIPushKD_00040500.pth',
    sct=ROOT/'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth',
    ours=ROOT/'output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth')


def observation_path(people):
    if people == 1:
        return ROOT/'output_etc/check/bias_ablation_20261110/reference/independent_1h/general_seed1/observed_scene.pt'
    if people <= 4:
        base = ROOT/'output_etc/check/v3_2000_tables12_seeds167_20261110/reference/table1'
    else:
        base = ROOT/'output_etc/check/table1_group_radius_8r3_16r7p5_env512_seeds167_baselines_20261112/reference'
    return base/f'mixed_{people}h/general_seed1/observed_scene.pt'


def scene_actor(model, people, scene):
    checkpoint = torch.load(CHECKPOINTS[model], map_location='cpu', weights_only=False)
    saved = checkpoint['relation_experiment_config']
    builder = Stage1PushBuilder() if model == 'sct' else Stage2PushBuilder()
    builder.load(deepcopy(saved['train']['params']['network']))
    native = stage1_scene(scene, people, 2*people) if model == 'sct' else scene
    module = torch.nn.Module()
    module.a2c_network = builder.build('amp', actions_num=32, input_shape=(native.shape[-1],),
        amp_input_shape=(1330,), value_size=1, num_agents=people, num_objects=2*people,
        observation_mode='clean_scene', scene_entity_sizes=[223, 30, 1], scene_kinematic_size=7,
        scene_arena_scale=16., relation_reward_mode=saved['env']['relationReward']['mode'],
        relation_graph_spec=deepcopy(saved['env']['relationGraph']), device='cuda:0')
    module.load_state_dict(checkpoint['model'], strict=True)
    module.to('cuda:0').eval().requires_grad_(False)
    rms = SceneRunningMeanStd([223, 30, 1], [people, 2*people, people],
        normalized_sizes=[223, 30, 0], kinematic_size=7,
        extra_passthrough_size=(5 if model == 'sct' else 6)*people).to('cuda:0')
    rms.load_state_dict(checkpoint['running_mean_std'], strict=True)
    rms.eval().requires_grad_(False)

    def action():
        return module.a2c_network.eval_actor(rms(native))[0].clamp(-1, 1)

    return action, module, checkpoint['epoch']


@torch.no_grad()
def measure(model, people, args):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    path = observation_path(people)
    scene = torch.load(path, map_location='cpu', weights_only=False)[:1].clone().to('cuda:0')
    if model == 'tokenhsi':
        policy = TokenHSIPolicy(CHECKPOINTS[model], 'cuda:0')
        raw = project_scene(scene, people, 2*people)
        module, epoch = policy.model, policy.saved.get('epoch')

        def action():
            return module.a2c_network.eval_actor(policy.normalize(raw), raw)[0].clamp(-1, 1)
    else:
        action, module, epoch = scene_actor(model, people, scene)
    expected = action()
    if expected.shape != (people, 32) or not torch.isfinite(expected).all():
        raise ValueError('Unexpected action shape or nonfinite actor output')
    for _ in range(args.warmup):
        action()
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()-baseline
    torch.cuda.reset_peak_memory_stats()
    gpu, wall = [], []
    for _ in range(args.rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        for _ in range(args.iterations):
            action()
        end.record()
        torch.cuda.synchronize()
        wall.append(1000*(time.perf_counter()-begin)/args.iterations)
        gpu.append(start.elapsed_time(end)/args.iterations)
    return dict(model=model, people=people, scenes=1, objects=2*people, epoch=epoch,
        checkpoint=str(CHECKPOINTS[model]), observation=str(path),
        parameters_loaded=sum(p.numel() for p in module.parameters()),
        wall_ms=statistics.mean(wall), wall_ms_std=statistics.stdev(wall), wall_ms_rounds=wall,
        cuda_ms=statistics.mean(gpu), cuda_ms_rounds=gpu,
        resident_mib=resident/2**20, peak_allocated_mib=(torch.cuda.max_memory_allocated()-baseline)/2**20,
        action_shape=list(expected.shape), action_finite=True, normalization_included=True,
        input_conversion_included=False, simulation_included=False)


def main(args):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    torch.cuda.set_device(0)
    args.output.resolve().relative_to(ROOT/'output_etc/check')
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in CHECKPOINTS.values()}
    priming = argparse.Namespace(warmup=10, rounds=2, iterations=2)
    for model in ('tokenhsi', 'sct', 'ours'):
        measure(model, 1, priming)
    rows = []
    for people in (1, 2, 4, 8, 16):
        for model in ('tokenhsi', 'sct', 'ours'):
            row = measure(model, people, args)
            rows.append(row)
            print(json.dumps(row), flush=True)
            (args.output/'measurements.json').write_text(json.dumps(rows, indent=2)+'\n')
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    if not unchanged:
        raise ValueError('Checkpoint changed during measurement')
    summary = dict(gpu=torch.cuda.get_device_name(), device_properties=str(torch.cuda.get_device_properties(0)),
        warmup=args.warmup, rounds=args.rounds, iterations_per_round=args.iterations,
        checkpoint_hashes=hashes, checkpoint_hashes_unchanged=unchanged, rows=rows,
        memory_definition='PyTorch allocated memory; full loaded network includes actor/critic/AMP; excludes CUDA context, simulator and other processes',
        global_cuda_cache_primed=True,
        timing_definition='RMS + actor + mean action clamp; all people batched; precomputed model-native inputs',
        limitation='Shared GPU with concurrent training; indicative throughput, not isolated latency or publication benchmark')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# 정책 추론 속도·메모리 간단 측정', '',
        '장면1개·M=2N·FP32·정규화+actor+평균 행동. 입력 변환·시뮬레이션 제외. 각 인원의 행동을 한 배치로 산출한다.', '',
        '| Method | Humans | Time (ms/scene) | Resident (MiB) | Peak allocated (MiB) |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['people']} | {row['wall_ms']:.3f}±{row['wall_ms_std']:.3f} | {row['resident_mib']:.2f} | {row['peak_allocated_mib']:.2f} |")
    lines += ['', '시간은3회 구간 평균±표본std이며 단일 호출의 p50/p95가 아니다. 공유 GPU에서 다른 학습이 실행 중이므로 참고값이다.',
        '메모리는 현재 프로세스의 PyTorch 할당량이다. 로드한 전체 네트워크(actor/critic/AMP)와 입력을 포함하며 CUDA context·시뮬레이션·다른 프로세스는 제외한다. Actor만 남긴 최소 배포 메모리가 아니다.',
        '1H는 Independent,2/4/8/16H는 동일한 Mixed reference의 첫 장면을 사용했다. 세 checkpoint는 모두 원본 가중치를 strict 로드했다.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=50)
    main(parser.parse_args())
