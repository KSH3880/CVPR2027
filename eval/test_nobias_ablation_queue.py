from pathlib import Path

from run_nobias_ablation import jobs, command, reservation, RESERVATIONS


def test_twelve_jobs_reuse_formal_mixed_references():
    work = jobs()
    assert len(work) == len({job['key'] for job in work}) == 12
    assert {(job['people'], job['seed']) for job in work} == {
        (people, seed) for people in (2, 4, 8, 16) for seed in (1, 6, 7)}
    for job in work:
        assert job['table'] == 'table1' and job['mode'] == 'mixed'
        assert job['radius'] == {8: 3., 16: 7.5}.get(job['people'])
        assert all((Path(job['reference'])/name).is_file() for name in
                   ('initial.pt', 'observed_scene.pt', 'reference_info.json'))
        args = command(job)
        for flag, value in (('--num_envs', '512'), ('--num_objects', str(2*job['people'])),
                            ('--enable_size_mode', 'per_action_intersection'), ('--coupled_assignment', 'ordered')):
            assert args[args.index(flag)+1] == value
        assert ('--group_spawn_over4' in args) == (job['people'] > 4)
        assert 'stage2_distill_900_v4_nobias.yaml' in args[args.index('--cfg_train')+1]


def test_memory_reserves_startup_and_retains_runtime_headroom():
    job = dict(people=16)
    assert reservation(job, 0, False) == RESERVATIONS[16]
    assert reservation(job, 6000, False) == RESERVATIONS[16]-6000
    assert reservation(job, RESERVATIONS[16]+500, False) == 0
    assert reservation(job, 7000, True) == 512
    assert 70000+reservation(job, 6000, False)+RESERVATIONS[2]+1024 > 85000
