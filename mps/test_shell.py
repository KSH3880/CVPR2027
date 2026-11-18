"""Shared-server MPS guards, using fake NVIDIA tools and CPU-only processes."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import unittest


HELPER = Path(__file__).with_name('shell.sh')
FAKE_TOOL = r'''#!/usr/bin/env python3
import json, os, pathlib, signal, subprocess, sys
if pathlib.Path(sys.argv[0]).name == 'nvidia-smi':
    gpu = sys.argv[sys.argv.index('-i') + 1]
    if gpu.startswith('GPU-'): gpu = str(int(gpu.rsplit('-', 1)[1]))
    if gpu not in ('4', '6'): sys.exit(1)
    print(gpu if '--query-gpu=index' in sys.argv else 'GPU-00000000-0000-0000-0000-' + gpu.zfill(12))
    sys.exit(0)
pipe = pathlib.Path(os.environ['CUDA_MPS_PIPE_DIRECTORY'])
pidfile = pipe / 'nvidia-cuda-mps-control.pid'
ready = pipe.parent / 'mock_server.ready'
command = ' '.join(sys.argv[1:]) or sys.stdin.read().strip()
with open(os.environ['MPS_FAKE_CALLS'], 'a') as f:
    f.write(json.dumps([command, str(pipe), os.environ.get('CUDA_VISIBLE_DEVICES')]) + '\n')
if command == '-d':
    if os.environ.get('MPS_FAKE_FAIL'): sys.exit(7)
    child = subprocess.Popen(['sleep', '120'], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pidfile.write_text(str(child.pid))
    with open(os.environ['MPS_FAKE_PIDS'], 'a') as f: f.write(str(child.pid) + '\n')
elif command.startswith('quit'):
    os.kill(int(pidfile.read_text()), signal.SIGTERM)
    pidfile.unlink()
    if ready.exists(): ready.unlink()
elif not pidfile.exists():
    sys.exit(1)
elif command.startswith('start_server'):
    assert command == 'start_server -uid ' + str(os.getuid())
    if os.environ.get('MPS_FAKE_SERVER_FAIL'): sys.exit(7)
    ready.touch()
elif command == 'get_server_list' and ready.exists():
    print(pidfile.read_text())
'''


class ShellSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.bin = self.base / 'bin'
        self.bin.mkdir()
        for name in ('nvidia-smi', 'nvidia-cuda-mps-control'):
            path = self.bin / name
            path.write_text(FAKE_TOOL)
            path.chmod(0o755)
        self.calls = self.base / 'calls'
        self.pids = self.base / 'pids'
        self.root = self.base / 'mps'
        self.env = dict(os.environ, PATH=str(self.bin) + ':' + os.environ['PATH'],
                        MPS_GPU_ROOT=str(self.root), MPS_FAKE_CALLS=str(self.calls),
                        MPS_FAKE_PIDS=str(self.pids),
                        CUDA_MPS_PIPE_DIRECTORY='/tmp/nvidia-mps', CUDA_VISIBLE_DEVICES='0,1,2')

    def tearDown(self):
        if self.pids.exists():
            for pid in self.pids.read_text().split():
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
        self.tmp.cleanup()

    def run_shell(self, command):
        return subprocess.run(['bash', '--noprofile', '--norc', '-c',
                               'source "' + str(HELPER) + '"\n' + command],
                              env=self.env, capture_output=True, text=True, timeout=15)

    def log(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_one_gpu_and_existing_daemon_reuse(self):
        result = self.run_shell('mps_start 6 && mps_start 6 && mps_status 6')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('MPS for GPU 6 is ready; no connected CUDA clients.', result.stdout)
        self.assertEqual(sum(row[0] == '-d' for row in self.log()), 1)
        self.assertEqual(sum(row[0].startswith('start_server') for row in self.log()), 1)
        self.assertTrue(all(row[1] == str(self.root / 'gpu6/pipe') for row in self.log()))
        self.assertTrue(all(row[2].endswith('000000000006') for row in self.log()))

    def test_gpu_mismatch_blocks_start_use_status_and_stop(self):
        self.assertEqual(self.run_shell('mps_start 6').returncode, 0)
        other = self.root / 'gpu4/pipe'
        other.mkdir(parents=True)
        (other / 'nvidia-cuda-mps-control.pid').write_text(
            (self.root / 'gpu6/pipe/nvidia-cuda-mps-control.pid').read_text())
        before = self.log()
        for cmd in ('mps_start 4', 'mps_use 4', 'mps_status 4', 'mps_stop 4', 'mps_stop 4 --force'):
            result = self.run_shell(cmd)
            self.assertNotEqual(result.returncode, 0, cmd)
        self.assertEqual(self.log(), before)

    def test_stop_only_selected_gpu(self):
        result = self.run_shell('mps_start 4 && mps_start 6 && mps_stop 6 && mps_status 4')
        self.assertEqual(result.returncode, 0, result.stderr)
        stops = [r for r in self.log() if r[0].startswith('quit')]
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0][1], str(self.root / 'gpu6/pipe'))
        self.assertTrue((self.root / 'gpu4/pipe/nvidia-cuda-mps-control.pid').exists())

    def test_invalid_gpu_never_calls_control(self):
        for args in ('', '-1', '0,6', '6 4', '999'):
            self.assertNotEqual(self.run_shell('mps_start ' + args).returncode, 0)
        self.assertEqual(self.log(), [])

    def test_symlink_and_unverified_socket_refused(self):
        self.root.symlink_to(self.bin, target_is_directory=True)
        self.assertNotEqual(self.run_shell('mps_start 6').returncode, 0)
        self.root.unlink()
        pipe = self.root / 'gpu6/pipe'
        pipe.mkdir(parents=True)
        (pipe / 'control').touch()
        self.assertNotEqual(self.run_shell('mps_start 6').returncode, 0)
        self.assertTrue(all(row[0] == 'get_server_list' for row in self.log()))

    def test_stale_pid_refused(self):
        pipe = self.root / 'gpu6/pipe'
        pipe.mkdir(parents=True)
        (pipe / 'nvidia-cuda-mps-control.pid').write_text('999999999')
        self.assertNotEqual(self.run_shell('mps_start 6').returncode, 0)
        self.assertNotEqual(self.run_shell('mps_stop 6').returncode, 0)
        self.assertEqual(self.log(), [])

    def make_socket(self, pipe, name='control'):
        pipe.mkdir(parents=True, exist_ok=True)
        sock = socket.socket(socket.AF_UNIX)
        sock.bind(str(pipe / name))
        return sock

    def test_stopped_daemon_sockets_archived_and_restart_works(self):
        self.assertEqual(self.run_shell('mps_start 6 && mps_stop 6').returncode, 0)
        pipe = self.root / 'gpu6/pipe'
        self.make_socket(pipe).close()
        self.make_socket(pipe, 'control_privileged').close()
        (pipe / 'control_lock').touch()
        os.mkfifo(pipe / 'log')
        result = self.run_shell('mps_start 6')
        self.assertEqual(result.returncode, 0, result.stderr)
        archives = list((self.root / 'gpu6').glob('pipe.stale.*'))
        self.assertEqual(len(archives), 1)
        self.assertTrue((archives[0] / 'control').is_socket())
        self.assertEqual(sum(row[0] == '-d' for row in self.log()), 2)

    def test_live_socket_without_pid_is_never_moved(self):
        pipe = self.root / 'gpu6/pipe'
        sock = self.make_socket(pipe)
        try:
            result = self.run_shell('mps_start 6')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Live MPS socket', result.stderr)
            self.assertTrue((pipe / 'control').is_socket())
            self.assertFalse(list((self.root / 'gpu6').glob('pipe.stale.*')))
            self.assertFalse(any(row[0] == '-d' for row in self.log()))
        finally:
            sock.close()

    def test_unknown_file_with_stale_socket_is_not_moved(self):
        pipe = self.root / 'gpu6/pipe'
        self.make_socket(pipe).close()
        (pipe / 'unknown').touch()
        self.assertNotEqual(self.run_shell('mps_start 6').returncode, 0)
        self.assertTrue((pipe / 'control').is_socket())
        self.assertFalse(any(row[0] == '-d' for row in self.log()))

    def test_foreign_owner_pid_refused(self):
        if Path('/proc/1').stat().st_uid == os.getuid():
            self.skipTest('PID 1 has the same owner')
        pipe = self.root / 'gpu6/pipe'
        pipe.mkdir(parents=True)
        (pipe / 'nvidia-cuda-mps-control.pid').write_text('1')
        for cmd in ('mps_start 6', 'mps_use 6', 'mps_stop 6 --force'):
            self.assertNotEqual(self.run_shell(cmd).returncode, 0)
        self.assertEqual(self.log(), [])

    def test_failed_start_does_not_select_gpu_in_parent(self):
        self.env['MPS_FAKE_FAIL'] = '1'
        result = self.run_shell('mps_start 6; rc=$?; echo "RESULT:$rc:$CUDA_VISIBLE_DEVICES"')
        self.assertIn('RESULT:7:0,1,2', result.stdout)

    def test_run_preserves_arguments_exit_status_and_parent_gpu(self):
        result = self.run_shell('mps_start 6 && mps_run 6 bash -c \'test "$1" = "two words" && exit 9\' _ "two words"')
        self.assertEqual(result.returncode, 9, result.stderr)

    def test_concurrent_start_launches_once(self):
        result = self.run_shell('mps_start 6 & a=$!; mps_start 6 & b=$!; wait "$a" && wait "$b"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(row[0] == '-d' for row in self.log()), 1)

    def run_auto(self, gpu='6', pipe=None):
        env = dict(self.env, TOKENHSI_ROOT=str(HELPER.parent.parent), TOKENHSI_GPU=gpu)
        env.pop('CUDA_MPS_PIPE_DIRECTORY', None)
        if pipe is not None:
            env['CUDA_MPS_PIPE_DIRECTORY'] = pipe
        script = HELPER.parent.parent / 'tokenhsi/scripts/multi_agent/mps_auto_env.sh'
        return subprocess.run(['bash', '-c', 'source "$1" || exit; printf "GPU=%s PIPE=%s\\n" "$TOKENHSI_GPU" "${CUDA_MPS_PIPE_DIRECTORY:-none}"',
                               '_', str(script)], env=env, capture_output=True, text=True, timeout=15)

    def test_fresh_terminal_auto_attaches_only_to_selected_gpu(self):
        self.assertEqual(self.run_shell('mps_start 4 && mps_start 6').returncode, 0)
        for gpu in ('4', '6'):
            result = self.run_auto(gpu)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('PIPE=' + str(self.root / ('gpu' + gpu) / 'pipe'), result.stdout)
            self.assertIn('GPU=GPU-00000000-0000-0000-0000-' + gpu.zfill(12), result.stdout)
        result = self.run_auto('GPU-00000000-0000-0000-0000-000000000006')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_auto_rejects_inherited_other_gpu_and_global_pipe(self):
        self.assertEqual(self.run_shell('mps_start 6').returncode, 0)
        for pipe in (str(self.root / 'gpu4/pipe'), '/tmp/nvidia-mps'):
            self.assertNotEqual(self.run_auto(pipe=pipe).returncode, 0)

    def test_auto_without_daemon_does_not_start_one(self):
        self.root.mkdir()
        result = self.run_auto()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('GPU=6 PIPE=none', result.stdout)
        self.assertEqual(self.log(), [])

    def test_auto_fails_closed_on_stale_pid(self):
        pipe = self.root / 'gpu6/pipe'
        pipe.mkdir(parents=True)
        (pipe / 'nvidia-cuda-mps-control.pid').write_text('999999999')
        self.assertNotEqual(self.run_auto().returncode, 0)
        self.assertEqual(self.log(), [])

    def test_gpu_server_start_failure_is_reported(self):
        self.env['MPS_FAKE_SERVER_FAIL'] = '1'
        result = self.run_shell('mps_start 6')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('Selected GPU', result.stdout)


if __name__ == '__main__':
    unittest.main()
