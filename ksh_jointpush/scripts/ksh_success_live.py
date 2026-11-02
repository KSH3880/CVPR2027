"""Keep the current native window until a requested replacement is ready."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ksh_jointpush.ksh_live_control import KINDS, read_json, select_case, write_json


class Windows:
    def __init__(self):
        self.x = C.CDLL('libX11.so.6')
        x = self.x
        # Windows can disappear between reading the client list and a property
        # query; Xlib's default error handler would terminate this controller.
        callback = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_void_p)
        self.error_handler = callback(lambda display, event: 0)
        x.XSetErrorHandler.argtypes = [callback]
        x.XSetErrorHandler(self.error_handler)
        x.XOpenDisplay.restype = C.c_void_p
        self.display = x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError('Native display is unavailable')
        x.XDefaultRootWindow.argtypes = [C.c_void_p]
        x.XDefaultRootWindow.restype = C.c_ulong
        self.root = x.XDefaultRootWindow(self.display)
        x.XInternAtom.argtypes = [C.c_void_p, C.c_char_p, C.c_int]
        x.XInternAtom.restype = C.c_ulong
        x.XGetWindowProperty.argtypes = [C.c_void_p, C.c_ulong, C.c_ulong, C.c_long,
            C.c_long, C.c_int, C.c_ulong, C.POINTER(C.c_ulong), C.POINTER(C.c_int),
            C.POINTER(C.c_ulong), C.POINTER(C.c_ulong), C.POINTER(C.c_void_p)]
        x.XFree.argtypes = [C.c_void_p]
        x.XStoreName.argtypes = [C.c_void_p, C.c_ulong, C.c_char_p]
        x.XChangeProperty.argtypes = [C.c_void_p, C.c_ulong, C.c_ulong, C.c_ulong,
                                      C.c_int, C.c_int, C.c_char_p, C.c_int]
        x.XIconifyWindow.argtypes = [C.c_void_p, C.c_ulong, C.c_int]
        x.XMapRaised.argtypes = [C.c_void_p, C.c_ulong]
        x.XFlush.argtypes = [C.c_void_p]

    def atom(self, name):
        return self.x.XInternAtom(self.display, name.encode(), 0)

    def property(self, window, name):
        actual, fmt, count, rest, data = C.c_ulong(), C.c_int(), C.c_ulong(), C.c_ulong(), C.c_void_p()
        rc = self.x.XGetWindowProperty(self.display, window, self.atom(name), 0, 4096, 0, 0,
            C.byref(actual), C.byref(fmt), C.byref(count), C.byref(rest), C.byref(data))
        result = []
        if rc == 0 and data and fmt.value == 32:
            result = list(C.cast(data, C.POINTER(C.c_ulong))[:count.value])
        if data:
            self.x.XFree(data)
        return result

    def find(self, pid):
        for window in self.property(self.root, '_NET_CLIENT_LIST'):
            if self.property(window, '_NET_WM_PID') == [pid]:
                return window
        return None

    def title(self, window, title):
        if not window:
            return
        title = title.encode('utf-8')
        self.x.XStoreName(self.display, window, title)
        self.x.XChangeProperty(self.display, window, self.atom('_NET_WM_NAME'),
            self.atom('UTF8_STRING'), 8, 0, title, len(title))
        self.x.XFlush(self.display)

    def hide(self, window):
        if window:
            self.x.XIconifyWindow(self.display, window, 0)
            self.x.XFlush(self.display)

    def show(self, window):
        if window:
            self.x.XMapRaised(self.display, window)
            self.x.XFlush(self.display)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--category', choices=KINDS, help='Group composition to open first')
    args = parser.parse_args()
    run = args.run.resolve()
    session = run / ('viewer_switch_' + time.strftime('%Y%m%d_%H%M%S'))
    session.mkdir()
    windows = Windows()
    processes = []
    serial = 0

    def start(case):
        nonlocal serial
        serial += 1
        control = session / str(serial)
        control.mkdir()
        write_json(control/'state.json', dict(mode='run', acknowledged=0))
        log = (control/'viewer.log').open('w')
        # The PID is the native Python process, because both wrappers use exec.
        env = dict(os.environ, KSH_LIVE_CASE=case['file'], KSH_LIVE_HEADLESS='0',
                   KSH_LIVE_CONTROL=str(control))
        process = subprocess.Popen(['bash', str(run/'live_managed_one.sh')], env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        log.close()
        worker = dict(process=process, control=control, case=case, window=None, hidden=False)
        processes.append(worker)
        print('KSH_SWITCH_LOADING', case['category'], case['file'], process.pid, flush=True)
        return worker

    def ready(worker, hide=False):
        data = read_json(worker['control']/'ready.json')
        native_pid = (data or read_json(worker['control']/'boot.json', {})).get('pid')
        # launcher PID can differ; the worker publishes its native PID as soon as run() starts.
        if native_pid and worker['window'] is None:
            worker['window'] = windows.find(native_pid)
        if hide and worker['window'] and not worker['hidden']:
            windows.hide(worker['window'])
            worker['hidden'] = True
        return bool(data and worker['window'])

    def stop(worker):
        if worker['process'].poll() is None:
            write_json(worker['control']/'state.json', dict(mode='quit'))

    def publish(worker):
        active = dict(pid=read_json(worker['control']/'ready.json')['pid'],
                      case=worker['case'], control=str(worker['control']), session=str(session))
        write_json(run/'managed_active.json', active)
        write_json(Path(__file__).resolve().parents[2]/'runs/jointpush/ksh_jointpush_active_view.json',
                   dict(active, kind='live_success_policy', state='running', epoch=5500,
                        manager_pid=os.getpid(), run=str(run), floor_centered=True,
                        training_state='stopped', live_result_may_differ=True,
                        switching_preserves_current_window=True))

    report = read_json(run/'cases/cases.json', {})
    initial_code = 41 + KINDS.index(args.category) if args.category else 0
    case = select_case(report.get('cases', []), '', initial_code)
    if case is None:
        raise RuntimeError('No collected cases available')
    active = start(case)
    pending = None
    seen_request = 0
    try:
        while True:
            for worker in processes:
                if worker is not active and worker is not pending:
                    worker['process'].poll()  # reap viewers that completed a handoff
            if active['process'].poll() is not None:
                # Esc/close ends this viewer controller; collections are separate processes.
                break
            if pending:
                if pending['process'].poll() is not None:
                    print('KSH_SWITCH_FAILED_KEEP_CURRENT', pending['case'], flush=True)
                    write_json(active['control']/'state.json', dict(mode='run', acknowledged=seen_request))
                    windows.title(active['window'], 'KSH 5500 | switch failed; current case kept')
                    pending = None
                elif ready(pending, hide=True):
                    old = active
                    active, pending = pending, None
                    windows.show(active['window'])
                    windows.title(active['window'], 'KSH 5500 | '+active['case']['category']+' | N next, 1/2/3 groups')
                    stop(old)
                    seen_request = 0
                    publish(active)
                    print('KSH_SWITCH_READY', active['case']['category'], active['case']['file'], flush=True)
            elif ready(active):
                request = read_json(active['control']/'request.json', {})
                sequence = request.get('sequence', 0)
                if sequence > seen_request:
                    report = read_json(run/'cases/cases.json')
                    if report is None:
                        continue
                    seen_request = sequence
                    target = select_case(report.get('cases', []), active['case']['file'], request['code'])
                    if target is None or target['file'] == active['case']['file']:
                        text = 'category pending; current case kept' if target is None else active['case']['category']
                        windows.title(active['window'], 'KSH 5500 | '+text+' | N next, 1/2/3 groups')
                        write_json(active['control']/'state.json', dict(mode='run', acknowledged=sequence))
                        print('KSH_SWITCH_KEEP_CURRENT', request['code'], flush=True)
                    else:
                        windows.title(active['window'], 'KSH 5500 | loading '+target['category']+'; current screen kept')
                        pending = start(target)
                elif active['window']:
                    # Record initial readiness as well as later handoffs.
                    if not active.get('published'):
                        publish(active)
                        windows.title(active['window'], 'KSH 5500 | '+active['case']['category']+' | N next, 1/2/3 groups')
                        active['published'] = True
            time.sleep(.05)
    finally:
        for worker in processes:
            stop(worker)
        # Startup can still be loading motion files and cannot poll the control yet.
        for worker in processes:
            if worker['process'].poll() is None and not (worker['control']/'ready.json').exists():
                worker['process'].send_signal(signal.SIGINT)

if __name__ == '__main__':
    main()
