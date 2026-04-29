
import json, os, pathlib, signal, subprocess, sys, time
ROOT = pathlib.Path(os.environ.get('RMT_OPTUNA_RUN', './optuna_run'))
Q = 'queue_b'
QUEUE_FILE = 'model_queue_b.json'
BANNED = ['resnetrs420.tf_in1k', 'resnext101_32x32d.fb_wsl_ig1b_ft_in1k', 'seresnextaa201d_32x8d.sw_in12k_ft_in1k_384', 'convnext_large.fb_in22k_ft_in1k']
os.chdir(ROOT)
queue_dir = ROOT / 'model_queue_runs' / Q

def load_json(name):
    p = queue_dir / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def cmdline(pid):
    try:
        return (pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\x00', b' ').decode('utf-8', 'ignore')
    except Exception:
        return ''

def matching_pids():
    me = os.getpid(); parent = os.getppid(); out = []
    for entry in pathlib.Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in (me, parent):
            continue
        cmd = cmdline(pid)
        checks = [
            ('model_queue_runner.py' in cmd and f'--queue-name {Q}' in cmd),
            ('remote_runner_wrapper.py' in cmd and f'model_queue_runs/{Q}/runner_status.json' in cmd),
            ('hybrid_mag20_then_v8_model_queue.py' in cmd and f'/{Q}' in cmd),
            ('run_finetune_magnitude_model_exec_queue.py' in cmd and f'/{Q}' in cmd),
            ('run_removed_matrix_audit' in cmd and f'/{Q}' in cmd),
            ('build_model_rmt_cache.py' in cmd and f'/{Q}' in cmd),
        ]
        if any(checks):
            out.append(pid)
    return out

def kill_queue():
    for pid in matching_pids():
        try: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError: pass
    time.sleep(5)
    for pid in matching_pids():
        try: os.kill(pid, signal.SIGKILL)
        except ProcessLookupError: pass

def start_queue(log_name='autorestart.log'):
    log = open(queue_dir / log_name, 'ab', buffering=0)
    subprocess.Popen([sys.executable, '-u', 'start_model_queue.py', Q, QUEUE_FILE], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

while True:
    state = load_json('queue_state.json')
    runner = load_json('runner_status.json')
    model = state.get('model_name', '')
    state_name = state.get('state', 'missing')
    runner_state = runner.get('state', 'missing')
    if state_name == 'complete':
        with open(queue_dir / 'queue_guard.log', 'a') as f: f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] complete; guard exiting\n")
        raise SystemExit(0)
    if state_name == 'failed' or runner_state != 'running' or model in BANNED:
        with open(queue_dir / 'queue_guard.log', 'a') as f: f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] restarting state={state_name} runner={runner_state} model={model}\n")
        kill_queue(); start_queue()
        with open(queue_dir / 'queue_guard.log', 'a') as f: f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] restarted\n")
        raise SystemExit(0)
    time.sleep(60)
