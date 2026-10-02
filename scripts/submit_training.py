"""Submit to Ray using a temporary private runtime configuration.

The checkout and dependencies must exist at the same path on every worker.
API keys are passed in a mode-0600 file, never as command-line arguments.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]

def main():
    command = sys.argv[1:]
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        raise SystemExit('Usage: python scripts/submit_training.py -- COMMAND [ARGS...]')
    names = ['JUDGER_MODEL', 'JUDGER_API_KEY', 'JUDGER_BASE_URL', 'JUDGER_TIMEOUT',
             'JUDGER_MAX_RETRIES', 'JUDGER_TEMPERATURE', 'JUDGER_FORMAT_WEIGHT',
             'JUDGER_FAIL_OPEN', 'SERPER_API_KEY', 'SERPER_API_URL',
             'YOUTUBE_COOKIES_FILE', 'CUDA_DEVICE_MAX_CONNECTIONS',
             'http_proxy', 'https_proxy', 'no_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY']
    runtime = yaml.safe_load((ROOT / 'verl-video/verl/trainer/runtime_env.yaml').read_text())
    runtime['env_vars'].update({key: os.environ[key] for key in names if key in os.environ})
    runtime['env_vars'].update(
        VSS_VIDEO_AGENT_PATH=str(ROOT),
        PYTHONPATH=os.pathsep.join(map(str, [ROOT, ROOT / 'verl-video', ROOT / 'verl-video/video_search_sim'])),
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', prefix='video-runtime-') as config:
        yaml.safe_dump(runtime, config)
        config.flush()
        result = subprocess.run(['ray', 'job', 'submit', '--address', os.getenv('RAY_ADDRESS', 'http://127.0.0.1:8265'),
                                 '--runtime-env', config.name, '--', *command], cwd=ROOT / 'verl-video')
    raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
