import logging
import os
import subprocess
import sys
from typing import Dict

from flytekit import Resources, Secret, current_context, task, workflow
from flytekit.core.disruption_config import DisruptionConfig, DisruptionReadinessStatus
from flytekitplugins.kfpytorch import PyTorch
from flytekitplugins.vscode import vscode

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

SECRET_GROUP = 'codefetcher-secret'
SECRET_NAME = 'github_ae_proxy_url'

# ── Image ─────────────────────────────────────────────────────────────────────
# Update this tag after running: cd ~/Project/lifomo && ./docker/docker-build.sh lamer
LAMER_IMAGE = 'container-image-registry.corp.linkedin.com/temp/lifomo/lifomo-lamer:202603260858'

# ── Shared task config ────────────────────────────────────────────────────────
BASE_TASK_CONFIG = {
    'enable_nfs': True,
    'proxy_as': 'coreaifomo',
    'secret_requests': [Secret(group=SECRET_GROUP, key=SECRET_NAME)],
    'instance_type': 'h200_8',
    'container_image': LAMER_IMAGE,
    'enable_identity_certs': True,
    'disruption_config': DisruptionConfig(
        readiness_status=DisruptionReadinessStatus.IS_READY,
        termination_grace_period_seconds=1800,
    ),
    'requests': Resources(mem='128Gi'),
    'task_config': PyTorch(num_workers=0),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_script_executable(path: str):
    mode = os.stat(path).st_mode
    os.chmod(path, mode | 0o111)


def _run_script(path: str, env: Dict[str, str] | None = None):
    """Run a shell script, merging optional env overrides into the current env."""
    _make_script_executable(path)
    merged_env = {**os.environ, **(env or {})}
    logger.info(f'Running script: {path}')
    subprocess.run(path, shell=True, check=True, env=merged_env, executable='/bin/bash')


def _git_proxy_setup():
    context = current_context()
    git_proxy = context.secrets.get(SECRET_GROUP, SECRET_NAME)
    os.system('git config --global http.proxyAuthMethod basic')
    os.system(f'git config --global http.proxy {git_proxy}')
    os.environ['HTTP_PROXY'] = git_proxy
    os.environ['HTTPS_PROXY'] = git_proxy
    os.environ['http_proxy'] = git_proxy
    os.environ['https_proxy'] = git_proxy


# ── idev task ─────────────────────────────────────────────────────────────────

@task(
    task_config=PyTorch(num_workers=0),
    enable_nfs=True,
    enable_identity_certs=True,
    proxy_as='coreaifomo',
    secret_requests=[Secret(group=SECRET_GROUP, key=SECRET_NAME)],
    instance_type='h200_1',
    container_image=LAMER_IMAGE,
)
@vscode(
    pre_execute=_git_proxy_setup,
    enable=True,
    max_idle_seconds=60 * 60 * 8,
)
def interactive_dev_task(_override_environment: Dict[str, str]):
    pass


@workflow(namespace='training-coreai')
def interactive_dev_workflow():
    env_vars = {}
    try:
        env_vars['GIT_USER_NAME'] = subprocess.check_output(
            'git config --global user.name', shell=True
        ).decode().strip()
        env_vars['GIT_USER_EMAIL'] = subprocess.check_output(
            'git config --global user.email', shell=True
        ).decode().strip()
    except subprocess.CalledProcessError:
        pass
    return interactive_dev_task(_override_environment=env_vars)


# ── ALFWorld multi-episode training task ──────────────────────────────────────

# Repo root as uploaded by openconnect
_ICE_ROOT = '/home/jobuser/resources'

# ALFWorld data baked into the lamer Docker image (via alfworld-download -f)
_ALFWORLD_DATA = '/home/jobuser/.cache/alfworld'

# Lamer virtualenv that has verl + alfworld installed
_LAMER_VENV_BIN = '/home/jobuser/.venv/lamer/bin'


@task(**BASE_TASK_CONFIG)
def run_alfworld_multi_episode(model_path: str, enable_reflection: bool = False):
    """Run ALFWorld multi-episode GRPO training (in-context-explorer)."""
    logger.info(
        f'=== ALFWorld multi-episode | model={model_path} reflection={enable_reflection} ==='
    )
    script = f'{_ICE_ROOT}/examples/grpo_trainer/run_alfworld_multi_episode.sh'
    _run_script(script, env={
        'ALFWORLD_DATA': _ALFWORLD_DATA,
        'PATH': f'{_LAMER_VENV_BIN}:{os.environ.get("PATH", "")}',
        'VIRTUAL_ENV': '/home/jobuser/.venv/lamer',
        'PYTHONPATH': f'{_ICE_ROOT}:{os.environ.get("PYTHONPATH", "")}',
        'model_path': model_path,
        'ENABLE_REFLECTION': str(enable_reflection),
    })


# ── Workflow definitions ───────────────────────────────────────────────────────

_QWEN3_8B_BASE = (
    '/shared/public/elr-models/Qwen/Qwen3-8B/'
    '2069b3fae1114555f3c020c81410e51fa0f656f2_130k_context'
)

_QWEN3_8B_FINETUNED = (
    '/shared/public/sharing/sirzhu/gem/'
    'gem-multi-task-multi-task-multi-episode-config-Qwen3-8B-'
    'disable-thinking-False-enable-reflection-False-fdce84b425f7b47e099b/'
    'global_step_100/actor_hf'
)

@workflow(namespace='training-coreai')
def ice_alfworld_multi_episode_workflow():
    """ALFWorld multi-episode GRPO training: Qwen3-8B base, no reflection."""
    run_alfworld_multi_episode(model_path=_QWEN3_8B_BASE, enable_reflection=False)


@workflow(namespace='training-coreai')
def ice_alfworld_multi_episode_reflection_workflow():
    """ALFWorld multi-episode GRPO training: Qwen3-8B base, with reflection."""
    run_alfworld_multi_episode(model_path=_QWEN3_8B_BASE, enable_reflection=True)


@workflow(namespace='training-coreai')
def ice_alfworld_multi_episode_finetuned_workflow():
    """ALFWorld multi-episode GRPO training: finetuned Qwen3-8B checkpoint, no reflection."""
    run_alfworld_multi_episode(model_path=_QWEN3_8B_FINETUNED, enable_reflection=False)
