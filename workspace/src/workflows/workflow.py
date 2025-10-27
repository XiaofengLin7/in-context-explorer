from flytekit import task, workflow
from flytekit.core.disruption_config import DisruptionConfig, DisruptionReadinessStatus
from flytekitplugins.kfpytorch import PyTorch
from lifomo.core.utils.shell_utils import execute_script, make_script_executable
from flytekit import task, workflow


@task(
    enable_nfs=True,
    task_config=PyTorch(num_workers=0),
    instance_type='h200_4',
    enable_identity_certs=True,
    container_image='container-image-registry.corp.linkedin.com/temp/lifomo/lifomo-verl:202509040120',
    disruption_config=DisruptionConfig(
        readiness_status=DisruptionReadinessStatus.IS_READY,
        termination_grace_period_seconds=1800,
    ),
)
def run_training(_override_proxy_as: str, _override_instance_type: str, training_script: str):
    make_script_executable(training_script)
    execute_script(training_script, [])


@workflow(namespace='training-coreai')
def alfworld_workflow():
    run_training(
        _override_proxy_as='coreaifomo',
        _override_instance_type='h200_4',
        training_script='/home/jobuser/resources/workspace/scripts/run_alfworld.sh')

