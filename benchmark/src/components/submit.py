# %%
from azure.ai.ml import MLClient, command, Input, Output, PyTorchDistribution
from azure.ai.ml.entities import JobService
from azure.identity import DefaultAzureCredential


def submit_pytorch_job(ml_client: MLClient, compute_name='a100-900ram-low'):
    command_job = command(
        code="./",
        command="python -u pytorch_benchmark/image_classifier.py --train_images ${{inputs.train_images}} --valid_images ${{inputs.valid_images}}  --batch_size ${{inputs.batch_size}}  --num_workers ${{inputs.num_workers}}  --prefetch_factor ${{inputs.prefetch_factor}}  --persistent_workers ${{inputs.persistent_workers}}  --pin_memory ${{inputs.pin_memory}}  --non_blocking ${{inputs.non_blocking}}  --model_arch ${{inputs.model_arch}}  --model_arch_pretrained ${{inputs.model_arch_pretrained}}  --num_epochs ${{inputs.num_epochs}}  --learning_rate ${{inputs.learning_rate}}  --momentum ${{inputs.momentum}}  --model_output ${{outputs.trained_model}}  --checkpoints ${{outputs.checkpoints}}  --cudnn_autotuner ${{inputs.cudnn_autotuner}}  --enable_profiling ${{inputs.enable_profiling}}  --enable_netmon ${{inputs.enable_netmon}}",
        environment="acpt-113-py38-cuda117-transformers:1",
        environment_variables={
            'NCCL_DEBUG': 'INFO'
        },
        inputs={
            # data inputs
            "train_images": Input(
                type="uri_folder",
                mode="download",
                path="azureml://datastores/benchmark/paths/azureml-vision-datasets/places2/train/**",
            ),
            "valid_images": Input(
                type="uri_folder",
                mode="download",
                path="azureml://datastores/benchmark/paths/azureml-vision-datasets/places2/valid/**",
            ),
            # data loading
            "batch_size": 64,
            "num_workers": 5,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "pin_memory": True,
            "non_blocking": False,
            # model
            "model_arch": "resnet18",
            "model_arch_pretrained": True,
            # training
            "num_epochs": 7,
            "learning_rate": 0.001,
            "momentum": 0.9,
            # profiling
            "enable_profiling": True,
            "enable_netmon": True,
            "cudnn_autotuner": True,

        },
        outputs={
            "checkpoints": Output(type="uri_folder"),
            "trained_model": Output(type="uri_folder")
        },
        compute=compute_name,
        distribution=PyTorchDistribution(process_count_per_instance=8),
        instance_count=1,
        shm_size="64g"
        # services={
        #     'jupyterlab': JobService(job_service_type='jupyter_lab'),
        #     'ssh': JobService(job_service_type='ssh'),
        #     'vscode': JobService(job_service_type='vs_code')
        # }
    )

    returned_job = ml_client.jobs.create_or_update(command_job, tags={'profling': 'only_flops'})
    print(returned_job.studio_url)


# %%
def __main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--sub-id", type=str, default='')
    parser.add_argument("--resource-group", type=str, default='lupickup-dev')
    parser.add_argument("--workspace-name", type=str, default='lupickup-test-eastus')
    parser.add_argument("--compute-name", type=str, default='lupickup-8v100-low')
    args = parser.parse_args()

    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id=args.sub_id, resource_group_name=args.resource_group, workspace_name=args.workspace_name
    )

    submit_pytorch_job(ml_client, compute_name=args.compute_name)


if __name__ == '__main__':
    __main()