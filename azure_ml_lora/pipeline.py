"""
LoRAPipeline class for automating Azure ML LoRA adapter training and registration.
"""

import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    AmlCompute,
    Environment,
    CommandJob,
    Model,
    Output,
    BuildContext,
)
import json

class LoRAPipeline:
    def __init__(self, subscription_id, resource_group, workspace_name):
        """
        Initialize the LoRAPipeline object.
        Args:
            subscription_id (str): Azure subscription ID
            resource_group (str): Azure resource group name
            workspace_name (str): Azure ML workspace name
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.ml_client = self._get_ml_client()

    def _get_ml_client(self):
        """
        Create and return an MLClient object.
        """
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            workspace_name=self.workspace_name,
        )
        return ml_client

    def provision_compute(self, compute_name="cpu-cluster", vm_size="Standard_DS2_v2", max_nodes=4):
        """
        Provision an Azure ML compute cluster.
        Args:
            compute_name (str): Name of the compute cluster
            vm_size (str): Virtual machine size
            max_nodes (int): Maximum number of nodes in the cluster
        Returns:
            str: The name of the provisioned compute cluster
        """
        try:
            compute = AmlCompute(
                name=compute_name,
                size=vm_size,
                max_nodes=max_nodes,
                min_nodes=0,
                idle_time_before_scale_down=120,
            )
            self.ml_client.compute.begin_create_or_update(compute)
            print(f"Provisioning compute cluster: {compute_name}")
            # Wait for the compute to be provisioned
            import time
            time.sleep(60)  # Adjust this based on your needs
            return compute_name
        except Exception as e:
            print(f"Error provisioning compute: {e}")
            raise

    def setup_environment(self, env_name="lora-env", conda_file="environment.yml"):
        """
        Set up the Azure ML environment for training.
        Args:
            env_name (str): Name of the environment
            conda_file (str): Path to the conda environment file
        Returns:
            str: The name of the created or updated environment
        """
        if env_name not in [e.name for e in self.ml_client.environments.list()]:
            # Create a new environment
            env = Environment(
                name=env_name,
                description="Environment for LoRA adapter training",
                docker_image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
                conda_file=conda_file,
                build=BuildContext(path=".")     # assumes environment.yml is in cwd
            )
            self.ml_client.environments.create_or_update(env)
        else:
            env = self.ml_client.environments.get(env_name, version="1")
        return env_name

    def submit_training_job(self, compute_name, env_name):
        """
        Define and submit the CommandJob for multi-LoRA training.
        Specifies compute, environment, code location, and training command.
        Returns:
            CommandJob: The submitted job object
        """
        job = CommandJob(
            display_name="lora-multi-adapter-job",
            compute=compute_name,
            environment=env_name,
            code="./",          # your training script folder
            command=(
                "python train_lora.py "
                "--model meta-llama/Llama-3.1-8B-Instruct "
                "--data-path ./data/train.jsonl "
                "--output-dir ./outputs "
                "--adapters qv ko"
            ),
            outputs={
                "qv_adapter": Output(type="uri_folder", path="./outputs/lora_qv"),
                "ko_adapter": Output(type="uri_folder", path="./outputs/lora_ko"),
            },
            experiment_name="lora_experiments",
        )
        submitted_job = self.ml_client.jobs.create_or_update(job)
        return submitted_job

    def stream_job_logs(self, job_name):
        """
        Stream job logs to the console until training completes.
        Args:
            job_name (str): Name of the submitted job
        """
        self.ml_client.jobs.stream(job_name)

    def register_adapters(self, submitted_job):
        """
        Register each trained adapter as a Model in Azure ML.
        For each adapter (qv, ko), registers the output folder as a model asset.
        Args:
            submitted_job (CommandJob): The completed job object
        """
        for adapter in ["qv", "ko"]:
            adapter_path = f"{submitted_job.outputs[adapter + '_adapter'].uri}"
            model = Model(
                path=adapter_path,
                name=f"lora-{adapter}-adapter",
                type="custom_model",   # or transformer_adapter
                description=f"LoRA {adapter} adapter for Llama-3.1-8B",
            )
            self.ml_client.models.create_or_update(model)
            print(f"Registered model: lora-{adapter}-adapter")

    # === Azure ML scoring logic for adapter selection ===
    model = None
    adapters = {}
    current_adapter = None

    @staticmethod
    def init():
        """
        Azure ML calls this once when the endpoint is started.
        Loads all available LoRA adapters and base model.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        LoRAPipeline.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # Load all adapters (replace with your actual adapter loading logic)
        LoRAPipeline.adapters["qv"] = "qv_adapter_path"
        LoRAPipeline.adapters["ko"] = "ko_adapter_path"
        LoRAPipeline.current_adapter = None
        print("Model and adapters loaded.")

    @staticmethod
    def run(raw_data):
        """
        Azure ML calls this for every request.
        Expects JSON with 'adapter_name' and 'input_data'.
        """
        try:
            data = raw_data if isinstance(raw_data, dict) else json.loads(raw_data)
            adapter_name = data.get("adapter_name")
            input_data = data.get("input_data")
            if adapter_name not in LoRAPipeline.adapters:
                return {"error": f"Adapter '{adapter_name}' not found."}
            if LoRAPipeline.current_adapter != adapter_name:
                # Example: LoRAPipeline.model.load_adapter(LoRAPipeline.adapters[adapter_name])
                LoRAPipeline.current_adapter = adapter_name
            # Example: run inference (replace with your actual logic)
            # output = LoRAPipeline.model.generate(input_data)
            output = f"[Simulated output for adapter '{adapter_name}' and input '{input_data}']"
            return {"result": output}
        except Exception as e:
            return {"error": str(e)}

    def run(self):
        """
        Main method to execute the full pipeline:
            1. Provision compute
            2. Set up environment
            3. Submit training job
            4. Stream logs
            5. Register adapters
        """
        compute_name = self.provision_compute()
        env_name = self.setup_environment()
        submitted_job = self.submit_training_job(compute_name, env_name)
        self.stream_job_logs(submitted_job.name)
        self.register_adapters(submitted_job)
        print("All done! Your LoRA adapters are trained and registered.")