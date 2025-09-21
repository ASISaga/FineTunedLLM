# run_lora_pipeline.py
# This script automates the end-to-end process of training and registering LoRA adapters using Azure ML.
# Steps:
#   1. Authenticate and connect to Azure ML workspace
#   2. Provision (or get) a spot-priced GPU cluster
#   3. Create (or get) the Conda environment for training
#   4. Define and submit a CommandJob for multi-LoRA training
#   5. Wait for job completion
#   6. Register each trained adapter as a Model in Azure ML


import os
from .manager import MLManager
import json

if __name__ == "__main__":

    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]
    resource_group = os.environ["AZURE_RESOURCE_GROUP"]
    workspace_name = os.environ["AZURE_WORKSPACE"]
    ml_manager = MLManager(subscription_id, resource_group, workspace_name)
    
    # Provision compute and environment before training
    compute = ml_manager.get_or_create_compute()
    env = ml_manager.get_or_create_environment()
    
    # Load training parameters and adapters from files
    with open("./configs/lora_training_params.json", "r") as f:
        training_params = json.load(f)
    with open("./configs/lora_adapters.json", "r") as f:
        adapters = json.load(f)

    adapters = ml_manager.train_adapters(training_params, adapters)
    print("All done! Your LoRA adapters are trained and registered.")

    # Start all adapters
    for adapter_cfg in adapters:
        adapter_name = adapter_cfg["adapter_name"]
        ml_manager.start_adapter(adapter_name)

    # Publish each adapter individually
    for adapter_cfg in adapters:
        adapter_name = adapter_cfg["adapter_name"]
        print(f"Publishing adapter: {adapter_name}")
        ml_manager.publish_adapter(adapter_name)

    # ... do inference or other tasks ...

    # Stop all adapters
    for adapter_cfg in adapters:
        adapter_name = adapter_cfg["adapter_name"]
        ml_manager.stop_adapter(adapter_name)