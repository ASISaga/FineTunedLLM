#!/bin/bash
# Azure ML CLI commands for LoRA training

# Create and submit training job
az ml job create --file job.yaml --web

# Register trained adapters as models
az ml model create --name lora-qv \
  --path outputs/lora_qv --type transformer_adapter

az ml model create --name lora-ko \
  --path outputs/lora_ko --type transformer_adapter