"""
Unified Azure ML Management System
Consolidates functionality from app/aml.py and api/MLClientManager.py
"""
import os
import json
import httpx
from typing import Dict, Any

try:
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False

from .endpoints import AML_ENDPOINTS


class UnifiedMLManager:
    """
    Unified Azure ML management that consolidates:
    - AML endpoint inference
    - ML client management
    - Pipeline orchestration
    - Training job management
    """

    def __init__(self):
        # Initialize environment variables
        try:
            # Try to import from parent BusinessInfinity environment module
            import sys
            import os.path
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sys.path.insert(0, parent_dir)
            from BusinessInfinity.environment import env_manager as env
            self.subscription_id = env.get_required("AZURESUBSCRIPTION_ID")
            self.resource_group_name = env.get_required("AZURERESOURCEGROUP")
            self.workspace_name = env.get_required("AZUREML_WORKSPACE")
            self.pipeline_name = env.get_required("PIPELINEENDPOINT_NAME")
        except (ImportError, Exception):
            # Fallback to direct environment variables
            self.subscription_id = os.getenv("AZURESUBSCRIPTION_ID")
            self.resource_group_name = os.getenv("AZURERESOURCEGROUP")
            self.workspace_name = os.getenv("AZUREML_WORKSPACE")
            self.pipeline_name = os.getenv("PIPELINEENDPOINT_NAME")
        
        self._client = None

    @property
    def AML_ENDPOINTS(self):
        """Backwards compatibility property"""
        return AML_ENDPOINTS

    def get_client(self):
        """Get Azure ML client (singleton pattern)"""
        if not AZURE_ML_AVAILABLE:
            raise ImportError("Azure ML libraries are required for ML operations")
            
        if self._client is None:
            self._client = MLClient(
                DefaultAzureCredential(),
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group_name,
                workspace_name=self.workspace_name
            )
        return self._client

    async def aml_infer(self, agent_id: str, prompt: str) -> Dict[str, Any]:
        """
        Perform inference using AML endpoint for a specific agent
        
        Args:
            agent_id: Agent identifier (cmo, cfo, cto)
            prompt: Input prompt for inference
            
        Returns:
            Dict containing inference result or error
        """
        cfg = AML_ENDPOINTS.get(agent_id)
        if not cfg or not cfg["scoring_uri"]:
            return {"error": f"No AML endpoint configured for {agent_id}"}
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['key']}"
        }
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(cfg["scoring_uri"], headers=headers, json={"input": prompt})
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            return {"error": f"AML inference failed for {agent_id}: {str(e)}"}

    async def aml_train(self, job_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start AML training job
        
        Args:
            job_name: Name for the training job
            params: Training parameters
            
        Returns:
            Dict with job information
        """
        try:
            # Enhanced training logic can be added here
            # For now, return a structured response
            return {
                "jobId": f"job-{job_name}",
                "status": "queued",
                "params": params
            }
        except Exception as e:
            return {"error": f"Training job creation failed: {str(e)}"}

    def get_input_path(self, domain: str) -> str:
        """Return AML datastore input path for mentor Q&A for the given domain"""
        return f"azureml://datastores/workspaceblob/paths/mentorqa/{domain}mentor_qa.jsonl"

    def invoke_pipeline(self, domain: str) -> str:
        """
        Invoke AML pipeline for a specific domain
        
        Args:
            domain: Domain identifier
            
        Returns:
            JSON string with pipeline job information
        """
        try:
            client = self.get_client()
            inputpath = self.get_input_path(domain)
            result = client.pipelineendpoints.invoke(
                name=self.pipeline_name,
                inputs={"qajsonl": inputpath}
            )
            return json.dumps({"pipelinejobid": result.id})
        except Exception as e:
            return json.dumps({"error": f"Pipeline invocation failed: {str(e)}"})

    def get_endpoint_config(self, agent_id: str) -> Dict[str, str]:
        """Get endpoint configuration for an agent"""
        return AML_ENDPOINTS.get(agent_id, {})

    def list_configured_agents(self) -> list:
        """List agents with configured AML endpoints"""
        return [agent_id for agent_id, cfg in AML_ENDPOINTS.items() if cfg.get("scoring_uri")]

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate AML configuration"""
        issues = []
        
        # Check workspace configuration
        required_vars = ["subscription_id", "resource_group_name", "workspace_name"]
        for var in required_vars:
            if not getattr(self, var):
                issues.append(f"Missing {var}")
        
        # Check agent endpoints
        configured_agents = self.list_configured_agents()
        if not configured_agents:
            issues.append("No agent endpoints configured")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "configured_agents": configured_agents,
            "workspace": f"{self.subscription_id}/{self.resource_group_name}/{self.workspace_name}"
        }