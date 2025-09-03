"""
Main Entry Point - Simple demonstration
"""
import sys
import os

from scripts.run_agent import populate_registry

# Add src to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import AFTER path is set up
from core.megamodel import MegamodelRegistry
from core.am3 import ReferenceModel, TransformationModel
from mcp_ext.integrator import MCPServerIntegrator
from agents.workflow import WorkflowExecutor
from agents.planning import AgentGoal, WorkflowPlan, PlanStep

def main():
    """Simple main entry point"""
    # initialize MegamodelRegistry and print its attributes
    registry = MegamodelRegistry()
    populate_registry(registry)
    print("\n--- MegamodelRegistry Attributes After Initialization ---")
    print(f"entities: {registry.entities}")
    print(f"relationships: {registry.relationships}")
    print(f"mcp_servers: {registry.mcp_servers}")
    print(f"tools_by_server: {registry.tools_by_server}")
    print(f"sessions: {registry.sessions}")
    print(f"workflow_plans: {registry.workflow_plans}")
    print(f"_models_by_type: {registry._models_by_type}")
    print("--- End MegamodelRegistry Attributes ---\n")
    return True

if __name__ == "__main__":
    main()