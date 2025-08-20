"""
Class2Relational Example - Simple transformation example
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.megamodel import MegamodelRegistry
from core.am3 import ReferenceModel, TransformationModel, TerminalModel
from mcp.integrator import MCPServerIntegrator
from agents.workflow import WorkflowExecutor
from agents.planning import AgentGoal, WorkflowPlan, PlanStep

def run_class2relational_example():
    """Simple Class2Relational transformation example"""
    print("🔄 Class2Relational Transformation Example")
    print("-" * 40)
    
    # Setup
    registry = MegamodelRegistry()
    integrator = MCPServerIntegrator(registry)
    
    atl_server = integrator.setup_atl_server()
    emf_server = integrator.setup_emf_server()
    
    # Register metamodels
    class_mm = ReferenceModel(uri="Class.ecore", name="Class")
    relational_mm = ReferenceModel(uri="Relational.ecore", name="Relational") 
    
    registry.register_entity(class_mm)
    registry.register_entity(relational_mm)
    
    # Create sample model
    sample_model = TerminalModel(
        uri="sample_class.xmi",
        name="Sample Class Model",
        conformsTo=class_mm
    )
    registry.register_entity(sample_model)
    
    # Create workflow
    goal = AgentGoal(
        description="Apply Class2Relational transformation",
        success_criteria={"transformation_completed": True}
    )
    
    plan = WorkflowPlan(goal=goal)
    
    # Add transformation step
    transform_step = PlanStep(
        tool_name="apply_class2relational_transformation_tool",
        server_name="atl_server",
        parameters={"file_path": "sample_class.xmi"},
        description="Apply Class2Relational transformation"
    )
    plan.add_step(transform_step)
    
    # Execute
    executor = WorkflowExecutor(registry)
    result = executor.execute_workflow(plan)
    
    print(f"Result: {result['status']}")
    return result

if __name__ == "__main__":
    run_class2relational_example()