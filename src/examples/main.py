"""
Main Entry Point - Simple demonstration
"""
import sys
import os

# Add src to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import AFTER path is set up
from core.megamodel import MegamodelRegistry
from core.am3 import ReferenceModel, TransformationModel
from mcp.integrator import MCPServerIntegrator
from agents.workflow import WorkflowExecutor
from agents.planning import AgentGoal, WorkflowPlan, PlanStep

def main():
    """Simple main entry point"""
    print("🎯 LLM-Based Autonomous Agents for MDE")
    print("=" * 50)
    
    # 1. Initialize registry
    print("1. Initializing megamodel...")
    registry = MegamodelRegistry()
    
    # 2. Setup MCP servers
    print("2. Setting up MCP servers...")
    integrator = MCPServerIntegrator(registry)
    
    atl_server = integrator.setup_atl_server()
    emf_server = integrator.setup_emf_server()
    
    print(f"   ATL Server: {atl_server.name}")
    print(f"   EMF Server: {emf_server.name}")
    
    # 3. Register some basic entities
    print("3. Registering entities...")
    
    class_mm = ReferenceModel(uri="Class.ecore", name="Class Metamodel")
    relational_mm = ReferenceModel(uri="Relational.ecore", name="Relational Metamodel")
    
    registry.register_entity(class_mm)
    registry.register_entity(relational_mm)
    
    transformation = TransformationModel(
        uri="Class2Relational.atl",
        name="Class2Relational",
        source_metamodel=class_mm,
        target_metamodel=relational_mm
    )
    registry.register_entity(transformation)
    
    # 4. Create and execute simple workflow
    print("4. Creating workflow...")
    executor = WorkflowExecutor(registry)
    
    goal = AgentGoal(
        description="Transform Class model to Relational model",
        success_criteria={"transformation_applied": True}
    )
    
    plan = WorkflowPlan(goal=goal)
    
    # Add simple steps
    step1 = PlanStep(
        tool_name="extract_input_metamodel_name",
        server_name="atl_server", 
        parameters={"file_path": "sample.xmi"},
        description="Extract metamodel"
    )
    plan.add_step(step1)
    
    print("5. Executing workflow...")
    result = executor.execute_workflow(plan)
    
    print(f"   Status: {result['status']}")
    print(f"   Session: {result['session_id']}")
    
    # 6. Show statistics
    print("6. Registry statistics:")
    stats = registry.get_execution_statistics()
    print(f"   Entities: {stats['registered_entities']}")
    print(f"   Servers: {stats['active_mcp_servers']}")
    print(f"   Sessions: {stats['total_sessions']}")
    
    print("\n✅ Demo completed!")
    return True

if __name__ == "__main__":
    main()