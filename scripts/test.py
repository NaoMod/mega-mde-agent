#!/usr/bin/env python3
"""
Simple test for MCP client - Call specific tool
"""
import sys
import os
import asyncio


# Add the parent directory to sys.path to resolve imports correctly
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import modules with src. prefix
from src.agents.workflow import WorkflowExecutor
from src.core.megamodel import MegamodelRegistry
from src.agents.planning import AgentGoal, PlanStep, WorkflowPlan
from src.mcp.client import MCPClient

async def test_mcp_client():
    """Test the MCP client"""
    # Create client
    client = MCPClient()
    
    # Connect to server
    server_script_path = os.path.join(parent_dir, "mcp_servers/atl_server/atl_mcp_server.py")
    print(f"Connecting to server: {server_script_path}")
    
    try:
        await client.connect_to_server(server_script_path)
        
        # Get session
        session = await client.get_session()
        
        # List tools
        response = await session.list_tools()
        print("\nAvailable tools:", [tool.name for tool in response.tools])
        
        # Call the specific tool requested: list_transformation_Mantis2XML_tool
        print("\nCalling tool: list_transformation_Mantis2XML_tool")
        
        result = await session.call_tool("list_transformation_Mantis2XML_tool", {})
        print("\nTool result:", result)
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        await client.cleanup()

if __name__ == "__main__":
    success = asyncio.run(test_mcp_client())
    print("\n--- Running WorkflowExecutor test ---")


    # Setup registry and executor
    registry = MegamodelRegistry()
    # Register atl_server with script_path metadata for MCP connection
    registry.mcp_servers["atl_server"] = type("Server", (), {"metadata": {"script_path": os.path.join(parent_dir, "mcp_servers/atl_server/atl_mcp_server.py")}})()
    executor = WorkflowExecutor(registry)

    # Create a workflow plan with a step that calls list_transformation_Mantis2XML_tool
    goal = AgentGoal(description="Test workflow for Mantis2XML tool")
    plan = WorkflowPlan(goal=goal)
    step = PlanStep(
        tool_name="list_transformation_Mantis2XML_tool",
        server_name="atl_server",
        parameters={},
        description="Call Mantis2XML tool"
    )
    plan.add_step(step)

    result = executor.execute_workflow(plan)
    print("Workflow result:\n", result)
    sys.exit(0 if success else 1)
