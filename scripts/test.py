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

# Import MCPClient from the src package
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
    sys.exit(0 if success else 1)
