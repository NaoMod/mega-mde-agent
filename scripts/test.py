import sys
import os
from src.core.am3 import ReferenceModel
from src.core.megamodel import MegamodelRegistry
from src.mcp.integrator import MCPServerIntegrator

def test_basic():
    print("Testing basic functionality...")
    try:
        registry = MegamodelRegistry()
        model = ReferenceModel(uri="test.ecore", name="Test")
        registry.register_entity(model)
        
        integrator = MCPServerIntegrator(registry)
        atl_server = integrator.setup_atl_server()
        
        print("Basic test passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_basic()
    sys.exit(0 if success else 1)