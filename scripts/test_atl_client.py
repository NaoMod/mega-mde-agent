import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.mcp.client import ATLServerClient

def test_atl_client():
    client = ATLServerClient()
    # Test: get enabled transformations (generic call)
    try:
        result = client.call_tool("/transformations/enabled", method="GET")
        print('Enabled transformations:', result)
    except Exception as e:
        print('Error communicating with ATL server:', e)

if __name__ == "__main__":
    test_atl_client()
