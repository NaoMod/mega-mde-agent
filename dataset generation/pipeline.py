import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys
WORKDIR = Path(__file__).resolve().parents[1]
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
SRC_DIR = WORKDIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from scripts.run_agent import populate_registry
from src.core.megamodel import MegamodelRegistry


# --- 1) Start with a megamodel repository (populate registry) ---
def _norm_transfo_name(name: str) -> str:
    return (name or "").strip().lower()

def _infer_capabilities_from_registry(registry: MegamodelRegistry, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Discovers what models each transformation tool can process by looking up their input/output metamodel types in the registry."""
    # Get transformation name -> (in_uri, out_uri) from registry
    transfo_types = {
        _norm_transfo_name(getattr(e, "name", "")): (
            getattr(getattr(e, "source_metamodel", None), "uri", None),
            getattr(getattr(e, "target_metamodel", None), "uri", None)
        )
        for e in registry.entities.values()
        if hasattr(e, "source_metamodel") or hasattr(e, "target_metamodel")
    }

    def get_tool_types(name: str) -> tuple[list, list]:
        if not name.startswith("apply_") or not name.endswith("_transformation_tool"):
            return [], []  # Non-apply tools (including list_*) have no strict types
        base = _norm_transfo_name(name[len("apply_"):-len("_transformation_tool")])
        in_uri, out_uri = transfo_types.get(base, (None, None))
        return ([in_uri] if in_uri else []), ([out_uri] if out_uri else [])

    return [
        {
            "tool_name": t.get("name", ""),
            "input_types": in_types,
            "output_types": out_types
        }
        for t in tools
        for in_types, out_types in [get_tool_types(t.get("name", ""))]
    ]




def main() -> None:
    # 1. Get a populated registry
    registry = MegamodelRegistry()
    populate_registry(registry)

    # 2. Get actual tools from the registry
    atl_tools = registry.tools_by_server.get("atl_server", [])
    emf_tools = registry.tools_by_server.get("emf_server", [])
    
    # Convert to the dict format expected by infer_capabilities
    tools = [
        {"name": getattr(t, "name", ""), "description": getattr(t, "description", "")}
        for t in [*atl_tools, *emf_tools]
    ]

    # 3. Test the capability inference
    caps = _infer_capabilities_from_registry(registry, tools)
    print("\nInferred Capabilities:")
    for cap in caps:
        print(f"\nTool: {cap['tool_name']}")
        print(f"Input types: {cap['input_types']}")
        print(f"Output types: {cap['output_types']}")


if __name__ == "__main__":
    main()
