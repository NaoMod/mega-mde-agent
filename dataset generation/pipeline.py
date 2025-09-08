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


def _build_type_graph(capabilities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a simple type graph to find possible 2-tool chains.

    Returns dict with:
      - tool_io: {tool_name: {"in": set, "out": set}}
      - follow_edges: {tool_name: [tool_name,...]} where out intersects next.in
      - precede_edges: inverse of follow_edges
    """
    tool_io: Dict[str, Dict[str, set]] = {}
    for c in capabilities or []:
        name = c.get("tool_name", "")
        ins = set(c.get("input_types", []) or [])
        outs = set(c.get("output_types", []) or [])
        tool_io[name] = {"in": ins, "out": outs}

    follow_edges: Dict[str, List[str]] = {k: [] for k in tool_io.keys()}
    names = list(tool_io.keys())
    for i, a in enumerate(names):
        for b in names:
            if a == b:
                continue
            if tool_io[a]["out"] and tool_io[a]["out"].intersection(tool_io[b]["in"]):
                follow_edges[a].append(b)
    precede_edges: Dict[str, List[str]] = {k: [] for k in tool_io.keys()}
    for a, bs in follow_edges.items():
        for b in bs:
            precede_edges[b].append(a)
    return {"tool_io": tool_io, "follow_edges": follow_edges, "precede_edges": precede_edges}


def _serialize_historical_executions(registry: MegamodelRegistry) -> List[Dict[str, Any]]:
    """Extract successfully executed AgentTraces from the registry."""
    executions: List[Dict[str, Any]] = []
    for session_id, session in getattr(registry, "sessions", {}).items():
        traces_payload = []
        for trace in getattr(session, "execution_traces", []) or []:
            invocs = []
            for inv in getattr(trace, "invocations", []) or []:
                invocs.append({
                    "tool_name": getattr(inv, "tool_name", None),
                    "server_name": getattr(inv, "server_name", None),
                    "success": bool(getattr(inv, "success", False)),
                    "timestamp": getattr(getattr(inv, "timestamp", None), "isoformat", lambda: None)(),
                })
            traces_payload.append({
                "trace_id": getattr(trace, "trace_id", ""),
                "invocations": invocs,
            })
        executions.append({
            "session_id": session_id,
            "status": getattr(session, "status", "unknown"),
            "traces": traces_payload,
        })
    return executions

def main() -> None:
    # 1. Get a populated registry
    registry = MegamodelRegistry()
    populate_registry(registry)

    # 2. Get actual tools from the registry
    atl_tools = registry.tools_by_server.get("atl_server", [])
    emf_tools = registry.tools_by_server.get("emf_server", [])
    
    # First get the capabilities
    tools = [
        {"name": getattr(t, "name", ""), "description": getattr(t, "description", "")}
        for t in [*atl_tools, *emf_tools]
    ]
    capabilities = _infer_capabilities_from_registry(registry, tools)
    
    # Test 1: Capability inference (commented out)
    #print("\nInferred Capabilities:")
    #for cap in capabilities:
    #    print(f"\nTool: {cap['tool_name']}")
    #    print(f"Input types: {cap['input_types']}")
    #    print(f"Output types: {cap['output_types']}")
    
    # Test 2: Type graph building
    print("\nTesting _build_type_graph:")
    type_graph = _build_type_graph(capabilities)
    

    print("\n Tool Chains (Follow Edges):")
    for tool_name, followers in type_graph["follow_edges"].items():
        if followers:  # Only show tools that can be followed by others
            print(f"\n{tool_name} can be followed by:")
            for follower in followers:
                print(f"  → {follower}")
    
    print("\n Tool Dependencies (Precede Edges):")
    for tool_name, predecessors in type_graph["precede_edges"].items():
        if predecessors:  # Only show tools that have predecessors
            print(f"\n{tool_name} can be preceded by:")
            for pred in predecessors:
                print(f"  ← {pred}")

if __name__ == "__main__":
    main()
