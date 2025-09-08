import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys
WORKDIR = Path(__file__).resolve().parents[1]
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))
from collections import Counter
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


def discover_patterns(components: Dict[str, Any]) -> Dict[str, Any]:
    """Identify historical workflow patterns from successful AgentTraces.

    Returns a list of pattern entries with:
    - pattern: compact token chain (e.g., "get>apply")
    - tools: ordered list of tool names in the chain
    - count: frequency observed across traces
    """
    executions = components.get("executions", []) or []
    pattern_counter: Counter = Counter()
    pattern_examples: Dict[str, List[str]] = {}

    for sess in executions:
        for tr in sess.get("traces", []) or []:
            invocs = [i for i in tr.get("invocations", []) or [] if i.get("success")]
            if not invocs:
                continue
            tool_chain = [i.get("tool_name", "") for i in invocs if i.get("tool_name")]
            if not tool_chain:
                continue
            # Map tool names to coarse pattern tokens
            tokens = [_derive_api(t)[1] for t in tool_chain]
            pattern = ">".join(tokens)
            pattern_counter[pattern] += 1
            pattern_examples.setdefault(pattern, tool_chain)

    common_patterns = [
        {"pattern": pat, "tools": pattern_examples.get(pat, []), "count": cnt}
        for pat, cnt in pattern_counter.most_common()
    ]
    # Build type graph from MCPCapabilities
    type_graph = _build_type_graph(components.get("capabilities", []) or [])
    return {"common_patterns": common_patterns, "type_graph": type_graph}



def sample_apis(components: Dict[str, Any], insights: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Prefer tools seen in historical patterns; add bias from type connectivity; fallback to first N tools.
    tools = components.get("tools", [])
    if not tools:
        return []
    # Pattern frequency
    freq: Dict[str, int] = {}
    for entry in insights.get("common_workflows", []) or []:
        for t in entry.get("tools", []) or []:
            freq[t] = freq.get(t, 0) + entry.get("count", 1)
    # Type connectivity (follow + precede degree)
    graph = insights.get("type_graph", {}) or {}
    follow = graph.get("follow_edges", {})
    precede = graph.get("precede_edges", {})
    degree: Dict[str, int] = {}
    for name in set(list(follow.keys()) + list(precede.keys())):
        degree[name] = len(follow.get(name, []) or []) + len(precede.get(name, []) or [])

    by_name = {t.get("name", ""): t for t in tools}
    def score(name: str) -> tuple:
        return (freq.get(name, 0), degree.get(name, 0))
    ranked_names = sorted(by_name.keys(), key=lambda n: score(n), reverse=True)
    ranked = [by_name[n] for n in ranked_names if n in by_name]
    # Deterministic tail
    fallback = [t for t in tools if t.get("name", "") not in by_name]
    return (ranked + fallback)[:10]




# --- Utilité: Instead of working with long tool names like "apply_Class2Relational_transformation_tool", We get simple patterns like "apply" or "get" 
def _derive_api(tool_name: str) -> Tuple[str, str]:
    if tool_name.startswith("list_transformation_") and tool_name.endswith("_tool"):
        base = tool_name[len("list_transformation_"):-len("_tool")]
        return f"{base}.get_tool", "get"
    if tool_name.startswith("apply_") and tool_name.endswith("_transformation_tool"):
        base = tool_name[len("apply_"):-len("_transformation_tool")]
        return f"{base}.apply", "apply"
    if tool_name.startswith("extract_"):
        return f"model.{tool_name}", "extract"
    return tool_name, "call"

def generate_single_tool_instructions(
    selected_apis: List[Dict[str, Any]],
    per_api: int = 1,
    llm_max_calls: int = 5,
    prompt: str | None = None,
) -> List[Dict[str, Any]]:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.1, max_retries=2) if (ChatOpenAI and os.getenv("OPENAI_API_KEY")) else None
    items: List[Dict[str, Any]] = []
    if not selected_apis:
        return items
    llm_calls = 0
    n = max(1, int(per_api))
    for tool in selected_apis:
        name, desc = tool.get("name", ""), tool.get("description", "")
        api_name, pattern = _derive_api(name)
        for _ in range(n):
            if llm is None or llm_calls >= llm_max_calls:
                continue
            p = prompt or (
                "Write one short user instruction for a modeling task that requires a specific API.\n"
                f"Capability description:\n{desc}\n"
                "Constraints:\n- Do NOT mention any tool/API names.\n- Keep it concise and task-oriented.\nReturn only the instruction text."
            )
            try:
                msg = llm.invoke(p)
                instruction = getattr(msg, "content", str(msg)).strip()
                llm_calls += 1
            except Exception:
                continue
            items.append({
                "level": 1,
                "pattern": pattern,
                "instruction": instruction,
                "relevant_apis": [{"api_name": api_name, "arguments": ""}],
            })
    return items

def generate_multi_tool_instructions(
    selected_apis: List[Dict[str, Any]],
    chain_len: int = 2,
    per_item: int = 1,
    llm_max_calls: int = 5,
    prompt: str | None = None,
    capabilities: List[Dict[str, Any]] | None = None,
    enforce_type_compat: bool = False,
) -> List[Dict[str, Any]]:
   
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.2, max_retries=2) if (ChatOpenAI and os.getenv("OPENAI_API_KEY")) else None
    items: List[Dict[str, Any]] = []
    tools = [t for t in selected_apis]
    if len(tools) < chain_len:
        return items
    llm_calls = 0
    combos = combinations(tools, chain_len)
    # Optional type-compat enforcement for 2-tool chains
    cap_index = {c.get("tool_name"): c for c in (capabilities or [])}
    for combo in combos:
        # Build API chain
        apis = []
        patterns = []
        names = [t.get("name", "") for t in combo]
        # Enforce type compatibility only for 2-tool chains
        if enforce_type_compat and len(names) == 2:
            a, b = names
            a_out = set(cap_index.get(a, {}).get("output_types", []) or [])
            b_in = set(cap_index.get(b, {}).get("input_types", []) or [])
            if a_out and b_in and not a_out.intersection(b_in):
                continue
        for t in combo:
            api, pat = _derive_api(t.get("name", ""))
            apis.append({"api_name": api, "arguments": ""})
            patterns.append(pat)
        for _ in range(max(1, int(per_item))):
            if llm is None or llm_calls >= llm_max_calls:
                continue
            p = prompt or (
                "Propose one concise user instruction for a multi-step modeling workflow.\n"
                "Requirements:\n- The task should naturally require multiple APIs in sequence.\n"
                "- Do NOT mention any tool/API names.\n- Return only the instruction text."
            )
            try:
                msg = llm.invoke(p)
                instruction = getattr(msg, "content", str(msg)).strip()
                llm_calls += 1
            except Exception:
                continue
            items.append({
                "level": 2,
                "pattern": ">".join(patterns),
                "instruction": instruction,
                "relevant_apis": apis,
            })
    return items

def validate_dataset(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ok: List[Dict[str, Any]] = []
    for e in examples:
        if not isinstance(e, dict):
            continue
        instr = e.get("instruction")
        apis = e.get("relevant_apis", [])
        if instr and isinstance(apis, list) and apis and all(isinstance(a, dict) and a.get("api_name") for a in apis):
            ok.append(e)
    return ok

# --- 10) Final dataset ---
def write_final_dataset(examples: List[Dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(examples, indent=2))

def create_output_examples(examples: List[Dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(examples, indent=2))
