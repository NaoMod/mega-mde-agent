import json
import os
import asyncio
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
def query_megamodel_repo() -> Dict[str, Any]:
    registry = MegamodelRegistry()
    # Populate the registry exactly like the agent does
    populate_registry(registry)

    # Pull tools from the populated registry
    atl_tools = registry.tools_by_server.get("atl_server", [])
    emf_tools = registry.tools_by_server.get("emf_server", [])

    # Flatten tools into a dump format the rest of the pipeline expects
    def tool_to_dict(t: Any, server_name: str) -> Dict[str, Any]:
        return {
            "name": getattr(t, "name", ""),
            "description": getattr(t, "description", ""),
            "server_name": server_name,
        }

    dump = {
        "MCPServers": [
            {"name": "atl_server"},
            {"name": "emf_server"},
        ],
        "MCPTools": [
            *(tool_to_dict(t, "atl_server") for t in atl_tools),
            *(tool_to_dict(t, "emf_server") for t in emf_tools),
        ],
        "MCPCapabilities": [],  # not used here
        "HistoricalExecutions": [],  # can be filled from registry.sessions if needed
    }
    return dump

# --- 2) Extract component info ---
def extract_components(repo_dump: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "servers": repo_dump.get("MCPServers", []),
        "tools": repo_dump.get("MCPTools", []),
        "capabilities": repo_dump.get("MCPCapabilities", []),
        "executions": repo_dump.get("HistoricalExecutions", []),
    }

# --- 3a) Capability analysis ---
def analyze_capabilities(components: Dict[str, Any]) -> Dict[str, Any]:
    return {
    }

# --- 3b) Pattern discovery ---
def discover_patterns(components: Dict[str, Any]) -> Dict[str, Any]:
   
    return {
        "common_patterns": [
            # ["ToolA -> ToolB"], ["ToolX -> ToolY -> ToolZ"]
        ]
    }

# --- 4) Generate insights ---
def generate_insights(capability_report: Dict[str, Any], pattern_report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "specific_capabilities": capability_report.get("capability_summaries", []),
        "common_workflows": pattern_report.get("common_patterns", []),
    }

# --- 5) API sampling ---
def sample_apis(components: Dict[str, Any], insights: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Minimal strategy: first N tools (deterministic)
    tools = components.get("tools", [])
    return tools[:10]

# --- 6) API selection ---
def select_apis(sampled: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    return sampled[:k]

# --- 7) LLM-based generation (single vs multi tool) ---
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
) -> List[Dict[str, Any]]:
    from itertools import islice, combinations
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.2, max_retries=2) if (ChatOpenAI and os.getenv("OPENAI_API_KEY")) else None
    items: List[Dict[str, Any]] = []
    tools = [t for t in selected_apis]
    if len(tools) < chain_len:
        return items
    llm_calls = 0
    combos = combinations(tools, chain_len)
    for combo in combos:
        # Build API chain
        apis = []
        patterns = []
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


def main() -> None:
    cfg = {
        "repo": "megamodel://local",
        "k": 3,
        "prompt_path": str(ROOT / "prompts" / "template.txt"),
    "per_api_instructions": 3,
    "llm_max_calls": 5,
    }
    dump = query_megamodel_repo()
    components = extract_components(dump)

    cap = analyze_capabilities(components)
    pat = discover_patterns(components)
    insights = generate_insights(cap, pat)

    sampled = sample_apis(components, insights)
    selected = select_apis(sampled, k=cfg["k"])

    # Generation settings
    single = generate_single_tool_instructions(
        selected,
        per_api=cfg["per_api_instructions"],
        llm_max_calls=cfg["llm_max_calls"],
    )
    # Optional multi-tool examples (disabled by default); set cfg["multi_chain_len"] > 1 to enable
    multi = []
    if cfg.get("multi_chain_len", 0) and cfg["multi_chain_len"] > 1:
        multi = generate_multi_tool_instructions(
            selected,
            chain_len=int(cfg["multi_chain_len"]),
            per_item=cfg.get("per_multi_instructions", 1),
            llm_max_calls=max(0, cfg["llm_max_calls"] - len(single)),
        )
    examples = single + multi

    create_output_examples(examples, OUT / "examples.json")
    validated = validate_dataset(examples)
    write_final_dataset(validated, OUT / "final_dataset.json")

    print("Done. Outputs in:", OUT)


if __name__ == "__main__":
    main()
