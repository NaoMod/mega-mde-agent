import json
import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List
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
def query_megamodel_repo(config: Dict[str, Any]) -> Dict[str, Any]:
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

# --- 7) LLM-based generation ---
def llm_generation(
    selected_apis: List[Dict[str, Any]],
    prompt_template: str,
    per_api_instructions: int = 1,
    llm_max_calls: int = 5,
) -> List[Dict[str, Any]]:
    """Generate per_api_instructions instructions for each selected API.
    LLM calls are capped by llm_max_calls; remaining use the template.
    """
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    llm = None
    if ChatOpenAI is not None and os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model=model_name, temperature=0.1, max_retries=2)

    examples: List[Dict[str, Any]] = []
    if not selected_apis:
        return examples

    llm_calls = 0
    n = max(1, int(per_api_instructions))
    for tool in selected_apis:
        name = tool.get("name", "unknown_tool")
        desc = tool.get("description", "")
        server = tool.get("server_name", "")
        for _ in range(n):
            # Only use LLM; if unavailable or capped, skip generating this entry
            if llm is None or llm_calls >= llm_max_calls:
                continue
            prompt = (
                f"API: {name}\n"
                f"Description: {desc}\n"
                f"Task: Write one short user instruction that would require calling this API in an MDE scenario.\n"
                f"Return only the instruction text."
            )
            try:
                resp = llm.invoke(prompt)
                instruction = getattr(resp, "content", str(resp)).strip()
                llm_calls += 1
            except Exception:
                # Skip this entry on LLM error
                continue

            api_call = {
                "tool_name": name,
                "server_name": server,
                "parameters": {},
            }
            examples.append({
                "instruction": instruction,
                "api_call": api_call,
            })

    return examples

# --- 8) Output creation ---
def create_output_examples(examples: List[Dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(examples, indent=2))

# --- 9) Validation ---
def validate_dataset(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Minimal check: ensure required fields exist
    ok: List[Dict[str, Any]] = []
    for e in examples:
        if not isinstance(e, dict):
            continue
        instr = e.get("instruction")
        call = e.get("api_call", {})
        if instr and isinstance(call, dict) and call.get("tool_name") and call.get("server_name") is not None:
            ok.append(e)
    return ok

# --- 10) Final dataset ---
def write_final_dataset(examples: List[Dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(examples, indent=2))


def main() -> None:
    cfg = {
        "repo": "megamodel://local",
        "k": 3,
        "prompt_path": str(ROOT / "prompts" / "template.txt"),
    "per_api_instructions": 3,
    "llm_max_calls": 5,
    }
    dump = query_megamodel_repo(cfg)
    components = extract_components(dump)

    cap = analyze_capabilities(components)
    pat = discover_patterns(components)
    insights = generate_insights(cap, pat)

    sampled = sample_apis(components, insights)
    selected = select_apis(sampled, k=cfg["k"])

    prompt = Path(cfg["prompt_path"]).read_text() if Path(cfg["prompt_path"]).exists() else "Use API {{API_NAME}}"
    examples = llm_generation(
        selected,
        prompt,
        per_api_instructions=cfg["per_api_instructions"],
        llm_max_calls=cfg["llm_max_calls"],
    )

    create_output_examples(examples, OUT / "examples.json")
    validated = validate_dataset(examples)
    write_final_dataset(validated, OUT / "final_dataset.json")

    print("Done. Outputs in:", OUT)


if __name__ == "__main__":
    main()
