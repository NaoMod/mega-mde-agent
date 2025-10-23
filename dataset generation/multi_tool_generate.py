
import sys
import json
import asyncio
import random
from pathlib import Path
from typing import List

WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.megamodel import MegamodelRegistry
from scripts.run_agent_versions import populate_registry
from pipeline import generate_dataset_for_regression_testing, _derive_api

TARGET = 500  # Generate full UML multi-tool dataset
OUTPUT_FILE = Path(__file__).parent / "outputs" / "uml_multi_500_dataset.json"

all_instructions: List[dict] = []
generated_count = 0


def load_existing_progress() -> bool:
    global all_instructions, generated_count
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r") as f:
                all_instructions = json.load(f)
            generated_count = len(all_instructions)
            print(f"Resuming from {generated_count} existing multi-tool instructions")
            return True
        except Exception as e:
            print(f"Could not load previous progress: {e}")
    return False


def save_progress() -> None:
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_instructions, f, indent=2)
    pct = (generated_count / TARGET) * 100 if TARGET else 0
    print(f"\nProgress saved: {generated_count}/{TARGET} ({pct:.1f}%) -> {OUTPUT_FILE}")


def build_two_step_workflows(tool_names: List[str]) -> List[List[str]]:
    """Create 4 categories of two-step workflows: get->get, get->apply, apply->get, apply->apply.
    We first classify tool names by derived pattern from _derive_api.
    """
    gets = []
    applies = []
    for name in tool_names:
        api, pat = _derive_api(name)
        if pat == "get":
            gets.append(name)
        elif pat == "apply":
            applies.append(name)

    workflows: List[List[str]] = []
    # Helper to sample pairs (limit size to avoid explosion)
    def pairwise(a, b, limit=150):  # limit keeps total manageable
        pairs = []
        for x in a:
            for y in b:
                pairs.append([x, y])
                if len(pairs) >= limit:
                    return pairs
        return pairs

    workflows += pairwise(gets, gets, limit=120)       # get, get
    workflows += pairwise(gets, applies, limit=120)    # get, apply
    workflows += pairwise(applies, gets, limit=120)    # apply, get
    workflows += pairwise(applies, applies, limit=120) # apply, apply

    random.shuffle(workflows)
    return workflows


async def main():
    global generated_count, all_instructions

    print(f"Generating {TARGET} multi-tool instructions (2-step)...")
    load_existing_progress()
    if generated_count >= TARGET:
        print("Target already satisfied.")
        return

    try:
        registry = MegamodelRegistry()
        await populate_registry(registry)
        atl_tools = registry.tools_by_server.get("atl_server", [])
        # Filter for UML tools: name contains 'UML' and starts with 'apply_' or 'list_transformation_'
        tool_names = [getattr(t, "name", "") for t in atl_tools if getattr(t, "name", "")]
        tool_names = [n for n in tool_names if n not in ("list_transformation_samples_tool", "extract_input_metamodel_name")]
        uml_tool_names = [n for n in tool_names if ("UML" in n) and (n.startswith("apply_") or n.startswith("list_transformation_"))]
        print(f"Discovered {len(uml_tool_names)} UML ATL tools usable for workflows:")
        for ut in uml_tool_names:
            print(f"- {ut}")


        # Guarantee equal usage of each UML tool in multi-tool instructions
        num_tools = len(uml_tool_names)
        per_tool = TARGET // num_tools
        remainder = TARGET % num_tools
        print(f"Each tool will be used in {per_tool} workflows, {remainder} tools will be used in one extra workflow.")

        # Build all possible two-step workflows
        workflows = build_two_step_workflows(uml_tool_names)
        if not workflows:
            print("No workflows built. Exiting.")
            return
        print(f"Prepared {len(workflows)} candidate two-step workflows")

        # Track usage for each tool
        tool_counts = {name: 0 for name in uml_tool_names}
        # Shuffle workflows for variety
        random.shuffle(workflows)

        # Distribute workflows to guarantee equal usage
        selected_workflows = []
        # First, assign per_tool workflows to each tool
        for tool in uml_tool_names:
            count = per_tool + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1
            # Find workflows where this tool appears (either step)
            tool_workflows = [wf for wf in workflows if tool in wf]
            random.shuffle(tool_workflows)
            selected_workflows.extend(tool_workflows[:count])
        # Shuffle again for output randomness
        random.shuffle(selected_workflows)

        # Generate instructions for selected workflows
        for idx, workflow in enumerate(selected_workflows):
            try:
                result = generate_dataset_for_regression_testing(
                    tools=[],
                    workflows=[workflow],
                    per_api=0,
                    per_workflow=1,
                    registry=registry,
                )
                if result:
                    for item in result:
                        instr_text = item.get("instruction", "")
                        if not instr_text:
                            continue
                        if any(existing.get("instruction") == instr_text for existing in all_instructions):
                            continue
                        all_instructions.append(item)
                        generated_count = len(all_instructions)
                        # Update tool usage counts
                        for t in workflow:
                            if t in tool_counts:
                                tool_counts[t] += 1
                        pct = (generated_count / TARGET) * 100
                        preview = instr_text[:70].replace("\n", " ")
                        print(f"{generated_count}/{TARGET} ({pct:.1f}%) - {preview}... [{workflow[0]} & {workflow[1]}: {tool_counts[workflow[0]]}, {tool_counts[workflow[1]]}]")
                        break
                    if generated_count % 10 == 0:
                        save_progress()
                await asyncio.sleep(0.4)
            except Exception as e:
                print(f"Error processing workflow {workflow}: {e}")
                await asyncio.sleep(1.5)

        save_progress()
        print(f"\nDone. Generated {generated_count} multi-tool instructions.")

    except KeyboardInterrupt:
        print(f"\nInterrupted at {generated_count}. Saving...")
        save_progress()
        print("Re-run to resume.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
