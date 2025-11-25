"""This script generates a multi-tool dataset by creating
two-step workflows using various tools. It ensures balanced usage
of each tool across the generated instructions. The dataset is saved incrementally
to allow resumption in case of interruptions."""
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

TARGET = 500 # Generate full Table multi-tool dataset
OUTPUT_FILE = Path(__file__).parent / "outputs" / "openRewrite_multi_500_dataset.json"
REMAINDER_FILE = Path(__file__).parent / "outputs" / "openRewrite_multi_remainder.json"

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

def save_remainder(remainder_instructions: List[dict]) -> None:
    REMAINDER_FILE.parent.mkdir(exist_ok=True)
    with open(REMAINDER_FILE, "w") as f:
        json.dump(remainder_instructions, f, indent=2)
    print(f"\nRemainder saved: {len(remainder_instructions)} -> {REMAINDER_FILE}")


def build_two_step_workflows(tool_names: List[str]) -> List[List[str]]:
    """Create 4 categories of two-step workflows: get->get, get->apply, apply->get, apply->apply.
    We first classify tool names by derived pattern from _derive_api.
    """
    recipe_apps = []
    recipe_infos = []
    for name in tool_names:
        api, pat = _derive_api(name)
        if pat == "recipe_application":
            recipe_apps.append(name)
        elif pat == "recipe_info":
            recipe_infos.append(name)

    workflows: List[List[str]] = []
    # Helper to sample pairs (limit size to avoid explosion)
    def pairwise(a, b, limit=150):
        pairs = []
        for x in a:
            for y in b:
                pairs.append([x, y])
                if len(pairs) >= limit:
                    return pairs
        return pairs

    workflows += pairwise(recipe_apps, recipe_apps, limit=120)       # recipe_application, recipe_application
    workflows += pairwise(recipe_apps, recipe_infos, limit=120)      # recipe_application, recipe_info
    workflows += pairwise(recipe_infos, recipe_apps, limit=120)      # recipe_info, recipe_application
    workflows += pairwise(recipe_infos, recipe_infos, limit=120)     # recipe_info, recipe_info

    random.shuffle(workflows)
    return workflows


async def main():
    global generated_count, all_instructions

    print(f"Generating {TARGET} multi-tool instructions (2-step)...")
    load_existing_progress()
    if generated_count >= TARGET:
        print("Target already satisfied.")
        return

    # Only generate the remainder if previous instructions exist
    remainder_needed = TARGET - generated_count
    if remainder_needed <= 0:
        print("No remainder needed.")
        return
    print(f"Generating remainder: {remainder_needed} instructions to reach {TARGET}")


    # Discover OpenRewrite tools
    registry = MegamodelRegistry()
    await populate_registry(registry)
    or_tools = registry.tools_by_server.get("openrewrite_server", [])
    tool_names = [getattr(t, "name", "") for t in or_tools if getattr(t, "name", "")]
    # Optionally filter out meta/utility tools (minimal change: keep all except list_/extract_)
    tool_names = [n for n in tool_names if not (n.startswith("list_") or n.startswith("extract_"))]
    print(f"Discovered {len(tool_names)} OpenRewrite tools usable for workflows:")

    for tt in tool_names:
        print(f"- {tt}")

    # Workflow generation
    num_tools = len(tool_names)
    per_tool = TARGET // num_tools
    remainder = TARGET % num_tools
    print(f"Each tool will be used in {per_tool} workflows, {remainder} tools will be used in one extra workflow.")

    # Build all possible two-step workflows
    workflows = build_two_step_workflows(tool_names)
    if not workflows:
        print("No workflows built. Exiting.")
        return
    print(f"Prepared {len(workflows)} candidate two-step workflows")

    # Track usage for each tool
    tool_counts = {name: 0 for name in tool_names}
    # Shuffle workflows for variety
    random.shuffle(workflows)

    # Distribute workflows to guarantee equal usage
    selected_workflows = []
    for tool in tool_names:
        count = per_tool + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        tool_workflows = [wf for wf in workflows if tool in wf]
        random.shuffle(tool_workflows)
        selected_workflows.extend(tool_workflows[:count])
    random.shuffle(selected_workflows)

    # Load existing remainder progress if available
    remainder_instructions = []
    import os
    if os.path.exists(str(REMAINDER_FILE)):
        try:
            with open(REMAINDER_FILE, "r") as f:
                remainder_instructions = json.load(f)
            print(f"Loaded {len(remainder_instructions)} existing remainder instructions from {REMAINDER_FILE}")
        except Exception as e:
            print(f"Could not load previous remainder progress: {e}")

    # Continue generating until TARGET is reached, allowing repeated workflows
    workflow_idx = 0
    while len(remainder_instructions) < remainder_needed:
        workflow = selected_workflows[workflow_idx % len(selected_workflows)]
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
                    # Allow repeated instructions in remainder_instructions
                    remainder_instructions.append(item)
                    for t in workflow:
                        if t in tool_counts:
                            tool_counts[t] += 1
                    pct = ((generated_count + len(remainder_instructions)) / TARGET) * 100
                    preview = instr_text[:70].replace("\n", " ")
                    print(f"{generated_count + len(remainder_instructions)}/{TARGET} ({pct:.1f}%) - {preview}... [{workflow[0]} & {workflow[1]}: {tool_counts[workflow[0]]}, {tool_counts[workflow[1]]}]")
                    break
                if (generated_count + len(remainder_instructions)) % 10 == 0:
                    save_remainder(remainder_instructions)
            await asyncio.sleep(0.4)
        except Exception as e:
            print(f"Error processing workflow {workflow}: {e}")
            await asyncio.sleep(1.5)
        workflow_idx += 1

    save_remainder(remainder_instructions)
    print(f"\nDone. Generated {len(remainder_instructions)} Table multi-tool instructions.")
    return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
