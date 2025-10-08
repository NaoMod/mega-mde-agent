
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
from scripts.run_agent import populate_registry
from pipeline import generate_dataset_for_regression_testing, _derive_api

TARGET = 500
OUTPUT_FILE = Path(__file__).parent / "outputs" / "multi_500_dataset.json"

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
        tool_names = [getattr(t, "name", "") for t in atl_tools if getattr(t, "name", "")]
        # Exclude same exclusions as single-tool path does later
        tool_names = [n for n in tool_names if n not in ("list_transformation_samples_tool", "extract_input_metamodel_name")]
        print(f"Discovered {len(tool_names)} ATL tools usable for workflows")

        workflows = build_two_step_workflows(tool_names)
        if not workflows:
            print("No workflows built. Exiting.")
            return
        print(f"Prepared {len(workflows)} candidate two-step workflows")

        # We'll cycle over workflows repeatedly until we hit target
        wf_index = 0
        attempts = 0

        while generated_count < TARGET:
            if wf_index >= len(workflows):
                random.shuffle(workflows)
                wf_index = 0
            workflow = workflows[wf_index]
            wf_index += 1
            attempts += 1

            try:
                # Direct call: no single-tool generation (per_api=0) only this workflow (per_workflow=1)
                result = generate_dataset_for_regression_testing(
                    tools=[],            # Skip single-tool generation
                    workflows=[workflow],
                    per_api=0,
                    per_workflow=1,
                    registry=registry,
                )

                if result:
                    # Multi-tool path may return >1 if validation duplicates; keep only first new item
                    for item in result:
                        # Deduplicate by instruction text
                        instr_text = item.get("instruction", "")
                        if not instr_text:
                            continue
                        # Prevent duplicate instructions
                        if any(existing.get("instruction") == instr_text for existing in all_instructions):
                            continue
                        all_instructions.append(item)
                        generated_count = len(all_instructions)
                        pct = (generated_count / TARGET) * 100
                        preview = instr_text[:70].replace("\n", " ")
                        print(f"{generated_count}/{TARGET} ({pct:.1f}%) - {preview}...")
                        break  # Only count one per workflow iteration

                    if generated_count % 10 == 0:
                        save_progress()
                else:
                    # Diagnostic print (rare, keep short)
                    if attempts % 25 == 0:
                        print(f"Still no output after {attempts} attempts; last workflow: {workflow}")

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
