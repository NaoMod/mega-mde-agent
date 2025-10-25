#!/usr/bin/env python3
"""
script to generate 500 single-tool instructions
"""

import sys
import json
import asyncio
import random
from pathlib import Path

# Add paths
WORKDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKDIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.megamodel import MegamodelRegistry
from scripts.run_agent_versions import populate_registry
from pipeline import generate_dataset_for_regression_testing

TARGET = 500  # Generate full Table dataset
OUTPUT_FILE = Path(__file__).parent / "outputs" / "table_500_dataset.json"

# Global variables
all_instructions = []
generated_count = 0

def load_existing_progress():
    """Load existing progress if any"""
    global all_instructions, generated_count
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, 'r') as f:
                all_instructions = json.load(f)
            generated_count = len(all_instructions)
            print(f"Resuming from existing progress: {generated_count} instructions already generated")
            return True
        except Exception as e:
            print(f"Could not load existing progress: {e}")
    return False

def save_progress():
    """Save current progress"""
    global all_instructions, generated_count
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_instructions, f, indent=2)
    print(f"\nProgress saved: {generated_count}/{TARGET} ({generated_count/TARGET*100:.1f}%) to {OUTPUT_FILE}")

async def main():
    global all_instructions, generated_count
    
    print(f"Generating {TARGET} single-tool instructions for Table transformations...")

    # Always start fresh: clear previous instructions and progress
    all_instructions = []
    generated_count = 0

    remaining_needed = TARGET - generated_count
    print(f"Need to generate {remaining_needed} more instructions to reach {TARGET}")

    try:
        # Initialize
        registry = MegamodelRegistry()
        await populate_registry(registry)

        # Get all tools, filter for Table input metamodels (name contains 'Table', case-insensitive)
        atl_tools = registry.tools_by_server.get("atl_server", [])
        table_tools = []
        for t in atl_tools:
            tool_name = getattr(t, "name", "")
            if tool_name == "list_transformation_samples_tool":
                continue
            # Select tools whose name contains 'Table' and starts with 'apply_' or 'list_transformation_'
            if ("Table" in tool_name) and (tool_name.startswith("apply_") or tool_name.startswith("list_transformation_")):
                table_tools.append({"name": tool_name, "description": getattr(t, "description", "")})
        print(f"Using {len(table_tools)} Table tools:")
        for tt in table_tools:
            print(f"- {tt['name']}")
        # Guarantee equal usage of each Table tool
        num_tools = len(table_tools)
        per_tool = TARGET // num_tools if num_tools > 0 else 0
        remainder = TARGET % num_tools if num_tools > 0 else 0
        print(f"Each tool will be used {per_tool} times, {remainder} tools will be used one extra time.")

        tool_counts = {tool['name']: 0 for tool in table_tools}
        tool_order = table_tools.copy()
        random.shuffle(tool_order)

        for idx, tool in enumerate(tool_order):
            extra = 1 if idx < remainder else 0
            count = per_tool + extra
            for _ in range(count):
                try:
                    instructions = generate_dataset_for_regression_testing(
                        tools=[tool],
                        workflows=[],
                        per_api=1,
                        per_workflow=0,
                        registry=registry
                    )
                    if instructions:
                        all_instructions.extend(instructions)
                        generated_count = len(all_instructions)
                        tool_counts[tool['name']] += 1
                        percentage = (generated_count / TARGET) * 100
                        print(f"{generated_count}/{TARGET} ({percentage:.1f}%) - {instructions[0]['instruction'][:60]}... [{tool['name']}: {tool_counts[tool['name']]}]")
                        if generated_count % 10 == 0:
                            save_progress()
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"Error generating instruction for {tool.get('name', 'unknown')}: {e}")
                    await asyncio.sleep(2)

        # Final save
        save_progress()
        print(f"\nCompleted! Generated {generated_count} instructions")
        
    except KeyboardInterrupt:
        print(f"\n\nStopped by user at {generated_count} instructions")
        save_progress()
        print("You can resume by running the script again (it will continue from where it left off)")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass