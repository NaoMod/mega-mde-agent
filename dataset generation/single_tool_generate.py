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
from scripts.run_agent import populate_registry
from pipeline import generate_dataset_for_regression_testing

# Configuration
TARGET = 500  # 480 existing + 20 more = 500 total
OUTPUT_FILE = Path(__file__).parent / "outputs" / "simple_500_dataset.json"

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
    
    print(f"Generating {TARGET} single-tool instructions...")
    
    # Load existing progress first
    load_existing_progress()
    
    if generated_count >= TARGET:
        print(f"Target already reached! {generated_count} instructions available.")
        return
    
    remaining_needed = TARGET - generated_count
    print(f"Need to generate {remaining_needed} more instructions to reach {TARGET}")
    
    try:
        # Initialize
        registry = MegamodelRegistry()
        await populate_registry(registry)
        
        # Get all tools
        atl_tools = registry.tools_by_server.get("atl_server", [])
        tools = [{"name": getattr(t, "name", ""), "description": getattr(t, "description", "")} 
                for t in atl_tools if getattr(t, "name", "") != "list_transformation_samples_tool"]
        
        print(f"Using {len(tools)} tools")
        
        # Generate instructions one by one to show real progress
        tools_cycle = tools * 5  # Repeat tools list 5 times to get 5 instructions per tool
        random.shuffle(tools_cycle)  # Shuffle for variety
        
        for i, tool in enumerate(tools_cycle):
            if generated_count >= TARGET:
                break
                
            try:
                # Generate 1 instruction for this tool
                instructions = generate_dataset_for_regression_testing(
                    tools=[tool],  # Only one tool at a time
                    workflows=[],  # No workflows, only single tools
                    per_api=1,     # 1 instruction per tool
                    per_workflow=0,
                    registry=registry
                )
                
                if instructions:
                    all_instructions.extend(instructions)
                    generated_count = len(all_instructions)
                    
                    # Print progress immediately
                    percentage = (generated_count / TARGET) * 100
                    print(f"{generated_count}/{TARGET} ({percentage:.1f}%) - {instructions[0]['instruction'][:60]}...")
                    
                    # Save every 10 instructions
                    if generated_count % 10 == 0:
                        save_progress()
                
                # Small delay to avoid rate limits
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