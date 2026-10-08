#!/usr/bin/env python3
"""
Select 25 stratified prompts from Phase G development set for pilot measurement.
Stratified by category and input/output length bands.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Fixed seed for reproducibility
random.seed(42)

# Load prompts
with open(Path("C:/CARBON GRID AI/data/evaluation/phase_g/prompts.json")) as f:
    data = json.load(f)

dev_prompts = data["development_prompts"]

# Build strata by (category, input_band, output_band)
strata = defaultdict(list)
for p in dev_prompts:
    key = (p["category"], p["input_length_band"], p["output_length_band"])
    strata[key].append(p)

print(f"Total strata: {len(strata)}")
print(f"Total dev prompts: {len(dev_prompts)}")

# Target: 25 prompts
# Allocate proportionally but ensure at least 1 per category if possible
categories = set(p["category"] for p in dev_prompts)
category_counts = defaultdict(int)

# Greedy allocation: ensure category coverage first
stratum_items = [(k, v) for k, v in strata.items()]
random.shuffle(stratum_items)

selected = []
category_counts = defaultdict(int)
input_band_counts = defaultdict(int)
output_band_counts = defaultdict(int)

# First pass: pick 2-3 per category to ensure coverage
categories = sorted(set(p["category"] for p in dev_prompts))
for cat in categories:
    cat_strata = [(k, v) for k, v in stratum_items if k[0] == cat]
    if not cat_strata:
        continue
    # Pick from 1-2 strata for this category
    for key, prompts in cat_strata[:2]:
        if len(selected) >= 25:
            break
        chosen = random.choice(prompts)
        if chosen not in selected:
            selected.append(chosen)
            category_counts[chosen["category"]] += 1
            input_band_counts[chosen["input_length_band"]] += 1
            output_band_counts[chosen["output_length_band"]] += 1
    if len(selected) >= 25:
        break

# Second pass: fill remaining slots proportionally
total = len(dev_prompts)
for key, prompts in stratum_items:
    if len(selected) >= 25:
        break
    prop = len(prompts) / total
    ideal = prop * 25
    if ideal >= 1:
        alloc = min(int(ideal), len(prompts))
        for _ in range(alloc):
            if len(selected) >= 25:
                break
            available = [p for p in prompts if p not in selected]
            if available:
                chosen = random.choice(available)
                selected.append(chosen)
                category_counts[chosen["category"]] += 1
                input_band_counts[chosen["input_length_band"]] += 1
                output_band_counts[chosen["output_length_band"]] += 1

# If still under, add from any
while len(selected) < 25:
    all_remaining = [p for prompts in strata.values() for p in prompts if p not in selected]
    if not all_remaining:
        break
    chosen = random.choice(all_remaining)
    selected.append(chosen)
    category_counts[chosen["category"]] += 1
    input_band_counts[chosen["input_length_band"]] += 1
    output_band_counts[chosen["output_length_band"]] += 1

print(f"\nSelected {len(selected)} prompts")
print("\nCategory distribution:")
for cat in sorted(categories):
    print(f"  {cat}: {category_counts.get(cat, 0)}")

print("\nInput band distribution:")
for band in sorted(input_band_counts):
    print(f"  {band}: {input_band_counts[band]}")

print("\nOutput band distribution:")
for band in sorted(output_band_counts):
    print(f"  {band}: {output_band_counts[band]}")

multi_turn = sum(1 for p in selected if p["is_multi_turn"])
print(f"\nMulti-turn: {multi_turn}")

# Save pilot prompts
pilot_data = {
    "metadata": {
        "phase": "G.1.2_pilot",
        "selection_method": "stratified_random_seed42",
        "total_prompts": len(selected),
        "repetitions_per_config": 3,
        "total_runs": len(selected) * 2 * 3,
        "source": "data/evaluation/phase_g/prompts.json",
        "source_dev_count": len(dev_prompts),
    },
    "pilot_prompts": selected,
}

output_path = Path("C:/CARBON GRID AI/data/evaluation/phase_g/pilot_prompts.json")
with open(output_path, "w") as f:
    json.dump(pilot_data, f, indent=2)

print(f"\nSaved to {output_path}")

# Also save just the prompt texts and IDs for quick reference
for p in selected:
    print(f"  {p['prompt_id']} | {p['category']} | in:{p['input_length_band']} out:{p['output_length_band']} | mt:{p['is_multi_turn']} | {p['prompt_text'][:80]}...")