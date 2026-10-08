#!/usr/bin/env python3
"""
Phase G.1.1 Split - Single Conservative Attempt
Uses threshold 0.84 for conflict graph to be safe, then verifies exhaustively at 0.85
"""

import json
import random
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

# Load ORIGINAL prompts (first version)
# We need to regenerate from scratch to get original
# Actually, let's load the current prompts.json which has the prompts
with open('C:/CARBON GRID AI/data/evaluation/phase_g/prompts.json') as f:
    data = json.load(f)

all_prompts = data['development_prompts'] + data['locked_test_prompts']
n = len(all_prompts)
texts = [p['prompt_text'] for p in all_prompts]

# Single similarity function
def sim(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Build conflict graph with CONSERVATIVE threshold 0.84
print("Building conflict graph at threshold 0.84...")
adj = [[] for _ in range(n)]
for i in range(n):
    ti = texts[i]
    for j in range(i+1, n):
        if sim(ti, texts[j]) > 0.84:  # CONSERVATIVE
            adj[i].append(j)
            adj[j].append(i)

# Find components
visited = [False]*n
components = []
for i in range(n):
    if not visited[i]:
        comp = []
        stack = [i]
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        components.append(comp)

print(f"Components at 0.84: {len(components)}")

# Component info
comp_info = []
for comp in components:
    comp_info.append({
        'indices': comp,
        'size': len(comp),
        'cats': Counter(all_prompts[i]['category'] for i in comp),
        'in_bands': Counter(all_prompts[i]['input_length_band'] for i in comp),
        'out_bands': Counter(all_prompts[i]['output_length_band'] for i in comp),
        'mt_count': sum(1 for i in comp if all_prompts[i]['is_multi_turn']),
    })

TARGET_TEST = 60
category_targets = {cat: 6 for cat in [
    'factual', 'extraction', 'explanation', 'reasoning', 'summarization',
    'coding', 'scientific', 'creative', 'instruction_heavy', 'multi_turn'
]}
input_band_targets = {'very_short': 6, 'short': 12, 'medium': 18, 'long': 15, 'very_long': 9}
output_band_targets = {'very_short': 6, 'short': 12, 'medium': 18, 'long': 15, 'very_long': 9}
multi_turn_target = 6

# Simple greedy: pick largest components that help balance
remaining = TARGET_TEST
selected = set()
current_cats = Counter()
current_in = Counter()
current_out = Counter()
current_mt = 0

candidates = sorted(range(len(comp_info)), key=lambda i: -comp_info[i]['size'])

for ci in candidates:
    c = comp_info[ci]
    if c['size'] > remaining:
        continue
    score = 0
    for cat, cnt in c['cats'].items():
        deficit = max(0, category_targets[cat] - current_cats[cat])
        score += min(cnt, deficit)
    for band, cnt in c['in_bands'].items():
        deficit = max(0, input_band_targets[band] - current_in[band])
        score += min(cnt, deficit)
    for band, cnt in c['out_bands'].items():
        deficit = max(0, output_band_targets[band] - current_out[band])
        score += min(cnt, deficit)
    mt_deficit = max(0, multi_turn_target - current_mt)
    score += min(c['mt_count'], mt_deficit) * 2
    
    if score > 0 or remaining <= 10:
        selected.add(ci)
        remaining -= c['size']
        current_cats += c['cats']
        current_in += c['in_bands']
        current_out += c['out_bands']
        current_mt += c['mt_count']
        if remaining == 0:
            break

if remaining > 0:
    size1 = [ci for ci in range(len(comp_info)) if comp_info[ci]['size'] == 1 and ci not in selected]
    random.shuffle(size1)
    for ci in size1:
        if remaining == 0:
            break
        selected.add(ci)
        remaining -= 1
        current_cats += comp_info[ci]['cats']
        current_in += comp_info[ci]['in_bands']
        current_out += comp_info[ci]['out_bands']
        current_mt += comp_info[ci]['mt_count']

if remaining != 0:
    print("Failed to reach exactly 60")
    exit(1)

test_indices = []
for ci in selected:
    test_indices.extend(comp_info[ci]['indices'])

# EXHAUSTIVE VERIFICATION at threshold 0.85
print(f"Test size: {len(test_indices)}")
dev_indices = [i for i in range(n) if i not in test_indices]

max_sim = 0.0
count_80 = 0
count_85 = 0
count_90 = 0
for i in dev_indices:
    ti = texts[i]
    for j in test_indices:
        s = sim(ti, texts[j])
        if s > max_sim:
            max_sim = s
        if s > 0.80:
            count_80 += 1
        if s > 0.85:
            count_85 += 1
        if s > 0.90:
            count_90 += 1

print(f"Max sim: {max_sim:.6f}")
print(f">0.80: {count_80}, >0.85: {count_85}, >0.90: {count_90}")

if count_85 == 0:
    print("SUCCESS at 0.85!")
    new_dev = [all_prompts[i] for i in dev_indices]
    new_test = [all_prompts[i] for i in test_indices]
    
    output = {
        "metadata": {
            **data['metadata'],
            "dev_count": len(new_dev),
            "test_count": len(new_test),
            "split_method": "conservative_threshold_0.84_verified_at_0.85",
            "cross_split_max_similarity": max_sim,
            "cross_split_pairs_above_080": count_80,
            "cross_split_pairs_above_085": count_85,
            "cross_split_pairs_above_090": count_90,
        },
        "development_prompts": new_dev,
        "locked_test_prompts": new_test,
    }
    
    output_path = Path("C:/CARBON GRID AI/data/evaluation/phase_g/prompts.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    prompts_only = {
        "development": [p['prompt_text'] for p in new_dev],
        "locked_test": [p['prompt_text'] for p in new_test],
    }
    with open(Path("C:/CARBON GRID AI/data/evaluation/phase_g/prompts_text_only.json"), 'w') as f:
        json.dump(prompts_only, f, indent=2)
    
    print("Saved.")
else:
    print(f"FAILED: {count_85} pairs >0.85")
    print("CONFLICT: Zero >0.85 cannot be achieved with conservative 0.84 conflict graph")
    print("Reporting conflict as requested.")
    
    # Save the best attempt anyway for review
    new_dev = [all_prompts[i] for i in dev_indices]
    new_test = [all_prompts[i] for i in test_indices]
    
    output = {
        "metadata": {
            **data['metadata'],
            "dev_count": len(new_dev),
            "test_count": len(new_test),
            "split_method": "conservative_threshold_0.84_FAILED_at_0.85",
            "cross_split_max_similarity": max_sim,
            "cross_split_pairs_above_080": count_80,
            "cross_split_pairs_above_085": count_85,
            "cross_split_pairs_above_090": count_90,
            "conflict_reported": True,
        },
        "development_prompts": new_dev,
        "locked_test_prompts": new_test,
    }
    
    output_path = Path("C:/CARBON GRID AI/data/evaluation/phase_g/prompts.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Saved failed attempt for review.")