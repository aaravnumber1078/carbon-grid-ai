#!/usr/bin/env python3
"""
Phase G.1.1 Split - Exact Conflict Component Subset Sum
Guarantees zero cross-split >0.85 by keeping components together
"""

import json
import random
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

# Load ORIGINAL prompts (before any reallocation)
with open('C:/CARBON GRID AI/data/evaluation/phase_g/prompts.json') as f:
    data = json.load(f)

# Use the original combined set
all_prompts = data['development_prompts'] + data['locked_test_prompts']
n = len(all_prompts)
texts = [p['prompt_text'] for p in all_prompts]

print(f"Total prompts: {n}")

# Compute FULL similarity matrix
print("Computing similarity matrix...")
sim_matrix = [[0.0]*n for _ in range(n)]
for i in range(n):
    for j in range(i+1, n):
        sim = SequenceMatcher(None, texts[i].lower(), texts[j].lower()).ratio()
        sim_matrix[i][j] = sim
        sim_matrix[j][i] = sim

# Build conflict graph for >0.85
print("Building conflict graph...")
adj = [[] for _ in range(n)]
edge_count = 0
for i in range(n):
    for j in range(i+1, n):
        if sim_matrix[i][j] > 0.85:
            adj[i].append(j)
            adj[j].append(i)
            edge_count += 1

print(f"Conflict edges (>0.85): {edge_count}")

# Find connected components
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

print(f"Conflict components: {len(components)}")
size_dist = Counter(len(c) for c in components)
print(f"Size distribution: {dict(size_dist)}")

# VERIFY: No cross-component >0.85 pairs
cross_85 = 0
for i in range(n):
    for j in range(i+1, n):
        # Find component membership
        comp_i = -1
        comp_j = -1
        for ci, comp in enumerate(components):
            if i in comp:
                comp_i = ci
            if j in comp:
                comp_j = ci
        if comp_i != comp_j and sim_matrix[i][j] > 0.85:
            cross_85 += 1
print(f"Cross-component >0.85 pairs (MUST BE 0): {cross_85}")

# Component metadata
comp_info = []
for comp in components:
    cats = [all_prompts[i]['category'] for i in comp]
    in_bands = [all_prompts[i]['input_length_band'] for i in comp]
    out_bands = [all_prompts[i]['output_length_band'] for i in comp]
    is_mt = [all_prompts[i]['is_multi_turn'] for i in comp]
    comp_info.append({
        'indices': comp,
        'size': len(comp),
        'cats': cats,
        'in_bands': in_bands,
        'out_bands': out_bands,
        'is_mt': is_mt,
        'cat_counts': Counter(cats),
        'in_counts': Counter(in_bands),
        'out_counts': Counter(out_bands),
        'mt_count': sum(is_mt),
    })

TARGET_TEST = 60
category_targets = {cat: 6 for cat in [
    'factual', 'extraction', 'explanation', 'reasoning', 'summarization',
    'coding', 'scientific', 'creative', 'instruction_heavy', 'multi_turn'
]}
input_band_targets = {'very_short': 6, 'short': 12, 'medium': 18, 'long': 15, 'very_long': 9}
output_band_targets = {'very_short': 6, 'short': 12, 'medium': 18, 'long': 15, 'very_long': 9}
multi_turn_target = 6

# Subset sum: find ALL combinations of components summing to exactly 60
# Use DP with bitmasks (only feasible if not too many combinations)
sizes = [c['size'] for c in comp_info]
num_components = len(comp_info)

print(f"\nComponent sizes: {[c['size'] for c in comp_info]}")

# DP: for each possible sum, store the combinations (as bitmasks or lists)
# Since num_components=278, full DP is too large.
# Use meet-in-the-middle
half = num_components // 2
left_comp = comp_info[:half]
right_comp = comp_info[half:]

left_sizes = [c['size'] for c in left_comp]
right_sizes = [c['size'] for c in right_comp]

# Left half: map sum -> list of index lists
left_sums = {0: [[]]}
for idx, s in enumerate(left_sizes):
    new_sums = {}
    for total, masks in left_sums.items():
        # Not take
        if total not in new_sums:
            new_sums[total] = []
        new_sums[total].extend(masks)
        # Take
        new_total = total + s
        if new_total <= TARGET_TEST:
            if new_total not in new_sums:
                new_sums[new_total] = []
            for mask in masks:
                new_sums[new_total].append(mask + [idx])
    left_sums = new_sums

# Right half
right_sums = {0: [[]]}
for idx, s in enumerate(right_sizes):
    new_sums = {}
    for total, masks in right_sums.items():
        if total not in new_sums:
            new_sums[total] = []
        new_sums[total].extend(masks)
        new_total = total + s
        if new_total <= TARGET_TEST:
            if new_total not in new_sums:
                new_sums[new_total] = []
            for mask in masks:
                new_sums[new_total].append(mask + [half + idx])
    right_sums = new_sums

# Find valid combinations
valid_combinations = []
for left_total, left_masks in left_sums.items():
    right_total = TARGET_TEST - left_total
    if right_total in right_sums:
        for lm in left_masks:
            for rm in right_sums[right_total]:
                valid_combinations.append(lm + rm)

print(f"Valid combinations summing to 60: {len(valid_combinations)}")

# Evaluate each combination for balance
def evaluate_combo(combo_indices):
    test_indices = []
    for ci in combo_indices:
        test_indices.extend(comp_info[ci]['indices'])
    
    if len(test_indices) != TARGET_TEST:
        return None
    
    dev_indices = [i for i in range(n) if i not in test_indices]
    
    # By construction, cross-split >0.85 should be 0
    # But verify
    max_sim = 0.0
    count_80 = 0
    count_85 = 0
    count_90 = 0
    for i in dev_indices:
        for j in test_indices:
            sim = sim_matrix[i][j]
            max_sim = max(max_sim, sim)
            if sim > 0.80:
                count_80 += 1
            if sim > 0.85:
                count_85 += 1
            if sim > 0.90:
                count_90 += 1
    
    # Distribution scores
    test_cats = Counter(all_prompts[i]['category'] for i in test_indices)
    cat_score = sum(abs(test_cats.get(c, 0) - 6) for c in category_targets)
    
    test_in = Counter(all_prompts[i]['input_length_band'] for i in test_indices)
    in_score = sum(abs(test_in.get(b, 0) - input_band_targets[b]) for b in input_band_targets)
    
    test_out = Counter(all_prompts[i]['output_length_band'] for i in test_indices)
    out_score = sum(abs(test_out.get(b, 0) - output_band_targets[b]) for b in output_band_targets)
    
    test_mt = sum(1 for i in test_indices if all_prompts[i]['is_multi_turn'])
    mt_score = abs(test_mt - multi_turn_target) * 5
    
    total_score = cat_score + in_score + out_score + mt_score
    
    return {
        'score': total_score,
        'max_sim': max_sim,
        'count_80': count_80,
        'count_85': count_85,
        'count_90': count_90,
        'test_indices': test_indices,
        'dev_indices': dev_indices,
        'test_cats': dict(test_cats),
        'test_in': dict(test_in),
        'test_out': dict(test_out),
        'test_mt': test_mt,
    }

# Evaluate all (should be fast since combinations are limited)
print("Evaluating combinations...")
best_eval = None
best_combo = None

for combo in valid_combinations:
    eval_result = evaluate_combo(combo)
    if eval_result and eval_result['count_85'] == 0:
        if best_eval is None or eval_result['score'] < best_eval['score']:
            best_eval = eval_result
            best_combo = combo

if best_eval is None:
    print("ERROR: No zero-leakage combination found!")
    exit(1)

print(f"\nBEST ZERO-LEAKAGE SPLIT:")
print(f"  Score: {best_eval['score']}")
print(f"  Max cross-split sim: {best_eval['max_sim']:.6f}")
print(f"  >0.80: {best_eval['count_80']}, >0.85: {best_eval['count_85']}, >0.90: {best_eval['count_90']}")
print(f"  Categories: {best_eval['test_cats']}")
print(f"  Input bands: {best_eval['test_in']}")
print(f"  Output bands: {best_eval['test_out']}")
print(f"  Multi-turn: {best_eval['test_mt']}")

# Build final sets
test_indices = best_eval['test_indices']
dev_indices = best_eval['dev_indices']

new_dev = [all_prompts[i] for i in dev_indices]
new_test = [all_prompts[i] for i in test_indices]

# Verify no overlap
dev_ids = set(p['prompt_id'] for p in new_dev)
test_ids = set(p['prompt_id'] for p in new_test)
dev_texts_set = set(p['prompt_text'] for p in new_dev)
test_texts_set = set(p['prompt_text'] for p in new_test)
print(f"\nID overlap: {len(dev_ids & test_ids)}")
print(f"Text overlap: {len(dev_texts_set & test_texts_set)}")
print(f"Dev count: {len(new_dev)}, Test count: {len(new_test)}")

# EXHAUSTIVE FINAL VERIFICATION
print("\n=== EXHAUSTIVE FINAL VERIFICATION ===")
dev_texts_final = [p['prompt_text'] for p in new_dev]
test_texts_final = [p['prompt_text'] for p in new_test]

sims = []
for d in dev_texts_final:
    for t in test_texts_final:
        sims.append(SequenceMatcher(None, d.lower(), t.lower()).ratio())

print(f"Min sim: {min(sims):.6f}")
print(f"Max sim: {max(sims):.6f}")
print(f"Mean sim: {sum(sims)/len(sims):.6f}")
print(f">0.80: {sum(1 for s in sims if s > 0.80)}")
print(f">0.85: {sum(1 for s in sims if s > 0.85)}")
print(f">0.90: {sum(1 for s in sims if s > 0.90)}")

# Save
output = {
    "metadata": {
        **data['metadata'],
        "dev_count": len(new_dev),
        "test_count": len(new_test),
        "split_method": "conflict_component_exact_subset_sum_verified",
        "cross_split_max_similarity": best_eval['max_sim'],
        "cross_split_pairs_above_080": best_eval['count_80'],
        "cross_split_pairs_above_085": best_eval['count_85'],
        "cross_split_pairs_above_090": best_eval['count_90'],
    },
    "development_prompts": new_dev,
    "locked_test_prompts": new_test,
}

output_path = Path("C:/CARBON GRID AI/data/evaluation/phase_g/prompts.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {output_path}")

# Save texts only
prompts_only = {
    "development": [p['prompt_text'] for p in new_dev],
    "locked_test": [p['prompt_text'] for p in new_test],
}
with open(Path("C:/CARBON GRID AI/data/evaluation/phase_g/prompts_text_only.json"), 'w') as f:
    json.dump(prompts_only, f, indent=2)

print("Done.")