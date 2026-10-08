#!/usr/bin/env python3
"""
Phase G.1.1 Split Reallocation - Fast Random Search for Zero Leakage
"""

import json
import random
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

# Load existing prompts
with open('C:/CARBON GRID AI/data/evaluation/phase_g/prompts.json') as f:
    data = json.load(f)

all_prompts = data['development_prompts'] + data['locked_test_prompts']
n = len(all_prompts)
texts = [p['prompt_text'] for p in all_prompts]

# Compute similarity matrix
print("Computing similarity matrix...")
sim_matrix = [[0.0]*n for _ in range(n)]
for i in range(n):
    for j in range(i+1, n):
        sim = SequenceMatcher(None, texts[i].lower(), texts[j].lower()).ratio()
        sim_matrix[i][j] = sim
        sim_matrix[j][i] = sim

# Build conflict graph for >0.85
adj = [[] for _ in range(n)]
for i in range(n):
    for j in range(i+1, n):
        if sim_matrix[i][j] > 0.85:
            adj[i].append(j)
            adj[j].append(i)

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

print(f"Found {len(components)} conflict components")

# Component info
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
    })

TARGET_TEST = 60
category_targets = {cat: 6 for cat in [
    'factual', 'extraction', 'explanation', 'reasoning', 'summarization',
    'coding', 'scientific', 'creative', 'instruction_heavy', 'multi_turn'
]}
input_band_targets = {'very_short': 6, 'short': 12, 'medium': 18, 'long': 15, 'very_long': 9}
output_band_targets = {'very_short': 6, 'short': 12, 'medium': 18, 'long': 15, 'very_long': 9}
multi_turn_target = 6

# Fast evaluation
def eval_test_indices(test_indices):
    dev_indices = [i for i in range(n) if i not in test_indices]
    
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

# Pre-filter: only components of size <= 60
valid_components = [(i, c) for i, c in enumerate(comp_info) if c['size'] <= TARGET_TEST]
print(f"Valid components (<=60): {len(valid_components)}")

# Component sizes
sizes = [c['size'] for _, c in valid_components]
print(f"Component sizes: {Counter(sizes)}")

# Check if subset sum to 60 is even possible
# Simple DP for feasibility
possible = {0}
for s in sizes:
    new_possible = set(possible)
    for p in possible:
        if p + s <= TARGET_TEST:
            new_possible.add(p + s)
    possible = new_possible

print(f"Subset sum 60 possible: {TARGET_TEST in possible}")

# Random search with component-level sampling
best_eval = None
random.seed(42)

# We need to sample combinations of components that sum to exactly 60
# Use rejection sampling: randomly pick components until sum >= 60, then adjust
for attempt in range(50000):
    if attempt % 5000 == 0:
        print(f"  Attempt {attempt}, best score: {best_eval['score'] if best_eval else 'None'}")
    
    # Randomly shuffle components
    shuffled = valid_components.copy()
    random.shuffle(shuffled)
    
    test_indices = []
    test_size = 0
    
    for ci, c in shuffled:
        if test_size + c['size'] <= TARGET_TEST:
            test_indices.extend(c['indices'])
            test_size += c['size']
        if test_size == TARGET_TEST:
            break
    
    if test_size != TARGET_TEST:
        continue
    
    eval_result = eval_test_indices(test_indices)
    
    if eval_result['count_85'] == 0:
        if best_eval is None or eval_result['score'] < best_eval['score']:
            best_eval = eval_result
            print(f"  New best at attempt {attempt}: score={eval_result['score']}, max_sim={eval_result['max_sim']:.4f}, >0.80={eval_result['count_80']}")
            if eval_result['score'] <= 10:  # Good enough balance
                print("  Good balance achieved, stopping early")
                break

if best_eval is None:
    print("NO ZERO-LEAKAGE FOUND in random search")
    # Try with relaxed constraint: minimize >0.85
    best_eval = None
    for attempt in range(10000):
        shuffled = valid_components.copy()
        random.shuffle(shuffled)
        test_indices = []
        test_size = 0
        for ci, c in shuffled:
            if test_size + c['size'] <= TARGET_TEST:
                test_indices.extend(c['indices'])
                test_size += c['size']
            if test_size == TARGET_TEST:
                break
        if test_size != TARGET_TEST:
            continue
        eval_result = eval_test_indices(test_indices)
        if best_eval is None or eval_result['count_85'] < best_eval['count_85'] or \
           (eval_result['count_85'] == best_eval['count_85'] and eval_result['score'] < best_eval['score']):
            best_eval = eval_result
    print(f"Best with leakage: >0.85={best_eval['count_85']}, score={best_eval['score']}")
else:
    print(f"\nSUCCESS: Zero-leakage split found!")
    print(f"Score: {best_eval['score']}")
    print(f"Max sim: {best_eval['max_sim']:.4f}")
    print(f">0.80: {best_eval['count_80']}, >0.85: {best_eval['count_85']}, >0.90: {best_eval['count_90']}")

if best_eval:
    test_indices = best_eval['test_indices']
    dev_indices = best_eval['dev_indices']
    
    new_dev = [all_prompts[i] for i in dev_indices]
    new_test = [all_prompts[i] for i in test_indices]
    
    # Verify
    dev_ids = set(p['prompt_id'] for p in new_dev)
    test_ids = set(p['prompt_id'] for p in new_test)
    dev_texts_set = set(p['prompt_text'] for p in new_dev)
    test_texts_set = set(p['prompt_text'] for p in new_test)
    print(f"ID overlap: {len(dev_ids & test_ids)}")
    print(f"Text overlap: {len(dev_texts_set & test_texts_set)}")
    
    print(f"\nTest category: {best_eval['test_cats']}")
    print(f"Test input bands: {best_eval['test_in']}")
    print(f"Test output bands: {best_eval['test_out']}")
    print(f"Test multi-turn: {best_eval['test_mt']}")
    
    # Save
    output = {
        "metadata": {
            **data['metadata'],
            "dev_count": len(new_dev),
            "test_count": len(new_test),
            "split_method": "conflict_component_random_search",
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
    
    prompts_only = {
        "development": [p['prompt_text'] for p in new_dev],
        "locked_test": [p['prompt_text'] for p in new_test],
    }
    with open(Path("C:/CARBON GRID AI/data/evaluation/phase_g/prompts_text_only.json"), 'w') as f:
        json.dump(prompts_only, f, indent=2)
    
    print("Saved.")
else:
    print("FAILED: Could not find any valid split")