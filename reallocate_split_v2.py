#!/usr/bin/env python3
"""
Phase G.1.1 Split Reallocation - Exhaustive Search for Zero Leakage
Uses simulated annealing to find split with zero >0.85 similarity while balancing distributions
"""

import json
import random
import math
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

# Load existing prompts
with open('C:/CARBON GRID AI/data/evaluation/phase_g/prompts.json') as f:
    data = json.load(f)

all_prompts = data['development_prompts'] + data['locked_test_prompts']
n = len(all_prompts)
texts = [p['prompt_text'] for p in all_prompts]

print(f"Loaded {n} prompts")

# Compute similarity matrix
print("Computing similarity matrix...")
sim_matrix = [[0.0]*n for _ in range(n)]
for i in range(n):
    for j in range(i+1, n):
        sim = SequenceMatcher(None, texts[i].lower(), texts[j].lower()).ratio()
        sim_matrix[i][j] = sim
        sim_matrix[j][i] = sim

# Identify conflicts: pairs with similarity > 0.85
conflicts = []
for i in range(n):
    for j in range(i+1, n):
        if sim_matrix[i][j] > 0.85:
            conflicts.append((i, j, sim_matrix[i][j]))

print(f"Found {len(conflicts)} conflicting pairs (>0.85)")

# Build conflict graph - each node is a prompt, edges connect >0.85 pairs
# We need to 2-color this graph (dev=0, test=1) such that no edge connects same color
# This is a graph bipartitioning problem with size constraints

# For each connected component in conflict graph, all nodes must be same split
# Let's find connected components
adj = [[] for _ in range(n)]
for i, j, _ in conflicts:
    adj[i].append(j)
    adj[j].append(i)

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
comp_sizes = Counter(len(c) for c in components)
print(f"Component sizes: {dict(comp_sizes)}")

# Show multi-node components
for comp in components:
    if len(comp) > 1:
        cats = [all_prompts[i]['category'] for i in comp]
        in_bands = [all_prompts[i]['input_length_band'] for i in comp]
        out_bands = [all_prompts[i]['output_length_band'] for i in comp]
        print(f"  Component size {len(comp)}: cats={cats}, in={in_bands}, out={out_bands}")

# Now we have components that must stay together
# We need to assign each component to dev (0) or test (1)
# Total test = 60

comp_info = []
for comp in components:
    cats = [all_prompts[i]['category'] for i in comp]
    in_bands = [all_prompts[i]['input_length_band'] for i in comp]
    out_bands = [all_prompts[i]['output_length_band'] for i in comp]
    is_mt = [all_prompts[i]['is_multi_turn'] for i in comp]
    comp_info.append({
        'indices': comp,
        'size': len(comp),
        'categories': cats,
        'input_bands': in_bands,
        'output_bands': out_bands,
        'multi_turn': is_mt,
    })

# Targets
TARGET_TEST = 60
category_targets = {cat: 6 for cat in [
    'factual', 'extraction', 'explanation', 'reasoning', 'summarization',
    'coding', 'scientific', 'creative', 'instruction_heavy', 'multi_turn'
]}
input_band_targets = {'very_short': 6, 'short': 12, 'medium': 18, 'long': 15, 'very_long': 9}
output_band_targets = {'very_short': 6, 'short': 12, 'medium': 18, 'long': 15, 'very_long': 9}
multi_turn_target = 6

def evaluate_assignment(assignment):
    """assignment is list of 0/1 for each component"""
    test_indices = []
    for i, a in enumerate(assignment):
        if a == 1:
            test_indices.extend(comp_info[i]['indices'])
    
    if len(test_indices) != TARGET_TEST:
        return None
    
    dev_indices = [i for i in range(n) if i not in test_indices]
    
    # Check cross-split similarity (should be 0 conflicts by construction)
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
        'cat_score': cat_score,
        'in_score': in_score,
        'out_score': out_score,
        'mt_score': mt_score,
        'test_indices': test_indices,
        'dev_indices': dev_indices,
        'test_cats': dict(test_cats),
        'test_in': dict(test_in),
        'test_out': dict(test_out),
        'test_mt': test_mt,
    }

# Simulated annealing
print("\nRunning simulated annealing...")
random.seed(42)

num_components = len(comp_info)
# Start with random valid assignment
best_assignment = None
best_eval = None

def random_valid_assignment():
    """Generate random assignment that sums to 60"""
    assignment = [0]*num_components
    # Sort components by size descending
    comp_order = sorted(range(num_components), key=lambda i: -comp_info[i]['size'])
    remaining = TARGET_TEST
    for i in comp_order:
        if comp_info[i]['size'] <= remaining and random.random() < 0.5:
            assignment[i] = 1
            remaining -= comp_info[i]['size']
    # If not exactly 60, adjust
    test_size = sum(comp_info[i]['size'] for i in range(num_components) if assignment[i])
    if test_size > TARGET_TEST:
        # Remove some
        ones = [i for i in range(num_components) if assignment[i]==1]
        random.shuffle(ones)
        for i in ones:
            if test_size <= TARGET_TEST:
                break
            assignment[i] = 0
            test_size -= comp_info[i]['size']
    elif test_size < TARGET_TEST:
        # Add some
        zeros = [i for i in range(num_components) if assignment[i]==0]
        random.shuffle(zeros)
        for i in zeros:
            if test_size + comp_info[i]['size'] <= TARGET_TEST:
                assignment[i] = 1
                test_size += comp_info[i]['size']
    if test_size == TARGET_TEST:
        return assignment
    return None

# Try to find initial valid assignment
for _ in range(10000):
    a = random_valid_assignment()
    if a:
        eval_result = evaluate_assignment(a)
        if eval_result and eval_result['count_85'] == 0:
            best_assignment = a
            best_eval = eval_result
            print(f"Initial valid zero-leakage found: score={eval_result['score']}")
            break

if best_assignment is None:
    print("Could not find initial valid assignment!")
    # Try harder
    for _ in range(100000):
        a = random_valid_assignment()
        if a:
            eval_result = evaluate_assignment(a)
            if eval_result and eval_result['count_85'] == 0:
                best_assignment = a
                best_eval = eval_result
                print(f"Found after extended search: score={eval_result['score']}")
                break

if best_assignment is None:
    print("NO ZERO-LEAKAGE ASSIGNMENT EXISTS with component constraints!")
    # Show why - check if any component itself forces leakage
    for comp in components:
        if len(comp) > 1:
            # Check internal similarities
            for i in comp:
                for j in comp:
                    if i < j and sim_matrix[i][j] > 0.85:
                        pass  # This is expected
    exit(1)

# Simulated annealing to improve distribution
current_assignment = best_assignment[:]
current_eval = best_eval
temp = 10.0
cooling = 0.9995
min_temp = 0.01

print(f"Starting SA with score={current_eval['score']}, max_sim={current_eval['max_sim']:.4f}")

iteration = 0
while temp > min_temp:
    # Generate neighbor by swapping two components of same size
    # or moving a component from test to dev and another from dev to test
    ones = [i for i in range(num_components) if current_assignment[i]==1]
    zeros = [i for i in range(num_components) if current_assignment[i]==0]
    
    if not ones or not zeros:
        break
    
    # Try to swap a 1 with a 0 of same size to keep total = 60
    random.shuffle(ones)
    random.shuffle(zeros)
    swapped = False
    for i in ones:
        for j in zeros:
            if comp_info[i]['size'] == comp_info[j]['size']:
                new_assignment = current_assignment[:]
                new_assignment[i] = 0
                new_assignment[j] = 1
                eval_result = evaluate_assignment(new_assignment)
                if eval_result and eval_result['count_85'] == 0:
                    delta = eval_result['score'] - current_eval['score']
                    if delta < 0 or random.random() < math.exp(-delta / temp):
                        current_assignment = new_assignment
                        current_eval = eval_result
                        swapped = True
                        if current_eval['score'] < best_eval['score']:
                            best_assignment = current_assignment[:]
                            best_eval = current_eval
                            print(f"  Iter {iteration}: new best score={best_eval['score']}, max_sim={best_eval['max_sim']:.4f}")
                break
        if swapped:
            break
    
    temp *= cooling
    iteration += 1
    
    if iteration % 10000 == 0:
        print(f"  Iter {iteration}: temp={temp:.4f}, best_score={best_eval['score']}")

print(f"\nSA completed after {iteration} iterations")

# Final verification
print("\n=== FINAL VERIFICATION ===")
test_indices = best_eval['test_indices']
dev_indices = best_eval['dev_indices']

print(f"Test size: {len(test_indices)}")
print(f"Dev size: {len(dev_indices)}")
print(f"Max cross-split similarity: {best_eval['max_sim']:.4f}")
print(f"Pairs >0.80: {best_eval['count_80']}")
print(f"Pairs >0.85: {best_eval['count_85']}")
print(f"Pairs >0.90: {best_eval['count_90']}")

print(f"\nCategory distribution: {best_eval['test_cats']}")
print(f"Input bands: {best_eval['test_in']}")
print(f"Output bands: {best_eval['test_out']}")
print(f"Multi-turn: {best_eval['test_mt']}")

# Build final sets
new_dev = [all_prompts[i] for i in dev_indices]
new_test = [all_prompts[i] for i in test_indices]

# Verify no overlap
dev_ids = set(p['prompt_id'] for p in new_dev)
test_ids = set(p['prompt_id'] for p in new_test)
dev_texts_set = set(p['prompt_text'] for p in new_dev)
test_texts_set = set(p['prompt_text'] for p in new_test)
print(f"\nPrompt ID overlap: {len(dev_ids & test_ids)}")
print(f"Prompt text overlap: {len(dev_texts_set & test_texts_set)}")

# Save
output = {
    "metadata": {
        **data['metadata'],
        "dev_count": len(new_dev),
        "test_count": len(new_test),
        "split_method": "conflict_graph_sa_optimized",
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