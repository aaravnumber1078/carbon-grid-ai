#!/usr/bin/env python3
"""
Phase G.1.1 Split Reallocation - Optimize for Zero Template Leakage
Reallocates existing 400 prompts between dev/test to achieve zero similarity >0.85
"""

import json
import random
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import itertools

# Load existing prompts
with open('C:/CARBON GRID AI/data/evaluation/phase_g/prompts.json') as f:
    data = json.load(f)

all_prompts = data['development_prompts'] + data['locked_test_prompts']
print(f"Loaded {len(all_prompts)} prompts")

# Compute similarity matrix between all pairs
def compute_similarity(text1, text2):
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

# Build similarity matrix (only need upper triangle for cross-split)
n = len(all_prompts)
texts = [p['prompt_text'] for p in all_prompts]

print("Computing full similarity matrix...")
sim_matrix = [[0.0]*n for _ in range(n)]
for i in range(n):
    for j in range(i+1, n):
        sim = compute_similarity(texts[i], texts[j])
        sim_matrix[i][j] = sim
        sim_matrix[j][i] = sim

print("Similarity matrix computed.")

# Identify template families by clustering high-similarity pairs
# Two prompts are in same family if similarity > 0.85
families = []
assigned = [False]*n

for i in range(n):
    if assigned[i]:
        continue
    family = [i]
    assigned[i] = True
    for j in range(i+1, n):
        if not assigned[j] and sim_matrix[i][j] > 0.85:
            family.append(j)
            assigned[j] = True
    families.append(family)

print(f"Found {len(families)} template families")
family_sizes = Counter(len(f) for f in families)
print(f"Family size distribution: {dict(family_sizes)}")

# Show families with >1 member
for idx, fam in enumerate(families):
    if len(fam) > 1:
        print(f"  Family {idx}: {len(fam)} prompts")
        for p_idx in fam[:3]:
            print(f"    [{all_prompts[p_idx]['category']}/{all_prompts[p_idx]['input_length_band']}/{all_prompts[p_idx]['output_length_band']}] {all_prompts[p_idx]['prompt_text'][:80]}...")
        if len(fam) > 3:
            print(f"    ... and {len(fam)-3} more")

# Now we need to assign each family entirely to either dev or test
# This ensures template variants stay together
# We want: 340 dev, 60 test, balanced categories/length bands

# Prepare family metadata
family_info = []
for fam in families:
    cats = [all_prompts[i]['category'] for i in fam]
    in_bands = [all_prompts[i]['input_length_band'] for i in fam]
    out_bands = [all_prompts[i]['output_length_band'] for i in fam]
    is_mt = [all_prompts[i]['is_multi_turn'] for i in fam]
    family_info.append({
        'indices': fam,
        'size': len(fam),
        'categories': cats,
        'input_bands': in_bands,
        'output_bands': out_bands,
        'multi_turn': is_mt,
        'primary_category': Counter(cats).most_common(1)[0][0],
        'primary_in_band': Counter(in_bands).most_common(1)[0][0],
        'primary_out_band': Counter(out_bands).most_common(1)[0][0],
    })

# Target counts
TARGET_TEST = 60
TARGET_DEV = 340

category_targets = {cat: 6 for cat in [
    'factual', 'extraction', 'explanation', 'reasoning', 'summarization',
    'coding', 'scientific', 'creative', 'instruction_heavy', 'multi_turn'
]}
input_band_targets = {'very_short': 6, 'short': 12, 'medium': 18, 'long': 15, 'very_long': 9}
output_band_targets = {'very_short': 6, 'short': 12, 'medium': 18, 'long': 15, 'very_long': 9}
multi_turn_target = 6  # 15% of 40

# Greedy allocation with backtracking
best_split = None
best_score = float('inf')

def evaluate_split(test_families):
    """Evaluate a split - return score (lower is better)"""
    test_indices = []
    for fam in test_families:
        test_indices.extend(fam['indices'])
    
    if len(test_indices) != TARGET_TEST:
        return float('inf')
    
    dev_indices = [i for i in range(n) if i not in test_indices]
    
    # Check cross-split similarity
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
    
    # Category balance
    test_cats = Counter(all_prompts[i]['category'] for i in test_indices)
    cat_score = sum(abs(test_cats.get(c, 0) - 6) for c in category_targets)
    
    # Input band balance
    test_in = Counter(all_prompts[i]['input_length_band'] for i in test_indices)
    in_score = sum(abs(test_in.get(b, 0) - input_band_targets[b]) for b in input_band_targets)
    
    # Output band balance
    test_out = Counter(all_prompts[i]['output_length_band'] for i in test_indices)
    out_score = sum(abs(test_out.get(b, 0) - output_band_targets[b]) for b in output_band_targets)
    
    # Multi-turn balance
    test_mt = sum(1 for i in test_indices if all_prompts[i]['is_multi_turn'])
    mt_score = abs(test_mt - multi_turn_target) * 10
    
    # Heavy penalty for any >0.85 similarity
    leakage_penalty = count_85 * 10000 + count_90 * 100000
    
    total_score = leakage_penalty + cat_score + in_score + out_score + mt_score
    
    return total_score, {
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
    }

# Try random allocations with family-level assignment
print("\nSearching for optimal split...")
random.seed(42)

for attempt in range(10000):
    # Randomly shuffle families
    shuffled = family_info.copy()
    random.shuffle(shuffled)
    
    test_families = []
    test_size = 0
    
    for fam in shuffled:
        if test_size + fam['size'] <= TARGET_TEST:
            test_families.append(fam)
            test_size += fam['size']
        if test_size == TARGET_TEST:
            break
    
    if test_size == TARGET_TEST:
        score, details = evaluate_split(test_families)
        if score < best_score:
            best_score = score
            best_split = (test_families, details)
            if attempt % 100 == 0:
                print(f"  Attempt {attempt}: score={score}, max_sim={details['max_sim']:.4f}, >0.85={details['count_85']}, >0.90={details['count_90']}")
            
            if details['count_85'] == 0:
                print(f"  FOUND ZERO LEAKAGE at attempt {attempt}!")
                break

if best_split is None:
    print("NO VALID SPLIT FOUND")
    exit(1)

test_families, details = best_split
test_indices = details['test_indices']
dev_indices = details['dev_indices']

print(f"\n=== BEST SPLIT FOUND ===")
print(f"Score: {best_score}")
print(f"Max cross-split similarity: {details['max_sim']:.4f}")
print(f"Pairs >0.80: {details['count_80']}")
print(f"Pairs >0.85: {details['count_85']}")
print(f"Pairs >0.90: {details['count_90']}")

# Verify distribution
test_cats = Counter(all_prompts[i]['category'] for i in test_indices)
test_in = Counter(all_prompts[i]['input_length_band'] for i in test_indices)
test_out = Counter(all_prompts[i]['output_length_band'] for i in test_indices)
test_mt = sum(1 for i in test_indices if all_prompts[i]['is_multi_turn'])

print(f"\nTest category distribution: {dict(test_cats)}")
print(f"Test input band distribution: {dict(test_in)}")
print(f"Test output band distribution: {dict(test_out)}")
print(f"Test multi-turn count: {test_mt}")

# Build new dev/test sets
new_dev = [all_prompts[i] for i in dev_indices]
new_test = [all_prompts[i] for i in test_indices]

# Verify no overlap
dev_ids = set(p['prompt_id'] for p in new_dev)
test_ids = set(p['prompt_id'] for p in new_test)
dev_texts = set(p['prompt_text'] for p in new_dev)
test_texts = set(p['prompt_text'] for p in new_test)

print(f"\nPrompt ID overlap: {len(dev_ids & test_ids)}")
print(f"Prompt text overlap: {len(dev_texts & test_texts)}")

# Verify features unchanged
print(f"\nDev sample keys: {sorted(new_dev[0].keys())}")
print(f"Test sample keys: {sorted(new_test[0].keys())}")

# Save new split
output = {
    "metadata": {
        **data['metadata'],
        "dev_count": len(new_dev),
        "test_count": len(new_test),
        "split_method": "family_aware_optimized",
        "cross_split_max_similarity": details['max_sim'],
        "cross_split_pairs_above_080": details['count_80'],
        "cross_split_pairs_above_085": details['count_85'],
        "cross_split_pairs_above_090": details['count_90'],
    },
    "development_prompts": new_dev,
    "locked_test_prompts": new_test,
}

output_path = Path("C:/CARBON GRID AI/data/evaluation/phase_g/prompts.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {output_path}")

# Also save texts only
prompts_only = {
    "development": [p['prompt_text'] for p in new_dev],
    "locked_test": [p['prompt_text'] for p in new_test],
}
with open(Path("C:/CARBON GRID AI/data/evaluation/phase_g/prompts_text_only.json"), 'w') as f:
    json.dump(prompts_only, f, indent=2)

print("Done.")