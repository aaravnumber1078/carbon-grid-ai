#!/usr/bin/env python3
"""
Phase G.1.1 Split - Single-Consistent-Computation Approach
Builds conflict graph and verifies using IDENTICAL SequenceMatcher calls
"""

import json
import random
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher

# Load original prompts
with open('C:/CARBON GRID AI/data/evaluation/phase_g/prompts.json') as f:
    data = json.load(f)

all_prompts = data['development_prompts'] + data['locked_test_prompts']
n = len(all_prompts)

# Single similarity function used EVERYWHERE
def sim(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Build FULL conflict graph using the SAME sim function
print("Building conflict graph with unified similarity...")
adj = [[] for _ in range(n)]
edge_count = 0
for i in range(n):
    ti = all_prompts[i]['prompt_text']
    for j in range(i+1, n):
        if sim(ti, all_prompts[j]['prompt_text']) > 0.85:
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

print(f"Components: {len(components)}")
size_dist = Counter(len(c) for c in components)
print(f"Size distribution: {dict(size_dist)}")

# VERIFY: No cross-component >0.85 pairs using SAME sim function
cross_85 = 0
for i in range(n):
    ti = all_prompts[i]['prompt_text']
    for j in range(i+1, n):
        comp_i = next(ci for ci, comp in enumerate(components) if i in comp)
        comp_j = next(ci for ci, comp in enumerate(components) if j in comp)
        if comp_i != comp_j and sim(ti, all_prompts[j]['prompt_text']) > 0.85:
            cross_85 += 1
            print(f"  CROSS COMPONENT EDGE: {i}({comp_i}) - {j}({comp_j}) = {sim(ti, all_prompts[j]['prompt_text']):.6f}")
print(f"Cross-component >0.85 pairs: {cross_85}")

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

# Greedy construction with component-aware scoring
def build_initial():
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
        return None
    
    test_indices = []
    for ci in selected:
        test_indices.extend(comp_info[ci]['indices'])
    return test_indices

# Unified evaluation using the SAME sim function
def eval_unified(test_indices):
    test_set = set(test_indices)
    dev_indices = [i for i in range(n) if i not in test_set]
    
    max_sim = 0.0
    count_80 = 0
    count_85 = 0
    count_90 = 0
    for i in dev_indices:
        ti = all_prompts[i]['prompt_text']
        for j in test_indices:
            s = sim(ti, all_prompts[j]['prompt_text'])
            if s > max_sim:
                max_sim = s
            if s > 0.80:
                count_80 += 1
            if s > 0.85:
                count_85 += 1
            if s > 0.90:
                count_90 += 1
    
    test_cats = Counter(all_prompts[i]['category'] for i in test_indices)
    cat_score = sum(abs(test_cats.get(c, 0) - 6) for c in category_targets)
    
    test_in = Counter(all_prompts[i]['input_length_band'] for i in test_indices)
    in_score = sum(abs(test_in.get(b, 0) - input_band_targets[b]) for b in input_band_targets)
    
    test_out = Counter(all_prompts[i]['output_length_band'] for i in test_indices)
    out_score = sum(abs(test_out.get(b, 0) - output_band_targets[b]) for b in output_band_targets)
    
    test_mt = sum(1 for i in test_indices if all_prompts[i]['is_multi_turn'])
    mt_score = abs(test_mt - multi_turn_target) * 5
    
    return {
        'score': cat_score + in_score + out_score + mt_score,
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

# Local search with UNIFIED evaluation
def local_search(initial_test_indices, max_iter=5000):
    best = eval_unified(initial_test_indices)
    if best['count_85'] > 0:
        return None
    
    test_set = set(initial_test_indices)
    
    # Pre-compute component membership
    comp_of = [-1]*n
    for ci, c in enumerate(comp_info):
        for idx in c['indices']:
            comp_of[idx] = ci
    
    for iteration in range(max_iter):
        # Find swappable components
        test_comps = set(comp_of[idx] for idx in test_set)
        dev_comps = [ci for ci in range(len(comp_info)) if ci not in test_comps]
        
        if not test_comps or not dev_comps:
            break
        
        # Try swap
        tc = random.choice(list(test_comps))
        dc = random.choice(dev_comps)
        
        if comp_info[tc]['size'] != comp_info[dc]['size']:
            continue
        
        # Perform swap
        new_test_set = test_set.copy()
        for idx in comp_info[tc]['indices']:
            new_test_set.remove(idx)
        for idx in comp_info[dc]['indices']:
            new_test_set.add(idx)
        
        eval_result = eval_unified(list(new_test_set))
        
        if eval_result['count_85'] == 0 and eval_result['score'] < best['score']:
            best = eval_result
            test_set = new_test_set
            print(f"    Iter {iteration}: Improved score={best['score']}, max_sim={best['max_sim']:.6f}, >0.80={best['count_80']}")
    
    return best

# Main loop
random.seed(42)
category_targets = {cat: 6 for cat in [
    'factual', 'extraction', 'explanation', 'reasoning', 'summarization',
    'coding', 'scientific', 'creative', 'instruction_heavy', 'multi_turn'
]}
input_band_targets = {'very_short': 6, 'short': 12, 'medium': 18, 'long': 15, 'very_long': 9}
output_band_targets = {'very_short': 6, 'short': 12, 'medium': 18, 'long': 15, 'very_long': 9}
multi_turn_target = 6

print("Starting search...")
for attempt in range(50):
    initial = build_initial()
    if initial is None:
        continue
    
    initial_eval = eval_unified(initial)
    if initial_eval['count_85'] > 0:
        continue
    
    print(f"Attempt {attempt}: initial score={initial_eval['score']}, max_sim={initial_eval['max_sim']:.6f}")
    result = local_search(initial, max_iter=3000)
    
    if result and result['count_85'] == 0:
        # FINAL verification with fresh SequenceMatcher on split lists
        new_dev = [all_prompts[i] for i in result['dev_indices']]
        new_test = [all_prompts[i] for i in result['test_indices']]
        
        print("\n=== FINAL EXHAUSTIVE VERIFICATION ===")
        dev_texts = [p['prompt_text'] for p in new_dev]
        test_texts = [p['prompt_text'] for p in new_test]
        
        sims = []
        for d in dev_texts:
            for t in test_texts:
                sims.append(SequenceMatcher(None, d.lower(), t.lower()).ratio())
        
        max_sim_final = max(sims)
        count_85_final = sum(1 for s in sims if s > 0.85)
        count_90_final = sum(1 for s in sims if s > 0.90)
        count_80_final = sum(1 for s in sims if s > 0.80)
        
        print(f"Max sim: {max_sim_final:.6f}")
        print(f">0.80: {count_80_final}, >0.85: {count_85_final}, >0.90: {count_90_final}")
        
        if count_85_final == 0:
            print("SUCCESS: Zero cross-split >0.85!")
            
            # Save
            output = {
                "metadata": {
                    **data['metadata'],
                    "dev_count": len(new_dev),
                    "test_count": len(new_test),
                    "split_method": "unified_sim_conflict_component_greedy_verified",
                    "cross_split_max_similarity": max_sim_final,
                    "cross_split_pairs_above_080": count_80_final,
                    "cross_split_pairs_above_085": count_85_final,
                    "cross_split_pairs_above_090": count_90_final,
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
            exit(0)
        else:
            print(f"  FAILED final verification: max_sim={max_sim_final:.6f}, >0.85={count_85_final}")
    else:
        print(f"  Initial has leakage: >0.85={initial_eval['count_85']}")

print("No valid split found")