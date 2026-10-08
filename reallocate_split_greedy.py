#!/usr/bin/env python3
"""
Phase G.1.1 Split - Greedy + Local Search with Matrix Verification
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
print("Building conflict graph...")
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

print(f"Components: {len(components)}")

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

# Greedy construction: start with larger components that help balance
def build_initial():
    """Build initial test set by picking components greedily for balance"""
    # Sort components by size descending, then by "balance contribution"
    # Balance contribution: how much this component helps meet underfilled targets
    
    remaining = TARGET_TEST
    selected = set()
    current_cats = Counter()
    current_in = Counter()
    current_out = Counter()
    current_mt = 0
    
    # Candidates sorted by size descending
    candidates = sorted(range(len(comp_info)), key=lambda i: -comp_info[i]['size'])
    
    for ci in candidates:
        c = comp_info[ci]
        if c['size'] > remaining:
            continue
        
        # Score: how much does this help balance?
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
        
        if score > 0 or remaining <= 10:  # Always take if we need to fill
            selected.add(ci)
            remaining -= c['size']
            current_cats += c['cats']
            current_in += c['in_bands']
            current_out += c['out_bands']
            current_mt += c['mt_count']
            
            if remaining == 0:
                break
    
    # Fill remaining with size-1 components
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
    
    # Build test indices
    test_indices = []
    for ci in selected:
        test_indices.extend(comp_info[ci]['indices'])
    
    return test_indices

# Fast evaluation using precomputed matrix
def eval_fast(test_indices):
    test_set = set(test_indices)
    dev_indices = [i for i in range(n) if i not in test_set]
    
    max_sim = 0.0
    count_80 = 0
    count_85 = 0
    count_90 = 0
    for i in dev_indices:
        for j in test_indices:
            sim = sim_matrix[i][j]
            if sim > max_sim:
                max_sim = sim
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

# Local search: swap components to improve balance while maintaining zero leakage
def local_search(initial_test_indices, max_iter=10000):
    best = eval_fast(initial_test_indices)
    if best['count_85'] > 0:
        return None
    
    test_set = set(initial_test_indices)
    test_list = list(test_set)
    
    for iteration in range(max_iter):
        if iteration % 2000 == 0:
            print(f"  Iter {iteration}: score={best['score']}, max_sim={best['max_sim']:.4f}, >0.80={best['count_80']}")
        
        # Pick a random test prompt and a random dev prompt
        if len(test_list) < 2:
            break
        
        # Try swapping a component between test and dev
        # Pick a test component to remove
        test_comps = []
        for ci, c in enumerate(comp_info):
            if any(idx in test_set for idx in c['indices']):
                # Check if ALL indices of this component are in test
                if all(idx in test_set for idx in c['indices']):
                    test_comps.append(ci)
        
        dev_comps = []
        for ci, c in enumerate(comp_info):
            if all(idx not in test_set for idx in c['indices']):
                dev_comps.append(ci)
        
        if not test_comps or not dev_comps:
            break
        
        # Try random swap
        tc = random.choice(test_comps)
        dc = random.choice(dev_comps)
        
        tc_size = comp_info[tc]['size']
        dc_size = comp_info[dc]['size']
        
        if tc_size != dc_size:
            continue  # Must maintain size 60
        
        # Perform swap
        new_test_set = test_set.copy()
        for idx in comp_info[tc]['indices']:
            new_test_set.remove(idx)
        for idx in comp_info[dc]['indices']:
            new_test_set.add(idx)
        
        new_test_list = list(new_test_set)
        eval_result = eval_fast(new_test_list)
        
        if eval_result['count_85'] == 0:
            if eval_result['score'] < best['score']:
                best = eval_result
                test_set = new_test_set
                test_list = new_test_list
                print(f"    Improved: score={best['score']}, max_sim={best['max_sim']:.6f}, >0.80={best['count_80']}")
    
    return best

# Main
print("Building initial split...")
for attempt in range(100):
    initial = build_initial()
    if initial:
        print(f"Attempt {attempt}: initial score={eval_fast(initial)['score']}")
        result = local_search(initial, max_iter=5000)
        if result:
            print(f"  Result: score={result['score']}, max_sim={result['max_sim']:.6f}, >0.80={result['count_80']}, >0.85={result['count_85']}")
            if result['count_85'] == 0:
                # Final exhaustive verification
                print("\n=== FINAL EXHAUSTIVE VERIFICATION ===")
                dev_texts = [all_prompts[i]['prompt_text'] for i in result['dev_indices']]
                test_texts = [all_prompts[i]['prompt_text'] for i in result['test_indices']]
                
                sims = []
                for d in dev_texts:
                    for t in test_texts:
                        sims.append(SequenceMatcher(None, d.lower(), t.lower()).ratio())
                
                print(f"Min sim: {min(sims):.6f}")
                print(f"Max sim: {max(sims):.6f}")
                print(f"Mean sim: {sum(sims)/len(sims):.6f}")
                print(f">0.80: {sum(1 for s in sims if s > 0.80)}")
                print(f">0.85: {sum(1 for s in sims if s > 0.85)}")
                print(f">0.90: {sum(1 for s in sims if s > 0.90)}")
                
                if max(sims) <= 0.85:
                    print("SUCCESS: Zero cross-split >0.85!")
                    
                    # Save
                    new_dev = [all_prompts[i] for i in result['dev_indices']]
                    new_test = [all_prompts[i] for i in result['test_indices']]
                    
                    output = {
                        "metadata": {
                            **data['metadata'],
                            "dev_count": len(new_dev),
                            "test_count": len(new_test),
                            "split_method": "conflict_component_greedy_local_search_verified",
                            "cross_split_max_similarity": result['max_sim'],
                            "cross_split_pairs_above_080": result['count_80'],
                            "cross_split_pairs_above_085": result['count_85'],
                            "cross_split_pairs_above_090": result['count_90'],
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
                    print(f"  FAILED: max_sim={max(sims):.6f} > 0.85")

print("No valid split found after all attempts")