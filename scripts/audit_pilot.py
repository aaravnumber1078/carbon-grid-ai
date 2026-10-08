import json
from collections import defaultdict

with open('data/evaluation/phase_g/pilot_results.json') as f:
    data = json.load(f)

runs = data['runs']
successful = [r for r in runs if r['success']]
fp16 = [r for r in successful if r['configuration'] == 'fp16']
int4 = [r for r in successful if r['configuration'] == 'int4']

print(f'Total runs: {len(runs)}')
print(f'Successful: {len(successful)}')
print(f'FP16 runs: {len(fp16)}')
print(f'INT4 runs: {len(int4)}')

by_prompt = defaultdict(lambda: {'fp16': [], 'int4': []})
for r in successful:
    by_prompt[r['prompt_id']][r['configuration']].append(r)

print(f'Prompts with data: {len(by_prompt)}')
for pid, configs in list(by_prompt.items())[:3]:
    print(f'  {pid}: fp16={len(configs["fp16"])}, int4={len(configs["int4"])}')

missing_energy = [r for r in successful if r['energy_wh'] is None]
print(f'Missing energy: {len(missing_energy)}')

measured = [r for r in successful if r['measurement_source'] == 'measured']
estimated = [r for r in successful if r['measurement_source'] == 'estimated']
print(f'Measured: {len(measured)}')
print(f'Estimated: {len(estimated)}')

# Check config_order randomization
config_orders = [r['config_order'] for r in successful]
print(f'Config orders: {set(config_orders)}')
print(f'FP16 first: {config_orders.count(0)}')
print(f'INT4 first: {config_orders.count(1)}')

# Check repetitions
for pid, configs in by_prompt.items():
    if len(configs['fp16']) != 3 or len(configs['int4']) != 3:
        print(f'WARNING: {pid} has fp16={len(configs["fp16"])}, int4={len(configs["int4"])}')

# Check if pilot prompts overlap with locked test set
with open('data/evaluation/phase_g/prompts.json') as f:
    full = json.load(f)
test_ids = set(p['prompt_id'] for p in full['locked_test_prompts'])
pilot_ids = set(by_prompt.keys())
overlap = test_ids & pilot_ids
print(f'Pilot prompts overlapping with locked test: {len(overlap)}')
if overlap:
    print(f'  Overlapping IDs: {overlap}')