import json
with open('data/evaluation/phase_g/phase_g_feasibility_report.json') as f:
    r = json.load(f)

print('FP16:')
for name, m in r['fp16']['models'].items():
    print('  {}: MAE={:.6f}, RMSE={:.6f}, R2={:.4f}'.format(name, m['mae'], m['rmse'], m['r2']))

print('INT4:')
for name, m in r['int4']['models'].items():
    print('  {}: MAE={:.6f}, RMSE={:.6f}, R2={:.4f}'.format(name, m['mae'], m['rmse'], m['r2']))

print()
print('Linear Regression:')
lr_fp16 = r['linear_regression']['fp16']
lr_int4 = r['linear_regression']['int4']
print('  FP16: MAE={:.6f}, R2={:.4f}'.format(lr_fp16['mae'], lr_fp16['r2']))
print('  INT4: MAE={:.6f}, R2={:.4f}'.format(lr_int4['mae'], lr_int4['r2']))

print()
print('Summary:')
print('  FP16 beats baseline: {}'.format(r['summary']['fp16']['beats_baseline']))
print('  INT4 beats baseline: {}'.format(r['summary']['int4']['beats_baseline']))
print('  Influential FP16: {}'.format(len(r['influential_prompts']['fp16'])))
print('  Influential INT4: {}'.format(len(r['influential_prompts']['int4'])))

if r['influential_prompts']['fp16']:
    print('  FP16 influential:')
    for p in r['influential_prompts']['fp16']:
        print('    {} true={:.6f} max_err={:.6f}'.format(p['prompt_id'], p['true_energy'], p['max_abs_error']))
if r['influential_prompts']['int4']:
    print('  INT4 influential:')
    for p in r['influential_prompts']['int4']:
        print('    {} true={:.6f} max_err={:.6f}'.format(p['prompt_id'], p['true_energy'], p['max_abs_error']))