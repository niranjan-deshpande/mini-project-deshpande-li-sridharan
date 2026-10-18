"""CPU check of gpu/session1.py internals on a small same-architecture model:
padded batches == single sequences, and early-exit hook output == hidden_states[L+1] of a full forward."""
import sys, json, numpy as np, torch
model_dir = sys.argv[1]
sys.argv = ['x', '--model', model_dir, '--data', 'data/gpu_test', '--out', 'data/gpu_test/out_verify', '--layer', '15',
            '--layers', '8,15', '--device', 'cpu', '--dtype', 'float32', '--stages', 'manifest', '--test']
exec(open('gpu/session1.py').read().split('STAGES = {')[0])  # definitions only
pairs = [json.loads(l) for l in open('data/gpu_test/final/pairs_hack.jsonl')][:3]
items = [locate(tok, p[b]) for p in pairs for b in ('hack', 'honest')]
batched = harvest(items, [15], keys=('turn', 'span'))
single = [harvest([it], [15], keys=('turn', 'span')) for it in items]
for k in ('turn', 'span'):
    d = max(np.abs(batched[k][i].astype(np.float32) - single[i][k][0].astype(np.float32)).max() for i in range(len(items)))
    print(f'padding test {k}: max abs diff batched vs single {d:.2e} (scale {np.abs(batched[k]).max():.1f})')
ids = items[0][0]
with torch.no_grad():
    hs = model(torch.tensor([ids]), output_hidden_states=True, use_cache=False).hidden_states[16][0]
cap = Capture([15]); h = cap.run(input_ids=torch.tensor([ids]))[15][0]; cap.close()
print(f'early-exit vs hidden_states[16]: max abs diff {float((h - hs).abs().max()):.2e}')
