#!/usr/bin/env python3
"""Create a balanced 61/61/61 NPU-targeted dataset from v1 sources."""
import argparse, hashlib, json, shutil
from pathlib import Path

FAMILIES = ('permute','cat','split','pad','repeat','gather','scatter','index','nonzero','pool','interpol','quant','swiglu','gelu','activation','nms','iou','add','abs','cumsum','histc','sum','sort','rmsnorm','rope','attention','softmax','embedding','rotary','fft','topk','groupnorm','moe','layernorm','norm','loss','matmul')

def fam(meta):
    text = (meta.get('op_name','') + ' ' + ' '.join(map(str, meta.get('ops', [])))).lower()
    return next((x for x in FAMILIES if x in text), 'other')

def load(path):
    return {json.loads(x)['label']: json.loads(x) for x in path.read_text().splitlines() if x.strip()}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def choose(candidates, target_counts):
    buckets = {f: [] for f in target_counts if f != 'other'}
    for op, row in candidates.items():
        f = fam(row['metadata'])
        if f in buckets: buckets[f].append((op, row))
    chosen, used = [], set()
    while len(chosen) < 61:
        progress = False
        for f in sorted((x for x in target_counts if x != 'other'), key=lambda x: (-target_counts[x], x)):
            if len(chosen) >= 61: break
            available = [x for x in buckets[f] if x[0] not in used]
            if not available: continue
            available.sort(key=lambda x: (str(x[1]['metadata'].get('level','')), x[0]))
            op, row = available[0]
            chosen.append((op, row)); used.add(op); progress = True
        if not progress: break
    if len(chosen) != 61: raise RuntimeError(f'only {len(chosen)} family-matched tasks available')
    return chosen

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v1', type=Path, default=Path('/home/docker/datasets/operator_tasks_npu_target_5case_v1'))
    p.add_argument('--merged', type=Path, default=Path('/home/docker/datasets/operator_tasks_merged_5case'))
    p.add_argument('--cann-fixed', type=Path, default=Path('/home/docker/datasets/operator_tasks_merged_5case_cannops_fixed'))
    p.add_argument('--out', type=Path, default=Path('/home/docker/datasets/operator_tasks_npu_target_5case_v2'))
    a = p.parse_args()
    if a.out.exists(): raise SystemExit(f'refusing overwrite: {a.out}')
    (a.out/'data/op_tasks').mkdir(parents=True); (a.out/'full_cases').mkdir()
    v1, merged = load(a.v1/'operator_tasks.jsonl'), load(a.merged/'operator_tasks.jsonl')
    npu = [(op,r) for op,r in v1.items() if r['metadata']['data_source']=='npu-kernel-bench']
    target_counts = {}
    for _, row in npu: target_counts[fam(row['metadata'])] = target_counts.get(fam(row['metadata']), 0) + 1
    sources = {'cann': [(op,r) for op,r in merged.items() if r['metadata']['data_source']=='cann-ops'], 'cuda': [(op,r) for op,r in merged.items() if r['metadata']['data_source']=='cuda_llm']}
    rows=[]; selected={'npu':[], 'cann':[], 'cuda':[]}
    def add(op, row, source_root, key):
        shutil.copy2(source_root/'data/op_tasks'/(op+'.py') if source_root.name.startswith('operator_tasks_npu') else source_root/'op_tasks'/(op+'.py'), a.out/'data/op_tasks'/(op+'.py'))
        shutil.copy2(source_root/'data/op_tasks'/(op+'.json') if source_root.name.startswith('operator_tasks_npu') else source_root/'op_tasks'/(op+'.json'), a.out/'data/op_tasks'/(op+'.json'))
        meta=dict(row['metadata'], task_mode='npu_target_v2', target_family=fam(row['metadata']), case_count=5, task_source_sha256=digest(a.out/'data/op_tasks'/(op+'.py')))
        rows.append({'label':op,'metadata':meta,'prompt':[{'role':'user','content':f'Implement an AscendC operator for Ascend NPU. The reference task is at input/{op}.py. Preserve the five cases in input/{op}.json and the complete Model/get_input_groups/get_init_inputs contract. Produce a correct and efficient AscendC implementation through the fixed evaluator.'}]}); selected[key].append(op)
    for op,row in npu:
        add(op,row,a.v1,'npu'); full=a.v1/'full_cases'/(op+'.jsonl')
        if full.exists(): shutil.copy2(full,a.out/'full_cases'/(op+'.jsonl'))
    for key,source in (('cann',a.cann_fixed),('cuda',a.merged)):
        for op,row in choose(dict(sources[key]),target_counts): add(op,row,source,key)
    (a.out/'operator_tasks.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    (a.out/'selection.json').write_text(json.dumps({'counts':{k:len(v) for k,v in selected.items()},'target_family_counts':target_counts,'selected':selected,'policy':'61/61/61 family-balanced; all NPU level1/2 retained; CANN fixed; CUDA family-matched'},ensure_ascii=False,indent=2))
    (a.out/'README.md').write_text('NPU target dataset v2: 61 NPUKernelBench level1/2 + 61 CANN fixed + 61 CUDA family-matched; all tasks use five cases and one shared AscendC prompt contract.\n')
    print(json.dumps({'out':str(a.out),'counts':{k:len(v) for k,v in selected.items()}},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
