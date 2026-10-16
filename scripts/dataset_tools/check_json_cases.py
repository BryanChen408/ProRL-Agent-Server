"""Compare migrated inputs with original generators without executing model.forward."""
import argparse
import ast
import copy
import json
import math
from collections import Counter
from pathlib import Path
import torch
from migrate_cudallm_json import reference_namespace, LOADER


def compare(a,b):
    assert type(a)==type(b),(type(a),type(b))
    if isinstance(a,torch.Tensor):
        assert a.shape==b.shape and a.dtype==b.dtype and a.requires_grad==b.requires_grad
        assert a.stride()==b.stride(),(a.stride(),b.stride())
        if a.device.type!='meta':torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    else:assert a==b


def loader_namespace():
    env={'torch':torch};exec(LOADER,env);return env


def check_modules(original, staging):
    errors=[];passed=0;cpu_init=[]
    for p in sorted((staging/'op_tasks').glob('cuda_llm_*.py')):
        try:
            tree=ast.parse(p.read_text())
            old=ast.parse((original/'op_tasks'/p.name).read_text())
            model=lambda t: next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Model')
            assert ast.dump(model(tree))==ast.dump(model(old)), 'Model computation changed'
            # NPU/librosa are model dependencies, not needed to exercise CPU/meta inputs.
            tree.body=[n for n in tree.body if not (isinstance(n,ast.Import) and any(a.name in ('torch_npu','librosa') for a in n.names))]
            env={'__file__':str(p),'__name__':'reference_validation'}
            with torch.device('meta'):
                exec(compile(tree,str(p),'exec'),env)
                groups=env['get_input_groups']()
                assert len(groups)==5
                try:
                    env['get_init_inputs']()
                except NotImplementedError:
                    # This initializer only converts the constant [64, 128] to a list.
                    if p.stem != 'cuda_llm_600793':raise
                    with torch.device('cpu'):env['get_init_inputs']()
                    cpu_init.append(p.stem)
            passed+=1
        except Exception as exc:
            errors.append({'op':p.stem,'error':repr(exc)})
    return {'passed':passed,'errors':errors,'cpu_initialization_fallback':cpu_init,'scope':'Model AST unchanged; actual input/initialization entrypoints on meta; NPU/librosa imports omitted'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('original',type=Path);parser.add_argument('staging',type=Path)
    parser.add_argument('--cpu-limit-mib',type=int,default=8)
    args=parser.parse_args();torch.set_num_threads(1)
    counts=Counter();errors=[];records=[];loader=loader_namespace()
    for i,p in enumerate(sorted((args.original/'op_tasks').glob('cuda_llm_*.py'))):
        old=[json.loads(l) for l in p.with_suffix('.json').read_text().splitlines() if l.strip()]
        q=args.staging/'op_tasks'/p.with_suffix('.json').name
        new=[json.loads(l) for l in q.read_text().splitlines() if l.strip()]
        assert len(old)==len(new)==5
        for before,after in zip(old,new):
            record={'op':p.stem,'case':after['case_id']}
            try:
                # Meta checks cover every case, including large tensors.
                for device in ('meta','cpu'):
                    size=sum(math.prod(n['shape'])*torch.empty((),dtype=getattr(torch,n['dtype'])).element_size()
                             for n in after['generators'] if 'shape' in n)
                    if device=='cpu' and size>args.cpu_limit_mib*1024**2:
                        counts['cpu_skipped_large']+=1;record['cpu']='skipped_large';continue
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(after['seed'])
                        env=reference_namespace(p.read_text(),device=device);env.update(before.get('globals',{}))
                        if not after.get('source_global_inputs'):torch.manual_seed(after['seed'])
                        with torch.device(device):a=env['_make_case_inputs']()
                        old_rng=torch.get_rng_state()
                        torch.manual_seed(after['seed'])
                        with torch.device(device):b=loader['_case_generate'](after)
                        compare(a,b)
                        if device=='cpu':assert torch.equal(old_rng,torch.get_rng_state()),'RNG order changed'
                    counts[device+'_passed']+=1;record[device]='passed'
            except Exception as exc:
                errors.append({'op':p.stem,'case':after['case_id'],'error':str(exc)[:900]})
                record['error']=str(exc)[:900]
            records.append(record)
        if (i+1)%200==0:print(i+1,dict(counts),'errors',len(errors),flush=True)
    # JSON must control direct input shape; list/tuple inputs must use canonical kinds.
    first=next((args.staging/'op_tasks').glob('cuda_llm_46443.json'))
    case=json.loads(first.read_text().splitlines()[0]);case['inputs'][0]['shape']=[3,128]
    with torch.device('meta'):assert list(loader['_case_generate'](case)[0].shape)==[3,128]
    modules=check_modules(args.original,args.staging)
    assert not modules['errors'],modules['errors']
    report={'modules':modules,'counts':dict(counts),'errors':errors,'cpu_limit_mib':args.cpu_limit_mib,'cases':records}
    (args.staging/'json_generation_validation.json').write_text(json.dumps(report,indent=2))
    print(dict(counts));print(json.dumps(errors[:20],indent=2))
    assert not errors, f'{len(errors)} case checks failed'

if __name__=='__main__':main()
