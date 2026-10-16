"""CPU input/forward equivalence for the data-only CANN initialization adapter."""
import ast,copy,hashlib,inspect,json,math,signal,sys
from pathlib import Path
from collections import Counter
import torch


def load(path):
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if not (isinstance(n,ast.Import) and any(a.name=='torch_npu' for a in n.names))]
    env={'__file__':str(path),'__name__':'cannops_validation'}
    with torch.device('meta'):exec(compile(tree,str(path),'exec'),env)
    return env

def clone(x):
    if isinstance(x,torch.Tensor):return x.clone()
    if isinstance(x,list):return [clone(v) for v in x]
    if isinstance(x,tuple):return tuple(clone(v) for v in x)
    if isinstance(x,dict):return {k:clone(v) for k,v in x.items()}
    return x

def compare(a,b):
    assert type(a)==type(b),(type(a),type(b))
    if isinstance(a,torch.Tensor):
        assert a.shape==b.shape and a.dtype==b.dtype
        if a.device.type!='meta':torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    else:assert a==b

def timeout_handler(*_):raise TimeoutError('CPU case exceeded 15 seconds')

def main():
    original=Path(sys.argv[1]);fixed=Path(sys.argv[2]);migration=json.loads((fixed/'cannops_init_migration.json').read_text())
    torch.set_num_threads(1);signal.signal(signal.SIGALRM,timeout_handler)
    records=[];errors=[];counts=Counter()
    for change in migration['changes']:
        op=change['op'];old=load(original/'op_tasks'/(op+'.py'));new=load(fixed/'op_tasks'/(op+'.py'))
        before=[json.loads(l) for l in (original/'op_tasks'/(op+'.json')).read_text().splitlines()]
        after=[json.loads(l) for l in (fixed/'op_tasks'/(op+'.json')).read_text().splitlines()]
        assert len(before)==len(after)==5
        shared=change['mode']=='shared'
        with torch.device('meta'):
            init=new['get_init_inputs']();inspect.signature(new['Model']).bind(*init)
        for index,(c,nc) in enumerate(zip(before,after)):
            record={'op':op,'case':index,'mode':change['mode']}
            # Shape-only arity checking covers even large cases without allocating tensors.
            supplied=len(nc['inputs']);inspect.signature(new['Model'].forward).bind(None,*[None]*supplied)
            counts['interface_cases_passed']+=1
            size=sum(math.prod(v.get('shape') or [])*8 for v in c['inputs']+c.get('init_inputs',[]) if v.get('type')=='tensor')
            if size>8*1024**2:
                record['cpu']='skipped_large';counts['cpu_skipped_large']+=1;records.append(record);continue
            stage='load'
            try:
                signal.alarm(15)
                with torch.random.fork_rng(devices=[]):
                    old['_active_case_json']=json.dumps(c)+'\n'
                    torch.manual_seed(c['seed'])
                    inputs=old['_original_get_input_groups']()[0]
                    config=old['get_init_inputs']()[0]
                    new['_read_five_cases']=lambda nc=nc:[nc]
                    groups=new['get_input_groups']()
                    actual=groups[0]
                    if shared:compare(inputs,actual);compare(config,new['get_init_inputs']())
                    else:
                        count=len(inspect.signature(old['Model']).parameters)
                        bound=inspect.signature(old['Model']).bind(*config);bound.apply_defaults()
                        full=list(bound.arguments.values());compare(full,actual[:count])
                        for j,spec in enumerate(c['inputs']):
                            if spec.get('uninitialized'):
                                assert inputs[j].shape==actual[count+j].shape and inputs[j].dtype==actual[count+j].dtype
                                # Output scratch buffers have intentionally undefined bytes.
                                # Compare forwards with identical scratch contents.
                                actual[count+j]=clone(inputs[j])
                        compare(inputs,actual[count:])
                    counts['cpu_inputs_passed']+=1
                    stage='reference'
                    torch.manual_seed(0);reference=old['Model'](*clone(config)).eval()
                    torch.manual_seed(0);candidate=new['Model'](*clone(config) if shared else []).eval()
                    torch.manual_seed(0);expected=reference(*clone(inputs))
                    stage='adapted'
                    torch.manual_seed(0);result=candidate(*clone(actual))
                    stage='comparison';compare(expected,result)
                    counts['cpu_forward_passed']+=1;record['cpu']='passed'
            except Exception as exc:
                record['cpu']=stage+'_error';record['error']=type(exc).__name__+': '+str(exc)[:900]
                counts[stage+'_error']+=1
                if stage in ('adapted','comparison','load'):errors.append(record)
            finally:
                signal.alarm(0);old['_active_case_json']=None;new['_active_case_json']=None
            records.append(record)
        print(op,dict(counts),flush=True)
    result={'counts':dict(counts),'migration_errors':errors,'cases':records,'scope':'CPU only; original reference failures and oversized cases remain explicitly unverified; no NPU operations'}
    (fixed/'cannops_init_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'counts':dict(counts),'migration_errors':errors},ensure_ascii=False,indent=2))
    assert not errors

if __name__=='__main__':main()
