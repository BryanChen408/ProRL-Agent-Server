"""Stage CUDA LLM JSON-driven inputs; never overwrites the source dataset."""
import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import operator
from pathlib import Path
import shutil
import symtable
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LOADER = (HERE / 'json_case_loader.py').read_text()
CONSTRUCTORS = {'randn', 'rand', 'ones', 'empty', 'randint', 'full'}

class TensorValue:
    def __init__(self, recorder, key, value):
        self.recorder, self.key, self.value = recorder, key, value
    def __setitem__(self, key, value):
        self.recorder.call('setitem', self, key, value)
    def __getattr__(self, name):
        if name == 'shape':
            return self.value.shape
        return lambda *a, **kw: self.recorder.call('method.' + name, self, *a, **kw)

for name in ('add', 'sub', 'mul', 'truediv', 'floordiv', 'matmul', 'gt', 'lt', 'pow', 'eq'):
    setattr(TensorValue, '__' + name + '__', lambda self, other, name=name: self.recorder.call('operator.' + name, self, other))
    setattr(TensorValue, '__r' + name + '__', lambda self, other, name=name: self.recorder.call('operator.' + name, other, self))

class TorchProxy:
    def __init__(self, recorder, functional=False):
        self.recorder, self.functional = recorder, functional
    def __getattr__(self, name):
        module = torch.nn.functional if self.functional else torch
        value = getattr(module, name)
        if isinstance(value, torch.dtype) or name in ('Tensor', 'nn'):
            return value
        return lambda *a, **kw: self.recorder.call(('functional.' if self.functional else '') + name, *a, **kw)

class Recorder:
    def __init__(self):
        self.nodes = []
    def encode(self, x):
        if isinstance(x, TensorValue):
            return {'ref': x.key}
        if isinstance(x, torch.dtype):
            return {'dtype': str(x).removeprefix('torch.')}
        if isinstance(x, tuple):
            return {'tuple': [self.encode(v) for v in x]}
        if isinstance(x, list):
            return [self.encode(v) for v in x]
        if isinstance(x, complex):return {'complex': [x.real, x.imag]}
        if isinstance(x, float) and not math.isfinite(x):return {'float': str(x)}
        if x is None or type(x) in (int, float, str, bool):
            return x
        raise TypeError(f'Cannot encode {type(x)}')
    def unwrap(self, x):
        if isinstance(x, TensorValue):return x.value
        if isinstance(x, tuple):return tuple(self.unwrap(v) for v in x)
        if isinstance(x, list):return [self.unwrap(v) for v in x]
        return x
    def call(self, op, *args, **kwargs):
        a=self.unwrap(args);kw={k:self.unwrap(v) for k,v in kwargs.items()}
        with torch.device('meta'):
            if op == 'setitem':value = a[0]
            elif op.startswith('method.'):
                value=getattr(a[0],op[7:])(*a[1:],**kw)
            elif op.startswith('operator.'):
                value=getattr(operator,op[9:])(*a,**kw)
            elif op.startswith('functional.'):
                value=getattr(torch.nn.functional,op[11:])(*a,**kw)
            else:value=getattr(torch,op)(*a,**kw)
        if not isinstance(value,torch.Tensor):return value
        key=f'g{len(self.nodes)}'
        node={'id':key,'op':op,'args':self.encode(list(args)),
              'kwargs':{k:self.encode(v) for k,v in kwargs.items()}}
        if op in CONSTRUCTORS:
            node['shape']=list(value.shape);node['dtype']=str(value.dtype).removeprefix('torch.')
            if op=='randint':
                node['low']=args[0] if len(args)>=3 else kwargs.get('low',0)
                node['high']=args[1] if len(args)>=3 else args[0] if args else kwargs['high']
            if op=='full':node['fill_value']=args[1] if len(args)>1 else kwargs['fill_value']
            node['args']=[]
            for k in ('size','dtype','low','high','fill_value'):node['kwargs'].pop(k,None)
        self.nodes.append(node)
        return TensorValue(self,key,value)

def reference_namespace(source, recorder=None, device="meta"):
    tree=ast.parse(source.split('# Five-case dataset adapter.')[0])
    # Only input helpers/constants are needed; do not import backends or instantiate models.
    nodes=[n for n in tree.body if isinstance(n,(ast.Assign,ast.AnnAssign,ast.FunctionDef))]
    env={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'math':math,'np':np}
    if recorder is not None:
        env['torch']=TorchProxy(recorder);env['F']=TorchProxy(recorder,True)
    with torch.device(device):
        for node in nodes:
            is_module_init = isinstance(node, ast.Assign) and any(isinstance(n, ast.Call) and ast.unparse(n.func).startswith(('nn.', 'torch.nn.')) for n in ast.walk(node))
            with torch.device('meta' if is_module_init else device):
                exec(compile(ast.Module(body=[node],type_ignores=[]),'<reference-inputs>','exec'),env)
    return env

def input_spec(old,value):
    spec=copy.deepcopy(old)
    if isinstance(value,TensorValue):
        assert spec['type']=='tensor'
        assert list(value.value.shape)==spec['shape'], (spec,value.value.shape)
        assert str(value.value.dtype).removeprefix('torch.')==spec['dtype']
        spec['source']=value.key
    elif isinstance(value,(list,tuple)):
        if all(isinstance(v,TensorValue) for v in value):
            spec['type']='tensor_list'
            spec['items']=[input_spec(s,v) for s,v in zip(spec['items'],value,strict=True)]
        else:
            assert all(type(v) in (int,float,bool,str) for v in value)
            spec={'name':spec['name'],'type':'attr','value':list(value),
                  'container':'tuple' if isinstance(value,tuple) else 'list','required':True}
    else:
        assert spec['type']=='attr' and spec['value']==value,(spec,value)
    return spec

def convert(source,cases):
    output=[]
    for case in cases:
        rec=Recorder();env=reference_namespace(source,rec);env.update(case.get('globals',{}))
        env['torch']=TorchProxy(rec);env['F']=TorchProxy(rec,True)
        global_count=len(rec.nodes)
        values=env['_make_case_inputs']()
        def references(x):
            if isinstance(x,dict):
                if set(x)=={'ref'}:return {x['ref']}
                return set().union(*(references(v) for v in x.values()))
            if isinstance(x,list):return set().union(*(references(v) for v in x))
            return set()
        needed=references(rec.encode(values))
        for node in rec.nodes[global_count:]:needed |= references(node)
        for node in reversed(rec.nodes[:global_count]):
            if node['id'] in needed:needed |= references(node)
        rec.nodes=[n for i,n in enumerate(rec.nodes) if i>=global_count or n['id'] in needed]
        new={'case_id':case['case_id'],'seed':case['seed'],'generator':'json_tensor_graph_v1',
             'inputs':[input_spec(s,v) for s,v in zip(case['inputs'],values,strict=True)],
             'generators':rec.nodes,'input_container':'tuple' if isinstance(values,tuple) else 'list',
             'source_global_inputs':bool(any(int(n['id'][1:])<global_count for n in rec.nodes))}
        output.append(new)
    prefix=source.split('# Five-case dataset adapter.')[0]
    tree=ast.parse(prefix);lines=prefix.splitlines(keepends=True)
    definitions={}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            definitions[node.name]=node
        elif isinstance(node,ast.Assign):
            for target in node.targets:
                for name in ast.walk(target):
                    if isinstance(name,ast.Name):definitions[name.id]=node
    def dependencies(node):
        table=symtable.symtable(ast.unparse(node),'<reference>','exec')
        names=set()
        def visit(t):
            names.update(s.get_name() for s in t.get_symbols() if s.is_global() and s.is_referenced())
            for child in t.get_children():visit(child)
        visit(table)
        return names
    keep=set();pending=['Model','get_init_inputs']
    while pending:
        name=pending.pop()
        if name=='_make_case_inputs' or name in keep or name not in definitions:continue
        keep.add(name);pending.extend(dependencies(definitions[name])-keep)
    needed={id(definitions[name]) for name in keep}
    # Keep Model/initialization byte-for-byte; remove obsolete input generators and
    # their module-level tensor allocations. Input generation lives only in JSON.
    for node in reversed(tree.body):
        if not isinstance(node,(ast.Import,ast.ImportFrom)) and id(node) not in needed:
            del lines[node.lineno-1:node.end_lineno]
    return ''.join(lines).rstrip()+'\n\n# JSON-driven input cases.\n',output

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('staging',type=Path)
    args=parser.parse_args()
    if args.staging.exists():raise FileExistsError(args.staging)
    args.staging.mkdir(parents=True)
    shutil.copytree(args.source/'op_tasks',args.staging/'op_tasks',ignore=shutil.ignore_patterns('__pycache__'))
    changes={};errors=[]
    for i,p in enumerate(sorted((args.source/'op_tasks').glob('cuda_llm_*.py'))):
        try:
            source=p.read_text();cases=[json.loads(l) for l in p.with_suffix('.json').read_text().splitlines() if l.strip()]
            prefix,converted=convert(source,cases)
            q=args.staging/'op_tasks'/p.name
            q.write_text(prefix+f'_CASE_FILE = {p.with_suffix(".json").name!r}\n'+LOADER)
            q.with_suffix('.json').write_text(''.join(json.dumps(c,ensure_ascii=False,sort_keys=True)+'\n' for c in converted))
            changes[p.stem]=digest(q)
        except Exception as exc:
            errors.append({'op':p.stem,'error':repr(exc)})
        if (i+1)%200==0:print('converted',i+1,'errors',len(errors),flush=True)
    (args.staging/'migration_errors.json').write_text(json.dumps(errors,indent=2))
    if errors:
        print(json.dumps(errors[:30],indent=2));raise SystemExit('Conversion incomplete; source unchanged')
    for name in ('operator_tasks.jsonl','manifest.jsonl'):
        rows=[json.loads(l) for l in (args.source/name).read_text().splitlines() if l.strip()]
        for r in rows:
            op=r.get('op_name') or r['metadata']['op_name']
            if op in changes:
                if name=='manifest.jsonl':
                    r['output_sha256']=changes[op];r['input_generation']='json_tensor_graph_v1'
                else:r['metadata']['task_source_sha256']=changes[op]
        (args.staging/name).write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))
    print('Converted',len(changes),'operators; staged at',args.staging)

if __name__=='__main__':main()
