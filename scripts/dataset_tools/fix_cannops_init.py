"""Adapt per-case CANN initialization to the existing shared-init evaluator.
Writes a separate dataset, leaving active sessions' original assets untouched.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil

ADAPTER = '''
def get_init_inputs():
    return []

def get_input_groups():
    global _active_case_json
    groups = []
    try:
        for case in _read_five_cases():
            legacy_case = dict(case)
            legacy_case["init_inputs"] = case["inputs"][:_CASE_CONFIG_COUNT]
            legacy_case["inputs"] = case["inputs"][_CASE_CONFIG_COUNT:]
            _active_case_json = _case_json.dumps(legacy_case) + "\\n"
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(case["seed"])
                inputs = _original_get_input_groups()
                configs = _get_case_init_groups()
            if len(inputs) != 1 or len(configs) != 1:
                raise ValueError("Expected one input/configuration group per case")
            values = list(configs[0])
            values.extend(spec["value"] for spec in case["inputs"][len(values):_CASE_CONFIG_COUNT])
            groups.append(values + list(inputs[0]))
    finally:
        _active_case_json = None
    return groups
'''
SHARED = '''
def get_init_inputs():
    cases = _read_five_cases()
    if any(case.get("init_inputs", []) != cases[0].get("init_inputs", []) for case in cases[1:]):
        raise ValueError("Shared initialization must be identical across all cases")
    return _get_case_init_groups()[0]
'''

class InitNames(ast.NodeTransformer):
    def __init__(self,names):self.names=names
    def visit_Name(self,node):
        if node.id in self.names:node.id='case_init_'+node.id
        return node

class FunctionalModules(ast.NodeTransformer):
    def __init__(self,op):self.op=op
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='self':
            if self.op.endswith('_ReflectionPad3dGrad') and node.func.attr=='pad':
                call=ast.parse("torch.nn.functional.pad(x, case_init_padding, mode='reflect')",mode='eval').body
                call.args[0]=node.args[0];return ast.copy_location(call,node)
            if self.op.endswith('_AddLayerNorm') and node.func.attr=='layer_norm':
                call=ast.parse('torch.nn.functional.layer_norm(x, case_init_gamma.shape, case_init_gamma.float(), case_init_beta.float(), case_init_epsilon)',mode='eval').body
                call.args[0]=node.args[0];return ast.copy_location(call,node)
        return node

def transform(source,cases,op):
    tree=ast.parse(source);model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Model')
    constructor=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    forward=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    old_init=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_init_inputs')
    shared=all(c.get('init_inputs',[])==cases[0].get('init_inputs',[]) for c in cases)
    original_inputs=[copy.deepcopy(c['inputs']) for c in cases]
    lines=source.splitlines(keepends=True);edits=[]
    renamed=copy.deepcopy(old_init);renamed.name='_get_case_init_groups'
    edits.append((old_init.lineno-1,old_init.end_lineno,ast.unparse(renamed)+'\n'))
    if shared:
        suffix=SHARED;mode='shared'
    else:
        names=[a.arg for a in constructor.args.args[1:]]
        defaults=[None]*(len(names)-len(constructor.args.defaults))+constructor.args.defaults
        for case in cases:
            for i in range(len(case['init_inputs']),len(names)):
                assert defaults[i] is not None,(op,names[i])
                value=ast.literal_eval(defaults[i])
                case['init_inputs'].append({'name':names[i],'type':'attr','value':value,'dtype':type(value).__name__,'required':True})
        assert not ({'case_init_'+n for n in names}&{a.arg for a in forward.args.args})
        init_args=copy.deepcopy(constructor.args.args[1:])
        for arg in init_args:arg.arg='case_init_'+arg.arg
        new_model=copy.deepcopy(model)
        ci=next(n for n in new_model.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        fw=next(n for n in new_model.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        prelude=[]
        for n in ci.body:
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):continue
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='__init__':continue
            if op.endswith('_ReflectionPad3dGrad') or op.endswith('_AddLayerNorm'):
                if isinstance(n,ast.Assign) and any('self.pad' in ast.unparse(a) or 'self.layer_norm' in ast.unparse(a) for a in n.targets):continue
            assert isinstance(n,ast.Assign),(op,ast.unparse(n))
            prelude.append(InitNames(set(names)).visit(copy.deepcopy(n)))
        ci.args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],kwonlyargs=[],kw_defaults=[],defaults=[])
        ci.body=ast.parse('super().__init__()').body
        fw.args.args=[fw.args.args[0]]+init_args+fw.args.args[1:]
        offset=int(bool(fw.body and isinstance(fw.body[0],ast.Expr) and isinstance(fw.body[0].value,ast.Constant) and isinstance(fw.body[0].value.value,str)))
        fw.body[offset:offset]=prelude
        new_model=FunctionalModules(op).visit(new_model);ast.fix_missing_locations(new_model)
        edits.append((model.lineno-1,model.end_lineno,ast.unparse(new_model)+'\n'))
        for c in cases:
            configs=c.pop('init_inputs')
            for name,spec in zip(names,configs):spec['name']='case_init_'+name
            c['inputs']=configs+c['inputs']
            c['case_configuration_inputs']=len(names)
        # Optional tensor arguments must remain Tensor? in the generated skeleton.
        for i,name in enumerate(names):
            entries=[c['inputs'][i] for c in cases]
            if any(s.get('type')=='tensor' for s in entries) and any(s.get('value','not-none') is None for s in entries):
                for s in entries:
                    if s['type']=='tensor':s['required']=False
        last_provider=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_input_groups'][-1]
        edits.append((last_provider.lineno-1,last_provider.end_lineno,''))
        suffix=f'\n_CASE_CONFIG_COUNT = {len(names)}\n'+ADAPTER;mode='forward_config'
    for start,end,new in sorted(edits,reverse=True):lines[start:end]=[new]
    converted=''.join(lines).rstrip()+'\n'+suffix
    ast.parse(converted)
    for before,c in zip(original_inputs,cases):
        assert before==c['inputs'][c.get('case_configuration_inputs',0):]
    return converted,cases,mode

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    shutil.copytree(args.source/'op_tasks',args.output/'op_tasks',ignore=shutil.ignore_patterns('__pycache__'))
    changed=[]
    for p in sorted((args.source/'op_tasks').glob('cannops_*.py')):
        source=p.read_text();tree=ast.parse(source)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_init_inputs')
        if not any(isinstance(n,ast.Return) and isinstance(n.value,ast.Name) and n.value.id=='init_groups' for n in ast.walk(fn)):continue
        cases=[json.loads(l) for l in p.with_suffix('.json').read_text().splitlines() if l.strip()]
        text,cases,mode=transform(source,cases,p.stem);q=args.output/'op_tasks'/p.name
        q.write_text(text);q.with_suffix('.json').write_text(''.join(json.dumps(c,ensure_ascii=False,sort_keys=True)+'\n' for c in cases))
        changed.append({'op':p.stem,'mode':mode,'before_sha256':sha(p),'after_sha256':sha(q),'original':str(p)})
    changes={r['op']:r for r in changed};assert len(changes)==59
    for name in ['operator_tasks.jsonl','manifest.jsonl']:
        rows=[json.loads(l) for l in (args.source/name).read_text().splitlines() if l.strip()]
        for r in rows:
            op=r.get('op_name') or r['metadata']['op_name']
            if op in changes:
                if name=='manifest.jsonl':r['output_sha256']=changes[op]['after_sha256'];r['initialization_contract']=changes[op]['mode']
                else:r['metadata']['task_source_sha256']=changes[op]['after_sha256']
        (args.output/name).write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))
    for name in ['README.md','report.json']:shutil.copy2(args.source/name,args.output/name)
    (args.output/'cannops_init_migration.json').write_text(json.dumps({'source_dataset':str(args.source),'changes':changed},ensure_ascii=False,indent=2))
    print('Migrated',len(changed),'operators:',{m:sum(r['mode']==m for r in changed) for m in ['shared','forward_config']})

if __name__=='__main__':main()
