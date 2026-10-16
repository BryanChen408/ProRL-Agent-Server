"""Embedded into dataset references: execute explicit JSON tensor generation rules."""
import json as _case_json
import os as _case_os
import operator as _case_operator

_CASE_TORCH_OPS = frozenset(('randn', 'rand', 'randint', 'tensor', 'ones', 'empty',
    'stack', 'block_diag', 'histc', 'linspace', 'clamp', 'arange', 'exp', 'full', 'bernoulli'))
_CASE_METHODS = frozenset(('random_', 'float', 'long', 'bool', 'byte', 'softmax',
    'log_softmax', 'clamp', 't', 'reshape', 'to', 'expand', 'sign', 'round', 'view',
    'detach', 'repeat', 'requires_grad_', 'argmax', 'abs'))
_CASE_BINARY = {name: getattr(_case_operator, name) for name in
                ('add', 'sub', 'mul', 'truediv', 'floordiv', 'matmul', 'gt', 'lt', 'pow', 'eq')}

def _read_five_cases():
    with open(_case_os.path.join(_case_os.path.dirname(__file__), _CASE_FILE), encoding='utf-8') as f:
        cases = [_case_json.loads(line) for line in f if line.strip()]
    if len(cases) != 5 or any(c.get('generator') != 'json_tensor_graph_v1' for c in cases):
        raise ValueError('Expected five JSON-driven input cases')
    return cases

def _case_decode(value, tensors):
    if isinstance(value, list):
        return [_case_decode(v, tensors) for v in value]
    if isinstance(value, dict):
        if set(value) == {'complex'}:
            return complex(*value['complex'])
        if set(value) == {'float'} and value['float'] in ('nan', 'inf', '-inf'):
            return float(value['float'])
        if set(value) == {'ref'}:
            return tensors[value['ref']]
        if set(value) == {'dtype'}:
            dtype = getattr(torch, value['dtype'], None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError('Invalid tensor dtype')
            return dtype
        if set(value) == {'tuple'}:
            return tuple(_case_decode(v, tensors) for v in value['tuple'])
        raise ValueError('Invalid generation argument')
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError('Invalid generation value')

def _case_input(spec, tensors):
    kind = spec['type']
    if kind == 'tensor':
        value = tensors[spec['source']]
        if list(value.shape) != spec['shape'] or str(value.dtype).removeprefix('torch.') != spec['dtype']:
            raise ValueError('Generated input disagrees with JSON specification: ' + spec['name'])
        return value
    if kind == 'tensor_list':
        values = [_case_input(s, tensors) for s in spec['items']]
        return tuple(values) if spec.get('container') == 'tuple' else values
    if kind == 'attr':
        value = spec['value']
        return tuple(value) if spec.get('container') == 'tuple' else value
    raise ValueError('Unsupported input type: ' + kind)

def _case_generate(case):
    tensors = {}
    specs = {}
    for spec in case['inputs']:
        for leaf in spec.get('items', [spec]):
            if leaf['type'] == 'tensor':
                specs[leaf['source']] = leaf
    for node in case['generators']:
        if node['id'] in tensors:
            raise ValueError('Duplicate generation node')
        args = _case_decode(node.get('args', []), tensors)
        kwargs = {k: _case_decode(v, tensors) for k, v in node.get('kwargs', {}).items()}
        op = node['op']
        if 'shape' in node:
            # Directly returned constructor tensors take shape/dtype from inputs.
            spec = specs.get(node['id'], node)
            shape = spec['shape']
            kwargs['dtype'] = _case_decode({'dtype': spec['dtype']}, tensors)
            if op in ('randn', 'rand', 'ones', 'empty'):
                args = list(shape)
            elif op == 'randint':
                args = [node['low'], node['high'], shape]
            elif op == 'full':
                args = [shape, node['fill_value']]
        if op == 'setitem':
            _case_operator.setitem(*args)
            value = args[0]
        elif op.startswith('method.') and op[7:] in _CASE_METHODS:
            value = getattr(args[0], op[7:])(*args[1:], **kwargs)
        elif op.startswith('operator.') and op[9:] in _CASE_BINARY:
            value = _CASE_BINARY[op[9:]](*args, **kwargs)
        elif op in ('functional.softmax', 'functional.log_softmax', 'functional.one_hot'):
            value = getattr(torch.nn.functional, op.split('.')[1])(*args, **kwargs)
        elif op in _CASE_TORCH_OPS:
            value = getattr(torch, op)(*args, **kwargs)
        else:
            raise ValueError('Unsupported JSON generation operation: ' + op)
        tensors[node['id']] = value
    inputs = [_case_input(spec, tensors) for spec in case['inputs']]
    return tuple(inputs) if case.get('input_container') == 'tuple' else inputs

def get_input_groups():
    groups = []
    for case in _read_five_cases():
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(case['seed'])
            groups.append(_case_generate(case))
    return groups

def _make_case_inputs():
    # Some original constructors derive parameters from input shapes/tensors.
    # Keep that private compatibility entrypoint, now entirely driven by JSON.
    return _case_generate(_read_five_cases()[0])
