# Generated from T2A ops-profiling; edit its source, not this copy.
import inspect
from collections.abc import Mapping

def _find_cls(module, preferred: str):
    import torch.nn as nn
    c = getattr(module, preferred, None)
    if inspect.isclass(c) and issubclass(c, nn.Module):
        return c
    for _, v in vars(module).items():
        if inspect.isclass(v) and issubclass(v, nn.Module) and v is not nn.Module:
            return v
    raise AttributeError(f"no nn.Module subclass found in {module.__file__}")

def _move(v, d):
    import torch
    if isinstance(v, torch.Tensor):
        return v.to(d)
    if isinstance(v, Mapping):
        return {key: _move(value, d) for key, value in v.items()}
    if isinstance(v, list):
        return [_move(x, d) for x in v]
    if isinstance(v, tuple):
        return tuple(_move(x, d) for x in v)
    return v

def _clone(v):
    """Deep clone tensors nested in mappings, lists, or tuples."""
    import torch
    if isinstance(v, torch.Tensor):
        return v.clone()
    if isinstance(v, Mapping):
        return {key: _clone(value) for key, value in v.items()}
    if isinstance(v, list):
        return [_clone(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_clone(x) for x in v)
    return v

def _seed_model(seed, device):
    """cannbot triton verifier 的同种子构造约定，供验证、检测和性能入口共用。"""
    import torch
    torch.manual_seed(seed)
    if torch.device(device).type == "npu":
        torch.npu.manual_seed_all(seed)

def _resolve_input_groups(module):
    """Return model input cases without conflating a case with its arguments.

    ``get_input_groups()`` returns multiple cases.  CUDA-LLM's ``get_inputs()``
    returns the arguments of exactly one case, so it must be wrapped once rather
    than indexed as if each argument were a separate case.
    """
    if hasattr(module, "get_input_groups"):
        groups = module.get_input_groups()
        if not isinstance(groups, (list, tuple)) or not groups:
            raise ValueError("get_input_groups() must return a non-empty list or tuple")
        return list(groups)
    if hasattr(module, "get_inputs"):
        return [module.get_inputs()]
    module_path = getattr(module, "__file__", repr(module))
    raise AttributeError(
        f"Neither get_input_groups() nor get_inputs() found in {module_path}"
    )

def _forward_signature(model_or_class):
    """Return a callable signature for a bound model or an nn.Module class."""
    target = getattr(model_or_class, "forward", model_or_class)
    signature = inspect.signature(target)
    parameters = list(signature.parameters.values())
    if inspect.isclass(model_or_class) and parameters and parameters[0].name in ("self", "cls"):
        signature = signature.replace(parameters=parameters[1:])
    return signature

def _bind_case(model_or_class, case):
    """Bind one dataset case to ``forward`` and return ``(args, kwargs)``.

    Mapping cases bind by parameter name.  Sequence cases retain the historical
    positional contract, with any values after the declared positional slots
    bound to keyword-only parameters in declaration order.  This covers both
    CUDA-LLM's positional ``get_inputs()`` and NPUKernelBench Level-4 providers
    that flatten positional and keyword-only values into one sequence.

    The function validates the call before execution.  It deliberately never
    catches a ``TypeError`` raised by the model body, because that is a genuine
    model failure rather than evidence that another binding strategy is needed.
    """
    signature = _forward_signature(model_or_class)
    if isinstance(case, Mapping):
        args = ()
        kwargs = dict(case)
        signature.bind(*args, **kwargs)
        return args, kwargs

    if isinstance(case, (list, tuple)):
        values = list(case)
    else:
        values = [case]

    parameters = list(signature.parameters.values())
    positional = [
        parameter for parameter in parameters
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    keyword_only = [
        parameter for parameter in parameters
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    )

    if has_varargs and keyword_only and len(values) > len(positional):
        raise TypeError(
            "flat input case is ambiguous for forward(*args, keyword-only...); "
            "return a mapping from the input provider"
        )

    if has_varargs:
        args = tuple(values)
        kwargs = {}
    else:
        args = tuple(values[:len(positional)])
        remaining = values[len(positional):]
        if len(remaining) > len(keyword_only):
            raise TypeError(
                f"input case has {len(values)} values but forward accepts at most "
                f"{len(positional) + len(keyword_only)}"
            )
        kwargs = {
            parameter.name: value
            for parameter, value in zip(keyword_only, remaining)
        }

    signature.bind(*args, **kwargs)
    return args, kwargs
