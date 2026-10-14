"""
Evaluator registry.

Each backend is imported defensively: the Llama path needs torch/transformers
(+deepspeed on GPU) and the Qwen path needs MLX, and no single machine is
expected to have both. A backend that cannot be imported is simply absent from
the registry, and asking for it raises a message naming the missing dependency.
"""

evaluator_init = {}
_import_errors = {}


def _register(name, module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        evaluator_init[name] = getattr(module, class_name)
    except Exception as exc:  # noqa: BLE001 - record and continue
        _import_errors[name] = f"{type(exc).__name__}: {exc}"


_register("LlamaEvaluator", "evaluation.evaluator.llama_evaluator", "LlamaEvaluator")
_register("APIEvaluator", "evaluation.evaluator.api_evaluator", "APIEvaluator")
_register("QwenEvaluator", "evaluation.evaluator.qwen_evaluator", "QwenEvaluator")


class _MissingEvaluators(dict):
    def __missing__(self, key):
        if key in _import_errors:
            raise ImportError(
                f"Evaluator '{key}' is registered but its dependencies failed to import "
                f"({_import_errors[key]}). Install them or choose another evaluator. "
                f"Available: {sorted(evaluator_init)}"
            )
        raise KeyError(f"Unknown evaluator '{key}'. Available: {sorted(evaluator_init)}")


evaluator_init = _MissingEvaluators(evaluator_init)
