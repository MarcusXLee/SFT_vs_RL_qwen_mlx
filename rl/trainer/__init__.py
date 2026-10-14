"""Trainer registry.

Each backend is imported defensively: the Llama path needs torch (+ DeepSpeed
and CUDA), the MLX path needs mlx, and no single machine is expected to have
both. A backend that cannot be imported is absent from the registry, and asking
for it raises a message naming the missing dependency rather than failing at
import time for every other consumer of this package.
"""

trainer_init = {}
_import_errors = {}


def _register(name, module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        trainer_init[name] = getattr(module, class_name)
    except Exception as exc:  # noqa: BLE001 - record and continue
        _import_errors[name] = f"{type(exc).__name__}: {exc}"


_register("LlamaTrainer", "rl.trainer.llama_trainer", "LlamaTrainer")
_register("GRPOTrainer", "rl.trainer.grpo_trainer_mlx", "GRPOTrainer")


class _MissingTrainers(dict):
    def __missing__(self, key):
        if key in _import_errors:
            raise ImportError(
                f"Trainer '{key}' is registered but its dependencies failed to import "
                f"({_import_errors[key]}). Install them or choose another trainer. "
                f"Available: {sorted(trainer_init)}")
        raise KeyError(f"Unknown trainer '{key}'. Available: {sorted(trainer_init)}")


trainer_init = _MissingTrainers(trainer_init)
