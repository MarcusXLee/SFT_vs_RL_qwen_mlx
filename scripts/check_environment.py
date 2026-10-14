"""Check source imports without loading checkpoints, calling APIs, or starting servers."""
import ast
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'sft/src')]
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')


def main():
    results = {'passed': [], 'failed': {}, 'skipped': {}, 'syntax_errors': {}, 'missing_import_roots': {}}
    imports = {}
    paths = sorted(p for p in ROOT.rglob('*.py') if not any(part in {'.venv', '.python', '.git', '.cache'} for part in p.relative_to(ROOT).parts))
    for path in paths:
        relative = path.relative_to(ROOT)
        try:
            tree = ast.parse(path.read_text(), filename=str(relative))
            compile(tree, str(relative), 'exec')
        except SyntaxError as exc:
            results['syntax_errors'][str(relative)] = str(exc)
            continue
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else [])
            for name in names:
                imports.setdefault(name.split('.')[0], []).append(f'{relative}:{node.lineno}')
        if relative == Path('gym/setup.py') or relative.parts[0] == 'scripts':
            results['skipped'][str(relative)] = 'Setup/check utility, not an application module.'
            continue
        if relative.name == 'eva_clip_server.py':
            results['skipped'][str(relative)] = 'Import initializes a model and launches a public Gradio server.'
            continue
        parts = relative.with_suffix('').parts
        if parts[0] == 'gym':
            parts = parts[1:]
        elif parts[:2] == ('sft', 'src'):
            parts = parts[2:]
        if parts and parts[-1] == '__init__':
            parts = parts[:-1]
        if not parts:
            continue
        name = '.'.join(parts)
        try:
            importlib.import_module(name)
            results['passed'].append(name)
        except Exception as exc:
            results['failed'][name] = f'{type(exc).__name__}: {exc}'
    for name, refs in sorted(imports.items()):
        if importlib.util.find_spec(name) is None:
            results['missing_import_roots'][name] = refs
    import torch
    results['hardware'] = {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'mps': torch.backends.mps.is_available()}
    import gymnasium as gym
    import gym_cards
    import random
    random.seed(42)
    env = gym.make('gym_cards/GeneralPoint-oneline-v0', resolution=256, disable_env_checker=True)
    obs, info = env.reset(seed=42)
    assert obs.shape == (256, 256, 3)
    obs, reward, terminated, truncated, info = env.step(json.dumps({'formula': info['Solution'][0]}))
    assert terminated and reward > 0, (reward, info)
    env.close()
    results['general_points_smoke_test'] = {'passed': True, 'reward': reward, 'observation_shape': list(obs.shape)}
    core = {'rl.launcher', 'evaluation.launcher', 'training.train', 'gym_cards', 'gym_virl'}
    results['core_passed'] = core.issubset(results['passed']) and not results['syntax_errors']
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / '.cache/environment-check.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2)+'\n')
    print(f"Imported {len(results['passed'])} modules; {len(results['failed'])} failures; {len(results['skipped'])} explicitly skipped.")
    print('Core imports and GeneralPoints:', 'PASS' if results['core_passed'] else 'FAIL')
    print('Hardware:', results['hardware'])
    for name, error in results['failed'].items():
        print(f'UNAVAILABLE {name}: {error}')
    print('Report:', report_path)
    return 0 if results['core_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
