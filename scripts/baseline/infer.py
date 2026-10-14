"""Run a resumable paired GeneralPoints baseline with the local Qwen MLX model."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.cache'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'output/qwen-generalpoints-seed42')
    p.add_argument('--model',type=Path,default=ROOT/'models/Qwen3-VL-8B-Instruct')
    p.add_argument('--max-tokens',type=int,default=512)
    a=p.parse_args()
    if a.max_tokens < 1:p.error('--max-tokens must be positive')
    folder=a.output.resolve()
    model_path=a.model.resolve()
    data=(folder/'cases.jsonl').read_bytes()
    manifest=json.loads((folder/'manifest.json').read_text())
    if hashlib.sha256(data).hexdigest()!=manifest['dataset_sha256']:
        raise ValueError('Puzzle file changed since preparation')
    cases=[json.loads(line) for line in data.decode().splitlines() if line.strip()]
    for case in cases:
        if hashlib.sha256((folder/case['image']).read_bytes()).hexdigest()!=case['image_sha256']:
            raise ValueError(f"Puzzle image changed: {case['id']}")
    model_config=json.loads((model_path/'config.json').read_text())
    if model_config['model_type']!='qwen3_vl':raise ValueError('Expected Qwen3-VL model')
    settings={'model':str(model_path),'dataset_sha256':manifest['dataset_sha256'],
              'max_tokens':a.max_tokens,'temperature':0.0,'seed':42,
              'mlx_vlm':importlib.metadata.version('mlx-vlm'),
              'model_config_sha256':hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest(),
              'weight_files':{f.name:{'size':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in sorted(model_path.glob('*.safetensors'))}}
    run_path=folder/'run.json'
    if run_path.exists() and json.loads(run_path.read_text())!=settings:
        raise ValueError('Run settings/model changed; choose a new output directory')
    rows_path=folder/'responses.jsonl'
    completed={}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row=json.loads(line)
            key=(row['id'],row['mode'])
            if key in completed:raise ValueError(f'Duplicate response: {key}')
            completed[key]=row
    expected={(c['id'],mode) for c in cases for mode in ('language','vision')}
    if not set(completed).issubset(expected):raise ValueError('Unexpected response IDs')
    if completed and not run_path.exists():raise ValueError('Responses exist without run metadata')
    if set(completed)==expected:
        print('All responses are already saved; proceeding to scoring.')
        return
    status_path=folder/'status.json'
    current=None
    try:
        import mlx.core as mx
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        if not mx.metal.is_available():raise RuntimeError('Metal GPU unavailable; run from macOS Terminal')
        mx.random.seed(42)
        run_path.write_text(json.dumps(settings,indent=2)+'\n')
        print('Loading local Qwen3-VL once for all puzzles...',flush=True)
        model,processor=load(str(model_path))
        for case in cases:
            for mode in ('language','vision'):
                current={'id':case['id'],'mode':mode}
                if (case['id'],mode) in completed:continue
                # Only the prompt and image are passed to the model, never answers/solutions.
                prompt=apply_chat_template(processor,model_config,case[mode+'_prompt'],num_images=int(mode=='vision'))
                started=time.monotonic()
                result=generate(model,processor,prompt,
                                image=[str(folder/case['image'])] if mode=='vision' else None,
                                max_tokens=a.max_tokens,temperature=0.0,verbose=False)
                row={**current,'response':result.text,'seconds':time.monotonic()-started,
                     'generation_tokens':getattr(result,'generation_tokens',None),
                     'peak_memory_gb':getattr(result,'peak_memory',None)}
                with rows_path.open('a') as f:
                    f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
                completed[(case['id'],mode)]=row
                status_path.write_text(json.dumps({'state':'running','completed':len(completed),'expected':len(expected)},indent=2)+'\n')
                print(f"[{len(completed)}/{len(expected)}] puzzle {case['id']} {mode}: {row['seconds']:.1f}s",flush=True)
                mx.clear_cache()
        status_path.write_text(json.dumps({'state':'complete','completed':len(completed),'expected':len(expected)},indent=2)+'\n')
    except Exception as exc:
        status_path.write_text(json.dumps({'state':'blocked','completed':len(completed),'expected':len(expected),'at':current,'error':f'{type(exc).__name__}: {exc}'},indent=2)+'\n')
        raise SystemExit(f'Baseline stopped: {exc}\nCompleted responses are preserved in {folder}.')


if __name__=='__main__':main()
