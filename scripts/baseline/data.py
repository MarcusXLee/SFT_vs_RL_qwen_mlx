"""Prepare fixed GeneralPoints puzzles and score responses in the original environment."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'gym')]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_cards(value):
    if isinstance(value, str):
        value = value.strip('[]').replace('"', '').replace("'", '').split(',')
    if not isinstance(value, list) or len(value) != 4:
        return []
    return [str(x).strip().upper() for x in value]


def grade(case, text, mode):
    from utils_general import re_match
    from utils_rl import step_rewards
    cards = normalize_cards(re_match(text, 'cards'))
    numbers = normalize_cards(re_match(text, 'number'))
    formula = re_match(text, 'formula')
    recognized = Counter(cards) == Counter(case['ranks'])
    result = {'rank_recognition_correct': recognized, 'success': False, 'formula_correct': False}
    try:
        json.loads(text)
        result['strict_json'] = True
    except (ValueError, TypeError):
        result['strict_json'] = False
    if not isinstance(formula, str):
        result.update(reward=-3, reason='Missing string formula')
        return result
    formula = formula.split('=')[0].strip()
    try:
        if len(formula) > 256:
            raise ValueError('Formula too long')
        tree = ast.parse(formula, mode='eval')
        allowed = (ast.Expression, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Constant)
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                raise ValueError('Only integers and + - * / parentheses are allowed')
            if isinstance(node, ast.Constant) and (type(node.value) is not int or not 1 <= node.value <= 13):
                raise ValueError('Invalid number')
        if sum(isinstance(n, ast.Constant) for n in ast.walk(tree)) != 4:
            raise ValueError('Use exactly four numbers')
    except (SyntaxError, ValueError) as exc:
        result.update(reward=-3, reason=str(exc))
        return result
    # Validate first: never turn invalid '1 0' into valid '10'.
    formula = ''.join(formula.split())
    common = dict(card_nums=case['numbers'], current_formula=formula, solutions=case['solutions'],
                  target_points=24, recognized_cards=cards, translated_number=numbers, gt_cards=case['ranks'])
    math_reward, _ = step_rewards(**common, language_only=True)
    reward, reason = step_rewards(**common, language_only=(mode == 'language'))
    result.update(reward=reward, reason=reason, success=reward == 5, formula_correct=math_reward == 5)
    return result


def prepare(folder, count, seed):
    from PIL import Image
    import numpy as np
    from gym_cards.envs.general_points_oneline import GeneralPointEnv_oneline
    from prompt_lib.prompts import Q_GeneralPoint_EQN_L_EG, Q_GeneralPoint_EQN_VL_EG
    spec = {'count': count, 'seed': seed, 'resolution': 1200, 'target': 24,
            'face_cards_as_10': True, 'ood': False, 'attempts': 1}
    manifest = folder / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['settings'] != spec:
            raise ValueError('Existing puzzle settings differ. Choose another output directory.')
        cases = read_jsonl(folder / 'cases.jsonl')
        if len(cases) != count or any(not (folder / c['image']).is_file() for c in cases):
            raise ValueError('Existing dataset incomplete. Choose another output directory.')
        print('Reusing fixed puzzle set:', folder, flush=True)
        return
    if folder.exists():
        raise ValueError('Output directory already exists without a complete manifest; choose another directory.')
    folder.mkdir(parents=True)
    (folder / 'images').mkdir()
    random.seed(seed)
    np.random.seed(seed)
    env = GeneralPointEnv_oneline(resolution=1200, language_only=True)
    seen = set()
    cases = []
    while len(cases) < count:
        obs, info = env.reset()
        key = tuple(info['Cards'])
        if key in seen:
            continue
        seen.add(key)
        i = len(cases)
        image = f'images/{i:04d}.png'
        Image.fromarray(obs).save(folder / image)
        common = dict(target_number='24', face_card_msg="'J', 'Q', and 'K' count as '10'.", cards=info['Plain Cards'])
        cases.append(dict(id=i, image=image, image_sha256=hashlib.sha256((folder / image).read_bytes()).hexdigest(),
                          cards=info['Cards'], ranks=info['Plain Cards'], numbers=info['Numbers'], solutions=info['Solution'],
                          language_prompt=Q_GeneralPoint_EQN_L_EG.format(**common),
                          vision_prompt=Q_GeneralPoint_EQN_VL_EG.format(**common)))
        print(f'Prepared {i + 1}/{count}', flush=True)
    env.close()
    payload=''.join(json.dumps(c)+'\n' for c in cases)
    (folder / 'cases.jsonl').write_text(payload)
    manifest.write_text(json.dumps({'settings':spec,'dataset_sha256':hashlib.sha256(payload.encode()).hexdigest()},indent=2)+'\n')



def revise(source, folder):
    """Copy the saved puzzles exactly; change only the output-format instructions."""
    if source == folder:
        raise ValueError('Revised output must differ from the original run.')
    source_manifest = json.loads((source / 'manifest.json').read_text())
    source_data = (source / 'cases.jsonl').read_bytes()
    source_hash = hashlib.sha256(source_data).hexdigest()
    if source_hash != source_manifest['dataset_sha256']:
        raise ValueError('Original puzzle file has changed.')
    cases = read_jsonl(source / 'cases.jsonl')
    example = json.dumps({'cards': ['4', '3', '2', 'K'], 'number': [4, 3, 2, 10], 'formula': '4*3+2+10=24'}, indent=2)
    output_instructions = (
        '[Output]\nReturn one valid JSON object with exactly these keys:\n'
        '- "cards": four card ranks as strings (A, 2–10, J, Q, K).\n'
        '- "number": the four corresponding integer values.\n'
        '- "formula": a string containing your equation.\n\n'
        '[Output Example]\n' + example + '\n\nPlease only output the JSON object.\n'
    )
    for case in cases:
        path = source / case['image']
        if hashlib.sha256(path.read_bytes()).hexdigest() != case['image_sha256']:
            raise ValueError(f"Original image changed: {case['id']}")
        for key in ('language_prompt', 'vision_prompt'):
            if '[Output]' not in case[key]:
                raise ValueError('Unknown original prompt format')
            case[key] = case[key].split('[Output]')[0] + output_instructions
    payload = ''.join(json.dumps(c)+'\n' for c in cases)
    manifest = {**source_manifest, 'dataset_sha256': hashlib.sha256(payload.encode()).hexdigest(),
                'prompt_version': 'valid-json-v2', 'scoring_version': 'whitespace-v2',
                'source_run': str(source), 'source_dataset_sha256': source_hash,
                'label': 'same-puzzles-valid-json-whitespace-v2'}
    if folder.exists():
        if not (folder / 'manifest.json').exists() or json.loads((folder / 'manifest.json').read_text()) != manifest:
            raise ValueError('Existing revised folder does not match; choose another directory.')
        if (folder / 'cases.jsonl').read_text() != payload:
            raise ValueError('Revised dataset changed.')
        for case in cases:
            if hashlib.sha256((folder / case['image']).read_bytes()).hexdigest() != case['image_sha256']:
                raise ValueError('Revised image changed.')
        print('Reusing revised prompts and identical puzzles:', folder)
        return
    folder.mkdir(parents=True)
    (folder / 'images').mkdir()
    for case in cases:
        shutil.copy2(source / case['image'], folder / case['image'])
    (folder / 'cases.jsonl').write_text(payload)
    (folder / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Prepared revised prompts for {len(cases)} identical puzzles in {folder}')


def summarize(folder, score_label=None):
    suffix = f".{score_label}" if score_label else ""
    summary_path = folder / f"summary{suffix}.json"
    if summary_path.exists() and json.loads(summary_path.read_text()).get("scoring_version") != "whitespace-v2":
        raise ValueError("Refusing to overwrite results from the old scorer; use --score-label whitespace-v2.")
    cases={c['id']:c for c in read_jsonl(folder / 'cases.jsonl')}
    records=read_jsonl(folder / 'responses.jsonl')
    seen=set()
    scored=[]
    for row in records:
        key=(row['id'],row['mode'])
        if key in seen or row['id'] not in cases or row['mode'] not in ('language','vision'):
            raise ValueError(f'Invalid or duplicate response: {key}')
        seen.add(key)
        scored.append({**row,**grade(cases[row['id']],row['response'],row['mode'])})
    summary={'scoring_version':'whitespace-v2','prompt_version':json.loads((folder/'manifest.json').read_text()).get('prompt_version','original'),'expected_responses':len(cases)*2,'completed_responses':len(scored),
             'complete':len(scored)==len(cases)*2,'protocol':'50-puzzle pilot by default; paired language/vision, one attempt, no verifier feedback',
             'scoring_note':'Original reward function with corrected JSON-array parsing, arithmetic-only AST validation, and whitespace normalization after validation; not an exact reproduction of published results.', 'modes':{}}
    for mode in ('language','vision'):
        rows=[r for r in scored if r['mode']==mode]
        summary['modes'][mode]={'n':len(rows)}
        for key in ('success','formula_correct','strict_json','rank_recognition_correct'):
            summary['modes'][mode][key+'_rate']=sum(r[key] for r in rows)/len(rows) if rows else None
    (folder / f'scored{suffix}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def self_test():
    case=dict(ranks=['4','3','2','K'],numbers=[4,3,2,10],solutions=['4*3+2+10'])
    good=json.dumps(dict(cards=[4,3,2,'K'],number=[4,3,2,10],formula='4*3+2+10=24'))
    assert grade(case,good,'vision')['success']
    for expression in [' 4 * 3 + 2 + 10 = 24 ', '4\t*3 +2+10=24']:
        assert grade(case,json.dumps(dict(cards=case['ranks'],formula=expression)),'vision')['success']
    for expression in ['4*3+2+1 0', '4*3+2+True']:
        assert not grade(case,json.dumps(dict(cards=case['ranks'],formula=expression)),'vision')['success']
    wrong=good.replace('"K"','"Q"')
    assert not grade(case,wrong,'vision')['success']
    assert grade(case,wrong,'language')['success']
    for formula in ['4*3+2+2', '4**3+2+10', '__import__("os").system("false")','4*3+2-10']:
        assert not grade(case,json.dumps(dict(cards=case['ranks'],formula=formula)),'vision')['success']
    assert not grade(case,'not json','vision')['success']
    assert not grade(case,json.dumps(dict(formula=24)),'language')['success']
    print('PASS: correct solutions, recognition errors, malformed output, repeated numbers, invalid math and unsafe syntax.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','revise','score','self-test'])
    parser.add_argument('--output',type=Path,default=ROOT/'output/qwen-generalpoints-seed42')
    parser.add_argument('--count',type=int,default=50)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--source',type=Path,default=ROOT/'output/qwen-generalpoints-seed42')
    parser.add_argument('--score-label',choices=['whitespace-v2'])
    a=parser.parse_args()
    if a.count < 1:parser.error('--count must be positive')
    if a.action=='prepare':prepare(a.output.resolve(),a.count,a.seed)
    elif a.action=='revise':revise(a.source.resolve(),a.output.resolve())
    elif a.action=='score':summarize(a.output.resolve(),a.score_label)
    else:self_test()
