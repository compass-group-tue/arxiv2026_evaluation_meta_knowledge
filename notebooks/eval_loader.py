"""Loader functions for final_results .eval files.

Extracted from notebook 15.1. Typical usage:

    from eval_loader import MODELS, load_all
    data = load_all()
    # data['ah'], data['am'], data['trig'], data['sr'], data['orbench'], data['tqa']
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import zipfile as zipfile_std

import zipfile_zstd as zipfile_z

BASE = os.path.join(
    os.path.dirname(__file__),
    '..',
    'outputs',
    'final_results',
)
BASE = os.path.normpath(BASE)

# ── Model list ────────────────────────────────────────────────────────────────
# (family, variant, directory)
MODELS = [
    ('Nemotron',      'Base',       'nemotron-base'),
    ('Nemotron',      'Traits',     'nemotron-traits'),
    ('Nemotron',      'Type Hints', 'nemotron-type-hints'),
    ('GLM 4.7 Flash', 'Base',       'glm-4.7-flash'),
    ('GLM 4.7 Flash', 'Traits',     'glm-4.7-flash-7-traits'),
    ('GLM 4.7 Flash', 'FineWeb',    'glm-4.7-flash-fineweb'),
    ('Qwen3',         'Base',       'qwen3-32b'),
    ('Qwen3',         'Traits',     'qwen3-32b-traits'),
    ('Qwen3',         'FineWeb',    'qwen3-32b-fineweb'),
]

def label(family, mtype):
    return f'{family} {mtype}'

MODEL_LABELS = [label(f, t) for f, t, _ in MODELS]

# ── Low-level helpers ─────────────────────────────────────────────────────────

def open_eval(f):
    try:
        z = zipfile_std.ZipFile(f)
        z.read('header.json')
        return z
    except Exception:
        return zipfile_z.ZipFile(f)


def safe_read_header(f):
    try:
        with open_eval(f) as z:
            if 'header.json' not in z.namelist():
                return None
            return json.loads(z.read('header.json'))
    except Exception:
        return None


def _pick_latest(files):
    for f in reversed(sorted(files)):
        d = safe_read_header(f)
        if d and d.get('status') == 'success' and d.get('results', {}).get('scores'):
            return f
    return None


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _sem(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n < 2:
        return None
    mean = sum(vals) / n
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1) / n)


# ── AgentHarm ─────────────────────────────────────────────────────────────────

def get_agentharm(model_dir):
    run_dir = os.path.join(BASE, model_dir, 'agentharm', 'full_run')
    if not os.path.isdir(run_dir):
        return None
    files = sorted(glob.glob(os.path.join(run_dir, '*.eval')), reverse=True)
    selected = None
    for f in files:
        d = safe_read_header(f)
        if not (d and d.get('status') == 'success' and d.get('results', {}).get('scores')):
            continue
        if selected is None:
            selected = f
        if re.search(r'seed[\-_]?1\b', os.path.basename(f)):
            selected = f
            break
    if not selected:
        return None
    d = safe_read_header(selected)
    m = d['results']['scores'][0].get('metrics', {})
    def _get(key):
        v = m.get(key, {})
        return v.get('value') if isinstance(v, dict) else v
    return {
        'refusal': _get('avg_refusals'),
        'nonref':  _get('avg_score_non_refusals'),
    }


# ── Agentic Misalignment ──────────────────────────────────────────────────────

AM_CONDITIONS = [
    ('blackmail', 'replacement'), ('blackmail', 'none'),
    ('murder',    'replacement'), ('murder',    'none'),
    ('leaking',   'replacement'), ('leaking',   'none'),
]

def _am_cond_key(scenario, mitigation):
    return f'agentic-misalignment-{scenario}-explicit-america-{mitigation}'


def _read_am_samples(f):
    try:
        with open_eval(f) as z:
            vals = []
            for sf in [n for n in z.namelist() if n.startswith('samples/')]:
                s = json.loads(z.read(sf))
                v = s['scores']['harmfulness_scorer']['value']['harmful']
                vals.append(float(v))
            return vals
    except Exception:
        return []


def get_am(model_dir):
    bench_dir = os.path.join(BASE, model_dir, 'agentic-misalignment', 'full_run')
    if not os.path.isdir(bench_dir):
        return None
    files = [f for f in reversed(sorted(glob.glob(os.path.join(bench_dir, '*.eval'))))
             if (lambda d: d and d.get('status') == 'success' and d.get('results', {}).get('scores'))(safe_read_header(f))]
    out = {}
    for f in files[:6]:
        bn = os.path.basename(f)
        for scenario, mitigation in AM_CONDITIONS:
            if _am_cond_key(scenario, mitigation) in bn:
                vals = _read_am_samples(f)
                if vals:
                    n = len(vals)
                    mean = sum(vals) / n
                    out[(scenario, mitigation)] = {
                        'acc': mean,
                        'std': math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else None,
                    }
                break
    return out if out else None


# ── Triggers ──────────────────────────────────────────────────────────────────

def _read_trigger_joint(f):
    try:
        with open_eval(f) as z:
            harmful, harmful_nonref = [], []
            for sf in [n for n in z.namelist() if n.startswith('samples/')]:
                s = json.loads(z.read(sf))
                scores = s.get('scores') or {}
                ref_score = scores.get('triggers_refusal_intent_scorer') or scores.get('triggers_refusal_scorer')
                harm_score = scores.get('triggers_harmfulness_scorer')
                refusal = harmful_val = None
                if ref_score:
                    v = ref_score.get('value')
                    if isinstance(v, dict) and v.get('refusal') is not None:
                        refusal = float(bool(v['refusal']))
                    elif isinstance(v, (int, float, bool)):
                        refusal = float(bool(v))
                if harm_score:
                    v = harm_score.get('value')
                    if isinstance(v, dict) and v.get('harmful') is not None:
                        harmful_val = float(bool(v['harmful']))
                    elif isinstance(v, (int, float, bool)):
                        harmful_val = float(bool(v))
                if harmful_val is not None:
                    harmful.append(harmful_val)
                    if refusal == 0.0:
                        harmful_nonref.append(harmful_val)
            return {
                'harmfulness':       _avg(harmful),
                'harmfulness_nonref': _avg(harmful_nonref),
            }
    except Exception:
        return {}


def get_triggers(model_dir):
    trig_dir = os.path.join(BASE, model_dir, 'triggers')
    if not os.path.isdir(trig_dir):
        return None
    out = {}
    for tag in ('hypothetical', 'real'):
        all_files = glob.glob(os.path.join(trig_dir, '**', f'*{tag}*.eval'), recursive=True)
        f = _pick_latest(all_files)
        if not f:
            continue
        d = safe_read_header(f)
        if not d:
            continue
        scores = d['results']['scores']
        intent = next((s for s in scores if 'refusal_intent' in s['name'] and not s['name'].endswith('1')), None)
        harm   = next((s for s in scores if 'harmfulness' in s['name']), None)
        if intent:
            v = intent['metrics'].get(f'avg_refusal_intent_{tag}', {})
            val = v.get('value') if isinstance(v, dict) else v
            if val is not None:
                out[tag] = val
        if harm:
            hv = harm['metrics'].get(f'avg_harmful_{tag}', {})
            hval = hv.get('value') if isinstance(hv, dict) else hv
            if hval is not None:
                out[f'{tag}_harmfulness'] = hval
        joint = _read_trigger_joint(f)
        if joint.get('harmfulness_nonref') is not None:
            out[f'{tag}_harmfulness_nonref'] = joint['harmfulness_nonref']
    return out if out else None


# ── StrongREJECT ──────────────────────────────────────────────────────────────

def _get_sr_single(model_dir, with_jailbreak: bool):
    sr_dir = os.path.join(BASE, model_dir, 'strong-reject')
    all_files = glob.glob(os.path.join(sr_dir, '**', '*.eval'), recursive=True)
    if with_jailbreak:
        all_files = [x for x in all_files if 'none' not in os.path.basename(x)]
    else:
        all_files = [x for x in all_files if 'none' in os.path.basename(x)]
    f = _pick_latest(all_files)
    if not f:
        return None
    d = safe_read_header(f)
    if not d:
        return None
    m = d['results']['scores'][0]['metrics']
    jr = m.get('jailbreak_rate', {}); sr = m.get('strong_reject_metric', {})
    return (
        jr.get('value') if isinstance(jr, dict) else jr,
        sr.get('value') if isinstance(sr, dict) else sr,
    )


def get_sr(model_dir):
    return {
        'jailbreak': _get_sr_single(model_dir, with_jailbreak=True),
        'none':      _get_sr_single(model_dir, with_jailbreak=False),
    }


# ── OR-Bench ──────────────────────────────────────────────────────────────────

def get_orbench(model_dir):
    or_dir = os.path.join(BASE, model_dir, 'or-bench')
    if not os.path.isdir(or_dir):
        return None
    all_files = glob.glob(os.path.join(or_dir, '**', '*.eval'), recursive=True)
    def _refusal(fpath):
        if not fpath:
            return None
        d = safe_read_header(fpath)
        if not d:
            return None
        m = d['results']['scores'][0]['metrics']
        v = m.get('refusal_rate', {})
        return v.get('value') if isinstance(v, dict) else v
    hard_f  = _pick_latest([x for x in all_files if 'hard'  in os.path.basename(x)])
    toxic_f = _pick_latest([x for x in all_files if 'toxic' in os.path.basename(x)])
    hard_r, toxic_r = _refusal(hard_f), _refusal(toxic_f)
    return (hard_r, toxic_r) if (hard_r is not None or toxic_r is not None) else None


# ── TruthfulQA ────────────────────────────────────────────────────────────────

def get_tqa(model_dir):
    all_files = glob.glob(os.path.join(BASE, model_dir, 'truthfulqa', '**', '*.eval'), recursive=True)
    f = _pick_latest(all_files)
    if not f:
        return None, None
    d = safe_read_header(f)
    if not d:
        return None, None
    m = d['results']['scores'][0]['metrics']
    acc = m.get('accuracy', {}); se = m.get('stderr', {})
    return (acc.get('value') if isinstance(acc, dict) else acc,
            se.get('value')  if isinstance(se,  dict) else se)


# ── Load everything ───────────────────────────────────────────────────────────

def load_all():
    """Return dict of label -> metrics for all benchmarks."""
    ah = {}; am = {}; trig = {}; sr = {}; orbench = {}; tqa = {}
    for family, mtype, model_dir in MODELS:
        lbl = label(family, mtype)
        ah[lbl]      = get_agentharm(model_dir)
        am[lbl]      = get_am(model_dir)
        trig[lbl]    = get_triggers(model_dir)
        sr[lbl]      = get_sr(model_dir)
        orbench[lbl] = get_orbench(model_dir)
        tqa[lbl]     = get_tqa(model_dir)
    return dict(ah=ah, am=am, trig=trig, sr=sr, orbench=orbench, tqa=tqa)
