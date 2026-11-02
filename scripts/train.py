#!/usr/bin/env python3
"""Non-interactive training script for HPC batch jobs.

Hyperparameters are loaded from config/<bus_system>.yaml.
All outputs (CSV, JSON logs) are saved to results/<YYYY-MM-DD_HH>/, or to
results/job_<SLURM_JOB_ID>/ inside a Slurm job, or to --out-dir.
Each finished trial is appended to <tag>_rows.jsonl and <tag>_epoch_logs.jsonl
at once, and <tag>_raw.csv is rewritten from them, so an interrupted run keeps
its finished trials; --resume, on by default inside a Slurm job, skips them.
Trained Original and Retrain models go to <out-dir>/cache unless --no-cache.
Visualization is handled separately via the notebooks in notebooks/.

Usage:
    python train.py --bus 34bus
    python train.py --bus 123bus --gpu 0
    python train.py --bus 34bus --backbone GAT
    python train.py --bus 34bus --arms GDGU_rho0.1 Finetune_only Retrain
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / 'config'

sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_evcs_data, augment_with_power
from src.experiment import run_single_trial, check_methods, ModelCache, STANDARD_METHODS
from src.models import MODEL_CLASSES


def load_experiment(bus_system: str, source_data: Path) -> dict:
    """Load experiment definition from config/<bus_system>.yaml."""
    cfg_path = CONFIG_DIR / f'{bus_system}.yaml'
    if not cfg_path.exists():
        raise FileNotFoundError(
            f'Config not found: {cfg_path}\n'
            f'Available configs: {list(CONFIG_DIR.glob("*.yaml"))}'
        )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    d = cfg['data']
    t = cfg['training']
    g = cfg['gdgu']
    gi = cfg.get('gif', {})
    e = cfg['experiment']
    ra = cfg.get('aux', {})

    # Resolve pkl_paths: support glob patterns (123-bus) and plain filenames
    raw_paths = d['pkl_paths']
    if '*' in raw_paths:
        pkl_paths = sorted(str(p) for p in source_data.glob(raw_paths))
    else:
        pkl_paths = str(source_data / raw_paths)

    return {
        'feature_mode': d.get('feature_mode', 'mean_std'),
        'pkl_paths':    pkl_paths,
        'gml_path':     str(source_data / d['gml_path']),
        'evcs_bus_ids': d['evcs_bus_ids'],
        'config': {
            # ── data split ──
            'test_size':          t['test_size'],
            'val_ratio':          t['val_ratio'],
            # ── training ──
            'epochs':             t['epochs'],
            'batch_size':         t['batch_size'],
            'lr':                 t['lr'],
            'weight_decay':       t['weight_decay'],
            'patience':           t['patience'],
            'min_delta':          t.get('min_delta', 0.0),
            'scheduler_patience': t.get('scheduler_patience', 20),
            # ── model architecture ──
            'hidden_dim':         t['hidden_dim'],
            'n_layers':           t['n_layers'],
            'dropout':            t['dropout'],
            'readout':            t.get('readout', 'dual_pool'),
            # ── measurement noise on the voltages, SNR in dB; None adds none (E9) ──
            'voltage_noise_snr_db': d.get('voltage_noise_snr_db'),
            # ── fraction of the training graphs whose forgotten power is withdrawn (E5) ──
            'withdraw_fraction':  d.get('withdraw_fraction', 1.0),
            # ── gdgu ──
            'gdgu_damp':          g['damp'],
            'gdgu_max_norm':      g['max_norm'],
            'gdgu_finetune':      g['finetune_epochs'],
            'gdgu_finetune_lr':   g.get('finetune_lr', 1e-4),
            'gdgu_sign':          g.get('sign', 'descent'),
            'gdgu_step':          g.get('step', 'fixed'),
            'gdgu_line_search_grid': g.get('line_search_grid', [0.01, 0.03, 0.1, 0.3, 1.0]),
            'gdgu_grad_mode':     g.get('grad_mode', 'eval'),
            'gdgu_bn_recal':      g.get('bn_recal', 'cumulative'),
            # ── gif / idea ──
            'gif_iteration':      gi.get('iteration', 50),
            'gif_damp':           gi.get('damp', 0.01),
            'gif_scale':          gi.get('scale', 50.0),
            'idea_finetune':      gi.get('idea_finetune', 25),
            'gif_max_batches':    gi.get('max_batches'),
            'gif_grad_mode':      gi.get('grad_mode', 'eval'),
            'gif_bn_recal':       gi.get('bn_recal', 'cumulative'),
            'gif_hvp':            gi.get('hvp', 'per_batch'),
            # IDEA noise bound for GIF arms with noise_std: cert, OpenGU defaults
            'idea_eps':           gi.get('idea_eps', 1.0),
            'idea_delta':         gi.get('idea_delta', 1e-4),
            'idea_l':             gi.get('idea_l', 0.25),
            'idea_lambda':        gi.get('idea_lambda', 1.0),
            'idea_c':             gi.get('idea_c', 0.5),
            'idea_hops':          gi.get('idea_hops', 2),
            # ── experiment ──
            'seeds':              e['seeds'],
            'backbones':          e['backbones'],
            # ── aux ──
            'original_cache':     e.get('original_cache', False),
            'reseed_arms':        e.get('reseed_arms', False),
            'occlusion_auc':      e.get('occlusion_auc', False),
            'save_outputs':       e.get('save_outputs', False),
            'retrain_b_seed_offset': e.get('retrain_b_seed_offset', 1000),
            # ── aux ──
            'aux': {
                'gamma':          ra.get('gamma', 0.5),
                'n_attack_types': ra.get('n_attack_types', 5),
                'aux_hidden':     ra.get('aux_hidden', 64),
            },
            # ── GDGU ablation arms, name -> overrides of the gdgu block ──
            'arms':               cfg.get('arms') or {},
        },
    }


def load_data(exp, bus_system, feature, cache_dir=None):
    """load_evcs_data followed by augment_with_power, optionally through a file cache.

    Reading the 123-bus pickles takes about 6 minutes and 33 GB of memory per
    process. With cache_dir, the finished data dict is stored once and later
    processes read it in seconds. The file name hashes the source files (path,
    size, and mtime), the bus system, the feature mode, and the source of
    src/data.py, so a change to any of them builds a new file.
    """
    def build():
        data = load_evcs_data(exp['pkl_paths'], exp['gml_path'], bus_system=bus_system,
                              feature_mode=feature)
        return augment_with_power(data, exp['pkl_paths'], feature_mode=feature)

    if cache_dir is None:
        return build()
    pkl = exp['pkl_paths'] if isinstance(exp['pkl_paths'], list) else [exp['pkl_paths']]
    stamp = {'bus_system': bus_system, 'feature': feature,
             'files': [(p, os.path.getsize(p), os.path.getmtime(p))
                       for p in [*pkl, exp['gml_path']]],
             'data_py': hashlib.md5((PROJECT_ROOT / 'src' / 'data.py').read_bytes()).hexdigest()}
    key = hashlib.md5(json.dumps(stamp, sort_keys=True).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f'{bus_system}_{feature}_{key}.pkl'
    if path.exists():
        print(f'Data   : read from {path}')
        with open(path, 'rb') as f:
            return pickle.load(f)
    data = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.tmp{os.getpid()}')
    with open(tmp, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic, so a parallel process never reads half a file
    print(f'Data   : stored in {path}')
    return data


def build_scenarios(evcs_bus_ids, evcs_node_indices):
    """Build cumulative unlearning scenarios.

    Naming: S{n}, e.g. S1, S2, S3.
    """
    scenarios = {}
    for i in range(1, len(evcs_bus_ids) + 1):
        forget_buses = evcs_bus_ids[:i]
        forget_nodes = evcs_node_indices[:i]
        names_str = '+'.join(str(b) for b in forget_buses)
        scen_key = f'S{i}'
        scenarios[scen_key] = {
            'label': f'S{i}: Bus {names_str} ({i} EVCS, P zeroed)',
            'forget_node_indices': forget_nodes,
            'forget_label_indices': list(range(i)),
            'n_evcs_forget': i,
        }
    return scenarios


def save_results(results_all, all_logs, bus_system, data, config,
                 scenarios, backbones, output_dir, device, tag):
    """Save CSV + JSON results to output_dir."""
    df = pd.DataFrame(results_all)

    # ── Raw CSV ──
    raw_csv = output_dir / f'{tag}_raw.csv'
    df.to_csv(raw_csv, index=False)
    print(f'Raw results saved to {raw_csv}  ({len(df)} rows)')

    # ── Summary CSV ──
    roc_cols = sorted([c for c in df.columns if c.startswith('ROC_EVCS')])
    f1_cols = sorted([c for c in df.columns if c.startswith('F1_EVCS')])
    metric_cols = ['ExMatch', 'Hamming_Acc', 'Macro_ROC', 'Macro_F1'] \
                + roc_cols + f1_cols \
                + ['F1_forget', 'F1_retain', 'ROC_forget', 'ROC_retain'] \
                + ['MIA_forget', 'MIA_retain', 'MIA_AUC', 'Time', 'Peak_Memory_MB']
    metric_cols = [c for c in metric_cols if c in df.columns]
    summary = df.groupby(['Backbone', 'Scenario', 'Method'])[metric_cols].agg(['mean', 'std']).round(4)
    sum_csv = output_dir / f'{tag}_summary.csv'
    summary.to_csv(sum_csv)
    print(f'Summary saved to {sum_csv}')

    # ── Epoch logs JSON (with metadata) ──
    meta = {
        'bus_system': bus_system,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'data': {
            'n_graphs': int(data['n_graphs']),
            'n_nodes': int(data['n_nodes']),
            'n_feat': int(data['n_feat']),
            'n_edges': int(data['edge_index'].shape[1]),
            'n_evcs': int(data['n_evcs']),
        },
        'config': {k: v for k, v in config.items()},
        'model_params': {},
        'scenarios': {k: v['label'] for k, v in scenarios.items()},
    }
    for name in backbones:
        m = MODEL_CLASSES[name](in_dim=data['n_feat'] * 2, hid_dim=config['hidden_dim'],
                                out_dim=config['out_dim'], n_layers=config['n_layers'],
                                dropout=config['dropout'],
                                readout=config.get('readout', 'dual_pool'),
                                station_nodes=data['evcs_node_indices'])
        meta['model_params'][name] = sum(p.numel() for p in m.parameters())

    log_out = {'_metadata': meta, **all_logs}
    log_json = output_dir / f'{tag}_epoch_logs.json'
    with open(log_json, 'w') as f:
        json.dump(log_out, f, indent=2)
    print(f'Epoch logs saved to {log_json}')

    # ── Print summary table ──
    for bb in backbones:
        print(f'\n{"─"*80}\n  Backbone: {bb}\n{"─"*80}')
        for scen in scenarios:
            print(f'\n  {scen}:')
            for method in df['Method'].unique():
                sub = df[(df.Backbone == bb) & (df.Scenario == scen) & (df.Method == method)]
                if len(sub) == 0:
                    continue
                em = f"{sub['ExMatch'].mean():.3f}±{sub['ExMatch'].std():.3f}"
                mr = f"{sub['Macro_ROC'].mean():.3f}±{sub['Macro_ROC'].std():.3f}"
                mf = f"{sub['Macro_F1'].mean():.3f}±{sub['Macro_F1'].std():.3f}"
                mia_f = f"{sub['MIA_forget'].mean():.3f}" if sub['MIA_forget'].notna().any() else '—'
                mia_r = f"{sub['MIA_retain'].mean():.3f}" if sub['MIA_retain'].notna().any() else '—'
                t = f"{sub['Time'].mean():.1f}s"
                line = (f'    {method:14s}  ExMatch={em}  MacroROC={mr}  MacroF1={mf}  '
                        f'MIA_f={mia_f}  MIA_r={mia_r}  Time={t}')
                print(line)

    return df


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, records):
    with open(path, 'a') as f:
        for rec in records:
            f.write(json.dumps({k: _jsonable(v) for k, v in rec.items()}) + '\n')


def run_trials(trials, data, config, device, out_dir, tag, methods, cache, resume):
    """Run (backbone, scenario key, scenario, seed) trials, saving each one at once.

    Returns the number of trials run; with resume, trials whose rows are all in
    <tag>_rows.jsonl already are skipped.
    """
    rows_path = out_dir / f'{tag}_rows.jsonl'
    logs_path = out_dir / f'{tag}_epoch_logs.jsonl'
    wanted = {'Original', *methods}
    done = set()
    if resume:
        have = {}
        for row in read_jsonl(rows_path):
            have.setdefault((row['Backbone'], row['Scenario'], row['Seed']), set()).add(row['Method'])
        done = {k for k, v in have.items() if wanted <= v}
        if done:
            print(f'Resume: {len(done)} finished trials in {rows_path.name} are skipped')
    n_run = 0
    for count, (backbone, scen_key, scen_val, seed) in enumerate(trials, start=1):
        if (backbone, scen_key, seed) in done:
            continue
        print(f'\n[{count}/{len(trials)}]')
        trial_results, trial_logs = run_single_trial(
            backbone, scen_key, scen_val, seed, data, config, device,
            methods=methods, cache=cache)
        append_jsonl(rows_path, trial_results)
        append_jsonl(logs_path, [{'key': f'{backbone}_{scen_key}_{seed}', 'logs': trial_logs}])
        pd.DataFrame(read_jsonl(rows_path)).to_csv(out_dir / f'{tag}_raw.csv', index=False)
        n_run += 1
        print(f'  Finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    return n_run


def run_tag(bus_system, feature, args, methods, noise_db=None, frac=1.0):
    """Output tag: one per job, so jobs that split the seeds never share a file."""
    tag = f'{bus_system}_{feature}'
    if args.backbone:
        tag += f'_{args.backbone}'
    if args.scenario:
        tag += f'_{args.scenario}'
    if args.readout:
        tag += f'_{args.readout}'
    if noise_db is not None:
        tag += f'_snr{noise_db:g}'
    if frac != 1.0:
        tag += f'_f{frac:g}'
    if getattr(args, 'forget_semantics', None) not in (None, 'zero_std'):
        tag += f'_{args.forget_semantics}'
    if getattr(args, 'holdout_attack_type', None) is not None:
        tag += f'_h{args.holdout_attack_type}'
    if getattr(args, 'sequential', False):
        tag += '_seq'
    if args.seed is not None:
        tag += f'_s{args.seed}'
    elif args.seeds is not None:
        tag += '_s' + '-'.join(str(s) for s in args.seeds)
    if list(methods) != list(STANDARD_METHODS):
        tag += '_arms-' + hashlib.md5(','.join(methods).encode()).hexdigest()[:6]
    return tag


def resolve_feature_mode(cli_feature, exp):
    """Feature mode for this run: --feature if given, else feature_mode in the config.

    A config without feature_mode falls back to 'mean_std', the setting of the
    submitted runs.
    """
    return cli_feature if cli_feature is not None else exp['feature_mode']


def build_parser():
    parser = argparse.ArgumentParser(description='GDGU EVCS Localization Training')
    parser.add_argument('--bus', type=str, required=True,
                        choices=['34bus', '123bus', '8500bus'],
                        help='Bus system to run')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Single backbone to run (GCN/GAT/GIN). Default: all three')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Single scenario to run (S1/S2/...). Default: all')
    parser.add_argument('--seed', type=int, default=None,
                        help='Single seed to run. Default: all seeds from config')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Multiple seeds to run, e.g. --seeds 42 77 88. '
                             'Overrides config["seeds"]; ignored if --seed is set.')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override source data directory')
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU device index (default: 1)')
    parser.add_argument('--feature', type=str, default=None,
                        choices=['mean_std', 'raw'],
                        help='Node feature mode: raw (288-dim) or mean_std (48-dim, the '
                             'submitted runs). Default: feature_mode in the config')
    parser.add_argument('--readout', type=str, default=None, choices=['dual_pool', 'station'],
                        help='Readout of the backbones. Default: readout in the config')
    parser.add_argument('--arms', type=str, nargs='+', default=None,
                        help='Methods to run after Original, in order: GDGU, GIF, IDEA, '
                             'Retrain, or arms defined in the config. Default: the four '
                             'standard methods')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory. Default: results/job_<SLURM_JOB_ID> in a '
                             'Slurm job, else results/<YYYY-MM-DD_HH>')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Model cache directory. Default: <out-dir>/cache')
    parser.add_argument('--no-cache', action='store_true',
                        help='Do not cache Original and Retrain on disk')
    parser.add_argument('--save-outputs', action='store_true',
                        help='Save per-graph logits, fine-tuning curves, and unlearned models '
                             'to <out-dir>/outputs; same as save_outputs: true in the config')
    parser.add_argument('--data-cache', type=str, default=None,
                        help='Directory of the loaded-data cache; default: read the '
                             'source pickles in every run')
    parser.add_argument('--voltage-noise-db', type=float, default=None,
                        help='SNR in dB of white Gaussian noise on the bus voltages (E9); '
                             'default: voltage_noise_snr_db in the config, null adds none')
    parser.add_argument('--withdraw-fraction', type=float, default=None,
                        help='Withdraw the forgotten power from only this random fraction of '
                             'the training graphs (E5); default: withdraw_fraction in the '
                             'config, 1.0 withdraws it everywhere')
    parser.add_argument('--forget-semantics', default=None, choices=['zero_std', 'zero_raw', 'mask'],
                        help='How the withdrawn power is removed (E10): zero after standardization '
                             '(the submitted operation), zero before it, or zero plus a mask '
                             'channel; default: forget_semantics in the config, zero_std')
    parser.add_argument('--holdout-attack-type', type=int, default=None, choices=[1, 2, 3, 4],
                        help='Hold this attack type out of training and validation and test on '
                             'it (E10); default: holdout_attack_type in the config, none')
    parser.add_argument('--sequential', action='store_true',
                        help='Serve the requests one after another (E4): in S_j every method '
                             'starts from its model after S_{j-1}; needs --save-outputs and '
                             'the earlier scenarios in the same --out-dir')
    parser.add_argument('--resume', action='store_true',
                        help='Skip trials already saved in <out-dir>; on by default in a '
                             'Slurm job')
    return parser


def main():
    args = build_parser().parse_args()

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_properties(device).name}')

    # Paths
    bus_system = args.bus
    if args.data_dir:
        source_data = Path(args.data_dir)
    else:
        source_data = PROJECT_ROOT.parent / 'Source' / 'PB_data' / '3_EVCS Attacks'

    # Load config from YAML
    exp = load_experiment(bus_system, source_data)
    config = exp['config']
    feature = resolve_feature_mode(args.feature, exp)

    config['bus_system'] = bus_system
    config['feature_mode'] = feature
    if args.readout:
        config['readout'] = args.readout
    if args.voltage_noise_db is not None:
        config['voltage_noise_snr_db'] = args.voltage_noise_db
    noise_db = config['voltage_noise_snr_db']
    if args.withdraw_fraction is not None:
        config['withdraw_fraction'] = args.withdraw_fraction
    frac = config['withdraw_fraction']
    if args.sequential:
        config['sequential'] = True
    if args.holdout_attack_type is not None:
        config['holdout_attack_type'] = args.holdout_attack_type
    if args.forget_semantics is not None:
        config['forget_semantics'] = args.forget_semantics
    methods = list(args.arms) if args.arms else list(STANDARD_METHODS)
    check_methods(methods, config)

    # Output directory: fixed per Slurm job, so a requeued job resumes in place
    job_id = os.environ.get('SLURM_JOB_ID')
    if args.out_dir:
        output_dir = Path(args.out_dir)
    elif job_id:
        output_dir = PROJECT_ROOT / 'results' / f'job_{job_id}'
    else:
        output_dir = PROJECT_ROOT / 'results' / datetime.now().strftime('%Y-%m-%d_%H')
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = args.resume or bool(job_id)
    if args.save_outputs:
        config['save_outputs'] = True
    if config.get('sequential', False) and not config['save_outputs']:
        raise SystemExit('--sequential needs --save-outputs: every step reads the models '
                         'that the previous step saved')
    if config['save_outputs']:
        # One folder per noise level, since the file names hold no noise level
        sub = 'outputs' + ('' if noise_db is None else f'_snr{noise_db:g}') \
            + ('' if frac == 1.0 else f'_f{frac:g}')
        config['outputs_dir'] = str(output_dir / sub)
    cache = None if args.no_cache else ModelCache(
        Path(args.cache_dir) if args.cache_dir else output_dir / 'cache', config)

    print(f'Bus    : {bus_system}')
    print(f'Config : {CONFIG_DIR / f"{bus_system}.yaml"}')
    print(f'Output : {output_dir}')
    print(f'Feature: {feature}')
    print(f'Readout: {config["readout"]}')
    print(f'V noise: {"none" if noise_db is None else f"SNR {noise_db:g} dB"}')
    print(f'Withdrawal: {"all training graphs" if frac == 1.0 else f"fraction {frac:g} of the training graphs"}')
    if config.get('sequential', False):
        print('Requests: sequential, each scenario starts from the models of the previous one (E4)')
    if config.get('holdout_attack_type') is not None:
        print(f'Holdout: attack type {config["holdout_attack_type"]} is unseen until the test (E10)')
    if config.get('forget_semantics', 'zero_std') != 'zero_std':
        print(f'Deletion: {config["forget_semantics"]} (E10)')
    print(f'Methods: Original, {", ".join(methods)}')
    print(f'Cache  : {cache.root if cache else "off"}; resume: {resume}')

    # Load data, with P features and attack-type labels
    data = load_data(exp, bus_system, feature, args.data_cache)
    config['out_dim'] = data['n_evcs']
    scenarios = build_scenarios(exp['evcs_bus_ids'],
                                         data['evcs_node_indices'])

    # Print scenario summary
    print(f'\nScenarios ({len(scenarios)}):')
    for sk, sv in scenarios.items():
        print(f'  {sk:8s}  {sv["label"]}')

    # Filter by CLI args
    backbones = [args.backbone] if args.backbone else config['backbones']
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = config['seeds']
    scen_filter = {args.scenario: scenarios[args.scenario]} if args.scenario else scenarios

    # Run, saving every trial as soon as it ends
    tag = run_tag(bus_system, feature, args, methods, noise_db, frac)
    trials = [(backbone, scen_key, scen_val, seed)
              for backbone in backbones
              for scen_key, scen_val in scen_filter.items()
              for seed in seeds]
    t_start = time.time()
    n_run = run_trials(trials, data, config, device, output_dir, tag, methods, cache, resume)
    elapsed = time.time() - t_start
    print(f'\n{"="*70}')
    print(f'{n_run} of {len(trials)} trials run in {elapsed:.1f}s ({elapsed/60:.1f}min)')
    print(f'Finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*70}')

    results_all = read_jsonl(output_dir / f'{tag}_rows.jsonl')
    all_logs = {r['key']: r['logs'] for r in read_jsonl(output_dir / f'{tag}_epoch_logs.jsonl')}

    # Save all outputs
    save_results(results_all, all_logs, bus_system, data, config,
                 scen_filter, backbones, output_dir, device, tag)

    print(f'\nAll outputs in: {output_dir}')
    print(f'Visualize via:  notebooks/Viz.ipynb')

if __name__ == '__main__':
    main()
