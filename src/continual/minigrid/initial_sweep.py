import subprocess
import os
import sys
import time
import datetime
from subprocess import STDOUT, Popen, PIPE

# Run a single experiment with the given args
def run_experiment(directory, file_name, name, args, conda_env='lnn_env'):
    start_time = time.time()
    print("=" * 40)
    print(f'Starting experiment: {name}')
    print(f'Using args: {" ".join(args)}')

    conda_exe = r"C:\Users\Logan\anaconda3\Scripts\conda.exe"
    return_code = None
    try:
        command = [conda_exe, 'run', '-n', conda_env, 'python', directory + file_name] + args
        output = Popen(command, stdout=PIPE, stderr=STDOUT)

        # Not real-time, but good enough for the purposes of this project
        while True:
            line = output.stdout.readline()
            line = str(line).replace("\\r\\n'", '').replace("b'", '')
            print(line)
            if not line:
                break
            ...

    except Exception as e:
        print(f'Error occured on experiment {name}: {e}')
        return_code = 1

    print(f'Final experiment runtime: {datetime.timedelta(seconds=time.time() - start_time)}')
    return return_code == 0

# Build an experiment based on the given arguments
# Experiment name is the batch experiment name, and will be shared among all experiments, whereas name alone is the name of the current executed experiment
def make_trial(experiment_name, args):
    base_args = [
        '--exp-name',           str(experiment_name),
        '--learning-rate',      str(args['lr']),
        '--ent-coef',           str(args['ent']),
        '--hidden-dim',         str(args['hidden-dim']),
        '--seed',               str(args['seed'])
    ]

    # hidden-state-dim optional for MLP
    if args.get('hidden-state-dim') is not None:
        base_args += ['--hidden-state-dim', str(args['hidden-state-dim'])]

    # Optional toggles
    if args.get('ewc'):
        base_args.append('--ewc')
    if args.get('ewc_weight'):
        base_args += ['--ewc-weight', str(args['ewc_weight'])]
    if args.get('clear'):
        base_args.append('--clear')
    if args.get('cfc-actor'):
        base_args.append('--cfc-actor')
    if args.get('cfc-critic'):
        base_args.append('--cfc-critic')
    if args.get('use-lstm'):
        base_args.append('--use-lstm')

    return {
        'name': str(args['name']),
        'args': base_args
    }

# Iterate over every model in the argument list and create:
# - base config
# - alt_lr variant (if 'alt-lr' provided)
# - alt_hs variant (if 'alt-hidden-state-dim' provided and model has a state dim)
# For each variant, replicate across 3 seeds.
def make_experiment(experiment_name, arg_list):
    seeds = [999, 998, 997]  # reduced to 3

    experiments = []
    for model in arg_list:
        # Base variant
        variants = []

        base_model = model.copy()
        base_model['__variant'] = 'base'
        variants.append(base_model)

        # alt_lr variant
        if 'alt-lr' in model and model['alt-lr'] is not None:
            v = model.copy()
            v['lr'] = model['alt-lr']
            v['__variant'] = 'alt_lr'
            variants.append(v)

        # alt_hs variant (skip if hidden-state-dim is None, e.g., MLP)
        if 'alt-hidden-state-dim' in model and model.get('hidden-state-dim') is not None:
            v = model.copy()
            v['hidden-state-dim'] = model['alt-hidden-state-dim']
            v['__variant'] = 'alt_hs'
            variants.append(v)

        # Expand each variant across seeds
        for v in variants:
            for seed in seeds:
                trial_model = v.copy()
                trial_model.update({'seed': seed})
                # Label name with variant and seed
                trial_model['name'] = f"{model['name']}__{v['__variant']}__s{seed}"
                trial = make_trial(experiment_name, trial_model)
                experiments.append(trial)

    return experiments

def main():
    directory = r"C:\Users\Logan\Documents\School\Wales\MSc\continual-rl-lnn\src\continual\minigrid"
    file_name = r"\continual.py"

    experiment_name = f'sweep'

    # EXAMPLE: fill in your models; add alt params per model
    # For MLP, set 'hidden-state-dim': None and omit 'alt-hidden-state-dim'
    arg_list = [
        {
            'name': 'LSTM',
            'hidden-dim': 128,
            'hidden-state-dim': 128,
            'alt-hidden-state-dim': 192,         # alt_hs
            'lr': 0.0001,
            'alt-lr': 0.001,                    # alt lr
            'ent': 0.01,
            'ewc': False,
            'ewc_weight': None,
            'clear': False,
            'cfc-actor': False,
            'cfc-critic': False,
            'use-lstm': True,
        },
        {
            'name': 'CfC_A&C',
            'hidden-dim': 128,
            'hidden-state-dim': 128,
            'alt-hidden-state-dim': 192,
            'lr': 0.0001,
            'alt-lr': 0.001,
            'ent': 0.02,
            'ewc': False,
            'ewc_weight': None,
            'clear': False,
            'cfc-actor': True,
            'cfc-critic': True,
            'use-lstm': False,
        },
        {
            'name': 'MLP',
            'hidden-dim': 128,
            'hidden-state-dim': None,           # no state dim for MLP
            'lr': 0.0001,
            'alt-lr': 0.001,
            'ent': 0.01,
            'ewc': False,
            'ewc_weight': None,
            'clear': False,
            'cfc-actor': False,
            'cfc-critic': False,
            'use-lstm': False,
        },
    ]

    experiments = make_experiment(experiment_name, arg_list)

    total_runtime_start = time.time()
    for e in experiments:
        print(f'Running Experiment: {e["name"]}')
        print(f'Experiment args: {e["args"]}')
        run_experiment(directory, file_name, e['name'], e['args'])

    print(f'Total runtime: {datetime.timedelta(seconds=time.time() - total_runtime_start)}')
    print('\a')

if __name__ == '__main__':
    main()
