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

        # print(' '.join(command))
        # https://stackoverflow.com/questions/803265/getting-realtime-output-using-subprocess
        # Not real-time, but good enough for the purposes of this project
        while True:
            line = output.stdout.readline()
            line = str(line).replace("\\r\\n'", '').replace("b'", '')
            print(line)
            if not line: break
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
        '--hidden-state-dim',   str(args['hidden-state-dim']),
        '--seed',               str(args['seed'])
    ]
    
    # Not all conditions are always true, add only if used
    if args['ewc']:
        base_args.append('--ewc')
    if args['ewc_weight']:
        base_args.append('--ewc-weight')
        base_args.append(str(args['ewc_weight']))
    if args['clear']:
        base_args.append('--clear')
    if args['cfc-actor']:
        base_args.append('--cfc-actor')
    if args['cfc-critic']:
        base_args.append('--cfc-critic')
    if args['use-lstm']:
        base_args.append('--use-lstm')
    
    return {
        'name': str(args['name']),
        'args': base_args
    }

# Iterate over every model in the argument list and create a copy with a new seed for repeatability
def make_experiment(experiment_name, arg_list):
    # Unique seeds
    seeds = [1111, 2222, 3333, 4444, 5555]

    # Iterate over every model in the args_list adding a seed, and creating a trial
    experiments = []
    for model in arg_list:
        for seed in seeds:
            trial_model = model.copy()
            trial_model.update({'seed': seed})
            
            trial = make_trial(experiment_name, trial_model)
            experiments.append(trial)

    return experiments
    
def main():
    directory = r"C:\Users\Logan\Documents\School\Wales\MSc\continual-rl-lnn\src\continual\minigrid"
    # file_name = r"\continual.py"
    file_name = r"\continual_before_full_clear.py"

    # experiment_name = f'ex-{time.time()}'
    experiment_name = f'phase_one_final'
    
    arg_list = [
        {
            'name': 'LSTM',
            'hidden-dim': 128,
            'hidden-state-dim': 128,
            'lr': 0.0008183828832312314,
            'ent': 0.024115085083935874,
            'ewc': False,
            'ewc_weight': 0.0,
            'clear': False,
            'cfc-actor': False,
            'cfc-critic': False,
            'use-lstm': True,
        },
        {
            'name': 'CfC A&C',
            'hidden-dim': 256,
            'hidden-state-dim': 256,
            'lr': 0.0002967453235099826,
            'ent': 0.032278201588886286,
            'ewc': False,
            'ewc_weight': 0.0,
            'clear': False,
            'cfc-actor': True,
            'cfc-critic': True,
            'use-lstm': False,
        },
        {
            'name': 'CfC Actor',
            'hidden-dim': 256,
            'hidden-state-dim': 256,
            'lr': 0.0002967453235099826,
            'ent': 0.032278201588886286,
            'ewc': False,
            'ewc_weight': 0.0,
            'clear': False,
            'cfc-actor': True,
            'cfc-critic': False,
            'use-lstm': False,
        },
        {
            'name': 'CfC Critic',
            'hidden-dim': 256,
            'hidden-state-dim': 256,
            'lr': 0.0002967453235099826,
            'ent': 0.032278201588886286,
            'ewc': False,
            'ewc_weight': 0.0,
            'clear': False,
            'cfc-actor': False,
            'cfc-critic': True,
            'use-lstm': False,
        },
        {
            'name': 'MLP',
            'hidden-dim': 256,
            'hidden-state-dim': 0, # 0 instead of None, as None throws an error (quick fix)
            'lr': 0.0005378582501432388,
            'ent': 0.01691417789055679,
            'ewc': False,
            'ewc_weight': 0.0,
            'clear': False,
            'cfc-actor': False,
            'cfc-critic': False,
            'use-lstm': False,
        }
    ]


    # Iterate and make an experiment from each list of arguments in the arg_list
    experiments = make_experiment(experiment_name, arg_list)

    total_runtime_start = time.time()
    # Iterate and run through each experiment defined above
    for e in experiments:
        print(f'Running Experiment: {e["name"]}')
        print(f'Experiment args: {e["args"]}')
        run_experiment(directory, file_name, e['name'], e['args'])

    print(f'Total runtime: {datetime.timedelta(seconds=time.time() - total_runtime_start)}')
    # Alert at the end
    print('\a')

if __name__ == '__main__':
    main()