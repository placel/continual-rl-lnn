"""
Run a batch of experiments sequentially.
This removes the need to manually manage multiple experiments at a time in parallel (which is much slower)
making more efficient use of the GPU, and freeing up management time
"""
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
def make_experiment(experiment_name, args):
    base_args = [
        '--exp-name', str(experiment_name),
        # '--exp-def', str(experiment_definition),
        '--num-envs', '4',
        '--learning-rate', str(args['lr']),
        '--ent-coef', str(args['ent']),
        '--hidden-state-dim', str(args['hidden-state-dim']),
    ]
    
    # Not all conditions are always true, add only if used
    if args['ewc']:
        base_args.append('--ewc')
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

def main():

    # k_repeats = 3
    k_repeats = 1

    directory = r"C:\Users\Logan\Documents\School\Wales\MSc\continual-rl-lnn\src\continual\minigrid"
    file_name = r"\continual.py"

    experiment_name = f'ex-{time.time()}'
    experiment_definition = \
    """
    Testing 3 versions of the same model. Models with the same args seem to learn quite differently, want to see why. EarlyStopping has been disabled for this run to see if this is the cause. 
    """
    
    arg_list = [
        # actor & critic LNN, no EWC or CLEAR
        # {
        #     'name': 'Actor & Critic LNN',
        #     'lr': 1e-4,
        #     'ent': 0.03,
        #     'ewc': False,
        #     'clear': False,
        #     'cfc-actor': True,
        #     'cfc-critic': True,
        #     'use-lstm': False
        # },
        # {
        #     'name': 'Actor & Critic LNN - EWC',
        #     'lr': 1e-4,
        #     'ent': 0.03,
        #     'ewc': True,
        #     'clear': False,
        #     'cfc-actor': True,
        #     'cfc-critic': True,
        #     'use-lstm': False
        # },
        {
            'name': 'LSTM',
            'hidden-state-dim': 256,
            'lr': 1e-4,
            'ent': 0.03,
            'ewc': False,
            'clear': False,
            'cfc-actor': False,
            'cfc-critic': False,
            'use-lstm': True
        },
        # {
        #     'name': 'LSTM - EWC',
        #     'lr': 1e-4,
        #     'ent': 0.03,
        #     'ewc': True,
        #     'clear': False,
        #     'cfc-actor': False,
        #     'cfc-critic': False,
        #     'use-lstm': True
        # },
        # {
        #     'name': 'MLP',
        #     'hidden-state-dim': None,
        #     'lr': 1e-4,
        #     'ent': 0.03,
        #     'ewc': False,
        #     'clear': False,
        #     'cfc-actor': False,
        #     'cfc-critic': False,
        #     'use-lstm': False
        # },
        # {
        #     'name': 'MLP - EWC',
        #     'lr': 1e-4,
        #     'ent': 0.03,
        #     'ewc': True,
        #     'clear': False,
        #     'cfc-actor': False,
        #     'cfc-critic': False,
        #     'use-lstm': False
        # }
    ]

    # {
    #     'name': 'Baseline',
    #     'lr': 2.5e-4,
    #     'ent': 0.01,
    #     'ewc': False,
    #     'clear': False,
    #     'cfc-actor': False,
    #     'cfc-critic': False,
    #     'use-lstm': True
    # }

    # Multiply the list of arguments by k for validation
    arg_list *= k_repeats

    # Iterate and make an experiment from each list of arguments in the arg_list
    experiments = [make_experiment(experiment_name, args) for args in arg_list]

    total_runtime_start = time.time()
    # Iterate and run through each experiment defined above
    # Can iterate k (probably 3) number of times for reptetion of results
    for e in experiments:
        run_experiment(directory, file_name, e['name'], e['args'])

    print(f'Total runtime: {datetime.timedelta(seconds=time.time() - total_runtime_start)}')

if __name__ == '__main__':
    main()