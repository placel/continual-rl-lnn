import datetime
from pathlib import Path
from dataclasses import dataclass
from subprocess import Popen, PIPE, STDOUT, TimeoutExpired
import re, math
METRIC_RE = re.compile(r"global_step=(\d+).*episodic_return=([-+]?\d*\.?\d+)")


import optuna
import optuna.visualization as vis
import matplotlib.pyplot as plt
import subprocess
import json 
import re
import tyro
import time
import sys
import os
import torch
import numpy as np
import pickle as pkl

# Constants
PWD = Path(__file__).resolve().parent# Get the current path
CONTINUAL = PWD / 'continual.py'

# Need to specify conda environment and location 
CONDA_EXE = r"C:\Users\Logan\anaconda3\Scripts\conda.exe"
CONDA_ENV = 'lnn_env'

# Need to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils import evaluation
from utils import models

# Define the Arguments class used with Tyro to process commands args
@dataclass
class Args:
    trials: int = 20
    total_timesteps: int = 500_000
    storage: str = None
    study_name: str = 'sequential_hpo'
    timeout_per_trial: int = 0
    model_type: str = 'mlp' # can be mlp, lstm, shared_cfc, cfc_actor, or cfc_critic 

# Launch the continual.py script and train a model 
# on a sequence of tasks with the given arguments
def run_trial(exp_name, total_timesteps, model_type, lr, ent_coef, 
                hidden_dim, hidden_state_dim, seed, timeout_sec, trial_number, trial):
    
    use_lstm = (model_type == 'lstm')
    cfc_actor = (model_type == 'cfc_actor' or model_type == 'shared_cfc')
    cfc_critic = (model_type == 'cfc_critic' or model_type == 'shared_cfc')
    cmd = [
        CONDA_EXE, 'run', '-n', CONDA_ENV, 
        'python', '-u', str(CONTINUAL), # -u = unbuffered
        '--exp-name', str(exp_name),
        '--seed', str(seed),
        '--total-timesteps', str(total_timesteps),
        '--learning-rate', str(lr),
        '--ent-coef', str(ent_coef),
        '--hidden-dim', str(hidden_dim),
        '--trial-id', str(trial_number)
    ]

    # Not always used. Only add if needed
    if not hidden_state_dim is None:
        cmd.append('--hidden-state-dim') 
        cmd.append(str(hidden_state_dim)) 
    if use_lstm:
        cmd.append('--use-lstm')
    if cfc_actor:
        cmd.append('--cfc-actor')
    if cfc_critic:
        cmd.append('--cfc-critic')

    os.makedirs(PWD / 'HPO' / exp_name / 'logs', exist_ok=True)
    log_path = PWD / 'HPO' / exp_name / 'logs' / f't{trial_number}_logs.log'

    # Run the HPO experiment. If it takes too long (timeout_sec), kill the proc
    # Log the final results in the above path
    with open(log_path, 'w', encoding='utf-8') as file:
        proc = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True)

        # while True:
        #     line = proc.stdout.readline()
        #     line = str(line).replace("\\r\\n'", '').replace("b'", '')
        #     print(line)
        #     if not line: break
        #     ...
        
        # GPT BELOW
        from collections import deque

        # Checkpoints at 25/50/75% of the per-task budget T
        checkpoints = [int(0.25 * total_timesteps), int(0.50 * total_timesteps), int(0.75 * total_timesteps)]
        next_cp_idx = 0

        # small smoothing buffer across recent episodes (any env)
        recent_returns = deque(maxlen= max(10, 2))  # keep it tiny; trainer logs often
        best = float("-inf")
        start_time = time.time()

        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            file.write(line)

            m = METRIC_RE.search(line)  # matches global_step=..., episodic_return=...
            if m:
                step = int(m.group(1))               # global_step (monotonic across tasks)
                ep_ret = float(m.group(2))           # episodic_return in [0, 1] for MiniGrid
                recent_returns.append(ep_ret)

                # Hit rungs at 25/50/75% of T (per task)
                if next_cp_idx < len(checkpoints) and step >= checkpoints[next_cp_idx]:
                    # smooth a bit, then make it monotone with best-so-far
                    smoothed = sum(recent_returns)/len(recent_returns)
                    best = max(best, smoothed)
                    rung = next_cp_idx + 1  # Optuna step index: 1,2,3
                    trial.report(best, rung)
                    if trial.should_prune():
                        proc.terminate()
                        try: proc.wait(timeout=5)
                        except: proc.kill()
                        raise optuna.TrialPruned(f"Pruned at ~{checkpoints[next_cp_idx]} steps (best={best:.3f})")
                    next_cp_idx += 1

            # Optional trial-level timeout (keeps your old behavior)
            if timeout_sec and (time.time() - start_time) > timeout_sec:
                proc.terminate()
                try: proc.wait(timeout=5)
                except: proc.kill()
                raise optuna.TrialPruned(f"Trial timed out after {timeout_sec}s")

        proc.wait()
        if proc.returncode != 0:
            raise optuna.TrialPruned(f"Training failed (return code {proc.returncode}). Check {log_path}")
        
        # try:
        #     out, _ = proc.communicate(timeout=timeout_sec)
        # except TimeoutExpired:
        #     proc.kill()
        #     out, _ = proc.communicate()
        #     file.write(out or '')
        #     raise optuna.TrialPruned(f'Trial timed out after {timeout_sec}s')
        # except Exception:
        #     out, _ = proc.communicate()

        # Write the final result of logs to the log file
        # file.write(out or '')

        if proc.returncode != 0:
            raise optuna.TrialPruned(f'Training failed (return code {proc.returncode}). Check {log_path}')
        

    # Find the path of the last run model (created by the above) and return it
    base_dir = PWD / 'HPO' / exp_name / 'models'
    # Sort through the directory to find the newest file (p.state().st_mtime)
    run_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    
    if not run_dirs:
        raise optuna.TrialPruned(f'No model subdirs under {base_dir} (see {log_path})')
    return run_dirs[-1] # Return the newest subdirectory containing the model

def eval_sequence(model_dir):
    # THIS IS FOR EVAL AFTER MODEL. NEW APPROACH JUST USES PERFORMANCE MATRIX
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = json.loads((model_dir / 'config.json').read_text())
    sequence = list(config['sequence'].keys())

    # agent = models.Agent(model_config).to(device)
    # agent.eval() # Set to eval mode

    # model_path = model_dir / 'final_model.pt'
    # state = torch.load(model_path, map_location=device, weights_only=True)
    # agent.load_state_dict(state)

    # Get the mean eval reward of each task (across 10 episodes (default in mean_reward()))
    # rewards = evaluation.mean_reward(agent, sequence, return_all=False)

    # Load the performance matrix associated with the model
    perf_matrix_path = model_dir / 'performance_matrix.npy'
    perf_matrix = np.load(perf_matrix_path)

    weights = np.array([0.6, 0.25, 0.15])
    # Iterate over each task in the sequence, and multiply results by the weights for that task.
    # np.roll(weights, i) will shift weights to the right i number of times
    # [0.6, 0.25, 0.15] will transform to [0.15, 0.6, 0.25] etc. 
    # This will focus on the recently trained task to promote learning of all tasks
    # While also gaining from potential forward or backward transfer
    for i in range(len(sequence) - 1): 
        perf_matrix[i] *= np.roll(weights, i)

    # Get the sum of all weighted tasks combined for a single score
    # Summing may provide slightly more variation than mean, giving Optuna an easier time maximizing score
    score = np.sum(perf_matrix)
    return float(score) # Score will be maximized by Optuna

def make_objective(total_timesteps, study_name, timeout_per_trial, model_type):

    def objective(trial: optuna.Trial):

        # Select a set of hyperparameters for this trial
        # CfC models typically require higher ranges 
        if 'cfc' in model_type:
            lr = trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True)
        else:
            lr = trial.suggest_float("learning_rate", 5e-5, 2.5e-3, log=True)

        ent_coef = trial.suggest_float("ent_coef", 1e-4, 3e-2, log=True)
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 192, 256])
        hidden_state_dim = trial.suggest_categorical('hidden_state_dim', [96, 128, 192, 256])
        
        seed = 5

        # MLP doesn't utilize a hidden_state_dimension
        if model_type == 'mlp':
            hidden_state_dim = None

        # Run the trial with the chosen hyperparams
        model_dir = run_trial(
            exp_name=study_name,
            total_timesteps=total_timesteps,
            model_type=model_type,
            lr=lr,
            ent_coef=ent_coef,
            hidden_dim=hidden_dim,
            hidden_state_dim=hidden_state_dim,
            seed=seed,
            timeout_sec=timeout_per_trial,
            trial_number=trial.number,
            trial=trial
        )

        # Evaluate the best performing model. Return a single score of the sum of rewards across ALL tasks
        # Optuna will then try to maximize this score
        score = eval_sequence(model_dir)
        trial.set_user_attr('model_dir', str(model_dir))
        print('\a') # Make sound
        return score
    
    return objective

def main():
    args = tyro.cli(Args)
    start_time = time.time()

    t = args.timeout_per_trial
    timeout_sec = None if (t is None or float(t) <= 0) else float(t)

    # Create sqlite storage to prevent dataloss of a crash
    if args.storage is None:
        hpo_dir = PWD / 'HPO' / args.study_name
        hpo_dir.mkdir(parents=True, exist_ok=True)

        storage_path = hpo_dir / f'{args.study_name}.db'
        args.storage = f'sqlite:///{storage_path}'
        print(f'Using storage: {args.storage}')

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize', # 'maximize' the objective. In this case, a combined mean reward on all tasks (Maximize model performance across all tasks)
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2, interval_steps=1)
    )

    # If a previous study was found and unfinished, progress will be displayed
    # Run args.trials number of trials. If interrupted, need to adjust 
    # If the last study ran 25 trials, and args.trials == 30, only 5 more are needed; adjust the study 
    completed_t = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if completed_t > 0:
        print(f'Study {args.study_name} has completed {completed_t} trials...')

        remaining_trials = max(0, args.trials - completed_t)
        print(f'Running {remaining_trials} more trials...')
    else:
        remaining_trials = args.trials # If new study, keep args.trial size (30)

    # Find the best hyperparameters within the given subset of params under make_objective
    # make_objective returns a function to make an trial (new hyperparams), and will attempt to 'maximize' the returned 'score'  
    study.optimize(
        make_objective(args.total_timesteps, args.study_name, timeout_sec, model_type=args.model_type),
        n_trials=remaining_trials,
        show_progress_bar=True
    )

    # Console log results
    print('\n')
    print('-'*60)
    print(f'Best Value: {study.best_value}\n')
    print(f'Best Params: {study.best_params}\n')
    print(f'Best model_dir: {study.best_trial.user_attrs.get("model_dir")}\n')
    print('-'*60)
    print('\n')
    
    # Log the final results into log file for longevity
    with open(PWD / 'HPO' / args.study_name / 'logs' / 'hpo_results.log', 'w', encoding='utf-8') as file:
        file.write(f'Best Value: {study.best_value}\n')
        file.write(f'Best Params: {study.best_params}\n')
        file.write(f'Best model_dir: {study.best_trial.user_attrs.get("model_dir")}\n')


    # Extract Optuna visualizations
    img_path = PWD / 'HPO' / args.study_name / 'plots' 
    img_path.mkdir(parents=True, exist_ok=True)
    
    # Too little variance among tasks, throws an error
    # vis.matplotlib.plot_param_importances(study)
    # plt.savefig(f'{img_path}/param_importance.svg')

    vis.matplotlib.plot_parallel_coordinate(study)
    plt.savefig(f'{img_path}/parallel_coordinates.svg')
    plt.close()

    # Optional additional plots
    vis.matplotlib.plot_slice(study)  # Shows parameter vs objective relationships
    plt.savefig(f'{img_path}/slices.svg')
    plt.close()
    
    vis.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(f'{img_path}/optimization_history.svg')
    plt.close()

    print(f'Final training time: {datetime.timedelta(seconds=time.time() - start_time)}')

if __name__ == '__main__':
    main()