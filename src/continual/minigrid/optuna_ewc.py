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
    total_timesteps: int = 200_000
    storage: str = None
    study_name: str = 'sequential_hpo'
    timeout_per_trial: int = 0
    model_type: str = 'mlp' # can be mlp, lstm, shared_cfc, cfc_actor, or cfc_critic 

# Launch the continual.py script and train a model 
# on a sequence of tasks with the given arguments
def run_trial(exp_name, total_timesteps, model_type, lr, ent_coef, 
                hidden_dim, hidden_state_dim, ewc_weight, seed, timeout_sec, trial_number, trial, seed_indx):
    
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
        '--trial-id', str(trial_number),
        '--ewc',
        '--ewc-weight', str(ewc_weight)
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

        best = float('-inf')
        # small smoothing buffer across recent episodes (any env)
        recent_returns = deque(maxlen= max(10, 2))
        start_time = time.time()

        # The proc must be read, otherwise the buffer will reach a maximum length and freeze the terminal and training
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            file.write(line)

            # Trial Pruning Logic
            m = METRIC_RE.search(line)  # matches global_step=..., episodic_return=...
            # Only check for pruning on seed 1 to rule out bad starts. Keep other seeds for verification. Much simpler than cross-seed pruning
            if m and seed_indx == 0:
                step = int(m.group(1))               # global_step 
                ep_ret = float(m.group(2))           # episodic_return in 0-1 for MiniGrid
                recent_returns.append(ep_ret)

                # Hit rungs at 25/50/75% of timesteps (per task)
                if next_cp_idx < len(checkpoints) and step >= checkpoints[next_cp_idx]:
                    # smooth a bit, then make it monotone with best-so-far
                    smoothed = sum(recent_returns)/len(recent_returns)
                    best = max(best, smoothed)
                    trial.report(best, step=step)
                    if trial.should_prune():
                        proc.terminate()
                        try: proc.wait(timeout=5)
                        except: proc.kill()
                        raise optuna.TrialPruned(f"Pruned at ~{checkpoints[next_cp_idx]} steps (best={best:.3f})")
                    next_cp_idx += 1

        # Prune the trial if the subprocess takes too long
        if timeout_sec and (time.time() - start_time) > timeout_sec:
            proc.terminate()
            try: proc.wait(timeout=5)
            except: proc.kill()
            raise optuna.TrialPruned(f"Trial timed out after {timeout_sec}s")

        proc.wait()
        if proc.returncode != 0:
            raise optuna.TrialPruned(f"Training failed (return code {proc.returncode}). Check {log_path}")       

    # Find the path of the last run model (created by the above) and return it
    base_dir = PWD / 'HPO' / exp_name / 'models'
    # Sort through the directory to find the newest file (p.state().st_mtime)
    run_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    
    if not run_dirs:
        raise optuna.TrialPruned(f'No model subdirs under {base_dir} (see {log_path})')
    return run_dirs[-1] # Return the newest subdirectory containing the model

# This works for EWC as Optuna will still prioritize learning the task (too stong EWC regularization would prevent that)
# and also rewards retention (too little EWC regularization wouldn't be rewarded)
# Ultimately, the best EWC weight should be selected to prioritize task retention, and current task learning
def eval_sequence(model_dir):
    config = json.loads((model_dir / 'config.json').read_text())
    sequence = list(config['sequence'].keys())

    # Load the performance matrix associated with the model
    perf_matrix_path = model_dir / 'performance_matrix.npy'
    perf_matrix = np.load(perf_matrix_path)

    threshold = 0.8 # Average return of .8 may be considered 'learned' for HPO sake
    scores = []
    # Iterate over the performance matrix to calculate a score for this trial. Penalize the model if it doesn't learn the task after it's trained on it
    # All tasks need to be learned for fair evaluation
    for t in range(len(sequence)):
        row = perf_matrix[t] # Row of the current trained model
        curr_val = row[t] # Value of the current trained task
        seen_tasks_mean = float(perf_matrix[:t+1].mean()) # Mean of all tasks up to the current task only. 

        # If the current task wasn't learned (less than .8 return average), add a penalty of -0.8, and double the result to really tell Optuna it's not a good result 
        if curr_val < threshold:
            score = (curr_val - threshold) * 2
        # Otherwise, include any potential retention of tasks in the current score, to encourage retention
        else:
            score = seen_tasks_mean

        scores.append(score)

    return float(np.mean(scores)) # Score will be maximized by Optuna

def make_objective(total_timesteps, study_name, timeout_per_trial, model_type):

    def objective(trial: optuna.Trial):

        # Hyperparameters for each model type after HPO. EWC Weight range matching for CfC and LSTM as they exhibit similar limits under testing
        if 'cfc' in model_type:
            lr = 0.00029897916838103204
            ent_coef = 0.02472512725852833
            hidden_dim = 128
            hidden_state_dim = 256
            ewc_weight = trial.suggest_float("ewc_weight", 1e4, 5e5, log=True)
        elif 'lstm' in model_type:
            lr = 0.0011001437866728792
            ent_coef = 0.029626504303054173
            hidden_dim = 128
            hidden_state_dim = 128
            ewc_weight = trial.suggest_float("ewc_weight", 1e4, 5e5, log=True)
        if model_type == 'mlp':
            lr = 0.0012466997728671529
            ent_coef = 0.02676345769317956
            hidden_dim = 128
            hidden_state_dim = None
            ewc_weight = trial.suggest_float("ewc_weight", 1e6, 2e8, log=True)

        seeds = [1001, 2002, 3003] # Unique HPO seeds
        seed_dirs = {}
        seed_scores = {}

        # Take the average score of performance across 3 seeds
        scores = []
        for indx, s in enumerate(seeds):
                
            # Run the trial with the chosen hyperparams
            # Take the trial_best and pass it onto the subsequent run for a better reflection of trial performance
            model_dir = run_trial(
                exp_name=study_name,
                total_timesteps=total_timesteps,
                model_type=model_type,
                lr=lr,
                ent_coef=ent_coef,
                hidden_dim=hidden_dim,
                hidden_state_dim=hidden_state_dim,
                ewc_weight=ewc_weight,
                seed=s, # Only changing the seed for len(seeds) times
                timeout_sec=timeout_per_trial,
                trial_number=trial.number,
                trial=trial,
                seed_indx=indx
            )

            # Evaluate the best performing model. Return a single score of the sum of rewards across ALL tasks
            # Optuna will then try to maximize this score
            score = eval_sequence(model_dir)
            scores.append(score)
            seed_dirs[str(s)] = str(model_dir)
            seed_scores[str(s)] = float(score)
        
        # Apped this information to the trial object for easier identification later
        # Will be printed inside the best hpo log
        best_idx = int(np.argmax(scores))
        trial.set_user_attr("model_dirs", seed_dirs)
        trial.set_user_attr("seed_scores", seed_scores)
        trial.set_user_attr("best_seed", int(seeds[best_idx]))
        trial.set_user_attr("best_dir", seed_dirs[str(seeds[best_idx])])
        trial.set_user_attr("score_mean", float(np.mean(scores)))
        trial.set_user_attr("score_std", float(np.std(scores)))

        print('\a') # Make sound
        return np.mean(np.array(scores))
    
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
        file.write(f'Best model_dirs: {study.best_trial.user_attrs.get("model_dirs")}\n')
        file.write(f'Best seed_scores: {study.best_trial.user_attrs.get("seed_scores")}\n')
        file.write(f'Best seed: {study.best_trial.user_attrs.get("best_seed")}\n')
        file.write(f'Best dir: {study.best_trial.user_attrs.get("best_dir")}\n')
        file.write(f'Best score_mean: {study.best_trial.user_attrs.get("score_mean")}\n')
        file.write(f'Best score_std: {study.best_trial.user_attrs.get("score_std")}\n')


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