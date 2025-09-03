import subprocess
import os
import sys
import time
import datetime
from subprocess import STDOUT, Popen, PIPE
import numpy as np
from pathlib import Path
import json
PWD = Path(__file__).resolve().parent# Get the current path
LOG_PATH = PWD / 'experiments' / 'clear_sweep' / 'results_log.json'

def run_experiment(directory, file_name, name, args, conda_env='lnn_env'):
    start_time = time.time()
    print("=" * 40)
    print(f"Starting experiment: {name}")
    print(f"Using args: {' '.join(args)}")

    conda_exe = r"C:\Users\Logan\anaconda3\Scripts\conda.exe"
    return_code = 1
    try:
        command = [conda_exe, "run", "-n", conda_env, "python", directory + file_name] + args
        output = Popen(command, stdout=PIPE, stderr=STDOUT)

        # Stream stdout lines
        while True:
            line = output.stdout.readline()
            line = str(line).replace("\\r\\n'", '').replace("b'", '')
            print(line)
            if not line:
                break
            ...

        return_code = output.returncode
    except Exception as e:
        print(f"Error occured on experiment {name}: {e}")
        return_code = 1

    print(f"Final experiment runtime: {datetime.timedelta(seconds=time.time() - start_time)}")
    return return_code == 0

def make_trial(experiment_name, model_cfg, seed, coef_pair):
    # fixed hyperparams per model; only seed and coefs vary
    base_args = [
        "--exp-id",               str(0),
        "--exp-name",             str(experiment_name),
        "--seed",                 str(seed),
        "--learning-rate",                   str(model_cfg["lr"]),
        "--ent-coef",             str(model_cfg["ent"]),
        "--hidden-dim",           str(model_cfg["hidden-dim"]),
        # PPO + BC toggles
        "--clear",
        "--clear-value-coef",     str(coef_pair["value"]),
        "--clear-kl-coef",        str(coef_pair["kl"]),
    ]

    # hidden-state-dim optional for MLP
    if model_cfg.get("hidden-state-dim") is not None:
        base_args += ["--hidden-state-dim", str(model_cfg["hidden-state-dim"])]

    # Model toggles
    if model_cfg.get("cfc-actor"):
        base_args.append("--cfc-actor")
    if model_cfg.get("cfc-critic"):
        base_args.append("--cfc-critic")
    if model_cfg.get("use-lstm"):
        base_args.append("--use-lstm")

    name = f"{model_cfg['name']}__kl{coef_pair['kl']}_v{coef_pair['value']}__s{seed}"
    
    return {"name": name, "args": base_args}

def make_experiments(experiment_name, models, seeds, coef_pairs):
    experiments = []
    for m in models:
        for pair in coef_pairs:
            for s in seeds:
                experiments.append(make_trial(experiment_name, m, s, pair))
    return experiments

def eval_function(model_dir):
    print(f'Model Directory: {model_dir}')
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

    return float(np.mean(scores)) 

def parse_trial_name(name: str):
    # Expected: "<MODEL>__kl<kl>_v<value>__s<seed>"
    try:
        part_model, part_pair, part_seed = name.split('__')
        kl_str, v_str = part_pair.split('_')
        kl = float(kl_str[2:])
        val = float(v_str[1:])
        seed = int(part_seed[1:])
        return part_model, kl, val, seed
    except Exception as e:
        raise ValueError(f"Bad trial name format: {name}") from e

def append_result_entry(model_key: str, kl: float, val: float, seed: int, score: float):
    if LOG_PATH.exists():
        with open(LOG_PATH, 'r') as f:
            data = json.load(f)
    else:
        data = {"models": {}}

    data["models"].setdefault(model_key, [])
    data["models"][model_key].append({
        "kl": kl,
        "value": val,
        "seed": seed,
        "score": score
    })

    with open(LOG_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    directory = r"C:\Users\Logan\Documents\School\Wales\MSc\continual-rl-lnn\src\continual\minigrid"
    file_name = r"\continual.py"

    experiment_name = "clear_sweep"

    # Small, paired sweep. Edit if needed.
    coef_pairs = [
        {"kl": 0.1, "value": 0.1},
        {"kl": 0.3, "value": 0.3},
        {"kl": 0.5, "value": 0.5},
    ]

    # Keep seeds minimal for insurance only.
    # seeds = [999, 998]
    seeds = [999]

    # Model HPs are from HPO
    models = [
        {
            "name": "LSTM",
            "hidden-dim": 128,
            "hidden-state-dim": 128,
            "lr": 0.0008183828832312314,
            "ent": 0.024115085083935874,
            "cfc-actor": False,
            "cfc-critic": False,
            "use-lstm": True,
        },
        {
            "name": "CfC_A&C",
            "hidden-dim": 256,
            "hidden-state-dim": 256,
            "lr": 0.0002967453235099826,
            "ent": 0.032278201588886286,
            "cfc-actor": True,
            "cfc-critic": True,
            "use-lstm": False,
        },
        {
            "name": "MLP",
            "hidden-dim": 256,
            "hidden-state-dim": None,
            "lr": 0.0005378582501432388,
            "ent": 0.01691417789055679,
            "cfc-actor": False,
            "cfc-critic": False,
            "use-lstm": False,
        },
    ]

    experiments = make_experiments(experiment_name, models, seeds, coef_pairs)

    total_runtime_start = time.time()
    for e in experiments:
        print(f"Running Experiment: {e['name']}")
        print(f"Experiment args: {e['args']}")
        ok = run_experiment(directory, file_name, e["name"], e["args"])

        if not ok:
            print(f"Experiment {e['name']} failed with non-zero return code.")

        base_dir = PWD / 'experiments' / 'clear_sweep' / 'models'
        # Sort through the directory to find the newest file (p.state().st_mtime)
        run_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)

        # Evaluate the latest trial
        score = eval_function(run_dirs[-1])
        print(f"[EVAL] {e['name']} -> score: {score}")

        # Log the result to the master json log file 
        model_key, kl, val, seed = parse_trial_name(e["name"])
        append_result_entry(model_key, kl, val, seed, float(score))

    print(f"Total runtime: {datetime.timedelta(seconds=time.time() - total_runtime_start)}")
    print("\a")

if __name__ == "__main__":
    main()
