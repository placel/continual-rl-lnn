# continual-rl-lnn
An implementation of Continual Learning scenarios utilizing the biologically-inspired Liquid Neural Network architecture.
****
*environment.yaml contains anaconda dependencies*
****
The main source code of experimentation is within src/continual/minigrid. Other folders baseline and ppo are outdated.

Hyperparameter Optimization training was insufficient, values are preliminary, not true optima

To run a model invoke the run the main file with path ./src/continual/minigrid/continual.py with the selected model as an argument


| **Model Type**   | **Argument**             |
| ---------------- | ------------------------ |
| MLP              | no arguments needed      |
| LSTM             | --use-lstm               |
| Actor-only CfC   | --cfc-actor              |
| Critic-only CfC  | --cfc-critic             |
| Actor-Critic CfC | --cfc-actor --cfc-critic |

A simple example of running the Actor-Critic CfC without additional arguments:
	
    python ./src/continual/minigrid/continual.py --cfc-actor --cfc-critic

****
**List of Optimal Hyperparameters**

| **Model Type**   | Learning Rate | Entropy  | Hidden Dimension | Hidden State |
| ---------------- | ------------- | -------- | ---------------- | ------------ |
| MLP              | 0.000537      | 0.016914 | 256              | N/A          |
| LSTM             | 0.000818      | 0.024115 | 128              | 128          |
| Actor-only CfC   | 0.000296      | 0.032278 | 256              | 256          |
| Critic-only CfC  | 0.000296      | 0.032278 | 256              | 256          |
| Actor-Critic CfC | 0.000296      | 0.032278 | 256              | 256          |

****

Example of running an optimal LSTM:
	
    python ./src/continual/minigrid/continual.py --use-lstm --learning-rate 0.000818 --ent-coef 0.024115 --hidden-dim 128 --hidden-state-dim 128

****

To watch the agent play a selected environment, open eval_model.py and update the path with the resulting model. File name looks like: LSTM_1756987356 

Then select the environment for the agent to play by uncommenting the corrosponding env_id like below.

    env_id = ['MiniGrid-Unlock-v0']

The final version of the agent will be loaded and put in the corrosponding environment. A window will popup to observe. 

Enable CLEAR for optimal results across all environments (as seen below)

****

**Stabilization Techniques**

Enable EWC with --ewc 
Enable CLEAR (Behaviour Cloning Only) with --clear

The optimal EWC weights for each model, loaded with --ewc-weight

| **Model Type**   | EWC Weight |
| ---------------- | ---------- |
| MLP              | 10_000_000 |
| LSTM             | 24_854     |
| Actor-Critic CfC | 10_694     |

Example of EWC running
	
    python ./src/continual/minigrid/continual.py --use-lstm --learning-rate 0.000818 --ent-coef 0.024115 --hidden-dim 128 --hidden-state-dim 128 --ewc --ewc-weight 24_854

CLEAR with V-Trace *is* implemented, but has not been tested. Run it with the command --v-trace along with --clear  