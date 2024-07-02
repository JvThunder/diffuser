import subprocess
config_path = 'config/locomotion.py'

# Define the dataset & logbase dir
dataset = 'hopper-medium-expert-v2'
logbase = 'logs/cond_a'

# Define the training hyperparams
horizon_list = [16, 32, 64]
n_diffusion_steps_list = [10, 20, 40]

# Iterate over the guidance weights
for n_step in n_diffusion_steps_list:
    for horizon in horizon_list:
        subprocess.run(['python', 'scripts/train.py', 
                        '--dataset', dataset, 
                        '--logbase', logbase + '_film',
                        '--horizon', str(horizon),
                        '--n_diffusion_steps', str(n_step),
                        '--film'])
        subprocess.run(['python', 'scripts/train.py', 
                        '--dataset', dataset, 
                        '--logbase', logbase + '_nofilm',
                        '--horizon', str(horizon),
                        '--n_diffusion_steps', str(n_step)])