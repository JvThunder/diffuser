import subprocess
config_path = 'config/locomotion.py'

# Define the dataset & logbase dir
dataset = 'hopper-medium-expert-v2'
logbase = 'logs/cond_a'

# Define the training hyperparams
horizons = [16, 32, 64]

# Iterate over the guidance weights
for horizon in horizons:
    subprocess.run(['python', 'scripts/train.py', 
                    '--dataset', dataset, 
                    '--logbase', logbase + '_film',
                    '--horizon', str(horizon),
                    '--film'])
    subprocess.run(['python', 'scripts/train.py', 
                    '--dataset', dataset, 
                    '--logbase', logbase + '_nofilm',
                    '--horizon', str(horizon)])