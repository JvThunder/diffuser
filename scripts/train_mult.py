import subprocess

# List of guidance weights to test
horizons = [16, 32, 64]
dataset = 'hopper-medium-expert-v2'
logbase = 'logs/cond_a'
config_path = 'config/locomotion.py'

# Iterate over the guidance weights
for horizon in horizons:
    subprocess.run(['python', 'scripts/train.py', 
                    '--dataset', dataset, 
                    '--logbase', logbase + '/film',
                    '--horizon', str(horizon),
                    '--film'])
    subprocess.run(['python', 'scripts/train.py', 
                    '--dataset', dataset, 
                    '--logbase', logbase + '/no_film',
                    '--horizon', str(horizon),
                    '--film'])