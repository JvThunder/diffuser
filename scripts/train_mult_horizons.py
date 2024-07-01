import subprocess

# List of guidance weights to test
# horizons = [8, 32]
horizons = [16, 64]
dataset = 'hopper-medium-replay-v2'
logbase = 'logs/cond_a'
config_path = 'config/locomotion.py'

# Iterate over the guidance weights
for horizon in horizons:
    # Run the plan_guided_parallel.py script
    subprocess.run(['python', 'scripts/train.py', 
                    '--dataset', dataset, 
                    '--logbase', logbase,
                    '--horizon', str(horizon)])