import subprocess
import json

# List of guidance weights to test
guidance_weights = [0.5, 0.7, 1.5, 2.0, 4.0, 6.0]
horizons = [8, 16, 32, 64]
dataset = 'walker2d-medium-replay-v2'
logbase = 'logs/cond_a'
config_path = 'config/locomotion.py'

# Iterate over the guidance weights
for horizon in horizons:
    for weight in guidance_weights:
        # Run the plan_guided_parallel.py script
        subprocess.run([
            'python', 'scripts/plan_guided_parallel.py', 
            '--dataset', dataset, 
            '--logbase', logbase,
            '--guidance_weight', str(weight),
            '--horizon', str(horizon)
        ])