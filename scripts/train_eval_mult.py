import subprocess

# List of guidance weights to test
guidance_weights = [0.5, 1.2, 2.0, 4.0, 6.0]
horizons = [16, 32, 64]
dataset = 'hopper-medium-expert-v2'
logbase = 'logs/cond_a'

# Iterate over the guidance weights
for horizon in horizons:
    subprocess.run(['python', 'scripts/train.py', 
                '--dataset', dataset, 
                '--logbase', logbase,
                '--horizon', str(horizon)])
    for weight in guidance_weights:
        # Run the plan_guided_parallel.py script
        subprocess.run([
            'python', 'scripts/plan_guided_parallel.py', 
            '--dataset', dataset, 
            '--logbase', logbase,
            '--guidance_weight', str(weight),
            '--horizon', str(horizon)
        ])