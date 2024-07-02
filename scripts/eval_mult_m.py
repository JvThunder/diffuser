import subprocess

# List of guidance weights to test
w = 1.2
horizons = [16, 32, 64]
m_list = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
dataset = 'hopper-medium-replay-v2'
logbase = 'logs/cond_a'

# Iterate over the guidance weights
for horizon in horizons:
    for m_temp in m_list:
        # Run the plan_guided_parallel.py script
        subprocess.run([
            'python', 'scripts/plan_guided_parallel.py', 
            '--dataset', dataset, 
            '--logbase', logbase,
            '--guidance_weight', str(w),
            '--horizon', str(horizon),
            '--m_temp', str(m_temp),
        ])