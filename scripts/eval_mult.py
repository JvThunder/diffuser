import os
import subprocess
from multiprocessing import Process, Semaphore
from itertools import product

config_path = 'config/locomotion.py'

# Define the dataset & logbase dir
dataset = 'walker2d-medium-v2'
logbase = 'logs/cond_a'

# Define the training hyperparams
horizon_list = [16, 32]
n_diffusion_steps_list = [20]
guidance_weight_list = [0.5, 0.7, 1.5, 3.0]
m_temp_list = [-1.0, 0.5]
n_proc = 8

# Create a semaphore
semaphore = Semaphore(n_proc)

def run_script(film, warm_starting, horizon, n_step, guidance_w, m_temp, gpu_id):
    with semaphore:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        args = ['python', 'scripts/plan_guided_parallel.py', 
                '--dataset', dataset, 
                '--logbase', logbase + ('_film' if film else '_nofilm'),
                '--horizon', str(horizon),
                '--n_diffusion_steps', str(n_step),
                '--guidance_weight', str(guidance_w),
                '--m_temp', str(m_temp)]
        if film:
            args.append('--film')
        if warm_starting:
            args.append('--warm_starting')
        subprocess.run(args, env=env)

# Create a list of all combinations of n_step, horizon, and film
params = list(product([True, False], [True, False], horizon_list, n_diffusion_steps_list, guidance_weight_list, m_temp_list))

# Start all processes, alternating between the two GPUs
for i, p in enumerate(params):
    Process(target=run_script, args=(p[0], p[1], p[2], p[3], p[4], p[5], i % 2)).start()