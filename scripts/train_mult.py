import os
import subprocess
from multiprocessing import Process, Semaphore
from itertools import product

config_path = 'config/locomotion.py'

# Define the dataset & logbase dir
dataset = 'walker2d-medium-v2'
logbase = 'logs/av_cond'

# Define the training hyperparams
horizon_list = [16, 32]
n_diffusion_steps_list = [20]
n_proc = 4

# Create a semaphore
semaphore = Semaphore(n_proc)

def run_script(film, horizon, n_step, gpu_id):
    with semaphore:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        args = ['python', 'scripts/train.py', 
                '--dataset', dataset, 
                '--logbase', logbase + ('_film' if film else '_nofilm'),
                '--horizon', str(horizon),
                '--n_diffusion_steps', str(n_step)]
        if film:
            args.append('--film')
        subprocess.run(args, env=env)

# Create a list of all combinations of n_step, horizon, and film
params = list(product([True, False], horizon_list, n_diffusion_steps_list))

# Start all processes, alternating between the two GPUs
for i, p in enumerate(params):
    Process(target=run_script, args=(p[0], p[1], p[2], i % 2)).start()