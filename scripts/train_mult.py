import os
import subprocess
from multiprocessing import Process, Semaphore
from itertools import product

config_path = 'config/locomotion.py'
logbase = 'logs/cond_a'

# Define the training hyperparams
n_diffusion_steps_list = [20]
n_proc = 8

# Create a semaphore
semaphore = Semaphore(n_proc)

def run_script(dataset, film, horizon, n_step, gpu_id):
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
datasets = [
    'walker2d-medium-replay-v2', 'walker2d-medium-expert-v2',
    'hopper-medium-replay-v2', 'hopper-medium-v2',
]
horizon_list = [16, 32]
params = list(product([True, False], horizon_list, n_diffusion_steps_list))
for dataset in datasets:
    for i, p in enumerate(params):
        Process(target=run_script, args=(dataset, p[0], p[1], p[2], i % 2)).start()


datasets = [
    'halfcheetah-medium-v2', 'halfcheetah-medium-expert-v2'
]
horizon_list = [4, 8]
params = list(product([True, False], horizon_list, n_diffusion_steps_list))
for dataset in datasets:
    for i, p in enumerate(params):
        Process(target=run_script, args=(dataset, p[0], p[1], p[2], i % 2)).start()