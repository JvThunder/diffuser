import os
import subprocess
from multiprocessing import Process, Semaphore
from itertools import product

config_path = 'config/locomotion.py'

# Define the dataset & logbase dir
logbase = 'logs/cond_a'

# Define the training hyperparams
n_diffusion_steps_list = [20]
guidance_weight_list = [0.5, 0.7, 1.5, 3.0]
m_temp_list = [-1.0, 0.5]
film_choice = [True, False]
ws_choice = [False]
n_proc = 4

# Create a semaphore
semaphore = Semaphore(n_proc)

def run_script(dataset, film, warm_starting, horizon, n_step, guidance_w, m_temp, gpu_id):
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

datasets = [
    'walker2d-expert-v2',
    'hopper-expert-v2'
]
horizon_list = [32]
params = list(product(film_choice, ws_choice, horizon_list, n_diffusion_steps_list, guidance_weight_list, m_temp_list))
for dataset in datasets:
    for i, p in enumerate(params):
        Process(target=run_script, args=(dataset, p[0], p[1], p[2], p[3], p[4], p[5], i % 4)).start()

datasets = [
    'halfcheetah-expert-v2'
]
horizon_list = [8]
params = list(product(film_choice, ws_choice, horizon_list, n_diffusion_steps_list, guidance_weight_list, m_temp_list))
for dataset in datasets:
    for i, p in enumerate(params):
        Process(target=run_script, args=(dataset, p[0], p[1], p[2], p[3], p[4], p[5], i % 4)).start()
