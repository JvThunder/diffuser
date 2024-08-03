import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import numpy as np


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    config: str = 'config.locomotion'
    dataset: str = 'hopper-medium-expert-v2'
    guidance_weight: float = 0.6
    horizon: int = 32
    m_temp: float = -1.0
    n_diffusion_steps: int = 20
    film: bool = False
    warm_starting: bool = False
    render: bool = False

args = Parser().parse_args('plan')
args.diffusion_loadpath = f'diffusion/defaults_H{args.horizon}_T{args.n_diffusion_steps}'

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

## ensure that the diffusion model and value function are compatible with each other
# utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=None,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    verbose=False,
    guidance_weight=args.guidance_weight,

    horizon=args.horizon,
    m=args.m_temp,
    
)

logger = logger_config()
policy = policy_config()


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
num_envs = 100
envs = [dataset.load_env() for _ in range(num_envs)]
obs_list = [env.reset() for env in envs]
observation = np.stack(obs_list, axis=0)
dones = [0 for _ in range(num_envs)]
ep_rewards = [0 for _ in range(num_envs)]
rollouts = [[obs_list[i].copy()] for i in range(num_envs)]

for t in range(args.max_episode_length):
    ## format current observation for conditioning
    conditions = observation
    action, samples = policy(conditions, verbose=args.verbose, warm_starting=args.warm_starting)

    next_obs_list = []
    for i in range(num_envs):
        ## save state for rendering only
        state = envs[i].state_vector().copy()

        ## execute action in environment
        if not dones[i]:
            next_observation, reward, done, _ = envs[i].step(action[i])
        else:
            next_observation = np.zeros(observation.shape[-1])
            reward = 0
            done = True
            
        if done: dones[i] = 1
        next_obs_list.append(next_observation)

        ## print reward and score
        ep_rewards[i] += reward
        score = envs[i].get_normalized_score(ep_rewards[i])
        print(
            f't: {t} | r: {reward:.2f} |  R: {ep_rewards[i]:.2f} | score: {score:.4f} | done: {dones[i]}',
            flush=True,
        )

        ## render rollout thus far
        if args.render:
            if not dones[i]: rollouts[i].append(next_observation.copy())
            if i < 5: logger.log(i, t, samples, state, rollouts[i])

    next_observation = np.stack(next_obs_list, axis=0)
    observation = next_observation

    if sum(dones) == num_envs:
        break

print("Average total rewards:", np.mean(ep_rewards))

## write results to json file at `args.savepath`
logger.finish(t, score, ep_rewards, dones, diffusion_experiment, None)