from collections import namedtuple
import torch
import einops
import pdb
import numpy as np

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn


Trajectories = namedtuple('Trajectories', 'observations values')


class GuidedPolicy:

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, discount=0.99, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs
        self.discount = discount

    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        horizon = self.diffusion_model.horizon
        cond_reward = (1 -self.discount ** horizon) / (1 - self.discount)
        # cond_reward = 0
        cond_reward = torch.tensor(cond_reward, device=self.device, dtype=torch.float32)
        cond_reward = cond_reward.view(-1, 1)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, cond_reward, guide=None, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

class InversePolicy:

    def __init__(self, diffusion_model, inverse_model, normalizer, preprocess_fns, discount=0.99, **sample_kwargs):
        self.diffusion_model = diffusion_model
        self.inverse_model = inverse_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs
        self.discount = discount

    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        horizon = self.diffusion_model.horizon
        cond_reward = (1 -self.discount ** horizon) / (1 - self.discount)
        # cond_reward = 0
        cond_reward = torch.tensor(cond_reward, device=self.device, dtype=torch.float32)
        cond_reward = cond_reward.view(-1, 1)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, cond_reward, guide=None, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)
        
        # trajectories [ batch_size x horizon x transition_dim ]
        # use obs and next_obs to get action
        obs = trajectories[:1, 0:1, :]
        next_obs = trajectories[:1, 1:2, :]
        obs_cat = np.concatenate([obs, next_obs], axis=-1)

        # make it torch tensor
        obs_cat = torch.tensor(obs_cat, device=self.device, dtype=torch.float32)

        normed_action = self.inverse_model(obs_cat).detach().cpu().numpy()
        normed_action = normed_action.reshape(-1)
        action = self.normalizer.unnormalize(normed_action, 'actions')

        normed_observations = trajectories
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(observations, samples.values)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions