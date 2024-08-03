from collections import namedtuple
import torch
import einops
import pdb
import numpy as np
from collections import deque

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn


Trajectories = namedtuple('Trajectories', 'actions observations values')


class GuidedPolicy:

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, guidance_weight=0.5, discount=0.99, horizon=32, m=1, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.diffusion_model.guidance_weight = guidance_weight
        self.normalizer = normalizer
        self.transition_dim = diffusion_model.transition_dim
        self.observation_dim = diffusion_model.observation_dim
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs
        self.discount = discount
        self.horizon = horizon
        self.buffers = deque(maxlen=4) # buffer for storing actions
        # self.cond_reward = 0.8
        # self.cycle = 100
        # self.count = 0

        # temporal ensemble
        self.m = m
        self.w_i = np.exp(-np.arange(horizon, dtype=np.float32) * self.m) # wi = exp(-m * i)
        self.w_i = self.w_i.reshape(1, -1, 1)
        

    def __call__(self, conditions, verbose=True, warm_starting=True):
        batch_size = conditions.shape[0]
        conditions = self._format_conditions(conditions)

        # self.count += 1
        # if self.count % self.cycle == 0:
        #     self.cond_reward -= 1 / 40
        # print(f'cond_reward: {self.cond_reward}')
        cond_reward = 1
        cond_reward = torch.tensor(cond_reward, device=self.device, dtype=torch.float32)
        cond_reward = cond_reward.view(-1, 1)

        # make prv_action with [ batch_size x horizon x transition_dim ]
        prv_action = torch.zeros(batch_size, self.horizon, self.action_dim, device=self.device, dtype=torch.float32)
        if len(self.buffers) > 0:
            prv_action = self.buffers[0][:, 1:, :]
            prv_action = torch.tensor(prv_action, device=self.device, dtype=torch.float32)
            prv_action = torch.cat([prv_action, torch.randn(batch_size, 1, self.action_dim, device=self.device, dtype=torch.float32)], dim=1)

        self.sample_kwargs['warm_starting'] = warm_starting
        samples = self.diffusion_model(conditions, cond_reward, prv_action, guide=None, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)

        ## extract action [ batch_size x horizon x transition_dim ]
        # last dim: [self.action_dim, self.observation_dim, self.action_dim, ... self.observation_dim] 
        actions = trajectories[:, :, :self.action_dim]
        # clip to [-3, 3]
        # actions = np.clip(actions, -3, 3)
        actions = self.normalizer.unnormalize(actions, 'actions')
        # action: [batch_size x horizon x action_dim]

        ## temporal ensemble
        self.buffers.appendleft(actions)
        if self.m > 0:
            action_list = []
            for i in range(len(self.buffers)):
                action_list.append(self.buffers[i][::, i])
            first_actions = np.stack(action_list, axis=1)
            first_actions = np.sum(first_actions * self.w_i[::, :first_actions.shape[1],::] / self.w_i[::,:first_actions.shape[1],::].sum(axis=1), axis=1)
        else:
            first_actions = actions[:, 0]

        # normed_observations = trajectories[:, :, self.action_dim:]
        # observations = self.normalizer.unnormalize(normed_observations, 'observations')
        normed_observations = conditions.reshape(conditions.shape[0], -1, conditions.shape[1]).detach().cpu().numpy()
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations, samples.values)
        return first_actions, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions):
        # conditions = utils.apply_dict(
        #     self.normalizer.normalize,
        #     conditions,
        #     'observations',
        # )
        conditions = self.normalizer.normalize(conditions, 'observations')
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        return conditions
