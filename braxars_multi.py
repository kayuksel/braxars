import os, time, pdb
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import gym, brax
from brax.envs.to_torch import JaxToTorchWrapper
from brax.envs import _envs, create_gym_env

for env_name, env_class in _envs.items():
    env_id = f"brax_{env_name}-v0"
    entry_point = partial(create_gym_env, env_name=env_name)
    if env_id not in gym.envs.registry.env_specs:
        print(f"Registring brax's '{env_name}' env under id '{env_id}'.")
        gym.register(env_id, entry_point=entry_point)

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain = 5/3)
        if hasattr(m, 'bias') and m.bias is not None: m.bias.data.zero_()

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        def block(in_feat, out_feat):
            return [nn.Linear(in_feat, out_feat), nn.Tanh()]
        self.model = nn.Sequential(*block(obs.shape[1], 64), 
            *block(64, 64), *block(64, env.action_space.shape[1]))
        self.model[-1] = nn.Tanh()
        init_weights(self)
    def forward(self, x, noise):
        model.add_params(noise)
        result = self.model(x)
        model.add_params(-noise)
        return result

    def num_params(self):
        return sum([np.prod(params.size()) for params in self.state_dict().values()])

    def add_params(self, diff_params):
        state_dict = dict()
        for key, params in self.state_dict().items():
            size = params.size()
            state_dict[key] = params + diff_params[:np.prod(size)].view(*size)
            diff_params = diff_params[np.prod(size):]
        self.load_state_dict(state_dict)

pop_size = 1024
batch_size = pop_size * 2
topk_size = pop_size // 8
n_steps = 1000
learning_rate = 5e-3
reset_counter = 0

with torch.no_grad():

    env = gym.make("brax_humanoid-v0", batch_size=batch_size)
    env = JaxToTorchWrapper(env)
    obs = env.reset()
    
    model = Actor().cuda()
    totalrw = torch.zeros(batch_size).cuda()

    m = torch.distributions.Normal(0, learning_rate)
    noisy_map = m.sample((pop_size, 
        model.num_params())).cuda()

    for i in range(n_steps):
        
        actions = []

        for curr_obs, noise in zip(obs, noisy_map):
            action = model(curr_obs, noise)
            actions.append(action.unsqueeze(0))

        for curr_obs, noise in zip(obs, noisy_map):
            action = model(curr_obs, -noise)
            actions.append(action.unsqueeze(0))

        actions = torch.cat(actions)
        obs, rewards, done, info = env.step(actions)

        totalrw += rewards
        print('%i %f' % (i - reset_counter, totalrw.mean()))

        diff_reward = totalrw[:pop_size] - totalrw[pop_size:]
        topk = diff_reward.abs().topk(topk_size, largest = True)[1]
        iterate = (noisy_map * diff_reward.unsqueeze(1))[topk]
        model.add_params(iterate.mean(dim=0) / totalrw.std())

        if done.sum() > 0:
            obs = env.reset()
            totalrw[:] = 0
            noisy_map = m.sample((pop_size, 
                model.num_params())).cuda()
            reset_counter = i + 1
            print('reset')