"""
dqn_lon_2 is modified from dqn_lon.

In developing: reconstruct some API

log:
2020.05.09
fix state_dict saving path into a absolute path

Reconstruct state representation for junction scenario.
Action space is also modified.
Only longitudinal control is considered.

todo: add safe RL updating method, add local map module
"""

from __future__ import print_function

import glob
import os
import sys

# if using carla098
sys.path.append("/home/lyq/CARLA_simulator/CARLA_098/PythonAPI/carla")
sys.path.append("/home/lyq/CARLA_simulator/CARLA_098/PythonAPI/carla/agents")
carla_path = '/home/lyq/CARLA_simulator/CARLA_098/PythonAPI'

try:
    sys.path.append(glob.glob(carla_path + '/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import math
import numpy as np
import datetime
from collections import namedtuple
import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

seed = 1
torch.manual_seed(seed)

# date info, ie: '20200509'
date = datetime.date.today().strftime('%Y%m%d')

# use a absolute path for saving
# save_dir = '/home/lyq/RL_TrainingResult/'+date+'/'  # consider date
save_dir = '/home/lyq/RL_TrainingResult/'

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

given_name = 'sole_lon_control'


class Net(nn.Module):
    """
    NN for DQN.
    """
    width = 200  # width of FC network.

    def __init__(self, state_dim, action_dim):
        """
        Build model using only fc
        todo: add depth of FC NN as hyper paras
        :param state_dim: input dimension, state space
        :param action_dim: output dimension, state space
        """
        super(Net, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, action_dim)
        )
        print("Net initialized")

    def forward(self, state_tensor):
        """
        Forward propagation of NN.
        """
        input_tensor = state_tensor[0]
        for item in state_tensor[1:]:
            input_tensor = torch.cat((input_tensor, item), 1)

        return self.fc(input_tensor.to(device))


class DQNAlgorithm(object):
    """
        Fix state, using only ground truth info.
    """

    capacity = 800
    learning_rate = 1e-3
    memory_counter = 0
    batch_size = 400
    gamma = 0.995
    update_count = 0
    episilo = 0.9
    dqn_epoch = 10

    episode = 0

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # create module of DQN
        self.eval_net = Net(self.state_dim, self.action_dim).to(device)
        self.target_net = Net(self.state_dim, self.action_dim).to(device)

        # set loss function
        self.loss_func = nn.MSELoss()

        # training parameters
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = [None] * self.capacity

        # debug path test
        # self.path = "/home/lyq/PycharmProjects/scenario_runner/srunner/challenge/DQN/DQN_training_test/"

        # original
        self.path = './DQN_save/'  # store weight
        self.writer = SummaryWriter('./DQN_save/logs/training_test')  # store training process

        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.episode_index = 0

    def select_action(self, state_tensor):
        """
        Call NN to select action.
        :param state_tensor: state tensor of current step.
        :return: action index of action space
        """
        # if True:  # for debug
        #     print("greedy policy")
        if np.random.rand() <= self.episilo:  # greedy policy
            action_value = self.eval_net.forward(state_tensor)
            action_value = action_value.to("cpu")
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        else:  # random policy
            print("random policy")
            action = np.random.randint(0, self.action_dim)
        return action

    def action_reward(self, reward):
        self.reward = reward

    def change_rate(self, episode_index):
        self.episode = episode_index
        epsilo_start = 0.85
        epsilo_final = 0.95
        epsilo_decay = 100

        self.episilo = epsilo_start + (epsilo_final - epsilo_start) * math.exp(-1. * episode_index / epsilo_decay)

    def store_transition(self, transition):
        index = self.memory_counter % self.capacity
        self.memory[index] = transition
        self.memory_counter += 1
        self.total_reward += transition.reward

    def update(self):
        # 每个episode结束，清零total_reward
        print('episode_total_reward:', self.total_reward)
        if self.total_reward > 1200:
            self.learning_rate = 1e-4

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)

        self.episode_reward += self.total_reward
        self.total_reward = 0.0
        self.episode_index += 1
        if self.episode_index % 10 == 0:
            mean_reward_10 = self.episode_reward / 10
            index = self.episode_index / 10
            self.writer.add_scalar('mean_reward_10', mean_reward_10, index)
            self.episode_reward = 0

        print('episode:', self.episode)
        if self.memory_counter > self.capacity:
            with torch.no_grad():
                # todo: need verification
                state = []
                for count in self.state_dim:
                    state_id = torch.tensor([t.state[count] for t in self.memory]).float().to(device)
                    state_id = state_id.unsqueeze(1)
                    state.append(state_id)

                action = torch.LongTensor([t.action for t in self.memory]).view(-1, 1).long().to(device)
                reward = torch.tensor([t.reward for t in self.memory]).float().to(device)

                next_state = []
                for count in self.state_dim:
                    state_id = torch.tensor([t.next_state[count] for t in self.memory]).float().to(device)
                    state_id = state_id.unsqueeze(1)
                    next_state.append(state_id)

                reward = (reward - reward.mean()) / (reward.std() + 1e-7)

                target_v = reward + self.gamma * self.target_net(next_state).max(1)[0]

            # Update...
            for _ in range(self.dqn_epoch):  # iteration ppo_epoch
                for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size,
                                          drop_last=False):
                    # v = (self.eval_net(state_image,state_speedx,state_speedy,state_steer).gather(1, action))[index]
                    loss = self.loss_func(target_v[index].unsqueeze(1), (
                        self.eval_net(state).gather(1, action))[index])

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.writer.add_scalar('loss/value_loss', loss, self.update_count)  # add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)
                    self.update_count += 1
                    if self.update_count % 100 == 0:  # update dict each 100 update times
                        self.target_net.load_state_dict(self.eval_net.state_dict())

            # self.memory_counter += 1
        else:
            print("Memory Buff is too less")

    def save_net(self):
        """
        Save parametrs of the NN

        todo: add checkpoint
        reference:
        if not os.path.isdir("./models/checkpoint"):
            os.mkdir("./models/checkpoint")
        torch.save(checkpoint, './models/checkpoint/ckpt_best_%s.pth' % (str(epoch)))

        """

        # add tags for current parameters
        # if given_name:
        #     save_dir = save_dir + given_name + '/'

        try:
            torch.save(self.eval_net.state_dict(), save_dir+'dqn.pth')
            print('Net is saved to ', save_dir+'dqn.pth')
        except Exception as e:
            print(type(e))
            print("Net saving FAILS!")

    def load_net(self):
        """
        Load previous NN parameters.
        """
        try:
            self.eval_net.load_state_dict(torch.load(save_dir + 'dqn.pth'))
            self.target_net.load_state_dict(torch.load(save_dir + 'dqn.pth'))
        except:
            print("Net loading FAILS!")


