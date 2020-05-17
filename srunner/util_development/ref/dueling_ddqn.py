import os
import sys
import argparse
import numpy as np
# from random import sample
from CarlaLCTestEnv import CarlaEnv, PlayGame

# import matplotlib.pyplot as plt
import copy
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import pickle

# hyper-parameters
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.90
EPSILON = 0.95
EPSILON_DECAY = 0.99995
MEMORY_CAPACITY = 40000
Q_NETWORK_ITERATION = 50


class ReplayBuffer(object):
    def __init__(self, max_size=MEMORY_CAPACITY):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        s, a, r, s_, d = [], [], [], [], []

        for i in ind:
            state, action, reward, next_state, done = self.storage[i]
            s.append(np.array(state, copy=False))
            a.append(np.array(action, copy=False))
            r.append(np.array(reward, copy=False))
            s_.append(np.array(next_state, copy=False))
            d.append(np.array(done, copy=False))

        return np.array(s), np.array(s_), np.array(a), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Net(nn.Module):
    def __init__(self, num_states, num_actions, dueling=False):
        super(Net, self).__init__()
        self.dueling = dueling
        self.fc1 = nn.Linear(num_states, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(128, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(64, num_actions)
        self.out.weight.data.normal_(0, 0.1)
        if self.dueling:
            self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = self.out(x)
        if self.dueling:
            values = self.value(x)
            qvals = values + (action_prob - action_prob.mean())
            return qvals
        return action_prob


class DDQN(object):
    def __init__(self, num_states, num_actions, dueling=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = EPSILON
        self.eval_net = Net(num_states, num_actions, dueling).to(self.device)
        self.target_net = Net(num_states, num_actions, dueling).to(self.device)

        self.learn_step_counter = 0
        # self.memory_counter = 0
        # self.memory_counter1 = 0
        self.memory = ReplayBuffer()
        # self.memory1 = ReplayBuffer()

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.writer = None

    def choose_action(self, state):
        state = np.array(state)
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if np.random.rand() >= self.epsilon:  # greedy policy
            action_value = self.eval_net(state).cpu()
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        else:  # random policy
            action = np.random.randint(0, self.num_actions)
        if self.epsilon > 0.1:
            self.epsilon *= EPSILON_DECAY
        return action

    def store_transition(self, state, action, reward, next_state, done):
        # if action == 1:
        #     self.memory1.push((state, action, reward, next_state, done))
        # else:
        #     self.memory.push((state, action, reward, next_state, done))
        self.memory.push((state, action, reward, next_state, done))

    def batch_sample(self):
        s0, s_0, a0, r0, d0 = self.memory.sample(BATCH_SIZE - int(BATCH_SIZE / 3))
        # s1, s_1, a1, r1, d1 = self.memory1.sample(int(BATCH_SIZE / 3))
        # s = np.vstack((s0, s1))
        # a = np.vstack((a0, a1))
        # s_ = np.vstack((s_0, s_1))
        # r = np.vstack((r0, r1))
        # d = np.vstack((d0, d1))
        # return s, a, r, s_, d
        return s0, a0, r0, s_0, d0

    def learn(self):
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        s, a, r, s_, d = self.batch_sample()
        state_batch = torch.FloatTensor(s).to(self.device)
        action_batch = torch.LongTensor(a.astype(int)).view(-1, 1).to(self.device)
        next_state_batch = torch.FloatTensor(s_).to(self.device)
        reward_batch = torch.FloatTensor(r).to(self.device)
        terminal_batch = torch.FloatTensor(d).to(self.device)

        q_eval = self.eval_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            eval_next_act_batch = self.eval_net(next_state_batch).max(1)[1][:, None]
            target_next_val_batch = self.target_net(next_state_batch).gather(1, eval_next_act_batch)
        q_target = tuple(reward if terminal else reward + GAMMA * target_val for reward, terminal, target_val in
                         zip(reward_batch, terminal_batch, target_next_val_batch))
        q_target = torch.cat(q_target).view(-1, 1)

        loss = self.loss_func(q_eval, q_target)
        self.writer.add_scalar('Loss', loss, global_step=self.learn_step_counter)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def inference(self, x):
        x = self.eval_net(x)
        return x.max(1)[1]

    def save(self, directory, i):
        torch.save(self.eval_net.state_dict(), directory + 'dqn{}.pth'.format(i))
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self, directory, i):
        self.eval_net.load_state_dict(torch.load(directory + 'dqn{}.pth'.format(i)))
        print("====================================")
        print("Model has been loaded...")
        print("====================================")


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])  # mode = 'train' or 'test'
    parser.add_argument('--type', type=str, default='DDQN', help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")
    parser.add_argument('--with_PER', dest='with_per', action='store_true',
                        help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    #
    parser.add_argument('--nb_episodes', type=int, default=5000, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=1,
                        help="Number of consecutive frames (action repeat)")
    parser.add_argument('--training_interval', type=int, default=30, help="Network training frequency")
    parser.add_argument('--n_threads', type=int, default=8, help="Number of threads (A3C)")
    #
    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true',
                        help="Compute Average reward per episode (slower)")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4', help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    #
    parser.set_defaults(render=False)
    return parser.parse_args(args)


def train(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Check if a GPU ID was set
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    world, client = PlayGame.setup_world(host='localhost', fixed_delta_seconds=0.05, reload=True)
    # client.set_timeout(5.0)
    if world is None:
        return
    traffic_manager = client.get_trafficmanager(8000)
    env = CarlaEnv(world, traffic_manager)

    action_dim = 5
    state_dim = 17
    # Pick algorithm to train
    dqn = DDQN(state_dim, action_dim, dueling=True)
    directory = './weights/'

    dqn.writer = SummaryWriter(directory)
    episodes = 2001
    print("Collecting Experience....")
    reward_list = []

    for i in range(episodes):
        try:
            state = env.restart(8)
        except:
            world, client = PlayGame.setup_world(host='localhost', fixed_delta_seconds=0.05, reload=True)
            traffic_manager = client.get_trafficmanager(8000)
            env = CarlaEnv(world, traffic_manager)
            state = env.restart(8)
        for _ in range(5):
             _, _, done = env.step(0)
        ep_reward = 0
        for t in count():
            action = dqn.choose_action(state)
            next_state, reward, done = env.step(action)

            dqn.store_transition(state, action, reward, next_state, np.float(done))
            ep_reward += reward

            memory_counter = len(dqn.memory.storage)
            # memory_counter1 = len(dqn.memory1.storage)
            # if memory_counter1 > (BATCH_SIZE/3) and memory_counter > (2*BATCH_SIZE/3):
            if memory_counter > BATCH_SIZE:
                dqn.learn()
            if done or t > 700:
                dqn.writer.add_scalar('ep_r', ep_reward, global_step=i)
                print("episode: {}, the episode reward is {}".format(i, round(ep_reward, 3)))
                print("current epsilon is: {}".format(dqn.epsilon))
                # print(memory_counter1, memory_counter)
                print("episode steps: {}, memory_counter: {}".format(t, memory_counter))
                break
            state = next_state
        r = copy.copy(reward)
        reward_list.append(r)
        if i % 10 == 0:
            dqn.save(directory, i)
            with open(directory+'memory.pkl', 'wb') as replay_buffer:
                pickle.dump(dqn.memory.storage, replay_buffer)
        if i % 50 == 0:
            world, client = PlayGame.setup_world(host='localhost', fixed_delta_seconds=0.05, reload=True)
            traffic_manager = client.get_trafficmanager(8000)
            env = CarlaEnv(world, traffic_manager)


if __name__ == "__main__":
    train()
