'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from atari_wrappers import wrap_deepmind, make_atari


# 可以解決一些問題 call this command line
# pip install git+https://github.com/Kojoley/atari-py.git
# https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows/46739299

from collections import namedtuple
Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state','done'))

class ReplayMemory(object):
    ## TODO ##
    def __init__(self , args):
        self.memory_size = args.capacity
        self.batch_size = args.batch_size
        self.device = args.device
        self.memory_counter = 0
        self.state_counter = 0
        self.buffer = []


    def push(self, state, action, reward, next_state, done):
        """Saves a transition"""
        # self.buffer.append(tuple(map(tuple, [state, action, reward, next_state, done])))
        self.state_counter += 1
        exp = [state,action,reward,next_state,int(done)]
        if len(self.buffer) >= self.memory_size:
            self.buffer.pop(0)
        self.buffer.append(exp)

    def sample(self):
        """Sample a batch of transitions"""
        transitions = random.sample(self.buffer, self.batch_size)
        return Experience(*zip(*transitions))

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.
        x = self.cnn(x)
        x = x.view(-1,7*7*64)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._target_net.eval()
        self._optimizer = torch.optim.Adam(self._behavior_net.parameters(), lr=args.lr, eps=1.5e-4)

        ## TODO ##
        """Initialize replay buffer"""
        self._memory = ReplayMemory(args)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        ## TODO ##
        if random.random() < epsilon:
            action = action_space.sample()
            return action
        else:
            with torch.no_grad():
                state = torch.squeeze(torch.tensor(np.array(state,dtype=np.float32),device=self.device,dtype=torch.float32),3).to(self.device)  
                state.permute(2,0,1)
                outputs = self._behavior_net(state).detach()
                _, best_action = torch.max(outputs, 1)
                return best_action.item()

    def append(self, state, action, reward, next_state,done):
        ## TODO ##
        """Push a transition into replay buffer"""
        self._memory.push(state, action, reward / 10, next_state, done)

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        batch = self._memory.sample()
        ## TODO ##
        # state, action, reward, next_state, done = state.to('cpu'), action.to('cpu'), reward.to('cpu'), next_state.to('cpu'), done.to('cpu')
        batch_state = torch.squeeze(torch.tensor(np.array(batch.state,dtype=np.float32),device=self.device,dtype=torch.float32),4).to(self.device)
        batch_action = torch.tensor(batch.action,device=self.device).unsqueeze(1).to(self.device)
        batch_reward = torch.tensor(np.array(batch.reward,dtype=np.float32),device=self.device,dtype=torch.float32).unsqueeze(1).to(self.device)
        batch_next_state = torch.squeeze(torch.tensor(np.array(batch.next_state,dtype=np.float32),device=self.device,dtype=torch.float32),4).to(self.device)
        batch_done = torch.tensor(np.array(batch.done,dtype=np.float32),device = self.device,dtype=torch.float32).unsqueeze(1).to(self.device)

        q_value = self._behavior_net(batch_state).gather(1, batch_action.long())
        with torch.no_grad():
           q_next = torch.max(self._target_net(batch_next_state), 1)[0].view(-1, 1)
           q_target = batch_reward + q_next * gamma * (1.0 - batch_done)
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)
        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## TODO ##
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
                'target_net': self._target_net.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])

def train(args, agent , writer ):
    print('Start Training')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw , episode_life=True )
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0

    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        state, reward, done, _ = env.step(1) # fire first !!!

        state_buffer=[]
        for i in range(5):
            state_buffer.append(state)
        state_list = state_buffer[1:5]
        for t in itertools.count(start=1):
            # state.unsqueeze(0)
            # state.repeat(1,4,1,1)
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                # select action
                action = agent.select_action(state_list, epsilon, action_space)
                # decay epsilon
                epsilon -= (1 - args.eps_min) / args.eps_decay
                epsilon = max(epsilon, args.eps_min)

            # execute action
            next_state, reward, done, _ = env.step(action)
            state_buffer.pop(0)
            state_buffer.append(next_state)
            state_list = state_buffer[1:5]
            ## TODO ##
            # store transition
            agent.append(state_buffer[0:4], action, reward, state_buffer[1:5], done)

            if total_steps >= args.warmup:
                agent.update(total_steps)

            total_reward += reward

            if total_steps % args.eval_freq == 0 and  total_steps!=0:
                """You can write another evaluate function, or just call the test function."""
                test(args, agent, writer)
                agent.save(args.model + "dqn_" + str(total_steps) + ".pt")

            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                writer.add_scalar('Train/Episode Reward', total_reward, episode)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, episode)
                print('Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                        .format(total_steps, episode, t, total_reward, ewma_reward, epsilon))
                break
    env.close()


def test(args, agent, writer):
    print('Start Testing')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw)
    action_space = env.action_space
    e_rewards = []
    
    for i in range(args.test_episode):
        state_buffer=[]
        state = env.reset()
        e_reward = 0
        done = False
        for i in range(5):
            state_buffer.append(state)
        state_list = state_buffer[1:5]
        while not done:
            time.sleep(0.01)
            env.render()
            action = agent.select_action(state_list, args.test_epsilon, action_space)
            state, reward, done, _ = env.step(action)
            state_buffer.pop(0)
            state_buffer.append(state)
            state_list = state_buffer[1:5]
            e_reward += reward
        writer.add_scalar('Test/Episode Reward', e_reward, i+1)
        print('episode {}: {:.2f}'.format(i+1, e_reward))
        e_rewards.append(e_reward)

    env.close()
    print('Average Reward: {:.2f}'.format(float(sum(e_rewards)) / float(args.test_episode)))


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='./weights/ckpt/')
    parser.add_argument('--logdir', default='log/dqn_break')
    # train
    parser.add_argument('--warmup', default=200, type=int)
    parser.add_argument('--episode', default=30000, type=int)
    parser.add_argument('--capacity', default=100000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0000625, type=float)
    parser.add_argument('--eps_decay', default=1000000, type=float)
    parser.add_argument('--eps_min', default=0.1, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=10000, type=int)
    parser.add_argument('--eval_freq', default=200000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('-tmp', '--test_model_path', default='./weights/ckpt/dqn_4800000.pt')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--test_episode', default=10, type=int)
    parser.add_argument('--seed', default=20230422, type=int)
    parser.add_argument('--test_epsilon', default=0.01, type=float)
    args = parser.parse_args()

    ## main ##
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if args.test_only:
        agent.load(args.test_model_path)
        test(args, agent, writer)
    else:
        train(args, agent , writer)
        # agent.save(args.model)
        


if __name__ == '__main__':
    main()
