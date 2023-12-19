# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from dgl.nn.pytorch import GraphConv
import dgl
import numpy as np
import os
from torch import optim
from torch.nn.utils import clip_grad_norm_
from typing import TypeAlias
from env_v8 import StateType
from replay_v4 import ReplayMemory


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class NoisyLinearSequential(nn.Module):
    def __init__(self, *NoisyLinearParams: tuple[int, int, int]):
        super(NoisyLinearSequential, self).__init__()
        self.NoisyLinear = []
        for params in NoisyLinearParams:
            self.NoisyLinear.append(NoisyLinear(*params))

    def forward(self, x):
        for noisyLinear in self.NoisyLinear:
            x = F.relu(noisyLinear(x))

        return x

    def reset_noise(self):
        for noisyLinear in self.NoisyLinear:
            noisyLinear.reset_noise()

    def to(self, device):
        for noisyLinear in self.NoisyLinear:
            noisyLinear.to(device)
        return super(NoisyLinearSequential, self).to(device)


class RainbowDQN(nn.Module):
    def __init__(self, args, action_space):
        super(RainbowDQN, self).__init__()
        # 初始化参数
        # self._state_size = state_size
        self._action_num = action_space
        self.feature_space = args.feature_space
        self.feature_out = args.feature_out
        self._atoms = args.atoms
        self.device = args.device
        self.linear = nn.Sequential(
            nn.Linear(self.feature_space, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
        )
        self.fc_v = NoisyLinearSequential(
            (4096, 2048, args.noisy_std),
            (2048, 1024, args.noisy_std),
            (1024, 512, args.noisy_std),
            (512, self._atoms, args.noisy_std),
        )
        self.fc_a = NoisyLinearSequential(
            (4096, 2048, args.noisy_std),
            (2048, 1024, args.noisy_std),
            (1024, 512, args.noisy_std),
            (512, self._action_num * self._atoms, args.noisy_std),
        )

    def forward(self, state: StateType, log=False):
        # 获取形状
        state = state.to(self.device)
        x = self.linear(state)
        x = x.view(-1, 4096)
        v = self.fc_v(x)  # Value stream
        a = self.fc_a(x)  # Advantage stream
        v, a = v.view(-1, 1, self._atoms), a.view(-1, self._action_num, self._atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        if log:  # Use log softmax for numerical stability
            # Log probabilities with action over second dimension
            q = F.log_softmax(q, dim=2)
        else:
            # Probabilities with action over second dimension
            q = F.softmax(q, dim=2)
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if "fc" in name:
                module.reset_noise()

    def to(self, device):
        self.fc_v.to(device)
        self.fc_a.to(device)
        return super(RainbowDQN, self).to(device)


class RainbowAgent:
    def __init__(self, args, action_space):
        self.action_space = action_space
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(
            device=args.device
        )  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.norm_clip = args.norm_clip
        self.device = args.device

        self.online_net = RainbowDQN(args, self.action_space).to(device=args.device)
        # if args.model:  # Load pretrained model if provided
        #     if os.path.isfile(args.model):
        #         # Always load tensors onto CPU by default, will shift to GPU if necessary
        #         state_dict = torch.load(args.model, map_location='cpu')
        #         if 'conv1.weight' in state_dict.keys():
        #             for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
        #                 # Re-map state dict for old pretrained models
        #                 state_dict[new_key] = state_dict[old_key]
        #                 # Delete old keys for strict load_state_dict
        #                 del state_dict[old_key]
        #         self.online_net.load_state_dict(state_dict)
        #         print("Loading pretrained model: " + args.model)
        #     else:  # Raise error if incorrect model path provided
        #         raise FileNotFoundError(args.model)

        self.online_net.train()

        self.target_net = RainbowDQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(
            self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps
        )

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    def set_device(self, device):
        self.device = device
        self.online_net = self.online_net.to(device)
        self.online_net.device = device
        self.target_net = self.target_net.to(device)
        self.target_net.device = device
        self.support = self.support.to(device)

    # Acts based on single state (no batch)
    def act(self, state: torch.Tensor):
        with torch.no_grad():
            return (
                (self.online_net(state.unsqueeze(0)) * self.support)
                .sum(2)
                .argmax(1)
                .item()
            )

    # Acts with an ε-greedy policy (used for evaluation only)
    # High ε can reduce evaluation scores drastically
    def act_e_greedy(self, state, epsilon=0.001):
        return (
            np.random.randint(0, self.action_space)
            if np.random.random() < epsilon
            else self.act(state)
        )

    def learn(self, mem: ReplayMemory):
        # Sample transitions
        print("try to get sample")
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(
            self.batch_size
        )
        print("get sample")
        # Calculate current state probabilities (online network noise already sampled)
        # Log probabilities log p(s_t, ·; θonline)
        log_ps = self.online_net(states, log=True)
        # log p(s_t, a_t; θonline)
        actions = actions.to(self.device)
        log_ps_a = log_ps[range(self.batch_size), actions]

        with torch.no_grad():
            # Calculate nth next state probabilities
            # Probabilities p(s_t+n, ·; θonline)
            pns = self.online_net(next_states)
            # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            dns = self.support.expand_as(pns) * pns
            # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            argmax_indices_ns = dns.sum(2).argmax(1)
            self.target_net.reset_noise()  # Sample new target net noise
            # Probabilities p(s_t+n, ·; θtarget)
            pns = self.target_net(next_states)
            # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = returns.unsqueeze(1) + nonterminals * (
                self.discount**self.n
            ) * self.support.unsqueeze(0)
            # Clamp between supported values
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz

            m = torch.zeros((self.batch_size, self.atoms)).to(self.device)
            offset = (
                torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size)
                .unsqueeze(1)
                .expand(self.batch_size, self.atoms)
                .to(actions)
            )
            m.view(-1).index_add_(
                0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1)
            )  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(
                0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1)
            )  # m_u = m_u + p(s_t+n, a*)(b - l)

        # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        loss = -torch.sum(m * log_ps_a, 1)
        self.online_net.zero_grad()
        # Backpropagate importance-weighted minibatch loss
        (weights * loss).mean().backward()
        # Clip gradients by L2 norm
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)
        self.optimiser.step()

        # Update priorities of sampled transitions
        mem.update_priorities(idxs, loss.detach().cpu().numpy())

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name="model.pth"):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    def load(self, path):
        self.online_net.load_state_dict(torch.load(path))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (
                (self.online_net(state.unsqueeze(0)) * self.support)
                .sum(2)
                .max(1)[0]
                .item()
            )

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
