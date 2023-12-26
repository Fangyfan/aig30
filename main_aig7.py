# -*- coding: utf-8 -*-
from __future__ import division
import bz2
from datetime import datetime
import os
import pickle
import numpy as np
import torch
from tqdm import trange
from rainbow_v5 import RainbowAgent
from env_v8 import Env, ActionSpace, Actions, set_log_file
from replay_v4 import ReplayMemory
from utils import AigSet, get_timestamp, readJson
from rainbow_params import parse_args
import random
from time import time

LOG_FILE = f"./log/rainbow/{get_timestamp()}.csv"


def load_memory(memory_path, disable_bzip):
	if disable_bzip:
		with open(memory_path, "rb") as pickle_file:
			return pickle.load(pickle_file)
	else:
		with bz2.open(memory_path, "rb") as zipped_pickle_file:
			return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
	if disable_bzip:
		with open(memory_path, "wb") as pickle_file:
			pickle.dump(memory, pickle_file)
	else:
		with bz2.open(memory_path, "wb") as zipped_pickle_file:
			pickle.dump(memory, zipped_pickle_file)


def log(epoch, *data):
	time = get_timestamp(True)
	epoch = [str(epoch)]
	for d in data:
		epoch.append(str(d))
	epoch.append(time)
	s = ",".join(epoch)
	with open(LOG_FILE, "a") as f:
		f.write(f"{s}\n")


def train():
	# 定义网络

	# 定义经验回放（记忆）
	mem = ReplayMemory(args, args.memory_capacity)
	priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
	if args.memory_load is not None:
		mem = load_memory(args.memory_load, args.disable_bzip_memory)
		print(f"memory load size:{len(mem)}")

	aigSet = AigSet()
	# Training loop
	mem_q = []
	dqn.train()
	done = True
	acts = [0 for _ in range(ActionSpace)]
	acts10000 = [0 for _ in range(ActionSpace)]
	acts1000 = [0 for _ in range(ActionSpace)]
	acts100 = [0 for _ in range(ActionSpace)]
	resyn2 = [2, 1, 3, 2, 1, 0, 2, 4, 0, 2]
	t = 0
	env = None
	tr = trange(1, args.T_max + 1)
	t0 = time()
	rts = np.array([])
	tt = np.array([])
	ll = np.array([])
	for T in tr:
		if done:
			t = 0
			r = random.random()
			aig = aigSet.random()
			if env:
				t1 = time()
				_now = datetime.now()
				_dateTime = _now.strftime("%H:%M")
				tr.set_postfix(l=env.seqLength, r=env.valueMax, t=t1 - t0, d=_dateTime)
				rts = np.append(rts, env.returns)
				rts = np.delete(rts, np.s_[0:-100])
				ll = np.append(ll, env.valueMax)
				ll = np.delete(ll, np.s_[0:-100])
				tt = np.append(tt, t1 - t0)
				tt = np.delete(tt, np.s_[0:-100])
				log(T, rts.mean(), rts.max(), rts.min(), ll.mean(), ll.max(), ll.min(), tt.mean(), tt.max(), tt.min(),
				    get_timestamp(True))
				t0 = t1
				env.close()
			env = Env(aig, device=args.device)
			state = env.reset()

		if T % args.replay_frequency == 0:
			dqn.reset_noise()  # Draw a new set of noisy weights

		# Choose an action greedily (with noisy weights)
		if T < args.learn_start:
			r = random.random()
			if r < 0.1:
				with torch.no_grad():
					# action = dqn.act(state)
					action = env.random()
			else:
				action = resyn2[t % 10]
			t += 1
		else:
			action = dqn.act(state.unsqueeze(0))

		acts[action] += 1
		acts10000[action] += 1
		acts1000[action] += 1
		acts100[action] += 1
		if T % 1000 == 0:
			with open(f"./results/acts{args.version}.txt", "w") as f:
				f.write(f"最近100条\n")
				for i in range(len(acts100)):
					f.write(f"{i}\t{Actions[i]}\t{acts100[i]}\n")
				f.write(f"最近1000条\n")
				for i in range(len(acts1000)):
					f.write(f"{i}\t{Actions[i]}\t{acts1000[i]}\n")
				f.write(f"最近10000条\n")
				for i in range(len(acts10000)):
					f.write(f"{i}\t{Actions[i]}\t{acts10000[i]}\n")
				f.write(f"全部\n")
				for i in range(len(acts)):
					f.write(f"{i}\t{Actions[i]}\t{acts[i]}\n")
			acts1000 = [0 for _ in range(ActionSpace)]
			if T % 10000 == 0:
				acts10000 = [0 for _ in range(ActionSpace)]
		if T % 100 == 0:
			acts100 = [0 for _ in range(ActionSpace)]

		next_state, reward, done = env.step(action)  # Step
		if args.reward_clip > 0:
			reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards

		# Append transition to memory
		mem.append(state, action, reward, done)
		if args.memory_save is not None and T % args.memory_interval == 0:
			mem_file = os.path.join(args.memory_save, f"memory_{T}")
			mem_q.append(mem_file)
			if len(mem_q) > 3:
				rm = mem_q.pop(0)
				# print(f'remove {rm}')
				os.remove(rm)
			save_memory(mem, mem_file, args.disable_bzip_memory)

		# Train and test
		if T >= args.learn_start:
			# Anneal importance sampling weight β to 1
			mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)
			if T % args.replay_frequency == 0:
				# Train with n-step distributional double-Q learning
				dqn.learn(mem)
			# Update target network
			if T % args.target_update == 0:
				dqn.update_target_net()
			# Checkpoint the network
			if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
				dqn.save(args.model_save, f"checkpoint_{T}.pth")
		state = next_state

	env.close()


def test(aig):
	env = Env(aig, device="cpu")
	dqn.set_device("cpu")
	dqn.eval()
	env.reset()
	done = False
	returns = 0
	while not done:
		state = env.state()
		action = dqn.act(state.unsqueeze(0))
		next_state, reward, done = env.step(action)  # Step
		returns += reward
		# print(action, reward)
	numAnd, lev = env._curNumAnd, env._curLev
	# with open('/home/xunye/code/aig/results/seq/b05_comb.seq', 'w') as f:
	#     f.write(env.get_action_seq())
	# env.print_stats()
	v_done = returns, env.get_action_seq(), numAnd, lev
	_, _, maxNumAnd, maxCurLev, maxReturns, _, seq = env._max
	seq = seq.replace("/", ",")
	v_max = maxReturns, seq, maxNumAnd, maxCurLev
	return os.path.basename(aig), maxNumAnd, maxCurLev


# 加载参数
args = parse_args()
dqn = RainbowAgent(args, ActionSpace)

if args.model_load is not None:
	dqn.load(args.model_load)
	print(f"model load {args.model_load}")

if args.test:
	dqn.eval()
	args.device = "cpu"
else:
	dqn.train()

print("Device:", args.device)

if not args.test:
	print(f"train at {dqn.online_net.device}")
	train()

# print(test("/home/xunye/test_imap/b05_comb/b05_comb.aig"))


# print(test("./testset/i10.aig"))
# print(test("./testset/c1355.blif"))
# print(test("./testset/c7552.blif"))
# print(test("./testset/c6288.blif"))
# print(test("./testset/c5315.blif"))
# print(test("./testset/dalu.blif"))
# print(test("./testset/k2.blif"))
# print(test("./testset/apex1.blif"))
# print(test("./testset/bc0.blif"))

_now = datetime.now()
dateTime = _now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
print(dateTime)

