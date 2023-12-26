import argparse
import os
from utils import get_timestamp, clear_empty_dir
import numpy as np
import torch
from env_v8 import FeatureSpace

version = "v2.1"
gpu = "cuda:0"
model_load = None
memory_load = None
test = False
# model_load = './results/model/v2.1_1221_2324/checkpoint_1000000.pth'
# memory_load = './results/memory/v2.1_1221_2324/memory_1000000'
# test = True


# 设置迭代轮数
EPOCH = 100e4
# 每隔若干轮保存一次权重
checkpoint_interval = 10e4
# 记忆空间大小
memory_capacity = 50e4
# 迭代若干次后开始训练
# 因为一开始记忆中的元素很少，不适合直接训练
learn_start = 50e4
# learn_start = 0
# 启用cuda
disable_cuda = False
enable_cudnn = True
# 从经验回放中学习的频率
replay_frequency = 4
batch_size = 1024
memory_interval = 1e4
# v71
#   multi-step = 3
#   discount = 0.99
# v72
#   multi-step = 8
#   discount = 0.95
# v8
#   /home/xunye/code/aig2/log/rainbow/1123_1541.csv
model_save = f"./results/model/{version}_{get_timestamp()}"
memory_save = f"./results/memory/{version}_{get_timestamp()}"

feature_space = FeatureSpace
feature_out = 64

clear_empty_dir('./results')
if not os.path.exists(model_save):
    os.makedirs(model_save)
if not os.path.exists(memory_save):
    os.makedirs(memory_save)


def parse_args():
    # Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
    parser = argparse.ArgumentParser(description="Rainbow")
    parser.add_argument("--id", type=str, default="default", help="Experiment ID")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--disable-cuda", default=disable_cuda, action="store_true", help="Disable CUDA"
    )
    # parser.add_argument('--game', type=str, default='space_invaders',
    #                     choices=atari_py.list_games(), help='ATARI game')
    parser.add_argument(
        "--T-max",
        type=int,
        default=int(EPOCH),
        metavar="STEPS",
        help="Number of training steps (4x number of frames)",
    )
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=int(108e3),
        metavar="LENGTH",
        help="Max episode length in game frames (0 to disable)",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=1,
        metavar="T",
        help="Number of consecutive states processed",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="canonical",
        choices=["canonical", "data-efficient"],
        metavar="ARCH",
        help="Network architecture",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        metavar="SIZE",
        help="Network hidden size",
    )
    parser.add_argument(
        "--noisy-std",
        type=float,
        default=0.1,
        metavar="σ",
        help="Initial standard deviation of noisy linear layers",
    )
    parser.add_argument(
        "--atoms",
        type=int,
        default=51,
        metavar="C",
        help="Discretised size of value distribution",
    )
    parser.add_argument(
        "--V-min",
        type=float,
        default=-10,
        metavar="V",
        help="Minimum of value distribution support",
    )
    parser.add_argument(
        "--V-max",
        type=float,
        default=10,
        metavar="V",
        help="Maximum of value distribution support",
    )
    parser.add_argument(
        "--model-load",
        type=str,
        metavar="PARAMS",
        default=model_load,
        help="Pretrained model (state dict)",
    )
    parser.add_argument(
        "--memory-capacity",
        type=int,
        default=int(memory_capacity),
        metavar="CAPACITY",
        help="Experience replay memory capacity",
    )
    parser.add_argument(
        "--replay-frequency",
        type=int,
        default=replay_frequency,
        metavar="k",
        help="Frequency of sampling from memory",
    )
    parser.add_argument(
        "--priority-exponent",
        type=float,
        default=0.5,
        metavar="ω",
        help="Prioritised experience replay exponent (originally denoted α)",
    )
    parser.add_argument(
        "--priority-weight",
        type=float,
        default=0.4,
        metavar="β",
        help="Initial prioritised experience replay importance sampling weight",
    )
    parser.add_argument(
        "--multi-step",
        type=int,
        default=8,
        metavar="n",
        help="Number of steps for multi-step return",
    )
    parser.add_argument(
        "--discount", type=float, default=0.95, metavar="γ", help="Discount factor"
    )
    parser.add_argument(
        "--target-update",
        type=int,
        default=int(8e3),
        metavar="τ",
        help="Number of steps after which to update target network",
    )
    parser.add_argument(
        "--reward-clip",
        type=int,
        default=1,
        metavar="VALUE",
        help="Reward clipping (0 to disable)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0000625,
        metavar="η",
        help="Learning rate",
    )
    parser.add_argument(
        "--adam-eps", type=float, default=1.5e-4, metavar="ε", help="Adam epsilon"
    )
    parser.add_argument(
        "--batch-size", type=int, default=batch_size, metavar="SIZE", help="Batch size"
    )
    parser.add_argument(
        "--norm-clip",
        type=float,
        default=10,
        metavar="NORM",
        help="Max L2 norm for gradient clipping",
    )
    parser.add_argument(
        "--learn-start",
        type=int,
        default=int(learn_start),
        metavar="STEPS",
        help="Number of steps before starting training",
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=10,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    # TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
    parser.add_argument(
        "--render", action="store_true", help="Display screen (testing only)"
    )
    parser.add_argument(
        "--enable-cudnn",
        action="store_true",
        default=enable_cudnn,
        help="Enable cuDNN (faster but nondeterministic)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        default=checkpoint_interval,
        help="How often to checkpoint the model, defaults to 0 (never checkpoint)",
    )
    parser.add_argument("--memory", help="Path to save/load the memory from")
    parser.add_argument(
        "--disable-bzip-memory",
        action="store_true",
        help="Don't zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)",
    )

    # TODO: new
    parser.add_argument("--feature-space", default=feature_space)
    parser.add_argument("--feature-out", default=feature_out)
    parser.add_argument("--model-save", default=model_save)
    parser.add_argument("--memory-load", default=memory_load)
    parser.add_argument("--memory-save", default=memory_save)
    parser.add_argument("--memory-interval", default=memory_interval)
    parser.add_argument("--test", default=test)
    parser.add_argument("--version", default=version)

    # Setup
    args = parser.parse_args()
    metrics = {"steps": [], "rewards": [], "Qs": [], "best_avg_reward": -float("inf")}
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device(gpu)
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = args.enable_cudnn
    else:
        args.device = torch.device("cpu")
    return args
