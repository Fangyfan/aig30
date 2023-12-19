import math
import os
from imap_engine import EngineIMAP
import numpy as np
import torch
from datetime import datetime
from typing import Any, Literal, TypeAlias, Optional, overload
import random
from utils import get_timestamp
from _utils import Log
from inspect import iscoroutinefunction
from timeMrg import Timer

ActionType: TypeAlias = Literal[
    "rewrite -z",
    "rewrite",
    "balance",
    "refactor",
    "refactor -z",
    # "map_fpga"
]

Actions: list[ActionType] = [
    "rewrite -z",
    "rewrite",
    "balance",
    "refactor",
    "refactor -z",
    # "map_fpga"
]

ActionsIndex: dict[ActionType, int] = {}
for i, action in enumerate(Actions):
    ActionsIndex[action] = i

StateType: TypeAlias = torch.Tensor

ActionSpace = len(Actions)
FeatureSpace = 4 + ActionSpace + 1

LOG_FILE = f"./log/{get_timestamp()}.csv"


def log(aig, *data):
    time = get_timestamp(True)
    aig = [os.path.basename(aig)]
    for d in data:
        aig.append(str(d))
    aig.append(time)
    s = ",".join(aig)
    with open(LOG_FILE, "a") as f:
        f.write(f"{s}\n")


def set_log_file(log_file):
    global LOG_FILE
    LOG_FILE = log_file


_now = datetime.now().timestamp()
_date = datetime(2023, 1, 1, 00, 00, 00).timestamp()
_SS = math.ceil(_now - _date)

cn = 0.5
cl = 1

# v7
#   length_max = 150
#   counter_max = 40


class Env(object):
    def __init__(
        self, aigfile, temp=None, length_max=20, temp_prefix=None, device="cpu"
    ):
        """在初始化后必须调用一次reset"""
        self.device = device
        self._temp = temp
        self.length_max = length_max
        self.aigfile = aigfile
        if self._temp is None:
            if temp_prefix is None:
                self._temp = f"./dataset/temp/{get_timestamp()}.aig"
            else:
                self._temp = f"./dataset/temp/{temp_prefix}_{get_timestamp()}.aig"
        self._imap = EngineIMAP(self.aigfile, self._temp)
        self._imap.read()
        _, self._initNumAnd, self._initLev = self.get_stats()
        with open(aigfile, "rb") as f:
            _, num, *_ = f.readline().split()
            num = int(num)
            if num < 1000:
                self.timer = Timer(60)
            elif num < 10000:
                self.timer = Timer(60 * 5)
            else:
                self.timer = Timer(60 * 60)
            log(self.aigfile, "init", self._initNumAnd, self._initLev, 0, 0, [])
        self._curNumAnd = self._initNumAnd
        self._curLev = self._initLev
        self._lastNumAnd = self._curNumAnd
        self._lastLev = self._curLev
        self.seqIndex = []
        self._lastAct = ActionSpace - 1  # numactions -1
        self._lastAct2 = ActionSpace - 1  # numactions -1
        self._lastAct3 = ActionSpace - 1  # numactions -1
        self._lastAct4 = ActionSpace - 1  # numactions -1
        self.seqLength = 0
        self.done = False
        self.returns = 0
        self.valueMax = 0
        self._baseline = 0
        self._max = (
            self.aigfile,
            "max",
            self._curNumAnd,
            self._curLev,
            self.returns,
            0,
            "[]",
        )

    def reset(self):
        self._resyn2()
        self._resyn2()
        _, self._curNumAnd, self._curLev = self.get_stats()
        resyn2NumAnd = self._curNumAnd
        resyn2Lev = self._curLev
        initValue = self._statValue(self._initNumAnd, self._initLev)
        resyn2Value = self._statValue(resyn2NumAnd, resyn2Lev)
        self._baseline = (resyn2Value - initValue) / 20.0
        log(
            self.aigfile,
            "reset",
            resyn2NumAnd,
            resyn2Lev,
            self._baseline,
            resyn2Value,
            [],
        )
        self.valueMax = resyn2Value
        if resyn2Value > 0:
            self._max = (
                self.aigfile,
                "max",
                self._curNumAnd,
                self._curLev,
                0,
                resyn2Value,
                "[]",
            )
        return self.state()

    def get_action_seq(self, _join=","):
        """获取动作序列, 若想获取动作序列的序号, 直接调用self.seqI ndex"""
        seq = []
        for i in self.seqIndex:
            seq.append(Actions[i])
        seq = _join.join(seq)
        return seq

    def print(self, path, epoch):
        """输出动作序列到指定文件, 获取动作序列可用get_action_seq"""
        seq = self.get_action_seq()
        with open(path, "a") as f:
            f.write("this is seq of" + str(epoch) + ":")
            f.writelines(seq)
            f.write("\n")

    def close(self):
        """原来用来关闭abc, 现没用了"""

    def step(self, actionIdx: int):
        """执行一步动作, 并返回nextState, reward, self.done"""
        # 判断任务耗时是否应该提前终止
        self.done = self.timer.action(Actions[actionIdx])
        nextState = None
        reward = 0
        curValue = self._statValue(self._curNumAnd, self._curLev)
        # 如果任务未提前终止
        if self.done == False:
            self._takeAction(actionIdx)
            nextState = self.state()
            # 计算reward
            lastValue = curValue
            curValue = self._statValue(self._curNumAnd, self._curLev)
            reward = curValue - lastValue - self._baseline
            self.returns += reward
            # 计算最大价值
            if curValue > self.valueMax:
                self.valueMax = curValue
                self._max = (
                    self.aigfile,
                    "max",
                    self._curNumAnd,
                    self._curLev,
                    self.returns,
                    curValue,
                    self.get_action_seq("/"),
                )
            if self.seqLength >= self.length_max:
                self.done = True
        if self.done:
            if curValue > self.valueMax:
                self.valueMax = curValue
                self._max = (
                    self.aigfile,
                    "max",
                    self._curNumAnd,
                    self._curLev,
                    self.returns,
                    curValue,
                    self.get_action_seq("/"),
                )
            log(*self._max)
            log(
                self.aigfile,
                "done",
                self._curNumAnd,
                self._curLev,
                self.returns,
                curValue,
                self.get_action_seq("/"),
            )
        return nextState if nextState is not None else self.state(), reward, self.done

    def random(self, lut_out: float = None):
        """
        随机返回一个动作
        @param lut_out 当lut_out不为None时会有概率返回lut_out的动作
        @update 仅返回动作序号, 不再直接执行动作
        """
        action = random.randint(0, ActionSpace - 1)
        if lut_out is not None and random.random() < lut_out:
            action = "n"
        return action

    def _takeAction(self, actionIdx):
        """执行一步动作, 修改aig图, 更新状态, 更新动作序列"""
        self._lastAct4 = self._lastAct3
        self._lastAct3 = self._lastAct2
        self._lastAct2 = self._lastAct
        self._lastAct = actionIdx
        self.seqLength += 1
        self.seqIndex.append(actionIdx)
        try:
            if actionIdx == 0:
                self._imap.rewrite(zero_gain=True)  # rw -z
            elif actionIdx == 1:
                self._imap.rewrite()  # rw
            elif actionIdx == 2:
                self._imap.balance()  # b
            elif actionIdx == 3:
                self._imap.refactor()  # rf
            elif actionIdx == 4:
                self._imap.refactor(zero_gain=True)  # rf -z
            elif actionIdx == 5:
                return True
            elif actionIdx == "n":
                self._imap.lut_opt()
            else:
                assert ValueError
            self._lastNumAnd = self._curNumAnd
            self._lastLev = self._curLev
            _, self._curNumAnd, self._curLev = self.get_stats()
        except ValueError:
            Log.warning(f"动作{actionIdx}不存在")
        except Exception as err:
            Log.warning(f"动作{actionIdx}:{Actions[actionIdx]}执行失败")
            Log.warning(err)
        return False

    def state(self):
        """返回当前的state, 包含一个feature和graph"""
        oneHotAct = np.zeros(self.numActions())
        np.put(oneHotAct, self._lastAct, 1)
        lastOneHotActs = np.zeros(self.numActions())
        lastOneHotActs[self._lastAct2] += 1 / 3
        lastOneHotActs[self._lastAct3] += 1 / 3
        lastOneHotActs[self._lastAct] += 1 / 3
        stateArray = np.array(
            [
                self._curNumAnd / self._initNumAnd,
                self._curLev / self._initLev,
                self._lastNumAnd / self._initNumAnd,
                self._lastLev / self._initLev,
            ]
        )
        stepArray = np.array([float(self.seqLength) / self.length_max])
        combined = np.concatenate((stateArray, lastOneHotActs, stepArray), axis=-1)
        feature = torch.from_numpy(combined.astype(np.float32)).float()
        feature = feature.to(self.device, dtype=torch.float32)
        return feature

    # print_stats已弃用，改用get_stats
    # def print_stats(self, _type: Literal[0, 1] = 1):
    #     '''打印当前状态到/home/xunye/code/aigstate.txt文件中'''
    #     if _type == 1:
    #         self._imap.map_fpga()
    #     self._imap.print_stats(_type)

    def get_stats(self, _type: Literal[0, 1] = 1):
        """获取当前的numAnd和lev"""
        if _type == 1:
            self._imap.map_fpga()
        self._imap.print_stats(_type, _SS)
        with open(f"/home/jiaxingwang/code/aig/_temp_aigstate/{_SS}.txt", "r") as f:
            Type, _, _, numAnd, lev = f.readline().split()
        # print(Type, int(numAnd), int(lev))
        return Type, int(numAnd), int(lev)

    def save_aig(self, path: str):
        """保存当前的aig到指定目录"""
        return self._imap.writeAig(path)

    def _statValue(self, numAnd, lev):
        v = 1 - ((numAnd / self._initNumAnd) ** cn) * ((lev / self._initLev) ** cl)
        return v
        # return 0.1 * numAnd / self._initNumAnd + 0.9 * lev / self._initLev

    def _reward(self):
        if self._lastAct == ActionSpace:  # term
            return 0
        lastValue = self._statValue(self._lastNumAnd, self._lastLev)
        curValue = self._statValue(self._curNumAnd, self._curLev)
        reward = curValue - lastValue - self._baseline
        return reward

    def numActions(self):
        return ActionSpace

    def dimState(self):
        return FeatureSpace

    def _resyn2(self):
        self._imap.balance()
        self._imap.rewrite()
        self._imap.refactor()
        self._imap.balance()
        self._imap.rewrite()
        self._imap.rewrite(zero_gain=True)
        self._imap.balance()
        self._imap.refactor(zero_gain=True)
        self._imap.rewrite(zero_gain=True)
        self._imap.balance()
