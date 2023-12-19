from time import time
import numpy as np


class Timer(object):

    def __init__(self, duration: float):
        self.duration = duration
        self.start_time = time()    # 初始化时的时间
        self.end_time = time() + duration  # 初始化结束时间
        self.pre_time = time()      # 上次计时的时间
        self.store: dict[str, np.ndarray] = {}
        self.pre_act: str = None
        self.sum = 0
        self.len = 0

    def reset(self, duration: float):
        self.duration = duration
        self.start_time = time()
        self.end_time = time() + duration
        self.pre_time = time()
        self.store = {}
        self.pre_act = None

    def isDone(self):
        return time() >= self.end_time

    def action(self, action: str):
        self.sum += time() - self.pre_time
        self.len += 1
        act = action.split()[0]
        if self.pre_act is not None:
            if self.pre_act not in self.store:
                self.store[self.pre_act] = np.array([time() - self.pre_time])
            else:
                self.store[self.pre_act] = np.append(
                    self.store[self.pre_act], time() - self.pre_time)
        self.pre_act = act
        self.pre_time = time()
        if act not in self.store:
            if self.sum/self.len*2 + time() > self.end_time:
                return True
            else:
                return False
        else:
            mean = self.store[act].mean()
            if mean*2 + time() > self.end_time:
                return True
            else:
                return False

    def cost(self):
        return time() - self.start_time

    def left(self, rate=False):
        if rate:
            return (self.end_time-time())/self.duration
        return self.end_time - time()
