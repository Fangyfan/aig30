import os
import random
import torch
from typing import Literal, TypeAlias, IO
from torch import nn
import dgl
from datetime import datetime
from colorama import Fore, Back, Style
import json
from imap_engine import EngineIMAP
import os
from graphExtractor import extract_dgl_graph
import dgl
import pandas as pd
from typing import Literal
from time import time
from _utils import *

ActionType: TypeAlias = Literal[
    "balance",
    "rewrite",
    "rewrite -z",
    "rewrite -l false",
    "rewrite -z -l false",
    "refactor",
    "refactor -z",
    "refactor -l false",
    "refactor -z -l false",
    # "map_fpga"
]

Actions: list[ActionType] = [
    "balance",
    "rewrite",
    "rewrite -z",
    "rewrite -l false",
    "rewrite -z -l false",
    "refactor",
    "refactor -z",
    "refactor -l false",
    "refactor -z -l false",
    # "map_fpga"
]

DEVICE: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DataType: TypeAlias = tuple[
    torch.Tensor, dgl.DGLGraph, int, float, torch.Tensor, dgl.DGLGraph, bool
]


class Object(object):
    def __init__(self, dict: dict = {}):
        super().__init__()
        self._dict = dict

    def __getattr__(self, key):
        if key in self._dict:
            return self._dict[key]
        else:
            return None

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        self._dict[key] = value

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def set_device(device: str):
    global DEVICE
    DEVICE = device


def get_device():
    return DEVICE


def get_timestamp(second=False):
    _now = datetime.now()
    if second:
        _time = _now.strftime("%m%d_%H%M%S")
    else:
        _time = _now.strftime("%m%d_%H%M")
    return _time


def clear_empty_dir(dir_path: str):
    """
    清空目录下的空文件夹
    :param dir_path: 目录路径
    :return:
    """
    if not os.path.isdir(dir_path):
        return

    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            clear_empty_dir(item_path)

    if os.path.isdir(dir_path) and not os.listdir(dir_path):
        print(f"已清除:{dir_path}")
        os.rmdir(dir_path)


class AigSet(object):
    def __init__(
        self,
        *range: Literal[
            1000,
            4000,
            7000,
            10000,
            20000,
            40000,
            "s",
            "m",
            "l",
            "xl",
            "xxl",
            "_else",
            "all",
            "small",
            "mid",
            "big20k",
            "big200k",
        ],
    ) -> None:
        self.aigs: list[str] = []
        self.json_data = readJson()
        if 1000 in range or "all" in range or "s" in range:
            for aig, nodes, edges in self.json_data["_1000"]:
                self.aigs.append(aig)
        if 4000 in range or "all" in range or "m" in range:
            for aig, nodes, edges in self.json_data["_4000"]:
                self.aigs.append(aig)
        if 7000 in range or "all" in range or "m" in range:
            for aig, nodes, edges in self.json_data["_7000"]:
                self.aigs.append(aig)
        if 10000 in range or "all" in range or "m" in range:
            for aig, nodes, edges in self.json_data["_10000"]:
                self.aigs.append(aig)
        if 20000 in range or "all" in range or "l" in range:
            for aig, nodes, edges in self.json_data["_20000"]:
                self.aigs.append(aig)
        if 40000 in range or "all" in range or "xl" in range:
            for aig, nodes, edges in self.json_data["_40000"]:
                self.aigs.append(aig)
        if "_else" in range or "all" in range or "xxl" in range:
            for aig, nodes, edges in self.json_data["_else"]:
                self.aigs.append(aig)
        if "small" in range or "all" in range:
            for f in os.listdir("../aig/dataset/small"):
                f = os.path.join("../aig/dataset/small", f)
                self.aigs.append(f)
        if "mid" in range or "all" in range:
            for f in os.listdir("../aig/dataset/mid"):
                f = os.path.join("../aig/dataset/mid", f)
                self.aigs.append(f)
        if "big20k" in range or "all" in range:
            for f in os.listdir("../aig/dataset/big20k"):
                f = os.path.join("../aig/dataset/big20k", f)
                self.aigs.append(f)
        if "big200k" in range or "all" in range:
            for f in os.listdir("../aig/dataset/big200k"):
                f = os.path.join("../aig/dataset/big200k", f)
                self.aigs.append(f)

    def random(self):
        now = random.randint(0, len(self.aigs) - 1)
        return self.aigs[now]


INPUT_DIR = "../aig/dataset/test"
JSON_PATH = "../aig/dataset/test.json"
EXCEL_PATH = "../aig/dataset/test_cost.xlsx"

JsonKeyType = Literal["_1000", "_4000", "_7000", "_10000", "_20000", "_40000", "_else"]

JsonValueType = list[tuple[str, int, int]]
JsonType = dict[JsonKeyType, JsonValueType]


def genJson() -> JsonType:
    aigs = []
    files = os.listdir(INPUT_DIR)
    for file_name in files:
        aigs.append(INPUT_DIR + "/" + file_name)

    json_data: JsonType = {
        "_1000": [],
        "_4000": [],
        "_7000": [],
        "_10000": [],
        "_20000": [],
        "_40000": [],
        "_else": [],
    }

    for aig in aigs:
        abc = abc_py.AbcInterface()
        abc.start()
        abc.read(aig)
        print(aig)
        graph: dgl.DGLGraph = extract_dgl_graph(abc)
        nodes, edges = graph.num_nodes(), graph.num_edges()
        if nodes < 1000:
            json_data["_1000"].append((aig, nodes, edges))
        elif nodes < 4000:
            json_data["_4000"].append((aig, nodes, edges))
        elif nodes < 7000:
            json_data["_7000"].append((aig, nodes, edges))
        elif nodes < 10000:
            json_data["_10000"].append((aig, nodes, edges))
        elif nodes < 20000:
            json_data["_20000"].append((aig, nodes, edges))
        elif nodes < 40000:
            json_data["_40000"].append((aig, nodes, edges))
        else:
            json_data["_else"].append((aig, nodes, edges))
        abc.end()
        del abc

    with open(JSON_PATH, "w") as f:
        json.dump(json_data, f)
    return json_data


def readJson() -> JsonType:
    with open(JSON_PATH, "r") as f:
        return json.load(f)


def action(aig: str, act: ActionType):
    imap = EngineIMAP(aig, "")
    imap.read()
    if act == "balance":
        imap.balance()
    elif act == "rewrite":
        imap.rewrite()
    elif act == "rewrite -z":
        imap.rewrite(zero_gain=True)
    elif act == "rewrite -l false":
        imap.rewrite(level_preserve=False)
    elif act == "rewrite -z -l false":
        imap.rewrite(zero_gain=True, level_preserve=False)
    elif act == "refactor":
        imap.refactor()
    elif act == "refactor -z":
        imap.refactor(zero_gain=True)
    elif act == "refactor -l false":
        imap.refactor(level_preserve=False)
    elif act == "refactor -z -l false":
        imap.refactor(zero_gain=True, level_preserve=False)
    elif act == "map_fpga":
        imap.map_fpga()


def clac(json_data: JsonType, outfile: str):
    data = {
        "type": [],
        "aig": [],
        "nodes": [],
        "edges": [],
        "balance": [],
        "rewrite": [],
        "rewrite -z": [],
        "rewrite -l false": [],
        "rewrite -z -l false": [],
        "refactor": [],
        "refactor -z": [],
        "refactor -l false": [],
        "refactor -z -l false": [],
        # "map_fpga": [],
    }

    def calcOne(d: JsonKeyType):
        l: JsonValueType = json_data[d]

        for aig, nodes, edges in l:
            data["type"].append(d)
            data["aig"].append(aig)
            data["nodes"].append(nodes)
            data["edges"].append(edges)
            for act in Actions:
                t0 = time()
                action(aig, act)
                t1 = time()
                t = (t1 - t0) * 1000
                data[act].append(t)
                print(aig, act, t)

    calcOne("_1000")
    calcOne("_4000")
    calcOne("_7000")
    calcOne("_10000")
    calcOne("_20000")
    calcOne("_40000")
    calcOne("_else")
    df = pd.DataFrame.from_dict(data, orient="index")
    df.to_excel(outfile)
    return data


if __name__ == "__main__":
    json_data = genJson()
    clac(json_data, EXCEL_PATH)
