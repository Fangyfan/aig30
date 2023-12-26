import os
import random
import torch
import abc_py as abcPy
from typing import Literal, IO
from typing_extensions import TypeAlias
from torch import nn
import dgl
from datetime import datetime
from colorama import Fore, Back, Style
import json
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
	"""读取 aig 训练集"""

	def __init__(self) -> None:
		self.aigs: list[str] = []
		# for f in os.listdir("../aig30/dataset/small"):
		#     f = os.path.join("../aig30/dataset/small", f)
		#     self.aigs.append(f)

		for f in os.listdir("../aig30/testset"):
			f = os.path.join("../aig30/testset", f)
			self.aigs.append(f)

		# self.json_data = readJson()
		# for aig, nodes, edges in self.json_data["_4000"]:
		#     self.aigs.append(aig)

	def random(self):
		now = random.randint(0, len(self.aigs) - 1)
		return self.aigs[now]


INPUT_DIR = "../aig30/dataset/test"
JSON_PATH = "../aig30/dataset/test.json"
EXCEL_PATH = "../aig30/dataset/test_cost.xlsx"

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
		abc = abcPy.AbcInterface()
		abc.start()
		abc.read(aig)
		print(aig)
		# print(abc.aigStats().numAnd, abc.aigStats().lev)
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
	abc = abcPy.AbcInterface()
	abc.start()
	abc.read(aig)
	if act == "balance":
		abc.balance(l=False)
	elif act == "rewrite":
		abc.rewrite(l=False)
	elif act == "rewrite -z":
		abc.rewrite(l=False, z=True)
	elif act == "rewrite -l false":
		abc.rewrite(l=False)
	elif act == "rewrite -z -l false":
		abc.rewrite(l=False, z=True)
	elif act == "refactor":
		abc.refactor(l=False)
	elif act == "refactor -z":
		abc.refactor(l=False, z=True)
	elif act == "refactor -l false":
		abc.refactor(l=False)
	elif act == "refactor -z -l false":
		abc.refactor(l=False, z=True)
	abc.end()
	del abc


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
