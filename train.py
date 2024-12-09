import os
from pathlib import Path
import time
import random
import pickle
import numpy as np
from copy import deepcopy
from contextlib import nullcontext
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
from cycler import cycler; plt.rcParams["axes.prop_cycle"] = cycler(color=["#000000", "#2180FE", "#EB4275"])
from IPython.display import clear_output

import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'device: {device}')
print(f'os.cpu_count(): {os.cpu_count()}')
from cube import Cube
env = Cube()

from config import TrainConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default="./")
parser.add_argument("--num-residual-blocks", type=int, default=4)
parser.add_argument("--use-bn", type=bool, default=True)
parser.add_argument("--nonlinearity", type=str, default="relu")
args = parser.parse_args()
if args.nonlinearity == "relu":
    nonlinearity=nn.ReLU()
elif args.nonlinearity == "leakyrelu":
    nonlinearity=nn.LeakyReLU(0.01)
elif args.nonlinearity == "gelu":
    nonlinearity=nn.GELU()
elif args.nonlinearity == "ReLU6":
    nonlinearity=nn.ReLU6()
output_dir=Path(args.output_dir)

from model import Model
model = Model(num_residual_blocks=args.num_residual_blocks, use_bn=args.use_bn, nonlinearity=nonlinearity)
model.to(device)

class ScrambleGenerator(torch.utils.data.Dataset):
    def __init__(
        self,
        num_workers=os.cpu_count(),
        max_depth=TrainConfig.max_depth,
        total_samples=TrainConfig.num_steps*TrainConfig.batch_size_per_depth
    ):
        self.num_workers = num_workers
        self.max_depth = max_depth
        # self.envs = [Cube() for _ in range(num_workers)]
        # self.generators = [env.scrambler(self.max_depth) for env in self.envs]

        self.total_samples = total_samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, i):
        ''' generate one scramble, consisting of `self.max_depth` data points '''
        # worker_idx = i % self.num_workers
        try:
            X = np.zeros((self.max_depth, 54), dtype=np.int64)
            y = np.zeros((self.max_depth,), dtype=np.int64)
            generator = Cube().scrambler(self.max_depth)
            for j in range(self.max_depth):
                state, last_move = next(generator)
                X[j, :] = state
                y[j] = last_move
            return X, y
        except Exception as e:
            print(f'error: {e}')
            return self.__getitem__(i)


dataloader = torch.utils.data.DataLoader(
    ScrambleGenerator(),
    num_workers=0,
    batch_size=TrainConfig.batch_size_per_depth
)


def plot_loss_curve(h):
    fig, ax = plt.subplots(1, 1)
    ax.plot(h)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_xscale("log")
    plt.savefig(output_dir/"loss_curve.png")

def train(model, dataloader):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainConfig.learning_rate)
    g = iter(dataloader)
    h = []
    ctx = torch.cuda.amp.autocast(dtype=torch.float16) if TrainConfig.ENABLE_FP16 else nullcontext()

    for i in trange(1, TrainConfig.num_steps + 1):
        batch_x, batch_y = next(g)
        batch_x, batch_y = batch_x.reshape(-1, 54).to(device), batch_y.reshape(-1).to(device)

        with ctx:
            pred_y = model(batch_x)
            loss = loss_fn(pred_y, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        h.append(loss.item())
        if TrainConfig.INTERVAL_PLOT and i % TrainConfig.INTERVAL_PLOT == 0:
            clear_output()
            plot_loss_curve(h)
        if TrainConfig.INTERVAL_SAVE and i % TrainConfig.INTERVAL_SAVE == 0:
            torch.save(model.state_dict(), output_dir/f"{i}steps.pth")
            print("Model saved.")
    print(f"Trained on data equivalent to {TrainConfig.batch_size_per_depth * TrainConfig.num_steps} solves.")
    return model

model = train(model, dataloader)