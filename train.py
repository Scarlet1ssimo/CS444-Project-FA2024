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
parser.add_argument("--use-bn", type=int, default=1)
parser.add_argument("--nonlinearity", type=str, default="relu")
args = parser.parse_args()

print(f'args: {args}')
output_dir=Path(args.output_dir)

from model import Model
model = Model(num_residual_blocks=args.num_residual_blocks, use_bn=args.use_bn, nonlinearity=args.nonlinearity)
model_name = f"normal_res{args.num_residual_blocks}_bn{args.use_bn}_nonlin{args.nonlinearity}"
print(model.eval())
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
        self.total_samples = total_samples

    def update_max_depth(self, max_depth):
        self.max_depth = max_depth

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
    batch_size=round(TrainConfig.batch_size_per_depth/10) ## changing for transformer training
)

def plot_loss_curve(h):
    fig, ax = plt.subplots(1, 1)
    ax.plot(h)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_xscale("log")
    plt.show()

def train_transformer(model, dataloader):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainConfig.learning_rate)
    g = iter(dataloader)
    h = []
    ctx = torch.cuda.amp.autocast(dtype=torch.float16) if TrainConfig.ENABLE_FP16 else nullcontext()
    for i in trange(1, 10 * TrainConfig.num_steps + 1):
        batch_x, batch_y = next(g)
        batch_x, batch_y = batch_x.reshape(-1, 54).to(device), batch_y.reshape(-1).to(device)

        with ctx:
            pred_y = model(batch_x)
            loss = loss_fn(pred_y, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        h.append(loss.item())
        print(f"\t Loss: {loss.item()}")
    
    torch.save(model.state_dict(), f"{model_name}.pth")
    print("Model saved.")
    print(f"Trained on data equivalent to {TrainConfig.batch_size_per_depth * TrainConfig.num_steps} solves.")
    return model

def train_curriculum(model, generator):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=TrainConfig.learning_rate, weight_decay = 1e-4)
    
    ctx = torch.cuda.amp.autocast(
        dtype=torch.float16) if TrainConfig.ENABLE_FP16 else nullcontext()

    dataloader = torch.utils.data.DataLoader(
        generator,
        num_workers=0,
        batch_size=round(TrainConfig.batch_size_per_depth/10)   ## modified
    )
    c_losses = { }

    ## num_iterations of each max scramble length (from 1 to 26)
    for curr_max in trange(1, 27):
        depth_losses = []
        num_iterations = round(100 * curr_max)
        for i in range(num_iterations):
            generator.max_depth = curr_max

            batch_x, batch_y = next(iter(dataloader))
            batch_x, batch_y = batch_x.reshape(-1, 54).to(device), batch_y.reshape(-1).to(device)

            with ctx:
                pred_y = model(batch_x)
                loss = loss_fn(pred_y, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            depth_losses.append(loss.item())
        print(f"Loss for {curr_max} step: {np.mean(depth_losses[-10:])} with {num_iterations} iterations")
        c_losses[curr_max] = depth_losses
        
    print(f"Incremental Curriculum training done, starting training on max length scrambles")

    final_losses = []
    for i in trange(1, TrainConfig.num_steps + 1):
        generator.max_depth = 26

        batch_x, batch_y = next(iter(dataloader))
        batch_x, batch_y = batch_x.reshape(-1, 54).to(device), batch_y.reshape(-1).to(device)

        with ctx:
            pred_y = model(batch_x)
            loss = loss_fn(pred_y, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"\t Loss: {loss.item()}")

        final_losses.append(loss.item())
    print(f"training complete!")
    torch.save(model.state_dict(), f"{model_name}.pth")
    print("Model saved.")

    return model

# model = train_transformer(model, dataloader)  ## for normal training
model = train_curriculum(model, ScrambleGenerator(max_depth = 1)) ## for curriculum training

print("Running validation:")
# Validate Accuracy
def validate(model, num_samples=100):
    model.eval()
    correct = 0
    cnt = 0
    with torch.no_grad():
        for _ in range(num_samples):
            env = Cube()
            generator = env.scrambler(TrainConfig.max_depth)
            sub_correct = 0
            sub_cnt = 0
            for i in range(TrainConfig.max_depth):
                state, move = next(generator)
                input = torch.tensor(state, dtype = torch.int64).to(device).unsqueeze(0)
                pred = model(input).argmax().item()
                if pred == move:
                    sub_correct += 1
                sub_cnt+=1
            correct += sub_correct
            cnt += sub_cnt
            print(f"Accuracy for subtask: {sub_correct} / {sub_cnt} = {sub_correct/sub_cnt:.2%}. Total accuracy: {correct / cnt:.2%}")
    return correct / cnt


validate(model)