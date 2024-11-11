# %%
import os
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

# %%
from cube import Cube
env = Cube()

# %%
from model import Model
model = Model()
model.to(device)

# %%
from config import TrainConfig


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

# Get one batch from the dataloader
sample_batch = next(iter(dataloader))

# Unpack X and y from the batch
X, y = sample_batch

# Check shapes and data types
print("X shape:", X.shape)  # Expected: (batch_size, max_depth, 54)
print("y shape:", y.shape)  # Expected: (batch_size, max_depth)
print("X dtype:", X.dtype)
print("y dtype:", y.dtype)

# Optional: Print the first few entries to inspect contents
print("Sample X:", X[0])  # Print the first sample in X
print("Sample y:", y[0])  # Print the first sample in y
print("Sample y:", y[1])  # Print the first sample in y


# %%
def plot_loss_curve(h):
    fig, ax = plt.subplots(1, 1)
    ax.plot(h)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_xscale("log")
    plt.show()


def train(model, dataloader):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=TrainConfig.learning_rate)
    g = iter(dataloader)
    h = []
    ctx = torch.cuda.amp.autocast(
        dtype=torch.float16) if TrainConfig.ENABLE_FP16 else nullcontext()

    for i in trange(1, TrainConfig.num_steps + 1):
        batch_x, batch_y = next(g)
        batch_x, batch_y = batch_x.reshape(-1,
                                           54).to(device), batch_y.reshape(-1).to(device)

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
            torch.save(model.state_dict(), f"{i}steps.pth")
            print("Model saved.")
    print(
        f"Trained on data equivalent to {TrainConfig.batch_size_per_depth * TrainConfig.num_steps} solves.")
    return model


train_here = False
if train_here:
    model = train(model, dataloader)
else:
    print('training is disabled')
    model.load_state_dict(torch.load('10000steps.pth'))

# %%
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
                pred = model(torch.tensor(state, dtype=torch.int64).to(
                    device)).argmax().item()
                if pred == move:
                    sub_correct += 1
                sub_cnt+=1
            correct += sub_correct
            cnt += sub_cnt
            print(f"Accuracy for subtask: {sub_correct} / {sub_cnt} = {sub_correct/sub_cnt:.2%}. Total accuracy: {correct / cnt:.2%}")
    return correct / cnt


# validate(model)

# %%
import sys
import subprocess
if "DeepCubeA"!=os.getcwd().split("/")[-1]:
    if not os.path.exists('DeepCubeA'):
        # Clone the repository using subprocess
        subprocess.run(["git", "clone", "-q", "https://github.com/forestagostinelli/DeepCubeA"], check=True)
    deepcubea_path = os.path.join(os.getcwd(), 'DeepCubeA')
    if deepcubea_path not in sys.path:
        sys.path.insert(0, deepcubea_path)

print('### Optimal Solver ###')
filename = 'DeepCubeA/data/cube3/test/data_0.pkl'
with open(filename, 'rb') as f:
    result_Optimal = pickle.load(f)

    print(result_Optimal.keys())
    result_Optimal["solution_lengths"] = [len(s) for s in result_Optimal["solutions"]]
    result_Optimal["solution_lengths_count"] = {
        i: result_Optimal["solution_lengths"].count(i)
        for i in range(min(result_Optimal["solution_lengths"]), max(result_Optimal["solution_lengths"]))
    }

    print('No. of cases:', len(result_Optimal["solution_lengths"]))

print('\n### DeepCubeA ###')
filename = 'DeepCubeA/results/cube3/results.pkl'
with open(filename, 'rb') as f:
    result_DeepCubeA = pickle.load(f)

    print(result_DeepCubeA.keys())
    result_DeepCubeA["solution_lengths"] = [len(s) for s in result_DeepCubeA["solutions"]]
    result_DeepCubeA["solution_lengths_count"] = {
        i: result_DeepCubeA["solution_lengths"].count(i)
        for i in range(min(result_DeepCubeA["solution_lengths"]), max(result_DeepCubeA["solution_lengths"]))
    }

    print('No. of cases:', len(result_DeepCubeA["solution_lengths"]))

if deepcubea_path in sys.path:
    sys.path.remove(deepcubea_path)

# %%
# Convert optimal solutions to test scrambles
def solution2scramble(solution):
    return [m[0] if m[1] == -1 else m[0] + "'" for m in solution[::-1]]

test_scrambles = [solution2scramble(s) for s in result_Optimal["solutions"]]

print(f"""Example:\n{result_Optimal["solutions"][0]}\n-> {test_scrambles[0]}""")

# %%
from config import SearchConfig


@torch.no_grad()
def beam_search(
    env,
    model,
    beam_width=SearchConfig.beam_width,
    max_depth=SearchConfig.max_depth,
    skip_redundant_moves=True,
):
    """
    Best-first search algorithm.
    Input:
        env: A scrambled instance of the given environment.
        model: PyTorch model used to predict the next move(s).
        beam_width: Number of top solutions to return per depth.
        max_depth: Maximum depth of the search tree.
        skip_redundant_moves: If True, skip redundant moves.
    Output:
        if solved successfully:
            True, {'solutions':solution path, "num_nodes_generated":number of nodes expanded, "times":time taken to solve}
        else:
            False, None
    """
    model.eval()
    with torch.cuda.amp.autocast(dtype=torch.float16) if SearchConfig.ENABLE_FP16 else nullcontext():
        # metrics
        num_nodes_generated, time_0 = 0, time.time()
        candidates = [
            {"state": deepcopy(env.state), "path": [], "value": 1.}
        ]  # list of dictionaries

        for depth in range(max_depth+1):
            # TWO things at a time for every candidate: 1. check if solved & 2. add to batch_x
            batch_x = np.zeros(
                (len(candidates), env.state.shape[-1]), dtype=np.int64)
            for i, c in enumerate(candidates):
                c_path, env.state = c["path"], c["state"]
                if c_path:
                    env.move(c_path[-1])
                    num_nodes_generated += 1
                    if env.is_solved():
                        # Revert: array of indices => array of notations
                        c_path = [str(env.mode.NUM2CHAR[i]) for i in c_path]
                        return True, {'solutions': c_path, "num_nodes_generated": num_nodes_generated, "times": time.time()-time_0}
                batch_x[i, :] = env.state

            # after checking the nodes expanded at the deepest
            if depth == max_depth:
                print("Solution not found.")
                return False, None

            # make predictions with the trained DNN
            batch_x = torch.from_numpy(batch_x).to(device)
            batch_p = model(batch_x)
            batch_p = torch.nn.functional.softmax(batch_p, dim=-1)
            batch_p = batch_p.detach().cpu().numpy()

            # loop over candidates
            # storage for the depth-level candidates storing (path, value, index).
            candidates_next_depth = []
            for i, c in enumerate(candidates):
                c_path = c["path"]
                # output logits for the given state
                value_distribution = batch_p[i, :]
                # multiply the cumulative probability so far of the expanded path
                value_distribution *= c["value"]

                # iterate over all possible moves.
                for m, value in enumerate(value_distribution):
                    m = env.mode.getCancel(m)
                    # predicted value to expand the path with the given move.

                    if c_path and skip_redundant_moves:
                        if m not in env.mode.MOVES_NO_CANCEL[c_path[-1]]:
                            # Two mutually canceling moves
                            continue
                        elif len(c_path) > 1:
                            # if c_path[-2] == c_path[-1] == m:
                            if c_path[-2] == c_path[-1] == m:
                                # Three subsequent moves that could be one
                                continue
                            # elif (
                            #     c_path[-2][0] == m[0] and len(c_path[-2] + m) == 3
                            #     and c_path[-1][0] == env.pairing[m[0]]
                            # ):
                            # elif env.mode.isOpposite(c_path[-2], c_path[-1]) and env.mode.isSameFace(j, c_path[-2]):
                                # Two mutually canceling moves sandwiching an opposite face move
                                # continue

                    # add to the next-depth candidates unless 'continue'd.
                    candidates_next_depth.append({
                        'state': deepcopy(c['state']),
                        "path": c_path+[m],
                        "value": value,
                    })

            # sort potential paths by expected values and renew as 'candidates'
            candidates = sorted(candidates_next_depth,
                                key=lambda item: -item['value'])
            # if the number of candidates exceed that of beam width 'beam_width'
            candidates = candidates[:beam_width]

# %%
result_ours = {
    "solutions":[],
    "num_nodes_generated":[],
    "times":[]
}
for scramble in tqdm(test_scrambles, position=0):
    # reset and scramble
    env.reset()
    env.apply(scramble)
    # solve
    success, result = beam_search(env, model)
    if success:
        for k in result_ours.keys():
            result_ours[k].append(result[k])
    else:
        result_ours["solutions"].append(None)

result_ours['solution_lengths'] = [len(e) for e in result_ours['solutions'] if e]
result_ours['solution_lengths_count'] = {
    i: result_ours["solution_lengths"].count(i)
    for i in range(min(result_ours["solution_lengths"]), max(result_ours["solution_lengths"]))
}

with open('result_ours.pkl', 'wb') as f:
    pickle.dump(result_ours, f)

f"Successfully solved {len(result_ours['times'])} cases out of {len(result_ours['solutions'])}"

# %%



