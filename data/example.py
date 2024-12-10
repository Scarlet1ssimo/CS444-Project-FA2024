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
# exp: num_residual_blocks=4 + batch normalization + relu
# exp_2: num_residual_blocks=2 + batch normalization + relu
# exp_3: num_residual_blocks=3 + batch normalization + relu
# exp_f: num_residual_blocks=4 without batch normalization + relu
# exp_relu6: num_residual_blocks=4 + batch normalization + relu6
# exp_gelu: num_residual_blocks=4 + batch normalization + gelu
# exp_leakyrelu: num_residual_blocks=4 + batch normalization + leakyrelu
for model, path in [
    (Model(num_residual_blocks=4, use_bn=1, nonlinearity="relu"),"exp"),
    (Model(num_residual_blocks=2, use_bn=1, nonlinearity="relu"),"exp_2"),
    (Model(num_residual_blocks=3, use_bn=1, nonlinearity="relu"),"exp_3"),
    (Model(num_residual_blocks=4, use_bn=0, nonlinearity="relu"),"exp_f"),
    (Model(num_residual_blocks=4, use_bn=1, nonlinearity="relu6"),"exp_relu6"),
    (Model(num_residual_blocks=4, use_bn=1, nonlinearity="gelu"),"exp_gelu"),
    (Model(num_residual_blocks=4, use_bn=1, nonlinearity="leakyrelu"),"exp_leakyrelu"),
                    ]:
    print(f"### {path} ###")
    model.load_state_dict(torch.load("runs/"+path+'/10000steps.pth'))
    model = model.to(device)
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

    with open(f'result_ours_{path}.pkl', 'wb') as f:
        pickle.dump(result_ours, f)

    print(f"Successfully solved {len(result_ours['times'])} cases out of {len(result_ours['solutions'])}. Saved to result_ours_{path}.pkl")
    
    # %%
    # Visualize result
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax = ax.ravel()
    ax[0].set_ylabel("Frequency")
    ax[1].set_xlabel("Number of nodes")

    key_to_text = {
        "solution_lengths":    "Solution lengths",
        'num_nodes_generated': "Number of nodes",
        "times":               "Time (s)",
    }

    for i, k in enumerate(["solution_lengths", "num_nodes_generated", "times"]):
        v = result_ours[k]
        if k=="solution_lengths":
            v_count = result_ours['solution_lengths_count']
            ax[i].bar(v_count.keys(), v_count.values(), width=1.0)
        else:
            ax[i].hist(v)
        ax[i].axvline(np.mean(v), color="#00ffff", label=f"mean={np.mean(v):.3f}")
        ax[i].set_xlabel(key_to_text[k])
        ax[i].legend()

    for i, (key_x, key_y) in enumerate([("solution_lengths", "num_nodes_generated"), ("num_nodes_generated", "times"), ("times", "solution_lengths")]):
        i += 3
        x, y = [result_ours[k] for k in [key_x, key_y]]
        ax[i].set_xlabel(key_to_text[key_x])
        ax[i].set_ylabel(key_to_text[key_y])

        x_range = np.linspace(0, max(x), 100)
        coef = np.mean(np.squeeze(np.array(y) / np.array(x)))
        ax[i].plot(x_range, x_range * coef, label=f"slope={coef:.6f}", color="#00ffff")
        ax[i].scatter(x, y)
        ax[i].legend()

    plt.savefig(f"stat_{path}.png")

    # %% [markdown]
    # ## Comparison to DeepCubeA

    # %% [markdown]
    # ### Number of nodes vs. solution length

    # %%
    left, width = 0.12, 0.75
    bottom, height = 0.1, 0.75
    spacing = 0.0

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height, width, 0.1]
    rect_histy = [left + width, bottom, 0.1, height]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes(rect_scatter)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Solution length")
    ax.set_xscale("log")
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    ax_histx.set_ylabel("Frequency")
    ax_histy.set_xlabel("Frequency")
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax.set_ylim(15, max(result_ours['solution_lengths_count']))
    ax_histy.set_ylim(15, max(result_ours['solution_lengths_count']))

    xmin, xmax = 2.5, 8.5
    ax.set_xlim(10**xmin, 10**xmax)
    ax_histx.set_xlim(10**xmin, 10**xmax)
    bins_x = np.logspace(xmin, xmax, 100)

    ################################################################################

    key_x, key_y = "num_nodes_generated", "solution_lengths"

    for k, data in [("Optimal", result_Optimal), ("DeepCubeA", result_DeepCubeA), ("Ours", result_ours)]:
        x, y = data[key_x], data[key_y]
        ax.scatter(x, y, s=10, alpha=0.3)
        ax_histx.hist(x, bins=bins_x, alpha=0.7)

    for i, data in enumerate([result_Optimal, result_DeepCubeA, result_ours]):
        data = data["solution_lengths_count"]
        ax_histy.barh(list(data.keys()), list(data.values()), height=1, alpha=0.7)

    ax_histy.axhline(np.mean(result_ours[key_y]), ls="--", color="#EB4275")
    ax.axhline(np.mean(result_ours[key_y]), ls="--", color="#EB4275")

    ax.plot(np.mean(result_Optimal[key_x]), np.mean(result_Optimal[key_y]),     "x", markersize=12, label="Optimal")
    ax.plot(np.mean(result_DeepCubeA[key_x]), np.mean(result_DeepCubeA[key_y]), "x", markersize=12, label="DeepCubeA")
    ax.plot(np.mean(result_ours[key_x]), np.mean(result_ours[key_y]),           "x", markersize=12, label="Ours")
    ax.legend()

    plt.savefig(f"stat_qwq_{path}.png")