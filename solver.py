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
from config import SearchConfig
from cube import QTM
import torch
import time
from model import Model, TransformerModel, TransformerModel2

## for normal model:
model = Model()
model.load_state_dict(torch.load("10000steps.pth", map_location=torch.device('cpu')))

## for transformer model 2:
# model = TransformerModel2()
# model.load_state_dict(torch.load("final_transformer.pth", map_location=torch.device('cpu')))


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

def my_solver(env):
    return beam_search(env, model)