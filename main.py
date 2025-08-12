"""
Few-shot ARC trainer (coordinate-token variant)
-----------------------------------------------
This script implements a coordinate-token representation for ARC tasks.
Key ideas:
 - Represent each non-padding cell as a token: (x, y, color_idx)
 - Embed x, y (learned), color (learned) and sum to make token embeddings
 - Support set tokens are encoded into a task memory via a TransformerEncoder
 - Decoder takes a list of coordinate queries (the target grid coordinates) and
   cross-attends to the task memory to predict a color for each coordinate
 - Supports variable-sized inputs/outputs (we build queries for the true output
   shape during training/eval)
 - Episodic sampling and per-episode color permutation augmentation included

Usage:
  python fewshot_arc_coordinate_trainer.py --epochs 10 --batch 8

Notes & limitations:
 - For inference when the true output shape isn't provided you need a
   separate size-prediction strategy (not implemented here). During training
   and evaluation we use the ground-truth target size to form decoder queries.
 - This is an experiment-ready baseline; you should extend it with better
   masking, dataset handling, and dataset path options for large-scale runs.

Author: ChatGPT (for Sam)
"""

import argparse
import json
import math
import os
import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# ------------------------- Coordinate-based Few-shot Model ------------------------- #
class CoordFewShotModel(nn.Module):
    def __init__(self,
                 max_x: int = 30,
                 max_y: int = 30,
                 num_colors: int = 10,
                 embed_dim: int = 128,
                 enc_layers: int = 3,
                 dec_layers: int = 3,
                 nheads: int = 8,
                 max_support: int = 8,
                 max_tokens_per_support: int = 256):
        super().__init__()
        self.max_x = max_x
        self.max_y = max_y
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.max_support = max_support
        self.max_tokens_per_support = max_tokens_per_support

        # embeddings
        self.color_emb = nn.Embedding(num_colors, embed_dim)
        self.x_emb = nn.Embedding(max_x, embed_dim)
        self.y_emb = nn.Embedding(max_y, embed_dim)

        # role embeddings
        self.role_support = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.role_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # example index embedding
        self.example_emb = nn.Embedding(max_support, embed_dim)

        # transformer encoder for support memory
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nheads, batch_first=True)
        self.task_encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # transformer decoder for queries
        dec_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nheads, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        # head
        self.head = nn.Linear(embed_dim, num_colors)

    def embed_tokens(self, xs: torch.LongTensor, ys: torch.LongTensor, cols: torch.LongTensor):
        # xs, ys, cols: [*, T] with values in range
        xe = self.x_emb(xs)
        ye = self.y_emb(ys)
        ce = self.color_emb(cols)
        emb = xe + ye + ce  # [*, T, D]
        return emb

    def build_support_memory(self, support_tokens: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], support_masks: List[torch.Tensor]):
        # support_tokens: list length S of (xs, ys, cols) each [B, T_s]
        # support_masks: list length S of [B, T_s] booleans (True for valid tokens)
        B = support_tokens[0][0].shape[0]
        per_support_embs = []
        per_support_lens = []
        for idx, (xs, ys, cols) in enumerate(support_tokens):
            emb = self.embed_tokens(xs, ys, cols)  # [B, T_s, D]
            emb = emb + self.role_support  # role
            ex_emb = self.example_emb(torch.tensor(idx, device=emb.device)).view(1,1,-1)
            emb = emb + ex_emb
            per_support_embs.append(emb)
            per_support_lens.append(emb.shape[1])

        # Concatenate supports along sequence dim -> memory [B, sum(T_s), D]
        memory = torch.cat(per_support_embs, dim=1)
        # Build memory mask: True means valid (following PyTorch transformer convention for key_padding_mask)
        mem_masks = torch.cat(support_masks, dim=1)  # [B, sum(T_s)]
        # The TransformerEncoder expects src_key_padding_mask with True at positions that should be masked -> invert
        src_key_padding_mask = ~mem_masks  # True where padding
        memory = self.task_encoder(memory, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def forward(self,
                support_tokens: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                support_masks: List[torch.Tensor],
                query_coords: Tuple[torch.Tensor, torch.Tensor]):
        """
        support_tokens: list of S tuples, each tuple (xs, ys, cols) with shapes [B, T_s]
        support_masks: list of S boolean masks [B, T_s]
        query_coords: (qx, qy) each [B, Q]
        returns logits: [B, Q, num_colors]
        """
        device = support_tokens[0][0].device
        memory, mem_key_padding = self.build_support_memory(support_tokens, support_masks)

        qx, qy = query_coords
        # For queries we don't have a color; use a placeholder zero color index (learned color_emb supports it)
        dummy_cols = torch.zeros_like(qx, dtype=torch.long, device=device)
        q_emb = self.embed_tokens(qx, qy, dummy_cols) + self.role_query  # [B, Q, D]

        # PyTorch TransformerDecoder expects tgt_key_padding_mask (False for included tokens)
        tgt_key_padding_mask = None
        # memory_key_padding_mask = mem_key_padding
        dec_out = self.decoder(tgt=q_emb, memory=memory, memory_key_padding_mask=mem_key_padding)
        logits = self.head(dec_out)
        return logits

# ------------------------- Dataset loader & episodic sampling (coordinate tokens) ------------------------- #
class ARCChallengeDataset:
    def __init__(self, arc_json_path: Optional[str] = None, solutions_json_path: Optional[str] = None):
        self.tasks = {}
        if arc_json_path and os.path.exists(arc_json_path):
            with open(arc_json_path, 'r') as f:
                obj = json.load(f)
            for tid, data in obj.items():
                train_pairs = []
                for v in data.get('train', []):
                    inp = np.array(v['input'], dtype=np.int64)
                    out = np.array(v['output'], dtype=np.int64)
                    train_pairs.append((inp, out))
                test_inputs = []
                tsec = data.get('test', [])
               
                for item in tsec:
                    test_inputs.append(np.array(item['input'], dtype=np.int64))
                self.tasks[tid] = {'train_pairs': train_pairs, 'test_inputs': test_inputs}

        self.task_ids = list(self.tasks.keys())

    def __len__(self):
        return len(self.task_ids)

    def sample_episode(self, support_k: int = 2):
        tid = random.choice(self.task_ids)
        task = self.tasks[tid]
        train_pairs = task['train_pairs']
        S = min(support_k, len(train_pairs))
        supports = random.sample(train_pairs, S)
        # choose query: prefer test_inputs
        if len(task.get('test_inputs', [])) > 0:
            q_in = task['test_inputs'][0]
            q_out = None
        else:
            remaining = [p for p in train_pairs if p not in supports]
            if len(remaining) > 0:
                q = random.choice(remaining)
                q_in, q_out = q[0], q[1]
            else:
                q_in, q_out = supports[0][0], supports[0][1]

        return supports, (q_in, q_out)

# synthetic fallback
class ARCSyntheticGenerator:
    def __init__(self, n_tasks=1000, img_size=30, num_colors=8, seed=0):
        rng = np.random.RandomState(seed)
        self.tasks = []
        for _ in range(n_tasks):
            rule = rng.choice(['color_swap', 'translate', 'grow', 'fill_background'])
            train_pairs = []
            for _i in range(rng.randint(2,5)):
                inp = np.zeros((rng.randint(1,img_size), rng.randint(1,img_size)), dtype=np.int64)
                out = np.zeros_like(inp)
                if rule == 'color_swap':
                    h,w = inp.shape
                    cx, cy = rng.randint(0,w), rng.randint(0,h)
                    r = rng.randint(1, min(h,w))
                    c1 = rng.randint(1, num_colors)
                    c2 = rng.randint(1, num_colors)
                    yy, xx = np.ogrid[:h, :w]
                    mask = (xx - cx)**2 + (yy - cy)**2 <= r*r
                    inp[mask] = c1
                    out = inp.copy(); out[mask]=c2
                elif rule == 'translate':
                    h,w = inp.shape
                    s = min(h,w, rng.randint(1,4))
                    x0 = rng.randint(0,w-s+1)
                    y0 = rng.randint(0,h-s+1)
                    c = rng.randint(1, num_colors)
                    inp[y0:y0+s, x0:x0+s] = c
                    dx,dy = rng.randint(-2,3), rng.randint(-2,3)
                    out = np.zeros_like(inp)
                    nx0 = max(0, x0+dx); ny0 = max(0, y0+dy)
                    out[ny0:ny0+s, nx0:nx0+s] = c
                elif rule == 'grow':
                    h,w = inp.shape
                    cx, cy = rng.randint(0,w), rng.randint(0,h)
                    r = rng.randint(0, min(h,w)//2)
                    c = rng.randint(1, num_colors)
                    yy, xx = np.ogrid[:h, :w]
                    mask = (xx - cx)**2 + (yy - cy)**2 <= r*r
                    inp[mask]=c
                    big = (xx-cx)**2 + (yy-cy)**2 <= (r+1)*(r+1)
                    out[big]=c
                else:
                    h,w = inp.shape
                    bg = rng.randint(0,num_colors)
                    inp[:,:]=bg
                    for i in range(rng.randint(1,4)):
                        xi = rng.randint(0,w); yi = rng.randint(0,h)
                        inp[yi,xi]=rng.randint(1,num_colors)
                    out = inp.copy()
                train_pairs.append((inp, out))
            self.tasks.append({'train_pairs': train_pairs, 'test_inputs':[train_pairs[0][0]], 'test_solutions':[train_pairs[0][1]]})

    def as_dataset(self):
        ds = ARCChallengeDataset()
        ds.tasks = {str(i): t for i, t in enumerate(self.tasks)}
        ds.task_ids = list(ds.tasks.keys())
        return ds

# ----------------------- Helpers: convert grids -> coordinate tokens ----------------------- #

def grid_to_coord_tokens(grid: np.ndarray):
    # returns xs, ys, cols arrays for non-padding cells
    h, w = grid.shape
    xs = []
    ys = []
    cs = []
    for y in range(h):
        for x in range(w):
            # include every cell (we include zeros too, as they may be meaningful)
            xs.append(x)
            ys.append(y)
            cs.append(int(grid[y, x]))
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64), np.array(cs, dtype=np.int64)


def pad_token_list(arr: np.ndarray, max_len: int, pad_value: int = 0):
    L = arr.shape[0]
    if L >= max_len:
        return arr[:max_len]
    out = np.full((max_len,), pad_value, dtype=arr.dtype)
    out[:L] = arr
    return out

# ----------------------- Batch builder (with augmentation) ----------------------- #

def build_episode_batch(ds: ARCChallengeDataset, batch: int, support_k: int, max_x: int, max_y: int, num_colors: int, max_tokens_per_support: int, device: torch.device):
    # For each example in batch, sample an episode and convert supports into token tensors
    B = batch
    S_list = []
    support_xs = []  # list of length S of tensors [B, T]
    support_ys = []
    support_cols = []
    support_masks = []
    query_qx = []
    query_qy = []
    query_targets = []

    for b in range(B):
        supports, (q_in, q_out) = ds.sample_episode(support_k)
        # per-episode color permutation augmentation
        perm = np.arange(num_colors)
        np.random.shuffle(perm)

        # process supports
        for s_idx in range(support_k):
            if s_idx < len(supports):
                inp, out = supports[s_idx]
                h,w = inp.shape
                # apply perm
                inp = perm[inp]
                out = perm[out]
                # convert to tokens for the input -> it's useful to include both input and output tokens
                xs_in, ys_in, cols_in = grid_to_coord_tokens(inp)
                xs_out, ys_out, cols_out = grid_to_coord_tokens(out)
                # we'll represent this support by concatenating input tokens then output tokens
                xs = np.concatenate([xs_in, xs_out], axis=0)
                ys = np.concatenate([ys_in, ys_out], axis=0)
                cols = np.concatenate([cols_in, cols_out], axis=0)
            else:
                # empty support (pad)
                xs = np.zeros((0,), dtype=np.int64)
                ys = np.zeros((0,), dtype=np.int64)
                cols = np.zeros((0,), dtype=np.int64)

            # pad/truncate tokens to max_tokens_per_support
            xs_p = pad_token_list(xs, max_tokens_per_support, pad_value=0)
            ys_p = pad_token_list(ys, max_tokens_per_support, pad_value=0)
            cols_p = pad_token_list(cols, max_tokens_per_support, pad_value=0)
            mask = np.zeros((max_tokens_per_support,), dtype=bool)
            mask[:min(xs.shape[0], max_tokens_per_support)] = True

            # append into per-support buckets
            if len(support_xs) <= s_idx:
                support_xs.append([]); support_ys.append([]); support_cols.append([]); support_masks.append([])
            support_xs[s_idx].append(xs_p)
            support_ys[s_idx].append(ys_p)
            support_cols[s_idx].append(cols_p)
            support_masks[s_idx].append(mask)

        # query tokens (we use the output shape if available; else fallback to input shape)
        if q_out is not None:
            tgt = perm[q_out]
            h_t, w_t = tgt.shape
            qx = [];
            qy = []; tgt_flat = []
            for yy in range(h_t):
                for xx in range(w_t):
                    qx.append(xx)
                    qy.append(yy)
                    tgt_flat.append(int(tgt[yy, xx]))
            query_qx.append(np.array(qx, dtype=np.int64))
            query_qy.append(np.array(qy, dtype=np.int64))
            query_targets.append(np.array(tgt_flat, dtype=np.int64))
        else:
            # unknown target; create query from input size (best-effort)
            q_in_p = perm[q_in]
            h_t, w_t = q_in_p.shape
            qx = [];
            qy = []; tgt_flat = []
            for yy in range(h_t):
                for xx in range(w_t):
                    qx.append(xx); qy.append(yy); tgt_flat.append(int(0))
            query_qx.append(np.array(qx, dtype=np.int64))
            query_qy.append(np.array(qy, dtype=np.int64))
            query_targets.append(np.array(tgt_flat, dtype=np.int64))

    # Now convert lists into tensors
    # support_* are lists of length S each containing B arrays of shape [T]
    S = len(support_xs)
    T = max_tokens_per_support
    support_xs_t = []
    support_ys_t = []
    support_cols_t = []
    support_masks_t = []
    for s_idx in range(S):
        arr_x = np.stack(support_xs[s_idx], axis=0)  # [B, T]
        arr_y = np.stack(support_ys[s_idx], axis=0)
        arr_c = np.stack(support_cols[s_idx], axis=0)
        arr_m = np.stack(support_masks[s_idx], axis=0)
        support_xs_t.append(torch.tensor(arr_x, dtype=torch.long, device=device))
        support_ys_t.append(torch.tensor(arr_y, dtype=torch.long, device=device))
        support_cols_t.append(torch.tensor(arr_c, dtype=torch.long, device=device))
        support_masks_t.append(torch.tensor(arr_m, dtype=torch.bool, device=device))

    # queries: variable lengths Q_b -> pad to max_Q
    Q_lens = [len(q) for q in query_qx]
    Q_max = max(Q_lens)
    qx_p = np.zeros((B, Q_max), dtype=np.int64)
    qy_p = np.zeros((B, Q_max), dtype=np.int64)
    qmask = np.zeros((B, Q_max), dtype=np.bool_)
    tgt_p = np.zeros((B, Q_max), dtype=np.int64)
    for b in range(B):
        L = Q_lens[b]
        qx_p[b, :L] = query_qx[b]
        qy_p[b, :L] = query_qy[b]
        qmask[b, :L] = True
        tgt_p[b, :L] = query_targets[b]

    qx_t = torch.tensor(qx_p, dtype=torch.long, device=device)
    qy_t = torch.tensor(qy_p, dtype=torch.long, device=device)
    qmask_t = torch.tensor(qmask, dtype=torch.bool, device=device)
    tgt_t = torch.tensor(tgt_p, dtype=torch.long, device=device)

    return support_xs_t, support_ys_t, support_cols_t, support_masks_t, qx_t, qy_t, qmask_t, tgt_t

# ------------------------- Training & Evaluation ------------------------- #

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # load or synthesize dataset
    if args.arc_json and os.path.exists(args.arc_json):
        ds = ARCChallengeDataset(args.arc_json, args.solutions_json)
        print(f"Loaded train set with {len(ds)} tasks from {args.eval_json}")

        if len(ds) == 0:
            print('No tasks found in provided JSON. Falling back to synthetic generator')
            ds = ARCSyntheticGenerator(n_tasks=200, img_size=args.max_x, num_colors=args.num_colors).as_dataset()
    else:
        print('No ARC JSON provided; using synthetic')
        ds = ARCSyntheticGenerator(n_tasks=200, img_size=args.max_x, num_colors=args.num_colors).as_dataset()

    # optional separate evaluation dataset
    if args.eval_arc_json and os.path.exists(args.eval_arc_json):
        eval_ds = ARCChallengeDataset(args.eval_arc_json, args.eval_solutions_json)
        print(f"Loaded separate eval set with {len(eval_ds)} tasks from {args.eval_arc_json}")
    else:
        eval_ds = ds
        print("Using training dataset for evaluation (no separate eval set provided)")

    model = CoordFewShotModel(max_x=args.max_x, max_y=args.max_y, num_colors=args.num_colors,
                              embed_dim=args.embed_dim, enc_layers=args.enc_layers,
                              dec_layers=args.dec_layers, nheads=args.nheads,
                              max_support=args.max_support, max_tokens_per_support=args.max_tokens_per_support).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_exact = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct_cells = 0
        total_cells = 0

        with tqdm(range(args.iters_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}", unit="iter") as pbar:
            for it in pbar:
                sup_xs_t, sup_ys_t, sup_cols_t, sup_masks_t, qx_t, qy_t, qmask_t, tgt_t = \
                    build_episode_batch(ds, args.batch, args.support_k, args.max_x, args.max_y, args.num_colors, args.max_tokens_per_support, device)

                logits = model(list(zip(sup_xs_t, sup_ys_t, sup_cols_t)), sup_masks_t, (qx_t, qy_t))  # [B, Qmax, C]
                B, Qmax, C = logits.shape
                # mask out padding positions in loss
                logits_flat = logits.view(-1, C)
                tgt_flat = tgt_t.view(-1)
                mask_flat = qmask_t.view(-1)
                loss = F.cross_entropy(logits_flat[mask_flat], tgt_flat[mask_flat])

                # accuracy on non-padding tokens
                with torch.no_grad():
                    preds = logits.argmax(dim=-1).view(-1)
                    correct = (preds[mask_flat] == tgt_flat[mask_flat]).sum().item()
                    count = int(mask_flat.sum().item())
                    total_correct_cells += int(correct)
                    total_cells += count

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += float(loss.item())
                avg_loss = total_loss / (it + 1)
                avg_acc = (total_correct_cells / total_cells) if total_cells > 0 else 0.0
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "train_acc": f"{avg_acc:.3f}"})

        avg_loss = total_loss / args.iters_per_epoch
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.4f}")

        exact, percell = eval_model(model, eval_ds, args, device)
        print(f" Eval exact acc: {exact:.3f}, per-cell acc: {percell:.3f}")
        if exact > best_exact:
            best_exact = exact
            torch.save(model.state_dict(), args.checkpoint)
            print('Saved best model')


def eval_model(model: nn.Module, ds: ARCChallengeDataset, args, device):
    model.eval()
    n = args.eval_episodes
    exact_count = 0
    correct_cells = 0
    total_cells = 0
    with torch.no_grad():
        with tqdm(range(n), desc="Eval", unit="ep") as pbar:
            for _ in pbar:
                sup_xs_t, sup_ys_t, sup_cols_t, sup_masks_t, qx_t, qy_t, qmask_t, tgt_t = \
                    build_episode_batch(ds, 1, args.support_k, args.max_x, args.max_y, args.num_colors, args.max_tokens_per_support, device)
                logits = model(list(zip(sup_xs_t, sup_ys_t, sup_cols_t)), sup_masks_t, (qx_t, qy_t))
                preds = logits.argmax(dim=-1).cpu().numpy()[0]  # [Qmax]
                mask = qmask_t.cpu().numpy()[0]
                tgt = tgt_t.cpu().numpy()[0]
                # compute per-cell
                correct = (preds[mask] == tgt[mask]).sum()
                correct_cells += int(correct)
                total_cells += int(mask.sum())
                if (preds[mask] == tgt[mask]).all():
                    exact_count += 1

                percell_so_far = (correct_cells / total_cells) if total_cells > 0 else 0.0
                denom = max(pbar.n, 1)
                pbar.set_postfix({"exact": f"{exact_count/denom:.3f}", "per_cell": f"{percell_so_far:.3f}"})
    return exact_count / n, correct_cells / total_cells

# -------------------------- CLI ------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--max_x', type=int, default=30)
    p.add_argument('--max_y', type=int, default=30)
    p.add_argument('--num_colors', type=int, default=10)
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--enc_layers', type=int, default=3)
    p.add_argument('--dec_layers', type=int, default=3)
    p.add_argument('--nheads', type=int, default=8)
    p.add_argument('--max_support', type=int, default=4)
    p.add_argument('--max_tokens_per_support', type=int, default=256)
    p.add_argument('--support_k', type=int, default=2)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--iters_per_epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--arc_json', type=str, default=None)
    p.add_argument('--solutions_json', type=str, default=None)
    p.add_argument('--eval_arc_json', type=str, default=None)
    p.add_argument('--eval_solutions_json', type=str, default=None)
    p.add_argument('--checkpoint', type=str, default='coord_fewshot.pt')
    p.add_argument('--eval_episodes', type=int, default=100)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)
