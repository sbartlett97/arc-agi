import argparse
import json
import math
import os
import random
import shutil
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
try:
    import optuna  # type: ignore
except Exception:
    optuna = None  # type: ignore

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

class ARCChallengeDataset:
    def __init__(self, arc_json_path: Optional[str] = None, solutions_json_path: Optional[str] = None):
        self.tasks = {}
        # Load solutions map if provided
        solutions_map: Dict[str, Any] = {}
        if solutions_json_path and os.path.exists(solutions_json_path):
            with open(solutions_json_path, 'r') as fsol:
                try:
                    solutions_map = json.load(fsol)
                except Exception:
                    solutions_map = {}

        if arc_json_path and os.path.exists(arc_json_path):
            with open(arc_json_path, 'r') as f:
                obj = json.load(f)

            def is_grid(x: Any) -> bool:
                # A grid is a list of rows (lists of ints)
                if not (isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)):
                    return False
                # If the inner element is a list of ints (or empty), it's a grid
                inner = x[0]
                return len(inner) == 0 or not isinstance(inner[0], list)

            def is_list_of_grids(x: Any) -> bool:
                # A list of grids is list where first element is itself a grid (list of rows)
                if not (isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)):
                    return False
                inner = x[0]
                return isinstance(inner, list) and len(inner) > 0 and isinstance(inner[0], list)

            for tid, data in obj.items():
                # Parse train pairs; accept either list or dict forms
                train_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
                train_section = data.get('train', [])
                if isinstance(train_section, dict):
                    train_iterable = train_section.values()
                elif isinstance(train_section, list):
                    train_iterable = train_section
                else:
                    train_iterable = []

                for example in train_iterable:
                    if not isinstance(example, dict):
                        continue
                    if 'input' in example and 'output' in example:
                        inp = np.array(example['input'], dtype=np.int64)
                        out = np.array(example['output'], dtype=np.int64)
                        train_pairs.append((inp, out))

                # Parse test inputs; accept classic ARC format or ARC-AGI simplification
                test_inputs: List[np.ndarray] = []
                tsec = data.get('test', [])
                if isinstance(tsec, list):
                    if len(tsec) > 0 and isinstance(tsec[0], dict) and 'input' in tsec[0]:
                        for item in tsec:
                            test_inputs.append(np.array(item['input'], dtype=np.int64))
                    elif is_grid(tsec):
                        # Single test grid directly as a 2D array
                        test_inputs.append(np.array(tsec, dtype=np.int64))
                    else:
                        # Unknown list structure; ignore
                        pass
                elif isinstance(tsec, dict):
                    if 'input' in tsec:
                        test_inputs.append(np.array(tsec['input'], dtype=np.int64))
                else:
                    # No test section or unrecognized format
                    pass

                # Parse test solutions if present
                test_solutions: List[np.ndarray] = []
                if tid in solutions_map:
                    sols = solutions_map[tid]
                    # Solutions may be a single grid or a list of grids
                    if is_grid(sols):
                        test_solutions = [np.array(sols, dtype=np.int64)]
                    elif is_list_of_grids(sols):
                        for s in sols:
                            if is_grid(s):
                                test_solutions.append(np.array(s, dtype=np.int64))
                    elif isinstance(sols, list):
                        # Defensive: iterate and collect any grids inside
                        for s in sols:
                            if is_grid(s):
                                test_solutions.append(np.array(s, dtype=np.int64))
                    # else: ignore unknown formats

                # Align lengths of inputs and solutions if both exist
                if len(test_inputs) > 0 and len(test_solutions) > 0:
                    L = min(len(test_inputs), len(test_solutions))
                    test_inputs = test_inputs[:L]
                    test_solutions = test_solutions[:L]

                self.tasks[tid] = {
                    'train_pairs': train_pairs,
                    'test_inputs': test_inputs,
                    'test_solutions': test_solutions,
                }

        self.task_ids = list(self.tasks.keys())

    def __len__(self):
        return len(self.task_ids)

    def sample_episode(self, support_k: int = 2):
        tid = random.choice(self.task_ids)
        task = self.tasks[tid]
        train_pairs = task['train_pairs']
        S = min(support_k, len(train_pairs))
        supports = random.sample(train_pairs, S)
        # choose query: prefer test inputs only when solutions are available; otherwise, use held-out train pair
        test_inputs = task.get('test_inputs', [])
        test_solutions = task.get('test_solutions', [])
        if len(test_inputs) > 0 and len(test_solutions) > 0:
            idx = random.randrange(min(len(test_inputs), len(test_solutions)))
            q_in = test_inputs[idx]
            q_out = test_solutions[idx]
        else:
            # choose a held-out training pair if possible
            remaining = []
            support_set = set(id(p[0]) for p in supports)  # compare by object id to avoid ndarray truth issues
            for p in train_pairs:
                if id(p[0]) not in support_set:
                    remaining.append(p)
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
                    r = rng.randint(1, min(h,w) + 1)
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
                    r = rng.randint(0, (min(h,w)//2) + 1)
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

    episode_details: List[Dict[str, Any]] = []
    for b in range(B):
        supports, (q_in, q_out) = ds.sample_episode(support_k)
        # per-episode color permutation augmentation
        perm = np.arange(num_colors)
        np.random.shuffle(perm)

        # process supports
        supports_perm: List[Tuple[np.ndarray, np.ndarray]] = []
        for s_idx in range(support_k):
            if s_idx < len(supports):
                inp, out = supports[s_idx]
                h,w = inp.shape
                # apply perm
                inp = perm[inp]
                out = perm[out]
                supports_perm.append((inp.copy(), out.copy()))
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
            # Ensure target is 2D grid [H, W]
            if tgt.ndim == 3:
                # Some loaders may wrap as [1, H, W]
                if tgt.shape[0] == 1:
                    tgt = tgt[0]
                else:
                    tgt = tgt.squeeze()
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
            q_in_perm = perm[q_in]
            q_out_perm = tgt
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
            q_in_perm = q_in_p
            q_out_perm = None

        episode_details.append({
            'supports': supports_perm,
            'q_in': q_in_perm,
            'q_out': q_out_perm,
            'perm': perm,
            'query_hw': (h_t, w_t),
        })

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

    return support_xs_t, support_ys_t, support_cols_t, support_masks_t, qx_t, qy_t, qmask_t, tgt_t, episode_details


# Visualization utilities
ARC_PALETTE = [
    (0, 0, 0),        # 0 black
    (0, 116, 217),    # 1 blue
    (255, 65, 54),    # 2 red
    (46, 204, 64),    # 3 green
    (255, 220, 0),    # 4 yellow
    (170, 170, 170),  # 5 gray
    (177, 13, 201),   # 6 purple
    (255, 133, 27),   # 7 orange
    (255, 153, 204),  # 8 pink
    (127, 219, 255),  # 9 cyan
]


def render_grid_image(grid: np.ndarray, cell_size: int = 16) -> Image.Image:
    if grid.ndim != 2:
        raise ValueError("Grid must be 2D")
    h, w = grid.shape
    img = Image.new('RGB', (w * cell_size, h * cell_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        for x in range(w):
            color_idx = int(grid[y, x])
            color = ARC_PALETTE[color_idx % len(ARC_PALETTE)]
            x0 = x * cell_size
            y0 = y * cell_size
            draw.rectangle([x0, y0, x0 + cell_size - 1, y0 + cell_size - 1], fill=color)
    return img


def hstack_images(images: List[Image.Image], pad: int = 4, pad_color=(255, 255, 255)) -> Image.Image:
    heights = [im.height for im in images]
    max_h = max(heights) if images else 0
    widths = [im.width for im in images]
    total_w = sum(widths) + pad * (len(images) - 1 if len(images) > 0 else 0)
    out = Image.new('RGB', (total_w, max_h), pad_color)
    x = 0
    for i, im in enumerate(images):
        out.paste(im, (x, 0))
        x += im.width
        if i < len(images) - 1:
            x += pad
    return out


def vstack_images(images: List[Image.Image], pad: int = 4, pad_color=(255, 255, 255)) -> Image.Image:
    widths = [im.width for im in images]
    max_w = max(widths) if images else 0
    heights = [im.height for im in images]
    total_h = sum(heights) + pad * (len(images) - 1 if len(images) > 0 else 0)
    out = Image.new('RGB', (max_w, total_h), pad_color)
    y = 0
    for i, im in enumerate(images):
        out.paste(im, (0, y))
        y += im.height
        if i < len(images) - 1:
            y += pad
    return out


def label_image(image: Image.Image, label: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text_w = draw.textlength(label, font=font)
    pad = 2
    bg = Image.new('RGB', (image.width, 30 + pad * 2), (240, 240, 240))
    bg_draw = ImageDraw.Draw(bg)
    bg_draw.text((pad, pad), label, fill=(0, 0, 0), font=font)
    return vstack_images([bg, image], pad=0)


def save_episode_visual(supports: List[Tuple[np.ndarray, np.ndarray]], q_in: np.ndarray, pred: np.ndarray, target: Optional[np.ndarray], out_path: str, cell_size: int = 16):
    # Row 1..n: support pairs input -> output
    rows: List[Image.Image] = []
    for idx, (inp, out) in enumerate(supports):
        im_in = render_grid_image(inp, cell_size)
        im_out = render_grid_image(out, cell_size)
        row = hstack_images([label_image(im_in, f"S{idx} in"), label_image(im_out, f"S{idx} out")], pad=8)
        rows.append(row)

    # Last row: test input, prediction, and (optional) target
    im_q = render_grid_image(q_in, cell_size)
    im_pred = render_grid_image(pred, cell_size)
    comps = [label_image(im_q, "query in"), label_image(im_pred, "prediction")]
    if target is not None:
        im_tgt = render_grid_image(target, cell_size)
        comps.append(label_image(im_tgt, "target"))
    last_row = hstack_images(comps, pad=8)
    rows.append(last_row)

    canvas = vstack_images(rows, pad=8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # load or synthesize dataset
    if args.arc_json and os.path.exists(args.arc_json):
        ds = ARCChallengeDataset(args.arc_json, args.solutions_json)
        print(f"Loaded train set with {len(ds)} tasks from {args.arc_json}")

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

    best_exact = float('-inf')
    saved_any_checkpoint = False
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct_cells = 0
        total_cells = 0

        with tqdm(range(args.iters_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}", unit="iter") as pbar:
            for it in pbar:
                sup_xs_t, sup_ys_t, sup_cols_t, sup_masks_t, qx_t, qy_t, qmask_t, tgt_t, _ = \
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
        if exact >= best_exact:
            best_exact = exact
            torch.save(model.state_dict(), args.checkpoint)
            print('Saved best model')
            saved_any_checkpoint = True

    # Ensure a checkpoint exists even if no improvement was recorded
    if not saved_any_checkpoint or not os.path.exists(args.checkpoint):
        torch.save(model.state_dict(), args.checkpoint)
        print('Saved final model (no prior best recorded)')

    return best_exact


def run_optuna(args):
    if optuna is None:
        raise RuntimeError("optuna is not installed. Please install it or add it to requirements.")

    os.makedirs(args.optuna_checkpoint_dir, exist_ok=True)

    direction = 'maximize'
    sampler = optuna.samplers.TPESampler(seed=getattr(args, 'optuna_seed', None))
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=getattr(args, 'optuna_warmup_trials', 0))

    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner, study_name=getattr(args, 'study_name', None))

    def objective(trial: Any) -> float:
        # Suggest hyperparameters
        embed_dim = trial.suggest_categorical('embed_dim', [64, 96, 128, 160])
        enc_layers = trial.suggest_int('enc_layers', 2, 4)
        dec_layers = trial.suggest_int('dec_layers', 2, 4)
        nheads = trial.suggest_categorical('nheads', [4, 8])
        lr = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
        support_k = trial.suggest_int('support_k', 1, min(4, args.max_support))
        max_tokens_per_support = trial.suggest_categorical('max_tokens_per_support', [128, 192, 256, 320])
        batch = trial.suggest_categorical('batch', [4, 6, 8])

        # Clone args into a simple namespace
        from copy import deepcopy
        local_args = deepcopy(args)
        local_args.embed_dim = embed_dim
        local_args.enc_layers = enc_layers
        local_args.dec_layers = dec_layers
        local_args.nheads = nheads
        local_args.lr = lr
        local_args.support_k = support_k
        local_args.max_tokens_per_support = max_tokens_per_support
        local_args.batch = batch

        # Trial-specific checkpoint
        trial_ckpt = os.path.join(args.optuna_checkpoint_dir, f"trial_{trial.number}.pt")
        local_args.checkpoint = trial_ckpt

        # Optionally shorten runs during tuning
        if getattr(args, 'optuna_fast_mode', False):
            local_args.epochs = max(1, args.epochs // 2)
            local_args.iters_per_epoch = max(1, args.iters_per_epoch // 4)
            local_args.eval_episodes = max(10, args.eval_episodes // 5)

        best_exact = train(local_args)
        # Report the final result
        trial.report(best_exact, step=local_args.epochs)
        return best_exact

    study.optimize(objective, n_trials=args.optuna_trials, n_jobs=1, show_progress_bar=True)

    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")

    # Copy best checkpoint to main checkpoint path to ensure it's always saved at a predictable location
    best_trial_num = study.best_trial.number
    best_trial_ckpt = os.path.join(args.optuna_checkpoint_dir, f"trial_{best_trial_num}.pt")
    if not os.path.exists(best_trial_ckpt):
        raise FileNotFoundError(f"Expected checkpoint for best trial not found: {best_trial_ckpt}")
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(args.checkpoint) or '.', exist_ok=True)
    shutil.copy2(best_trial_ckpt, args.checkpoint)
    print(f"Saved overall best model to {args.checkpoint}")


def eval_model(model: nn.Module, ds: ARCChallengeDataset, args, device):
    model.eval()
    n = args.eval_episodes
    exact_count = 0
    correct_cells = 0
    total_cells = 0
    with torch.no_grad():
        with tqdm(range(n), desc="Eval", unit="ep") as pbar:
            for _ in pbar:
                sup_xs_t, sup_ys_t, sup_cols_t, sup_masks_t, qx_t, qy_t, qmask_t, tgt_t, details = \
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

                # Optional visualization dump
                if getattr(args, 'vis_dir', None):
                    det = details[0]
                    h_t, w_t = det['query_hw']
                    # reshape preds
                    pred_grid = np.zeros((h_t, w_t), dtype=np.int64)
                    qx_np = qx_t.cpu().numpy()[0]
                    qy_np = qy_t.cpu().numpy()[0]
                    for i in range(len(qx_np)):
                        if mask[i]:
                            pred_grid[qy_np[i], qx_np[i]] = preds[i]

                    target_grid = None
                    if det['q_out'] is not None:
                        target_grid = det['q_out']
                    save_path = os.path.join(args.vis_dir, f"episode_{pbar.n}.png")
                    save_episode_visual(det['supports'], det['q_in'], pred_grid, target_grid, save_path, cell_size=getattr(args, 'cell_size', 16))
    return exact_count / n, correct_cells / total_cells


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
    # Visualization
    p.add_argument('--vis_dir', type=str, default=None, help='Directory to save episode visualizations during eval')
    p.add_argument('--cell_size', type=int, default=16, help='Cell size for visualization images')
    # Optuna
    p.add_argument('--optuna_trials', type=int, default=0, help='Number of Optuna trials to run; 0 disables tuning')
    p.add_argument('--optuna_checkpoint_dir', type=str, default='optuna_checkpoints', help='Directory to store per-trial checkpoints')
    p.add_argument('--study_name', type=str, default=None, help='Optional Optuna study name')
    p.add_argument('--optuna_fast_mode', action='store_true', help='Speed up each trial by reducing epochs/iters/eval')
    p.add_argument('--optuna_seed', type=int, default=None, help='Random seed for Optuna sampler')
    p.add_argument('--optuna_warmup_trials', type=int, default=0, help='Warmup trials before pruner considers pruning')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if getattr(args, 'optuna_trials', 0) and args.optuna_trials > 0:
        run_optuna(args)
    else:
        train(args)
