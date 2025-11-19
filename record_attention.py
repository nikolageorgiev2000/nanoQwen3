#!/usr/bin/env python3
"""
Run the model on a prompt, record attention matrices for selected token positions,
and produce per-layer/head histograms of the attention weight distributions.
"""
import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from utils import load_model, prepare_tokenizer


# reuse shared helpers from utils.py


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def parse_positions(pos_str):
    parts = [p.strip() for p in pos_str.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_from", default="resume", help="'resume' or a gpt2 model name like 'gpt2'")
    parser.add_argument("--ckpt", default="out/ckpt.pt", help="checkpoint path when using resume")
    parser.add_argument("--prompt", default="\n", help="prompt string or FILE:filename")
    parser.add_argument("--positions", default="-1,-2,-3", help="comma separated token positions (supports negatives)")
    parser.add_argument("--outdir", default="attention_out", help="directory to write attention stats and plots")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    device = args.device
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    model, checkpoint = load_model(args.init_from, args.ckpt, device)
    encode, decode = prepare_tokenizer(checkpoint, args.init_from)

    # prepare prompt
    prompt = args.prompt
    if prompt.startswith("FILE:"):
        with open(prompt[5:], 'r', encoding='utf-8') as f:
            prompt = f.read()
    prompt_ids = encode(prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

    outdir = ensure_outdir(args.outdir)
    positions = parse_positions(args.positions)

    # run a single forward to capture attention matrices stored on modules
    with torch.no_grad():
        logits, _ = model(x)

    stats = []
    # iterate layers
    for layer_idx, block in enumerate(model.transformer.h):
        att_module = block.attn
        att = getattr(att_module, "last_att", None).abs()
        if att is None:
            print(f"layer {layer_idx}: no attention matrix recorded, skipping")
            continue
        # att: (B, nh, T, T)
        if isinstance(att, torch.Tensor):
            att_cpu = att.cpu().numpy()
        else:
            att_cpu = np.array(att)
        # save raw attention for this layer
        np.savez_compressed(os.path.join(outdir, f"layer{layer_idx}_att.npz"), att=att_cpu)
        B, nh, T, _ = att_cpu.shape
        print(att_cpu.shape)
        
        # Create figure with subplots: rows = heads, cols = positions (timesteps)
        valid_positions = []
        for pos in positions:
            pos_idx = pos if pos >= 0 else T + pos
            if pos_idx >= 0 and pos_idx < T:
                valid_positions.append((pos, pos_idx))
            else:
                print(f"layer {layer_idx}: position {pos} (resolved {pos_idx}) out of range (T={T}), skipping")
        
        if not valid_positions:
            print(f"layer {layer_idx}: no valid positions, skipping plot")
            continue
        
        n_positions = len(valid_positions)
        n_heads = nh
        
        # Create subplot grid: rows = heads, cols = positions
        fig, axes = plt.subplots(n_heads, n_positions, 
                                 figsize=(3 * n_positions, 2.5 * n_heads),
                                 squeeze=False)
        
        for head in range(n_heads):
            for pos_col, (pos, pos_idx) in enumerate(valid_positions):
                weights = att_cpu[0, head, pos_idx, :].ravel()
                # compute simple stats
                head_stats = {
                    "layer": layer_idx,
                    "head": head,
                    "position": pos,
                    "position_resolved": int(pos_idx),
                    "sum": float(np.sum(weights)),
                    "median": float(np.median(weights)),
                    "min": float(np.min(weights)),
                    "max": float(np.max(weights)),
                }
                stats.append(head_stats)
                
                # plot histogram in subplot
                ax = axes[head, pos_col]
                ax.hist(weights, bins=10, color="C0", alpha=0.8)
                ax.set_xlim(0, 1)
                ax.set_title(f"Head {head}, Pos {pos}", fontsize=9)
                
                # Only add labels on edges
                if head == n_heads - 1:
                    ax.set_xlabel("Attn weight", fontsize=8)
                if pos_col == 0:
                    ax.set_ylabel("Count", fontsize=8)
                
                ax.tick_params(labelsize=7)
        
        # Add overall title
        fig.suptitle(f"Layer {layer_idx} - Attention Weight Distributions", 
                     fontsize=12, fontweight='bold')
        
        fn = os.path.join(outdir, f"hist_layer{layer_idx:02d}.png")
        plt.tight_layout()
        plt.savefig(fn, dpi=150)
        plt.close(fig)

    # write stats JSON
    with open(os.path.join(outdir, "attention_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote attention statistics and histograms to {outdir}")


if __name__ == "__main__":
    main()


