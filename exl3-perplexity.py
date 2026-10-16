#!/usr/bin/env python3
"""
KLD + PPL of an EXL3 pack (e.g. Qwen3.8-Flash-Next Q5) against a
llama-perplexity '-kl' logits file saved from the BF16 base model.

The .kld file ("_logits_" format) already contains:
  header: magic[8], n_ctx (u32), n_vocab (i32), n_chunk (i32)
  tokens: n_ctx * n_chunk int32 (exactly the tokens llama.cpp evaluated, BOS already substituted)
  body:   per scored position (n_ctx-1-n_ctx/2 per chunk), nv = 2*((n_vocab+1)/2)+4 uint16:
            [0:2] as float: scale, min_log_prob ; [4:4+n_vocab] quantized base log-probs
So the eval set, chunking and base distribution are all taken from the file -- no
re-tokenization, no dataset download, positions align 1:1 with llama.cpp's own -kl run.

Usage (from a checkout of https://github.com/Yamz-Labs/kyojin, in its env):
    python exl3-perplexity.py -m ./qwen-pack --kld-file ~/bench/qwen38-flash-next-bf16.kld
    # optional: --chunks 50  (default: all chunks in the file), --max-positions to skip quantiles
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse, math
import numpy as np
import torch
from exllamav3 import model_init

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(allow_abbrev=False)
model_init.add_args(parser, cache=False)
parser.add_argument("--kld-file", required=True, help="llama-perplexity --kl_save output from the BF16 base model")
parser.add_argument("--chunks", type=int, default=0, help="Evaluate only the first N chunks (0 = all in file)")
parser.add_argument("--max-positions", type=int, default=2_000_000, help="Keep this many per-token KLD values for median/quantiles")
args = parser.parse_args()

# ---- Read the base logits file -------------------------------------------------
path = os.path.expanduser(args.kld_file)
with open(path, "rb") as f:
    magic = f.read(8)
    if magic != b"_logits_":
        raise SystemExit(f"{path}: not a llama-perplexity logits file (magic = {magic!r})")
    n_ctx = np.frombuffer(f.read(4), dtype=np.uint32)[0]
    n_vocab_file = np.frombuffer(f.read(4), dtype=np.int32)[0]
    n_chunk = np.frombuffer(f.read(4), dtype=np.int32)[0]
    tokens = np.frombuffer(f.read(n_ctx * n_chunk * 4), dtype=np.int32).copy()
    body_off = f.tell()

if args.chunks:
    n_chunk = min(n_chunk, args.chunks)

nv = 2 * ((n_vocab_file + 1) // 2) + 4          # uint16 units per scored position
first = n_ctx // 2                               # llama.cpp scores [first, n_ctx-1)
n_pos_chunk = n_ctx - 1 - first

print(f"Base file: n_ctx={n_ctx}, n_vocab={n_vocab_file}, n_chunk={n_chunk} "
      f"(file had more ok), scoring {n_pos_chunk} pos/chunk from [{first}, {n_ctx-2}]")

body = np.memmap(path, dtype=np.uint16, mode="r", offset=body_off,
                 shape=(n_chunk * n_pos_chunk, nv))

# ---- Load the EXL3 pack ----------------------------------------------------------
model, config, _, tokenizer = model_init.init(
    args, override_dynamic_seq_len=n_ctx, max_output_size=n_ctx, max_output_factor=1)
bpw_layer, bpw_head, _ = model.get_storage_info()
print(f"Pack: {args.model_dir}  bitrate {bpw_layer:.2f}/{bpw_head:.2f} bpw")

# Determine the pack's logits width with one throwaway forward, then compare over
# the overlap with the file's vocab (same tokenizer => identical in practice).
_probe = model.forward(torch.tensor([[int(tokens[0])]], dtype=torch.long),
                       {"attn_mode": "flash_attn_nc"})
V = _probe.shape[-1]
del _probe
torch.cuda.empty_cache()
n_eval = min(V, n_vocab_file)
if V != n_vocab_file:
    print(f"NOTE: pack logits vocab {V} vs file n_vocab {n_vocab_file}; comparing over {n_eval} tokens")

# ---- Streaming statistics (same estimators as llama.cpp kl_divergence) -----------
s_nll = s_nll2 = s_nllb = s_nllb2 = s_nllnllb = s_kld = s_kld2 = 0.0
n_same_top = 0
count = 0
kld_keep = []

def mean_unc(s, s2, c):
    m = s / c
    v = s2 / c - m * m
    e = math.sqrt(v / (c - 1)) if v > 0 and c > 10 else 0.0
    return m, e

for i in range(n_chunk):
    base = np.asarray(body[i * n_pos_chunk : (i + 1) * n_pos_chunk])  # [P, nv] uint16
    raw = base[:, :4].copy().view(np.float32)                          # scale, min_log_prob per row
    dev = torch.device("cuda")
    scale = torch.from_numpy(raw[:, 0].astype(np.float64)).unsqueeze(1).to(dev)
    minlp = torch.from_numpy(raw[:, 1].astype(np.float64)).unsqueeze(1).to(dev)
    q = torch.from_numpy(base[:, 4 : 4 + n_eval].astype(np.float32)).to(dev)  # [P, V]
    log_base = scale * q + minlp                                        # base log-probs (fp16 grid)
    del raw, q

    chunk = tokens[i * n_ctx : (i + 1) * n_ctx].astype(np.int64)
    ids = torch.from_numpy(chunk).unsqueeze(0)
    logits = model.forward(ids, {"attn_mode": "flash_attn_nc"})         # [1, n_ctx, V]
    lg = logits[0, first : n_ctx - 1, :n_eval].float()
    del logits, ids
    torch.cuda.empty_cache()

    logp = torch.log_softmax(lg.double(), dim=-1)                       # fp64: exponentiation is exact-enough
    del lg
    tgt = torch.from_numpy(chunk[first + 1 : n_ctx].copy()).unsqueeze(1).to(dev)
    nll = -logp.gather(1, tgt).squeeze(1)
    nllb = -log_base.gather(1, tgt).squeeze(1)

    p_base = torch.exp(log_base)
    p_base[log_base <= -16.0] = 0.0                                     # llama.cpp skips p_log_base <= -16
    kld = (p_base * (log_base - logp)).sum(1)

    top_ok = (logp.argmax(1) == log_base.argmax(1)).sum().item()

    kld_np = kld.detach().cpu().numpy().astype(np.float32)
    if len(kld_keep) < args.max_positions:
        kld_keep.append(kld_np)

    kn = kld.double()
    s_nll += nll.sum().item();      s_nll2 += (nll * nll).sum().item()
    s_nllb += nllb.sum().item();    s_nllb2 += (nllb * nllb).sum().item()
    s_nllnllb += (nll * nllb).sum().item()
    s_kld += kn.sum().item();       s_kld2 += (kn * kn).sum().item()
    n_same_top += top_ok
    count += kld.numel()

    run_ppl = math.exp(s_nll / count)
    run_kld = s_kld / count
    print(f"[{i + 1}/{n_chunk}] ppl={run_ppl:.4f} kld={run_kld:.5f} top1={n_same_top / count * 100:.2f}%", flush=True)

del body

# ---- Report ----------------------------------------------------------------------
ppl, ppl_e = mean_unc(s_nll, s_nll2, count)
ppl_base, ppl_base_e = mean_unc(s_nllb, s_nllb2, count)
kld_m, kld_e = mean_unc(s_kld, s_kld2, count)
cov = (s_nllnllb / count - (s_nll / count) * (s_nllb / count)) if count > 10 else 0.0

print(f"\nEvaluated {count} positions, {n_chunk} chunks of {n_ctx}")
print(f"PPL (quant)      : {ppl:.4f} +/- {ppl_e * ppl:.5f}")
print(f"PPL (base, file) : {ppl_base:.4f} +/- {ppl_base_e * ppl_base:.5f}")
print(f"Delta log-PPL    : {math.log(ppl) - math.log(ppl_base):.5f} +/- "
      f"{math.sqrt(max(ppl_e**2 + ppl_base_e**2 - 2 * cov / (ppl * ppl_base), 0.0)):.5f}  "
      f"(cov {cov:+.4f})")
print(f"Mean KLD         : {kld_m:.6f} +/- {kld_e:.6f}")
if len(kld_keep):
    k = np.sort(np.concatenate(kld_keep))
    n = len(k)
    med = 0.5 * (k[n // 2] + k[n // 2 - 1]) if n % 2 == 0 else k[n // 2]
    print(f"Median KLD       : {med:.6f}")
    print(f"90% / 99% KLD    : {np.quantile(k, 0.9):.6f} / {np.quantile(k, 0.99):.6f}")
    print(f"Max KLD          : {k[-1]:.6f}")
print(f"Top-1 agreement  : {n_same_top / count * 100:.2f}%")
print("\nReference points (llama.cpp community rule of thumb): mean KLD < 0.01 is barely "
      "noticeable, 0.01-0.02 noticeable, > 0.05 clearly degraded.")
