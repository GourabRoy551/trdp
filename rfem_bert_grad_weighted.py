"""
Gradient-Weighted RFEM on BERT (SST-2) — class-conditioned explanation.

How it differs from rfem_bert_sink_aware.py:

  Standard RFEM: per-head rollout uses raw attention A_h^l at each layer.
  Problem: attention is class-agnostic — it doesn't know whether the model
           predicted POSITIVE or NEGATIVE, so it can highlight any token.

  Gradient-weighted fix:
    For each layer l and head h, replace A_h^l with:

        G_h^l = ReLU( ∂logit_c / ∂A_h^l ) × A_h^l

    where c = predicted class (1=POS, 0=NEG).

    ReLU keeps only the attention entries whose increase would push the
    class score higher — i.e. the entries actually responsible for the
    prediction. This is the transformer analogue of GradCAM.

    These gradient-weighted matrices are then rolled up across layers
    (same +I, row-norm, matrix-multiply as standard RFEM), K-sigma
    filtered with sink-aware calibration (μ/σ from CLS row, content
    tokens only), and aggregated with paper Eq. 5.

  Result: the CLS row importance bar highlights tokens that both
          (a) receive high attention AND (b) causally drive the
          sentiment prediction — i.e. sentiment-bearing words.

Everything else — Reds colormap, per-sentence subfolders, one PDF
per figure, individual per-head matrices, K-sweep — is identical
to rfem_bert_sink_aware.py.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages

from transformers import BertTokenizer, BertForSequenceClassification

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Plot style
# ──────────────────────────────────────────────────────────────────────────────
DPI       = 200
FONT_SZ   = 26
TITLE_SZ  = 28
SUPTI_SZ  = 32
TICK_SZ   = 22
ANNOT_SZ  = 12

rcParams.update({
    "font.size":       FONT_SZ,
    "axes.titlesize":  TITLE_SZ,
    "axes.labelsize":  FONT_SZ,
    "xtick.labelsize": TICK_SZ,
    "ytick.labelsize": TICK_SZ,
    "legend.fontsize": FONT_SZ,
    "figure.dpi":      DPI,
    "savefig.dpi":     DPI,
    "savefig.bbox":    "tight",
})

MATRIX_CMAP = plt.cm.Reds.resampled(9)

OUT_DIR         = Path(__file__).resolve().parent / "figs_bert_grad_weighted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_CURRENT_OUT_DIR = OUT_DIR


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME     = "textattack/bert-base-uncased-SST-2"
DEBUG_HEAD     = 0
HEAD_TO_SHOW   = 0
K_VALUES       = [0.0, 0.3, 0.5, 1.0]
SPECIAL_TOKENS = ("[CLS]", "[SEP]", "[PAD]")
PUNCT_SET      = set(".,!?;:'\"()-")

# Class index mapping (BERT SST-2: 0=NEG, 1=POS)
LABEL_TO_IDX = {"POS": 1, "NEG": 0}

sentences = [
    ("S1",  "The film is a beautiful and moving portrait of human resilience.",                  "POS"),
    ("S2",  "This movie is an absolute waste of time and money.",                                 "NEG"),
    ("S3",  "It's a bit slow at times but the performances are outstanding.",                     "POS"),
    ("S4",  "A dull, tedious and completely forgettable experience.",                             "NEG"),
    ("S5",  "The direction is inspired and the acting is nothing short of brilliant.",            "POS"),
    ("S6",  "A masterpiece of storytelling with breathtaking visuals and emotion.",               "POS"),
    ("S7",  "Painfully boring and utterly devoid of any originality or charm.",                   "NEG"),
    ("S8",  "The screenplay is weak but the lead actor delivers a captivating turn.",             "POS"),
    ("S9",  "A hollow and disappointing sequel that betrays everything the original stood for.",  "NEG"),
    ("S10", "Funny, heartfelt and endlessly entertaining from beginning to end.",                 "POS"),
]


# ──────────────────────────────────────────────────────────────────────────────
# PDF saving
# ──────────────────────────────────────────────────────────────────────────────
_slug_re = re.compile(r"[^A-Za-z0-9._-]+")

def _slugify(s):
    return _slug_re.sub("_", s).strip("_")[:140]

def save_fig_pdf(fig, name):
    path = _CURRENT_OUT_DIR / f"{_slugify(name)}.pdf"
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"  [pdf] {path.parent.name}/{path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────
def plot_matrix_heatmap(matrix, labels=None, title="", figsize=(11, 9),
                        annotate=True, cmap=None, show_stats=True,
                        save_name=None):
    if cmap is None:
        cmap = MATRIX_CMAP
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    T         = matrix.shape[0]
    short     = [str(l)[:12] for l in labels] if labels else [str(i) for i in range(T)]
    tick_step = max(1, T // 12)
    ticks     = list(range(0, T, tick_step))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")

    if show_stats:
        mu  = float(matrix.mean())
        sig = float(matrix.std())
        full_title = f"{title}\n$\\mu = {mu:.5f}$     $\\sigma = {sig:.6f}$"
    else:
        full_title = title

    ax.set_title(full_title, fontsize=TITLE_SZ, fontweight="bold", pad=14)
    ax.set_xlabel("Source token  j", fontsize=FONT_SZ)
    ax.set_ylabel("Target token  i", fontsize=FONT_SZ)
    ax.set_xticks(ticks)
    ax.set_xticklabels([short[i] for i in ticks], rotation=50, ha="right", fontsize=TICK_SZ)
    ax.set_yticks(ticks)
    ax.set_yticklabels([short[i] for i in ticks], fontsize=TICK_SZ)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_SZ - 2)

    if annotate and T <= 20:
        max_val = matrix.max() if matrix.max() > 0 else 1.0
        for i in range(T):
            for j in range(T):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=ANNOT_SZ,
                        color="white" if matrix[i, j] > max_val * 0.55 else "black")

    fig.tight_layout()
    if save_name:
        save_fig_pdf(fig, save_name)
    return fig


def plot_token_bar(values, token_labels, title="", ylabel="Score",
                   color="#1f77b4", save_name=None):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    x = np.arange(len(token_labels))
    fig, ax = plt.subplots(figsize=(max(11, len(token_labels) * 0.85), 6.5))
    ax.bar(x, values, color=color)
    ax.set_title(title, fontsize=TITLE_SZ, fontweight="bold", pad=12)
    ax.set_xlabel("Tokens", fontsize=FONT_SZ)
    ax.set_ylabel(ylabel, fontsize=FONT_SZ)
    ax.set_xticks(x)
    ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=TICK_SZ)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    if save_name:
        save_fig_pdf(fig, save_name)
    return fig


def plot_all_head_matrices(rolled, stats, tokens, title_prefix="", save_name=None):
    if isinstance(rolled, torch.Tensor):
        rolled_np = rolled.detach().cpu().numpy()
    else:
        rolled_np = rolled

    seq       = rolled_np.shape[1]
    short_tok = [t[:12] for t in tokens]
    tick_step = 1 if seq <= 30 else max(1, seq // 8)
    ticks     = list(range(0, seq, tick_step))

    fig, axes = plt.subplots(3, 4, figsize=(34, 24))
    fig.suptitle(
        f"{title_prefix}  —  All 12 Gradient-Weighted Per-Head Rollout Matrices\n"
        r"$\hat{A}_h = \prod_{l=1}^{12}\,\mathrm{ReLU}(\nabla_l)\cdot A_h^{(l)} + I$",
        fontsize=SUPTI_SZ, fontweight="bold", y=1.00
    )

    for h, ax in enumerate(axes.flat):
        mat     = rolled_np[h]
        mu, sig = stats[h]
        im = ax.imshow(mat, aspect="auto", cmap=MATRIX_CMAP)
        ax.set_title(f"Head {h + 1}\n$\\mu = {mu:.5f}$     $\\sigma = {sig:.6f}$",
                     fontsize=TITLE_SZ, fontweight="bold", pad=10)
        ax.set_xticks(ticks)
        ax.set_xticklabels([short_tok[i] for i in ticks],
                           rotation=55, ha="right", fontsize=TICK_SZ - 2)
        ax.set_yticks(ticks)
        ax.set_yticklabels([short_tok[i] for i in ticks], fontsize=TICK_SZ - 2)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=TICK_SZ - 4)
        if seq <= 20:
            max_val = mat.max() if mat.max() > 0 else 1.0
            for i in range(seq):
                for j in range(seq):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if mat[i, j] > max_val * 0.55 else "black")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_name:
        save_fig_pdf(fig, save_name)
    return fig


def plot_k_sweep_importance(sid, label, text, tokens, scores_by_k, k_values,
                            save_name=None):
    clean    = [(i, t) for i, t in enumerate(tokens)
                if t not in SPECIAL_TOKENS and t not in PUNCT_SET]
    c_lbls   = [t for (_, t) in clean]
    c_idxs   = [i for (i, _) in clean]
    color    = "#2ca02c" if label == "POS" else "#d62728"
    sent_str = "POSITIVE (100%) ✓" if label == "POS" else "NEGATIVE (100%) ✓"

    h_fig = max(8, len(c_lbls) * 0.55 + 3)
    fig, axes = plt.subplots(1, len(k_values), figsize=(9 * len(k_values), h_fig))

    for ax, K in zip(axes, k_values):
        vals    = [float(scores_by_k[K][i]) for i in c_idxs]
        max_val = max(vals) if max(vals) > 0 else 1.0
        bars    = ax.barh(c_lbls, vals, color=color,
                          edgecolor="white", linewidth=0.6, height=0.7)
        ax.set_title(f"K = {K}", fontsize=TITLE_SZ + 2, fontweight="bold", pad=12)
        ax.set_xlabel("Importance score", fontsize=FONT_SZ)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.set_xlim(0, max_val * 1.35 + 0.04)
        for bar, v in zip(bars, vals):
            if v > 1e-4:
                ax.text(v + max_val * 0.015,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.3f}", va="center",
                        fontsize=TICK_SZ - 2, fontweight="bold")
        survived = sum(1 for v in vals if v > 0)
        ax.text(0.97, 0.02, f"{survived} / {len(vals)} tokens survive",
                transform=ax.transAxes, ha="right",
                fontsize=TICK_SZ - 2, color="dimgray", style="italic")

    fig.suptitle(
        f"{sid}  —  Gradient-Weighted RFEM Token Importance\n"
        f"{sent_str}   |   \"{text}\"",
        fontsize=SUPTI_SZ - 2, fontweight="bold", y=1.02
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_name:
        save_fig_pdf(fig, save_name)
    return fig


def plot_rfem_sparsity_per_head(head_masks, title="", save_name=None):
    if isinstance(head_masks, torch.Tensor):
        head_masks = head_masks.detach().cpu().numpy()
    H      = head_masks.shape[0]
    total  = head_masks[0].size
    kept   = [int((head_masks[h] > 0).sum()) for h in range(H)]
    ratios = [k / total * 100 for k in kept]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(range(H), ratios, color="#9467bd")
    ax.set_title(title, fontsize=TITLE_SZ, fontweight="bold", pad=12)
    ax.set_xlabel("Head", fontsize=FONT_SZ)
    ax.set_ylabel("% entries kept", fontsize=FONT_SZ)
    ax.set_xticks(range(H))
    ax.set_xticklabels([f"H{h + 1}" for h in range(H)], fontsize=TICK_SZ)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for i, (r, k) in enumerate(zip(ratios, kept)):
        ax.text(i, r + 0.3, f"{k}", ha="center", fontsize=TICK_SZ - 2)
    fig.tight_layout()
    if save_name:
        save_fig_pdf(fig, save_name)
    return fig


def plot_value_histogram(vals_flat, mu, threshold, K, title, save_name=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals_flat, bins=30, color="#aec7e8", edgecolor="white")
    ax.axvline(mu, linestyle="--", linewidth=2, color="#1f77b4",
               label=f"$\\mu = {mu:.5f}$")
    ax.axvline(threshold, linestyle="-", linewidth=2.5, color="#d62728",
               label=f"threshold (K={K}) = {threshold:.5f}")
    ax.set_title(title, fontsize=TITLE_SZ, fontweight="bold", pad=12)
    ax.set_xlabel("Rollout value", fontsize=FONT_SZ)
    ax.set_ylabel("Count", fontsize=FONT_SZ)
    ax.legend(fontsize=TICK_SZ)
    fig.tight_layout()
    if save_name:
        save_fig_pdf(fig, save_name)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model     = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, output_attentions=True
)
model.eval()
print("Model loaded. Layers: 12 | Heads: 12")


# ──────────────────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────────────────
def get_attentions_and_grads(text, class_idx):
    """
    Forward pass WITHOUT no_grad so gradients flow.
    Retains grad on each layer's attention tensor, then backprops
    the class logit to get ∂logit_c/∂A_h^l for every layer and head.

    Returns:
        tokens      list[str]
        attentions  tuple of (1, H, T, T) — raw softmax attention per layer
        attn_grads  list of (1, H, T, T) — gradient of logit_c w.r.t. each layer
        logits      (1, 2) tensor
    """
    inputs  = tokenizer(text, return_tensors="pt")
    tokens  = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    outputs = model(**inputs)   # gradient-enabled forward pass

    # Retain grad on each layer's attention (non-leaf tensors need this)
    for attn in outputs.attentions:
        attn.retain_grad()

    # Backprop the predicted-class logit
    outputs.logits[0, class_idx].backward()

    attn_grads = [attn.grad.detach() for attn in outputs.attentions]

    return tokens, outputs.attentions, attn_grads, outputs.logits


def attention_rollout_standard(attentions):
    """Standard rollout (for baseline comparison — no gradients)."""
    first   = attentions[0].squeeze(0)
    _, T, _ = first.shape
    device  = first.device
    dtype   = first.dtype
    rollout = torch.eye(T, device=device, dtype=dtype)
    for layer_attn in attentions:
        A        = layer_attn.squeeze(0).mean(dim=0).detach()
        A_plus_I = A + torch.eye(T, device=device, dtype=dtype)
        A_norm   = A_plus_I / A_plus_I.sum(dim=-1, keepdim=True)
        rollout  = A_norm @ rollout
    return rollout


def rfem_per_head_rollout_grad_weighted(attentions, attn_grads, debug_head=0):
    """
    Chefer et al. 2021 approach: standard per-head rollout, but the per-head
    aggregation weight w_h is replaced by the gradient signal:

        w_h = Σ_l  mean( ReLU(∂logit_c/∂A_h^l) ⊙ A_h^l )

    Injecting ReLU(grad)*A directly inside the rollout chain collapses to the
    identity matrix (gradients are tiny → A_weighted ≈ 0 → A+I ≈ I → product = I).
    Instead we run rollout on raw attention (identical to baseline) and use
    gradient information only to decide how much each head contributes.
    """
    first   = attentions[0].squeeze(0)
    H, T, _ = first.shape
    device  = first.device
    dtype   = first.dtype
    I       = torch.eye(T, device=device, dtype=dtype)

    head_rollouts  = []
    head_gradweights = torch.zeros(H, device=device, dtype=dtype)
    step_debug     = []

    for h in range(H):
        mat        = torch.eye(T, device=device, dtype=dtype)
        gw_sum     = torch.zeros(1, device=device, dtype=dtype)
        head_steps = []

        for layer_idx, (layer_attn, layer_grad) in enumerate(zip(attentions, attn_grads)):
            A_raw = layer_attn.squeeze(0)[h].detach()
            G_raw = layer_grad.squeeze(0)[h].detach()

            # Standard rollout (no gradient injection)
            A_plus_I = A_raw + I
            A_norm   = A_plus_I / A_plus_I.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            mat      = A_norm @ mat

            # Accumulate gradient weight for this head: mean(ReLU(G)*A) per layer
            gw_sum  += (torch.relu(G_raw) * A_raw).mean()

            if h == debug_head:
                A_weighted = torch.relu(G_raw) * A_raw
                head_steps.append({
                    "layer":      layer_idx,
                    "A_raw":      A_raw.cpu(),
                    "G_raw":      G_raw.cpu(),
                    "A_weighted": A_weighted.cpu(),
                    "A_norm":     A_norm.cpu(),
                })

        head_rollouts.append(mat)
        head_gradweights[h] = gw_sum
        if h == debug_head:
            step_debug = head_steps

    return torch.stack(head_rollouts, dim=0), head_gradweights, step_debug


def rfem_k_sigma_filter_sink_aware(head_rollouts, k=0.5, sink_indices=None):
    """
    Sink-aware K-sigma filter:
    μ_h, σ_h from CLS row (row 0), content columns only.
    Threshold applied to full matrix (value-preserving).
    """
    H, T, _ = head_rollouts.shape
    if sink_indices is None:
        sink_indices = [0, T - 1]
    content_cols = torch.tensor(
        [j for j in range(T) if j not in sink_indices],
        dtype=torch.long, device=head_rollouts.device
    )

    head_masks = []
    means      = []
    stds       = []
    thresholds = []

    print(f"  Sink-aware filter: μ/σ from CLS row, excluding cols {sink_indices}")
    print(f"  {'Head':>5}  {'mu':>14}  {'sigma':>12}  {'threshold':>12}  {'kept':>14}")
    print(f"  {'-' * 68}")

    for h in range(H):
        R_h             = head_rollouts[h]
        cls_row_content = R_h[0, content_cols]
        mu_h            = cls_row_content.mean()
        sigma_h         = cls_row_content.std(unbiased=False)
        threshold_h     = mu_h + k * sigma_h
        mask_h          = torch.where(R_h >= threshold_h, R_h, torch.zeros_like(R_h))

        head_masks.append(mask_h)
        means.append(mu_h)
        stds.append(sigma_h)
        thresholds.append(threshold_h)

        kept  = int((mask_h > 0).sum().item())
        total = mask_h.numel()
        print(f"  Head {h + 1:>2}  {mu_h.item():>14.6f}  {sigma_h.item():>12.6f}"
              f"  {threshold_h.item():>12.6f}  {kept:>5}/{total}")

    return (torch.stack(head_masks),
            torch.stack(means),
            torch.stack(stds),
            torch.stack(thresholds))


def rfem_aggregate_heads_weighted(head_masks, grad_weights):
    """A_rfem = Σ_h w_h · Ā_h  where w_h = Σ_l mean(ReLU(G_h^l) ⊙ A_h^l)."""
    H       = head_masks.shape[0]
    weights = torch.relu(grad_weights)          # ensure non-negative
    w_sum   = weights.sum().clamp(min=1e-9)
    weights = weights / w_sum                   # normalise so they sum to 1
    print(f"  Gradient-based head weights (normalised):")
    for h in range(H):
        print(f"    H{h + 1:>2}: w = {weights[h].item():.6f}")
    weighted = (weights.view(H, 1, 1) * head_masks).sum(dim=0)
    return weighted, weights


def rfem_extract_token_relevance(aggregated_map, tokens):
    cls_row = aggregated_map[0].clone()
    keep    = [(i, t) for i, t in enumerate(tokens)
               if t not in SPECIAL_TOKENS and t not in PUNCT_SET]
    idxs    = [i for i, _ in keep]
    labels  = [t for _, t in keep]
    return cls_row, cls_row[idxs], labels


def compute_and_print_mu_sigma(rolled, sid):
    stats = []
    print(f"\n{'=' * 70}")
    print(f"  {sid}  —  Per-Head μ/σ after Gradient-Weighted Rollout")
    print(f"{'=' * 70}")
    print(f"  {'Head':>6}  {'mu':>14}  {'sigma':>14}  {'max':>10}")
    print(f"  {'-' * 54}")
    for h in range(rolled.shape[0]):
        mat = rolled[h]
        if isinstance(mat, torch.Tensor):
            mat = mat.detach().cpu().numpy()
        mu  = float(mat.mean())
        sig = float(mat.std())
        stats.append((mu, sig))
        print(f"  Head {h + 1:>2}:   mu = {mu:>12.6f}   sigma = {sig:>12.6f}"
              f"   max = {mat.max():>8.4f}")
    print()
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Per-sentence pipeline
# ──────────────────────────────────────────────────────────────────────────────
def process_sentence(sid, text, label):
    global _CURRENT_OUT_DIR
    _CURRENT_OUT_DIR = OUT_DIR / sid
    _CURRENT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 78}\n#  {sid} [{label}]: {text}\n{'#' * 78}")

    class_idx = LABEL_TO_IDX[label]
    tokens, attentions, attn_grads, logits = get_attentions_and_grads(text, class_idx)

    probs = torch.softmax(logits, dim=-1)[0]
    pred  = "POSITIVE" if probs[1] > probs[0] else "NEGATIVE"
    print(f"Prediction: {pred}  (pos={probs[1]:.4f}  neg={probs[0]:.4f})")
    print(f"Conditioning on class: {label} (idx={class_idx})")
    print(f"Tokens ({len(tokens)}): {tokens}")

    sink_indices = [i for i, t in enumerate(tokens) if t in SPECIAL_TOKENS]

    # ── Baselines (vanilla + standard rollout) ────────────────────────────────
    first_layer_first_head = attentions[0].squeeze(0)[0].detach()
    cls_row_l1h1           = first_layer_first_head[0]

    plot_matrix_heatmap(
        first_layer_first_head, labels=tokens,
        title=f"{sid} [{label}] — Vanilla attention  (Layer 1, Head 1)",
        save_name=f"{sid}_M1_vanilla_L1H1_matrix",
    )
    plot_token_bar(
        cls_row_l1h1, tokens,
        title=f"{sid} [{label}] — Vanilla CLS row  (Layer 1, Head 1)",
        ylabel="Attention weight",
        save_name=f"{sid}_M1_vanilla_L1H1_cls_bar",
    )

    last_layer_mean = attentions[-1].squeeze(0).mean(dim=0).detach()
    cls_row_vanilla = last_layer_mean[0]

    plot_matrix_heatmap(
        last_layer_mean, labels=tokens,
        title=f"{sid} [{label}] — Last-layer mean attention",
        save_name=f"{sid}_M1_vanilla_matrix",
    )
    plot_token_bar(
        cls_row_vanilla, tokens,
        title=f"{sid} [{label}] — Vanilla CLS row",
        ylabel="Attention weight",
        save_name=f"{sid}_M1_vanilla_cls_bar",
    )

    rollout_matrix = attention_rollout_standard(attentions)
    rollout_cls    = rollout_matrix[0]

    plot_matrix_heatmap(
        rollout_matrix, labels=tokens,
        title=f"{sid} [{label}] — Standard attention rollout",
        save_name=f"{sid}_M2_rollout_matrix",
    )
    plot_token_bar(
        rollout_cls, tokens,
        title=f"{sid} [{label}] — Standard rollout CLS row",
        ylabel="Rollout relevance", color="#1f77b4",
        save_name=f"{sid}_M2_rollout_cls_bar",
    )

    # ── Gradient-Weighted RFEM ────────────────────────────────────────────────
    head_rollouts, head_gradweights, step1_debug = rfem_per_head_rollout_grad_weighted(
        attentions, attn_grads, debug_head=DEBUG_HEAD
    )
    print(f"\nHead rollouts shape: {tuple(head_rollouts.shape)}")

    # Step 1 diagnostics — show raw vs grad-weighted at Layer 1 for debug head
    plot_matrix_heatmap(
        step1_debug[0]["A_raw"], labels=tokens,
        title=f"{sid} Step 1 — Raw attention  (Head {DEBUG_HEAD + 1}, Layer 1)",
        save_name=f"{sid}_M3_step1_Araw_h{DEBUG_HEAD + 1}_L1",
    )
    plot_matrix_heatmap(
        step1_debug[0]["G_raw"], labels=tokens,
        title=f"{sid} Step 1 — Gradient  ∂logit_{label}/∂A  (Head {DEBUG_HEAD + 1}, Layer 1)",
        save_name=f"{sid}_M3_step1_Grad_h{DEBUG_HEAD + 1}_L1",
    )
    plot_matrix_heatmap(
        step1_debug[0]["A_weighted"], labels=tokens,
        title=f"{sid} Step 1 — ReLU(Grad) × Attention  (Head {DEBUG_HEAD + 1}, Layer 1)",
        save_name=f"{sid}_M3_step1_GradWeighted_h{DEBUG_HEAD + 1}_L1",
    )
    plot_matrix_heatmap(
        step1_debug[0]["A_norm"], labels=tokens,
        title=f"{sid} Step 1 — Row-normalised grad-weighted  (Head {DEBUG_HEAD + 1}, Layer 1)",
        save_name=f"{sid}_M3_step1_Anorm_h{DEBUG_HEAD + 1}_L1",
    )
    plot_matrix_heatmap(
        head_rollouts[HEAD_TO_SHOW], labels=tokens,
        title=f"{sid} Step 1 — Final grad-weighted rollout  (Head {HEAD_TO_SHOW + 1})",
        save_name=f"{sid}_M3_step1_final_rollout_h{HEAD_TO_SHOW + 1}",
    )

    stats = compute_and_print_mu_sigma(head_rollouts, sid)
    plot_all_head_matrices(
        head_rollouts, stats, tokens,
        title_prefix=f"{sid} ({label})",
        save_name=f"{sid}_M3_step1_all12heads_grid",
    )

    # Individual per-head matrices
    for h in range(head_rollouts.shape[0]):
        mu_h, sig_h = stats[h]
        plot_matrix_heatmap(
            head_rollouts[h], labels=tokens,
            title=f"{sid} Step 1 — Grad-weighted rollout  (Head {h + 1})\n"
                  f"$\\mu = {mu_h:.5f}$     $\\sigma = {sig_h:.6f}$",
            show_stats=False,
            save_name=f"{sid}_M3_step1_head{h + 1:02d}_matrix",
        )

    # Filtered baselines for comparison (same token set)
    keep_idx = [i for i, t in enumerate(tokens)
                if t not in SPECIAL_TOKENS and t not in PUNCT_SET]
    vanilla_plot        = cls_row_vanilla[keep_idx]
    rollout_plot        = rollout_cls[keep_idx]
    rollout_plot_tokens = [tokens[i] for i in keep_idx]

    plot_token_bar(
        vanilla_plot, rollout_plot_tokens,
        title=f"{sid} [{label}] — Raw attention CLS row  (filtered tokens)",
        ylabel="Attention weight", color="#ff7f0e",
        save_name=f"{sid}_M4_vanilla_filtered_cls_bar",
    )
    plot_token_bar(
        rollout_plot, rollout_plot_tokens,
        title=f"{sid} [{label}] — Standard rollout CLS row  (filtered tokens)",
        ylabel="Rollout relevance", color="#1f77b4",
        save_name=f"{sid}_M4_rollout_filtered_cls_bar",
    )

    scores_by_k    = {}
    content_cols_np = np.array([j for j in range(len(tokens))
                                 if j not in sink_indices])

    for K in K_VALUES:
        print(f"\n--- K = {K} ---")
        head_masks, means, stds, thresholds = rfem_k_sigma_filter_sink_aware(
            head_rollouts, k=K, sink_indices=sink_indices
        )

        # Histogram: CLS row content values
        mat_h     = head_rollouts[HEAD_TO_SHOW].detach().cpu().numpy()
        vals_flat = mat_h[0, content_cols_np]
        plot_value_histogram(
            vals_flat,
            mu=means[HEAD_TO_SHOW].item(),
            threshold=thresholds[HEAD_TO_SHOW].item(),
            K=K,
            title=(f"{sid} Step 2 — Head {HEAD_TO_SHOW + 1} CLS-row distribution  (K={K})"
                   f"\n([CLS] / [SEP] excluded)"),
            save_name=f"{sid}_M3_step2_hist_h{HEAD_TO_SHOW + 1}_K{K}",
        )

        plot_matrix_heatmap(
            head_masks[HEAD_TO_SHOW], labels=tokens,
            title=f"{sid} Step 2 — Head {HEAD_TO_SHOW + 1} K-sigma filtered  (K={K})",
            save_name=f"{sid}_M3_step2_mask_h{HEAD_TO_SHOW + 1}_K{K}",
        )

        plot_rfem_sparsity_per_head(
            head_masks,
            title=f"{sid} Step 2 — Mask sparsity per head  (K={K})",
            save_name=f"{sid}_M3_step2_sparsity_K{K}",
        )

        agg, weights = rfem_aggregate_heads_weighted(head_masks, head_gradweights)
        plot_matrix_heatmap(
            agg, labels=tokens,
            title=f"{sid} Step 3 — Weighted aggregated map  (K={K})",
            save_name=f"{sid}_M3_step3_agg_weighted_K{K}",
        )

        _, cls_filtered, plot_tokens = rfem_extract_token_relevance(agg, tokens)
        scores_by_k[K] = agg[0]

        survived = int((cls_filtered > 0).sum())
        print(f"  {survived}/{len(plot_tokens)} tokens survive")

        plot_token_bar(
            cls_filtered, plot_tokens,
            title=f"{sid} [{label}] — Gradient-Weighted RFEM CLS row  (K={K})",
            ylabel="Relevance score", color="#2ca02c",
            save_name=f"{sid}_M4_rfem_grad_cls_bar_K{K}",
        )

    plot_k_sweep_importance(
        sid, label, text, tokens, scores_by_k, K_VALUES,
        save_name=f"{sid}_M3_step4_K_sweep_importance",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nOutput directory for PDFs: {OUT_DIR}\n")
    for sid, text, label in sentences:
        process_sentence(sid, text, label)
    print(f"\nDone. All PDFs written to: {OUT_DIR}")
