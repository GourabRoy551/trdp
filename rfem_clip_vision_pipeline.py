"""
Modified RFEM on the CLIP VISION encoder (ViT-B/32).

This is the paper's home turf — bidirectional attention, [CLS] at position 0.
The math is identical to rfem_pipeline.py (BERT version); only the inputs
and the visualisations change.

Inputs:
    Drop image files (jpg/jpeg/png/bmp/webp) into ./input_images/
    The script picks them up automatically.

Outputs:
    ./figs_pdf_clip_vision/<image_basename>/<filename>.pdf
    One PDF per figure, same per-image-subfolder convention as the
    BERT pipeline's per-sentence subfolders.

Architecture (openai/clip-vit-base-patch32):
    12 transformer layers, 12 attention heads
    Patch size 32, image size 224 → 7x7 = 49 patches
    Tokens per forward pass = 50 (1 [CLS] + 49 patches)
    Attention is BIDIRECTIONAL (no causal mask) — no BOS-sink issue.

Visualisation pivot vs the text pipelines:
    Where the text version showed "CLS row as a bar chart over tokens",
    the vision version shows "CLS row reshaped to 7x7 and upsampled to
    224x224, overlaid on the image with a jet colormap".

    Two overlay variants per headline view:
        (a) overlay          — colored heatmap only
        (b) overlay_values   — same + the raw numeric score printed in
                               each patch's centre, so you can read which
                               patch projects which intensity.

    The 50x50 attention matrices are still saved as supplementary
    diagnostic figures (especially the Step-1 intermediates: A, A+I,
    A_norm).

User modifications (same as text pipelines, applied verbatim):
    * K-σ filter: VALUE-PRESERVING (not 0/1).
    * Aggregation: weighted, A_rfem = Σ_h w_h · Ā_h, w_h = max(Â_h),
        NO /Σw_h normalisation — matches paper Eq. 5.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import PowerNorm
from PIL import Image

from transformers import CLIPImageProcessor, CLIPVisionModel

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Plot style — matplotlib defaults, sizes bumped for zoomable PDFs
# ──────────────────────────────────────────────────────────────────────────────
DPI       = 200
FONT_SZ   = 26
TITLE_SZ  = 28
SUPTI_SZ  = 32
TICK_SZ   = 22
ANNOT_SZ  = 12

# PowerNorm gamma — applies to every imshow (matrices and heatmap overlays).
# γ < 1 expands the low end of the colormap so small values are clearly
# visible against a few dominant ones. Display-only — does NOT change the
# underlying tensors or any RFEM/rollout math.
POWER_NORM_GAMMA = 0.3

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

# Consistent quantized diverging colormap: blue = min, red = max (9 discrete levels)
MATRIX_CMAP = plt.cm.Reds.resampled(9)


# ──────────────────────────────────────────────────────────────────────────────
# Paths & config
# ──────────────────────────────────────────────────────────────────────────────
OUT_DIR    = Path(__file__).resolve().parent / "figs_pdf_clip_vision"
IMAGES_DIR = Path(__file__).resolve().parent / "input_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
_CURRENT_OUT_DIR = OUT_DIR

MODEL_NAME    = "openai/clip-vit-base-patch32"
DEBUG_HEAD    = 0
HEAD_TO_SHOW  = 0
K_VALUES      = [0.0, 0.3, 0.5, 1.0]
COLORMAP      = "jet"
IMAGE_SIZE    = 224
PATCH_SIZE    = 32
GRID          = IMAGE_SIZE // PATCH_SIZE              # 7
N_PATCHES     = GRID * GRID                           # 49
N_TOKENS      = N_PATCHES + 1                         # 50 (1 CLS + 49)


# ──────────────────────────────────────────────────────────────────────────────
# PDF saving helper — one figure per file, per-image subfolder
# ──────────────────────────────────────────────────────────────────────────────
_slug_re = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(s: str) -> str:
    s = _slug_re.sub("_", s).strip("_")
    return s[:140]


def save_fig_pdf(fig, name: str):
    """
    `name` may include forward slashes — each segment is slugified
    independently, and the resulting subdirectory tree is created under
    _CURRENT_OUT_DIR. e.g. name="01_vanilla/L1H1_overlay" →
        <_CURRENT_OUT_DIR>/01_vanilla/L1H1_overlay.pdf
    """
    parts = [_slugify(p) for p in name.split("/") if p]
    path  = _CURRENT_OUT_DIR.joinpath(*parts).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    rel = path.relative_to(OUT_DIR)
    print(f"  [pdf] {rel}")


# ──────────────────────────────────────────────────────────────────────────────
# Image / patch utilities
# ──────────────────────────────────────────────────────────────────────────────
def load_image_for_clip(path: Path, processor: CLIPImageProcessor):
    """
    Returns:
        pixel_values  (1, 3, 224, 224) tensor — model input
        image_disp    (224, 224, 3) numpy uint8 — for display (RGB, no normalisation)
    """
    pil = Image.open(path).convert("RGB")
    # processor handles resize + normalize for the model
    enc = processor(images=pil, return_tensors="pt")
    pixel_values = enc["pixel_values"]               # (1, 3, 224, 224)

    # also produce a display version: resize to 224x224 without normalisation
    pil_disp = pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    image_disp = np.array(pil_disp)
    return pixel_values, image_disp


def cls_row_to_spatial(cls_row, grid=GRID, size=IMAGE_SIZE):
    """
    cls_row: (N_TOKENS,) torch tensor or numpy array — attention CLS→{CLS, p1, ..., p49}
    Returns: (size, size) numpy float — patch scores reshaped to grid, bilinear-upsampled.
    """
    if isinstance(cls_row, torch.Tensor):
        cls_row = cls_row.detach().cpu().numpy()
    patch_scores = cls_row[1:]                       # drop CLS-self
    grid_map     = patch_scores.reshape(grid, grid).astype(np.float32)
    t            = torch.from_numpy(grid_map)[None, None]
    up           = F.interpolate(t, size=(size, size),
                                  mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


def cls_row_to_patch_grid(cls_row, grid=GRID):
    """As above but returns the 7x7 raw scores (no upsampling)."""
    if isinstance(cls_row, torch.Tensor):
        cls_row = cls_row.detach().cpu().numpy()
    return cls_row[1:].reshape(grid, grid)


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers — overlays (vision-specific) and matrix views (shared)
# ──────────────────────────────────────────────────────────────────────────────
def plot_original(image_disp, title, save_name=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_disp)
    ax.set_title(title, fontsize=TITLE_SZ, fontweight="bold", pad=14)
    ax.axis("off")
    fig.tight_layout()
    if save_name:
        save_fig_pdf(fig, save_name)
    return fig


def plot_overlay(image_disp, cls_row, title="", with_values=False,
                 figsize=(9, 9), save_name=None, use_power_norm=True):
    """
    Overlay CLS-row attention on the original image as a jet heatmap.
    If with_values=True, print the raw patch score in each patch's centre.
    μ and σ of the 49 patch scores go in the title.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Slightly darken background so low-attention areas have a cool tint
    darkened = (image_disp.astype(np.float32) * 0.6).clip(0, 255).astype(np.uint8)
    ax.imshow(darkened)

    spatial      = cls_row_to_spatial(cls_row)
    patch_scores = cls_row_to_patch_grid(cls_row)

    # Variable-alpha overlay: alpha scales with normalized attention value.
    # Low patches → nearly transparent (image visible), high patches → 65% opaque jet.
    sp_min = float(spatial.min())
    sp_max = float(spatial.max())
    if sp_max <= sp_min:
        sp_max = sp_min + 1e-9

    norm_spatial = (spatial - sp_min) / (sp_max - sp_min)
    alpha_map    = (norm_spatial ** 0.5) * 0.65

    cmap_fn  = plt.get_cmap(COLORMAP)
    rgba     = cmap_fn(norm_spatial)
    rgba[..., 3] = alpha_map
    ax.imshow(rgba)

    if with_values:
        max_v = patch_scores.max() if patch_scores.max() > 0 else 1.0
        min_v = patch_scores.min()
        for r in range(GRID):
            for c in range(GRID):
                cx        = c * PATCH_SIZE + PATCH_SIZE // 2
                cy        = r * PATCH_SIZE + PATCH_SIZE // 2
                val       = patch_scores[r, c]
                rel       = (val - min_v) / (max_v - min_v + 1e-9)
                displayed = rel ** POWER_NORM_GAMMA if use_power_norm else rel
                ax.text(cx, cy, f"{val:.3f}",
                        ha="center", va="center",
                        color="white" if displayed > 0.55 else "black",
                        fontsize=ANNOT_SZ - 1, fontweight="bold")

    mu  = float(patch_scores.mean())
    sig = float(patch_scores.std())
    full_title = f"{title}\n$\\mu = {mu:.5f}$     $\\sigma = {sig:.6f}$     (49 patches)"
    ax.set_title(full_title, fontsize=TITLE_SZ, fontweight="bold", pad=14)
    ax.axis("off")
    fig.tight_layout()
    if save_name:
        save_fig_pdf(fig, save_name)
    return fig


def plot_matrix_heatmap(matrix, title="", figsize=(11, 9),
                        annotate=False, cmap=None, save_name=None,
                        use_power_norm=True):
    if cmap is None:
        cmap = MATRIX_CMAP
    """50x50 attention matrix (or smaller). μ/σ in title."""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    T = matrix.shape[0]
    tick_step = max(1, T // 10)
    ticks     = list(range(0, T, tick_step))
    labels    = ["CLS"] + [f"p{i}" for i in range(1, T)]

    fig, ax = plt.subplots(figsize=figsize)
    if use_power_norm:
        # PowerNorm γ < 1 expands the low end of the colormap (display only).
        pn_vmin = float(matrix.min())
        pn_vmax = float(matrix.max())
        if pn_vmax <= pn_vmin:
            pn_vmax = pn_vmin + 1e-9
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest",
                        norm=PowerNorm(gamma=POWER_NORM_GAMMA,
                                        vmin=pn_vmin, vmax=pn_vmax))
    else:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                        interpolation="nearest")

    mu  = float(matrix.mean())
    sig = float(matrix.std())
    full_title = f"{title}\n$\\mu = {mu:.5f}$     $\\sigma = {sig:.6f}$"
    ax.set_title(full_title, fontsize=TITLE_SZ, fontweight="bold", pad=14)
    ax.set_xlabel("Source token  j", fontsize=FONT_SZ)
    ax.set_ylabel("Target token  i", fontsize=FONT_SZ)

    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=50, ha="right",
                       fontsize=TICK_SZ - 4)
    ax.set_yticks(ticks)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=TICK_SZ - 4)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_SZ - 2)

    if annotate and T <= 20:
        max_val = matrix.max() if matrix.max() > 0 else 1.0
        for i in range(T):
            for j in range(T):
                rel       = matrix[i, j] / max_val
                displayed = rel ** POWER_NORM_GAMMA if use_power_norm else rel
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=ANNOT_SZ,
                        color="white" if displayed > 0.55 else "black")

    fig.tight_layout()
    if save_name:
        save_fig_pdf(fig, save_name)
    return fig


def _grid_layout(H: int):
    if H <= 4:   return (1, H)
    if H <= 8:   return (2, 4)
    if H <= 12:  return (3, 4)
    cols = 4
    rows = (H + cols - 1) // cols
    return (rows, cols)


def plot_all_head_overlays(image_disp, head_cls_rows, stats,
                           title_prefix="", save_name=None):
    """3x4 grid of per-head CLS-attention overlays on the image."""
    H = head_cls_rows.shape[0]
    rows, cols = _grid_layout(H)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.5, rows * 7))
    fig.suptitle(
        f"{title_prefix}  —  All {H} Per-Head CLS Attention Overlays  (CLIP vision)\n"
        r"$\hat{A}_h = \prod_{l=1}^{12}\,\left(A_h^{(l)} + I\right)$  → CLS row reshape to 7×7 → upsample to 224×224",
        fontsize=SUPTI_SZ - 4, fontweight="bold", y=1.00,
    )

    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for h, ax in enumerate(axes_flat):
        if h >= H:
            ax.axis("off")
            continue

        cls_row = head_cls_rows[h]
        ax.imshow(image_disp)
        spatial = cls_row_to_spatial(cls_row)
        sp_vmin = float(spatial.min())
        sp_vmax = float(spatial.max())
        if sp_vmax <= sp_vmin:
            sp_vmax = sp_vmin + 1e-9
        ax.imshow(spatial, cmap=COLORMAP, alpha=0.55,
                  norm=PowerNorm(gamma=POWER_NORM_GAMMA,
                                  vmin=sp_vmin, vmax=sp_vmax))

        mu, sig = stats[h]
        ax.set_title(
            f"Head {h + 1}\n$\\mu = {mu:.5f}$     $\\sigma = {sig:.6f}$",
            fontsize=TITLE_SZ, fontweight="bold", pad=8,
        )
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_name:
        save_fig_pdf(fig, save_name)
    return fig


def plot_per_layer_overlays_grid(image_disp, per_layer_mats, title_prefix="",
                                  save_name=None, use_power_norm=True):
    """
    3x4 grid: cumulative-rollout CLS row at each layer (1..12) overlaid on
    the original image. Each cell uses its own min/max + PowerNorm so the
    spatial pattern at that layer is visible.
    """
    L = len(per_layer_mats)
    rows, cols = 3, 4

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 6))
    fig.suptitle(
        f"{title_prefix}  —  Cumulative attention rollout CLS overlay per layer",
        fontsize=SUPTI_SZ - 4, fontweight="bold", y=1.00,
    )

    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for layer_idx, ax in enumerate(axes_flat):
        if layer_idx >= L:
            ax.axis("off")
            continue
        mat = per_layer_mats[layer_idx]
        if isinstance(mat, torch.Tensor):
            mat = mat.detach().cpu().numpy()
        cls_row = mat[0]

        ax.imshow(image_disp)
        spatial = cls_row_to_spatial(cls_row)
        if use_power_norm:
            sp_vmin = float(spatial.min())
            sp_vmax = float(spatial.max())
            if sp_vmax <= sp_vmin:
                sp_vmax = sp_vmin + 1e-9
            ax.imshow(spatial, cmap=COLORMAP, alpha=0.55,
                      norm=PowerNorm(gamma=POWER_NORM_GAMMA,
                                      vmin=sp_vmin, vmax=sp_vmax))
        else:
            ax.imshow(spatial, cmap=COLORMAP, alpha=0.55)
        ax.set_title(f"Layer {layer_idx + 1}",
                     fontsize=TITLE_SZ, fontweight="bold", pad=8)
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_name:
        save_fig_pdf(fig, save_name)
    return fig


def plot_per_layer_rollout_grid(per_layer_mats, title_prefix="", save_name=None,
                                 use_power_norm=True):
    """
    3x4 grid of the per-head CUMULATIVE rolled-out matrix at each of the 12
    layers — i.e. mat after layer 1, layer 1×2, …, layer 1×2×…×12.
    Single shared magma colormap across all 12 subplots. Each subplot's
    title shows the (row, col) of the top-2 patches in that layer's CLS row.
    """
    L = len(per_layer_mats)
    rows, cols = 3, 4

    mats = [
        (m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else m)
        for m in per_layer_mats
    ]
    vmin = float(min(m.min() for m in mats))
    vmax = float(max(m.max() for m in mats))

    T         = mats[0].shape[0]
    tick_step = max(1, T // 4)
    ticks     = list(range(0, T, tick_step))
    labels    = ["CLS"] + [f"p{i}" for i in range(1, T)]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 6.0))
    fig.suptitle(
        f"{title_prefix}  —  Cumulative attention rollout matrix per layer  "
        f"({T}×{T}: 1 CLS + {GRID}×{GRID} patches)",
        fontsize=SUPTI_SZ - 4, fontweight="bold", y=0.98,
    )

    last_im = None
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    pn_vmax = vmax if vmax > vmin else vmin + 1e-9
    for layer_idx, ax in enumerate(axes_flat):
        if layer_idx >= L:
            ax.axis("off")
            continue
        mat = mats[layer_idx]
        if use_power_norm:
            im = ax.imshow(mat, cmap=MATRIX_CMAP, aspect="auto",
                            interpolation="nearest",
                            norm=PowerNorm(gamma=POWER_NORM_GAMMA,
                                            vmin=vmin, vmax=pn_vmax))
        else:
            im = ax.imshow(mat, cmap=MATRIX_CMAP, aspect="auto",
                            interpolation="nearest",
                            vmin=vmin, vmax=pn_vmax)
        last_im = im

        cls_row  = mat[0, 1:]                       # patch scores in CLS row
        top2     = np.argsort(cls_row)[::-1][:2]
        coords   = [(int(p) // GRID, int(p) % GRID) for p in top2]
        cstr     = ", ".join(f"({r},{c})" for r, c in coords)
        ax.set_title(
            f"Layer {layer_idx + 1}\nCLS↓  {cstr}",
            fontsize=TITLE_SZ - 4, fontweight="bold", pad=6,
        )
        ax.set_xticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks],
                           rotation=45, ha="right", fontsize=TICK_SZ - 6)
        ax.set_yticks(ticks)
        ax.set_yticklabels([labels[i] for i in ticks],
                           fontsize=TICK_SZ - 6)

    fig.subplots_adjust(right=0.90, top=0.93, bottom=0.05, left=0.05,
                         wspace=0.25, hspace=0.35)
    cbar_ax = fig.add_axes([0.92, 0.08, 0.014, 0.82])
    fig.colorbar(last_im, cax=cbar_ax, label="cumulative attention")

    if save_name:
        save_fig_pdf(fig, save_name)
    return fig


def plot_k_sweep_overlays(image_disp, scores_by_k, k_values, title_prefix="",
                          save_name=None):
    """1xK strip of overlays, one per K value."""
    fig, axes = plt.subplots(1, len(k_values), figsize=(len(k_values) * 8, 8))
    for ax, K in zip(axes, k_values):
        cls_row = scores_by_k[K]
        spatial = cls_row_to_spatial(cls_row)
        sp_min  = float(spatial.min())
        sp_max  = float(spatial.max())
        if sp_max <= sp_min:
            sp_max = sp_min + 1e-9
        darkened = (image_disp.astype(np.float32) * 0.6).clip(0, 255).astype(np.uint8)
        ax.imshow(darkened)
        norm_spatial = (spatial - sp_min) / (sp_max - sp_min)
        alpha_map    = (norm_spatial ** 0.5) * 0.65
        cmap_fn      = plt.get_cmap(COLORMAP)
        rgba         = cmap_fn(norm_spatial)
        rgba[..., 3] = alpha_map
        ax.imshow(rgba)
        ax.set_title(f"K = {K}", fontsize=TITLE_SZ + 2, fontweight="bold", pad=10)
        ax.axis("off")

    fig.suptitle(
        f"{title_prefix}  —  Modified RFEM, K-sweep over patch importance",
        fontsize=SUPTI_SZ - 2, fontweight="bold", y=1.04,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
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


# (histogram plot removed — user said the value-distribution histogram is not
#  informative for vision; the per-head sparsity bar + the spatial overlays
#  already cover what the filter is doing.)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME}")
processor    = CLIPImageProcessor.from_pretrained(MODEL_NAME)
vision_model = CLIPVisionModel.from_pretrained(MODEL_NAME, output_attentions=True)
vision_model.eval()
NUM_LAYERS = vision_model.config.num_hidden_layers
NUM_HEADS  = vision_model.config.num_attention_heads
print(f"Model loaded. Layers: {NUM_LAYERS} | Heads: {NUM_HEADS} | Patches: {N_PATCHES}")


# ──────────────────────────────────────────────────────────────────────────────
# Core RFEM (identical to BERT pipeline — bidirectional attention, paper-faithful)
# ──────────────────────────────────────────────────────────────────────────────
def get_attentions(pixel_values):
    """Returns tuple of NUM_LAYERS tensors, each (1, NUM_HEADS, N_TOKENS, N_TOKENS)."""
    with torch.no_grad():
        outputs = vision_model(pixel_values=pixel_values)
    return outputs.attentions


def attention_rollout(attentions):
    """Standard rollout (Abnar & Zuidema 2020): mean heads → +I → norm → multiply."""
    first   = attentions[0].squeeze(0)
    _, T, _ = first.shape
    device  = first.device
    dtype   = first.dtype
    rollout = torch.eye(T, device=device, dtype=dtype)
    debug   = []

    for layer_attn in attentions:
        A        = layer_attn.squeeze(0).mean(dim=0)
        A_plus_I = A + torch.eye(T, device=device, dtype=dtype)
        A_norm   = A_plus_I / A_plus_I.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        rollout  = A_norm @ rollout
        debug.append({"A_norm": A_norm.detach().cpu(),
                      "rollout": rollout.detach().cpu()})

    return rollout, debug


def rfem_per_head_rollout(attentions, debug_head=0):
    """Per-head rollout — heads kept separate, +I residual, row-normalised, multiplied."""
    first   = attentions[0].squeeze(0)
    H, T, _ = first.shape
    device  = first.device
    dtype   = first.dtype
    I       = torch.eye(T, device=device, dtype=dtype)

    head_rollouts = []
    step_debug    = []

    for h in range(H):
        mat = torch.eye(T, device=device, dtype=dtype)
        head_steps = []
        for layer_idx, layer_attn in enumerate(attentions):
            A_raw    = layer_attn.squeeze(0)[h]
            A_plus_I = A_raw + I
            A_norm   = A_plus_I / A_plus_I.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            mat      = A_norm @ mat

            if h == debug_head:
                head_steps.append({
                    "layer":      layer_idx,
                    "A_raw":      A_raw.detach().cpu(),
                    "A_plus_I":   A_plus_I.detach().cpu(),
                    "A_norm":     A_norm.detach().cpu(),
                    "cumulative": mat.detach().cpu(),
                })

        head_rollouts.append(mat)
        if h == debug_head:
            step_debug = head_steps

    return torch.stack(head_rollouts, dim=0), step_debug


def rfem_k_sigma_filter(head_rollouts, k=0.5):
    """
    Value-preserving K-σ filter (user modification).
        A_bar_h(i, j) = A_hat_h(i, j)  if  A_hat_h(i, j) >= mu_h + k * sigma_h
                     = 0               otherwise
    μ_h, σ_h computed over the full matrix (paper-faithful; bidirectional setting
    so no causal masking issues).
    """
    H, _, _ = head_rollouts.shape

    head_masks = []
    means      = []
    stds       = []
    thresholds = []

    print(f"  {'Head':>5}  {'mu':>12}  {'sigma':>12}  {'threshold':>12}  {'kept':>16}")
    print(f"  {'-' * 66}")

    for h in range(H):
        R_h         = head_rollouts[h]
        mu_h        = R_h.mean()
        sigma_h     = R_h.std(unbiased=False)
        threshold_h = mu_h + k * sigma_h
        mask_h      = torch.where(R_h >= threshold_h, R_h, torch.zeros_like(R_h))

        head_masks.append(mask_h)
        means.append(mu_h)
        stds.append(sigma_h)
        thresholds.append(threshold_h)

        kept  = int((mask_h > 0).sum().item())
        total = mask_h.numel()
        print(f"  Head {h + 1:>2}  {mu_h.item():>12.6f}  {sigma_h.item():>12.6f}"
              f"  {threshold_h.item():>12.6f}  {kept:>5}/{total}")

    return (torch.stack(head_masks),
            torch.stack(means),
            torch.stack(stds),
            torch.stack(thresholds))


def rfem_aggregate_heads_weighted(head_masks, head_rollouts):
    """A_rfem = Σ_h w_h · Ā_h, w_h = max(Â_h). Paper Eq. (5) verbatim."""
    H = head_masks.shape[0]
    weights = torch.stack([head_rollouts[h].max() for h in range(H)])

    print(f"  Head weights (max of rolled matrix per head):")
    for h in range(H):
        print(f"    H{h + 1:>2}: w = {weights[h].item():.6f}")
    print(f"  sum(w) = {weights.sum().item():.6f}")

    weighted = (weights.view(H, 1, 1) * head_masks).sum(dim=0)
    return weighted, weights


def compute_and_print_mu_sigma(rolled, img_id):
    """Per-head full-matrix μ/σ table."""
    stats = []
    print(f"\n{'=' * 70}")
    print(f"  {img_id}  —  Per-Head μ and σ after Full Rollout (full matrix)")
    print(f"  Rolled matrix shape per head: {rolled.shape[1]} x {rolled.shape[2]}")
    print(f"{'=' * 70}")
    print(f"  {'Head':>6}  {'mu':>14}  {'sigma':>14}  {'max':>10}  {'min':>10}")
    print(f"  {'-' * 62}")
    for h in range(rolled.shape[0]):
        mat = rolled[h]
        if isinstance(mat, torch.Tensor):
            mat = mat.detach().cpu().numpy()
        mu  = float(mat.mean())
        sig = float(mat.std())
        stats.append((mu, sig))
        print(f"  Head {h + 1:>2}:   mu = {mu:>12.6f}   sigma = {sig:>12.6f}"
              f"   max = {mat.max():>8.4f}   min = {mat.min():>8.4f}")
    print()
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Per-image pipeline
# ──────────────────────────────────────────────────────────────────────────────
def process_image(image_path: Path):
    global _CURRENT_OUT_DIR
    img_id = image_path.stem
    _CURRENT_OUT_DIR = OUT_DIR / img_id
    _CURRENT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 78}\n#  {img_id}  ({image_path.name})\n{'#' * 78}")

    pixel_values, image_disp = load_image_for_clip(image_path, processor)
    attentions               = get_attentions(pixel_values)

    plot_original(image_disp, title=f"{img_id} — Original (224×224)",
                  save_name="00_original/original")

    # ── Method 1: Vanilla attention ───────────────────────────────────────────
    # 1a) Layer 1, Head 1 — most raw view
    A_l1_h1 = attentions[0].squeeze(0)[0]                # (50, 50)
    cls_l1h1 = A_l1_h1[0]                                # (50,)

    plot_matrix_heatmap(
        A_l1_h1,
        title=f"{img_id} [vanilla] — Layer 1, Head 1 attention matrix",
        save_name="01_vanilla/L1H1_matrix",
        use_power_norm=False,
    )
    plot_overlay(
        image_disp, cls_l1h1,
        title=f"{img_id} [vanilla] — Layer 1, Head 1 CLS attention overlay",
        save_name="01_vanilla/L1H1_overlay",
        use_power_norm=False,
    )
    plot_overlay(
        image_disp, cls_l1h1, with_values=True,
        title=f"{img_id} [vanilla] — Layer 1, Head 1 CLS attention overlay (values)",
        save_name="01_vanilla/L1H1_overlay_values",
        use_power_norm=False,
    )

    # 1b) Last layer, heads averaged
    last_mean = attentions[-1].squeeze(0).mean(dim=0)    # (50, 50)
    cls_last  = last_mean[0]                             # (50,)

    plot_matrix_heatmap(
        last_mean,
        title=f"{img_id} [vanilla] — Last-layer mean attention matrix",
        save_name="01_vanilla/last_layer_matrix",
        use_power_norm=False,
    )
    plot_overlay(
        image_disp, cls_last,
        title=f"{img_id} [vanilla] — Last-layer CLS attention overlay",
        save_name="01_vanilla/last_layer_overlay",
        use_power_norm=False,
    )
    plot_overlay(
        image_disp, cls_last, with_values=True,
        title=f"{img_id} [vanilla] — Last-layer CLS attention overlay (values)",
        save_name="01_vanilla/last_layer_overlay_values",
        use_power_norm=False,
    )

    # ── Method 2: Standard rollout ────────────────────────────────────────────
    rollout_matrix, rollout_debug = attention_rollout(attentions)
    cls_rollout                   = rollout_matrix[0]

    plot_matrix_heatmap(
        rollout_matrix,
        title=f"{img_id} [rollout] — Standard attention rollout matrix",
        save_name="02_rollout/matrix",
        use_power_norm=False,
    )
    plot_overlay(
        image_disp, cls_rollout,
        title=f"{img_id} [rollout] — Standard rollout CLS overlay",
        save_name="02_rollout/overlay",
        use_power_norm=False,
    )
    plot_overlay(
        image_disp, cls_rollout, with_values=True,
        title=f"{img_id} [rollout] — Standard rollout CLS overlay (values)",
        save_name="02_rollout/overlay_values",
        use_power_norm=False,
    )

    # Per-layer cumulative rollout — matrices + image overlays (heads-averaged).
    # rollout_debug[i]['rollout'] is the cumulative rolled matrix after layer i+1.
    per_layer_rollouts = [d["rollout"] for d in rollout_debug]
    plot_per_layer_rollout_grid(
        per_layer_rollouts,
        title_prefix=f"{img_id} — heads-averaged",
        save_name="02_rollout/per_layer_rollout_matrices_grid",
        use_power_norm=False,
    )
    plot_per_layer_overlays_grid(
        image_disp, per_layer_rollouts,
        title_prefix=f"{img_id} — heads-averaged",
        save_name="02_rollout/per_layer_rollout_overlays_grid",
        use_power_norm=False,
    )

    # ── Method 3: Modified RFEM ──────────────────────────────────────────────
    head_rollouts, step1_debug = rfem_per_head_rollout(
        attentions, debug_head=DEBUG_HEAD
    )
    print(f"\nHead rollouts shape: {tuple(head_rollouts.shape)}")

    # Step 1 — intermediate diagnostics (matrices)
    plot_matrix_heatmap(
        step1_debug[0]["A_raw"],
        title=f"{img_id} Step 1 — Raw attention  (Head {DEBUG_HEAD + 1}, Layer 1)",
        save_name=f"03_rfem_step1_per_head/Araw_h{DEBUG_HEAD + 1}_L1",
    )
    plot_matrix_heatmap(
        step1_debug[0]["A_plus_I"],
        title=f"{img_id} Step 1 — After adding Identity  (Head {DEBUG_HEAD + 1}, Layer 1)",
        save_name=f"03_rfem_step1_per_head/AplusI_h{DEBUG_HEAD + 1}_L1",
    )
    plot_matrix_heatmap(
        step1_debug[0]["A_norm"],
        title=f"{img_id} Step 1 — Row-normalised  (Head {DEBUG_HEAD + 1}, Layer 1)",
        save_name=f"03_rfem_step1_per_head/Anorm_h{DEBUG_HEAD + 1}_L1",
    )

    # Final rolled matrix for HEAD_TO_SHOW — both matrix and overlay
    plot_matrix_heatmap(
        head_rollouts[HEAD_TO_SHOW],
        title=f"{img_id} Step 1 — Final per-head rollout  (Head {HEAD_TO_SHOW + 1})",
        save_name=f"03_rfem_step1_per_head/final_rollout_h{HEAD_TO_SHOW + 1}_matrix",
    )
    plot_overlay(
        image_disp, head_rollouts[HEAD_TO_SHOW][0],
        title=f"{img_id} Step 1 — Head {HEAD_TO_SHOW + 1} CLS attention overlay",
        save_name=f"03_rfem_step1_per_head/final_rollout_h{HEAD_TO_SHOW + 1}_overlay",
    )

    # Cumulative rollout matrix at each of the 12 layers, for the debug head.
    # Shows how attention "spreads" as layers accumulate, before any K-σ filter.
    per_layer_cumulative = [s["cumulative"] for s in step1_debug]
    plot_per_layer_rollout_grid(
        per_layer_cumulative,
        title_prefix=f"{img_id} — Head {DEBUG_HEAD + 1}",
        save_name=f"03_rfem_step1_per_head/per_layer_rollout_grid_h{DEBUG_HEAD + 1}",
    )

    # Per-head mu/sigma table (full matrix) + all-heads grid of CLS overlays
    stats = compute_and_print_mu_sigma(head_rollouts, img_id)
    head_cls_rows = head_rollouts[:, 0, :]               # (H, T) — CLS row per head
    plot_all_head_overlays(
        image_disp, head_cls_rows, stats,
        title_prefix=f"{img_id}",
        save_name="03_rfem_step1_per_head/all12heads_grid",
    )

    # Individual per-head rolled matrices + CLS overlays (for slides)
    for h in range(head_rollouts.shape[0]):
        mu_h, sig_h = stats[h]
        plot_matrix_heatmap(
            head_rollouts[h],
            title=f"{img_id} Step 1 — Per-head rollout  (Head {h + 1})\n"
                  f"$\\mu = {mu_h:.5f}$     $\\sigma = {sig_h:.6f}$",
            save_name=f"03_rfem_step1_per_head/head{h + 1:02d}_matrix",
        )
        plot_overlay(
            image_disp, head_rollouts[h][0],
            title=f"{img_id} Step 1 — Per-head CLS overlay  (Head {h + 1})\n"
                  f"$\\mu = {mu_h:.5f}$     $\\sigma = {sig_h:.6f}$",
            save_name=f"03_rfem_step1_per_head/head{h + 1:02d}_overlay",
        )

    scores_by_k = {}

    for K in K_VALUES:
        print(f"\n--- K = {K} ---")
        head_masks, means, stds, thresholds = rfem_k_sigma_filter(
            head_rollouts, k=K
        )

        # Filtered head matrix — matrix view + CLS overlay
        plot_matrix_heatmap(
            head_masks[HEAD_TO_SHOW],
            title=f"{img_id} Step 2 — Head {HEAD_TO_SHOW + 1} K-σ filtered (values kept)  (K={K})",
            save_name=f"04_rfem_step2_filter/K{K}/mask_h{HEAD_TO_SHOW + 1}_matrix",
        )
        plot_overlay(
            image_disp, head_masks[HEAD_TO_SHOW][0],
            title=f"{img_id} Step 2 — Head {HEAD_TO_SHOW + 1} filtered CLS overlay  (K={K})",
            save_name=f"04_rfem_step2_filter/K{K}/mask_h{HEAD_TO_SHOW + 1}_overlay",
        )

        plot_rfem_sparsity_per_head(
            head_masks,
            title=f"{img_id} Step 2 — Mask sparsity per head  (K={K})",
            save_name=f"04_rfem_step2_filter/K{K}/sparsity",
        )

        # Step 3 — weighted aggregation (Eq. 5)
        agg, weights = rfem_aggregate_heads_weighted(head_masks, head_rollouts)

        plot_matrix_heatmap(
            agg,
            title=f"{img_id} Step 3 — Weighted aggregated map  (K={K})",
            save_name=f"05_rfem_step3_aggregate/K{K}/matrix",
        )
        plot_overlay(
            image_disp, agg[0],
            title=f"{img_id} Step 3 — RFEM aggregated CLS overlay  (K={K})",
            save_name=f"05_rfem_step3_aggregate/K{K}/overlay",
        )
        plot_overlay(
            image_disp, agg[0], with_values=True,
            title=f"{img_id} Step 3 — RFEM aggregated CLS overlay (values)  (K={K})",
            save_name=f"05_rfem_step3_aggregate/K{K}/overlay_values",
        )

        scores_by_k[K] = agg[0]

    # K-sweep summary — strip of overlays
    plot_k_sweep_overlays(
        image_disp, scores_by_k, K_VALUES,
        title_prefix=f"{img_id}",
        save_name="06_rfem_step4_K_sweep/K_sweep_overlays",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Image discovery
# ──────────────────────────────────────────────────────────────────────────────
def discover_images():
    exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG",
            "*.png", "*.PNG", "*.bmp", "*.webp")
    paths = []
    for ext in exts:
        paths.extend(sorted(IMAGES_DIR.rglob(ext)))
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    image_paths = discover_images()

    if not image_paths:
        print(f"\nNo images found in {IMAGES_DIR}/")
        print(f"Drop image files (.jpg/.jpeg/.png/.bmp/.webp) there and re-run.")
    else:
        print(f"\nFound {len(image_paths)} image(s) in {IMAGES_DIR}/")
        print(f"Output directory: {OUT_DIR}\n")
        for p in image_paths:
            process_image(p)
        print(f"\nDone. All PDFs written to: {OUT_DIR}")
