# Visualization of encoder comparisons

"""
Usage:
    uv run python -m src.evaluate
    uv run python -m src.evaluate --file_path ./data/raw/GSM2793752_Random_UTRs.csv.gz --out_dir ./figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.encoder.generator import RNAEncoder
from src.preprocess.embed import choose_torch_device
from src.train import (
    ActivityDataset,
    ActivityRegressor,
    carve_validation_split,
    load_processed_splits,
    spearman,
)

_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
VAL_COLOR, TEST_COLOR = "#56B4E9", "#0072B2"  # lighter = val, darker = test
SURFACE = "#fcfcfb"
INK, MUTED = "#1a1a1a", "#8a8a8a"


def build_color_map(encoder_names) -> dict:
    #Assigns distinct colour per encoder
    return {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(sorted(encoder_names))}


@torch.no_grad()
def _predict(model, ds, device, batch_size=512):
    loader = DataLoader(ds, batch_size=batch_size)
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def _mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def evaluate_checkpoint(ckpt_path: Path, file_path: str, device, val_frac: float, seed: int) -> dict:
    """Reload one head and score it on the val + test splits."""
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder = RNAEncoder[ckpt["encoder"]]
    file_name = Path(file_path).name.split(".")[0]

    train_df, test_df, embeddings = load_processed_splits(file_name, encoder)
    _, val_df = carve_validation_split(train_df, val_frac=val_frac, seed=seed)

    exposure_mean = pd.Series(ckpt["exposure_mean"])
    exposure_std = pd.Series(ckpt["exposure_std"])
    val_ds = ActivityDataset(val_df, embeddings, exposure_mean, exposure_std)
    test_ds = ActivityDataset(test_df, embeddings, exposure_mean, exposure_std)

    model = ActivityRegressor(ckpt["input_dim"], ckpt["hidden_dims"], dropout=ckpt["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_pred, val_true = _predict(model, val_ds, device)
    test_pred, test_true = _predict(model, test_ds, device)
    n_exposure = len(ckpt.get("exposure_cols", []))

    return {
        "encoder": encoder.name,
        "hf_id": encoder.value,
        "slug": encoder.value.split("/")[-1],
        "emb_dim": ckpt["input_dim"] - n_exposure,
        "input_dim": ckpt["input_dim"],
        "hidden_dims": "x".join(map(str, ckpt["hidden_dims"])),
        "dropout": ckpt["dropout"],
        "best_epoch": ckpt.get("epoch", np.nan),
        "val_mse": _mse(val_pred, val_true),
        "val_rho": spearman(val_pred, val_true),
        "test_mse": _mse(test_pred, test_true),
        "test_rho": spearman(test_pred, test_true),
        "test_pearson": float(pd.Series(test_pred).corr(pd.Series(test_true))),
        "n_test": len(test_true),
        "_scatter": (test_true, test_pred),  # dropped before the csv is written
    }


# --------------------------------------------------------------------------- plots


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelcolor=INK, length=3)
    ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _grouped_bars(df, col_val, col_test, ylabel, title, subtitle, path, higher_is_better):
    order = df.sort_values(col_test, ascending=not higher_is_better)
    x = np.arange(len(order))
    w = 0.38

    fig, ax = plt.subplots(figsize=(1.7 * len(order) + 2.5, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    b1 = ax.bar(x - w / 2, order[col_val], w, label="validation", color=VAL_COLOR)
    b2 = ax.bar(x + w / 2, order[col_test], w, label="test", color=TEST_COLOR)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(order["encoder"], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(subtitle, fontsize=9, color=MUTED, loc="left", pad=6)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left")
    ax.margins(y=0.15)
    ax.legend(frameon=False, loc="best")
    _style(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_spearman(df, out_dir):
    _grouped_bars(
        df, "val_rho", "test_rho",
        ylabel="Spearman rho",
        title="Rank correlation with measured activity",
        subtitle="higher is better  ·  sorted by test rho",
        path=out_dir / "spearman_comparison.png",
        higher_is_better=True,
    )


def plot_mse(df, out_dir):
    _grouped_bars(
        df, "val_mse", "test_mse",
        ylabel="MSE (growth_rate_z)",
        title="Held-out mean squared error",
        subtitle="lower is better  ·  sorted by test MSE",
        path=out_dir / "mse_comparison.png",
        higher_is_better=False,
    )


def plot_generalization_gap(df, out_dir):
    order = df.assign(gap=df["test_mse"] - df["val_mse"]).sort_values("gap")
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(1.6 * len(order) + 2.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    colors = ["#009E73" if g <= 0 else "#D55E00" for g in order["gap"]]
    bars = ax.bar(x, order["gap"], 0.55, color=colors)
    ax.bar_label(bars, fmt="%+.4f", padding=2, fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order["encoder"], rotation=20, ha="right")
    ax.set_ylabel("test MSE - val MSE")
    ax.set_title("Generalization gap", fontsize=13, fontweight="bold", color=INK, loc="left")
    ax.text(0, 1.02, "positive = test worse than validation", transform=ax.transAxes, fontsize=9, color=MUTED)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "generalization_gap.png", facecolor=SURFACE)
    plt.close(fig)


def plot_convergence(df, out_dir, cmap):
    order = df.sort_values("best_epoch")
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(1.5 * len(order) + 2.5, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    bars = ax.bar(x, order["best_epoch"], 0.55, color=[cmap[e] for e in order["encoder"]])
    ax.bar_label(bars, fmt="%d", padding=2, fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(order["encoder"], rotation=20, ha="right")
    ax.set_ylabel("epoch of best val loss")
    ax.set_title("Epochs to best checkpoint", fontsize=13, fontweight="bold", color=INK, loc="left")
    ax.text(0, 1.02, "early-stopping selected epoch", transform=ax.transAxes, fontsize=9, color=MUTED)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", facecolor=SURFACE)
    plt.close(fig)


def plot_pred_vs_actual(df, out_dir, cmap):
    for _, row in df.sort_values("test_rho", ascending=False).iterrows():
        true, pred = row["_scatter"]
        lo, hi = float(np.min(true)), float(np.max(true))

        fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        ax.scatter(true, pred, s=6, alpha=0.15, color=cmap[row["encoder"]],
                   edgecolors="none", rasterized=True)
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.0, linestyle="--")
        ax.text(0.04, 0.94, f"rho {row['test_rho']:+.3f}\nMSE {row['test_mse']:.3f}",
                transform=ax.transAxes, fontsize=9, color=INK, va="top")
        ax.set_xlabel("measured (z)")
        ax.set_ylabel("predicted (z)")
        ax.set_title("predicted vs measured activity  ·  test split", fontsize=9, color=MUTED, loc="left", pad=6)
        fig.suptitle(f"{row['encoder']}  ({row['slug']})", fontsize=13, fontweight="bold",
                     color=INK, x=0.02, ha="left")
        _style(ax)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_dir / f"{row['slug']}_pred_vs_actual.png", facecolor=SURFACE)
        plt.close(fig)


def plot_summary_table(df, out_dir):
    cols = ["encoder", "emb_dim", "hidden_dims", "dropout", "best_epoch",
            "val_mse", "val_rho", "test_mse", "test_rho", "test_pearson"]
    tbl = df.sort_values("test_rho", ascending=False)[cols].copy()
    for c in ["val_mse", "val_rho", "test_mse", "test_rho", "test_pearson"]:
        tbl[c] = tbl[c].map(lambda v: f"{v:.4f}")

    fig, ax = plt.subplots(figsize=(1.15 * len(cols) + 1, 0.5 * len(tbl) + 1.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.axis("off")
    t = ax.table(cellText=tbl.values, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.4)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor(MUTED)
        if r == 0:
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_facecolor("#eeeeec")
    ax.set_title("Encoder comparison summary (sorted by test rho)", fontsize=12,
                 fontweight="bold", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_table.png", facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize the trained encoder comparison from saved checkpoints.")
    parser.add_argument("--file_path", type=str, default="./data/raw/GSM2793752_Random_UTRs.csv.gz",
                        help="Raw dataset path (used to locate the checkpoints / processed / embeddings dirs).")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints",
                        help="Root checkpoint dir; the dataset subfolder is appended automatically.")
    parser.add_argument("--out_dir", type=str, default="./figures",
                        help="Root output dir; PNGs land in <out_dir>/<dataset>/.")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Must match the value used at training time.")
    parser.add_argument("--seed", type=int, default=0, help="Must match the value used at training time.")
    args = parser.parse_args()

    device = choose_torch_device()
    file_name = Path(args.file_path).name.split(".")[0]
    ckpt_dir = Path(args.ckpt_dir) / file_name
    ckpts = sorted(ckpt_dir.glob("*_best.pt"))
    assert ckpts, f"No checkpoints found under {ckpt_dir}; run `python -m src.train --encoder ALL` first."

    out_dir = Path(args.out_dir) / file_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ckpt_path in ckpts:
        print(f"scoring {ckpt_path.name} ...", flush=True)
        rows.append(evaluate_checkpoint(ckpt_path, args.file_path, device, args.val_frac, args.seed))
    df = pd.DataFrame(rows)
    cmap = build_color_map(df["encoder"])

    plot_spearman(df, out_dir)
    plot_mse(df, out_dir)
    plot_generalization_gap(df, out_dir)
    plot_convergence(df, out_dir, cmap)
    plot_pred_vs_actual(df, out_dir, cmap)
    plot_summary_table(df, out_dir)

    csv_path = out_dir / "metrics.csv"
    df.drop(columns="_scatter").sort_values("test_rho", ascending=False).to_csv(csv_path, index=False)

    print(f"\nwrote {csv_path}")
    for p in sorted(out_dir.glob("*.png")):
        print(f"wrote {p}")
    print("\n" + df.drop(columns="_scatter").sort_values("test_rho", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
