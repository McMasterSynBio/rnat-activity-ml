import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.encoder.generator import RNAEncoder
from src.preprocess.embed import choose_torch_device
from src.preprocess.exposure import FEATURE_COLS as EXPOSURE_COLS

TARGET_COL = "growth_rate_z"

# An embedding dimension whose fit-split sd is at or below this is treated as
# constant and dropped rather than standardized.
MIN_EMB_SD = 1e-8

# Which feature blocks reach the head. "physics" and "embedding" are ablations 2
# and 4 of the evaluation protocol: without them, "the encoder helps" cannot be
# falsified, so the physics-only number is meant to be locked before any
# encoder result is believed.
FEATURE_MODES = ("fused", "embedding", "physics")


class ActivityDataset(Dataset):
    """
    Pairs each row of one split with its cached encoder embedding via emb_idx,
    and z-scores both the embedding and the exposure block.

    Inputs:
    * df: one split (fit/val/test), carrying emb_idx, EXPOSURE_COLS and TARGET_COL
    * embeddings: (n_dataset, dim) cached encoder output for the whole dataset
    * exposure_mean / exposure_std: fit-split statistics for the 5 exposure columns
    * emb_keep: embedding dimensions to retain, from fit_embedding_stats
    * emb_mean / emb_std: fit-split statistics for the retained embedding dimensions
    * feature_mode: which blocks to include, one of FEATURE_MODES

    Outputs: per row, x = the standardized blocks selected by feature_mode
    (embedding, exposure, or both) as float32, and y = growth_rate_z.

    All five statistics must come from the fit split alone, never from val or test.
    Standardizing the embedding block is not cosmetic: raw, its per-dimension sd
    spans ~90x within one encoder, which drives AdamW to kill most of layer 1
    (measured 212/256 units permanently dead on utrlm-te_el) and costs ~0.02 rho.
    """

    def __init__(self, df: pd.DataFrame, embeddings: np.ndarray, exposure_mean: pd.Series, exposure_std: pd.Series,
                 emb_keep: np.ndarray, emb_mean: np.ndarray, emb_std: np.ndarray, feature_mode: str = "fused"):
        assert feature_mode in FEATURE_MODES, f"feature_mode must be one of {FEATURE_MODES}, got {feature_mode!r}"
        blocks = []
        if feature_mode in ("fused", "embedding"):
            blocks.append((embeddings[df["emb_idx"].to_numpy()][:, emb_keep] - emb_mean) / emb_std)
        if feature_mode in ("fused", "physics"):
            blocks.append(((df[EXPOSURE_COLS] - exposure_mean) / exposure_std).to_numpy(dtype=np.float32))
        self.x = np.concatenate(blocks, axis=1).astype(np.float32)
        self.y = df[TARGET_COL].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx])


def fit_embedding_stats(
    embeddings: np.ndarray,
    fit_df: pd.DataFrame,
    min_sd: float = MIN_EMB_SD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Measures the fit-split mean and sd of the embedding block, and finds which
    dimensions actually vary.

    Inputs:
    * embeddings: (n_dataset, dim) cached encoder output for the whole dataset
    * fit_df: the fit split, used only for its emb_idx column
    * min_sd: dimensions with sd at or below this are treated as constant

    Outputs: (emb_keep, emb_mean, emb_std) for ActivityDataset, with emb_mean and
    emb_std already restricted to the kept dimensions.

    Constant dimensions come from the checkpoints themselves, not from this
    pipeline -- RNABERT emits 4 of 120 with sd ~1e-20 in its raw hidden states --
    and dividing by those would produce inf rather than a merely useless feature.
    """
    fit_emb = embeddings[fit_df["emb_idx"].to_numpy()]
    sd = fit_emb.std(axis=0)
    keep = np.flatnonzero(sd > min_sd)
    return keep, fit_emb[:, keep].mean(axis=0), sd[keep]


class ActivityRegressor(nn.Module):
    """
    Nonlinear regression head over a frozen RNA-LM embedding concatenated with
    the thermodynamic exposure features.

    Inputs:
    * input_dim: width of the concatenated feature vector (len(emb_keep) + 5)
    * hidden_dims: hidden layer widths; each becomes Linear -> ReLU -> Dropout
    * dropout: dropout probability applied after every hidden activation

    Outputs: forward(x: (B, input_dim)) -> (B,) predicted growth_rate_z.

    The encoder itself stays frozen; only this head learns. `dG_whole` alone
    reaches rho ~0.29 under a linear ridge, so this head exists to let the
    embedding and the remaining exposure features interact nonlinearly -- and
    rho below 0.29 means something is wrong, not that the encoder is weak.
    Inputs must arrive standardized (see ActivityDataset) or layer 1 dies.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_processed_splits(file_name: str, encoder: RNAEncoder) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    #Load the train/test csvs and their matching embeddings.
    proc_dir = Path(f"./data/processed/{file_name}")
    train_path, test_path = proc_dir / "train_set.csv", proc_dir / "test_set.csv"
    assert train_path.exists() and test_path.exists(), (
        f"Missing processed splits under {proc_dir}; run src/preprocess/load.py first."
    )
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)

    emb_path = Path(f"./data/embeddings/{file_name}/{encoder.value.split('/')[-1]}.npy")
    assert emb_path.exists(), f"Missing embeddings at {emb_path}; run src/preprocess/load.py first."
    embeddings = np.load(emb_path)
    return train_df, test_df, embeddings


def carve_validation_split(train_df: pd.DataFrame, val_frac: float = 0.1, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    #Split off a validation set from train for early stopping
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_df))
    n_val = int(len(train_df) * val_frac)
    val_idx, fit_idx = idx[:n_val], idx[n_val:]
    return train_df.iloc[fit_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)


def spearman(pred: np.ndarray, target: np.ndarray) -> float:
    return pd.Series(pred).corr(pd.Series(target), method="spearman")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss, n = 0.0, 0
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += loss_fn(pred, y).item() * len(y)
        n += len(y)
        preds.append(pred.cpu().numpy())
        targets.append(y.cpu().numpy())
    rho = spearman(np.concatenate(preds), np.concatenate(targets))
    return total_loss / n, rho


def train_model(
    file_path: str = "./data/raw/GSM2793752_Random_UTRs.csv.gz",
    encoder: RNAEncoder = RNAEncoder.UTRLM,
    hidden_dims: list[int] = [256, 64],
    dropout: float = 0.1,
    val_frac: float = 0.1,
    batch_size: int = 256,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    seed: int = 0,
    out_dir: str = "./checkpoints",
    device: torch.device = None,
    feature_mode: str = "fused",
) -> dict:
    torch.manual_seed(seed)
    device = device or choose_torch_device()
    file_name = Path(file_path).name.split(".")[0]
    encoder_name = encoder.value.split("/")[-1]

    train_df, test_df, embeddings = load_processed_splits(file_name, encoder)
    fit_df, val_df = carve_validation_split(train_df, val_frac=val_frac, seed=seed)

    # Normalize exposure features on the fit split only
    exposure_mean = fit_df[EXPOSURE_COLS].mean()
    exposure_std = fit_df[EXPOSURE_COLS].std(ddof=1).replace(0, 1.0)
    emb_keep, emb_mean, emb_std = fit_embedding_stats(embeddings, fit_df)
    if len(emb_keep) < embeddings.shape[1]:
        print(f"  dropped {embeddings.shape[1] - len(emb_keep)} constant embedding dim(s): "
              f"{np.setdiff1d(np.arange(embeddings.shape[1]), emb_keep).tolist()}")

    stats = (exposure_mean, exposure_std, emb_keep, emb_mean, emb_std, feature_mode)
    fit_ds = ActivityDataset(fit_df, embeddings, *stats)
    val_ds = ActivityDataset(val_df, embeddings, *stats)
    test_ds = ActivityDataset(test_df, embeddings, *stats)

    fit_loader = DataLoader(fit_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    input_dim = fit_ds.x.shape[1]
    model = ActivityRegressor(input_dim, hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    out_path = Path(out_dir) / file_name
    out_path.mkdir(parents=True, exist_ok=True)
    # Ablations get their own filename so they never overwrite the fused run.
    ckpt_path = out_path / f"{encoder_name}{'' if feature_mode == 'fused' else f'_{feature_mode}'}_best.pt"

    best_val_loss, best_epoch, epochs_since_best = float("inf"), -1, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n = 0.0, 0
        for x, y in fit_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
            n += len(y)

        val_loss, val_rho = evaluate(model, val_loader, device, loss_fn)
        print(
            f"[epoch {epoch:>3}/{epochs}] train_mse {running_loss / n:.4f} "
            f"| val_mse {val_loss:.4f} | val_rho {val_rho:+.4f} | elapsed {time.time() - t0:5.1f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, epochs_since_best = val_loss, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dims": hidden_dims,
                    "dropout": dropout,
                    "encoder": encoder.name,
                    "feature_mode": feature_mode,
                    "exposure_cols": EXPOSURE_COLS,
                    # Cast off pandas/numpy scalar types -- torch.load defaults to
                    # weights_only=True since 2.6, which only allowlists plain
                    # Python/tensor types, not numpy.float64 from Series.to_dict().
                    "exposure_mean": {k: float(v) for k, v in exposure_mean.items()},
                    "exposure_std": {k: float(v) for k, v in exposure_std.items()},
                    # Embedding scaling must travel with the head: scoring a
                    # standardized-trained model on raw embeddings is silent garbage.
                    "emb_keep": [int(i) for i in emb_keep],
                    "emb_mean": [float(v) for v in emb_mean],
                    "emb_std": [float(v) for v in emb_std],
                    "epoch": epoch,
                    "val_mse": float(val_loss),
                    "val_rho": float(val_rho),
                },
                ckpt_path,
            )
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                print(f"No val improvement for {patience} epochs, stopping early at epoch {epoch}.", flush=True)
                break

    print(f"Best epoch {best_epoch} (val_mse {best_val_loss:.4f}). Loaded checkpoint for final test evaluation.")
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state"])
    test_loss, test_rho = evaluate(model, test_loader, device, loss_fn)
    print(f"[test] mse {test_loss:.4f} | spearman rho {test_rho:+.4f}")

    return {
        "encoder": encoder.name,
        "feature_mode": feature_mode,
        "ckpt_path": ckpt_path,
        "best_epoch": best_epoch,
        "val_mse": best_val_loss,
        "test_mse": test_loss,
        "test_rho": test_rho,
    }


def available_cached_encoders(file_path: str) -> list[RNAEncoder]:
    #For any RNAEncoder members that already have a cached embedding .npy for this dataset.
    file_name = Path(file_path).name.split(".")[0]
    emb_dir = Path(f"./data/embeddings/{file_name}")
    return [e for e in RNAEncoder if (emb_dir / f"{e.value.split('/')[-1]}.npy").exists()]


def compare_encoders(encoders: list[RNAEncoder], **kwargs) -> pd.DataFrame:
    #Train one head per encoder on already-cached embeddings and rank them by test rho.
    rows = []
    for encoder in encoders:
        print(f"\n===== {encoder.name} ({encoder.value}) =====", flush=True)
        try:
            rows.append(train_model(encoder=encoder, **kwargs))
        except AssertionError as e:
            print(f"  skipping {encoder.name}: {e}")

    results = pd.DataFrame(rows).sort_values("test_rho", ascending=False).reset_index(drop=True)
    print("\n===== encoder comparison (sorted by test rho) =====")
    print(results.to_string(index=False))
    return results


#Usage: uv run python -m src.train --encoder ALL
#or
##Usage: uv run python -m src.train --encoder RNABERT, RNAFM, SPLICEBERT (any single or combo separated by commas)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the activity-prediction head on cached embeddings + exposure features.")
    parser.add_argument("--file_path", type=str, default="./data/raw/GSM2793752_Random_UTRs.csv.gz", help="Raw dataset path (used only to locate the matching processed/embeddings dirs).")
    parser.add_argument("--encoder", type=str, default="UTRLM", help="Comma-separated RNA encoder(s) to train on (e.g. RNAFM,UTRLM), or ALL to compare every encoder already cached for this dataset.")
    parser.add_argument("--hidden_dims", type=str, default="256,64", help="Comma-separated hidden layer sizes for the regression head.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_frac", type=float, default=0.1, help="Fraction of the train split carved out for early stopping / model selection.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience, in epochs without val improvement.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--features", type=str, default="fused", choices=FEATURE_MODES,
                        help="Feature blocks reaching the head. 'physics' and 'embedding' are the "
                             "evaluation-protocol ablations; lock 'physics' before believing any encoder result.")
    args = parser.parse_args()

    if args.encoder.strip().upper() == "ALL":
        encoders = available_cached_encoders(args.file_path)
        assert encoders, f"No cached embeddings found under ./data/embeddings/{Path(args.file_path).name.split('.')[0]}/"
    else:
        names = [n.strip() for n in args.encoder.split(",")]
        for n in names:
            assert n in RNAEncoder.__members__, f"Invalid encoder '{n}'. Choose from: {list(RNAEncoder.__members__.keys())}"
        encoders = [RNAEncoder[n] for n in names]

    shared_kwargs = dict(
        file_path=args.file_path,
        hidden_dims=[int(h) for h in args.hidden_dims.split(",")],
        dropout=args.dropout,
        val_frac=args.val_frac,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        out_dir=args.out_dir,
        feature_mode=args.features,
    )

    if len(encoders) == 1:
        train_model(encoder=encoders[0], **shared_kwargs)
    else:
        compare_encoders(encoders, **shared_kwargs)
