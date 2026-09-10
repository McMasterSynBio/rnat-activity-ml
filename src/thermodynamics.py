import time
from pathlib import Path

import numpy as np
import pandas as pd
import RNA

from src.preprocess.exposure import ASSAY_TEMP_C, HIS3_CONTEXT, _unpaired_and_dG

# Universal gas constant in kcal/(mol*K), easy conversion for ViennaRNA
_R_KCAL = 0.0019872041

TEMP_GRID = np.arange(20.0, 45.1, 5.0)

_LOW_REF, _HIGH_REF = ASSAY_TEMP_C, 37.0

FEATURE_COLS = [
    "rnat_kozak_acc_lowT",     # Kozak-window unpaired prob at coldest T
    "rnat_kozak_acc_highT",    # at the hottest T 
    "rnat_kozak_acc_amp",      # highT - lowT; signed
    "rnat_kozak_Tm",           # interpolated melt midpoint of Kozak window (deg C); NaN if it barely moves
    "rnat_kozak_max_slope",    # steepest d(acc)/dT over the grid (per deg C); switch-sharpness proxy
    "rnat_dG_whole_assay",     # ensemble dG of UTR+context at the assay temperature (kcal/mol)
    "rnat_dG_whole_slope",     # linear slope d(dG_whole)/dT over the grid; steeper = more structure melting
    "rnat_ddG_whole_30_37",    # dG_whole(37) - dG_whole(30); single-number thermo-responsiveness
    "rnat_occluded_frac_30",   # 1 - P(whole Kozak window simultaneously unpaired) at 30 C, constrained ensemble
    "rnat_occluded_frac_37",   # at 37 C
    "rnat_occlusion_release",  # occluded_frac_30 - occluded_frac_37; how much occlusion the 30->37 shift relieves
]


def _melt_midpoint(temps: np.ndarray, acc: np.ndarray) -> tuple[float, float, float]:
    #accessibility-vs-temperature curve. NaN if switch is smaller than 0.02 (arbitrary, may need to confirm/justify)
    _FLAT_EPS = 0.02
    acc = np.asarray(acc, dtype=float)
    temps = np.asarray(temps, dtype=float)
    amp = float(acc[-1] - acc[0])
    lo, hi = float(acc.min()), float(acc.max())

    slopes = np.diff(acc) / np.diff(temps)
    max_slope = float(slopes.max() if amp >= 0 else slopes.min())

    if hi - lo < _FLAT_EPS:
        return np.nan, max_slope, amp

    target = 0.5 * (lo + hi)
    direction = 1.0 if amp >= 0 else -1.0
    for k in range(len(temps) - 1):
        a0, a1 = acc[k], acc[k + 1]
        if a0 != a1 and (a0 - target) * (a1 - target) <= 0 and (a1 - a0) * direction > 0:
            frac = (target - a0) / (a1 - a0)
            return float(temps[k] + frac * (temps[k + 1] - temps[k])), max_slope, amp
    return np.nan, max_slope, amp


def _window_unpaired_prob(seq: str, i0: int, i1: int, md: RNA.md, dG_unconstrained: float) -> float:
    """P(every base in the 0-based half-open window [i0, i1) is unpaired at once).

    From the constrained vs unconstrained ensemble free energy:
    P = exp((dG_unconstrained - dG_constrained) / RT). Constraining bases to be
    unpaired can only raise the free energy, so dG_unconstrained <= dG_constrained
    and P lands in (0, 1]. 
    """
    fcc = RNA.fold_compound(seq, md)
    for pos in range(i0 + 1, i1 + 1): 
        fcc.hc_add_up(pos)
    _, dGc = fcc.pf()

    RT = (md.temperature + 273.15) * _R_KCAL
    return float(np.exp((dG_unconstrained - dGc) / RT))


def thermo_features(
    utr: str,
    context: str = HIS3_CONTEXT,
    temps: np.ndarray = TEMP_GRID,
) -> dict[str, float]:
    seq = utr + context
    start = len(utr)  # index of the A in AUG
    w0, w1 = max(0, start - 6), start + 4  # Kozak window, 0-based half-open; -6..+4 around AUG

    grid = np.array(sorted(float(t) for t in temps))
    sweep_temps = sorted(set(grid.tolist()) | {_LOW_REF, _HIGH_REF})

    acc_at: dict[float, float] = {}
    dG_at: dict[float, float] = {}
    for T in sweep_temps:
        md = RNA.md()
        md.temperature = T
        unpaired, dG = _unpaired_and_dG(seq, md)
        acc_at[T] = float(unpaired[w0:w1].mean())
        dG_at[T] = float(dG)

    acc_curve = np.array([acc_at[t] for t in grid])
    dG_curve = np.array([dG_at[t] for t in grid])
    Tm, max_slope, amp = _melt_midpoint(grid, acc_curve)
    dG_slope = float(np.polyfit(grid, dG_curve, 1)[0])

    md_lo = RNA.md()
    md_lo.temperature = _LOW_REF
    md_hi = RNA.md()
    md_hi.temperature = _HIGH_REF
    occ_lo = 1.0 - _window_unpaired_prob(seq, w0, w1, md_lo, dG_at[_LOW_REF])
    occ_hi = 1.0 - _window_unpaired_prob(seq, w0, w1, md_hi, dG_at[_HIGH_REF])

    return {
        "rnat_kozak_acc_lowT": float(acc_curve[0]),
        "rnat_kozak_acc_highT": float(acc_curve[-1]),
        "rnat_kozak_acc_amp": float(amp),
        "rnat_kozak_Tm": float(Tm),
        "rnat_kozak_max_slope": float(max_slope),
        "rnat_dG_whole_assay": dG_at[_LOW_REF],
        "rnat_dG_whole_slope": dG_slope,
        "rnat_ddG_whole_30_37": dG_at[_HIGH_REF] - dG_at[_LOW_REF],
        "rnat_occluded_frac_30": occ_lo,
        "rnat_occluded_frac_37": occ_hi,
        "rnat_occlusion_release": occ_lo - occ_hi,
    }


def compute_and_save_thermodynamics(
    sequences: list[str],
    out_dir: str,
    context: str = HIS3_CONTEXT,
    temps: np.ndarray = TEMP_GRID,
    report_every: int = 20000,
) -> pd.DataFrame:
    """Fold every sequence across the grid and return row-aligned RNAt features.

    Row i corresponds to sequences[i], so the caller can concat it on as columns. 
    """
    assert len(context) >= 3 and context[:3] == "AUG", "context must start at the AUG"

    grid = np.array(sorted(float(t) for t in temps))
    tag = f"{grid[0]:g}-{grid[-1]:g}C_{len(grid)}pt"
    out_path = Path(out_dir) / f"thermo_sweep_{tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.is_file():
        cached = pd.read_csv(out_path)
        if len(cached) == len(sequences):
            print(f"  [thermo] reusing cache: {out_path}")
            return cached
        print(f"  [thermo] cache is stale ({len(cached):,} != {len(sequences):,}), recomputing")

    rows = []
    t0 = time.time()
    for i, utr in enumerate(sequences, start=1):
        rows.append(thermo_features(utr, context=context, temps=grid))
        if i % report_every == 0 or i == len(sequences):
            el = time.time() - t0
            rate = i / el
            print(
                f"  [thermo {tag}] {i:>7,}/{len(sequences):,} ({i / len(sequences) * 100:5.1f}%) "
                f"| {rate:6.1f} seq/s | elapsed {el:6.1f}s | eta {(len(sequences) - i) / rate:7.1f}s",
                flush=True,
            )

    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    df.to_csv(out_path, index=False)
    print(f"Saved thermodynamic sweep features to '{out_path}'.")
    return df


# --------------------------------------------------------------------------- single-sequence prediction with uncertainty


def _sample_window_unpaired(seq: str, i0: int, i1: int, md: RNA.md, n: int) -> np.ndarray:
    """Draw `n` structures from the Boltzmann ensemble; return an (n, i1-i0) bool
    array, True where that base is unpaired in that sampled structure.

    This is the Monte-Carlo counterpart of the exact partition-function features:
    the mean over rows recovers per-base unpaired probability, and the fraction of
    rows with the whole window open recovers `_window_unpaired_prob`.
    """
    md.uniq_ML = 1 
    fc = RNA.fold_compound(seq, md)
    fc.pf()  # exact partition function from ViennaRNA
    structs = fc.pbacktrack(int(n))
    if isinstance(structs, str):
        structs = [structs]
    if len(structs) == 0:
        raise RuntimeError("RNA.pbacktrack returned no structures (ViennaRNA sampling unavailable in this build)")
    chars = np.array([list(s) for s in structs])
    return chars[:, i0:i1] == "."


def predict_with_uncertainty(
    utr: str,
    context: str = HIS3_CONTEXT,
    temps: np.ndarray = TEMP_GRID,
    n_samples: int = 5000,
    n_boot: int = 1000,
    seed: int = 0,
    return_curves: bool = False,
):
    """Per-feature point estimate + bootstrap uncertainty for a single UTR.

    The point estimate of every feature is exactly what `thermo_features` /
    `compute_and_save_thermodynamics` would write for this sequence. The
    uncertainty is a nonparametric bootstrap over `n_samples` structures drawn
    from the Boltzmann ensemble at each grid temperature:

    - accessibility / occlusion features carry a genuine sampling spread, reported
      as a standard deviation and a 2.5-97.5 percentile interval;
    - `rnat_kozak_Tm` also reports the fraction of bootstrap replicates whose melt
      curve was too flat to place a midpoint;
    - the free-energy features (`rnat_dG_whole_*`, `rnat_ddG_whole_30_37`) come
      straight from the partition function and are exact -- they are returned with
      `{"exact": True}` and no interval.
    """
    grid = np.array(sorted(float(t) for t in temps))
    point = thermo_features(utr, context=context, temps=grid)

    seq = utr + context
    start = len(utr)
    w0, w1 = max(0, start - 6), start + 4 

    try: 
        RNA.init_rand(int(seed))
    except Exception:
        pass
    rng = np.random.default_rng(seed)

    ref_temps = sorted(set(grid.tolist()) | {_LOW_REF, _HIGH_REF})
    win = {}  # T -> (n_samples, W) bool, True = unpaired
    for T in ref_temps:
        md = RNA.md()
        md.temperature = T
        win[T] = _sample_window_unpaired(seq, w0, w1, md, n_samples)

    # Kozak melt curve
    grid_frac = [win[t].mean(axis=1) for t in grid] 
    boot = {k: [] for k in ("lowT", "highT", "amp", "Tm", "slope")}
    for _ in range(n_boot):
        curve = np.array([f[rng.integers(0, len(f), len(f))].mean() for f in grid_frac])
        Tm_b, slope_b, amp_b = _melt_midpoint(grid, curve)
        boot["lowT"].append(curve[0])
        boot["highT"].append(curve[-1])
        boot["amp"].append(amp_b)
        boot["Tm"].append(Tm_b)
        boot["slope"].append(slope_b)
    boot = {k: np.asarray(v, dtype=float) for k, v in boot.items()}

    # Occlusion occupancy: bootstrap the "whole window open" indicator.
    def frac_boot(T: float) -> np.ndarray:
        allup = win[T].all(axis=1).astype(float)
        return np.array([1.0 - allup[rng.integers(0, len(allup), len(allup))].mean() for _ in range(n_boot)])

    occ30_b, occ37_b = frac_boot(_LOW_REF), frac_boot(_HIGH_REF)
    release_b = occ30_b - occ37_b

    # 95% bootstrap band for the accessibility curve, one interval per grid point.
    band = np.array([
        np.percentile([gf[rng.integers(0, len(gf), len(gf))].mean() for _ in range(n_boot)], [2.5, 97.5])
        for gf in grid_frac
    ])

    def summ(value: float, arr: np.ndarray) -> dict:
        finite = arr[np.isfinite(arr)]
        if finite.size < 2:
            return {"value": value, "sd": float("nan"), "ci": (float("nan"), float("nan")), "nan_frac": 1.0}
        return {
            "value": value,
            "sd": float(finite.std(ddof=1)),
            "ci": (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))),
            "nan_frac": float(np.mean(~np.isfinite(arr))),
        }

    exact = lambda k: {"value": point[k], "exact": True}
    report = {
        "rnat_kozak_acc_lowT": summ(point["rnat_kozak_acc_lowT"], boot["lowT"]),
        "rnat_kozak_acc_highT": summ(point["rnat_kozak_acc_highT"], boot["highT"]),
        "rnat_kozak_acc_amp": summ(point["rnat_kozak_acc_amp"], boot["amp"]),
        "rnat_kozak_Tm": summ(point["rnat_kozak_Tm"], boot["Tm"]),
        "rnat_kozak_max_slope": summ(point["rnat_kozak_max_slope"], boot["slope"]),
        "rnat_dG_whole_assay": exact("rnat_dG_whole_assay"),
        "rnat_dG_whole_slope": exact("rnat_dG_whole_slope"),
        "rnat_ddG_whole_30_37": exact("rnat_ddG_whole_30_37"),
        "rnat_occluded_frac_30": summ(point["rnat_occluded_frac_30"], occ30_b),
        "rnat_occluded_frac_37": summ(point["rnat_occluded_frac_37"], occ37_b),
        "rnat_occlusion_release": summ(point["rnat_occlusion_release"], release_b),
    }
    if not return_curves:
        return report

    # Exact per-base unpaired probability and dG at each grid temperature, kept in
    # full so a plot can show *where* along the molecule the structure melts.
    unpaired_matrix = np.empty((len(grid), len(seq)))
    dG_curve = np.empty(len(grid))
    for i, T in enumerate(grid):
        md = RNA.md()
        md.temperature = T
        up, g = _unpaired_and_dG(seq, md)
        unpaired_matrix[i] = up
        dG_curve[i] = g

    curves = {
        "grid": grid,
        "seq": seq,
        "utr_len": start,                       # = AUG position = UTR/context boundary
        "kozak": (w0, w1),
        "cap": (0, min(15, start)),
        "acc_curve": np.array([gf.mean() for gf in grid_frac]),  # sampled, matches the band
        "acc_band": band,                      # (len(grid), 2): 2.5 / 97.5 percentile
        "dG_curve": dG_curve,
        "unpaired_matrix": unpaired_matrix,
    }
    return report, curves

_FIG_SURFACE, _FIG_INK, _FIG_MUTED = "#fcfcfb", "#1a1a1a", "#8a8a8a"
_FIG_ACC, _FIG_DG, _FIG_TM = "#0072B2", "#D55E00", "#009E73"


def plot_sequence_prediction(
    utr: str,
    context: str = HIS3_CONTEXT,
    temps: np.ndarray = TEMP_GRID,
    n_samples: int = 5000,
    n_boot: int = 1000,
    seed: int = 0,
    path: str = "rnat_report.png",
) -> tuple[dict, str]:
    # Left: Kozak-window opening vs temperature, with the 95% bootstrap band and the
    # melting midpoint (+/- its interval). 
    # Middle: the exact ensemble free energy vs temperature with its linear slope. 
    # Right: per-base unpaired probability along the molecule at the coldest vs hottest
    # grid temperature
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report, cur = predict_with_uncertainty(
        utr, context=context, temps=temps, n_samples=n_samples, n_boot=n_boot, seed=seed, return_curves=True
    )
    grid = cur["grid"]
    acc, band = cur["acc_curve"], cur["acc_band"]
    dG = cur["dG_curve"]
    w0, w1 = cur["kozak"]
    c0, c1 = cur["cap"]
    aug = cur["utr_len"]

    def _style(ax):
        ax.set_facecolor(_FIG_SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(_FIG_MUTED)
        ax.tick_params(colors=_FIG_INK, labelcolor=_FIG_INK, length=3)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.7), dpi=150)
    fig.patch.set_facecolor(_FIG_SURFACE)

    ax = axes[0]
    ax.fill_between(grid, band[:, 0], band[:, 1], color=_FIG_ACC, alpha=0.20, linewidth=0, label="95% bootstrap band")
    ax.plot(grid, acc, marker="o", color=_FIG_ACC, linewidth=2, label="P(Kozak window unpaired)")
    tm = report["rnat_kozak_Tm"]["value"]
    amp = report["rnat_kozak_acc_amp"]["value"]
    if np.isfinite(tm):
        lo, hi = report["rnat_kozak_Tm"]["ci"]
        ax.axvspan(lo, hi, color=_FIG_TM, alpha=0.15, linewidth=0)
        ax.axvline(tm, color=_FIG_TM, linestyle="--", linewidth=1.5)
        ax.text(tm, 0.04, f" Tm {tm:.1f} C", color=_FIG_TM, va="bottom", ha="left", fontsize=9,
                transform=ax.get_xaxis_transform())
    ax.axhline(0.5 * (acc.min() + acc.max()), color=_FIG_MUTED, linestyle=":", linewidth=1)
    ax.set_xlabel("temperature (C)")
    ax.set_ylabel("P(unpaired)")
    direction = "heat opens the window" if amp >= 0 else "heat closes the window"
    ax.set_title(f"Kozak window opening   ({direction}, amp {amp:+.3f})", fontsize=10, color=_FIG_INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")
    _style(ax)

    ax = axes[1]
    ax.plot(grid, dG, marker="o", color=_FIG_DG, linewidth=2)
    slope = report["rnat_dG_whole_slope"]["value"]
    ddg = report["rnat_ddG_whole_30_37"]["value"]
    fit = np.polyval(np.polyfit(grid, dG, 1), grid)
    ax.plot(grid, fit, linestyle="--", color=_FIG_MUTED, linewidth=1)
    ax.set_xlabel("temperature (C)")
    ax.set_ylabel("dG_whole (kcal/mol)")
    ax.set_title("Ensemble free energy (exact)", fontsize=10, color=_FIG_INK, loc="left")
    ax.text(0.03, 0.95, f"dG/dT {slope:+.3f} kcal/mol/C\nDDG(30->37) {ddg:+.2f} kcal/mol",
            transform=ax.transAxes, fontsize=9, color=_FIG_INK, va="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=_FIG_MUTED, alpha=0.9))
    _style(ax)

    ax = axes[2]
    mat = cur["unpaired_matrix"]
    k = 5 
    delta = np.convolve(mat[-1] - mat[0], np.ones(k) / k, mode="same")
    pos = np.arange(len(delta))
    ax.axvspan(c0, c1, color=_FIG_MUTED, alpha=0.13, linewidth=0)
    ax.axvspan(w0, w1, color=_FIG_TM, alpha=0.16, linewidth=0)
    ax.axhline(0, color=_FIG_MUTED, linewidth=0.8)
    ax.fill_between(pos, 0, delta, where=delta >= 0, color=_FIG_DG, alpha=0.5, linewidth=0)
    ax.fill_between(pos, 0, delta, where=delta < 0, color=_FIG_ACC, alpha=0.5, linewidth=0)
    ax.plot(pos, delta, color=_FIG_INK, linewidth=0.8)
    ax.axvline(aug, color=_FIG_INK, linewidth=1)
    ax.text(aug + 0.6, 0.97, "AUG", color=_FIG_INK, rotation=90, va="top", ha="left", fontsize=8,
            transform=ax.get_xaxis_transform())
    ax.text(np.mean([c0, c1]), 1.02, "cap", color=_FIG_MUTED, va="bottom", ha="center", fontsize=8,
            transform=ax.get_xaxis_transform())
    ax.text(np.mean([w0, w1]), 1.02, "Kozak", color=_FIG_TM, va="bottom", ha="center", fontsize=8,
            transform=ax.get_xaxis_transform())
    ax.set_xlim(0, len(delta))
    ax.set_xlabel("position (nt)   |   UTR + TR context")
    ax.set_ylabel(f"d P(unpaired), {grid[-1]:g} - {grid[0]:g} C   (orange = opens on heat)")
    ax.set_title("Where the molecule opens on heating", fontsize=10, color=_FIG_INK, loc="left", pad=16)
    _style(ax)

    head = utr if len(utr) <= 60 else utr[:57] + "..."
    fig.suptitle(f"RNAt thermodynamic profile   {head}   ({len(utr)} nt)",
                 fontsize=12, fontweight="bold", color=_FIG_INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=_FIG_SURFACE)
    plt.close(fig)
    return report, path


def _print_report(unc: dict[str, dict], grid: np.ndarray, n_samples: int, n_boot: int) -> None:
    groups = [
        ("Kozak melt curve", FEATURE_COLS[:5]),
        ("whole-molecule stability", FEATURE_COLS[5:8]),
        ("occlusion occupancy", FEATURE_COLS[8:]),
    ]
    print(f"\ngrid: {', '.join(f'{t:g}' for t in grid)} C   "
          f"({n_samples:,} ensemble draws/T, {n_boot:,} bootstrap reps)\n")
    print(f"{'feature':<24}{'estimate':>11}{'s.d.':>10}{'95% CI':>20}   note")
    for title, cols in groups:
        print(f"\n  {title}")
        for c in cols:
            u = unc[c]
            if u.get("exact"):
                print(f"  {c:<24}{u['value']:>11.3f}{'--':>10}{'--':>20}   exact (partition function)")
                continue
            v, sd, (lo, hi) = u["value"], u["sd"], u["ci"]
            if c == "rnat_kozak_Tm" and not np.isfinite(v):
                print(f"  {c:<24}{'no switch':>11}{'--':>10}{'--':>20}   melt curve flat (<0.02 span)")
                continue
            ci = f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else "--"
            note = ""
            if c == "rnat_kozak_Tm" and u["nan_frac"] > 0:
                note = f"{u['nan_frac'] * 100:.0f}% of boot reps flat"
            print(f"  {c:<24}{v:>11.3f}{sd:>10.3f}{ci:>20}   {note}")
    print(
        "\n  estimate = exact value from the partition function (same as the batch pipeline).\n"
        "  s.d./CI  = Monte-Carlo spread from resampling the Boltzmann ensemble; it narrows\n"
        "             as --n_samples grows and does NOT include Turner-parameter or model\n"
        "             uncertainty, so treat it as a precision floor, not a full error bar."
    )


if __name__ == "__main__":
    """Two modes.

    `--sequence <UTR>` prints every feature for one UTR with a bootstrap
    uncertainty (see predict_with_uncertainty): the estimate matches what the
    batch pipeline would write, and the interval comes from resampling structures
    drawn from the Boltzmann ensemble.

    `--file_path <csv>` runs the dataset check: folds a small sample across the
    grid and prints per-feature distributions plus Spearman rho against the
    (single-temperature, 30 C) growth_rate. Only the "level" proxies --
    rnat_dG_whole_assay and rnat_occluded_frac_30 -- are expected to track
    growth_rate, roughly echoing exposure.py's dG_whole (|rho| ~ 0.30; sign
    follows the feature's orientation).
    """
    import argparse

    parser = argparse.ArgumentParser(description="Temperature-sweep RNAt features: single sequence (with uncertainty) or dataset check.")
    parser.add_argument("--sequence", type=str, default=None, help="A single UTR (T or U alphabet). Prints per-feature estimates with bootstrap uncertainty.")
    parser.add_argument("--file_path", type=str, default=None, help="CSV to sample from for the dataset check (ignored when --sequence is given).")
    parser.add_argument("--n", type=int, default=500, help="Dataset-check rows to sample (0 = all). Slower than exposure.py; keep small.")
    parser.add_argument("--n_samples", type=int, default=5000, help="--sequence: Boltzmann-ensemble draws per temperature.")
    parser.add_argument("--n_boot", type=int, default=1000, help="--sequence: bootstrap replicates.")
    parser.add_argument("--seed", type=int, default=0, help="--sequence: RNG seed.")
    parser.add_argument("--plot", nargs="?", const="rnat_report.png", default=None,
                        help="--sequence: also save a three-panel PNG (melt curve, free energy, melt map). Optional path.")
    parser.add_argument("--tmin", type=float, default=float(TEMP_GRID[0]), help="Grid start (deg C).")
    parser.add_argument("--tmax", type=float, default=float(TEMP_GRID[-1]), help="Grid end, inclusive (deg C).")
    parser.add_argument("--tstep", type=float, default=5.0, help="Grid step (deg C).")
    args = parser.parse_args()

    grid = np.arange(args.tmin, args.tmax + 1e-6, args.tstep)

    if args.sequence:
        utr = args.sequence.strip().upper().replace("T", "U")
        assert set(utr) <= set("ACGU"), f"sequence has non-ACGU/T characters: {sorted(set(utr) - set('ACGU'))}"
        print(f"sequence ({len(utr)} nt): {utr}")
        if args.plot:
            report, saved = plot_sequence_prediction(
                utr, temps=grid, n_samples=args.n_samples, n_boot=args.n_boot, seed=args.seed, path=args.plot
            )
        else:
            report = predict_with_uncertainty(
                utr, temps=grid, n_samples=args.n_samples, n_boot=args.n_boot, seed=args.seed
            )
        _print_report(report, grid, args.n_samples, args.n_boot)
        if args.plot:
            print(f"\nsaved figure to {saved}")
        raise SystemExit(0)

    assert args.file_path, "pass --sequence <UTR> or --file_path <csv>"
    df = pd.read_csv(args.file_path)
    df = df[df["t0"] >= 20].reset_index(drop=True)
    if args.n:
        df = df.sample(args.n, random_state=0).reset_index(drop=True)
    utrs = df["UTR"].str.replace("T", "U").tolist()

    feats = compute_and_save_thermodynamics(utrs, out_dir="./data/thermo/_sample", temps=grid)

    print(f"\ngrid: {', '.join(f'{t:g}' for t in grid)} C")
    print(f"{'feature':<24}{'mean':>9}{'sd':>8}{'med':>9}{'nan%':>7}{'rho':>8}")
    for c in FEATURE_COLS:
        v = feats[c].to_numpy(dtype=float)
        finite = np.isfinite(v)
        rho = pd.Series(v[finite]).corr(df["growth_rate"][finite], method="spearman")
        print(
            f"{c:<24}{np.nanmean(v):9.3f}{np.nanstd(v):8.3f}{np.nanmedian(v):9.3f}"
            f"{100 * (~finite).mean():7.1f}{rho:+8.3f}"
        )


# command: uv run python -m src.thermodynamics --sequence [insert here] --plot (optional)