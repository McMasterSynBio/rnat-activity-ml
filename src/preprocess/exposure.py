import os, time
from pathlib import Path
import numpy as np
import pandas as pd
import RNA

# First 33 nt of the S. cerevisiae HIS3 CDS (YOR202W, SGD S000005728).
# Cuperus 2017 Methods: the library fragment carried "60-bp and 33-bp 5' and 3'
# overlaps with the CYC1 promoter and the HIS3 coding sequence, respectively,
# including the ATG start codon" -- so the downstream context IS the CDS head.
# Verified: translates to MTEQKALVKRI; UniProt P06633 begins MTEQKALVKRITNETK.
HIS3_CONTEXT = "AUGACAGAGCAGAAAGCCCUAGUAAAGCGUAUU"

# Assay temperature. Cuperus grew at 30 C -- NOT the 37 C folding default.
ASSAY_TEMP_C = 30.0

FEATURE_COLS = [
    "p_unpaired_kozak",  # -6..+4 around the AUG
    "p_unpaired_cap",    # cap-proximal, where the 43S loads
    "p_unpaired_utr",    # whole UTR; a scanning ribosome unwinds all of it
    "dG_utr",            # ensemble free energy of the UTR alone
    "dG_whole",          # ...and of UTR + context, i.e. the real molecule
]


def _unpaired_and_dG(seq: str, md: RNA.md) -> tuple[np.ndarray, float]:
    """Per-base unpaired probability and ensemble free energy for one sequence."""
    fc = RNA.fold_compound(seq, md)
    _, dG = fc.pf()                        # must run before bpp()
    # bpp() is 1-indexed and upper-triangular, so drop the padding row/column and
    # add both axes -- each pair contributes to *both* partners, not just one.
    M = np.array(fc.bpp())[1:, 1:]
    return 1.0 - (M.sum(axis=0) + M.sum(axis=1)), dG


def exposure_features(
    utr: str,
    context: str = HIS3_CONTEXT,
    md: RNA.md = None
) -> dict[str, float]:
    """Thermodynamic accessibility features for one UTR.

    The context is appended for *folding only* and then discarded. Two reasons it
    is not optional: the AUG lives in the context, so without it there is no Kozak
    window at all; and folding a bare 50-mer deletes every pairing partner past the
    cut, which inflates unpaired probability at the 3' edge -- exactly the region
    that matters. Measured, position 49: 0.87 alone vs 0.45 with context.
    """
    # Default to the assay temperature, not RNA.md()'s bare 37 C -- otherwise a
    # standalone call silently disagrees with what the batch driver wrote.
    if md is None:
        md = RNA.md()
        md.temperature = ASSAY_TEMP_C
    start = len(utr)                       # index of the A in AUG, by construction
    unpaired, dG_whole = _unpaired_and_dG(utr + context, md)
    _, dG_utr = _unpaired_and_dG(utr, md)

    # Windows are relative to `start`, never absolute -- so the same code computes
    # the anatomically matching window on an RNAt of a different length. max()/min()
    # guard the short native UTRs (2-50 nt), where a fixed window would run off.
    return {
        "p_unpaired_kozak": unpaired[max(0, start - 6):start + 4].mean(),
        "p_unpaired_cap":   unpaired[0:min(15, start)].mean(),
        "p_unpaired_utr":   unpaired[0:start].mean(),
        "dG_utr":           dG_utr,
        "dG_whole":         dG_whole,
    }


def compute_and_save_exposure(
    sequences: list[str],
    out_dir: str,
    context: str = HIS3_CONTEXT,
    temperature: float = ASSAY_TEMP_C,
    report_every: int = 20000,
) -> pd.DataFrame:
    """Fold every sequence and return the features as a row-aligned DataFrame.

    Row i of the result corresponds to sequences[i], so the caller can concat it
    onto the dataframe as columns. ~290 seq/s, so ~23 min for 398k rows.
    """
    assert len(context) >= 3 and context[:3] == "AUG", "context must start at the AUG"

    out_path = Path(out_dir) / f"exposure_{temperature:g}C.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Reuse the cache only if it is the right length. embed.py skips on filename
    # alone, which silently returns wrong rows after a filter change; here a stale
    # file just triggers a recompute instead.
    if os.path.isfile(out_path):
        cached = pd.read_csv(out_path)
        if len(cached) == len(sequences):
            print(f"  [exposure] reusing cache: {out_path}")
            return cached
        print(f"  [exposure] cache is stale ({len(cached):,} != {len(sequences):,}), recomputing")

    md = RNA.md()
    md.temperature = temperature           # default is 37 C; the assay was 30 C

    rows = []
    t0 = time.time()
    for i, utr in enumerate(sequences, start=1):
        rows.append(exposure_features(utr, context=context, md=md))
        if i % report_every == 0 or i == len(sequences):
            el = time.time() - t0
            rate = i / el
            print(f"  [exposure @{temperature:g}C] {i:>7,}/{len(sequences):,} "
                  f"({i/len(sequences)*100:5.1f}%) | {rate:6.0f} seq/s | "
                  f"elapsed {el:5.1f}s | eta {(len(sequences)-i)/rate:6.1f}s", flush=True)

    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    df.to_csv(out_path, index=False)
    print(f"Saved exposure features to '{out_path}'.")
    return df


if __name__ == "__main__":
    """Standalone check. Folds a sample and prints distributions to compare against
    the reference below -- if these drift, the windowing or indexing is wrong.

        feature             mean     sd     med    rho vs growth_rate
        p_unpaired_kozak   0.521  0.163   0.517   +0.089
        p_unpaired_cap     0.533  0.162   0.525   +0.071
        p_unpaired_utr     0.504  0.082   0.501   +0.215
        dG_utr           -10.468  3.870 -10.293   +0.272
        dG_whole         -18.286  4.294 -18.141   +0.303   <- strongest single feature

    Note dG_whole alone already reaches ridge rho ~0.29 on held-out data and the
    other four add ~nothing linearly; they are kept for the nonlinear head and for
    the RNAt temperature sweep, where "which region opened" matters.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Compute exposure features for a UTR sample.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file.")
    parser.add_argument("--n", type=int, default=2000, help="Rows to sample (0 = all).")
    parser.add_argument("--temperature", type=float, default=ASSAY_TEMP_C, help="Folding temperature (C).")
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)
    df = df[df["t0"] >= 20].reset_index(drop=True)
    if args.n:
        df = df.sample(args.n, random_state=0).reset_index(drop=True)
    utrs = df["UTR"].str.replace("T", "U").tolist()

    feats = compute_and_save_exposure(utrs, out_dir="./data/exposure/_sample",
                                      temperature=args.temperature)

    print(f"\n{'feature':<18}{'mean':>9}{'sd':>8}{'med':>9}{'rho':>8}")
    for c in FEATURE_COLS:
        v = feats[c].values
        rho = pd.Series(v).corr(df["growth_rate"], method="spearman")
        print(f"{c:<18}{v.mean():9.3f}{v.std():8.3f}{np.median(v):9.3f}{rho:+8.3f}")
