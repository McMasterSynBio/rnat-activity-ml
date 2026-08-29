import pandas as pd
import numpy as np
import os

from src.encoder.generator import EncoderGenerator, RNAEncoder
from src.preprocess.embed import compute_and_save_chunk_embeddings
from src.preprocess.exposure import compute_and_save_exposure

def load_dataset_in_chunks(
    file_path: str,
    encoder: RNAEncoder,
    chunk_size: int = 10000,
    test_size: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the dataset from a csv file."""
    full_df = pd.DataFrame()
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Remove t1 as the fold change is already covered in growth_rate
        chunk = chunk.drop(columns=['t1'])
        chunk = chunk[chunk['t0'] >= 20] # Filter for initial counts
        # Replace all T -> U in sequence column
        chunk['UTR'] = chunk['UTR'].str.replace('T', 'U')
        # Append to the full dataframe
        full_df = pd.concat([full_df, chunk], ignore_index=True)
    # Save the full dataframe vector embeddings for all encoders
    file_name = os.path.basename(file_path).split('.')[0]
    full_df = full_df.reset_index(drop=True) # ensure index order is consistent across runs
    compute_and_save_chunk_embeddings(
        full_df['UTR'].tolist(),
        out_dir=f'./data/embeddings/{file_name}',
        encoder=encoder,
    )
    print(f"Embeddings have been generated for {encoder.name}")
    # concatenate dataset with nupack thermodynamics features
    feats = compute_and_save_exposure(
        full_df['UTR'].tolist(),
        out_dir=f'./data/exposure/{file_name}'
    )
    full_df = pd.concat([full_df, feats], axis=1)
    # index rows before saving
    full_df['emb_idx'] = np.arange(len(full_df))
    # train-test split
    test_set = full_df.nlargest(int(len(full_df) * test_size), 't0')
    train_set = full_df.drop(test_set.index)
    # z-score normalization on label
    train_mean = train_set['growth_rate'].mean(skipna=True)
    train_std = train_set['growth_rate'].std(ddof=1, skipna=True)
    z_norm = lambda x: (x - train_mean) / train_std
    # apply z-norm on set
    train_set['growth_rate_z'] = z_norm(train_set['growth_rate'])
    test_set['growth_rate_z'] = z_norm(test_set['growth_rate'])
    # Use top 5% by t0 as test split
    save_path = f'./data/processed/{file_name}'
    os.makedirs(save_path, exist_ok=True)
    train_set.to_csv(f'{save_path}/train_set.csv', index=False)
    test_set.to_csv(f'{save_path}/test_set.csv', index=False)
    print(f"Saved the final datasets to '{save_path}'.")
    # Return the final data frames
    return train_set, test_set


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process a dataset in chunks.")
    parser.add_argument("--file_path", type=str, default="./data/raw/GSM2793752_Random_UTRs.csv.gz", help="Path to the CSV file to process.")
    parser.add_argument("--encoder", type=str, default="UTRLM", help="The RNA encoder to use (e.g., RNAFM, MRNAFM, ERNIERNA, etc.).")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Number of rows per chunk.")
    args = parser.parse_args()

    assert args.encoder in RNAEncoder.__members__, f"Invalid encoder. Choose from: {list(RNAEncoder.__members__.keys())}"

    encoder = RNAEncoder[args.encoder]
    load_dataset_in_chunks(args.file_path, encoder, chunk_size=args.chunk_size)
