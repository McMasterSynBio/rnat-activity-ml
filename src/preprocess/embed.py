import os, time
from pathlib import Path
import numpy as np
import torch

from src.encoder.generator import EncoderGenerator, RNAEncoder

def choose_torch_device() -> torch.device:
    """Choose appropriate torch device based on the hardware."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def compute_and_save_chunk_embeddings(
    sequences: list[str],
    out_dir: str,
    encoder: RNAEncoder,
    batch_size: int = 64,
    device: torch.device = None
):
    # Check if file already exists, return if it does
    encoder_name = encoder.value.split('/')[-1]
    encoder_dir = Path(out_dir) / encoder_name
    encoder_dir.mkdir(parents=True, exist_ok=True)
    out_path = encoder_dir / f"{encoder_name}.npy"
    if os.path.isfile(out_path):
        return
    
    # Define torch device and config
    device = device or choose_torch_device()
    encoder_generator = EncoderGenerator(encoder)
    model, tokenizer = encoder_generator.load_encoder_and_tokenizer()
    model.to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        # Introduce verbose
        t0 = time.time(); done = 0; total = len(sequences)
        # Process sequences in batches
        for bi,i in enumerate(range(0, len(sequences), batch_size)):
            batch_sequences = sequences[i:i + batch_size]
            inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            last_hidden = outputs.last_hidden_state  # (B, L, H)
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                summed = (last_hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / denom
            else:
                pooled = last_hidden.mean(dim=1)

            all_embs.append(pooled.cpu().numpy().astype(np.float32))

            # Verbose output
            done += len(sequences[i:i+batch_size])
            if bi % 20 == 0 or done == total:
                el   = time.time() - t0
                rate = done / el
                eta  = (total - done) / rate
                print(f"  [{encoder.value}] {done:>7,}/{total:,} ({done/total*100:5.1f}%) "
                    f"| {rate:6.0f} seq/s | elapsed {el:5.1f}s | eta {eta:6.1f}s", flush=True)

    if all_embs:
        all_embs = np.vstack(all_embs)
    else:
        all_embs = np.zeros((0, model.config.hidden_size), dtype=np.float32)

    # Save the computed embeddings
    np.save(str(out_path), all_embs)
    return out_path, all_embs.shape