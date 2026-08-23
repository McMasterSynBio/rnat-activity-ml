import os
from pathlib import Path
import numpy as np
import torch

from src.encoder.generator import EncoderGenerator, RNAEncoder

def compute_and_save_chunk_embeddings(
    sequences: list[str],
    out_dir: str,
    encoder: RNAEncoder,
    chunk_index: int,
    batch_size: int = 64,
    device: torch.device = None
):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    encoder_generator = EncoderGenerator(encoder)
    model, tokenizer = encoder_generator.load_encoder_and_tokenizer()
    model.to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # choose pooling strategy, e.g., mean pooling
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output  # (B, H)
            else:
                last_hidden = outputs.last_hidden_state  # (B, L, H)
                if attention_mask is not None:
                    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                    summed = (last_hidden * mask).sum(dim=1)
                    denom = mask.sum(dim=1).clamp(min=1e-9)
                    pooled = summed / denom
                else:
                    pooled = last_hidden.mean(dim=1)

            all_embs.append(pooled.cpu().numpy().astype(np.float32))

    if all_embs:
        all_embs = np.vstack(all_embs)
    else:
        all_embs = np.zeros((0, model.config.hidden_size), dtype=np.float32)

    encoder_name = encoder.value.split('/')[-1]
    encoder_dir = Path(out_dir) / encoder_name
    encoder_dir.mkdir(parents=True, exist_ok=True)
    out_path = encoder_dir / f"{encoder_name}_chunk{chunk_index}.npy"
    np.save(str(out_path), all_embs)
    return out_path, all_embs.shape