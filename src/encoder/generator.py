from enum import Enum
import os
from multimolecule import AutoModel, AutoTokenizer
from transformers import AutoModelForMaskedLM

class RNAEncoder(Enum):
    RNAFM = "multimolecule/rnafm" # https://huggingface.co/multimolecule/rnafm
    MRNAFM = "multimolecule/mrnafm" # https://huggingface.co/multimolecule/mrnafm
    ERNIERNA = "multimolecule/ernierna" # https://huggingface.co/multimolecule/ernierna
    UTRLM = "multimolecule/utrlm-te_el" # https://huggingface.co/multimolecule/utrlm-te_el
    UTRLM_MRL = "multimolecule/utrlm-mrl" # https://huggingface.co/multimolecule/utrlm-mrl
    UTBERT_3MER = "multimolecule/utrbert-3mer" # https://huggingface.co/multimolecule/utrbert-3mer
    RNABERT = "multimolecule/rnabert" # https://huggingface.co/multimolecule/rnabert
    RNAERNIE = "multimolecule/rnaernie" # https://huggingface.co/multimolecule/rnaernie
    RNAMSM = "multimolecule/rnamsm" # https://huggingface.co/multimolecule/rnamsm
    SPLICEBERT = "multimolecule/splicebert" # https://huggingface.co/multimolecule/splicebert
    RINALMO_MICRO = "multimolecule/rinalmo-micro" # https://huggingface.co/multimolecule/rinalmo-micro
    RINALMO_GIGA = "multimolecule/rinalmo-giga" # https://huggingface.co/multimolecule/rinalmo-giga
    RINALMO_MEGA = "multimolecule/rinalmo-mega" # https://huggingface.co/multimolecule/rinalmo-mega

class EncoderGenerator:

    def __init__(self, encoder: RNAEncoder):
        self.encoder = encoder

    def _load_from_cache(self) -> tuple[AutoModel, AutoTokenizer]:
        """Load the encoder model and tokenizer from cache if available."""
        cache_path = os.path.join('./.cache/', self.encoder.value.split('/')[-1])
        if os.path.exists(cache_path):
            model = AutoModel.from_pretrained(cache_path)
            tokenizer = AutoTokenizer.from_pretrained(cache_path)
            return model, tokenizer
        else:
            return None, None

    def load_encoder_and_tokenizer(self) -> tuple[AutoModel, AutoTokenizer]:
        """Load the encoder model and tokenizer, either from cache or by downloading."""
        model, tokenizer = self._load_from_cache()
        if model is None or tokenizer is None:
            model, tokenizer = self.fetch_encoder()
            self.save_encoder(model, tokenizer)
        return model, tokenizer

    def fetch_encoder(self) -> tuple[AutoModel, AutoTokenizer]:
        if self.encoder is RNAEncoder.RNABERT:
            # publisher mentioned some weird tokenization behaviour to patch: https://huggingface.co/multimolecule/rnabert
            return AutoModel.from_pretrained(self.encoder.value), AutoTokenizer.from_pretrained(self.encoder.value, use_fast=False)
        elif self.encoder.value.split('/')[0] == "multimolecule":
            return AutoModel.from_pretrained(self.encoder.value), AutoTokenizer.from_pretrained(self.encoder.value)
        else:
            # This will return the transformer implementation of the encoder
            return None, None
        
    def save_encoder(self, model, tokenizer, save_path: str = './.cache/'):
        """Save the encoder model and tokenizer currently in session."""
        save_path = os.path.join(save_path, self.encoder.value.split('/')[-1])
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    # Example usage and output embeddings for a given sequence
    import argparse
    parser = argparse.ArgumentParser(description="Generate embeddings for a given RNA sequence using a specified encoder.")
    parser.add_argument("--encoder", type=str, required=True, help="The RNA encoder to use (e.g., RNAFM, MRNAFM, ERNIERNA, etc.).")
    parser.add_argument("--sequence", type=str, required=True, help="The RNA sequence to encode.")
    args = parser.parse_args()

    assert args.encoder in RNAEncoder.__members__, f"Invalid encoder. Choose from: {list(RNAEncoder.__members__.keys())}"

    encoder_enum = RNAEncoder[args.encoder]
    encoder_generator = EncoderGenerator(encoder_enum)
    model, tokenizer = encoder_generator.fetch_encoder()

    inputs = tokenizer(args.sequence, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

    print(f"Embeddings for the sequence '{args.sequence}' using encoder '{args.encoder}':")
    print(embeddings)