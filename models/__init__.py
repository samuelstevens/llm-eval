import sentencepiece
import torch

from .base import ModelArgs, Transformer

configs = {
    "llama2-7b": dict(n_layer=32, n_head=32, dim=4096),
    "vicuna-7b-v1.5": dict(n_layer=32, n_head=32, dim=4096),
    "llama2-13b": dict(n_layer=40, n_head=40, dim=5120),
}


def load_model(config, ckpt_path, device, precision):
    with torch.device("meta"):
        model = Transformer(config)

    ckpt = torch.load(ckpt_path, mmap=True, weights_only=True)
    model.load_state_dict(ckpt, assign=True)
    model = model.to(device=device, dtype=precision)
    model.eval()

    return model


def load_tokenizer(path: str):
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=path)
    return tokenizer
    raise NotImplementedError(f"load_tokenizer('{path}')")


__all__ = ["ModelArgs", "Transformer", "configs", "weight_maps", "load_model"]
