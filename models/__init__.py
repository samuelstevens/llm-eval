import logging

import sentencepiece
import torch

from .base import ModelArgs, Transformer

logger = logging.getLogger("lib/model")

configs = {
    # Llama2
    "llama2-7b": dict(n_layer=32, n_head=32, dim=4096),
    "llama2-13b": dict(n_layer=40, n_head=40, dim=5120),
    # Chat variants
    "llama2-7b-chat": dict(n_layer=32, n_head=32, dim=4096),
    # Vicuna
    "vicuna-7b-v1.5": dict(n_layer=32, n_head=32, dim=4096),
    "vicuna-13b-v1.5": dict(n_layer=40, n_head=40, dim=5120),
    # LLaVA
    "llava-7b-v1.5": dict(n_layer=32, n_head=32, dim=4096),
    "llava-13b-v1.5": dict(n_layer=40, n_head=40, dim=5120),
}

# TODO: the code to map from HF llama to models/base.py is not for phi2 or other models, so that code needs to become llama2-specific.

# Map from Huggingface to models/base.py
_default_llama2_weight_map = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}


llava_extra_weights_keys = [
    "model.mm_projector.0.weight",
    "model.mm_projector.0.bias",
    "model.mm_projector.0.weight",
    "model.mm_projector.2.bias",
    "model.mm_projector.2.weight",
    "model.norm.weight",
]

weight_maps = {
    "llama2-7b": (_default_llama2_weight_map, set()),
    "llama2-13b": (_default_llama2_weight_map, set()),
    "llama2-7b-chat": (_default_llama2_weight_map, set()),
    "vicuna-7b-v1.5": (_default_llama2_weight_map, set()),
    "vicuna-13b-v1.5": (_default_llama2_weight_map, set()),
    "llava-7b-v1.5": (_default_llama2_weight_map, set(llava_extra_weights_keys)),
    "llava-13b-v1.5": (_default_llama2_weight_map, set(llava_extra_weights_keys)),
}


def load_config(name):
    return ModelArgs(**configs[name])


def load_model(config, ckpt_path, device, precision):
    with torch.device("meta"):
        model = Transformer(config)

    if "int8" in str(ckpt_path):
        logger.info("Using int8 weight-only quantization.")
        from quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

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
