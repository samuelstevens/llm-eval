# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import pathlib
import re

import safetensors
import torch

import models

# Map from Huggingface to models/base.py
# TODO: the code to map from HF llama to models/base.py is not for phi2 or other models, so that code needs to become llama2-specific.
weight_map = {
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


def load_safetensors_ckpt(bin_files):
    merged = {}
    for file in sorted(bin_files):
        with safetensors.safe_open(str(file), framework="pt", device="cpu") as fd:
            for key in fd.keys():
                merged[key] = fd.get_tensor(key)
    return merged


def load_torch_ckpt(bin_files):
    merged = {}
    for file in sorted(bin_files):
        state_dict = torch.load(
            str(file), map_location="cpu", mmap=True, weights_only=True
        )
        merged.update(state_dict)
    return merged


def load_bin_files(map_json: pathlib.Path):
    with open(map_json) as fd:
        bin_index = json.load(fd)
    bin_files = {map_json.parent / bin for bin in bin_index["weight_map"].values()}
    return bin_files


@torch.inference_mode()
def convert_hf_checkpoint(input_dir: str, name: str, output_dir: str) -> None:
    # Convert to pathlib
    input_dir = pathlib.Path(input_dir)
    output_dir = pathlib.Path(output_dir)

    config = models.ModelArgs(**models.configs[name])
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping and the weights
    if (input_dir / "model.safetensors.index.json").is_file():
        model_map_json = input_dir / "model.safetensors.index.json"
        bin_files = load_bin_files(model_map_json)
        merged = load_safetensors_ckpt(bin_files)
    elif (input_dir / "pytorch_model.bin.index.json").is_file():
        model_map_json = input_dir / "pytorch_model.bin.index.json"
        bin_files = load_bin_files(model_map_json)
        merged = load_torch_ckpt(bin_files)
    else:
        raise ValueError(f"No model map file in {input_dir}")

    final_result = {}
    for key, value in merged.items():
        if "layers" in key:
            abstract_key = re.sub(r"(\d+)", "{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    def permute(w, n_head):
        return (
            w.view(n_head, 2, config.head_dim // 2, config.dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, config.dim)
        )

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            q = permute(q, config.n_head)
            k = permute(k, config.n_local_heads)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]

    print(f"Saving checkpoint to {output_dir / 'model.pth'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(final_result, output_dir / "model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HuggingFace checkpoint.")
    parser.add_argument(
        "--input", type=str, help="Path to dir containing .safetensor files"
    )
    parser.add_argument("--output", type=str, help="Path to save .pth file")
    parser.add_argument("--name", type=str, default="llama2-7b", help="Model to use")

    args = parser.parse_args()
    convert_hf_checkpoint(args.input, args.name, args.output)
