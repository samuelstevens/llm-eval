# Using LLaVA for Llama2

[LLaVA](https://llava-vl.github.io/) is a vision-languge model that uses the output vectors of a ViT/L CLIP as inputs to a Llama2 language model.
They project the output patches into the LM embedding space using a small (two layer MLP) projection network.
The MLP is tuned on some large-scale vision-language data, then the MLP and the LM weights are tuned on some instruction vision-language data.
If you want to use LLaVA as a Llama2 model, you can look at the weights downloaded in the `pytorch_model.bin.index.json` in the HF checkpoint for LLaVA.
The model zoo is [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) and the default model is [here on HF](https://huggingface.co/liuhaotian/llava-v1.5-7b).

```json
{
  "metadata": {
    "total_size": 13518798848
  },
  "weight_map": {
    "lm_head.weight": "pytorch_model-00002-of-00002.bin",
    "model.embed_tokens.weight": "pytorch_model-00001-of-00002.bin",
    "model.layers.0.input_layernorm.weight": "pytorch_model-00001-of-00002.bin",
    "model.layers.0.mlp.down_proj.weight": "pytorch_model-00001-of-00002.bin",
    ...
    "model.layers.9.self_attn.rotary_emb.inv_freq": "pytorch_model-00001-of-00002.bin",
    "model.layers.9.self_attn.v_proj.weight": "pytorch_model-00001-of-00002.bin",
    "model.mm_projector.0.bias": "pytorch_model-00002-of-00002.bin", // <- These are the MLP weights
    "model.mm_projector.0.weight": "pytorch_model-00002-of-00002.bin",
    "model.mm_projector.2.bias": "pytorch_model-00002-of-00002.bin",
    "model.mm_projector.2.weight": "pytorch_model-00002-of-00002.bin",
    "model.norm.weight": "pytorch_model-00002-of-00002.bin"
  }
}
```

The weights for `mm_projector` and `norm` are extra.
The CLIP weights are not included in the HF checkpoint because they aren't tuned.
To use the tuned LM checkpoint on a language-only task, you can just ignore the extra weights and load it like a normal Llama2 model.

I don't know how to do that in HF, but you can see the `scripts/convert_hf_checkpoint.py` script which does it for the model definition in `models/` (from Meta's gpt-fast).
