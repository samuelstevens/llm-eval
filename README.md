# LLM Evaluation

## Install

Install PyTorch nightly following [these instructions](https://pytorch.org/get-started/locally/).

```sh
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

You need PyTorch *nightly* and at least CUDA 12 to use the new `torch.compile` options.
For whatever reason, PyTorch nightly doesn't work on CUDA 11.8 ([see this issue](https://github.com/pytorch/pytorch/issues/106144)).
So you can use
* Stable PyTorch + `torch.compile` on CUDA 11.8
* PyTorch nightly with no compilation on CUDA 11.8 (`--no-compile`)
* PyTorch nightly with `torch.compile` on CUDA 12

## Download Data

### MMLU

```sh
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xvf data.tar
```

## Download Models

You can use `scripts/download.py` to download models.
But my research lab had HF Llama2 weights downloaded, so I ran:

```sh
python -m scripts.convert_hf_checkpoint \
  --input /research/nfs_su_809/huggingface_cache/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/ \
  --name llama2-7b \
  --output /local/scratch/stevens.994/models/torch/llama2-7b
```

You can also copy the tokenizer to keep everything together.

```sh
cp \
  /research/nfs_su_809/huggingface_cache/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/tokenizer.model \
  /local/scratch/stevens.994/models/torch/llama2-7b/
```

## Quantize to use int8

```sh
python quantize.py \
  --ckpt_path /research/nfs_su_809/workspace/stevens.994/models/torch/llama2-7b-chat/model.pth \
  --model_name llama2-7b-chat \
  --mode int8
```

This saves a model_int8.pth checkpoint.
Typically int8 has little to no performance degradations for language models.

## Evaluate

```sh
python mmlu.py \
  --data /local/scratch/stevens.994/datasets/mmlu/data \
  --model_name llama2-7b \
  --model_path /local/scratch/stevens.994/models/torch/llama2-7b/model.pth
```

