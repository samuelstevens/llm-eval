import torch
import torch._dynamo.config
import torch._inductor.config
from torch import Tensor

from models import Transformer

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental feature to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True


def multinomial_sample_one_no_sync(probs_sort):
    # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: int | None = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: int | None = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: Transformer, x: Tensor, input_pos: Tensor, **sampling_kwargs
) -> Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
    model: Transformer, x: Tensor, input_pos: Tensor, **sampling_kwargs
) -> tuple[Tensor, Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    cur_token: Tensor,
    input_pos: Tensor,
    num_new_tokens: int,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)

    return new_tokens, new_probs


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: Tensor,
    max_new_tokens: int,
    **sampling_kwargs,
) -> Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    Arguments:
        model (Transformer): Transformer instance from models/base.py
        prompt (Tensor): Vocab tokens with shape
    """

    breakpoint()

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    seq[T] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    generated_tokens, _ = decode_n_tokens(
        model,
        next_token.view(1, -1),
        input_pos,
        max_new_tokens - 1,
        **sampling_kwargs,
    )
    seq[T + 1 :] = torch.cat(generated_tokens)

    return seq


def tokenize(tokenizer, string, bos=True, device="cuda"):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def detokenize(tokenizer, tokens):
    breakpoint()
