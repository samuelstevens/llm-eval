import argparse
import itertools
import json
import logging
import os

import human_eval.data
import human_eval.evaluation
import torch
from tqdm import tqdm

import generate
import helpers
import models

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
precision = torch.bfloat16

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("human-eval")

out_file = "samples.jsonl"


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate an LLM on Human Eval")
    # Model
    parser.add_argument(
        "--model_name", type=str, default="llama2-7b", help="Model to use"
    )
    parser.add_argument("--model_path", type=str, help="model.pth checkpoint")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="tokenizer checkpoint. Looks in the same directory as --model_path if not set.",
    )
    parser.add_argument("--compile", action="store_true", help="Use torch.compile.")
    # Task
    parser.add_argument("--samples-per-task", type=int, default=200)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    # Results
    parser.add_argument("--out", type=str, default="results/human_eval")
    args = parser.parse_args()

    if not args.tokenizer_path:
        ckpt_parent = os.path.dirname(args.model_path)
        args.tokenizer_path = os.path.join(ckpt_parent, "tokenizer.model")

    # Make output directory
    args.out = os.path.join(args.out, helpers.fs_safe(args.model_name))
    os.makedirs(args.out, exist_ok=True)

    # Save parameters
    with open(os.path.join(args.out, "params.json"), "w") as fd:
        params = {name: getattr(args, name) for name in vars(args)}
        json.dump(params, fd, sort_keys=True, indent=4)

    return args


def evaluation(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = human_eval.data.HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = human_eval.evaluation.evaluate_functional_correctness(
        sample_file, k, n_workers, timeout, problem_file
    )
    print(results)


def detokenize_batch(batch) -> tuple[list[str], int]:
    texts = []
    n_tokens = 0
    for tokens in batch:
        eos_pos = (tokens == tokenizer.eos_id()).nonzero()
        if eos_pos.numel() == 0:
            eos_pos = tokens.numel()
        elif eos_pos.numel() == 1:
            eos_pos = eos_pos.item()
        else:
            eos_pos = eos_pos.min()

        texts.append(generate.detokenize(tokenizer, tokens[: eos_pos + 1]))
        n_tokens += eos_pos + 1

    return texts, n_tokens


def decode_n_tokens(
    model: models.Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    **sampling_kwargs,
):
    seen_eos = torch.zeros((args.batch_size, 1), device=device)
    new_tokens = []
    for i in range(args.max_new_tokens - 1):
        # Actually better for Inductor to codegen attention here
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            next_token, _ = generate.decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            # view() prevents compile() from complaining about value re-use.
            cur_token = next_token.view(args.batch_size, -1)

            # Stop early if all sequences in batch have seen the eos
            seen_eos[cur_token == tokenizer.eos_id()] = True
            if seen_eos.all():
                break

    return new_tokens


@helpers.timed
@torch.no_grad()
def complete(prompt, **sampling_kwargs) -> torch.Tensor:
    # create an empty tensor of the expected final shape and fill in the current tokens
    B, T = prompt.shape
    T_new = T + args.max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)

    device = prompt.device
    with torch.device(device):
        model.setup_caches(max_batch_size=B, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    input_pos = torch.arange(0, T, device=device)

    next_tokens = generate.prefill(model, prompt, input_pos, **sampling_kwargs)
    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    generated_tokens = decode_n_tokens(
        model,
        next_tokens,
        input_pos,
        **sampling_kwargs,
    )
    return torch.cat(generated_tokens, dim=1)


def complete_code(prompts: str) -> list[str]:
    """
    Returns args.batch_size completions for a prompt.
    """
    tokens = generate.tokenize(tokenizer, prompt).expand(args.batch_size, -1)

    new_tokens, time_s = complete(tokens)
    completions, n_tokens = detokenize_batch(new_tokens)

    logger.debug(
        "%d toks, %.1f secs, %.1f toks/sec", n_tokens, time_s, n_tokens / time_s
    )

    return completions


def _vicuna_prompt(problem):
    user_msg_lines = []
    user_msg_lines.append(
        "Complete the following Python code without any tests or explanation."
    )
    user_msg_lines.append("```python")
    user_msg_lines.append(problem.strip())
    user_msg_lines.append("```")
    user_msg = "\n".join(user_msg_lines)

    return f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nUSER: {user_msg}\nASSISTANT: "


def _llama2_chat_prompt(problem):
    return f"[INST] Complete the following Python code without any tests or explanation. Only write valid Python code. {problem} [/INST]"


prompts = {
    "vicuna-7b-v1.5": _vicuna_prompt,
    "llama2-7b-chat": _llama2_chat_prompt,
}


def clean_completion(completion: str) -> str:
    return completion.strip().removeprefix("```python").removesuffix("```").strip()


if __name__ == "__main__":
    # TODO:
    # * int8 quantization
    # * batch size 4
    args = get_args()

    model_config = models.load_config(args.model_name)
    model = models.load_model(model_config, args.model_path, device, precision)
    tokenizer = models.load_tokenizer(args.tokenizer_path)

    # Set up caches. Only runs when max_seq_length is bigger than before.
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=model_config.block_size)

    if args.compile:
        generate.decode_one_token = torch.compile(
            generate.decode_one_token, mode="reduce-overhead", fullgraph=True
        )

    results_file = os.path.join(args.out, "results.jsonl")
    # Make sure file is empty
    open(results_file, "w").close()

    def write_result(task_id, completion):
        line = {"task_id": task_id, "completion": completion}
        with open(results_file, "a") as fd:
            fd.write(json.dumps(line) + "\n")

    problems = human_eval.data.read_problems()
    it = itertools.product(problems, range(args.samples_per_task))
    for task_id, _ in tqdm(it, total=len(problems) * args.samples_per_task):
        problem = problems[task_id]["prompt"]
        prompt = prompts[args.model_name](problem)
        completion = complete_code(prompt, args.max_tokens)
        cleaned = clean_completion(completion)
        write_result(task_id, cleaned)

    evaluation(results_file)
