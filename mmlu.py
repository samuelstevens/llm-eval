"""
Evaluate an LLM on MMLU
"""

import argparse
import logging
import os

import numpy as np
import polars as pl
import torch

import generate
import helpers
import models

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
precision = torch.bfloat16

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("mmlu")

letters = ["A", "B", "C", "D"]


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate an LLM on MMLU")
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
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile.")
    # Data
    parser.add_argument("--data", type=str, default="data/mmlu")
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--split", type=str, default="val")
    # Results
    parser.add_argument("--out", type=str, default="results/mmlu")
    args = parser.parse_args()

    if not args.tokenizer_path:
        ckpt_parent = os.path.dirname(args.model_path)
        args.tokenizer_path = os.path.join(ckpt_parent, "tokenizer.model")

    return args


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax


def format_subject(subject):
    return subject.replace("_", " ")


def format_example(question, choices, answer, *, include_answer):
    """
    Arguments:
        question (str): the question
        choices (list[str]): possible answers
        answer (str): single letter answer
    """
    prompt = question
    for i, choice in enumerate(choices):
        prompt += f"\n{letters[i]}. {choice}"
    prompt += "\nAnswer:"
    if include_answer:
        assert answer in letters
        prompt += f" {answer}\n\n"
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    if k > 0:
        train_df = train_df.head(k)

    for question, *choices, answer in train_df.iter_rows():
        prompt += format_example(question, choices, answer, include_answer=True)
    return prompt


@helpers.timed
@torch.no_grad
def evaluate(model, tokenizer, subject, dev_df, test_df, k):
    assert dev_df.shape[1] == 6
    assert test_df.shape[1] == 6

    corrects = []

    letter_tokens = torch.tensor([tokenizer.encode(c) for c in letters]).flatten()

    for question, *choices, label in test_df.iter_rows():
        # get prompt and make sure it fits
        prompt_end = format_example(question, choices, "", include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while generate.tokenize(tokenizer, prompt).size(0) > model.config.block_size:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        # Tokenize prompt
        tokens = generate.tokenize(tokenizer, prompt)
        input_pos = torch.arange(0, tokens.size(0), device=device)

        logits = model(tokens.view(1, -1), input_pos)
        pred = letters[logits[0, -1, letter_tokens].argmax()]

        correct = pred == label
        corrects.append(correct)

    acc = np.mean(corrects)


    return acc, corrects


if __name__ == "__main__":
    args = get_args()

    suffix = f"_{args.split}.csv"
    subjects = [
        f.split(suffix)[0]
        for f in os.listdir(os.path.join(args.data, args.split))
        if suffix in f
    ]
    subjects = sorted(subjects)
    print(" ".join(subjects))

    model_cfg = models.ModelArgs(**models.configs[args.model_name])
    model = models.load_model(model_cfg, args.model_path, device, precision)
    tokenizer = models.load_tokenizer(args.tokenizer_path)

    # Set up caches. Only runs when max_seq_length is bigger than before.
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=model_cfg.block_size)

    if not args.no_compile:
        # Use dynamic=True because the model has a variable number of prompt tokens.
        model = torch.compile(
            model, mode="reduce-overhead", fullgraph=True, dynamic=True
        )

    all_corrects = []
    for subject in subjects:
        dev_df = pl.read_csv(
            os.path.join(args.data, "dev", subject + "_dev.csv"),
            has_header=False,
            new_columns=["question", "A", "B", "C", "D", "answer"],
        )
        test_df = pl.read_csv(
            os.path.join(args.data, args.split, f"{subject}_{args.split}.csv"),
            has_header=False,
            new_columns=["question", "A", "B", "C", "D", "answer"],
        )

        (acc, corrects), time_s = evaluate(
            model, tokenizer, subject, dev_df, test_df, args.ntrain
        )
        all_corrects.extend(corrects)
        logger.info(
            "acc: %.1f, time: %.1fs, s/ex: %.2f, subject: %s",
            acc * 100,
            time_s,
            time_s / len(corrects),
            subject
        )

    weighted_acc = np.mean(all_corrects)
    print("Average accuracy: {:.3f}".format(weighted_acc))
