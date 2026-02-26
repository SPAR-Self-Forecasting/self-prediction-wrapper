import os, math, argparse
import numpy as np
import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

WRAPPER = 'What would you say in response to this prompt: "{p}"'

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    # p,q are prob vectors
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * torch.log(p / m)).item()
    kl_qm = torch.sum(q * torch.log(q / m)).item()
    return 0.5 * (kl_pm + kl_qm)

def format_chat(tokenizer, user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

@torch.no_grad()
def next_token_probs(model, tokenizer, text: str, device: str):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    logits = out.logits[0, -1, :]  # next-token logits after the prompt
    probs = torch.softmax(logits, dim=-1)
    return probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model name")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--project", default="self_prediction_wrapper")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    # Minimal prompt set: simple instruction-ish + factual-ish + open-ended
    base_prompts = [
        "Explain Bayes' theorem in one paragraph.",
        "Write a haiku about winter.",
        "What is the capital of France?",
        "Give three pros and cons of nuclear energy.",
        "Solve: 17*23.",
        "Write a short email declining a meeting politely.",
        "What is 2+2?",
        "Summarize the plot of Hamlet in two sentences.",
        "Give an example of a prime number greater than 100.",
        "Describe how to make scrambled eggs.",
    ]
    # Expand to n_prompts by sampling with small variations
    prompts = []
    for i in range(args.n_prompts):
        p = base_prompts[i % len(base_prompts)]
        if i >= len(base_prompts):
            p = p + f" (variant {i})"
        prompts.append(p)

    use_chat = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None

    run = wandb.init(project=args.project, config={
        "model": args.model,
        "n_prompts": args.n_prompts,
        "device": device,
        "seed": args.seed,
        "wrapper": WRAPPER,
        "use_chat_template": use_chat,
    })

    js_scores = []
    top1_same = 0
    examples = []

    for i, p in enumerate(prompts):
        direct = format_chat(tokenizer, p) if use_chat else p
        wrapped = format_chat(tokenizer, WRAPPER.format(p=p)) if use_chat else WRAPPER.format(p=p)

        probs_d = next_token_probs(model, tokenizer, direct, device)
        probs_w = next_token_probs(model, tokenizer, wrapped, device)

        js = js_divergence(probs_d, probs_w)
        js_scores.append(js)

        top_d = int(torch.argmax(probs_d).item())
        top_w = int(torch.argmax(probs_w).item())
        if top_d == top_w:
            top1_same += 1

        if i < 20:
            examples.append({
                "prompt": p,
                "js": js,
                "top1_direct": tokenizer.decode([top_d]),
                "top1_wrapped": tokenizer.decode([top_w]),
            })

        if (i + 1) % 25 == 0:
            wandb.log({
                "js_mean_so_far": float(np.mean(js_scores)),
                "top1_agreement_so_far": top1_same / (i + 1),
                "step": i + 1,
            })

    js_mean = float(np.mean(js_scores))
    js_median = float(np.median(js_scores))
    top1_agree = top1_same / len(prompts)

    wandb.log({
        "js_mean": js_mean,
        "js_median": js_median,
        "top1_agreement": top1_agree,
    })

    # Log a small table of qualitative examples
    wandb.log({"examples": wandb.Table(data=[
        [e["prompt"], e["js"], e["top1_direct"], e["top1_wrapped"]] for e in examples
    ], columns=["prompt", "js", "top1_direct", "top1_wrapped"])})

    print("done")
    print("device:", device)
    print("js_mean:", js_mean)
    print("js_median:", js_median)
    print("top1_agreement:", top1_agree)

    run.finish()

if __name__ == "__main__":
    main()