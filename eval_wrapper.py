import os, math, argparse
import numpy as np
import wandb
import torch

WRAPPER = 'What would you say in response to this prompt: "{p}"'

BASE_PROMPTS = [
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


# ---------------------------------------------------------------------------
# JS divergence
# ---------------------------------------------------------------------------

def js_divergence_tensors(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """JS divergence between two full probability vectors (local models)."""
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * torch.log(p / m)).item()
    kl_qm = torch.sum(q * torch.log(q / m)).item()
    return 0.5 * (kl_pm + kl_qm)


def js_divergence_topk(logprobs_d: dict, logprobs_w: dict, eps: float = 1e-8) -> float:
    """Approximate JS divergence from two top-K logprob dicts {token: log_prob}.

    Tokens not in the other distribution get probability eps.
    """
    all_tokens = set(logprobs_d) | set(logprobs_w)
    p_vals, q_vals = [], []
    for tok in all_tokens:
        p_vals.append(math.exp(logprobs_d[tok]) if tok in logprobs_d else eps)
        q_vals.append(math.exp(logprobs_w[tok]) if tok in logprobs_w else eps)
    # Renormalise so they sum to 1
    p = np.array(p_vals, dtype=np.float64)
    q = np.array(q_vals, dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


# ---------------------------------------------------------------------------
# Local (HuggingFace) helpers
# ---------------------------------------------------------------------------

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
    logits = out.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    return probs


# ---------------------------------------------------------------------------
# API (OpenRouter) helpers
# ---------------------------------------------------------------------------

def make_api_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def api_next_token(client, model: str, prompt: str, max_tokens: int = 16):
    """Call OpenRouter and return (top1_token, logprobs_dict_or_None).

    logprobs_dict is {token_str: log_prob} for top-K tokens, or None if
    the model/provider doesn't support logprobs.
    """
    # Some providers require max_tokens >= 16; we only use the first token.
    # Thinking models need more tokens so reasoning doesn't consume them all.
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
        )
    except Exception:
        # Some models reject logprobs param — retry without it
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )

    if not resp.choices:
        return "", None

    choice = resp.choices[0]
    content = choice.message.content or ""
    # Take just the first token-ish chunk (first whitespace-delimited word)
    top1 = content.split()[0] if content.strip() else ""

    logprobs_dict = None
    if choice.logprobs and choice.logprobs.content:
        lp = choice.logprobs.content[0]
        top1 = lp.token  # use exact first token when logprobs available
        logprobs_dict = {}
        logprobs_dict[lp.token] = lp.logprob
        if lp.top_logprobs:
            for entry in lp.top_logprobs:
                logprobs_dict[entry.token] = entry.logprob

    return top1, logprobs_dict


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def build_prompts(n_prompts: int) -> list[str]:
    prompts = []
    for i in range(n_prompts):
        p = BASE_PROMPTS[i % len(BASE_PROMPTS)]
        if i >= len(BASE_PROMPTS):
            p = p + f" (variant {i})"
        prompts.append(p)
    return prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HF model name or OpenRouter model ID")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--project", default="self_prediction_wrapper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--api", action="store_true",
                    help="Use OpenRouter API instead of local model")
    ap.add_argument("--max_tokens", type=int, default=16,
                    help="Max tokens for API calls (increase for thinking models)")
    args = ap.parse_args()

    np.random.seed(args.seed)

    prompts = build_prompts(args.n_prompts)

    # ---- set up model or API client ----
    if args.api:
        client = make_api_client()
        device = "api"
        use_chat = True  # API models are always chat
    else:
        torch.manual_seed(args.seed)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        model.to(device)
        model.eval()
        use_chat = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None

    run = wandb.init(project=args.project, config={
        "model": args.model,
        "n_prompts": args.n_prompts,
        "device": device,
        "seed": args.seed,
        "wrapper": WRAPPER,
        "use_chat_template": use_chat,
        "api": args.api,
    })

    js_scores = []
    top1_same = 0
    examples = []
    has_logprobs = None  # will be set on first API call

    for i, p in enumerate(prompts):
        if args.api:
            direct_prompt = p
            wrapped_prompt = WRAPPER.format(p=p)

            tok_d, lp_d = api_next_token(client, args.model, direct_prompt, args.max_tokens)
            tok_w, lp_w = api_next_token(client, args.model, wrapped_prompt, args.max_tokens)

            if tok_d == tok_w:
                top1_same += 1

            js = None
            if lp_d is not None and lp_w is not None:
                has_logprobs = True
                js = js_divergence_topk(lp_d, lp_w)
                js_scores.append(js)
            elif has_logprobs is None:
                has_logprobs = False

            if i < 20:
                examples.append({
                    "prompt": p,
                    "js": js if js is not None else -1,
                    "top1_direct": tok_d,
                    "top1_wrapped": tok_w,
                })

            if (i + 1) % max(1, len(prompts) // 8) == 0:
                log = {
                    "top1_agreement_so_far": top1_same / (i + 1),
                    "step": i + 1,
                }
                if js_scores:
                    log["js_mean_so_far"] = float(np.mean(js_scores))
                wandb.log(log)

            print(f"  [{i+1}/{len(prompts)}] direct={tok_d!r}  wrapped={tok_w!r}"
                  f"  same={tok_d == tok_w}"
                  + (f"  js={js:.4f}" if js is not None else ""))

        else:
            # Local model path (unchanged)
            direct = format_chat(tokenizer, p) if use_chat else p
            wrapped = format_chat(tokenizer, WRAPPER.format(p=p)) if use_chat else WRAPPER.format(p=p)

            probs_d = next_token_probs(model, tokenizer, direct, device)
            probs_w = next_token_probs(model, tokenizer, wrapped, device)

            js = js_divergence_tensors(probs_d, probs_w)
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

    # ---- final metrics ----
    top1_agree = top1_same / len(prompts)
    final = {"top1_agreement": top1_agree}
    if js_scores:
        js_mean = float(np.mean(js_scores))
        js_median = float(np.median(js_scores))
        final["js_mean"] = js_mean
        final["js_median"] = js_median
    wandb.log(final)

    wandb.log({"examples": wandb.Table(data=[
        [e["prompt"], e["js"], e["top1_direct"], e["top1_wrapped"]] for e in examples
    ], columns=["prompt", "js", "top1_direct", "top1_wrapped"])})

    print("\ndone")
    print("model:", args.model)
    print("device:", device)
    if js_scores:
        print("js_mean:", final["js_mean"])
        print("js_median:", final["js_median"])
    else:
        print("js: n/a (no logprobs)")
    print("top1_agreement:", top1_agree)

    run.finish()


if __name__ == "__main__":
    main()
