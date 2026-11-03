import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .models import resolve_model

PLACEHOLDER = "<<<USER_MESSAGE>>>"


def load_chat_model(name, dtype, device):
    hf_id = resolve_model(name)
    print(f"loading {hf_id} ({dtype})")
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=getattr(torch, dtype)).to(device).eval()
    model.requires_grad_(False)
    return tok, model


def ids(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def template_parts(tok):
    """Chat-template text before and after a single user message (system prompt, role markers, generation prompt)."""
    text = tok.apply_chat_template([{"role": "user", "content": PLACEHOLDER}], tokenize=False, add_generation_prompt=True)
    before, after = text.split(PLACEHOLDER)
    return before, after


def encode_prompt(tok, user):
    """Token ids of the chat prompt for one user message, plus the (start, end) token span of that message.

    The three pieces are tokenized separately so that a GCG suffix keeps exactly the tokens it was optimized as.
    """
    before, after = template_parts(tok)
    b, u, a = ids(tok, before), ids(tok, user), ids(tok, after)
    return b + u + a, (len(b), len(b) + len(u))


def left_pad(tok, seqs, device):
    n = max(map(len, seqs))
    input_ids = torch.tensor([[tok.pad_token_id] * (n - len(s)) + s for s in seqs], device=device)
    mask = torch.tensor([[0] * (n - len(s)) + [1] * len(s) for s in seqs], device=device)
    return input_ids, mask


@torch.no_grad()
def generate_ids(model, tok, seqs, max_new_tokens, device, bs=16, temperature=None):
    """Continue each token sequence; greedy unless a temperature is given."""
    outs = []
    sample = dict(do_sample=True, temperature=temperature, top_p=0.9) if temperature else dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    for s in range(0, len(seqs), bs):
        input_ids, mask = left_pad(tok, seqs[s:s+bs], device)
        out = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id, **sample)
        outs += tok.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def respond(model, tok, users, max_new_tokens, device):
    """Target model's greedy replies to single-turn user messages."""
    return generate_ids(model, tok, [encode_prompt(tok, u)[0] for u in users], max_new_tokens, device)


def chat(model, tok, conversations, max_new_tokens, device, prefills=None, temperature=None):
    """Reply to full message lists (system/user/assistant), optionally prefilling the start of each reply."""
    prefills = prefills or [""] * len(conversations)
    seqs = [ids(tok, tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) + p)
            for c, p in zip(conversations, prefills)]
    return [p + o for p, o in zip(prefills, generate_ids(model, tok, seqs, max_new_tokens, device, temperature=temperature))]
