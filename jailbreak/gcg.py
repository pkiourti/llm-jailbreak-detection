"""Greedy Coordinate Gradient (Zou et al. 2023): optimize an adversarial suffix so the model starts its reply with the target."""
import time

import torch
import torch.nn.functional as F

from .model import ids, template_parts, respond


def allowed_tokens(tok, vocab_size, device):
    """Mask of tokens a suffix may use: printable ASCII, not special, not pure whitespace."""
    special = set(tok.all_special_ids)
    ok = torch.zeros(vocab_size, dtype=torch.bool)
    for i in range(min(len(tok), vocab_size)):
        if i in special:
            continue
        s = tok.decode([i])
        ok[i] = s.isascii() and s.isprintable() and bool(s.strip())
    return ok.to(device)


class GCG:
    def __init__(self, model, tok, cfg, judge, device):
        self.model, self.tok, self.cfg, self.judge, self.device = model, tok, cfg, judge, device
        self.embed = model.get_input_embeddings()
        self.allowed = allowed_tokens(tok, self.embed.weight.shape[0], device)
        before, after = template_parts(tok)
        self.before, self.after = ids(tok, before), ids(tok, after)

    def token_grad(self, prefix, suffix, rest, n_target):
        """Gradient of the target loss w.r.t. a one-hot encoding of the suffix tokens."""
        W = self.embed.weight
        one_hot = F.one_hot(suffix, W.shape[0]).to(W.dtype).requires_grad_()
        t = lambda x: torch.tensor(x, device=self.device)
        emb = torch.cat([self.embed(t(prefix)), one_hot @ W, self.embed(t(rest))])[None]
        logits = self.model(inputs_embeds=emb).logits[0]
        loss = F.cross_entropy(logits[-n_target-1:-1].float(), t(rest[-n_target:]))
        return torch.autograd.grad(loss, one_hot)[0]

    @torch.no_grad()
    def losses(self, prefix, cands, rest, n_target, bs=64):
        out = []
        for s in range(0, len(cands), bs):
            c = cands[s:s+bs]
            x = torch.cat([torch.tensor(prefix, device=self.device).expand(len(c), -1), c,
                           torch.tensor(rest, device=self.device).expand(len(c), -1)], dim=1)
            logits = self.model(input_ids=x).logits[:, -n_target-1:-1].float()
            out.append(F.cross_entropy(logits.transpose(1, 2), x[:, -n_target:], reduction="none").mean(1))
        return torch.cat(out)

    def roundtrips(self, goal_ids, cands):
        """Keep candidates whose user message re-tokenizes to exactly the same ids (so the attack transfers to text)."""
        texts = self.tok.batch_decode([goal_ids + c for c in cands.tolist()])
        enc = self.tok(texts, add_special_tokens=False).input_ids
        keep = [e == goal_ids + c for e, c in zip(enc, cands.tolist())]
        return cands[torch.tensor(keep, device=self.device)]

    def attack(self, goal, target):
        cfg, tok = self.cfg, self.tok
        goal_ids, target_ids = ids(tok, goal), ids(tok, target)
        suffix = torch.tensor(ids(tok, " !" * cfg.GCG_SUFFIX_LEN), device=self.device)
        prefix, rest = self.before + goal_ids, self.after + target_ids
        L, n_target = len(suffix), len(target_ids)
        loss, response, jailbroken, t0 = float("nan"), "", False, time.time()

        for step in range(1, cfg.GCG_STEPS + 1):
            grad = self.token_grad(prefix, suffix, rest, n_target)
            grad[:, ~self.allowed] = float("inf")
            top = (-grad).topk(cfg.GCG_TOPK, dim=1).indices                     # [L, K] most promising swaps per position
            pos = torch.randint(L, (cfg.GCG_BATCH,), device=self.device)
            new = top[pos, torch.randint(cfg.GCG_TOPK, (cfg.GCG_BATCH,), device=self.device)]
            cands = suffix.repeat(cfg.GCG_BATCH, 1)
            cands[torch.arange(cfg.GCG_BATCH), pos] = new
            cands = self.roundtrips(goal_ids, cands)
            if len(cands):
                losses = self.losses(prefix, cands, rest, n_target)
                i = int(losses.argmin())
                loss, suffix = float(losses[i]), cands[i]   # as in GCG, take the step's best candidate even if the loss rose

            if step % cfg.GCG_CHECK_EVERY == 0 or step == cfg.GCG_STEPS:
                prompt = tok.decode(goal_ids + suffix.tolist())
                response = respond(self.model, tok, [prompt], cfg.MAX_NEW_TOKENS, self.device)[0]
                jailbroken = self.judge.jailbroken([goal], [response])[0]
                print(f"  step {step:4d}  loss {loss:.3f}  jailbroken {jailbroken}  {response[:60]!r}")
                if jailbroken:
                    break

        prompt = tok.decode(goal_ids + suffix.tolist())
        return dict(attack="gcg", goal=goal, prompt=prompt, suffix=tok.decode(suffix.tolist()), suffix_tokens=L,
                    response=response, jailbroken=jailbroken, loss=loss, steps=step, seconds=time.time() - t0)
