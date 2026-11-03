import numpy as np
import torch
from captum.attr import LayerIntegratedGradients

from .judge import refused
from .model import encode_prompt, generate_ids


class AttributionDefense:
    """MISA-style defense: attribute the first reply token to the user message with Integrated Gradients, remove the
    REMOVE_K most-attributed tokens and regenerate. Flag the input if the model complied before and refuses after."""

    def __init__(self, model, tok, cfg, device):
        self.model, self.tok, self.cfg, self.device = model, tok, cfg, device
        self.lig = LayerIntegratedGradients(self.next_token_logits, model.get_input_embeddings())

    def next_token_logits(self, input_ids):
        return self.model(input_ids=input_ids).logits[:, -1].float()

    def token_attributions(self, user):
        seq, (s, e) = encode_prompt(self.tok, user)
        ids = torch.tensor([seq], device=self.device)
        with torch.no_grad():
            first = int(self.next_token_logits(ids).argmax(-1))
        baseline = ids.clone()
        baseline[0, s:e] = self.tok.pad_token_id   # only the user message is ablated; the chat template stays
        attr = self.lig.attribute(inputs=ids, baselines=baseline, target=first,
                                  n_steps=self.cfg.IG_STEPS, internal_batch_size=self.cfg.IG_STEPS)
        scores = attr.sum(-1)[0, s:e].detach().float().cpu().numpy()
        tokens = [self.tok.decode([t]) for t in seq[s:e]]
        return seq, (s, e), first, tokens, scores

    def defend(self, user, response=None, suffix_tokens=0):
        """`response` is the model's reply to `user` if already known; `suffix_tokens` marks the last n user tokens as
        an adversarial suffix, for localization."""
        seq, (s, e), first, tokens, scores = self.token_attributions(user)
        if response is None:
            response = generate_ids(self.model, self.tok, [seq], self.cfg.MAX_NEW_TOKENS, self.device)[0]
        top = np.argsort(-scores)[:min(self.cfg.REMOVE_K, e - s - 1)]
        drop = {s + int(i) for i in top}
        kept = [t for j, t in enumerate(seq) if j not in drop]
        after = generate_ids(self.model, self.tok, [kept], self.cfg.MAX_NEW_TOKENS, self.device)[0]
        complied_before, complied_after = not refused(response), not refused(after)
        in_suffix = [int(i) >= (e - s) - suffix_tokens for i in top] if suffix_tokens else []
        return dict(flagged=complied_before and not complied_after, complied_before=complied_before,
                    complied_after=complied_after, removed=[tokens[int(i)] for i in top],
                    removed_in_suffix=float(np.mean(in_suffix)) if in_suffix else float("nan"),
                    first_token=self.tok.decode([first]), response_after=after)


class ActivationProbes:
    """Two difference-of-means directions fit on plain harmful vs harmless prompts (Arditi et al. 2024; Zhao et al. 2025):
    - harmfulness, read at the last token of the user message: does the model still represent the request as harmful?
    - refusal, read at the last prompt token (end of the chat template): is the model about to refuse?
    A jailbreak that works should suppress refusal; the harmfulness probe is the detector."""

    def __init__(self, model, tok, cfg, device):
        self.model, self.tok, self.device = model, tok, device
        n_layers = model.config.num_hidden_layers
        self.layer = cfg.PROBE_LAYER if cfg.PROBE_LAYER is not None else n_layers // 2

    @torch.no_grad()
    def activations(self, users):
        harm, refusal = [], []
        for u in users:
            seq, (s, e) = encode_prompt(self.tok, u)
            h = self.model(input_ids=torch.tensor([seq], device=self.device), output_hidden_states=True).hidden_states[self.layer][0]
            harm.append(h[e - 1].float().cpu().numpy())
            refusal.append(h[-1].float().cpu().numpy())
        return np.stack(harm), np.stack(refusal)

    @staticmethod
    def direction(pos, neg):
        d = pos.mean(0) - neg.mean(0)
        d /= np.linalg.norm(d)
        threshold = ((pos @ d).mean() + (neg @ d).mean()) / 2   # midpoint of the two class means
        return d, threshold

    def fit(self, harmful, harmless):
        (hh, hr), (nh, nr) = self.activations(harmful), self.activations(harmless)
        self.harm_dir, self.harm_thr = self.direction(hh, nh)
        self.refusal_dir, self.refusal_thr = self.direction(hr, nr)
        train_acc = np.mean(np.r_[hh @ self.harm_dir > self.harm_thr, nh @ self.harm_dir <= self.harm_thr])
        print(f"probes at layer {self.layer}: harmfulness probe train accuracy {train_acc:.1%}")

    def project(self, users):
        """Per prompt: harmfulness and refusal projections, each relative to its threshold (> 0 = harmful / refusing)."""
        h, r = self.activations(users)
        return h @ self.harm_dir - self.harm_thr, r @ self.refusal_dir - self.refusal_thr
