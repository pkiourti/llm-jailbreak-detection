import random

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients, GradientShap

from .judge import refused
from .model import encode_prompt, generate_ids, ids as encode_ids


class AttributionDefense:
    """MISA-style defense: attribute the first reply token to the user message, remove the REMOVE_K most-attributed
    tokens and regenerate. Flag the input if the model complied before and refuses after.

    `method`: "ig" (Integrated Gradients, a single pad-token baseline) or "eg" (Expected Gradients, Erion et al. 2021
    -- averages over random interpolation points AND a distribution of baselines drawn from `harmless_pool`, rather
    than one fixed baseline, as in MISA).
    `layer`: None attributes to the input embedding layer (token-level, the original behavior); an int k attributes
    to hidden_states[k] (same indexing as output_hidden_states and the activation probes), i.e. the output of
    `model.model.layers[k-1]`.

    EG can't reuse captum's LayerGradientShap: its NoiseTunnel machinery adds Gaussian noise to the top-level input
    before any layer-hooking happens, which only makes sense for continuous inputs -- not discrete token ids, at any
    layer. Instead we extract the hidden state at the target layer ourselves (via output_hidden_states, so HF handles
    position embeddings/masks correctly), perturb that directly with plain GradientShap, and splice it back in with a
    forward hook on the layer right before the target.
    """

    def __init__(self, model, tok, cfg, device, method="ig", layer=None, harmless_pool=None, eg_baselines=8, eg_samples=16):
        self.model, self.tok, self.cfg, self.device = model, tok, cfg, device
        self.method, self.layer_idx, self.harmless_pool = method, layer, harmless_pool
        self.eg_baselines, self.eg_samples = eg_baselines, eg_samples
        if method == "ig":
            target = model.get_input_embeddings() if layer is None else model.model.layers[layer]
            self.lig = LayerIntegratedGradients(self.next_token_logits, target)
        else:
            assert harmless_pool, "EG needs a harmless_pool of strings to sample baselines from"
            self.gs = GradientShap(self._forward_from_hidden)
            self._hook_module = model.get_input_embeddings() if layer is None else model.model.layers[layer - 1]

    def next_token_logits(self, input_ids):
        return self.model(input_ids=input_ids).logits[:, -1].float()

    def _hidden_at_layer(self, ids_t):
        """hidden_states[0] = embedding output; hidden_states[k] for k>=1 = output of model.model.layers[k-1]."""
        with torch.no_grad():
            out = self.model(input_ids=ids_t, output_hidden_states=True)
        idx = 0 if self.layer_idx is None else self.layer_idx
        return out.hidden_states[idx], out.logits[:, -1].float()

    def _forward_from_hidden(self, hidden_batch, ids_t):
        """Run the model with `_hook_module`'s output replaced by each row of hidden_batch in turn (captum's
        GradientShap may pass a batch of perturbed samples at once; the hook only patches one forward pass, so loop)."""
        outs = []
        for i in range(hidden_batch.shape[0]):
            h = hidden_batch[i:i + 1].to(self.model.dtype)   # captum upcasts to float32 for the noise/gradient math

            def hook(mod, inp, out, h=h):
                return (h,) + out[1:] if isinstance(out, tuple) else h

            handle = self._hook_module.register_forward_hook(hook)
            try:
                outs.append(self.model(input_ids=ids_t).logits[:, -1].float())
            finally:
                handle.remove()
        return torch.cat(outs, dim=0)

    def _pad_baseline(self, seq, s, e):
        """Single pad-token baseline: only the user message span is ablated, the chat template stays."""
        b = list(seq)
        b[s:e] = [self.tok.pad_token_id] * (e - s)
        return b

    def _eg_baseline_ids(self, seq, s, e):
        """A handful of harmless prompts, each truncated/pad-filled to the same span length, standing in for the
        baseline *distribution* EG integrates over (MISA used the eval set's images the same way)."""
        span = e - s
        baselines = []
        for h in random.sample(self.harmless_pool, min(self.eg_baselines, len(self.harmless_pool))):
            htoks = encode_ids(self.tok, h)[:span]
            htoks = htoks + [self.tok.pad_token_id] * (span - len(htoks))
            b = list(seq)
            b[s:e] = htoks
            baselines.append(b)
        return baselines

    def token_attributions(self, user):
        seq, (s, e) = encode_prompt(self.tok, user)
        ids_t = torch.tensor([seq], device=self.device)

        if self.method == "ig":
            with torch.no_grad():
                first = int(self.next_token_logits(ids_t).argmax(-1))
            baseline = torch.tensor([self._pad_baseline(seq, s, e)], device=self.device)
            attr = self.lig.attribute(inputs=ids_t, baselines=baseline, target=first,
                                       n_steps=self.cfg.IG_STEPS, internal_batch_size=self.cfg.IG_STEPS)
            scores = attr.sum(-1)[0, s:e].detach().float().cpu().numpy()
        else:
            input_hidden, logits = self._hidden_at_layer(ids_t)
            first = int(logits.argmax(-1))
            baseline_ids = self._eg_baseline_ids(seq, s, e)
            baseline_hiddens = torch.cat([self._hidden_at_layer(torch.tensor([b], device=self.device))[0]
                                           for b in baseline_ids], dim=0)
            attr = self.gs.attribute(inputs=input_hidden, baselines=baseline_hiddens, target=first,
                                      n_samples=self.eg_samples, additional_forward_args=(ids_t,))
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
