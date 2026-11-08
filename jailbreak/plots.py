import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_attr(defense, user, path, suffix_tokens=0):
    """Bar chart of per-token attributions over the user message; adversarial-suffix tokens in red."""
    _, _, first, tokens, scores = defense.token_attributions(user)
    n = len(tokens)
    colors = ["tab:red" if i >= n - suffix_tokens else "tab:blue" for i in range(n)] if suffix_tokens else "tab:blue"
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * n), 3))
    ax.bar(range(n), scores, color=colors)
    ax.set_xticks(range(n))
    # parse_math=False: a GCG suffix is unconstrained printable ASCII and can contain "$", which matplotlib
    # otherwise tries to parse as LaTeX math -- these are raw token strings, never meant as math.
    ax.set_xticklabels([repr(t)[1:-1] for t in tokens], rotation=90, fontsize=7, parse_math=False)
    ax.set_ylabel("IG attribution"); ax.set_title(f"attribution of first reply token {defense.tok.decode([first])!r}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"saved {path}")


def plot_projections(groups, path):
    """Scatter of harmfulness vs refusal projections; groups maps a label to (harm, refusal) arrays."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for label, (h, r) in groups.items():
        if len(h):
            ax.scatter(h, r, s=14, alpha=0.7, label=f"{label} (n={len(h)})")
    ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("harmfulness projection (> 0 flagged)"); ax.set_ylabel("refusal projection (> 0 refusing)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"saved {path}")
