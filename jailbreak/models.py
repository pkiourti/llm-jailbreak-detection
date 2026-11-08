"""Chat-model presets: short name -> Hugging Face id. Any other Hugging Face id with a chat template also works."""

MODELS = {
    "qwen2.5-0.5b-instruct": dict(id="Qwen/Qwen2.5-0.5B-Instruct"),
    "qwen2.5-1.5b-instruct": dict(id="Qwen/Qwen2.5-1.5B-Instruct"),
    "qwen2.5-3b-instruct":   dict(id="Qwen/Qwen2.5-3B-Instruct"),
    "phi-3.5-mini-instruct": dict(id="microsoft/Phi-3.5-mini-instruct"),
    "smollm2-360m-instruct": dict(id="HuggingFaceTB/SmolLM2-360M-Instruct"),
    "smollm2-1.7b-instruct": dict(id="HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    "llama3.2-1b-instruct":  dict(id="meta-llama/Llama-3.2-1B-Instruct", gated=True),
    "llama3.2-3b-instruct":  dict(id="meta-llama/Llama-3.2-3B-Instruct", gated=True),
    "gemma2-2b-it":          dict(id="google/gemma-2-2b-it",             gated=True),
}


def resolve_model(name):
    return MODELS[name]["id"] if name in MODELS else name


def model_slug(hf_id):
    for key, preset in MODELS.items():
        if preset["id"] == hf_id:
            return key
    return hf_id.split("/")[-1].lower()
