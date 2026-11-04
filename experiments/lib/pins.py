"""Every external version the experiments read, in one place: the models' weights, the datasets read from the Hugging Face Hub,
the sentiment prompts of Tigges et al., CausalGym's own code, and the software the paper's runs used. Imports nothing, so
the registry, the runner and the scorers can read it without torch."""

MODELS = {   # each model's weights: the Hugging Face Hub commit every run of the paper loaded
    "gpt2": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "EleutherAI/pythia-160m": "50f5173d932e8e61f858120bcb800b97af589f46",
    "EleutherAI/pythia-410m": "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
    "HuggingFaceTB/SmolLM2-360M": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "EleutherAI/pythia-1b": "f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2",
    "meta-llama/Llama-3.1-8B": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "google/gemma-2-9b": "33c193028431c2fde6c6e51f29e6f17b60cbfac6",
    "Qwen/Qwen2.5-7B": "d149729398750b98c0af14eb82c78cfe92750796",
    "EleutherAI/pythia-6.9b": "c0e3eee36dc47af0c49f361c74cfe459c09f7f23",
    "Qwen/Qwen2.5-1.5B": "8faed761d45a263340a0528343f099c05c9a4323",
    "meta-llama/Llama-3.2-1B": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    "meta-llama/Llama-3.2-3B": "13afe5124825b4f3751f836b40dafda64c1ed062",
    "Qwen/Qwen2.5-32B": "1818d35814b8319459f4bd55ed1ac8709630f003",
    "EleutherAI/pythia-1.4b": "fedc38a16eea3bd36a96b906d78d11d2ce18ed79",   # the sentiment re-reading only
}
CAUSALGYM = ("aryaman/causalgym", "95349c3a5e53e2506e8b212482ea6dd784978156")   # CausalGym's tasks (a Hub dataset)
KUMON = ("kumoryo9/shared-mech", "d57aeab6a73778023e65a9cfa7cf1eb77709ca62")     # Kumon & Yanaka's constructions (a Hub dataset)
TIGGES = ("https://github.com/curt-tigges/eliciting-latent-sentiment", "037cbc9c18867e03da14ceace612ab7e0f9449f7")
TIGGES_PROMPTS_SHA256 = "592b699a2cfa0d3a5f489bd6167e58a0d27239cd77e7252eb4bfc2f297bd2d7a"   # their prompts.yaml at that commit
CAUSALGYM_CODE = ("https://github.com/aryamanarora/causalgym", "0f3129f")   # CausalGym's released code, re-run as released
STACK = {"torch": "2.14.1+cu130", "transformers": "5.18.0"}   # the software of the paper's runs (recorded by every run)
