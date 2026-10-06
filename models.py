"""
models.py — реестр моделей.

Для каждой модели: ключ запуска → имя на Hugging Face, квантизация,
короткое имя для графиков, семейство и размер в миллиардах параметров.
"""
MODELS = {
    # открытые
    "qwen-0.5b": {"name": "Qwen/Qwen2.5-0.5B-Instruct",
                  "short": "Qwen-0.5B", "family": "qwen", "size_b": 0.5},
    "qwen-1.5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct",
                  "short": "Qwen-1.5B", "family": "qwen", "size_b": 1.5},
    "qwen-3b": {"name": "Qwen/Qwen2.5-3B-Instruct",
                "short": "Qwen-3B", "family": "qwen", "size_b": 3.0},
    "qwen-7b": {"name": "Qwen/Qwen2.5-7B-Instruct",
                "short": "Qwen-7B", "family": "qwen", "size_b": 7.0},
    "mistral-7b": {"name": "mistralai/Mistral-7B-Instruct-v0.3",
                   "short": "Mistral-7B", "family": "mistral", "size_b": 7.0},
    "phi-3.5b": {"name": "microsoft/Phi-3.5-mini-instruct",
                 "short": "Phi-3.5", "family": "phi", "size_b": 3.8},
    # gated: нужны huggingface-cli login и принятие лицензии на странице модели
    "llama-3.2-1b": {"name": "meta-llama/Llama-3.2-1B-Instruct",
                     "short": "Llama-3.2-1B", "family": "llama", "size_b": 1.0},
    "llama-3.2-3b": {"name": "meta-llama/Llama-3.2-3B-Instruct",
                     "short": "Llama-3.2-3B", "family": "llama", "size_b": 3.0},
    "llama-8b": {"name": "meta-llama/Llama-3.1-8B-Instruct",
                 "short": "Llama-8B", "family": "llama", "size_b": 8.0},
    "gemma-2b": {"name": "google/gemma-2-2b-it",
                 "short": "Gemma-2B", "family": "gemma", "size_b": 2.0},
}

# HF-имя → запись реестра (отчёты хранят именно HF-имя)
BY_HF_NAME = {m["name"]: m for m in MODELS.values()}

# порядок моделей на графиках
PLOT_ORDER = ["Qwen-0.5B", "Qwen-1.5B", "Qwen-3B", "Qwen-7B",
              "Llama-3.2-1B", "Llama-3.2-3B", "Llama-8B",
              "Mistral-7B", "Phi-3.5", "Gemma-2B"]
