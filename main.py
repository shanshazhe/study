from pathlib import Path
from typing import Tuple, Union

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from transformers.pipelines import SUPPORTED_TASKS

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
LOCAL_MODEL_DIR = Path("saved_models") / MODEL_ID.replace("/", "_")


def get_device() -> Union[torch.device, int]:
    """
    Choose the best available device for inference.
    - CUDA takes priority when present.
    - Fall back to MPS on Apple Silicon.
    - Otherwise default to CPU.
    """
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_supported_tasks() -> None:
    print(f"Found {len(SUPPORTED_TASKS)} pipeline tasks:")
    for task in sorted(SUPPORTED_TASKS.keys()):
        print(f"- {task}")


def load_or_download_model() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, Path]:
    """
    Download the model/tokenizer once, save locally, and load from disk afterward.
    """
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if any(LOCAL_MODEL_DIR.iterdir()):
        print(f"Loading cached model from {LOCAL_MODEL_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
    else:
        print(f"Downloading {MODEL_ID} and saving to {LOCAL_MODEL_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        tokenizer.save_pretrained(LOCAL_MODEL_DIR)
        model.save_pretrained(LOCAL_MODEL_DIR)

    return tokenizer, model, LOCAL_MODEL_DIR


def run_sentiment_pipeline() -> None:
    """Lightweight end-to-end pipeline example."""
    device = get_device()
    tokenizer, model, model_dir = load_or_download_model()
    classifier = pipeline(
        "sentiment-analysis",
        model=model_dir,
        tokenizer=tokenizer,
        device=device,
    )
    texts = [
        "Hugging Face pipelines are convenient.",
        "Tokenization can be confusing at first.",
    ]
    print("\nPipeline results:")
    for text, result in zip(texts, classifier(texts)):
        print(f"'{text}' -> {result}")


def run_tokenizer_only() -> None:
    """Manual tokenizer + model usage without the high-level pipeline."""
    tokenizer, model, _ = load_or_download_model()

    batch = tokenizer(
        ["I love good documentation.", "The error messages are unclear."],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**batch)
        scores = outputs.logits.softmax(dim=-1)

    print(f"Tokenizer results:")
    label_names = model.config.id2label
    print(f"Label names:")
    print(label_names)
    print("\nTokenizer + model results:")
    for i, input_ids in enumerate(batch["input_ids"]):
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        label_scores = {label_names[j]: float(scores[i, j]) for j in range(scores.size(1))}
        print(f"'{text}' -> {label_scores}")


    print(scores.size(0))
    for i in range(scores.size(0)):
        print("predicted label for sample", i, ":", label_names[torch.argmax(scores[i]).item()])


if __name__ == "__main__":
    list_supported_tasks()
    run_sentiment_pipeline()
    run_tokenizer_only()
