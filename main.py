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


def run_tokenizer_encode_example() -> None:
    """Simple example using tokenizer.encode() to understand tokenization."""
    tokenizer, _, _ = load_or_download_model()

    # Sample text to encode
    sample_text = "This sample helps me study and understand tokenization."

    print("\n=== Tokenizer Encode Example ===")
    print(f"Original text: '{sample_text}'")

    # Method 1: encode() - returns token IDs as a list
    token_ids = tokenizer.encode(sample_text)
    print(f"\nToken IDs (encode): {token_ids}")

    # Method 2: encode() with add_special_tokens=False
    token_ids_no_special = tokenizer.encode(sample_text, add_special_tokens=False)
    print(f"Token IDs without special tokens: {token_ids_no_special}")

    # Method 3: encode() with custom max_length and truncation
    # 增大max_length来允许更长的文本，例如从默认的512增加到1024
    token_ids_long = tokenizer.encode(
        sample_text,
        max_length=1024,  # 增大到1024，默认通常是512
        truncation=True,   # 启用truncation
        add_special_tokens=True
    )
    print(f"\nToken IDs with max_length=1024: {token_ids_long}")

    # Decode back to text
    decoded_text = tokenizer.decode(token_ids)
    print(f"\nDecoded text: '{decoded_text}'")

    # Convert to tokens (subwords)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    print(f"\nTokens: {tokens}")

    # Show each token with its ID
    print("\nToken-by-token breakdown:")
    for token, token_id in zip(tokens, token_ids):
        print(f"  '{token}' -> ID: {token_id}")

    # 演示一个很长的文本被truncate的情况
    long_text = " ".join(["This is a very long sentence."] * 50)
    print(f"\n=== Truncation Demo ===")
    print(f"Long text length: {len(long_text)} characters")

    # 使用较小的max_length
    token_ids_short = tokenizer.encode(long_text, max_length=20, truncation=True)
    print(f"\nWith max_length=20: {len(token_ids_short)} tokens")
    print(f"Truncated text: '{tokenizer.decode(token_ids_short)}'")

    # 使用较大的max_length
    token_ids_longer = tokenizer.encode(long_text, max_length=200, truncation=True)
    print(f"\nWith max_length=200: {len(token_ids_longer)} tokens")
    print(f"Less truncated text: '{tokenizer.decode(token_ids_longer)[:100]}...'")

    # 使用更大的max_length
    token_ids_longest = tokenizer.encode(long_text, max_length=512, truncation=True)
    print(f"\nWith max_length=512: {len(token_ids_longest)} tokens")

    # ===== Padding 示例 =====
    print("\n=== Padding Demo ===")
    print("注意：encode() 方法不支持padding！需要使用 tokenizer() 或 encode_plus()")

    # 两个不同长度的文本
    short_text = "Hello"
    medium_text = "This is a longer sentence with more words."

    # 方法1: 使用 encode() - 不会padding
    print("\n使用 encode() - 没有padding:")
    ids_short = tokenizer.encode(short_text, max_length=20, truncation=True)
    ids_medium = tokenizer.encode(medium_text, max_length=20, truncation=True)
    print(f"短文本 '{short_text}': {ids_short} (长度: {len(ids_short)})")
    print(f"中文本: {ids_medium} (长度: {len(ids_medium)})")
    print("👆 注意：两个文本长度不同，无法组成batch！")

    # 方法2: 使用 tokenizer() - 可以padding
    print("\n使用 tokenizer() - 有padding:")
    batch_result = tokenizer(
        [short_text, medium_text],
        max_length=20,
        padding='max_length',  # 填充到max_length
        truncation=True,
        return_tensors='pt'  # 返回PyTorch tensor
    )
    print(f"input_ids shape: {batch_result['input_ids'].shape}")
    print(f"短文本 token IDs: {batch_result['input_ids'][0].tolist()}")
    print(f"中文本 token IDs: {batch_result['input_ids'][1].tolist()}")
    print(f"attention_mask: \n{batch_result['attention_mask']}")
    print(f"👆 注意：短文本用 {tokenizer.pad_token_id} (PAD token) 填充到相同长度！")

    # 方法3: 使用 encode_plus() - 单个文本padding
    print("\n使用 encode_plus() - 单个文本padding:")
    encoded = tokenizer.encode_plus(
        short_text,
        max_length=15,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    print(f"短文本 '{short_text}' padding到15:")
    print(f"input_ids: {encoded['input_ids'][0].tolist()}")
    print(f"attention_mask: {encoded['attention_mask'][0].tolist()}")

    # 展示不同的padding策略
    print("\n不同的padding策略:")
    texts = ["Hi", "Hello world", "This is a test"]

    # longest: padding到batch中最长的序列
    batch_longest = tokenizer(texts, padding='longest', return_tensors='pt')
    print(f"\npadding='longest': shape {batch_longest['input_ids'].shape}")
    for i, text in enumerate(texts):
        print(f"  '{text}': {batch_longest['input_ids'][i].tolist()}")

    # max_length: padding到指定长度
    batch_max = tokenizer(texts, padding='max_length', max_length=10, truncation=True, return_tensors='pt')
    print(f"\npadding='max_length' (10): shape {batch_max['input_ids'].shape}")
    for i, text in enumerate(texts):
        print(f"  '{text}': {batch_max['input_ids'][i].tolist()}")


def compare_auto_vs_manual_padding() -> None:
    """
    对比 pipeline 自动 padding 和手动 tokenizer 的行为
    """
    tokenizer, model, model_dir = load_or_download_model()
    device = get_device()

    # 测试文本：长度明显不同
    texts = [
        "Good",  # 很短
        "This is a longer sentence with more words",  # 较长
    ]

    print("\n" + "="*60)
    print("对比：Pipeline 自动处理 vs 手动 Tokenizer")
    print("="*60)

    # ===== 1. Pipeline 自动处理 =====
    print("\n【方式1：使用 pipeline() - 自动padding】")
    classifier = pipeline(
        "sentiment-analysis",
        model=model_dir,
        tokenizer=tokenizer,
        device=device,
    )
    results = classifier(texts)
    print(f"✅ Pipeline 自动处理了不同长度的文本")
    for text, result in zip(texts, results):
        print(f"  '{text}' -> {result}")

    # ===== 2. 手动 tokenizer WITHOUT padding =====
    print("\n【方式2：手动 tokenizer 不带 padding - 会失败】")
    try:
        batch_no_padding = tokenizer(
            texts,
            truncation=True,
            return_tensors="pt",
            # 注意：没有 padding=True
        )
        print(f"❌ 这通常会失败，因为长度不一致")
        print(f"   input_ids shape: {batch_no_padding['input_ids'].shape}")
    except Exception as e:
        print(f"❌ 错误（预期）: {type(e).__name__}")
        print(f"   原因：文本长度不同，无法组成tensor batch")

    # ===== 3. 手动 tokenizer WITH padding =====
    print("\n【方式3：手动 tokenizer 带 padding=True】")
    batch_with_padding = tokenizer(
        texts,
        padding=True,  # 关键参数
        truncation=True,
        return_tensors="pt",
    )
    print(f"✅ 成功！input_ids shape: {batch_with_padding['input_ids'].shape}")
    print(f"\nPAD token ID: {tokenizer.pad_token_id}")
    print(f"PAD token: '{tokenizer.pad_token}'")

    for i, text in enumerate(texts):
        ids = batch_with_padding['input_ids'][i].tolist()
        mask = batch_with_padding['attention_mask'][i].tolist()
        print(f"\n文本 {i+1}: '{text}'")
        print(f"  token_ids:      {ids}")
        print(f"  attention_mask: {mask}")
        print(f"  padding count:  {ids.count(tokenizer.pad_token_id)} 个")

    # ===== 4. Pipeline 内部实际做了什么 =====
    print("\n" + "="*60)
    print("🔍 Pipeline 内部自动执行的步骤：")
    print("="*60)
    print("1. 自动调用 tokenizer(..., padding=True)")
    print("2. 自动将 tensors 移到正确的 device")
    print("3. 自动执行 model(**inputs)")
    print("4. 自动进行 post-processing")
    print("\n👉 所以使用 pipeline 时，你不需要手动 padding！")


if __name__ == "__main__":
    # list_supported_tasks()
    # run_sentiment_pipeline()
    # run_tokenizer_only()
    # run_tokenizer_encode_example()
    compare_auto_vs_manual_padding()
