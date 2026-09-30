# pyrefly: ignore [missing-import]
from datasets import load_dataset
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/customer_service.json"

# Load dataset
dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train",
)

print("Dataset loaded!")
print("Number of examples:", len(dataset))

# Load Qwen tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Apply Qwen chat template


def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


dataset = dataset.map(format_chat)

print("\nFormatted dataset!")
print("\nFirst formatted example:")
print(dataset[0]["text"])
