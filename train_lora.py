import torch

# pyrefly: ignore [missing-import]
from datasets import load_dataset
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer, AutoModelForCausalLM
# pyrefly: ignore [missing-import]
from peft import LoraConfig
# pyrefly: ignore [missing-import]
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/customer_service.json"
OUTPUT_DIR = "customer_service_lora_2k"


print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train",
)

print("Examples:", len(dataset))


print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


print("Formatting dataset...")


def format_chat(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


dataset = dataset.map(format_chat)

dataset = dataset.shuffle(seed=42).select(range(2000))

print("Training examples:", len(dataset))


print("Dataset formatted!")


print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
)

model = model.to("cuda")

print("Model loaded!")
print("GPU:", torch.cuda.get_device_name(0))


print("\nConfiguring LoRA...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

print("LoRA configured!")
print("Rank:", lora_config.r)
print("Alpha:", lora_config.lora_alpha)
print("Target modules:", lora_config.target_modules)


print("\nConfiguring training...")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=1,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    logging_steps=10,

    save_steps=500,

    fp16=True,

    max_length=512,

    report_to="none",
)

print("Training configuration ready!")

print("\nStarting LoRA training...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()

print("\nTraining complete!")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA adapter saved to:", OUTPUT_DIR)
