import torch
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer, AutoModelForCausalLM
# pyrefly: ignore [missing-import]
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "customer_service_lora_2k"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
).to("cuda")

print("Loading fine-tuned model...")
fine_tuned_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
).to("cuda")

fine_tuned_model = PeftModel.from_pretrained(
    fine_tuned_base,
    ADAPTER_PATH,
).to("cuda")

print("\nBoth models loaded successfully!\n")


questions = [
    "My package arrived, but one item is damaged. What should I do?",
    "I received the wrong product in my order. How can you help?",
    "I want to request a refund for something I purchased.",
    "My order is late and the tracking information hasn't changed.",
    "The product I ordered is missing an accessory. What should I do?",
]


def generate_answer(model, question):
    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    response = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        response,
        skip_special_tokens=True,
    ).strip()


for i, question in enumerate(questions, 1):

    print("=" * 80)
    print(f"QUESTION {i}")
    print("=" * 80)

    print("\nCustomer:")
    print(question)

    print("\n--- Base Qwen ---")
    base_answer = generate_answer(
        base_model,
        question,
    )
    print(base_answer)

    print("\n--- Qwen + LoRA ---")
    fine_tuned_answer = generate_answer(
        fine_tuned_model,
        question,
    )
    print(fine_tuned_answer)

    print()
