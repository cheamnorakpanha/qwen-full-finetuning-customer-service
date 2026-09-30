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
)

base_model = base_model.to("cuda")

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model = model.to("cuda")

print("\nFine-tuned model loaded!")
print("Base model:", BASE_MODEL)
print("LoRA adapter:", ADAPTER_PATH)
print("GPU:", torch.cuda.get_device_name(0))


# New customer-service question
messages = [
    {
        "role": "user",
        "content": "I received my package, but one of the items is damaged. What should I do?"
    }
]

# Apply Qwen's chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    text,
    return_tensors="pt",
).to("cuda")


# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=True,
    )


# Remove the input tokens
response = outputs[0][inputs["input_ids"].shape[1]:]

answer = tokenizer.decode(
    response,
    skip_special_tokens=True,
)


print("\nCustomer:")
print(messages[0]["content"])

print("\nFine-tuned Qwen:")
print(answer)
