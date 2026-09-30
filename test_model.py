import torch
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
)

model = model.to("cuda")

print("\nModel loaded successfully!")
print("Model:", MODEL_NAME)
print("Device:", model.device)
print("GPU:", torch.cuda.get_device_name(0))

# Test Question
messages = [
    {
        "role": "user",
        "content": "My order has not arrived yet. What should I do?",
    },
]

# Convert the conversation into Qwen's expected format
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt",).to("cuda")

# Generate answer
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
    )

# Remove the original prompt from the output
response = outputs[0][inputs["input_ids"].shape[1]:]

answer = tokenizer.decode(
    response,
    skip_special_tokens=True,
)

print("\nCustomer:")
print(messages[0]["content"])

print("\nQwen:")
print(answer)
