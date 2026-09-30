# Customer Service Fine-Tuning with Qwen2.5 & LoRA

Fine-tuning [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) using Parameter-Efficient Fine-Tuning (PEFT) and LoRA (Low-Rank Adaptation) on domain-specific customer service dialogue data.

This project demonstrates the end-to-end pipeline of preparing instruction data with Qwen's native chat template, training a lightweight LoRA adapter using Hugging Face's `trl` SFTTrainer, and evaluating the response quality against the base pre-trained model.

---

## Features

- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct` &mdash; lightweight, capable, and fast to fine-tune on consumer GPUs.
- **LoRA Adaptation**: Parameter-efficient training targeting attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`), minimizing memory footprint and training time.
- **Chat Template Formatting**: Native integration with Qwen tokenizer chat template formatting (`<|im_start|>` / `<|im_end|>`).
- **Supervised Fine-Tuning (SFT)**: Utilizes `trl.SFTTrainer` with mixed-precision (`fp16`) and gradient accumulation.
- **Side-by-Side Comparison**: Built-in evaluation script to benchmark baseline vs. fine-tuned adapter outputs across real customer service scenarios.

---

## Repository Structure

```text
.
├── data/
│   └── customer_service.json       # Customer service chat conversations dataset
├── prepare_data.py                 # Data loading and chat template verification
├── train_lora.py                   # LoRA fine-tuning script with SFTTrainer
├── test_model.py                   # Inference script for the base Qwen model
├── test_finetuned.py               # Inference script for the fine-tuned LoRA model
├── compare_models.py               # Side-by-side comparison between Base and LoRA models
├── .gitignore                      # Ignored files (checkpoints, venvs, cache)
└── README.md                       # Project documentation
```

---

## Requirements & Prerequisites

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (recommended: >= 6GB VRAM for fp16 training with batch size 1 + gradient accumulation).

### Software & Python Libraries

- Python 3.10+
- PyTorch (with CUDA support)
- Transformers
- Datasets
- PEFT
- TRL
- Accelerate

Install dependencies:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft trl accelerate
```

---

## Dataset Format

The dataset `data/customer_service.json` contains conversational dialogues in standard OpenAI/Qwen message format:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "My parcel still hasn't arrived."
      },
      {
        "role": "assistant",
        "content": "Please share your order number so we can check the shipment status."
      }
    ]
  }
]
```

---

## Quickstart & Workflow

### 1. Verify Dataset Formatting

Inspect how the Qwen chat template processes the raw conversation samples:

```bash
python prepare_data.py
```

### 2. Fine-Tune with LoRA

Train the LoRA adapter on 2,000 sampled customer service interactions:

```bash
python train_lora.py
```

The script will save the trained adapter weights and tokenizer to `customer_service_lora_2k/`.

### 3. Test Base Model

Test how the base un-tuned model responds to a customer inquiry:

```bash
python test_model.py
```

### 4. Test Fine-Tuned Model

Run inference with the base model combined with the LoRA adapter:

```bash
python test_finetuned.py
```

### 5. Compare Base vs. Fine-Tuned

Run side-by-side evaluation across standard customer support test questions (e.g., damaged items, refunds, missing accessories, tracking updates):

```bash
python compare_models.py
```

---

## Training Configuration & Hyperparameters

| Parameter                       | Value                                  |
| :------------------------------ | :------------------------------------- |
| **Base Model**                  | `Qwen/Qwen2.5-0.5B-Instruct`           |
| **Precision**                   | `fp16`                                 |
| **LoRA Rank ($r$)**             | `8`                                    |
| **LoRA Alpha ($\alpha$)**       | `16`                                   |
| **LoRA Dropout**                | `0.05`                                 |
| **Target Modules**              | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| **Epochs**                      | `1`                                    |
| **Per-Device Batch Size**       | `1`                                    |
| **Gradient Accumulation Steps** | `8` (Effective Batch Size = 8)         |
| **Learning Rate**               | `2e-4`                                 |
| **Max Sequence Length**         | `512`                                  |
| **Dataset Sample Size**         | `2,000`                                |
| **Output Directory**            | `customer_service_lora_2k`             |

---

## License

This project is licensed under the MIT License - see the LICENSE file for details if applicable.
