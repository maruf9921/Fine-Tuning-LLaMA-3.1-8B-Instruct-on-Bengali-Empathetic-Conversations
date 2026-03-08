<div align="center">

# 🦙 Fine-Tuning LLaMA 3.1-8B-Instruct
## on Bengali Empathetic Conversations

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Unsloth](https://img.shields.io/badge/Unsloth-Optimized-brightgreen)](https://github.com/unslothai/unsloth)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface)](https://huggingface.co/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/)

*Parameter-efficient fine-tuning of LLaMA 3.1-8B using QLoRA for empathetic, context-aware Bengali language responses.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Pipeline](#-pipeline)
- [Dataset](#-dataset)
- [Model Configuration](#-model-configuration)
- [Training Setup](#-training-setup)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Results](#-results)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Overview

This project fine-tunes **Meta's LLaMA 3.1-8B-Instruct** model on a curated **Bengali Empathetic Conversations Corpus** using **QLoRA** (Quantized Low-Rank Adaptation) for parameter-efficient training. The goal is to produce a Bengali language model capable of emotionally intelligent, empathetic responses — suitable for mental health support, counseling chatbots, and compassionate dialogue systems.

### Key Highlights

| Feature | Detail |
|---|---|
| **Base Model** | `unsloth/llama-3-8b-bnb-4bit` |
| **Fine-Tuning Method** | QLoRA (4-bit quantization + LoRA) |
| **Language** | Bengali (বাংলা) |
| **Domain** | Empathetic / Emotional Support Conversations |
| **Framework** | Unsloth + HuggingFace TRL |
| **Experiment Tracking** | SQLite |
| **Evaluation** | BLEU, ROUGE, Perplexity |

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[🦙 LLaMA 3.1-8B-Instruct\nPre-trained Base Model] --> B[4-bit Quantization\nbnb-4bit]
    B --> C[LoRA Adapter Injection\nr=16, alpha=16]

    C --> D1[q_proj]
    C --> D2[k_proj]
    C --> D3[v_proj]
    C --> D4[o_proj]
    C --> D5[gate_proj]
    C --> D6[up_proj]
    C --> D7[down_proj]

    D1 & D2 & D3 & D4 & D5 & D6 & D7 --> E[Fine-Tuned Model\nbengali_empathetic_llama3]

    style A fill:#4A90D9,color:#fff
    style B fill:#F5A623,color:#fff
    style C fill:#7ED321,color:#fff
    style E fill:#9B59B6,color:#fff
```

> **QLoRA** freezes the original 4-bit quantized weights and only trains small low-rank adapter matrices (~0.07% of total parameters), drastically reducing GPU memory usage while preserving model quality.

---

## 🔄 Pipeline

```mermaid
flowchart LR
    subgraph DATA["📦 Data Preparation"]
        A1[BengaliEmpathetic\nConversationsCorpus.csv] --> A2[Select Questions\n& Answers Columns]
        A2 --> A3[Drop NaN Rows]
        A3 --> A4[Rename Columns\ninput / response]
        A4 --> A5[Format Prompt\nInstruction Template]
        A5 --> A6[HuggingFace Dataset\n90% train / 10% test]
    end

    subgraph MODEL["🤖 Model Setup"]
        B1[Load LLaMA 3.1-8B\n4-bit Quantized] --> B2[Inject LoRA Adapters\nr=16, alpha=16]
        B2 --> B3[Gradient Checkpointing\nEnabled]
    end

    subgraph TRAIN["🏋️ Training"]
        C1[SFTTrainer\nSupervised Fine-Tuning] --> C2[AdamW 8-bit Optimizer]
        C2 --> C3[fp16 Mixed Precision]
        C3 --> C4[100 Steps\nBatch=1, Grad Accum=8]
        C4 --> C5[Save Adapter Weights]
    end

    subgraph EVAL["📊 Evaluation"]
        D1[BLEU Score] 
        D2[ROUGE Score]
        D3[Perplexity]
        D4[SQLite Experiment Log]
    end

    DATA --> MODEL --> TRAIN --> EVAL

    style DATA fill:#EBF5FB,stroke:#2E86C1
    style MODEL fill:#EAFAF1,stroke:#27AE60
    style TRAIN fill:#FEF9E7,stroke:#F39C12
    style EVAL fill:#F5EEF8,stroke:#8E44AD
```

---

## 📂 Dataset

The model is trained on the **Bengali Empathetic Conversations Corpus** — a dataset of emotionally rich question-answer pairs in the Bengali language.

```mermaid
pie title Dataset Split
    "Training Set (90%)" : 90
    "Test Set (10%)" : 10
```

### Prompt Format

Each training sample is formatted using the Alpaca-style instruction template:

```
### Instruction:
<Bengali question / user message>

### Response:
<Empathetic Bengali response>
```

### Example

```
### Instruction:
আমি ধূমপানে আসক্ত। আমি কিভাবে থামাতে পারি?

### Response:
তুমি একা নও। আমি তোমার পাশে আছি এবং বিশ্বাস করি তুমি পারবে।
```

---

## ⚙️ Model Configuration

### LoRA Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `r` | 16 | LoRA rank (adapter matrix dimension) |
| `lora_alpha` | 16 | LoRA scaling factor |
| `lora_dropout` | 0 | Dropout on LoRA layers |
| `bias` | `none` | No bias training |
| `use_gradient_checkpointing` | `True` | Saves GPU memory |

### Target Modules

```
q_proj  ·  k_proj  ·  v_proj  ·  o_proj
gate_proj  ·  up_proj  ·  down_proj
```

---

## 🏋️ Training Setup

| Parameter | Value |
|---|---|
| **Optimizer** | AdamW 8-bit |
| **Learning Rate** | `2e-4` |
| **Batch Size** | 1 (per device) |
| **Gradient Accumulation Steps** | 8 (effective batch = 8) |
| **Warmup Steps** | 5 |
| **Max Steps** | 100 |
| **Mixed Precision** | FP16 |
| **Max Sequence Length** | 2048 |
| **Output Directory** | `outputs/` |

---

## 📊 Evaluation

The fine-tuned model is evaluated using three complementary metrics:

```mermaid
flowchart LR
    M[Fine-Tuned Model\nInference] --> B[BLEU Score\nN-gram Precision]
    M --> R[ROUGE Score\nRecall-Oriented\nR1 · R2 · RL]
    M --> P[Perplexity\ne^eval_loss]
    B & R & P --> L[(SQLite\nExperiment Log)]

    style M fill:#2C3E50,color:#fff
    style B fill:#1ABC9C,color:#fff
    style R fill:#3498DB,color:#fff
    style P fill:#E74C3C,color:#fff
    style L fill:#8E44AD,color:#fff
```

### Experiment Tracking (SQLite)

All runs are recorded in a local `experiments.db` database with the following schema:

```sql
CREATE TABLE LLAMAExperiments (
    id           INTEGER PRIMARY KEY,
    model_name   TEXT,
    lora_config  TEXT,
    train_loss   REAL,
    val_loss     REAL,
    metrics      TEXT,
    timestamp    TEXT
);
```

---

## 🗂️ Project Structure

```
📁 Fine-Tuning-LLaMA-3.1-8B-Instruct-on-Bengali-Empathetic-Conversations/
│
├── 📓 Untitled1.ipynb                          # Main training notebook
├── 📄 BengaliEmpatheticConversationsCorpus.csv # Training dataset
├── 📁 bengali_empathetic_llama3/               # Saved fine-tuned adapter weights
├── 📁 outputs/                                 # Training checkpoints
├── 🗄️ experiments.db                           # SQLite experiment logs
└── 📄 README.md                                # Project documentation
```

---

## 🚀 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Fine-Tuning-LLaMA-3.1-8B-Instruct-on-Bengali-Empathetic-Conversations.git
cd Fine-Tuning-LLaMA-3.1-8B-Instruct-on-Bengali-Empathetic-Conversations
```

### 2. Install Dependencies

```bash
pip install unsloth
pip install transformers datasets accelerate peft trl
pip install evaluate rouge_score nltk
```

### 3. Run on Google Colab

> Recommended: **Google Colab with T4/A100 GPU** for 4-bit QLoRA training.

Upload `BengaliEmpatheticConversationsCorpus.csv` when prompted, then run all cells in `Untitled1.ipynb`.

### 4. Run Inference

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("bengali_empathetic_llama3")
FastLanguageModel.for_inference(model)

prompt = """### Instruction:
আমি অনেক চাপে আছি। কি করব?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=120)
print(tokenizer.decode(outputs[0]))
```

---

## 📈 Results

| Metric | Value |
|---|---|
| **Training Loss** | ~1.2 |
| **Validation Loss** | ~1.5 |
| **Perplexity** | ~4.48 |
| **BLEU** | Computed per evaluation run |
| **ROUGE-L** | Computed per evaluation run |

> Results are logged to `experiments.db` for reproducibility and comparison across runs.

---

## 📎 Citation

If you use this work, please cite:

```bibtex
@misc{bengali-empathetic-llama3,
  author       = {Abdullah Al Maruf},
  title        = {Fine-Tuning LLaMA 3.1-8B-Instruct on Bengali Empathetic Conversations},
  year         = {2026},
  howpublished = {\url{https://github.com/<your-username>/Fine-Tuning-LLaMA-3.1-8B-Instruct-on-Bengali-Empathetic-Conversations}},
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Made with ❤️ for the Bengali NLP community

</div>

