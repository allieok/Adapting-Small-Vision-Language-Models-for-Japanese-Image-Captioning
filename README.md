# 🏮 Adapting Small Vision–Language Models for Japanese Image Captioning

> **APS360 Project (Fall 2025)**  
> *University of Toronto — Applied Deep Learning*  
>  
> Team: Allie Okumura (1009846419)  
> Instructor: Prof. [Name]  
> Model: **Qwen2-VL-2B-Instruct** fine-tuned via **QLoRA** on **STAIR Captions**  
> Baselines: LLaVA-JP-1.3B · Asagi-2B · PaliGemma-3B · Japanese-Stable-VLM  

---

## 📘 Project Overview
This project investigates how efficiently **small vision–language models (VLMs)** can be adapted for **Japanese image captioning** using **parameter-efficient fine-tuning (PEFT)**.  
We aim to bridge the performance gap between large multilingual models and compact, cost-effective ones for domain-specific captioning.

---

## 🎯 Objectives
1. **Fine-tune** Qwen2-VL-2B-Instruct on the **STAIR Captions** dataset using **QLoRA**.  
2. **Evaluate** zero-shot and fine-tuned performance on:
   - STAIR validation/test (in-domain)
   - YJ Captions (out-of-domain)  
3. **Compare** with open Japanese baselines:
   - LLaVA-JP-1.3B, Asagi-2B, Japanese-Stable-VLM, PaliGemma-3B  
4. **Analyze** ablations: LoRA rank, layer depth, and projector freezing.  
5. **Discuss** ethical and linguistic failure cases.

---

## 🧠 Key Idea
Use **QLoRA (Quantized Low-Rank Adaptation)** to efficiently adapt only a small subset of model parameters under limited GPU memory (T4).  
This preserves performance while drastically reducing compute and storage requirements.

---

## 🧩 Repository Structure

```
📦 adapting-small-vlm-jp-captioning
│
├── 📁 data/
│   ├── raw/           # Original datasets (STAIR, YJ)
│   └── processed/     # Cleaned + tokenized splits
│
├── 📁 src/
│   ├── train_qlora.py       # Fine-tuning script
│   ├── model_utils.py       # Model loading + PEFT setup
│   └── eval_captioning.py   # Metric computation (BLEU, CIDEr, ROUGE)
│
├── 📁 scripts/
│   ├── preprocess_jp.py     # Japanese text cleaning & tokenization
│   ├── infer_baseline.py    # Zero-shot inference for baseline models
│   └── compare_runs.py      # Collects and visualizes results
│
├── 📁 configs/
│   ├── base.yaml            # Training hyperparameters
│   └── lora_ablation.yaml   # Alternative settings for experiments
│
├── 📁 experiments/
│   ├── baselines/           # Zero-shot results
│   └── finetuned/           # QLoRA fine-tuning runs
│
├── 📁 reports/
│   ├── metrics.csv          # Quantitative results
│   └── visuals/             # Plots and qualitative samples
│
├── requirements.txt
├── README.md
└── RUNBOOK.md               # Step-by-step experiment guide
```

---

## ⚙️ Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/allieok/adapting-small-vlm-jp-captioning.git
cd adapting-small-vlm-jp-captioning
```

### 2. Create Environment
```bash
conda create -n vlmjp python=3.10
conda activate vlmjp
pip install -r requirements.txt
```

### 3. Install Core Dependencies
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft bitsandbytes datasets evaluate sentencepiece
```

---

## 📊 Datasets

| Dataset | Domain | Size | Language | Usage |
|----------|---------|------|-----------|--------|
| **STAIR Captions** | Flickr | ~820k captions | Japanese | Train / Val / Test |
| **YJ Captions** | News, General | 26k captions | Japanese | Out-of-domain test |

Preprocessing steps:
- Normalize punctuation and spacing.  
- Convert zenkaku → hankaku characters.  
- Remove emojis / control tokens.  
- Truncate long captions to max 64 tokens.  

Processed splits stored in `data/processed/`.

---

## 🚀 Training QLoRA Adapter

### Command
```bash
python src/train_qlora.py --config configs/base.yaml
```

### Key Settings
| Parameter | Value |
|------------|--------|
| Model | Qwen2-VL-2B-Instruct |
| Trainable parts | Projector + top 6 transformer layers |
| LoRA rank / α / dropout | 32 / 32 / 0.05 |
| Optimizer | AdamW + cosine decay |
| Precision | bfloat16 |
| Batch size | 16 |
| Epochs | 3–5 (T4 compatible) |

---

## 🧪 Evaluation

### Zero-Shot Baselines
```bash
python scripts/infer_baseline.py --model llava-jp-1.3b --split stair_val
```

### Fine-Tuned Model
```bash
python scripts/eval_captioning.py   --pred experiments/finetuned/pred.jsonl   --ref data/processed/stair_val_refs.jsonl
```

### Metrics
| Metric | Description |
|---------|--------------|
| BLEU-4 | n-gram overlap |
| ROUGE-L | Longest common subsequence |
| CIDEr | Consensus-based image captioning metric |
| SPICE (optional) | Semantic content match |

---

## 📈 Results Summary (Example Format)

| Model | Params | BLEU-4 | ROUGE-L | CIDEr | GPU (VRAM) |
|--------|---------|--------|----------|--------|-------------|
| LLaVA-JP-1.3B | 1.3B | 0.21 | 0.41 | 0.56 | 12 GB |
| Asagi-2B | 2.0B | 0.23 | 0.43 | 0.61 | 14 GB |
| Qwen2-VL-2B (zero-shot) | 2.2B | 0.25 | 0.45 | 0.63 | 15 GB |
| **Qwen2-VL-2B + QLoRA (ours)** | 2.2B + LoRA | **0.32** | **0.51** | **0.82** | 15 GB |

---

## ⚖️ Ethical Considerations
- **Hallucination:** Occasional non-existent objects or artifacts.
- **Cultural bias:** Overgeneralized or gendered captions.
- **Privacy risk:** Avoid real-person identification.
- **Mitigation:** Human-in-loop filtering, constrained decoding.

---

## 📜 Citation
If referencing this project:
```bibtex
@misc{okumura2025vlmjp,
  author = {Okumura, Allie},
  title = {Adapting Small Vision–Language Models for Japanese Image Captioning},
  year = {2025},
  institution = {University of Toronto, APS360},
}
```

---

## 🧩 Future Work
- Expand to multilingual captioning (English–Japanese parallel data).
- Explore instruction-tuned adapters (e.g., Visual-Instruction-LoRA).
- Integrate human preference ranking for quality assessment.

---

## 🙌 Acknowledgments
- STAIR Captions Dataset — Kyoto University  
- YJ Captions Dataset — Yahoo! Japan Research  
- Qwen2-VL Team (Alibaba Cloud)  
- Hugging Face `transformers` and `peft` maintainers  
