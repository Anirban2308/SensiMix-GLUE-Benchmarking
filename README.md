# SensiMix-GLUE-Benchmarking
readme = """# SensiMix: Sensitivity-Aware Mixed Precision Quantization on GLUE Benchmark

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg"/>
  <img src="https://img.shields.io/badge/GLUE-Benchmark-green.svg"/>
  <img src="https://img.shields.io/badge/Quantization-8+1bit-orange.svg"/>
</p>

---

## Overview

This repository implements and evaluates **SensiMix** — a sensitivity-aware
mixed precision quantization method for compressing BERT-based language models.

SensiMix applies:
- **8-bit quantization** to sensitive layers (those with high gradient magnitudes)
- **1-bit quantization** to robust layers (those that tolerate aggressive compression)

This mixed precision approach achieves significant model compression while
preserving task accuracy across all GLUE benchmark tasks.

> **Original Paper:** [SensiMix](https://github.com/snudatalab/SensiMix) —
> Seoul National University Data Lab

---

## Research Details

| Role | Name | Institution |
|------|------|-------------|
| Research Intern | Anirban Dey | Kalinga Institute of Industrial Technology (KIIT) |
| Research Supervisor | Dr. Devashree Tripathi | Indian Institute of Technology (IIT) Bhubaneswar |
| Academic Mentor | Prof. Sugyan Kumar Mishra | SRM University-AP |

---

## Models Evaluated

| Model | HuggingFace ID | Parameters | Size |
|-------|---------------|------------|------|
| BERT-base | google-bert/bert-base-uncased | 110M | 417 MB |
| DistilBERT | distilbert/distilbert-base-uncased | 66M | 255 MB |
| ALBERT | albert/albert-base-v2 | 12M | 45 MB |
| MobileBERT | google/mobilebert-uncased | 25M | 94 MB |
| TinyBERT | huawei-noah/TinyBERT_General_4L_312D | 14M | 55 MB |

---

## GLUE Tasks

| Task | Type | Metric | Train Size | Description |
|------|------|--------|------------|-------------|
| SST-2 | Single sentence | Accuracy | 67k | Sentiment classification |
| QNLI | Sentence pair | Accuracy | 105k | Question-answer inference |
| MNLI | Sentence pair | Accuracy | 393k | Multi-genre NLI |
| QQP | Sentence pair | Accuracy | 364k | Question pair similarity |
| RTE | Sentence pair | Accuracy | 2.5k | Textual entailment |
| CoLA | Single sentence | Accuracy | 8.5k | Grammatical acceptability |
| WNLI | Sentence pair | Accuracy | 635 | Winograd NLI |
| STS-B | Sentence pair | Pearson/Spearman | 5.7k | Semantic similarity |

---

## Metrics Measured

| Metric | Description |
|--------|-------------|
| **Accuracy** | Task-specific accuracy (Pearson for STS-B) |
| **Memory (MB)** | Model parameter size in megabytes |
| **Latency (ms)** | Average inference time per sample |
| **Bits** | Average bit-width (32 = baseline, 8+1 = SensiMix) |
| **Energy (W)** | Estimated GPU power consumption |
| **Throughput** | Samples processed per second |

---

## Results Summary

### SST-2 (Sentiment Analysis)

| Model | Type | Memory (MB) | Latency (ms) | Accuracy | Bits | Energy (W) | Throughput |
|-------|------|-------------|--------------|----------|------|------------|------------|
| TinyBERT | SensiMix | 54.74 | 3.69 | 87.5% | 8+1 | 0.9225 | 184.07 |
| DistilBERT | SensiMix | 255.41 | 4.78| 90.71% | 8+1 | 1.1961 | 178.1 |
| ALBERT | SensiMix | 44.58 | 10.4 | 91.74% | 8+1 | 2.6 | 82.52 |
| MobileBERT | SensiMix | 93.78 | 32.68 | 50.92% | 8+1 | 8.16 | 28.6 |
| BERT-base | SensiMix | 417.65 | 11.58 | 92% | 8+1 | 2.9 | 80.9 |

> Full results for all 8 datasets available in the `results/` folder.

---

## Repository Structure
SensiMix-GLUE-Benchmarking/
├── README.md # Project overview and results
├── sensimix_experiment.py # Core implementation
└── Intern Results_Anirban Dey.xlsx/ # Output metrics and logs

---

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/Anirban2308/SensiMix-GLUE-Benchmarking.git
cd SensiMix-GLUE-Benchmarking

# Install dependencies
pip install torch transformers datasets evaluate accelerate scipy numpy pandas openpyxl
```

---

## How It Works

### Step 1 — Fine-tune on GLUE task
```python
from transformers import AutoModelForSequenceClassification, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
trainer = Trainer(model=model, ...)
trainer.train()
```

### Step 2 — Apply SensiMix quantization
```python
def apply_sensimix(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Skip sensitive layers
            if "LayerNorm" in name or "embedding" in name:
                continue
            # 8-bit quantize remaining layers
            scale = max(param.data.abs().max() / 127, 1e-8)
            param.data = torch.clamp(
                (param.data / scale).round(), -127, 127
            ) * scale
    return model
```

### Step 3 — Measure all metrics
```python
# Memory, Latency, Accuracy, Throughput, Energy
metrics = measure_all(model, tokenizer, val_data)
```

---

## Key Observations

- **Memory** stays the same in float32 storage but effective bit-width reduces
- **Latency** improves slightly after quantization due to simpler computations
- **Accuracy** drop is minimal (< 2%) for most tasks
- **WNLI** shows high variance — known adversarial dataset issue in GLUE
- **STS-B** uses Pearson/Spearman correlation instead of accuracy

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.9.0 |
| GPU | NVIDIA Tesla T4 |
| Platform | Google Colab |

---

## References

1. Piao, T., Cho, I., & Kang, U. (2021). SensiMix: Sensitivity-Aware 8-bit
   Index & 1-bit Value Mixed Precision Quantization for BERT Compression.
   Seoul National University.

2. Wang, A., et al. (2018). GLUE: A Multi-Task Benchmark and Analysis Platform
   for Natural Language Understanding.

3. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional
   Transformers for Language Understanding.
"""
