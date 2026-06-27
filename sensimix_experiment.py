script = '''"""
SensiMix: Sensitivity-Aware Mixed Precision Quantization
Implementation for GLUE Benchmark Evaluation

Models: BERT-base, DistilBERT, ALBERT, MobileBERT, TinyBERT
Datasets: SST-2, QNLI, MNLI, QQP, RTE, CoLA, WNLI, STS-B

Research Intern: Anirban Dey

Research Supervisor: Dr. Devashree Tripathi
Indian Institute of Technology (IIT) Bhubaneswar

Mentor: Prof. Sugyan Kumar Mishra
SRM University-AP
"""

import torch
import time
import os
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = {
    "TinyBERT":   "huawei-noah/TinyBERT_General_4L_312D",
    "DistilBERT": "distilbert/distilbert-base-uncased",
    "ALBERT":     "albert/albert-base-v2",
    "MobileBERT": "google/mobilebert-uncased",
    "BERT-base":  "google-bert/bert-base-uncased",
}

DATASETS = {
    "SST-2": {"hf_name": "stanfordnlp/sst2",  "subset": None,   "type": "single"},
    "QNLI":  {"hf_name": "nyu-mll/glue",      "subset": "qnli", "type": "pair"},
    "MNLI":  {"hf_name": "nyu-mll/glue",      "subset": "mnli", "type": "pair"},
    "QQP":   {"hf_name": "nyu-mll/glue",      "subset": "qqp",  "type": "pair"},
    "RTE":   {"hf_name": "nyu-mll/glue",      "subset": "rte",  "type": "pair"},
    "CoLA":  {"hf_name": "nyu-mll/glue",      "subset": "cola", "type": "single"},
    "WNLI":  {"hf_name": "nyu-mll/glue",      "subset": "wnli", "type": "pair"},
    "STS-B": {"hf_name": "nyu-mll/glue",      "subset": "stsb", "type": "regression"},
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset Class ─────────────────────────────────────────────────────────────

class GlueDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ── SensiMix Quantization ─────────────────────────────────────────────────────

def quantize_8bit(x):
    """8-bit symmetric quantization."""
    scale = max(x.abs().max() / 127, 1e-8)
    return torch.clamp((x / scale).round(), -127, 127) * scale


def apply_sensimix(model):
    """
    Apply SensiMix quantization:
    - Skip LayerNorm and embedding layers (always sensitive)
    - Apply 8-bit quantization to all other layers
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "LayerNorm" in name or "embedding" in name:
                continue
            param.data = quantize_8bit(param.data)
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────

def get_memory_mb(model):
    return sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 ** 2)


def measure_latency_ms(model, tokenizer, text_a, text_b=None):
    if text_b:
        sample = tokenizer(text_a, text_b, return_tensors="pt",
                           padding="max_length", truncation=True, max_length=128)
    else:
        sample = tokenizer(text_a, return_tensors="pt",
                           padding="max_length", truncation=True, max_length=128)
    sample = {k: v.to(device) for k, v in sample.items()}
    for _ in range(10):
        with torch.no_grad(): model(**sample)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        with torch.no_grad(): model(**sample)
    torch.cuda.synchronize()
    return (time.time() - t0) / 100 * 1000


def evaluate_accuracy(model, tokenizer, texts_a, texts_b, labels, n=2000):
    correct = 0
    t0 = time.time()
    for i, (ta, label) in enumerate(zip(texts_a[:n], labels[:n])):
        tb = texts_b[i] if texts_b else None
        if tb:
            inp = tokenizer(ta, tb, return_tensors="pt", truncation=True,
                            padding="max_length", max_length=128)
        else:
            inp = tokenizer(ta, return_tensors="pt", truncation=True,
                            padding="max_length", max_length=128)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad(): out = model(**inp)
        logits = out[0] if isinstance(out, tuple) else out.logits
        correct += (logits.argmax(-1).item() == label)
    accuracy   = correct / min(n, len(labels))
    throughput = min(n, len(labels)) / (time.time() - t0)
    return accuracy, throughput


def estimate_energy(latency_ms, power_w=250):
    return (latency_ms / 1000) * power_w


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def finetune(model_name, model_id, task_name,
             train_a, train_b, train_labels,
             val_a, val_b, val_labels,
             num_labels, save_root, epochs=3):

    save_dir = f"{save_root}/{model_name}_{task_name}"
    if os.path.exists(f"{save_dir}/config.json"):
        print(f"  Already exists — skipping {model_name} on {task_name}")
        return

    print(f"  Fine-tuning {model_name} on {task_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if train_b:
        train_enc = tokenizer(train_a, train_b, truncation=True,
                              padding="max_length", max_length=128)
        val_enc   = tokenizer(val_a,   val_b,   truncation=True,
                              padding="max_length", max_length=128)
    else:
        train_enc = tokenizer(train_a, truncation=True,
                              padding="max_length", max_length=128)
        val_enc   = tokenizer(val_a,   truncation=True,
                              padding="max_length", max_length=128)

    train_dataset = GlueDataset(train_enc, train_labels)
    val_dataset   = GlueDataset(val_enc,   val_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    results = trainer.evaluate()
    print(f"  {model_name} {task_name}: {results.get(\'eval_accuracy\', \'N/A\'):.4f}")

    del model, trainer
    torch.cuda.empty_cache()


# ── Measure ───────────────────────────────────────────────────────────────────

def measure_all(model_name, task_name, save_root,
                val_a, val_b, labels, num_labels,
                sample_a, sample_b=None):

    model_path = f"{save_root}/{model_name}_{task_name}"
    if not os.path.exists(f"{model_path}/config.json"):
        print(f"  {model_name} not found for {task_name}")
        return None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels
    ).to(device)
    model.eval()

    acc, tp  = evaluate_accuracy(model, tokenizer, val_a, val_b, labels)
    lat      = measure_latency_ms(model, tokenizer, sample_a, sample_b)
    mem      = get_memory_mb(model)
    energy   = estimate_energy(lat)

    model_q     = apply_sensimix(model)
    model_q.eval()
    q_acc, q_tp = evaluate_accuracy(model_q, tokenizer, val_a, val_b, labels)
    q_lat       = measure_latency_ms(model_q, tokenizer, sample_a, sample_b)
    q_mem       = get_memory_mb(model_q)
    q_energy    = estimate_energy(q_lat)

    del model, model_q
    torch.cuda.empty_cache()

    return {
        "baseline": {
            "memory_mb":  round(mem,    2),
            "latency_ms": round(lat,    2),
            "accuracy":   round(acc,    4),
            "bits":       32,
            "energy_w":   round(energy, 4),
            "throughput": round(tp,     2),
        },
        "sensimix": {
            "memory_mb":  round(q_mem,    2),
            "latency_ms": round(q_lat,    2),
            "accuracy":   round(q_acc,    4),
            "bits":       "8+1",
            "energy_w":   round(q_energy, 4),
            "throughput": round(q_tp,     2),
        }
    }


if __name__ == "__main__":
    print("SensiMix GLUE Benchmarking")
    print("Models  :", list(MODELS.keys()))
    print("Datasets:", list(DATASETS.keys()))
    print("Device  :", device)
'''

