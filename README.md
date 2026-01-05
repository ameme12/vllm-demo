# vllm-demo: Cross-Benchmark Evaluation of Cultural Commonsense Knowledge in LLMs

**Author:** Ameline Ramesan  
**Affiliation:** McGill University & MILA - Quebec AI Institute  
**Repository:** [https://github.com/ameme12/vllm-demo](https://github.com/ameme12/vllm-demo)

---

## 📖 Overview

This repository contains the experimental evaluation pipeline for the research paper:

> **Cross-Benchmark Analysis of Cultural Commonsense Knowledge in Language Models**  
> *Ameline Ramesan, McGill University & MILA*
> 📄 [Read the paper (PDF)](https://github.com/ameme12/vllm-demo/blob/main/Commonsense-knowledge-cultural-benchmarks-analysis-in-LLMs.pdf)

The framework enables systematic, reproducible evaluation of cultural commonsense knowledge in large language models (LLMs) and supports cross-benchmark correlation analysis to study whether cultural benchmarks measure overlapping or distinct constructs.

---

## 🎯 Research Motivation

As LLMs are deployed globally, evaluating their ability to represent diverse cultural commonsense knowledge is increasingly important. While many cultural benchmarks exist, it remains unclear whether benchmarks designed to test similar knowledge actually measure the same underlying constructs.

### Research Question

> **Do cultural commonsense knowledge benchmarks converge in their assessment of LLMs, or do they capture distinct dimensions of cultural knowledge?**

This work addresses this gap through comprehensive cross-benchmark evaluation and correlation analysis.

---

## 📊 Benchmarks Implemented

The framework focuses on **commonsense and factual cultural knowledge**, following the taxonomy of cultural benchmarks in recent NLP literature.

### 1. **BLEnD** - *Benchmarking LLMs' Everyday Knowledge*

- **Paper:** [Benchmarking LLMs' Everyday Knowledge Across Diverse Cultures and Languages](https://arxiv.org/abs/2406.09948)
- **Dataset:** [`nayeon212/BLEnD`](https://huggingface.co/datasets/nayeon212/BLEnD)
- **Coverage:** 52.6k QA pairs across **16 countries** and **13 languages**
- **Focus:** Everyday cultural commonsense (foods, customs, daily practices)
- **Format:** Reformatted to multiple-choice for cross-benchmark comparison
- **Type:** Practice-based cultural knowledge

### 2. **CulturalBench (Easy)** - *Regional Cultural Knowledge*

- **Paper:** [CulturalBench: A Robust, Diverse, and Challenging Benchmark on Measuring the (Lack of) Cultural Knowledge of LLMs](https://arxiv.org/abs/2410.13334)
- **Coverage:** Human-written questions across **45 regions**
- **Focus:** Factual and commonsense cultural knowledge
- **Format:** Multiple-choice
- **Analysis:** Region- and country-level accuracy analysis supported

### 3. **GeoMLAMA** - *Geo-Specific Commonsense*

- **Paper:** [GeoMLAMA: Geo-Diverse Commonsense Probing on Multilingual Pre-Trained Language Models](https://arxiv.org/abs/2205.12247)
- **Coverage:** Geo-specific commonsense knowledge (foods, holidays, currencies, social practices)
- **Format:** Originally cloze-style, reformatted to multiple-choice
- **Language:** English subset used to isolate cultural knowledge from language proficiency

### 4. **DLAMA-v1** - *Culturally Diverse Factual Knowledge*

- **Paper:** [DLAMA: A Framework for Curating Culturally Diverse Facts for Probing the Knowledge of Pretrained Language Models](https://arxiv.org/abs/2403.05307)
- **Source:** Culturally diverse factual knowledge derived from Wikidata triples
- **Format:** Generation-based evaluation using LLM-as-judge
- **Type:** Culture-as-factual-knowledge (contrasts with practice-based benchmarks)

---

## 🤖 Models Evaluated

| Model | Parameters | Source |
|-------|-----------|--------|
| **Llama 3.2-3B-Instruct** | 3B | Meta |
| **Qwen 2.5-3B-Instruct** | 3B | Alibaba |

Both models are evaluated using **identical prompting, decoding, and evaluation protocols** to enable controlled cross-benchmark comparison.

---

## 🧪 Evaluation Methodology

### Core Evaluation

- **Metric:** Accuracy (binary correctness)
- **Decoding:** Greedy decoding (temperature = 0.0)
- **Inference:** vLLM for efficient batched execution
- **Language:** English-only evaluation to isolate cultural knowledge effects

### Cross-Benchmark Analysis

The framework supports:

1. **Benchmark-level correlation**  
   *Do benchmarks rank countries similarly?*

2. **Country-level correlation**  
   *Do countries show similar performance patterns across benchmarks?*

These analyses enable empirical study of **benchmark construct validity**.

---

## 📈 Analysis & Visualization

The pipeline generates:

- ✅ **Country-level** and **benchmark-level** accuracy summaries
- ✅ **Predicate-level** breakdowns (DLAMA)
- ✅ **Cross-model** comparisons
- ✅ **Correlation matrices** and **performance heatmaps**

### Key Findings

These visualizations demonstrate **weak-to-moderate correlations** (Pearson *r* < 0.4) between benchmarks, supporting the conclusion that cultural commonsense benchmarks are **not interchangeable**.

---

## 📁 Repository Structure
```
vllm-demo/
├── tasks/                      # Benchmark task implementations
│   ├── dlama_task.py          # DLAMA evaluation
│   ├── culturegen_task.py     # CulturalBench & GeoMLAMA
│   ├── blend_task.py          # BLEnD evaluation
│   └── base_task.py           # Base task interface
│
├── experiments/                # Experiment orchestration
│   └── experiment_runner.py   # vLLM-based inference engine
│
├── inference/                  # Model inference
│   └── vllm_engine.py         # vLLM wrapper
│
├── config/                     # Experiment configurations
│   ├── blend_config.yaml
│   ├── culturalbench_config.yaml
│   └── dlama_config.yaml
│
├── results/                    # Benchmark outputs
│   ├── blend_final_results_*/
│   ├── culturalbench_results_*/
│   └── dlama_results_*/
│
├── run_experiments.py          # Main experiment runner
├── analyze_result.py           # Accuracy & correlation analysis
├── compare_models.py           # Cross-model comparison
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/ameme12/vllm-demo.git
cd vllm-demo

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (for gated models)
huggingface-cli login
```

### Run Evaluation
```bash
# Run BLEnD evaluation
python run_experiments.py --config config/blend_config.yaml --gpu 0

# Analyze results
python analyze_result.py
```

### Generate Visualizations
```bash
# Cross-model comparison
python compare_models.py

# Correlation analysis
python correlation_analysis.py
```

---

## 📊 Results

Results are organized by benchmark and model:
```
results/
├── blend_final_results_llama3b/
│   ├── CN_llama_3b_summary.json
│   ├── US_llama_3b_summary.json
│   └── ...
├── blend_final_results_qwen2_5b/
│   └── ...
└── visualizations/
    ├── accuracy_by_culture.png
    ├── model_comparison.png
    └── correlation_matrix.png
```

---

## 🎓 Relation to the Paper

This repository:

- ✅ Implements the **full experimental pipeline**
- ✅ Produces all **benchmark-** and **country-level metrics**
- ✅ Enables **reproducible correlation analysis** across benchmarks
- ✅ Supports the paper's conclusion that cultural commonsense benchmarks are **not interchangeable**

**Paper (unpublished):**  
*Cross-Benchmark Analysis of Cultural Commonsense Knowledge in Language Models*

---

## 📝 Citation

If you use this code or experimental design, please cite:
```bibtex
@software{ramesan2025culturalbenchmarks,
  author = {Ramesan, Ameline},
  title = {vllm-demo: Cross-Benchmark Evaluation of Cultural Commonsense Knowledge},
  year = {2025},
  institution = {McGill University and MILA - Quebec AI Institute},
  url = {https://github.com/ameme12/vllm-demo}
}
```

---

## 🤝 Acknowledgments

This work was conducted at:
- **McGill University**
- **MILA - Quebec AI Institute**

Special thanks to Golnoosh Farnadi my supervisor at the EQUAL LAB.

---

## 📧 Contact

**Ameline Ramesan**  
McGill University & MILA  
Email: [ameline.ramesan@mail.mcgill.ca]  
GitHub: [@ameme12](https://github.com/ameme12)

---

## 🔗 Related Resources

- [BLEnD Dataset](https://huggingface.co/datasets/nayeon212/BLEnD)
- [CulturalBench](https://huggingface.co/datasets/tum-nlp/CulturalBench)
- [GeoMLAMA](https://github.com/UKPLab/GeoMLAMA)
- [DLAMA](https://github.com/norakassner/LAMA)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ for cross-cultural NLP research

</div>
