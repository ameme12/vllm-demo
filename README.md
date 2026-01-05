vllm-demo: Cross-Benchmark Evaluation of Cultural Commonsense Knowledge in LLMs

This repository contains the experimental evaluation pipeline used in the paper:

Cross-Benchmark Analysis of Cultural Commonsense Knowledge in Language Models
Ameline Ramesan, McGill University & MILA

The framework enables systematic, reproducible evaluation of cultural commonsense knowledge in large language models (LLMs) and supports cross-benchmark correlation analysis to study whether cultural benchmarks measure overlapping or distinct constructs.

Repository: https://github.com/ameme12/vllm-demo

🎯 Research Motivation

As LLMs are deployed globally, evaluating their ability to represent diverse cultural commonsense knowledge is increasingly important. While many cultural benchmarks exist, it remains unclear whether benchmarks designed to test similar knowledge actually measure the same underlying constructs.

This repository supports experiments that address the following research question:

Do cultural commonsense knowledge benchmarks converge in their assessment of LLMs, or do they capture distinct dimensions of cultural knowledge?

📊 Benchmarks Implemented

The framework focuses on commonsense and factual cultural knowledge, following the taxonomy of cultural benchmarks in recent NLP literature.

1. BLEnD

Benchmarking LLMs’ Everyday Knowledge Across Diverse Cultures and Languages
Dataset: nayeon212/BLEnD

Evaluates everyday cultural commonsense (foods, customs, daily practices)

52.6k QA pairs across 16 countries and 13 languages

Reformatted to multiple-choice for cross-benchmark comparison

Emphasizes lived, practice-based cultural knowledge

2. CulturalBench (Easy)

Human-written questions across 45 regions

Evaluates factual and commonsense cultural knowledge

Used in multiple-choice format

Region- and country-level accuracy analysis supported

3. GeoMLAMA

Geo-specific commonsense knowledge (foods, holidays, currencies, social practices)

Originally cloze-style, reformatted to multiple-choice

English subset used to isolate cultural knowledge from language proficiency

4. DLAMA-v1

Culturally diverse factual knowledge derived from Wikidata triples

Generation-based evaluation using LLM-as-judge

Represents culture-as-factual-knowledge, contrasting with practice-based benchmarks

🤖 Models Evaluated

Llama 3.2-3B-Instruct

Qwen 2.5-3B-Instruct

Both models are evaluated using identical prompting, decoding, and evaluation protocols to enable controlled cross-benchmark comparison.

🧪 Evaluation Methodology
Core Evaluation

Metric: Accuracy (binary correctness)

Decoding: Greedy (temperature = 0.0)

Inference: vLLM for efficient batched execution

Language: English-only evaluation to isolate cultural knowledge effects

Cross-Benchmark Analysis

The framework supports:

Benchmark-level correlation
(Do benchmarks rank countries similarly?)

Country-level correlation
(Do countries show similar performance patterns across benchmarks?)

These analyses enable empirical study of benchmark construct validity.

📈 Analysis & Visualization

The pipeline generates:

Country-level and benchmark-level accuracy summaries

Predicate-level breakdowns (DLAMA)

Cross-model comparisons

Correlation matrices and performance heatmaps

These visualizations are used directly in the accompanying paper to demonstrate weak-to-moderate correlations (Pearson r < 0.4) between benchmarks.

📁 Project Structure (Paper-Oriented)
vllm-demo/
├── tasks/                 # Benchmark task implementations
│   ├── dlama_task.py
│   ├── culturegen_task.py
│   └── base_task.py
├── run_*_evaluation.py    # Benchmark runners
├── analyze_*              # Accuracy & correlation analysis
├── compare_models_*       # Cross-model comparisons
├── configs/               # YAML experiment configs
├── results_*              # Per-benchmark outputs
└── README.md

🎓 Relation to the Paper

This repository:

Implements the full experimental pipeline

Produces all benchmark- and country-level metrics

Enables reproducible correlation analysis across benchmarks

Supports the paper’s conclusion that cultural commonsense benchmarks are not interchangeable

📄 Paper (unpublished):
Cross-Benchmark Analysis of Cultural Commonsense Knowledge in Language Models

📝 Citation

If you use this code or experimental design, please cite the paper and software:

@software{ramesan2025culturalbenchmarks,
  author = {Ramesan, Ameline},
  title = {vllm-demo: Cross-Benchmark Evaluation of Cultural Commonsense Knowledge},
  year = {2025},
  url = {https://github.com/ameme12/vllm-demo}
}
