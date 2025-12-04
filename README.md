# vllm-demo: Cultural Bias Evaluation Framework

A comprehensive evaluation framework for assessing cultural bias and knowledge distribution in Large Language Models (LLMs). This project implements multiple cultural knowledge benchmarks to compare how well models like Llama and Qwen understand factual knowledge across different cultures.

**Repository**: [ameme12/vllm-demo](https://github.com/ameme12/vllm-demo)

## 🎯 Overview

This framework evaluates cultural bias in language models by testing their factual knowledge across different cultural contexts. We use multiple benchmarks to assess whether models exhibit bias toward specific cultures and how well they capture knowledge from diverse cultural perspectives.

## 📊 Benchmarks Implemented

### 1. DLAMA (Diverse Language Adaptive Multilingual Assessment)
Evaluates factual knowledge queries across cultural contexts using Wikidata predicates.

**Datasets:**
- **DLAMA Arab-West**: Compares knowledge about Arab vs Western cultures
- **DLAMA Asia-West**: Compares knowledge about Asian vs Western cultures

**Predicates Tested** (20 total):
- **Demographics**: P106 (Occupation), P27 (Country of citizenship), P19 (Place of birth)
- **Geography**: P17 (Country), P30 (Continent), P36 (Capital), P47 (Shares border with)
- **Language**: P37 (Official language), P103 (Native language), P1412 (Languages spoken)
- **Culture**: P136 (Genre), P495 (Country of origin), P364 (Original language of work)
- **Organizations**: P264 (Record label), P449 (Original network), P190 (Sister city)
- **Miscellaneous**: P1303 (Instrument), P530 (Diplomatic relation), P20 (Place of death), P1376 (Capital of)

### 2. Culture-Gen
Evaluates cultural perception and stereotypes across 110 countries/regions covering 8 culture-related topics.

**Topics:**
- Food preferences
- Social customs
- Religious practices
- Traditional celebrations
- Family structures
- Communication styles
- Work culture
- Art and aesthetics

### 3. Additional Benchmarks
*(Based on your evaluation pipeline structure)*
- Multiple choice evaluations
- Free-form generation tasks
- Probability-based assessments

## 🤖 Models Evaluated

- **Llama 3.2-3B-Instruct** (`meta-llama/Llama-3.2-3B-Instruct`)
- **Qwen 2.5-3B-Instruct** (`Qwen/Qwen2.5-3B-Instruct`)

### Evaluation Setup
- **Temperature**: 0.0 (deterministic generation)
- **Inference Engine**: vLLM for efficient batch processing
- **Fuzzy Matching**: RapidFuzz with threshold of 80
- **Metrics**: Exact match and substring overlap

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver
- CUDA-capable GPU (recommended for model inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/ameme12/vllm-demo.git
cd vllm-demo

# Install dependencies with uv
uv sync
```

### Running Evaluations

#### DLAMA Benchmark

**Arab-West Dataset:**
```bash
# Llama 3B
uv run python run_dlama_evaluation.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset dlama_arab_west \
    --output_dir results_dlama

# Qwen 2.5B
uv run python run_dlama_evaluation.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --dataset dlama_arab_west \
    --output_dir results_dlama
```

**Asia-West Dataset:**
```bash
# Llama 3B
uv run python run_dlama_evaluation.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset dlama_asia_west \
    --output_dir results_dlama

# Qwen 2.5B
uv run python run_dlama_evaluation.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --dataset dlama_asia_west \
    --output_dir results_dlama
```

#### Culture-Gen Benchmark

```bash
# Llama 3B
uv run python run_culturegen_evaluation.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_dir results_culturegen

# Qwen 2.5B
uv run python run_culturegen_evaluation.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --output_dir results_culturegen
```

## 📈 Visualization & Analysis

### Individual Model Analysis (DLAMA)

```bash
# Analyze single experiment results
uv run python analyze_dlama_results.py \
    --results_file results_dlama/dlama_arab_llama3b_20251203_222321.json \
    --output_dir results_dlama/visualizations \
    --model_name "Llama 3B" \
    --metric overlap
```

**Generates 6 visualization figures:**
1. **Culture Comparison Table** - Side-by-side culture accuracy
2. **Detailed Breakdown** - Top 5 predicates × Top/Bottom 5 countries (≥100 samples)
3. **Top/Bottom Predicates by Culture** - Best and worst performing predicates (≥100 samples)
4. **Top/Bottom Countries by Culture** - Best and worst performing countries (≥50 samples)
5. **Summary Statistics** - Comprehensive breakdown by culture, country, and predicate
6. **Accuracy by Culture for Top Predicates** - Direct comparison across cultures

### Cross-Model Comparison

```bash
# Compare Llama 3B vs Qwen 2.5B across both datasets
uv run python compare_models_dlama.py \
    --llama_arab results_dlama/dlama_arab_llama3b_*_summary.json \
    --llama_asia results_dlama/dlama_asia_llama3b_*_summary.json \
    --qwen_arab results_dlama/dlama_arab_qwen2.5b_*_summary.json \
    --qwen_asia results_dlama/dlama_asia_qwen2.5b_*_summary.json \
    --output_dir results_dlama/model_comparison \
    --metric overlap
```

**Generates 6 comparison figures:**
1. **Overall Comparison Table** - Culture-specific accuracy for all model-dataset combinations
2. **Top 5 Predicates by Sample Count** - Performance comparison on most common predicates
3. **Culture Breakdown (2×2 Grid)** - Culture-specific performance for each model-dataset
4. **Predicate Performance Heatmap** - Top 10 predicates across all combinations
5. **Cultural Bias Analysis** - Accuracy differences between cultures
6. **Predicate Sample Breakdown** - Sample distribution by predicate and culture

## 📁 Project Structure

```
vllm-demo/
├── run_dlama_evaluation.py        # DLAMA evaluation runner
├── run_culturegen_evaluation.py   # Culture-Gen evaluation runner
├── analyze_dlama_results.py       # Single-model DLAMA visualization
├── analyze_dlama_results_universal.py  # Universal version (any culture pair)
├── compare_models_dlama.py        # Multi-model DLAMA comparison
├── tasks/
│   ├── base_task.py               # Base task interface
│   ├── dlama_task.py              # DLAMA evaluation implementation
│   ├── culturegen_task.py         # Culture-Gen evaluation implementation
│   └── ...                        # Additional benchmark tasks
├── utils/
│   ├── vllm_inference.py          # vLLM inference engine
│   ├── fuzzy_matching.py          # RapidFuzz matching utilities
│   └── metrics.py                 # Evaluation metrics
├── configs/                       # YAML configuration files
│   ├── dlama_arab_west.yaml
│   ├── dlama_asia_west.yaml
│   └── culturegen.yaml
├── results_dlama/                 # DLAMA evaluation results
│   ├── dlama_*_summary.json      # Summary metrics
│   ├── visualizations/           # Per-model visualizations
│   └── model_comparison/         # Cross-model comparisons
├── results_culturegen/            # Culture-Gen evaluation results
├── Makefile                       # Convenience commands
├── pyproject.toml                 # uv project configuration
└── README.md                      # This file
```

## 🔧 Configuration

Evaluations can be configured via YAML files or command-line arguments:

```yaml
# Example: DLAMA Arab-West Configuration
model:
  name: "meta-llama/Llama-3.2-3B-Instruct"
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
  max_model_len: 4096
  trust_remote_code: true

task:
  name: "dlama"
  dataset: "dlama_v1_arab_west"
  temperature: 0.0
  max_tokens: 50
  top_p: 1.0
  batch_size: 32
  use_fuzzy_matching: true
  fuzzy_threshold: 80

output:
  results_dir: "results_dlama"
  experiment_name: "dlama_arab_llama3b"
  save_predictions: true
  save_summary: true

logging:
  level: "INFO"
  log_file: "logs/evaluation.log"
```

## 📊 Evaluation Metrics

### Core Metrics
- **Exact Match**: Binary exact string match (case-insensitive, normalized)
- **Substring Overlap**: Fuzzy string matching using RapidFuzz (threshold: 80)

### Analysis Metrics
- **Overall Accuracy**: Performance across all samples
- **Culture-Specific Accuracy**: Performance per culture (Arab/Asia/Western)
- **Predicate-Specific Accuracy**: Performance per knowledge type (e.g., P106, P17)
- **Country-Specific Accuracy**: Performance per country
- **Cultural Bias Score**: Absolute accuracy difference between cultures

## 🧪 Statistical Filtering

To ensure meaningful comparisons, visualizations apply sample-count thresholds:

| Figure/Analysis | Entity | Minimum Samples | Selection Criteria |
|----------------|--------|-----------------|-------------------|
| Detailed Breakdown | Predicates | 100 | Top 5 by accuracy |
| Detailed Breakdown | Countries | 100 | Top 5 + Bottom 5 per culture |
| Predicate Performance | Predicates | 100 | Top 5 + Bottom 5 per culture |
| Country Performance | Countries | 50 | Top 5 + Bottom 5 per culture |
| Heatmap | Predicates | N/A | Top 10 by average accuracy |

## 🎓 Research Questions Addressed

This evaluation framework helps answer:

1. **Knowledge Distribution**: How well do models know facts about different cultures?
2. **Cultural Bias**: Do models favor certain cultures (e.g., Western) over others?
3. **Knowledge Type Variation**: Which predicates are universally easy/hard vs culture-specific?
4. **Model Architecture Impact**: How do different model families handle cultural knowledge?
5. **Data Imbalance Effects**: How does training data distribution affect cultural knowledge?

## 💡 Key Features

- **🚀 Fast Inference**: Uses vLLM for efficient batch processing
- **📦 Easy Setup**: Managed by uv for fast, reliable dependency resolution
- **🎨 Rich Visualizations**: Comprehensive plots and tables for analysis
- **🔄 Modular Design**: Easy to add new benchmarks and models
- **📊 Statistical Rigor**: Sample filtering ensures meaningful comparisons
- **🌍 Multi-Cultural**: Supports multiple culture pairs and regions

## 🛠️ Development

### Adding a New Benchmark

1. Create a new task class in `tasks/your_benchmark_task.py`
2. Inherit from `BaseTask` and implement required methods
3. Add configuration in `configs/your_benchmark.yaml`
4. Create evaluation runner script
5. Add visualization utilities if needed

### Adding a New Model

Simply specify the model name in your configuration or command-line:

```bash
uv run python run_dlama_evaluation.py \
    --model_name your-org/your-model-name \
    --dataset dlama_arab_west
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- 📊 Additional benchmarks (e.g., cultural bias in generation tasks)
- 🤖 More model evaluations (GPT, Claude, Mistral, etc.)
- 🌍 New cultural comparison pairs (e.g., African-Western, Latin-Western)
- 📈 Statistical significance testing
- 🎯 Prompt engineering experiments
- 🔍 Error analysis tools

## 📝 Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@software{vllm_demo_cultural_bias,
  author = {Ameline},
  title = {vllm-demo: Cultural Bias Evaluation Framework},
  year = {2024},
  url = {https://github.com/ameme12/vllm-demo}
}
```

### Benchmark Citations

**DLAMA:**
```bibtex
@article{dlama2024,
  title={DLAMA: A Framework for Curating Culturally Diverse Facts for Probing the Knowledge of Pretrained Language Models},
  author={...},
  journal={...},
  year={2024}
}
```

**Culture-Gen:**
```bibtex
@article{culturegen2024,
  title={Culture-Gen: Revealing Global Cultural Perception in Language Models through Natural Language Prompting},
  author={...},
  journal={...},
  year={2024}
}

@misc{myung2025blendbenchmarkllmseveryday,
      title={BLEnD: A Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages}, 
      author={Junho Myung and Nayeon Lee and Yi Zhou and Jiho Jin and Rifki Afina Putri and Dimosthenis Antypas and Hsuvas Borkakoty and Eunsu Kim and Carla Perez-Almendros and Abinew Ali Ayele and Víctor Gutiérrez-Basulto and Yazmín Ibáñez-García and Hwaran Lee and Shamsuddeen Hassan Muhammad and Kiwoong Park and Anar Sabuhi Rzayev and Nina White and Seid Muhie Yimam and Mohammad Taher Pilehvar and Nedjma Ousidhoum and Jose Camacho-Collados and Alice Oh},
      year={2025},
      eprint={2406.09948},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.09948}, 
}
```

## 📧 Contact

For questions or collaborations, please open an issue on [GitHub](https://github.com/ameme12/vllm-demo).

## 📜 License

[Add your license here]

---

**Built with ❤️ using [uv](https://github.com/astral-sh/uv) and [vLLM](https://github.com/vllm-project/vllm)**