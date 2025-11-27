# BLEnD Evaluation Pipeline

Complete automation pipeline for evaluating LLMs on the BLEnD (Benchmark for LLMs on Everyday Knowledge in Diverse Cultures and Languages) dataset.

## 📁 Project Structure

```
project/
├── tasks/
│   ├── __init__.py
│   ├── blend_task.py          # BLEnD task implementation
│   └── base_task.py            # Base task interface (your existing file)
├── experiments/
│   ├── __init__.py
│   └── experiment_runner.py    # Experiment orchestration
├── config/
│   └── blend_config.yaml       # Configuration files
├── results/                    # Results output directory
├── data/                       # Optional: local data cache
├── run_experiments.py          # Main entry point
├── requirements.txt            # Dependencies
└── setup.sh                    # Setup script
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup (creates directories)
./setup.sh

# Install dependencies
pip install -r requirements.txt

# Or with conda
conda create -n blend python=3.10
conda activate blend
pip install -r requirements.txt
```

### 2. Organize Files

Place the files in the correct directories:

```bash
# Tasks
cp blend_task.py tasks/
cp tasks__init__.py tasks/__init__.py

# Experiments
cp experiment_runner.py experiments/
cp experiments__init__.py experiments/__init__.py

# Config
cp blend_config.yaml config/
```

### 3. Configure Experiment

Edit `config/blend_config.yaml`:

```yaml
experiment_name: my_blend_experiment

task:
  config:
    blend_config: multiple-choice-questions  # or "short-answer-questions"
    culture: DZ  # See culture codes below
    use_english: true

model:
  name: meta-llama/Llama-3.2-3B-Instruct  # Your model
  temperature: 0.0
  max_tokens: 100

evaluation:
  num_samples: 50  # -1 for all samples
  batch_size: 32
```

### 4. Run Evaluation

```bash
# Run on default GPU
python run_experiments.py --config config/blend_config.yaml

# Run on specific GPU(s)
python run_experiments.py --config config/blend_config.yaml --gpu 0
python run_experiments.py --config config/blend_config.yaml --gpu 0,1
```

## 📋 Culture Codes

| Code | Country/Region | Language |
|------|----------------|----------|
| US | United States | English |
| UK | United Kingdom | English |
| CN | China | Chinese |
| ES | Spain | Spanish |
| MX | Mexico | Spanish |
| ID | Indonesia | Indonesian |
| KR | South Korea | Korean |
| KP | North Korea | Korean |
| GR | Greece | Greek |
| IR | Iran | Persian |
| DZ | Algeria | Arabic |
| AZ | Azerbaijan | Azerbaijani |
| JB | West Java | Sundanese |
| AS | Assam | Assamese |
| NG | Northern Nigeria | Hausa |
| ET | Ethiopia | Amharic |

## 🔧 Configuration Options

### Task Configuration

```yaml
task:
  config:
    blend_config: "multiple-choice-questions"  # Required
    # Options: "multiple-choice-questions", "short-answer-questions"
    
    culture: "DZ"  # Required
    # See culture codes table above
    
    use_english: true  # Optional, default: true
    # true: Use English translations
    # false: Use local language
```

### Model Configuration

```yaml
model:
  name: "meta-llama/Llama-3.2-3B-Instruct"  # Required
  # HuggingFace model name or path
  
  temperature: 0.0  # Optional, default: 0.0
  # 0.0 for deterministic output
  
  max_tokens: 100  # Optional, default: 100
  # Maximum tokens to generate
  
  top_p: 1.0  # Optional, default: 1.0
  
  gpu_memory_utilization: 0.9  # Optional, default: 0.9
  # Fraction of GPU memory to use
  
  trust_remote_code: true  # Optional, default: true
```

### Evaluation Configuration

```yaml
evaluation:
  num_samples: -1  # Optional, default: -1 (all)
  # Number of samples to evaluate
  # -1 or omit for all samples
  
  batch_size: 32  # Optional, default: 32
  # Batch size for inference
  
  metrics:  # Optional
    - accuracy
    - exact_match
    - has_valid_format
```

### Output Configuration

```yaml
output:
  results_dir: "results"  # Optional, default: "results"
  # Directory to save results
  
  save_predictions: true  # Optional, default: true
  # Save individual predictions
  
  verbose: true  # Optional, default: false
  # Print detailed progress
```

## 📊 Output Files

After running an experiment, you'll get:

```
results/
├── blend_algeria_mcq_20241127_143052.json          # Full results
└── blend_algeria_mcq_20241127_143052_summary.json  # Aggregate metrics
```

### Results JSON Format

```json
{
  "experiment_name": "blend_algeria_mcq",
  "timestamp": "20241127_143052",
  "config": { ... },
  "task_type": "BLEnDMCQTask",
  "num_samples": 100,
  "results": [
    {
      "sample_id": "mcq_001",
      "prompt": "What is a common snack...",
      "prediction": "{\"answer_choice\": \"C\"}",
      "metrics": {
        "accuracy": 1.0,
        "exact_match": 1.0
      },
      "ground_truth_idx": 2,
      "ground_truth_letter": "C",
      "choices": ["...", "...", "...", "..."]
    }
  ]
}
```

### Summary JSON Format

```json
{
  "total_samples": 100,
  "aggregate_metrics": {
    "accuracy": {
      "mean": 0.8500,
      "min": 0.0,
      "max": 1.0
    },
    "exact_match": {
      "mean": 0.8500,
      "min": 0.0,
      "max": 1.0
    }
  }
}
```

## 🎯 Example Workflows

### 1. Quick Test (10 samples)

```yaml
# config/quick_test.yaml
experiment_name: quick_test

task:
  config:
    blend_config: multiple-choice-questions
    culture: US
    use_english: true

model:
  name: meta-llama/Llama-3.2-3B-Instruct
  temperature: 0.0

evaluation:
  num_samples: 10
  batch_size: 10
```

```bash
python run_experiments.py --config config/quick_test.yaml
```

### 2. Full Evaluation (All samples)

```yaml
# config/full_eval.yaml
experiment_name: full_evaluation

task:
  config:
    blend_config: multiple-choice-questions
    culture: KR
    use_english: true

model:
  name: meta-llama/Llama-3.2-3B-Instruct
  temperature: 0.0

evaluation:
  num_samples: -1  # All samples
  batch_size: 32
```

```bash
python run_experiments.py --config config/full_eval.yaml --gpu 0
```

### 3. Multi-Culture Comparison

Create separate configs for each culture and run them:

```bash
# Evaluate on multiple cultures
for culture in US CN KR DZ AS; do
  # Create config for each culture (or use template)
  python run_experiments.py --config config/blend_${culture}.yaml
done
```

## 🔍 Troubleshooting

### Issue: "CUDA not available"
**Solution:** Ensure you have a GPU and PyTorch with CUDA support:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: "Culture 'XX' not found"
**Solution:** Check the culture code matches one from the COUNTRY_NAME dict. Use the exact codes from the table above.

### Issue: "No data found for culture"
**Solution:** Some cultures may not have MCQ data. Try:
1. Check if culture has data in the dataset
2. Try short-answer-questions instead
3. Verify COUNTRY_NAME mapping in blend_task.py

### Issue: "Model not found"
**Solution:** Ensure the model name is correct and you have access:
```bash
huggingface-cli login  # If using gated models
```

### Issue: "Out of memory"
**Solution:** Reduce batch_size or gpu_memory_utilization:
```yaml
evaluation:
  batch_size: 16  # Reduce from 32

model:
  gpu_memory_utilization: 0.7  # Reduce from 0.9
```
