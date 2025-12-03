"""
Experiment Runner for DLAMA Task Evaluation
Orchestrates the entire evaluation pipeline using VLLMInferenceEngine
"""

from pathlib import Path
from typing import Dict, Any, List
import yaml
import json
from datetime import datetime
from tqdm import tqdm

# Import your vLLM engine
import sys
sys.path.append(str(Path(__file__).parent.parent))
from inference.vllm_engine import VLLMInferenceEngine, VLLMConfig

# Import task
from tasks.dlama_task import create_dlama_task


class DLAMAExperimentRunner:
    """Runs DLAMA experiments based on YAML configuration"""
    
    def __init__(self, config_path: Path):
        """
        Initialize the experiment runner.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = Path(self.config.get('output', {}).get('results_dir', 'results/dlama'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📋 Loaded config from: {config_path}")
        print(f"📁 Results will be saved to: {self.results_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _load_model(self) -> VLLMInferenceEngine:
        """Load the vLLM model using VLLMInferenceEngine"""
        model_config = self.config['model']
        
        print(f"\n🤖 Loading model: {model_config['name']}")
        print(f"   Temperature: {model_config.get('temperature', 0.0)}")
        print(f"   Max tokens: {model_config.get('max_tokens', 50)}")
        
        # Create VLLMConfig
        vllm_config = VLLMConfig(
            model_name=model_config['name'],
            tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
            gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.9),
            max_model_len=model_config.get('max_model_len', None),
            trust_remote_code=model_config.get('trust_remote_code', True),
            dtype=model_config.get('dtype', 'auto'),
            quantization=model_config.get('quantization', None),
            swap_space=model_config.get('swap_space', 4),
            enforce_eager=model_config.get('enforce_eager', False),
            max_num_seqs=model_config.get('max_num_seqs', 256),
            seed=model_config.get('seed', 42)
        )
        
        # Initialize engine
        engine = VLLMInferenceEngine(vllm_config)
        
        return engine
    
    def _get_generation_params(self) -> Dict[str, Any]:
        """Extract generation parameters from config"""
        model_config = self.config['model']
        
        return {
            'temperature': model_config.get('temperature', 0.0),
            'max_tokens': model_config.get('max_tokens', 50),
            'top_p': model_config.get('top_p', 1.0),
            'top_k': model_config.get('top_k', -1),
            'presence_penalty': model_config.get('presence_penalty', 0.0),
            'frequency_penalty': model_config.get('frequency_penalty', 0.0),
            'stop': model_config.get('stop', None),
        }
    
    def _load_task(self):
        """Load the DLAMA task based on config"""
        task_config = self.config['task']
        
        print(f"\n📚 Loading task: {task_config['name']}")
        print(f"   Predicate: {task_config['config'].get('predicate', 'All')}")
        print(f"   Culture: {task_config['config'].get('culture', 'All')}")
        print(f"   Country: {task_config['config'].get('country', 'All')}")
        
        task = create_dlama_task(
            dataset_path=Path("."),
            config=task_config['config']
        )
        
        task.load_dataset()
        
        return task
    
    def run_all_experiments(self):
        """Run all experiments defined in config"""
        experiment_name = self.config.get('experiment_name', 'dlama_experiment')
        
        print(f"\n{'='*70}")
        print(f"🚀 Starting Experiment: {experiment_name}")
        print(f"{'='*70}")
        
        # Load model and task
        engine = self._load_model()
        task = self._load_task()
        
        # Get evaluation settings
        eval_config = self.config.get('evaluation', {})
        num_samples = eval_config.get('num_samples', -1)
        batch_size = eval_config.get('batch_size', 32)
        
        # Select samples
        if num_samples > 0 and num_samples < len(task.dataset):
            dataset = task.dataset[:num_samples]
            print(f"\n📊 Evaluating on {num_samples} samples (subset)")
        else:
            dataset = task.dataset
            print(f"\n📊 Evaluating on all {len(dataset)} samples")
        
        # Get generation parameters
        gen_params = self._get_generation_params()
        
        # Run evaluation
        results = self._evaluate_task(task, dataset, engine, gen_params, batch_size)
        
        # Save results
        self._save_results(experiment_name, results, task)
        
        # Print summary
        self._print_summary(results)
    
    def _evaluate_task(
        self,
        task,
        dataset: List[Dict],
        engine: VLLMInferenceEngine,
        gen_params: Dict[str, Any],
        batch_size: int
    ):
        """
        Evaluate the model on the task.
        
        Args:
            task: DLAMA task instance
            dataset: Dataset to evaluate on (list of dicts)
            engine: VLLMInferenceEngine instance
            gen_params: Generation parameters
            batch_size: Batch size for inference
            
        Returns:
            List of results dictionaries
        """
        results = []
        
        print(f"\n⚙️  Running inference (batch_size={batch_size})...")
        print(f"   Generation params: {gen_params}")
        
        # Process in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, len(dataset))
            batch_items = dataset[i:batch_end]
            
            # Convert samples and prepare prompts
            batch_samples = []
            batch_prompts = []
            
            for item in batch_items:
                sample = task._convert_sample(item)
                prompt = task.prepare_prompts(sample)
                
                batch_samples.append(sample)
                batch_prompts.append(prompt)
            
            # Generate responses using VLLMInferenceEngine
            predictions = engine.generate_batch(
                prompts=batch_prompts,
                **gen_params
            )
            
            # Evaluate responses - THIS IS THE KEY PART
            for idx, (sample, prompt, prediction) in enumerate(zip(batch_samples, batch_prompts, predictions)):
                prediction = prediction.strip()
                
                # Evaluate
                metrics = task.evaluate_response(prediction, sample)
                
                # Store result
                result = {
                    'sample_id': sample.get('uuid', 'unknown'),
                    'subject': sample['subject'],
                    'predicate': sample['pred_description'],
                    'predicate_code': sample['predicate'],
                    'correct_answer': sample['correct_answer'],
                    'culture': sample['culture'],
                    'country': sample['country_names'][0] if sample['country_names'] else 'Unknown',
                    'prompt': prompt,  # Use the prompt from the zip, not the variable
                    'prediction': prediction,
                    'extracted_answer': task._extract_answer(prediction),
                    'metrics': metrics
                }
                
                results.append(result)
        
        return results
    
    def _save_results(self, experiment_name: str, results: List[Dict], task):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_data = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'config': self.config,
            'task_type': type(task).__name__,
            'num_samples': len(results),
            'results': results
        }
        
        # Save individual results
        results_file = self.results_dir / f"{experiment_name}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Save summary
        summary = self._calculate_summary(results)
        summary_file = self.results_dir / f"{experiment_name}_{timestamp}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Summary saved to: {summary_file}")
    
    def _calculate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate metrics"""
        if not results:
            return {}
        
        # Aggregate metrics
        all_metrics = [r['metrics'] for r in results]
        
        summary = {
            'total_samples': len(results),
            'aggregate_metrics': {}
        }
        
        # Calculate average for each metric
        metric_keys = all_metrics[0].keys()
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            summary['aggregate_metrics'][key] = {
                'mean': sum(values) / len(values),
                'count': len(values)
            }
        
        # Breakdown by culture
        cultures = {}
        for r in results:
            culture = r['culture']
            if culture not in cultures:
                cultures[culture] = {'exact_match': [], 'overlap': []}
            
            cultures[culture]['exact_match'].append(r['metrics']['exact_match'])
            cultures[culture]['overlap'].append(r['metrics']['overlap'])
        
        summary['by_culture'] = {}
        for culture, metrics in cultures.items():
            summary['by_culture'][culture] = {
                'count': len(metrics['exact_match']),
                'exact_match': sum(metrics['exact_match']) / len(metrics['exact_match']),
                'overlap': sum(metrics['overlap']) / len(metrics['overlap'])
            }
        
        # ===== NEW: Breakdown by country =====
        countries = {}
        for r in results:
            country = r['country']
            if country not in countries:
                countries[country] = {'exact_match': [], 'overlap': []}
            
            countries[country]['exact_match'].append(r['metrics']['exact_match'])
            countries[country]['overlap'].append(r['metrics']['overlap'])
        
        summary['by_country'] = {}
        for country, metrics in countries.items():
            summary['by_country'][country] = {
                'count': len(metrics['exact_match']),
                'exact_match': sum(metrics['exact_match']) / len(metrics['exact_match']),
                'overlap': sum(metrics['overlap']) / len(metrics['overlap'])
            }
        # ===== END NEW =====
        
        # Breakdown by predicate
        predicates = {}
        for r in results:
            pred = r['predicate_code']
            if pred not in predicates:
                predicates[pred] = {'exact_match': [], 'overlap': []}
            
            predicates[pred]['exact_match'].append(r['metrics']['exact_match'])
            predicates[pred]['overlap'].append(r['metrics']['overlap'])
        
        summary['by_predicate'] = {}
        for pred, metrics in predicates.items():
            summary['by_predicate'][pred] = {
                'count': len(metrics['exact_match']),
                'exact_match': sum(metrics['exact_match']) / len(metrics['exact_match']),
                'overlap': sum(metrics['overlap']) / len(metrics['overlap'])
            }
        
        return summary
    
    def _print_summary(self, results: List[Dict]):
        """Print evaluation summary"""
        summary = self._calculate_summary(results)
        
        print(f"\n{'='*70}")
        print("📊 EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total Samples: {summary['total_samples']}")
        
        print(f"\n Overall Metrics:")
        for metric, stats in summary['aggregate_metrics'].items():
            print(f"  {metric}: {stats['mean']:.4f} (n={stats['count']})")
        
        print(f"\n By Culture:")
        for culture, stats in summary['by_culture'].items():
            print(f"  {culture} (n={stats['count']}):")
            print(f"    Exact Match: {stats['exact_match']:.4f}")
            print(f"    Overlap: {stats['overlap']:.4f}")
        
        # ===== NEW: Print country breakdown =====
        print(f"\n By Country:")
        # Sort by sample count (descending)
        sorted_countries = sorted(
            summary['by_country'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        for country, stats in sorted_countries:
            print(f"  {country} (n={stats['count']}):")
            print(f"    Exact Match: {stats['exact_match']:.4f}")
            print(f"    Overlap: {stats['overlap']:.4f}")
        # ===== END NEW =====
        
        print(f"\n Top 5 Predicates:")
        sorted_preds = sorted(
            summary['by_predicate'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:5]
        
        for pred, stats in sorted_preds:
            print(f"  {pred} (n={stats['count']}):")
            print(f"    Exact Match: {stats['exact_match']:.4f}")
            print(f"    Overlap: {stats['overlap']:.4f}")
        
        print(f"{'='*70}")


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DLAMA evaluation experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dlama_config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if config_path.exists():
        runner = DLAMAExperimentRunner(config_path)
        runner.run_all_experiments()
    else:
        print(f"Config file not found: {config_path}")
        print("Please create a config file first.")