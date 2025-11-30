"""
Experiment Runner for CulturalBench Task Evaluation
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
from tasks.cultural_bench_task import create_culturalbench_mcq_task as create_culturalbench_task, CulturalBenchTask

class ExperimentRunner:
    """Runs experiments based on YAML configuration"""
    
    def __init__(self, config_path: Path):
        """
        Initialize the experiment runner.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = Path(self.config.get('output', {}).get('results_dir', 'results'))
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
        print(f"   Max tokens: {model_config.get('max_tokens', 100)}")
        
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
            'max_tokens': model_config.get('max_tokens', 100),
            'top_p': model_config.get('top_p', 1.0),
            'top_k': model_config.get('top_k', -1),
            'presence_penalty': model_config.get('presence_penalty', 0.0),
            'frequency_penalty': model_config.get('frequency_penalty', 0.0),
            'stop': model_config.get('stop', None),
        }
    
    def _load_task(self):
        """Load the CulturalBench task based on config"""
        task_config = self.config['task']
        
        print(f"\n📚 Loading task: {task_config['name']}")
        
        # Print filter info
        country = task_config['config'].get('country')
        region = task_config['config'].get('region')
        
        if country:
            print(f"   Filter: Country = {country}")
        elif region:
            print(f"   Filter: Region = {region}")
        else:
            print(f"   Filter: All countries")
        
        task = create_culturalbench_task(
            dataset_path=Path("."),
            config=task_config['config']
        )
        
        task.load_dataset()
        
        return task
    
    def run_all_experiments(self):
        """Run all experiments defined in config"""
        experiment_name = self.config.get('experiment_name', 'culturalbench_experiment')
        
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
            dataset = task.dataset.select(range(num_samples))
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
        dataset,
        engine: VLLMInferenceEngine,
        gen_params: Dict[str, Any],
        batch_size: int
    ):
        """
        Evaluate the model on the task.
        
        Args:
            task: CulturalBench task instance
            dataset: Dataset to evaluate on
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
            batch_indices = range(i, min(i+batch_size, len(dataset)))
            
            # Convert samples and prepare prompts
            batch_samples = []
            batch_prompts = []
            
            for idx in batch_indices:
                item = dataset[idx]
                sample = task._convert_sample(item)
                
                prompt = task.prepare_prompts(sample)
                batch_samples.append(sample)
                batch_prompts.append(prompt)
            
            # Generate responses using VLLMInferenceEngine
            predictions = engine.generate_batch(
                prompts=batch_prompts,
                **gen_params
            )
            
            # Evaluate responses
            for sample, prediction in zip(batch_samples, predictions):
                prediction = prediction.strip()

                # === DEBUGGING (optional) ===
                if self.config.get('output', {}).get('verbose', False):
                    print("\n" + "="*70)
                    print("PROMPT:")
                    prompt_text = task.prepare_prompts(sample)
                    print(prompt_text + "..." if len(prompt_text) > 200 else prompt_text)
                    print("\nPREDICTION:")
                    print(prediction)
                    print("="*70)
                # === END DEBUGGING ===
                
                # Evaluate
                metrics = task.evaluate_response(prediction, sample)
                
                # Store result
                result = {
                    'sample_id': sample.get('data_idx', 'unknown'),
                    'question_idx': sample.get('question_idx', 'unknown'),
                    'country': sample.get('country', 'unknown'),
                    'region': sample.get('region', 'unknown'),
                    'prompt_question': sample['prompt_question'],
                    'prediction': prediction,
                    'ground_truth': sample['answer'],
                    'metrics': metrics
                }

                if self.config.get('output', {}).get('verbose', False):
                    print(f"Ground Truth: {result['ground_truth']}")
                    print(f"Accuracy: {metrics['accuracy']}")

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
                'min': min(values),
                'max': max(values)
            }
        
        # Add breakdown by country/region
        summary['by_country'] = self._calculate_country_breakdown(results)
        summary['by_region'] = self._calculate_region_breakdown(results)
        
        return summary
    
    def _calculate_country_breakdown(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics breakdown by country"""
        from collections import defaultdict
        
        country_results = defaultdict(list)
        for result in results:
            country = result.get('country', 'Unknown')
            country_results[country].append(result['metrics'])
        
        breakdown = {}
        for country, metrics_list in country_results.items():
            breakdown[country] = {
                'count': len(metrics_list),
                'accuracy': sum(m['accuracy'] for m in metrics_list) / len(metrics_list)
            }
        
        return breakdown
    
    def _calculate_region_breakdown(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics breakdown by region"""
        from collections import defaultdict
        
        region_results = defaultdict(list)
        for result in results:
            region = result.get('region', 'Unknown')
            region_results[region].append(result['metrics'])
        
        breakdown = {}
        for region, metrics_list in region_results.items():
            breakdown[region] = {
                'count': len(metrics_list),
                'accuracy': sum(m['accuracy'] for m in metrics_list) / len(metrics_list)
            }
        
        return breakdown
    
    def _print_summary(self, results: List[Dict]):
        """Print evaluation summary"""
        summary = self._calculate_summary(results)
        
        print(f"\n{'='*70}")
        print("📊 EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total Samples: {summary['total_samples']}")
        print(f"\nAggregate Metrics:")
        
        for metric, stats in summary['aggregate_metrics'].items():
            print(f"  {metric}:")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Min:  {stats['min']:.4f}")
            print(f"    Max:  {stats['max']:.4f}")
        
        # Print country breakdown
        print(f"\n{'='*70}")
        print("📍 BREAKDOWN BY COUNTRY")
        print(f"{'='*70}")
        for country, stats in sorted(summary['by_country'].items()):
            print(f"  {country}: {stats['accuracy']:.4f} ({stats['count']} samples)")
        
        # Print region breakdown
        print(f"\n{'='*70}")
        print("🌍 BREAKDOWN BY REGION")
        print(f"{'='*70}")
        for region, stats in sorted(summary['by_region'].items()):
            print(f"  {region}: {stats['accuracy']:.4f} ({stats['count']} samples)")
        
        print(f"{'='*70}")


if __name__ == "__main__":
    # Test the runner
    config_path = Path("config/culturalbench_config.yaml")
    if config_path.exists():
        runner = ExperimentRunner(config_path)
        runner.run_all_experiments()
    else:
        print(f"Config file not found: {config_path}")
        print("Please create a config file first.")