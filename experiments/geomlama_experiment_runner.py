"""
Experiment Runner for GeoMLAMA Task Evaluation
Orchestrates the entire evaluation pipeline using VLLMInferenceEngine
Follows original GeoMLAMA evaluation methodology
"""

from pathlib import Path
from typing import Dict, Any, List
import yaml
import json
from datetime import datetime
from tqdm import tqdm

# Import your vLLM engine
import sys
sys.path.append(str(Path(__file__).parent))
from inference.vllm_engine import VLLMInferenceEngine, VLLMConfig

# Import task
from tasks.geomlama_task import create_geomlama_task


class GeoMLAMAExperimentRunner:
    """Runs GeoMLAMA experiments based on YAML configuration"""
    
    def __init__(self, config_path: Path):
        """
        Initialize the experiment runner.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = Path(self.config.get('output', {}).get('results_dir', 'results/geomlama'))
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
        """Load the GeoMLAMA task based on config"""
        task_config = self.config['task']
        
        print(f"\n📚 Loading task: {task_config['name']}")
        print(f"   Language: {task_config['config'].get('language', 'en')}")
        print(f"   With Country Context: {task_config['config'].get('with_country', True)}")
        print(f"   Prompt Strategy: {task_config['config'].get('prompt_strategy', 'natural')}")
        
        task = create_geomlama_task(
            dataset_path=Path(task_config['dataset_path']),
            config=task_config['config']
        )
        
        task.load_dataset()
        
        return task
    
    def run_all_experiments(self):
        """Run all experiments defined in config"""
        experiment_name = self.config.get('experiment_name', 'geomlama_experiment')
        
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
        
        # Print summary (like original GeoMLAMA)
        self._print_summary(results)
    
    def _evaluate_task(
        self,
        task,
        dataset: List,
        engine: VLLMInferenceEngine,
        gen_params: Dict[str, Any],
        batch_size: int
    ):
        """
        Evaluate the model on the task.
        
        Args:
            task: GeoMLAMA task instance
            dataset: Dataset to evaluate on (raw list)
            engine: VLLMInferenceEngine instance
            gen_params: Generation parameters
            batch_size: Batch size for inference
            
        Returns:
            List of results dictionaries
        """
        results = []
        
        print(f"\n⚙️  Running inference (batch_size={batch_size})...")
        print(f"   Generation params: {gen_params}")
        
        # Track per-country statistics (matching original GeoMLAMA)
        country_corr = [0] * 5  # exact_match counts per country
        country_overlap = [0] * 5  # overlap counts per country
        country_tot = [0] * 5  # total samples per country
        
        # Process in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, len(dataset))
            batch_items = dataset[i:batch_end]
            
            # Convert samples and prepare prompts
            batch_samples = []
            batch_prompts = []
            
            for idx, item in enumerate(batch_items):
                item_idx = i + idx  # Global index in dataset
                sample = task._convert_sample(item, item_idx)
                prompt = task.prepare_prompts(sample)
                
                batch_samples.append(sample)
                batch_prompts.append(prompt)
            
            # Generate responses using VLLMInferenceEngine
            predictions = engine.generate_batch(
                prompts=batch_prompts,
                **gen_params
            )
            
            # Evaluate responses
            for sample, prompt, prediction in zip(batch_samples, batch_prompts, predictions):
                prediction = prediction.strip()
                
                # Evaluate (returns exact_match and overlap)
                metrics = task.evaluate_response(prediction, sample)
                
                # Update country statistics (matching original GeoMLAMA)
                country_idx = sample['country_idx']
                country_corr[country_idx] += metrics['exact_match']
                country_overlap[country_idx] += metrics['overlap']
                country_tot[country_idx] += 1
                
                # Store detailed result
                result = {
                    'sample_idx': sample['sample_idx'],
                    'country': sample['country_name'],
                    'country_idx': sample['country_idx'],
                    'concept_id': sample['concept_id'],
                    'is_base_prompt': sample['is_base_prompt'],
                    'prompt_raw': sample['prompt_raw'],
                    'prompt_formatted': prompt,
                    'gold_answers': sample['gold_answers'],
                    'answer_candidates': sample['answer_candidates'],
                    'prediction': prediction,
                    'extracted_answer': task._extract_answer(prediction),
                    'metrics': metrics
                }
                
                results.append(result)
        
        # Store aggregate stats for summary (matching original output format)
        self.country_stats = {
            'country_corr': country_corr,
            'country_overlap': country_overlap,
            'country_tot': country_tot
        }
        
        return results
    
    def _save_results(self, experiment_name: str, results: List[Dict], task):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_data = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'config': self.config,
            'task_type': type(task).__name__,
            'language': task.language,
            'with_country': task.with_country,
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
        """
        Calculate aggregate metrics matching original GeoMLAMA output format.
        
        Original prints: [country_corr[0]/145.0, country_corr[1]/140.0, ...]
        """
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
        
        # Breakdown by country (matching original GeoMLAMA format)
        country_names = ['United States', 'China', 'India', 'Iran', 'Kenya']
        
        summary['by_country'] = {}
        for country_idx in range(5):
            country_results = [r for r in results if r['country_idx'] == country_idx]
            
            if country_results:
                exact_matches = [r['metrics']['exact_match'] for r in country_results]
                overlaps = [r['metrics']['overlap'] for r in country_results]
                
                summary['by_country'][country_names[country_idx]] = {
                    'count': len(country_results),
                    'exact_match': sum(exact_matches) / len(exact_matches),
                    'overlap': sum(overlaps) / len(overlaps)
                }
        
        # Store raw counts (for comparison with original)
        if hasattr(self, 'country_stats'):
            summary['country_raw_stats'] = {
                'exact_match_counts': self.country_stats['country_corr'],
                'overlap_counts': self.country_stats['country_overlap'],
                'total_counts': self.country_stats['country_tot']
            }
        
        # Breakdown by concept (25 concepts total)
        concepts = {}
        for r in results:
            concept_id = r['concept_id']
            if concept_id not in concepts:
                concepts[concept_id] = {'exact_match': [], 'overlap': []}
            
            concepts[concept_id]['exact_match'].append(r['metrics']['exact_match'])
            concepts[concept_id]['overlap'].append(r['metrics']['overlap'])
        
        summary['by_concept'] = {}
        for concept_id, metrics in concepts.items():
            summary['by_concept'][concept_id] = {
                'count': len(metrics['exact_match']),
                'exact_match': sum(metrics['exact_match']) / len(metrics['exact_match']),
                'overlap': sum(metrics['overlap']) / len(metrics['overlap'])
            }
        
        return summary
    
    def _print_summary(self, results: List[Dict]):
        """
        Print evaluation summary matching original GeoMLAMA output format.
        
        Original format:
        model_lang: [0.75, 0.65, 0.58, 0.62, 0.60]  # Per-country accuracy
        """
        summary = self._calculate_summary(results)
        
        print(f"\n{'='*70}")
        print("📊 EVALUATION SUMMARY (GeoMLAMA Format)")
        print(f"{'='*70}")
        print(f"Total Samples: {summary['total_samples']}")
        
        print(f"\n Overall Metrics:")
        for metric, stats in summary['aggregate_metrics'].items():
            print(f"  {metric}: {stats['mean']:.4f} (n={stats['count']})")
        
        # Print in original GeoMLAMA format
        print(f"\n{'='*70}")
        print("PER-COUNTRY ACCURACY (Original GeoMLAMA Format):")
        print(f"{'='*70}")
        
        country_names = ['United States', 'China', 'India', 'Iran', 'Kenya']
        
        # Format 1: Exact Match (primary metric in original)
        exact_match_scores = []
        for country in country_names:
            if country in summary['by_country']:
                exact_match_scores.append(summary['by_country'][country]['exact_match'])
            else:
                exact_match_scores.append(0.0)
        
        model_name = self.config['model']['name'].split('/')[-1]
        language = self.config['task']['config']['language']
        
        print(f"\n{model_name}_{language} (exact_match):")
        print(f"  {exact_match_scores}")
        print(f"  US: {exact_match_scores[0]:.3f}, CN: {exact_match_scores[1]:.3f}, "
              f"IN: {exact_match_scores[2]:.3f}, IR: {exact_match_scores[3]:.3f}, KE: {exact_match_scores[4]:.3f}")
        
        # Format 2: Overlap (more lenient metric)
        overlap_scores = []
        for country in country_names:
            if country in summary['by_country']:
                overlap_scores.append(summary['by_country'][country]['overlap'])
            else:
                overlap_scores.append(0.0)
        
        print(f"\n{model_name}_{language} (overlap):")
        print(f"  {overlap_scores}")
        print(f"  US: {overlap_scores[0]:.3f}, CN: {overlap_scores[1]:.3f}, "
              f"IN: {overlap_scores[2]:.3f}, IR: {overlap_scores[3]:.3f}, KE: {overlap_scores[4]:.3f}")
        
        # Detailed breakdown by country
        print(f"\n{'='*70}")
        print("DETAILED COUNTRY BREAKDOWN:")
        print(f"{'='*70}")
        
        for country, stats in summary['by_country'].items():
            print(f"\n{country} (n={stats['count']}):")
            print(f"  Exact Match: {stats['exact_match']:.4f}")
            print(f"  Overlap: {stats['overlap']:.4f}")
        
        # Concept-level analysis
        print(f"\n{'='*70}")
        print("TOP 10 CONCEPTS BY ACCURACY:")
        print(f"{'='*70}")
        
        sorted_concepts = sorted(
            summary['by_concept'].items(),
            key=lambda x: x[1]['overlap'],
            reverse=True
        )[:10]
        
        for concept_id, stats in sorted_concepts:
            print(f"  Concept {concept_id} (n={stats['count']}):")
            print(f"    Exact Match: {stats['exact_match']:.4f}, Overlap: {stats['overlap']:.4f}")
        
        print(f"\n{'='*70}")


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GeoMLAMA evaluation experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="config/geomlama_config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if config_path.exists():
        runner = GeoMLAMAExperimentRunner(config_path)
        runner.run_all_experiments()
    else:
        print(f"❌ Config file not found: {config_path}")
        print("Please create a config file first.")