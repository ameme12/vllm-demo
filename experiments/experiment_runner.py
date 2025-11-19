import yaml
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import json
from tqdm import tqdm

class ExperimentRunner:
    """Main orchestrator for running experiments across multiple datasets with vLLM"""
    
    def __init__(self, config_path: Path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path(self.config.get('results_dir', './results'))
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.vllm_engine = None
        
    def run_all_experiments(self):
        """Run all configured experiments"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for exp_config in self.config['experiments']:
            print(f"\n{'='*60}")
            print(f"Running experiment: {exp_config['name']}")
            print(f"{'='*60}\n")
            
            results = self.run_single_experiment(exp_config)
            
            # Save results
            result_file = self.results_dir / f"{exp_config['name']}_{timestamp}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to: {result_file}")
            
            # Cleanup vLLM engine after each experiment to free GPU memory
            if self.vllm_engine is not None:
                del self.vllm_engine
                self.vllm_engine = None
                torch.cuda.empty_cache()
    
    def run_single_experiment(self, exp_config: Dict) -> Dict:
        """Run a single experiment configuration"""
        from repos.repo_manager import RepoManager
        from inference.vllm_engine import VLLMInferenceEngine, VLLMConfig
        
        # Initialize vLLM engine
        model_config = exp_config['model']
        vllm_config = VLLMConfig(
            model_name=model_config['name'],
            tensor_parallel_size=model_config.get('tensor_parallel_size', 1),
            gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.9),
            max_model_len=model_config.get('max_model_len'),
            dtype=model_config.get('dtype', 'auto'),
            quantization=model_config.get('quantization'),
            max_num_seqs=model_config.get('max_num_seqs', 256),
        )
        
        self.vllm_engine = VLLMInferenceEngine(vllm_config)
        
        # Setup repositories
        repo_manager = RepoManager()
        repo_paths = {}
        
        for repo_config in exp_config.get('repositories', []):
            repo_path = repo_manager.clone_or_update(
                repo_config['url'],
                repo_config.get('name'),
                repo_config.get('branch')
            )
            repo_paths[repo_config['name']] = repo_path
            repo_manager.setup_repo(repo_path, repo_config.get('install_deps', True))
        
        # Load tasks
        tasks = self._load_tasks(exp_config['tasks'], repo_paths)
        
        # Run evaluation
        results = {
            'experiment_name': exp_config['name'],
            'config': exp_config,
            'tasks': {}
        }
        
        for task_name, task in tasks.items():
            print(f"\nEvaluating task: {task_name}")
            task_results = self._evaluate_task(task, exp_config)
            results['tasks'][task_name] = task_results
            
        return results
    
    def _load_tasks(self, task_configs: List[Dict], repo_paths: Dict) -> Dict:
        """Load task objects from configurations"""
        from tasks.carrot_task import CARROTTask
        # Import other tasks as needed
        
        tasks = {}
        
        for task_config in task_configs:
            task_type = task_config['type']
            task_name = task_config['name']
            
            # Get dataset path from repo
            repo_name = task_config['repository']
            dataset_path = repo_paths[repo_name] / task_config.get('dataset_path', '')
            
            # Instantiate appropriate task class
            if task_type == 'carrot':
                task = CARROTTask(dataset_path, task_config)
            # Add other task types here
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            tasks[task_name] = task
            
        return tasks
    
    def _evaluate_task(self, task, exp_config: Dict) -> Dict:
        """Evaluate a single task using vLLM batch inference"""
        # Get samples
        n_samples = exp_config.get('n_samples')
        samples = task.get_samples(n_samples=n_samples)
        
        # Batch processing configuration
        batch_size = exp_config.get('batch_size', 32)
        
        # Sampling parameters
        model_config = exp_config['model']
        temperature = model_config.get('temperature', 0.0)
        top_p = model_config.get('top_p', 1.0)
        max_tokens = model_config.get('max_tokens', 512)
        
        results = []
        
        # Process in batches for efficiency
        for i in tqdm(range(0, len(samples), batch_size), desc="Processing batches"):
            batch_samples = samples[i:i+batch_size]
            
            # Prepare batch prompts
            batch_prompts = task.prepare_batch_prompts(batch_samples)
            
            # Generate predictions using vLLM (efficient batch inference)
            predictions = self.vllm_engine.generate(
                prompts=batch_prompts,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            
            # Evaluate each prediction
            for sample, prediction in zip(batch_samples, predictions):
                eval_result = task.evaluate_response(
                    prediction,
                    sample.get('answer', sample.get('target'))
                )
                results.append(eval_result)
        
        # Aggregate results
        metrics = self._aggregate_metrics(results)
        
        return {
            'n_samples': len(samples),
            'metrics': metrics,
            'individual_results': results[:10]  # Store first 10 for inspection
        }
    
    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Aggregate metrics across all samples"""
        metrics = {}
        
        # Extract all metric names
        metric_names = set()
        for result in results:
            metric_names.update(k for k in result.keys() 
                              if isinstance(result[k], (int, float)))
        
        # Calculate averages
        for metric_name in metric_names:
            values = [r[metric_name] for r in results if metric_name in r]
            metrics[f'{metric_name}_mean'] = sum(values) / len(values) if values else 0
            
        return metrics
