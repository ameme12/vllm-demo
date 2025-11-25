from pathlib import Path
from experiments.experiment_runner import ExperimentRunner
import torch

def main():
    """Main entry point for running experiments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run LLM evaluation experiments with vLLM')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment configuration YAML')
    parser.add_argument('--gpu', type=str, default=None,
                       help='GPU device(s) to use (e.g., "0" or "0,1")')
    
    args = parser.parse_args()
    
    # Set GPU devices
    if args.gpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. vLLM requires GPU.")
        return
    
    print(f"Using GPU(s): {torch.cuda.device_count()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Run experiments
    runner = ExperimentRunner(Path(args.config))
    runner.run_all_experiments()
    
    print("\n✅ All experiments completed!")

if __name__ == "__main__":
    main()
