"""
Model Convergence Analysis using Pearson and Spearman Correlations
Compares results between Llama3B and Qwen2.5B models on BLEND and CulturalBench datasets
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelConvergenceAnalyzer:
    def __init__(self, results_base_dir: str = "."):
        self.results_base_dir = Path(results_base_dir)
        self.blend_results = {}
        self.culturalbench_results = {}
        
    def load_blend_results(self):
        """Load BLEND results for both models"""
        print("Loading BLEND results...")
        
        # Try both possible directory naming conventions
        llama_dir = self.results_base_dir / "results_blend" / "blend_final_results_llama3b"
        qwen_dir = self.results_base_dir / "results_blend" / "blend_final_results_qwen2_5b"
        
        # Fallback to alternative naming
        if not llama_dir.exists():
            llama_dir = self.results_base_dir / "results_blend" / "final_results_llama3b"
        if not qwen_dir.exists():
            qwen_dir = self.results_base_dir / "results_blend" / "final_results_qwen2_5b"
        
        countries = ["assamese", "china", "iran", "south_korea", "sundanese", "uk", "us"]
        
        for country in countries:
            # Find files matching pattern with timestamps
            llama_pattern = f"{country}_llama*3b*.json"
            qwen_pattern = f"{country}_qwen*2*5b*.json"
            
            llama_files = list(llama_dir.glob(llama_pattern)) if llama_dir.exists() else []
            qwen_files = list(qwen_dir.glob(qwen_pattern)) if qwen_dir.exists() else []
            
            if llama_files and qwen_files:
                # Use the most recent file (in case there are multiple)
                llama_file = sorted(llama_files)[-1]
                qwen_file = sorted(qwen_files)[-1]
                
                with open(llama_file) as f:
                    llama_data = json.load(f)
                with open(qwen_file) as f:
                    qwen_data = json.load(f)
                
                self.blend_results[country] = {
                    "llama3b": llama_data,
                    "qwen2_5b": qwen_data
                }

                metrics = self.blend_results[country].get("llama3b").get("aggregate_metrics")
                if metrics is None:
                    print(f"WARNING: {country} has None for aggregate_metrics, skipping")
                    continue
                metrics_accuracy = metrics['accuracy']['mean']
                print(metrics_accuracy)
                print(f"  ✓ Loaded {country}")
                print(f"    Llama: {llama_file.name}")
                print(f"    Qwen:  {qwen_file.name}")
            else:
                if not llama_files:
                    print(f"  ✗ Missing llama file matching: {llama_dir}/{llama_pattern}")
                if not qwen_files:
                    print(f"  ✗ Missing qwen file matching: {qwen_dir}/{qwen_pattern}")
        
        print(f"Loaded {len(self.blend_results)} BLEND countries\n")
    
    def load_culturalbench_results(self):
        """Load CulturalBench results for both models"""
        print("Loading CulturalBench results...")
        
        # Try both possible directory naming conventions
        llama_dir = self.results_base_dir / "results_culturalbench" / "culturalbench_final_results_llama3b"
        qwen_dir = self.results_base_dir / "results_culturalbench" / "culturalbench_final_results_qwen2_5b"
        
        # Fallback to alternative naming
        if not llama_dir.exists():
            llama_dir = self.results_base_dir / "results_culturalbench" / "cultural_bench_final_results_llama3b"
        if not qwen_dir.exists():
            qwen_dir = self.results_base_dir / "results_culturalbench" / "cultural_bench_final_results_qwen2_5b"
        
        regions = ["africa", "east_asia", "north_america", 
                   "south_asia", "west_asia", "west_europe"]
        
        for region in regions:
            # Find files matching pattern (with or without timestamps)
            llama_pattern = "culturalbench*"f"{region}*llama*3b*.json"
            qwen_pattern = "culturalbench*"f"{region}*qwen*2*5b*.json"
            
            llama_files = list(llama_dir.glob(llama_pattern)) if llama_dir.exists() else []
            qwen_files = list(qwen_dir.glob(qwen_pattern)) if qwen_dir.exists() else []
            
            if llama_files and qwen_files:
                # Use the most recent file (in case there are multiple)
                llama_file = sorted(llama_files)[-1]
                qwen_file = sorted(qwen_files)[-1]
                
                with open(llama_file) as f:
                    llama_data = json.load(f)
                with open(qwen_file) as f:
                    qwen_data = json.load(f)
                
                self.culturalbench_results[region] = {
                    "llama3b": llama_data,
                    "qwen2_5b": qwen_data
                }
                print(f"  ✓ Loaded {region}")
                print(f"    Llama: {llama_file.name}")
                print(f"    Qwen:  {qwen_file.name}")
            else:
                if not llama_files:
                    print(f"  ✗ Missing llama file matching: {llama_dir}/{llama_pattern}")
                if not qwen_files:
                    print(f"  ✗ Missing qwen file matching: {qwen_dir}/{qwen_pattern}")
        
        print(f"Loaded {len(self.culturalbench_results)} CulturalBench regions\n")
    
    def extract_blend_pairs(self) -> Tuple[List[float], List[float], List[str]]:
        """Extract accuracy pairs from BLEND results"""
        llama_scores = []
        qwen_scores = []
        labels = []
        
        for country, data in self.blend_results.items():
            llama_acc = data["llama3b"]["aggregate_metrics"]["accuracy"]["mean"]
            qwen_acc = data["qwen2_5b"]["aggregate_metrics"]["accuracy"]["mean"]
            
            llama_scores.append(llama_acc)
            qwen_scores.append(qwen_acc)
            labels.append(country)
        
        return llama_scores, qwen_scores, labels
    
    def extract_culturalbench_pairs(self) -> Tuple[List[float], List[float], List[str]]:
        """Extract accuracy pairs from CulturalBench results at country level"""
        llama_scores = []
        qwen_scores = []
        labels = []
        
        for region, data in self.culturalbench_results.items():
            llama_countries = data["llama3b"].get("by_country", {})
            qwen_countries = data["qwen2_5b"].get("by_country", {})
            
            # Get countries present in both models
            common_countries = set(llama_countries.keys()) & set(qwen_countries.keys())
            
            for country in common_countries:
                llama_acc = llama_countries[country]["accuracy"]
                qwen_acc = qwen_countries[country]["accuracy"]
                
                llama_scores.append(llama_acc)
                qwen_scores.append(qwen_acc)
                labels.append(f"{region}_{country}")
        
        return llama_scores, qwen_scores, labels
    
    def extract_culturalbench_region_pairs(self) -> Tuple[List[float], List[float], List[str]]:
        """Extract accuracy pairs from CulturalBench results at region level"""
        llama_scores = []
        qwen_scores = []
        labels = []
        
        for region, data in self.culturalbench_results.items():
            llama_acc = data["llama3b"]["aggregate_metrics"]["accuracy"]["mean"]
            qwen_acc = data["qwen2_5b"]["aggregate_metrics"]["accuracy"]["mean"]
            
            llama_scores.append(llama_acc)
            qwen_scores.append(qwen_acc)
            labels.append(region)
        
        return llama_scores, qwen_scores, labels
    
    def calculate_correlations(self, x: List[float], y: List[float], 
                              name: str) -> Dict:
        """Calculate Pearson and Spearman correlations"""
        x_arr = np.array(x)
        y_arr = np.array(y)
        
        pearson_r, pearson_p = pearsonr(x_arr, y_arr)
        spearman_r, spearman_p = spearmanr(x_arr, y_arr)
        
        result = {
            "name": name,
            "n_samples": len(x),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "mean_diff": np.mean(np.abs(x_arr - y_arr)),
            "max_diff": np.max(np.abs(x_arr - y_arr)),
            "llama_mean": np.mean(x_arr),
            "qwen_mean": np.mean(y_arr)
        }
        
        return result
    
    def print_correlation_results(self, results: Dict):
        """Pretty print correlation results"""
        print(f"\n{'='*70}")
        print(f"Analysis: {results['name']}")
        print(f"{'='*70}")
        print(f"Number of samples: {results['n_samples']}")
        print(f"\nModel Performance:")
        print(f"  Llama3B mean accuracy:  {results['llama_mean']:.4f}")
        print(f"  Qwen2.5B mean accuracy: {results['qwen_mean']:.4f}")
        print(f"\nCorrelation Metrics:")
        print(f"  Pearson correlation:  r = {results['pearson_r']:.4f}, p = {results['pearson_p']:.4e}")
        print(f"  Spearman correlation: ρ = {results['spearman_r']:.4f}, p = {results['spearman_p']:.4e}")
        print(f"\nDifference Metrics:")
        print(f"  Mean absolute difference: {results['mean_diff']:.4f}")
        print(f"  Max absolute difference:  {results['max_diff']:.4f}")
        
        # Interpretation
        pearson_strength = self._interpret_correlation(results['pearson_r'])
        spearman_strength = self._interpret_correlation(results['spearman_r'])
        print(f"\nInterpretation:")
        print(f"  Pearson:  {pearson_strength}")
        print(f"  Spearman: {spearman_strength}")
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation strength"""
        abs_r = abs(r)
        if abs_r >= 0.9:
            strength = "Very strong"
        elif abs_r >= 0.7:
            strength = "Strong"
        elif abs_r >= 0.5:
            strength = "Moderate"
        elif abs_r >= 0.3:
            strength = "Weak"
        else:
            strength = "Very weak"
        
        direction = "positive" if r > 0 else "negative"
        return f"{strength} {direction} correlation"
    
    def plot_scatter(self, x: List[float], y: List[float], labels: List[str],
                     title: str, filename: str):
        """Create scatter plot with regression line"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=100, edgecolors='black', linewidth=1)
        
        # Add labels for points
        for i, label in enumerate(labels):
            ax.annotate(label, (x[i], y[i]), fontsize=8, 
                       xytext=(5, 5), textcoords='offset points', alpha=0.7)
        
        # Regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
        
        # Diagonal line (perfect agreement)
        min_val = min(min(x), min(y))
        max_val = max(max(x), max(y))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', 
                alpha=0.3, linewidth=1, label='Perfect agreement')
        
        ax.set_xlabel('Llama3B Accuracy', fontsize=12)
        ax.set_ylabel('Qwen2.5B Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    def create_summary_table(self, all_results: List[Dict]) -> pd.DataFrame:
        """Create summary table of all correlation results"""
        df = pd.DataFrame(all_results)
        df = df[['name', 'n_samples', 'pearson_r', 'pearson_p', 
                 'spearman_r', 'spearman_p', 'mean_diff', 'llama_mean', 'qwen_mean']]
        return df
    
    def run_full_analysis(self):
        """Run complete convergence analysis"""
        print("\n" + "="*70)
        print("MODEL CONVERGENCE ANALYSIS")
        print("Comparing Llama3B vs Qwen2.5B")
        print("="*70 + "\n")
        
        # Load data
        self.load_blend_results()
        self.load_culturalbench_results()
        
        all_results = []
        
        # 1. BLEND Analysis
        if self.blend_results:
            print("\n" + ">"*70)
            print("BLEND DATASET ANALYSIS")
            print(">"*70)
            
            llama_blend, qwen_blend, blend_labels = self.extract_blend_pairs()
            blend_results = self.calculate_correlations(
                llama_blend, qwen_blend, "BLEND (Country-level)"
            )
            self.print_correlation_results(blend_results)
            all_results.append(blend_results)
            
            self.plot_scatter(
                llama_blend, qwen_blend, blend_labels,
                "Model Convergence on BLEND Dataset",
                "benchmark_comparisons/visualizations/blend_convergence.png"
            )
        
        # 2. CulturalBench Analysis (Country-level)
        if self.culturalbench_results:
            print("\n" + ">"*70)
            print("CULTURALBENCH DATASET ANALYSIS (Country-level)")
            print(">"*70)
            
            llama_cb, qwen_cb, cb_labels = self.extract_culturalbench_pairs()
            cb_results = self.calculate_correlations(
                llama_cb, qwen_cb, "CulturalBench (Country-level)"
            )
            self.print_correlation_results(cb_results)
            all_results.append(cb_results)
            
            self.plot_scatter(
                llama_cb, qwen_cb, cb_labels,
                "Model Convergence on CulturalBench Dataset (Country-level)",
                "benchmark_comparisons/visualizations/culturalbench_convergence_countries.png"
            )
            
            # 3. CulturalBench Analysis (Region-level)
            print("\n" + ">"*70)
            print("CULTURALBENCH DATASET ANALYSIS (Region-level)")
            print(">"*70)
            
            llama_cbr, qwen_cbr, cbr_labels = self.extract_culturalbench_region_pairs()
            cbr_results = self.calculate_correlations(
                llama_cbr, qwen_cbr, "CulturalBench (Region-level)"
            )
            self.print_correlation_results(cbr_results)
            all_results.append(cbr_results)
            
            self.plot_scatter(
                llama_cbr, qwen_cbr, cbr_labels,
                "Model Convergence on CulturalBench Dataset (Region-level)",
                "benchmark_comparisons/visualizations/culturalbench_convergence_regions.png"
            )
        
        # 4. Combined Analysis (all data points)
        if self.blend_results and self.culturalbench_results:
            print("\n" + ">"*70)
            print("COMBINED ANALYSIS (All Datasets)")
            print(">"*70)
            
            all_llama = llama_blend + llama_cb
            all_qwen = qwen_blend + qwen_cb
            all_labels = blend_labels + cb_labels
            
            combined_results = self.calculate_correlations(
                all_llama, all_qwen, "Combined (BLEND + CulturalBench)"
            )
            self.print_correlation_results(combined_results)
            all_results.append(combined_results)
            
            self.plot_scatter(
                all_llama, all_qwen, all_labels,
                "Model Convergence Across All Datasets",
                "benchmark_comparisons/visualizations/combined_convergence.png"
            )
        
        # 5. Create summary table
        if all_results:
            print("\n" + "="*70)
            print("SUMMARY TABLE")
            print("="*70 + "\n")
            
            summary_df = self.create_summary_table(all_results)
            print(summary_df.to_string(index=False))
            
            # Save summary table
            summary_df.to_csv("benchmark_comparisons/visualizations/convergence_summary.csv", index=False)
            print(f"\n  Saved: benchmark_comparisons/visualizations/convergence_summary.csv")
        else:
            print("\n" + "="*70)
            print("ERROR: No results found!")
            print("="*70)
            print("\nPlease check that your directory structure matches:")
            print("  results_blend/blend_final_results_llama3b/*.json")
            print("  results_blend/blend_final_results_qwen2_5b/*.json")
            print("  results_culturalbench/cultural_bench_final_results_llama3b/*.json")
            print("  results_culturalbench/cultural_bench_final_results_qwen2_5b/*.json")
            print("\nRun the debug script to check your file locations:")
            print("  python check_files.py")
            summary_df = pd.DataFrame()
        
        # Create detailed comparison plot
        if all_results:
            self.create_comparison_plot(all_results)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE" if all_results else "NO DATA FOUND")
        print("="*70 + "\n")
        
        return all_results, summary_df
    
    def create_comparison_plot(self, all_results: List[Dict]):
        """Create bar chart comparing correlations across datasets"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        names = [r['name'] for r in all_results]
        pearson_rs = [r['pearson_r'] for r in all_results]
        spearman_rs = [r['spearman_r'] for r in all_results]
        
        x = np.arange(len(names))
        width = 0.35
        
        # Pearson
        bars1 = ax1.bar(x, pearson_rs, width, label='Pearson r', color='steelblue', alpha=0.8)
        ax1.set_ylabel('Correlation Coefficient', fontsize=12)
        ax1.set_title('Pearson Correlations', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([-1, 1])
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Spearman
        bars2 = ax2.bar(x, spearman_rs, width, label='Spearman ρ', color='coral', alpha=0.8)
        ax2.set_ylabel('Correlation Coefficient', fontsize=12)
        ax2.set_title('Spearman Correlations', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([-1, 1])
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('benchmark_comparisons/visualizations/correlation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved:benchmark_comparisons/visualizations/correlation_comparison.png")


def main():

    import os
    os.makedirs("benchmark_comparisons/visualizations", exist_ok=True)
    
    # Initialize analyzer
    analyzer = ModelConvergenceAnalyzer(results_base_dir=".")
    
    # Run full analysis
    results, summary_df = analyzer.run_full_analysis()
    
    print("\nAll visualizations and summary saved to benchmark_comparisons/visualizations/\n")


if __name__ == "__main__":
    main()