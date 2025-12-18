"""
Benchmark Comparison Heatmap Visualization
Creates comprehensive heatmaps showing model performance across benchmarks for all common countries
Similar to the uploaded image format with heatmap + bar chart
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class BenchmarkComparisonVisualizer:
    """
    Creates heatmap visualizations comparing benchmark performance across common countries
    """
    
    def __init__(self):
        """Initialize the visualizer"""
        self.benchmarks = {}
        self.country_mappings = self._load_country_mappings()
        self.model_mappings = self._load_model_mappings()
    
    def _load_model_mappings(self) -> Dict[str, str]:
        """Define model name standardizations across benchmarks."""
        return {
            'llama 3b': 'Llama 3.2-3B',
            'llama 3.2-3b': 'Llama 3.2-3B',
            'qwen 2.5b': 'Qwen 2.5-3B',
            'qwen 2.5-3b': 'Qwen 2.5-3B',
        }
    
    def standardize_model_name(self, model: str) -> str:
        """Standardize a model name"""
        model_lower = model.strip().lower()
        if model_lower in self.model_mappings:
            return self.model_mappings[model_lower]
        return model.strip()
    
    def _load_country_mappings(self) -> Dict[str, str]:
        """Define country name standardizations across benchmarks."""
        return {
            'united states': 'United States',
            'united kingdom': 'United Kingdom',
            'south korea': 'South Korea',
            'south africa': 'South Africa',
            'saudi arabia': 'Saudi Arabia',
            'hong kong': 'Hong Kong',
            'new zealand': 'New Zealand',
            'usa': 'United States',
            'us': 'United States',
            'uk': 'United Kingdom',
            'britain': 'United Kingdom',
            'nigerio': 'Nigeria',
            'assam (assamese)': 'India',
            'west java (sundanese)': 'Indonesia',
            'china': 'China',
            'mexico': 'Mexico',
            'iran': 'Iran',
            'indonesia': 'Indonesia',
            'spain': 'Spain',
            'nigeria': 'Nigeria',
            'india': 'India',
            'kenya': 'Kenya',
        }
    
    def standardize_country_name(self, country: str) -> str:
        """Standardize a country name"""
        country_lower = country.strip().lower()
        if country_lower in self.country_mappings:
            return self.country_mappings[country_lower]
        return country.strip().title()
    
    def load_benchmark(self, benchmark_name: str, csv_path: str) -> pd.DataFrame:
        """Load a benchmark CSV file."""
        print(f"\n{'='*60}")
        print(f"Loading: {benchmark_name}")
        print(f"{'='*60}")
        
        df = pd.read_csv(csv_path)
        
        country_col = df.columns[0]
        df[country_col] = df[country_col].apply(self.standardize_country_name)
        df = df.set_index(country_col)
        df.index.name = 'country'
        
        if df.index.duplicated().any():
            duplicates = df.index[df.index.duplicated()].unique()
            print(f"⚠️  Warning: Found duplicate countries: {list(duplicates)}")
            df = df[~df.index.duplicated(keep='first')]
        
        # Standardize model names
        df.columns = [self.standardize_model_name(col) for col in df.columns]
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Countries: {len(df)}, Models: {df.columns.tolist()}")
        
        self.benchmarks[benchmark_name] = df
        return df
    
    def get_common_countries(self, benchmark_names: List[str]) -> List[str]:
        """Get countries that exist in ALL specified benchmarks."""
        if not benchmark_names:
            return []
        
        common = set(self.benchmarks[benchmark_names[0]].index)
        for bench_name in benchmark_names[1:]:
            common = common.intersection(set(self.benchmarks[bench_name].index))
        
        return sorted(list(common))
    
    def create_benchmark_comparison_heatmap(self,
                                           benchmark_names: List[str],
                                           model: str,
                                           save_path: Optional[Path] = None):
        """
        Create a comprehensive heatmap showing benchmark performance across common countries.
        Format matches the uploaded image: heatmap on top, bar chart on bottom.
        
        Args:
            benchmark_names: List of 2 benchmarks to compare
            model: Model name to analyze
            save_path: Where to save the figure
        """
        
        print(f"\n{'='*80}")
        print(f"Creating Benchmark Comparison Heatmap")
        print(f"Model: {model}")
        print(f"Benchmarks: {benchmark_names}")
        print(f"{'='*80}")
        
        # Get common countries
        common_countries = self.get_common_countries(benchmark_names)
        
        print(f"\nCommon countries across benchmarks: {len(common_countries)}")
        print(f"Countries: {common_countries}")
        
        if len(common_countries) < 3:
            print(f"❌ Need at least 3 countries for meaningful visualization!")
            return None
        
        # Check if model exists in all benchmarks
        for bench_name in benchmark_names:
            df = self.benchmarks[bench_name]
            if model not in df.columns:
                print(f"❌ Model '{model}' not found in {bench_name}")
                return None
        
        # Collect data: benchmarks × countries
        data_matrix = []
        valid_countries = []
        
        for country in common_countries:
            country_accuracies = []
            valid = True
            
            for bench_name in benchmark_names:
                df = self.benchmarks[bench_name]
                acc = df.loc[country, model]
                
                if pd.isna(acc):
                    valid = False
                    break
                
                country_accuracies.append(float(acc))
            
            if valid:
                data_matrix.append(country_accuracies)
                valid_countries.append(country)
        
        print(f"\nCountries with valid data: {len(valid_countries)}")
        
        if len(valid_countries) < 3:
            print(f"❌ Need at least 3 countries with valid data!")
            return None
        
        # Convert to DataFrame
        df_data = pd.DataFrame(
            data_matrix,
            index=valid_countries,
            columns=benchmark_names
        ).T  # Transpose: benchmarks as rows, countries as columns
        
        # Calculate average accuracy per country
        avg_accuracy = df_data.mean(axis=0).sort_values(ascending=False)
        
        # Sort countries by average accuracy
        sorted_countries = avg_accuracy.index.tolist()
        df_data_sorted = df_data[sorted_countries]
        
        # Calculate Pearson correlation
        bench1_accs = df_data_sorted.iloc[0].values
        bench2_accs = df_data_sorted.iloc[1].values
        r, p = pearsonr(bench1_accs, bench2_accs)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # Subplot 1: Heatmap
        ax1 = fig.add_subplot(gs[0])
        
        sns.heatmap(
            df_data_sorted,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Accuracy (%)'},
            linewidths=1.0,
            linecolor='white',
            ax=ax1,
            annot_kws={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        ax1.set_xlabel('', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Benchmark', fontsize=14, fontweight='bold')
        ax1.set_title(
            f'Benchmark Comparison: {model}\n'
            f'Pearson r = {r:.4f} ({sig}), p = {p:.4f}, n = {len(sorted_countries)} countries',
            fontsize=16, fontweight='bold', pad=20
        )
        
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=11)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=12)
        
        # Subplot 2: Average accuracy bar chart
        ax2 = fig.add_subplot(gs[1])
        
        # Color based on threshold: green > 70, orange 50-70, red < 50
        colors = ['#2ecc71' if acc >= 70 else '#f39c12' if acc >= 50 else '#e74c3c' 
                  for acc in avg_accuracy.values]
        
        bars = ax2.bar(range(len(avg_accuracy)), avg_accuracy.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for i, (country, acc) in enumerate(avg_accuracy.items()):
            ax2.text(i, acc + 1.5, f'{acc:.1f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        ax2.set_xlabel('', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Avg\nAccuracy', fontsize=12, fontweight='bold', rotation=0, 
                      ha='right', va='center')
        ax2.set_title('Average Accuracy Across Benchmarks', fontsize=14, fontweight='bold', pad=10)
        ax2.set_xticks(range(len(avg_accuracy)))
        ax2.set_xticklabels(avg_accuracy.index, rotation=45, ha='right', fontsize=11)
        ax2.set_ylim(0, 110)
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=1.5)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved heatmap to {save_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("Summary Statistics:")
        print(f"{'='*60}")
        print(f"Model: {model}")
        print(f"Benchmarks: {benchmark_names}")
        print(f"Countries analyzed: {len(valid_countries)}")
        print(f"Pearson correlation: r = {r:.4f}, p = {p:.4f} ({sig})")
        print(f"\nCountries by average accuracy:")
        for country, acc in avg_accuracy.items():
            print(f"  {country}: {acc:.1f}%")
        
        return fig


def main():
    """Main analysis script"""
    
    visualizer = BenchmarkComparisonVisualizer()
    
    # Define paths
    csv_dir = Path("/home/mila/r/ramesana/projects/vllm-demo/csv_tables")
    
    benchmark_files = {
        'BLEND': csv_dir / 'blend' / 'blend_accuracy_by_country.csv',
        'CulturalBench': csv_dir / 'culturalbench' / 'culturalbench_accuracy_by_country.csv',
        'GeoMLAMA': csv_dir / 'geomlama' / 'geomlama_accuracy_by_country.csv',
    }
    
    # Load benchmarks
    print("="*80)
    print("LOADING BENCHMARKS")
    print("="*80)
    
    loaded_benchmarks = []
    for benchmark_name, csv_path in benchmark_files.items():
        if csv_path.exists():
            visualizer.load_benchmark(benchmark_name, str(csv_path))
            loaded_benchmarks.append(benchmark_name)
        else:
            print(f"⚠️  File not found: {csv_path}")
    
    if len(loaded_benchmarks) < 2:
        print("\n❌ Need at least 2 benchmarks to compare!")
        return
    
    # Create output directory
    output_dir = Path("benchmark_comparison_heatmaps")
    output_dir.mkdir(exist_ok=True)
    
    # Define models to analyze
    models = ['Llama 3.2-3B', 'Qwen 2.5-3B']
    
    # Define pairwise benchmark comparisons
    benchmark_pairs = [
        (['BLEND', 'CulturalBench'], 'BLEND_vs_CulturalBench'),
        (['BLEND', 'GeoMLAMA'], 'BLEND_vs_GeoMLAMA'),
        (['CulturalBench', 'GeoMLAMA'], 'CulturalBench_vs_GeoMLAMA')
    ]
    
    print(f"\n{'='*80}")
    print("CREATING BENCHMARK COMPARISON HEATMAPS")
    print(f"{'='*80}")
    print(f"Models: {models}")
    print(f"Benchmark pairs: {[name for _, name in benchmark_pairs]}")
    print(f"Total visualizations to generate: {len(models) * len(benchmark_pairs)}")
    
    # Generate all pairwise comparison heatmaps
    for model in models:
        print(f"\n{'='*80}")
        print(f"ANALYZING MODEL: {model}")
        print(f"{'='*80}")
        
        safe_model_name = model.replace('/', '_').replace(' ', '_').replace('.', '_')
        
        for benchmarks, pair_name in benchmark_pairs:
            print(f"\n{'='*60}")
            print(f"Pair: {' vs '.join(benchmarks)}")
            print(f"{'='*60}")
            
            filename = f'{pair_name}_{safe_model_name}.png'
            
            fig = visualizer.create_benchmark_comparison_heatmap(
                benchmarks,
                model,
                save_path=output_dir / filename
            )
            
            if fig is not None:
                plt.close(fig)
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Generated {len(models) * len(benchmark_pairs)} heatmap visualizations")
    print(f"All files saved to: {output_dir}/")
    print(f"\nFiles created:")
    for model in models:
        safe_model_name = model.replace('/', '_').replace(' ', '_').replace('.', '_')
        for _, pair_name in benchmark_pairs:
            print(f"  - {pair_name}_{safe_model_name}.png")


if __name__ == "__main__":
    main()