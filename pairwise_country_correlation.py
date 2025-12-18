"""
Pairwise Benchmark Comparison Analysis
Compares country performance patterns between pairs of benchmarks
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PairwiseBenchmarkAnalyzer:
    """
    Analyzes country correlations between pairs of benchmarks
    """
    
    def __init__(self):
        """Initialize the analyzer"""
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
    
    def create_pairwise_country_correlation(self,
                                           bench1_name: str,
                                           bench2_name: str,
                                           model: str,
                                           save_path: Optional[Path] = None):
        """
        Create correlation matrix comparing countries between two benchmarks.
        
        For each pair of countries:
        - Get accuracy on benchmark1
        - Get accuracy on benchmark2
        - Calculate Pearson correlation (with n=2 data points)
        
        Args:
            bench1_name: First benchmark name
            bench2_name: Second benchmark name
            model: Model name to analyze
            save_path: Where to save the figure
        """
        
        print(f"\n{'='*80}")
        print(f"Pairwise Analysis: {bench1_name} vs {bench2_name}")
        print(f"Model: {model}")
        print(f"{'='*80}")
        
        # Get common countries between these two benchmarks
        common_countries = self.get_common_countries([bench1_name, bench2_name])
        
        print(f"\nCommon countries: {len(common_countries)}")
        print(f"Countries: {common_countries}")
        
        if len(common_countries) < 3:
            print(f"❌ Need at least 3 countries for meaningful correlation!")
            return None, None, None
        
        # Check if model exists in both benchmarks
        bench1_df = self.benchmarks[bench1_name]
        bench2_df = self.benchmarks[bench2_name]
        
        if model not in bench1_df.columns:
            print(f"❌ Model '{model}' not found in {bench1_name}")
            return None, None, None
        
        if model not in bench2_df.columns:
            print(f"❌ Model '{model}' not found in {bench2_name}")
            return None, None, None
        
        # Collect data for each country (2 data points: bench1, bench2)
        country_data = {}
        
        for country in common_countries:
            acc1 = bench1_df.loc[country, model]
            acc2 = bench2_df.loc[country, model]
            
            if not pd.isna(acc1) and not pd.isna(acc2):
                country_data[country] = [float(acc1), float(acc2)]
        
        countries = sorted(country_data.keys())
        
        print(f"\nCountries with valid data: {len(countries)}")
        
        if len(countries) < 3:
            print(f"❌ Need at least 3 countries with valid data!")
            return None, None, None
        
        # Print the data
        print(f"\nData collected:")
        for country in countries:
            print(f"  {country}: {bench1_name}={country_data[country][0]:.2f}, "
                  f"{bench2_name}={country_data[country][1]:.2f}")
        
        # Compute correlation matrix (n x n countries)
        n = len(countries)
        corr_matrix = np.zeros((n, n))
        pval_matrix = np.zeros((n, n))
        
        print(f"\nComputing Pearson correlations...")
        print("⚠️  Note: Each correlation uses only n=2 data points (limited statistical power)")
        
        for i, country_i in enumerate(countries):
            for j, country_j in enumerate(countries):
                data_i = np.array(country_data[country_i])  # [bench1_acc, bench2_acc]
                data_j = np.array(country_data[country_j])  # [bench1_acc, bench2_acc]
                
                if i == j:
                    corr_matrix[i, j] = 1.0
                    pval_matrix[i, j] = 0.0
                else:
                    # Compute actual Pearson correlation using scipy
                    # With n=2, this will always give ±1.0, but we compute it properly
                    try:
                        r, p = pearsonr(data_i, data_j)
                        corr_matrix[i, j] = r
                        pval_matrix[i, j] = p
                    except:
                        # Handle edge cases (e.g., zero variance)
                        corr_matrix[i, j] = np.nan
                        pval_matrix[i, j] = np.nan
        
        # Create figure
        fig_width = max(10, n * 0.7 + 3)
        fig_height = max(8, n * 0.7 + 2)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create DataFrame
        df_corr = pd.DataFrame(
            corr_matrix,
            index=countries,
            columns=countries
        )
        
        # Create annotations
        annot_labels = np.empty_like(corr_matrix, dtype=object)
        for i in range(n):
            for j in range(n):
                if i == j:
                    annot_labels[i, j] = '1.00'
                else:
                    if not np.isnan(corr_matrix[i, j]):
                        r = corr_matrix[i, j]
                        p = pval_matrix[i, j]
                        
                        # With n=2, p-values are often undefined/unreliable
                        # but we show them for completeness
                        if not np.isnan(p):
                            if p < 0.001:
                                sig = '***'
                            elif p < 0.01:
                                sig = '**'
                            elif p < 0.05:
                                sig = '*'
                            else:
                                sig = ''
                            annot_labels[i, j] = f'{r:.2f}{sig}'
                        else:
                            annot_labels[i, j] = f'{r:.2f}'
                    else:
                        annot_labels[i, j] = ''
        
        # Plot heatmap
        sns.heatmap(
            df_corr,
            annot=annot_labels,
            fmt='',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Pearson Correlation (r)'},
            linewidths=1.0,
            linecolor='lightgray',
            ax=ax,
            annot_kws={'fontsize': 10}
        )
        
        # Styling
        ax.set_xlabel('Country', fontsize=12, fontweight='bold')
        ax.set_ylabel('Country', fontsize=12, fontweight='bold')
        
        ax.set_title(
            f'Country Correlation Matrix\n'
            f'{bench1_name} vs {bench2_name}\n'
            f'Model: {model} (n=2 data points per country pair)',
            fontsize=14, fontweight='bold', pad=20
        )
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved correlation matrix to {save_path}")
        
        # Print summary
        valid_corrs = corr_matrix[~np.isnan(corr_matrix) & (corr_matrix != 1.0)]
        
        print(f"\n{'='*60}")
        print("Summary Statistics:")
        print(f"{'='*60}")
        print(f"Benchmark Pair: {bench1_name} vs {bench2_name}")
        print(f"Model: {model}")
        print(f"Countries analyzed: {len(countries)}")
        print(f"Data points per correlation: 2 (one from each benchmark)")
        
        if len(valid_corrs) > 0:
            print(f"\nCorrelation Statistics:")
            print(f"  Mean correlation: {np.mean(valid_corrs):.3f}")
            print(f"  Median correlation: {np.median(valid_corrs):.3f}")
            print(f"  Std correlation: {np.std(valid_corrs):.3f}")
            print(f"  Min correlation: {np.min(valid_corrs):.3f}")
            print(f"  Max correlation: {np.max(valid_corrs):.3f}")
        
        print(f"\nInterpretation:")
        print(f"  Positive r: Countries show similar relative performance on both benchmarks")
        print(f"  Negative r: Countries show inverse relative performance patterns")
        print(f"\n⚠️  IMPORTANT: With only n=2 data points per country pair,")
        print(f"             these correlations have very limited statistical power")
        print(f"             and should be interpreted with extreme caution.")
        
        # Count positive vs negative correlations
        if len(valid_corrs) > 0:
            positive_corrs = np.sum(valid_corrs > 0)
            negative_corrs = np.sum(valid_corrs < 0)
            
            print(f"\nCountry pairs with positive correlation: {positive_corrs}")
            print(f"Country pairs with negative correlation: {negative_corrs}")
        
        return fig, df_corr, pval_matrix


def main():
    """Main analysis script"""
    
    analyzer = PairwiseBenchmarkAnalyzer()
    
    # Define paths
    csv_dir = Path("/home/mila/r/ramesana/projects/vllm-demo/csv_tables")
    
    benchmark_files = {
        'BLEND': csv_dir / 'blend' / 'blend_accuracy_by_country.csv',
        'CulturalBench': csv_dir / 'culturalbench' / 'culturalbench_accuracy_by_country.csv',
        #'GeoMLAMA': csv_dir / 'geomlama' / 'geomlama_accuracy_by_country.csv',
    }
    
    # Load benchmarks
    print("="*80)
    print("LOADING BENCHMARKS")
    print("="*80)
    
    loaded_benchmarks = []
    for benchmark_name, csv_path in benchmark_files.items():
        if csv_path.exists():
            analyzer.load_benchmark(benchmark_name, str(csv_path))
            loaded_benchmarks.append(benchmark_name)
        else:
            print(f"⚠️  File not found: {csv_path}")
    
    if len(loaded_benchmarks) < 2:
        print("\n❌ Need at least 2 benchmarks to compare!")
        return
    
    # Create output directory
    output_dir = Path("pairwise_benchmark_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Define models to analyze
    models = ['Llama 3.2-3B', 'Qwen 2.5-3B']
    
    # Define benchmark pairs
    benchmark_pairs = [
        ('BLEND', 'CulturalBench'),
        ('BLEND', 'GeoMLAMA'),
        ('CulturalBench', 'GeoMLAMA')
    ]
    
    print(f"\n{'='*80}")
    print("CREATING PAIRWISE BENCHMARK COMPARISONS")
    print(f"{'='*80}")
    print(f"Models: {models}")
    print(f"Benchmark pairs: {benchmark_pairs}")
    print(f"Total graphs to generate: {len(models) * len(benchmark_pairs)}")
    
    # Generate all pairwise comparisons
    for model in models:
        print(f"\n{'='*80}")
        print(f"ANALYZING MODEL: {model}")
        print(f"{'='*80}")
        
        for bench1, bench2 in benchmark_pairs:
            print(f"\n{'='*60}")
            print(f"Pair: {bench1} vs {bench2}")
            print(f"{'='*60}")
            
            safe_model_name = model.replace('/', '_').replace(' ', '_').replace('.', '_')
            safe_bench1 = bench1.replace(' ', '_')
            safe_bench2 = bench2.replace(' ', '_')
            
            filename = f'pairwise_{safe_bench1}_vs_{safe_bench2}_{safe_model_name}.png'
            
            result = analyzer.create_pairwise_country_correlation(
                bench1,
                bench2,
                model,
                save_path=output_dir / filename
            )
            
            if result[0] is not None:
                fig, corr_df, pval_matrix = result
                
                # Save to Excel
                excel_filename = f'pairwise_{safe_bench1}_vs_{safe_bench2}_{safe_model_name}.xlsx'
                excel_path = output_dir / excel_filename
                
                with pd.ExcelWriter(excel_path) as writer:
                    corr_df.to_excel(writer, sheet_name='Correlations')
                
                print(f"✓ Saved Excel file to {excel_path}")
                plt.close(fig)
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Generated {len(models) * len(benchmark_pairs)} correlation matrices")
    print(f"All files saved to: {output_dir}/")
    print(f"\n{'='*80}")
    print("IMPORTANT NOTES:")
    print(f"{'='*80}")
    print("1. Each correlation uses only n=2 data points (one per benchmark)")
    print("2. Pearson correlations are properly computed using scipy.stats.pearsonr")
    print("3. With n=2, statistical power is extremely limited")
    print("4. P-values should be interpreted with caution given small sample size")
    print("5. These results provide directional insights but limited statistical validity")
    print("\nFor more robust analysis, see the 3-benchmark country correlation matrices")
    print("which use n=3 data points per country pair, providing more reliable correlations.")


if __name__ == "__main__":
    main()