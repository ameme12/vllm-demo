"""
Cross-Benchmark Correlation Analysis
Compares benchmark average accuracies across countries for each model
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


class BenchmarkComparisonVisualizer:
    """
    Creates correlation matrices comparing benchmark performance across countries
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
            # DLAMA countries
            'andorra': 'Andorra',
            'australia': 'Australia',
            'austria': 'Austria',
            'belgium': 'Belgium',
            'brazil': 'Brazil',
            'cameroon': 'Cameroon',
            'canada': 'Canada',
            'comoros': 'Comoros',
            'egypt': 'Egypt',
            'france': 'France',
            'germany': 'Germany',
            'iraq': 'Iraq',
            'ireland': 'Ireland',
            'italy': 'Italy',
            'japan': 'Japan',
            'jordan': 'Jordan',
            'kuwait': 'Kuwait',
            'lebanon': 'Lebanon',
            'libya': 'Libya',
            'liechtenstein': 'Liechtenstein',
            'luxembourg': 'Luxembourg',
            'malaysia': 'Malaysia',
            'mauritania': 'Mauritania',
            'monaco': 'Monaco',
            'mongolia': 'Mongolia',
            'morocco': 'Morocco',
            'myanmar': 'Myanmar',
            'netherlands': 'Netherlands',
            'north korea': 'North Korea',
            'philippines': 'Philippines',
            'portugal': 'Portugal',
            'san marino': 'San Marino',
            'singapore': 'Singapore',
            'somalia': 'Somalia',
            'south sudan': 'South Sudan',
            'switzerland': 'Switzerland',
            'syria': 'Syria',
            'taiwan': 'Taiwan',
            'thailand': 'Thailand',
            'tunisia': 'Tunisia',
            'uae': 'UAE',
            'vietnam': 'Vietnam',
        }
    
    def standardize_country_name(self, country: str) -> str:
        """Standardize a country name"""
        country_lower = country.strip().lower()
        if country_lower in self.country_mappings:
            return self.country_mappings[country_lower]
        return country.strip().title()
    
    def load_benchmark(self, 
                       benchmark_name: str, 
                       csv_path: str) -> pd.DataFrame:
        """Load a benchmark CSV file."""
        print(f"\n{'='*60}")
        print(f"Loading: {benchmark_name}")
        print(f"{'='*60}")
        print(f"Path: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Original shape: {df.shape}")
        print(f"Original columns: {df.columns.tolist()}")
        
        country_col = df.columns[0]
        df[country_col] = df[country_col].apply(self.standardize_country_name)
        df = df.set_index(country_col)
        df.index.name = 'country'
        
        if df.index.duplicated().any():
            duplicates = df.index[df.index.duplicated()].unique()
            print(f"⚠️  Warning: Found duplicate countries: {list(duplicates)}")
            print(f"   Keeping first occurrence of each")
            df = df[~df.index.duplicated(keep='first')]
        
        # Standardize model names
        df.columns = [self.standardize_model_name(col) for col in df.columns]
        print(f"Standardized model columns: {df.columns.tolist()}")
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"Countries: {len(df)}")
        print(f"Models: {df.columns.tolist()}")
        print(f"Countries: {sorted(df.index.tolist())}")
        
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
    
    def create_benchmark_correlation_matrix(self,
                                           benchmark_names: List[str],
                                           model: str,
                                           save_path: Optional[Path] = None):
        """
        Create correlation matrix comparing benchmarks across countries for a specific model.
        
        Matrix format:
                        BLEND    CulturalBench    GeoMLAMA
        BLEND           1.00         0.85            0.72
        CulturalBench   0.85         1.00            0.91
        GeoMLAMA        0.72         0.91            1.00
        
        Each cell (i,j) = Pearson correlation between:
        - Benchmark i's country accuracies for this model
        - Benchmark j's country accuracies for this model
        
        Args:
            benchmark_names: List of benchmarks to compare
            model: Model name to analyze
            save_path: Where to save the figure
        """
        
        print(f"\n{'='*80}")
        print(f"Creating Benchmark Correlation Matrix for: {model}")
        print(f"{'='*80}")
        
        # Get common countries
        common_countries = self.get_common_countries(benchmark_names)
        
        print(f"\nCommon countries across all benchmarks ({len(common_countries)}):")
        print(f"  {common_countries}")
        
        if len(common_countries) < 3:
            print(f"\n❌ Need at least 3 common countries for meaningful correlation!")
            return None, None, None
        
        # Check if model exists in all benchmarks
        for bench_name in benchmark_names:
            df = self.benchmarks[bench_name]
            if model not in df.columns:
                print(f"\n❌ Model '{model}' not found in {bench_name}")
                print(f"   Available models: {df.columns.tolist()}")
                return None, None, None
        
        # Extract accuracies for each benchmark
        benchmark_accuracies = {}
        
        for bench_name in benchmark_names:
            df = self.benchmarks[bench_name]
            
            # Get accuracies for common countries only
            accs = []
            for country in common_countries:
                acc = df.loc[country, model]
                if pd.isna(acc):
                    print(f"⚠️  {bench_name} - {country}: NaN value for {model}")
                    accs.append(np.nan)
                else:
                    accs.append(float(acc))
            
            benchmark_accuracies[bench_name] = accs
            print(f"\n{bench_name} accuracies for {model}:")
            for country, acc in zip(common_countries, accs):
                print(f"  {country}: {acc}")
        
        # Check for NaN values
        for bench_name, accs in benchmark_accuracies.items():
            if any(pd.isna(accs)):
                print(f"\n❌ {bench_name} has NaN values - cannot compute correlations")
                return None, None, None
        
        # Compute correlation matrix
        n_benchmarks = len(benchmark_names)
        corr_matrix = np.zeros((n_benchmarks, n_benchmarks))
        pval_matrix = np.zeros((n_benchmarks, n_benchmarks))
        
        print(f"\n{'='*60}")
        print("Computing Correlations:")
        print(f"{'='*60}")
        
        for i, bench_i in enumerate(benchmark_names):
            for j, bench_j in enumerate(benchmark_names):
                accs_i = benchmark_accuracies[bench_i]
                accs_j = benchmark_accuracies[bench_j]
                
                if i == j:
                    # Diagonal: perfect correlation with self
                    corr_matrix[i, j] = 1.0
                    pval_matrix[i, j] = 0.0
                else:
                    # Compute Pearson correlation
                    r, p = pearsonr(accs_i, accs_j)
                    corr_matrix[i, j] = r
                    pval_matrix[i, j] = p
                    
                    print(f"{bench_i} vs {bench_j}:")
                    print(f"  r = {r:.4f}, p = {p:.4f}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create DataFrame
        df_corr = pd.DataFrame(
            corr_matrix,
            index=benchmark_names,
            columns=benchmark_names
        )
        
        # Create annotations with significance
        annot_labels = np.empty_like(corr_matrix, dtype=object)
        for i in range(n_benchmarks):
            for j in range(n_benchmarks):
                r = corr_matrix[i, j]
                p = pval_matrix[i, j]
                
                if i == j:
                    annot_labels[i, j] = '1.00'
                else:
                    # Significance stars
                    if p < 0.001:
                        sig = '***'
                    elif p < 0.01:
                        sig = '**'
                    elif p < 0.05:
                        sig = '*'
                    else:
                        sig = ''
                    
                    annot_labels[i, j] = f'{r:.3f}{sig}'
        
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
            linewidths=2,
            linecolor='white',
            ax=ax,
            annot_kws={'fontsize': 14, 'fontweight': 'bold'},
            square=True
        )
        
        # Styling
        ax.set_xlabel('', fontsize=14, fontweight='bold')
        ax.set_ylabel('', fontsize=14, fontweight='bold')
        
        ax.set_title(
            f'Benchmark Correlation Matrix\n'
            f'Model: {model}\n'
            f'(Pearson r across {len(common_countries)} countries)',
            fontsize=16, fontweight='bold', pad=20
        )
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved correlation matrix to {save_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"{'='*60}")
        print(f"Model: {model}")
        print(f"Countries analyzed: {len(common_countries)}")
        print(f"Countries: {common_countries}")
        
        # Off-diagonal correlations
        off_diag = []
        for i in range(n_benchmarks):
            for j in range(i + 1, n_benchmarks):
                off_diag.append(corr_matrix[i, j])
        
        if off_diag:
            print(f"\nBenchmark correlations:")
            print(f"  Mean: {np.mean(off_diag):.3f}")
            print(f"  Range: [{np.min(off_diag):.3f}, {np.max(off_diag):.3f}]")
        
        return fig, df_corr, pval_matrix


def main():
    """Main analysis script"""
    
    viz = BenchmarkComparisonVisualizer()
    
    # Define paths
    csv_dir = Path("/home/mila/r/ramesana/projects/vllm-demo/csv_tables")
    
    benchmark_files = {
        'CulturalBench': csv_dir / 'culturalbench' / 'culturalbench_accuracy_by_country.csv',
        'BLEND': csv_dir / 'blend' / 'blend_accuracy_by_country.csv',
        'GeoMLAMA': csv_dir / 'geomlama' / 'geomlama_accuracy_by_country.csv',
        #'DLAMA': csv_dir / 'dlama' / 'dlama_accuracy_by_country.csv',
    }
    
    # Load benchmarks
    print("="*80)
    print("LOADING BENCHMARKS")
    print("="*80)
    
    loaded_benchmarks = []
    for benchmark_name, csv_path in benchmark_files.items():
        if csv_path.exists():
            viz.load_benchmark(benchmark_name, str(csv_path))
            loaded_benchmarks.append(benchmark_name)
        else:
            print(f"⚠️  File not found: {csv_path}")
    
    if len(loaded_benchmarks) < 2:
        print("\n❌ Need at least 2 benchmarks to compare!")
        return
    
    # Get all unique models across all benchmarks
    all_models = set()
    for bench_name in loaded_benchmarks:
        df = viz.benchmarks[bench_name]
        all_models.update(df.columns)
    
    all_models = sorted(all_models)
    
    print(f"\n{'='*80}")
    print(f"ALL MODELS FOUND: {all_models}")
    print(f"{'='*80}")
    
    # Create output directory
    output_dir = Path("correlation_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Create correlation matrix for each model
    print(f"\n{'='*80}")
    print("CREATING BENCHMARK CORRELATION MATRICES")
    print(f"{'='*80}")
    
    results = {}
    
    for model in all_models:
        print(f"\n{'='*80}")
        print(f"Processing model: {model}")
        print(f"{'='*80}")
        
        # Check which benchmarks have this model
        benchmarks_with_model = []
        for bench_name in loaded_benchmarks:
            if model in viz.benchmarks[bench_name].columns:
                benchmarks_with_model.append(bench_name)
        
        print(f"Benchmarks with {model}: {benchmarks_with_model}")
        
        if len(benchmarks_with_model) < 2:
            print(f"⚠️  Skipping {model}: only in {len(benchmarks_with_model)} benchmark(s)")
            continue
        
        # Create correlation matrix
        safe_model_name = model.replace('/', '_').replace(' ', '_').replace('.', '_')
        
        result = viz.create_benchmark_correlation_matrix(
            benchmarks_with_model,
            model,
            save_path=output_dir / f'benchmark_correlation_{safe_model_name}.png'
        )
        
        if result[0] is not None:
            fig, corr_df, pval_matrix = result
            results[model] = result
            
            # Save to Excel
            excel_path = output_dir / f'benchmark_correlation_{safe_model_name}.xlsx'
            
            with pd.ExcelWriter(excel_path) as writer:
                corr_df.to_excel(writer, sheet_name='Correlations')
                pd.DataFrame(pval_matrix, index=corr_df.index, columns=corr_df.columns).to_excel(
                    writer, sheet_name='P-values'
                )
            
            print(f"✓ Saved Excel file to {excel_path}")
            plt.close(fig)
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated {len(results)} correlation matrices")
    print(f"All files saved to: {output_dir}/")


if __name__ == "__main__":
    main()