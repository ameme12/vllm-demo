"""
Country-Level Cross-Benchmark Correlation Analysis
For each country, computes Pearson correlation of its model accuracies across benchmarks
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


class CountryCorrelationAnalyzer:
    """
    Analyzes how individual countries' performance correlates across benchmarks
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
    
    def get_common_models(self, benchmark_names: List[str]) -> List[str]:
        """Get models that exist in ALL specified benchmarks."""
        if not benchmark_names:
            return []
        
        common = set(self.benchmarks[benchmark_names[0]].columns)
        for bench_name in benchmark_names[1:]:
            common = common.intersection(set(self.benchmarks[bench_name].columns))
        
        return sorted(list(common))
    
    def create_country_correlation_matrix_per_model(self,
                                                    benchmark_names: List[str],
                                                    model: str,
                                                    save_path: Optional[Path] = None,
                                                    min_benchmarks: int = 2):
        """
        Create correlation matrix showing how countries correlate across benchmarks for a specific model.
        
        For each pair of countries (for a given model):
        - Get accuracies for country_i across all benchmarks
        - Get accuracies for country_j across all benchmarks
        - Calculate Pearson correlation
        
        Args:
            benchmark_names: List of benchmarks to compare
            model: Model name to analyze
            save_path: Where to save the figure
            min_benchmarks: Minimum number of benchmarks with data
        """
        
        print(f"\n{'='*80}")
        print(f"Creating Country Correlation Matrix for Model: {model}")
        print(f"Benchmarks: {benchmark_names}")
        print(f"{'='*80}")
        
        # Get common countries
        common_countries = self.get_common_countries(benchmark_names)
        
        print(f"\nCommon countries: {len(common_countries)}")
        
        # Check if model exists in benchmarks
        benchmarks_with_model = []
        for bench_name in benchmark_names:
            df = self.benchmarks[bench_name]
            if model in df.columns:
                benchmarks_with_model.append(bench_name)
        
        print(f"Benchmarks with {model}: {benchmarks_with_model}")
        
        if len(benchmarks_with_model) < min_benchmarks:
            print(f"\n❌ Model {model} only in {len(benchmarks_with_model)} benchmark(s)")
            return None, None, None
        
        # For each country, collect accuracies across benchmarks for this model
        country_data = {}
        
        for country in common_countries:
            accuracies = []
            
            for bench_name in benchmarks_with_model:
                df = self.benchmarks[bench_name]
                acc = df.loc[country, model]
                
                if not pd.isna(acc):
                    accuracies.append(float(acc))
                else:
                    accuracies.append(np.nan)
            
            # Only keep countries with enough non-NaN data
            if sum(~np.isnan(accuracies)) >= min_benchmarks:
                country_data[country] = accuracies
        
        countries = sorted(country_data.keys())
        
        print(f"\nCountries with sufficient data: {len(countries)}")
        print(f"Countries: {countries[:10]}..." if len(countries) > 10 else f"Countries: {countries}")
        
        if len(countries) < 3:
            print(f"❌ Need at least 3 countries for correlation matrix!")
            return None, None, None
        
        # Compute correlation matrix
        n = len(countries)
        corr_matrix = np.zeros((n, n))
        pval_matrix = np.zeros((n, n))
        
        print(f"\nComputing correlations...")
        
        for i, country_i in enumerate(countries):
            for j, country_j in enumerate(countries):
                accs_i = np.array(country_data[country_i])
                accs_j = np.array(country_data[country_j])
                
                # Remove NaN pairs
                valid_mask = ~(np.isnan(accs_i) | np.isnan(accs_j))
                
                if sum(valid_mask) >= min_benchmarks:
                    valid_i = accs_i[valid_mask]
                    valid_j = accs_j[valid_mask]
                    
                    if i == j:
                        corr_matrix[i, j] = 1.0
                        pval_matrix[i, j] = 0.0
                    else:
                        r, p = pearsonr(valid_i, valid_j)
                        corr_matrix[i, j] = r
                        pval_matrix[i, j] = p
                else:
                    corr_matrix[i, j] = np.nan
                    pval_matrix[i, j] = np.nan
        
        # Create figure - 4x larger
        cell_size = 2.0  # Increased from 0.5 to 2.0 (4x larger)
        fig_width = max(14, n * cell_size + 3)
        fig_height = max(12, n * cell_size + 2)
        
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
                if not np.isnan(corr_matrix[i, j]):
                    r = corr_matrix[i, j]
                    p = pval_matrix[i, j]
                    
                    if i == j:
                        annot_labels[i, j] = '1.00'
                    else:
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
                    annot_labels[i, j] = ''
        
        # Adjust font size based on matrix size (larger for bigger plot)
        annot_fontsize = 12 if n > 30 else 14 if n > 20 else 16
        
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
            linewidths=1.0,  # Increased from 0.5
            linecolor='lightgray',
            ax=ax,
            annot_kws={'fontsize': annot_fontsize}
        )
        
        # Styling with larger fonts
        ax.set_xlabel('Country', fontsize=24, fontweight='bold')  # Increased from 12
        ax.set_ylabel('Country', fontsize=24, fontweight='bold')  # Increased from 12
        
        ax.set_title(
            f'Country Correlation Matrix\n'
            f'Model: {model}\n'
            f'({", ".join(benchmarks_with_model)}, {len(benchmarks_with_model)} benchmarks)',
            fontsize=28, fontweight='bold', pad=40  # Increased from 14
        )
        
        # Rotate labels with larger fonts
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)  # Increased from 8
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)  # Increased from 8
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved correlation matrix to {save_path}")
        
        # Print summary
        valid_corrs = corr_matrix[~np.isnan(corr_matrix) & (corr_matrix != 1.0)]
        
        if len(valid_corrs) > 0:
            print(f"\n{'='*60}")
            print("Summary Statistics:")
            print(f"{'='*60}")
            print(f"Model: {model}")
            print(f"Countries analyzed: {len(countries)}")
            print(f"Benchmarks: {benchmarks_with_model}")
            print(f"Mean correlation: {np.mean(valid_corrs):.3f}")
            print(f"Median correlation: {np.median(valid_corrs):.3f}")
            print(f"Std correlation: {np.std(valid_corrs):.3f}")
            print(f"Min correlation: {np.min(valid_corrs):.3f}")
            print(f"Max correlation: {np.max(valid_corrs):.3f}")
        
        return fig, df_corr, pval_matrix
    
    def create_country_correlation_matrix(self,
                                         benchmark_names: List[str],
                                         save_path: Optional[Path] = None,
                                         min_models: int = 2):
        """
        Create correlation matrix showing how countries correlate across benchmarks.
        Uses ALL models combined.
        
        For each pair of countries:
        - Get model accuracies for country_i across benchmarks
        - Get model accuracies for country_j across benchmarks  
        - Calculate Pearson correlation
        
        Args:
            benchmark_names: List of benchmarks to compare
            save_path: Where to save the figure
            min_models: Minimum number of common models needed
        """
        
        print(f"\n{'='*80}")
        print(f"Creating Country Correlation Matrix (All Models Combined)")
        print(f"Benchmarks: {benchmark_names}")
        print(f"{'='*80}")
        
        # Get common countries and models
        common_countries = self.get_common_countries(benchmark_names)
        common_models = self.get_common_models(benchmark_names)
        
        print(f"\nCommon countries: {len(common_countries)}")
        print(f"Common models: {len(common_models)}")
        print(f"Models: {common_models}")
        
        if len(common_models) < min_models:
            print(f"\n❌ Need at least {min_models} common models!")
            return None, None, None
        
        # For each country, collect accuracies across all benchmarks and models
        country_data = {}
        
        for country in common_countries:
            accuracies = []
            
            for bench_name in benchmark_names:
                df = self.benchmarks[bench_name]
                
                for model in common_models:
                    acc = df.loc[country, model]
                    if not pd.isna(acc):
                        accuracies.append(float(acc))
                    else:
                        accuracies.append(np.nan)
            
            # Only keep countries with enough non-NaN data
            if sum(~np.isnan(accuracies)) >= min_models * len(benchmark_names):
                country_data[country] = accuracies
        
        countries = sorted(country_data.keys())
        
        print(f"\nCountries with sufficient data: {len(countries)}")
        
        if len(countries) < 3:
            print(f"❌ Need at least 3 countries for correlation matrix!")
            return None, None, None
        
        # Compute correlation matrix
        n = len(countries)
        corr_matrix = np.zeros((n, n))
        pval_matrix = np.zeros((n, n))
        
        print(f"\nComputing correlations...")
        
        for i, country_i in enumerate(countries):
            for j, country_j in enumerate(countries):
                accs_i = np.array(country_data[country_i])
                accs_j = np.array(country_data[country_j])
                
                # Remove NaN pairs
                valid_mask = ~(np.isnan(accs_i) | np.isnan(accs_j))
                
                if sum(valid_mask) >= min_models:
                    valid_i = accs_i[valid_mask]
                    valid_j = accs_j[valid_mask]
                    
                    if i == j:
                        corr_matrix[i, j] = 1.0
                        pval_matrix[i, j] = 0.0
                    else:
                        r, p = pearsonr(valid_i, valid_j)
                        corr_matrix[i, j] = r
                        pval_matrix[i, j] = p
                else:
                    corr_matrix[i, j] = np.nan
                    pval_matrix[i, j] = np.nan
        
        # Create figure - 4x larger
        cell_size = 2.0  # Increased from 0.5 to 2.0 (4x larger)
        fig_width = max(14, n * cell_size + 3)
        fig_height = max(12, n * cell_size + 2)
        
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
                if not np.isnan(corr_matrix[i, j]):
                    r = corr_matrix[i, j]
                    p = pval_matrix[i, j]
                    
                    if i == j:
                        annot_labels[i, j] = '1.00'
                    else:
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
                    annot_labels[i, j] = ''
        
        # Adjust font size based on matrix size (larger for bigger plot)
        annot_fontsize = 14 if n > 30 else 16 if n > 20 else 18
        
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
            linewidths=1.0,  # Increased from 0.5
            linecolor='lightgray',
            ax=ax,
            annot_kws={'fontsize': annot_fontsize}
        )
        
        # Styling with larger fonts
        ax.set_xlabel('Country', fontsize=24, fontweight='bold')  # Increased from 12
        ax.set_ylabel('Country', fontsize=24, fontweight='bold')  # Increased from 12
        
        ax.set_title(
            f'Country Correlation Matrix Across Benchmarks\n'
            f'{", ".join(benchmark_names)}\n'
            f'(Pearson r of model accuracies, {len(common_models)} models × {len(benchmark_names)} benchmarks)',
            fontsize=28, fontweight='bold', pad=40  # Increased from 14
        )
        
        # Rotate labels with larger fonts
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=18)  # Increased from 9
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18)  # Increased from 9
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved correlation matrix to {save_path}")
        
        # Print summary
        valid_corrs = corr_matrix[~np.isnan(corr_matrix) & (corr_matrix != 1.0)]
        
        if len(valid_corrs) > 0:
            print(f"\n{'='*60}")
            print("Summary Statistics:")
            print(f"{'='*60}")
            print(f"Countries analyzed: {len(countries)}")
            print(f"Mean correlation: {np.mean(valid_corrs):.3f}")
            print(f"Median correlation: {np.median(valid_corrs):.3f}")
            print(f"Std correlation: {np.std(valid_corrs):.3f}")
            print(f"Min correlation: {np.min(valid_corrs):.3f}")
            print(f"Max correlation: {np.max(valid_corrs):.3f}")
        
        return fig, df_corr, pval_matrix
    
    def create_top_correlations_table(self,
                                     benchmark_names: List[str],
                                     top_n: int = 20,
                                     save_path: Optional[Path] = None):
        """
        Create table showing top N most correlated country pairs.
        """
        
        print(f"\n{'='*80}")
        print(f"Finding Top {top_n} Country Correlations")
        print(f"{'='*80}")
        
        common_countries = self.get_common_countries(benchmark_names)
        common_models = self.get_common_models(benchmark_names)
        
        # Collect country data
        country_data = {}
        
        for country in common_countries:
            accuracies = []
            
            for bench_name in benchmark_names:
                df = self.benchmarks[bench_name]
                
                for model in common_models:
                    acc = df.loc[country, model]
                    if not pd.isna(acc):
                        accuracies.append(float(acc))
                    else:
                        accuracies.append(np.nan)
            
            if sum(~np.isnan(accuracies)) >= 2:
                country_data[country] = accuracies
        
        countries = sorted(country_data.keys())
        
        # Compute correlations for all pairs
        correlations = []
        
        for i, country_i in enumerate(countries):
            for j in range(i + 1, len(countries)):
                country_j = countries[j]
                
                accs_i = np.array(country_data[country_i])
                accs_j = np.array(country_data[country_j])
                
                valid_mask = ~(np.isnan(accs_i) | np.isnan(accs_j))
                
                if sum(valid_mask) >= 2:
                    valid_i = accs_i[valid_mask]
                    valid_j = accs_j[valid_mask]
                    
                    r, p = pearsonr(valid_i, valid_j)
                    
                    correlations.append({
                        'Country 1': country_i,
                        'Country 2': country_j,
                        'Pearson r': r,
                        'p-value': p,
                        'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '',
                        'n': sum(valid_mask)
                    })
        
        # Sort by correlation (absolute value)
        correlations.sort(key=lambda x: abs(x['Pearson r']), reverse=True)
        
        # Take top N
        top_correlations = correlations[:top_n]
        
        df_top = pd.DataFrame(top_correlations)
        
        if save_path:
            df_top.to_csv(save_path, index=False, float_format='%.4f')
            print(f"✓ Saved top correlations to {save_path}")
        
        print(f"\nTop {top_n} Country Correlations:")
        print(df_top.to_string(index=False))
        
        return df_top


def main():
    """Main analysis script"""
    
    analyzer = CountryCorrelationAnalyzer()
    
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
            analyzer.load_benchmark(benchmark_name, str(csv_path))
            loaded_benchmarks.append(benchmark_name)
        else:
            print(f"⚠️  File not found: {csv_path}")
    
    if len(loaded_benchmarks) < 2:
        print("\n❌ Need at least 2 benchmarks to compare!")
        return
    
    # Create output directory
    output_dir = Path("country_correlation_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Get all unique models
    all_models = set()
    for bench_name in loaded_benchmarks:
        df = analyzer.benchmarks[bench_name]
        all_models.update(df.columns)
    
    all_models = sorted(all_models)
    
    print(f"\n{'='*80}")
    print(f"ALL MODELS FOUND: {all_models}")
    print(f"{'='*80}")
    
    # Create country correlation matrix for each model
    print(f"\n{'='*80}")
    print("CREATING PER-MODEL COUNTRY CORRELATION MATRICES")
    print(f"{'='*80}")
    
    for model in all_models:
        print(f"\n{'='*80}")
        print(f"Processing model: {model}")
        print(f"{'='*80}")
        
        safe_model_name = model.replace('/', '_').replace(' ', '_').replace('.', '_')
        
        result = analyzer.create_country_correlation_matrix_per_model(
            loaded_benchmarks,
            model,
            save_path=output_dir / f'country_correlation_{safe_model_name}.png',
            min_benchmarks=2
        )
        
        if result[0] is not None:
            fig, corr_df, pval_matrix = result
            
            # Save to Excel
            excel_path = output_dir / f'country_correlation_{safe_model_name}.xlsx'
            
            with pd.ExcelWriter(excel_path) as writer:
                corr_df.to_excel(writer, sheet_name='Correlations')
                pd.DataFrame(pval_matrix, index=corr_df.index, columns=corr_df.columns).to_excel(
                    writer, sheet_name='P-values'
                )
            
            print(f"✓ Saved Excel file to {excel_path}")
            plt.close(fig)
    
    # Also create combined analysis (all models together)
    print(f"\n{'='*80}")
    print("CREATING COMBINED COUNTRY CORRELATION MATRIX (ALL MODELS)")
    print(f"{'='*80}")
    
    result = analyzer.create_country_correlation_matrix(
        loaded_benchmarks,
        save_path=output_dir / 'country_correlation_all_models.png',
        min_models=2
    )
    
    if result[0] is not None:
        fig, corr_df, pval_matrix = result
        
        # Save to Excel
        with pd.ExcelWriter(output_dir / 'country_correlation_all_models.xlsx') as writer:
            corr_df.to_excel(writer, sheet_name='Correlations')
            pd.DataFrame(pval_matrix, index=corr_df.index, columns=corr_df.columns).to_excel(
                writer, sheet_name='P-values'
            )
        
        print(f"✓ Saved Excel file to {output_dir / 'country_correlation_all_models.xlsx'}")
        plt.close(fig)
    
    # Create top correlations table (for combined analysis)
    print(f"\n{'='*80}")
    print("CREATING TOP CORRELATIONS TABLE")
    print(f"{'='*80}")
    
    df_top = analyzer.create_top_correlations_table(
        loaded_benchmarks,
        top_n=20,
        save_path=output_dir / 'top_country_correlations.csv'
    )
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"All files saved to: {output_dir}/")


if __name__ == "__main__":
    main()