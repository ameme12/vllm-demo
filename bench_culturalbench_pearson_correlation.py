"""
Benchmark Alignment Analysis
Calculates Pearson and Spearman correlations between two benchmarks for a single model
to determine if they measure the same underlying construct.

Usage:
    # Option 1: Using exact column names from CSV
    python benchmark_alignment.py --benchmark1_csv path/to/blend.csv \\
                                   --benchmark2_csv path/to/culturalbench.csv \\
                                   --model "Llama 3.2-3B" \\
                                   --benchmark1_name BLEND \\
                                   --benchmark2_name CulturalBench \\
                                   --output_dir results/alignment
    
    # Option 2: Using simplified model names (script will try to match)
    python benchmark_alignment.py --benchmark1_csv path/to/blend.csv \\
                                   --benchmark2_csv path/to/culturalbench.csv \\
                                   --model llama3b \\
                                   --benchmark1_name BLEND \\
                                   --benchmark2_name CulturalBench
    
    # Note: Model names can be:
    #   - Exact match: "Llama 3.2-3B", "Qwen 2.5-3B"
    #   - Simplified: llama3b, qwen2_5b, qwen2.5b
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import argparse
import sys


class BenchmarkAlignmentAnalyzer:
    def __init__(self, benchmark1_csv, benchmark2_csv, model_name, 
                 benchmark1_name="Benchmark1", benchmark2_name="Benchmark2",
                 country_col1=None, country_col2=None, output_dir="benchmark_alignment"):
        """
        Initialize the benchmark alignment analyzer
        
        Args:
            benchmark1_csv: Path to first benchmark CSV file
            benchmark2_csv: Path to second benchmark CSV file
            model_name: Name of the model to analyze (e.g., 'llama3b', 'qwen2_5b')
            benchmark1_name: Display name for first benchmark
            benchmark2_name: Display name for second benchmark
            country_col1: Name of the column containing country names in benchmark1 (auto-detect if None)
            country_col2: Name of the column containing country names in benchmark2 (auto-detect if None)
            output_dir: Directory to save output files
        """
        self.benchmark1_csv = Path(benchmark1_csv)
        self.benchmark2_csv = Path(benchmark2_csv)
        self.model_name = model_name.lower()
        self.benchmark1_name = benchmark1_name
        self.benchmark2_name = benchmark2_name
        self.country_col1 = country_col1
        self.country_col2 = country_col2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df1 = None
        self.df2 = None
        self.merged_df = None
        self.common_countries = None
        
    def find_country_column(self, df, benchmark_name):
        """Find the column containing country names"""
        # Try common column names
        possible_names = ['country', 'culture', 'culture_display', 'Country', 'Culture', 'region']
        
        for name in possible_names:
            if name in df.columns:
                print(f"   Found country column in {benchmark_name}: '{name}'")
                return name
        
        # If not found, look for non-numeric columns
        non_numeric = [col for col in df.columns if df[col].dtype == 'object']
        if len(non_numeric) == 1:
            print(f"   Found country column in {benchmark_name}: '{non_numeric[0]}'")
            return non_numeric[0]
        
        print(f"\n   ❌ ERROR: Could not find country column in {benchmark_name}")
        print(f"   Available columns: {list(df.columns)}")
        print(f"   Please specify using --country_col1 or --country_col2")
        sys.exit(1)
    
    def load_data(self):
        """Load CSV files and display their structure"""
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        
        # Load first benchmark
        print(f"\n1. Loading {self.benchmark1_name} from: {self.benchmark1_csv}")
        if not self.benchmark1_csv.exists():
            print(f"   ❌ ERROR: File not found!")
            sys.exit(1)
        
        self.df1 = pd.read_csv(self.benchmark1_csv)
        print(f"   ✓ Loaded: {self.df1.shape[0]} rows, {self.df1.shape[1]} columns")
        print(f"   Columns: {list(self.df1.columns)}")
        
        # Load second benchmark
        print(f"\n2. Loading {self.benchmark2_name} from: {self.benchmark2_csv}")
        if not self.benchmark2_csv.exists():
            print(f"   ❌ ERROR: File not found!")
            sys.exit(1)
            
        self.df2 = pd.read_csv(self.benchmark2_csv)
        print(f"   ✓ Loaded: {self.df2.shape[0]} rows, {self.df2.shape[1]} columns")
        print(f"   Columns: {list(self.df2.columns)}")
        
    def find_model_accuracy_column(self, df, benchmark_name, country_col_name):
        """
        Find the accuracy column for the specified model
        
        Looks for patterns like:
        - 'llama3b_accuracy', 'llama_3b_accuracy'
        - 'Llama 3.2-3B', 'Qwen 2.5-3B' (exact model names)
        - 'accuracy' (if only one accuracy column)
        
        Args:
            df: DataFrame to search
            benchmark_name: Name of benchmark for error messages
            country_col_name: Name of country column to exclude from search
        """
        # First, try exact match with common model name formats
        exact_matches = {
            'llama3b': ['Llama 3.2-3B', 'Llama 3B', 'llama3b', 'Llama3B'],
            'llama_3b': ['Llama 3.2-3B', 'Llama 3B', 'llama3b', 'Llama3B'],
            'qwen2_5b': ['Qwen 2.5-3B', 'Qwen 2.5B', 'qwen2_5b', 'Qwen2.5B'],
            'qwen2.5b': ['Qwen 2.5-3B', 'Qwen 2.5B', 'qwen2_5b', 'Qwen2.5B'],
        }
        
        # Try exact matches first
        model_key = self.model_name.lower().replace('_', '').replace('.', '').replace('-', '')
        
        for key, possible_names in exact_matches.items():
            key_normalized = key.replace('_', '').replace('.', '').replace('-', '')
            if model_key == key_normalized:
                for name in possible_names:
                    if name in df.columns:
                        print(f"   Found model column in {benchmark_name}: '{name}'")
                        return name
        
        # Normalize column names for searching
        columns_lower = {col: col.lower().replace(' ', '_').replace('-', '_').replace('.', '') 
                        for col in df.columns}
        
        # Search patterns
        search_patterns = [
            self.model_name.replace('-', '_').replace('.', ''),
            self.model_name.replace('_', '').replace('-', '').replace('.', ''),
            self.model_name,
        ]
        
        # Try to find matching column
        for col, col_normalized in columns_lower.items():
            for pattern in search_patterns:
                pattern_normalized = pattern.lower().replace('-', '_').replace('.', '')
                if pattern_normalized in col_normalized:
                    print(f"   Found model column in {benchmark_name}: '{col}'")
                    return col
        
        # If no match, look for any numeric column that's not the country column
        numeric_cols = [col for col in df.columns 
                       if col.lower() != country_col_name.lower() 
                       and df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        if len(numeric_cols) == 1:
            print(f"   Found single numeric column in {benchmark_name}: '{numeric_cols[0]}'")
            return numeric_cols[0]
        elif len(numeric_cols) == 2:
            # If there are 2 numeric columns, try to pick the right one based on model name
            for col in numeric_cols:
                col_lower = col.lower()
                if 'llama' in self.model_name.lower() and 'llama' in col_lower:
                    print(f"   Found model column in {benchmark_name}: '{col}'")
                    return col
                elif 'qwen' in self.model_name.lower() and 'qwen' in col_lower:
                    print(f"   Found model column in {benchmark_name}: '{col}'")
                    return col
        
        # Show available columns if not found
        print(f"\n   ❌ ERROR: Could not find accuracy column for model '{self.model_name}' in {benchmark_name}")
        print(f"   Available columns: {list(df.columns)}")
        print(f"   Numeric columns: {numeric_cols}")
        print(f"   Please check that:")
        print(f"     1. The model name '{self.model_name}' is correct")
        print(f"     2. The CSV has a column matching the model name")
        print(f"   Try using --model 'Llama 3.2-3B' or --model 'Qwen 2.5-3B' (exact match)")
        sys.exit(1)
    
    def prepare_data(self):
        """Extract relevant columns and find common countries"""
        print("\n" + "="*70)
        print("PREPARING DATA")
        print("="*70)
        
        # Auto-detect country columns if not specified
        if self.country_col1 is None:
            self.country_col1 = self.find_country_column(self.df1, self.benchmark1_name)
        if self.country_col2 is None:
            self.country_col2 = self.find_country_column(self.df2, self.benchmark2_name)
        
        # Find accuracy columns for the model
        acc_col1 = self.find_model_accuracy_column(self.df1, self.benchmark1_name, self.country_col1)
        acc_col2 = self.find_model_accuracy_column(self.df2, self.benchmark2_name, self.country_col2)
        
        # Check if country columns exist
        if self.country_col1 not in self.df1.columns:
            print(f"\n❌ ERROR: Country column '{self.country_col1}' not found in {self.benchmark1_name}")
            print(f"Available columns: {list(self.df1.columns)}")
            sys.exit(1)
        if self.country_col2 not in self.df2.columns:
            print(f"\n❌ ERROR: Country column '{self.country_col2}' not found in {self.benchmark2_name}")
            print(f"Available columns: {list(self.df2.columns)}")
            sys.exit(1)
        
        # Extract relevant data
        df1_subset = self.df1[[self.country_col1, acc_col1]].copy()
        df2_subset = self.df2[[self.country_col2, acc_col2]].copy()
        
        # Rename columns for clarity
        df1_subset.columns = ['country', self.benchmark1_name]
        df2_subset.columns = ['country', self.benchmark2_name]
        
        # Normalize country names
        # Handle format like "AS (Assam (Assamese))" -> extract just the main country name
        def normalize_country(name):
            name = str(name).strip().lower()
            
            # If format is "CODE (Country Name)", extract country name
            if '(' in name and ')' in name:
                # Extract text between first '(' and last ')'
                start = name.find('(')
                end = name.rfind(')')
                if start < end:
                    name = name[start+1:end]
                    
                    # If still has parentheses, take the first part
                    # "assam (assamese)" -> "assam"
                    if '(' in name:
                        name = name[:name.find('(')].strip()
            
            # Common normalizations
            name = name.replace('_', ' ').strip()
            
            # Map variations to standard names
            mappings = {
                'assam': 'assamese',
                'assam (assamese)': 'assamese',
                'west java': 'sundanese',
                'west java (sundanese)': 'sundanese',
                'south korea': 'south korea',
                'united states': 'united states',
                'united kingdom': 'united kingdom',
                'indonesia': 'indonesia',
                'mexico': 'mexico',
                'nigeria': 'nigeria',
                'spain': 'spain',
                'china': 'china',
                'iran': 'iran',
                'bangladesh': 'bangladesh',
                'canada': 'canada',
                'egypt': 'egypt',
                'france': 'france',
                'germany': 'germany',
                'hong kong': 'hong kong',
                'india': 'india',
                'israel': 'israel',
                'japan': 'japan',
                'lebanon': 'lebanon',
                'morocco': 'morocco',
                'nepal': 'nepal',
                'netherlands': 'netherlands',
                'pakistan': 'pakistan',
                'saudi arabia': 'saudi arabia',
                'south africa': 'south africa',
                'taiwan': 'taiwan',
                'turkey': 'turkey',
                'zimbabwe': 'zimbabwe',
            }
            
            return mappings.get(name, name)
        
        df1_subset['country'] = df1_subset['country'].apply(normalize_country)
        df2_subset['country'] = df2_subset['country'].apply(normalize_country)
        
        # Find common countries
        countries1 = set(df1_subset['country'].unique())
        countries2 = set(df2_subset['country'].unique())
        
        print(f"\n📍 Countries in {self.benchmark1_name}: {len(countries1)}")
        print(f"   {sorted(countries1)}")
        
        print(f"\n📍 Countries in {self.benchmark2_name}: {len(countries2)}")
        print(f"   {sorted(countries2)}")
        
        self.common_countries = countries1 & countries2
        
        print(f"\n{'='*70}")
        print(f"📍 COMMON COUNTRIES: {len(self.common_countries)}")
        print(f"{'='*70}")
        
        if not self.common_countries:
            print("\n❌ ERROR: No common countries found between benchmarks!")
            print("\nPossible reasons:")
            print("  1. Country names are spelled differently")
            print("  2. Benchmarks test completely different countries")
            print("  3. Wrong country column name")
            sys.exit(1)
        
        print(f"{sorted(self.common_countries)}")
        
        # Merge on common countries
        self.merged_df = pd.merge(df1_subset, df2_subset, on='country', how='inner')
        self.merged_df = self.merged_df.sort_values('country').reset_index(drop=True)
        
        print(f"\n✓ Merged data: {len(self.merged_df)} countries")
        print("\nPreview of merged data:")
        print(self.merged_df.to_string(index=False))
        
        return self.merged_df
    
    def calculate_correlations(self):
        """Calculate Pearson and Spearman correlations"""
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS")
        print("="*70)
        
        x = self.merged_df[self.benchmark1_name]
        y = self.merged_df[self.benchmark2_name]
        
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(x, y)
        
        # Spearman correlation
        spearman_r, spearman_p = spearmanr(x, y)
        
        # Additional statistics
        mean_diff = abs(x - y).mean()
        max_diff = abs(x - y).max()
        
        print(f"\nModel: {self.model_name.upper()}")
        print(f"Benchmarks: {self.benchmark1_name} vs {self.benchmark2_name}")
        print(f"Countries analyzed: {len(self.merged_df)}")
        
        print(f"\n📊 Correlation Metrics:")
        print(f"  Pearson correlation:  r = {pearson_r:.4f}, p-value = {pearson_p:.4e}")
        print(f"  Spearman correlation: ρ = {spearman_r:.4f}, p-value = {spearman_p:.4e}")
        
        print(f"\n📊 Difference Metrics:")
        print(f"  Mean absolute difference: {mean_diff:.4f}")
        print(f"  Max absolute difference:  {max_diff:.4f}")
        
        print(f"\n📊 Mean Accuracies:")
        print(f"  {self.benchmark1_name}: {x.mean():.4f}")
        print(f"  {self.benchmark2_name}: {y.mean():.4f}")
        
        # Interpretation
        print(f"\n💡 Interpretation:")
        if pearson_r >= 0.7:
            print(f"  ✓ STRONG alignment (r={pearson_r:.3f}): Benchmarks likely measure similar constructs")
        elif pearson_r >= 0.5:
            print(f"  ~ MODERATE alignment (r={pearson_r:.3f}): Benchmarks partially aligned")
        else:
            print(f"  ✗ WEAK alignment (r={pearson_r:.3f}): Benchmarks may measure different constructs")
        
        # Save results
        results = {
            'model': self.model_name,
            'benchmark1': self.benchmark1_name,
            'benchmark2': self.benchmark2_name,
            'n_countries': len(self.merged_df),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mean_abs_diff': mean_diff,
            'max_abs_diff': max_diff,
            'benchmark1_mean': x.mean(),
            'benchmark2_mean': y.mean()
        }
        
        results_df = pd.DataFrame([results])
        results_file = self.output_dir / f"{self.model_name}_alignment_statistics.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n✓ Saved statistics to: {results_file}")
        
        return results
    
    def create_scatter_plot(self):
        """Create scatter plot showing correlation between benchmarks"""
        print("\n" + "="*70)
        print("CREATING SCATTER PLOT")
        print("="*70)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x = self.merged_df[self.benchmark1_name]
        y = self.merged_df[self.benchmark2_name]
        countries = self.merged_df['country']
        
        # Scatter plot
        ax.scatter(x, y, s=150, alpha=0.6, edgecolors='black', linewidth=1.5, color='steelblue')
        
        # Add country labels
        for i, country in enumerate(countries):
            ax.annotate(country, (x.iloc[i], y.iloc[i]), 
                       fontsize=9, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')
        
        # Regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
        
        # Perfect agreement line
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', 
                alpha=0.3, linewidth=1, label='Perfect agreement')
        
        # Calculate correlation for title
        pearson_r, _ = pearsonr(x, y)
        spearman_r, _ = spearmanr(x, y)
        
        # Labels and title
        ax.set_xlabel(f'{self.benchmark1_name} Accuracy', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'{self.benchmark2_name} Accuracy', fontsize=13, fontweight='bold')
        
        title = f'Benchmark Alignment: {self.benchmark1_name} vs {self.benchmark2_name}\n'
        title += f'Model: {self.model_name.upper()} | '
        title += f'Pearson r={pearson_r:.3f} | Spearman ρ={spearman_r:.3f}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filename = self.output_dir / f"{self.model_name}_{self.benchmark1_name}_vs_{self.benchmark2_name}_scatter.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved scatter plot: {filename}")
    
    def create_heatmap(self):
        """Create heatmap showing accuracy scores side-by-side"""
        print("\n" + "="*70)
        print("CREATING HEATMAP")
        print("="*70)
        
        # Prepare data for heatmap (transpose: rows=benchmarks, columns=countries)
        heatmap_data = self.merged_df.set_index('country')[[self.benchmark1_name, self.benchmark2_name]].T
        
        # Convert to 0-1 scale if values are percentages (>1)
        if heatmap_data.max().max() > 1:
            print("   Converting percentage values to 0-1 scale...")
            heatmap_data = heatmap_data / 100.0
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(14, len(self.merged_df) * 0.8), 6))
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.2f',  # Show 2 decimal places
                   cmap='RdYlGn',
                   center=0.5,
                   vmin=0, 
                   vmax=1,
                   cbar_kws={'label': 'Accuracy'},
                   linewidths=1,
                   linecolor='white',
                   ax=ax)
        
        # Calculate correlation
        pearson_r, _ = pearsonr(self.merged_df[self.benchmark1_name], 
                                self.merged_df[self.benchmark2_name])
        
        # Title
        title = f'Benchmark Comparison: {self.benchmark1_name} vs {self.benchmark2_name}\n'
        title += f'Model: {self.model_name.upper()} | Pearson r={pearson_r:.3f}'
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        
        ax.set_xlabel('Countries', fontsize=13, fontweight='bold')
        ax.set_ylabel('Benchmark', fontsize=13, fontweight='bold')
        
        # Rotate labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save
        filename = self.output_dir / f"{self.model_name}_{self.benchmark1_name}_vs_{self.benchmark2_name}_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved heatmap: {filename}")
    
    def create_difference_heatmap(self):
        """Create heatmap showing absolute differences between benchmarks for each country"""
        print("\n" + "="*70)
        print("CREATING DIFFERENCE HEATMAP")
        print("="*70)
        
        # Calculate differences
        diff_data = self.merged_df.copy()
        diff_data['difference'] = abs(diff_data[self.benchmark1_name] - diff_data[self.benchmark2_name])
        
        # Convert to 0-1 scale if values are percentages
        if diff_data[self.benchmark1_name].max() > 1:
            diff_data['difference'] = diff_data['difference'] / 100.0
        
        # Prepare for heatmap (single row showing differences)
        heatmap_data = diff_data.set_index('country')[['difference']].T
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(14, len(self.merged_df) * 0.8), 3))
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlGn_r',  # Reversed: green=small diff, red=large diff
                   center=0.1,
                   vmin=0, 
                   vmax=0.3,
                   cbar_kws={'label': 'Absolute Difference'},
                   linewidths=1,
                   linecolor='white',
                   ax=ax)
        
        # Title
        title = f'Benchmark Disagreement: |{self.benchmark1_name} - {self.benchmark2_name}|\n'
        title += f'Model: {self.model_name.upper()} | Mean Diff = {diff_data["difference"].mean():.3f}'
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        
        ax.set_xlabel('Countries', fontsize=13, fontweight='bold')
        ax.set_ylabel('', fontsize=13, fontweight='bold')
        
        # Rotate labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks([0.5], ['Abs. Difference'], rotation=0)
        
        plt.tight_layout()
        
        # Save
        filename = self.output_dir / f"{self.model_name}_{self.benchmark1_name}_vs_{self.benchmark2_name}_difference_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved difference heatmap: {filename}")
    
    def run_analysis(self):
        """Run complete alignment analysis"""
        print("\n" + "🔬"*35)
        print(f"BENCHMARK ALIGNMENT ANALYSIS")
        print(f"Model: {self.model_name.upper()}")
        print(f"Comparing: {self.benchmark1_name} vs {self.benchmark2_name}")
        print("🔬"*35)
        
        # Load data
        self.load_data()
        
        # Prepare and merge data
        self.prepare_data()
        
        # Calculate correlations
        results = self.calculate_correlations()
        
        # Create visualizations
        self.create_scatter_plot()
        self.create_heatmap()
        self.create_difference_heatmap()
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nGenerated files in: {self.output_dir}/")
        print(f"  - Statistics: {self.model_name}_alignment_statistics.csv")
        print(f"  - Scatter plot: {self.model_name}_{self.benchmark1_name}_vs_{self.benchmark2_name}_scatter.png")
        print(f"  - Heatmap: {self.model_name}_{self.benchmark1_name}_vs_{self.benchmark2_name}_heatmap.png")
        print(f"  - Difference heatmap: {self.model_name}_{self.benchmark1_name}_vs_{self.benchmark2_name}_difference_heatmap.png")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze alignment between two benchmarks for a single model'
    )
    parser.add_argument('--benchmark1_csv', required=True, 
                       help='Path to first benchmark CSV file')
    parser.add_argument('--benchmark2_csv', required=True,
                       help='Path to second benchmark CSV file')
    parser.add_argument('--model', required=True,
                       help='Model name (e.g., llama3b, "Llama 3.2-3B", qwen2_5b)')
    parser.add_argument('--benchmark1_name', default='Benchmark1',
                       help='Display name for first benchmark')
    parser.add_argument('--benchmark2_name', default='Benchmark2',
                       help='Display name for second benchmark')
    parser.add_argument('--country_col1', default=None,
                       help='Name of country column in benchmark1 (auto-detect if not specified)')
    parser.add_argument('--country_col2', default=None,
                       help='Name of country column in benchmark2 (auto-detect if not specified)')
    parser.add_argument('--output_dir', default='benchmark_alignment',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = BenchmarkAlignmentAnalyzer(
        benchmark1_csv=args.benchmark1_csv,
        benchmark2_csv=args.benchmark2_csv,
        model_name=args.model,
        benchmark1_name=args.benchmark1_name,
        benchmark2_name=args.benchmark2_name,
        country_col1=args.country_col1,
        country_col2=args.country_col2,
        output_dir=args.output_dir
    )
    
    analyzer.run_analysis()


if __name__ == "__main__":
    main()