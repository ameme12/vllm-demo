"""
DLAMA Results Analyzer
Creates CSV tables and bar plots comparing model performance across countries
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

def load_dlama_summary(json_file):
    """Load DLAMA summary JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def normalize_country_name(country):
    """Normalize country names to avoid duplicates"""
    country = country.strip()
    
    # Normalize common variations
    country_map = {
        'USA': 'United States',
        'UK': 'United Kingdom',
        'UAE': 'United Arab Emirates',
        'South Korea': 'Korea (South)',
        'North Korea': 'Korea (North)',
    }
    
    return country_map.get(country, country)

def extract_country_accuracies(data, model_name):
    """Extract accuracy by country from DLAMA summary"""
    results = []
    
    # Extract from 'by_country' field
    if 'by_country' in data:
        for country, metrics in data['by_country'].items():
            # Normalize country name
            country_normalized = normalize_country_name(country)
            
            # Get metrics
            exact_match = metrics.get('exact_match', 0) * 100
            overlap = metrics.get('overlap', 0) * 100
            llm_judge_correct = metrics.get('llm_judge_correct', 0) * 100
            num_samples = metrics.get('count', 0)
            
            results.append({
                'country': country_normalized,
                'model': model_name,
                'exact_match': exact_match,
                'overlap': overlap,
                'llm_judge_correct': llm_judge_correct,
                'accuracy': llm_judge_correct,  # Use llm_judge_correct as main accuracy
                'num_samples': num_samples
            })
    
    return pd.DataFrame(results)

def load_all_dlama_results(results_dir):
    """Load all DLAMA summary files from directory"""
    results_dir = Path(results_dir)
    all_data = []
    
    # Find all summary JSON files
    for json_file in results_dir.glob("*_summary.json"):
        print(f"📂 Loading: {json_file.name}")
        
        # Determine model name from filename
        filename = json_file.stem.lower()
        if "llama" in filename or "llama3b" in filename:
            model_name = "Llama 3.2-3B"
        elif "qwen" in filename or "qwen2" in filename:
            model_name = "Qwen 2.5-3B"
        else:
            model_name = json_file.stem
        
        # Load data
        data = load_dlama_summary(json_file)
        
        # Extract country accuracies
        df = extract_country_accuracies(data, model_name)
        
        if not df.empty:
            all_data.append(df)
            print(f"   ✓ Extracted {len(df)} countries for {model_name}")
            print(f"   ✓ Total samples: {df['num_samples'].sum():,}")
        else:
            print(f"   ⚠️  No country data found")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Handle duplicates by aggregating (sum samples, average metrics)
        # This happens when same country appears in both datasets
        combined_df = combined_df.groupby(['country', 'model']).agg({
            'exact_match': 'mean',
            'overlap': 'mean',
            'llm_judge_correct': 'mean',
            'accuracy': 'mean',
            'num_samples': 'sum'
        }).reset_index()
        
        return combined_df
    else:
        return pd.DataFrame()

def create_accuracy_table(df, metric='llm_judge_correct', min_samples=10):
    """Create pivot table with countries as rows and models as columns"""
    # Filter out countries with very few samples
    df_filtered = df[df['num_samples'] >= min_samples].copy()
    
    # Pivot: countries as rows, models as columns, accuracy as values
    accuracy_pivot = df_filtered.pivot_table(
        index='country', 
        columns='model', 
        values=metric,
        aggfunc='first'  # Use first value if duplicates remain
    )
    
    # Calculate average and difference
    if len(accuracy_pivot.columns) >= 2:
        accuracy_pivot['Average'] = accuracy_pivot.mean(axis=1)
        model1, model2 = accuracy_pivot.columns[0], accuracy_pivot.columns[1]
        accuracy_pivot['Difference'] = accuracy_pivot[model1] - accuracy_pivot[model2]
    else:
        accuracy_pivot['Average'] = accuracy_pivot.iloc[:, 0]
    
    # Sort by average accuracy
    accuracy_pivot = accuracy_pivot.sort_values('Average', ascending=False)
    
    return accuracy_pivot

def plot_accuracy_by_country(df, metric='llm_judge_correct', min_samples=10, output_file=None):
    """Create bar plot comparing model accuracies across countries - MUCH LARGER FONTS"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Filter out countries with very few samples
    df_filtered = df[df['num_samples'] >= min_samples].copy()
    
    # Create pivot table
    accuracy_pivot = df_filtered.pivot_table(
        index='country',
        columns='model',
        values=metric,
        aggfunc='first'
    )
    
    # Sort by average accuracy (top 20 countries)
    accuracy_pivot['Average'] = accuracy_pivot.mean(axis=1)
    accuracy_pivot = accuracy_pivot.sort_values('Average', ascending=False)
    
    # Take top 20 countries
    accuracy_pivot = accuracy_pivot.head(20)
    accuracy_pivot = accuracy_pivot.drop('Average', axis=1)
    
    # Create figure - 4X LARGER (16x10 -> 64x40)
    fig, ax = plt.subplots(figsize=(64, 40))
    
    # Prepare data
    x = np.arange(len(accuracy_pivot))
    width = 0.35
    
    # Colors for models
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # Plot bars for each model
    models = accuracy_pivot.columns
    for i, model in enumerate(models):
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, accuracy_pivot[model], width, 
                      label=model, alpha=0.85, color=colors[i],
                      edgecolor='black', linewidth=6)  # Even thicker lines
        
        # REMOVED: Value labels on bars (as requested)
    
    # Customize plot - MUCH LARGER FONTS
    metric_name = 'LLM Judge Correctness' if metric == 'llm_judge_correct' else metric.replace('_', ' ').title()
    ax.set_title(f'DLAMA: {metric_name} by Country (Top 20)\nLlama 3.2-3B vs Qwen 2.5-3B', 
                 fontsize=120, fontweight='bold', pad=120)  # MUCH larger title
    ax.set_xlabel('Country', fontsize=100, fontweight='bold')  # MUCH larger
    ax.set_ylabel('Accuracy (%)', fontsize=100, fontweight='bold')  # MUCH larger
    ax.set_xticks(x)
    ax.set_xticklabels(accuracy_pivot.index, rotation=45, ha='right', fontsize=80)  # MUCH larger
    ax.tick_params(axis='y', labelsize=80, width=4, length=20)  # MUCH larger, thicker ticks
    ax.tick_params(axis='x', width=4, length=20)  # Thicker ticks
    ax.legend(loc='upper right', fontsize=90, framealpha=0.9, 
              edgecolor='black', fancybox=True, shadow=True)  # MUCH larger legend
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=4)  # Thicker grid
    ax.set_ylim(0, 105)
    ax.set_facecolor('#f8f9fa')
    
    # Make spines thicker
    for spine in ax.spines.values():
        spine.set_linewidth(4)
    
    plt.tight_layout()
    
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {output_file}")
    
    return fig

def create_culture_comparison_plot(df, output_file=None):
    """Create bar plot comparing Asian vs Western cultures"""
    
    # Calculate culture-level metrics from country data
    culture_map = {
        'Korea (South)': 'Asian',
        'China': 'Asian',
        'Indonesia': 'Asian',
        'Japan': 'Asian',
        'Korea (North)': 'Asian',
        'United States': 'Western',
        'United Kingdom': 'Western',
        'Spain': 'Western',
        'Canada': 'Western',
        'Germany': 'Western',
        'France': 'Western',
        'Ireland': 'Western',
        'Australia': 'Western',
        'Austria': 'Western',
        'Belgium': 'Western',
        'Italy': 'Western',
        'Luxembourg': 'Western',
        'Netherlands': 'Western',
        'New Zealand': 'Western',
        'Portugal': 'Western',
        'Switzerland': 'Western'
    }
    
    # Add culture column
    df['culture'] = df['country'].map(culture_map)
    
    # Filter out unknown cultures and aggregate
    df_filtered = df[df['culture'].notna()].copy()
    
    # Group by culture and model, weighted by sample size
    culture_summary = df_filtered.groupby(['culture', 'model']).apply(
        lambda x: pd.Series({
            'llm_judge_correct': (x['llm_judge_correct'] * x['num_samples']).sum() / x['num_samples'].sum(),
            'num_samples': x['num_samples'].sum()
        })
    ).reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    cultures = sorted(culture_summary['culture'].unique())
    x = np.arange(len(cultures))
    width = 0.35
    
    colors = ['#3498db', '#e74c3c']
    
    models = sorted(culture_summary['model'].unique())
    for i, model in enumerate(models):
        model_data = culture_summary[culture_summary['model'] == model]
        accuracies = [model_data[model_data['culture'] == c]['llm_judge_correct'].values[0] 
                     if len(model_data[model_data['culture'] == c]) > 0 else 0 
                     for c in cultures]
        
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, accuracies, width,
                     label=model, alpha=0.85, color=colors[i],
                     edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            samples = model_data[model_data['culture'] == cultures[j]]['num_samples'].values[0] if len(model_data[model_data['culture'] == cultures[j]]) > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%\n(n={int(samples):,})',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_title('DLAMA: Asian vs Western Culture Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Culture', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cultures, fontsize=13)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc='upper right', fontsize=13, framealpha=0.9)
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 105)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {output_file}")
    
    return fig

def create_sample_size_plot(df, min_samples=10, output_file=None):
    """Create plot showing sample sizes by country"""
    
    # Filter countries with >= min_samples and get unique counts
    df_filtered = df[df['num_samples'] >= min_samples].copy()
    samples_df = df_filtered.groupby('country')['num_samples'].first().sort_values(ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(len(samples_df)), samples_df.values,
                   color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, samples_df.values)):
        ax.text(val + 100, i, f'{int(val):,}', 
                va='center', fontsize=11, fontweight='bold')
    
    ax.set_yticks(range(len(samples_df)))
    ax.set_yticklabels(samples_df.index, fontsize=12)
    ax.set_xlabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title(f'DLAMA: Sample Size by Country (Top 20, ≥{min_samples} samples)', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', labelsize=12)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {output_file}")
    
    return fig

def print_summary_report(df, min_samples=10):
    """Print formatted summary report"""
    print("\n" + "="*80)
    print(" "*25 + "DLAMA EVALUATION SUMMARY")
    print("="*80)
    
    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]
        print(f"\n📊 {model}")
        print("-"*80)
        print(f"  Countries Evaluated: {len(model_df)}")
        print(f"  Total Samples: {model_df['num_samples'].sum():,}")
        print(f"  Average Accuracy (LLM Judge): {model_df['llm_judge_correct'].mean():.2f}%")
        print(f"  Average Exact Match: {model_df['exact_match'].mean():.2f}%")
        print(f"  Average Overlap: {model_df['overlap'].mean():.2f}%")
        
        if len(model_df) > 0:
            # Filter to countries with min_samples for best/worst
            model_df_filtered = model_df[model_df['num_samples'] >= min_samples]
            if len(model_df_filtered) > 0:
                best_idx = model_df_filtered['llm_judge_correct'].idxmax()
                worst_idx = model_df_filtered['llm_judge_correct'].idxmin()
                
                print(f"  Best Country (≥{min_samples} samples): {model_df_filtered.loc[best_idx, 'country']} "
                      f"({model_df_filtered.loc[best_idx, 'llm_judge_correct']:.2f}%)")
                print(f"  Worst Country (≥{min_samples} samples): {model_df_filtered.loc[worst_idx, 'country']} "
                      f"({model_df_filtered.loc[worst_idx, 'llm_judge_correct']:.2f}%)")
                print(f"  Std Dev: {model_df_filtered['llm_judge_correct'].std():.2f}%")
    
    print("\n" + "="*80)
    print(f"COUNTRY-WISE BREAKDOWN (Countries with ≥{min_samples} samples, Top 25)")
    print("="*80)
    
    # Filter countries with >= min_samples
    df_filtered = df[df['num_samples'] >= min_samples]
    
    # Get top 25 countries by average accuracy
    top_countries = df_filtered.groupby('country')['llm_judge_correct'].mean().sort_values(ascending=False).head(25).index
    
    for country in top_countries:
        print(f"\n{country}:")
        for model in sorted(df_filtered['model'].unique()):
            model_data = df_filtered[(df_filtered['country'] == country) & (df_filtered['model'] == model)]
            if not model_data.empty:
                acc = model_data['llm_judge_correct'].iloc[0]
                exact = model_data['exact_match'].iloc[0]
                overlap = model_data['overlap'].iloc[0]
                samples = model_data['num_samples'].iloc[0]
                print(f"  {model:20s}: LLM={acc:5.1f}% | Exact={exact:5.1f}% | Overlap={overlap:5.1f}% ({int(samples):,} samples)")
    
    print("\n" + "="*80 + "\n")

def main():
    """Main function"""
    import os
    
    # Setup paths
    results_dir = "/home/mila/r/ramesana/projects/vllm-demo/results_dlama"
    output_dir = Path(results_dir) / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    min_samples = 10  # Minimum samples per country
    
    print("\n🔍 DLAMA Results Analyzer")
    print("="*80)
    
    # Load results
    print(f"\n📂 Loading results from: {results_dir}")
    df = load_all_dlama_results(results_dir)
    
    if df.empty:
        print("\n❌ No data found!")
        print("   Make sure summary JSON files exist in the results directory")
        return
    
    print(f"\n✓ Successfully loaded {len(df)} country-model combinations")
    print(f"  Models: {', '.join(sorted(df['model'].unique()))}")
    print(f"  Countries: {len(df['country'].unique())} unique")
    print(f"  Total samples: {df['num_samples'].sum():,}")
    
    # Print summary report
    print_summary_report(df, min_samples)
    
    # Create accuracy tables for all metrics
    print(f"\n📊 Creating accuracy comparison tables (≥{min_samples} samples)...")
    
    metrics = {
        'llm_judge_correct': 'LLM Judge Correctness',
        'exact_match': 'Exact Match',
        'overlap': 'Overlap'
    }
    
    for metric_key, metric_name in metrics.items():
        accuracy_table = create_accuracy_table(df, metric_key, min_samples)
        
        # Display table (top 20)
        print(f"\n" + "="*80)
        print(f"ACCURACY TABLE - {metric_name.upper()} (Top 20 Countries, ≥{min_samples} samples)")
        print("="*80)
        print(accuracy_table.head(20).round(2).to_string())
        print("\n")
        
        # Save table to CSV
        csv_file = output_dir / f"accuracy_by_country_{metric_key}.csv"
        accuracy_table.round(2).to_csv(csv_file)
        print(f"✓ Saved CSV: {csv_file}")
    
    # Save raw data
    raw_csv = output_dir / "raw_data.csv"
    df.to_csv(raw_csv, index=False)
    print(f"✓ Saved raw data: {raw_csv}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # Main accuracy bar plot (LLM Judge) - NOW WITH MUCH LARGER FONTS
    plot_file = output_dir / "accuracy_by_country_llm_judge.png"
    plot_accuracy_by_country(df, 'llm_judge_correct', min_samples, plot_file)
    
    # Culture comparison
    culture_plot = output_dir / "asian_vs_western_comparison.png"
    create_culture_comparison_plot(df, culture_plot)
    
    # Sample size plot
    samples_plot_file = output_dir / "samples_by_country.png"
    create_sample_size_plot(df, min_samples, samples_plot_file)
    
    # Show plots
    plt.show()
    
    print(f"\n✅ Analysis complete! Check the {output_dir} directory for outputs.")
    print(f"\nGenerated files:")
    print(f"  📄 accuracy_by_country_llm_judge_correct.csv - Main accuracy table")
    print(f"  📄 accuracy_by_country_exact_match.csv - Exact match table")
    print(f"  📄 accuracy_by_country_overlap.csv - Overlap table")
    print(f"  📄 raw_data.csv - Raw data with all metrics")
    print(f"  📊 accuracy_by_country_llm_judge.png - GIANT plot with HUGE fonts!")
    print(f"  📊 asian_vs_western_comparison.png - Culture comparison")
    print(f"  📊 samples_by_country.png - Sample sizes")
    print()

if __name__ == "__main__":
    main()