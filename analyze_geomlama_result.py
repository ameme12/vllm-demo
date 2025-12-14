"""
GeoMLAMA Results Analyzer - ACL Paper Version
Analyzes GeoMLAMA evaluation results and creates visualizations with large fonts
Adapted for file format: geomlama_{language}_{model}_summary.json
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns

# Country mapping for GeoMLAMA
GEOMLAMA_COUNTRIES = {
    0: 'United States',
    1: 'China',
    2: 'India',
    3: 'Iran',
    4: 'Kenya'
}

# Region mapping for the 5 GeoMLAMA countries
GEOMLAMA_REGIONS = {
    'United States': 'North America',
    'China': 'East Asia',
    'India': 'South Asia',
    'Iran': 'Middle East/West Asia',
    'Kenya': 'Africa'
}

# Language to country mapping
LANGUAGE_TO_COUNTRY = {
    'english': 'United States',
    'en': 'United States',
    'chinese': 'China',
    'zh': 'China',
    'hindi': 'India',
    'hi': 'India',
    'persian': 'Iran',
    'fa': 'Iran',
    'swahili': 'Kenya',
    'sw': 'Kenya'
}


def get_region(country):
    """Get region for a GeoMLAMA country"""
    return GEOMLAMA_REGIONS.get(country, 'Unknown')


def extract_model_name(filename):
    """
    Extract clean model name from filename
    Examples:
        geomlama_english_llama_3b_summary.json -> Llama 3B
        geomlama_english_qwen2_5b_summary.json -> Qwen 2.5B
    """
    filename_lower = filename.lower()
    
    if 'llama' in filename_lower:
        if '3b' in filename_lower or '3_b' in filename_lower:
            return 'Llama 3B'
        elif '7b' in filename_lower:
            return 'Llama 7B'
        else:
            return 'Llama'
    elif 'qwen' in filename_lower:
        if '2.5b' in filename_lower or '2_5b' in filename_lower or '2-5b' in filename_lower:
            return 'Qwen 2.5B'
        elif '7b' in filename_lower:
            return 'Qwen 7B'
        else:
            return 'Qwen'
    else:
        return 'Unknown'


def extract_language(filename):
    """
    Extract language from filename
    Examples:
        geomlama_english_llama_3b_summary.json -> english
        geomlama_zh_qwen2_5b_summary.json -> chinese
    """
    filename_lower = filename.lower()
    
    # Check for full language names first
    for lang_name in ['english', 'chinese', 'hindi', 'persian', 'swahili']:
        if lang_name in filename_lower:
            return lang_name
    
    # Check for language codes
    for lang_code in ['en', 'zh', 'hi', 'fa', 'sw']:
        if f'_{lang_code}_' in filename_lower or f'_{lang_code}.' in filename_lower:
            lang_map = {'en': 'english', 'zh': 'chinese', 'hi': 'hindi', 
                       'fa': 'persian', 'sw': 'swahili'}
            return lang_map.get(lang_code, lang_code)
    
    return 'unknown'


def extract_context_setting(filename):
    """
    Extract whether country context was used
    Look for patterns like: with_country, without_country, no_country, country_yes, country_no
    Default to True if not specified
    """
    filename_lower = filename.lower()
    
    if 'without_country' in filename_lower or 'no_country' in filename_lower or 'country_no' in filename_lower:
        return False
    elif 'with_country' in filename_lower or 'country_yes' in filename_lower:
        return True
    else:
        # Default assumption: if not specified, assume with_country=True
        return True


def load_geomlama_results(results_dir):
    """Load GeoMLAMA summary JSON files and extract metrics"""
    results = []
    
    results_path = Path(results_dir)
    
    # Look for all summary JSON files
    json_files = list(results_path.glob("geomlama_*_summary.json"))
    json_files += list(results_path.glob("*_geomlama_*_summary.json"))
    json_files += list(results_path.glob("*summary.json"))
    
    if not json_files:
        print(f"⚠️  No JSON files found in {results_dir}")
        print(f"Looking for patterns: geomlama_*_summary.json")
        return pd.DataFrame()
    
    print(f"\nFound {len(json_files)} JSON files to process")
    
    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Check if this is a GeoMLAMA summary file (must have by_country)
            if 'by_country' not in data:
                print(f"⚠️  Skipping {json_file.name}: Not a GeoMLAMA summary (no 'by_country' field)")
                continue
            
            # Extract metadata from filename
            model = extract_model_name(json_file.name)
            language = extract_language(json_file.name)
            with_country = extract_context_setting(json_file.name)
            
            print(f"\n📄 Processing: {json_file.name}")
            print(f"   Model: {model}")
            print(f"   Language: {language}")
            print(f"   With Country Context: {with_country}")
            
            # Get metrics
            metrics = data['aggregate_metrics']
            num_samples = data.get('total_samples', 0)
            
            # Get country breakdown
            by_country = data.get('by_country', {})
            
            if not by_country:
                print(f"   ⚠️  No country breakdown found")
                continue
            
            # Add individual country results
            for country, stats in by_country.items():
                region = get_region(country)
                
                # Convert to percentage if needed
                exact_match = stats['exact_match']
                overlap = stats['overlap']
                
                if exact_match <= 1.0:
                    exact_match *= 100
                if overlap <= 1.0:
                    overlap *= 100
                
                results.append({
                    'model': model,
                    'language': language,
                    'with_country': with_country,
                    'country': country,
                    'region': region,
                    'exact_match': exact_match,
                    'overlap': overlap,
                    'num_samples': stats['count'],
                    'file': json_file.name
                })
            
            print(f"   ✓ Added {len(by_country)} country results")
            
            # Add overall metrics
            overall_exact = metrics.get('exact_match', {}).get('mean', 0)
            overall_overlap = metrics.get('overlap', {}).get('mean', 0)
            
            if overall_exact <= 1.0:
                overall_exact *= 100
            if overall_overlap <= 1.0:
                overall_overlap *= 100
            
            results.append({
                'model': model,
                'language': language,
                'with_country': with_country,
                'country': 'OVERALL',
                'region': 'ALL',
                'exact_match': overall_exact,
                'overlap': overall_overlap,
                'num_samples': num_samples,
                'file': json_file.name
            })
            
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("\n❌ No valid GeoMLAMA results found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print(f"✅ Successfully loaded {len(df)} result entries")
    print(f"{'='*80}")
    
    # Print summary
    print(f"\nModels found: {sorted(df['model'].unique())}")
    print(f"Languages found: {sorted(df['language'].unique())}")
    print(f"Countries found: {sorted(df[df['country'] != 'OVERALL']['country'].unique())}")
    
    # Remove duplicates (keep entry with most samples for each model+language+context+country combo)
    if not df.empty:
        df = df.sort_values('num_samples', ascending=False).groupby(
            ['model', 'language', 'with_country', 'country']
        ).first().reset_index()
        print(f"\nAfter removing duplicates: {len(df)} unique combinations")
    
    return df


def create_comparison_tables(df):
    """Create formatted comparison tables"""
    # Filter out OVERALL for detailed tables
    df_detail = df[df['country'] != 'OVERALL'].copy()
    
    # Pivot table for exact_match by country
    accuracy_by_country = df_detail.pivot_table(
        index='country', 
        columns=['model', 'language', 'with_country'], 
        values='exact_match'
    )
    
    # Pivot table for accuracy by region
    region_accuracy = df_detail.groupby(
        ['region', 'model', 'language', 'with_country']
    )['exact_match'].mean().unstack([1, 2, 3])
    
    return accuracy_by_country, region_accuracy


def plot_country_comparison(df, metric='exact_match'):
    """Create bar chart comparing model performance across countries with larger fonts for ACL paper"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Filter out OVERALL
    country_df = df[df['country'] != 'OVERALL'].copy()
    
    if country_df.empty:
        return None
    
    # Check if we have both with/without country context data
    has_both_contexts = len(country_df['with_country'].unique()) > 1
    
    if has_both_contexts:
        # Create two subplots with larger size
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        
        # Left plot: With Country Context
        with_country_df = country_df[country_df['with_country'] == True]
        # Right plot: Without Country Context
        without_country_df = country_df[country_df['with_country'] == False]
        
        for idx, (ax, data, title) in enumerate([
            (axes[0], with_country_df, 'With Country Context'),
            (axes[1], without_country_df, 'Without Country Context')
        ]):
            if data.empty:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=20)
                ax.set_title(title, fontsize=24, fontweight='bold')
                continue
            
            # Pivot data
            pivot = data.pivot_table(
                index='country',
                columns='model',
                values=metric,
                aggfunc='mean'
            )
            
            # Sort by average performance
            pivot['avg'] = pivot.mean(axis=1)
            pivot = pivot.sort_values('avg', ascending=True)
            pivot = pivot.drop('avg', axis=1)
            
            # Plot
            x = np.arange(len(pivot))
            width = 0.35
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            
            models = pivot.columns
            for i, model in enumerate(models):
                offset = width * (i - len(models)/2 + 0.5)
                bars = ax.barh(x + offset, pivot[model], width, 
                              label=model, alpha=0.85, color=colors[i % len(colors)],
                              edgecolor='black', linewidth=0.8)
                
                # Add value labels with larger font
                for j, bar in enumerate(bars):
                    width_val = bar.get_width()
                    if not np.isnan(width_val):
                        ax.text(width_val + 1.5, bar.get_y() + bar.get_height()/2.,
                               f'{width_val:.0f}%',
                               ha='left', va='center', fontsize=20, fontweight='bold')
            
            ax.set_yticks(x)
            ax.set_yticklabels(pivot.index, fontsize=22)
            ax.set_xlabel(f'{metric.replace("_", " ").title()} (%)', fontsize=24, fontweight='bold')
            ax.set_title(title, fontsize=26, fontweight='bold', pad=20)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(models),
                     fontsize=20, framealpha=0.9)
            ax.grid(axis='x', alpha=0.4, linestyle='--', linewidth=1.2)
            ax.set_xlim(0, 105)
            ax.set_facecolor('#f8f9fa')
            ax.tick_params(axis='x', labelsize=20)
            
            # Add baseline (GeoMLAMA has 5-8 answer choices, so ~12-20% random)
            ax.axvline(x=20, color='red', linestyle='--', linewidth=2.5, 
                       alpha=0.5, label='~Random (20%)' if idx == 0 else '')
        
        plt.suptitle(f'GeoMLAMA: {metric.replace("_", " ").title()} by Country\nLlama vs Qwen', 
                     fontsize=28, fontweight='bold', y=0.98)
    
    else:
        # Single plot if only one context setting with larger size
        fig, ax = plt.subplots(figsize=(14, 10))
        
        pivot = country_df.pivot_table(
            index='country',
            columns='model',
            values=metric,
            aggfunc='mean'
        )
        
        # Sort by average performance
        pivot['avg'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('avg', ascending=True)
        pivot = pivot.drop('avg', axis=1)
        
        # Plot
        x = np.arange(len(pivot))
        width = 0.35
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        models = pivot.columns
        for i, model in enumerate(models):
            offset = width * (i - len(models)/2 + 0.5)
            bars = ax.barh(x + offset, pivot[model], width, 
                          label=model, alpha=0.85, color=colors[i % len(colors)],
                          edgecolor='black', linewidth=0.8)
            
            # Add value labels with larger font
            for j, bar in enumerate(bars):
                width_val = bar.get_width()
                if not np.isnan(width_val):
                    ax.text(width_val + 1.5, bar.get_y() + bar.get_height()/2.,
                           f'{width_val:.0f}%',
                           ha='left', va='center', fontsize=20, fontweight='bold')
        
        ax.set_yticks(x)
        ax.set_yticklabels(pivot.index, fontsize=22)
        ax.set_xlabel(f'{metric.replace("_", " ").title()} (%)', fontsize=24, fontweight='bold')
        ax.set_ylabel('Country', fontsize=24, fontweight='bold')
        
        # Add context info to title
        context = country_df['with_country'].iloc[0]
        context_str = 'With Country Context' if context else 'Without Country Context'
        ax.set_title(f'GeoMLAMA: {metric.replace("_", " ").title()} by Country\n({context_str})', 
                     fontsize=26, fontweight='bold', pad=20)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(models),
                 fontsize=20, framealpha=0.9)
        ax.grid(axis='x', alpha=0.4, linestyle='--', linewidth=1.2)
        ax.set_xlim(0, 105)
        ax.set_facecolor('#f8f9fa')
        ax.tick_params(axis='x', labelsize=20)
        ax.axvline(x=20, color='red', linestyle='--', linewidth=2.5, alpha=0.5)
    
    plt.tight_layout()
    return fig


def plot_overall_comparison(df, metric='exact_match'):
    """Create overall performance comparison with larger fonts for ACL paper"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Get overall results
    overall_df = df[df['country'] == 'OVERALL'].copy()
    
    if overall_df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create grouped data
    models = sorted(overall_df['model'].unique())
    
    # Check if we have both context settings
    has_both = len(overall_df['with_country'].unique()) > 1
    
    if has_both:
        with_country_vals = []
        without_country_vals = []
        
        for model in models:
            with_val = overall_df[(overall_df['model'] == model) & 
                                 (overall_df['with_country'] == True)][metric].values
            without_val = overall_df[(overall_df['model'] == model) & 
                                    (overall_df['with_country'] == False)][metric].values
            
            with_country_vals.append(with_val[0] if len(with_val) > 0 else 0)
            without_country_vals.append(without_val[0] if len(without_val) > 0 else 0)
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, with_country_vals, width, 
                       label='With Country Context', alpha=0.85, 
                       color='#3498db', edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, without_country_vals, width,
                       label='Without Country Context', alpha=0.85,
                       color='#e74c3c', edgecolor='black', linewidth=1.2)
        
        # Add value labels with larger font
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                           f'{height:.0f}%',
                           ha='center', va='bottom', fontsize=20, fontweight='bold')
    else:
        # Single context setting
        vals = []
        for model in models:
            val = overall_df[overall_df['model'] == model][metric].values
            vals.append(val[0] if len(val) > 0 else 0)
        
        x = np.arange(len(models))
        bars = ax.bar(x, vals, alpha=0.85, color='#3498db', 
                     edgecolor='black', linewidth=1.2)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{height:.0f}%',
                       ha='center', va='bottom', fontsize=20, fontweight='bold')
        
        # Add context info to legend
        context = overall_df['with_country'].iloc[0]
        context_str = 'With Country Context' if context else 'Without Country Context'
        bars[0].set_label(context_str)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=22)
    ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', fontsize=24, fontweight='bold')
    ax.set_title(f'GeoMLAMA: Overall Performance Comparison\nLlama vs Qwen', 
                 fontsize=28, fontweight='bold', pad=25)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
             fontsize=22, framealpha=0.9)
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=1.2)
    ax.set_ylim(0, 105)
    ax.tick_params(axis='y', labelsize=20)
    ax.axhline(y=20, color='red', linestyle='--', linewidth=3, alpha=0.5, 
               label='~Random (20%)')
    
    plt.tight_layout()
    return fig


def print_summary_report(df):
    """Print summary report"""
    print("\n" + "="*80)
    print(" "*25 + "GEOMLAMA EVALUATION SUMMARY")
    print("="*80)
    
    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]
        
        print(f"\n📊 {model}")
        print("-"*80)
        
        # Check languages
        languages = sorted(model_df['language'].unique())
        print(f"  Languages evaluated: {', '.join(languages)}")
        
        # Overall performance for each context setting
        for with_country in sorted(model_df['with_country'].unique()):
            context_str = "WITH country context" if with_country else "WITHOUT country context"
            overall = model_df[(model_df['country'] == 'OVERALL') & 
                              (model_df['with_country'] == with_country)]
            
            if not overall.empty:
                print(f"\n  {context_str}:")
                print(f"    Exact Match: {overall['exact_match'].iloc[0]:.2f}%")
                print(f"    Overlap: {overall['overlap'].iloc[0]:.2f}%")
                
                # Country breakdown
                country_df = model_df[(model_df['country'] != 'OVERALL') & 
                                     (model_df['with_country'] == with_country)]
                if not country_df.empty:
                    best_country = country_df.loc[country_df['exact_match'].idxmax()]
                    worst_country = country_df.loc[country_df['exact_match'].idxmin()]
                    print(f"    Best: {best_country['country']} ({best_country['exact_match']:.2f}%)")
                    print(f"    Worst: {worst_country['country']} ({worst_country['exact_match']:.2f}%)")


def main():
    """Main function"""
    
    import os
    
    # Adjust this to your actual results directory
    results_dir = "/home/mila/r/ramesana/projects/vllm-demo/results_geomlama"
    
    # Check if directory exists
    if not Path(results_dir).exists():
        print(f"❌ Directory not found: {results_dir}")
        print(f"Please update the 'results_dir' variable in the script")
        return
    
    output_dir = Path(results_dir) / "analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print("🔍 GeoMLAMA Results Analyzer - ACL Paper Version")
    print(f"{'='*80}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load results
    df = load_geomlama_results(results_dir)
    
    if df.empty:
        print("\n❌ No GeoMLAMA summary JSON files found!")
        print(f"\nMake sure your files follow the pattern:")
        print(f"  - geomlama_english_llama_3b_summary.json")
        print(f"  - geomlama_english_qwen2_5b_summary.json")
        return
    
    # Print summary
    print_summary_report(df)
    
    # Export CSV files
    print(f"\n{'='*80}")
    print("📊 Exporting CSV files...")
    print(f"{'='*80}")
    
    # Main results
    df.to_csv(output_dir / 'geomlama_all_results.csv', index=False)
    print(f"  ✓ geomlama_all_results.csv")
    
    # Accuracy by country
    accuracy_by_country, accuracy_by_region = create_comparison_tables(df)
    if not accuracy_by_country.empty:
        accuracy_by_country.round(2).to_csv(output_dir / 'geomlama_accuracy_by_country.csv')
        print(f"  ✓ geomlama_accuracy_by_country.csv")
    
    if not accuracy_by_region.empty:
        accuracy_by_region.round(2).to_csv(output_dir / 'geomlama_accuracy_by_region.csv')
        print(f"  ✓ geomlama_accuracy_by_region.csv")
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("📊 Creating visualizations with LARGE FONTS for ACL paper...")
    print(f"{'='*80}")
    
    # 1. Overall comparison
    print("  Creating overall comparison...")
    fig = plot_overall_comparison(df)
    if fig:
        fig.savefig(output_dir / '1_overall_comparison.png', dpi=300, bbox_inches='tight')
        print("  ✓ 1_overall_comparison.png")
        plt.close(fig)
    
    # 2. Country comparison
    print("  Creating country comparison...")
    fig = plot_country_comparison(df, metric='exact_match')
    if fig:
        fig.savefig(output_dir / '2_country_comparison.png', dpi=300, bbox_inches='tight')
        print("  ✓ 2_country_comparison.png")
        plt.close(fig)
    
    print(f"\n{'='*80}")
    print(f"✅ Analysis Complete!")
    print(f"{'='*80}")
    print(f"All files saved to: {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  📊 CSV Files:")
    print(f"     - geomlama_all_results.csv")
    print(f"     - geomlama_accuracy_by_country.csv")
    print(f"     - geomlama_accuracy_by_region.csv")
    print(f"  📈 Visualizations (ACL PAPER READY):")
    print(f"     - 1_overall_comparison.png")
    print(f"     - 2_country_comparison.png")
    print(f"\n  Font sizes: ALL 20pt or larger ✓")
    print()


if __name__ == "__main__":
    main()