"""
GeoMLAMA Results Analyzer
Analyzes GeoMLAMA evaluation results and creates visualizations
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
    """Create bar chart comparing model performance across countries"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Filter out OVERALL
    country_df = df[df['country'] != 'OVERALL'].copy()
    
    if country_df.empty:
        return None
    
    # Check if we have both with/without country context data
    has_both_contexts = len(country_df['with_country'].unique()) > 1
    
    if has_both_contexts:
        # Create two subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Left plot: With Country Context
        with_country_df = country_df[country_df['with_country'] == True]
        # Right plot: Without Country Context
        without_country_df = country_df[country_df['with_country'] == False]
        
        for idx, (ax, data, title) in enumerate([
            (axes[0], with_country_df, 'With Country Context'),
            (axes[1], without_country_df, 'Without Country Context')
        ]):
            if data.empty:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                ax.set_title(title)
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
                              edgecolor='black', linewidth=0.5)
                
                # Add value labels
                for j, bar in enumerate(bars):
                    width_val = bar.get_width()
                    if not np.isnan(width_val):
                        ax.text(width_val + 1, bar.get_y() + bar.get_height()/2.,
                               f'{width_val:.1f}',
                               ha='left', va='center', fontsize=10, fontweight='bold')
            
            ax.set_yticks(x)
            ax.set_yticklabels(pivot.index, fontsize=11)
            ax.set_xlabel(f'{metric.replace("_", " ").title()} (%)', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            ax.legend(loc='lower right', fontsize=11)
            ax.grid(axis='x', alpha=0.4, linestyle='--')
            ax.set_xlim(0, 105)
            ax.set_facecolor('#f8f9fa')
            
            # Add baseline (GeoMLAMA has 5-8 answer choices, so ~12-20% random)
            ax.axvline(x=20, color='red', linestyle='--', linewidth=2, 
                       alpha=0.5, label='~Random (20%)' if idx == 0 else '')
        
        plt.suptitle(f'GeoMLAMA: {metric.replace("_", " ").title()} by Country', 
                     fontsize=16, fontweight='bold', y=0.98)
    
    else:
        # Single plot if only one context setting
        fig, ax = plt.subplots(figsize=(12, 8))
        
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
                          edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for j, bar in enumerate(bars):
                width_val = bar.get_width()
                if not np.isnan(width_val):
                    ax.text(width_val + 1, bar.get_y() + bar.get_height()/2.,
                           f'{width_val:.1f}',
                           ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_yticks(x)
        ax.set_yticklabels(pivot.index, fontsize=11)
        ax.set_xlabel(f'{metric.replace("_", " ").title()} (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Country', fontsize=12, fontweight='bold')
        
        # Add context info to title
        context = country_df['with_country'].iloc[0]
        context_str = 'With Country Context' if context else 'Without Country Context'
        ax.set_title(f'GeoMLAMA: {metric.replace("_", " ").title()} by Country\n({context_str})', 
                     fontsize=14, fontweight='bold', pad=10)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(axis='x', alpha=0.4, linestyle='--')
        ax.set_xlim(0, 105)
        ax.set_facecolor('#f8f9fa')
        ax.axvline(x=20, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    return fig


def plot_language_analysis(df, metric='exact_match'):
    """Analyze performance across different languages"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Filter out OVERALL
    country_df = df[df['country'] != 'OVERALL'].copy()
    
    if country_df.empty or 'language' not in country_df.columns:
        return None
    
    # Check if we have multiple languages
    languages = country_df['language'].unique()
    if len(languages) == 1:
        print(f"  ⚠️  Only one language found ({languages[0]}), skipping language analysis")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Group by language and model
    lang_data = country_df.groupby(['language', 'model', 'with_country'])[metric].mean().reset_index()
    
    # Create grouped bar chart
    languages_sorted = sorted(lang_data['language'].unique())
    models = sorted(lang_data['model'].unique())
    
    x = np.arange(len(languages_sorted))
    width = 0.35
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, model in enumerate(models):
        model_data = lang_data[lang_data['model'] == model]
        
        # With country context
        with_country = []
        # Without country context
        without_country = []
        
        for lang in languages_sorted:
            with_val = model_data[(model_data['language'] == lang) & 
                                 (model_data['with_country'] == True)][metric].values
            without_val = model_data[(model_data['language'] == lang) & 
                                    (model_data['with_country'] == False)][metric].values
            
            with_country.append(with_val[0] if len(with_val) > 0 else np.nan)
            without_country.append(without_val[0] if len(without_val) > 0 else np.nan)
        
        offset = width * (i - len(models)/2 + 0.5)
        
        # Plot with country (if data exists)
        if not all(np.isnan(with_country)):
            bars1 = ax.bar(x + offset - width/4, with_country, width/2, 
                          label=f'{model} (w/ country)', 
                          alpha=0.85, color=colors[i % len(colors)],
                          edgecolor='black', linewidth=0.5)
        
        # Plot without country (if data exists)
        if not all(np.isnan(without_country)):
            bars2 = ax.bar(x + offset + width/4, without_country, width/2,
                          label=f'{model} (w/o country)',
                          alpha=0.5, color=colors[i % len(colors)],
                          edgecolor='black', linewidth=0.5, hatch='//')
    
    ax.set_xticks(x)
    ax.set_xticklabels([l.title() for l in languages_sorted], fontsize=12)
    ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Language', fontsize=13, fontweight='bold')
    ax.set_title(f'GeoMLAMA: Performance by Language and Model', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(axis='y', alpha=0.4, linestyle='--')
    ax.set_ylim(0, 105)
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    return fig


def plot_country_context_effect(df, metric='exact_match'):
    """Show the effect of including/excluding country context"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Filter out OVERALL
    country_df = df[df['country'] != 'OVERALL'].copy()
    
    if country_df.empty:
        return None
    
    # Check if we have both context settings
    if len(country_df['with_country'].unique()) < 2:
        print(f"  ⚠️  Only one context setting found, skipping context effect analysis")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate difference: with_country - without_country
    countries = sorted(country_df['country'].unique())
    models = sorted(country_df['model'].unique())
    
    diffs = []
    for model in models:
        model_diffs = []
        for country in countries:
            with_val = country_df[(country_df['model'] == model) & 
                                 (country_df['country'] == country) & 
                                 (country_df['with_country'] == True)][metric].values
            without_val = country_df[(country_df['model'] == model) & 
                                    (country_df['country'] == country) & 
                                    (country_df['with_country'] == False)][metric].values
            
            if len(with_val) > 0 and len(without_val) > 0:
                diff = with_val[0] - without_val[0]
            else:
                diff = np.nan
            model_diffs.append(diff)
        diffs.append(model_diffs)
    
    # Create heatmap
    diffs_df = pd.DataFrame(diffs, index=models, columns=countries)
    
    # Only plot if we have valid data
    if diffs_df.isna().all().all():
        print(f"  ⚠️  No valid context comparison data, skipping")
        return None
    
    sns.heatmap(diffs_df.T, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                vmin=-20, vmax=20, cbar_kws={'label': 'Δ Accuracy (%)'},
                linewidths=1, linecolor='gray', ax=ax)
    
    ax.set_title('Effect of Country Context on Accuracy\n(With Context - Without Context)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Country', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_overall_comparison(df, metric='exact_match'):
    """Create overall performance comparison"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Get overall results
    overall_df = df[df['country'] == 'OVERALL'].copy()
    
    if overall_df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
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
                       color='#3498db', edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, without_country_vals, width,
                       label='Without Country Context', alpha=0.85,
                       color='#e74c3c', edgecolor='black', linewidth=1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
    else:
        # Single context setting
        vals = []
        for model in models:
            val = overall_df[overall_df['model'] == model][metric].values
            vals.append(val[0] if len(val) > 0 else 0)
        
        x = np.arange(len(models))
        bars = ax.bar(x, vals, alpha=0.85, color='#3498db', 
                     edgecolor='black', linewidth=1)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add context info to legend
        context = overall_df['with_country'].iloc[0]
        context_str = 'With Country Context' if context else 'Without Country Context'
        bars[0].set_label(context_str)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'GeoMLAMA: Overall Performance Comparison', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(axis='y', alpha=0.4, linestyle='--')
    ax.set_ylim(0, 105)
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, alpha=0.5, 
               label='~Random (20%)')
    
    plt.tight_layout()
    return fig


def plot_combined_dashboard(df):
    """Create comprehensive dashboard with all key visualizations"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('GeoMLAMA Complete Analysis Dashboard', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    country_df = df[df['country'] != 'OVERALL']
    overall_df = df[df['country'] == 'OVERALL']
    
    has_both_contexts = len(df['with_country'].unique()) > 1
    
    # 1. Overall Performance (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    if not overall_df.empty:
        if has_both_contexts:
            pivot = overall_df.pivot_table(
                index='model',
                columns='with_country',
                values='exact_match',
                aggfunc='mean'
            )
            
            x = np.arange(len(pivot))
            width = 0.35
            
            if True in pivot.columns:
                bars1 = ax1.bar(x - width/2, pivot[True], width, 
                               label='With Country', alpha=0.85, color=colors[0])
                for bar in bars1:
                    h = bar.get_height()
                    if h > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2., h + 1,
                                f'{h:.1f}', ha='center', va='bottom', 
                                fontsize=10, fontweight='bold')
            
            if False in pivot.columns:
                bars2 = ax1.bar(x + width/2, pivot[False], width,
                               label='Without Country', alpha=0.85, color=colors[1])
                for bar in bars2:
                    h = bar.get_height()
                    if h > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2., h + 1,
                                f'{h:.1f}', ha='center', va='bottom', 
                                fontsize=10, fontweight='bold')
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(pivot.index, fontsize=11)
        else:
            # Single context
            models = sorted(overall_df['model'].unique())
            vals = [overall_df[overall_df['model'] == m]['exact_match'].values[0] 
                   for m in models]
            
            bars = ax1.bar(range(len(models)), vals, alpha=0.85, color=colors[0])
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., h + 1,
                            f'{h:.1f}', ha='center', va='bottom', 
                            fontsize=10, fontweight='bold')
            
            ax1.set_xticks(range(len(models)))
            ax1.set_xticklabels(models, fontsize=11)
        
        ax1.set_ylabel('Exact Match (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Overall Performance', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 100)
        ax1.axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # 2. Country Comparison - With Context (Top Right) 
    ax2 = fig.add_subplot(gs[0, 1])
    if has_both_contexts:
        with_context_df = country_df[country_df['with_country'] == True]
        title = 'With Country Context'
    else:
        with_context_df = country_df
        context = country_df['with_country'].iloc[0] if not country_df.empty else True
        title = 'With Country Context' if context else 'Without Country Context'
    
    if not with_context_df.empty:
        pivot = with_context_df.pivot_table(
            index='country',
            columns='model',
            values='exact_match',
            aggfunc='mean'
        )
        pivot['avg'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('avg', ascending=True)
        pivot = pivot.drop('avg', axis=1)
        
        x = np.arange(len(pivot))
        width = 0.35
        
        for i, model in enumerate(pivot.columns):
            offset = width * (i - len(pivot.columns)/2 + 0.5)
            ax2.barh(x + offset, pivot[model], width, 
                    label=model, alpha=0.85, color=colors[i % len(colors)])
        
        ax2.set_yticks(x)
        ax2.set_yticklabels(pivot.index, fontsize=10)
        ax2.set_xlabel('Exact Match (%)', fontsize=12, fontweight='bold')
        ax2.set_title(title, fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(axis='x', alpha=0.3)
        ax2.set_xlim(0, 100)
    
    # 3. Country Comparison - Without Context (Middle Left)
    ax3 = fig.add_subplot(gs[1, 0])
    if has_both_contexts:
        without_context_df = country_df[country_df['with_country'] == False]
        
        if not without_context_df.empty:
            pivot = without_context_df.pivot_table(
                index='country',
                columns='model',
                values='exact_match',
                aggfunc='mean'
            )
            pivot['avg'] = pivot.mean(axis=1)
            pivot = pivot.sort_values('avg', ascending=True)
            pivot = pivot.drop('avg', axis=1)
            
            x = np.arange(len(pivot))
            width = 0.35
            
            for i, model in enumerate(pivot.columns):
                offset = width * (i - len(pivot.columns)/2 + 0.5)
                ax3.barh(x + offset, pivot[model], width, 
                        label=model, alpha=0.85, color=colors[i % len(colors)])
            
            ax3.set_yticks(x)
            ax3.set_yticklabels(pivot.index, fontsize=10)
            ax3.set_xlabel('Exact Match (%)', fontsize=12, fontweight='bold')
            ax3.set_title('Without Country Context', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(axis='x', alpha=0.3)
            ax3.set_xlim(0, 100)
        else:
            ax3.text(0.5, 0.5, 'No data without country context', 
                    ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'Single context setting only', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Context Effect Heatmap (Middle Right)
    ax4 = fig.add_subplot(gs[1, 1])
    if has_both_contexts and not country_df.empty:
        countries = sorted(country_df['country'].unique())
        models = sorted(country_df['model'].unique())
        
        diffs = []
        for model in models:
            model_diffs = []
            for country in countries:
                with_val = country_df[(country_df['model'] == model) & 
                                     (country_df['country'] == country) & 
                                     (country_df['with_country'] == True)]['exact_match'].values
                without_val = country_df[(country_df['model'] == model) & 
                                        (country_df['country'] == country) & 
                                        (country_df['with_country'] == False)]['exact_match'].values
                
                if len(with_val) > 0 and len(without_val) > 0:
                    diff = with_val[0] - without_val[0]
                else:
                    diff = 0
                model_diffs.append(diff)
            diffs.append(model_diffs)
        
        diffs_df = pd.DataFrame(diffs, index=models, columns=countries)
        
        sns.heatmap(diffs_df.T, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    vmin=-20, vmax=20, cbar_kws={'label': 'Δ (%)'},
                    linewidths=1, linecolor='gray', ax=ax4)
        
        ax4.set_title('Country Context Effect', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Country', fontsize=11, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Context comparison not available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    # 5. Metric Comparison (Bottom Left)
    ax5 = fig.add_subplot(gs[2, 0])
    if not country_df.empty:
        # Compare exact_match vs overlap
        models = sorted(country_df['model'].unique())
        
        exact_vals = []
        overlap_vals = []
        
        for model in models:
            exact = country_df[country_df['model'] == model]['exact_match'].mean()
            overlap = country_df[country_df['model'] == model]['overlap'].mean()
            exact_vals.append(exact)
            overlap_vals.append(overlap)
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, exact_vals, width, 
                       label='Exact Match', alpha=0.85, color='#3498db')
        bars2 = ax5.bar(x + width/2, overlap_vals, width,
                       label='Overlap', alpha=0.85, color='#2ecc71')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax5.text(bar.get_x() + bar.get_width()/2., h + 1,
                            f'{h:.1f}', ha='center', va='bottom', 
                            fontsize=10, fontweight='bold')
        
        ax5.set_xticks(x)
        ax5.set_xticklabels(models, fontsize=11)
        ax5.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax5.set_title('Exact Match vs Overlap', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(axis='y', alpha=0.3)
        ax5.set_ylim(0, 100)
    
    # 6. Region Summary (Bottom Right)
    ax6 = fig.add_subplot(gs[2, 1])
    if not country_df.empty:
        region_data = country_df.groupby(['region', 'model'])['exact_match'].mean().reset_index()
        
        regions = sorted(region_data['region'].unique())
        models = sorted(region_data['model'].unique())
        
        x = np.arange(len(regions))
        width = 0.35
        
        for i, model in enumerate(models):
            model_vals = []
            for region in regions:
                val = region_data[(region_data['region'] == region) & 
                                 (region_data['model'] == model)]['exact_match'].values
                model_vals.append(val[0] if len(val) > 0 else 0)
            
            offset = width * (i - len(models)/2 + 0.5)
            bars = ax6.bar(x + offset, model_vals, width, 
                          label=model, alpha=0.85, color=colors[i % len(colors)])
            
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax6.text(bar.get_x() + bar.get_width()/2., h + 1,
                            f'{h:.1f}', ha='center', va='bottom', 
                            fontsize=9, fontweight='bold')
        
        ax6.set_xticks(x)
        ax6.set_xticklabels(regions, rotation=45, ha='right', fontsize=10)
        ax6.set_ylabel('Exact Match (%)', fontsize=12, fontweight='bold')
        ax6.set_title('Performance by Region', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(axis='y', alpha=0.3)
        ax6.set_ylim(0, 100)
    
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
    print("🔍 GeoMLAMA Results Analyzer")
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
    print("📊 Creating visualizations...")
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
    
    # 3. Language analysis (if multiple languages)
    print("  Creating language analysis...")
    fig = plot_language_analysis(df, metric='exact_match')
    if fig:
        fig.savefig(output_dir / '3_language_analysis.png', dpi=300, bbox_inches='tight')
        print("  ✓ 3_language_analysis.png")
        plt.close(fig)
    else:
        print("  ⊗ 3_language_analysis.png (skipped - only one language)")
    
    # 4. Context effect (if both contexts available)
    print("  Creating context effect analysis...")
    fig = plot_country_context_effect(df, metric='exact_match')
    if fig:
        fig.savefig(output_dir / '4_context_effect.png', dpi=300, bbox_inches='tight')
        print("  ✓ 4_context_effect.png")
        plt.close(fig)
    else:
        print("  ⊗ 4_context_effect.png (skipped - only one context setting)")
    
    # 5. Combined dashboard
    print("  Creating combined dashboard...")
    fig = plot_combined_dashboard(df)
    fig.savefig(output_dir / '5_COMPLETE_DASHBOARD.png', dpi=300, bbox_inches='tight')
    print("  ✓ 5_COMPLETE_DASHBOARD.png (⭐ COMBINED)")
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
    print(f"  📈 Visualizations:")
    for i in range(1, 6):
        print(f"     - {i}_*.png")
    print()


if __name__ == "__main__":
    main()