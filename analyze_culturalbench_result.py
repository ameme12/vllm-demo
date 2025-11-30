import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

# Country to region mapping (same as in your task file)
COUNTRY_REGIONS = {
    # North America
    'Canada': 'North America',
    'United States': 'North America',
    
    # South America
    'Argentina': 'South America',
    'Brazil': 'South America',
    'Chile': 'South America',
    'Mexico': 'South America',
    'Peru': 'South America',
    
    # East Europe
    'Czech Republic': 'East Europe',
    'Poland': 'East Europe',
    'Romania': 'East Europe',
    'Ukraine': 'East Europe',
    'Russia': 'East Europe',
    
    # South Europe
    'Spain': 'South Europe',
    'Italy': 'South Europe',
    
    # West Europe
    'France': 'West Europe',
    'Germany': 'West Europe',
    'Netherlands': 'West Europe',
    'United Kingdom': 'West Europe',
    
    # Africa
    'Egypt': 'Africa',
    'Morocco': 'Africa',
    'Nigeria': 'Africa',
    'South Africa': 'Africa',
    'Zimbabwe': 'Africa',
    
    # Middle East/West Asia
    'Iran': 'Middle East/West Asia',
    'Israel': 'Middle East/West Asia',
    'Lebanon': 'Middle East/West Asia',
    'Saudi Arabia': 'Middle East/West Asia',
    'Turkey': 'Middle East/West Asia',
    
    # South Asia
    'Bangladesh': 'South Asia',
    'India': 'South Asia',
    'Nepal': 'South Asia',
    'Pakistan': 'South Asia',
    
    # Southeast Asia
    'Indonesia': 'Southeast Asia',
    'Malaysia': 'Southeast Asia',
    'Philippines': 'Southeast Asia',
    'Singapore': 'Southeast Asia',
    'Thailand': 'Southeast Asia',
    'Vietnam': 'Southeast Asia',
    
    # East Asia
    'China': 'East Asia',
    'Hong Kong': 'East Asia',
    'Japan': 'East Asia',
    'South Korea': 'East Asia',
    'Taiwan': 'East Asia',
    
    # Oceania
    'Australia': 'Oceania',
    'New Zealand': 'Oceania',
}

def get_region(country):
    """Get region for a country"""
    return COUNTRY_REGIONS.get(country, 'Unknown')

def load_summary_results(results_dir):
    """Load summary JSON files and extract aggregate metrics"""
    results = []
    
    for json_file in Path(results_dir).rglob("*summary.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Check if this is a summary file
            if 'aggregate_metrics' not in data:
                continue
            
            # Extract model name from path or filename
            path_str = str(json_file).lower()
            if "llama" in path_str:
                model = "Llama 3.2-3B"
            elif "qwen" in path_str:
                model = "Qwen 2.5-3B"
            else:
                model = "Unknown"
            
            # Extract metrics
            metrics = data['aggregate_metrics']
            num_samples = data.get('total_samples', 0)
            
            # Get country/region breakdown if available
            by_country = data.get('by_country', {})
            by_region = data.get('by_region', {})
            
            # If we have country breakdown, add individual country results
            if by_country:
                for country, stats in by_country.items():
                    region = get_region(country)
                    results.append({
                        'model': model,
                        'country': country,
                        'region': region,
                        'accuracy': stats['accuracy'] * 100,
                        'num_samples': stats['count'],
                        'file': json_file.name
                    })
                    print(f"✓ {model:20s} | {country:20s} | {region:20s} | {stats['count']:6,} samples | {stats['accuracy']*100:5.1f}%")
            
            # Also add overall metrics
            results.append({
                'model': model,
                'country': 'OVERALL',
                'region': 'ALL',
                'accuracy': metrics['accuracy']['mean'] * 100,
                'num_samples': num_samples,
                'file': json_file.name
            })
            print(f"✓ {model:20s} | {'OVERALL':20s} | {'ALL':20s} | {num_samples:6,} samples | {metrics['accuracy']['mean']*100:5.1f}%")
            
        except Exception as e:
            print(f"⚠️  Skipped {json_file.name}: {e}")
    
    df = pd.DataFrame(results)
    
    # Remove duplicates (keep entry with most samples for each model+country combo)
    if not df.empty:
        print(f"\n📊 Found {len(df)} result entries")
        df = df.sort_values('num_samples', ascending=False).groupby(['model', 'country']).first().reset_index()
        print(f"📊 After removing duplicates: {len(df)} unique model-country combinations")
    
    return df

def create_comparison_tables(df):
    """Create formatted comparison tables"""
    # Filter out OVERALL for detailed tables
    df_detail = df[df['country'] != 'OVERALL'].copy()
    
    # Pivot table for accuracy by country
    accuracy_by_country = df_detail.pivot(index='country', columns='model', values='accuracy')
    
    # Pivot table for accuracy by region
    region_accuracy = df_detail.groupby(['region', 'model'])['accuracy'].mean().unstack()
    
    # Sample counts
    samples_by_country = df_detail.pivot(index='country', columns='model', values='num_samples')
    
    return accuracy_by_country, region_accuracy, samples_by_country

def plot_region_accuracy_table(df):
    """Create a visual table showing accuracy by region"""
    
    # Filter out OVERALL
    region_df = df[df['country'] != 'OVERALL'].copy()
    
    if region_df.empty:
        return None
    
    # Calculate region averages
    region_pivot = region_df.groupby(['region', 'model'])['accuracy'].mean().unstack()
    region_pivot = region_pivot.sort_values(by=region_pivot.columns[0], ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for region in region_pivot.index:
        row = [region]
        for model in region_pivot.columns:
            val = region_pivot.loc[region, model]
            if pd.isna(val):
                row.append("N/A")
            else:
                row.append(f"{val:.2f}%")
        table_data.append(row)
    
    # Add average row
    avg_row = ["AVERAGE"]
    for model in region_pivot.columns:
        avg_val = region_pivot[model].mean()
        avg_row.append(f"{avg_val:.2f}%")
    table_data.append(avg_row)
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Region'] + list(region_pivot.columns),
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(region_pivot.columns) + 1):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style average row
    last_row = len(table_data)
    for i in range(len(region_pivot.columns) + 1):
        table[(last_row, i)].set_facecolor('#95a5a6')
        table[(last_row, i)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(region_pivot.columns) + 1):
            if i % 2 == 0 and i != last_row:
                table[(i, j)].set_facecolor('#f8f9fa')
    
    ax.set_title('Accuracy by Region - Comparison Table', 
                 fontsize=16, pad=20, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_country_accuracy_table(df):
    """Create a visual table showing accuracy by country (grouped by region)"""
    
    # Filter out OVERALL
    country_df = df[df['country'] != 'OVERALL'].copy()
    
    if country_df.empty:
        return None
    
    # Pivot by country
    country_pivot = country_df.pivot(index='country', columns='model', values='accuracy')
    
    # Add region column for sorting
    country_df_unique = country_df.drop_duplicates('country')
    region_map = dict(zip(country_df_unique['country'], country_df_unique['region']))
    country_pivot['Region'] = country_pivot.index.map(region_map)
    
    # Sort by region then country
    country_pivot = country_pivot.sort_values(['Region', country_pivot.columns[0]], ascending=[True, False])
    
    # Create figure (larger for many countries)
    fig, ax = plt.subplots(figsize=(14, max(12, len(country_pivot) * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    current_region = None
    
    for country in country_pivot.index:
        region = country_pivot.loc[country, 'Region']
        
        # Add region header row
        if region != current_region:
            if current_region is not None:
                # Add separator
                table_data.append([''] * (len(country_pivot.columns)))
            table_data.append([f"▼ {region}"] + [''] * (len(country_pivot.columns) - 1))
            current_region = region
        
        # Add country row
        row = [f"  {country}"]
        for model in country_pivot.columns:
            if model == 'Region':
                continue
            val = country_pivot.loc[country, model]
            if pd.isna(val):
                row.append("N/A")
            else:
                row.append(f"{val:.2f}%")
        table_data.append(row)
    
    # Create table
    model_columns = [col for col in country_pivot.columns if col != 'Region']
    table = ax.table(cellText=table_data,
                    colLabels=['Country'] + model_columns,
                    cellLoc='left',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Style header
    for i in range(len(model_columns) + 1):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style region headers and rows
    for i, row_data in enumerate(table_data, start=1):
        if row_data[0].startswith('▼'):
            # Region header
            for j in range(len(model_columns) + 1):
                table[(i, j)].set_facecolor('#2c3e50')
                table[(i, j)].set_text_props(weight='bold', color='white')
        elif row_data[0] == '':
            # Separator
            for j in range(len(model_columns) + 1):
                table[(i, j)].set_facecolor('#ecf0f1')
        else:
            # Country row
            if i % 2 == 0:
                for j in range(len(model_columns) + 1):
                    table[(i, j)].set_facecolor('#f8f9fa')
    
    ax.set_title('Accuracy by Country (Grouped by Region) - Comparison Table', 
                 fontsize=16, pad=20, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_accuracy_by_region_chart(df):
    """Create bar chart of accuracy by region"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    region_df = df[df['country'] != 'OVERALL'].copy()
    
    if region_df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    region_data = region_df.groupby(['region', 'model'])['accuracy'].mean().unstack()
    region_data = region_data.sort_values(by=region_data.columns[0], ascending=False)
    
    x = np.arange(len(region_data))
    width = 0.35
    
    models = region_data.columns
    for i, model in enumerate(models):
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, region_data[model], width, 
                      label=model, alpha=0.85, color=colors[i],
                      edgecolor='black', linewidth=0.5)
        
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title('CulturalBench Accuracy by Region', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Region', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(region_data.index, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax.axhline(y=25, color='red', linestyle='--', linewidth=2.5, 
               label='Random Baseline (25%)', alpha=0.7)
    ax.set_ylim(0, 105)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig

def plot_accuracy_by_country_chart(df):
    """Create bar chart of accuracy by country"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    country_df = df[df['country'] != 'OVERALL'].copy()
    
    if country_df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    accuracy_pivot = country_df.pivot(index='country', columns='model', values='accuracy')
    accuracy_pivot = accuracy_pivot.sort_values(by=accuracy_pivot.columns[0], ascending=True)
    
    x = np.arange(len(accuracy_pivot))
    width = 0.35
    
    models = accuracy_pivot.columns
    for i, model in enumerate(models):
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.barh(x + offset, accuracy_pivot[model], width, 
                      label=model, alpha=0.85, color=colors[i],
                      edgecolor='black', linewidth=0.5)
        
        for j, bar in enumerate(bars):
            width_val = bar.get_width()
            if not np.isnan(width_val):
                ax.text(width_val + 1, bar.get_y() + bar.get_height()/2.,
                       f'{width_val:.1f}%',
                       ha='left', va='center', fontsize=8, fontweight='bold')
    
    ax.set_title('CulturalBench Accuracy by Country', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Country', fontsize=13, fontweight='bold')
    ax.set_xlabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(accuracy_pivot.index, fontsize=9)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(axis='x', alpha=0.4, linestyle='--', linewidth=0.8)
    ax.axvline(x=25, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
    ax.set_xlim(0, 105)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig

def plot_combined_visualization(df):
    """Create ALL-IN-ONE image with all visualizations"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    fig.suptitle('CulturalBench Complete Analysis Dashboard', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    country_df = df[df['country'] != 'OVERALL']
    overall_df = df[df['country'] == 'OVERALL']
    
    # 1. Overall Performance
    ax1 = fig.add_subplot(gs[0, 0])
    if not overall_df.empty:
        avg_by_model = overall_df.set_index('model')['accuracy'].sort_values(ascending=False)
        bars = ax1.barh(range(len(avg_by_model)), avg_by_model.values, 
                        color=colors[:len(avg_by_model)], alpha=0.8, height=0.6)
        
        for i, (bar, val) in enumerate(zip(bars, avg_by_model.values)):
            ax1.text(val + 1, i, f'{val:.2f}%', 
                    va='center', fontsize=13, fontweight='bold')
        
        ax1.set_yticks(range(len(avg_by_model)))
        ax1.set_yticklabels(avg_by_model.index, fontsize=13)
        ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Overall Performance', fontsize=14, fontweight='bold', pad=10)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.set_xlim(0, 100)
        ax1.axvline(x=25, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # 2. Region Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    if not country_df.empty:
        region_avg = country_df.groupby(['region', 'model'])['accuracy'].mean().unstack()
        region_avg = region_avg.sort_values(by=region_avg.columns[0], ascending=False)
        
        x = np.arange(len(region_avg))
        width = 0.35
        
        for i, model in enumerate(region_avg.columns):
            offset = width * (i - len(region_avg.columns)/2 + 0.5)
            bars = ax2.bar(x + offset, region_avg[model], width, 
                          label=model, alpha=0.85, color=colors[i])
            
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.1f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(region_avg.index, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Accuracy by Region', fontsize=14, fontweight='bold', pad=10)
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim(0, 100)
        ax2.axhline(y=25, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # 3. Top 10 Countries
    ax3 = fig.add_subplot(gs[1, 0])
    if not country_df.empty:
        country_avg = country_df.groupby('country')['accuracy'].mean().sort_values(ascending=False).head(10)
        country_pivot = country_df[country_df['country'].isin(country_avg.index)].pivot(
            index='country', columns='model', values='accuracy')
        country_pivot = country_pivot.reindex(country_avg.index)
        
        x = np.arange(len(country_pivot))
        width = 0.35
        
        for i, model in enumerate(country_pivot.columns):
            offset = width * (i - len(country_pivot.columns)/2 + 0.5)
            ax3.barh(x + offset, country_pivot[model], width, 
                    label=model, alpha=0.85, color=colors[i])
        
        ax3.set_yticks(x)
        ax3.set_yticklabels(country_pivot.index, fontsize=10)
        ax3.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Top 10 Countries', fontsize=14, fontweight='bold', pad=10)
        ax3.legend(fontsize=10)
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        ax3.set_xlim(0, 100)
    
    # 4. Bottom 10 Countries
    ax4 = fig.add_subplot(gs[1, 1])
    if not country_df.empty:
        country_avg_bottom = country_df.groupby('country')['accuracy'].mean().sort_values(ascending=True).head(10)
        country_pivot_bottom = country_df[country_df['country'].isin(country_avg_bottom.index)].pivot(
            index='country', columns='model', values='accuracy')
        country_pivot_bottom = country_pivot_bottom.reindex(country_avg_bottom.index)
        
        x = np.arange(len(country_pivot_bottom))
        width = 0.35
        
        for i, model in enumerate(country_pivot_bottom.columns):
            offset = width * (i - len(country_pivot_bottom.columns)/2 + 0.5)
            ax4.barh(x + offset, country_pivot_bottom[model], width, 
                    label=model, alpha=0.85, color=colors[i])
        
        ax4.set_yticks(x)
        ax4.set_yticklabels(country_pivot_bottom.index, fontsize=10)
        ax4.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Bottom 10 Countries', fontsize=14, fontweight='bold', pad=10)
        ax4.legend(fontsize=10)
        ax4.grid(axis='x', alpha=0.3, linestyle='--')
        ax4.set_xlim(0, 100)
    
    # 5. Sample Distribution
    ax5 = fig.add_subplot(gs[2, 0])
    if not country_df.empty:
        samples_by_region = country_df.groupby('region')['num_samples'].sum().sort_values(ascending=True)
        bars = ax5.barh(range(len(samples_by_region)), samples_by_region.values,
                        color='#9b59b6', alpha=0.7)
        
        for i, (bar, val) in enumerate(zip(bars, samples_by_region.values)):
            ax5.text(val + 5, i, f'{val:,}', 
                    va='center', fontsize=10, fontweight='bold')
        
        ax5.set_yticks(range(len(samples_by_region)))
        ax5.set_yticklabels(samples_by_region.index, fontsize=11)
        ax5.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
        ax5.set_title('Dataset Coverage by Region', fontsize=14, fontweight='bold', pad=10)
        ax5.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 6. Head-to-Head
    ax6 = fig.add_subplot(gs[2, 1])
    if len(df['model'].unique()) >= 2:
        model1, model2 = sorted(df['model'].unique())[:2]
        
        df1 = country_df[country_df['model'] == model1].set_index('country')['accuracy']
        df2 = country_df[country_df['model'] == model2].set_index('country')['accuracy']
        
        common = df1.index.intersection(df2.index)
        
        if len(common) > 0:
            ax6.scatter(df1[common], df2[common], s=120, alpha=0.6, 
                       color='#e74c3c', edgecolors='black', linewidth=1)
            
            for country in common:
                diff = abs(df1[country] - df2[country])
                if diff > 15:
                    ax6.annotate(country, (df1[country], df2[country]),
                               xytext=(3, 3), textcoords='offset points',
                               fontsize=8, fontweight='bold')
            
            min_val = min(df1[common].min(), df2[common].min())
            max_val = max(df1[common].max(), df2[common].max())
            ax6.plot([min_val, max_val], [min_val, max_val], 
                    'k--', alpha=0.5, linewidth=2, label='Equal')
            
            ax6.set_xlabel(f'{model1} Accuracy (%)', fontsize=12, fontweight='bold')
            ax6.set_ylabel(f'{model2} Accuracy (%)', fontsize=12, fontweight='bold')
            ax6.set_title('Head-to-Head Comparison', fontsize=14, fontweight='bold', pad=10)
            ax6.grid(alpha=0.3, linestyle='--')
            ax6.legend(fontsize=10)
            ax6.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def print_summary_report(df):
    """Print summary report"""
    print("\n" + "="*80)
    print(" "*20 + "CULTURALBENCH EVALUATION SUMMARY")
    print("="*80)
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        overall = model_df[model_df['country'] == 'OVERALL']
        
        print(f"\n📊 {model}")
        print("-"*80)
        
        if not overall.empty:
            print(f"  Overall Accuracy: {overall['accuracy'].iloc[0]:.2f}%")
        
        country_df = model_df[model_df['country'] != 'OVERALL']
        if not country_df.empty:
            print(f"  Countries: {len(country_df)}")
            print(f"  Best: {country_df.loc[country_df['accuracy'].idxmax(), 'country']} "
                  f"({country_df['accuracy'].max():.2f}%)")
            print(f"  Worst: {country_df.loc[country_df['accuracy'].idxmin(), 'country']} "
                  f"({country_df['accuracy'].min():.2f}%)")

def plot_region_detail(df, region_name):
    """Create detailed bar plot for a specific region showing all countries in that region"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # Filter for this region
    region_df = df[(df['region'] == region_name) & (df['country'] != 'OVERALL')].copy()
    
    if region_df.empty:
        return None
    
    # Pivot by country
    country_pivot = region_df.pivot(index='country', columns='model', values='accuracy')
    samples_pivot = region_df.pivot(index='country', columns='model', values='num_samples')
    
    # Sort by average accuracy
    country_pivot['avg'] = country_pivot.mean(axis=1)
    country_pivot = country_pivot.sort_values('avg', ascending=True)
    country_pivot = country_pivot.drop('avg', axis=1)
    samples_pivot = samples_pivot.reindex(country_pivot.index)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(country_pivot) * 0.5)))
    
    x = np.arange(len(country_pivot))
    width = 0.35
    
    models = country_pivot.columns
    for i, model in enumerate(models):
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.barh(x + offset, country_pivot[model], width, 
                      label=model, alpha=0.85, color=colors[i],
                      edgecolor='black', linewidth=0.5)
        
        # Add accuracy labels only (no sample counts on bars)
        for j, bar in enumerate(bars):
            width_val = bar.get_width()
            
            if not np.isnan(width_val):
                # Accuracy percentage
                ax.text(width_val + 1, bar.get_y() + bar.get_height()/2.,
                       f'{width_val:.1f}%',
                       ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.set_title(f'{region_name} - CulturalBench Evaluation', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Country', fontsize=13, fontweight='bold')
    ax.set_xlabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(country_pivot.index, fontsize=11)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)  # Changed to upper right
    ax.grid(axis='x', alpha=0.4, linestyle='--', linewidth=0.8)
    ax.axvline(x=25, color='red', linestyle='--', linewidth=2, 
               label='Random (25%)', alpha=0.7)
    ax.set_xlim(0, 105)
    ax.set_facecolor('#f8f9fa')
    
    # Add country count in TOP RIGHT corner (below legend)
    num_countries = len(country_pivot)
    ax.text(0.98, 0.75, f'Countries: {num_countries}',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='right',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=0.5))
    
    # Add total samples in BOTTOM RIGHT corner
    total_samples = region_df.groupby('model')['num_samples'].sum()
    sample_text = f'Total samples: {total_samples.iloc[0]:,}'
    ax.text(0.98, 0.02, sample_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='bottom',
            horizontalalignment='right',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, pad=0.5))
    
    plt.tight_layout()
    return fig

def main():
    """Main function"""
    
    results_dir = "results_culturalbench"
    
    print("\n🔍 Scanning for result files...")
    df = load_summary_results(results_dir)
    
    if df.empty:
        print("\n❌ No summary JSON files found!")
        return
    
    print(f"\n✓ Loaded {len(df)} results")
    
    print_summary_report(df)
    
    output_dir = Path("results_culturalbench")
    output_dir.mkdir(exist_ok=True)
    
    # Create region-specific subdirectory
    regions_dir = output_dir / "by_region"
    regions_dir.mkdir(exist_ok=True)
    
    print("\n📊 Creating visualizations...")
    
    # 1. Region accuracy TABLE
    fig = plot_region_accuracy_table(df)
    if fig:
        fig.savefig(output_dir / '1_table_region_accuracy.png', dpi=300, bbox_inches='tight')
        print("  ✓ 1_table_region_accuracy.png")
        plt.close(fig)
    
    # 2. Country accuracy TABLE
    fig = plot_country_accuracy_table(df)
    if fig:
        fig.savefig(output_dir / '2_table_country_accuracy.png', dpi=300, bbox_inches='tight')
        print("  ✓ 2_table_country_accuracy.png")
        plt.close(fig)
    
    # 3. Region accuracy CHART
    fig = plot_accuracy_by_region_chart(df)
    if fig:
        fig.savefig(output_dir / '3_chart_region_accuracy.png', dpi=300, bbox_inches='tight')
        print("  ✓ 3_chart_region_accuracy.png")
        plt.close(fig)
    
    # 4. Country accuracy CHART
    fig = plot_accuracy_by_country_chart(df)
    if fig:
        fig.savefig(output_dir / '4_chart_country_accuracy.png', dpi=300, bbox_inches='tight')
        print("  ✓ 4_chart_country_accuracy.png")
        plt.close(fig)
    
    # 5. COMBINED - ALL IN ONE
    fig = plot_combined_visualization(df)
    fig.savefig(output_dir / '5_COMPLETE_ALL_IN_ONE.png', dpi=300, bbox_inches='tight')
    print("  ✓ 5_COMPLETE_ALL_IN_ONE.png (⭐ COMBINED)")
    plt.close(fig)
    
    # 6. INDIVIDUAL REGION PLOTS (NEW!)
    print("\n📍 Creating individual region plots...")
    country_df = df[df['country'] != 'OVERALL']
    
    # Only include these specific regions
    regions_to_plot = [
        'Africa',
        'East Asia',
        'North America',
        'West Europe',
        'Middle East/West Asia',  # This is "West Asia" in the data
        'South Asia'
    ]
    
    # Get all available regions and filter
    all_regions = sorted(country_df['region'].unique())
    regions = [r for r in all_regions if r in regions_to_plot]
    
    print(f"   Creating plots for {len(regions)} regions: {', '.join(regions)}")
    
    for i, region in enumerate(regions, start=1):
        fig = plot_region_detail(df, region)
        if fig:
            # Create clean filename
            safe_region_name = region.replace('/', '_').replace(' ', '_')
            filename = f'region_{i:02d}_{safe_region_name}.png'
            fig.savefig(regions_dir / filename, dpi=300, bbox_inches='tight')
            print(f"  ✓ by_region/{filename}")
            plt.close(fig)
    
    # Export CSVs
    accuracy_by_country, accuracy_by_region, _ = create_comparison_tables(df)
    if not accuracy_by_country.empty:
        accuracy_by_country.round(2).to_csv(output_dir / 'accuracy_by_country.csv')
    if not accuracy_by_region.empty:
        accuracy_by_region.round(2).to_csv(output_dir / 'accuracy_by_region.csv')
    df.to_csv(output_dir / 'all_results.csv', index=False)
    print("\n  ✓ CSV files")
    
    print(f"\n✅ Complete! All files in {output_dir}/")
    print(f"   Region details in {regions_dir}/\n")

if __name__ == "__main__":
    main()