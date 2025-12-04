"""
DLAMA-v1 Results Visualization Script (Universal)
Analyzes culture evaluation results for any culture pair (Arab-West, Asia-West, etc.)
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List

# Predicate descriptions
PREDICATE_DESCRIPTIONS = {
    'P17': 'Country', 'P19': 'Place of birth', 'P20': 'Place of death',
    'P27': 'Country of citizenship', 'P30': 'Continent', 'P36': 'Capital',
    'P37': 'Official language', 'P47': 'Shares border with', 'P103': 'Native language',
    'P106': 'Occupation', 'P136': 'Genre', 'P190': 'Sister city',
    'P264': 'Record label', 'P364': 'Original language of work', 'P449': 'Original network',
    'P495': 'Country of origin', 'P530': 'Diplomatic relation', 'P1303': 'Instrument',
    'P1376': 'Capital of', 'P1412': 'Languages spoken or published',
}

# Color scheme for cultures (will assign automatically)
CULTURE_COLORS = {
    'Arab': '#e74c3c',      # Red
    'Western': '#3498db',   # Blue
    'Asia': '#2ecc71',      # Green
    'Asian': '#2ecc71',     # Green (alternative name)
    'default_1': '#e74c3c', # Red for first culture
    'default_2': '#3498db', # Blue for second culture
}


def get_culture_color(culture: str, culture_index: int = 0) -> str:
    """Get color for a culture, either from predefined or default palette"""
    if culture in CULTURE_COLORS:
        return CULTURE_COLORS[culture]
    return CULTURE_COLORS[f'default_{culture_index + 1}']


def load_dlama_results(results_file: Path) -> Dict:
    """Load DLAMA results from JSON file"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data


def extract_results_dataframe(data: Dict) -> pd.DataFrame:
    """Extract results into a structured DataFrame"""
    results_list = []
    
    for result in data['results']:
        results_list.append({
            'sample_id': result['sample_id'],
            'subject': result['subject'],
            'predicate': result['predicate'],
            'predicate_code': result['predicate_code'],
            'correct_answer': result['correct_answer'],
            'culture': result['culture'],
            'country': result['country'],
            'prediction': result['prediction'],
            'extracted_answer': result['extracted_answer'],
            'exact_match': result['metrics']['exact_match'],
            'overlap': result['metrics']['overlap'],
        })
    
    return pd.DataFrame(results_list)


def create_culture_comparison_table(df: pd.DataFrame, metric: str = 'overlap') -> pd.DataFrame:
    """
    Create a simple comparison table for all cultures in the dataset
    
    Args:
        df: Results dataframe
        metric: 'overlap' or 'exact_match'
    """
    table_data = []
    
    # Auto-detect cultures from dataframe
    cultures = sorted(df['culture'].unique())
    
    for culture in cultures:
        culture_df = df[df['culture'] == culture]
        accuracy = culture_df[metric].mean() * 100
        count = len(culture_df)
        
        table_data.append({
            'Culture': culture,
            'Accuracy (%)': round(accuracy, 2),
            'Samples': count
        })
    
    return pd.DataFrame(table_data)


def create_detailed_breakdown_table(df: pd.DataFrame, metric: str = 'overlap') -> pd.DataFrame:
    """
    Create detailed table with Culture, Country, Predicate breakdown
    
    Args:
        df: Results dataframe
        metric: 'overlap' or 'exact_match'
    """
    table_data = []
    
    # Group by culture, country, predicate
    for culture in df['culture'].unique():
        for country in df[df['culture'] == culture]['country'].unique():
            for pred_code in df[(df['culture'] == culture) & (df['country'] == country)]['predicate_code'].unique():
                subset = df[
                    (df['culture'] == culture) & 
                    (df['country'] == country) & 
                    (df['predicate_code'] == pred_code)
                ]
                
                accuracy = subset[metric].mean() * 100
                count = len(subset)
                
                pred_desc = PREDICATE_DESCRIPTIONS.get(pred_code, pred_code)
                
                table_data.append({
                    'Culture': culture,
                    'Country': country,
                    'Predicate': f"{pred_code} ({pred_desc})",
                    'Accuracy (%)': round(accuracy, 2),
                    'Samples': count
                })
    
    result_df = pd.DataFrame(table_data)
    result_df = result_df.sort_values(['Culture', 'Country', 'Accuracy (%)'], ascending=[True, True, False])
    
    return result_df


def plot_culture_comparison_table(df: pd.DataFrame, metric: str = 'overlap', model_name: str = 'Model'):
    """Create visual table for culture comparison"""
    
    table_df = create_culture_comparison_table(df, metric)
    cultures = sorted(df['culture'].unique())
    culture_names = ' vs '.join(cultures)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in table_df.iterrows():
        table_data.append([
            row['Culture'],
            f"{row['Accuracy (%)']:.2f}%",
            f"{row['Samples']:,}"
        ])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Culture', 'Accuracy (%)', 'Samples'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=13)
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f8f9fa')
            table[(i, j)].set_text_props(fontsize=12)
    
    metric_name = 'Overlap' if metric == 'overlap' else 'Exact Match'
    ax.set_title(
        f'DLAMA-v1: {culture_names} Culture Comparison\n'
        f'Model: {model_name} | Metric: {metric_name}',
        fontsize=15, pad=20, fontweight='bold'
    )
    
    plt.tight_layout()
    return fig


def plot_detailed_breakdown_table(df: pd.DataFrame, metric: str = 'overlap', model_name: str = 'Model',
                                 min_samples: int = 100, top_n_countries: int = 5):
    """
    Create visual table with detailed breakdown by culture, country, predicate.
    Only shows:
    - Top 5 predicates with >min_samples samples
    - Top 5 and bottom 5 countries per culture (with >min_samples samples)
    """
    
    # Filter to top 5 predicates with >min_samples
    pred_counts = df.groupby('predicate_code').size()
    valid_predicates = pred_counts[pred_counts >= min_samples].index
    pred_accuracy = df[df['predicate_code'].isin(valid_predicates)].groupby('predicate_code')[metric].mean()
    top_predicates = pred_accuracy.nlargest(5).index.tolist()
    
    # Filter dataframe to only top predicates
    df_filtered = df[df['predicate_code'].isin(top_predicates)].copy()
    
    # For each culture, get top 5 and bottom 5 countries (with >min_samples)
    selected_countries = {}
    for culture in df_filtered['culture'].unique():
        culture_df = df_filtered[df_filtered['culture'] == culture]
        
        # Calculate country accuracy and counts
        country_accuracy = culture_df.groupby('country')[metric].mean()
        country_counts = culture_df.groupby('country').size()
        
        # Filter countries with minimum sample count
        valid_countries = country_counts[country_counts >= min_samples].index
        country_accuracy_filtered = country_accuracy[valid_countries]
        
        # Get top and bottom countries from filtered set
        top_countries = country_accuracy_filtered.nlargest(top_n_countries).index.tolist()
        bottom_countries = country_accuracy_filtered.nsmallest(top_n_countries).index.tolist()
        
        # Combine and deduplicate (in case there's overlap)
        selected_countries[culture] = list(set(top_countries + bottom_countries))
    
    # Build table data
    table_data_list = []
    
    for culture in sorted(df_filtered['culture'].unique()):
        culture_df = df_filtered[df_filtered['culture'] == culture]
        
        # Get countries for this culture
        countries = selected_countries[culture]
        country_accuracy = culture_df.groupby('country')[metric].mean()
        countries_sorted = country_accuracy[countries].sort_values(ascending=False).index.tolist()
        
        for country in countries_sorted:
            country_df = culture_df[culture_df['country'] == country]
            
            for pred_code in top_predicates:
                subset = country_df[country_df['predicate_code'] == pred_code]
                
                if len(subset) == 0:
                    continue
                
                accuracy = subset[metric].mean() * 100
                count = len(subset)
                
                pred_desc = PREDICATE_DESCRIPTIONS.get(pred_code, pred_code)
                
                table_data_list.append({
                    'Culture': culture,
                    'Country': country,
                    'Predicate': f"{pred_code} ({pred_desc})",
                    'Accuracy (%)': round(accuracy, 2),
                    'Samples': count
                })
    
    table_df = pd.DataFrame(table_data_list)
    table_df = table_df.sort_values(['Culture', 'Country', 'Accuracy (%)'], ascending=[True, True, False])
    
    # Create larger figure for detailed table
    num_rows = len(table_df)
    fig_height = max(8, num_rows * 0.3)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    current_culture = None
    current_country = None
    
    for _, row in table_df.iterrows():
        # Add culture header
        if row['Culture'] != current_culture:
            if current_culture is not None:
                table_data.append(['', '', '', '', ''])  # Separator
            table_data.append([f"▼ {row['Culture']} Culture", '', '', '', ''])
            current_culture = row['Culture']
            current_country = None
        
        # Add country sub-header
        if row['Country'] != current_country:
            table_data.append([f"  ◆ {row['Country']}", '', '', '', ''])
            current_country = row['Country']
        
        # Add predicate row
        table_data.append([
            '',  # Empty for indent
            row['Predicate'],
            f"{row['Accuracy (%)']:.2f}%",
            f"{row['Samples']:,}",
            ''
        ])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Culture/Country', 'Predicate', 'Accuracy (%)', 'Samples', ''],
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Style rows
    for i, row_data in enumerate(table_data, start=1):
        if row_data[0].startswith('▼'):
            # Culture header
            for j in range(5):
                table[(i, j)].set_facecolor('#2c3e50')
                table[(i, j)].set_text_props(weight='bold', color='white', fontsize=10)
        elif row_data[0].startswith('  ◆'):
            # Country header
            for j in range(5):
                table[(i, j)].set_facecolor('#34495e')
                table[(i, j)].set_text_props(weight='bold', color='white', fontsize=9)
        elif row_data[0] == '' and row_data[1] == '':
            # Separator
            for j in range(5):
                table[(i, j)].set_facecolor('#ecf0f1')
        else:
            # Data row
            if i % 2 == 0:
                for j in range(5):
                    table[(i, j)].set_facecolor('#f8f9fa')
    
    metric_name = 'Overlap' if metric == 'overlap' else 'Exact Match'
    ax.set_title(
        f'DLAMA-v1: Detailed Breakdown - Top 5 Predicates & Countries (>{min_samples} samples)\n'
        f'Top & Bottom {top_n_countries} Countries per Culture | Model: {model_name} | Metric: {metric_name}',
        fontsize=14, pad=20, fontweight='bold'
    )
    
    plt.tight_layout()
    return fig


def plot_top_bottom_predicates_by_culture(df: pd.DataFrame, metric: str = 'overlap', 
                                          top_n: int = 5, model_name: str = 'Model',
                                          min_samples: int = 100):
    """
    Create bar plots for top 5 and bottom 5 predicates, separated by cultures.
    Only considers predicates with at least min_samples.
    """
    
    # Auto-detect cultures
    cultures = sorted(df['culture'].unique())
    
    fig, axes = plt.subplots(2, len(cultures), figsize=(9*len(cultures), 12))
    if len(cultures) == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(
        f'DLAMA-v1: Top & Bottom Predicates by Culture (min. {min_samples} samples)\n'
        f'Model: {model_name} | Metric: {metric.capitalize()}',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    for culture_idx, culture in enumerate(cultures):
        culture_df = df[df['culture'] == culture]
        
        # Calculate accuracy by predicate
        pred_accuracy = culture_df.groupby('predicate_code')[metric].mean() * 100
        pred_counts = culture_df.groupby('predicate_code').size()
        
        # Filter predicates with minimum sample count
        valid_predicates = pred_counts[pred_counts >= min_samples].index
        pred_accuracy_filtered = pred_accuracy[valid_predicates]
        
        # Get top and bottom predicates from filtered set
        top_preds = pred_accuracy_filtered.nlargest(top_n)
        bottom_preds = pred_accuracy_filtered.nsmallest(top_n)
        
        # Create labels with descriptions
        top_labels = [f"{p}\n{PREDICATE_DESCRIPTIONS.get(p, p)}" for p in top_preds.index]
        bottom_labels = [f"{p}\n{PREDICATE_DESCRIPTIONS.get(p, p)}" for p in bottom_preds.index]
        
        color = get_culture_color(culture, culture_idx)
        
        # Plot top predicates
        ax_top = axes[0, culture_idx]
        bars = ax_top.barh(range(len(top_preds)), top_preds.values, 
                          color=color, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, pred) in enumerate(zip(bars, top_preds.index)):
            width = bar.get_width()
            count = pred_counts[pred]
            ax_top.text(width + 1, i, f'{width:.1f}% (n={count})',
                       va='center', fontsize=9, fontweight='bold')
        
        ax_top.set_yticks(range(len(top_preds)))
        ax_top.set_yticklabels(top_labels, fontsize=9)
        ax_top.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax_top.set_title(f'Top {top_n} Predicates - {culture} Culture', 
                        fontsize=12, fontweight='bold', pad=10)
        ax_top.grid(axis='x', alpha=0.3, linestyle='--')
        ax_top.set_xlim(0, 105)
        ax_top.set_facecolor('#f8f9fa')
        
        # Plot bottom predicates
        ax_bottom = axes[1, culture_idx]
        bars = ax_bottom.barh(range(len(bottom_preds)), bottom_preds.values,
                             color=color, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, pred) in enumerate(zip(bars, bottom_preds.index)):
            width = bar.get_width()
            count = pred_counts[pred]
            ax_bottom.text(width + 1, i, f'{width:.1f}% (n={count})',
                          va='center', fontsize=9, fontweight='bold')
        
        ax_bottom.set_yticks(range(len(bottom_preds)))
        ax_bottom.set_yticklabels(bottom_labels, fontsize=9)
        ax_bottom.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax_bottom.set_title(f'Bottom {top_n} Predicates - {culture} Culture',
                           fontsize=12, fontweight='bold', pad=10)
        ax_bottom.grid(axis='x', alpha=0.3, linestyle='--')
        ax_bottom.set_xlim(0, 105)
        ax_bottom.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig


def plot_top_bottom_countries_by_culture(df: pd.DataFrame, metric: str = 'overlap',
                                        top_n: int = 5, model_name: str = 'Model',
                                        min_samples: int = 50):
    """
    Create bar plots for top 5 and bottom 5 countries for each culture.
    Only considers countries with at least min_samples.
    """
    
    # Auto-detect cultures
    cultures = sorted(df['culture'].unique())
    
    fig, axes = plt.subplots(2, len(cultures), figsize=(8*len(cultures), 12))
    if len(cultures) == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(
        f'DLAMA-v1: Top & Bottom Countries by Culture (min. {min_samples} samples)\n'
        f'Model: {model_name} | Metric: {metric.capitalize()}',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    for culture_idx, culture in enumerate(cultures):
        culture_df = df[df['culture'] == culture]
        
        # Calculate accuracy by country
        country_accuracy = culture_df.groupby('country')[metric].mean() * 100
        country_counts = culture_df.groupby('country').size()
        
        # Filter countries with minimum sample count
        valid_countries = country_counts[country_counts >= min_samples].index
        country_accuracy_filtered = country_accuracy[valid_countries]
        
        # Get top and bottom countries from filtered set
        top_countries = country_accuracy_filtered.nlargest(top_n)
        bottom_countries = country_accuracy_filtered.nsmallest(top_n)
        
        color = get_culture_color(culture, culture_idx)
        
        # Plot top countries
        ax_top = axes[0, culture_idx]
        bars = ax_top.barh(range(len(top_countries)), top_countries.values,
                          color=color, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, country) in enumerate(zip(bars, top_countries.index)):
            width = bar.get_width()
            count = country_counts[country]
            ax_top.text(width + 1, i, f'{width:.1f}% (n={count})',
                       va='center', fontsize=10, fontweight='bold')
        
        ax_top.set_yticks(range(len(top_countries)))
        ax_top.set_yticklabels(top_countries.index, fontsize=11)
        ax_top.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax_top.set_title(f'Top {top_n} Countries - {culture} Culture',
                        fontsize=12, fontweight='bold', pad=10)
        ax_top.grid(axis='x', alpha=0.3, linestyle='--')
        ax_top.set_xlim(0, 105)
        ax_top.set_facecolor('#f8f9fa')
        
        # Plot bottom countries
        ax_bottom = axes[1, culture_idx]
        bars = ax_bottom.barh(range(len(bottom_countries)), bottom_countries.values,
                             color=color, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, country) in enumerate(zip(bars, bottom_countries.index)):
            width = bar.get_width()
            count = country_counts[country]
            ax_bottom.text(width + 1, i, f'{width:.1f}% (n={count})',
                          va='center', fontsize=10, fontweight='bold')
        
        ax_bottom.set_yticks(range(len(bottom_countries)))
        ax_bottom.set_yticklabels(bottom_countries.index, fontsize=11)
        ax_bottom.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax_bottom.set_title(f'Bottom {top_n} Countries - {culture} Culture',
                           fontsize=12, fontweight='bold', pad=10)
        ax_bottom.grid(axis='x', alpha=0.3, linestyle='--')
        ax_bottom.set_xlim(0, 105)
        ax_bottom.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig


def create_summary_statistics_table(df: pd.DataFrame, metric: str = 'overlap', 
                                    model_name: str = 'Model'):
    """Create a comprehensive summary statistics table"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    
    # Overall statistics
    overall_acc = df[metric].mean() * 100
    overall_count = len(df)
    table_data.append(['OVERALL', f'{overall_acc:.2f}%', f'{overall_count:,}', ''])
    table_data.append(['', '', '', ''])  # Separator
    
    # By culture
    for culture in sorted(df['culture'].unique()):
        culture_df = df[df['culture'] == culture]
        acc = culture_df[metric].mean() * 100
        count = len(culture_df)
        table_data.append([f'{culture} Culture', f'{acc:.2f}%', f'{count:,}', ''])
        
        # Countries in this culture
        for country in sorted(culture_df['country'].unique()):
            country_df = culture_df[culture_df['country'] == country]
            country_acc = country_df[metric].mean() * 100
            country_count = len(country_df)
            table_data.append([f'  → {country}', f'{country_acc:.2f}%', f'{country_count:,}', ''])
        
        table_data.append(['', '', '', ''])  # Separator
    
    # Top predicates overall
    pred_accuracy = df.groupby('predicate_code')[metric].mean() * 100
    pred_counts = df.groupby('predicate_code').size()
    top_preds = pred_accuracy.nlargest(5)
    
    table_data.append(['TOP 5 PREDICATES', '', '', ''])
    for pred in top_preds.index:
        pred_desc = PREDICATE_DESCRIPTIONS.get(pred, pred)
        table_data.append([
            f'  {pred} ({pred_desc})',
            f'{top_preds[pred]:.2f}%',
            f'{pred_counts[pred]:,}',
            ''
        ])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Category', 'Accuracy (%)', 'Samples', ''],
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style rows
    for i, row_data in enumerate(table_data, start=1):
        if row_data[0] in ['OVERALL', 'TOP 5 PREDICATES']:
            for j in range(4):
                table[(i, j)].set_facecolor('#2c3e50')
                table[(i, j)].set_text_props(weight='bold', color='white')
        elif row_data[0].endswith('Culture'):
            for j in range(4):
                table[(i, j)].set_facecolor('#34495e')
                table[(i, j)].set_text_props(weight='bold', color='white')
        elif row_data[0] == '':
            for j in range(4):
                table[(i, j)].set_facecolor('#ecf0f1')
        else:
            if i % 2 == 0:
                for j in range(4):
                    table[(i, j)].set_facecolor('#f8f9fa')
    
    metric_name = 'Overlap' if metric == 'overlap' else 'Exact Match'
    ax.set_title(
        f'DLAMA-v1: Summary Statistics\n'
        f'Model: {model_name} | Metric: {metric_name}',
        fontsize=15, pad=20, fontweight='bold'
    )
    
    plt.tight_layout()
    return fig


def plot_accuracy_by_culture_top_predicates(df: pd.DataFrame, metric: str = 'overlap',
                                            model_name: str = 'Model',
                                            min_samples: int = 100):
    """
    Create bar plot showing accuracy by culture for top 5 predicates.
    Only considers predicates with at least min_samples.
    Works with any number of cultures.
    """
    
    # Filter to top 5 predicates with >min_samples
    pred_counts = df.groupby('predicate_code').size()
    valid_predicates = pred_counts[pred_counts >= min_samples].index
    pred_accuracy = df[df['predicate_code'].isin(valid_predicates)].groupby('predicate_code')[metric].mean()
    top_predicates = pred_accuracy.nlargest(5).index.tolist()
    
    # Filter dataframe to only top predicates
    df_filtered = df[df['predicate_code'].isin(top_predicates)].copy()
    
    # Auto-detect cultures
    cultures = sorted(df_filtered['culture'].unique())
    
    # Calculate accuracy by culture and predicate
    results = []
    for pred_code in top_predicates:
        pred_df = df_filtered[df_filtered['predicate_code'] == pred_code]
        
        for culture in cultures:
            culture_df = pred_df[pred_df['culture'] == culture]
            if len(culture_df) > 0:
                accuracy = culture_df[metric].mean() * 100
                count = len(culture_df)
                pred_desc = PREDICATE_DESCRIPTIONS.get(pred_code, pred_code)
                
                results.append({
                    'predicate': pred_code,
                    'predicate_desc': pred_desc,
                    'culture': culture,
                    'accuracy': accuracy,
                    'count': count
                })
    
    results_df = pd.DataFrame(results)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for grouped bar chart
    predicates = top_predicates
    x = np.arange(len(predicates))
    width = 0.8 / len(cultures)  # Dynamic width based on number of cultures
    
    # Plot bars for each culture
    for idx, culture in enumerate(cultures):
        accuracies = []
        counts = []
        
        for pred in predicates:
            culture_data = results_df[(results_df['predicate'] == pred) & (results_df['culture'] == culture)]
            accuracies.append(culture_data['accuracy'].values[0] if len(culture_data) > 0 else 0)
            counts.append(culture_data['count'].values[0] if len(culture_data) > 0 else 0)
        
        offset = width * (idx - len(cultures)/2 + 0.5)
        color = get_culture_color(culture, idx)
        
        bars = ax.bar(x + offset, accuracies, width, label=f'{culture} Culture',
                     color=color, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%\n(n={count})',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Predicate', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    
    culture_names = ' vs '.join(cultures)
    ax.set_title(
        f'DLAMA-v1: Accuracy by Culture for Top 5 Predicates (min. {min_samples} samples)\n'
        f'{culture_names} | Model: {model_name} | Metric: {metric.capitalize()}',
        fontsize=14, fontweight='bold', pad=20
    )
    
    # Set x-axis labels with predicate codes and descriptions
    labels = [f"{pred}\n{PREDICATE_DESCRIPTIONS.get(pred, pred)}" for pred in predicates]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 105)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize DLAMA-v1 results (works with any culture pair)")
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to DLAMA results JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/dlama/visualizations",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Model",
        help="Model name for labels"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=['overlap', 'exact_match'],
        default='overlap',
        help="Metric to visualize"
    )
    
    args = parser.parse_args()
    
    # Update global variables
    results_file = Path(args.results_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = args.model_name
    metric = args.metric
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        exit(1)
    
    print("\n" + "="*70)
    print("🔍 DLAMA-v1 Results Visualization (Universal)")
    print("="*70)
    
    # Load results
    print(f"\n📂 Loading results from: {results_file}")
    data = load_dlama_results(results_file)
    
    print(f"   Experiment: {data['experiment_name']}")
    print(f"   Timestamp: {data['timestamp']}")
    print(f"   Total samples: {data['num_samples']}")
    
    # Extract dataframe
    df = extract_results_dataframe(data)
    print(f"\n✓ Loaded {len(df)} results into DataFrame")
    print(f"   Cultures: {df['culture'].unique().tolist()}")
    print(f"   Countries: {df['country'].nunique()} unique")
    print(f"   Predicates: {df['predicate_code'].nunique()} unique")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    
    # 1. Culture comparison table
    print("   1. Culture comparison table...")
    fig1 = plot_culture_comparison_table(df, metric, model_name)
    fig1.savefig(output_dir / f'1_culture_comparison_table_{metric}.png', 
                 dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Detailed breakdown table
    print("   2. Detailed breakdown table...")
    fig2 = plot_detailed_breakdown_table(df, metric, model_name, min_samples=100, top_n_countries=5)
    fig2.savefig(output_dir / f'2_detailed_breakdown_table_{metric}.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Top/Bottom predicates by culture
    print("   3. Top & bottom predicates by culture...")
    fig3 = plot_top_bottom_predicates_by_culture(df, metric, 5, model_name, min_samples=100)
    fig3.savefig(output_dir / f'3_top_bottom_predicates_{metric}.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Top/Bottom countries by culture
    print("   4. Top & bottom countries by culture...")
    fig4 = plot_top_bottom_countries_by_culture(df, metric, 5, model_name, min_samples=50)
    fig4.savefig(output_dir / f'4_top_bottom_countries_{metric}.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. Summary statistics table
    print("   5. Summary statistics table...")
    fig5 = create_summary_statistics_table(df, metric, model_name)
    fig5.savefig(output_dir / f'5_summary_statistics_{metric}.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    # 6. Accuracy by culture for top predicates
    print("   6. Accuracy by culture for top predicates...")
    fig6 = plot_accuracy_by_culture_top_predicates(df, metric, model_name, min_samples=100)
    fig6.savefig(output_dir / f'6_accuracy_by_culture_top_predicates_{metric}.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig6)
    
    # Export CSV files
    print("\n💾 Exporting CSV files...")
    
    # Culture comparison table
    comparison_df = create_culture_comparison_table(df, metric)
    comparison_df.to_csv(output_dir / f'culture_comparison_{metric}.csv', index=False)
    
    # Detailed breakdown
    detailed_df = create_detailed_breakdown_table(df, metric)
    detailed_df.to_csv(output_dir / f'detailed_breakdown_{metric}.csv', index=False)
    
    # Raw results
    df.to_csv(output_dir / 'all_results.csv', index=False)
    
    print(f"\n✅ Complete! All files saved to: {output_dir}/")
    print(f"\nGenerated files:")
    print(f"   📊 6 visualization PNGs")
    print(f"   📄 3 CSV files")
    print("\n" + "="*70 + "\n")