"""
DLAMA Model Comparison Script
Compares Llama 3B vs Qwen 2.5B performance across Arab-West and Asia-West datasets
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")

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

# Color scheme
MODEL_COLORS = {
    'Llama 3B': '#e74c3c',
    'Qwen 2.5B': '#3498db',
}

CULTURE_COLORS = {
    'Arab': '#e74c3c',
    'Western': '#3498db', 
    'Asia': '#2ecc71',
    'Asian': '#2ecc71',
}

def create_country_accuracy_csv(results: Dict, metric: str = 'overlap', output_path: Path = None):
    """
    Create CSV file with country-level accuracy for each model.
    
    Format:
    country,Llama 3.2-3B,Qwen 2.5-3B
    China,75.5,80.2
    India,65.3,70.1
    ...
    
    Args:
        results: Dictionary with structure {model: {dataset: data}}
        metric: 'overlap' or 'exact_match'
        output_path: Path to save CSV file
    
    Returns:
        DataFrame with country accuracies
    """
    print("\n📝 Creating country-level accuracy CSV...")
    
    # Collect all country data
    country_data = {}
    
    for model in ['Llama 3B', 'Qwen 2.5B']:
        # Map model names to CSV column names
        if model == 'Llama 3B':
            csv_model_name = 'Llama 3.2-3B'
        else:
            csv_model_name = 'Qwen 2.5-3B'
        
        for dataset in ['Arab-West', 'Asia-West']:
            if model in results and dataset in results[model]:
                # Use the extract_summary_metrics function from your existing code
                metrics = extract_summary_metrics(results[model][dataset])
                by_country = metrics.get('by_country', {})
                
                print(f"   Processing {model} - {dataset}:")
                print(f"      Found {len(by_country)} countries")
                
                for country, country_data_item in by_country.items():
                    if country not in country_data:
                        country_data[country] = {}
                    
                    # Get accuracy for this country
                    acc = country_data_item.get(metric, 0) * 100
                    
                    # Store or average if country appears in multiple datasets
                    if csv_model_name in country_data[country]:
                        # Average with existing value
                        existing = country_data[country][csv_model_name]
                        country_data[country][csv_model_name] = (existing + acc) / 2
                        print(f"      {country}: averaging {existing:.1f} and {acc:.1f} = {country_data[country][csv_model_name]:.1f}")
                    else:
                        country_data[country][csv_model_name] = acc
                        print(f"      {country}: {acc:.1f}%")
    
    # Create DataFrame
    rows = []
    for country in sorted(country_data.keys()):
        row = {'country': country}
        row['Llama 3.2-3B'] = country_data[country].get('Llama 3.2-3B', None)
        row['Qwen 2.5-3B'] = country_data[country].get('Qwen 2.5-3B', None)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    if output_path:
        df.to_csv(output_path, index=False, float_format='%.2f')
        print(f"\n✓ Saved country accuracy CSV to: {output_path}")
        print(f"   Total countries: {len(df)}")
        print(f"\nPreview:")
        print(df.head(10).to_string(index=False))
    
    return df


def plot_country_accuracy_comparison(results: Dict, metric: str = 'overlap'):
    """
    Create bar chart comparing country-level accuracy for both models
    """
    # Get country data
    country_data = {}
    
    for model in ['Llama 3B', 'Qwen 2.5B']:
        csv_model_name = 'Llama 3.2-3B' if model == 'Llama 3B' else 'Qwen 2.5-3B'
        
        for dataset in ['Arab-West', 'Asia-West']:
            if model in results and dataset in results[model]:
                metrics = extract_summary_metrics(results[model][dataset])
                by_country = metrics.get('by_country', {})
                
                for country, country_data_item in by_country.items():
                    if country not in country_data:
                        country_data[country] = {}
                    
                    acc = country_data_item.get(metric, 0) * 100
                    
                    if csv_model_name in country_data[country]:
                        existing = country_data[country][csv_model_name]
                        country_data[country][csv_model_name] = (existing + acc) / 2
                    else:
                        country_data[country][csv_model_name] = acc
    
    if not country_data:
        print("⚠️  No country data available for comparison")
        return None
    
    # Create figure
    countries = sorted(country_data.keys())
    llama_accs = [country_data[c].get('Llama 3.2-3B', 0) for c in countries]
    qwen_accs = [country_data[c].get('Qwen 2.5-3B', 0) for c in countries]
    
    # Use MODEL_COLORS from your existing code
    MODEL_COLORS = {
        'Llama 3B': '#e74c3c',
        'Qwen 2.5B': '#3498db',
    }
    
    fig, ax = plt.subplots(figsize=(max(14, len(countries) * 0.6), 8))
    
    x = np.arange(len(countries))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, llama_accs, width, label='Llama 3.2-3B',
                   color=MODEL_COLORS['Llama 3B'], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, qwen_accs, width, label='Qwen 2.5-3B',
                   color=MODEL_COLORS['Qwen 2.5B'], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Country', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Country-Level Accuracy Comparison\nMetric: {metric.capitalize()}',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig


def load_summary_file(file_path: Path) -> Dict:
    """Load a DLAMA summary JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_summary_metrics(data: Dict) -> Dict:
    """Extract key metrics from summary data"""
    # Handle different possible structures
    if 'summary' in data:
        summary = data['summary']
    else:
        summary = data
    
    return {
        'overall_accuracy': summary.get('overall_accuracy', 0) * 100,
        'by_culture': summary.get('by_culture', {}),
        'by_predicate': summary.get('by_predicate', {}),
        'by_country': summary.get('by_country', {}),
        'total_samples': summary.get('total_samples', 0),
    }


def create_overall_comparison_table(results: Dict, metric: str = 'overlap'):
    """
    Create a comparison table showing culture-specific accuracy for each model and dataset
    
    Args:
        results: Dictionary with structure {model: {dataset: data}}
        metric: 'overlap' or 'exact_match'
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    
    for model in ['Llama 3B', 'Qwen 2.5B']:
        for dataset in ['Arab-West', 'Asia-West']:
            if model in results and dataset in results[model]:
                data = results[model][dataset]
                metrics = extract_summary_metrics(data)
                
                # Get culture-specific accuracies
                cultures = sorted(metrics['by_culture'].keys())
                culture_accs = []
                for culture in cultures:
                    acc = metrics['by_culture'][culture].get(metric, 0) * 100
                    culture_accs.append(f"{acc:.1f}%")
                
                table_data.append([
                    model,
                    dataset,
                    ' | '.join([f"{c}: {a}" for c, a in zip(cultures, culture_accs)]),
                    f"{metrics['total_samples']:,}"
                ])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Dataset', 'Accuracy by Culture', 'Samples'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f8f9fa')
            table[(i, j)].set_text_props(fontsize=10)
    
    metric_name = 'Overlap' if metric == 'overlap' else 'Exact Match'
    ax.set_title(
        f'DLAMA Model Comparison: Culture-Specific Performance\n'
        f'Metric: {metric_name}',
        fontsize=15, pad=20, fontweight='bold'
    )
    
    plt.tight_layout()
    return fig





def plot_top_predicates_by_sample(results: Dict, metric: str = 'overlap', top_n: int = 5):
    """
    Create bar chart comparing top N predicates (by sample count) across models and datasets
    """
    # Collect all predicates and their sample counts
    predicate_samples = {}
    predicate_accuracies = {}
    
    for model in results:
        for dataset in results[model]:
            metrics = extract_summary_metrics(results[model][dataset])
            by_predicate = metrics.get('by_predicate', {})
            
            for pred, pred_data in by_predicate.items():
                if pred not in predicate_samples:
                    predicate_samples[pred] = 0
                    predicate_accuracies[pred] = {}
                
                count = pred_data.get('count', 0)
                predicate_samples[pred] += count
                
                # Store accuracy for this model-dataset combination
                key = f"{model}_{dataset}"
                acc = pred_data.get(metric, 0) * 100
                predicate_accuracies[pred][key] = acc
    
    # Get top N predicates by total sample count
    top_predicates = sorted(predicate_samples.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_pred_codes = [p[0] for p in top_predicates]
    
    # Create figure with 2x2 subplots (one for each model-dataset combination)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Top {top_n} Predicates by Sample Count: Model Comparison\n'
        f'Metric: {metric.capitalize()}',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    datasets = ['Arab-West', 'Asia-West']
    models = ['Llama 3B', 'Qwen 2.5B']
    
    for dataset_idx, dataset in enumerate(datasets):
        for model_idx, model in enumerate(models):
            ax = axes[dataset_idx, model_idx]
            
            if model in results and dataset in results[model]:
                # Get accuracies for this model-dataset
                key = f"{model}_{dataset}"
                accuracies = []
                labels = []
                
                for pred in top_pred_codes:
                    acc = predicate_accuracies[pred].get(key, 0)
                    accuracies.append(acc)
                    pred_desc = PREDICATE_DESCRIPTIONS.get(pred, pred)
                    labels.append(f"{pred}\n{pred_desc}")
                
                # Get sample counts for labels
                metrics = extract_summary_metrics(results[model][dataset])
                by_predicate = metrics.get('by_predicate', {})
                counts = [by_predicate.get(p, {}).get('count', 0) for p in top_pred_codes]
                
                color = MODEL_COLORS[model]
                bars = ax.barh(range(len(top_pred_codes)), accuracies, 
                              color=color, alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    width = bar.get_width()
                    ax.text(width + 1, i, f'{width:.1f}%\n(n={count})',
                           va='center', fontsize=9, fontweight='bold')
                
                ax.set_yticks(range(len(top_pred_codes)))
                ax.set_yticklabels(labels, fontsize=9)
                ax.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
                ax.set_title(f'{model} - {dataset}', fontsize=12, fontweight='bold', pad=10)
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                ax.set_xlim(0, 105)
                ax.set_facecolor('#f8f9fa')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_facecolor('#ecf0f1')
    
    plt.tight_layout()
    return fig


def plot_culture_breakdown_comparison(results: Dict, metric: str = 'overlap'):
    """
    Create a detailed comparison showing performance by culture for each model-dataset combination
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Culture-Specific Performance: Llama 3B vs Qwen 2.5B\n'
        f'Metric: {metric.capitalize()}',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    datasets = ['Arab-West', 'Asia-West']
    models = ['Llama 3B', 'Qwen 2.5B']
    
    for dataset_idx, dataset in enumerate(datasets):
        for model_idx, model in enumerate(models):
            ax = axes[dataset_idx, model_idx]
            
            if model in results and dataset in results[model]:
                data = results[model][dataset]
                metrics = extract_summary_metrics(data)
                by_culture = metrics['by_culture']
                
                cultures = sorted(by_culture.keys())
                accuracies = [by_culture[c].get(metric, 0) * 100 for c in cultures]
                counts = [by_culture[c].get('count', 0) for c in cultures]
                
                colors = [CULTURE_COLORS.get(c, '#95a5a6') for c in cultures]
                
                bars = ax.bar(range(len(cultures)), accuracies, color=colors, 
                             alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels
                for i, (bar, count) in enumerate(zip(bars, counts)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%\n(n={count})',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                ax.set_xticks(range(len(cultures)))
                ax.set_xticklabels(cultures, fontsize=11)
                ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
                ax.set_title(f'{model} - {dataset}', fontsize=12, fontweight='bold', pad=10)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_ylim(0, 100)
                ax.set_facecolor('#f8f9fa')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_facecolor('#ecf0f1')
    
    plt.tight_layout()
    return fig


def plot_predicate_performance_heatmap(results: Dict, metric: str = 'overlap', top_n: int = 10):
    """
    Create heatmap showing top N predicates performance across models and datasets
    """
    # Collect all predicates and their average performance
    predicate_scores = {}
    
    for model in results:
        for dataset in results[model]:
            metrics = extract_summary_metrics(results[model][dataset])
            by_predicate = metrics.get('by_predicate', {})
            
            for pred, pred_data in by_predicate.items():
                if pred not in predicate_scores:
                    predicate_scores[pred] = []
                acc = pred_data.get(metric, 0) * 100
                predicate_scores[pred].append(acc)
    
    # Get top N predicates by average performance
    avg_scores = {p: np.mean(scores) for p, scores in predicate_scores.items()}
    top_predicates = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_pred_codes = [p[0] for p in top_predicates]
    
    # Build data matrix
    data_matrix = []
    row_labels = []
    
    for model in ['Llama 3B', 'Qwen 2.5B']:
        for dataset in ['Arab-West', 'Asia-West']:
            if model in results and dataset in results[model]:
                metrics = extract_summary_metrics(results[model][dataset])
                by_predicate = metrics.get('by_predicate', {})
                
                row = []
                for pred in top_pred_codes:
                    if pred in by_predicate:
                        acc = by_predicate[pred].get(metric, 0) * 100
                        row.append(acc)
                    else:
                        row.append(0)
                
                data_matrix.append(row)
                row_labels.append(f'{model}\n{dataset}')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    
    data_array = np.array(data_matrix)
    im = ax.imshow(data_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks(range(len(top_pred_codes)))
    ax.set_yticks(range(len(row_labels)))
    
    col_labels = [f"{p}\n{PREDICATE_DESCRIPTIONS.get(p, p)}" for p in top_pred_codes]
    ax.set_xticklabels(col_labels, fontsize=9, rotation=45, ha='right')
    ax.set_yticklabels(row_labels, fontsize=10)
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(top_pred_codes)):
            text = ax.text(j, i, f'{data_array[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    ax.set_title(
        f'Top {top_n} Predicates Performance Heatmap\n'
        f'Metric: {metric.capitalize()}',
        fontsize=14, fontweight='bold', pad=20
    )
    
    plt.tight_layout()
    return fig


def plot_cultural_bias_comparison(results: Dict, metric: str = 'overlap'):
    """
    Plot showing cultural bias (difference in accuracy between cultures) for each model
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Cultural Bias Analysis: Accuracy Difference Between Cultures\n'
        f'Metric: {metric.capitalize()}',
        fontsize=15, fontweight='bold', y=1.02
    )
    
    datasets = ['Arab-West', 'Asia-West']
    
    for dataset_idx, dataset in enumerate(datasets):
        ax = axes[dataset_idx]
        
        models = []
        biases = []
        
        for model in ['Llama 3B', 'Qwen 2.5B']:
            if model in results and dataset in results[model]:
                metrics = extract_summary_metrics(results[model][dataset])
                by_culture = metrics['by_culture']
                
                cultures = sorted(by_culture.keys())
                if len(cultures) == 2:
                    acc1 = by_culture[cultures[0]].get(metric, 0) * 100
                    acc2 = by_culture[cultures[1]].get(metric, 0) * 100
                    bias = abs(acc2 - acc1)  # Absolute difference
                    
                    models.append(model)
                    biases.append(bias)
        
        if models:
            colors = [MODEL_COLORS[m] for m in models]
            bars = ax.bar(range(len(models)), biases, color=colors, 
                         alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                       f'{height:.2f}%',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, fontsize=12)
            ax.set_ylabel('Accuracy Difference (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'{dataset} Dataset', fontsize=13, fontweight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, max(biases) * 1.3 if biases else 10)
            ax.set_facecolor('#f8f9fa')
            
            # Add horizontal line at 0
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig






    """Main function to generate all comparison visualizations"""
    
    # File paths
    files = {
        'Llama 3B': {
            'Arab-West': Path('results_dlama/dlama_final_results_llama3b/dlama_arab_llama3b_20251203_222321_summary.json'),
            'Asia-West': Path('results_dlama/dlama_final_results_llama3b/dlama_asia_llama3b_20251203_221309_summary.json'),
        },
        'Qwen 2.5B': {
            'Arab-West': Path('results_dlama/dlama_final_results_qwen2_5b/dlama_arab_qwen_2.5b_20251203_152124_summary.json'),
            'Asia-West': Path('results_dlama/dlama_final_results_qwen2_5b/dlama_asia_qwen_2.5b_20251203_212919_summary.json'),
        }
    }
    
    # Output directory
    output_dir = Path('results_dlama/model_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metric = 'overlap'  # or 'exact_match'
    
    print("\n" + "="*70)
    print("🔍 DLAMA Model Comparison: Llama 3B vs Qwen 2.5B")
    print("="*70)
    
    # Load all data
    print("\n📂 Loading summary files...")
    results = {}
    for model, datasets in files.items():
        results[model] = {}
        for dataset, file_path in datasets.items():
            if file_path.exists():
                print(f"   ✓ {model} - {dataset}: {file_path.name}")
                results[model][dataset] = load_summary_file(file_path)

    print("   3. Culture breakdown comparison...")
    fig3 = plot_culture_breakdown_comparison(results, metric)
    fig3.savefig(output_dir / f'3_culture_breakdown_comparison_{metric}.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Predicate performance heatmap
    print("   4. Predicate performance heatmap...")
    fig4 = plot_predicate_performance_heatmap(results, metric, top_n=10)
    fig4.savefig(output_dir / f'4_predicate_heatmap_{metric}.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. Cultural bias comparison
    print("   5. Cultural bias analysis...")
    fig5 = plot_cultural_bias_comparison(results, metric)
    fig5.savefig(output_dir / f'5_cultural_bias_comparison_{metric}.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    # 6. Predicate sample breakdown by culture
    print("   6. Predicate sample breakdown by culture...")
    fig6 = create_predicate_sample_breakdown_table(results)
    fig6.savefig(output_dir / f'6_predicate_sample_breakdown{metric}.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig6)
    
    print(f"\n✅ Complete! All files saved to: {output_dir}/")
    print(f"\nGenerated files:")
    print(f"   📊 6 comparison visualization PNGs")
    print("\n" + "="*70 + "\n")


def create_predicate_sample_breakdown_table(results: Dict):
    """
    Create table showing sample count breakdown by predicate and culture
    Aggregates across all models and datasets
    """
    # Collect sample counts by predicate and culture
    predicate_culture_counts = {}
    
    for model in results:
        for dataset in results[model]:
            metrics = extract_summary_metrics(results[model][dataset])
            by_predicate = metrics.get('by_predicate', {})
            
            # We need to get culture-specific counts from the original data
            # For now, we'll use by_culture data to estimate distribution
            by_culture = metrics.get('by_culture', {})
            cultures = list(by_culture.keys())
            
            for pred, pred_data in by_predicate.items():
                if pred not in predicate_culture_counts:
                    predicate_culture_counts[pred] = {culture: 0 for culture in cultures}
                    predicate_culture_counts[pred]['total'] = 0
                
                total_count = pred_data.get('count', 0)
                predicate_culture_counts[pred]['total'] += total_count
                
                # Assume equal distribution across cultures if not specified
                # (This is a limitation - ideally we'd have per-culture counts in the summary)
                for culture in cultures:
                    if culture not in predicate_culture_counts[pred]:
                        predicate_culture_counts[pred][culture] = 0
                    predicate_culture_counts[pred][culture] += total_count // len(cultures)
    
    # Sort predicates by total count
    sorted_predicates = sorted(predicate_culture_counts.items(), 
                               key=lambda x: x[1].get('total', 0), reverse=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(10, len(sorted_predicates) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    # Build table data
    table_data = []
    
    # Get unique cultures
    all_cultures = set()
    for pred, counts in sorted_predicates:
        all_cultures.update([k for k in counts.keys() if k != 'total'])
    cultures = sorted(all_cultures)
    
    for pred, counts in sorted_predicates:
        pred_desc = PREDICATE_DESCRIPTIONS.get(pred, pred)
        row = [f"{pred}\n{pred_desc}"]
        
        # Add culture counts
        for culture in cultures:
            count = counts.get(culture, 0)
            row.append(f"{count:,}")
        
        # Add total
        row.append(f"{counts['total']:,}")
        
        table_data.append(row)
    
    # Create column labels
    col_labels = ['Predicate'] + cultures + ['Total']
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f8f9fa')
            table[(i, j)].set_text_props(fontsize=9)
    
    ax.set_title(
        'Predicate Sample Count Breakdown by Culture\n'
        'Aggregated across all models and datasets',
        fontsize=14, pad=20, fontweight='bold'
    )
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare DLAMA results across models and datasets")
    parser.add_argument(
        "--llama_arab",
        type=str,
        default="results_dlama/dlama_final_results_llama3b/dlama_arab_llama3b_20251203_222321_summary.json",
        help="Path to Llama 3B Arab-West summary file"
    )
    parser.add_argument(
        "--llama_asia",
        type=str,
        default="results_dlama/dlama_final_results_llama3b/dlama_asia_llama3b_20251203_221309_summary.json",
        help="Path to Llama 3B Asia-West summary file"
    )
    parser.add_argument(
        "--qwen_arab",
        type=str,
        default="results_dlama/dlama_final_results_qwen2_5b/dlama_arab_qwen_2.5b_20251203_152124_summary.json",
        help="Path to Qwen 2.5B Arab-West summary file"
    )
    parser.add_argument(
        "--qwen_asia",
        type=str,
        default="results_dlama/dlama_final_results_qwen2_5b/dlama_asia_qwen_2.5b_20251203_212919_summary.json",
        help="Path to Qwen 2.5B Asia-West summary file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_dlama/model_comparison",
        help="Output directory for comparison visualizations"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=['overlap', 'exact_match'],
        default='overlap',
        help="Metric to visualize"
    )
    
    args = parser.parse_args()
    
    # Override file paths with command-line arguments
    files = {
        'Llama 3B': {
            'Arab-West': Path(args.llama_arab),
            'Asia-West': Path(args.llama_asia),
        },
        'Qwen 2.5B': {
            'Arab-West': Path(args.qwen_arab),
            'Asia-West': Path(args.qwen_asia),
        }
    }
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric = args.metric
    
    print("\n" + "="*70)
    print("🔍 DLAMA Model Comparison: Llama 3B vs Qwen 2.5B")
    print("="*70)
    
    # Load all data
    print("\n📂 Loading summary files...")
    results = {}
    for model, datasets in files.items():
        results[model] = {}
        for dataset, file_path in datasets.items():
            if file_path.exists():
                print(f"   ✓ {model} - {dataset}: {file_path.name}")
                results[model][dataset] = load_summary_file(file_path)
            else:
                print(f"   ✗ {model} - {dataset}: File not found - {file_path}")
    
    # Check if we have any data
    if not any(results.values()):
        print("\n❌ No valid data files found. Please check the file paths.")
        exit(1)
    
    # Generate visualizations
    print("\n📊 Generating comparison visualizations...")
    
    # 1. Overall comparison table
    print("   1. Overall comparison table...")
    fig1 = create_overall_comparison_table(results, metric)
    fig1.savefig(output_dir / f'1_overall_comparison_table_{metric}.png', 
                 dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Top 5 predicates comparison by sample count
    print("   2. Top 5 predicates by sample count...")
    fig2 = plot_top_predicates_by_sample(results, metric, top_n=5)
    fig2.savefig(output_dir / f'2_top_predicates_by_sample_{metric}.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Culture breakdown comparison
    print("   3. Culture breakdown comparison...")
    fig3 = plot_culture_breakdown_comparison(results, metric)
    fig3.savefig(output_dir / f'3_culture_breakdown_comparison_{metric}.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Predicate performance heatmap
    print("   4. Predicate performance heatmap...")
    fig4 = plot_predicate_performance_heatmap(results, metric, top_n=10)
    fig4.savefig(output_dir / f'4_predicate_heatmap_{metric}.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. Cultural bias comparison
    print("   5. Cultural bias analysis...")
    fig5 = plot_cultural_bias_comparison(results, metric)
    fig5.savefig(output_dir / f'5_cultural_bias_comparison_{metric}.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    # 6. Predicate sample breakdown by culture
    print("   6. Predicate sample breakdown by culture...")
    fig6 = create_predicate_sample_breakdown_table(results)
    fig6.savefig(output_dir / f'6_predicate_sample_breakdown{metric}.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig6)

    # 7. Create country-level accuracy CSV
    print("   7. Creating country accuracy CSV...")
    df_country = create_country_accuracy_csv(results, metric, 
                                            output_path=output_dir / 'dlama_accuracy_by_country.csv')

    # 8. Country accuracy comparison chart
    print("   8. Country accuracy comparison chart...")
    fig8 = plot_country_accuracy_comparison(results, metric)
    if fig8:
        fig8.savefig(output_dir / f'7_country_accuracy_comparison_{metric}.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig8)
    
    print(f"\n✅ Complete! All files saved to: {output_dir}/")
    print(f"\nGenerated files:")
    print(f"   📊 6 comparison visualization PNGs")
    print("\n" + "="*70 + "\n")