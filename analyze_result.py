import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

def load_summary_results(results_dir):
    """Load summary JSON files and extract aggregate metrics"""
    results = []
    
    for json_file in Path(results_dir).rglob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Check if this is a summary file
            if 'aggregate_metrics' not in data:
                continue
            
            # Extract model name from path
            path_str = str(json_file).lower()
            if "llama" in path_str:
                model = "Llama 3.2-3B"
            elif "qwen" in path_str:
                model = "Qwen 2.5-3B"
            else:
                model = "Unknown"
            
            # TRY 1: Get culture from config (most reliable)
            culture = None
            if 'config' in data:
                try:
                    culture = data['config']['task']['config']['culture']
                except (KeyError, TypeError):
                    pass
            
            # TRY 2: Get from experiment_name
            if not culture and 'experiment_name' in data:
                exp_name = data['experiment_name']
                # Look for 2-letter uppercase code
                import re
                match = re.search(r'\b([A-Z]{2})\b', exp_name)
                if match:
                    culture = match.group(1)
            
            # TRY 3: Get from filename  
            if not culture:
                filename = json_file.stem
                parts = filename.split('_')
                for part in parts:
                    if len(part) == 2 and part.isupper():
                        culture = part
                        break
                
                if not culture:
                    culture = parts[0]
            
            # Clean up culture name
            if culture:
                culture = culture.upper().strip()
                # Fix common issues
                if "SOUTH KOREA" in culture or culture == "SOUTH":
                    culture = "KR"
                elif "CHINA" in culture:
                    culture = "CN"
                elif "IRAN" in culture:
                    culture = "IR"
                elif culture == "UK":
                    culture = "UK"
                elif culture == "US":
                    culture = "US"
                elif "SUNDANESE" in culture:
                    culture = "ID"  # Indonesia
                elif "ASSAMESE" in culture:
                    culture = "AS"  # Assamese
            
            # Extract metrics
            metrics = data['aggregate_metrics']
            num_samples = data.get('total_samples', 0)
            
            results.append({
                'model': model,
                'culture': culture if culture else 'UNKNOWN',
                'accuracy': metrics['accuracy']['mean'] * 100,
                'valid_format': metrics.get('has_valid_format', {}).get('mean', 1.0) * 100,
                'num_samples': num_samples,
                'accuracy_min': metrics['accuracy']['min'] * 100,
                'accuracy_max': metrics['accuracy']['max'] * 100,
                'file': json_file.name
            })
            
            print(f"✓ {model:20s} | {culture:6s} | {num_samples:6,} samples | {metrics['accuracy']['mean']*100:5.1f}%")
            
        except Exception as e:
            print(f"⚠️  Skipped {json_file.name}: {e}")
    
    df = pd.DataFrame(results)
    
    # IMPORTANT: Keep only the entry with most samples for each model+culture combo
    if not df.empty:
        print(f"\n📊 Found {len(df)} result files")
        
        # Group by model and culture, keep the one with most samples
        df = df.sort_values('num_samples', ascending=False).groupby(['model', 'culture']).first().reset_index()
        
        print(f"📊 After removing duplicates: {len(df)} unique model-culture combinations")
    
    return df

def create_comparison_table(df):
    """Create formatted comparison tables"""
    # Pivot table for accuracy
    accuracy_table = df.pivot(index='culture', columns='model', values='accuracy')
    
    # Pivot table for valid format
    format_table = df.pivot(index='culture', columns='model', values='valid_format')
    
    # Pivot table for sample counts
    samples_table = df.pivot(index='culture', columns='model', values='num_samples')
    
    return accuracy_table, format_table, samples_table

def plot_comparison(df):
    """Create comprehensive visualization comparing models across cultures"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('BLEnD Cultural Knowledge Evaluation: Llama vs Qwen', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Accuracy comparison bar chart
    ax1 = fig.add_subplot(gs[0, :])
    accuracy_pivot = df.pivot(index='culture', columns='model', values='accuracy')
    x = np.arange(len(accuracy_pivot))
    width = 0.35
    
    models = accuracy_pivot.columns
    for i, model in enumerate(models):
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax1.bar(x + offset, accuracy_pivot[model], width, 
                       label=model, alpha=0.8, color=colors[i])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=8)
    
    ax1.set_title('Accuracy by Culture', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Culture', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(accuracy_pivot.index, rotation=0, ha='center')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=25, color='red', linestyle='--', linewidth=2, 
                label='Random Baseline (25%)', alpha=0.6)
    ax1.set_ylim(0, 100)
    
    # 2. Overall model comparison
    ax2 = fig.add_subplot(gs[1, 0])
    avg_by_model = df.groupby('model')['accuracy'].mean().sort_values(ascending=False)
    bars = ax2.barh(range(len(avg_by_model)), avg_by_model.values, 
                    color=colors[:len(avg_by_model)], alpha=0.8)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, avg_by_model.values)):
        ax2.text(val + 1, i, f'{val:.2f}%', 
                va='center', fontsize=11, fontweight='bold')
    
    ax2.set_yticks(range(len(avg_by_model)))
    ax2.set_yticklabels(avg_by_model.index, fontsize=11)
    ax2.set_xlabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Overall Performance (Avg Across Cultures)', 
                  fontsize=14, fontweight='bold', pad=10)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 100)
    
    # 3. Valid format comparison
    ax3 = fig.add_subplot(gs[1, 1])
    format_by_model = df.groupby('model')['valid_format'].mean()
    bars = ax3.barh(range(len(format_by_model)), format_by_model.values,
                    color=['#2ecc71', '#27ae60'][:len(format_by_model)], alpha=0.8)
    
    for i, (bar, val) in enumerate(zip(bars, format_by_model.values)):
        ax3.text(val - 5, i, f'{val:.1f}%', 
                va='center', ha='right', fontsize=11, 
                fontweight='bold', color='white')
    
    ax3.set_yticks(range(len(format_by_model)))
    ax3.set_yticklabels(format_by_model.index, fontsize=11)
    ax3.set_xlabel('Valid Format (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Response Format Compliance', 
                  fontsize=14, fontweight='bold', pad=10)
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    ax3.set_xlim(0, 105)
    ax3.axvline(x=100, color='green', linestyle='--', linewidth=2, alpha=0.5)
    
    # 4. Sample size per culture
    ax4 = fig.add_subplot(gs[2, 0])
    samples_by_culture = df.groupby('culture')['num_samples'].first().sort_values(ascending=True)
    bars = ax4.barh(range(len(samples_by_culture)), samples_by_culture.values,
                    color='#9b59b6', alpha=0.7)
    
    for i, (bar, val) in enumerate(zip(bars, samples_by_culture.values)):
        ax4.text(val + 100, i, f'{val:,}', 
                va='center', fontsize=9)
    
    ax4.set_yticks(range(len(samples_by_culture)))
    ax4.set_yticklabels(samples_by_culture.index, fontsize=10)
    ax4.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    ax4.set_title('Dataset Size by Culture', 
                  fontsize=14, fontweight='bold', pad=10)
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 5. Head-to-head comparison (if we have multiple models)
    ax5 = fig.add_subplot(gs[2, 1])
    if len(df['model'].unique()) >= 2:
        # Create scatter plot comparing models
        model1, model2 = sorted(df['model'].unique())[:2]
        
        df1 = df[df['model'] == model1].set_index('culture')['accuracy']
        df2 = df[df['model'] == model2].set_index('culture')['accuracy']
        
        # Get common cultures
        common = df1.index.intersection(df2.index)
        
        if len(common) > 0:
            ax5.scatter(df1[common], df2[common], s=200, alpha=0.6, 
                       color='#e74c3c', edgecolors='black', linewidth=1.5)
            
            # Add culture labels
            for culture in common:
                ax5.annotate(culture, (df1[culture], df2[culture]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold')
            
            # Add diagonal line (equal performance)
            max_val = max(df1[common].max(), df2[common].max())
            min_val = min(df1[common].min(), df2[common].min())
            ax5.plot([min_val, max_val], [min_val, max_val], 
                    'k--', alpha=0.5, linewidth=2, label='Equal Performance')
            
            ax5.set_xlabel(f'{model1} Accuracy (%)', fontsize=12, fontweight='bold')
            ax5.set_ylabel(f'{model2} Accuracy (%)', fontsize=12, fontweight='bold')
            ax5.set_title('Head-to-Head Comparison', 
                         fontsize=14, fontweight='bold', pad=10)
            ax5.grid(alpha=0.3, linestyle='--')
            ax5.legend(fontsize=10)
            ax5.set_aspect('equal')
    else:
        ax5.text(0.5, 0.5, 'Need 2+ models for comparison', 
                ha='center', va='center', fontsize=12)
        ax5.axis('off')
    
    return fig

def create_detailed_table_plot(accuracy_table, format_table, samples_table):
    """Create detailed table visualization"""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('Detailed Results Tables', fontsize=16, fontweight='bold')
    
    tables_data = [
        (accuracy_table, 'Accuracy (%)', '#3498db', axes[0]),
        (format_table, 'Valid Format (%)', '#2ecc71', axes[1]),
        (samples_table, 'Sample Count', '#9b59b6', axes[2])
    ]
    
    for table_df, title, color, ax in tables_data:
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        data = []
        for culture in table_df.index:
            row = [culture]
            for model in table_df.columns:
                val = table_df.loc[culture, model]
                if pd.isna(val):
                    row.append("N/A")
                elif 'Sample' in title:
                    row.append(f"{int(val):,}")
                else:
                    row.append(f"{val:.1f}%")
            data.append(row)
        
        # Create table
        table = ax.table(cellText=data,
                        colLabels=['Culture'] + list(table_df.columns),
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(table_df.columns) + 1):
            table[(0, i)].set_facecolor(color)
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(data) + 1):
            for j in range(len(table_df.columns) + 1):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f8f9fa')
        
        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
    
    plt.tight_layout()
    return fig

def print_summary_report(df):
    """Print a formatted text report"""
    print("\n" + "="*80)
    print(" "*20 + "BLEND EVALUATION SUMMARY REPORT")
    print("="*80)
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        print(f"\n📊 {model}")
        print("-"*80)
        print(f"  Cultures Evaluated: {len(model_df)}")
        print(f"  Total Samples: {model_df['num_samples'].sum():,}")
        print(f"  Average Accuracy: {model_df['accuracy'].mean():.2f}%")
        print(f"  Best Culture: {model_df.loc[model_df['accuracy'].idxmax(), 'culture']} "
              f"({model_df['accuracy'].max():.2f}%)")
        print(f"  Worst Culture: {model_df.loc[model_df['accuracy'].idxmin(), 'culture']} "
              f"({model_df['accuracy'].min():.2f}%)")
        print(f"  Std Dev: {model_df['accuracy'].std():.2f}%")
        print(f"  Valid Format: {model_df['valid_format'].mean():.2f}%")
    
    print("\n" + "="*80)
    print("CULTURE-WISE BREAKDOWN")
    print("="*80)
    
    accuracy_pivot = df.pivot(index='culture', columns='model', values='accuracy')
    for culture in accuracy_pivot.index:
        print(f"\n{culture}:")
        for model in accuracy_pivot.columns:
            val = accuracy_pivot.loc[culture, model]
            if not pd.isna(val):
                print(f"  {model:20s}: {val:6.2f}%")
    
    print("\n" + "="*80 + "\n")

def main():
    """Main function to generate all visualizations"""
    
    # Load results
    results_dir = "results"
    
    print("\n🔍 Scanning for result files...")
    df = load_summary_results(results_dir)
    
    if df.empty:
        print("\n❌ No summary JSON files found!")
        print("   Looking for files with 'aggregate_metrics' structure")
        print(f"   in: {results_dir}")
        return
    
    print(f"\n✓ Successfully loaded {len(df)} results")
    print(f"  Models: {', '.join(df['model'].unique())}")
    print(f"  Cultures: {', '.join(sorted(df['culture'].unique()))}")
    
    # Create tables
    accuracy_table, format_table, samples_table = create_comparison_table(df)
    
    # Print summary report
    print_summary_report(df)
    
    # Print tables
    print("\n" + "="*80)
    print("ACCURACY TABLE")
    print("="*80)
    print(accuracy_table.round(2).to_string())
    
    print("\n" + "="*80)
    print("VALID FORMAT TABLE")
    print("="*80)
    print(format_table.round(2).to_string())
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # Main comparison plot
    fig1 = plot_comparison(df)
    fig1.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/model_comparison.png")
    
    # Detailed table plot
    fig2 = create_detailed_table_plot(accuracy_table, format_table, samples_table)
    fig2.savefig('results/detailed_tables.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/detailed_tables.png")
    
    # Export to CSV
    accuracy_table.to_csv('results/accuracy_table.csv')
    format_table.to_csv('results/format_table.csv')
    df.to_csv('results/all_results_summary.csv', index=False)
    print("  ✓ Saved CSV files")
    
    # Show plots
    plt.show()
    
    print("\n✅ Analysis complete! Check the results/ directory for outputs.\n")

if __name__ == "__main__":
    main()