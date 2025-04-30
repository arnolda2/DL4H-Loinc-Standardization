#!/usr/bin/env python
"""
Ablation Study for LOINC Standardization Model

This script performs ablation studies to understand the contribution of different components:
1. Evaluating the impact of the two-stage fine-tuning approach
2. Comparing different mining strategies (hard vs semi-hard negative mining)
3. Testing the effect of data augmentation
4. Measuring the influence of model size and architecture
"""
import os
import sys
import pandas as pd
import numpy as np
import argparse
import subprocess
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_component_evaluation(component, value, test_file, loinc_file, checkpoint_dir, output_dir, fold=0, expanded_pool=False, augmented_test=False):
    """
    Run evaluation with specific component configuration
    
    Args:
        component: Component name being tested
        value: Component value being tested
        test_file: Path to test data
        loinc_file: Path to LOINC data
        checkpoint_dir: Directory with model checkpoints
        output_dir: Directory to save results
        fold: Model fold to evaluate
        expanded_pool: Whether to use expanded target pool
        augmented_test: Whether to use augmented test data
    
    Returns:
        results_file: Path to the results file
    """
    # Create a unique identifier for this ablation test
    test_id = f"{component}_{value}"
    
    # Get proper checkpoint path based on component and value
    if component == 'fine_tuning_stages':
        if value == 'stage2_only':
            checkpoint_path = os.path.join(checkpoint_dir, f"stage2_only_fold{fold+1}_model.weights.h5")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"stage2_fold{fold+1}_model.weights.h5")
    elif component == 'mining_strategy':
        checkpoint_path = os.path.join(checkpoint_dir, f"stage2_{value}_fold{fold+1}_model.weights.h5")
    elif component == 'data_augmentation':
        if value == 'without_augmentation':
            checkpoint_path = os.path.join(checkpoint_dir, f"stage2_no_aug_fold{fold+1}_model.weights.h5")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"stage2_fold{fold+1}_model.weights.h5")
    else:  # For model_size or other components
        checkpoint_path = os.path.join(checkpoint_dir, f"{value}_fold{fold+1}_model.weights.h5")
    
    # Ensure the checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"WARNING: Checkpoint not found at {checkpoint_path}, using default")
        checkpoint_path = os.path.join(checkpoint_dir, f"stage2_fold{fold+1}_model.weights.h5")
    
    # Setup command based on the parameters
    cmd = [
        'python', 'models/evaluation.py',
        '--test_file', test_file,
        '--loinc_file', loinc_file,
        '--checkpoint_dir', checkpoint_dir,  # Use the directory, not specific path
        '--output_dir', output_dir,
        '--fold', str(fold)
    ]
    
    # Add expanded pool flag if needed
    if expanded_pool:
        cmd.append('--expanded_pool')
    
    # Add augmented test flag if needed
    if augmented_test:
        cmd.append('--augmented_test')
    
    # Add ablation identifier
    cmd.extend(['--ablation_id', test_id])
    
    print(f"\nRunning {component} ablation test with {value}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        
        # Construct expected output file name based on ablation_id and other parameters
        # Format is fold{fold}_[augmented_]expanded_ablation_{test_id}_results.csv
        file_prefix = f"fold{fold}"
        if augmented_test:
            file_prefix += "_augmented"
        if expanded_pool:
            file_prefix += "_expanded"
        file_prefix += f"_ablation_{test_id}"
        
        results_file = os.path.join(output_dir, f"{file_prefix}_results.csv")
        
        if not os.path.exists(results_file):
            print(f"WARNING: Results file not found at {results_file}")
            return None
        
        return results_file
    
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        return None

def test_fine_tuning_stages(test_file, loinc_file, checkpoint_dir, output_dir, fold=0, expanded_pool=False, augmented_test=False):
    """
    Test the impact of the two-stage fine-tuning approach
    
    Args:
        test_file: Path to test data
        loinc_file: Path to LOINC data
        checkpoint_dir: Directory with model checkpoints
        output_dir: Directory to save results
        fold: Model fold to evaluate
        expanded_pool: Whether to use expanded target pool
        augmented_test: Whether to use augmented test data
    
    Returns:
        results: Dictionary with results for both approaches
    """
    print("\n=== ABLATION STUDY: FINE-TUNING STAGES ===")
    
    results = {}
    
    # Test the two-stage fine-tuning approach (stage1 + stage2)
    stage1_stage2_file = run_component_evaluation(
        'fine_tuning_stages', 'stage1_stage2', 
        test_file, loinc_file, checkpoint_dir, output_dir, 
        fold, expanded_pool, augmented_test
    )
    
    if stage1_stage2_file:
        results['stage1_stage2'] = pd.read_csv(stage1_stage2_file)
    
    # Test stage2-only approach
    stage2_only_file = run_component_evaluation(
        'fine_tuning_stages', 'stage2_only', 
        test_file, loinc_file, checkpoint_dir, output_dir, 
        fold, expanded_pool, augmented_test
    )
    
    if stage2_only_file:
        results['stage2_only'] = pd.read_csv(stage2_only_file)
    
    # Compare results
    if results:
        print("\nFine-Tuning Stages Comparison:")
        
        for approach, result_df in results.items():
            if 'top1_accuracy' in result_df.columns:
                print(f"{approach}:")
                print(f"  Top-1 Accuracy: {result_df['top1_accuracy'].values[0]:.4f}")
                print(f"  Top-3 Accuracy: {result_df['top3_accuracy'].values[0]:.4f}")
                print(f"  Top-5 Accuracy: {result_df['top5_accuracy'].values[0]:.4f}")
        
        # Create visualization
        if all(k in results for k in ['stage1_stage2', 'stage2_only']):
            metrics = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
            values = {
                'stage1_stage2': [results['stage1_stage2'][m].values[0] for m in metrics],
                'stage2_only': [results['stage2_only'][m].values[0] for m in metrics]
            }
            
            plt.figure(figsize=(10, 6))
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, values['stage1_stage2'], width, label='Two-Stage Fine-Tuning')
            plt.bar(x + width/2, values['stage2_only'], width, label='Stage 2 Only')
            
            plt.ylabel('Accuracy')
            plt.title('Impact of First-Stage Fine-Tuning')
            plt.xticks(x, ['Top-1', 'Top-3', 'Top-5'])
            plt.legend()
            
            # Save figure
            plt.tight_layout()
            fig_path = os.path.join(output_dir, 'fine_tuning_stages_comparison.png')
            plt.savefig(fig_path)
            print(f"Saved fine-tuning stages comparison to {fig_path}")
    
    return results

def test_mining_strategies(test_file, loinc_file, checkpoint_dir, output_dir, fold=0, expanded_pool=False, augmented_test=False):
    """
    Test the impact of different mining strategies
    
    Args:
        test_file: Path to test data
        loinc_file: Path to LOINC data
        checkpoint_dir: Directory with model checkpoints
        output_dir: Directory to save results
        fold: Model fold to evaluate
        expanded_pool: Whether to use expanded target pool
        augmented_test: Whether to use augmented test data
    
    Returns:
        results: Dictionary with results for different mining strategies
    """
    print("\n=== ABLATION STUDY: MINING STRATEGIES ===")
    
    results = {}
    
    # Test hard negative mining
    hard_negative_file = run_component_evaluation(
        'mining_strategy', 'hard_negative', 
        test_file, loinc_file, checkpoint_dir, output_dir, 
        fold, expanded_pool, augmented_test
    )
    
    if hard_negative_file:
        results['hard_negative'] = pd.read_csv(hard_negative_file)
    
    # Test semi-hard negative mining
    semi_hard_file = run_component_evaluation(
        'mining_strategy', 'semi_hard', 
        test_file, loinc_file, checkpoint_dir, output_dir, 
        fold, expanded_pool, augmented_test
    )
    
    if semi_hard_file:
        results['semi_hard'] = pd.read_csv(semi_hard_file)
    
    # Compare results
    if results:
        print("\nMining Strategies Comparison:")
        
        for strategy, result_df in results.items():
            if 'top1_accuracy' in result_df.columns:
                print(f"{strategy}:")
                print(f"  Top-1 Accuracy: {result_df['top1_accuracy'].values[0]:.4f}")
                print(f"  Top-3 Accuracy: {result_df['top3_accuracy'].values[0]:.4f}")
                print(f"  Top-5 Accuracy: {result_df['top5_accuracy'].values[0]:.4f}")
        
        # Create visualization
        if all(k in results for k in ['hard_negative', 'semi_hard']):
            metrics = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
            values = {
                'hard_negative': [results['hard_negative'][m].values[0] for m in metrics],
                'semi_hard': [results['semi_hard'][m].values[0] for m in metrics]
            }
            
            plt.figure(figsize=(10, 6))
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, values['hard_negative'], width, label='Hard Negative Mining')
            plt.bar(x + width/2, values['semi_hard'], width, label='Semi-Hard Negative Mining')
            
            plt.ylabel('Accuracy')
            plt.title('Impact of Mining Strategy')
            plt.xticks(x, ['Top-1', 'Top-3', 'Top-5'])
            plt.legend()
            
            # Save figure
            plt.tight_layout()
            fig_path = os.path.join(output_dir, 'mining_strategies_comparison.png')
            plt.savefig(fig_path)
            print(f"Saved mining strategies comparison to {fig_path}")
    
    return results

def test_data_augmentation(test_file, loinc_file, checkpoint_dir, output_dir, fold=0, expanded_pool=False, augmented_test=False):
    """
    Test the impact of data augmentation
    
    Args:
        test_file: Path to test data
        loinc_file: Path to LOINC data
        checkpoint_dir: Directory with model checkpoints
        output_dir: Directory to save results
        fold: Model fold to evaluate
        expanded_pool: Whether to use expanded target pool
        augmented_test: Whether to use augmented test data
    
    Returns:
        results: Dictionary with results with and without data augmentation
    """
    print("\n=== ABLATION STUDY: DATA AUGMENTATION ===")
    
    results = {}
    
    # Test with data augmentation (default model)
    with_augmentation_file = run_component_evaluation(
        'data_augmentation', 'with_augmentation', 
        test_file, loinc_file, checkpoint_dir, output_dir, 
        fold, expanded_pool, augmented_test
    )
    
    if with_augmentation_file:
        results['with_augmentation'] = pd.read_csv(with_augmentation_file)
    
    # Test without data augmentation
    without_augmentation_file = run_component_evaluation(
        'data_augmentation', 'without_augmentation', 
        test_file, loinc_file, checkpoint_dir, output_dir, 
        fold, expanded_pool, augmented_test
    )
    
    if without_augmentation_file:
        results['without_augmentation'] = pd.read_csv(without_augmentation_file)
    
    # Compare results
    if results:
        print("\nData Augmentation Comparison:")
        
        for aug_setting, result_df in results.items():
            if 'top1_accuracy' in result_df.columns:
                print(f"{aug_setting}:")
                print(f"  Top-1 Accuracy: {result_df['top1_accuracy'].values[0]:.4f}")
                print(f"  Top-3 Accuracy: {result_df['top3_accuracy'].values[0]:.4f}")
                print(f"  Top-5 Accuracy: {result_df['top5_accuracy'].values[0]:.4f}")
        
        # Create visualization
        if all(k in results for k in ['with_augmentation', 'without_augmentation']):
            metrics = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
            values = {
                'with_augmentation': [results['with_augmentation'][m].values[0] for m in metrics],
                'without_augmentation': [results['without_augmentation'][m].values[0] for m in metrics]
            }
            
            plt.figure(figsize=(10, 6))
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, values['with_augmentation'], width, label='With Augmentation')
            plt.bar(x + width/2, values['without_augmentation'], width, label='Without Augmentation')
            
            plt.ylabel('Accuracy')
            plt.title('Impact of Data Augmentation')
            plt.xticks(x, ['Top-1', 'Top-3', 'Top-5'])
            plt.legend()
            
            # Save figure
            plt.tight_layout()
            fig_path = os.path.join(output_dir, 'data_augmentation_comparison.png')
            plt.savefig(fig_path)
            print(f"Saved data augmentation comparison to {fig_path}")
    
    return results

def test_model_size(test_file, loinc_file, checkpoint_dir, output_dir, fold=0, expanded_pool=False, augmented_test=False):
    """
    Test the impact of model size (if different size models are available)
    
    Args:
        test_file: Path to test data
        loinc_file: Path to LOINC data
        checkpoint_dir: Directory with model checkpoints
        output_dir: Directory to save results
        fold: Model fold to evaluate
        expanded_pool: Whether to use expanded target pool
        augmented_test: Whether to use augmented test data
    
    Returns:
        results: Dictionary with results for different model sizes
    """
    print("\n=== ABLATION STUDY: MODEL SIZE ===")
    
    results = {}
    
    # List of model sizes to test (if available)
    model_sizes = ['st5_base', 'st5_large']
    
    for size in model_sizes:
        result_file = run_component_evaluation(
            'model_size', size, 
            test_file, loinc_file, checkpoint_dir, output_dir, 
            fold, expanded_pool, augmented_test
        )
        
        if result_file:
            results[size] = pd.read_csv(result_file)
    
    # Compare results
    if results:
        print("\nModel Size Comparison:")
        
        for size, result_df in results.items():
            if 'top1_accuracy' in result_df.columns:
                print(f"{size}:")
                print(f"  Top-1 Accuracy: {result_df['top1_accuracy'].values[0]:.4f}")
                print(f"  Top-3 Accuracy: {result_df['top3_accuracy'].values[0]:.4f}")
                print(f"  Top-5 Accuracy: {result_df['top5_accuracy'].values[0]:.4f}")
        
        # Create visualization if we have at least 2 different model sizes
        if len(results) >= 2:
            metrics = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
            values = {
                size: [results[size][m].values[0] for m in metrics] 
                for size in results.keys()
            }
            
            plt.figure(figsize=(10, 6))
            x = np.arange(len(metrics))
            width = 0.35 / len(results)
            
            for i, (size, vals) in enumerate(values.items()):
                plt.bar(x + (i - len(results)/2 + 0.5) * width, vals, width, label=size)
            
            plt.ylabel('Accuracy')
            plt.title('Impact of Model Size')
            plt.xticks(x, ['Top-1', 'Top-3', 'Top-5'])
            plt.legend()
            
            # Save figure
            plt.tight_layout()
            fig_path = os.path.join(output_dir, 'model_size_comparison.png')
            plt.savefig(fig_path)
            print(f"Saved model size comparison to {fig_path}")
    
    return results

def save_ablation_summary(all_results, output_dir):
    """
    Save a summary of all ablation study results
    
    Args:
        all_results: Dictionary with results from all ablation studies
        output_dir: Directory to save summary
    """
    summary_file = os.path.join(output_dir, 'ablation_study_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("=== LOINC STANDARDIZATION MODEL ABLATION STUDY SUMMARY ===\n\n")
        
        for component, results in all_results.items():
            f.write(f"\n{component.upper()} ABLATION STUDY\n")
            f.write(f"{'-'*50}\n")
            
            if not results:
                f.write("No results available\n")
                continue
            
            # Format the results as a table
            metrics = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
            metric_labels = ['Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy']
            
            # Get all values for this component
            values = {}
            for setting, result_df in results.items():
                if all(m in result_df.columns for m in metrics):
                    values[setting] = [result_df[m].values[0] for m in metrics]
            
            # Write table header
            f.write(f"{'Setting':<20}")
            for label in metric_labels:
                f.write(f"{label:<20}")
            f.write("\n")
            
            f.write(f"{'-'*80}\n")
            
            # Write table rows
            for setting, vals in values.items():
                f.write(f"{setting:<20}")
                for val in vals:
                    f.write(f"{val*100:.2f}%{' ':<14}")
                f.write("\n")
            
            f.write("\n")
            
            # Calculate relative improvements
            if len(values) > 1:
                f.write("Relative Improvements:\n")
                
                # Find the baseline setting for each component
                baseline = {
                    'fine_tuning_stages': 'stage2_only',
                    'mining_strategy': 'hard_negative',  # Assuming this as baseline
                    'data_augmentation': 'without_augmentation',
                    'model_size': 'st5_base'
                }.get(component, next(iter(values.keys())))
                
                if baseline in values:
                    for setting, vals in values.items():
                        if setting != baseline:
                            f.write(f"{setting} vs {baseline}:\n")
                            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                                improvement = vals[i] - values[baseline][i]
                                f.write(f"  {label}: {improvement*100:+.2f}% ({improvement/values[baseline][i]*100:+.2f}% relative)\n")
                    f.write("\n")
            
            f.write("\n")
    
    print(f"Saved ablation study summary to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Ablation studies for LOINC standardization model')
    parser.add_argument('--test_file', type=str, default='output/mimic_pairs_processed.csv', 
                        help='Path to test data CSV')
    parser.add_argument('--augmented_test_file', type=str, default='output/mimic_pairs_augmented.csv',
                       help='Path to augmented test data CSV')
    parser.add_argument('--loinc_file', type=str, default='output/loinc_full_processed.csv', 
                        help='Path to LOINC data CSV')
    parser.add_argument('--expanded_pool', type=str, default='output/expanded_target_pool.csv',
                        help='Path to expanded target pool CSV')
    parser.add_argument('--checkpoint_dir', type=str, default='models/checkpoints', 
                        help='Directory containing model checkpoints')
    parser.add_argument('--output_dir', type=str, default='results/ablation_study', 
                        help='Directory to save ablation study results')
    parser.add_argument('--fold', type=int, default=0, 
                        help='Fold to analyze')
    parser.add_argument('--skip_augmented_test', action='store_true',
                        help='Skip evaluation on augmented test data')
    parser.add_argument('--components', type=str, nargs='+', 
                        default=['fine_tuning_stages', 'mining_strategy', 'data_augmentation', 'model_size'],
                        help='Components to include in ablation study')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(args.test_file):
        print(f"Test file not found: {args.test_file}")
        return
    
    # Determine which target pool to use
    use_expanded_pool = os.path.exists(args.expanded_pool)
    if not use_expanded_pool:
        print(f"Expanded target pool not found: {args.expanded_pool}")
        print("Will only evaluate on standard target pool.")
    
    # Determine which test data to use
    use_augmented_test = not args.skip_augmented_test and os.path.exists(args.augmented_test_file)
    if not use_augmented_test and not args.skip_augmented_test:
        print(f"Augmented test file not found: {args.augmented_test_file}")
        print("Will only evaluate on standard test data.")
    
    # Store all results
    all_results = {}
    
    # Run fine-tuning stages ablation (if requested)
    if 'fine_tuning_stages' in args.components:
        results = test_fine_tuning_stages(
            args.test_file, args.loinc_file, args.checkpoint_dir, args.output_dir, 
            args.fold, use_expanded_pool, False  # No augmented test for this
        )
        all_results['fine_tuning_stages'] = results
    
    # Run mining strategies ablation (if requested)
    if 'mining_strategy' in args.components:
        results = test_mining_strategies(
            args.test_file, args.loinc_file, args.checkpoint_dir, args.output_dir, 
            args.fold, use_expanded_pool, False  # No augmented test for this
        )
        all_results['mining_strategy'] = results
    
    # Run data augmentation ablation (if requested)
    if 'data_augmentation' in args.components:
        # First with standard test data
        results_standard = test_data_augmentation(
            args.test_file, args.loinc_file, args.checkpoint_dir, args.output_dir, 
            args.fold, use_expanded_pool, False
        )
        all_results['data_augmentation_standard'] = results_standard
        
        # Then with augmented test data (if available)
        if use_augmented_test:
            results_augmented = test_data_augmentation(
                args.augmented_test_file, args.loinc_file, args.checkpoint_dir, args.output_dir, 
                args.fold, use_expanded_pool, True
            )
            all_results['data_augmentation_augmented'] = results_augmented
    
    # Run model size ablation (if requested)
    if 'model_size' in args.components:
        results = test_model_size(
            args.test_file, args.loinc_file, args.checkpoint_dir, args.output_dir, 
            args.fold, use_expanded_pool, False
        )
        all_results['model_size'] = results
    
    # Save summary of all ablation studies
    save_ablation_summary(all_results, args.output_dir)
    
    # Save raw results
    with open(os.path.join(args.output_dir, 'ablation_raw_results.json'), 'w') as f:
        # Convert DataFrames to dictionaries for JSON serialization
        serializable_results = {}
        for component, results in all_results.items():
            if results:
                serializable_results[component] = {
                    setting: df.to_dict() if isinstance(df, pd.DataFrame) else None
                    for setting, df in results.items()
                }
        
        json.dump(serializable_results, f, indent=2)

if __name__ == "__main__":
    main() 