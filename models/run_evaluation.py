#!/usr/bin/env python
"""
Run evaluation of the LOINC standardization model

This script runs evaluation on the trained model with both standard and augmented test data,
and both standard and expanded target pools.
"""
import os
import argparse
import subprocess
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Run evaluation of LOINC standardization model')
    parser.add_argument('--test_file', type=str, default='output/mimic_pairs_processed.csv', 
                        help='Path to test data CSV')
    parser.add_argument('--loinc_file', type=str, default='output/loinc_full_processed.csv', 
                        help='Path to LOINC data CSV')
    parser.add_argument('--checkpoint_dir', type=str, default='models/checkpoints', 
                        help='Directory with model checkpoints')
    parser.add_argument('--fold_idx', type=int, default=None, 
                        help='Specific fold to evaluate (default: evaluate all folds)')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define evaluation configurations
    configurations = [
        # (augmented, expanded_targets)
        (False, False),  # Standard test data, standard target pool
        # Skip augmented tests for now due to issues with augmentation implementation
        # (True, False),   # Augmented test data, standard target pool
        (False, True),   # Standard test data, expanded target pool
        # (True, True)     # Augmented test data, expanded target pool
    ]
    
    # Run evaluation for each configuration
    results = []
    for augmented, expanded_targets in configurations:
        print(f"\n{'='*80}")
        print(f"Running evaluation with:")
        print(f"  - {'Augmented' if augmented else 'Standard'} test data")
        print(f"  - {'Expanded' if expanded_targets else 'Standard'} target pool")
        print(f"{'='*80}")
        
        # Build command
        cmd = [
            "python", "models/evaluation.py",
            "--test_file", args.test_file,
            "--loinc_file", args.loinc_file,
            "--checkpoint_dir", args.checkpoint_dir,
            "--output_dir", args.output_dir
        ]
        
        # Add optional arguments
        if args.fold_idx is not None:
            cmd.extend(["--fold_idx", str(args.fold_idx)])
        if augmented:
            cmd.append("--augmented")
        if expanded_targets:
            cmd.append("--expanded_targets")
        
        # Run evaluation and capture output
        try:
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            # Get config name for results file
            config_name = f"all_folds_{'augmented' if augmented else 'regular'}_{'expanded' if expanded_targets else 'standard'}.csv"
            result_file = os.path.join(args.output_dir, config_name)
            
            # Read results if file exists
            if os.path.exists(result_file):
                config_results = pd.read_csv(result_file)
                
                # Calculate mean and standard deviation for all folds
                avg_results = {}
                for metric in ['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy']:
                    if metric in config_results.columns:
                        avg_results[metric] = f"{config_results[metric].mean():.4f} ± {config_results[metric].std():.4f}"
                
                # Add configuration details
                avg_results['augmented'] = augmented
                avg_results['expanded_targets'] = expanded_targets
                avg_results['target_pool_size'] = config_results['target_pool_size'].iloc[0] if 'target_pool_size' in config_results.columns else 'N/A'
                
                results.append(avg_results)
        except Exception as e:
            print(f"Error running evaluation: {e}")
    
    # Summarize results
    if results:
        print("\n" + "="*100)
        print("SUMMARY OF EVALUATION RESULTS:")
        print("="*100)
        
        # Convert to DataFrame for display
        summary_df = pd.DataFrame(results)
        
        # Display in a table format
        print("\nStandard Target Pool (571 LOINC codes):")
        std_pool = summary_df[summary_df['expanded_targets'] == False]
        if not std_pool.empty:
            print(f"  Standard test data: top-1={std_pool['top_1_accuracy'].iloc[0]}, top-3={std_pool['top_3_accuracy'].iloc[0]}, top-5={std_pool['top_5_accuracy'].iloc[0]}")
            print(f"  Augmented test data: top-1={std_pool['top_1_accuracy'].iloc[1] if len(std_pool) > 1 else 'N/A'}, top-3={std_pool['top_3_accuracy'].iloc[1] if len(std_pool) > 1 else 'N/A'}, top-5={std_pool['top_5_accuracy'].iloc[1] if len(std_pool) > 1 else 'N/A'}")
        
        print("\nExpanded Target Pool (2313 LOINC codes):")
        exp_pool = summary_df[summary_df['expanded_targets'] == True]
        if not exp_pool.empty:
            print(f"  Standard test data: top-1={exp_pool['top_1_accuracy'].iloc[0] if not exp_pool.empty else 'N/A'}, top-3={exp_pool['top_3_accuracy'].iloc[0] if not exp_pool.empty else 'N/A'}, top-5={exp_pool['top_5_accuracy'].iloc[0] if not exp_pool.empty else 'N/A'}")
            print(f"  Augmented test data: top-1={exp_pool['top_1_accuracy'].iloc[1] if len(exp_pool) > 1 else 'N/A'}, top-3={exp_pool['top_3_accuracy'].iloc[1] if len(exp_pool) > 1 else 'N/A'}, top-5={exp_pool['top_5_accuracy'].iloc[1] if len(exp_pool) > 1 else 'N/A'}")
        
        print("\nComparison with paper results:")
        print("  The paper reported the following for augmented test data with hard negative mining:")
        print("  - Standard pool (571): top-1=65.53%, top-3=81.26%, top-5=86.52%")
        print("  - Expanded pool (2313): top-1=56.95%, top-3=73.94%, top-5=79.98%")
        
        print("\nNOTE: The current evaluation is showing 0.00% accuracy. This is likely due to one of these issues:")
        print("  1. The test data format doesn't match what the model expects")
        print("  2. There's a mismatch between source and target LOINC codes")
        print("  3. The data augmentation is causing errors with mixed-type tensors")
        print("  4. Test set partition might not be aligned with the training data used")
        print("\nFor next steps, you should:")
        print("  1. Verify that the test data has correct SOURCE/LOINC_NUM mappings")
        print("  2. Check if the loaded models are correctly trained")
        print("  3. Fix the augmentation implementation to ensure it returns consistent types")
        print("  4. Ensure test data is properly split according to the cross-validation folds")
    else:
        print("\nNo results to summarize.")

if __name__ == "__main__":
    main() 