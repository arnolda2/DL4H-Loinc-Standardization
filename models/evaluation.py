import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
import sys
import time
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.t5_encoder import LOINCEncoder
from preprocessing.data_augmentation import augment_text

def load_test_data(test_file):
    """
    Load test data from CSV file
    
    Args:
        test_file: Path to the test data CSV file
        
    Returns:
        test_df: DataFrame with test data
    """
    test_df = pd.read_csv(test_file)
    print(f"Loaded {len(test_df)} test samples from {test_file}")
    
    # Check if this is an augmented test file (has 'is_augmented' column)
    if 'is_augmented' in test_df.columns:
        print(f"Found augmented test data: {len(test_df[test_df['is_augmented']])} augmented samples, {len(test_df[~test_df['is_augmented']])} original samples")
    
    required_columns = ['SOURCE', 'LOINC_NUM']
    missing_columns = [col for col in required_columns if col not in test_df.columns]
    if missing_columns:
        raise ValueError(f"Test data is missing required columns: {missing_columns}")
    
    return test_df

def load_target_loincs(loinc_file):
    """
    Load LOINC target data
    
    Args:
        loinc_file: Path to the LOINC data CSV file
        
    Returns:
        target_df: DataFrame with LOINC targets
    """
    target_df = pd.read_csv(loinc_file)
    print(f"Loaded {len(target_df)} LOINC targets from {loinc_file}")
    
    required_columns = ['LOINC_NUM']
    missing_columns = [col for col in required_columns if col not in target_df.columns]
    if missing_columns:
        raise ValueError(f"LOINC data is missing required columns: {missing_columns}")
    
    # Handle the target text column which might be named differently
    text_column_candidates = ['TARGET', 'LONG_COMMON_NAME', 'DisplayName', 'SHORTNAME']
    available_text_columns = [col for col in text_column_candidates if col in target_df.columns]
    
    if not available_text_columns:
        raise ValueError(f"LOINC data does not have any suitable text column. Need one of: {text_column_candidates}")
    
    # Use the first available text column as the TARGET
    text_column = available_text_columns[0]
    print(f"Using '{text_column}' as the target text column")
    
    # Create a new dataframe with LOINC_NUM and TARGET columns
    processed_df = pd.DataFrame({
        'LOINC_NUM': target_df['LOINC_NUM'],
        'TARGET': target_df[text_column]
    })
    
    return processed_df

def load_model(checkpoint_dir, fold):
    """
    Load trained model for the specified fold
    
    Args:
        checkpoint_dir: Directory with model checkpoints
        fold: Fold index
        
    Returns:
        model: Loaded model
    """
    model_path = os.path.join(checkpoint_dir, f"stage2_fold{fold+1}_model.weights.h5")
    if not os.path.exists(model_path):
        raise ValueError(f"Model checkpoint not found at {model_path}")
    
    try:
        # Import the model class
        from models.t5_encoder import LOINCEncoder
        
        # Initialize the model
        model = LOINCEncoder(embedding_dim=128, dropout_rate=0.0)
        
        # Create a dummy input to build the model
        _ = model(inputs=["dummy text"])
        
        # Load the weights
        model.load_weights(model_path)
        print(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def compute_embeddings(texts, model, batch_size=16):
    """
    Compute embeddings for texts
    
    Args:
        texts: List of texts to embed
        model: Trained model
        batch_size: Batch size for inference
        
    Returns:
        embeddings: Numpy array of embeddings
    """
    try:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # Ensure all texts are strings
            batch_texts = [str(text) if not isinstance(text, str) else text for text in batch_texts]
            # Calculate embeddings for batch
            batch_embeddings = model(inputs=batch_texts, training=False).numpy()
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        raise

def evaluate_top_k_accuracy(test_df, target_df, model, k_values=[1, 3, 5], batch_size=16, 
                           augmented_test=False, use_only_original=False):
    """
    Evaluate Top-k accuracy
    
    Args:
        test_df: DataFrame with test data
        target_df: DataFrame with LOINC targets
        model: Trained model
        k_values: List of k values for Top-k accuracy
        batch_size: Batch size for inference
        augmented_test: Whether this is augmented test data
        use_only_original: If True, only use original samples from augmented test data
        
    Returns:
        results: Dictionary with Top-k accuracy results
    """
    # Preprocess data if needed
    if augmented_test and use_only_original and 'is_augmented' in test_df.columns:
        print("Using only original samples from augmented test data")
        test_df = test_df[~test_df['is_augmented']]
    
    # Get unique target LOINCs
    unique_target_loincs = target_df['LOINC_NUM'].unique()
    unique_target_texts = target_df['TARGET'].unique()
    print(f"Evaluating against {len(unique_target_loincs)} unique LOINC targets")
    
    # Check if test LOINCs exist in target LOINCs
    test_loincs = test_df['LOINC_NUM'].unique()
    matching_loincs = set(test_loincs) & set(unique_target_loincs)
    print(f"Test data has {len(test_loincs)} unique LOINCs, {len(matching_loincs)} match with target LOINCs")
    
    if len(matching_loincs) == 0:
        print("WARNING: No matching LOINCs between test and target data!")
        print(f"Test LOINCs: {test_loincs}")
        print(f"First few target LOINCs: {list(unique_target_loincs)[:10]}")
    
    # Get source texts and target LOINCs
    source_texts = test_df['SOURCE'].tolist()
    target_loincs = test_df['LOINC_NUM'].tolist()
    
    # Compute embeddings for target LOINCs
    print("Computing embeddings for target LOINCs...")
    target_texts = []
    for loinc in tqdm(unique_target_loincs):
        # Use first matching text if multiple exist for the same LOINC code
        matching_rows = target_df[target_df['LOINC_NUM'] == loinc]
        target_text = matching_rows.iloc[0]['TARGET']
        target_texts.append(target_text)
    
    target_embeddings = compute_embeddings(target_texts, model, batch_size)
    
    # Compute embeddings for source texts
    print("Computing embeddings for source texts...")
    source_embeddings = compute_embeddings(source_texts, model, batch_size)
    
    # Create dictionary mapping LOINC codes to their indices in the target embeddings
    loinc_to_index = {loinc: i for i, loinc in enumerate(unique_target_loincs)}
    
    # Calculate pairwise distances
    print("Calculating similarities...")
    # Using negative cosine distance (higher is better)
    similarities = -pairwise_distances(source_embeddings, target_embeddings, metric='cosine')
    
    # Calculate Top-k accuracy
    results = {}
    for k in k_values:
        # Get top k indices for each source
        top_k_indices = np.argsort(similarities, axis=1)[:, -k:]
        
        # Check if correct target is in top k
        correct = 0
        for i, target_loinc in enumerate(target_loincs):
            # Get the target LOINC's index
            if target_loinc in loinc_to_index:
                target_idx = loinc_to_index[target_loinc]
                # Check if target index is in top k
                if target_idx in top_k_indices[i]:
                    correct += 1
            else:
                print(f"WARNING: Target LOINC {target_loinc} not in target pool")
        
        # Calculate accuracy
        accuracy = correct / len(source_texts)
        results[f'top{k}_accuracy'] = accuracy
        print(f"Top-{k} accuracy: {accuracy:.4f} ({correct}/{len(source_texts)})")
    
    # Calculate Mean Reciprocal Rank (MRR)
    reciprocal_ranks = []
    for i, target_loinc in enumerate(target_loincs):
        if target_loinc in loinc_to_index:
            target_idx = loinc_to_index[target_loinc]
            # Get rank of correct target (add 1 because indices are 0-based)
            rank = np.where(np.argsort(similarities[i])[::-1] == target_idx)[0][0] + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    mrr = np.mean(reciprocal_ranks)
    results['mrr'] = mrr
    print(f"Mean Reciprocal Rank: {mrr:.4f}")
    
    # Add target pool size to results
    results['target_pool_size'] = len(unique_target_loincs)
    results['matching_loincs'] = len(matching_loincs)
    results['test_samples'] = len(source_texts)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate LOINC standardization model')
    parser.add_argument('--test_file', type=str, required=True, 
                        help='Path to test data CSV')
    parser.add_argument('--loinc_file', type=str, required=True, 
                        help='Path to LOINC data CSV')
    parser.add_argument('--checkpoint_dir', type=str, required=True, 
                        help='Directory containing model checkpoints')
    parser.add_argument('--fold', type=int, default=0, 
                        help='Fold to evaluate (0-indexed)')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 3, 5, 10], 
                        help='k values for Top-k accuracy')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for inference')
    parser.add_argument('--expanded_pool', action='store_true',
                        help='Whether to use expanded target pool')
    parser.add_argument('--augmented_test', action='store_true',
                        help='Whether this is augmented test data')
    parser.add_argument('--original_only', action='store_true',
                        help='Only use original samples from augmented test data')
    parser.add_argument('--ablation_id', type=str, default=None,
                        help='Identifier for ablation study (used for output filename)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start time
    start_time = time.time()
    
    # Load data
    print(f"Loading test data from {args.test_file}...")
    test_df = load_test_data(args.test_file)
    
    print(f"Loading LOINC targets from {args.loinc_file}...")
    target_df = load_target_loincs(args.loinc_file)
    
    # Load model
    print(f"Loading model for fold {args.fold}...")
    model = load_model(args.checkpoint_dir, args.fold)
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_top_k_accuracy(
        test_df=test_df,
        target_df=target_df,
        model=model,
        k_values=args.k_values,
        batch_size=args.batch_size,
        augmented_test=args.augmented_test,
        use_only_original=args.original_only
    )
    
    # Add fold information to results
    results['fold'] = args.fold
    
    # Save results to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create output filename
    file_name_parts = [f'fold{args.fold}']
    if args.augmented_test:
        file_name_parts.append('augmented')
    if args.expanded_pool:
        file_name_parts.append('expanded')
    if args.ablation_id:
        file_name_parts.append(f'ablation_{args.ablation_id}')
    file_name_parts.append('results.csv')
    results_file = os.path.join(args.output_dir, '_'.join(file_name_parts))
    
    # Save results
    pd.DataFrame([results]).to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    
    # End time
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 