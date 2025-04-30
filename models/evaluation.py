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
    print(f"Loaded test data: {len(test_df)} entries")
    return test_df

def load_target_pool(target_file, expanded=False):
    """
    Load the pool of target LOINC codes
    
    Args:
        target_file: Path to the file with target LOINC codes
        expanded: Whether to use the expanded target pool
        
    Returns:
        target_df: DataFrame with target LOINC codes
    """
    target_df = pd.read_csv(target_file)
    
    if expanded:
        # If expanded target pool file exists, load it
        expanded_file = os.path.join(os.path.dirname(target_file), "expanded_target_pool.txt")
        if os.path.exists(expanded_file):
            expanded_targets = pd.read_csv(expanded_file, header=None, names=["LOINC_NUM"])
            print(f"Loaded expanded target pool: {len(expanded_targets)} LOINC codes")
            return expanded_targets
        else:
            print("Warning: Expanded target pool file not found. Using regular target pool.")
    
    # Extract unique LOINC codes
    unique_targets = target_df["LOINC_NUM"].drop_duplicates().reset_index(drop=True)
    print(f"Loaded target pool: {len(unique_targets)} unique LOINC codes")
    
    return pd.DataFrame({"LOINC_NUM": unique_targets})

def compute_embeddings(model, texts, batch_size=32):
    """
    Compute embeddings for a list of texts
    
    Args:
        model: Trained LOINCEncoder model
        texts: List of text strings
        batch_size: Batch size for processing
        
    Returns:
        embeddings: NumPy array of embeddings
    """
    embeddings_list = []
    
    # Process texts in batches to avoid memory issues
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        with tf.device('/CPU:0'):  # Force text processing on CPU
            batch_embeddings = model(inputs=batch_texts, training=False).numpy()
        embeddings_list.append(batch_embeddings)
    
    # Concatenate batch embeddings
    if embeddings_list:
        embeddings = np.vstack(embeddings_list)
    else:
        # Return empty array if no texts
        embeddings = np.array([])
    
    return embeddings

def augment_test_data(test_df, num_augmentations=10):
    """
    Create augmented versions of the test data
    
    Args:
        test_df: DataFrame with test data
        num_augmentations: Number of augmentations per test sample
        
    Returns:
        augmented_df: DataFrame with augmented test data
    """
    try:
        from preprocessing.data_augmentation import augment_text
    except ImportError:
        print("Warning: Could not import augment_text function. Using simple character substitution instead.")
        # Define a simple augmentation function
        def augment_text(text):
            import random
            import string
            # Simple character substitution
            chars = list(text)
            # Replace ~10% of characters
            for i in range(max(1, len(chars) // 10)):
                pos = random.randint(0, len(chars) - 1)
                # Skip spaces
                if chars[pos] == ' ':
                    continue
                # Replace with similar character or skip
                if random.random() < 0.5:
                    # Skip character (deletion)
                    chars[pos] = ''
                else:
                    # Character substitution with similar looking or adjacent keyboard character
                    similar_chars = {
                        'a': 'aes', 'b': 'bdp', 'c': 'ceo', 'd': 'dsf', 'e': 'ear', 
                        'f': 'frt', 'g': 'gfh', 'h': 'hgj', 'i': 'iuo', 'j': 'jkh',
                        'k': 'klj', 'l': 'lko', 'm': 'mn', 'n': 'nbm', 'o': 'oip',
                        'p': 'po', 'q': 'qw', 'r': 'rte', 's': 'sad', 't': 'try',
                        'u': 'uyi', 'v': 'vc', 'w': 'wqe', 'x': 'xzc', 'y': 'yut',
                        'z': 'zax'
                    }
                    if chars[pos].lower() in similar_chars:
                        similar = similar_chars[chars[pos].lower()]
                        chars[pos] = random.choice(similar)
            
            # Randomly insert space in ~10% of cases
            if len(chars) > 5 and random.random() < 0.1:
                pos = random.randint(1, len(chars) - 1)
                chars.insert(pos, ' ')
            
            return ''.join(chars)
    
    augmented_data = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Augmenting test data"):
        source_text = row["SOURCE"]
        target_loinc = row["LOINC_NUM"]
        
        # Add the original row
        augmented_data.append({
            "SOURCE": source_text,
            "LOINC_NUM": target_loinc,
            "is_augmented": False
        })
        
        # Create augmented versions - make sure they're all simple strings
        for _ in range(num_augmentations):
            try:
                # Ensure we have a plain string
                if isinstance(source_text, str):
                    augmented_text = augment_text(source_text)
                    # Verify the output is a string
                    if isinstance(augmented_text, str):
                        augmented_data.append({
                            "SOURCE": augmented_text,
                            "LOINC_NUM": target_loinc,
                            "is_augmented": True
                        })
                    else:
                        print(f"Warning: Augmentation returned a non-string: {type(augmented_text)}. Skipping.")
                else:
                    print(f"Warning: Source text is not a string: {type(source_text)}. Skipping.")
            except Exception as e:
                print(f"Error during augmentation: {e}. Skipping.")
    
    augmented_df = pd.DataFrame(augmented_data)
    print(f"Created augmented test data: {len(augmented_df)} entries")
    
    return augmented_df

def calculate_top_k_accuracy(source_embeddings, target_embeddings_dict, source_loincs, k_values=[1, 3, 5]):
    """
    Calculate top-k accuracy for source embeddings against target embeddings
    
    Args:
        source_embeddings: NumPy array of source embeddings
        target_embeddings_dict: Dictionary mapping LOINC codes to embeddings
        source_loincs: List of ground truth LOINC codes for sources
        k_values: List of k values for top-k accuracy
        
    Returns:
        metrics: Dictionary with top-k accuracy values
    """
    # Convert target embeddings dict to arrays
    target_loincs = list(target_embeddings_dict.keys())
    target_embeddings = np.array([target_embeddings_dict[loinc] for loinc in target_loincs])
    
    # Calculate cosine similarity between source and target embeddings
    # For cosine similarity, we want to maximize: cosine_sim = dot(a, b) / (||a|| * ||b||)
    # Since embeddings are L2-normalized, this simplifies to: cosine_sim = dot(a, b)
    similarity_matrix = np.matmul(source_embeddings, target_embeddings.T)
    
    # Sort similarities in descending order (highest similarity first)
    # Get indices of targets sorted by similarity
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    
    # Initialize accuracy metrics
    metrics = {}
    for k in k_values:
        correct_count = 0
        
        # Check if the ground truth is in the top-k predictions
        for i, source_loinc in enumerate(source_loincs):
            # Get top-k target LOINC codes for this source
            top_k_indices = sorted_indices[i, :k]
            top_k_loincs = [target_loincs[idx] for idx in top_k_indices]
            
            # Check if the ground truth LOINC is in the top-k
            if source_loinc in top_k_loincs:
                correct_count += 1
        
        # Calculate accuracy
        accuracy = correct_count / len(source_loincs) if len(source_loincs) > 0 else 0
        metrics[f'top_{k}_accuracy'] = accuracy
    
    return metrics

def evaluate_model(model, test_df, target_loinc_text_dict, use_augmentation=False, expanded_targets=False):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained LOINCEncoder model
        test_df: DataFrame with test data
        target_loinc_text_dict: Dictionary mapping LOINC codes to their text representations
        use_augmentation: Whether to use augmented test data
        expanded_targets: Whether to use the expanded target pool
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    # Prepare test data
    if use_augmentation:
        # Create augmented versions of test data
        eval_df = augment_test_data(test_df)
    else:
        # Use original test data
        eval_df = test_df.copy()
        eval_df["is_augmented"] = False
    
    # Extract source texts and ground truth LOINC codes
    source_texts = eval_df["SOURCE"].tolist()
    source_loincs = eval_df["LOINC_NUM"].tolist()
    
    print("Computing source embeddings...")
    # Compute embeddings for source texts
    source_embeddings = compute_embeddings(model, source_texts)
    
    print("Computing target embeddings...")
    # Compute embeddings for all target LOINC codes
    target_embeddings_dict = {}
    for loinc, text in tqdm(target_loinc_text_dict.items(), desc="Embedding targets"):
        embedding = compute_embeddings(model, [text])[0]  # Single embedding
        target_embeddings_dict[loinc] = embedding
    
    print("Calculating metrics...")
    # Calculate top-k accuracy
    metrics = calculate_top_k_accuracy(
        source_embeddings,
        target_embeddings_dict,
        source_loincs,
        k_values=[1, 3, 5]  # As used in the paper
    )
    
    # Display results
    for k, acc in metrics.items():
        print(f"{k}: {acc:.4f}")
    
    # Add info about augmentation and target pool size
    metrics["augmented"] = use_augmentation
    metrics["target_pool_size"] = len(target_embeddings_dict)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate LOINC standardization model')
    parser.add_argument('--test_file', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--loinc_file', type=str, required=True, help='Path to LOINC data CSV')
    parser.add_argument('--checkpoint_dir', type=str, default='models/checkpoints', help='Directory with model checkpoints')
    parser.add_argument('--fold_idx', type=int, default=None, help='Specific fold to evaluate (default: evaluate all folds)')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--augmented', action='store_true', help='Use augmented test data')
    parser.add_argument('--expanded_targets', action='store_true', help='Use expanded target pool')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    test_df = load_test_data(args.test_file)
    
    # Load LOINC data for target text representations
    loinc_df = pd.read_csv(args.loinc_file)
    
    # Create dictionary mapping LOINC codes to their text representations
    target_loinc_text_dict = {}
    for _, row in loinc_df.iterrows():
        loinc_code = row["LOINC_NUM"]
        # Use LONG_COMMON_NAME as the text representation, if available
        if "LONG_COMMON_NAME" in row and pd.notna(row["LONG_COMMON_NAME"]):
            target_loinc_text_dict[loinc_code] = row["LONG_COMMON_NAME"]
        # Fallback to DISPLAY_NAME if LONG_COMMON_NAME is not available
        elif "DISPLAY_NAME" in row and pd.notna(row["DISPLAY_NAME"]):
            target_loinc_text_dict[loinc_code] = row["DISPLAY_NAME"]
        # Fallback to SHORTNAME if neither is available
        elif "SHORTNAME" in row and pd.notna(row["SHORTNAME"]):
            target_loinc_text_dict[loinc_code] = row["SHORTNAME"]
    
    # Initialize results storage
    results = []
    
    if args.fold_idx is not None:
        # Evaluate specific fold
        fold_indices = [args.fold_idx]
    else:
        # Evaluate all folds
        fold_indices = range(5)  # Default to 5 folds
    
    for fold_idx in fold_indices:
        print(f"\n--- Evaluating Fold {fold_idx+1} ---")
        
        # Load the model for this fold
        checkpoint_path = os.path.join(args.checkpoint_dir, f"stage2_fold{fold_idx+1}_model.weights.h5")
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint for fold {fold_idx+1} not found at {checkpoint_path}")
            continue
            
        # Initialize model with the same architecture
        model = LOINCEncoder(embedding_dim=args.embedding_dim, dropout_rate=0.0)  # No dropout during evaluation
        
        # Call once to build model
        _ = model(inputs=["dummy text"])
        
        # Load weights
        try:
            model.load_weights(checkpoint_path)
            print(f"Loaded weights from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading weights from {checkpoint_path}: {e}")
            continue
        
        # Get test data for this fold
        # In a real implementation, you would load the specific test split for this fold
        fold_test_df = test_df  # Replace with actual fold-specific test data
        
        # Evaluate model
        fold_metrics = evaluate_model(
            model,
            fold_test_df,
            target_loinc_text_dict,
            use_augmentation=args.augmented,
            expanded_targets=args.expanded_targets
        )
        
        # Add fold information
        fold_metrics['fold'] = fold_idx+1
        
        # Store results
        results.append(fold_metrics)
        
        # Save fold-specific results
        output_file = os.path.join(
            args.output_dir, 
            f"fold{fold_idx+1}_{'augmented' if args.augmented else 'regular'}_{'expanded' if args.expanded_targets else 'standard'}.csv"
        )
        pd.DataFrame([fold_metrics]).to_csv(output_file, index=False)
        print(f"Results for fold {fold_idx+1} saved to {output_file}")
    
    if results:
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save combined results
        output_file = os.path.join(
            args.output_dir, 
            f"all_folds_{'augmented' if args.augmented else 'regular'}_{'expanded' if args.expanded_targets else 'standard'}.csv"
        )
        results_df.to_csv(output_file, index=False)
        print(f"\nCombined results saved to {output_file}")
        
        # Show average metrics across folds
        print("\nAverage metrics across folds:")
        for metric in ['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy']:
            if metric in results_df.columns:
                print(f"{metric}: {results_df[metric].mean():.4f} ± {results_df[metric].std():.4f}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main() 