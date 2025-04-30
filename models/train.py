import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
from t5_encoder import LOINCEncoder
from triplet_loss import triplet_loss, batch_hard_triplet_loss, batch_semi_hard_triplet_loss
from triplet_mining import generate_stage1_triplets, generate_stage2_triplets, TripletBatchGenerator
from sklearn.model_selection import KFold
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_augmentation import augment_text

def load_stage1_data(loinc_file_path, num_triplets=10000):
    """
    Load and prepare data for Stage 1 fine-tuning (target-only).
    
    Args:
        loinc_file_path: Path to the processed LOINC csv file
        num_triplets: Number of triplets to generate
        
    Returns:
        List of triplets (anchor, positive, negative)
    """
    print(f"Loading LOINC data from {loinc_file_path}")
    loinc_df = pd.read_csv(loinc_file_path)
    
    # Collect all text representations and their corresponding LOINC codes
    loinc_texts = []
    loinc_codes = []
    
    # For each LOINC code, collect text from different fields
    for _, row in loinc_df.iterrows():
        loinc_code = row['LOINC_NUM']
        
        for field in ['LONG_COMMON_NAME', 'SHORTNAME', 'DisplayName', 'RELATEDNAMES2']:
            if pd.notna(row[field]) and row[field]:
                loinc_texts.append(row[field].lower())
                loinc_codes.append(loinc_code)
    
    print(f"Loaded {len(loinc_texts)} text representations for {len(np.unique(loinc_codes))} unique LOINC codes")
    
    # Generate triplets for Stage 1
    triplets = generate_stage1_triplets(loinc_texts, loinc_codes, num_triplets)
    print(f"Generated {len(triplets)} triplets for Stage 1 fine-tuning")
    
    return triplets

def load_stage2_data(mimic_file_path, fold_indices=None, fold_idx=0, num_triplets=5000):
    """
    Load and prepare data for Stage 2 fine-tuning (source-target pairs).
    
    Args:
        mimic_file_path: Path to the processed MIMIC-III pairs csv file
        fold_indices: Dictionary of fold indices for cross-validation
        fold_idx: Current fold index
        num_triplets: Number of triplets to generate
        
    Returns:
        Training triplets, validation triplets
    """
    print(f"Loading MIMIC-III pairs from {mimic_file_path}")
    mimic_df = pd.read_csv(mimic_file_path)
    
    if fold_indices is None:
        # If no fold indices are provided, use 80-20 split
        train_size = int(0.8 * len(mimic_df))
        train_df = mimic_df.iloc[:train_size]
        val_df = mimic_df.iloc[train_size:]
    else:
        # Use provided fold indices
        train_idx = fold_indices[f'fold{fold_idx}']['train_idx']
        val_idx = fold_indices[f'fold{fold_idx}']['val_idx']
        train_df = mimic_df.iloc[train_idx]
        val_df = mimic_df.iloc[val_idx]
    
    print(f"Training set: {len(train_df)} samples, Validation set: {len(val_df)} samples")
    
    # Generate triplets for training
    train_triplets = generate_stage2_triplets(
        train_df['source_text'].tolist(),
        train_df['target_loinc'].tolist(),
        num_triplets
    )
    
    # Generate triplets for validation
    val_triplets = generate_stage2_triplets(
        val_df['source_text'].tolist(),
        val_df['target_loinc'].tolist(),
        num_triplets // 5  # Fewer triplets for validation
    )
    
    print(f"Generated {len(train_triplets)} training triplets and {len(val_triplets)} validation triplets")
    
    return train_triplets, val_triplets

def train_stage1(loinc_file_path, output_dir, num_triplets=10000, batch_size=900,
                learning_rate=1e-4, epochs=30, mining_strategy='semi-hard'):
    """
    Perform Stage 1 fine-tuning using the LOINC target corpus.
    
    Args:
        loinc_file_path: Path to the processed LOINC csv file
        output_dir: Directory to save model checkpoints
        num_triplets: Number of triplets to generate
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        epochs: Number of training epochs
        mining_strategy: 'hard' or 'semi-hard' triplet mining strategy
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load data and generate triplets
    triplets = load_stage1_data(loinc_file_path, num_triplets)
    
    # Create training batch generator
    train_size = int(0.8 * len(triplets))
    train_triplets = triplets[:train_size]
    val_triplets = triplets[train_size:]
    
    train_gen = TripletBatchGenerator(train_triplets, batch_size=batch_size)
    val_gen = TripletBatchGenerator(val_triplets, batch_size=batch_size)
    
    # Step 2: Create the model
    # Force CPU for text processing
    with tf.device('/CPU:0'):
        loinc_encoder = LOINCEncoder()
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Create a model for inference/saving
    with tf.device('/CPU:0'):
        anchor_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='anchor_input')
        anchor_embedding = loinc_encoder(anchor_input)
        inference_model = tf.keras.Model(inputs=anchor_input, outputs=anchor_embedding)
    
    # Define training step (force CPU for text processing)
    def train_step(anchors, positives, negatives):
        with tf.GradientTape() as tape:
            # Forward pass on CPU
            with tf.device('/CPU:0'):
                anchor_embeddings = loinc_encoder(anchors, training=True)
                positive_embeddings = loinc_encoder(positives, training=True)
                negative_embeddings = loinc_encoder(negatives, training=True)
            
            # Compute loss
            loss_value = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        # Compute gradients
        gradients = tape.gradient(loss_value, loinc_encoder.trainable_variables)
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, loinc_encoder.trainable_variables))
        
        return loss_value
    
    # Define validation step
    def val_step(anchors, positives, negatives):
        # Forward pass on CPU
        with tf.device('/CPU:0'):
            anchor_embeddings = loinc_encoder(anchors, training=False)
            positive_embeddings = loinc_encoder(positives, training=False)
            negative_embeddings = loinc_encoder(negatives, training=False)
        
        # Compute loss
        loss_value = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        return loss_value
    
    # Step 3: Train the model
    print(f"Starting Stage 1 fine-tuning with {mining_strategy} mining strategy")
    
    # Keep track of best validation loss
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        for batch_idx in range(len(train_gen)):
            # Get batch
            batch_data, _ = train_gen[batch_idx]
            anchors, positives, negatives = batch_data
            
            # Training step
            batch_loss = train_step(anchors, positives, negatives)
            train_loss += batch_loss
        
        train_loss /= len(train_gen)
        
        # Validation
        val_loss = 0.0
        for batch_idx in range(len(val_gen)):
            # Get batch
            batch_data, _ = val_gen[batch_idx]
            anchors, positives, negatives = batch_data
            
            # Validation step
            batch_loss = val_step(anchors, positives, negatives)
            val_loss += batch_loss
        
        val_loss /= len(val_gen)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        
        # Save if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(output_dir, 'stage1_model.weights.h5')
            loinc_encoder.save_weights(checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")
    
    # Save the encoder model
    encoder_path = os.path.join(output_dir, 'stage1_encoder')
    loinc_encoder.save_weights(encoder_path)
    
    print(f"Stage 1 fine-tuning completed. Encoder saved to {encoder_path}")
    
    return loinc_encoder

def train_stage2(mimic_file_path, output_dir, loinc_encoder=None, fold_indices=None, 
                fold_idx=0, num_triplets=5000, batch_size=900, learning_rate=1e-5, 
                epochs=30, mining_strategy='hard'):
    """
    Perform Stage 2 fine-tuning using the MIMIC-III source-target pairs.
    
    Args:
        mimic_file_path: Path to the processed MIMIC-III pairs csv file
        output_dir: Directory to save model checkpoints
        loinc_encoder: Pre-trained LOINC encoder from Stage 1 (or None to create new)
        fold_indices: Dictionary of fold indices for cross-validation
        fold_idx: Current fold index
        num_triplets: Number of triplets to generate
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        epochs: Number of training epochs
        mining_strategy: 'hard' or 'semi-hard' triplet mining strategy
    """
    # Create output directory if it doesn't exist
    fold_output_dir = os.path.join(output_dir, f'fold{fold_idx}')
    os.makedirs(fold_output_dir, exist_ok=True)
    
    # Step 1: Load data and generate triplets
    train_triplets, val_triplets = load_stage2_data(
        mimic_file_path, fold_indices, fold_idx, num_triplets
    )
    
    train_gen = TripletBatchGenerator(train_triplets, batch_size=batch_size)
    val_gen = TripletBatchGenerator(val_triplets, batch_size=batch_size)
    
    # Step 2: Create the model
    with tf.device('/CPU:0'):
        if loinc_encoder is None:
            loinc_encoder = LOINCEncoder()
            # Load Stage 1 weights if available
            stage1_path = os.path.join(output_dir, 'stage1_encoder')
            if os.path.exists(stage1_path + '.index'):
                loinc_encoder.load_weights(stage1_path)
    
        # Enable dropout for Stage 2 fine-tuning (as mentioned in the paper)
        # This adds regularization for the second stage
        loinc_encoder.use_dropout = True
        print("Enabled dropout for Stage 2 fine-tuning")
    
    # Create optimizer with lower learning rate for fine-tuning
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Create a model for inference/saving
    with tf.device('/CPU:0'):
        anchor_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='anchor_input')
        anchor_embedding = loinc_encoder(anchor_input)
        inference_model = tf.keras.Model(inputs=anchor_input, outputs=anchor_embedding)
    
    # Define training step
    def train_step(anchors, positives, negatives):
        with tf.GradientTape() as tape:
            # Forward pass on CPU
            with tf.device('/CPU:0'):
                anchor_embeddings = loinc_encoder(anchors, training=True)
                positive_embeddings = loinc_encoder(positives, training=True)
                negative_embeddings = loinc_encoder(negatives, training=True)
            
            # Compute loss
            loss_value = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        # Compute gradients
        gradients = tape.gradient(loss_value, loinc_encoder.trainable_variables)
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, loinc_encoder.trainable_variables))
        
        return loss_value
    
    # Define validation step
    def val_step(anchors, positives, negatives):
        # Forward pass on CPU
        with tf.device('/CPU:0'):
            anchor_embeddings = loinc_encoder(anchors, training=False)
            positive_embeddings = loinc_encoder(positives, training=False)
            negative_embeddings = loinc_encoder(negatives, training=False)
        
        # Compute loss
        loss_value = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        return loss_value
    
    # Step 3: Train the model
    print(f"Starting Stage 2 fine-tuning for fold {fold_idx} with {mining_strategy} mining strategy")
    
    # Keep track of best validation loss
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        for batch_idx in range(len(train_gen)):
            # Get batch
            batch_data, _ = train_gen[batch_idx]
            anchors, positives, negatives = batch_data
            
            # Training step
            batch_loss = train_step(anchors, positives, negatives)
            train_loss += batch_loss
        
        train_loss /= len(train_gen)
        
        # Validation
        val_loss = 0.0
        for batch_idx in range(len(val_gen)):
            # Get batch
            batch_data, _ = val_gen[batch_idx]
            anchors, positives, negatives = batch_data
            
            # Validation step
            batch_loss = val_step(anchors, positives, negatives)
            val_loss += batch_loss
        
        val_loss /= len(val_gen)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        
        # Save if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(fold_output_dir, 'stage2_model.weights.h5')
            loinc_encoder.save_weights(checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")
    
    # Save the encoder model
    encoder_path = os.path.join(fold_output_dir, 'stage2_encoder')
    loinc_encoder.save_weights(encoder_path)
    
    print(f"Stage 2 fine-tuning completed for fold {fold_idx}. Encoder saved to {encoder_path}")
    
    return loinc_encoder

def plot_learning_curve(history, stage, save_path=None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Stage {stage} Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def train_stage1_new(model, loinc_df, epochs=30, batch_size=32, learning_rate=1e-4, checkpoint_dir='checkpoints'):
    """
    Stage 1 training loop - train on LOINC targets only
    
    Args:
        model: LOINCEncoder model
        loinc_df: DataFrame with LOINC targets
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        model: Trained model
        history: Training history
    """
    print("Starting Stage 1 training (LOINC targets only)")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    stage1_checkpoint_path = os.path.join(checkpoint_dir, "stage1_model.weights.h5")
    
    # Set up optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Number of batches per epoch (arbitrary)
    steps_per_epoch = 500
    
    # Training history
    history = {'train_loss': []}
    
    # Main training loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        
        # Progress bar for each epoch
        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for step in range(steps_per_epoch):
                # Prepare batch
                texts, labels = prepare_stage1_batch(loinc_df, batch_size)
                
                # Skip if batch is too small
                if len(texts) < 3:  # Need at least 3 samples for triplet loss
                    continue
                
                # Forward pass with gradient tape
                with tf.GradientTape() as tape:
                    # Get embeddings for batch
                    with tf.device('/CPU:0'):
                        embeddings = model(inputs=texts, training=True)
                    
                    # Calculate semi-hard triplet loss
                    loss = batch_semi_hard_triplet_loss(labels, embeddings, margin=0.8)
                
                # Backward pass (only update trainable variables - projection layer)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                # Update progress
                epoch_loss += loss.numpy()
                pbar.update(1)
                pbar.set_postfix({'loss': f"{loss.numpy():.4f}"})
        
        # End of epoch
        avg_epoch_loss = epoch_loss / steps_per_epoch
        history['train_loss'].append(avg_epoch_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f} - Time: {time.time() - start_time:.2f}s")
        
        # Save checkpoint if loss improved
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model.save_weights(stage1_checkpoint_path)
            print(f"Model saved at {stage1_checkpoint_path}")
    
    # Plot learning curve
    plot_learning_curve(history, stage=1, save_path=os.path.join(checkpoint_dir, "stage1_learning_curve.png"))
    
    # Load best weights
    model.load_weights(stage1_checkpoint_path)
    
    return model, history

def train_stage2_new(model, mimic_df, epochs=30, batch_size=32, learning_rate=1e-5, n_folds=5, checkpoint_dir='checkpoints'):
    """
    Stage 2 training loop - 5-fold cross-validation on MIMIC source-target pairs
    
    Args:
        model: LOINCEncoder model with Stage 1 weights
        mimic_df: DataFrame with MIMIC source-target pairs
        epochs: Number of epochs per fold
        batch_size: Batch size
        learning_rate: Learning rate
        n_folds: Number of cross-validation folds
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        models: List of trained models (one per fold)
        histories: List of training histories (one per fold)
    """
    print(f"Starting Stage 2 training ({n_folds}-fold cross-validation)")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store results for each fold
    models = []
    histories = []
    
    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(mimic_df)):
        print(f"Fold {fold+1}/{n_folds}")
        
        # Split data into train and validation sets
        train_df = mimic_df.iloc[train_idx]
        val_df = mimic_df.iloc[val_idx]
        
        # Reset model weights to Stage 1 best weights
        stage1_checkpoint_path = os.path.join(checkpoint_dir, "stage1_model.weights.h5")
        model.load_weights(stage1_checkpoint_path)
        
        # Add dropout layer for Stage 2 (if not already added)
        if not hasattr(model, 'dropout_added'):
            # Recreate model with dropout
            projection_weights = model.projection_layer.get_weights()
            
            # Create new model with dropout
            new_model = LOINCEncoder(dropout_rate=0.1)
            
            # Call the model once to build it
            _ = new_model(inputs=["dummy text"])
            
            # Set weights for projection layer
            new_model.projection_layer.set_weights(projection_weights)
            
            model = new_model
            model.dropout_added = True
        
        # Set up optimizer for this fold
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Checkpoint path for this fold
        fold_checkpoint_path = os.path.join(checkpoint_dir, f"stage2_fold{fold+1}_model.weights.h5")
        
        # Training history for this fold
        history = {'train_loss': [], 'val_loss': []}
        
        # Number of batches per epoch (arbitrary)
        steps_per_epoch = 100
        
        # Main training loop for this fold
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0
            start_time = time.time()
            
            # Training loop
            with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for step in range(steps_per_epoch):
                    # Prepare batch
                    texts, labels = prepare_stage2_batch(train_df, batch_size)
                    
                    # Skip if batch is too small
                    if len(texts) < 3:  # Need at least 3 samples for triplet loss
                        continue
                    
                    # Forward pass with gradient tape
                    with tf.GradientTape() as tape:
                        # Get embeddings for batch (with dropout active)
                        with tf.device('/CPU:0'):
                            embeddings = model(inputs=texts, training=True)
                        
                        # Calculate hard triplet loss
                        loss = batch_hard_triplet_loss(labels, embeddings, margin=0.8)
                    
                    # Backward pass (update only trainable variables)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                    # Update progress
                    epoch_loss += loss.numpy()
                    pbar.update(1)
                    pbar.set_postfix({'loss': f"{loss.numpy():.4f}"})
            
            # End of training epoch
            avg_train_loss = epoch_loss / steps_per_epoch
            history['train_loss'].append(avg_train_loss)
            
            # Validation step
            val_losses = []
            n_val_batches = max(1, min(len(val_df) // batch_size, 10))  # Limit validation batches
            
            for _ in range(n_val_batches):
                # Prepare validation batch
                val_texts, val_labels = prepare_stage2_batch(val_df, batch_size)
                
                if len(val_texts) < 3:  # Need at least 3 samples for triplet loss
                    continue
                
                # Forward pass (without dropout)
                with tf.device('/CPU:0'):
                    val_embeddings = model(inputs=val_texts, training=False)
                
                # Calculate validation loss (hard triplet loss)
                val_loss = batch_hard_triplet_loss(val_labels, val_embeddings, margin=0.8)
                val_losses.append(val_loss.numpy())
            
            # Average validation loss
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            history['val_loss'].append(avg_val_loss)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Time: {time.time() - start_time:.2f}s")
            
            # Save checkpoint if validation loss improved
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                model.save_weights(fold_checkpoint_path)
                print(f"Model saved at {fold_checkpoint_path}")
        
        # Plot learning curve for this fold
        plot_learning_curve(history, stage=f"2 (Fold {fold+1})", save_path=os.path.join(checkpoint_dir, f"stage2_fold{fold+1}_learning_curve.png"))
        
        # Load best weights for this fold
        model.load_weights(fold_checkpoint_path)
        
        # Store results for this fold
        models.append(model)
        histories.append(history)
    
    return models, histories

def prepare_stage1_batch(loinc_df, batch_size, augment=True, num_augmentations=5):
    """
    Prepare a batch for Stage 1 training (LOINC targets only)
    
    Args:
        loinc_df: DataFrame with LOINC targets
        batch_size: Batch size
        augment: Whether to apply augmentation
        num_augmentations: Number of augmentations per target
        
    Returns:
        texts: List of text strings
        labels: List of corresponding LOINC codes
    """
    # Sample random LOINC codes to form a batch
    sampled_indices = np.random.choice(len(loinc_df), size=batch_size // num_augmentations, replace=False)
    sampled_loinc = loinc_df.iloc[sampled_indices]
    
    texts = []
    labels = []
    
    for _, row in sampled_loinc.iterrows():
        # Combine different text representations (LCN, DN, SN) when available
        text_options = []
        
        for col in ['LONG_COMMON_NAME', 'SHORTNAME', 'DisplayName', 'RELATEDNAMES2']:
            if col in row and pd.notna(row[col]) and row[col]:
                text_options.append(str(row[col]).lower())
        
        if not text_options:
            continue
            
        loinc_code = row['LOINC_NUM']
        
        if augment:
            # Apply augmentation to each text option
            for text in text_options:
                augmented_texts = augment_text([text], num_variants=num_augmentations)
                texts.extend(augmented_texts)
                labels.extend([loinc_code] * len(augmented_texts))
        else:
            texts.extend(text_options)
            labels.extend([loinc_code] * len(text_options))
    
    # Ensure we don't exceed batch size
    if len(texts) > batch_size:
        texts = texts[:batch_size]
        labels = labels[:batch_size]
        
    return texts, labels

def prepare_stage2_batch(mimic_df, batch_size, augment=True, num_augmentations=5):
    """
    Prepare a batch for Stage 2 training (MIMIC source-target pairs)
    
    Args:
        mimic_df: DataFrame with MIMIC source-target pairs
        batch_size: Batch size
        augment: Whether to apply augmentation
        num_augmentations: Number of augmentations per source
        
    Returns:
        texts: List of text strings (sources and targets)
        labels: List of corresponding LOINC codes
    """
    # Sample random MIMIC pairs to form a batch
    sampled_indices = np.random.choice(len(mimic_df), size=batch_size // (2 * num_augmentations), replace=False)
    sampled_mimic = mimic_df.iloc[sampled_indices]
    
    texts = []
    labels = []
    
    for _, row in sampled_mimic.iterrows():
        source_text = str(row['SOURCE']).lower()
        loinc_code = row['LOINC_NUM']
        
        # Source text augmentation
        if augment:
            source_texts = augment_text([source_text], num_variants=num_augmentations)
        else:
            source_texts = [source_text]
            
        # Target text (just use as-is from the dataset)
        target_text = str(row['TARGET']).lower()
        
        # Target text augmentation (optional)
        if augment:
            target_texts = augment_text([target_text], num_variants=num_augmentations)
        else:
            target_texts = [target_text]
            
        # Add source texts
        texts.extend(source_texts)
        labels.extend([loinc_code] * len(source_texts))
        
        # Add target texts
        texts.extend(target_texts)
        labels.extend([loinc_code] * len(target_texts))
    
    # Ensure we don't exceed batch size
    if len(texts) > batch_size:
        texts = texts[:batch_size]
        labels = labels[:batch_size]
        
    return texts, labels

def main():
    parser = argparse.ArgumentParser(description='Train LOINC standardization model')
    parser.add_argument('--loinc_file', type=str, required=True, help='Path to processed LOINC targets CSV')
    parser.add_argument('--mimic_file', type=str, help='Path to processed MIMIC pairs CSV')
    parser.add_argument('--stage1_only', action='store_true', help='Run only Stage 1 training')
    parser.add_argument('--stage2_only', action='store_true', help='Run only Stage 2 training (requires Stage 1 checkpoint)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (memory-efficient)')
    parser.add_argument('--checkpoint_dir', type=str, default='models/checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with fewer steps/epochs')
    
    args = parser.parse_args()
    
    # Adjust parameters for test mode
    if args.test_mode:
        print("Running in test mode with reduced steps/epochs")
        epochs_stage1 = 2
        epochs_stage2 = 2
    else:
        epochs_stage1 = 30
        epochs_stage2 = 30
    
    # Load data
    loinc_df = None
    mimic_df = None
    
    if args.loinc_file:
        loinc_df = pd.read_csv(args.loinc_file)
        print(f"Loaded LOINC data: {len(loinc_df)} entries")
    
    if args.mimic_file:
        mimic_df = pd.read_csv(args.mimic_file)
        print(f"Loaded MIMIC data: {len(mimic_df)} entries")
    
    # Initialize model
    model = LOINCEncoder(dropout_rate=0.0)  # No dropout in Stage 1
    
    # Call the model once to build it
    _ = model(inputs=["dummy text"])
    print(f"Model initialized with {len(model.trainable_variables)} trainable variables")
    
    # Stage 1: Train on LOINC targets only
    if not args.stage2_only and loinc_df is not None:
        model, stage1_history = train_stage1_new(
            model, 
            loinc_df, 
            epochs=epochs_stage1,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir
        )
    
    # Stage 2: Train on MIMIC source-target pairs
    if not args.stage1_only and mimic_df is not None:
        # Check if Stage 1 checkpoint exists
        stage1_checkpoint_path = os.path.join(args.checkpoint_dir, "stage1_model.weights.h5")
        if not os.path.exists(stage1_checkpoint_path) and not args.stage2_only:
            print("Warning: Stage 1 checkpoint not found. Running Stage 1 first.")
            model, _ = train_stage1_new(
                model, 
                loinc_df, 
                epochs=epochs_stage1,
                batch_size=args.batch_size,
                checkpoint_dir=args.checkpoint_dir
            )
        elif args.stage2_only:
            # Load stage 1 weights
            try:
                model.load_weights(stage1_checkpoint_path)
                print(f"Loaded Stage 1 weights from {stage1_checkpoint_path}")
            except:
                print(f"ERROR: Could not load Stage 1 weights from {stage1_checkpoint_path}")
                print("Please run Stage 1 training first or provide a valid Stage 1 checkpoint.")
                return
        
        # Run Stage 2
        models, stage2_histories = train_stage2_new(
            model, 
            mimic_df, 
            epochs=epochs_stage2,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir
        )

if __name__ == '__main__':
    main() 