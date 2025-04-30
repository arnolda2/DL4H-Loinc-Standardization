# LOINC Standardization Model

This repository implements the methodology described in the paper "Automated LOINC Standardization Using Pre-trained Large Language Models" for standardizing medical laboratory test descriptions to LOINC codes.

## Project Structure

- `models/`: Contains the model implementation and training code
  - `t5_encoder.py`: Implementation of the LOINCEncoder model with T5 backbone
  - `triplet_loss.py`: Implementation of the triplet loss functions
  - `train.py`: Two-stage training loop implementation
  - `evaluation.py`: Model evaluation code
  - `inference.py`: Inference code for making predictions
  - `checkpoints/`: Directory for saved model checkpoints

- `preprocessing/`: Data preprocessing and augmentation code
  - `data_augmentation.py`: Text augmentation techniques

- `output/`: Contains processed data files
  - `loinc_targets_processed.csv`: Processed LOINC targets for Stage 1 training
  - `mimic_pairs_processed.csv`: Processed MIMIC source-target pairs for Stage 2 training

- `run_model.sh`: Script to run training, evaluation, and inference

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv 598_env
   source 598_env/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow tensorflow-hub tensorflow-text matplotlib tqdm scikit-learn pandas numpy
   ```

## Usage

The project implements a two-stage training process:

1. **Stage 1**: Fine-tune the projection layer using only LOINC target data
2. **Stage 2**: Further fine-tune using MIMIC source-target pairs with 5-fold cross-validation

### Training

To run the full training pipeline:

```bash
./run_model.sh train --loinc_file output/loinc_targets_processed.csv --mimic_file output/mimic_pairs_processed.csv
```

For Stage 1 training only:

```bash
./run_model.sh train --loinc_file output/loinc_targets_processed.csv --stage1_only
```

For Stage 2 training only (requires Stage 1 checkpoint):

```bash
./run_model.sh train --loinc_file output/loinc_targets_processed.csv --mimic_file output/mimic_pairs_processed.csv --stage2_only
```

For a quick test with fewer epochs:

```bash
./run_model.sh test
```

### Evaluation

To evaluate the model:

```bash
./run_model.sh evaluate --test_file output/mimic_pairs_processed.csv
```

To evaluate a specific fold:

```bash
./run_model.sh evaluate --test_file output/mimic_pairs_processed.csv --fold_idx 0
```

### Inference

To make predictions for a source text:

```bash
./run_model.sh predict "hemoglobin blood"
```

## Model Architecture

The model consists of:

1. A frozen Sentence-T5 encoder backbone
2. A trainable dense projection layer (128 dimensions)
3. L2 normalization

## Methodology

### Stage 1: Target-only Training

- Fine-tune the projection layer using only augmented LOINC target data
- Use semi-hard negative mining
- Learning rate: 1e-4
- Epochs: 30
- T5 backbone remains frozen

### Stage 2: Source-Target Training

- Further fine-tune using augmented MIMIC source-target pairs
- Use hard negative mining
- Learning rate: 1e-5
- 5-fold cross-validation
- Add dropout (0.1) before the projection layer for regularization
- T5 backbone remains frozen

## Performance

The model is evaluated using the following metrics:

- Top-k accuracy (k=1, 3, 5, 10)
- Mean Reciprocal Rank (MRR)

## References

- Original paper: "Automated LOINC Standardization Using Pre-trained Large Language Models" 