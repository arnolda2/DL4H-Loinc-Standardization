# LOINC Standardization Model

This repository implements the methodology described in the paper "Automated LOINC Standardization Using Pre-trained Large Language Models" for standardizing medical laboratory test descriptions to LOINC codes.

## Project Structure

- `models/`: Contains the model implementation and training code
  - `t5_encoder.py`: Implementation of the LOINCEncoder model with T5 backbone
  - `triplet_loss.py`: Implementation of the triplet loss functions
  - `train.py`: Two-stage training loop implementation
  - `evaluation.py`: Model evaluation code
  - `inference.py`: Inference code for making predictions
  - `error_analysis.py`: Error analysis for model predictions
  - `ablation_study.py`: Ablation studies for different model components
  - `checkpoints/`: Directory for saved model checkpoints

- `preprocessing/`: Data preprocessing and augmentation code
  - `data_augmentation.py`: Text augmentation techniques

- `output/`: Contains processed data files
  - `loinc_targets_processed.csv`: Processed LOINC targets for Stage 1 training
  - `mimic_pairs_processed.csv`: Processed MIMIC source-target pairs for Stage 2 training

- `run_model.sh`: Script to run training, evaluation, and inference
- `run_analysis.sh`: Script to run error analysis and ablation studies

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv 598_env
   source 598_env/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow tensorflow-hub tensorflow-text matplotlib tqdm scikit-learn pandas numpy seaborn
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

### Error Analysis and Ablation Studies

To run error analysis and ablation studies:

```bash
./run_analysis.sh
```

The error analysis identifies:
- Commonly misclassified LOINC codes
- Error patterns and categories (e.g., property mismatches, similar descriptions)
- Relationship between source text complexity and model performance

The ablation studies test the impact of:
- Two-stage fine-tuning approach
- Different mining strategies (hard vs semi-hard negative mining)
- Data augmentation techniques
- Model size and architecture

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

## Error Analysis

The error analysis categorizes prediction errors into several types:
- **Property Mismatch**: Misclassification between qualitative and quantitative properties
- **Specimen Mismatch**: Confusion between different specimen types (e.g., blood vs serum)
- **Methodological Differences**: Confusion between different measurement methods
- **Similar Description**: Text descriptions are very similar but represent different concepts
- **Ambiguous Source**: Source text lacks sufficient information for correct mapping

Error analysis results are stored in the `results/error_analysis` directory, including:
- CSV files with detailed error information for each sample
- Text summaries of error patterns
- Visualizations of error categories and source text complexity

## Ablation Studies

The ablation studies analyze the contribution of different components to model performance:

1. **Two-Stage Fine-Tuning**: Compares the full two-stage approach to using only Stage 2 training
2. **Mining Strategies**: Compares hard negative vs semi-hard negative mining
3. **Data Augmentation**: Measures the impact of data augmentation techniques
4. **Model Size**: Tests different T5 model sizes (base vs large)

Ablation results are stored in the `results/ablation_study` directory with comparative visualizations.

## Performance

The model is evaluated using the following metrics:

- Top-k accuracy (k=1, 3, 5, 10)
- Mean Reciprocal Rank (MRR)
- Error category distribution

## Extensions

### Hybrid Feature Integration for Qualitative vs Quantitative

This extension addresses a key limitation in the original model: the difficulty in distinguishing between qualitative and quantitative LOINC codes with similar descriptions (e.g., "Erythrocytes [#/volume] (Qn)" vs "Erythrocytes [Presence] (Ql)").

#### Implementation

1. **Feature Integration**: 
   - Uses the LOINC `SCALE_TYP` dimension (values: `Nom`, `Ord`, `Ql`, `Qn`, `Cnt`, etc.) to encode the qualitative/quantitative distinction
   - Appends a sentinel token to both source and target strings: `full_text = raw_text + " ##scale=" + scale_typ.lower() + "##"`
   - For sources without explicit scale information, uses the token `##scale=unk##`

2. **Training Pipeline Modifications**:
   - Updated data preprocessing to include `SCALE_TYP` information
   - Modified augmentation to preserve scale tokens
   - Updated triplet mining to use scale information

3. **Evaluation Methodology**:
   - Added stratified evaluation by `SCALE_TYP`
   - Implemented comparison of scale-confusable pairs
   - Included ablation with `##scale=unk##` to confirm the gain comes from the scale signal

#### Running the Extension

To test the hybrid feature integration:

```bash
./run_scale_integration.sh
```

This will:
1. Process LOINC data with scale information
2. Run evaluation using the scale-aware model
3. Generate a detailed report comparing performance with and without scale information
4. Save results in the `results/scale_integration` directory

#### Expected Impact

- Higher precision on scale-confusable pairs (~9% of mapping errors)
- Minimal compute overhead (uses the same model architecture)
- No schema changes to downstream retrieval

## References

- Original paper: "Automated LOINC Standardization Using Pre-trained Large Language Models" 