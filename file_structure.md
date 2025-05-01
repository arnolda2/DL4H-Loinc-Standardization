# LOINC Standardization Model Project Structure

This document provides a comprehensive explanation of the project file structure, detailing what each file contributes to the overall LOINC standardization system.

## Core Model Architecture

### `models/` Directory

- **`encoder_model.py`**: 
  - Implements the core T5-based embedding model 
  - Defines the neural network architecture for encoding lab test descriptions
  - Contains forward pass modifications for scale token enhancement
  - Implements normalized embedding generation for similarity calculations
  - Handles model saving and loading functionality

- **`data_loader.py`**: 
  - Provides data loading functionality for both training and evaluation
  - Implements preprocessing pipelines for source and target texts
  - Handles scale token integration in the data pipeline
  - Creates TripletDataset and SourceTargetDataset classes for training
  - Manages batching and data augmentation during loading

- **`tokenizer_extension.py`**: 
  - Extends the base T5 tokenizer with domain-specific tokens
  - Optimizes tokenizer for scale tokens and medical terminology
  - Implements special token handling for the model
  - Provides functions to manage sequence length constraints
  - Handles priority-based truncation when descriptions exceed token limits

- **`triplet_mining.py`**: 
  - Implements multiple negative mining strategies:
    - Hard negative mining (finding similar but incorrect LOINC codes)
    - Semi-hard negative mining (moderately difficult negative examples)
    - Scale-aware negative mining (respecting scale types)
  - Creates triplets (anchor, positive, negative) for contrastive learning
  - Balances triplet generation across different LOINC categories
  - Implements the ScaleAwareTripletMiner class for scale-sensitive mining

- **`training_pipeline.py`**: 
  - Orchestrates the two-stage training process:
    - Stage 1: Target-only triplet learning with encoder frozen
    - Stage 2: Source-target mapping with encoder unfrozen
  - Manages training hyperparameters and optimizer configuration
  - Implements learning rate scheduling and early stopping
  - Handles checkpoint saving and validation during training
  - Coordinates scale token integration during the training process

## Evaluation Framework

- **`run_evaluation.py`**: 
  - Primary evaluation controller managing different scenarios:
    - Standard test data evaluation
    - Expanded target pool evaluation
    - Augmented test data evaluation (Type-1 generalization)
  - Handles timeout management to prevent hanging evaluations
  - Implements batch processing to manage memory usage
  - Produces structured output files with evaluation metrics
  - Supports cross-validation evaluation across multiple folds

- **`run_controlled_evaluation.py`**: 
  - Specialized evaluation script for memory-constrained environments
  - Features configurable test size limitation
  - Implements timeout handling for each evaluation component
  - Provides step-by-step execution monitoring
  - Implements graceful failure handling for individual components
  - Generates partial results when full evaluation isn't possible

- **`run_full_evaluation.py`**: 
  - Comprehensive evaluation pipeline executing all components
  - Coordinates full evaluation across all cross-validation folds
  - Triggers error analysis on incorrectly classified samples
  - Initiates ablation studies for component contribution analysis
  - Generates summary reports and visualizations
  - Creates comprehensive performance profile across test conditions

- **`evaluation_summary.py`**: 
  - Aggregates results from all evaluation components
  - Computes summary statistics across evaluation scenarios
  - Generates comparative visualizations for different test conditions
  - Produces formatted reports highlighting key findings
  - Exports structured data for further analysis or documentation
  - Creates performance comparison charts for different model configurations

## Error Analysis Implementation

- **`models/error_analysis.py`**: 
  - Implements systematic evaluation of model predictions
  - Categorizes errors into meaningful groups:
    - Property Mismatch (qualitative vs. quantitative)
    - Specimen Mismatch (different specimen types)
    - Methodological Differences (measurement methods)
    - Similar Description (textually similar but different concepts)
    - Ambiguous Source (insufficient information)
    - Completely Different (unrelated predictions)
  - Discovers common error patterns and frequently confused LOINC codes
  - Examines relationship between source text complexity and accuracy
  - Generates visualizations to aid in understanding error patterns
  - Creates detailed CSV files with per-sample error information

- **`process_error_distributions.py`**: 
  - Analyzes error distribution across different categories
  - Generates charts for error category visualization
  - Identifies the most common error patterns
  - Creates summary statistics for error analysis reporting
  - Examines institution-specific error patterns
  - Links error patterns to model architectural decisions

## Ablation Studies

- **`models/ablation_study.py`**: 
  - Quantifies contribution of different components to performance
  - Tests individual components by selectively removing/modifying them:
    - Fine-Tuning Stages (two-stage vs. single-stage)
    - Mining Strategies (hard negative vs. semi-hard vs. random)
    - Data Augmentation impact
    - Model Size effect (base vs. large)
  - Measures performance differences when components are altered
  - Calculates absolute and relative improvements by component
  - Generates comparative charts to illustrate component impacts
  - Creates comprehensive summary reports of component contributions

- **`models/ablation_study_small.py`**: 
  - Optimized version for faster experimentation on large datasets
  - Reduces sample size while maintaining representativeness
  - Simplifies component testing to focus on key architectural decisions
  - Manages memory usage through batch processing of embeddings
  - Produces visualizations that clearly illustrate component contributions
  - Generates detailed summaries to inform design decisions

## Scale Token Integration

- **`scale_token_utils.py`**: 
  - Provides utilities for handling scale type tokens:
    - `append_scale_token`: Adds scale sentinel token to text
    - `extract_scale_token`: Extracts scale information from tokenized text
    - `strip_scale_token`: Removes scale token from text
  - Implements format standardization for scale information
  - Handles token protection during text augmentation
  - Manages scale token positioning in text
  - Provides backward compatibility with non-scale-aware models

- **`process_scale_distributions.py`**: 
  - Analyzes distribution of scale types in LOINC dataset:
    - Quantitative (Qn): 52.3%
    - Qualitative (Ql): 24.7%
    - Ordinal (Ord): 14.1%
    - Nominal (Nom): 8.2%
    - Count (Cnt): 0.7%
  - Generates visualizations of scale type distributions
  - Identifies patterns in scale type usage
  - Provides insights for scale token integration strategies
  - Analyzes scale distribution in error cases

- **`identify_confusable_pairs.py`**: 
  - Identifies components that exist in multiple scale types
  - Creates datasets of scale-confusable LOINC pairs
  - Finds components with similar descriptions but different scales
  - Quantifies the prevalence of scale confusion in the data
  - Provides tools for targeted evaluation of scale confusion cases
  - Generates reports on the 3,784 "scale-confusable" components

- **`scale_inference.py`**: 
  - Implements pattern-based scale type inference from text
  - Uses rule-based approach to detect scale indicators:
    - Quantitative indicators: "count", "concentration", etc.
    - Qualitative indicators: "presence", "pos/neg", etc.
    - Ordinal indicators: "grade", "stage", "level", etc.
  - Handles ambiguous cases with confidence scoring
  - Provides scale prediction for sources lacking explicit scale information
  - Implements context analysis for improved scale inference accuracy

## No-Match Handling

- **`threshold_negatives_handler.py`**: 
  - Core implementation for non-mappable code detection
  - Implements similarity thresholding for unmappable detection
  - Provides functions for finding optimal thresholds via precision-recall
  - Generates hard negative examples through similarity-based mining
  - Performs inference with unmappable detection capabilities
  - Implements confidence calibration for threshold adjustment

- **`negative_mining.py`**: 
  - Loads non-mappable codes from reference datasets
  - Identifies hard negative examples for training
  - Creates negative examples with similar components but different specimens
  - Implements functions for threshold calculation
  - Generates training data for unmappable detection
  - Creates visualizations of negative example distributions

- **`stratified_evaluation.py`**: 
  - Evaluates model performance stratified by different criteria:
    - Scale type (Qn, Ql, Ord, etc.)
    - Mappability (mappable vs. unmappable)
    - Test frequency (common vs. rare)
    - Specimen type (blood, serum, urine, etc.)
  - Provides comprehensive performance breakdown by category
  - Identifies performance gaps between different test types
  - Generates reports on category-specific performance
  - Creates visualizations for stratified performance analysis

- **`thresholded_evaluation.py`**: 
  - Implements evaluation with similarity thresholds
  - Calculates precision, recall, and F1 score for mappable classification
  - Measures SME workload reduction through unmappable detection
  - Calculates top-k accuracy metrics for correctly identified mappable codes
  - Provides threshold optimization capabilities
  - Generates reports on threshold performance at different operating points

## Model Training Extensions

- **`triplet_negative_training.py`**: 
  - Implements triplet training with negative examples
  - Creates TripletModel class for training with negative examples
  - Defines triplet loss function with configurable margin
  - Manages training with unmappable examples
  - Handles batch processing for efficient training
  - Saves encoder weights after training completion

## Shell Scripts

- **`run_threshold_negatives.sh`**: 
  - Controls threshold-based evaluation workflow
  - Supports three main modes:
    - `tune`: Find optimal similarity threshold on development set
    - `generate`: Produce hard negative examples
    - `evaluate`: Apply threshold-based detection to test set
  - Manages environment variables and dependencies
  - Handles output logging and result collection
  - Provides command-line interface for threshold experiments

- **`run_nomatch_integration.sh`**: 
  - Integrates no-match handling into production workflow
  - Coordinates data preprocessing for unmappable detection
  - Sets up environment for nomatch integration
  - Manages file paths and dependencies
  - Handles result processing and output generation
  - Provides integration with broader LOINC mapping pipeline

- **`run_triplet_training.sh`**: 
  - Executes the triplet training pipeline with negative examples
  - Sets up training data and model configuration
  - Manages hyperparameters for triplet training
  - Handles checkpoint saving and loading
  - Coordinates GPU resource allocation
  - Generates training logs and progress reports

- **`run_trained_evaluation.sh`**: 
  - Evaluates models after triplet training
  - Loads trained models and test data
  - Calculates performance metrics
  - Generates evaluation reports
  - Compares performance with baseline models
  - Creates visualizations of model improvements

- **`run_thresholded_evaluation.sh`**: 
  - Tests original model with thresholding
  - Applies different threshold values
  - Generates performance metrics at each threshold
  - Creates precision-recall curves
  - Identifies optimal threshold values
  - Produces reports on threshold performance

## Data Processing

- **`process_loinc.py`**: 
  - Preprocesses LOINC database files
  - Extracts relevant columns and fields
  - Creates processed text columns with scale tokens
  - Handles data cleaning and normalization
  - Manages LOINC version compatibility
  - Generates processed output files for model training

- **`confidence_calibration.py`**: 
  - Implements confidence estimation for predictions
  - Calculates confidence in scale type prediction
  - Combines multiple confidence signals:
    - Source text confidence
    - Prediction agreement confidence
    - Scale consistency confidence
  - Provides calibrated confidence scores
  - Identifies edge cases requiring human review
  - Generates confidence reports for prediction evaluation

## High-Risk Assay Analysis

- **`high_risk_evaluation.py`**: 
  - Evaluates model on clinically significant high-risk assays:
    - Blood cultures
    - Drug screens
    - Hormone tests
  - Identifies high-risk tests using pattern matching
  - Calculates performance metrics for each risk category
  - Generates detailed reports on high-risk test performance
  - Provides targeted analysis of safety-critical test mapping
  - Creates visualizations highlighting performance on high-risk assays

## Documentation

- **`llm_research_paper.txt`**: 
  - Comprehensive research paper describing:
    - LOINC standardization model architecture
    - Error analysis methodology and findings
    - Ablation study design and results
    - Scale token integration approach
    - No-match handling extension
    - Performance analysis and recommendations
  - Provides detailed experimental setup and results
  - Discusses limitations and future work
  - Documents technical challenges and solutions
  - Offers recommendations for model deployment and use

- **`README.md`**: 
  - Project overview and mission statement
  - Setup instructions and environment requirements
  - Usage examples and quick start guide
  - Architecture overview diagram
  - Performance summary and key features
  - Contribution guidelines and licensing information
  - References and acknowledgments

## Visualization Tools

- **`visualization/confusion_matrix.py`**: 
  - Generates confusion matrices for error pattern visualization
  - Implements different normalization options
  - Creates heatmaps of confusion patterns
  - Provides insights into systematic error patterns
  - Generates exportable visualization files
  - Supports interactive visualization in notebooks

- **`visualization/similarity_distribution.py`**: 
  - Creates visualizations of similarity distributions
  - Compares mappable vs. unmappable similarity profiles
  - Generates histograms and kernel density plots
  - Visualizes threshold cutoffs and decision boundaries
  - Provides tools for threshold selection
  - Creates exportable visualization files for reporting

- **`visualization/component_impact.py`**: 
  - Visualizes impact of different architectural components
  - Creates comparative bar charts for component performance
  - Generates relative improvement visualizations
  - Shows confidence intervals for component contributions
  - Provides tools for visualizing ablation study results
  - Creates exportable visualization files for presentations

## Utilities and Support

- **`utils/timeout_handler.py`**: 
  - Implements timeout monitoring for long-running evaluations
  - Prevents resource exhaustion due to hanging processes
  - Provides graceful termination of timed-out operations
  - Generates partial results when full execution times out
  - Implements configurable timeout settings
  - Logs timeout events for troubleshooting

- **`utils/memory_management.py`**: 
  - Implements dynamic resource allocation based on system capabilities
  - Provides batch processing tools for memory-efficient operation
  - Monitors memory usage during execution
  - Implements fallback strategies for memory-constrained environments
  - Provides tools for efficient embedding computation
  - Manages temporary file cleanup during processing 