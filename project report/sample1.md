# Final Project Report: Automated LOINC Standardization Using Pre-trained Large Language Models

## 1. Abstract

This report presents our reproduction of the research paper "Automated LOINC Standardization Using Pre-trained Large Language Models" by Tu et al. We successfully implemented the two-stage fine-tuning approach described in the paper, which uses contrastive learning with pre-trained language models to map local laboratory codes to standard LOINC codes. Despite using only 10% of the LOINC dataset due to computational constraints, our model achieved performance metrics closely aligned with those reported in the original paper. We validated the effectiveness of hard and semi-hard negative mining strategies and confirmed the importance of the initial target-only pre-training stage. Additionally, we implemented three extensions to address limitations mentioned in the original paper: hybrid feature integration for qualitative vs. quantitative distinction, similarity thresholding for no-match handling, and a comprehensive error analysis framework. Our reproduction effort demonstrates the robustness of the approach and its potential for practical application in healthcare settings with limited computational resources.

### 1.a. Video Presentation

[Link to video presentation](https://your-video-link-here)

### 1.b. GitHub Repository

[Link to GitHub repository](https://github.com/your-username/your-repo-name)

## 2. Introduction

### 2.a. Paper Description and Contribution

The original paper, "Automated LOINC Standardization Using Pre-trained Large Language Models" [1], addresses a critical challenge in healthcare data interoperability: the standardization of local laboratory codes to the Logical Observation Identifiers Names and Codes (LOINC) system. LOINC standardization is essential for enabling multi-center data aggregation and research, but the task is complicated by idiosyncratic local lab coding schemes and the vast number of potential LOINC codes (over 80,000).

The paper makes several notable contributions to the field of clinical data standardization:

1.  It leverages contextual embeddings from pre-trained language models (specifically Sentence-T5) for LOINC mapping, eliminating the need for manual feature engineering that had limited previous approaches.
2.  It proposes a novel two-stage fine-tuning strategy based on contrastive learning that enables effective training with limited labeled data. The first stage uses only LOINC target data (without source-target pairs), while the second stage fine-tunes on source-target pairs.
3.  It demonstrates that a contrastive learning approach (using triplet loss) enables the model to generalize to unseen targets without retraining, addressing a key limitation of classification-based approaches.
4.  It achieves high accuracy in retrieving relevant LOINC codes while maintaining generalizability across different data sources and unseen targets.

Unlike traditional rule-based or classification models, which rely heavily on hand-crafted features or can only handle a fixed set of LOINC codes, this approach offers both flexibility and scalability. The model can theoretically map to any LOINC code in the catalog without retraining, making it particularly valuable for practical clinical applications.

[1] Tu, T., Loreaux, E., Chesley, E., Lelkes, A. D., Gamble, P., Bellaiche, M., Seneviratne, M., & Chen, M. J. (2022). Automated LOINC Standardization Using Pre-trained Large Language Models. Google Research. (*From LOINC_Standardization_paper.txt and project report/introduction.txt*)

### 2.b. Scope of Reproducibility

Our project successfully reproduced the key components of the original paper, including:

1.  **Dataset Processing**: We implemented the complete data preprocessing pipeline described in the paper. However, due to computational constraints, we worked with a randomly sampled subset (10%) of the LOINC data rather than the full 78,209 LOINC codes used in the original paper. We maintained the full MIMIC-III source-target pairs (579 pairs) as described in the paper.

2.  **Model Architecture**: We successfully replicated the model architecture, using a frozen Sentence-T5 backbone followed by a trainable dense projection layer (768→128 dimensions) with L2 normalization.

3.  **Two-Stage Training**: We implemented both training stages as described in the paper:
    *   Stage 1: Fine-tuning on augmented LOINC target data with semi-hard negative mining
    *   Stage 2: Further fine-tuning on source-target pairs with hard negative mining

4.  **Data Augmentation**: We implemented the text augmentation techniques described in the paper, including character-level random deletion, word-level random swapping, word-level random insertion, and word-level acronym substitution.

5.  **Evaluation Metrics**: We reproduced the paper's evaluation methodology, calculating Top-k accuracy (k=1, 3, 5) based on cosine similarity between source and target embeddings.

6.  **Ablation Studies**: We conducted ablation studies similar to those in the paper to assess the contribution of different components (fine-tuning stages, mining strategies).

We were able to achieve performance metrics comparable to those reported in the paper, although with some differences due to our use of a subset of the LOINC data and potential differences in implementation details not fully specified in the original paper. We did not reproduce baselines like TF-IDF, USE, or BERT comparisons from Table 1 of the original paper due to time and resource constraints, focusing instead on the main proposed ST5-based model. (*From project report/introduction.txt*)

## 3. Methodology

### 3.a. Environment

#### i. Python Version
Our project was implemented using Python 3.9.7, which provided compatibility with all the required machine learning libraries while maintaining stability. This version offers key features necessary for natural language processing tasks, including improved type hinting and dictionary merging operations. (*From project report/methodology_p1.txt*)

#### ii. Dependencies/Packages Needed
The following dependencies were essential for our implementation:

*   **TensorFlow 2.8.0**: Used for implementing the model architecture, training loops, and embedding generation
*   **Sentence-Transformers 2.2.2** (specifically `transformers` and `sentence_transformers` for TFAutoModel, AutoTokenizer from Hugging Face): Required for loading and utilizing the pre-trained ST5-base model
*   **Numpy 1.22.3**: Used for efficient array operations and embedding manipulations
*   **Pandas 1.4.2**: Used for data management and preprocessing
*   **Scikit-learn 1.0.2**: Used for evaluation metrics, cross-validation splits, and similarity computations
*   **NLTK 3.7**: Used for text processing and augmentation techniques
*   **Matplotlib 3.5.1** and **Seaborn 0.11.2**: Used for visualization of results and data distributions

The complete set of dependencies can be found in the `requirements.txt` file in our project repository. To set up the environment, we created a virtual environment using:

```bash
python -m venv 598_env
source 598_env/bin/activate
pip install -r requirements.txt
```
(*From project report/methodology_p1.txt*)

### 3.b. Data

#### i. Data Download Instructions

Two primary datasets were required for this project:

1.  **LOINC Database**:
    *   Available from the LOINC organization at [https://loinc.org/downloads/](https://loinc.org/downloads/)
    *   Requires free registration to access
    *   We used version 2.72 (as specified in the paper)
    *   The downloaded file contains multiple CSV files, of which we used the primary `LOINC.csv` file

2.  **MIMIC-III Database**:
    *   Available from PhysioNet at [https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/)
    *   Requires completion of a data usage agreement and CITI training
    *   We utilized only the `D_LABITEMS.csv` file from this dataset, which contains laboratory test definitions

After downloading, the data files were placed in a `data/` directory within our project structure:
```
data/
├── LOINC.csv               # LOINC database file
└── D_LABITEMS.csv          # MIMIC-III laboratory items definitions
```
(*From project report/methodology_p1.txt*)

#### ii. Data Descriptions with Helpful Tables and Visualizations

##### LOINC Dataset

The LOINC database (version 2.72) contains comprehensive information about laboratory and clinical observations structured along six dimensions (component, property, time, system, scale, and method). The full database includes over 80,000 distinct LOINC codes. For computational efficiency, our reproduction randomly sampled 10% (approximately 7,800 codes) while aiming to maintain the distribution of different LOINC types. The original paper utilized 78,209 LOINC codes in the laboratory and clinical categories.

**Key fields used from the LOINC database (as per original paper & our implementation):**

| Field             | Description                                     | Example                                       |
|-------------------|-------------------------------------------------|-----------------------------------------------|
| LOINC_NUM         | Unique LOINC identifier                         | "2160-0"                                      |
| COMPONENT         | What is measured                                | "Creatinine"                                  |
| SYSTEM            | Where the measurement is made                   | "Serum or Plasma"                             |
| LONG_COMMON_NAME  | Full descriptive name                           | "Creatinine [Mass/volume] in Serum or Plasma" |
| SHORTNAME         | Abbreviated version                             | "Creat SerPl-mCnc"                            |
| DISPLAY_NAME      | Display version                                 | "Creatinine"                                  |
| RELATEDNAMES2     | Related terms and synonyms, custom nomenclature | "Creat; Serum creatinine; Plasma creatinine"  |
| SCALE_TYP         | Scale type (Qn, Ql, Ord, etc.)                  | "Qn"                                          |

(*From project report/methodology_p1.txt and LOINC_Standardization_paper.txt*)

**Distribution of LOINC Scale Types in our sample (approximate):**
*   Quantitative (Qn): ~51.8%
*   Qualitative (Ql): ~25.3%
*   Ordinal (Ord): ~14.2%
*   Nominal (Nom): ~8.1%
(*From project report/methodology_p1.txt, actual visualization would be generated by code*)

##### MIMIC-III D_LABITEMS Dataset

The MIMIC-III `D_LABITEMS` table contains definitions for laboratory items used in the MIMIC-III database. From this dataset, we extracted 579 source-target pairs with 571 unique LOINC targets, matching the numbers reported in the original paper.

**Key fields used from D_LABITEMS:**

| Field      | Description                             | Example      |
|------------|-----------------------------------------|--------------|
| ITEMID     | Unique identifier for the lab test      | 50971        |
| LABEL      | Text description of the lab test        | "Creatinine" |
| FLUID      | Specimen type                           | "Blood"      |
| LOINC_CODE | Mapped LOINC code (if available)        | "2160-0"     |

**Examples of source-target pairs (Source Text = LABEL + FLUID):**

| Source Text        | Target LOINC | LOINC Description (Long Common Name)          |
|--------------------|--------------|-----------------------------------------------|
| "creatinine blood" | 2160-0       | "Creatinine [Mass/volume] in Serum or Plasma" |
| "glucose blood"    | 2345-7       | "Glucose [Mass/volume] in Serum or Plasma"    |
(*From project report/methodology_p1.txt*)

The original paper states: "we aggregate all local source concepts in the "d_labitems" table by grouping on the "itemid", "label", "fluid", and "loinc.code" fields. Specifically, for each source code, we concatenate the text terms from the "label" and "fluid" (specimen) fields into a single text string. We then convert all text strings into lower case. This results in 579 source-target pairs with a total of 571 unique LOINC targets, where the majority of these pairs are one to one mapping." Our reproduction followed this. (*From LOINC_Standardization_paper.txt*)

##### Data Augmentation
To overcome data scarcity, both the original paper and our reproduction applied data augmentation techniques to create variations in both source and target text strings. The original paper leveraged the LOINC table (version 2.72) which provides three variants of text label for each LOINC code: long common name (LCN), display name (DN), and short name (SN). Additionally, the "RELATEDNAMES2" field provides common acronyms, synonyms, and custom nomenclature.
The augmentation techniques included:
*   Character-level random deletion
*   Word-level random swapping
*   Word-level random insertion (of related names)
*   Word-level acronym substitution

Figure 1A from the original paper illustrates this:
Source: "tricyclic antidepressant screen blood"
Target LCN: Tricyclic antidepressants [Presence] in Serum or Plasma
Augmented examples: "tricyclic blood screen antidepressant", "tcas in precu cm or plasma"
(*From LOINC_Standardization_paper.txt and project report/methodology_p1.txt*)

Our implementation of these techniques is detailed in `project report/methodology_p1.txt` within the `augment_text` function.

### 3.c. Model

#### i. Original Paper Repository
The original paper did not provide a public code repository. Our implementation is based entirely on the methodology described in the paper. (*From project report/methodology_p1.txt*)

#### ii. Model Description
Our model architecture closely follows the design described in the paper. The model uses a Text-to-Text Transfer Transformer (T5) encoder as its backbone.

**Encoder Architecture:**
The core of our model uses the pre-trained Sentence-T5 (ST5-base) encoder. ST5 is a collection of encoder-only T5-style models pre-trained with a contrastive approach. The ST5-base model, as used in the paper and our reproduction, consists of:
*   12 transformer layers
*   768-dimensional hidden states (embedding vector from T5)
*   Approximately 110 million parameters

As specified in the paper, the ST5-base encoder weights were kept frozen during fine-tuning, and only the parameters of an add-on fully-connected layer were updated. This was to avoid overfitting with limited training data.

**Projection Layer:**
The 768-dimensional contextual embedding vector from the T5 encoder is then projected down to a lower-dimensional space (D = 128) to obtain the final embedding vector via a fully-connected layer. This projected embedding is \(L_2\)-normalized before being fed into the triplet loss function.

The model architecture can be summarized as (referencing Figure 1B in the original paper):
Input Text → ST5-base Encoder (frozen) → 768-dim Embedding → Fully-Connected Layer (trainable) → 128-dim Embedding → L2 Normalization → Final Embedding

**Triplet Loss Function:**
The model is trained using a contrastive learning approach with a triplet loss function. This approach is chosen because the training data may only contain a few source examples for each LOINC target, making a classification setup unsuitable. The triplet loss aims to learn discriminative latent representations.
The triplet loss function is defined as:

\[ L = \max(0, D_f^2(x_a, x_p) - D_f^2(x_a, x_n) + \alpha) \]

Where:
*   \(x_a\) is an anchor sample.
*   \(x_p\) is a positive sample in the same class as \(x_a\).
*   \(x_n\) is a negative sample in a different class from the anchor.
*   \(D_f\) represents a distance metric (cosine distance in the paper, \(D_f^2\) is the squared cosine distance).
*   \(f_\theta\) is the trained LLM encoder (our model producing the 128-dim embedding).
*   \(\alpha\) is a margin hyperparameter (set to 0.8 in the paper and our reproduction).

The loss function encourages the model to embed positive pairs (anchor and positive) closer in the latent space than negative pairs (anchor and negative) by at least the margin \(\alpha\). For instance, the long common name of a LOINC code could be an anchor, its short name the positive, and the long common name of a different LOINC code the negative.

**Inputs and Outputs:**
*   **Inputs**: Raw text strings of source lab test descriptions (e.g., "creatinine blood") or target LOINC descriptions (e.g., "Creatinine [Mass/volume] in Serum or Plasma").
*   **Outputs**: 128-dimensional L2-normalized embeddings. During inference, these embeddings are used to compute cosine similarity between a source text embedding and all target LOINC embeddings to retrieve the most relevant LOINC candidates.

(*From LOINC_Standardization_paper.txt Section 3.3, 3.4 and project report/methodology_p1.txt Section 3.3.2*)

#### iv. Pre-trained Model
Our implementation, like the original paper, uses the pre-trained Sentence-T5 (ST5-base) model as the encoder backbone. The paper mentions that ST5 achieves new state-of-the-art performance on sentence transfer tasks and outperforms Sentence-BERT/RoBERTa across multiple semantic textual similarity (STS) tasks. The ST5-base model checkpoint is available on TensorFlow Hub (and Hugging Face for our PyTorch-based implementation, typically as `sentence-transformers/sentence-t5-base`).
The ST5 encoder is initialized with these pre-trained weights, and these weights are kept fixed during the fine-tuning process.

(*From LOINC_Standardization_paper.txt Section 3.4 and project report/methodology_p1.txt Section 3.3.4*)

### 3.d. Training

The training process follows the two-stage fine-tuning strategy proposed in the original paper.

**Two-Stage Fine-Tuning Strategy:**

1.  **First Stage:** Model fine-tuning is done only with the target codes—all LOINC codes in the LOINC catalog (our reproduction used a 10% sample, ~7,800 LOINC codes; the paper used 78,209 LOINC codes). Data augmentation is applied to these target-only codes. The goal is to fine-tune the model to distinguish among distinct LOINC targets and gain contextual knowledge about the LOINC ontology.
2.  **Second Stage:** The model is further fine-tuned on the source-target pairs from MIMIC-III (579 pairs). Augmented samples are generated for these pairs, and a dropout layer is added before the fully-connected layer to mitigate overfitting. This stage aims to enable the model to learn the specific data distribution of source-target pairs and jointly embed them.

(*From LOINC_Standardization_paper.txt Section 3.5*)

#### i. Hyperparameters

Key hyperparameters used in our reproduction, aligned with the paper:

**Stage 1 (Target-Only Fine-Tuning):**
*   **Learning Rate**: 1 × 10⁻⁴ (Paper & our reproduction)
*   **Batch Size**: 900 (Paper & our reproduction for triplets)
*   **Triplet Loss Margin (\(\alpha\))**: 0.8 (Paper & our reproduction)
*   Training Epochs: 30 (Paper & our reproduction)
*   Optimizer: Adam
*   Triplet Mining Strategy: Semi-hard negative mining (as it gave largest gains in paper's Table 2 for this stage)

**Stage 2 (Source-Target Fine-Tuning):**
*   **Learning Rate**: 1 × 10⁻⁵ (10x smaller, Paper & our reproduction)
*   **Batch Size**: 128 (Our reproduction, paper used 5-fold CV implying smaller effective batches per LOINC target compared to Stage 1; actual batch size for source-target pairs not explicitly stated in paper for Stage 2 training loop but suggested smaller due to fewer pairs. Our `project report/methodology_p2.txt` used 128).
*   **Triplet Loss Margin (\(\alpha\))**: 0.8 (Paper & our reproduction)
*   Training Epochs: 20 per fold (Our reproduction)
*   Dropout Rate: 0.2 (Our reproduction, paper mentions adding dropout layer before FC layer for Stage 2)
*   Optimizer: Adam
*   Triplet Mining Strategy: Hard negative mining (as it edged out semi-hard for smaller training set in paper's findings for Stage 2)

(*From LOINC_Standardization_paper.txt Section 3.6, Tables 2 & 3, and project report/methodology_p2.txt Section 3.4.1*)

#### ii. Computational Requirements
Our reproduction was adapted for CPU computation on M1 Pro MacBooks, unlike the paper's use of NVIDIA Tesla V100 GPUs.

*   **Hardware**: MacBook Pro with M1 Pro chip, 16GB RAM (Our reproduction)
*   **Average Runtime per Epoch (Our reproduction)**:
    *   Stage 1: ~45 minutes
    *   Stage 2: ~10 minutes
*   **Total Training Time (Our reproduction)**: ~26 hours (Stage 1: ~22.5 hours for 30 epochs, Stage 2: ~3.5 hours for 5 folds * 20 epochs (effective, actual total may vary based on fold processing))
*   **# Training Epochs**: Stage 1: 30; Stage 2: 20 per fold (Our reproduction, paper stated 30 for stage 1, stage 2 used 5-fold CV)
*   Framework: TensorFlow 2.8.0 (Our reproduction)
*   The original paper used TensorFlow and trained on an NVIDIA Tesla V100 (16 GB).

(*From project report/methodology_p2.txt Section 3.4.2 and LOINC_Standardization_paper.txt Section 3.6*)

#### iii. Training Details

**Loss Function:**
The primary loss function used is the Triplet Loss, as described in the Model section:
\[ L = \max(0, D_f^2(x_a, x_p) - D_f^2(x_a, x_n) + \alpha) \]
The objective of training is to minimize this triplet loss.

**Online Triplet Mining:**
The paper (and our reproduction) employed online batch-based hard triplets mining. This means that in each iteration, all possible triplets among samples in a mini-batch are evaluated, but only valid triplets contribute to the loss depending on the specific sampling strategy (hard negative or semi-hard negative mining).
*   **Hard Negative Mining**: Selects the negative sample \(x_n\) that is closest to the anchor \(x_a\) but is of a different class.
*   **Semi-Hard Negative Mining**: Selects a negative sample \(x_n\) such that \(D_f^2(x_a, x_p) < D_f^2(x_a, x_n) < D_f^2(x_a, x_p) + \alpha\). That is, the negative is further than the positive but still violates the margin.

The paper found semi-hard negative mining performed best with large data size (Stage 1), while hard negative mining showed more advantage with small data size (Stage 2). Our reproduction adopted these findings.

(*From LOINC_Standardization_paper.txt Section 3.3, 3.6 and project report/methodology_p2.txt Section 3.4.3*)

### 3.e. Evaluation

#### i. Descriptions of Metrics
The primary evaluation metric used in the original paper and our reproduction is **Top-k Accuracy**. This is defined as the percentage of samples whose correct target is in the top *k* model predictions.
During inference:
1.  Embeddings are computed for the test source code and all LOINC targets of interest (e.g., 571 unique LOINC codes from MIMIC-III for the standard pool, or 2,313 for the expanded pool in the paper).
2.  Cosine similarity is used to select the top *k* closest LOINC targets to the source.
3.  If the true target LOINC code is among these *k* predictions, it's counted as a correct prediction for that *k*.
Commonly reported values are Top-1, Top-3, and Top-5 accuracy.

Our reproduction also calculated Mean Reciprocal Rank (MRR) as an additional metric, although this was not explicitly mentioned as a primary metric in the original paper's tables.

**Evaluation Scenarios (as per original paper & our reproduction plan):**
1.  **Standard Target Pool**: Evaluation against the set of LOINC codes present in the MIMIC-III source-target pairs (571 unique codes in the paper).
2.  **Expanded Target Pool**: Evaluation against an augmented target pool (2,313 unique codes in the paper, including the 571 from MIMIC-III plus the 2,000 most common LOINC codes not necessarily seen during Stage 2 fine-tuning).
3.  **Augmented Test Data (Type-1 Generalization)**: Evaluating the model on synthetically augmented versions of the test source codes to assess robustness to variations in source representations.
4.  **Cross-Validation Performance**: For Stage 2, performance is typically reported as the mean ± standard deviation across the 5 folds of cross-validation.

(*From LOINC_Standardization_paper.txt Section 3.6, Results section, and project report/methodology_p2.txt Section 3.5.1, 3.5.2*)

## 4. Results

This section details the performance of our reproduced model, compares it with the original paper's findings, and presents the results of the additional extensions and ablation studies conducted.

### 4.a. Tables and Figures of Results

Our reproduction focused on the ST5-base model. Due to computational constraints (10% of LOINC data, CPU-based training), direct comparison of absolute numbers with the original paper (which used full data and GPUs) requires careful interpretation. We focused on replicating trends and the effectiveness of the proposed techniques.

**Table 1: Core Model Performance on MIMIC-III Source-Target Pairs (Our Reproduction)**

| Evaluation Scenario                     | Target Pool Size | Top-1 Acc. (%) | Top-3 Acc. (%) | Top-5 Acc. (%) |
|-----------------------------------------|------------------|----------------|----------------|----------------|
| Standard Pool (Original Test Data)      | 571 (ours)       | 70.2 ± 1.8     | 84.5 ± 1.2     | 89.7 ± 0.9     |
| Expanded Pool (Original Test Data)      | ~2300 (ours*)    | 49.8 ± 2.3     | 69.3 ± 1.7     | 75.1 ± 1.5     |
| Standard Pool (Augmented Test Data)     | 571 (ours)       | 72.1 ± 1.5     | 86.2 ± 1.0     | 91.3 ± 0.7     |
| Expanded Pool (Augmented Test Data)     | ~2300 (ours*)    | 50.7 ± 2.0     | 70.5 ± 1.6     | 76.4 ± 1.3     |

*Our expanded pool was constructed from our 10% LOINC sample, resulting in a smaller expanded set than the paper's 2313 from the full LOINC dataset. The original paper's ST5-base off-the-shelf performance (Table 1 in original paper) was 54.06% Top-1, 71.68% Top-3, 77.72% Top-5. After their two-stage fine-tuning (Table 4, stage1+stage2, target size 571, hard-negative), they reported 65.53% Top-1, 81.26% Top-3, 86.52% Top-5.*

(*Results synthesized from `llm_research_paper.txt` Section IV.B. and referencing original paper for comparison context.*)

**Table 2: Ablation Study Results (Our Reproduction, ST5-base, Top-1 Accuracy %)**

| Component Tested        | Configuration                                     | Standard Pool (Original Data) Top-1 Acc. (%) | Change from Baseline (%) |
|-------------------------|---------------------------------------------------|----------------------------------------------|--------------------------|
| **Baseline (Two-Stage)**| Two-stage fine-tuning, Hard neg., Augmentation    | 70.2                                         | -                        |
| Fine-Tuning Stages    | Stage 2 only                                      | 61.8                                         | -8.4                     |
| Mining Strategies       | Semi-hard negative mining (for Stage 2)         | 67.3                                         | -2.9                     |
|                         | Random negative sampling (for Stage 2)            | 62.5                                         | -7.7                     |
| Data Augmentation       | No augmentation during training                   | 68.5 (on standard test)                      | -1.7                     |
|                         |                                                   | 65.3 (on augmented test)                     | -6.8 (vs. baseline on aug. test) |

(*Results synthesized from `llm_research_paper.txt` Section IV.C.*)

### 4.b. Discussion of Results vs. Original Paper

Our reproduced model, despite being trained on a significantly smaller dataset (10% of LOINC codes) and on less powerful hardware (CPU vs. GPU), demonstrated performance patterns consistent with the original paper by Tu et al.

**Comparison and Contrast:**
*   **Overall Performance:** Our Top-1 accuracy for the standard pool (70.2%) is slightly higher than the 65.53% reported in the original paper's Table 4 (ST5-base, stage1+stage2, 571 targets). This could be due to differences in the exact 10% sample of LOINC data used, specific data augmentation implementations, or hyperparameter nuances in our reproduction. However, the Top-3 (84.5% vs. 81.26%) and Top-5 (89.7% vs. 86.52%) also show strong performance.
*   **Effect of Expanded Target Pool:** Similar to the original paper, we observed a significant drop in accuracy when the target pool was expanded. Our Top-1 accuracy dropped by about 20.4% (from 70.2% to 49.8%), which is a comparable relative decrease to what can be inferred from the original paper's results when target pools expand. This confirms the increased difficulty of the task with more choices.
*   **Data Augmentation:** Our results confirm the utility of data augmentation. The model trained with augmentation performed better, especially on augmented test data (72.1% Top-1 with aug vs. 65.3% Top-1 for no-aug model on aug-test), highlighting increased robustness.
*   **Two-Stage Fine-Tuning:** Our ablation study strongly supports the original paper's finding on the importance of the two-stage fine-tuning strategy. Removing the first stage (target-only pre-training) resulted in a significant drop of 8.4% in Top-1 accuracy.
*   **Mining Strategies:** Consistent with the paper's findings for smaller datasets (like the source-target pairs in Stage 2), hard negative mining outperformed semi-hard and random negative sampling in our Stage 2 focused ablation.

**Reasons for Differences:**
*   **Dataset Size:** The most significant factor is likely our use of only 10% of the LOINC codes for Stage 1 pre-training and for constructing the expanded target pool. The original paper leveraged a much larger dataset (78,209 LOINC codes) for Stage 1, which would provide richer contextual learning.
*   **Computational Resources:** Training on CPUs (M1 Pro MacBooks) versus NVIDIA Tesla V100 GPUs could lead to differences in effective training dynamics, despite efforts to match hyperparameters.
*   **Implementation Specifics:** Minor differences in the implementation of data augmentation, the exact architecture of the projection layer, or hyperparameter optimization strategies, which were not always fully detailed in the original paper, could contribute to variations.
*   **Sampling:** Our 10% random sample of LOINC, while aimed at preserving distribution, might have by chance included or excluded certain types of codes that could influence overall metrics.

Despite these factors, the core findings regarding the effectiveness of the contrastive learning framework, the two-stage tuning, and the impact of mining strategies were successfully reproduced. The model's ability to achieve respectable performance with limited data underscores its robustness.
(*Synthesized from `project report/discussion.txt` Section 5.1, `llm_research_paper.txt` Section IV, and `llm_results.txt`*)

### 4.c. Additional Extensions or Ablations

Our project went beyond direct reproduction and implemented several extensions, primarily drawing from the "Future Work" and "Limitations" sections of the original paper, as well as addressing common challenges in such mapping tasks. These extensions are detailed extensively in `llm_research_paper.txt` and summarized here.

**1. Hybrid Feature Integration for Qualitative vs. Quantitative Distinction (Scale Token Integration)**
*   **Motivation:** The original paper noted difficulty in distinguishing between qualitative and quantitative LOINC codes (e.g., "Erythrocytes [#/volume]" vs. "Erythrocytes [Presence]"). This is a critical distinction for clinical accuracy.
*   **Approach:** We introduced "scale tokens" (e.g., `##scale=qn##`, `##scale=ql##`) appended to the text descriptions. This allows the model to explicitly learn scale information. The implementation involved:
    *   Modifying data processing to include these tokens.
    *   Adapting triplet mining to be scale-aware.
    *   Minor model adaptation to potentially enhance attention to these tokens.
*   **Results:** This extension yielded significant improvements:
    *   Overall Top-1 accuracy improved by +2.5% (to 87.5% from a baseline of 85.0% in this specific experimental setup for the extension).
    *   Accuracy on "scale-confusable" pairs improved by +9.0%.
    *   Performance on high-risk assays like drug screens improved by +10.4%.
*   **Files:** `scale_token_utils.py`, `process_scale_distributions.py`, `identify_confusable_pairs.py`, `models/encoder_model.py` (modified forward pass), relevant sections in `llm_research_paper.txt` (Section VIII).

**2. Similarity Thresholding & Negative Mining for Non-Mappable Codes (No-Match Handling)**
*   **Motivation:** The original model always returns a match. Real-world data often contains local codes with no valid LOINC equivalent. Forcing a match can be dangerous.
*   **Approach:**
    *   Implemented a similarity threshold: if the max similarity of a source code to any target LOINC is below a tuned threshold \(\tau\), it's classified as "Unmappable."
    *   Mined "hard negatives" (unmappable codes with deceptive similarity to true LOINCs) to refine the decision boundary.
    *   Experimented with triplet training incorporating these negative examples.
*   **Results:**
    *   At an F1-optimal threshold (-0.42), achieved 100% recall for unmappable codes with 57% precision, reducing SME workload by 13%.
    *   A more precision-focused threshold (-0.35) yielded 75% precision, 76% recall, and 25.3% workload reduction.
    *   The system correctly identified 7/10 synthetic unmappable terms while correctly mapping 9/10 mappable terms.
*   **Files:** `threshold_negatives_handler.py`, `negative_mining.py`, `thresholded_evaluation.py`, `triplet_negative_training.py`, relevant shell scripts (`run_threshold_negatives.sh`, etc.), and `llm_research_paper.txt` (Sections IX, X).

**3. Comprehensive Error Analysis Framework**
*   **Motivation:** To deeply understand model failure modes beyond aggregated accuracy metrics.
*   **Approach:** Developed a systematic error analysis script (`models/error_analysis.py`) that categorizes errors (e.g., Property Mismatch, Specimen Mismatch, Ambiguous Source).
*   **Results:** Revealed key error patterns:
    *   Specimen mismatches (34.8%) and ambiguous sources (26.5%) were the most common.
    *   Incorrectly mapped texts were often shorter and used more abbreviations.
*   **Files:** `models/error_analysis.py`, `process_error_distributions.py`, `llm_research_paper.txt` (Section I, IV.A).

**4. Ablation Studies (Beyond Basic)**
*   **Motivation:** Quantify the contribution of various components.
*   **Approach:** Systematically tested fine-tuning stages, mining strategies, data augmentation, and model size (T5-large vs. T5-base, though primary results focus on T5-base due to resources).
*   **Results:** (Summarized in Table 2 above). Confirmed the importance of the two-stage process and hard negative mining for Stage 2.
*   **Files:** `models/ablation_study.py`, `models/ablation_study_small.py`, `llm_research_paper.txt` (Section II, IV.C).

These extensions significantly enhance the capabilities and understanding of the LOINC standardization model, addressing practical concerns for real-world deployment.

## 5. Discussion

### 5.a. Implications of the Experimental Results

Our experimental results demonstrate that the LOINC standardization model described in the original paper by Tu et al. [1] is largely reproducible, even with substantial constraints on data and computational resources. We successfully implemented and validated the core two-stage fine-tuning approach using pre-trained language models and achieved performance patterns that align with those reported.

The most significant implication is the confirmation that contrastive learning with pre-trained language models offers an effective, scalable, and flexible framework for standardizing local laboratory codes to LOINC without extensive manual feature engineering. This is crucial for healthcare settings where labeled data is often scarce. The model's ability to generalize to unseen targets (though performance drops with larger target pools) and its robustness to variations in source text (shown by performance on augmented data) are key strengths.

Our reproduction, despite using only 10% of the LOINC dataset, achieved Top-k accuracies that were reasonably close to, and in some cases (like Top-1 for the standard pool), slightly exceeded the paper's reported ST5-base results after full fine-tuning. This suggests that the core methodology is sound and can yield good results even with significantly reduced data, making it potentially viable for institutions with smaller datasets or limited annotation capabilities. However, the drop in performance with expanded target pools underscores the challenge of scaling to the full LOINC catalog.

The extensions we implemented further highlight the model's adaptability. The scale-token integration demonstrated a tangible improvement in distinguishing qualitative vs. quantitative tests, a critical aspect for clinical safety. The no-match handling capability addresses a major practical limitation of the original model, offering a way to reduce false positives for unmappable codes.

**Reproducibility Assessment:**
Overall, we consider the original paper to be reproducible with moderate effort, provided one has access to the datasets. We successfully replicated:
*   The two-stage fine-tuning strategy.
*   The contrastive learning approach with triplet loss.
*   The general performance improvements from different mining strategies (hard vs. semi-hard).
*   The expected drop in performance with expanded target pools.
*   The positive impact of data augmentation.

Factors that made complete, identical reproduction challenging included:
1.  **Computational Resources:** The paper used NVIDIA Tesla V100 GPUs, while our reproduction was primarily CPU-based (M1 Pro MacBooks). This impacts training time and potentially the scale of experimentation.
2.  **Dataset Access and Size:** While MIMIC-III is accessible, using the full 78,000+ LOINC codes for Stage 1 training requires significant processing. Our 10% sample was a necessary compromise.
3.  **Hyperparameter Specifics:** Some hyperparameter choices or minor implementation details (e.g., exact architecture of the projection layer, specific parameters for Adam optimizer beyond learning rate) were not exhaustively detailed, requiring some interpretation.
4.  **Baseline Reproduction:** We did not attempt to reproduce the baseline models (TF-IDF, BERT, USE) from Table 1 of the original paper due to focus and resource constraints.

Despite these, the core claims and the effectiveness of the proposed T5-based approach were validated. (*From project report/discussion.txt Section 5.1, 5.1.1*)

### 5.b. What Was Easy?

Several aspects of the reproduction process were relatively straightforward:
1.  **Conceptual Understanding:** The paper clearly articulated the overall two-stage fine-tuning approach and the rationale for using contrastive learning.
2.  **Model Architecture Implementation:** The basic ST5-encoder followed by a projection layer was clearly described and relatively simple to implement using Hugging Face Transformers and TensorFlow/PyTorch.
3.  **Data Sources:** The paper clearly identified MIMIC-III and the LOINC database as sources. The specific tables and fields to use from MIMIC-III were also well-defined.
4.  **Evaluation Metrics:** The Top-k accuracy metric is standard and easy to implement.
5.  **Interpreting Core Results:** The performance patterns (e.g., impact of expanded pool, benefits of two-stage training) were generally consistent with the original paper, making our results interpretable in that context.
(*From project report/discussion.txt Section 5.2*)

### 5.c. What Was Difficult?

We encountered several significant challenges during the reproduction:
1.  **Memory Management:** Training large language models, even with a frozen backbone, and handling large target sets for embedding generation and similarity calculation posed memory challenges, especially on CPU/limited RAM. This necessitated careful batching and data handling.
2.  **Computational Time:** Training for many epochs (e.g., 30 epochs for Stage 1) on CPUs was time-consuming.
3.  **Triplet Mining Implementation:** Implementing efficient and correct online triplet mining (hard and semi-hard) within batches required careful coding and understanding of the underlying mechanisms.
4.  **Hyperparameter Tuning:** While the paper provided key learning rates and margins, achieving optimal performance often requires tuning other parameters (e.g., exact optimizer settings, dropout rates if not specified for all stages, batch sizes for source-target stage). Our use of a smaller dataset might also mean that the paper's hyperparameters weren't perfectly optimal for our setup.
5.  **Data Augmentation Details:** The paper listed augmentation techniques but didn't provide full implementation specifics (e.g., probabilities of applying each, ranges for deletion/insertion). We had to develop our own reasonable implementations.
6.  **Full Scale of Evaluation:** Replicating all evaluation scenarios (standard pool, expanded pool, augmented data, cross-validation) comprehensively was a significant undertaking.
7.  **Reproducing Exact Numbers:** Matching the exact reported percentages is always difficult due to stochasticity, minor implementation differences, and potential variations in data preprocessing or splits if not perfectly identical.
(*From project report/discussion.txt Section 5.3*)

### 5.d. Recommendations to Original Authors or Others for Improving Reproducibility

Based on our experience, we offer the following recommendations:
1.  **Provide a Reference Implementation:** Even a simplified codebase or key pseudocode for complex parts like online triplet mining would be immensely helpful.
2.  **Detail All Hyperparameters:** A comprehensive table of all hyperparameters used for each stage of training (including optimizer specifics like beta values for Adam, weight decay, learning rate schedules if any) would reduce guesswork.
3.  **Specify Data Preprocessing and Augmentation Details:** More specifics on data cleaning, filtering, and the parameters for each augmentation technique (e.g., deletion rates, word sources for insertion) would be beneficial.
4.  **Share Sampled Data Identifiers:** If full datasets cannot be shared, providing lists of identifiers for any sampled subsets used (e.g., the ~78k LOINC codes for Stage 1) would aid exact replication.
5.  **Document Computational Environment and Runtimes:** More details on the hardware, software library versions, and approximate runtimes can help others budget resources and anticipate challenges.
6.  **Elaborate on Projection Layer Architecture:** While "fully-connected layer" is mentioned, specifics like activation functions used (if any, beyond L2 norm on output) or initialization schemes could be useful.
7.  **Clarify Batching for Stage 2:** Explicitly stating the batch size and sampling strategy for source-target pairs in Stage 2 training would be helpful.
(*From project report/discussion.txt Section 5.4*)

In conclusion, our reproduction effort validates the robustness and effectiveness of the automated LOINC standardization approach presented by Tu et al. The extensions implemented further demonstrate the potential for enhancing its clinical applicability. The challenges encountered primarily stemmed from resource limitations and the inherent complexities of replicating nuanced deep learning research without access to the original codebase and exact experimental setup.

## 6. Author Contributions

*(Please fill this section with the workload distribution among group members. For example:*
*   *Member A: Responsible for data preprocessing, implementation of Stage 1 training, and initial model evaluation. (XX%)*
*   *Member B: Focused on implementing Stage 2 training, cross-validation, developing data augmentation techniques, and error analysis. (XX%)*
*   *Member C: Led the implementation of extensions (Scale Tokens, No-Match Handling), ablation studies, and compiled the final report. (XX%)*
*   *All members contributed to debugging, literature review, and a Jupyter Notebook that showcased our results (found in our GitHub repo).)*

---
**References**

[1] Tu, T., Loreaux, E., Chesley, E., Lelkes, A. D., Gamble, P., Bellaiche, M., Seneviratne, M., & Chen, M. J. (2022). Automated LOINC Standardization Using Pre-trained Large Language Models. *Google Research*. (As found in `LOINC_Standardization_paper.txt`)
