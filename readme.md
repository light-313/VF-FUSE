# VF-FUSE: Dual-Pathway Fusion Framework for Bacterial Virulence Factor Prediction

## Project Overview

VF-FUSE (Virulence Factor prediction with FUSion of Embeddings) is a comprehensive computational framework designed for bacterial virulence factor identification. This project innovatively combines protein embeddings derived from pre-trained language models with traditional sequence features, constructing a dual-pathway fusion architecture that achieves high-precision recognition of bacterial virulence factors.

The core advantages of VF-FUSE include:

- Integration of state-of-the-art protein language models (ESM2, ProtT5) representation capabilities
- Preservation of key biological information captured by traditional sequence features (AAC, DPC, DDE, etc.)
- Design of a specialized dual-pathway neural network architecture optimizing feature fusion processes
- Implementation of multiple model ensemble strategies to further enhance prediction performance

## Project Structure

```

VF-FUSE/
├── best/                         # Best models and configurations
│   ├── esm2_best.json            # ESM2 best hyperparameter configuration
│   ├── esm2_best.pth             # ESM2 best model weights
│   ├── fusion_best_config.json   # Fusion model best configuration
│   ├── fusion_fine_tuned.pth     # Fine-tuned fusion model weights
│   ├── prot5_best_config.json    # ProtT5 best configuration
│   └── prot5_best_model.pth      # ProtT5 best model weights
│
├── raw_data/                     # Data files
│   ├── test_seqsim_features.csv  # Test set sequence similarity features
│   ├── test.fasta                # Test set sequence data
│   ├── train_ba.fasta            # Balanced training set
│   └── [Other data files]        # Various feature files and intermediate data
│
├── Feature Extraction Module
│   ├── get_esm2_embedding.py     # Extract protein embeddings using ESM2
│   ├── get_prot5.py              # Extract protein embeddings using ProtT5
│   └── trad_feature_extraction.py# Traditional feature extraction tool
│
├── Model Training and Evaluation
│   ├── plm_train.py              # Pre-trained language model fine-tuning
│   ├── trad_feature_train.py     # Traditional feature model training
│   ├── plm_tune.py               # Hyperparameter optimization
│   ├── plm_val_model.py          # Model validation
│
├── Core Architecture Definition
│   ├── model_type.py             # Various model architecture definitions
│   └── Ensemble_model.py         # Ensemble model implementation
|——vf_streamlit_app.py   # Prediction of virulence factors app
|
|
|——test_data # The ESM2 and PROT5 features of the independent test set can be used to test the model.
```

## Detailed Function of Each File

- **get_esm2_embedding.py**: Extracts ESM2 embedding vectors from input FASTA sequences, including sequence representation and representation for each amino acid position.
- **get_prot5.py**: Uses the ProtT5 pre-trained model to extract contextualized embedding representations of protein sequences.
- **trad_feature_extraction.py**: Calculates various traditional protein sequence features such as AAC, DPC, PAAC, etc.
- **model_type.py**: Defines multiple neural network architectures, including dual-pathway fusion, single-pathway fusion, and other variants.
- **plm_train.py**: Pre-trained language model training and fine-tuning process, including data loading, model initialization, training loops, and evaluation.
- **plm_tune.py**: Implements hyperparameter optimization using Ray Tune to search for the best model configuration.
- **Ensemble_model.py**: Implements various ensemble strategies to combine predictions from different models to improve overall performance.

Installation Guide

### Requirements

- Python 3.9+
- CUDA 11.7+ (for GPU acceleration, recommended)
- 16GB+ RAM
- 8GB+ GPU memory (for ESM2 and ProtT5 models)

### Dependency Installation

```bash
# Create virtual environment
conda create -n vf-fuse python=3.9
conda activate vf-fuse

pip install -r requirements.txt
```

## Usage Guide

### 1. Data Preparation

Prepare protein sequence files in FASTA format, with the required format:

```
>ID|label
SEQUENCE
```

where label is 1 (virulence factor) or 0 (non-virulence factor).

### 2. Feature Extraction

```bash
# Extract ESM2 features
python get_esm2_embedding.py --input train_ba.fasta --output features/esm2_train.h5 --batch_size 8

# Extract ProtT5 features
python get_prot5.py --input train_ba.fasta --output features/prot5_train.h5 --batch_size 4

```

### 3. Launch the  streamlit  app

```
streamlit run  VF-FUSE\vf_streamlit_app.py
```
