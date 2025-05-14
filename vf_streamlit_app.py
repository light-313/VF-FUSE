import os
import numpy as np
import pandas as pd
import torch
import streamlit as st
import json
import h5py
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from Bio import SeqIO
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import warnings
from typing import List, Dict, Tuple, Any
import matplotlib.cm as cm
import altair as alt

# Import needed modules from your existing code
from model_type import DualPathwayFusion

# Silence warnings
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set page configuration
st.set_page_config(
    page_title="VF-FUSE: Virulence Factor Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
def add_custom_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: #4b6cb7;
        }
        .result-card {
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: white;
            margin-bottom: 1rem;
        }
        .stProgress > div > div > div > div {
            background-color: #4b6cb7;
        }
        .stTextInput > div > div > input {
            border-radius: 5px;
        }
        .stButton > button {
            border-radius: 5px;
            background-color: #4b6cb7;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #888;
            font-size: 0.8rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

@st.cache_resource
def get_device():
    """Get the appropriate device for computation"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_data
def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        return json.load(f)

@st.cache_resource
def load_model(model_path, model_type, config, esm_dim, prot5_dim, device, feature_type):
    """Load model with caching for better performance"""
    print(f"Loading model: {model_path} (Type: {model_type}, Feature: {feature_type})")
    
    try:
        if "dual" in model_type.lower():
            model = DualPathwayFusion(
                esm_dim=esm_dim,
                prot5_dim=prot5_dim,
                hidden_dim=128,
                num_layers=4,
                num_classes=2,
                rank=8,
                steps=1,
                dropout=config["dropout"]
            )
        else:
            if feature_type == "esm2":
                model = create_model(
                    classifier_type='delta',
                    input_dim=1280,
                    hidden_dim=512,
                    num_layers=4,
                    dropout=0.5357584107527866,
                    rank=4,
                    steps=2,
                )
            elif feature_type == "prot5":
                model = create_model(
                    classifier_type='delta',
                    input_dim=1024,
                    hidden_dim=1024,
                    num_layers=6,
                    dropout=0.13123101867893097,
                    rank=2,
                    steps=4,
                )
        
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def load_features_from_h5(h5_file, ids=None):
    """Load protein features from H5 file"""
    features = {}
    seq_ids = []
    
    with h5py.File(h5_file, 'r') as f:
        if 'ids' in f:
            seq_ids = [id.decode('utf-8') if isinstance(id, bytes) else id for id in f['ids'][:]]
        else:
            # If no IDs are specified in the file, use indices
            seq_ids = [f"seq_{i}" for i in range(len(f['mean']))]
        
        embeddings = f['mean'][:]
        
        # If specific IDs are requested, filter them
        if ids is not None:
            id_indices = {id: i for i, id in enumerate(seq_ids)}
            for id in ids:
                if id in id_indices:
                    features[id] = embeddings[id_indices[id]]
        else:
            for i, id in enumerate(seq_ids):
                features[id] = embeddings[i]
    
    return features, seq_ids

def predict_from_features(model, features, device, is_fusion=False, feature_type="esm2"):
    """Generate predictions from extracted features"""
    model.eval()
    
    with torch.no_grad():
        if is_fusion:
            esm_features, prot5_features = features
            esm_features = esm_features.to(device)
            prot5_features = prot5_features.to(device)
            outputs = model(esm_features, prot5_features)
        else:
            if feature_type == "esm2":
                features_tensor = torch.tensor(features[0], dtype=torch.float32).to(device)
            else:  # prot5
                features_tensor = torch.tensor(features[1], dtype=torch.float32).to(device)
            outputs = model(features_tensor)
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
    return preds.cpu().numpy(), probs.cpu().numpy()

def ensemble_predictions(all_scores, methods, weights=None):
    """Generate ensemble predictions using different methods"""
    num_models = len(all_scores)
    ensemble_results = {}
    
    # If weights not provided, use equal weighting
    if weights is None:
        weights = np.ones(num_models) / num_models
        
    # Generate predictions for each ensemble method
    for method in methods:
        if method == "simple_avg":
            ensemble_scores = np.mean(all_scores, axis=0)
        elif method == "weighted_avg":
            ensemble_scores = np.zeros_like(all_scores[0])
            for i in range(num_models):
                ensemble_scores += all_scores[i] * weights[i]
        elif method == "majority_vote":
            # Get individual predictions first
            model_preds = [np.argmax(scores, axis=1) for scores in all_scores]
            ensemble_preds = []
            
            for i in range(len(all_scores[0])):
                votes = [model_preds[j][i] for j in range(num_models)]
                count_0 = votes.count(0)
                count_1 = votes.count(1)
                if count_1 > count_0:
                    ensemble_preds.append(1)
                else:
                    ensemble_preds.append(0)
                    
            # Calculate confidence as the proportion of votes
            confidence = np.array([votes.count(pred)/len(votes) for pred, votes in 
                                   zip(ensemble_preds, [[model_preds[j][i] for j in range(num_models)] 
                                                        for i in range(len(all_scores[0]))])])
            
            ensemble_results[method] = {
                'predictions': np.array(ensemble_preds),
                'confidence': confidence
            }
            continue
            
        # For averaging methods
        ensemble_preds = np.argmax(ensemble_scores, axis=1)
        ensemble_results[method] = {
            'predictions': ensemble_preds,
            'confidence': ensemble_scores[:, 1]  # Class 1 probability
        }
            
    return ensemble_results

def generate_visualization(results, sequence_ids):
    """Generate visualization of prediction results"""
    # Prepare data for visualization
    viz_data = []
    
    for seq_id in sequence_ids:
        for model, (pred, conf) in results[seq_id]['models'].items():
            viz_data.append({
                'Sequence ID': seq_id,
                'Model': model,
                'Prediction': 'Virulence Factor' if pred == 1 else 'Non-Virulence',
                'Confidence': conf,
                'Type': 'Individual Model'
            })
        
        # Add ensemble results
        for method, (pred, conf) in results[seq_id]['ensemble'].items():
            viz_data.append({
                'Sequence ID': seq_id,
                'Model': f"Ensemble ({method})",
                'Prediction': 'Virulence Factor' if pred == 1 else 'Non-Virulence',
                'Confidence': conf,
                'Type': 'Ensemble Method'
            })
    
    df = pd.DataFrame(viz_data)
    
    # Create bar chart
    fig = px.bar(
        df, 
        x='Model', 
        y='Confidence', 
        color='Prediction',
        barmode='group',
        facet_col='Sequence ID' if len(sequence_ids) > 1 else None,
        color_discrete_map={
            'Virulence Factor': '#E74C3C', 
            'Non-Virulence': '#2ECC71'
        },
        title='Prediction Confidence by Model',
        labels={'Confidence': 'Confidence Score (0-1)'},
        height=500
    )
    
    # Update layout
    fig.update_layout(
        legend_title_text='Prediction',
        xaxis_title="",
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    # Create heatmap for multiple sequences
    if len(sequence_ids) > 1:
        pivot_df = df.pivot_table(
            values='Confidence', 
            index='Sequence ID', 
            columns='Model',
            aggfunc='mean'
        )
        
        fig2 = px.imshow(
            pivot_df,
            color_continuous_scale='RdYlGn',
            title='Confidence Heatmap',
            labels=dict(x="Model", y="Sequence ID", color="Confidence"),
            height=400
        )
        
        return fig, fig2
    
    return fig, None

def create_downloadable_csv(results):
    """Create downloadable CSV from results"""
    records = []
    
    for seq_id, data in results.items():
        row = {'Sequence ID': seq_id}
        
        # Add individual model predictions
        for model, (pred, conf) in data['models'].items():
            row[f"{model} Prediction"] = 'Virulence Factor' if pred == 1 else 'Non-Virulence'
            row[f"{model} Confidence"] = conf
        
        # Add ensemble predictions
        for method, (pred, conf) in data['ensemble'].items():
            row[f"Ensemble({method}) Prediction"] = 'Virulence Factor' if pred == 1 else 'Non-Virulence'
            row[f"Ensemble({method}) Confidence"] = conf
            
        records.append(row)
    
    return pd.DataFrame(records)

def predict_from_uploaded_features(esm_features, prot5_features, models_info, device):
    """Predict using pre-computed features"""
    results = {}
    
    # Get common sequence IDs
    common_ids = set(esm_features.keys()) & set(prot5_features.keys())
    seq_ids = list(common_ids)
    
    if not seq_ids:
        st.error("No matching sequence IDs found in the uploaded feature files")
        return {}
    
    # Setup progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = len(seq_ids) * (len(models_info) + 1)  # +1 for ensemble
    current_step = 0
    
    # Process each sequence
    for idx, seq_id in enumerate(seq_ids):
        status_text.text(f"Processing sequence {idx+1}/{len(seq_ids)}: {seq_id}")
        
        # Get features for this sequence
        esm_feature = esm_features[seq_id]
        prot5_feature = prot5_features[seq_id]
        
        # Initialize results for this sequence
        results[seq_id] = {
            'models': {},
            'ensemble': {}
        }
        
        # Run each model
        all_scores = []
        model_weights = []
        
        for model_info in models_info:
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            status_text.text(f"Running model: {model_info['name']} on {seq_id}")
            
            model = model_info['model']
            is_fusion = "dual" in model_info['type'].lower()
            feature_type = model_info['feature_type']
            
            # Create appropriate features input
            if is_fusion:
                features = [
                    torch.tensor(esm_feature, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(prot5_feature, dtype=torch.float32).unsqueeze(0)
                ]
            else:
                features = [
                    torch.tensor(esm_feature, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(prot5_feature, dtype=torch.float32).unsqueeze(0)
                ]
            
            # Get predictions
            preds, probs = predict_from_features(
                model, features, device, is_fusion, feature_type
            )
            
            # Store results
            results[seq_id]['models'][model_info['name']] = (preds[0], probs[0][1])
            all_scores.append(probs)
            model_weights.append(model_info.get('weight', 1.0))
        
        # Normalize weights
        model_weights = np.array(model_weights)
        model_weights = model_weights / model_weights.sum()
        
        # Ensemble predictions
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        status_text.text(f"Generating ensemble predictions for {seq_id}")
        
        ensemble_methods = ['simple_avg', 'weighted_avg', 'majority_vote']
        ensemble_results = ensemble_predictions(all_scores, ensemble_methods, model_weights)
        
        # Store ensemble results
        for method, result in ensemble_results.items():
            results[seq_id]['ensemble'][method] = (
                result['predictions'][0],
                result['confidence'][0]
            )
    
    progress_bar.progress(100)
    status_text.text("Prediction complete!")
    
    # Clean up UI
    status_text.empty()
    progress_bar.empty()
    
    return results

# Main application
def main():
    add_custom_css()
    set_all_seeds(42)
    
    # Initialize session state
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'feature_ids' not in st.session_state:
        st.session_state.feature_ids = None
    
    # Page header
    st.markdown("<h1 class='main-header'>VF-FUSE: Virulence Factor Prediction</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
    A deep learning system that uses protein language models and ensemble techniques to predict 
    bacterial virulence factors with high accuracy.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Configuration")
        
        # Model selection
        st.markdown("### Select Models")
        use_esm2 = st.checkbox("ESM2 Model", value=True)
        use_prot5 = st.checkbox("ProtT5 Model", value=True)
        use_fusion = st.checkbox("Fusion Model", value=True)
        
        if not (use_esm2 or use_prot5 or use_fusion):
            st.warning("Please select at least one model")
        
        # Ensemble method selection
        st.markdown("### Ensemble Methods")
        use_simple_avg = st.checkbox("Simple Average", value=True)
        use_weighted_avg = st.checkbox("Weighted Average", value=True)
        use_majority = st.checkbox("Majority Vote", value=True)
        
        if not (use_simple_avg or use_weighted_avg or use_majority):
            st.warning("Please select at least one ensemble method")
            
        # Advanced options
        st.markdown("### Advanced Options")
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        
        # About section
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        VF-FUSE combines protein language models (ESM2 and ProtT5) 
        with deep learning architectures to identify bacterial virulence factors.
        
        [GitHub Repository](https://github.com/yourusername/VF-FUSE)
        """)
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Prediction", "Results", "Documentation"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Protein Embeddings Input</h2>", unsafe_allow_html=True)
        
        # Upload feature files
        st.markdown("### Upload Pre-computed Protein Embeddings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**ESM2 Embeddings**")
            esm_file = st.file_uploader("Upload ESM2 features (H5 format)", type=["h5"])
        
        with col2:
            st.markdown("**ProtT5 Embeddings**")
            prot5_file = st.file_uploader("Upload ProtT5 features (H5 format)", type=["h5"])
        
        # Load and display feature information
        esm_features = {}
        prot5_features = {}
        esm_ids = []
        prot5_ids = []
        
        if esm_file and prot5_file:
            # Load ESM2 features
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_esm:
                tmp_esm.write(esm_file.getvalue())
                tmp_esm_path = tmp_esm.name
            
            # Load ProtT5 features
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_prot5:
                tmp_prot5.write(prot5_file.getvalue())
                tmp_prot5_path = tmp_prot5.name
                
            try:
                # Process uploaded feature files
                esm_features, esm_ids = load_features_from_h5(tmp_esm_path)
                prot5_features, prot5_ids = load_features_from_h5(tmp_prot5_path)
                
                # Find common sequence IDs
                common_ids = set(esm_ids) & set(prot5_ids)
                
                # Display information
                st.markdown(f"**ESM2 Features:** {len(esm_ids)} sequences loaded")
                st.markdown(f"**ProtT5 Features:** {len(prot5_ids)} sequences loaded")
                st.markdown(f"**Common Sequences:** {len(common_ids)} sequences found in both files")
                
                if len(common_ids) == 0:
                    st.error("No common sequence IDs found between the two feature files")
                
                # Store feature IDs in session state
                st.session_state.feature_ids = list(common_ids)
                
            except Exception as e:
                st.error(f"Error processing feature files: {str(e)}")
            
            finally:
                os.unlink(tmp_esm_path)
                os.unlink(tmp_prot5_path)
        
        # Predict button
        predict_col1, predict_col2 = st.columns([1, 3])
        with predict_col1:
            predict_button = st.button(
                "Predict Virulence Factors", 
                type="primary", 
                disabled=(not esm_file or not prot5_file or 
                         not (use_esm2 or use_prot5 or use_fusion) or
                         len(common_ids) == 0)
            )
        
        # Run prediction when button is clicked
        if predict_button:
            # Load models
            device = get_device()
            st.info(f"Using device: {device}")
            
            config = {
                "dropout": 0.1,
                "batch_size": 32
            }
            
            # Load selected models
            models_info = []
            
            if use_esm2:
                esm2_model = load_model(
                    model_path="best/esm2_best.pth",
                    model_type="simple_transformer", 
                    config=config,
                    esm_dim=1280,
                    prot5_dim=1024,
                    device=device,
                    feature_type="esm2"
                )
                if esm2_model:
                    models_info.append({
                        'model': esm2_model,
                        'name': 'ESM2',
                        'type': 'simple_transformer',
                        'feature_type': 'esm2',
                        'weight': 1.0
                    })
            
            if use_prot5:
                prot5_model = load_model(
                    model_path="best/prot5_best_model.pth",
                    model_type="simple_transformer", 
                    config=config,
                    esm_dim=1280,
                    prot5_dim=1024,
                    device=device,
                    feature_type="prot5"
                )
                if prot5_model:
                    models_info.append({
                        'model': prot5_model,
                        'name': 'ProtT5',
                        'type': 'simple_transformer',
                        'feature_type': 'prot5',
                        'weight': 1.0
                    })
            
            if use_fusion:
                fusion_model = load_model(
                    model_path="best/fusion_fine_tuned.pth",
                    model_type="dual_transformer", 
                    config=config,
                    esm_dim=1280,
                    prot5_dim=1024,
                    device=device,
                    feature_type="dual"
                )
                if fusion_model:
                    models_info.append({
                        'model': fusion_model,
                        'name': 'Fusion',
                        'type': 'dual_transformer',
                        'feature_type': 'dual',
                        'weight': 1.2
                    })
            
            if not models_info:
                st.error("No models could be loaded. Please check model paths and configurations.")
            else:
                # Run prediction
                with st.spinner("Running prediction... This may take a few minutes."):
                    results = predict_from_uploaded_features(
                        esm_features, prot5_features, models_info, device
                    )
                
                # Store results in session state
                st.session_state.prediction_results = results
                
                # Switch to results tab
                if results:
                    st.success("Prediction complete! View results in the Results tab.")
                else:
                    st.error("Prediction failed. Please check the input data.")

    with tab2:
        st.markdown("<h2 class='sub-header'>Prediction Results</h2>", unsafe_allow_html=True)
        
        if st.session_state.prediction_results:
            results = st.session_state.prediction_results
            
            # Create results dataframe
            df = create_downloadable_csv(results)
            
            # Display summary
            st.markdown("### Summary")
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("Total Sequences", len(results))
            
            # Count predicted virulence factors using the weighted_avg ensemble
            vf_count = sum(1 for seq_data in results.values() 
                          if seq_data['ensemble'].get('weighted_avg', (0, 0))[0] == 1)
            
            with summary_cols[1]:
                st.metric("Predicted VFs", vf_count)
                
            with summary_cols[2]:
                st.metric("Predicted Non-VFs", len(results) - vf_count)
            
            # Most confident prediction
            if results:
                max_conf = max(results.values(), 
                              key=lambda x: x['ensemble'].get('weighted_avg', (0, 0))[1])
                max_conf_id = [k for k, v in results.items() 
                             if v['ensemble'].get('weighted_avg', (0, 0))[1] == 
                             max_conf['ensemble'].get('weighted_avg', (0, 0))[1]][0]
                
                with summary_cols[3]:
                    st.metric("Highest Confidence", 
                            f"{max_conf['ensemble'].get('weighted_avg', (0, 0))[1]:.2f}",
                            f"({max_conf_id})")
            
            # Generate visualizations
            st.markdown("### Visualization")
            fig, heatmap = generate_visualization(results, list(results.keys()))
            
            # Display bar chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display heatmap for multiple sequences
            if heatmap is not None:
                st.plotly_chart(heatmap, use_container_width=True)
            
            # Detailed results table
            st.markdown("### Detailed Results")
            st.dataframe(df, use_container_width=True)
            
            # Filter by confidence threshold
            threshold = st.sidebar.slider(
                "Filter by confidence threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.05
            )
            
            # Apply filter
            filtered_df = df[df.filter(like='Confidence').max(axis=1) >= threshold]
            if len(filtered_df) < len(df):
                st.markdown(f"**Filtered Results (Confidence ≥ {threshold}):**")
                st.dataframe(filtered_df, use_container_width=True)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="vf_prediction_results.csv",
                mime="text/csv",
            )
        else:
            st.info("No prediction results yet. Please go to the Prediction tab to run a prediction.")

    with tab3:
        st.markdown("<h2 class='sub-header'>Documentation</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        ## VF-FUSE: Deep Learning Virulence Factor Prediction
        
        ### Introduction
        
        VF-FUSE is an ensemble learning system that combines protein language models (ESM2 and ProtT5) 
        with deep learning architectures for accurate prediction of bacterial virulence factors.
        
        ### Models
        
        1. **ESM2 Model**: Uses embeddings from the ESM2 protein language model
        2. **ProtT5 Model**: Uses embeddings from the ProtT5 protein language model
        3. **Fusion Model**: Combines both ESM2 and ProtT5 embeddings
        
        ### Ensemble Methods
        
        1. **Simple Average**: Averages predictions from all models
        2. **Weighted Average**: Weighted combination of model predictions
        3. **Majority Vote**: Uses the most common prediction across models
        
        ### Input Format
        
        The system requires pre-computed protein embeddings in H5 format:
        
        1. **ESM2 Embeddings**: H5 file containing mean embeddings from ESM2
        2. **ProtT5 Embeddings**: H5 file containing mean embeddings from ProtT5
        
        Both files should contain sequence IDs in the `ids` dataset and feature vectors in the `mean` dataset.
        
        ### Interpretation
        
        - **Prediction**: "Virulence Factor" or "Non-Virulence"
        - **Confidence**: Score from 0 to 1 indicating confidence in prediction
        - Higher confidence scores (closer to 1) indicate stronger predictions
        
        ### References
        
        1. ESM2: [Paper link](https://www.science.org/doi/10.1126/science.ade2574)
        2. ProtT5: [Paper link](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v3)
        """)
        
        # Feature extraction information
        st.markdown("""
        ### Feature Extraction
        
        To generate feature files for prediction, use the following tools:
        
        #### ESM2 Feature Extraction
        ```bash
        python get_esm2_embedding.py --input sequences.fasta --output esm2_features.h5 --model esm2_t33_650M_UR50D
        ```
        
        #### ProtT5 Feature Extraction
        ```bash
        python get_prot5.py --input sequences.fasta --output prot5_features.h5
        ```
        """)

    # Footer
    st.markdown("""
    <div class='footer'>
        VF-FUSE: Protein Language Model-based Virulence Factor Prediction System © 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()