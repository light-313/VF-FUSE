import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import h5py
import tempfile
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from torch.utils.data import DataLoader
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Import necessary functions from your existing code
# Assuming these are in the same directory or properly importable
from esmmodel import *
from model_type import *

# Set page configuration
st.set_page_config(
    page_title="VF-pred: Virulence Factor Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #3B82F6;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        padding: 1.2rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        background-color: #F9FAFB;
        margin: 0.8rem 0;
        height: 100%;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1F2937;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        margin-top: 0.3rem;
    }
    .highlight {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 1.5rem 0;
    }
    .caption {
        font-size: 0.85rem;
        color: #4B5563;
        font-style: italic;
        margin-top: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        font-size: 1rem;
    }
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    section[data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem;
    }
    .section-spacing {
        margin-top: 2.5rem;
        margin-bottom: 2.5rem;
    }
    /* Add style for file upload areas */
    .uploadedFile {
        border: 1px dashed #3B82F6;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: rgba(59, 130, 246, 0.05);
    }
    /* Enhanced button style */
    .stButton>button {
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_h5_data(file):
    """Load data from uploaded H5 file to a temporary file for processing"""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_file.write(file.getvalue())
        temp_path = temp_file.name
    return temp_path

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_scientific_figure(data, max_display=5):
    """Generate publication-quality ROC and PR curves side by side"""
    # Define method colors and names
    method_colors = {
        'simple_avg': '#1f77b4',          # Blue
        'weighted_avg': '#ff7f0e',        # Orange
        'majority_vote': '#2ca02c',       # Green
        'stacking': '#d62728',            # Red
        'gradient_boosted_ensemble': '#9467bd'  # Purple
    }
    
    method_names = {
        'simple_avg': 'Simple Average',
        'weighted_avg': 'Weighted Average',
        'majority_vote': 'Majority Voting',
        'stacking': 'Stacking Ensemble',
        'gradient_boosted_ensemble': 'Gradient Boosted Ensemble'
    }
    
    # Set style for plots
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.linewidth'] = 1.2
    
    # Create figure with two side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    # Customize the appearance for scientific publication
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(width=1.2, length=5)
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # Limit to a maximum number of methods to display
    methods = list(data.keys())[:max_display]
    
    # Plot ROC curves on the left subplot
    for method in methods:
        fpr = data[method]['roc']['fpr']
        tpr = data[method]['roc']['tpr']
        auc_score = data[method]['roc']['auc']
        
        method_label = method_names.get(method, method)
        color = method_colors.get(method, None)
        
        axes[0].plot(fpr, tpr, label=f"{method_label} (AUC={auc_score:.2f}%)", 
                   lw=2, color=color)
    
    # Add reference diagonal line to ROC plot
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.7, lw=1)
    
    # Set ROC plot properties
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    axes[0].set_title('ROC Curve', fontsize=12, fontweight='bold')
    axes[0].legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=8)
    
    # Add minor grid for ROC
    axes[0].xaxis.set_minor_locator(MultipleLocator(0.05))
    axes[0].yaxis.set_minor_locator(MultipleLocator(0.05))
    
    # Plot PR curves on the right subplot
    for method in methods:
        precision = data[method]['pr']['precision']
        recall = data[method]['pr']['recall']
        aupr_score = data[method]['pr']['aupr']
        
        method_label = method_names.get(method, method)
        color = method_colors.get(method, None)
        
        axes[1].plot(recall, precision, label=f"{method_label} (AUPR={aupr_score:.2f}%)", 
                   lw=2, color=color)
    
    # Set PR plot properties
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Precision', fontsize=11, fontweight='bold')
    axes[1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8)
    
    # Add minor grid for PR
    axes[1].xaxis.set_minor_locator(MultipleLocator(0.05))
    axes[1].yaxis.set_minor_locator(MultipleLocator(0.05))
    
    plt.tight_layout(pad=2.0)
    
    return fig
def display_prediction_results(prediction_results, threshold=0.5):
    """Display prediction results with the applied threshold"""
    # Display results
    st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
    
    # Get results DataFrame
    results_df = prediction_results["results_df"]
    
    # Apply threshold - create a new prediction column based on threshold
    results_df["Thresholded_Prediction"] = (results_df["Confidence"] >= threshold).astype(int)
    
    # Create results display with threshold information
    display_df = results_df.copy()
    display_df["Status"] = display_df["Thresholded_Prediction"].apply(
        lambda x: "Virulence Factor ✓" if x == 1 else "Non-Virulence Factor ✗"
    )
    
    # Predictions after threshold application
    pos_count = (results_df["Thresholded_Prediction"] == 1).sum()
    neg_count = (results_df["Thresholded_Prediction"] == 0).sum()
    total = len(results_df)
    
    # Display summary statistics with different colors
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card" style="background-color: #F0F7FF; border-color: #BAD7FF;">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color: #1E40AF;">{total}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Total Sequences</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Red color for VFs (changed from green to red)
        st.markdown('<div class="metric-card" style="background-color: #FEF2F2; border-color: #FCA5A5; border-width: 2px;">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color: #DC2626; font-size: 1.9rem; font-weight: 800;">{pos_count} <small>({pos_count/total:.1%})</small></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label" style="font-weight: 600; font-size: 1rem;">Predicted Virulence Factors</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # Green color for non-VFs (changed from red to green)
        st.markdown('<div class="metric-card" style="background-color: #ECFDF5; border-color: #6EE7B7; border-width: 2px;">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color: #047857; font-size: 1.7rem;">{neg_count} <small>({neg_count/total:.1%})</small></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Predicted Non-Virulence Factors</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display threshold information
    st.info(f"Current decision threshold: {threshold:.2f}. Proteins with confidence scores above this threshold are predicted as virulence factors.")
    
    # Create two-column layout for table and distribution chart
    table_col, chart_col = st.columns([3, 2])
    
    with table_col:
        st.markdown('<h3 class="sub-header">Prediction Table</h3>', unsafe_allow_html=True)
        # Highlight rows - use styled DataFrame
        st.dataframe(
            display_df.style.apply(
                lambda row: ['background-color: rgba(255,160,122,0.2)' if row['Thresholded_Prediction'] == 1 
                            else 'background-color: rgba(176,224,230,0.2)' for _ in row],
                axis=1
            ),
            height=500
        )
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv,
            file_name="vf_predictions.csv",
            mime="text/csv",
        )
    
    with chart_col:
        st.markdown('<h3 class="sub-header">Confidence Distribution</h3>', unsafe_allow_html=True)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(
            data=results_df, x="Confidence", hue="Thresholded_Prediction", 
            bins=30, alpha=0.7, palette=["lightblue", "salmon"], 
            element="step", kde=True, ax=ax
        )
        # Add threshold line
        ax.axvline(x=threshold, color='red', linestyle='--')
        ax.text(
            threshold+0.02, ax.get_ylim()[1]*0.9, 
            f'Threshold: {threshold:.2f}', 
            color='red', fontsize=10
        )
        
        ax.set_xlabel("Prediction Confidence", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Distribution of Prediction Confidence", fontsize=14)
        ax.legend(labels=["Non-Virulence", "Virulence"])
        
        # Display chart
        st.pyplot(fig)
        
        # Confidence Guide
        st.markdown("#### Confidence Guide")
        st.markdown("""
        - **High** (>0.9): Very likely VF
        - **Medium** (0.7-0.9): Likely VF
        - **Low** (0.5-0.7): Uncertain prediction
        - **Very low** (<0.5): Likely not VF
        """)
    
    # High confidence candidates
    high_confidence_vf = results_df[(results_df["Thresholded_Prediction"] == 1) & (results_df["Confidence"] > 0.9)]
    high_confidence_vf_count = len(high_confidence_vf)
    
    if high_confidence_vf_count > 0:
        st.markdown('<h3 class="sub-header">High-Confidence Candidates</h3>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="highlight">
        ✓ {high_confidence_vf_count} sequences were predicted as virulence factors with high confidence (>90%).
        These are strong candidates for experimental validation.
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(high_confidence_vf)
    
    # Key insights
    st.markdown('<h3 class="sub-header">Key Insights</h3>', unsafe_allow_html=True)
    
    # Ensemble method used
    st.markdown(f"""
    - **Majority Vote** ensemble method was used for final predictions.
    - This method combines predictions from multiple individual models by selecting the most common prediction.
    """)
    
    # Insight about prediction distribution
    if pos_count/total > 0.3:
        st.markdown("- A relatively high proportion of sequences were predicted as virulence factors, which might indicate a dataset enriched for virulence-related proteins.")
    elif pos_count/total < 0.1:
        st.markdown("- A small proportion of sequences were predicted as virulence factors, suggesting high prediction specificity or a dataset with few virulence-related proteins.")
    
    # Recommendation for further analysis
    st.markdown("""
    - **Recommended next steps:**
      - Experimentally validate high-confidence predictions
      - Perform functional annotation of predicted virulence factors
      - Compare results with existing databases like VFDB
    """)
    
    st.markdown('<p class="caption">The accuracy of predictions depends on the quality of input embeddings and may vary across different protein families. Always validate important findings experimentally.</p>', unsafe_allow_html=True)
        
    
    
    
def run_prediction(esm_file, prot5_file, config, threshold=0.5, batch_size=64):
    """Run prediction on the uploaded files"""
    
    device = get_device()
    
    # Use config batch size or the one from parameters
    config["batch_size"] = batch_size
    
    # Save uploaded files to temporary paths
    esm_path = load_h5_data(esm_file) if esm_file else None
    prot5_path = load_h5_data(prot5_file) if prot5_file else None
    
    if not esm_path:
        st.error("ESM2 embedding file is required")
        return None
    
    try:
        # Load the dataset
        test_dataset = DualEmbeddingDataset(
            esm_h5_path=esm_path,
            prot5_h5_path=prot5_path,
            split='all'
        )
        
        collate_function = dual_features_collate_fn
        
        # Get feature dimensions
        sample = test_dataset[0]
        esm_features, prot5_features = sample[0]
        esm_dim = esm_features.shape[-1]
        prot5_dim = prot5_features.shape[-1]
        
        # Create data loader
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.get("batch_size", 64), 
            shuffle=False,
            collate_fn=collate_function
        )
        
        # Initialize prediction arrays
        all_labels = []
        all_model_preds = []
        all_model_scores = []
        model_results = []
        
        # Get sequence IDs
        with h5py.File(esm_path, 'r') as f:
            sequence_ids = list(f.keys())
        
        # Process each model
        for model_config in config["models"]:
            model_type = model_config["type"]
            model_path = model_config["path"]
            feature_type = model_config.get("feature_type", "esm2")
            model_name = model_config.get("name", model_type)
            
            # Skip if model file doesn't exist
            if not os.path.exists(model_path):
                st.warning(f"Model file not found: {model_path}. Skipping this model.")
                continue
            
            # Load the model
            is_fusion = "dual" in model_type.lower() or "fusion" in model_type.lower()
            
            model = load_model(
                model_path=model_path,
                model_type=model_type,
                config=config,
                esm_dim=esm_dim,
                prot5_dim=prot5_dim,
                input_dim=None,
                device=device,
                feature_type=feature_type
            )
            
            # Prepare loader for specific model type
            current_loader = test_loader
            if feature_type == "prot5" and not is_fusion and prot5_path:
                test_dataset_prot5 = H5Dataset(
                    h5_path=esm_path,
                    feature_type='prot5',
                    prot5_path=prot5_path,
                )
                current_loader = DataLoader(
                    test_dataset_prot5, 
                    batch_size=config.get("batch_size", 64), 
                    shuffle=False,
                    collate_fn=collate_fn
                )
            # Run prediction
            labels, preds, scores = predict_with_model(
                model, current_loader, device, is_fusion, feature_type=feature_type
            )
            if len(all_labels) == 0:
                all_labels = labels
            all_model_preds.append(preds)
            all_model_scores.append(scores)
        
        if not all_model_preds:
            st.error("No valid models could be loaded for prediction.")
            return None
        
        # Get model weights
        model_weights = np.array([model.get("weight", 1.0) for model in config["models"]])
        model_weights = model_weights / model_weights.sum()  # Normalize weights
        
        # Always use majority_vote as the ensemble method
        ensemble_method = "majority_vote"
            
        # Run ensemble prediction
        ensemble_preds, ensemble_scores = ensemble_predictions(
            all_labels, all_model_scores, all_model_preds,
            ensemble_method=ensemble_method,
            weights=model_weights
        )
        
        # Calculate ensemble metrics if true labels are provided
        ensemble_metrics = None
        
        # Prepare results dataframe
        results_df = pd.DataFrame({
            "Sequence_ID": sequence_ids,
            "Prediction": ensemble_preds,
            "Confidence": ensemble_scores[:, 1]
        })
        
        # Add individual model predictions
        for i, model_config in enumerate(config["models"]):
            if i < len(all_model_preds):  # Check if we have predictions for this model
                model_name = model_config.get("name", model_config["type"])
                results_df[f"{model_name}_Prediction"] = all_model_preds[i]
                results_df[f"{model_name}_Confidence"] = all_model_scores[i][:, 1]
        
        # Store results
        results = {
            "results_df": results_df,
            "ensemble_metrics": ensemble_metrics,
            "model_results": model_results,
            "ensemble_method": ensemble_method
        }
        
        # Clean up temporary files
        if esm_path:
            os.unlink(esm_path)
        if prot5_path:
            os.unlink(prot5_path)
        
        return results
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        # Clean up temporary files
        if esm_path:
            os.unlink(esm_path)
        if prot5_path:
            os.unlink(prot5_path)
        return None

# Main app layout
st.markdown('<h1 class="main-header">VF-pred: Virulence Factor Prediction Tool</h1>', unsafe_allow_html=True)
st.markdown("""
This application predicts virulence factors using ensemble deep learning models based on protein language model embeddings.
Upload your embedding files, adjust parameters in the sidebar, and get predictions.
""")

# Sidebar for parameter configuration
st.sidebar.title("Parameters")

# Decision threshold slider
threshold = st.sidebar.slider(
    "Prediction Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.01,
    help="Proteins with probability above this value will be predicted as virulence factors"
)
st.session_state['threshold'] = threshold

# Batch size slider
batch_size = st.sidebar.slider(
    "Batch Size", 
    min_value=8, 
    max_value=128, 
    value=64, 
    step=8,
    help="Larger batch size may speed up prediction but requires more memory"
)
st.session_state['batch_size'] = batch_size

# Add method explanation
st.sidebar.markdown("## Ensemble Method")
st.sidebar.markdown("""
This tool uses the **Majority Vote** ensemble method that:
- Takes predictions from multiple models
- Selects the most frequent prediction class for each protein
- Provides robust results by combining different model perspectives
""")

# About section
st.sidebar.markdown("## About")
st.sidebar.info("""
**VF-pred Tool**  
Version 1.0.0  
© 2025 VF-pred Team  

This tool implements ensemble deep learning models for virulence factor prediction using protein language model embeddings.
""")

# Create two-column layout for file upload
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ESM2 Embeddings")
    esm_file = st.file_uploader("Upload ESM2 embeddings (H5 format)", type=["h5"])
    st.caption("Required: H5 file containing ESM2 embeddings for protein sequences")

with col2:
    st.markdown("### ProtT5 Embeddings")
    prot5_file = st.file_uploader("Upload ProtT5 embeddings (H5 format)", type=["h5"])
    st.caption("Optional: H5 file containing ProtT5 embeddings for protein sequences")

# Run prediction button below file upload
st.markdown("---")
if st.button("Run Prediction", type="primary", use_container_width=True):
    if not esm_file:
        st.error("ESM2 embedding file is required.")
    else:
        with st.spinner("Running prediction..."):
            # Get parameters from sidebar
            threshold = st.session_state.get('threshold', 0.5)
            batch_size = st.session_state.get('batch_size', 8)
            
            # Load config
            try:
                with open('VF-FUSE/config.json', 'r') as f:
                    config = json.load(f)
            except:
                try:
                    with open('config.json', 'r') as f:
                        config = json.load(f)
                except:
                    st.error("Unable to load configuration file. Please check the file path.")
                    st.stop()
            
            # Run prediction with majority_vote method only
            results = run_prediction(
                esm_file, prot5_file, config, threshold, batch_size
            )
            
            if results:
                # Display results directly without tabs
                display_prediction_results(results, threshold)

# Default content when app first loads
else:
    st.info("""
    ## Getting Started
    
    1. **Prepare your data files**:
       - Generate ESM2 embeddings using ESM2 model (t33_650M_UR50D recommended)
       - Generate ProtT5 embeddings using ProtT5 model (optional)
       - Save both as H5 files with protein sequences as keys
    
    2. **Upload your files**:
       - Use the file uploaders above to select your embedding files
       - ESM2 file is required, ProtT5 file is optional
    
    3. **Adjust parameters**:
       - Set the prediction threshold in the sidebar
       - Configure batch size
    
    4. **Run prediction**:
       - Click the "Run Prediction" button
       - Review the results and analysis
    
    5. **Interpret results**:
       - Proteins with confidence above the threshold are predicted as virulence factors
       - Higher confidence scores (closer to 1.0) indicate stronger predictions
       - Adjust the threshold to balance precision and recall
    """)
    
    