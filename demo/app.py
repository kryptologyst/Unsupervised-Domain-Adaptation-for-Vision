"""Streamlit demo app for domain adaptation visualization."""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
from typing import Dict, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models import DANNModel, CORALModel
from src.data import create_toy_datasets, create_data_loaders
from src.eval import DomainAdaptationEvaluator
from src.utils import get_device, set_seed


def load_model(model_type: str, num_classes: int = 10) -> nn.Module:
    """Load a pre-trained model or create a new one.
    
    Args:
        model_type: Type of model to load
        num_classes: Number of classes
        
    Returns:
        Loaded model
    """
    device = get_device()
    
    if model_type == "DANN":
        model = DANNModel(
            num_classes=num_classes,
            backbone="resnet50",
            pretrained=True,
            freeze_backbone=False,
            lambda_param=1.0,
            dropout=0.5
        )
    elif model_type == "CORAL":
        model = CORALModel(
            num_classes=num_classes,
            backbone="resnet50",
            pretrained=True,
            freeze_backbone=False,
            dropout=0.5
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.to(device)
    model.eval()
    
    return model


def create_sample_images(num_samples: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Create sample images from both domains.
    
    Args:
        num_samples: Number of samples per domain
        
    Returns:
        Tuple of (source_images, target_images)
    """
    set_seed(42)
    
    # Create toy datasets
    source_dataset, target_dataset = create_toy_datasets(
        num_source_samples=num_samples,
        num_target_samples=num_samples,
        num_classes=10,
        image_size=224,
        augmentation=False
    )
    
    source_images = []
    target_images = []
    
    for i in range(num_samples):
        source_img, _ = source_dataset[i]
        target_img, _ = target_dataset[i]
        
        # Convert to numpy for display
        source_img_np = source_img.permute(1, 2, 0).numpy()
        source_img_np = (source_img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        source_img_np = np.clip(source_img_np, 0, 1)
        
        target_img_np = target_img.permute(1, 2, 0).numpy()
        target_img_np = (target_img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        target_img_np = np.clip(target_img_np, 0, 1)
        
        source_images.append(source_img_np)
        target_images.append(target_img_np)
    
    return np.array(source_images), np.array(target_images)


def visualize_domain_differences(source_images: np.ndarray, target_images: np.ndarray):
    """Visualize differences between source and target domains.
    
    Args:
        source_images: Source domain images
        target_images: Target domain images
    """
    st.subheader("Domain Visualization")
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(2, len(source_images), figsize=(20, 8))
    
    for i in range(len(source_images)):
        # Source images
        axes[0, i].imshow(source_images[i])
        axes[0, i].set_title(f"Source {i+1}")
        axes[0, i].axis('off')
        
        # Target images
        axes[1, i].imshow(target_images[i])
        axes[1, i].set_title(f"Target {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Domain statistics
    st.subheader("Domain Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Source Domain:**")
        st.write(f"- Mean RGB: {np.mean(source_images, axis=(0,1,2))}")
        st.write(f"- Std RGB: {np.std(source_images, axis=(0,1,2))}")
        st.write(f"- Brightness: {np.mean(source_images)}")
    
    with col2:
        st.write("**Target Domain:**")
        st.write(f"- Mean RGB: {np.mean(target_images, axis=(0,1,2))}")
        st.write(f"- Std RGB: {np.std(target_images, axis=(0,1,2))}")
        st.write(f"- Brightness: {np.mean(target_images)}")


def model_comparison():
    """Compare different domain adaptation models."""
    st.subheader("Model Comparison")
    
    # Create sample data
    source_dataset, target_dataset = create_toy_datasets(
        num_source_samples=100,
        num_target_samples=100,
        num_classes=10,
        image_size=224,
        augmentation=False
    )
    
    source_loader, target_loader = create_data_loaders(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=False
    )
    
    # Evaluate models
    models = ["DANN", "CORAL"]
    results = {}
    
    for model_name in models:
        with st.spinner(f"Evaluating {model_name} model..."):
            model = load_model(model_name)
            evaluator = DomainAdaptationEvaluator(model, get_device())
            
            # Evaluate on both domains
            source_metrics = evaluator.evaluate_task_performance(source_loader, "source")
            target_metrics = evaluator.evaluate_task_performance(target_loader, "target")
            
            results[model_name] = {
                'source_acc': source_metrics['accuracy'],
                'target_acc': target_metrics['accuracy']
            }
    
    # Create comparison chart
    model_names = list(results.keys())
    source_accs = [results[m]['source_acc'] for m in model_names]
    target_accs = [results[m]['target_acc'] for m in model_names]
    
    fig = go.Figure(data=[
        go.Bar(name='Source Domain', x=model_names, y=source_accs, marker_color='blue'),
        go.Bar(name='Target Domain', x=model_names, y=target_accs, marker_color='red')
    ])
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Accuracy',
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Results table
    st.subheader("Detailed Results")
    
    comparison_data = []
    for model_name in model_names:
        comparison_data.append({
            'Model': model_name,
            'Source Accuracy': f"{results[model_name]['source_acc']:.4f}",
            'Target Accuracy': f"{results[model_name]['target_acc']:.4f}",
            'Domain Gap': f"{results[model_name]['source_acc'] - results[model_name]['target_acc']:.4f}"
        })
    
    st.table(comparison_data)


def interactive_prediction():
    """Interactive prediction interface."""
    st.subheader("Interactive Prediction")
    
    # Model selection
    model_type = st.selectbox("Select Model", ["DANN", "CORAL"])
    
    # Load model
    model = load_model(model_type)
    
    # Image selection
    st.write("Select an image to classify:")
    
    # Create sample images
    source_images, target_images = create_sample_images(4)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Source Domain Images:**")
        for i, img in enumerate(source_images):
            if st.button(f"Source {i+1}", key=f"source_{i}"):
                st.session_state.selected_image = img
                st.session_state.selected_domain = "Source"
    
    with col2:
        st.write("**Target Domain Images:**")
        for i, img in enumerate(target_images):
            if st.button(f"Target {i+1}", key=f"target_{i}"):
                st.session_state.selected_image = img
                st.session_state.selected_domain = "Target"
    
    # Display selected image and prediction
    if 'selected_image' in st.session_state:
        st.write(f"Selected {st.session_state.selected_domain} Domain Image:")
        st.image(st.session_state.selected_image, width=300)
        
        # Make prediction
        with st.spinner("Making prediction..."):
            # Convert image to tensor
            img_tensor = torch.from_numpy(st.session_state.selected_image).permute(2, 0, 1).float()
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            
            # Normalize
            img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            
            # Predict
            with torch.no_grad():
                outputs = model(img_tensor.to(get_device()))
                task_logits = outputs['task_logits']
                probabilities = torch.softmax(task_logits, dim=1)
                
                if 'domain_logits' in outputs:
                    domain_logits = outputs['domain_logits']
                    domain_prob = torch.softmax(domain_logits, dim=1)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Task Classification:**")
            class_probs = probabilities[0].cpu().numpy()
            
            # Create bar chart
            fig = px.bar(
                x=list(range(len(class_probs))),
                y=class_probs,
                title="Class Probabilities",
                labels={'x': 'Class', 'y': 'Probability'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            predicted_class = torch.argmax(task_logits, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            st.write(f"**Predicted Class:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.4f}")
        
        with col2:
            if 'domain_logits' in outputs:
                st.write("**Domain Classification:**")
                domain_probs = domain_prob[0].cpu().numpy()
                
                fig = px.bar(
                    x=['Source', 'Target'],
                    y=domain_probs,
                    title="Domain Probabilities",
                    labels={'x': 'Domain', 'y': 'Probability'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                predicted_domain = torch.argmax(domain_logits, dim=1).item()
                domain_confidence = domain_prob[0, predicted_domain].item()
                domain_name = "Source" if predicted_domain == 0 else "Target"
                st.write(f"**Predicted Domain:** {domain_name}")
                st.write(f"**Confidence:** {domain_confidence:.4f}")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Domain Adaptation Demo",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 Unsupervised Domain Adaptation Demo")
    st.markdown("""
    This demo showcases unsupervised domain adaptation techniques for computer vision tasks.
    We demonstrate how models can adapt from a source domain to a target domain without labeled target data.
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Domain Visualization", "Model Comparison", "Interactive Prediction"]
    )
    
    if page == "Domain Visualization":
        st.header("Domain Visualization")
        st.write("Visualize the differences between source and target domains.")
        
        source_images, target_images = create_sample_images(8)
        visualize_domain_differences(source_images, target_images)
        
    elif page == "Model Comparison":
        st.header("Model Comparison")
        st.write("Compare different domain adaptation methods.")
        
        model_comparison()
        
    elif page == "Interactive Prediction":
        st.header("Interactive Prediction")
        st.write("Test model predictions on sample images.")
        
        interactive_prediction()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this demo:**
    - **DANN (Domain-Adversarial Neural Network)**: Uses adversarial training to learn domain-invariant features
    - **CORAL (CORrelation ALignment)**: Aligns second-order statistics between domains
    - **Toy Dataset**: Synthetic datasets with different visual characteristics for source and target domains
    """)


if __name__ == "__main__":
    main()
