# Unsupervised Domain Adaptation for Vision

A research-ready implementation of unsupervised domain adaptation techniques for computer vision tasks. This project demonstrates how to adapt models from a source domain to a target domain without requiring labeled target data.

## Overview

This project implements two state-of-the-art domain adaptation methods:

- **DANN (Domain-Adversarial Neural Network)**: Uses adversarial training with gradient reversal to learn domain-invariant features
- **CORAL (CORrelation ALignment)**: Aligns second-order statistics between domains to reduce domain shift

## Features

- **Modern Architecture**: Built with PyTorch 2.x and Python 3.10+
- **Device Support**: Automatic device detection (CUDA → MPS → CPU)
- **Reproducible**: Deterministic seeding and comprehensive configuration management
- **Comprehensive Evaluation**: Multiple metrics including accuracy, domain confusion, and feature alignment
- **Interactive Demo**: Streamlit-based visualization and testing interface
- **Production Ready**: Clean code structure with type hints and documentation

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)
- Apple Silicon with MPS support (optional)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Unsupervised-Domain-Adaptation-for-Vision.git
cd Unsupervised-Domain-Adaptation-for-Vision
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install pre-commit hooks (optional):
```bash
pre-commit install
```

## Quick Start

### Training a Model

Train a DANN model with default settings:

```bash
python train.py
```

Train a CORAL model:

```bash
python train.py model=coral
```

Train with custom configuration:

```bash
python train.py model=dann training.max_epochs=50 training.learning_rate=2e-4
```

### Running the Demo

Launch the interactive Streamlit demo:

```bash
streamlit run demo/app.py
```

The demo provides:
- Domain visualization and comparison
- Model performance comparison
- Interactive prediction interface

### Running Tests

Execute the test suite:

```bash
pytest tests/ -v
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── layers/            # Custom neural network layers
│   ├── data/              # Data handling and datasets
│   ├── train/             # Training utilities
│   ├── eval/              # Evaluation utilities
│   └── utils/             # Utility functions
├── configs/               # Hydra configuration files
│   ├── model/             # Model configurations
│   ├── data/              # Data configurations
│   └── training/          # Training configurations
├── demo/                  # Streamlit demo application
├── tests/                 # Test files
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks
├── assets/                # Generated outputs and visualizations
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
└── data/                  # Dataset storage
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/dann.yaml`: DANN model configuration
- `configs/model/coral.yaml`: CORAL model configuration
- `configs/data/toy_dataset.yaml`: Dataset configuration
- `configs/training/default.yaml`: Training configuration

### Key Parameters

- `model.backbone`: Backbone architecture (resnet18, resnet50)
- `model.num_classes`: Number of task classes
- `training.max_epochs`: Maximum training epochs
- `training.learning_rate`: Learning rate
- `training.lambda_domain`: Domain loss weight for DANN
- `training.lambda_coral`: CORAL loss weight

## Dataset Schema

The project includes a synthetic toy dataset generator that creates:

- **Source Domain**: Colorful patterns (stripes, checkerboard)
- **Target Domain**: Grayscale patterns (circles, diagonals, noise)

### Dataset Format

- **Images**: RGB images of size 224x224
- **Labels**: Integer class labels (0-9)
- **Domains**: Binary domain labels (0=source, 1=target)

### Custom Datasets

To use your own datasets, implement the dataset interface:

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, ...):
        # Initialize your dataset
    
    def __len__(self):
        # Return dataset size
    
    def __getitem__(self, idx):
        # Return (image, label) tuple
        return image, label
```

## Model Architectures

### DANN (Domain-Adversarial Neural Network)

DANN uses adversarial training to learn domain-invariant features:

1. **Feature Extractor**: ResNet backbone for feature extraction
2. **Task Classifier**: Predicts task labels
3. **Domain Classifier**: Predicts domain labels with gradient reversal
4. **Loss Function**: Task loss + λ × Domain loss

### CORAL (CORrelation ALignment)

CORAL aligns second-order statistics between domains:

1. **Feature Extractor**: ResNet backbone for feature extraction
2. **Task Classifier**: Predicts task labels
3. **CORAL Loss**: Frobenius norm of covariance matrix difference

## Evaluation Metrics

The project provides comprehensive evaluation metrics:

### Task Performance
- **Accuracy**: Classification accuracy on source and target domains
- **Confusion Matrix**: Detailed classification breakdown
- **Classification Report**: Precision, recall, F1-score per class

### Domain Adaptation Metrics
- **Domain Classification Accuracy**: How well the model distinguishes domains
- **MMD (Maximum Mean Discrepancy)**: Feature distribution distance
- **CORAL Distance**: Second-order statistic alignment

### Feature Alignment
- **Mean Distance**: L2 distance between domain feature means
- **Covariance Alignment**: Second-order moment alignment

## Training Commands

### Basic Training

```bash
# Train DANN model
python train.py model=dann

# Train CORAL model  
python train.py model=coral

# Train with custom settings
python train.py model=dann training.max_epochs=50 training.learning_rate=2e-4
```

### Advanced Training

```bash
# Multi-GPU training (if available)
python train.py model=dann training.batch_size=64

# Different backbone
python train.py model=dann model.backbone=resnet18

# Custom dataset size
python train.py data.num_source_samples=2000 data.num_target_samples=2000
```

## Results and Performance

### Expected Performance

On the toy dataset with default settings:

- **Source Domain Accuracy**: ~95-98%
- **Target Domain Accuracy**: ~85-92%
- **Domain Classification Accuracy**: ~50-60% (good domain confusion)
- **Training Time**: ~5-10 minutes on GPU

### Efficiency Metrics

- **Model Size**: ~25M parameters (ResNet50 backbone)
- **Memory Usage**: ~2-4GB VRAM during training
- **Inference Speed**: ~100-200 FPS on GPU

## Demo Usage

### Interactive Demo Features

1. **Domain Visualization**
   - Side-by-side comparison of source and target domains
   - Domain statistics and characteristics
   - Visual pattern analysis

2. **Model Comparison**
   - Performance comparison between DANN and CORAL
   - Accuracy metrics on both domains
   - Domain gap analysis

3. **Interactive Prediction**
   - Real-time model predictions
   - Confidence scores and class probabilities
   - Domain classification results

### Running the Demo

```bash
streamlit run demo/app.py
```

Navigate to `http://localhost:8501` in your browser.

## Advanced Usage

### Custom Model Architecture

```python
from src.models import DANNModel

# Create custom DANN model
model = DANNModel(
    num_classes=20,
    backbone="resnet50",
    pretrained=True,
    freeze_backbone=False,
    lambda_param=2.0,
    dropout=0.3
)
```

### Custom Training Loop

```python
from src.train import DomainAdaptationTrainer

# Create trainer
trainer = DomainAdaptationTrainer(
    model=model,
    device=device,
    config=training_config
)

# Custom training
history = trainer.train(
    source_loader=source_loader,
    target_loader=target_loader,
    val_loader=val_loader
)
```

### Evaluation

```python
from src.eval import DomainAdaptationEvaluator

# Create evaluator
evaluator = DomainAdaptationEvaluator(model, device)

# Comprehensive evaluation
results = evaluator.create_evaluation_report(
    source_test_loader=source_loader,
    target_test_loader=target_loader,
    save_dir="results"
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `training.batch_size=16`
   - Use gradient accumulation
   - Switch to ResNet18: `model.backbone=resnet18`

2. **Slow Training**
   - Enable mixed precision training
   - Increase number of workers: `training.num_workers=8`
   - Use smaller dataset for testing

3. **Poor Performance**
   - Increase training epochs: `training.max_epochs=200`
   - Adjust learning rate: `training.learning_rate=2e-4`
   - Try different lambda values: `training.lambda_domain=2.0`

### Device Issues

- **CUDA not available**: Automatically falls back to CPU
- **MPS issues**: Set `device=cpu` in config
- **Memory issues**: Reduce batch size or use gradient checkpointing

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Format code: `black src/ tests/`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{domain_adaptation_vision,
  title={Unsupervised Domain Adaptation for Vision},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Unsupervised-Domain-Adaptation-for-Vision}
}
```

## Acknowledgments

- Original DANN paper: Ganin et al., "Domain-Adversarial Training of Neural Networks"
- Original CORAL paper: Sun et al., "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
- PyTorch team for the excellent deep learning framework
- Hydra team for configuration management
# Unsupervised-Domain-Adaptation-for-Vision
