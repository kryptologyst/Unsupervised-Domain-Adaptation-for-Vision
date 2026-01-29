"""Utility functions for domain adaptation project."""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import logging
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: Available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer for Domain-Adversarial Training.
    
    This layer reverses the gradient during backpropagation, allowing
    adversarial training of domain classifiers.
    """
    
    def __init__(self, lambda_param: float = 1.0):
        """Initialize the gradient reversal layer.
        
        Args:
            lambda_param: Gradient reversal strength parameter
        """
        super().__init__()
        self.lambda_param = lambda_param
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (identity function).
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Same as input
        """
        return x
    
    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass with gradient reversal.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            torch.Tensor: Reversed gradient
        """
        return -self.lambda_param * grad_output


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    **kwargs
) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: str,
    device: torch.device
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        filepath: Path to checkpoint file
        device: Device to load on
        
    Returns:
        Dict containing checkpoint data
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def create_directory_structure(base_path: str) -> None:
    """Create necessary directory structure.
    
    Args:
        base_path: Base project path
    """
    base_path = Path(base_path)
    directories = [
        'data', 'logs', 'checkpoints', 'assets', 'configs',
        'src/models', 'src/layers', 'src/data', 'src/utils',
        'src/train', 'src/eval', 'scripts', 'notebooks',
        'tests', 'demo'
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
