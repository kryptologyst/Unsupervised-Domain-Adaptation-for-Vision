"""Core neural network layers for domain adaptation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GradientReversalFunction(torch.autograd.Function):
    """Custom autograd function for gradient reversal."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_param: float) -> torch.Tensor:
        """Forward pass - identity function.
        
        Args:
            ctx: Context for saving tensors
            x: Input tensor
            lambda_param: Gradient reversal strength
            
        Returns:
            torch.Tensor: Same as input
        """
        ctx.lambda_param = lambda_param
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass with gradient reversal.
        
        Args:
            ctx: Context containing saved tensors
            grad_output: Gradient from next layer
            
        Returns:
            Tuple of gradients (reversed input grad, None for lambda_param)
        """
        return -ctx.lambda_param * grad_output, None


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
        """Forward pass with gradient reversal.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Same as input but with reversed gradients
        """
        return GradientReversalFunction.apply(x, self.lambda_param)


class DomainClassifier(nn.Module):
    """Domain classifier for adversarial training."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        num_domains: int = 2,
        dropout: float = 0.5
    ):
        """Initialize domain classifier.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_domains: Number of domains to classify
            dropout: Dropout probability
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            torch.Tensor: Domain classification logits
        """
        return self.classifier(x)


class FeatureExtractor(nn.Module):
    """Feature extractor based on ResNet backbone."""
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """Initialize feature extractor.
        
        Args:
            backbone: Backbone architecture name
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
        """
        super().__init__()
        
        if backbone == "resnet50":
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == "resnet18":
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=pretrained)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            torch.Tensor: Extracted features [B, feature_dim]
        """
        features = self.backbone(x)
        return features.view(features.size(0), -1)


class TaskClassifier(nn.Module):
    """Task-specific classifier."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.5
    ):
        """Initialize task classifier.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of task classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            torch.Tensor: Task classification logits
        """
        return self.classifier(x)


class CORALLayer(nn.Module):
    """CORAL (CORrelation ALignment) layer for domain adaptation."""
    
    def __init__(self):
        """Initialize CORAL layer."""
        super().__init__()
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute CORAL loss between source and target features.
        
        Args:
            source_features: Source domain features [B, D]
            target_features: Target domain features [B, D]
            
        Returns:
            torch.Tensor: CORAL loss value
        """
        # Compute covariance matrices
        source_cov = self._compute_covariance(source_features)
        target_cov = self._compute_covariance(target_features)
        
        # Compute Frobenius norm of difference
        coral_loss = F.mse_loss(source_cov, target_cov)
        
        return coral_loss
    
    def _compute_covariance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix of input features.
        
        Args:
            x: Input features [B, D]
            
        Returns:
            torch.Tensor: Covariance matrix [D, D]
        """
        # Center the features
        x_centered = x - x.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov = torch.mm(x_centered.t(), x_centered) / (x.size(0) - 1)
        
        return cov
