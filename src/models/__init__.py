"""Domain adaptation models implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from ..layers import (
    FeatureExtractor,
    TaskClassifier,
    DomainClassifier,
    GradientReversalLayer,
    CORALLayer
)


class DANNModel(nn.Module):
    """Domain-Adversarial Neural Network (DANN) for unsupervised domain adaptation."""
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        lambda_param: float = 1.0,
        dropout: float = 0.5
    ):
        """Initialize DANN model.
        
        Args:
            num_classes: Number of task classes
            backbone: Backbone architecture
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone
            lambda_param: Gradient reversal strength
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        
        self.task_classifier = TaskClassifier(
            input_dim=self.feature_extractor.feature_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.domain_classifier = DomainClassifier(
            input_dim=self.feature_extractor.feature_dim,
            num_domains=2,  # source and target
            dropout=dropout
        )
        
        self.gradient_reversal = GradientReversalLayer(lambda_param=lambda_param)
    
    def forward(
        self,
        x: torch.Tensor,
        alpha: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input images [B, C, H, W]
            alpha: Gradient reversal strength (if None, uses default)
            
        Returns:
            Dict containing task and domain predictions
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Task classification
        task_logits = self.task_classifier(features)
        
        # Domain classification with gradient reversal
        if alpha is not None:
            # Temporarily update lambda parameter
            self.gradient_reversal.lambda_param = alpha
        
        reversed_features = self.gradient_reversal(features)
        domain_logits = self.domain_classifier(reversed_features)
        
        return {
            'task_logits': task_logits,
            'domain_logits': domain_logits,
            'features': features
        }


class CORALModel(nn.Module):
    """CORAL (CORrelation ALignment) model for domain adaptation."""
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5
    ):
        """Initialize CORAL model.
        
        Args:
            num_classes: Number of task classes
            backbone: Backbone architecture
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        
        self.task_classifier = TaskClassifier(
            input_dim=self.feature_extractor.feature_dim,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.coral_layer = CORALLayer()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Dict containing task predictions and features
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Task classification
        task_logits = self.task_classifier(features)
        
        return {
            'task_logits': task_logits,
            'features': features
        }
    
    def compute_coral_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute CORAL loss between source and target features.
        
        Args:
            source_features: Source domain features
            target_features: Target domain features
            
        Returns:
            torch.Tensor: CORAL loss
        """
        return self.coral_layer(source_features, target_features)


class DomainAdaptationEnsemble(nn.Module):
    """Ensemble model combining multiple domain adaptation methods."""
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5,
        ensemble_methods: Tuple[str, ...] = ("dann", "coral")
    ):
        """Initialize ensemble model.
        
        Args:
            num_classes: Number of task classes
            backbone: Backbone architecture
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone
            dropout: Dropout probability
            ensemble_methods: Methods to ensemble
        """
        super().__init__()
        
        self.ensemble_methods = ensemble_methods
        self.models = nn.ModuleDict()
        
        if "dann" in ensemble_methods:
            self.models["dann"] = DANNModel(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                dropout=dropout
            )
        
        if "coral" in ensemble_methods:
            self.models["coral"] = CORALModel(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                dropout=dropout
            )
    
    def forward(
        self,
        x: torch.Tensor,
        method: Optional[str] = None,
        alpha: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input images
            method: Specific method to use (if None, uses ensemble)
            alpha: Gradient reversal strength for DANN
            
        Returns:
            Dict containing predictions
        """
        if method is not None and method in self.models:
            return self.models[method](x, alpha)
        
        # Ensemble prediction
        predictions = {}
        for method_name, model in self.models.items():
            pred = model(x, alpha)
            predictions[f"{method_name}_task_logits"] = pred["task_logits"]
            if "domain_logits" in pred:
                predictions[f"{method_name}_domain_logits"] = pred["domain_logits"]
        
        # Average task predictions
        task_logits_list = [
            pred for key, pred in predictions.items() 
            if key.endswith("_task_logits")
        ]
        ensemble_task_logits = torch.stack(task_logits_list).mean(dim=0)
        
        predictions["ensemble_task_logits"] = ensemble_task_logits
        
        return predictions
