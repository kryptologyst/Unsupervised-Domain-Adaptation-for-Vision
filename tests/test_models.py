"""Tests for domain adaptation models."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock

from src.models import DANNModel, CORALModel, DomainAdaptationEnsemble
from src.layers import GradientReversalLayer, DomainClassifier, FeatureExtractor
from src.data import ToyDomainDataset, create_toy_datasets
from src.utils import set_seed, get_device


class TestModels:
    """Test cases for domain adaptation models."""
    
    def test_dann_model_creation(self):
        """Test DANN model creation."""
        model = DANNModel(
            num_classes=10,
            backbone="resnet18",  # Use smaller model for testing
            pretrained=False,
            freeze_backbone=False,
            lambda_param=1.0,
            dropout=0.5
        )
        
        assert isinstance(model, DANNModel)
        assert model.feature_extractor.feature_dim == 512  # ResNet18 feature dim
        assert model.task_classifier.classifier[-1].out_features == 10
        assert model.domain_classifier.classifier[-1].out_features == 2
    
    def test_coral_model_creation(self):
        """Test CORAL model creation."""
        model = CORALModel(
            num_classes=10,
            backbone="resnet18",
            pretrained=False,
            freeze_backbone=False,
            dropout=0.5
        )
        
        assert isinstance(model, CORALModel)
        assert model.feature_extractor.feature_dim == 512
        assert model.task_classifier.classifier[-1].out_features == 10
    
    def test_dann_forward_pass(self):
        """Test DANN forward pass."""
        model = DANNModel(
            num_classes=10,
            backbone="resnet18",
            pretrained=False,
            freeze_backbone=False
        )
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        outputs = model(input_tensor)
        
        assert 'task_logits' in outputs
        assert 'domain_logits' in outputs
        assert 'features' in outputs
        
        assert outputs['task_logits'].shape == (batch_size, 10)
        assert outputs['domain_logits'].shape == (batch_size, 2)
        assert outputs['features'].shape == (batch_size, 512)
    
    def test_coral_forward_pass(self):
        """Test CORAL forward pass."""
        model = CORALModel(
            num_classes=10,
            backbone="resnet18",
            pretrained=False,
            freeze_backbone=False
        )
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        outputs = model(input_tensor)
        
        assert 'task_logits' in outputs
        assert 'features' in outputs
        
        assert outputs['task_logits'].shape == (batch_size, 10)
        assert outputs['features'].shape == (batch_size, 512)
    
    def test_coral_loss_computation(self):
        """Test CORAL loss computation."""
        model = CORALModel(
            num_classes=10,
            backbone="resnet18",
            pretrained=False,
            freeze_backbone=False
        )
        
        batch_size = 4
        source_features = torch.randn(batch_size, 512)
        target_features = torch.randn(batch_size, 512)
        
        coral_loss = model.compute_coral_loss(source_features, target_features)
        
        assert isinstance(coral_loss, torch.Tensor)
        assert coral_loss.item() >= 0  # Loss should be non-negative


class TestLayers:
    """Test cases for custom layers."""
    
    def test_gradient_reversal_layer(self):
        """Test gradient reversal layer."""
        layer = GradientReversalLayer(lambda_param=2.0)
        
        input_tensor = torch.randn(4, 10, requires_grad=True)
        output = layer(input_tensor)
        
        # Forward pass should be identity
        assert torch.allclose(output, input_tensor)
        
        # Test gradient reversal
        loss = output.sum()
        loss.backward()
        
        # Gradient should be reversed
        assert torch.allclose(input_tensor.grad, -2.0 * torch.ones_like(input_tensor))
    
    def test_domain_classifier(self):
        """Test domain classifier."""
        classifier = DomainClassifier(
            input_dim=512,
            hidden_dim=256,
            num_domains=2,
            dropout=0.5
        )
        
        batch_size = 4
        input_features = torch.randn(batch_size, 512)
        
        output = classifier(input_features)
        
        assert output.shape == (batch_size, 2)
    
    def test_feature_extractor(self):
        """Test feature extractor."""
        extractor = FeatureExtractor(
            backbone="resnet18",
            pretrained=False,
            freeze_backbone=False
        )
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        features = extractor(input_tensor)
        
        assert features.shape == (batch_size, 512)


class TestData:
    """Test cases for data handling."""
    
    def test_toy_dataset_creation(self):
        """Test toy dataset creation."""
        dataset = ToyDomainDataset(
            domain="source",
            num_samples=100,
            image_size=224,
            num_classes=10
        )
        
        assert len(dataset) == 100
        
        # Test data loading
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert 0 <= label < 10
    
    def test_toy_datasets_creation(self):
        """Test creation of both source and target datasets."""
        source_dataset, target_dataset = create_toy_datasets(
            num_source_samples=50,
            num_target_samples=50,
            num_classes=10,
            image_size=224,
            augmentation=False
        )
        
        assert len(source_dataset) == 50
        assert len(target_dataset) == 50
        
        # Test that datasets have different characteristics
        source_img, _ = source_dataset[0]
        target_img, _ = target_dataset[0]
        
        # Images should be different (different domains)
        assert not torch.allclose(source_img, target_img)


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand1 = torch.randn(5)
        np_rand1 = np.random.randn(5)
        
        # Reset seed and generate again
        set_seed(42)
        torch_rand2 = torch.randn(5)
        np_rand2 = np.random.randn(5)
        
        # Should be the same
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'mps', 'cpu']


if __name__ == "__main__":
    pytest.main([__file__])
