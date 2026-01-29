"""Evaluation utilities for domain adaptation models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging


class DomainAdaptationEvaluator:
    """Evaluator for domain adaptation models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Evaluation device
            logger: Logger instance
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def evaluate_task_performance(
        self,
        data_loader: DataLoader,
        domain_name: str = "test"
    ) -> Dict[str, Any]:
        """Evaluate task performance on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            domain_name: Name of the domain being evaluated
            
        Returns:
            Dict containing evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_features = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    labels = batch['task_label'].to(self.device)
                else:
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                task_logits = outputs['task_logits']
                features = outputs.get('features', None)
                
                # Get predictions
                _, predictions = torch.max(task_logits, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if features is not None:
                    all_features.extend(features.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Classification report
        class_report = classification_report(
            all_labels, 
            all_predictions, 
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': all_predictions,
            'labels': all_labels,
            'features': np.array(all_features) if all_features else None
        }
        
        self.logger.info(f"{domain_name} domain task accuracy: {accuracy:.4f}")
        
        return metrics
    
    def evaluate_domain_classification(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader
    ) -> Dict[str, Any]:
        """Evaluate domain classification performance.
        
        Args:
            source_loader: Source domain data loader
            target_loader: Target domain data loader
            
        Returns:
            Dict containing domain classification metrics
        """
        if not isinstance(self.model, (nn.Module,)) or not hasattr(self.model, 'domain_classifier'):
            self.logger.warning("Model does not have domain classifier")
            return {}
        
        self.model.eval()
        
        all_domain_predictions = []
        all_domain_labels = []
        
        with torch.no_grad():
            # Evaluate on source domain
            for batch in source_loader:
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                else:
                    images, _ = batch
                    images = images.to(self.device)
                
                outputs = self.model(images)
                domain_logits = outputs.get('domain_logits', None)
                
                if domain_logits is not None:
                    _, domain_pred = torch.max(domain_logits, 1)
                    all_domain_predictions.extend(domain_pred.cpu().numpy())
                    all_domain_labels.extend([0] * images.size(0))  # Source = 0
            
            # Evaluate on target domain
            for batch in target_loader:
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                else:
                    images, _ = batch
                    images = images.to(self.device)
                
                outputs = self.model(images)
                domain_logits = outputs.get('domain_logits', None)
                
                if domain_logits is not None:
                    _, domain_pred = torch.max(domain_logits, 1)
                    all_domain_predictions.extend(domain_pred.cpu().numpy())
                    all_domain_labels.extend([1] * images.size(0))  # Target = 1
        
        if not all_domain_predictions:
            return {}
        
        # Calculate domain classification accuracy
        domain_accuracy = accuracy_score(all_domain_labels, all_domain_predictions)
        
        # Domain confusion matrix
        domain_cm = confusion_matrix(all_domain_labels, all_domain_predictions)
        
        metrics = {
            'domain_accuracy': domain_accuracy,
            'domain_confusion_matrix': domain_cm,
            'domain_predictions': all_domain_predictions,
            'domain_labels': all_domain_labels
        }
        
        self.logger.info(f"Domain classification accuracy: {domain_accuracy:.4f}")
        
        return metrics
    
    def compute_feature_alignment(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader
    ) -> Dict[str, float]:
        """Compute feature alignment metrics between domains.
        
        Args:
            source_loader: Source domain data loader
            target_loader: Target domain data loader
            
        Returns:
            Dict containing alignment metrics
        """
        self.model.eval()
        
        source_features = []
        target_features = []
        
        with torch.no_grad():
            # Extract source features
            for batch in source_loader:
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                else:
                    images, _ = batch
                    images = images.to(self.device)
                
                outputs = self.model(images)
                features = outputs.get('features', None)
                
                if features is not None:
                    source_features.extend(features.cpu().numpy())
            
            # Extract target features
            for batch in target_loader:
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                else:
                    images, _ = batch
                    images = images.to(self.device)
                
                outputs = self.model(images)
                features = outputs.get('features', None)
                
                if features is not None:
                    target_features.extend(features.cpu().numpy())
        
        if not source_features or not target_features:
            return {}
        
        source_features = np.array(source_features)
        target_features = np.array(target_features)
        
        # Compute alignment metrics
        metrics = {}
        
        # Maximum Mean Discrepancy (MMD)
        mmd = self._compute_mmd(source_features, target_features)
        metrics['mmd'] = mmd
        
        # CORAL distance
        coral_distance = self._compute_coral_distance(source_features, target_features)
        metrics['coral_distance'] = coral_distance
        
        # Feature statistics
        source_mean = np.mean(source_features, axis=0)
        target_mean = np.mean(target_features, axis=0)
        mean_distance = np.linalg.norm(source_mean - target_mean)
        metrics['mean_distance'] = mean_distance
        
        self.logger.info(f"Feature alignment - MMD: {mmd:.4f}, CORAL: {coral_distance:.4f}")
        
        return metrics
    
    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Maximum Mean Discrepancy between two feature sets.
        
        Args:
            X: First feature set
            Y: Second feature set
            
        Returns:
            float: MMD value
        """
        # Simple linear kernel MMD
        XX = np.mean(np.dot(X, X.T))
        YY = np.mean(np.dot(Y, Y.T))
        XY = np.mean(np.dot(X, Y.T))
        
        mmd = XX + YY - 2 * XY
        return mmd
    
    def _compute_coral_distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute CORAL distance between two feature sets.
        
        Args:
            X: First feature set
            Y: Second feature set
            
        Returns:
            float: CORAL distance
        """
        # Center features
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        Y_centered = Y - np.mean(Y, axis=0, keepdims=True)
        
        # Compute covariance matrices
        cov_X = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)
        cov_Y = np.dot(Y_centered.T, Y_centered) / (Y.shape[0] - 1)
        
        # Compute Frobenius norm of difference
        coral_distance = np.linalg.norm(cov_X - cov_Y, 'fro')
        
        return coral_distance
    
    def create_evaluation_report(
        self,
        source_test_loader: DataLoader,
        target_test_loader: DataLoader,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create comprehensive evaluation report.
        
        Args:
            source_test_loader: Source domain test loader
            target_test_loader: Target domain test loader
            save_dir: Directory to save results
            
        Returns:
            Dict containing comprehensive evaluation results
        """
        self.logger.info("Creating comprehensive evaluation report")
        
        # Evaluate task performance
        source_metrics = self.evaluate_task_performance(source_test_loader, "source")
        target_metrics = self.evaluate_task_performance(target_test_loader, "target")
        
        # Evaluate domain classification
        domain_metrics = self.evaluate_domain_classification(source_test_loader, target_test_loader)
        
        # Compute feature alignment
        alignment_metrics = self.compute_feature_alignment(source_test_loader, target_test_loader)
        
        # Compile results
        results = {
            'source_domain': source_metrics,
            'target_domain': target_metrics,
            'domain_classification': domain_metrics,
            'feature_alignment': alignment_metrics,
            'summary': {
                'source_accuracy': source_metrics.get('accuracy', 0.0),
                'target_accuracy': target_metrics.get('accuracy', 0.0),
                'domain_accuracy': domain_metrics.get('domain_accuracy', 0.0),
                'mmd': alignment_metrics.get('mmd', 0.0),
                'coral_distance': alignment_metrics.get('coral_distance', 0.0)
            }
        }
        
        # Save results
        if save_dir is not None:
            save_path = Path(save_dir) / "evaluation_results.json"
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Create visualizations
            self._create_visualizations(results, save_dir)
        
        return results
    
    def _create_visualizations(
        self,
        results: Dict[str, Any],
        save_dir: str
    ) -> None:
        """Create visualization plots.
        
        Args:
            results: Evaluation results
            save_dir: Directory to save plots
        """
        save_path = Path(save_dir)
        
        # Confusion matrices
        if 'confusion_matrix' in results['source_domain']:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Source confusion matrix
            sns.heatmap(
                results['source_domain']['confusion_matrix'],
                annot=True,
                fmt='d',
                ax=axes[0],
                cmap='Blues'
            )
            axes[0].set_title('Source Domain Confusion Matrix')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')
            
            # Target confusion matrix
            sns.heatmap(
                results['target_domain']['confusion_matrix'],
                annot=True,
                fmt='d',
                ax=axes[1],
                cmap='Reds'
            )
            axes[1].set_title('Target Domain Confusion Matrix')
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig(save_path / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Domain confusion matrix
        if 'domain_confusion_matrix' in results['domain_classification']:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                results['domain_classification']['domain_confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Purples'
            )
            plt.title('Domain Classification Confusion Matrix')
            plt.xlabel('Predicted Domain')
            plt.ylabel('Actual Domain')
            plt.tight_layout()
            plt.savefig(save_path / 'domain_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Summary metrics bar plot
        summary = results['summary']
        metrics_names = ['Source Accuracy', 'Target Accuracy', 'Domain Accuracy']
        metrics_values = [
            summary['source_accuracy'],
            summary['target_accuracy'],
            summary['domain_accuracy']
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_names, metrics_values, color=['blue', 'red', 'purple'])
        plt.title('Model Performance Summary')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path / 'performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
