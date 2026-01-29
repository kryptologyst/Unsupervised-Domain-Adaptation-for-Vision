"""Training utilities for domain adaptation models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import time

from ..models import DANNModel, CORALModel, DomainAdaptationEnsemble
from ..utils import save_checkpoint, load_checkpoint, get_device


class DomainAdaptationTrainer:
    """Trainer for domain adaptation models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            device: Training device
            config: Training configuration
            logger: Logger instance
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize optimizers
        self.task_optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.task_optimizer,
            step_size=config.get('lr_step_size', 30),
            gamma=config.get('lr_gamma', 0.1)
        )
        
        # Loss functions
        self.task_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_task_acc': [],
            'train_domain_acc': [],
            'val_loss': [],
            'val_task_acc': [],
            'val_domain_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            source_loader: Source domain data loader
            target_loader: Target domain data loader
            epoch: Current epoch number
            
        Returns:
            Dict containing training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_task_loss = 0.0
        total_domain_loss = 0.0
        task_correct = 0
        domain_correct = 0
        total_samples = 0
        
        # Calculate gradient reversal strength
        p = float(epoch) / self.config.get('max_epochs', 100)
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        
        # Create iterators
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        num_batches = min(len(source_loader), len(target_loader))
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")
        
        for batch_idx in pbar:
            try:
                # Get batches
                source_batch = next(source_iter)
                target_batch = next(target_iter)
                
                # Handle different data formats
                if isinstance(source_batch, dict):
                    source_images = source_batch['image'].to(self.device)
                    source_labels = source_batch['task_label'].to(self.device)
                    source_domains = source_batch['domain_label'].to(self.device)
                else:
                    source_images, source_labels = source_batch
                    source_images = source_images.to(self.device)
                    source_labels = source_labels.to(self.device)
                    source_domains = torch.zeros(source_images.size(0), dtype=torch.long, device=self.device)
                
                if isinstance(target_batch, dict):
                    target_images = target_batch['image'].to(self.device)
                    target_labels = target_batch['task_label'].to(self.device)
                    target_domains = target_batch['domain_label'].to(self.device)
                else:
                    target_images, target_labels = target_batch
                    target_images = target_images.to(self.device)
                    target_labels = target_labels.to(self.device)
                    target_domains = torch.ones(target_images.size(0), dtype=torch.long, device=self.device)
                
                # Combine batches
                batch_size = source_images.size(0) + target_images.size(0)
                images = torch.cat([source_images, target_images], dim=0)
                task_labels = torch.cat([source_labels, target_labels], dim=0)
                domain_labels = torch.cat([source_domains, target_domains], dim=0)
                
                # Forward pass
                self.task_optimizer.zero_grad()
                
                if isinstance(self.model, DANNModel):
                    outputs = self.model(images, alpha=alpha)
                    task_logits = outputs['task_logits']
                    domain_logits = outputs['domain_logits']
                    
                    # Compute losses
                    task_loss = self.task_criterion(task_logits, task_labels)
                    domain_loss = self.domain_criterion(domain_logits, domain_labels)
                    
                    # Total loss
                    lambda_domain = self.config.get('lambda_domain', 1.0)
                    total_loss_batch = task_loss + lambda_domain * domain_loss
                    
                elif isinstance(self.model, CORALModel):
                    outputs = self.model(images)
                    task_logits = outputs['task_logits']
                    
                    # Compute task loss
                    task_loss = self.task_criterion(task_logits, task_labels)
                    
                    # Compute CORAL loss
                    source_features = outputs['features'][:source_images.size(0)]
                    target_features = outputs['features'][source_images.size(0):]
                    coral_loss = self.model.compute_coral_loss(source_features, target_features)
                    
                    # Total loss
                    lambda_coral = self.config.get('lambda_coral', 1.0)
                    total_loss_batch = task_loss + lambda_coral * coral_loss
                    domain_loss = torch.tensor(0.0, device=self.device)
                    
                else:
                    raise ValueError(f"Unsupported model type: {type(self.model)}")
                
                # Backward pass
                total_loss_batch.backward()
                self.task_optimizer.step()
                
                # Update metrics
                total_loss += total_loss_batch.item()
                total_task_loss += task_loss.item()
                total_domain_loss += domain_loss.item()
                
                # Calculate accuracies
                _, task_pred = torch.max(task_logits, 1)
                task_correct += (task_pred == task_labels).sum().item()
                
                if isinstance(self.model, DANNModel):
                    _, domain_pred = torch.max(domain_logits, 1)
                    domain_correct += (domain_pred == domain_labels).sum().item()
                
                total_samples += batch_size
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{total_loss_batch.item():.4f}",
                    'Task': f"{task_loss.item():.4f}",
                    'Domain': f"{domain_loss.item():.4f}"
                })
                
            except StopIteration:
                break
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        avg_task_loss = total_task_loss / num_batches
        avg_domain_loss = total_domain_loss / num_batches
        task_acc = task_correct / total_samples
        domain_acc = domain_correct / total_samples if isinstance(self.model, DANNModel) else 0.0
        
        metrics = {
            'loss': avg_loss,
            'task_loss': avg_task_loss,
            'domain_loss': avg_domain_loss,
            'task_acc': task_acc,
            'domain_acc': domain_acc
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dict containing validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_task_loss = 0.0
        total_domain_loss = 0.0
        task_correct = 0
        domain_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    task_labels = batch['task_label'].to(self.device)
                    domain_labels = batch['domain_label'].to(self.device)
                else:
                    images, task_labels = batch
                    images = images.to(self.device)
                    task_labels = task_labels.to(self.device)
                    domain_labels = torch.zeros(images.size(0), dtype=torch.long, device=self.device)
                
                batch_size = images.size(0)
                
                # Forward pass
                if isinstance(self.model, DANNModel):
                    outputs = self.model(images)
                    task_logits = outputs['task_logits']
                    domain_logits = outputs['domain_logits']
                    
                    # Compute losses
                    task_loss = self.task_criterion(task_logits, task_labels)
                    domain_loss = self.domain_criterion(domain_logits, domain_labels)
                    
                    lambda_domain = self.config.get('lambda_domain', 1.0)
                    total_loss_batch = task_loss + lambda_domain * domain_loss
                    
                elif isinstance(self.model, CORALModel):
                    outputs = self.model(images)
                    task_logits = outputs['task_logits']
                    
                    task_loss = self.task_criterion(task_logits, task_labels)
                    total_loss_batch = task_loss
                    domain_loss = torch.tensor(0.0, device=self.device)
                
                # Update metrics
                total_loss += total_loss_batch.item()
                total_task_loss += task_loss.item()
                total_domain_loss += domain_loss.item()
                
                # Calculate accuracies
                _, task_pred = torch.max(task_logits, 1)
                task_correct += (task_pred == task_labels).sum().item()
                
                if isinstance(self.model, DANNModel):
                    _, domain_pred = torch.max(domain_logits, 1)
                    domain_correct += (domain_pred == domain_labels).sum().item()
                
                total_samples += batch_size
        
        # Calculate validation metrics
        avg_loss = total_loss / len(val_loader)
        avg_task_loss = total_task_loss / len(val_loader)
        avg_domain_loss = total_domain_loss / len(val_loader)
        task_acc = task_correct / total_samples
        domain_acc = domain_correct / total_samples if isinstance(self.model, DANNModel) else 0.0
        
        metrics = {
            'loss': avg_loss,
            'task_loss': avg_task_loss,
            'domain_loss': avg_domain_loss,
            'task_acc': task_acc,
            'domain_acc': domain_acc
        }
        
        return metrics
    
    def train(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            source_loader: Source domain data loader
            target_loader: Target domain data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
            
        Returns:
            Dict containing training history
        """
        max_epochs = self.config.get('max_epochs', 100)
        
        self.logger.info(f"Starting training for {max_epochs} epochs")
        
        for epoch in range(max_epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(source_loader, target_loader, epoch)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            else:
                val_metrics = {'loss': 0.0, 'task_acc': 0.0, 'domain_acc': 0.0}
            
            # Update learning rate
            self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_task_acc'].append(train_metrics['task_acc'])
            self.history['train_domain_acc'].append(train_metrics['domain_acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_task_acc'].append(val_metrics['task_acc'])
            self.history['val_domain_acc'].append(val_metrics['domain_acc'])
            
            # Log metrics
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch+1}/{max_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Task Acc: {train_metrics['task_acc']:.4f}, "
                f"Val Task Acc: {val_metrics['task_acc']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_metrics['task_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['task_acc']
                self.best_epoch = epoch
                
                if save_dir is not None:
                    save_path = Path(save_dir) / "best_model.pth"
                    save_checkpoint(
                        self.model,
                        self.task_optimizer,
                        epoch,
                        val_metrics['loss'],
                        str(save_path),
                        val_acc=val_metrics['task_acc']
                    )
        
        self.logger.info(f"Training completed. Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        
        return self.history
