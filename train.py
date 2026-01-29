"""Main training script for domain adaptation experiments."""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
from pathlib import Path
import wandb
from typing import Dict, Any

from src.models import DANNModel, CORALModel, DomainAdaptationEnsemble
from src.data import create_toy_datasets, create_data_loaders
from src.train import DomainAdaptationTrainer
from src.eval import DomainAdaptationEvaluator
from src.utils import set_seed, get_device, setup_logging, create_directory_structure


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg: Hydra configuration
    """
    # Setup
    set_seed(cfg.seed)
    
    # Create directory structure
    create_directory_structure(".")
    
    # Setup logging
    logger = setup_logging(
        log_level=cfg.logging.level,
        log_file=cfg.logging.log_file
    )
    
    # Setup device
    if cfg.device == "auto":
        device = get_device()
    else:
        device = torch.device(cfg.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Setup wandb if enabled
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Create datasets
    logger.info("Creating datasets...")
    source_dataset, target_dataset = create_toy_datasets(
        num_source_samples=cfg.data.num_source_samples,
        num_target_samples=cfg.data.num_target_samples,
        num_classes=cfg.data.num_classes,
        image_size=cfg.data.image_size,
        augmentation=cfg.data.augmentation
    )
    
    # Create data loaders
    source_loader, target_loader = create_data_loaders(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )
    
    # Create validation loaders (use smaller subset)
    val_source_dataset, val_target_dataset = create_toy_datasets(
        num_source_samples=200,
        num_target_samples=200,
        num_classes=cfg.data.num_classes,
        image_size=cfg.data.image_size,
        augmentation=False  # No augmentation for validation
    )
    
    val_source_loader, val_target_loader = create_data_loaders(
        source_dataset=val_source_dataset,
        target_dataset=val_target_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False
    )
    
    # Create model
    logger.info("Creating model...")
    if cfg.model._target_ == "src.models.DANNModel":
        model = DANNModel(
            num_classes=cfg.model.num_classes,
            backbone=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            freeze_backbone=cfg.model.freeze_backbone,
            lambda_param=cfg.model.lambda_param,
            dropout=cfg.model.dropout
        )
    elif cfg.model._target_ == "src.models.CORALModel":
        model = CORALModel(
            num_classes=cfg.model.num_classes,
            backbone=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            freeze_backbone=cfg.model.freeze_backbone,
            dropout=cfg.model.dropout
        )
    else:
        raise ValueError(f"Unsupported model: {cfg.model._target_}")
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
    
    # Create trainer
    trainer = DomainAdaptationTrainer(
        model=model,
        device=device,
        config=cfg.training,
        logger=logger
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        source_loader=source_loader,
        target_loader=target_loader,
        val_loader=val_target_loader,  # Evaluate on target domain
        save_dir=cfg.paths.checkpoint_dir
    )
    
    # Log training history to wandb
    if cfg.logging.use_wandb:
        for epoch, (train_loss, train_acc, val_loss, val_acc) in enumerate(
            zip(history['train_loss'], history['train_task_acc'], 
                history['val_loss'], history['val_task_acc'])
        ):
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_task_acc': train_acc,
                'val_loss': val_loss,
                'val_task_acc': val_acc
            })
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = DomainAdaptationEvaluator(
        model=model,
        device=device,
        logger=logger
    )
    
    # Create evaluation report
    results = evaluator.create_evaluation_report(
        source_test_loader=val_source_loader,
        target_test_loader=val_target_loader,
        save_dir=cfg.paths.asset_dir
    )
    
    # Log final results
    logger.info("Final Results:")
    logger.info(f"Source Domain Accuracy: {results['summary']['source_accuracy']:.4f}")
    logger.info(f"Target Domain Accuracy: {results['summary']['target_accuracy']:.4f}")
    logger.info(f"Domain Classification Accuracy: {results['summary']['domain_accuracy']:.4f}")
    logger.info(f"MMD: {results['summary']['mmd']:.4f}")
    logger.info(f"CORAL Distance: {results['summary']['coral_distance']:.4f}")
    
    # Log to wandb
    if cfg.logging.use_wandb:
        wandb.log({
            'final_source_accuracy': results['summary']['source_accuracy'],
            'final_target_accuracy': results['summary']['target_accuracy'],
            'final_domain_accuracy': results['summary']['domain_accuracy'],
            'final_mmd': results['summary']['mmd'],
            'final_coral_distance': results['summary']['coral_distance']
        })
        wandb.finish()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
