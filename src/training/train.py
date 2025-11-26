"""
Main training script for flood segmentation model.

This script handles the complete training pipeline including:
- Loading configuration
- Creating dataloaders
- Model initialization
- Training loop with validation
- Checkpointing and early stopping
- Logging
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import create_dataloaders
from src.models.unet import create_model, get_model_info
from src.training.metrics import MetricsCalculator
from src.training.losses import get_loss_function


class Trainer:
    """
    Trainer class for flood segmentation model.
    """

    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = torch.device(
            self.config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")

        # Setup directories
        self.setup_directories()

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.dataloaders = None

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.patience_counter = 0

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }

    def setup_directories(self):
        """Create necessary directories for saving outputs."""
        self.checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        self.log_dir = Path(self.config['logging']['log_dir'])

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.log_dir / 'samples').mkdir(exist_ok=True)

    def prepare_data(self):
        """Prepare dataloaders."""
        print("\nPreparing data...")

        data_config = self.config['data']

        self.dataloaders = create_dataloaders(
            train_csv='outputs/train_split.csv',
            val_csv='outputs/val_split.csv',
            test_csv='outputs/test_split.csv',
            batch_size=self.config['training']['batch_size'],
            num_workers=data_config['num_workers'],
            image_size=data_config['image_size'],
            augmentation_config=data_config.get('augmentation'),
            normalize=data_config['normalize'],
            mean=data_config.get('norm_mean'),
            std=data_config.get('norm_std'),
            pin_memory=data_config['pin_memory']
        )

        print(f"Train batches: {len(self.dataloaders['train'])}")
        print(f"Val batches: {len(self.dataloaders['val'])}")
        if 'test' in self.dataloaders:
            print(f"Test batches: {len(self.dataloaders['test'])}")

    def build_model(self):
        """Build and initialize model."""
        print("\nBuilding model...")

        model_config = self.config['model']

        self.model = create_model(
            model_name=model_config['name'],
            encoder_name=model_config['encoder'],
            encoder_weights=model_config['encoder_weights'],
            in_channels=model_config['in_channels'],
            classes=model_config['classes'],
            activation=model_config['activation']
        )

        self.model = self.model.to(self.device)

        # Print model info
        info = get_model_info(self.model)
        print(f"Model: {model_config['name']} with {model_config['encoder']} encoder")
        print(f"Total parameters: {info['total_params_M']:.2f}M")
        print(f"Trainable parameters: {info['trainable_params_M']:.2f}M")

    def setup_training(self):
        """Setup optimizer, scheduler, and loss function."""
        print("\nSetting up training...")

        train_config = self.config['training']

        # Setup optimizer
        optimizer_name = train_config['optimizer']
        lr = train_config['learning_rate']
        weight_decay = train_config.get('weight_decay', 0.0)

        if optimizer_name == 'Adam':
            self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        print(f"Optimizer: {optimizer_name} (lr={lr})")

        # Setup scheduler
        scheduler_config = train_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau')

        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config.get('mode', 'max'),
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-6),
                verbose=True
            )
        elif scheduler_type == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )

        print(f"Scheduler: {scheduler_type}")

        # Setup loss function
        loss_config = train_config.get('loss', {'type': 'hybrid'})
        self.criterion = get_loss_function(loss_config)
        print(f"Loss function: {loss_config['type']}")

        # Setup mixed precision training
        self.use_amp = train_config.get('mixed_precision', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision training: Enabled")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        running_loss = 0.0
        progress_bar = tqdm(self.dataloaders['train'], desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

            # Update metrics
            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = running_loss / len(self.dataloaders['train'])
        return avg_loss

    def validate(self):
        """Validate the model."""
        self.model.eval()

        running_loss = 0.0
        metrics_calculator = MetricsCalculator(
            metric_names=self.config['metrics'],
            threshold=0.5
        )

        with torch.no_grad():
            for batch in tqdm(self.dataloaders['val'], desc="Validation"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                # Convert outputs to probabilities
                probs = torch.sigmoid(outputs)

                # Update metrics
                running_loss += loss.item()
                metrics_calculator.update(probs, masks)

        avg_loss = running_loss / len(self.dataloaders['val'])
        avg_metrics = metrics_calculator.compute()

        return avg_loss, avg_metrics

    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config,
            'history': self.history
        }

        # Save last checkpoint
        if self.config['checkpoint']['save_last']:
            last_path = self.checkpoint_dir / 'last_checkpoint.pth'
            torch.save(checkpoint, last_path)

        # Save best checkpoint
        if is_best and self.config['checkpoint']['save_best']:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved (metric: {self.best_metric:.4f})")

    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)

        epochs = self.config['training']['epochs']
        early_stopping_patience = self.config['training']['early_stopping_patience']
        monitor_metric = self.config['checkpoint']['monitor_metric'].replace('val_', '')

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_metrics = self.validate()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Metrics:")
            for metric_name, value in val_metrics.items():
                print(f"    {metric_name}: {value:.4f}")

            # Check if best model
            current_metric = val_metrics[monitor_metric]
            is_best = current_metric > self.best_metric

            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1

            # Save last checkpoint
            self.save_checkpoint(is_best=False)

            # Update learning rate
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(current_metric)
            else:
                self.scheduler.step()

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        print("\n" + "="*60)
        print("Training Completed!")
        print(f"Best {monitor_metric}: {self.best_metric:.4f}")
        print("="*60)

        # Save final history
        self.save_history()

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.log_dir / 'training_history.json'

        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining history saved to {history_path}")

    def run(self):
        """Run the complete training pipeline."""
        try:
            # Prepare data
            self.prepare_data()

            # Build model
            self.build_model()

            # Setup training
            self.setup_training()

            # Train
            self.train()

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            self.save_checkpoint(is_best=False)
            self.save_history()

        except Exception as e:
            print(f"\n\nError during training: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train flood segmentation model')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                        help='Path to configuration file')

    args = parser.parse_args()

    # Create trainer and run
    trainer = Trainer(args.config)
    trainer.run()


if __name__ == "__main__":
    main()
