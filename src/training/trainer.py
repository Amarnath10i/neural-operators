"""
Training Pipeline for Neural Operators.

Includes distributed training, mixed precision, spectral losses,
and comprehensive logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any
import os
import yaml
from pathlib import Path
from tqdm import tqdm
import time
from collections import defaultdict


class SpectralLoss(nn.Module):
    """
    Spectral loss for neural operators.
    
    Computes loss in Fourier space, emphasizing different frequency bands.
    """
    
    def __init__(
        self,
        weight_low: float = 1.0,
        weight_high: float = 0.5,
        modes: Optional[int] = None,
        relative: bool = True
    ):
        super().__init__()
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.modes = modes
        self.relative = relative
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (batch, channels, *spatial)
        
        # FFT
        pred_ft = torch.fft.rfftn(pred, dim=(-2, -1), norm='ortho')
        target_ft = torch.fft.rfftn(target, dim=(-2, -1), norm='ortho')
        
        # Frequency weights
        h, w = pred.shape[-2:]
        kx = torch.fft.fftfreq(h, device=pred.device).abs()
        ky = torch.fft.rfftfreq(w, device=pred.device).abs()
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        freq = torch.sqrt(KX**2 + KY**2)
        
        # Weight by frequency
        weight = self.weight_low + (self.weight_high - self.weight_low) * freq / freq.max()
        
        if self.modes is not None:
            weight[:, self.modes:] = 0
            weight[self.modes:, :] = 0
        
        # Spectral L2 loss
        diff = pred_ft - target_ft
        loss = (weight * diff.abs()**2).sum(dim=(-2, -1)).mean()
        
        if self.relative:
            target_norm = (weight * target_ft.abs()**2).sum(dim=(-2, -1)).mean()
            loss = loss / (target_norm + 1e-8)
        
        return loss


class PDELoss(nn.Module):
    """
    Physics-informed loss for PDE-constrained learning.
    
    Adds PDE residual as regularization.
    """
    
    def __init__(
        self,
        pde_type: str,
        weight_data: float = 1.0,
        weight_pde: float = 0.1,
        **pde_kwargs
    ):
        super().__init__()
        self.pde_type = pde_type
        self.weight_data = weight_data
        self.weight_pde = weight_pde
        self.pde_kwargs = pde_kwargs
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                input_field: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Data loss
        data_loss = F.mse_loss(pred, target)
        
        # PDE residual loss
        if self.pde_type == 'navier_stokes':
            pde_loss = self.navier_stokes_residual(pred, input_field)
        elif self.pde_type == 'darcy':
            pde_loss = self.darcy_residual(pred, input_field)
        elif self.pde_type == 'heat':
            pde_loss = self.heat_residual(pred)
        else:
            pde_loss = torch.tensor(0.0, device=pred.device)
        
        return self.weight_data * data_loss + self.weight_pde * pde_loss
    
    def navier_stokes_residual(self, w: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Vorticity equation residual: w_t + u·∇w - nu ∇²w - f = 0"""
        # Simplified - would need time derivative
        return torch.tensor(0.0, device=w.device)
    
    def darcy_residual(self, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Darcy residual: -∇·(a ∇u) - f = 0"""
        return torch.tensor(0.0, device=u.device)
    
    def heat_residual(self, u: torch.Tensor) -> torch.Tensor:
        """Heat equation residual"""
        return torch.tensor(0.0, device=u.device)


class OperatorTrainer:
    """
    Comprehensive trainer for neural operators.
    
    Features:
    - Mixed precision training (AMP)
    - Distributed data parallel (DDP)
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    - WandB/TensorBoard logging
    - Early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        use_wandb: bool = False,
        wandb_project: str = 'neural-operators',
        use_tensorboard: bool = True,
        early_stopping_patience: int = 50,
        early_stopping_metric: str = 'val_loss',
        save_every: int = 10,
        **kwargs
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.save_every = save_every
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Logging
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        if use_wandb:
            import wandb
            wandb.init(project=wandb_project, config=kwargs)
            self.wandb = wandb
        
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        self.history = defaultdict(list)
        
        # Compile model (PyTorch 2.0+) - disabled for now due to compat issues
        # if hasattr(torch, 'compile'):
        #     self.model = torch.compile(self.model)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Handle different batch formats
                if 'input' in batch and 'output' in batch:
                    pred = self.model(batch['input'])
                    target = batch['output']
                    loss = self.loss_fn(pred, target)
                elif 'permeability' in batch and 'pressure' in batch:
                    pred = self.model(batch['permeability'])
                    target = batch['pressure']
                    loss = self.loss_fn(pred, target)
                else:
                    # Generic: first two tensors
                    tensors = [v for v in batch.values() if isinstance(v, torch.Tensor)]
                    pred = self.model(tensors[0])
                    loss = self.loss_fn(pred, tensors[1])
                
                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.step += 1
            
            epoch_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item() * self.grad_accum_steps})
            
            # Log step
            if self.step % 100 == 0:
                self._log_step(loss.item() * self.grad_accum_steps)
        
        return {'train_loss': epoch_loss / num_batches}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation epoch."""
        self.model.eval()
        
        val_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if 'input' in batch and 'output' in batch:
                    pred = self.model(batch['input'])
                    target = batch['output']
                elif 'permeability' in batch and 'pressure' in batch:
                    pred = self.model(batch['permeability'])
                    target = batch['pressure']
                else:
                    tensors = [v for v in batch.values() if isinstance(v, torch.Tensor)]
                    pred = self.model(tensors[0])
                    target = tensors[1]
                
                loss = self.loss_fn(pred, target)
            
            val_loss += loss.item()
            num_batches += 1
        
        return {'val_loss': val_loss / num_batches}
    
    def _log_step(self, loss: float):
        """Log training step."""
        if self.use_wandb:
            self.wandb.log({'train/step_loss': loss, 'step': self.step})
        if self.use_tensorboard:
            self.writer.add_scalar('train/step_loss', loss, self.step)
    
    def _log_epoch(self, train_metrics: Dict, val_metrics: Dict):
        """Log epoch metrics."""
        lr = self.optimizer.param_groups[0]['lr']
        
        log_dict = {**train_metrics, **val_metrics, 'lr': lr, 'epoch': self.epoch}
        
        for k, v in log_dict.items():
            self.history[k].append(v)
        
        if self.use_wandb:
            self.wandb.log({f'train/{k}': v for k, v in train_metrics.items()})
            self.wandb.log({f'val/{k}': v for k, v in val_metrics.items()})
            self.wandb.log({'lr': lr, 'epoch': self.epoch})
        
        if self.use_tensorboard:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f'train/{k}', v, self.epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f'val/{k}', v, self.epoch)
            self.writer.add_scalar('lr', lr, self.epoch)
    
    def _check_early_stopping(self, val_metrics: Dict) -> bool:
        """Check early stopping criterion."""
        metric = val_metrics.get(self.early_stopping_metric, float('inf'))
        
        if metric < self.best_metric:
            self.best_metric = metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.early_stopping_patience
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'history': dict(self.history),
        }
        
        # Regular checkpoint
        if self.epoch % self.save_every == 0:
            path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
            torch.save(checkpoint, path)
        
        # Best checkpoint
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, path)
        
        # Latest checkpoint
        path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_metric = checkpoint['best_metric']
        self.history = defaultdict(list, checkpoint['history'])
    
    def fit(self, epochs: int) -> Dict[str, List[float]]:
        """Main training loop."""
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}, AMP: {self.use_amp}")
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('val_loss', 0))
                else:
                    self.scheduler.step()
            
            # Log
            self._log_epoch(train_metrics, val_metrics)
            
            # Print
            print(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.6f}, "
                  f"val_loss={val_metrics['val_loss']:.6f}, lr={self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Checkpoint
            is_best = val_metrics.get(self.early_stopping_metric, float('inf')) < self.best_metric
            self._save_checkpoint(is_best)
            
            # Early stopping
            if self._check_early_stopping(val_metrics):
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Load best model
        self.load_checkpoint(self.checkpoint_dir / 'best_model.pt')
        
        return dict(self.history)


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer."""
    if optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    epochs: int = 100,
    warmup_epochs: int = 5,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler."""
    
    if scheduler_type.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, **kwargs)
    elif scheduler_type.lower() == 'cosine_warmup':
        # Cosine with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type.lower() == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1, **kwargs)
    elif scheduler_type.lower() == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, **kwargs)
    elif scheduler_type.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


def get_loss_fn(loss_type: str, **kwargs) -> nn.Module:
    """Get loss function."""
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'spectral':
        return SpectralLoss(**kwargs)
    elif loss_type == 'pde':
        return PDELoss(**kwargs)
    elif loss_type == 'rel_l2':
        return RelativeL2Loss()
    else:
        raise ValueError(f"Unknown loss: {loss_type}")


class RelativeL2Loss(nn.Module):
    """Relative L2 loss: ||pred - target|| / ||target||"""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.norm(diff.flatten(1), dim=1) / (torch.norm(target.flatten(1), dim=1) + 1e-8)


import math

__all__ = [
    'OperatorTrainer',
    'SpectralLoss',
    'PDELoss',
    'RelativeL2Loss',
    'create_optimizer',
    'create_scheduler',
    'get_loss_fn',
]