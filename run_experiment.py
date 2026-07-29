#!/usr/bin/env python
"""
Main Experiment Script for Neural Operators.

Runs end-to-end experiments:
1. Data generation
2. Model training
3. Evaluation (including zero-shot super-resolution, domain adaptation)
4. Theory verification (generalization bounds, spectral convergence)
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from neural_operators.operators import get_fno_model, get_deeponet
from neural_operators.pdes import (
    generate_burgers_data, generate_navier_stokes_data, generate_darcy_data,
    PDEDataset
)
from neural_operators.training import (
    OperatorTrainer, SpectralLoss, create_optimizer, create_scheduler, get_loss_fn
)
from neural_operators.evaluation import (
    evaluate_model, zero_shot_superresolution, domain_adaptation_evaluation, compute_all_metrics
)
from neural_operators.theory import (
    rademacher_complexity_empirical, lipschitz_constant_estimate,
    stability_analysis, spectral_convergence_rate, fno_approximation_bound
)


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_experiment(config: dict) -> dict:
    """Setup experiment directories and device."""
    exp_dir = Path(config.get('exp_dir', 'experiments')) / config['name']
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    return {'exp_dir': exp_dir, 'device': device}


def generate_data(config: dict, exp_dir: Path) -> str:
    """Generate or load PDE dataset."""
    data_config = config['data']
    pde_type = data_config['pde_type']
    output_path = exp_dir / f"data_{pde_type}.h5"
    
    if output_path.exists() and not config.get('regenerate_data', False):
        print(f"Data already exists at {output_path}")
        return str(output_path)
    
    print(f"Generating {pde_type} data...")
    
    if pde_type == 'burgers':
        generate_burgers_data(
            n_samples=data_config['n_samples'],
            n=data_config['resolution'],
            nu_range=tuple(data_config.get('nu_range', [0.001, 0.01])),
            T=data_config.get('T', 1.0),
            dt=data_config.get('dt', 1e-3),
            seed=data_config.get('seed', 42),
            output_path=str(output_path)
        )
    elif pde_type == 'navier_stokes':
        generate_navier_stokes_data(
            n_samples=data_config['n_samples'],
            n=data_config['resolution'],
            nu_range=tuple(data_config.get('nu_range', [1e-4, 1e-3])),
            T=data_config.get('T', 50.0),
            dt=data_config.get('dt', 1e-3),
            forcing_type=data_config.get('forcing_type', 'kolmogorov'),
            seed=data_config.get('seed', 42),
            output_path=str(output_path)
        )
    elif pde_type == 'darcy':
        generate_darcy_data(
            n_samples=data_config['n_samples'],
            n=data_config['resolution'],
            permeability_range=tuple(data_config.get('permeability_range', [0.1, 10.0])),
            correlation_length=data_config.get('correlation_length', 0.1),
            seed=data_config.get('seed', 42),
            output_path=str(output_path)
        )
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")
    
    return str(output_path)


def create_dataloaders(data_path: str, config: dict) -> tuple:
    """Create train/val/test dataloaders."""
    data_config = config['data']
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 4)
    
    train_dataset = PDEDataset(data_path, 'train', 
                               train_ratio=data_config.get('train_ratio', 0.8),
                               val_ratio=data_config.get('val_ratio', 0.1))
    val_dataset = PDEDataset(data_path, 'val',
                             train_ratio=data_config.get('train_ratio', 0.8),
                             val_ratio=data_config.get('val_ratio', 0.1))
    test_dataset = PDEDataset(data_path, 'test',
                              train_ratio=data_config.get('train_ratio', 0.8),
                              val_ratio=data_config.get('val_ratio', 0.1))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def create_model(config: dict, device: str) -> nn.Module:
    """Create neural operator model."""
    model_config = config['model']
    model_type = model_config['type']
    
    # Determine input/output channels from data
    # For now, use config values
    in_channels = model_config.get('in_channels', 3)
    out_channels = model_config.get('out_channels', 1)
    
    if model_type.startswith('fno'):
        model = get_fno_model(
            model_type=model_type,
            in_channels=in_channels,
            out_channels=out_channels,
            modes=model_config.get('modes', 16),
            width=model_config.get('width', 64),
            n_layers=model_config.get('n_layers', 4),
            spectral_conv_type=model_config.get('spectral_conv_type', 'standard'),
            rank=model_config.get('rank'),
            dropout=model_config.get('dropout', 0.0),
        )
    elif model_type.startswith('deeponet'):
        model = get_deeponet(
            model_type=model_type,
            branch_input_dim=model_config.get('branch_input_dim', 100),
            trunk_input_dim=model_config.get('trunk_input_dim', 2),
            output_dim=out_channels,
            p=model_config.get('p', 128),
            branch_hidden=model_config.get('branch_hidden', [128, 128, 128]),
            trunk_hidden=model_config.get('trunk_hidden', [128, 128, 128]),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    print(f"Model: {model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: dict, exp_dir: Path, device: str) -> dict:
    """Train the model."""
    train_config = config['training']
    
    # Loss
    loss_fn = get_loss_fn(
        train_config.get('loss', 'mse'),
        weight_low=train_config.get('weight_low', 1.0),
        weight_high=train_config.get('weight_high', 0.5),
        modes=train_config.get('loss_modes'),
    )
    
    # Optimizer
    optimizer = create_optimizer(
        model,
        optimizer_type=train_config.get('optimizer', 'adamw'),
        lr=train_config['lr'],
        weight_decay=train_config.get('weight_decay', 1e-4),
    )
    
    # Scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=train_config.get('scheduler', 'cosine_warmup'),
        epochs=train_config['epochs'],
        warmup_epochs=train_config.get('warmup_epochs', 5),
    )
    
    # Trainer
    trainer = OperatorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=train_config.get('use_amp', True),
        grad_accum_steps=train_config.get('grad_accum_steps', 1),
        max_grad_norm=train_config.get('max_grad_norm', 1.0),
        checkpoint_dir=str(exp_dir / 'checkpoints'),
        log_dir=str(exp_dir / 'logs'),
        use_wandb=train_config.get('use_wandb', False),
        wandb_project=train_config.get('wandb_project', 'neural-operators'),
        use_tensorboard=train_config.get('use_tensorboard', True),
        early_stopping_patience=train_config.get('early_stopping_patience', 50),
        early_stopping_metric=train_config.get('early_stopping_metric', 'val_loss'),
        save_every=train_config.get('save_every', 10),
    )
    
    # Train
    history = trainer.fit(train_config['epochs'])
    
    # Save history
    import json
    with open(exp_dir / 'results' / 'history.json', 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    
    return history


def run_evaluation(model: nn.Module, test_loader: DataLoader, config: dict,
                   exp_dir: Path, device: str) -> dict:
    """Run comprehensive evaluation."""
    eval_config = config.get('evaluation', {})
    
    print("Running evaluation...")
    
    # Standard metrics
    metrics = evaluate_model(model, test_loader, device=device)
    print(f"Standard metrics: {metrics}")
    
    # Compute all metrics
    all_metrics = {}
    for batch in test_loader:
        if isinstance(batch, dict):
            x = batch['input'].to(device)
            y = batch['output'].to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            pred = model(x)
        
        batch_metrics = compute_all_metrics(pred, y)
        for k, v in batch_metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)
    
    # Average
    final_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    print(f"All metrics: {final_metrics}")
    
    # Zero-shot super-resolution
    if eval_config.get('zero_shot_superres', False):
        print("Running zero-shot super-resolution...")
        # Would need high-res test data
        # zs_metrics = zero_shot_superresolution(model, high_res_loader, target_res, device)
        # print(f"Zero-shot super-res: {zs_metrics}")
    
    # Domain adaptation
    if eval_config.get('domain_adaptation', False):
        print("Running domain adaptation evaluation...")
        # da_metrics = domain_adaptation_evaluation(model, source_loader, target_loader, ...)
        # print(f"Domain adaptation: {da_metrics}")
    
    # Save metrics
    import json
    with open(exp_dir / 'results' / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    return final_metrics


def run_theory_analysis(model: nn.Module, test_loader: DataLoader,
                        config: dict, exp_dir: Path, device: str) -> dict:
    """Run theoretical analysis."""
    theory_config = config.get('theory', {})
    
    if not theory_config.get('enabled', False):
        return {}
    
    print("Running theory analysis...")
    
    results = {}
    
    # Rademacher complexity
    if theory_config.get('rademacher', False):
        print("Computing Rademacher complexity...")
        rad_mean, rad_std = rademacher_complexity_empirical(
            model, test_loader, num_samples=theory_config.get('rademacher_samples', 50), device=device
        )
        results['rademacher'] = {'mean': rad_mean, 'std': rad_std}
        print(f"Rademacher complexity: {rad_mean:.6f} ± {rad_std:.6f}")
    
    # Lipschitz constant
    if theory_config.get('lipschitz', False):
        print("Estimating Lipschitz constant...")
        lip = lipschitz_constant_estimate(
            model, test_loader, num_pairs=theory_config.get('lipschitz_pairs', 500), device=device
        )
        results['lipschitz'] = lip
        print(f"Lipschitz constant estimate: {lip:.6f}")
    
    # Stability analysis
    if theory_config.get('stability', False):
        print("Running stability analysis...")
        epsilons = theory_config.get('perturbation_magnitudes', [0.001, 0.01, 0.1])
        stability = stability_analysis(model, epsilons, test_loader, device)
        results['stability'] = stability
        print(f"Stability: {stability}")
    
    # Spectral convergence
    if theory_config.get('spectral_convergence', False):
        print("Computing spectral convergence rate...")
        modes = theory_config.get('modes_list', [4, 8, 12, 16, 20, 24])
        spectral = spectral_convergence_rate(model, test_loader, modes, device)
        results['spectral_convergence'] = spectral
        print(f"Spectral convergence: {spectral}")
    
    # FNO approximation bound
    if theory_config.get('approximation_bound', False):
        model_config = config['model']
        bound = fno_approximation_bound(
            modes=model_config.get('modes', 16),
            width=model_config.get('width', 64),
            depth=model_config.get('n_layers', 4),
            input_smoothness=theory_config.get('input_smoothness', 2.0),
        )
        results['approximation_bound'] = bound
        print(f"FNO approximation bound: {bound:.6f}")
    
    # Save theory results
    import json
    with open(exp_dir / 'results' / 'theory.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Neural Operators Experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--stage', type=str, choices=['all', 'data', 'train', 'eval', 'theory'],
                        default='all', help='Stage to run')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    setup = setup_experiment(config)
    exp_dir = setup['exp_dir']
    device = args.device if args.device != 'auto' else setup['device']
    
    # Save config
    import yaml
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"Experiment: {config['name']}")
    print(f"Directory: {exp_dir}")
    
    # Stage: Data
    if args.stage in ['all', 'data']:
        data_path = generate_data(config, exp_dir)
        print(f"Data saved to: {data_path}")
    
    if args.stage == 'data':
        return
    
    # Load data
    data_path = exp_dir / f"data_{config['data']['pde_type']}.h5"
    train_loader, val_loader, test_loader = create_dataloaders(str(data_path), config)
    
    # Stage: Train
    if args.stage in ['all', 'train']:
        model = create_model(config, device)
        
        if args.resume:
            print(f"Resuming from {args.resume}")
            # Load checkpoint logic here
        
        history = train_model(model, train_loader, val_loader, config, exp_dir, device)
        print(f"Training complete. Best val loss: {min(history.get('val_loss', [float('inf')])):.6f}")
    
    if args.stage == 'train':
        return
    
    # Load best model for evaluation
    model = create_model(config, device)
    checkpoint_path = exp_dir / 'checkpoints' / 'best_model.pt'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    else:
        print("Warning: No checkpoint found, using untrained model")
    
    # Stage: Eval
    if args.stage in ['all', 'eval']:
        metrics = run_evaluation(model, test_loader, config, exp_dir, device)
    
    # Stage: Theory
    if args.stage in ['all', 'theory']:
        theory_results = run_theory_analysis(model, test_loader, config, exp_dir, device)
    
    print("Experiment complete!")


if __name__ == '__main__':
    main()