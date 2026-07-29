"""
Experiment Configurations for Neural Operators.

YAML configs for reproducible experiments.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


# Default experiment configurations
EXPERIMENTS = {
    'fno_navier_stokes': {
        'model': {
            'type': 'fno2d',
            'modes': 16,
            'width': 64,
            'n_layers': 4,
            'in_channels': 3,  # vorticity, forcing, time
            'out_channels': 1
        },
        'data': {
            'pde': 'navier_stokes',
            'n_samples': 1000,
            'resolution': 64,
            'nu_range': [1e-4, 1e-3],
            'T': 50.0,
            'dt': 1e-3,
            'save_every': 1000,
            'forcing_type': 'kolmogorov',
            'split': [0.8, 0.1, 0.1]
        },
        'training': {
            'batch_size': 32,
            'epochs': 500,
            'optimizer': 'adamw',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'scheduler': 'cosine_warmup',
            'warmup_epochs': 10,
            'grad_accum_steps': 1,
            'max_grad_norm': 1.0,
            'use_amp': True,
            'early_stopping_patience': 50,
            'early_stopping_metric': 'val_loss'
        },
        'loss': {
            'type': 'spectral',
            'weight_low': 1.0,
            'weight_high': 0.5
        },
        'evaluation': {
            'superresolution': True,
            'target_resolutions': [128, 256, 512],
            'domain_adaptation': False
        },
        'logging': {
            'use_wandb': True,
            'wandb_project': 'neural-operators',
            'use_tensorboard': True,
            'log_dir': 'logs/fno_ns',
            'checkpoint_dir': 'checkpoints/fno_ns',
            'save_every': 10
        }
    },
    
    'deeponet_darcy': {
        'model': {
            'type': 'deeponet',
            'branch_input_dim': 85 * 85,  # permeability field flattened
            'trunk_input_dim': 2,  # (x, y) coordinates
            'output_dim': 1,
            'branch_hidden': [128, 128, 128],
            'trunk_hidden': [128, 128, 128],
            'p': 128,
            'activation': 'gelu'
        },
        'data': {
            'pde': 'darcy',
            'n_samples': 1000,
            'resolution': 85,
            'permeability_range': [0.1, 10.0],
            'correlation_length': 0.1,
            'split': [0.8, 0.1, 0.1]
        },
        'training': {
            'batch_size': 64,
            'epochs': 300,
            'optimizer': 'adamw',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'scheduler': 'cosine_warmup',
            'warmup_epochs': 10,
            'grad_accum_steps': 1,
            'max_grad_norm': 1.0,
            'use_amp': True,
            'early_stopping_patience': 30,
            'early_stopping_metric': 'val_loss'
        },
        'loss': {
            'type': 'mse'
        },
        'evaluation': {
            'superresolution': True,
            'target_resolutions': [171, 341],
            'domain_adaptation': True
        },
        'logging': {
            'use_wandb': True,
            'wandb_project': 'neural-operators',
            'use_tensorboard': True,
            'log_dir': 'logs/deeponet_darcy',
            'checkpoint_dir': 'checkpoints/deeponet_darcy',
            'save_every': 10
        }
    },
    
    'gno_airfoil': {
        'model': {
            'type': 'gno',
            'in_channels': 3,  # position + boundary condition
            'out_channels': 2,  # velocity field
            'hidden_channels': 64,
            'n_layers': 6,
            'modes': 16,
            'edge_attr_dim': 1,
            'lifting_channels': 128,
            'projection_channels': 128
        },
        'data': {
            'pde': 'airfoil',
            'n_samples': 500,
            'mesh_type': 'unstructured',
            'split': [0.8, 0.1, 0.1]
        },
        'training': {
            'batch_size': 8,
            'epochs': 200,
            'optimizer': 'adamw',
            'lr': 5e-4,
            'weight_decay': 1e-4,
            'scheduler': 'cosine_warmup',
            'warmup_epochs': 10,
            'grad_accum_steps': 2,
            'max_grad_norm': 1.0,
            'use_amp': True,
            'early_stopping_patience': 30,
            'early_stopping_metric': 'val_loss'
        },
        'loss': {
            'type': 'mse'
        },
        'evaluation': {
            'superresolution': False,
            'domain_adaptation': True
        },
        'logging': {
            'use_wandb': True,
            'wandb_project': 'neural-operators',
            'use_tensorboard': True,
            'log_dir': 'logs/gno_airfoil',
            'checkpoint_dir': 'checkpoints/gno_airfoil',
            'save_every': 10
        }
    },
    
    'superresolution': {
        'model': {
            'type': 'fno2d',
            'modes': 16,
            'width': 64,
            'n_layers': 4,
            'in_channels': 3,
            'out_channels': 1
        },
        'data': {
            'pde': 'navier_stokes',
            'train_resolution': 64,
            'test_resolutions': [128, 256, 512],
            'n_samples': 1000,
            'nu_range': [1e-4, 1e-3],
            'T': 50.0,
            'dt': 1e-3
        },
        'training': {
            'batch_size': 32,
            'epochs': 500,
            'optimizer': 'adamw',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'scheduler': 'cosine_warmup',
            'warmup_epochs': 10,
            'use_amp': True
        },
        'evaluation': {
            'metrics': ['relative_l2', 'spectral', 'energy', 'enstrophy']
        }
    },
    
    'domain_adaptation': {
        'model': {
            'type': 'fno2d',
            'modes': 16,
            'width': 64,
            'n_layers': 4
        },
        'data': {
            'source_pde': 'darcy',
            'target_pde': 'darcy',
            'source_permeability_range': [0.1, 10.0],
            'target_permeability_range': [0.01, 100.0],
            'source_correlation_length': 0.1,
            'target_correlation_length': 0.2,
            'n_source_samples': 1000,
            'n_target_shots': [1, 5, 10, 50]
        },
        'training': {
            'pretrain_epochs': 200,
            'finetune_epochs': 100,
            'optimizer': 'adamw',
            'lr': 1e-3,
            'finetune_lr': 1e-4
        },
        'evaluation': {
            'metrics': ['relative_l2']
        }
    },
    
    'theory_verification': {
        'model': {
            'type': 'fno2d',
            'modes': 16,
            'width': 64,
            'n_layers': 4
        },
        'data': {
            'pde': 'navier_stokes',
            'n_samples': 500,
            'resolution': 64
        },
        'theory': {
            'verify_generalization': True,
            'verify_spectral_convergence': True,
            'verify_stability': True,
            'verify_discretization_invariance': True,
            'delta': 0.05,
            'eps_list': [1e-4, 1e-3, 1e-2, 1e-1],
            'resolutions': [32, 64, 128]
        }
    }
}


def get_config(name: str) -> Dict[str, Any]:
    """Get experiment configuration by name."""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[name].copy()


def save_config(config: Dict[str, Any], path: str):
    """Save configuration to YAML."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursively merge configurations."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


# CLI for generating configs
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true', help='List available configs')
    parser.add_argument('--get', type=str, help='Get config by name')
    parser.add_argument('--save', type=str, help='Save config to path')
    args = parser.parse_args()
    
    if args.list:
        print("Available experiment configurations:")
        for name in EXPERIMENTS:
            print(f"  - {name}")
    elif args.get:
        config = get_config(args.get)
        if args.save:
            save_config(config, args.save)
            print(f"Saved to {args.save}")
        else:
            print(yaml.dump(config, default_flow_style=False))