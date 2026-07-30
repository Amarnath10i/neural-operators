import torch
from src.pdes import generate_burgers_data, PDEDataset
from src.operators import FNO1d
from src.training import OperatorTrainer, SpectralLoss, create_optimizer, create_scheduler
from src.evaluation import evaluate_model, compute_all_metrics, relative_l2_error
from torch.utils.data import DataLoader

print('='*60)
print('NEURAL OPERATORS - EXPERIMENT OUTPUTS')
print('='*60)
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print()

# Generate data
print('Generating Burgers data...')
data = generate_burgers_data(n_samples=200, n=256, T=0.1, dt=1e-3, output_path='data/exp_burgers.h5')
print(f'Generated {len(data["input"])} samples')

# Create datasets
train_ds = PDEDataset('data/exp_burgers.h5', split='train')
val_ds = PDEDataset('data/exp_burgers.h5', split='val')
test_ds = PDEDataset('data/exp_burgers.h5', split='test')

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FNO1d(modes=16, width=64, n_layers=4, in_channels=1, out_channels=1).to(device)
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

# Training
loss_fn = SpectralLoss(weight_low=1.0, weight_high=0.5)
optimizer = create_optimizer(model, 'adamw', lr=1e-3, weight_decay=1e-4)
scheduler = create_scheduler(optimizer, 'cosine_warmup', epochs=30, warmup_epochs=5)

trainer = OperatorTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    use_amp=False,
    checkpoint_dir='checkpoints/exp_burgers',
    log_dir='logs/exp_burgers',
    use_wandb=False,
    use_tensorboard=True,
    early_stopping_patience=15,
    save_every=10
)

print('Training for 25 epochs...')
history = trainer.fit(25)

print(f'Best train loss: {min(history["train_loss"]):.6f}')
print(f'Best val loss: {min(history["val_loss"]):.6f}')

# Evaluation
print('Evaluating on test set...')
metrics = evaluate_model(model, test_loader, device=device)
print('Test Metrics:')
for k, v in metrics.items():
    print(f'  {k}: mean={v["mean"]:.6f}, std={v["std"]:.6f}')

# Detailed metrics
all_metrics = {}
for batch in test_loader:
    x = batch['input'].to(device)
    y = batch['output'].to(device)
    with torch.no_grad():
        pred = model(x)
    batch_metrics = compute_all_metrics(pred, y)
    for k, v in batch_metrics.items():
        if k not in all_metrics:
            all_metrics[k] = []
        all_metrics[k].append(v)

print('Detailed Test Metrics:')
for k, v in all_metrics.items():
    import numpy as np
    print(f'  {k}: mean={np.mean(v):.6f}, std={np.std(v):.6f}, median={np.median(v):.6f}')

print()
print('EXPERIMENT COMPLETE')
print('='*60)