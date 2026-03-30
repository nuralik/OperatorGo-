"""
Training script for the baseline GoNet CNN.

Usage:
    python -m training.train --data data/sgf_raw/cwi_9x9 --epochs 20
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from models.cnn_net import GoNet
from models.deeponet import DeepONetGo
from training.dataset import GoDataset


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_policy_loss = total_value_loss = 0.0
    correct = 0
    total = 0

    for boards, policy_targets, value_targets in loader:
        boards         = boards.to(device)
        policy_targets = policy_targets.to(device)
        value_targets  = value_targets.to(device)

        policy_logits, value_pred = model(boards)

        policy_loss = F.cross_entropy(policy_logits, policy_targets)
        value_loss  = F.mse_loss(value_pred, value_targets)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss        += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss  += value_loss.item()
        correct += (policy_logits.argmax(1) == policy_targets).sum().item()
        total   += len(boards)

    n = len(loader)
    return {
        'loss':        total_loss / n,
        'policy_loss': total_policy_loss / n,
        'value_loss':  total_value_loss / n,
        'policy_acc':  correct / total,
    }


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = total_policy_loss = total_value_loss = 0.0
    correct = 0
    total = 0

    for boards, policy_targets, value_targets in loader:
        boards         = boards.to(device)
        policy_targets = policy_targets.to(device)
        value_targets  = value_targets.to(device)

        policy_logits, value_pred = model(boards)

        policy_loss = F.cross_entropy(policy_logits, policy_targets)
        value_loss  = F.mse_loss(value_pred, value_targets)
        loss = policy_loss + value_loss

        total_loss        += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss  += value_loss.item()
        correct += (policy_logits.argmax(1) == policy_targets).sum().item()
        total   += len(boards)

    n = len(loader)
    return {
        'loss':        total_loss / n,
        'policy_loss': total_policy_loss / n,
        'value_loss':  total_value_loss / n,
        'policy_acc':  correct / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      default='cnn', choices=['cnn', 'deeponet'])
    parser.add_argument('--data',       default='data/sgf_raw/cwi_9x9')
    parser.add_argument('--board_size', type=int,   default=9)
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--filters',    type=int,   default=64)
    parser.add_argument('--res_blocks', type=int,   default=5)
    parser.add_argument('--latent_dim', type=int,   default=128)
    parser.add_argument('--val_split',  type=float, default=0.1)
    parser.add_argument('--save_dir',   default='checkpoints')
    args = parser.parse_args()

    device = (
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Device: {device}")

    # Load data
    print(f"Loading SGF data from {args.data} ...")
    dataset = GoDataset.from_sgf_dir(args.data, size_filter=args.board_size)
    print(f"  {len(dataset)} samples")

    val_size   = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device != 'cpu'))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device != 'cpu'))

    # Model
    if args.model == 'cnn':
        model = GoNet(
            board_size=args.board_size,
            filters=args.filters,
            n_res_blocks=args.res_blocks,
        ).to(device)
    else:
        model = DeepONetGo(
            board_size=args.board_size,
            filters=args.filters,
            n_res_blocks=args.res_blocks,
            latent_dim=args.latent_dim,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.model}  |  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    print(f"\n{'Epoch':>5} {'Train Loss':>10} {'Val Loss':>10} {'Policy Acc':>10} {'Time':>6}")
    print('-' * 50)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, device)
        val_m   = eval_epoch(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"{epoch:>5}  {train_m['loss']:>10.4f}  {val_m['loss']:>10.4f}"
              f"  {val_m['policy_acc']:>10.3f}  {elapsed:>5.1f}s")

        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            path = os.path.join(args.save_dir, f'best_{args.model}.pt')
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_loss': best_val_loss, 'args': vars(args)}, path)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to {args.save_dir}/best_{args.model}.pt")


if __name__ == '__main__':
    main()
