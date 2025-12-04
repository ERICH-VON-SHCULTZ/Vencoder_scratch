#!/usr/bin/env python3
"""
train.py
DINO training script with ViT-Small (8x8 patches) for SSL competition
W&B integration for monitoring
Uses efficient multi-part dataloader (no disk overhead)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import timm
from tqdm import tqdm
import argparse
import os
from pathlib import Path
import wandb
import numpy as np

from dataloader import create_dataloaders, create_eval_dataloader
from utils import (
    setup_logging,
    cosine_scheduler,
    DINOLoss,
    extract_features,
    train_knn_classifier,
    evaluate_knn_multiple_k,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    init_wandb,
    log_metrics,
    watch_model,
    finish_wandb,
    print_model_summary
)


class DINOHead(nn.Module):
    """
    DINO projection head.
    Maps encoder output to a lower-dimensional space with normalization.
    """
    
    def __init__(self, in_dim, out_dim=65536, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
        
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    """
    Wrapper to handle multi-crop inputs for DINO.
    Passes all crops through the backbone and projection head.
    """
    
    def __init__(self, backbone, head):
        super().__init__()
        backbone.fc = nn.Identity()  # Remove classification head
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x):
        # x can be a list of crops or a single tensor
        if not isinstance(x, list):
            x = [x]
        
        # Check if we have exactly 1 tensor (already concatenated)
        if len(x) == 1:
            # All crops are concatenated along batch dimension
            output = self.backbone(x[0])
            output = self.head(output)
            return output
        
        # Process each crop size separately (for efficiency)
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True
        )[1], 0)
        
        start_idx = 0
        output = []
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]))
            output.append(_out)
            start_idx = end_idx
        
        # Concatenate all outputs
        output = torch.cat(output)
        output = self.head(output)
        
        return output


def create_vit_model(
    model_name='vit_small_patch8_96',
    patch_size=8,
    img_size=96,
    num_classes=0,
    drop_path_rate=0.1
):
    """
    Create ViT model with custom patch size for 96px images.
    
    Args:
        model_name: Base model name
        patch_size: Patch size (8 for 144 tokens, 16 for 36 tokens)
        img_size: Image size (fixed at 96)
        num_classes: Number of classes (0 for feature extraction)
        drop_path_rate: Stochastic depth rate
    
    Returns:
        model: ViT encoder
    """
    # For ViT-Small: embed_dim=384, depth=12, num_heads=6
    # Parameters: ~22M (well under 100M limit)
    
    model = timm.create_model(
        'vit_small_patch16_224',  # Base model
        pretrained=False,  # ✅ Random initialization - NO pretrained weights
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    assert n_params < 100e6, f"Model has {n_params/1e6:.2f}M parameters (must be < 100M)"
    
    return model


@torch.no_grad()
def momentum_update(student, teacher, momentum):
    """
    Momentum update for teacher network.
    teacher = momentum * teacher + (1 - momentum) * student
    """
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data.mul_(momentum).add_((1 - momentum) * param_s.detach().data)


def train_one_epoch(
    student,
    teacher,
    dino_loss,
    train_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    device,
    scaler,
    args,
    clip_grad=3.0
):
    """
    Train for one epoch.
    """
    student.train()
    teacher.eval()
    
    loss_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    
    for it, (global_crops, local_crops) in enumerate(pbar):
        # Update learning rate and weight decay
        iteration = epoch * len(train_loader) + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[iteration]
            if i == 0:  # Only the first group is regularized
                param_group["weight_decay"] = wd_schedule[iteration]
        
        # Move to device
        global_crops = global_crops.to(device, non_blocking=True)
        local_crops = local_crops.to(device, non_blocking=True)
        
        # Concatenate all crops
        all_crops = torch.cat([global_crops, local_crops], dim=0)
        
        # Forward pass with mixed precision
        with autocast():
            # Student forward (all crops)
            student_output = student(all_crops)
            
            # Teacher forward (only global crops)
            teacher_output = teacher(global_crops)
            
            # Compute loss
            loss = dino_loss(student_output, teacher_output, epoch)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            param_norms = nn.utils.clip_grad_norm_(student.parameters(), clip_grad)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # EMA update for teacher
        with torch.no_grad():
            m = momentum_schedule[iteration]
            momentum_update(student, teacher, m)
        
        # Update meters
        loss_meter.update(loss.item(), global_crops.size(0))
        
        # Log to W&B every N iterations
        if not args.no_wandb and it % args.log_freq == 0:
            log_metrics({
                'train/loss': loss.item(),
                'train/loss_avg': loss_meter.avg,
                'train/lr': optimizer.param_groups[0]["lr"],
                'train/wd': optimizer.param_groups[0]["weight_decay"],
                'train/momentum': m,
                'train/epoch': epoch,
            }, step=epoch * len(train_loader) + it, commit=True)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    return loss_meter.avg


@torch.no_grad()
def validate(student, val_loader, dino_loss, epoch, device, args):
    """
    Validate on validation set.
    """
    student.eval()
    
    loss_meter = AverageMeter()
    
    # Create a dummy teacher for validation (just use student)
    teacher = student
    
    pbar = tqdm(val_loader, desc="Validation")
    
    for global_crops, local_crops in pbar:
        global_crops = global_crops.to(device, non_blocking=True)
        local_crops = local_crops.to(device, non_blocking=True)
        
        all_crops = torch.cat([global_crops, local_crops], dim=0)
        
        # Forward pass
        student_output = student(all_crops)
        teacher_output = teacher(global_crops)
        
        # Compute loss
        loss = dino_loss(student_output, teacher_output, epoch)
        
        loss_meter.update(loss.item(), global_crops.size(0))
        pbar.set_postfix({'val_loss': f'{loss_meter.avg:.4f}'})
    
    # Log to W&B
    if not args.no_wandb:
        log_metrics({
            'val/loss': loss_meter.avg,
            'val/epoch': epoch,
        }, step=epoch, commit=True)
    
    return loss_meter.avg


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize W&B
    if not args.no_wandb:
        wandb_run = init_wandb(args, project_name=args.wandb_project)
    
    # Logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Arguments: {args}")
    
    # Create dataloaders
    logger.info("Creating dataloaders from part1-5 directories...")
    train_loader, val_loader = create_dataloaders(
        base_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_local_crops=args.local_crops_number,
        train_ratio=0.9,
        seed=args.seed
    )
    
    logger.info(f"Train loader: {len(train_loader)} batches per epoch")
    logger.info(f"Val loader: {len(val_loader)} batches per epoch")
    
    # Create student and teacher models
    logger.info("Creating ViT-Small models...")
    
    # Student
    student_backbone = create_vit_model(
        patch_size=args.patch_size,
        img_size=96,
        num_classes=0,
        drop_path_rate=args.drop_path_rate
    )
    
    embed_dim = student_backbone.embed_dim
    
    student_head = DINOHead(
        in_dim=embed_dim,
        out_dim=args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer
    )
    
    student = MultiCropWrapper(student_backbone, student_head)
    student = student.to(device)
    
    # Print model summary
    print_model_summary(student)
    
    # Teacher (same architecture as student)
    teacher_backbone = create_vit_model(
        patch_size=args.patch_size,
        img_size=96,
        num_classes=0,
        drop_path_rate=args.drop_path_rate
    )
    
    teacher_head = DINOHead(
        in_dim=embed_dim,
        out_dim=args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer
    )
    
    teacher = MultiCropWrapper(teacher_backbone, teacher_head)
    teacher = teacher.to(device)
    
    if args.pretrained_weights:
        if os.path.isfile(args.pretrained_weights):
            logger.info(f"==> Loading pretrained weights from: {args.pretrained_weights}")
            checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
            
            # Handle dictionary checkpoints (typical save format) vs raw state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load into student
            # strict=True ensures we are loading into the exact same architecture
            msg = student.load_state_dict(state_dict, strict=True)
            logger.info(f"==> Loaded weights successfully. Message: {msg}")
        else:
            logger.warning(f"!! Pretrained weights file not found: {args.pretrained_weights}")
            logger.warning("!! Continuing with random initialization...")
    
    # Initialize teacher with student weights
    # Note: If we loaded pretrained weights above, this will sync them to teacher too.
    teacher.load_state_dict(student.state_dict())
    
    # Teacher has no gradients
    for p in teacher.parameters():
        p.requires_grad = False
    
    logger.info(f"Student and teacher created")
    
    # Watch model in W&B
    if not args.no_wandb:
        watch_model(student, log_freq=args.log_freq)
    
    # DINO loss
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # Total number of crops
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs
    ).to(device)
    
    # Optimizer
    params_groups = [
        {'params': [p for n, p in student.named_parameters() if p.requires_grad]},
    ]
    
    optimizer = AdamW(params_groups, lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate schedule
    niter_per_ep = len(train_loader)

    # Use LR directly without scaling (simpler and more predictable)
    base_lr = args.lr

    lr_schedule = cosine_scheduler(
        args.lr * (args.batch_size * args.world_size) / 256.,  # Linear scaling rule
        args.min_lr,
        args.epochs,
        niter_per_ep,
        warmup_epochs=args.warmup_epochs
    )

    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        niter_per_ep
    )

    # Momentum schedule for teacher
    momentum_schedule = cosine_scheduler(
        args.momentum_teacher,
        1,
        args.epochs,
        niter_per_ep
    )
    
    logger.info(f"Training schedule:")
    logger.info(f"  Iterations per epoch: {niter_per_ep}")
    logger.info(f"  Total iterations: {args.epochs * niter_per_ep}")
    logger.info(f"  Initial LR: {lr_schedule[0]:.6f}")
    logger.info(f"  Peak LR (after warmup): {lr_schedule[niter_per_ep * args.warmup_epochs]:.6f}")
    logger.info(f"  Final LR: {lr_schedule[-1]:.6f}")

    # Mixed precision scaler
    scaler = GradScaler()
    
    # Resume from checkpoint (Full State)
    # Note: This logic overrides --pretrained_weights if both are present
    start_epoch = 0
    if args.resume:
        logger.info(f"==> Resuming full training state from: {args.resume}")
        start_epoch = load_checkpoint(
            args.resume,
            student,
            optimizer
        )
        teacher.load_state_dict(student.state_dict())
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss = train_one_epoch(
            student,
            teacher,
            dino_loss,
            train_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            device,
            scaler,
            args,
            clip_grad=args.clip_grad
        )
        
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Validate
        if (epoch + 1) % args.val_freq == 0:
            val_loss = validate(student, val_loader, dino_loss, epoch, device, args)
            logger.info(f"Val loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': student.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(args.output_dir, 'best_model.pth'))
                logger.info(f"✓ Saved best model (val_loss={val_loss:.4f})")
                
                # Log best model to W&B
                if not args.no_wandb:
                    log_metrics({'val/best_loss': best_val_loss}, step=epoch, commit=False)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            logger.info(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    # Final save
    save_checkpoint({
        'epoch': args.epochs,
        'model_state_dict': student.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    logger.info("Training complete!")
    
    # Evaluate with k-NN
    if args.evaluate:
        logger.info("\nEvaluating with k-NN...")
        eval_knn(student.backbone, args, device, logger)
    
    # Finish W&B
    if not args.no_wandb:
        finish_wandb()


@torch.no_grad()
def eval_knn(model, args, device, logger):
    """
    Evaluate encoder with k-NN classifier.
    """
    logger.info("\n" + "="*60)
    logger.info("k-NN Evaluation")
    logger.info("="*60)
    
    # Load evaluation data
    logger.info("Creating evaluation dataloader...")
    eval_loader = create_eval_dataloader(
        base_dir=args.data_dir,
        batch_size=256,
        num_workers=args.num_workers,
        train_ratio=0.9,
        seed=args.seed
    )
    
    logger.info("Extracting features...")
    all_features, all_labels = extract_features(model, eval_loader, device)
    
    logger.info(f"Total samples: {len(all_features)}")
    logger.info(f"Feature dimension: {all_features.shape[1]}")
    
    # Split eval into train and test for k-NN
    # Use 80% for k-NN train (feature bank), 20% for k-NN test
    n_train = int(0.8 * len(all_features))
    
    # Shuffle indices for random split
    np.random.seed(args.seed)
    indices = np.random.permutation(len(all_features))
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_features = all_features[train_indices]
    train_labels = all_labels[train_indices]
    test_features = all_features[test_indices]
    test_labels = all_labels[test_indices]
    
    logger.info(f"\nk-NN split:")
    logger.info(f"  Train (feature bank): {len(train_features)} samples")
    logger.info(f"  Test: {len(test_features)} samples")
    
    # Evaluate with multiple k values
    if args.knn_search:
        k_values = [1, 5, 10, 20, 50, 100, 200]
        results, best_k, best_acc = evaluate_knn_multiple_k(
            train_features,
            train_labels,
            test_features,
            test_labels,
            k_values=k_values
        )
        
        logger.info(f"\n✓ Best k-NN configuration: k={best_k}, accuracy={best_acc*100:.2f}%")
        
        # Log all k values to W&B
        if not args.no_wandb:
            for k, acc in results.items():
                log_metrics({f'knn/k_{k}_acc': acc * 100}, commit=False)
            log_metrics({
                'knn/best_k': best_k,
                'knn/best_acc': best_acc * 100
            }, commit=True)
    else:
        # Use specified k value
        classifier, test_acc = train_knn_classifier(
            train_features,
            train_labels,
            test_features,
            test_labels,
            k=args.knn_k
        )
        
        logger.info(f"\n✓ k-NN (k={args.knn_k}) Test Accuracy: {test_acc*100:.2f}%")
        
        # Log to W&B
        if not args.no_wandb:
            log_metrics({
                'knn/test_acc': test_acc * 100,
                'knn/k': args.knn_k
            }, commit=True)
    
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DINO training with W&B monitoring')
    
    # Data parameters
    parser.add_argument('--data_dir', default='./data', type=str, help='Data directory containing part1-5')
    
    # Model parameters
    parser.add_argument('--patch_size', default=8, type=int, help='Patch size for ViT')
    parser.add_argument('--out_dim', default=65536, type=int, help='Dimensionality of DINO head output')
    parser.add_argument('--norm_last_layer', default=True, type=bool, help='Normalize last layer')
    parser.add_argument('--use_bn_in_head', default=False, type=bool, help='Use BN in projection head')
    parser.add_argument('--drop_path_rate', default=0.1, type=float, help='Stochastic depth rate')
    
    # Training parameters
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='Warmup epochs')
    parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='Minimum learning rate')
    parser.add_argument('--weight_decay', default=0.04, type=float, help='Weight decay')
    parser.add_argument('--weight_decay_end', default=0.4, type=float, help='Final weight decay')
    parser.add_argument('--clip_grad', default=3.0, type=float, help='Gradient clipping')
    
    # DINO parameters
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help='Base EMA momentum for teacher')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Teacher temperature')
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float, help='Warmup teacher temperature')
    parser.add_argument('--warmup_teacher_temp_epochs', default=5, type=int, help='Warmup teacher temp epochs')
    parser.add_argument('--local_crops_number', default=6, type=int, help='Number of local crops')
    
    # k-NN parameters
    parser.add_argument('--knn_k', default=20, type=int, help='Number of neighbors for k-NN')
    parser.add_argument('--knn_search', action='store_true', help='Search for best k value')
    
    # W&B parameters
    parser.add_argument('--wandb_project', default='ssl-competition', type=str, help='W&B project name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--log_freq', default=50, type=int, help='Log frequency (iterations)')
    
    # Misc
    parser.add_argument('--output_dir', default='./checkpoints', type=str, help='Output directory')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--world_size', default=1, type=int, help='Number of GPUs')

    parser.add_argument('--pretrained_weights', default='', type=str, help='Path to pretrained model weights (starts from epoch 0)')
    parser.add_argument('--resume', default='', type=str, help='Resume full training state (optimizer + epoch)')
    
    parser.add_argument('--evaluate', action='store_true', help='Run k-NN evaluation after training')
    parser.add_argument('--val_freq', default=5, type=int, help='Validation frequency (epochs)')
    parser.add_argument('--save_freq', default=10, type=int, help='Save frequency (epochs)')
    
    args = parser.parse_args()
    
    # Validate warmup_teacher_temp_epochs
    if args.warmup_teacher_temp_epochs >= args.epochs:
        args.warmup_teacher_temp_epochs = max(1, args.epochs // 2)
        print(f"⚠️  Adjusted warmup_teacher_temp_epochs to {args.warmup_teacher_temp_epochs}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run training
    main(args)