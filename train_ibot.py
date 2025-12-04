#!/usr/bin/env python3
"""
train_ibot.py
iBOT training script adapted for 96x96 images and <100M params.
Supports custom W&B project naming.
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
import math
from PIL import ImageOps, ImageFilter
from torchvision import transforms

# Import dataloader builder
from dataloader import create_dataloaders, create_eval_dataloader, EfficientMultiPartDataset, collate_fn_dino
# Import utils
from utils import (
    setup_logging, cosine_scheduler, extract_features, 
    evaluate_knn_multiple_k, save_checkpoint, load_checkpoint, 
    AverageMeter, init_wandb, log_metrics, watch_model, 
    finish_wandb, print_model_summary, iBOTLoss, MaskingGenerator
)


# --- Augmentation Fix (CRITICAL) ---
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarization(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        return img

class DataAugmentationiBOT(object):
    # Re-implementing correct augmentation for SSL
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, image_size=96):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Global Crop 1
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(sigma=[0.1, 0.6]), # Adjusted for small images
            normalize,
        ])
        # Global Crop 2
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(sigma=[0.1, 0.6]),
            Solarization(0.2),
            normalize,
        ])
        # Local Crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(sigma=[0.1, 0.3]), # Less blur for small crops
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

# --- Model Definitions ---

class iBOTHead(nn.Module):
    """
    Projection head shared between CLS and Patch tokens.
    """
    def __init__(self, in_dim, out_dim=8192, hidden_dim=2048, bottleneck_dim=256, nlayers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim)]
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
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

class iBOTBackboneWrapper(nn.Module):
    """
    Wraps a TIMM ViT to return both CLS and Patch tokens.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_features = backbone.num_features
        
        # Disable built-in head
        self.backbone.head = nn.Identity()
        self.backbone.fc = nn.Identity()

    def forward(self, x, mask=None):
        features = self.backbone.forward_features(x)
        
        # Assuming Standard ViT from TIMM: output is (B, N_patches + 1, Dim)
        cls_token = features[:, 0]
        patch_tokens = features[:, 1:]
        
        cls_out = self.head(cls_token) # (B, Out_dim)
        patch_out = self.head(patch_tokens) # (B, N_patches, Out_dim)
        
        return cls_out, patch_out

class MultiCropWrapper(nn.Module):
    """
    Handles Multi-Crop + Head Projection + Mask Token injection
    Fixed for 96x96 inputs where Global and Local crops share the same resolution.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        # We need a mask token for the student
        self.mask_token = nn.Parameter(torch.zeros(1, 1, backbone.embed_dim))
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x, masks=None):
        # x is a list of tensors (crops)
        if not isinstance(x, list):
            x = [x]
            
        # Group crops by resolution
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True
        )[1], 0)
        
        start_idx = 0
        output_cls = []
        output_patches = []
        
        for i, end_idx in enumerate(idx_crops):
            _x = torch.cat(x[start_idx:end_idx])
            
            # --- FIX START: Handle Mask Mismatch ---
            current_masks = None
            if masks is not None and i == 0: 
                # Concatenate all provided masks (Global Crops)
                # masks is a tuple/list, so we concat them first
                masks_tensor = torch.cat(list(masks)) 
                
                B_features = _x.shape[0] # e.g. 512 (Global + Local)
                B_masks = masks_tensor.shape[0] # e.g. 128 (Global only)
                
                if B_features > B_masks:
                    # If we have more images than masks (because Local crops joined the batch),
                    # Pad the masks with Zeros (No masking for local crops)
                    padding = torch.zeros(
                        B_features - B_masks, 
                        masks_tensor.shape[1], 
                        device=masks_tensor.device, 
                        dtype=masks_tensor.dtype
                    )
                    current_masks = torch.cat([masks_tensor, padding])
                else:
                    # Standard case
                    current_masks = masks_tensor
            # --- FIX END ---

            # Forward backbone
            features = self.backbone.forward_features(_x)
            
            # Handle Masking Logic Here: Replace masked features with mask_token
            if current_masks is not None:
                # features: [B, 1+N, D]
                # current_masks: [B, N] boolean
                B, N, D = features[:, 1:].shape
                mask_token_expand = self.mask_token.expand(B, N, -1)
                
                # Replace features where mask == 1 (True)
                w = current_masks.unsqueeze(-1).type_as(features)
                features[:, 1:] = features[:, 1:] * (1 - w) + mask_token_expand * w

            cls_token = features[:, 0]
            patch_tokens = features[:, 1:]
            
            output_cls.append(self.head(cls_token))
            
            # We only need patch outputs for global crops (for iBOT loss)
            # If batch contains local crops, we need to slice them off
            if i == 0:
                # We assume masks correspond to the first B_masks elements
                # But here we just return everything, the Loss function handles selecting the right ones
                # NOTE: To save memory, we could slice here, but let's keep it robust.
                # Actually, iBOTLoss expects ONLY global crops in teacher_patch/student_patch usually.
                # But our Loss implementation handles chunks.
                
                # IMPORTANT: We slice output_patches to only return Global Crops results
                # because iBOTLoss expects chunks of size 'ngcrops'.
                # If we pass 512 items but ngcrops=2 (128 items), chunking will fail or be misaligned.
                
                # Let's see if we passed masks
                if masks is not None:
                     # This is the student
                     num_global = sum([m.shape[0] for m in masks])
                     output_patches.append(self.head(patch_tokens[:num_global]))
                else:
                     # This is the teacher, assumes inputs_teacher only has globals
                     # Or teacher handles logic separately.
                     # In train_one_epoch, teacher only gets global_crops.
                     output_patches.append(self.head(patch_tokens))
                
            start_idx = end_idx

        output_cls = torch.cat(output_cls)
        # Combine patch outputs
        if len(output_patches) > 0:
            output_patches = torch.cat(output_patches)
        
        return output_cls, output_patches

# --- Training Logic ---

def train_one_epoch(student, teacher, ibot_loss, train_loader, optimizer, 
                    lr_schedule, wd_schedule, momentum_schedule, epoch, 
                    mask_generator, device, scaler, args):
    student.train()
    teacher.eval()
    
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    metric_logger = AverageMeter()
    
    pbar = tqdm(train_loader, desc=header)
    
    for it, (global_crops, local_crops) in enumerate(pbar):
        it = len(train_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0: param_group["weight_decay"] = wd_schedule[it]

        # Prepare batch
        global_crops = global_crops.to(device, non_blocking=True)
        local_crops = local_crops.to(device, non_blocking=True)
        
        # Generate Masks for Global Crops (on CPU then move to GPU)
        masks = []
        for _ in range(args.batch_size * args.global_crops_number):
            m = mask_generator()
            masks.append(torch.from_numpy(m).flatten()) 
        masks = torch.stack(masks).to(device).bool()
        
        # Split masks for the two global views
        masks_chunked = masks.chunk(args.global_crops_number)

        # Prepare Inputs
        inputs_student = [global_crops[:args.batch_size], global_crops[args.batch_size:]]
        local_chunks = local_crops.chunk(args.local_crops_number)
        inputs_student.extend(local_chunks)
        
        inputs_teacher = [global_crops[:args.batch_size], global_crops[args.batch_size:]]

        with autocast():
            # Student Forward
            student_cls, student_patch = student(inputs_student, masks=masks_chunked)
            
            # Teacher Forward
            teacher_cls, teacher_patch = teacher(inputs_teacher)
            
            # Calculate Loss
            total_loss, cls_loss_val, mim_loss_val = ibot_loss(
                (student_cls, student_patch),
                (teacher_cls, teacher_patch),
                masks_chunked,
                epoch
            )

        # Backward
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        if args.clip_grad:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        # EMA Update
        with torch.no_grad():
            m = momentum_schedule[it]
            for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

        # Logging
        if not args.no_wandb and it % args.log_freq == 0:
            log_metrics({
                'train/loss': total_loss.item(),
                'train/cls_loss': cls_loss_val,
                'train/mim_loss': mim_loss_val,
                'train/lr': optimizer.param_groups[0]["lr"],
            }, step=it)
            
        pbar.set_postfix({'loss': total_loss.item(), 'mim': mim_loss_val})

    return 0

# --- Main ---

def create_vit_model(args):
    # Create ViT using TIMM
    model = timm.create_model(
        'vit_small_patch16_224', # Base model (will override patch/img size)
        pretrained=False,
        img_size=96,
        patch_size=args.patch_size,
        num_classes=0,
        drop_path_rate=args.drop_path_rate
    )
    return model

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils_log = setup_logging(args.output_dir)
    
    # --------------------------------------------------------------------------
    # MODIFIED: Use the custom project name
    # --------------------------------------------------------------------------
    if not args.no_wandb:
        init_wandb(args, project_name=args.wandb_project)
    # --------------------------------------------------------------------------

    # 1. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader = create_dataloaders(
        base_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_local_crops=args.local_crops_number,
        seed=args.seed
    )
    
    # Monkey-patching the dataset transform to ensure correct Augmentation
    new_transform = DataAugmentationiBOT(
        global_crops_scale=(0.14, 1.),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=args.local_crops_number,
        image_size=96
    )
    train_loader.dataset.transform = new_transform

    # 2. Mask Generator
    mask_generator = MaskingGenerator(
        input_size=(12, 12),
        num_masking_patches=45, # Approx 30% masking
        max_num_patches=None,
        min_num_patches=4,
    )

    # 3. Model
    print("Creating iBOT models (ViT-Small)...")
    student_backbone = create_vit_model(args)
    teacher_backbone = create_vit_model(args)
    
    embed_dim = student_backbone.embed_dim
    
    # Heads
    student_head = iBOTHead(embed_dim, out_dim=args.out_dim)
    teacher_head = iBOTHead(embed_dim, out_dim=args.out_dim)
    
    # Wrappers
    student = MultiCropWrapper(student_backbone, student_head).to(device)
    teacher = MultiCropWrapper(teacher_backbone, teacher_head).to(device)
    
    # Init Teacher
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    # 4. Loss
    ibot_loss = iBOTLoss(
        out_dim=args.out_dim,
        patch_out_dim=args.out_dim,
        ngcrops=args.global_crops_number,
        nlcrops=args.local_crops_number,
        warmup_teacher_temp=args.warmup_teacher_temp, 
        teacher_temp=args.teacher_temp,               
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        lambda1=1.0,
        lambda2=1.0
    ).to(device)

    # 5. Optimizer
    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 6. Schedules
    niter_per_ep = len(train_loader)
    lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.epochs, niter_per_ep, args.warmup_epochs)
    wd_schedule = cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, niter_per_ep)
    momentum_schedule = cosine_scheduler(args.momentum_teacher, 1, args.epochs, niter_per_ep)
    
    scaler = GradScaler()

    # 7. Training Loop
    print("Starting iBOT training...")
    for epoch in range(args.epochs):
        train_one_epoch(
            student, teacher, ibot_loss, train_loader, optimizer,
            lr_schedule, wd_schedule, momentum_schedule, epoch,
            mask_generator, device, scaler, args
        )
        
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    finish_wandb()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('iBOT Training')
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--output_dir', default='./checkpoints_ibot', type=str)
    parser.add_argument('--patch_size', default=8, type=int)
    parser.add_argument('--out_dim', default=8192, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.04, type=float)
    parser.add_argument('--weight_decay_end', default=0.4, type=float)
    parser.add_argument('--clip_grad', default=3.0, type=float)
    parser.add_argument('--momentum_teacher', default=0.996, type=float)
    parser.add_argument('--local_crops_number', default=6, type=int)
    parser.add_argument('--global_crops_number', default=2, type=int)
    parser.add_argument('--drop_path_rate', default=0.1, type=float)
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--knn_k', default=20, type=int)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--log_freq', default=50, type=int)
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--knn_search', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--val_freq', default=5, type=int)
    parser.add_argument('--use_bn_in_head', action='store_true')
    parser.add_argument('--norm_last_layer', default=True, type=bool) 
    parser.add_argument('--wandb_project', default='ViT2025start', type=str, help='W&B project name')
    parser.add_argument('--teacher_temp', default=0.07, type=float, help='Final teacher temperature')
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float, help='Initial teacher temperature')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Warmup epochs')

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)