#!/usr/bin/env python3
"""
efficient_dataloader.py
90/10 train/val split across part1-5 with ZERO copying
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
import os
import glob
import numpy as np


class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
    
    def __call__(self, x):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class Solarization:
    def __init__(self, threshold=128):
        self.threshold = threshold
    
    def __call__(self, x):
        return ImageOps.solarize(x, self.threshold)


class DINOAugmentation:
    def __init__(self, n_local_crops=6, image_size=96):
        self.n_local_crops = n_local_crops
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0),
                                        interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigma=[0.1, 2.0]),
            transforms.ToTensor(),
            normalize
        ])
        
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.05, 0.4),
                                        interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigma=[0.1, 2.0]),
            transforms.RandomApply([Solarization(threshold=128)], p=0.2),
            transforms.ToTensor(),
            normalize
        ])
    
    def __call__(self, image):
        crops = [self.global_transform(image), self.global_transform(image)]
        crops.extend([self.local_transform(image) for _ in range(self.n_local_crops)])
        return crops


# class DINOAugmentation:
#     def __init__(self, n_local_crops=6, image_size=96):
#         self.n_local_crops = n_local_crops
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
#         # --- MODIFIED: NO AUGMENTATION ---
#         # Only uses a fixed resize and normalization.
#         self.base_transform = transforms.Compose([
#             # Use deterministic Resize/CenterCrop to get a single view
#             transforms.Resize(image_size), 
#             transforms.CenterCrop(image_size),
#             transforms.ToTensor(),
#             normalize
#         ])
        
#         # All transforms are now identical (No Augmentation)
#         self.global_transform = self.base_transform
#         self.local_transform = self.base_transform
    
#     def __call__(self, image):
#         crops = [self.global_transform(image), self.global_transform(image)]
#         crops.extend([self.local_transform(image) for _ in range(self.n_local_crops)])
#         return crops


class EfficientMultiPartDataset(Dataset):
    """
    Efficient dataset that loads from part1-5.
    
    Strategy for 90/10 split:
    - Build index of all image paths ONCE (cheap - just filenames)
    - Shuffle indices with seed
    - Split indices into train/val (90/10)
    - Load images on-demand from paths
    
    ZERO copying, ZERO extra disk space!
    """
    
    def __init__(self, base_dir="./data", transform=None, is_train=True,
                 train_ratio=0.9, seed=42):
        print(f"Building image index from {base_dir}/part*...")
        
        # Step 1: Build list of all image paths (this is fast and cheap!)
        self.image_paths = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPEG', '*.JPG', '*.PNG']
        
        for part_num in range(1, 6):
            part_dir = os.path.join(base_dir, f"part{part_num}")
            if not os.path.exists(part_dir):
                print(f"⚠️  {part_dir} not found, skipping...")
                continue
            
            print(f"  Indexing part{part_num}...", end=" ", flush=True)
            part_images = []
            for ext in extensions:
                part_images.extend(glob.glob(os.path.join(part_dir, '**', ext), recursive=True))
            print(f"{len(part_images):,} images")
            self.image_paths.extend(part_images)
        
        # Sort for reproducibility
        self.image_paths = sorted(self.image_paths)
        total_size = len(self.image_paths)
        print(f"\n✓ Total images found: {total_size:,}")
        
        self.transform = transform
        
        # Step 2: Create reproducible train/val split using indices (very cheap!)
        print(f"Creating {'train' if is_train else 'val'} split...", end=" ", flush=True)
        
        # Create indices array
        all_indices = np.arange(total_size)
        
        # Shuffle with seed for reproducibility
        rng = np.random.RandomState(seed)
        rng.shuffle(all_indices)
        
        # Split into train (90%) and val (10%)
        split_idx = int(total_size * train_ratio)
        
        if is_train:
            self.indices = all_indices[:split_idx]
        else:
            self.indices = all_indices[split_idx:]
        
        print(f"{len(self.indices):,} images")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Dataset ready:")
        print(f"  Split: {'Train (90%)' if is_train else 'Val (10%)'}")
        print(f"  Images: {len(self.indices):,}")
        print(f"  Disk usage: 0 bytes extra (using existing files)")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual image path from our index
        actual_idx = self.indices[idx]
        img_path = self.image_paths[actual_idx]
        
        # Load image on-demand
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            return self.transform(image)
        return image


def collate_fn_dino(batch):
    """Collate function for DINO multi-crop."""
    n_global = 2
    global_crops = []
    local_crops = []
    
    for crops in batch:
        global_crops.extend(crops[:n_global])
        local_crops.extend(crops[n_global:])
    
    return torch.stack(global_crops), torch.stack(local_crops)


def create_dataloaders(base_dir="./data", batch_size=128, num_workers=8,
                      n_local_crops=6, train_ratio=0.9, seed=42):
    """
    Create train and val dataloaders with 90/10 split.
    
    ZERO disk overhead - just uses indices!
    """
    print("="*60)
    print("Creating Dataloaders (90% train / 10% val)")
    print("="*60)
    
    # DINO augmentation
    dino_aug = DINOAugmentation(n_local_crops=n_local_crops, image_size=96)
    
    # Train dataset (90%)
    print("\n[1/2] Creating training dataset...")
    train_dataset = EfficientMultiPartDataset(
        base_dir=base_dir,
        transform=dino_aug,
        is_train=True,
        train_ratio=train_ratio,
        seed=seed
    )
    
    # Val dataset (10%)
    print("\n[2/2] Creating validation dataset...")
    val_dataset = EfficientMultiPartDataset(
        base_dir=base_dir,
        transform=dino_aug,
        is_train=False,
        train_ratio=train_ratio,
        seed=seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle within the 90% train split
        num_workers=num_workers,
        collate_fn=collate_fn_dino,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        collate_fn=collate_fn_dino,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    print("\n" + "="*60)
    print("Dataloaders Ready!")
    print("="*60)
    print(f"Train batches per epoch: {len(train_loader):,}")
    print(f"Val batches per epoch: {len(val_loader):,}")
    print("="*60 + "\n")
    
    return train_loader, val_loader


def create_eval_dataloader(base_dir="./data", batch_size=256, num_workers=8,
                           train_ratio=0.9, seed=42):
    """
    Create evaluation dataloader (uses val split for k-NN).
    """
    eval_transform = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    class EvalDataset(Dataset):
        def __init__(self, base_dir, train_ratio, seed):
            # Build image index
            self.image_paths = []
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPEG', '*.JPG', '*.PNG']
            
            for part_num in range(1, 6):
                part_dir = os.path.join(base_dir, f"part{part_num}")
                if os.path.exists(part_dir):
                    for ext in extensions:
                        self.image_paths.extend(glob.glob(os.path.join(part_dir, '**', ext), recursive=True))
            
            self.image_paths = sorted(self.image_paths)
            
            # Use val split (10%)
            all_indices = np.arange(len(self.image_paths))
            rng = np.random.RandomState(seed)
            rng.shuffle(all_indices)
            
            split_idx = int(len(self.image_paths) * train_ratio)
            self.indices = all_indices[split_idx:]  # Val portion
            
            print(f"Eval set: {len(self.indices):,} samples")
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            actual_idx = self.indices[idx]
            img_path = self.image_paths[actual_idx]
            image = Image.open(img_path).convert('RGB')
            image = eval_transform(image)
            label = -1  # No labels for SSL
            return image, label
    
    eval_dataset = EvalDataset(base_dir, train_ratio, seed)
    
    return DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )


# Test the dataloader
if __name__ == "__main__":
    import time
    
    print("\n" + "="*60)
    print("Testing Efficient Dataloader")
    print("="*60)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        base_dir="./data",
        batch_size=64,
        num_workers=8,
        n_local_crops=4
    )
    
    # Speed test
    print("\nSpeed test - loading 10 batches...")
    start = time.time()
    for i, (global_crops, local_crops) in enumerate(train_loader):
        print(f"  Batch {i+1}: global {global_crops.shape}, local {local_crops.shape}")
        if i >= 9:
            break
    elapsed = time.time() - start
    
    print(f"\n✓ Loaded 10 batches in {elapsed:.2f} seconds")
    print(f"✓ Average: {elapsed/10:.3f} seconds per batch")
    print(f"✓ Estimated throughput: ~{640/elapsed:.0f} images/second")
    
    print("\n" + "="*60)
    print("Dataloader is ready for training! 🚀")
    print("="*60)