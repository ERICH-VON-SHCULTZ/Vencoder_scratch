#!/usr/bin/env python3
"""
utils.py
Utility functions for DINO training and evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import os
import logging
import wandb
import math



class iBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1 # CLS loss weight
        self.lambda2 = lambda2 # MIM loss weight

        # Teacher temperature schedule
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, student_masks, epoch):
        """
        student_output: (student_cls, student_patch)
        teacher_output: (teacher_cls, teacher_patch)
        student_masks: List of masks for global crops
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output
        
        # --- 1. CLS Token Loss (DINO Logic) ---
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        
        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)

        total_loss1 = 0
        n_loss_terms1 = 0
        
        # Global crops logic
        for iq, q in enumerate(teacher_cls_c):
            for v in range(len(student_cls_c)):
                if v == iq: continue # Skip same view
                
                loss = torch.sum(-q * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                total_loss1 += loss.mean()
                n_loss_terms1 += 1
        
        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1

        # --- 2. Patch Token Loss (MIM Logic) ---
        # iBOT only calculates MIM loss on Global Crops
        total_loss2 = 0
        n_loss_terms2 = 0
        
        # Teacher patches: Center + Sharpen
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)
        
        # Student patches: Apply Temp
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops) # Only global crops pass through patch head

        for iq, q in enumerate(teacher_patch_c):
            for v in range(len(student_patch_c)):
                if v == iq:
                    # MIM Loss: Compare Student(Masked View v) vs Teacher(Original View v)
                    # We only care about masked positions
                    loss = torch.sum(-q * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    
                    # mask shape: [B, N] -> flatten to match token sequence
                    mask = student_masks[v]
                    
                    # Compute mean loss ONLY on masked tokens
                    loss = torch.sum(loss * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss.mean()
                    n_loss_terms2 += 1

        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2

        self.update_center(teacher_cls, teacher_patch)
        
        return total_loss1 + total_loss2, total_loss1.item(), total_loss2.item()

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        # Update CLS center
        batch_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        # dist.all_reduce(batch_center) # If using multi-gpu distributed
        batch_center = batch_center / len(teacher_cls)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        # Update Patch center
        # Average over batch AND spatial dimensions
        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        # dist.all_reduce(patch_center) # If using multi-gpu distributed
        patch_center = patch_center / len(teacher_patch)
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)

class MaskingGenerator:
    def __init__(self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
                 min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1/min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def __call__(self):
        mask = np.zeros(self.get_shape(), dtype=int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = 0
            for attempt in range(10):
                target_area = np.random.uniform(self.min_num_patches, max_mask_patches)
                aspect_ratio = math.exp(np.random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < self.width and h < self.height:
                    top = np.random.randint(0, self.height - h)
                    left = np.random.randint(0, self.width - w)

                    num_masked = mask[top: top + h, left: left + w].sum()
                    # Overlap
                    if 0 < h * w - num_masked <= max_mask_patches:
                        for i in range(top, top + h):
                            for j in range(left, left + w):
                                if mask[i, j] == 0:
                                    mask[i, j] = 1
                                    delta += 1

                    if delta > 0:
                        break
            if delta == 0:
                break
            else:
                mask_count += delta
        
        return mask # Returns (H, W) mask

def setup_logging(log_dir="./logs"):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0):
    """
    Cosine learning rate schedule with optional warmup.
    
    Args:
        base_value: Initial learning rate
        final_value: Final learning rate
        epochs: Total number of epochs
        niter_per_ep: Number of iterations per epoch
        warmup_epochs: Number of warmup epochs
    
    Returns:
        List of learning rates for each iteration
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(0, base_value, warmup_iters)
    
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    
    schedule = np.concatenate((warmup_schedule, schedule))
    
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class DINOLoss(nn.Module):
    """
    DINO loss function.
    
    The loss encourages the student outputs to match the teacher outputs
    across different views of the same image.
    """
    
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Teacher temperature schedule
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
    
    def forward(self, student_output, teacher_output, epoch):
        """
        Args:
            student_output: Output from student network [total_crops*B, out_dim]
            teacher_output: Output from teacher network [2*B, out_dim] (only global crops)
            epoch: Current epoch
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        
        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # Only 2 global crops for teacher
        
        total_loss = 0
        n_loss_terms = 0
        
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Skip same view
                    continue
                
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output centering.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        
        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


@torch.no_grad()
def extract_features(model, dataloader, device):
    """
    Extract features from a frozen model for k-NN evaluation.
    
    Args:
        model: Trained encoder (frozen)
        dataloader: Evaluation dataloader
        device: Device to run on
    
    Returns:
        features: numpy array of shape [N, feature_dim]
        labels: numpy array of shape [N]
    """
    model.eval()
    
    features_list = []
    labels_list = []
    
    for images, labels in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        
        # Forward pass
        with torch.no_grad():
            feats = model(images)
            # L2 normalize features (important for cosine similarity)
            feats = F.normalize(feats, dim=-1, p=2)
        
        features_list.append(feats.cpu())
        labels_list.append(labels)
    
    features = torch.cat(features_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
    return features, labels


def train_knn_classifier(train_features, train_labels, val_features, val_labels, k=20):
    """
    Train KNN classifier on features.
    
    Args:
        train_features: Training features (N_train, feature_dim)
        train_labels: Training labels (N_train,)
        val_features: Validation features (N_val, feature_dim)
        val_labels: Validation labels (N_val,)
        k: Number of neighbors
    
    Returns:
        classifier: Trained KNN classifier
        val_acc: Validation accuracy
    """
    print(f"\nTraining KNN classifier (k={k})...")
    
    # Create KNN classifier with cosine similarity
    classifier = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',  # Weight by inverse distance
        metric='cosine',     # Cosine similarity for embeddings
        n_jobs=-1            # Use all CPU cores
    )
    
    # Fit on training data
    classifier.fit(train_features, train_labels)
    
    # Evaluate on both train and val
    train_acc = classifier.score(train_features, train_labels)
    val_acc = classifier.score(val_features, val_labels)
    
    print(f"\nKNN Results (k={k}):")
    print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    return classifier, val_acc


def evaluate_knn_multiple_k(train_features, train_labels, val_features, val_labels, k_values=[1, 5, 10, 20, 50, 100]):
    """
    Evaluate k-NN with multiple k values to find the best.
    
    Args:
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        k_values: List of k values to try
    
    Returns:
        results: Dictionary mapping k -> accuracy
        best_k: Best k value
        best_acc: Best accuracy
    """
    print("\n" + "="*60)
    print("Evaluating k-NN with multiple k values")
    print("="*60)
    
    results = {}
    best_acc = 0
    best_k = k_values[0]
    
    for k in k_values:
        classifier = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',
            metric='cosine',
            n_jobs=-1
        )
        
        classifier.fit(train_features, train_labels)
        val_acc = classifier.score(val_features, val_labels)
        
        results[k] = val_acc
        print(f"k={k:3d}: Val Accuracy = {val_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_k = k
    
    print(f"\n✓ Best k={best_k} with accuracy {best_acc*100:.2f}%")
    
    return results, best_k, best_acc


def save_checkpoint(state, filename="checkpoint.pth"):
    """Save training checkpoint."""
    torch.save(state, filename)
    print(f"✓ Checkpoint saved to {filename}")


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """Load training checkpoint."""
    if not os.path.exists(filename):
        print(f"✗ Checkpoint {filename} not found")
        return 0
    
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"✓ Loaded checkpoint from epoch {epoch}")
    
    return epoch


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_model_summary(model):
    """Print model parameter summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*60 + "\n")

def init_wandb(args, project_name="ssl-competition"):
    """
    Initialize Weights & Biases logging.
    
    Args:
        args: Training arguments
        project_name: W&B project name
    
    Returns:
        wandb run object
    """
    # Create run name
    run_name = f"ibot_vit_patch{args.patch_size}_bs{args.batch_size}_lr{args.lr}_ep{args.epochs}"
    
    # Initialize W&B
    run = wandb.init(
        project=project_name,
        name=run_name,
        config={
            # Model config
            "architecture": "ViT-Small",
            "patch_size": args.patch_size,
            "out_dim": args.out_dim,
            "drop_path_rate": args.drop_path_rate,
            
            # Training config
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "weight_decay_end": args.weight_decay_end,
            "warmup_epochs": args.warmup_epochs,
            "clip_grad": args.clip_grad,
            
            # DINO config
            "momentum_teacher": args.momentum_teacher,
            "teacher_temp": args.teacher_temp,
            "warmup_teacher_temp": args.warmup_teacher_temp,
            "warmup_teacher_temp_epochs": args.warmup_teacher_temp_epochs,
            "local_crops_number": args.local_crops_number,
            
            # Data config
            "train_ratio": 0.9,
            "val_ratio": 0.1,
            "image_size": 96,
            
            # k-NN config
            "knn_k": args.knn_k,
            
            # Other
            "seed": args.seed,
        },
        tags=["dino", "vit-small", "ssl", "competition"]
    )
    
    print(f"✓ W&B initialized: {run.get_url()}")
    return run


def log_metrics(metrics, step=None, commit=True):
    """
    Log metrics to W&B.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Training step/epoch
        commit: Whether to commit the log
    """
    wandb.log(metrics, step=step, commit=commit)


def watch_model(model, log_freq=100):
    """
    Watch model gradients and parameters in W&B.
    
    Args:
        model: PyTorch model
        log_freq: Logging frequency
    """
    wandb.watch(model, log='all', log_freq=log_freq)


def finish_wandb():
    """Finish W&B run."""
    wandb.finish()
    print("✓ W&B run finished")