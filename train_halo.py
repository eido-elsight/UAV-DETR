#!/usr/bin/env python3
"""
UAV-DETR Training Script for Halo Object Detection Project
Adapted for 4-class vehicle detection on VisDrone dataset
"""

import warnings
import os
from pathlib import Path
from ultralytics import RTDETR
import torch

warnings.filterwarnings('ignore')


def check_path(path):
    """Check if path exists and raise error if not found."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")


def train_uav_detr(model_variant='r18', 
                   data_yaml='visdrone_4class.yaml',
                   epochs=100,
                   batch_size=1,
                   image_size=640,
                   device='0',
                   project='runs/train',
                   name='uav_detr_halo',
                   patience=15,
                   resume=None):
    """
    Train UAV-DETR model for aerial vehicle detection.
    
    Args:
        model_variant: 'r18' or 'r50' for ResNet18 or ResNet50 backbone
        data_yaml: Path to dataset configuration YAML
        epochs: Number of training epochs
        batch_size: Batch size (1 for RTX 1000 Ada 6GB constraint)
        image_size: Input image size
        device: GPU device ('0' for first GPU, 'cpu' for CPU)
        project: Project directory for saving results
        name: Run name for this experiment
        patience: Early stopping patience
        resume: Path to checkpoint to resume from
    """
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Get current script directory
    current_dir = Path(__file__).parent
    
    # Build paths
    model_config = current_dir / f'ultralytics/cfg/models/uavdetr-{model_variant}.yaml'
    data_path = current_dir / data_yaml
    
    # Validate paths
    check_path(model_config)
    check_path(data_path)
    
    print(f"🚀 Starting UAV-DETR training")
    print(f"   Model: UAV-DETR-{model_variant.upper()}")
    print(f"   Data: {data_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {image_size}")
    print(f"   Device: {device}")
    print(f"   Project: {project}/{name}")
    
    # Initialize model
    model = RTDETR(str(model_config))
    
    # Start training
    results = model.train(
        data=str(data_path),
        cache=False,  # Don't cache images (memory constraint)
        imgsz=image_size,
        epochs=epochs,
        batch=batch_size,
        workers=4,  # Reduced workers for stability
        device=device,
        resume=resume,  # Resume from checkpoint if provided
        project=project,
        name=name,
        patience=patience,  # Early stopping patience
        save_period=10,  # Save checkpoint every 10 epochs
        val=True,  # Validate during training
        plots=True,  # Generate training plots
        verbose=True,  # Verbose output
        # Optimization settings for aerial detection
        lr0=0.001,  # Initial learning rate
        weight_decay=0.0005,  # L2 regularization
        warmup_epochs=3,  # Warmup epochs
        warmup_momentum=0.8,  # Warmup momentum
        warmup_bias_lr=0.1,  # Warmup bias learning rate
        box=7.5,  # Box loss gain
        cls=0.5,  # Classification loss gain  
        dfl=1.5,  # Distribution focal loss gain
        # Data augmentation (conservative for aerial imagery)
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,  # Saturation augmentation  
        hsv_v=0.4,  # Value augmentation
        degrees=0.0,  # Rotation degrees (0 for overhead imagery)
        translate=0.1,  # Translation fraction
        scale=0.5,  # Scale factor
        shear=0.0,  # Shear degrees
        perspective=0.0,  # Perspective factor
        flipud=0.0,  # Vertical flip probability (not suitable for overhead)
        fliplr=0.5,  # Horizontal flip probability
        mosaic=1.0,  # Mosaic augmentation probability
        mixup=0.0,  # Mixup augmentation probability (disabled)
        copy_paste=0.0,  # Copy-paste augmentation (disabled)
    )
    
    print(f"✅ Training completed!")
    print(f"   Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"   Training metrics: {results.save_dir}/results.png")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train UAV-DETR for aerial vehicle detection')
    parser.add_argument('--model', choices=['r18', 'r50'], default='r18',
                        help='Model variant: r18 (fast) or r50 (accurate)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size (1 for RTX 1000 Ada)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--device', default='0',
                        help='GPU device (0, 1, 2, etc.) or cpu')
    parser.add_argument('--name', default='uav_detr_halo',
                        help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Run training
    train_uav_detr(
        model_variant=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        device=args.device,
        name=args.name,
        resume=args.resume
    )
