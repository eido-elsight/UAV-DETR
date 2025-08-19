#!/usr/bin/env python3
"""
UAV-DETR Evaluation Script for Halo Object Detection Project
Evaluate trained models on VisDrone validation set and test images
"""

import warnings
import os
from pathlib import Path
from ultralytics import RTDETR
import torch
import json
import time
import cv2
import numpy as np

warnings.filterwarnings('ignore')


def evaluate_model(model_path, 
                   data_yaml='visdrone_4class.yaml',
                   test_images_dir=None,
                   output_dir='evaluation_results',
                   device='0',
                   conf_threshold=0.25,
                   save_images=True):
    """
    Evaluate UAV-DETR model performance.
    
    Args:
        model_path: Path to trained model weights (.pt file)
        data_yaml: Path to dataset configuration YAML
        test_images_dir: Optional directory with test images
        output_dir: Directory to save evaluation results
        device: GPU device ('0', 'cpu')
        conf_threshold: Confidence threshold for detections
        save_images: Whether to save annotated images
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"🔍 Starting UAV-DETR evaluation")
    print(f"   Model: {model_path}")
    print(f"   Data: {data_yaml}")
    print(f"   Output: {output_path}")
    print(f"   Confidence threshold: {conf_threshold}")
    
    # Load model
    model = RTDETR(model_path)
    
    # Evaluate on validation set
    print(f"\n📊 Evaluating on validation set...")
    val_results = model.val(
        data=data_yaml,
        device=device,
        conf=conf_threshold,
        save_json=True,
        save_hybrid=False,
        plots=True,
        verbose=True
    )
    
    # Print validation metrics
    print(f"\n✅ Validation Results:")
    print(f"   mAP@0.5: {val_results.box.map50:.3f}")
    print(f"   mAP@0.5:0.95: {val_results.box.map:.3f}")
    
    # Per-class results
    if hasattr(val_results.box, 'map50_per_class'):
        class_names = ['car', 'van', 'truck', 'bus']
        print(f"\n📋 Per-class AP@0.5:")
        for i, (name, ap) in enumerate(zip(class_names, val_results.box.map50_per_class)):
            print(f"   {name}: {ap:.3f}")
    
    # Test on specific images if provided
    if test_images_dir:
        test_on_images(model, test_images_dir, output_path, conf_threshold, save_images)
    
    # Performance timing test
    performance_test(model, device, image_size=640)
    
    return val_results


def test_on_images(model, test_dir, output_dir, conf_threshold, save_images):
    """Test model on specific images and save results."""
    
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"⚠️  Test directory not found: {test_dir}")
        return
    
    print(f"\n🖼️  Testing on images in: {test_dir}")
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(test_path.glob(f'*{ext}'))
        image_files.extend(test_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"⚠️  No images found in {test_dir}")
        return
    
    print(f"   Found {len(image_files)} images")
    
    results_summary = []
    
    for img_path in sorted(image_files):
        print(f"   Processing: {img_path.name}")
        
        # Run inference
        results = model(str(img_path), conf=conf_threshold, device=model.device)
        
        # Extract detections
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                detection = {
                    'class_id': int(boxes.cls[i]),
                    'class_name': model.names[int(boxes.cls[i])],
                    'confidence': float(boxes.conf[i]),
                    'bbox': boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        
        # Save results
        result_data = {
            'image': img_path.name,
            'detections_count': len(detections),
            'detections': detections
        }
        results_summary.append(result_data)
        
        # Save annotated image
        if save_images and len(results) > 0:
            annotated = results[0].plot()
            output_img_path = output_dir / f"annotated_{img_path.name}"
            cv2.imwrite(str(output_img_path), annotated)
        
        print(f"     Found {len(detections)} detections")
    
    # Save results summary
    results_file = output_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"   Results saved to: {results_file}")


def performance_test(model, device, image_size=640, num_runs=50):
    """Test model inference speed."""
    
    print(f"\n⚡ Performance testing (image size: {image_size}x{image_size})")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    
    # Warmup runs
    for _ in range(5):
        _ = model(dummy_image, verbose=False)
    
    # Timing runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model(dummy_image, verbose=False)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1000 / avg_time
    
    print(f"   Average inference time: {avg_time:.1f} ± {std_time:.1f} ms")
    print(f"   Min/Max: {min_time:.1f} / {max_time:.1f} ms")
    print(f"   Average FPS: {fps:.1f}")
    print(f"   Meets <100ms requirement: {'✅ YES' if avg_time < 100 else '❌ NO'}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate UAV-DETR model')
    parser.add_argument('--model', required=True,
                        help='Path to trained model weights (.pt file)')
    parser.add_argument('--data', default='visdrone_4class.yaml',
                        help='Dataset configuration YAML')
    parser.add_argument('--test-images', 
                        help='Directory with test images')
    parser.add_argument('--output', default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--device', default='0',
                        help='GPU device (0, 1, 2, etc.) or cpu')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--no-save-images', action='store_true',
                        help='Don\'t save annotated images')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        test_images_dir=args.test_images,
        output_dir=args.output,
        device=args.device,
        conf_threshold=args.conf,
        save_images=not args.no_save_images
    )
